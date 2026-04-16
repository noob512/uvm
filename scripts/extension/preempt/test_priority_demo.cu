// Priority scheduling demo: LC (short kernel) vs BE (long kernel)
//
// Modes:
//   be            — Launch ONE persistent kernel that occupies ALL SMs (runs until killed)
//   lc [N]        — Launch N short kernels, measure submission-to-completion latency
//   be_loop       — Launch long kernels in a loop (lower GPU saturation)
//
// The "be" mode launches enough blocks to fill every SM slot, so LC
// CANNOT run alongside BE — LC must wait for hardware preemption timeslice.
// This is where BPF preempt helps: reduces LC wait from timeslice (~ms) to ~300us.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

static volatile int running = 1;
static void sighandler(int s) { running = 0; }

// =============================================================
// BE: persistent kernel that saturates ALL SMs
// Uses maximum blocks to fill every SM slot.
// Runs until host sets *stop = 1.
// =============================================================
__global__ void be_persistent_kernel(volatile int *stop) {
    // Spin until host signals stop
    while (!(*stop)) {
        // Busy spin — holds SM slot occupied
        long long start = clock64();
        while (clock64() - start < 100000) {} // ~50us per spin
    }
}

// =============================================================
// BE loop: launch discrete long kernels (less saturation between launches)
// =============================================================
__global__ void be_long_kernel(volatile int *flag, int iters) {
    long long start = clock64();
    while (clock64() - start < (long long)iters * 100000LL) {}
    if (threadIdx.x == 0 && blockIdx.x == 0 && flag)
        *flag = 1;
}

// =============================================================
// LC: minimal-work kernel
// =============================================================
__global__ void lc_short_kernel(float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)i * 1.001f;
}

static long time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000L + ts.tv_nsec / 1000;
}

// Query GPU for max occupancy
static int get_max_blocks() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, be_persistent_kernel, 256, 0);

    int total = blocks_per_sm * prop.multiProcessorCount;
    printf("  GPU: %s (%d SMs, %d blocks/SM, total=%d blocks)\n",
           prop.name, prop.multiProcessorCount, blocks_per_sm, total);
    return total;
}

void run_be_persistent() {
    int max_blocks = get_max_blocks();
    printf("BE: PID=%d, launching persistent kernel (%d blocks × 256 threads)...\n",
           getpid(), max_blocks);
    printf("BE: This fills ALL SM slots — LC MUST wait for hardware preemption.\n");
    printf("BE: Kill with Ctrl-C or kill %d\n", getpid());
    fflush(stdout);

    volatile int *d_stop;
    cudaMalloc((void**)&d_stop, sizeof(int));
    int zero = 0;
    cudaMemcpy((void*)d_stop, &zero, sizeof(int), cudaMemcpyHostToDevice);

    be_persistent_kernel<<<max_blocks, 256>>>(d_stop);

    // Wait for signal
    while (running) {
        sleep(1);
    }

    // Stop kernel
    int one = 1;
    cudaMemcpy((void*)d_stop, &one, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaFree((void*)d_stop);
    printf("BE: stopped\n");
}

void run_be_loop() {
    printf("BE: PID=%d, running long GPU kernels in loop...\n", getpid());
    fflush(stdout);

    volatile int *d_flag;
    cudaMalloc((void**)&d_flag, sizeof(int));

    int count = 0;
    while (running) {
        int zero = 0;
        cudaMemcpy((void*)d_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);
        long t0 = time_us();
        be_long_kernel<<<512, 256>>>(d_flag, 1000);
        cudaDeviceSynchronize();
        long elapsed = time_us() - t0;
        count++;
        if (count % 10 == 0)
            printf("BE: %d kernels done (last=%ld us)\n", count, elapsed);
    }

    cudaFree((void*)d_flag);
    printf("BE: stopped after %d kernels\n", count);
}

void run_lc(int n_trials) {
    printf("LC: PID=%d, measuring %d short kernel latencies...\n",
           getpid(), n_trials);
    fflush(stdout);

    float *d_buf;
    cudaMalloc(&d_buf, 256 * sizeof(float));

    // Warmup (3 kernels)
    for (int i = 0; i < 3; i++) {
        lc_short_kernel<<<1, 256>>>(d_buf, 256);
        cudaDeviceSynchronize();
    }

    long latencies[1024];
    int actual = n_trials > 1024 ? 1024 : n_trials;

    for (int i = 0; i < actual; i++) {
        long t0 = time_us();
        lc_short_kernel<<<1, 256>>>(d_buf, 256);
        cudaDeviceSynchronize();
        long elapsed = time_us() - t0;
        latencies[i] = elapsed;
        printf("  LC kernel [%d]: %ld us\n", i, elapsed);
        usleep(200000);  // 200ms between launches
    }

    // Sort for percentiles
    for (int i = 0; i < actual - 1; i++)
        for (int j = i + 1; j < actual; j++)
            if (latencies[j] < latencies[i]) {
                long t = latencies[i];
                latencies[i] = latencies[j];
                latencies[j] = t;
            }

    long sum = 0;
    for (int i = 0; i < actual; i++) sum += latencies[i];
    long avg = sum / actual;
    long median = latencies[actual / 2];
    long p50 = latencies[(int)(actual * 0.50)];
    long p95 = latencies[(int)(actual * 0.95)];
    long p99 = latencies[(int)(actual * 0.99)];

    printf("\n=== LC Kernel Latency (%d trials) ===\n", actual);
    printf("  avg=%ld us, median=%ld us, min=%ld us, max=%ld us\n",
           avg, median, latencies[0], latencies[actual - 1]);
    printf("  P50=%ld us, P95=%ld us, P99=%ld us\n", p50, p95, p99);

    cudaFree(d_buf);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <be|be_loop|lc> [N_trials]\n", argv[0]);
        fprintf(stderr, "  be      — persistent kernel, fills ALL SMs\n");
        fprintf(stderr, "  be_loop — loop of long kernels (less saturation)\n");
        fprintf(stderr, "  lc [N]  — measure N short kernel latencies\n");
        return 1;
    }

    signal(SIGINT, sighandler);
    signal(SIGTERM, sighandler);

    if (strcmp(argv[1], "be") == 0) {
        run_be_persistent();
    } else if (strcmp(argv[1], "be_loop") == 0) {
        run_be_loop();
    } else if (strcmp(argv[1], "lc") == 0) {
        int n = argc > 2 ? atoi(argv[2]) : 20;
        run_lc(n);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        return 1;
    }
    return 0;
}
