/*
 * Uprobe Prefetch Benchmark: measures benefit of application-guided prefetch
 *
 * Allocates oversubscribed UVM memory, processes in chunks.
 * Before processing chunk N, calls request_prefetch(chunk N+1) so BPF
 * can start migrating ahead of GPU access.
 *
 * Usage: ./uprobe_bench [total_MB] [chunk_MB] [iterations] [--no-hint]
 *
 * Build: nvcc -O2 -o uprobe_bench uprobe_bench.cu -lcudart -Wno-deprecated-gpu-targets
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <sys/time.h>

/* Hook point for BPF uprobe. Must be noinline so it has a symbol. */
extern "C" __attribute__((noinline))
void request_prefetch(void *addr, size_t length) {
    asm volatile("" :: "r"(addr), "r"(length) : "memory");
}

/* Simple sequential kernel: read input, light compute, write output */
__global__ void seq_process(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        float val = input[i];
        val = val * 1.5f + 2.0f;
        output[i] = val;
    }
}

static double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {
    size_t total_mb = 48 * 1024;  /* 48 GB default (1.5x oversubscription on 32GB) */
    size_t chunk_mb = 4;          /* 4 MB chunks */
    int iterations = 5;
    int enable_hint = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-hint") == 0) {
            enable_hint = 0;
        } else if (i == 1) {
            total_mb = strtoull(argv[1], NULL, 10);
        } else if (i == 2) {
            chunk_mb = strtoull(argv[2], NULL, 10);
        } else if (i == 3) {
            iterations = atoi(argv[3]);
        }
    }

    size_t total_bytes = total_mb * 1024ULL * 1024;
    size_t chunk_bytes = chunk_mb * 1024ULL * 1024;
    size_t array_bytes = total_bytes / 2;  /* input + output */
    size_t N = array_bytes / sizeof(float);
    size_t chunk_elems = chunk_bytes / sizeof(float);
    size_t num_chunks = (N + chunk_elems - 1) / chunk_elems;

    printf("Uprobe Prefetch Benchmark\n");
    printf("=========================\n");
    printf("Total: %zu MB, Chunk: %zu MB, Chunks: %zu, Iterations: %d\n",
           total_mb, chunk_mb, num_chunks, iterations);
    printf("Hint: %s\n\n", enable_hint ? "ENABLED (request_prefetch)" : "DISABLED");

    /* Allocate UVM memory */
    float *input, *output;
    cudaError_t err;

    err = cudaMallocManaged(&input, array_bytes, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged input failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMallocManaged(&output, array_bytes, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged output failed: %s\n", cudaGetErrorString(err));
        cudaFree(input);
        return 1;
    }

    /* Initialize on CPU */
    printf("Initializing %zu MB on CPU...\n", array_bytes / (1024*1024));
    for (size_t i = 0; i < N; i++) {
        input[i] = 1.0f;
    }

    /* Warmup: touch first chunk to establish va_space for kprobe */
    printf("Warmup (establishing va_space)...\n");
    size_t warmup_n = chunk_elems < N ? chunk_elems : N;
    seq_process<<<256, 256>>>(input, output, warmup_n);
    cudaDeviceSynchronize();

    /* Timed iterations */
    printf("\nRunning %d iterations...\n", iterations);

    for (int iter = 0; iter < iterations; iter++) {
        /* Ensure data starts on CPU for fair comparison */
        /* (For iter>0, some data may already be on GPU from previous iter) */

        double t_start = now_ms();

        for (size_t c = 0; c < num_chunks; c++) {
            size_t offset = c * chunk_elems;
            size_t this_n = (offset + chunk_elems <= N) ? chunk_elems : (N - offset);

            /* Pipeline: hint NEXT chunk before processing current */
            if (enable_hint && c + 1 < num_chunks) {
                size_t next_offset = (c + 1) * chunk_elems;
                size_t next_n = (next_offset + chunk_elems <= N) ? chunk_elems : (N - next_offset);
                /* Hint both input and output for next chunk */
                request_prefetch(input + next_offset, next_n * sizeof(float));
                request_prefetch(output + next_offset, next_n * sizeof(float));
            }

            /* Process current chunk */
            int blocks = (this_n + 255) / 256;
            if (blocks > 2048) blocks = 2048;
            seq_process<<<blocks, 256>>>(input + offset, output + offset, this_n);
        }
        cudaDeviceSynchronize();

        double t_end = now_ms();
        double elapsed = t_end - t_start;

        double bw = (double)(array_bytes * 2) / (elapsed / 1000.0) / 1e9;
        printf("  iter %d: %.1f ms  (%.2f GB/s)\n", iter, elapsed, bw);
    }

    cudaFree(input);
    cudaFree(output);
    printf("\nDone.\n");
    return 0;
}
