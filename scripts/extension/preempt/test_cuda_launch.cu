// Simple CUDA program that calls cuLaunchKernel via the runtime API
#include <stdio.h>
#include <unistd.h>

__global__ void dummy_kernel(float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)i;
}

int main(int argc, char **argv) {
    int n_launches = argc > 1 ? atoi(argv[1]) : 5;
    float *d_buf;
    cudaMalloc(&d_buf, 1024 * sizeof(float));

    printf("PID=%d, launching %d kernels...\n", getpid(), n_launches);
    for (int i = 0; i < n_launches; i++) {
        dummy_kernel<<<1, 256>>>(d_buf, 256);
        cudaDeviceSynchronize();
        printf("  kernel %d launched\n", i + 1);
        usleep(500000); // 500ms between launches
    }

    cudaFree(d_buf);
    printf("Done.\n");
    return 0;
}
