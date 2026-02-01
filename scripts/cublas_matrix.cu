#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    srand(time(NULL));
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = N * N * sizeof(float);

    float *A = NULL, *B = NULL, *C = NULL;
    float *devA = NULL, *devB = NULL, *devC = NULL;
    
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    cudaMalloc(&devA, size);
    cudaMalloc(&devB, size);
    cudaMalloc(&devC, size);

    cudaMemcpy(devA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        devB, N,
        devA, N,
        &beta,
        devC, N
    );

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(C, devC, size, cudaMemcpyDeviceToHost);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error: %d\n", status);
    }

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("cuBLAS SGEMM time (N=%d): %f ms\n", N, ms);

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
