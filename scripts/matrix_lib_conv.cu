#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TILE_WIDTH 16

//----------------------Functions Start
//Error Checkers to avoid silent errors, found in stack overflow
// CUDA Error Checking
#define cuda_check(err) \
{ \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) \
                  << " in " << __FILE__ \
                  << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}

//-------matrix multiplication
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

// Exposed C function for Python
extern "C"
void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    cuda_check(cudaMalloc((void**)&d_A, size));
    cuda_check(cudaMalloc((void**)&d_B, size));
    cuda_check(cudaMalloc((void**)&d_C, size));

    cuda_check(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cuda_check(cudaDeviceSynchronize());

    cuda_check(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


//Image Convolution
// -------------------- PGM I/O --------------------
//using python functions for the input/output so no cuda library functions needed

__global__ void imageConvNaive(
    float const *input_image,
    float const *c_filter,
    float *output_image,
    int const n_rows,
    int const n_cols,
    int const filter_dim
) {
    //defining thread outputs
    int const outputCol = (blockIdx.x * blockDim.x) + threadIdx.x;
    int const outputRow = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (outputRow >= n_rows || outputCol >= n_cols)
        return;

    float Pvalue = 0.0f;

    int input_col_edge_top = max(outputCol - filter_dim, 0);
    int input_col_edge_bottom = min(outputCol + filter_dim, n_cols - 1);
    int input_row_edge_left = max(outputRow - filter_dim, 0);
    int input_row_edge_right = min(outputRow + filter_dim, n_rows - 1);

    //try to speed up by unrolling loop where possible to avoid unnecessary conditionalcheck
    #pragma unroll
    //outer loop walks along row
    for (int input_row = input_row_edge_left;
         input_row <= input_row_edge_right;
         ++input_row) {

        int kernel_row = input_row - outputRow + filter_dim;

        #pragma unroll
        //inner loop walks along column
        for (int input_col = input_col_edge_top;
             input_col <= input_col_edge_bottom;
             ++input_col) {

            int kernel_col = input_col - outputCol + filter_dim;

            //computing the point value of the overall product for this thread
            Pvalue +=
                c_filter[kernel_row * (2 * filter_dim + 1) + kernel_col] *
                input_image[input_row * n_cols + input_col];
        }
    }

    output_image[outputRow * n_cols + outputCol] = Pvalue;
}

// Exposed C function for Python
extern "C"
void gpu_image_conv(
    float* h_input,
    float* h_kernel,
    float* h_output,
    int n_rows,
    int n_cols,
    int filter_dim
) {
    float *d_input, *d_kernel, *d_output;

    size_t img_bytes = n_rows * n_cols * sizeof(float);
    size_t ker_bytes = (2 * filter_dim + 1) * (2 * filter_dim + 1) * sizeof(float);

    cuda_check(cudaMalloc((void**)&d_input, img_bytes));
    cuda_check(cudaMalloc((void**)&d_kernel, ker_bytes));
    cuda_check(cudaMalloc((void**)&d_output, img_bytes));

    cuda_check(cudaMemcpy(d_input, h_input, img_bytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_kernel, h_kernel, ker_bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n_cols + block.x - 1) / block.x,
              (n_rows + block.y - 1) / block.y);

    imageConvNaive<<<grid, block>>>(
        d_input, d_kernel, d_output,
        n_rows, n_cols, filter_dim
    );

    cuda_check(cudaDeviceSynchronize());
    cuda_check(cudaMemcpy(h_output, d_output, img_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
