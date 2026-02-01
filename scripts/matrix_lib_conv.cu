#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16
//----------------------Functions Start
//Error Checkers to avoid silent errors, found in stack overflow
// CUDA Error Checking
#define cuda_check(err) { \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}

// CUBLAS Error Checking
#define cublas_check(status) { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error" << std::endl; \
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

    float Pvalue = 0.0;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < N && (m*TILE_WIDTH+tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (Col < N && (m*TILE_WIDTH+ty) < N)
            ds_B[ty][tx] = B[(m*TILE_WIDTH + ty) * N + Col];
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
extern "C" void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
} 


//Image Convolution
// -------------------- PGM I/O --------------------
//using python functions for the input/output so no cuda library functions needed

__global__ void imageConvNaive(float const *input_image, float const *c_filter, float *output_image, int const n_rows, int const n_cols, int const filter_dim) {
    //defining thread outputs
    int const outputCol = (blockIdx.x*blockDim.x) + threadIdx.x;
    int const outputRow = (blockIdx.y*blockDim.y)+ threadIdx.y;
    if(outputRow>=n_rows || outputCol>=n_cols)
        return;
    float Pvalue = 0.0;

    int input_col_edge_top = max(outputCol-filter_dim,0);
    int input_col_edge_bottom = min(outputCol + filter_dim,n_cols-1);
    int input_row_edge_left = max(outputRow- filter_dim,0);
    int input_row_edge_right = min(outputRow + filter_dim,n_rows-1);
    
    //try to speed up by unrolling loop where possible to avoid unnecessary conditionalcheck
    #pragma unroll
    //outer loop walks along row
    for( int input_row = input_row_edge_left; input_row<=input_row_edge_right; ++input_row){
        int kernel_row = input_row-outputRow + filter_dim;
        #pragma unroll
        //inner loop walks along column
        for(int input_col = input_col_edge_top; input_col <=input_col_edge_bottom; ++input_col){
            int kernel_col = input_col-outputCol + filter_dim;
            //computing the point value of the overall product for this thread
            Pvalue += c_filter[kernel_row * (2*filter_dim +1)+kernel_col]*input_image[input_row*n_cols+input_col];
        }}
        output_image[outputRow*n_cols+outputCol] = Pvalue;
    }
    
extern "C" void imageConvNaive(float* input_image, float* c_kernel, float* output_image, int n_rows, int n_cols, int filter_dim){
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n_cols + block.x-1)/block.x,(n_rows+block.y-1)/block.y);
    imageConvNaive<<<grid,block>>>(
        input_image, c_kernel,output_image,n_rows,n_cols, filter_dim
    );
}


