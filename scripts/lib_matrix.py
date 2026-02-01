import ctypes
import numpy as np
import time

# Load shared library
lib = ctypes.cdll.LoadLibrary("./libmatrix.so")

# Define argument types 
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]
lib.imageConvNaive.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  
    ctypes.c_int, 
    ctypes.c_int,  
    ctypes.c_int   
]

N = 1024
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

start = time.time()
lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
end = time.time()



print(f"Python call to multiply matrix completed in {end - start:.4f} seconds") 

H, W = 512, 512
filter_dim = 1  # radius → 3x3 filter

input_img = np.random.rand(H, W).astype(np.float32)
kernel = np.random.rand(2 * filter_dim + 1,
                         2 * filter_dim + 1).astype(np.float32)
output_img = np.zeros((H, W), dtype=np.float32)

start = time.time()
lib.imageConvNaive(
    input_img.ravel(),
    kernel.ravel(),
    output_img.ravel(),
    H,
    W,
    filter_dim
)
end = time.time()

print(f"imageConvNaive completed in {end - start:.4f} seconds")