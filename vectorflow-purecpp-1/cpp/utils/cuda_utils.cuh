// cuda_utils.cuh
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>

inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) 
                  << " in file " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

inline void cudaMemcpyAsyncWrapper(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dst, src, count, kind, stream));
}

inline void cudaFreeWrapper(void* ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}

inline void cudaMallocWrapper(void** ptr, size_t size) {
    CHECK_CUDA_ERROR(cudaMalloc(ptr, size));
}

#endif // CUDA_UTILS_H