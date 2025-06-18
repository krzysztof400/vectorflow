#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Utility function to get grid size
inline int getGridSize(int size, int blockSize) {
    return (size + blockSize - 1) / blockSize;
}

// Utility function to initialize CUDA device
inline void initializeCUDA() {
    CHECK_CUDA(cudaSetDevice(0));
}

#endif // CUDA_UTILS_H