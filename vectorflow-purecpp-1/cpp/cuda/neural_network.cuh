#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

// Define constants for CUDA kernel dimensions
#define BLOCK_SIZE 256

// Function declarations for CUDA kernels
__global__ void forwardPropagationKernel(float* input, float* weights, float* biases, float* output, int inputSize, int outputSize);
__global__ void backwardPropagationKernel(float* input, float* output, float* weights, float* biases, float learningRate, int inputSize, int outputSize);

// Utility functions for CUDA memory management
void cudaAllocate(float** devicePtr, size_t size);
void cudaCopyToDevice(float* hostPtr, float* devicePtr, size_t size);
void cudaCopyToHost(float* devicePtr, float* hostPtr, size_t size);
void cudaFreeMemory(float* devicePtr);

#endif // NEURAL_NETWORK_CUH