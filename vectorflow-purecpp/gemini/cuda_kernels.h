#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

// CUDA kernel declarations
__global__ void relu_kernel(float* data, int size);
__global__ void relu_derivative_kernel(float* data, float* original, int size);
__global__ void add_bias_kernel(float* output, float* bias, int size);
__global__ void compute_output_error_kernel(float* error, float* predicted, float* actual, int size);
__global__ void hadamard_product_kernel(float* result, float* a, float* b, int size);
__global__ void update_weights_kernel(float* weights, float* gradients, float learning_rate, int size);
__global__ void update_biases_kernel(float* biases, float* errors, float learning_rate, int size);

#endif // CUDA_KERNELS_H