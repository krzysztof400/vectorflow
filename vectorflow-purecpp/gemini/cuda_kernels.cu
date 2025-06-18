#include "cuda_kernels.h"

// CUDA kernels implementation
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void relu_derivative_kernel(float* data, float* original, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = original[idx] > 0.0f ? 1.0f : 0.0f;
    }
}

__global__ void add_bias_kernel(float* output, float* bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += bias[idx];
    }
}

__global__ void compute_output_error_kernel(float* error, float* predicted, float* actual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predicted[idx] - actual[idx];
    }
}

__global__ void hadamard_product_kernel(float* result, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void update_weights_kernel(float* weights, float* gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

__global__ void update_biases_kernel(float* biases, float* errors, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        biases[idx] -= learning_rate * errors[idx];
    }
}