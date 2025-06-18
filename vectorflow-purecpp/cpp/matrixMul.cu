#include <cuda_runtime.h>
#include "matrix_utils.h"

__global__ void matrixMultiplyKernel(const float* input, const float* weights, const float* biases, float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < outputSize) {
        float z_val = biases[idx];
        for (int k = 0; k < inputSize; k++) {
            z_val += input[k] * weights[k * outputSize + idx];
        }
        output[idx] = z_val;
    }
}

void matrixMultiplyCUDA(const float* input, const float* weights, const float* biases, float* output, int inputSize, int outputSize) {
    // Allocate device memory
    float *d_input, *d_weights, *d_biases, *d_output;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
    cudaMalloc(&d_biases, outputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, inputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, outputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (outputSize + blockSize - 1) / blockSize;
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_input, d_weights, d_biases, d_output, inputSize, outputSize);

    // Copy result back to host
    cudaMemcpy(output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);
}