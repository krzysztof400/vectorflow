// filepath: /vectorflow-purecpp/vectorflow-purecpp/cpp/cuda/neural_network.cu
#include "neural_network.cuh"
#include "cuda_utils.cuh"

__global__ void forwardPropagationKernel(float* input, float* weights, float* biases, float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        float z_val = biases[idx];
        for (int j = 0; j < inputSize; j++) {
            z_val += input[j] * weights[j * outputSize + idx];
        }
        output[idx] = activationFunction(z_val);
    }
}

__global__ void backwardPropagationKernel(float* input, float* output, float* weights, float* d_errors, float* d_activations, float learningRate, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        float error = d_errors[idx] * derivativeActivationFunction(d_activations[idx]);
        for (int j = 0; j < inputSize; j++) {
            weights[j * outputSize + idx] -= learningRate * error * input[j];
        }
    }
}

void NeuralNetwork::trainWithCUDA(float* X_train, float* Y_train, int dataSize, int epochs, float learningRate) {
    float *d_input, *d_weights, *d_biases, *d_output, *d_errors, *d_activations;

    // Allocate device memory
    cudaMalloc(&d_input, sizeof(float) * inputSize);
    cudaMalloc(&d_weights, sizeof(float) * weightSize);
    cudaMalloc(&d_biases, sizeof(float) * outputSize);
    cudaMalloc(&d_output, sizeof(float) * outputSize);
    cudaMalloc(&d_errors, sizeof(float) * outputSize);
    cudaMalloc(&d_activations, sizeof(float) * outputSize);

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < dataSize; i++) {
            // Copy input data to device
            cudaMemcpy(d_input, X_train + i * inputSize, sizeof(float) * inputSize, cudaMemcpyHostToDevice);

            // Launch forward propagation kernel
            forwardPropagationKernel<<<(outputSize + 255) / 256, 256>>>(d_input, d_weights, d_biases, d_output, inputSize, outputSize);
            cudaDeviceSynchronize();

            // Calculate errors and launch backward propagation kernel
            cudaMemcpy(d_activations, d_output, sizeof(float) * outputSize, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_errors, d_activations, sizeof(float) * outputSize, cudaMemcpyDeviceToDevice);
            backwardPropagationKernel<<<(outputSize + 255) / 256, 256>>>(d_input, d_output, d_weights, d_errors, d_activations, learningRate, inputSize, outputSize);
            cudaDeviceSynchronize();
        }
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);
    cudaFree(d_errors);
    cudaFree(d_activations);
}