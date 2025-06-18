#include "NeuralNetwork.h"
#include "cuda_kernels.h"
#include "cuda_utils.h"
#include <iostream>
#include <cmath>
#include <cstdio>

NeuralNetwork::NeuralNetwork(const int Layers[], int numLayers)
    : numOfLayers(numLayers) {
    
    // Initialize cuBLAS
    cublasCreate(&cublasHandle);
    
    shape = new int[numOfLayers];
    for (int i = 0; i < numOfLayers; i++) {
        shape[i] = Layers[i];
    }

    // Allocate host memory for weights
    h_weights = new float*[numOfLayers - 1];
    d_weights = new float*[numOfLayers - 1];
    
    for (int i = 0; i < numOfLayers - 1; i++) {
        int inputSize = shape[i];
        int outputSize = shape[i + 1];
        int weightSize = inputSize * outputSize;
        
        // Allocate host memory
        h_weights[i] = new float[weightSize];
        
        // Initialize weights
        for (int j = 0; j < weightSize; j++) {
            h_weights[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Xavier-like initialization
        }
        
        // Allocate device memory and copy
        CHECK_CUDA(cudaMalloc(&d_weights[i], weightSize * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_weights[i], h_weights[i], weightSize * sizeof(float), cudaMemcpyHostToDevice));
    }
    printf("Weights initialized and copied to GPU.\n");

    // Allocate host memory for biases
    h_biases = new float*[numOfLayers - 1];
    d_biases = new float*[numOfLayers - 1];
    
    for (int i = 0; i < numOfLayers - 1; i++) {
        int biasSize = shape[i + 1];
        
        // Allocate host memory
        h_biases[i] = new float[biasSize];
        
        // Initialize biases
        for (int j = 0; j < biasSize; j++) {
            h_biases[i][j] = 0.0f; // Initialize to zero
        }
        
        // Allocate device memory and copy
        CHECK_CUDA(cudaMalloc(&d_biases[i], biasSize * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_biases[i], h_biases[i], biasSize * sizeof(float), cudaMemcpyHostToDevice));
    }
    printf("Biases initialized and copied to GPU.\n");
    
    // Allocate memory for activations, z-values, and errors
    d_activations = new float*[numOfLayers];
    d_z_values = new float*[numOfLayers - 1];
    d_errors = new float*[numOfLayers - 1];
    
    for (int i = 0; i < numOfLayers; i++) {
        CHECK_CUDA(cudaMalloc(&d_activations[i], shape[i] * sizeof(float)));
    }
    
    for (int i = 0; i < numOfLayers - 1; i++) {
        CHECK_CUDA(cudaMalloc(&d_z_values[i], shape[i + 1] * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_errors[i], shape[i + 1] * sizeof(float)));
    }
    
    printf("Activation and error arrays allocated on GPU.\n");
}

NeuralNetwork::~NeuralNetwork() {
    // Free host memory
    for (int i = 0; i < numOfLayers - 1; i++) {
        delete[] h_weights[i];
        delete[] h_biases[i];
    }
    delete[] h_weights;
    delete[] h_biases;
    
    // Free device memory
    for (int i = 0; i < numOfLayers - 1; i++) {
        cudaFree(d_weights[i]);
        cudaFree(d_biases[i]);
        cudaFree(d_z_values[i]);
        cudaFree(d_errors[i]);
    }
    delete[] d_weights;
    delete[] d_biases;
    delete[] d_z_values;
    delete[] d_errors;
    
    for (int i = 0; i < numOfLayers; i++) {
        cudaFree(d_activations[i]);
    }
    delete[] d_activations;
    
    delete[] shape;
    
    cublasDestroy(cublasHandle);
    printf("All memory deallocated.\n");
}

void NeuralNetwork::train(int epochs, float learningRate, float X[], float Y[], int dataSize) {
    printf("Training started for %d epochs with learning rate %f.\n", epochs, learningRate);
    
    // Allocate device memory for input and target
    float* d_input;
    float* d_target;
    CHECK_CUDA(cudaMalloc(&d_input, shape[0] * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_target, shape[numOfLayers - 1] * sizeof(float)));
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < dataSize; i++) {
            // Copy current sample to device
            CHECK_CUDA(cudaMemcpy(d_input, X + i * shape[0], shape[0] * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target, Y + i * shape[numOfLayers - 1], shape[numOfLayers - 1] * sizeof(float), cudaMemcpyHostToDevice));
            
            // Forward propagation
            forwardPropagation(d_input);
            
            // Backward propagation
            backwardPropagation(d_target, learningRate);
        }
        
        if (epoch % 100 == 0) {
            printf("Epoch %d completed.\n", epoch);
        }
    }
    
    cudaFree(d_input);
    cudaFree(d_target);
    printf("Training completed.\n");
}

void NeuralNetwork::forwardPropagation(float* d_input) {
    // Copy input to first activation layer
    CHECK_CUDA(cudaMemcpy(d_activations[0], d_input, shape[0] * sizeof(float), cudaMemcpyDeviceToDevice));
    
    const float alpha = 1.0f, beta = 0.0f;
    
    for (int l = 0; l < numOfLayers - 1; l++) {
        int inputSize = shape[l];
        int outputSize = shape[l + 1];
        
        // Matrix multiplication: z = W^T * a + b
        // cuBLAS uses column-major format, so we need to transpose
        cublasSgemv(cublasHandle, CUBLAS_OP_T, inputSize, outputSize,
                   &alpha, d_weights[l], inputSize,
                   d_activations[l], 1,
                   &beta, d_z_values[l], 1);
        
        // Add bias
        int blockSize = 256;
        int gridSize = getGridSize(outputSize, blockSize);
        add_bias_kernel<<<gridSize, blockSize>>>(d_z_values[l], d_biases[l], outputSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Apply activation function (ReLU for hidden layers)
        CHECK_CUDA(cudaMemcpy(d_activations[l + 1], d_z_values[l], outputSize * sizeof(float), cudaMemcpyDeviceToDevice));
        
        if (l < numOfLayers - 2) { // Hidden layers use ReLU
            relu_kernel<<<gridSize, blockSize>>>(d_activations[l + 1], outputSize);
        }
        // Output layer uses linear activation (no change needed)
        
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

void NeuralNetwork::backwardPropagation(float* d_target, float learningRate) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Compute output layer error
    int outputSize = shape[numOfLayers - 1];
    int blockSize = 256;
    int gridSize = getGridSize(outputSize, blockSize);
    
    compute_output_error_kernel<<<gridSize, blockSize>>>(
        d_errors[numOfLayers - 2], d_activations[numOfLayers - 1], d_target, outputSize);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Backpropagate errors
    for (int l = numOfLayers - 2; l > 0; l--) {
        int currentSize = shape[l + 1];
        int prevSize = shape[l];
        
        // Allocate temporary memory for derivatives
        float* d_derivatives;
        CHECK_CUDA(cudaMalloc(&d_derivatives, currentSize * sizeof(float)));
        
        // Compute activation derivatives
        gridSize = getGridSize(currentSize, blockSize);
        relu_derivative_kernel<<<gridSize, blockSize>>>(d_derivatives, d_z_values[l], currentSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Element-wise multiplication of errors and derivatives
        hadamard_product_kernel<<<gridSize, blockSize>>>(d_errors[l], d_errors[l], d_derivatives, currentSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        if (l > 0) {
            // Propagate error to previous layer
            cublasSgemv(cublasHandle, CUBLAS_OP_N, prevSize, currentSize,
                       &alpha, d_weights[l], prevSize,
                       d_errors[l], 1,
                       &beta, d_errors[l - 1], 1);
        }
        
        cudaFree(d_derivatives);
    }
    
    // Update weights and biases
    updateWeights(learningRate);
}

void NeuralNetwork::updateWeights(float learningRate) {
    const float alpha = 1.0f;
    
    for (int l = 0; l < numOfLayers - 1; l++) {
        int inputSize = shape[l];
        int outputSize = shape[l + 1];
        
        // Allocate temporary memory for weight gradients
        float* d_weight_gradients;
        CHECK_CUDA(cudaMalloc(&d_weight_gradients, inputSize * outputSize * sizeof(float)));
        
        // Compute weight gradients: dW = a * error^T
        cublasSger(cublasHandle, inputSize, outputSize,
                  &alpha, d_activations[l], 1,
                  d_errors[l], 1,
                  d_weight_gradients, inputSize);
        
        // Update weights
        int weightSize = inputSize * outputSize;
        int blockSize = 256;
        int gridSize = getGridSize(weightSize, blockSize);
        update_weights_kernel<<<gridSize, blockSize>>>(d_weights[l], d_weight_gradients, learningRate, weightSize);
        
        // Update biases
        gridSize = getGridSize(outputSize, blockSize);
        update_biases_kernel<<<gridSize, blockSize>>>(d_biases[l], d_errors[l], learningRate, outputSize);
        
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_weight_gradients);
    }
}

void NeuralNetwork::printNN() {
    printf("\n--- Neural Network Weights ---\n");
    for (int l = 0; l < numOfLayers - 1; l++) {
        int inputSize = shape[l];
        int outputSize = shape[l + 1];
        int weightSize = inputSize * outputSize;
        
        // Copy weights back to host for printing
        CHECK_CUDA(cudaMemcpy(h_weights[l], d_weights[l], weightSize * sizeof(float), cudaMemcpyDeviceToHost));
        
        printf("Layer %d to Layer %d:\n", l, l + 1);
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                printf("%.6f ", h_weights[l][i * outputSize + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

float* NeuralNetwork::predict(float* input) {
    float* d_input;
    float* h_output = new float[shape[numOfLayers - 1]];
    
    CHECK_CUDA(cudaMalloc(&d_input, shape[0] * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, input, shape[0] * sizeof(float), cudaMemcpyHostToDevice));
    
    forwardPropagation(d_input);
    
    CHECK_CUDA(cudaMemcpy(h_output, d_activations[numOfLayers - 1], 
                         shape[numOfLayers - 1] * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    return h_output;
}