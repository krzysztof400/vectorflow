#include "nn/model.h"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// The class definition has been removed from this file and is now only in 'nn/model.h'.
// This file contains the implementations of the NeuralNetwork methods.

NeuralNetwork::NeuralNetwork(const int Layers[], int numLayers,  int opt)
    : numOfLayers(numLayers), option(opt) {
    
    srand(time(0)); 

    shape = new int[numOfLayers];
    for (int i = 0; i < numOfLayers; i++) {
        shape[i] = Layers[i];
    }

    weights = new float*[numOfLayers - 1];
    biases = new float*[numOfLayers - 1];
    d_activations = new float*[numOfLayers -1];
    d_errors = new float*[numOfLayers - 1];
    
    for (int i = 0; i < numOfLayers - 1; i++) {
        int inputSize = shape[i];
        int outputSize = shape[i + 1];

        weights[i] = new float[inputSize * outputSize];
        for (int j = 0; j < inputSize * outputSize; j++) {
            weights[i][j] = ((float)rand() / RAND_MAX) * sqrt(2.0f / inputSize);
        }
        
        biases[i] = new float[outputSize];
        for (int j = 0; j < outputSize; j++) {
            biases[i][j] = ((float)rand() / RAND_MAX) * sqrt(2.0f / inputSize); 
        }
        
        d_activations[i] = new float[outputSize];
        d_errors[i] = new float[outputSize];
    }
}
    
NeuralNetwork::~NeuralNetwork() {
    for (int i = 0; i < numOfLayers - 1; i++) {
        delete[] weights[i];
        delete[] biases[i];
        delete[] d_activations[i];
        delete[] d_errors[i];
    }
    delete[] weights;
    delete[] biases;
    delete[] d_activations;
    delete[] d_errors;
    delete[] shape;
}
    
void NeuralNetwork::printNN() {
    printf("\n--- Neural Network Weights and Biases ---\n");
    for (int l = 0; l < numOfLayers - 1; l++) {
        int in = shape[l];
        int out = shape[l + 1];
        printf("Layer %d to Layer %d (%d -> %d):\n", l, l + 1, in, out);
        printf("  Weights:\n");
        for (int i = 0; i < in; i++) {
            printf("    Node %d: [", i);
            for (int j = 0; j < out; j++) {
                printf("%f", weights[l][i * out + j]);
                if (j < out - 1) printf(", ");
            }
            printf("]\n");
        }
        printf("  Biases: [");
        for (int j = 0; j < out; j++) {
            printf("%f", biases[l][j]);
            if (j < out - 1) printf(", ");
        }
        printf("]\n\n");
    }
}

void NeuralNetwork::train(int epochs, float learningRate, float X_train[], float Y_train[], int dataSize) {
    printf("Training started for %d epochs with learning rate %f. Data size: %d. Input: %d, Output: %d\n",
        epochs, learningRate, dataSize, shape[0], shape[numOfLayers - 1]);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_epoch_loss = 0.0f;
        for (int i = 0; i < dataSize; i++) {
            float* current_X = X_train + i * shape[0];
            float* current_Y = Y_train + i * shape[numOfLayers - 1];

            forwardPropagation(current_X);

            for(int k=0; k < shape[numOfLayers-1]; ++k) {
                float error = d_activations[numOfLayers - 2][k] - current_Y[k];
                total_epoch_loss += error * error;
            }

            backwardPropagation(current_X, current_Y, learningRate);
        }
        if ((epoch + 1) % 100 == 0 || epoch == epochs - 1) {
            printf("Epoch %d/%d completed. Average Loss: %f\n", epoch + 1, epochs, total_epoch_loss / dataSize);
        }
    }
    printf("Training completed.\n");
}

float* NeuralNetwork::predict(float* X_sample) {
    forwardPropagation(X_sample);
    return d_activations[numOfLayers - 2];
}

void NeuralNetwork::forwardPropagation(float* X_input) {
    float* current_layer_input = X_input;

    for (int l = 0; l < numOfLayers - 1; l++) {
        int inputSize = shape[l];
        int outputSize = shape[l + 1];

        for (int j = 0; j < outputSize; j++) {
            float z = biases[l][j];
            for (int k = 0; k < inputSize; k++) {
                z += current_layer_input[k] * weights[l][k * outputSize + j];
            }
            d_activations[l][j] = activationFunction(option, z);
        }

        current_layer_input = d_activations[l];
    }
}

void NeuralNetwork::backwardPropagation(float* X_sample_input, float* Y_true, float learningRate) {
    int outputLayerIndex = numOfLayers - 2;
    int outputSize = shape[numOfLayers - 1];

    for (int j = 0; j < outputSize; j++) {
        float activation_output = d_activations[outputLayerIndex][j];
        d_errors[outputLayerIndex][j] = (activation_output - Y_true[j]) * derivativeActivationFunction(option, activation_output);
    }

    for (int l = numOfLayers - 3; l >= 0; l--) {
        int currentLayerOutputSize = shape[l + 1];
        int nextLayerOutputSize = shape[l + 2];

        for (int j = 0; j < currentLayerOutputSize; j++) {
            float error_sum = 0.0f;
            for (int k = 0; k < nextLayerOutputSize; k++) {
                error_sum += d_errors[l + 1][k] * weights[l + 1][j * nextLayerOutputSize + k];
            }
            d_errors[l][j] = error_sum * derivativeActivationFunction(option, d_activations[l][j]);
        }
    }

    for (int l = 0; l < numOfLayers - 1; l++) {
        int inputSize = shape[l];
        int outputSize = shape[l + 1];
        
        float* prev_layer_activations = (l == 0) ? X_sample_input : d_activations[l - 1];

        for (int j = 0; j < outputSize; j++) {
            for (int k = 0; k < inputSize; k++) {
                weights[l][k * outputSize + j] -= learningRate * d_errors[l][j] * prev_layer_activations[k];
            }
            biases[l][j] -= learningRate * d_errors[l][j];
        }
    }
}

float NeuralNetwork::activationFunction(int option, float x) {
    switch (option) {
        case 0: return Relu(x);
        case 1: return Sigmoid(x);
        case 2: return LeakyRelu(x);
        default: return Relu(x);
    }
}

float NeuralNetwork::derivativeActivationFunction(int option, float x_activated) {
    switch (option) {
        case 0: return derivativeRelu(x_activated);
        case 1: return derivativeSigmoid(x_activated);
        case 2: return derivativeLeakyRelu(x_activated);
        default: return derivativeRelu(x_activated);
    }
}

float NeuralNetwork::Relu(float x) {
    return x > 0 ? x : 0;
}

float NeuralNetwork::derivativeRelu(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

float NeuralNetwork::LeakyRelu(float x) {
    return x > 0 ? x : 0.01f * x;
}
float NeuralNetwork::derivativeLeakyRelu(float x) {
    return x > 0 ? 1.0f : 0.01f;
}

float NeuralNetwork::Sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float NeuralNetwork::derivativeSigmoid(float x_activated) { 
    return x_activated * (1.0f - x_activated);
}

float* NeuralNetwork::Softmax(float* logits, int size) {
    float max_logit = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_exp += exp(logits[i] - max_logit);
    }

    float* probabilities = new float[size];
    for (int i = 0; i < size; i++) {
        probabilities[i] = exp(logits[i] - max_logit) / sum_exp;
    }
    return probabilities;
}