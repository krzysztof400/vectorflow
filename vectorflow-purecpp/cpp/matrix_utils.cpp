#include "matrix_utils.h"

void matrixMultiply(const float* input, const float* weights, const float* biases, float* output, int inputSize, int outputSize) {
    for (int j = 0; j < outputSize; j++) { // For each neuron in the current layer
        float z_val = biases[j];
        for (int k = 0; k < inputSize; k++) { // For each input to this neuron
            z_val += input[k] * weights[k * outputSize + j];
        }
        output[j] = z_val; // Store the result
    }
}