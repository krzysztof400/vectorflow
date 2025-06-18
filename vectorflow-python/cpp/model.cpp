#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <ctime>
#include "matrix_utils.h"

class Model {
private:
    float** weights;
    float** biases;
    int numOfLayers;
    std::vector<int> shape;
    int option;

    float** d_activations;
    float** d_errors;

public:
    Model(const std::vector<int>& layers, int opt = 1)
        : numOfLayers(layers.size()), shape(layers), option(opt) {
        
        if (numOfLayers < 2) {
            throw std::invalid_argument("A neural network must have at least 2 layers (input and output).");
        }

        srand(time(0)); 

        weights = new float*[numOfLayers - 1];
        biases = new float*[numOfLayers - 1];
        d_activations = new float*[numOfLayers - 1];
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
    
    ~Model() {
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
    }
    
    void printNN() {
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

    void train(const std::vector<std::vector<float>>& X_train, const std::vector<std::vector<float>>& Y_train, int epochs, float learningRate) {
        if (X_train.empty() || Y_train.empty() || X_train.size() != Y_train.size()) {
            throw std::invalid_argument("Training data and labels must be non-empty and have the same number of samples.");
        }
        if (X_train[0].size() != shape[0] || Y_train[0].size() != shape.back()) {
            throw std::invalid_argument("Training data dimensions do not match model shape.");
        }

        int dataSize = X_train.size();
        printf("Training started for %d epochs with learning rate %f. Data size: %d. Input: %d, Output: %d\n",
            epochs, learningRate, dataSize, shape[0], shape.back());

        std::vector<float> flat_X_train;
        flat_X_train.reserve(dataSize * shape[0]);
        for(const auto& row : X_train) flat_X_train.insert(flat_X_train.end(), row.begin(), row.end());
        
        std::vector<float> flat_Y_train;
        flat_Y_train.reserve(dataSize * shape.back());
        for(const auto& row : Y_train) flat_Y_train.insert(flat_Y_train.end(), row.begin(), row.end());

        for (int epoch = 0; epoch < epochs; epoch++) {
            float total_epoch_loss = 0.0f;
            for (int i = 0; i < dataSize; i++) {
                float* current_X = flat_X_train.data() + i * shape[0];
                float* current_Y = flat_Y_train.data() + i * shape.back();

                forwardPropagation(current_X);

                for(int k=0; k < shape.back(); ++k) {
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

    std::vector<float> predict(std::vector<float>& X_sample) {
        if (X_sample.size() != shape[0]) {
            throw std::invalid_argument("Input sample dimensions do not match model input shape.");
        }
        forwardPropagation(X_sample.data());
        
        int outputSize = shape.back();
        std::vector<float> result(d_activations[numOfLayers - 2], d_activations[numOfLayers - 2] + outputSize);
        return result;
    }

private:
    void forwardPropagation(float* X_input) {
        float* current_layer_input = X_input;

        for (int l = 0; l < numOfLayers - 1; l++) {
            int inputSize = shape[l];
            int outputSize = shape[l + 1];

            matrixMultiplyCUDA(current_layer_input, weights[l], biases[l], d_activations[l], inputSize, outputSize);

            current_layer_input = d_activations[l];
        }
    }

    void backwardPropagation(float* X_sample_input, float* Y_true, float learningRate) {
        int outputLayerIndex = numOfLayers - 2;
        int outputSize = shape.back();

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

    float activationFunction(int option, float x) {
        switch (option) {
            case 0: return Relu(x);
            case 1: return Sigmoid(x);
            case 2: return LeakyRelu(x);
            default: return Relu(x);
        }
    }

    float derivativeActivationFunction(int option, float x_activated) {
        switch (option) {
            case 0: return derivativeRelu(x_activated);
            case 1: return derivativeSigmoid(x_activated);
            case 2: return derivativeLeakyRelu(x_activated);
            default: return derivativeRelu(x_activated);
        }
    }

    float Relu(float x) { return x > 0 ? x : 0; }
    float derivativeRelu(float x) { return x > 0 ? 1.0f : 0.0f; }
    float LeakyRelu(float x) { return x > 0 ? x : 0.01f * x; }
    float derivativeLeakyRelu(float x) { return x > 0 ? 1.0f : 0.01f; }
    float Sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
    float derivativeSigmoid(float x_activated) { return x_activated * (1.0f - x_activated); }

    float* Softmax(float* logits, int size) {
        float max_logit = logits[0];
        for (int i = 1; i < size; i++) if (logits[i] > max_logit) max_logit = logits[i];

        float sum_exp = 0.0f;
        for (int i = 0; i < size; i++) sum_exp += exp(logits[i] - max_logit);

        float* probabilities = new float[size];
        for (int i = 0; i < size; i++) probabilities[i] = exp(logits[i] - max_logit) / sum_exp;
        return probabilities;
    }
};