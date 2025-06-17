#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "cuda/neural_network.cuh" // Include the CUDA header for the neural network

#define train_examples 10000
#define test_examples 100

class NeuralNetwork {
private:
    float** weights;
    float** biases;
    int numOfLayers;
    int* shape;
    int option;

    float** d_activations;
    float** d_errors;

public:
    NeuralNetwork(const int Layers[], int numLayers, int opt = 1)
        : numOfLayers(numLayers) {
        
        srand(time(0));

        shape = new int[numOfLayers];
        for (int i = 0; i < numOfLayers; i++) {
            shape[i] = Layers[i];
        }

        weights = new float*[numOfLayers - 1];
        biases = new float*[numOfLayers - 1];
        d_activations = new float*[numOfLayers - 1];
        d_errors = new float*[numOfLayers - 1];
        
        for (int i = 0; i < numOfLayers - 1; i++) {
            int inputSize = shape[i];
            int outputSize = shape[i + 1];

            weights[i] = new float[inputSize * outputSize];
            for (int j = 0; j < inputSize * outputSize; j++) {
                weights[i][j] = ((float)rand() / RAND_MAX) * 1.0f - 0.5f;
            }
            
            biases[i] = new float[outputSize];
            for (int j = 0; j < outputSize; j++) {
                biases[i][j] = ((float)rand() / RAND_MAX) * 1.0f - 0.5f;
            }
            
            d_activations[i] = new float[outputSize];
            d_errors[i] = new float[outputSize];
        }
    }
    
    ~NeuralNetwork() {
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

    void train(int epochs, float learningRate, float X_train[], float Y_train[], int dataSize) {
        printf("Training started for %d epochs with learning rate %f. Data size: %d. Input: %d, Output: %d\n",
            epochs, learningRate, dataSize, shape[0], shape[numOfLayers - 1]);

        for (int epoch = 0; epoch < epochs; epoch++) {
            float total_epoch_loss = 0.0f;
            for (int i = 0; i < dataSize; i++) {
                float* current_X = X_train + i * shape[0];
                float* current_Y = Y_train + i * shape[numOfLayers - 1];

                forwardPropagation(current_X);

                for (int k = 0; k < shape[numOfLayers - 1]; ++k) {
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

    float* predict(float* X_sample) {
        forwardPropagation(X_sample);
        return d_activations[numOfLayers - 2];
    }

private:
    void forwardPropagation(float* X_input) {
        float* current_layer_input = X_input;

        for (int l = 0; l < numOfLayers - 1; l++) {
            int inputSize = shape[l];
            int outputSize = shape[l + 1];
            
            // Compute z = W * current_layer_input + b
            for (int j = 0; j < outputSize; j++) { // For each neuron in the current layer (l+1)
                float z_val = biases[l][j];
                for (int k = 0; k < inputSize; k++) { // For each input to this neuron
                    z_val += current_layer_input[k] * weights[l][k * outputSize + j];
                }
                // Apply activation function
                d_activations[l][j] = activationFunction(option, z_val);
            }
            // The output of this layer becomes the input for the next layer
            current_layer_input = d_activations[l];
        }
    }

    void backwardPropagation(float* X_sample_input, float* Y_true, float learningRate) {
        // Calculate errors for the output layer
        int outputLayerIndex = numOfLayers - 2; // Index for d_errors and d_activations
        int outputSize = shape[numOfLayers - 1];

        for (int j = 0; j < outputSize; j++) {
            float activation_output = d_activations[outputLayerIndex][j];
            d_errors[outputLayerIndex][j] = (activation_output - Y_true[j]) * derivativeActivationFunction(option, activation_output);
        }

        // Backpropagate errors to hidden layers
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

        // Update weights and biases
        for (int l = 0; l < numOfLayers - 1; l++) {
            int inputSize = shape[l];
            int outputSize = shape[l + 1];
            
            float* prev_layer_activations;
            if (l == 0) {
                prev_layer_activations = X_sample_input;
            } else {
                prev_layer_activations = d_activations[l - 1];
            }

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
            default: return Relu(x);
        }
    }

    float derivativeActivationFunction(int option, float x_activated) {
        switch (option) {
            case 0: return derivativeRelu(x_activated);
            case 1: return derivativeSigmoid(x_activated);
            default: return derivativeRelu(x_activated);
        }
    }

    float Relu(float x) {
        return x > 0 ? x : 0;
    }

    float derivativeRelu(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }

    float Sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    float derivativeSigmoid(float x_activated) { 
        return x_activated * (1.0f - x_activated);
    }
};

int main() {
    int layers[] = {2, 4, 1}; // Input layer: 2 neurons, Hidden layer: 4 neurons, Output layer: 1 neuron
    NeuralNetwork nn(layers, 3, 0);

    float trainX[train_examples * 2];
    float trainY[train_examples];

    srand(time(0));
    for (int i = 0; i < train_examples; ++i) {
        float x1 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; 
        float x2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; 
        trainX[i * 2] = x1;
        trainX[i * 2 + 1] = x2;

        trainY[i] = (x1 * x2 > 0) ? 1.0f : 0.0f; 
    }

    nn.train(2000, 0.001f, trainX, trainY, train_examples);

    printf("\n--- Final Neural Network State ---\n");
    nn.printNN();

    printf("\n--- Predictions for Test Data ---\n");
    float testX[test_examples * 2];
    float testY[test_examples];

    for (int i = 0; i < test_examples; ++i) {
        float x1 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float x2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        testX[i * 2] = x1;
        testX[i * 2 + 1] = x2;

        testY[i] = (x1 * x2 > 0) ? 1.0f : 0.0f; 
    }

    int correct_predictions = 0;

    for (int i = 0; i < test_examples; ++i) {
        float* input_sample = testX + i * layers[0];
        float* prediction = nn.predict(input_sample);
        printf("Input: [%.1f, %.1f], Predicted: %.4f, Expected: %.1f\n",
               input_sample[0], input_sample[1], prediction[0], testY[i]);

        if ((prediction[0] >= 0.5f && testY[i] == 1.0f) || (prediction[0] < 0.5f && testY[i] == 0.0f)) {
            correct_predictions++;
        }
    }

    float accuracy = (float)correct_predictions / test_examples * 100.0f;
    printf("\nAccuracy: %.2f%%\n", accuracy);

    return 0;
}