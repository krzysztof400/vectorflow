#include <iostream>
#include <cmath>
#include <cstdio>




class NeuralNetwork {
private:
    float** weights;   // Each layer has one weight matrix (flattened)
    float** biases;
    int numOfLayers;
    int* shape;

    float** d_activations;
    // float** d_z_values;
    float** d_errors;

public:
    NeuralNetwork(const int Layers[], int numLayers)
        : numOfLayers(numLayers) {
        
        shape = new int[numOfLayers];
        for (int i = 0; i < numOfLayers; i++) {
            shape[i] = Layers[i];
        }

        // Allocate weights
        weights = new float*[numOfLayers - 1];
        for (int i = 0; i < numOfLayers - 1; i++) {
            int inputSize = shape[i];
            int outputSize = shape[i + 1];
            weights[i] = new float[inputSize * outputSize];
            for (int j = 0; j < inputSize * outputSize; j++) {
                weights[i][j] = ((float)rand() / RAND_MAX) - 0.5f;
            }
        }
        printf("Weights initialized.\n");

        // Allocate biases
        biases = new float*[numOfLayers - 1];
        for (int i = 0; i < numOfLayers - 1; i++) {
            biases[i] = new float[shape[i + 1]];
            for (int j = 0; j < shape[i + 1]; j++) {
                biases[i][j] =  ((float)rand() / RAND_MAX) - 0.5f; 
            }
        }
        printf("Biases initialized.\n");

        // Allocate d_activations and d_errors
        d_activations = new float*[numOfLayers - 1];
        d_errors = new float*[numOfLayers - 1];
        for (int i = 0; i < numOfLayers - 1; i++) {
            d_activations[i] = new float[shape[i + 1]];
            d_errors[i] = new float[shape[i + 1]];
        }
        printf("d_activations and d_errors initialized.\n");
    }

    ~NeuralNetwork() {
        for (int i = 0; i < numOfLayers - 1; i++) {
            delete[] weights[i];
        }
        delete[] weights;
        printf("Weights deallocated.\n");

        for (int i = 0; i < numOfLayers - 1; i++) {
            delete[] biases[i];
        }
        delete[] biases;
        printf("Biases deallocated.\n");

        delete[] shape;
        printf("Shape deallocated.\n");
    }

    void printNN() {
        printf("\n--- Neural Network Weights ---\n");
        for (int l = 0; l < numOfLayers - 1; l++) {
            int in = shape[l];
            int out = shape[l + 1];
            printf("Layer %d to Layer %d:\n", l, l + 1);
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    printf("%f ", weights[l][i * out + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    void train(int epochs, float learningRate, float X[], float Y[], int dataSize) {
        printf("Training started for %d epochs with learning rate %f.\n", epochs, learningRate);
        printf("Data size: %d\n", dataSize);
        printf("Input shape: %d, Output shape: %d\n", shape[0], shape[numOfLayers - 1]);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < dataSize; i++) {
                float* current_X = X + i * shape[0]; 
                float* current_Y = Y + i * shape[numOfLayers - 1]; 
                printf("Epoch %d, Sample %d\n", epoch, i);
                forwardPropagation(current_X);
                backwardPropagation(current_X, current_Y, learningRate);
            }
        }
        printf("Training completed.\n");
    }

private:
    void forwardPropagation(float* X) {
        printf("Forward propagation started.\n");
        for(int l = 0; l < numOfLayers - 1; l++) {
            int inputSize = shape[l];
            int outputSize = shape[l + 1];
            float* z = new float[outputSize];
            d_activations[l] = new float[outputSize];

            // Compute z = W * X + b
            for (int j = 0; j < outputSize; j++) {
                z[j] = biases[l][j];
                for (int k = 0; k < inputSize; k++) {
                    z[j] += weights[l][k * outputSize + j] * X[k];
                }
            }

            // Apply activation function (ReLU or Sigmoid)
            for (int j = 0; j < outputSize; j++) {
                d_activations[l][j] = Relu(z[j]);
            }

            // Prepare for next layer
            X = d_activations[l];
            delete[] z;
        }
        printf("Forward propagation completed.\n");
    }

    void backwardPropagation(float* X, float* Y, float learningRate) {
        printf("Backward propagation started.\n");
        int outputSize = shape[numOfLayers - 1];
        d_errors[numOfLayers - 2] = new float[outputSize];
        for (int j = 0; j < outputSize; j++) {
            float output = d_activations[numOfLayers - 2][j];
            d_errors[numOfLayers - 2][j] = (output - Y[j]) * derivativeRelu(output);
        }

        // Backpropagate through layers
        for (int l = numOfLayers - 3; l >= 0; l--) {
            int currentSize = shape[l + 1];
            int prevSize = shape[l];
            d_errors[l] = new float[currentSize];

            for (int j = 0; j < currentSize; j++) {
            float error = 0.0f;
            for (int k = 0; k < shape[l + 2]; k++) {
                error += d_errors[l + 1][k] * weights[l + 1][j * shape[l + 2] + k];
            }
            d_errors[l][j] = error * derivativeRelu(d_activations[l][j]);
            }
        }

        // Update weights and biases
        for (int l = 0; l < numOfLayers - 1; l++) {
            int inputSize = shape[l];
            int outputSize = shape[l + 1];

            for (int j = 0; j < outputSize; j++) {
            for (int k = 0; k < inputSize; k++) {
                weights[l][k * outputSize + j] -= learningRate * d_errors[l][j] * (l == 0 ? X[k] : d_activations[l - 1][k]);
            }
            biases[l][j] -= learningRate * d_errors[l][j];
            }
        }

        // Free memory for errors
        for (int l = 0; l < numOfLayers - 1; l++) {
            delete[] d_errors[l];
        }
        printf("Backward propagation completed.\n");
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

    float derivativeSigmoid(float x) {
        return x * (1.0f - x);
    }

    float Softmax(float x[], int size) {
        float maxVal = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > maxVal) {
                maxVal = x[i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += exp(x[i] - maxVal);
        }
        return exp(x[0] - maxVal) / sum; // Example for the first element
    }
};

int main() {
    int layers[] = {2, 4, 1};
    NeuralNetwork nn(layers, 3);
    // nn.printNN();

    float X[] = {
        0.0f, 0.0f,  // Input: [0, 0]
        0.0f, 1.0f,  // Input: [0, 1]
        1.0f, 0.0f,  // Input: [1, 0]
        1.0f, 1.0f   // Input: [1, 1]
    };

    float Y[] = {
        0.0f,  // Target: 0
        1.0f,  // Target: 1
        1.0f,  // Target: 1
        0.0f   // Target: 0
    };

    nn.train(2000, 0.1f, X, Y, 4);

    return 0;
}
