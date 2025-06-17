#include <iostream>
#include <cmath>
#include <cstdio>

class NeuralNetwork {
private:
    float** weights;   // Each layer has one weight matrix (flattened)
    float** biases;
    int numOfLayers;
    int* shape;

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
                weights[i][j] = 0.0f;
            }
        }
        printf("Weights initialized.\n");

        // Allocate biases
        biases = new float*[numOfLayers - 1];
        for (int i = 0; i < numOfLayers - 1; i++) {
            biases[i] = new float[shape[i + 1]];
            for (int j = 0; j < shape[i + 1]; j++) {
                biases[i][j] = 0.0f;
            }
        }
        printf("Biases initialized.\n");
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

    void train(int epochs, float learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            forwardPropagation();
            backwardPropagation();
        }
        printf("Training completed.\n");
    }

private:
    void forwardPropagation() {
        // Placeholder
    }

    void backwardPropagation() {
        // Placeholder
    }
};

int main() {
    int layers[] = {3, 5, 2, 5, 6, 8};
    NeuralNetwork nn(layers, 6);
    nn.printNN();
    return 0;
}
