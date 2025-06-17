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
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < dataSize; i++) {
                float* current_X = X + i * shape[0]; 
                float* current_Y = Y + i * shape[numOfLayers - 1]; 
                forwardPropagation(current_X);
                backwardPropagation(current_X, current_Y, learningRate);
            }
        }
        printf("Training completed.\n");
    }

private:
    void forwardPropagation(float* X) {
        for(int l = 0; l < numOfLayers - 1; l++) {
            int in = shape[l];
            int out = shape[l + 1];
            float* nextLayerOutput = new float[out];

            for (int j = 0; j < out; j++) {
                nextLayerOutput[j] = biases[l][j];
                for (int i = 0; i < in; i++) {
                    nextLayerOutput[j] += X[i] * weights[l][i * out + j];
                }
                nextLayerOutput[j] = Relu(nextLayerOutput[j]);
            }
            X = nextLayerOutput;
        }
    }

    void backwardPropagation(float* X, float* Y, float learningRate) {
         
    }

    void updateWeights(float* X, float* Y, float learningRate) {

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
    NeuralNetwork nn(layers, 6);
    nn.printNN();
    return 0;
}
