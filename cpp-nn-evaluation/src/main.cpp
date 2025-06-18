#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "nn/model.h"
#include "nn/data_utils.h"

#define train_examples 10000
#define test_examples 100

void evaluateNeuralNetwork(int activationOption, float learningRate) {
    int layers[] = {2, 4, 1}; // Input layer: 2 neurons, Hidden layer: 4 neurons, Output layer: 1 neuron
    NeuralNetwork nn(layers, 3, activationOption);

    float trainX[train_examples * 2];
    float trainY[train_examples];

    // Generate training data
    for (int i = 0; i < train_examples; ++i) {
        float x1 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; 
        float x2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; 
        trainX[i * 2] = x1;
        trainX[i * 2 + 1] = x2;
        trainY[i] = (x1 * x2 > 0) ? 1.0f : 0.0f; // 1 if same sign, 0 otherwise
    }

    nn.train(100, learningRate, trainX, trainY, train_examples);

    printf("\n--- Final Neural Network State with Learning Rate %.3f and Activation Option %d ---\n", learningRate, activationOption);
    nn.printNN();

    printf("\n--- Predictions for Test Data ---\n");
    float testX[test_examples * 2];
    float testY[test_examples];

    for (int i = 0; i < test_examples; ++i) {
        float x1 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float x2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        testX[i * 2] = x1;
        testX[i * 2 + 1] = x2;
        testY[i] = (x1 * x2 > 0) ? 1.0f : 0.0f; // 1 if same sign, 0 otherwise
    }

    int correct_predictions = 0;

    for (int i = 0; i < test_examples; ++i) {
        float* input_sample = testX + i * 2;
        float* prediction = nn.predict(input_sample);
        printf("Input: [%.1f, %.1f], Predicted: %.4f, Expected: %.1f\n",
               input_sample[0], input_sample[1], prediction[0], testY[i]);

        if ((prediction[0] >= 0.5f && testY[i] == 1.0f) || (prediction[0] < 0.5f && testY[i] == 0.0f)) {
            correct_predictions++;
        }
    }

    float accuracy = (float)correct_predictions / test_examples * 100.0f;
    printf("\nAccuracy: %.2f%%\n", accuracy);
}

int main() {
    srand(time(0)); // Seed for random number generation

    // Evaluate with unnormalized data
    printf("\n--- Evaluating with Unnormalized Data ---\n");
    evaluateNeuralNetwork(0, 0.001f); // ReLU, Learning Rate 0.001
    evaluateNeuralNetwork(1, 0.001f); // Sigmoid, Learning Rate 0.001

    // Evaluate with L1 normalized data
    printf("\n--- Evaluating with L1 Normalized Data ---\n");
    // Implement L1 normalization and call evaluateNeuralNetwork

    // Evaluate with L2 normalized data
    printf("\n--- Evaluating with L2 Normalized Data ---\n");
    // Implement L2 normalization and call evaluateNeuralNetwork

    return 0;
}