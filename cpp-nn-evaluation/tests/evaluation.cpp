#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "nn/model.h"
#include "nn/data_utils.h"

#define TRAIN_EXAMPLES 10000
#define TEST_EXAMPLES 100

void evaluateNeuralNetwork(NeuralNetwork& nn, float learningRate, const char* activationFunction) {
    float trainX[TRAIN_EXAMPLES * 2];
    float trainY[TRAIN_EXAMPLES];

    srand(time(0));
    for (int i = 0; i < TRAIN_EXAMPLES; ++i) {
        float x1 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float x2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        trainX[i * 2] = x1;
        trainX[i * 2 + 1] = x2;
        trainY[i] = (x1 * x2 > 0) ? 1.0f : 0.0f;
    }

    nn.train(100, learningRate, trainX, trainY, TRAIN_EXAMPLES);

    float testX[TEST_EXAMPLES * 2];
    float testY[TEST_EXAMPLES];

    for (int i = 0; i < TEST_EXAMPLES; ++i) {
        float x1 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float x2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        testX[i * 2] = x1;
        testX[i * 2 + 1] = x2;
        testY[i] = (x1 * x2 > 0) ? 1.0f : 0.0f;
    }

    int correct_predictions = 0;

    for (int i = 0; i < TEST_EXAMPLES; ++i) {
        float* input_sample = testX + i * 2;
        float* prediction = nn.predict(input_sample);
        if ((prediction[0] >= 0.5f && testY[i] == 1.0f) || (prediction[0] < 0.5f && testY[i] == 0.0f)) {
            correct_predictions++;
        }
    }

    float accuracy = (float)correct_predictions / TEST_EXAMPLES * 100.0f;
    std::cout << "Activation Function: " << activationFunction << ", Learning Rate: " << learningRate 
              << ", Accuracy: " << accuracy << "%" << std::endl;
}

int main() {
    int layers[] = {2, 4, 1};

    // Test with unnormalized data
    NeuralNetwork nn1(layers, 3, 0); // ReLU
    evaluateNeuralNetwork(nn1, 0.001f, "ReLU");

    NeuralNetwork nn2(layers, 3, 1); // Sigmoid
    evaluateNeuralNetwork(nn2, 0.001f, "Sigmoid");

    // Test with L1 normalized data
    float trainX_L1[TRAIN_EXAMPLES * 2];
    float trainY_L1[TRAIN_EXAMPLES];
    // Fill trainX_L1 with L1 normalized data...

    NeuralNetwork nn3(layers, 3, 0); // ReLU
    evaluateNeuralNetwork(nn3, 0.001f, "ReLU (L1 Normalized)");

    NeuralNetwork nn4(layers, 3, 1); // Sigmoid
    evaluateNeuralNetwork(nn4, 0.001f, "Sigmoid (L1 Normalized)");

    // Test with L2 normalized data
    float trainX_L2[TRAIN_EXAMPLES * 2];
    float trainY_L2[TRAIN_EXAMPLES];
    // Fill trainX_L2 with L2 normalized data...

    NeuralNetwork nn5(layers, 3, 0); // ReLU
    evaluateNeuralNetwork(nn5, 0.001f, "ReLU (L2 Normalized)");

    NeuralNetwork nn6(layers, 3, 1); // Sigmoid
    evaluateNeuralNetwork(nn6, 0.001f, "Sigmoid (L2 Normalized)");

    // Analyze learning rate effect
    for (float lr : {0.01f, 0.001f, 0.0001f}) {
        NeuralNetwork nn7(layers, 3, 0); // ReLU
        evaluateNeuralNetwork(nn7, lr, ("ReLU (Learning Rate: " + std::to_string(lr) + ")").c_str());

        NeuralNetwork nn8(layers, 3, 1); // Sigmoid
        evaluateNeuralNetwork(nn8, lr, ("Sigmoid (Learning Rate: " + std::to_string(lr) + ")").c_str());
    }

    return 0;
}