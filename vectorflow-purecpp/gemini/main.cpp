#include "NeuralNetwork.h"
#include "cuda_utils.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Initialize random seed
    srand(time(nullptr));
    
    // Initialize CUDA
    initializeCUDA();
    
    // Define network architecture
    int layers[] = {2, 4, 1};
    int numLayers = 3;
    
    // Create neural network
    NeuralNetwork nn(layers, numLayers);
    
    // Create sample training data (XOR problem)
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
    
    int dataSize = 4;
    
    printf("Training XOR problem...\n");
    
    // Train the network
    nn.train(2000, 0.1f, X, Y, dataSize);
    
    // Print final weights
    nn.printNN();
    
    // Test predictions
    printf("\n--- Testing Predictions ---\n");
    
    float testInputs[][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    float expectedOutputs[] = {0.0f, 1.0f, 1.0f, 0.0f};
    
    for (int i = 0; i < 4; i++) {
        float* result = nn.predict(testInputs[i]);
        printf("Input: [%.1f, %.1f] -> Predicted: %.6f, Expected: %.1f\n", 
               testInputs[i][0], testInputs[i][1], result[0], expectedOutputs[i]);
        delete[] result;
    }
    
    return 0;
}