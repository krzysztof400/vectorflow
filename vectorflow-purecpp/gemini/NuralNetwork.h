#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

class NeuralNetwork {
private:
    float** d_weights;   // Device weights
    float** d_biases;    // Device biases
    float** h_weights;   // Host weights (for initialization)
    float** h_biases;    // Host biases (for initialization)
    
    // Activations and pre-activations for each layer
    float** d_activations;
    float** d_z_values;   // Pre-activation values
    float** d_errors;     // Error terms for backprop
    
    int numOfLayers;
    int* shape;
    cublasHandle_t cublasHandle;

    // Private methods
    void forwardPropagation(float* d_input);
    void backwardPropagation(float* d_target, float learningRate);
    void updateWeights(float learningRate);

public:
    NeuralNetwork(const int Layers[], int numLayers);
    ~NeuralNetwork();
    
    void train(int epochs, float learningRate, float X[], float Y[], int dataSize);
    void printNN();
    float* predict(float* input);
};

#endif // NEURAL_NETWORK_H