// filepath: cpp-nn-evaluation/include/nn/model.h
#ifndef MODEL_H
#define MODEL_H

class NeuralNetwork {
private:
    float** weights;
    float** biases;
    int numOfLayers;
    int* shape;
    int option;

    float** d_activations;
    float** d_errors;

    // Private methods
    void forwardPropagation(float* X_input);
    void backwardPropagation(float* X_sample_input, float* Y_true, float learningRate);
    float activationFunction(int option, float x);
    float derivativeActivationFunction(int option, float x_activated);
    float Relu(float x);
    float derivativeRelu(float x);
    float LeakyRelu(float x);
    float derivativeLeakyRelu(float x);
    float Sigmoid(float x);
    float derivativeSigmoid(float x_activated);
    float* Softmax(float* logits, int size);

public:
    NeuralNetwork(const int Layers[], int numLayers, int opt = 1);
    ~NeuralNetwork();
    void printNN();
    void train(int epochs, float learningRate, float X_train[], float Y_train[], int dataSize);
    float* predict(float* X_sample);
};

#endif // MODEL_H