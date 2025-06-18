#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

void matrixMultiply(const float* input, const float* weights, const float* biases, float* output, int inputSize, int outputSize);
void matrixMultiplyCUDA(const float* input, const float* weights, const float* biases, float* output, int inputSize, int outputSize);

#endif // MATRIX_UTILS_H