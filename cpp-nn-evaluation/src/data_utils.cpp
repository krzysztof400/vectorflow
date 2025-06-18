#include "nn/data_utils.h"
#include <cmath>
#include <cstdlib>

void L1Normalize(float* data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += std::fabs(data[i]);
    }
    if (sum > 0) {
        for (int i = 0; i < size; ++i) {
            data[i] /= sum;
        }
    }
}

void L2Normalize(float* data, int size) {
    float sum_of_squares = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum_of_squares += data[i] * data[i];
    }
    float norm = std::sqrt(sum_of_squares);
    if (norm > 0) {
        for (int i = 0; i < size; ++i) {
            data[i] /= norm;
        }
    }
}