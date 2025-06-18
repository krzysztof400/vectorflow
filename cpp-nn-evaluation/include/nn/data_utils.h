#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <vector>

// Function to perform L1 normalization on a dataset
std::vector<float> l1Normalize(const std::vector<float>& data);

// Function to perform L2 normalization on a dataset
std::vector<float> l2Normalize(const std::vector<float>& data);

// Function to generate unnormalized training data
void generateUnnormalizedData(float* trainX, float* trainY, int dataSize);

// Function to generate L1 normalized training data
void generateL1NormalizedData(float* trainX, float* trainY, int dataSize);

// Function to generate L2 normalized training data
void generateL2NormalizedData(float* trainX, float* trainY, int dataSize);

#endif // DATA_UTILS_H