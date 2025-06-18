#include "matrix_utils.h"
#include <vector>

std::vector<std::vector<double>> add_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    std::vector<std::vector<double>> C(rows, std::vector<double>(cols));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

std::vector<std::vector<double>> multiply_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t colsB = B[0].size();
    std::vector<std::vector<double>> C(rowsA, std::vector<double>(colsB, 0));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}