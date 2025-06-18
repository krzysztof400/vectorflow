#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>

class MatrixUtils {
public:
    static std::vector<std::vector<double>> add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
    static std::vector<std::vector<double>> subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
    static std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
    static std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A);
};

#endif // MATRIX_UTILS_H