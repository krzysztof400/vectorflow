#include <pybind11/pybind11.h>
#include "matrix_utils.h"
#include "model.h"

namespace py = pybind11;

PYBIND11_MODULE(vectorflow, m) {
    m.doc() = "VectorFlow: A Python library for matrix operations and model logic using C++";

    // Bind matrix utility functions
    m.def("matrix_add", &matrix_add, "Add two matrices");
    m.def("matrix_subtract", &matrix_subtract, "Subtract two matrices");
    m.def("matrix_multiply", &matrix_multiply, "Multiply two matrices");

    // Bind model functions
    m.def("create_model", &create_model, "Create a new model");
    m.def("train_model", &train_model, "Train the model with given data");
    m.def("predict", &predict, "Make predictions using the trained model");
}