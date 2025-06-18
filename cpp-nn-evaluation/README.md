# Neural Network Evaluation Project

This project implements a simple neural network in C++ and evaluates its performance based on various criteria. The neural network is designed to classify data points based on their coordinates, using both ReLU and Sigmoid activation functions. The evaluation includes testing with unnormalized data, L1 normalized data, and L2 normalized data, as well as analyzing the effect of different learning rates on the training speed.

## Project Structure

- **include/nn/model.h**: Header file declaring the `NeuralNetwork` class, including methods for training, predicting, and managing weights and biases, as well as activation functions.
  
- **include/nn/data_utils.h**: Header file declaring utility functions for data normalization (L1 and L2) and other preprocessing functions.

- **src/main.cpp**: Entry point for the application. Initializes the neural network, generates training data, and calls the training method.

- **src/model.cpp**: Implementation of the `NeuralNetwork` class, including forward and backward propagation methods, weight updates, and activation functions.

- **src/data_utils.cpp**: Implements data utility functions declared in `data_utils.h`, including normalization functions for L1 and L2.

- **tests/evaluation.cpp**: Testing script that evaluates the neural network implementation based on specified criteria, including testing with unnormalized data, L1 normalized data, L2 normalized data, and using both Sigmoid and ReLU activation functions. Analyzes the learning rate's effect on learning speed.

- **CMakeLists.txt**: Configuration file for CMake, specifying how to build the project, including source files and include directories.

- **README.md**: Documentation for the project, including setup instructions, usage, and relevant information about the neural network implementation.

- **REPORT.md**: Report addressing the posed questions regarding the neural network's performance and learning characteristics.

## Setup Instructions

1. **Clone the Repository**: 
   ```
   git clone <repository-url>
   cd cpp-nn-evaluation
   ```

2. **Build the Project**:
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

3. **Run the Evaluation**:
   After building, run the evaluation script:
   ```
   ./tests/evaluation
   ```

## Usage

The neural network can be trained and evaluated using the provided scripts. Modify the parameters in `src/main.cpp` and `tests/evaluation.cpp` to experiment with different configurations, such as the number of epochs, learning rates, and activation functions.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.