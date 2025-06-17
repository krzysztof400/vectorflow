# VectorFlow Pure C++ Neural Network

## Overview
VectorFlow is a pure C++ implementation of a neural network that utilizes CUDA for accelerated computations. This project includes the core neural network functionality, as well as CUDA implementations to enhance performance during training and prediction.

## Project Structure
```
vectorflow-purecpp
├── cpp
│   ├── model.cpp            # Main implementation of the neural network
│   ├── cuda
│   │   ├── neural_network.cu # CUDA implementation of neural network functions
│   │   └── neural_network.cuh # Header file for CUDA functions and kernels
│   └── utils
│       └── cuda_utils.cuh    # Utility functions for CUDA operations
├── Makefile                  # Build instructions for the project
└── README.md                 # Project documentation
```

## Requirements
- CUDA Toolkit
- A compatible NVIDIA GPU
- C++ compiler (g++ or similar)

## Building the Project
To build the project, navigate to the root directory of the project and run the following command:

```
make
```

This will compile the C++ and CUDA files and create the executable.

## Running the Project
After building the project, you can run the executable generated in the build process. The program will train the neural network on randomly generated data and then evaluate its performance on test data.

## Dependencies
Ensure that you have the following installed:
- CUDA Toolkit (version compatible with your GPU)
- CMake (optional, for managing builds)
- A C++ compiler that supports C++11 or later

## Usage
1. Clone the repository or download the project files.
2. Install the required dependencies.
3. Build the project using the provided Makefile.
4. Execute the compiled program to see the results of the neural network training and predictions.

## Contributing
Contributions are welcome! If you have suggestions for improvements or additional features, feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.