# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -O2 -Wall -I./cpp/cuda -I./cpp/utils -I/usr/local/cuda/include
NVCCFLAGS = -O2 -arch=sm_35

# Directories
SRC_DIR = cpp
CUDA_DIR = $(SRC_DIR)/cuda
UTILS_DIR = $(SRC_DIR)/utils
OBJ_DIR = obj

# Source files
CPP_SRC = $(SRC_DIR)/model.cpp
CUDA_SRC = $(CUDA_DIR)/neural_network.cu
CUDA_UTILS = $(UTILS_DIR)/cuda_utils.cuh

# Object files
OBJ_CPP = $(OBJ_DIR)/model.o
OBJ_CUDA = $(OBJ_DIR)/neural_network.o

# Executable
TARGET = vectorflow

# Create object directory if it doesn't exist
$(shell mkdir -p $(OBJ_DIR))

# Build the target
all: $(TARGET)

$(TARGET): $(OBJ_CPP) $(OBJ_CUDA)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_CPP): $(CPP_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_CUDA): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean