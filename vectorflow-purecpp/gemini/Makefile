# Makefile for CUDA Neural Network

# Compiler settings
NVCC = nvcc
CXX = g++

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_50 -std=c++11
CXX_FLAGS = -O3 -std=c++11

# Libraries
LIBS = -lcublas -lcudart

# Source files
CUDA_SOURCES = cuda_kernels.cu
CPP_SOURCES = model.cpp main.cpp
HEADERS = NeuralNetwork.h cuda_kernels.h cuda_utils.h

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
OBJECTS = $(CUDA_OBJECTS) $(CPP_OBJECTS)

# Target executable
TARGET = neural_network

# Default target
all: $(TARGET)

# Link all objects
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install nvidia-cuda-toolkit

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: $(TARGET)

# Release build (default)
release: $(TARGET)

# Print build info
info:
	@echo "CUDA Compiler: $(NVCC)"
	@echo "C++ Compiler: $(CXX)"
	@echo "NVCC Flags: $(NVCC_FLAGS)"
	@echo "Libraries: $(LIBS)"
	@echo "Source Files: $(CUDA_SOURCES) $(CPP_SOURCES)"
	@echo "Target: $(TARGET)"

# Check CUDA installation
check-cuda:
	nvcc --version
	nvidia-smi

.PHONY: all clean install-deps debug release info check-cuda