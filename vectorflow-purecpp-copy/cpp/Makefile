# Compiler settings
CXX = g++
NVCC = nvcc

# Compiler flags
CXX_FLAGS = -O3 -std=c++11
NVCC_FLAGS = -O3 -arch=sm_50 -std=c++11

# Libraries
LIBS = -lcublas -lcudart

# Source files
CPP_SOURCES = model.cpp matrix_utils.cpp
CUDA_SOURCES = matrixMul.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Target executable
TARGET = vectorflow

# Default target
all: $(TARGET)

# Link all objects
$(TARGET): $(CPP_OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(CPP_OBJECTS) $(CUDA_OBJECTS) $(TARGET)

# Debug build
debug: CXX_FLAGS += -g -DDEBUG
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: clean all

.PHONY: all clean debug