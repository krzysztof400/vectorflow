cmake_minimum_required(VERSION 3.10)

project(cpp_nn_evaluation)

set(CMAKE_CXX_STANDARD 11)

# Include directories
include_directories(include)

# Library source files
set(LIB_SOURCES
    src/model.cpp
    src/data_utils.cpp
)

# Create a static library from the shared sources
add_library(nn_lib STATIC ${LIB_SOURCES})

# Main executable
add_executable(cpp_nn_evaluation src/main.cpp)

# Link the main executable against the library
target_link_libraries(cpp_nn_evaluation nn_lib)

# Test executable
add_executable(evaluation_test tests/evaluation.cpp)

# Link the test executable against the library
target_link_libraries(evaluation_test nn_lib)