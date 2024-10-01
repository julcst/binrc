cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# FIXME: Why does this not happen automatically?
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang++)

# Set the project name and enable CUDA
project(biNRC LANGUAGES CXX CUDA)

# Add the executable
add_executable(${PROJECT_NAME} main.cu)

# Find and link the CUDA toolkit
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} CUDA::cudart)

# Set compiler flags
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_EXTENSIONS OFF
)