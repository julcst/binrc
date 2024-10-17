cmake_minimum_required(VERSION 3.28)

# FIXME: These should be cmake -D arguments but I am lazy
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang++)

# Set the project name and enable CUDA
project(biNRC LANGUAGES CXX CUDA)

set(SRC
    src/main.cpp
    src/mainapp.cu
)

# Add the executable
add_executable(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PRIVATE src)

add_compile_definitions(CUDA_VERSION=12060)
add_compile_definitions(GLM_FORCE_CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 20012")

# Fetch framework
include(FetchContent)
FetchContent_Declare(
    framework
    URL https://github.com/julcst/gltemplate/archive/v1.5.tar.gz
)
FetchContent_MakeAvailable(framework)

# Find and link the CUDA toolkit
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} CUDA::cudart framework)

# Set compiler flags
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)