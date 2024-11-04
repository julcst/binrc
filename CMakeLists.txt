cmake_minimum_required(VERSION 3.28...3.31)

# FIXME: These should be cmake -D arguments but I am lazy
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for" FORCE)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE PATH "Path to the CUDA compiler" FORCE)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang++ CACHE PATH "Path to the host C++ compiler" FORCE)
set(OPTIX_DIR ../NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64 CACHE PATH "Path to the OptiX SDK" FORCE)

# Config
add_compile_definitions(CUDA_VERSION=12060)
add_compile_definitions(GLM_FORCE_CUDA)
set(GLM_ENABLE_FAST_MATH ON)
set(GLM_ENABLE_CXX_20 ON)
# FIXME: Clang 19.1.0 is technically not supported
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 20012 --use_fast_math -allow-unsupported-compiler") # Disables glm warnings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(biNRC LANGUAGES CXX CUDA)

# Fin dependencies
find_package(CUDAToolkit 12 REQUIRED)

include(FetchContent)
FetchContent_Declare(
    framework
    URL https://github.com/julcst/gltemplate/archive/v1.5.tar.gz
)
FetchContent_Declare(
    fastgltf
    URL https://github.com/spnda/fastgltf/archive/v0.8.0.tar.gz
)
FetchContent_MakeAvailable(framework fastgltf)

# Build the OptiX IR
set(OPTIXIR
    src/optixpathtracer.cu
)
add_library(optixir OBJECT ${OPTIXIR})
target_link_libraries(optixir CUDA::cudart glm)
target_include_directories(optixir PRIVATE ${OPTIX_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})
set_property(TARGET optixir PROPERTY CUDA_OPTIX_COMPILATION ON)

# TODO: Embed the file content
set(CONTENT "#pragma once\n#include <filesystem>\nnamespace optixir {\n")
foreach(obj $<TARGET_OBJECTS:optixir>)
    string(APPEND CONTENT "static const std::filesystem::path $<PATH:GET_STEM,${obj}>_path = \"$<TARGET_OBJECTS:optixir>\"\;\n")
endforeach()
string(APPEND CONTENT "}\n")
file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/include/optixir.hpp CONTENT ${CONTENT})

set(SRC
    src/main.cpp
    src/mainapp.cpp
    src/optixrenderer.cpp
    src/scene.cpp
)

# Add the executable
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::curand framework fastgltf)
target_include_directories(${PROJECT_NAME} PUBLIC src ${OPTIX_DIR}/include ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/include)

# Set compiler flags
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_EXTENSIONS OFF
    CUDA_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)