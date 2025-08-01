cmake_minimum_required(VERSION 3.28...3.31)

set(DEBUG_CUDA ON CACHE BOOL "Enable debugging for CUDA" FORCE)

set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for" FORCE)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "Path to the host C++ compiler" FORCE)

if (UNIX)
    set(OPTIX_DIR ../NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64 CACHE PATH "Path to the OptiX SDK" FORCE)
elseif (WIN32)
    set(OPTIX_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0" CACHE PATH "Path to the OptiX SDK" FORCE)
endif()

# Config
set(GLM_ENABLE_FAST_MATH ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 20012 --use_fast_math") # Disables glm warnings
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --debug") # Enables host debugging
if (DEBUG_CUDA)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --generate-line-info") # Enables line info
    #set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --device-debug") # Enables device debugging, disable optimizations
    message(STATUS "CUDA debugging enabled: " ${CMAKE_CUDA_FLAGS_DEBUG} ${CMAKE_CUDA_FLAGS})
    add_compile_definitions(OPTIX_DEBUG)
endif()
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(biNRC LANGUAGES CXX)

# Find dependencies
find_package(CUDAToolkit 12 REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE} CACHE PATH "Path to the CUDA compiler" FORCE)
enable_language(CUDA)

include(FetchContent)
FetchContent_Declare(
    framework
    URL https://github.com/julcst/gltemplate/archive/v1.6b.tar.gz
)
FetchContent_Declare(
    fastgltf
    URL https://github.com/spnda/fastgltf/archive/v0.8.0.tar.gz
)
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
FetchContent_Declare(
    tinycudann
    GIT_REPOSITORY https://github.com/julcst/tiny-cuda-nn
)
FetchContent_Declare(
    cuco
    GIT_REPOSITORY https://github.com/NVIDIA/cuCollections
    GIT_TAG dev
)
FetchContent_MakeAvailable(framework fastgltf tinycudann cuco)

# Build the OptiX IR
set(OPTIXIR
    src/optix/hit.cu
    src/optix/inference.cu
    src/optix/reference.cu
    src/optix/train_bidir.cu
    src/optix/train_light.cu
    src/optix/train_eye.cu
    src/optix/sppm_eye_pass.cu
    src/optix/sppm_light_pass.cu
    src/optix/sppm_rtx.cu
)
add_library(optixir OBJECT ${OPTIXIR})
add_compile_definitions(_CCCL_DISABLE_INT128 _LIBCUDACXX_HAS_NO_INT128)
target_link_libraries(optixir CUDA::cudart CUDA::curand tiny-cuda-nn cuco)
target_include_directories(optixir PRIVATE src ${OPTIX_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(optixir PROPERTIES
    CUDA_OPTIX_COMPILATION ON
)

set(SRC
    src/brdflut.cu
    src/main.cu
    src/mainapp.cu
    src/optixrenderer.cu
    src/scene.cpp
)

# Add the executable
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::curand framework fastgltf tiny-cuda-nn cuco)
target_include_directories(${PROJECT_NAME} PUBLIC src ${OPTIX_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})

# Set compiler flags
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_EXTENSIONS OFF
    CUDA_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

# TODO: Embed the file content
string(APPEND CONTENT "#pragma once\n\n#include <filesystem>\n#include <array>\n\nnamespace optixir {")
string(APPEND CONTENT "\n\tconstexpr std::array paths = {\n\t\t\"")
string(APPEND CONTENT $<JOIN:$<LIST:SORT,$<TARGET_OBJECTS:optixir>>,$<QUOTE>$<COMMA>\n\t\t$<QUOTE>>)
string(APPEND CONTENT "\"\n\t};")
string(APPEND CONTENT "\n\tenum Modules {\n\t\t")
string(APPEND CONTENT $<JOIN:$<UPPER_CASE:$<PATH:GET_STEM,$<LIST:SORT,$<TARGET_OBJECTS:optixir>>>>,$<COMMA>\n\t\t>)
string(APPEND CONTENT "\n\t};")
string(APPEND CONTENT "\n}\n")
set(OPTIXIR_DIR "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>/include")
file(GENERATE OUTPUT ${OPTIXIR_DIR}/optixir.hpp CONTENT "${CONTENT}" TARGET ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PRIVATE ${OPTIXIR_DIR})