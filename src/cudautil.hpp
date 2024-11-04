#pragma once

#include <optix.h>
#include <cuda.h>
#include <curand.h>

#include <iostream>
#include <format>

// OPTIX check as constexpr function
constexpr void check(OptixResult res) {
    if (res != OPTIX_SUCCESS) {
        throw std::runtime_error(optixGetErrorName(res));
    }
}

// CUDA check as constexpr function
constexpr void check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorName(error));
    }
}

// CURAND check as constexpr function
constexpr void check(curandStatus_t status) {
    switch (status) {
        case CURAND_STATUS_SUCCESS:
            return;
        case CURAND_STATUS_VERSION_MISMATCH:
            throw std::runtime_error("CURAND_STATUS_VERSION_MISMATCH");
        case CURAND_STATUS_NOT_INITIALIZED:
            throw std::runtime_error("CURAND_STATUS_NOT_INITIALIZED");
        case CURAND_STATUS_ALLOCATION_FAILED:
            throw std::runtime_error("CURAND_STATUS_ALLOCATION_FAILED");
        case CURAND_STATUS_TYPE_ERROR:
            throw std::runtime_error("CURAND_STATUS_TYPE_ERROR");
        case CURAND_STATUS_OUT_OF_RANGE:
            throw std::runtime_error("CURAND_STATUS_OUT_OF_RANGE");
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            throw std::runtime_error("CURAND_STATUS_LENGTH_NOT_MULTIPLE");
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            throw std::runtime_error("CURAND_STATUS_DOUBLE_PRECISION_REQUIRED");
        case CURAND_STATUS_LAUNCH_FAILURE:
            throw std::runtime_error("CURAND_STATUS_LAUNCH_FAILURE");
        case CURAND_STATUS_PREEXISTING_FAILURE:
            throw std::runtime_error("CURAND_STATUS_PREEXISTING_FAILURE");
        case CURAND_STATUS_INITIALIZATION_FAILED:
            throw std::runtime_error("CURAND_STATUS_INITIALIZATION_FAILED");
        case CURAND_STATUS_ARCH_MISMATCH:
            throw std::runtime_error("CURAND_STATUS_ARCH_MISMATCH");
        case CURAND_STATUS_INTERNAL_ERROR:
            throw std::runtime_error("CURAND_STATUS_INTERNAL_ERROR");
        default:
            throw std::runtime_error("Unknown CURAND error");
    }
}

static void printCudaDevices() {
    int deviceCount, device;
    check(cudaGetDeviceCount(&deviceCount));
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, device));
        std::cout << std::format("Device {} - {}, compute capability {}.{}, cores {}, warp size {}", device, prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.warpSize) << std::endl;
    }
}