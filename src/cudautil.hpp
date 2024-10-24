#pragma once

#include <optix.h>
#include <cuda.h>

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

static void printCudaDevices() {
    int deviceCount, device;
    check(cudaGetDeviceCount(&deviceCount));
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, device));
        std::cout << std::format("Device {} - {}, compute capability {}.{}, cores {}, warp size {}", device, prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.warpSize) << std::endl;
    }
}