#pragma once

#include <optix.h>
#include <optix_host.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_types.h>

std::tuple<OptixTraversableHandle, CUdeviceptr> buildPhotonAS(OptixDeviceContext ctx, const std::vector<OptixBuildInput>& buildInputs) {
    // Allocate memory for acceleration structure
    OptixAccelBuildOptions accelOptions = {
        .buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE, // TODO: Test OPTIX_BUIILD_FLAG_PREFER_FAST_BUILD
        .operation = OPTIX_BUILD_OPERATION_BUILD,
        .motionOptions = {
            .numKeys = 0,
        },
    };

    OptixAccelBufferSizes bufferSizes;
    check(optixAccelComputeMemoryUsage(ctx, &accelOptions, buildInputs.data(), buildInputs.size(), &bufferSizes));
    CUdeviceptr tempBuffer, gasBuffer; // TODO: Keep buffers around for faster build
    check(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), bufferSizes.tempSizeInBytes));
    check(cudaMalloc(reinterpret_cast<void**>(&gasBuffer), bufferSizes.outputSizeInBytes));

    OptixTraversableHandle handle;
    check(optixAccelBuild(ctx, nullptr, &accelOptions, buildInputs.data(), buildInputs.size(), tempBuffer, bufferSizes.tempSizeInBytes, gasBuffer, bufferSizes.outputSizeInBytes, &handle, nullptr, 0));
    // NOTE: We do not do compaction to reduce build time

    check(cudaFree(reinterpret_cast<void*>(tempBuffer)));

    return {handle, gasBuffer};
}

std::tuple<OptixTraversableHandle, CUdeviceptr> buildPhotonAS(OptixDeviceContext ctx, CUdeviceptr aabbBuffer, uint32_t photonQueryCount) {
    std::array<uint, 1> flags = { OPTIX_GEOMETRY_FLAG_NONE };

    const auto buildInput = OptixBuildInput {
        .type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
        .customPrimitiveArray = {
            .aabbBuffers = &aabbBuffer,
            .numPrimitives = photonQueryCount,
            .strideInBytes = 0,
            .flags = flags.data(),
            .numSbtRecords = 1,
            .sbtIndexOffsetBuffer = 0, // No sbt index offsets
            .sbtIndexOffsetSizeInBytes = 0,
            .sbtIndexOffsetStrideInBytes = 0,
            .primitiveIndexOffset = 0,
        }
    };

    return buildPhotonAS(ctx, {buildInput});
}