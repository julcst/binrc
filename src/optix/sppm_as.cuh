#pragma once

#include <thrust/device_vector.h>

#include "sppm_rtx.cuh"

struct SPPMRTX {
    OptixTraversableHandle handle = 0;
    uint32_t size = 0;
    uint32_t totalPhotonCount = 0;
    float alpha = 0.7f;
    float initialRadius = 0.05f;
    thrust::device_vector<uint8_t> tempBuffer;
    thrust::device_vector<uint8_t> gasBuffer;
    thrust::device_vector<OptixAabb> aabbBuffer;
    thrust::device_vector<PhotonQuery> queryBuffer;
    thrust::device_vector<PhotonQueryView::Atomics> atomicBuffer {1};

    // inline void initHitRecords(OptixProgramGroup pg, size_t numRecords) {
    //     std::vector<PhotonQueryRecord> hitRecordsHost(numRecords);
    //     for (auto& record : hitRecordsHost) optixSbtRecordPackHeader(pg, &record);
    //     hitRecords = hitRecordsHost;
    //     aabbBuffer.resize(numRecords);
    // }

    SPPMRTX(uint32_t n) : aabbBuffer(n), queryBuffer(n), size(n) {}

    inline void resize(uint32_t n) {
        size = n;
        aabbBuffer.reserve(n);
        queryBuffer.reserve(n);
    }

    inline void resetQueries(const PhotonQuery& q = {}) {
        queryBuffer.assign(size, q);
    }

    inline void resetCollectedPhotons() {
        thrust::for_each(queryBuffer.begin(), queryBuffer.end(), [] __device__ (PhotonQuery& q) {
            q.collectedPhotons = 0;
        });
    }

    inline PhotonQueryView getDeviceView() {
        return PhotonQueryView { 
            .queries = queryBuffer.data().get(),
            .aabbs = aabbBuffer.data().get(),
            .atomics = atomicBuffer.data().get(), // TODO: Pull request for better syntax
            .queryCount = size,
            .handle = handle,
            .totalPhotonCount = totalPhotonCount,
            .alpha = alpha,
            .initialRadius = initialRadius,
        };
    }

    inline void updatePhotonAS(OptixDeviceContext ctx, const std::vector<OptixBuildInput>& buildInputs) {
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
        // NOTE: We keep the buffers around for faster rebuilds
        tempBuffer.reserve(bufferSizes.tempSizeInBytes);
        gasBuffer.reserve(bufferSizes.outputSizeInBytes);

        check(optixAccelBuild(ctx, nullptr, &accelOptions, buildInputs.data(), buildInputs.size(), reinterpret_cast<CUdeviceptr>(tempBuffer.data().get()), bufferSizes.tempSizeInBytes, reinterpret_cast<CUdeviceptr>(gasBuffer.data().get()), bufferSizes.outputSizeInBytes, &handle, nullptr, 0));
        // NOTE: We do not compact to reduce build time
    }

    inline void updatePhotonAS(OptixDeviceContext ctx) {
        std::array<uint32_t, 1> flags = { OPTIX_GEOMETRY_FLAG_NONE | OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
        CUdeviceptr aabbBufferPtr = reinterpret_cast<CUdeviceptr>(aabbBuffer.data().get());

        const auto buildInput = OptixBuildInput {
            .type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
            .customPrimitiveArray = {
                .aabbBuffers = &aabbBufferPtr,
                .numPrimitives = size,
                .strideInBytes = 0,
                .flags = flags.data(),
                .numSbtRecords = 1,
                .sbtIndexOffsetBuffer = 0, // No sbt index offsets
                .sbtIndexOffsetSizeInBytes = 0,
                .sbtIndexOffsetStrideInBytes = 0,
                .primitiveIndexOffset = 0,
            }
        };

        updatePhotonAS(ctx, { buildInput });
    }
};