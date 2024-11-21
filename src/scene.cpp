#include "scene.hpp"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>

#include <cuda_runtime.h>
#include <cudautil.hpp>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_types.h>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>

#include <glm/glm.hpp>
using namespace glm;

#include <vector>
#include <array>

Scene::~Scene() {
    check(cudaFree(reinterpret_cast<void*>(instances)));
    check(cudaFree(reinterpret_cast<void*>(gasBuffer)));
    check(cudaFree(reinterpret_cast<void*>(iasBuffer)));
}

OptixTraversableHandle Scene::loadGLTF(OptixDeviceContext ctx, const std::filesystem::path& path) {
    std::vector<float3> vertices = {
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
    };
    CUdeviceptr vertexBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&vertexBuffer), vertices.size() * sizeof(float3)));
    check(cudaMemcpy(reinterpret_cast<void*>(vertexBuffer), vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));

    std::vector<uint3> indices = {
        make_uint3(0, 1, 2),
    };
    CUdeviceptr indexBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&indexBuffer), indices.size() * sizeof(uint3)));
    check(cudaMemcpy(reinterpret_cast<void*>(indexBuffer), indices.data(), indices.size() * sizeof(uint3), cudaMemcpyHostToDevice));

    std::array<uint, 1> flags = { OPTIX_GEOMETRY_FLAG_NONE };

    const auto buildInput = OptixBuildInput {
        .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
        .triangleArray = {
            .vertexBuffers = &vertexBuffer,
            .numVertices = static_cast<unsigned int>(vertices.size()),
            .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
            .vertexStrideInBytes = sizeof(vec3),
            .indexBuffer = indexBuffer,
            .numIndexTriplets = static_cast<unsigned int>(indices.size()),
            .indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
            .indexStrideInBytes = 0,
            .flags = flags.data(),
            .numSbtRecords = 1,
        },
    };
    const auto handle = buildGAS(ctx, {buildInput});

    nInstances = 1;
    const std::vector<OptixInstance> instancesHost = {
        OptixInstance {
            .transform = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
            },
            .instanceId = 0,
            .sbtOffset = 0,
            .visibilityMask = 255,
            .flags = OPTIX_INSTANCE_FLAG_NONE,
            .traversableHandle = handle,
        },
    };
    check(cudaMalloc(reinterpret_cast<void**>(&instances), nInstances * sizeof(OptixInstance)));
    check(cudaMemcpy(reinterpret_cast<void*>(instances), instancesHost.data(), nInstances * sizeof(OptixInstance), cudaMemcpyHostToDevice));

    return buildIAS(ctx);
}

OptixTraversableHandle Scene::buildGAS(OptixDeviceContext ctx, const std::vector<OptixBuildInput>& buildInputs) {
    // Allocate memory for acceleration structure
    OptixAccelBuildOptions accelOptions = {
        .buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        .operation = OPTIX_BUILD_OPERATION_BUILD,
        .motionOptions = {
            .numKeys = 0,
        },
    };
    OptixAccelBufferSizes bufferSizes;
    check(optixAccelComputeMemoryUsage(ctx, &accelOptions, buildInputs.data(), buildInputs.size(), &bufferSizes));
    CUdeviceptr tempBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), bufferSizes.tempSizeInBytes));

    // TODO: Only reallocate if necessary
    check(cudaFree(reinterpret_cast<void*>(gasBuffer)));
    check(cudaMalloc(reinterpret_cast<void**>(&gasBuffer), bufferSizes.outputSizeInBytes));

    OptixTraversableHandle handle;
    optixAccelBuild(ctx, nullptr, &accelOptions, buildInputs.data(), buildInputs.size(), tempBuffer, bufferSizes.tempSizeInBytes, gasBuffer, bufferSizes.outputSizeInBytes, &handle, nullptr, 0);

    check(cudaFree(reinterpret_cast<void*>(tempBuffer)));

    return handle;
}

OptixTraversableHandle Scene::buildIAS(OptixDeviceContext ctx) {
    OptixBuildInput buildInput = {
        .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = {
            .instances = reinterpret_cast<CUdeviceptr>(instances),
            .numInstances = nInstances,
        },
    };

    OptixAccelBuildOptions accelOptions = {
        .buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        .operation = OPTIX_BUILD_OPERATION_BUILD,
        .motionOptions = {
            .numKeys = 0,
        },
    };
    OptixAccelBufferSizes bufferSizes;
    check(optixAccelComputeMemoryUsage(ctx, &accelOptions, &buildInput, 1, &bufferSizes));
    CUdeviceptr tempBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), bufferSizes.tempSizeInBytes));

    // TODO: Only reallocate if necessary
    check(cudaFree(reinterpret_cast<void*>(iasBuffer)));
    check(cudaMalloc(reinterpret_cast<void**>(&iasBuffer), bufferSizes.outputSizeInBytes));

    OptixTraversableHandle handle;
    optixAccelBuild(ctx, nullptr, &accelOptions, &buildInput, 1, tempBuffer, bufferSizes.tempSizeInBytes, iasBuffer, bufferSizes.outputSizeInBytes, &handle, nullptr, 0);

    check(cudaFree(reinterpret_cast<void*>(tempBuffer)));

    return handle;
}

OptixTraversableHandle Scene::updateIAS(OptixDeviceContext ctx) {
    OptixBuildInput buildInput = {
        .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = {
            .instances = reinterpret_cast<CUdeviceptr>(instances),
            .numInstances = nInstances,
        },
    };

    OptixAccelBuildOptions accelOptions = {
        .buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        .operation = OPTIX_BUILD_OPERATION_UPDATE,
        .motionOptions = {
            .numKeys = 0,
        },
    };
    OptixAccelBufferSizes bufferSizes;
    check(optixAccelComputeMemoryUsage(ctx, &accelOptions, nullptr, 0, &bufferSizes));
    CUdeviceptr d_tempBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), bufferSizes.tempUpdateSizeInBytes));

    OptixTraversableHandle handle;
    optixAccelBuild(ctx, nullptr, &accelOptions, &buildInput, 1, d_tempBuffer, bufferSizes.tempSizeInBytes, iasBuffer, bufferSizes.outputSizeInBytes, &handle, nullptr, 0);

    check(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));

    return handle;
}