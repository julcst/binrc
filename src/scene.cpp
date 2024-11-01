#include "scene.hpp"

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
    // Parse GLTF file
    auto parser = fastgltf::Parser(fastgltf::Extensions::None);
    auto data = fastgltf::GltfDataBuffer::FromPath(path);
    if (auto e = data.error(); e != fastgltf::Error::None) throw std::runtime_error(std::format("Error: {}", fastgltf::getErrorMessage(e)));
    auto asset = parser.loadGltf(data.get(), path.parent_path(), fastgltf::Options::GenerateMeshIndices);
    if (auto e = asset.error(); e != fastgltf::Error::None) throw std::runtime_error(std::format("Error: {}", fastgltf::getErrorMessage(e)));

    // Count number of instances
    nInstances = 0;
    for (const auto& node : asset->nodes) {
        if (auto i = node.meshIndex; i.has_value()) {
            nInstances += asset->meshes[i.value()].primitives.size();
        }
    }
    check(cudaMallocManaged(reinterpret_cast<void**>(&instances), nInstances * sizeof(OptixInstance)));

    // Count number of primitives
    size_t nPrimitives = 0;
    for (const auto& mesh : asset->meshes) {
        nPrimitives += mesh.primitives.size();
    }

    // Build geometry
    std::vector<OptixBuildInput> buildInputs;
    buildInputs.reserve(nPrimitives);
    std::array<uint, 1> flags = { OPTIX_GEOMETRY_FLAG_NONE };
    std::vector<std::array<CUdeviceptr, 1>> vertexBuffers; // Store pointers to device buffers in host memory
    vertexBuffers.reserve(nPrimitives);

    for (const auto& mesh : asset->meshes) {
        for (const auto& primitive : mesh.primitives) {

            auto& posAcc = asset->accessors[primitive.findAttribute("POSITION")->accessorIndex];
            auto nVertices = posAcc.count;
            vec4* vertices;
            // TODO: Free
            check(cudaMallocManaged(reinterpret_cast<void**>(&vertices), nVertices * sizeof(vec4)));
            fastgltf::iterateAccessorWithIndex<vec3>(asset.get(), posAcc, [&](const vec3& vertex, auto i) {
                vertices[i] = vec4(vertex, 1.0f);
            });
            vertexBuffers.push_back({ reinterpret_cast<CUdeviceptr>(vertices) });

            auto& indexAcc = asset->accessors[primitive.indicesAccessor.value()];
            auto nTriangles = indexAcc.count / 3;
            uint* indices;
            // TODO: Free
            check(cudaMallocManaged(reinterpret_cast<void**>(&indices), nTriangles * sizeof(uint3)));
            fastgltf::iterateAccessorWithIndex<uint>(asset.get(), indexAcc, [&](const uint& index, auto i) {
                indices[i] = index;
            });

            buildInputs.push_back(OptixBuildInput {
                .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                .triangleArray = {
                    .vertexBuffers = vertexBuffers.back().data(),
                    .numVertices = static_cast<unsigned int>(nVertices),
                    .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
                    .vertexStrideInBytes = sizeof(vec4),
                    .indexBuffer = reinterpret_cast<CUdeviceptr>(indices),
                    .numIndexTriplets = static_cast<unsigned int>(nTriangles),
                    .indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
                    .indexStrideInBytes = 0,
                    .flags = flags.data(),
                    .numSbtRecords = 1,
                },
            });
        }
    }

    for (const auto& node : asset->nodes) {
        if (auto i = node.meshIndex; i.has_value()) {
            auto mesh = asset->meshes[i.value()];
            for (auto primitive : mesh.primitives) {
                
            }
        }
    }

    const auto handle = buildGAS(ctx, buildInputs);

    // Free vertex and index buffers
    for (auto buildInput : buildInputs) {
        check(cudaFree(reinterpret_cast<void*>(buildInput.triangleArray.vertexBuffers[0])));
        check(cudaFree(reinterpret_cast<void*>(buildInput.triangleArray.indexBuffer)));
    }

    for (uint i = 0; i < nInstances; i++) {
        instances[i] = OptixInstance {
            .transform = {
                1.0f, 0.0f, 0.0f, i * 1.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
            },
            .instanceId = i,
            .sbtOffset = 0,
            .visibilityMask = 255,
            .flags = OPTIX_INSTANCE_FLAG_NONE,
            .traversableHandle = handle,
        };
    }

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