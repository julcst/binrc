#include "scene.hpp"

#include <cuda_runtime.h>
#include <cudautil.hpp>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_types.h>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/math.hpp>

#include <glm/glm.hpp>
using namespace glm;

#include <vector>
#include <array>
#include <variant>
#include <format>

Scene::~Scene() {
    free();
}

void Scene::free() {
    check(cudaFree(reinterpret_cast<void*>(instances)));
    for (auto buffer : gasBuffers) {
        check(cudaFree(reinterpret_cast<void*>(buffer)));
    }
    check(cudaFree(reinterpret_cast<void*>(iasBuffer)));
}

OptixTraversableHandle Scene::loadGLTF(OptixDeviceContext ctx, const std::filesystem::path& path) {
    // Free previous scene
    free();

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
    std::array<uint, 1> flags = { OPTIX_GEOMETRY_FLAG_NONE };
    std::vector<std::vector<OptixTraversableHandle>> gasHandles;
    gasHandles.reserve(nPrimitives);

    for (const auto& mesh : asset->meshes) {
        gasHandles.emplace_back();
        for (const auto& primitive : mesh.primitives) {
            auto& posAcc = asset->accessors[primitive.findAttribute("POSITION")->accessorIndex];
            auto nVertices = posAcc.count;
            vec4* vertices;
            // TODO: Free
            check(cudaMallocManaged(reinterpret_cast<void**>(&vertices), nVertices * sizeof(vec4)));
            fastgltf::iterateAccessorWithIndex<vec3>(asset.get(), posAcc, [&](const vec3& vertex, auto i) {
                vertices[i] = vec4(vertex, 1.0f);
            });

            auto& indexAcc = asset->accessors[primitive.indicesAccessor.value()];
            auto nTriangles = indexAcc.count / 3;
            uint* indices;
            // TODO: Free
            check(cudaMallocManaged(reinterpret_cast<void**>(&indices), nTriangles * sizeof(uint3)));
            fastgltf::iterateAccessorWithIndex<uint>(asset.get(), indexAcc, [&](const uint& index, auto i) {
                indices[i] = index;
            });

            const auto handle = buildGAS(ctx, { OptixBuildInput {
                .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                .triangleArray = {
                    .vertexBuffers = reinterpret_cast<CUdeviceptr*>(&vertices),
                    .numVertices = static_cast<unsigned int>(nVertices),
                    .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
                    .vertexStrideInBytes = sizeof(vec4), // 16 byte stride for better performance
                    .indexBuffer = reinterpret_cast<CUdeviceptr>(indices),
                    .numIndexTriplets = static_cast<unsigned int>(nTriangles),
                    .indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
                    .indexStrideInBytes = 0,
                    .flags = flags.data(),
                    .numSbtRecords = 1,
                },
            }});
            gasHandles.back().push_back(handle);

            check(cudaFree(reinterpret_cast<void*>(vertices)));
            check(cudaFree(reinterpret_cast<void*>(indices)));
        }
    }

    uint i = 0;
    for (const auto& node : asset->nodes) {
        if (auto m = node.meshIndex; m.has_value()) {
            auto mesh = asset->meshes[m.value()];
            for (uint j = 0; j < mesh.primitives.size(); j++) {
                const auto& primitive = mesh.primitives[j];
                fastgltf::math::fmat4x4 mat;
                auto* trs = std::get_if<fastgltf::TRS>(&node.transform);
                auto* matrix = std::get_if<fastgltf::math::fmat4x4>(&node.transform);
                if (trs) {
                    mat = fastgltf::math::scale(fastgltf::math::rotate(fastgltf::math::translate(fastgltf::math::fmat4x4(1.0f), trs->translation), trs->rotation), trs->scale);
                } else if (matrix) {
                    mat = *matrix;
                }
                auto& gasHandle = gasHandles[m.value()][j];
                instances[i] = OptixInstance {
                    .transform = {
                        mat.row(0)[0], mat.row(0)[1], mat.row(0)[2], mat.row(0)[3],
                        mat.row(1)[0], mat.row(1)[1], mat.row(1)[2], mat.row(1)[3],
                        mat.row(2)[0], mat.row(2)[1], mat.row(2)[2], mat.row(2)[3],
                    },
                    .instanceId = i,
                    .sbtOffset = 0,
                    .visibilityMask = 255,
                    .flags = OPTIX_INSTANCE_FLAG_NONE,
                    .traversableHandle = gasHandle,
                };
                i++;
            }
        }
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
    CUdeviceptr tempBuffer, gasBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), bufferSizes.tempSizeInBytes));
    check(cudaMalloc(reinterpret_cast<void**>(&gasBuffer), bufferSizes.outputSizeInBytes));

    OptixTraversableHandle handle;
    optixAccelBuild(ctx, nullptr, &accelOptions, buildInputs.data(), buildInputs.size(), tempBuffer, bufferSizes.tempSizeInBytes, gasBuffer, bufferSizes.outputSizeInBytes, &handle, nullptr, 0);

    check(cudaFree(reinterpret_cast<void*>(tempBuffer)));

    gasBuffers.push_back(gasBuffer);

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