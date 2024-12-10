#include "scene.hpp"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/math.hpp>

#include <cuda_runtime.h>
#include <cudautil.hpp>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_types.h>

#include <stb_image.h>

#include <glm/glm.hpp>
using namespace glm;

#include <vector>
#include <array>
#include <variant>
#include <format>
#include <tuple>

#include <framework/context.hpp>

#include "optixparams.cuh"
#include "cudaglm.cuh"

Geometry::~Geometry() {
    check(cudaFree(reinterpret_cast<void*>(gasBuffer)));
    check(cudaFree(reinterpret_cast<void*>(indexBuffer)));
    check(cudaFree(reinterpret_cast<void*>(vertexData)));
}

Scene::~Scene() {
    check(cudaFree(reinterpret_cast<void*>(iasBuffer)));
    for (auto& image : images) {
        check(cudaFreeArray(image));
    }
    for (auto& texture : textures) {
        check(cudaDestroyTextureObject(texture));
    }
}

std::tuple<OptixTraversableHandle, CUdeviceptr> buildGAS(OptixDeviceContext ctx, const std::vector<OptixBuildInput>& buildInputs) {
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

    // TODO: Compact

    check(cudaFree(reinterpret_cast<void*>(tempBuffer)));

    return {handle, gasBuffer};
}

std::tuple<OptixTraversableHandle, CUdeviceptr, CUdeviceptr> buildGAS(OptixDeviceContext ctx, const std::vector<vec4>& vertices, const std::vector<uint>& indices) {
    std::array<uint, 1> flags = { OPTIX_GEOMETRY_FLAG_NONE };

    CUdeviceptr vertexBuffer, indexBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&vertexBuffer), vertices.size() * sizeof(float4)));
    check(cudaMemcpy(reinterpret_cast<void*>(vertexBuffer), vertices.data(), vertices.size() * sizeof(float4), cudaMemcpyHostToDevice));
    check(cudaMalloc(reinterpret_cast<void**>(&indexBuffer), indices.size() * sizeof(uint)));
    check(cudaMemcpy(reinterpret_cast<void*>(indexBuffer), indices.data(), indices.size() * sizeof(uint), cudaMemcpyHostToDevice));

    const auto buildInput = OptixBuildInput {
        .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
        .triangleArray = {
            .vertexBuffers = &vertexBuffer,
            .numVertices = static_cast<unsigned int>(vertices.size()),
            .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
            .vertexStrideInBytes = sizeof(vec4), // 16 byte stride for better performance
            .indexBuffer = indexBuffer,
            .numIndexTriplets = static_cast<unsigned int>(indices.size() / 3),
            .indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
            .indexStrideInBytes = 0,
            .preTransform = 0,
            .flags = flags.data(),
            .numSbtRecords = 1,
            .sbtIndexOffsetBuffer = 0,
            .sbtIndexOffsetSizeInBytes = 0,
            .sbtIndexOffsetStrideInBytes = 0,
            .primitiveIndexOffset = 0,
            .transformFormat = OPTIX_TRANSFORM_FORMAT_NONE,
        },
    };

    const auto [handle, gasBuffer] = buildGAS(ctx, {buildInput});

    check(cudaFree(reinterpret_cast<void*>(vertexBuffer)));

    return {handle, gasBuffer, indexBuffer};
}

std::tuple<OptixTraversableHandle, CUdeviceptr> buildIAS(OptixDeviceContext ctx, const std::vector<OptixInstance>& instances) {
    CUdeviceptr instanceBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&instanceBuffer), instances.size() * sizeof(OptixInstance)));
    check(cudaMemcpy(reinterpret_cast<void*>(instanceBuffer), instances.data(), instances.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));

    OptixBuildInput buildInput = {
        .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = {
            .instances = instanceBuffer,
            .numInstances = static_cast<unsigned int>(instances.size()),
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
    CUdeviceptr tempBuffer, iasBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), bufferSizes.tempSizeInBytes));
    check(cudaMalloc(reinterpret_cast<void**>(&iasBuffer), bufferSizes.outputSizeInBytes));

    OptixTraversableHandle handle;
    optixAccelBuild(ctx, nullptr, &accelOptions, &buildInput, 1, tempBuffer, bufferSizes.tempSizeInBytes, iasBuffer, bufferSizes.outputSizeInBytes, &handle, nullptr, 0);

    // TODO: Compact

    check(cudaFree(reinterpret_cast<void*>(instanceBuffer)));
    check(cudaFree(reinterpret_cast<void*>(tempBuffer)));

    return {handle, iasBuffer};
}

float3 toVec3(const fastgltf::math::fvec4& v) {
    return make_float3(v.x(), v.y(), v.z());
}

cudaTextureObject_t createTextureObject(cudaArray_t image, int srgb) {
    cudaResourceDesc resDesc = {
        .resType = cudaResourceTypeArray,
        .res = { .array = image },
    };
    cudaTextureDesc texDesc = {
        .addressMode = { cudaAddressModeWrap, cudaAddressModeWrap, cudaAddressModeWrap },
        .filterMode = cudaFilterModeLinear,
        .readMode = cudaReadModeNormalizedFloat,
        .sRGB = srgb,
        .borderColor = { 0, 0, 0, 0 },
        .normalizedCoords = 1,
        .maxAnisotropy = 0,
        .mipmapFilterMode = cudaFilterModePoint,
        .mipmapLevelBias = 0,
        .minMipmapLevelClamp = 0.0f,
        .maxMipmapLevelClamp = 0.0f,
        .disableTrilinearOptimization = 0,
        .seamlessCubemap = 0,
    };
    cudaTextureObject_t texObj;
    check(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    return texObj;
}

void Scene::loadGLTF(OptixDeviceContext ctx, Params* params, OptixProgramGroup& program, OptixShaderBindingTable& sbt, const std::filesystem::path& path) {
    // Parse GLTF file
    auto parser = fastgltf::Parser(fastgltf::Extensions::KHR_materials_transmission);
    auto data = fastgltf::GltfDataBuffer::FromPath(path);
    if (auto e = data.error(); e != fastgltf::Error::None) throw std::runtime_error(std::format("Error: {}", fastgltf::getErrorMessage(e)));
    auto asset = parser.loadGltf(data.get(), path.parent_path(), fastgltf::Options::GenerateMeshIndices);
    if (auto e = asset.error(); e != fastgltf::Error::None) throw std::runtime_error(std::format("Error: {}", fastgltf::getErrorMessage(e)));

    // NOTE: Images are not freed before loading new images -> Potential memory leak
    images.reserve(asset->images.size());
    for (const auto& image : asset->images) {
        const auto& data = std::get<fastgltf::sources::BufferView>(image.data);
        const auto& bufferView = asset->bufferViews.at(data.bufferViewIndex);
        const auto& buffer = asset->buffers.at(bufferView.bufferIndex);
        const auto& array = std::get<fastgltf::sources::Array>(buffer.data);
        int width, height, channels;
        auto* imageData = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(array.bytes.data() + bufferView.byteOffset), bufferView.byteLength, &width, &height, &channels, STBI_rgb_alpha);
        std::cout << "Loaded image: " << image.name << " (" << width << "x" << height << "x" << channels << ")" << std::endl;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

        cudaArray_t cuArray;
        check(cudaMallocArray(&cuArray, &channelDesc, width, height));
        const size_t spitch = width * sizeof(uchar4);
        check(cudaMemcpy2DToArray(cuArray, 0, 0, imageData, spitch, spitch, height, cudaMemcpyHostToDevice));

        stbi_image_free(imageData);
        images.push_back(cuArray);
    }

    // NOTE: Textures are not freed before loading new textures -> Potential memory leak
    textures.reserve(asset->textures.size());

    // Count number of instances
    uint nInstances = 0;
    for (const auto& node : asset->nodes) {
        if (auto i = node.meshIndex; i.has_value()) {
            nInstances += asset->meshes[i.value()].primitives.size();
        }
    }

    // Count number of geometries
    uint nGeometries = 0;
    for (const auto& mesh : asset->meshes) {
        nGeometries += mesh.primitives.size();
    }
    std::vector<HitRecord> hitRecords(nGeometries);

    // Create materials
    uint nMaterials = asset->materials.size();
    std::vector<Material> materials(nMaterials);
    for (uint i = 0; i < nMaterials; i++) {
        const auto& material = asset->materials[i];

        materials[i] = Material {
            .color = toVec3(material.pbrData.baseColorFactor),
            .roughness = material.pbrData.roughnessFactor,
            .metallic = material.pbrData.metallicFactor,
            .transmission = 0.0f,
            .baseMap = 0,
            .normalMap = 0,
            .mrMap = 0,
        };

        if (material.transmission) {
            materials[i].transmission = material.transmission->transmissionFactor;
            std::cout << "Transmission factor: " << material.transmission->transmissionFactor;
        }

        if (auto& tex = material.pbrData.baseColorTexture; tex.has_value()) {
            const auto imageIdx = asset->textures.at(tex.value().textureIndex).imageIndex;
            if (!imageIdx.has_value()) throw std::runtime_error("Texture has no image index");
            const auto& image = images.at(imageIdx.value());
            materials[i].baseMap = createTextureObject(image, 1);
            textures.push_back(materials[i].baseMap);
        }

        if (auto& tex = material.normalTexture; tex.has_value()) {
            const auto imageIdx = asset->textures.at(tex.value().textureIndex).imageIndex;
            std::cout << "Ignored normal texture scale: " << tex.value().scale << "\n";
            if (!imageIdx.has_value()) throw std::runtime_error("Texture has no image index");
            const auto& image = images.at(imageIdx.value());
            materials[i].normalMap = createTextureObject(image, 0);
            textures.push_back(materials[i].normalMap);
        }

        if (auto& tex = material.pbrData.metallicRoughnessTexture; tex.has_value()) {
            const auto imageIdx = asset->textures.at(tex.value().textureIndex).imageIndex;
            if (!imageIdx.has_value()) throw std::runtime_error("Texture has no image index");
            const auto& image = images.at(imageIdx.value());
            materials[i].mrMap = createTextureObject(image, 0);
            textures.push_back(materials[i].mrMap);
        }
    }

    // Build geometry and free previous geometry buffers
    geometryTable = std::vector<std::vector<Geometry>>(asset->meshes.size());

    uint geometryID = 0;
    for (uint i = 0; i < asset->meshes.size(); i++) {
        const auto& mesh = asset->meshes[i];
        for (const auto& primitive : mesh.primitives) {
            auto& posAcc = asset->accessors[primitive.findAttribute("POSITION")->accessorIndex];
            std::vector<vec4> vertices(posAcc.count);
            fastgltf::iterateAccessorWithIndex<vec3>(asset.get(), posAcc, [&](const vec3& vertex, auto i) {
                vertices[i] = vec4(vertex, 1.0f);
            });

            auto& indexAcc = asset->accessors[primitive.indicesAccessor.value()];
            std::vector<uint> indices(indexAcc.count);
            fastgltf::iterateAccessorWithIndex<uint>(asset.get(), indexAcc, [&](const uint& index, auto i) {
                indices[i] = index;
            });

            const auto [handle, gasBuffer, indexBuffer] = buildGAS(ctx, vertices, indices);

            std::vector<VertexData> vertexData(vertices.size());
            auto& normalAcc = asset->accessors.at(primitive.findAttribute("NORMAL")->accessorIndex);
            fastgltf::iterateAccessorWithIndex<vec3>(asset.get(), normalAcc, [&](const vec3& normal, auto i) {
                vertexData[i].normal = glmToCuda(normal);
            });
            auto& texCoordAcc = asset->accessors.at(primitive.findAttribute("TEXCOORD_0")->accessorIndex);
            fastgltf::iterateAccessorWithIndex<vec2>(asset.get(), texCoordAcc, [&](const vec2& texCoord, auto i) {
                vertexData[i].texCoord = glmToCuda(texCoord);
            });
            auto& tangentAcc = asset->accessors.at(primitive.findAttribute("TANGENT")->accessorIndex);
            fastgltf::iterateAccessorWithIndex<vec4>(asset.get(), tangentAcc, [&](const vec4& tangent, auto i) {
                vertexData[i].tangent = glmToCuda(tangent);
            });

            CUdeviceptr vertexDataBuffer;
            check(cudaMalloc(reinterpret_cast<void**>(&vertexDataBuffer), vertexData.size() * sizeof(VertexData)));
            check(cudaMemcpy(reinterpret_cast<void*>(vertexDataBuffer), vertexData.data(), vertexData.size() * sizeof(VertexData), cudaMemcpyHostToDevice));

            check(optixSbtRecordPackHeader(program, &hitRecords[geometryID]));
            uint materialID = primitive.materialIndex.value_or(0);
            hitRecords[geometryID].data = HitData {
                .indexBuffer = reinterpret_cast<uint3*>(indexBuffer),
                .vertexData = reinterpret_cast<VertexData*>(vertexDataBuffer),
                .materialID = materialID,
            };

            geometryTable[i].emplace_back(handle, gasBuffer, vertexDataBuffer, indexBuffer, geometryID);
            geometryID++;

            std::cout << "Loaded geometry " << geometryID << " with " << vertices.size() << " vertices and " << indices.size() / 3 << " triangles" << std::endl;
        }
    }

    std::vector<OptixInstance> instances(nInstances);
    uint i = 0;
    for (const auto& node : asset->nodes) {
        if (auto m = node.meshIndex; m.has_value()) {
            auto mesh = asset->meshes[m.value()];
            auto mat = fastgltf::math::fmat4x4(1.0f);
            auto* trs = std::get_if<fastgltf::TRS>(&node.transform);
            auto* matrix = std::get_if<fastgltf::math::fmat4x4>(&node.transform);
            if (trs) {
                mat = fastgltf::math::scale(fastgltf::math::rotate(fastgltf::math::translate(mat, trs->translation), trs->rotation), trs->scale);
            } else if (matrix) {
                mat = *matrix;
            }
            for (uint j = 0; j < mesh.primitives.size(); j++) {
                const auto& primitive = mesh.primitives[j];
                auto& geometry = geometryTable[m.value()][j];
                instances[i] = OptixInstance {
                    .transform = {
                        mat.row(0)[0], mat.row(0)[1], mat.row(0)[2], mat.row(0)[3],
                        mat.row(1)[0], mat.row(1)[1], mat.row(1)[2], mat.row(1)[3],
                        mat.row(2)[0], mat.row(2)[1], mat.row(2)[2], mat.row(2)[3],
                    },
                    .instanceId = i,
                    .sbtOffset = geometry.sbtOffset,
                    .visibilityMask = 255,
                    .flags = OPTIX_INSTANCE_FLAG_NONE,
                    .traversableHandle = geometry.handle,
                };
                i++;
            }
        }
    }

    const auto [handle, newIASBuffer] = buildIAS(ctx, instances);

    params->handle = handle;
    check(cudaFree(reinterpret_cast<void*>(iasBuffer)));
    iasBuffer = newIASBuffer;

    Material* materialBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&materialBuffer), materials.size() * sizeof(Material)));
    check(cudaMemcpy(reinterpret_cast<void*>(materialBuffer), materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    check(cudaFree(reinterpret_cast<void*>(params->materials))); // Free previous materials buffer
    params->materials = materialBuffer;

    HitRecord* hitRecordBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&hitRecordBuffer), hitRecords.size() * sizeof(HitRecord)));
    check(cudaMemcpy(reinterpret_cast<void*>(hitRecordBuffer), hitRecords.data(), hitRecords.size() * sizeof(HitRecord), cudaMemcpyHostToDevice));

    check(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase))); // Free previous hitgroup record buffer
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitRecordBuffer);
    sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    sbt.hitgroupRecordCount = hitRecords.size();
}