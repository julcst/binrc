#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

#include <vector>
#include <filesystem>
#include <optional>

#include <glm/glm.hpp>
using namespace glm;

#include <tiny-cuda-nn/gpu_memory.h>

#include "optix/params.cuh"
#include "cudautil.hpp"

struct Emitter {
    float emission;
    std::vector<vec4> vertices;
    std::vector<uint> indices;
    std::vector<VertexData> vertexData;
};

struct AABB {
    vec3 min = { INFINITY, INFINITY, INFINITY };
    vec3 max = { -INFINITY, -INFINITY, -INFINITY };
    inline void extend(const vec3& p) {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }
    inline void extend(const AABB& aabb) {
        min = glm::min(min, aabb.min);
        max = glm::max(max, aabb.max);
    }
    inline void transform(const mat4& m) {
        const auto p0 = m * vec4(min.x, min.y, min.z, 1.0f);
        const auto p1 = m * vec4(min.x, min.y, max.z, 1.0f);
        const auto p2 = m * vec4(min.x, max.y, min.z, 1.0f);
        const auto p3 = m * vec4(min.x, max.y, max.z, 1.0f);
        const auto p4 = m * vec4(max.x, min.y, min.z, 1.0f);
        const auto p5 = m * vec4(max.x, min.y, max.z, 1.0f);
        const auto p6 = m * vec4(max.x, max.y, min.z, 1.0f);
        const auto p7 = m * vec4(max.x, max.y, max.z, 1.0f);
        extend(vec3(p0) / p0.w);
        extend(vec3(p1) / p1.w);
        extend(vec3(p2) / p2.w);
        extend(vec3(p3) / p3.w);
        extend(vec3(p4) / p4.w);
        extend(vec3(p5) / p5.w);
        extend(vec3(p6) / p6.w);
        extend(vec3(p7) / p7.w);
    }
};

struct Geometry {
    OptixTraversableHandle handle = 0;
    CUdeviceptr gasBuffer = 0;
    CUdeviceptr indexBuffer = 0;
    CUdeviceptr vertexBuffer = 0;
    CUdeviceptr cdfBuffer = 0;
    float totalArea = 0.0f;
    uint sbtOffset;
    std::optional<Emitter> emitter;
    AABB aabb;

    // RAII
    Geometry() = default;
    Geometry(const Geometry&) = delete;
    Geometry& operator=(const Geometry&) = delete;
    Geometry(Geometry&& other) :
        handle(other.handle), gasBuffer(other.gasBuffer), indexBuffer(other.indexBuffer),
        vertexBuffer(other.vertexBuffer), cdfBuffer(other.cdfBuffer), totalArea(other.totalArea),
        sbtOffset(other.sbtOffset), emitter(std::move(other.emitter)), aabb(other.aabb) {
        other.handle = 0;
        other.gasBuffer = 0;
        other.indexBuffer = 0;
        other.vertexBuffer = 0;
        other.cdfBuffer = 0;
    }
    Geometry& operator=(Geometry&& other) = delete;
    ~Geometry() {
        check(cudaFree(reinterpret_cast<void*>(gasBuffer)));
        check(cudaFree(reinterpret_cast<void*>(indexBuffer)));
        check(cudaFree(reinterpret_cast<void*>(vertexBuffer)));
        check(cudaFree(reinterpret_cast<void*>(cdfBuffer)));
    }
};

struct SceneData {
    std::vector<HitRecord> hitRecords;
    std::vector<Material> materials;
    std::vector<EmissiveTriangle> lightTable;
    std::vector<Instance> instances;
    OptixTraversableHandle handle;
    float totalArea = 0.0f;
};

struct cudaArray_RAII {
    cudaArray_t array;
    cudaArray_RAII(cudaArray_t array = nullptr) : array(array) {}
    ~cudaArray_RAII() { check(cudaFreeArray(array)); }
    cudaArray_RAII(const cudaArray_RAII&) = delete;
    cudaArray_RAII& operator=(const cudaArray_RAII&) = delete;
    cudaArray_RAII(cudaArray_RAII&& other) : array(other.array) { other.array = nullptr; }
    cudaArray_RAII& operator=(cudaArray_RAII&& other) {
        if (this != &other) {
            check(cudaFreeArray(array));
            array = other.array;
            other.array = nullptr;
        }
        return *this;
    }
};

struct cudaTextureObject_RAII {
    cudaTextureObject_t texture;
    cudaTextureObject_RAII(cudaTextureObject_t texture = 0) : texture(texture) {}
    ~cudaTextureObject_RAII() { check(cudaDestroyTextureObject(texture)); }
    cudaTextureObject_RAII(const cudaTextureObject_RAII&) = delete;
    cudaTextureObject_RAII& operator=(const cudaTextureObject_RAII&) = delete;
    cudaTextureObject_RAII(cudaTextureObject_RAII&& other) : texture(other.texture) { other.texture = 0; }
    cudaTextureObject_RAII& operator=(cudaTextureObject_RAII&& other) {
        if (this != &other) {
            check(cudaDestroyTextureObject(texture));
            texture = other.texture;
            other.texture = 0;
        }
        return *this;
    }
};

struct Scene {
    std::vector<OptixInstance> instances;
    std::vector<std::pair<std::string, glm::mat4>> cameras;

    // Buffers managed by the scene
    CUdeviceptr iasBuffer = 0; // NOTE: This is owned memory and must be freed
    std::vector<std::vector<Geometry>> geometryTable;
    std::vector<cudaArray_RAII> images; // NOTE: This is owned memory and must be freed
    std::vector<cudaTextureObject_RAII> textures; // NOTE: This is owned memory and must be freed

    Scene() = default;
    ~Scene() {
        check(cudaFree(reinterpret_cast<void*>(iasBuffer)));
    }

    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(Scene&&) = delete;

    bool isEmpty() const { return geometryTable.empty(); }
    SceneData loadGLTF(OptixDeviceContext ctx, const std::filesystem::path& path);
    AABB getAABB() const;
};