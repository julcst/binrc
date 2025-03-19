#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

#include <vector>
#include <filesystem>
#include <optional>

#include <glm/glm.hpp>
using namespace glm;

#include "optixparams.cuh"

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
    OptixTraversableHandle handle;
    CUdeviceptr gasBuffer; // NOTE: This is owned memory and must be freed
    CUdeviceptr indexBuffer; // NOTE: This is owned memory and must be freed
    CUdeviceptr vertexData; // NOTE: This is owned memory and must be freed
    uint sbtOffset;
    std::optional<Emitter> emitter;
    AABB aabb;
    Geometry(OptixTraversableHandle handle, CUdeviceptr gasBuffer, CUdeviceptr indexBuffer, CUdeviceptr vertexData, uint sbtOffset, std::optional<Emitter> emitter, AABB aabb)
        : handle(handle), gasBuffer(gasBuffer), indexBuffer(indexBuffer), vertexData(vertexData), sbtOffset(sbtOffset), emitter(emitter), aabb(aabb) {}
    ~Geometry();
};

struct Scene {
    std::vector<OptixInstance> instances;
    std::vector<std::pair<std::string, glm::mat4>> cameras;

    // Buffers managed by the scene
    CUdeviceptr iasBuffer = 0;
    std::vector<std::vector<Geometry>> geometryTable;
    std::vector<cudaArray_t> images;
    std::vector<cudaTextureObject_t> textures;

    Scene() = default;
    ~Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(Scene&&) = delete;

    void free();
    bool isEmpty() const { return geometryTable.empty(); }
    void loadGLTF(OptixDeviceContext ctx, Params* params, OptixProgramGroup& o, OptixShaderBindingTable& sbt, const std::filesystem::path& path);
    AABB getAABB() const;
};