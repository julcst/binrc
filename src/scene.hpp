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

struct Geometry {
    OptixTraversableHandle handle;
    CUdeviceptr gasBuffer; // NOTE: This is owned memory and must be freed
    CUdeviceptr indexBuffer; // NOTE: This is owned memory and must be freed
    CUdeviceptr vertexData; // NOTE: This is owned memory and must be freed
    uint sbtOffset;
    std::optional<Emitter> emitter;
    Geometry(OptixTraversableHandle handle, CUdeviceptr gasBuffer, CUdeviceptr indexBuffer, CUdeviceptr vertexData, uint sbtOffset, std::optional<Emitter> emitter);
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
};