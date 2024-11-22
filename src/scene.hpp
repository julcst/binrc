#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

#include <vector>
#include <filesystem>

#include <glm/glm.hpp>
using namespace glm;

#include "optixparams.cuh"

struct Geometry {
    OptixTraversableHandle handle;
    CUdeviceptr gasBuffer; // NOTE: This is owned memory and must be freed
    CUdeviceptr indexBuffer; // NOTE: This is owned memory and must be freed
    CUdeviceptr vertexData; // NOTE: This is owned memory and must be freed
    uint sbtOffset;
    ~Geometry();
};

struct Scene {
    std::vector<OptixInstance> instances;

    // Buffers managed by the scene
    CUdeviceptr iasBuffer = 0;
    std::vector<std::vector<Geometry>> geometryTable;

    Scene() = default;
    ~Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(Scene&&) = delete;

    void loadGLTF(OptixDeviceContext ctx, Params* params, OptixProgramGroup& o, OptixShaderBindingTable& sbt, const std::filesystem::path& path);
};