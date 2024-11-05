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
    CUdeviceptr gasBuffer;
    uint sbtOffset;
};

struct Scene {
    uint nInstances = 0;
    OptixInstance* instances = nullptr;
    
    uint nGeometries = 0;
    HitRecord* hitRecords = nullptr;

    uint nMaterials = 0;
    Material* materials = nullptr;

    CUdeviceptr iasBuffer = 0;
    std::vector<std::vector<Geometry>> meshToGeometries;

    std::vector<uint3*> indexBuffers;
    std::vector<VertexData*> vertexDatas;

    Scene() = default;
    ~Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(Scene&&) = delete;

    void loadGLTF(OptixDeviceContext ctx, Params* params, OptixProgramGroup& o, OptixShaderBindingTable& sbt, const std::filesystem::path& path);

  private:
    void free();
    OptixTraversableHandle buildIAS(OptixDeviceContext ctx);
    OptixTraversableHandle updateIAS(OptixDeviceContext ctx);
};