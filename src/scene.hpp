#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

#include <vector>
#include <filesystem>

#include <glm/glm.hpp>
using namespace glm;

struct Scene {
    OptixInstance* instances = nullptr;
    uint nInstances = 0;
    CUdeviceptr iasBuffer = 0;
    CUdeviceptr gasBuffer = 0;

    Scene() = default;
    ~Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(Scene&&) = delete;

    OptixTraversableHandle loadGLTF(OptixDeviceContext ctx, const std::filesystem::path& path);
    OptixTraversableHandle buildGAS(OptixDeviceContext ctx, const std::vector<OptixBuildInput>& buildInputs);
    OptixTraversableHandle buildIAS(OptixDeviceContext ctx);
    OptixTraversableHandle updateIAS(OptixDeviceContext ctx);
};