#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

#include <array>
#include <vector>
#include <filesystem>

#include <glm/glm.hpp>
using namespace glm;

#include "optixparams.cuh"
#include "scene.hpp"

class OptixRenderer {
public:
    OptixRenderer();
    ~OptixRenderer();
    OptixRenderer(const OptixRenderer&) = delete;
    OptixRenderer& operator=(const OptixRenderer&) = delete;
    OptixRenderer(OptixRenderer&&) = delete;
    OptixRenderer& operator=(OptixRenderer&&) = delete;

    void render(vec4* image, uvec2 dim);
    void setCamera(const mat4& clipToWorld);
    void loadGLTF(const std::filesystem::path& path);
    void resize(uvec2 dim);
    
    Params* params;

private:
    OptixDeviceContext context;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    std::array<OptixProgramGroup, 3> programGroups;
    RaygenRecord* raygenRecord;
    MissRecord* missRecord;
    Scene scene;

    void reset();
    void generateSobol(uint offset, uint n);
    void ensureSobol(uint sample);
};