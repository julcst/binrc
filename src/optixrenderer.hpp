#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

#include <array>
#include <vector>
#include <filesystem>

#include <glm/glm.hpp>
using namespace glm;

#include "optixparams.hpp"
#include "scene.hpp"

struct RaygenData {};
struct MissData {};
struct HitData {};

template <typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) std::array<char, OPTIX_SBT_RECORD_HEADER_SIZE> header;
    T data;
};

using RaygenRecord = Record<RaygenData>;
using MissRecord = Record<MissData>;
using HitRecord = Record<HitData>;

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
private:
    OptixDeviceContext context;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    RaygenRecord* raygenRecord;
    MissRecord* missRecord;
    HitRecord* hitRecord;
    Params* params;
    Scene scene;
};