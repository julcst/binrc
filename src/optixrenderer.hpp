#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

#include <vector>

#include "optixparams.hpp"

struct RaygenData {};
struct MissData {};
struct HitData {};

template <typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RaygenRecord = Record<RaygenData>;
using MissRecord = Record<MissData>;
using HitRecord = Record<HitData>;

class OptixRenderer {
public:
    OptixRenderer();
    ~OptixRenderer();
    void render(float4* image, uint2 dim);
    void buildGAS(const std::vector<float3>& vertices, const std::vector<uint3>& indices);
private:
    OptixDeviceContext context;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    RaygenRecord* raygenRecord;
    MissRecord* missRecord;
    HitRecord* hitRecord;
    Params* params;
};