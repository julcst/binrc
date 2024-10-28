#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

#include "optixparams.hpp"

template <typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RaygenRecord = Record<int>;
using MissRecord = Record<int>;

class OptixRenderer {
public:
    OptixRenderer();
    ~OptixRenderer();
    void render(float4* image, int width, int height);

private:
    OptixDeviceContext context;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    RaygenRecord* raygenRecord;
    MissRecord* missRecord;
    Params* params;
};