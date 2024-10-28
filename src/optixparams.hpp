#pragma once

#include <cuda_runtime.h>

struct Params {
    float4* image;
    uint2 dim;
    OptixTraversableHandle handle;
};
extern "C" __constant__ Params params;