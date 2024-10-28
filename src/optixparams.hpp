#pragma once

#include <cuda_runtime.h>

struct Params {
    float4* image;
    uint2 dim;
};
extern "C" __constant__ Params params;