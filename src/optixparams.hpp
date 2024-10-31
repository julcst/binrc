#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>
using namespace glm;

struct Params {
    vec4* image;
    uvec2 dim;
    OptixTraversableHandle handle;
    mat4 clipToWorld;
};
extern "C" __constant__ Params params;