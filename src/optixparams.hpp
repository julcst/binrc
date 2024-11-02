#pragma once

#include <cuda_runtime.h>

#include <array>

#include <glm/glm.hpp>
using namespace glm;

struct Params {
    vec4* image;
    uvec2 dim;
    OptixTraversableHandle handle;
    mat4 clipToWorld;
};
extern "C" __constant__ Params params;

struct VertexData {
    vec3 normal;
    vec2 texCoord;
};

struct GASData {
    uint3* indexBuffer;        // Pointer to triangle indices
    VertexData* vertexBuffer;  // Pointer to vertex data
    uint materialIndex;         // Material index or identifier
};

struct RaygenData {};
struct MissData {};

template <typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) std::array<char, OPTIX_SBT_RECORD_HEADER_SIZE> header;
    T data;
};

using RaygenRecord = Record<RaygenData>;
using MissRecord = Record<MissData>;
using HitRecord = Record<GASData>;