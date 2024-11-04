#pragma once

#include <cuda_runtime.h>

#include <array>

#include <glm/glm.hpp>
using namespace glm;

// NOTE: Because this includes pointers this should be zero-initialized using cudaMemset
struct Params {
    vec4* image; // A copied pointer to the image buffer
    float* randSequence; // Quasi-random Sobol sequence             // NOTE: This is owned memory and must be freed
    vec4* rotationTable; // Cranley-Patterson-Rotation per pixel    // NOTE: This is owned memory and must be freed
    uvec2 dim;
    OptixTraversableHandle handle;
    mat4 clipToWorld;
    uint sequenceOffset; // Offset into the Sobol sequence
    uint sequenceStride; // Stride between different dimensions
    uint sample; // Current sample
    float weight; // Weight of the current sample (= 1 / (sample + 1))
};
extern "C" __constant__ Params params;

struct VertexData {
    vec3 normal;
    vec2 texCoord;
};

struct RaygenData {};
struct MissData {};
struct HitData {
    uint3* indexBuffer;      // Pointer to triangle indices         // NOTE: This is owned memory and must be freed
    VertexData* vertexData;  // Pointer to vertex data              // NOTE: This is owned memory and must be freed
    uint materialIndex;      // Material index or identifier
};

template <typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) std::array<char, OPTIX_SBT_RECORD_HEADER_SIZE> header;
    T data;
};

using RaygenRecord = Record<RaygenData>;
using MissRecord = Record<MissData>;
using HitRecord = Record<HitData>;