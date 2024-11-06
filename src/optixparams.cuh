#pragma once

#include <cuda_runtime.h>
#include <array>

#include "cudamathtypes.cuh"

constexpr int PAYLOAD_SIZE = 12;
constexpr float MAX_T = 1e32f;
constexpr uint MAX_BOUNCES = 16;
constexpr uint RANDS_PER_PIXEL = 2;
constexpr uint RANDS_PER_BOUNCE = 4;
constexpr uint RAND_SEQUENCE_DIMS = RANDS_PER_PIXEL + RANDS_PER_BOUNCE * MAX_BOUNCES;
constexpr uint RAND_SEQUENCE_CACHE_SIZE = 1024;

// NOTE: Because this includes pointers this should be zero-initialized using cudaMemset
struct Params {
    float4* image; // A copied pointer to the image buffer
    float* randSequence; // Quasi-random Sobol sequence             // NOTE: This is owned memory and must be freed
    float4* rotationTable; // Cranley-Patterson-Rotation per pixel    // NOTE: This is owned memory and must be freed
    uint2 dim;
    OptixTraversableHandle handle;
    float4x4 clipToWorld;
    uint sequenceOffset; // Offset into the Sobol sequence
    uint sequenceStride; // Stride between different dimensions
    uint sample; // Current sample
    float weight; // Weight of the current sample (= 1 / (sample + 1))
    float russianRouletteWeight; // Weight for Russian Roulette
};
extern "C" __constant__ Params params;

__device__ inline float getRand(uint dim) {
    const uint i = params.sample - params.sequenceOffset + params.sequenceStride * dim;
    return params.randSequence[i];
}

__device__ inline float getRand(uint depth, uint i) {
    return getRand(RANDS_PER_PIXEL + depth * RANDS_PER_BOUNCE + i);
}

struct VertexData {
    float3 normal;
    float4 tangent;
    float2 texCoord;
};

struct Material {
    float3 color;
    float roughness;
    float metallic;
};

struct RaygenData {};
struct MissData {};
struct HitData {
    uint3* indexBuffer;      // Pointer to triangle indices         // NOTE: This is owned memory and must be freed
    VertexData* vertexData;  // Pointer to vertex data              // NOTE: This is owned memory and must be freed
    Material* material;      // Pointer to material data
};

template <typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) std::array<char, OPTIX_SBT_RECORD_HEADER_SIZE> header;
    T data;
};

using RaygenRecord = Record<RaygenData>;
using MissRecord = Record<MissData>;
using HitRecord = Record<HitData>;