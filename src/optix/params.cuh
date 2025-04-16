#pragma once

#include <cuda_runtime.h>
#include <array>
#include <string_view>

#include <tiny-cuda-nn/common.h>
#include <json/json.hpp>

#include "cudamath.cuh"

constexpr float MAX_T = 1e32f;
constexpr uint MAX_BOUNCES = 12;
constexpr uint RANDS_PER_PIXEL = 5;
constexpr uint RANDS_PER_BOUNCE = 9;
constexpr uint ROTATIONS_PER_PIXEL = 2;
constexpr uint RAND_SEQUENCE_DIMS = RANDS_PER_PIXEL + RANDS_PER_BOUNCE * MAX_BOUNCES;
constexpr uint RAND_SEQUENCE_CACHE_SIZE = 4096;

constexpr uint TRAIN_DEPTH = 6;
constexpr uint TRANSMISSION_FLAG = 1 << 0;
constexpr uint NEE_FLAG = 1 << 1;
constexpr uint TRAINING_NEE_FLAG = 1 << 2;
constexpr uint INFERENCE_NEE_FLAG = 1 << 3;
constexpr uint FORWARD_RR_FLAG = 1 << 4;
constexpr uint BACKWARD_RR_FLAG = 1 << 5;

// TODO: Use half instead of float
struct NRCInput {
    float3 position = {NAN, 0.0f, 0.0f};
    float2 wo = {0.0f, 0.0f};
    float2 wn = {0.0f, 0.0f};
    float roughness = 0.0f;
    float3 diffuse = {0.0f, 0.0f, 0.0f};
    float3 specular = {0.0f, 0.0f, 0.0f};
};

struct NRCOutput {
    float3 radiance = {0.0f, 0.0f, 0.0f};
};

constexpr int PAYLOAD_SIZE = 4 * 3 + 5;
constexpr uint NRC_INPUT_SIZE = 3 * 3 + 2 * 2 + 1;
constexpr uint NRC_OUTPUT_SIZE = 3;
constexpr uint NRC_SUBBATCH_SIZE = tcnn::BATCH_SIZE_GRANULARITY * 64 * 8 * 4;
constexpr uint STEPS_PER_BATCH = 1;
constexpr uint NRC_BATCH_SIZE = NRC_SUBBATCH_SIZE * STEPS_PER_BATCH;

struct VertexData {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 texCoord;
};

struct Material {
    float3 baseColor;
    float3 emission;
    float roughness;
    float metallic;
    float transmission;
    cudaTextureObject_t baseMap;
    cudaTextureObject_t normalMap;
    cudaTextureObject_t mrMap;
};

struct EmissiveTriangle {
    float3 v0;
    float cdf;
    float3 v1;
    float weight;
    float3 v2;
    uint materialID;
    float3 n0;
    float area;
    float3 n1;
    float3 n2;
};

constexpr std::array<std::string_view, 6> INFERENCE_MODES = {
    "No Inference",
    "Raw Cache",
    "1st Vertex",
    "1st Vertex + NEE",
    "1st Diffuse",
    "Variance Heuristic",
};

enum class InferenceMode : u_int8_t {
    NO_INFERENCE,
    RAW_CACHE,
    FIRST_VERTEX,
    FIRST_VERTEX_WITH_NEE,
    FIRST_DIFFUSE,
    VARIANCE_HEURISTIC,
};

// NOTE: Because this includes pointers this should be zero-initialized using cudaMemset
struct Params {
    float4* image; // A copied pointer to the image buffer
    uint2 dim;
    OptixTraversableHandle handle;
    float4x4 clipToWorld;
    uint sequenceOffset; // Offset into the Sobol sequence
    uint sequenceStride; // Stride between different dimensions
    uint sample; // Current sample
    float weight = 1.0f; // Weight of the current sample (= 1 / (sample + 1))
    float russianRouletteWeight = 10.0f; // Weight for Russian Roulette
    float sceneEpsilon = 1e-4f; // Scene epsilon
    uint flags = NEE_FLAG | TRANSMISSION_FLAG; // Flags
    InferenceMode inferenceMode;
    float3 sceneMin;
    float sceneScale;

////////////////// OWNED MEMORY //////////////////
// NOTE: This is owned memory and must be freed //
//////////////////////////////////////////////////
    float* randSequence; // Quasi-random Sobol sequence
    float2* rotationTable; // Cranley-Patterson-Rotation per pixel
    Material* materials; // materials
    EmissiveTriangle* lightTable; // lightTable
    uint lightTableSize; // lightTableSize
//////////////////////////////////////////////////

    uint* trainingIndexPtr;
    float* trainingInput;
    float* trainingTarget;
    float* inferenceInput;
    float* inferenceOutput;
    float3* inferenceThroughput;
};
extern "C" __constant__ const Params params;

struct RaygenData {};
struct MissData {};
struct HitData {
    uint3* indexBuffer;      // Pointer to triangle indices         // NOTE: This is owned memory and must be freed
    VertexData* vertexData;  // Pointer to vertex data              // NOTE: This is owned memory and must be freed
    uint materialID;         // Index into the materials array
};

template <typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) std::array<char, OPTIX_SBT_RECORD_HEADER_SIZE> header;
    T data;
};

using RaygenRecord = Record<RaygenData>;
using MissRecord = Record<MissData>;
using HitRecord = Record<HitData>;

__device__ __forceinline__ float getRand(uint dim) {
    const uint i = params.sample - params.sequenceOffset + params.sequenceStride * dim;
    return params.randSequence[i];
}

__device__ __forceinline__ float getRand(uint depth, uint dim) {
    return getRand(RANDS_PER_PIXEL + depth * RANDS_PER_BOUNCE + dim);
}

__device__ __forceinline__ float getRand(uint depth, uint dim, float rotation) {
    return fract(getRand(depth, dim) + rotation);
}

__device__ __forceinline__ float2 getRand2(uint dim) {
    const uint i = params.sample - params.sequenceOffset + params.sequenceStride * dim;
    return {
        params.randSequence[i],
        params.randSequence[i + params.sequenceStride]
    };
}

__device__ __forceinline__ float2 getRand2(uint depth, uint dim) {
    return getRand2(RANDS_PER_PIXEL + depth * RANDS_PER_BOUNCE + dim);
}

__device__ __forceinline__ float2 getRand2(uint depth, uint offset, float2 rotation) {
    return fract(make_float2(getRand(depth, offset), getRand(depth, offset + 1)) + rotation);
}

#define PR1(dim) fract(getRand(dim) + rotation.x)
#define PR2(dim) fract(getRand2(dim) + rotation)
#define BR1(dim) getRand(depth, dim, rotation.x)
#define BR2(dim) getRand2(depth, dim, rotation)

#define RND_JITTER PR2(0)
#define RND_TRAIN1 PR1(2)
#define RND_TRAIN2 PR1(3)

#define RND_ROULETTE BR1(0)
#define RND_LSRC BR1(1)
#define RND_LSAMP BR2(2)
#define RND_BSDF BR1(4)
#define RND_MICROFACET BR2(5)
#define RND_DIFFUSE BR2(7)