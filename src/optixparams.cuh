#pragma once

#include <cuda_runtime.h>
#include <array>

#include <tiny-cuda-nn/common.h>
#include <json/json.hpp>

#include "cudamathtypes.cuh"

constexpr int PAYLOAD_SIZE = 17;
constexpr float MAX_T = 1e32f;
constexpr uint MAX_BOUNCES = 31;
constexpr uint RANDS_PER_PIXEL = 2;
constexpr uint RANDS_PER_BOUNCE = 4;
constexpr uint RAND_SEQUENCE_DIMS = RANDS_PER_PIXEL + RANDS_PER_BOUNCE * MAX_BOUNCES;
constexpr uint RAND_SEQUENCE_CACHE_SIZE = 4096;

constexpr uint TRANSMISSION_FLAG = 1 << 0;
constexpr uint NEE_FLAG = 1 << 1;
constexpr uint NRC_INFERENCE_FLAG = 1 << 2;

struct NRCInput {
    float3 position = make_float3(NAN, 0.0f, 0.0f);
};

struct NRCOutput {
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
};

constexpr uint NRC_INPUT_SIZE = sizeof(NRCInput) / sizeof(float);
constexpr uint NRC_OUTPUT_SIZE = sizeof(NRCOutput) / sizeof(float);
constexpr uint NRC_BATCH_SIZE = tcnn::BATCH_SIZE_GRANULARITY * 22;

const nlohmann::json NRC_CONFIG = {
	{"loss", {
		{"otype", "RelativeL2Luminance"}
	}},
	{"optimizer", {
		{"otype", "Average"},
        {"nested", {
            {"otype", "Adam"},
            {"learning_rate", 1e-2f},
        }},
	}},
	{"encoding", {
		{"otype", "Composite"},
        {"nested", {
            {
                {"n_dims_to_encode", 3},
                {"otype", "Grid"},
                {"type", "Hash"},
            },
        }},
    }},
	{"network", {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", 32},
		{"n_hidden_layers", 5},
	}},
};

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

// NOTE: Because this includes pointers this should be zero-initialized using cudaMemset
struct Params {
    float4* image; // A copied pointer to the image buffer
    uint2 dim;
    OptixTraversableHandle handle;
    float4x4 clipToWorld;
    uint sequenceOffset; // Offset into the Sobol sequence
    uint sequenceStride; // Stride between different dimensions
    uint sample; // Current sample
    float weight; // Weight of the current sample (= 1 / (sample + 1))
    float russianRouletteWeight; // Weight for Russian Roulette
    float sceneEpsilon; // Scene epsilon
    uint flags;

////////////////// OWNED MEMORY //////////////////
// NOTE: This is owned memory and must be freed //
//////////////////////////////////////////////////
    float* randSequence; // Quasi-random Sobol sequence
    float4* rotationTable; // Cranley-Patterson-Rotation per pixel
    Material* materials; // materials
    EmissiveTriangle* lightTable; // lightTable
    uint lightTableSize; // lightTableSize
//////////////////////////////////////////////////

    float* trainingInput;
    float* trainingTarget;
    float* inferenceInput;
    float* inferenceOutput;
};
extern "C" __constant__ Params params;

__host__ inline void initParams(Params* params) {
    params->clipToWorld = make_float4x4(make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                                        make_float4(0.0f, 1.0f, 0.0f, 0.0f),
                                        make_float4(0.0f, 0.0f, 1.0f, 0.0f),
                                        make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    params->weight = 1.0f;
    params->russianRouletteWeight = 10.0f;
    params->sceneEpsilon = 1e-4f;
    params->flags = TRANSMISSION_FLAG | NEE_FLAG;
}

__device__ inline float getRand(uint dim) {
    const uint i = params.sample - params.sequenceOffset + params.sequenceStride * dim;
    return params.randSequence[i];
}

__device__ inline float getRand(uint depth, uint i) {
    return getRand(RANDS_PER_PIXEL + depth * RANDS_PER_BOUNCE + i);
}

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