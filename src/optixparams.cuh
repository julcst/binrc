#pragma once

#include <cuda_runtime.h>
#include <array>
#include <string_view>

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

// TODO: Use half instead of float
struct NRCInput {
    float3 position = make_float3(NAN, 0.0f, 0.0f);
    float2 wo = make_float2(0.0f, 0.0f);
    float2 wn = make_float2(0.0f, 0.0f);
    float roughness = 0.0f;
    float3 diffuse = make_float3(0.0f, 0.0f, 0.0f);
    float3 specular = make_float3(0.0f, 0.0f, 0.0f);
};

struct NRCOutput {
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
};

constexpr uint NRC_INPUT_SIZE = sizeof(NRCInput) / sizeof(float);
constexpr uint NRC_OUTPUT_SIZE = sizeof(NRCOutput) / sizeof(float);
constexpr uint NRC_BATCH_SIZE = tcnn::BATCH_SIZE_GRANULARITY * 64 * 4 * 8;

const nlohmann::json NRC_CONFIG = {
	{"loss", {
		{"otype", "RelativeL2Luminance"},
	}},
	{"optimizer", {
		{"otype", "EMA"},
        {"decay", 0.95f},
        {"nested", {
            {"otype", "Adam"},
            {"learning_rate", 1e-3f},
            {"beta1", 0.9f},
            {"beta2", 0.999f},
            {"epsilon", 1e-8f},
            {"l2_reg", 0.0f},
            {"relative_decay", 0.0f},
            {"absolute_decay", 0.0f},
            {"adabound", false},
        }},
	}},
	{"encoding", {
		{"otype", "Composite"},
        {"nested", {
            {
                {"n_dims_to_encode", 3},
                {"otype", "Grid"},
                {"type", "Hash"},
                {"n_levels", 16},
                {"n_feature_per_level", 2},
                {"log2_hashmap_size", 15},
                {"base_resolution", 16},
                {"per_level_scale", 2.0f},
                {"interpolation", "Linear"},
            },
            {
                {"n_dims_to_encode", 5},
                {"otype", "OneBlob"},
                {"n_bins", 4},
            },
            {
                {"otype", "Identity"},
            },
        }},
    }},
	{"network", {
		{"otype", "FullyFusedMLP"},
        //{"otype", "CutlassMLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", 64},
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
    float weight; // Weight of the current sample (= 1 / (sample + 1))
    float russianRouletteWeight; // Weight for Russian Roulette
    float sceneEpsilon; // Scene epsilon
    uint flags;
    InferenceMode inferenceMode;

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
    float3* inferenceThroughput;
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