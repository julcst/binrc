#pragma once

#include <optix_device.h>
#include <cuda_runtime.h>

struct Payload {
    float3 baseColor; // Linear RGB base color
    float3 normal; // World space normal, guaranteed to be normalized
    float3 tangent; // World space tenagent, not normalized
    float3 emission; // Linear RGB emission color
    float roughness;
    float metallic;
    float transmission;
    float area;
    float t; // Distance of intersection on ray, set to INFINITY if no intersection
};

__device__ inline void setBaseColor(const float3& value) {
    optixSetPayload_0(__float_as_uint(value.x));
    optixSetPayload_1(__float_as_uint(value.y));
    optixSetPayload_2(__float_as_uint(value.z));
}

__device__ inline void setNormal(const float3& value) {
    optixSetPayload_3(__float_as_uint(value.x));
    optixSetPayload_4(__float_as_uint(value.y));
    optixSetPayload_5(__float_as_uint(value.z));
}

__device__ inline void setTangent(const float3& value) {
    optixSetPayload_6(__float_as_uint(value.x));
    optixSetPayload_7(__float_as_uint(value.y));
    optixSetPayload_8(__float_as_uint(value.z));
}

__device__ inline void setEmission(const float3& value) {
    optixSetPayload_9(__float_as_uint(value.x));
    optixSetPayload_10(__float_as_uint(value.y));
    optixSetPayload_11(__float_as_uint(value.z));
}

__device__ inline void setRoughness(const float value) {
    optixSetPayload_12(__float_as_uint(value));
}

__device__ inline void setMetallic(const float value) {
    optixSetPayload_13(__float_as_uint(value));
}

__device__ inline void setTransmission(const float value) {
    optixSetPayload_14(__float_as_uint(value));
}

__device__ inline void setArea(const float value) {
    optixSetPayload_15(__float_as_uint(value));
}

__device__ inline void setT(const float value) {
    optixSetPayload_16(__float_as_uint(value));
}

__device__ constexpr inline Payload getPayload(const std::array<uint, 17>& values) {
    return {
        make_float3(__uint_as_float(values[0]), __uint_as_float(values[1]), __uint_as_float(values[2])),
        make_float3(__uint_as_float(values[3]), __uint_as_float(values[4]), __uint_as_float(values[5])),
        make_float3(__uint_as_float(values[6]), __uint_as_float(values[7]), __uint_as_float(values[8])),
        make_float3(__uint_as_float(values[9]), __uint_as_float(values[10]), __uint_as_float(values[11])),
        __uint_as_float(values[12]),
        __uint_as_float(values[13]),
        __uint_as_float(values[14]),
        __uint_as_float(values[15]),
        __uint_as_float(values[16]),
    };
}