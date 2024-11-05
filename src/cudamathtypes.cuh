#pragma once

#include <cuda_runtime.h>

struct float2x2 {
    float2 m[2];
    __host__ __device__ constexpr float2& operator[](uint i) { return m[i]; }
    __host__ __device__ constexpr const float2& operator[](uint i) const { return m[i]; }
};

struct float3x3 {
    float3 m[3];
    __host__ __device__ constexpr float3& operator[](uint i) { return m[i]; }
    __host__ __device__ constexpr const float3& operator[](uint i) const { return m[i]; }
};

struct float4x4 {
    float4 m[4];
    __host__ __device__ constexpr float4& operator[](uint i) { return m[i]; }
    __host__ __device__ constexpr const float4& operator[](uint i) const { return m[i]; }
};

__host__ __device__ constexpr float4x4 make_float4x4(const float4& a, const float4& b, const float4& c, const float4& d) {
    return float4x4{a, b, c, d};
}