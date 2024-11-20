#pragma once

#include <cuda_runtime.h>

using uint = unsigned int;

struct float2x2 {
    float2 m[2];
    __host__ __device__ constexpr float2& operator[](uint i) { return m[i]; }
    __host__ __device__ constexpr const float2& operator[](uint i) const { return m[i]; }
};

__host__ __device__ constexpr float2x2 make_float2x2(const float2& a, const float2& b) {
    return float2x2{a, b};
}

struct float3x3 {
    float3 m[3];
    __host__ __device__ constexpr float3& operator[](uint i) { return m[i]; }
    __host__ __device__ constexpr const float3& operator[](uint i) const { return m[i]; }
};

__host__ __device__ constexpr float3x3 make_float3x3(const float3& a, const float3& b, const float3& c) {
    return float3x3{a, b, c};
}

struct float4x4 {
    float4 m[4];
    __host__ __device__ constexpr float4& operator[](uint i) { return m[i]; }
    __host__ __device__ constexpr const float4& operator[](uint i) const { return m[i]; }
};

__host__ __device__ constexpr float4x4 make_float4x4(const float4& a, const float4& b, const float4& c, const float4& d) {
    return float4x4{a, b, c, d};
}