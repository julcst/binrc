#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

// Conversion glm -> CUDA

__host__ __device__ constexpr float2 glmToCuda(const glm::vec2& v) {
    return make_float2(v.x, v.y);
}

__host__ __device__ constexpr float3 glmToCuda(const glm::vec3& v) {
    //return reinterpret_cast<const float3&>(v);
    return make_float3(v.x, v.y, v.z);
}

__host__ __device__ constexpr float4 glmToCuda(const glm::vec4& v) {
    return make_float4(v.x, v.y, v.z, v.w);
}

__host__ __device__ constexpr int2 glmToCuda(const glm::ivec2& v) {
    return make_int2(v.x, v.y);
}

__host__ __device__ constexpr int3 glmToCuda(const glm::ivec3& v) {
    return make_int3(v.x, v.y, v.z);
}

__host__ __device__ constexpr int4 glmToCuda(const glm::ivec4& v) {
    return make_int4(v.x, v.y, v.z, v.w);
}

__host__ __device__ constexpr uint2 glmToCuda(const glm::uvec2& v) {
    return make_uint2(v.x, v.y);
}

__host__ __device__ constexpr uint3 glmToCuda(const glm::uvec3& v) {
    return make_uint3(v.x, v.y, v.z);
}

__host__ __device__ constexpr uint4 glmToCuda(const glm::uvec4& v) {
    return make_uint4(v.x, v.y, v.z, v.w);
}

// Conversion CUDA -> glm

__host__ __device__ constexpr glm::vec2 cudaToGlm(const float2& v) {
    return glm::vec2(v.x, v.y);
}

__host__ __device__ constexpr glm::vec3 cudaToGlm(const float3& v) {
    return glm::vec3(v.x, v.y, v.z);
}

__host__ __device__ constexpr glm::vec4 cudaToGlm(const float4& v) {
    return glm::vec4(v.x, v.y, v.z, v.w);
}

__host__ __device__ constexpr glm::ivec2 cudaToGlm(const int2& v) {
    return glm::ivec2(v.x, v.y);
}

__host__ __device__ constexpr glm::ivec3 cudaToGlm(const int3& v) {
    return glm::ivec3(v.x, v.y, v.z);
}

__host__ __device__ constexpr glm::ivec4 cudaToGlm(const int4& v) {
    return glm::ivec4(v.x, v.y, v.z, v.w);
}

__host__ __device__ constexpr glm::uvec2 cudaToGlm(const uint2& v) {
    return glm::uvec2(v.x, v.y);
}

__host__ __device__ constexpr glm::uvec3 cudaToGlm(const uint3& v) {
    return glm::uvec3(v.x, v.y, v.z);
}

__host__ __device__ constexpr glm::uvec4 cudaToGlm(const uint4& v) {
    return glm::uvec4(v.x, v.y, v.z, v.w);
}

// Operators on float

__host__ __device__ constexpr float fract(float x) {
    return x - truncf(x);
}

// Operators on CUDA float2

__host__ __device__ constexpr float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__host__ __device__ constexpr float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__host__ __device__ constexpr float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

__host__ __device__ constexpr float2 operator/(const float2& a, const float2& b) {
    return make_float2(a.x / b.x, a.y / b.y);
}

__host__ __device__ constexpr float2 operator+(const float2& a, float b) {
    return make_float2(a.x + b, a.y + b);
}

__host__ __device__ constexpr float2 operator-(const float2& a, float b) {
    return make_float2(a.x - b, a.y - b);
}

__host__ __device__ constexpr float2 operator*(const float2& a, float b) {
    return make_float2(a.x * b, a.y * b);
}

__host__ __device__ constexpr float2 operator/(const float2& a, float b) {
    return make_float2(a.x / b, a.y / b);
}

__host__ __device__ constexpr float2 operator+(float a, const float2& b) {
    return make_float2(a + b.x, a + b.y);
}

__host__ __device__ constexpr float2 operator-(float a, const float2& b) {
    return make_float2(a - b.x, a - b.y);
}

__host__ __device__ constexpr float2 operator*(float a, const float2& b) {
    return make_float2(a * b.x, a * b.y);
}

__host__ __device__ constexpr float2 operator/(float a, const float2& b) {
    return make_float2(a / b.x, a / b.y);
}

__host__ __device__ constexpr float dot(const float2& a, const float2& b) {
    return a.x * b.x + a.y * b.y;
}

__host__ __device__ constexpr float dot2(const float2& a) {
    return dot(a, a);
}

__host__ __device__ constexpr float length(const float2& v) {
    return sqrtf(dot2(v));
}

__host__ __device__ constexpr float2 normalize(const float2& v) {
    return v / length(v);
}

__host__ __device__ constexpr float2 fract(const float2& v) {
    return make_float2(fract(v.x), fract(v.y));
}

// Operators on CUDA float3

__host__ __device__ constexpr float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ constexpr float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ constexpr float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ constexpr float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ constexpr float3 operator+(const float3& a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__host__ __device__ constexpr float3 operator-(const float3& a, float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__host__ __device__ constexpr float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ constexpr float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ constexpr float3 operator+(float a, const float3& b) {
    return make_float3(a + b.x, a + b.y, a + b.z);
}

__host__ __device__ constexpr float3 operator-(float a, const float3& b) {
    return make_float3(a - b.x, a - b.y, a - b.z);
}

__host__ __device__ constexpr float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ constexpr float3 operator/(float a, const float3& b) {
    return make_float3(a / b.x, a / b.y, a / b.z);
}

__host__ __device__ constexpr float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ constexpr float dot2(const float3& a) {
    return dot(a, a);
}

__host__ __device__ constexpr float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__ constexpr float length(const float3& v) {
    return norm3df(v.x, v.y, v.z);
}

__host__ __device__ constexpr float3 normalize(const float3& v) {
    return v / length(v);
}

__host__ __device__ constexpr float3 fract(const float3& v) {
    return make_float3(fract(v.x), fract(v.y), fract(v.z));
}

// Operators on CUDA float4

__host__ __device__ constexpr float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ constexpr float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ constexpr float4 operator*(const float4& a, const float4& b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__host__ __device__ constexpr float4 operator/(const float4& a, const float4& b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__host__ __device__ constexpr float4 operator+(const float4& a, float b) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__host__ __device__ constexpr float4 operator-(const float4& a, float b) {
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

__host__ __device__ constexpr float4 operator*(const float4& a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__host__ __device__ constexpr float4 operator/(const float4& a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__host__ __device__ constexpr float4 operator+(float a, const float4& b) {
    return make_float4(a + b.x, a + b.y, a + b.z, a + b.w);
}

__host__ __device__ constexpr float4 operator-(float a, const float4& b) {
    return make_float4(a - b.x, a - b.y, a - b.z, a - b.w);
}

__host__ __device__ constexpr float4 operator*(float a, const float4& b) {
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__host__ __device__ constexpr float4 operator/(float a, const float4& b) {
    return make_float4(a / b.x, a / b.y, a / b.z, a / b.w);
}

__host__ __device__ constexpr float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ constexpr float dot2(const float4& a) {
    return dot(a, a);
}

__host__ __device__ constexpr float length(const float4& v) {
    return norm4df(v.x, v.y, v.z, v.w);
}

__host__ __device__ constexpr float4 normalize(const float4& v) {
    return v / length(v);
}

__host__ __device__ constexpr float4 fract(const float4& v) {
    return make_float4(fract(v.x), fract(v.y), fract(v.z), fract(v.w));
}

// Operators on CUDA int2

__host__ __device__ constexpr int2 operator+(const int2& a, const int2& b) {
    return make_int2(a.x + b.x, a.y + b.y);
}

__host__ __device__ constexpr int2 operator-(const int2& a, const int2& b) {
    return make_int2(a.x - b.x, a.y - b.y);
}

__host__ __device__ constexpr int2 operator*(const int2& a, const int2& b) {
    return make_int2(a.x * b.x, a.y * b.y);
}

__host__ __device__ constexpr int2 operator/(const int2& a, const int2& b) {
    return make_int2(a.x / b.x, a.y / b.y);
}

__host__ __device__ constexpr int2 operator+(const int2& a, int b) {
    return make_int2(a.x + b, a.y + b);
}

__host__ __device__ constexpr int2 operator-(const int2& a, int b) {
    return make_int2(a.x - b, a.y - b);
}

__host__ __device__ constexpr int2 operator*(const int2& a, int b) {
    return make_int2(a.x * b, a.y * b);
}

__host__ __device__ constexpr int2 operator/(const int2& a, int b) {
    return make_int2(a.x / b, a.y / b);
}

__host__ __device__ constexpr int2 operator+(int a, const int2& b) {
    return make_int2(a + b.x, a + b.y);
}

__host__ __device__ constexpr int2 operator-(int a, const int2& b) {
    return make_int2(a - b.x, a - b.y);
}

__host__ __device__ constexpr int2 operator*(int a, const int2& b) {
    return make_int2(a * b.x, a * b.y);
}

__host__ __device__ constexpr int2 operator/(int a, const int2& b) {
    return make_int2(a / b.x, a / b.y);
}

__host__ __device__ constexpr int dot(const int2& a, const int2& b) {
    return a.x * b.x + a.y * b.y;
}

// Operators on CUDA int3

__host__ __device__ constexpr int3 operator+(const int3& a, const int3& b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ constexpr int3 operator-(const int3& a, const int3& b) {
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ constexpr int3 operator*(const int3& a, const int3& b) {
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ constexpr int3 operator/(const int3& a, const int3& b) {
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ constexpr int3 operator+(const int3& a, int b) {
    return make_int3(a.x + b, a.y + b, a.z + b);
}

__host__ __device__ constexpr int3 operator-(const int3& a, int b) {
    return make_int3(a.x - b, a.y - b, a.z - b);
}

__host__ __device__ constexpr int3 operator*(const int3& a, int b) {
    return make_int3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ constexpr int3 operator/(const int3& a, int b) {
    return make_int3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ constexpr int3 operator+(int a, const int3& b) {
    return make_int3(a + b.x, a + b.y, a + b.z);
}

__host__ __device__ constexpr int3 operator-(int a, const int3& b) {
    return make_int3(a - b.x, a - b.y, a - b.z);
}

__host__ __device__ constexpr int3 operator*(int a, const int3& b) {
    return make_int3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ constexpr int3 operator/(int a, const int3& b) {
    return make_int3(a / b.x, a / b.y, a / b.z);
}

__host__ __device__ constexpr int dot(const int3& a, const int3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ constexpr int3 cross(const int3& a, const int3& b) {
    return make_int3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Operators on CUDA int4

__host__ __device__ constexpr int4 operator+(const int4& a, const int4& b) {
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ constexpr int4 operator-(const int4& a, const int4& b) {
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ constexpr int4 operator*(const int4& a, const int4& b) {
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__host__ __device__ constexpr int4 operator/(const int4& a, const int4& b) {
    return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__host__ __device__ constexpr int4 operator+(const int4& a, int b) {
    return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__host__ __device__ constexpr int4 operator-(const int4& a, int b) {
    return make_int4(a.x - b, a.y - b, a.z - b, a.w - b);
}

__host__ __device__ constexpr int4 operator*(const int4& a, int b) {
    return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__host__ __device__ constexpr int4 operator/(const int4& a, int b) {
    return make_int4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__host__ __device__ constexpr int4 operator+(int a, const int4& b) {
    return make_int4(a + b.x, a + b.y, a + b.z, a + b.w);
}

__host__ __device__ constexpr int4 operator-(int a, const int4& b) {
    return make_int4(a - b.x, a - b.y, a - b.z, a - b.w);
}

__host__ __device__ constexpr int4 operator*(int a, const int4& b) {
    return make_int4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__host__ __device__ constexpr int4 operator/(int a, const int4& b) {
    return make_int4(a / b.x, a / b.y, a / b.z, a / b.w);
}

__host__ __device__ constexpr int dot(const int4& a, const int4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Operators on CUDA uint2

__host__ __device__ constexpr uint2 operator+(const uint2& a, const uint2& b) {
    return make_uint2(a.x + b.x, a.y + b.y);
}

__host__ __device__ constexpr uint2 operator-(const uint2& a, const uint2& b) {
    return make_uint2(a.x - b.x, a.y - b.y);
}

__host__ __device__ constexpr uint2 operator*(const uint2& a, const uint2& b) {
    return make_uint2(a.x * b.x, a.y * b.y);
}

__host__ __device__ constexpr uint2 operator/(const uint2& a, const uint2& b) {
    return make_uint2(a.x / b.x, a.y / b.y);
}

__host__ __device__ constexpr uint2 operator+(const uint2& a, unsigned int b) {
    return make_uint2(a.x + b, a.y + b);
}

__host__ __device__ constexpr uint2 operator-(const uint2& a, unsigned int b) {
    return make_uint2(a.x - b, a.y - b);
}

__host__ __device__ constexpr uint2 operator*(const uint2& a, unsigned int b) {
    return make_uint2(a.x * b, a.y * b);
}

__host__ __device__ constexpr uint2 operator/(const uint2& a, unsigned int b) {
    return make_uint2(a.x / b, a.y / b);
}

__host__ __device__ constexpr uint2 operator+(unsigned int a, const uint2& b) {
    return make_uint2(a + b.x, a + b.y);
}

__host__ __device__ constexpr uint2 operator-(unsigned int a, const uint2& b) {
    return make_uint2(a - b.x, a - b.y);
}

__host__ __device__ constexpr uint2 operator*(unsigned int a, const uint2& b) {
    return make_uint2(a * b.x, a * b.y);
}

__host__ __device__ constexpr uint2 operator/(unsigned int a, const uint2& b) {
    return make_uint2(a / b.x, a / b.y);
}

__host__ __device__ constexpr unsigned int dot(const uint2& a, const uint2& b) {
    return a.x * b.x + a.y * b.y;
}

// Operators on CUDA uint3

__host__ __device__ constexpr uint3 operator+(const uint3& a, const uint3& b) {
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ constexpr uint3 operator-(const uint3& a, const uint3& b) {
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ constexpr uint3 operator*(const uint3& a, const uint3& b) {
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ constexpr uint3 operator/(const uint3& a, const uint3& b) {
    return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ constexpr uint3 operator+(const uint3& a, unsigned int b) {
    return make_uint3(a.x + b, a.y + b, a.z + b);
}

__host__ __device__ constexpr uint3 operator-(const uint3& a, unsigned int b) {
    return make_uint3(a.x - b, a.y - b, a.z - b);
}

__host__ __device__ constexpr uint3 operator*(const uint3& a, unsigned int b) {
    return make_uint3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ constexpr uint3 operator/(const uint3& a, unsigned int b) {
    return make_uint3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ constexpr uint3 operator+(unsigned int a, const uint3& b) {
    return make_uint3(a + b.x, a + b.y, a + b.z);
}

__host__ __device__ constexpr uint3 operator-(unsigned int a, const uint3& b) {
    return make_uint3(a - b.x, a - b.y, a - b.z);
}

__host__ __device__ constexpr uint3 operator*(unsigned int a, const uint3& b) {
    return make_uint3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ constexpr uint3 operator/(unsigned int a, const uint3& b) {
    return make_uint3(a / b.x, a / b.y, a / b.z);
}

__host__ __device__ constexpr unsigned int dot(const uint3& a, const uint3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Operators on CUDA uint4

__host__ __device__ constexpr uint4 operator+(const uint4& a, const uint4& b) {
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ constexpr uint4 operator-(const uint4& a, const uint4& b) {
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ constexpr uint4 operator*(const uint4& a, const uint4& b) {
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__host__ __device__ constexpr uint4 operator/(const uint4& a, const uint4& b) {
    return make_uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__host__ __device__ constexpr uint4 operator+(const uint4& a, unsigned int b) {
    return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__host__ __device__ constexpr uint4 operator-(const uint4& a, unsigned int b) {
    return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b);
}

__host__ __device__ constexpr uint4 operator*(const uint4& a, unsigned int b) {
    return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__host__ __device__ constexpr uint4 operator/(const uint4& a, unsigned int b) {
    return make_uint4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__host__ __device__ constexpr uint4 operator+(unsigned int a, const uint4& b) {
    return make_uint4(a + b.x, a + b.y, a + b.z, a + b.w);
}

__host__ __device__ constexpr uint4 operator-(unsigned int a, const uint4& b) {
    return make_uint4(a - b.x, a - b.y, a - b.z, a - b.w);
}

__host__ __device__ constexpr uint4 operator*(unsigned int a, const uint4& b) {
    return make_uint4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__host__ __device__ constexpr uint4 operator/(unsigned int a, const uint4& b) {
    return make_uint4(a / b.x, a / b.y, a / b.z, a / b.w);
}

__host__ __device__ constexpr unsigned int dot(const uint4& a, const uint4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}