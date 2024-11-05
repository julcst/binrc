#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "cudamathtypes.cuh"

// Conversion glm -> CUDA

__host__ __device__ constexpr float2 glmToCuda(const glm::vec2& v) {
    return make_float2(v.x, v.y);
}

__host__ __device__ constexpr float3 glmToCuda(const glm::vec3& v) {
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

__host__ __device__ constexpr float4x4 glmToCuda(const glm::mat4& m) {
    return make_float4x4(
        glmToCuda(m[0]),
        glmToCuda(m[1]),
        glmToCuda(m[2]),
        glmToCuda(m[3])
    );
}

// Conversion CUDA -> glm

__host__ __device__ constexpr glm::vec2 cudaToGlm(const float2& v) {
    return {v.x, v.y};
}

__host__ __device__ constexpr glm::vec3 cudaToGlm(const float3& v) {
    return {v.x, v.y, v.z};
}

__host__ __device__ constexpr glm::vec4 cudaToGlm(const float4& v) {
    return {v.x, v.y, v.z, v.w};
}

__host__ __device__ constexpr glm::ivec2 cudaToGlm(const int2& v) {
    return {v.x, v.y};
}

__host__ __device__ constexpr glm::ivec3 cudaToGlm(const int3& v) {
    return {v.x, v.y, v.z};
}

__host__ __device__ constexpr glm::ivec4 cudaToGlm(const int4& v) {
    return {v.x, v.y, v.z, v.w};
}

__host__ __device__ constexpr glm::uvec2 cudaToGlm(const uint2& v) {
    return {v.x, v.y};
}

__host__ __device__ constexpr glm::uvec3 cudaToGlm(const uint3& v) {
    return {v.x, v.y, v.z};
}

__host__ __device__ constexpr glm::uvec4 cudaToGlm(const uint4& v) {
    return {v.x, v.y, v.z, v.w};
}

// Special functions

/**
 * Reorthogonalizes a tangent space using the Gram-Schmidt process and returns an orthonormal tangent space matrix
 * Note: n needs to be normalized and t must be linearly independent from n
 */
__host__ __device__ constexpr glm::mat3 buildTBN(const glm::vec3& n, const glm::vec3& t) {
    const auto nt = glm::normalize(t - dot(t, n) * n);
    const auto b = glm::cross(n, nt);
    return glm::mat3(nt, b, n);
}

/**
 * Builds an orthogonal tangent space to world space matrix from a normalized normal
 */
__host__ __device__ constexpr glm::mat3 buildTBN(const glm::vec3& n) {
    if (glm::abs(n.y) > 0.99f) {
        glm::vec3 t = normalize(glm::cross(n, glm::vec3(1.0f, 0.0f, 0.0f)));
        return {t, glm::cross(n, t), n};
    } else {
        glm::vec3 t = glm::normalize(glm::cross(n, glm::vec3(0.0f, 1.0f, 0.0f)));
        return {t, glm::cross(n, t), n};
    }
}