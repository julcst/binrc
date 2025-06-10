#pragma once

#include <cuda_runtime.h>

#include "cudamath.cuh"
#include "params.cuh"
#include "payload.cuh"

struct Ray {
    float3 origin;
    float3 direction;
};

__device__ inline Ray makeCameraRay(const float2& uv) {
    const float4 origin = params.clipToWorld[3]; // = params.clipToWorld * make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    const float4 clipTarget = make_float4(-2.0f * uv + 1.0f, -1.0f, 1.0f);
    const float4 target = params.clipToWorld * clipTarget;
    const float3 origin3 = make_float3(origin) / origin.w;
    const float3 dir3 = normalize(origin3 - make_float3(target) / target.w);
    return {origin3, dir3};
}

__device__ inline Payload trace(const Ray& ray, uint hint) {
    std::array<uint, 17> p;
    optixTraverse(
        params.handle,
        ray.origin, ray.direction,
        0.0f, MAX_T, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        1, 1, 0, // SBT offset, stride, miss index // NOTE: HitRecord 0 is used for photon mapping
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16]
    );
    optixReorder(hint, 1);
    optixInvoke(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16]);
    return getPayload(p);
}

__device__ inline Payload trace(const Ray& ray) {
    std::array<uint, 17> p;
    optixTraverse(
        params.handle,
        ray.origin, ray.direction,
        0.0f, MAX_T, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        1, 1, 0, // SBT offset, stride, miss index // NOTE: HitRecord 0 is used for photon mapping
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16]
    );
    optixReorder();
    optixInvoke(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16]);
    return getPayload(p);
}

__device__ inline bool traceOcclusion(const float3& a, const float3& b) {
    const auto dir = b - a;
    optixTraverse(
        params.handle,
        a, dir,
        0.0f, 1.0f, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        1, 1, 1 // SBT offset, stride, miss index // NOTE: HitRecord 0 is used for photon mapping, MissRecord 1 is null
    );
    return optixHitObjectIsHit();
}