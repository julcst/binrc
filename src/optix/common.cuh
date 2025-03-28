#pragma once

#include <cuda_runtime.h>

#include "cudamath.cuh"
#include "params.cuh"

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

__device__ inline float getRand(uint dim) {
    const uint i = params.sample - params.sequenceOffset + params.sequenceStride * dim;
    return params.randSequence[i];
}

__device__ inline float getRand(uint depth, uint i) {
    return getRand(RANDS_PER_PIXEL + depth * RANDS_PER_BOUNCE + i);
}

__device__ inline float getRand(uint depth, uint offset, float rotation) {
    return fract(getRand(depth, offset) + rotation);
}

__device__ inline float2 getRand(uint depth, uint offset, float r0, float r1) {
    return fract(make_float2(getRand(depth, offset + 0) + r0, getRand(depth, offset + 1) + r1));
}

__device__ inline float3 getRand(uint depth, uint offset, float r0, float r1, float r2) {
    return fract(make_float3(getRand(depth, offset + 0) + r0, getRand(depth, offset + 1) + r1, getRand(depth, offset + 2) + r2));
}

__device__ inline NRCInput encodeInput(const float3& position, const float3& wo, const float3& wn, const float3& diffuse, const float3& specular, float alpha) {
    return {
        .position = params.sceneScale * (position - params.sceneMin),
        .wo = toNormSpherical(wo), // Switch to Octahedral
        .wn = toNormSpherical(wn), // TODO: Switch to Octahedral
        //.roughness = 1 - exp(-alpha),
        .roughness = alpha,
        .diffuse = diffuse,
        .specular = specular, // TODO: directional albedo FDG
    };
}

__device__ inline NRCInput encodeInput(const float3& position, const float3& wo, const Payload& payload) {
    const auto F0 = mix(make_float3(0.04f), payload.baseColor, payload.metallic);
    const auto albedo = (1.0f - payload.metallic) * payload.baseColor;
    return encodeInput(position, wo, payload.normal, albedo, F0, payload.roughness * payload.roughness);
}

__device__ inline void pushNRCInput(float* to, const NRCInput& input) {
    to[0] = input.position.x;
    to[1] = input.position.y;
    to[2] = input.position.z;
    to[3] = input.wo.x;
    to[4] = input.wo.y;
    to[5] = input.wn.x;
    to[6] = input.wn.y;
    to[7] = input.roughness;
    to[8] = input.diffuse.x;
    to[9] = input.diffuse.y;
    to[10] = input.diffuse.z;
    to[11] = input.specular.x;
    to[12] = input.specular.y;
    to[13] = input.specular.z;
}

__device__ inline uint pushNRCTrainInput(const NRCInput& input) {
    const auto i = atomicAdd(params.trainingIndexPtr, 1u);
    pushNRCInput(params.trainingInput + (i % NRC_BATCH_SIZE) * NRC_INPUT_SIZE, input);
    return i;
}

__device__ inline void pushNRCOutput(float* to, const NRCOutput& output) {
    to[0] = output.radiance.x;
    to[1] = output.radiance.y;
    to[2] = output.radiance.z;
}

__device__ inline Payload trace(const Ray& ray, uint hint) {
    std::array<uint, 17> p;
    optixTraverse(
        params.handle,
        ray.origin, ray.direction,
        0.0f, MAX_T, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT offset, stride, miss index
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
        0, 1, 0, // SBT offset, stride, miss index
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
        0, 1, 0 // SBT offset, stride, miss index
    );
    return optixHitObjectIsHit();
}