#pragma once

#include <cuda_runtime.h>

#include "cudamath.cuh"
#include "params.cuh"

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

__device__ inline void writeNRCInput(float* to, const NRCInput& input) {
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

__device__ inline void writeNRCInput(float* to, uint idx, const NRCInput& input) {
    writeNRCInput(to + idx * NRC_INPUT_SIZE, input);
}

__device__ inline uint pushNRCTrainInput(const NRCInput& input) {
    const auto i = atomicAdd(params.trainingIndexPtr, 1u);
    writeNRCInput(params.trainingInput + (i % NRC_BATCH_SIZE) * NRC_INPUT_SIZE, input);
    return i % NRC_BATCH_SIZE;
}

__device__ inline void writeNRCOutput(float* to, const float3& radiance) {
    to[0] = radiance.x;
    to[1] = radiance.y;
    to[2] = radiance.z;
}

__device__ inline void writeNRCOutput(float* to, const NRCOutput& output) {
    writeNRCOutput(to, output.radiance);
}