#include <optix.h>

#include "params.cuh"

extern "C" __global__ void __raygen__() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    params.photonMap.store(idx.x, PhotonQuery {
        .pos = {(float) idx.x, 0, 0},
        .wo = {1, 0, 0},
        .n = {0, 1, 0},
        .mat = MaterialProperties {
            .F0 = {1.0f, 0.0f, 0.0f},
            .albedo = {1.0f, 0.0f, 0.0f},
            .alpha2 = 0.1f,
            .transmission = 0.0f
        },
        .radius = 2.0f
    });
}