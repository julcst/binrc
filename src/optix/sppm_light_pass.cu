#include <optix.h>

#include "params.cuh"

extern "C" __global__ void __raygen__() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    params.photonMap.recordPhoton({
        .pos = {(float) idx.x, 0, 0}, // Assuming a fixed position for photons
        .wi = {1, 0, 0}, // Assuming a fixed direction for photons
        .flux = make_float3(1.0f, 1.0f, 1.0f), // Assuming a fixed flux for photons
    });
}