#include <optix.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "params.cuh"
#include "payload.cuh"
#include "principled_brdf.cuh"
#include "common.cuh"

extern "C" __global__ void __raygen__() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;
    
    curandStatePhilox4_32_10_t state;
    constexpr uint32_t N_RANDS = 64;
    curand_init(0, i, params.trainingRound * N_RANDS, &state);

    auto ray = makeCameraRay({curand_uniform(&state), curand_uniform(&state)});

    float3 throughput = {1.0f};

    for (uint depth = 0; depth <= 6; depth++) {

        const auto r = curand_uniform4(&state);

        // Russian roulette
        const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        if (r.x >= pContinue) break;
        throughput /= pContinue;

        const auto payload = trace(ray);

        if (isinf(payload.t)) break; // Skybox

        const auto wo = -ray.direction;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;

        params.photonMap.store({
            .pos = hitPoint,
            .wo = wo,
            .n = payload.normal,
            .mat = calcMaterialProperties(payload.baseColor, payload.metallic, alpha, payload.transmission),
            .radius = 0.01f, // TODO: Radius reduction
            .totalPhotonCountAtBirth = params.photonMap.totalPhotonCount,
        });

        const auto sample = sampleDisney(r.y, {r.z, r.w}, {r.z, r.w}, wo, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
        
        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};
        throughput *= sample.throughput;
    }
}