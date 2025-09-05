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

    float3 throughput = make_float3(1.0f);

    for (uint depth = 0; depth <= 6; depth++) {

        const auto r = curand_uniform4(&state);

        // Russian roulette
        // const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        // if (r.x >= pContinue) break;
        // throughput /= pContinue;

        const auto payload = trace(ray);

        if (isinf(payload.t)) break; // Skybox

        const auto wo = -ray.direction;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto mat = calcMaterialProperties(payload.baseColor, payload.metallic, alpha, payload.transmission);
        const auto sample = sampleDisney(r.y, {r.z, r.w}, {r.z, r.w}, wo, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);

        // Alternative: Check pdf <= 1 / PI
        if (mat.isDiffuse()) {
            params.photonMap.store({
                .pos = hitPoint,
                .wo = wo,
                .n = payload.normal,
                .mat = mat,
                .radius = params.photonMap.initialRadius,
                .totalPhotonCountAtBirth = params.photonMap.totalPhotonCount,
            });
            break;
        }
        
        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};
        throughput *= sample.throughput;
    }
}

extern "C" __global__ void __raygen__full() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;
    if (i > params.photonMap.queryCount) return; // Safety check

    curandStatePhilox4_32_10_t state;
    constexpr uint32_t N_RANDS = 64;
    curand_init(0, i, params.trainingRound * N_RANDS, &state);

    const auto uv = (make_float2(idx.x, idx.y) + make_float2(curand_uniform(&state), curand_uniform(&state))) / make_float2(dim.x, dim.y);
    auto ray = makeCameraRay(uv);

    float3 throughput = make_float3(1.0f);
    float3 radiancePlus = make_float3(0.0f);

    for (uint depth = 0; depth < MAX_BOUNCES; depth++) {
        const auto r = curand_uniform4(&state);

        const auto payload = trace(ray);

        if (isinf(payload.t)) {// Skybox => No photon query
            params.photonMap.queries[i].throughput = {0.0f, 0.0f, 0.0f};
            params.photonMap.markAABBInvalid(i);
            radiancePlus += throughput * payload.emission;
            break;
        }

        if (luminance(payload.emission) > 0.0f) { // Add self-emission
            radiancePlus += throughput * payload.emission;
        }

        const auto wo = -ray.direction;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto mat = calcMaterialProperties(payload.baseColor, payload.metallic, alpha, payload.transmission);
        const auto sample = sampleDisney(r.y, {r.z, r.w}, {r.z, r.w}, wo, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);

        //if (sample.pdf <= INV_PI + 1e-2f) {
        if (mat.isDiffuse()) {
            auto query = params.photonMap.queries[i];
            query.pos = hitPoint;
            query.wo = wo;
            query.n = payload.normal;
            query.mat = mat;
            query.throughput = throughput;
            query.collectedPhotons = 0;
            params.photonMap.store(i, query);
            break;
        }
        
        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};
        throughput *= sample.throughput;
    }

    // No valid query found -> Zero it out
    // TODO: On termination, store previous query
    params.image[i] = mix(params.image[i], make_float4(radiancePlus, 1.0f), params.weight);
}