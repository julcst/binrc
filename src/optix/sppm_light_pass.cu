#include <optix.h>
#include <curand_kernel.h>

#include "params.cuh"
#include "sampling.cuh"
#include "common.cuh"
#include "principled_brdf.cuh"

extern "C" __global__ void __raygen__() {
    if (!params.lightTable) return; // Cannot sample light without lights
    
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;

    curandStatePhilox4_32_10_t state;
    constexpr uint32_t N_RANDS = 4 * MAX_BOUNCES + 4; // Number of random numbers to generate at once
    curand_init(0, i, params.trainingRound * N_RANDS, &state);

    const auto r = curand_uniform4(&state); // Note curand generates in (0, 1] not [0, 1)
    const auto lightSample = samplePhoton(curand_uniform(&state), make_float2(r.x, r.y), make_float2(r.z, r.w));
    auto ray = Ray{lightSample.position + lightSample.n * copysignf(params.sceneEpsilon, dot(lightSample.wo, lightSample.n)), lightSample.wo};
    auto flux = lightSample.emission;

    Payload payload;

    for (uint depth = 0; depth < MAX_BOUNCES; depth++) {
        const auto r = curand_uniform4(&state);

        // Russian roulette
        const float pContinue = min(luminance(flux) * params.russianRouletteWeight, 1.0f);
        if (r.x >= pContinue) break;
        flux /= pContinue;

        payload = trace(ray);

        if (isinf(payload.t)) break; // Skybox

        const auto wi = -ray.direction;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;

        params.photonMap.recordPhoton({
            .pos = hitPoint,
            .wi = wi,
            .flux = flux,
        });

        const auto sample = sampleDisney(r.y, {r.z, r.w}, {r.z, r.w}, wi, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
        flux *= sample.throughput;
        //flux += payload.emission; // TODO: Is this correct?

        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};
    }
}