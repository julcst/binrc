#include <optix_device.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <array>

#include "params.cuh"
#include "payload.cuh"
#include "common.cuh"
#include "nrc.cuh"
#include "cudamath.cuh"
#include "sampling.cuh"

constexpr uint N_RANDS = 5 + (TRAIN_DEPTH + 1) * 6;

extern "C" __global__ void __raygen__() {
    if (!params.lightTable) return; // Cannot sample light without lights
    
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(0, i, params.trainingRound * N_RANDS, &state);

    const auto r = curand_uniform4(&state); // Note curand generates in (0, 1] not [0, 1)
    const auto lightSample = sampleLight(curand_uniform(&state), make_float2(r.x, r.y), make_float2(r.z, r.w));
    auto ray = Ray{lightSample.position + lightSample.n * copysignf(params.sceneEpsilon, dot(lightSample.wo, lightSample.n)), lightSample.wo};
    auto radiance = lightSample.emission * INV_PI;
    // printf("Light sample: %f %f %f\n", lightSample.emission.x, lightSample.emission.y, lightSample.emission.z);
    radiance *= 2.0f; // Balancing

    Payload payload;

    for (uint depth = 1; depth <= TRAIN_DEPTH; depth++) {
        // Russian roulette
        if (params.flags & BACKWARD_RR_FLAG) {
            const float pContinue = min(luminance(radiance) * params.russianRouletteWeight, 1.0f);
            if (curand_uniform(&state) >= pContinue) break; // FIXME: use random numbers independent from sampling
            radiance /= pContinue;
        }

        payload = trace(ray);

        if (isinf(payload.t)) break; // Skybox

        auto n = payload.normal;
        const auto wi = -ray.direction;
        const auto inside = dot(n, wi) < 0.0f;
        n = inside ? -n : n;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;

        const auto r = curand_uniform4(&state);
        const auto sample = sampleDisney(curand_uniform(&state), {r.x, r.y}, {r.z, r.w}, wi, n, inside, payload.baseColor, payload.metallic, alpha, payload.transmission);

        radiance *= sample.throughput;

        // printf("Sample throughput: %f\n", pow2(sample.throughput));

        ray = Ray{hitPoint + n * copysignf(params.sceneEpsilon, dot(sample.direction, n)), sample.direction};

        //if (depth > 1) {
            const auto trainInput = encodeInput(hitPoint, !sample.isSpecular && (params.flags & DIFFUSE_ENCODING_FLAG) ? make_float3(NAN) : sample.direction, n, payload);
            const auto trainIdx = pushNRCTrainInput(trainInput);
            const auto reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
            writeNRCOutput(params.trainingTarget, trainIdx, reflectanceFactorizationTerm * radiance);
            atomicAdd(params.lightSamples, 1u);
        //}

        radiance += payload.emission;
    }
}