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
#include "principled_brdf.cuh"

constexpr uint N_RANDS = 5 + (TRAIN_DEPTH + 1) * 6;

extern "C" __global__ void __raygen__() {
    if (!params.lightTable) return; // Cannot sample light without lights
    
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(0, i, params.trainingRound * N_RANDS, &state);

    const auto r = curand_uniform4(&state); // Note curand generates in (0, 1] not [0, 1)
    const auto lightSample = samplePhoton(curand_uniform(&state), make_float2(r.x, r.y), make_float2(r.z, r.w));
    auto ray = Ray{lightSample.position + lightSample.n * copysignf(params.sceneEpsilon, dot(lightSample.wo, lightSample.n)), lightSample.wo};
    auto radiance = params.flags & LIGHT_TRACE_FIX_FLAG ? lightSample.emission : lightSample.emission * INV_PI; // FIXME: Why / PI² ?
    // printf("Light sample: %f %f %f\n", lightSample.emission.x, lightSample.emission.y, lightSample.emission.z);
    radiance *= params.balanceWeight; // Balancing
    uint lightSamples = 0;

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

        const auto wi = -ray.direction;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;

        const auto r = curand_uniform4(&state);
        // FIXME: Wrong IOR
        const auto sample = sampleDisney(curand_uniform(&state), {r.x, r.y}, {r.z, r.w}, wi, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);

        

        // printf("Sample throughput: %f\n", pow2(sample.throughput));

        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};

        //if (depth > 1) {
        const auto trainInput = encodeInput(hitPoint, !sample.isSpecular && (params.flags & DIFFUSE_ENCODING_FLAG), sample.direction, payload);
        const auto reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
        const auto mat = calcMaterialProperties(payload.baseColor, payload.metallic, payload.roughness, payload.transmission);
        const auto brdf = min(evalDisneyBRDFCosine(wi, sample.direction, payload.normal, mat), 1e0f); // TODO: cosineThetaI correct?
        const auto output = reflectanceFactorizationTerm * radiance * brdf;
        radiance *= sample.throughput;

        const auto trainIdx = pushNRCTrainInput(trainInput);
        writeNRCOutput(params.trainingTarget, trainIdx, output);
        lightSamples++;
        //}

        //radiance += payload.emission; // TODO: Do not train bounce emission
    }

    atomicAdd(params.lightSamples, lightSamples); // TODO: Aggregate
}