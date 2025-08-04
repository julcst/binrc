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

constexpr uint N_RANDS = 5 + (MAX_BOUNCES) * 6;

extern "C" __global__ void __raygen__() {
    if (!params.lightTable) return; // Cannot sample light without lights
    
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(0, i, params.trainingRound * N_RANDS, &state);
    const auto r = curand_uniform4(&state); // Note curand generates in (0, 1] not [0, 1)

    const auto lightSample = samplePhoton(curand_uniform(&state), make_float2(r.x, r.y), make_float2(r.z, r.w));

    // radiance along path from x_0 to x_i-1 divided by the pdf of path x_0 to x_i-1
    auto radiance = lightSample.emission * INV_PI; // Divide by cos(wo) / pdf(wo) = cos(wo) / (cos(wo) / PI) = PI
    radiance *= params.balanceWeight; // Balancing

    // Probability of sampling the incoming light direction
    float p_wi = abs(dot(lightSample.wo, lightSample.n)) * INV_PI;
    //radiance *= abs(dot(lightSample.wo, lightSample.n)); // Scale by the pdf of the sampled direction

    uint lightSamples = 0;
    auto ray = Ray{lightSample.position + lightSample.n * copysignf(params.sceneEpsilon, dot(lightSample.wo, lightSample.n)), lightSample.wo};
    Payload payload;
    float3 prevPoint = lightSample.position;

    for (uint depth = 0; depth < params.maxPathLength - 1; depth++) {
        // Russian roulette
        // if (params.flags & BACKWARD_RR_FLAG) {
        //     const float pContinue = min(luminance(radiance) * params.russianRouletteWeight, 1.0f);
        //     if (curand_uniform(&state) >= pContinue) break; // FIXME: use random numbers independent from sampling
        //     radiance /= pContinue;
        // }

        payload = trace(ray);

        if (isinf(payload.t)) break; // Skybox

        const auto wi = -ray.direction;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const float dist2 = length2(hitPoint - prevPoint);
        prevPoint = hitPoint;
        const auto alpha = payload.roughness * payload.roughness;

        const auto r = curand_uniform4(&state);
        const auto sample = sampleDisney(curand_uniform(&state), {r.x, r.y}, {r.z, r.w}, wi, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
        const auto mat = calcMaterialProperties(payload.baseColor, payload.metallic, payload.roughness, payload.transmission);
        const auto brdf = evalDisneyBRDF(wi, sample.direction, payload.normal, mat); // TODO: cosineThetaI correct?
        const auto cosThetaI = abs(dot(wi, payload.normal));
        const auto cosThetaO = abs(dot(sample.direction, payload.normal));

        //const auto Lo = radiance * brdf * cosThetaI / max(dist2, 1.0f);
        const auto Lo = radiance * brdf; // Naive solution
        // TODO: Handle dirac
        //radiance *= brdf * cosThetaO / max(p_wi, 1.0f);
        radiance *= sample.throughput; // Naive solution
        p_wi = sample.pdf;

        const auto trainInput = encodeInput(hitPoint, !sample.isSpecular && (params.flags & DIFFUSE_ENCODING_FLAG), sample.direction, payload);
        const auto reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
        const auto output = min(reflectanceFactorizationTerm * Lo , 10.0f);
        const auto trainIdx = pushNRCTrainInput(trainInput);
        writeNRCOutput(params.trainingTarget, trainIdx, output);
        lightSamples++;

        // printf("Sample throughput: %f\n", length2(sample.throughput));

        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};

        //radiance += payload.emission; // TODO: Do not train bounce emission
    }

    atomicAdd(params.lightSamples, lightSamples); // TODO: Aggregate
}