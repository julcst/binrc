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

    auto queryRay = makeCameraRay({curand_uniform(&state), curand_uniform(&state)});
    Payload queryPayload = trace(queryRay);
    if (isinf(queryPayload.t)) return; // Nothing to train
    float3 queryX = queryRay.origin + queryPayload.t * queryRay.direction;
    float3 queryWo = -queryRay.direction;
    float3 queryLo = {0.0f, 0.0f, 0.0f};
    auto queryMat = calcMaterialProperties(queryPayload);
    if(queryMat.alpha2 < 1e-2f) printf("Query material F0 %.2f albedo %.2f alpha2 %.2f\n", length2(queryMat.F0), length2(queryMat.albedo), queryMat.alpha2);

    const auto r = curand_uniform4(&state); // Note curand generates in (0, 1] not [0, 1)
    const auto lightSample = samplePhoton(curand_uniform(&state), make_float2(r.x, r.y), make_float2(r.z, r.w));

    // radiance along path from x_0 to x_i-1 divided by the pdf of path x_0 to x_i-1
    auto radiance = lightSample.emission * INV_PI; // Divide by cos(wo) / pdf(wo) = cos(wo) / (cos(wo) / PI) = PI
    radiance *= params.balanceWeight; // Balancing

    uint lightSamples = 0;
    auto ray = Ray{lightSample.position + lightSample.n * copysignf(params.sceneEpsilon, dot(lightSample.wo, lightSample.n)), lightSample.wo};
    Payload payload;

    // Connect the light sample to the query point
    auto connectionDir = lightSample.position - queryX;
    const float dist2 = length2(connectionDir);
    if (dist2 > params.sceneEpsilon && !traceOcclusion(lightSample.position, lightSample.n, queryX, queryPayload.normal)) {
        connectionDir /= sqrtf(dist2); // Normalize direction
        const auto queryBRDF = evalDisneyBRDFCosine(queryWo, connectionDir, queryPayload.normal, queryMat);
        const auto Lo = radiance * queryBRDF * abs(dot(connectionDir, lightSample.n)) / dist2;
        queryLo += Lo;
    }

    radiance *= PI; // Multiply by cos(wo) / pdf(wo) = cos(wo) / (cos(wo) / PI) = PI

    for (uint depth = 0; depth < 6; depth++) {
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
        const auto alpha = payload.roughness * payload.roughness;
        const auto mat = calcMaterialProperties(payload);

        auto connectionDir = hitPoint - queryX;
        const float dist2 = length2(connectionDir);
        if (dist2 > params.sceneEpsilon && !traceOcclusion(hitPoint, payload.normal, queryX, queryPayload.normal)) {
            connectionDir /= sqrtf(dist2); // Normalize direction
            const auto queryBRDF = evalDisneyBRDFCosine(queryWo, connectionDir, queryPayload.normal, queryMat);
            const auto hitBRDF = evalDisneyBRDFCosine(wi, -connectionDir, payload.normal, mat);
            const auto Lo = radiance * queryBRDF * hitBRDF / dist2;
            queryLo += Lo;
        }

        const auto r = curand_uniform4(&state);
        const auto sample = sampleDisney(curand_uniform(&state), {r.x, r.y}, {r.z, r.w}, wi, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
        const auto brdf = evalDisneyBRDFCosine(wi, sample.direction, payload.normal, mat);
        radiance *= sample.throughput;

        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};

        //radiance += payload.emission; // NOTE: Do not train bounce emission
    }

    const auto trainInput = encodeInput(queryX, false, queryWo, queryPayload);
    const auto reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
    const auto output = min(reflectanceFactorizationTerm * queryLo , 10.0f);
    const auto trainIdx = pushNRCTrainInput(trainInput);
    writeNRCOutput(params.trainingTarget, trainIdx, output);
    lightSamples++;
    atomicAdd(params.lightSamples, lightSamples); // TODO: Aggregate
}