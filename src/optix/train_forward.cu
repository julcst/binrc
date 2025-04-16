#include <optix_device.h>
#include <cuda_runtime.h>
#include <array>

#include "params.cuh"
#include "payload.cuh"
#include "common.cuh"
#include "nrc.cuh"
#include "cudamath.cuh"
#include "sampling.cuh"

struct TrainBounce {
    float3 radiance = make_float3(0.0f);
    float3 throughput = make_float3(1.0f);
    float3 reflectanceFactorizationTerm = make_float3(1.0f);
    uint index = 0;
};

extern "C" __global__ void __raygen__() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;
    const auto rotation = params.rotationTable[i];

    const auto uv = RND_JITTER;
    auto ray = makeCameraRay(uv);

    const auto nee = params.lightTable && (params.flags & NEE_FLAG);

    Payload payload;
    auto throughput = make_float3(1.0f);
    auto prevBrdfPdf = 1.0f;
    auto lightPdfIsZero = true;
    auto trainBounceIdx = 0;

    std::array<TrainBounce, TRAIN_DEPTH> trainBounces;
    
    for (uint depth = 1; depth <= TRAIN_DEPTH; depth++) {
        trainBounceIdx = depth - 1;

        // Russian roulette
        const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        if (RND_ROULETTE >= pContinue) break;
        for (uint i = 0; i < trainBounceIdx; i++) {
            trainBounces[i].throughput /= pContinue;
        }
        throughput /= pContinue;

        payload = trace(ray);

        if (isinf(payload.t)) {
            for (uint i = 0; i < trainBounceIdx; i++) {
                trainBounces[i].radiance += trainBounces[i].throughput * payload.emission;
            }
            break; // Skybox
        }

        auto n = payload.normal;
        const auto wo = -ray.direction;
        const auto inside = dot(n, wo) < 0.0f;
        n = inside ? -n : n;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto metallic = payload.metallic;
        const auto baseColor = payload.baseColor; // baseColor

        const auto trainInput = encodeInput(hitPoint, wo, payload);
        const auto trainIdx = pushNRCTrainInput(trainInput);
        const auto reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
        trainBounces[trainBounceIdx].index = trainIdx;
        trainBounces[trainBounceIdx].reflectanceFactorizationTerm = reflectanceFactorizationTerm;

        if (luminance(payload.emission) > 0.0f) {
            auto weight = 1.0f;
            if (nee && !lightPdfIsZero) {
                // NOTE: Maybe calculating the prevBrdfPdf here only when necessary is faster
                const auto lightPdf = lightPdfUniform(wo, payload.t, n, payload.area);
                weight = balanceHeuristic(prevBrdfPdf, lightPdf);
            }
            for (uint i = 0; i < trainBounceIdx; i++) {
                trainBounces[i].radiance += trainBounces[i].throughput * payload.emission * weight;
            }
        }

        // Next event estimation
        if (nee) {
            const auto sample = sampleLight(RND_LSRC, RND_LSAMP, hitPoint);
            const auto cosThetaS = dot(sample.wi, n);
            //if (abs(cosThetaS) > 0.0f && abs(sample.cosThetaL) > 0.0f) {
                const auto brdf = evalDisney(wo, sample.wi, n, baseColor, metallic, alpha, payload.transmission, inside);
                const auto surfacePoint = hitPoint + n * copysignf(params.sceneEpsilon, cosThetaS);
                const auto lightPoint = sample.position - sample.n * copysignf(params.sceneEpsilon, dot(sample.wi, sample.n));
                if (!brdf.isDirac && brdf.pdf > 0.0f && !traceOcclusion(surfacePoint, lightPoint)) {
                    const auto weight = balanceHeuristic(sample.pdf, brdf.pdf);
                    const auto weightedEmission = brdf.throughput * sample.emission * weight / sample.pdf;
                    for (uint i = 0; i <= trainBounceIdx; i++) {
                        trainBounces[i].radiance += trainBounces[i].throughput * weightedEmission;
                    }
                }
            //}
        }

        const auto sample = sampleDisney(RND_BSDF, RND_MICROFACET, RND_DIFFUSE, wo, n, inside, payload.baseColor, payload.metallic, alpha, payload.transmission);
        
        ray = Ray{hitPoint + n * copysignf(params.sceneEpsilon, dot(sample.direction, n)), sample.direction};
        prevBrdfPdf = sample.pdf;
        lightPdfIsZero = sample.isDirac || payload.transmission > 0.0f;
        for (uint i = 0; i <= trainBounceIdx; i++) {
            trainBounces[i].throughput *= sample.throughput;
        }
        throughput *= sample.throughput;
    }

    for (uint i = 0; i < trainBounceIdx; i++) {
        writeNRCOutput(params.trainingTarget + trainBounces[i].index * NRC_OUTPUT_SIZE, trainBounces[i].radiance * trainBounces[i].reflectanceFactorizationTerm);
    }
}