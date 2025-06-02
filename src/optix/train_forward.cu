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

constexpr uint N_RANDS = 2 + (TRAIN_DEPTH) * 9;

extern "C" __global__ void __raygen__() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;
    
    curandStatePhilox4_32_10_t state;
    curand_init(0, i, params.trainingRound * N_RANDS, &state);

    auto ray = makeCameraRay({curand_uniform(&state), curand_uniform(&state)});

    const auto nee = params.lightTable && (params.flags & NEE_FLAG);

    Payload payload;
    auto throughput = make_float3(1.0f);
    auto lightPdfIsZero = true;
    auto trainBounceIdx = 0;
    SampleResult sample {ray.direction, make_float3(1.0f), 1.0f, true, true};

    std::array<TrainBounce, TRAIN_DEPTH> trainBounces;
    for (uint i = 0; i < TRAIN_DEPTH; i++) {
        trainBounces[i].isValid = false;
    }
    
    for (uint depth = 1; depth <= TRAIN_DEPTH; depth++) {
        trainBounceIdx = depth - 1;

        // Russian roulette
        if (params.flags & FORWARD_RR_FLAG) {
            const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
            if (curand_uniform(&state) >= pContinue) break;
            for (uint i = 0; i < trainBounceIdx; i++) {
                trainBounces[i].throughput /= pContinue;
            }
            throughput /= pContinue;
        }

        payload = trace(ray);

        if (isinf(payload.t)) {
            for (uint i = 0; i < trainBounceIdx; i++) {
                trainBounces[i].radiance += trainBounces[i].throughput * sample.throughput * payload.emission;
            }
            break; // Skybox
        }

        for (uint i = 0; i <= trainBounceIdx; i++) {
            trainBounces[i].throughput *= sample.throughput;
        }

        auto n = payload.normal;
        const auto wo = -ray.direction;
        const auto inside = dot(n, wo) < 0.0f;
        n = inside ? -n : n;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto metallic = payload.metallic;
        const auto baseColor = payload.baseColor; // baseColor

        if (luminance(payload.emission) > 0.0f) {
            auto weight = 1.0f;
            if (nee && !lightPdfIsZero) {
                // NOTE: Maybe calculating the prevBrdfPdf here only when necessary is faster
                const auto lightPdf = lightPdfUniform(wo, payload.t, n, payload.area);
                weight = balanceHeuristic(sample.pdf, lightPdf);
            }
            for (uint i = 0; i < trainBounceIdx; i++) {
                trainBounces[i].radiance += trainBounces[i].throughput * payload.emission * weight;
            }
        }

        // Next event estimation
        if (nee) {
            const auto sample = sampleLight(curand_uniform(&state), {curand_uniform(&state), curand_uniform(&state)}, hitPoint);
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

        const auto r = curand_uniform4(&state);
        sample = sampleDisney(curand_uniform(&state), {r.x, r.y}, {r.z, r.w}, wo, n, inside, payload.baseColor, payload.metallic, alpha, payload.transmission);
        
        ray = Ray{hitPoint + n * copysignf(params.sceneEpsilon, dot(sample.direction, n)), sample.direction};
        lightPdfIsZero = sample.isDirac;
        for (uint i = 0; i <= trainBounceIdx; i++) {
            trainBounces[i].throughput *= sample.throughput;
        }
        throughput *= sample.throughput;

        const auto trainInput = encodeInput(hitPoint, !sample.isSpecular && (params.flags & DIFFUSE_ENCODING_FLAG) ? make_float3(NAN) : wo, n, payload);
        const auto trainIdx = pushNRCTrainInput(trainInput);
        const auto reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
        trainBounces[trainBounceIdx].index = trainIdx;
        trainBounces[trainBounceIdx].reflectanceFactorizationTerm = reflectanceFactorizationTerm;
        trainBounces[trainBounceIdx].isValid = true;
    }

    // TODO: Keep 1/16 of learning paths unbiased from self-learning
    if (params.flags & SELF_LEARNING_FLAG) {
        params.selfLearningBounces[i] = trainBounces;
        for (uint j = 0; j < NRC_INPUT_SIZE; j++) {
            params.selfLearningQueries[i * NRC_INPUT_SIZE + j] = params.trainingInput[trainBounces[trainBounceIdx].index * NRC_INPUT_SIZE + j];
        }
    } else {
        for (uint i = 0; i < trainBounceIdx; i++) {
            writeNRCOutput(params.trainingTarget + trainBounces[i].index * NRC_OUTPUT_SIZE, trainBounces[i].radiance * trainBounces[i].reflectanceFactorizationTerm);
        }
    }
}