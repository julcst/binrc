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
    const auto pixelIdx = idx.y * params.dim.x + idx.x;
    
    curandStatePhilox4_32_10_t state;
    curand_init(0, pixelIdx, params.trainingRound * N_RANDS, &state);

    auto ray = makeCameraRay({curand_uniform(&state), curand_uniform(&state)});

    const auto nee = params.lightTable && (params.flags & NEE_FLAG);

    Payload payload;
    auto throughput = make_float3(1.0f);
    auto lightPdfIsZero = true;
    SampleResult sample {ray.direction, make_float3(1.0f), 1.0f, true, true};

    std::array<TrainBounce, TRAIN_DEPTH> trainBounces;
    for (uint i = 0; i < TRAIN_DEPTH; i++) {
        trainBounces[i].isValid = false;
    }

    uint32_t nTrainingSamples = 0;
    for (uint32_t depth = 0; depth < TRAIN_DEPTH; depth++) {

        // Russian roulette
        if (params.flags & FORWARD_RR_FLAG) {
            const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
            if (curand_uniform(&state) >= pContinue) break;
            if (!(params.flags & SELF_LEARNING_FLAG)) {
                if (depth > 0) trainBounces[depth - 1].throughput /= pContinue;
                throughput /= pContinue;
            }
        }

        payload = trace(ray);

        if (isinf(payload.t)) {
            auto radiance = payload.emission;
            // NOTE: This is logically equivalent to for(i = n; i >= 0; i--) but avoids unsigned underflow
            for (uint i = depth - 1; i < TRAIN_DEPTH; i--) {
                radiance *= trainBounces[i].throughput;
                trainBounces[i].radiance += radiance;
            }
            break; // Skybox
        }

        nTrainingSamples++;
        trainBounces[depth].throughput = sample.throughput;
        const auto wo = -ray.direction;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto metallic = payload.metallic;
        const auto baseColor = payload.baseColor; // baseColor

        if (luminance(payload.emission) > 0.0f) {
            auto weight = 1.0f;
            if (nee && !lightPdfIsZero) {
                // NOTE: Maybe calculating the prevBrdfPdf here only when necessary is faster
                const auto lightPdf = lightPdfUniform(wo, payload.t, payload.normal, payload.area);
                weight = balanceHeuristic(sample.pdf, lightPdf);
            }
            auto radiance = payload.emission * weight;
            for (uint i = depth - 1; i < TRAIN_DEPTH; i--) {
                radiance *= trainBounces[i].throughput;
                trainBounces[i].radiance += radiance;
            }
        }

        // Next event estimation
        if (nee) {
            const auto sample = sampleLight(curand_uniform(&state), {curand_uniform(&state), curand_uniform(&state)}, hitPoint);
            //if (abs(cosThetaS) > 0.0f && abs(sample.cosThetaL) > 0.0f) {
                const auto brdf = evalDisney(wo, sample.wi, payload.normal, baseColor, metallic, alpha, payload.transmission);
                const auto surfacePoint = hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.wi, payload.normal));
                const auto lightPoint = sample.position - sample.n * copysignf(params.sceneEpsilon, dot(sample.wi, sample.n));
                if (!brdf.isDirac && brdf.pdf > 0.0f && !traceOcclusion(surfacePoint, lightPoint)) {
                    const auto weight = balanceHeuristic(sample.pdf, brdf.pdf);
                    auto radiance = brdf.throughput * sample.emission * weight / sample.pdf;
                    for (uint i = depth; i < TRAIN_DEPTH; i--) {
                        radiance *= trainBounces[i].throughput;
                        trainBounces[i].radiance += radiance;
                    }
                }
            //}
        }

        const auto r = curand_uniform4(&state);
        sample = sampleDisney(curand_uniform(&state), {r.x, r.y}, {r.z, r.w}, wo, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
        
        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};
        lightPdfIsZero = sample.isDirac;
        trainBounces[depth].throughput = sample.throughput;
        throughput *= sample.throughput;

        const auto trainInput = encodeInput(hitPoint, !sample.isSpecular && (params.flags & DIFFUSE_ENCODING_FLAG), wo, payload);
        const auto trainIdx = pushNRCTrainInput(trainInput);
        const auto reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
        trainBounces[depth].index = trainIdx;
        trainBounces[depth].reflectanceFactorizationTerm = reflectanceFactorizationTerm;
        trainBounces[depth].isValid = true;
    }

    // TODO: Keep 1/16 of learning paths unbiased from self-learning
    if (params.flags & SELF_LEARNING_FLAG && pixelIdx % 16 != 0) {
        params.selfLearningBounces[pixelIdx] = trainBounces;
        // Component-wise copy of the training input to self-learning queries
        auto terminalBounceInputIdx = nTrainingSamples > 0 ? trainBounces[nTrainingSamples - 1].index : trainBounces[0].index;
        for (uint i = 0; i < NRC_INPUT_SIZE; i++) {
            params.selfLearningQueries[pixelIdx * NRC_INPUT_SIZE + i] = params.trainingInput[terminalBounceInputIdx * NRC_INPUT_SIZE + i];
        }
    } else {
        // TODO: Aggregated atomics
        for (uint32_t i = 0; i < nTrainingSamples; i++) {
            writeNRCOutput(params.trainingTarget + trainBounces[i].index * NRC_OUTPUT_SIZE, trainBounces[i].radiance * trainBounces[i].reflectanceFactorizationTerm);
        }
    }
}