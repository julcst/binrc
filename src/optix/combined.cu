#include <optix_device.h>
#include <cuda_runtime.h>

#include "params.cuh"
#include "payload.cuh"
#include "common.cuh"
#include "nrc.cuh"
#include "cudamath.cuh"
#include "sampling.cuh"

extern "C" __global__ void __raygen__combined() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;
    const auto rotation = params.rotationTable[i];

    const auto uv = (make_float2(idx.x, idx.y) + RND_JITTER) / make_float2(dim.x, dim.y);
    auto ray = makeCameraRay(uv);

    const auto nee = params.lightTable && (params.flags & NEE_FLAG);

    Payload payload;
    auto color = make_float3(0.0f);
    auto throughput = make_float3(1.0f);
    auto prevBrdfPdf = 1.0f;
    auto diracEvent = true;

    auto trainDepth = -1;
    // TODO: Train the whole path
    // TODO: Use stratified sampling
    if (RND_TRAIN1 < NRC_BATCH_SIZE / float(params.dim.x * params.dim.y)) {
        trainDepth = int(RND_TRAIN2 * 6) + 1;
    }
    auto trainTarget = NRCOutput{};
    uint trainIndex = 0;
    auto reflectanceFactorizationTerm = make_float3(1.0f);
    auto trainThroughput = make_float3(1.0f);
    bool isTrainingPath = trainDepth >= 0;
    bool writeTrainingSample = false;
    float3 inferenceThroughput = make_float3(1.0f);
    float3 inferencePlus = make_float3(0.0f);

    NRCInput nrcQuery {};
    
    for (uint depth = 1; depth < MAX_BOUNCES; depth++) {
        if (depth == trainDepth) {
            trainTarget.radiance = make_float3(0.0f);
            trainThroughput = make_float3(1.0f);
        }

        // Russian roulette
        const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        if (RND_ROULETTE >= pContinue) break;
        throughput /= pContinue;
        trainThroughput /= pContinue;

        payload = trace(ray, isTrainingPath ? 1u : 0u);

        if (isinf(payload.t)) {
            color += throughput * payload.emission;
            trainTarget.radiance += trainThroughput * payload.emission;
            break; // Skybox
        }

        auto n = payload.normal;
        const auto wo = -ray.direction;
        const auto inside = dot(n, wo) < 0.0f;
        n = inside ? -n : n;

        if (luminance(payload.emission) > 0.0f) {
            auto weight = 1.0f;
            if (nee && !diracEvent) {
                // NOTE: Maybe calculating the prevBrdfPdf here only when necessary is faster
                const auto lightPdf = lightPdfUniform(wo, payload.t, n, payload.area);
                weight = balanceHeuristic(prevBrdfPdf, lightPdf);
            }
            color += throughput * payload.emission * weight;
            trainTarget.radiance += trainThroughput * payload.emission * weight;
        }

        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto metallic = payload.metallic;
        const auto baseColor = payload.baseColor; // baseColor

        if (depth == trainDepth && luminance(payload.emission) < 1e-3f) { // NOTE: Skipping emissive vertices reduces variance
            const auto F0 = mix(make_float3(0.04f), baseColor, metallic);
            const auto albedo = (1.0f - metallic) * baseColor;
            auto trainInput = encodeInput(hitPoint, wo, n, albedo, F0, alpha);
            reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
            //const auto inputIdx = (i % NRC_BATCH_SIZE) * NRC_INPUT_SIZE;
            //pushNRCInput(params.trainingInput + inputIdx, trainInput);
            trainIndex = pushNRCTrainInput(trainInput);
            writeTrainingSample = true;
        }

        // NRC Inference Input
        if ((params.inferenceMode == InferenceMode::FIRST_VERTEX || params.inferenceMode == InferenceMode::RAW_CACHE) && depth == 1) {
            const auto F0 = mix(make_float3(0.04f), baseColor, metallic);
            const auto albedo = (1.0f - metallic) * baseColor;
            nrcQuery = encodeInput(hitPoint, wo, n, albedo, F0, alpha);
            inferenceThroughput = throughput;
            inferencePlus = color;
            if (!isTrainingPath) break;
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
                    color += throughput * brdf.throughput * sample.emission * weight / sample.pdf;
                    trainTarget.radiance += trainThroughput * brdf.throughput * sample.emission * weight / sample.pdf;
                }
            //}
        }

        // NRC Inference Input
        if (params.inferenceMode == InferenceMode::FIRST_VERTEX_WITH_NEE && depth == 1) {
            const auto F0 = mix(make_float3(0.04f), baseColor, metallic);
            const auto albedo = (1.0f - metallic) * baseColor;
            nrcQuery = encodeInput(hitPoint, wo, n, albedo, F0, alpha);
            inferenceThroughput = throughput;
            inferencePlus = color;
            if (!isTrainingPath) break;
        }

        const auto sample = sampleDisney(RND_BSDF, RND_MICROFACET, RND_DIFFUSE, wo, n, inside, payload.baseColor, payload.metallic, alpha, payload.transmission);
        
        ray = Ray{hitPoint + n * copysignf(params.sceneEpsilon, dot(sample.direction, n)), sample.direction};
        throughput *= sample.throughput;
        trainThroughput *= sample.throughput;
        prevBrdfPdf = sample.pdf;
        diracEvent = sample.isDirac;

        // NRC Inference Input
        if (params.inferenceMode == InferenceMode::FIRST_DIFFUSE && !sample.isSpecular) {
            const auto F0 = mix(make_float3(0.04f), baseColor, metallic);
            const auto albedo = (1.0f - metallic) * baseColor;
            nrcQuery = encodeInput(hitPoint, wo, n, albedo, F0, alpha);
            inferenceThroughput = throughput;
            inferencePlus = color;
            if (!isTrainingPath) break;
        }
    }

    if (writeTrainingSample) {
        trainTarget.radiance *= reflectanceFactorizationTerm;
        const auto outputIdx = (trainIndex % NRC_BATCH_SIZE) * NRC_OUTPUT_SIZE;
        pushNRCOutput(params.trainingTarget + outputIdx, trainTarget);
    }

    const auto inputIdx = i * NRC_INPUT_SIZE;
    pushNRCInput(params.inferenceInput + inputIdx, nrcQuery);
    params.inferenceThroughput[i] = inferenceThroughput;

    // NOTE: We should not need to prevent NaNs
    // FIXME: NaNs
    //if (isfinite(color))
    if (params.inferenceMode == InferenceMode::NO_INFERENCE) {
        params.image[i] = mix(params.image[i], make_float4(max(color, 0.0f), 1.0f), params.weight); // FIXME: Negative colors
    } else {
        params.image[i] = mix(params.image[i], make_float4(max(inferencePlus, 0.0f), 1.0f), params.weight); // FIXME: Negative colors
    }
}