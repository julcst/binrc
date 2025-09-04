#include <optix_device.h>
#include <cuda_runtime.h>

#include "params.cuh"
#include "payload.cuh"
#include "common.cuh"
#include "nrc.cuh"
#include "cudamath.cuh"
#include "sampling.cuh"
#include "principled_brdf.cuh"

extern "C" __global__ void __raygen__() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;
    const auto rotation = params.rotationTable[i];

    const auto uv = (make_float2(idx.x, idx.y) + RND_JITTER) / make_float2(dim.x, dim.y);
    auto ray = makeCameraRay(uv);

    const auto nee = params.lightTable && (params.flags & INFERENCE_NEE_FLAG);
    //const auto nee = false;

    Payload payload;
    auto isPayloadValid = false;
    auto prevBrdfPdf = 1.0f;
    auto lightPdfIsZero = true;
    auto hitPoint = ray.origin;
    auto wo = ray.direction;
    auto diffuse = false;
    float primaryVariance = 0.0f;
    VarianceHeuristic varianceHeuristic;

    float3 throughput = make_float3(1.0f);
    float3 inferenceThroughput = make_float3(0.0f);
    float3 inferencePlus = make_float3(0.0f);
    
    for (uint depth = 0; depth < MAX_BOUNCES; depth++) {
        // Russian roulette
        // const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        // if (RND_ROULETTE >= pContinue) break;
        // throughput /= pContinue;

        if (params.inferenceMode == InferenceMode::FIRST_VERTEX ||
            params.inferenceMode == InferenceMode::RAW_CACHE) {
            if (depth > 0) break; // Only trace the first vertex
        }

        // Trace
        const auto next = trace(ray);
        payload = next; // Putting this here does not inference when there is no hit

        // Skybox
        if (isinf(next.t)) { // Use previous bounce for inference
            inferencePlus += throughput * next.emission;
            break; // Skybox
        }

        // if (params.inferenceMode == InferenceMode::SAH) {
        //     if (depth == 0) {
        //         primaryVariance = calcPrimaryVariance(pow2(next.t), abs(dot(ray.direction, next.normal))) * params.varianceTradeoff;
        //     } else {
        //         varianceHeuristic.add(pow2(next.t), prevBrdfPdf, abs(dot(ray.direction, next.normal)));
        //         if (varianceHeuristic.get() > primaryVariance) break; // Stop if variance is too high
        //     }
        // }

        isPayloadValid = true;
        wo = -ray.direction;

        // Emission
        if (luminance(payload.emission) > 0.0f) {
            auto weight = 1.0f;
            if (nee && !lightPdfIsZero) {
                // NOTE: Maybe calculating the prevBrdfPdf here only when necessary is faster
                const auto lightPdf = lightPdfUniform(wo, payload.t, payload.normal, payload.area);
                weight = powerHeuristic(prevBrdfPdf, lightPdf);
            }
            inferencePlus += throughput * payload.emission * weight;
        }

        hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;

        // Next event estimation
        if (nee) {
            const auto sample = sampleLight(RND_LSRC, RND_LSAMP, hitPoint);
            //if (abs(cosThetaS) > 0.0f && abs(sample.cosThetaL) > 0.0f) {
            const auto brdf = evalDisney(wo, sample.wi, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
            const auto surfacePoint = hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.wi, payload.normal));
            const auto lightPoint = sample.position - sample.n * copysignf(params.sceneEpsilon, dot(sample.wi, sample.n));
            if (!brdf.isDirac && brdf.pdf > 0.0f && !traceOcclusion(surfacePoint, lightPoint)) {
                const auto weight = powerHeuristic(sample.pdf, brdf.pdf);
                const auto contribution = throughput * brdf.throughput * sample.emission * weight / sample.pdf;
                inferencePlus += contribution;
            }
            //}
        }

        const auto mat = calcMaterialProperties(payload.baseColor, payload.metallic, alpha, payload.transmission);
        diffuse = mat.isDiffuse();

        inferenceThroughput = throughput;
        if (nee && !lightPdfIsZero) {
            const auto lightPdf = lightPdfUniform(wo, payload.t, payload.normal, payload.area);
            inferenceThroughput *= powerHeuristic(prevBrdfPdf, lightPdf);
        }

        if (params.inferenceMode == InferenceMode::FIRST_DIFFUSE && diffuse) {
            break; // Stop at the first diffuse surface
        }

        if (params.inferenceMode == InferenceMode::SAH || params.inferenceMode == InferenceMode::BTH) {
            if (depth == 0) {
                primaryVariance = calcPrimaryVariance(pow2(payload.t), abs(dot(wo, payload.normal))) * params.varianceTradeoff;
            } else {
                varianceHeuristic.add(pow2(payload.t), prevBrdfPdf, abs(dot(wo, payload.normal)));
                if (varianceHeuristic.get() > primaryVariance) break; // Stop if variance is too high
            }
        }

        // Sampling
        const auto sample = sampleDisney(RND_BSDF, RND_MICROFACET, RND_DIFFUSE, wo, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);

        if (params.inferenceMode == InferenceMode::BTH) {
            float pd = abs(dot(sample.direction, payload.normal)) * INV_PI; // Cosine hemisphere PDF
            float pcontinue = powerHeuristic(sample.pdf, params.K * pd);
            if (RND_ROULETTE >= pcontinue) break;
        }

        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};
        throughput *= sample.throughput;
        prevBrdfPdf = sample.pdf;
        lightPdfIsZero = sample.isDirac;
    }

    if (!isPayloadValid) {
        params.inferenceThroughput[i] = make_float3(0.0f);
    } else {
        // FIXME: Diffuse encoding too bright
        const auto nrcQuery = encodeInput(hitPoint, diffuse && (params.flags & DIFFUSE_ENCODING_FLAG), wo, payload);
        writeNRCInput(params.inferenceInput, i, nrcQuery);
        params.inferenceThroughput[i] = inferenceThroughput;
    }

    params.image[i] = mix(params.image[i], make_float4(inferencePlus, 1.0f), params.weight);
}