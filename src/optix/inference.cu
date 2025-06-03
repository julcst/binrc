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

    //const auto nee = params.lightTable && (params.flags & NEE_FLAG);
    const auto nee = false;

    Payload payload;
    auto isPayloadValid = false;
    auto prevBrdfPdf = 1.0f;
    auto lightPdfIsZero = true;
    auto hitPoint = ray.origin;
    auto wo = ray.direction;
    auto diffuse = false;

    float3 throughput = make_float3(1.0f);
    float3 prevThroughput = make_float3(0.0f);
    float3 inferencePlus = make_float3(0.0f);
    
    for (uint depth = 1; depth < 2; depth++) {
        // Russian roulette
        const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        if (RND_ROULETTE >= pContinue) break;
        throughput /= pContinue;

        // Trace
        const auto next = trace(ray);

        // Skybox
        if (isinf(next.t)) { // TODO: Use previous bounce for inference
            inferencePlus += throughput * next.emission;
            break; // Skybox
        }

        payload = next;
        isPayloadValid = true;
        wo = -ray.direction;

        // Emission
        if (luminance(payload.emission) > 0.0f) {
            auto weight = 1.0f;
            if (nee && !lightPdfIsZero) {
                // NOTE: Maybe calculating the prevBrdfPdf here only when necessary is faster
                const auto lightPdf = lightPdfUniform(wo, payload.t, payload.normal, payload.area);
                weight = balanceHeuristic(prevBrdfPdf, lightPdf);
            }
            inferencePlus += throughput * payload.emission * weight;
        }

        hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;

        // Next event estimation
        if (nee) {
            const auto sample = sampleLight(RND_LSRC, RND_LSAMP, hitPoint);
            //if (payload.transmission > 0.0f || cosThetaS > 0.0f) {
                const auto brdf = evalDisney(wo, sample.wi, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
                const auto surfacePoint = hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.wi, payload.normal));
                const auto lightPoint = sample.position - sample.n * copysignf(params.sceneEpsilon, dot(sample.wi, sample.n));
                if (!brdf.isDirac && brdf.pdf > 0.0f && !traceOcclusion(surfacePoint, lightPoint)) {
                    const auto weight = balanceHeuristic(sample.pdf, brdf.pdf);
                    inferencePlus += throughput * brdf.throughput * sample.emission * weight;
                }
            //}
        }

        // Sampling
        const auto sample = sampleDisney(RND_BSDF, RND_MICROFACET, RND_DIFFUSE, wo, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
        diffuse = !sample.isSpecular;

        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};
        prevThroughput = throughput;
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
        params.inferenceThroughput[i] = prevThroughput;
    }

    params.image[i] = mix(params.image[i], make_float4(max(inferencePlus, 0.0f), 1.0f), params.weight);
}