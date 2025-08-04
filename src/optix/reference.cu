#include <optix_device.h>
#include <cuda_runtime.h>

#include "params.cuh"
#include "payload.cuh"
#include "common.cuh"
#include "cudamath.cuh"
#include "sampling.cuh"
#include "principled_brdf.cuh"

extern "C" __global__ void __raygen__reference() {
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
    auto lightPdfIsZero = true;
    
    for (uint depth = 1; depth < params.maxPathLength; depth++) {
        // Russian roulette
        const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        if (RND_ROULETTE >= pContinue) break;
        throughput /= pContinue;

        payload = trace(ray);

        if (isinf(payload.t)) {
            color += throughput * payload.emission;
            break; // Skybox
        }

        const auto wo = -ray.direction;

        if (luminance(payload.emission) > 0.0f) {
            auto weight = 1.0f;
            if (nee && !lightPdfIsZero) {
                // NOTE: Maybe calculating the prevBrdfPdf here only when necessary is faster
                const auto lightPdf = lightPdfUniform(wo, payload.t, payload.normal, payload.area);
                weight = powerHeuristic(prevBrdfPdf, lightPdf);
            }
            color += throughput * payload.emission * weight;
        }

        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto metallic = payload.metallic;
        const auto baseColor = payload.baseColor; // baseColor

        // Next event estimation
        if (nee) {
            const auto sample = sampleLight(RND_LSRC, RND_LSAMP, hitPoint);
            //if (abs(cosThetaS) > 0.0f && abs(sample.cosThetaL) > 0.0f) {
            const auto brdf = evalDisney(wo, sample.wi, payload.normal, baseColor, metallic, alpha, payload.transmission);
            const auto surfacePoint = hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.wi, payload.normal));
            const auto lightPoint = sample.position - sample.n * copysignf(params.sceneEpsilon, dot(sample.wi, sample.n));
            if (!brdf.isDirac && brdf.pdf > 0.0f && !traceOcclusion(surfacePoint, lightPoint)) {
                const auto weight = powerHeuristic(sample.pdf, brdf.pdf);
                const auto contribution = throughput * brdf.throughput * sample.emission * weight / sample.pdf;
                color += contribution;
            }
            //}
        }

        const auto sample = sampleDisney(RND_BSDF, RND_MICROFACET, RND_DIFFUSE, wo, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
        
        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};
        throughput *= sample.throughput;
        prevBrdfPdf = sample.pdf;
        lightPdfIsZero = sample.isDirac;
    }

    params.image[i] = mix(params.image[i], make_float4(color, 1.0f), params.weight);
}