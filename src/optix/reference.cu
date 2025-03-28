#include <optix_device.h>
#include <cuda_runtime.h>

#include "params.cuh"
#include "payload.cuh"
#include "common.cuh"
#include "cudamath.cuh"
#include "sampling.cuh"

extern "C" __global__ void __raygen__reference() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;

    float pixelRnd[RANDS_PER_PIXEL];
    getPixelRands(i, pixelRnd);

    const float2 jitter = {pixelRnd[0], pixelRnd[1]};
    const auto uv = (make_float2(idx.x, idx.y) + jitter) / make_float2(dim.x, dim.y);
    auto ray = makeCameraRay(uv);

    const auto nee = params.lightTable && (params.flags & NEE_FLAG);

    Payload payload;
    auto color = make_float3(0.0f);
    auto throughput = make_float3(1.0f);
    auto prevBrdfPdf = 1.0f;
    auto diracEvent = true;
    
    for (uint depth = 1; depth < MAX_BOUNCES; depth++) {
        float bounceRnd[RANDS_PER_BOUNCE];
        getBounceRands(i, depth, bounceRnd);

        // Russian roulette
        const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        if (bounceRnd[0] >= pContinue) break;
        throughput /= pContinue;

        payload = trace(ray);

        if (isinf(payload.t)) {
            color += throughput * payload.emission;
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
        }

        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto metallic = payload.metallic;
        const auto baseColor = payload.baseColor; // baseColor

        // Next event estimation
        if (nee) {
            const auto sample = sampleLight({bounceRnd[1], bounceRnd[2], bounceRnd[3]}, hitPoint);
            const auto cosThetaS = dot(sample.wi, n);
            //if (abs(cosThetaS) > 0.0f && abs(sample.cosThetaL) > 0.0f) {
                const auto brdf = evalDisney(wo, sample.wi, n, baseColor, metallic, alpha, payload.transmission, inside);
                const auto surfacePoint = hitPoint + n * copysignf(params.sceneEpsilon, cosThetaS);
                const auto lightPoint = sample.position - sample.n * copysignf(params.sceneEpsilon, dot(sample.wi, sample.n));
                if (!brdf.isDirac && brdf.pdf > 0.0f && !traceOcclusion(surfacePoint, lightPoint)) {
                    const auto weight = balanceHeuristic(sample.pdf, brdf.pdf);
                    color += throughput * brdf.throughput * sample.emission * weight / sample.pdf;
                }
            //}
        }

        const auto sample = sampleDisney(bounceRnd[4], {bounceRnd[5], bounceRnd[6]}, {bounceRnd[5], bounceRnd[6]}, wo, n, inside, payload.baseColor, payload.metallic, alpha, payload.transmission);
        
        ray = Ray{hitPoint + n * copysignf(params.sceneEpsilon, dot(sample.direction, n)), sample.direction};
        throughput *= sample.throughput;
        prevBrdfPdf = sample.pdf;
        diracEvent = sample.isDirac;
    }

    // NOTE: We should not need to prevent NaNs
    // FIXME: NaNs
    //if (isfinite(color))
    params.image[i] = mix(params.image[i], make_float4(max(color, 0.0f), 1.0f), params.weight); // FIXME: Negative colors
}