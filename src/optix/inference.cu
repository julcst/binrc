#include <optix_device.h>
#include <cuda_runtime.h>

#include "params.cuh"
#include "payload.cuh"
#include "common.cuh"
#include "cudamath.cuh"
#include "sampling.cuh"

extern "C" __global__ void __raygen__inference() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;
    const auto rotation = params.rotationTable[i];

    const auto jitter = fract(make_float2(getRand(0), getRand(1)) + make_float2(rotation));
    const auto uv = (make_float2(idx.x, idx.y) + jitter) / make_float2(dim.x, dim.y);
    auto ray = makeCameraRay(uv);

    const auto nee = params.lightTable && (params.flags & NEE_FLAG);

    Payload payload;
    auto prevBrdfPdf = 1.0f;
    auto diracEvent = true;
    auto hitPoint = ray.origin;
    auto n = make_float3(0.0f);
    auto wo = ray.direction;

    float3 throughput = make_float3(1.0f);
    float3 inferencePlus = make_float3(0.0f);
    float3 inferenceThroughput = make_float3(1.0f);
    
    for (uint depth = 1; depth < MAX_BOUNCES; depth++) {

        // Russian roulette
        const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        if (getRand(depth, 3, rotation.z) >= pContinue) break;
        throughput /= pContinue;

        // Trace
        const auto next = trace(ray);

        // Skybox
        if (isinf(next.t)) { // TODO: Use previous bounce for inference
            inferencePlus += throughput * next.emission;
            break; // Skybox
        }

        payload = next;
        inferenceThroughput = throughput;
        n = payload.normal;
        wo = -ray.direction;
        const auto inside = dot(n, wo) < 0.0f;
        n = inside ? -n : n;

        // Emission
        if (luminance(payload.emission) > 0.0f) {
            auto weight = 1.0f;
            if (nee && !diracEvent) {
                // NOTE: Maybe calculating the prevBrdfPdf here only when necessary is faster
                const auto lightPdf = lightPdfUniform(wo, payload.t, n, payload.area);
                weight = balanceHeuristic(prevBrdfPdf, lightPdf);
            }
            inferencePlus += throughput * payload.emission * weight;
        }

        hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;

        // Next event estimation
        if (nee) {
            const auto sample = sampleLight(getRand(depth, 0, rotation.w, rotation.x, rotation.y), hitPoint);
            const auto cosThetaS = dot(sample.wi, n);
            //if (payload.transmission > 0.0f || cosThetaS > 0.0f) {
                const auto brdf = evalDisney(wo, sample.wi, n, payload.baseColor, payload.metallic, alpha, payload.transmission, inside);
                const auto surfacePoint = hitPoint + n * copysignf(params.sceneEpsilon, cosThetaS);
                const auto lightPoint = sample.position - sample.n * copysignf(params.sceneEpsilon, dot(sample.wi, sample.n));
                if (!brdf.isDirac && brdf.pdf > 0.0f && !traceOcclusion(surfacePoint, lightPoint)) {
                    const auto weight = balanceHeuristic(sample.pdf, brdf.pdf);
                    inferencePlus += throughput * brdf.throughput * sample.emission * weight / sample.pdf;
                }
            //}
        }

        // Sampling
        const auto sample = sampleDisney(getRand(depth, 0, rotation.w), getRand(depth, 1, rotation.x, rotation.y), getRand(depth, 1, rotation.z, rotation.w), wo, n, inside, payload.baseColor, payload.metallic, alpha, payload.transmission);
        
        ray = Ray{hitPoint + n * copysignf(params.sceneEpsilon, dot(sample.direction, n)), sample.direction};
        throughput *= sample.throughput;
        prevBrdfPdf = sample.pdf;
        diracEvent = sample.isDirac;
    }

    payload = trace(ray);
    const auto inputIdx = i * NRC_INPUT_SIZE;
    const auto nrcQuery = encodeInput(hitPoint, wo, payload);
    pushNRCInput(params.inferenceInput + inputIdx, nrcQuery);
    params.inferenceThroughput[i] = throughput;

    // NOTE: We should not need to prevent NaNs
    // FIXME: NaNs
    //if (isfinite(color))
    params.image[i] = mix(params.image[i], make_float4(max(inferencePlus, 0.0f), 1.0f), params.weight); // FIXME: Negative colors
}