#include <optix_device.h>
#include <cuda_runtime.h>
#include <array>

#include "params.cuh"
#include "payload.cuh"
#include "common.cuh"
#include "nrc.cuh"
#include "cudamath.cuh"
#include "sampling.cuh"

extern "C" __global__ void __raygen__() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;
    const auto rotation = params.rotationTable[i];

    const auto lightSample = sampleLight(PR1(0), PR2(1), PR2(3));
    auto ray = Ray{lightSample.position + lightSample.n * copysignf(params.sceneEpsilon, dot(lightSample.wo, lightSample.n)), lightSample.wo};
    auto radiance = lightSample.emission;
    // printf("Light sample: %f %f %f\n", lightSample.emission.x, lightSample.emission.y, lightSample.emission.z); TODO: Is this in right scale?

    Payload payload;

    for (uint depth = 1; depth <= TRAIN_DEPTH; depth++) {
        // Russian roulette
        const float pContinue = min(luminance(radiance) * params.russianRouletteWeight, 1.0f);
        if (RND_ROULETTE >= pContinue) break;
        radiance /= pContinue;

        payload = trace(ray);

        if (isinf(payload.t)) break; // Skybox

        auto n = payload.normal;
        const auto wi = -ray.direction;
        const auto inside = dot(n, wi) < 0.0f;
        n = inside ? -n : n;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;

        const auto sample = sampleDisney(RND_BSDF, RND_MICROFACET, RND_DIFFUSE, wi, n, inside, payload.baseColor, payload.metallic, alpha, payload.transmission);

        radiance *= sample.throughput;

        // printf("Sample throughput: %f\n", pow2(sample.throughput)); FIXME: throughput > 1 makes radiance explode

        ray = Ray{hitPoint + n * copysignf(params.sceneEpsilon, dot(sample.direction, n)), sample.direction};

        if (depth > 1) {
            const auto trainInput = encodeInput(hitPoint, sample.direction, payload);
            const auto trainIdx = pushNRCTrainInput(trainInput);
            const auto reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
            writeNRCOutput(params.trainingTarget, trainIdx, reflectanceFactorizationTerm * radiance);
        }

        radiance += payload.emission;
    }
}