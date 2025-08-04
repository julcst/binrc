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

constexpr uint N_RANDS = 5 + (MAX_BOUNCES) * 6;

struct PathVertex {
    float3 position; // Position of the vertex
    float3 normal; // Normal at the vertex
    float3 wi; // Incoming direction at the vertex
    MaterialProperties mat; // Material properties at the vertex
    float3 radiance; // Radiance at the vertex
};

extern "C" __global__ void __raygen__() {
    if (!params.lightTable) return; // Cannot sample light without lights
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(0, i, params.trainingRound * N_RANDS, &state);

    const auto r = curand_uniform4(&state); // Note curand generates in (0, 1] not [0, 1)
    const auto lightSample = samplePhoton(curand_uniform(&state), make_float2(r.x, r.y), make_float2(r.z, r.w));

    // radiance along path from x_0 to x_i-1 divided by the pdf of path x_0 to x_i-1
    auto radiance = lightSample.emission * INV_PI; // Divide by cos(wo) / pdf(wo) = cos(wo) / (cos(wo) / PI) = PI
    const auto lightEmission = radiance;
    radiance *= params.balanceWeight; // Balancing

    uint lightSamples = 0;
    auto ray = Ray{lightSample.position + lightSample.n * copysignf(params.sceneEpsilon, dot(lightSample.wo, lightSample.n)), lightSample.wo};
    Payload payload;

    radiance *= PI; // Multiply by cos(wo) / pdf(wo) = cos(wo) / (cos(wo) / PI) = PI

    PathVertex path[MAX_BOUNCES];

    for (uint depth = 0; depth < params.maxPathLength - 1; depth++) {
        // Russian roulette
        // if (params.flags & BACKWARD_RR_FLAG) {
        //     const float pContinue = min(luminance(radiance) * params.russianRouletteWeight, 1.0f);
        //     if (curand_uniform(&state) >= pContinue) break; // FIXME: use random numbers independent from sampling
        //     radiance /= pContinue;
        // }

        payload = trace(ray);

        if (isinf(payload.t)) break; // Skybox

        const auto wi = -ray.direction;
        const auto queryX = ray.origin + payload.t * ray.direction;
        const auto queryN = payload.normal;
        const auto alpha = payload.roughness * payload.roughness;
        const auto queryMat = calcMaterialProperties(payload);

        const auto r = curand_uniform4(&state);
        const auto sample = sampleDisney(curand_uniform(&state), {r.x, r.y}, {r.z, r.w}, wi, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
        const auto queryWo = sample.direction;
        const auto brdf = evalDisneyBRDFCosine(wi, sample.direction, payload.normal, queryMat);
        auto queryLo = make_float3(0.0f);

        // Connect the light sample to the query point
        auto connectionDir = lightSample.position - queryX;
        const float dist2 = length2(connectionDir);
        if (dist2 > params.sceneEpsilon && !traceOcclusion(lightSample.position, lightSample.n, queryX, queryN)) {
            connectionDir /= sqrtf(dist2); // Normalize direction
            const auto queryBRDF = evalDisneyBRDFCosine(queryWo, connectionDir, queryN, queryMat);
            const auto Lo = lightEmission * queryBRDF * abs(dot(connectionDir, lightSample.n)) / dist2;
            queryLo += Lo;
        }

        // Connect previous path vertices to the query point
        for (uint i = 0; i < depth; i++) {
            const auto& vertex = path[i];
            auto connectionDir = vertex.position - queryX;
            const float dist2 = length2(connectionDir);
            if (dist2 > params.sceneEpsilon && !traceOcclusion(vertex.position, vertex.normal, queryX, queryN)) {
                connectionDir /= sqrtf(dist2); // Normalize direction
                const auto queryBRDF = evalDisneyBRDFCosine(queryWo, connectionDir, queryN, queryMat);
                const auto vertexBRDF = evalDisneyBRDFCosine(vertex.wi, -connectionDir, vertex.normal, vertex.mat);
                const auto Lo = vertex.radiance * queryBRDF * vertexBRDF;
                queryLo += Lo;
            }
        }

        const auto trainInput = encodeInput(queryX, false, queryWo, payload);
        const auto reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
        const auto output = min(reflectanceFactorizationTerm * queryLo , 10.0f);
        const auto trainIdx = pushNRCTrainInput(trainInput);
        writeNRCOutput(params.trainingTarget, trainIdx, output);
        lightSamples++;

        path[depth] = {
            .position = queryX,
            .normal = queryN,
            .wi = wi,
            .mat = queryMat,
            .radiance = radiance
        };
        radiance *= sample.throughput;

        ray = Ray{queryX + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};

        //radiance += payload.emission; // NOTE: Do not train bounce emission
    }

    atomicAdd(params.lightSamples, lightSamples); // TODO: Aggregate
}