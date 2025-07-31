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

constexpr uint N_RANDS = 5 + (TRAIN_DEPTH + 1) * 6;

extern "C" __global__ void __raygen__() {
    if (!params.lightTable) return; // Cannot sample light without lights
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(0, i, params.trainingRound * N_RANDS, &state);

    auto queryRay = makeCameraRay({curand_uniform(&state), curand_uniform(&state)});
    Payload queryPayload = trace(queryRay);
    if (isinf(queryPayload.t)) return; // Nothing to train
    float3 queryX = queryRay.origin + queryPayload.t * queryRay.direction;
    float3 queryWo = -queryRay.direction;
    float3 queryLo = {0.0f, 0.0f, 0.0f};
    //printf("Query payload albedo metallic roughness transmission: %.2f %.2f %.2f %.2f\n", length2(queryPayload.baseColor), queryPayload.metallic, queryPayload.roughness, queryPayload.transmission);
    auto queryMat = calcMaterialProperties(queryPayload);
    if(queryMat.alpha2 < 1e-2f) printf("Query material F0 %.2f albedo %.2f alpha2 %.2f\n", length2(queryMat.F0), length2(queryMat.albedo), queryMat.alpha2);

    const auto r = curand_uniform4(&state); // Note curand generates in (0, 1] not [0, 1)
    const auto lightSample = samplePhoton(curand_uniform(&state), make_float2(r.x, r.y), make_float2(r.z, r.w));

    // radiance along path from x_0 to x_i-1 divided by the pdf of path x_0 to x_i-1
    auto radiance = params.flags & LIGHT_TRACE_FIX_FLAG ? lightSample.emission : lightSample.emission * INV_PI; // FIXME: Why / PI² ?
    radiance *= params.balanceWeight; // Balancing

    // Probability of sampling the incoming light direction
    float p_wi = abs(dot(lightSample.wo, lightSample.n)) * INV_PI;
    //radiance *= p_wi; // FIXME: Integrate in sampling

    uint lightSamples = 0;
    auto ray = Ray{lightSample.position + lightSample.n * copysignf(params.sceneEpsilon, dot(lightSample.wo, lightSample.n)), lightSample.wo};
    Payload payload;

    // Connect the light sample to the query point
    auto connectionDir = lightSample.position - queryX;
    const float dist2 = length2(connectionDir);
    if (dist2 > params.sceneEpsilon && !traceOcclusion(lightSample.position, lightSample.n, queryX, queryPayload.normal)) {
        connectionDir /= sqrtf(dist2); // Normalize direction
        const auto queryBRDF = evalDisneyBRDFCosine(queryWo, connectionDir, queryPayload.normal, queryMat);
        const auto Lo = radiance * queryBRDF * abs(dot(connectionDir, lightSample.n)) / dist2;
        queryLo += Lo;
    }

    radiance *= PI;

    for (uint depth = 0; depth < 8; depth++) {
        // Russian roulette
        // if (params.flags & BACKWARD_RR_FLAG) {
        //     const float pContinue = min(luminance(radiance) * params.russianRouletteWeight, 1.0f);
        //     if (curand_uniform(&state) >= pContinue) break; // FIXME: use random numbers independent from sampling
        //     radiance /= pContinue;
        // }

        // auto connectionDir = hitPoint - queryX;
        // const float dist2 = length2(connectionDir);
        // connectionDir /= sqrtf(dist2); // Normalize direction

        // const auto queryBRDF = evalDisneyBRDFCosine(queryWo, connectionDir, queryPayload.normal, queryMat);
        // //const auto queryBRDF = abs(dot(queryWo, queryPayload.normal)) * INV_PI; // Lambertian BRDF
        // const auto Lo = radiance * queryBRDF * abs(dot(connectionDir, prevNormal)) / max(dist2, 1e-6f);
        // //if (length2(Lo) < 1e-2f) printf("Lo is zero radiance %.2f brdf %.2f cosO %.2f cosI %.2f dist2 %.2f F0 %.2f albedo %.2f alpha2 %.2f\n", length2(radiance), length2(queryBRDF), dot(queryWo, queryPayload.normal), dot(connectionDir, queryPayload.normal), dist2, length2(queryMat.F0), length2(queryMat.albedo), queryMat.alpha2);
        // //else printf("Lo: %.2f\n", length2(Lo));
        // //const auto Lo = radiance * brdf;
        // queryLo += Lo;

        payload = trace(ray);

        if (isinf(payload.t)) break; // Skybox

        const auto wi = -ray.direction;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto mat = calcMaterialProperties(payload);

        auto connectionDir = hitPoint - queryX;
        const float dist2 = length2(connectionDir);
        if (dist2 > params.sceneEpsilon && !traceOcclusion(hitPoint, payload.normal, queryX, queryPayload.normal)) {
            connectionDir /= sqrtf(dist2); // Normalize direction
            const auto queryBRDF = evalDisneyBRDFCosine(queryWo, connectionDir, queryPayload.normal, queryMat);
            //const auto hitBRDF = evalDisneyBRDFCosine(wi, -connectionDir, payload.normal, mat) * PI; // Almost correct
            const auto hitBRDF = evalDisneyBRDFCosine(wi, -connectionDir, payload.normal, mat);
            //const auto hitBRDF = evalDisneyWeighted(wi, -connectionDir, payload.normal, mat).throughput;
            const auto Lo = radiance * queryBRDF * hitBRDF / dist2;
            queryLo += Lo;
        }

        const auto r = curand_uniform4(&state);
        const auto sample = sampleDisney(curand_uniform(&state), {r.x, r.y}, {r.z, r.w}, wi, payload.normal, payload.baseColor, payload.metallic, alpha, payload.transmission);
        const auto brdf = evalDisneyBRDFCosine(wi, sample.direction, payload.normal, mat);
        //radiance *= brdf / p_wi;
        radiance *= sample.throughput;
        p_wi = sample.pdf;

        // printf("Sample throughput: %f\n", length2(sample.throughput));

        ray = Ray{hitPoint + payload.normal * copysignf(params.sceneEpsilon, dot(sample.direction, payload.normal)), sample.direction};

        //radiance += payload.emission; // TODO: Do not train bounce emission
    }

    const auto trainInput = encodeInput(queryX, false, queryWo, queryPayload);
    const auto reflectanceFactorizationTerm = 1.0f / max(trainInput.diffuse + trainInput.specular, 1e-3f);
    const auto output = min(reflectanceFactorizationTerm * queryLo , 10.0f);
    const auto trainIdx = pushNRCTrainInput(trainInput);
    writeNRCOutput(params.trainingTarget, trainIdx, output);
    //printf("Training sample %d: pos=(%.2f, %.2f, %.2f), wo=(%.2f, %.2f, %.2f), Lo=(%.2f, %.2f, %.2f)\n", trainIdx, queryX.x, queryX.y, queryX.z, queryWo.x, queryWo.y, queryWo.z, output.x, output.y, output.z);
    lightSamples++;

    atomicAdd(params.lightSamples, lightSamples); // TODO: Aggregate
}