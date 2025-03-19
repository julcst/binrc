#include <optix_device.h>
#include <cuda_runtime.h>

#include "optixparams.cuh"
#include "cudamath.cuh"
#include "sampling.cuh"

struct Ray {
    float3 origin;
    float3 direction;
};

__device__ inline Ray makeCameraRay(const float2& uv) {
    const float4 origin = params.clipToWorld[3]; // = params.clipToWorld * make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    const float4 clipTarget = make_float4(-2.0f * uv + 1.0f, -1.0f, 1.0f);
    const float4 target = params.clipToWorld * clipTarget;
    const float3 origin3 = make_float3(origin) / origin.w;
    const float3 dir3 = normalize(origin3 - make_float3(target) / target.w);
    return {origin3, dir3};
}

__device__ inline float getRand(uint depth, uint offset, float rotation) {
    return fract(getRand(depth, offset) + rotation);
}

__device__ inline float2 getRand(uint depth, uint offset, float r0, float r1) {
    return fract(make_float2(getRand(depth, offset + 0) + r0, getRand(depth, offset + 1) + r1));
}

__device__ inline float3 getRand(uint depth, uint offset, float r0, float r1, float r2) {
    return fract(make_float3(getRand(depth, offset + 0) + r0, getRand(depth, offset + 1) + r1, getRand(depth, offset + 2) + r2));
}

__device__ inline void setBaseColor(const float3& value) {
    optixSetPayload_0(__float_as_uint(value.x));
    optixSetPayload_1(__float_as_uint(value.y));
    optixSetPayload_2(__float_as_uint(value.z));
}

__device__ inline void setNormal(const float3& value) {
    optixSetPayload_3(__float_as_uint(value.x));
    optixSetPayload_4(__float_as_uint(value.y));
    optixSetPayload_5(__float_as_uint(value.z));
}

__device__ inline void setTangent(const float3& value) {
    optixSetPayload_6(__float_as_uint(value.x));
    optixSetPayload_7(__float_as_uint(value.y));
    optixSetPayload_8(__float_as_uint(value.z));
}

__device__ inline void setEmission(const float3& value) {
    optixSetPayload_9(__float_as_uint(value.x));
    optixSetPayload_10(__float_as_uint(value.y));
    optixSetPayload_11(__float_as_uint(value.z));
}

__device__ inline void setRoughness(const float value) {
    optixSetPayload_12(__float_as_uint(value));
}

__device__ inline void setMetallic(const float value) {
    optixSetPayload_13(__float_as_uint(value));
}

__device__ inline void setTransmission(const float value) {
    optixSetPayload_14(__float_as_uint(value));
}

__device__ inline void setArea(const float value) {
    optixSetPayload_15(__float_as_uint(value));
}

__device__ inline void setT(const float value) {
    optixSetPayload_16(__float_as_uint(value));
}

__device__ constexpr inline Payload getPayload(const std::array<uint, 17>& values) {
    return {
        make_float3(__uint_as_float(values[0]), __uint_as_float(values[1]), __uint_as_float(values[2])),
        make_float3(__uint_as_float(values[3]), __uint_as_float(values[4]), __uint_as_float(values[5])),
        make_float3(__uint_as_float(values[6]), __uint_as_float(values[7]), __uint_as_float(values[8])),
        make_float3(__uint_as_float(values[9]), __uint_as_float(values[10]), __uint_as_float(values[11])),
        __uint_as_float(values[12]),
        __uint_as_float(values[13]),
        __uint_as_float(values[14]),
        __uint_as_float(values[15]),
        __uint_as_float(values[16]),
    };
}

__device__ inline Payload trace(const Ray& ray) {
    uint a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q;
    optixTraverse(
        params.handle,
        ray.origin, ray.direction,
        0.0f, MAX_T, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT offset, stride, miss index
        a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q // Payload
    );
    //const auto data = reinterpret_cast<HitData*>(optixGetSbtDataPointer());
    //optixReorder(data->materialID, 3); // TODO: Provide coherence hints
    optixInvoke(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q);
    return getPayload({a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q});
}

__device__ inline bool traceOcclusion(const float3& a, const float3& b) {
    const auto dir = b - a;
    optixTraverse(
        params.handle,
        a, dir,
        0.0f, 1.0f, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0 // SBT offset, stride, miss index
    );
    return optixHitObjectIsHit();
}

__device__ inline NRCInput encodeInput(const float3& position, const float3& wo, const float3& wn, const float3& diffuse, const float3& specular, float alpha) {
    return {
        .position = position * 0.1f + 0.5f, // TODO: Normalize position
        .wo = toNormSpherical(wo), // Switch to Octahedral
        .wn = toNormSpherical(wn), // Switch to Octahedral
        //.roughness = 1 - exp(-alpha),
        .roughness = alpha,
        .diffuse = diffuse,
        .specular = specular, // directional albedo FDG
    };
}

__device__ inline void pushNRCInput(float* to, const NRCInput& input) {
    to[0] = input.position.x;
    to[1] = input.position.y;
    to[2] = input.position.z;
    to[3] = input.wo.x;
    to[4] = input.wo.y;
    to[5] = input.wn.x;
    to[6] = input.wn.y;
    to[7] = input.roughness;
    to[8] = input.diffuse.x;
    to[9] = input.diffuse.y;
    to[10] = input.diffuse.z;
    to[11] = input.specular.x;
    to[12] = input.specular.y;
    to[13] = input.specular.z;
}

__device__ inline void pushNRCTrainInput(const NRCInput& input) {
    const auto i = atomicAdd(params.trainingIndexPtr, 1u);
    pushNRCInput(params.trainingInput + ((i + 1) % NRC_BATCH_SIZE) * NRC_INPUT_SIZE, input);
}

__device__ inline void pushNRCOutput(float* to, const NRCOutput& output) {
    to[0] = output.radiance.x;
    to[1] = output.radiance.y;
    to[2] = output.radiance.z;
}

extern "C" __global__ void __raygen__rg() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;
    const auto rotation = params.rotationTable[i];

    const auto jitter = fract(make_float2(getRand(0), getRand(1)) + make_float2(rotation));
    const auto uv = (make_float2(idx.x, idx.y) + jitter) / make_float2(dim.x, dim.y);
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
    if (getRand(1, 2, rotation.y) < NRC_BATCH_SIZE / float(params.dim.x * params.dim.y)) {
        trainDepth = int(getRand(1, 3, rotation.z) * 6) + 1;
    }
    auto trainTarget = NRCOutput{};
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
        if (getRand(depth, 3, rotation.z) >= pContinue) break;
        throughput /= pContinue;
        trainThroughput /= pContinue;

        payload = trace(ray);

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
#ifdef DEBUGPRINT
    printf("Weight: %.3f BRDF: %.3f Light: %.3f\n", weight, prevBrdfPdf, lightPdf);
#endif
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
            const auto inputIdx = (i % NRC_BATCH_SIZE) * NRC_INPUT_SIZE;
            pushNRCInput(params.trainingInput + inputIdx, trainInput);
            //pushNRCTrainInput(trainInput);
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
        // TODO: Dirac check
        if (nee) {
            const auto sample = sampleLight(getRand(depth, 0, rotation.w, rotation.x, rotation.y), hitPoint);
            const auto cosThetaS = dot(sample.wi, n);
            //if (abs(cosThetaS) > 0.0f && abs(sample.cosThetaL) > 0.0f) {
                const auto brdf = evalDisney(wo, sample.wi, n, baseColor, metallic, alpha, payload.transmission, inside);
                const auto surfacePoint = hitPoint + n * copysignf(params.sceneEpsilon, cosThetaS);
                const auto lightPoint = sample.position - sample.n * copysignf(params.sceneEpsilon, dot(sample.wi, sample.n));
                if (!brdf.isDirac && brdf.pdf > 0.0f && !traceOcclusion(surfacePoint, lightPoint)) {
                    const auto weight = balanceHeuristic(sample.pdf, brdf.pdf);
                    color += throughput * brdf.throughput * sample.emission * weight / sample.pdf;
                    trainTarget.radiance += trainThroughput * brdf.throughput * sample.emission * weight / sample.pdf;
#ifdef DEBUGPRINT
    if (getRand(depth, 0, rotation.y) < 0.001f) printf("\t\t\t\t\t\tNEE We: %.3f BRDF: %.3f Light: %.3f\n", weight, brdf.pdf, sample.pdf);
#endif
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

        // TODO: Move sampling into closesthit to benefit from reordering
        const auto sample = sampleDisney(getRand(depth, 0, rotation.w), getRand(depth, 1, rotation.x, rotation.y), getRand(depth, 1, rotation.z, rotation.w), wo, n, inside, baseColor, metallic, alpha, payload.transmission);
        
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
        const auto outputIdx = (i % NRC_BATCH_SIZE) * NRC_OUTPUT_SIZE;
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

extern "C" __global__ void __closesthit__ch() {
    // Get optix built-in variables
    const auto bary2 = optixGetTriangleBarycentrics();
    const auto bary = make_float3(1.0f - bary2.x - bary2.y, bary2);
    const auto data = reinterpret_cast<HitData*>(optixGetSbtDataPointer());

    // Get triangle vertices
    const auto idx = data->indexBuffer[optixGetPrimitiveIndex()];
    const auto v0 = data->vertexData[idx.x];
    const auto v1 = data->vertexData[idx.y];
    const auto v2 = data->vertexData[idx.z];

    const auto e0 = v1.position - v0.position;
    const auto e1 = v2.position - v0.position;
    const auto area = 0.5f * length(optixTransformVectorFromObjectToWorldSpace(cross(e0, e1)));

    // Interpolate normal
    const auto objectSpaceNormal = bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal;

    // Interpolate tangent
    const auto objectSpaceTangentWithOrientation = bary.x * v0.tangent + bary.y * v1.tangent + bary.z * v2.tangent;
    const auto objectSpaceTangent = make_float3(objectSpaceTangentWithOrientation);
    const auto worldSpaceTangent = optixTransformVectorFromObjectToWorldSpace(objectSpaceTangent);

    const auto texCoord = bary.x * v0.texCoord + bary.y * v1.texCoord + bary.z * v2.texCoord;

    // Get material
    const auto material = params.materials[data->materialID];
    
    auto baseColor = material.baseColor;
    if (material.baseMap) baseColor *= make_float3(tex2D<float4>(material.baseMap, texCoord.x, texCoord.y));

    auto mr = make_float2(material.metallic, material.roughness);
    if (material.mrMap) mr *= make_float2(tex2D<float4>(material.mrMap, texCoord.x, texCoord.y));

    // NOTE: Normal mapping produces artifacts with pathtracing: See Microfacet-based Normal Mapping for Robust Monte Carlo Path Tracing by Schüssler et al. 2017 for a solution
    if (material.normalMap) { // MikkTSpace normal mapping
        const auto tangentOrientation = objectSpaceTangentWithOrientation.w;
        const auto tangentSpaceNormal = make_float3(tex2D<float4>(material.normalMap, texCoord.x, texCoord.y)) * 2.0f - 1.0f;
        const auto objectSpaceBitangent = cross(objectSpaceNormal, objectSpaceTangent) * tangentOrientation;
        setNormal(normalize(optixTransformNormalFromObjectToWorldSpace(tangentSpaceNormal.x * objectSpaceTangent + tangentSpaceNormal.y * objectSpaceBitangent + tangentSpaceNormal.z * objectSpaceNormal)));
    } else {
        setNormal(normalize(optixTransformNormalFromObjectToWorldSpace(objectSpaceNormal)));
    }
    
    setBaseColor(baseColor);
    setTangent(worldSpaceTangent);
    setMetallic(mr.x);
    setEmission(material.emission);
    setRoughness(mr.y);
    setTransmission(params.flags & TRANSMISSION_FLAG ? material.transmission : 0.0f);
    setArea(area);
    setT(optixGetRayTmax());
}

extern "C" __global__ void __miss__ms() {
    const auto dir = optixGetWorldRayDirection();
    auto sky = make_float3(0.03f);

    setEmission(sky);
    setT(INFINITY);
}