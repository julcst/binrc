#include <optix_device.h>
#include <cuda_runtime.h>

#include "optixparams.cuh"
#include "cudamath.cuh"
#include "sampling.cuh"

struct Ray {
    float3 origin;
    float3 direction;
};

__device__ Ray makeCameraRay(const float2& uv) {
    const float4 origin = params.clipToWorld[3]; // = params.clipToWorld * make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    const float4 clipTarget = make_float4(-2.0f * uv + 1.0f, -1.0f, 1.0f);
    const float4 target = params.clipToWorld * clipTarget;
    const float3 origin3 = make_float3(origin) / origin.w;
    const float3 dir3 = normalize(origin3 - make_float3(target) / target.w);
    return Ray{origin3, dir3};
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

struct Payload {
    float3 baseColor; // Linear RGB base color
    float3 normal; // World space normal, guaranteed to be normalized
    float3 tangent; // World space tenagent, not normalized
    float3 emission; // Linear RGB emission color
    float roughness;
    float metallic;
    float transmission;
    float area;
    float t; // Distance of intersection on ray, set to INFINITY if no intersection
};

__device__ void setBaseColor(const float3& value) {
    optixSetPayload_0(__float_as_uint(value.x));
    optixSetPayload_1(__float_as_uint(value.y));
    optixSetPayload_2(__float_as_uint(value.z));
}

__device__ void setNormal(const float3& value) {
    optixSetPayload_3(__float_as_uint(value.x));
    optixSetPayload_4(__float_as_uint(value.y));
    optixSetPayload_5(__float_as_uint(value.z));
}

__device__ void setTangent(const float3& value) {
    optixSetPayload_6(__float_as_uint(value.x));
    optixSetPayload_7(__float_as_uint(value.y));
    optixSetPayload_8(__float_as_uint(value.z));
}

__device__ void setEmission(const float3& value) {
    optixSetPayload_9(__float_as_uint(value.x));
    optixSetPayload_10(__float_as_uint(value.y));
    optixSetPayload_11(__float_as_uint(value.z));
}

__device__ void setRoughness(const float value) {
    optixSetPayload_12(__float_as_uint(value));
}

__device__ void setMetallic(const float value) {
    optixSetPayload_13(__float_as_uint(value));
}

__device__ void setTransmission(const float value) {
    optixSetPayload_14(__float_as_uint(value));
}

__device__ void setArea(const float value) {
    optixSetPayload_15(__float_as_uint(value));
}

__device__ void setT(const float value) {
    optixSetPayload_16(__float_as_uint(value));
}

__device__ Payload getPayload(uint a, uint b, uint c, uint d, uint e, uint f, uint g, uint h, uint i, uint j, uint k, uint l, uint m, uint n, uint o, uint p, uint q) {
    return Payload{
        make_float3(__uint_as_float(a), __uint_as_float(b), __uint_as_float(c)),
        make_float3(__uint_as_float(d), __uint_as_float(e), __uint_as_float(f)),
        make_float3(__uint_as_float(g), __uint_as_float(h), __uint_as_float(i)),
        make_float3(__uint_as_float(j), __uint_as_float(k), __uint_as_float(l)),
        __uint_as_float(m),
        __uint_as_float(n),
        __uint_as_float(o),
        __uint_as_float(p),
        __uint_as_float(q),
    };
}

__device__ Payload trace(const Ray& ray) {
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
    return getPayload(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q);
}

__device__ bool traceOcclusion(const float3& a, const float3& b) {
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

__device__ void pushTrainingSample(float3 pos, float3 color) {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = (idx.y * params.dim.x + idx.x) % NRC_BATCH_SIZE;
    const auto inputIdx = i * NRC_INPUT_SIZE;
    const auto outputIdx = i * NRC_OUTPUT_SIZE;
    params.trainingInput[inputIdx + 0] = pos.x;
    params.trainingInput[inputIdx + 1] = pos.y;
    params.trainingInput[inputIdx + 2] = pos.z;
    params.trainingTarget[outputIdx + 0] = color.x;
    params.trainingTarget[outputIdx + 1] = color.y;
    params.trainingTarget[outputIdx + 2] = color.z;
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
    if (getRand(1, 2, rotation.y) < NRC_BATCH_SIZE / float(params.dim.x * params.dim.y)) {
        trainDepth = int(getRand(1, 3, rotation.z) * 6) + 1;
    }
    auto trainInput = NRCInput{};
    auto trainTarget = NRCOutput{};
    auto trainThroughput = make_float3(1.0f);

    auto nrcQuery = NRCInput{};
    
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

        // NRC Inference Input
        if (depth == 1) {
            const auto F0 = mix(make_float3(0.04f), baseColor, metallic);
            const auto albedo = (1.0f - metallic) * baseColor;
            nrcQuery.position = hitPoint;
            nrcQuery.wo = toNormSpherical(wo);
            nrcQuery.wn = toNormSpherical(n);
            nrcQuery.roughness = 1 - exp(-alpha);
            nrcQuery.diffuse = albedo;
            nrcQuery.specular = F0;
        }

        if (depth == trainDepth) {
            const auto F0 = mix(make_float3(0.04f), baseColor, metallic);
            const auto albedo = (1.0f - metallic) * baseColor;
            trainInput.position = hitPoint;
            trainInput.wo = toNormSpherical(wo);
            trainInput.wn = toNormSpherical(n);
            trainInput.roughness = 1 - exp(-alpha);
            trainInput.diffuse = albedo;
            trainInput.specular = F0;
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

        // TODO: Move sampling into closesthit to benefit from reordering
        const auto sample = sampleDisney(getRand(depth, 0, rotation.w), getRand(depth, 1, rotation.x, rotation.y), getRand(depth, 1, rotation.z, rotation.w), wo, n, inside, baseColor, metallic, alpha, payload.transmission);
        
        ray = Ray{hitPoint + n * copysignf(params.sceneEpsilon, dot(sample.direction, n)), sample.direction};
        throughput *= sample.throughput;
        trainThroughput *= sample.throughput;
        prevBrdfPdf = sample.pdf;
        diracEvent = sample.isDirac;
    }

    trainTarget.radiance = trainTarget.radiance / max(trainInput.diffuse + trainInput.specular, 1e-3f);
    
    // NRC Training
    if (isfinite(trainInput.position)
        && isfinite(trainInput.wo)
        && isfinite(trainInput.wn)
        && isfinite(trainInput.roughness)
        && isfinite(trainInput.diffuse)
        && isfinite(trainInput.specular)
        && isfinite(trainTarget.radiance)
        && getRand(1, 2, rotation.y) < NRC_BATCH_SIZE / float(params.dim.x * params.dim.y)) {
        const auto trainIdx = i % NRC_BATCH_SIZE;
        const auto inputIdx = trainIdx * NRC_INPUT_SIZE;
        const auto outputIdx = trainIdx * NRC_OUTPUT_SIZE;
        params.trainingInput[inputIdx + 0] = trainInput.position.x;
        params.trainingInput[inputIdx + 1] = trainInput.position.y;
        params.trainingInput[inputIdx + 2] = trainInput.position.z;
        params.trainingInput[inputIdx + 3] = trainInput.wo.x;
        params.trainingInput[inputIdx + 4] = trainInput.wo.y;
        params.trainingInput[inputIdx + 5] = trainInput.wn.x;
        params.trainingInput[inputIdx + 6] = trainInput.wn.y;
        params.trainingInput[inputIdx + 7] = trainInput.roughness;
        params.trainingInput[inputIdx + 8] = trainInput.diffuse.x;
        params.trainingInput[inputIdx + 9] = trainInput.diffuse.y;
        params.trainingInput[inputIdx + 10] = trainInput.diffuse.z;
        params.trainingInput[inputIdx + 11] = trainInput.specular.x;
        params.trainingInput[inputIdx + 12] = trainInput.specular.y;
        params.trainingInput[inputIdx + 13] = trainInput.specular.z;
        params.trainingTarget[outputIdx + 0] = trainTarget.radiance.x;
        params.trainingTarget[outputIdx + 1] = trainTarget.radiance.y;
        params.trainingTarget[outputIdx + 2] = trainTarget.radiance.z;
    }

    // NRC Inference
    const auto inputIdx = i * NRC_INPUT_SIZE;
    params.inferenceInput[inputIdx + 0] = nrcQuery.position.x;
    params.inferenceInput[inputIdx + 1] = nrcQuery.position.y;
    params.inferenceInput[inputIdx + 2] = nrcQuery.position.z;
    params.inferenceInput[inputIdx + 3] = nrcQuery.wo.x;
    params.inferenceInput[inputIdx + 4] = nrcQuery.wo.y;
    params.inferenceInput[inputIdx + 5] = nrcQuery.wn.x;
    params.inferenceInput[inputIdx + 6] = nrcQuery.wn.y;
    params.inferenceInput[inputIdx + 7] = nrcQuery.roughness;
    params.inferenceInput[inputIdx + 8] = nrcQuery.diffuse.x;
    params.inferenceInput[inputIdx + 9] = nrcQuery.diffuse.y;
    params.inferenceInput[inputIdx + 10] = nrcQuery.diffuse.z;
    params.inferenceInput[inputIdx + 11] = nrcQuery.specular.x;
    params.inferenceInput[inputIdx + 12] = nrcQuery.specular.y;
    params.inferenceInput[inputIdx + 13] = nrcQuery.specular.z;

    // NOTE: We should not need to prevent NaNs
    // FIXME: NaNs
    //if (isfinite(color))
    if (!(NRC_INFERENCE_FLAG & params.flags))
    params.image[i] = mix(params.image[i], make_float4(max(color, 0.0f), 1.0f), params.weight); // FIXME: Negative colors
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
    auto sky = make_float3(0.0f);

    setEmission(sky);
    setT(INFINITY);
}