#include <optix_device.h>
#include <cuda_runtime.h>

#include "optixparams.cuh"
#include "cudamath.cuh"
#include "brdf.cuh"

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
    float3 albedo; // Linear RGB base color
    float3 normal; // World space normal, guaranteed to be normalized
    float3 tangent; // World space tenagent, not normalized
    float3 emission; // Linear RGB emission color
    float roughness;
    float metallic;
    float transmission;
    float t; // Distance of intersection on ray, set to INFINITY if no intersection
};

__device__ void setAlbedo(const float3& value) {
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

__device__ void setT(const float value) {
    optixSetPayload_15(__float_as_uint(value));
}

__device__ Payload getPayload(uint a, uint b, uint c, uint d, uint e, uint f, uint g, uint h, uint i, uint j, uint k, uint l, uint m, uint n, uint o, uint p) {
    return Payload{
        make_float3(__uint_as_float(a), __uint_as_float(b), __uint_as_float(c)),
        make_float3(__uint_as_float(d), __uint_as_float(e), __uint_as_float(f)),
        make_float3(__uint_as_float(g), __uint_as_float(h), __uint_as_float(i)),
        make_float3(__uint_as_float(j), __uint_as_float(k), __uint_as_float(l)),
        __uint_as_float(m),
        __uint_as_float(n),
        __uint_as_float(o),
        __uint_as_float(p),
    };
}

__device__ Payload trace(const Ray& ray) {
    uint a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;
    optixTraverse(
        params.handle,
        ray.origin, ray.direction,
        0.0f, MAX_T, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT offset, stride, miss index
        a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p // Payload
    );
    //const auto data = reinterpret_cast<HitData*>(optixGetSbtDataPointer());
    //optixReorder(data->materialID, 3); // TODO: Provide coherence hints
    optixInvoke(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    return getPayload(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
}

__device__ bool traceOcclusion(const Ray& ray, float dist) {
    optixTraverse(
        params.handle,
        ray.origin, ray.direction,
        0.0f, dist - 1e-4f, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0 // SBT offset, stride, miss index
    );
    return optixHitObjectIsHit();
}

extern "C" __global__ void __raygen__rg() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;
    const auto rotation = params.rotationTable[i];

    const auto jitter = fract(make_float2(getRand(0), getRand(1)) + make_float2(rotation));
    const auto uv = (make_float2(idx.x, idx.y) + jitter) / make_float2(dim.x, dim.y);
    auto ray = makeCameraRay(uv);

    Payload payload;
    auto color = make_float3(0.0f);
    auto throughput = make_float3(1.0f);
    auto isGlossy = false;
    
    for (uint depth = 1; depth < MAX_BOUNCES; depth++) {
        payload = trace(ray);

        const auto nee = !isinf(payload.t) && depth > 1 && params.lightTable;
        
        if (!nee || isGlossy) color += throughput * payload.emission; // FIXME: * dot(payload.normal, -ray.direction); ???

        if (isinf(payload.t)) break; // Skybox

        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto metallic = payload.metallic;
        const auto albedo = payload.albedo;

        auto n = payload.normal;
        const auto wo = -ray.direction;
        const auto inside = dot(n, wo) < 0.0f;
        n = inside ? -n : n;
        const auto cosThetaO = dot(wo, n);
        const auto baseSpecular = mix(make_float3(0.04f), albedo, metallic);
        const auto baseDiffuse = (1.0f - metallic) * albedo;

        // Importance sampling weights // TODO: Use precomputed
        const auto wSpecular = luminance(F_SchlickApprox(cosThetaO, baseSpecular));
        const auto wDiffuse = luminance(baseDiffuse);
        const auto pSpecular = wSpecular / (wSpecular + wDiffuse);

        // TODO: Move sampling into closesthit to benefit from reordering
        if (getRand(depth, 0, rotation.w) < pSpecular) { 
            // Sample Trowbridge-Reitz specular
            isGlossy = true;
            const auto rand = getRand(depth, 1, rotation.x, rotation.y);
            const auto sample = sampleTrowbridgeReitz(rand, wo, cosThetaO, n, alpha, baseSpecular);
            ray.direction = sample.direction;
            ray.origin = hitPoint + 1e-4f * n; // Prevent self intersection
            throughput *= sample.throughput / pSpecular;
        } else {
            // TODO: Proper weighting
            if (payload.transmission < 0.5f) { // Sample Brent-Burley diffuse
                isGlossy = false;
                const auto rand = getRand(depth, 1, rotation.z, rotation.w); 
                const auto tangentToWorld = buildTBN(n, payload.tangent);
                const auto sample = sampleBrentBurley(rand, wo, cosThetaO, n, alpha, tangentToWorld, baseDiffuse);
                ray.direction = sample.direction;
                ray.origin = hitPoint + 1e-4f * n; // Prevent self intersection
                throughput *= sample.throughput / (1.0f - pSpecular);
            } else {
                isGlossy = true;
                const auto rand = getRand(depth, 1, rotation.x, rotation.y);
                const auto sample = sampleTrowbridgeReitzTransmission(rand, wo, cosThetaO, n, alpha, baseSpecular, albedo, inside);
                ray.direction = sample.direction;
                ray.origin = hitPoint + 1e-4f * ray.direction; // Prevent self intersection
                throughput *= sample.throughput / (1.0f - pSpecular);
            }
        }

        // Russian roulette
        const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        if (fract(getRand(depth, 3) + rotation.z) >= pContinue) break;
        throughput /= pContinue;

        // Next event estimation
        // TODO: MIS
        if (params.lightTable) {
            const auto sample = sampleLight(getRand(depth, 0, rotation.w, rotation.x, rotation.y), hitPoint);
            const auto cosThetaS = dot(sample.wi, n);
            const auto shadowRay = Ray{hitPoint + 1e-4f * ray.direction, sample.wi};
            if (cosThetaS > 0.0f && sample.cosThetaL > 0.0f && !traceOcclusion(shadowRay, sample.dist)) {
                const auto brdf = disneyBRDF(wo, sample.wi, n, albedo, metallic, alpha);
                color += throughput * brdf * cosThetaS * INV_PI * sample.emission / sample.pdf;
            }
        }
    }

    // NOTE: We should not need to prevent NaNs
    // FIXME: NaNs
    // if (isfinite(throughput))
    params.image[i] = mix(params.image[i], make_float4(color, 1.0f), params.weight);
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

    // Interpolate normal
    const auto objectSpaceNormal = bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal;

    // Interpolate tangent
    const auto objectSpaceTangentWithOrientation = bary.x * v0.tangent + bary.y * v1.tangent + bary.z * v2.tangent;
    const auto objectSpaceTangent = make_float3(objectSpaceTangentWithOrientation);
    const auto worldSpaceTangent = optixTransformVectorFromObjectToWorldSpace(objectSpaceTangent);

    const auto texCoord = bary.x * v0.texCoord + bary.y * v1.texCoord + bary.z * v2.texCoord;

    // Get material
    const auto material = params.materials[data->materialID];
    
    auto albedo = material.albedo;
    if (material.baseMap) albedo *= make_float3(tex2D<float4>(material.baseMap, texCoord.x, texCoord.y));

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
    
    setAlbedo(albedo);
    setTangent(worldSpaceTangent);
    setMetallic(mr.x);
    setEmission(material.emission);
    setRoughness(mr.y);
    setTransmission(material.transmission);
    setT(optixGetRayTmax());
}

extern "C" __global__ void __miss__ms() {
    const auto dir = optixGetWorldRayDirection();
    auto sky = make_float3(0.05f);

    setEmission(sky);
    setT(INFINITY);
}