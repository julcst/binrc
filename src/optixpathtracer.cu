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
    const float4 origin = params.clipToWorld[3]; // = params.clipToWorld[3] * make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    const float4 clipTarget = make_float4(-2.0f * uv + 1.0f, -1.0f, 1.0f);
    const float4 target = params.clipToWorld * clipTarget;
    const float3 origin3 = make_float3(origin) / origin.w;
    const float3 dir3 = normalize(origin3 - make_float3(target) / target.w);
    return Ray{origin3, dir3};
}

struct Payload {
    float3 color; // Linear RGB base color
    float3 normal; // World space normal, guaranteed to be normalized
    float3 tangent; // World space tenagent, not normalized
    float roughness;
    float metallic;
    float t; // Distonce of intersection on ray, set to INFINITY if no intersection
};

__device__ void setColor(const float3& value) {
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

__device__ void setRoughness(const float value) {
    optixSetPayload_9(__float_as_uint(value));
}

__device__ void setMetallic(const float value) {
    optixSetPayload_10(__float_as_uint(value));
}

__device__ void setT(const float value) {
    optixSetPayload_11(__float_as_uint(value));
}

__device__ Payload getPayload(uint a, uint b, uint c, uint d, uint e, uint f, uint g, uint h, uint i, uint j, uint k, uint l) {
    return Payload{
        make_float3(__uint_as_float(a), __uint_as_float(b), __uint_as_float(c)),
        make_float3(__uint_as_float(d), __uint_as_float(e), __uint_as_float(f)),
        make_float3(__uint_as_float(g), __uint_as_float(h), __uint_as_float(i)),
        __uint_as_float(j),
        __uint_as_float(k),
        __uint_as_float(l)
    };
}

__device__ Payload trace(const Ray& ray) {
    uint a, b, c, d, e, f, g, h, i, j, k, l;
    optixTraverse(
        params.handle,
        ray.origin, ray.direction,
        0.0f, MAX_T, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT offset, stride, miss index
        a, b, c, d, e, f, g, h, i, j, k, l // Payload
    );
    //optixReorder(); // TODO: Provide coherence hints
    optixInvoke(a, b, c, d, e, f, g, h, i, j, k, l);
    return getPayload(a, b, c, d, e, f, g, h, i, j, k, l);
}

__device__ float luminance(const float3& linearRGB) {
    return dot(make_float3(0.2126f, 0.7152f, 0.0722f), linearRGB);
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
    auto throughput = make_float3(1.0f);
    for (uint depth = 0; depth < MAX_BOUNCES; depth++) {
        payload = trace(ray);

        if (isinf(payload.t)) {
            throughput *= payload.color;
            break;
        }

        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto metallic = payload.metallic;
        const auto albedo = payload.color;

        const auto n = payload.normal;
        const auto wo = -ray.direction;
        const auto cosThetaO = dot(wo, n);
        const auto baseSpecular = mix(make_float3(0.04f), albedo, metallic);
        const auto baseDiffuse = (1.0f - metallic) * albedo;

        // Importance sampling weights // TODO: Use precomputed
        const auto wSpecular = luminance(F_SchlickApprox(cosThetaO, baseSpecular));
        const auto wDiffuse = luminance(baseDiffuse);
        const auto pSpecular = wSpecular / (wSpecular + wDiffuse);

        if (fract(getRand(depth, 0) + rotation.w) < pSpecular) { 
            // Sample Trowbridge-Reitz specular
            const auto rand = fract(make_float2(getRand(depth, 1) + rotation.x, getRand(depth, 2) + rotation.y));
            const auto sample = sampleTrowbridgeReitz(rand, wo, cosThetaO, n, alpha, baseSpecular);
            ray.direction = sample.direction;
            throughput *= sample.throughput / pSpecular;
        } else {
            // Sample Brent-Burley diffuse
            const auto rand = fract(make_float2(getRand(depth, 1) + rotation.z, getRand(depth, 2) + rotation.w)); 
            const auto tangentToWorld = buildTBN(n, payload.tangent);
            const auto sample = sampleBrentBurley(rand, wo, cosThetaO, n, alpha, tangentToWorld, baseDiffuse);
            ray.direction = sample.direction;
            throughput *= sample.throughput / (1.0f - pSpecular);
        }

        // Russian roulette
        const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        if (fract(getRand(depth, 3) + rotation.z) >= pContinue) {
            throughput = make_float3(0.0f);
            break;
        }
        throughput /= pContinue;

        ray.origin = hitPoint + 1e-2f * n;
    }

    params.image[i] = mix(params.image[i], make_float4(throughput, 1.0f), params.weight);
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
    const auto worldSpaceNormal = optixTransformNormalFromObjectToWorldSpace(objectSpaceNormal);

    // Interpolate tangent
    const auto objectSpaceTangent = bary.x * v0.tangent + bary.y * v1.tangent + bary.z * v2.tangent;
    const auto worldSpaceTangent = optixTransformVectorFromObjectToWorldSpace(make_float3(objectSpaceTangent));
    //const auto tangentOrientation = objectSpaceTangent.w; // Used for MikkTSpace normal mapping

    // Get material
    const auto material = data->material;

    setColor(material->color);
    setNormal(normalize(worldSpaceNormal));
    setTangent(worldSpaceTangent);
    setRoughness(material->roughness);
    setMetallic(material->metallic);
    setT(optixGetRayTmax());
}

extern "C" __global__ void __miss__ms() {
    const auto dir = optixGetWorldRayDirection();
    auto sky = make_float3(0.1f);
    const auto sundir = normalize(make_float3(0.5f, 0.5f, 0.5f));
    sky += clamp(powf(dot(dir, sundir), 100.0f), 0.0f, 1.0f) * make_float3(0.8f, 0.9f, 1.0f) * 5.0f;

    setColor(sky);
    setT(INFINITY);
}