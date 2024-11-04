#include <optix_device.h>
#include <cuda_runtime.h>

#include "optixparams.hpp"
#include "cudaglm.hpp"
#include "brdf.hpp"

struct Ray {
    vec3 origin;
    vec3 direction;
};

__device__ Ray makeCameraRay(const vec2& uv) {
    const vec4 origin = params.clipToWorld[3]; // = params.clipToWorld[3] * vec4(0.0f, 0.0f, 0.0f, 1.0f);
    const vec4 clipTarget(-2.0f * uv + 1.0f, -1.0f, 1.0f);
    const vec4 target = params.clipToWorld * clipTarget;
    const vec3 origin3 = vec3(origin) / origin.w;
    const vec3 dir3 = normalize(origin3 - vec3(target) / target.w);
    return Ray{origin3, dir3};
}

struct Payload {
    vec3 color;
    vec3 normal;
    float roughness;
    float metallic;
    float t;
};

__device__ void setColor(const vec3& value) {
    optixSetPayload_0(__float_as_uint(value.x));
    optixSetPayload_1(__float_as_uint(value.y));
    optixSetPayload_2(__float_as_uint(value.z));
}

__device__ void setNormal(const vec3& value) {
    optixSetPayload_3(__float_as_uint(value.x));
    optixSetPayload_4(__float_as_uint(value.y));
    optixSetPayload_5(__float_as_uint(value.z));
}

__device__ void setRoughness(const float value) {
    optixSetPayload_6(__float_as_uint(value));
}

__device__ void setMetallic(const float value) {
    optixSetPayload_7(__float_as_uint(value));
}

__device__ void setT(const float value) {
    optixSetPayload_8(__float_as_uint(value));
}

__device__ Payload getPayload(uint a, uint b, uint c, uint d, uint e, uint f, uint g, uint h, uint i) {
    return Payload{
        vec3(__uint_as_float(a), __uint_as_float(b), __uint_as_float(c)),
        vec3(__uint_as_float(d), __uint_as_float(e), __uint_as_float(f)),
        __uint_as_float(g),
        __uint_as_float(h),
        __uint_as_float(i),
    };
}

__device__ Payload trace(const Ray& ray) {
    uint a, b, c, d, e, f, g, h, i;
    optixTraverse(
        params.handle,
        glmToCuda(ray.origin), glmToCuda(ray.direction),
        0.0f, MAX_T, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT offset, stride, miss index
        a, b, c, d, e, f, g, h, i // Payload
    );
    //optixReorder(); // TODO: Provide coherence hints
    optixInvoke(a, b, c, d, e, f, g, h, i);
    return getPayload(a, b, c, d, e, f, g, h, i);
}

__device__ float luminance(const vec3& linearRGB) {
    return dot(vec3(0.2126f, 0.7152f, 0.0722f), linearRGB);
}

extern "C" __global__ void __raygen__rg() {
    const uvec3 idx = cudaToGlm(optixGetLaunchIndex());
    const uvec3 dim = cudaToGlm(optixGetLaunchDimensions());
    const uint i = idx.y * params.dim.x + idx.x;
    const vec4 rotation = params.rotationTable[i];

    const vec2 jitter = fract(vec2(getRand(0), getRand(1)) + vec2(rotation));
    const vec2 uv = (vec2(idx) + jitter) / vec2(dim);
    auto ray = makeCameraRay(uv);

    Payload payload;
    vec3 throughput = vec3(1.0f);
    for (uint depth = 0; depth < MAX_BOUNCES; depth++) {
        payload = trace(ray);

        if (isinf(payload.t)) {
            throughput *= payload.color;
            break;
        }

        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto alpha = payload.roughness * payload.roughness;
        const auto alpha2 = alpha * alpha;
        const auto wo = -ray.direction;
        const auto cosThetaO = dot(wo, payload.normal);
        const auto F0 = mix(vec3(0.04f), payload.color, payload.metallic);

        // Importance sampling weights // TODO: Use precomputed
        const auto wSpecular = luminance(F_SchlickApprox(cosThetaO, F0));
        const auto wDiffuse = (1.0f - payload.metallic) * luminance(payload.color);
        const auto pSpecular = wSpecular / (wSpecular + wDiffuse);
        const auto pDiffuse = 1.0f - pSpecular;

        if (fract(getRand(depth, 0) + rotation.w) < pSpecular) { 
            // Sample Trowbridge-Reitz specular
            const auto vndfRand = fract(vec2(getRand(depth, 1) + rotation.x, getRand(depth, 2) + rotation.y));
            const auto microfacetNormal = sampleVNDFTrowbridgeReitz(vndfRand, wo, alpha, payload.normal);
            ray.direction = reflect(-wo, microfacetNormal);
            const auto cosThetaD = dot(wo, microfacetNormal); // = dot(ray.direction, microfacetNormal)
            const auto cosThetaI = dot(ray.direction, payload.normal);
            const auto F = F_SchlickApprox(cosThetaD, F0);
            const auto LambdaL = Lambda_TrowbridgeReitz(cosThetaI, alpha2);
            const auto LambdaV = Lambda_TrowbridgeReitz(cosThetaO, alpha2);
            const auto specular = F * (1.0f + LambdaV) / (1.0f + LambdaL + LambdaV); // = F * (G2 / G1)
            throughput *= specular / pSpecular;
        } else {
            // Sample Brent-Burley diffuse
            const auto hemisphereRand = fract(vec2(getRand(depth, 1) + rotation.z, getRand(depth, 2) + rotation.w)); 
            const auto tangentToWorld = buildTBN(payload.normal);
            ray.direction = tangentToWorld * sampleCosineHemisphere(hemisphereRand);
            const auto microfacetNormal = normalize(ray.direction + wo);
            const auto cosThetaD = dot(wo, microfacetNormal); // = dot(ray.direction, microfacetNormal)
            const auto cosThetaI = dot(ray.direction, payload.normal);
            const auto FD90 = 0.5f + 2.0f * alpha * cosThetaD * cosThetaD;
            const auto response = (1.0f + (FD90 - 1.0f) * pow5(1.0f - cosThetaI)) * (1.0f + (FD90 - 1.0f) * pow5(1.0f - cosThetaO));
            // NOTE: We drop the 1.0 / PI prefactor
            const auto diffuse = (1.0f - payload.metallic) * payload.color * response;
            throughput *= diffuse / pDiffuse;
        }

        // Russian roulette
        const float pContinue = min(luminance(throughput) * params.russianRouletteWeight, 1.0f);
        if (fract(getRand(depth, 3) + rotation.z) > pContinue) {
            throughput = vec3(0.0f);
            break;
        }
        throughput /= pContinue;

        ray.origin = hitPoint + 1e-2f * payload.normal;
    }

    params.image[i] = mix(params.image[i], vec4(throughput, 1.0f), params.weight);
}

extern "C" __global__ void __closesthit__ch() {
    // Get optix built-in variables
    const auto bary2 = cudaToGlm(optixGetTriangleBarycentrics());
    const auto bary = vec3(1.0f - bary2.x - bary2.y, bary2);
    const auto data = reinterpret_cast<HitData*>(optixGetSbtDataPointer());

    // Get triangle vertices
    const auto idx = data->indexBuffer[optixGetPrimitiveIndex()];
    const auto v0 = data->vertexData[idx.x];
    const auto v1 = data->vertexData[idx.y];
    const auto v2 = data->vertexData[idx.z];

    // Interpolate normal
    const auto objectSpaceNormal = bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal;
    const auto worldSpaceNormal = cudaToGlm(optixTransformNormalFromObjectToWorldSpace(glmToCuda(objectSpaceNormal)));

    // Get material
    const auto material = data->material;

    setColor(material->color);
    setNormal(normalize(worldSpaceNormal));
    setRoughness(material->roughness);
    setMetallic(material->metallic);
    setT(optixGetRayTmax());
}

extern "C" __global__ void __miss__ms() {
    const vec3 dir = cudaToGlm(optixGetWorldRayDirection());

    setColor(0.5f * (dir + 1.0f));
    setT(INFINITY);
}