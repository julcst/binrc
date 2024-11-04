#include <optix_device.h>
#include <cuda_runtime.h>

#include "optixparams.hpp"
#include "cudaglm.hpp"

struct Ray {
    vec3 origin;
    vec3 direction;
};

__device__ Ray makeCameraRay(const vec2& uv) {
    const vec4 origin = params.clipToWorld * vec4(0.0f, 0.0f, 0.0f, 1.0f);
    const vec4 clipTarget(-2.0f * uv + 1.0f, -1.0f, 1.0f);
    const vec4 target = params.clipToWorld * clipTarget;
    const vec3 origin3 = vec3(origin) / origin.w;
    const vec3 dir3 = normalize(origin3 - vec3(target) / target.w);
    return Ray{origin3, dir3};
}

struct Payload {
    vec3 color;
    vec3 normal;
    float t;
};

__device__ void setPayload(const Payload& value) {
    optixSetPayload_0(__float_as_uint(value.color.x));
    optixSetPayload_1(__float_as_uint(value.color.y));
    optixSetPayload_2(__float_as_uint(value.color.z));
    optixSetPayload_3(__float_as_uint(value.normal.x));
    optixSetPayload_4(__float_as_uint(value.normal.y));
    optixSetPayload_5(__float_as_uint(value.normal.z));
    optixSetPayload_6(__float_as_uint(value.t));
}

__device__ Payload getPayload() {
    return Payload{
        vec3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2())),
        vec3(__uint_as_float(optixGetPayload_3()), __uint_as_float(optixGetPayload_4()), __uint_as_float(optixGetPayload_5())),
        __uint_as_float(optixGetPayload_6()),
    };
}

__device__ Payload getPayload(uint a, uint b, uint c, uint d, uint e, uint f, uint g) {
    return Payload{
        vec3(__uint_as_float(a), __uint_as_float(b), __uint_as_float(c)),
        vec3(__uint_as_float(d), __uint_as_float(e), __uint_as_float(f)),
        __uint_as_float(g),
    };
}

__device__ Payload trace(const Ray& ray) {
    uint a, b, c, d, e, f, g;
    optixTrace(
        params.handle,
        glmToCuda(ray.origin), glmToCuda(ray.direction),
        0.0f, MAX_T, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT offset, stride, miss index
        a, b, c, d, e, f, g // Payload
    );
    return getPayload(a, b, c, d, e, f, g);
}

// Has to be called in raygen
__device__ Payload rtrace(const Ray& ray) {
    uint a, b, c, d, e, f, g;
    optixTraverse(
        params.handle,
        glmToCuda(ray.origin), glmToCuda(ray.direction),
        0.0f, MAX_T, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT offset, stride, miss index
        a, b, c, d, e, f, g // Payload
    );
    optixReorder(); // TODO: Provide coherence hints
    optixInvoke(a, b, c, d, e, f, g);
    return getPayload(a, b, c, d, e, f, g);
}

__device__ float getRand(uint dim) {
    const uint i = params.sample - params.sequenceOffset + params.sequenceStride * dim;
    return params.randSequence[i];
}

__device__ vec2 getRand4(uint dim, vec4 rotation) {
    return fract(vec4(getRand(dim), getRand(dim + 1), getRand(dim + 2), getRand(dim + 3)) + rotation);
}

__device__ vec3 SampleVndf_GGX(vec2 u, vec3 wi, float alpha, vec3 n) {
    // Dirac function for alpha = 0
    if (alpha == 0.0f) return n;
    // decompose the vector in parallel and perpendicular components
    vec3 wi_z = n * dot(wi, n);
    vec3 wi_xy = wi - wi_z;
    // warp to the hemisphere configuration
    vec3 wiStd = normalize(wi_z - alpha * wi_xy);
    // sample a spherical cap in (-wiStd.z, 1]
    float wiStd_z = dot(wiStd, n);
    float phi = (2.0f * u.x - 1.0f) * 3.1415926f;
    float z = (1.0f - u.y) * (1.0f + wiStd_z) - wiStd_z;
    float sinTheta = sqrt(clamp(1.0f - z * z, 1e-6f, 1.0f));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    vec3 cStd = vec3(x, y, z);
    // reflect sample to align with normal
    vec3 up = vec3(0.0f, 0.0f, 1.0f);
    vec3 wr = n + up;
    // prevent division by zero
    float wrz_safe = max(wr.z, 1e-6f);
    vec3 c = dot(wr, cStd) * wr / wrz_safe - cStd;
    // compute halfway direction as standard normal
    vec3 wmStd = c + wiStd;
    vec3 wmStd_z = n * dot(n, wmStd);
    vec3 wmStd_xy = wmStd_z - wmStd;
    // warp back to the ellipsoid configuration
    vec3 wm = normalize(wmStd_z + alpha * wmStd_xy);
    // return final normal
    return wm;
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
    for (uint depth = 0; depth < MAX_RAY_DEPTH; depth++) {
        payload = trace(ray);
        throughput *= payload.color;
        if (isinf(payload.t)) break;
        const auto hitPoint = ray.origin + payload.t * ray.direction;
        const auto vndfRand = fract(vec2(getRand(2), getRand(3)) + vec2(rotation));
        const auto microfacetNormal = SampleVndf_GGX(vndfRand, -ray.direction, 0.1f, payload.normal);
        ray.origin = hitPoint + 1e-2f * payload.normal;
        ray.direction = reflect(ray.direction, microfacetNormal);
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

    Payload payload;
    payload.color = vec3(0.7f);
    payload.normal = normalize(worldSpaceNormal);
    payload.t = optixGetRayTmax();
    setPayload(payload);
}

extern "C" __global__ void __miss__ms() {
    const vec3 dir = cudaToGlm(optixGetWorldRayDirection());

    Payload payload;
    payload.color = 0.5f * (dir + 1.0f);
    payload.t = INFINITY;
    setPayload(payload);
}