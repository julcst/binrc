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
    uint depth;
};

__device__ void setPayload(const Payload& value) {
    optixSetPayload_0(__float_as_uint(value.color.x));
    optixSetPayload_1(__float_as_uint(value.color.y));
    optixSetPayload_2(__float_as_uint(value.color.z));
    optixSetPayload_3(value.depth);
}

__device__ Payload getPayload() {
    return Payload{
        vec3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2())), optixGetPayload_3()
    };
}

__device__ Payload getPayload(uint a, uint b, uint c, uint d) {
    return Payload{
        vec3(__uint_as_float(a), __uint_as_float(b), __uint_as_float(c)), d

    };
}

__device__ Payload trace(const Ray& ray, uint depth = 0) {
    uint a, b, c;
    optixTrace(
        params.handle,
        glmToCuda(ray.origin), glmToCuda(ray.direction),
        0.0f, 1e32f, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT offset, stride, miss index
        a, b, c, depth // Payload
    );
    return getPayload(a, b, c, depth);
}

// Has to be called in raygen
__device__ Payload rtrace(const Ray& ray, uint depth = 0) {
    uint a, b, c;
    optixTraverse(
        params.handle,
        glmToCuda(ray.origin), glmToCuda(ray.direction),
        0.0f, 1e32f, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT offset, stride, miss index
        a, b, c, depth // Payload
    );
    optixReorder(); // TODO: Provide coherence hints
    optixInvoke(a, b, c, depth);
    return getPayload(a, b, c, depth);
}

extern "C" __global__ void __raygen__rg() {
    const uvec3 idx = cudaToGlm(optixGetLaunchIndex());
    const uvec3 dim = cudaToGlm(optixGetLaunchDimensions());
    const vec2 uv = vec2(idx) / vec2(dim);
    const uint i = idx.y * params.dim.x + idx.x;

    const auto ray = makeCameraRay(uv);

    const auto payload = trace(ray);

    // TODO: Reorder

    params.image[i] = vec4(payload.color, 1.0f);
}

extern "C" __global__ void __closesthit__ch() {
    // Get optix built-in variables
    const auto bary2 = cudaToGlm(optixGetTriangleBarycentrics());
    const auto bary = vec3(1.0f - bary2.x - bary2.y, bary2);
    const auto data = reinterpret_cast<HitData*>(optixGetSbtDataPointer());
    const auto rayOrigin = cudaToGlm(optixGetWorldRayOrigin());
    const auto rayDir = cudaToGlm(optixGetWorldRayDirection());
    const auto hitPoint = rayOrigin + optixGetRayTmax() * rayDir;

    // Get triangle vertices
    const auto idx = data->indexBuffer[optixGetPrimitiveIndex()];
    const auto v0 = data->vertexData[idx.x];
    const auto v1 = data->vertexData[idx.y];
    const auto v2 = data->vertexData[idx.z];

    // Interpolate normal
    const auto objectSpaceNormal = bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal;
    const auto worldSpaceNormal = cudaToGlm(optixTransformNormalFromObjectToWorldSpace(glmToCuda(objectSpaceNormal)));
    const auto normal = normalize(worldSpaceNormal);

    const auto reflectOrigin = hitPoint + 1e-3f * normal;
    const auto reflectDir = reflect(rayDir, normal);
    const auto reflectRay = Ray{reflectOrigin, reflectDir};

    Payload payload;
    payload.depth = optixGetPayload_3();

    if (payload.depth < 16 - 1) {
        const auto payload2 = trace(reflectRay, payload.depth + 1);
        payload.color = payload2.color * 0.7f; 
    }

    setPayload(payload);
}

extern "C" __global__ void __miss__ms() {
    const vec3 dir = cudaToGlm(optixGetWorldRayDirection());

    Payload payload;
    payload.color = 0.5f * (dir + 1.0f);
    setPayload(payload);
}