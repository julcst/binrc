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
};

__device__ void setPayload(const Payload& value) {
    optixSetPayload_0(__float_as_uint(value.color.x));
    optixSetPayload_1(__float_as_uint(value.color.y));
    optixSetPayload_2(__float_as_uint(value.color.z));
}

__device__ Payload getPayload() {
    return Payload{
        vec3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()))
    };
}

__device__ Payload getPayload(uint a, uint b, uint c) {
    return Payload{
        vec3(__uint_as_float(a), __uint_as_float(b), __uint_as_float(c))
    };
}

extern "C" __global__ void __raygen__rg() {
    const uvec3 idx = cudaToGlm(optixGetLaunchIndex());
    const uvec3 dim = cudaToGlm(optixGetLaunchDimensions());
    const vec2 uv = vec2(idx) / vec2(dim);

    const Ray ray = makeCameraRay(uv);
    uint a, b, c;
    optixTrace(params.handle, glmToCuda(ray.origin), glmToCuda(ray.direction), 0.0f, 1e32f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 0, 0, a, b, c);
    const Payload payload = getPayload(a, b, c);

    const uint i = idx.y * params.dim.x + idx.x;
    params.image[i] = vec4(payload.color, 1.0f);
}

extern "C" __global__ void __closesthit__ch() {
    const vec2 bary = cudaToGlm(optixGetTriangleBarycentrics());

    Payload payload;
    payload.color = vec3(bary.x, bary.y, 1.0f - bary.x - bary.y); 
    setPayload(payload);
}

extern "C" __global__ void __miss__ms() {
    const vec3 dir = cudaToGlm(optixGetWorldRayDirection());

    Payload payload;
    payload.color = 0.5f * (dir + 1.0f);
    setPayload(payload);
}