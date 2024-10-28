#include <optix_device.h>

#include "optixparams.hpp"

void __device__ makeCameraRay(const uint3& idx, const uint3& dim, float3& rayOrigin, float3& rayDir) {
    rayOrigin = make_float3(0.0f, 0.0f, -1.0f);
    rayDir = make_float3(
        (2.0f * idx.x / dim.x - 1.0f) * dim.y / dim.x,
        -1.0f + 2.0f * idx.y / dim.y,
        1.0f
    );
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    uint3 payload;
    float3 rayOrigin, rayDir;
    makeCameraRay(idx, dim, rayOrigin, rayDir);
    optixTrace(params.handle, rayOrigin, rayDir, 0.0f, 1e32f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 0, 0, payload.x, payload.y, payload.z);

    const uint i = idx.y * params.dim.x + idx.x;
    params.image[i] = make_float4(__uint_as_float(payload.x), __uint_as_float(payload.y), __uint_as_float(payload.z), 1.0f);
}

void __device__ setPayload(const float3& value) {
    optixSetPayload_0(__float_as_uint(value.x));
    optixSetPayload_1(__float_as_uint(value.y));
    optixSetPayload_2(__float_as_uint(value.z));
}

extern "C" __global__ void __closesthit__ch() {
    const float2 bary = optixGetTriangleBarycentrics();
 
    const float3 c = make_float3(bary.x, bary.y, 1.0f - bary.x - bary.y); 
    setPayload(c);
}

extern "C" __global__ void __miss__ms() {
    setPayload(make_float3(0.0f, 0.5f, 0.0f));
}