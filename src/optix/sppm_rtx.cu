#include <optix_device.h>
#include <cuda_runtime.h>

#include "principled_brdf.cuh"
#include "cudamath.cuh"
#include "sppm_rtx.cuh"
#include "params.cuh"
#include "common.cuh"

// Point-Sphere interection test
extern "C" __global__ void __intersection__() {
    const auto idx = optixGetPrimitiveIndex();
    PhotonQuery* query = params.photonMap.queries + idx;

    const auto rayOrigin = optixGetWorldRayOrigin();

    float dist2 = length2(rayOrigin - query->pos);

    if (dist2 > pow2(query->radius)) {
        //optixIgnoreIntersection();
    } else {
        optixReportIntersection(optixGetRayTmin(), 0);
    }
}

// TODO: Disable AnyHit for performance?
extern "C" __global__ void __anyhit__() {
    PhotonQuery* query = params.photonMap.queries + optixGetPrimitiveIndex();
    
    const float3 wi = {__uint_as_float(optixGetPayload_0()),
                       __uint_as_float(optixGetPayload_1()),
                       __uint_as_float(optixGetPayload_2())};
    const float3 flux = {__uint_as_float(optixGetPayload_3()),
                         __uint_as_float(optixGetPayload_4()),
                         __uint_as_float(optixGetPayload_5())};

    // FIXME: Multiply with PI / cosThetaI
    const auto radiance = evalDisneyBRDF(wi, query->wo, query->n, query->mat) * flux;

    // NOTE: Does atomicAdd hurt performance because of serialization?
    // Probably not so much, because we have rather sparse photon queries
    // Maybe warp aggregated atomics could help?
    atomicAdd(&query->collectedPhotons, 1u);
    atomicAdd(&query->flux.x, radiance.x);
    atomicAdd(&query->flux.y, radiance.y);
    atomicAdd(&query->flux.z, radiance.z);
}

// NOTE: Never use empty miss and closest hit programs in OptiX, use nullptr instead to minimize overhead.
// extern "C" __global__ void __miss__() {/*Empty*/}
// extern "C" __global__ void __closesthit__() {/*Empty*/}

extern "C" __global__ void __raygen__visualize() {
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.x + idx.y * dim.x;

    const auto uv = (make_float2(idx.x, idx.y)) / make_float2(dim.x, dim.y);
    auto ray = makeCameraRay(uv);

    float3 color = {0.0f, 0.0f, 0.0f};
    float weight = 0.0f;

    // TODO: Use PayloadTypeID
    std::array p = {
        __float_as_uint(color.x), __float_as_uint(color.y), __float_as_uint(color.z), __float_as_uint(weight),
    };
    optixTrace(params.photonMap.handle,
        ray.origin, ray.direction, // origin, direction
        0.0f, 1e6f, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(1), 
        OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0, 1, 1, // SBT offset, stride, miss index
        p[0], p[1], p[2], p[3] // payload
    );
    color = {
        __uint_as_float(p[0]),
        __uint_as_float(p[1]),
        __uint_as_float(p[2])
    };
    weight = __uint_as_float(p[3]);
    if (weight > 0.0f) color /= weight; // Normalize color by weight

    params.image[i] = make_float4(color, 1.0f);
}

__device__ __forceinline__ float diskIntersect(float3 rayOrigin, float3 rayDirection, float3 center, float3 normal, float radius) {
    const auto o = rayOrigin - center;
    const auto denominator = -dot(rayDirection, normal);
    if (denominator < 1e-6f) return NAN; // Ray is parallel or back-facing to the disk
    const float t = dot(normal, o) / denominator;
    const auto q = o + rayDirection * t;
    const auto dist2 = dot(q, q);
    if (dist2 > radius * radius) return NAN; // Outside the disk
    return t;
}

extern "C" __global__ void __intersection__visualize() {
    const auto idx = optixGetPrimitiveIndex();
    PhotonQuery* query = params.photonMap.queries + idx;

    const auto rayOrigin = optixGetWorldRayOrigin();
    const auto rayDirection = optixGetWorldRayDirection();
    const auto center = query->pos;
    const auto radius = query->radius;
    const auto normal = sign(dot(query->n, query->wo)) * query->n; // Ensure normal is facing wo

    const auto t = diskIntersect(rayOrigin, rayDirection, center, normal, radius);
    if (isfinite(t) && t <= optixGetRayTmax() + 2.0f * radius) {
        optixReportIntersection(t, 0);
        float3 output = {__uint_as_float(optixGetPayload_0()),
                         __uint_as_float(optixGetPayload_1()),
                         __uint_as_float(optixGetPayload_2())};
        float weight = __uint_as_float(optixGetPayload_3());

        float3 radiance = query->calcRadiance(params.photonMap.totalPhotonCount);
        float localWeight = max(dot(query->wo, -rayDirection), 0.0f) + 1e-2f;
        output += radiance * localWeight;
        weight += localWeight;

        optixSetPayload_0(__float_as_uint(output.x));
        optixSetPayload_1(__float_as_uint(output.y));
        optixSetPayload_2(__float_as_uint(output.z));
        optixSetPayload_3(__float_as_uint(weight));
    }

    // const auto a = length2(rayDirection);
    // const auto b = 2.0f * dot(rayDirection, rayOrigin - center);
    // const auto c = length2(rayOrigin - center) - pow2(radius);
    // const auto discriminant = pow2(b) - 4.0f * a * c;
    // if (discriminant < 0.0f) {
    //     // No intersection
    //     return;
    // }
    // const auto sqrtDiscriminant = sqrtf(discriminant);
    // const auto t1 = (-b - sqrtDiscriminant) / (2.0f * a);
    // optixReportIntersection(t1, 0);
}

extern "C" __global__ void __closesthit__visualize() {
    PhotonQuery* query = params.photonMap.queries + optixGetPrimitiveIndex();
    const auto radiance = query->calcRadiance(params.photonMap.totalPhotonCount);
    
    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
    optixSetPayload_2(__float_as_uint(radiance.z));
}