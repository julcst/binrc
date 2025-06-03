#include <optix_device.h>
#include <cuda_runtime.h>

#include "principled_brdf.cuh"
#include "cudamath.cuh"

struct Photon {
    float3 pos; // Position of the photon
    float3 wi; // Incoming direction of the photon
    float3 flux; // Incoming flux (irradiance)
};

struct PhotonQuery {
    float3 pos; // Position of query
    float3 wo; // Outgoing direction for which to accumulate photons
    float3 n; // Normal at the query position
    MaterialProperties mat; // Material properties at the query position
    float radius; // Radius of accumulation TODO: Radius reduction
    float3 flux = {0.0f}; // Accumulated outgoing flux
    uint32_t count = 0; // Number of photons found

    float3 calcRadiance() const {
        if (count == 0) return {0.0f}; // No photons found
        return flux / (static_cast<float>(count) * PI * pow2(radius));
    }
};

__device__ __forceinline__ void recordPhoton(OptixTraversableHandle queries, const Photon& photon) {
    constexpr float EPS = 0.0f;
    std::array p = {
        __float_as_uint(photon.wi.x), __float_as_uint(photon.wi.y), __float_as_uint(photon.wi.z),
        __float_as_uint(photon.flux.x), __float_as_uint(photon.flux.y), __float_as_uint(photon.flux.z)
    };
    optixTrace(queries,
        photon.pos, {EPS}, // origin, direction
        0.0f, EPS, // tmin, tmax
        0.0f, // rayTime
        OptixVisibilityMask(1), 
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 
        0, 1, 0, // SBT offset, stride, miss index
        p[0], p[1], p[2], p[3], p[4], p[5] // payload
    );
    // TODO: SER for cache efficiency
    // optixInvoke();
}

// Point-Sphere interection test
extern "C" __global__ void __intersection__() {
    const auto idx = optixGetPrimitiveIndex();
    auto* query = reinterpret_cast<PhotonQuery*>(optixGetSbtDataPointer());

    const auto rayOrigin = optixGetWorldRayOrigin();

    float dist2 = pow2(rayOrigin - query->pos);

    if (dist2 > pow2(query->radius)) {
        return; // No intersection
    }

    optixReportIntersection(optixGetRayTmin(), 0);
}

// TODO: Disable AnyHit for performance?
extern "C" __global__ void __anyhit__() {
    auto* query = reinterpret_cast<PhotonQuery*>(optixGetSbtDataPointer());
    
    const float3 wi = {__uint_as_float(optixGetPayload_0()),
                       __uint_as_float(optixGetPayload_1()),
                       __uint_as_float(optixGetPayload_2())};
    const float3 flux = {__uint_as_float(optixGetPayload_3()),
                         __uint_as_float(optixGetPayload_4()),
                         __uint_as_float(optixGetPayload_5())};

    const auto radiance = evalDisneyBRDFOnly(query->wo, wi, query->n, query->mat) * flux;

    // NOTE: Does atomicAdd hurt performance beacuse of serialization?
    // Maybe warp aggregated atomics could help?
    atomicAdd(&query->count, 1u);
    atomicAdd(&query->flux.x, radiance.x);
    atomicAdd(&query->flux.y, radiance.y);
    atomicAdd(&query->flux.z, radiance.z);
}

extern "C" __global__ void __miss__() {/*Empty*/}
extern "C" __global__ void __closesthit__() {/*Empty*/}