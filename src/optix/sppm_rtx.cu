#include <optix_device.h>
#include <cuda_runtime.h>

#include "principled_brdf.cuh"
#include "cudamath.cuh"
#include "sppm_rtx.cuh"
#include "params.cuh"

// Point-Sphere interection test
extern "C" __global__ void __intersection__() {
    const auto idx = optixGetPrimitiveIndex();
    PhotonQuery* query = params.photonMap.queries + idx;

    const auto rayOrigin = optixGetWorldRayOrigin();

    float dist2 = pow2(rayOrigin - query->pos);

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

    const auto radiance = evalDisneyBRDFOnly(query->wo, wi, query->n, query->mat) * flux;

    // NOTE: Does atomicAdd hurt performance because of serialization?
    // Probably not so much, because we have rather sparse photon queries
    // Maybe warp aggregated atomics could help?
    atomicAdd(&query->count, 1u);
    atomicAdd(&query->flux.x, radiance.x);
    atomicAdd(&query->flux.y, radiance.y);
    atomicAdd(&query->flux.z, radiance.z);
}

// NOTE: Never use empty miss and closest hit programs in OptiX, use nullptr instead to minimize overhead.
// extern "C" __global__ void __miss__() {/*Empty*/}
// extern "C" __global__ void __closesthit__() {/*Empty*/}