#pragma once

#ifdef __CUDA_ARCH__
#include <optix_device.h>
#endif

#include "prinicpled_brdf_types.cuh"

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
    float radius = 0.0f; // Radius of accumulation TODO: Radius reduction
    float3 flux = {0.0f}; // Accumulated outgoing flux
    uint32_t count = 0; // Number of photons found

    // float3 calcRadiance() const {
    //     if (count == 0) return {0.0f}; // No photons found
    //     // TODO: Calculate totalCount by reduction
    //     return flux / (static_cast<float>(totalCount) * PI * pow2(radius));
    // }

    // float calcRadius() const {
    //     return initialRadius * exp(-count);
    // }
};

struct PhotonQueryView {
    PhotonQuery* queries = nullptr;
    OptixAabb* aabbs = nullptr;
    OptixTraversableHandle handle = 0;

#ifdef __CUDA_ARCH__
    __device__ __forceinline__ void store(const uint32_t idx, const PhotonQuery& query) const {
        queries[idx] = query;
        aabbs[idx] = {
            .minX = query.pos.x - query.radius,
            .minY = query.pos.y - query.radius,
            .minZ = query.pos.z - query.radius,
            .maxX = query.pos.x + query.radius,
            .maxY = query.pos.y + query.radius,
            .maxZ = query.pos.z + query.radius
        };
    }

    __device__ __forceinline__ void recordPhoton(const Photon& photon) const {
        constexpr float EPS = 0.0f;
        std::array p = {
            __float_as_uint(photon.wi.x), __float_as_uint(photon.wi.y), __float_as_uint(photon.wi.z),
            __float_as_uint(photon.flux.x), __float_as_uint(photon.flux.y), __float_as_uint(photon.flux.z)
        };
        // TODO: Use PayloadTypeID
        optixTrace(handle,
            photon.pos, {EPS}, // origin, direction
            0.0f, EPS, // tmin, tmax
            0.0f, // rayTime
            OptixVisibilityMask(1), 
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            1, 0, 1, // SBT offset, stride, miss index
            p[0], p[1], p[2], p[3], p[4], p[5] // payload
        );
        // TODO: SER for cache efficiency
        // optixInvoke();
    }
#endif
};