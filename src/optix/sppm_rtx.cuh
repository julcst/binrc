#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

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
    float totalCollectedPhotons = 0; // Total number of photons collected
    uint32_t collectedPhotons = 0; // Number of photons collected in this pass
    uint32_t totalPhotonCountAtBirth = 0; // Total number of photons at initialization
    float3 throughput = {0.0f, 0.0f, 0.0f}; // Throughput (only used for SPPM)

    __device__ float3 calcRadiance(uint32_t totalPhotonCount) const {
        // TODO: Handle totalPhotonCountAtBirth >= totalPhotonCount
        uint32_t emittedPhotons = totalPhotonCount - totalPhotonCountAtBirth;
        if (emittedPhotons == 0) return {0.0f, 0.0f, 0.0f}; // Avoid division by zero
        return flux / (static_cast<float>(emittedPhotons) * PI * pow2(radius));
    }

    __device__ void applyRadiusReduction(float alpha) {
        if (collectedPhotons == 0) return; // No photons collected, no radius reduction needed
        const float newCollectedPhotons = totalCollectedPhotons + collectedPhotons * alpha;
        const float factor = newCollectedPhotons / (totalCollectedPhotons + collectedPhotons);
        totalCollectedPhotons = newCollectedPhotons;
        radius *= sqrt(factor);
        // TODO: radius2 *= factor; // If using squared radius
        flux *= factor;
    }

    // float calcRadius() const {
    //     return initialRadius * exp(-count);
    // }
};

struct PhotonQueryView {
    struct Atomics {
        uint32_t index = 0;
    };
    PhotonQuery* queries = nullptr;
    OptixAabb* aabbs = nullptr;
    Atomics* atomics = nullptr;
    uint32_t queryCount = 0;
    OptixTraversableHandle handle = 0;
    uint32_t totalPhotonCount = 0;
    float alpha; // Radius reduction factor
    float initialRadius; // Initial radius for photon queries
    float photonRecordingProbability; // Probability that a non-caustic photon is recorded

    __device__ __forceinline__ void updateAABB(const uint32_t idx, const PhotonQuery& query) const {
        aabbs[idx] = {
            .minX = query.pos.x - query.radius,
            .minY = query.pos.y - query.radius,
            .minZ = query.pos.z - query.radius,
            .maxX = query.pos.x + query.radius,
            .maxY = query.pos.y + query.radius,
            .maxZ = query.pos.z + query.radius
        };
    }

    __device__ __forceinline__ void markAABBInvalid(const uint32_t idx) const {
        aabbs[idx] = {
            .minX = 0.0f,
            .minY = 0.0f,
            .minZ = 0.0f,
            .maxX = 0.0f,
            .maxY = 0.0f,
            .maxZ = 0.0f
        };
    }

    __device__ __forceinline__ void store(const uint32_t idx, const PhotonQuery& query) const {
        queries[idx] = query;
        updateAABB(idx, query);
    }

#ifdef __CUDA_ARCH__
    __device__ __forceinline__ void store(const PhotonQuery& query) const {
        const auto idx = atomicAdd(&(atomics->index), 1) % queryCount;
        store(idx, query);
    }

    __device__ __forceinline__ void recordPhoton(const Photon& photon) const {
        constexpr float EPS = 1e-6f; // Small epsilon, because zero is not allowed
        std::array p = {
            __float_as_uint(photon.wi.x), __float_as_uint(photon.wi.y), __float_as_uint(photon.wi.z),
            __float_as_uint(photon.flux.x), __float_as_uint(photon.flux.y), __float_as_uint(photon.flux.z)
        };
        // TODO: Use PayloadTypeID
        optixTraverse(handle,
            photon.pos, {EPS}, // origin, direction
            0.0f, EPS, // tmin, tmax
            0.0f, // rayTime
            OptixVisibilityMask(1), 
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0, 1, 1, // SBT offset, stride, miss index
            p[0], p[1], p[2], p[3], p[4], p[5] // payload
        );
        optixReorder();
        optixInvoke(p[0], p[1], p[2], p[3], p[4], p[5]);
    }
#endif
};