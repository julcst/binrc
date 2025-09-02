#pragma once

#include <cuda_runtime.h>
#include <cudamath.cuh>

struct MaterialProperties {
    float3 F0;
    float3 albedo;
    float alpha2;
    float transmission;

    __device__ constexpr bool isDiffuse() const {
        return luminance(albedo * (1 - transmission)) > 1e-6f;
    }
};