#pragma once

#include <cuda_runtime.h>

struct MaterialProperties {
    float3 F0;
    float3 albedo;
    float alpha2;
    float transmission;
};