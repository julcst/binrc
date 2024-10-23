#include <optix_device.h>

#include "optixparams.hpp"

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const uint i = idx.y * params.dim.x + idx.x;
    params.image[i] = make_uchar4(idx.x, idx.y, 0, 255);
}