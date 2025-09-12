#include <optix_device.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "params.cuh"
#include "payload.cuh"
#include "common.cuh"
#include "nrc.cuh"
#include "cudamath.cuh"

constexpr uint TRIALS = 1;
constexpr uint N_RANDS = TRIALS * 2;

extern "C" __global__ void __raygen__() {
    if (!params.lightTable) return; // Cannot sample light without lights
    
    const auto idx = optixGetLaunchIndex();
    const auto dim = optixGetLaunchDimensions();
    const auto i = idx.y * params.dim.x + idx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(0, i, params.trainingRound * N_RANDS, &state);

    for (int j = 0; j < TRIALS; j++) { // Dummy loop to align with self-learning skipping
        auto queryRay = makeCameraRay({curand_uniform(&state), curand_uniform(&state)});
        Payload queryPayload = trace(queryRay);
        if (isinf(queryPayload.t)) continue; // Try again
        float3 queryX = queryRay.origin + queryPayload.t * queryRay.direction;
        float3 queryWo = -queryRay.direction;
        float3 queryLo = {0.0f, 0.0f, 0.0f};

        const auto trainInput = encodeInput(queryX, false, queryWo, queryPayload);
        const auto trainIdx = pushNRCTrainInput(trainInput);
        writeNRCOutput(params.trainingTarget, trainIdx, queryLo);
        break;
    }
}