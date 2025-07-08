#include "optixrenderer.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix.h>

#include <framework/common.hpp>

#include <iostream>
#include <array>
#include <vector>
#include <fstream>

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/config.h>

#include "optixir.hpp"
#include "cudautil.hpp"
#include "cudaglm.cuh"
#include "optix/params.cuh"
#include "cudamath.cuh"
#include "optix/sampling.cuh"
#include "optix/sppm_as.cuh"

OptixRenderer::OptixRenderer() {
    check(cudaFree(nullptr)); // Initialize CUDA for this device on this thread
    check(optixInit()); // Initialize the OptiX API
    
    // Initialize the OptiX device context
    OptixDeviceContextOptions options = {
        .logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void*) {
            std::cerr << "[" << tag << "] " << message << std::endl;
        },
        .logCallbackLevel = 4, // Print all log messages
        .validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF,
    };
#ifdef OPTIX_DEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL; // Enable all validation checks
#endif
    CUcontext cuCtx = nullptr; // zero means take the current context
    check(optixDeviceContextCreate(cuCtx, &options, &context));

    // Create module
    OptixModuleCompileOptions moduleCompileOptions = {
        .maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL,
        .numPayloadTypes = 0,
        .payloadTypes = nullptr,
    };
#ifdef OPTIX_DEBUG
    //moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // Disable optimizations
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE; // Generate debug information
#endif
    const OptixPipelineCompileOptions pipelineCompileOptions = {
        .usesMotionBlur = false,
        .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
        .numPayloadValues = PAYLOAD_SIZE,
        .numAttributeValues = 2,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "params",
        .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM),
    };

    for (size_t i = 0; i < optixir::paths.size(); i++) {
        const auto binary = Common::readBinaryFile(optixir::paths[i]);
        check(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions, binary.data(), binary.size(), nullptr, nullptr, &modules[i]));
    }

    // Create program groups
    OptixProgramGroupOptions pgOptions = {};
    const std::array programDecriptions = {
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = modules[optixir::REFERENCE],
                .entryFunctionName = "__raygen__reference",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = modules[optixir::TRAIN_EYE],
                .entryFunctionName = "__raygen__",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = modules[optixir::TRAIN_LIGHT],
                .entryFunctionName = "__raygen__",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = modules[optixir::INFERENCE],
                .entryFunctionName = "__raygen__",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = modules[optixir::SPPM_EYE_PASS],
                .entryFunctionName = "__raygen__",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = modules[optixir::SPPM_LIGHT_PASS],
                .entryFunctionName = "__raygen__",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = modules[optixir::SPPM_RTX],
                .entryFunctionName = "__raygen__visualize",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = modules[optixir::SPPM_EYE_PASS],
                .entryFunctionName = "__raygen__full",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
            .miss = {
                .module = modules[optixir::HIT],
                .entryFunctionName = "__miss__ms",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            .hitgroup = {
                .moduleCH = modules[optixir::HIT],
                .entryFunctionNameCH = "__closesthit__ch",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            .hitgroup = {
                .moduleCH = nullptr, // No closest hit program for SPPM
                .entryFunctionNameCH = nullptr,
                .moduleAH = modules[optixir::SPPM_RTX],
                .entryFunctionNameAH = "__anyhit__",
                .moduleIS = modules[optixir::SPPM_RTX],
                .entryFunctionNameIS = "__intersection__",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
            .miss = {
                .module = nullptr, // No miss program for SPPM
                .entryFunctionName = nullptr,
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            .hitgroup = {
                .moduleCH = modules[optixir::SPPM_RTX],
                .entryFunctionNameCH = "__closesthit__visualize",
                .moduleAH = nullptr, // No any hit program for SPPM
                .entryFunctionNameAH = nullptr,
                .moduleIS = modules[optixir::SPPM_RTX],
                .entryFunctionNameIS = "__intersection__visualize",
            },
        },
    };
    check(optixProgramGroupCreate(context, programDecriptions.data(), programDecriptions.size(), &pgOptions, nullptr, nullptr, programGroups.data()));

    // Create pipeline
    OptixPipelineLinkOptions pipelineLinkOptions = {
        .maxTraceDepth = MAX_BOUNCES,
    };
    check(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), programGroups.size(), nullptr, nullptr, &pipeline));

    // TODO: optixUtilComputeStackSizesSimplePathtracer?

    // Set up shader binding table
    std::vector<RaygenRecord> raygenRecordsHost(sbts.size());
    check(optixSbtRecordPackHeader(programGroups[REFERENCE], &raygenRecordsHost[REFERENCE]));
    check(optixSbtRecordPackHeader(programGroups[TRAIN_EYE], &raygenRecordsHost[TRAIN_EYE]));
    check(optixSbtRecordPackHeader(programGroups[TRAIN_LIGHT], &raygenRecordsHost[TRAIN_LIGHT]));
    check(optixSbtRecordPackHeader(programGroups[INFERENCE], &raygenRecordsHost[INFERENCE]));
    check(optixSbtRecordPackHeader(programGroups[SPPM_EYE_PASS], &raygenRecordsHost[SPPM_EYE_PASS]));
    check(optixSbtRecordPackHeader(programGroups[SPPM_LIGHT_PASS], &raygenRecordsHost[SPPM_LIGHT_PASS]));
    check(optixSbtRecordPackHeader(programGroups[SPPM_VIS_RAYGEN], &raygenRecordsHost[SPPM_VIS_RAYGEN]));
    check(optixSbtRecordPackHeader(programGroups[SPPM_FULL], &raygenRecordsHost[SPPM_FULL]));
    raygenRecords.resize_and_copy_from_host(raygenRecordsHost);

    std::vector<MissRecord> missRecordsHost(sbts.size());
    check(optixSbtRecordPackHeader(programGroups[MISS], &missRecordsHost[0]));
    check(optixSbtRecordPackHeader(programGroups[NO_MISS], &missRecordsHost[1]));
    missRecords.resize_and_copy_from_host(missRecordsHost);

    for (size_t i = 0; i < sbts.size(); i++) {
        sbts[i] = {
            .raygenRecord = reinterpret_cast<CUdeviceptr>(&raygenRecords[i]),
            .missRecordBase = reinterpret_cast<CUdeviceptr>(missRecords.data()),
            .missRecordStrideInBytes = sizeof(MissRecord),
            .missRecordCount = static_cast<uint32_t>(missRecords.size()),
            .hitgroupRecordBase = 0,
            .hitgroupRecordStrideInBytes = sizeof(HitRecord),
            .hitgroupRecordCount = 0,
        };
    }

    nrcModel = tcnn::create_from_config(NRC_INPUT_SIZE, NRC_OUTPUT_SIZE, nlohmann::json::parse(Common::readFile("nrc.json"), nullptr, true, true));
    nrcTrainInput = tcnn::GPUMatrix<float>(NRC_INPUT_SIZE, NRC_BATCH_SIZE);
    nrcTrainOutput = tcnn::GPUMatrix<float>(NRC_OUTPUT_SIZE, NRC_BATCH_SIZE);

    std::cout << "Network: " << std::setw(2) << nrcModel.network->hyperparams()
              << "\nTrainer: " << std::setw(2) << nrcModel.trainer->hyperparams()
              << std::endl;
    
    params.trainingInput = nrcTrainInput.data();
    params.trainingTarget = nrcTrainOutput.data();
    params.selfLearningBounces = selfLearningBounces.data();
    params.selfLearningQueries = selfLearningQueries.data();

    nrcTrainIndex.memset(0);
    params.trainingIndexPtr = nrcTrainIndex.data();

    nrcLightSamples.memset(0);
    params.lightSamples = nrcLightSamples.data();

    params.brdfLUT = brdfLUT.texObj;

    HeaderOnlyRecord sppmVisRecord;
    optixSbtRecordPackHeader(programGroups[SPPM_VIS_HIT], &sppmVisRecord);
    sppmVisRecords = {sppmVisRecord};
    sppmVisSBT = {
        .raygenRecord = reinterpret_cast<CUdeviceptr>(&raygenRecords[(size_t) SPPM_VIS_RAYGEN]),
        .missRecordBase = reinterpret_cast<CUdeviceptr>(missRecords.data()),
        .missRecordStrideInBytes = sizeof(MissRecord),
        .missRecordCount = static_cast<uint32_t>(missRecords.size()),
        .hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(sppmVisRecords.data().get()),
        .hitgroupRecordStrideInBytes = sizeof(HeaderOnlyRecord),
        .hitgroupRecordCount = 1,
    };
    params.photonMap = sppmBVH.getDeviceView();
}

OptixRenderer::~OptixRenderer() {
    for (auto& module : modules) check(optixModuleDestroy(module));
    check(optixPipelineDestroy(pipeline));
    check(optixDeviceContextDestroy(context));
}

__global__ void testSceneSampling(const uint sampleCount, const Instance* instances, const uint instanceCount, const Material* materials) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sampleCount) return;

    curandStatePhilox4_32_10_t state;
    curand_init(0, i, 0, &state);
    const auto rand = curand_uniform4(&state);

    const auto surf = sampleScene(instances, instanceCount, materials, rand.x, make_float2(rand.z, rand.w));
    printf("Sample %d: %f %f %f %f %f %f %f %f %f\n", i, surf.position.x, surf.position.y, surf.position.z, surf.normal.x, surf.normal.y, surf.normal.z, surf.baseColor.x, surf.baseColor.y, surf.baseColor.z);
}

void OptixRenderer::loadGLTF(const std::filesystem::path& path) {
    auto sceneData = scene.loadGLTF(context, path);
    const auto aabb = scene.getAABB();
    const auto size = aabb.max - aabb.min;

    for (auto& hitRecord : sceneData.hitRecords) optixSbtRecordPackHeader(programGroups[CLOSEST_HIT], &hitRecord);

    HitRecord sppmRecord;
    optixSbtRecordPackHeader(programGroups[SPPM_RTX], &sppmRecord);
    sceneData.hitRecords.insert(sceneData.hitRecords.begin(), sppmRecord);

    hitRecords.resize_and_copy_from_host(sceneData.hitRecords);
    materials.resize_and_copy_from_host(sceneData.materials);
    lightTable.resize_and_copy_from_host(sceneData.lightTable);
    instances.resize_and_copy_from_host(sceneData.instances);

    for (auto& sbt : sbts) {
        sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitRecords.data());
        sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
        sbt.hitgroupRecordCount = hitRecords.size();
    }

    params.sceneMin = {aabb.min.x, aabb.min.y, aabb.min.z};
    params.sceneScale = 1.0f / std::max(size.x, std::max(size.y, size.z));
    params.materials = materials.data();
    params.lightTable = lightTable.data();
    params.lightTableSize = lightTable.size();
    params.handle = sceneData.handle;

    std::cout << "Min: (" << params.sceneMin.x << ", " << params.sceneMin.y << ", " << params.sceneMin.z << ") Scale: " << params.sceneScale << std::endl;

    // Test scene sampling
    const uint sampleCount = 100;
    const uint blockSize = 256;
    const uint blockCount = (sampleCount + blockSize - 1) / blockSize;
    testSceneSampling<<<blockCount, blockSize>>>(sampleCount, instances.data(), instances.size(), materials.data());

    reset();
    lossHistory.clear();
}

void OptixRenderer::setCamera(const mat4& clipToWorld) {
    params.clipToWorld = glmToCuda(clipToWorld);
}

__global__ void visualizeInference(Params* params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params->dim.x || y >= params->dim.y) return;

    const int i = y * params->dim.x + x;
    const int idxIn = i * NRC_INPUT_SIZE;
    const int idxOut = i * NRC_OUTPUT_SIZE;
    auto inference = make_float3(params->inferenceOutput[idxOut + 0], params->inferenceOutput[idxOut + 1], params->inferenceOutput[idxOut + 2]);

    const auto throughput = params->inferenceThroughput[i];

    if (throughput.x <= 0.0f && throughput.y <= 0.0f && throughput.z <= 0.0f) return;

    if (params->inferenceMode == InferenceMode::RAW_CACHE) {
        params->image[i] = make_float4(inference, 1.0f);
    } else {
        const auto diffuse = make_float3(params->inferenceInput[idxIn + 8], params->inferenceInput[idxIn + 9], params->inferenceInput[idxIn + 10]);
        const auto specular = make_float3(params->inferenceInput[idxIn + 11], params->inferenceInput[idxIn + 12], params->inferenceInput[idxIn + 13]);
        params->image[i] += params->weight * make_float4(inference * (diffuse + specular) * throughput, 1.0f);
        // params->image[i] = make_float4(throughput, 1.0f);
    }
}

__device__ inline void writeNRCInput(float* dest, uint idx, const NRCInput& input) {
    auto to = dest + idx * NRC_INPUT_SIZE;
    to[0] = input.position.x;
    to[1] = input.position.y;
    to[2] = input.position.z;
    to[3] = input.wo.x;
    to[4] = input.wo.y;
    to[5] = input.wn.x;
    to[6] = input.wn.y;
    to[7] = input.roughness;
    to[8] = input.diffuse.x;
    to[9] = input.diffuse.y;
    to[10] = input.diffuse.z;
    to[11] = input.specular.x;
    to[12] = input.specular.y;
    to[13] = input.specular.z;
}

__device__ inline void writeNRCOutput(float* dest, uint idx, const float3& radiance) {
    auto to = dest + idx * NRC_OUTPUT_SIZE;
    to[0] = radiance.x;
    to[1] = radiance.y;
    to[2] = radiance.z;
}

// TODO: Diffuse encoding
__device__ inline NRCInput encodeInput(const Params* params, const float3& wo, const Surface& surf, const cudaTextureObject_t brdfLUT) {
    const auto F0 = mix(make_float3(0.04f), surf.baseColor, surf.metallic);
    const auto lut = tex2D<float4>(brdfLUT, surf.roughness, dot(surf.normal, wo));
    const auto specular = F0 * lut.x + lut.y;
    const auto albedo = (1.0f - surf.metallic) * surf.baseColor;
    return {
        .position = params->sceneScale * (surf.position - params->sceneMin),
        .wo = toNormSpherical(wo),
        .wn = toNormSpherical(surf.normal),
        .roughness = pow2(surf.roughness),
        .diffuse = albedo,
        .specular = specular,
    };
}

__device__ inline NRCInput encodeInput(const Params* params, const float3& x, const float3& wo, const float3& n, const MaterialProperties& mat, const cudaTextureObject_t brdfLUT) {
    const auto roughness = powf(mat.alpha2, 1.0f / 4.0f);
    const auto lut = tex2D<float4>(brdfLUT, roughness, dot(n, wo));
    const auto specular = mat.F0 * lut.x + lut.y;
    return {
        .position = params->sceneScale * (x - params->sceneMin),
        .wo = toNormSpherical(wo),
        .wn = toNormSpherical(n),
        .roughness = pow2(roughness),
        .diffuse = mat.albedo,
        .specular = mat.F0,
    };
}

__global__ void generateDummySamples(const uint sampleCount, Params* params, const Instance* instances, const uint instanceCount, const Material* materials) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sampleCount) return;

    curandStatePhilox4_32_10_t state;
    curand_init(0, i, params->trainingRound * 5, &state);
    const auto rand = curand_uniform4(&state);

    const auto surf = sampleScene(instances, instanceCount, materials, rand.x, {rand.y, rand.z});
    const auto wo = buildTBN(surf.normal) * sampleCosineHemisphere({rand.w, curand_uniform(&state)});

    const auto input = encodeInput(params, wo, surf, params->brdfLUT);
    const auto idx = atomicAdd(params->trainingIndexPtr, 1u) % NRC_BATCH_SIZE; // TODO: This is a bad and unnecessary use of atomic operations
    writeNRCInput(params->trainingInput, idx, input);
    writeNRCOutput(params->trainingTarget, idx, make_float3(0.0f));
}

__global__ void applySelfLearning(unsigned int numQueries, std::array<TrainBounce, TRAIN_DEPTH>* selfLearningBounces, float* nrcQueries, float* nrcOutput, float* trainTarget) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numQueries) return;

    const int idxIn = i * NRC_INPUT_SIZE;
    const int idxOut = i * NRC_OUTPUT_SIZE;

    const auto bounces = selfLearningBounces[i];
    auto inference = make_float3(nrcOutput[idxOut + 0], nrcOutput[idxOut + 1], nrcOutput[idxOut + 2]);
    const auto diffuse = make_float3(nrcQueries[idxIn + 8], nrcQueries[idxIn + 9], nrcQueries[idxIn + 10]);
    const auto specular = make_float3(nrcQueries[idxIn + 11], nrcQueries[idxIn + 12], nrcQueries[idxIn + 13]);
    inference *= (diffuse + specular);

    for (uint32_t j = TRAIN_DEPTH - 1; j < TRAIN_DEPTH; j--) {
        const auto& bounce = bounces[j];
        if (!bounce.isValid) continue;
        inference *= bounce.throughput;
        /*printf("Bounce %d: (%f %f %f) * Inference (%f %f %f) + Radiance (%f %f %f)\n", bounce.index, bounce.throughput.x, bounce.throughput.y, bounce.throughput.z, 
               inference.x, inference.y, inference.z, bounce.radiance.x, bounce.radiance.y, bounce.radiance.z);*/
        writeNRCOutput(trainTarget, bounce.index, bounce.reflectanceFactorizationTerm * (inference + bounce.radiance));
    }
}

__global__ void writePhotonQueriesToTrainingSet(Params* params, float* nrcQueries, float* trainTarget) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params->photonMap.queryCount) return;

    const auto& photonQuery = params->photonMap.queries[i];

    const auto trainIdx = atomicAdd(params->trainingIndexPtr, 1u) % NRC_BATCH_SIZE;

    const auto input = encodeInput(params, photonQuery.pos, photonQuery.wo, photonQuery.n, photonQuery.mat, params->brdfLUT);
    const auto reflectanceFactorizationTerm = max(input.diffuse + input.specular, 1e-3f);
    const auto radiance = photonQuery.calcRadiance(params->photonMap.totalPhotonCount) / reflectanceFactorizationTerm;

    writeNRCInput(nrcQueries, trainIdx, input);
    writeNRCOutput(trainTarget, trainIdx, radiance);
}

__global__ void accumulatePhotonsToFramebuffer(Params* params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params->dim.x || y >= params->dim.y) return;

    const int i = y * params->dim.x + x;
    if (i >= params->photonMap.queryCount) return; // Safety check

    const auto photonQuery = params->photonMap.queries[i];
    const auto radiance = photonQuery.calcRadiance(params->photonMap.totalPhotonCount);

    params->image[i] *= make_float4(radiance, 1.0f);
}

// TODO: Could do multiple smaller training steps per frame
void OptixRenderer::train() {
    nrcLightSamples.memset(0);
    check(cudaDeviceSynchronize());

    // Calculate sample amounts per training methods
    // FIXME: Fix crash when changing trainigDirection
    const float balanceRatio = 1.0f / params.balanceWeight;
    const auto totalTrainingSamples = NRC_BATCH_SIZE / TRAIN_DEPTH;
    const uint forwardSamples = totalTrainingSamples * (1.0f - trainingDirection);
    const uint backwardSamples = (totalTrainingSamples - forwardSamples) * balanceRatio * (1.0f - photonMappingAmount);
    const uint32_t photonQueries = (totalTrainingSamples - forwardSamples) * TRAIN_DEPTH * photonMappingAmount;
    const uint32_t photonQuerySamples = photonQueries / 6 * photonQueryReplacement;
    //std::cout << "Training samples: " << totalTrainingSamples << " (forward: " << forwardSamples << ", backward: " << backwardSamples << ", photon queries: " << photonQueries << ")" << std::endl;

    sppmBVH.size = photonQueries;
    params.photonMap = sppmBVH.getDeviceView(); // Update handle in params
    paramsBuffer.copy_from_host(&params, 1);
    check(cudaDeviceSynchronize()); // Wait for the copy to finish

    if (photonQueries) {
        if (photonQuerySamples) check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer.data()), sizeof(Params), &sbts[SPPM_EYE_PASS], photonQuerySamples, 1, 1));
        check(cudaDeviceSynchronize()); // Wait for the renderer to finish
        sppmBVH.updatePhotonAS(context);
        check(cudaDeviceSynchronize());
        if (photonCount) check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer.data()), sizeof(Params), &sbts[SPPM_LIGHT_PASS], photonCount, 1, 1));
        check(cudaDeviceSynchronize()); // Wait for the renderer to finish
        sppmBVH.totalPhotonCount += photonCount;
        params.photonMap = sppmBVH.getDeviceView(); // Update handle in params
        paramsBuffer.copy_from_host(&params, 1);
        check(cudaDeviceSynchronize()); // Wait for the copy to finish
        writePhotonQueriesToTrainingSet<<<(photonQueries + 255) / 256, 256>>>(paramsBuffer.data(), nrcTrainInput.data(), nrcTrainOutput.data());
        check(cudaDeviceSynchronize());
    }

    // Generate training samples
    if (forwardSamples) check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer.data()), sizeof(Params), &sbts[TRAIN_EYE], forwardSamples, 1, 1));
    if (backwardSamples) {
        check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer.data()), sizeof(Params), &sbts[TRAIN_LIGHT], backwardSamples, 1, 1));
    }
    check(cudaDeviceSynchronize()); // Wait for the renderer to finish
    
    uint nL = 0; // Real number of light samples
    nrcLightSamples.copy_to_host(&nL);
    // (nL + nD) * r = nL    and    r = 1 / w    =>    nD = nL * w - nL
    uint nD = nL * params.balanceWeight - nL;
    //std::cout << "nL: " << nL << " nD: " << nD << std::endl;
    // FIXME: Upsides too dark
    if (nD) generateDummySamples<<<(nD + 255) / 256, 256>>>(nD, paramsBuffer.data(), instances.data(), instances.size(), materials.data());
    check(cudaDeviceSynchronize()); // Wait for the renderer to finish

    if (params.flags & SELF_LEARNING_FLAG) {
        nrcModel.network->inference(selfLearningQueries, selfLearningInference, false); // Do not apply EMA here
        const auto block = 256;
        const auto grid = (forwardSamples + block - 1) / block;
        applySelfLearning<<<grid, block>>>(forwardSamples, selfLearningBounces.data(), selfLearningQueries.data(), selfLearningInference.data(), nrcTrainOutput.data());
    }
    
    params.trainingRound++;

    // Perform training steps
    for (uint32_t offset = 0; offset < NRC_BATCH_SIZE; offset += NRC_SUBBATCH_SIZE) {
        // TODO: Use pdf
        // TODO: Limit training to the samples generated in this step to improve performance
        // TODO: Shuffling
        auto ctx = nrcModel.trainer->training_step(nrcTrainInput.slice_cols(offset, NRC_SUBBATCH_SIZE), nrcTrainOutput.slice_cols(offset, NRC_SUBBATCH_SIZE));
        float loss = nrcModel.trainer->loss(*ctx);
        lossHistory.push_back(loss);
    }
}

void OptixRenderer::render(vec4* image, uvec2 dim) {

    // Update parameters
    params.image = reinterpret_cast<float4*>(image);
    params.dim = make_uint2(dim.x, dim.y);
    ensureSobol(params.sample);

    // Copy host parameters to device
    paramsBuffer.copy_from_host(&params, 1);
    check(cudaDeviceSynchronize()); // Wait for the copy to finish

    if (!scene.isEmpty() && enableTraining) train();

    switch (params.inferenceMode) {
        case InferenceMode::NO_INFERENCE:
            check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer.data()), sizeof(Params), &sbts[REFERENCE], dim.x, dim.y, 1));
            check(cudaDeviceSynchronize()); // Wait for the renderer to finish
            break;
        case InferenceMode::RAW_PHOTON_MAP:
            check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer.data()), sizeof(Params), &sppmVisSBT, dim.x, dim.y, 1));
            check(cudaDeviceSynchronize()); // Wait for the renderer to finish
            break;
        case InferenceMode::PHOTON_MAPPING:
            sppmBVH.resize(dim.x * dim.y);
            params.photonMap = sppmBVH.getDeviceView(); // Update handle in params
            paramsBuffer.copy_from_host(&params, 1);
            check(cudaDeviceSynchronize()); // Wait for the copy to finish
            check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer.data()), sizeof(Params), &sbts[SPPM_FULL], dim.x, dim.y, 1));
            check(cudaDeviceSynchronize()); // Wait for the renderer to finish
            sppmBVH.updatePhotonAS(context);
            check(cudaDeviceSynchronize());
            if (photonCount) check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer.data()), sizeof(Params), &sbts[SPPM_LIGHT_PASS], photonCount, 1, 1));
            check(cudaDeviceSynchronize()); // Wait for the renderer to finish
            sppmBVH.totalPhotonCount += photonCount;
            params.photonMap = sppmBVH.getDeviceView(); // Update handle in params
            paramsBuffer.copy_from_host(&params, 1);
            check(cudaDeviceSynchronize()); // Wait for the copy to finish
            accumulatePhotonsToFramebuffer<<<dim3((dim.x + 15) / 16, (dim.y + 15) / 16), dim3(16, 16)>>>(paramsBuffer.data());
            params.trainingRound++;
            break;
        default:
            check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer.data()), sizeof(Params), &sbts[INFERENCE], dim.x, dim.y, 1));
            check(cudaDeviceSynchronize()); // Wait for the renderer to finish

            nrcModel.network->inference(nrcInferenceInput, nrcInferenceOutput);

            dim3 block(16, 16);
            dim3 grid((dim.x + block.x - 1) / block.x, (dim.y + block.y - 1) / block.y);
            visualizeInference<<<grid, block>>>(paramsBuffer.data());
            check(cudaDeviceSynchronize()); // Wait for the visualization to finish
            break;
    }

    params.sample++;
    params.weight = 1.0f / static_cast<float>(params.sample);
}

void OptixRenderer::generateSobol(uint offset, uint n) {
    randSequence.resize(n * RAND_SEQUENCE_DIMS);

    params.sequenceStride = n;
    params.sequenceOffset = offset;
    params.randSequence = randSequence.data();

    // NOTE: We rebuild the generator, this makes regeneration slow but saves memory
    curandGenerator_t generator;
    check(curandCreateGenerator(&generator, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
    check(curandSetQuasiRandomGeneratorDimensions(generator, RAND_SEQUENCE_DIMS)); // 4 dimensions for 4D Sobol sequence
    check(curandSetGeneratorOffset(generator, offset)); // Reset the sequence
    check(curandGenerateUniform(generator, randSequence.data(), randSequence.size()));
    check(cudaDeviceSynchronize()); // Wait for the generator to finish
    check(curandDestroyGenerator(generator));
}

void OptixRenderer::ensureSobol(uint sample) {
    if (sample < params.sequenceOffset || sample >= params.sequenceOffset + params.sequenceStride) {
        // std::cout << std::format("Regenerating Sobol sequence for samples [{},{})", sample, sample + RAND_SEQUENCE_CACHE_SIZE) << std::endl;
        // C++17
        std::cout << "Regenerating Sobol sequence for samples [" << sample << "," << sample + RAND_SEQUENCE_CACHE_SIZE << ")" << std::endl;
        generateSobol(sample, RAND_SEQUENCE_CACHE_SIZE);
    }
}

void OptixRenderer::resize(uvec2 dim) {
    // Generate inference input and output buffers
    auto inferenceBatchSize = dim.x * dim.y;
    inferenceBatchSize += tcnn::BATCH_SIZE_GRANULARITY - inferenceBatchSize % tcnn::BATCH_SIZE_GRANULARITY; // Round up to the next multiple of BATCH_SIZE_GRANULARITY

    nrcInferenceInput = tcnn::GPUMatrix<float>(NRC_INPUT_SIZE, inferenceBatchSize);
    nrcInferenceOutput = tcnn::GPUMatrix<float>(NRC_OUTPUT_SIZE, inferenceBatchSize);
    nrcInferenceThroughput = tcnn::GPUMemory<float3>(inferenceBatchSize);

    // Generate the Cranley-Patterson-Rotation per pixel
    // NOTE: We rebuild the generator on resize, this makes resize slow but saves memory
    rotationTable.resize(dim.x * dim.y);

    curandGenerator_t generator;
    check(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW));
    check(curandGenerateUniform(generator, reinterpret_cast<float*>(rotationTable.data()), rotationTable.size() * 2));
    check(curandDestroyGenerator(generator));

    params.inferenceInput = nrcInferenceInput.data();
    params.inferenceOutput = nrcInferenceOutput.data();
    params.inferenceThroughput = nrcInferenceThroughput.data();
    params.rotationTable = rotationTable.data();

    check(cudaDeviceSynchronize()); // Wait for the generator to finish
}

void OptixRenderer::reset() {
    params.sample = 0;
    params.weight = 1.0f;
}

void OptixRenderer::resetNRC() {
    nrcModel.trainer->initialize_params();
    lossHistory.clear();
}