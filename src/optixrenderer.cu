#include "optixrenderer.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <optix.h>
#include <optix_host.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_types.h>

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
        .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        .numPayloadValues = PAYLOAD_SIZE,
        .numAttributeValues = 2,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "params",
        .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE),
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
                .module = modules[optixir::COMBINED],
                .entryFunctionName = "__raygen__combined",
            },
        },
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
                .module = modules[optixir::TRAIN_FORWARD],
                .entryFunctionName = "__raygen__",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = modules[optixir::TRAIN_BACKWARD],
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
    };
    check(optixProgramGroupCreate(context, programDecriptions.data(), programDecriptions.size(), &pgOptions, nullptr, nullptr, programGroups.data()));

    // Create pipeline
    OptixPipelineLinkOptions pipelineLinkOptions = {
        .maxTraceDepth = MAX_BOUNCES,
    };
    check(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), programGroups.size(), nullptr, nullptr, &pipeline));

    // TODO: optixUtilComputeStackSizesSimplePathtracer?

    // Set up shader binding table
    std::vector<RaygenRecord> raygenRecord(sbts.size());
    check(optixSbtRecordPackHeader(programGroups[COMBINED], &raygenRecord[COMBINED]));
    check(optixSbtRecordPackHeader(programGroups[REFERENCE], &raygenRecord[REFERENCE]));
    check(optixSbtRecordPackHeader(programGroups[TRAIN_FORWARD], &raygenRecord[TRAIN_FORWARD]));
    check(optixSbtRecordPackHeader(programGroups[TRAIN_BACKWARD], &raygenRecord[TRAIN_BACKWARD]));
    check(optixSbtRecordPackHeader(programGroups[INFERENCE], &raygenRecord[INFERENCE]));
    raygenRecords.resize_and_copy_from_host(raygenRecord);

    MissRecord missRecord;
    check(optixSbtRecordPackHeader(programGroups[MISS], &missRecord));
    missRecords.resize_and_copy_from_host({missRecord});

    for (size_t i = 0; i < sbts.size(); i++) {
        sbts[i] = {
            .raygenRecord = reinterpret_cast<CUdeviceptr>(&raygenRecords[i]),
            .missRecordBase = reinterpret_cast<CUdeviceptr>(missRecords.data()),
            .missRecordStrideInBytes = sizeof(MissRecord),
            .missRecordCount = 1,
            .hitgroupRecordBase = 0,
            .hitgroupRecordStrideInBytes = sizeof(HitRecord),
            .hitgroupRecordCount = 0,
        };
    }

    params.copy_from_host({Params{}});

    nrcModel = tcnn::create_from_config(NRC_INPUT_SIZE, NRC_OUTPUT_SIZE, nlohmann::json::parse(Common::readFile("nrc.json"), nullptr, true, true));
    nrcTrainInput = tcnn::GPUMatrix<float>(NRC_INPUT_SIZE, NRC_BATCH_SIZE);
    nrcTrainOutput = tcnn::GPUMatrix<float>(NRC_OUTPUT_SIZE, NRC_BATCH_SIZE);

    std::cout << "Network: " << std::setw(2) << nrcModel.network->hyperparams()
              << "\nTrainer: " << std::setw(2) << nrcModel.trainer->hyperparams()
              << std::endl;

    getParams().trainingInput = nrcTrainInput.data();
    getParams().trainingTarget = nrcTrainOutput.data();

    nrcTrainIndex = tcnn::GPUMemory<uint>(1, true);
    nrcTrainIndex.memset(0);
    getParams().trainingIndexPtr = nrcTrainIndex.data();
}

OptixRenderer::~OptixRenderer() {
    for (auto& module : modules) check(optixModuleDestroy(module));
    check(optixPipelineDestroy(pipeline));
    check(optixDeviceContextDestroy(context));
}

void OptixRenderer::loadGLTF(const std::filesystem::path& path) {
    auto sceneData = scene.loadGLTF(context, path);
    const auto aabb = scene.getAABB();
    const auto size = aabb.max - aabb.min;

    for (auto& hitRecord : sceneData.hitRecords) optixSbtRecordPackHeader(programGroups[CLOSEST_HIT], &hitRecord);

    hitRecords.resize_and_copy_from_host(sceneData.hitRecords);
    materials.resize_and_copy_from_host(sceneData.materials);
    lightTable.resize_and_copy_from_host(sceneData.lightTable);

    for (auto& sbt : sbts) {
        sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitRecords.data());
        sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
        sbt.hitgroupRecordCount = hitRecords.size();
    }

    getParams().sceneMin = {aabb.min.x, aabb.min.y, aabb.min.z};
    getParams().sceneScale = 1.0f / std::max(size.x, std::max(size.y, size.z));
    getParams().materials = materials.data();
    getParams().lightTable = lightTable.data();
    getParams().lightTableSize = lightTable.size();
    getParams().handle = sceneData.handle;

    std::cout << "Min: (" << getParams().sceneMin.x << ", " << getParams().sceneMin.y << ", " << getParams().sceneMin.z << ") Scale: " << getParams().sceneScale << std::endl;

    reset();
    lossHistory.clear();
}

void OptixRenderer::setCamera(const mat4& clipToWorld) {
    getParams().clipToWorld = glmToCuda(clipToWorld);
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
        // params->image[i] = make_float4(throughput, 1.0f); // FIXME: Looks wrong
    }
}

void OptixRenderer::train() {
    // Generate training samples
    const auto totalTrainingSamples = NRC_BATCH_SIZE / TRAIN_DEPTH;
    const uint backwardSamples = totalTrainingSamples * trainingDirection;
    const uint forwardSamples = totalTrainingSamples - backwardSamples;
    if (forwardSamples) check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params.data()), sizeof(Params), &sbts[TRAIN_FORWARD], forwardSamples, 1, 1));
    if (backwardSamples) check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params.data()), sizeof(Params), &sbts[TRAIN_BACKWARD], backwardSamples, 1, 1));
    check(cudaDeviceSynchronize()); // Wait for the renderer to finish

    // Perform training steps
    for (uint32_t offset = 0; offset < NRC_BATCH_SIZE; offset += NRC_SUBBATCH_SIZE) {
        auto ctx = nrcModel.trainer->training_step(nrcTrainInput.slice_cols(offset, NRC_SUBBATCH_SIZE), nrcTrainOutput.slice_cols(offset, NRC_SUBBATCH_SIZE));
        float loss = nrcModel.trainer->loss(*ctx);
        lossHistory.push_back(loss);
    }
}

void OptixRenderer::render(vec4* image, uvec2 dim) {

    // Update parameters
    getParams().image = reinterpret_cast<float4*>(image);
    getParams().dim = make_uint2(dim.x, dim.y);
    ensureSobol(getParams().sample);

    if (!scene.isEmpty() && enableTraining) train();

    if (getParams().inferenceMode == InferenceMode::NO_INFERENCE) { // Reference
        check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params.data()), sizeof(Params), &sbts[REFERENCE], dim.x, dim.y, 1));
        check(cudaDeviceSynchronize()); // Wait for the renderer to finish
    } else {
        check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params.data()), sizeof(Params), &sbts[INFERENCE], dim.x, dim.y, 1));
        check(cudaDeviceSynchronize()); // Wait for the renderer to finish

        nrcModel.network->inference(nrcInferenceInput, nrcInferenceOutput);

        dim3 block(16, 16);
        dim3 grid((dim.x + block.x - 1) / block.x, (dim.y + block.y - 1) / block.y);
        visualizeInference<<<grid, block>>>(params.data());
        check(cudaDeviceSynchronize()); // Wait for the visualization to finish
    }
    
    getParams().sample++;
    getParams().weight = 1.0f / static_cast<float>(getParams().sample);
}

void OptixRenderer::generateSobol(uint offset, uint n) {
    randSequence.resize(n * RAND_SEQUENCE_DIMS);

    getParams().sequenceStride = n;
    getParams().sequenceOffset = offset;
    getParams().randSequence = randSequence.data();

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
    if (sample < getParams().sequenceOffset || sample >= getParams().sequenceOffset + getParams().sequenceStride) {
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

    getParams().inferenceInput = nrcInferenceInput.data();
    getParams().inferenceOutput = nrcInferenceOutput.data();
    getParams().inferenceThroughput = nrcInferenceThroughput.data();
    getParams().rotationTable = rotationTable.data();

    check(cudaDeviceSynchronize()); // Wait for the generator to finish
}

void OptixRenderer::reset() {
    getParams().sample = 0;
    getParams().weight = 1.0f;
}

void OptixRenderer::resetNRC() {
    nrcModel.trainer->initialize_params();
    lossHistory.clear();
}