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

std::vector<char> readBinaryFile(const std::filesystem::path& filepath) {
    std::ifstream stream{filepath, std::ios::binary};
    std::cout << "Loading " << std::filesystem::absolute(filepath) << std::endl;
    if (stream.fail()) throw std::runtime_error("Could not open file: " + std::filesystem::absolute(filepath).string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

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
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // Disable optimizations
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL; // Generate debug information
#endif
    OptixPipelineCompileOptions pipelineCompileOptions = {
        .usesMotionBlur = false,
        .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        .numPayloadValues = PAYLOAD_SIZE,
        .numAttributeValues = 2,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "params",
        .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE),
    };

    OptixModule combinedModule, hitModule = nullptr;
    const auto combined = readBinaryFile(optixir::paths[0]);
    check(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions, combined.data(), combined.size(), nullptr, nullptr, &combinedModule));
    const auto hit = readBinaryFile(optixir::paths[1]);
    check(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions, hit.data(), hit.size(), nullptr, nullptr, &hitModule));

    // Create program groups
    OptixProgramGroupOptions pgOptions = {};
    std::array programDecriptions = {
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = combinedModule,
                .entryFunctionName = "__raygen__rg",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
            .miss = {
                .module = hitModule,
                .entryFunctionName = "__miss__ms",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            .hitgroup = {
                .moduleCH = hitModule,
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

    // Set up shader binding table
    RaygenRecord raygenRecord;
    check(optixSbtRecordPackHeader(programGroups[0], &raygenRecord));
    CUdeviceptr raygenRecordDevice;
    check(cudaMalloc(reinterpret_cast<void**>(&raygenRecordDevice), sizeof(RaygenRecord)));
    check(cudaMemcpy(reinterpret_cast<void*>(raygenRecordDevice), &raygenRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice));

    MissRecord missRecord;
    check(optixSbtRecordPackHeader(programGroups[1], &missRecord));
    CUdeviceptr missRecordDevice;
    check(cudaMalloc(reinterpret_cast<void**>(&missRecordDevice), sizeof(MissRecord)));
    check(cudaMemcpy(reinterpret_cast<void*>(missRecordDevice), &missRecord, sizeof(MissRecord), cudaMemcpyHostToDevice));

    sbt = {
        .raygenRecord = raygenRecordDevice,
        .missRecordBase = missRecordDevice,
        .missRecordStrideInBytes = sizeof(MissRecord),
        .missRecordCount = 1,
        .hitgroupRecordBase = 0,
        .hitgroupRecordStrideInBytes = sizeof(HitRecord),
        .hitgroupRecordCount = 0,
    };

    check(cudaMallocManaged(reinterpret_cast<void**>(&params), sizeof(Params)));
    check(cudaMemset(params, 0, sizeof(Params)));
    initParams(params);

    nrcModel = tcnn::create_from_config(NRC_INPUT_SIZE, NRC_OUTPUT_SIZE, NRC_CONFIG);
    nrcTrainInput = tcnn::GPUMatrix<float>(NRC_INPUT_SIZE, NRC_BATCH_SIZE);
    nrcTrainOutput = tcnn::GPUMatrix<float>(NRC_OUTPUT_SIZE, NRC_BATCH_SIZE);

    std::cout << "Network: " << std::setw(2) << nrcModel.network->hyperparams()
              << "\nTrainer: " << std::setw(2) << nrcModel.trainer->hyperparams()
              << std::endl;

    params->trainingInput = nrcTrainInput.data();
    params->trainingTarget = nrcTrainOutput.data();

    nrcTrainIndex = tcnn::GPUMemory<uint>(1, true);
    nrcTrainIndex.memset(0);
    params->trainingIndexPtr = nrcTrainIndex.data();
}

OptixRenderer::~OptixRenderer() {
    check(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    check(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
    check(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
    check(cudaFree(reinterpret_cast<void*>(params->randSequence)));
    check(cudaFree(reinterpret_cast<void*>(params->rotationTable)));
    check(cudaFree(reinterpret_cast<void*>(params->materials)));
    check(cudaFree(reinterpret_cast<void*>(params->lightTable)));
    check(cudaFree(reinterpret_cast<void*>(params)));
    check(optixPipelineDestroy(pipeline));
    check(optixDeviceContextDestroy(context));
}

void OptixRenderer::loadGLTF(const std::filesystem::path& path) {
    scene.loadGLTF(context, params, programGroups[2], sbt, path);
    const auto aabb = scene.getAABB();
    params->sceneMin = {aabb.min.x, aabb.min.y, aabb.min.z};
    const auto size = aabb.max - aabb.min;
    params->sceneScale = 1.0f / std::max(size.x, std::max(size.y, size.z));
    std::cout << "Min: (" << params->sceneMin.x << ", " << params->sceneMin.y << ", " << params->sceneMin.z << ") Scale: " << params->sceneScale << std::endl;
    reset();
    lossHistory.clear();
}

void OptixRenderer::setCamera(const mat4& clipToWorld) {
    params->clipToWorld = glmToCuda(clipToWorld);
}

__global__ void visualizeInference(Params* params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params->dim.x || y >= params->dim.y) return;
    const int i = y * params->dim.x + x;
    const int idxIn = i * NRC_INPUT_SIZE;
    const int idxOut = i * NRC_OUTPUT_SIZE;
    auto inference = make_float3(params->inferenceOutput[idxOut + 0], params->inferenceOutput[idxOut + 1], params->inferenceOutput[idxOut + 2]);
    if (!isfinite(inference)) return;
    if (params->inferenceMode == InferenceMode::RAW_CACHE) {
        params->image[i] = make_float4(inference, 1.0f);
    } else {
        const auto diffuse = make_float3(params->inferenceInput[idxIn + 8], params->inferenceInput[idxIn + 9], params->inferenceInput[idxIn + 10]);
        const auto specular = make_float3(params->inferenceInput[idxIn + 11], params->inferenceInput[idxIn + 12], params->inferenceInput[idxIn + 13]);
        const auto throughput = params->inferenceThroughput[i];
        params->image[i] += params->weight * make_float4(inference * (diffuse + specular) * throughput, 1.0f);
    }
}

void OptixRenderer::render(vec4* image, uvec2 dim) {
    params->image = reinterpret_cast<float4*>(image);
    params->dim = make_uint2(dim.x, dim.y);
    const auto prevTrainIndex = nrcTrainIndex.at(0);
    
    ensureSobol(params->sample);
    check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params), sizeof(Params), &sbt, dim.x, dim.y, 1));
    check(cudaDeviceSynchronize()); // Wait for the renderer to finish

    if (!scene.isEmpty()) train();

    if (params->inferenceMode != InferenceMode::NO_INFERENCE) {
        nrcModel.network->inference(nrcInferenceInput, nrcInferenceOutput);

        dim3 block(16, 16);
        dim3 grid((dim.x + block.x - 1) / block.x, (dim.y + block.y - 1) / block.y);
        visualizeInference<<<grid, block>>>(params);
        check(cudaDeviceSynchronize()); // Wait for the visualization to finish
    }
    
    params->sample++;
    params->weight = 1.0f / static_cast<float>(params->sample);
    //std::cout << nrcTrainIndex.at(0) - prevTrainIndex << std::endl;
}

void OptixRenderer::generateSobol(uint offset, uint n) {
    // NOTE: We rebuild the generator, this makes regeneration slow but saves memory
    const uint nfloats = n * RAND_SEQUENCE_DIMS;
    params->sequenceStride = n;
    params->sequenceOffset = offset;
    curandGenerator_t generator;
    check(curandCreateGenerator(&generator, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
    check(curandSetQuasiRandomGeneratorDimensions(generator, RAND_SEQUENCE_DIMS)); // 4 dimensions for 4D Sobol sequence
    check(curandSetGeneratorOffset(generator, offset)); // Reset the sequence
    check(cudaFree(reinterpret_cast<void*>(params->randSequence)));
    check(cudaMalloc(reinterpret_cast<void**>(&params->randSequence), nfloats * sizeof(float)));
    check(curandGenerateUniform(generator, reinterpret_cast<float*>(params->randSequence), nfloats));
    check(cudaDeviceSynchronize()); // Wait for the generator to finish
    check(curandDestroyGenerator(generator));
}

void OptixRenderer::ensureSobol(uint sample) {
    if (sample < params->sequenceOffset || sample >= params->sequenceOffset + params->sequenceStride) {
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
    params->inferenceInput = nrcInferenceInput.data();
    params->inferenceOutput = nrcInferenceOutput.data();
    params->inferenceThroughput = nrcInferenceThroughput.data();

    // Generate the Cranley-Patterson-Rotation per pixel
    // NOTE: We rebuild the generator on resize, this makes resize slow but saves memory
    const size_t n = static_cast<size_t>(dim.x * dim.y) * 4;
    curandGenerator_t generator;
    check(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW));
    check(cudaFree(reinterpret_cast<void*>(params->rotationTable)));
    check(cudaMalloc(reinterpret_cast<void**>(&params->rotationTable), n * sizeof(float)));
    check(curandGenerateUniform(generator, reinterpret_cast<float*>(params->rotationTable), n));
    check(cudaDeviceSynchronize()); // Wait for the generator to finish
    check(curandDestroyGenerator(generator));
}

void OptixRenderer::reset() {
    params->sample = 0;
    params->weight = 1.0f;
}

void OptixRenderer::resetNRC() {
    nrcModel.trainer->initialize_params();
    lossHistory.clear();
}

void OptixRenderer::train() {
    for (uint32_t offset = 0; offset < NRC_BATCH_SIZE; offset += NRC_SUBBATCH_SIZE) {
        auto ctx = nrcModel.trainer->training_step(nrcTrainInput.slice_cols(offset, NRC_SUBBATCH_SIZE), nrcTrainOutput.slice_cols(offset, NRC_SUBBATCH_SIZE));
        float loss = nrcModel.trainer->loss(*ctx);
        lossHistory.push_back(loss);
    }
}