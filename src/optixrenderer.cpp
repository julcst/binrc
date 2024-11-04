#include "optixrenderer.hpp"

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

#include "optixir.hpp"
#include "cudautil.hpp"

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
#ifndef NDEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL; // Enable all validation checks
#endif
    CUcontext cuCtx = nullptr; // zero means take the current context
    check(optixDeviceContextCreate(cuCtx, &options, &context));

    // Create module
    OptixModuleCompileOptions moduleCompileOptions = {
        .maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL,
        .numPayloadTypes = 0,
        .payloadTypes = nullptr,
    };
#ifndef NDEBUG
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

    OptixModule module = nullptr;
    const std::string source = Common::readFile(optixir::optixpathtracer_path);
    check(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions, source.c_str(), source.size(), nullptr, nullptr, &module));

    // Create program groups
    OptixProgramGroupOptions pgOptions = {};
    std::array programDecriptions = {
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = module,
                .entryFunctionName = "__raygen__rg",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
            .miss = {
                .module = module,
                .entryFunctionName = "__miss__ms",
            },
        },
        OptixProgramGroupDesc {
            .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            .hitgroup = {
                .moduleCH = module,
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
    check(cudaMallocManaged(reinterpret_cast<void**>(&raygenRecord), sizeof(RaygenRecord)));
    check(optixSbtRecordPackHeader(programGroups[0], reinterpret_cast<void*>(raygenRecord)));

    check(cudaMallocManaged(reinterpret_cast<void**>(&missRecord), sizeof(MissRecord)));
    check(optixSbtRecordPackHeader(programGroups[1], reinterpret_cast<void*>(missRecord)));

    sbt = {
        .raygenRecord = reinterpret_cast<CUdeviceptr>(raygenRecord),
        .missRecordBase = reinterpret_cast<CUdeviceptr>(missRecord),
        .missRecordStrideInBytes = sizeof(MissRecord),
        .missRecordCount = 1,
        .hitgroupRecordBase = 0,
        .hitgroupRecordStrideInBytes = sizeof(HitRecord),
        .hitgroupRecordCount = 1,
    };

    check(cudaMallocManaged(reinterpret_cast<void**>(&params), sizeof(Params)));
    params->image = nullptr;
    params->randSequence = nullptr;
    params->rotationTable = nullptr;
    params->russianRouletteWeight = 3.0f;
}

OptixRenderer::~OptixRenderer() {
    check(cudaFree(reinterpret_cast<void*>(raygenRecord)));
    check(cudaFree(reinterpret_cast<void*>(missRecord)));
    check(cudaFree(reinterpret_cast<void*>(params->randSequence)));
    check(cudaFree(reinterpret_cast<void*>(params->rotationTable)));
    check(cudaFree(reinterpret_cast<void*>(params)));
    check(optixPipelineDestroy(pipeline));
    check(optixDeviceContextDestroy(context));
}

void OptixRenderer::loadGLTF(const std::filesystem::path& path) {
    scene.loadGLTF(context, params, programGroups[2], sbt, path);
    reset();
}

void OptixRenderer::setCamera(const mat4& clipToWorld) {
    params->clipToWorld = clipToWorld;
    reset();
}

void OptixRenderer::render(vec4* image, uvec2 dim) {
    params->image = image;
    params->dim = dim;

    ensureSobol(params->sample);
    check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params), sizeof(Params), &sbt, dim.x, dim.y, 1));

    params->sample++;
    params->weight = 1.0f / static_cast<float>(params->sample);
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
    check(curandDestroyGenerator(generator));
}

void OptixRenderer::ensureSobol(uint sample) {
    if (sample < params->sequenceOffset || sample >= params->sequenceOffset + params->sequenceStride) {
        std::cout << std::format("Regenerating Sobol sequence for samples [{},{})", sample, sample + RAND_SEQUENCE_CACHE_SIZE) << std::endl;
        generateSobol(sample, RAND_SEQUENCE_CACHE_SIZE);
    }
}

void OptixRenderer::resize(uvec2 dim) {
    // NOTE: We rebuild the generator on resize, this makes resize slow but saves memory
    curandGenerator_t generator;
    check(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW));
    check(cudaFree(reinterpret_cast<void*>(params->rotationTable)));
    size_t n = static_cast<size_t>(dim.x * dim.y) * 4;
    check(cudaMalloc(reinterpret_cast<void**>(&params->rotationTable), n * sizeof(float)));
    check(curandGenerateUniform(generator, reinterpret_cast<float*>(params->rotationTable), n));
    check(curandDestroyGenerator(generator));
    reset();
}

void OptixRenderer::reset() {
    params->sample = 0;
    params->weight = 1.0f;
}