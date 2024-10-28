#include "optixrenderer.hpp"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_host.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_types.h>

#include <framework/common.hpp>

#include <iostream>
#include <array>

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
        .numPayloadValues = 2,
        .numAttributeValues = 2,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "params",
        .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE),
    };

    OptixModule module = nullptr;
    const std::string source = Common::readFile(optixir::optixpathtracer_path);
    check(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions, source.c_str(), source.size(), nullptr, nullptr, &module));

    // Create program group
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        .raygen = {
            .module = module,
            .entryFunctionName = "__raygen__rg",
        },
    };

    OptixProgramGroup programGroup = nullptr;
    check(optixProgramGroupCreate(context, &pgDesc, 1, &pgOptions, nullptr, nullptr, &programGroup));

    // Create pipeline
    OptixPipelineLinkOptions pipelineLinkOptions = {
        .maxTraceDepth = 1,
    };
    check(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions, &programGroup, 1, nullptr, nullptr, &pipeline));

    // Set up shader binding table
    check(cudaMallocManaged(reinterpret_cast<void**>(&raygenRecord), sizeof(RaygenRecord)));
    check(optixSbtRecordPackHeader(programGroup, reinterpret_cast<void*>(raygenRecord)));

    check(cudaMallocManaged(reinterpret_cast<void**>(&missRecord), sizeof(MissRecord)));
    check(optixSbtRecordPackHeader(programGroup, reinterpret_cast<void*>(missRecord)));

    sbt = {
        .raygenRecord = reinterpret_cast<CUdeviceptr>(raygenRecord),
        .missRecordBase = reinterpret_cast<CUdeviceptr>(missRecord),
        .missRecordStrideInBytes = sizeof(MissRecord),
        .missRecordCount = 1,
    };

    check(cudaMallocManaged(reinterpret_cast<void**>(&params), sizeof(Params)));
}

OptixRenderer::~OptixRenderer() {
    check(cudaFree(reinterpret_cast<void*>(raygenRecord)));
    check(cudaFree(reinterpret_cast<void*>(params)));
    check(optixPipelineDestroy(pipeline));
    check(optixDeviceContextDestroy(context));
}

void OptixRenderer::render(float4* image, int width, int height) {
    params->image = image;
    params->dim = make_uint2(width, height);

    check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params), sizeof(Params), &sbt, width, height, 1));
}