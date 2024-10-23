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

// OPTIX check as constexpr function
constexpr void check(OptixResult res) {
    if (res != OPTIX_SUCCESS) {
        throw std::runtime_error(optixGetErrorName(res));
    }
}

// CUDA check as constexpr function
constexpr void check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorName(error));
    }
}

OptixRenderer::OptixRenderer() {
    check(cudaFree(nullptr)); // Initialize CUDA for this device on this thread
    check(optixInit()); // Initialize the OptiX API
    
    // Initialize the OptiX device context
    OptixDeviceContextOptions options = {};
#ifndef NDEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL; // Enable all validation checks
    options.logCallbackLevel = 4; // Print all log messages
    options.logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void*) {
        std::cerr << "[" << tag << "] " << message << std::endl;
    };
#endif
    CUcontext cuCtx = nullptr; // zero means take the current context
    check(optixDeviceContextCreate(cuCtx, &options, &context));

    // Create module
    OptixModuleCompileOptions moduleCompileOptions = {
        .maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, //OPTIX_COMPILE_OPTIMIZATION_LEVEL_3
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL, //OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL
        .numPayloadTypes = 0,
        .payloadTypes = nullptr,
    };
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
    std::array<char, 2048> logString;
    auto logSize = logString.size();
    check(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions, source.c_str(), source.size(), logString.data(), &logSize, &module));
    std::cerr << "Module Log: " << logString.data() << std::endl;

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
    logSize = logString.size();
    check(optixProgramGroupCreate(context, &pgDesc, 1, &pgOptions, logString.data(), &logSize, &programGroup));
    std::cerr << "Program Group Log: " << logString.data() << std::endl;

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

    sbt = {};
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygenRecord);
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(missRecord);
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);

    check(cudaMallocManaged(reinterpret_cast<void**>(&params), sizeof(Params)));
}

OptixRenderer::~OptixRenderer() {
    check(cudaFree(reinterpret_cast<void*>(raygenRecord)));
    check(cudaFree(reinterpret_cast<void*>(params)));
    check(optixPipelineDestroy(pipeline));
    check(optixDeviceContextDestroy(context));
}

void OptixRenderer::render(uchar4* image, int width, int height) {
    params->image = image;
    params->dim = make_uint2(width, height);

    check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params), sizeof(Params), &sbt, width, height, 1));

    check(cudaDeviceSynchronize()); // Necessary?
}