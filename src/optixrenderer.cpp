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
#include <vector>
#include <fstream>

#include "optixir.hpp"
#include "cudautil.hpp"

using uint = unsigned int;

std::vector<char> readBinaryFile(const std::filesystem::path& filepath) {
    std::ifstream stream{filepath, std::ios::binary};
    std::cout << "Loading " << std::filesystem::absolute(filepath) << std::endl;
    if (stream.fail()) throw std::runtime_error("Could not open file: " + std::filesystem::absolute(filepath).string());
    return std::vector<char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
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
        .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
        .numPayloadValues = 3,
        .numAttributeValues = 2,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "params",
        .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE),
    };

    OptixModule module = nullptr;
    const auto source = readBinaryFile(optixir::optixpathtracer_path);
    check(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions, source.data(), source.size(), nullptr, nullptr, &module));

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
    std::array<OptixProgramGroup, programDecriptions.size()> programGroups;
    check(optixProgramGroupCreate(context, programDecriptions.data(), programDecriptions.size(), &pgOptions, nullptr, nullptr, programGroups.data()));

    // Create pipeline
    OptixPipelineLinkOptions pipelineLinkOptions = {
        .maxTraceDepth = 1,
    };
    check(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), programGroups.size(), nullptr, nullptr, &pipeline));

    // Set up shader binding table
    check(cudaMallocManaged(reinterpret_cast<void**>(&raygenRecord), sizeof(RaygenRecord)));
    check(optixSbtRecordPackHeader(programGroups[0], reinterpret_cast<void*>(raygenRecord)));

    check(cudaMallocManaged(reinterpret_cast<void**>(&missRecord), sizeof(MissRecord)));
    check(optixSbtRecordPackHeader(programGroups[1], reinterpret_cast<void*>(missRecord)));

    check(cudaMallocManaged(reinterpret_cast<void**>(&hitRecord), sizeof(HitRecord)));
    check(optixSbtRecordPackHeader(programGroups[2], reinterpret_cast<void*>(hitRecord)));

    sbt = {
        .raygenRecord = reinterpret_cast<CUdeviceptr>(raygenRecord),
        .missRecordBase = reinterpret_cast<CUdeviceptr>(missRecord),
        .missRecordStrideInBytes = sizeof(MissRecord),
        .missRecordCount = 1,
        .hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitRecord),
        .hitgroupRecordStrideInBytes = sizeof(HitRecord),
        .hitgroupRecordCount = 1,
    };

    check(cudaMallocManaged(reinterpret_cast<void**>(&params), sizeof(Params)));
}

OptixRenderer::~OptixRenderer() {
    check(cudaFree(reinterpret_cast<void*>(raygenRecord)));
    check(cudaFree(reinterpret_cast<void*>(params)));
    check(optixPipelineDestroy(pipeline));
    check(optixDeviceContextDestroy(context));
}

void OptixRenderer::buildGAS(const std::vector<float3>& vertices, const std::vector<uint3>& indices) {
    // Move data to GPU
    CUdeviceptr d_vertices, d_indices;
    check(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices.size() * sizeof(float3)));
    check(cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));
    check(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices.size() * sizeof(uint3)));
    check(cudaMemcpy(reinterpret_cast<void*>(d_indices), indices.data(), indices.size() * sizeof(uint3), cudaMemcpyHostToDevice));

    OptixAccelBuildOptions accelOptions = {
        .buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        .operation = OPTIX_BUILD_OPERATION_BUILD,
    };
    std::array<uint, 1> flags = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput buildInput = {
        .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
        .triangleArray = {
            .vertexBuffers = &d_vertices,
            .numVertices = static_cast<unsigned int>(vertices.size()),
            .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
            .vertexStrideInBytes = sizeof(float3),
            .indexBuffer = d_indices,
            .numIndexTriplets = static_cast<unsigned int>(indices.size()),
            .indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
            .indexStrideInBytes = sizeof(uint3),
            .flags = flags.data(),
            .numSbtRecords = 1,
        },
    };

    // Allocate memory for acceleration structure
    OptixAccelBufferSizes bufferSizes;
    check(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &bufferSizes));
    CUdeviceptr d_tempBuffer, d_outputBuffer;
    check(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), bufferSizes.tempSizeInBytes));
    check(cudaMalloc(reinterpret_cast<void**>(&d_outputBuffer), bufferSizes.outputSizeInBytes));

    OptixTraversableHandle gasHandle;
    optixAccelBuild(context, nullptr, &accelOptions, &buildInput, 1, d_tempBuffer, bufferSizes.tempSizeInBytes, d_outputBuffer, bufferSizes.outputSizeInBytes, &gasHandle, nullptr, 0);

    check(cudaFree(reinterpret_cast<void*>(d_vertices)));

    params->handle = gasHandle;
}

void OptixRenderer::render(float4* image, uint2 dim) {
    params->image = image;
    params->dim = dim;

    check(optixLaunch(pipeline, nullptr, reinterpret_cast<CUdeviceptr>(params), sizeof(Params), &sbt, dim.x, dim.y, 1));
}