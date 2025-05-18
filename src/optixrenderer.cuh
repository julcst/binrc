#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_types.h>

#include <array>
#include <vector>
#include <filesystem>

#include <glm/glm.hpp>
using namespace glm;

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/config.h>

#include <json/json.hpp>

#include "optix/params.cuh"
#include "scene.hpp"
#include "optixir.hpp"
#include "brdflut.cuh"

enum ProgramGroup {
    COMBINED,
    REFERENCE,
    TRAIN_FORWARD,
    TRAIN_BACKWARD,
    INFERENCE,
    MISS,
    CLOSEST_HIT,
};

constexpr size_t RAYGEN_COUNT = 5;
constexpr size_t PROGRAM_GROUP_COUNT = RAYGEN_COUNT + 2;

class OptixRenderer {
public:
    OptixRenderer();
    ~OptixRenderer();
    OptixRenderer(const OptixRenderer&) = delete;
    OptixRenderer& operator=(const OptixRenderer&) = delete;
    OptixRenderer(OptixRenderer&&) = delete;
    OptixRenderer& operator=(OptixRenderer&&) = delete;

    void reset();
    void resetNRC();
    void render(vec4* image, uvec2 dim);
    void setCamera(const mat4& clipToWorld);
    void loadGLTF(const std::filesystem::path& path);
    void resize(uvec2 dim);

    Scene scene;
    std::vector<float> lossHistory;
    bool enableTraining = false;
    float trainingDirection = 0.5f;
    Params params = {}; // NOTE: Initialization is important to prevent invalid pointers

private:
    OptixDeviceContext context;
    OptixPipeline pipeline;

    std::array<OptixModule, optixir::paths.size()> modules;
    std::array<OptixShaderBindingTable, RAYGEN_COUNT> sbts;
    std::array<OptixProgramGroup, PROGRAM_GROUP_COUNT> programGroups;

    tcnn::GPUMemory<Params> paramsBuffer {1};
    tcnn::GPUMemory<HitRecord> hitRecords;
    tcnn::GPUMemory<RaygenRecord> raygenRecords;
    tcnn::GPUMemory<MissRecord> missRecords;
    tcnn::GPUMemory<float> randSequence;
    tcnn::GPUMemory<float2> rotationTable;
    tcnn::GPUMemory<Material> materials;
    tcnn::GPUMemory<EmissiveTriangle> lightTable;
    tcnn::GPUMemory<Instance> instances;

    tcnn::TrainableModel nrcModel; // TODO: Explore half inputs and outputs
    tcnn::GPUMatrix<float> nrcTrainInput;
    tcnn::GPUMatrix<float> nrcTrainOutput;
    tcnn::GPUMemory<uint> nrcTrainIndex {1};
    tcnn::GPUMemory<uint> nrcLightSamples {1};
    tcnn::GPUMatrix<float> nrcInferenceInput;
    tcnn::GPUMatrix<float> nrcInferenceOutput;
    tcnn::GPUMemory<float3> nrcInferenceThroughput;

    BRDFLUT brdfLUT;

    void generateSobol(uint offset, uint n);
    void ensureSobol(uint sample);
    void train();
};