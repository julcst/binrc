#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

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
#include "optix/sppm_as.cuh"

enum ProgramGroup {
// Raygeneration programs
    REFERENCE,
    TRAIN_EYE,
    TRAIN_LIGHT,
    TRAIN_BIDIR,
    INFERENCE,
    SPPM_EYE_PASS,
    SPPM_LIGHT_PASS,
    SPPM_VIS_RAYGEN,
    SPPM_FULL,
// Other
    MISS,
    CLOSEST_HIT,
    SPPM_RTX,
    NO_MISS,
    SPPM_VIS_HIT,
};

constexpr size_t RAYGEN_COUNT = 9;
constexpr size_t PROGRAM_GROUP_COUNT = RAYGEN_COUNT + 5;

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
    float trainingDirection = 1.0f;
    float photonMappingAmount = 0.0f; // Propoertion of backward samples that are generated using photon mapping
    float photonQueryReplacement = 0.5f; // Proportion of photon queries that are kept between frames
    uint32_t photonCount = 1 << 17; // Number of photons to generate
    Params params = {}; // NOTE: Initialization is important to prevent invalid pointers
    SPPMRTX sppmBVH {NRC_BATCH_SIZE};

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

    tcnn::GPUMemory<std::array<TrainBounce, TRAIN_DEPTH>> selfLearningBounces {NRC_BATCH_SIZE};
    tcnn::GPUMatrix<float> selfLearningInference {NRC_OUTPUT_SIZE, NRC_BATCH_SIZE};
    tcnn::GPUMatrix<float> selfLearningQueries {NRC_INPUT_SIZE, NRC_BATCH_SIZE};

    BRDFLUT brdfLUT;

    thrust::device_vector<HeaderOnlyRecord> sppmVisRecords;
    OptixShaderBindingTable sppmVisSBT;

    void generateSobol(uint offset, uint n);
    void ensureSobol(uint sample);
    void train();
};