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
    TRAIN_LIGHT_NAIVE,
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

NLOHMANN_JSON_SERIALIZE_ENUM(ProgramGroup, {
    {REFERENCE, "reference"},
    {TRAIN_EYE, "train_eye"},
    {TRAIN_LIGHT, "train_light"},
    {TRAIN_LIGHT_NAIVE, "train_light_naive"},
    {TRAIN_BIDIR, "train_bidir"},
    {INFERENCE, "inference"},
    {SPPM_EYE_PASS, "sppm_eye_pass"},
    {SPPM_LIGHT_PASS, "sppm_light_pass"},
    {SPPM_VIS_RAYGEN, "sppm_vis_raygen"},
    {SPPM_FULL, "sppm_full"},
    {MISS, "miss"},
    {CLOSEST_HIT, "closest_hit"},
    {SPPM_RTX, "sppm_rtx"},
    {NO_MISS, "no_miss"},
    {SPPM_VIS_HIT, "sppm_vis_hit"}
});

struct FrameBreakdown {
    float photonQueryGeneration = 0.0f;
    float photonQueryMapBuildTime = 0.0f;
    float photonGeneration = 0.0f;
    float photonPostprocessing = 0.0f;
    float forwardSampleGeneration = 0.0f;
    float backwardSampleGeneration = 0.0f;
    float balanceSampleGeneration = 0.0f;
    float selfLearningInference = 0.0f;
    float selfLearningPostprocessing = 0.0f;
    float training = 0.0f;
    float pathtracing = 0.0f;
    float inference = 0.0f;
    float visualization = 0.0f;
    float total = 0.0f;

    void operator+=(const FrameBreakdown& other) {
        photonQueryGeneration += other.photonQueryGeneration;
        photonQueryMapBuildTime += other.photonQueryMapBuildTime;
        photonGeneration += other.photonGeneration;
        photonPostprocessing += other.photonPostprocessing;
        forwardSampleGeneration += other.forwardSampleGeneration;
        backwardSampleGeneration += other.backwardSampleGeneration;
        balanceSampleGeneration += other.balanceSampleGeneration;
        selfLearningInference += other.selfLearningInference;
        selfLearningPostprocessing += other.selfLearningPostprocessing;
        training += other.training;
        pathtracing += other.pathtracing;
        inference += other.inference;
        visualization += other.visualization;
        total += other.total;
    }

    FrameBreakdown operator*(float factor) const {
        return {
            photonQueryGeneration * factor,
            photonQueryMapBuildTime * factor,
            photonGeneration * factor,
            photonPostprocessing * factor,
            forwardSampleGeneration * factor,
            backwardSampleGeneration * factor,
            balanceSampleGeneration * factor,
            selfLearningInference * factor,
            selfLearningPostprocessing * factor,
            training * factor,
            pathtracing * factor,
            inference * factor,
            visualization * factor,
            total * factor,
        };
    }
};

struct AverageFrameBreakdown {
    FrameBreakdown sum;
    uint32_t count = 0;
    void add(const FrameBreakdown& breakdown) {
        sum += breakdown;
        count++;
    }
    FrameBreakdown average() const {
        return count > 1 ? sum * (1.0f / count) : sum;
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FrameBreakdown,
    photonQueryGeneration, photonQueryMapBuildTime, photonGeneration, photonPostprocessing,
    forwardSampleGeneration, backwardSampleGeneration, balanceSampleGeneration,
    selfLearningInference, selfLearningPostprocessing, training, pathtracing, inference,
    visualization, total
);

constexpr size_t RAYGEN_COUNT = 10;
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
    FrameBreakdown render(vec4* image, uvec2 dim);
    void setCamera(const mat4& clipToWorld);
    void loadGLTF(const std::filesystem::path& path);
    void resize(uvec2 dim);
    void configure(const nlohmann::json& config);
    nlohmann::json getConfig() const;

    Scene scene;
    std::vector<float> lossHistory;
    bool enableTraining = false;
    float trainingDirection = 1.0f;
    float photonMappingAmount = 0.0f; // Propoertion of backward samples that are generated using photon mapping
    float photonQueryReplacement = 0.5f; // Proportion of photon queries that are kept between frames
    uint32_t photonCount = 1 << 17; // Number of photons to generate
    Params params = {}; // NOTE: Initialization is important to prevent invalid pointers
    SPPMRTX sppmBVH {NRC_BATCH_SIZE};
    ProgramGroup backwardTrainer = TRAIN_LIGHT;
    bool useJITFusion = true;
    bool useFusedInference = true;

private:
    OptixDeviceContext context;
    OptixPipeline pipeline;

    std::array<OptixModule, optixir::paths.size()> modules;
    std::array<OptixShaderBindingTable, RAYGEN_COUNT> sbts;
    std::array<OptixProgramGroup, PROGRAM_GROUP_COUNT> programGroups;
    std::array<CudaEvent, 23> events;

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

    std::unique_ptr<tcnn::CudaRtcKernel> fusedInferenceKernel;

    tcnn::GPUMemory<std::array<TrainBounce, TRAIN_DEPTH>> selfLearningBounces {NRC_BATCH_SIZE};
    tcnn::GPUMatrix<float> selfLearningInference {NRC_OUTPUT_SIZE, NRC_BATCH_SIZE};
    tcnn::GPUMatrix<float> selfLearningQueries {NRC_INPUT_SIZE, NRC_BATCH_SIZE};

    BRDFLUT brdfLUT;

    thrust::device_vector<HeaderOnlyRecord> sppmVisRecords;
    OptixShaderBindingTable sppmVisSBT;

    void generateSobol(uint offset, uint n);
    void ensureSobol(uint sample);
    void train();
    void loadModel(const tcnn::TrainableModel& model);
};