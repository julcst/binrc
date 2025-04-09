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

enum ProgramGroup {
    COMBINED,
    REFERENCE,
    MISS,
    CLOSEST_HIT,
};

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

    inline Params& getParams() { return params.at(0); }

private:
    OptixDeviceContext context;
    OptixPipeline pipeline;

    std::array<OptixModule, optixir::paths.size()> modules;
    std::array<OptixShaderBindingTable, 2> sbts;
    std::array<OptixProgramGroup, 4> programGroups;

    tcnn::GPUMemory<Params> params {1, true};
    tcnn::GPUMemory<HitRecord> hitRecords;
    tcnn::GPUMemory<RaygenRecord> raygenRecords;
    tcnn::GPUMemory<MissRecord> missRecords;
    tcnn::GPUMemory<float> randSequence;
    tcnn::GPUMemory<float2> rotationTable;
    tcnn::GPUMemory<Material> materials;
    tcnn::GPUMemory<EmissiveTriangle> lightTable;

    tcnn::TrainableModel nrcModel; // TODO: Explore half inputs and outputs
    tcnn::GPUMatrix<float> nrcTrainInput;
    tcnn::GPUMatrix<float> nrcTrainOutput;
    tcnn::GPUMemory<uint> nrcTrainIndex;
    tcnn::GPUMatrix<float> nrcInferenceInput;
    tcnn::GPUMatrix<float> nrcInferenceOutput;
    tcnn::GPUMemory<float3> nrcInferenceThroughput;

    void generateSobol(uint offset, uint n);
    void ensureSobol(uint sample);
    void train();
};