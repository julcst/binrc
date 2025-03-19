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

#include "optixparams.cuh"
#include "scene.hpp"

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
    
    Params* params; // NOTE: This is owned memory and must be properly freed
    Scene scene;
    std::vector<float> lossHistory;

private:
    OptixDeviceContext context;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt; // NOTE: This contains owned memory and must be properly freed
    std::array<OptixProgramGroup, 3> programGroups;
    tcnn::TrainableModel nrcModel;
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