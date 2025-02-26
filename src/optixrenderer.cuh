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

const nlohmann::json NRC_CONFIG = {
	{"loss", {
		{"otype", "RelativeL2Luminance"}
	}},
	{"optimizer", {
		{"otype", "Adam"},
		{"learning_rate", 1e-3},
	}},
	{"encoding", {
		{"otype", "HashGrid"},
		{"n_levels", 16},
		{"n_features_per_level", 2},
		{"log2_hashmap_size", 19},
		{"base_resolution", 16},
		{"per_level_scale", 2.0},
	}},
	{"network", {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", 64},
		{"n_hidden_layers", 2},
	}},
};
const uint NRC_INPUT_SIZE = 3;
const uint NRC_OUTPUT_SIZE = 3;

class OptixRenderer {
public:
    OptixRenderer();
    ~OptixRenderer();
    OptixRenderer(const OptixRenderer&) = delete;
    OptixRenderer& operator=(const OptixRenderer&) = delete;
    OptixRenderer(OptixRenderer&&) = delete;
    OptixRenderer& operator=(OptixRenderer&&) = delete;

    void reset();
    void render(vec4* image, uvec2 dim);
    void setCamera(const mat4& clipToWorld);
    void loadGLTF(const std::filesystem::path& path);
    void resize(uvec2 dim);
    
    Params* params; // NOTE: This is owned memory and must be properly freed
    Scene scene;

private:
    OptixDeviceContext context;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt; // NOTE: This contains owned memory and must be properly freed
    std::array<OptixProgramGroup, 3> programGroups;
    //tcnn::TrainableModel nrcModel;
    tcnn::GPUMatrix<float> nrcTrainInput;
    tcnn::GPUMatrix<float> nrcTrainOutput;

    void generateSobol(uint offset, uint n);
    void ensureSobol(uint sample);
};