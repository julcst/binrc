#include "mainapp.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <optix_types.h>

#include <framework/gl/buffer.hpp>
#include <framework/imguiutil.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace glm;

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <iostream>
#include <format>
#include <filesystem>

#include "cudautil.hpp"

const std::string vs = R"(#version 460 core
layout(location = 0) in vec2 position;
out vec2 texCoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    texCoord = position * 0.5 + 0.5;
})";

const std::string fs = R"(#version 460 core
in vec2 texCoord;
out vec4 fragColor;
layout(location = 0) uniform sampler2D tex;
layout(location = 1) uniform float exposure;
void main() {
    fragColor = texture(tex, texCoord) * exposure;
})";

std::vector<std::filesystem::path> scanFolder(const std::filesystem::path& folder) {
    std::vector<std::filesystem::path> files;
    try {
        for (auto& f : std::filesystem::directory_iterator(folder)) {
            if (f.path().extension() == ".glb"|| f.path().extension() == ".gltf") files.push_back(f.path());
        } 
        return files;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << e.what() << std::endl;
        return files;
    }
}

MainApp::MainApp() : App(800, 600) {
    printCudaDevices();

    fullscreenTriangle.load(Mesh::FULLSCREEN_VERTICES, Mesh::FULLSCREEN_INDICES);

    blitProgram.loadSource(vs, fs);
    blitProgram.use();
    blitProgram.set(1, exposure);

    folder = std::filesystem::current_path().parent_path().string();
    scenes = scanFolder(folder);

    setVSync(false);
}

MainApp::~MainApp() {
    check(cudaGraphicsUnregisterResource(cudaPboResource));
}

void MainApp::resize(const ivec2& res) {
    bufferDim = uvec2(res);
    if (cudaPboResource) check(cudaGraphicsUnregisterResource(cudaPboResource)); // Unregister the old resource to prevent memory leak
    pbo.allocate(res.x * res.y * sizeof(vec4), GL_STREAM_DRAW);
    check(cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo.handle, cudaGraphicsMapFlagsWriteDiscard));
    blitTexture = Texture<GL_TEXTURE_2D>();
    blitTexture.allocate2D(GL_RGBA32F, res.x, res.y);
    blitTexture.bindTextureUnit(0);
    camera.resize(float(res.x) / float(res.y));
    renderer.reset();
    renderer.resize(uvec2(res));
}

void MainApp::resizeCallback(const vec2& res) { 
    resize(res); // TODO: Fix scaling issues
}

void MainApp::keyCallback(Key key, Action action, Modifier modifier) {
    if (action == Action::PRESS && key == Key::ESC) close();
}

void MainApp::scrollCallback(float xamount, float yamount) {
    camera.zoom(yamount);
}

void MainApp::moveCallback(const vec2& movement, bool leftButton, bool rightButton, bool middleButton) {
    if (leftButton) camera.orbit(movement * 0.01f);
}

void MainApp::buildImGui() {
    ImGui::StatisticsWindow(delta, resolution);

    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    std::string folder_str = folder.string();
    if (ImGui::InputText("Folder", &folder_str)) {
        folder = folder_str;
        scenes = scanFolder(folder);
        sceneID = 0;
    }
    if (ImGui::FileCombo("Scene", &sceneID, scenes)) {
        renderer.loadGLTF(scenes.at(sceneID));
    }
    ImGui::Text("Sample: %d", renderer.params.sample);
    if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 10.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) blitProgram.set(1, exposure);
    ImGui::SliderFloat("Russian Roulette", &renderer.params.russianRouletteWeight, 1.0f, 10.0f, "%.1f");
    ImGui::SliderFloat("Scene Epsilon", &renderer.params.sceneEpsilon, 1e-6f, 1e-1f, "%f", ImGuiSliderFlags_Logarithmic);
    bool reset = ImGui::FlagCheckbox("Enable NEE", &renderer.params.flags, NEE_FLAG);
    reset |= ImGui::FlagCheckbox("Enable Transmission", &renderer.params.flags, TRANSMISSION_FLAG);
    for (size_t i = 0; i < renderer.scene.cameras.size(); i++) {
        if (ImGui::Button(renderer.scene.cameras[i].first.c_str())) {
            auto scale = mat4(1.0f);
            scale[0][0] = camera.aspectRatio;
            const auto clipToWorld = renderer.scene.cameras[i].second * scale;
            renderer.setCamera(clipToWorld);
            reset = true;
        }
        if (i < renderer.scene.cameras.size() - 1) ImGui::SameLine();
    }

    ImGui::SeparatorText("NRC");
    ImGui::Checkbox("Enable Training", &renderer.enableTraining);
    ImGui::FlagCheckbox("Enable Backward RR", &renderer.params.flags, BACKWARD_RR_FLAG);
    ImGui::FlagCheckbox("Enable Forward RR", &renderer.params.flags, FORWARD_RR_FLAG);
    ImGui::SliderFloat("Training Direction", &renderer.trainingDirection, 0.0f, 1.0f, "%.2f");
    reset |= ImGui::EnumCombo("Inference Mode", &renderer.params.inferenceMode, INFERENCE_MODES);
    ImGui::PlotLines("Loss", renderer.lossHistory.data(), renderer.lossHistory.size());
    if (ImGui::Button("Reset NRC")) {
        renderer.resetNRC();
        reset = true;
    }

    ImGui::End();

    if (reset) renderer.reset();
}

void MainApp::render() {
    // Map the buffer to CUDA
    vec4* image;
    size_t size;
    check(cudaGraphicsMapResources(1, &cudaPboResource));
    check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&image), &size, cudaPboResource));

    if(camera.updateIfChanged()) {
        renderer.setCamera(inverse(camera.projectionMatrix * camera.viewMatrix));
        renderer.reset();
    }

    renderer.render(image, bufferDim);

    // Unmap the buffer
    check(cudaGraphicsUnmapResources(1, &cudaPboResource));

    // Map the buffer to a texture
    pbo.bind();
    glTextureSubImage2D(blitTexture.handle, 0, 0, 0, bufferDim.x, bufferDim.y, GL_RGBA, GL_FLOAT, nullptr);

    // Blit the texture using OpenGL
    fullscreenTriangle.draw();
}