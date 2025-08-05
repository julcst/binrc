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

std::filesystem::path getNumberedPath(const std::filesystem::path& basePath, const std::string& extension) {
    int number = 0;
    std::filesystem::path path;
    do {
        // C++20
        path = basePath / (std::format("screenshot_{:03d}.{}", number, extension));
        // C++17
        // std::stringstream ss;
        // ss << basePath.string() << std::setw(3) << std::setfill('0') << number << "." << extension;
        // path = ss.str();
        number++;
    } while (std::filesystem::exists(path));
    return path;
}

MainApp::MainApp() : App(800, 800) {
    printCudaDevices();

    fullscreenTriangle.load(Mesh::FULLSCREEN_VERTICES, Mesh::FULLSCREEN_INDICES);

    blitProgram.loadSource(vs, fs);
    blitProgram.use();
    blitProgram.set(1, exposure);

    camera.worldPosition = vec3(0.0f, 0.0f, -2.0f);
    camera.invalidate();

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
    if (action != Action::PRESS) return;
    switch (key) {
        case Key::ESC:
            close();
            break;
        case Key::T:
            imguiEnabled = !imguiEnabled;
            break;
        case Key::C:
            takeScreenshot(getNumberedPath("screenshot_", "png").string());
            break;
        case Key::X:
            blitTexture.writeToFile(getNumberedPath("screenshot_", "hdr").string());
            break;
        case Key::J:
            std::cout << renderer.getConfig().dump(4) << std::endl;
            break;
        case Key::F:
            camera.target = vec3(0.0f, 0.0f, 0.0f);
            camera.invalidate();
            break;
        default:
            break;
    }
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
    bool reset = false;

    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::string folder_str = folder.string();
        if (ImGui::InputText("Folder", &folder_str)) {
            folder = folder_str;
            scenes = scanFolder(folder);
            sceneID = 0;
        }
        if (ImGui::FileCombo("Scene##2", &sceneID, scenes)) {
            renderer.loadGLTF(scenes.at(sceneID));
        }
        ImGui::SliderFloat("Scene Epsilon", &renderer.params.sceneEpsilon, 1e-6f, 1e-1f, "%f", ImGuiSliderFlags_Logarithmic);
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
    }

    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Sample: %d", renderer.params.sample);
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 10.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) blitProgram.set(1, exposure);
        reset |= ImGui::EnumCombo("Inference Mode", &renderer.params.inferenceMode, INFERENCE_MODES);
        reset |= ImGui::FlagCheckbox("Enable Light Tracing Fix", &renderer.params.flags, LIGHT_TRACE_FIX_FLAG);
        reset |= ImGui::SliderFloat("Variance Tradeoff", &renderer.params.varianceTradeoff, 0.0f, 1.0f, "%.3f");

        switch (renderer.params.inferenceMode) {
            case InferenceMode::NO_INFERENCE:
                ImGui::SliderFloat("Russian Roulette", &renderer.params.russianRouletteWeight, 1.0f, 10.0f, "%.1f");
                reset |= ImGui::FlagCheckbox("Enable NEE", &renderer.params.flags, NEE_FLAG);
                reset |= ImGui::FlagCheckbox("Enable Transmission", &renderer.params.flags, TRANSMISSION_FLAG);
                break;
        }
        reset |= ImGui::SliderInt("Max Path Length", reinterpret_cast<int*>(&renderer.params.maxPathLength), 1, MAX_BOUNCES);
    }

    if (ImGui::CollapsingHeader("NRC", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Training", &renderer.enableTraining);
        ImGui::FlagCheckbox("Enable Diffuse Encoding", &renderer.params.flags, DIFFUSE_ENCODING_FLAG);
        ImGui::FlagCheckbox("Enable Self Learning", &renderer.params.flags, SELF_LEARNING_FLAG);
        ImGui::SliderFloat("Training Direction", &renderer.trainingDirection, 0.0f, 1.0f, "%.2f");
        ImGui::PlotLines("Loss", renderer.lossHistory.data(), renderer.lossHistory.size());
        if (ImGui::Button("Reset NRC")) {
            renderer.resetNRC();
            reset = true;
        }
    }

    if (ImGui::CollapsingHeader("Eye Training", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::FlagCheckbox("Russian Roulette##2", &renderer.params.flags, FORWARD_RR_FLAG);
    }

    if (ImGui::CollapsingHeader("Light Training", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Combo("Training Mode", &renderer.backwardTrainer, {
            {TRAIN_LIGHT, "Light"},
            {TRAIN_LIGHT_NAIVE, "Light Naive"},
            {TRAIN_BIDIR, "Bidirectional"},
        });
        ImGui::FlagCheckbox("Russian Roulette##3", &renderer.params.flags, BACKWARD_RR_FLAG);
        float balancing = 100.0f - 100.0f / renderer.params.balanceWeight;
        if (ImGui::SliderFloat("Balancing Samples", &balancing, 0.0f, 100.0f, "%.0f%%")) {
            renderer.params.balanceWeight = 100.0f / (100.0f - balancing);
        }
        ImGui::Text("(Balancing Weight: %.2f)", renderer.params.balanceWeight);
        ImGui::SliderFloat("Photon Query Samples", &renderer.photonMappingAmount, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Photon Query Replacement Factor", &renderer.photonQueryReplacement, 0.0f, 1.0f, "%.2f");
        ImGui::SliderInt("Photon Count", reinterpret_cast<int*>(&renderer.photonCount), 1 << 10, 1 << 20, "%d");
        ImGui::SliderFloat("Photon Radius", &renderer.sppmBVH.initialRadius, 0.01f, 1.0f, "%.2f");
        ImGui::SliderFloat("Alpha", &renderer.sppmBVH.alpha, 0.0f, 1.0f, "%.2f");
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

    const float camDelta = 2.5f * delta;
    if (isKeyDown(Key::W)) camera.moveInEyeSpace(vec3(0.0f, 0.0f, -camDelta));
    if (isKeyDown(Key::S)) camera.moveInEyeSpace(vec3(0.0f, 0.0f, camDelta));
    if (isKeyDown(Key::A)) camera.moveInEyeSpace(vec3(-camDelta, 0.0f, 0.0f));
    if (isKeyDown(Key::D)) camera.moveInEyeSpace(vec3(camDelta, 0.0f, 0.0f));
    if (isKeyDown(Key::Q)) camera.moveInEyeSpace(vec3(0.0f, -camDelta, 0.0f));
    if (isKeyDown(Key::E)) camera.moveInEyeSpace(vec3(0.0f, camDelta, 0.0f));

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