#include "mainapp.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <optix_types.h>

#include <framework/gl/buffer.hpp>
#include <framework/imguiutil.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace glm;

#include <imgui.h>

#include <iostream>
#include <format>

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

MainApp::MainApp() : App(800, 600) {
    printCudaDevices();

    fullscreenTriangle.load(Mesh::FULLSCREEN_VERTICES, Mesh::FULLSCREEN_INDICES);

    blitProgram.loadSource(vs, fs);
    blitProgram.use();
    blitProgram.set(1, exposure);

    std::vector<float3> vertices = {
        make_float3(-0.5f, -0.5f, 0.0f),
        make_float3(0.5f, -0.5f, 0.0f),
        make_float3(0.0f, 0.5f, 0.0f),
    };

    std::vector<uint3> indices = {
        make_uint3(0, 1, 2),
    };

    const auto handle = renderer.buildGAS(vertices, indices);
    const auto instances = {
        OptixInstance {
            .transform = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
            },
            .instanceId = 0,
            .sbtOffset = 0,
            .visibilityMask = 255,
            .flags = OPTIX_INSTANCE_FLAG_NONE,
            .traversableHandle = handle,
        },
    };
    renderer.buildIAS(instances);
}

MainApp::~MainApp() {
    check(cudaGraphicsUnregisterResource(cudaPboResource));
}

void MainApp::resizeCallback(const vec2& res) {
    pbo.allocate(res.x * res.y * sizeof(vec4), GL_STREAM_DRAW);
    check(cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo.handle, cudaGraphicsMapFlagsWriteDiscard));
    blitTexture = Texture<GL_TEXTURE_2D>();
    blitTexture.allocate2D(GL_RGBA32F, res.x, res.y);
    blitTexture.bindTextureUnit(0);
    camera.resize(res.x / res.y);
}

void MainApp::keyCallback(Key key, Action action, Modifier modifier) {
    if (action == Action::PRESS && key == Key::ESC) close();
}

void MainApp::scrollCallback(float amount) {
    camera.zoom(amount);
}

void MainApp::moveCallback(const vec2& movement, bool leftButton, bool rightButton, bool middleButton) {
    if (leftButton) camera.orbit(movement * 0.01f);
}

void MainApp::buildImGui() {
    ImGui::StatisticsWindow(delta, resolution);
    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 10.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) blitProgram.set(1, exposure);
    ImGui::End();
}

void MainApp::render() {
    // Map the buffer to CUDA
    vec4* image;
    size_t size;
    check(cudaGraphicsMapResources(1, &cudaPboResource));
    check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&image), &size, cudaPboResource));
    const auto dim = uvec2(resolution);

    if(camera.updateIfChanged()) renderer.setCamera(inverse(camera.projectionMatrix * camera.viewMatrix));

    renderer.render(image, dim);

    check(cudaDeviceSynchronize());

    // Unmap the buffer
    check(cudaGraphicsUnmapResources(1, &cudaPboResource));

    // Map the buffer to a texture
    pbo.bind();
    glTextureSubImage2D(blitTexture.handle, 0, 0, 0, dim.x, dim.y, GL_RGBA, GL_FLOAT, nullptr);

    // Blit the texture using OpenGL
    fullscreenTriangle.draw();
}