#include "mainapp.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <framework/gl/buffer.hpp>
#include <framework/imguiutil.hpp>

#include <glm/glm.hpp>
using namespace glm;

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
void main() {
    fragColor = texture(tex, texCoord);
})";

MainApp::MainApp() : App(800, 600) {
    printCudaDevices();

    fullscreenTriangle.load(Mesh::FULLSCREEN_VERTICES, Mesh::FULLSCREEN_INDICES);

    blitProgram.loadSource(vs, fs);
    blitProgram.use();
}

MainApp::~MainApp() {
    cudaGraphicsUnregisterResource(cudaPboResource);
}

void MainApp::resizeCallback(const vec2& res) {
    pbo.allocate(res.x * res.y * 4, GL_STREAM_DRAW);
    check(cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo.handle, cudaGraphicsMapFlagsWriteDiscard));
    blitTexture = Texture<GL_TEXTURE_2D>();
    blitTexture.allocate2D(GL_RGBA8, res.x, res.y);
    blitTexture.bindTextureUnit(0);
}

void MainApp::buildImGui() {
    ImGui::StatisticsWindow(delta, resolution);
}

void MainApp::render() {
    // Map the buffer to CUDA
    uchar4* devPtr;
    size_t size;
    check(cudaGraphicsMapResources(1, &cudaPboResource));
    check(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource));

    renderer.render(devPtr, resolution.x, resolution.y);

    check(cudaDeviceSynchronize());

    // Unmap the buffer
    check(cudaGraphicsUnmapResources(1, &cudaPboResource));

    // Map the buffer to a texture
    pbo.bind();
    glTextureSubImage2D(blitTexture.handle, 0, 0, 0, resolution.x, resolution.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    // Blit the texture using OpenGL
    fullscreenTriangle.draw();
}