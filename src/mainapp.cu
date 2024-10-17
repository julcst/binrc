#include "mainapp.hpp"

#include <cuda_runtime.h>

#include <framework/gl/buffer.hpp>

#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
using namespace glm;

__global__ void fillPattern(uchar4* pos, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        pos[idx] = make_uchar4(blockIdx.x * 128, blockIdx.y * 128, 0, 255);
    }
}

void printCudaDevices() {
    int deviceCount, device;
    cudaGetDeviceCount(&deviceCount);
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf ("Device %d has compute capability %d.%d and %d cores.\n", device,
            deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    }
}

const std::string vs = R"(#version 460 core

layout(location = 0) in vec2 position;

out vec2 texCoord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    texCoord = position * 0.5 + 0.5;
}
)";

const std::string fs = R"(#version 460 core

in vec2 texCoord;
out vec4 fragColor;

layout(location = 0) uniform sampler2D tex;

void main() {
    fragColor = texture(tex, texCoord);
}
)";

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
    cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo.handle, cudaGraphicsMapFlagsWriteDiscard);
    blitTexture = Texture<GL_TEXTURE_2D>();
    blitTexture.allocate2D(GL_RGBA8, res.x, res.y);
    blitTexture.bindTextureUnit(0);
}

void MainApp::render() {
    // Map the buffer to CUDA
    uchar4* devPtr;
    size_t size;
    cudaGraphicsMapResources(1, &cudaPboResource);
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPboResource);

    // Launch the CUDA kernel to fill the buffer with green
    dim3 block(16, 16);
    dim3 grid((resolution.x + block.x - 1) / block.x, (resolution.y + block.y - 1) / block.y);
    fillPattern<<<grid, block>>>(devPtr, resolution.x, resolution.y);
    cudaDeviceSynchronize();

    // Unmap the buffer
    cudaGraphicsUnmapResources(1, &cudaPboResource);

    // Map the buffer to a texture
    pbo.bind();
    glTextureSubImage2D(blitTexture.handle, 0, 0, 0, resolution.x, resolution.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    // Blit the texture using OpenGL
    blitProgram.use();
    fullscreenTriangle.draw();
}