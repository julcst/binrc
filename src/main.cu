#include "mainapp.cuh"

#include <iostream>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <tiny-cuda-nn/gpu_memory.h>
#include <stb_image_write.h>

#include "optixrenderer.cuh"

int saveImage(const std::string& filename, const glm::uvec2& dim, const glm::vec4* image) {
    std::vector<glm::vec4> hostImage(dim.x * dim.y);
    cudaMemcpy(hostImage.data(), image, dim.x * dim.y * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
    return stbi_write_hdr(filename.c_str(), dim.x, dim.y, 4, (float*) hostImage.data());
}

int main(int argc, char** argv) {
    if (argc == 1) {
        MainApp app;
        app.run();
    } else if (argc == 3) {
        OptixRenderer renderer;
        renderer.loadGLTF(argv[1]);

        glm::uvec2 dim = {512, 512};
        renderer.resize(dim);

        if (renderer.scene.cameras.empty()) {
            std::cerr << "No cameras found in the scene." << std::endl;
            return 1;
        }
        const auto clipToWorld = renderer.scene.cameras[0].second;
        renderer.setCamera(clipToWorld);

        const auto samples = atoi(argv[2]);

        tcnn::GPUMemory<glm::vec4> image(dim.x * dim.y);
        for (auto i = 0; i < samples; i++) {
            renderer.render(image.data(), dim);
            std::cout << "Sample " << i + 1 << " / " << samples << "\r" << std::flush;
            if (i % 10000 == 0) saveImage("output_" + std::to_string(i) + ".hdr", dim, image.data());
        }

        saveImage("output.hdr", dim, image.data());
    } else {
        std::cerr << "Usage: " << argv[0] << " [scene.glb] [samples]" << std::endl;
        return 1;
    }
}   