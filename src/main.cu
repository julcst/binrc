#include "mainapp.cuh"

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <tiny-cuda-nn/gpu_memory.h>
#include <stb_image_write.h>

#include "optixrenderer.cuh"

int saveImage(const std::string& filename, uint samples, const glm::uvec2& dim, const glm::vec4* image) {
    std::vector<glm::vec4> hostImage(dim.x * dim.y);
    cudaMemcpy(hostImage.data(), image, dim.x * dim.y * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
    return stbi_write_hdr((filename + "_" + std::to_string(samples) + ".hdr").c_str(), dim.x, dim.y, 4, (float*) hostImage.data());
}

int main(int argc, char** argv) {
    if (argc == 1) {
        MainApp app;
        app.run();
    } else if (argc >= 3) {
        OptixRenderer renderer;
        const std::filesystem::path scenePath = argv[1];
        const auto filename = scenePath.stem().string();
        renderer.loadGLTF(scenePath);

        glm::uvec2 dim = {512, 512};
        renderer.resize(dim);

        if (renderer.scene.cameras.empty()) {
            std::cerr << "No cameras found in the scene." << std::endl;
            return 1;
        }
        const auto clipToWorld = renderer.scene.cameras[0].second;
        renderer.setCamera(clipToWorld);

        std::vector<uint> runs;
        for (int i = 2; i < argc; i++) {
            runs.push_back(std::stoi(argv[i]));
        }

        tcnn::GPUMemory<glm::vec4> image(dim.x * dim.y);
        uint spp = 0;
        for (const auto& run : runs) {
            while (spp < run) {
                renderer.render(image.data(), dim);
                spp++;
                std::cout << "SPP: " << spp << " / " << run << "\r" << std::flush;
            }
            saveImage(filename, spp, dim, image.data());
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " [scene.glb] [samples...]" << std::endl;
        return 1;
    }
}   