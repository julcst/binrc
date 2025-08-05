#include "mainapp.cuh"

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <variant>
#include <glm/glm.hpp>
#include <tiny-cuda-nn/gpu_memory.h>
#include <stb_image_write.h>
#include <chrono>
#include <sstream>

#include "optixrenderer.cuh"

int saveImage(const std::string& path, const glm::uvec2& dim, const glm::vec4* image) {
    std::vector<glm::vec4> hostImage(dim.x * dim.y);
    cudaMemcpy(hostImage.data(), image, dim.x * dim.y * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
    return stbi_write_hdr(path.c_str(), dim.x, dim.y, 4, (float*) hostImage.data());
}

// Define types
struct BreakConditionDuration {
    std::chrono::steady_clock::duration duration;

    bool isMet(uint /*samples*/, std::chrono::steady_clock::duration elapsed) const {
        return elapsed >= duration;
    }

    std::string toString() const {
        std::ostringstream ss;
        if (duration < std::chrono::seconds(1)) {
            ss << std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        } else if (duration < std::chrono::minutes(1)) {
            ss << std::chrono::duration_cast<std::chrono::seconds>(duration);
        } else if (duration < std::chrono::hours(1)) {
            ss << std::chrono::duration_cast<std::chrono::minutes>(duration);
        } else {
            ss << std::chrono::duration_cast<std::chrono::hours>(duration);
        }
        return ss.str();
    }
};

struct BreakConditionSample {
    uint32_t sampleCount;

    bool isMet(uint samples, std::chrono::steady_clock::duration /*elapsed*/) const {
        return samples >= sampleCount;
    }

    std::string toString() const {
        return std::to_string(sampleCount) + "spp";
    }
};

using BreakCondition = std::variant<std::monostate, BreakConditionDuration, BreakConditionSample>;

inline BreakCondition parseRenderCondition(const std::string& conditionStr) {
    if (conditionStr.empty() || conditionStr == "null") {
        return std::monostate{};
    }
    switch (conditionStr.back()) {
        case 's':
            return BreakConditionDuration{std::chrono::seconds(std::stoi(conditionStr))};
        case 'm':
            return BreakConditionDuration{std::chrono::minutes(std::stoi(conditionStr))};
        case 'h':
            return BreakConditionDuration{std::chrono::hours(std::stoi(conditionStr))};
        default:
            return BreakConditionSample{static_cast<uint32_t>(std::stoi(conditionStr.substr(0, conditionStr.size() - 3)))};
    }
}

void from_json(const nlohmann::json& j, BreakCondition& condition) {
    if (j.is_null()) {
        condition = std::monostate{};
    } else if (j.is_string()) {
        condition = parseRenderCondition(j.get<std::string>());
    } else if (j.is_number_unsigned()) {
        condition = BreakConditionSample{j.get<uint32_t>()};
    } else if (j.is_number_float()) {
        condition = BreakConditionDuration{
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(j.get<double>())
            )};
    } else {
        throw std::runtime_error("Invalid render condition format: " + j.dump());
    }
}

bool isConditionMet(const BreakCondition& condition, uint samples, std::chrono::steady_clock::duration duration) {
    return std::visit([&](const auto& c) -> bool {
        using T = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return true;
        } else {
            return c.isMet(samples, duration);
        }
    }, condition);
}

std::string conditionToString(const BreakCondition& condition) {
    return std::visit([](const auto& c) -> std::string {
        using T = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return "null";
        } else {
            return c.toString();
        }
    }, condition);
}

int main(int argc, char** argv) {
    if (argc == 1) {
        MainApp app;
        app.run();
    } else if (argc == 2) {
        OptixRenderer renderer;
        const std::filesystem::path configPath = argv[1];
        const auto config = nlohmann::json::parse(Common::readFile(configPath), nullptr, true, true);
        renderer.configure(config);

        if (renderer.scene.cameras.empty()) {
            std::cerr << "No cameras found in the scene." << std::endl;
            return 1;
        }
        const auto clipToWorld = renderer.scene.cameras[0].second;
        renderer.setCamera(clipToWorld);

        glm::vec4 xAxis = clipToWorld[0];
        glm::vec4 yAxis = clipToWorld[1];
        float aspectRatio = glm::length(xAxis) / glm::length(yAxis);
        std::cout << "Aspect Ratio: " << aspectRatio << std::endl;
        glm::uvec2 dim = {static_cast<uint>(aspectRatio * 512), 512};
        renderer.resize(dim);
        tcnn::GPUMemory<glm::vec4> image(dim.x * dim.y);

        BreakCondition pretraining = config.contains("pretraining")
            ? config.at("pretraining").get<BreakCondition>()
            : BreakCondition{};

        uint32_t spp = 0;
        auto startTime = std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration renderTime = {};
        while (!isConditionMet(pretraining, spp, renderTime)) {
            renderer.render(image.data(), dim);
            spp++;
            std::cout << "Pretraining: " << spp << "\r" << std::flush;
        }

        renderer.reset();

        std::vector<BreakCondition> conditions = config.at("export").get<std::vector<BreakCondition>>();
        for (auto& condition : conditions) {
            std::cout << "Condition: " << conditionToString(condition) << std::endl;
        }

        spp = 0;
        startTime = std::chrono::steady_clock::now();
        while (!conditions.empty()) {
            renderer.render(image.data(), dim);
            auto renderTime = std::chrono::steady_clock::now() - startTime;
            spp++;
            std::cout << "Rendering: " << spp << "\r" << std::flush;

            for (auto it = conditions.begin(); it != conditions.end();) {
                if (isConditionMet(*it, spp, renderTime)) {
                    auto path = configPath.parent_path() / configPath.stem();
                    path += "_" + conditionToString(*it);
                    saveImage(path.concat(".hdr"), dim, image.data());
                    nlohmann::json metadata = {
                        {"condition", conditionToString(*it)},
                        {"samples", spp},
                        {"duration", renderTime.count()},
                    };
                    Common::writeToFile(metadata.dump(4), path.concat(".json"));
                    it = conditions.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
}