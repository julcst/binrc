#pragma once

#include <framework/app.hpp>
#include <framework/gl/buffer.hpp>
#include <framework/gl/texture.hpp>
#include <framework/mesh.hpp>
#include <framework/gl/program.hpp>
#include <framework/camera.hpp>

#include <cuda_runtime.h>

#include <glm/glm.hpp>
using namespace glm;

#include "optixrenderer.hpp"

class MainApp : public App {
  public:
    MainApp();
    ~MainApp() override;
    MainApp(const MainApp&) = delete;
    MainApp& operator=(const MainApp&) = delete;
    MainApp(MainApp&&) = delete;
    MainApp& operator=(MainApp&&) = delete;
  protected:
    void render() override;
    void buildImGui() override;
    void keyCallback(Key key, Action action, Modifier modifier) override;
    // void clickCallback(Button button, Action action, Modifier modifier) override;
    void scrollCallback(float amount) override;
    void moveCallback(const vec2& movement, bool leftButton, bool rightButton, bool middleButton) override;
    void resizeCallback(const vec2& resolution) override;
  private:
  void resize(int width, int height);
    Buffer<GL_PIXEL_UNPACK_BUFFER> pbo;
    Texture<GL_TEXTURE_2D> blitTexture;
    Mesh fullscreenTriangle;
    Program blitProgram;
    cudaGraphicsResource_t cudaPboResource = nullptr;
    OptixRenderer renderer;
    Camera camera;
    float exposure = 1.0f;
};