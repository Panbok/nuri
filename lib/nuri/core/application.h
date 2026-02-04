#pragma once

#include "nuri/defines.h"
#include "nuri/pch.h"

namespace nuri {

class GPUDevice;
class Renderer;
class Window;

class NURI_API Application {
public:
  Application(const std::string &title, std::int32_t width,
              std::int32_t height);
  virtual ~Application();

  Application(const Application &) = delete;
  Application &operator=(const Application &) = delete;
  Application(Application &&) = delete;
  Application &operator=(Application &&) = delete;

  void run();
  double getTime() const;

  virtual void onInit() = 0;
  virtual void onUpdate(float deltaTime) = 0;
  virtual void onDraw() = 0;
  virtual void onResize(std::int32_t width, std::int32_t height) = 0;
  virtual void onShutdown() = 0;

  GPUDevice &getGPU();
  const GPUDevice &getGPU() const;
  Window &getWindow();
  const Window &getWindow() const;

  inline float getAspectRatio() const {
    return width_ / static_cast<float>(height_);
  }
  inline std::int32_t getWidth() const { return width_; }
  inline std::int32_t getHeight() const { return height_; }

  Renderer &getRenderer();
  const Renderer &getRenderer() const;

private:
  std::string title_;
  std::int32_t width_;
  std::int32_t height_;
  std::unique_ptr<Window> window_;
  std::unique_ptr<GPUDevice> gpu_;
  std::unique_ptr<Renderer> renderer_;
};

} // namespace nuri

