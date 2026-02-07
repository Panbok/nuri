#pragma once

#include "nuri/core/layer_stack.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/renderer.h"

namespace nuri {
struct NURI_API ApplicationConfig {
  std::string_view title = "Nuri";
  std::int32_t width = 960;
  std::int32_t height = 540;

  // Window mode:
  // - fullscreen=true,  borderlessFullscreen=false: exclusive fullscreen
  // - fullscreen=true,  borderlessFullscreen=true:  borderless monitor-sized
  //   windowed mode
  // - fullscreen=false, width=0 and height=0:      windowed with max screen
  //   size coverage
  bool fullscreen = false;
  bool borderlessFullscreen = false;
};

class NURI_API Application {
public:
  explicit Application(const ApplicationConfig &config);
  Application(const std::string &title, std::int32_t width, std::int32_t height,
              bool fullscreen = false, bool borderlessFullscreen = false);
  virtual ~Application();

  Application(const Application &) = delete;
  Application &operator=(const Application &) = delete;
  Application(Application &&) = delete;
  Application &operator=(Application &&) = delete;

  void run();
  double getTime() const;

  virtual void onInit() = 0;
  virtual void onUpdate(double deltaTime) = 0;
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
  LayerStack &getLayerStack();
  const LayerStack &getLayerStack() const;

private:
  std::string title_;
  std::int32_t width_;
  std::int32_t height_;
  bool fullscreen_ = false;
  bool borderlessFullscreen_ = false;
  std::unique_ptr<Window> window_;
  std::unique_ptr<GPUDevice> gpu_;
  std::unique_ptr<Renderer> renderer_;
  LayerStack layerStack_;
};

} // namespace nuri
