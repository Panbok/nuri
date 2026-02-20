#pragma once

#include "nuri/core/event_manager.h"
#include "nuri/core/input_events.h"
#include "nuri/core/input_system.h"
#include "nuri/core/layer_stack.h"
#include "nuri/core/log.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/renderer.h"

namespace nuri {
struct NURI_API ApplicationConfig {
  std::string title = "Nuri";
  std::int32_t width = 960;
  std::int32_t height = 540;

  // Window mode (mutually exclusive).
  // Windowed with width=0 and height=0 uses max screen size coverage.
  WindowMode windowMode = WindowMode::Windowed;
};

class NURI_API Application {
public:
  explicit Application(const ApplicationConfig &config);
  Application(const std::string &title, std::int32_t width, std::int32_t height,
              WindowMode windowMode = WindowMode::Windowed);
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
  virtual bool onInput(const InputEvent &event);
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
  const ApplicationConfig &config() const;

  Renderer &getRenderer();
  const Renderer &getRenderer() const;
  LayerStack &getLayerStack();
  const LayerStack &getLayerStack() const;
  EventManager &getEventManager();
  const EventManager &getEventManager() const;
  InputSystem &getInput();
  const InputSystem &getInput() const;

protected:
  [[nodiscard]] std::pmr::memory_resource *layerMemoryResource() noexcept {
    return &layerMemory_;
  }

private:
  struct LogLifetimeGuard {
    explicit LogLifetimeGuard(const LogConfig &config);
    ~LogLifetimeGuard();
    LogLifetimeGuard(const LogLifetimeGuard &) = delete;
    LogLifetimeGuard &operator=(const LogLifetimeGuard &) = delete;
  };

  static LogConfig makeDefaultLogConfig();

  static bool dispatchInputEvent(const InputEvent &event, void *user);
  bool handleInputEvent(const InputEvent &event);

  LogLifetimeGuard logLifetimeGuard_;
  ApplicationConfig appConfig_{};
  std::int32_t width_;
  std::int32_t height_;
  WindowMode windowMode_ = WindowMode::Windowed;
  std::unique_ptr<Window> window_;
  std::unique_ptr<GPUDevice> gpu_;
  std::pmr::unsynchronized_pool_resource rendererMemory_;
  std::unique_ptr<Renderer> renderer_;
  std::pmr::unsynchronized_pool_resource layerMemory_;
  LayerStack layerStack_;
  std::pmr::unsynchronized_pool_resource eventMemory_;
  EventManager eventManager_;
  InputSystem input_;
  SubscriptionToken inputDispatchSubscription_{};
};

} // namespace nuri
