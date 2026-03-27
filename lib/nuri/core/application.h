#pragma once

#include "nuri/core/event_manager.h"
#include "nuri/core/input_events.h"
#include "nuri/core/input_system.h"
#include "nuri/core/log.h"
#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderer.h"

#include <optional>

namespace nuri {

// Controls how final scene composition is handled. `PipelineOnly` means the
// render pipeline owns composition end-to-end today; extend this enum if the
// application later adds CPU-composed or hybrid composition modes.
enum class RenderCompositionMode : uint8_t {
  PipelineOnly = 0,
};

struct NURI_API ApplicationConfig {
  std::string title = "Nuri";
  std::int32_t width = 960;
  std::int32_t height = 540;

  // Window mode (mutually exclusive).
  // Windowed with width=0 and height=0 uses max screen size coverage.
  WindowMode windowMode = WindowMode::Windowed;
  std::optional<RuntimeShaderConfig> shaderConfig{};
  RenderCompositionMode renderComposition = RenderCompositionMode::PipelineOnly;
};

[[nodiscard]] NURI_API ApplicationConfig
makeApplicationConfig(const RuntimeConfig &config);

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
  [[nodiscard]] RenderCompositionMode renderCompositionMode() const noexcept {
    return appConfig_.renderComposition;
  }

  Renderer &getRenderer();
  const Renderer &getRenderer() const;
  RenderPipeline &getRenderPipeline();
  const RenderPipeline &getRenderPipeline() const;
  EventManager &getEventManager();
  const EventManager &getEventManager() const;
  InputSystem &getInput();
  const InputSystem &getInput() const;

protected:
  [[nodiscard]] std::pmr::memory_resource *pipelineMemoryResource() noexcept {
    return &pipelineMemory_;
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
  [[nodiscard]] Result<bool, std::string>
  registerConfiguredDefaultRenderPipeline();
  // On success, `true` means the default pipeline was registered in this call.
  // `false` is reserved for a future "already registered / skipped" path.
  [[nodiscard]] Result<bool, std::string>
  registerDefaultRenderPipeline(const RuntimeShaderConfig &shaderConfig);

  LogLifetimeGuard logLifetimeGuard_;
  ApplicationConfig appConfig_{};
  std::int32_t width_;
  std::int32_t height_;
  WindowMode windowMode_ = WindowMode::Windowed;
  std::unique_ptr<Window> window_;
  std::unique_ptr<GPUDevice> gpu_;
  std::pmr::unsynchronized_pool_resource rendererMemory_;
  std::unique_ptr<Renderer> renderer_;
  std::unique_ptr<RenderPipeline> renderPipeline_;
  std::pmr::unsynchronized_pool_resource pipelineMemory_;
  std::pmr::unsynchronized_pool_resource eventMemory_;
  EventManager eventManager_;
  InputSystem input_;
  SubscriptionToken inputDispatchSubscription_{};
};

} // namespace nuri
