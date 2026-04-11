#include "nuri/pch.h"

#include "nuri/core/application.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/features/composite_feature.h"
#include "nuri/gfx/pipeline/features/debug_feature.h"
#include "nuri/gfx/pipeline/features/opaque_feature.h"
#include "nuri/gfx/pipeline/features/skybox_feature.h"
#include "nuri/gfx/pipeline/features/transmission_feature.h"
#include "nuri/gfx/pipeline/features/transparent_feature.h"
#include "nuri/gfx/pipeline/providers/frame_composition_provider.h"
#include "nuri/gfx/pipeline/providers/material_table_gpu_provider.h"
#include "nuri/gfx/pipeline/providers/scene_lighting_provider.h"
#include "nuri/gfx/renderer.h"

namespace nuri {

ApplicationConfig makeApplicationConfig(const RuntimeConfig &config) {
  return ApplicationConfig{
      .title = config.window.title,
      .width = config.window.width,
      .height = config.window.height,
      .windowMode = config.window.mode,
      .shaderConfig = config.shaders,
  };
}

Application::LogLifetimeGuard::LogLifetimeGuard(const LogConfig &config) {
  Log::initialize(config);
}

Application::LogLifetimeGuard::~LogLifetimeGuard() { Log::shutdown(); }

LogConfig Application::makeDefaultLogConfig() {
  std::filesystem::create_directories("logs");
  return {
      .filePath =
          std::filesystem::path(
              std::format(
                  "logs/{}_nuri.log",
                  std::chrono::system_clock::now().time_since_epoch().count()))
              .string(),
      .logLevel = LogLevel::Debug,
      .consoleLevel = LogLevel::Debug,
      .threadNames = false,
  };
}

Application::Application(const ApplicationConfig &config)
    : logLifetimeGuard_(makeDefaultLogConfig()), appConfig_(config),
      width_(appConfig_.width), height_(appConfig_.height),
      windowMode_(appConfig_.windowMode), eventManager_(eventMemory_),
      input_(eventManager_) {
  inputDispatchSubscription_ = eventManager_.subscribe<InputEvent>(
      EventChannel::Input, &Application::dispatchInputEvent, this);

  window_ = Window::create(appConfig_.title, width_, height_, windowMode_);
  NURI_ASSERT(window_ != nullptr, "Failed to create window");
  window_->bindEventManager(&eventManager_);

  // Sync initial size to the actual window framebuffer size (important for
  // fullscreen / monitor-sized window creation which can ignore the requested
  // width/height).
  int32_t fbw = 0;
  int32_t fbh = 0;
  window_->getFramebufferSize(fbw, fbh);
  if (fbw > 0 && fbh > 0) {
    width_ = fbw;
    height_ = fbh;
  }

  gpu_ = GPUDevice::create(*window_);
  NURI_ASSERT(gpu_ != nullptr, "Failed to create GPU device");
  renderer_ = Renderer::create(*gpu_, rendererMemory_);
  NURI_ASSERT(renderer_ != nullptr, "Failed to create renderer");
  renderPipeline_ = std::make_unique<RenderPipeline>(&rendererMemory_);
  auto pipelineResult = registerConfiguredDefaultRenderPipeline();
  NURI_ASSERT(!pipelineResult.hasError(),
              "Failed to register default render pipeline: %s",
              pipelineResult.error().c_str());
}

Application::Application(const std::string &title, std::int32_t width,
                         std::int32_t height, WindowMode windowMode)
    : Application(ApplicationConfig{
          .title = title,
          .width = width,
          .height = height,
          .windowMode = windowMode,
          .shaderConfig = std::nullopt,
      }) {}

Application::~Application() {
  (void)eventManager_.unsubscribe(inputDispatchSubscription_);
  renderPipeline_.reset();
  renderer_.reset();
  gpu_.reset();
  window_.reset();
}

void Application::run() {
  NURI_LOG_DEBUG("Application::run: Application started");
  NURI_PROFILER_THREAD("Main");

  onInit();
  double lastTime = getTime();

  while (!window_->shouldClose()) {
    NURI_PROFILER_FRAME("Frame");

    input_.beginFrame();
    {
      NURI_PROFILER_ZONE("Window::pollEvents", NURI_PROFILER_COLOR_WAIT);
      window_->pollEvents();
      NURI_PROFILER_ZONE_END();
    }
    {
      NURI_PROFILER_ZONE("EventManager::dispatch(RawInput)",
                         NURI_PROFILER_COLOR_WAIT);
      eventManager_.dispatch(EventChannel::RawInput);
      NURI_PROFILER_ZONE_END();
    }
    {
      NURI_PROFILER_ZONE("EventManager::dispatch(Input)",
                         NURI_PROFILER_COLOR_WAIT);
      eventManager_.dispatch(EventChannel::Input);
      NURI_PROFILER_ZONE_END();
    }

    std::int32_t newWidth = 0;
    std::int32_t newHeight = 0;
    {
      NURI_PROFILER_ZONE("GPUDevice::getFramebufferSize",
                         NURI_PROFILER_COLOR_WAIT);
      gpu_->getFramebufferSize(newWidth, newHeight);
      NURI_PROFILER_ZONE_END();
    }
    if (!newWidth || !newHeight) {
      width_ = newWidth;
      height_ = newHeight;
      input_.endFrame();
      continue;
    }

    if (newWidth != width_ || newHeight != height_) {
      NURI_PROFILER_ZONE("Resize", NURI_PROFILER_COLOR_CREATE);
      width_ = newWidth;
      height_ = newHeight;
      onResize(width_, height_);
      renderer_->onResize(width_, height_);
      NURI_PROFILER_ZONE_END();
    }

    double currentTime = getTime();
    double deltaTime = currentTime - lastTime;
    lastTime = currentTime;
    {
      NURI_PROFILER_ZONE("onUpdate", NURI_PROFILER_COLOR_SUBMIT);
      onUpdate(deltaTime);
      NURI_PROFILER_ZONE_END();
    }
    {
      NURI_PROFILER_ZONE("onDraw", NURI_PROFILER_COLOR_CMD_DRAW);
      onDraw();
      NURI_PROFILER_ZONE_END();
    }

    input_.endFrame();
  }

  NURI_LOG_DEBUG("Application::run: Application shutdown");
  onShutdown();
}

double Application::getTime() const { return gpu_->getTime(); }

GPUDevice &Application::getGPU() { return *gpu_; }

const GPUDevice &Application::getGPU() const { return *gpu_; }

Window &Application::getWindow() { return *window_; }

const Window &Application::getWindow() const { return *window_; }

Renderer &Application::getRenderer() { return *renderer_; }

const Renderer &Application::getRenderer() const { return *renderer_; }

RenderPipeline &Application::getRenderPipeline() {
  NURI_ASSERT(renderPipeline_ != nullptr, "Render pipeline is null");
  return *renderPipeline_;
}

const RenderPipeline &Application::getRenderPipeline() const {
  NURI_ASSERT(renderPipeline_ != nullptr, "Render pipeline is null");
  return *renderPipeline_;
}

EventManager &Application::getEventManager() { return eventManager_; }

const EventManager &Application::getEventManager() const {
  return eventManager_;
}

InputSystem &Application::getInput() { return input_; }

const InputSystem &Application::getInput() const { return input_; }

const ApplicationConfig &Application::config() const { return appConfig_; }

bool Application::dispatchInputEvent(const InputEvent &event, void *user) {
  if (!user) {
    return false;
  }
  return static_cast<Application *>(user)->handleInputEvent(event);
}

bool Application::handleInputEvent(const InputEvent &event) {
  return onInput(event);
}

Result<bool, std::string>
Application::registerConfiguredDefaultRenderPipeline() {
  if (renderPipeline_ == nullptr || !appConfig_.shaderConfig.has_value()) {
    return Result<bool, std::string>::makeResult(true);
  }
  return registerDefaultRenderPipeline(*appConfig_.shaderConfig);
}

bool Application::onInput(const InputEvent &event) {
  if (event.type == InputEventType::Key &&
      event.payload.key.key == Key::Escape &&
      event.payload.key.action == KeyAction::Press) {
    if (window_) {
      window_->requestClose();
    }
    return true;
  }
  return false;
}

Result<bool, std::string> Application::registerDefaultRenderPipeline(
    const RuntimeShaderConfig &shaderConfig) {
  if (renderPipeline_ == nullptr) {
    return Result<bool, std::string>::makeError(
        "Application::registerDefaultRenderPipeline: render pipeline is null");
  }

  renderPipeline_->addProvider(
      std::make_unique<MaterialTableGpuProvider>(getGPU()));
  renderPipeline_->addProvider(std::make_unique<FrameCompositionProvider>(
      getGPU(), pipelineMemoryResource()));
  renderPipeline_->addProvider(
      std::make_unique<SceneLightingProvider>(getGPU()));
  renderPipeline_->addFeature(
      std::make_unique<SkyboxFeature>(getGPU(), shaderConfig.skybox));
  renderPipeline_->addFeature(std::make_unique<OpaqueFeature>(
      getGPU(), shaderConfig.opaque, pipelineMemoryResource()));
  renderPipeline_->addFeature(
      std::make_unique<FrameCompositionFeature>(getGPU(), shaderConfig.opaque));
  renderPipeline_->addFeature(std::make_unique<TransmissionFeature>(
      getGPU(), shaderConfig.opaque, pipelineMemoryResource()));
  renderPipeline_->addFeature(std::make_unique<TransparentFeature>(
      getGPU(), shaderConfig.opaque, pipelineMemoryResource()));
  renderPipeline_->addFeature(std::make_unique<DebugFeature>(
      getGPU(), shaderConfig.debugGrid, pipelineMemoryResource()));
  renderPipeline_->addFeature(
      std::make_unique<FramePresentFeature>(getGPU(), shaderConfig.opaque));

  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
