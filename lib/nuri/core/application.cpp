#include "nuri/core/application.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/renderer.h"

namespace nuri {

Application::Application(const ApplicationConfig &config)
    : title_(config.title), width_(config.width), height_(config.height),
      windowMode_(config.windowMode) {
  Log::initialize({
      .filePath =
          std::filesystem::path(
              std::format(
                  "logs/{}_nuri.log",
                  std::chrono::system_clock::now().time_since_epoch().count()))
              .string(),
      .logLevel = LogLevel::Debug,
      .consoleLevel = LogLevel::Debug,
      .threadNames = false,
  });
  window_ = Window::create(title_, width_, height_, windowMode_);

  // Sync initial size to the actual window framebuffer size (important for
  // fullscreen / monitor-sized window creation which can ignore the requested
  // width/height).
  if (window_) {
    int32_t fbw = 0;
    int32_t fbh = 0;
    window_->getFramebufferSize(fbw, fbh);
    if (fbw > 0 && fbh > 0) {
      width_ = fbw;
      height_ = fbh;
    }
  }

  gpu_ = GPUDevice::create(*window_);
  renderer_ = Renderer::create(*gpu_);
}

Application::Application(const std::string &title, std::int32_t width,
                         std::int32_t height, WindowMode windowMode)
    : Application(ApplicationConfig{
          .title = title,
          .width = width,
          .height = height,
          .windowMode = windowMode,
      }) {}

Application::~Application() {
  layerStack_.clear();
  renderer_.reset();
  gpu_.reset();
  window_.reset();
  Log::shutdown();
}

void Application::run() {
  NURI_LOG_INFO("Application::run: Application started");
  NURI_PROFILER_THREAD("Main");

  onInit();
  double lastTime = getTime();

  while (!gpu_->shouldClose()) {
    NURI_PROFILER_FRAME("Frame");

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
      continue;
    }

    if (newWidth != width_ || newHeight != height_) {
      NURI_PROFILER_ZONE("Resize", NURI_PROFILER_COLOR_CREATE);
      width_ = newWidth;
      height_ = newHeight;
      onResize(width_, height_);
      renderer_->onResize(width_, height_);
      layerStack_.onResize(width_, height_);
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
      NURI_PROFILER_ZONE("LayerStack::onUpdate", NURI_PROFILER_COLOR_SUBMIT);
      layerStack_.onUpdate(deltaTime);
      NURI_PROFILER_ZONE_END();
    }
    {
      NURI_PROFILER_ZONE("GPUDevice::pollEvents", NURI_PROFILER_COLOR_WAIT);
      gpu_->pollEvents();
      NURI_PROFILER_ZONE_END();
    }

    {
      NURI_PROFILER_ZONE("onDraw", NURI_PROFILER_COLOR_CMD_DRAW);
      onDraw();
      NURI_PROFILER_ZONE_END();
    }
  }

  NURI_LOG_INFO("Application::run: Application shutdown");
  onShutdown();
}

double Application::getTime() const { return gpu_->getTime(); }

GPUDevice &Application::getGPU() { return *gpu_; }

const GPUDevice &Application::getGPU() const { return *gpu_; }

Window &Application::getWindow() { return *window_; }

const Window &Application::getWindow() const { return *window_; }

Renderer &Application::getRenderer() { return *renderer_; }

const Renderer &Application::getRenderer() const { return *renderer_; }

LayerStack &Application::getLayerStack() { return layerStack_; }

const LayerStack &Application::getLayerStack() const { return layerStack_; }

} // namespace nuri
