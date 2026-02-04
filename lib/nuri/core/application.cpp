#include "nuri/core/application.h"
#include "nuri/core/log.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/renderer.h"

namespace nuri {

Application::Application(const std::string &title, std::int32_t width,
                         std::int32_t height)
    : title_(title), width_(width), height_(height) {
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
  window_ = Window::create(title_, width_, height_);
  gpu_ = GPUDevice::create(*window_);
  renderer_ = Renderer::create(*gpu_);
}

Application::~Application() {
  renderer_.reset();
  gpu_.reset();
  window_.reset();
  Log::shutdown();
}

void Application::run() {
  onInit();

  while (!gpu_->shouldClose()) {
    onUpdate(0.0f);
    gpu_->pollEvents();
    std::int32_t newWidth = 0;
    std::int32_t newHeight = 0;
    gpu_->getFramebufferSize(newWidth, newHeight);
    if (!newWidth || !newHeight) {
      width_ = newWidth;
      height_ = newHeight;
      continue;
    }

    if (newWidth != width_ || newHeight != height_) {
      width_ = newWidth;
      height_ = newHeight;
      onResize(width_, height_);
      renderer_->onResize(width_, height_);
    }

    onDraw();
  }

  onShutdown();
}

double Application::getTime() const { return gpu_->getTime(); }

GPUDevice &Application::getGPU() { return *gpu_; }

const GPUDevice &Application::getGPU() const { return *gpu_; }

Window &Application::getWindow() { return *window_; }

const Window &Application::getWindow() const { return *window_; }

Renderer &Application::getRenderer() { return *renderer_; }

const Renderer &Application::getRenderer() const { return *renderer_; }

} // namespace nuri
