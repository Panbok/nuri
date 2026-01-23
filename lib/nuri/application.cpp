#include "nuri/application.h"
#include "minilog/minilog.h"

namespace nuri {

Application::Application(const std::string &title, std::int32_t width,
                         std::int32_t height)
    : title_(title), width_(width), height_(height), window_(nullptr) {
  minilog::initialize(nullptr, {.threadNames = false});
  window_ = lvk::initWindow(title_.c_str(), width_, height_, false, false);
  context_ = lvk::createVulkanContextWithSwapchain(
      window_, static_cast<std::uint32_t>(width_),
      static_cast<std::uint32_t>(height_), lvk::ContextConfig{});
}

Application::~Application() {
  context_.reset();
  glfwDestroyWindow(window_);
  glfwTerminate();
  minilog::deinitialize();
}

void Application::run() {
  while (!glfwWindowShouldClose(window_)) {
    onUpdate(0.0f);
    glfwPollEvents();
    glfwGetFramebufferSize(window_, &width_, &height_);
    if (!width_ || !height_)
      continue;
    onRender();
    lvk::ICommandBuffer &commandBuffer = context_->acquireCommandBuffer();
    context_->submit(commandBuffer, context_->getCurrentSwapchainTexture());
  }

  onShutdown();
}

} // namespace nuri
