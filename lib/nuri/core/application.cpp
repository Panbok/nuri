#include "nuri/core/application.h"

namespace nuri {

Application::Application(const std::string &title, std::int32_t width,
                         std::int32_t height)
    : title_(title), width_(width), height_(height), window_(nullptr) {
  minilog::initialize(nullptr, {.threadNames = false});
  window_ = lvk::initWindow(title_.c_str(), width_, height_, true, false);
  context_ = lvk::createVulkanContextWithSwapchain(
      window_, static_cast<std::uint32_t>(width_),
      static_cast<std::uint32_t>(height_), lvk::ContextConfig{});
  renderer_ = nuri::Renderer::create(*context_);

  glfwSetWindowUserPointer(window_, this);

  glfwSetWindowSizeCallback(
      window_, [](GLFWwindow *window, int width, int height) {
        Application *app =
            static_cast<Application *>(glfwGetWindowUserPointer(window));
        app->getRenderer()->onResize(width, height);
      });
}

Application::~Application() {
  context_.reset();
  glfwDestroyWindow(window_);
  glfwTerminate();
  minilog::deinitialize();
}

void Application::run() {
  onInit();

  while (!glfwWindowShouldClose(window_)) {
    onUpdate(0.0f);
    glfwPollEvents();
    glfwGetFramebufferSize(window_, &width_, &height_);
    if (!width_ || !height_)
      continue;
    onDraw();
  }

  onShutdown();
}

double Application::getTime() const { return glfwGetTime(); }

} // namespace nuri
