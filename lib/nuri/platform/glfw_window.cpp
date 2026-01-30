#include "nuri/platform/glfw_window.h"

#include <atomic>
#include <GLFW/glfw3.h>

namespace nuri {

namespace {

std::atomic<int> s_glfwRefCount{0};

} // namespace

struct GlfwWindow::Impl {
  GLFWwindow *window = nullptr;
};

GlfwWindow::GlfwWindow() : impl_(std::make_unique<Impl>()) {}

GlfwWindow::~GlfwWindow() {
  if (impl_ && impl_->window) {
    glfwDestroyWindow(impl_->window);
    impl_->window = nullptr;
    if (--s_glfwRefCount == 0) {
      glfwTerminate();
    }
  }
}

std::unique_ptr<GlfwWindow> GlfwWindow::create(std::string_view title,
                                               int32_t width,
                                               int32_t height) {
  int const prev = s_glfwRefCount.fetch_add(1);
  if (prev == 0) {
    if (!glfwInit()) {
      s_glfwRefCount--;
      return nullptr;
    }
  }

  auto window = std::unique_ptr<GlfwWindow>(new GlfwWindow());
  std::string titleStr(title);

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  window->impl_->window =
      glfwCreateWindow(width, height, titleStr.c_str(), nullptr, nullptr);
  if (!window->impl_->window) {
    if (--s_glfwRefCount == 0) {
      glfwTerminate();
    }
    return nullptr;
  }

  return window;
}

std::unique_ptr<Window> Window::create(std::string_view title, int32_t width,
                                       int32_t height) {
  return GlfwWindow::create(title, width, height);
}

void GlfwWindow::pollEvents() { glfwPollEvents(); }

bool GlfwWindow::shouldClose() const {
  return impl_->window && glfwWindowShouldClose(impl_->window);
}

void GlfwWindow::getFramebufferSize(int32_t &outWidth,
                                    int32_t &outHeight) const {
  if (!impl_->window) {
    outWidth = 0;
    outHeight = 0;
    return;
  }
  int fbw = 0;
  int fbh = 0;
  glfwGetFramebufferSize(impl_->window, &fbw, &fbh);
  outWidth = static_cast<int32_t>(fbw);
  outHeight = static_cast<int32_t>(fbh);
}

double GlfwWindow::getTime() const { return glfwGetTime(); }

void *GlfwWindow::nativeHandle() const { return impl_->window; }

} // namespace nuri
