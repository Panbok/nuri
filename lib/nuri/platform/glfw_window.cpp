#include "nuri/platform/glfw_window.h"

#include "nuri/core/log.h"

#include <GLFW/glfw3.h>
#include <atomic>

namespace nuri {

namespace {

std::atomic<int> s_glfwRefCount{0};

void glfwKeyCallback(GLFWwindow *window, int key, int /*scancode*/, int action,
                     int /*mods*/) {
  if (!window) {
    return;
  }
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

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
                                               int32_t width, int32_t height,
                                               bool fullscreen,
                                               bool borderlessFullscreen) {
  int const prev = s_glfwRefCount.fetch_add(1);
  if (prev == 0) {
    if (!glfwInit()) {
      s_glfwRefCount--;
      NURI_LOG_WARNING("GlfwWindow::create: glfwInit failed");
      return nullptr;
    }
  }

  auto window = std::unique_ptr<GlfwWindow>(new GlfwWindow());
  std::string titleStr(title);

  glfwDefaultWindowHints();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  auto fail = [&]() -> std::unique_ptr<GlfwWindow> {
    if (--s_glfwRefCount == 0) {
      glfwTerminate();
    }
    return nullptr;
  };

  const bool wantExclusiveFullscreen = fullscreen && !borderlessFullscreen;
  const bool wantBorderlessMonitorWindow = fullscreen && borderlessFullscreen;
  const bool wantMaxCoverageWindow = !fullscreen && (width == 0 && height == 0);

  GLFWmonitor *primaryMonitor = nullptr;
  const GLFWvidmode *primaryMode = nullptr;
  if (wantExclusiveFullscreen || wantBorderlessMonitorWindow ||
      wantMaxCoverageWindow) {
    primaryMonitor = glfwGetPrimaryMonitor();
    if (!primaryMonitor) {
      NURI_LOG_WARNING("GlfwWindow::create: glfwGetPrimaryMonitor failed");
      return fail();
    }
    primaryMode = glfwGetVideoMode(primaryMonitor);
    if (!primaryMode) {
      NURI_LOG_WARNING("GlfwWindow::create: glfwGetVideoMode failed");
      return fail();
    }
  }

  int32_t createWidth = width;
  int32_t createHeight = height;
  if (wantExclusiveFullscreen || wantBorderlessMonitorWindow) {
    createWidth = static_cast<int32_t>(primaryMode->width);
    createHeight = static_cast<int32_t>(primaryMode->height);
  } else if (wantMaxCoverageWindow) {
    int workX = 0;
    int workY = 0;
    int workW = 0;
    int workH = 0;
    glfwGetMonitorWorkarea(primaryMonitor, &workX, &workY, &workW, &workH);
    if (workW > 0 && workH > 0) {
      createWidth = static_cast<int32_t>(workW);
      createHeight = static_cast<int32_t>(workH);
    } else {
      // Fallback: use full monitor mode if work area is unavailable.
      createWidth = static_cast<int32_t>(primaryMode->width);
      createHeight = static_cast<int32_t>(primaryMode->height);
    }
  } else {
    if (width <= 0 || height <= 0) {
      NURI_LOG_WARNING("GlfwWindow::create: invalid window size (%d x %d)",
                       width, height);
      return fail();
    }
  }

  GLFWmonitor *createMonitor =
      wantExclusiveFullscreen ? primaryMonitor : nullptr;
  glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);
  glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  if (wantExclusiveFullscreen) {
    glfwWindowHint(GLFW_REFRESH_RATE, primaryMode->refreshRate);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  } else if (wantBorderlessMonitorWindow) {
    // Borderless monitor-sized windowed mode (not exclusive fullscreen).
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  } else if (wantMaxCoverageWindow) {
    // Windowed with max screen size coverage (decorated).
  }

  window->impl_->window = glfwCreateWindow(
      createWidth, createHeight, titleStr.c_str(), createMonitor, nullptr);
  if (!window->impl_->window) {
    NURI_LOG_WARNING(
        "GlfwWindow::create: glfwCreateWindow failed (%d x %d)%s", createWidth,
        createHeight,
        wantExclusiveFullscreen
            ? " [exclusive fullscreen]"
            : (wantBorderlessMonitorWindow
                   ? " [borderless monitor-sized]"
                   : (wantMaxCoverageWindow ? " [max coverage]" : "")));
    return fail();
  }

  glfwSetKeyCallback(window->impl_->window, glfwKeyCallback);

  if ((wantBorderlessMonitorWindow || wantMaxCoverageWindow) &&
      primaryMonitor) {
    int targetX = 0;
    int targetY = 0;
    if (wantMaxCoverageWindow) {
      int workX = 0;
      int workY = 0;
      int workW = 0;
      int workH = 0;
      glfwGetMonitorWorkarea(primaryMonitor, &workX, &workY, &workW, &workH);
      targetX = workX;
      targetY = workY;
    } else {
      glfwGetMonitorPos(primaryMonitor, &targetX, &targetY);
    }
    glfwSetWindowPos(window->impl_->window, targetX, targetY);
    glfwFocusWindow(window->impl_->window);
  }

  NURI_LOG_INFO("Window::create: Creating window '%.*s' (%d x %d)%s%s",
                static_cast<int>(title.size()), title.data(), createWidth,
                createHeight, fullscreen ? " [fullscreen]" : "",
                (fullscreen && borderlessFullscreen) ? " [borderless]" : "");

  return window;
}

std::unique_ptr<Window> Window::create(std::string_view title, int32_t width,
                                       int32_t height, bool fullscreen,
                                       bool borderlessFullscreen) {
  return GlfwWindow::create(title, width, height, fullscreen,
                            borderlessFullscreen);
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
