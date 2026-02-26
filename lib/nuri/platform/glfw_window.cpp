#include "nuri/platform/glfw_window.h"

#include "nuri/core/event_manager.h"
#include "nuri/core/input_events.h"
#include "nuri/core/log.h"

#include <GLFW/glfw3.h>
#include <atomic>
#include <mutex>

namespace nuri {

namespace {

std::atomic<int> s_glfwRefCount{0};
std::once_flag s_glfwInitOnce;
std::atomic<bool> s_glfwInitSucceeded{false};

void doGlfwInit() { s_glfwInitSucceeded.store(glfwInit()); }

} // namespace

struct GlfwWindow::Impl {
  GLFWwindow *window = nullptr;
  CursorMode cursorMode = CursorMode::Normal;
};

GlfwWindow::GlfwWindow() : impl_(std::make_unique<Impl>()) {}

namespace {

Key mapGlfwKey(int key) {
  if (key == GLFW_KEY_UNKNOWN) {
    return Key::Unknown;
  }
  if (key < 0 || key >= static_cast<int>(Key::Count)) {
    return Key::Unknown;
  }
  return static_cast<Key>(static_cast<uint16_t>(key));
}

KeyAction mapGlfwKeyAction(int action) {
  switch (action) {
  case GLFW_PRESS:
    return KeyAction::Press;
  case GLFW_RELEASE:
    return KeyAction::Release;
  case GLFW_REPEAT:
    return KeyAction::Repeat;
  default:
    return KeyAction::Release;
  }
}

MouseButton mapGlfwMouseButton(int button) {
  if (button < 0 || button >= static_cast<int>(MouseButton::Count)) {
    return MouseButton::Unknown;
  }
  return static_cast<MouseButton>(static_cast<uint8_t>(button));
}

MouseAction mapGlfwMouseAction(int action) {
  return action == GLFW_PRESS ? MouseAction::Press : MouseAction::Release;
}

KeyMod mapGlfwMods(int mods) {
  KeyMod out = KeyMod::None;
  if ((mods & GLFW_MOD_SHIFT) != 0) {
    out |= KeyMod::Shift;
  }
  if ((mods & GLFW_MOD_CONTROL) != 0) {
    out |= KeyMod::Control;
  }
  if ((mods & GLFW_MOD_ALT) != 0) {
    out |= KeyMod::Alt;
  }
  if ((mods & GLFW_MOD_SUPER) != 0) {
    out |= KeyMod::Super;
  }
#if defined(GLFW_MOD_CAPS_LOCK)
  if ((mods & GLFW_MOD_CAPS_LOCK) != 0) {
    out |= KeyMod::CapsLock;
  }
#endif
#if defined(GLFW_MOD_NUM_LOCK)
  if ((mods & GLFW_MOD_NUM_LOCK) != 0) {
    out |= KeyMod::NumLock;
  }
#endif
  return out;
}

int toGlfwCursorMode(CursorMode mode) {
  switch (mode) {
  case CursorMode::Normal:
    return GLFW_CURSOR_NORMAL;
  case CursorMode::Hidden:
    return GLFW_CURSOR_HIDDEN;
  case CursorMode::Disabled:
    return GLFW_CURSOR_DISABLED;
  }
  return GLFW_CURSOR_NORMAL;
}

EventManager *getEventManager(GLFWwindow *window) {
  if (!window) {
    return nullptr;
  }
  return static_cast<EventManager *>(glfwGetWindowUserPointer(window));
}

template <typename T> void emitRaw(GLFWwindow *window, const T &event) {
  EventManager *events = getEventManager(window);
  if (!events) {
    return;
  }
  events->emit(event, EventChannel::RawInput);
}

void emitRawKeyEvent(GLFWwindow *window, int key, int scancode, int action,
                     int mods) {
  const RawKeyEvent event{
      .key = mapGlfwKey(key),
      .scancode = scancode,
      .action = mapGlfwKeyAction(action),
      .mods = mapGlfwMods(mods),
  };
  emitRaw(window, event);
}

void emitRawCharEvent(GLFWwindow *window, unsigned int codepoint) {
  const RawCharEvent event{
      .codepoint = static_cast<uint32_t>(codepoint),
  };
  emitRaw(window, event);
}

void emitRawMouseButtonEvent(GLFWwindow *window, int button, int action,
                             int mods) {
  const RawMouseButtonEvent event{
      .button = mapGlfwMouseButton(button),
      .action = mapGlfwMouseAction(action),
      .mods = mapGlfwMods(mods),
  };
  emitRaw(window, event);
}

void emitRawMouseMoveEvent(GLFWwindow *window, double x, double y) {
  const RawMouseMoveEvent event{
      .x = x,
      .y = y,
  };
  emitRaw(window, event);
}

void emitRawMouseScrollEvent(GLFWwindow *window, double xOffset,
                             double yOffset) {
  const RawMouseScrollEvent event{
      .xOffset = xOffset,
      .yOffset = yOffset,
  };
  emitRaw(window, event);
}

void emitRawFocusEvent(GLFWwindow *window, int focused) {
  const RawFocusEvent event{
      .focused = focused != 0,
  };
  emitRaw(window, event);
}

void emitRawCursorEnterEvent(GLFWwindow *window, int entered) {
  const RawCursorEnterEvent event{
      .entered = entered != 0,
  };
  emitRaw(window, event);
}

} // namespace

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
                                               WindowMode mode) {
  s_glfwRefCount.fetch_add(1);
  std::call_once(s_glfwInitOnce, doGlfwInit);
  if (!s_glfwInitSucceeded.load()) {
    s_glfwRefCount--;
    NURI_LOG_WARNING("GlfwWindow::create: glfwInit failed");
    return nullptr;
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

  const bool wantExclusiveFullscreen = (mode == WindowMode::Fullscreen);
  const bool wantBorderlessMonitorWindow =
      (mode == WindowMode::BorderlessFullscreen);
  const bool wantMaxCoverageWindow =
      (mode == WindowMode::Windowed && width == 0 && height == 0);

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

  int workAreaX = 0;
  int workAreaY = 0;

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
    workAreaX = workX;
    workAreaY = workY;
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

  glfwSetWindowUserPointer(window->impl_->window, nullptr);
  glfwSetKeyCallback(window->impl_->window, emitRawKeyEvent);
  glfwSetCharCallback(window->impl_->window, emitRawCharEvent);
  glfwSetMouseButtonCallback(window->impl_->window, emitRawMouseButtonEvent);
  glfwSetCursorPosCallback(window->impl_->window, emitRawMouseMoveEvent);
  glfwSetScrollCallback(window->impl_->window, emitRawMouseScrollEvent);
  glfwSetWindowFocusCallback(window->impl_->window, emitRawFocusEvent);
  glfwSetCursorEnterCallback(window->impl_->window, emitRawCursorEnterEvent);
  glfwSetInputMode(window->impl_->window, GLFW_CURSOR,
                   toGlfwCursorMode(window->impl_->cursorMode));

  if ((wantBorderlessMonitorWindow || wantMaxCoverageWindow) &&
      primaryMonitor) {
    int targetX = 0;
    int targetY = 0;
    if (wantMaxCoverageWindow) {
      targetX = workAreaX;
      targetY = workAreaY;
    } else {
      glfwGetMonitorPos(primaryMonitor, &targetX, &targetY);
    }
    glfwSetWindowPos(window->impl_->window, targetX, targetY);
    glfwFocusWindow(window->impl_->window);
  }

  const char *modeStr = (mode == WindowMode::Fullscreen) ? " [fullscreen]"
                        : (mode == WindowMode::BorderlessFullscreen)
                            ? " [borderless fullscreen]"
                            : "";
  NURI_LOG_DEBUG("Window::create: Creating window '%.*s' (%d x %d)%s",
                 static_cast<int>(title.size()), title.data(), createWidth,
                 createHeight, modeStr);

  return window;
}

std::unique_ptr<Window> Window::create(std::string_view title, int32_t width,
                                       int32_t height, WindowMode mode) {
  return GlfwWindow::create(title, width, height, mode);
}

void GlfwWindow::pollEvents() { glfwPollEvents(); }

bool GlfwWindow::shouldClose() const {
  return impl_->window && glfwWindowShouldClose(impl_->window);
}

void GlfwWindow::getWindowSize(int32_t &outWidth, int32_t &outHeight) const {
  if (!impl_->window) {
    outWidth = 0;
    outHeight = 0;
    return;
  }
  int width = 0;
  int height = 0;
  glfwGetWindowSize(impl_->window, &width, &height);
  outWidth = static_cast<int32_t>(width);
  outHeight = static_cast<int32_t>(height);
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

void GlfwWindow::requestClose() {
  if (impl_ && impl_->window) {
    glfwSetWindowShouldClose(impl_->window, GLFW_TRUE);
  }
}

void GlfwWindow::setCursorMode(CursorMode mode) {
  if (!impl_ || !impl_->window) {
    return;
  }

  impl_->cursorMode = mode;
  glfwSetInputMode(impl_->window, GLFW_CURSOR, toGlfwCursorMode(mode));
}

CursorMode GlfwWindow::getCursorMode() const {
  if (!impl_) {
    return CursorMode::Normal;
  }
  return impl_->cursorMode;
}

void GlfwWindow::bindEventManager(EventManager *events) {
  if (impl_ && impl_->window) {
    glfwSetWindowUserPointer(impl_->window, events);
  }
}

} // namespace nuri
