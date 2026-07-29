#include "nuri/core/event_manager.h"
#include "nuri/core/input_events.h"
#include "nuri/core/log.h"
#include "nuri/core/window.h"
#include <GLFW/glfw3.h>
#include <array>
#include <atomic>
#include <mutex>
namespace nuri {

class GlfwWindow final : public Window {
public:
  static std::unique_ptr<GlfwWindow> create(std::string_view title,
                                            int32_t width, int32_t height,
                                            WindowMode mode);
  ~GlfwWindow() override;
  void pollEvents() override;
  bool shouldClose() const override;
  void getWindowSize(int32_t &width, int32_t &height) const override;
  void getFramebufferSize(int32_t &width, int32_t &height) const override;
  void setWindowSize(int32_t width, int32_t height) override;
  double getTime() const override;
  void *nativeHandle() const override;
  void requestClose() override;
  void setCursorMode(CursorMode mode) override;
  CursorMode getCursorMode() const override;
  void bindEventManager(EventManager *events) override;

private:
  GLFWwindow *window_ = nullptr;
  CursorMode cursorMode_ = CursorMode::Normal;
};

namespace {
struct WindowModePlan {
  bool exclusive;
  bool monitorSized;
  bool visible;
  bool decorated;
  bool resizable;
  const char *logSuffix;
};

constexpr std::array kWindowModePlans{
    WindowModePlan{false, false, true, true, true, ""},
    WindowModePlan{false, false, false, true, true, " [hidden]"},
    WindowModePlan{true, true, true, false, false, " [fullscreen]"},
    WindowModePlan{false, true, true, false, false, " [borderless fullscreen]"},
};

[[nodiscard]] constexpr const WindowModePlan &windowModePlan(WindowMode mode) {
  return kWindowModePlans[static_cast<size_t>(mode)];
}

std::atomic<int> s_glfwRefCount{0};
std::mutex s_glfwMutex;
bool s_glfwInitialized = false;
[[nodiscard]] bool acquireGlfw() {
  std::lock_guard lock(s_glfwMutex);
  if (s_glfwRefCount.load() == 0) {
    s_glfwInitialized = glfwInit() == GLFW_TRUE;
    if (!s_glfwInitialized) {
      return false;
    }
  }
  s_glfwRefCount.fetch_add(1);
  return true;
}
void releaseGlfw() {
  std::lock_guard lock(s_glfwMutex);
  const int oldCount = s_glfwRefCount.load();
  if (oldCount <= 0) {
    return;
  }
  s_glfwRefCount.store(oldCount - 1);
  if (oldCount == 1 && s_glfwInitialized) {
    glfwTerminate();
    s_glfwInitialized = false;
  }
}
} // namespace

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
void emitInput(GLFWwindow *window, const InputEvent &event) {
  EventManager *events = getEventManager(window);
  if (!events) {
    return;
  }
  events->emit(event, EventChannel::RawInput);
}
void emitKeyEvent(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  InputEvent event{};
  event.type = InputEventType::Key;
  event.payload.key = {.key = mapGlfwKey(key),
                       .scancode = scancode,
                       .action = mapGlfwKeyAction(action),
                       .mods = mapGlfwMods(mods)};
  emitInput(window, event);
}
void emitCharacterEvent(GLFWwindow *window, unsigned int codepoint) {
  InputEvent event{};
  event.type = InputEventType::Character;
  event.payload.character = {.codepoint = static_cast<uint32_t>(codepoint)};
  emitInput(window, event);
}
void emitMouseButtonEvent(GLFWwindow *window, int button, int action,
                          int mods) {
  InputEvent event{};
  event.type = InputEventType::MouseButton;
  event.payload.mouseButton = {.button = mapGlfwMouseButton(button),
                               .action = mapGlfwMouseAction(action),
                               .mods = mapGlfwMods(mods)};
  emitInput(window, event);
}
void emitMouseMoveEvent(GLFWwindow *window, double x, double y) {
  InputEvent event{};
  event.type = InputEventType::MouseMove;
  event.payload.mouseMove = {.x = x, .y = y};
  emitInput(window, event);
}
void emitMouseScrollEvent(GLFWwindow *window, double xOffset, double yOffset) {
  InputEvent event{};
  event.type = InputEventType::MouseScroll;
  event.payload.mouseScroll = {.x = xOffset, .y = yOffset};
  emitInput(window, event);
}
void emitFocusEvent(GLFWwindow *window, int focused) {
  InputEvent event{};
  event.type = InputEventType::Focus;
  event.payload.focus = {.focused = focused != 0};
  emitInput(window, event);
}
void emitCursorEnterEvent(GLFWwindow *window, int entered) {
  InputEvent event{};
  event.type = InputEventType::CursorEnter;
  event.payload.cursorEnter = {.entered = entered != 0};
  emitInput(window, event);
}
} // namespace

GlfwWindow::~GlfwWindow() {
  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
    releaseGlfw();
  }
}

std::unique_ptr<GlfwWindow> GlfwWindow::create(std::string_view title,
                                               int32_t width, int32_t height,
                                               WindowMode mode) {
  if (!acquireGlfw()) {
    NURI_LOG_WARNING("GlfwWindow::create: glfwInit failed");
    return nullptr;
  }
  auto window = std::unique_ptr<GlfwWindow>(new GlfwWindow());
  std::string titleStr(title);
  glfwDefaultWindowHints();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  auto fail = [&]() -> std::unique_ptr<GlfwWindow> {
    releaseGlfw();
    return nullptr;
  };
  const WindowModePlan &plan = windowModePlan(mode);
  const bool wantMaxCoverageWindow =
      (mode == WindowMode::Windowed && width == 0 && height == 0);
  GLFWmonitor *primaryMonitor = nullptr;
  const GLFWvidmode *primaryMode = nullptr;
  if (plan.monitorSized || wantMaxCoverageWindow) {
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
  if (plan.monitorSized) {
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
  GLFWmonitor *createMonitor = plan.exclusive ? primaryMonitor : nullptr;
  glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);
  glfwWindowHint(GLFW_DECORATED, plan.decorated ? GLFW_TRUE : GLFW_FALSE);
  glfwWindowHint(GLFW_RESIZABLE, plan.resizable ? GLFW_TRUE : GLFW_FALSE);
  glfwWindowHint(GLFW_VISIBLE, plan.visible ? GLFW_TRUE : GLFW_FALSE);
  if (plan.exclusive) {
    glfwWindowHint(GLFW_REFRESH_RATE, primaryMode->refreshRate);
  }
  window->window_ = glfwCreateWindow(createWidth, createHeight,
                                     titleStr.c_str(), createMonitor, nullptr);
  if (!window->window_) {
    NURI_LOG_WARNING(
        "GlfwWindow::create: glfwCreateWindow failed (%d x %d)%s", createWidth,
        createHeight,
        plan.exclusive
            ? " [exclusive fullscreen]"
            : (plan.monitorSized
                   ? " [borderless monitor-sized]"
                   : (wantMaxCoverageWindow ? " [max coverage]" : "")));
    return fail();
  }
  glfwSetWindowUserPointer(window->window_, nullptr);
  glfwSetKeyCallback(window->window_, emitKeyEvent);
  glfwSetCharCallback(window->window_, emitCharacterEvent);
  glfwSetMouseButtonCallback(window->window_, emitMouseButtonEvent);
  glfwSetCursorPosCallback(window->window_, emitMouseMoveEvent);
  glfwSetScrollCallback(window->window_, emitMouseScrollEvent);
  glfwSetWindowFocusCallback(window->window_, emitFocusEvent);
  glfwSetCursorEnterCallback(window->window_, emitCursorEnterEvent);
  glfwSetInputMode(window->window_, GLFW_CURSOR,
                   toGlfwCursorMode(window->cursorMode_));
  if (((!plan.exclusive && plan.monitorSized) || wantMaxCoverageWindow) &&
      primaryMonitor) {
    int targetX = 0;
    int targetY = 0;
    if (wantMaxCoverageWindow) {
      targetX = workAreaX;
      targetY = workAreaY;
    } else {
      glfwGetMonitorPos(primaryMonitor, &targetX, &targetY);
    }
    glfwSetWindowPos(window->window_, targetX, targetY);
    glfwFocusWindow(window->window_);
  }
  NURI_LOG_DEBUG("Window::create: Creating window '%.*s' (%d x %d)%s",
                 static_cast<int>(title.size()), title.data(), createWidth,
                 createHeight, plan.logSuffix);
  return window;
}

std::unique_ptr<Window> Window::create(std::string_view title, int32_t width,
                                       int32_t height, WindowMode mode) {
  return GlfwWindow::create(title, width, height, mode);
}

void GlfwWindow::pollEvents() { glfwPollEvents(); }

bool GlfwWindow::shouldClose() const {
  return window_ && glfwWindowShouldClose(window_);
}

void GlfwWindow::getWindowSize(int32_t &outWidth, int32_t &outHeight) const {
  if (!window_) {
    outWidth = 0;
    outHeight = 0;
    return;
  }
  int width = 0;
  int height = 0;
  glfwGetWindowSize(window_, &width, &height);
  outWidth = static_cast<int32_t>(width);
  outHeight = static_cast<int32_t>(height);
}

void GlfwWindow::getFramebufferSize(int32_t &outWidth,
                                    int32_t &outHeight) const {
  if (!window_) {
    outWidth = 0;
    outHeight = 0;
    return;
  }
  int fbw = 0;
  int fbh = 0;
  glfwGetFramebufferSize(window_, &fbw, &fbh);
  outWidth = static_cast<int32_t>(fbw);
  outHeight = static_cast<int32_t>(fbh);
}

void GlfwWindow::setWindowSize(int32_t width, int32_t height) {
  if (window_ && width > 0 && height > 0) {
    glfwSetWindowSize(window_, width, height);
  }
}

double GlfwWindow::getTime() const { return glfwGetTime(); }

void *GlfwWindow::nativeHandle() const { return window_; }

void GlfwWindow::requestClose() {
  if (window_) {
    glfwSetWindowShouldClose(window_, GLFW_TRUE);
  }
}

void GlfwWindow::setCursorMode(CursorMode mode) {
  if (!window_) {
    return;
  }
  cursorMode_ = mode;
  glfwSetInputMode(window_, GLFW_CURSOR, toGlfwCursorMode(mode));
}

CursorMode GlfwWindow::getCursorMode() const { return cursorMode_; }

void GlfwWindow::bindEventManager(EventManager *events) {
  if (window_) {
    glfwSetWindowUserPointer(window_, events);
  }
}

} // namespace nuri
