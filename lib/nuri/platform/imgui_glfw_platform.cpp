#include "nuri/platform/imgui_glfw_platform.h"

#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>

namespace nuri {

namespace {

int toGlfwKey(Key key) {
  if (key == Key::Unknown) {
    return GLFW_KEY_UNKNOWN;
  }
  return static_cast<int>(key);
}

int toGlfwKeyAction(KeyAction action) {
  switch (action) {
  case KeyAction::Press:
    return GLFW_PRESS;
  case KeyAction::Release:
    return GLFW_RELEASE;
  case KeyAction::Repeat:
    return GLFW_REPEAT;
  default:
    return GLFW_RELEASE;
  }

  return GLFW_RELEASE;
}

int toGlfwMouseButton(MouseButton button) {
  if (button == MouseButton::Unknown) {
    return -1;
  }
  return static_cast<int>(button);
}

int toGlfwMouseAction(MouseAction action) {
  switch (action) {
  case MouseAction::Press:
    return GLFW_PRESS;
  case MouseAction::Release:
    return GLFW_RELEASE;
  default:
    return GLFW_RELEASE;
  }
  return GLFW_RELEASE;
}

int toGlfwMods(KeyMod mods) { return static_cast<int>(mods); }

} // namespace

std::unique_ptr<ImGuiGlfwPlatform>
ImGuiGlfwPlatform::create(Window &window, EventManager &events) {
  return std::unique_ptr<ImGuiGlfwPlatform>(
      new ImGuiGlfwPlatform(window, events));
}

ImGuiGlfwPlatform::ImGuiGlfwPlatform(Window &window, EventManager &events)
    : window_(window), events_(events) {
  glfwWindow_ = static_cast<GLFWwindow *>(window_.nativeHandle());
  ImGui_ImplGlfw_InitForVulkan(glfwWindow_, /*install_callbacks=*/false);

  keySub_ = events_.subscribe<RawKeyEvent>(
      EventChannel::RawInput,
      &onRaw<RawKeyEvent, &ImGuiGlfwPlatform::handleRawKey>, this);
  charSub_ = events_.subscribe<RawCharEvent>(
      EventChannel::RawInput,
      &onRaw<RawCharEvent, &ImGuiGlfwPlatform::handleRawChar>, this);
  mouseButtonSub_ = events_.subscribe<RawMouseButtonEvent>(
      EventChannel::RawInput,
      &onRaw<RawMouseButtonEvent, &ImGuiGlfwPlatform::handleRawMouseButton>,
      this);
  mouseMoveSub_ = events_.subscribe<RawMouseMoveEvent>(
      EventChannel::RawInput,
      &onRaw<RawMouseMoveEvent, &ImGuiGlfwPlatform::handleRawMouseMove>, this);
  mouseScrollSub_ = events_.subscribe<RawMouseScrollEvent>(
      EventChannel::RawInput,
      &onRaw<RawMouseScrollEvent, &ImGuiGlfwPlatform::handleRawMouseScroll>,
      this);
  focusSub_ = events_.subscribe<RawFocusEvent>(
      EventChannel::RawInput,
      &onRaw<RawFocusEvent, &ImGuiGlfwPlatform::handleRawFocus>, this);
  cursorEnterSub_ = events_.subscribe<RawCursorEnterEvent>(
      EventChannel::RawInput,
      &onRaw<RawCursorEnterEvent, &ImGuiGlfwPlatform::handleRawCursorEnter>,
      this);
}

ImGuiGlfwPlatform::~ImGuiGlfwPlatform() {
  (void)events_.unsubscribe(keySub_);
  (void)events_.unsubscribe(charSub_);
  (void)events_.unsubscribe(mouseButtonSub_);
  (void)events_.unsubscribe(mouseMoveSub_);
  (void)events_.unsubscribe(mouseScrollSub_);
  (void)events_.unsubscribe(focusSub_);
  (void)events_.unsubscribe(cursorEnterSub_);

  ImGui_ImplGlfw_Shutdown();
  glfwWindow_ = nullptr;
}

void ImGuiGlfwPlatform::newFrame() { ImGui_ImplGlfw_NewFrame(); }

bool ImGuiGlfwPlatform::handleRawKey(const RawKeyEvent &event) {
  if (!glfwWindow_) {
    return false;
  }
  ImGui_ImplGlfw_KeyCallback(glfwWindow_, toGlfwKey(event.key), event.scancode,
                             toGlfwKeyAction(event.action),
                             toGlfwMods(event.mods));
  return false;
}

bool ImGuiGlfwPlatform::handleRawChar(const RawCharEvent &event) {
  if (!glfwWindow_) {
    return false;
  }
  ImGui_ImplGlfw_CharCallback(glfwWindow_, event.codepoint);
  return false;
}

bool ImGuiGlfwPlatform::handleRawMouseButton(const RawMouseButtonEvent &event) {
  if (!glfwWindow_) {
    return false;
  }
  const int button = toGlfwMouseButton(event.button);
  if (button < 0) {
    return false;
  }
  ImGui_ImplGlfw_MouseButtonCallback(glfwWindow_, button,
                                     toGlfwMouseAction(event.action),
                                     toGlfwMods(event.mods));
  return false;
}

bool ImGuiGlfwPlatform::handleRawMouseMove(const RawMouseMoveEvent &event) {
  if (!glfwWindow_) {
    return false;
  }
  ImGui_ImplGlfw_CursorPosCallback(glfwWindow_, event.x, event.y);
  return false;
}

bool ImGuiGlfwPlatform::handleRawMouseScroll(const RawMouseScrollEvent &event) {
  if (!glfwWindow_) {
    return false;
  }
  ImGui_ImplGlfw_ScrollCallback(glfwWindow_, event.xOffset, event.yOffset);
  return false;
}

bool ImGuiGlfwPlatform::handleRawFocus(const RawFocusEvent &event) {
  if (!glfwWindow_) {
    return false;
  }
  ImGui_ImplGlfw_WindowFocusCallback(glfwWindow_, event.focused ? 1 : 0);
  return false;
}

bool ImGuiGlfwPlatform::handleRawCursorEnter(const RawCursorEnterEvent &event) {
  if (!glfwWindow_) {
    return false;
  }
  ImGui_ImplGlfw_CursorEnterCallback(glfwWindow_, event.entered ? 1 : 0);
  return false;
}

} // namespace nuri
