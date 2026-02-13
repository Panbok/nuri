#pragma once

#include "nuri/core/event_manager.h"
#include "nuri/core/input_events.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"

#include <GLFW/glfw3.h>
#include <memory>

namespace nuri {

class NURI_API ImGuiGlfwPlatform final {
public:
  static std::unique_ptr<ImGuiGlfwPlatform> create(Window &window,
                                                   EventManager &events);
  ~ImGuiGlfwPlatform();

  ImGuiGlfwPlatform(const ImGuiGlfwPlatform &) = delete;
  ImGuiGlfwPlatform &operator=(const ImGuiGlfwPlatform &) = delete;
  ImGuiGlfwPlatform(ImGuiGlfwPlatform &&) = delete;
  ImGuiGlfwPlatform &operator=(ImGuiGlfwPlatform &&) = delete;

  void newFrame();

private:
  ImGuiGlfwPlatform(Window &window, EventManager &events);

  template <typename T, bool (ImGuiGlfwPlatform::*Fn)(const T &)>
  static bool onRaw(const T &event, void *user) {
    if (!user) {
      return false;
    }
    return (static_cast<ImGuiGlfwPlatform *>(user)->*Fn)(event);
  }

  bool handleRawKey(const RawKeyEvent &event);
  bool handleRawChar(const RawCharEvent &event);
  bool handleRawMouseButton(const RawMouseButtonEvent &event);
  bool handleRawMouseMove(const RawMouseMoveEvent &event);
  bool handleRawMouseScroll(const RawMouseScrollEvent &event);
  bool handleRawFocus(const RawFocusEvent &event);
  bool handleRawCursorEnter(const RawCursorEnterEvent &event);

  Window &window_;
  EventManager &events_;
  GLFWwindow *glfwWindow_ = nullptr;
  SubscriptionToken keySub_{};
  SubscriptionToken charSub_{};
  SubscriptionToken mouseButtonSub_{};
  SubscriptionToken mouseMoveSub_{};
  SubscriptionToken mouseScrollSub_{};
  SubscriptionToken focusSub_{};
  SubscriptionToken cursorEnterSub_{};
};

} // namespace nuri
