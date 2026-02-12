#pragma once

#include "nuri/core/event_manager.h"
#include "nuri/core/input_events.h"
#include "nuri/defines.h"

#include <bitset>

#include <glm/vec2.hpp>

namespace nuri {

class NURI_API InputSystem {
public:
  explicit InputSystem(EventManager &events);
  ~InputSystem();

  InputSystem(const InputSystem &) = delete;
  InputSystem &operator=(const InputSystem &) = delete;
  InputSystem(InputSystem &&) = delete;
  InputSystem &operator=(InputSystem &&) = delete;

  void beginFrame();
  void endFrame();

  bool isKeyDown(Key key) const;
  bool wasKeyPressed(Key key) const;
  bool wasKeyReleased(Key key) const;

  bool isMouseButtonDown(MouseButton button) const;
  bool wasMouseButtonPressed(MouseButton button) const;
  bool wasMouseButtonReleased(MouseButton button) const;

  glm::dvec2 mousePosition() const;
  glm::dvec2 mouseDelta() const;
  glm::dvec2 scrollDelta() const;

private:
  template <typename T, bool (InputSystem::*Fn)(const T &)>
  static bool onRaw(const T &event, void *user) {
    if (!user) {
      return false;
    }
    return (static_cast<InputSystem *>(user)->*Fn)(event);
  }

  bool handleRawKey(const RawKeyEvent &event);
  bool handleRawChar(const RawCharEvent &event);
  bool handleRawMouseButton(const RawMouseButtonEvent &event);
  bool handleRawMouseMove(const RawMouseMoveEvent &event);
  bool handleRawMouseScroll(const RawMouseScrollEvent &event);
  bool handleRawFocus(const RawFocusEvent &event);
  bool handleRawCursorEnter(const RawCursorEnterEvent &event);

  static size_t keyIndex(Key key);
  static size_t mouseIndex(MouseButton button);

  EventManager &events_;
  SubscriptionToken keySub_{};
  SubscriptionToken charSub_{};
  SubscriptionToken mouseButtonSub_{};
  SubscriptionToken mouseMoveSub_{};
  SubscriptionToken mouseScrollSub_{};
  SubscriptionToken focusSub_{};
  SubscriptionToken cursorEnterSub_{};

  std::bitset<static_cast<size_t>(Key::Count)> keyDown_{};
  std::bitset<static_cast<size_t>(Key::Count)> keyPressed_{};
  std::bitset<static_cast<size_t>(Key::Count)> keyReleased_{};
  std::bitset<static_cast<size_t>(MouseButton::Count)> mouseDown_{};
  std::bitset<static_cast<size_t>(MouseButton::Count)> mousePressed_{};
  std::bitset<static_cast<size_t>(MouseButton::Count)> mouseReleased_{};

  glm::dvec2 mousePosition_{0.0, 0.0};
  glm::dvec2 mouseDelta_{0.0, 0.0};
  glm::dvec2 scrollDelta_{0.0, 0.0};
  bool hasMousePosition_ = false;
};

} // namespace nuri
