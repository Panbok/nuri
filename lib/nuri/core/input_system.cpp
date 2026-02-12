#include "nuri/core/input_system.h"

namespace nuri {

namespace {
constexpr int32_t kRawInputPriority = 1000;
}

InputSystem::InputSystem(EventManager &events) : events_(events) {
  keySub_ = events_.subscribe<RawKeyEvent>(
      EventChannel::RawInput, &onRaw<RawKeyEvent, &InputSystem::handleRawKey>,
      this, kRawInputPriority);
  charSub_ = events_.subscribe<RawCharEvent>(
      EventChannel::RawInput, &onRaw<RawCharEvent, &InputSystem::handleRawChar>,
      this, kRawInputPriority);
  mouseButtonSub_ = events_.subscribe<RawMouseButtonEvent>(
      EventChannel::RawInput,
      &onRaw<RawMouseButtonEvent, &InputSystem::handleRawMouseButton>, this,
      kRawInputPriority);
  mouseMoveSub_ = events_.subscribe<RawMouseMoveEvent>(
      EventChannel::RawInput,
      &onRaw<RawMouseMoveEvent, &InputSystem::handleRawMouseMove>, this,
      kRawInputPriority);
  mouseScrollSub_ = events_.subscribe<RawMouseScrollEvent>(
      EventChannel::RawInput,
      &onRaw<RawMouseScrollEvent, &InputSystem::handleRawMouseScroll>, this,
      kRawInputPriority);
  focusSub_ = events_.subscribe<RawFocusEvent>(
      EventChannel::RawInput,
      &onRaw<RawFocusEvent, &InputSystem::handleRawFocus>, this,
      kRawInputPriority);
  cursorEnterSub_ = events_.subscribe<RawCursorEnterEvent>(
      EventChannel::RawInput,
      &onRaw<RawCursorEnterEvent, &InputSystem::handleRawCursorEnter>, this,
      kRawInputPriority);
  NURI_LOG_DEBUG("InputSystem::InputSystem: Input system created");
}

InputSystem::~InputSystem() {
  (void)events_.unsubscribe(keySub_);
  (void)events_.unsubscribe(charSub_);
  (void)events_.unsubscribe(mouseButtonSub_);
  (void)events_.unsubscribe(mouseMoveSub_);
  (void)events_.unsubscribe(mouseScrollSub_);
  (void)events_.unsubscribe(focusSub_);
  (void)events_.unsubscribe(cursorEnterSub_);
  NURI_LOG_DEBUG("InputSystem::~InputSystem: Input system destroyed");
}

void InputSystem::beginFrame() {
  keyPressed_.reset();
  keyReleased_.reset();
  mousePressed_.reset();
  mouseReleased_.reset();
  mouseDelta_ = glm::dvec2(0.0, 0.0);
  scrollDelta_ = glm::dvec2(0.0, 0.0);
}

void InputSystem::endFrame() {}

bool InputSystem::isKeyDown(Key key) const {
  const size_t idx = keyIndex(key);
  return idx < keyDown_.size() ? keyDown_.test(idx) : false;
}

bool InputSystem::wasKeyPressed(Key key) const {
  const size_t idx = keyIndex(key);
  return idx < keyPressed_.size() ? keyPressed_.test(idx) : false;
}

bool InputSystem::wasKeyReleased(Key key) const {
  const size_t idx = keyIndex(key);
  return idx < keyReleased_.size() ? keyReleased_.test(idx) : false;
}

bool InputSystem::isMouseButtonDown(MouseButton button) const {
  const size_t idx = mouseIndex(button);
  return idx < mouseDown_.size() ? mouseDown_.test(idx) : false;
}

bool InputSystem::wasMouseButtonPressed(MouseButton button) const {
  const size_t idx = mouseIndex(button);
  return idx < mousePressed_.size() ? mousePressed_.test(idx) : false;
}

bool InputSystem::wasMouseButtonReleased(MouseButton button) const {
  const size_t idx = mouseIndex(button);
  return idx < mouseReleased_.size() ? mouseReleased_.test(idx) : false;
}

glm::dvec2 InputSystem::mousePosition() const { return mousePosition_; }

glm::dvec2 InputSystem::mouseDelta() const { return mouseDelta_; }

glm::dvec2 InputSystem::scrollDelta() const { return scrollDelta_; }

bool InputSystem::handleRawKey(const RawKeyEvent &event) {
  const size_t idx = keyIndex(event.key);
  if (idx < keyDown_.size()) {
    switch (event.action) {
    case KeyAction::Press:
      if (!keyDown_.test(idx)) {
        keyPressed_.set(idx);
      }
      keyDown_.set(idx);
      break;
    case KeyAction::Repeat:
      keyDown_.set(idx);
      break;
    case KeyAction::Release:
      if (keyDown_.test(idx)) {
        keyReleased_.set(idx);
      }
      keyDown_.reset(idx);
      break;
    default:
      break;
    }
  }

  InputEvent out{};
  out.type = InputEventType::Key;
  out.deviceId = event.deviceId;
  out.payload.key.key = event.key;
  out.payload.key.scancode = event.scancode;
  out.payload.key.action = event.action;
  out.payload.key.mods = event.mods;
  events_.emit(out, EventChannel::Input);
  return false;
}

bool InputSystem::handleRawChar(const RawCharEvent &event) {
  InputEvent out{};
  out.type = InputEventType::Character;
  out.deviceId = event.deviceId;
  out.payload.character.codepoint = event.codepoint;
  events_.emit(out, EventChannel::Input);
  return false;
}

bool InputSystem::handleRawMouseButton(const RawMouseButtonEvent &event) {
  const size_t idx = mouseIndex(event.button);
  if (idx < mouseDown_.size()) {
    switch (event.action) {
    case MouseAction::Press:
      if (!mouseDown_.test(idx)) {
        mousePressed_.set(idx);
      }
      mouseDown_.set(idx);
      break;
    case MouseAction::Release:
      if (mouseDown_.test(idx)) {
        mouseReleased_.set(idx);
      }
      mouseDown_.reset(idx);
      break;
    default:
      break;
    }
  }

  InputEvent out{};
  out.type = InputEventType::MouseButton;
  out.deviceId = event.deviceId;
  out.payload.mouseButton.button = event.button;
  out.payload.mouseButton.action = event.action;
  out.payload.mouseButton.mods = event.mods;
  events_.emit(out, EventChannel::Input);
  return false;
}

bool InputSystem::handleRawMouseMove(const RawMouseMoveEvent &event) {
  double dx = 0.0;
  double dy = 0.0;
  if (hasMousePosition_) {
    dx = event.x - mousePosition_.x;
    dy = event.y - mousePosition_.y;
  } else {
    hasMousePosition_ = true;
  }

  mousePosition_ = glm::dvec2(event.x, event.y);
  mouseDelta_ += glm::dvec2(dx, dy);

  InputEvent out{};
  out.type = InputEventType::MouseMove;
  out.deviceId = event.deviceId;
  out.payload.mouseMove.x = event.x;
  out.payload.mouseMove.y = event.y;
  out.payload.mouseMove.dx = dx;
  out.payload.mouseMove.dy = dy;
  events_.emit(out, EventChannel::Input);
  return false;
}

bool InputSystem::handleRawMouseScroll(const RawMouseScrollEvent &event) {
  scrollDelta_ += glm::dvec2(event.xOffset, event.yOffset);

  InputEvent out{};
  out.type = InputEventType::MouseScroll;
  out.deviceId = event.deviceId;
  out.payload.mouseScroll.x = event.xOffset;
  out.payload.mouseScroll.y = event.yOffset;
  events_.emit(out, EventChannel::Input);
  return false;
}

bool InputSystem::handleRawFocus(const RawFocusEvent &event) {
  if (!event.focused) {
    keyDown_.reset();
    mouseDown_.reset();
    hasMousePosition_ = false;
    mousePosition_ = glm::dvec2(0.0, 0.0);
    mouseDelta_ = glm::dvec2(0.0, 0.0);
  }

  InputEvent out{};
  out.type = InputEventType::Focus;
  out.deviceId = event.deviceId;
  out.payload.focus.focused = event.focused;
  events_.emit(out, EventChannel::Input);
  return false;
}

bool InputSystem::handleRawCursorEnter(const RawCursorEnterEvent &event) {
  if (!event.entered) {
    hasMousePosition_ = false;
    mousePosition_ = glm::dvec2(0.0, 0.0);
    mouseDelta_ = glm::dvec2(0.0, 0.0);
  }

  InputEvent out{};
  out.type = InputEventType::CursorEnter;
  out.deviceId = event.deviceId;
  out.payload.cursorEnter.entered = event.entered;
  events_.emit(out, EventChannel::Input);
  return false;
}

size_t InputSystem::keyIndex(Key key) {
  if (key == Key::Unknown) {
    return static_cast<size_t>(-1);
  }
  return static_cast<size_t>(key);
}

size_t InputSystem::mouseIndex(MouseButton button) {
  if (button == MouseButton::Unknown) {
    return static_cast<size_t>(-1);
  }
  return static_cast<size_t>(button);
}

} // namespace nuri
