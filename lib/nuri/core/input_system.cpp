#include "nuri/core/input_system.h"
namespace nuri {

namespace {
constexpr int32_t kRawInputPriority = 1000;
}

InputSystem::InputSystem(EventManager &events) : events_(events) {
  inputSub_ = events_.subscribe<InputEvent>(
      EventChannel::RawInput, &dispatchInput, this, kRawInputPriority);
  NURI_LOG_DEBUG("InputSystem::InputSystem: Input system created");
}

InputSystem::~InputSystem() {
  (void)events_.unsubscribe(inputSub_);
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

bool InputSystem::dispatchInput(const InputEvent &event, void *user) {
  return user ? static_cast<InputSystem *>(user)->handleInput(event) : false;
}

bool InputSystem::handleInput(InputEvent event) {
  switch (event.type) {
  case InputEventType::Key: {
    const size_t idx = keyIndex(event.payload.key.key);
    if (idx >= keyDown_.size()) {
      break;
    }
    switch (event.payload.key.action) {
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
    }
    break;
  }
  case InputEventType::MouseButton: {
    const size_t idx = mouseIndex(event.payload.mouseButton.button);
    if (idx >= mouseDown_.size()) {
      break;
    }
    if (event.payload.mouseButton.action == MouseAction::Press) {
      if (!mouseDown_.test(idx)) {
        mousePressed_.set(idx);
      }
      mouseDown_.set(idx);
    } else {
      if (mouseDown_.test(idx)) {
        mouseReleased_.set(idx);
      }
      mouseDown_.reset(idx);
    }
    break;
  }
  case InputEventType::MouseMove: {
    auto &move = event.payload.mouseMove;
    move.dx = hasMousePosition_ ? move.x - mousePosition_.x : 0.0;
    move.dy = hasMousePosition_ ? move.y - mousePosition_.y : 0.0;
    hasMousePosition_ = true;
    mousePosition_ = {move.x, move.y};
    mouseDelta_ += glm::dvec2(move.dx, move.dy);
    break;
  }
  case InputEventType::MouseScroll:
    scrollDelta_ +=
        glm::dvec2(event.payload.mouseScroll.x, event.payload.mouseScroll.y);
    break;
  case InputEventType::Focus:
    if (!event.payload.focus.focused) {
      keyDown_.reset();
      mouseDown_.reset();
      hasMousePosition_ = false;
      mousePosition_ = {0.0, 0.0};
      mouseDelta_ = {0.0, 0.0};
    }
    break;
  case InputEventType::CursorEnter:
    if (!event.payload.cursorEnter.entered) {
      hasMousePosition_ = false;
      mousePosition_ = {0.0, 0.0};
      mouseDelta_ = {0.0, 0.0};
    }
    break;
  case InputEventType::Character:
    break;
  }
  events_.emit(event, EventChannel::Input);
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
