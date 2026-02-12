#pragma once

#include "nuri/core/input_codes.h"

#include <type_traits>

namespace nuri {

struct RawKeyEvent {
  Key key = Key::Unknown;
  int32_t scancode = 0;
  KeyAction action = KeyAction::Release;
  KeyMod mods = KeyMod::None;
  uint32_t deviceId = 0;
};

struct RawCharEvent {
  uint32_t codepoint = 0;
  uint32_t deviceId = 0;
};

struct RawMouseButtonEvent {
  MouseButton button = MouseButton::Left;
  MouseAction action = MouseAction::Release;
  KeyMod mods = KeyMod::None;
  uint32_t deviceId = 0;
};

struct RawMouseMoveEvent {
  double x = 0.0;
  double y = 0.0;
  uint32_t deviceId = 0;
};

struct RawMouseScrollEvent {
  double xOffset = 0.0;
  double yOffset = 0.0;
  uint32_t deviceId = 0;
};

struct RawFocusEvent {
  bool focused = false;
  uint32_t deviceId = 0;
};

struct RawCursorEnterEvent {
  bool entered = false;
  uint32_t deviceId = 0;
};

enum class InputEventType : uint8_t {
  Key,
  Character,
  MouseButton,
  MouseMove,
  MouseScroll,
  Focus,
  CursorEnter,
};

struct InputKeyData {
  Key key = Key::Unknown;
  int32_t scancode = 0;
  KeyAction action = KeyAction::Release;
  KeyMod mods = KeyMod::None;
};

struct InputCharacterData {
  uint32_t codepoint = 0;
};

struct InputMouseButtonData {
  MouseButton button = MouseButton::Unknown;
  MouseAction action = MouseAction::Release;
  KeyMod mods = KeyMod::None;
};

struct InputMouseMoveData {
  double x = 0.0;
  double y = 0.0;
  double dx = 0.0;
  double dy = 0.0;
};

struct InputMouseScrollData {
  double x = 0.0;
  double y = 0.0;
};

struct InputFocusData {
  bool focused = false;
};

struct InputCursorEnterData {
  bool entered = false;
};

union InputEventPayload {
  InputKeyData key;
  InputCharacterData character;
  InputMouseButtonData mouseButton;
  InputMouseMoveData mouseMove;
  InputMouseScrollData mouseScroll;
  InputFocusData focus;
  InputCursorEnterData cursorEnter;

  constexpr InputEventPayload() : key() {}
};

struct InputEvent {
  InputEventType type = InputEventType::Key;
  uint32_t deviceId = 0;
  InputEventPayload payload{};
};

static_assert(std::is_trivially_copyable_v<RawKeyEvent>);
static_assert(std::is_trivially_copyable_v<RawCharEvent>);
static_assert(std::is_trivially_copyable_v<RawMouseButtonEvent>);
static_assert(std::is_trivially_copyable_v<RawMouseMoveEvent>);
static_assert(std::is_trivially_copyable_v<RawMouseScrollEvent>);
static_assert(std::is_trivially_copyable_v<RawFocusEvent>);
static_assert(std::is_trivially_copyable_v<RawCursorEnterEvent>);
static_assert(std::is_trivially_copyable_v<InputEventPayload>);
static_assert(std::is_trivially_copyable_v<InputEvent>);

static_assert(std::is_trivially_destructible_v<RawKeyEvent>);
static_assert(std::is_trivially_destructible_v<RawCharEvent>);
static_assert(std::is_trivially_destructible_v<RawMouseButtonEvent>);
static_assert(std::is_trivially_destructible_v<RawMouseMoveEvent>);
static_assert(std::is_trivially_destructible_v<RawMouseScrollEvent>);
static_assert(std::is_trivially_destructible_v<RawFocusEvent>);
static_assert(std::is_trivially_destructible_v<RawCursorEnterEvent>);
static_assert(std::is_trivially_destructible_v<InputEventPayload>);
static_assert(std::is_trivially_destructible_v<InputEvent>);

} // namespace nuri
