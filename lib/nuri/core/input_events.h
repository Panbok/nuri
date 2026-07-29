#pragma once
#include "nuri/core/input_codes.h"
#include <cstdint>
#include <type_traits>
namespace nuri {

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

static_assert(std::is_trivially_copyable_v<InputEventPayload>);
static_assert(std::is_trivially_copyable_v<InputEvent>);

static_assert(std::is_trivially_destructible_v<InputEventPayload>);
static_assert(std::is_trivially_destructible_v<InputEvent>);

} // namespace nuri
