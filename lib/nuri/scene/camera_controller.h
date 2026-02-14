#pragma once

#include "nuri/core/input_events.h"
#include "nuri/core/input_system.h"
#include "nuri/core/result.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"
#include "nuri/scene/camera.h"

#include <cstdint>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace nuri {

enum class CameraPreset : uint8_t {
  FpsDirect = 0,
  FpsMoveTo = 1,
  Count,
};

enum class MoveToEasing : uint8_t {
  Linear = 0,
  Smoothstep = 1,
  Count,
};

struct FpsDirectConfig {
  float lookSensitivity = 0.0025f;
  float acceleration = 150.0f;
  float damping = 0.2f;
  float maxSpeed = 10.0f;
  float fastSpeedMultiplier = 10.0f;
  glm::vec3 worldUp{0.0f, 1.0f, 0.0f};
};

struct PitchLimitConstraintConfig {
  float pitchLimitRadians = glm::radians(89.0f);
};

struct MoveToRequest {
  glm::vec3 targetPosition{0.0f, 0.0f, 0.0f};
  glm::quat targetOrientation{1.0f, 0.0f, 0.0f, 0.0f};
  float durationSeconds = 1.0f;
  MoveToEasing easing = MoveToEasing::Smoothstep;
};

struct CameraControllerConfig {
  FpsDirectConfig fps{};
  PitchLimitConstraintConfig pitch{};
};

class NURI_API CameraController final {
public:
  explicit CameraController(const CameraControllerConfig &config = {});

  bool onInput(const InputEvent &event, Window &window);
  void onActivate(Window &window);
  void onDeactivate(Window &window);
  void update(double deltaTime, const InputSystem &input, Camera &camera);
  void reset();

  void setPreset(CameraPreset preset);
  [[nodiscard]] CameraPreset preset() const noexcept { return preset_; }

  Result<bool, std::string> startMoveTo(const MoveToRequest &request);
  void cancelMoveTo() noexcept;
  [[nodiscard]] bool isMoveToActive() const noexcept {
    return moveTo_.active || moveTo_.queued;
  }

private:
  struct MovementState {
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;
    bool shiftLeft = false;
    bool shiftRight = false;
  };

  struct MoveToState {
    bool queued = false;
    bool active = false;
    bool anglesInitialized = false;
    MoveToRequest request{};
    glm::vec3 startPosition{0.0f, 0.0f, 0.0f};
    float elapsedSeconds = 0.0f;
    float startYawRadians = 0.0f;
    float startPitchRadians = 0.0f;
    float targetYawRadians = 0.0f;
    float targetPitchRadians = 0.0f;
  };

  [[nodiscard]] bool *stateForKey(Key key);
  bool handleKeyEvent(const InputKeyData &data);
  bool handleMouseButtonEvent(const InputMouseButtonData &data, Window &window);
  void syncKeyStateFromPolling(const InputSystem &input);
  void forceCursorNormal(Window &window);
  void clearInputState();
  void initializeAnglesFromCamera(const Camera &camera);
  void beginMoveToFromCamera(const Camera &camera);

  CameraControllerConfig config_{};
  CameraPreset preset_ = CameraPreset::FpsDirect;
  MovementState movement_{};
  MoveToState moveTo_{};

  bool looking_ = false;
  bool ignoreNextMouseDelta_ = false;
  bool anglesInitialized_ = false;

  float yawRadians_ = 0.0f;
  float pitchRadians_ = 0.0f;
  glm::vec3 velocity_{0.0f, 0.0f, 0.0f};
};

NURI_API CameraController
makeFpsDirectController(const CameraControllerConfig &config = {});

NURI_API CameraController
makeFpsMoveToController(const CameraControllerConfig &config = {});

} // namespace nuri
