#include "nuri/scene/camera_controller.h"

#include "nuri/core/log.h"

#include "nuri/pch.h"

namespace nuri {

namespace {

constexpr float kEpsilon = 1e-6f;
constexpr float kMinDamping = 0.001f;
constexpr float kMinMoveToDurationSeconds = 0.0001f;
constexpr float kTwoPi = glm::two_pi<float>();
constexpr std::array<Key, 8> kMovementKeys = {
    Key::W, Key::S, Key::A,         Key::D,
    Key::E, Key::Q, Key::LeftShift, Key::RightShift,
};

struct CameraIntent {
  glm::dvec2 lookDelta{0.0, 0.0};
  glm::vec3 moveAxis{0.0f, 0.0f, 0.0f};
  bool fast = false;
  bool hasManualLook = false;
  bool hasManualMovement = false;
};

struct WorldBasis {
  glm::vec3 up{0.0f, 1.0f, 0.0f};
  glm::vec3 forward{0.0f, 0.0f, -1.0f};
  glm::vec3 right{1.0f, 0.0f, 0.0f};
};

WorldBasis makeWorldBasis(const glm::vec3 &configuredUp) {
  WorldBasis basis{};
  if (glm::dot(configuredUp, configuredUp) > kEpsilon) {
    basis.up = glm::normalize(configuredUp);
  }

  const glm::vec3 fallbackForward = glm::vec3(0.0f, 0.0f, -1.0f);
  const glm::vec3 alternateForward = glm::vec3(1.0f, 0.0f, 0.0f);
  const glm::vec3 seed = std::abs(glm::dot(basis.up, fallbackForward)) < 0.999f
                             ? fallbackForward
                             : alternateForward;

  glm::vec3 flattenedForward = seed - basis.up * glm::dot(seed, basis.up);
  if (glm::dot(flattenedForward, flattenedForward) <= kEpsilon) {
    flattenedForward = fallbackForward;
  }
  basis.forward = glm::normalize(flattenedForward);
  basis.right = glm::normalize(glm::cross(basis.forward, basis.up));
  return basis;
}

glm::quat orientationFromAngles(float yawRadians, float pitchRadians,
                                const WorldBasis &basis) {
  const glm::quat yawRotation = glm::angleAxis(yawRadians, basis.up);
  glm::vec3 rightAxis = yawRotation * basis.right;
  if (glm::dot(rightAxis, rightAxis) <= kEpsilon) {
    rightAxis = basis.right;
  } else {
    rightAxis = glm::normalize(rightAxis);
  }
  const glm::quat pitchRotation = glm::angleAxis(pitchRadians, rightAxis);
  return glm::normalize(pitchRotation * yawRotation);
}

bool anglesFromOrientation(const glm::quat &orientation,
                           const WorldBasis &basis, float &inOutYawRadians,
                           float &outPitchRadians) {
  const glm::vec3 forward = orientation * glm::vec3(0.0f, 0.0f, -1.0f);
  const float upDot = glm::clamp(glm::dot(forward, basis.up), -1.0f, 1.0f);
  outPitchRadians = std::asin(upDot);

  glm::vec3 flatForward = forward - basis.up * upDot;
  if (glm::dot(flatForward, flatForward) <= kEpsilon) {
    return false;
  }
  flatForward = glm::normalize(flatForward);

  inOutYawRadians = std::atan2(glm::dot(flatForward, basis.right),
                               glm::dot(flatForward, basis.forward));
  return true;
}

float applyEasing(float t, MoveToEasing easing) {
  const float clamped = glm::clamp(t, 0.0f, 1.0f);
  switch (easing) {
  case MoveToEasing::Linear:
    return clamped;
  case MoveToEasing::Smoothstep:
    return clamped * clamped * (3.0f - 2.0f * clamped);
  }
  return clamped;
}

float wrapAngleRadians(float radians) {
  return std::remainder(radians, kTwoPi);
}

float lerpAngleShortest(float fromRadians, float toRadians, float t) {
  const float delta = wrapAngleRadians(toRadians - fromRadians);
  return wrapAngleRadians(fromRadians + delta * glm::clamp(t, 0.0f, 1.0f));
}

} // namespace

CameraController::CameraController(const CameraControllerConfig &config)
    : config_(config) {}

bool *CameraController::stateForKey(Key key) {
  switch (key) {
  case Key::W:
    return &movement_.forward;
  case Key::S:
    return &movement_.backward;
  case Key::A:
    return &movement_.left;
  case Key::D:
    return &movement_.right;
  case Key::E:
    return &movement_.up;
  case Key::Q:
    return &movement_.down;
  case Key::LeftShift:
    return &movement_.shiftLeft;
  case Key::RightShift:
    return &movement_.shiftRight;
  default:
    return nullptr;
  }
}

bool CameraController::handleKeyEvent(const InputKeyData &data) {
  const bool isDown =
      (data.action == KeyAction::Press || data.action == KeyAction::Repeat);
  const bool isRelease = (data.action == KeyAction::Release);
  if (!isDown && !isRelease) {
    return false;
  }

  bool *state = stateForKey(data.key);
  if (!state) {
    return false;
  }

  *state = isDown;
  return true;
}

bool CameraController::handleMouseButtonEvent(const InputMouseButtonData &data,
                                              Window &window) {
  if (data.button != MouseButton::Right) {
    return false;
  }

  if (data.action == MouseAction::Press) {
    if (!looking_) {
      looking_ = true;
      ignoreNextMouseDelta_ = true;
      window.setCursorMode(CursorMode::Disabled);
    }
    return true;
  }

  if (data.action == MouseAction::Release) {
    if (looking_) {
      looking_ = false;
      window.setCursorMode(CursorMode::Normal);
    }
    return true;
  }

  return false;
}

bool CameraController::onInput(const InputEvent &event, Window &window) {
  switch (event.type) {
  case InputEventType::Key:
    return handleKeyEvent(event.payload.key);
  case InputEventType::MouseButton:
    return handleMouseButtonEvent(event.payload.mouseButton, window);
  case InputEventType::Focus:
    if (!event.payload.focus.focused) {
      clearInputState();
      cancelMoveTo();
      forceCursorNormal(window);
      NURI_LOG_INFO(
          "CameraController::onInput: Focus lost, camera input reset");
      return true;
    }
    return false;
  case InputEventType::Character:
  case InputEventType::MouseMove:
  case InputEventType::MouseScroll:
  case InputEventType::CursorEnter:
    return false;
  }
  return false;
}

void CameraController::syncKeyStateFromPolling(const InputSystem &input) {
  for (Key key : kMovementKeys) {
    bool *state = stateForKey(key);
    if (state && *state && !input.isKeyDown(key)) {
      *state = false;
    }
  }
}

void CameraController::initializeAnglesFromCamera(const Camera &camera) {
  const WorldBasis basis = makeWorldBasis(config_.fps.worldUp);
  if (!anglesFromOrientation(camera.orientation(), basis, yawRadians_,
                             pitchRadians_)) {
    const glm::vec3 forward = camera.forward();
    pitchRadians_ =
        std::asin(glm::clamp(glm::dot(forward, basis.up), -1.0f, 1.0f));
  }
  anglesInitialized_ = true;
}

void CameraController::beginMoveToFromCamera(const Camera &camera) {
  if (!moveTo_.queued) {
    return;
  }
  moveTo_.queued = false;
  moveTo_.active = true;
  moveTo_.anglesInitialized = false;
  moveTo_.elapsedSeconds = 0.0f;
  moveTo_.startPosition = camera.position();
  moveTo_.startYawRadians = yawRadians_;
  moveTo_.startPitchRadians = pitchRadians_;
}

void CameraController::update(double deltaTime, const InputSystem &input,
                              Camera &camera) {
  if (deltaTime <= 0.0) {
    return;
  }

  if (!anglesInitialized_) {
    initializeAnglesFromCamera(camera);
  }

  syncKeyStateFromPolling(input);
  beginMoveToFromCamera(camera);

  CameraIntent intent{};
  if (looking_) {
    glm::dvec2 mouseDelta = input.mouseDelta();
    if (ignoreNextMouseDelta_) {
      mouseDelta = glm::dvec2(0.0, 0.0);
      ignoreNextMouseDelta_ = false;
    }
    intent.lookDelta = mouseDelta;
    intent.hasManualLook = glm::dot(mouseDelta, mouseDelta) > 0.0;
  }

  const float fb =
      (movement_.forward ? 1.0f : 0.0f) - (movement_.backward ? 1.0f : 0.0f);
  const float lr =
      (movement_.right ? 1.0f : 0.0f) - (movement_.left ? 1.0f : 0.0f);
  const float ud =
      (movement_.up ? 1.0f : 0.0f) - (movement_.down ? 1.0f : 0.0f);
  intent.moveAxis = glm::vec3(lr, ud, fb);
  intent.hasManualMovement = glm::dot(intent.moveAxis, intent.moveAxis) > 0.0f;
  intent.fast = movement_.shiftLeft || movement_.shiftRight;

  if (moveTo_.active && (intent.hasManualLook || intent.hasManualMovement)) {
    cancelMoveTo();
    NURI_LOG_DEBUG(
        "CameraController::update: MoveTo cancelled by manual input");
  }

  const float dtSeconds = static_cast<float>(deltaTime);
  const WorldBasis basis = makeWorldBasis(config_.fps.worldUp);
  glm::vec3 position = camera.position();
  glm::vec3 velocity = velocity_;
  float yawRadians = yawRadians_;
  float pitchRadians = pitchRadians_;

  yawRadians -=
      static_cast<float>(intent.lookDelta.x) * config_.fps.lookSensitivity;
  yawRadians = wrapAngleRadians(yawRadians);
  pitchRadians -=
      static_cast<float>(intent.lookDelta.y) * config_.fps.lookSensitivity;
  const float pitchLimit = std::max(config_.pitch.pitchLimitRadians, 0.0f);
  pitchRadians = std::clamp(pitchRadians, -pitchLimit, pitchLimit);

  glm::quat orientation =
      orientationFromAngles(yawRadians, pitchRadians, basis);
  const glm::vec3 forward = orientation * glm::vec3(0.0f, 0.0f, -1.0f);
  const glm::vec3 right = orientation * glm::vec3(1.0f, 0.0f, 0.0f);

  glm::vec3 upAxis = glm::cross(right, forward);
  if (glm::dot(upAxis, upAxis) <= kEpsilon) {
    upAxis = basis.up;
  } else {
    upAxis = glm::normalize(upAxis);
  }

  glm::vec3 accel = (forward * intent.moveAxis.z) +
                    (right * intent.moveAxis.x) + (upAxis * intent.moveAxis.y);
  if (intent.fast) {
    accel *= config_.fps.fastSpeedMultiplier;
  }

  if (glm::dot(accel, accel) <= kEpsilon) {
    const float damping = std::max(config_.fps.damping, kMinDamping);
    velocity -= velocity * std::min((1.0f / damping) * dtSeconds, 1.0f);
  } else {
    velocity += accel * config_.fps.acceleration * dtSeconds;
    const float maxSpeed =
        intent.fast ? config_.fps.maxSpeed * config_.fps.fastSpeedMultiplier
                    : config_.fps.maxSpeed;
    const float speed = glm::length(velocity);
    if (speed > maxSpeed && speed > kEpsilon) {
      velocity = glm::normalize(velocity) * maxSpeed;
    }
  }
  position += velocity * dtSeconds;

  if (moveTo_.active) {
    if (!moveTo_.anglesInitialized) {
      float targetYaw = moveTo_.startYawRadians;
      float targetPitch = moveTo_.startPitchRadians;
      if (!anglesFromOrientation(moveTo_.request.targetOrientation, basis,
                                 targetYaw, targetPitch)) {
        const glm::vec3 targetForward = glm::normalize(
            moveTo_.request.targetOrientation * glm::vec3(0.0f, 0.0f, -1.0f));
        targetPitch = std::asin(
            glm::clamp(glm::dot(targetForward, basis.up), -1.0f, 1.0f));
      }
      moveTo_.targetYawRadians = wrapAngleRadians(targetYaw);
      moveTo_.targetPitchRadians = targetPitch;
      moveTo_.anglesInitialized = true;
    }

    moveTo_.elapsedSeconds += dtSeconds;
    const float duration =
        std::max(moveTo_.request.durationSeconds, kMinMoveToDurationSeconds);
    const float linearT =
        glm::clamp(moveTo_.elapsedSeconds / duration, 0.0f, 1.0f);
    const float t = applyEasing(linearT, moveTo_.request.easing);

    position =
        glm::mix(moveTo_.startPosition, moveTo_.request.targetPosition, t);
    yawRadians =
        lerpAngleShortest(moveTo_.startYawRadians, moveTo_.targetYawRadians, t);
    pitchRadians =
        glm::mix(moveTo_.startPitchRadians, moveTo_.targetPitchRadians, t);
    orientation = orientationFromAngles(yawRadians, pitchRadians, basis);
    velocity = glm::vec3(0.0f);

    if (linearT >= 1.0f) {
      moveTo_.active = false;
      position = moveTo_.request.targetPosition;
      yawRadians = moveTo_.targetYawRadians;
      pitchRadians = moveTo_.targetPitchRadians;
      orientation = orientationFromAngles(yawRadians, pitchRadians, basis);
    }
  }

  orientation = glm::normalize(orientation);
  if (!moveTo_.active) {
    pitchRadians = std::clamp(pitchRadians, -pitchLimit, pitchLimit);
    orientation = orientationFromAngles(yawRadians, pitchRadians, basis);
  }

  camera.setPosition(position);
  camera.setOrientation(orientation);
  velocity_ = velocity;
  yawRadians_ = yawRadians;
  pitchRadians_ = pitchRadians;
}

void CameraController::setPreset(CameraPreset preset) {
  if (preset_ == preset) {
    return;
  }

  preset_ = preset;
  if (preset_ == CameraPreset::FpsDirect) {
    cancelMoveTo();
  }
}

Result<bool, std::string>
CameraController::startMoveTo(const MoveToRequest &request) {
  if (preset_ != CameraPreset::FpsMoveTo) {
    return Result<bool, std::string>::makeError(
        "MoveTo is available only in FPS + MoveTo preset");
  }

  if (!std::isfinite(request.durationSeconds) ||
      request.durationSeconds <= 0.0f) {
    return Result<bool, std::string>::makeError(
        "MoveTo durationSeconds must be > 0");
  }

  if (!std::isfinite(request.targetPosition.x) ||
      !std::isfinite(request.targetPosition.y) ||
      !std::isfinite(request.targetPosition.z)) {
    return Result<bool, std::string>::makeError(
        "MoveTo targetPosition must be finite");
  }

  moveTo_.request = request;
  moveTo_.queued = true;
  moveTo_.active = false;
  moveTo_.elapsedSeconds = 0.0f;
  return Result<bool, std::string>::makeResult(true);
}

void CameraController::cancelMoveTo() noexcept { moveTo_ = {}; }

void CameraController::forceCursorNormal(Window &window) {
  looking_ = false;
  ignoreNextMouseDelta_ = false;
  window.setCursorMode(CursorMode::Normal);
}

void CameraController::clearInputState() {
  movement_ = {};
  velocity_ = glm::vec3(0.0f, 0.0f, 0.0f);
}

void CameraController::onActivate(Window &window) {
  clearInputState();
  cancelMoveTo();
  forceCursorNormal(window);
  NURI_LOG_DEBUG("CameraController::onActivate: Controller activated");
}

void CameraController::onDeactivate(Window &window) {
  clearInputState();
  cancelMoveTo();
  forceCursorNormal(window);
  NURI_LOG_DEBUG("CameraController::onDeactivate: Controller deactivated");
}

void CameraController::reset() {
  clearInputState();
  cancelMoveTo();
  yawRadians_ = 0.0f;
  pitchRadians_ = 0.0f;
  anglesInitialized_ = false;
  NURI_LOG_DEBUG("CameraController::reset: Controller state reset");
}

CameraController makeFpsDirectController(const CameraControllerConfig &config) {
  CameraController controller(config);
  controller.setPreset(CameraPreset::FpsDirect);
  return controller;
}

CameraController makeFpsMoveToController(const CameraControllerConfig &config) {
  CameraController controller(config);
  controller.setPreset(CameraPreset::FpsMoveTo);
  return controller;
}

} // namespace nuri
