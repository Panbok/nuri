#include "nuri/pch.h"

#include "nuri/scene/camera.h"

#include "nuri/core/log.h"

namespace nuri {

namespace {

const PerspectiveParams kDefaultPerspective{};
const OrthographicParams kDefaultOrthographic{};
constexpr float kEpsilon = 1e-6f;

float sanitizeAspect(float aspect) {
  if (!std::isfinite(aspect) || aspect <= kEpsilon) {
    NURI_LOG_WARNING(
        "Camera::projectionMatrix: Invalid aspect ratio %.6f, using 1.0",
        aspect);
    return 1.0f;
  }
  return aspect;
}

} // namespace

Camera::Camera(const glm::vec3 &position, const glm::quat &orientation)
    : position_(position) {
  setOrientation(orientation);
  setPerspective(kDefaultPerspective);
  setOrthographic(kDefaultOrthographic);
}

void Camera::setOrientation(const glm::quat &orientation) {
  const float norm = glm::length(orientation);
  if (!std::isfinite(norm) || norm <= kEpsilon) {
    NURI_LOG_WARNING(
        "Camera::setOrientation: Invalid quaternion, using identity");
    orientation_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    return;
  }
  orientation_ = glm::normalize(orientation);
}

glm::vec3 Camera::forward() const {
  return glm::normalize(orientation_ * glm::vec3(0.0f, 0.0f, -1.0f));
}

glm::vec3 Camera::right() const {
  return glm::normalize(orientation_ * glm::vec3(1.0f, 0.0f, 0.0f));
}

glm::vec3 Camera::up() const {
  return glm::normalize(orientation_ * glm::vec3(0.0f, 1.0f, 0.0f));
}

void Camera::setPerspective(const PerspectiveParams &params) {
  perspective_ = sanitizePerspective(params);
}

void Camera::setOrthographic(const OrthographicParams &params) {
  orthographic_ = sanitizeOrthographic(params);
}

glm::mat4 Camera::viewMatrix() const {
  const glm::mat4 rotation = glm::mat4_cast(glm::conjugate(orientation_));
  const glm::mat4 translation = glm::translate(glm::mat4(1.0f), -position_);
  return rotation * translation;
}

glm::mat4 Camera::projectionMatrix(float aspect) const {
  const float safeAspect = sanitizeAspect(aspect);
  if (projectionType_ == ProjectionType::Orthographic) {
    const float halfHeight = 0.5f * orthographic_.height;
    const float halfWidth = halfHeight * safeAspect;
    return glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight,
                      orthographic_.nearPlane, orthographic_.farPlane);
  }

  return glm::perspective(perspective_.fovYRadians, safeAspect,
                          perspective_.nearPlane, perspective_.farPlane);
}

glm::mat4 Camera::viewProjectionMatrix(float aspect) const {
  return projectionMatrix(aspect) * viewMatrix();
}

void Camera::setLookAt(const glm::vec3 &eye, const glm::vec3 &target,
                       const glm::vec3 &worldUp) {
  const glm::vec3 direction = target - eye;
  if (glm::dot(direction, direction) <= kEpsilon) {
    NURI_LOG_WARNING("Camera::setLookAt: Eye and target are too close");
    position_ = eye;
    return;
  }

  glm::vec3 up = worldUp;
  const float upLengthSquared = glm::dot(up, up);
  if (!std::isfinite(upLengthSquared) || upLengthSquared <= kEpsilon) {
    NURI_LOG_WARNING("Camera::setLookAt: worldUp is invalid or zero, using +Y");
    up = glm::vec3(0.0f, 1.0f, 0.0f);
  }

  const glm::vec3 normalizedDirection = glm::normalize(direction);
  glm::vec3 normalizedUp = glm::normalize(up);
  if (std::abs(glm::dot(normalizedDirection, normalizedUp)) >=
      1.0f - kEpsilon) {
    NURI_LOG_WARNING("Camera::setLookAt: direction and worldUp are collinear, "
                     "using fallback up axis");
    normalizedUp = glm::vec3(0.0f, 0.0f, 1.0f);
    if (std::abs(glm::dot(normalizedDirection, normalizedUp)) >=
        1.0f - kEpsilon) {
      normalizedUp = glm::vec3(1.0f, 0.0f, 0.0f);
    }
  }

  position_ = eye;
  const glm::mat4 view = glm::lookAt(eye, target, normalizedUp);
  const glm::mat4 world = glm::inverse(view);
  setOrientation(glm::quat_cast(world));
}

PerspectiveParams Camera::sanitizePerspective(const PerspectiveParams &params) {
  PerspectiveParams out = params;

  if (!std::isfinite(out.fovYRadians) || out.fovYRadians <= kEpsilon ||
      out.fovYRadians >= glm::pi<float>() - kEpsilon) {
    NURI_LOG_WARNING("Camera: Invalid perspective FOV %.6f, using default %.6f",
                     out.fovYRadians, kDefaultPerspective.fovYRadians);
    out.fovYRadians = kDefaultPerspective.fovYRadians;
  }

  if (!std::isfinite(out.nearPlane) || out.nearPlane <= kEpsilon) {
    NURI_LOG_WARNING(
        "Camera: Invalid perspective near plane %.6f, using default "
        "%.6f",
        out.nearPlane, kDefaultPerspective.nearPlane);
    out.nearPlane = kDefaultPerspective.nearPlane;
  }

  if (!std::isfinite(out.farPlane) ||
      out.farPlane <= out.nearPlane + kEpsilon) {
    NURI_LOG_WARNING(
        "Camera: Invalid perspective far plane %.6f, using default "
        "%.6f",
        out.farPlane, kDefaultPerspective.farPlane);
    out.farPlane = std::max(kDefaultPerspective.farPlane, out.nearPlane + 1.0f);
  }

  return out;
}

OrthographicParams
Camera::sanitizeOrthographic(const OrthographicParams &params) {
  OrthographicParams out = params;

  if (!std::isfinite(out.height) || out.height <= kEpsilon) {
    NURI_LOG_WARNING("Camera: Invalid ortho height %.6f, using default %.6f",
                     out.height, kDefaultOrthographic.height);
    out.height = kDefaultOrthographic.height;
  }

  if (!std::isfinite(out.nearPlane) || std::abs(out.nearPlane) <= kEpsilon) {
    NURI_LOG_WARNING(
        "Camera: Invalid ortho near plane %.6f, using default %.6f",
        out.nearPlane, kDefaultOrthographic.nearPlane);
    out.nearPlane = kDefaultOrthographic.nearPlane;
  }

  if (!std::isfinite(out.farPlane) ||
      out.farPlane <= out.nearPlane + kEpsilon) {
    NURI_LOG_WARNING("Camera: Invalid ortho far plane %.6f, using default %.6f",
                     out.farPlane, kDefaultOrthographic.farPlane);
    out.farPlane =
        std::max(kDefaultOrthographic.farPlane, out.nearPlane + 1.0f);
  }

  return out;
}

} // namespace nuri
