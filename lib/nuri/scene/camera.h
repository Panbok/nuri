#pragma once

#include "nuri/defines.h"

#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace nuri {

enum class ProjectionType : uint8_t {
  Perspective = 0,
  Orthographic = 1,
};

struct PerspectiveParams {
  float fovYRadians = glm::radians(60.0f);
  float nearPlane = 0.1f;
  float farPlane = 1000.0f;
};

struct OrthographicParams {
  float height = 10.0f;
  float nearPlane = 0.1f;
  float farPlane = 1000.0f;
};

class NURI_API Camera {
public:
  Camera() = default;
  explicit Camera(const glm::vec3 &position,
                  const glm::quat &orientation = glm::quat(1.0f, 0.0f, 0.0f,
                                                           0.0f));

  [[nodiscard]] const glm::vec3 &position() const noexcept { return position_; }
  [[nodiscard]] const glm::quat &orientation() const noexcept {
    return orientation_;
  }

  void setPosition(const glm::vec3 &position) noexcept { position_ = position; }
  void setOrientation(const glm::quat &orientation);

  [[nodiscard]] glm::vec3 forward() const;
  [[nodiscard]] glm::vec3 right() const;
  [[nodiscard]] glm::vec3 up() const;

  [[nodiscard]] ProjectionType projectionType() const noexcept {
    return projectionType_;
  }
  void setProjectionType(ProjectionType type) noexcept {
    projectionType_ = type;
  }

  [[nodiscard]] const PerspectiveParams &perspective() const noexcept {
    return perspective_;
  }
  void setPerspective(const PerspectiveParams &params);

  [[nodiscard]] const OrthographicParams &orthographic() const noexcept {
    return orthographic_;
  }
  void setOrthographic(const OrthographicParams &params);

  [[nodiscard]] glm::mat4 viewMatrix() const;
  [[nodiscard]] glm::mat4 projectionMatrix(float aspect) const;
  [[nodiscard]] glm::mat4 viewProjectionMatrix(float aspect) const;

  void setLookAt(const glm::vec3 &eye, const glm::vec3 &target,
                 const glm::vec3 &worldUp);

private:
  static PerspectiveParams sanitizePerspective(const PerspectiveParams &params);
  static OrthographicParams
  sanitizeOrthographic(const OrthographicParams &params);

  glm::vec3 position_{0.0f, 0.0f, 0.0f};
  glm::quat orientation_{1.0f, 0.0f, 0.0f, 0.0f};
  ProjectionType projectionType_ = ProjectionType::Perspective;
  PerspectiveParams perspective_{};
  OrthographicParams orthographic_{};
};

} // namespace nuri
