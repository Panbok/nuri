#pragma once

#include "nuri/math/utils.h"
#include "nuri/scene/light.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/constants.hpp>

namespace nuri {

inline constexpr float kLightBasisMinLength = 1.0e-6f;

[[nodiscard]] inline glm::quat
rotationFromMatrixOrIdentity(const glm::mat4 &matrix) {
  glm::vec3 basisX(matrix[0].x, matrix[0].y, matrix[0].z);
  glm::vec3 basisY(matrix[1].x, matrix[1].y, matrix[1].z);
  glm::vec3 basisZ(matrix[2].x, matrix[2].y, matrix[2].z);

  const float xLength = glm::length(basisX);
  const float yLength = glm::length(basisY);
  const float zLength = glm::length(basisZ);
  if (!std::isfinite(xLength) || !std::isfinite(yLength) ||
      !std::isfinite(zLength) || xLength <= kLightBasisMinLength ||
      yLength <= kLightBasisMinLength || zLength <= kLightBasisMinLength) {
    return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }

  basisX /= xLength;
  basisY /= yLength;
  basisZ /= zLength;

  glm::mat3 basis(1.0f);
  basis[0] = basisX;
  basis[1] = basisY;
  basis[2] = basisZ;
  if (glm::determinant(basis) < 0.0f) {
    basis[0] = -basis[0];
  }
  return glm::normalize(glm::quat_cast(basis));
}

[[nodiscard]] inline LightDesc sanitizeLightDesc(const LightDesc &desc) {
  LightDesc sanitized = desc;
  sanitized.position = sanitizeFiniteVec3(desc.position, glm::vec3(0.0f));
  sanitized.rotation = sanitizeRotation(desc.rotation);
  sanitized.color = glm::max(sanitizeFiniteVec3(desc.color, glm::vec3(1.0f)),
                             glm::vec3(0.0f));
  sanitized.intensity = sanitizeNonNegative(desc.intensity, 1.0f);
  sanitized.enabled = desc.enabled;

  switch (sanitized.type) {
  case LightType::Directional:
    sanitized.range = 0.0f;
    sanitized.innerConeAngleRadians = 0.0f;
    sanitized.outerConeAngleRadians = 0.0f;
    break;
  case LightType::Point:
    sanitized.range = sanitizeNonNegative(desc.range, 0.0f);
    sanitized.innerConeAngleRadians = 0.0f;
    sanitized.outerConeAngleRadians = 0.0f;
    break;
  case LightType::Spot:
    sanitized.range = sanitizeNonNegative(desc.range, 0.0f);
    sanitized.outerConeAngleRadians =
        std::clamp(sanitizeNonNegative(desc.outerConeAngleRadians,
                                       glm::quarter_pi<float>()),
                   0.0f, glm::half_pi<float>() - 1.0e-4f);
    sanitized.innerConeAngleRadians =
        std::clamp(sanitizeNonNegative(desc.innerConeAngleRadians, 0.0f), 0.0f,
                   sanitized.outerConeAngleRadians);
    break;
  }

  return sanitized;
}

[[nodiscard]] inline LightDesc lightLocalFromWorld(const LightDesc &worldDesc,
                                                      const glm::mat4 &nodeWorld) {
  LightDesc local = worldDesc;
  const glm::mat4 inverseNodeWorld = safeInverseOrIdentity(nodeWorld);
  local.position =
      glm::vec3(inverseNodeWorld * glm::vec4(worldDesc.position, 1.0f));
  const glm::quat nodeRotation = rotationFromMatrixOrIdentity(nodeWorld);
  local.rotation =
      sanitizeRotation(glm::inverse(nodeRotation) * worldDesc.rotation);
  return sanitizeLightDesc(local);
}

[[nodiscard]] inline LightDesc
transformLightDesc(const LightDesc &source, const glm::mat4 &modelMatrix) {
  LightDesc transformed = source;
  transformed.position =
      glm::vec3(modelMatrix * glm::vec4(source.position, 1.0f));
  transformed.rotation =
      rotationFromMatrixOrIdentity(modelMatrix) * source.rotation;
  return transformed;
}

[[nodiscard]] inline glm::vec3
lightDirectionFromRotation(const glm::quat &rotation) {
  const glm::vec3 direction =
      sanitizeRotation(rotation) * glm::vec3(0.0f, 0.0f, -1.0f);
  const float length = glm::length(direction);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return glm::vec3(0.0f, 0.0f, -1.0f);
  }
  return direction / length;
}

} // namespace nuri
