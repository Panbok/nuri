#pragma once

#include "nuri/scene/light.h"

#include <cmath>

#include <glm/gtc/quaternion.hpp>

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

[[nodiscard]] inline LightDesc
transformLightDesc(const LightDesc &source, const glm::mat4 &modelMatrix) {
  LightDesc transformed = source;
  transformed.position =
      glm::vec3(modelMatrix * glm::vec4(source.position, 1.0f));
  transformed.rotation =
      rotationFromMatrixOrIdentity(modelMatrix) * source.rotation;
  return transformed;
}

} // namespace nuri
