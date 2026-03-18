#include "nuri/pch.h"

#include "nuri/math/utils.h"

#include <algorithm>
#include <cmath>

namespace nuri {

namespace {

constexpr glm::quat kIdentityRotation(1.0f, 0.0f, 0.0f, 0.0f);

} // namespace

glm::vec3 sanitizeFiniteVec3(const glm::vec3 &value,
                             const glm::vec3 &fallback) {
  if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
      !std::isfinite(value.z)) {
    return fallback;
  }
  return value;
}

glm::quat sanitizeRotation(const glm::quat &rotation) {
  const float length = glm::length(rotation);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return kIdentityRotation;
  }
  return glm::normalize(rotation);
}

float sanitizeNonNegative(float value, float fallback) {
  if (!std::isfinite(value)) {
    return fallback;
  }
  return std::max(value, 0.0f);
}

} // namespace nuri