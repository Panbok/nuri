#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/ext/matrix_relational.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
namespace nuri {

[[nodiscard]] inline glm::vec3 sanitizeFiniteVec3(const glm::vec3 &value,
                                                  const glm::vec3 &fallback) {
  if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
      !std::isfinite(value.z)) {
    return fallback;
  }
  return value;
}

[[nodiscard]] inline glm::quat sanitizeRotation(const glm::quat &rotation) {
  const float length = glm::length(rotation);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }
  return glm::normalize(rotation);
}

[[nodiscard]] inline float sanitizeNonNegative(float value,
                                               float fallback = 0.0f) {
  if (!std::isfinite(value)) {
    return fallback;
  }
  return std::max(value, 0.0f);
}

[[nodiscard]] inline bool vec3ExactEqual(const glm::vec3 &lhs,
                                         const glm::vec3 &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

[[nodiscard]] inline bool quatExactEqual(const glm::quat &lhs,
                                         const glm::quat &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
}

[[nodiscard]] inline bool mat4ExactEqual(const glm::mat4 &lhs,
                                         const glm::mat4 &rhs) {
  return glm::all(glm::equal(lhs, rhs));
}

[[nodiscard]] inline glm::mat4 safeInverseOrIdentity(const glm::mat4 &matrix) {
  const float determinant = glm::determinant(matrix);
  if (!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-8f) {
    return glm::mat4(1.0f);
  }
  return glm::inverse(matrix);
}

[[nodiscard]] inline uint32_t floatBitsToUint(float value) {
  return std::bit_cast<uint32_t>(value);
}

[[nodiscard]] inline size_t alignUp(size_t value, size_t alignment) {
  if (alignment == 0u) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0u ? value : value + alignment - remainder;
}

[[nodiscard]] inline bool alignUpU64(uint64_t value, uint64_t alignment,
                                     uint64_t &out) {
  const uint64_t remainder = alignment > 1u ? value % alignment : 0u;
  const uint64_t increment = remainder ? alignment - remainder : 0u;
  if (value > std::numeric_limits<uint64_t>::max() - increment) {
    return false;
  }
  out = value + increment;
  return true;
}

} // namespace nuri
