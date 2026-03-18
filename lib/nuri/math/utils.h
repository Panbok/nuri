// Common math sanitization helpers used to keep renderer inputs finite/sane.
#pragma once

#include <bit>
#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace nuri {

[[nodiscard]] glm::vec3 sanitizeFiniteVec3(const glm::vec3 &value,
                                           const glm::vec3 &fallback);

[[nodiscard]] glm::quat sanitizeRotation(const glm::quat &rotation);

[[nodiscard]] float sanitizeNonNegative(float value, float fallback = 0.0f);

[[nodiscard]] inline uint32_t floatBitsToUint(float value) {
  return std::bit_cast<uint32_t>(value);
}

[[nodiscard]] inline bool vec3Equal(const glm::vec3 &lhs,
                                    const glm::vec3 &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

[[nodiscard]] inline bool quatEqual(const glm::quat &lhs,
                                    const glm::quat &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
}

} // namespace nuri