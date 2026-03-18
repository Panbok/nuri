// Common math sanitization helpers used to keep renderer inputs finite/sane.
#pragma once

#include <glm/glm.hpp>

namespace nuri {

[[nodiscard]] glm::vec3 sanitizeFiniteVec3(const glm::vec3 &value,
                                           const glm::vec3 &fallback);

[[nodiscard]] glm::quat sanitizeRotation(const glm::quat &rotation);

[[nodiscard]] float sanitizeNonNegative(float value, float fallback = 0.0f);

} // namespace nuri