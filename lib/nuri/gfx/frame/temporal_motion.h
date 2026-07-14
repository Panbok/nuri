#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <limits>

namespace nuri {

// Canonical temporal-motion convention used by every color/AO consumer:
// historyUv = currentUv + velocityUv. Matrices must be unjittered.
struct TemporalMotionEndpoint {
  glm::vec2 currentUv{0.0f};
  glm::vec2 previousUv{0.0f};
  glm::vec2 velocityUv{0.0f};
  bool valid = false;
};

[[nodiscard]] constexpr glm::vec2
temporalScreenUvFromClipNdc(glm::vec2 ndc) noexcept {
  return glm::vec2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
}

[[nodiscard]] inline TemporalMotionEndpoint projectTemporalMotionEndpoint(
    const glm::vec3 &currentWorldPosition,
    const glm::vec3 &previousWorldPosition,
    const glm::mat4 &currentUnjitteredViewProjection,
    const glm::mat4 &previousUnjitteredViewProjection) noexcept {
  const glm::vec4 currentClip =
      currentUnjitteredViewProjection * glm::vec4(currentWorldPosition, 1.0f);
  const glm::vec4 previousClip =
      previousUnjitteredViewProjection * glm::vec4(previousWorldPosition, 1.0f);
  constexpr float kMinimumClipW = 1.0e-8f;
  if (!std::isfinite(currentClip.w) || !std::isfinite(previousClip.w) ||
      std::abs(currentClip.w) <= kMinimumClipW ||
      std::abs(previousClip.w) <= kMinimumClipW) {
    return {};
  }

  const glm::vec2 currentUv =
      temporalScreenUvFromClipNdc(glm::vec2(currentClip) / currentClip.w);
  const glm::vec2 previousUv =
      temporalScreenUvFromClipNdc(glm::vec2(previousClip) / previousClip.w);
  const glm::vec2 velocityUv = previousUv - currentUv;
  const bool finite =
      std::isfinite(currentUv.x) && std::isfinite(currentUv.y) &&
      std::isfinite(previousUv.x) && std::isfinite(previousUv.y) &&
      std::isfinite(velocityUv.x) && std::isfinite(velocityUv.y);
  return TemporalMotionEndpoint{
      .currentUv = currentUv,
      .previousUv = previousUv,
      .velocityUv = velocityUv,
      .valid = finite,
  };
}

[[nodiscard]] inline float
temporalMotionEndpointErrorPixels(glm::vec2 measuredVelocityUv,
                                  const TemporalMotionEndpoint &expected,
                                  glm::uvec2 renderExtent) noexcept {
  if (!expected.valid || renderExtent.x == 0u || renderExtent.y == 0u ||
      !std::isfinite(measuredVelocityUv.x) ||
      !std::isfinite(measuredVelocityUv.y)) {
    return std::numeric_limits<float>::infinity();
  }
  const glm::vec2 errorPixels =
      (measuredVelocityUv - expected.velocityUv) * glm::vec2(renderExtent);
  return glm::length(errorPixels);
}

} // namespace nuri
