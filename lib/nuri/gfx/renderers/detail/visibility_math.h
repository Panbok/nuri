#pragma once

#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/math/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <glm/glm.hpp>

namespace nuri::visibility_detail {

enum class DepthClipConvention : uint8_t {
  MinusOneToOne = 0,
  ZeroToOne = 1,
};

enum class VisibilityClassification : uint8_t {
  Outside = 0,
  Intersects = 1,
  Inside = 2,
};

struct FrustumPlanes {
  std::array<glm::vec4, 6> planes{};
};

[[nodiscard]] inline glm::vec4 matrixRow(const glm::mat4 &m,
                                         uint32_t row) noexcept {
  return glm::vec4(m[0][row], m[1][row], m[2][row], m[3][row]);
}

[[nodiscard]] inline glm::vec4 normalizePlane(glm::vec4 plane) noexcept {
  const float len = glm::length(glm::vec3(plane));
  if (!std::isfinite(len) || len <= 1.0e-8f) {
    return glm::vec4(0.0f, 0.0f, 0.0f, -std::numeric_limits<float>::max());
  }
  return plane / len;
}

[[nodiscard]] inline FrustumPlanes
buildFrustumPlanes(const glm::mat4 &viewProj,
                   DepthClipConvention depthConvention =
                       DepthClipConvention::MinusOneToOne) noexcept {
  const glm::vec4 row0 = matrixRow(viewProj, 0u);
  const glm::vec4 row1 = matrixRow(viewProj, 1u);
  const glm::vec4 row2 = matrixRow(viewProj, 2u);
  const glm::vec4 row3 = matrixRow(viewProj, 3u);

  FrustumPlanes out{};
  out.planes[0] = normalizePlane(row3 + row0);
  out.planes[1] = normalizePlane(row3 - row0);
  out.planes[2] = normalizePlane(row3 + row1);
  out.planes[3] = normalizePlane(row3 - row1);
  out.planes[4] = normalizePlane(depthConvention ==
                                         DepthClipConvention::ZeroToOne
                                     ? row2
                                     : row3 + row2);
  out.planes[5] = normalizePlane(row3 - row2);
  return out;
}

[[nodiscard]] inline FrustumPlanes
buildCameraFrustumPlanes(const CameraFrameState &camera) noexcept {
  return buildFrustumPlanes(cameraCurrentUnjitteredViewProjection(camera),
                            DepthClipConvention::MinusOneToOne);
}

[[nodiscard]] inline float maxAxisScale(const glm::mat4 &transform) noexcept {
  const float sx = glm::length(glm::vec3(transform[0]));
  const float sy = glm::length(glm::vec3(transform[1]));
  const float sz = glm::length(glm::vec3(transform[2]));
  return std::max(std::max(sx, sy), sz);
}

[[nodiscard]] inline glm::vec4
transformBoundingSphere(const BoundingBox &bounds,
                        const glm::mat4 &worldFromLocal) noexcept {
  const glm::vec3 localCenter = bounds.getCenter();
  const float localRadius = 0.5f * glm::length(bounds.getSize());
  const glm::vec3 worldCenter =
      glm::vec3(worldFromLocal * glm::vec4(localCenter, 1.0f));
  const float worldRadius = localRadius * maxAxisScale(worldFromLocal);
  return glm::vec4(worldCenter, std::max(worldRadius, 0.0f));
}

[[nodiscard]] inline VisibilityClassification
classifySphere(const FrustumPlanes &frustum, glm::vec3 center,
               float radius) noexcept {
  VisibilityClassification result = VisibilityClassification::Inside;
  const float safeRadius =
      std::isfinite(radius) ? std::max(radius, 0.0f) : 0.0f;
  for (const glm::vec4 &plane : frustum.planes) {
    const float distance = glm::dot(glm::vec3(plane), center) + plane.w;
    if (distance < -safeRadius) {
      return VisibilityClassification::Outside;
    }
    if (distance < safeRadius) {
      result = VisibilityClassification::Intersects;
    }
  }
  return result;
}

[[nodiscard]] inline VisibilityClassification
classifyAabb(const FrustumPlanes &frustum,
             const BoundingBox &worldBounds) noexcept {
  VisibilityClassification result = VisibilityClassification::Inside;
  const glm::vec3 min = worldBounds.min_;
  const glm::vec3 max = worldBounds.max_;
  for (const glm::vec4 &plane : frustum.planes) {
    const glm::vec3 normal = glm::vec3(plane);
    const glm::vec3 positive(
        normal.x >= 0.0f ? max.x : min.x,
        normal.y >= 0.0f ? max.y : min.y,
        normal.z >= 0.0f ? max.z : min.z);
    if (glm::dot(normal, positive) + plane.w < 0.0f) {
      return VisibilityClassification::Outside;
    }

    const glm::vec3 negative(
        normal.x >= 0.0f ? min.x : max.x,
        normal.y >= 0.0f ? min.y : max.y,
        normal.z >= 0.0f ? min.z : max.z);
    if (glm::dot(normal, negative) + plane.w < 0.0f) {
      result = VisibilityClassification::Intersects;
    }
  }
  return result;
}

[[nodiscard]] inline VisibilityClassification
classifyTransformedBounds(const FrustumPlanes &frustum,
                          const BoundingBox &localBounds,
                          const glm::mat4 &worldFromLocal) noexcept {
  const glm::vec4 sphere = transformBoundingSphere(localBounds, worldFromLocal);
  const VisibilityClassification sphereClass =
      classifySphere(frustum, glm::vec3(sphere), sphere.w);
  if (sphereClass == VisibilityClassification::Outside ||
      sphereClass == VisibilityClassification::Inside) {
    return sphereClass;
  }
  return classifyAabb(frustum, localBounds.getTransformed(worldFromLocal));
}

[[nodiscard]] inline bool
isVisible(VisibilityClassification classification) noexcept {
  return classification != VisibilityClassification::Outside;
}

} // namespace nuri::visibility_detail
