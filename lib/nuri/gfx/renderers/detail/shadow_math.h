#pragma once

#include "nuri/gfx/frame/render_frame_context.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>

namespace nuri::shadow_detail {

struct DirectionalShadowFit {
  float splitNear = 0.0f;
  float splitFar = 0.0f;
  float texelWorldSize = 0.0f;
  glm::mat4 lightView{1.0f};
  glm::mat4 lightProj{1.0f};
  glm::mat4 lightViewProj{1.0f};
  glm::vec3 lightSpaceBoundsMin{0.0f};
  glm::vec3 lightSpaceBoundsMax{0.0f};
  glm::vec3 frustumCenter{0.0f};
  std::array<glm::vec3, 8> frustumCorners{};
};

[[nodiscard]] inline glm::vec3 normalizeSafe(glm::vec3 value,
                                             glm::vec3 fallback) {
  const float length = glm::length(value);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return fallback;
  }
  return value / length;
}

[[nodiscard]] inline glm::vec3 chooseLightUp(glm::vec3 lightDirection) {
  const glm::vec3 worldUp = std::abs(lightDirection.y) < 0.99f
                                ? glm::vec3(0.0f, 1.0f, 0.0f)
                                : glm::vec3(0.0f, 0.0f, 1.0f);
  const glm::vec3 right = glm::cross(worldUp, lightDirection);
  if (glm::dot(right, right) <= 1.0e-8f) {
    return glm::vec3(1.0f, 0.0f, 0.0f);
  }
  return worldUp;
}

[[nodiscard]] inline glm::mat4
makeDirectionalLightView(glm::vec3 lightDirection) {
  lightDirection = normalizeSafe(lightDirection, glm::vec3(0.0f, -1.0f, 0.0f));
  return glm::lookAt(glm::vec3(0.0f), lightDirection,
                     chooseLightUp(lightDirection));
}

struct CameraBasis {
  glm::vec3 right{1.0f, 0.0f, 0.0f};
  glm::vec3 up{0.0f, 1.0f, 0.0f};
  glm::vec3 forward{0.0f, 0.0f, -1.0f};
};

[[nodiscard]] inline CameraBasis
cameraBasisFromInvView(const glm::mat4 &invView) {
  return CameraBasis{
      .right =
          normalizeSafe(glm::vec3(invView[0]), glm::vec3(1.0f, 0.0f, 0.0f)),
      .up = normalizeSafe(glm::vec3(invView[1]), glm::vec3(0.0f, 1.0f, 0.0f)),
      .forward =
          normalizeSafe(-glm::vec3(invView[2]), glm::vec3(0.0f, 0.0f, -1.0f)),
  };
}

[[nodiscard]] inline std::array<glm::vec3, 8>
computeCameraSliceCorners(const CameraFrameState &camera, float splitNear,
                          float splitFar) {
  const glm::mat4 invView = glm::inverse(camera.view);
  const CameraBasis basis = cameraBasisFromInvView(invView);
  const glm::vec3 cameraPos = glm::vec3(camera.cameraPos);
  const glm::vec3 forward = basis.forward;
  const glm::vec3 right = basis.right;
  const glm::vec3 up = basis.up;
  const float nearPlane = std::max(splitNear, 0.01f);
  const float farPlane = std::max(nearPlane + 0.01f, splitFar);
  std::array<glm::vec3, 8> corners{};

  if (camera.projectionType == ProjectionType::Orthographic) {
    const float halfHeight = 0.5f * std::max(camera.orthoHeight, 0.01f);
    const float halfWidth = halfHeight * std::max(camera.aspectRatio, 0.01f);
    const glm::vec3 nearCenter = cameraPos + forward * nearPlane;
    const glm::vec3 farCenter = cameraPos + forward * farPlane;
    const glm::vec3 dx = right * halfWidth;
    const glm::vec3 dy = up * halfHeight;
    corners[0] = nearCenter - dx - dy;
    corners[1] = nearCenter + dx - dy;
    corners[2] = nearCenter + dx + dy;
    corners[3] = nearCenter - dx + dy;
    corners[4] = farCenter - dx - dy;
    corners[5] = farCenter + dx - dy;
    corners[6] = farCenter + dx + dy;
    corners[7] = farCenter - dx + dy;
    return corners;
  }

  const float tanHalfFov = std::tan(std::max(camera.fovYRadians, 0.01f) * 0.5f);
  const float nearHalfHeight = tanHalfFov * nearPlane;
  const float nearHalfWidth =
      nearHalfHeight * std::max(camera.aspectRatio, 0.01f);
  const float farHalfHeight = tanHalfFov * farPlane;
  const float farHalfWidth =
      farHalfHeight * std::max(camera.aspectRatio, 0.01f);
  const glm::vec3 nearCenter = cameraPos + forward * nearPlane;
  const glm::vec3 farCenter = cameraPos + forward * farPlane;
  const glm::vec3 nearDx = right * nearHalfWidth;
  const glm::vec3 nearDy = up * nearHalfHeight;
  const glm::vec3 farDx = right * farHalfWidth;
  const glm::vec3 farDy = up * farHalfHeight;

  corners[0] = nearCenter - nearDx - nearDy;
  corners[1] = nearCenter + nearDx - nearDy;
  corners[2] = nearCenter + nearDx + nearDy;
  corners[3] = nearCenter - nearDx + nearDy;
  corners[4] = farCenter - farDx - farDy;
  corners[5] = farCenter + farDx - farDy;
  corners[6] = farCenter + farDx + farDy;
  corners[7] = farCenter - farDx + farDy;
  return corners;
}

[[nodiscard]] inline std::array<glm::vec3, 8>
computeBoundsCorners(glm::vec3 boundsMin, glm::vec3 boundsMax) {
  return {
      glm::vec3(boundsMin.x, boundsMin.y, boundsMin.z),
      glm::vec3(boundsMax.x, boundsMin.y, boundsMin.z),
      glm::vec3(boundsMax.x, boundsMax.y, boundsMin.z),
      glm::vec3(boundsMin.x, boundsMax.y, boundsMin.z),
      glm::vec3(boundsMin.x, boundsMin.y, boundsMax.z),
      glm::vec3(boundsMax.x, boundsMin.y, boundsMax.z),
      glm::vec3(boundsMax.x, boundsMax.y, boundsMax.z),
      glm::vec3(boundsMin.x, boundsMax.y, boundsMax.z),
  };
}

[[nodiscard]] inline DirectionalShadowFit
fitSingleDirectionalShadowMap(const CameraFrameState &camera,
                              glm::vec3 lightDirection, float maxDistance,
                              uint32_t shadowMapSize, bool stabilize = false) {
  if (stabilize) {
    // TODO: Stabilize the orthographic fit with texel snapping to reduce
    // shadow swimming during camera motion.
  }
  DirectionalShadowFit fit{};
  fit.splitNear = std::max(camera.nearPlane, 0.01f);
  const float requestedFar =
      std::min(std::max(maxDistance, fit.splitNear + 0.01f),
               std::max(camera.farPlane, fit.splitNear + 0.01f));
  fit.splitFar = std::max(fit.splitNear + 0.01f, requestedFar);
  fit.frustumCorners =
      computeCameraSliceCorners(camera, fit.splitNear, fit.splitFar);

  for (const glm::vec3 corner : fit.frustumCorners) {
    fit.frustumCenter += corner;
  }
  fit.frustumCenter /= static_cast<float>(fit.frustumCorners.size());

  float radius = 0.0f;
  for (const glm::vec3 corner : fit.frustumCorners) {
    radius = std::max(radius, glm::length(corner - fit.frustumCenter));
  }
  radius = std::max(radius, 1.0f);

  lightDirection = normalizeSafe(lightDirection, glm::vec3(0.0f, -1.0f, 0.0f));
  fit.lightView = makeDirectionalLightView(lightDirection);

  glm::vec3 lightMin(std::numeric_limits<float>::max());
  glm::vec3 lightMax(std::numeric_limits<float>::lowest());
  for (const glm::vec3 corner : fit.frustumCorners) {
    const glm::vec3 lightSpace =
        glm::vec3(fit.lightView * glm::vec4(corner, 1.0f));
    lightMin = glm::min(lightMin, lightSpace);
    lightMax = glm::max(lightMax, lightSpace);
  }

  const float xyPadding = std::max(radius * 0.1f, 1.0f);
  lightMin.x -= xyPadding;
  lightMin.y -= xyPadding;
  lightMax.x += xyPadding;
  lightMax.y += xyPadding;

  float width = std::max(lightMax.x - lightMin.x, 0.01f);
  float height = std::max(lightMax.y - lightMin.y, 0.01f);
  fit.texelWorldSize =
      std::max(width, height) / static_cast<float>(std::max(shadowMapSize, 1u));
  const float depth = std::max(lightMax.z - lightMin.z, 0.01f);
  const float depthPadding = std::max(depth * 0.01f, 0.01f);
  const float nearPlane = -lightMax.z - depthPadding;
  float farPlane = -lightMin.z + depthPadding;
  if (farPlane <= nearPlane + 0.01f) {
    farPlane = nearPlane + 0.01f;
  }
  fit.lightProj = glm::orthoRH_ZO(lightMin.x, lightMax.x, lightMin.y,
                                  lightMax.y, nearPlane, farPlane);
  fit.lightViewProj = fit.lightProj * fit.lightView;
  fit.lightSpaceBoundsMin = glm::vec3(lightMin.x, lightMin.y, -farPlane);
  fit.lightSpaceBoundsMax = glm::vec3(lightMax.x, lightMax.y, -nearPlane);
  return fit;
}

[[nodiscard]] inline DirectionalShadowFit
fitDirectionalShadowMapToBounds(const CameraFrameState &camera,
                                glm::vec3 boundsMin, glm::vec3 boundsMax,
                                glm::vec3 lightDirection, float maxDistance,
                                uint32_t shadowMapSize, bool stabilize) {
  if (stabilize) {
    // TODO: Apply the same texel-snapped stabilization path used for camera
    // frustum fits once the bounds-based fit is stabilized.
  }
  DirectionalShadowFit fit{};
  fit.splitNear = std::max(camera.nearPlane, 0.01f);
  fit.splitFar =
      std::max(fit.splitNear + 0.01f, std::max(maxDistance, fit.splitNear));
  const glm::vec3 originalMin = boundsMin;
  const glm::vec3 originalMax = boundsMax;
  boundsMin = glm::min(originalMin, originalMax);
  boundsMax = glm::max(originalMin, originalMax);
  fit.frustumCorners = computeBoundsCorners(boundsMin, boundsMax);
  fit.frustumCenter = (boundsMin + boundsMax) * 0.5f;

  lightDirection = normalizeSafe(lightDirection, glm::vec3(0.0f, -1.0f, 0.0f));
  fit.lightView = makeDirectionalLightView(lightDirection);

  glm::vec3 lightMin(std::numeric_limits<float>::max());
  glm::vec3 lightMax(std::numeric_limits<float>::lowest());
  for (const glm::vec3 corner : fit.frustumCorners) {
    const glm::vec3 lightSpace =
        glm::vec3(fit.lightView * glm::vec4(corner, 1.0f));
    lightMin = glm::min(lightMin, lightSpace);
    lightMax = glm::max(lightMax, lightSpace);
  }

  const float unpaddedWidth = std::max(lightMax.x - lightMin.x, 0.01f);
  const float unpaddedHeight = std::max(lightMax.y - lightMin.y, 0.01f);
  const float xyPadding =
      std::max(std::max(unpaddedWidth, unpaddedHeight) * 0.001f, 0.01f);
  lightMin.x -= xyPadding;
  lightMin.y -= xyPadding;
  lightMax.x += xyPadding;
  lightMax.y += xyPadding;

  const float width = std::max(lightMax.x - lightMin.x, 0.01f);
  const float height = std::max(lightMax.y - lightMin.y, 0.01f);
  fit.texelWorldSize =
      std::max(width, height) / static_cast<float>(std::max(shadowMapSize, 1u));
  const float depth = std::max(lightMax.z - lightMin.z, 0.01f);
  const float depthPadding = std::max(depth * 0.001f, 0.01f);
  const float nearPlane = -lightMax.z - depthPadding;
  float farPlane = -lightMin.z + depthPadding;
  if (farPlane <= nearPlane + 0.01f) {
    farPlane = nearPlane + 0.01f;
  }
  fit.lightProj = glm::orthoRH_ZO(lightMin.x, lightMax.x, lightMin.y,
                                  lightMax.y, nearPlane, farPlane);
  fit.lightViewProj = fit.lightProj * fit.lightView;
  fit.lightSpaceBoundsMin = glm::vec3(lightMin.x, lightMin.y, -farPlane);
  fit.lightSpaceBoundsMax = glm::vec3(lightMax.x, lightMax.y, -nearPlane);
  return fit;
}

} // namespace nuri::shadow_detail
