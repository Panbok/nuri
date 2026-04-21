#pragma once

#include "nuri/gfx/frame/render_frame_context.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>

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
  glm::vec3 unsnappedCenter{0.0f};
  glm::vec3 snappedCenter{0.0f};
  glm::vec2 unsnappedLightSpaceCenter{0.0f};
  glm::vec2 snappedLightSpaceCenter{0.0f};
  std::array<glm::vec3, 8> frustumCorners{};
};

struct ShadowSdsmHistogramSourceSelection {
  uint32_t level = 0u;
  glm::uvec2 dimensions{0u};
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

[[nodiscard]] inline bool shadowCasterOverlapsLightSpaceBounds(
    std::span<const glm::vec3, 8> casterWorldCorners,
    const glm::mat4 &lightView, glm::vec3 lightSpaceBoundsMin,
    glm::vec3 lightSpaceBoundsMax, float padding) {
  glm::vec3 casterLightMin(std::numeric_limits<float>::max());
  glm::vec3 casterLightMax(std::numeric_limits<float>::lowest());
  for (const glm::vec3 corner : casterWorldCorners) {
    const glm::vec3 lightSpace = glm::vec3(lightView * glm::vec4(corner, 1.0f));
    casterLightMin = glm::min(casterLightMin, lightSpace);
    casterLightMax = glm::max(casterLightMax, lightSpace);
  }

  const glm::vec3 normalizedMin =
      glm::min(lightSpaceBoundsMin, lightSpaceBoundsMax);
  const glm::vec3 normalizedMax =
      glm::max(lightSpaceBoundsMin, lightSpaceBoundsMax);
  const glm::vec3 conservativePadding(std::max(padding, 0.0f));
  const glm::vec3 paddedMin = normalizedMin - conservativePadding;
  const glm::vec3 paddedMax = normalizedMax + conservativePadding;
  lightSpaceBoundsMin = paddedMin;
  lightSpaceBoundsMax = paddedMax;

  return casterLightMax.x >= lightSpaceBoundsMin.x &&
         casterLightMin.x <= lightSpaceBoundsMax.x &&
         casterLightMax.y >= lightSpaceBoundsMin.y &&
         casterLightMin.y <= lightSpaceBoundsMax.y &&
         casterLightMax.z >= lightSpaceBoundsMin.z &&
         casterLightMin.z <= lightSpaceBoundsMax.z;
}

[[nodiscard]] inline glm::vec3 lightSpaceCenter(glm::vec3 boundsMin,
                                                glm::vec3 boundsMax) {
  return (boundsMin + boundsMax) * 0.5f;
}

inline void stabilizeOrthoBounds(glm::vec3 &lightMin, glm::vec3 &lightMax,
                                 uint32_t shadowMapSize, bool stabilize,
                                 DirectionalShadowFit &fit,
                                 const glm::mat4 &inverseLightView) {
  const float width = std::max(lightMax.x - lightMin.x, 0.01f);
  const float height = std::max(lightMax.y - lightMin.y, 0.01f);
  fit.texelWorldSize =
      std::max(width, height) / static_cast<float>(std::max(shadowMapSize, 1u));

  const glm::vec3 unsnappedLightCenter = lightSpaceCenter(lightMin, lightMax);
  fit.unsnappedLightSpaceCenter = glm::vec2(unsnappedLightCenter);

  glm::vec3 snappedLightCenter = unsnappedLightCenter;
  if (stabilize && fit.texelWorldSize > 0.0f) {
    const float texelStep = fit.texelWorldSize;
    snappedLightCenter.x =
        std::round(unsnappedLightCenter.x / texelStep) * texelStep;
    snappedLightCenter.y =
        std::round(unsnappedLightCenter.y / texelStep) * texelStep;
    lightMin.x = snappedLightCenter.x - width * 0.5f;
    lightMax.x = snappedLightCenter.x + width * 0.5f;
    lightMin.y = snappedLightCenter.y - height * 0.5f;
    lightMax.y = snappedLightCenter.y + height * 0.5f;
  }
  fit.snappedLightSpaceCenter = glm::vec2(snappedLightCenter);

  fit.unsnappedCenter =
      glm::vec3(inverseLightView * glm::vec4(unsnappedLightCenter, 1.0f));
  fit.snappedCenter =
      glm::vec3(inverseLightView * glm::vec4(snappedLightCenter, 1.0f));
}

[[nodiscard]] inline float
linearizeDeviceDepthToViewDepth(float deviceDepth,
                                const CameraFrameState &camera) {
  const float clampedDepth = std::clamp(deviceDepth, 0.0f, 1.0f);
  const float nearPlane = std::max(camera.nearPlane, 0.01f);
  const float farPlane = std::max(camera.farPlane, nearPlane + 0.01f);
  if (camera.projectionType == ProjectionType::Orthographic) {
    return glm::mix(nearPlane, farPlane, clampedDepth);
  }
  const float denominator = farPlane - clampedDepth * (farPlane - nearPlane);
  if (!std::isfinite(denominator) || denominator <= 1.0e-6f) {
    return farPlane;
  }
  return nearPlane * farPlane / denominator;
}

[[nodiscard]] inline std::array<float, kMaxShadowCascades + 1u>
computeCascadeSplitDepthsForRange(float nearDepth, float farDepth,
                                  uint32_t cascadeCount,
                                  ShadowCascadeSplitMode splitMode,
                                  float lambda) {
  const uint32_t safeCascadeCount =
      std::clamp(cascadeCount, 1u, kMaxShadowCascades);
  const float effectiveNear = std::max(nearDepth, 0.01f);
  const float effectiveFar = std::max(effectiveNear + 0.01f, farDepth);
  const float clampedLambda = std::clamp(lambda, 0.0f, 1.0f);

  std::array<float, kMaxShadowCascades + 1u> splits{};
  splits.fill(effectiveFar);
  splits[0] = effectiveNear;
  for (uint32_t i = 1u; i < safeCascadeCount; ++i) {
    const float t =
        static_cast<float>(i) / static_cast<float>(safeCascadeCount);
    const float uniformSplit =
        effectiveNear + (effectiveFar - effectiveNear) * t;
    const float logSplit =
        effectiveNear * std::pow(effectiveFar / effectiveNear, t);
    switch (sanitizeShadowCascadeSplitMode(splitMode)) {
    case ShadowCascadeSplitMode::Uniform:
      splits[i] = uniformSplit;
      break;
    case ShadowCascadeSplitMode::Logarithmic:
      splits[i] = logSplit;
      break;
    case ShadowCascadeSplitMode::Practical:
    default:
      splits[i] = glm::mix(uniformSplit, logSplit, clampedLambda);
      break;
    }
  }
  splits[safeCascadeCount] = effectiveFar;
  return splits;
}

[[nodiscard]] inline std::array<float, kMaxShadowCascades + 1u>
computeCascadeSplitDepths(const CameraFrameState &camera, float maxDistance,
                          uint32_t cascadeCount,
                          ShadowCascadeSplitMode splitMode, float lambda) {
  const float effectiveNear = std::max(camera.nearPlane, 0.01f);
  const float requestedFar =
      std::min(std::max(maxDistance, effectiveNear + 0.01f),
               std::max(camera.farPlane, effectiveNear + 0.01f));
  const float effectiveFar = std::max(effectiveNear + 0.01f, requestedFar);
  return computeCascadeSplitDepthsForRange(effectiveNear, effectiveFar,
                                           cascadeCount, splitMode, lambda);
}

[[nodiscard]] inline ShadowSdsmHistogramSourceSelection
selectSdsmHistogramSourceLevel(std::span<const glm::uvec2> levelDimensions,
                               uint32_t levelCount,
                               uint32_t maxTexelCount = 4096u) {
  const uint32_t safeLevelCount =
      std::min(levelCount, static_cast<uint32_t>(levelDimensions.size()));
  if (safeLevelCount == 0u) {
    return {};
  }
  for (uint32_t level = 0u; level < safeLevelCount; ++level) {
    const glm::uvec2 dimensions =
        glm::max(levelDimensions[level], glm::uvec2(1u));
    const uint64_t texelCount = static_cast<uint64_t>(dimensions.x) *
                                static_cast<uint64_t>(dimensions.y);
    if (texelCount <= static_cast<uint64_t>(std::max(maxTexelCount, 1u))) {
      return ShadowSdsmHistogramSourceSelection{
          .level = level,
          .dimensions = dimensions,
      };
    }
  }
  return ShadowSdsmHistogramSourceSelection{
      .level = safeLevelCount - 1u,
      .dimensions =
          glm::max(levelDimensions[safeLevelCount - 1u], glm::uvec2(1u)),
  };
}

inline void accumulateSdsmHistogramInterval(std::span<float> histogramBuckets,
                                            float rangeNear, float rangeFar,
                                            float intervalMin,
                                            float intervalMax,
                                            float weight = 1.0f) {
  if (histogramBuckets.empty() || !std::isfinite(rangeNear) ||
      !std::isfinite(rangeFar) || !std::isfinite(intervalMin) ||
      !std::isfinite(intervalMax) || !std::isfinite(weight) || weight <= 0.0f) {
    return;
  }

  const float clampedNear = std::max(rangeNear, 0.0f);
  const float clampedFar = std::max(clampedNear + 1.0e-4f, rangeFar);
  const float clampedMin =
      std::clamp(std::min(intervalMin, intervalMax), clampedNear, clampedFar);
  const float clampedMax =
      std::clamp(std::max(intervalMin, intervalMax), clampedNear, clampedFar);
  const float rangeSpan = clampedFar - clampedNear;
  if (rangeSpan <= 1.0e-6f) {
    histogramBuckets.front() += weight;
    return;
  }

  const size_t bucketCount = histogramBuckets.size();
  if (clampedMax <= clampedMin + 1.0e-6f) {
    const float normalized =
        std::clamp((clampedMin - clampedNear) / rangeSpan, 0.0f, 0.99999994f);
    const size_t bucketIndex = std::min(
        static_cast<size_t>(normalized * static_cast<float>(bucketCount)),
        bucketCount - 1u);
    histogramBuckets[bucketIndex] += weight;
    return;
  }

  const float bucketWidth = rangeSpan / static_cast<float>(bucketCount);
  const size_t startBucket = std::min(
      static_cast<size_t>(std::floor((clampedMin - clampedNear) / bucketWidth)),
      bucketCount - 1u);
  const size_t endBucket = std::min(
      static_cast<size_t>(std::floor((clampedMax - clampedNear) / bucketWidth)),
      bucketCount - 1u);
  const float intervalSpan = clampedMax - clampedMin;
  for (size_t bucketIndex = startBucket; bucketIndex <= endBucket;
       ++bucketIndex) {
    const float bucketStart =
        clampedNear + bucketWidth * static_cast<float>(bucketIndex);
    const float bucketEnd = std::min(bucketStart + bucketWidth, clampedFar);
    const float overlap = std::max(std::min(clampedMax, bucketEnd) -
                                       std::max(clampedMin, bucketStart),
                                   0.0f);
    if (overlap > 0.0f) {
      histogramBuckets[bucketIndex] += weight * (overlap / intervalSpan);
    }
  }
}

[[nodiscard]] inline float
sdsmHistogramPercentileDepth(std::span<const float> histogramBuckets,
                             float rangeNear, float rangeFar,
                             float percentile) {
  if (histogramBuckets.empty() || !std::isfinite(rangeNear) ||
      !std::isfinite(rangeFar)) {
    return rangeNear;
  }
  const float clampedNear = std::max(rangeNear, 0.0f);
  const float clampedFar = std::max(clampedNear + 1.0e-4f, rangeFar);
  const float bucketWidth =
      (clampedFar - clampedNear) / static_cast<float>(histogramBuckets.size());
  float totalWeight = 0.0f;
  for (const float weight : histogramBuckets) {
    if (std::isfinite(weight) && weight > 0.0f) {
      totalWeight += weight;
    }
  }
  if (totalWeight <= 1.0e-6f) {
    return clampedNear;
  }
  const float targetWeight = std::clamp(percentile, 0.0f, 1.0f) * totalWeight;
  float accumulatedWeight = 0.0f;
  for (size_t bucketIndex = 0u; bucketIndex < histogramBuckets.size();
       ++bucketIndex) {
    const float bucketWeight = std::isfinite(histogramBuckets[bucketIndex]) &&
                                       histogramBuckets[bucketIndex] > 0.0f
                                   ? histogramBuckets[bucketIndex]
                                   : 0.0f;
    if (accumulatedWeight + bucketWeight >= targetWeight) {
      const float localWeight =
          bucketWeight > 1.0e-6f
              ? std::clamp((targetWeight - accumulatedWeight) / bucketWeight,
                           0.0f, 1.0f)
              : 0.0f;
      return std::clamp(clampedNear +
                            (static_cast<float>(bucketIndex) + localWeight) *
                                bucketWidth,
                        clampedNear, clampedFar);
    }
    accumulatedWeight += bucketWeight;
  }
  return clampedFar;
}

inline void enforceMonotonicShadowSplitDepths(
    std::array<float, kMaxShadowCascades + 1u> &splitDepths,
    uint32_t cascadeCount, float rangeNear, float rangeFar,
    float minimumSpacing = 1.0e-4f) {
  const uint32_t safeCascadeCount =
      std::clamp(cascadeCount, 1u, kMaxShadowCascades);
  const float clampedNear = std::max(rangeNear, 0.0f);
  const float clampedFar = std::max(
      clampedNear + minimumSpacing * static_cast<float>(safeCascadeCount),
      rangeFar);
  splitDepths[0] = clampedNear;
  splitDepths[safeCascadeCount] = clampedFar;
  for (uint32_t i = 1u; i < safeCascadeCount; ++i) {
    const float minDepth = splitDepths[i - 1u] + minimumSpacing;
    const float maxDepth =
        clampedFar - minimumSpacing * static_cast<float>(safeCascadeCount - i);
    splitDepths[i] = std::clamp(splitDepths[i], minDepth, maxDepth);
  }
}

[[nodiscard]] inline DirectionalShadowFit fitDirectionalShadowCascadeSlice(
    const CameraFrameState &camera, float splitNear, float splitFar,
    glm::vec3 lightDirection, uint32_t shadowMapSize,
    std::span<const glm::vec3> casterPoints = {}, bool stabilize = false) {
  DirectionalShadowFit fit{};
  fit.splitNear = std::max(splitNear, 0.01f);
  fit.splitFar = std::max(fit.splitNear + 0.01f, splitFar);
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
  const glm::mat4 inverseLightView = glm::inverse(fit.lightView);
  const glm::vec3 lightSpaceCenter =
      glm::vec3(fit.lightView * glm::vec4(fit.frustumCenter, 1.0f));

  glm::vec3 lightMin(lightSpaceCenter.x - radius, lightSpaceCenter.y - radius,
                     std::numeric_limits<float>::max());
  glm::vec3 lightMax(lightSpaceCenter.x + radius, lightSpaceCenter.y + radius,
                     std::numeric_limits<float>::lowest());

  const auto accumulateDepth = [&](glm::vec3 point) {
    const float z = glm::vec3(fit.lightView * glm::vec4(point, 1.0f)).z;
    lightMin.z = std::min(lightMin.z, z);
    lightMax.z = std::max(lightMax.z, z);
  };
  for (const glm::vec3 corner : fit.frustumCorners) {
    accumulateDepth(corner);
  }
  for (const glm::vec3 point : casterPoints) {
    accumulateDepth(point);
  }
  if (!std::isfinite(lightMin.z) || !std::isfinite(lightMax.z)) {
    lightMin.z = lightSpaceCenter.z - radius;
    lightMax.z = lightSpaceCenter.z + radius;
  }

  stabilizeOrthoBounds(lightMin, lightMax, shadowMapSize, stabilize, fit,
                       inverseLightView);
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
fitSingleDirectionalShadowMap(const CameraFrameState &camera,
                              glm::vec3 lightDirection, float maxDistance,
                              uint32_t shadowMapSize, bool stabilize = false) {
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
  const glm::mat4 inverseLightView = glm::inverse(fit.lightView);

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

  stabilizeOrthoBounds(lightMin, lightMax, shadowMapSize, stabilize, fit,
                       inverseLightView);
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
  const glm::mat4 inverseLightView = glm::inverse(fit.lightView);

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

  stabilizeOrthoBounds(lightMin, lightMax, shadowMapSize, stabilize, fit,
                       inverseLightView);
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
