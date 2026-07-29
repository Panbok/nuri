#include "nuri/gfx/renderers/debug_renderer.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/debug_draw_3d.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderers/detail/visibility_math.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"
namespace nuri {
namespace {
constexpr uint32_t kGridDrawDebugColor = 0xff66aaff;
constexpr uint32_t kGridVertexCount = 6;
constexpr std::string_view kGridPipelineName = "debug_grid";
constexpr std::string_view kGridDrawLabel = "DebugGrid Draw";
const glm::vec4 kDirectionalLightIconColor(1.0f, 0.88f, 0.25f, 1.0f);
const glm::vec4 kPointLightIconColor(0.35f, 0.82f, 1.0f, 1.0f);
const glm::vec4 kSpotLightIconColor(1.0f, 0.55f, 0.24f, 1.0f);
const glm::vec4 kSelectedLightIconColor(1.0f, 0.28f, 0.16f, 1.0f);
const std::array<glm::vec4, kMaxShadowCascades> kShadowCascadeColors = {
    glm::vec4(0.15f, 1.0f, 0.3f, 1.0f),
    glm::vec4(0.0f, 0.85f, 1.0f, 1.0f),
    glm::vec4(1.0f, 0.9f, 0.1f, 1.0f),
    glm::vec4(1.0f, 0.25f, 1.0f, 1.0f),
};
const glm::vec4 kShadowLightBoundsColor(1.0f, 0.45f, 0.15f, 1.0f);
const glm::vec4 kShadowLightRayColor(1.0f, 0.92f, 0.35f, 1.0f);
const glm::vec4 kShadowTexelSnapColor(0.95f, 0.95f, 1.0f, 1.0f);
const glm::vec4 kModelBoundsColor(1.0f, 1.0f, 0.0f, 1.0f);
const glm::vec4 kVisibilityBoundsColor(0.25f, 0.85f, 1.0f, 1.0f);
const glm::vec4 kVisibilityMeshletBoundsColor(0.35f, 0.62f, 1.0f, 1.0f);
const glm::vec4 kVisibilityInsideColor(0.18f, 1.0f, 0.32f, 1.0f);
const glm::vec4 kVisibilityIntersectingColor(1.0f, 0.78f, 0.16f, 1.0f);
const glm::vec4 kVisibilityOutsideColor(1.0f, 0.18f, 0.12f, 1.0f);
const glm::vec4 kVisibilityConservativeColor(0.88f, 0.42f, 1.0f, 1.0f);
const glm::vec4 kDDGIVolumeColor(0.2f, 0.72f, 1.0f, 0.9f);
const glm::vec4 kSelectedDDGIVolumeColor(1.0f, 0.5f, 0.12f, 1.0f);
const glm::vec4 kDDGIGridColor(0.18f, 0.58f, 0.82f, 0.45f);
[[nodiscard]] glm::vec3 safeNormalize(const glm::vec3 &value,
                                      const glm::vec3 &fallback) {
  const float length = glm::length(value);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return fallback;
  }
  return value / length;
}
[[nodiscard]] float lightIconScale(const CameraFrameState &camera,
                                   const glm::vec3 &position) {
  const float distance = glm::length(glm::vec3(camera.cameraPos) - position);
  return std::clamp(distance * 0.08f, 0.2f, 3.0f);
}
[[nodiscard]] glm::vec4 visibilityBoundsColor(
    const RenderFrameContext &frame, const Renderable &renderable,
    const BoundingBox &bounds,
    const visibility_detail::FrustumPlanes &frustum) noexcept {
  if (!frame.settings.visibility.debug.visualizeCullReason) {
    return kVisibilityBoundsColor;
  }
  if (!renderable.morphWeights.empty() || !renderable.skinPalette.empty()) {
    return kVisibilityConservativeColor;
  }
  const visibility_detail::VisibilityClassification classification =
      visibility_detail::classifyTransformedBounds(frustum, bounds,
                                                   renderable.modelMatrix);
  switch (classification) {
  case visibility_detail::VisibilityClassification::Outside:
    return kVisibilityOutsideColor;
  case visibility_detail::VisibilityClassification::Intersects:
    return kVisibilityIntersectingColor;
  case visibility_detail::VisibilityClassification::Inside:
    return kVisibilityInsideColor;
  }
  return kVisibilityBoundsColor;
}
[[nodiscard]] glm::vec4 visibilityMeshletBoundsColor(
    const RenderFrameContext &frame, const Renderable &renderable,
    const glm::vec4 &boundsSphere,
    const visibility_detail::FrustumPlanes &frustum) noexcept {
  if (!frame.settings.visibility.debug.visualizeCullReason) {
    return kVisibilityMeshletBoundsColor;
  }
  if (!renderable.morphWeights.empty() || !renderable.skinPalette.empty()) {
    return kVisibilityConservativeColor;
  }
  const float localRadius =
      std::isfinite(boundsSphere.w) ? std::max(boundsSphere.w, 0.0f) : 0.0f;
  const glm::vec3 worldCenter = glm::vec3(
      renderable.modelMatrix * glm::vec4(glm::vec3(boundsSphere), 1.0f));
  const float worldRadius =
      localRadius * visibility_detail::maxAxisScale(renderable.modelMatrix);
  const visibility_detail::VisibilityClassification classification =
      visibility_detail::classifySphere(frustum, worldCenter, worldRadius);
  switch (classification) {
  case visibility_detail::VisibilityClassification::Outside:
    return kVisibilityOutsideColor;
  case visibility_detail::VisibilityClassification::Intersects:
    return kVisibilityIntersectingColor;
  case visibility_detail::VisibilityClassification::Inside:
    return kVisibilityInsideColor;
  }
  return kVisibilityMeshletBoundsColor;
}
void buildLightBasis(const glm::vec3 &direction, glm::vec3 &outRight,
                     glm::vec3 &outUp) {
  const glm::vec3 dir = safeNormalize(direction, glm::vec3(0.0f, 0.0f, -1.0f));
  const glm::vec3 fallbackUp = std::abs(dir.y) > 0.95f
                                   ? glm::vec3(1.0f, 0.0f, 0.0f)
                                   : glm::vec3(0.0f, 1.0f, 0.0f);
  outRight =
      safeNormalize(glm::cross(fallbackUp, dir), glm::vec3(1.0f, 0.0f, 0.0f));
  outUp = safeNormalize(glm::cross(dir, outRight), glm::vec3(0.0f, 1.0f, 0.0f));
}
void drawDirectionalLightIcon(DebugDraw3D &debugDraw, const glm::vec3 &position,
                              const glm::vec3 &direction, float scale,
                              const glm::vec4 &color) {
  const glm::vec3 dir = safeNormalize(direction, glm::vec3(0.0f, 0.0f, -1.0f));
  glm::vec3 right(1.0f, 0.0f, 0.0f);
  glm::vec3 up(0.0f, 1.0f, 0.0f);
  buildLightBasis(dir, right, up);
  const glm::vec3 end = position + dir * (scale * 2.4f);
  const glm::vec3 headBase = end - dir * (scale * 0.8f);
  debugDraw.line(position, end, color);
  debugDraw.line(end, headBase + right * (scale * 0.45f), color);
  debugDraw.line(end, headBase - right * (scale * 0.45f), color);
  debugDraw.line(end, headBase + up * (scale * 0.45f), color);
  debugDraw.line(end, headBase - up * (scale * 0.45f), color);
}
void drawPointLightIcon(DebugDraw3D &debugDraw, const glm::vec3 &position,
                        float scale, const glm::vec4 &color) {
  const glm::vec3 x(scale, 0.0f, 0.0f);
  const glm::vec3 y(0.0f, scale, 0.0f);
  const glm::vec3 z(0.0f, 0.0f, scale);
  debugDraw.line(position - x, position + x, color);
  debugDraw.line(position - y, position + y, color);
  debugDraw.line(position - z, position + z, color);
  debugDraw.line(position + x, position + y, color);
  debugDraw.line(position + y, position - x, color);
  debugDraw.line(position - x, position - y, color);
  debugDraw.line(position - y, position + x, color);
}
void drawSpotLightIcon(DebugDraw3D &debugDraw, const glm::vec3 &position,
                       const glm::vec3 &direction, float scale,
                       const glm::vec4 &color) {
  const glm::vec3 dir = safeNormalize(direction, glm::vec3(0.0f, 0.0f, -1.0f));
  glm::vec3 right(1.0f, 0.0f, 0.0f);
  glm::vec3 up(0.0f, 1.0f, 0.0f);
  buildLightBasis(dir, right, up);
  const glm::vec3 baseCenter = position + dir * (scale * 2.2f);
  const float radius = scale * 0.9f;
  const glm::vec3 p0 = baseCenter + right * radius;
  const glm::vec3 p1 = baseCenter + up * radius;
  const glm::vec3 p2 = baseCenter - right * radius;
  const glm::vec3 p3 = baseCenter - up * radius;
  debugDraw.line(position, p0, color);
  debugDraw.line(position, p1, color);
  debugDraw.line(position, p2, color);
  debugDraw.line(position, p3, color);
  debugDraw.line(p0, p1, color);
  debugDraw.line(p1, p2, color);
  debugDraw.line(p2, p3, color);
  debugDraw.line(p3, p0, color);
}
[[nodiscard]] bool shadowOverlayEnabled(const RenderFrameContext &frame) {
  if (!frame.settings.shadow.enabled) {
    return false;
  }
  const RenderSettings::ShadowDebugSettings &debug =
      frame.settings.shadow.debug;
  return debug.showCascadeFrusta || debug.showLightViewBounds ||
         debug.showTexelGridSnap;
}
[[nodiscard]] bool shadowTexelGridSnapEnabled(const RenderFrameContext &frame) {
  return frame.settings.shadow.enabled &&
         frame.settings.shadow.debug.showTexelGridSnap;
}
void accumulateSortDepth(float &sortDepth, const glm::mat4 &view,
                         const glm::vec3 &position) {
  sortDepth = std::max(sortDepth, -(view * glm::vec4(position, 1.0f)).z);
}
[[nodiscard]] glm::vec3 dehomogenize(const glm::vec4 &value) {
  if (std::abs(value.w) <= 1.0e-6f) {
    return glm::vec3(value);
  }
  return glm::vec3(value) / value.w;
}
void drawCornerBox(DebugDraw3D &debugDraw,
                   const std::array<glm::vec3, 8> &corners,
                   const glm::vec4 &color) {
  debugDraw.line(corners[0], corners[1], color);
  debugDraw.line(corners[1], corners[2], color);
  debugDraw.line(corners[2], corners[3], color);
  debugDraw.line(corners[3], corners[0], color);
  debugDraw.line(corners[4], corners[5], color);
  debugDraw.line(corners[5], corners[6], color);
  debugDraw.line(corners[6], corners[7], color);
  debugDraw.line(corners[7], corners[4], color);
  debugDraw.line(corners[0], corners[4], color);
  debugDraw.line(corners[1], corners[5], color);
  debugDraw.line(corners[2], corners[6], color);
  debugDraw.line(corners[3], corners[7], color);
}
void drawShadowCascadeFrustum(DebugDraw3D &debugDraw,
                              const ShadowCascadeDebugFrameData &cascade,
                              const glm::vec4 &color, const glm::mat4 &view,
                              float &sortDepth) {
  std::array<glm::vec3, 8> corners{};
  for (size_t i = 0; i < corners.size(); ++i) {
    corners[i] = dehomogenize(cascade.worldFrustumCorners[i]);
    accumulateSortDepth(sortDepth, view, corners[i]);
  }
  drawCornerBox(debugDraw, corners, color);
  const glm::vec3 nearCenter =
      (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;
  const glm::vec3 farCenter =
      (corners[4] + corners[5] + corners[6] + corners[7]) * 0.25f;
  const glm::vec4 guideColor(color.r, color.g, color.b, 0.65f);
  debugDraw.line(corners[4], corners[6], guideColor);
  debugDraw.line(corners[5], corners[7], guideColor);
  debugDraw.line(nearCenter, farCenter, guideColor);
  accumulateSortDepth(sortDepth, view, nearCenter);
  accumulateSortDepth(sortDepth, view, farCenter);
}
void drawShadowLightBounds(DebugDraw3D &debugDraw,
                           const ShadowCascadeDebugFrameData &cascade,
                           const glm::vec4 &color, const glm::mat4 &view,
                           float &sortDepth) {
  const glm::vec3 boundsMin = glm::min(glm::vec3(cascade.lightSpaceBoundsMin),
                                       glm::vec3(cascade.lightSpaceBoundsMax));
  const glm::vec3 boundsMax = glm::max(glm::vec3(cascade.lightSpaceBoundsMin),
                                       glm::vec3(cascade.lightSpaceBoundsMax));
  const std::array<glm::vec3, 8> lightSpaceCorners = {
      glm::vec3(boundsMin.x, boundsMin.y, boundsMin.z),
      glm::vec3(boundsMax.x, boundsMin.y, boundsMin.z),
      glm::vec3(boundsMax.x, boundsMax.y, boundsMin.z),
      glm::vec3(boundsMin.x, boundsMax.y, boundsMin.z),
      glm::vec3(boundsMin.x, boundsMin.y, boundsMax.z),
      glm::vec3(boundsMax.x, boundsMin.y, boundsMax.z),
      glm::vec3(boundsMax.x, boundsMax.y, boundsMax.z),
      glm::vec3(boundsMin.x, boundsMax.y, boundsMax.z),
  };
  std::array<glm::vec3, 8> worldCorners{};
  for (size_t i = 0; i < worldCorners.size(); ++i) {
    worldCorners[i] = glm::vec3(cascade.inverseLightView *
                                glm::vec4(lightSpaceCorners[i], 1.0f));
    accumulateSortDepth(sortDepth, view, worldCorners[i]);
  }
  drawCornerBox(debugDraw, worldCorners, color);
}
void drawShadowTexelSnap(DebugDraw3D &debugDraw,
                         const ShadowCascadeDebugFrameData &cascade,
                         const glm::vec4 &color, const glm::mat4 &view,
                         float &sortDepth) {
  const glm::vec3 center = dehomogenize(cascade.snappedCenter);
  const float texelSize = std::max(cascade.texelWorldSize, 0.001f);
  const glm::vec3 boundsMin = glm::min(glm::vec3(cascade.lightSpaceBoundsMin),
                                       glm::vec3(cascade.lightSpaceBoundsMax));
  const glm::vec3 boundsMax = glm::max(glm::vec3(cascade.lightSpaceBoundsMin),
                                       glm::vec3(cascade.lightSpaceBoundsMax));
  const float width = std::max(boundsMax.x - boundsMin.x, texelSize);
  const float height = std::max(boundsMax.y - boundsMin.y, texelSize);
  const float maxExtent = std::max(width, height);
  constexpr float kTargetGridLines = 16.0f;
  const float spacing =
      texelSize *
      std::max(1.0f, std::ceil(maxExtent / (texelSize * kTargetGridLines)));
  const glm::vec3 snappedLight =
      glm::vec3(cascade.lightView * glm::vec4(center, 1.0f));
  const float gridZ = boundsMax.z;
  const glm::vec4 gridColor(color.r, color.g, color.b, 0.55f);
  const auto toWorld = [&](float x, float y, float z) {
    return glm::vec3(cascade.inverseLightView * glm::vec4(x, y, z, 1.0f));
  };
  const float firstX =
      snappedLight.x +
      std::floor((boundsMin.x - snappedLight.x) / spacing) * spacing;
  const float firstY =
      snappedLight.y +
      std::floor((boundsMin.y - snappedLight.y) / spacing) * spacing;
  constexpr int kMaxGridLinesPerAxis = 48;
  for (int i = 0; i < kMaxGridLinesPerAxis; ++i) {
    const float x = firstX + static_cast<float>(i) * spacing;
    if (x > boundsMax.x) {
      break;
    }
    if (x >= boundsMin.x) {
      const glm::vec3 p0 = toWorld(x, boundsMin.y, gridZ);
      const glm::vec3 p1 = toWorld(x, boundsMax.y, gridZ);
      debugDraw.line(p0, p1, gridColor);
      accumulateSortDepth(sortDepth, view, p0);
      accumulateSortDepth(sortDepth, view, p1);
    }
  }
  for (int i = 0; i < kMaxGridLinesPerAxis; ++i) {
    const float y = firstY + static_cast<float>(i) * spacing;
    if (y > boundsMax.y) {
      break;
    }
    if (y >= boundsMin.y) {
      const glm::vec3 p0 = toWorld(boundsMin.x, y, gridZ);
      const glm::vec3 p1 = toWorld(boundsMax.x, y, gridZ);
      debugDraw.line(p0, p1, gridColor);
      accumulateSortDepth(sortDepth, view, p0);
      accumulateSortDepth(sortDepth, view, p1);
    }
  }
  std::array<glm::vec3, 4> faceCorners = {
      toWorld(boundsMin.x, boundsMin.y, gridZ),
      toWorld(boundsMax.x, boundsMin.y, gridZ),
      toWorld(boundsMax.x, boundsMax.y, gridZ),
      toWorld(boundsMin.x, boundsMax.y, gridZ),
  };
  debugDraw.line(faceCorners[0], faceCorners[1], color);
  debugDraw.line(faceCorners[1], faceCorners[2], color);
  debugDraw.line(faceCorners[2], faceCorners[3], color);
  debugDraw.line(faceCorners[3], faceCorners[0], color);
  const glm::mat4 inverseView = glm::inverse(view);
  const glm::vec3 cameraPos = glm::vec3(inverseView[3]);
  const glm::vec3 cameraRight =
      safeNormalize(glm::vec3(inverseView[0]), glm::vec3(1.0f, 0.0f, 0.0f));
  const glm::vec3 cameraUp =
      safeNormalize(glm::vec3(inverseView[1]), glm::vec3(0.0f, 1.0f, 0.0f));
  const glm::vec3 cameraForward =
      safeNormalize(-glm::vec3(inverseView[2]), glm::vec3(0.0f, 0.0f, -1.0f));
  const glm::vec3 markerCenter = cameraPos + cameraForward * 2.0f;
  constexpr int kMarkerHalfLines = 3;
  constexpr float kMarkerSpacing = 0.08f;
  const float markerExtent =
      kMarkerSpacing * static_cast<float>(kMarkerHalfLines);
  const glm::vec4 markerColor(0.1f, 1.0f, 1.0f, 1.0f);
  for (int i = -kMarkerHalfLines; i <= kMarkerHalfLines; ++i) {
    const float offset = static_cast<float>(i) * kMarkerSpacing;
    debugDraw.line(
        markerCenter + cameraRight * offset - cameraUp * markerExtent,
        markerCenter + cameraRight * offset + cameraUp * markerExtent,
        markerColor);
    debugDraw.line(
        markerCenter + cameraUp * offset - cameraRight * markerExtent,
        markerCenter + cameraUp * offset + cameraRight * markerExtent,
        markerColor);
  }
  const glm::vec3 unsnapped = dehomogenize(cascade.unsnappedCenter);
  if (glm::length(unsnapped - center) > 1.0e-5f) {
    debugDraw.line(unsnapped, center, color);
  }
  accumulateSortDepth(sortDepth, view, center);
  accumulateSortDepth(sortDepth, view, markerCenter);
}
void drawSelectedShadowLightRay(DebugDraw3D &debugDraw,
                                const RenderFrameContext &frame,
                                const ShadowCascadeDebugFrameData &cascade,
                                const glm::mat4 &view, float &sortDepth) {
  if (frame.scene == nullptr ||
      !frame.sharedResources.selectedShadowLightId.has_value() ||
      !isValid(*frame.sharedResources.selectedShadowLightId)) {
    return;
  }
  LightDesc light{};
  if (!frame.scene->graph().getCachedLightWorldDesc(
          *frame.sharedResources.selectedShadowLightId, light) ||
      light.type != LightType::Directional) {
    return;
  }
  const glm::vec3 center = dehomogenize(cascade.snappedCenter);
  const glm::vec3 direction =
      safeNormalize(light.rotation * glm::vec3(0.0f, 0.0f, -1.0f),
                    glm::vec3(0.0f, -1.0f, 0.0f));
  const glm::vec3 boundsExtent =
      glm::abs(glm::vec3(cascade.lightSpaceBoundsMax) -
               glm::vec3(cascade.lightSpaceBoundsMin));
  const float extent = std::max(glm::length(boundsExtent), 1.0f);
  const glm::vec3 start = center - direction * extent;
  const glm::vec3 end = center + direction * extent;
  debugDraw.line(start, end, kShadowLightRayColor);
  glm::vec3 right(1.0f, 0.0f, 0.0f);
  glm::vec3 up(0.0f, 1.0f, 0.0f);
  buildLightBasis(direction, right, up);
  const glm::vec3 headBase = end - direction * (extent * 0.08f);
  debugDraw.line(end, headBase + right * (extent * 0.04f),
                 kShadowLightRayColor);
  debugDraw.line(end, headBase - right * (extent * 0.04f),
                 kShadowLightRayColor);
  debugDraw.line(end, headBase + up * (extent * 0.04f), kShadowLightRayColor);
  debugDraw.line(end, headBase - up * (extent * 0.04f), kShadowLightRayColor);
  accumulateSortDepth(sortDepth, view, center);
}
bool drawShadowDebugOverlay(DebugDraw3D &debugDraw,
                            const RenderFrameContext &frame,
                            const glm::mat4 &view, float &sortDepth) {
  if (!shadowOverlayEnabled(frame) ||
      !frame.sharedResources.shadowDebugFrameData.has_value()) {
    return false;
  }
  const ShadowDebugFrameData &debugData =
      *frame.sharedResources.shadowDebugFrameData;
  if (debugData.cascadeCount == 0u) {
    return false;
  }
  const RenderSettings::ShadowDebugSettings &debug =
      frame.settings.shadow.debug;
  const uint32_t cascadeCount =
      std::min(debugData.cascadeCount, kMaxShadowCascades);
  const uint32_t selectedCascade =
      std::min(debug.debugCascadeIndex, cascadeCount - 1u);
  bool hasLines = false;
  if (debug.showCascadeFrusta) {
    for (uint32_t i = 0; i < cascadeCount; ++i) {
      drawShadowCascadeFrustum(debugDraw, debugData.cascades[i],
                               kShadowCascadeColors[i], view, sortDepth);
    }
    hasLines = true;
  }
  const ShadowCascadeDebugFrameData &selected =
      debugData.cascades[selectedCascade];
  if (debug.showLightViewBounds) {
    drawShadowLightBounds(debugDraw, selected, kShadowLightBoundsColor, view,
                          sortDepth);
    drawSelectedShadowLightRay(debugDraw, frame, selected, view, sortDepth);
    hasLines = true;
  }
  if (debug.showTexelGridSnap) {
    drawShadowTexelSnap(debugDraw, selected, kShadowTexelSnapColor, view,
                        sortDepth);
    hasLines = true;
  }
  return hasLines;
}

bool drawDDGIVolumeOverlays(DebugDraw3D &debugDraw,
                            const RenderFrameContext &frame,
                            const glm::mat4 &view, float &sortDepth) {
  if (frame.scene == nullptr) {
    return false;
  }
  const RenderSettings::DDGISettings &settings = frame.settings.ddgi;
  const DDGIVolumeId selected =
      frame.sharedResources.selectedDDGIVolumeId.value_or(kInvalidDDGIVolumeId);
  if (!settings.showVolumes && !isValid(selected)) {
    return false;
  }
  bool drew = false;
  for (const RenderDDGIVolume &volume : frame.scene->ddgiVolumes()) {
    const bool isSelected = isValid(selected) && volume.id == selected;
    if (!settings.showVolumes && !isSelected) {
      continue;
    }
    const glm::vec3 extents =
        0.5f * glm::vec3(volume.probeCounts - 1u) * volume.probeSpacing;
    glm::vec3 trackedCenter(0.0f);
    if (volume.mode == DDGIVolumeMode::CameraTracked) {
      const glm::vec3 cameraLocal = glm::vec3(
          glm::inverse(volume.worldFromLocal) * frame.camera.cameraPos);
      trackedCenter =
          glm::vec3(ddgiCameraCell(cameraLocal, volume.probeSpacing)) *
          volume.probeSpacing;
    }
    const glm::mat4 transform =
        volume.worldFromLocal * glm::translate(glm::mat4(1.0f), trackedCenter);
    debugDraw.box(transform, extents,
                  isSelected ? kSelectedDDGIVolumeColor : kDDGIVolumeColor);
    const glm::vec3 center = glm::vec3(transform[3]);
    sortDepth = std::max(sortDepth, -(view * glm::vec4(center, 1.0f)).z);
    drew = true;
    if (!isSelected) {
      continue;
    }
    const glm::vec3 axisX = glm::normalize(glm::vec3(transform[0]));
    const glm::vec3 axisY = glm::normalize(glm::vec3(transform[1]));
    const glm::vec3 axisZ = glm::normalize(glm::vec3(transform[2]));
    const float axisLength =
        std::max(0.25f, std::min({volume.probeSpacing.x, volume.probeSpacing.y,
                                  volume.probeSpacing.z}));
    debugDraw.line(center, center + axisX * axisLength,
                   glm::vec4(1.0f, 0.15f, 0.12f, 1.0f));
    debugDraw.line(center, center + axisY * axisLength,
                   glm::vec4(0.15f, 1.0f, 0.2f, 1.0f));
    debugDraw.line(center, center + axisZ * axisLength,
                   glm::vec4(0.2f, 0.45f, 1.0f, 1.0f));
    const int gridX = static_cast<int>(std::min(volume.probeCounts.x, 16u));
    const int gridY = static_cast<int>(std::min(volume.probeCounts.y, 16u));
    const int gridZ = static_cast<int>(std::min(volume.probeCounts.z, 16u));
    const glm::vec4 outline = kSelectedDDGIVolumeColor * 0.75f;
    debugDraw.plane(center, axisX, axisZ, gridX, gridZ, 2.0f * extents.x,
                    2.0f * extents.z, kDDGIGridColor, outline);
    debugDraw.plane(center, axisX, axisY, gridX, gridY, 2.0f * extents.x,
                    2.0f * extents.y, kDDGIGridColor, outline);
    debugDraw.plane(center, axisZ, axisY, gridZ, gridY, 2.0f * extents.z,
                    2.0f * extents.y, kDDGIGridColor, outline);
  }
  return drew;
}
} // namespace

DebugRenderer::DebugRenderer(GPUDevice &gpu, DebugRendererConfig config,
                             std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      debugDraw3D_(std::make_unique<DebugDraw3D>(gpu, memory_)),
      transparentSortableDraws_(memory_), transparentFixedDraws_(memory_),
      transparentDependencyBuffers_(memory_) {}

DebugRenderer::~DebugRenderer() = default;

Result<bool, std::string> DebugRenderer::createGridShaders() {
  if (nuri::isValid(gridVertexShader_) && nuri::isValid(gridFragmentShader_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (config_.grid.vertex.empty() || config_.grid.fragment.empty()) {
    return Result<bool, std::string>::makeError(
        "DebugRenderer::createGridShaders: vertex or fragment shader path is "
        "empty");
  }
  const std::string vertexShaderPath = config_.grid.vertex.string();
  auto vertexResult = compileShaderFile(gpu_, "debug_grid", vertexShaderPath,
                                        ShaderStage::Vertex);
  if (vertexResult.hasError()) {
    gridVertexShader_ = {};
    gridFragmentShader_ = {};
    return Result<bool, std::string>::makeError(vertexResult.error());
  }
  const std::string fragmentShaderPath = config_.grid.fragment.string();
  auto fragmentResult = compileShaderFile(
      gpu_, "debug_grid", fragmentShaderPath, ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    gridVertexShader_ = {};
    gridFragmentShader_ = {};
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }
  gridVertexShader_ = vertexResult.value();
  gridFragmentShader_ = fragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugRenderer::ensureGridPipeline(Format colorFormat, Format depthFormat) {
  auto shaderResult = createGridShaders();
  if (shaderResult.hasError()) {
    return shaderResult;
  }
  if (gridPipeline_.valid() && gridPipelineColorFormat_ == colorFormat &&
      gridPipelineDepthFormat_ == depthFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  gridPipeline_.reset();
  const RenderPipelineDesc desc{
      .vertexInput = {},
      .vertexShader = gridVertexShader_,
      .fragmentShader = gridFragmentShader_,
      .colorFormats = {colorFormat},
      .depthFormat = depthFormat,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = true,
      .rasterState = depthFormat != Format::Count
                         ? makeRasterPipelineState(
                               DepthState{.compareOp = CompareOp::LessEqual,
                                          .isDepthWriteEnabled = false})
                         : RasterPipelineState{},
  };
  auto pipelineResult = gpu_.createRenderPipeline(desc, kGridPipelineName);
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  gridPipeline_.reset(gpu_, pipelineResult.value());
  gridPipelineColorFormat_ = colorFormat;
  gridPipelineDepthFormat_ = depthFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugRenderer::ensureDDGIProbePipeline(Format colorFormat, Format depthFormat) {
  if (!nuri::isValid(ddgiProbeVertexShader_) ||
      !nuri::isValid(ddgiProbeFragmentShader_)) {
    if (config_.ddgi.probeDebugVertex.empty() ||
        config_.ddgi.probeDebugFragment.empty()) {
      return Result<bool, std::string>::makeError(
          "DebugRenderer: DDGI probe debug shader path is empty");
    }
    auto vertex = compileShaderFile(gpu_, "ddgi_probe_debug",
                                    config_.ddgi.probeDebugVertex.string(),
                                    ShaderStage::Vertex);
    if (vertex.hasError()) {
      return Result<bool, std::string>::makeError(vertex.error());
    }
    auto fragment = compileShaderFile(gpu_, "ddgi_probe_debug",
                                      config_.ddgi.probeDebugFragment.string(),
                                      ShaderStage::Fragment);
    if (fragment.hasError()) {
      return Result<bool, std::string>::makeError(fragment.error());
    }
    ddgiProbeVertexShader_ = vertex.value();
    ddgiProbeFragmentShader_ = fragment.value();
  }
  if (ddgiProbePipeline_.valid() &&
      ddgiProbePipelineColorFormat_ == colorFormat &&
      ddgiProbePipelineDepthFormat_ == depthFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  ddgiProbePipeline_.reset();
  const RenderPipelineDesc desc{
      .vertexInput = {},
      .vertexShader = ddgiProbeVertexShader_,
      .fragmentShader = ddgiProbeFragmentShader_,
      .colorFormats = {colorFormat},
      .depthFormat = depthFormat,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = true,
      .rasterState = depthFormat != Format::Count
                         ? makeRasterPipelineState(
                               DepthState{.compareOp = CompareOp::LessEqual,
                                          .isDepthWriteEnabled = false})
                         : RasterPipelineState{},
  };
  auto pipeline = gpu_.createRenderPipeline(desc, "DDGI Probe Debug Pipeline");
  if (pipeline.hasError()) {
    return Result<bool, std::string>::makeError(pipeline.error());
  }
  ddgiProbePipeline_.reset(gpu_, pipeline.value());
  ddgiProbePipelineColorFormat_ = colorFormat;
  ddgiProbePipelineDepthFormat_ = depthFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugRenderer::prepareDDGIProbeDraws(const RenderFrameContext &frame,
                                     TextureHandle depthTexture) {
  ddgiProbeDrawCount_ = 0u;
  if (!frame.settings.ddgi.showProbes ||
      !frame.sharedResources.ddgiFrameGpuData.has_value()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const DDGIFrameGpuDataHandle &ddgi = *frame.sharedResources.ddgiFrameGpuData;
  if (!nuri::isValid(ddgi.buffer) || ddgi.bufferAddress == 0u ||
      ddgi.activeVolumeCount == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }
  const Format depthFormat = nuri::isValid(depthTexture)
                                 ? gpu_.getTextureFormat(depthTexture)
                                 : Format::Count;
  const TextureHandle colorTexture =
      frame.sharedResources[FrameTextureSlot::FrameColor].texture;
  const Format colorFormat = nuri::isValid(colorTexture)
                                 ? gpu_.getTextureFormat(colorTexture)
                                 : gpu_.getSwapchainFormat();
  auto pipeline = ensureDDGIProbePipeline(colorFormat, depthFormat);
  if (pipeline.hasError()) {
    return pipeline;
  }
  const glm::mat4 worldFromView = glm::inverse(frame.camera.view);
  const glm::vec3 cameraRight = glm::normalize(glm::vec3(worldFromView[0]));
  const glm::vec3 cameraUp = glm::normalize(glm::vec3(worldFromView[1]));
  ddgiProbeDrawCount_ = std::min(ddgi.activeVolumeCount, kMaxDDGIVolumes);
  for (uint32_t slot = 0u; slot < ddgiProbeDrawCount_; ++slot) {
    const float size =
        std::clamp(ddgi.minimumProbeSpacing[slot] * 0.075f, 0.025f, 1.0f);
    ddgiProbePushConstants_[slot] = DDGIProbePushConstants{
        .viewProjection = cameraCurrentUnjitteredViewProjection(frame.camera),
        .cameraRightAndScale = glm::vec4(cameraRight, size),
        .cameraUpAndDebugView =
            glm::vec4(cameraUp, static_cast<float>(ddgi.debugView)),
        .frameAddress = ddgi.bufferAddress,
        .volumeSlot = slot,
        .submittedSequence =
            static_cast<uint32_t>(frame.metrics.ddgi.submittedSequence),
    };
    DrawItem &draw = ddgiProbeDrawItems_[slot];
    draw = DrawItem{};
    draw.pipeline = ddgiProbePipeline_.get();
    draw.vertexCount = 6u;
    draw.instanceCount = ddgi.probeCounts[slot];
    draw.pushConstants = std::as_bytes(
        std::span(&ddgiProbePushConstants_[slot], static_cast<size_t>(1u)));
    draw.debugLabel = "DDGI Probe Debug Draw";
    draw.debugColor = 0xff44ccffu;
    if (nuri::isValid(depthTexture)) {
      draw.useDepthState = true;
      draw.depthState = {.compareOp = CompareOp::LessEqual,
                         .isDepthWriteEnabled = false};
    }
  }
  for (BufferHandle buffer : ddgi.dependencyBuffers) {
    if (nuri::isValid(buffer) &&
        std::ranges::find(transparentDependencyBuffers_, buffer) ==
            transparentDependencyBuffers_.end()) {
      transparentDependencyBuffers_.push_back(buffer);
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugRenderer::ensureDDGIRayPipeline(Format colorFormat, Format depthFormat) {
  if (!nuri::isValid(ddgiRayVertexShader_) ||
      !nuri::isValid(ddgiRayFragmentShader_)) {
    if (config_.ddgi.rayDebugVertex.empty() ||
        config_.ddgi.rayDebugFragment.empty()) {
      return Result<bool, std::string>::makeError(
          "DebugRenderer: DDGI ray debug shader path is empty");
    }
    auto vertex = compileShaderFile(gpu_, "ddgi_ray_debug",
                                    config_.ddgi.rayDebugVertex.string(),
                                    ShaderStage::Vertex);
    if (vertex.hasError()) {
      return Result<bool, std::string>::makeError(vertex.error());
    }
    auto fragment = compileShaderFile(gpu_, "ddgi_ray_debug",
                                      config_.ddgi.rayDebugFragment.string(),
                                      ShaderStage::Fragment);
    if (fragment.hasError()) {
      return Result<bool, std::string>::makeError(fragment.error());
    }
    ddgiRayVertexShader_ = vertex.value();
    ddgiRayFragmentShader_ = fragment.value();
  }
  if (ddgiRayPipeline_.valid() && ddgiRayPipelineColorFormat_ == colorFormat &&
      ddgiRayPipelineDepthFormat_ == depthFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  ddgiRayPipeline_.reset();
  const RenderPipelineDesc desc{
      .vertexInput = {},
      .vertexShader = ddgiRayVertexShader_,
      .fragmentShader = ddgiRayFragmentShader_,
      .colorFormats = {colorFormat},
      .depthFormat = depthFormat,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Line,
      .blendEnabled = true,
      .rasterState = depthFormat != Format::Count
                         ? makeRasterPipelineState(
                               DepthState{.compareOp = CompareOp::LessEqual,
                                          .isDepthWriteEnabled = false})
                         : RasterPipelineState{},
  };
  auto pipeline = gpu_.createRenderPipeline(desc, "DDGI Ray Debug Pipeline");
  if (pipeline.hasError()) {
    return Result<bool, std::string>::makeError(pipeline.error());
  }
  ddgiRayPipeline_.reset(gpu_, pipeline.value());
  ddgiRayPipelineColorFormat_ = colorFormat;
  ddgiRayPipelineDepthFormat_ = depthFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugRenderer::prepareDDGIRayDraw(const RenderFrameContext &frame,
                                  TextureHandle depthTexture) {
  ddgiRayDrawItem_ = {};
  if (!frame.settings.ddgi.showSelectedProbeRays ||
      !frame.sharedResources.ddgiFrameGpuData.has_value()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const DDGIFrameGpuDataHandle &ddgi = *frame.sharedResources.ddgiFrameGpuData;
  if (!nuri::isValid(ddgi.diagnosticBuffer) ||
      ddgi.diagnosticRayAddress == 0u || ddgi.diagnosticRayCount == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }
  const Format depthFormat = nuri::isValid(depthTexture)
                                 ? gpu_.getTextureFormat(depthTexture)
                                 : Format::Count;
  const TextureHandle colorTexture =
      frame.sharedResources[FrameTextureSlot::FrameColor].texture;
  const Format colorFormat = nuri::isValid(colorTexture)
                                 ? gpu_.getTextureFormat(colorTexture)
                                 : gpu_.getSwapchainFormat();
  auto pipeline = ensureDDGIRayPipeline(colorFormat, depthFormat);
  if (pipeline.hasError()) {
    return pipeline;
  }
  ddgiRayPushConstants_ = DDGIRayPushConstants{
      .viewProjection = cameraCurrentUnjitteredViewProjection(frame.camera),
      .diagnosticAddress = ddgi.diagnosticRayAddress,
      .rayCount = ddgi.diagnosticRayCount,
  };
  ddgiRayDrawItem_.pipeline = ddgiRayPipeline_.get();
  ddgiRayDrawItem_.vertexCount = ddgi.diagnosticRayCount * 2u;
  ddgiRayDrawItem_.instanceCount = 1u;
  ddgiRayDrawItem_.pushConstants =
      std::as_bytes(std::span(&ddgiRayPushConstants_, static_cast<size_t>(1u)));
  ddgiRayDrawItem_.debugLabel = "DDGI Selected Probe Rays";
  ddgiRayDrawItem_.debugColor = 0xffff55aau;
  if (nuri::isValid(depthTexture)) {
    ddgiRayDrawItem_.useDepthState = true;
    ddgiRayDrawItem_.depthState = {.compareOp = CompareOp::LessEqual,
                                   .isDepthWriteEnabled = false};
  }
  if (std::ranges::find(transparentDependencyBuffers_, ddgi.diagnosticBuffer) ==
      transparentDependencyBuffers_.end()) {
    transparentDependencyBuffers_.push_back(ddgi.diagnosticBuffer);
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugRenderer::prepareGridDraw(const RenderFrameContext &frame,
                               TextureHandle depthTexture) {
  const bool hasDepth = nuri::isValid(depthTexture);
  const Format depthFormat =
      hasDepth ? gpu_.getTextureFormat(depthTexture) : Format::Count;
  const TextureHandle colorTexture =
      frame.sharedResources[FrameTextureSlot::FrameColor].texture;
  const Format colorFormat = nuri::isValid(colorTexture)
                                 ? gpu_.getTextureFormat(colorTexture)
                                 : gpu_.getSwapchainFormat();
  auto pipelineResult = ensureGridPipeline(colorFormat, depthFormat);
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  gridPushConstants_ = GridPushConstants{
      .mvp = cameraCurrentUnjitteredViewProjection(frame.camera),
      .cameraPos = frame.camera.cameraPos,
      .origin = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f),
  };
  gridDrawItem_ = DrawItem{};
  gridDrawItem_.pipeline = gridPipeline_.get();
  gridDrawItem_.vertexCount = kGridVertexCount;
  gridDrawItem_.instanceCount = 1;
  gridDrawItem_.pushConstants = std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(&gridPushConstants_),
      sizeof(gridPushConstants_));
  gridDrawItem_.debugLabel = kGridDrawLabel;
  gridDrawItem_.debugColor = kGridDrawDebugColor;
  if (hasDepth) {
    gridDrawItem_.useDepthState = true;
    gridDrawItem_.depthState = {.compareOp = CompareOp::LessEqual,
                                .isDepthWriteEnabled = false};
  }
  return Result<bool, std::string>::makeResult(true);
}

bool DebugRenderer::hasDebugWork(const RenderFrameContext &frame) const {
  const RenderSettings &settings = frame.settings;
  const RenderSettings::DebugSettings &debug = settings.debug;
  const VisibilityDebugSettings &visibilityDebug = settings.visibility.debug;
  return debug.enabled || debug.modelBounds || debug.grid || debug.lightIcons ||
         visibilityDebug.showObjectBounds ||
         visibilityDebug.showMeshletBounds || shadowOverlayEnabled(frame) ||
         settings.ddgi.showVolumes || settings.ddgi.showProbes ||
         settings.ddgi.showSelectedProbeRays ||
         frame.sharedResources.selectedDDGIVolumeId.has_value();
}

bool DebugRenderer::buildSceneDebugLines(const RenderFrameContext &frame,
                                         TextureHandle depthTexture,
                                         float &outSortDepth) {
  outSortDepth = 0.0f;
  if (!nuri::isValid(depthTexture)) {
    return false;
  }
  debugDraw3D_->clear();
  debugDraw3D_->setMatrix(cameraCurrentUnjitteredViewProjection(frame.camera));
  const LightId selectedLightId = resolveSelectedLightId(frame);
  bool hasLines = false;
  const glm::mat4 view = frame.camera.view;
  const RenderSettings &settings = frame.settings;
  if (frame.scene != nullptr && frame.resources != nullptr &&
      (settings.debug.modelBounds ||
       settings.visibility.debug.showObjectBounds ||
       settings.visibility.debug.showMeshletBounds)) {
    const std::span<const Renderable> renderables = frame.scene->renderables();
    const bool drawVisibilityBounds =
        settings.visibility.debug.showObjectBounds;
    const bool drawMeshletBounds = settings.visibility.debug.showMeshletBounds;
    const visibility_detail::FrustumPlanes frustum =
        visibility_detail::buildCameraFrustumPlanes(frame.camera);
    for (const Renderable &renderable : renderables) {
      const ModelRecord *modelRecord =
          frame.resources->tryGet(renderable.model);
      if (!modelRecord || !modelRecord->model) {
        continue;
      }
      const BoundingBox &bounds = modelRecord->model->bounds();
      if (settings.debug.modelBounds || drawVisibilityBounds) {
        const glm::vec4 color =
            drawVisibilityBounds
                ? visibilityBoundsColor(frame, renderable, bounds, frustum)
                : kModelBoundsColor;
        debugDraw3D_->box(renderable.modelMatrix, bounds, color);
        const glm::vec3 center = glm::vec3(renderable.modelMatrix *
                                           glm::vec4(bounds.getCenter(), 1.0f));
        outSortDepth =
            std::max(outSortDepth, -(view * glm::vec4(center, 1.0f)).z);
        hasLines = true;
      }
      if (drawMeshletBounds) {
        for (const glm::vec4 &boundsSphere :
             modelRecord->model->meshletBoundsSpheres()) {
          const float radius = std::isfinite(boundsSphere.w)
                                   ? std::max(boundsSphere.w, 0.0f)
                                   : 0.0f;
          if (radius <= 0.0f) {
            continue;
          }
          glm::mat4 meshletTransform = renderable.modelMatrix;
          meshletTransform[3] =
              renderable.modelMatrix * glm::vec4(glm::vec3(boundsSphere), 1.0f);
          const glm::vec4 color = visibilityMeshletBoundsColor(
              frame, renderable, boundsSphere, frustum);
          debugDraw3D_->box(meshletTransform, glm::vec3(radius), color);
          const glm::vec3 center = glm::vec3(meshletTransform[3]);
          outSortDepth =
              std::max(outSortDepth, -(view * glm::vec4(center, 1.0f)).z);
          hasLines = true;
        }
      }
    }
  }
  if (frame.scene != nullptr && settings.debug.lightIcons) {
    frame.scene->forEachLightId([&](LightId lightId) {
      LightDesc light{};
      if (!frame.scene->graph().getCachedLightWorldDesc(lightId, light)) {
        return;
      }
      const float scale = lightIconScale(frame.camera, light.position);
      const bool selected =
          isValid(selectedLightId) && selectedLightId == lightId;
      const glm::vec4 color =
          selected
              ? kSelectedLightIconColor
              : (light.type == LightType::Directional
                     ? kDirectionalLightIconColor
                     : (light.type == LightType::Point ? kPointLightIconColor
                                                       : kSpotLightIconColor));
      const glm::vec3 direction = light.rotation * glm::vec3(0.0f, 0.0f, -1.0f);
      switch (light.type) {
      case LightType::Directional:
        drawDirectionalLightIcon(*debugDraw3D_, light.position, direction,
                                 scale, color);
        break;
      case LightType::Point:
        drawPointLightIcon(*debugDraw3D_, light.position, scale, color);
        break;
      case LightType::Spot:
        drawSpotLightIcon(*debugDraw3D_, light.position, direction, scale,
                          color);
        break;
      }
      outSortDepth =
          std::max(outSortDepth, -(view * glm::vec4(light.position, 1.0f)).z);
      hasLines = true;
    });
  }
  if (drawShadowDebugOverlay(*debugDraw3D_, frame, view, outSortDepth)) {
    hasLines = true;
  }
  if (drawDDGIVolumeOverlays(*debugDraw3D_, frame, view, outSortDepth)) {
    hasLines = true;
  }
  return hasLines;
}

Result<bool, std::string> DebugRenderer::buildTransparentStageContribution(
    RenderFrameContext &frame, TransparentStageContribution &out) {
  NURI_PROFILER_FUNCTION();
  out = {};
  transparentSortableDraws_.clear();
  transparentFixedDraws_.clear();
  transparentDependencyBuffers_.clear();
  if (!hasDebugWork(frame)) {
    return Result<bool, std::string>::makeResult(true);
  }
  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  if (nuri::isValid(depthTexture)) {
    float debugSortDepth = 0.0f;
    const bool built =
        buildSceneDebugLines(frame, depthTexture, debugSortDepth);
    if (built) {
      const TextureHandle frameColor =
          frame.sharedResources[FrameTextureSlot::FrameColor].texture;
      const Format targetColorFormat = nuri::isValid(frameColor)
                                           ? gpu_.getTextureFormat(frameColor)
                                           : gpu_.getSwapchainFormat();
      DrawItem lineDraw{};
      BufferHandle lineBuffer{};
      auto lineResult = debugDraw3D_->prepareDraw(
          frame.frameIndex, depthTexture, targetColorFormat,
          !shadowTexelGridSnapEnabled(frame), lineDraw, lineBuffer);
      if (lineResult.hasError()) {
        return Result<bool, std::string>::makeError(lineResult.error());
      }
      if (lineResult.value()) {
        transparentSortableDraws_.push_back(TransparentStageSortableDraw{
            .draw = lineDraw,
            .sortDepth = debugSortDepth,
            .stableOrder = 0u,
        });
        transparentDependencyBuffers_.push_back(lineBuffer);
      }
    }
  }
  if (frame.settings.debug.grid) {
    auto gridResult = prepareGridDraw(frame, depthTexture);
    if (gridResult.hasError()) {
      return gridResult;
    }
    transparentFixedDraws_.push_back(gridDrawItem_);
  }
  auto probeResult = prepareDDGIProbeDraws(frame, depthTexture);
  if (probeResult.hasError()) {
    return probeResult;
  }
  for (uint32_t slot = 0u; slot < ddgiProbeDrawCount_; ++slot) {
    if (ddgiProbeDrawItems_[slot].instanceCount != 0u) {
      transparentFixedDraws_.push_back(ddgiProbeDrawItems_[slot]);
    }
  }
  auto rayResult = prepareDDGIRayDraw(frame, depthTexture);
  if (rayResult.hasError()) {
    return rayResult;
  }
  if (ddgiRayDrawItem_.vertexCount != 0u) {
    transparentFixedDraws_.push_back(ddgiRayDrawItem_);
  }
  out.sortableDraws = std::span<const TransparentStageSortableDraw>(
      transparentSortableDraws_.data(), transparentSortableDraws_.size());
  out.fixedDraws = std::span<const DrawItem>(transparentFixedDraws_.data(),
                                             transparentFixedDraws_.size());
  out.dependencyBuffers =
      std::span<const BufferHandle>(transparentDependencyBuffers_.data(),
                                    transparentDependencyBuffers_.size());
  out.textureReads = {};
  return Result<bool, std::string>::makeResult(true);
}

void DebugRenderer::onFrameSubmitted(SubmissionHandle submission) noexcept {
  debugDraw3D_->onFrameSubmitted(submission);
}

void DebugRenderer::onFrameAbandoned() noexcept {
  debugDraw3D_->onFrameAbandoned();
}

void registerDebugStages(RenderPipeline &pipeline, GPUDevice &gpu,
                         DebugRendererConfig config,
                         std::pmr::memory_resource *memory) {
  pipeline.addComponent(
      std::make_unique<DebugRenderer>(gpu, std::move(config), memory),
      PipelineComponentDesc{
          .publish =
              [](void *state, FrameBuildContext &ctx) {
                ctx.frame.transparentContributors.publish(
                    TransparentContributionCollector{
                        .user = state,
                        .collect =
                            [](void *user, RenderFrameContext &frame,
                               TransparentStageContribution &out) {
                              return static_cast<DebugRenderer *>(user)
                                  ->buildTransparentStageContribution(frame,
                                                                      out);
                            },
                    });
                return Result<bool, std::string>::makeResult(true);
              },
          .submitted =
              [](void *state, const RenderFrameContext &frame) noexcept {
                static_cast<DebugRenderer *>(state)->onFrameSubmitted(
                    frame.submission);
              },
          .abandoned =
              [](void *state, const RenderFrameContext &) noexcept {
                static_cast<DebugRenderer *>(state)->onFrameAbandoned();
              },
      });
}

} // namespace nuri
