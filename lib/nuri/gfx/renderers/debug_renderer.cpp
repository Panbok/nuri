#include "nuri/gfx/renderers/debug_renderer.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/debug_draw_3d.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"

namespace nuri {
namespace {

constexpr uint32_t kGridPassDebugColor = 0xff66aaff;
constexpr uint32_t kGridDrawDebugColor = 0xff66aaff;
constexpr uint32_t kLightIconPassDebugColor = 0xfff0c040;
constexpr uint32_t kGridVertexCount = 6;
constexpr std::string_view kGridPipelineName = "debug_grid";
constexpr std::string_view kGridPassLabel = "DebugGrid Pass";
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

[[nodiscard]] bool isSameTextureHandle(TextureHandle a, TextureHandle b) {
  return a.index == b.index && a.generation == b.generation;
}

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
  if (frame.settings == nullptr || !frame.settings->shadow.enabled) {
    return false;
  }
  const RenderSettings::ShadowDebugSettings &debug =
      frame.settings->shadow.debug;
  return debug.showCascadeFrusta || debug.showLightViewBounds ||
         debug.showTexelGridSnap;
}

[[nodiscard]] bool shadowTexelGridSnapEnabled(const RenderFrameContext &frame) {
  return frame.settings != nullptr && frame.settings->shadow.enabled &&
         frame.settings->shadow.debug.showTexelGridSnap;
}

void accumulateSortDepth(float &sortDepth, const glm::mat4 &view,
                         const glm::vec3 &position) {
  sortDepth = std::max(sortDepth, -(view * glm::vec4(position, 1.0f)).z);
}

[[nodiscard]] glm::vec3 dehomogenize(const glm::vec4 &value) {
  // dehomogenize intentionally skips division when value.w is near zero to
  // avoid divide-by-zero and treat it as a direction/point at infinity.
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

uint32_t drawShadowTexelSnap(DebugDraw3D &debugDraw,
                             const ShadowCascadeDebugFrameData &cascade,
                             const glm::vec4 &color, const glm::mat4 &view,
                             float &sortDepth) {
  uint32_t lineCount = 0u;
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
      ++lineCount;
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
      ++lineCount;
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
  lineCount += 4u;

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
    lineCount += 2u;
  }

  const glm::vec3 unsnapped = dehomogenize(cascade.unsnappedCenter);
  if (glm::length(unsnapped - center) > 1.0e-5f) {
    debugDraw.line(unsnapped, center, color);
    ++lineCount;
  }
  accumulateSortDepth(sortDepth, view, center);
  accumulateSortDepth(sortDepth, view, markerCenter);
  return lineCount;
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
      frame.settings->shadow.debug;
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
    const uint32_t snapLineCount = drawShadowTexelSnap(
        debugDraw, selected, kShadowTexelSnapColor, view, sortDepth);
    static std::atomic<uint32_t> loggedTexelSnapOverlays{0u};
    if (loggedTexelSnapOverlays.fetch_add(1u, std::memory_order_relaxed) <
        16u) {
      NURI_LOG_DEBUG(
          "DebugRenderer::drawShadowDebugOverlay texel grid: requested=%u "
          "effective=%u cascadeCount=%u lines=%u texelWorldSize=%.6f",
          debug.debugCascadeIndex, selectedCascade, cascadeCount, snapLineCount,
          selected.texelWorldSize);
    }
    hasLines = true;
  }

  return hasLines;
}

} // namespace

DebugRenderer::DebugRenderer(GPUDevice &gpu, DebugRendererConfig config,
                             std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      debugDraw3D_(std::make_unique<DebugDraw3D>(gpu, memory_)),
      transparentSortableDraws_(memory_), transparentFixedDraws_(memory_),
      transparentDependencyBuffers_(memory_) {}

DebugRenderer::~DebugRenderer() { onDetach(); }

void DebugRenderer::onDetach() {
  debugDraw3D_.reset();
  resetGridState();
  preparedSceneDepthTexture_ = {};
  preparedFrameColorTexture_ = {};
  preparedSceneDepthGraphTexture_ = {};
  preparedHasPriorColorPass_ = false;
  preparedGridPass_ = false;
  preparedSceneOverlayPass_ = false;
  preparedSceneOverlayDepthTest_ = true;
  transparentSortableDraws_.clear();
  transparentFixedDraws_.clear();
  transparentDependencyBuffers_.clear();
}

Result<bool, std::string> DebugRenderer::ensureGridInitialized() {
  return createGridShaders();
}

Result<bool, std::string> DebugRenderer::createGridShaders() {
  if (gridShader_ && nuri::isValid(gridVertexShader_) &&
      nuri::isValid(gridFragmentShader_)) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (config_.vertex.empty() || config_.fragment.empty()) {
    return Result<bool, std::string>::makeError(
        "DebugRenderer::createGridShaders: vertex or fragment shader path is "
        "empty");
  }

  gridShader_ = Shader::create("debug_grid", gpu_);
  if (!gridShader_) {
    return Result<bool, std::string>::makeError(
        "DebugRenderer::createGridShaders: failed to create grid shader "
        "wrapper");
  }

  const std::string vertexShaderPath = config_.vertex.string();
  auto vertexResult =
      gridShader_->compileFromFile(vertexShaderPath, ShaderStage::Vertex);
  if (vertexResult.hasError()) {
    gridVertexShader_ = {};
    gridFragmentShader_ = {};
    gridShader_.reset();
    return Result<bool, std::string>::makeError(vertexResult.error());
  }
  const std::string fragmentShaderPath = config_.fragment.string();
  auto fragmentResult =
      gridShader_->compileFromFile(fragmentShaderPath, ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    gridVertexShader_ = {};
    gridFragmentShader_ = {};
    gridShader_.reset();
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }

  gridVertexShader_ = vertexResult.value();
  gridFragmentShader_ = fragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugRenderer::ensureGridPipeline(Format colorFormat, Format depthFormat) {
  auto shaderResult = ensureGridInitialized();
  if (shaderResult.hasError()) {
    return shaderResult;
  }

  if (nuri::isValid(gridPipelineHandle_) &&
      gridPipelineColorFormat_ == colorFormat &&
      gridPipelineDepthFormat_ == depthFormat) {
    return Result<bool, std::string>::makeResult(true);
  }

  gridPipeline_.reset();
  gridPipelineHandle_ = {};

  gridPipeline_ = Pipeline::create(gpu_);
  if (!gridPipeline_) {
    return Result<bool, std::string>::makeError(
        "DebugRenderer::ensureGridPipeline: failed to create grid pipeline "
        "wrapper");
  }

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
  };

  auto pipelineResult =
      gridPipeline_->createRenderPipeline(desc, kGridPipelineName);
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }

  gridPipelineHandle_ = pipelineResult.value();
  gridPipelineColorFormat_ = colorFormat;
  gridPipelineDepthFormat_ = depthFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugRenderer::prepareGridDraw(const RenderFrameContext &frame,
                               TextureHandle depthTexture) {
  const bool hasDepth = nuri::isValid(depthTexture);
  const Format depthFormat =
      hasDepth ? gpu_.getTextureFormat(depthTexture) : Format::Count;
  TextureHandle colorTexture = preparedFrameColorTexture_;
  if (!nuri::isValid(colorTexture)) {
    colorTexture = resolveFrameColorTexture(frame);
  }
  const Format colorFormat = nuri::isValid(colorTexture)
                                 ? gpu_.getTextureFormat(colorTexture)
                                 : gpu_.getSwapchainFormat();
  auto pipelineResult = ensureGridPipeline(colorFormat, depthFormat);
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }

  gridPushConstants_ = GridPushConstants{
      .mvp = frame.camera.proj * frame.camera.view,
      .cameraPos = frame.camera.cameraPos,
      .origin = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f),
  };

  gridDrawItem_ = DrawItem{};
  gridDrawItem_.pipeline = gridPipelineHandle_;
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

void DebugRenderer::resetGridState() {
  gridPipeline_.reset();
  gridShader_.reset();

  gridVertexShader_ = {};
  gridFragmentShader_ = {};
  gridPipelineHandle_ = {};
  gridPipelineColorFormat_ = Format::Count;
  gridPipelineDepthFormat_ = Format::Count;

  gridPushConstants_ = GridPushConstants{};
  gridDrawItem_ = DrawItem{};
}

Result<bool, std::string> DebugRenderer::appendModelBoundsGraphPass(
    const RenderFrameContext &frame, RenderGraphBuilder &graph,
    TextureHandle sceneDepthTexture,
    RenderGraphTextureId sceneDepthGraphTexture) {
  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  if (!debugDraw3D_ || !frame.scene || !nuri::isValid(depthTexture) ||
      !frame.resources) {
    return Result<bool, std::string>::makeResult(true);
  }

  const std::span<const Renderable> renderables = frame.scene->renderables();
  if (renderables.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  debugDraw3D_->clear();
  debugDraw3D_->setMatrix(frame.camera.proj * frame.camera.view);
  for (const Renderable &renderable : renderables) {
    const ModelRecord *modelRecord = frame.resources->tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      continue;
    }
    debugDraw3D_->box(renderable.modelMatrix, modelRecord->model->bounds(),
                      glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
  }

  auto linePassResult =
      debugDraw3D_->buildGraphPass(frame.frameIndex, depthTexture);
  if (linePassResult.hasError()) {
    return Result<bool, std::string>::makeError(linePassResult.error());
  }

  DebugDraw3D::PreparedGraphPass pass = linePassResult.value();
  TextureHandle colorTexture = resolveFrameColorTexture(frame);
  if (!nuri::isValid(colorTexture)) {
    colorTexture = pass.colorTextureHandle;
  }
  if (nuri::isValid(colorTexture)) {
    auto colorImportResult =
        graph.importTexture(colorTexture, "debug_pass_color_texture");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    pass.desc.colorTexture = colorImportResult.value();
  }
  if (nuri::isValid(pass.depthTextureHandle)) {
    const bool useDepthOverride =
        nuri::isValid(sceneDepthTexture) &&
        nuri::isValid(sceneDepthGraphTexture) &&
        isSameTextureHandle(pass.depthTextureHandle, sceneDepthTexture);
    if (useDepthOverride) {
      pass.desc.depthTexture = sceneDepthGraphTexture;
    } else {
      auto depthImportResult = graph.importTexture(pass.depthTextureHandle,
                                                   "debug_pass_depth_texture");
      if (depthImportResult.hasError()) {
        return Result<bool, std::string>::makeError(depthImportResult.error());
      }
      pass.desc.depthTexture = depthImportResult.value();
    }
  }

  auto addResult = graph.addGraphicsPass(pass.desc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }

  return Result<bool, std::string>::makeResult(true);
}

bool DebugRenderer::hasDebugWork(const RenderFrameContext &frame) const {
  if (frame.settings == nullptr) {
    return false;
  }
  const RenderSettings::DebugSettings &debug = frame.settings->debug;
  return debug.enabled || debug.modelBounds || debug.grid || debug.lightIcons ||
         shadowOverlayEnabled(frame);
}

Result<bool, std::string>
DebugRenderer::buildSceneDebugLines(const RenderFrameContext &frame,
                                    TextureHandle depthTexture,
                                    float &outSortDepth) {
  outSortDepth = 0.0f;
  if (!debugDraw3D_ || !nuri::isValid(depthTexture)) {
    static std::atomic<uint32_t> loggedSceneDebugEarlyOuts{0u};
    if (loggedSceneDebugEarlyOuts.fetch_add(1u, std::memory_order_relaxed) <
        16u) {
      NURI_LOG_DEBUG(
          "DebugRenderer::buildSceneDebugLines early-out: frame=%llu "
          "debugDraw=%u scene=%u depthValid=%u",
          static_cast<unsigned long long>(frame.frameIndex),
          debugDraw3D_ != nullptr ? 1u : 0u, frame.scene != nullptr ? 1u : 0u,
          nuri::isValid(depthTexture) ? 1u : 0u);
    }
    return Result<bool, std::string>::makeResult(false);
  }

  debugDraw3D_->clear();
  debugDraw3D_->setMatrix(frame.camera.proj * frame.camera.view);

  const LightId selectedLightId = resolveSelectedLightId(frame);
  bool hasLines = false;
  const glm::mat4 view = frame.camera.view;

  if (frame.scene != nullptr && frame.settings != nullptr &&
      frame.settings->debug.modelBounds && frame.resources != nullptr) {
    const std::span<const Renderable> renderables = frame.scene->renderables();
    for (const Renderable &renderable : renderables) {
      const ModelRecord *modelRecord =
          frame.resources->tryGet(renderable.model);
      if (!modelRecord || !modelRecord->model) {
        continue;
      }
      debugDraw3D_->box(renderable.modelMatrix, modelRecord->model->bounds(),
                        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
      const glm::vec3 center =
          glm::vec3(renderable.modelMatrix *
                    glm::vec4(modelRecord->model->bounds().getCenter(), 1.0f));
      outSortDepth =
          std::max(outSortDepth, -(view * glm::vec4(center, 1.0f)).z);
      hasLines = true;
    }
  }

  if (frame.scene != nullptr && frame.settings != nullptr &&
      frame.settings->debug.lightIcons) {
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

  return Result<bool, std::string>::makeResult(hasLines);
}

Result<bool, std::string>
DebugRenderer::prepareDebugPasses(RenderFrameContext &frame) {
  preparedSceneDepthTexture_ = {};
  preparedFrameColorTexture_ = {};
  preparedSceneDepthGraphTexture_ = {};
  preparedHasPriorColorPass_ = false;
  preparedGridPass_ = false;
  preparedSceneOverlayPass_ = false;

  if (frame.sharedResources.transparentStageEnabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (!hasDebugWork(frame)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const TextureHandle sceneDepthTexture = resolveFrameDepthTexture(frame);
  preparedSceneDepthGraphTexture_ = resolveSceneDepthGraphTexture(frame);
  TextureHandle frameColorTexture = resolveFrameColorTexture(frame);
  preparedSceneDepthTexture_ = sceneDepthTexture;
  preparedFrameColorTexture_ = frameColorTexture;
  preparedHasPriorColorPass_ = nuri::isValid(frameColorTexture);

  if (frame.settings->debug.grid) {
    auto gridResult = prepareGridDraw(frame, sceneDepthTexture);
    if (gridResult.hasError()) {
      return gridResult;
    }
    preparedGridPass_ = true;
  }

  float debugSortDepth = 0.0f;
  auto buildLinesResult =
      buildSceneDebugLines(frame, sceneDepthTexture, debugSortDepth);
  if (buildLinesResult.hasError()) {
    return buildLinesResult;
  }
  preparedSceneOverlayPass_ = buildLinesResult.value();
  preparedSceneOverlayDepthTest_ = !shadowTexelGridSnapEnabled(frame);

  return Result<bool, std::string>::makeResult(true);
}

bool DebugRenderer::hasPreparedDebugGridPass() const noexcept {
  return preparedGridPass_;
}

bool DebugRenderer::hasPreparedDebugSceneOverlayPass() const noexcept {
  return preparedSceneOverlayPass_;
}

Result<bool, std::string>
DebugRenderer::appendDebugGridPass(RenderFrameContext &frame,
                                   RenderGraphBuilder &graph) {
  if (!preparedGridPass_) {
    return Result<bool, std::string>::makeResult(true);
  }

  const bool hasDepth = nuri::isValid(preparedSceneDepthTexture_);
  RenderGraphTextureId depthTextureId{};
  if (hasDepth) {
    if (nuri::isValid(preparedSceneDepthGraphTexture_)) {
      depthTextureId = preparedSceneDepthGraphTexture_;
    } else {
      auto importResult = graph.importTexture(preparedSceneDepthTexture_,
                                              "debug_depth_texture");
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }
      depthTextureId = importResult.value();
    }
  }

  RenderGraphGraphicsPassDesc gridPass{};
  gridPass.color = {.loadOp = preparedHasPriorColorPass_ ? LoadOp::Load
                                                         : LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}};
  if (nuri::isValid(preparedFrameColorTexture_)) {
    auto colorImportResult = graph.importTexture(preparedFrameColorTexture_,
                                                 "debug_grid_color_texture");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    gridPass.colorTexture = colorImportResult.value();
  }
  if (hasDepth) {
    gridPass.depth = {.loadOp = LoadOp::Load,
                      .storeOp = StoreOp::Store,
                      .clearDepth = 1.0f,
                      .clearStencil = 0};
    gridPass.depthTexture = depthTextureId;
  }
  gridPass.draws = std::span<const DrawItem>(&gridDrawItem_, 1u);
  gridPass.debugLabel = kGridPassLabel;
  gridPass.debugColor = kGridPassDebugColor;

  auto addResult = graph.addGraphicsPass(gridPass);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugRenderer::appendDebugSceneOverlayPass(RenderFrameContext &frame,
                                           RenderGraphBuilder &graph) {
  if (!preparedSceneOverlayPass_) {
    return Result<bool, std::string>::makeResult(true);
  }

  const Format overlayColorFormat =
      nuri::isValid(preparedFrameColorTexture_)
          ? gpu_.getTextureFormat(preparedFrameColorTexture_)
          : Format::Count;
  auto linePassResult = debugDraw3D_->buildGraphPass(
      frame.frameIndex, preparedSceneDepthTexture_, overlayColorFormat,
      preparedSceneOverlayDepthTest_);
  if (linePassResult.hasError()) {
    return Result<bool, std::string>::makeError(linePassResult.error());
  }

  DebugDraw3D::PreparedGraphPass pass = linePassResult.value();
  TextureHandle colorTexture = preparedFrameColorTexture_;
  if (!nuri::isValid(colorTexture)) {
    colorTexture = pass.colorTextureHandle;
  }
  if (nuri::isValid(colorTexture)) {
    auto colorImportResult =
        graph.importTexture(colorTexture, "debug_pass_color_texture");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    pass.desc.colorTexture = colorImportResult.value();
  }
  if (nuri::isValid(pass.depthTextureHandle)) {
    const bool useDepthOverride =
        nuri::isValid(preparedSceneDepthTexture_) &&
        nuri::isValid(preparedSceneDepthGraphTexture_) &&
        isSameTextureHandle(pass.depthTextureHandle,
                            preparedSceneDepthTexture_);
    if (useDepthOverride) {
      pass.desc.depthTexture = preparedSceneDepthGraphTexture_;
    } else {
      auto depthImportResult = graph.importTexture(pass.depthTextureHandle,
                                                   "debug_pass_depth_texture");
      if (depthImportResult.hasError()) {
        return Result<bool, std::string>::makeError(depthImportResult.error());
      }
      pass.desc.depthTexture = depthImportResult.value();
    }
  }
  pass.desc.debugColor = kLightIconPassDebugColor;

  auto addResult = graph.addGraphicsPass(pass.desc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> DebugRenderer::buildTransparentStageContribution(
    RenderFrameContext &frame, TransparentStageContribution &out) {
  NURI_PROFILER_FUNCTION();
  out = {};
  transparentSortableDraws_.clear();
  transparentFixedDraws_.clear();
  transparentDependencyBuffers_.clear();

  static std::atomic<uint32_t> loggedTransparentDebugBuilds{0u};
  const bool shouldLogTransparentDebugBuild =
      loggedTransparentDebugBuilds.fetch_add(1u, std::memory_order_relaxed) <
      16u;
  if (shouldLogTransparentDebugBuild) {
    NURI_LOG_DEBUG(
        "DebugRenderer::buildTransparentStageContribution: frame=%llu "
        "hasDebugWork=%u transparentStageEnabled=%u",
        static_cast<unsigned long long>(frame.frameIndex),
        hasDebugWork(frame) ? 1u : 0u,
        frame.sharedResources.transparentStageEnabled ? 1u : 0u);
  }

  if (!hasDebugWork(frame)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  if (shouldLogTransparentDebugBuild) {
    NURI_LOG_DEBUG(
        "DebugRenderer::buildTransparentStageContribution depth: frame=%llu "
        "depthValid=%u",
        static_cast<unsigned long long>(frame.frameIndex),
        nuri::isValid(depthTexture) ? 1u : 0u);
  }
  if (nuri::isValid(depthTexture)) {
    float debugSortDepth = 0.0f;
    auto buildLinesResult =
        buildSceneDebugLines(frame, depthTexture, debugSortDepth);
    if (buildLinesResult.hasError()) {
      return Result<bool, std::string>::makeError(buildLinesResult.error());
    }
    if (shouldLogTransparentDebugBuild) {
      NURI_LOG_DEBUG(
          "DebugRenderer::buildTransparentStageContribution lines: frame=%llu "
          "built=%u sortDepth=%.5f",
          static_cast<unsigned long long>(frame.frameIndex),
          buildLinesResult.value() ? 1u : 0u, debugSortDepth);
    }
    if (buildLinesResult.value()) {
      const TextureHandle frameColor = resolveFrameColorTexture(frame);
      const Format targetColorFormat = nuri::isValid(frameColor)
                                           ? gpu_.getTextureFormat(frameColor)
                                           : gpu_.getSwapchainFormat();
      auto linePassResult = debugDraw3D_->buildGraphPass(
          frame.frameIndex, depthTexture, targetColorFormat,
          !shadowTexelGridSnapEnabled(frame));
      if (linePassResult.hasError()) {
        return Result<bool, std::string>::makeError(linePassResult.error());
      }
      const DebugDraw3D::PreparedGraphPass pass = linePassResult.value();
      if (shouldLogTransparentDebugBuild) {
        NURI_LOG_DEBUG(
            "DebugRenderer::buildTransparentStageContribution pass: frame=%llu "
            "draws=%zu deps=%zu depthHandleValid=%u",
            static_cast<unsigned long long>(frame.frameIndex),
            pass.desc.draws.size(), pass.desc.dependencyBuffers.size(),
            nuri::isValid(pass.depthTextureHandle) ? 1u : 0u);
      }
      if (!pass.desc.draws.empty()) {
        for (size_t i = 0; i < pass.desc.draws.size(); ++i) {
          transparentSortableDraws_.push_back(TransparentStageSortableDraw{
              .draw = pass.desc.draws[i],
              .sortDepth = debugSortDepth,
              .stableOrder = static_cast<uint32_t>(i),
          });
        }
      }
      for (const BufferHandle buffer : pass.desc.dependencyBuffers) {
        if (nuri::isValid(buffer)) {
          transparentDependencyBuffers_.push_back(buffer);
        }
      }
    }
  }

  if (frame.settings->debug.grid) {
    auto gridResult = prepareGridDraw(frame, depthTexture);
    if (gridResult.hasError()) {
      return gridResult;
    }
    transparentFixedDraws_.push_back(gridDrawItem_);
  }

  out.sortableDraws = std::span<const TransparentStageSortableDraw>(
      transparentSortableDraws_.data(), transparentSortableDraws_.size());
  out.fixedDraws = std::span<const DrawItem>(transparentFixedDraws_.data(),
                                             transparentFixedDraws_.size());
  out.dependencyBuffers =
      std::span<const BufferHandle>(transparentDependencyBuffers_.data(),
                                    transparentDependencyBuffers_.size());
  out.textureReads = {};
  if (shouldLogTransparentDebugBuild) {
    NURI_LOG_DEBUG(
        "DebugRenderer::buildTransparentStageContribution out: frame=%llu "
        "sortable=%zu fixed=%zu deps=%zu",
        static_cast<unsigned long long>(frame.frameIndex),
        out.sortableDraws.size(), out.fixedDraws.size(),
        out.dependencyBuffers.size());
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
