#pragma once

#include "nuri/core/log.h"
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/sim/animation_scene_frame_data.h"
#include "nuri/scene/camera.h"
#include "nuri/scene/light.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory_resource>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

class RenderScene;
class ResourceManager;
struct RenderFrameContext;

enum class OpaqueDebugVisualization : uint8_t {
  None = 0,
  WireframeOverlay = 1,
  WireframeOnly = 2,
  TessPatchEdgesHeatmap = 3,
};

enum class TextureFilterMode : uint8_t {
  Bilinear = 0,
  Trilinear = 1,
  Anisotropic = 2,
};

enum class ToneMapper : uint8_t {
  ACES2_SDR = 0,
  AgX = 1,
};

enum class ShadowFilterMode : uint8_t {
  Hard = 0,
  PCF3x3 = 1,
  PoissonPCF = 2,
  PCSS = 3,
};

enum class ShadowCascadeSplitMode : uint8_t {
  Uniform = 0,
  Logarithmic = 1,
  Practical = 2,
};

enum class ShadowSdsmMode : uint8_t {
  Disabled = 0,
  PreviousFrameMinMax = 1,
  Histogram = 2,
};

enum class ShadowSdsmReductionBackend : uint8_t {
  Auto = 0,
  Cpu = 1,
  Gpu = 2,
};

enum class ShadowSdsmStatus : uint8_t {
  Disabled = 0,
  Active = 1,
  Unavailable = 2,
  Stale = 3,
  Invalid = 4,
  FallbackFixed = 5,
};

enum class ShadowPreviewMode : uint8_t {
  SelectedCascade = 0,
  TiledAllCascades = 1,
};

static constexpr float kDefaultToneMapExposureEv = 0.0f;
static constexpr float kDefaultAcesExposureOffsetEv = 0.35f;
static constexpr float kDefaultAgxExposureOffsetEv = -0.35f;
static constexpr float kDefaultToneMapCompareSplit = 0.5f;
static constexpr float kMinToneMapCompareSplit = 0.1f;
static constexpr float kMaxToneMapCompareSplit = 0.9f;

static constexpr uint32_t kMaxSceneDepthPyramidLevels = 16u;
static constexpr uint32_t kSceneDepthPyramidTexIdPackWidth = 4u;
static constexpr uint32_t kSceneDepthPyramidArraySize =
    (kMaxSceneDepthPyramidLevels + kSceneDepthPyramidTexIdPackWidth - 1u) /
    kSceneDepthPyramidTexIdPackWidth;
static constexpr uint32_t kMaxShadowPcfSamples = 64u;
static_assert(kSceneDepthPyramidArraySize * kSceneDepthPyramidTexIdPackWidth >=
              kMaxSceneDepthPyramidLevels);
static constexpr uint32_t kFrameCompositionSceneColorMipCount = 3u;
static constexpr Format kFrameCompositionSceneColorFormat =
    Format::RGBA16_FLOAT;
static constexpr Format kFrameCompositionFrameColorFormat =
    Format::RGBA16_FLOAT;
static constexpr Format kFrameCompositionDepthFormat = Format::D32_FLOAT;
static constexpr Format kDefaultShadowMapDepthFormat = Format::D16_UNORM;
static constexpr uint32_t kMaxShadowCascades = 4u;
static constexpr uint32_t kDefaultShadowSdsmHistogramBucketCount = 64u;
static constexpr uint32_t kMinShadowSdsmHistogramBucketCount = 8u;
static constexpr uint32_t kMaxShadowSdsmHistogramBucketCount = 128u;
static constexpr uint32_t kInvalidShadowBindlessIndex = 0xFFFFFFFFu;
static constexpr uint32_t kShadowFrameFlagEnabled = 1u << 0u;
static constexpr uint32_t kShadowFrameFlagVisualizeShadowFactor = 1u << 1u;
static constexpr uint32_t kShadowFrameFlagVisualizeCascadeIndex = 1u << 2u;
static constexpr uint32_t kShadowFrameFlagVisualizePCSSBlockers = 1u << 3u;
static constexpr uint32_t kShadowFrameFlagFixedPoissonRotation = 1u << 4u;
static constexpr uint32_t kShadowFrameFlagVisualizePCFResult = 1u << 5u;
static constexpr uint32_t kShadowFrameFlagVisualizeReceiverDepth = 1u << 6u;
static constexpr uint32_t kShadowFrameFlagVisualizeShadowMapDepth = 1u << 7u;
static constexpr uint32_t kShadowFrameFlagVisualizePCSSAverageBlockerDepth =
    1u << 8u;
static constexpr uint32_t kShadowFrameFlagVisualizePCSSFilterRadius = 1u << 9u;
static constexpr float kDefaultPcssLightRadiusScale = 0.35f;
static constexpr float kDefaultPcssSearchRadiusClampTexels = 12.0f;
static constexpr float kDefaultPcssFilterRadiusClampTexels = 18.0f;

[[nodiscard]] constexpr uint8_t
sanitizeTextureFilterAnisotropy(uint8_t anisotropy) noexcept {
  switch (anisotropy) {
  case 2u:
  case 4u:
  case 8u:
  case 16u:
    return anisotropy;
  default:
    return 8u;
  }
}

[[nodiscard]] constexpr TextureFilterMode
sanitizeTextureFilterMode(TextureFilterMode mode) noexcept {
  switch (mode) {
  case TextureFilterMode::Bilinear:
  case TextureFilterMode::Trilinear:
  case TextureFilterMode::Anisotropic:
    return mode;
  default:
    return TextureFilterMode::Trilinear;
  }
}

[[nodiscard]] constexpr ToneMapper
sanitizeToneMapper(ToneMapper mapper) noexcept {
  switch (mapper) {
  case ToneMapper::ACES2_SDR:
  case ToneMapper::AgX:
    return mapper;
  default:
    return ToneMapper::ACES2_SDR;
  }
}

struct RenderSettings {
  struct SkyboxSettings {
    bool enabled = true;
  };

  struct OpaqueSettings {
    bool enabled = true;
    // Depth pre-pass is workload dependent. Keep it opt-in because high
    // instance-count scenes can become vertex-bound and regress badly.
    bool enableDepthPrepass = false;
    // The pyramid is useful for later sampling consumers, but it adds one
    // graph graphics pass per depth level; keep it opt-in until consumed.
    bool enableDepthPyramid = false;
    OpaqueDebugVisualization debugVisualization =
        OpaqueDebugVisualization::None;
    bool enableInstanceCompute = true;
    bool enableIndirectDraw = true;
    bool enableInstancedDraw = true;
    bool enableMeshLod = true;
    int32_t forcedMeshLod = -1;
    glm::vec3 meshLodDistanceThresholds{8.0f, 16.0f, 32.0f};
    bool enableInstanceAnimation = true;
    bool enableTessellation = false;
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 6.0f;
    // 0 means "no cap".
    uint32_t tessMaxInstances = 256;
  };

  struct DebugSettings {
    bool enabled = false;
    bool modelBounds = false;
    bool grid = false;
    bool lightIcons = true;
  };

  struct ShadowDebugSettings {
    bool showCascadeFrusta = false;
    bool showLightViewBounds = false;
    bool showTexelGridSnap = false;
    bool showShadowMapViewport = false;
    bool showLightPerspectiveViewport = false;
    uint32_t debugCascadeIndex = 0;
    bool freezeCascades = false;
    bool freezeLightView = false;
    bool visualizeCascadeIndex = false;
    bool visualizeShadowFactor = false;
    bool visualizePCFResult = false;
    bool visualizeReceiverDepth = false;
    bool visualizeShadowMapDepth = false;
    bool visualizePCSSBlockers = false;
    bool visualizePCSSAverageBlockerDepth = false;
    bool visualizePCSSFilterRadius = false;
    bool fixedPoissonRotation = false;
    uint32_t poissonRotationSeed = 0u;
    bool visualizeSDSMHistogram = false;
    bool logDiagnostics = false;
    LogLevel diagnosticLogLevel = LogLevel::Trace;
    uint32_t diagnosticLogIntervalFrames = 1u;
    bool diagnosticLogOnlyOnChange = false;
    bool enableCascadeCasterCulling = true;
    ShadowPreviewMode previewMode = ShadowPreviewMode::SelectedCascade;
    float previewDepthMin = 0.0f;
    float previewDepthMax = 1.0f;
    bool previewDepthInvert = false;
    bool previewDepthLog = false;
  };

  struct ShadowSettings {
    bool enabled = true;
    uint32_t cascadeCount = kMaxShadowCascades;
    uint32_t shadowMapSize = 2048;
    float maxDistance = 150.0f;
    ShadowCascadeSplitMode splitMode = ShadowCascadeSplitMode::Practical;
    float splitLambda = 0.75f;
    bool stabilizeCascades = true;
    float cascadeBlendFraction = 0.08f;
    float constantBias = 0.0005f;
    float slopeBias = 1.5f;
    float normalBias = 0.0f;
    ShadowFilterMode filterMode = ShadowFilterMode::Hard;
    uint32_t pcfSampleCount = 9;
    uint32_t pcssBlockerSampleCount = 16;
    uint32_t pcssFilterSampleCount = 32;
    float pcssLightRadiusScale = kDefaultPcssLightRadiusScale;
    float pcssSearchRadiusClampTexels = kDefaultPcssSearchRadiusClampTexels;
    float pcssFilterRadiusClampTexels = kDefaultPcssFilterRadiusClampTexels;
    ShadowSdsmMode sdsmMode = ShadowSdsmMode::Disabled;
    ShadowSdsmReductionBackend sdsmReductionBackend =
        ShadowSdsmReductionBackend::Auto;
    float sdsmTemporalBlend = 0.85f;
    uint32_t sdsmHistogramBucketCount = kDefaultShadowSdsmHistogramBucketCount;
    float sdsmHistogramTrimLowPercent = 0.5f;
    float sdsmHistogramTrimHighPercent = 1.0f;
    ShadowDebugSettings debug{};
  };

  struct TransparentSettings {
    bool enabled = true;
  };

  struct TransmissionSettings {
    bool enabled = true;
  };

  struct TextureFilteringSettings {
    TextureFilterMode mode = TextureFilterMode::Trilinear;
    uint8_t anisotropy = 8u;
  };

  struct ToneMapSettings {
    ToneMapper operator_ = ToneMapper::ACES2_SDR;
    float exposureEv = kDefaultToneMapExposureEv;
    float acesExposureOffsetEv = kDefaultAcesExposureOffsetEv;
    float agxExposureOffsetEv = kDefaultAgxExposureOffsetEv;
    bool grayCardDebug = false;
    bool sideBySideCompare = false;
    float compareSplit = kDefaultToneMapCompareSplit;
  };

  SkyboxSettings skybox{};
  OpaqueSettings opaque{};
  TransmissionSettings transmission{};
  TransparentSettings transparent{};
  DebugSettings debug{};
  ShadowSettings shadow{};
  TextureFilteringSettings textureFiltering{};
  ToneMapSettings toneMap{};
};

[[nodiscard]] inline uint32_t
shadowDebugFrameFlags(const RenderSettings::ShadowDebugSettings &debug) {
  uint32_t flags = 0u;
  if (debug.visualizeShadowFactor) {
    flags |= kShadowFrameFlagVisualizeShadowFactor;
  }
  if (debug.visualizeCascadeIndex) {
    flags |= kShadowFrameFlagVisualizeCascadeIndex;
  }
  if (debug.visualizePCFResult) {
    flags |= kShadowFrameFlagVisualizePCFResult;
  }
  if (debug.visualizeReceiverDepth) {
    flags |= kShadowFrameFlagVisualizeReceiverDepth;
  }
  if (debug.visualizeShadowMapDepth) {
    flags |= kShadowFrameFlagVisualizeShadowMapDepth;
  }
  if (debug.visualizePCSSBlockers) {
    flags |= kShadowFrameFlagVisualizePCSSBlockers;
  }
  if (debug.visualizePCSSAverageBlockerDepth) {
    flags |= kShadowFrameFlagVisualizePCSSAverageBlockerDepth;
  }
  if (debug.visualizePCSSFilterRadius) {
    flags |= kShadowFrameFlagVisualizePCSSFilterRadius;
  }
  if (debug.fixedPoissonRotation) {
    flags |= kShadowFrameFlagFixedPoissonRotation;
  }
  return flags;
}

inline void sanitizeTextureFilteringSettings(
    RenderSettings::TextureFilteringSettings &settings) {
  settings.mode = sanitizeTextureFilterMode(settings.mode);
  settings.anisotropy = sanitizeTextureFilterAnisotropy(settings.anisotropy);
}

inline void sanitizeToneMapSettings(RenderSettings::ToneMapSettings &settings) {
  settings.operator_ = sanitizeToneMapper(settings.operator_);
  settings.compareSplit = std::clamp(
      settings.compareSplit, kMinToneMapCompareSplit, kMaxToneMapCompareSplit);
}

[[nodiscard]] constexpr ShadowFilterMode
sanitizeShadowFilterMode(ShadowFilterMode mode) noexcept {
  switch (mode) {
  case ShadowFilterMode::Hard:
  case ShadowFilterMode::PCF3x3:
  case ShadowFilterMode::PoissonPCF:
  case ShadowFilterMode::PCSS:
    return mode;
  default:
    return ShadowFilterMode::Hard;
  }
}

[[nodiscard]] constexpr ShadowSdsmMode
sanitizeShadowSdsmMode(ShadowSdsmMode mode) noexcept {
  switch (mode) {
  case ShadowSdsmMode::Disabled:
  case ShadowSdsmMode::PreviousFrameMinMax:
  case ShadowSdsmMode::Histogram:
    return mode;
  default:
    return ShadowSdsmMode::Disabled;
  }
}

[[nodiscard]] constexpr ShadowSdsmReductionBackend
sanitizeShadowSdsmReductionBackend(
    ShadowSdsmReductionBackend backend) noexcept {
  switch (backend) {
  case ShadowSdsmReductionBackend::Auto:
  case ShadowSdsmReductionBackend::Cpu:
  case ShadowSdsmReductionBackend::Gpu:
    return backend;
  default:
    return ShadowSdsmReductionBackend::Auto;
  }
}

[[nodiscard]] constexpr ShadowPreviewMode
sanitizeShadowPreviewMode(ShadowPreviewMode mode) noexcept {
  switch (mode) {
  case ShadowPreviewMode::SelectedCascade:
  case ShadowPreviewMode::TiledAllCascades:
    return mode;
  default:
    return ShadowPreviewMode::SelectedCascade;
  }
}

[[nodiscard]] constexpr ShadowCascadeSplitMode
sanitizeShadowCascadeSplitMode(ShadowCascadeSplitMode mode) noexcept {
  switch (mode) {
  case ShadowCascadeSplitMode::Uniform:
  case ShadowCascadeSplitMode::Logarithmic:
  case ShadowCascadeSplitMode::Practical:
    return mode;
  default:
    return ShadowCascadeSplitMode::Practical;
  }
}

inline void sanitizeShadowSettings(RenderSettings::ShadowSettings &settings) {
  settings.cascadeCount =
      std::clamp(settings.cascadeCount, 1u, kMaxShadowCascades);
  settings.shadowMapSize = std::max(settings.shadowMapSize, 1u);
  settings.maxDistance =
      std::isfinite(settings.maxDistance) && settings.maxDistance > 0.0f
          ? settings.maxDistance
          : 150.0f;
  settings.splitLambda = std::clamp(settings.splitLambda, 0.0f, 1.0f);
  settings.cascadeBlendFraction =
      std::clamp(settings.cascadeBlendFraction, 0.0f, 1.0f);
  settings.filterMode = sanitizeShadowFilterMode(settings.filterMode);
  settings.splitMode = sanitizeShadowCascadeSplitMode(settings.splitMode);
  settings.sdsmMode = sanitizeShadowSdsmMode(settings.sdsmMode);
  settings.sdsmReductionBackend =
      sanitizeShadowSdsmReductionBackend(settings.sdsmReductionBackend);
  settings.debug.previewMode =
      sanitizeShadowPreviewMode(settings.debug.previewMode);
  // Light-perspective shadow preview is kept in settings but has no renderer
  // path yet, so sanitize it off instead of accepting a no-op debug flag.
  settings.debug.showLightPerspectiveViewport = false;
  settings.pcfSampleCount =
      std::clamp(settings.pcfSampleCount, 1u, kMaxShadowPcfSamples);
  settings.pcssBlockerSampleCount =
      std::clamp(settings.pcssBlockerSampleCount, 1u, kMaxShadowPcfSamples);
  settings.pcssFilterSampleCount =
      std::clamp(settings.pcssFilterSampleCount, 1u, kMaxShadowPcfSamples);
  settings.pcssLightRadiusScale =
      std::isfinite(settings.pcssLightRadiusScale)
          ? std::max(settings.pcssLightRadiusScale, 0.0f)
          : kDefaultPcssLightRadiusScale;
  settings.pcssSearchRadiusClampTexels =
      std::isfinite(settings.pcssSearchRadiusClampTexels)
          ? std::max(settings.pcssSearchRadiusClampTexels, 0.0f)
          : kDefaultPcssSearchRadiusClampTexels;
  settings.pcssFilterRadiusClampTexels =
      std::isfinite(settings.pcssFilterRadiusClampTexels)
          ? std::max(settings.pcssFilterRadiusClampTexels, 0.0f)
          : kDefaultPcssFilterRadiusClampTexels;
  settings.sdsmTemporalBlend =
      std::clamp(settings.sdsmTemporalBlend, 0.0f, 1.0f);
  settings.sdsmHistogramBucketCount = std::clamp(
      settings.sdsmHistogramBucketCount, kMinShadowSdsmHistogramBucketCount,
      kMaxShadowSdsmHistogramBucketCount);
  settings.sdsmHistogramTrimLowPercent =
      std::isfinite(settings.sdsmHistogramTrimLowPercent)
          ? std::clamp(settings.sdsmHistogramTrimLowPercent, 0.0f, 20.0f)
          : 0.5f;
  settings.sdsmHistogramTrimHighPercent =
      std::isfinite(settings.sdsmHistogramTrimHighPercent)
          ? std::clamp(settings.sdsmHistogramTrimHighPercent, 0.0f, 20.0f)
          : 1.0f;
  switch (settings.debug.diagnosticLogLevel) {
  case LogLevel::Trace:
  case LogLevel::Debug:
  case LogLevel::Info:
  case LogLevel::Warning:
  case LogLevel::Error:
  case LogLevel::Fatal:
    break;
  default:
    settings.debug.diagnosticLogLevel = LogLevel::Trace;
    break;
  }
  settings.debug.diagnosticLogIntervalFrames =
      std::max(settings.debug.diagnosticLogIntervalFrames, 1u);
  settings.debug.debugCascadeIndex =
      std::min(settings.debug.debugCascadeIndex, settings.cascadeCount - 1u);
  if (!std::isfinite(settings.debug.previewDepthMin) ||
      !std::isfinite(settings.debug.previewDepthMax)) {
    settings.debug.previewDepthMin = 0.0f;
    settings.debug.previewDepthMax = 1.0f;
  }
  settings.debug.previewDepthMin =
      std::clamp(settings.debug.previewDepthMin, 0.0f, 1.0f);
  settings.debug.previewDepthMax =
      std::clamp(settings.debug.previewDepthMax, 0.0f, 1.0f);
  if (settings.debug.previewDepthMax <= settings.debug.previewDepthMin) {
    settings.debug.previewDepthMax =
        std::min(settings.debug.previewDepthMin + 1.0e-4f, 1.0f);
    settings.debug.previewDepthMin =
        std::min(settings.debug.previewDepthMin,
                 settings.debug.previewDepthMax - 1.0e-4f);
  }
}

[[nodiscard]] constexpr uint32_t
shadowFilterSampleBudget(ShadowFilterMode filterMode, uint32_t pcfSampleCount,
                         uint32_t pcssBlockerSampleCount,
                         uint32_t pcssFilterSampleCount) noexcept {
  switch (sanitizeShadowFilterMode(filterMode)) {
  case ShadowFilterMode::Hard:
    return 1u;
  case ShadowFilterMode::PCF3x3:
  case ShadowFilterMode::PoissonPCF:
    return std::clamp(pcfSampleCount, 1u, kMaxShadowPcfSamples);
  case ShadowFilterMode::PCSS:
    return std::clamp(pcssBlockerSampleCount, 1u, kMaxShadowPcfSamples) +
           std::clamp(pcssFilterSampleCount, 1u, kMaxShadowPcfSamples);
  }
  return 1u;
}

[[nodiscard]] inline TextureFilterMode effectiveTextureFilterMode(
    const RenderSettings::TextureFilteringSettings &settings,
    uint8_t maxSamplerAnisotropy) {
  const TextureFilterMode mode = sanitizeTextureFilterMode(settings.mode);
  if (mode != TextureFilterMode::Anisotropic) {
    return mode;
  }
  return maxSamplerAnisotropy > 1u ? TextureFilterMode::Anisotropic
                                   : TextureFilterMode::Trilinear;
}

struct CameraFrameState {
  glm::mat4 view{1.0f};
  glm::mat4 proj{1.0f};
  glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
  float aspectRatio = 1.0f;
  ProjectionType projectionType = ProjectionType::Perspective;
  float nearPlane = 0.1f;
  float farPlane = 1000.0f;
  float fovYRadians = glm::radians(60.0f);
  float orthoHeight = 10.0f;
};

[[nodiscard]] inline CameraFrameState makeCameraFrameState(const Camera &camera,
                                                           float aspectRatio) {
  CameraFrameState state{};
  state.view = camera.viewMatrix();
  state.proj = camera.projectionMatrix(aspectRatio);
  state.cameraPos = glm::vec4(camera.position(), 1.0f);
  state.aspectRatio = aspectRatio;
  state.projectionType = camera.projectionType();
  if (state.projectionType == ProjectionType::Orthographic) {
    const OrthographicParams &ortho = camera.orthographic();
    state.nearPlane = ortho.nearPlane;
    state.farPlane = ortho.farPlane;
    state.orthoHeight = ortho.height;
  } else {
    const PerspectiveParams &perspective = camera.perspective();
    state.nearPlane = perspective.nearPlane;
    state.farPlane = perspective.farPlane;
    state.fovYRadians = perspective.fovYRadians;
  }
  return state;
}

struct alignas(16) ShadowCascadeGpuData {
  glm::mat4 lightViewProj{1.0f};
  glm::mat4 lightView{1.0f};
  glm::vec4 splitDepthTexelSize{0.0f};
  glm::vec4 uvScaleBias{1.0f, 1.0f, 0.0f, 0.0f};
  glm::vec4 biasParams{0.0f};
  glm::vec4 pcssParams{0.0f};
  glm::uvec4 textureSampler{kInvalidShadowBindlessIndex,
                            kInvalidShadowBindlessIndex, 0u, 0u};
};
static_assert(sizeof(ShadowCascadeGpuData) == 208u,
              "ShadowCascadeGpuData must match shader layout");

struct alignas(16) ShadowFrameGpuData {
  glm::uvec4 flagsCascadeCountLightIndex{0u};
  glm::vec4 fadeParams{0.0f};
  glm::uvec4 filterParams{0u};
  std::array<ShadowCascadeGpuData, kMaxShadowCascades> cascades{};
};
static_assert(sizeof(ShadowFrameGpuData) == 880u,
              "ShadowFrameGpuData must match shader layout");

struct ShadowFrameGpuDataHandle {
  BufferHandle buffer{};
  uint64_t bufferAddress = 0u;
};

struct ShadowSdsmGpuReduceTargetHandle {
  BufferHandle buffer{};
  uint64_t bufferAddress = 0u;
};

struct ShadowCascadeDebugFrameData {
  float splitNear = 0.0f;
  float splitFar = 0.0f;
  float texelWorldSize = 0.0f;
  glm::mat4 lightView{1.0f};
  glm::mat4 lightProj{1.0f};
  glm::mat4 lightViewProj{1.0f};
  glm::mat4 inverseLightView{1.0f};
  glm::mat4 inverseLightProj{1.0f};
  std::array<glm::vec4, 8> worldFrustumCorners{};
  glm::vec4 lightSpaceBoundsMin{0.0f};
  glm::vec4 lightSpaceBoundsMax{0.0f};
  glm::vec4 unsnappedCenter{0.0f};
  glm::vec4 snappedCenter{0.0f};
  TextureHandle texture{};
  uint32_t textureBindlessId = kInvalidShadowBindlessIndex;
  uint32_t drawCount = 0u;
  uint32_t culledCount = 0u;
  uint32_t staticDrawCount = 0u;
  uint32_t dynamicDrawCount = 0u;
  enum class StaticOnlyReuseStatus : uint8_t {
    None = 0u,
    Reused,
    StaticCacheRebuilt,
    HasDynamicCasters,
    NoPreviousStaticOnlyPass,
    RasterStateChanged,
    AdaptiveRefresh,
  };
  StaticOnlyReuseStatus staticOnlyReuseStatus = StaticOnlyReuseStatus::None;
  bool staticOnlyReuseCandidate = false;
  bool staticOnlyReusePreviousValid = false;
  bool staticOnlyReuseLightViewProjChanged = false;
  bool staticOnlyReuseBiasChanged = false;
  bool staticOnlyReuseCasterSignatureChanged = false;
  bool staticOnlyReuseAdaptiveRefresh = false;
  uint64_t currentStaticOnlyRasterSignature = 0u;
  uint64_t previousStaticOnlyRasterSignature = 0u;
  uint64_t currentStaticOnlyLightViewProjSignature = 0u;
  uint64_t previousStaticOnlyLightViewProjSignature = 0u;
  uint64_t currentStaticOnlyBiasSignature = 0u;
  uint64_t previousStaticOnlyBiasSignature = 0u;
  uint64_t currentStaticOnlyCasterSignature = 0u;
  uint64_t previousStaticOnlyCasterSignature = 0u;
};

struct ShadowSdsmDebugFrameData {
  ShadowSdsmMode mode = ShadowSdsmMode::Disabled;
  ShadowSdsmReductionBackend requestedReductionBackend =
      ShadowSdsmReductionBackend::Auto;
  ShadowSdsmReductionBackend activeReductionBackend =
      ShadowSdsmReductionBackend::Cpu;
  ShadowSdsmStatus status = ShadowSdsmStatus::Disabled;
  bool reductionFallbackActive = false;
  bool fixedFallbackActive = false;
  uint64_t sourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint32_t histogramSourceLevel = 0u;
  glm::uvec2 histogramSourceDimensions{0u};
  uint32_t histogramBucketCount = 0u;
  uint32_t histogramValidTileCount = 0u;
  uint32_t gpuResultRingSlotCount = 0u;
  uint32_t gpuResultSelectedSlot = std::numeric_limits<uint32_t>::max();
  bool gpuReductionResultAvailable = false;
  bool gpuSplitPayloadValid = false;
  uint64_t gpuResultSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint32_t splitCount = 0u;
  float fixedRangeNear = 0.0f;
  float fixedRangeFar = 0.0f;
  float rawDeviceMin = 0.0f;
  float rawDeviceMax = 0.0f;
  float rawLinearMin = 0.0f;
  float rawLinearMax = 0.0f;
  float smoothedLinearMin = 0.0f;
  float smoothedLinearMax = 0.0f;
  float histogramTotalWeight = 0.0f;
  float histogramTrimLowPercent = 0.0f;
  float histogramTrimHighPercent = 0.0f;
  float histogramTrimmedRangeNear = 0.0f;
  float histogramTrimmedRangeFar = 0.0f;
  float effectiveRangeNear = 0.0f;
  float effectiveRangeFar = 0.0f;
  std::array<float, kMaxShadowCascades + 1u> fixedSplitDepths{};
  std::array<float, kMaxShadowCascades + 1u> minMaxSplitDepths{};
  std::array<float, kMaxShadowCascades + 1u> histogramSplitDepths{};
  std::array<float, kMaxShadowCascades + 1u> effectiveSplitDepths{};
  std::array<float, kMaxShadowSdsmHistogramBucketCount>
      histogramBucketWeights{};
};

struct ShadowDebugFrameData {
  LightId selectedShadowLightId = kInvalidLightId;
  uint32_t cascadeCount = 0u;
  uint32_t rawSamplerId = kInvalidShadowBindlessIndex;
  uint32_t compareSamplerId = kInvalidShadowBindlessIndex;
  ShadowSdsmDebugFrameData sdsm{};
  std::array<ShadowCascadeDebugFrameData, kMaxShadowCascades> cascades{};
};

struct ForwardSceneFrameData {
  glm::mat4 view{1.0f};
  glm::mat4 proj{1.0f};
  glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
  uint32_t cubemapTexId = 0;
  uint32_t hasCubemap = 0;
  uint32_t irradianceTexId = 0;
  uint32_t prefilteredGgxTexId = 0;
  uint32_t prefilteredCharlieTexId = 0;
  uint32_t brdfLutTexId = 0;
  uint32_t flags = 0;
  uint32_t cubemapSamplerId = 0;
  uint32_t materialSamplerId = 0;
  uint32_t sceneColorTexId = 0;
  uint32_t sceneColorSamplerId = 0;
  uint32_t sceneColorHalfResTexId = 0;
  uint32_t sceneColorQuarterResTexId = 0;
  uint32_t sceneDepthTexId = 0;
  uint32_t sceneDepthSamplerId = 0;
  uint32_t sceneDepthPyramidLevelCount = 0;
  std::array<glm::uvec4, kSceneDepthPyramidArraySize> sceneDepthPyramidTexIds{};
  uint64_t directionalLightBufferAddress = 0;
  uint64_t localLightBufferAddress = 0;
  uint64_t materialHeaderBufferAddress = 0;
  uint64_t materialClearcoatBufferAddress = 0;
  uint64_t materialSheenBufferAddress = 0;
  uint64_t materialTransmissionBufferAddress = 0;
  uint64_t materialSpecularBufferAddress = 0;
  uint32_t directionalLightCount = 0;
  uint32_t localLightCount = 0;
  uint64_t shadowFrameBufferAddress = 0;
  uint32_t shadowFlags = 0;
  uint32_t shadowReserved0 = 0;

  [[nodiscard]] bool
  operator==(const ForwardSceneFrameData &other) const noexcept {
    return std::memcmp(this, &other, sizeof(ForwardSceneFrameData)) == 0;
  }

  [[nodiscard]] bool
  operator!=(const ForwardSceneFrameData &other) const noexcept {
    return !(*this == other);
  }
};
static_assert(sizeof(ForwardSceneFrameData) == 352,
              "ForwardSceneFrameData must match shader FrameDataBuffer layout");
static_assert(offsetof(ForwardSceneFrameData, directionalLightBufferAddress) ==
              272u);
static_assert(offsetof(ForwardSceneFrameData, localLightBufferAddress) == 280u);
static_assert(offsetof(ForwardSceneFrameData, materialHeaderBufferAddress) ==
              288u);
static_assert(offsetof(ForwardSceneFrameData, materialClearcoatBufferAddress) ==
              296u);
static_assert(offsetof(ForwardSceneFrameData, materialSheenBufferAddress) ==
              304u);
static_assert(offsetof(ForwardSceneFrameData,
                       materialTransmissionBufferAddress) == 312u);
static_assert(offsetof(ForwardSceneFrameData, materialSpecularBufferAddress) ==
              320u);
static_assert(offsetof(ForwardSceneFrameData, directionalLightCount) == 328u);
static_assert(offsetof(ForwardSceneFrameData, localLightCount) == 332u);
static_assert(offsetof(ForwardSceneFrameData, shadowFrameBufferAddress) ==
              336u);
static_assert(offsetof(ForwardSceneFrameData, shadowFlags) == 344u);

// GPU-side forwarding of the light metadata carried in ForwardSceneFrameData.
// The CPU owns allocation and updates of ForwardSceneFrameData, then derives
// the resolved GPU addresses/counts below before publishing them for consumers.
// Keep this struct in sync with ForwardSceneFrameData's layout/semantics for
// directional/local light buffer addresses and counts to avoid
// desynchronization bugs when the CPU-side contract changes.
struct ForwardSceneGpuData {
  BufferHandle buffer{};
  uint64_t frameDataAddress = 0;
  uint64_t directionalLightBufferAddress = 0;
  uint64_t localLightBufferAddress = 0;
  uint64_t shadowFrameBufferAddress = 0;
  uint32_t directionalLightCount = 0;
  uint32_t localLightCount = 0;
  uint32_t shadowFlags = 0;
};

// Keep this one-to-one with ForwardSceneFrameData's material buffer address
// fields: headerBufferAddress, clearcoatBufferAddress, sheenBufferAddress,
// transmissionBufferAddress, and specularBufferAddress. Reordering, renaming,
// or layout changes here must be mirrored there; update version with structural
// changes so consumers do not desynchronize cached table data.
struct MaterialTableGpuData {
  BufferHandle headerBuffer{};
  BufferHandle clearcoatBuffer{};
  BufferHandle sheenBuffer{};
  BufferHandle transmissionBuffer{};
  BufferHandle specularBuffer{};
  uint64_t headerBufferAddress = 0;
  uint64_t clearcoatBufferAddress = 0;
  uint64_t sheenBufferAddress = 0;
  uint64_t transmissionBufferAddress = 0;
  uint64_t specularBufferAddress = 0;
  uint64_t version = 0;
};

struct OpaqueFrameMetrics {
  uint32_t totalInstances = 0;
  uint32_t visibleInstances = 0;
  uint32_t instancedDraws = 0;
  uint32_t indirectDrawCalls = 0;
  uint32_t indirectCommands = 0;
  uint32_t tessellatedDraws = 0;
  uint32_t tessellatedInstances = 0;
  uint32_t debugOverlayDraws = 0;
  uint32_t debugOverlayFallbackDraws = 0;
  uint32_t debugPatchHeatmapDraws = 0;
  uint32_t computeDispatches = 0;
  uint32_t computeDispatchX = 0;
  uint32_t depthPrepassDraws = 0;
  uint32_t depthPrepassIndirectDraws = 0;
  uint32_t depthPyramidLevels = 0;
  uint32_t depthPrepassEnabled = 0;
};

struct ShadowFrameMetrics {
  uint32_t cascadeCount = 0;
  uint32_t shadowMapSize = 0;
  uint32_t totalDraws = 0;
  uint32_t totalCulledDraws = 0;
  uint32_t totalIndexCountEstimate = 0;
  uint32_t staticCasterEntries = 0;
  uint32_t dynamicCasterEntries = 0;
  uint32_t staticCacheReused = 0;
  uint32_t staticOnlyCandidateCount = 0;
  uint32_t reusedStaticOnlyCascadeCount = 0;
  uint32_t staticOnlyReuseMissStaticCacheRebuiltCount = 0;
  uint32_t staticOnlyReuseMissDynamicCasterCount = 0;
  uint32_t staticOnlyReuseMissNoPreviousCount = 0;
  uint32_t staticOnlyReuseMissRasterStateChangedCount = 0;
  uint32_t staticOnlyReuseMissAdaptiveRefreshCount = 0;
  uint64_t cascadeTextureBytes = 0;
  uint32_t filterSampleBudget = 0;
  uint32_t pcssBlockerSampleBudget = 0;
  uint32_t pcssFilterSampleBudget = 0;
  uint32_t pcssMaxSamplesPerReceiver = 0;
  uint32_t pcssMaxSamplesPerBlendedReceiver = 0;
  uint32_t sdsmComputePassCount = 0;
  uint32_t sdsmReadbackBytes = 0;
  uint32_t sdsmReductionSourceSamples = 0;
  uint32_t sdsmHistogramSourceSamples = 0;
  ShadowSdsmReductionBackend sdsmActiveReductionBackend =
      ShadowSdsmReductionBackend::Cpu;
  float gpuTimeMs = 0.0f;
  float depthGpuTimeMs = 0.0f;
  float sdsmGpuTimeMs = 0.0f;
  float sdsmCpuReductionTimeMs = 0.0f;
  float sdsmCpuHistogramTimeMs = 0.0f;
  uint64_t gpuTimingSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t depthGpuTimingSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t sdsmGpuTimingSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint32_t gpuTimingAvailable = 0;
  uint32_t depthGpuTimingAvailable = 0;
  uint32_t sdsmGpuTimingAvailable = 0;
};

struct RenderFrameMetrics {
  OpaqueFrameMetrics opaque{};
  ShadowFrameMetrics shadow{};
  struct TransparentFrameMetrics {
    uint32_t meshDraws = 0;
    uint32_t contributorSortableDraws = 0;
    uint32_t contributorFixedDraws = 0;
    uint32_t pickDraws = 0;
  } transparent{};
};

struct OpaquePickRequest {
  uint32_t x = 0;
  uint32_t y = 0;
  uint64_t requestId = 0;
};

struct OpaquePickResult {
  uint64_t requestId = 0;
  bool hit = false;
  uint32_t renderableIndex = 0;
};

struct ShadowInspectRequest {
  uint32_t x = 0;
  uint32_t y = 0;
  uint64_t requestId = 0;
};

struct ShadowInspectResult {
  uint64_t requestId = 0;
  bool valid = false;
  float receiverDepth = 0.0f;
  float receiverCompareDepth = 0.0f;
  float sampledDepth = 0.0f;
  uint32_t cascadeIndex = 0u;
  float cascadeBlendFactor = 0.0f;
};

enum class FrameTextureRequirementFlags : uint32_t {
  None = 0u,
  SceneColor = 1u << 0u,
  FrameColor = 1u << 1u,
  SceneDepth = 1u << 2u,
  SceneColorMipChain = 1u << 3u,
  HistoryColor = 1u << 4u,
  MotionVectors = 1u << 5u,
  Normals = 1u << 6u,
  Exposure = 1u << 7u,
};

[[nodiscard]] constexpr FrameTextureRequirementFlags
operator|(FrameTextureRequirementFlags lhs, FrameTextureRequirementFlags rhs) {
  return static_cast<FrameTextureRequirementFlags>(static_cast<uint32_t>(lhs) |
                                                   static_cast<uint32_t>(rhs));
}

[[nodiscard]] constexpr FrameTextureRequirementFlags
operator&(FrameTextureRequirementFlags lhs, FrameTextureRequirementFlags rhs) {
  return static_cast<FrameTextureRequirementFlags>(static_cast<uint32_t>(lhs) &
                                                   static_cast<uint32_t>(rhs));
}

constexpr FrameTextureRequirementFlags &
operator|=(FrameTextureRequirementFlags &lhs,
           FrameTextureRequirementFlags rhs) {
  lhs = lhs | rhs;
  return lhs;
}

constexpr FrameTextureRequirementFlags &
operator&=(FrameTextureRequirementFlags &lhs,
           FrameTextureRequirementFlags rhs) {
  lhs = lhs & rhs;
  return lhs;
}

[[nodiscard]] constexpr bool
hasFrameTextureRequirementFlag(FrameTextureRequirementFlags flags,
                               FrameTextureRequirementFlags flag) {
  return static_cast<uint32_t>(flags & flag) != 0u;
}

static constexpr FrameTextureRequirementFlags
    kBaselineFrameTextureRequirements =
        FrameTextureRequirementFlags::SceneColor |
        FrameTextureRequirementFlags::FrameColor |
        FrameTextureRequirementFlags::SceneDepth |
        FrameTextureRequirementFlags::SceneColorMipChain |
        FrameTextureRequirementFlags::HistoryColor;

struct FrameSharedResources {
  std::optional<ForwardSceneGpuData> forwardSceneGpuData{};
  std::optional<MaterialTableGpuData> materialTableGpuData{};
  std::optional<AnimationSceneFrameData> animationSceneGpuData{};
  std::optional<ShadowFrameGpuDataHandle> shadowFrameGpuData{};
  std::optional<ShadowSdsmGpuReduceTargetHandle> shadowSdsmGpuReduceTarget{};
  ComputePipelineHandle shadowSdsmGpuReducePipeline{};
  std::optional<ShadowDebugFrameData> shadowDebugFrameData{};
  FrameTextureRequirementFlags textureRequirements =
      kBaselineFrameTextureRequirements;
  TextureHandle sceneDepthTexture{};
  RenderGraphTextureId sceneDepthGraphTexture{};
  std::array<TextureHandle, kMaxSceneDepthPyramidLevels>
      sceneDepthPyramidTextures{};
  std::array<RenderGraphTextureId, kMaxSceneDepthPyramidLevels>
      sceneDepthPyramidGraphTextures{};
  std::optional<uint64_t> sceneDepthPyramidSourceFrameIndex{};
  std::array<TextureHandle, kMaxShadowCascades> shadowCascadeTextures{};
  std::array<RenderGraphTextureId, kMaxShadowCascades>
      shadowCascadeGraphTextures{};
  TextureHandle shadowDebugPreviewTexture{};
  RenderGraphTextureId shadowDebugPreviewGraphTexture{};
  uint32_t shadowRawSamplerId = kInvalidShadowBindlessIndex;
  uint32_t shadowCompareSamplerId = kInvalidShadowBindlessIndex;
  uint32_t sceneDepthPyramidLevelCount = 0;
  uint32_t sceneDepthSamplerId = 0;
  TextureHandle sceneColorTexture{};
  TextureHandle sceneColorHalfResTexture{};
  TextureHandle sceneColorQuarterResTexture{};
  RenderGraphTextureId sceneColorGraphTexture{};
  TextureHandle frameColorTexture{};
  RenderGraphTextureId frameColorGraphTexture{};
  TextureHandle historyColorReadTexture{};
  TextureHandle historyColorWriteTexture{};
  RenderGraphTextureId opaquePickGraphTexture{};
  RenderGraphTextureId opaquePickDepthGraphTexture{};
  std::optional<LightId> selectedLightId{};
  std::optional<LightId> selectedShadowLightId{};
  bool transparentStageEnabled = false;
};

struct TransparentStageSortableDraw {
  DrawItem draw{};
  float sortDepth = 0.0f;
  uint32_t stableOrder = 0;
};

// These spans are non-owning views into contributor-managed frame storage.
// They are valid only for the current frame and may be invalidated by the
// contributor's next clear()/beginFrame()-style reset. Copy the data if it
// must outlive the current frame.
struct TransparentStageContribution {
  std::span<const TransparentStageSortableDraw> sortableDraws{};
  std::span<const DrawItem> fixedDraws{};
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const TextureHandle> textureReads{};
};

using TransparentContributionCollectFn = Result<bool, std::string> (*)(
    void *user, RenderFrameContext &frame, TransparentStageContribution &out);

struct TransparentContributionCollector {
  void *user = nullptr;
  TransparentContributionCollectFn collect = nullptr;
};

class TransparentContributionRegistry {
public:
  explicit TransparentContributionRegistry(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : collectors_(memory != nullptr ? memory
                                      : std::pmr::get_default_resource()) {}

  void clear() { collectors_.clear(); }

  void publish(TransparentContributionCollector collector) {
    if (collector.user == nullptr || collector.collect == nullptr) {
      return;
    }
    collectors_.push_back(collector);
  }

  [[nodiscard]] std::span<const TransparentContributionCollector>
  collectors() const noexcept {
    return std::span<const TransparentContributionCollector>(
        collectors_.data(), collectors_.size());
  }

  [[nodiscard]] bool empty() const noexcept { return collectors_.empty(); }

private:
  std::pmr::vector<TransparentContributionCollector> collectors_;
};

struct RenderFrameContext {
  const RenderScene *scene = nullptr;
  CameraFrameState camera{};
  RenderSettings *settings = nullptr;
  RenderFrameMetrics metrics{};
  // Frame-scoped one-shot opaque pick request/result channel.
  std::optional<OpaquePickRequest> opaquePickRequest{};
  std::optional<OpaquePickResult> opaquePickResult{};
  // Frame-scoped one-shot shadow inspect request/result channel.
  std::optional<ShadowInspectRequest> shadowInspectRequest{};
  std::optional<ShadowInspectResult> shadowInspectResult{};
  FrameSharedResources sharedResources{};
  TransparentContributionRegistry transparentContributors{};
  TextureHandle sharedDepthTexture{};
  const ResourceManager *resources = nullptr;
  double timeSeconds = 0.0;
  uint64_t frameIndex = 0;
};

[[nodiscard]] inline const RenderSettings &
renderSettingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? *frame.settings : kDefaultSettings;
}

[[nodiscard]] inline bool
expectsCurrentFrameSceneColor(const RenderFrameContext &frame) {
  return nuri::isValid(frame.sharedResources.sceneColorTexture);
}

inline void resetFrameSharedResources(RenderFrameContext &frame) {
  frame.sharedResources = {};
  frame.transparentContributors.clear();
}

[[nodiscard]] inline TextureHandle
resolveFrameDepthTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.sceneDepthTexture)) {
    return frame.sharedResources.sceneDepthTexture;
  }
  return frame.sharedDepthTexture;
}

[[nodiscard]] inline TextureHandle
resolveFrameColorTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.frameColorTexture)) {
    return frame.sharedResources.frameColorTexture;
  }
  return {};
}

[[nodiscard]] inline TextureHandle
resolveSceneColorTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.sceneColorTexture)) {
    return frame.sharedResources.sceneColorTexture;
  }
  return {};
}

[[nodiscard]] inline TextureHandle
resolveSceneColorMipTexture(const RenderFrameContext &frame,
                            uint32_t mipLevel) {
  switch (mipLevel) {
  case 0u:
    return nuri::isValid(frame.sharedResources.sceneColorTexture)
               ? frame.sharedResources.sceneColorTexture
               : TextureHandle{};
  case 1u:
    return nuri::isValid(frame.sharedResources.sceneColorHalfResTexture)
               ? frame.sharedResources.sceneColorHalfResTexture
               : TextureHandle{};
  case 2u:
    return nuri::isValid(frame.sharedResources.sceneColorQuarterResTexture)
               ? frame.sharedResources.sceneColorQuarterResTexture
               : TextureHandle{};
  default:
    return {};
  }
}

[[nodiscard]] inline TextureHandle
resolveSceneColorHalfResTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.sceneColorHalfResTexture)) {
    return frame.sharedResources.sceneColorHalfResTexture;
  }
  return {};
}

[[nodiscard]] inline TextureHandle
resolveSceneColorQuarterResTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.sceneColorQuarterResTexture)) {
    return frame.sharedResources.sceneColorQuarterResTexture;
  }
  return {};
}

[[nodiscard]] inline bool
resolveTransparentStageEnabled(const RenderFrameContext &frame) {
  return frame.sharedResources.transparentStageEnabled;
}

[[nodiscard]] inline TextureHandle
resolveHistoryColorReadTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.historyColorReadTexture)) {
    return frame.sharedResources.historyColorReadTexture;
  }
  return {};
}

[[nodiscard]] inline TextureHandle
resolveHistoryColorWriteTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.historyColorWriteTexture)) {
    return frame.sharedResources.historyColorWriteTexture;
  }
  return {};
}

[[nodiscard]] inline RenderGraphTextureId
resolveSceneDepthGraphTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.sceneDepthGraphTexture)) {
    return frame.sharedResources.sceneDepthGraphTexture;
  }
  return {};
}

[[nodiscard]] inline LightId
resolveSelectedLightId(const RenderFrameContext &frame) {
  if (frame.sharedResources.selectedLightId.has_value() &&
      isValid(*frame.sharedResources.selectedLightId)) {
    return *frame.sharedResources.selectedLightId;
  }
  return kInvalidLightId;
}

} // namespace nuri
