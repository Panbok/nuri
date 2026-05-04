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
#include <string_view>
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

enum class AntiAliasingMode : uint8_t {
  None = 0,
  TAA = 1,
  SpatialFallback = 2,
  MSAA4x = 3,
};

enum class AntiAliasingDebugView : uint8_t {
  None = 0,
  Settings = 1,
  MotionVectors = 2,
  VelocityMagnitude = 3,
  TAACurrentColor = 4,
  TAAPreviousHistory = 5,
  TAAResolved = 6,
  TAAHistoryValidity = 7,
  TAARejectionMask = 8,
  TAABlendFactor = 9,
  TAAClampDelta = 10,
  TAAPixelInspector = 11,
  TAAReactiveMask = 12,
  TAADisocclusionMask = 13,
  TAAVelocityDilation = 14,
  TAASceneColorHalfRes = 15,
  TAASceneColorQuarterRes = 16,
  TAATransmissionMipSource = 17,
  TAAReprojectedHistory = 18,
  TAAResolveConfidence = 19,
  TAAClampDiagnostics = 20,
  TAAPreviousVelocity = 21,
  TAAHdrWeight = 22,
  TAAHistoryFilterDelta = 23,
  TAADisocclusionFallback = 24,
  TAASplitCompare = 25,
  SpatialAAEdges = 26,
  SpatialAABlendWeights = 27,
  SpatialAACleanupMask = 28,
  SpatialAASplitCompare = 29,
};

enum class TemporalAAClampMode : uint8_t {
  Clamp = 0,
  Clip = 1,
  Variance = 2,
};

enum class TemporalAAHdrWeightingMode : uint8_t {
  None = 0,
  Luminance = 1,
  LogLuminance = 2,
};

enum class TemporalAAVelocityDilationMode : uint8_t {
  None = 0,
  ClosestDepth = 1,
  LargestMagnitude = 2,
};

enum class TemporalAAHistoryFilterMode : uint8_t {
  CatmullRom = 0,
  Bilinear = 1,
};

enum class TemporalHistoryResetReason : uint8_t {
  None = 0,
  FirstFrame = 1,
  HistoryResetRequested = 2,
  AntiAliasingModeChanged = 3,
  Resize = 4,
  ProjectionChanged = 5,
  RenderScaleChanged = 6,
  CameraCut = 7,
  InvalidHistoryTexture = 8,
};

enum class ShadowFilterMode : uint8_t {
  Hard = 0,
  PCF3x3 = 1,
  PoissonPCF = 2,
  PCSS = 3,
};

enum class ShadowQualityPreset : uint8_t {
  Custom = 0,
  Low = 1,
  Medium = 2,
  High = 3,
  Ultra = 4,
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
static constexpr uint32_t kMsaa4xSampleCount = 4u;
static constexpr Format kFrameCompositionMotionVectorFormat =
    Format::RG16_FLOAT;
static constexpr Format kFrameCompositionReactiveMaskFormat = Format::R32_FLOAT;
// Motion vectors are normalized UV-space history lookup offsets:
// historyUv = currentUv + velocity. Current and previous jitter are excluded.
static constexpr ClearColor kFrameCompositionMotionVectorClearValue{
    .r = 0.0f, .g = 0.0f, .b = 0.0f, .a = 0.0f};
static constexpr ClearColor kFrameCompositionReactiveMaskClearValue{
    .r = 0.0f, .g = 0.0f, .b = 0.0f, .a = 0.0f};
static constexpr uint32_t kTemporalJitterSequenceLength = 8u;
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

[[nodiscard]] constexpr AntiAliasingMode
sanitizeAntiAliasingMode(AntiAliasingMode mode) noexcept {
  switch (mode) {
  case AntiAliasingMode::None:
  case AntiAliasingMode::TAA:
  case AntiAliasingMode::SpatialFallback:
  case AntiAliasingMode::MSAA4x:
    return mode;
  default:
    return AntiAliasingMode::None;
  }
}

[[nodiscard]] constexpr AntiAliasingDebugView
sanitizeAntiAliasingDebugView(AntiAliasingDebugView view) noexcept {
  switch (view) {
  case AntiAliasingDebugView::None:
  case AntiAliasingDebugView::Settings:
  case AntiAliasingDebugView::MotionVectors:
  case AntiAliasingDebugView::VelocityMagnitude:
  case AntiAliasingDebugView::TAACurrentColor:
  case AntiAliasingDebugView::TAAPreviousHistory:
  case AntiAliasingDebugView::TAAResolved:
  case AntiAliasingDebugView::TAAHistoryValidity:
  case AntiAliasingDebugView::TAARejectionMask:
  case AntiAliasingDebugView::TAABlendFactor:
  case AntiAliasingDebugView::TAAClampDelta:
  case AntiAliasingDebugView::TAAPixelInspector:
  case AntiAliasingDebugView::TAAReactiveMask:
  case AntiAliasingDebugView::TAADisocclusionMask:
  case AntiAliasingDebugView::TAAVelocityDilation:
  case AntiAliasingDebugView::TAASceneColorHalfRes:
  case AntiAliasingDebugView::TAASceneColorQuarterRes:
  case AntiAliasingDebugView::TAATransmissionMipSource:
  case AntiAliasingDebugView::TAAReprojectedHistory:
  case AntiAliasingDebugView::TAAResolveConfidence:
  case AntiAliasingDebugView::TAAClampDiagnostics:
  case AntiAliasingDebugView::TAAPreviousVelocity:
  case AntiAliasingDebugView::TAAHdrWeight:
  case AntiAliasingDebugView::TAAHistoryFilterDelta:
  case AntiAliasingDebugView::TAADisocclusionFallback:
  case AntiAliasingDebugView::TAASplitCompare:
  case AntiAliasingDebugView::SpatialAAEdges:
  case AntiAliasingDebugView::SpatialAABlendWeights:
  case AntiAliasingDebugView::SpatialAACleanupMask:
  case AntiAliasingDebugView::SpatialAASplitCompare:
    return view;
  default:
    return AntiAliasingDebugView::None;
  }
}

[[nodiscard]] constexpr TemporalAAClampMode
sanitizeTemporalAAClampMode(TemporalAAClampMode mode) noexcept {
  switch (mode) {
  case TemporalAAClampMode::Clamp:
  case TemporalAAClampMode::Clip:
  case TemporalAAClampMode::Variance:
    return mode;
  default:
    return TemporalAAClampMode::Variance;
  }
}

[[nodiscard]] constexpr TemporalAAHdrWeightingMode
sanitizeTemporalAAHdrWeightingMode(TemporalAAHdrWeightingMode mode) noexcept {
  switch (mode) {
  case TemporalAAHdrWeightingMode::None:
  case TemporalAAHdrWeightingMode::Luminance:
  case TemporalAAHdrWeightingMode::LogLuminance:
    return mode;
  default:
    return TemporalAAHdrWeightingMode::Luminance;
  }
}

[[nodiscard]] constexpr TemporalAAVelocityDilationMode
sanitizeTemporalAAVelocityDilationMode(
    TemporalAAVelocityDilationMode mode) noexcept {
  switch (mode) {
  case TemporalAAVelocityDilationMode::None:
  case TemporalAAVelocityDilationMode::ClosestDepth:
  case TemporalAAVelocityDilationMode::LargestMagnitude:
    return mode;
  default:
    return TemporalAAVelocityDilationMode::ClosestDepth;
  }
}

[[nodiscard]] constexpr TemporalAAHistoryFilterMode
sanitizeTemporalAAHistoryFilterMode(TemporalAAHistoryFilterMode mode) noexcept {
  switch (mode) {
  case TemporalAAHistoryFilterMode::Bilinear:
  case TemporalAAHistoryFilterMode::CatmullRom:
    return mode;
  default:
    return TemporalAAHistoryFilterMode::CatmullRom;
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
    ShadowQualityPreset qualityPreset = ShadowQualityPreset::Custom;
    uint32_t cascadeCount = kMaxShadowCascades;
    uint32_t shadowMapSize = 2048;
    Format depthFormat = kDefaultShadowMapDepthFormat;
    float maxDistance = 150.0f;
    float maxDistanceFadeFraction = 0.0f;
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

  struct AntiAliasingDebugSettings {
    bool jitterEnabled = false;
    bool freezeJitter = false;
    bool resetHistoryRequested = false;
    bool logDiagnostics = false;
    bool spatialPostTaaCleanup = false;
    bool spatialPostMsaaCleanup = true;
    AntiAliasingDebugView view = AntiAliasingDebugView::None;
    float diagnosticLogIntervalSeconds = 0.25f;
    float taaJitterScale = 0.75f;
    float taaCurrentFrameWeight = 0.06f;
    float taaDepthDiscontinuityThreshold = 0.01f;
    float taaVelocityRejectionThreshold = 0.0015f;
    float taaVelocityBlendScale = 0.35f;
    float taaMotionCurrentWeight = 0.35f;
    float taaDisocclusionCurrentWeight = 0.65f;
    float taaClampCurrentWeight = 0.50f;
    float taaClampBlendAttenuation = 0.35f;
    float taaVarianceGamma = 1.50f;
    float taaHdrWeightStrength = 0.50f;
    float taaReactiveCurrentWeight = 0.85f;
    float taaReactiveStrength = 1.0f;
    float taaVelocityDilationDepthThreshold = 0.01f;
    TemporalAAClampMode taaClampMode = TemporalAAClampMode::Variance;
    TemporalAAHdrWeightingMode taaHdrWeightingMode =
        TemporalAAHdrWeightingMode::Luminance;
    TemporalAAVelocityDilationMode taaVelocityDilationMode =
        TemporalAAVelocityDilationMode::ClosestDepth;
    TemporalAAHistoryFilterMode taaHistoryFilterMode =
        TemporalAAHistoryFilterMode::CatmullRom;
  };

  struct AntiAliasingSettings {
    AntiAliasingMode mode = AntiAliasingMode::None;
    AntiAliasingDebugSettings debug{};
  };

  SkyboxSettings skybox{};
  OpaqueSettings opaque{};
  TransmissionSettings transmission{};
  TransparentSettings transparent{};
  DebugSettings debug{};
  ShadowSettings shadow{};
  AntiAliasingSettings antiAliasing{};
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

inline void
sanitizeAntiAliasingSettings(RenderSettings::AntiAliasingSettings &settings) {
  settings.mode = sanitizeAntiAliasingMode(settings.mode);
  settings.debug.view = sanitizeAntiAliasingDebugView(settings.debug.view);
  settings.debug.taaClampMode =
      sanitizeTemporalAAClampMode(settings.debug.taaClampMode);
  settings.debug.taaHdrWeightingMode =
      sanitizeTemporalAAHdrWeightingMode(settings.debug.taaHdrWeightingMode);
  settings.debug.taaVelocityDilationMode =
      sanitizeTemporalAAVelocityDilationMode(
          settings.debug.taaVelocityDilationMode);
  if (!std::isfinite(settings.debug.taaCurrentFrameWeight)) {
    settings.debug.taaCurrentFrameWeight = 0.06f;
  }
  settings.debug.taaCurrentFrameWeight =
      std::clamp(settings.debug.taaCurrentFrameWeight, 0.0f, 1.0f);
  if (!std::isfinite(settings.debug.diagnosticLogIntervalSeconds)) {
    settings.debug.diagnosticLogIntervalSeconds = 0.25f;
  }
  settings.debug.diagnosticLogIntervalSeconds =
      std::clamp(settings.debug.diagnosticLogIntervalSeconds, 0.033f, 5.0f);
  if (!std::isfinite(settings.debug.taaJitterScale)) {
    settings.debug.taaJitterScale = 0.75f;
  }
  settings.debug.taaJitterScale =
      std::clamp(settings.debug.taaJitterScale, 0.0f, 1.0f);
  if (!std::isfinite(settings.debug.taaDepthDiscontinuityThreshold)) {
    settings.debug.taaDepthDiscontinuityThreshold = 0.01f;
  }
  settings.debug.taaDepthDiscontinuityThreshold =
      std::clamp(settings.debug.taaDepthDiscontinuityThreshold, 0.0f, 1.0f);
  if (!std::isfinite(settings.debug.taaVelocityRejectionThreshold)) {
    settings.debug.taaVelocityRejectionThreshold = 0.0015f;
  }
  settings.debug.taaVelocityRejectionThreshold =
      std::clamp(settings.debug.taaVelocityRejectionThreshold, 0.0f, 1.0f);
  if (!std::isfinite(settings.debug.taaVelocityBlendScale)) {
    settings.debug.taaVelocityBlendScale = 0.35f;
  }
  settings.debug.taaVelocityBlendScale =
      std::clamp(settings.debug.taaVelocityBlendScale, 0.0f, 4.0f);
  if (!std::isfinite(settings.debug.taaMotionCurrentWeight)) {
    settings.debug.taaMotionCurrentWeight = 0.35f;
  }
  settings.debug.taaMotionCurrentWeight =
      std::clamp(settings.debug.taaMotionCurrentWeight, 0.0f, 1.0f);
  if (!std::isfinite(settings.debug.taaDisocclusionCurrentWeight)) {
    settings.debug.taaDisocclusionCurrentWeight = 0.65f;
  }
  settings.debug.taaDisocclusionCurrentWeight =
      std::clamp(settings.debug.taaDisocclusionCurrentWeight, 0.0f, 1.0f);
  if (!std::isfinite(settings.debug.taaClampCurrentWeight)) {
    settings.debug.taaClampCurrentWeight = 0.50f;
  }
  settings.debug.taaClampCurrentWeight =
      std::clamp(settings.debug.taaClampCurrentWeight, 0.0f, 1.0f);
  if (!std::isfinite(settings.debug.taaClampBlendAttenuation)) {
    settings.debug.taaClampBlendAttenuation = 0.35f;
  }
  settings.debug.taaClampBlendAttenuation =
      std::clamp(settings.debug.taaClampBlendAttenuation, 0.0f, 1.0f);
  if (!std::isfinite(settings.debug.taaVarianceGamma)) {
    settings.debug.taaVarianceGamma = 1.50f;
  }
  settings.debug.taaVarianceGamma =
      std::clamp(settings.debug.taaVarianceGamma, 0.0f, 4.0f);
  if (!std::isfinite(settings.debug.taaHdrWeightStrength)) {
    settings.debug.taaHdrWeightStrength = 0.50f;
  }
  settings.debug.taaHdrWeightStrength =
      std::clamp(settings.debug.taaHdrWeightStrength, 0.0f, 1.0f);
  if (!std::isfinite(settings.debug.taaReactiveCurrentWeight)) {
    settings.debug.taaReactiveCurrentWeight = 0.85f;
  }
  settings.debug.taaReactiveCurrentWeight =
      std::clamp(settings.debug.taaReactiveCurrentWeight, 0.0f, 1.0f);
  if (!std::isfinite(settings.debug.taaReactiveStrength)) {
    settings.debug.taaReactiveStrength = 1.0f;
  }
  settings.debug.taaReactiveStrength =
      std::clamp(settings.debug.taaReactiveStrength, 0.0f, 4.0f);
  if (!std::isfinite(settings.debug.taaVelocityDilationDepthThreshold)) {
    settings.debug.taaVelocityDilationDepthThreshold = 0.01f;
  }
  settings.debug.taaVelocityDilationDepthThreshold =
      std::clamp(settings.debug.taaVelocityDilationDepthThreshold, 0.0f, 1.0f);
  settings.debug.taaHistoryFilterMode =
      sanitizeTemporalAAHistoryFilterMode(settings.debug.taaHistoryFilterMode);
  if (settings.mode != AntiAliasingMode::TAA) {
    settings.debug.jitterEnabled = false;
    settings.debug.freezeJitter = false;
  }
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

[[nodiscard]] constexpr ShadowQualityPreset
sanitizeShadowQualityPreset(ShadowQualityPreset preset) noexcept {
  switch (preset) {
  case ShadowQualityPreset::Custom:
  case ShadowQualityPreset::Low:
  case ShadowQualityPreset::Medium:
  case ShadowQualityPreset::High:
  case ShadowQualityPreset::Ultra:
    return preset;
  default:
    return ShadowQualityPreset::Custom;
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

inline void sanitizeShadowSettings(RenderSettings::ShadowSettings &settings);

inline void applyShadowQualityPreset(RenderSettings::ShadowSettings &settings,
                                     ShadowQualityPreset preset) {
  settings.qualityPreset = sanitizeShadowQualityPreset(preset);
  switch (settings.qualityPreset) {
  case ShadowQualityPreset::Custom:
    break;
  case ShadowQualityPreset::Low:
    settings.cascadeCount = 1u;
    settings.shadowMapSize = 1024u;
    settings.depthFormat = kDefaultShadowMapDepthFormat;
    settings.maxDistance = 80.0f;
    settings.maxDistanceFadeFraction = 0.0f;
    settings.splitLambda = 0.35f;
    settings.constantBias = 0.0008f;
    settings.slopeBias = 2.0f;
    settings.normalBias = 0.25f;
    settings.filterMode = ShadowFilterMode::PCF3x3;
    settings.pcfSampleCount = 9u;
    settings.sdsmMode = ShadowSdsmMode::Disabled;
    settings.debug.fixedPoissonRotation = false;
    break;
  case ShadowQualityPreset::Medium:
    settings.cascadeCount = 3u;
    settings.shadowMapSize = 2048u;
    settings.depthFormat = kDefaultShadowMapDepthFormat;
    settings.maxDistance = 120.0f;
    settings.maxDistanceFadeFraction = 0.0f;
    settings.splitLambda = 0.30f;
    settings.constantBias = 0.0006f;
    settings.slopeBias = 1.75f;
    settings.normalBias = 0.35f;
    settings.filterMode = ShadowFilterMode::PoissonPCF;
    settings.pcfSampleCount = 16u;
    settings.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
    settings.sdsmReductionBackend = ShadowSdsmReductionBackend::Auto;
    settings.debug.fixedPoissonRotation = true;
    break;
  case ShadowQualityPreset::High:
    settings.cascadeCount = 4u;
    settings.shadowMapSize = 4096u;
    settings.depthFormat = kDefaultShadowMapDepthFormat;
    settings.maxDistance = 150.0f;
    settings.maxDistanceFadeFraction = 0.0f;
    settings.splitLambda = 0.25f;
    settings.constantBias = 0.0005f;
    settings.slopeBias = 1.5f;
    settings.normalBias = 0.50f;
    settings.filterMode = ShadowFilterMode::PoissonPCF;
    settings.pcfSampleCount = 24u;
    settings.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
    settings.sdsmReductionBackend = ShadowSdsmReductionBackend::Auto;
    settings.debug.fixedPoissonRotation = true;
    break;
  case ShadowQualityPreset::Ultra:
    settings.cascadeCount = 4u;
    settings.shadowMapSize = 8192u;
    settings.depthFormat = Format::D32_FLOAT;
    settings.maxDistance = 220.0f;
    settings.maxDistanceFadeFraction = 0.0f;
    settings.splitLambda = 0.50f;
    settings.constantBias = 0.00035f;
    settings.slopeBias = 1.15f;
    settings.normalBias = 0.40f;
    settings.filterMode = ShadowFilterMode::PoissonPCF;
    settings.pcfSampleCount = 32u;
    settings.pcssBlockerSampleCount = 16u;
    settings.pcssFilterSampleCount = 32u;
    settings.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
    settings.sdsmReductionBackend = ShadowSdsmReductionBackend::Auto;
    settings.debug.fixedPoissonRotation = true;
    break;
  }
  sanitizeShadowSettings(settings);
}

inline void sanitizeShadowSettings(RenderSettings::ShadowSettings &settings) {
  settings.qualityPreset = sanitizeShadowQualityPreset(settings.qualityPreset);
  settings.cascadeCount =
      std::clamp(settings.cascadeCount, 1u, kMaxShadowCascades);
  settings.shadowMapSize = std::max(settings.shadowMapSize, 1u);
  switch (settings.depthFormat) {
  case Format::D16_UNORM:
  case Format::D32_FLOAT:
    break;
  default:
    settings.depthFormat = kDefaultShadowMapDepthFormat;
    break;
  }
  settings.maxDistance =
      std::isfinite(settings.maxDistance) && settings.maxDistance > 0.0f
          ? settings.maxDistance
          : 150.0f;
  settings.maxDistanceFadeFraction =
      std::isfinite(settings.maxDistanceFadeFraction)
          ? std::clamp(settings.maxDistanceFadeFraction, 0.0f, 1.0f)
          : 0.0f;
  settings.splitLambda = std::clamp(settings.splitLambda, 0.0f, 1.0f);
  settings.cascadeBlendFraction =
      std::clamp(settings.cascadeBlendFraction, 0.0f, 1.0f);
  settings.constantBias =
      std::isfinite(settings.constantBias) ? settings.constantBias : 0.0005f;
  settings.slopeBias = std::isfinite(settings.slopeBias)
                           ? std::max(settings.slopeBias, 0.0f)
                           : 1.5f;
  settings.normalBias = std::isfinite(settings.normalBias)
                            ? std::max(settings.normalBias, 0.0f)
                            : 0.0f;
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

[[nodiscard]] inline uint32_t
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
  default:
    // sanitizeShadowFilterMode normalizes all values; this is only defensive.
#if defined(_MSC_VER)
    __assume(false);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_unreachable();
#else
    return 1u;
#endif
  }
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
  // Current projection used by scene rendering. This is jittered when temporal
  // jitter is active; use currentUnjitteredProj for non-temporal overlays.
  glm::mat4 proj{1.0f};
  glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
  float aspectRatio = 1.0f;
  ProjectionType projectionType = ProjectionType::Perspective;
  float nearPlane = 0.1f;
  float farPlane = 1000.0f;
  float fovYRadians = glm::radians(60.0f);
  float orthoHeight = 10.0f;
  glm::mat4 currentUnjitteredProj{1.0f};
  glm::mat4 currentUnjitteredViewProj{1.0f};
  glm::mat4 currentJitteredViewProj{1.0f};
  glm::mat4 previousUnjitteredViewProj{1.0f};
  glm::mat4 previousJitteredViewProj{1.0f};
  glm::vec2 jitterPixelOffset{0.0f};
  glm::vec2 previousJitterPixelOffset{0.0f};
  glm::vec4 previousCameraPos{0.0f, 0.0f, 0.0f, 1.0f};
  glm::uvec2 renderExtent{0u, 0u};
  uint32_t jitterIndex = 0u;
  uint32_t jitterSequenceLength = kTemporalJitterSequenceLength;
  uint32_t framesSinceHistoryReset = 0u;
  uint32_t historyResetCount = 0u;
  uint32_t jitterOutOfBoundsCount = 0u;
  TemporalHistoryResetReason historyResetReason =
      TemporalHistoryResetReason::None;
  bool temporalDataValid = false;
  bool jitterEnabled = false;
  bool jitterFrozen = false;
  bool jitterOutOfBounds = false;
  bool historyValid = false;
};

struct TemporalCameraHistoryState {
  glm::mat4 previousUnjitteredViewProj{1.0f};
  glm::mat4 previousJitteredViewProj{1.0f};
  glm::vec2 previousJitterPixelOffset{0.0f};
  glm::vec4 previousCameraPos{0.0f, 0.0f, 0.0f, 1.0f};
  glm::uvec2 previousRenderExtent{0u, 0u};
  glm::vec2 previousRenderScale{1.0f, 1.0f};
  ProjectionType previousProjectionType = ProjectionType::Perspective;
  AntiAliasingMode previousAntiAliasingMode = AntiAliasingMode::None;
  float previousAspectRatio = 1.0f;
  float previousNearPlane = 0.1f;
  float previousFarPlane = 1000.0f;
  float previousFovYRadians = glm::radians(60.0f);
  float previousOrthoHeight = 10.0f;
  uint32_t nextJitterIndex = 0u;
  uint32_t currentJitterIndex = 0u;
  uint32_t framesSinceHistoryReset = 0u;
  uint32_t historyResetCount = 0u;
  uint32_t jitterOutOfBoundsCount = 0u;
  bool initialized = false;
};

struct TemporalCameraFrameDesc {
  glm::uvec2 renderExtent{0u, 0u};
  glm::vec2 renderScale{1.0f, 1.0f};
  bool cameraCutRequested = false;
  bool historyTextureValid = true;
};

[[nodiscard]] inline CameraFrameState makeCameraFrameState(const Camera &camera,
                                                           float aspectRatio) {
  CameraFrameState state{};
  state.view = camera.viewMatrix();
  state.proj = camera.projectionMatrix(aspectRatio);
  state.currentUnjitteredProj = state.proj;
  state.currentUnjitteredViewProj = state.proj * state.view;
  state.currentJitteredViewProj = state.currentUnjitteredViewProj;
  state.previousUnjitteredViewProj = state.currentUnjitteredViewProj;
  state.previousJitteredViewProj = state.currentJitteredViewProj;
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

[[nodiscard]] inline float halton(uint32_t index, uint32_t base) noexcept {
  float result = 0.0f;
  float invBase = 1.0f / static_cast<float>(base);
  float fraction = invBase;
  while (index > 0u) {
    result += static_cast<float>(index % base) * fraction;
    index /= base;
    fraction *= invBase;
  }
  return result;
}

[[nodiscard]] inline glm::vec2
temporalJitterPixelOffset(uint32_t jitterIndex) noexcept {
  const uint32_t sample = (jitterIndex % kTemporalJitterSequenceLength) + 1u;
  return glm::vec2(halton(sample, 2u) - 0.5f, halton(sample, 3u) - 0.5f);
}

[[nodiscard]] inline bool jitterOffsetWithinHalfPixel(glm::vec2 offset) {
  return std::abs(offset.x) <= 0.5f && std::abs(offset.y) <= 0.5f;
}

[[nodiscard]] inline glm::mat4
applyProjectionJitter(glm::mat4 projection, glm::vec2 jitterPixelOffset,
                      glm::uvec2 renderExtent, ProjectionType projectionType) {
  if (renderExtent.x == 0u || renderExtent.y == 0u) {
    return projection;
  }

  const glm::vec2 jitterNdc{
      2.0f * jitterPixelOffset.x / static_cast<float>(renderExtent.x),
      2.0f * jitterPixelOffset.y / static_cast<float>(renderExtent.y),
  };
  if (projectionType == ProjectionType::Orthographic) {
    projection[3][0] += jitterNdc.x;
    projection[3][1] += jitterNdc.y;
  } else {
    projection[2][0] -= jitterNdc.x;
    projection[2][1] -= jitterNdc.y;
  }
  return projection;
}

[[nodiscard]] inline bool
projectionMetadataChanged(const TemporalCameraHistoryState &history,
                          const CameraFrameState &state) {
  constexpr float kProjectionEpsilon = 1.0e-6f;
  return history.previousProjectionType != state.projectionType ||
         std::abs(history.previousAspectRatio - state.aspectRatio) >
             kProjectionEpsilon ||
         std::abs(history.previousNearPlane - state.nearPlane) >
             kProjectionEpsilon ||
         std::abs(history.previousFarPlane - state.farPlane) >
             kProjectionEpsilon ||
         std::abs(history.previousFovYRadians - state.fovYRadians) >
             kProjectionEpsilon ||
         std::abs(history.previousOrthoHeight - state.orthoHeight) >
             kProjectionEpsilon;
}

[[nodiscard]] inline bool
temporalRenderScaleChanged(const TemporalCameraHistoryState &history,
                           glm::vec2 renderScale) {
  constexpr float kRenderScaleEpsilon = 1.0e-6f;
  return std::abs(history.previousRenderScale.x - renderScale.x) >
             kRenderScaleEpsilon ||
         std::abs(history.previousRenderScale.y - renderScale.y) >
             kRenderScaleEpsilon;
}

[[nodiscard]] inline CameraFrameState makeTemporalCameraFrameState(
    const Camera &camera, float aspectRatio,
    const RenderSettings::AntiAliasingSettings &antiAliasing,
    const TemporalCameraFrameDesc &desc, TemporalCameraHistoryState &history) {
  CameraFrameState state = makeCameraFrameState(camera, aspectRatio);
  state.temporalDataValid = true;
  state.renderExtent = desc.renderExtent;

  const AntiAliasingMode mode = sanitizeAntiAliasingMode(antiAliasing.mode);
  const bool taaSelected = mode == AntiAliasingMode::TAA;
  const bool hasRenderExtent =
      desc.renderExtent.x > 0u && desc.renderExtent.y > 0u;
  state.jitterEnabled =
      taaSelected && antiAliasing.debug.jitterEnabled && hasRenderExtent;
  state.jitterFrozen = state.jitterEnabled && antiAliasing.debug.freezeJitter;

  TemporalHistoryResetReason resetReason = TemporalHistoryResetReason::None;
  if (!history.initialized) {
    resetReason = TemporalHistoryResetReason::FirstFrame;
  } else if (antiAliasing.debug.resetHistoryRequested) {
    resetReason = TemporalHistoryResetReason::HistoryResetRequested;
  } else if (history.previousAntiAliasingMode != mode) {
    resetReason = TemporalHistoryResetReason::AntiAliasingModeChanged;
  } else if (desc.cameraCutRequested) {
    resetReason = TemporalHistoryResetReason::CameraCut;
  } else if (history.previousRenderExtent != desc.renderExtent) {
    resetReason = TemporalHistoryResetReason::Resize;
  } else if (temporalRenderScaleChanged(history, desc.renderScale)) {
    resetReason = TemporalHistoryResetReason::RenderScaleChanged;
  } else if (!desc.historyTextureValid) {
    resetReason = TemporalHistoryResetReason::InvalidHistoryTexture;
  } else if (projectionMetadataChanged(history, state)) {
    resetReason = TemporalHistoryResetReason::ProjectionChanged;
  }

  if (state.jitterEnabled) {
    if (state.jitterFrozen && history.initialized) {
      state.jitterIndex = history.currentJitterIndex;
    } else {
      state.jitterIndex =
          history.nextJitterIndex % kTemporalJitterSequenceLength;
    }
    const float jitterScale =
        std::isfinite(antiAliasing.debug.taaJitterScale)
            ? std::clamp(antiAliasing.debug.taaJitterScale, 0.0f, 1.0f)
            : 0.75f;
    state.jitterPixelOffset =
        temporalJitterPixelOffset(state.jitterIndex) * jitterScale;
    if (!state.jitterFrozen) {
      history.nextJitterIndex =
          (state.jitterIndex + 1u) % kTemporalJitterSequenceLength;
    }
  } else {
    state.jitterIndex = 0u;
    state.jitterPixelOffset = glm::vec2(0.0f);
  }

  state.jitterOutOfBounds =
      !jitterOffsetWithinHalfPixel(state.jitterPixelOffset);
  if (state.jitterOutOfBounds) {
    ++history.jitterOutOfBoundsCount;
  }
  state.jitterOutOfBoundsCount = history.jitterOutOfBoundsCount;
  state.proj = applyProjectionJitter(state.currentUnjitteredProj,
                                     state.jitterPixelOffset, desc.renderExtent,
                                     state.projectionType);
  state.currentJitteredViewProj = state.proj * state.view;

  const bool resetHistory = resetReason != TemporalHistoryResetReason::None;
  state.historyValid = taaSelected && history.initialized && !resetHistory;
  state.previousUnjitteredViewProj = state.historyValid
                                         ? history.previousUnjitteredViewProj
                                         : state.currentUnjitteredViewProj;
  state.previousJitteredViewProj = state.historyValid
                                       ? history.previousJitteredViewProj
                                       : state.currentJitteredViewProj;
  state.previousJitterPixelOffset = state.historyValid
                                        ? history.previousJitterPixelOffset
                                        : state.jitterPixelOffset;
  state.previousCameraPos =
      state.historyValid ? history.previousCameraPos : state.cameraPos;
  state.historyResetReason = resetReason;
  if (resetHistory) {
    ++history.historyResetCount;
    history.framesSinceHistoryReset = 0u;
  } else if (taaSelected) {
    ++history.framesSinceHistoryReset;
  }
  state.framesSinceHistoryReset = history.framesSinceHistoryReset;
  state.historyResetCount = history.historyResetCount;

  history.previousUnjitteredViewProj = state.currentUnjitteredViewProj;
  history.previousJitteredViewProj = state.currentJitteredViewProj;
  history.previousJitterPixelOffset = state.jitterPixelOffset;
  history.previousCameraPos = state.cameraPos;
  history.previousRenderExtent = desc.renderExtent;
  history.previousRenderScale = desc.renderScale;
  history.previousProjectionType = state.projectionType;
  history.previousAntiAliasingMode = mode;
  history.previousAspectRatio = state.aspectRatio;
  history.previousNearPlane = state.nearPlane;
  history.previousFarPlane = state.farPlane;
  history.previousFovYRadians = state.fovYRadians;
  history.previousOrthoHeight = state.orthoHeight;
  history.currentJitterIndex = state.jitterIndex;
  history.initialized = true;

  return state;
}

[[nodiscard]] inline const glm::mat4 &
cameraCurrentUnjitteredProjection(const CameraFrameState &camera) noexcept {
  return camera.temporalDataValid ? camera.currentUnjitteredProj : camera.proj;
}

[[nodiscard]] inline glm::mat4
cameraCurrentUnjitteredViewProjection(const CameraFrameState &camera) noexcept {
  return camera.temporalDataValid ? camera.currentUnjitteredViewProj
                                  : camera.proj * camera.view;
}

[[nodiscard]] constexpr std::string_view
temporalHistoryResetReasonName(TemporalHistoryResetReason reason) noexcept {
  switch (reason) {
  case TemporalHistoryResetReason::None:
    return "None";
  case TemporalHistoryResetReason::FirstFrame:
    return "First Frame";
  case TemporalHistoryResetReason::HistoryResetRequested:
    return "History Reset Requested";
  case TemporalHistoryResetReason::AntiAliasingModeChanged:
    return "AA Mode Changed";
  case TemporalHistoryResetReason::Resize:
    return "Resize";
  case TemporalHistoryResetReason::ProjectionChanged:
    return "Projection Changed";
  case TemporalHistoryResetReason::RenderScaleChanged:
    return "Render Scale Changed";
  case TemporalHistoryResetReason::CameraCut:
    return "Camera Cut";
  case TemporalHistoryResetReason::InvalidHistoryTexture:
    return "Invalid History Texture";
  }
  return "Unknown";
}

struct alignas(16) ShadowCascadeGpuData {
  glm::mat4 lightViewProj{1.0f};
  glm::mat4 lightView{1.0f};
  glm::vec4 splitDepthTexelSize{0.0f};
  glm::vec4 uvScaleBias{1.0f, 1.0f, 0.0f, 0.0f};
  glm::vec4 biasParams{0.0f};
  // x: PCSS receiver-depth world scale, y/z: search/filter radius clamps.
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
  uint32_t staticBatchFullEmitCount = 0u;
  uint32_t staticLightGridQueryCellCount = 0u;
  uint32_t staticLightGridCandidateCount = 0u;
  uint32_t staticLightGridFallbackScanCount = 0u;
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
  float maxDistanceFadeStart = 0.0f;
  float maxDistanceFadeEnd = 0.0f;
  uint32_t rawSamplerId = kInvalidShadowBindlessIndex;
  uint32_t compareSamplerId = kInvalidShadowBindlessIndex;
  ShadowSdsmDebugFrameData sdsm{};
  std::array<ShadowCascadeDebugFrameData, kMaxShadowCascades> cascades{};
};

struct ForwardSceneFrameData {
  glm::mat4 view{1.0f};
  // Current scene projection. This is jittered when temporal jitter is active.
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
  float gpuTimeMs = 0.0f;
  uint64_t gpuTimingSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint32_t gpuTimingAvailable = 0u;
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
  uint32_t staticBatchTemplateCount = 0;
  uint32_t shadowBatchEntryCount = 0;
  uint32_t shadowInstanceRemapCount = 0;
  uint32_t staticBatchFullEmitCount = 0;
  uint32_t staticLightGridQueryCount = 0;
  uint32_t staticLightGridFallbackScanCount = 0;
  uint32_t staticLightGridQueryCellCount = 0;
  uint32_t staticLightGridCandidateCount = 0;
  uint32_t staticOnlyCandidateCount = 0;
  uint32_t reusedStaticOnlyCascadeCount = 0;
  uint32_t staticOnlyReuseMissStaticCacheRebuiltCount = 0;
  uint32_t staticOnlyReuseMissDynamicCasterCount = 0;
  uint32_t staticOnlyReuseMissNoPreviousCount = 0;
  uint32_t staticOnlyReuseMissRasterStateChangedCount = 0;
  uint32_t staticOnlyReuseMissAdaptiveRefreshCount = 0;
  uint64_t cascadeTextureBytes = 0;
  float minCascadeTexelWorldSize = 0.0f;
  float averageCascadeTexelWorldSize = 0.0f;
  float maxCascadeTexelWorldSize = 0.0f;
  float farCascadeTexelWorldSize = 0.0f;
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

struct AntiAliasingFrameMetrics {
  glm::vec2 jitterPixelOffset{0.0f};
  glm::vec2 previousJitterPixelOffset{0.0f};
  glm::vec2 jitterDeltaPixelOffset{0.0f};
  glm::vec4 cameraPosition{0.0f, 0.0f, 0.0f, 1.0f};
  glm::vec4 previousCameraPosition{0.0f, 0.0f, 0.0f, 1.0f};
  Format motionVectorFormat = Format::Count;
  uint32_t jitterIndex = 0u;
  uint32_t jitterSequenceLength = kTemporalJitterSequenceLength;
  uint32_t jitterOutOfBoundsCount = 0u;
  uint32_t historyResetCount = 0u;
  uint32_t framesSinceHistoryReset = 0u;
  uint32_t motionVectorWidth = 0u;
  uint32_t motionVectorHeight = 0u;
  uint32_t motionVectorTextureCount = 0u;
  uint32_t motionVectorAllocationCount = 0u;
  uint32_t motionVectorReallocationCount = 0u;
  uint32_t motionVectorRg32FallbackCount = 0u;
  uint32_t motionVectorClearPassCount = 0u;
  uint32_t velocityPassCount = 0u;
  uint32_t velocityDrawCount = 0u;
  uint32_t velocityInstanceCount = 0u;
  uint32_t velocityPreviousTransformValidCount = 0u;
  uint32_t velocityMissingPreviousTransformCount = 0u;
  uint32_t velocityAnimatedResponsiveCount = 0u;
  uint32_t velocityTessellatedSkippedDrawCount = 0u;
  uint32_t velocityDebugPassCount = 0u;
  uint32_t reactiveMaskWidth = 0u;
  uint32_t reactiveMaskHeight = 0u;
  uint32_t reactiveMaskTextureCount = 0u;
  uint32_t reactiveMaskAllocationCount = 0u;
  uint32_t reactiveMaskReallocationCount = 0u;
  uint32_t reactiveMaskPassCount = 0u;
  uint32_t reactiveMaskDrawCount = 0u;
  uint32_t reactiveAlphaMaskedDrawCount = 0u;
  uint32_t reactiveSkippedTessellatedDrawCount = 0u;
  uint32_t taaResolvePassCount = 0u;
  uint32_t taaCopyBackPassCount = 0u;
  uint32_t taaPostResolveSceneColorMipPassCount = 0u;
  uint32_t taaTransmissionStaleSceneColorFrameCount = 0u;
  uint32_t taaResolveGpuTimingAvailable = 0u;
  uint32_t taaDebugGpuTimingAvailable = 0u;
  uint32_t taaSceneColorDownsampleGpuTimingAvailable = 0u;
  uint32_t taaTransmissionGpuTimingAvailable = 0u;
  uint32_t taaTransmissionMipDebugPassCount = 0u;
  uint32_t taaTransparentPostTaaDrawCount = 0u;
  uint32_t taaTransparentPostTaaMeshDrawCount = 0u;
  uint32_t taaTransparentPostTaaContributorDrawCount = 0u;
  uint32_t taaTransparentPostTaaFixedDrawCount = 0u;
  uint32_t taaOverlayPostTaaDrawCount = 0u;
  uint32_t taaOverlayHistoryContaminationFrameCount = 0u;
  uint32_t taaResolveWidth = 0u;
  uint32_t taaResolveHeight = 0u;
  uint32_t spatialAAWidth = 0u;
  uint32_t spatialAAHeight = 0u;
  uint32_t spatialAAPassCount = 0u;
  uint32_t spatialAAEdgePassCount = 0u;
  uint32_t spatialAABlendPassCount = 0u;
  uint32_t spatialAANeighborhoodPassCount = 0u;
  uint32_t spatialAACopyBackPassCount = 0u;
  uint32_t spatialAAFallbackFrameCount = 0u;
  uint32_t spatialAACleanupFrameCount = 0u;
  uint32_t spatialAADebugPassCount = 0u;
  uint32_t spatialAATextureCount = 0u;
  uint32_t spatialAALutTextureCount = 0u;
  uint32_t spatialAAGpuTimingAvailable = 0u;
  uint32_t msaaSampleCount = 1u;
  uint32_t msaaWidth = 0u;
  uint32_t msaaHeight = 0u;
  uint32_t msaaResolvePassCount = 0u;
  uint32_t msaaAlphaMaskedDrawCount = 0u;
  uint32_t msaaResolveGpuTimingAvailable = 0u;
  uint64_t taaResolveGpuTimingSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t taaDebugGpuTimingSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t taaSceneColorDownsampleGpuTimingSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t taaTransmissionGpuTimingSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t spatialAAGpuTimingSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t msaaResolveGpuTimingSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint32_t taaCurrentFallbackFrameCount = 0u;
  uint64_t motionVectorTextureBytes = 0u;
  uint64_t previousMotionVectorTextureBytes = 0u;
  uint64_t motionVectorTotalBytes = 0u;
  uint64_t motionVectorClearBytes = 0u;
  uint64_t velocityPassBandwidthEstimateBytes = 0u;
  uint64_t velocityDebugBandwidthEstimateBytes = 0u;
  uint64_t reactiveMaskTextureBytes = 0u;
  uint64_t reactiveMaskTotalBytes = 0u;
  uint64_t reactiveMaskPassBandwidthEstimateBytes = 0u;
  uint64_t taaHistoryBandwidthEstimateBytes = 0u;
  uint64_t spatialAATextureBytes = 0u;
  uint64_t spatialAATotalBytes = 0u;
  uint64_t spatialAALutTextureBytes = 0u;
  uint64_t spatialAABandwidthEstimateBytes = 0u;
  uint64_t msaaColorTextureBytes = 0u;
  uint64_t msaaDepthTextureBytes = 0u;
  uint64_t msaaTotalBytes = 0u;
  uint64_t msaaResolveBandwidthEstimateBytes = 0u;
  float velocityAverageObjectMotion = 0.0f;
  float velocityMaxObjectMotion = 0.0f;
  float velocityEstimatedAverageMagnitude = 0.0f;
  float velocityEstimatedMaxMagnitude = 0.0f;
  float velocityStaticResidualEstimate = 0.0f;
  float velocityCameraMatrixDelta = 0.0f;
  float cameraPositionDelta = 0.0f;
  float jitterDeltaMagnitude = 0.0f;
  float velocityMissingPreviousRatio = 0.0f;
  float velocityEdgeDiscontinuityEstimate = 0.0f;
  float taaJitterScale = 0.75f;
  float taaCurrentFrameWeight = 0.06f;
  float taaHistoryFrameWeight = 0.94f;
  float taaHistoryValidPercent = 0.0f;
  float taaOutOfBoundsReprojectionPercent = 0.0f;
  float taaDepthDiscontinuityThreshold = 0.01f;
  float taaVelocityRejectionThreshold = 0.0015f;
  float taaVelocityBlendScale = 0.35f;
  float taaMotionCurrentWeight = 0.35f;
  float taaDisocclusionCurrentWeight = 0.65f;
  float taaClampCurrentWeight = 0.50f;
  float taaClampBlendAttenuation = 0.35f;
  float taaVarianceGamma = 1.50f;
  float taaHdrWeightStrength = 0.50f;
  float taaReactiveCurrentWeight = 0.85f;
  float taaReactiveStrength = 1.0f;
  float taaReactiveCoverageEstimate = 0.0f;
  float taaAlphaMaskedCoverageEstimate = 0.0f;
  float taaDisocclusionRejectionEstimate = 0.0f;
  float taaVelocityDilationDepthThreshold = 0.01f;
  float taaVelocityDilationAffectedEstimate = 0.0f;
  float taaResolveGpuTimeMs = 0.0f;
  float taaDebugGpuTimeMs = 0.0f;
  float taaSceneColorDownsampleGpuTimeMs = 0.0f;
  float taaTransmissionGpuTimeMs = 0.0f;
  float taaTransmissionFlickerEstimate = 0.0f;
  float taaTransparentEdgeJitterEstimate = 0.0f;
  float spatialAAGpuTimeMs = 0.0f;
  float msaaResolveGpuTimeMs = 0.0f;
  float spatialAAEdgePixelEstimate = 0.0f;
  float spatialAAModifiedPixelEstimate = 0.0f;
  TemporalAAClampMode taaClampMode = TemporalAAClampMode::Variance;
  TemporalAAHdrWeightingMode taaHdrWeightingMode =
      TemporalAAHdrWeightingMode::Luminance;
  TemporalAAVelocityDilationMode taaVelocityDilationMode =
      TemporalAAVelocityDilationMode::ClosestDepth;
  TemporalAAHistoryFilterMode taaHistoryFilterMode =
      TemporalAAHistoryFilterMode::CatmullRom;
  TemporalHistoryResetReason historyResetReason =
      TemporalHistoryResetReason::None;
  bool jitterEnabled = false;
  bool jitterFrozen = false;
  bool taaQualityValidationInvalidatedByFrozenJitter = false;
  bool jitterOutOfBounds = false;
  bool historyValid = false;
  bool temporalDataValid = false;
  bool motionVectorAllocated = false;
  bool motionVectorFormatSupported = false;
  bool previousMotionVectorValid = false;
  bool motionVectorGraphPublished = false;
  bool previousMotionVectorGraphPublished = false;
  bool reactiveMaskAllocated = false;
  bool reactiveMaskGraphPublished = false;
  bool reactiveMaskFormatSupported = false;
  bool opaqueVelocityGenerated = false;
  bool velocityDebugViewRendered = false;
  bool previousTransformCacheValid = false;
  bool taaResolvedSceneColorPublished = false;
  bool taaDebugViewRendered = false;
  bool taaHistoryValidityDebugViewRendered = false;
  bool taaOutOfBoundsFallbackEnabled = false;
  bool taaBilinearHistorySampling = false;
  bool taaDepthRejectionEnabled = false;
  bool taaVelocityRejectionEnabled = false;
  bool taaPreviousVelocityDisocclusionEnabled = false;
  bool taaNeighborhoodClampEnabled = false;
  bool taaAdaptiveBlendEnabled = false;
  bool taaClampBlendAttenuationEnabled = false;
  bool taaNeighborhoodFallbackEnabled = false;
  bool taaHdrWeightingEnabled = false;
  bool taaPixelInspectorDebugViewRendered = false;
  bool taaReactiveMaskEnabled = false;
  bool taaReactiveMaskDebugViewRendered = false;
  bool taaDisocclusionMaskDebugViewRendered = false;
  bool taaVelocityDilationEnabled = false;
  bool taaVelocityDilationDebugViewRendered = false;
  bool taaPreviousVelocityDebugViewRendered = false;
  bool taaHdrWeightDebugViewRendered = false;
  bool taaHistoryFilterDeltaDebugViewRendered = false;
  bool taaDisocclusionFallbackDebugViewRendered = false;
  bool taaSplitCompareDebugViewRendered = false;
  bool taaPostResolveSceneColorMipChainGenerated = false;
  bool taaTransmissionPostResolveSceneColorConsumed = false;
  bool taaSceneColorMipDebugViewRendered = false;
  bool taaTransmissionMipDebugViewRendered = false;
  bool taaTransparentEdgeJitterTracked = false;
  bool spatialAAEnabled = false;
  bool spatialAAFallbackActive = false;
  bool spatialAACleanupActive = false;
  bool msaaEnabled = false;
  bool msaaColorAllocated = false;
  bool msaaDepthAllocated = false;
  bool msaaColorGraphPublished = false;
  bool msaaDepthGraphPublished = false;
  bool msaaColorResolveTargetBound = false;
  bool msaaDepthResolveTargetBound = false;
  bool msaaAlphaToCoverageEnabled = false;
  bool msaaSampleShadingEnabled = false;
  bool msaaSpatialCleanupEnabled = false;
  bool msaaSpatialCleanupActive = false;
  bool spatialAAEdgesDebugViewRendered = false;
  bool spatialAABlendWeightsDebugViewRendered = false;
  bool spatialAACleanupMaskDebugViewRendered = false;
  bool spatialAASplitCompareDebugViewRendered = false;
};

[[nodiscard]] inline AntiAliasingFrameMetrics
makeAntiAliasingFrameMetrics(const CameraFrameState &camera) noexcept {
  const glm::vec2 jitterDelta =
      camera.jitterPixelOffset - camera.previousJitterPixelOffset;
  return AntiAliasingFrameMetrics{
      .jitterPixelOffset = camera.jitterPixelOffset,
      .previousJitterPixelOffset = camera.previousJitterPixelOffset,
      .jitterDeltaPixelOffset = jitterDelta,
      .cameraPosition = camera.cameraPos,
      .previousCameraPosition = camera.previousCameraPos,
      .jitterIndex = camera.jitterIndex,
      .jitterSequenceLength = camera.jitterSequenceLength,
      .jitterOutOfBoundsCount = camera.jitterOutOfBoundsCount,
      .historyResetCount = camera.historyResetCount,
      .framesSinceHistoryReset = camera.framesSinceHistoryReset,
      .cameraPositionDelta = glm::length(glm::vec3(camera.cameraPos) -
                                         glm::vec3(camera.previousCameraPos)),
      .jitterDeltaMagnitude = glm::length(jitterDelta),
      .historyResetReason = camera.historyResetReason,
      .jitterEnabled = camera.jitterEnabled,
      .jitterFrozen = camera.jitterFrozen,
      .taaQualityValidationInvalidatedByFrozenJitter = camera.jitterFrozen,
      .jitterOutOfBounds = camera.jitterOutOfBounds,
      .historyValid = camera.historyValid,
      .temporalDataValid = camera.temporalDataValid,
  };
}

struct RenderFrameMetrics {
  uint64_t frameIndex = 0u;
  OpaqueFrameMetrics opaque{};
  ShadowFrameMetrics shadow{};
  AntiAliasingFrameMetrics antiAliasing{};
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
  ReactiveMask = 1u << 8u,
  MsaaSceneColor = 1u << 9u,
  MsaaSceneDepth = 1u << 10u,
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
  TextureHandle msaaSceneDepthTexture{};
  RenderGraphTextureId msaaSceneDepthGraphTexture{};
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
  TextureHandle msaaSceneColorTexture{};
  TextureHandle sceneColorHalfResTexture{};
  TextureHandle sceneColorQuarterResTexture{};
  RenderGraphTextureId sceneColorGraphTexture{};
  RenderGraphTextureId msaaSceneColorGraphTexture{};
  TextureHandle frameColorTexture{};
  RenderGraphTextureId frameColorGraphTexture{};
  TextureHandle historyColorReadTexture{};
  TextureHandle historyColorWriteTexture{};
  TextureHandle motionVectorTexture{};
  TextureHandle previousMotionVectorTexture{};
  TextureHandle reactiveMaskTexture{};
  RenderGraphTextureId motionVectorGraphTexture{};
  RenderGraphTextureId previousMotionVectorGraphTexture{};
  RenderGraphTextureId reactiveMaskGraphTexture{};
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
resolveMsaaSceneColorTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.msaaSceneColorTexture) &&
      nuri::isValid(frame.sharedResources.msaaSceneColorGraphTexture)) {
    return frame.sharedResources.msaaSceneColorTexture;
  }
  return {};
}

[[nodiscard]] inline TextureHandle
resolveMsaaSceneDepthTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.msaaSceneDepthTexture) &&
      nuri::isValid(frame.sharedResources.msaaSceneDepthGraphTexture)) {
    return frame.sharedResources.msaaSceneDepthTexture;
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

[[nodiscard]] inline TextureHandle
resolveMotionVectorTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.motionVectorTexture)) {
    return frame.sharedResources.motionVectorTexture;
  }
  return {};
}

[[nodiscard]] inline TextureHandle
resolvePreviousMotionVectorTexture(const RenderFrameContext &frame) {
  if (nuri::isValid(frame.sharedResources.previousMotionVectorTexture)) {
    return frame.sharedResources.previousMotionVectorTexture;
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
