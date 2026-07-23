#pragma once
#include "nuri/core/log.h"
#include "nuri/core/result.h"
#include "nuri/gfx/ddgi/ddgi_coverage.h"
#include "nuri/gfx/ddgi/ddgi_types.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/gfx/ray_tracing/ray_tracing_types.h"
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
#include <glm/glm.hpp>
#include <limits>
#include <memory_resource>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>
namespace nuri {

class SceneDrawDatabase;

class RenderScene;
class ResourceManager;
class TemporalFrameService;
struct RenderFrameContext;

enum class OpaqueDebugVisualization : uint8_t {
  None = 0,
  WireframeOverlay = 1,
  WireframeOnly = 2,
  TessPatchEdgesHeatmap = 3,
  MeshletId = 4,
  MeshletSelectedLod = 5,
};

enum class MeshletRenderMode : uint8_t {
  Disabled = 0,
  Opportunistic = 1,
  Required = 2,
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
  MSAA8x = 4,
};

enum class TemporalReconstructionProvider : uint8_t {
  Legacy = 0,
  Reference = 1,
  External = 2,
};

enum class CoverageMode : uint8_t {
  Sample1 = 0,
  Sample4 = 1,
  Sample8 = 2,
  Count,
};

enum class ColorReconstruction : uint8_t {
  Off = 0,
  ReferenceTAA = 1,
  LegacyTAA = 2,
  ExternalTemporal = 3,
};

enum class SpatialCleanupPoint : uint8_t {
  Off = 0,
  PreComposition = 1,
  PostTransparency = 2,
};

enum class AlphaCoveragePolicy : uint8_t {
  Off = 0,
  ThresholdedAlphaToCoverage = 1,
};

enum class TransparencyAAPolicy : uint8_t {
  InheritCoverage = 0,
  SingleSamplePostResolve = 1,
};

enum class PresentationAAUnsupportedReason : uint8_t {
  None = 0,
  Sample4Color = 1,
  Sample4Depth = 2,
  DepthResolveMin = 3,
  AlphaToCoverage = 4,
  Sample8Color = 5,
  Sample8Depth = 6,
};

struct PresentationAAProviderCapabilities {
  bool referenceTemporal = true;
  bool externalTemporal = false;
  bool reactiveMask = true;
  bool compositionMask = false;
};

using PresentationAAGpuCapabilities = GpuMultisampleCapabilities;

struct PresentationAAPlan {
  CoverageMode coverage = CoverageMode::Sample1;
  ColorReconstruction reconstruction = ColorReconstruction::Off;
  SpatialCleanupPoint spatialCleanup = SpatialCleanupPoint::Off;
  AlphaCoveragePolicy alphaCoverage = AlphaCoveragePolicy::Off;
  TransparencyAAPolicy transparency = TransparencyAAPolicy::InheritCoverage;
  bool sampleShadingSupported = false;
  bool sampleShadingEnabled = false;
  bool jitterScene = false;
  bool needsMotion = false;
  bool needsReactiveMask = false;
  bool needsCompositionMask = false;
  bool needsMotionClass = false;
  bool gtaoTemporal = false;
  bool valid = false;
  bool operator==(const PresentationAAPlan &) const = default;
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
  TAATemporalConfidence = 26,
  TAAPreviousDepthRejection = 27,
  TAAStabilityDiagnostics = 28,
  TAAStabilityOwnership = 29,
  TAAPatchProbe = 30,
  TAAMotionFilter = 31,
  SpatialAAEdges = 32,
  SpatialAABlendWeights = 33,
  SpatialAACleanupMask = 34,
  SpatialAASplitCompare = 35,
};

enum class AmbientOcclusionMode : uint8_t {
  Disabled = 0,
  GTAO = 1,
};

enum class AmbientOcclusionPreset : uint8_t {
  Low = 0,
  Balanced = 1,
  High = 2,
  Ultra = 3,
  Custom = 4,
};

enum class AmbientOcclusionDebugView : uint8_t {
  None = 0,
  Visibility = 1,
  BentNormal = 2,
  Normals = 3,
};

enum class HDRPostProcessDebugView : uint8_t {
  None = 0,
  BloomPrefilter = 1,
  BloomFinal = 2,
  LogAverageLuminance = 3,
  AdaptedExposure = 4,
};

enum class HDRExposureMeteringMode : uint8_t {
  FullFrame = 0,
  CenterWeighted = 1,
};

enum class AmbientOcclusionDisabledReason : uint8_t {
  None = 0,
  ModeDisabled = 1,
  OpaqueDisabled = 2,
  Msaa4x = 3,
  MissingResources = 4,
  Unsupported = 5,
};

enum class TemporalAAClampMode : uint8_t {
  Clamp = 0,
  Clip = 1,
  Variance = 2,
  ClampYCoCg = 3,
  ClipYCoCg = 4,
  VarianceYCoCg = 5,
};

enum class TemporalAAHdrWeightingMode : uint8_t {
  None = 0,
  Luminance = 1,
  LogLuminance = 2,
  ToneMapped = 3,
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

enum class TemporalAAQualityPreset : uint8_t {
  Performance = 0,
  Balanced = 1,
  Quality = 2,
  Ultra = 3,
  Custom = 4,
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
  SceneContentChanged = 9,
};

enum class ShadowQualityPreset : uint8_t {
  Custom = 0,
  Low = 1,
  Medium = 2,
  High = 3,
  Ultra = 4,
};

enum class ShadowSdsmReductionBackend : uint8_t {
  Cpu = 0,
  Gpu = 1,
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

enum class DDGIQualityPreset : uint8_t {
  Low = 0,
  Balanced = 1,
  High = 2,
  Custom = 3,
};

struct DDGICommandEpochs {
  uint64_t resetHistory = 0u;
  uint64_t forceFullUpdate = 0u;
  uint64_t rebuildRayTracingScene = 0u;
  constexpr bool operator==(const DDGICommandEpochs &) const = default;
};

[[nodiscard]] constexpr bool ddgiEpochIsPending(uint64_t requested,
                                                uint64_t consumed) noexcept {
  return requested > consumed;
}

enum class VisibilityCullingMode : uint8_t {
  Disabled = 0,
  CpuCoarse = 1,
  Hybrid = 2,
  GpuDriven = 3,
};

enum class VisibilityOcclusionMode : uint8_t {
  Disabled = 0,
  PreviousFrameHiZ = 1,
  CurrentFrameHiZExperimental = 2,
};

struct VisibilityDebugSettings {
  bool showObjectBounds = false;
  bool showMeshletBounds = false;
  bool visualizeCullReason = false;
  bool logCounters = false;
  uint32_t forcedVisibleListCapacity = std::numeric_limits<uint32_t>::max();
};

struct VisibilitySettings {
  VisibilityCullingMode mainViewMode = VisibilityCullingMode::GpuDriven;
  VisibilityCullingMode shadowMode = VisibilityCullingMode::Hybrid;
  VisibilityOcclusionMode occlusionMode = VisibilityOcclusionMode::Disabled;
  bool enableMeshletFrustumCulling = false;
  bool enableMeshletConeCulling = false;
  bool enableGpuInstanceCulling = true;
  bool enableGpuIndirectDraw = true;
  bool enableIndirectMeshDispatch = true;
  bool enableMeshletPreTaskCompaction = false;
  bool visibleOnUncertain = true;
  VisibilityDebugSettings debug{};
};

static constexpr float kDefaultToneMapExposureEv = 0.0f;
static constexpr float kDefaultAcesExposureOffsetEv = 0.35f;
static constexpr float kDefaultAgxExposureOffsetEv = -0.35f;
static constexpr float kDefaultToneMapCompareSplit = 0.5f;
static constexpr float kMinToneMapCompareSplit = 0.1f;
static constexpr float kMaxToneMapCompareSplit = 0.9f;
static constexpr float kDefaultHDRBloomStrength = 0.08f;
static constexpr float kDefaultHDRBloomThreshold = 1.0f;
static constexpr float kDefaultHDRBloomSoftKnee = 0.5f;
static constexpr uint32_t kDefaultHDRBloomMaxMipCount = 6u;
// The percentile meter sees scene-linear luminance rather than an 18% chart
// patch. A calibrated 10% target avoids lifting dark-dominant outdoor frames
// by roughly one stop while retaining explicit 0.18 overrides for test charts.
static constexpr float kDefaultHDRAdaptationTargetGray = 0.10f;
static constexpr float kDefaultHDRAdaptationSpeed = 3.0f;
static constexpr float kDefaultHDRAdaptationMaxEvChange = 1.0f;
static constexpr float kDefaultHDRHistogramLowPercentile = 0.01f;
static constexpr float kDefaultHDRHistogramHighPercentile = 0.99f;
static constexpr float kDefaultHDRHistogramMinLogLuminance = -12.0f;
static constexpr float kDefaultHDRHistogramMaxLogLuminance = 12.0f;
static constexpr float kDefaultHDRAdaptationMinEv = -8.0f;
static constexpr float kDefaultHDRAdaptationMaxEv = 8.0f;
static constexpr float kMaxDDGIPersistedRadianceLuminance = 4096.0f;

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
static constexpr uint32_t kMsaa8xSampleCount = 8u;
static constexpr Format kFrameCompositionMotionVectorFormat =
    Format::RG16_FLOAT;
static constexpr Format kFrameCompositionReactiveMaskFormat = Format::R8_UNORM;
static constexpr Format kFrameCompositionMotionClassFormat = Format::R8_UNORM;
static constexpr Format kFrameCompositionNormalFormat = Format::RGBA16_FLOAT;
static constexpr Format kFrameCompositionAmbientOcclusionFormat =
    Format::R16_UNORM;
static constexpr Format kFrameCompositionExposureFormat = Format::R32_FLOAT;
static constexpr ClearColor kFrameCompositionMotionVectorClearValue{
    .r = 0.0f, .g = 0.0f, .b = 0.0f, .a = 0.0f};
static constexpr ClearColor kFrameCompositionReactiveMaskClearValue{
    .r = 0.0f, .g = 0.0f, .b = 0.0f, .a = 0.0f};

enum class MotionClass : uint8_t {
  Invalid = 0u,
  ProvenStaticCameraOnly = 1u,
  Full = 2u,
  BackgroundRotation = 3u,
};

[[nodiscard]] constexpr float encodeMotionClass(MotionClass value) noexcept {
  return static_cast<float>(value) / 255.0f;
}

static constexpr ClearColor kFrameCompositionMotionClassClearValue{
    .r = encodeMotionClass(MotionClass::Invalid),
    .g = 0.0f,
    .b = 0.0f,
    .a = 0.0f};
static constexpr ClearColor kFrameCompositionNormalClearValue{
    .r = 0.0f, .g = 0.0f, .b = 1.0f, .a = 0.0f};
static constexpr ClearColor kFrameCompositionAmbientOcclusionClearValue{
    .r = 1.0f, .g = 0.0f, .b = 0.0f, .a = 0.0f};
static constexpr uint32_t kAmbientOcclusionFlagScalarAo = 1u << 0u;
static constexpr uint32_t kAmbientOcclusionFlagBentNormal = 1u << 1u;
static constexpr uint32_t kAmbientOcclusionDebugViewShift = 8u;
static constexpr uint32_t kAmbientOcclusionDebugViewMask = 0xffu;
static constexpr uint32_t kTemporalJitterSequenceLength = 8u;
static constexpr Format kDefaultShadowMapDepthFormat = Format::D16_UNORM;
static constexpr uint32_t kMaxShadowCascades = 4u;
static constexpr uint32_t kInvalidShadowBindlessIndex = 0xFFFFFFFFu;
static constexpr uint32_t kInvalidSamplerBindlessIndex = 0xFFFFFFFFu;
static constexpr uint32_t kShadowFrameFlagEnabled = 1u << 0u;
static constexpr uint32_t kShadowFrameFlagVisualizeShadowFactor = 1u << 1u;
static constexpr uint32_t kShadowFrameFlagVisualizeCascadeIndex = 1u << 2u;
static constexpr uint32_t kShadowFrameFlagVisualizePCFResult = 1u << 3u;
static constexpr uint32_t kShadowFrameFlagVisualizeReceiverDepth = 1u << 4u;
static constexpr uint32_t kShadowFrameFlagVisualizeShadowMapDepth = 1u << 5u;

template <typename Enum>
[[nodiscard]] constexpr Enum sanitizeContiguousEnum(Enum value, Enum last,
                                                    Enum fallback) noexcept {
  using Value = std::underlying_type_t<Enum>;
  return static_cast<Value>(value) <= static_cast<Value>(last) ? value
                                                               : fallback;
}

[[nodiscard]] constexpr uint8_t
sanitizeTextureFilterAnisotropy(uint8_t anisotropy) noexcept {
  return anisotropy == 2u || anisotropy == 4u || anisotropy == 8u ||
                 anisotropy == 16u
             ? anisotropy
             : 8u;
}

[[nodiscard]] constexpr TextureFilterMode
sanitizeTextureFilterMode(TextureFilterMode mode) noexcept {
  return sanitizeContiguousEnum(mode, TextureFilterMode::Anisotropic,
                                TextureFilterMode::Trilinear);
}

[[nodiscard]] constexpr ToneMapper
sanitizeToneMapper(ToneMapper mapper) noexcept {
  return sanitizeContiguousEnum(mapper, ToneMapper::AgX, ToneMapper::ACES2_SDR);
}

[[nodiscard]] constexpr AntiAliasingMode
sanitizeAntiAliasingMode(AntiAliasingMode mode) noexcept {
  return sanitizeContiguousEnum(mode, AntiAliasingMode::MSAA8x,
                                AntiAliasingMode::None);
}

[[nodiscard]] constexpr bool isMsaaMode(AntiAliasingMode mode) noexcept {
  const AntiAliasingMode sanitized = sanitizeAntiAliasingMode(mode);
  return sanitized == AntiAliasingMode::MSAA4x ||
         sanitized == AntiAliasingMode::MSAA8x;
}

inline constexpr size_t kCoverageModeCount =
    static_cast<size_t>(CoverageMode::Count);

[[nodiscard]] constexpr size_t
coverageModeIndex(CoverageMode coverage) noexcept {
  return static_cast<size_t>(coverage);
}

[[nodiscard]] constexpr uint32_t
coverageSampleCount(CoverageMode coverage) noexcept {
  switch (coverage) {
  case CoverageMode::Sample4:
    return kMsaa4xSampleCount;
  case CoverageMode::Sample8:
    return kMsaa8xSampleCount;
  case CoverageMode::Sample1:
  case CoverageMode::Count:
    return 1u;
  }
  return 1u;
}

[[nodiscard]] constexpr TemporalReconstructionProvider
sanitizeTemporalReconstructionProvider(
    TemporalReconstructionProvider provider) noexcept {
  return sanitizeContiguousEnum(provider,
                                TemporalReconstructionProvider::External,
                                TemporalReconstructionProvider::Legacy);
}

[[nodiscard]] constexpr AmbientOcclusionMode
sanitizeAmbientOcclusionMode(AmbientOcclusionMode mode) noexcept {
  return sanitizeContiguousEnum(mode, AmbientOcclusionMode::GTAO,
                                AmbientOcclusionMode::GTAO);
}

[[nodiscard]] constexpr AmbientOcclusionPreset
sanitizeAmbientOcclusionPreset(AmbientOcclusionPreset preset) noexcept {
  return sanitizeContiguousEnum(preset, AmbientOcclusionPreset::Custom,
                                AmbientOcclusionPreset::Balanced);
}

[[nodiscard]] constexpr AmbientOcclusionDebugView
sanitizeAmbientOcclusionDebugView(AmbientOcclusionDebugView view) noexcept {
  return sanitizeContiguousEnum(view, AmbientOcclusionDebugView::Normals,
                                AmbientOcclusionDebugView::None);
}

[[nodiscard]] constexpr HDRPostProcessDebugView
sanitizeHDRPostProcessDebugView(HDRPostProcessDebugView view) noexcept {
  return sanitizeContiguousEnum(view, HDRPostProcessDebugView::AdaptedExposure,
                                HDRPostProcessDebugView::None);
}

[[nodiscard]] constexpr AntiAliasingDebugView
sanitizeAntiAliasingDebugView(AntiAliasingDebugView view) noexcept {
  return sanitizeContiguousEnum(view,
                                AntiAliasingDebugView::SpatialAASplitCompare,
                                AntiAliasingDebugView::None);
}

[[nodiscard]] constexpr TemporalAAClampMode
sanitizeTemporalAAClampMode(TemporalAAClampMode mode) noexcept {
  return sanitizeContiguousEnum(mode, TemporalAAClampMode::VarianceYCoCg,
                                TemporalAAClampMode::VarianceYCoCg);
}

[[nodiscard]] constexpr TemporalAAHdrWeightingMode
sanitizeTemporalAAHdrWeightingMode(TemporalAAHdrWeightingMode mode) noexcept {
  return sanitizeContiguousEnum(mode, TemporalAAHdrWeightingMode::ToneMapped,
                                TemporalAAHdrWeightingMode::Luminance);
}

[[nodiscard]] constexpr TemporalAAVelocityDilationMode
sanitizeTemporalAAVelocityDilationMode(
    TemporalAAVelocityDilationMode mode) noexcept {
  return sanitizeContiguousEnum(
      mode, TemporalAAVelocityDilationMode::LargestMagnitude,
      TemporalAAVelocityDilationMode::ClosestDepth);
}

[[nodiscard]] constexpr TemporalAAHistoryFilterMode
sanitizeTemporalAAHistoryFilterMode(TemporalAAHistoryFilterMode mode) noexcept {
  return sanitizeContiguousEnum(mode, TemporalAAHistoryFilterMode::Bilinear,
                                TemporalAAHistoryFilterMode::CatmullRom);
}

[[nodiscard]] constexpr TemporalAAQualityPreset
sanitizeTemporalAAQualityPreset(TemporalAAQualityPreset preset) noexcept {
  return sanitizeContiguousEnum(preset, TemporalAAQualityPreset::Custom,
                                TemporalAAQualityPreset::Quality);
}

[[nodiscard]] constexpr DDGIQualityPreset
sanitizeDDGIQualityPreset(DDGIQualityPreset preset) noexcept {
  return sanitizeContiguousEnum(preset, DDGIQualityPreset::Custom,
                                DDGIQualityPreset::Balanced);
}

[[nodiscard]] constexpr DDGIDebugView
sanitizeDDGIDebugView(DDGIDebugView view) noexcept {
  return sanitizeContiguousEnum(view, DDGIDebugView::LeakRisk,
                                DDGIDebugView::None);
}

struct RenderSettings {
  struct SkyboxSettings {
    bool enabled = true;
  };
  struct OpaqueSettings {
    bool enabled = true;
    bool enableDepthPrepass = false;
    bool enableDepthPyramid = false;
    OpaqueDebugVisualization debugVisualization =
        OpaqueDebugVisualization::None;
    bool enableInstanceCompute = true;
    bool enableIndirectDraw = true;
    bool enableInstancedDraw = true;
    bool enableMeshLod = true;
    bool enableCpuFrustumCulling = true;
    MeshletRenderMode meshletMode = MeshletRenderMode::Disabled;
    bool enableMeshletFrustumCulling = true;
    bool enableMeshletConeCulling = true;
    uint32_t hybridClassicMaxMeshlets = 96u;
    int32_t forcedMeshLod = -1;
    float meshLodTargetPixelError = 1.0f;
    float meshLodHysteresisRatio = 0.2f;
    glm::vec3 meshLodDistanceThresholds{8.0f, 16.0f, 32.0f};
    bool enableInstanceAnimation = true;
    bool enableTessellation = false;
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 6.0f;
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
    uint32_t debugCascadeIndex = 0;
    bool freezeCascades = false;
    bool freezeLightView = false;
    bool visualizeCascadeIndex = false;
    bool visualizeShadowFactor = false;
    bool visualizePCFResult = false;
    bool visualizeReceiverDepth = false;
    bool visualizeShadowMapDepth = false;
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
    float splitLambda = 0.75f;
    float cascadeBlendFraction = 0.08f;
    float constantBias = 0.0005f;
    float slopeBias = 1.5f;
    float normalBias = 0.0f;
    uint32_t pcfSampleCount = 9;
    float sdsmTemporalBlend = 0.85f;
    ShadowDebugSettings debug{};
  };
  struct TransparentSettings {
    bool enabled = true;
  };
  struct TransmissionSettings {
    bool enabled = true;
    bool taaJitterPrefilter = true;
    float taaJitterPrefilterMaxLod = 1.0f;
    float taaJitterDepthBiasConstant = -8.0f;
  };
  struct AmbientOcclusionSettings {
    AmbientOcclusionMode mode = AmbientOcclusionMode::GTAO;
    AmbientOcclusionPreset preset = AmbientOcclusionPreset::Balanced;
    AmbientOcclusionDebugView debugView = AmbientOcclusionDebugView::None;
    float strength = 1.0f;
    bool active = true;
    bool temporalAccumulation = true;
    AmbientOcclusionDisabledReason disabledReason =
        AmbientOcclusionDisabledReason::None;
    uint32_t sliceCount = 2u;
    uint32_t stepCount = 4u;
    uint32_t denoisePassCount = 2u;
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
  struct HDRPostProcessSettings {
    bool bloomEnabled = true;
    float bloomStrength = kDefaultHDRBloomStrength;
    float bloomThreshold = kDefaultHDRBloomThreshold;
    float bloomSoftKnee = kDefaultHDRBloomSoftKnee;
    uint32_t bloomMaxMipCount = kDefaultHDRBloomMaxMipCount;
    bool adaptationEnabled = false;
    HDRExposureMeteringMode meteringMode =
        HDRExposureMeteringMode::CenterWeighted;
    float adaptationTargetGray = kDefaultHDRAdaptationTargetGray;
    // Retained as the legacy/profile seed. Runtime adaptation uses the two
    // directional speeds below.
    float adaptationSpeed = kDefaultHDRAdaptationSpeed;
    float adaptationBrightenSpeed = kDefaultHDRAdaptationSpeed;
    float adaptationDarkenSpeed = kDefaultHDRAdaptationSpeed;
    float adaptationMaxEvChange = kDefaultHDRAdaptationMaxEvChange;
    float adaptationMinEv = kDefaultHDRAdaptationMinEv;
    float adaptationMaxEv = kDefaultHDRAdaptationMaxEv;
    float histogramLowPercentile = kDefaultHDRHistogramLowPercentile;
    float histogramHighPercentile = kDefaultHDRHistogramHighPercentile;
    float histogramMinLogLuminance = kDefaultHDRHistogramMinLogLuminance;
    float histogramMaxLogLuminance = kDefaultHDRHistogramMaxLogLuminance;
    HDRPostProcessDebugView debugView = HDRPostProcessDebugView::None;
  };
  struct AntiAliasingDebugSettings {
    bool jitterEnabled = true;
    bool freezeJitter = false;
    bool resetHistoryRequested = false;
    bool logDiagnostics = false;
    bool spatialPostTaaCleanup = true;
    bool spatialPostMsaaCleanup = false;
    bool taaSharpenEnabled = true;
    bool taaMaterialMipBiasEnabled = false;
    bool transparentPostTaaSpatialCleanup = true;
    AntiAliasingDebugView view = AntiAliasingDebugView::None;
    float diagnosticLogIntervalSeconds = 0.25f;
    float taaJitterScale = 0.75f;
    float taaCurrentFrameWeight = 0.045f;
    float taaSharpenStrength = 0.14f;
    float taaSharpenConfidenceThreshold = 0.82f;
    float taaMaterialMipBias = 0.0f;
    float taaDepthDiscontinuityThreshold = 0.01f;
    float taaVelocityRejectionThreshold = 1.5f;
    float taaVelocityBlendScale = 0.22f;
    float taaMotionCurrentWeight = 0.22f;
    float taaDisocclusionCurrentWeight = 0.62f;
    float taaClampCurrentWeight = 0.38f;
    float taaClampBlendAttenuation = 0.35f;
    float taaVarianceGamma = 1.85f;
    float taaHdrWeightStrength = 0.50f;
    float taaReactiveCurrentWeight = 0.85f;
    float taaReactiveStrength = 1.0f;
    float taaVelocityDilationDepthThreshold = 0.01f;
    TemporalAAClampMode taaClampMode = TemporalAAClampMode::VarianceYCoCg;
    TemporalAAHdrWeightingMode taaHdrWeightingMode =
        TemporalAAHdrWeightingMode::Luminance;
    TemporalAAVelocityDilationMode taaVelocityDilationMode =
        TemporalAAVelocityDilationMode::ClosestDepth;
    TemporalAAHistoryFilterMode taaHistoryFilterMode =
        TemporalAAHistoryFilterMode::CatmullRom;
  };
  struct AntiAliasingSettings {
    AntiAliasingMode mode = AntiAliasingMode::None;
    TemporalReconstructionProvider temporalProvider =
        TemporalReconstructionProvider::Legacy;
    TemporalAAQualityPreset qualityPreset = TemporalAAQualityPreset::Quality;
    AntiAliasingDebugSettings debug{};
  };
  struct DDGISettings {
    bool enabled = false;
    DDGIQualityPreset preset = DDGIQualityPreset::Balanced;
    uint32_t raysPerProbe = 128u;
    uint32_t classificationRaysPerProbe = 16u;
    uint32_t maxProbeUpdatesPerFrame = 512u;
    uint32_t maxRayQueriesPerFrame = 65'536u;
    uint32_t maxLocalLightsPerHit = 8u;
    uint32_t maxCandidateIntersectionsPerRay = 64u;
    float irradianceHysteresis = 0.97f;
    float distanceHysteresis = 0.98f;
    float changeIrradianceHysteresisScale = 0.50f;
    float changeDistanceHysteresisScale = 0.50f;
    // Receiver reconstruction bias remains separate from ray-origin domains.
    float selfShadowBias = 0.30f;
    float primaryProbeBias = 0.0001f;
    float localShadowBias = 0.30f;
    float directionalShadowBias = 0.30f;
    float classificationBias = 0.30f;
    float multiBounceLuminanceClamp = 32.0f;
    bool relocation = true;
    bool classification = true;
    bool multiBounce = true;
    bool diagnosticCounters = true;
    bool freezeUpdates = false;
    DDGIDebugView debugView = DDGIDebugView::None;
    bool showVolumes = false;
    bool showProbes = false;
    bool showSelectedProbeRays = false;
    DDGICommandEpochs requestedEpochs{};
    DDGICoverageSettings coverage{};
  };
  SkyboxSettings skybox{};
  OpaqueSettings opaque{};
  TransmissionSettings transmission{};
  TransparentSettings transparent{};
  DebugSettings debug{};
  ShadowSettings shadow{};
  VisibilitySettings visibility{};
  AntiAliasingSettings antiAliasing{};
  AmbientOcclusionSettings ambientOcclusion{};
  TextureFilteringSettings textureFiltering{};
  ToneMapSettings toneMap{};
  HDRPostProcessSettings hdrPostProcess{};
  DDGISettings ddgi{};
};

[[nodiscard]] constexpr VisibilityCullingMode
sanitizeVisibilityCullingMode(VisibilityCullingMode mode) noexcept {
  return sanitizeContiguousEnum(mode, VisibilityCullingMode::GpuDriven,
                                VisibilityCullingMode::Disabled);
}

[[nodiscard]] constexpr VisibilityOcclusionMode
sanitizeVisibilityOcclusionMode(VisibilityOcclusionMode mode) noexcept {
  return sanitizeContiguousEnum(
      mode, VisibilityOcclusionMode::CurrentFrameHiZExperimental,
      VisibilityOcclusionMode::Disabled);
}

inline void sanitizeVisibilitySettings(VisibilitySettings &settings) noexcept {
  settings.mainViewMode = sanitizeVisibilityCullingMode(settings.mainViewMode);
  settings.shadowMode = sanitizeVisibilityCullingMode(settings.shadowMode);
  settings.occlusionMode =
      sanitizeVisibilityOcclusionMode(settings.occlusionMode);
}

[[nodiscard]] inline float finiteClamp(float value, float minimum,
                                       float maximum, float fallback) noexcept {
  return std::clamp(std::isfinite(value) ? value : fallback, minimum, maximum);
}

inline void applyDDGIQualityPreset(RenderSettings::DDGISettings &settings,
                                   DDGIQualityPreset preset) noexcept {
  settings.preset = sanitizeDDGIQualityPreset(preset);
  if (settings.preset == DDGIQualityPreset::Custom) {
    return;
  }
  struct Preset {
    uint32_t raysPerProbe;
    uint32_t classificationRaysPerProbe;
    uint32_t maxProbeUpdatesPerFrame;
    uint32_t maxRayQueriesPerFrame;
    float irradianceHysteresis;
    float distanceHysteresis;
  };
  static constexpr std::array presets{
      Preset{64u, 16u, 512u, 32'768u, 0.98f, 0.99f},
      Preset{128u, 24u, 512u, 65'536u, 0.97f, 0.98f},
      Preset{256u, 32u, 512u, 131'072u, 0.95f, 0.98f},
  };
  const Preset &values = presets[static_cast<uint8_t>(settings.preset)];
  settings.raysPerProbe = values.raysPerProbe;
  settings.classificationRaysPerProbe = values.classificationRaysPerProbe;
  settings.maxProbeUpdatesPerFrame = values.maxProbeUpdatesPerFrame;
  settings.maxRayQueriesPerFrame = values.maxRayQueriesPerFrame;
  settings.irradianceHysteresis = values.irradianceHysteresis;
  settings.distanceHysteresis = values.distanceHysteresis;
}

inline void
sanitizeDDGISettings(RenderSettings::DDGISettings &settings,
                     uint32_t maxRayQueriesHardCap =
                         std::numeric_limits<uint32_t>::max()) noexcept {
  settings.preset = sanitizeDDGIQualityPreset(settings.preset);
  applyDDGIQualityPreset(settings, settings.preset);
  settings.debugView = sanitizeDDGIDebugView(settings.debugView);
  sanitizeDDGICoverageSettings(settings.coverage);
  settings.raysPerProbe = std::clamp(settings.raysPerProbe, 16u, 1024u);
  settings.classificationRaysPerProbe = std::clamp(
      settings.classificationRaysPerProbe, 8u, settings.raysPerProbe);
  settings.maxProbeUpdatesPerFrame =
      std::clamp(settings.maxProbeUpdatesPerFrame, 1u, 65'536u);
  const uint32_t minimumQueries = 2u * settings.raysPerProbe;
  const uint32_t effectiveHardCap =
      std::max(maxRayQueriesHardCap, minimumQueries);
  settings.maxRayQueriesPerFrame = std::clamp(settings.maxRayQueriesPerFrame,
                                              minimumQueries, effectiveHardCap);
  settings.maxLocalLightsPerHit =
      std::clamp(settings.maxLocalLightsPerHit, 0u, 16u);
  settings.maxCandidateIntersectionsPerRay =
      std::clamp(settings.maxCandidateIntersectionsPerRay, 8u, 256u);
  settings.irradianceHysteresis =
      finiteClamp(settings.irradianceHysteresis, 0.0f, 0.9999f, 0.97f);
  settings.distanceHysteresis =
      finiteClamp(settings.distanceHysteresis, 0.0f, 0.9999f, 0.98f);
  settings.changeIrradianceHysteresisScale =
      finiteClamp(settings.changeIrradianceHysteresisScale, 0.0f, 1.0f, 0.50f);
  settings.changeDistanceHysteresisScale =
      finiteClamp(settings.changeDistanceHysteresisScale, 0.0f, 1.0f, 0.50f);
  settings.selfShadowBias =
      finiteClamp(settings.selfShadowBias, 0.0f, 2.0f, 0.30f);
  settings.primaryProbeBias =
      finiteClamp(settings.primaryProbeBias, 0.0f, 0.25f, 0.0001f);
  settings.localShadowBias =
      finiteClamp(settings.localShadowBias, 0.0f, 2.0f, 0.30f);
  settings.directionalShadowBias =
      finiteClamp(settings.directionalShadowBias, 0.0f, 2.0f, 0.30f);
  settings.classificationBias =
      finiteClamp(settings.classificationBias, 0.0f, 2.0f, 0.30f);
  if (!std::isfinite(settings.multiBounceLuminanceClamp) ||
      settings.multiBounceLuminanceClamp <= 0.0f) {
    settings.multiBounceLuminanceClamp = 32.0f;
  } else {
    settings.multiBounceLuminanceClamp = std::min(
        settings.multiBounceLuminanceClamp, kMaxDDGIPersistedRadianceLuminance);
  }
}

inline void
copyDDGIQualityPresetToCustom(RenderSettings::DDGISettings &settings) noexcept {
  const DDGIQualityPreset preset = sanitizeDDGIQualityPreset(settings.preset);
  applyDDGIQualityPreset(settings, preset);
  settings.preset = DDGIQualityPreset::Custom;
}

inline void sanitizeAmbientOcclusionSettings(
    RenderSettings::AmbientOcclusionSettings &settings,
    const RenderSettings::OpaqueSettings &opaque,
    const RenderSettings::AntiAliasingSettings &antiAliasing) {
  settings.mode = sanitizeAmbientOcclusionMode(settings.mode);
  settings.preset = sanitizeAmbientOcclusionPreset(settings.preset);
  settings.debugView = sanitizeAmbientOcclusionDebugView(settings.debugView);
  settings.strength = finiteClamp(settings.strength, 0.0f, 4.0f, 1.0f);
  const bool temporalAccumulationRequested = settings.temporalAccumulation;
  settings.active = false;
  settings.temporalAccumulation = false;
  settings.disabledReason = AmbientOcclusionDisabledReason::None;
  if (settings.mode == AmbientOcclusionMode::Disabled) {
    settings.disabledReason = AmbientOcclusionDisabledReason::ModeDisabled;
  } else if (!opaque.enabled) {
    settings.disabledReason = AmbientOcclusionDisabledReason::OpaqueDisabled;
  } else {
    settings.active = true;
    settings.temporalAccumulation = temporalAccumulationRequested;
  }
  static constexpr std::array<std::array<uint32_t, 3>, 4> presets{{
      {1u, 3u, 1u},
      {2u, 4u, 2u},
      {3u, 6u, 2u},
      {4u, 8u, 2u},
  }};
  if (settings.preset != AmbientOcclusionPreset::Custom) {
    const auto &values = presets[static_cast<uint8_t>(settings.preset)];
    settings.sliceCount = values[0];
    settings.stepCount = values[1];
    settings.denoisePassCount = values[2];
  }
}

[[nodiscard]] inline bool
isAmbientOcclusionActive(const RenderSettings &settings) noexcept {
  RenderSettings::AmbientOcclusionSettings ao = settings.ambientOcclusion;
  sanitizeAmbientOcclusionSettings(ao, settings.opaque, settings.antiAliasing);
  return ao.active;
}

[[nodiscard]] inline uint32_t
shadowDebugFrameFlags(const RenderSettings::ShadowDebugSettings &debug) {
  return (debug.visualizeShadowFactor ? kShadowFrameFlagVisualizeShadowFactor
                                      : 0u) |
         (debug.visualizeCascadeIndex ? kShadowFrameFlagVisualizeCascadeIndex
                                      : 0u) |
         (debug.visualizePCFResult ? kShadowFrameFlagVisualizePCFResult : 0u) |
         (debug.visualizeReceiverDepth ? kShadowFrameFlagVisualizeReceiverDepth
                                       : 0u) |
         (debug.visualizeShadowMapDepth
              ? kShadowFrameFlagVisualizeShadowMapDepth
              : 0u);
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

inline void sanitizeHDRPostProcessSettings(
    RenderSettings::HDRPostProcessSettings &settings) {
  settings.bloomStrength =
      finiteClamp(settings.bloomStrength, 0.0f, 1.0f, kDefaultHDRBloomStrength);
  settings.bloomThreshold = finiteClamp(settings.bloomThreshold, 0.0f, 64.0f,
                                        kDefaultHDRBloomThreshold);
  settings.bloomSoftKnee =
      finiteClamp(settings.bloomSoftKnee, 0.0f, 4.0f, kDefaultHDRBloomSoftKnee);
  settings.bloomMaxMipCount =
      std::clamp(settings.bloomMaxMipCount, 1u, kMaxSceneDepthPyramidLevels);
  settings.adaptationTargetGray =
      finiteClamp(settings.adaptationTargetGray, 0.001f, 16.0f,
                  kDefaultHDRAdaptationTargetGray);
  settings.adaptationSpeed = finiteClamp(settings.adaptationSpeed, 0.0f, 64.0f,
                                         kDefaultHDRAdaptationSpeed);
  settings.adaptationBrightenSpeed =
      finiteClamp(settings.adaptationBrightenSpeed, 0.0f, 64.0f,
                  kDefaultHDRAdaptationSpeed);
  settings.adaptationDarkenSpeed = finiteClamp(
      settings.adaptationDarkenSpeed, 0.0f, 64.0f, kDefaultHDRAdaptationSpeed);
  settings.adaptationMaxEvChange =
      finiteClamp(settings.adaptationMaxEvChange, 0.0f, 16.0f,
                  kDefaultHDRAdaptationMaxEvChange);
  settings.adaptationMinEv = finiteClamp(settings.adaptationMinEv, -32.0f,
                                         32.0f, kDefaultHDRAdaptationMinEv);
  settings.adaptationMaxEv = finiteClamp(settings.adaptationMaxEv, -32.0f,
                                         32.0f, kDefaultHDRAdaptationMaxEv);
  if (settings.adaptationMinEv > settings.adaptationMaxEv) {
    std::swap(settings.adaptationMinEv, settings.adaptationMaxEv);
  }
  settings.histogramLowPercentile =
      finiteClamp(settings.histogramLowPercentile, 0.0f, 0.49f,
                  kDefaultHDRHistogramLowPercentile);
  settings.histogramHighPercentile =
      finiteClamp(settings.histogramHighPercentile, 0.51f, 1.0f,
                  kDefaultHDRHistogramHighPercentile);
  settings.histogramMinLogLuminance =
      finiteClamp(settings.histogramMinLogLuminance, -32.0f, 31.0f,
                  kDefaultHDRHistogramMinLogLuminance);
  settings.histogramMaxLogLuminance =
      finiteClamp(settings.histogramMaxLogLuminance, -31.0f, 32.0f,
                  kDefaultHDRHistogramMaxLogLuminance);
  if (settings.histogramMinLogLuminance >= settings.histogramMaxLogLuminance) {
    settings.histogramMinLogLuminance = kDefaultHDRHistogramMinLogLuminance;
    settings.histogramMaxLogLuminance = kDefaultHDRHistogramMaxLogLuminance;
  }
  settings.meteringMode = sanitizeContiguousEnum(
      settings.meteringMode, HDRExposureMeteringMode::CenterWeighted,
      HDRExposureMeteringMode::CenterWeighted);
  settings.debugView = sanitizeHDRPostProcessDebugView(settings.debugView);
}

inline void
sanitizeTransmissionSettings(RenderSettings::TransmissionSettings &settings) {
  settings.taaJitterPrefilterMaxLod =
      finiteClamp(settings.taaJitterPrefilterMaxLod, 0.0f, 2.0f, 1.0f);
  settings.taaJitterDepthBiasConstant =
      finiteClamp(settings.taaJitterDepthBiasConstant, -64.0f, 0.0f, -8.0f);
}

inline void
sanitizeAntiAliasingSettings(RenderSettings::AntiAliasingSettings &settings) {
  settings.mode = sanitizeAntiAliasingMode(settings.mode);
  settings.qualityPreset =
      sanitizeTemporalAAQualityPreset(settings.qualityPreset);
  settings.debug.view = sanitizeAntiAliasingDebugView(settings.debug.view);
  settings.debug.taaClampMode =
      sanitizeTemporalAAClampMode(settings.debug.taaClampMode);
  settings.debug.taaHdrWeightingMode =
      sanitizeTemporalAAHdrWeightingMode(settings.debug.taaHdrWeightingMode);
  settings.debug.taaVelocityDilationMode =
      sanitizeTemporalAAVelocityDilationMode(
          settings.debug.taaVelocityDilationMode);
  using Debug = RenderSettings::AntiAliasingDebugSettings;
  struct FloatSetting {
    float Debug::*field;
    float minimum;
    float maximum;
    float fallback;
  };
  static constexpr std::array floatSettings{
      FloatSetting{&Debug::taaCurrentFrameWeight, 0.0f, 1.0f, 0.045f},
      FloatSetting{&Debug::diagnosticLogIntervalSeconds, 0.033f, 5.0f, 0.25f},
      FloatSetting{&Debug::taaJitterScale, 0.0f, 1.0f, 0.75f},
      FloatSetting{&Debug::taaSharpenStrength, 0.0f, 1.0f, 0.14f},
      FloatSetting{&Debug::taaSharpenConfidenceThreshold, 0.0f, 1.0f, 0.82f},
      FloatSetting{&Debug::taaMaterialMipBias, -1.0f, 0.0f, 0.0f},
      FloatSetting{&Debug::taaDepthDiscontinuityThreshold, 0.0f, 1.0f, 0.01f},
      FloatSetting{&Debug::taaVelocityRejectionThreshold, 0.0f, 64.0f, 1.5f},
      FloatSetting{&Debug::taaVelocityBlendScale, 0.0f, 4.0f, 0.22f},
      FloatSetting{&Debug::taaMotionCurrentWeight, 0.0f, 1.0f, 0.22f},
      FloatSetting{&Debug::taaDisocclusionCurrentWeight, 0.0f, 1.0f, 0.62f},
      FloatSetting{&Debug::taaClampCurrentWeight, 0.0f, 1.0f, 0.38f},
      FloatSetting{&Debug::taaClampBlendAttenuation, 0.0f, 1.0f, 0.35f},
      FloatSetting{&Debug::taaVarianceGamma, 0.0f, 4.0f, 1.85f},
      FloatSetting{&Debug::taaHdrWeightStrength, 0.0f, 1.0f, 0.50f},
      FloatSetting{&Debug::taaReactiveCurrentWeight, 0.0f, 1.0f, 0.85f},
      FloatSetting{&Debug::taaReactiveStrength, 0.0f, 4.0f, 1.0f},
      FloatSetting{&Debug::taaVelocityDilationDepthThreshold, 0.0f, 1.0f,
                   0.01f},
  };
  for (const FloatSetting &spec : floatSettings) {
    float &value = settings.debug.*spec.field;
    value = finiteClamp(value, spec.minimum, spec.maximum, spec.fallback);
  }
  settings.debug.taaHistoryFilterMode =
      sanitizeTemporalAAHistoryFilterMode(settings.debug.taaHistoryFilterMode);
  if (settings.mode != AntiAliasingMode::TAA) {
    settings.debug.jitterEnabled = false;
    settings.debug.freezeJitter = false;
  }
}

inline void copyTemporalAAPresetTuning(
    RenderSettings::AntiAliasingDebugSettings &dst,
    const RenderSettings::AntiAliasingDebugSettings &src) {
  dst.spatialPostTaaCleanup = src.spatialPostTaaCleanup;
  dst.taaSharpenEnabled = src.taaSharpenEnabled;
  dst.taaMaterialMipBiasEnabled = src.taaMaterialMipBiasEnabled;
  dst.transparentPostTaaSpatialCleanup = src.transparentPostTaaSpatialCleanup;
  dst.taaJitterScale = src.taaJitterScale;
  dst.taaCurrentFrameWeight = src.taaCurrentFrameWeight;
  dst.taaSharpenStrength = src.taaSharpenStrength;
  dst.taaSharpenConfidenceThreshold = src.taaSharpenConfidenceThreshold;
  dst.taaMaterialMipBias = src.taaMaterialMipBias;
  dst.taaDepthDiscontinuityThreshold = src.taaDepthDiscontinuityThreshold;
  dst.taaVelocityRejectionThreshold = src.taaVelocityRejectionThreshold;
  dst.taaVelocityBlendScale = src.taaVelocityBlendScale;
  dst.taaMotionCurrentWeight = src.taaMotionCurrentWeight;
  dst.taaDisocclusionCurrentWeight = src.taaDisocclusionCurrentWeight;
  dst.taaClampCurrentWeight = src.taaClampCurrentWeight;
  dst.taaClampBlendAttenuation = src.taaClampBlendAttenuation;
  dst.taaVarianceGamma = src.taaVarianceGamma;
  dst.taaHdrWeightStrength = src.taaHdrWeightStrength;
  dst.taaReactiveCurrentWeight = src.taaReactiveCurrentWeight;
  dst.taaReactiveStrength = src.taaReactiveStrength;
  dst.taaVelocityDilationDepthThreshold = src.taaVelocityDilationDepthThreshold;
  dst.taaClampMode = src.taaClampMode;
  dst.taaHdrWeightingMode = src.taaHdrWeightingMode;
  dst.taaVelocityDilationMode = src.taaVelocityDilationMode;
  dst.taaHistoryFilterMode = src.taaHistoryFilterMode;
}

[[nodiscard]] inline RenderSettings::AntiAliasingDebugSettings
temporalAAQualityPresetDebugSettings(TemporalAAQualityPreset preset) {
  RenderSettings::AntiAliasingDebugSettings debug{};
  debug.spatialPostTaaCleanup = true;
  debug.transparentPostTaaSpatialCleanup = true;
  debug.taaClampMode = TemporalAAClampMode::VarianceYCoCg;
  debug.taaHdrWeightingMode = TemporalAAHdrWeightingMode::Luminance;
  debug.taaVelocityDilationMode = TemporalAAVelocityDilationMode::ClosestDepth;
  debug.taaHistoryFilterMode = TemporalAAHistoryFilterMode::CatmullRom;
  struct Preset {
    float jitterScale;
    float currentWeight;
    float motionWeight;
    float disocclusionWeight;
    float clampWeight;
    float velocityBlend;
    float varianceGamma;
    float sharpenStrength;
    float sharpenConfidence;
    TemporalAAHistoryFilterMode historyFilter =
        TemporalAAHistoryFilterMode::CatmullRom;
  };
  static constexpr std::array presets{
      Preset{0.60f, 0.075f, 0.30f, 0.68f, 0.48f, 0.34f, 1.35f, 0.18f, 0.74f,
             TemporalAAHistoryFilterMode::Bilinear},
      Preset{0.70f, 0.055f, 0.27f, 0.64f, 0.43f, 0.28f, 1.60f, 0.16f, 0.78f},
      Preset{0.75f, 0.045f, 0.22f, 0.62f, 0.38f, 0.22f, 1.85f, 0.14f, 0.82f},
      Preset{0.75f, 0.035f, 0.18f, 0.58f, 0.32f, 0.18f, 2.15f, 0.10f, 0.88f},
  };
  const TemporalAAQualityPreset sanitized =
      sanitizeTemporalAAQualityPreset(preset);
  const Preset &values =
      presets[static_cast<uint8_t>(sanitized == TemporalAAQualityPreset::Custom
                                       ? TemporalAAQualityPreset::Quality
                                       : sanitized)];
  debug.taaJitterScale = values.jitterScale;
  debug.taaCurrentFrameWeight = values.currentWeight;
  debug.taaMotionCurrentWeight = values.motionWeight;
  debug.taaDisocclusionCurrentWeight = values.disocclusionWeight;
  debug.taaClampCurrentWeight = values.clampWeight;
  debug.taaVelocityBlendScale = values.velocityBlend;
  debug.taaVarianceGamma = values.varianceGamma;
  debug.taaSharpenStrength = values.sharpenStrength;
  debug.taaSharpenConfidenceThreshold = values.sharpenConfidence;
  debug.taaHistoryFilterMode = values.historyFilter;
  RenderSettings::AntiAliasingSettings settings{
      .mode = AntiAliasingMode::TAA,
      .qualityPreset = TemporalAAQualityPreset::Custom,
      .debug = debug,
  };
  sanitizeAntiAliasingSettings(settings);
  return settings.debug;
}

[[nodiscard]] inline RenderSettings::AntiAliasingDebugSettings
effectiveTemporalAADebugSettings(
    const RenderSettings::AntiAliasingSettings &settings) {
  RenderSettings::AntiAliasingSettings sanitized = settings;
  sanitizeAntiAliasingSettings(sanitized);
  RenderSettings::AntiAliasingDebugSettings effective = sanitized.debug;
  const TemporalAAQualityPreset preset =
      sanitizeTemporalAAQualityPreset(sanitized.qualityPreset);
  if (preset != TemporalAAQualityPreset::Custom) {
    const RenderSettings::AntiAliasingDebugSettings presetDebug =
        temporalAAQualityPresetDebugSettings(preset);
    copyTemporalAAPresetTuning(effective, presetDebug);
  }
  return effective;
}

[[nodiscard]] inline bool isTemporalAAResolvedSceneColorOutput(
    const RenderSettings::AntiAliasingSettings &settings) {
  if (sanitizeAntiAliasingMode(settings.mode) != AntiAliasingMode::TAA) {
    return false;
  }
  const RenderSettings::AntiAliasingDebugSettings aaDebug =
      effectiveTemporalAADebugSettings(settings);
  const AntiAliasingDebugView debugView =
      sanitizeAntiAliasingDebugView(aaDebug.view);
  return debugView == AntiAliasingDebugView::None ||
         debugView == AntiAliasingDebugView::TAAResolved ||
         debugView == AntiAliasingDebugView::TAASceneColorHalfRes ||
         debugView == AntiAliasingDebugView::TAASceneColorQuarterRes ||
         debugView == AntiAliasingDebugView::TAATransmissionMipSource;
}

inline void copyTemporalAAQualityPresetToCustom(
    RenderSettings::AntiAliasingSettings &settings) {
  RenderSettings::AntiAliasingSettings sanitized = settings;
  sanitizeAntiAliasingSettings(sanitized);
  const TemporalAAQualityPreset preset =
      sanitizeTemporalAAQualityPreset(sanitized.qualityPreset);
  settings = sanitized;
  if (preset != TemporalAAQualityPreset::Custom) {
    const RenderSettings::AntiAliasingDebugSettings presetDebug =
        temporalAAQualityPresetDebugSettings(preset);
    copyTemporalAAPresetTuning(settings.debug, presetDebug);
  }
  settings.qualityPreset = TemporalAAQualityPreset::Custom;
}

[[nodiscard]] constexpr ShadowQualityPreset
sanitizeShadowQualityPreset(ShadowQualityPreset preset) noexcept {
  return sanitizeContiguousEnum(preset, ShadowQualityPreset::Ultra,
                                ShadowQualityPreset::Custom);
}

[[nodiscard]] constexpr ShadowPreviewMode
sanitizeShadowPreviewMode(ShadowPreviewMode mode) noexcept {
  return sanitizeContiguousEnum(mode, ShadowPreviewMode::TiledAllCascades,
                                ShadowPreviewMode::SelectedCascade);
}

inline void sanitizeShadowSettings(RenderSettings::ShadowSettings &settings);

inline void applyShadowQualityPreset(RenderSettings::ShadowSettings &settings,
                                     ShadowQualityPreset preset) {
  settings.qualityPreset = sanitizeShadowQualityPreset(preset);
  struct Preset {
    uint32_t cascadeCount;
    uint32_t mapSize;
    Format depthFormat;
    float maxDistance;
    float fadeFraction;
    float splitLambda;
    float constantBias;
    float slopeBias;
    float normalBias;
    uint32_t pcfSamples;
  };
  static constexpr std::array presets{
      Preset{1u, 1024u, kDefaultShadowMapDepthFormat, 80.0f, 0.15f, 0.35f,
             0.0008f, 2.0f, 0.25f, 9u},
      Preset{3u, 2048u, kDefaultShadowMapDepthFormat, 120.0f, 0.12f, 0.30f,
             0.0006f, 1.75f, 0.35f, 16u},
      Preset{4u, 4096u, kDefaultShadowMapDepthFormat, 150.0f, 0.10f, 0.25f,
             0.0005f, 1.5f, 0.50f, 24u},
      Preset{4u, 8192u, Format::D32_FLOAT, 220.0f, 0.08f, 0.50f, 0.00035f,
             1.15f, 0.40f, 32u},
  };
  if (settings.qualityPreset != ShadowQualityPreset::Custom) {
    const Preset &values =
        presets[static_cast<uint8_t>(settings.qualityPreset) - 1u];
    settings.cascadeBlendFraction = 0.08f;
    settings.sdsmTemporalBlend = 0.85f;
    settings.cascadeCount = values.cascadeCount;
    settings.shadowMapSize = values.mapSize;
    settings.depthFormat = values.depthFormat;
    settings.maxDistance = values.maxDistance;
    settings.maxDistanceFadeFraction = values.fadeFraction;
    settings.splitLambda = values.splitLambda;
    settings.constantBias = values.constantBias;
    settings.slopeBias = values.slopeBias;
    settings.normalBias = values.normalBias;
    settings.pcfSampleCount = values.pcfSamples;
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
      finiteClamp(settings.maxDistanceFadeFraction, 0.0f, 1.0f, 0.0f);
  settings.splitLambda = std::clamp(settings.splitLambda, 0.0f, 1.0f);
  settings.cascadeBlendFraction =
      std::clamp(settings.cascadeBlendFraction, 0.0f, 1.0f);
  settings.constantBias =
      std::isfinite(settings.constantBias) ? settings.constantBias : 0.0005f;
  settings.slopeBias = finiteClamp(settings.slopeBias, 0.0f,
                                   std::numeric_limits<float>::max(), 1.5f);
  settings.normalBias = finiteClamp(settings.normalBias, 0.0f,
                                    std::numeric_limits<float>::max(), 0.0f);
  settings.debug.previewMode =
      sanitizeShadowPreviewMode(settings.debug.previewMode);
  settings.pcfSampleCount =
      std::clamp(settings.pcfSampleCount, 1u, kMaxShadowPcfSamples);
  settings.sdsmTemporalBlend =
      std::clamp(settings.sdsmTemporalBlend, 0.0f, 1.0f);
  settings.debug.diagnosticLogLevel = sanitizeContiguousEnum(
      settings.debug.diagnosticLogLevel, LogLevel::Fatal, LogLevel::Trace);
  settings.debug.diagnosticLogIntervalFrames =
      std::max(settings.debug.diagnosticLogIntervalFrames, 1u);
  settings.debug.debugCascadeIndex =
      std::min(settings.debug.debugCascadeIndex, settings.cascadeCount - 1u);
  settings.debug.previewDepthMin =
      finiteClamp(settings.debug.previewDepthMin, 0.0f, 1.0f, 0.0f);
  settings.debug.previewDepthMax =
      finiteClamp(settings.debug.previewDepthMax, 0.0f, 1.0f, 1.0f);
  if (settings.debug.previewDepthMax <= settings.debug.previewDepthMin) {
    settings.debug.previewDepthMax =
        std::min(settings.debug.previewDepthMin + 1.0e-4f, 1.0f);
    settings.debug.previewDepthMin =
        std::min(settings.debug.previewDepthMin,
                 settings.debug.previewDepthMax - 1.0e-4f);
  }
}

[[nodiscard]] inline uint32_t
shadowFilterSampleBudget(uint32_t pcfSampleCount) noexcept {
  return std::clamp(pcfSampleCount, 1u, kMaxShadowPcfSamples);
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
  bool cameraContinuityValid = false;
  bool historyValid = false;
};

[[nodiscard]] inline glm::mat4
makeBackgroundRotationReprojection(const CameraFrameState &camera) {
  if (!camera.historyValid) {
    return glm::mat4(1.0f);
  }
  const glm::mat4 previousView = glm::inverse(camera.currentUnjitteredProj) *
                                 camera.previousUnjitteredViewProj;
  const glm::mat4 currentRotation = glm::mat4(glm::mat3(camera.view));
  const glm::mat4 previousRotation = glm::mat4(glm::mat3(previousView));
  const glm::mat4 currentJitteredRotationViewProj =
      camera.proj * currentRotation;
  const glm::mat4 previousUnjitteredRotationViewProj =
      camera.currentUnjitteredProj * previousRotation;
  return previousUnjitteredRotationViewProj *
         glm::inverse(currentJitteredRotationViewProj);
}

struct TemporalSceneContentState {
  uint64_t lightTopologyVersion = 0u;
  uint64_t lightTransformVersion = 0u;
  uint64_t materialTableVersion = 0u;
  uint64_t environmentVersion = 0u;
  bool operator==(const TemporalSceneContentState &) const = default;
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
  TemporalSceneContentState previousSceneContent{};
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
  TemporalSceneContentState sceneContent{};
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
  const RenderSettings::AntiAliasingDebugSettings aaDebug =
      effectiveTemporalAADebugSettings(antiAliasing);
  const bool taaSelected = mode == AntiAliasingMode::TAA;
  const bool hasRenderExtent =
      desc.renderExtent.x > 0u && desc.renderExtent.y > 0u;
  state.jitterEnabled = taaSelected && aaDebug.jitterEnabled && hasRenderExtent;
  state.jitterFrozen = state.jitterEnabled && aaDebug.freezeJitter;
  TemporalHistoryResetReason resetReason = TemporalHistoryResetReason::None;
  if (!history.initialized) {
    resetReason = TemporalHistoryResetReason::FirstFrame;
  } else if (aaDebug.resetHistoryRequested) {
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
  } else if (history.previousSceneContent != desc.sceneContent) {
    resetReason = TemporalHistoryResetReason::SceneContentChanged;
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
        std::isfinite(aaDebug.taaJitterScale)
            ? std::clamp(aaDebug.taaJitterScale, 0.0f, 1.0f)
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
  state.cameraContinuityValid = history.initialized && !resetHistory;
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
  history.previousSceneContent = desc.sceneContent;
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
  case TemporalHistoryResetReason::SceneContentChanged:
    return "Scene Content Changed";
  }
  return "Unknown";
}

struct alignas(16) ShadowCascadeGpuData {
  glm::mat4 lightViewProj{1.0f};
  glm::vec4 splitDepthTexelSize{0.0f};
};
static_assert(sizeof(ShadowCascadeGpuData) == 80u);

struct alignas(16) ShadowFrameGpuData {
  glm::uvec4 flagsCascadeCountLightIndex{0u};
  glm::vec4 fadeParams{0.0f};
  glm::vec4 sharedBiasParams{0.0f};
  glm::uvec4 sharedSamplerMapSize{kInvalidShadowBindlessIndex,
                                  kInvalidShadowBindlessIndex, 0u, 0u};
  glm::uvec4 cascadeTextureIds{
      kInvalidShadowBindlessIndex, kInvalidShadowBindlessIndex,
      kInvalidShadowBindlessIndex, kInvalidShadowBindlessIndex};
  std::array<ShadowCascadeGpuData, kMaxShadowCascades> cascades{};
};
static_assert(sizeof(ShadowFrameGpuData) == 400u);

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
  ShadowSdsmReductionBackend activeReductionBackend =
      ShadowSdsmReductionBackend::Cpu;
  ShadowSdsmStatus status = ShadowSdsmStatus::Disabled;
  bool reductionFallbackActive = false;
  bool fixedFallbackActive = false;
  uint64_t sourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint32_t gpuResultRingSlotCount = 0u;
  uint32_t gpuResultSelectedSlot = std::numeric_limits<uint32_t>::max();
  bool gpuReductionResultAvailable = false;
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
  float effectiveRangeNear = 0.0f;
  float effectiveRangeFar = 0.0f;
  std::array<float, kMaxShadowCascades + 1u> fixedSplitDepths{};
  std::array<float, kMaxShadowCascades + 1u> minMaxSplitDepths{};
  std::array<float, kMaxShadowCascades + 1u> effectiveSplitDepths{};
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
  uint32_t ambientOcclusionTexId = 0;
  uint32_t ambientOcclusionSamplerId = 0;
  uint32_t ambientOcclusionFlags = 0;
  uint32_t ambientOcclusionReserved0 = 0;
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
  uint32_t materialCoverageSamplerId = kInvalidSamplerBindlessIndex;
  uint32_t materialDataSamplerId = kInvalidSamplerBindlessIndex;
  uint32_t materialSamplerReserved0 = 0;
  uint32_t materialSamplerReserved1 = 0;
  uint32_t materialSamplerReserved2 = 0;
  uint64_t ddgiFrameBufferAddress = 0u;
  uint32_t ddgiFlags = 0u;
  uint32_t ddgiDebugView = 0u;
  uint32_t ddgiReserved0 = 0u;
  uint32_t ddgiReserved1 = 0u;
  glm::mat4 previousViewProj{1.0f};
  glm::uvec4 sceneDepthPyramidInfo{0u};
  [[nodiscard]] bool
  operator==(const ForwardSceneFrameData &other) const noexcept {
    return std::memcmp(this, &other, sizeof(ForwardSceneFrameData)) == 0;
  }
};
static_assert(sizeof(ForwardSceneFrameData) == 488u);

struct ForwardSceneGpuData {
  BufferHandle buffer{};
  ForwardSceneFrameData frameData{};
  ForwardSceneFrameData postTaaFrameData{};
  uint64_t frameDataAddress = 0;
  uint64_t postTaaFrameDataAddress = 0;
  uint64_t directionalLightBufferAddress = 0;
  uint64_t localLightBufferAddress = 0;
  uint64_t shadowFrameBufferAddress = 0;
  uint32_t directionalLightCount = 0;
  uint32_t localLightCount = 0;
  uint32_t shadowFlags = 0;
  std::span<const BufferHandle> indirectDependencyBuffers{};
  std::span<const TextureHandle> indirectDependencyTextures{};
};

struct DDGIFrameGpuDataHandle {
  BufferHandle buffer{};
  uint64_t bufferAddress = 0u;
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const TextureHandle> dependencyTextures{};
  uint32_t activeVolumeCount = 0u;
  uint32_t flags = 0u;
  DDGIDebugView debugView = DDGIDebugView::None;
  std::array<DDGIEffectiveVolumeKey, kMaxDDGIVolumes> volumeKeys{};
  std::array<DDGIVolumeId, kMaxDDGIVolumes> volumeIds{};
  std::array<uint32_t, kMaxDDGIVolumes> probeCounts{};
  std::array<float, kMaxDDGIVolumes> minimumProbeSpacing{};
  std::array<DDGICaptureMetadata, kMaxDDGIVolumes> captureMetadata{};
  uint64_t coverageGeneration = 0u;
  uint64_t sceneBoundsGeneration = 0u;
  BufferHandle diagnosticBuffer{};
  uint64_t diagnosticRayAddress = 0u;
  uint32_t diagnosticRayCount = 0u;
};

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
  uint32_t meshletDispatches = 0;
  uint32_t meshletTaskGroups = 0;
  uint32_t meshletCandidateCount = 0;
  uint32_t meshletModeRequired = 0;
  uint32_t meshletModeActive = 0;
  uint32_t meshletHybridActive = 0;
  uint32_t meshletHybridClassicBatches = 0;
  uint32_t meshletHybridClassicInstances = 0;
  uint32_t meshletHybridCoverageClassicBatches = 0;
  uint32_t meshletHybridCoverageClassicInstances = 0;
  uint32_t meshletHybridMeshletBatches = 0;
  uint32_t meshletHybridMeshletInstances = 0;
  uint32_t autoLodActive = 0;
  uint32_t autoLodHistoryReset = 0;
  uint32_t autoLodTransitions = 0;
  uint32_t autoLodLod0Instances = 0;
  uint32_t autoLodLod1Instances = 0;
  uint32_t meshletRejectedMissingFeature = 0;
  uint32_t meshletRejectedMissingAssetData = 0;
  uint32_t meshletRejectedIncompatibleFrame = 0;
  uint32_t depthPrepassDraws = 0;
  uint32_t depthPyramidLevels = 0;
  uint32_t depthPrepassEnabled = 0;
  float gpuTimeMs = 0.0f;
  uint64_t gpuTimingSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint32_t gpuTimingAvailable = 0u;
};

struct ShadowFrameMetrics {
  uint32_t cascadeCount = 0;
  uint32_t shadowMapSize = 0;
  uint32_t frameGpuBytes = 0;
  uint32_t totalDraws = 0;
  uint32_t totalCulledDraws = 0;
  uint32_t totalIndexCountEstimate = 0;
  uint32_t staticCasterEntries = 0;
  uint32_t dynamicCasterEntries = 0;
  uint32_t staticCacheReused = 0;
  uint32_t staticBatchTemplateCount = 0;
  uint32_t shadowBatchEntryCount = 0;
  uint32_t shadowInstanceRemapCount = 0;
  uint32_t submittedDrawItemCount = 0;
  uint32_t indirectCommandCount = 0;
  uint32_t drawPacketBytes = 0;
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
  uint32_t sdsmComputePassCount = 0;
  uint32_t sdsmReadbackBytes = 0;
  uint32_t sdsmReductionSourceSamples = 0;
  ShadowSdsmReductionBackend sdsmActiveReductionBackend =
      ShadowSdsmReductionBackend::Cpu;
  float gpuTimeMs = 0.0f;
  float depthGpuTimeMs = 0.0f;
  float sdsmGpuTimeMs = 0.0f;
  float sdsmCpuReductionTimeMs = 0.0f;
  uint64_t gpuTimingSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t depthGpuTimingSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t sdsmGpuTimingSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint32_t gpuTimingAvailable = 0;
  uint32_t depthGpuTimingAvailable = 0;
  uint32_t sdsmGpuTimingAvailable = 0;
};

struct VisibilityFrameMetrics {
  uint32_t cpuMainCandidates = 0;
  uint32_t cpuMainVisibleCandidates = 0;
  uint32_t cpuMainRejected = 0;
  uint32_t gpuMainCandidates = 0;
  uint32_t gpuMainVisibleCandidates = 0;
  uint32_t gpuMainRejectedFrustum = 0;
  uint32_t gpuMainRejectedOcclusion = 0;
  uint32_t gpuOutputOverflowCount = 0;
  uint32_t gpuMainReadbackAvailable = 0;
  uint32_t gpuMainReadbackSourceFrame = 0;
  uint32_t gpuMainReadbackStaleFrameCount = 0;
  uint32_t gpuMainReadbackErrorCount = 0;
  uint32_t gpuMainReadbackVisibleCandidates = 0;
  uint32_t gpuMainVisibleListMismatches = 0;
  uint32_t gpuIndirectDrawUsed = 0;
  uint32_t gpuIndirectDrawFallback = 0;
  uint32_t gpuIndirectDrawCommands = 0;
  uint32_t gpuIndirectDrawReadbackCommands = 0;
  uint32_t gpuIndirectDrawReadbackTombstoned = 0;
  uint32_t gpuIndirectDrawReadbackVisible = 0;
  uint32_t meshletCandidates = 0;
  uint32_t meshletRejectedFrustum = 0;
  uint32_t meshletRejectedCone = 0;
  uint32_t meshletRejectedOcclusion = 0;
  uint32_t meshletOcclusionAvailable = 0;
  uint32_t meshletOcclusionMode = 0;
  uint32_t meshletOcclusionSourceFrame = 0;
  uint32_t meshletOcclusionSourceAge = 0;
  uint32_t currentFrameHiZActive = 0;
  uint32_t meshletPreTaskCompactionActive = 0;
  uint32_t meshletPreTaskCandidatesInput = 0;
  uint32_t meshletPreTaskCandidatesOutput = 0;
  uint32_t meshletPreTaskTaskGroupsInput = 0;
  uint32_t meshletPreTaskTaskGroupsOutput = 0;
  uint32_t meshletPreTaskTaskGroupsSaved = 0;
  uint32_t meshletPreTaskOverflowCount = 0;
  uint32_t meshletPreTaskMismatchCount = 0;
  uint32_t meshletPayloadOverflowCount = 0;
  uint32_t meshletReadbackAvailable = 0;
  uint32_t meshletReadbackSourceFrame = 0;
  uint32_t meshletReadbackStaleFrameCount = 0;
  uint32_t meshletReadbackErrorCount = 0;
  uint32_t meshletEmitted = 0;
  uint32_t meshletTaskGroupsExecuted = 0;
  uint32_t shadowCpuCandidates = 0;
  uint32_t shadowCpuRejected = 0;
  uint32_t indirectDrawCount = 0;
  uint32_t indirectMeshDispatchCount = 0;
  uint32_t uncertainVisible = 0;
  uint32_t occlusionAvailable = 0;
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
  uint32_t motionVectorDepthReprojectionPassCount = 0u;
  uint32_t velocityPassCount = 0u;
  uint32_t velocityDrawCount = 0u;
  uint32_t velocityInstanceCount = 0u;
  uint32_t velocityPreviousTransformValidCount = 0u;
  uint32_t velocityMissingPreviousTransformCount = 0u;
  uint32_t velocityAnimatedResponsiveCount = 0u;
  uint32_t velocityAnimatedPreviousGeometryCount = 0u;
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
  uint32_t reactiveMotionUncertainDrawCount = 0u;
  uint32_t reactiveSkippedTessellatedDrawCount = 0u;
  uint32_t motionClassTextureCount = 0u;
  uint32_t historyColorTextureCount = 0u;
  uint32_t taaResolvePassCount = 0u;
  uint32_t taaCopyBackPassCount = 0u;
  uint32_t taaPostResolveSceneColorMipPassCount = 0u;
  uint32_t taaTransmissionStaleSceneColorFrameCount = 0u;
  uint32_t taaResolveGpuTimingAvailable = 0u;
  uint32_t taaDebugGpuTimingAvailable = 0u;
  uint32_t taaSceneColorDownsampleGpuTimingAvailable = 0u;
  uint32_t taaTransmissionGpuTimingAvailable = 0u;
  uint32_t taaTransmissionMipDebugPassCount = 0u;
  uint32_t transparentTransmissionFeedbackRefreshCount = 0u;
  uint32_t transparentTransmissionBlendDrawCount = 0u;
  uint32_t transparentTransmissionFeedbackSourceAvailable = 0u;
  uint32_t taaTransparentPostTaaDrawCount = 0u;
  uint32_t taaTransparentPostTaaMeshDrawCount = 0u;
  uint32_t taaTransparentPostTaaContributorDrawCount = 0u;
  uint32_t taaTransparentPostTaaFixedDrawCount = 0u;
  uint32_t taaTransparentPostSpatialAAPassCount = 0u;
  uint32_t taaResolveWidth = 0u;
  uint32_t taaResolveHeight = 0u;
  uint32_t spatialAAWidth = 0u;
  uint32_t spatialAAHeight = 0u;
  uint32_t spatialAAPassCount = 0u;
  uint32_t spatialAAEdgePassCount = 0u;
  uint32_t spatialAABlendPassCount = 0u;
  uint32_t spatialAANeighborhoodPassCount = 0u;
  uint32_t spatialAACopyBackPassCount = 0u;
  uint32_t spatialAACleanupFrameCount = 0u;
  uint32_t spatialAADebugPassCount = 0u;
  uint32_t spatialAAGpuTimingAvailable = 0u;
  uint32_t msaaSampleCount = 1u;
  uint32_t msaaWidth = 0u;
  uint32_t msaaHeight = 0u;
  uint32_t msaaColorTextureCount = 0u;
  uint32_t msaaDepthTextureCount = 0u;
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
  uint64_t previousSceneDepthTextureBytes = 0u;
  uint64_t velocityPassBandwidthEstimateBytes = 0u;
  uint64_t velocityDebugBandwidthEstimateBytes = 0u;
  uint64_t reactiveMaskTextureBytes = 0u;
  uint64_t reactiveMaskTotalBytes = 0u;
  uint64_t reactiveMaskPassBandwidthEstimateBytes = 0u;
  uint64_t motionClassTotalBytes = 0u;
  uint64_t historyColorTextureBytes = 0u;
  uint64_t historyColorTotalBytes = 0u;
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
  float cameraDirectionDelta = 0.0f;
  float jitterDeltaMagnitude = 0.0f;
  float velocityMissingPreviousRatio = 0.0f;
  float velocityEdgeDiscontinuityEstimate = 0.0f;
  float taaJitterScale = 0.75f;
  float taaCurrentFrameWeight = 0.045f;
  float taaHistoryFrameWeight = 0.955f;
  float taaHistoryValidPercent = 0.0f;
  float taaSharpenStrength = 0.14f;
  float taaSharpenConfidenceThreshold = 0.82f;
  float taaMaterialMipBias = 0.0f;
  float taaDepthDiscontinuityThreshold = 0.01f;
  float taaVelocityRejectionThreshold = 1.5f;
  float taaVelocityBlendScale = 0.22f;
  float taaMotionCurrentWeight = 0.22f;
  float taaDisocclusionCurrentWeight = 0.62f;
  float taaClampCurrentWeight = 0.38f;
  float taaClampBlendAttenuation = 0.35f;
  float taaVarianceGamma = 1.85f;
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
  float taaTransmissionJitterMinLod = 0.0f;
  float taaTransmissionDepthBiasConstant = 0.0f;
  float taaTransparentEdgeJitterEstimate = 0.0f;
  float spatialAAGpuTimeMs = 0.0f;
  float msaaResolveGpuTimeMs = 0.0f;
  float spatialAAEdgePixelEstimate = 0.0f;
  float spatialAAModifiedPixelEstimate = 0.0f;
  TemporalAAClampMode taaClampMode = TemporalAAClampMode::VarianceYCoCg;
  TemporalAAHdrWeightingMode taaHdrWeightingMode =
      TemporalAAHdrWeightingMode::Luminance;
  TemporalAAVelocityDilationMode taaVelocityDilationMode =
      TemporalAAVelocityDilationMode::ClosestDepth;
  TemporalAAHistoryFilterMode taaHistoryFilterMode =
      TemporalAAHistoryFilterMode::CatmullRom;
  TemporalAAQualityPreset taaQualityPreset = TemporalAAQualityPreset::Quality;
  TemporalHistoryResetReason historyResetReason =
      TemporalHistoryResetReason::None;
  bool jitterEnabled = false;
  bool jitterFrozen = false;
  bool taaQualityValidationInvalidatedByFrozenJitter = false;
  bool jitterOutOfBounds = false;
  bool cameraContinuityValid = false;
  bool historyValid = false;
  bool temporalDataValid = false;
  bool motionVectorAllocated = false;
  bool motionVectorFormatSupported = false;
  bool previousMotionVectorValid = false;
  bool previousSceneDepthValid = false;
  bool motionVectorGraphPublished = false;
  bool previousMotionVectorGraphPublished = false;
  bool previousSceneDepthGraphPublished = false;
  bool motionVectorDepthReprojectionGenerated = false;
  bool reactiveMaskAllocated = false;
  bool reactiveMaskGraphPublished = false;
  bool reactiveMaskFormatSupported = false;
  bool opaqueVelocityGenerated = false;
  bool velocityDebugViewRendered = false;
  bool previousTransformCacheValid = false;
  bool taaResolvedSceneColorPublished = false;
  bool taaDebugViewRendered = false;
  bool taaSharpenEnabled = false;
  bool taaSharpenActive = false;
  bool taaMaterialMipBiasEnabled = false;
  bool taaMaterialMipBiasApplied = false;
  bool taaHistoryValidityDebugViewRendered = false;
  bool taaOutOfBoundsFallbackEnabled = false;
  bool taaBilinearHistorySampling = false;
  bool taaDepthRejectionEnabled = false;
  bool taaPreviousDepthRejectionEnabled = false;
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
  bool taaVelocityDilationDebugViewRendered = false;
  bool taaPreviousVelocityDebugViewRendered = false;
  bool taaHdrWeightDebugViewRendered = false;
  bool taaHistoryFilterDeltaDebugViewRendered = false;
  bool taaDisocclusionFallbackDebugViewRendered = false;
  bool taaSplitCompareDebugViewRendered = false;
  bool taaTemporalConfidenceDebugViewRendered = false;
  bool taaPreviousDepthRejectionDebugViewRendered = false;
  bool taaPostResolveSceneColorMipChainGenerated = false;
  bool taaTransmissionPostResolveSceneColorConsumed = false;
  bool taaTransmissionStableVisibilityDepth = false;
  bool taaSceneColorMipDebugViewRendered = false;
  bool taaTransmissionMipDebugViewRendered = false;
  bool taaTransparentPostSpatialCleanupEnabled = false;
  bool taaTransparentPostSpatialCleanupActive = false;
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
  bool msaaSample4ColorSupported = false;
  bool msaaSample4DepthSupported = false;
  bool msaaSample8ColorSupported = false;
  bool msaaSample8DepthSupported = false;
  bool msaaDepthResolveMinSupported = false;
  bool msaaAlphaToCoverageSupported = false;
  bool msaaSampleRateShadingSupported = false;
  bool msaaSpatialCleanupEnabled = false;
  bool msaaSpatialCleanupActive = false;
  PresentationAAUnsupportedReason msaaUnsupportedReason =
      PresentationAAUnsupportedReason::None;
  AlphaCoveragePolicy msaaAlphaCoveragePolicy = AlphaCoveragePolicy::Off;
  TransparencyAAPolicy msaaTransparencyPolicy =
      TransparencyAAPolicy::InheritCoverage;
  bool spatialAAEdgesDebugViewRendered = false;
  bool spatialAABlendWeightsDebugViewRendered = false;
  bool spatialAACleanupMaskDebugViewRendered = false;
  bool spatialAASplitCompareDebugViewRendered = false;
  bool taaStabilityDiagnosticsDebugViewRendered = false;
  bool taaStabilityOwnershipDebugViewRendered = false;
  bool taaPatchProbeDebugViewRendered = false;
  bool taaMotionFilterDebugViewRendered = false;
  bool taaStaticFrameVelocitySanitizationEnabled = false;
};

struct AmbientOcclusionFrameMetrics {
  AmbientOcclusionPreset activePreset = AmbientOcclusionPreset::Balanced;
  AmbientOcclusionDisabledReason disabledReason =
      AmbientOcclusionDisabledReason::None;
  Format normalFormat = Format::Count;
  Format ambientOcclusionFormat = Format::Count;
  uint32_t width = 0u;
  uint32_t height = 0u;
  uint32_t normalPrepassDraws = 0u;
  uint32_t depthPrefilterPassCount = 0u;
  uint32_t mainPassCount = 0u;
  uint32_t temporalPassCount = 0u;
  uint32_t depthMipCount = 0u;
  uint32_t requestedSliceCount = 0u;
  uint32_t requestedStepCount = 0u;
  uint32_t requestedDenoisePassCount = 0u;
  uint32_t sliceCount = 0u;
  uint32_t stepCount = 0u;
  uint32_t denoisePassCount = 0u;
  uint32_t textureCount = 0u;
  uint32_t normalTextureCount = 0u;
  uint32_t ambientOcclusionTextureCount = 0u;
  uint32_t normalTextureAllocationCount = 0u;
  uint32_t normalTextureReallocationCount = 0u;
  uint32_t ambientOcclusionTextureAllocationCount = 0u;
  uint32_t ambientOcclusionTextureReallocationCount = 0u;
  uint64_t normalTextureBytes = 0u;
  uint64_t ambientOcclusionTextureBytes = 0u;
  uint64_t depthPrefilterTextureBytes = 0u;
  uint64_t edgeTextureBytes = 0u;
  uint64_t scratchTextureBytes = 0u;
  uint64_t totalTextureBytes = 0u;
  float strength = 1.0f;
  float gpuTimeMs = 0.0f;
  uint64_t gpuTimingSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint32_t gpuTimingAvailable = 0u;
  bool enabled = false;
  bool active = false;
  bool normalsAllocated = false;
  bool ambientOcclusionAllocated = false;
  bool normalGraphPublished = false;
  bool ambientOcclusionGraphPublished = false;
  bool temporalAccumulationEnabled = false;
  bool temporalAccumulationActive = false;
  bool temporalHistoryInvalidated = false;
  bool temporalHistoryValid = false;
  bool temporalMotionVectorsConsumed = false;
  bool temporalMotionClassConsumed = false;
  bool temporalPreviousDepthConsumed = false;
  bool scalarAoAvailable = false;
  bool bentNormalAvailable = false;
};

struct HDRPostProcessFrameMetrics {
  uint32_t width = 0u;
  uint32_t height = 0u;
  uint32_t bloomMipCount = 0u;
  uint32_t bloomPassCount = 0u;
  uint32_t luminancePassCount = 0u;
  uint32_t adaptationPassCount = 0u;
  uint32_t textureCount = 0u;
  uint32_t exposureTextureAllocationCount = 0u;
  uint32_t exposureTextureReallocationCount = 0u;
  uint32_t exposureHistoryAllocationCount = 0u;
  uint32_t exposureHistoryReallocationCount = 0u;
  uint32_t gpuTimingAvailable = 0u;
  uint64_t textureBytes = 0u;
  uint64_t gpuTimingSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t exposureTelemetrySourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  float adaptedExposureEv = 0.0f;
  float automaticExposureEv = 0.0f;
  float exposureTargetEv = 0.0f;
  float exposureMeteredLuminance = 0.0f;
  float effectiveExposureEv = 0.0f;
  float exposureInvalidSampleFraction = 0.0f;
  float gpuTimeMs = 0.0f;
  uint32_t exposureTelemetryStaleFrames = 0u;
  uint32_t exposureTelemetryPendingSlots = 0u;
  uint32_t exposureTelemetryDroppedSamples = 0u;
  bool bloomEnabled = false;
  bool bloomActive = false;
  bool adaptationEnabled = false;
  bool adaptationActive = false;
  bool exposureHistoryValid = false;
  bool exposureTelemetryAvailable = false;
};

[[nodiscard]] inline AntiAliasingFrameMetrics
makeAntiAliasingFrameMetrics(const CameraFrameState &camera) noexcept {
  const glm::vec2 jitterDelta =
      camera.jitterPixelOffset - camera.previousJitterPixelOffset;
  const glm::mat4 previousView = glm::inverse(camera.currentUnjitteredProj) *
                                 camera.previousUnjitteredViewProj;
  const glm::vec3 currentForward = -glm::normalize(
      glm::vec3(camera.view[0][2], camera.view[1][2], camera.view[2][2]));
  const glm::vec3 previousForward = -glm::normalize(
      glm::vec3(previousView[0][2], previousView[1][2], previousView[2][2]));
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
      .cameraDirectionDelta =
          camera.cameraContinuityValid
              ? glm::length(currentForward - previousForward)
              : 0.0f,
      .jitterDeltaMagnitude = glm::length(jitterDelta),
      .historyResetReason = camera.historyResetReason,
      .jitterEnabled = camera.jitterEnabled,
      .jitterFrozen = camera.jitterFrozen,
      .taaQualityValidationInvalidatedByFrozenJitter = camera.jitterFrozen,
      .jitterOutOfBounds = camera.jitterOutOfBounds,
      .cameraContinuityValid = camera.cameraContinuityValid,
      .historyValid = camera.historyValid,
      .temporalDataValid = camera.temporalDataValid,
  };
}

struct RenderFrameMetrics {
  struct AssetStreamingFrameMetrics {
    uint32_t cpuCompletions = 0u;
    uint32_t cpuWorkers = 0u;
    uint32_t cpuActiveWorkerLimit = 0u;
    uint32_t cpuInteractiveMode = 0u;
    uint32_t cpuQueuedJobs = 0u;
    uint32_t cpuRunningJobs = 0u;
    uint32_t cpuRunningIo = 0u;
    uint32_t cpuRunningDecode = 0u;
    uint32_t cpuRunningCook = 0u;
    uint32_t cpuRunningTranscode = 0u;
    uint32_t cpuRunningMetadata = 0u;
    uint32_t dedicatedCopyQueue = 0u;
    uint32_t gpuMaterialized = 0u;
    uint32_t published = 0u;
    uint32_t cancelled = 0u;
    uint32_t failed = 0u;
    uint32_t scenePatches = 0u;
    uint32_t sceneCommits = 0u;
    uint32_t deferredCpuCompletions = 0u;
    uint32_t publicationDeadlineExceeded = 0u;
    double publicationMainThreadMilliseconds = 0.0;
    double publicationMaxOperationMilliseconds = 0.0;
    uint64_t cpuInFlightBytes = 0u;
    uint64_t uploadBytes = 0u;
    uint64_t submittedJobs = 0u;
    uint64_t completedJobs = 0u;
    uint64_t cancelledJobs = 0u;
    uint64_t rejectedJobs = 0u;
  } assets{};
  uint64_t frameIndex = 0u;
  OpaqueFrameMetrics opaque{};
  ShadowFrameMetrics shadow{};
  VisibilityFrameMetrics visibility{};
  AntiAliasingFrameMetrics antiAliasing{};
  AmbientOcclusionFrameMetrics ambientOcclusion{};
  HDRPostProcessFrameMetrics hdrPostProcess{};
  RayTracingSceneFrameMetrics rayTracingScene{};
  DDGIFrameMetrics ddgi{};
  struct TransparentFrameMetrics {
    uint32_t meshDraws = 0;
    uint32_t contributorSortableDraws = 0;
    uint32_t contributorFixedDraws = 0;
    uint32_t pickDraws = 0;
  } transparent{};
};

struct ResolvedGeometryWorkMetrics {
  uint32_t instanceCandidates = 0u;
  uint32_t visibleInstances = 0u;
  uint32_t indirectCommands = 0u;
  uint32_t visibleIndirectCommands = 0u;
  uint32_t meshletCandidates = 0u;
  uint32_t emittedMeshlets = 0u;
  uint32_t executedMeshletTaskGroups = 0u;
  bool mainReadbackAvailable = false;
  bool indirectReadbackAvailable = false;
  bool meshletReadbackAvailable = false;
};

[[nodiscard]] constexpr ResolvedGeometryWorkMetrics
resolveGeometryWorkMetrics(const RenderFrameMetrics &metrics) noexcept {
  const bool mainReadbackAvailable =
      metrics.visibility.gpuMainReadbackAvailable != 0u;
  const bool indirectReadbackAvailable =
      mainReadbackAvailable && metrics.visibility.gpuIndirectDrawUsed != 0u;
  const bool meshletReadbackAvailable =
      metrics.visibility.meshletReadbackAvailable != 0u;
  return {
      .instanceCandidates = mainReadbackAvailable
                                ? metrics.visibility.gpuMainCandidates
                                : metrics.opaque.totalInstances,
      .visibleInstances = mainReadbackAvailable
                              ? metrics.visibility.gpuMainVisibleCandidates
                              : metrics.opaque.visibleInstances,
      .indirectCommands =
          indirectReadbackAvailable
              ? metrics.visibility.gpuIndirectDrawReadbackCommands
              : metrics.opaque.indirectCommands,
      .visibleIndirectCommands =
          indirectReadbackAvailable
              ? metrics.visibility.gpuIndirectDrawReadbackVisible
              : metrics.opaque.indirectCommands,
      .meshletCandidates = metrics.visibility.meshletCandidates,
      .emittedMeshlets = meshletReadbackAvailable
                             ? metrics.visibility.meshletEmitted
                             : metrics.visibility.meshletCandidates,
      .executedMeshletTaskGroups =
          meshletReadbackAvailable
              ? metrics.visibility.meshletTaskGroupsExecuted
              : metrics.opaque.meshletTaskGroups,
      .mainReadbackAvailable = mainReadbackAvailable,
      .indirectReadbackAvailable = indirectReadbackAvailable,
      .meshletReadbackAvailable = meshletReadbackAvailable,
  };
}

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

struct DDGIProbeInspectRequest {
  uint64_t requestId = 0u;
  DDGIVolumeId volume = kInvalidDDGIVolumeId;
  uint32_t probeId = 0u;
  uint32_t rayCount = 128u;
};

struct DDGIProbeInspectResult {
  uint64_t requestId = 0u;
  uint64_t sceneId = 0u;
  DDGIVolumeId volume = kInvalidDDGIVolumeId;
  uint32_t probeId = 0u;
  uint32_t volumeSlot = 0u;
  glm::uvec3 volumeCoordinate{0u};
  glm::uvec2 irradianceAtlasCoordinate{0u};
  glm::uvec2 distanceAtlasCoordinate{0u};
  uint32_t probeState = 0u;
  uint32_t lastSuccessfulUpdate = 0u;
  uint32_t updateAge = 0u;
  uint32_t hitCount = 0u;
  uint32_t missCount = 0u;
  uint32_t rejectedAlphaCount = 0u;
  uint32_t rejectedBackfaceCount = 0u;
  uint32_t candidateOverflowCount = 0u;
  uint32_t diagnosticEventOverflowCount = 0u;
  uint32_t rayCount = 0u;
  glm::vec3 nominalWorldPosition{0.0f};
  glm::vec3 relocatedWorldPosition{0.0f};
  glm::vec3 irradiance{0.0f};
  glm::vec2 distanceMoments{0.0f};
  uint64_t layoutGeneration = 0u;
  uint64_t resourceGeneration = 0u;
  uint64_t deviceEpoch = 0u;
  uint64_t submissionSequence = 0u;
  bool active = false;
  bool inside = false;
  bool valid = false;
};

enum class RenderCaptureValueKind : uint8_t {
  Color,
  LinearHdrColor,
  Depth,
  ShadowDepth,
  Normal,
  Velocity,
  Mask,
  Scalar,
  DebugPreview,
};

enum class RenderCaptureLifetimeClass : uint8_t {
  FrameSharedRingTexture,
  FeaturePersistentTexture,
  ToolCaptureTexture,
  CaptureCopyTexture,
};

struct RenderCapturePoint {
  std::string_view name{};
  uint32_t version = 1u;
  TextureHandle texture{};
  Format format = Format::Count;
  TextureDimensions dimensions{};
  uint64_t frameIndex = 0u;
  uint32_t mip = 0u;
  uint32_t layer = 0u;
  RenderCaptureValueKind kind = RenderCaptureValueKind::Color;
  RenderCaptureLifetimeClass lifetime =
      RenderCaptureLifetimeClass::FrameSharedRingTexture;
  std::string_view colorSpace{};
  std::string_view defaultCompareProfile{};
  std::string_view producerPassLabel{};
  std::string_view debugLabel{};
  DDGICaptureMetadata ddgiMetadata{};
};

class RenderCaptureRequest {
public:
  explicit RenderCaptureRequest(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : names_(memory != nullptr ? memory : std::pmr::get_default_resource()) {}
  void clear() { names_.clear(); }
  void request(std::string_view name) {
    if (name.empty() || contains(name)) {
      return;
    }
    names_.push_back(name);
  }
  [[nodiscard]] bool contains(std::string_view name) const noexcept {
    for (const std::string_view entry : names_) {
      if (entry == name) {
        return true;
      }
    }
    return false;
  }
  [[nodiscard]] bool empty() const noexcept { return names_.empty(); }
  [[nodiscard]] std::span<const std::string_view> names() const noexcept {
    return std::span<const std::string_view>(names_.data(), names_.size());
  }

private:
  std::pmr::vector<std::string_view> names_;
};

class RenderCaptureRegistry {
public:
  explicit RenderCaptureRegistry(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : points_(memory != nullptr ? memory : std::pmr::get_default_resource()) {
  }
  void clear() { points_.clear(); }
  void publish(const RenderCapturePoint &point) {
    if (point.name.empty() || !nuri::isValid(point.texture)) {
      return;
    }
    for (RenderCapturePoint &entry : points_) {
      if (entry.name == point.name) {
        entry = point;
        return;
      }
    }
    points_.push_back(point);
  }
  [[nodiscard]] const RenderCapturePoint *
  find(std::string_view name) const noexcept {
    for (const RenderCapturePoint &point : points_) {
      if (point.name == name) {
        return &point;
      }
    }
    return nullptr;
  }
  [[nodiscard]] std::span<const RenderCapturePoint> points() const noexcept {
    return std::span<const RenderCapturePoint>(points_.data(), points_.size());
  }

private:
  std::pmr::vector<RenderCapturePoint> points_;
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
  AmbientOcclusion = 1u << 11u,
  PresentCapture = 1u << 12u,
  MotionClass = 1u << 13u,
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
        FrameTextureRequirementFlags::SceneColorMipChain;

struct FrameSharedResources {
  const SceneDrawDatabase *sceneDrawDatabase = nullptr;
  std::optional<RayTracingSceneFrameView> rayTracingScene{};
  std::optional<DDGIFrameGpuDataHandle> ddgiFrameGpuData{};
  std::optional<ForwardSceneGpuData> forwardSceneGpuData{};
  std::optional<MaterialTableGpuData> materialTableGpuData{};
  std::optional<AnimationSceneFrameData> animationSceneGpuData{};
  std::optional<ShadowFrameGpuDataHandle> shadowFrameGpuData{};
  std::optional<ShadowSdsmGpuReduceTargetHandle> shadowSdsmGpuReduceTarget{};
  ComputePipelineHandle shadowSdsmGpuReducePipeline{};
  std::optional<ShadowDebugFrameData> shadowDebugFrameData{};
  FrameTextureRequirementFlags textureRequirements =
      kBaselineFrameTextureRequirements;
  FrameTextureRequirementFlags historyWriteRequirements =
      FrameTextureRequirementFlags::None;
  TextureHandle sceneDepthTexture{};
  TextureHandle previousSceneDepthTexture{};
  TextureHandle transmissionVisibilityDepthTexture{};
  RenderGraphTextureId sceneDepthGraphTexture{};
  RenderGraphTextureId previousSceneDepthGraphTexture{};
  RenderGraphTextureId transmissionVisibilityDepthGraphTexture{};
  TextureHandle msaaSceneDepthTexture{};
  RenderGraphTextureId msaaSceneDepthGraphTexture{};
  TextureHandle normalTexture{};
  RenderGraphTextureId normalGraphTexture{};
  TextureHandle ambientOcclusionTexture{};
  TextureHandle previousAmbientOcclusionTexture{};
  RenderGraphTextureId ambientOcclusionGraphTexture{};
  std::array<TextureHandle, kMaxSceneDepthPyramidLevels>
      sceneDepthPyramidTextures{};
  std::array<RenderGraphTextureId, kMaxSceneDepthPyramidLevels>
      sceneDepthPyramidGraphTextures{};
  std::optional<uint64_t> sceneDepthPyramidSourceFrameIndex{};
  std::optional<glm::mat4> sceneDepthPyramidSourceViewProj{};
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
  TextureHandle presentCaptureTexture{};
  RenderGraphTextureId presentCaptureGraphTexture{};
  TextureHandle historyColorReadTexture{};
  TextureHandle historyColorWriteTexture{};
  bool historyColorReadValid = false;
  TextureHandle exposureReadTexture{};
  TextureHandle exposureWriteTexture{};
  RenderGraphTextureId exposureReadGraphTexture{};
  RenderGraphTextureId exposureWriteGraphTexture{};
  TextureHandle motionVectorTexture{};
  TextureHandle previousMotionVectorTexture{};
  TextureHandle reactiveMaskTexture{};
  TextureHandle motionClassTexture{};
  RenderGraphTextureId motionVectorGraphTexture{};
  RenderGraphTextureId previousMotionVectorGraphTexture{};
  RenderGraphTextureId reactiveMaskGraphTexture{};
  RenderGraphTextureId motionClassGraphTexture{};
  RenderGraphTextureId opaquePickGraphTexture{};
  RenderGraphTextureId opaquePickDepthGraphTexture{};
  std::optional<LightId> selectedLightId{};
  std::optional<LightId> selectedShadowLightId{};
  std::optional<DDGIVolumeId> selectedDDGIVolumeId{};
  bool transparentStageEnabled = false;
  bool transparentTransmissionStageEnabled = false;
  bool exposureHistoryValid = false;
};

enum class TransparentStageFeedbackRefreshMode : uint8_t {
  BeforeEachDraw = 0,
  OnceBeforeFirstDraw = 1,
};

using TransparentStageAppendFeedbackRefreshFn = Result<bool, std::string> (*)(
    void *user, RenderFrameContext &frame, RenderGraphBuilder &graph);

struct TransparentStageFeedbackRefresh {
  void *user = nullptr;
  TransparentStageAppendFeedbackRefreshFn appendRefresh = nullptr;
  TransparentStageFeedbackRefreshMode mode =
      TransparentStageFeedbackRefreshMode::BeforeEachDraw;
};

struct TransparentStageSortableDraw {
  DrawItem draw{};
  float sortDepth = 0.0f;
  uint32_t stableOrder = 0;
  bool requiresFrameColorFeedback = false;
  uint32_t dependencyOffset = 0;
  uint32_t dependencyCount = 0;
};

struct TransparentStageContribution {
  std::span<const TransparentStageSortableDraw> sortableDraws{};
  std::span<const DrawItem> fixedDraws{};
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const TextureHandle> textureReads{};
  TransparentStageFeedbackRefresh feedbackRefresh{};
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
  RenderScene *scene = nullptr;
  CameraFrameState camera{};
  PresentationAAPlan presentationAA{};
  GpuTimingReport gpuTiming{};
  TemporalFrameService *temporalFrameService = nullptr;
  const RenderSettings *settings = nullptr;
  RenderSettings resolvedSettings{};
  bool settingsResolved = false;
  RenderFrameMetrics metrics{};
  std::optional<OpaquePickRequest> opaquePickRequest{};
  std::optional<OpaquePickResult> opaquePickResult{};
  std::optional<ShadowInspectRequest> shadowInspectRequest{};
  std::optional<ShadowInspectResult> shadowInspectResult{};
  std::optional<DDGIProbeInspectRequest> ddgiProbeInspectRequest{};
  std::optional<DDGIProbeInspectResult> ddgiProbeInspectResult{};
  FrameSharedResources sharedResources{};
  RenderCaptureRequest captureRequests{};
  RenderCaptureRegistry captureRegistry{};
  TransparentContributionRegistry transparentContributors{};
  TextureHandle sharedDepthTexture{};
  const ResourceManager *resources = nullptr;
  double timeSeconds = 0.0;
  double deltaSeconds = 1.0 / 60.0;
  uint64_t frameIndex = 0;
  SubmissionHandle submission{};
};

[[nodiscard]] inline RenderSettings
resolveRenderSettings(const RenderSettings &source) {
  RenderSettings resolved = source;
  sanitizeVisibilitySettings(resolved.visibility);
  resolved.antiAliasing.temporalProvider =
      sanitizeTemporalReconstructionProvider(
          resolved.antiAliasing.temporalProvider);
  sanitizeAntiAliasingSettings(resolved.antiAliasing);
  sanitizeAmbientOcclusionSettings(resolved.ambientOcclusion, resolved.opaque,
                                   resolved.antiAliasing);
  sanitizeTextureFilteringSettings(resolved.textureFiltering);
  sanitizeToneMapSettings(resolved.toneMap);
  sanitizeHDRPostProcessSettings(resolved.hdrPostProcess);
  sanitizeTransmissionSettings(resolved.transmission);
  sanitizeShadowSettings(resolved.shadow);
  sanitizeDDGISettings(resolved.ddgi);
  return resolved;
}

inline void resolveRenderSettingsForFrame(RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  frame.resolvedSettings = resolveRenderSettings(
      frame.settings ? *frame.settings : kDefaultSettings);
  frame.settingsResolved = true;
}

[[nodiscard]] inline bool
isRenderCaptureRequested(const RenderFrameContext &frame,
                         std::string_view name) noexcept {
  return !frame.captureRequests.empty() && frame.captureRequests.contains(name);
}

[[nodiscard]] inline const RenderSettings &
renderSettingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  if (frame.settingsResolved) {
    return frame.resolvedSettings;
  }
  return frame.settings ? *frame.settings : kDefaultSettings;
}

inline void resetFrameSharedResources(RenderFrameContext &frame) {
  frame.sharedResources = {};
  frame.captureRegistry.clear();
  frame.transparentContributors.clear();
}

[[nodiscard]] inline TextureHandle
resolveFrameDepthTexture(const RenderFrameContext &frame) {
  const TextureHandle depth = frame.sharedResources.sceneDepthTexture;
  return nuri::isValid(depth) ? depth : frame.sharedDepthTexture;
}

[[nodiscard]] inline TextureHandle
resolveSceneColorMipTexture(const RenderFrameContext &frame,
                            uint32_t mipLevel) {
  const std::array textures{
      frame.sharedResources.sceneColorTexture,
      frame.sharedResources.sceneColorHalfResTexture,
      frame.sharedResources.sceneColorQuarterResTexture,
  };
  return mipLevel < textures.size() ? textures[mipLevel] : TextureHandle{};
}

[[nodiscard]] inline LightId
resolveSelectedLightId(const RenderFrameContext &frame) {
  return frame.sharedResources.selectedLightId.value_or(kInvalidLightId);
}

} // namespace nuri
