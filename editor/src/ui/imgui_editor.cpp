#include "nuri/editor_pch.h"

#include "nuri/ui/imgui_editor.h"

#include "nuri/app/editor_animation_player_service.h"
#include "nuri/bakery/bakery_system.h"
#include "nuri/core/application.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/imgui_gpu_renderer.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/render_graph/render_graph_telemetry.h"
#include "nuri/platform/imgui_glfw_platform.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/storage/font/nfont_compiler.h"
#include "nuri/scene/render_scene.h"
#include "nuri/text/text_system.h"
#include "nuri/ui/camera_controller_widget.h"
#include "nuri/ui/file_dialog_widget.h"
#include "nuri/ui/linear_graph.h"
#include "nuri/utils/frame_time_display.h"
#include "scene_light_editor.h"

#include <limits>
#include <unordered_set>

#include <ImGuizmo.h>

namespace nuri {

namespace {

constexpr size_t kMaxLogLines = 2000;
constexpr float kLogFilterWidth = 200.0f;
constexpr float kPassListWidth = 140.0f;
constexpr double kMetricGraphUpdateIntervalSeconds = 0.04;
constexpr double kFrameTimeDisplayUpdateIntervalSeconds = 0.25;
constexpr double kLogUpdateIntervalSeconds = 0.10;
constexpr double kPassMetricsUpdateIntervalSeconds = 0.50;
constexpr float kMetricGraphWindowWidth = 300.0f;
constexpr float kMetricGraphWindowHeight = 280.0f;
constexpr double kMetricSampleMinDeltaSeconds = 1.0e-6;
constexpr std::size_t kMetricGraphSampleCount = 240;
constexpr std::size_t kHierarchyBatchSize = 30;
constexpr uint32_t kUiMaxTessInstances = 65536u;
constexpr const char *kDockspaceWindowName = "NuriDockspace";
constexpr const char *kDockspaceRootId = "NuriDockspace##Root";
constexpr const char *kLogWindowName = "Log";
constexpr const char *kRenderGraphTelemetryWindowName =
    "Render Graph Telemetry";
constexpr const char *kPassMetricsWindowName = "Passes Metrics";
constexpr const char *kPassMetricsRecordingWindowName =
    "Passes Metrics Recording";
constexpr const char *kFontCompilerWindowName = "Font Compiler";
constexpr const char *kBakeryWindowName = "Bakery";
constexpr const char *kLightsWindowName = "Lights";
constexpr const char *kRenderPassesWindowName = "Render Passes";
constexpr const char *kHierarchyWindowName = "Hierarchy";
constexpr const char *kInspectorWindowName = "Inspector";
constexpr const char *kAnimationPlayerWindowName = "Animation Player";
constexpr const char *kTextureFilteringWindowName = "Texture Filtering";
constexpr const char *kAntiAliasingWindowName = "Anti-Aliasing";
constexpr const char *kAmbientOcclusionWindowName = "Ambient Occlusion";
constexpr const char *kHDRPostProcessWindowName = "HDR Postprocess";
constexpr const char *kShadowsWindowName = "Shadows";
constexpr const char *kCameraControllerWindowName = "Camera Controller";
constexpr const char *kCameraHelpWindowName = "Camera Help";
constexpr const char *kGizmoControlsWindowName = "Gizmo Controls";
constexpr const char *kTelemetryWindowName = "Telemetry";
constexpr const char *kFramePacingWindowName = "Frame Pacing";
constexpr std::array<uint8_t, 4> kTextureFilterAnisotropyLevels = {2u, 4u, 8u,
                                                                   16u};
constexpr std::array<const char *, 4> kTextureFilterAnisotropyLabels = {
    "2x", "4x", "8x", "16x"};
constexpr std::array<const char *, 3> kTextureFilterModeLabels = {
    "Bilinear", "Trilinear", "Anisotropic"};
constexpr std::array<const char *, 4> kAntiAliasingModeLabels = {
    "None", "TAA", "Spatial Fallback", "MSAA 4x"};
constexpr std::array<const char *, 5> kTemporalAAQualityPresetLabels = {
    "Performance", "Balanced", "Quality", "Ultra", "Custom"};
constexpr std::array<const char *, 2> kAmbientOcclusionModeLabels = {"Disabled",
                                                                     "GTAO"};

[[nodiscard]] std::string_view
presentModeLabel(SwapchainPresentMode mode) noexcept {
  switch (mode) {
  case SwapchainPresentMode::Immediate:
    return "Immediate";
  case SwapchainPresentMode::Mailbox:
    return "Mailbox";
  case SwapchainPresentMode::Fifo:
    return "FIFO";
  case SwapchainPresentMode::Unknown:
  default:
    return "Unknown";
  }
}
constexpr std::array<const char *, 5> kAmbientOcclusionPresetLabels = {
    "Low", "Balanced", "High", "Ultra", "Custom"};
constexpr std::array<const char *, 4> kAmbientOcclusionDebugViewLabels = {
    "None", "Visibility", "Bent Normal", "Normals"};
constexpr std::array<const char *, 5> kHDRPostProcessDebugViewLabels = {
    "None", "Bloom Prefilter", "Bloom Final", "Log Average Luminance",
    "Adapted Exposure"};
constexpr std::array<const char *, 36> kAntiAliasingDebugViewLabels = {
    "None",
    "Settings",
    "Motion Vectors",
    "Velocity Magnitude",
    "TAA Current",
    "TAA History",
    "TAA Resolved",
    "TAA History Validity",
    "TAA Rejection Mask",
    "TAA Blend Factor",
    "TAA Clamp Delta",
    "TAA Pixel Inspector",
    "TAA Reactive Mask",
    "TAA Disocclusion Mask",
    "TAA Velocity Dilation",
    "TAA Scene Half",
    "TAA Scene Quarter",
    "TAA Transmission Mip",
    "TAA Reprojected History",
    "TAA Resolve Confidence",
    "TAA Clamp Diagnostics",
    "TAA Previous Velocity",
    "TAA HDR Weight",
    "TAA History Filter Delta",
    "TAA Disocclusion Fallback",
    "TAA Split Compare",
    "TAA Temporal Confidence",
    "TAA Previous Depth Rejection",
    "TAA Stability Diagnostics",
    "TAA Stability Ownership",
    "TAA Patch Probe",
    "TAA Motion Filter",
    "Spatial AA Edges",
    "Spatial AA Blend Weights",
    "Spatial AA Cleanup Mask",
    "Spatial AA Split Compare"};
constexpr std::array<const char *, 6> kTemporalAAClampModeLabels = {
    "Clamp RGB",   "Clip RGB",   "Variance RGB",
    "Clamp YCoCg", "Clip YCoCg", "Variance YCoCg"};
constexpr std::array<const char *, 4> kTemporalAAHdrWeightingModeLabels = {
    "None", "Luminance", "Log Luminance", "Tone Mapped"};
constexpr std::array<const char *, 3> kTemporalAAVelocityDilationModeLabels = {
    "None", "Closest Depth", "Largest Magnitude"};
constexpr std::array<const char *, 2> kTemporalAAHistoryFilterModeLabels = {
    "Catmull-Rom", "Bilinear"};
constexpr std::array<uint32_t, 4> kShadowMapResolutions = {1024u, 2048u, 4096u,
                                                           8192u};
constexpr std::array<const char *, 4> kShadowMapResolutionLabels = {"1K", "2K",
                                                                    "4K", "8K"};
const std::array<ImVec4, kMaxShadowCascades> kShadowCascadeColorsUi = {
    ImVec4(0.15f, 1.0f, 0.3f, 1.0f),
    ImVec4(0.0f, 0.85f, 1.0f, 1.0f),
    ImVec4(1.0f, 0.9f, 0.1f, 1.0f),
    ImVec4(1.0f, 0.25f, 1.0f, 1.0f),
};

enum class PassInspectorKind : uint8_t {
  Skybox,
  Shadow,
  Opaque,
  Transmission,
  Transparent,
  Composite,
  AntiAliasing,
  AmbientOcclusion,
  Debug,
  Generic,
};

PassInspectorKind classifyPassInspector(std::string_view featureName,
                                        std::string_view passName) {
  if (featureName == "SkyboxFeature" || passName == "SkyboxPass") {
    return PassInspectorKind::Skybox;
  }
  if (featureName == "ShadowFeature" || passName == "ShadowDepthPass") {
    return PassInspectorKind::Shadow;
  }
  if (featureName == "OpaqueFeature" || featureName == "OpaquePrepassFeature" ||
      featureName == "OpaqueMainFeature" || passName == "OpaqueMainPass" ||
      passName == "OpaqueMainLightingPass" || passName == "OpaquePrepassPass" ||
      passName == "OpaquePickPass") {
    return PassInspectorKind::Opaque;
  }
  if (featureName == "GTAOFeature" || passName == "GTAOPass") {
    return PassInspectorKind::AmbientOcclusion;
  }
  if (featureName == "TemporalAAFeature" ||
      featureName == "MsaaResolveFeature" || passName == "MsaaResolvePass" ||
      passName == "TemporalAAMotionVectorClearPass") {
    return PassInspectorKind::AntiAliasing;
  }
  if (featureName == "TransmissionFeature" ||
      passName == "TransmissionMainPass") {
    return PassInspectorKind::Transmission;
  }
  if (featureName == "TransparentFeature" ||
      passName == "TransparentMainPass" || passName == "TransparentPickPass") {
    return PassInspectorKind::Transparent;
  }
  if (featureName == "FrameCompositionFeature" ||
      featureName == "HDRPostProcessFeature" ||
      featureName == "FramePresentFeature" ||
      passName == "SceneColorDownsamplePass" ||
      passName == "SceneResolvePass" || passName == "HDRExposureAdaptPass" ||
      passName == "HDRBloomCompositePass" || passName == "PresentToneMapPass") {
    return PassInspectorKind::Composite;
  }
  if (featureName == "DebugFeature" || passName == "DebugGridPass" ||
      passName == "DebugSceneOverlayPass") {
    return PassInspectorKind::Debug;
  }
  return PassInspectorKind::Generic;
}

const char *formatDisplayName(Format format) {
  switch (format) {
  case Format::R32_UINT:
    return "R32_UINT";
  case Format::R8_UNORM:
    return "R8_UNORM";
  case Format::R32_FLOAT:
    return "R32_FLOAT";
  case Format::RG32_FLOAT:
    return "RG32_FLOAT";
  case Format::RG16_FLOAT:
    return "RG16_FLOAT";
  case Format::RGBA8_UNORM:
    return "RGBA8_UNORM";
  case Format::RGBA8_SRGB:
    return "RGBA8_SRGB";
  case Format::RGBA8_UINT:
    return "RGBA8_UINT";
  case Format::RGBA16_FLOAT:
    return "RGBA16_FLOAT";
  case Format::RGBA32_FLOAT:
    return "RGBA32_FLOAT";
  case Format::BC7_RGBA_UNORM:
    return "BC7_RGBA_UNORM";
  case Format::BC7_RGBA_SRGB:
    return "BC7_RGBA_SRGB";
  case Format::ETC2_RGB8_UNORM:
    return "ETC2_RGB8_UNORM";
  case Format::ETC2_RGB8_SRGB:
    return "ETC2_RGB8_SRGB";
  case Format::D16_UNORM:
    return "D16_UNORM";
  case Format::D32_FLOAT:
    return "D32_FLOAT";
  case Format::Count:
    return "Invalid";
  }
  return "Unknown";
}

const char *textureFilterModeDisplayName(TextureFilterMode mode) {
  switch (sanitizeTextureFilterMode(mode)) {
  case TextureFilterMode::Bilinear:
    return "Bilinear";
  case TextureFilterMode::Trilinear:
    return "Trilinear";
  case TextureFilterMode::Anisotropic:
    return "Anisotropic";
  }
  return "Unknown";
}

[[nodiscard]] size_t shadowMapResolutionIndex(uint32_t shadowMapSize) {
  size_t bestIndex = 0u;
  uint32_t bestDistance = shadowMapSize > kShadowMapResolutions.front()
                              ? shadowMapSize - kShadowMapResolutions.front()
                              : kShadowMapResolutions.front() - shadowMapSize;
  for (size_t i = 1; i < kShadowMapResolutions.size(); ++i) {
    const uint32_t candidate = kShadowMapResolutions[i];
    const uint32_t distance = shadowMapSize > candidate
                                  ? shadowMapSize - candidate
                                  : candidate - shadowMapSize;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

[[nodiscard]] uint32_t snapShadowMapResolution(uint32_t shadowMapSize) {
  return kShadowMapResolutions[shadowMapResolutionIndex(shadowMapSize)];
}

[[nodiscard]] std::optional<float> shadowDebugCascadeTexelWorldSize(
    const RenderSettings::ShadowSettings &shadow,
    const std::optional<ShadowDebugFrameData> *shadowDebugFrameData) {
  if (shadowDebugFrameData == nullptr || !shadowDebugFrameData->has_value() ||
      (*shadowDebugFrameData)->cascadeCount == 0u) {
    return std::nullopt;
  }

  const uint32_t cascadeIndex =
      std::min(shadow.debug.debugCascadeIndex,
               (*shadowDebugFrameData)->cascadeCount - 1u);
  const float texelWorldSize =
      (*shadowDebugFrameData)->cascades[cascadeIndex].texelWorldSize;
  if (!std::isfinite(texelWorldSize) || texelWorldSize <= 0.0f) {
    return std::nullopt;
  }
  return texelWorldSize;
}

[[nodiscard]] float normalBiasWorldUnits(float normalBiasTexels,
                                         float texelWorldSize) {
  return normalBiasTexels * texelWorldSize;
}

const char *toneMapperDisplayName(ToneMapper mapper) {
  switch (mapper) {
  case ToneMapper::ACES2_SDR:
    return "ACES 2 SDR";
  case ToneMapper::AgX:
    return "AgX";
  }
  return "Unknown";
}

const char *antiAliasingModeDisplayName(AntiAliasingMode mode) {
  switch (sanitizeAntiAliasingMode(mode)) {
  case AntiAliasingMode::None:
    return "None";
  case AntiAliasingMode::TAA:
    return "TAA";
  case AntiAliasingMode::SpatialFallback:
    return "Spatial Fallback";
  case AntiAliasingMode::MSAA4x:
    return "MSAA 4x";
  }
  return "Unknown";
}

const char *temporalAAQualityPresetDisplayName(TemporalAAQualityPreset preset) {
  switch (sanitizeTemporalAAQualityPreset(preset)) {
  case TemporalAAQualityPreset::Performance:
    return kTemporalAAQualityPresetLabels[0];
  case TemporalAAQualityPreset::Balanced:
    return kTemporalAAQualityPresetLabels[1];
  case TemporalAAQualityPreset::Quality:
    return kTemporalAAQualityPresetLabels[2];
  case TemporalAAQualityPreset::Ultra:
    return kTemporalAAQualityPresetLabels[3];
  case TemporalAAQualityPreset::Custom:
    return kTemporalAAQualityPresetLabels[4];
  }
  return "Unknown";
}

const char *antiAliasingDebugViewDisplayName(AntiAliasingDebugView view) {
  switch (sanitizeAntiAliasingDebugView(view)) {
  case AntiAliasingDebugView::None:
    return "None";
  case AntiAliasingDebugView::Settings:
    return "Settings";
  case AntiAliasingDebugView::MotionVectors:
    return "Motion Vectors";
  case AntiAliasingDebugView::VelocityMagnitude:
    return "Velocity Magnitude";
  case AntiAliasingDebugView::TAACurrentColor:
    return "TAA Current";
  case AntiAliasingDebugView::TAAPreviousHistory:
    return "TAA History";
  case AntiAliasingDebugView::TAAResolved:
    return "TAA Resolved";
  case AntiAliasingDebugView::TAAHistoryValidity:
    return "TAA History Validity";
  case AntiAliasingDebugView::TAARejectionMask:
    return "TAA Rejection Mask";
  case AntiAliasingDebugView::TAABlendFactor:
    return "TAA Blend Factor";
  case AntiAliasingDebugView::TAAClampDelta:
    return "TAA Clamp Delta";
  case AntiAliasingDebugView::TAAPixelInspector:
    return "TAA Pixel Inspector";
  case AntiAliasingDebugView::TAAReactiveMask:
    return "TAA Reactive Mask";
  case AntiAliasingDebugView::TAADisocclusionMask:
    return "TAA Disocclusion Mask";
  case AntiAliasingDebugView::TAAVelocityDilation:
    return "TAA Velocity Dilation";
  case AntiAliasingDebugView::TAASceneColorHalfRes:
    return "TAA Scene Half";
  case AntiAliasingDebugView::TAASceneColorQuarterRes:
    return "TAA Scene Quarter";
  case AntiAliasingDebugView::TAATransmissionMipSource:
    return "TAA Transmission Mip";
  case AntiAliasingDebugView::TAAReprojectedHistory:
    return "TAA Reprojected History";
  case AntiAliasingDebugView::TAAResolveConfidence:
    return "TAA Resolve Confidence";
  case AntiAliasingDebugView::TAAClampDiagnostics:
    return "TAA Clamp Diagnostics";
  case AntiAliasingDebugView::TAAPreviousVelocity:
    return "TAA Previous Velocity";
  case AntiAliasingDebugView::TAAHdrWeight:
    return "TAA HDR Weight";
  case AntiAliasingDebugView::TAAHistoryFilterDelta:
    return "TAA History Filter Delta";
  case AntiAliasingDebugView::TAADisocclusionFallback:
    return "TAA Disocclusion Fallback";
  case AntiAliasingDebugView::TAASplitCompare:
    return "TAA Split Compare";
  case AntiAliasingDebugView::SpatialAAEdges:
    return "Spatial AA Edges";
  case AntiAliasingDebugView::SpatialAABlendWeights:
    return "Spatial AA Blend Weights";
  case AntiAliasingDebugView::SpatialAACleanupMask:
    return "Spatial AA Cleanup Mask";
  case AntiAliasingDebugView::SpatialAASplitCompare:
    return "Spatial AA Split Compare";
  case AntiAliasingDebugView::TAATemporalConfidence:
    return "TAA Temporal Confidence";
  case AntiAliasingDebugView::TAAPreviousDepthRejection:
    return "TAA Previous Depth Rejection";
  case AntiAliasingDebugView::TAAStabilityDiagnostics:
    return "TAA Stability Diagnostics";
  case AntiAliasingDebugView::TAAStabilityOwnership:
    return "TAA Stability Ownership";
  case AntiAliasingDebugView::TAAPatchProbe:
    return "TAA Patch Probe";
  case AntiAliasingDebugView::TAAMotionFilter:
    return "TAA Motion Filter";
  }
  return "Unknown";
}

const char *ambientOcclusionModeDisplayName(AmbientOcclusionMode mode) {
  switch (sanitizeAmbientOcclusionMode(mode)) {
  case AmbientOcclusionMode::Disabled:
    return kAmbientOcclusionModeLabels[0];
  case AmbientOcclusionMode::GTAO:
    return kAmbientOcclusionModeLabels[1];
  }
  return "Unknown";
}

const char *ambientOcclusionPresetDisplayName(AmbientOcclusionPreset preset) {
  switch (sanitizeAmbientOcclusionPreset(preset)) {
  case AmbientOcclusionPreset::Low:
    return kAmbientOcclusionPresetLabels[0];
  case AmbientOcclusionPreset::Balanced:
    return kAmbientOcclusionPresetLabels[1];
  case AmbientOcclusionPreset::High:
    return kAmbientOcclusionPresetLabels[2];
  case AmbientOcclusionPreset::Ultra:
    return kAmbientOcclusionPresetLabels[3];
  case AmbientOcclusionPreset::Custom:
    return kAmbientOcclusionPresetLabels[4];
  }
  return "Unknown";
}

const char *
ambientOcclusionDebugViewDisplayName(AmbientOcclusionDebugView view) {
  switch (sanitizeAmbientOcclusionDebugView(view)) {
  case AmbientOcclusionDebugView::None:
    return kAmbientOcclusionDebugViewLabels[0];
  case AmbientOcclusionDebugView::Visibility:
    return kAmbientOcclusionDebugViewLabels[1];
  case AmbientOcclusionDebugView::BentNormal:
    return kAmbientOcclusionDebugViewLabels[2];
  case AmbientOcclusionDebugView::Normals:
    return kAmbientOcclusionDebugViewLabels[3];
  }
  return "Unknown";
}

const char *ambientOcclusionDisabledReasonDisplayName(
    AmbientOcclusionDisabledReason reason) {
  switch (reason) {
  case AmbientOcclusionDisabledReason::None:
    return "None";
  case AmbientOcclusionDisabledReason::ModeDisabled:
    return "Mode Disabled";
  case AmbientOcclusionDisabledReason::OpaqueDisabled:
    return "Opaque Disabled";
  case AmbientOcclusionDisabledReason::Msaa4x:
    return "MSAA 4x";
  case AmbientOcclusionDisabledReason::MissingResources:
    return "Missing Resources";
  case AmbientOcclusionDisabledReason::Unsupported:
    return "Unsupported";
  }
  return "Unknown";
}

const char *temporalAAClampModeDisplayName(TemporalAAClampMode mode) {
  switch (sanitizeTemporalAAClampMode(mode)) {
  case TemporalAAClampMode::Clamp:
    return "Clamp RGB";
  case TemporalAAClampMode::Clip:
    return "Clip RGB";
  case TemporalAAClampMode::Variance:
    return "Variance RGB";
  case TemporalAAClampMode::ClampYCoCg:
    return "Clamp YCoCg";
  case TemporalAAClampMode::ClipYCoCg:
    return "Clip YCoCg";
  case TemporalAAClampMode::VarianceYCoCg:
    return "Variance YCoCg";
  }
  return "Unknown";
}

const char *
temporalAAHdrWeightingModeDisplayName(TemporalAAHdrWeightingMode mode) {
  switch (sanitizeTemporalAAHdrWeightingMode(mode)) {
  case TemporalAAHdrWeightingMode::None:
    return "None";
  case TemporalAAHdrWeightingMode::Luminance:
    return "Luminance";
  case TemporalAAHdrWeightingMode::LogLuminance:
    return "Log Luminance";
  case TemporalAAHdrWeightingMode::ToneMapped:
    return "Tone Mapped";
  }
  return "Unknown";
}

const char *
temporalAAVelocityDilationModeDisplayName(TemporalAAVelocityDilationMode mode) {
  switch (sanitizeTemporalAAVelocityDilationMode(mode)) {
  case TemporalAAVelocityDilationMode::None:
    return "None";
  case TemporalAAVelocityDilationMode::ClosestDepth:
    return "Closest Depth";
  case TemporalAAVelocityDilationMode::LargestMagnitude:
    return "Largest Magnitude";
  }
  return "Unknown";
}

const char *
temporalAAHistoryFilterModeDisplayName(TemporalAAHistoryFilterMode mode) {
  switch (sanitizeTemporalAAHistoryFilterMode(mode)) {
  case TemporalAAHistoryFilterMode::Bilinear:
    return "Bilinear";
  case TemporalAAHistoryFilterMode::CatmullRom:
    return "Catmull-Rom";
  }
  return "Unknown";
}

const char *boolLogValue(bool value) { return value ? "true" : "false"; }

const char *shadowQualityPresetDisplayName(ShadowQualityPreset preset) {
  switch (sanitizeShadowQualityPreset(preset)) {
  case ShadowQualityPreset::Custom:
    return "Custom";
  case ShadowQualityPreset::Low:
    return "Low";
  case ShadowQualityPreset::Medium:
    return "Medium";
  case ShadowQualityPreset::High:
    return "High";
  case ShadowQualityPreset::Ultra:
    return "Ultra";
  }
  return "Unknown";
}

const char *shadowSdsmStatusDisplayName(ShadowSdsmStatus status) {
  switch (status) {
  case ShadowSdsmStatus::Disabled:
    return "Disabled";
  case ShadowSdsmStatus::Active:
    return "Active";
  case ShadowSdsmStatus::Unavailable:
    return "Unavailable";
  case ShadowSdsmStatus::Stale:
    return "Stale";
  case ShadowSdsmStatus::Invalid:
    return "Invalid";
  case ShadowSdsmStatus::FallbackFixed:
    return "Fallback Fixed";
  }
  return "Unknown";
}

const char *shadowStaticOnlyReuseStatusDisplayName(
    ShadowCascadeDebugFrameData::StaticOnlyReuseStatus status) {
  switch (status) {
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::None:
    return "None";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::Reused:
    return "Reused";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::StaticCacheRebuilt:
    return "Static cache rebuilt";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::HasDynamicCasters:
    return "Dynamic casters present";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::
      NoPreviousStaticOnlyPass:
    return "No previous static-only pass";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::RasterStateChanged:
    return "Raster state changed";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::AdaptiveRefresh:
    return "Adaptive refresh";
  }
  return "Unknown";
}

const char *
shadowSdsmReductionBackendDisplayName(ShadowSdsmReductionBackend backend) {
  switch (backend) {
  case ShadowSdsmReductionBackend::Cpu:
    return "CPU";
  case ShadowSdsmReductionBackend::Gpu:
    return "GPU";
  }
  return "Unknown";
}

[[nodiscard]] bool isSdsmWarningStatus(ShadowSdsmStatus status) {
  return status == ShadowSdsmStatus::Unavailable ||
         status == ShadowSdsmStatus::Stale ||
         status == ShadowSdsmStatus::Invalid;
}

void drawShadowSplitGraph(const ShadowSdsmDebugFrameData &sdsm) {
  const float graphWidth = ImGui::GetContentRegionAvail().x;
  const ImVec2 graphSize(std::max(graphWidth, 160.0f), 72.0f);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const ImVec2 max(origin.x + graphSize.x, origin.y + graphSize.y);
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  const ImU32 borderColor = IM_COL32(110, 110, 110, 255);
  const ImU32 barColor = IM_COL32(36, 36, 36, 255);
  const ImU32 fixedColor = IM_COL32(150, 150, 150, 255);
  const ImU32 minMaxColor = IM_COL32(90, 170, 255, 255);
  drawList->AddRectFilled(origin, max, barColor, 4.0f);
  drawList->AddRect(origin, max, borderColor, 4.0f);

  const float rangeMin = sdsm.fixedRangeNear;
  const float rangeMax = std::max(sdsm.fixedRangeFar, rangeMin + 1.0e-4f);
  const auto normalizedX = [&](float depth) {
    const float t =
        std::clamp((depth - rangeMin) / (rangeMax - rangeMin), 0.0f, 1.0f);
    return origin.x + t * graphSize.x;
  };
  const uint32_t splitCount =
      std::clamp(sdsm.splitCount, 1u, kMaxShadowCascades);
  for (uint32_t i = 1u; i < splitCount; ++i) {
    const float fixedX = normalizedX(sdsm.fixedSplitDepths[i]);
    drawList->AddLine(ImVec2(fixedX, origin.y + 10.0f),
                      ImVec2(fixedX, max.y - 10.0f), fixedColor, 1.0f);
  }
  for (uint32_t i = 1u; i < splitCount; ++i) {
    const float minMaxX = normalizedX(sdsm.minMaxSplitDepths[i]);
    drawList->AddLine(ImVec2(minMaxX, origin.y + 16.0f),
                      ImVec2(minMaxX, max.y - 16.0f), minMaxColor, 1.0f);
  }
  for (uint32_t i = 1u; i < splitCount; ++i) {
    const float effectiveX = normalizedX(sdsm.effectiveSplitDepths[i]);
    drawList->AddLine(ImVec2(effectiveX, origin.y + 4.0f),
                      ImVec2(effectiveX, max.y - 4.0f),
                      ImGui::ColorConvertFloat4ToU32(
                          kShadowCascadeColorsUi[static_cast<size_t>(i - 1u)]),
                      2.0f);
  }
  ImGui::InvisibleButton("ShadowSplitGraph", graphSize);
  ImGui::TextUnformatted(
      "Split graph: gray=fixed, blue=min/max, cascade colors=effective.");
}

const char *shadowPreviewModeDisplayName(ShadowPreviewMode mode) {
  switch (sanitizeShadowPreviewMode(mode)) {
  case ShadowPreviewMode::SelectedCascade:
    return "Selected Cascade";
  case ShadowPreviewMode::TiledAllCascades:
    return "Tiled All Cascades";
  }
  return "Unknown";
}

const char *bakeJobKindName(bakery::BakeJobKind kind) {
  switch (kind) {
  case bakery::BakeJobKind::BrdfLut:
    return "BRDF LUT";
  case bakery::BakeJobKind::EnvmapPrefilter:
    return "Envmap Prefilter";
  case bakery::BakeJobKind::SceneTextureArtifacts:
    return "Scene Texture Artifacts";
  }
  return "Unknown";
}

const char *bakeJobStateName(bakery::BakeJobState state) {
  switch (state) {
  case bakery::BakeJobState::Queued:
    return "Queued";
  case bakery::BakeJobState::CacheCheck:
    return "CacheCheck";
  case bakery::BakeJobState::GpuSetup:
    return "GpuSetup";
  case bakery::BakeJobState::GpuStep:
    return "GpuStep";
  case bakery::BakeJobState::WriteQueued:
    return "WriteQueued";
  case bakery::BakeJobState::WriteInFlight:
    return "WriteInFlight";
  case bakery::BakeJobState::Succeeded:
    return "Succeeded";
  case bakery::BakeJobState::Skipped:
    return "Skipped";
  case bakery::BakeJobState::Failed:
    return "Failed";
  case bakery::BakeJobState::Canceled:
    return "Canceled";
  }
  return "Unknown";
}

struct LogLevelMeta {
  LogLevel level;
  std::string_view tag;
};

constexpr LogLevelMeta kLogLevels[] = {
    {LogLevel::Trace, "[Trace]"}, {LogLevel::Debug, "[Debug]"},
    {LogLevel::Info, "[Info]"},   {LogLevel::Warning, "[Warn]"},
    {LogLevel::Error, "[Error]"}, {LogLevel::Fatal, "[Fatal]"},
};

float sanitizeSample(float value) {
  return std::isfinite(value) ? value : 0.0f;
}

struct LogLine {
  LogLevel level = LogLevel::Info;
  std::string message;
};

struct LogFilterState {
  bool autoScroll = true;
  bool requestScroll = false;
  ImGuiTextFilter textFilter;
  bool showTrace = true;
  bool showDebug = true;
  bool showInfo = true;
  bool showWarning = true;
  bool showFatal = true;

  bool levelEnabled(LogLevel level) const {
    switch (level) {
    case LogLevel::Trace:
      return showTrace;
    case LogLevel::Debug:
      return showDebug;
    case LogLevel::Info:
      return showInfo;
    case LogLevel::Warning:
      return showWarning;
    case LogLevel::Error:
      return true;
    case LogLevel::Fatal:
      return showFatal;
    }
    return true;
  }
};

struct FontCompilerUiState {
  std::filesystem::path outputDirectory;
  std::vector<std::filesystem::path> availableNfonts;
  std::array<char, 512> sourcePath = {};
  std::array<char, 512> outputPath = {};
  std::array<char, 512> selectedNfontPath = {};
  bool autoOutputName = true;
  int charsetPreset = 0;
  float minimumEmSize = 40.0f;
  float pxRange = 4.0f;
  float outerPixelPadding = 2.0f;
  int atlasSpacing = 1;
  bool useRgba16fAtlas = true;
  int atlasWidthPreset = 1;
  int atlasHeightPreset = 1;
  int maxAtlasWidth = 2048;
  int maxAtlasHeight = 2048;
  int threadCount = 0;
  int selectedNfontIndex = -1;
  float globalFontSizePx = 42.0f;
  std::shared_future<Result<NFontCompileReport, std::string>> compileFuture;
  bool compileInFlight = false;
  bool nfontListInitialized = false;
  std::string status;
  std::string error;
  std::string globalStatus;
  std::string globalError;
  NFontCompileReport lastReport{};
  FileDialogWidget fileDialog{};

  FontCompilerUiState() {
    auto runtimeConfigResult = loadRuntimeConfigFromEnvOrDefault();
    if (runtimeConfigResult.hasError()) {
      outputDirectory = std::filesystem::path("assets") / "fonts";
    } else {
      outputDirectory = runtimeConfigResult.value().roots.fonts;
    }
    outputDirectory = outputDirectory.lexically_normal();

    const std::string defaultOutput = (outputDirectory / "generated_ui.nfont")
                                          .lexically_normal()
                                          .generic_string();
    std::memcpy(outputPath.data(), defaultOutput.c_str(),
                std::min(outputPath.size() - 1, defaultOutput.size()));
  }
};

struct BakeryUiState {
  std::array<char, 512> envHdrPath = {};
  std::array<char, 512> scenePath = {};
  bool forceRebuild = false;
  bool prebuildBc7 = false;
  bool prebuildEtc2 = false;
  bool prebuildRgba8 = false;
  std::string status{};
  std::string error{};
  FileDialogWidget fileDialog{};

  BakeryUiState() {
    constexpr std::string_view kDefaultEnvHdr = "piazza_bologni_1k.hdr";
    const size_t copyCount =
        std::min(envHdrPath.size() - 1u, kDefaultEnvHdr.size());
    if (copyCount > 0) {
      std::memcpy(envHdrPath.data(), kDefaultEnvHdr.data(), copyCount);
    }
    envHdrPath[copyCount] = '\0';
  }
};

struct RenderGraphTelemetryUiState {
  std::array<char, 512> outputPath = {};
  std::string status{};
  std::string error{};
  std::string lastSuggestedPath{};
  FileDialogWidget fileDialog{};
  bool initializedOutputPath = false;
};

struct PassMetricsRow {
  std::string name{};
  float cpuTimeMs = 0.0f;
  float gpuTimeMs = 0.0f;
  bool hasCpuTiming = false;
  bool hasGpuTiming = false;
};

struct PassMetricAggregate {
  uint32_t orderedPassIndex = UINT32_MAX;
  std::string name{};
  float cpuMinMs = std::numeric_limits<float>::max();
  float cpuMaxMs = 0.0f;
  double cpuSumMs = 0.0;
  uint32_t cpuSampleCount = 0u;
  float gpuMinMs = std::numeric_limits<float>::max();
  float gpuMaxMs = 0.0f;
  double gpuSumMs = 0.0;
  uint32_t gpuSampleCount = 0u;
};

struct PassMetricsUiState {
  std::vector<PassMetricsRow> rows{};
  std::vector<PassMetricAggregate> aggregates{};
  double lastUpdateSeconds = -kPassMetricsUpdateIntervalSeconds;
  uint64_t renderGraphFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t gpuFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t recordingStartFrameIndex = 0u;
  uint64_t lastRecordedCpuFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t lastRecordedGpuFrameIndex = std::numeric_limits<uint64_t>::max();
  uint32_t gpuPassTimingCount = 0u;
  uint32_t recordedCpuFrameCount = 0u;
  uint32_t recordedGpuReportCount = 0u;
  bool recording = false;
  bool showRecordingWindow = false;
};

struct TelemetryOverlayUiState {
  bool overlayEnabled = true;
  bool showFpsMs = true;
  bool showInstanceStats = true;
  bool showDrawTessStats = true;
  bool showIndirectStats = true;
  bool showDebugDrawStats = true;
  bool showPatchHeatmap = true;
  bool showDispatchStats = true;
  bool showGraphs = true;
  bool showImGuiMetricsWindow = false;
};

struct SceneSelectionUiState {
  std::vector<std::string> ids{};
  std::vector<std::string> names{};
  std::vector<const char *> nameViews{};
  std::string hotkeyHint = "Toggle Editor: F6";
  int selectedIndex = 0;
  std::optional<std::string> pendingSelectionRequest{};
  std::string pendingSceneId{};
  std::string loadPhase{};
  std::string loadError{};
  float loadProgress = 0.0f;
  bool loadCancellable = false;
  bool loadFailed = false;
  bool pendingCancelRequest = false;
  uint64_t version = 0;

  void set(std::span<const EditorSceneSelectionOption> scenes,
           std::string_view selectedSceneId, uint64_t newVersion,
           std::string_view hotkeyHintIn, const EditorSceneLoadUiState &load) {
    if (version != newVersion) {
      version = newVersion;
      ids.clear();
      names.clear();
      ids.reserve(scenes.size());
      names.reserve(scenes.size());
      for (const EditorSceneSelectionOption &scene : scenes) {
        ids.emplace_back(scene.id);
        names.emplace_back(scene.label);
      }
      nameViews.clear();
      nameViews.reserve(names.size());
      for (const std::string &name : names) {
        nameViews.push_back(name.c_str());
      }
    }
    if (hotkeyHintIn.empty()) {
      hotkeyHint = "Toggle Editor: F6";
    } else {
      hotkeyHint.assign(hotkeyHintIn.data(), hotkeyHintIn.size());
    }
    if (nameViews.empty()) {
      selectedIndex = 0;
      pendingSelectionRequest.reset();
      return;
    }
    auto it = std::find(ids.begin(), ids.end(), selectedSceneId);
    selectedIndex =
        it == ids.end() ? 0 : static_cast<int>(std::distance(ids.begin(), it));
    pendingSceneId.assign(load.pendingSceneId.data(),
                          load.pendingSceneId.size());
    loadPhase.assign(load.phase.data(), load.phase.size());
    loadError.assign(load.error.data(), load.error.size());
    loadProgress = std::clamp(load.progress, 0.0f, 1.0f);
    loadCancellable = load.cancellable;
    loadFailed = load.failed;
  }
};

struct RenderableInspectorState {
  RenderableId renderableId = kInvalidRenderableId;
  MaterialRef ownedOverride = kInvalidMaterialRef;
  int baselineSlotIndex = 0;
  int selectedTextureIndex = 0;
};

struct HierarchyNodeStats {
  uint32_t renderableCount = 0u;
  uint32_t lightCount = 0u;
};

struct HierarchyNodeTopology {
  std::string labelName{};
  std::vector<NodeId> children{};
};

enum class HierarchyRowKind : uint8_t {
  SceneRoot,
  Node,
  Batch,
};

struct HierarchyVisibleRow {
  HierarchyRowKind kind = HierarchyRowKind::Node;
  int depth = 0;
  NodeId node = kInvalidNodeId;
  size_t beginIndex = 0u;
  size_t endIndex = 0u;
};

[[nodiscard]] constexpr size_t hierarchyNodeSlot(NodeId node) {
  return static_cast<size_t>(indexOf(node));
}

struct MaterialSourceEntry {
  MaterialRef ref = kInvalidMaterialRef;
  std::string label{};
};

struct MaterialTextureEntry {
  const char *label = "";
  TextureRef ref = kInvalidTextureRef;
};

template <typename T = ImTextureID>
inline T toImTextureId(uint32_t bindlessIndex) {
  if constexpr (std::is_pointer_v<T>) {
    return static_cast<T>(
        reinterpret_cast<void *>(static_cast<uintptr_t>(bindlessIndex)));
  } else {
    return static_cast<T>(static_cast<uintptr_t>(bindlessIndex));
  }
}

[[nodiscard]] std::string nodeDisplayName(const SceneGraph &graph,
                                          NodeId node) {
  std::string_view name{};
  if (graph.getNodeName(node, name) && !name.empty()) {
    return std::string(name);
  }
  return "Node #" + std::to_string(indexOf(node));
}

[[nodiscard]] bool selectionNodeStillValid(const SceneGraph &graph,
                                           const SceneEditorSelectionState &s) {
  if (!isValid(s.node)) {
    return false;
  }
  glm::mat4 dummy(1.0f);
  return graph.getNodeLocalTransform(s.node, dummy);
}

[[nodiscard]] std::optional<uint32_t>
findRenderableIndexById(const RenderScene &scene, RenderableId id) {
  return scene.findRenderableIndex(id);
}

void applyNodeSelection(const RenderScene &scene, NodeId node,
                        SceneEditorSelectionState &selection) {
  const RenderableId previousRenderable = selection.renderableId;
  selection.clear();
  if (!isValid(node)) {
    return;
  }

  selection.node = node;
  RenderableId firstRenderable = kInvalidRenderableId;
  RenderableId matchedRenderable = kInvalidRenderableId;
  scene.graph().forEachRenderableOnNode(node, [&](RenderableId renderableId) {
    if (!isValid(firstRenderable)) {
      firstRenderable = renderableId;
    }
    if (renderableId == previousRenderable) {
      matchedRenderable = renderableId;
    }
  });
  if (!isValid(matchedRenderable)) {
    matchedRenderable = firstRenderable;
  }
  if (isValid(matchedRenderable)) {
    const auto renderableIndex =
        findRenderableIndexById(scene, matchedRenderable);
    if (renderableIndex.has_value()) {
      selection.kind = SceneSelectionKind::NodeRenderable;
      selection.renderableId = matchedRenderable;
      selection.renderableIndex = *renderableIndex;
      return;
    }
  }

  LightId firstLight = kInvalidLightId;
  scene.graph().forEachLightOnNode(node, [&](LightId lightId) {
    if (!isValid(firstLight)) {
      firstLight = lightId;
    }
  });
  if (isValid(firstLight)) {
    selection.kind = SceneSelectionKind::Light;
    selection.lightId = firstLight;
    return;
  }
  selection.kind = SceneSelectionKind::Node;
}

[[nodiscard]] std::vector<NodeId> collectChildNodes(const SceneGraph &graph,
                                                    NodeId node) {
  std::vector<NodeId> out;
  NodeId child = kInvalidNodeId;
  if (!graph.getNodeFirstChild(node, child)) {
    return out;
  }
  while (isValid(child)) {
    out.push_back(child);
    NodeId next = kInvalidNodeId;
    if (!graph.getNodeNextSibling(child, next)) {
      break;
    }
    child = next;
  }
  std::reverse(out.begin(), out.end());
  return out;
}

[[nodiscard]] std::vector<MaterialSourceEntry>
buildMaterialSourceEntries(const Renderable &renderable,
                           const ResourceManager *resources) {
  std::vector<MaterialSourceEntry> entries;
  if (resources == nullptr) {
    entries.push_back(
        MaterialSourceEntry{.ref = renderable.material, .label = "Fallback"});
    return entries;
  }

  const ModelRecord *modelRecord = resources->tryGet(renderable.model);
  if (modelRecord == nullptr || modelRecord->model == nullptr ||
      modelRecord->model->sourceMaterialCount() == 0u) {
    entries.push_back(
        MaterialSourceEntry{.ref = renderable.material, .label = "Fallback"});
    return entries;
  }

  entries.reserve(modelRecord->model->sourceMaterialCount());
  for (uint32_t sourceIndex = 0;
       sourceIndex < modelRecord->model->sourceMaterialCount(); ++sourceIndex) {
    const MaterialRef mapped = modelRecord->materialForSource(sourceIndex);
    const MaterialRef resolved = isValid(mapped) ? mapped : renderable.material;
    MaterialSourceEntry entry{};
    entry.ref = resolved;
    entry.label = "Source Slot " + std::to_string(sourceIndex);
    if (const MaterialRecord *record = resources->tryGet(resolved);
        record != nullptr && !record->debugName.empty()) {
      entry.label += " - " + std::string(record->debugName);
    }
    entries.push_back(std::move(entry));
  }

  return entries;
}

[[nodiscard]] std::vector<MaterialTextureEntry>
buildMaterialTextureEntries(const MaterialRecord &record) {
  struct Spec {
    const char *label;
    MaterialTextureSlot slot;
  };
  constexpr Spec kSpecs[] = {
      {"Base Color", kMaterialTextureSlotBaseColor},
      {"Metallic Roughness", kMaterialTextureSlotMetallicRoughness},
      {"Normal", kMaterialTextureSlotNormal},
      {"Emissive", kMaterialTextureSlotEmissive},
      {"Occlusion", kMaterialTextureSlotOcclusion},
  };

  std::vector<MaterialTextureEntry> entries;
  for (const Spec &spec : kSpecs) {
    const TextureRef ref = record.textureRefs[spec.slot];
    if (!isValid(ref)) {
      continue;
    }
    entries.push_back(MaterialTextureEntry{.label = spec.label, .ref = ref});
  }
  return entries;
}

constexpr std::array<int, 5> kAtlasResolutionSteps = {1024, 2048, 3072, 4096,
                                                      8192};

constexpr std::array<const char *, 5> kAtlasResolutionStepLabels = {
    "1K (1024)", "2K (2048)", "3K (3072)", "4K (4096)", "8K (8192)"};

void setPathText(std::array<char, 512> &buffer, std::string_view value) {
  buffer.fill('\0');
  const size_t copyCount = std::min(buffer.size() - 1u, value.size());
  if (copyCount > 0) {
    std::memcpy(buffer.data(), value.data(), copyCount);
  }
  buffer[copyCount] = '\0';
}

void syncOutputPathFromSource(FontCompilerUiState &state) {
  const std::filesystem::path sourcePath{std::string(state.sourcePath.data())};
  const std::string stem = sourcePath.stem().string();
  if (stem.empty()) {
    return;
  }

  if (state.outputDirectory.empty()) {
    state.outputDirectory = std::filesystem::path("assets") / "fonts";
  }
  const std::filesystem::path resolved =
      (state.outputDirectory / (stem + ".nfont")).lexically_normal();
  setPathText(state.outputPath, resolved.generic_string());
}

void refreshNfontAssetList(FontCompilerUiState &state) {
  state.availableNfonts.clear();
  std::error_code ec;
  if (!std::filesystem::exists(state.outputDirectory, ec) ||
      !std::filesystem::is_directory(state.outputDirectory, ec)) {
    state.selectedNfontIndex = -1;
    return;
  }

  for (const auto &entry :
       std::filesystem::directory_iterator(state.outputDirectory, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file(ec)) {
      continue;
    }
    const std::filesystem::path path = entry.path();
    if (path.extension() == ".nfont") {
      state.availableNfonts.push_back(path.lexically_normal());
    }
  }
  std::sort(state.availableNfonts.begin(), state.availableNfonts.end());

  const std::filesystem::path currentPath{
      std::string(state.selectedNfontPath.data())};
  state.selectedNfontIndex = -1;
  for (size_t i = 0; i < state.availableNfonts.size(); ++i) {
    if (state.availableNfonts[i] == currentPath) {
      state.selectedNfontIndex = static_cast<int>(i);
      break;
    }
  }
  if (state.selectedNfontIndex < 0 && !state.availableNfonts.empty()) {
    state.selectedNfontIndex = 0;
    setPathText(state.selectedNfontPath,
                state.availableNfonts.front().generic_string());
  }
}

[[nodiscard]] int closestAtlasStepIndex(int value) {
  int bestIndex = 0;
  int bestDistance = std::abs(kAtlasResolutionSteps[0] - value);
  for (size_t i = 1; i < kAtlasResolutionSteps.size(); ++i) {
    const int distance = std::abs(kAtlasResolutionSteps[i] - value);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = static_cast<int>(i);
    }
  }
  return bestIndex;
}

struct LogModel {
  std::deque<LogLine> lines;
  std::uint64_t lastSequence = 0;
  bool seededFromFile = false;

  void clear() {
    lines.clear();
    lastSequence = 0;
    seededFromFile = false;
  }

  void trimLinesToCapacity() {
    while (lines.size() > kMaxLogLines) {
      lines.pop_front();
    }
  }

  static std::filesystem::path findLatestLogFile() {
    std::error_code ec;
    const std::filesystem::path logDir("logs");
    if (!std::filesystem::exists(logDir, ec)) {
      return {};
    }

    std::filesystem::path latest;
    std::filesystem::file_time_type latestTime{};
    for (const auto &entry : std::filesystem::directory_iterator(logDir, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_regular_file(ec)) {
        continue;
      }
      const auto path = entry.path();
      if (path.extension() != ".log") {
        continue;
      }
      const auto writeTime = entry.last_write_time(ec);
      if (ec) {
        continue;
      }
      if (latest.empty() || writeTime > latestTime) {
        latest = path;
        latestTime = writeTime;
      }
    }
    return latest;
  }

  static std::pair<LogLevel, std::string> parseLevelTag(std::string_view line) {
    for (const auto &meta : kLogLevels) {
      if (line.size() >= meta.tag.size() &&
          line.substr(0, meta.tag.size()) == meta.tag) {
        std::string msg(line.substr(meta.tag.size()));
        if (!msg.empty() && msg.front() == ' ') {
          msg.erase(0, 1);
        }
        return {meta.level, std::move(msg)};
      }
    }
    return {LogLevel::Info, std::string(line)};
  }

  void seedFromFileIfNeeded(LogFilterState &filterState) {
    if (seededFromFile) {
      return;
    }
    seededFromFile = true;

    const std::filesystem::path logPath = findLatestLogFile();
    if (logPath.empty()) {
      return;
    }

    std::ifstream file(logPath);
    if (!file.is_open()) {
      return;
    }

    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      auto [level, message] = parseLevelTag(line);
      LogLine entry{};
      entry.level = level;
      entry.message = std::move(message);
      lines.push_back(std::move(entry));
    }

    trimLinesToCapacity();
    if (!lines.empty()) {
      filterState.requestScroll = true;
    }
  }

  void appendEntries(const std::vector<LogEntry> &entries) {
    for (const auto &entry : entries) {
      LogLine line{};
      line.level = entry.level;
      line.message = entry.message;
      lines.push_back(std::move(line));
    }
    trimLinesToCapacity();
  }

  void update(LogFilterState &filterState) {
    seedFromFileIfNeeded(filterState);

    std::vector<LogEntry> entries;
    const LogReadResult result = readLogEntriesSince(lastSequence, entries);
    if (result.truncated) {
      lines.clear();
      lastSequence = result.lastSequence;
    }
    if (!entries.empty()) {
      lastSequence = result.lastSequence;
      appendEntries(entries);
      filterState.requestScroll = true;
    }
  }
};

std::string_view logTagFor(LogLevel level) {
  for (const auto &meta : kLogLevels) {
    if (meta.level == level) {
      return meta.tag;
    }
  }
  return "[Info]";
}

int logLevelComboIndex(LogLevel level) {
  for (int i = 0; i < static_cast<int>(std::size(kLogLevels)); ++i) {
    if (kLogLevels[static_cast<size_t>(i)].level == level) {
      return i;
    }
  }
  return 0;
}

void setLogLevelFromCombo(LogLevel &level, int comboIndex) {
  const int safeIndex =
      std::clamp(comboIndex, 0, static_cast<int>(std::size(kLogLevels)) - 1);
  level = kLogLevels[static_cast<size_t>(safeIndex)].level;
}

void drawInlineCheckbox(const char *label, bool &value) {
  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::Checkbox(label, &value);
}

void drawLogToolbar(LogModel &model, LogFilterState &filterState) {
  if (ImGui::Button("Clear")) {
    model.clear();
    filterState.requestScroll = true;
  }

  drawInlineCheckbox("Auto-scroll", filterState.autoScroll);

  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  filterState.textFilter.Draw("Filter", kLogFilterWidth);

  struct Toggle {
    const char *label;
    bool *enabled;
  };
  Toggle toggles[] = {
      {"Trace", &filterState.showTrace}, {"Debug", &filterState.showDebug},
      {"Info", &filterState.showInfo},   {"Warn", &filterState.showWarning},
      {"Fatal", &filterState.showFatal},
  };
  for (const Toggle &toggle : toggles) {
    drawInlineCheckbox(toggle.label, *toggle.enabled);
  }
}

void drawLogMessages(const LogModel &model, LogFilterState &filterState,
                     std::pmr::memory_resource *scratchResource) {
  std::pmr::vector<size_t> visibleIndices(
      scratchResource ? scratchResource : std::pmr::get_default_resource());
  visibleIndices.reserve(model.lines.size());
  for (size_t lineIndex = 0; lineIndex < model.lines.size(); ++lineIndex) {
    const auto &line = model.lines[lineIndex];
    if (!filterState.levelEnabled(line.level)) {
      continue;
    }
    if (!filterState.textFilter.PassFilter(line.message.c_str())) {
      continue;
    }
    visibleIndices.push_back(lineIndex);
  }

  ImGui::BeginChild("LogScroll", ImVec2(0.0f, 0.0f), false,
                    ImGuiWindowFlags_HorizontalScrollbar);

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(visibleIndices.size()));
  while (clipper.Step()) {
    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
      const size_t visibleLineIndex = visibleIndices[static_cast<size_t>(i)];
      const LogLine &line = model.lines[visibleLineIndex];
      const std::string_view tag = logTagFor(line.level);
      ImGui::TextUnformatted(tag.data(), tag.data() + tag.size());
      ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
      ImGui::TextUnformatted(line.message.c_str());
    }
  }

  if (filterState.autoScroll && filterState.requestScroll) {
    ImGui::SetScrollHereY(1.0f);
    filterState.requestScroll = false;
  }
  ImGui::EndChild();
}

void drawInspectorHeader(std::string_view label) {
  ImGui::TextUnformatted(label.data(), label.data() + label.size());
  ImGui::Separator();
}

bool passKindUsesFeatureToggle(PassInspectorKind kind) {
  switch (kind) {
  case PassInspectorKind::Skybox:
  case PassInspectorKind::Shadow:
  case PassInspectorKind::Opaque:
  case PassInspectorKind::Transmission:
  case PassInspectorKind::Transparent:
  case PassInspectorKind::AmbientOcclusion:
  case PassInspectorKind::Debug:
    return true;
  case PassInspectorKind::Composite:
  case PassInspectorKind::AntiAliasing:
  case PassInspectorKind::Generic:
    return false;
  }
  return false;
}

bool *renderSettingToggleForPassKind(RenderSettings &renderSettings,
                                     PassInspectorKind kind) {
  switch (kind) {
  case PassInspectorKind::Skybox:
    return &renderSettings.skybox.enabled;
  case PassInspectorKind::Shadow:
    return &renderSettings.shadow.enabled;
  case PassInspectorKind::Opaque:
    return &renderSettings.opaque.enabled;
  case PassInspectorKind::Transmission:
    return &renderSettings.transmission.enabled;
  case PassInspectorKind::Transparent:
    return &renderSettings.transparent.enabled;
  case PassInspectorKind::Debug:
    return &renderSettings.debug.enabled;
  case PassInspectorKind::AmbientOcclusion:
  case PassInspectorKind::Composite:
  case PassInspectorKind::AntiAliasing:
  case PassInspectorKind::Generic:
    return nullptr;
  }
  return nullptr;
}

bool isPassInFamily(const RenderPipelinePassInfo &candidate,
                    const RenderPipelinePassInfo &selected,
                    PassInspectorKind selectedKind) {
  if (!selected.featureName.empty() &&
      candidate.featureName == selected.featureName) {
    return true;
  }
  return classifyPassInspector(candidate.featureName, candidate.passName) ==
         selectedKind;
}

bool isPassFamilyEnabled(RenderPipeline *renderPipeline,
                         const RenderPipelinePassInfo &selected,
                         PassInspectorKind kind) {
  if (renderPipeline == nullptr) {
    return false;
  }
  bool sawFamilyPass = false;
  for (size_t passIndex = 0; passIndex < renderPipeline->passCount();
       ++passIndex) {
    const auto candidate = renderPipeline->passInfo(passIndex);
    if (!candidate.has_value() || !isPassInFamily(*candidate, selected, kind)) {
      continue;
    }
    sawFamilyPass = true;
    if (!candidate->enabled) {
      return false;
    }
  }
  return sawFamilyPass;
}

void setPassFamilyEnabled(RenderPipeline *renderPipeline,
                          const RenderPipelinePassInfo &selected,
                          PassInspectorKind kind, bool enabled) {
  if (renderPipeline == nullptr) {
    return;
  }
  for (size_t passIndex = 0; passIndex < renderPipeline->passCount();
       ++passIndex) {
    const auto candidate = renderPipeline->passInfo(passIndex);
    if (!candidate.has_value() || !isPassInFamily(*candidate, selected, kind)) {
      continue;
    }
    renderPipeline->setPassEnabled(candidate->index, enabled);
  }
}

void syncFeatureToggleToRenderSettings(RenderSettings &renderSettings,
                                       PassInspectorKind kind, bool enabled) {
  if (kind == PassInspectorKind::AmbientOcclusion) {
    renderSettings.ambientOcclusion.mode =
        enabled ? AmbientOcclusionMode::GTAO : AmbientOcclusionMode::Disabled;
    sanitizeAmbientOcclusionSettings(renderSettings.ambientOcclusion,
                                     renderSettings.opaque,
                                     renderSettings.antiAliasing);
    return;
  }
  if (bool *const toggle = renderSettingToggleForPassKind(renderSettings, kind);
      toggle != nullptr) {
    *toggle = enabled;
  }
}

void drawSkyboxSettings(RenderSettings::SkyboxSettings &skybox) {
  ImGui::Text("Skybox background: %s", skybox.enabled ? "enabled" : "disabled");
}

[[nodiscard]] std::string_view
meshletModeLabel(MeshletRenderMode mode) noexcept {
  switch (mode) {
  case MeshletRenderMode::Disabled:
    return "Indexed";
  case MeshletRenderMode::Opportunistic:
    return "Hybrid";
  case MeshletRenderMode::Required:
    return "Mesh shaders only";
  default:
    return "Unknown";
  }
}

[[nodiscard]] std::string_view
activeGeometryRouteLabel(MeshletRenderMode requestedMode,
                         const OpaqueFrameMetrics &metrics) noexcept {
  if (metrics.meshletModeActive != 0u) {
    return metrics.meshletHybridActive != 0u ? "Hybrid" : "Mesh shaders";
  }
  if (metrics.meshletRejectedMissingFeature != 0u) {
    return "Indexed fallback (mesh shaders unavailable)";
  }
  if (metrics.meshletRejectedMissingAssetData != 0u) {
    return "Indexed fallback (meshlet data missing)";
  }
  if (metrics.meshletRejectedIncompatibleFrame != 0u) {
    return "Indexed fallback (frame incompatible)";
  }
  return requestedMode == MeshletRenderMode::Disabled ? "Indexed"
                                                      : "Indexed fallback";
}

void drawOpaqueSettings(RenderSettings::OpaqueSettings &opaque,
                        const RenderSettings::ShadowSettings &shadow,
                        const OpaqueFrameMetrics &metrics) {
  constexpr const char *kDebugModes[] = {
      "None",
      "Wire Overlay",
      "Wireframe Only",
      "Tess Patch (Edges + Heatmap)",
  };
  int debugMode = static_cast<int>(opaque.debugVisualization);
  debugMode =
      std::clamp(debugMode, 0, static_cast<int>(IM_ARRAYSIZE(kDebugModes)) - 1);
  if (ImGui::Combo("Debug Visualization##OpaquePass", &debugMode, kDebugModes,
                   IM_ARRAYSIZE(kDebugModes))) {
    opaque.debugVisualization =
        static_cast<OpaqueDebugVisualization>(debugMode);
  }
  if (opaque.debugVisualization ==
      OpaqueDebugVisualization::TessPatchEdgesHeatmap) {
    ImGui::TextUnformatted(
        "Patch mode auto-enables tessellation for visualization.");
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Geometry Route");
  constexpr const char *kGeometryRoutes[] = {
      "Indexed",
      "Hybrid",
      "Mesh shaders only",
  };
  int geometryRoute = std::clamp(static_cast<int>(opaque.meshletMode), 0, 2);
  if (ImGui::Combo("Route##OpaquePass", &geometryRoute, kGeometryRoutes,
                   IM_ARRAYSIZE(kGeometryRoutes))) {
    opaque.meshletMode = static_cast<MeshletRenderMode>(geometryRoute);
  }
  const std::string_view requestedRoute = meshletModeLabel(opaque.meshletMode);
  const std::string_view activeRoute =
      activeGeometryRouteLabel(opaque.meshletMode, metrics);
  ImGui::Text("Requested: %.*s", static_cast<int>(requestedRoute.size()),
              requestedRoute.data());
  ImGui::Text("Active: %.*s", static_cast<int>(activeRoute.size()),
              activeRoute.data());
  if (metrics.meshletHybridActive != 0u) {
    ImGui::Text("Hybrid batches: %u indexed / %u meshlet",
                metrics.meshletHybridClassicBatches,
                metrics.meshletHybridMeshletBatches);
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Depth");
  ImGui::Checkbox("Enable Depth Pre-pass##OpaquePass",
                  &opaque.enableDepthPrepass);
  ImGui::Checkbox("Enable Depth Pyramid##OpaquePass",
                  &opaque.enableDepthPyramid);
  ImGui::Text("Effective Depth Pre-pass: %s",
              opaque.enableDepthPrepass ? "enabled" : "disabled");
  const bool forcedDepthPyramid = shadow.enabled && !opaque.enableDepthPyramid;
  ImGui::Text("Effective Depth Pyramid: %s",
              (opaque.enableDepthPyramid || forcedDepthPyramid) ? "enabled"
                                                                : "disabled");
  if (forcedDepthPyramid) {
    ImGui::TextUnformatted(
        "Depth pyramid is forced on while previous-frame SDSM is active.");
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Mesh LOD");
  ImGui::Checkbox("Enable Indirect Draws##OpaquePass",
                  &opaque.enableIndirectDraw);
  ImGui::Checkbox("Enable Instanced Draws##OpaquePass",
                  &opaque.enableInstancedDraw);
  ImGui::Checkbox("Enable Mesh LOD##OpaquePass", &opaque.enableMeshLod);
  ImGui::SliderInt("Forced LOD##OpaquePass", &opaque.forcedMeshLod, -1, 3);

  ImGui::SliderFloat("LOD Pixel Error##OpaquePass",
                     &opaque.meshLodTargetPixelError, 0.1f, 8.0f, "%.2f");
  ImGui::SliderFloat("LOD Hysteresis##OpaquePass",
                     &opaque.meshLodHysteresisRatio, 0.0f, 0.75f, "%.2f");
  ImGui::Text("Auto LOD: %s, L0 %u / L1 %u, transitions %u",
              metrics.autoLodActive != 0u ? "active" : "inactive",
              metrics.autoLodLod0Instances, metrics.autoLodLod1Instances,
              metrics.autoLodTransitions);

  ImGui::Separator();
  ImGui::TextUnformatted("Tessellation");
  ImGui::Checkbox("Enable Tessellation##OpaquePass",
                  &opaque.enableTessellation);
  ImGui::SliderFloat("Tess Near##OpaquePass", &opaque.tessNearDistance, 0.0f,
                     256.0f, "%.2f");
  ImGui::SliderFloat("Tess Far##OpaquePass", &opaque.tessFarDistance, 0.0f,
                     512.0f, "%.2f");
  ImGui::SliderFloat("Tess Min##OpaquePass", &opaque.tessMinFactor, 1.0f, 64.0f,
                     "%.2f");
  ImGui::SliderFloat("Tess Max##OpaquePass", &opaque.tessMaxFactor, 1.0f, 64.0f,
                     "%.2f");
  int tessMaxInstances = static_cast<int>(
      std::min<uint32_t>(opaque.tessMaxInstances, kUiMaxTessInstances));
  if (ImGui::SliderInt("Tess Max Inst##OpaquePass", &tessMaxInstances, 0,
                       4096)) {
    opaque.tessMaxInstances =
        static_cast<uint32_t>(std::max(tessMaxInstances, 0));
  }
  opaque.tessNearDistance = std::max(0.0f, opaque.tessNearDistance);
  opaque.tessFarDistance =
      std::max(opaque.tessFarDistance, opaque.tessNearDistance + 1.0e-3f);
  opaque.tessMinFactor = std::clamp(opaque.tessMinFactor, 1.0f, 64.0f);
  opaque.tessMaxFactor =
      std::clamp(opaque.tessMaxFactor, opaque.tessMinFactor, 64.0f);
  opaque.tessMaxInstances =
      std::min<uint32_t>(opaque.tessMaxInstances, kUiMaxTessInstances);
}

void drawDebugSettings(RenderSettings::DebugSettings &debug) {
  ImGui::Checkbox("Model Bounds##DebugPasses", &debug.modelBounds);
  ImGui::Checkbox("Grid##DebugPasses", &debug.grid);
  ImGui::Checkbox("Light Icons##DebugPasses", &debug.lightIcons);
}

void drawShadowSettings(
    RenderSettings::ShadowSettings &shadow,
    const std::optional<ShadowDebugFrameData> *shadowDebugFrameData = nullptr) {
  sanitizeShadowSettings(shadow);
  shadow.shadowMapSize = snapShadowMapResolution(shadow.shadowMapSize);
  const auto markCustomPreset = [&]() {
    shadow.qualityPreset = ShadowQualityPreset::Custom;
  };
  ImGui::Checkbox("Enable Shadows##ShadowPass", &shadow.enabled);

  constexpr const char *kQualityPresetLabels[] = {
      "Custom", "Low", "Medium", "High", "Ultra",
  };
  int qualityPreset =
      static_cast<int>(sanitizeShadowQualityPreset(shadow.qualityPreset));
  if (ImGui::Combo("Quality Preset##ShadowPass", &qualityPreset,
                   kQualityPresetLabels, IM_ARRAYSIZE(kQualityPresetLabels))) {
    applyShadowQualityPreset(shadow,
                             static_cast<ShadowQualityPreset>(qualityPreset));
    shadow.shadowMapSize = snapShadowMapResolution(shadow.shadowMapSize);
  }

  int cascadeCount = static_cast<int>(shadow.cascadeCount);
  if (ImGui::SliderInt("Cascade Count##ShadowPass", &cascadeCount, 1,
                       static_cast<int>(kMaxShadowCascades))) {
    shadow.cascadeCount = static_cast<uint32_t>(std::max(cascadeCount, 1));
    markCustomPreset();
  }

  int shadowMapResolution =
      static_cast<int>(shadowMapResolutionIndex(shadow.shadowMapSize));
  shadowMapResolution =
      std::clamp(shadowMapResolution, 0,
                 static_cast<int>(kShadowMapResolutions.size()) - 1);
  if (ImGui::SliderInt("Shadow Map Size##ShadowPass", &shadowMapResolution, 0,
                       static_cast<int>(kShadowMapResolutions.size()) - 1,
                       kShadowMapResolutionLabels[shadowMapResolution])) {
    shadowMapResolution =
        std::clamp(shadowMapResolution, 0,
                   static_cast<int>(kShadowMapResolutions.size()) - 1);
    shadow.shadowMapSize =
        kShadowMapResolutions[static_cast<size_t>(shadowMapResolution)];
    markCustomPreset();
  }

  if (ImGui::SliderFloat("Max Distance##ShadowPass", &shadow.maxDistance, 1.0f,
                         1000.0f, "%.1f")) {
    markCustomPreset();
  }
  if (ImGui::SliderFloat("Max Distance Fade##ShadowPass",
                         &shadow.maxDistanceFadeFraction, 0.0f, 1.0f, "%.2f")) {
    markCustomPreset();
  }
  if (shadowDebugFrameData != nullptr && shadowDebugFrameData->has_value()) {
    ImGui::Text("Resolved Fade: %.3f .. %.3f",
                (*shadowDebugFrameData)->maxDistanceFadeStart,
                (*shadowDebugFrameData)->maxDistanceFadeEnd);
  }
  if (ImGui::SliderFloat("Split Lambda##ShadowPass", &shadow.splitLambda, 0.0f,
                         1.0f, "%.2f")) {
    markCustomPreset();
  }
  if (ImGui::SliderFloat("Cascade Blend Fraction##ShadowPass",
                         &shadow.cascadeBlendFraction, 0.0f, 1.0f, "%.2f")) {
    markCustomPreset();
  }
  if (shadow.cascadeCount == 1u) {
    ImGui::TextUnformatted("Cascade blending is only visible when more than "
                           "one cascade is active.");
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Bias");
  if (ImGui::DragFloat("Constant Bias##ShadowPass", &shadow.constantBias,
                       0.0001f, -0.1f, 0.1f, "%.5f")) {
    markCustomPreset();
  }
  if (ImGui::DragFloat("Slope Bias##ShadowPass", &shadow.slopeBias, 0.05f, 0.0f,
                       16.0f, "%.2f")) {
    markCustomPreset();
  }
  if (ImGui::DragFloat("Normal Bias (texels)##ShadowPass", &shadow.normalBias,
                       0.005f, 0.0f, 8.0f, "%.3f")) {
    markCustomPreset();
  }
  const std::optional<float> debugCascadeTexelWorldSize =
      shadowDebugCascadeTexelWorldSize(shadow, shadowDebugFrameData);
  if (debugCascadeTexelWorldSize.has_value()) {
    float debugCascadeWorldBias =
        normalBiasWorldUnits(shadow.normalBias, *debugCascadeTexelWorldSize);
    const float dragSpeed =
        std::max(*debugCascadeTexelWorldSize * 0.25f, 1.0e-5f);
    if (ImGui::DragFloat("Normal Bias (world, debug cascade)##ShadowPass",
                         &debugCascadeWorldBias, dragSpeed, 0.0f, 1000.0f,
                         "%.6f")) {
      shadow.normalBias =
          std::max(debugCascadeWorldBias / *debugCascadeTexelWorldSize, 0.0f);
      markCustomPreset();
    }
    ImGui::Text("Debug Cascade Bias Scale: %.6f world units per texel",
                *debugCascadeTexelWorldSize);
  } else {
    ImGui::TextUnformatted(
        "Normal bias is stored in texels. World-unit conversion appears once "
        "shadow debug data is available.");
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Filtering");
  int pcfSamples = static_cast<int>(shadow.pcfSampleCount);
  if (ImGui::SliderInt("PCF Samples##ShadowPass", &pcfSamples, 1, 64)) {
    shadow.pcfSampleCount = static_cast<uint32_t>(std::max(pcfSamples, 1));
    markCustomPreset();
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Previous-frame min/max SDSM");
  if (ImGui::SliderFloat("SDSM Temporal Blend##ShadowPass",
                         &shadow.sdsmTemporalBlend, 0.0f, 1.0f, "%.2f")) {
    markCustomPreset();
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Debug");
  ImGui::Checkbox("Cascade Frusta##ShadowPass",
                  &shadow.debug.showCascadeFrusta);
  ImGui::Checkbox("Light View Bounds##ShadowPass",
                  &shadow.debug.showLightViewBounds);
  ImGui::Checkbox("Texel Grid Snap##ShadowPass",
                  &shadow.debug.showTexelGridSnap);
  ImGui::Checkbox("Shadow Map Viewport##ShadowPass",
                  &shadow.debug.showShadowMapViewport);
  int debugCascade = static_cast<int>(shadow.debug.debugCascadeIndex);
  if (ImGui::SliderInt("Debug Cascade##ShadowPass", &debugCascade, 0,
                       static_cast<int>(kMaxShadowCascades) - 1)) {
    shadow.debug.debugCascadeIndex =
        static_cast<uint32_t>(std::max(debugCascade, 0));
  }
  ImGui::Checkbox("Freeze Cascades##ShadowPass", &shadow.debug.freezeCascades);
  ImGui::Checkbox("Freeze Light View##ShadowPass",
                  &shadow.debug.freezeLightView);
  ImGui::Checkbox("Cull Casters Per Cascade (debug)##ShadowPass",
                  &shadow.debug.enableCascadeCasterCulling);
  ImGui::Checkbox("Visualize Cascade Index##ShadowPass",
                  &shadow.debug.visualizeCascadeIndex);
  ImGui::Checkbox("Visualize Shadow Factor##ShadowPass",
                  &shadow.debug.visualizeShadowFactor);
  ImGui::Checkbox("Visualize PCF Result##ShadowPass",
                  &shadow.debug.visualizePCFResult);
  ImGui::Checkbox("Visualize Receiver Depth##ShadowPass",
                  &shadow.debug.visualizeReceiverDepth);
  ImGui::Checkbox("Visualize Shadow Map Depth##ShadowPass",
                  &shadow.debug.visualizeShadowMapDepth);
  ImGui::Separator();
  ImGui::TextUnformatted("Diagnostic Logs");
  ImGui::Checkbox("Log Shadow Diagnostics##ShadowPass",
                  &shadow.debug.logDiagnostics);
  int diagnosticLogLevel = logLevelComboIndex(shadow.debug.diagnosticLogLevel);
  if (ImGui::Combo(
          "Log Level##ShadowPass", &diagnosticLogLevel,
          [](void *, int idx, const char **outText) {
            if (idx < 0 || idx >= static_cast<int>(std::size(kLogLevels))) {
              return false;
            }
            *outText = kLogLevels[static_cast<size_t>(idx)].tag.data();
            return true;
          },
          nullptr, static_cast<int>(std::size(kLogLevels)))) {
    setLogLevelFromCombo(shadow.debug.diagnosticLogLevel, diagnosticLogLevel);
  }
  int diagnosticLogInterval = static_cast<int>(std::min<uint32_t>(
      shadow.debug.diagnosticLogIntervalFrames,
      static_cast<uint32_t>(std::numeric_limits<int>::max())));
  if (ImGui::SliderInt("Log Interval (frames)##ShadowPass",
                       &diagnosticLogInterval, 1, 240)) {
    shadow.debug.diagnosticLogIntervalFrames =
        static_cast<uint32_t>(std::max(diagnosticLogInterval, 1));
  }
  ImGui::Checkbox("Log Only On Change##ShadowPass",
                  &shadow.debug.diagnosticLogOnlyOnChange);
  ImGui::TextUnformatted(
      "Logs emit one frame summary plus one line per active cascade.");

  ImGui::Separator();
  ImGui::TextUnformatted("Depth Preview");
  constexpr const char *kPreviewModeLabels[] = {
      "Selected Cascade",
      "Tiled All Cascades",
  };
  int previewMode =
      static_cast<int>(sanitizeShadowPreviewMode(shadow.debug.previewMode));
  if (ImGui::Combo("Preview Mode##ShadowPass", &previewMode, kPreviewModeLabels,
                   IM_ARRAYSIZE(kPreviewModeLabels))) {
    shadow.debug.previewMode = static_cast<ShadowPreviewMode>(previewMode);
  }
  float previewRange[2] = {
      shadow.debug.previewDepthMin,
      shadow.debug.previewDepthMax,
  };
  if (ImGui::SliderFloat2("Preview Range##ShadowPass", previewRange, 0.0f, 1.0f,
                          "%.4f")) {
    shadow.debug.previewDepthMin = previewRange[0];
    shadow.debug.previewDepthMax = previewRange[1];
  }
  if (ImGui::Button("Full 0-1##ShadowPreviewPreset")) {
    shadow.debug.previewDepthMin = 0.0f;
    shadow.debug.previewDepthMax = 1.0f;
  }
  ImGui::SameLine();
  if (ImGui::Button("Near 0.90-1##ShadowPreviewPreset")) {
    shadow.debug.previewDepthMin = 0.90f;
    shadow.debug.previewDepthMax = 1.0f;
  }
  ImGui::SameLine();
  if (ImGui::Button("Tight 0.98-1##ShadowPreviewPreset")) {
    shadow.debug.previewDepthMin = 0.98f;
    shadow.debug.previewDepthMax = 1.0f;
  }
  ImGui::Checkbox("Invert Preview##ShadowPass",
                  &shadow.debug.previewDepthInvert);
  ImGui::SameLine();
  ImGui::Checkbox("Log Preview##ShadowPass", &shadow.debug.previewDepthLog);

  sanitizeShadowSettings(shadow);
  ImGui::Separator();
  ImGui::Text("Quality Preset: %s",
              shadowQualityPresetDisplayName(shadow.qualityPreset));
}

void drawTransparentSettings(RenderSettings::TransparentSettings &transparent) {
  ImGui::Text("Transparent blending: %s",
              transparent.enabled ? "enabled" : "disabled");
}

void drawTransmissionSettings(
    RenderSettings::TransmissionSettings &transmission) {
  ImGui::Text("Transmission shading: %s",
              transmission.enabled ? "enabled" : "disabled");
  ImGui::Checkbox("TAA Jitter Prefilter##Transmission",
                  &transmission.taaJitterPrefilter);
  ImGui::SliderFloat("TAA Jitter Prefilter Max LOD##Transmission",
                     &transmission.taaJitterPrefilterMaxLod, 0.0f, 2.0f,
                     "%.2f");
  ImGui::SliderFloat("TAA Jitter Depth Bias##Transmission",
                     &transmission.taaJitterDepthBiasConstant, -64.0f, 0.0f,
                     "%.1f");
}

void drawAmbientOcclusionSettings(
    RenderSettings::AmbientOcclusionSettings &ao,
    const RenderSettings::OpaqueSettings &opaque,
    const RenderSettings::AntiAliasingSettings &antiAliasing,
    const RenderFrameMetrics &frameMetrics) {
  sanitizeAmbientOcclusionSettings(ao, opaque, antiAliasing);

  int modeIndex = static_cast<int>(sanitizeAmbientOcclusionMode(ao.mode));
  modeIndex = std::clamp(
      modeIndex, 0, static_cast<int>(kAmbientOcclusionModeLabels.size()) - 1);
  if (ImGui::Combo("Mode##AmbientOcclusion", &modeIndex,
                   kAmbientOcclusionModeLabels.data(),
                   static_cast<int>(kAmbientOcclusionModeLabels.size()))) {
    ao.mode = static_cast<AmbientOcclusionMode>(modeIndex);
    sanitizeAmbientOcclusionSettings(ao, opaque, antiAliasing);
  }

  int presetIndex = static_cast<int>(sanitizeAmbientOcclusionPreset(ao.preset));
  presetIndex =
      std::clamp(presetIndex, 0,
                 static_cast<int>(kAmbientOcclusionPresetLabels.size()) - 1);
  if (ImGui::Combo("Preset##AmbientOcclusion", &presetIndex,
                   kAmbientOcclusionPresetLabels.data(),
                   static_cast<int>(kAmbientOcclusionPresetLabels.size()))) {
    ao.preset = static_cast<AmbientOcclusionPreset>(presetIndex);
    sanitizeAmbientOcclusionSettings(ao, opaque, antiAliasing);
  }

  ImGui::SliderFloat("Strength##AmbientOcclusion", &ao.strength, 0.0f, 1.0f,
                     "%.2f");
  const bool temporalAccumulationDisabled =
      ao.mode == AmbientOcclusionMode::Disabled || !opaque.enabled ||
      sanitizeAntiAliasingMode(antiAliasing.mode) == AntiAliasingMode::MSAA4x;
  ImGui::BeginDisabled(temporalAccumulationDisabled);
  ImGui::Checkbox("Temporal Accumulation##AmbientOcclusion",
                  &ao.temporalAccumulation);
  ImGui::EndDisabled();

  int debugViewIndex =
      static_cast<int>(sanitizeAmbientOcclusionDebugView(ao.debugView));
  debugViewIndex =
      std::clamp(debugViewIndex, 0,
                 static_cast<int>(kAmbientOcclusionDebugViewLabels.size()) - 1);
  if (ImGui::Combo("Debug View##AmbientOcclusion", &debugViewIndex,
                   kAmbientOcclusionDebugViewLabels.data(),
                   static_cast<int>(kAmbientOcclusionDebugViewLabels.size()))) {
    ao.debugView = static_cast<AmbientOcclusionDebugView>(debugViewIndex);
  }

  sanitizeAmbientOcclusionSettings(ao, opaque, antiAliasing);
  const AmbientOcclusionFrameMetrics &metrics = frameMetrics.ambientOcclusion;

  ImGui::Separator();
  ImGui::Text("Mode: %s", ambientOcclusionModeDisplayName(ao.mode));
  ImGui::Text("Preset: %s", ambientOcclusionPresetDisplayName(ao.preset));
  ImGui::Text("Debug View: %s",
              ambientOcclusionDebugViewDisplayName(ao.debugView));
  ImGui::Text("Requested Active: %s", ao.active ? "yes" : "no");
  ImGui::Text("Effective Active: %s", metrics.active ? "yes" : "no");
  ImGui::Text("Settings Disabled Reason: %s",
              ambientOcclusionDisabledReasonDisplayName(ao.disabledReason));
  ImGui::Text(
      "Frame Disabled Reason: %s",
      ambientOcclusionDisabledReasonDisplayName(metrics.disabledReason));

  if (ImGui::CollapsingHeader("Metrics##AmbientOcclusion",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    constexpr double MiB = 1024.0 * 1024.0;
    ImGui::Text("Dimensions: %u x %u", metrics.width, metrics.height);
    ImGui::Text("Normal Format: %s", formatDisplayName(metrics.normalFormat));
    ImGui::Text("AO Format: %s",
                formatDisplayName(metrics.ambientOcclusionFormat));
    ImGui::Text("Normal Prepass Draws: %u", metrics.normalPrepassDraws);
    ImGui::Text("Passes: depth %u, main %u, denoise %u, temporal %u",
                metrics.depthPrefilterPassCount, metrics.mainPassCount,
                metrics.denoisePassCount, metrics.temporalPassCount);
    ImGui::Text("Requested Sampling: slices %u, steps %u, denoise %u",
                metrics.requestedSliceCount, metrics.requestedStepCount,
                metrics.requestedDenoisePassCount);
    ImGui::Text("Effective Sampling: slices %u, steps %u, denoise %u, mips %u",
                metrics.sliceCount, metrics.stepCount, metrics.denoisePassCount,
                metrics.depthMipCount);
    ImGui::Text(
        "GPU Time: %.3f ms (%s, frame %llu)", metrics.gpuTimeMs,
        metrics.gpuTimingAvailable != 0u ? "ready" : "pending",
        static_cast<unsigned long long>(metrics.gpuTimingSourceFrameIndex));
    ImGui::Text("Textures: %u, %.2f MiB", metrics.textureCount,
                static_cast<double>(metrics.totalTextureBytes) / MiB);
    ImGui::Text("GTAO Textures: depth %.2f, edge %.2f, work %.2f MiB",
                static_cast<double>(metrics.depthPrefilterTextureBytes) / MiB,
                static_cast<double>(metrics.edgeTextureBytes) / MiB,
                static_cast<double>(metrics.scratchTextureBytes) / MiB);
    ImGui::Text("Normal Texture Churn: alloc %u, realloc %u",
                metrics.normalTextureAllocationCount,
                metrics.normalTextureReallocationCount);
    ImGui::Text("AO Texture Churn: alloc %u, realloc %u",
                metrics.ambientOcclusionTextureAllocationCount,
                metrics.ambientOcclusionTextureReallocationCount);
    ImGui::Text("Outputs: scalar %s, bent normal %s",
                metrics.scalarAoAvailable ? "yes" : "no",
                metrics.bentNormalAvailable ? "yes" : "no");
    ImGui::Text("Temporal: requested %s, active %s, history %s, motion %s",
                metrics.temporalAccumulationEnabled ? "yes" : "no",
                metrics.temporalAccumulationActive ? "yes" : "no",
                metrics.temporalHistoryInvalidated
                    ? "invalidated"
                    : (metrics.temporalHistoryValid ? "valid" : "invalid"),
                metrics.temporalMotionVectorsConsumed ? "consumed"
                                                      : "not used");
    ImGui::Text("Graph: normals %s, AO %s",
                metrics.normalGraphPublished ? "published" : "missing",
                metrics.ambientOcclusionGraphPublished ? "published"
                                                       : "missing");
  }
}

void drawToneMapSettings(RenderSettings::ToneMapSettings &toneMap) {
  sanitizeToneMapSettings(toneMap);
  constexpr const char *kToneMapperLabels[] = {"ACES 2 SDR", "AgX"};
  int toneMapperIndex = static_cast<int>(toneMap.operator_);
  toneMapperIndex =
      std::clamp(toneMapperIndex, 0,
                 static_cast<int>(IM_ARRAYSIZE(kToneMapperLabels)) - 1);
  if (ImGui::Combo("Tone Mapper##CompositePass", &toneMapperIndex,
                   kToneMapperLabels, IM_ARRAYSIZE(kToneMapperLabels))) {
    toneMap.operator_ = static_cast<ToneMapper>(toneMapperIndex);
  }
  ImGui::SliderFloat("Exposure (EV)##CompositePass", &toneMap.exposureEv,
                     -10.0f, 10.0f, "%.2f");
  ImGui::SliderFloat("ACES Offset (EV)##CompositePass",
                     &toneMap.acesExposureOffsetEv, -4.0f, 4.0f, "%.2f");
  ImGui::SliderFloat("AgX Offset (EV)##CompositePass",
                     &toneMap.agxExposureOffsetEv, -4.0f, 4.0f, "%.2f");
  ImGui::Text("Effective ACES EV: %.2f",
              toneMap.exposureEv + toneMap.acesExposureOffsetEv);
  ImGui::Text("Effective AgX EV: %.2f",
              toneMap.exposureEv + toneMap.agxExposureOffsetEv);
  ImGui::Checkbox("Gray Card Debug##CompositePass", &toneMap.grayCardDebug);
  ImGui::Checkbox("Side-by-Side Compare##CompositePass",
                  &toneMap.sideBySideCompare);
  if (toneMap.sideBySideCompare) {
    ImGui::SliderFloat("Compare Split##CompositePass", &toneMap.compareSplit,
                       kMinToneMapCompareSplit, kMaxToneMapCompareSplit,
                       "%.2f");
    const ToneMapper compareMapper = toneMap.operator_ == ToneMapper::AgX
                                         ? ToneMapper::ACES2_SDR
                                         : ToneMapper::AgX;
    ImGui::Text("Left: %s", toneMapperDisplayName(toneMap.operator_));
    ImGui::Text("Right: %s", toneMapperDisplayName(compareMapper));
  }
}

void drawHDRPostProcessSettings(RenderSettings::HDRPostProcessSettings &hdr,
                                const RenderFrameMetrics &frameMetrics) {
  sanitizeHDRPostProcessSettings(hdr);
  ImGui::Separator();
  ImGui::TextUnformatted("HDR Postprocess");
  ImGui::Checkbox("Bloom##HDRPostProcess", &hdr.bloomEnabled);
  if (hdr.bloomEnabled) {
    ImGui::SliderFloat("Bloom Strength##HDRPostProcess", &hdr.bloomStrength,
                       0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Bloom Threshold##HDRPostProcess", &hdr.bloomThreshold,
                       0.0f, 16.0f, "%.2f");
    ImGui::SliderFloat("Bloom Soft Knee##HDRPostProcess", &hdr.bloomSoftKnee,
                       0.0f, 4.0f, "%.2f");
    int bloomMipCount = static_cast<int>(hdr.bloomMaxMipCount);
    if (ImGui::SliderInt("Bloom Mips##HDRPostProcess", &bloomMipCount, 1,
                         static_cast<int>(kMaxSceneDepthPyramidLevels))) {
      hdr.bloomMaxMipCount = static_cast<uint32_t>(std::max(bloomMipCount, 1));
    }
  }
  ImGui::Checkbox("Adaptation##HDRPostProcess", &hdr.adaptationEnabled);
  if (hdr.adaptationEnabled) {
    ImGui::SliderFloat("Target Gray##HDRPostProcess", &hdr.adaptationTargetGray,
                       0.01f, 2.0f, "%.3f");
    ImGui::SliderFloat("Adaptation Speed##HDRPostProcess", &hdr.adaptationSpeed,
                       0.0f, 16.0f, "%.2f");
    ImGui::SliderFloat("Min EV##HDRPostProcess", &hdr.adaptationMinEv, -16.0f,
                       16.0f, "%.2f");
    ImGui::SliderFloat("Max EV##HDRPostProcess", &hdr.adaptationMaxEv, -16.0f,
                       16.0f, "%.2f");
  }
  int hdrDebugIndex = static_cast<int>(hdr.debugView);
  hdrDebugIndex =
      std::clamp(hdrDebugIndex, 0,
                 static_cast<int>(kHDRPostProcessDebugViewLabels.size()) - 1);
  if (ImGui::Combo("HDR Debug##HDRPostProcess", &hdrDebugIndex,
                   kHDRPostProcessDebugViewLabels.data(),
                   static_cast<int>(kHDRPostProcessDebugViewLabels.size()))) {
    hdr.debugView = static_cast<HDRPostProcessDebugView>(hdrDebugIndex);
  }
  sanitizeHDRPostProcessSettings(hdr);

  const HDRPostProcessFrameMetrics &hdrMetrics = frameMetrics.hdrPostProcess;
  ImGui::Text("Bloom active: %s", hdrMetrics.bloomActive ? "yes" : "no");
  ImGui::Text("Adaptation active: %s",
              hdrMetrics.adaptationActive ? "yes" : "no");
  ImGui::Text("Exposure history: %s",
              hdrMetrics.exposureHistoryValid ? "valid" : "invalid");
  ImGui::Text("Adapted EV: %.2f  Effective EV: %.2f",
              hdrMetrics.adaptedExposureEv, hdrMetrics.effectiveExposureEv);
  ImGui::Text("Exposure textures: alloc %u, realloc %u",
              hdrMetrics.exposureTextureAllocationCount,
              hdrMetrics.exposureTextureReallocationCount);
  ImGui::Text("Exposure history: alloc %u, realloc %u",
              hdrMetrics.exposureHistoryAllocationCount,
              hdrMetrics.exposureHistoryReallocationCount);
}

void drawCompositeSettings(RenderSettings::ToneMapSettings &toneMap,
                           RenderSettings::HDRPostProcessSettings &hdr,
                           const RenderFrameMetrics &frameMetrics) {
  drawToneMapSettings(toneMap);
  drawHDRPostProcessSettings(hdr, frameMetrics);
  ImGui::Separator();
  ImGui::TextUnformatted(
      "Frame composition is always active and presents the HDR frame");
  ImGui::TextUnformatted("through downsample, resolve, and tone-map passes.");
}

bool hasTemporalAAFeature(RenderPipeline *renderPipeline) {
  if (renderPipeline == nullptr) {
    return false;
  }
  for (size_t passIndex = 0; passIndex < renderPipeline->passCount();
       ++passIndex) {
    const auto pass = renderPipeline->passInfo(passIndex);
    if (pass.has_value() && pass->featureName == "TemporalAAFeature") {
      return true;
    }
  }
  return false;
}

std::string antiAliasingSettingsSummary(
    const RenderSettings::AntiAliasingSettings &settings,
    bool temporalFeaturePresent) {
  const RenderSettings::AntiAliasingDebugSettings effectiveDebug =
      effectiveTemporalAADebugSettings(settings);
  std::string summary = "AA mode=";
  summary += antiAliasingModeDisplayName(settings.mode);
  summary += " taaPreset=";
  summary += temporalAAQualityPresetDisplayName(settings.qualityPreset);
  summary += " jitter=";
  summary += settings.debug.jitterEnabled ? "enabled" : "disabled";
  summary += " freezeJitter=";
  summary += settings.debug.freezeJitter ? "true" : "false";
  summary += " logDiagnostics=";
  summary += settings.debug.logDiagnostics ? "true" : "false";
  summary += " logIntervalSec=";
  summary += std::format("{:.2f}", settings.debug.diagnosticLogIntervalSeconds);
  summary += " debugView=";
  summary += antiAliasingDebugViewDisplayName(settings.debug.view);
  summary += " taaJitterScale=";
  summary += std::format("{:.2f}", effectiveDebug.taaJitterScale);
  summary += " taaCurrentWeight=";
  summary += std::format("{:.2f}", effectiveDebug.taaCurrentFrameWeight);
  summary += " taaSharpen=";
  summary += effectiveDebug.taaSharpenEnabled ? "enabled" : "disabled";
  summary += " taaSharpenStrength=";
  summary += std::format("{:.2f}", effectiveDebug.taaSharpenStrength);
  summary += " taaMaterialMipBias=";
  summary += effectiveDebug.taaMaterialMipBiasEnabled ? "enabled" : "disabled";
  summary += "/";
  summary += std::format("{:.2f}", effectiveDebug.taaMaterialMipBias);
  summary += " transparentPostTaaSpatialCleanup=";
  summary +=
      effectiveDebug.transparentPostTaaSpatialCleanup ? "enabled" : "disabled";
  summary += " spatialPostTaaCleanup=";
  summary += effectiveDebug.spatialPostTaaCleanup ? "enabled" : "disabled";
  summary += " taaMotionWeight=";
  summary += std::format("{:.2f}", effectiveDebug.taaMotionCurrentWeight);
  summary += " taaClampWeight=";
  summary += std::format("{:.2f}", effectiveDebug.taaClampCurrentWeight);
  summary += " taaHistoryFilter=";
  summary += temporalAAHistoryFilterModeDisplayName(
      effectiveDebug.taaHistoryFilterMode);
  summary += " taaClampMode=";
  summary += temporalAAClampModeDisplayName(effectiveDebug.taaClampMode);
  summary += " taaHdrWeighting=";
  summary +=
      temporalAAHdrWeightingModeDisplayName(effectiveDebug.taaHdrWeightingMode);
  summary += " resetHistoryRequested=";
  summary += settings.debug.resetHistoryRequested ? "true" : "false";
  summary += " temporalFeaturePresent=";
  summary += temporalFeaturePresent ? "true" : "false";
  return summary;
}

struct AAMemoryAndBandwidthMetrics {
  uint64_t residentBytes = 0u;
  uint64_t bandwidthBytes = 0u;
};

AAMemoryAndBandwidthMetrics
ComputeAAMetrics(const AntiAliasingFrameMetrics &metrics) {
  return {
      .residentBytes =
          metrics.motionVectorTotalBytes + metrics.reactiveMaskTotalBytes +
          metrics.spatialAATotalBytes + metrics.spatialAALutTextureBytes +
          metrics.msaaTotalBytes,
      .bandwidthBytes = metrics.motionVectorClearBytes +
                        metrics.velocityPassBandwidthEstimateBytes +
                        metrics.velocityDebugBandwidthEstimateBytes +
                        metrics.reactiveMaskPassBandwidthEstimateBytes +
                        metrics.taaHistoryBandwidthEstimateBytes +
                        metrics.spatialAABandwidthEstimateBytes +
                        metrics.msaaResolveBandwidthEstimateBytes,
  };
}

std::string
antiAliasingMetricsSummary(const AntiAliasingFrameMetrics &metrics) {
  const std::string resetReason(
      temporalHistoryResetReasonName(metrics.historyResetReason));
  std::string summary = std::format(
      "frame=[jitter={{enabled={} frozen={} index={}/{} offset=({:.4f},{:.4f}) "
      "previousOffset=({:.4f},{:.4f}) delta=({:.4f},{:.4f}) "
      "deltaMag={:.4f} bounds={} oobCount={}}} "
      "camera={{pos=({:.3f},{:.3f},{:.3f}) "
      "previous=({:.3f},{:.3f},{:.3f}) delta={:.6f}}} "
      "history={{valid={} temporalDataValid={} "
      "resetReason={} framesSinceReset={} resetCount={}}} "
      "motionVectors={{allocated={} format={} formatSupported={} "
      "dimensions={}x{} ring={} currentBytes={} previousValid={} "
      "previousBytes={} totalBytes={} graphPublished={} "
      "previousGraphPublished={} previousDepthValid={} "
      "previousDepthBytes={} previousDepthGraphPublished={} "
      "clearPasses={} clearBytes={}}} "
      "opaqueVelocity={{generated={} passCount={} debugPassCount={} draws={} "
      "instances={} previousCacheValid={} previousValid={} previousMissing={} "
      "animatedResponsive={} animatedPreviousGeometry={} "
      "tessellatedSkipped={} objectMotionAvg={:.5f} "
      "objectMotionMax={:.5f} velocityAvg={:.5f} velocityMax={:.5f} "
      "staticResidual={:.6f} cameraDelta={:.6f} missingPreviousRatio={:.3f} "
      "edgeDiscontinuity={:.3f} passBytes={} debugBytes={} "
      "debugViewRendered={}}} "
      "resolve={{passes={} copyBackPasses={} dimensions={}x{} "
      "resolveGpuMs={:.3f} resolveTiming={} debugGpuMs={:.3f} "
      "debugTiming={} "
      "postTaaMipPasses={} postTaaMipChain={} transmissionPostTaa={} "
      "staleTransmissionFrames={} mipDebugView={} "
      "downsampleGpuMs={:.3f} downsampleTiming={} "
      "transmissionGpuMs={:.3f} transmissionTiming={} "
      "transmissionMipDebug={} transmissionFlicker={:.3f} "
      "transmissionJitterMinLod={:.2f} transmissionDepthBias={:.1f} "
      "transmissionStableDepth={} "
      "transparentPostTaaDraws={} transparentMesh={} "
      "transparentContributors={} "
      "transparentFixed={} transparentEdgeJitter={:.3f} "
      "jitterScale={:.2f} preset={} currentWeight={:.3f} "
      "historyWeight={:.3f} motionWeight={:.3f} clampWeight={:.3f} "
      "historyFilter={} historyTextureValid={} "
      "oobReprojectionMeasured=false qualityValidation={} "
      "currentFallbackFrames={} bandwidthBytes={} resolvedSceneColor={} "
      "debugViewRendered={} historyValidityView={} pixelInspectorView={} "
      "stabilityDiagnosticsView={} stabilityOwnershipView={} patchProbeView={} "
      "motionFilterView={} "
      "oobFallback={} bilinearHistory={} "
      "depthReject={} depthThreshold={:.4f} "
      "previousDepthReject={} velocityReject={} "
      "staticVelocitySanitization={} velocityThresholdPx={:.2f} "
      "previousVelocityDisocclusion={} "
      "neighborhoodClamp={} adaptiveBlend={} motionBlendPerPixel={:.2f} "
      "disocclusionWeight={:.2f} clampAttenuationEnabled={} "
      "clampAttenuation={:.2f} neighborhoodFallback={} clampMode={} "
      "varianceGamma={:.2f} hdrWeighting={} hdrStrength={:.2f} "
      "hdrWeightingEnabled={}}}]",
      boolLogValue(metrics.jitterEnabled), boolLogValue(metrics.jitterFrozen),
      metrics.jitterIndex, metrics.jitterSequenceLength,
      metrics.jitterPixelOffset.x, metrics.jitterPixelOffset.y,
      metrics.previousJitterPixelOffset.x, metrics.previousJitterPixelOffset.y,
      metrics.jitterDeltaPixelOffset.x, metrics.jitterDeltaPixelOffset.y,
      metrics.jitterDeltaMagnitude,
      metrics.jitterOutOfBounds ? "out-of-bounds" : "valid",
      metrics.jitterOutOfBoundsCount, metrics.cameraPosition.x,
      metrics.cameraPosition.y, metrics.cameraPosition.z,
      metrics.previousCameraPosition.x, metrics.previousCameraPosition.y,
      metrics.previousCameraPosition.z, metrics.cameraPositionDelta,
      boolLogValue(metrics.historyValid),
      boolLogValue(metrics.temporalDataValid), resetReason,
      metrics.framesSinceHistoryReset, metrics.historyResetCount,
      boolLogValue(metrics.motionVectorAllocated),
      formatDisplayName(metrics.motionVectorFormat),
      boolLogValue(metrics.motionVectorFormatSupported),
      metrics.motionVectorWidth, metrics.motionVectorHeight,
      metrics.motionVectorTextureCount,
      static_cast<unsigned long long>(metrics.motionVectorTextureBytes),
      boolLogValue(metrics.previousMotionVectorValid),
      static_cast<unsigned long long>(metrics.previousMotionVectorTextureBytes),
      static_cast<unsigned long long>(metrics.motionVectorTotalBytes),
      boolLogValue(metrics.motionVectorGraphPublished),
      boolLogValue(metrics.previousMotionVectorGraphPublished),
      boolLogValue(metrics.previousSceneDepthValid),
      static_cast<unsigned long long>(metrics.previousSceneDepthTextureBytes),
      boolLogValue(metrics.previousSceneDepthGraphPublished),
      metrics.motionVectorClearPassCount,
      static_cast<unsigned long long>(metrics.motionVectorClearBytes),
      boolLogValue(metrics.opaqueVelocityGenerated), metrics.velocityPassCount,
      metrics.velocityDebugPassCount, metrics.velocityDrawCount,
      metrics.velocityInstanceCount,
      boolLogValue(metrics.previousTransformCacheValid),
      metrics.velocityPreviousTransformValidCount,
      metrics.velocityMissingPreviousTransformCount,
      metrics.velocityAnimatedResponsiveCount,
      metrics.velocityAnimatedPreviousGeometryCount,
      metrics.velocityTessellatedSkippedDrawCount,
      metrics.velocityAverageObjectMotion, metrics.velocityMaxObjectMotion,
      metrics.velocityEstimatedAverageMagnitude,
      metrics.velocityEstimatedMaxMagnitude,
      metrics.velocityStaticResidualEstimate, metrics.velocityCameraMatrixDelta,
      metrics.velocityMissingPreviousRatio,
      metrics.velocityEdgeDiscontinuityEstimate,
      static_cast<unsigned long long>(
          metrics.velocityPassBandwidthEstimateBytes),
      static_cast<unsigned long long>(
          metrics.velocityDebugBandwidthEstimateBytes),
      boolLogValue(metrics.velocityDebugViewRendered),
      metrics.taaResolvePassCount, metrics.taaCopyBackPassCount,
      metrics.taaResolveWidth, metrics.taaResolveHeight,
      metrics.taaResolveGpuTimeMs, metrics.taaResolveGpuTimingAvailable,
      metrics.taaDebugGpuTimeMs, metrics.taaDebugGpuTimingAvailable,
      metrics.taaPostResolveSceneColorMipPassCount,
      boolLogValue(metrics.taaPostResolveSceneColorMipChainGenerated),
      boolLogValue(metrics.taaTransmissionPostResolveSceneColorConsumed),
      metrics.taaTransmissionStaleSceneColorFrameCount,
      boolLogValue(metrics.taaSceneColorMipDebugViewRendered),
      metrics.taaSceneColorDownsampleGpuTimeMs,
      metrics.taaSceneColorDownsampleGpuTimingAvailable,
      metrics.taaTransmissionGpuTimeMs,
      metrics.taaTransmissionGpuTimingAvailable,
      boolLogValue(metrics.taaTransmissionMipDebugViewRendered),
      metrics.taaTransmissionFlickerEstimate,
      metrics.taaTransmissionJitterMinLod,
      metrics.taaTransmissionDepthBiasConstant,
      boolLogValue(metrics.taaTransmissionStableVisibilityDepth),
      metrics.taaTransparentPostTaaDrawCount,
      metrics.taaTransparentPostTaaMeshDrawCount,
      metrics.taaTransparentPostTaaContributorDrawCount,
      metrics.taaTransparentPostTaaFixedDrawCount,
      metrics.taaTransparentEdgeJitterEstimate, metrics.taaJitterScale,
      temporalAAQualityPresetDisplayName(metrics.taaQualityPreset),
      metrics.taaCurrentFrameWeight, metrics.taaHistoryFrameWeight,
      metrics.taaMotionCurrentWeight, metrics.taaClampCurrentWeight,
      temporalAAHistoryFilterModeDisplayName(metrics.taaHistoryFilterMode),
      boolLogValue(metrics.historyValid),
      metrics.taaQualityValidationInvalidatedByFrozenJitter
          ? "invalid-freeze-jitter"
          : "valid",
      metrics.taaCurrentFallbackFrameCount,
      static_cast<unsigned long long>(metrics.taaHistoryBandwidthEstimateBytes),
      boolLogValue(metrics.taaResolvedSceneColorPublished),
      boolLogValue(metrics.taaDebugViewRendered),
      boolLogValue(metrics.taaHistoryValidityDebugViewRendered),
      boolLogValue(metrics.taaPixelInspectorDebugViewRendered),
      boolLogValue(metrics.taaStabilityDiagnosticsDebugViewRendered),
      boolLogValue(metrics.taaStabilityOwnershipDebugViewRendered),
      boolLogValue(metrics.taaPatchProbeDebugViewRendered),
      boolLogValue(metrics.taaMotionFilterDebugViewRendered),
      boolLogValue(metrics.taaOutOfBoundsFallbackEnabled),
      boolLogValue(metrics.taaBilinearHistorySampling),
      boolLogValue(metrics.taaDepthRejectionEnabled),
      metrics.taaDepthDiscontinuityThreshold,
      boolLogValue(metrics.taaPreviousDepthRejectionEnabled),
      boolLogValue(metrics.taaVelocityRejectionEnabled),
      boolLogValue(metrics.taaStaticFrameVelocitySanitizationEnabled),
      metrics.taaVelocityRejectionThreshold,
      boolLogValue(metrics.taaPreviousVelocityDisocclusionEnabled),
      boolLogValue(metrics.taaNeighborhoodClampEnabled),
      boolLogValue(metrics.taaAdaptiveBlendEnabled),
      metrics.taaVelocityBlendScale, metrics.taaDisocclusionCurrentWeight,
      boolLogValue(metrics.taaClampBlendAttenuationEnabled),
      metrics.taaClampBlendAttenuation,
      boolLogValue(metrics.taaNeighborhoodFallbackEnabled),
      temporalAAClampModeDisplayName(metrics.taaClampMode),
      metrics.taaVarianceGamma,
      temporalAAHdrWeightingModeDisplayName(metrics.taaHdrWeightingMode),
      metrics.taaHdrWeightStrength,
      boolLogValue(metrics.taaHdrWeightingEnabled));
  summary += std::format(
      " taaP2={{sharpenEnabled={} sharpenActive={} sharpenStrength={:.2f} "
      "sharpenConfidence={:.2f} mipBiasEnabled={} mipBiasApplied={} "
      "mipBias={:.2f} transparentSpatialEnabled={} "
      "transparentSpatialActive={} transparentSpatialPasses={}}}",
      boolLogValue(metrics.taaSharpenEnabled),
      boolLogValue(metrics.taaSharpenActive), metrics.taaSharpenStrength,
      metrics.taaSharpenConfidenceThreshold,
      boolLogValue(metrics.taaMaterialMipBiasEnabled),
      boolLogValue(metrics.taaMaterialMipBiasApplied),
      metrics.taaMaterialMipBias,
      boolLogValue(metrics.taaTransparentPostSpatialCleanupEnabled),
      boolLogValue(metrics.taaTransparentPostSpatialCleanupActive),
      metrics.taaTransparentPostSpatialAAPassCount);
  summary += std::format(
      " reactiveMask={{enabled={} draws={} alphaDraws={} "
      "motionUncertainDraws={} skippedTessellated={} reactiveCoverage={:.3f} "
      "alphaCoverage={:.3f} currentWeight={:.2f} strength={:.2f}}}",
      boolLogValue(metrics.taaReactiveMaskEnabled),
      metrics.reactiveMaskDrawCount, metrics.reactiveAlphaMaskedDrawCount,
      metrics.reactiveMotionUncertainDrawCount,
      metrics.reactiveSkippedTessellatedDrawCount,
      metrics.taaReactiveCoverageEstimate,
      metrics.taaAlphaMaskedCoverageEstimate, metrics.taaReactiveCurrentWeight,
      metrics.taaReactiveStrength);
  return summary;
}

std::string spatialAAMetricsSummary(const AntiAliasingFrameMetrics &metrics) {
  return std::format(
      "spatialAA={{enabled={} fallback={} cleanup={} dimensions={}x{} "
      "passes={} edge={} blend={} neighborhood={} copyBack={} debug={} "
      "gpuMs={:.3f} gpuTiming={} sourceFrame={} textureBytes={} "
      "lutBytes={} bandwidthBytes={} edgeEstimate={:.3f} "
      "modifiedEstimate={:.3f} edgesDebug={} blendDebug={} "
      "cleanupDebug={} splitDebug={} transparentPostEnabled={} "
      "transparentPostActive={} transparentPostPasses={}}}",
      boolLogValue(metrics.spatialAAEnabled),
      boolLogValue(metrics.spatialAAFallbackActive),
      boolLogValue(metrics.spatialAACleanupActive), metrics.spatialAAWidth,
      metrics.spatialAAHeight, metrics.spatialAAPassCount,
      metrics.spatialAAEdgePassCount, metrics.spatialAABlendPassCount,
      metrics.spatialAANeighborhoodPassCount,
      metrics.spatialAACopyBackPassCount, metrics.spatialAADebugPassCount,
      metrics.spatialAAGpuTimeMs, metrics.spatialAAGpuTimingAvailable,
      metrics.spatialAAGpuTimingSourceFrameIndex,
      static_cast<unsigned long long>(metrics.spatialAATotalBytes),
      static_cast<unsigned long long>(metrics.spatialAALutTextureBytes),
      static_cast<unsigned long long>(metrics.spatialAABandwidthEstimateBytes),
      metrics.spatialAAEdgePixelEstimate,
      metrics.spatialAAModifiedPixelEstimate,
      boolLogValue(metrics.spatialAAEdgesDebugViewRendered),
      boolLogValue(metrics.spatialAABlendWeightsDebugViewRendered),
      boolLogValue(metrics.spatialAACleanupMaskDebugViewRendered),
      boolLogValue(metrics.spatialAASplitCompareDebugViewRendered),
      boolLogValue(metrics.taaTransparentPostSpatialCleanupEnabled),
      boolLogValue(metrics.taaTransparentPostSpatialCleanupActive),
      metrics.taaTransparentPostSpatialAAPassCount);
}

std::string msaaMetricsSummary(const AntiAliasingFrameMetrics &metrics) {
  return std::format(
      "msaa={{enabled={} samples={} dimensions={}x{} colorAllocated={} "
      "depthAllocated={} colorGraph={} depthGraph={} colorResolve={} "
      "depthResolve={} resolvePasses={} alphaMaskedDraws={} "
      "alphaToCoverage={} sampleShading={} spatialCleanupEnabled={} "
      "spatialCleanupActive={} resolveGpuMs={:.3f} resolveTiming={} "
      "resolveTimingSourceFrame={} colorBytes={} depthBytes={} totalBytes={} "
      "bandwidthBytes={}}}",
      boolLogValue(metrics.msaaEnabled), metrics.msaaSampleCount,
      metrics.msaaWidth, metrics.msaaHeight,
      boolLogValue(metrics.msaaColorAllocated),
      boolLogValue(metrics.msaaDepthAllocated),
      boolLogValue(metrics.msaaColorGraphPublished),
      boolLogValue(metrics.msaaDepthGraphPublished),
      boolLogValue(metrics.msaaColorResolveTargetBound),
      boolLogValue(metrics.msaaDepthResolveTargetBound),
      metrics.msaaResolvePassCount, metrics.msaaAlphaMaskedDrawCount,
      boolLogValue(metrics.msaaAlphaToCoverageEnabled),
      boolLogValue(metrics.msaaSampleShadingEnabled),
      boolLogValue(metrics.msaaSpatialCleanupEnabled),
      boolLogValue(metrics.msaaSpatialCleanupActive),
      metrics.msaaResolveGpuTimeMs, metrics.msaaResolveGpuTimingAvailable,
      metrics.msaaResolveGpuTimingSourceFrameIndex,
      static_cast<unsigned long long>(metrics.msaaColorTextureBytes),
      static_cast<unsigned long long>(metrics.msaaDepthTextureBytes),
      static_cast<unsigned long long>(metrics.msaaTotalBytes),
      static_cast<unsigned long long>(
          metrics.msaaResolveBandwidthEstimateBytes));
}

std::string opaqueMetricsSummary(const OpaqueFrameMetrics &metrics) {
  return std::format(
      "opaque={{gpuMs={:.3f} gpuTiming={} sourceFrame={} totalInstances={} "
      "visibleInstances={} submittedDrawPackets={} submittedIndirectCalls={} "
      "submittedIndirectCommands={} "
      "computeDispatches={} computeDispatchX={} depthPrepass={} "
      "depthPrepassDraws={} tessDraws={} tessInstances={} overlays={}}}",
      metrics.gpuTimeMs, metrics.gpuTimingAvailable,
      metrics.gpuTimingSourceFrameIndex, metrics.totalInstances,
      metrics.visibleInstances, metrics.instancedDraws,
      metrics.indirectDrawCalls, metrics.indirectCommands,
      metrics.computeDispatches, metrics.computeDispatchX,
      metrics.depthPrepassEnabled, metrics.depthPrepassDraws,
      metrics.tessellatedDraws, metrics.tessellatedInstances,
      metrics.debugOverlayDraws);
}

std::string antiAliasingDiagnosticsSummary(
    const RenderSettings::AntiAliasingSettings &settings,
    const RenderFrameMetrics &frameMetrics, bool temporalFeaturePresent) {
  return std::format(
      "AA diagnostics: {} {} {} {} {}",
      antiAliasingSettingsSummary(settings, temporalFeaturePresent),
      antiAliasingMetricsSummary(frameMetrics.antiAliasing),
      spatialAAMetricsSummary(frameMetrics.antiAliasing),
      msaaMetricsSummary(frameMetrics.antiAliasing),
      opaqueMetricsSummary(frameMetrics.opaque));
}

void drawAntiAliasingSettings(RenderSettings::AntiAliasingSettings &aa,
                              RenderPipeline *renderPipeline,
                              const RenderFrameMetrics &frameMetrics) {
  sanitizeAntiAliasingSettings(aa);
  int modeIndex = static_cast<int>(aa.mode);
  modeIndex = std::clamp(modeIndex, 0,
                         static_cast<int>(kAntiAliasingModeLabels.size()) - 1);
  if (ImGui::Combo("Mode##AntiAliasing", &modeIndex,
                   kAntiAliasingModeLabels.data(),
                   static_cast<int>(kAntiAliasingModeLabels.size()))) {
    aa.mode = static_cast<AntiAliasingMode>(modeIndex);
    sanitizeAntiAliasingSettings(aa);
  }

  const bool temporalControlsDisabled =
      sanitizeAntiAliasingMode(aa.mode) != AntiAliasingMode::TAA;
  ImGui::BeginDisabled(temporalControlsDisabled);
  int presetIndex = static_cast<int>(aa.qualityPreset);
  presetIndex =
      std::clamp(presetIndex, 0,
                 static_cast<int>(kTemporalAAQualityPresetLabels.size()) - 1);
  if (ImGui::Combo("TAA Quality Preset##AntiAliasing", &presetIndex,
                   kTemporalAAQualityPresetLabels.data(),
                   static_cast<int>(kTemporalAAQualityPresetLabels.size()))) {
    aa.qualityPreset = static_cast<TemporalAAQualityPreset>(presetIndex);
    sanitizeAntiAliasingSettings(aa);
  }
  const bool customTemporalPreset =
      sanitizeTemporalAAQualityPreset(aa.qualityPreset) ==
      TemporalAAQualityPreset::Custom;
  ImGui::BeginDisabled(customTemporalPreset);
  if (ImGui::Button("Copy Preset to Custom##AntiAliasing")) {
    copyTemporalAAQualityPresetToCustom(aa);
  }
  ImGui::EndDisabled();
  ImGui::Checkbox("Jitter Enabled##AntiAliasing", &aa.debug.jitterEnabled);
  ImGui::Checkbox("Freeze Jitter##AntiAliasing", &aa.debug.freezeJitter);
  ImGui::Checkbox("Log TAA Diagnostics##AntiAliasing",
                  &aa.debug.logDiagnostics);
  ImGui::SliderFloat("TAA Log Interval##AntiAliasing",
                     &aa.debug.diagnosticLogIntervalSeconds, 0.033f, 5.0f,
                     "%.2f s");
  ImGui::EndDisabled();

  RenderSettings::AntiAliasingDebugSettings effectiveDebug =
      effectiveTemporalAADebugSettings(aa);
  RenderSettings::AntiAliasingDebugSettings presetPreview = effectiveDebug;
  RenderSettings::AntiAliasingDebugSettings *tuningDebug =
      customTemporalPreset ? &aa.debug : &presetPreview;
  ImGui::BeginDisabled(temporalControlsDisabled || !customTemporalPreset);
  ImGui::Checkbox("Post-TAA Spatial Cleanup##AntiAliasing",
                  &tuningDebug->spatialPostTaaCleanup);
  ImGui::Checkbox("Transparent Post-TAA Spatial Cleanup##AntiAliasing",
                  &tuningDebug->transparentPostTaaSpatialCleanup);
  ImGui::Checkbox("TAA Sharpen##AntiAliasing", &tuningDebug->taaSharpenEnabled);
  ImGui::Checkbox("TAA Material Mip Bias##AntiAliasing",
                  &tuningDebug->taaMaterialMipBiasEnabled);
  ImGui::SliderFloat("TAA Jitter Scale##AntiAliasing",
                     &tuningDebug->taaJitterScale, 0.0f, 1.0f, "%.2f");
  ImGui::SliderFloat("TAA Current Weight##AntiAliasing",
                     &tuningDebug->taaCurrentFrameWeight, 0.0f, 1.0f, "%.2f");
  ImGui::SliderFloat("TAA Sharpen Strength##AntiAliasing",
                     &tuningDebug->taaSharpenStrength, 0.0f, 1.0f, "%.2f");
  ImGui::SliderFloat("TAA Sharpen Confidence##AntiAliasing",
                     &tuningDebug->taaSharpenConfidenceThreshold, 0.0f, 1.0f,
                     "%.2f");
  ImGui::SliderFloat("TAA Material Mip Bias Value##AntiAliasing",
                     &tuningDebug->taaMaterialMipBias, -1.0f, 0.0f, "%.2f");
  ImGui::SliderFloat("TAA Motion Current Weight##AntiAliasing",
                     &tuningDebug->taaMotionCurrentWeight, 0.0f, 1.0f, "%.2f");
  ImGui::SliderFloat("TAA Clamp Current Weight##AntiAliasing",
                     &tuningDebug->taaClampCurrentWeight, 0.0f, 1.0f, "%.2f");
  int historyFilterModeIndex =
      static_cast<int>(tuningDebug->taaHistoryFilterMode);
  historyFilterModeIndex = std::clamp(
      historyFilterModeIndex, 0,
      static_cast<int>(kTemporalAAHistoryFilterModeLabels.size()) - 1);
  if (ImGui::Combo(
          "TAA History Filter##AntiAliasing", &historyFilterModeIndex,
          kTemporalAAHistoryFilterModeLabels.data(),
          static_cast<int>(kTemporalAAHistoryFilterModeLabels.size()))) {
    tuningDebug->taaHistoryFilterMode =
        static_cast<TemporalAAHistoryFilterMode>(historyFilterModeIndex);
  }
  ImGui::SliderFloat("TAA Depth Reject##AntiAliasing",
                     &tuningDebug->taaDepthDiscontinuityThreshold, 0.0f, 0.1f,
                     "%.4f");
  ImGui::SliderFloat("TAA Velocity Reject (px)##AntiAliasing",
                     &tuningDebug->taaVelocityRejectionThreshold, 0.0f, 16.0f,
                     "%.2f");
  ImGui::SliderFloat("TAA Motion Blend / px##AntiAliasing",
                     &tuningDebug->taaVelocityBlendScale, 0.0f, 4.0f, "%.2f");
  ImGui::SliderFloat("TAA Disocclusion Weight##AntiAliasing",
                     &tuningDebug->taaDisocclusionCurrentWeight, 0.0f, 1.0f,
                     "%.2f");
  ImGui::SliderFloat("TAA Clamp Attenuation##AntiAliasing",
                     &tuningDebug->taaClampBlendAttenuation, 0.0f, 1.0f,
                     "%.2f");
  int clampModeIndex = static_cast<int>(tuningDebug->taaClampMode);
  clampModeIndex =
      std::clamp(clampModeIndex, 0,
                 static_cast<int>(kTemporalAAClampModeLabels.size()) - 1);
  if (ImGui::Combo("TAA Clamp Mode##AntiAliasing", &clampModeIndex,
                   kTemporalAAClampModeLabels.data(),
                   static_cast<int>(kTemporalAAClampModeLabels.size()))) {
    tuningDebug->taaClampMode =
        static_cast<TemporalAAClampMode>(clampModeIndex);
  }
  ImGui::SliderFloat("TAA Variance Gamma##AntiAliasing",
                     &tuningDebug->taaVarianceGamma, 0.0f, 4.0f, "%.2f");
  int hdrWeightingModeIndex =
      static_cast<int>(tuningDebug->taaHdrWeightingMode);
  hdrWeightingModeIndex = std::clamp(
      hdrWeightingModeIndex, 0,
      static_cast<int>(kTemporalAAHdrWeightingModeLabels.size()) - 1);
  if (ImGui::Combo(
          "TAA HDR Weighting##AntiAliasing", &hdrWeightingModeIndex,
          kTemporalAAHdrWeightingModeLabels.data(),
          static_cast<int>(kTemporalAAHdrWeightingModeLabels.size()))) {
    tuningDebug->taaHdrWeightingMode =
        static_cast<TemporalAAHdrWeightingMode>(hdrWeightingModeIndex);
  }
  ImGui::SliderFloat("TAA HDR Weight Strength##AntiAliasing",
                     &tuningDebug->taaHdrWeightStrength, 0.0f, 1.0f, "%.2f");
  ImGui::SliderFloat("TAA Reactive Current Weight##AntiAliasing",
                     &tuningDebug->taaReactiveCurrentWeight, 0.0f, 1.0f,
                     "%.2f");
  ImGui::SliderFloat("TAA Reactive Strength##AntiAliasing",
                     &tuningDebug->taaReactiveStrength, 0.0f, 4.0f, "%.2f");
  int dilationModeIndex =
      static_cast<int>(tuningDebug->taaVelocityDilationMode);
  dilationModeIndex = std::clamp(
      dilationModeIndex, 0,
      static_cast<int>(kTemporalAAVelocityDilationModeLabels.size()) - 1);
  if (ImGui::Combo(
          "TAA Velocity Dilation##AntiAliasing", &dilationModeIndex,
          kTemporalAAVelocityDilationModeLabels.data(),
          static_cast<int>(kTemporalAAVelocityDilationModeLabels.size()))) {
    tuningDebug->taaVelocityDilationMode =
        static_cast<TemporalAAVelocityDilationMode>(dilationModeIndex);
  }
  ImGui::SliderFloat("TAA Dilation Depth Threshold##AntiAliasing",
                     &tuningDebug->taaVelocityDilationDepthThreshold, 0.0f,
                     0.1f, "%.4f");
  ImGui::EndDisabled();
  if (temporalControlsDisabled) {
    aa.debug.jitterEnabled = false;
    aa.debug.freezeJitter = false;
  }
  const bool msaaControlsDisabled =
      sanitizeAntiAliasingMode(aa.mode) != AntiAliasingMode::MSAA4x;
  ImGui::BeginDisabled(msaaControlsDisabled);
  ImGui::Checkbox("Post-MSAA Spatial Cleanup##AntiAliasing",
                  &aa.debug.spatialPostMsaaCleanup);
  ImGui::EndDisabled();

  if (ImGui::Button("Reset History##AntiAliasing")) {
    aa.debug.resetHistoryRequested = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Clear Reset##AntiAliasing")) {
    aa.debug.resetHistoryRequested = false;
  }

  int debugViewIndex = static_cast<int>(aa.debug.view);
  debugViewIndex =
      std::clamp(debugViewIndex, 0,
                 static_cast<int>(kAntiAliasingDebugViewLabels.size()) - 1);
  if (ImGui::Combo("Debug View##AntiAliasing", &debugViewIndex,
                   kAntiAliasingDebugViewLabels.data(),
                   static_cast<int>(kAntiAliasingDebugViewLabels.size()))) {
    aa.debug.view = static_cast<AntiAliasingDebugView>(debugViewIndex);
  }
  sanitizeAntiAliasingSettings(aa);

  const bool temporalFeaturePresent = hasTemporalAAFeature(renderPipeline);
  ImGui::Separator();
  ImGui::Text("Active Mode: %s", antiAliasingModeDisplayName(aa.mode));
  ImGui::Text("TAA Preset: %s",
              temporalAAQualityPresetDisplayName(aa.qualityPreset));
  ImGui::Text("Jitter: %s", aa.debug.jitterEnabled ? "enabled" : "disabled");
  ImGui::Text("Freeze Jitter: %s", aa.debug.freezeJitter ? "yes" : "no");
  ImGui::Text("Interval Log: %s every %.2f s",
              aa.debug.logDiagnostics ? "enabled" : "disabled",
              aa.debug.diagnosticLogIntervalSeconds);
  if (aa.debug.jitterEnabled && aa.debug.freezeJitter) {
    ImGui::TextColored(
        ImVec4(1.0f, 0.72f, 0.22f, 1.0f),
        "TAA quality validation is invalid while jitter is frozen");
  }
  ImGui::Text("Debug View: %s",
              antiAliasingDebugViewDisplayName(aa.debug.view));
  ImGui::Text("History Reset Request: %s",
              aa.debug.resetHistoryRequested ? "pending" : "clear");
  ImGui::Text("TemporalAAFeature: %s",
              temporalFeaturePresent ? "registered" : "missing");
  const AntiAliasingFrameMetrics &metrics = frameMetrics.antiAliasing;
  if (ImGui::CollapsingHeader("TAA Capture HUD",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    const std::string_view hudResetReason =
        temporalHistoryResetReasonName(metrics.historyResetReason);
    ImGui::Text("Mode: %s", antiAliasingModeDisplayName(aa.mode));
    ImGui::Text("Preset: %s",
                temporalAAQualityPresetDisplayName(metrics.taaQualityPreset));
    ImGui::Text("View: %s", antiAliasingDebugViewDisplayName(aa.debug.view));
    ImGui::Text("Jitter: %u / %u  History: %s", metrics.jitterIndex,
                metrics.jitterSequenceLength,
                metrics.historyValid ? "valid" : "invalid");
    ImGui::Text("Reset: %.*s  Frames: %u",
                static_cast<int>(hudResetReason.size()), hudResetReason.data(),
                metrics.framesSinceHistoryReset);
    ImGui::Text("Resolve GPU: %.3f ms (%s)", metrics.taaResolveGpuTimeMs,
                metrics.taaResolveGpuTimingAvailable != 0u ? "ready"
                                                           : "pending");
    ImGui::Text("Debug GPU: %.3f ms (%s)", metrics.taaDebugGpuTimeMs,
                metrics.taaDebugGpuTimingAvailable != 0u ? "ready" : "pending");
    ImGui::Text(
        "Opaque GPU: %.3f ms (%s, frame %llu)", frameMetrics.opaque.gpuTimeMs,
        frameMetrics.opaque.gpuTimingAvailable != 0u ? "ready" : "pending",
        static_cast<unsigned long long>(
            frameMetrics.opaque.gpuTimingSourceFrameIndex));
    ImGui::Text("Spatial GPU: %.3f ms (%s)", metrics.spatialAAGpuTimeMs,
                metrics.spatialAAGpuTimingAvailable != 0u ? "ready"
                                                          : "pending");
    ImGui::Text("MSAA Resolve GPU: %.3f ms (%s)", metrics.msaaResolveGpuTimeMs,
                metrics.msaaResolveGpuTimingAvailable != 0u ? "ready"
                                                            : "pending");
    const AAMemoryAndBandwidthMetrics aaMetrics = ComputeAAMetrics(metrics);
    ImGui::Text(
        "AA Memory: %.2f MiB  Bandwidth Est: %.2f MiB",
        static_cast<double>(aaMetrics.residentBytes) / (1024.0 * 1024.0),
        static_cast<double>(aaMetrics.bandwidthBytes) / (1024.0 * 1024.0));
    ImGui::Text("Opaque Work: %u/%u visible, draws %u, indirect %u/%u",
                frameMetrics.opaque.visibleInstances,
                frameMetrics.opaque.totalInstances,
                frameMetrics.opaque.instancedDraws,
                frameMetrics.opaque.indirectDrawCalls,
                frameMetrics.opaque.indirectCommands);
    ImGui::Text("Opaque Compute: dispatches %u x%u, depth prepass %s (%u)",
                frameMetrics.opaque.computeDispatches,
                frameMetrics.opaque.computeDispatchX,
                frameMetrics.opaque.depthPrepassEnabled != 0u ? "on" : "off",
                frameMetrics.opaque.depthPrepassDraws);
    ImGui::Text(
        "Rejection: %.3f  Filter: %s", metrics.taaDisocclusionRejectionEstimate,
        temporalAAHistoryFilterModeDisplayName(metrics.taaHistoryFilterMode));
  }
  ImGui::Text("Jitter Index: %u / %u", metrics.jitterIndex,
              metrics.jitterSequenceLength);
  ImGui::Text("Jitter Offset: %.4f, %.4f px", metrics.jitterPixelOffset.x,
              metrics.jitterPixelOffset.y);
  ImGui::Text("Jitter Delta: %.4f, %.4f px (%.4f)",
              metrics.jitterDeltaPixelOffset.x,
              metrics.jitterDeltaPixelOffset.y, metrics.jitterDeltaMagnitude);
  ImGui::Text("Jitter Bounds: %s",
              metrics.jitterOutOfBounds ? "out of range" : "valid");
  ImGui::Text("Camera Delta: %.6f", metrics.cameraPositionDelta);
  ImGui::Text("History Valid: %s", metrics.historyValid ? "yes" : "no");
  const std::string_view resetReason =
      temporalHistoryResetReasonName(metrics.historyResetReason);
  ImGui::TextUnformatted("Reset Reason: ");
  ImGui::SameLine();
  ImGui::TextUnformatted(resetReason.data(),
                         resetReason.data() + resetReason.size());
  ImGui::Text("Frames Since Reset: %u", metrics.framesSinceHistoryReset);
  ImGui::Text("Reset Count: %u", metrics.historyResetCount);
  ImGui::Text("Jitter OOB Count: %u", metrics.jitterOutOfBoundsCount);
  if (ImGui::CollapsingHeader("Spatial AA", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Enabled: %s", metrics.spatialAAEnabled ? "yes" : "no");
    ImGui::Text("Fallback: %s", metrics.spatialAAFallbackActive ? "yes" : "no");
    ImGui::Text("Cleanup: %s", metrics.spatialAACleanupActive ? "yes" : "no");
    ImGui::Text("Dimensions: %u x %u", metrics.spatialAAWidth,
                metrics.spatialAAHeight);
    ImGui::Text("Passes: %u (edge %u, blend %u, neighborhood %u, copy %u)",
                metrics.spatialAAPassCount, metrics.spatialAAEdgePassCount,
                metrics.spatialAABlendPassCount,
                metrics.spatialAANeighborhoodPassCount,
                metrics.spatialAACopyBackPassCount);
    ImGui::Text("GPU Time: %.3f ms (%s, frame %llu)",
                metrics.spatialAAGpuTimeMs,
                metrics.spatialAAGpuTimingAvailable != 0u ? "ready" : "pending",
                static_cast<unsigned long long>(
                    metrics.spatialAAGpuTimingSourceFrameIndex));
    ImGui::Text(
        "Texture Bytes: %llu  LUT Bytes: %llu",
        static_cast<unsigned long long>(metrics.spatialAATotalBytes),
        static_cast<unsigned long long>(metrics.spatialAALutTextureBytes));
  }
  if (ImGui::CollapsingHeader("MSAA", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Enabled: %s", metrics.msaaEnabled ? "yes" : "no");
    ImGui::Text("Samples: %u", metrics.msaaSampleCount);
    ImGui::Text("Dimensions: %u x %u", metrics.msaaWidth, metrics.msaaHeight);
    ImGui::Text("Scene Targets: color %s, depth %s",
                metrics.msaaColorAllocated ? "allocated" : "missing",
                metrics.msaaDepthAllocated ? "allocated" : "missing");
    ImGui::Text("Graph Targets: color %s, depth %s",
                metrics.msaaColorGraphPublished ? "published" : "missing",
                metrics.msaaDepthGraphPublished ? "published" : "missing");
    ImGui::Text("Resolve Targets: color %s, depth %s",
                metrics.msaaColorResolveTargetBound ? "bound" : "missing",
                metrics.msaaDepthResolveTargetBound ? "bound" : "missing");
    ImGui::Text("Resolve Passes: %u", metrics.msaaResolvePassCount);
    ImGui::Text("Standalone Resolve GPU Time: %.3f ms (%s, frame %llu)",
                metrics.msaaResolveGpuTimeMs,
                metrics.msaaResolveGpuTimingAvailable != 0u ? "available"
                                                            : "pending",
                static_cast<unsigned long long>(
                    metrics.msaaResolveGpuTimingSourceFrameIndex));
    ImGui::Text("Alpha-Masked Draws: %u", metrics.msaaAlphaMaskedDrawCount);
    ImGui::Text("Alpha-to-Coverage: %s",
                metrics.msaaAlphaToCoverageEnabled ? "enabled" : "inactive");
    ImGui::Text("Sample Shading: %s",
                metrics.msaaSampleShadingEnabled ? "enabled" : "inactive");
    ImGui::Text("Spatial Cleanup: %s (%s)",
                metrics.msaaSpatialCleanupEnabled ? "enabled" : "disabled",
                metrics.msaaSpatialCleanupActive ? "active" : "inactive");
    ImGui::Text("Texture Bytes: color %llu, depth %llu, total %llu",
                static_cast<unsigned long long>(metrics.msaaColorTextureBytes),
                static_cast<unsigned long long>(metrics.msaaDepthTextureBytes),
                static_cast<unsigned long long>(metrics.msaaTotalBytes));
  }
  if (ImGui::CollapsingHeader("Motion Vector Resource",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Allocated: %s", metrics.motionVectorAllocated ? "yes" : "no");
    ImGui::Text("Format: %s", formatDisplayName(metrics.motionVectorFormat));
    ImGui::Text("Format Support: %s", metrics.motionVectorFormatSupported
                                          ? "supported"
                                          : "unsupported");
    ImGui::Text("Dimensions: %u x %u", metrics.motionVectorWidth,
                metrics.motionVectorHeight);
    ImGui::Text("Texture Ring Count: %u", metrics.motionVectorTextureCount);
    ImGui::Text(
        "Current Texture Bytes: %llu",
        static_cast<unsigned long long>(metrics.motionVectorTextureBytes));
    ImGui::Text("Previous Texture Bytes: %llu",
                static_cast<unsigned long long>(
                    metrics.previousMotionVectorTextureBytes));
    ImGui::Text(
        "Total Velocity Bytes: %llu",
        static_cast<unsigned long long>(metrics.motionVectorTotalBytes));
    ImGui::Text("Clear Value: %.3f, %.3f, %.3f, %.3f",
                kFrameCompositionMotionVectorClearValue.r,
                kFrameCompositionMotionVectorClearValue.g,
                kFrameCompositionMotionVectorClearValue.b,
                kFrameCompositionMotionVectorClearValue.a);
    ImGui::Text("Clear Passes: %u", metrics.motionVectorClearPassCount);
    ImGui::Text(
        "Clear Bandwidth Estimate: %llu bytes",
        static_cast<unsigned long long>(metrics.motionVectorClearBytes));
    ImGui::Text("Allocation Count: %u", metrics.motionVectorAllocationCount);
    ImGui::Text("Reallocation Count: %u",
                metrics.motionVectorReallocationCount);
    ImGui::Text("RG32 Fallback Count: %u",
                metrics.motionVectorRg32FallbackCount);
    ImGui::Text("Current Graph Resource: %s",
                metrics.motionVectorGraphPublished ? "published" : "missing");
    ImGui::Text("Previous Velocity: %s",
                metrics.previousMotionVectorValid ? "valid" : "invalid");
    ImGui::Text("Previous Graph Resource: %s",
                metrics.previousMotionVectorGraphPublished ? "published"
                                                           : "not imported");
    ImGui::Text("Opaque Velocity Passes: %u", metrics.velocityPassCount);
    ImGui::Text("Opaque Velocity Draws: %u", metrics.velocityDrawCount);
    ImGui::Text("Velocity Debug Passes: %u", metrics.velocityDebugPassCount);
    ImGui::Text("Velocity Instances: %u", metrics.velocityInstanceCount);
    ImGui::Text("Previous Transform Cache: %s",
                metrics.previousTransformCacheValid ? "valid" : "invalid");
    ImGui::Text("Previous Transforms: %u valid, %u missing",
                metrics.velocityPreviousTransformValidCount,
                metrics.velocityMissingPreviousTransformCount);
    ImGui::Text("Animated/Responsive Instances: %u",
                metrics.velocityAnimatedResponsiveCount);
    ImGui::Text("Animated Previous Geometry: %u",
                metrics.velocityAnimatedPreviousGeometryCount);
    ImGui::Text("Skipped Tessellated Draws: %u",
                metrics.velocityTessellatedSkippedDrawCount);
    ImGui::Text("Object Motion: avg %.5f, max %.5f",
                metrics.velocityAverageObjectMotion,
                metrics.velocityMaxObjectMotion);
    ImGui::Text("Estimated Velocity: avg %.5f, max %.5f",
                metrics.velocityEstimatedAverageMagnitude,
                metrics.velocityEstimatedMaxMagnitude);
    ImGui::Text("Static Residual Estimate: %.6f",
                metrics.velocityStaticResidualEstimate);
    ImGui::Text("Camera Matrix Delta: %.6f", metrics.velocityCameraMatrixDelta);
    ImGui::Text("Missing Previous Ratio: %.3f",
                metrics.velocityMissingPreviousRatio);
    ImGui::Text("Edge Discontinuity Estimate: %.3f",
                metrics.velocityEdgeDiscontinuityEstimate);
    ImGui::Text("Velocity Pass Bandwidth Estimate: %llu bytes",
                static_cast<unsigned long long>(
                    metrics.velocityPassBandwidthEstimateBytes));
    ImGui::Text("Velocity Debug Bandwidth Estimate: %llu bytes",
                static_cast<unsigned long long>(
                    metrics.velocityDebugBandwidthEstimateBytes));
    ImGui::Text("Velocity Debug View: %s",
                metrics.velocityDebugViewRendered ? "rendered" : "inactive");
    ImGui::TextUnformatted("Producer: Opaque velocity pass or clear fallback");
    ImGui::TextUnformatted("Consumer: future TAA resolve");
  }
  if (ImGui::CollapsingHeader("Reactive Mask Resource",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Allocated: %s", metrics.reactiveMaskAllocated ? "yes" : "no");
    ImGui::Text("Format: %s",
                formatDisplayName(kFrameCompositionReactiveMaskFormat));
    ImGui::Text("Format Support: %s", metrics.reactiveMaskFormatSupported
                                          ? "supported"
                                          : "unsupported");
    ImGui::Text("Dimensions: %u x %u", metrics.reactiveMaskWidth,
                metrics.reactiveMaskHeight);
    ImGui::Text("Texture Ring Count: %u", metrics.reactiveMaskTextureCount);
    ImGui::Text(
        "Current Texture Bytes: %llu",
        static_cast<unsigned long long>(metrics.reactiveMaskTextureBytes));
    ImGui::Text(
        "Total Reactive Bytes: %llu",
        static_cast<unsigned long long>(metrics.reactiveMaskTotalBytes));
    ImGui::Text("Graph Resource: %s",
                metrics.reactiveMaskGraphPublished ? "published" : "missing");
    ImGui::Text("Reactive Passes: %u", metrics.reactiveMaskPassCount);
    ImGui::Text("Reactive Draws: %u", metrics.reactiveMaskDrawCount);
    ImGui::Text("Alpha-Masked Draws: %u", metrics.reactiveAlphaMaskedDrawCount);
    ImGui::Text("Motion-Uncertain Draws: %u",
                metrics.reactiveMotionUncertainDrawCount);
    ImGui::Text("Skipped Tessellated Reactive Draws: %u",
                metrics.reactiveSkippedTessellatedDrawCount);
    ImGui::Text("Allocation Count: %u", metrics.reactiveMaskAllocationCount);
    ImGui::Text("Reallocation Count: %u",
                metrics.reactiveMaskReallocationCount);
    ImGui::Text("Reactive Bandwidth Estimate: %llu bytes",
                static_cast<unsigned long long>(
                    metrics.reactiveMaskPassBandwidthEstimateBytes));
  }
  if (ImGui::CollapsingHeader("TAA Resolve", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Resolve Passes: %u", metrics.taaResolvePassCount);
    ImGui::Text("Copy Back Passes: %u", metrics.taaCopyBackPassCount);
    ImGui::Text("Post-TAA Mip Passes: %u (%s)",
                metrics.taaPostResolveSceneColorMipPassCount,
                metrics.taaPostResolveSceneColorMipChainGenerated
                    ? "generated"
                    : "not generated");
    ImGui::Text("TAA Resolve GPU Time: %.3f ms (%s, frame %llu)",
                metrics.taaResolveGpuTimeMs,
                metrics.taaResolveGpuTimingAvailable != 0u ? "available"
                                                           : "pending",
                static_cast<unsigned long long>(
                    metrics.taaResolveGpuTimingSourceFrameIndex));
    ImGui::Text("TAA Debug GPU Time: %.3f ms (%s, frame %llu)",
                metrics.taaDebugGpuTimeMs,
                metrics.taaDebugGpuTimingAvailable != 0u ? "available"
                                                         : "pending",
                static_cast<unsigned long long>(
                    metrics.taaDebugGpuTimingSourceFrameIndex));
    ImGui::Text("Transmission Source: %s, stale frames %u",
                metrics.taaTransmissionPostResolveSceneColorConsumed
                    ? "post-TAA"
                    : "not confirmed",
                metrics.taaTransmissionStaleSceneColorFrameCount);
    ImGui::Text("Scene Mip Debug View: %s",
                metrics.taaSceneColorMipDebugViewRendered ? "rendered"
                                                          : "inactive");
    ImGui::Text("Downsample GPU Time: %.3f ms (%s, frame %llu)",
                metrics.taaSceneColorDownsampleGpuTimeMs,
                metrics.taaSceneColorDownsampleGpuTimingAvailable != 0u
                    ? "available"
                    : "pending",
                static_cast<unsigned long long>(
                    metrics.taaSceneColorDownsampleGpuTimingSourceFrameIndex));
    ImGui::Text("Transmission GPU Time: %.3f ms (%s, frame %llu)",
                metrics.taaTransmissionGpuTimeMs,
                metrics.taaTransmissionGpuTimingAvailable != 0u ? "available"
                                                                : "pending",
                static_cast<unsigned long long>(
                    metrics.taaTransmissionGpuTimingSourceFrameIndex));
    ImGui::Text("Transmission Mip Debug: %s (%u passes)",
                metrics.taaTransmissionMipDebugViewRendered ? "rendered"
                                                            : "inactive",
                metrics.taaTransmissionMipDebugPassCount);
    ImGui::Text("Transmission Flicker Estimate: %.2f",
                metrics.taaTransmissionFlickerEstimate);
    ImGui::Text("Transmission Jitter Min LOD: %.2f",
                metrics.taaTransmissionJitterMinLod);
    ImGui::Text("Transmission Depth Bias: %.1f",
                metrics.taaTransmissionDepthBiasConstant);
    ImGui::Text("Transmission Stable Depth: %s",
                metrics.taaTransmissionStableVisibilityDepth ? "active"
                                                             : "inactive");
    ImGui::Text("Transparent Edge Jitter: %.2f (%u post-TAA draws)",
                metrics.taaTransparentEdgeJitterEstimate,
                metrics.taaTransparentPostTaaDrawCount);
    ImGui::Text(
        "Transparent Post-TAA Split: mesh %u, contributors %u, fixed %u",
        metrics.taaTransparentPostTaaMeshDrawCount,
        metrics.taaTransparentPostTaaContributorDrawCount,
        metrics.taaTransparentPostTaaFixedDrawCount);
    ImGui::Text("Resolve Dimensions: %u x %u", metrics.taaResolveWidth,
                metrics.taaResolveHeight);
    ImGui::Text("Jitter Scale: %.2f", metrics.taaJitterScale);
    ImGui::Text("Blend Weights: current %.2f, history %.2f",
                metrics.taaCurrentFrameWeight, metrics.taaHistoryFrameWeight);
    ImGui::Text("TAA Sharpen: %s/%s strength %.2f confidence %.2f",
                metrics.taaSharpenEnabled ? "enabled" : "off",
                metrics.taaSharpenActive ? "active" : "inactive",
                metrics.taaSharpenStrength,
                metrics.taaSharpenConfidenceThreshold);
    ImGui::Text("TAA Material Mip Bias: %s/%s %.2f",
                metrics.taaMaterialMipBiasEnabled ? "enabled" : "off",
                metrics.taaMaterialMipBiasApplied ? "applied" : "inactive",
                metrics.taaMaterialMipBias);
    ImGui::Text(
        "Transparent Spatial Cleanup: %s/%s (%u passes)",
        metrics.taaTransparentPostSpatialCleanupEnabled ? "enabled" : "off",
        metrics.taaTransparentPostSpatialCleanupActive ? "active" : "inactive",
        metrics.taaTransparentPostSpatialAAPassCount);
    ImGui::Text("Motion/Clamp Targets: motion %.2f, clamp %.2f",
                metrics.taaMotionCurrentWeight, metrics.taaClampCurrentWeight);
    ImGui::Text("History Texture Valid: %s",
                metrics.historyValid ? "yes" : "no");
    ImGui::Text("Previous Depth: %s graph %s (%llu bytes)",
                metrics.previousSceneDepthValid ? "valid" : "invalid",
                metrics.previousSceneDepthGraphPublished ? "published"
                                                         : "not published",
                static_cast<unsigned long long>(
                    metrics.previousSceneDepthTextureBytes));
    ImGui::TextUnformatted("OOB Reprojection: not measured");
    ImGui::Text("Current Fallback Frames: %u",
                metrics.taaCurrentFallbackFrameCount);
    ImGui::Text("History Bandwidth Estimate: %llu bytes",
                static_cast<unsigned long long>(
                    metrics.taaHistoryBandwidthEstimateBytes));
    ImGui::Text("Resolved Scene Color: %s",
                metrics.taaResolvedSceneColorPublished ? "published"
                                                       : "not published");
    ImGui::Text("TAA Debug View: %s",
                metrics.taaDebugViewRendered ? "rendered" : "inactive");
    ImGui::Text("History Validity View: %s",
                metrics.taaHistoryValidityDebugViewRendered ? "rendered"
                                                            : "inactive");
    ImGui::Text("OOB Fallback: %s",
                metrics.taaOutOfBoundsFallbackEnabled ? "enabled" : "off");
    ImGui::Text("History Sampling: %s", temporalAAHistoryFilterModeDisplayName(
                                            metrics.taaHistoryFilterMode));
    ImGui::Text("Depth Rejection: %s threshold %.4f",
                metrics.taaDepthRejectionEnabled ? "enabled" : "off",
                metrics.taaDepthDiscontinuityThreshold);
    ImGui::Text("Previous Depth Rejection: %s",
                metrics.taaPreviousDepthRejectionEnabled ? "enabled" : "off");
    ImGui::Text("Velocity Rejection: %s threshold %.2f px",
                metrics.taaVelocityRejectionEnabled ? "enabled" : "off",
                metrics.taaVelocityRejectionThreshold);
    ImGui::Text("Previous Velocity Disocclusion: %s",
                metrics.taaPreviousVelocityDisocclusionEnabled ? "enabled"
                                                               : "off");
    ImGui::Text("Neighborhood Clamp: %s",
                metrics.taaNeighborhoodClampEnabled ? "enabled" : "off");
    ImGui::Text(
        "Adaptive Blend: %s motion/px %.2f motion %.2f disocclusion %.2f",
        metrics.taaAdaptiveBlendEnabled ? "enabled" : "off",
        metrics.taaVelocityBlendScale, metrics.taaMotionCurrentWeight,
        metrics.taaDisocclusionCurrentWeight);
    ImGui::Text("Clamp Attenuation: %s %.2f target %.2f",
                metrics.taaClampBlendAttenuationEnabled ? "enabled" : "off",
                metrics.taaClampBlendAttenuation,
                metrics.taaClampCurrentWeight);
    ImGui::Text("Neighborhood Fallback: %s",
                metrics.taaNeighborhoodFallbackEnabled ? "enabled" : "off");
    ImGui::Text("Clamp Mode: %s variance gamma %.2f",
                temporalAAClampModeDisplayName(metrics.taaClampMode),
                metrics.taaVarianceGamma);
    ImGui::Text(
        "HDR Weighting: %s strength %.2f (%s)",
        temporalAAHdrWeightingModeDisplayName(metrics.taaHdrWeightingMode),
        metrics.taaHdrWeightStrength,
        metrics.taaHdrWeightingEnabled ? "active" : "inactive");
    ImGui::Text("Reactive Mask: %s target %.2f strength %.2f coverage %.3f",
                metrics.taaReactiveMaskEnabled ? "enabled" : "off",
                metrics.taaReactiveCurrentWeight, metrics.taaReactiveStrength,
                metrics.taaReactiveCoverageEstimate);
    ImGui::Text("Velocity Dilation: %s threshold %.4f affected %.3f",
                temporalAAVelocityDilationModeDisplayName(
                    metrics.taaVelocityDilationMode),
                metrics.taaVelocityDilationDepthThreshold,
                metrics.taaVelocityDilationAffectedEstimate);
    ImGui::Text("Static Velocity Sanitization: %s",
                metrics.taaStaticFrameVelocitySanitizationEnabled ? "enabled"
                                                                  : "off");
    ImGui::Text("Disocclusion Estimate: %.3f",
                metrics.taaDisocclusionRejectionEstimate);
    ImGui::Text("Pixel Inspector View: %s",
                metrics.taaPixelInspectorDebugViewRendered ? "rendered"
                                                           : "inactive");
    ImGui::Text("Reactive Mask View: %s",
                metrics.taaReactiveMaskDebugViewRendered ? "rendered"
                                                         : "inactive");
    ImGui::Text("Disocclusion Mask View: %s",
                metrics.taaDisocclusionMaskDebugViewRendered ? "rendered"
                                                             : "inactive");
    ImGui::Text("Velocity Dilation View: %s",
                metrics.taaVelocityDilationDebugViewRendered ? "rendered"
                                                             : "inactive");
    ImGui::Text("Previous Velocity View: %s",
                metrics.taaPreviousVelocityDebugViewRendered ? "rendered"
                                                             : "inactive");
    ImGui::Text("HDR Weight View: %s", metrics.taaHdrWeightDebugViewRendered
                                           ? "rendered"
                                           : "inactive");
    ImGui::Text("History Filter Delta View: %s",
                metrics.taaHistoryFilterDeltaDebugViewRendered ? "rendered"
                                                               : "inactive");
    ImGui::Text("Disocclusion Fallback View: %s",
                metrics.taaDisocclusionFallbackDebugViewRendered ? "rendered"
                                                                 : "inactive");
    ImGui::Text("Split Compare View: %s",
                metrics.taaSplitCompareDebugViewRendered ? "rendered"
                                                         : "inactive");
    ImGui::Text("Temporal Confidence View: %s",
                metrics.taaTemporalConfidenceDebugViewRendered ? "rendered"
                                                               : "inactive");
    ImGui::Text("Previous Depth Rejection View: %s",
                metrics.taaPreviousDepthRejectionDebugViewRendered
                    ? "rendered"
                    : "inactive");
    ImGui::Text("Stability Diagnostics View: %s",
                metrics.taaStabilityDiagnosticsDebugViewRendered ? "rendered"
                                                                 : "inactive");
    ImGui::Text("Stability Ownership View: %s",
                metrics.taaStabilityOwnershipDebugViewRendered ? "rendered"
                                                               : "inactive");
    ImGui::Text("Patch Probe View: %s", metrics.taaPatchProbeDebugViewRendered
                                            ? "rendered"
                                            : "inactive");
    ImGui::Text("Motion Filter View: %s",
                metrics.taaMotionFilterDebugViewRendered ? "rendered"
                                                         : "inactive");
  }
  if (ImGui::Button("Dump AA Diagnostics To Log##AntiAliasing")) {
    const std::string summary = antiAliasingDiagnosticsSummary(
        aa, frameMetrics, temporalFeaturePresent);
    NURI_LOG_INFO("%s", summary.c_str());
  }
}

void drawAntiAliasingWindow(bool &open, RenderSettings &renderSettings,
                            RenderPipeline *renderPipeline,
                            const RenderFrameMetrics &frameMetrics) {
  if (!ImGui::Begin(kAntiAliasingWindowName, &open)) {
    ImGui::End();
    return;
  }
  drawAntiAliasingSettings(renderSettings.antiAliasing, renderPipeline,
                           frameMetrics);
  ImGui::End();
}

void drawAmbientOcclusionWindow(bool &open, RenderSettings &renderSettings,
                                const RenderFrameMetrics &frameMetrics) {
  if (!ImGui::Begin(kAmbientOcclusionWindowName, &open)) {
    ImGui::End();
    return;
  }
  drawAmbientOcclusionSettings(renderSettings.ambientOcclusion,
                               renderSettings.opaque,
                               renderSettings.antiAliasing, frameMetrics);
  ImGui::End();
}

void drawHDRPostProcessWindow(bool &open, RenderSettings &renderSettings,
                              const RenderFrameMetrics &frameMetrics) {
  if (!ImGui::Begin(kHDRPostProcessWindowName, &open)) {
    ImGui::End();
    return;
  }
  drawHDRPostProcessSettings(renderSettings.hdrPostProcess, frameMetrics);
  ImGui::End();
}

void drawTextureFilteringWindow(bool &open, RenderSettings &renderSettings,
                                const GPUDevice &gpu) {
  if (!ImGui::Begin(kTextureFilteringWindowName, &open)) {
    ImGui::End();
    return;
  }

  auto &settings = renderSettings.textureFiltering;
  sanitizeTextureFilteringSettings(settings);

  int modeIndex = static_cast<int>(settings.mode);
  modeIndex = std::clamp(modeIndex, 0,
                         static_cast<int>(kTextureFilterModeLabels.size()) - 1);
  if (ImGui::Combo("Mode", &modeIndex, kTextureFilterModeLabels.data(),
                   static_cast<int>(kTextureFilterModeLabels.size()))) {
    settings.mode = static_cast<TextureFilterMode>(modeIndex);
  }

  int anisotropyIndex = 2;
  for (int i = 0; i < static_cast<int>(kTextureFilterAnisotropyLevels.size());
       ++i) {
    if (settings.anisotropy ==
        kTextureFilterAnisotropyLevels[static_cast<size_t>(i)]) {
      anisotropyIndex = i;
      break;
    }
  }
  if (ImGui::Combo("Anisotropy", &anisotropyIndex,
                   kTextureFilterAnisotropyLabels.data(),
                   static_cast<int>(kTextureFilterAnisotropyLabels.size()))) {
    settings.anisotropy =
        kTextureFilterAnisotropyLevels[static_cast<size_t>(std::clamp(
            anisotropyIndex, 0,
            static_cast<int>(kTextureFilterAnisotropyLevels.size()) - 1))];
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Bilinear disables mip blending.");
  ImGui::TextUnformatted("Trilinear blends between mip levels.");
  ImGui::TextUnformatted(
      "Anisotropic improves oblique-angle texture sampling.");
  ImGui::Separator();

  const uint8_t maxAnisotropy = gpu.getMaxSamplerAnisotropy();
  const TextureFilterMode effectiveMode =
      effectiveTextureFilterMode(settings, maxAnisotropy);
  ImGui::Text("Requested: %s", textureFilterModeDisplayName(settings.mode));
  ImGui::Text("Effective: %s", textureFilterModeDisplayName(effectiveMode));
  ImGui::Text("Requested Anisotropy: %ux", settings.anisotropy);
  ImGui::Text("Max Backend Anisotropy: %ux", maxAnisotropy);
  if (settings.mode == TextureFilterMode::Anisotropic && maxAnisotropy <= 1u) {
    ImGui::TextUnformatted(
        "Backend fallback: anisotropy unavailable, using trilinear.");
  }

  ImGui::End();
}

void drawShadowsWindow(
    bool &open, RenderSettings &renderSettings, GPUDevice &gpu,
    const std::optional<ShadowDebugFrameData> &shadowDebugFrameData,
    const RenderFrameMetrics &frameMetrics,
    const std::optional<ShadowInspectResult> &shadowInspectResult,
    TextureHandle previewTexture,
    std::vector<TextureHandle> &dependencyTextures,
    std::vector<RenderGraphAccessMode> &dependencyTextureAccessModes) {
  if (!ImGui::Begin(kShadowsWindowName, &open)) {
    ImGui::End();
    return;
  }

  drawShadowSettings(renderSettings.shadow, &shadowDebugFrameData);
  ImGui::Separator();
  dependencyTextures.clear();
  dependencyTextureAccessModes.clear();

  if (!renderSettings.shadow.enabled) {
    ImGui::TextUnformatted("Shadows are disabled.");
    ImGui::End();
    return;
  }

  if (!shadowDebugFrameData.has_value() ||
      shadowDebugFrameData->cascadeCount == 0u) {
    ImGui::TextUnformatted("Shadow debug data is unavailable for this frame.");
    ImGui::End();
    return;
  }

  const uint32_t cascadeIndex =
      std::min(renderSettings.shadow.debug.debugCascadeIndex,
               shadowDebugFrameData->cascadeCount - 1u);
  const ShadowCascadeDebugFrameData &cascade =
      shadowDebugFrameData->cascades[cascadeIndex];
  ImGui::Text("Cascade Count: %u", shadowDebugFrameData->cascadeCount);
  ImGui::Text("Effective Debug Cascade: %u", cascadeIndex);
  if (renderSettings.shadow.debug.debugCascadeIndex != cascadeIndex) {
    ImGui::Text("Requested Debug Cascade %u is unavailable this frame.",
                renderSettings.shadow.debug.debugCascadeIndex);
  }
  if (isValid(shadowDebugFrameData->selectedShadowLightId)) {
    ImGui::Text("Selected Light: %u",
                shadowDebugFrameData->selectedShadowLightId.value);
  } else {
    ImGui::TextUnformatted("Selected Light: none");
  }
  ImGui::Text("Split Range: %.3f .. %.3f", cascade.splitNear, cascade.splitFar);
  ImGui::Text("Texel World Size: %.5f", cascade.texelWorldSize);
  ImGui::Text("Normal Bias: %.3f texels / %.6f world units",
              renderSettings.shadow.normalBias,
              normalBiasWorldUnits(renderSettings.shadow.normalBias,
                                   cascade.texelWorldSize));
  ImGui::Text("Draw Count: %u", cascade.drawCount);
  ImGui::Text("Culled Count: %u", cascade.culledCount);
  ImGui::Text("Static / Dynamic Draws: %u / %u", cascade.staticDrawCount,
              cascade.dynamicDrawCount);
  ImGui::Text("Depth Texture Bindless: %u", cascade.textureBindlessId);
  ImGui::Separator();
  ImGui::TextUnformatted("Static-Only Reuse");
  ImGui::Text("Status: %s", shadowStaticOnlyReuseStatusDisplayName(
                                cascade.staticOnlyReuseStatus));
  ImGui::Text("Candidate / Previous Valid: %s / %s",
              cascade.staticOnlyReuseCandidate ? "yes" : "no",
              cascade.staticOnlyReusePreviousValid ? "yes" : "no");
  ImGui::Text("Signature Changes: light=%s bias=%s casters=%s adaptive=%s",
              cascade.staticOnlyReuseLightViewProjChanged ? "yes" : "no",
              cascade.staticOnlyReuseBiasChanged ? "yes" : "no",
              cascade.staticOnlyReuseCasterSignatureChanged ? "yes" : "no",
              cascade.staticOnlyReuseAdaptiveRefresh ? "yes" : "no");
  ImGui::Text(
      "Raster Signature: 0x%016llX / 0x%016llX",
      static_cast<unsigned long long>(cascade.currentStaticOnlyRasterSignature),
      static_cast<unsigned long long>(
          cascade.previousStaticOnlyRasterSignature));
  ImGui::Text("Light Signature: 0x%016llX / 0x%016llX",
              static_cast<unsigned long long>(
                  cascade.currentStaticOnlyLightViewProjSignature),
              static_cast<unsigned long long>(
                  cascade.previousStaticOnlyLightViewProjSignature));
  ImGui::Text(
      "Caster Signature: 0x%016llX / 0x%016llX",
      static_cast<unsigned long long>(cascade.currentStaticOnlyCasterSignature),
      static_cast<unsigned long long>(
          cascade.previousStaticOnlyCasterSignature));
  ImGui::Separator();
  ImGui::TextUnformatted("Cascade Table");
  if (ImGui::BeginTable("ShadowCascadeTable", 8,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Index");
    ImGui::TableSetupColumn("Near");
    ImGui::TableSetupColumn("Far");
    ImGui::TableSetupColumn("Texel");
    ImGui::TableSetupColumn("Bias World");
    ImGui::TableSetupColumn("Draws");
    ImGui::TableSetupColumn("Culled");
    ImGui::TableSetupColumn("Bindless");
    ImGui::TableHeadersRow();
    const uint32_t cascadeTableCount =
        std::min(shadowDebugFrameData->cascadeCount, kMaxShadowCascades);
    for (uint32_t i = 0u; i < cascadeTableCount; ++i) {
      const ShadowCascadeDebugFrameData &rowCascade =
          shadowDebugFrameData->cascades[i];
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%u", i);
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%.3f", rowCascade.splitNear);
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%.3f", rowCascade.splitFar);
      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%.5f", rowCascade.texelWorldSize);
      ImGui::TableSetColumnIndex(4);
      ImGui::Text("%.6f", normalBiasWorldUnits(renderSettings.shadow.normalBias,
                                               rowCascade.texelWorldSize));
      ImGui::TableSetColumnIndex(5);
      ImGui::Text("%u", rowCascade.drawCount);
      ImGui::TableSetColumnIndex(6);
      ImGui::Text("%u", rowCascade.culledCount);
      ImGui::TableSetColumnIndex(7);
      ImGui::Text("%u", rowCascade.textureBindlessId);
    }
    ImGui::EndTable();
  }
  const glm::vec3 unsnappedCenter = glm::vec3(cascade.unsnappedCenter);
  const glm::vec3 snappedCenter = glm::vec3(cascade.snappedCenter);
  const glm::vec3 snapDelta = snappedCenter - unsnappedCenter;
  ImGui::Text("Unsnapped Center: %.3f %.3f %.3f", unsnappedCenter.x,
              unsnappedCenter.y, unsnappedCenter.z);
  ImGui::Text("Snapped Center: %.3f %.3f %.3f", snappedCenter.x,
              snappedCenter.y, snappedCenter.z);
  ImGui::Text("Snap Delta: %.5f %.5f %.5f", snapDelta.x, snapDelta.y,
              snapDelta.z);
  ImGui::Text("Freeze Cascades: %s", renderSettings.shadow.debug.freezeCascades
                                         ? "active"
                                         : "inactive");
  ImGui::Text("Freeze Light View: %s",
              renderSettings.shadow.debug.freezeLightView ? "active"
                                                          : "inactive");
  ImGui::TextUnformatted("Receiver Debug");
  if (!shadowInspectResult.has_value()) {
    ImGui::TextUnformatted(
        "Click the viewport to inspect the last opaque pixel.");
  } else if (!shadowInspectResult->valid) {
    ImGui::TextUnformatted(
        "Last inspected pixel did not produce a valid shadow sample.");
  } else {
    const float depthDelta = shadowInspectResult->receiverCompareDepth -
                             shadowInspectResult->sampledDepth;
    ImGui::Text("Active Cascade: %u", shadowInspectResult->cascadeIndex);
    ImGui::Text("Cascade Blend Factor: %.3f",
                shadowInspectResult->cascadeBlendFactor);
    ImGui::Text("Receiver Depth: %.6f", shadowInspectResult->receiverDepth);
    ImGui::Text("Receiver Compare Depth: %.6f",
                shadowInspectResult->receiverCompareDepth);
    ImGui::Text("Sampled Depth: %.6f", shadowInspectResult->sampledDepth);
    ImGui::Text("Compare Delta: %.6f", depthDelta);
  }

  ImGui::Separator();
  const ShadowSdsmDebugFrameData &sdsm = shadowDebugFrameData->sdsm;
  ImGui::TextUnformatted("Previous-frame min/max SDSM");
  ImGui::Text("Active Backend: %s", shadowSdsmReductionBackendDisplayName(
                                        sdsm.activeReductionBackend));
  if (sdsm.activeReductionBackend == ShadowSdsmReductionBackend::Gpu ||
      sdsm.gpuResultRingSlotCount > 0u) {
    ImGui::Text("GPU Result: %s",
                sdsm.gpuReductionResultAvailable ? "available" : "unavailable");
    if (sdsm.gpuResultSelectedSlot != std::numeric_limits<uint32_t>::max()) {
      ImGui::Text("GPU Ring: slot %u of %u", sdsm.gpuResultSelectedSlot,
                  sdsm.gpuResultRingSlotCount);
    } else {
      ImGui::Text("GPU Ring: no completed slot of %u",
                  sdsm.gpuResultRingSlotCount);
    }
    if (sdsm.gpuResultSourceFrameIndex !=
        std::numeric_limits<uint64_t>::max()) {
      ImGui::Text(
          "GPU Result Frame: %llu",
          static_cast<unsigned long long>(sdsm.gpuResultSourceFrameIndex));
    }
  }
  ImGui::Text("Reduction Fallback: %s",
              sdsm.reductionFallbackActive ? "active" : "inactive");
  ImGui::Text("Status: %s", shadowSdsmStatusDisplayName(sdsm.status));
  if (sdsm.sourceFrameIndex != std::numeric_limits<uint64_t>::max()) {
    ImGui::Text("Source Frame: %llu",
                static_cast<unsigned long long>(sdsm.sourceFrameIndex));
  } else {
    ImGui::TextUnformatted("Source Frame: unavailable");
  }
  ImGui::Text("Raw Device Min/Max: %.6f / %.6f", sdsm.rawDeviceMin,
              sdsm.rawDeviceMax);
  ImGui::Text("Raw Linear Min/Max: %.6f / %.6f", sdsm.rawLinearMin,
              sdsm.rawLinearMax);
  ImGui::Text("Smoothed Min/Max: %.6f / %.6f", sdsm.smoothedLinearMin,
              sdsm.smoothedLinearMax);
  ImGui::Text("Fixed Range: %.6f .. %.6f", sdsm.fixedRangeNear,
              sdsm.fixedRangeFar);
  ImGui::Text("Effective Range: %.6f .. %.6f", sdsm.effectiveRangeNear,
              sdsm.effectiveRangeFar);
  if (isSdsmWarningStatus(sdsm.status)) {
    ImGui::TextUnformatted(sdsm.fixedFallbackActive
                               ? "Warning: previous-frame SDSM data was not "
                                 "usable this frame; fixed splits are active."
                               : "Warning: previous-frame SDSM data was not "
                                 "usable this frame; reusing the last valid "
                                 "SDSM range.");
  }
  drawShadowSplitGraph(sdsm);

  ImGui::Separator();
  ImGui::TextUnformatted("Performance");
  ImGui::Text("Cascades / Map Size: %u / %u", frameMetrics.shadow.cascadeCount,
              frameMetrics.shadow.shadowMapSize);
  ImGui::Text("Draws / Culled: %u / %u", frameMetrics.shadow.totalDraws,
              frameMetrics.shadow.totalCulledDraws);
  ImGui::Text("Index Count Estimate: %u",
              frameMetrics.shadow.totalIndexCountEstimate);
  ImGui::Text(
      "Cascade Texture Bytes: %llu",
      static_cast<unsigned long long>(frameMetrics.shadow.cascadeTextureBytes));
  ImGui::Text("Texel World Size [min/avg/max/far]: %.5f / %.5f / %.5f / %.5f",
              frameMetrics.shadow.minCascadeTexelWorldSize,
              frameMetrics.shadow.averageCascadeTexelWorldSize,
              frameMetrics.shadow.maxCascadeTexelWorldSize,
              frameMetrics.shadow.farCascadeTexelWorldSize);
  ImGui::Text("Static / Dynamic Casters: %u / %u",
              frameMetrics.shadow.staticCasterEntries,
              frameMetrics.shadow.dynamicCasterEntries);
  ImGui::Text("Static Batches [templates/full]: %u / %u",
              frameMetrics.shadow.staticBatchTemplateCount,
              frameMetrics.shadow.staticBatchFullEmitCount);
  ImGui::Text("Shadow Batches / Remap: %u / %u",
              frameMetrics.shadow.shadowBatchEntryCount,
              frameMetrics.shadow.shadowInstanceRemapCount);
  ImGui::Text(
      "Static Grid [queries/fallback/cells/candidates]: %u / %u / %u / %u",
      frameMetrics.shadow.staticLightGridQueryCount,
      frameMetrics.shadow.staticLightGridFallbackScanCount,
      frameMetrics.shadow.staticLightGridQueryCellCount,
      frameMetrics.shadow.staticLightGridCandidateCount);
  ImGui::Text("Static Cache Reused: %s",
              frameMetrics.shadow.staticCacheReused != 0u ? "yes" : "no");
  ImGui::Text("Static-Only Candidates: %u",
              frameMetrics.shadow.staticOnlyCandidateCount);
  ImGui::Text("Reused Static Cascades: %u",
              frameMetrics.shadow.reusedStaticOnlyCascadeCount);
  ImGui::Text(
      "Reuse Misses [cache/dynamic/prev/raster/adapt]: %u / %u / %u / %u / %u",
      frameMetrics.shadow.staticOnlyReuseMissStaticCacheRebuiltCount,
      frameMetrics.shadow.staticOnlyReuseMissDynamicCasterCount,
      frameMetrics.shadow.staticOnlyReuseMissNoPreviousCount,
      frameMetrics.shadow.staticOnlyReuseMissRasterStateChangedCount,
      frameMetrics.shadow.staticOnlyReuseMissAdaptiveRefreshCount);
  ImGui::Text("Filter Sample Budget: %u",
              frameMetrics.shadow.filterSampleBudget);
  ImGui::Text("SDSM Compute Passes: %u",
              frameMetrics.shadow.sdsmComputePassCount);
  ImGui::Text("SDSM Readback Bytes: %u", frameMetrics.shadow.sdsmReadbackBytes);
  ImGui::Text("SDSM Reduction Source Samples: %u",
              frameMetrics.shadow.sdsmReductionSourceSamples);
  ImGui::Text("SDSM CPU Reduction Cost: %.3f ms",
              frameMetrics.shadow.sdsmCpuReductionTimeMs);
  if (frameMetrics.shadow.depthGpuTimingAvailable != 0u) {
    if (frameMetrics.shadow.depthGpuTimingSourceFrameIndex !=
        std::numeric_limits<uint64_t>::max()) {
      ImGui::Text("Shadow Depth GPU Time: %.3f ms (frame %llu)",
                  frameMetrics.shadow.depthGpuTimeMs,
                  static_cast<unsigned long long>(
                      frameMetrics.shadow.depthGpuTimingSourceFrameIndex));
    } else {
      ImGui::Text("Shadow Depth GPU Time: %.3f ms",
                  frameMetrics.shadow.depthGpuTimeMs);
    }
  } else {
    ImGui::TextUnformatted("Shadow Depth GPU Time: unavailable");
  }
  if (frameMetrics.shadow.sdsmGpuTimingAvailable != 0u) {
    if (frameMetrics.shadow.sdsmGpuTimingSourceFrameIndex !=
        std::numeric_limits<uint64_t>::max()) {
      ImGui::Text("SDSM GPU Time: %.3f ms (frame %llu)",
                  frameMetrics.shadow.sdsmGpuTimeMs,
                  static_cast<unsigned long long>(
                      frameMetrics.shadow.sdsmGpuTimingSourceFrameIndex));
    } else {
      ImGui::Text("SDSM GPU Time: %.3f ms", frameMetrics.shadow.sdsmGpuTimeMs);
    }
  } else {
    ImGui::TextUnformatted("SDSM GPU Time: unavailable");
  }
  if (frameMetrics.shadow.gpuTimingAvailable != 0u) {
    if (frameMetrics.shadow.gpuTimingSourceFrameIndex !=
        std::numeric_limits<uint64_t>::max()) {
      ImGui::Text("GPU Time: %.3f ms (frame %llu)",
                  frameMetrics.shadow.gpuTimeMs,
                  static_cast<unsigned long long>(
                      frameMetrics.shadow.gpuTimingSourceFrameIndex));
    } else {
      ImGui::Text("GPU Time: %.3f ms", frameMetrics.shadow.gpuTimeMs);
    }
  } else {
    ImGui::TextUnformatted("GPU Time: unavailable");
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Preview");
  if (!nuri::isValid(previewTexture)) {
    ImGui::TextUnformatted(
        "Enable 'Shadow Map Viewport' in Shadow debug settings to allocate "
        "the preview texture.");
    ImGui::End();
    return;
  }

  const uint32_t previewBindlessId =
      gpu.getTextureBindlessIndex(previewTexture);
  if (previewBindlessId == kInvalidTextureBindlessIndex) {
    ImGui::TextUnformatted("Shadow preview texture is unavailable.");
    ImGui::End();
    return;
  }

  dependencyTextures.push_back(previewTexture);
  dependencyTextureAccessModes.push_back(RenderGraphAccessMode::Read);
  ImGui::Text("Preview Format: %s",
              formatDisplayName(gpu.getTextureFormat(previewTexture)));
  const TextureDimensions previewDimensions =
      gpu.getTextureDimensions(previewTexture);
  ImGui::Text("Preview Resolution: %u x %u", previewDimensions.width,
              previewDimensions.height);
  ImGui::Text("Preview Mode: %s", shadowPreviewModeDisplayName(
                                      renderSettings.shadow.debug.previewMode));
  ImGui::Text("Preview Range: %.4f .. %.4f",
              renderSettings.shadow.debug.previewDepthMin,
              renderSettings.shadow.debug.previewDepthMax);
  ImGui::Text("Preview Transform: %s%s",
              renderSettings.shadow.debug.previewDepthInvert ? "invert"
                                                             : "linear",
              renderSettings.shadow.debug.previewDepthLog ? " + log" : "");
  ImGui::Image(toImTextureId(previewBindlessId), ImVec2(256.0f, 256.0f));
  ImGui::End();
}

std::string makePassListLabel(const RenderPipelinePassInfo &passInfo) {
  std::string label;
  label.reserve(passInfo.passName.size() + passInfo.featureName.size() + 4u);
  label.append(passInfo.passName.begin(), passInfo.passName.end());
  if (!passInfo.featureName.empty()) {
    label.append("##");
    label.append(passInfo.featureName.begin(), passInfo.featureName.end());
    label.push_back('_');
    label.append(std::to_string(passInfo.index));
  }
  return label;
}

void drawPassList(RenderSettings &renderSettings,
                  RenderPipeline *renderPipeline, size_t &selectedPassIndex) {
  ImGui::TextUnformatted("Passes");
  ImGui::Separator();
  if (renderPipeline == nullptr || renderPipeline->passCount() == 0u) {
    ImGui::TextDisabled("No pipeline passes registered.");
    return;
  }

  if (selectedPassIndex >= renderPipeline->passCount()) {
    selectedPassIndex = 0u;
  }

  for (size_t passIndex = 0; passIndex < renderPipeline->passCount();
       ++passIndex) {
    const std::optional<RenderPipelinePassInfo> passInfo =
        renderPipeline->passInfo(passIndex);
    if (!passInfo.has_value()) {
      continue;
    }
    const PassInspectorKind kind =
        classifyPassInspector(passInfo->featureName, passInfo->passName);
    bool enabled = passKindUsesFeatureToggle(kind)
                       ? isPassFamilyEnabled(renderPipeline, *passInfo, kind)
                       : passInfo->enabled;
    if (ImGui::Checkbox(
            ("##PassEnabled" + std::to_string(passInfo->index)).c_str(),
            &enabled)) {
      if (passKindUsesFeatureToggle(kind)) {
        setPassFamilyEnabled(renderPipeline, *passInfo, kind, enabled);
        syncFeatureToggleToRenderSettings(renderSettings, kind, enabled);
      } else {
        renderPipeline->setPassEnabled(passInfo->index, enabled);
      }
    }
    ImGui::SameLine();
    const std::string label = makePassListLabel(*passInfo);
    const bool isSelected = selectedPassIndex == passInfo->index;
    if (ImGui::Selectable(label.c_str(), isSelected)) {
      selectedPassIndex = passInfo->index;
    }
  }
}

void drawPassInspector(RenderSettings &renderSettings,
                       RenderPipeline *renderPipeline,
                       const RenderFrameMetrics &frameMetrics,
                       size_t &selectedPassIndex) {
  ImGui::BeginChild("PassPanel", ImVec2(0.0f, 0.0f), false,
                    ImGuiWindowFlags_NoScrollbar);

  if (ImGui::BeginTable("PassInspectorTable", 2,
                        ImGuiTableFlags_BordersInnerV |
                            ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("PassList", ImGuiTableColumnFlags_WidthFixed,
                            kPassListWidth);
    ImGui::TableSetupColumn("PassSettings", ImGuiTableColumnFlags_WidthStretch,
                            0.0f);

    ImGui::TableNextColumn();
    drawPassList(renderSettings, renderPipeline, selectedPassIndex);

    ImGui::TableNextColumn();
    if (renderPipeline == nullptr || renderPipeline->passCount() == 0u) {
      drawInspectorHeader("No Pass Selected");
      ImGui::TextUnformatted("RenderPipeline is unavailable.");
    } else {
      selectedPassIndex =
          std::min(selectedPassIndex, renderPipeline->passCount() - 1u);
      const std::optional<RenderPipelinePassInfo> passInfo =
          renderPipeline->passInfo(selectedPassIndex);
      if (!passInfo.has_value()) {
        drawInspectorHeader("No Pass Selected");
        ImGui::TextUnformatted("Selected pass entry is unavailable.");
      } else {
        std::string title(passInfo->passName);
        if (!passInfo->featureName.empty()) {
          title.append(" (");
          title.append(passInfo->featureName);
          title.push_back(')');
        }
        drawInspectorHeader(title);
        const PassInspectorKind kind =
            classifyPassInspector(passInfo->featureName, passInfo->passName);
        bool enabled =
            passKindUsesFeatureToggle(kind)
                ? isPassFamilyEnabled(renderPipeline, *passInfo, kind)
                : passInfo->enabled;
        if (ImGui::Checkbox("Enabled", &enabled)) {
          if (passKindUsesFeatureToggle(kind)) {
            setPassFamilyEnabled(renderPipeline, *passInfo, kind, enabled);
            syncFeatureToggleToRenderSettings(renderSettings, kind, enabled);
          } else {
            renderPipeline->setPassEnabled(passInfo->index, enabled);
          }
        }
        ImGui::Separator();

        switch (kind) {
        case PassInspectorKind::Skybox:
          drawSkyboxSettings(renderSettings.skybox);
          break;
        case PassInspectorKind::Shadow:
          drawShadowSettings(renderSettings.shadow);
          break;
        case PassInspectorKind::Opaque:
          drawOpaqueSettings(renderSettings.opaque, renderSettings.shadow,
                             frameMetrics.opaque);
          break;
        case PassInspectorKind::Transmission:
          drawTransmissionSettings(renderSettings.transmission);
          break;
        case PassInspectorKind::Transparent:
          drawTransparentSettings(renderSettings.transparent);
          break;
        case PassInspectorKind::Composite:
          drawCompositeSettings(renderSettings.toneMap,
                                renderSettings.hdrPostProcess, frameMetrics);
          break;
        case PassInspectorKind::AntiAliasing:
          drawAntiAliasingSettings(renderSettings.antiAliasing, renderPipeline,
                                   frameMetrics);
          break;
        case PassInspectorKind::AmbientOcclusion:
          drawAmbientOcclusionSettings(
              renderSettings.ambientOcclusion, renderSettings.opaque,
              renderSettings.antiAliasing, frameMetrics);
          break;
        case PassInspectorKind::Debug:
          drawDebugSettings(renderSettings.debug);
          break;
        case PassInspectorKind::Generic:
          ImGui::TextUnformatted("This pass has no dedicated inspector yet.");
          break;
        }
      }
    }

    ImGui::EndTable();
  }

  ImGui::EndChild();
}

void drawLogWindow(LogModel &model, LogFilterState &filterState,
                   std::pmr::memory_resource *scratchResource) {
  drawLogToolbar(model, filterState);
  ImGui::Separator();
  drawLogMessages(model, filterState, scratchResource);
}

void drawRenderPassesWindow(bool &open, RenderSettings &renderSettings,
                            RenderPipeline *renderPipeline,
                            const RenderFrameMetrics &frameMetrics,
                            size_t &selectedPassIndex) {
  if (!ImGui::Begin(kRenderPassesWindowName, &open)) {
    ImGui::End();
    return;
  }
  drawPassInspector(renderSettings, renderPipeline, frameMetrics,
                    selectedPassIndex);
  ImGui::End();
}

void drawFontCompilerWindow(bool &open, FontCompilerUiState &state,
                            TextSystem *textSystem, void *ownerWindowHandle) {
  if (!state.nfontListInitialized) {
    refreshNfontAssetList(state);
    state.nfontListInitialized = true;
  }

  if (state.compileInFlight && state.compileFuture.valid()) {
    const auto waitResult =
        state.compileFuture.wait_for(std::chrono::seconds(0));
    if (waitResult == std::future_status::ready) {
      auto compileResult = state.compileFuture.get();
      state.compileInFlight = false;
      if (compileResult.hasError()) {
        state.error = compileResult.error();
      } else {
        state.lastReport = compileResult.value();
        std::ostringstream status;
        status << "Generated " << state.lastReport.outputPath.string()
               << " | glyphs=" << state.lastReport.glyphCount
               << " atlas=" << state.lastReport.atlasWidth << "x"
               << state.lastReport.atlasHeight
               << " bytes=" << state.lastReport.bytesWritten;
        state.status = status.str();
        setPathText(
            state.selectedNfontPath,
            state.lastReport.outputPath.lexically_normal().generic_string());
        refreshNfontAssetList(state);
      }
    }
  }

  if (!ImGui::Begin(kFontCompilerWindowName, &open)) {
    ImGui::End();
    return;
  }

  bool sourceEdited = ImGui::InputText(
      "Source TTF/OTF", state.sourcePath.data(), state.sourcePath.size());
  ImGui::SameLine();
  if (ImGui::Button("Browse...##FontSource")) {
    if (const auto selectedPath =
            state.fileDialog.openFontFile(ownerWindowHandle)) {
      setPathText(state.sourcePath, selectedPath->generic_string());
      sourceEdited = true;
    }
  }

  if (sourceEdited && state.autoOutputName) {
    syncOutputPathFromSource(state);
  }

  ImGui::InputText("Output .nfont", state.outputPath.data(),
                   state.outputPath.size());
  ImGui::SameLine();
  if (ImGui::Checkbox("Auto name", &state.autoOutputName) &&
      state.autoOutputName) {
    syncOutputPathFromSource(state);
  }

  constexpr const char *kCharsetOptions[] = {"ASCII", "Latin-1"};
  ImGui::Combo("Charset", &state.charsetPreset, kCharsetOptions,
               IM_ARRAYSIZE(kCharsetOptions));

  ImGui::SliderFloat("Minimum EM Size", &state.minimumEmSize, 8.0f, 128.0f,
                     "%.1f");
  ImGui::SliderFloat("PX Range", &state.pxRange, 1.0f, 16.0f, "%.1f");
  ImGui::SliderFloat("Outer PX Padding", &state.outerPixelPadding, 0.0f, 16.0f,
                     "%.1f");
  ImGui::SliderInt("Atlas Spacing", &state.atlasSpacing, 0, 16);
  ImGui::Checkbox("RGBA16F Atlas", &state.useRgba16fAtlas);
  if (ImGui::SliderInt("Atlas Width Step", &state.atlasWidthPreset, 0,
                       static_cast<int>(kAtlasResolutionSteps.size()) - 1)) {
    state.atlasWidthPreset =
        std::clamp(state.atlasWidthPreset, 0,
                   static_cast<int>(kAtlasResolutionSteps.size()) - 1);
    state.maxAtlasWidth =
        kAtlasResolutionSteps[static_cast<size_t>(state.atlasWidthPreset)];
  }
  ImGui::SameLine();
  ImGui::TextUnformatted(kAtlasResolutionStepLabels[static_cast<size_t>(
      std::clamp(state.atlasWidthPreset, 0,
                 static_cast<int>(kAtlasResolutionStepLabels.size()) - 1))]);
  if (ImGui::InputInt("Max Atlas Width", &state.maxAtlasWidth)) {
    state.maxAtlasWidth = std::clamp(state.maxAtlasWidth, 1, 8192);
    state.atlasWidthPreset = closestAtlasStepIndex(state.maxAtlasWidth);
  }

  if (ImGui::SliderInt("Atlas Height Step", &state.atlasHeightPreset, 0,
                       static_cast<int>(kAtlasResolutionSteps.size()) - 1)) {
    state.atlasHeightPreset =
        std::clamp(state.atlasHeightPreset, 0,
                   static_cast<int>(kAtlasResolutionSteps.size()) - 1);
    state.maxAtlasHeight =
        kAtlasResolutionSteps[static_cast<size_t>(state.atlasHeightPreset)];
  }
  ImGui::SameLine();
  ImGui::TextUnformatted(kAtlasResolutionStepLabels[static_cast<size_t>(
      std::clamp(state.atlasHeightPreset, 0,
                 static_cast<int>(kAtlasResolutionStepLabels.size()) - 1))]);
  if (ImGui::InputInt("Max Atlas Height", &state.maxAtlasHeight)) {
    state.maxAtlasHeight = std::clamp(state.maxAtlasHeight, 1, 8192);
    state.atlasHeightPreset = closestAtlasStepIndex(state.maxAtlasHeight);
  }

  ImGui::SliderInt("Threads", &state.threadCount, 0, 32);

  const bool wasCompileInFlight = state.compileInFlight;
  if (wasCompileInFlight) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("Generate .nfont")) {
    state.status.clear();
    state.error.clear();

    NFontCompileConfig config{};
    config.sourceFontPath = std::string(state.sourcePath.data());
    config.outputFontPath = std::string(state.outputPath.data());
    config.charset = state.charsetPreset == 0 ? NFontCharsetPreset::Ascii
                                              : NFontCharsetPreset::Latin1;
    config.minimumEmSize = state.minimumEmSize;
    config.pxRange = state.pxRange;
    config.outerPixelPadding = state.outerPixelPadding;
    config.atlasSpacing =
        state.atlasSpacing > 0 ? static_cast<uint32_t>(state.atlasSpacing) : 0u;
    config.useRgba16fAtlas = state.useRgba16fAtlas;
    config.maxAtlasWidth = state.maxAtlasWidth > 0
                               ? static_cast<uint32_t>(state.maxAtlasWidth)
                               : 0u;
    config.maxAtlasHeight = state.maxAtlasHeight > 0
                                ? static_cast<uint32_t>(state.maxAtlasHeight)
                                : 0u;
    config.threadCount =
        state.threadCount > 0 ? static_cast<uint32_t>(state.threadCount) : 0u;

    state.compileFuture = std::async(std::launch::async, [config]() {
                            return compileNFontFromFontFile(config);
                          }).share();
    state.compileInFlight = true;
  }
  if (wasCompileInFlight) {
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextUnformatted("Compiling...");
  }

  if (!state.status.empty()) {
    ImGui::Spacing();
    ImGui::TextUnformatted(state.status.c_str());
  }
  if (!state.error.empty()) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                       state.error.c_str());
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Global Text Font");

  if (textSystem == nullptr) {
    ImGui::TextUnformatted("TextSystem is not available.");
    ImGui::End();
    return;
  }

  if (ImGui::Button("Refresh .nfont List")) {
    refreshNfontAssetList(state);
  }
  ImGui::SameLine();
  ImGui::Text("Dir: %s", state.outputDirectory.generic_string().c_str());

  const std::string previewLabel =
      state.selectedNfontIndex >= 0 &&
              static_cast<size_t>(state.selectedNfontIndex) <
                  state.availableNfonts.size()
          ? state.availableNfonts[static_cast<size_t>(state.selectedNfontIndex)]
                .filename()
                .string()
          : std::string("<none>");
  if (ImGui::BeginCombo("Available .nfont", previewLabel.c_str())) {
    for (size_t i = 0; i < state.availableNfonts.size(); ++i) {
      const bool selected = state.selectedNfontIndex == static_cast<int>(i);
      const std::string label = state.availableNfonts[i].filename().string();
      if (ImGui::Selectable(label.c_str(), selected)) {
        state.selectedNfontIndex = static_cast<int>(i);
        setPathText(state.selectedNfontPath,
                    state.availableNfonts[i].generic_string());
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  if (ImGui::InputText("Selected .nfont", state.selectedNfontPath.data(),
                       state.selectedNfontPath.size())) {
    const std::filesystem::path selectedPath(
        std::string(state.selectedNfontPath.data()));
    state.selectedNfontIndex = -1;
    for (size_t i = 0; i < state.availableNfonts.size(); ++i) {
      if (state.availableNfonts[i] == selectedPath) {
        state.selectedNfontIndex = static_cast<int>(i);
        break;
      }
    }
  }

  state.globalFontSizePx = std::clamp(state.globalFontSizePx, 8.0f, 256.0f);
  ImGui::SliderFloat("Global Font Size (px)", &state.globalFontSizePx, 8.0f,
                     256.0f, "%.1f");
  if (ImGui::Button("Apply Global Font")) {
    state.globalStatus.clear();
    state.globalError.clear();
    textSystem->setDefaultFontSizePx(state.globalFontSizePx);

    const std::filesystem::path selectedPath(
        std::string(state.selectedNfontPath.data()));
    if (selectedPath.empty()) {
      state.globalError = "No .nfont selected";
    } else {
      auto loadResult = textSystem->loadAndSetDefaultFont(
          selectedPath.generic_string(), selectedPath.stem().string());
      if (loadResult.hasError()) {
        state.globalError = loadResult.error();
      } else {
        std::ostringstream oss;
        oss << "Applied " << selectedPath.filename().string() << " at "
            << state.globalFontSizePx << "px";
        state.globalStatus = oss.str();
      }
    }
  }

  if (!state.globalStatus.empty()) {
    ImGui::Spacing();
    ImGui::TextUnformatted(state.globalStatus.c_str());
  }
  if (!state.globalError.empty()) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                       state.globalError.c_str());
  }

  ImGui::End();
}

void drawBakeryWindow(bool &open, BakeryUiState &state,
                      bakery::BakerySystem *bakery,
                      std::pmr::memory_resource *scratchResource,
                      void *ownerWindowHandle) {
  if (!ImGui::Begin(kBakeryWindowName, &open)) {
    ImGui::End();
    return;
  }

  if (bakery == nullptr) {
    ImGui::TextUnformatted("Bakery system is not available.");
    ImGui::End();
    return;
  }

  ImGui::Checkbox("Force Rebuild", &state.forceRebuild);

  if (ImGui::Button("Queue BRDF LUT")) {
    state.status.clear();
    state.error.clear();
    auto enqueueResult =
        bakery->enqueue(bakery::BakeRequest{bakery::BrdfLutBakeRequest{
            .forceRebuild = state.forceRebuild,
        }});
    if (enqueueResult.hasError()) {
      state.error = enqueueResult.error();
    } else {
      std::ostringstream oss;
      oss << "Queued BRDF LUT job #" << enqueueResult.value().value;
      state.status = oss.str();
    }
  }

  ImGui::InputText("Env HDR Path", state.envHdrPath.data(),
                   state.envHdrPath.size());
  ImGui::SameLine();
  if (ImGui::Button("Browse...##EnvHdr")) {
    static constexpr std::array<FileDialogFilter, 4> kHdrFilters = {
        FileDialogFilter{"HDR Files (*.hdr;*.exr)", "*.hdr;*.exr"},
        FileDialogFilter{"Radiance HDR (*.hdr)", "*.hdr"},
        FileDialogFilter{"OpenEXR (*.exr)", "*.exr"},
        FileDialogFilter{"All Files (*.*)", "*.*"},
    };
    OpenFileRequest request{};
    request.title = "Select Environment HDR";
    request.filters = kHdrFilters;
    request.defaultExtension = "hdr";
    request.ownerWindowHandle = ownerWindowHandle;
    if (const auto selectedPath = state.fileDialog.openFile(request)) {
      setPathText(state.envHdrPath, selectedPath->generic_string());
    }
  }
  if (ImGui::Button("Queue Env Prefilter")) {
    state.status.clear();
    state.error.clear();
    auto enqueueResult =
        bakery->enqueue(bakery::BakeRequest{bakery::EnvmapPrefilterBakeRequest{
            .environmentHdrPath =
                std::filesystem::path(std::string(state.envHdrPath.data())),
            .forceRebuild = state.forceRebuild,
        }});
    if (enqueueResult.hasError()) {
      state.error = enqueueResult.error();
    } else {
      std::ostringstream oss;
      oss << "Queued Env Prefilter job #" << enqueueResult.value().value;
      state.status = oss.str();
    }
  }

  ImGui::Separator();
  ImGui::InputText("Scene Path", state.scenePath.data(),
                   state.scenePath.size());
  ImGui::SameLine();
  if (ImGui::Button("Browse...##SceneBake")) {
    static constexpr std::array<FileDialogFilter, 3> kSceneFilters = {
        FileDialogFilter{"Scene Files (*.gltf;*.glb)", "*.gltf;*.glb"},
        FileDialogFilter{"glTF (*.gltf)", "*.gltf"},
        FileDialogFilter{"GLB (*.glb)", "*.glb"},
    };
    OpenFileRequest request{};
    request.title = "Select Scene for Texture Artifact Bake";
    request.filters = kSceneFilters;
    request.defaultExtension = "gltf";
    request.ownerWindowHandle = ownerWindowHandle;
    if (const auto selectedPath = state.fileDialog.openFile(request)) {
      setPathText(state.scenePath, selectedPath->generic_string());
    }
  }

  ImGui::Checkbox("Prebuild BC7", &state.prebuildBc7);
  ImGui::SameLine();
  ImGui::Checkbox("Prebuild ETC2", &state.prebuildEtc2);
  ImGui::SameLine();
  ImGui::Checkbox("Prebuild RGBA8", &state.prebuildRgba8);
  ImGui::TextUnformatted(
      "Scene Texture Artifacts writes target-native KTX2 cache entries.");
  ImGui::TextUnformatted(
      "External DDS textures stay authored and are not rebaked here.");

  if (ImGui::Button("Queue Scene Texture Artifacts")) {
    state.status.clear();
    state.error.clear();

    std::vector<bakery::SceneTextureArtifactTarget> prebuildTargets{};
    if (state.prebuildBc7) {
      prebuildTargets.push_back(bakery::SceneTextureArtifactTarget::BC7);
    }
    if (state.prebuildEtc2) {
      prebuildTargets.push_back(bakery::SceneTextureArtifactTarget::ETC2);
    }
    if (state.prebuildRgba8) {
      prebuildTargets.push_back(bakery::SceneTextureArtifactTarget::RGBA8);
    }

    auto enqueueResult = bakery->enqueue(
        bakery::BakeRequest{bakery::SceneTextureArtifactsBakeRequest{
            .scenePath =
                std::filesystem::path(std::string(state.scenePath.data())),
            .prebuildNativeTargets = std::move(prebuildTargets),
            .forceRebuild = state.forceRebuild,
        }});
    if (enqueueResult.hasError()) {
      state.error = enqueueResult.error();
    } else {
      std::ostringstream oss;
      oss << "Queued Scene Texture Artifacts job #"
          << enqueueResult.value().value;
      state.status = oss.str();
    }
  }

  if (!state.status.empty()) {
    ImGui::Spacing();
    ImGui::TextUnformatted(state.status.c_str());
  }
  if (!state.error.empty()) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                       state.error.c_str());
  }

  std::pmr::vector<bakery::BakeJobSnapshot> jobs =
      bakery->snapshotJobs(scratchResource);
  ImGui::Separator();
  ImGui::Text("Jobs: %d", static_cast<int>(jobs.size()));
  if (ImGui::BeginTable("BakeryJobs", 6,
                        ImGuiTableFlags_BordersInnerV |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_ScrollY,
                        ImVec2(0.0f, 240.0f))) {
    ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 52.0f);
    ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 120.0f);
    ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 110.0f);
    ImGui::TableSetupColumn("Progress", ImGuiTableColumnFlags_WidthFixed,
                            90.0f);
    ImGui::TableSetupColumn("Summary", ImGuiTableColumnFlags_WidthStretch,
                            0.0f);
    ImGui::TableSetupColumn("Error", ImGuiTableColumnFlags_WidthStretch, 0.0f);
    ImGui::TableHeadersRow();

    for (const bakery::BakeJobSnapshot &job : jobs) {
      ImGui::TableNextRow();

      ImGui::TableNextColumn();
      ImGui::Text("%llu", static_cast<unsigned long long>(job.id.value));

      ImGui::TableNextColumn();
      ImGui::TextUnformatted(bakeJobKindName(job.kind));

      ImGui::TableNextColumn();
      ImGui::TextUnformatted(bakeJobStateName(job.state));

      ImGui::TableNextColumn();
      ImGui::Text("%.0f%%", std::clamp(job.progress01, 0.0f, 1.0f) * 100.0f);

      ImGui::TableNextColumn();
      ImGui::TextUnformatted(job.summary.c_str());

      ImGui::TableNextColumn();
      if (job.error.empty()) {
        ImGui::TextUnformatted("-");
      } else {
        ImGui::TextWrapped("%s", job.error.c_str());
      }
    }
    ImGui::EndTable();
  }

  ImGui::End();
}

[[nodiscard]] std::string_view
resolveTelemetryPassName(const RenderGraphTelemetrySnapshot &snapshot,
                         uint32_t passIndex) {
  if (passIndex >= snapshot.compile.passDebugNames.size()) {
    return "unnamed_pass";
  }
  const std::pmr::string &name = snapshot.compile.passDebugNames[passIndex];
  return name.empty() ? std::string_view("unnamed_pass")
                      : std::string_view(name.data(), name.size());
}

const char *drawBufferBindingTargetName(
    RenderGraphCompileResult::DrawBufferBindingTarget target) {
  switch (target) {
  case RenderGraphCompileResult::DrawBufferBindingTarget::Vertex:
    return "vertex";
  case RenderGraphCompileResult::DrawBufferBindingTarget::Index:
    return "index";
  case RenderGraphCompileResult::DrawBufferBindingTarget::Indirect:
    return "indirect";
  case RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount:
    return "indirect_count";
  }
  return "unknown";
}

const char *passTextureBindingTargetName(
    RenderGraphCompileResult::PassTextureBindingTarget target) {
  switch (target) {
  case RenderGraphCompileResult::PassTextureBindingTarget::Color:
    return "color";
  case RenderGraphCompileResult::PassTextureBindingTarget::ColorResolve:
    return "color_resolve";
  case RenderGraphCompileResult::PassTextureBindingTarget::Depth:
    return "depth";
  case RenderGraphCompileResult::PassTextureBindingTarget::DepthResolve:
    return "depth_resolve";
  }
  return "unknown";
}

void syncTelemetryDumpPath(RenderGraphTelemetryUiState &state,
                           const RenderGraphTelemetryService *telemetry) {
  if (telemetry == nullptr) {
    return;
  }
  const std::string currentPath = state.outputPath.data();
  const std::string suggestedPath =
      telemetry->suggestDumpPath().generic_string();
  if (state.initializedOutputPath && currentPath != state.lastSuggestedPath) {
    return;
  }
  setPathText(state.outputPath, suggestedPath);
  state.lastSuggestedPath = suggestedPath;
  state.initializedOutputPath = true;
}

void drawTextView(std::string_view text) {
  ImGui::TextUnformatted(text.data(), text.data() + text.size());
}

[[nodiscard]] std::optional<float>
findPassCpuTimingMs(const RenderGraphTelemetrySnapshot &snapshot,
                    uint32_t orderedPassIndex) {
  if (orderedPassIndex < snapshot.execution.passTimings.size()) {
    const RenderGraphPassExecutionTiming &timing =
        snapshot.execution.passTimings[orderedPassIndex];
    if (timing.orderedPassIndex == orderedPassIndex) {
      return timing.cpuTimeMs;
    }
  }
  for (const RenderGraphPassExecutionTiming &timing :
       snapshot.execution.passTimings) {
    if (timing.orderedPassIndex == orderedPassIndex) {
      return timing.cpuTimeMs;
    }
  }
  return std::nullopt;
}

[[nodiscard]] const GpuTimingReport::PassTiming *
claimGpuPassTiming(const GpuTimingReport &report, std::vector<uint8_t> &claimed,
                   std::string_view passName, uint32_t orderedPassIndex,
                   uint32_t passCount) {
  const auto matchesName = [passName](
                               const GpuTimingReport::PassTiming &timing) {
    return std::string_view(timing.debugName.data(), timing.debugName.size()) ==
           passName;
  };

  if (orderedPassIndex < report.passTimings.size() &&
      orderedPassIndex < claimed.size() && claimed[orderedPassIndex] == 0u &&
      report.passTimings.size() == passCount &&
      matchesName(report.passTimings[orderedPassIndex])) {
    claimed[orderedPassIndex] = 1u;
    return &report.passTimings[orderedPassIndex];
  }

  for (size_t i = 0u; i < report.passTimings.size() && i < claimed.size();
       ++i) {
    if (claimed[i] != 0u || !matchesName(report.passTimings[i])) {
      continue;
    }
    claimed[i] = 1u;
    return &report.passTimings[i];
  }

  return nullptr;
}

[[nodiscard]] PassMetricAggregate &
ensurePassMetricAggregate(PassMetricsUiState &state, uint32_t orderedPassIndex,
                          std::string_view passName) {
  for (PassMetricAggregate &aggregate : state.aggregates) {
    if (aggregate.name == passName) {
      aggregate.orderedPassIndex =
          std::min(aggregate.orderedPassIndex, orderedPassIndex);
      return aggregate;
    }
  }

  PassMetricAggregate aggregate{};
  aggregate.orderedPassIndex = orderedPassIndex;
  aggregate.name.assign(passName.data(), passName.size());
  state.aggregates.push_back(std::move(aggregate));
  return state.aggregates.back();
}

void addPassMetricCpuSample(PassMetricAggregate &aggregate, float timeMs) {
  if (!std::isfinite(timeMs) || timeMs < 0.0f) {
    return;
  }
  aggregate.cpuMinMs = std::min(aggregate.cpuMinMs, timeMs);
  aggregate.cpuMaxMs = std::max(aggregate.cpuMaxMs, timeMs);
  aggregate.cpuSumMs += static_cast<double>(timeMs);
  ++aggregate.cpuSampleCount;
}

void addPassMetricGpuSample(PassMetricAggregate &aggregate, float timeMs) {
  if (!std::isfinite(timeMs) || timeMs < 0.0f) {
    return;
  }
  aggregate.gpuMinMs = std::min(aggregate.gpuMinMs, timeMs);
  aggregate.gpuMaxMs = std::max(aggregate.gpuMaxMs, timeMs);
  aggregate.gpuSumMs += static_cast<double>(timeMs);
  ++aggregate.gpuSampleCount;
}

[[nodiscard]] float passMetricSortMax(const PassMetricAggregate &aggregate) {
  float maxMs = 0.0f;
  if (aggregate.cpuSampleCount > 0u) {
    maxMs = std::max(maxMs, aggregate.cpuMaxMs);
  }
  if (aggregate.gpuSampleCount > 0u) {
    maxMs = std::max(maxMs, aggregate.gpuMaxMs);
  }
  return maxMs;
}

[[nodiscard]] float passMetricAverage(double sum, uint32_t sampleCount) {
  return sampleCount > 0u ? static_cast<float>(sum / sampleCount) : 0.0f;
}

void drawMetricValue(float value, uint32_t sampleCount) {
  if (sampleCount == 0u) {
    ImGui::TextUnformatted("--");
    return;
  }
  ImGui::Text("%.3f", value);
}

void recordCpuPassMetricSamples(PassMetricsUiState &state,
                                const RenderGraphTelemetrySnapshot &snapshot) {
  const uint64_t frameIndex = snapshot.summary.frameIndex;
  if (frameIndex < state.recordingStartFrameIndex ||
      frameIndex == state.lastRecordedCpuFrameIndex) {
    return;
  }

  const uint32_t passCount =
      static_cast<uint32_t>(snapshot.compile.orderedPassIndices.size());
  for (uint32_t orderedPassIndex = 0u; orderedPassIndex < passCount;
       ++orderedPassIndex) {
    const std::optional<float> cpuMs =
        findPassCpuTimingMs(snapshot, orderedPassIndex);
    if (!cpuMs.has_value()) {
      continue;
    }
    const uint32_t declaredPassIndex =
        snapshot.compile.orderedPassIndices[orderedPassIndex];
    const std::string_view passName =
        resolveTelemetryPassName(snapshot, declaredPassIndex);
    PassMetricAggregate &aggregate =
        ensurePassMetricAggregate(state, orderedPassIndex, passName);
    addPassMetricCpuSample(aggregate, *cpuMs);
  }

  state.lastRecordedCpuFrameIndex = frameIndex;
  ++state.recordedCpuFrameCount;
}

void recordGpuPassMetricReport(PassMetricsUiState &state,
                               const GpuTimingReport &report) {
  if (report.passTimings.empty()) {
    return;
  }

  const uint64_t frameIndex = report.passTimings.front().sourceFrameIndex;
  if (frameIndex < state.recordingStartFrameIndex ||
      frameIndex == state.lastRecordedGpuFrameIndex) {
    return;
  }

  for (uint32_t orderedPassIndex = 0u;
       orderedPassIndex < report.passTimings.size(); ++orderedPassIndex) {
    const GpuTimingReport::PassTiming &timing =
        report.passTimings[orderedPassIndex];
    const std::string_view passName(timing.debugName.data(),
                                    timing.debugName.size());
    PassMetricAggregate &aggregate =
        ensurePassMetricAggregate(state, orderedPassIndex, passName);
    addPassMetricGpuSample(aggregate, timing.timeMs);
  }

  state.lastRecordedGpuFrameIndex = frameIndex;
  ++state.recordedGpuReportCount;
}

void collectPassMetricRecordingSamples(
    PassMetricsUiState &state, const RenderGraphTelemetryService *telemetry,
    GPUDevice &gpu) {
  const RenderGraphTelemetrySnapshot *snapshot =
      telemetry != nullptr ? telemetry->latestSnapshot() : nullptr;
  if (snapshot != nullptr) {
    recordCpuPassMetricSamples(state, *snapshot);
  }

  std::array<GpuTimingReport, 64> reports{};
  for (;;) {
    const size_t reportCount =
        gpu.drainCompletedGpuTimingReports(std::span<GpuTimingReport>(reports));
    for (size_t i = 0u; i < reportCount; ++i) {
      recordGpuPassMetricReport(state, reports[i]);
      reports[i] = GpuTimingReport{};
    }
    if (reportCount < reports.size()) {
      break;
    }
  }
}

void startPassMetricRecording(PassMetricsUiState &state,
                              const RenderGraphTelemetryService *telemetry) {
  state.aggregates.clear();
  state.recordedCpuFrameCount = 0u;
  state.recordedGpuReportCount = 0u;
  state.lastRecordedCpuFrameIndex = std::numeric_limits<uint64_t>::max();
  state.lastRecordedGpuFrameIndex = std::numeric_limits<uint64_t>::max();
  state.showRecordingWindow = false;
  if (const RenderGraphTelemetrySnapshot *snapshot =
          telemetry != nullptr ? telemetry->latestSnapshot() : nullptr;
      snapshot != nullptr) {
    state.recordingStartFrameIndex = snapshot->summary.frameIndex;
  } else {
    state.recordingStartFrameIndex = 0u;
  }
  state.recording = true;
}

void stopPassMetricRecording(PassMetricsUiState &state) {
  state.recording = false;
  state.showRecordingWindow = !state.aggregates.empty();
}

void refreshPassMetrics(PassMetricsUiState &state,
                        const RenderGraphTelemetryService *telemetry,
                        GPUDevice &gpu) {
  state.rows.clear();
  state.renderGraphFrameIndex = std::numeric_limits<uint64_t>::max();
  state.gpuFrameIndex = std::numeric_limits<uint64_t>::max();
  state.gpuPassTimingCount = 0u;

  const RenderGraphTelemetrySnapshot *snapshot =
      telemetry != nullptr ? telemetry->latestSnapshot() : nullptr;
  if (snapshot == nullptr) {
    return;
  }

  const GpuTimingReport gpuReport = gpu.getLatestCompletedGpuTimingReport();
  state.renderGraphFrameIndex = snapshot->summary.frameIndex;
  state.gpuPassTimingCount =
      static_cast<uint32_t>(gpuReport.passTimings.size());
  if (!gpuReport.passTimings.empty()) {
    state.gpuFrameIndex = gpuReport.passTimings.front().sourceFrameIndex;
  }

  const uint32_t passCount =
      static_cast<uint32_t>(snapshot->compile.orderedPassIndices.size());
  std::vector<uint8_t> claimedGpuTimings(gpuReport.passTimings.size(), 0u);
  state.rows.reserve(passCount);

  for (uint32_t orderedPassIndex = 0u; orderedPassIndex < passCount;
       ++orderedPassIndex) {
    const uint32_t declaredPassIndex =
        snapshot->compile.orderedPassIndices[orderedPassIndex];
    const std::string_view passName =
        resolveTelemetryPassName(*snapshot, declaredPassIndex);

    PassMetricsRow row{};
    row.name.assign(passName.data(), passName.size());
    if (const std::optional<float> cpuMs =
            findPassCpuTimingMs(*snapshot, orderedPassIndex)) {
      row.cpuTimeMs = *cpuMs;
      row.hasCpuTiming = true;
    }

    if (const GpuTimingReport::PassTiming *gpuTiming =
            claimGpuPassTiming(gpuReport, claimedGpuTimings, passName,
                               orderedPassIndex, passCount);
        gpuTiming != nullptr) {
      row.gpuTimeMs = gpuTiming->timeMs;
      row.hasGpuTiming = true;
    }

    state.rows.push_back(std::move(row));
  }
}

void drawPassMetricRecordingWindow(PassMetricsUiState &state) {
  ImGui::Text("CPU Frames: %u", state.recordedCpuFrameCount);
  ImGui::SameLine();
  ImGui::Text("GPU Reports: %u", state.recordedGpuReportCount);

  if (state.aggregates.empty()) {
    ImGui::TextUnformatted("<none>");
    return;
  }

  std::vector<size_t> sortedIndices(state.aggregates.size());
  std::iota(sortedIndices.begin(), sortedIndices.end(), size_t{0});
  std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&](size_t lhs, size_t rhs) {
              const PassMetricAggregate &a = state.aggregates[lhs];
              const PassMetricAggregate &b = state.aggregates[rhs];
              return passMetricSortMax(a) > passMetricSortMax(b);
            });

  if (!ImGui::BeginTable("PassesMetricsRecordingTable", 10,
                         ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                             ImGuiTableFlags_Resizable |
                             ImGuiTableFlags_ScrollY,
                         ImVec2(0.0f, 520.0f))) {
    return;
  }

  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed, 56.0f);
  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("CPU Min", ImGuiTableColumnFlags_WidthFixed, 78.0f);
  ImGui::TableSetupColumn("CPU Avg", ImGuiTableColumnFlags_WidthFixed, 78.0f);
  ImGui::TableSetupColumn("CPU Max", ImGuiTableColumnFlags_WidthFixed, 78.0f);
  ImGui::TableSetupColumn("GPU Min", ImGuiTableColumnFlags_WidthFixed, 78.0f);
  ImGui::TableSetupColumn("GPU Avg", ImGuiTableColumnFlags_WidthFixed, 78.0f);
  ImGui::TableSetupColumn("GPU Max", ImGuiTableColumnFlags_WidthFixed, 78.0f);
  ImGui::TableSetupColumn("CPU N", ImGuiTableColumnFlags_WidthFixed, 62.0f);
  ImGui::TableSetupColumn("GPU N", ImGuiTableColumnFlags_WidthFixed, 62.0f);
  ImGui::TableHeadersRow();

  for (const size_t aggregateIndex : sortedIndices) {
    const PassMetricAggregate &aggregate = state.aggregates[aggregateIndex];
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::Text("%u", aggregate.orderedPassIndex);
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(aggregate.name.c_str());
    ImGui::TableNextColumn();
    drawMetricValue(aggregate.cpuMinMs, aggregate.cpuSampleCount);
    ImGui::TableNextColumn();
    drawMetricValue(
        passMetricAverage(aggregate.cpuSumMs, aggregate.cpuSampleCount),
        aggregate.cpuSampleCount);
    ImGui::TableNextColumn();
    drawMetricValue(aggregate.cpuMaxMs, aggregate.cpuSampleCount);
    ImGui::TableNextColumn();
    drawMetricValue(aggregate.gpuMinMs, aggregate.gpuSampleCount);
    ImGui::TableNextColumn();
    drawMetricValue(
        passMetricAverage(aggregate.gpuSumMs, aggregate.gpuSampleCount),
        aggregate.gpuSampleCount);
    ImGui::TableNextColumn();
    drawMetricValue(aggregate.gpuMaxMs, aggregate.gpuSampleCount);
    ImGui::TableNextColumn();
    ImGui::Text("%u", aggregate.cpuSampleCount);
    ImGui::TableNextColumn();
    ImGui::Text("%u", aggregate.gpuSampleCount);
  }

  ImGui::EndTable();
}

void drawPassMetricsWindow(PassMetricsUiState &state,
                           const RenderGraphTelemetryService *telemetry,
                           GPUDevice &gpu) {
  if (state.recording) {
    collectPassMetricRecordingSamples(state, telemetry, gpu);
  }

  const double nowSeconds = ImGui::GetTime();
  if (state.rows.empty() || nowSeconds - state.lastUpdateSeconds >=
                                kPassMetricsUpdateIntervalSeconds) {
    refreshPassMetrics(state, telemetry, gpu);
    state.lastUpdateSeconds = nowSeconds;
  }

  if (telemetry == nullptr || !telemetry->hasSnapshot()) {
    ImGui::TextUnformatted("No render-graph telemetry has been captured yet.");
    return;
  }

  if (state.recording) {
    if (ImGui::Button("Stop Recording##PassMetrics")) {
      stopPassMetricRecording(state);
    }
  } else {
    if (ImGui::Button("Start Recording##PassMetrics")) {
      startPassMetricRecording(state, telemetry);
    }
  }
  if (!state.aggregates.empty()) {
    ImGui::SameLine();
    if (ImGui::Button("Open Recording##PassMetrics")) {
      state.showRecordingWindow = true;
    }
  }
  ImGui::SameLine();
  ImGui::Text("CPU Frames: %u  GPU Reports: %u", state.recordedCpuFrameCount,
              state.recordedGpuReportCount);

  ImGui::Separator();
  ImGui::Text("Render Graph Frame: %llu",
              static_cast<unsigned long long>(state.renderGraphFrameIndex));
  ImGui::SameLine();
  if (state.gpuFrameIndex == std::numeric_limits<uint64_t>::max()) {
    ImGui::TextUnformatted("GPU Frame: pending");
  } else {
    ImGui::Text("GPU Frame: %llu",
                static_cast<unsigned long long>(state.gpuFrameIndex));
  }
  ImGui::SameLine();
  ImGui::Text("GPU Passes: %u", state.gpuPassTimingCount);

  if (state.rows.empty()) {
    ImGui::TextUnformatted("<none>");
    return;
  }

  if (!ImGui::BeginTable("PassesMetricsTable", 3,
                         ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                             ImGuiTableFlags_Resizable |
                             ImGuiTableFlags_ScrollY,
                         ImVec2(0.0f, 420.0f))) {
    return;
  }

  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("CPU", ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableSetupColumn("GPU", ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableHeadersRow();

  for (const PassMetricsRow &row : state.rows) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(row.name.c_str());
    ImGui::TableNextColumn();
    if (row.hasCpuTiming) {
      ImGui::Text("%.3f ms", row.cpuTimeMs);
    } else {
      ImGui::TextUnformatted("--");
    }
    ImGui::TableNextColumn();
    if (row.hasGpuTiming) {
      ImGui::Text("%.3f ms", row.gpuTimeMs);
    } else {
      ImGui::TextUnformatted("--");
    }
  }

  ImGui::EndTable();
}

void drawTelemetrySummary(
    const RenderGraphTelemetrySnapshot::Summary &summary) {
  if (!ImGui::BeginTable("RenderGraphTelemetrySummary", 2,
                         ImGuiTableFlags_BordersInnerV |
                             ImGuiTableFlags_SizingStretchProp)) {
    return;
  }

  const auto drawRow = [](const char *label, auto value) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::Text("%llu", static_cast<unsigned long long>(value));
  };

  drawRow("Frame Index", summary.frameIndex);
  drawRow("Declared Passes", summary.declaredPassCount);
  drawRow("Culled Passes", summary.culledPassCount);
  drawRow("Root Passes", summary.rootPassCount);
  drawRow("Pass Count", summary.passCount);
  drawRow("Edge Count", summary.edgeCount);
  drawRow("Imported Textures", summary.importedTextures);
  drawRow("Transient Textures", summary.transientTextures);
  drawRow("Imported Buffers", summary.importedBuffers);
  drawRow("Transient Buffers", summary.transientBuffers);
  drawRow("Texture Lifetimes", summary.transientTextureLifetimeCount);
  drawRow("Buffer Lifetimes", summary.transientBufferLifetimeCount);
  drawRow("Texture Physicals", summary.transientTexturePhysicalCount);
  drawRow("Buffer Physicals", summary.transientBufferPhysicalCount);
  drawRow("Resolved Dependency Slots",
          summary.resolvedDependencyBufferSlotCount);
  drawRow("Resolved Pre-Dispatch Slots",
          summary.resolvedPreDispatchDependencyBufferSlotCount);
  drawRow("Unresolved Texture Bindings", summary.unresolvedTextureBindingCount);
  drawRow("Unresolved Dependency Bindings",
          summary.unresolvedDependencyBufferBindingCount);
  drawRow("Unresolved Pre-Dispatch Bindings",
          summary.unresolvedPreDispatchDependencyBufferBindingCount);
  drawRow("Unresolved Draw Bindings", summary.unresolvedDrawBufferBindingCount);

  ImGui::EndTable();
}

template <typename Fn>
void drawTelemetryTableSection(const char *header, const char *tableId,
                               int columns, bool hasRows, float height,
                               Fn &&drawRows) {
  if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }
  if (!hasRows) {
    ImGui::TextUnformatted("<none>");
    return;
  }

  if (ImGui::BeginTable(tableId, columns,
                        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
                        ImVec2(0.0f, height))) {
    drawRows();
    ImGui::EndTable();
  }
}

void drawRenderGraphTelemetryWindow(RenderGraphTelemetryUiState &state,
                                    RenderGraphTelemetryService *telemetry,
                                    void *ownerWindowHandle) {
  syncTelemetryDumpPath(state, telemetry);

  ImGui::InputText("Output .txt", state.outputPath.data(),
                   state.outputPath.size());
  ImGui::SameLine();
  if (ImGui::Button("Browse...##RenderGraphTelemetry")) {
    static constexpr std::array<FileDialogFilter, 2> kTelemetryFilters = {
        FileDialogFilter{"Text Files (*.txt)", "*.txt"},
        FileDialogFilter{"All Files (*.*)", "*.*"},
    };
    SaveFileRequest request{};
    request.title = "Save Render Graph Telemetry";
    request.filters = kTelemetryFilters;
    request.defaultExtension = "txt";
    request.initialPath = state.outputPath.data();
    request.ownerWindowHandle = ownerWindowHandle;
    if (const auto selectedPath = state.fileDialog.saveFile(request)) {
      setPathText(state.outputPath, selectedPath->generic_string());
      state.initializedOutputPath = true;
    }
  }

  const bool canDump = telemetry != nullptr && telemetry->hasSnapshot();
  if (!canDump) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("Dump Current Telemetry")) {
    state.status.clear();
    state.error.clear();
    auto dumpResult = telemetry->writeLatestTextDump(state.outputPath.data());
    if (dumpResult.hasError()) {
      state.error = dumpResult.error();
    } else {
      state.status =
          std::string("Wrote telemetry to ") + state.outputPath.data();
    }
  }
  if (!canDump) {
    ImGui::EndDisabled();
  }

  if (!state.status.empty()) {
    ImGui::Spacing();
    ImGui::TextUnformatted(state.status.c_str());
  }
  if (!state.error.empty()) {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                       state.error.c_str());
  }

  const RenderGraphTelemetrySnapshot *snapshot =
      telemetry != nullptr ? telemetry->latestSnapshot() : nullptr;
  if (snapshot == nullptr) {
    ImGui::Spacing();
    ImGui::TextUnformatted("No render-graph telemetry has been captured yet.");
    return;
  }

  ImGui::Separator();
  drawTelemetrySummary(snapshot->summary);

  drawTelemetryTableSection(
      "Passes", "RenderGraphTelemetryPasses", 2,
      !snapshot->compile.passDebugNames.empty(), 160.0f, [&]() {
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed,
                                64.0f);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (uint32_t passIndex = 0;
             passIndex < snapshot->compile.passDebugNames.size(); ++passIndex) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("%u", passIndex);
          ImGui::TableNextColumn();
          drawTextView(resolveTelemetryPassName(*snapshot, passIndex));
        }
      });

  drawTelemetryTableSection(
      "Edges", "RenderGraphTelemetryEdges", 4, !snapshot->compile.edges.empty(),
      140.0f, [&]() {
        ImGui::TableSetupColumn("Before", ImGuiTableColumnFlags_WidthFixed,
                                60.0f);
        ImGui::TableSetupColumn("Before Name",
                                ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("After", ImGuiTableColumnFlags_WidthFixed,
                                60.0f);
        ImGui::TableSetupColumn("After Name",
                                ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (const auto &edge : snapshot->compile.edges) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("%u", edge.before);
          ImGui::TableNextColumn();
          drawTextView(resolveTelemetryPassName(*snapshot, edge.before));
          ImGui::TableNextColumn();
          ImGui::Text("%u", edge.after);
          ImGui::TableNextColumn();
          drawTextView(resolveTelemetryPassName(*snapshot, edge.after));
        }
      });

  drawTelemetryTableSection(
      "Execution Order", "RenderGraphTelemetryExecution", 3,
      !snapshot->compile.orderedPassIndices.empty(), 140.0f, [&]() {
        ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed,
                                60.0f);
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed,
                                60.0f);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (uint32_t rank = 0;
             rank < snapshot->compile.orderedPassIndices.size(); ++rank) {
          const uint32_t passIndex = snapshot->compile.orderedPassIndices[rank];
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("%u", rank);
          ImGui::TableNextColumn();
          ImGui::Text("%u", passIndex);
          ImGui::TableNextColumn();
          drawTextView(resolveTelemetryPassName(*snapshot, passIndex));
        }
      });

  drawTelemetryTableSection(
      "Transient Lifetimes", "RenderGraphTelemetryTextureLifetimes", 4,
      !snapshot->compile.transientTextureLifetimes.empty() ||
          !snapshot->compile.transientBufferLifetimes.empty(),
      180.0f, [&]() {
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("First", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Last", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableHeadersRow();
        for (const auto &lifetime :
             snapshot->compile.transientTextureLifetimes) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.resourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.firstExecutionIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.lastExecutionIndex);
        }
        for (const auto &lifetime :
             snapshot->compile.transientBufferLifetimes) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.resourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.firstExecutionIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", lifetime.lastExecutionIndex);
        }
      });

  drawTelemetryTableSection(
      "Allocations", "RenderGraphTelemetryAllocations", 4,
      !snapshot->compile.transientTextureAllocations.empty() ||
          !snapshot->compile.transientBufferAllocations.empty(),
      180.0f, [&]() {
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Physical", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Map", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (const auto &allocation :
             snapshot->compile.transientTextureAllocations) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", allocation.resourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", allocation.allocationIndex);
          ImGui::TableNextColumn();
          ImGui::Text("tex[%u] -> phys[%u]", allocation.resourceIndex,
                      allocation.allocationIndex);
        }
        for (const auto &allocation :
             snapshot->compile.transientBufferAllocations) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", allocation.resourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", allocation.allocationIndex);
          ImGui::TableNextColumn();
          ImGui::Text("buf[%u] -> phys[%u]", allocation.resourceIndex,
                      allocation.allocationIndex);
        }
      });

  drawTelemetryTableSection(
      "Allocation Maps", "RenderGraphTelemetryAllocationMaps", 3,
      !snapshot->compile.transientTextureAllocationByResource.empty() ||
          !snapshot->compile.transientBufferAllocationByResource.empty(),
      180.0f, [&]() {
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Physical", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableHeadersRow();
        for (uint32_t i = 0;
             i < snapshot->compile.transientTextureAllocationByResource.size();
             ++i) {
          const uint32_t physical =
              snapshot->compile.transientTextureAllocationByResource[i];
          if (physical == UINT32_MAX) {
            continue;
          }
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical);
        }
        for (uint32_t i = 0;
             i < snapshot->compile.transientBufferAllocationByResource.size();
             ++i) {
          const uint32_t physical =
              snapshot->compile.transientBufferAllocationByResource[i];
          if (physical == UINT32_MAX) {
            continue;
          }
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical);
        }
      });

  drawTelemetryTableSection(
      "Physical Allocations", "RenderGraphTelemetryPhysicalAllocations", 5,
      !snapshot->compile.transientTexturePhysicalAllocations.empty() ||
          !snapshot->compile.transientBufferPhysicalAllocations.empty(),
      200.0f, [&]() {
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Physical", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Representative",
                                ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Format/Usage",
                                ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Details", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (const auto &physical :
             snapshot->compile.transientTexturePhysicalAllocations) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical.allocationIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical.representativeResourceIndex);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted(formatDisplayName(physical.desc.format));
          ImGui::TableNextColumn();
          ImGui::Text("%ux%ux%u layers=%u samples=%u mips=%u",
                      physical.desc.dimensions.width,
                      physical.desc.dimensions.height,
                      physical.desc.dimensions.depth, physical.desc.numLayers,
                      physical.desc.numSamples, physical.desc.numMipLevels);
        }
        for (const auto &physical :
             snapshot->compile.transientBufferPhysicalAllocations) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical.allocationIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", physical.representativeResourceIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", static_cast<uint32_t>(physical.desc.usage));
          ImGui::TableNextColumn();
          ImGui::Text("storage=%u size=%zu",
                      static_cast<uint32_t>(physical.desc.storage),
                      physical.desc.size);
        }
      });

  drawTelemetryTableSection(
      "Bindings", "RenderGraphTelemetryBindings", 5,
      !snapshot->compile.unresolvedTextureBindings.empty() ||
          !snapshot->compile.resolvedDependencyBuffers.empty() ||
          !snapshot->compile.unresolvedDependencyBufferBindings.empty() ||
          !snapshot->compile.unresolvedPreDispatchDependencyBufferBindings
               .empty() ||
          !snapshot->compile.unresolvedDrawBufferBindings.empty(),
      220.0f, [&]() {
        ImGui::TableSetupColumn("Section", ImGuiTableColumnFlags_WidthFixed,
                                110.0f);
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed,
                                60.0f);
        ImGui::TableSetupColumn("Item", ImGuiTableColumnFlags_WidthFixed,
                                80.0f);
        ImGui::TableSetupColumn("Target", ImGuiTableColumnFlags_WidthFixed,
                                110.0f);
        ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (const auto &binding :
             snapshot->compile.unresolvedTextureBindings) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("pass_tex");
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.orderedPassIndex);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("-");
          ImGui::TableNextColumn();
          ImGui::TextUnformatted(passTextureBindingTargetName(binding.target));
          ImGui::TableNextColumn();
          ImGui::Text("tex[%u]", binding.textureResourceIndex);
        }
        for (uint32_t slot = 0;
             slot < snapshot->compile.resolvedDependencyBuffers.size();
             ++slot) {
          const BufferHandle handle =
              snapshot->compile.resolvedDependencyBuffers[slot];
          if (!nuri::isValid(handle)) {
            continue;
          }
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("dep_slot");
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("-");
          ImGui::TableNextColumn();
          ImGui::Text("%u", slot);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("resolved");
          ImGui::TableNextColumn();
          ImGui::Text("handle=(%u,%u)", handle.index, handle.generation);
        }
        for (const auto &binding :
             snapshot->compile.unresolvedDependencyBufferBindings) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("dep_buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.orderedPassIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.dependencyBufferIndex);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("dependency");
          ImGui::TableNextColumn();
          ImGui::Text("buf[%u]", binding.bufferResourceIndex);
        }
        for (const auto &binding :
             snapshot->compile.unresolvedPreDispatchDependencyBufferBindings) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("pre_dep");
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.orderedPassIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u/%u", binding.preDispatchIndex,
                      binding.dependencyBufferIndex);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("pre_dispatch");
          ImGui::TableNextColumn();
          ImGui::Text("buf[%u]", binding.bufferResourceIndex);
        }
        for (const auto &binding :
             snapshot->compile.unresolvedDrawBufferBindings) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("draw_buf");
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.orderedPassIndex);
          ImGui::TableNextColumn();
          ImGui::Text("%u", binding.drawIndex);
          ImGui::TableNextColumn();
          ImGui::TextUnformatted(drawBufferBindingTargetName(binding.target));
          ImGui::TableNextColumn();
          ImGui::Text("buf[%u]", binding.bufferResourceIndex);
        }
      });

  drawTelemetryTableSection(
      "Ranges", "RenderGraphTelemetryRanges", 4,
      !snapshot->compile.dependencyBufferRangesByPass.empty() ||
          !snapshot->compile.preDispatchRangesByPass.empty() ||
          !snapshot->compile.preDispatchDependencyRanges.empty() ||
          !snapshot->compile.drawRangesByPass.empty(),
      220.0f, [&]() {
        ImGui::TableSetupColumn("Section", ImGuiTableColumnFlags_WidthFixed,
                                110.0f);
        ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Offset", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed,
                                70.0f);
        ImGui::TableHeadersRow();
        for (uint32_t i = 0;
             i < snapshot->compile.dependencyBufferRangesByPass.size(); ++i) {
          const auto &range = snapshot->compile.dependencyBufferRangesByPass[i];
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("dep_pass");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.offset);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.count);
        }
        for (uint32_t i = 0;
             i < snapshot->compile.preDispatchRangesByPass.size(); ++i) {
          const auto &range = snapshot->compile.preDispatchRangesByPass[i];
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("pre_pass");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.offset);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.count);
        }
        for (uint32_t i = 0;
             i < snapshot->compile.preDispatchDependencyRanges.size(); ++i) {
          const auto &range = snapshot->compile.preDispatchDependencyRanges[i];
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("pre_dep");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.offset);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.count);
        }
        for (uint32_t i = 0; i < snapshot->compile.drawRangesByPass.size();
             ++i) {
          const auto &range = snapshot->compile.drawRangesByPass[i];
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextUnformatted("draw_pass");
          ImGui::TableNextColumn();
          ImGui::Text("%u", i);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.offset);
          ImGui::TableNextColumn();
          ImGui::Text("%u", range.count);
        }
      });
}

void setDockspaceWindowPlacement(const ImGuiViewport *viewport) {
  if (!viewport) {
    return;
  }
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);
}

void setLogWindowPlacementWithoutDock(const ImGuiViewport *viewport) {
  if (!viewport) {
    return;
  }
  const float height = std::max(180.0f, viewport->WorkSize.y * 0.25f);
  const ImVec2 position(viewport->WorkPos.x,
                        viewport->WorkPos.y + viewport->WorkSize.y - height);
  const ImVec2 size(viewport->WorkSize.x, height);
  ImGui::SetNextWindowPos(position, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(size, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowViewport(viewport->ID);
}

[[nodiscard]] std::optional<GpuFrameTimeSample>
overlayGpuFrameTimeSample(const GpuTimingReport &report) {
  if (!hasGpuTimingScope(report, GpuTimingScope::WholeFrame)) {
    return std::nullopt;
  }
  return GpuFrameTimeSample{
      .sourceFrame = report.wholeFrameSourceFrameIndex,
      .milliseconds = report.wholeFrameTimeMs,
  };
}

void drawFpsOverlay(LinearGraph &fpsGraph, LinearGraph &frametimeGraph,
                    const FrameTimeDisplayValues &frameTimes,
                    const RenderFrameMetrics &frameMetrics,
                    const RenderSettings &renderSettings,
                    const TelemetryOverlayUiState &telemetryState,
                    float overlayRightBoundaryX) {
  if (!telemetryState.overlayEnabled) {
    return;
  }
  if (const ImGuiViewport *viewport = ImGui::GetMainViewport()) {
    const float viewportRight = viewport->WorkPos.x + viewport->WorkSize.x;
    const float rightBoundary =
        overlayRightBoundaryX > 0.0f
            ? std::min(overlayRightBoundaryX, viewportRight) - 15.0f
            : viewportRight - 15.0f;
    ImGui::SetNextWindowPos({rightBoundary, viewport->WorkPos.y + 15.0f},
                            ImGuiCond_Always, {1.0f, 0.0f});
  }
  ImGui::SetNextWindowBgAlpha(0.30f);
  const float overlayHeight =
      telemetryState.showGraphs ? kMetricGraphWindowHeight : 158.0f;
  ImGui::SetNextWindowSize(ImVec2(kMetricGraphWindowWidth, overlayHeight),
                           ImGuiCond_Always);
  if (ImGui::Begin("##FPS", nullptr,
                   ImGuiWindowFlags_NoDecoration |
                       ImGuiWindowFlags_NoSavedSettings |
                       ImGuiWindowFlags_NoFocusOnAppearing |
                       ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove)) {
    const float fps = frameTimes.framesPerSecond;
    bool drewStats = false;
    if (telemetryState.showFpsMs) {
      ImGui::Text("FPS: %i", static_cast<int>(fps));
      if (frameTimes.cpuAvailable) {
        ImGui::Text("CPU: %.2f ms", frameTimes.cpuMilliseconds);
      } else {
        ImGui::TextUnformatted("CPU: sampling");
      }
      if (frameTimes.gpuAvailable) {
        ImGui::Text("GPU: %.2f ms", frameTimes.gpuMilliseconds);
      } else {
        ImGui::TextUnformatted("GPU: timing pending");
      }
      drewStats = true;
    }
    const ResolvedGeometryWorkMetrics geometryWork =
        resolveGeometryWorkMetrics(frameMetrics);
    if (telemetryState.showInstanceStats) {
      if (geometryWork.mainReadbackAvailable) {
        ImGui::Text("Visible instances: %u / %u", geometryWork.visibleInstances,
                    geometryWork.instanceCandidates);
      } else {
        ImGui::TextUnformatted("Visible instances: GPU readback pending");
      }
      drewStats = true;
    }
    if (telemetryState.showDrawTessStats) {
      if (frameMetrics.opaque.meshletModeActive != 0u) {
        if (geometryWork.meshletReadbackAvailable) {
          ImGui::Text("Emitted meshlets: %u / %u", geometryWork.emittedMeshlets,
                      geometryWork.meshletCandidates);
        } else {
          ImGui::TextUnformatted("Emitted meshlets: GPU readback pending");
        }
      } else {
        ImGui::Text("Submitted draws: %u (tess: %u)",
                    frameMetrics.opaque.instancedDraws,
                    frameMetrics.opaque.tessellatedDraws);
      }
      drewStats = true;
    }
    if (telemetryState.showIndirectStats) {
      if (geometryWork.indirectReadbackAvailable) {
        ImGui::Text("Visible indirect: %u / %u",
                    geometryWork.visibleIndirectCommands,
                    geometryWork.indirectCommands);
      } else if (frameMetrics.opaque.meshletModeActive == 0u) {
        ImGui::Text("Submitted indirect: %u calls / %u cmds",
                    frameMetrics.opaque.indirectDrawCalls,
                    frameMetrics.opaque.indirectCommands);
      }
      drewStats = true;
    }
    if (telemetryState.showDebugDrawStats) {
      ImGui::Text("Debug draws: %u (fallback: %u)",
                  frameMetrics.opaque.debugOverlayDraws,
                  frameMetrics.opaque.debugOverlayFallbackDraws);
      drewStats = true;
    }
    if (telemetryState.showPatchHeatmap) {
      ImGui::Text("Patch heatmap: %u",
                  frameMetrics.opaque.debugPatchHeatmapDraws);
      drewStats = true;
    }
    if (telemetryState.showDispatchStats) {
      ImGui::Text("Compute: %u dispatches (X=%u)",
                  frameMetrics.opaque.computeDispatches,
                  frameMetrics.opaque.computeDispatchX);
      const OpaqueFrameMetrics &opaqueMetrics = frameMetrics.opaque;
      const std::string_view activeRoute = activeGeometryRouteLabel(
          renderSettings.opaque.meshletMode, opaqueMetrics);
      if (opaqueMetrics.meshletModeActive != 0u) {
        ImGui::Text("Route: %.*s (%u packets)",
                    static_cast<int>(activeRoute.size()), activeRoute.data(),
                    opaqueMetrics.meshletDispatches);
        if (geometryWork.meshletReadbackAvailable) {
          ImGui::Text("Executed task groups: %u / %u",
                      geometryWork.executedMeshletTaskGroups,
                      opaqueMetrics.meshletTaskGroups);
        } else {
          ImGui::TextUnformatted("Task groups: GPU readback pending");
        }
      } else {
        ImGui::TextWrapped("Route: %.*s", static_cast<int>(activeRoute.size()),
                           activeRoute.data());
      }
      drewStats = true;
    }

    if (telemetryState.showGraphs) {
      if (drewStats) {
        ImGui::Separator();
      }
      const float availableGraphHeight = ImGui::GetContentRegionAvail().y;
      const float perGraphHeight = std::max(availableGraphHeight * 0.5f, 1.0f);
      const ImVec2 itemSpacing = ImGui::GetStyle().ItemSpacing;
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                          ImVec2(itemSpacing.x, 0.0f));

      LinearGraphStyle graphStyle{
          .heightPixels = perGraphHeight,
          .lineColorRgba = IM_COL32(64, 224, 128, 255),
          .fillUnderLine = true,
      };
      fpsGraph.draw("FPS Graph##Metrics", "FPS", graphStyle);

      graphStyle.lineColorRgba = IM_COL32(255, 160, 64, 255);
      frametimeGraph.draw("CPU Frametime Graph##Metrics", "CPU Frametime (ms)",
                          graphStyle);
      ImGui::PopStyleVar();
    }
  }
  ImGui::End();
}

ImGuiWindowFlags dockspaceWindowFlags() {
  return ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
         ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground |
         ImGuiWindowFlags_NoSavedSettings;
}

#ifdef IMGUI_HAS_DOCK
struct DockLayoutState {
  ImGuiID logDockId = 0;
  ImGuiID hierarchyDockId = 0;
  ImGuiID inspectorDockId = 0;
  bool built = false;

  void ensureLayout(ImGuiID dockspaceId, const ImGuiViewport *viewport) {
    if (built || dockspaceId == 0 || !viewport) {
      return;
    }

    const auto dockNodeFlags = static_cast<ImGuiDockNodeFlags>(
        static_cast<int>(ImGuiDockNodeFlags_DockSpace) |
        static_cast<int>(ImGuiDockNodeFlags_PassthruCentralNode));

    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, dockNodeFlags);
    ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

    ImGuiID dockMain = dockspaceId;
    ImGuiID dockBottom = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Down,
                                                     0.25f, nullptr, &dockMain);
    ImGuiID dockLeft = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Left,
                                                   0.22f, nullptr, &dockMain);
    ImGuiID dockRight = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Right,
                                                    0.28f, nullptr, &dockMain);

    logDockId = dockBottom;
    hierarchyDockId = dockLeft;
    inspectorDockId = dockRight;
    ImGui::DockBuilderDockWindow(kLogWindowName, logDockId);
    ImGui::DockBuilderDockWindow(kHierarchyWindowName, hierarchyDockId);
    ImGui::DockBuilderDockWindow(kInspectorWindowName, inspectorDockId);
    ImGui::DockBuilderDockWindow(kAnimationPlayerWindowName, inspectorDockId);
    ImGui::DockBuilderFinish(dockspaceId);
    built = true;
  }
};
#endif

#ifdef IMGUI_HAS_DOCK
using MaybeDockLayoutState = DockLayoutState;
#else
struct MaybeDockLayoutState {};
#endif

} // namespace

struct ImGuiEditor::Impl {
  enum class DeferredAnimationActionType : uint8_t {
    Start,
    Pause,
    Restart,
    Seek,
    SetPrimaryClip,
    SetSecondaryClip,
    SetBlendWeight,
  };

  struct DeferredAnimationAction {
    DeferredAnimationActionType type = DeferredAnimationActionType::Start;
    float floatValue = 0.0f;
    uint32_t clipIndex = 0u;
    bool enabled = false;
  };

  Impl(Window &windowIn, GPUDevice &gpuIn, const EditorServices &services)
      : window(windowIn), gpu(gpuIn), application(services.application),
        scene(services.scene), resources(services.resources),
        renderPipeline(services.renderPipeline),
        selectionState(services.selectionState != nullptr
                           ? services.selectionState
                           : &localSelectionState),
        textSystem(services.textSystem), cameraSystem(services.cameraSystem),
        bakery(services.bakery),
        renderGraphTelemetry(services.renderGraphTelemetry),
        animationPlayer(services.animationPlayer) {
    if (application != nullptr && application->frameRateLimit() != 0u) {
      frameRateLimitFps = static_cast<int>(application->frameRateLimit());
    }
  }

  ~Impl() { resetSceneUiState(); }

  RenderableInspectorState *findRenderableInspectorState(RenderableId id) {
    for (RenderableInspectorState &state : renderableInspectorStates) {
      if (state.renderableId == id) {
        return &state;
      }
    }
    return nullptr;
  }

  RenderableInspectorState &ensureRenderableInspectorState(RenderableId id) {
    if (RenderableInspectorState *state = findRenderableInspectorState(id);
        state != nullptr) {
      return *state;
    }
    renderableInspectorStates.push_back(RenderableInspectorState{});
    RenderableInspectorState &state = renderableInspectorStates.back();
    state.renderableId = id;
    state.selectedTextureIndex = 0;
    return state;
  }

  void releaseOwnedOverride(RenderableInspectorState &state) {
    if (resources != nullptr && isValid(state.ownedOverride)) {
      resources->release(state.ownedOverride);
    }
    state.ownedOverride = kInvalidMaterialRef;
  }

  void clearRenderableOverride(RenderableInspectorState &state) {
    if (scene != nullptr) {
      (void)scene->graph().clearRenderableMaterialOverride(state.renderableId);
    }
    releaseOwnedOverride(state);
  }

  void resetSceneUiState() {
    for (RenderableInspectorState &state : renderableInspectorStates) {
      releaseOwnedOverride(state);
    }
    renderableInspectorStates.clear();
    invalidateLightEditorDraft(lightEditorDraft);
    lastObservedSelectionNode = kInvalidNodeId;
    pendingRevealSelection = false;
    suppressRevealForNextSelectionChange = false;
    hierarchyTopologyCacheValid = false;
    hierarchyStatsCacheValid = false;
    cachedHierarchyTopologyVersion = std::numeric_limits<uint64_t>::max();
    hierarchyTopologyBuildQueue.clear();
    hierarchyTopologyBuildQueueCursor = 0u;
    hierarchyTopologyBuildNode = kInvalidNodeId;
    hierarchyTopologyBuildNextChild = kInvalidNodeId;
    hierarchyTopologyBuildNodeStarted = false;
    hierarchyStatsBuilding = false;
    hierarchyStatsBuildCursor = 0u;
    cachedSelectedPathLeaf = kInvalidNodeId;
    cachedLightCount = 0u;
    cachedRenderableCount = 0u;
    hierarchyNodeTopology.clear();
    hierarchyNodeStats.clear();
    selectedPathNodeFlags.clear();
    selectedPathChildIndices.clear();
    hierarchyVisibleRows.clear();
    hierarchyNodeOpenFlags.clear();
    hierarchyOpenBatchKeys.clear();
    hierarchySelectedRowIndex = -1;
    hierarchySceneRootOpen = true;
    deferredAnimationActions.clear();
    shadowInspectResult.reset();
    if (selectionState != nullptr) {
      selectionState->clear();
    }
  }

  void applyDeferredFramePacingActions() {
    if (pendingPresentMode.has_value()) {
      const SwapchainPresentMode requested = *pendingPresentMode;
      auto result = gpu.setSwapchainPresentMode(requested);
      if (result.hasError()) {
        framePacingStatus = result.error();
        framePacingStatusIsError = true;
      } else {
        const SwapchainPresentMode active = result.value();
        framePacingStatus =
            requested == active
                ? std::format("Present mode changed to {}.",
                              presentModeLabel(active))
                : std::format("Requested {}, but the device selected {}.",
                              presentModeLabel(requested),
                              presentModeLabel(active));
        framePacingStatusIsError = false;
      }
      pendingPresentMode.reset();
    }

    if (pendingFrameRateLimit.has_value()) {
      if (application != nullptr) {
        application->setFrameRateLimit(*pendingFrameRateLimit);
      }
      pendingFrameRateLimit.reset();
    }
  }

  void applyDeferredAnimationActions() {
    if (animationPlayer == nullptr || deferredAnimationActions.empty()) {
      return;
    }
    for (const DeferredAnimationAction &action : deferredAnimationActions) {
      switch (action.type) {
      case DeferredAnimationActionType::Start:
        (void)animationPlayer->startSelectionPlayback();
        break;
      case DeferredAnimationActionType::Pause:
        (void)animationPlayer->pauseSelectionPlayback();
        break;
      case DeferredAnimationActionType::Restart:
        (void)animationPlayer->restartSelectionPlayback();
        break;
      case DeferredAnimationActionType::Seek:
        (void)animationPlayer->seekSelectionPlayback(action.floatValue);
        break;
      case DeferredAnimationActionType::SetPrimaryClip:
        (void)animationPlayer->setSelectionPrimaryClip(action.clipIndex);
        break;
      case DeferredAnimationActionType::SetSecondaryClip:
        (void)animationPlayer->setSelectionSecondaryClip(
            action.enabled ? std::optional<uint32_t>(action.clipIndex)
                           : std::nullopt);
        break;
      case DeferredAnimationActionType::SetBlendWeight:
        (void)animationPlayer->setSelectionBlendWeight(action.floatValue);
        break;
      }
    }
    deferredAnimationActions.clear();
  }

  void applyDeferredUiActions() {
    applyDeferredFramePacingActions();
    applyDeferredAnimationActions();
  }

  void validateSelectionState() {
    if (scene == nullptr || selectionState == nullptr) {
      return;
    }
    if (!isValid(selectionState->node)) {
      selectionState->clear();
      return;
    }
    if (!selectionNodeStillValid(scene->graph(), *selectionState)) {
      selectionState->clear();
      return;
    }
    if (selectionState->kind == SceneSelectionKind::NodeRenderable) {
      const Renderable *renderable =
          scene->renderable(selectionState->renderableIndex);
      if (renderable == nullptr ||
          renderable->id != selectionState->renderableId ||
          renderable->node != selectionState->node) {
        selectionState->kind = SceneSelectionKind::Node;
        selectionState->renderableId = kInvalidRenderableId;
        selectionState->renderableIndex = 0u;
      }
    } else if (selectionState->kind == SceneSelectionKind::Light) {
      LightDesc light{};
      NodeId lightNode = kInvalidNodeId;
      if (!scene->graph().getLightDesc(selectionState->lightId, light) ||
          !scene->graph().getLightNode(selectionState->lightId, lightNode) ||
          lightNode != selectionState->node) {
        selectionState->kind = SceneSelectionKind::Node;
        selectionState->lightId = kInvalidLightId;
      }
    }
  }

  void syncSelectionWindows() {
    if (selectionState == nullptr) {
      lastObservedSelectionNode = kInvalidNodeId;
      pendingRevealSelection = false;
      suppressRevealForNextSelectionChange = false;
      return;
    }

    if (selectionState->node != lastObservedSelectionNode) {
      if (isValid(selectionState->node)) {
        showHierarchyWindow = true;
        showInspectorWindow = true;
        if (animationPlayer != nullptr &&
            animationPlayer->hasAnimatedSelection()) {
          showAnimationPlayerWindow = true;
          focusAnimationPlayerWindow = true;
        }
        pendingRevealSelection = !suppressRevealForNextSelectionChange;
      }
      suppressRevealForNextSelectionChange = false;
      lastObservedSelectionNode = selectionState->node;
    }
  }

  void rebuildHierarchyFrameCache() {
    if (scene == nullptr) {
      hierarchyTopologyCacheValid = false;
      hierarchyStatsCacheValid = false;
      cachedSelectedPathLeaf = kInvalidNodeId;
      hierarchyNodeTopology.clear();
      hierarchyNodeStats.clear();
      selectedPathNodeFlags.clear();
      selectedPathChildIndices.clear();
      hierarchyVisibleRows.clear();
      hierarchyNodeOpenFlags.clear();
      hierarchyOpenBatchKeys.clear();
      hierarchySelectedRowIndex = -1;
      hierarchyTopologyBuildQueue.clear();
      hierarchyTopologyBuildQueueCursor = 0u;
      hierarchyTopologyBuildNode = kInvalidNodeId;
      hierarchyTopologyBuildNextChild = kInvalidNodeId;
      hierarchyTopologyBuildNodeStarted = false;
      hierarchyStatsBuilding = false;
      hierarchyStatsBuildCursor = 0u;
      return;
    }

    uint32_t currentLightCount = 0u;
    scene->graph().forEachLightId([&](LightId lightId) {
      NodeId node = kInvalidNodeId;
      if (scene->graph().getLightNode(lightId, node) && isValid(node)) {
        ++currentLightCount;
      }
    });

    const uint32_t currentRenderableCount =
        static_cast<uint32_t>(scene->renderables().size());
    if (scene->topologyVersion() != cachedHierarchyTopologyVersion) {
      hierarchyTopologyCacheValid = false;
      hierarchyStatsCacheValid = false;
      hierarchyTopologyBuildQueue.clear();
      hierarchyTopologyBuildQueueCursor = 0u;
      hierarchyTopologyBuildNode = kInvalidNodeId;
      hierarchyTopologyBuildNextChild = kInvalidNodeId;
      hierarchyTopologyBuildNodeStarted = false;
      hierarchyStatsBuilding = false;
    }
    if (!hierarchyTopologyCacheValid) {
      NURI_PROFILER_ZONE("ImGuiEditor::HierarchyWindow::BuildTopologyCache",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (hierarchyTopologyBuildQueue.empty()) {
        hierarchyNodeTopology.clear();
        hierarchyTopologyBuildQueue.push_back(scene->graph().rootNode());
        hierarchyTopologyBuildQueueCursor = 0u;
      }
      constexpr uint32_t kTopologyOperationsPerFrame = 512u;
      uint32_t operations = 0u;
      while (operations < kTopologyOperationsPerFrame) {
        if (!hierarchyTopologyBuildNodeStarted) {
          if (hierarchyTopologyBuildQueueCursor >=
              hierarchyTopologyBuildQueue.size()) {
            hierarchyTopologyCacheValid = true;
            cachedHierarchyTopologyVersion = scene->topologyVersion();
            hierarchyTopologyBuildQueue.clear();
            hierarchyTopologyBuildQueueCursor = 0u;
            if (hierarchyNodeOpenFlags.size() < hierarchyNodeTopology.size()) {
              hierarchyNodeOpenFlags.resize(hierarchyNodeTopology.size(), 0u);
            }
            break;
          }
          hierarchyTopologyBuildNode =
              hierarchyTopologyBuildQueue[hierarchyTopologyBuildQueueCursor++];
          hierarchyTopologyBuildNodeStarted = true;
          const size_t nodeSlot = hierarchyNodeSlot(hierarchyTopologyBuildNode);
          if (nodeSlot >= hierarchyNodeTopology.size()) {
            hierarchyNodeTopology.resize(nodeSlot + 1u);
          }
          HierarchyNodeTopology &entry = hierarchyNodeTopology[nodeSlot];
          entry.labelName =
              nodeDisplayName(scene->graph(), hierarchyTopologyBuildNode);
          entry.children.clear();
          if (!scene->graph().getNodeFirstChild(
                  hierarchyTopologyBuildNode,
                  hierarchyTopologyBuildNextChild)) {
            hierarchyTopologyBuildNextChild = kInvalidNodeId;
          }
          ++operations;
          continue;
        }
        if (!isValid(hierarchyTopologyBuildNextChild)) {
          hierarchyTopologyBuildNodeStarted = false;
          hierarchyTopologyBuildNode = kInvalidNodeId;
          continue;
        }
        const NodeId child = hierarchyTopologyBuildNextChild;
        hierarchyNodeTopology[hierarchyNodeSlot(hierarchyTopologyBuildNode)]
            .children.push_back(child);
        hierarchyTopologyBuildQueue.push_back(child);
        if (!scene->graph().getNodeNextSibling(
                child, hierarchyTopologyBuildNextChild)) {
          hierarchyTopologyBuildNextChild = kInvalidNodeId;
        }
        ++operations;
      }
      NURI_PROFILER_ZONE_END();
      if (!hierarchyTopologyCacheValid) {
        return;
      }
    }
    if (!hierarchyStatsCacheValid ||
        currentRenderableCount != cachedRenderableCount ||
        currentLightCount != cachedLightCount) {
      bool statsIncomplete = false;
      NURI_PROFILER_ZONE("ImGuiEditor::HierarchyWindow::BuildNodeStats",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (!hierarchyStatsBuilding) {
        hierarchyNodeStats.assign(hierarchyNodeTopology.size(),
                                  HierarchyNodeStats{});
        hierarchyStatsBuildCursor = 0u;
        hierarchyStatsBuilding = true;
      }
      constexpr uint32_t kRenderableStatsPerFrame = 2048u;
      uint32_t processed = 0u;
      const std::span<const Renderable> renderables = scene->renderables();
      while (hierarchyStatsBuildCursor < renderables.size() &&
             processed < kRenderableStatsPerFrame) {
        const Renderable &renderable = renderables[hierarchyStatsBuildCursor++];
        ++processed;
        if (!isValid(renderable.node)) {
          continue;
        }
        const size_t nodeSlot = hierarchyNodeSlot(renderable.node);
        if (nodeSlot >= hierarchyNodeStats.size()) {
          hierarchyNodeStats.resize(nodeSlot + 1u);
        }
        ++hierarchyNodeStats[nodeSlot].renderableCount;
      }
      statsIncomplete = hierarchyStatsBuildCursor < renderables.size();
      if (!statsIncomplete) {
        scene->graph().forEachLightId([&](LightId lightId) {
          NodeId node = kInvalidNodeId;
          if (scene->graph().getLightNode(lightId, node) && isValid(node)) {
            const size_t nodeSlot = hierarchyNodeSlot(node);
            if (nodeSlot >= hierarchyNodeStats.size()) {
              hierarchyNodeStats.resize(nodeSlot + 1u);
            }
            ++hierarchyNodeStats[nodeSlot].lightCount;
          }
        });

        cachedRenderableCount = currentRenderableCount;
        cachedLightCount = currentLightCount;
        hierarchyStatsCacheValid = true;
        hierarchyStatsBuilding = false;
      }
      NURI_PROFILER_ZONE_END();
      if (statsIncomplete) {
        return;
      }
    }

    const NodeId selectedLeaf =
        selectionState != nullptr ? selectionState->node : kInvalidNodeId;
    if (selectedLeaf != cachedSelectedPathLeaf) {
      NURI_PROFILER_ZONE("ImGuiEditor::HierarchyWindow::BuildSelectedPath",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      selectedPathNodeFlags.assign(hierarchyNodeStats.size(), 0u);
      selectedPathChildIndices.assign(hierarchyNodeStats.size(), -1);
      if (isValid(selectedLeaf)) {
        NodeId current = selectedLeaf;
        while (isValid(current)) {
          const size_t currentSlot = hierarchyNodeSlot(current);
          if (currentSlot >= selectedPathNodeFlags.size()) {
            selectedPathNodeFlags.resize(currentSlot + 1u, 0u);
            selectedPathChildIndices.resize(currentSlot + 1u, -1);
          }
          selectedPathNodeFlags[currentSlot] = 1u;
          NodeId parent = kInvalidNodeId;
          if (!scene->graph().getNodeParent(current, parent)) {
            break;
          }
          if (isValid(parent)) {
            const size_t parentSlot = hierarchyNodeSlot(parent);
            if (parentSlot >= selectedPathChildIndices.size()) {
              selectedPathChildIndices.resize(parentSlot + 1u, -1);
            }
            const std::vector<NodeId> &siblings =
                hierarchyTopology(parent).children;
            const auto it =
                std::find(siblings.begin(), siblings.end(), current);
            if (it != siblings.end()) {
              selectedPathChildIndices[parentSlot] =
                  static_cast<int32_t>(std::distance(siblings.begin(), it));
            }
          }
          current = parent;
        }
      }
      cachedSelectedPathLeaf = selectedLeaf;
      NURI_PROFILER_ZONE_END();
    }
  }

  [[nodiscard]] const HierarchyNodeStats &hierarchyStats(NodeId node) const {
    static const HierarchyNodeStats kEmptyStats{};
    const size_t nodeSlot = hierarchyNodeSlot(node);
    return isValid(node) && nodeSlot < hierarchyNodeStats.size()
               ? hierarchyNodeStats[nodeSlot]
               : kEmptyStats;
  }

  [[nodiscard]] const HierarchyNodeTopology &
  hierarchyTopology(NodeId node) const {
    static const HierarchyNodeTopology kEmptyTopology{};
    const size_t nodeSlot = hierarchyNodeSlot(node);
    return isValid(node) && nodeSlot < hierarchyNodeTopology.size()
               ? hierarchyNodeTopology[nodeSlot]
               : kEmptyTopology;
  }

  [[nodiscard]] int32_t selectedChildIndexForParent(NodeId node) const {
    const size_t nodeSlot = hierarchyNodeSlot(node);
    return isValid(node) && nodeSlot < selectedPathChildIndices.size()
               ? selectedPathChildIndices[nodeSlot]
               : -1;
  }

  [[nodiscard]] bool nodeIsOnSelectedPath(NodeId node) const {
    const size_t nodeSlot = hierarchyNodeSlot(node);
    return isValid(node) && nodeSlot < selectedPathNodeFlags.size() &&
           selectedPathNodeFlags[nodeSlot] != 0u;
  }

  [[nodiscard]] bool childRangeContainsSelectedChild(NodeId parentNode,
                                                     size_t beginIndex,
                                                     size_t endIndex) const {
    const int32_t selectedChildIndex = selectedChildIndexForParent(parentNode);
    return selectedChildIndex >= 0 &&
           beginIndex <= static_cast<size_t>(selectedChildIndex) &&
           static_cast<size_t>(selectedChildIndex) < endIndex;
  }

  [[nodiscard]] static size_t hierarchyBatchSpan(size_t count) {
    size_t span = kHierarchyBatchSize;
    while (((count + span - 1u) / span) > kHierarchyBatchSize) {
      span *= kHierarchyBatchSize;
    }
    return span;
  }

  [[nodiscard]] static uint64_t
  hierarchyBatchKey(NodeId parentNode, size_t beginIndex, size_t endIndex) {
    uint64_t key = static_cast<uint64_t>(parentNode.value);
    key ^= 0x9e3779b97f4a7c15ull + (key << 6u) + (key >> 2u) +
           static_cast<uint64_t>(beginIndex);
    key ^= 0x9e3779b97f4a7c15ull + (key << 6u) + (key >> 2u) +
           static_cast<uint64_t>(endIndex);
    return key;
  }

  void drawNodeTransformEditor(NodeId node) {
    if (scene == nullptr || !isValid(node)) {
      ImGui::TextUnformatted("No node selected.");
      return;
    }

    SceneGraph &graph = scene->graph();
    (void)graph.syncWorldTransforms();
    glm::mat4 localMatrix(1.0f);
    glm::mat4 worldMatrix(1.0f);
    if (!graph.getNodeLocalTransform(node, localMatrix)) {
      ImGui::TextUnformatted("Selected node is no longer valid.");
      return;
    }
    (void)graph.getCachedNodeWorldTransform(node, worldMatrix);

    float translation[3]{};
    float rotation[3]{};
    float scale[3]{};
    ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(localMatrix),
                                          translation, rotation, scale);
    bool changed = false;
    changed |= ImGui::InputFloat3("Translation", translation, "%.3f");
    changed |= ImGui::InputFloat3("Rotation", rotation, "%.3f");
    changed |= ImGui::InputFloat3("Scale", scale, "%.3f");
    if (changed) {
      glm::mat4 recomposed(1.0f);
      ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale,
                                              glm::value_ptr(recomposed));
      (void)graph.setNodeLocalTransform(node, recomposed);
      localMatrix = recomposed;
      (void)graph.syncWorldTransforms();
      (void)graph.getCachedNodeWorldTransform(node, worldMatrix);
    }

    const glm::vec3 worldPosition = glm::vec3(worldMatrix[3]);
    ImGui::SeparatorText("World");
    ImGui::Text("Position: %.3f %.3f %.3f", worldPosition.x, worldPosition.y,
                worldPosition.z);
    ImGui::Text("Node Handle: %u", indexOf(node));
  }

  bool applyMaterialOverride(RenderableInspectorState &state,
                             const MaterialRecord &sourceRecord,
                             const MaterialDesc &desc,
                             const MaterialRequest::TextureRefs &textureRefs) {
    if (resources == nullptr || scene == nullptr) {
      return false;
    }

    auto acquireResult = resources->acquireMaterial(MaterialRequest{
        .desc = desc,
        .textureRefs = textureRefs,
        .debugName = !sourceRecord.debugName.empty()
                         ? std::string(sourceRecord.debugName) + " (Override)"
                         : std::string("Editor Override"),
        .sourceIdentity = "editor_override/renderable_" +
                          std::to_string(state.renderableId.value),
    });
    if (acquireResult.hasError()) {
      NURI_LOG_WARNING("ImGuiEditor: failed to acquire override material: %s",
                       acquireResult.error().c_str());
      return false;
    }

    const MaterialRef newRef = acquireResult.value();
    if (!scene->graph().setRenderableMaterialOverride(state.renderableId,
                                                      newRef)) {
      resources->release(newRef);
      NURI_LOG_WARNING("ImGuiEditor: failed to attach override material to "
                       "renderable %u",
                       state.renderableId.value);
      return false;
    }

    releaseOwnedOverride(state);
    state.ownedOverride = newRef;
    return true;
  }

  void drawMaterialViewer(const MaterialRecord &record,
                          RenderableInspectorState &state, bool editable) {
    MaterialDesc editedDesc = record.desc;
    const MaterialRequest::TextureRefs textureRefs = record.textureRefs;

    if (!editable) {
      ImGui::BeginDisabled();
    }

    bool changed = false;
    changed |= ImGui::ColorEdit4("Base Color",
                                 glm::value_ptr(editedDesc.baseColorFactor));
    changed |= ImGui::ColorEdit3("Emissive",
                                 glm::value_ptr(editedDesc.emissiveFactor));
    changed |= ImGui::SliderFloat(
        "Emissive Strength", &editedDesc.emissiveStrength, 0.0f, 32.0f, "%.3f");
    changed |= ImGui::SliderFloat("Metallic", &editedDesc.metallicFactor, 0.0f,
                                  1.0f, "%.3f");
    changed |= ImGui::SliderFloat("Roughness", &editedDesc.roughnessFactor,
                                  0.0f, 1.0f, "%.3f");
    constexpr const char *kAlphaModes[] = {"Opaque", "Mask", "Blend"};
    int alphaMode = static_cast<int>(editedDesc.alphaMode);
    if (ImGui::Combo("Alpha Mode", &alphaMode, kAlphaModes,
                     IM_ARRAYSIZE(kAlphaModes))) {
      editedDesc.alphaMode = static_cast<MaterialAlphaMode>(std::clamp(
          alphaMode, 0, static_cast<int>(IM_ARRAYSIZE(kAlphaModes)) - 1));
      changed = true;
    }
    changed |= ImGui::SliderFloat("Alpha Cutoff", &editedDesc.alphaCutoff, 0.0f,
                                  1.0f, "%.3f");
    changed |= ImGui::Checkbox("Double Sided", &editedDesc.doubleSided);

    if (!editable) {
      ImGui::EndDisabled();
    }

    if (editable && changed) {
      applyMaterialOverride(state, record, editedDesc, textureRefs);
    }

    ImGui::SeparatorText("Textures");
    const std::vector<MaterialTextureEntry> textures =
        buildMaterialTextureEntries(record);
    if (textures.empty()) {
      ImGui::TextUnformatted("No previewable textures.");
      return;
    }
    state.selectedTextureIndex = std::clamp(
        state.selectedTextureIndex, 0, static_cast<int>(textures.size()) - 1);
    if (ImGui::BeginListBox("Slots")) {
      for (int index = 0; index < static_cast<int>(textures.size()); ++index) {
        const bool selected = state.selectedTextureIndex == index;
        if (ImGui::Selectable(textures[index].label, selected)) {
          state.selectedTextureIndex = index;
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndListBox();
    }

    const MaterialTextureEntry &textureEntry =
        textures[static_cast<size_t>(state.selectedTextureIndex)];
    const TextureRecord *textureRecord =
        resources != nullptr ? resources->tryGet(textureEntry.ref) : nullptr;
    if (textureRecord == nullptr) {
      ImGui::TextUnformatted("Selected texture is unavailable.");
      return;
    }
    ImGui::Text("Preview: %s", textureEntry.label);
    if (textureRecord->bindlessIndex != kInvalidTextureBindlessIndex) {
      ImGui::Image(toImTextureId(textureRecord->bindlessIndex),
                   ImVec2(192.0f, 192.0f));
    }
    if (!textureRecord->debugName.empty()) {
      ImGui::Text("Debug: %s", textureRecord->debugName.c_str());
    }
    if (!textureRecord->canonicalPath.empty()) {
      ImGui::TextWrapped("Path: %s", textureRecord->canonicalPath.c_str());
    }
    ImGui::Text("Size: %ux%u", textureRecord->dimensions.width,
                textureRecord->dimensions.height);
    ImGui::Text("Format: %s", formatDisplayName(textureRecord->format));
    ImGui::Text("Mip Levels: %u", textureRecord->numMipLevels);
  }

  void drawRenderableInspector(const Renderable &renderable) {
    if (resources == nullptr) {
      ImGui::TextUnformatted("Resource manager unavailable.");
      return;
    }
    RenderableInspectorState &state =
        ensureRenderableInspectorState(renderable.id);
    std::vector<MaterialSourceEntry> sourceEntries =
        buildMaterialSourceEntries(renderable, resources);
    state.baselineSlotIndex = std::clamp(
        state.baselineSlotIndex, 0, static_cast<int>(sourceEntries.size()) - 1);

    const bool hasOverride = isValid(renderable.materialOverride);
    const MaterialRef displayedMaterial =
        hasOverride
            ? renderable.materialOverride
            : sourceEntries[static_cast<size_t>(state.baselineSlotIndex)].ref;
    const MaterialRecord *displayedRecord =
        resources->tryGet(displayedMaterial);

    ImGui::SeparatorText("Renderable");
    ImGui::Text("Renderable Index: %u", selectionState->renderableIndex);
    ImGui::Text("Renderable Handle: %u", renderable.id.value);

    if (const ModelRecord *modelRecord = resources->tryGet(renderable.model);
        modelRecord != nullptr) {
      if (!modelRecord->canonicalPath.empty()) {
        ImGui::TextWrapped("Model: %s", modelRecord->canonicalPath.c_str());
      }
    }

    ImGui::SeparatorText("Material");
    if (!hasOverride && sourceEntries.size() > 1u) {
      std::vector<const char *> labels;
      labels.reserve(sourceEntries.size());
      for (const MaterialSourceEntry &entry : sourceEntries) {
        labels.push_back(entry.label.c_str());
      }
      ImGui::Combo("Baseline Slot", &state.baselineSlotIndex, labels.data(),
                   static_cast<int>(labels.size()));
      for (const MaterialSourceEntry &entry : sourceEntries) {
        ImGui::BulletText("%s", entry.label.c_str());
      }
      ImGui::TextUnformatted(
          "Uniform override will replace all source-material slots.");
    }

    bool overrideEnabled = hasOverride;
    if (ImGui::Checkbox("Uniform Material Override", &overrideEnabled)) {
      if (overrideEnabled) {
        const MaterialRecord *sourceRecord = displayedRecord;
        if (sourceRecord != nullptr) {
          (void)applyMaterialOverride(state, *sourceRecord, sourceRecord->desc,
                                      sourceRecord->textureRefs);
        }
      } else {
        clearRenderableOverride(state);
      }
    }

    if (displayedRecord == nullptr) {
      ImGui::TextUnformatted("Material record unavailable.");
      return;
    }
    if (!displayedRecord->debugName.empty()) {
      ImGui::Text("Debug Name: %s", displayedRecord->debugName.c_str());
    }
    drawMaterialViewer(*displayedRecord, state, overrideEnabled);
  }

  void drawLightInspector(LightId lightId) {
    if (scene == nullptr) {
      return;
    }
    LightDesc light{};
    if (!scene->graph().getLightDesc(lightId, light)) {
      ImGui::TextUnformatted("Selected light is no longer valid.");
      return;
    }
    ImGui::SeparatorText("Light");
    ImGui::Text("Type: %s", lightTypeName(light.type));
    ImGui::Text("Light Slot: %u", indexOf(lightId));
    drawLightEditor(scene->graph(), lightId, light, lightEditorDraft);
  }

  void drawInspectorWindow() {
    if (!ImGui::Begin(kInspectorWindowName, &showInspectorWindow)) {
      inspectorWindowVisible = true;
      inspectorWindowMinX = ImGui::GetWindowPos().x;
      ImGui::End();
      return;
    }
    inspectorWindowVisible = true;
    inspectorWindowMinX = ImGui::GetWindowPos().x;
    if (scene == nullptr || selectionState == nullptr ||
        !isValid(selectionState->node)) {
      ImGui::TextUnformatted("No scene selection.");
      ImGui::End();
      return;
    }

    ImGui::TextUnformatted(
        nodeDisplayName(scene->graph(), selectionState->node).c_str());
    ImGui::Separator();
    drawNodeTransformEditor(selectionState->node);

    if (selectionState->kind == SceneSelectionKind::NodeRenderable) {
      if (const Renderable *renderable =
              scene->renderable(selectionState->renderableIndex);
          renderable != nullptr &&
          renderable->id == selectionState->renderableId) {
        drawRenderableInspector(*renderable);
      }
    } else if (selectionState->kind == SceneSelectionKind::Light) {
      drawLightInspector(selectionState->lightId);
    } else {
      uint32_t renderableCount = 0u;
      scene->graph().forEachRenderableOnNode(
          selectionState->node, [&](RenderableId) { ++renderableCount; });
      uint32_t lightCount = 0u;
      scene->graph().forEachLightOnNode(selectionState->node,
                                        [&](LightId) { ++lightCount; });
      ImGui::SeparatorText("Components");
      ImGui::Text("Renderables: %u", renderableCount);
      ImGui::Text("Lights: %u", lightCount);
    }

    ImGui::End();
  }

  void drawScenesSection() {
    ImGui::SeparatorText("Scenes");
    if (sceneSelectionState.nameViews.empty()) {
      ImGui::TextUnformatted("No scenes available.");
      return;
    }

    int selectedIndex = sceneSelectionState.selectedIndex;
    if (ImGui::Combo("Scene", &selectedIndex,
                     sceneSelectionState.nameViews.data(),
                     static_cast<int>(sceneSelectionState.nameViews.size())) &&
        selectedIndex != sceneSelectionState.selectedIndex) {
      sceneSelectionState.selectedIndex = selectedIndex;
      sceneSelectionState.pendingSelectionRequest =
          sceneSelectionState.ids[static_cast<size_t>(selectedIndex)];
    }
    if (!sceneSelectionState.hotkeyHint.empty()) {
      ImGui::TextUnformatted(sceneSelectionState.hotkeyHint.c_str());
    }
    if (!sceneSelectionState.pendingSceneId.empty()) {
      const auto pendingIt = std::find(sceneSelectionState.ids.begin(),
                                       sceneSelectionState.ids.end(),
                                       sceneSelectionState.pendingSceneId);
      const std::string_view pendingLabel =
          pendingIt != sceneSelectionState.ids.end()
              ? std::string_view(sceneSelectionState.names[static_cast<size_t>(
                    std::distance(sceneSelectionState.ids.begin(), pendingIt))])
              : std::string_view(sceneSelectionState.pendingSceneId);
      if (sceneSelectionState.loadFailed) {
        ImGui::Text("Failed to load %.*s",
                    static_cast<int>(pendingLabel.size()), pendingLabel.data());
        if (!sceneSelectionState.loadError.empty()) {
          ImGui::TextWrapped("%s", sceneSelectionState.loadError.c_str());
        }
        if (ImGui::Button("Retry scene load")) {
          sceneSelectionState.pendingSelectionRequest =
              sceneSelectionState.pendingSceneId;
        }
        ImGui::SameLine();
        if (ImGui::Button("Dismiss")) {
          sceneSelectionState.pendingCancelRequest = true;
        }
      } else {
        ImGui::Text("%s %.*s", sceneSelectionState.loadPhase.c_str(),
                    static_cast<int>(pendingLabel.size()), pendingLabel.data());
        ImGui::ProgressBar(sceneSelectionState.loadProgress,
                           ImVec2(-FLT_MIN, 0.0f));
        if (sceneSelectionState.loadCancellable &&
            ImGui::Button("Cancel scene load")) {
          sceneSelectionState.pendingCancelRequest = true;
        }
      }
    }
  }

  [[nodiscard]] bool isHierarchyNodeOpen(NodeId node) const {
    const size_t nodeSlot = hierarchyNodeSlot(node);
    return isValid(node) && nodeSlot < hierarchyNodeOpenFlags.size() &&
           hierarchyNodeOpenFlags[nodeSlot] != 0u;
  }

  void setHierarchyNodeOpen(NodeId node, bool open) {
    if (!isValid(node)) {
      return;
    }
    const size_t nodeSlot = hierarchyNodeSlot(node);
    if (nodeSlot >= hierarchyNodeOpenFlags.size()) {
      hierarchyNodeOpenFlags.resize(nodeSlot + 1u, 0u);
    }
    hierarchyNodeOpenFlags[nodeSlot] = open ? 1u : 0u;
  }

  [[nodiscard]] bool isHierarchyBatchOpen(NodeId parentNode, size_t beginIndex,
                                          size_t endIndex) const {
    return hierarchyOpenBatchKeys.contains(
        hierarchyBatchKey(parentNode, beginIndex, endIndex));
  }

  void setHierarchyBatchOpen(NodeId parentNode, size_t beginIndex,
                             size_t endIndex, bool open) {
    const uint64_t key = hierarchyBatchKey(parentNode, beginIndex, endIndex);
    if (open) {
      hierarchyOpenBatchKeys.insert(key);
    } else {
      hierarchyOpenBatchKeys.erase(key);
    }
  }

  void revealHierarchyBatchPath(NodeId parentNode, size_t childIndex) {
    const std::vector<NodeId> &children =
        hierarchyTopology(parentNode).children;
    if (childIndex >= children.size()) {
      return;
    }

    size_t beginIndex = 0u;
    size_t endIndex = children.size();
    while (endIndex - beginIndex > kHierarchyBatchSize) {
      const size_t childSpan = hierarchyBatchSpan(endIndex - beginIndex);
      const size_t relativeIndex = childIndex - beginIndex;
      const size_t subBegin =
          beginIndex + (relativeIndex / childSpan) * childSpan;
      const size_t subEnd = std::min(subBegin + childSpan, endIndex);
      setHierarchyBatchOpen(parentNode, subBegin, subEnd, true);
      beginIndex = subBegin;
      endIndex = subEnd;
    }
  }

  void applyPendingHierarchyReveal() {
    if (!pendingRevealSelection || selectionState == nullptr ||
        !isValid(selectionState->node)) {
      return;
    }

    hierarchySceneRootOpen = true;
    NodeId current = selectionState->node;
    while (isValid(current)) {
      if (!hierarchyTopology(current).children.empty()) {
        setHierarchyNodeOpen(current, true);
      }
      NodeId parent = kInvalidNodeId;
      if (!scene->graph().getNodeParent(current, parent)) {
        break;
      }
      if (isValid(parent)) {
        const int32_t childIndex = selectedChildIndexForParent(parent);
        if (childIndex >= 0) {
          revealHierarchyBatchPath(parent, static_cast<size_t>(childIndex));
          setHierarchyNodeOpen(parent, true);
        }
      }
      current = parent;
    }
  }

  void appendHierarchyVisibleRowsForChildren(
      NodeId parentNode, const std::vector<NodeId> &children, int depth,
      size_t beginIndex, size_t endIndex) {
    const size_t clampedEnd = std::min(endIndex, children.size());
    if (beginIndex >= clampedEnd) {
      return;
    }

    const size_t rangeCount = clampedEnd - beginIndex;
    if (rangeCount <= kHierarchyBatchSize) {
      for (size_t index = beginIndex; index < clampedEnd; ++index) {
        const NodeId child = children[index];
        hierarchyVisibleRows.push_back(HierarchyVisibleRow{
            .kind = HierarchyRowKind::Node,
            .depth = depth,
            .node = child,
        });
        if (selectionState != nullptr && selectionState->node == child) {
          hierarchySelectedRowIndex =
              static_cast<int>(hierarchyVisibleRows.size()) - 1;
        }
        if (isHierarchyNodeOpen(child)) {
          const HierarchyNodeTopology &childTopology = hierarchyTopology(child);
          appendHierarchyVisibleRowsForChildren(child, childTopology.children,
                                                depth + 1u, 0u,
                                                childTopology.children.size());
        }
      }
      return;
    }

    const size_t childSpan = hierarchyBatchSpan(rangeCount);
    for (size_t index = beginIndex; index < clampedEnd; index += childSpan) {
      const size_t subEnd = std::min(index + childSpan, clampedEnd);
      hierarchyVisibleRows.push_back(HierarchyVisibleRow{
          .kind = HierarchyRowKind::Batch,
          .depth = depth,
          .node = parentNode,
          .beginIndex = index,
          .endIndex = subEnd,
      });
      if (isHierarchyBatchOpen(parentNode, index, subEnd)) {
        appendHierarchyVisibleRowsForChildren(parentNode, children, depth + 1u,
                                              index, subEnd);
      }
    }
  }

  void rebuildHierarchyVisibleRows() {
    hierarchyVisibleRows.clear();
    hierarchySelectedRowIndex = -1;
    if (scene == nullptr) {
      return;
    }

    hierarchyVisibleRows.push_back(
        HierarchyVisibleRow{.kind = HierarchyRowKind::SceneRoot, .depth = 0});
    if (!hierarchySceneRootOpen) {
      return;
    }

    const std::vector<NodeId> &children =
        hierarchyTopology(scene->graph().rootNode()).children;
    appendHierarchyVisibleRowsForChildren(scene->graph().rootNode(), children,
                                          1, 0u, children.size());
  }

  void drawHierarchyRow(const HierarchyVisibleRow &row,
                        std::string_view sceneLabel) {
    if (scene == nullptr || selectionState == nullptr) {
      return;
    }

    const float indentWidth =
        static_cast<float>(row.depth) * ImGui::GetTreeNodeToLabelSpacing();
    if (indentWidth > 0.0f) {
      ImGui::Indent(indentWidth);
    }

    if (row.kind == HierarchyRowKind::SceneRoot) {
      ImGui::SetNextItemOpen(hierarchySceneRootOpen, ImGuiCond_Always);
      const bool open =
          ImGui::TreeNodeEx("active_scene_root",
                            ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                ImGuiTreeNodeFlags_SpanAvailWidth |
                                ImGuiTreeNodeFlags_OpenOnArrow |
                                ImGuiTreeNodeFlags_OpenOnDoubleClick,
                            "%s", std::string(sceneLabel).c_str());
      hierarchySceneRootOpen = open;
    } else if (row.kind == HierarchyRowKind::Batch) {
      const bool isOpen =
          isHierarchyBatchOpen(row.node, row.beginIndex, row.endIndex);
      const bool containsSelected = childRangeContainsSelectedChild(
          row.node, row.beginIndex, row.endIndex);
      ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                 ImGuiTreeNodeFlags_SpanAvailWidth |
                                 ImGuiTreeNodeFlags_OpenOnArrow |
                                 ImGuiTreeNodeFlags_OpenOnDoubleClick;
      if (containsSelected) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
      }
      const std::string label = "More " + std::to_string(row.beginIndex + 1u) +
                                "-" + std::to_string(row.endIndex) + " (" +
                                std::to_string(row.endIndex - row.beginIndex) +
                                ")";
      ImGui::PushID(static_cast<int>(row.node.value));
      ImGui::PushID(static_cast<int>(row.beginIndex));
      ImGui::PushID(static_cast<int>(row.endIndex));
      ImGui::SetNextItemOpen(isOpen, ImGuiCond_Always);
      const bool open = ImGui::TreeNodeEx("batch", flags, "%s", label.c_str());
      if (open != isOpen) {
        setHierarchyBatchOpen(row.node, row.beginIndex, row.endIndex, open);
      }
      ImGui::PopID();
      ImGui::PopID();
      ImGui::PopID();
    } else {
      const HierarchyNodeTopology &topology = hierarchyTopology(row.node);
      const HierarchyNodeStats &stats = hierarchyStats(row.node);
      const bool hasChildren = !topology.children.empty();
      const bool isOpen = hasChildren && isHierarchyNodeOpen(row.node);
      ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                 ImGuiTreeNodeFlags_SpanAvailWidth |
                                 ImGuiTreeNodeFlags_OpenOnArrow |
                                 ImGuiTreeNodeFlags_OpenOnDoubleClick;
      if (!hasChildren) {
        flags |= ImGuiTreeNodeFlags_Leaf;
      }
      if (selectionState->node == row.node) {
        flags |= ImGuiTreeNodeFlags_Selected;
      }
      std::string label = topology.labelName +
                          "  R:" + std::to_string(stats.renderableCount) +
                          "  L:" + std::to_string(stats.lightCount);
      ImGui::SetNextItemOpen(isOpen, ImGuiCond_Always);
      ImGui::PushID(static_cast<int>(row.node.value));
      const bool open = ImGui::TreeNodeEx("node", flags, "%s", label.c_str());
      if (hasChildren && open != isOpen) {
        setHierarchyNodeOpen(row.node, open);
      }
      if (ImGui::IsItemClicked()) {
        suppressRevealForNextSelectionChange = true;
        applyNodeSelection(*scene, row.node, *selectionState);
      }
      ImGui::PopID();
    }

    if (indentWidth > 0.0f) {
      ImGui::Unindent(indentWidth);
    }
  }

  void drawHierarchyWindow() {
    if (!ImGui::Begin(kHierarchyWindowName, &showHierarchyWindow)) {
      ImGui::End();
      return;
    }
    if (scene == nullptr) {
      ImGui::TextUnformatted("No scene available.");
      ImGui::End();
      return;
    }

    NURI_PROFILER_ZONE("ImGuiEditor::HierarchyWindow::ScenesSection",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    drawScenesSection();
    ImGui::SeparatorText("Hierarchy");
    NURI_PROFILER_ZONE_END();

    rebuildHierarchyFrameCache();
    applyPendingHierarchyReveal();
    rebuildHierarchyVisibleRows();

    NURI_PROFILER_ZONE("ImGuiEditor::HierarchyWindow::DrawTree",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    if (hierarchyVisibleRows.size() <= 1u) {
      ImGui::TextUnformatted("Scene graph is empty.");
    } else {
      const bool hasSceneName =
          !sceneSelectionState.names.empty() &&
          sceneSelectionState.selectedIndex >= 0 &&
          sceneSelectionState.selectedIndex <
              static_cast<int>(sceneSelectionState.names.size());
      const std::string sceneLabel =
          hasSceneName
              ? sceneSelectionState.names[sceneSelectionState.selectedIndex]
              : std::string("Active Scene");
      if (pendingRevealSelection && hierarchySelectedRowIndex >= 0) {
        const float lineHeight = ImGui::GetTextLineHeightWithSpacing();
        const float visibleHeight =
            std::max(ImGui::GetContentRegionAvail().y, lineHeight * 3.0f);
        const float targetY = std::max(
            0.0f, static_cast<float>(hierarchySelectedRowIndex) * lineHeight -
                      visibleHeight * 0.35f);
        ImGui::SetScrollY(targetY);
        pendingRevealSelection = false;
      }
      ImGuiListClipper clipper;
      clipper.Begin(static_cast<int>(hierarchyVisibleRows.size()),
                    ImGui::GetTextLineHeightWithSpacing());
      while (clipper.Step()) {
        for (int rowIndex = clipper.DisplayStart; rowIndex < clipper.DisplayEnd;
             ++rowIndex) {
          drawHierarchyRow(hierarchyVisibleRows[static_cast<size_t>(rowIndex)],
                           sceneLabel);
        }
      }
    }
    NURI_PROFILER_ZONE_END();
    ImGui::End();
  }

  void drawAnimationPlayerWindow() {
    if (animationPlayer == nullptr) {
      showAnimationPlayerWindow = false;
      return;
    }
    if (focusAnimationPlayerWindow) {
      ImGui::SetNextWindowFocus();
      focusAnimationPlayerWindow = false;
    }
    if (!ImGui::Begin(kAnimationPlayerWindowName, &showAnimationPlayerWindow)) {
      ImGui::End();
      return;
    }

    const EditorAnimationPlayerView view = animationPlayer->selectedView();
    switch (view.availability) {
    case EditorAnimationPlayerAvailability::NoSelection:
      ImGui::TextUnformatted("No scene selection.");
      ImGui::End();
      return;
    case EditorAnimationPlayerAvailability::NotAnimated:
      if (!view.selectionLabel.empty()) {
        ImGui::Text("Selected: %s", view.selectionLabel.c_str());
        ImGui::Separator();
      }
      ImGui::TextUnformatted("Selected object has no imported animations.");
      ImGui::End();
      return;
    case EditorAnimationPlayerAvailability::Animated:
      break;
    }

    ImGui::Text("Object: %s", view.instanceLabel.c_str());
    if (!view.selectionLabel.empty()) {
      ImGui::Text("Selected: %s", view.selectionLabel.c_str());
    }
    ImGui::Separator();

    if (ImGui::Button("Restart")) {
      deferredAnimationActions.push_back(DeferredAnimationAction{
          .type = DeferredAnimationActionType::Restart});
    }
    ImGui::SameLine();
    if (ImGui::Button("Start")) {
      deferredAnimationActions.push_back(
          DeferredAnimationAction{.type = DeferredAnimationActionType::Start});
    }
    ImGui::SameLine();
    if (ImGui::Button("Pause")) {
      deferredAnimationActions.push_back(
          DeferredAnimationAction{.type = DeferredAnimationActionType::Pause});
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(view.running ? "Running"
                                        : (view.paused ? "Paused" : "Stopped"));

    float blendWeight = view.blendWeight;
    if (!view.hasSecondaryClipSelection()) {
      ImGui::BeginDisabled();
    }
    if (ImGui::SliderFloat("Blend", &blendWeight, 0.0f, 1.0f, "%.2f",
                           ImGuiSliderFlags_AlwaysClamp)) {
      deferredAnimationActions.push_back(DeferredAnimationAction{
          .type = DeferredAnimationActionType::SetBlendWeight,
          .floatValue = blendWeight,
      });
    }
    if (!view.hasSecondaryClipSelection()) {
      ImGui::EndDisabled();
    }

    const float duration = std::max(view.timelineDurationSeconds, 0.0f);
    float timeline = glm::clamp(view.timelineTimeSeconds, 0.0f, duration);
    if (duration <= 0.0f) {
      ImGui::BeginDisabled();
    }
    if (ImGui::SliderFloat("Timeline", &timeline, 0.0f, duration, "%.3fs",
                           ImGuiSliderFlags_AlwaysClamp)) {
      deferredAnimationActions.push_back(DeferredAnimationAction{
          .type = DeferredAnimationActionType::Seek,
          .floatValue = timeline,
      });
    }
    if (duration <= 0.0f) {
      ImGui::EndDisabled();
    }
    ImGui::Text("%.3fs / %.3fs", timeline, duration);

    if (ImGui::BeginTable("AnimationPlayerClips", 4,
                          ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_BordersInnerH |
                              ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableSetupColumn("Primary", ImGuiTableColumnFlags_WidthFixed,
                              64.0f);
      ImGui::TableSetupColumn("Blend", ImGuiTableColumnFlags_WidthFixed, 64.0f);
      ImGui::TableSetupColumn("Clip", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Duration", ImGuiTableColumnFlags_WidthFixed,
                              90.0f);
      ImGui::TableHeadersRow();

      for (const EditorAnimationClipInfo &clip : view.clips) {
        const bool isPrimary = clip.clipIndex == view.primaryClipIndex;
        const bool isSecondary = clip.clipIndex == view.secondaryClipIndex;

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushID(static_cast<int>(clip.clipIndex));
        if (ImGui::RadioButton("##Primary", isPrimary) && !isPrimary) {
          deferredAnimationActions.push_back(DeferredAnimationAction{
              .type = DeferredAnimationActionType::SetPrimaryClip,
              .clipIndex = clip.clipIndex,
          });
        }

        ImGui::TableNextColumn();
        bool enableBlend = isSecondary;
        if (isPrimary) {
          ImGui::BeginDisabled();
        }
        if (ImGui::Checkbox("##Blend", &enableBlend)) {
          deferredAnimationActions.push_back(DeferredAnimationAction{
              .type = DeferredAnimationActionType::SetSecondaryClip,
              .clipIndex = clip.clipIndex,
              .enabled = enableBlend,
          });
        }
        if (isPrimary) {
          ImGui::EndDisabled();
        }

        ImGui::TableNextColumn();
        ImGui::TextUnformatted(clip.name.data(),
                               clip.name.data() + clip.name.size());

        ImGui::TableNextColumn();
        ImGui::Text("%.3fs", clip.durationSeconds);
        ImGui::PopID();
      }
      ImGui::EndTable();
    }

    ImGui::End();
  }

  void drawMainMenuBar() {
    if (!ImGui::BeginMainMenuBar()) {
      return;
    }

    if (ImGui::BeginMenu("Menu")) {
      ImGui::MenuItem("Bakery", nullptr, &showBakeryWindow);
      ImGui::MenuItem("Font Compiler", nullptr, &showFontCompilerWindow);
      ImGui::MenuItem("Hierarchy", nullptr, &showHierarchyWindow);
      ImGui::MenuItem("Inspector", nullptr, &showInspectorWindow);
      if (animationPlayer != nullptr) {
        ImGui::MenuItem("Animation Player", nullptr,
                        &showAnimationPlayerWindow);
      }
      ImGui::MenuItem("Render Passes", nullptr, &showRenderPassesWindow);
      ImGui::MenuItem("Lights", nullptr, &showLightsWindow);
      ImGui::MenuItem("Shadows", nullptr, &showShadowsWindow);
      ImGui::MenuItem("Anti-Aliasing", nullptr, &showAntiAliasingWindow);
      ImGui::MenuItem("Ambient Occlusion", nullptr,
                      &showAmbientOcclusionWindow);
      ImGui::MenuItem("HDR Postprocess", nullptr, &showHDRPostProcessWindow);
      if (ImGui::BeginMenu("Texture Filtering")) {
        auto &settings = renderSettings.textureFiltering;
        sanitizeTextureFilteringSettings(settings);
        const bool bilinear = settings.mode == TextureFilterMode::Bilinear;
        const bool trilinear = settings.mode == TextureFilterMode::Trilinear;
        const bool anisotropic =
            settings.mode == TextureFilterMode::Anisotropic;
        if (ImGui::MenuItem("Bilinear", nullptr, bilinear)) {
          settings.mode = TextureFilterMode::Bilinear;
        }
        if (ImGui::MenuItem("Trilinear", nullptr, trilinear)) {
          settings.mode = TextureFilterMode::Trilinear;
        }
        if (ImGui::MenuItem("Anisotropic", nullptr, anisotropic)) {
          settings.mode = TextureFilterMode::Anisotropic;
        }
        ImGui::Separator();
        ImGui::MenuItem("Settings Window", nullptr,
                        &showTextureFilteringWindow);
        ImGui::EndMenu();
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Debug")) {
      ImGui::MenuItem("Frame Pacing", nullptr, &showFramePacingWindow);
      ImGui::MenuItem("Render Graph Telemetry", nullptr,
                      &showRenderGraphTelemetryWindow);
      ImGui::MenuItem("Passes Metrics", nullptr, &showPassMetricsWindow);
      ImGui::MenuItem("Gizmo Controls", nullptr, &showGizmoControlsWindow);
      ImGui::MenuItem("Telemetry", nullptr, &showTelemetrySettingsWindow);
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Camera")) {
      ImGui::MenuItem("Controller", nullptr, &showCameraControllerWindow);
      ImGui::MenuItem("Help", nullptr, &showCameraHelpWindow);
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }

  void drawFramePacingWindow() {
    if (!ImGui::Begin(kFramePacingWindowName, &showFramePacingWindow)) {
      ImGui::End();
      return;
    }

    const SwapchainPresentMode activeMode = gpu.getSwapchainPresentMode();
    const std::string_view activeModeName = presentModeLabel(activeMode);
    ImGui::Text("Present mode: %.*s", static_cast<int>(activeModeName.size()),
                activeModeName.data());
    ImGui::Text("Swapchain images: %u", gpu.getSwapchainImageCount());

    const bool canChangePresentMode = gpu.supportsSwapchainPresentModeChange();
    bool vsyncEnabled = activeMode == SwapchainPresentMode::Mailbox ||
                        activeMode == SwapchainPresentMode::Fifo;
    ImGui::BeginDisabled(!canChangePresentMode);
    if (ImGui::Checkbox("VSync (Mailbox)", &vsyncEnabled)) {
      pendingPresentMode = vsyncEnabled ? SwapchainPresentMode::Mailbox
                                        : SwapchainPresentMode::Immediate;
      framePacingStatus.clear();
    }
    ImGui::EndDisabled();
    if (!canChangePresentMode) {
      ImGui::TextWrapped(
          "This graphics backend cannot change present mode at runtime.");
    } else {
      ImGui::TextDisabled(
          "VSync prefers Mailbox, falls back to FIFO, and recreates the "
          "swapchain.");
    }

    ImGui::Separator();
    const bool hasApplication = application != nullptr;
    bool frameLimitEnabled =
        hasApplication && application->frameRateLimit() != 0u;
    ImGui::BeginDisabled(!hasApplication);
    if (ImGui::Checkbox("Enable Frame Limit", &frameLimitEnabled)) {
      pendingFrameRateLimit =
          frameLimitEnabled ? static_cast<uint32_t>(frameRateLimitFps) : 0u;
    }
    ImGui::BeginDisabled(!frameLimitEnabled);
    if (ImGui::InputInt("FPS Limit", &frameRateLimitFps, 1, 10)) {
      frameRateLimitFps = std::clamp(frameRateLimitFps, 1, 1000);
      pendingFrameRateLimit = static_cast<uint32_t>(frameRateLimitFps);
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();

    if (hasApplication) {
      const uint32_t activeLimit = application->frameRateLimit();
      if (activeLimit != 0u) {
        ImGui::Text("Active limit: %u FPS", activeLimit);
      } else {
        ImGui::TextUnformatted("Active limit: unlimited");
      }
      ImGui::TextDisabled(
          "The limiter sleeps on the CPU and is independent of VSync.");
    }

    if (!framePacingStatus.empty()) {
      ImGui::Separator();
      if (framePacingStatusIsError) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s",
                           framePacingStatus.c_str());
      } else {
        ImGui::TextWrapped("%s", framePacingStatus.c_str());
      }
    }

    ImGui::End();
  }

  void updateMetricGraphs(double deltaSeconds) {
    graphSampleAccumulatorSeconds += deltaSeconds;
    if (graphSampleAccumulatorSeconds < kMetricGraphUpdateIntervalSeconds) {
      return;
    }

    const float frametimeMs =
        sanitizeSample(static_cast<float>(deltaSeconds * 1000.0));
    const float instantFps =
        sanitizeSample(deltaSeconds > kMetricSampleMinDeltaSeconds
                           ? static_cast<float>(1.0 / deltaSeconds)
                           : 0.0f);
    fpsGraph->pushSample(instantFps);
    frametimeGraph->pushSample(frametimeMs);
    graphSampleAccumulatorSeconds = std::fmod(
        graphSampleAccumulatorSeconds, kMetricGraphUpdateIntervalSeconds);
  }

  void updateAntiAliasingDiagnosticsLog(double deltaSeconds) {
    sanitizeAntiAliasingSettings(renderSettings.antiAliasing);
    const auto &aa = renderSettings.antiAliasing;
    if (!aa.debug.logDiagnostics) {
      antiAliasingLogAccumulatorSeconds = 0.0;
      return;
    }

    const double interval = static_cast<double>(
        std::clamp(aa.debug.diagnosticLogIntervalSeconds, 0.033f, 5.0f));
    antiAliasingLogAccumulatorSeconds += std::max(deltaSeconds, 0.0);
    if (antiAliasingLogAccumulatorSeconds < interval ||
        lastAntiAliasingLogFrame == frameMetrics.frameIndex) {
      return;
    }

    const bool temporalFeaturePresent = hasTemporalAAFeature(renderPipeline);
    const std::string diagnostics = antiAliasingDiagnosticsSummary(
        aa, frameMetrics, temporalFeaturePresent);
    NURI_LOG_INFO("AA interval diagnostics: frame=%llu dtMs=%.3f %s",
                  static_cast<unsigned long long>(frameMetrics.frameIndex),
                  frameDeltaSeconds * 1000.0, diagnostics.c_str());
    lastAntiAliasingLogFrame = frameMetrics.frameIndex;
    antiAliasingLogAccumulatorSeconds =
        std::fmod(antiAliasingLogAccumulatorSeconds, interval);
  }

  void beginFrame() {
    NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
    platform->newFrame();
    ImGui::NewFrame();
    inspectorWindowVisible = false;
    inspectorWindowMinX = 0.0f;
    uiDependencyTextures.clear();
    uiDependencyTextureAccessModes.clear();
    drawMainMenuBar();
    drawDockspaceRoot();
  }

  Result<RenderGraphGraphicsPassDesc, std::string> endFrame() {
    NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);

    if (renderGraphTelemetry != nullptr) {
      if (showRenderGraphTelemetryWindow) {
        renderGraphTelemetry->requestCapture(
            RenderGraphTelemetryLevel::Metadata);
      }
      if (showPassMetricsWindow || passMetricsState.recording) {
        renderGraphTelemetry->requestCapture(
            RenderGraphTelemetryLevel::PassTimings);
      }
    }

    if (telemetryOverlayState.showImGuiMetricsWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::ShowMetricsWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      ImGui::ShowMetricsWindow(&telemetryOverlayState.showImGuiMetricsWindow);
      NURI_PROFILER_ZONE_END();
    }

    NURI_PROFILER_ZONE("ImGuiEditor::UpdateMetricsAndLogs",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    frameTimeDisplaySampler.tick(
        frameDeltaSeconds,
        overlayGpuFrameTimeSample(gpu.getLatestCompletedGpuTimingReport()));
    updateMetricGraphs(std::max(frameDeltaSeconds, 0.0));
    updateAntiAliasingDiagnosticsLog(frameDeltaSeconds);

    logUpdateAccumulatorSeconds += std::max(frameDeltaSeconds, 0.0);
    if (logUpdateAccumulatorSeconds >= kLogUpdateIntervalSeconds) {
      logModel.update(logFilterState);
      logUpdateAccumulatorSeconds =
          std::fmod(logUpdateAccumulatorSeconds, kLogUpdateIntervalSeconds);
    }
    validateSelectionState();
    syncSelectionWindows();
    NURI_PROFILER_ZONE_END();

#ifdef IMGUI_HAS_DOCK
    if (dockLayoutState.logDockId != 0) {
      ImGui::SetNextWindowDockID(dockLayoutState.logDockId, ImGuiCond_Once);
    }
#endif

    const ImGuiViewport *viewport = ImGui::GetMainViewport();
#ifndef IMGUI_HAS_DOCK
    setLogWindowPlacementWithoutDock(viewport);
#endif

    ScopedScratch scopedScratch(scratchArena);
    if (showHierarchyWindow) {
#ifdef IMGUI_HAS_DOCK
      if (dockLayoutState.hierarchyDockId != 0) {
        ImGui::SetNextWindowDockID(dockLayoutState.hierarchyDockId,
                                   ImGuiCond_Once);
      }
#endif
      NURI_PROFILER_ZONE("ImGuiEditor::DrawHierarchyWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawHierarchyWindow();
      NURI_PROFILER_ZONE_END();
    }
    if (showInspectorWindow) {
#ifdef IMGUI_HAS_DOCK
      if (dockLayoutState.inspectorDockId != 0) {
        ImGui::SetNextWindowDockID(dockLayoutState.inspectorDockId,
                                   ImGuiCond_Once);
      }
#endif
      NURI_PROFILER_ZONE("ImGuiEditor::DrawInspectorWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawInspectorWindow();
      NURI_PROFILER_ZONE_END();
    }
    if (showAnimationPlayerWindow && animationPlayer != nullptr) {
#ifdef IMGUI_HAS_DOCK
      if (dockLayoutState.inspectorDockId != 0) {
        ImGui::SetNextWindowDockID(dockLayoutState.inspectorDockId,
                                   ImGuiCond_Once);
      }
#endif
      NURI_PROFILER_ZONE("ImGuiEditor::DrawAnimationPlayerWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawAnimationPlayerWindow();
      NURI_PROFILER_ZONE_END();
    }

    NURI_PROFILER_ZONE("ImGuiEditor::DrawLogWindow",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    ImGui::Begin(kLogWindowName);
    drawLogWindow(logModel, logFilterState, scopedScratch.resource());
    ImGui::End();

    if (showRenderGraphTelemetryWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawRenderGraphTelemetryWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (ImGui::Begin(kRenderGraphTelemetryWindowName,
                       &showRenderGraphTelemetryWindow)) {
        drawRenderGraphTelemetryWindow(telemetryState, renderGraphTelemetry,
                                       window.nativeHandle());
      }
      ImGui::End();
      NURI_PROFILER_ZONE_END();
    }
    if (showPassMetricsWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawPassMetricsWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (ImGui::Begin(kPassMetricsWindowName, &showPassMetricsWindow)) {
        drawPassMetricsWindow(passMetricsState, renderGraphTelemetry, gpu);
      }
      ImGui::End();
      NURI_PROFILER_ZONE_END();
    }
    if (passMetricsState.showRecordingWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawPassMetricsRecordingWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (ImGui::Begin(kPassMetricsRecordingWindowName,
                       &passMetricsState.showRecordingWindow)) {
        drawPassMetricRecordingWindow(passMetricsState);
      }
      ImGui::End();
      NURI_PROFILER_ZONE_END();
    }
    if (showFontCompilerWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawFontCompilerWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawFontCompilerWindow(showFontCompilerWindow, fontCompilerState,
                             textSystem, window.nativeHandle());
      NURI_PROFILER_ZONE_END();
    }
    if (showBakeryWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawBakeryWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawBakeryWindow(showBakeryWindow, bakeryState, bakery,
                       scopedScratch.resource(), window.nativeHandle());
      NURI_PROFILER_ZONE_END();
    }
    if (showRenderPassesWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawRenderPassesWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawRenderPassesWindow(showRenderPassesWindow, renderSettings,
                             renderPipeline, frameMetrics, selectedPassIndex);
      NURI_PROFILER_ZONE_END();
    }
    if (showTextureFilteringWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawTextureFilteringWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawTextureFilteringWindow(showTextureFilteringWindow, renderSettings,
                                 gpu);
      NURI_PROFILER_ZONE_END();
    }
    if (showAntiAliasingWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawAntiAliasingWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawAntiAliasingWindow(showAntiAliasingWindow, renderSettings,
                             renderPipeline, frameMetrics);
      NURI_PROFILER_ZONE_END();
    }
    if (showAmbientOcclusionWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawAmbientOcclusionWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawAmbientOcclusionWindow(showAmbientOcclusionWindow, renderSettings,
                                 frameMetrics);
      NURI_PROFILER_ZONE_END();
    }
    if (showHDRPostProcessWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawHDRPostProcessWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawHDRPostProcessWindow(showHDRPostProcessWindow, renderSettings,
                               frameMetrics);
      NURI_PROFILER_ZONE_END();
    }
    if (showShadowsWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawShadowsWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawShadowsWindow(showShadowsWindow, renderSettings, gpu,
                        shadowDebugFrameData, frameMetrics, shadowInspectResult,
                        shadowPreviewTexture, uiDependencyTextures,
                        uiDependencyTextureAccessModes);
      NURI_PROFILER_ZONE_END();
    }
    if (showCameraControllerWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawCameraControllerWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (cameraSystem != nullptr &&
          ImGui::Begin(kCameraControllerWindowName,
                       &showCameraControllerWindow)) {
        drawCameraControllerContents(*cameraSystem, cameraControllerState);
      }
      if (cameraSystem == nullptr) {
        showCameraControllerWindow = false;
      } else {
        ImGui::End();
      }
      NURI_PROFILER_ZONE_END();
    }
    if (showCameraHelpWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawCameraHelpWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (ImGui::Begin(kCameraHelpWindowName, &showCameraHelpWindow)) {
        drawCameraHelpContents();
      }
      ImGui::End();
      NURI_PROFILER_ZONE_END();
    }
    if (showFramePacingWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawFramePacingWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      drawFramePacingWindow();
      NURI_PROFILER_ZONE_END();
    }
    if (showTelemetrySettingsWindow) {
      NURI_PROFILER_ZONE("ImGuiEditor::DrawTelemetrySettingsWindow",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (ImGui::Begin(kTelemetryWindowName, &showTelemetrySettingsWindow)) {
        ImGui::Checkbox("Show Overlay", &telemetryOverlayState.overlayEnabled);
        ImGui::Separator();
        ImGui::Checkbox("FPS + ms", &telemetryOverlayState.showFpsMs);
        ImGui::Checkbox("Instance counts",
                        &telemetryOverlayState.showInstanceStats);
        ImGui::Checkbox("Draw / tess stats",
                        &telemetryOverlayState.showDrawTessStats);
        ImGui::Checkbox("Indirect stats",
                        &telemetryOverlayState.showIndirectStats);
        ImGui::Checkbox("Debug draw stats",
                        &telemetryOverlayState.showDebugDrawStats);
        ImGui::Checkbox("Patch heatmap",
                        &telemetryOverlayState.showPatchHeatmap);
        ImGui::Checkbox("Dispatch stats",
                        &telemetryOverlayState.showDispatchStats);
        ImGui::Checkbox("Graphs", &telemetryOverlayState.showGraphs);
        ImGui::Separator();
        ImGui::Checkbox("ImGui Metrics Window",
                        &telemetryOverlayState.showImGuiMetricsWindow);
      }
      ImGui::End();
      NURI_PROFILER_ZONE_END();
    }
    NURI_PROFILER_ZONE_END();

    NURI_PROFILER_ZONE("ImGuiEditor::DrawFpsOverlay",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    float overlayRightBoundaryX = 0.0f;
    if (inspectorWindowVisible) {
      if (const ImGuiViewport *viewport = ImGui::GetMainViewport();
          viewport != nullptr &&
          inspectorWindowMinX >
              viewport->WorkPos.x + viewport->WorkSize.x * 0.5f) {
        overlayRightBoundaryX = inspectorWindowMinX;
      }
    }
    drawFpsOverlay(*fpsGraph, *frametimeGraph, frameTimeDisplaySampler.values(),
                   frameMetrics, renderSettings, telemetryOverlayState,
                   overlayRightBoundaryX);
    NURI_PROFILER_ZONE_END();

    NURI_PROFILER_ZONE("ImGuiEditor::FinalizeImGuiFrame",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    ImGui::EndFrame();
    ImGui::Render();
    NURI_PROFILER_ZONE_END();

    auto passResult = [&]() {
      Result<RenderGraphGraphicsPassDesc, std::string> result =
          Result<RenderGraphGraphicsPassDesc, std::string>::makeError(
              std::string{});
      NURI_PROFILER_ZONE("ImGuiEditor::BuildGraphicsPassDesc",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      result =
          renderer->buildGraphicsPassDesc(gpu.getSwapchainFormat(), frameIndex);
      NURI_PROFILER_ZONE_END();
      return result;
    }();
    if (passResult.hasValue()) {
      RenderGraphGraphicsPassDesc &pass = passResult.value();
      pass.dependencyTextures = std::span<const TextureHandle>(
          uiDependencyTextures.data(), uiDependencyTextures.size());
      pass.dependencyTextureAccessModes =
          std::span<const RenderGraphAccessMode>(
              uiDependencyTextureAccessModes.data(),
              uiDependencyTextureAccessModes.size());
    }
    return passResult;
  }

  void drawDockspaceRoot() {
#ifdef IMGUI_HAS_DOCK
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    setDockspaceWindowPlacement(viewport);

    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

    ImGui::Begin(kDockspaceWindowName, nullptr, dockspaceWindowFlags());
    ImGui::PopStyleVar(3);

    const ImGuiDockNodeFlags dockFlags = ImGuiDockNodeFlags_PassthruCentralNode;
    const ImGuiID dockspaceId = ImGui::GetID(kDockspaceRootId);
    ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), dockFlags);
    dockLayoutState.ensureLayout(dockspaceId, viewport);

    ImGui::End();
#endif
  }

  Window &window;
  GPUDevice &gpu;
  std::unique_ptr<ImGuiGlfwPlatform> platform;
  std::unique_ptr<ImGuiGpuRenderer> renderer;
  FrameTimeDisplaySampler frameTimeDisplaySampler{
      kFrameTimeDisplayUpdateIntervalSeconds};
  std::unique_ptr<LinearGraph> fpsGraph =
      createImPlotLinearGraph(kMetricGraphSampleCount);
  std::unique_ptr<LinearGraph> frametimeGraph =
      createImPlotLinearGraph(kMetricGraphSampleCount);
  double graphSampleAccumulatorSeconds = kMetricGraphUpdateIntervalSeconds;
  double logUpdateAccumulatorSeconds = kLogUpdateIntervalSeconds;
  double antiAliasingLogAccumulatorSeconds = 0.0;
  uint64_t lastAntiAliasingLogFrame = std::numeric_limits<uint64_t>::max();
  bool showBakeryWindow = false;
  bool showFontCompilerWindow = false;
  bool showHierarchyWindow = true;
  bool showInspectorWindow = true;
  bool showAnimationPlayerWindow = false;
  bool showRenderPassesWindow = false;
  bool showLightsWindow = false;
  bool showTextureFilteringWindow = false;
  bool showAntiAliasingWindow = false;
  bool showAmbientOcclusionWindow = false;
  bool showHDRPostProcessWindow = false;
  bool showShadowsWindow = false;
  bool showRenderGraphTelemetryWindow = false;
  bool showPassMetricsWindow = false;
  bool showGizmoControlsWindow = false;
  bool showFramePacingWindow = false;
  bool showTelemetrySettingsWindow = false;
  bool showCameraControllerWindow = false;
  bool showCameraHelpWindow = false;
  bool focusAnimationPlayerWindow = false;
  bool inspectorWindowVisible = false;
  RenderSettings renderSettings{};
  RenderFrameMetrics frameMetrics{};
  std::optional<ShadowDebugFrameData> shadowDebugFrameData{};
  std::optional<ShadowInspectResult> shadowInspectResult{};
  uint64_t lastLoggedShadowInspectRequestId = 0;
  TextureHandle shadowPreviewTexture{};
  size_t selectedPassIndex = 0u;
  double frameDeltaSeconds = 0.0;
  uint64_t frameIndex = 0;
  float inspectorWindowMinX = 0.0f;
  LogModel logModel;
  LogFilterState logFilterState;
  RenderGraphTelemetryUiState telemetryState;
  PassMetricsUiState passMetricsState;
  TelemetryOverlayUiState telemetryOverlayState;
  FontCompilerUiState fontCompilerState;
  BakeryUiState bakeryState;
  SceneSelectionUiState sceneSelectionState;
  CameraControllerWidgetState cameraControllerState{};
  MaybeDockLayoutState dockLayoutState;
  ScratchArena scratchArena;
  std::vector<RenderableInspectorState> renderableInspectorStates{};
  LightEditorDraft lightEditorDraft{};
  SceneEditorSelectionState localSelectionState{};
  NodeId lastObservedSelectionNode = kInvalidNodeId;
  bool pendingRevealSelection = false;
  bool suppressRevealForNextSelectionChange = false;
  bool hierarchyStatsCacheValid = false;
  bool hierarchyTopologyCacheValid = false;
  uint32_t cachedRenderableCount = 0u;
  uint32_t cachedLightCount = 0u;
  uint64_t cachedHierarchyTopologyVersion =
      std::numeric_limits<uint64_t>::max();
  NodeId cachedSelectedPathLeaf = kInvalidNodeId;
  std::vector<NodeId> hierarchyTopologyBuildQueue{};
  size_t hierarchyTopologyBuildQueueCursor = 0u;
  NodeId hierarchyTopologyBuildNode = kInvalidNodeId;
  NodeId hierarchyTopologyBuildNextChild = kInvalidNodeId;
  bool hierarchyTopologyBuildNodeStarted = false;
  bool hierarchyStatsBuilding = false;
  size_t hierarchyStatsBuildCursor = 0u;
  std::vector<HierarchyNodeTopology> hierarchyNodeTopology{};
  std::vector<HierarchyNodeStats> hierarchyNodeStats{};
  std::vector<uint8_t> selectedPathNodeFlags{};
  std::vector<int32_t> selectedPathChildIndices{};
  std::vector<HierarchyVisibleRow> hierarchyVisibleRows{};
  std::vector<uint8_t> hierarchyNodeOpenFlags{};
  std::unordered_set<uint64_t> hierarchyOpenBatchKeys{};
  std::vector<TextureHandle> uiDependencyTextures{};
  std::vector<RenderGraphAccessMode> uiDependencyTextureAccessModes{};
  int hierarchySelectedRowIndex = -1;
  bool hierarchySceneRootOpen = true;
  std::vector<DeferredAnimationAction> deferredAnimationActions{};
  std::optional<SwapchainPresentMode> pendingPresentMode{};
  std::optional<uint32_t> pendingFrameRateLimit{};
  int frameRateLimitFps = 60;
  std::string framePacingStatus{};
  bool framePacingStatusIsError = false;
  Application *application = nullptr;
  RenderScene *scene = nullptr;
  ResourceManager *resources = nullptr;
  RenderPipeline *renderPipeline = nullptr;
  SceneEditorSelectionState *selectionState = nullptr;
  TextSystem *textSystem = nullptr;
  CameraSystem *cameraSystem = nullptr;
  bakery::BakerySystem *bakery = nullptr;
  RenderGraphTelemetryService *renderGraphTelemetry = nullptr;
  EditorAnimationPlayerService *animationPlayer = nullptr;
};

std::unique_ptr<ImGuiEditor>
ImGuiEditor::create(Window &window, GPUDevice &gpu,
                    const EditorServices &services) {
  return std::unique_ptr<ImGuiEditor>(new ImGuiEditor(window, gpu, services));
}

ImGuiEditor::ImGuiEditor(Window &window, GPUDevice &gpu,
                         const EditorServices &services)
    : impl_(std::make_unique<Impl>(window, gpu, services)) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();

  ImGuiIO &io = ImGui::GetIO();
  // Ensure the default font exists before the first NewFrame().
  // (Some builds/configs can end up with an empty font atlas otherwise.)
  if (io.Fonts && io.Fonts->Fonts.empty()) {
    io.Fonts->AddFontDefault();
  }
  io.FontDefault =
      io.Fonts && !io.Fonts->Fonts.empty() ? io.Fonts->Fonts[0] : nullptr;
  if (io.Fonts) {
    io.Fonts->Build();
  }
#if defined(ImGuiConfigFlags_DockingEnable) || defined(IMGUI_HAS_DOCK)
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  // Make docking work without needing to hold Shift.
  io.ConfigDockingWithShift = false;
#endif
  io.IniFilename = nullptr;

  impl_->platform = ImGuiGlfwPlatform::create(impl_->window);
  impl_->renderer = ImGuiGpuRenderer::create(impl_->gpu);
}

ImGuiEditor::~ImGuiEditor() {
  impl_->resetSceneUiState();
  impl_->renderer.reset();
  impl_->platform.reset();
  if (ImPlot::GetCurrentContext() != nullptr) {
    ImPlot::DestroyContext();
  }
  ImGui::DestroyContext();
}

void ImGuiEditor::setFrameDeltaSeconds(double deltaTime) {
  if (!impl_) {
    return;
  }
  if (!std::isfinite(deltaTime) || deltaTime < 0.0) {
    impl_->frameDeltaSeconds = 0.0;
    return;
  }
  impl_->frameDeltaSeconds = deltaTime;
}

void ImGuiEditor::setFrameIndex(uint64_t frameIndex) {
  if (!impl_) {
    return;
  }
  impl_->frameIndex = frameIndex;
}

void ImGuiEditor::setFrameMetrics(const RenderFrameMetrics &metrics) {
  if (!impl_) {
    return;
  }
  impl_->frameMetrics = metrics;
}

void ImGuiEditor::setRenderSettings(const RenderSettings &settings) {
  if (!impl_) {
    return;
  }
  impl_->renderSettings = settings;
  sanitizeTextureFilteringSettings(impl_->renderSettings.textureFiltering);
  sanitizeToneMapSettings(impl_->renderSettings.toneMap);
  sanitizeHDRPostProcessSettings(impl_->renderSettings.hdrPostProcess);
  sanitizeAntiAliasingSettings(impl_->renderSettings.antiAliasing);
  sanitizeAmbientOcclusionSettings(impl_->renderSettings.ambientOcclusion,
                                   impl_->renderSettings.opaque,
                                   impl_->renderSettings.antiAliasing);
  sanitizeShadowSettings(impl_->renderSettings.shadow);
  sanitizeTransmissionSettings(impl_->renderSettings.transmission);
}

void ImGuiEditor::setShadowDebugResources(
    const std::optional<ShadowDebugFrameData> &debug,
    TextureHandle previewTexture) {
  if (!impl_) {
    return;
  }
  impl_->shadowDebugFrameData = debug;
  impl_->shadowPreviewTexture = previewTexture;
}

void ImGuiEditor::setShadowInspectResult(
    const std::optional<ShadowInspectResult> &inspectResult) {
  if (!impl_) {
    return;
  }
  if (inspectResult.has_value()) {
    if (impl_->lastLoggedShadowInspectRequestId != inspectResult->requestId) {
      NURI_LOG_INFO(
          "ImGuiEditor: shadow inspect result request=%llu "
          "valid=%s receiverDepth=%.6f receiverCompareDepth=%.6f "
          "sampledDepth=%.6f cascade=%u cascadeBlend=%.6f",
          static_cast<unsigned long long>(inspectResult->requestId),
          inspectResult->valid ? "true" : "false", inspectResult->receiverDepth,
          inspectResult->receiverCompareDepth, inspectResult->sampledDepth,
          inspectResult->cascadeIndex, inspectResult->cascadeBlendFactor);
      impl_->lastLoggedShadowInspectRequestId = inspectResult->requestId;
    }
    impl_->shadowInspectResult = inspectResult;
  }
}

void ImGuiEditor::syncCameraControllerWidgetStateFromCamera(
    const Camera &camera) {
  if (!impl_) {
    return;
  }
  nuri::syncCameraControllerWidgetStateFromCamera(camera,
                                                  impl_->cameraControllerState);
}

void ImGuiEditor::setSceneSelectionUi(
    std::span<const EditorSceneSelectionOption> scenes,
    std::string_view selectedSceneId, uint64_t version,
    std::string_view hotkeyHint, const EditorSceneLoadUiState &load) {
  if (!impl_) {
    return;
  }
  impl_->sceneSelectionState.set(scenes, selectedSceneId, version, hotkeyHint,
                                 load);
}

void ImGuiEditor::resetSceneUiState() {
  if (!impl_) {
    return;
  }
  impl_->resetSceneUiState();
}

void ImGuiEditor::bindScene(RenderScene &scene) {
  if (!impl_) {
    return;
  }
  impl_->resetSceneUiState();
  impl_->scene = &scene;
}

std::optional<std::string> ImGuiEditor::takeSceneSelectionRequest() {
  return impl_
             ? std::exchange(impl_->sceneSelectionState.pendingSelectionRequest,
                             std::nullopt)
             : std::nullopt;
}

bool ImGuiEditor::takeSceneCancelRequest() {
  return impl_ != nullptr &&
         std::exchange(impl_->sceneSelectionState.pendingCancelRequest, false);
}

bool *ImGuiEditor::gizmoControlsWindowOpenState() {
  return impl_ ? &impl_->showGizmoControlsWindow : nullptr;
}

bool *ImGuiEditor::lightsWindowOpenState() {
  return impl_ ? &impl_->showLightsWindow : nullptr;
}

bool ImGuiEditor::isGizmoControlsWindowOpen() const {
  return impl_ != nullptr && impl_->showGizmoControlsWindow;
}

bool ImGuiEditor::isLightsWindowOpen() const {
  return impl_ != nullptr && impl_->showLightsWindow;
}

bool ImGuiEditor::isShadowsWindowOpen() const {
  return impl_ != nullptr && impl_->showShadowsWindow;
}

RenderSettings ImGuiEditor::renderSettings() const {
  if (!impl_) {
    return RenderSettings{};
  }
  RenderSettings settings = impl_->renderSettings;
  sanitizeTextureFilteringSettings(settings.textureFiltering);
  sanitizeToneMapSettings(settings.toneMap);
  sanitizeHDRPostProcessSettings(settings.hdrPostProcess);
  sanitizeTransmissionSettings(settings.transmission);
  sanitizeAntiAliasingSettings(settings.antiAliasing);
  sanitizeAmbientOcclusionSettings(settings.ambientOcclusion, settings.opaque,
                                   settings.antiAliasing);
  sanitizeShadowSettings(settings.shadow);
  return settings;
}

bool ImGuiEditor::wantsCaptureKeyboard() const {
  if (!ImGui::GetCurrentContext()) {
    return false;
  }
  const ImGuiIO &io = ImGui::GetIO();
  return io.WantCaptureKeyboard;
}

bool ImGuiEditor::wantsCaptureMouse() const {
  if (!ImGui::GetCurrentContext()) {
    return false;
  }
  const ImGuiIO &io = ImGui::GetIO();
  return io.WantCaptureMouse;
}

void ImGuiEditor::applyDeferredUiActions() {
  if (!impl_) {
    return;
  }
  impl_->applyDeferredUiActions();
}

void ImGuiEditor::beginFrame() { impl_->beginFrame(); }

Result<RenderGraphGraphicsPassDesc, std::string> ImGuiEditor::endFrame() {
  return impl_->endFrame();
}

} // namespace nuri
