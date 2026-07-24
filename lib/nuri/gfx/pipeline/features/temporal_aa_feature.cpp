#include "nuri/gfx/pipeline/features/temporal_aa_feature.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/fullscreen.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/pch.h"
#include <algorithm>
#include <array>
#include <bit>
namespace nuri {
namespace {
constexpr uint32_t kTaaResolveFlagHistoryValid = 1u << 0u;
constexpr uint32_t kTaaResolveFlagPreviousVelocityValid = 1u << 1u;
constexpr uint32_t kTaaResolveFlagDepthReject = 1u << 2u;
constexpr uint32_t kTaaResolveFlagVelocityReject = 1u << 3u;
constexpr uint32_t kTaaResolveFlagNeighborhoodClamp = 1u << 4u;
constexpr uint32_t kTaaResolveFlagAdaptiveBlend = 1u << 5u;
constexpr uint32_t kTaaResolveFlagClampBlendAttenuation = 1u << 6u;
constexpr uint32_t kTaaResolveFlagNeighborhoodFallback = 1u << 7u;
constexpr uint32_t kTaaResolveFlagReactiveMask = 1u << 8u;
constexpr uint32_t kTaaResolveFlagVelocityDilation = 1u << 9u;
constexpr uint32_t kTaaResolveFlagPreviousDepthValid = 1u << 10u;
constexpr uint32_t kTaaResolveFlagSharpen = 1u << 11u;
constexpr uint32_t kTaaResolveFlagStaticFrame = 1u << 12u;
constexpr uint32_t kTaaResolvePhase5Flags =
    kTaaResolveFlagDepthReject | kTaaResolveFlagVelocityReject |
    kTaaResolveFlagNeighborhoodClamp | kTaaResolveFlagAdaptiveBlend |
    kTaaResolveFlagClampBlendAttenuation | kTaaResolveFlagNeighborhoodFallback;
constexpr uint32_t kTaaResolveModeResolve = 0u;
constexpr uint32_t kTaaResolveModeCopyCurrent = 1u;
constexpr uint32_t kTaaResolveModePreviousHistory = 2u;
constexpr uint32_t kTaaResolveModeHistoryValidity = 3u;
constexpr uint32_t kTaaResolveModeRejectionMask = 4u;
constexpr uint32_t kTaaResolveModeBlendFactor = 5u;
constexpr uint32_t kTaaResolveModeClampDelta = 6u;
constexpr uint32_t kTaaResolveModePixelInspector = 7u;
constexpr uint32_t kTaaResolveModeVelocityMotionVectors = 8u;
constexpr uint32_t kTaaResolveModeVelocityMagnitude = 9u;
constexpr uint32_t kTaaResolveModeReactiveMask = 10u;
constexpr uint32_t kTaaResolveModeDisocclusionMask = 11u;
constexpr uint32_t kTaaResolveModeVelocityDilation = 12u;
constexpr uint32_t kTaaResolveModeReprojectedHistory = 13u;
constexpr uint32_t kTaaResolveModeResolveConfidence = 14u;
constexpr uint32_t kTaaResolveModeClampDiagnostics = 15u;
constexpr uint32_t kTaaResolveModePreviousVelocity = 16u;
constexpr uint32_t kTaaResolveModeHdrWeight = 17u;
constexpr uint32_t kTaaResolveModeHistoryFilterDelta = 18u;
constexpr uint32_t kTaaResolveModeDisocclusionFallback = 19u;
constexpr uint32_t kTaaResolveModeSplitCompare = 20u;
constexpr uint32_t kTaaResolveModeCopyHistoryToScene = 21u;
constexpr uint32_t kTaaResolveModeTemporalConfidence = 22u;
constexpr uint32_t kTaaResolveModePreviousDepthRejection = 23u;
constexpr uint32_t kTaaResolveModeStabilityDiagnostics = 24u;
constexpr uint32_t kTaaResolveModeStabilityOwnership = 25u;
constexpr uint32_t kTaaResolveModePatchProbe = 26u;
constexpr uint32_t kTaaResolveModeMotionFilter = 27u;
constexpr uint32_t kTaaHistoryFilterModeCatmullRom = 0u;
constexpr uint32_t kTaaHistoryFilterModeBilinear = 1u;
constexpr uint32_t kTaaResolvePassDebugColor = 0xffaa55ffu;
constexpr uint32_t kTaaCopyBackPassDebugColor = 0xff8844ffu;
constexpr uint32_t kTaaDebugPassDebugColor = 0xffcc77ffu;
constexpr uint32_t kDrawDebugColor = 0xffaa44ffu;
constexpr float kStaticFrameVelocityEpsilon = 1.0e-5f;
[[nodiscard]] constexpr uint32_t floatBits(float value) noexcept {
  return std::bit_cast<uint32_t>(value);
}
struct TAAResolvePushConstants {
  uint32_t currentTexId = 0u;
  uint32_t historyTexId = 0u;
  uint32_t depthTexId = 0u;
  uint32_t previousDepthTexId = 0u;
  uint32_t velocityTexId = 0u;
  uint32_t previousVelocityTexId = 0u;
  uint32_t reactiveMaskTexId = 0u;
  uint32_t linearSamplerId = 0u;
  uint32_t pointSamplerId = 0u;
  uint32_t flags = 0u;
  uint32_t mode = 0u;
  uint32_t currentWeightBits = 0u;
  uint32_t inverseWidthBits = 0u;
  uint32_t inverseHeightBits = 0u;
  uint32_t depthThresholdBits = 0u;
  uint32_t velocityThresholdBits = 0u;
  uint32_t velocityBlendScaleBits = 0u;
  uint32_t disocclusionWeightBits = 0u;
  uint32_t clampAttenuationBits = 0u;
  uint32_t varianceGammaBits = 0u;
  uint32_t hdrWeightStrengthBits = 0u;
  uint32_t reactiveCurrentWeightBits = 0u;
  uint32_t reactiveStrengthBits = 0u;
  uint32_t velocityDilationDepthThresholdBits = 0u;
  uint32_t clampMode = 0u;
  uint32_t hdrWeightingMode = 0u;
  uint32_t velocityDilationMode = 0u;
  uint32_t motionCurrentWeightBits = 0u;
  uint32_t clampCurrentWeightBits = 0u;
  uint32_t historyFilterMode = 0u;
  uint32_t previousRawJitterDeltaUvXBits = 0u;
  uint32_t previousRawJitterDeltaUvYBits = 0u;
};
static_assert(sizeof(TAAResolvePushConstants) <= 128);
[[nodiscard]] inline bool
isLegacyTAAEnabled(const RenderFrameContext &frame) noexcept {
  return frame.presentationAA.valid
             ? frame.presentationAA.reconstruction ==
                   ColorReconstruction::LegacyTAA
             : sanitizeAntiAliasingMode(
                   renderSettingsOrDefault(frame).antiAliasing.mode) ==
                   AntiAliasingMode::TAA;
}
[[nodiscard]] inline bool
isVelocityDebugView(AntiAliasingDebugView view) noexcept {
  return view == AntiAliasingDebugView::MotionVectors ||
         view == AntiAliasingDebugView::VelocityMagnitude;
}
[[nodiscard]] inline bool staticFrameVelocitySanitizationEligible(
    const AntiAliasingFrameMetrics &metrics) noexcept {
  return (metrics.velocityPassCount > 0u ||
          metrics.motionVectorDepthReprojectionGenerated) &&
         metrics.previousTransformCacheValid &&
         metrics.velocityMissingPreviousTransformCount == 0u &&
         metrics.velocityAnimatedResponsiveCount == 0u &&
         metrics.velocityAnimatedPreviousGeometryCount == 0u &&
         metrics.velocityTessellatedSkippedDrawCount == 0u &&
         metrics.velocityEstimatedMaxMagnitude <= kStaticFrameVelocityEpsilon &&
         metrics.velocityStaticResidualEstimate <=
             kStaticFrameVelocityEpsilon &&
         metrics.cameraPositionDelta <= kStaticFrameVelocityEpsilon;
}
struct TAAEvaluationDebugDesc {
  uint32_t mode = 0u;
  std::string_view label{};
  bool AntiAliasingFrameMetrics::*renderedMetric = nullptr;
  [[nodiscard]] explicit operator bool() const noexcept { return mode != 0u; }
};
[[nodiscard]] TAAEvaluationDebugDesc
taaEvaluationDebugDesc(AntiAliasingDebugView view) noexcept {
  switch (view) {
  case AntiAliasingDebugView::TAAHistoryValidity:
    return {kTaaResolveModeHistoryValidity, "TAA History Validity Debug Pass",
            &AntiAliasingFrameMetrics::taaHistoryValidityDebugViewRendered};
  case AntiAliasingDebugView::TAARejectionMask:
    return {kTaaResolveModeRejectionMask, "TAA Rejection Mask Debug Pass"};
  case AntiAliasingDebugView::TAABlendFactor:
    return {kTaaResolveModeBlendFactor, "TAA Blend Factor Debug Pass"};
  case AntiAliasingDebugView::TAAClampDelta:
    return {kTaaResolveModeClampDelta, "TAA Clamp Delta Debug Pass"};
  case AntiAliasingDebugView::TAAPixelInspector:
    return {kTaaResolveModePixelInspector, "TAA Pixel Inspector Debug Pass",
            &AntiAliasingFrameMetrics::taaPixelInspectorDebugViewRendered};
  case AntiAliasingDebugView::TAADisocclusionMask:
    return {kTaaResolveModeDisocclusionMask, "TAA Disocclusion Mask Debug Pass",
            &AntiAliasingFrameMetrics::taaDisocclusionMaskDebugViewRendered};
  case AntiAliasingDebugView::TAAVelocityDilation:
    return {kTaaResolveModeVelocityDilation, "TAA Velocity Dilation Debug Pass",
            &AntiAliasingFrameMetrics::taaVelocityDilationDebugViewRendered};
  case AntiAliasingDebugView::TAAReprojectedHistory:
    return {kTaaResolveModeReprojectedHistory,
            "TAA Reprojected History Debug Pass"};
  case AntiAliasingDebugView::TAAResolveConfidence:
    return {kTaaResolveModeResolveConfidence,
            "TAA Resolve Confidence Debug Pass"};
  case AntiAliasingDebugView::TAAClampDiagnostics:
    return {kTaaResolveModeClampDiagnostics,
            "TAA Clamp Diagnostics Debug Pass"};
  case AntiAliasingDebugView::TAAHdrWeight:
    return {kTaaResolveModeHdrWeight, "TAA HDR Weight Debug Pass",
            &AntiAliasingFrameMetrics::taaHdrWeightDebugViewRendered};
  case AntiAliasingDebugView::TAAHistoryFilterDelta:
    return {kTaaResolveModeHistoryFilterDelta,
            "TAA History Filter Delta Debug Pass",
            &AntiAliasingFrameMetrics::taaHistoryFilterDeltaDebugViewRendered};
  case AntiAliasingDebugView::TAADisocclusionFallback:
    return {
        kTaaResolveModeDisocclusionFallback,
        "TAA Disocclusion Fallback Debug Pass",
        &AntiAliasingFrameMetrics::taaDisocclusionFallbackDebugViewRendered};
  case AntiAliasingDebugView::TAASplitCompare:
    return {kTaaResolveModeSplitCompare, "TAA Split Compare Debug Pass",
            &AntiAliasingFrameMetrics::taaSplitCompareDebugViewRendered};
  case AntiAliasingDebugView::TAATemporalConfidence:
    return {kTaaResolveModeTemporalConfidence,
            "TAA Temporal Confidence Debug Pass",
            &AntiAliasingFrameMetrics::taaTemporalConfidenceDebugViewRendered};
  case AntiAliasingDebugView::TAAPreviousDepthRejection:
    return {
        kTaaResolveModePreviousDepthRejection,
        "TAA Previous Depth Rejection Debug Pass",
        &AntiAliasingFrameMetrics::taaPreviousDepthRejectionDebugViewRendered};
  case AntiAliasingDebugView::TAAStabilityDiagnostics:
    return {
        kTaaResolveModeStabilityDiagnostics,
        "TAA Stability Diagnostics Debug Pass",
        &AntiAliasingFrameMetrics::taaStabilityDiagnosticsDebugViewRendered};
  case AntiAliasingDebugView::TAAStabilityOwnership:
    return {kTaaResolveModeStabilityOwnership,
            "TAA Stability Ownership Debug Pass",
            &AntiAliasingFrameMetrics::taaStabilityOwnershipDebugViewRendered};
  case AntiAliasingDebugView::TAAPatchProbe:
    return {kTaaResolveModePatchProbe, "TAA Patch Probe Debug Pass",
            &AntiAliasingFrameMetrics::taaPatchProbeDebugViewRendered};
  case AntiAliasingDebugView::TAAMotionFilter:
    return {kTaaResolveModeMotionFilter, "TAA Motion Filter Debug Pass",
            &AntiAliasingFrameMetrics::taaMotionFilterDebugViewRendered};
  default:
    return {};
  }
}
struct TAAResolveTuning {
  float currentWeight;
  float sharpenStrength;
  float sharpenConfidenceThreshold;
  float depthDiscontinuityThreshold;
  float velocityRejectionThreshold;
  float velocityBlendScale;
  float motionCurrentWeight;
  float disocclusionCurrentWeight;
  float clampCurrentWeight;
  float clampBlendAttenuation;
  float varianceGamma;
  float hdrWeightStrength;
  float reactiveCurrentWeight;
  float reactiveStrength;
  float velocityDilationDepthThreshold;
  TemporalAAClampMode clampMode;
  TemporalAAHdrWeightingMode hdrWeightingMode;
  TemporalAAVelocityDilationMode velocityDilationMode;
  TemporalAAHistoryFilterMode historyFilterMode;
  bool sharpenEnabled;
};
[[nodiscard]] inline std::filesystem::path
resolveShaderBasePath(const RuntimeCompositeConfig &config) {
  if (!config.shaderBasePath.empty()) {
    return config.shaderBasePath;
  }
  if (!config.fullscreenVertex.empty()) {
    return config.fullscreenVertex.parent_path();
  }
  if (!config.sceneCopyFragment.empty()) {
    return config.sceneCopyFragment.parent_path();
  }
  if (!config.presentFragment.empty()) {
    return config.presentFragment.parent_path();
  }
  return {};
}
void destroyTemporalPipeline(GPUDevice &gpu, TemporalAAPipelineState &state) {
  if (nuri::isValid(state.pipeline)) {
    gpu.destroyRenderPipeline(state.pipeline);
  }
  for (ShaderHandle shader : state.shaders) {
    if (nuri::isValid(shader)) {
      gpu.destroyShaderModule(shader);
    }
  }
}
Result<bool, std::string>
createTemporalPipeline(GPUDevice &gpu, TemporalAAPipelineState &state,
                       const std::filesystem::path &vertexPath,
                       const std::filesystem::path &fragmentPath,
                       RenderPipelineDesc desc, std::string_view name) {
  Shader shader{name, gpu};
  auto vertex =
      shader.compileFromFile(vertexPath.string(), ShaderStage::Vertex);
  if (vertex.hasError()) {
    return Result<bool, std::string>::makeError(vertex.error());
  }
  state.shaders[0] = vertex.value();
  auto fragment =
      shader.compileFromFile(fragmentPath.string(), ShaderStage::Fragment);
  if (fragment.hasError()) {
    return Result<bool, std::string>::makeError(fragment.error());
  }
  state.shaders[1] = fragment.value();
  desc.vertexShader = state.shaders[0];
  desc.fragmentShader = state.shaders[1];
  auto pipeline = gpu.createRenderPipeline(desc, name);
  if (pipeline.hasError()) {
    return Result<bool, std::string>::makeError(pipeline.error());
  }
  state.pipeline = pipeline.value();
  return Result<bool, std::string>::makeResult(true);
}
[[nodiscard]] uint64_t textureStorageBytes(GPUDevice &gpu,
                                           TextureHandle texture) {
  if (!nuri::isValid(texture)) {
    return 0u;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  return static_cast<uint64_t>(std::max(dimensions.width, 1u)) *
         static_cast<uint64_t>(std::max(dimensions.height, 1u)) *
         formatTexelBytes(gpu.getTextureFormat(texture));
}
[[nodiscard]] TAAResolveTuning sanitizedResolveTuning(
    const RenderSettings::AntiAliasingSettings &settings) noexcept {
  const auto debug = effectiveTemporalAADebugSettings(settings);
  return {
      .currentWeight = debug.taaCurrentFrameWeight,
      .sharpenStrength = debug.taaSharpenStrength,
      .sharpenConfidenceThreshold = debug.taaSharpenConfidenceThreshold,
      .depthDiscontinuityThreshold = debug.taaDepthDiscontinuityThreshold,
      .velocityRejectionThreshold = debug.taaVelocityRejectionThreshold,
      .velocityBlendScale = debug.taaVelocityBlendScale,
      .motionCurrentWeight =
          std::max(debug.taaMotionCurrentWeight, debug.taaCurrentFrameWeight),
      .disocclusionCurrentWeight = std::max(debug.taaDisocclusionCurrentWeight,
                                            debug.taaCurrentFrameWeight),
      .clampCurrentWeight =
          std::max(debug.taaClampCurrentWeight, debug.taaCurrentFrameWeight),
      .clampBlendAttenuation = debug.taaClampBlendAttenuation,
      .varianceGamma = debug.taaVarianceGamma,
      .hdrWeightStrength = debug.taaHdrWeightStrength,
      .reactiveCurrentWeight = debug.taaReactiveCurrentWeight,
      .reactiveStrength = debug.taaReactiveStrength,
      .velocityDilationDepthThreshold = debug.taaVelocityDilationDepthThreshold,
      .clampMode = debug.taaClampMode,
      .hdrWeightingMode = debug.taaHdrWeightingMode,
      .velocityDilationMode = debug.taaVelocityDilationMode,
      .historyFilterMode = debug.taaHistoryFilterMode,
      .sharpenEnabled = debug.taaSharpenEnabled};
}
} // namespace

namespace {
[[nodiscard]] bool
temporalInputPlacementEnabled(const FrameBuildContext &ctx,
                              TemporalInputPlacement placement) {
  const PresentationAAPlan plan = presentationAAPlanForFrame(ctx.frame);
  return placement == TemporalInputPlacement::EarlyGtao
             ? plan.gtaoTemporal
             : usesTemporalColorReconstruction(plan);
}
} // namespace

bool TemporalAAClearPass::isEnabled(const FrameBuildContext &ctx) const {
  return temporalInputPlacementEnabled(ctx, placement_);
}

Result<bool, std::string> TemporalAAClearPass::prepare(FrameBuildContext &ctx) {
  (void)ctx;
  return Result<bool, std::string>::makeResult(false);
}

Result<bool, std::string> TemporalAAClearPass::build(FrameBuildContext &ctx) {
  const bool motion = input_ == TemporalInput::MotionVectors;
  TextureHandle texture =
      motion ? ctx.shared.motionVectorTexture : ctx.shared.reactiveMaskTexture;
  auto &graphTexture = motion ? ctx.shared.motionVectorGraphTexture
                              : ctx.shared.reactiveMaskGraphTexture;
  if (nuri::isValid(graphTexture) || !nuri::isValid(texture)) {
    return Result<bool, std::string>::makeResult(false);
  }
  graphTexture = ctx.graph
                     .importTexture(texture, motion ? "taa_motion_vectors"
                                                    : "taa_reactive_mask")
                     .value();
  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor =
                        motion ? kFrameCompositionMotionVectorClearValue
                               : kFrameCompositionReactiveMaskClearValue};
  passDesc.colorTexture = graphTexture;
  passDesc.gpuTimingScope =
      motion ? GpuTimingScope::Velocity : GpuTimingScope::ReactiveMask;
  passDesc.debugLabel = motion ? "Temporal AA Motion Vector Clear"
                               : "Temporal AA Reactive Mask Clear";
  passDesc.debugColor = motion ? 0xff44aaff : 0xff33cc88;
  (void)ctx.graph.addGraphicsPass(passDesc).value();
  auto &metrics = ctx.frame.metrics.antiAliasing;
  if (motion) {
    ctx.shared.historyWriteRequirements |=
        FrameTextureRequirementFlags::MotionVectors;
    metrics.motionVectorGraphPublished = true;
    metrics.motionVectorClearPassCount = 1u;
    metrics.motionVectorClearBytes = metrics.motionVectorTextureBytes;
  } else {
    metrics.reactiveMaskGraphPublished = true;
    metrics.reactiveMaskPassBandwidthEstimateBytes =
        metrics.reactiveMaskTextureBytes;
  }
  return Result<bool, std::string>::makeResult(true);
}

namespace {
struct BackgroundMotionPushConstants {
  glm::mat4 previousFromCurrentJitteredRotationClip{1.0f};
  uint32_t currentJitterUvXBits = 0u;
  uint32_t currentJitterUvYBits = 0u;
  uint32_t historyValid = 0u;
};
static_assert(sizeof(BackgroundMotionPushConstants) <= 128u);
static_assert(offsetof(BackgroundMotionPushConstants, currentJitterUvXBits) ==
              sizeof(glm::mat4));
} // namespace

TemporalAABackgroundMotionPass::TemporalAABackgroundMotionPass(
    GPUDevice &gpu, RuntimeCompositeConfig config,
    TemporalInputPlacement placement)
    : gpu_(gpu), placement_(placement) {
  const std::filesystem::path basePath = resolveShaderBasePath(config);
  vertexPath_ = basePath / "taa_background_motion.vert";
  fragmentPath_ = basePath / "taa_background_motion.frag";
}

TemporalAABackgroundMotionPass::~TemporalAABackgroundMotionPass() {
  destroyTemporalPipeline(gpu_, pipeline_);
}

bool TemporalAABackgroundMotionPass::isEnabled(
    const FrameBuildContext &ctx) const {
  const PresentationAAPlan plan = presentationAAPlanForFrame(ctx.frame);
  return temporalInputPlacementEnabled(ctx, placement_) &&
         plan.reconstruction == ColorReconstruction::ReferenceTAA &&
         plan.needsMotion;
}

Result<bool, std::string>
TemporalAABackgroundMotionPass::prepare(FrameBuildContext &ctx) {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  RenderPipelineDesc pipelineDesc =
      fullscreenPipelineDesc(kFrameCompositionMotionVectorFormat, {}, {});
  pipelineDesc.depthFormat = kFrameCompositionDepthFormat;
  pipelineDesc.rasterState = makeRasterPipelineState(DepthState{
      .compareOp = CompareOp::Equal,
      .isDepthWriteEnabled = false,
  });
  auto result =
      createTemporalPipeline(gpu_, pipeline_, vertexPath_, fragmentPath_,
                             pipelineDesc, "taa_background_motion");
  if (result.hasError()) {
    return result;
  }
  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TemporalAABackgroundMotionPass::build(FrameBuildContext &ctx) {
  if (nuri::isValid(ctx.shared.motionClassGraphTexture)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(ctx.shared.sceneDepthGraphTexture)) {
    ctx.shared.sceneDepthGraphTexture =
        ctx.graph.importTexture(ctx.shared.sceneDepthTexture, "scene_depth")
            .value();
  }
  const TextureDimensions dimensions =
      gpu_.getTextureDimensions(ctx.shared.motionVectorTexture);
  const float inverseWidth =
      1.0f / static_cast<float>(std::max(dimensions.width, 1u));
  const float inverseHeight =
      1.0f / static_cast<float>(std::max(dimensions.height, 1u));
  const glm::vec2 currentJitterUv{
      ctx.frame.camera.jitterPixelOffset.x * inverseWidth,
      -ctx.frame.camera.jitterPixelOffset.y * inverseHeight,
  };
  const BackgroundMotionPushConstants constants{
      .previousFromCurrentJitteredRotationClip =
          makeBackgroundRotationReprojection(ctx.frame.camera),
      .currentJitterUvXBits = floatBits(currentJitterUv.x),
      .currentJitterUvYBits = floatBits(currentJitterUv.y),
      .historyValid = ctx.frame.camera.historyValid ? 1u : 0u,
  };
  DrawItem draw = makeFullscreenDraw(
      pipeline_.pipeline,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&constants), sizeof(constants)),
      "TaaBackgroundMotion", kDrawDebugColor);
  draw.useDepthState = true;
  draw.depthState = {
      .compareOp = CompareOp::Equal,
      .isDepthWriteEnabled = false,
  };
  RenderGraphGraphicsPassDesc pass{};
  pass.color = {.loadOp = LoadOp::Load, .storeOp = StoreOp::Store};
  pass.colorTexture = ctx.shared.motionVectorGraphTexture;
  pass.depth = {.loadOp = LoadOp::Load,
                .storeOp = StoreOp::Store,
                .clearDepth = 1.0f,
                .clearStencil = 0u};
  pass.depthTexture = ctx.shared.sceneDepthGraphTexture;
  pass.draws = std::span<const DrawItem>(&draw, 1u);
  pass.debugLabel = "TAA Background Motion Pass";
  pass.debugColor = 0xff4488ff;
  (void)ctx.graph.addGraphicsPass(pass).value();
  ctx.shared.historyWriteRequirements |=
      FrameTextureRequirementFlags::MotionVectors;
  return Result<bool, std::string>::makeResult(true);
}

namespace {
struct MotionClassPushConstants {
  uint32_t depthTexId = 0u;
  uint32_t motionTexId = 0u;
  uint32_t reactiveTexId = 0u;
  uint32_t pointSamplerId = 0u;
  uint32_t forceInvalidGeometry = 0u;
  uint32_t provenStaticCameraOnly = 0u;
};
static_assert(sizeof(MotionClassPushConstants) == 24u);
} // namespace

TemporalAAMotionClassPass::TemporalAAMotionClassPass(
    GPUDevice &gpu, RuntimeCompositeConfig config,
    TemporalInputPlacement placement)
    : gpu_(gpu), placement_(placement) {
  const std::filesystem::path basePath = resolveShaderBasePath(config);
  vertexPath_ = config.fullscreenVertex.empty()
                    ? basePath / "fullscreen_copy.vert"
                    : config.fullscreenVertex;
  fragmentPath_ = basePath / "taa_motion_class.frag";
}

TemporalAAMotionClassPass::~TemporalAAMotionClassPass() {
  destroyTemporalPipeline(gpu_, pipeline_);
}

bool TemporalAAMotionClassPass::isEnabled(const FrameBuildContext &ctx) const {
  return temporalInputPlacementEnabled(ctx, placement_) &&
         presentationAAPlanForFrame(ctx.frame).needsMotionClass;
}

Result<bool, std::string>
TemporalAAMotionClassPass::prepare(FrameBuildContext &ctx) {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto result = createTemporalPipeline(
      gpu_, pipeline_, vertexPath_, fragmentPath_,
      fullscreenPipelineDesc(kFrameCompositionMotionClassFormat, {}, {}),
      "taa_motion_class");
  if (result.hasError()) {
    return result;
  }
  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TemporalAAMotionClassPass::build(FrameBuildContext &ctx) {
  if (nuri::isValid(ctx.shared.motionClassGraphTexture)) {
    return Result<bool, std::string>::makeResult(false);
  }
  const uint32_t depthTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.sceneDepthTexture);
  const uint32_t motionTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.motionVectorTexture);
  const uint32_t reactiveTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.reactiveMaskTexture);
  const uint32_t pointSamplerId = gpu_.getDefaultSamplerBindlessIndex();
  ctx.shared.motionClassGraphTexture =
      ctx.graph.importTexture(ctx.shared.motionClassTexture, "taa_motion_class")
          .value();
  const bool forceInvalidGeometry =
      !ctx.frame.metrics.antiAliasing.opaqueVelocityGenerated ||
      ctx.frame.metrics.antiAliasing.velocityTessellatedSkippedDrawCount > 0u;
  const MotionClassPushConstants constants{
      .depthTexId = depthTexId,
      .motionTexId = motionTexId,
      .reactiveTexId = reactiveTexId,
      .pointSamplerId = pointSamplerId,
      .forceInvalidGeometry = forceInvalidGeometry ? 1u : 0u,
      .provenStaticCameraOnly =
          ctx.frame.metrics.antiAliasing.motionVectorDepthReprojectionGenerated
              ? 1u
              : 0u,
  };
  const DrawItem draw = makeFullscreenDraw(
      pipeline_.pipeline,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&constants), sizeof(constants)),
      "TaaMotionClass", kDrawDebugColor);
  const std::array<TextureHandle, 3> reads{
      ctx.shared.sceneDepthTexture,
      ctx.shared.motionVectorTexture,
      ctx.shared.reactiveMaskTexture,
  };
  RenderGraphGraphicsPassDesc pass{};
  pass.color = {.loadOp = LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearColor = kFrameCompositionMotionClassClearValue};
  pass.colorTexture = ctx.shared.motionClassGraphTexture;
  pass.dependencyTextures = reads;
  pass.draws = std::span<const DrawItem>(&draw, 1u);
  pass.debugLabel = "TAA Motion Class Pass";
  pass.debugColor = 0xff55aaff;
  (void)ctx.graph.addGraphicsPass(pass).value();
  AntiAliasingFrameMetrics &metrics = ctx.frame.metrics.antiAliasing;
  ctx.shared.historyWriteRequirements |=
      FrameTextureRequirementFlags::MotionClass;
  if (isRenderCaptureRequested(ctx.frame, "motion_class")) {
    ctx.frame.captureRegistry.publish(RenderCapturePoint{
        .name = "motion_class",
        .version = 1u,
        .texture = ctx.shared.motionClassTexture,
        .format = gpu_.getTextureFormat(ctx.shared.motionClassTexture),
        .dimensions = gpu_.getTextureDimensions(ctx.shared.motionClassTexture),
        .frameIndex = ctx.frame.frameIndex,
        .mip = 0u,
        .layer = 0u,
        .kind = RenderCaptureValueKind::Mask,
        .lifetime = RenderCaptureLifetimeClass::FrameSharedRingTexture,
        .colorSpace = "motion_class_u8",
        .defaultCompareProfile = "mask",
        .producerPassLabel = "TAA Motion Class Pass",
        .debugLabel = "motion_class",
    });
  }
  return Result<bool, std::string>::makeResult(true);
}

TemporalAAResolvePass::TemporalAAResolvePass(GPUDevice &gpu,
                                             RuntimeCompositeConfig config)
    : gpu_(gpu) {
  const std::filesystem::path basePath = resolveShaderBasePath(config);
  vertexPath_ = config.fullscreenVertex.empty()
                    ? basePath / "fullscreen_copy.vert"
                    : config.fullscreenVertex;
  fragmentPath_ = basePath / "taa_resolve.frag";
}

TemporalAAResolvePass::~TemporalAAResolvePass() {
  if (nuri::isValid(linearClampSampler_)) {
    gpu_.destroySampler(linearClampSampler_);
  }
  destroyTemporalPipeline(gpu_, pipeline_);
}

bool TemporalAAResolvePass::isEnabled(const FrameBuildContext &ctx) const {
  return isLegacyTAAEnabled(ctx.frame);
}

Result<bool, std::string>
TemporalAAResolvePass::prepare(FrameBuildContext &ctx) {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto pipeline = createTemporalPipeline(
      gpu_, pipeline_, vertexPath_, fragmentPath_,
      fullscreenPipelineDesc(kFrameCompositionSceneColorFormat, {}, {}),
      "taa_resolve");
  if (pipeline.hasError()) {
    return pipeline;
  }
  const SamplerDesc linearClampDesc{
      .minFilter = SamplerFilter::Linear,
      .magFilter = SamplerFilter::Linear,
      .mipMode = SamplerMipMode::Disabled,
      .wrapU = SamplerWrapMode::Clamp,
      .wrapV = SamplerWrapMode::Clamp,
      .wrapW = SamplerWrapMode::Clamp,
      .mipLodMax = 0.0f,
  };
  auto sampler = gpu_.createSampler(linearClampDesc, "taa_linear_clamp");
  if (sampler.hasError()) {
    return Result<bool, std::string>::makeError(sampler.error());
  }
  linearClampSampler_ = sampler.value();
  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TemporalAAResolvePass::build(FrameBuildContext &ctx) {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  const RenderSettings::AntiAliasingDebugSettings aaDebug =
      effectiveTemporalAADebugSettings(settings.antiAliasing);
  const AntiAliasingDebugView debugView =
      sanitizeAntiAliasingDebugView(aaDebug.view);
  const TAAEvaluationDebugDesc evaluationDebug =
      taaEvaluationDebugDesc(debugView);
  const uint32_t sceneTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.sceneColorTexture);
  const uint32_t historyReadTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.historyColorReadTexture);
  const uint32_t historyWriteTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.historyColorWriteTexture);
  const uint32_t depthTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.sceneDepthTexture);
  const uint32_t velocityTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.motionVectorTexture);
  const uint32_t reactiveMaskTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.reactiveMaskTexture);
  const TAAResolveTuning tuning = sanitizedResolveTuning(settings.antiAliasing);
  const bool temporalHistoryValid =
      ctx.frame.camera.historyValid && ctx.shared.historyColorReadValid;
  const bool usePreviousDepth =
      temporalHistoryValid &&
      nuri::isValid(ctx.shared.previousSceneDepthTexture);
  const uint32_t previousDepthTexId =
      usePreviousDepth
          ? gpu_.getTextureBindlessIndex(ctx.shared.previousSceneDepthTexture)
          : depthTexId;
  const bool usePreviousVelocity =
      temporalHistoryValid &&
      nuri::isValid(ctx.shared.previousMotionVectorTexture);
  const uint32_t previousVelocityTexId =
      usePreviousVelocity
          ? gpu_.getTextureBindlessIndex(ctx.shared.previousMotionVectorTexture)
          : velocityTexId;
  const uint32_t linearSamplerId =
      gpu_.getSamplerBindlessIndex(linearClampSampler_);
  const uint32_t pointSamplerId = gpu_.getDefaultSamplerBindlessIndex();
  uint32_t resolveFlags =
      temporalHistoryValid ? kTaaResolveFlagHistoryValid : 0u;
  if (usePreviousVelocity) {
    resolveFlags |= kTaaResolveFlagPreviousVelocityValid;
  }
  if (usePreviousDepth) {
    resolveFlags |= kTaaResolveFlagPreviousDepthValid;
  }
  if (temporalHistoryValid) {
    resolveFlags |= kTaaResolvePhase5Flags;
  }
  const bool sanitizeStaticFrameVelocity =
      temporalHistoryValid &&
      staticFrameVelocitySanitizationEligible(ctx.frame.metrics.antiAliasing);
  if (sanitizeStaticFrameVelocity) {
    resolveFlags |= kTaaResolveFlagStaticFrame;
  }
  const bool useReactiveMask =
      temporalHistoryValid && nuri::isValid(ctx.shared.reactiveMaskTexture) &&
      tuning.reactiveStrength > 0.0f &&
      tuning.reactiveCurrentWeight > tuning.currentWeight;
  if (useReactiveMask) {
    resolveFlags |= kTaaResolveFlagReactiveMask;
  }
  const bool useVelocityDilation =
      temporalHistoryValid &&
      tuning.velocityDilationMode != TemporalAAVelocityDilationMode::None;
  if (useVelocityDilation) {
    resolveFlags |= kTaaResolveFlagVelocityDilation;
  }
  const TextureDimensions sceneDimensions =
      gpu_.getTextureDimensions(ctx.shared.sceneColorTexture);
  const float inverseWidth =
      1.0f / static_cast<float>(std::max(sceneDimensions.width, 1u));
  const float inverseHeight =
      1.0f / static_cast<float>(std::max(sceneDimensions.height, 1u));
  const glm::vec2 currentJitterUv{
      ctx.frame.camera.jitterPixelOffset.x * inverseWidth,
      -ctx.frame.camera.jitterPixelOffset.y * inverseHeight,
  };
  const glm::vec2 previousJitterUv{
      ctx.frame.camera.previousJitterPixelOffset.x * inverseWidth,
      -ctx.frame.camera.previousJitterPixelOffset.y * inverseHeight,
  };
  const glm::vec2 previousRawJitterDeltaUv = previousJitterUv - currentJitterUv;
  const auto sceneGraphTexture =
      ctx.graph
          .importTexture(ctx.shared.sceneColorTexture,
                         "taa_scene_color_current")
          .value();
  const auto historyWriteGraphTexture =
      ctx.graph
          .importTexture(ctx.shared.historyColorWriteTexture,
                         "taa_history_color_write")
          .value();
  if (usePreviousDepth) {
    ctx.shared.previousSceneDepthGraphTexture =
        ctx.graph
            .importTexture(ctx.shared.previousSceneDepthTexture,
                           "taa_previous_scene_depth")
            .value();
  }
  if (usePreviousVelocity) {
    ctx.shared.previousMotionVectorGraphTexture =
        ctx.graph
            .importTexture(ctx.shared.previousMotionVectorTexture,
                           "taa_previous_motion_vectors")
            .value();
  }
  std::array<TextureHandle, 7> resolveReads{};
  size_t resolveReadCount = 0u;
  resolveReads[resolveReadCount++] = ctx.shared.sceneColorTexture;
  resolveReads[resolveReadCount++] = ctx.shared.sceneDepthTexture;
  if (temporalHistoryValid) {
    resolveReads[resolveReadCount++] = ctx.shared.historyColorReadTexture;
    resolveReads[resolveReadCount++] = ctx.shared.motionVectorTexture;
    if (usePreviousDepth) {
      resolveReads[resolveReadCount++] = ctx.shared.previousSceneDepthTexture;
    }
    if (useReactiveMask) {
      resolveReads[resolveReadCount++] = ctx.shared.reactiveMaskTexture;
    }
    if (usePreviousVelocity) {
      resolveReads[resolveReadCount++] = ctx.shared.previousMotionVectorTexture;
    }
  }
  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  const GpuTimingReport &timingReport = ctx.frame.gpuTiming;
  if (hasGpuTimingScope(timingReport, GpuTimingScope::TemporalAAResolve)) {
    aaMetrics.taaResolveGpuTimeMs = timingReport.temporalAAResolveTimeMs;
    aaMetrics.taaResolveGpuTimingSourceFrameIndex =
        timingReport.temporalAAResolveSourceFrameIndex;
    aaMetrics.taaResolveGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::TemporalAADebug)) {
    aaMetrics.taaDebugGpuTimeMs = timingReport.temporalAADebugTimeMs;
    aaMetrics.taaDebugGpuTimingSourceFrameIndex =
        timingReport.temporalAADebugSourceFrameIndex;
    aaMetrics.taaDebugGpuTimingAvailable = 1u;
  }
  aaMetrics.taaResolveWidth = sceneDimensions.width;
  aaMetrics.taaResolveHeight = sceneDimensions.height;
  const float jitterScale = aaDebug.taaJitterScale;
  aaMetrics.taaJitterScale = jitterScale;
  aaMetrics.taaQualityPreset =
      sanitizeTemporalAAQualityPreset(settings.antiAliasing.qualityPreset);
  aaMetrics.taaCurrentFrameWeight = tuning.currentWeight;
  aaMetrics.taaHistoryFrameWeight = 1.0f - tuning.currentWeight;
  aaMetrics.taaSharpenEnabled = tuning.sharpenEnabled;
  aaMetrics.taaSharpenStrength = tuning.sharpenStrength;
  aaMetrics.taaSharpenConfidenceThreshold = tuning.sharpenConfidenceThreshold;
  aaMetrics.taaHistoryValidPercent =
      ctx.frame.camera.historyValid ? 100.0f : 0.0f;
  aaMetrics.taaOutOfBoundsFallbackEnabled = true;
  aaMetrics.taaBilinearHistorySampling =
      tuning.historyFilterMode == TemporalAAHistoryFilterMode::Bilinear;
  aaMetrics.previousSceneDepthGraphPublished = usePreviousDepth;
  aaMetrics.previousMotionVectorGraphPublished = usePreviousVelocity;
  aaMetrics.taaDepthDiscontinuityThreshold = tuning.depthDiscontinuityThreshold;
  aaMetrics.taaVelocityRejectionThreshold = tuning.velocityRejectionThreshold;
  aaMetrics.taaVelocityBlendScale = tuning.velocityBlendScale;
  aaMetrics.taaMotionCurrentWeight = tuning.motionCurrentWeight;
  aaMetrics.taaDisocclusionCurrentWeight = tuning.disocclusionCurrentWeight;
  aaMetrics.taaClampCurrentWeight = tuning.clampCurrentWeight;
  aaMetrics.taaClampBlendAttenuation = tuning.clampBlendAttenuation;
  aaMetrics.taaVarianceGamma = tuning.varianceGamma;
  aaMetrics.taaHdrWeightStrength = tuning.hdrWeightStrength;
  aaMetrics.taaReactiveCurrentWeight = tuning.reactiveCurrentWeight;
  aaMetrics.taaReactiveStrength = tuning.reactiveStrength;
  aaMetrics.taaVelocityDilationDepthThreshold =
      tuning.velocityDilationDepthThreshold;
  aaMetrics.taaClampMode = tuning.clampMode;
  aaMetrics.taaHdrWeightingMode = tuning.hdrWeightingMode;
  aaMetrics.taaVelocityDilationMode = tuning.velocityDilationMode;
  aaMetrics.taaHistoryFilterMode = tuning.historyFilterMode;
  aaMetrics.taaDepthRejectionEnabled = temporalHistoryValid;
  aaMetrics.taaPreviousDepthRejectionEnabled = usePreviousDepth;
  aaMetrics.taaVelocityRejectionEnabled = usePreviousVelocity;
  aaMetrics.taaPreviousVelocityDisocclusionEnabled = usePreviousVelocity;
  aaMetrics.taaNeighborhoodClampEnabled = temporalHistoryValid;
  aaMetrics.taaAdaptiveBlendEnabled = temporalHistoryValid;
  aaMetrics.taaStaticFrameVelocitySanitizationEnabled =
      sanitizeStaticFrameVelocity;
  aaMetrics.taaClampBlendAttenuationEnabled = temporalHistoryValid;
  aaMetrics.taaNeighborhoodFallbackEnabled = temporalHistoryValid;
  aaMetrics.taaHdrWeightingEnabled =
      tuning.hdrWeightingMode != TemporalAAHdrWeightingMode::None &&
      tuning.hdrWeightStrength > 0.0f;
  aaMetrics.taaReactiveMaskEnabled = useReactiveMask;
  aaMetrics.taaDisocclusionRejectionEstimate =
      std::max(aaMetrics.velocityMissingPreviousRatio,
               aaMetrics.velocityEdgeDiscontinuityEstimate);
  aaMetrics.taaVelocityDilationAffectedEstimate =
      useVelocityDilation ? aaMetrics.velocityEdgeDiscontinuityEstimate : 0.0f;
  const TAAResolvePushConstants resolveConstants{
      .currentTexId = sceneTexId,
      .historyTexId = historyReadTexId,
      .depthTexId = depthTexId,
      .previousDepthTexId = previousDepthTexId,
      .velocityTexId = velocityTexId,
      .previousVelocityTexId = previousVelocityTexId,
      .reactiveMaskTexId = reactiveMaskTexId,
      .linearSamplerId = linearSamplerId,
      .pointSamplerId = pointSamplerId,
      .flags = resolveFlags,
      .mode = kTaaResolveModeResolve,
      .currentWeightBits = floatBits(tuning.currentWeight),
      .inverseWidthBits = floatBits(inverseWidth),
      .inverseHeightBits = floatBits(inverseHeight),
      .depthThresholdBits = floatBits(tuning.depthDiscontinuityThreshold),
      .velocityThresholdBits = floatBits(tuning.velocityRejectionThreshold),
      .velocityBlendScaleBits = floatBits(tuning.velocityBlendScale),
      .disocclusionWeightBits = floatBits(tuning.disocclusionCurrentWeight),
      .clampAttenuationBits = floatBits(tuning.clampBlendAttenuation),
      .varianceGammaBits = floatBits(tuning.varianceGamma),
      .hdrWeightStrengthBits = floatBits(tuning.hdrWeightStrength),
      .reactiveCurrentWeightBits = floatBits(tuning.reactiveCurrentWeight),
      .reactiveStrengthBits = floatBits(tuning.reactiveStrength),
      .velocityDilationDepthThresholdBits =
          floatBits(tuning.velocityDilationDepthThreshold),
      .clampMode = static_cast<uint32_t>(tuning.clampMode),
      .hdrWeightingMode = static_cast<uint32_t>(tuning.hdrWeightingMode),
      .velocityDilationMode =
          static_cast<uint32_t>(tuning.velocityDilationMode),
      .motionCurrentWeightBits = floatBits(tuning.motionCurrentWeight),
      .clampCurrentWeightBits = floatBits(tuning.clampCurrentWeight),
      .historyFilterMode =
          tuning.historyFilterMode == TemporalAAHistoryFilterMode::Bilinear
              ? kTaaHistoryFilterModeBilinear
              : kTaaHistoryFilterModeCatmullRom,
      .previousRawJitterDeltaUvXBits = floatBits(previousRawJitterDeltaUv.x),
      .previousRawJitterDeltaUvYBits = floatBits(previousRawJitterDeltaUv.y),
  };
  const DrawItem resolveDraw = makeFullscreenDraw(
      pipeline_.pipeline,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&resolveConstants),
          sizeof(resolveConstants)),
      "TaaResolve", kDrawDebugColor);
  RenderGraphGraphicsPassDesc resolvePass{};
  resolvePass.color = {.loadOp = LoadOp::Clear,
                       .storeOp = StoreOp::Store,
                       .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  resolvePass.colorTexture = historyWriteGraphTexture;
  resolvePass.dependencyTextures =
      std::span<const TextureHandle>(resolveReads.data(), resolveReadCount);
  resolvePass.draws = std::span<const DrawItem>(&resolveDraw, 1u);
  resolvePass.gpuTimingScope = GpuTimingScope::TemporalAAResolve;
  resolvePass.debugLabel = "TAA Resolve Pass";
  resolvePass.debugColor = kTaaResolvePassDebugColor;
  (void)ctx.graph.addGraphicsPass(resolvePass).value();
  ctx.shared.historyWriteRequirements |=
      FrameTextureRequirementFlags::HistoryColor;
  ++aaMetrics.taaResolvePassCount;
  if (!ctx.frame.camera.historyValid) {
    ++aaMetrics.taaCurrentFallbackFrameCount;
  }
  aaMetrics.taaHistoryBandwidthEstimateBytes +=
      textureStorageBytes(gpu_, ctx.shared.sceneColorTexture) +
      textureStorageBytes(gpu_, ctx.shared.historyColorWriteTexture) +
      textureStorageBytes(gpu_, ctx.shared.sceneDepthTexture) +
      (temporalHistoryValid
           ? textureStorageBytes(gpu_, ctx.shared.historyColorReadTexture) +
                 textureStorageBytes(gpu_, ctx.shared.motionVectorTexture) +
                 (usePreviousDepth
                      ? textureStorageBytes(
                            gpu_, ctx.shared.previousSceneDepthTexture)
                      : 0u) +
                 (useReactiveMask ? textureStorageBytes(
                                        gpu_, ctx.shared.reactiveMaskTexture)
                                  : 0u) +
                 (usePreviousVelocity
                      ? textureStorageBytes(
                            gpu_, ctx.shared.previousMotionVectorTexture)
                      : 0u)
           : 0u);
  TextureHandle displaySource = ctx.shared.historyColorWriteTexture;
  uint32_t displaySourceTexId = historyWriteTexId;
  uint32_t displayMode = kTaaResolveModeCopyHistoryToScene;
  std::string_view displayLabel = "TAA Copy Back Pass";
  uint32_t displayColor = kTaaCopyBackPassDebugColor;
  bool debugDisplay = false;
  if (debugView == AntiAliasingDebugView::TAACurrentColor) {
    ctx.shared.sceneColorGraphTexture = sceneGraphTexture;
    aaMetrics.taaResolvedSceneColorPublished = false;
    aaMetrics.taaDebugViewRendered = true;
    return Result<bool, std::string>::makeResult(true);
  }
  if (debugView == AntiAliasingDebugView::TAAPreviousHistory) {
    displaySource = ctx.shared.historyColorReadTexture;
    displaySourceTexId = historyReadTexId;
    displayMode = kTaaResolveModePreviousHistory;
    displayLabel = "TAA Previous History Debug Pass";
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  } else if (isVelocityDebugView(debugView)) {
    displaySource = ctx.shared.motionVectorTexture;
    displaySourceTexId = velocityTexId;
    displayMode = debugView == AntiAliasingDebugView::MotionVectors
                      ? kTaaResolveModeVelocityMotionVectors
                      : kTaaResolveModeVelocityMagnitude;
    displayLabel = debugView == AntiAliasingDebugView::MotionVectors
                       ? "TAA Motion Vector Debug Pass"
                       : "TAA Velocity Debug Pass";
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  } else if (debugView == AntiAliasingDebugView::TAAPreviousVelocity) {
    displaySource = usePreviousVelocity ? ctx.shared.previousMotionVectorTexture
                                        : ctx.shared.motionVectorTexture;
    displaySourceTexId = previousVelocityTexId;
    displayMode = kTaaResolveModePreviousVelocity;
    displayLabel = "TAA Previous Velocity Debug Pass";
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  } else if (evaluationDebug) {
    displaySource = ctx.shared.sceneColorTexture;
    displaySourceTexId = sceneTexId;
    displayMode = evaluationDebug.mode;
    displayLabel = evaluationDebug.label;
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  } else if (debugView == AntiAliasingDebugView::TAAReactiveMask) {
    displaySource = ctx.shared.reactiveMaskTexture;
    displaySourceTexId = reactiveMaskTexId;
    displayMode = kTaaResolveModeReactiveMask;
    displayLabel = "TAA Reactive Mask Debug Pass";
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  }
  const bool displaySamplesReactiveMask =
      debugView == AntiAliasingDebugView::TAAReactiveMask;
  const bool displaySamplesTemporalEvaluation =
      static_cast<bool>(evaluationDebug);
  TAAResolvePushConstants copyConstants = resolveConstants;
  copyConstants.currentTexId =
      displaySamplesTemporalEvaluation ? sceneTexId : displaySourceTexId;
  copyConstants.mode = displayMode;
  const bool sharpenCopyBack =
      !debugDisplay && displayMode == kTaaResolveModeCopyHistoryToScene &&
      temporalHistoryValid && tuning.sharpenEnabled &&
      tuning.sharpenStrength > 0.0f;
  if (sharpenCopyBack) {
    copyConstants.flags |= kTaaResolveFlagSharpen;
    copyConstants.velocityThresholdBits = floatBits(tuning.sharpenStrength);
    copyConstants.velocityBlendScaleBits =
        floatBits(tuning.sharpenConfidenceThreshold);
  }
  const DrawItem copyDraw = makeFullscreenDraw(
      pipeline_.pipeline,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&copyConstants),
          sizeof(copyConstants)),
      debugDisplay ? "TaaResolveDebug" : "TaaCopyBack", kDrawDebugColor);
  std::array<TextureHandle, 8> copyReads{};
  size_t copyReadCount = 0u;
  if (displaySamplesReactiveMask) {
    copyReads[copyReadCount++] = ctx.shared.reactiveMaskTexture;
  } else if (displaySamplesTemporalEvaluation) {
    copyReads[copyReadCount++] = ctx.shared.sceneColorTexture;
    copyReads[copyReadCount++] = ctx.shared.sceneDepthTexture;
    if (temporalHistoryValid) {
      copyReads[copyReadCount++] = ctx.shared.historyColorReadTexture;
      copyReads[copyReadCount++] = ctx.shared.motionVectorTexture;
      if (usePreviousDepth) {
        copyReads[copyReadCount++] = ctx.shared.previousSceneDepthTexture;
      }
      if (useReactiveMask) {
        copyReads[copyReadCount++] = ctx.shared.reactiveMaskTexture;
      }
      if (usePreviousVelocity) {
        copyReads[copyReadCount++] = ctx.shared.previousMotionVectorTexture;
      }
    }
  } else {
    copyReads[copyReadCount++] = displaySource;
  }
  if (debugDisplay) {
    copyReads[copyReadCount++] = ctx.shared.historyColorWriteTexture;
  }
  if (displaySamplesTemporalEvaluation) {
    const uint32_t debugOutputTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.frameColorTexture);
    const auto debugOutput =
        ctx.graph
            .importTexture(ctx.shared.frameColorTexture, "taa_debug_output")
            .value();
    RenderGraphGraphicsPassDesc debugPass{};
    debugPass.color = {.loadOp = LoadOp::Clear,
                       .storeOp = StoreOp::Store,
                       .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    debugPass.colorTexture = debugOutput;
    debugPass.dependencyTextures =
        std::span<const TextureHandle>(copyReads.data(), copyReadCount);
    debugPass.draws = std::span<const DrawItem>(&copyDraw, 1u);
    debugPass.gpuTimingScope = GpuTimingScope::TemporalAADebug;
    debugPass.debugLabel = displayLabel;
    debugPass.debugColor = displayColor;
    (void)ctx.graph.addGraphicsPass(debugPass).value();
    TAAResolvePushConstants debugCopyConstants = copyConstants;
    debugCopyConstants.currentTexId = debugOutputTexId;
    debugCopyConstants.mode = kTaaResolveModeCopyCurrent;
    const DrawItem debugCopyDraw = makeFullscreenDraw(
        pipeline_.pipeline,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&debugCopyConstants),
            sizeof(debugCopyConstants)),
        "TaaDebugCopyBack", kDrawDebugColor);
    const std::array<TextureHandle, 1> debugCopyReads{
        ctx.shared.frameColorTexture};
    RenderGraphGraphicsPassDesc debugCopyPass{};
    debugCopyPass.color = {.loadOp = LoadOp::Clear,
                           .storeOp = StoreOp::Store,
                           .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    debugCopyPass.colorTexture = sceneGraphTexture;
    debugCopyPass.dependencyTextures = std::span<const TextureHandle>(
        debugCopyReads.data(), debugCopyReads.size());
    debugCopyPass.draws = std::span<const DrawItem>(&debugCopyDraw, 1u);
    debugCopyPass.gpuTimingScope = GpuTimingScope::TemporalAADebug;
    debugCopyPass.debugLabel = "TAA Debug Copy Back Pass";
    debugCopyPass.debugColor = kTaaCopyBackPassDebugColor;
    (void)ctx.graph.addGraphicsPass(debugCopyPass).value();
  } else {
    RenderGraphGraphicsPassDesc copyPass{};
    copyPass.color = {.loadOp = LoadOp::Clear,
                      .storeOp = StoreOp::Store,
                      .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    copyPass.colorTexture = sceneGraphTexture;
    copyPass.dependencyTextures =
        std::span<const TextureHandle>(copyReads.data(), copyReadCount);
    copyPass.draws = std::span<const DrawItem>(&copyDraw, 1u);
    copyPass.gpuTimingScope = debugDisplay ? GpuTimingScope::TemporalAADebug
                                           : GpuTimingScope::TemporalAACopyBack;
    copyPass.debugLabel = displayLabel;
    copyPass.debugColor = displayColor;
    (void)ctx.graph.addGraphicsPass(copyPass).value();
  }
  ctx.shared.sceneColorGraphTexture = sceneGraphTexture;
  aaMetrics.taaCopyBackPassCount += displaySamplesTemporalEvaluation ? 2u : 1u;
  aaMetrics.taaResolvedSceneColorPublished = !debugDisplay;
  aaMetrics.taaDebugViewRendered = debugDisplay;
  aaMetrics.taaSharpenActive = sharpenCopyBack;
  aaMetrics.taaReactiveMaskDebugViewRendered =
      debugView == AntiAliasingDebugView::TAAReactiveMask;
  aaMetrics.taaPreviousVelocityDebugViewRendered =
      debugView == AntiAliasingDebugView::TAAPreviousVelocity;
  if (evaluationDebug.renderedMetric) {
    aaMetrics.*evaluationDebug.renderedMetric = true;
  }
  if (isVelocityDebugView(debugView)) {
    aaMetrics.velocityDebugPassCount = 1u;
    aaMetrics.velocityDebugViewRendered = true;
    aaMetrics.velocityDebugBandwidthEstimateBytes =
        textureStorageBytes(gpu_, ctx.shared.motionVectorTexture) +
        textureStorageBytes(gpu_, ctx.shared.sceneColorTexture);
  }
  aaMetrics.taaHistoryBandwidthEstimateBytes +=
      textureStorageBytes(gpu_, displaySource) +
      textureStorageBytes(gpu_, ctx.shared.sceneColorTexture) +
      (displaySamplesTemporalEvaluation
           ? textureStorageBytes(gpu_, ctx.shared.frameColorTexture) * 2u +
                 textureStorageBytes(gpu_, ctx.shared.sceneDepthTexture)
           : 0u) +
      (displaySamplesTemporalEvaluation && temporalHistoryValid
           ? textureStorageBytes(gpu_, ctx.shared.historyColorReadTexture) +
                 textureStorageBytes(gpu_, ctx.shared.motionVectorTexture) +
                 (usePreviousDepth
                      ? textureStorageBytes(
                            gpu_, ctx.shared.previousSceneDepthTexture)
                      : 0u) +
                 (useReactiveMask ? textureStorageBytes(
                                        gpu_, ctx.shared.reactiveMaskTexture)
                                  : 0u) +
                 (usePreviousVelocity
                      ? textureStorageBytes(
                            gpu_, ctx.shared.previousMotionVectorTexture)
                      : 0u)
           : 0u);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TemporalAAClearPass::publishFrameData(FrameBuildContext &ctx) {
  const PresentationAAPlan plan = presentationAAPlanForFrame(ctx.frame);
  if (placement_ == TemporalInputPlacement::EarlyGtao) {
    if (!plan.gtaoTemporal) {
      return Result<bool, std::string>::makeResult(false);
    }
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::MotionVectors |
        FrameTextureRequirementFlags::ReactiveMask;
    return Result<bool, std::string>::makeResult(true);
  }
  if (plan.needsMotion) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::MotionVectors;
  }
  if (plan.needsReactiveMask) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::ReactiveMask;
  }
  if (plan.needsMotionClass) {
    ctx.shared.textureRequirements |= FrameTextureRequirementFlags::MotionClass;
  }
  if (isLegacyTAAEnabled(ctx.frame)) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::HistoryColor;
  }
  return Result<bool, std::string>::makeResult(true);
}

void registerTemporalInputStages(RenderPipeline &pipeline, GPUDevice &gpu,
                                 RuntimeCompositeConfig config) {
  pipeline.addStage(
      std::make_unique<TemporalAAClearPass>(TemporalInput::MotionVectors,
                                            TemporalInputPlacement::EarlyGtao),
      "TemporalInputFeature", "TemporalAAMotionVectorClearPass", false,
      PipelineComponentDesc{.publish = [](void *state, FrameBuildContext &ctx) {
        return static_cast<TemporalAAClearPass *>(state)->publishFrameData(ctx);
      }});
  pipeline.addStage(
      std::make_unique<TemporalAAClearPass>(TemporalInput::ReactiveMask,
                                            TemporalInputPlacement::EarlyGtao),
      "TemporalInputFeature", "TemporalAAReactiveMaskClearPass");
  pipeline.addStage(std::make_unique<TemporalAABackgroundMotionPass>(
                        gpu, config, TemporalInputPlacement::EarlyGtao),
                    "TemporalInputFeature", "TemporalAABackgroundMotionPass");
  pipeline.addStage(
      std::make_unique<TemporalAAMotionClassPass>(
          gpu, std::move(config), TemporalInputPlacement::EarlyGtao),
      "TemporalInputFeature", "TemporalAAMotionClassPass");
}

void registerTemporalAAStages(RenderPipeline &pipeline, GPUDevice &gpu,
                              RuntimeCompositeConfig config) {
  pipeline.addStage(
      std::make_unique<TemporalAAClearPass>(
          TemporalInput::MotionVectors,
          TemporalInputPlacement::ColorReconstruction),
      "TemporalAAFeature", "TemporalAAMotionVectorClearPass", false,
      PipelineComponentDesc{.publish = [](void *state, FrameBuildContext &ctx) {
        return static_cast<TemporalAAClearPass *>(state)->publishFrameData(ctx);
      }});
  pipeline.addStage(std::make_unique<TemporalAAClearPass>(
                        TemporalInput::ReactiveMask,
                        TemporalInputPlacement::ColorReconstruction),
                    "TemporalAAFeature", "TemporalAAReactiveMaskClearPass");
  pipeline.addStage(
      std::make_unique<TemporalAABackgroundMotionPass>(
          gpu, config, TemporalInputPlacement::ColorReconstruction),
      "TemporalAAFeature", "TemporalAABackgroundMotionPass");
  pipeline.addStage(
      std::make_unique<TemporalAAMotionClassPass>(
          gpu, config, TemporalInputPlacement::ColorReconstruction),
      "TemporalAAFeature", "TemporalAAMotionClassPass");
  pipeline.addStage(
      std::make_unique<TemporalAAResolvePass>(gpu, std::move(config)),
      "TemporalAAFeature", "TemporalAAResolvePass");
}

} // namespace nuri
