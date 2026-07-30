#include "nuri/gfx/pipeline/features/temporal_aa_feature.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/fullscreen.h"
#include "nuri/gfx/pipeline/owned_program_bundle.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include <algorithm>
#include <array>
#include <bit>
#include <filesystem>
namespace nuri {

enum class TemporalInputPlacement : uint8_t {
  EarlyGtao = 0u,
  ColorReconstruction = 1u,
};
enum class TemporalInput : uint8_t { MotionVectors, ReactiveMask };
class TemporalAAClearPass final {
public:
  TemporalAAClearPass(TemporalInput input, TemporalInputPlacement placement)
      : input_(input), placement_(placement) {}
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx);
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  TemporalInput input_ = TemporalInput::MotionVectors;
  TemporalInputPlacement placement_ =
      TemporalInputPlacement::ColorReconstruction;
};
class TemporalAABackgroundMotionPass final {
public:
  TemporalAABackgroundMotionPass(GPUDevice &gpu, RuntimeCompositeConfig config,
                                 TemporalInputPlacement placement);
  ~TemporalAABackgroundMotionPass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  GPUDevice &gpu_;
  OwnedProgramBundle pipeline_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
  TemporalInputPlacement placement_ =
      TemporalInputPlacement::ColorReconstruction;
};
class TemporalAAMotionClassPass final {
public:
  TemporalAAMotionClassPass(GPUDevice &gpu, RuntimeCompositeConfig config,
                            TemporalInputPlacement placement);
  ~TemporalAAMotionClassPass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  GPUDevice &gpu_;
  OwnedProgramBundle pipeline_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
  TemporalInputPlacement placement_ =
      TemporalInputPlacement::ColorReconstruction;
};
class TemporalAAResolvePass final {
public:
  TemporalAAResolvePass(GPUDevice &gpu, RuntimeCompositeConfig config);
  ~TemporalAAResolvePass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  GPUDevice &gpu_;
  OwnedProgramBundle pipeline_{};
  SamplerHandle linearClampSampler_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
};

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
             : frame.settings->antiAliasing.mode == AntiAliasingMode::TAA;
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
void destroyTemporalPipeline(OwnedProgramBundle &program) { program.reset(); }
Result<bool, std::string>
createTemporalPipeline(GPUDevice &gpu, OwnedProgramBundle &program,
                       const std::filesystem::path &vertexPath,
                       const std::filesystem::path &fragmentPath,
                       RenderPipelineDesc desc, std::string_view name) {
  const std::array shaderSpecs{
      ShaderSpec{name, vertexPath, ShaderStage::Vertex},
      ShaderSpec{name, fragmentPath, ShaderStage::Fragment}};
  auto shaders = program.compileShaders(gpu, shaderSpecs);
  if (shaders.hasError()) {
    return shaders;
  }
  desc.vertexShader = program.shader(0u);
  desc.fragmentShader = program.shader(1u);
  const GraphicsPipelineSpec pipelineSpec{name, desc};
  auto pipeline = program.replaceGraphicsPipelines(
      gpu, std::span<const GraphicsPipelineSpec>(&pipelineSpec, 1u));
  if (pipeline.hasError()) {
    program.reset();
  }
  return pipeline;
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
} // namespace

namespace {
[[nodiscard]] bool
temporalInputPlacementEnabled(const FrameBuildContext &ctx,
                              TemporalInputPlacement placement) {
  const PresentationAAPlan &plan = ctx.frame.presentationAA;
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
      motion ? ctx.shared[FrameTextureSlot::MotionVector].texture
             : ctx.shared[FrameTextureSlot::ReactiveMask].texture;
  auto &graphTexture = motion
                           ? ctx.shared[FrameTextureSlot::MotionVector].graph
                           : ctx.shared[FrameTextureSlot::ReactiveMask].graph;
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
  destroyTemporalPipeline(pipeline_);
}

bool TemporalAABackgroundMotionPass::isEnabled(
    const FrameBuildContext &ctx) const {
  const PresentationAAPlan &plan = ctx.frame.presentationAA;
  return temporalInputPlacementEnabled(ctx, placement_) &&
         plan.reconstruction == ColorReconstruction::ReferenceTAA &&
         plan.needsMotion;
}

Result<bool, std::string>
TemporalAABackgroundMotionPass::prepare(FrameBuildContext &) {
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
  if (nuri::isValid(ctx.shared[FrameTextureSlot::MotionClass].graph)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(ctx.shared[FrameTextureSlot::SceneDepth].graph)) {
    ctx.shared[FrameTextureSlot::SceneDepth].graph =
        ctx.graph
            .importTexture(ctx.shared[FrameTextureSlot::SceneDepth].texture,
                           "scene_depth")
            .value();
  }
  const TextureDimensions dimensions = gpu_.getTextureDimensions(
      ctx.shared[FrameTextureSlot::MotionVector].texture);
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
      pipeline_.graphicsPipeline(0u),
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
  pass.colorTexture = ctx.shared[FrameTextureSlot::MotionVector].graph;
  pass.depth = {.loadOp = LoadOp::Load,
                .storeOp = StoreOp::Store,
                .clearDepth = 1.0f,
                .clearStencil = 0u};
  pass.depthTexture = ctx.shared[FrameTextureSlot::SceneDepth].graph;
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
  destroyTemporalPipeline(pipeline_);
}

bool TemporalAAMotionClassPass::isEnabled(const FrameBuildContext &ctx) const {
  return temporalInputPlacementEnabled(ctx, placement_) &&
         ctx.frame.presentationAA.needsMotionClass;
}

Result<bool, std::string>
TemporalAAMotionClassPass::prepare(FrameBuildContext &) {
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
  if (nuri::isValid(ctx.shared[FrameTextureSlot::MotionClass].graph)) {
    return Result<bool, std::string>::makeResult(false);
  }
  const uint32_t depthTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::SceneDepth].texture);
  const uint32_t motionTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::MotionVector].texture);
  const uint32_t reactiveTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::ReactiveMask].texture);
  const uint32_t pointSamplerId = gpu_.getDefaultSamplerBindlessIndex();
  ctx.shared[FrameTextureSlot::MotionClass].graph =
      ctx.graph
          .importTexture(ctx.shared[FrameTextureSlot::MotionClass].texture,
                         "taa_motion_class")
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
      pipeline_.graphicsPipeline(0u),
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&constants), sizeof(constants)),
      "TaaMotionClass", kDrawDebugColor);
  const std::array<TextureHandle, 3> reads{
      ctx.shared[FrameTextureSlot::SceneDepth].texture,
      ctx.shared[FrameTextureSlot::MotionVector].texture,
      ctx.shared[FrameTextureSlot::ReactiveMask].texture,
  };
  RenderGraphGraphicsPassDesc pass{};
  pass.color = {.loadOp = LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearColor = kFrameCompositionMotionClassClearValue};
  pass.colorTexture = ctx.shared[FrameTextureSlot::MotionClass].graph;
  pass.draws = std::span<const DrawItem>(&draw, 1u);
  pass.debugLabel = "TAA Motion Class Pass";
  pass.debugColor = 0xff55aaff;
  const RenderGraphPassId motionClassPass =
      ctx.graph.addGraphicsPass(pass).value();
  (void)ctx.graph
      .addImportedTextureReads(motionClassPass, reads, pass.debugLabel)
      .value();
  ctx.shared.historyWriteRequirements |=
      FrameTextureRequirementFlags::MotionClass;
  if (isRenderCaptureRequested(ctx.frame, "motion_class")) {
    ctx.frame.captureRegistry.publish(RenderCapturePoint{
        .name = "motion_class",
        .version = 1u,
        .texture = ctx.shared[FrameTextureSlot::MotionClass].texture,
        .format = gpu_.getTextureFormat(
            ctx.shared[FrameTextureSlot::MotionClass].texture),
        .dimensions = gpu_.getTextureDimensions(
            ctx.shared[FrameTextureSlot::MotionClass].texture),
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
  destroyTemporalPipeline(pipeline_);
}

bool TemporalAAResolvePass::isEnabled(const FrameBuildContext &ctx) const {
  return isLegacyTAAEnabled(ctx.frame);
}

Result<bool, std::string> TemporalAAResolvePass::prepare(FrameBuildContext &) {
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
  const RenderSettings &settings = ctx.frame.settings.facts();
  const RenderSettings::AntiAliasingDebugSettings &aaDebug =
      settings.antiAliasing.debug;
  const AntiAliasingDebugView debugView = aaDebug.view;
  const TAAEvaluationDebugDesc evaluationDebug =
      taaEvaluationDebugDesc(debugView);
  const uint32_t sceneTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::SceneColor].texture);
  const uint32_t historyReadTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameHistoryTextureSlot::ColorRead].texture);
  const uint32_t historyWriteTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameHistoryTextureSlot::ColorWrite].texture);
  const uint32_t depthTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::SceneDepth].texture);
  const uint32_t velocityTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::MotionVector].texture);
  const uint32_t reactiveMaskTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::ReactiveMask].texture);
  const TemporalAATuning &tuning = settings.antiAliasing.temporalTuning;
  const bool temporalHistoryValid =
      ctx.frame.camera.historyValid && ctx.shared.historyColorReadValid;
  const bool usePreviousDepth =
      temporalHistoryValid &&
      nuri::isValid(
          ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture);
  const uint32_t previousDepthTexId =
      usePreviousDepth
          ? gpu_.getTextureBindlessIndex(
                ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture)
          : depthTexId;
  const bool usePreviousVelocity =
      temporalHistoryValid &&
      nuri::isValid(
          ctx.shared[FrameHistoryTextureSlot::PreviousMotionVector].texture);
  const uint32_t previousVelocityTexId =
      usePreviousVelocity
          ? gpu_.getTextureBindlessIndex(
                ctx.shared[FrameHistoryTextureSlot::PreviousMotionVector]
                    .texture)
          : velocityTexId;
  const uint32_t linearSamplerId =
      gpu_.getSamplerBindlessIndex(linearClampSampler_);
  const std::array recordingSamplers{linearClampSampler_};
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
      temporalHistoryValid &&
      nuri::isValid(ctx.shared[FrameTextureSlot::ReactiveMask].texture) &&
      tuning.reactiveStrength > 0.0f &&
      tuning.reactiveCurrentWeight > tuning.currentFrameWeight;
  if (useReactiveMask) {
    resolveFlags |= kTaaResolveFlagReactiveMask;
  }
  const bool useVelocityDilation =
      temporalHistoryValid &&
      tuning.velocityDilationMode != TemporalAAVelocityDilationMode::None;
  if (useVelocityDilation) {
    resolveFlags |= kTaaResolveFlagVelocityDilation;
  }
  const TextureDimensions sceneDimensions = gpu_.getTextureDimensions(
      ctx.shared[FrameTextureSlot::SceneColor].texture);
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
          .importTexture(ctx.shared[FrameTextureSlot::SceneColor].texture,
                         "taa_scene_color_current")
          .value();
  const auto historyWriteGraphTexture =
      ctx.graph
          .importTexture(
              ctx.shared[FrameHistoryTextureSlot::ColorWrite].texture,
              "taa_history_color_write")
          .value();
  if (usePreviousDepth) {
    ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].graph =
        ctx.graph
            .importTexture(
                ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture,
                "taa_previous_scene_depth")
            .value();
  }
  if (usePreviousVelocity) {
    ctx.shared[FrameHistoryTextureSlot::PreviousMotionVector].graph =
        ctx.graph
            .importTexture(
                ctx.shared[FrameHistoryTextureSlot::PreviousMotionVector]
                    .texture,
                "taa_previous_motion_vectors")
            .value();
  }
  std::array<TextureHandle, 7> resolveReads{};
  size_t resolveReadCount = 0u;
  resolveReads[resolveReadCount++] =
      ctx.shared[FrameTextureSlot::SceneColor].texture;
  resolveReads[resolveReadCount++] =
      ctx.shared[FrameTextureSlot::SceneDepth].texture;
  if (temporalHistoryValid) {
    resolveReads[resolveReadCount++] =
        ctx.shared[FrameHistoryTextureSlot::ColorRead].texture;
    resolveReads[resolveReadCount++] =
        ctx.shared[FrameTextureSlot::MotionVector].texture;
    if (usePreviousDepth) {
      resolveReads[resolveReadCount++] =
          ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture;
    }
    if (useReactiveMask) {
      resolveReads[resolveReadCount++] =
          ctx.shared[FrameTextureSlot::ReactiveMask].texture;
    }
    if (usePreviousVelocity) {
      resolveReads[resolveReadCount++] =
          ctx.shared[FrameHistoryTextureSlot::PreviousMotionVector].texture;
    }
  }
  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  const GpuTimingReport &timingReport = ctx.frame.gpuTiming;
  if (hasGpuTimingScope(timingReport, GpuTimingScope::TemporalAAResolve)) {
    aaMetrics.taaResolveGpuTimeMs =
        timingReport[GpuTimingScope::TemporalAAResolve].timeMs;
    aaMetrics.taaResolveGpuTimingSourceFrameIndex =
        timingReport[GpuTimingScope::TemporalAAResolve].sourceFrameIndex;
    aaMetrics.taaResolveGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::TemporalAADebug)) {
    aaMetrics.taaDebugGpuTimeMs =
        timingReport[GpuTimingScope::TemporalAADebug].timeMs;
    aaMetrics.taaDebugGpuTimingSourceFrameIndex =
        timingReport[GpuTimingScope::TemporalAADebug].sourceFrameIndex;
    aaMetrics.taaDebugGpuTimingAvailable = 1u;
  }
  aaMetrics.taaResolveWidth = sceneDimensions.width;
  aaMetrics.taaResolveHeight = sceneDimensions.height;
  const float jitterScale = settings.antiAliasing.temporalTuning.jitterScale;
  aaMetrics.taaJitterScale = jitterScale;
  aaMetrics.taaQualityPreset = settings.antiAliasing.qualityPreset;
  aaMetrics.taaCurrentFrameWeight = tuning.currentFrameWeight;
  aaMetrics.taaHistoryFrameWeight = 1.0f - tuning.currentFrameWeight;
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
      .currentWeightBits = floatBits(tuning.currentFrameWeight),
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
      pipeline_.graphicsPipeline(0u),
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&resolveConstants),
          sizeof(resolveConstants)),
      "TaaResolve", kDrawDebugColor);
  RenderGraphGraphicsPassDesc resolvePass{};
  resolvePass.color = {.loadOp = LoadOp::Clear,
                       .storeOp = StoreOp::Store,
                       .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  resolvePass.colorTexture = historyWriteGraphTexture;
  resolvePass.recordingSamplers = recordingSamplers;
  resolvePass.draws = std::span<const DrawItem>(&resolveDraw, 1u);
  resolvePass.gpuTimingScope = GpuTimingScope::TemporalAAResolve;
  resolvePass.debugLabel = "TAA Resolve Pass";
  resolvePass.debugColor = kTaaResolvePassDebugColor;
  const RenderGraphPassId resolvePassId =
      ctx.graph.addGraphicsPass(resolvePass).value();
  (void)ctx.graph
      .addImportedTextureReads(
          resolvePassId,
          std::span<const TextureHandle>(resolveReads.data(), resolveReadCount),
          resolvePass.debugLabel)
      .value();
  ctx.shared.historyWriteRequirements |=
      FrameTextureRequirementFlags::HistoryColor;
  ++aaMetrics.taaResolvePassCount;
  if (!ctx.frame.camera.historyValid) {
    ++aaMetrics.taaCurrentFallbackFrameCount;
  }
  aaMetrics.taaHistoryBandwidthEstimateBytes +=
      textureStorageBytes(gpu_,
                          ctx.shared[FrameTextureSlot::SceneColor].texture) +
      textureStorageBytes(
          gpu_, ctx.shared[FrameHistoryTextureSlot::ColorWrite].texture) +
      textureStorageBytes(gpu_,
                          ctx.shared[FrameTextureSlot::SceneDepth].texture) +
      (temporalHistoryValid
           ? textureStorageBytes(
                 gpu_, ctx.shared[FrameHistoryTextureSlot::ColorRead].texture) +
                 textureStorageBytes(
                     gpu_, ctx.shared[FrameTextureSlot::MotionVector].texture) +
                 (usePreviousDepth
                      ? textureStorageBytes(
                            gpu_,
                            ctx.shared
                                [FrameHistoryTextureSlot::PreviousSceneDepth]
                                    .texture)
                      : 0u) +
                 (useReactiveMask
                      ? textureStorageBytes(
                            gpu_,
                            ctx.shared[FrameTextureSlot::ReactiveMask].texture)
                      : 0u) +
                 (usePreviousVelocity
                      ? textureStorageBytes(
                            gpu_,
                            ctx.shared
                                [FrameHistoryTextureSlot::PreviousMotionVector]
                                    .texture)
                      : 0u)
           : 0u);
  TextureHandle displaySource =
      ctx.shared[FrameHistoryTextureSlot::ColorWrite].texture;
  uint32_t displaySourceTexId = historyWriteTexId;
  uint32_t displayMode = kTaaResolveModeCopyHistoryToScene;
  std::string_view displayLabel = "TAA Copy Back Pass";
  uint32_t displayColor = kTaaCopyBackPassDebugColor;
  bool debugDisplay = false;
  if (debugView == AntiAliasingDebugView::TAACurrentColor) {
    ctx.shared[FrameTextureSlot::SceneColor].graph = sceneGraphTexture;
    aaMetrics.taaResolvedSceneColorPublished = false;
    aaMetrics.taaDebugViewRendered = true;
    return Result<bool, std::string>::makeResult(true);
  }
  if (debugView == AntiAliasingDebugView::TAAPreviousHistory) {
    displaySource = ctx.shared[FrameHistoryTextureSlot::ColorRead].texture;
    displaySourceTexId = historyReadTexId;
    displayMode = kTaaResolveModePreviousHistory;
    displayLabel = "TAA Previous History Debug Pass";
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  } else if (isVelocityDebugView(debugView)) {
    displaySource = ctx.shared[FrameTextureSlot::MotionVector].texture;
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
    displaySource =
        usePreviousVelocity
            ? ctx.shared[FrameHistoryTextureSlot::PreviousMotionVector].texture
            : ctx.shared[FrameTextureSlot::MotionVector].texture;
    displaySourceTexId = previousVelocityTexId;
    displayMode = kTaaResolveModePreviousVelocity;
    displayLabel = "TAA Previous Velocity Debug Pass";
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  } else if (evaluationDebug) {
    displaySource = ctx.shared[FrameTextureSlot::SceneColor].texture;
    displaySourceTexId = sceneTexId;
    displayMode = evaluationDebug.mode;
    displayLabel = evaluationDebug.label;
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  } else if (debugView == AntiAliasingDebugView::TAAReactiveMask) {
    displaySource = ctx.shared[FrameTextureSlot::ReactiveMask].texture;
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
      pipeline_.graphicsPipeline(0u),
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&copyConstants),
          sizeof(copyConstants)),
      debugDisplay ? "TaaResolveDebug" : "TaaCopyBack", kDrawDebugColor);
  std::array<TextureHandle, 8> copyReads{};
  size_t copyReadCount = 0u;
  if (displaySamplesReactiveMask) {
    copyReads[copyReadCount++] =
        ctx.shared[FrameTextureSlot::ReactiveMask].texture;
  } else if (displaySamplesTemporalEvaluation) {
    copyReads[copyReadCount++] =
        ctx.shared[FrameTextureSlot::SceneColor].texture;
    copyReads[copyReadCount++] =
        ctx.shared[FrameTextureSlot::SceneDepth].texture;
    if (temporalHistoryValid) {
      copyReads[copyReadCount++] =
          ctx.shared[FrameHistoryTextureSlot::ColorRead].texture;
      copyReads[copyReadCount++] =
          ctx.shared[FrameTextureSlot::MotionVector].texture;
      if (usePreviousDepth) {
        copyReads[copyReadCount++] =
            ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture;
      }
      if (useReactiveMask) {
        copyReads[copyReadCount++] =
            ctx.shared[FrameTextureSlot::ReactiveMask].texture;
      }
      if (usePreviousVelocity) {
        copyReads[copyReadCount++] =
            ctx.shared[FrameHistoryTextureSlot::PreviousMotionVector].texture;
      }
    }
  } else {
    copyReads[copyReadCount++] = displaySource;
  }
  if (debugDisplay) {
    copyReads[copyReadCount++] =
        ctx.shared[FrameHistoryTextureSlot::ColorWrite].texture;
  }
  if (displaySamplesTemporalEvaluation) {
    const uint32_t debugOutputTexId = gpu_.getTextureBindlessIndex(
        ctx.shared[FrameTextureSlot::FrameColor].texture);
    const auto debugOutput =
        ctx.graph
            .importTexture(ctx.shared[FrameTextureSlot::FrameColor].texture,
                           "taa_debug_output")
            .value();
    RenderGraphGraphicsPassDesc debugPass{};
    debugPass.color = {.loadOp = LoadOp::Clear,
                       .storeOp = StoreOp::Store,
                       .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    debugPass.colorTexture = debugOutput;
    debugPass.recordingSamplers = recordingSamplers;
    debugPass.draws = std::span<const DrawItem>(&copyDraw, 1u);
    debugPass.gpuTimingScope = GpuTimingScope::TemporalAADebug;
    debugPass.debugLabel = displayLabel;
    debugPass.debugColor = displayColor;
    const RenderGraphPassId debugPassId =
        ctx.graph.addGraphicsPass(debugPass).value();
    (void)ctx.graph
        .addImportedTextureReads(
            debugPassId,
            std::span<const TextureHandle>(copyReads.data(), copyReadCount),
            debugPass.debugLabel)
        .value();
    TAAResolvePushConstants debugCopyConstants = copyConstants;
    debugCopyConstants.currentTexId = debugOutputTexId;
    debugCopyConstants.mode = kTaaResolveModeCopyCurrent;
    const DrawItem debugCopyDraw = makeFullscreenDraw(
        pipeline_.graphicsPipeline(0u),
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&debugCopyConstants),
            sizeof(debugCopyConstants)),
        "TaaDebugCopyBack", kDrawDebugColor);
    const std::array<TextureHandle, 1> debugCopyReads{
        ctx.shared[FrameTextureSlot::FrameColor].texture};
    RenderGraphGraphicsPassDesc debugCopyPass{};
    debugCopyPass.color = {.loadOp = LoadOp::Clear,
                           .storeOp = StoreOp::Store,
                           .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    debugCopyPass.colorTexture = sceneGraphTexture;
    debugCopyPass.recordingSamplers = recordingSamplers;
    debugCopyPass.draws = std::span<const DrawItem>(&debugCopyDraw, 1u);
    debugCopyPass.gpuTimingScope = GpuTimingScope::TemporalAADebug;
    debugCopyPass.debugLabel = "TAA Debug Copy Back Pass";
    debugCopyPass.debugColor = kTaaCopyBackPassDebugColor;
    const RenderGraphPassId debugCopyPassId =
        ctx.graph.addGraphicsPass(debugCopyPass).value();
    (void)ctx.graph
        .addImportedTextureReads(debugCopyPassId, debugCopyReads,
                                 debugCopyPass.debugLabel)
        .value();
  } else {
    RenderGraphGraphicsPassDesc copyPass{};
    copyPass.color = {.loadOp = LoadOp::Clear,
                      .storeOp = StoreOp::Store,
                      .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    copyPass.colorTexture = sceneGraphTexture;
    copyPass.recordingSamplers = recordingSamplers;
    copyPass.draws = std::span<const DrawItem>(&copyDraw, 1u);
    copyPass.gpuTimingScope = debugDisplay ? GpuTimingScope::TemporalAADebug
                                           : GpuTimingScope::TemporalAACopyBack;
    copyPass.debugLabel = displayLabel;
    copyPass.debugColor = displayColor;
    const RenderGraphPassId copyPassId =
        ctx.graph.addGraphicsPass(copyPass).value();
    (void)ctx.graph
        .addImportedTextureReads(
            copyPassId,
            std::span<const TextureHandle>(copyReads.data(), copyReadCount),
            copyPass.debugLabel)
        .value();
  }
  ctx.shared[FrameTextureSlot::SceneColor].graph = sceneGraphTexture;
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
        textureStorageBytes(
            gpu_, ctx.shared[FrameTextureSlot::MotionVector].texture) +
        textureStorageBytes(gpu_,
                            ctx.shared[FrameTextureSlot::SceneColor].texture);
  }
  aaMetrics.taaHistoryBandwidthEstimateBytes +=
      textureStorageBytes(gpu_, displaySource) +
      textureStorageBytes(gpu_,
                          ctx.shared[FrameTextureSlot::SceneColor].texture) +
      (displaySamplesTemporalEvaluation
           ? textureStorageBytes(
                 gpu_, ctx.shared[FrameTextureSlot::FrameColor].texture) *
                     2u +
                 textureStorageBytes(
                     gpu_, ctx.shared[FrameTextureSlot::SceneDepth].texture)
           : 0u) +
      (displaySamplesTemporalEvaluation && temporalHistoryValid
           ? textureStorageBytes(
                 gpu_, ctx.shared[FrameHistoryTextureSlot::ColorRead].texture) +
                 textureStorageBytes(
                     gpu_, ctx.shared[FrameTextureSlot::MotionVector].texture) +
                 (usePreviousDepth
                      ? textureStorageBytes(
                            gpu_,
                            ctx.shared
                                [FrameHistoryTextureSlot::PreviousSceneDepth]
                                    .texture)
                      : 0u) +
                 (useReactiveMask
                      ? textureStorageBytes(
                            gpu_,
                            ctx.shared[FrameTextureSlot::ReactiveMask].texture)
                      : 0u) +
                 (usePreviousVelocity
                      ? textureStorageBytes(
                            gpu_,
                            ctx.shared
                                [FrameHistoryTextureSlot::PreviousMotionVector]
                                    .texture)
                      : 0u)
           : 0u);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TemporalAAClearPass::publishFrameData(FrameBuildContext &ctx) {
  const PresentationAAPlan &plan = ctx.frame.presentationAA;
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
