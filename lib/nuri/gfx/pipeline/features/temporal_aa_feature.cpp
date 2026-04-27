#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/temporal_aa_feature.h"

#include "nuri/gfx/frame/render_frame_context.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

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
constexpr uint32_t kTaaHistoryFilterModeCatmullRom = 0u;
constexpr uint32_t kTaaHistoryFilterModeBilinear = 1u;
constexpr uint32_t kTaaResolvePassDebugColor = 0xffaa55ffu;
constexpr uint32_t kTaaCopyBackPassDebugColor = 0xff8844ffu;
constexpr uint32_t kTaaDebugPassDebugColor = 0xffcc77ffu;
constexpr uint32_t kDrawDebugColor = 0xffaa44ffu;
constexpr uint32_t kInvalidTextureBindlessIndex = 0xFFFFFFFFu;
constexpr float kDefaultTaaCurrentFrameWeight = 0.06f;
constexpr float kDefaultTaaVelocityRejectionThreshold = 0.0015f;
constexpr float kDefaultTaaVelocityBlendScale = 0.35f;
constexpr float kDefaultTaaVarianceGamma = 1.50f;

struct TAAResolvePushConstants {
  uint32_t currentTexId = 0u;
  uint32_t historyTexId = 0u;
  uint32_t depthTexId = 0u;
  uint32_t velocityTexId = 0u;
  uint32_t previousVelocityTexId = 0u;
  uint32_t reactiveMaskTexId = 0u;
  uint32_t currentSamplerId = 0u;
  uint32_t historySamplerId = 0u;
  uint32_t depthSamplerId = 0u;
  uint32_t velocitySamplerId = 0u;
  uint32_t reactiveMaskSamplerId = 0u;
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
};
static_assert(sizeof(TAAResolvePushConstants) <= 128);

[[nodiscard]] inline bool
isTAAEnabled(const RenderSettings &settings) noexcept {
  return sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
         AntiAliasingMode::TAA;
}

[[nodiscard]] inline bool
isVelocityDebugView(AntiAliasingDebugView view) noexcept {
  return view == AntiAliasingDebugView::MotionVectors ||
         view == AntiAliasingDebugView::VelocityMagnitude;
}

[[nodiscard]] inline bool
isTaaResolveEvaluationDebugView(AntiAliasingDebugView view) noexcept {
  return view == AntiAliasingDebugView::TAAHistoryValidity ||
         view == AntiAliasingDebugView::TAARejectionMask ||
         view == AntiAliasingDebugView::TAABlendFactor ||
         view == AntiAliasingDebugView::TAAClampDelta ||
         view == AntiAliasingDebugView::TAAPixelInspector ||
         view == AntiAliasingDebugView::TAAReactiveMask ||
         view == AntiAliasingDebugView::TAADisocclusionMask ||
         view == AntiAliasingDebugView::TAAVelocityDilation ||
         view == AntiAliasingDebugView::TAAReprojectedHistory ||
         view == AntiAliasingDebugView::TAAResolveConfidence ||
         view == AntiAliasingDebugView::TAAClampDiagnostics;
}

[[nodiscard]] inline std::string_view
taaResolveEvaluationDebugLabel(AntiAliasingDebugView view) noexcept {
  switch (view) {
  case AntiAliasingDebugView::TAAHistoryValidity:
    return "TAA History Validity Debug Pass";
  case AntiAliasingDebugView::TAARejectionMask:
    return "TAA Rejection Mask Debug Pass";
  case AntiAliasingDebugView::TAABlendFactor:
    return "TAA Blend Factor Debug Pass";
  case AntiAliasingDebugView::TAAClampDelta:
    return "TAA Clamp Delta Debug Pass";
  case AntiAliasingDebugView::TAAPixelInspector:
    return "TAA Pixel Inspector Debug Pass";
  case AntiAliasingDebugView::TAAReactiveMask:
    return "TAA Reactive Mask Debug Pass";
  case AntiAliasingDebugView::TAADisocclusionMask:
    return "TAA Disocclusion Mask Debug Pass";
  case AntiAliasingDebugView::TAAVelocityDilation:
    return "TAA Velocity Dilation Debug Pass";
  case AntiAliasingDebugView::TAAReprojectedHistory:
    return "TAA Reprojected History Debug Pass";
  case AntiAliasingDebugView::TAAResolveConfidence:
    return "TAA Resolve Confidence Debug Pass";
  case AntiAliasingDebugView::TAAClampDiagnostics:
    return "TAA Clamp Diagnostics Debug Pass";
  default:
    return "TAA Debug Display Pass";
  }
}

[[nodiscard]] inline uint32_t
taaResolveModeForDebugView(AntiAliasingDebugView view) noexcept {
  switch (view) {
  case AntiAliasingDebugView::TAAHistoryValidity:
    return kTaaResolveModeHistoryValidity;
  case AntiAliasingDebugView::TAARejectionMask:
    return kTaaResolveModeRejectionMask;
  case AntiAliasingDebugView::TAABlendFactor:
    return kTaaResolveModeBlendFactor;
  case AntiAliasingDebugView::TAAClampDelta:
    return kTaaResolveModeClampDelta;
  case AntiAliasingDebugView::TAAPixelInspector:
    return kTaaResolveModePixelInspector;
  case AntiAliasingDebugView::TAAReactiveMask:
    return kTaaResolveModeReactiveMask;
  case AntiAliasingDebugView::TAADisocclusionMask:
    return kTaaResolveModeDisocclusionMask;
  case AntiAliasingDebugView::TAAVelocityDilation:
    return kTaaResolveModeVelocityDilation;
  case AntiAliasingDebugView::TAAReprojectedHistory:
    return kTaaResolveModeReprojectedHistory;
  case AntiAliasingDebugView::TAAResolveConfidence:
    return kTaaResolveModeResolveConfidence;
  case AntiAliasingDebugView::TAAClampDiagnostics:
    return kTaaResolveModeClampDiagnostics;
  case AntiAliasingDebugView::MotionVectors:
    return kTaaResolveModeVelocityMotionVectors;
  case AntiAliasingDebugView::VelocityMagnitude:
    return kTaaResolveModeVelocityMagnitude;
  default:
    return kTaaResolveModeCopyCurrent;
  }
}

struct TAAResolveTuning {
  float currentWeight = kDefaultTaaCurrentFrameWeight;
  float depthDiscontinuityThreshold = 0.01f;
  float velocityRejectionThreshold = kDefaultTaaVelocityRejectionThreshold;
  float velocityBlendScale = kDefaultTaaVelocityBlendScale;
  float motionCurrentWeight = 0.35f;
  float disocclusionCurrentWeight = 0.65f;
  float clampCurrentWeight = 0.50f;
  float clampBlendAttenuation = 0.35f;
  float varianceGamma = kDefaultTaaVarianceGamma;
  float hdrWeightStrength = 0.50f;
  float reactiveCurrentWeight = 0.85f;
  float reactiveStrength = 1.0f;
  float velocityDilationDepthThreshold = 0.01f;
  TemporalAAClampMode clampMode = TemporalAAClampMode::Variance;
  TemporalAAHdrWeightingMode hdrWeightingMode =
      TemporalAAHdrWeightingMode::Luminance;
  TemporalAAVelocityDilationMode velocityDilationMode =
      TemporalAAVelocityDilationMode::ClosestDepth;
  TemporalAAHistoryFilterMode historyFilterMode =
      TemporalAAHistoryFilterMode::CatmullRom;
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

[[nodiscard]] RenderPipelineDesc fullscreenPipelineDesc(Format colorFormat,
                                                        ShaderHandle vertex,
                                                        ShaderHandle fragment) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertex,
      .fragmentShader = fragment,
      .colorFormats = {colorFormat},
      .depthFormat = Format::Count,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
  };
}

[[nodiscard]] DrawItem
makeFullscreenDraw(RenderPipelineHandle pipeline,
                   std::span<const std::byte> pushConstants,
                   std::string_view label) {
  DrawItem draw{};
  draw.pipeline = pipeline;
  draw.vertexCount = 3u;
  draw.instanceCount = 1u;
  draw.pushConstants = pushConstants;
  draw.debugLabel = label;
  draw.debugColor = kDrawDebugColor;
  return draw;
}

[[nodiscard]] uint64_t textureBytesPerPixel(Format format) {
  switch (format) {
  case Format::RG16_FLOAT:
    return sizeof(uint16_t) * 2u;
  case Format::RG32_FLOAT:
    return sizeof(float) * 2u;
  case Format::R32_UINT:
  case Format::R32_FLOAT:
  case Format::D32_FLOAT:
    return sizeof(uint32_t);
  case Format::D16_UNORM:
    return sizeof(uint16_t);
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return 4u;
  case Format::RGBA16_FLOAT:
    return 8u;
  case Format::RGBA32_FLOAT:
    return 16u;
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::Count:
    break;
  }
  return 0u;
}

[[nodiscard]] uint64_t textureStorageBytes(GPUDevice &gpu,
                                           TextureHandle texture) {
  if (!nuri::isValid(texture)) {
    return 0u;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  return static_cast<uint64_t>(std::max(dimensions.width, 1u)) *
         static_cast<uint64_t>(std::max(dimensions.height, 1u)) *
         textureBytesPerPixel(gpu.getTextureFormat(texture));
}

[[nodiscard]] float sanitizedCurrentWeight(
    const RenderSettings::AntiAliasingSettings &settings) noexcept {
  const float value = settings.debug.taaCurrentFrameWeight;
  if (!std::isfinite(value)) {
    return kDefaultTaaCurrentFrameWeight;
  }
  return std::clamp(value, 0.0f, 1.0f);
}

[[nodiscard]] TAAResolveTuning sanitizedResolveTuning(
    const RenderSettings::AntiAliasingSettings &settings) noexcept {
  TAAResolveTuning tuning{};
  tuning.currentWeight = sanitizedCurrentWeight(settings);
  tuning.depthDiscontinuityThreshold =
      std::isfinite(settings.debug.taaDepthDiscontinuityThreshold)
          ? std::clamp(settings.debug.taaDepthDiscontinuityThreshold, 0.0f,
                       1.0f)
          : 0.01f;
  tuning.velocityRejectionThreshold =
      std::isfinite(settings.debug.taaVelocityRejectionThreshold)
          ? std::clamp(settings.debug.taaVelocityRejectionThreshold, 0.0f, 1.0f)
          : kDefaultTaaVelocityRejectionThreshold;
  if (!std::isfinite(settings.debug.taaVelocityBlendScale) ||
      settings.debug.taaVelocityBlendScale > 4.0f) {
    tuning.velocityBlendScale = kDefaultTaaVelocityBlendScale;
  } else {
    tuning.velocityBlendScale =
        std::clamp(settings.debug.taaVelocityBlendScale, 0.0f, 4.0f);
  }
  tuning.motionCurrentWeight =
      std::isfinite(settings.debug.taaMotionCurrentWeight)
          ? std::clamp(settings.debug.taaMotionCurrentWeight, 0.0f, 1.0f)
          : 0.35f;
  tuning.motionCurrentWeight =
      std::max(tuning.motionCurrentWeight, tuning.currentWeight);
  tuning.disocclusionCurrentWeight =
      std::isfinite(settings.debug.taaDisocclusionCurrentWeight)
          ? std::clamp(settings.debug.taaDisocclusionCurrentWeight, 0.0f, 1.0f)
          : 0.65f;
  tuning.disocclusionCurrentWeight =
      std::max(tuning.disocclusionCurrentWeight, tuning.currentWeight);
  tuning.clampCurrentWeight =
      std::isfinite(settings.debug.taaClampCurrentWeight)
          ? std::clamp(settings.debug.taaClampCurrentWeight, 0.0f, 1.0f)
          : 0.50f;
  tuning.clampCurrentWeight =
      std::max(tuning.clampCurrentWeight, tuning.currentWeight);
  tuning.clampBlendAttenuation =
      std::isfinite(settings.debug.taaClampBlendAttenuation)
          ? std::clamp(settings.debug.taaClampBlendAttenuation, 0.0f, 1.0f)
          : 0.35f;
  tuning.varianceGamma =
      std::isfinite(settings.debug.taaVarianceGamma)
          ? std::clamp(settings.debug.taaVarianceGamma, 0.0f, 4.0f)
          : kDefaultTaaVarianceGamma;
  tuning.hdrWeightStrength =
      std::isfinite(settings.debug.taaHdrWeightStrength)
          ? std::clamp(settings.debug.taaHdrWeightStrength, 0.0f, 1.0f)
          : 0.50f;
  tuning.reactiveCurrentWeight =
      std::isfinite(settings.debug.taaReactiveCurrentWeight)
          ? std::clamp(settings.debug.taaReactiveCurrentWeight, 0.0f, 1.0f)
          : 0.85f;
  tuning.reactiveStrength =
      std::isfinite(settings.debug.taaReactiveStrength)
          ? std::clamp(settings.debug.taaReactiveStrength, 0.0f, 4.0f)
          : 1.0f;
  tuning.velocityDilationDepthThreshold =
      std::isfinite(settings.debug.taaVelocityDilationDepthThreshold)
          ? std::clamp(settings.debug.taaVelocityDilationDepthThreshold, 0.0f,
                       1.0f)
          : 0.01f;
  tuning.clampMode = sanitizeTemporalAAClampMode(settings.debug.taaClampMode);
  tuning.hdrWeightingMode =
      sanitizeTemporalAAHdrWeightingMode(settings.debug.taaHdrWeightingMode);
  tuning.velocityDilationMode = sanitizeTemporalAAVelocityDilationMode(
      settings.debug.taaVelocityDilationMode);
  tuning.historyFilterMode =
      sanitizeTemporalAAHistoryFilterMode(settings.debug.taaHistoryFilterMode);
  return tuning;
}

} // namespace

bool TemporalAAMotionVectorClearPass::isEnabled(
    const FrameBuildContext &ctx) const {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  return isTAAEnabled(settings);
}

Result<bool, std::string>
TemporalAAMotionVectorClearPass::prepare(FrameBuildContext &ctx) {
  (void)ctx;
  return Result<bool, std::string>::makeResult(false);
}

Result<bool, std::string>
TemporalAAMotionVectorClearPass::build(FrameBuildContext &ctx) {
  if (nuri::isValid(ctx.shared.motionVectorGraphTexture)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(ctx.shared.motionVectorTexture)) {
    return Result<bool, std::string>::makeResult(false);
  }

  auto motionVectorImport = ctx.graph.importTexture(
      ctx.shared.motionVectorTexture, "taa_motion_vectors");
  if (motionVectorImport.hasError()) {
    return Result<bool, std::string>::makeError(motionVectorImport.error());
  }
  ctx.shared.motionVectorGraphTexture = motionVectorImport.value();
  ctx.frame.metrics.antiAliasing.motionVectorGraphPublished = true;

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color =
      AttachmentColor{.loadOp = LoadOp::Clear,
                      .storeOp = StoreOp::Store,
                      .clearColor = kFrameCompositionMotionVectorClearValue};
  passDesc.colorTexture = ctx.shared.motionVectorGraphTexture;
  passDesc.debugLabel = "Temporal AA Motion Vector Clear";
  passDesc.debugColor = 0xff44aaff;

  auto addResult = ctx.graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  aaMetrics.motionVectorClearPassCount = 1u;
  aaMetrics.motionVectorClearBytes = aaMetrics.motionVectorTextureBytes;
  return Result<bool, std::string>::makeResult(true);
}

bool TemporalAAReactiveMaskClearPass::isEnabled(
    const FrameBuildContext &ctx) const {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  return isTAAEnabled(settings);
}

Result<bool, std::string>
TemporalAAReactiveMaskClearPass::prepare(FrameBuildContext &ctx) {
  (void)ctx;
  return Result<bool, std::string>::makeResult(false);
}

Result<bool, std::string>
TemporalAAReactiveMaskClearPass::build(FrameBuildContext &ctx) {
  if (nuri::isValid(ctx.shared.reactiveMaskGraphTexture)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(ctx.shared.reactiveMaskTexture)) {
    return Result<bool, std::string>::makeResult(false);
  }

  auto reactiveMaskImport = ctx.graph.importTexture(
      ctx.shared.reactiveMaskTexture, "taa_reactive_mask");
  if (reactiveMaskImport.hasError()) {
    return Result<bool, std::string>::makeError(reactiveMaskImport.error());
  }
  ctx.shared.reactiveMaskGraphTexture = reactiveMaskImport.value();
  ctx.frame.metrics.antiAliasing.reactiveMaskGraphPublished = true;

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color =
      AttachmentColor{.loadOp = LoadOp::Clear,
                      .storeOp = StoreOp::Store,
                      .clearColor = kFrameCompositionReactiveMaskClearValue};
  passDesc.colorTexture = ctx.shared.reactiveMaskGraphTexture;
  passDesc.debugLabel = "Temporal AA Reactive Mask Clear";
  passDesc.debugColor = 0xff33cc88;

  auto addResult = ctx.graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  aaMetrics.reactiveMaskPassBandwidthEstimateBytes =
      aaMetrics.reactiveMaskTextureBytes;
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
  if (nuri::isValid(pipeline_)) {
    gpu_.destroyRenderPipeline(pipeline_);
  }
  if (nuri::isValid(linearClampSampler_)) {
    gpu_.destroySampler(linearClampSampler_);
  }
  if (nuri::isValid(vertexShader_)) {
    gpu_.destroyShaderModule(vertexShader_);
  }
  if (nuri::isValid(fragmentShader_)) {
    gpu_.destroyShaderModule(fragmentShader_);
  }
}

bool TemporalAAResolvePass::isEnabled(const FrameBuildContext &ctx) const {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  return isTAAEnabled(settings);
}

Result<bool, std::string>
TemporalAAResolvePass::prepare(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(false);
  }

  if (!initialized_) {
    shader_ = Shader::create("taa_resolve", gpu_);
    if (!shader_) {
      return Result<bool, std::string>::makeError(
          "TemporalAAResolvePass::prepare: failed to create shader");
    }
    auto vertexResult =
        shader_->compileFromFile(vertexPath_.string(), ShaderStage::Vertex);
    if (vertexResult.hasError()) {
      return Result<bool, std::string>::makeError(vertexResult.error());
    }
    auto fragmentResult =
        shader_->compileFromFile(fragmentPath_.string(), ShaderStage::Fragment);
    if (fragmentResult.hasError()) {
      if (nuri::isValid(vertexResult.value())) {
        gpu_.destroyShaderModule(vertexResult.value());
      }
      return Result<bool, std::string>::makeError(fragmentResult.error());
    }
    vertexShader_ = vertexResult.value();
    fragmentShader_ = fragmentResult.value();
    const SamplerDesc linearClampDesc{
        .minFilter = SamplerFilter::Linear,
        .magFilter = SamplerFilter::Linear,
        .mipMode = SamplerMipMode::Disabled,
        .wrapU = SamplerWrapMode::Clamp,
        .wrapV = SamplerWrapMode::Clamp,
        .wrapW = SamplerWrapMode::Clamp,
        .mipLodMin = 0.0f,
        .mipLodMax = 0.0f,
        .maxAnisotropy = 1u,
        .depthCompareEnabled = false,
    };
    auto samplerResult =
        gpu_.createSampler(linearClampDesc, "taa_linear_clamp");
    if (samplerResult.hasError()) {
      if (nuri::isValid(fragmentShader_)) {
        gpu_.destroyShaderModule(fragmentShader_);
        fragmentShader_ = {};
      }
      if (nuri::isValid(vertexShader_)) {
        gpu_.destroyShaderModule(vertexShader_);
        vertexShader_ = {};
      }
      return Result<bool, std::string>::makeError(samplerResult.error());
    }
    linearClampSampler_ = samplerResult.value();
    initialized_ = true;
  }

  if (nuri::isValid(pipeline_) &&
      pipelineColorFormat_ == kFrameCompositionSceneColorFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (nuri::isValid(pipeline_)) {
    gpu_.destroyRenderPipeline(pipeline_);
    pipeline_ = {};
    pipelineColorFormat_ = Format::Count;
  }
  auto pipelineResult = gpu_.createRenderPipeline(
      fullscreenPipelineDesc(kFrameCompositionSceneColorFormat, vertexShader_,
                             fragmentShader_),
      "taa_resolve");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  pipeline_ = pipelineResult.value();
  pipelineColorFormat_ = kFrameCompositionSceneColorFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TemporalAAResolvePass::build(FrameBuildContext &ctx) {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  const AntiAliasingDebugView debugView =
      sanitizeAntiAliasingDebugView(settings.antiAliasing.debug.view);
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(false);
  }

  if (!nuri::isValid(pipeline_) ||
      !nuri::isValid(ctx.shared.sceneColorTexture) ||
      !nuri::isValid(ctx.shared.historyColorReadTexture) ||
      !nuri::isValid(ctx.shared.historyColorWriteTexture) ||
      !nuri::isValid(ctx.shared.sceneDepthTexture) ||
      !nuri::isValid(ctx.shared.motionVectorTexture) ||
      !nuri::isValid(ctx.shared.reactiveMaskTexture)) {
    return Result<bool, std::string>::makeResult(false);
  }

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
  const bool temporalHistoryValid = ctx.frame.camera.historyValid;
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
  if (sceneTexId == kInvalidTextureBindlessIndex ||
      historyReadTexId == kInvalidTextureBindlessIndex ||
      historyWriteTexId == kInvalidTextureBindlessIndex ||
      depthTexId == kInvalidTextureBindlessIndex ||
      velocityTexId == kInvalidTextureBindlessIndex ||
      reactiveMaskTexId == kInvalidTextureBindlessIndex ||
      (usePreviousVelocity &&
       previousVelocityTexId == kInvalidTextureBindlessIndex) ||
      linearSamplerId == kInvalidTextureBindlessIndex ||
      pointSamplerId == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "TemporalAAResolvePass::build: invalid bindless texture or sampler");
  }

  uint32_t resolveFlags =
      temporalHistoryValid ? kTaaResolveFlagHistoryValid : 0u;
  if (usePreviousVelocity) {
    resolveFlags |= kTaaResolveFlagPreviousVelocityValid;
  }
  if (temporalHistoryValid) {
    resolveFlags |= kTaaResolvePhase5Flags;
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
  const uint32_t currentWeightBits =
      std::bit_cast<uint32_t>(tuning.currentWeight);
  const TextureDimensions sceneDimensions =
      gpu_.getTextureDimensions(ctx.shared.sceneColorTexture);
  const float inverseWidth =
      1.0f / static_cast<float>(std::max(sceneDimensions.width, 1u));
  const float inverseHeight =
      1.0f / static_cast<float>(std::max(sceneDimensions.height, 1u));
  const uint32_t inverseWidthBits = std::bit_cast<uint32_t>(inverseWidth);
  const uint32_t inverseHeightBits = std::bit_cast<uint32_t>(inverseHeight);
  const uint32_t depthThresholdBits =
      std::bit_cast<uint32_t>(tuning.depthDiscontinuityThreshold);
  const uint32_t velocityThresholdBits =
      std::bit_cast<uint32_t>(tuning.velocityRejectionThreshold);
  const uint32_t velocityBlendScaleBits =
      std::bit_cast<uint32_t>(tuning.velocityBlendScale);
  const uint32_t motionCurrentWeightBits =
      std::bit_cast<uint32_t>(tuning.motionCurrentWeight);
  const uint32_t disocclusionWeightBits =
      std::bit_cast<uint32_t>(tuning.disocclusionCurrentWeight);
  const uint32_t clampCurrentWeightBits =
      std::bit_cast<uint32_t>(tuning.clampCurrentWeight);
  const uint32_t clampAttenuationBits =
      std::bit_cast<uint32_t>(tuning.clampBlendAttenuation);
  const uint32_t varianceGammaBits =
      std::bit_cast<uint32_t>(tuning.varianceGamma);
  const uint32_t hdrWeightStrengthBits =
      std::bit_cast<uint32_t>(tuning.hdrWeightStrength);
  const uint32_t reactiveCurrentWeightBits =
      std::bit_cast<uint32_t>(tuning.reactiveCurrentWeight);
  const uint32_t reactiveStrengthBits =
      std::bit_cast<uint32_t>(tuning.reactiveStrength);
  const uint32_t velocityDilationDepthThresholdBits =
      std::bit_cast<uint32_t>(tuning.velocityDilationDepthThreshold);
  const uint32_t clampMode =
      static_cast<uint32_t>(sanitizeTemporalAAClampMode(tuning.clampMode));
  const uint32_t hdrWeightingMode = static_cast<uint32_t>(
      sanitizeTemporalAAHdrWeightingMode(tuning.hdrWeightingMode));
  const uint32_t velocityDilationMode = static_cast<uint32_t>(
      sanitizeTemporalAAVelocityDilationMode(tuning.velocityDilationMode));
  const uint32_t historyFilterMode =
      sanitizeTemporalAAHistoryFilterMode(tuning.historyFilterMode) ==
              TemporalAAHistoryFilterMode::Bilinear
          ? kTaaHistoryFilterModeBilinear
          : kTaaHistoryFilterModeCatmullRom;

  auto importScene = ctx.graph.importTexture(ctx.shared.sceneColorTexture,
                                             "taa_scene_color_current");
  if (importScene.hasError()) {
    return Result<bool, std::string>::makeError(importScene.error());
  }
  auto importHistoryWrite = ctx.graph.importTexture(
      ctx.shared.historyColorWriteTexture, "taa_history_color_write");
  if (importHistoryWrite.hasError()) {
    return Result<bool, std::string>::makeError(importHistoryWrite.error());
  }
  if (usePreviousVelocity) {
    auto importPreviousVelocity = ctx.graph.importTexture(
        ctx.shared.previousMotionVectorTexture, "taa_previous_motion_vectors");
    if (importPreviousVelocity.hasError()) {
      return Result<bool, std::string>::makeError(
          importPreviousVelocity.error());
    }
    ctx.shared.previousMotionVectorGraphTexture =
        importPreviousVelocity.value();
  }

  std::array<TextureHandle, 6> resolveReads{};
  size_t resolveReadCount = 0u;
  resolveReads[resolveReadCount++] = ctx.shared.sceneColorTexture;
  if (temporalHistoryValid) {
    resolveReads[resolveReadCount++] = ctx.shared.historyColorReadTexture;
    resolveReads[resolveReadCount++] = ctx.shared.sceneDepthTexture;
    resolveReads[resolveReadCount++] = ctx.shared.motionVectorTexture;
    if (useReactiveMask) {
      resolveReads[resolveReadCount++] = ctx.shared.reactiveMaskTexture;
    }
    if (usePreviousVelocity) {
      resolveReads[resolveReadCount++] = ctx.shared.previousMotionVectorTexture;
    }
  }

  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  aaMetrics.taaResolveWidth = sceneDimensions.width;
  aaMetrics.taaResolveHeight = sceneDimensions.height;
  const float jitterScale =
      ctx.frame.settings != nullptr
          ? ctx.frame.settings->antiAliasing.debug.taaJitterScale
          : 0.75f;
  aaMetrics.taaJitterScale =
      std::isfinite(jitterScale) ? std::clamp(jitterScale, 0.0f, 1.0f) : 0.75f;
  aaMetrics.taaCurrentFrameWeight = tuning.currentWeight;
  aaMetrics.taaHistoryFrameWeight = 1.0f - tuning.currentWeight;
  aaMetrics.taaHistoryValidPercent =
      ctx.frame.camera.historyValid ? 100.0f : 0.0f;
  aaMetrics.taaOutOfBoundsFallbackEnabled = true;
  aaMetrics.taaBilinearHistorySampling =
      tuning.historyFilterMode == TemporalAAHistoryFilterMode::Bilinear;
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
  aaMetrics.taaVelocityRejectionEnabled = usePreviousVelocity;
  aaMetrics.taaPreviousVelocityDisocclusionEnabled = usePreviousVelocity;
  aaMetrics.taaNeighborhoodClampEnabled = temporalHistoryValid;
  aaMetrics.taaAdaptiveBlendEnabled = temporalHistoryValid;
  aaMetrics.taaClampBlendAttenuationEnabled = temporalHistoryValid;
  aaMetrics.taaNeighborhoodFallbackEnabled = temporalHistoryValid;
  aaMetrics.taaHdrWeightingEnabled =
      tuning.hdrWeightingMode != TemporalAAHdrWeightingMode::None &&
      tuning.hdrWeightStrength > 0.0f;
  aaMetrics.taaReactiveMaskEnabled = useReactiveMask;
  aaMetrics.taaVelocityDilationEnabled = useVelocityDilation;
  aaMetrics.taaReactiveCoverageEstimate =
      aaMetrics.taaAlphaMaskedCoverageEstimate;
  aaMetrics.taaDisocclusionRejectionEstimate =
      std::max(aaMetrics.velocityMissingPreviousRatio,
               aaMetrics.velocityEdgeDiscontinuityEstimate);
  aaMetrics.taaVelocityDilationAffectedEstimate =
      useVelocityDilation ? aaMetrics.velocityEdgeDiscontinuityEstimate : 0.0f;

  const TAAResolvePushConstants resolveConstants{
      .currentTexId = sceneTexId,
      .historyTexId = historyReadTexId,
      .depthTexId = depthTexId,
      .velocityTexId = velocityTexId,
      .previousVelocityTexId = previousVelocityTexId,
      .reactiveMaskTexId = reactiveMaskTexId,
      .currentSamplerId = linearSamplerId,
      .historySamplerId = linearSamplerId,
      .depthSamplerId = pointSamplerId,
      .velocitySamplerId = pointSamplerId,
      .reactiveMaskSamplerId = pointSamplerId,
      .flags = resolveFlags,
      .mode = kTaaResolveModeResolve,
      .currentWeightBits = currentWeightBits,
      .inverseWidthBits = inverseWidthBits,
      .inverseHeightBits = inverseHeightBits,
      .depthThresholdBits = depthThresholdBits,
      .velocityThresholdBits = velocityThresholdBits,
      .velocityBlendScaleBits = velocityBlendScaleBits,
      .disocclusionWeightBits = disocclusionWeightBits,
      .clampAttenuationBits = clampAttenuationBits,
      .varianceGammaBits = varianceGammaBits,
      .hdrWeightStrengthBits = hdrWeightStrengthBits,
      .reactiveCurrentWeightBits = reactiveCurrentWeightBits,
      .reactiveStrengthBits = reactiveStrengthBits,
      .velocityDilationDepthThresholdBits = velocityDilationDepthThresholdBits,
      .clampMode = clampMode,
      .hdrWeightingMode = hdrWeightingMode,
      .velocityDilationMode = velocityDilationMode,
      .motionCurrentWeightBits = motionCurrentWeightBits,
      .clampCurrentWeightBits = clampCurrentWeightBits,
      .historyFilterMode = historyFilterMode,
  };
  const DrawItem resolveDraw = makeFullscreenDraw(
      pipeline_,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&resolveConstants),
          sizeof(resolveConstants)),
      "TaaResolve");

  RenderGraphGraphicsPassDesc resolvePass{};
  resolvePass.color = {.loadOp = LoadOp::Clear,
                       .storeOp = StoreOp::Store,
                       .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  resolvePass.colorTexture = importHistoryWrite.value();
  resolvePass.dependencyTextures =
      std::span<const TextureHandle>(resolveReads.data(), resolveReadCount);
  resolvePass.draws = std::span<const DrawItem>(&resolveDraw, 1u);
  resolvePass.debugLabel = "TAA Resolve Pass";
  resolvePass.debugColor = kTaaResolvePassDebugColor;
  auto addResolve = ctx.graph.addGraphicsPass(resolvePass);
  if (addResolve.hasError()) {
    return Result<bool, std::string>::makeError(addResolve.error());
  }

  ++aaMetrics.taaResolvePassCount;
  if (!ctx.frame.camera.historyValid) {
    ++aaMetrics.taaCurrentFallbackFrameCount;
  }
  aaMetrics.taaHistoryBandwidthEstimateBytes +=
      textureStorageBytes(gpu_, ctx.shared.sceneColorTexture) +
      textureStorageBytes(gpu_, ctx.shared.historyColorWriteTexture) +
      (temporalHistoryValid
           ? textureStorageBytes(gpu_, ctx.shared.historyColorReadTexture) +
                 textureStorageBytes(gpu_, ctx.shared.sceneDepthTexture) +
                 textureStorageBytes(gpu_, ctx.shared.motionVectorTexture) +
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
  uint32_t displayMode = kTaaResolveModeCopyCurrent;
  std::string_view displayLabel = "TAA Copy Back Pass";
  uint32_t displayColor = kTaaCopyBackPassDebugColor;
  bool debugDisplay = false;

  if (debugView == AntiAliasingDebugView::TAACurrentColor) {
    ctx.shared.sceneColorGraphTexture = importScene.value();
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
    displayMode = taaResolveModeForDebugView(debugView);
    displayLabel = debugView == AntiAliasingDebugView::MotionVectors
                       ? "TAA Motion Vector Debug Pass"
                       : "TAA Velocity Debug Pass";
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  } else if (isTaaResolveEvaluationDebugView(debugView)) {
    if (debugView == AntiAliasingDebugView::TAAReactiveMask) {
      displaySource = ctx.shared.reactiveMaskTexture;
      displaySourceTexId = reactiveMaskTexId;
    } else {
      displaySource = ctx.shared.sceneColorTexture;
      displaySourceTexId = sceneTexId;
    }
    displayMode = taaResolveModeForDebugView(debugView);
    displayLabel = taaResolveEvaluationDebugLabel(debugView);
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  }

  const bool displayEvaluatesResolve =
      isTaaResolveEvaluationDebugView(debugView);
  const bool displaySamplesReactiveMask =
      debugView == AntiAliasingDebugView::TAAReactiveMask;
  const bool displaySamplesTemporalEvaluation =
      displayEvaluatesResolve && !displaySamplesReactiveMask;
  const TAAResolvePushConstants copyConstants{
      .currentTexId =
          displaySamplesTemporalEvaluation ? sceneTexId : displaySourceTexId,
      .historyTexId = historyReadTexId,
      .depthTexId = depthTexId,
      .velocityTexId = velocityTexId,
      .previousVelocityTexId = previousVelocityTexId,
      .reactiveMaskTexId = reactiveMaskTexId,
      .currentSamplerId = linearSamplerId,
      .historySamplerId = linearSamplerId,
      .depthSamplerId = pointSamplerId,
      .velocitySamplerId = pointSamplerId,
      .reactiveMaskSamplerId = pointSamplerId,
      .flags = resolveFlags,
      .mode = displayMode,
      .currentWeightBits = currentWeightBits,
      .inverseWidthBits = inverseWidthBits,
      .inverseHeightBits = inverseHeightBits,
      .depthThresholdBits = depthThresholdBits,
      .velocityThresholdBits = velocityThresholdBits,
      .velocityBlendScaleBits = velocityBlendScaleBits,
      .disocclusionWeightBits = disocclusionWeightBits,
      .clampAttenuationBits = clampAttenuationBits,
      .varianceGammaBits = varianceGammaBits,
      .hdrWeightStrengthBits = hdrWeightStrengthBits,
      .reactiveCurrentWeightBits = reactiveCurrentWeightBits,
      .reactiveStrengthBits = reactiveStrengthBits,
      .velocityDilationDepthThresholdBits = velocityDilationDepthThresholdBits,
      .clampMode = clampMode,
      .hdrWeightingMode = hdrWeightingMode,
      .velocityDilationMode = velocityDilationMode,
      .motionCurrentWeightBits = motionCurrentWeightBits,
      .clampCurrentWeightBits = clampCurrentWeightBits,
      .historyFilterMode = historyFilterMode,
  };
  const DrawItem copyDraw = makeFullscreenDraw(
      pipeline_,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&copyConstants),
          sizeof(copyConstants)),
      debugDisplay ? "TaaResolveDebug" : "TaaCopyBack");
  std::array<TextureHandle, 7> copyReads{};
  size_t copyReadCount = 0u;
  if (displaySamplesReactiveMask) {
    copyReads[copyReadCount++] = ctx.shared.reactiveMaskTexture;
  } else if (displaySamplesTemporalEvaluation) {
    copyReads[copyReadCount++] = ctx.shared.sceneColorTexture;
    if (temporalHistoryValid) {
      copyReads[copyReadCount++] = ctx.shared.historyColorReadTexture;
      copyReads[copyReadCount++] = ctx.shared.sceneDepthTexture;
      copyReads[copyReadCount++] = ctx.shared.motionVectorTexture;
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
    if (!nuri::isValid(ctx.shared.frameColorTexture)) {
      return Result<bool, std::string>::makeError(
          "TemporalAAResolvePass::build: missing TAA debug output texture");
    }

    const uint32_t debugOutputTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.frameColorTexture);
    if (debugOutputTexId == kInvalidTextureBindlessIndex) {
      return Result<bool, std::string>::makeError(
          "TemporalAAResolvePass::build: invalid TAA debug output texture");
    }

    auto importDebugOutput = ctx.graph.importTexture(
        ctx.shared.frameColorTexture, "taa_debug_output");
    if (importDebugOutput.hasError()) {
      return Result<bool, std::string>::makeError(importDebugOutput.error());
    }

    RenderGraphGraphicsPassDesc debugPass{};
    debugPass.color = {.loadOp = LoadOp::Clear,
                       .storeOp = StoreOp::Store,
                       .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    debugPass.colorTexture = importDebugOutput.value();
    debugPass.dependencyTextures =
        std::span<const TextureHandle>(copyReads.data(), copyReadCount);
    debugPass.draws = std::span<const DrawItem>(&copyDraw, 1u);
    debugPass.debugLabel = displayLabel;
    debugPass.debugColor = displayColor;
    auto addDebug = ctx.graph.addGraphicsPass(debugPass);
    if (addDebug.hasError()) {
      return Result<bool, std::string>::makeError(addDebug.error());
    }

    TAAResolvePushConstants debugCopyConstants = copyConstants;
    debugCopyConstants.currentTexId = debugOutputTexId;
    debugCopyConstants.mode = kTaaResolveModeCopyCurrent;
    const DrawItem debugCopyDraw = makeFullscreenDraw(
        pipeline_,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&debugCopyConstants),
            sizeof(debugCopyConstants)),
        "TaaDebugCopyBack");

    const std::array<TextureHandle, 1> debugCopyReads{
        ctx.shared.frameColorTexture};
    RenderGraphGraphicsPassDesc debugCopyPass{};
    debugCopyPass.color = {.loadOp = LoadOp::Clear,
                           .storeOp = StoreOp::Store,
                           .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    debugCopyPass.colorTexture = importScene.value();
    debugCopyPass.dependencyTextures = std::span<const TextureHandle>(
        debugCopyReads.data(), debugCopyReads.size());
    debugCopyPass.draws = std::span<const DrawItem>(&debugCopyDraw, 1u);
    debugCopyPass.debugLabel = "TAA Debug Copy Back Pass";
    debugCopyPass.debugColor = kTaaCopyBackPassDebugColor;
    auto addDebugCopy = ctx.graph.addGraphicsPass(debugCopyPass);
    if (addDebugCopy.hasError()) {
      return Result<bool, std::string>::makeError(addDebugCopy.error());
    }
  } else {
    RenderGraphGraphicsPassDesc copyPass{};
    copyPass.color = {.loadOp = LoadOp::Clear,
                      .storeOp = StoreOp::Store,
                      .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    copyPass.colorTexture = importScene.value();
    copyPass.dependencyTextures =
        std::span<const TextureHandle>(copyReads.data(), copyReadCount);
    copyPass.draws = std::span<const DrawItem>(&copyDraw, 1u);
    copyPass.debugLabel = displayLabel;
    copyPass.debugColor = displayColor;
    auto addCopy = ctx.graph.addGraphicsPass(copyPass);
    if (addCopy.hasError()) {
      return Result<bool, std::string>::makeError(addCopy.error());
    }
  }

  ctx.shared.sceneColorGraphTexture = importScene.value();
  aaMetrics.taaCopyBackPassCount += displaySamplesTemporalEvaluation ? 2u : 1u;
  aaMetrics.taaResolvedSceneColorPublished = !debugDisplay;
  aaMetrics.taaDebugViewRendered = debugDisplay;
  aaMetrics.taaHistoryValidityDebugViewRendered =
      debugView == AntiAliasingDebugView::TAAHistoryValidity;
  aaMetrics.taaPixelInspectorDebugViewRendered =
      debugView == AntiAliasingDebugView::TAAPixelInspector;
  aaMetrics.taaReactiveMaskDebugViewRendered =
      debugView == AntiAliasingDebugView::TAAReactiveMask;
  aaMetrics.taaDisocclusionMaskDebugViewRendered =
      debugView == AntiAliasingDebugView::TAADisocclusionMask;
  aaMetrics.taaVelocityDilationDebugViewRendered =
      debugView == AntiAliasingDebugView::TAAVelocityDilation;
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
           ? textureStorageBytes(gpu_, ctx.shared.frameColorTexture) * 2u
           : 0u) +
      (displaySamplesTemporalEvaluation && temporalHistoryValid
           ? textureStorageBytes(gpu_, ctx.shared.historyColorReadTexture) +
                 textureStorageBytes(gpu_, ctx.shared.sceneDepthTexture) +
                 textureStorageBytes(gpu_, ctx.shared.motionVectorTexture) +
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

TemporalAAFeature::TemporalAAFeature(GPUDevice &gpu,
                                     RuntimeCompositeConfig config)
    : resolvePass_(gpu, std::move(config)) {}

Result<bool, std::string>
TemporalAAFeature::publishFrameData(FrameBuildContext &ctx) {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  if (isTAAEnabled(settings)) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::MotionVectors |
        FrameTextureRequirementFlags::ReactiveMask;
  }
  return Result<bool, std::string>::makeResult(true);
}

std::span<RenderFeaturePass *const> TemporalAAFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
