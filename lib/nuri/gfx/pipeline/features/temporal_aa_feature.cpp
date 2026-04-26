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
constexpr uint32_t kTaaResolveModeResolve = 0u;
constexpr uint32_t kTaaResolveModeCopyCurrent = 1u;
constexpr uint32_t kTaaResolveModePreviousHistory = 2u;
constexpr uint32_t kTaaResolveModeHistoryValidity = 3u;
constexpr uint32_t kTaaResolvePassDebugColor = 0xffaa55ffu;
constexpr uint32_t kTaaCopyBackPassDebugColor = 0xff8844ffu;
constexpr uint32_t kTaaDebugPassDebugColor = 0xffcc77ffu;
constexpr uint32_t kDrawDebugColor = 0xffaa44ffu;
constexpr uint32_t kInvalidTextureBindlessIndex = 0xFFFFFFFFu;
constexpr float kDefaultTaaCurrentFrameWeight = 0.10f;

struct TAAResolvePushConstants {
  uint32_t currentTexId = 0u;
  uint32_t historyTexId = 0u;
  uint32_t depthTexId = 0u;
  uint32_t velocityTexId = 0u;
  uint32_t currentSamplerId = 0u;
  uint32_t historySamplerId = 0u;
  uint32_t depthSamplerId = 0u;
  uint32_t velocitySamplerId = 0u;
  uint32_t flags = 0u;
  uint32_t mode = 0u;
  uint32_t currentWeightBits = 0u;
  uint32_t reserved0 = 0u;
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
taaViewNeedsResolve(AntiAliasingDebugView view) noexcept {
  return view == AntiAliasingDebugView::None ||
         view == AntiAliasingDebugView::Settings ||
         view == AntiAliasingDebugView::TAAResolved;
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
  if (nuri::isValid(vertexShader_)) {
    gpu_.destroyShaderModule(vertexShader_);
  }
  if (nuri::isValid(fragmentShader_)) {
    gpu_.destroyShaderModule(fragmentShader_);
  }
}

bool TemporalAAResolvePass::isEnabled(const FrameBuildContext &ctx) const {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  if (!isTAAEnabled(settings)) {
    return false;
  }
  const AntiAliasingDebugView debugView =
      sanitizeAntiAliasingDebugView(settings.antiAliasing.debug.view);
  return !isVelocityDebugView(debugView);
}

Result<bool, std::string>
TemporalAAResolvePass::prepare(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(false);
  }

  const AntiAliasingDebugView debugView = sanitizeAntiAliasingDebugView(
      renderSettingsOrDefault(ctx.frame).antiAliasing.debug.view);
  if (debugView == AntiAliasingDebugView::TAACurrentColor) {
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
  if (debugView == AntiAliasingDebugView::TAACurrentColor) {
    AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
    if (nuri::isValid(ctx.shared.sceneColorTexture)) {
      const TextureDimensions sceneDimensions =
          gpu_.getTextureDimensions(ctx.shared.sceneColorTexture);
      aaMetrics.taaResolveWidth = sceneDimensions.width;
      aaMetrics.taaResolveHeight = sceneDimensions.height;
    }
    const float currentWeight = sanitizedCurrentWeight(settings.antiAliasing);
    aaMetrics.taaCurrentFrameWeight = currentWeight;
    aaMetrics.taaHistoryFrameWeight = 1.0f - currentWeight;
    aaMetrics.taaDebugViewRendered = true;
    return Result<bool, std::string>::makeResult(false);
  }

  if (!nuri::isValid(pipeline_) ||
      !nuri::isValid(ctx.shared.sceneColorTexture) ||
      !nuri::isValid(ctx.shared.historyColorReadTexture) ||
      !nuri::isValid(ctx.shared.historyColorWriteTexture) ||
      !nuri::isValid(ctx.shared.sceneDepthTexture) ||
      !nuri::isValid(ctx.shared.motionVectorTexture)) {
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
  const uint32_t linearSamplerId =
      gpu_.getLinearRepeatSamplerBindlessIndex(false, 1u);
  const uint32_t pointSamplerId = gpu_.getDefaultSamplerBindlessIndex();
  if (sceneTexId == kInvalidTextureBindlessIndex ||
      historyReadTexId == kInvalidTextureBindlessIndex ||
      historyWriteTexId == kInvalidTextureBindlessIndex ||
      depthTexId == kInvalidTextureBindlessIndex ||
      velocityTexId == kInvalidTextureBindlessIndex ||
      linearSamplerId == kInvalidTextureBindlessIndex ||
      pointSamplerId == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "TemporalAAResolvePass::build: invalid bindless texture or sampler");
  }

  const uint32_t historyFlag =
      ctx.frame.camera.historyValid ? kTaaResolveFlagHistoryValid : 0u;
  const float currentWeight = sanitizedCurrentWeight(settings.antiAliasing);
  const uint32_t currentWeightBits = std::bit_cast<uint32_t>(currentWeight);

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

  const std::array<TextureHandle, 4> resolveReads{
      ctx.shared.sceneColorTexture, ctx.shared.historyColorReadTexture,
      ctx.shared.sceneDepthTexture, ctx.shared.motionVectorTexture};

  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  const TextureDimensions sceneDimensions =
      gpu_.getTextureDimensions(ctx.shared.sceneColorTexture);
  aaMetrics.taaResolveWidth = sceneDimensions.width;
  aaMetrics.taaResolveHeight = sceneDimensions.height;
  aaMetrics.taaCurrentFrameWeight = currentWeight;
  aaMetrics.taaHistoryFrameWeight = 1.0f - currentWeight;
  aaMetrics.taaHistoryValidPercent =
      ctx.frame.camera.historyValid ? 100.0f : 0.0f;
  aaMetrics.taaOutOfBoundsFallbackEnabled = true;
  aaMetrics.taaBilinearHistorySampling = true;

  const bool needsResolve = taaViewNeedsResolve(debugView);
  if (needsResolve) {
    const TAAResolvePushConstants resolveConstants{
        .currentTexId = sceneTexId,
        .historyTexId = historyReadTexId,
        .depthTexId = depthTexId,
        .velocityTexId = velocityTexId,
        .currentSamplerId = linearSamplerId,
        .historySamplerId = linearSamplerId,
        .depthSamplerId = pointSamplerId,
        .velocitySamplerId = pointSamplerId,
        .flags = historyFlag,
        .mode = kTaaResolveModeResolve,
        .currentWeightBits = currentWeightBits,
        .reserved0 = 0u,
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
    resolvePass.dependencyTextures = std::span<const TextureHandle>(
        resolveReads.data(), resolveReads.size());
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
        textureStorageBytes(gpu_, ctx.shared.historyColorReadTexture) +
        textureStorageBytes(gpu_, ctx.shared.historyColorWriteTexture) +
        textureStorageBytes(gpu_, ctx.shared.sceneDepthTexture) +
        textureStorageBytes(gpu_, ctx.shared.motionVectorTexture);
  }

  TextureHandle displaySource = ctx.shared.historyColorWriteTexture;
  uint32_t displaySourceTexId = historyWriteTexId;
  uint32_t displayMode = kTaaResolveModeCopyCurrent;
  std::string_view displayLabel = "TAA Copy Back Pass";
  uint32_t displayColor = kTaaCopyBackPassDebugColor;
  bool debugDisplay = false;

  if (debugView == AntiAliasingDebugView::TAAPreviousHistory) {
    displaySource = ctx.shared.historyColorReadTexture;
    displaySourceTexId = historyReadTexId;
    displayMode = kTaaResolveModePreviousHistory;
    displayLabel = "TAA Previous History Debug Pass";
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  } else if (debugView == AntiAliasingDebugView::TAAHistoryValidity) {
    displaySource = ctx.shared.historyColorReadTexture;
    displaySourceTexId = historyReadTexId;
    displayMode = kTaaResolveModeHistoryValidity;
    displayLabel = "TAA History Validity Debug Pass";
    displayColor = kTaaDebugPassDebugColor;
    debugDisplay = true;
  }

  const TAAResolvePushConstants copyConstants{
      .currentTexId = displaySourceTexId,
      .historyTexId = historyReadTexId,
      .depthTexId = depthTexId,
      .velocityTexId = velocityTexId,
      .currentSamplerId = linearSamplerId,
      .historySamplerId = linearSamplerId,
      .depthSamplerId = pointSamplerId,
      .velocitySamplerId = pointSamplerId,
      .flags = historyFlag,
      .mode = displayMode,
      .currentWeightBits = currentWeightBits,
      .reserved0 = 0u,
  };
  const DrawItem copyDraw = makeFullscreenDraw(
      pipeline_,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&copyConstants),
          sizeof(copyConstants)),
      debugDisplay ? "TaaResolveDebug" : "TaaCopyBack");
  std::array<TextureHandle, 3> copyReads{};
  size_t copyReadCount = 0u;
  copyReads[copyReadCount++] = displaySource;
  if (displayMode == kTaaResolveModeHistoryValidity) {
    copyReads[copyReadCount++] = ctx.shared.sceneDepthTexture;
    copyReads[copyReadCount++] = ctx.shared.motionVectorTexture;
  }

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

  ctx.shared.sceneColorGraphTexture = importScene.value();
  ++aaMetrics.taaCopyBackPassCount;
  aaMetrics.taaResolvedSceneColorPublished = !debugDisplay;
  aaMetrics.taaDebugViewRendered = debugDisplay;
  aaMetrics.taaHistoryValidityDebugViewRendered =
      debugView == AntiAliasingDebugView::TAAHistoryValidity;
  aaMetrics.taaHistoryBandwidthEstimateBytes +=
      textureStorageBytes(gpu_, displaySource) +
      textureStorageBytes(gpu_, ctx.shared.sceneColorTexture);

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
        FrameTextureRequirementFlags::MotionVectors;
  }
  return Result<bool, std::string>::makeResult(true);
}

std::span<RenderFeaturePass *const> TemporalAAFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
