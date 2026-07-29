#include "nuri/gfx/pipeline/features/reference_taa_feature.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/fullscreen.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include <bit>
namespace nuri {
namespace {
constexpr uint32_t kInvalidBindlessIndex = 0xffffffffu;
constexpr uint32_t kReferenceFlagHistoryValid = 1u << 0u;
constexpr uint32_t kReferenceFlagPreviousDepthValid = 1u << 1u;
constexpr uint32_t kReferenceFlagOrthographicProjection = 1u << 2u;
constexpr uint32_t kReferenceModeResolve = 0u;
constexpr uint32_t kReferenceModeCopy = 1u;
struct ReferenceTAAPushConstants {
  uint32_t currentTexId = 0u;
  uint32_t opaqueSceneTexId = 0u;
  uint32_t historyTexId = 0u;
  uint32_t depthTexId = 0u;
  uint32_t previousDepthTexId = 0u;
  uint32_t motionTexId = 0u;
  uint32_t motionClassTexId = 0u;
  uint32_t reactiveTexId = 0u;
  uint32_t linearSamplerId = 0u;
  uint32_t pointSamplerId = 0u;
  uint32_t flags = 0u;
  uint32_t mode = 0u;
  uint32_t inverseWidthBits = 0u;
  uint32_t inverseHeightBits = 0u;
  uint32_t nearPlaneBits = 0u;
  uint32_t farPlaneBits = 0u;
  uint32_t currentWeightBits = 0u;
  uint32_t sharpenStrengthBits = 0u;
};
static_assert(sizeof(ReferenceTAAPushConstants) <= 128u);
[[nodiscard]] std::filesystem::path
shaderBasePath(const RuntimeCompositeConfig &config) {
  if (!config.shaderBasePath.empty()) {
    return config.shaderBasePath;
  }
  return config.fullscreenVertex.parent_path();
}
[[nodiscard]] DrawItem
makeFullscreenDraw(RenderPipelineHandle pipeline,
                   const ReferenceTAAPushConstants &constants,
                   std::string_view label) {
  DrawItem draw{};
  draw.pipeline = pipeline;
  draw.vertexCount = 3u;
  draw.instanceCount = 1u;
  draw.pushConstants = std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(&constants), sizeof(constants));
  draw.debugLabel = label;
  draw.debugColor = 0xff55ccff;
  return draw;
}
[[nodiscard]] float referenceCurrentWeight(TemporalAAQualityPreset preset) {
  switch (preset) {
  case TemporalAAQualityPreset::Performance:
    return 0.16f;
  case TemporalAAQualityPreset::Balanced:
    return 0.12f;
  case TemporalAAQualityPreset::Quality:
    return 0.09f;
  case TemporalAAQualityPreset::Ultra:
    return 0.07f;
  case TemporalAAQualityPreset::Custom:
    return 0.10f;
  }
  return 0.10f;
}
} // namespace

ReferenceTAAResolvePass::ReferenceTAAResolvePass(GPUDevice &gpu,
                                                 RuntimeCompositeConfig config)
    : gpu_(gpu) {
  const std::filesystem::path base = shaderBasePath(config);
  vertexPath_ = config.fullscreenVertex.empty() ? base / "fullscreen_copy.vert"
                                                : config.fullscreenVertex;
  fragmentPath_ = base / "taa_reference.frag";
}

ReferenceTAAResolvePass::~ReferenceTAAResolvePass() {
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

bool ReferenceTAAResolvePass::isEnabled(const FrameBuildContext &ctx) const {
  return ctx.frame.presentationAA.reconstruction ==
         ColorReconstruction::ReferenceTAA;
}

Result<bool, std::string>
ReferenceTAAResolvePass::prepare(FrameBuildContext &) {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto vertexResult = compileShaderFile(
      gpu_, "taa_reference", vertexPath_.string(), ShaderStage::Vertex);
  if (vertexResult.hasError()) {
    return Result<bool, std::string>::makeError(vertexResult.error());
  }
  auto fragmentResult = compileShaderFile(
      gpu_, "taa_reference", fragmentPath_.string(), ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    gpu_.destroyShaderModule(vertexResult.value());
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }
  vertexShader_ = vertexResult.value();
  fragmentShader_ = fragmentResult.value();
  const SamplerDesc samplerDesc{
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
  auto samplerResult = gpu_.createSampler(samplerDesc, "taa_reference_linear");
  if (samplerResult.hasError()) {
    return Result<bool, std::string>::makeError(samplerResult.error());
  }
  linearClampSampler_ = samplerResult.value();
  auto pipelineResult = gpu_.createRenderPipeline(
      fullscreenPipelineDesc(kFrameCompositionFrameColorFormat, vertexShader_,
                             fragmentShader_),
      "taa_reference");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  pipeline_ = pipelineResult.value();
  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ReferenceTAAResolvePass::build(FrameBuildContext &ctx) {
  if (!isEnabled(ctx) || !nuri::isValid(pipeline_) ||
      !nuri::isValid(ctx.shared[FrameTextureSlot::FrameColor].texture) ||
      !nuri::isValid(ctx.shared[FrameTextureSlot::SceneColor].texture) ||
      !nuri::isValid(ctx.shared[FrameHistoryTextureSlot::ColorRead].texture) ||
      !nuri::isValid(ctx.shared[FrameHistoryTextureSlot::ColorWrite].texture) ||
      !nuri::isValid(ctx.shared[FrameTextureSlot::SceneDepth].texture) ||
      !nuri::isValid(ctx.shared[FrameTextureSlot::MotionVector].texture) ||
      !nuri::isValid(ctx.shared[FrameTextureSlot::MotionClass].texture) ||
      !nuri::isValid(ctx.shared[FrameTextureSlot::ReactiveMask].texture)) {
    return Result<bool, std::string>::makeResult(false);
  }
  const bool historyValid =
      ctx.frame.camera.historyValid && ctx.shared.historyColorReadValid;
  const bool previousDepthValid =
      historyValid &&
      nuri::isValid(
          ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture);
  const TextureHandle previousDepth =
      previousDepthValid
          ? ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture
          : ctx.shared[FrameTextureSlot::SceneDepth].texture;
  const uint32_t currentTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::FrameColor].texture);
  const uint32_t opaqueSceneTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::SceneColor].texture);
  const uint32_t historyTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameHistoryTextureSlot::ColorRead].texture);
  const uint32_t depthTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::SceneDepth].texture);
  const uint32_t previousDepthTexId =
      gpu_.getTextureBindlessIndex(previousDepth);
  const uint32_t motionTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::MotionVector].texture);
  const uint32_t motionClassTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::MotionClass].texture);
  const uint32_t reactiveTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::ReactiveMask].texture);
  const uint32_t linearSamplerId =
      gpu_.getSamplerBindlessIndex(linearClampSampler_);
  const uint32_t pointSamplerId = gpu_.getDefaultSamplerBindlessIndex();
  if (currentTexId == kInvalidBindlessIndex ||
      opaqueSceneTexId == kInvalidBindlessIndex ||
      historyTexId == kInvalidBindlessIndex ||
      depthTexId == kInvalidBindlessIndex ||
      previousDepthTexId == kInvalidBindlessIndex ||
      motionTexId == kInvalidBindlessIndex ||
      motionClassTexId == kInvalidBindlessIndex ||
      reactiveTexId == kInvalidBindlessIndex ||
      linearSamplerId == kInvalidBindlessIndex ||
      pointSamplerId == kInvalidBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "ReferenceTAAResolvePass::build: invalid bindless resource");
  }
  auto historyOutput = ctx.graph.importTexture(
      ctx.shared[FrameHistoryTextureSlot::ColorWrite].texture,
      "reference_taa_history_write");
  if (historyOutput.hasError()) {
    return Result<bool, std::string>::makeError(historyOutput.error());
  }
  auto frameOutput =
      ctx.graph.importTexture(ctx.shared[FrameTextureSlot::FrameColor].texture,
                              "reference_taa_frame_output");
  if (frameOutput.hasError()) {
    return Result<bool, std::string>::makeError(frameOutput.error());
  }
  const TextureDimensions dimensions = gpu_.getTextureDimensions(
      ctx.shared[FrameTextureSlot::FrameColor].texture);
  const float inverseWidth =
      1.0f / static_cast<float>(std::max(dimensions.width, 1u));
  const float inverseHeight =
      1.0f / static_cast<float>(std::max(dimensions.height, 1u));
  const RenderSettings &settings = ctx.frame.settings;
  const TemporalAATuning &tuning = settings.antiAliasing.temporalTuning;
  const float sharpenStrength =
      tuning.sharpenEnabled ? tuning.sharpenStrength : 0.0f;
  ReferenceTAAPushConstants constants{
      .currentTexId = currentTexId,
      .opaqueSceneTexId = opaqueSceneTexId,
      .historyTexId = historyTexId,
      .depthTexId = depthTexId,
      .previousDepthTexId = previousDepthTexId,
      .motionTexId = motionTexId,
      .motionClassTexId = motionClassTexId,
      .reactiveTexId = reactiveTexId,
      .linearSamplerId = linearSamplerId,
      .pointSamplerId = pointSamplerId,
      .flags = (historyValid ? kReferenceFlagHistoryValid : 0u) |
               (previousDepthValid ? kReferenceFlagPreviousDepthValid : 0u) |
               (ctx.frame.camera.projectionType == ProjectionType::Orthographic
                    ? kReferenceFlagOrthographicProjection
                    : 0u),
      .mode = kReferenceModeResolve,
      .inverseWidthBits = std::bit_cast<uint32_t>(inverseWidth),
      .inverseHeightBits = std::bit_cast<uint32_t>(inverseHeight),
      .nearPlaneBits = std::bit_cast<uint32_t>(ctx.frame.camera.nearPlane),
      .farPlaneBits = std::bit_cast<uint32_t>(ctx.frame.camera.farPlane),
      .currentWeightBits = std::bit_cast<uint32_t>(
          referenceCurrentWeight(settings.antiAliasing.qualityPreset)),
      .sharpenStrengthBits = std::bit_cast<uint32_t>(sharpenStrength),
  };
  const DrawItem resolveDraw =
      makeFullscreenDraw(pipeline_, constants, "ReferenceTaaResolve");
  const std::array<TextureHandle, 8> resolveReads{
      ctx.shared[FrameTextureSlot::FrameColor].texture,
      ctx.shared[FrameTextureSlot::SceneColor].texture,
      ctx.shared[FrameHistoryTextureSlot::ColorRead].texture,
      ctx.shared[FrameTextureSlot::SceneDepth].texture,
      previousDepth,
      ctx.shared[FrameTextureSlot::MotionVector].texture,
      ctx.shared[FrameTextureSlot::MotionClass].texture,
      ctx.shared[FrameTextureSlot::ReactiveMask].texture,
  };
  RenderGraphGraphicsPassDesc resolvePass{};
  resolvePass.color = {.loadOp = LoadOp::Clear,
                       .storeOp = StoreOp::Store,
                       .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  resolvePass.colorTexture = historyOutput.value();
  resolvePass.dependencyTextures = resolveReads;
  resolvePass.draws = std::span<const DrawItem>(&resolveDraw, 1u);
  resolvePass.gpuTimingScope = GpuTimingScope::TemporalAAResolve;
  resolvePass.debugLabel = "Reference TAA Resolve Pass";
  resolvePass.debugColor = 0xff44bbff;
  auto addResolve = ctx.graph.addGraphicsPass(resolvePass);
  if (addResolve.hasError()) {
    return Result<bool, std::string>::makeError(addResolve.error());
  }
  constants.currentTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameHistoryTextureSlot::ColorWrite].texture);
  constants.mode = kReferenceModeCopy;
  const DrawItem copyDraw =
      makeFullscreenDraw(pipeline_, constants, "ReferenceTaaCopy");
  const std::array<TextureHandle, 1> copyReads{
      ctx.shared[FrameHistoryTextureSlot::ColorWrite].texture};
  RenderGraphGraphicsPassDesc copyPass{};
  copyPass.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  copyPass.colorTexture = frameOutput.value();
  copyPass.dependencyTextures = copyReads;
  copyPass.draws = std::span<const DrawItem>(&copyDraw, 1u);
  copyPass.gpuTimingScope = GpuTimingScope::TemporalAACopyBack;
  copyPass.debugLabel = "Reference TAA Copy Back Pass";
  copyPass.debugColor = 0xff4499dd;
  auto addCopy = ctx.graph.addGraphicsPass(copyPass);
  if (addCopy.hasError()) {
    return Result<bool, std::string>::makeError(addCopy.error());
  }
  ctx.shared[FrameTextureSlot::FrameColor].graph = frameOutput.value();
  ctx.frame.sharedResources[FrameTextureSlot::FrameColor].graph =
      frameOutput.value();
  ctx.shared.historyWriteRequirements |=
      FrameTextureRequirementFlags::HistoryColor;
  AntiAliasingFrameMetrics &metrics = ctx.frame.metrics.antiAliasing;
  ++metrics.taaResolvePassCount;
  ++metrics.taaCopyBackPassCount;
  metrics.taaResolveWidth = dimensions.width;
  metrics.taaResolveHeight = dimensions.height;
  metrics.taaCurrentFrameWeight =
      referenceCurrentWeight(settings.antiAliasing.qualityPreset);
  metrics.taaHistoryFrameWeight = 1.0f - metrics.taaCurrentFrameWeight;
  metrics.taaHistoryValidPercent = historyValid ? 100.0f : 0.0f;
  metrics.taaDepthRejectionEnabled = previousDepthValid;
  metrics.taaNeighborhoodClampEnabled = historyValid;
  metrics.taaAdaptiveBlendEnabled = historyValid;
  metrics.taaSharpenEnabled = tuning.sharpenEnabled;
  metrics.taaSharpenActive = historyValid && sharpenStrength > 0.0f;
  metrics.taaSharpenStrength = sharpenStrength;
  metrics.taaSharpenConfidenceThreshold = tuning.sharpenConfidenceThreshold;
  metrics.taaResolvedSceneColorPublished = true;
  if (!historyValid) {
    ++metrics.taaCurrentFallbackFrameCount;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ReferenceTAAResolvePass::publishFrameData(FrameBuildContext &ctx) {
  if (ctx.frame.presentationAA.reconstruction ==
      ColorReconstruction::ReferenceTAA) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::HistoryColor |
        FrameTextureRequirementFlags::MotionVectors |
        FrameTextureRequirementFlags::ReactiveMask |
        FrameTextureRequirementFlags::MotionClass;
  }
  return Result<bool, std::string>::makeResult(true);
}

void registerReferenceTAAStage(RenderPipeline &pipeline, GPUDevice &gpu,
                               RuntimeCompositeConfig config) {
  pipeline.addStage(
      std::make_unique<ReferenceTAAResolvePass>(gpu, std::move(config)),
      "ReferenceTAAFeature", "ReferenceTAAResolvePass", false,
      PipelineComponentDesc{.publish = [](void *state, FrameBuildContext &ctx) {
        return static_cast<ReferenceTAAResolvePass *>(state)->publishFrameData(
            ctx);
      }});
}

} // namespace nuri
