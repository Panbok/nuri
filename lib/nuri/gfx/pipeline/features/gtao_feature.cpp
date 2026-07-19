#include "nuri/gfx/pipeline/features/gtao_feature.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/render_capture.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/pch.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
namespace nuri {
namespace {
enum GtaoStage : size_t {
  DepthPrefilter,
  Edge,
  Main,
  Denoise,
  Temporal,
  StageCount
};
enum GtaoSampler : size_t { PointClamp, LinearClamp, SamplerCount };
struct GtaoShaderSpec {
  std::string_view name;
  std::string_view file;
};
constexpr std::array<GtaoShaderSpec, StageCount> kGtaoShaderSpecs{{
    {"gtao_depth_prefilter", "gtao_depth_prefilter.comp"},
    {"gtao_edges", "gtao_edges.comp"},
    {"gtao_main", "gtao_main.comp"},
    {"gtao_denoise", "gtao_denoise.comp"},
    {"gtao_temporal", "gtao_temporal.comp"},
}};
constexpr uint32_t kGTAOWorkgroupSizeX = 8u;
constexpr uint32_t kGTAOWorkgroupSizeY = 8u;
constexpr uint32_t kGTAODebugColor = 0xff44ddaa;
constexpr uint32_t kTemporalFlagsDefault = 1u;
constexpr float kTemporalBaseCurrentWeight = 0.20f;
constexpr float kTemporalRejectedCurrentWeight = 0.65f;
struct DepthPrefilterPushConstants {
  uint32_t sourceDepthTexId = kInvalidTextureBindlessIndex;
  std::array<uint32_t, GTAOPass::kViewDepthMipCount> outputTexIds{};
  uint32_t width = 1u;
  uint32_t height = 1u;
  uint32_t projectionType = 0u;
  uint32_t radiusBits = 0u;
  float nearPlane = 0.01f;
  float farPlane = 1000.0f;
};
static_assert(sizeof(DepthPrefilterPushConstants) <= 128u);
struct EdgePushConstants {
  uint32_t depthTexId = kInvalidTextureBindlessIndex;
  uint32_t outputTexId = kInvalidTextureBindlessIndex;
  uint32_t width = 1u;
  uint32_t height = 1u;
};
static_assert(sizeof(EdgePushConstants) <= 128u);
struct MainPushConstants {
  std::array<uint32_t, GTAOPass::kViewDepthMipCount> depthTexIds{};
  uint32_t normalTexId = kInvalidTextureBindlessIndex;
  uint32_t edgeTexId = kInvalidTextureBindlessIndex;
  uint32_t outputTexId = kInvalidTextureBindlessIndex;
  uint32_t width = 1u;
  uint32_t height = 1u;
  uint32_t noiseIndex = 0u;
  uint32_t sliceCount = 2u;
  uint32_t stepCount = 4u;
  uint32_t strengthBits = 0u;
  uint32_t radiusBits = 0u;
  float tanHalfFovY = 1.0f;
  float aspectRatio = 1.0f;
  float orthoHeight = 10.0f;
  uint32_t projectionType = 0u;
};
static_assert(sizeof(MainPushConstants) <= 128u);
struct DenoisePushConstants {
  uint32_t sourceTexId = kInvalidTextureBindlessIndex;
  uint32_t outputTexId = kInvalidTextureBindlessIndex;
  uint32_t edgeTexId = kInvalidTextureBindlessIndex;
  uint32_t width = 1u;
  uint32_t height = 1u;
};
static_assert(sizeof(DenoisePushConstants) <= 128u);
struct TemporalPushConstants {
  uint32_t currentTexId = kInvalidTextureBindlessIndex;
  uint32_t historyTexId = kInvalidTextureBindlessIndex;
  uint32_t motionVectorTexId = kInvalidTextureBindlessIndex;
  uint32_t motionClassTexId = kInvalidTextureBindlessIndex;
  uint32_t sceneDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t currentViewDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t previousSceneDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t edgeTexId = kInvalidTextureBindlessIndex;
  uint32_t outputTexId = kInvalidTextureBindlessIndex;
  uint32_t pointSamplerId = 0u;
  uint32_t historySamplerId = 0u;
  uint32_t width = 1u;
  uint32_t height = 1u;
  uint32_t flags = 0u;
  uint32_t baseCurrentWeightBits = 0u;
  uint32_t rejectedCurrentWeightBits = 0u;
  uint32_t projectionType = 0u;
  float nearPlane = 0.01f;
  float farPlane = 1000.0f;
};
static_assert(sizeof(TemporalPushConstants) <= 128u);
template <std::size_t Size, typename T>
std::span<const std::byte> copyPushConstants(std::array<std::byte, Size> &dst,
                                             const T &src) {
  static_assert(sizeof(T) <= Size);
  std::memcpy(dst.data(), &src, sizeof(T));
  return std::span<const std::byte>(dst.data(), sizeof(T));
}
[[nodiscard]] uint32_t divRoundUp(uint32_t value, uint32_t divisor) noexcept {
  return (value + divisor - 1u) / divisor;
}
[[nodiscard]] uint32_t levelDimension(uint32_t value, uint32_t level) noexcept {
  return std::max(1u, value >> std::min(level, 31u));
}
[[nodiscard]] uint64_t textureBytes(Format format, uint32_t width,
                                    uint32_t height) noexcept {
  return static_cast<uint64_t>(std::max(width, 1u)) *
         static_cast<uint64_t>(std::max(height, 1u)) *
         std::max(formatTexelBytes(format), 1u);
}
[[nodiscard]] TextureDesc
makeStorageSampledTextureDesc(Format format, uint32_t width, uint32_t height) {
  return TextureDesc{
      .type = TextureType::Texture2D,
      .format = format,
      .dimensions = {.width = std::max(width, 1u),
                     .height = std::max(height, 1u),
                     .depth = 1u},
      .usage = TextureUsage::StorageSampled,
      .storage = Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = 1u,
      .data = {},
      .dataNumMipLevels = 1u,
      .generateMipmaps = false,
  };
}
[[nodiscard]] std::filesystem::path
resolveShaderDir(const RuntimeOpaqueShaderConfig &config) {
  if (!config.meshFragment.empty()) {
    return config.meshFragment.parent_path();
  }
  if (!config.meshVertex.empty()) {
    return config.meshVertex.parent_path();
  }
  return {};
}
[[nodiscard]] RenderSettings::AmbientOcclusionSettings
resolvedAmbientOcclusionSettings(const RenderFrameContext &frame) {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  RenderSettings::AmbientOcclusionSettings ao = settings.ambientOcclusion;
  sanitizeAmbientOcclusionSettings(ao, settings.opaque, settings.antiAliasing);
  return ao;
}
[[nodiscard]] float
ambientOcclusionPresetRadius(AmbientOcclusionPreset preset) noexcept {
  switch (sanitizeAmbientOcclusionPreset(preset)) {
  case AmbientOcclusionPreset::Low:
    return 0.75f;
  case AmbientOcclusionPreset::Balanced:
    return 1.15f;
  case AmbientOcclusionPreset::High:
    return 1.6f;
  case AmbientOcclusionPreset::Ultra:
    return 2.2f;
  case AmbientOcclusionPreset::Custom:
    return 1.15f;
  }
  return 1.15f;
}
[[nodiscard]] uint64_t mixSignature(uint64_t seed, uint64_t value) noexcept {
  seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
  return seed;
}
[[nodiscard]] uint64_t ambientOcclusionTemporalPolicySignature(
    const RenderSettings::AmbientOcclusionSettings &ao) noexcept {
  uint64_t signature = 0xcbf29ce484222325ull;
  signature = mixSignature(signature, static_cast<uint32_t>(ao.mode));
  signature = mixSignature(signature, static_cast<uint32_t>(ao.preset));
  signature = mixSignature(signature, std::bit_cast<uint32_t>(ao.strength));
  signature = mixSignature(signature, ao.temporalAccumulation ? 1u : 0u);
  signature = mixSignature(signature, ao.sliceCount);
  signature = mixSignature(signature, ao.stepCount);
  signature = mixSignature(signature, ao.denoisePassCount);
  return signature;
}
} // namespace

GTAOPass::GTAOPass(GPUDevice &gpu, RuntimeOpaqueShaderConfig config)
    : gpu_(gpu), config_(std::move(config)) {
  auto result = initialize();
  if (result.hasError())
    initializationError_ = result.error();
}

GTAOPass::~GTAOPass() { destroyResources(); }

void GTAOPass::observeTemporalPolicy(
    const RenderSettings::AmbientOcclusionSettings &ao) noexcept {
  const uint64_t temporalPolicySignature =
      ambientOcclusionTemporalPolicySignature(ao);
  temporalPolicyChanged_ =
      hasLastTemporalPolicySignature_ &&
      lastTemporalPolicySignature_ != temporalPolicySignature;
  hasLastTemporalPolicySignature_ = true;
  lastTemporalPolicySignature_ = temporalPolicySignature;
}

bool GTAOPass::isEnabled(const FrameBuildContext &ctx) const {
  return resolvedAmbientOcclusionSettings(ctx.frame).active;
}

Result<bool, std::string> GTAOPass::prepare(FrameBuildContext &ctx) {
  (void)ctx;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GTAOPass::initialize() {
  for (size_t index = 0; index < samplers_.size(); ++index) {
    const SamplerFilter filter =
        index == PointClamp ? SamplerFilter::Nearest : SamplerFilter::Linear;
    auto result = gpu_.createSampler(
        SamplerDesc{.minFilter = filter,
                    .magFilter = filter,
                    .mipMode = SamplerMipMode::Disabled,
                    .wrapU = SamplerWrapMode::Clamp,
                    .wrapV = SamplerWrapMode::Clamp,
                    .wrapW = SamplerWrapMode::Clamp},
        index == PointClamp ? "gtao_point_clamp" : "gtao_linear_clamp");
    if (result.hasError())
      return Result<bool, std::string>::makeError(result.error());
    samplers_[index] = result.value();
  }
  const std::filesystem::path shaderDir = resolveShaderDir(config_);
  for (size_t index = 0; index < kGtaoShaderSpecs.size(); ++index) {
    const GtaoShaderSpec &spec = kGtaoShaderSpecs[index];
    auto shader = Shader::create(spec.name, gpu_);
    auto compiled = shader->compileFromFile((shaderDir / spec.file).string(),
                                            ShaderStage::Compute);
    if (compiled.hasError())
      return Result<bool, std::string>::makeError(compiled.error());
    shaders_[index] = compiled.value();
    auto pipeline = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = shaders_[index]}, spec.name);
    if (pipeline.hasError())
      return Result<bool, std::string>::makeError(pipeline.error());
    pipelines_[index] = pipeline.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
GTAOPass::ensureScratchTextures(FrameBuildContext &ctx) {
  if (!initializationError_.empty())
    return Result<bool, std::string>::makeError(initializationError_);
  const TextureDimensions sceneDimensions =
      gpu_.getTextureDimensions(ctx.shared.sceneDepthTexture);
  const uint32_t width = std::max(sceneDimensions.width, 1u);
  const uint32_t height = std::max(sceneDimensions.height, 1u);
  const uint32_t ringCount = std::max(1u, gpu_.getSwapchainImageCount());
  if (scratchWidth_ != width || scratchHeight_ != height ||
      scratchRingCount_ != ringCount || scratchTextures_.empty()) {
    auto recreateResult = recreateScratchTextures(width, height, ringCount);
    if (recreateResult.hasError()) {
      return recreateResult;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
GTAOPass::recreateScratchTextures(uint32_t width, uint32_t height,
                                  uint32_t ringCount) {
  destroyScratchTextures();
  scratchTextures_.resize(ringCount);
  for (uint32_t slot = 0u; slot < ringCount; ++slot) {
    FrameScratchTextures &textures = scratchTextures_[slot];
    const auto create = [&](size_t index, const TextureDesc &desc,
                            std::string name) -> Result<bool, std::string> {
      auto result = gpu_.createTexture(desc, name);
      if (result.hasError()) {
        destroyScratchTextures();
        return Result<bool, std::string>::makeError(result.error());
      }
      textures.textures[index] = result.value();
      return Result<bool, std::string>::makeResult(true);
    };
    for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
      auto result = create(level,
                           makeStorageSampledTextureDesc(
                               Format::R32_FLOAT, levelDimension(width, level),
                               levelDimension(height, level)),
                           "gtao_view_depth_" + std::to_string(slot) + "_" +
                               std::to_string(level));
      if (result.hasError())
        return result;
    }
    constexpr std::array formats{Format::R8_UNORM,
                                 kFrameCompositionAmbientOcclusionFormat,
                                 kFrameCompositionAmbientOcclusionFormat};
    constexpr std::array names{"gtao_edges_", "gtao_raw_ao_",
                               "gtao_denoise_scratch_"};
    for (size_t index = 0u; index < formats.size(); ++index) {
      auto result =
          create(kScratchEdges + index,
                 makeStorageSampledTextureDesc(formats[index], width, height),
                 names[index] + std::to_string(slot));
      if (result.hasError())
        return result;
    }
  }
  scratchWidth_ = width;
  scratchHeight_ = height;
  scratchRingCount_ = ringCount;
  return Result<bool, std::string>::makeResult(true);
}

void GTAOPass::destroyResources() {
  destroyScratchTextures();
  for (ComputePipelineHandle &pipeline : pipelines_)
    if (nuri::isValid(pipeline))
      gpu_.destroyComputePipeline(pipeline);
  for (SamplerHandle &sampler : samplers_)
    if (nuri::isValid(sampler))
      gpu_.destroySampler(sampler);
  pipelines_.fill({});
  samplers_.fill({});
  shaders_.fill({});
}

void GTAOPass::destroyScratchTextures() {
  for (FrameScratchTextures &textures : scratchTextures_) {
    for (TextureHandle &texture : textures.textures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
        texture = {};
      }
    }
  }
  scratchTextures_.clear();
  scratchWidth_ = 0u;
  scratchHeight_ = 0u;
  scratchRingCount_ = 0u;
}

GTAOPass::FrameScratchTextures &
GTAOPass::currentScratch(uint64_t frameIndex) noexcept {
  return scratchTextures_[static_cast<size_t>(frameIndex %
                                              scratchTextures_.size())];
}

Result<bool, std::string> GTAOPass::build(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  auto resourceResult = ensureScratchTextures(ctx);
  if (resourceResult.hasError()) {
    ctx.frame.metrics.ambientOcclusion.active = false;
    ctx.frame.metrics.ambientOcclusion.disabledReason =
        AmbientOcclusionDisabledReason::Unsupported;
    return resourceResult;
  }
  FrameScratchTextures &scratch = currentScratch(ctx.frame.frameIndex);
  const RenderSettings::AmbientOcclusionSettings ao =
      resolvedAmbientOcclusionSettings(ctx.frame);
  const TextureDimensions dimensions =
      gpu_.getTextureDimensions(ctx.shared.sceneDepthTexture);
  const uint32_t width = std::max(dimensions.width, 1u);
  const uint32_t height = std::max(dimensions.height, 1u);
  const DispatchSize dispatch{.x = divRoundUp(width, kGTAOWorkgroupSizeX),
                              .y = divRoundUp(height, kGTAOWorkgroupSizeY),
                              .z = 1u};
  AmbientOcclusionFrameMetrics &metrics = ctx.frame.metrics.ambientOcclusion;
  const GpuTimingReport &timingReport = ctx.frame.gpuTiming;
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAO)) {
    metrics.gpuTimeMs = timingReport.gtaoTimeMs;
    metrics.gpuTimingSourceFrameIndex = timingReport.gtaoSourceFrameIndex;
    metrics.gpuTimingAvailable = 1u;
  }
  metrics.enabled = true;
  metrics.active = true;
  metrics.activePreset = ao.preset;
  metrics.disabledReason = AmbientOcclusionDisabledReason::None;
  metrics.width = width;
  metrics.height = height;
  metrics.depthMipCount = kViewDepthMipCount;
  metrics.strength = ao.strength;
  metrics.depthPrefilterPassCount = 1u;
  metrics.mainPassCount = 1u;
  metrics.temporalAccumulationEnabled = ao.temporalAccumulation;
  metrics.scalarAoAvailable = true;
  metrics.bentNormalAvailable = false;
  metrics.requestedSliceCount = ao.sliceCount;
  metrics.requestedStepCount = ao.stepCount;
  metrics.requestedDenoisePassCount = ao.denoisePassCount;
  uint64_t depthBytes = 0u;
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    depthBytes += textureBytes(Format::R32_FLOAT, levelDimension(width, level),
                               levelDimension(height, level));
  }
  metrics.depthPrefilterTextureBytes = depthBytes;
  metrics.edgeTextureBytes = textureBytes(Format::R8_UNORM, width, height);
  metrics.scratchTextureBytes =
      textureBytes(kFrameCompositionAmbientOcclusionFormat, width, height) * 2u;
  metrics.textureCount += kViewDepthMipCount + 3u;
  metrics.totalTextureBytes += metrics.depthPrefilterTextureBytes +
                               metrics.edgeTextureBytes +
                               metrics.scratchTextureBytes;
  const uint32_t sourceDepthTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.sceneDepthTexture);
  const uint32_t normalTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.normalTexture);
  const uint32_t edgeTexId =
      gpu_.getTextureBindlessIndex(scratch.textures[kScratchEdges]);
  const uint32_t rawTexId = gpu_.getTextureBindlessIndex(
      scratch.textures[kScratchRawAmbientOcclusion]);
  const uint32_t scratchTexId =
      gpu_.getTextureBindlessIndex(scratch.textures[kScratchDenoise]);
  const uint32_t finalTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.ambientOcclusionTexture);
  bool temporalActive =
      presentationAAPlanForFrame(ctx.frame).gtaoTemporal &&
      hasTemporalCameraContinuity(ctx.frame.camera) &&
      ctx.frame.camera.temporalDataValid &&
      nuri::isValid(ctx.shared.previousAmbientOcclusionTexture) &&
      nuri::isValid(ctx.shared.motionVectorTexture) &&
      nuri::isValid(ctx.shared.motionVectorGraphTexture) &&
      nuri::isValid(ctx.shared.motionClassTexture) &&
      nuri::isValid(ctx.shared.motionClassGraphTexture) &&
      nuri::isValid(ctx.shared.previousSceneDepthTexture);
  uint32_t previousAoTexId = kInvalidTextureBindlessIndex;
  uint32_t motionVectorTexId = kInvalidTextureBindlessIndex;
  uint32_t motionClassTexId = kInvalidTextureBindlessIndex;
  uint32_t previousSceneDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t pointSamplerId = kInvalidTextureBindlessIndex;
  uint32_t linearSamplerId = kInvalidTextureBindlessIndex;
  if (temporalActive) {
    previousAoTexId = gpu_.getTextureBindlessIndex(
        ctx.shared.previousAmbientOcclusionTexture);
    motionVectorTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.motionVectorTexture);
    motionClassTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.motionClassTexture);
    previousSceneDepthTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.previousSceneDepthTexture);
    pointSamplerId = gpu_.getSamplerBindlessIndex(samplers_[PointClamp]);
    linearSamplerId = gpu_.getSamplerBindlessIndex(samplers_[LinearClamp]);
  }
  const bool temporalHistoryInvalidated = temporalPolicyChanged_;
  temporalPolicyChanged_ = false;
  if (temporalHistoryInvalidated) {
    temporalActive = false;
  }
  const uint32_t mainSliceCount = ao.sliceCount;
  const uint32_t mainStepCount = ao.stepCount;
  const uint32_t denoisePassCount = std::max(ao.denoisePassCount, 1u);
  const uint32_t mainNoiseIndex =
      temporalActive ? static_cast<uint32_t>(ctx.frame.frameIndex & 63u) : 0u;
  metrics.sliceCount = mainSliceCount;
  metrics.stepCount = mainStepCount;
  metrics.denoisePassCount = denoisePassCount;
  metrics.temporalHistoryInvalidated = temporalHistoryInvalidated;
  metrics.temporalHistoryValid =
      nuri::isValid(ctx.shared.previousAmbientOcclusionTexture);
  metrics.temporalAccumulationActive = temporalActive;
  metrics.temporalMotionVectorsConsumed = temporalActive;
  metrics.temporalMotionClassConsumed = temporalActive;
  metrics.temporalPreviousDepthConsumed = temporalActive;
  metrics.temporalPassCount = temporalActive ? 1u : 0u;
  DepthPrefilterPushConstants depthPc{
      .sourceDepthTexId = sourceDepthTexId,
      .width = width,
      .height = height,
      .projectionType = static_cast<uint32_t>(ctx.frame.camera.projectionType),
      .radiusBits =
          std::bit_cast<uint32_t>(ambientOcclusionPresetRadius(ao.preset)),
      .nearPlane = ctx.frame.camera.nearPlane,
      .farPlane = ctx.frame.camera.farPlane,
  };
  std::array<uint32_t, kViewDepthMipCount> depthTexIds{};
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    depthTexIds[level] = gpu_.getTextureBindlessIndex(scratch.textures[level]);
    depthPc.outputTexIds[level] = depthTexIds[level];
  }
  depthPrefilterDependencies_[0] = ctx.shared.sceneDepthTexture;
  depthPrefilterAccessModes_[0] = RenderGraphAccessMode::Read;
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    depthPrefilterDependencies_[level + 1u] = scratch.textures[level];
    depthPrefilterAccessModes_[level + 1u] = RenderGraphAccessMode::Write;
  }
  depthPrefilterDispatches_[0] = ComputeDispatchItem{
      .pipeline = pipelines_[DepthPrefilter],
      .dispatch = DispatchSize{.x = divRoundUp(width, 16u),
                               .y = divRoundUp(height, 16u),
                               .z = 1u},
      .pushConstants = copyPushConstants(depthPrefilterPushBytes_, depthPc),
      .dependencyTextures = std::span<const TextureHandle>(
          depthPrefilterDependencies_.data(), kViewDepthMipCount + 1u),
      .debugLabel = "GTAO Depth Prefilter",
      .debugColor = kGTAODebugColor,
  };
  RenderGraphGraphicsPassDesc depthPass{};
  depthPass.executionMode = RenderPassExecutionMode::ComputeOnly;
  depthPass.hasColorAttachment = false;
  depthPass.preDispatches = std::span<const ComputeDispatchItem>(
      depthPrefilterDispatches_.data(), depthPrefilterDispatches_.size());
  depthPass.dependencyTextures = std::span<const TextureHandle>(
      depthPrefilterDependencies_.data(), kViewDepthMipCount + 1u);
  depthPass.dependencyTextureAccessModes =
      std::span<const RenderGraphAccessMode>(depthPrefilterAccessModes_.data(),
                                             kViewDepthMipCount + 1u);
  depthPass.gpuTimingScope = GpuTimingScope::GTAO;
  depthPass.debugLabel = "GTAO Depth Prefilter Pass";
  depthPass.debugColor = kGTAODebugColor;
  [[maybe_unused]] const RenderGraphPassId depthPassId =
      ctx.graph.addGraphicsPass(depthPass).value();
  const EdgePushConstants edgePc{
      .depthTexId = depthTexIds[0],
      .outputTexId = edgeTexId,
      .width = width,
      .height = height,
  };
  edgeDependencies_[0] = scratch.textures[0];
  edgeAccessModes_[0] = RenderGraphAccessMode::Read;
  edgeDependencies_[1] = scratch.textures[kScratchEdges];
  edgeAccessModes_[1] = RenderGraphAccessMode::Write;
  edgeDispatches_[0] = ComputeDispatchItem{
      .pipeline = pipelines_[Edge],
      .dispatch = dispatch,
      .pushConstants = copyPushConstants(edgePushBytes_, edgePc),
      .dependencyTextures =
          std::span<const TextureHandle>(edgeDependencies_.data(), 2u),
      .debugLabel = "GTAO Edges",
      .debugColor = kGTAODebugColor,
  };
  RenderGraphGraphicsPassDesc edgePass{};
  edgePass.executionMode = RenderPassExecutionMode::ComputeOnly;
  edgePass.hasColorAttachment = false;
  edgePass.preDispatches = std::span<const ComputeDispatchItem>(
      edgeDispatches_.data(), edgeDispatches_.size());
  edgePass.dependencyTextures =
      std::span<const TextureHandle>(edgeDependencies_.data(), 2u);
  edgePass.dependencyTextureAccessModes =
      std::span<const RenderGraphAccessMode>(edgeAccessModes_.data(), 2u);
  edgePass.gpuTimingScope = GpuTimingScope::GTAO;
  edgePass.debugLabel = "GTAO Edge Pass";
  edgePass.debugColor = kGTAODebugColor;
  [[maybe_unused]] const RenderGraphPassId edgePassId =
      ctx.graph.addGraphicsPass(edgePass).value();
  MainPushConstants mainPc{
      .normalTexId = normalTexId,
      .edgeTexId = edgeTexId,
      .outputTexId = rawTexId,
      .width = width,
      .height = height,
      .noiseIndex = mainNoiseIndex,
      .sliceCount = mainSliceCount,
      .stepCount = mainStepCount,
      .strengthBits = std::bit_cast<uint32_t>(ao.strength),
      .radiusBits =
          std::bit_cast<uint32_t>(ambientOcclusionPresetRadius(ao.preset)),
      .tanHalfFovY = std::tan(ctx.frame.camera.fovYRadians * 0.5f),
      .aspectRatio = ctx.frame.camera.aspectRatio,
      .orthoHeight = ctx.frame.camera.orthoHeight,
      .projectionType = static_cast<uint32_t>(ctx.frame.camera.projectionType),
  };
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    mainPc.depthTexIds[level] = depthTexIds[level];
    mainDependencies_[level] = scratch.textures[level];
    mainAccessModes_[level] = RenderGraphAccessMode::Read;
  }
  mainDependencies_[kViewDepthMipCount] = ctx.shared.normalTexture;
  mainAccessModes_[kViewDepthMipCount] = RenderGraphAccessMode::Read;
  mainDependencies_[kViewDepthMipCount + 1u] = scratch.textures[kScratchEdges];
  mainAccessModes_[kViewDepthMipCount + 1u] = RenderGraphAccessMode::Read;
  mainDependencies_[kViewDepthMipCount + 2u] =
      scratch.textures[kScratchRawAmbientOcclusion];
  mainAccessModes_[kViewDepthMipCount + 2u] = RenderGraphAccessMode::Write;
  mainDispatches_[0] = ComputeDispatchItem{
      .pipeline = pipelines_[Main],
      .dispatch = dispatch,
      .pushConstants = copyPushConstants(mainPushBytes_, mainPc),
      .dependencyTextures = std::span<const TextureHandle>(
          mainDependencies_.data(), kViewDepthMipCount + 3u),
      .debugLabel = "GTAO Main",
      .debugColor = kGTAODebugColor,
  };
  RenderGraphGraphicsPassDesc mainPass{};
  mainPass.executionMode = RenderPassExecutionMode::ComputeOnly;
  mainPass.hasColorAttachment = false;
  mainPass.preDispatches = std::span<const ComputeDispatchItem>(
      mainDispatches_.data(), mainDispatches_.size());
  mainPass.dependencyTextures = std::span<const TextureHandle>(
      mainDependencies_.data(), kViewDepthMipCount + 3u);
  mainPass.dependencyTextureAccessModes =
      std::span<const RenderGraphAccessMode>(mainAccessModes_.data(),
                                             kViewDepthMipCount + 3u);
  mainPass.gpuTimingScope = GpuTimingScope::GTAO;
  mainPass.debugLabel = "GTAO Main Pass";
  mainPass.debugColor = kGTAODebugColor;
  [[maybe_unused]] const RenderGraphPassId mainPassId =
      ctx.graph.addGraphicsPass(mainPass).value();
  TextureHandle denoiseSource = scratch.textures[kScratchRawAmbientOcclusion];
  uint32_t denoiseSourceTexId = rawTexId;
  for (uint32_t passIndex = 0u; passIndex < denoisePassCount; ++passIndex) {
    const bool finalPass =
        !temporalActive && passIndex + 1u == denoisePassCount;
    TextureHandle outputTexture =
        finalPass ? ctx.shared.ambientOcclusionTexture
                  : (passIndex % 2u == 0u
                         ? scratch.textures[kScratchDenoise]
                         : scratch.textures[kScratchRawAmbientOcclusion]);
    const uint32_t outputTexId =
        finalPass ? finalTexId
                  : (passIndex % 2u == 0u ? scratchTexId : rawTexId);
    DenoisePushConstants denoisePc{
        .sourceTexId = denoiseSourceTexId,
        .outputTexId = outputTexId,
        .edgeTexId = edgeTexId,
        .width = width,
        .height = height,
    };
    denoiseDependencies_[0] = denoiseSource;
    denoiseDependencies_[1] = scratch.textures[kScratchEdges];
    denoiseDependencies_[2] = outputTexture;
    denoiseAccessModes_[0] = RenderGraphAccessMode::Read;
    denoiseAccessModes_[1] = RenderGraphAccessMode::Read;
    denoiseAccessModes_[2] = RenderGraphAccessMode::Write;
    denoiseDispatches_[0] = ComputeDispatchItem{
        .pipeline = pipelines_[Denoise],
        .dispatch = dispatch,
        .pushConstants = copyPushConstants(denoisePushBytes_, denoisePc),
        .dependencyTextures =
            std::span<const TextureHandle>(denoiseDependencies_.data(), 3u),
        .debugLabel = finalPass ? "GTAO Denoise Final" : "GTAO Denoise",
        .debugColor = kGTAODebugColor,
    };
    RenderGraphGraphicsPassDesc denoisePass{};
    denoisePass.executionMode = RenderPassExecutionMode::ComputeOnly;
    denoisePass.hasColorAttachment = false;
    denoisePass.preDispatches = std::span<const ComputeDispatchItem>(
        denoiseDispatches_.data(), denoiseDispatches_.size());
    denoisePass.dependencyTextures =
        std::span<const TextureHandle>(denoiseDependencies_.data(), 3u);
    denoisePass.dependencyTextureAccessModes =
        std::span<const RenderGraphAccessMode>(denoiseAccessModes_.data(), 3u);
    denoisePass.gpuTimingScope = GpuTimingScope::GTAO;
    denoisePass.debugLabel =
        finalPass ? "GTAO Denoise Final Pass" : "GTAO Denoise Pass";
    denoisePass.debugColor = kGTAODebugColor;
    [[maybe_unused]] const RenderGraphPassId denoisePassId =
        ctx.graph.addGraphicsPass(denoisePass).value();
    denoiseSource = outputTexture;
    denoiseSourceTexId = outputTexId;
  }
  if (temporalActive) {
    const TemporalPushConstants temporalPc{
        .currentTexId = denoiseSourceTexId,
        .historyTexId = previousAoTexId,
        .motionVectorTexId = motionVectorTexId,
        .motionClassTexId = motionClassTexId,
        .sceneDepthTexId = sourceDepthTexId,
        .currentViewDepthTexId = depthTexIds[0],
        .previousSceneDepthTexId = previousSceneDepthTexId,
        .edgeTexId = edgeTexId,
        .outputTexId = finalTexId,
        .pointSamplerId = pointSamplerId,
        .historySamplerId = linearSamplerId,
        .width = width,
        .height = height,
        .flags = kTemporalFlagsDefault,
        .baseCurrentWeightBits =
            std::bit_cast<uint32_t>(kTemporalBaseCurrentWeight),
        .rejectedCurrentWeightBits =
            std::bit_cast<uint32_t>(kTemporalRejectedCurrentWeight),
        .projectionType =
            static_cast<uint32_t>(ctx.frame.camera.projectionType),
        .nearPlane = ctx.frame.camera.nearPlane,
        .farPlane = ctx.frame.camera.farPlane,
    };
    temporalDependencies_[0] = denoiseSource;
    temporalDependencies_[1] = ctx.shared.previousAmbientOcclusionTexture;
    temporalDependencies_[2] = ctx.shared.motionVectorTexture;
    temporalDependencies_[3] = ctx.shared.motionClassTexture;
    temporalDependencies_[4] = ctx.shared.sceneDepthTexture;
    temporalDependencies_[5] = scratch.textures[0];
    temporalDependencies_[6] = ctx.shared.previousSceneDepthTexture;
    temporalDependencies_[7] = scratch.textures[kScratchEdges];
    temporalDependencies_[8] = ctx.shared.ambientOcclusionTexture;
    for (uint32_t i = 0u; i < 8u; ++i) {
      temporalAccessModes_[i] = RenderGraphAccessMode::Read;
    }
    temporalAccessModes_[8] = RenderGraphAccessMode::Write;
    temporalDispatches_[0] = ComputeDispatchItem{
        .pipeline = pipelines_[Temporal],
        .dispatch = dispatch,
        .pushConstants = copyPushConstants(temporalPushBytes_, temporalPc),
        .dependencyTextures =
            std::span<const TextureHandle>(temporalDependencies_.data(), 9u),
        .debugLabel = "GTAO Temporal",
        .debugColor = kGTAODebugColor,
    };
    RenderGraphGraphicsPassDesc temporalPass{};
    temporalPass.executionMode = RenderPassExecutionMode::ComputeOnly;
    temporalPass.hasColorAttachment = false;
    temporalPass.preDispatches = std::span<const ComputeDispatchItem>(
        temporalDispatches_.data(), temporalDispatches_.size());
    temporalPass.dependencyTextures =
        std::span<const TextureHandle>(temporalDependencies_.data(), 9u);
    temporalPass.dependencyTextureAccessModes =
        std::span<const RenderGraphAccessMode>(temporalAccessModes_.data(), 9u);
    temporalPass.gpuTimingScope = GpuTimingScope::GTAOTemporal;
    temporalPass.debugLabel = "GTAO Temporal Pass";
    temporalPass.debugColor = kGTAODebugColor;
    [[maybe_unused]] const RenderGraphPassId temporalPassId =
        ctx.graph.addGraphicsPass(temporalPass).value();
  }
  publishRequestedCapture(ctx.frame, gpu_, "gtao_edges",
                          scratch.textures[kScratchEdges],
                          RenderCaptureValueKind::Mask,
                          RenderCaptureLifetimeClass::FeaturePersistentTexture,
                          "packed_edge_mask_u8", "mask", "GTAO Edge Pass");
  publishRequestedCapture(ctx.frame, gpu_, "gtao_current", denoiseSource,
                          RenderCaptureValueKind::Scalar,
                          RenderCaptureLifetimeClass::FeaturePersistentTexture,
                          "linear_scalar", "scalar", "GTAO Denoise Pass");
  publishRequestedCapture(ctx.frame, gpu_, "gtao_history",
                          ctx.shared.previousAmbientOcclusionTexture,
                          RenderCaptureValueKind::Scalar,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "linear_scalar", "scalar", "GTAO Temporal Pass");
  publishRequestedCapture(ctx.frame, gpu_, "gtao_previous_depth",
                          ctx.shared.previousSceneDepthTexture,
                          RenderCaptureValueKind::Depth,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "device_depth", "depth", "GTAO Temporal Pass");
  const RenderGraphTextureId finalTexture =
      ctx.graph
          .importTexture(ctx.shared.ambientOcclusionTexture,
                         "gtao_final_ambient_occlusion")
          .value();
  ctx.shared.ambientOcclusionGraphTexture = finalTexture;
  ctx.frame.sharedResources.ambientOcclusionGraphTexture = finalTexture;
  ctx.shared.historyWriteRequirements |=
      FrameTextureRequirementFlags::AmbientOcclusion;
  ctx.frame.metrics.ambientOcclusion.ambientOcclusionGraphPublished = true;
  publishRequestedCapture(ctx.frame, gpu_, "ambient_occlusion",
                          ctx.shared.ambientOcclusionTexture,
                          RenderCaptureValueKind::Scalar,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "linear_scalar", "scalar", "GTAO Denoise Final Pass");
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GTAOPass::publishFrameData(FrameBuildContext &ctx) {
  const RenderSettings::AmbientOcclusionSettings ao =
      resolvedAmbientOcclusionSettings(ctx.frame);
  AmbientOcclusionFrameMetrics &metrics = ctx.frame.metrics.ambientOcclusion;
  metrics.enabled = ao.mode != AmbientOcclusionMode::Disabled;
  metrics.active = ao.active;
  metrics.activePreset = ao.preset;
  metrics.disabledReason = ao.disabledReason;
  metrics.strength = ao.strength;
  metrics.requestedSliceCount = ao.sliceCount;
  metrics.requestedStepCount = ao.stepCount;
  metrics.requestedDenoisePassCount = ao.denoisePassCount;
  metrics.sliceCount = ao.sliceCount;
  metrics.stepCount = ao.stepCount;
  metrics.denoisePassCount = ao.denoisePassCount;
  metrics.temporalAccumulationEnabled = ao.temporalAccumulation;
  observeTemporalPolicy(ao);
  if (ao.active) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::Normals |
        FrameTextureRequirementFlags::AmbientOcclusion;
    if (presentationAAPlanForFrame(ctx.frame).gtaoTemporal) {
      ctx.shared.textureRequirements |=
          FrameTextureRequirementFlags::MotionVectors |
          FrameTextureRequirementFlags::ReactiveMask |
          FrameTextureRequirementFlags::MotionClass;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

void registerGTAOStage(RenderPipeline &pipeline, GPUDevice &gpu,
                       RuntimeOpaqueShaderConfig config) {
  pipeline.addStage(
      std::make_unique<GTAOPass>(gpu, std::move(config)), "GTAOFeature",
      "GTAOPass", false,
      PipelineComponentDesc{.publish = [](void *state, FrameBuildContext &ctx) {
        return static_cast<GTAOPass *>(state)->publishFrameData(ctx);
      }});
}

} // namespace nuri
