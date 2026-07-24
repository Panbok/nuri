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
  ReconstructNormals,
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
    {"gtao_reconstruct_normals", "gtao_reconstruct_normals.comp"},
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
  uint32_t outputWidth = 1u;
  uint32_t outputHeight = 1u;
  uint32_t workingWidth = 1u;
  uint32_t workingHeight = 1u;
  uint32_t noiseIndex = 0u;
  uint32_t sliceCount = 2u;
  uint32_t stepCount = 4u;
  uint32_t strengthBits = 0u;
  uint32_t radiusBits = 0u;
  float tanHalfFovY = 1.0f;
  float aspectRatio = 1.0f;
  float orthoHeight = 10.0f;
  uint32_t projectionType = 0u;
  uint32_t inputMode = 0u;
};
static_assert(sizeof(MainPushConstants) <= 128u);
struct DenoisePushConstants {
  uint32_t sourceTexId = kInvalidTextureBindlessIndex;
  uint32_t outputTexId = kInvalidTextureBindlessIndex;
  uint32_t edgeTexId = kInvalidTextureBindlessIndex;
  uint32_t outputWidth = 1u;
  uint32_t outputHeight = 1u;
  uint32_t workingWidth = 1u;
  uint32_t workingHeight = 1u;
};
static_assert(sizeof(DenoisePushConstants) <= 128u);
struct TemporalPushConstants {
  uint32_t currentTexId = kInvalidTextureBindlessIndex;
  uint32_t historyTexId = kInvalidTextureBindlessIndex;
  uint32_t motionVectorTexId = kInvalidTextureBindlessIndex;
  uint32_t reactiveMaskTexId = kInvalidTextureBindlessIndex;
  uint32_t sceneDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t currentViewDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t previousSceneDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t edgeTexId = kInvalidTextureBindlessIndex;
  uint32_t outputTexId = kInvalidTextureBindlessIndex;
  uint32_t pointSamplerId = 0u;
  uint32_t historySamplerId = 0u;
  uint32_t width = 1u;
  uint32_t height = 1u;
  uint32_t currentWidth = 1u;
  uint32_t currentHeight = 1u;
  uint32_t flags = 0u;
  uint32_t baseCurrentWeightBits = 0u;
  uint32_t rejectedCurrentWeightBits = 0u;
  uint32_t projectionType = 0u;
  float nearPlane = 0.01f;
  float farPlane = 1000.0f;
  uint32_t forceInvalidGeometry = 0u;
  uint32_t provenStaticCameraOnly = 0u;
};
static_assert(sizeof(TemporalPushConstants) <= 128u);
struct ReconstructNormalsPushConstants {
  uint32_t depthTexId = kInvalidTextureBindlessIndex;
  uint32_t outputTexId = kInvalidTextureBindlessIndex;
  uint32_t width = 1u;
  uint32_t height = 1u;
  float tanHalfFovY = 1.0f;
  float aspectRatio = 1.0f;
  float orthoHeight = 10.0f;
  uint32_t projectionType = 0u;
};
static_assert(sizeof(ReconstructNormalsPushConstants) <= 128u);
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
    const AmbientOcclusionExecutionPlan &plan, float strength) noexcept {
  uint64_t signature = 0xcbf29ce484222325ull;
  signature = mixSignature(signature, plan.active ? 1u : 0u);
  signature = mixSignature(signature, static_cast<uint32_t>(plan.inputMode));
  signature = mixSignature(signature, static_cast<uint32_t>(plan.preset));
  signature =
      mixSignature(signature, static_cast<uint32_t>(plan.workingResolution));
  signature = mixSignature(signature, std::bit_cast<uint32_t>(strength));
  signature = mixSignature(signature, plan.temporal ? 1u : 0u);
  signature = mixSignature(signature, plan.sliceCount);
  signature = mixSignature(signature, plan.stepCount);
  signature = mixSignature(signature, plan.denoisePassCount);
  signature = mixSignature(signature, plan.outputWidth);
  signature = mixSignature(signature, plan.outputHeight);
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

void GTAOPass::observeTemporalPolicy(const AmbientOcclusionExecutionPlan &plan,
                                     float strength) noexcept {
  const uint64_t temporalPolicySignature =
      ambientOcclusionTemporalPolicySignature(plan, strength);
  temporalPolicyChanged_ =
      hasLastTemporalPolicySignature_ &&
      lastTemporalPolicySignature_ != temporalPolicySignature;
  hasLastTemporalPolicySignature_ = true;
  lastTemporalPolicySignature_ = temporalPolicySignature;
}

bool GTAOPass::isEnabled(const FrameBuildContext &ctx) const {
  return ctx.frame.ambientOcclusion.active;
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
  const AmbientOcclusionExecutionPlan &plan = ctx.frame.ambientOcclusion;
  const uint32_t workingWidth = plan.workingWidth;
  const uint32_t workingHeight = plan.workingHeight;
  const uint32_t ringCount = std::max(1u, gpu_.getSwapchainImageCount());
  if (scratchOutputWidth_ != width || scratchOutputHeight_ != height ||
      scratchWorkingWidth_ != workingWidth ||
      scratchWorkingHeight_ != workingHeight ||
      scratchRingCount_ != ringCount || scratchTextures_.empty()) {
    auto recreateResult = recreateScratchTextures(width, height, workingWidth,
                                                  workingHeight, ringCount);
    if (recreateResult.hasError()) {
      return recreateResult;
    }
  }
  if (ctx.frame.ambientOcclusion.inputMode ==
          AmbientOcclusionInputMode::DepthOnlyReconstructedNormal &&
      isRenderCaptureRequested(ctx.frame, "material_normals") &&
      !nuri::isValid(reconstructedNormalDebugTexture_)) {
    auto texture =
        gpu_.createTexture(makeStorageSampledTextureDesc(
                               kFrameCompositionNormalFormat, width, height),
                           "gtao_reconstructed_normals_debug");
    if (texture.hasError()) {
      return Result<bool, std::string>::makeError(texture.error());
    }
    reconstructedNormalDebugTexture_ = texture.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
GTAOPass::recreateScratchTextures(uint32_t width, uint32_t height,
                                  uint32_t workingWidth, uint32_t workingHeight,
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
      const bool fullResolution = index == 0u;
      auto result =
          create(kScratchEdges + index,
                 makeStorageSampledTextureDesc(
                     formats[index], fullResolution ? width : workingWidth,
                     fullResolution ? height : workingHeight),
                 names[index] + std::to_string(slot));
      if (result.hasError())
        return result;
    }
  }
  scratchOutputWidth_ = width;
  scratchOutputHeight_ = height;
  scratchWorkingWidth_ = workingWidth;
  scratchWorkingHeight_ = workingHeight;
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
  if (nuri::isValid(reconstructedNormalDebugTexture_)) {
    gpu_.destroyTexture(reconstructedNormalDebugTexture_);
    reconstructedNormalDebugTexture_ = {};
  }
  for (FrameScratchTextures &textures : scratchTextures_) {
    for (TextureHandle &texture : textures.textures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
        texture = {};
      }
    }
  }
  scratchTextures_.clear();
  scratchOutputWidth_ = 0u;
  scratchOutputHeight_ = 0u;
  scratchWorkingWidth_ = 0u;
  scratchWorkingHeight_ = 0u;
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
  const AmbientOcclusionExecutionPlan &plan = ctx.frame.ambientOcclusion;
  const RenderSettings::AmbientOcclusionSettings &ao =
      renderSettingsOrDefault(ctx.frame).ambientOcclusion;
  const TextureDimensions dimensions =
      gpu_.getTextureDimensions(ctx.shared.sceneDepthTexture);
  const uint32_t width = std::max(dimensions.width, 1u);
  const uint32_t height = std::max(dimensions.height, 1u);
  const bool halfResolution =
      plan.workingResolution == AmbientOcclusionWorkingResolution::Half;
  const uint32_t workingWidth = plan.workingWidth;
  const uint32_t workingHeight = plan.workingHeight;
  const DispatchSize fullDispatch{.x = divRoundUp(width, kGTAOWorkgroupSizeX),
                                  .y = divRoundUp(height, kGTAOWorkgroupSizeY),
                                  .z = 1u};
  const DispatchSize workingDispatch{
      .x = divRoundUp(workingWidth, kGTAOWorkgroupSizeX),
      .y = divRoundUp(workingHeight, kGTAOWorkgroupSizeY),
      .z = 1u};
  AmbientOcclusionFrameMetrics &metrics = ctx.frame.metrics.ambientOcclusion;
  const GpuTimingReport &timingReport = ctx.frame.gpuTiming;
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAO)) {
    metrics.gpuTimeMs = timingReport.gtaoTimeMs;
    metrics.gpuTimingSourceFrameIndex = timingReport.gtaoSourceFrameIndex;
    metrics.gpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAOPrefilterEdges)) {
    metrics.prefilterEdgesGpuTimeMs = timingReport.gtaoPrefilterEdgesTimeMs;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAOMain)) {
    metrics.mainGpuTimeMs = timingReport.gtaoMainTimeMs;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAODenoise)) {
    metrics.denoiseGpuTimeMs = timingReport.gtaoDenoiseTimeMs;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAOUpscale)) {
    metrics.upscaleGpuTimeMs = timingReport.gtaoUpscaleTimeMs;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAOTemporal)) {
    metrics.temporalGpuTimeMs = timingReport.gtaoTemporalTimeMs;
    metrics.upscaleGpuTimeMs = timingReport.gtaoTemporalTimeMs;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::OpaqueNormal)) {
    metrics.inputGpuTimeMs = timingReport.opaqueNormalTimeMs;
  }
  metrics.enabled = true;
  metrics.active = true;
  metrics.activePreset = plan.preset;
  metrics.inputMode = plan.inputMode;
  metrics.workingResolution = plan.workingResolution;
  metrics.disabledReason = AmbientOcclusionDisabledReason::None;
  metrics.width = width;
  metrics.height = height;
  metrics.workingWidth = workingWidth;
  metrics.workingHeight = workingHeight;
  metrics.depthMipCount = kViewDepthMipCount;
  metrics.strength = ao.strength;
  metrics.depthPrefilterPassCount = 1u;
  metrics.mainPassCount = 1u;
  metrics.temporalAccumulationEnabled = plan.temporal;
  metrics.scalarAoAvailable = true;
  metrics.bentNormalAvailable = false;
  metrics.requestedSliceCount = plan.sliceCount;
  metrics.requestedStepCount = plan.stepCount;
  metrics.requestedDenoisePassCount = plan.denoisePassCount;
  uint64_t depthBytes = 0u;
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    depthBytes += textureBytes(Format::R32_FLOAT, levelDimension(width, level),
                               levelDimension(height, level));
  }
  const uint64_t allocationCount = scratchRingCount_;
  metrics.depthPrefilterTextureBytes = depthBytes * allocationCount;
  metrics.edgeTextureBytes =
      textureBytes(Format::R8_UNORM, width, height) * allocationCount;
  metrics.scratchTextureBytes =
      textureBytes(kFrameCompositionAmbientOcclusionFormat, workingWidth,
                   workingHeight) *
      2u * allocationCount;
  metrics.textureCount +=
      (kViewDepthMipCount + 3u) * static_cast<uint32_t>(allocationCount);
  metrics.totalTextureBytes += metrics.depthPrefilterTextureBytes +
                               metrics.edgeTextureBytes +
                               metrics.scratchTextureBytes;
  metrics.featureTextureBytes = metrics.depthPrefilterTextureBytes +
                                metrics.edgeTextureBytes +
                                metrics.scratchTextureBytes;
  metrics.logicalActiveTextureBytes +=
      depthBytes + textureBytes(Format::R8_UNORM, width, height) +
      textureBytes(kFrameCompositionAmbientOcclusionFormat, workingWidth,
                   workingHeight) *
          2u;
  const uint32_t sourceDepthTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.sceneDepthTexture);
  const uint32_t normalTexId =
      nuri::isValid(ctx.shared.normalTexture)
          ? gpu_.getTextureBindlessIndex(ctx.shared.normalTexture)
          : sourceDepthTexId;
  const uint32_t edgeTexId =
      gpu_.getTextureBindlessIndex(scratch.textures[kScratchEdges]);
  const uint32_t rawTexId = gpu_.getTextureBindlessIndex(
      scratch.textures[kScratchRawAmbientOcclusion]);
  const uint32_t scratchTexId =
      gpu_.getTextureBindlessIndex(scratch.textures[kScratchDenoise]);
  const uint32_t finalTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.ambientOcclusionTexture);
  bool temporalActive =
      plan.temporal && hasTemporalCameraContinuity(ctx.frame.camera) &&
      ctx.frame.camera.temporalDataValid &&
      nuri::isValid(ctx.shared.previousAmbientOcclusionTexture) &&
      nuri::isValid(ctx.shared.motionVectorTexture) &&
      nuri::isValid(ctx.shared.motionVectorGraphTexture) &&
      nuri::isValid(ctx.shared.reactiveMaskTexture) &&
      nuri::isValid(ctx.shared.reactiveMaskGraphTexture) &&
      nuri::isValid(ctx.shared.previousSceneDepthTexture);
  uint32_t previousAoTexId = kInvalidTextureBindlessIndex;
  uint32_t motionVectorTexId = kInvalidTextureBindlessIndex;
  uint32_t reactiveMaskTexId = kInvalidTextureBindlessIndex;
  uint32_t previousSceneDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t pointSamplerId = kInvalidTextureBindlessIndex;
  uint32_t linearSamplerId = kInvalidTextureBindlessIndex;
  if (temporalActive) {
    previousAoTexId = gpu_.getTextureBindlessIndex(
        ctx.shared.previousAmbientOcclusionTexture);
    motionVectorTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.motionVectorTexture);
    reactiveMaskTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.reactiveMaskTexture);
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
  const uint32_t mainSliceCount = plan.sliceCount;
  const uint32_t mainStepCount = plan.stepCount;
  const uint32_t denoisePassCount = std::max(plan.denoisePassCount, 1u);
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
  metrics.temporalReactiveMaskConsumed = temporalActive;
  metrics.temporalMotionClassConsumed = false;
  metrics.temporalPreviousDepthConsumed = temporalActive;
  metrics.temporalPassCount = temporalActive ? 1u : 0u;
  DepthPrefilterPushConstants depthPc{
      .sourceDepthTexId = sourceDepthTexId,
      .width = width,
      .height = height,
      .projectionType = static_cast<uint32_t>(ctx.frame.camera.projectionType),
      .radiusBits =
          std::bit_cast<uint32_t>(ambientOcclusionPresetRadius(plan.preset)),
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
  depthPass.gpuTimingScope = GpuTimingScope::GTAOPrefilterEdges;
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
      .dispatch = fullDispatch,
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
  edgePass.gpuTimingScope = GpuTimingScope::GTAOPrefilterEdges;
  edgePass.debugLabel = "GTAO Edge Pass";
  edgePass.debugColor = kGTAODebugColor;
  [[maybe_unused]] const RenderGraphPassId edgePassId =
      ctx.graph.addGraphicsPass(edgePass).value();
  if (nuri::isValid(reconstructedNormalDebugTexture_)) {
    const ReconstructNormalsPushConstants reconstructPc{
        .depthTexId = depthTexIds[0],
        .outputTexId =
            gpu_.getTextureBindlessIndex(reconstructedNormalDebugTexture_),
        .width = width,
        .height = height,
        .tanHalfFovY = std::tan(ctx.frame.camera.fovYRadians * 0.5f),
        .aspectRatio = ctx.frame.camera.aspectRatio,
        .orthoHeight = ctx.frame.camera.orthoHeight,
        .projectionType =
            static_cast<uint32_t>(ctx.frame.camera.projectionType),
    };
    reconstructNormalDependencies_[0] = scratch.textures[0];
    reconstructNormalDependencies_[1] = reconstructedNormalDebugTexture_;
    reconstructNormalAccessModes_[0] = RenderGraphAccessMode::Read;
    reconstructNormalAccessModes_[1] = RenderGraphAccessMode::Write;
    reconstructNormalDispatches_[0] = ComputeDispatchItem{
        .pipeline = pipelines_[ReconstructNormals],
        .dispatch = fullDispatch,
        .pushConstants =
            copyPushConstants(reconstructNormalPushBytes_, reconstructPc),
        .dependencyTextures = reconstructNormalDependencies_,
        .debugLabel = "GTAO Reconstructed Normals Debug",
        .debugColor = kGTAODebugColor,
    };
    RenderGraphGraphicsPassDesc reconstructPass{};
    reconstructPass.executionMode = RenderPassExecutionMode::ComputeOnly;
    reconstructPass.hasColorAttachment = false;
    reconstructPass.preDispatches = reconstructNormalDispatches_;
    reconstructPass.dependencyTextures = reconstructNormalDependencies_;
    reconstructPass.dependencyTextureAccessModes =
        reconstructNormalAccessModes_;
    reconstructPass.debugLabel = "GTAO Reconstructed Normals Debug Pass";
    reconstructPass.debugColor = kGTAODebugColor;
    [[maybe_unused]] const RenderGraphPassId reconstructPassId =
        ctx.graph.addGraphicsPass(reconstructPass).value();
    publishRequestedCapture(
        ctx.frame, gpu_, "material_normals", reconstructedNormalDebugTexture_,
        RenderCaptureValueKind::Normal,
        RenderCaptureLifetimeClass::FeaturePersistentTexture, "view_normal",
        "normal", reconstructPass.debugLabel);
  }
  MainPushConstants mainPc{
      .normalTexId = normalTexId,
      .edgeTexId = edgeTexId,
      .outputTexId = rawTexId,
      .outputWidth = width,
      .outputHeight = height,
      .workingWidth = workingWidth,
      .workingHeight = workingHeight,
      .noiseIndex = mainNoiseIndex,
      .sliceCount = mainSliceCount,
      .stepCount = mainStepCount,
      .strengthBits = std::bit_cast<uint32_t>(ao.strength),
      .radiusBits =
          std::bit_cast<uint32_t>(ambientOcclusionPresetRadius(plan.preset)),
      .tanHalfFovY = std::tan(ctx.frame.camera.fovYRadians * 0.5f),
      .aspectRatio = ctx.frame.camera.aspectRatio,
      .orthoHeight = ctx.frame.camera.orthoHeight,
      .projectionType = static_cast<uint32_t>(ctx.frame.camera.projectionType),
      .inputMode = static_cast<uint32_t>(plan.inputMode),
  };
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    mainPc.depthTexIds[level] = depthTexIds[level];
    mainDependencies_[level] = scratch.textures[level];
    mainAccessModes_[level] = RenderGraphAccessMode::Read;
  }
  uint32_t mainDependencyCount = kViewDepthMipCount;
  if (plan.inputMode == AmbientOcclusionInputMode::MaterialNormalAndDepth) {
    mainDependencies_[mainDependencyCount] = ctx.shared.normalTexture;
    mainAccessModes_[mainDependencyCount++] = RenderGraphAccessMode::Read;
  }
  mainDependencies_[mainDependencyCount] = scratch.textures[kScratchEdges];
  mainAccessModes_[mainDependencyCount++] = RenderGraphAccessMode::Read;
  mainDependencies_[mainDependencyCount] =
      scratch.textures[kScratchRawAmbientOcclusion];
  mainAccessModes_[mainDependencyCount++] = RenderGraphAccessMode::Write;
  mainDispatches_[0] = ComputeDispatchItem{
      .pipeline = pipelines_[Main],
      .dispatch = workingDispatch,
      .pushConstants = copyPushConstants(mainPushBytes_, mainPc),
      .dependencyTextures = std::span<const TextureHandle>(
          mainDependencies_.data(), mainDependencyCount),
      .debugLabel = "GTAO Main",
      .debugColor = kGTAODebugColor,
  };
  RenderGraphGraphicsPassDesc mainPass{};
  mainPass.executionMode = RenderPassExecutionMode::ComputeOnly;
  mainPass.hasColorAttachment = false;
  mainPass.preDispatches = std::span<const ComputeDispatchItem>(
      mainDispatches_.data(), mainDispatches_.size());
  mainPass.dependencyTextures = std::span<const TextureHandle>(
      mainDependencies_.data(), mainDependencyCount);
  mainPass.dependencyTextureAccessModes =
      std::span<const RenderGraphAccessMode>(mainAccessModes_.data(),
                                             mainDependencyCount);
  mainPass.gpuTimingScope = GpuTimingScope::GTAOMain;
  mainPass.debugLabel = "GTAO Main Pass";
  mainPass.debugColor = kGTAODebugColor;
  [[maybe_unused]] const RenderGraphPassId mainPassId =
      ctx.graph.addGraphicsPass(mainPass).value();
  TextureHandle denoiseSource = scratch.textures[kScratchRawAmbientOcclusion];
  uint32_t denoiseSourceTexId = rawTexId;
  for (uint32_t passIndex = 0u; passIndex < denoisePassCount; ++passIndex) {
    const bool finalPass = !halfResolution && !temporalActive &&
                           passIndex + 1u == denoisePassCount;
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
        .outputWidth = width,
        .outputHeight = height,
        .workingWidth = workingWidth,
        .workingHeight = workingHeight,
    };
    denoiseDependencies_[0] = denoiseSource;
    denoiseDependencies_[1] = scratch.textures[kScratchEdges];
    denoiseDependencies_[2] = outputTexture;
    denoiseAccessModes_[0] = RenderGraphAccessMode::Read;
    denoiseAccessModes_[1] = RenderGraphAccessMode::Read;
    denoiseAccessModes_[2] = RenderGraphAccessMode::Write;
    denoiseDispatches_[0] = ComputeDispatchItem{
        .pipeline = pipelines_[Denoise],
        .dispatch = workingDispatch,
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
    denoisePass.gpuTimingScope = GpuTimingScope::GTAODenoise;
    denoisePass.debugLabel =
        finalPass ? "GTAO Denoise Final Pass" : "GTAO Denoise Pass";
    denoisePass.debugColor = kGTAODebugColor;
    [[maybe_unused]] const RenderGraphPassId denoisePassId =
        ctx.graph.addGraphicsPass(denoisePass).value();
    denoiseSource = outputTexture;
    denoiseSourceTexId = outputTexId;
  }
  if (halfResolution || temporalActive) {
    const TemporalPushConstants temporalPc{
        .currentTexId = denoiseSourceTexId,
        .historyTexId = previousAoTexId,
        .motionVectorTexId = motionVectorTexId,
        .reactiveMaskTexId = reactiveMaskTexId,
        .sceneDepthTexId = sourceDepthTexId,
        .currentViewDepthTexId = depthTexIds[0],
        .previousSceneDepthTexId = previousSceneDepthTexId,
        .edgeTexId = edgeTexId,
        .outputTexId = finalTexId,
        .pointSamplerId = pointSamplerId,
        .historySamplerId = linearSamplerId,
        .width = width,
        .height = height,
        .currentWidth = workingWidth,
        .currentHeight = workingHeight,
        .flags = temporalActive ? kTemporalFlagsDefault : 0u,
        .baseCurrentWeightBits =
            std::bit_cast<uint32_t>(kTemporalBaseCurrentWeight),
        .rejectedCurrentWeightBits =
            std::bit_cast<uint32_t>(kTemporalRejectedCurrentWeight),
        .projectionType =
            static_cast<uint32_t>(ctx.frame.camera.projectionType),
        .nearPlane = ctx.frame.camera.nearPlane,
        .farPlane = ctx.frame.camera.farPlane,
        .forceInvalidGeometry =
            (!ctx.frame.metrics.antiAliasing.opaqueVelocityGenerated ||
             ctx.frame.metrics.antiAliasing
                     .velocityTessellatedSkippedDrawCount > 0u)
                ? 1u
                : 0u,
        .provenStaticCameraOnly =
            ctx.frame.metrics.antiAliasing
                    .motionVectorDepthReprojectionGenerated
                ? 1u
                : 0u,
    };
    temporalDependencies_[0] = denoiseSource;
    temporalDependencies_[1] = scratch.textures[0];
    temporalDependencies_[2] = scratch.textures[kScratchEdges];
    temporalDependencies_[3] = ctx.shared.ambientOcclusionTexture;
    uint32_t temporalDependencyCount = 4u;
    if (temporalActive) {
      temporalDependencies_[4] = ctx.shared.previousAmbientOcclusionTexture;
      temporalDependencies_[5] = ctx.shared.motionVectorTexture;
      temporalDependencies_[6] = ctx.shared.reactiveMaskTexture;
      temporalDependencies_[7] = ctx.shared.sceneDepthTexture;
      temporalDependencies_[8] = ctx.shared.previousSceneDepthTexture;
      temporalDependencyCount = 9u;
    }
    for (uint32_t i = 0u; i < temporalDependencyCount; ++i) {
      temporalAccessModes_[i] = RenderGraphAccessMode::Read;
    }
    temporalAccessModes_[3] = RenderGraphAccessMode::Write;
    temporalDispatches_[0] = ComputeDispatchItem{
        .pipeline = pipelines_[Temporal],
        .dispatch = fullDispatch,
        .pushConstants = copyPushConstants(temporalPushBytes_, temporalPc),
        .dependencyTextures = std::span<const TextureHandle>(
            temporalDependencies_.data(), temporalDependencyCount),
        .debugLabel = temporalActive ? "GTAO Upscale Temporal" : "GTAO Upscale",
        .debugColor = kGTAODebugColor,
    };
    RenderGraphGraphicsPassDesc temporalPass{};
    temporalPass.executionMode = RenderPassExecutionMode::ComputeOnly;
    temporalPass.hasColorAttachment = false;
    temporalPass.preDispatches = std::span<const ComputeDispatchItem>(
        temporalDispatches_.data(), temporalDispatches_.size());
    temporalPass.dependencyTextures = std::span<const TextureHandle>(
        temporalDependencies_.data(), temporalDependencyCount);
    temporalPass.dependencyTextureAccessModes =
        std::span<const RenderGraphAccessMode>(temporalAccessModes_.data(),
                                               temporalDependencyCount);
    temporalPass.gpuTimingScope = temporalActive ? GpuTimingScope::GTAOTemporal
                                                 : GpuTimingScope::GTAOUpscale;
    temporalPass.debugLabel =
        temporalActive ? "GTAO Upscale Temporal Pass" : "GTAO Upscale Pass";
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
  const RenderSettings::AmbientOcclusionSettings &ao =
      renderSettingsOrDefault(ctx.frame).ambientOcclusion;
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  ctx.frame.ambientOcclusion = resolveAmbientOcclusionExecutionPlan(
      renderSettingsOrDefault(ctx.frame),
      presentationAAPlanForFrame(ctx.frame).coverage,
      static_cast<uint32_t>(std::max(framebufferWidth, 1)),
      static_cast<uint32_t>(std::max(framebufferHeight, 1)));
  const AmbientOcclusionExecutionPlan &plan = ctx.frame.ambientOcclusion;
  AmbientOcclusionFrameMetrics &metrics = ctx.frame.metrics.ambientOcclusion;
  metrics.enabled = ao.mode != AmbientOcclusionMode::Disabled;
  metrics.active = plan.active;
  metrics.activePreset = plan.preset;
  metrics.inputMode = plan.inputMode;
  metrics.workingResolution = plan.workingResolution;
  metrics.disabledReason = ao.disabledReason;
  metrics.strength = ao.strength;
  metrics.width = plan.outputWidth;
  metrics.height = plan.outputHeight;
  metrics.workingWidth = plan.workingWidth;
  metrics.workingHeight = plan.workingHeight;
  metrics.requestedSliceCount = plan.sliceCount;
  metrics.requestedStepCount = plan.stepCount;
  metrics.requestedDenoisePassCount = plan.denoisePassCount;
  metrics.sliceCount = plan.sliceCount;
  metrics.stepCount = plan.stepCount;
  metrics.denoisePassCount = plan.denoisePassCount;
  metrics.temporalAccumulationEnabled = plan.temporal;
  observeTemporalPolicy(plan, ao.strength);
  if (plan.active) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::AmbientOcclusion;
    if (plan.inputMode == AmbientOcclusionInputMode::MaterialNormalAndDepth) {
      ctx.shared.textureRequirements |= FrameTextureRequirementFlags::Normals;
    }
    if (presentationAAPlanForFrame(ctx.frame).gtaoTemporal) {
      ctx.shared.textureRequirements |=
          FrameTextureRequirementFlags::MotionVectors |
          FrameTextureRequirementFlags::ReactiveMask;
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
