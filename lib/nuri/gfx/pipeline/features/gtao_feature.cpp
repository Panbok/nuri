#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/gtao_feature.h"

#include "nuri/core/profiling.h"

#include <bit>
#include <cmath>
#include <cstring>

namespace nuri {
namespace {

constexpr uint32_t kInvalidTextureBindlessIndex = 0xffffffffu;
constexpr uint32_t kGTAOWorkgroupSizeX = 8u;
constexpr uint32_t kGTAOWorkgroupSizeY = 8u;
constexpr uint32_t kGTAODebugColor = 0xff44ddaa;

struct DepthPrefilterPushConstants {
  uint32_t sourceDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t sourceSamplerId = 0u;
  std::array<uint32_t, GTAOPass::kViewDepthMipCount> outputTexIds{};
  uint32_t width = 1u;
  uint32_t height = 1u;
  uint32_t mipCount = GTAOPass::kViewDepthMipCount;
  uint32_t projectionType = 0u;
  float nearPlane = 0.01f;
  float farPlane = 1000.0f;
};
static_assert(sizeof(DepthPrefilterPushConstants) <= 128u);

struct MainPushConstants {
  std::array<uint32_t, GTAOPass::kViewDepthMipCount> depthTexIds{};
  uint32_t normalTexId = kInvalidTextureBindlessIndex;
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
  uint32_t depthTexId = kInvalidTextureBindlessIndex;
  uint32_t width = 1u;
  uint32_t height = 1u;
  uint32_t passIndex = 0u;
};
static_assert(sizeof(DenoisePushConstants) <= 128u);

template <typename T>
std::span<const std::byte> copyPushConstants(std::array<std::byte, 128> &dst,
                                             const T &src) {
  static_assert(sizeof(T) <= 128u);
  std::memcpy(dst.data(), &src, sizeof(T));
  return std::span<const std::byte>(dst.data(), sizeof(T));
}

[[nodiscard]] uint32_t divRoundUp(uint32_t value, uint32_t divisor) noexcept {
  return (value + divisor - 1u) / divisor;
}

[[nodiscard]] uint32_t levelDimension(uint32_t value, uint32_t level) noexcept {
  return std::max(1u, value >> std::min(level, 31u));
}

[[nodiscard]] uint64_t bytesPerPixel(Format format) noexcept {
  switch (format) {
  case Format::R32_FLOAT:
  case Format::R32_UINT:
  case Format::RG16_FLOAT:
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
  case Format::D32_FLOAT:
    return 4u;
  case Format::D16_UNORM:
    return 2u;
  case Format::RG32_FLOAT:
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

[[nodiscard]] uint64_t textureBytes(Format format, uint32_t width,
                                    uint32_t height) noexcept {
  return static_cast<uint64_t>(std::max(width, 1u)) *
         static_cast<uint64_t>(std::max(height, 1u)) * bytesPerPixel(format);
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
  }
  return 1.15f;
}

} // namespace

GTAOPass::GTAOPass(GPUDevice &gpu, RuntimeOpaqueShaderConfig config)
    : gpu_(gpu), config_(std::move(config)) {}

GTAOPass::~GTAOPass() { destroyResources(); }

bool GTAOPass::isEnabled(const FrameBuildContext &ctx) const {
  const RenderSettings::AmbientOcclusionSettings ao =
      resolvedAmbientOcclusionSettings(ctx.frame);
  return ao.active && nuri::isValid(ctx.shared.sceneDepthTexture) &&
         nuri::isValid(ctx.shared.normalTexture) &&
         nuri::isValid(ctx.shared.normalGraphTexture) &&
         nuri::isValid(ctx.shared.ambientOcclusionTexture);
}

Result<bool, std::string> GTAOPass::prepare(FrameBuildContext &ctx) {
  (void)ctx;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GTAOPass::ensureResources(FrameBuildContext &ctx) {
  if (!nuri::isValid(pointClampSampler_)) {
    auto samplerResult =
        gpu_.createSampler(SamplerDesc{.minFilter = SamplerFilter::Nearest,
                                       .magFilter = SamplerFilter::Nearest,
                                       .mipMode = SamplerMipMode::Disabled,
                                       .wrapU = SamplerWrapMode::Clamp,
                                       .wrapV = SamplerWrapMode::Clamp,
                                       .wrapW = SamplerWrapMode::Clamp},
                           "gtao_point_clamp");
    if (samplerResult.hasError()) {
      return Result<bool, std::string>::makeError(samplerResult.error());
    }
    pointClampSampler_ = samplerResult.value();
  }

  if (!nuri::isValid(depthPrefilterPipeline_) ||
      !nuri::isValid(mainPipeline_) || !nuri::isValid(denoisePipeline_)) {
    const std::filesystem::path shaderDir = resolveShaderDir(config_);
    depthPrefilterShader_ = Shader::create("gtao_depth_prefilter", gpu_);
    mainShader_ = Shader::create("gtao_main", gpu_);
    denoiseShader_ = Shader::create("gtao_denoise", gpu_);
    if (!depthPrefilterShader_ || !mainShader_ || !denoiseShader_) {
      return Result<bool, std::string>::makeError(
          "GTAOPass::ensureResources: failed to create shader objects");
    }

    auto depthShaderResult = depthPrefilterShader_->compileFromFile(
        (shaderDir / "gtao_depth_prefilter.comp").string(),
        ShaderStage::Compute);
    auto mainShaderResult = mainShader_->compileFromFile(
        (shaderDir / "gtao_main.comp").string(), ShaderStage::Compute);
    auto denoiseShaderResult = denoiseShader_->compileFromFile(
        (shaderDir / "gtao_denoise.comp").string(), ShaderStage::Compute);
    if (depthShaderResult.hasError() || mainShaderResult.hasError() ||
        denoiseShaderResult.hasError()) {
      const std::string error =
          depthShaderResult.hasError()
              ? depthShaderResult.error()
              : (mainShaderResult.hasError() ? mainShaderResult.error()
                                             : denoiseShaderResult.error());
      return Result<bool, std::string>::makeError(error);
    }
    depthPrefilterShaderHandle_ = depthShaderResult.value();
    mainShaderHandle_ = mainShaderResult.value();
    denoiseShaderHandle_ = denoiseShaderResult.value();

    auto depthPipelineResult = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = depthPrefilterShaderHandle_},
        "gtao_depth_prefilter");
    auto mainPipelineResult = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = mainShaderHandle_}, "gtao_main");
    auto denoisePipelineResult = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = denoiseShaderHandle_},
        "gtao_denoise");
    if (depthPipelineResult.hasError() || mainPipelineResult.hasError() ||
        denoisePipelineResult.hasError()) {
      const std::string error =
          depthPipelineResult.hasError()
              ? depthPipelineResult.error()
              : (mainPipelineResult.hasError() ? mainPipelineResult.error()
                                               : denoisePipelineResult.error());
      return Result<bool, std::string>::makeError(error);
    }
    depthPrefilterPipeline_ = depthPipelineResult.value();
    mainPipeline_ = mainPipelineResult.value();
    denoisePipeline_ = denoisePipelineResult.value();
  }

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
    for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
      auto textureResult = gpu_.createTexture(
          makeStorageSampledTextureDesc(Format::R32_FLOAT,
                                        levelDimension(width, level),
                                        levelDimension(height, level)),
          "gtao_view_depth_" + std::to_string(slot) + "_" +
              std::to_string(level));
      if (textureResult.hasError()) {
        destroyScratchTextures();
        return Result<bool, std::string>::makeError(textureResult.error());
      }
      textures.viewDepthMips[level] = textureResult.value();
    }

    auto rawResult = gpu_.createTexture(
        makeStorageSampledTextureDesc(kFrameCompositionAmbientOcclusionFormat,
                                      width, height),
        "gtao_raw_ao_bent_" + std::to_string(slot));
    if (rawResult.hasError()) {
      destroyScratchTextures();
      return Result<bool, std::string>::makeError(rawResult.error());
    }
    textures.rawAmbientOcclusion = rawResult.value();

    auto scratchResult = gpu_.createTexture(
        makeStorageSampledTextureDesc(kFrameCompositionAmbientOcclusionFormat,
                                      width, height),
        "gtao_denoise_scratch_" + std::to_string(slot));
    if (scratchResult.hasError()) {
      destroyScratchTextures();
      return Result<bool, std::string>::makeError(scratchResult.error());
    }
    textures.denoiseScratch = scratchResult.value();
  }

  scratchWidth_ = width;
  scratchHeight_ = height;
  scratchRingCount_ = ringCount;
  return Result<bool, std::string>::makeResult(true);
}

void GTAOPass::destroyResources() {
  destroyScratchTextures();
  if (nuri::isValid(depthPrefilterPipeline_)) {
    gpu_.destroyComputePipeline(depthPrefilterPipeline_);
    depthPrefilterPipeline_ = {};
  }
  if (nuri::isValid(mainPipeline_)) {
    gpu_.destroyComputePipeline(mainPipeline_);
    mainPipeline_ = {};
  }
  if (nuri::isValid(denoisePipeline_)) {
    gpu_.destroyComputePipeline(denoisePipeline_);
    denoisePipeline_ = {};
  }
  if (nuri::isValid(pointClampSampler_)) {
    gpu_.destroySampler(pointClampSampler_);
    pointClampSampler_ = {};
  }
  depthPrefilterShader_.reset();
  mainShader_.reset();
  denoiseShader_.reset();
  depthPrefilterShaderHandle_ = {};
  mainShaderHandle_ = {};
  denoiseShaderHandle_ = {};
}

void GTAOPass::destroyScratchTextures() {
  for (FrameScratchTextures &textures : scratchTextures_) {
    for (TextureHandle &texture : textures.viewDepthMips) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
        texture = {};
      }
    }
    if (nuri::isValid(textures.rawAmbientOcclusion)) {
      gpu_.destroyTexture(textures.rawAmbientOcclusion);
      textures.rawAmbientOcclusion = {};
    }
    if (nuri::isValid(textures.denoiseScratch)) {
      gpu_.destroyTexture(textures.denoiseScratch);
      textures.denoiseScratch = {};
    }
  }
  scratchTextures_.clear();
  scratchWidth_ = 0u;
  scratchHeight_ = 0u;
  scratchRingCount_ = 0u;
}

GTAOPass::FrameScratchTextures *
GTAOPass::currentScratch(uint64_t frameIndex) noexcept {
  if (scratchTextures_.empty()) {
    return nullptr;
  }
  return &scratchTextures_[static_cast<size_t>(frameIndex %
                                               scratchTextures_.size())];
}

Result<bool, std::string> GTAOPass::build(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  auto resourceResult = ensureResources(ctx);
  if (resourceResult.hasError()) {
    ctx.frame.metrics.ambientOcclusion.active = false;
    ctx.frame.metrics.ambientOcclusion.disabledReason =
        AmbientOcclusionDisabledReason::Unsupported;
    return resourceResult;
  }

  FrameScratchTextures *scratch = currentScratch(ctx.frame.frameIndex);
  if (scratch == nullptr) {
    return Result<bool, std::string>::makeResult(false);
  }

  const RenderSettings::AmbientOcclusionSettings ao =
      resolvedAmbientOcclusionSettings(ctx.frame);
  const uint32_t pointSamplerId =
      gpu_.getSamplerBindlessIndex(pointClampSampler_);
  const TextureDimensions dimensions =
      gpu_.getTextureDimensions(ctx.shared.sceneDepthTexture);
  const uint32_t width = std::max(dimensions.width, 1u);
  const uint32_t height = std::max(dimensions.height, 1u);
  const DispatchSize dispatch{.x = divRoundUp(width, kGTAOWorkgroupSizeX),
                              .y = divRoundUp(height, kGTAOWorkgroupSizeY),
                              .z = 1u};

  AmbientOcclusionFrameMetrics &metrics = ctx.frame.metrics.ambientOcclusion;
  const GpuTimingReport timingReport = gpu_.getLatestCompletedGpuTimingReport();
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
  metrics.sliceCount = ao.sliceCount;
  metrics.stepCount = ao.stepCount;
  metrics.strength = ao.strength;
  metrics.depthPrefilterPassCount = 1u;
  metrics.mainPassCount = 1u;
  metrics.denoisePassCount = ao.denoisePassCount;
  metrics.scalarAoAvailable = true;
  metrics.bentNormalAvailable = true;

  uint64_t depthBytes = 0u;
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    depthBytes += textureBytes(Format::R32_FLOAT, levelDimension(width, level),
                               levelDimension(height, level));
  }
  metrics.depthPrefilterTextureBytes = depthBytes;
  metrics.scratchTextureBytes =
      textureBytes(kFrameCompositionAmbientOcclusionFormat, width, height) * 2u;
  metrics.textureCount += kViewDepthMipCount + 2u;
  metrics.totalTextureBytes +=
      metrics.depthPrefilterTextureBytes + metrics.scratchTextureBytes;

  const uint32_t sourceDepthTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.sceneDepthTexture);
  const uint32_t normalTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.normalTexture);
  const uint32_t rawTexId =
      gpu_.getTextureBindlessIndex(scratch->rawAmbientOcclusion);
  const uint32_t scratchTexId =
      gpu_.getTextureBindlessIndex(scratch->denoiseScratch);
  const uint32_t finalTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.ambientOcclusionTexture);
  if (sourceDepthTexId == kInvalidTextureBindlessIndex ||
      normalTexId == kInvalidTextureBindlessIndex ||
      rawTexId == kInvalidTextureBindlessIndex ||
      scratchTexId == kInvalidTextureBindlessIndex ||
      finalTexId == kInvalidTextureBindlessIndex) {
    metrics.active = false;
    metrics.disabledReason = AmbientOcclusionDisabledReason::MissingResources;
    return Result<bool, std::string>::makeResult(false);
  }

  DepthPrefilterPushConstants depthPc{
      .sourceDepthTexId = sourceDepthTexId,
      .sourceSamplerId = pointSamplerId,
      .width = width,
      .height = height,
      .mipCount = kViewDepthMipCount,
      .projectionType = static_cast<uint32_t>(ctx.frame.camera.projectionType),
      .nearPlane = ctx.frame.camera.nearPlane,
      .farPlane = ctx.frame.camera.farPlane,
  };
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    depthPc.outputTexIds[level] =
        gpu_.getTextureBindlessIndex(scratch->viewDepthMips[level]);
  }

  depthPrefilterDependencies_[0] = ctx.shared.sceneDepthTexture;
  depthPrefilterAccessModes_[0] = RenderGraphAccessMode::Read;
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    depthPrefilterDependencies_[level + 1u] = scratch->viewDepthMips[level];
    depthPrefilterAccessModes_[level + 1u] = RenderGraphAccessMode::Write;
  }
  depthPrefilterDispatches_[0] = ComputeDispatchItem{
      .pipeline = depthPrefilterPipeline_,
      .dispatch = dispatch,
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
  auto addDepth = ctx.graph.addGraphicsPass(depthPass);
  if (addDepth.hasError()) {
    return Result<bool, std::string>::makeError(addDepth.error());
  }

  MainPushConstants mainPc{
      .normalTexId = normalTexId,
      .outputTexId = rawTexId,
      .width = width,
      .height = height,
      .noiseIndex = 0u,
      .sliceCount = ao.sliceCount,
      .stepCount = ao.stepCount,
      .strengthBits = std::bit_cast<uint32_t>(ao.strength),
      .radiusBits =
          std::bit_cast<uint32_t>(ambientOcclusionPresetRadius(ao.preset)),
      .tanHalfFovY = std::tan(ctx.frame.camera.fovYRadians * 0.5f),
      .aspectRatio = ctx.frame.camera.aspectRatio,
      .orthoHeight = ctx.frame.camera.orthoHeight,
      .projectionType = static_cast<uint32_t>(ctx.frame.camera.projectionType),
  };
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    mainPc.depthTexIds[level] = depthPc.outputTexIds[level];
    mainDependencies_[level] = scratch->viewDepthMips[level];
    mainAccessModes_[level] = RenderGraphAccessMode::Read;
  }
  mainDependencies_[kViewDepthMipCount] = ctx.shared.normalTexture;
  mainAccessModes_[kViewDepthMipCount] = RenderGraphAccessMode::Read;
  mainDependencies_[kViewDepthMipCount + 1u] = scratch->rawAmbientOcclusion;
  mainAccessModes_[kViewDepthMipCount + 1u] = RenderGraphAccessMode::Write;
  mainDispatches_[0] = ComputeDispatchItem{
      .pipeline = mainPipeline_,
      .dispatch = dispatch,
      .pushConstants = copyPushConstants(mainPushBytes_, mainPc),
      .dependencyTextures = std::span<const TextureHandle>(
          mainDependencies_.data(), kViewDepthMipCount + 2u),
      .debugLabel = "GTAO Main",
      .debugColor = kGTAODebugColor,
  };

  RenderGraphGraphicsPassDesc mainPass{};
  mainPass.executionMode = RenderPassExecutionMode::ComputeOnly;
  mainPass.hasColorAttachment = false;
  mainPass.preDispatches = std::span<const ComputeDispatchItem>(
      mainDispatches_.data(), mainDispatches_.size());
  mainPass.dependencyTextures = std::span<const TextureHandle>(
      mainDependencies_.data(), kViewDepthMipCount + 2u);
  mainPass.dependencyTextureAccessModes =
      std::span<const RenderGraphAccessMode>(mainAccessModes_.data(),
                                             kViewDepthMipCount + 2u);
  mainPass.gpuTimingScope = GpuTimingScope::GTAO;
  mainPass.debugLabel = "GTAO Main Pass";
  mainPass.debugColor = kGTAODebugColor;
  auto addMain = ctx.graph.addGraphicsPass(mainPass);
  if (addMain.hasError()) {
    return Result<bool, std::string>::makeError(addMain.error());
  }

  TextureHandle denoiseSource = scratch->rawAmbientOcclusion;
  uint32_t denoiseSourceTexId = rawTexId;
  const uint32_t passCount = std::max(ao.denoisePassCount, 1u);
  for (uint32_t passIndex = 0u; passIndex < passCount; ++passIndex) {
    const bool finalPass = passIndex + 1u == passCount;
    TextureHandle outputTexture = finalPass ? ctx.shared.ambientOcclusionTexture
                                            : scratch->denoiseScratch;
    const uint32_t outputTexId = finalPass ? finalTexId : scratchTexId;
    DenoisePushConstants denoisePc{
        .sourceTexId = denoiseSourceTexId,
        .outputTexId = outputTexId,
        .depthTexId = depthPc.outputTexIds[0],
        .width = width,
        .height = height,
        .passIndex = passIndex,
    };

    denoiseDependencies_[0] = denoiseSource;
    denoiseDependencies_[1] = scratch->viewDepthMips[0];
    denoiseDependencies_[2] = outputTexture;
    denoiseAccessModes_[0] = RenderGraphAccessMode::Read;
    denoiseAccessModes_[1] = RenderGraphAccessMode::Read;
    denoiseAccessModes_[2] = RenderGraphAccessMode::Write;
    denoiseDispatches_[0] = ComputeDispatchItem{
        .pipeline = denoisePipeline_,
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
    auto addDenoise = ctx.graph.addGraphicsPass(denoisePass);
    if (addDenoise.hasError()) {
      return Result<bool, std::string>::makeError(addDenoise.error());
    }

    denoiseSource = outputTexture;
    denoiseSourceTexId = outputTexId;
  }

  auto importFinal = ctx.graph.importTexture(ctx.shared.ambientOcclusionTexture,
                                             "gtao_final_ambient_occlusion");
  if (importFinal.hasError()) {
    return Result<bool, std::string>::makeError(importFinal.error());
  }
  ctx.shared.ambientOcclusionGraphTexture = importFinal.value();
  ctx.frame.sharedResources.ambientOcclusionGraphTexture = importFinal.value();
  ctx.frame.metrics.ambientOcclusion.ambientOcclusionGraphPublished = true;
  return Result<bool, std::string>::makeResult(true);
}

GTAOFeature::GTAOFeature(GPUDevice &gpu, RuntimeOpaqueShaderConfig config)
    : pass_(gpu, std::move(config)) {}

Result<bool, std::string>
GTAOFeature::publishFrameData(FrameBuildContext &ctx) {
  const RenderSettings::AmbientOcclusionSettings ao =
      resolvedAmbientOcclusionSettings(ctx.frame);
  AmbientOcclusionFrameMetrics &metrics = ctx.frame.metrics.ambientOcclusion;
  metrics.enabled = ao.mode != AmbientOcclusionMode::Disabled;
  metrics.active = ao.active;
  metrics.activePreset = ao.preset;
  metrics.disabledReason = ao.disabledReason;
  metrics.strength = ao.strength;
  metrics.sliceCount = ao.sliceCount;
  metrics.stepCount = ao.stepCount;
  metrics.denoisePassCount = ao.denoisePassCount;
  if (ao.active) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::Normals |
        FrameTextureRequirementFlags::AmbientOcclusion;
  }
  return Result<bool, std::string>::makeResult(true);
}

std::span<RenderFeaturePass *const> GTAOFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
