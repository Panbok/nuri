#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/gtao_feature.h"

#include "nuri/core/profiling.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace nuri {
namespace {

constexpr uint32_t kInvalidTextureBindlessIndex = 0xffffffffu;
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
  uint32_t sceneDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t edgeTexId = kInvalidTextureBindlessIndex;
  uint32_t outputTexId = kInvalidTextureBindlessIndex;
  uint32_t pointSamplerId = 0u;
  uint32_t historySamplerId = 0u;
  uint32_t width = 1u;
  uint32_t height = 1u;
  uint32_t flags = 0u;
  uint32_t baseCurrentWeightBits = 0u;
  uint32_t rejectedCurrentWeightBits = 0u;
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

[[nodiscard]] uint64_t bytesPerPixel(Format format) noexcept {
  switch (format) {
  case Format::R8_UNORM:
    return 1u;
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
    NURI_LOG_WARNING(
        "GTAO texture byte metric requested unsupported format %u; using 1 "
        "byte per pixel sentinel",
        static_cast<uint32_t>(format));
    NURI_ASSERT(false, "Unsupported GTAO texture byte metric format %u",
                static_cast<uint32_t>(format));
    return 1u;
  }
  NURI_LOG_WARNING(
      "GTAO texture byte metric requested unknown format %u; using 1 byte per "
      "pixel sentinel",
      static_cast<uint32_t>(format));
  NURI_ASSERT(false, "Unknown GTAO texture byte metric format %u",
              static_cast<uint32_t>(format));
  return 1u;
}

[[nodiscard]] uint64_t textureBytes(Format format, uint32_t width,
                                    uint32_t height) noexcept {
  return static_cast<uint64_t>(std::max(width, 1u)) *
         static_cast<uint64_t>(std::max(height, 1u)) * bytesPerPixel(format);
}

void publishRequestedCapture(RenderFrameContext &frame, GPUDevice &gpu,
                             std::string_view name, TextureHandle texture,
                             RenderCaptureValueKind kind,
                             RenderCaptureLifetimeClass lifetime,
                             std::string_view colorSpace,
                             std::string_view compareProfile,
                             std::string_view producerPassLabel) {
  if (!isRenderCaptureRequested(frame, name) || !nuri::isValid(texture)) {
    return;
  }
  frame.captureRegistry.publish(RenderCapturePoint{
      .name = name,
      .version = 1u,
      .texture = texture,
      .format = gpu.getTextureFormat(texture),
      .dimensions = gpu.getTextureDimensions(texture),
      .frameIndex = frame.frameIndex,
      .mip = 0u,
      .layer = 0u,
      .kind = kind,
      .lifetime = lifetime,
      .colorSpace = colorSpace,
      .defaultCompareProfile = compareProfile,
      .producerPassLabel = producerPassLabel,
      .debugLabel = name,
  });
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
    : gpu_(gpu), config_(std::move(config)) {}

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

  if (!nuri::isValid(linearClampSampler_)) {
    auto samplerResult =
        gpu_.createSampler(SamplerDesc{.minFilter = SamplerFilter::Linear,
                                       .magFilter = SamplerFilter::Linear,
                                       .mipMode = SamplerMipMode::Disabled,
                                       .wrapU = SamplerWrapMode::Clamp,
                                       .wrapV = SamplerWrapMode::Clamp,
                                       .wrapW = SamplerWrapMode::Clamp},
                           "gtao_linear_clamp");
    if (samplerResult.hasError()) {
      return Result<bool, std::string>::makeError(samplerResult.error());
    }
    linearClampSampler_ = samplerResult.value();
  }

  if (!nuri::isValid(depthPrefilterPipeline_) ||
      !nuri::isValid(edgePipeline_) || !nuri::isValid(mainPipeline_) ||
      !nuri::isValid(denoisePipeline_) || !nuri::isValid(temporalPipeline_)) {
    const std::filesystem::path shaderDir = resolveShaderDir(config_);
    depthPrefilterShader_ = Shader::create("gtao_depth_prefilter", gpu_);
    edgeShader_ = Shader::create("gtao_edges", gpu_);
    mainShader_ = Shader::create("gtao_main", gpu_);
    denoiseShader_ = Shader::create("gtao_denoise", gpu_);
    temporalShader_ = Shader::create("gtao_temporal", gpu_);
    if (!depthPrefilterShader_ || !edgeShader_ || !mainShader_ ||
        !denoiseShader_ || !temporalShader_) {
      return Result<bool, std::string>::makeError(
          "GTAOPass::ensureResources: failed to create shader objects");
    }

    auto depthShaderResult = depthPrefilterShader_->compileFromFile(
        (shaderDir / "gtao_depth_prefilter.comp").string(),
        ShaderStage::Compute);
    auto edgeShaderResult = edgeShader_->compileFromFile(
        (shaderDir / "gtao_edges.comp").string(), ShaderStage::Compute);
    auto mainShaderResult = mainShader_->compileFromFile(
        (shaderDir / "gtao_main.comp").string(), ShaderStage::Compute);
    auto denoiseShaderResult = denoiseShader_->compileFromFile(
        (shaderDir / "gtao_denoise.comp").string(), ShaderStage::Compute);
    auto temporalShaderResult = temporalShader_->compileFromFile(
        (shaderDir / "gtao_temporal.comp").string(), ShaderStage::Compute);
    if (depthShaderResult.hasError() || edgeShaderResult.hasError() ||
        mainShaderResult.hasError() || denoiseShaderResult.hasError() ||
        temporalShaderResult.hasError()) {
      const auto getFirstShaderError = [&]() -> const std::string & {
        if (depthShaderResult.hasError()) {
          return depthShaderResult.error();
        }
        if (edgeShaderResult.hasError()) {
          return edgeShaderResult.error();
        }
        if (mainShaderResult.hasError()) {
          return mainShaderResult.error();
        }
        if (denoiseShaderResult.hasError()) {
          return denoiseShaderResult.error();
        }
        return temporalShaderResult.error();
      };
      const std::string error = getFirstShaderError();
      return Result<bool, std::string>::makeError(error);
    }
    depthPrefilterShaderHandle_ = depthShaderResult.value();
    edgeShaderHandle_ = edgeShaderResult.value();
    mainShaderHandle_ = mainShaderResult.value();
    denoiseShaderHandle_ = denoiseShaderResult.value();
    temporalShaderHandle_ = temporalShaderResult.value();

    auto depthPipelineResult = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = depthPrefilterShaderHandle_},
        "gtao_depth_prefilter");
    auto edgePipelineResult = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = edgeShaderHandle_}, "gtao_edges");
    auto mainPipelineResult = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = mainShaderHandle_}, "gtao_main");
    auto denoisePipelineResult = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = denoiseShaderHandle_},
        "gtao_denoise");
    auto temporalPipelineResult = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = temporalShaderHandle_},
        "gtao_temporal");
    if (depthPipelineResult.hasError() || edgePipelineResult.hasError() ||
        mainPipelineResult.hasError() || denoisePipelineResult.hasError() ||
        temporalPipelineResult.hasError()) {
      const std::string error =
          depthPipelineResult.hasError()
              ? depthPipelineResult.error()
              : (edgePipelineResult.hasError()
                     ? edgePipelineResult.error()
                     : (mainPipelineResult.hasError()
                            ? mainPipelineResult.error()
                            : (denoisePipelineResult.hasError()
                                   ? denoisePipelineResult.error()
                                   : temporalPipelineResult.error())));
      return Result<bool, std::string>::makeError(error);
    }
    depthPrefilterPipeline_ = depthPipelineResult.value();
    edgePipeline_ = edgePipelineResult.value();
    mainPipeline_ = mainPipelineResult.value();
    denoisePipeline_ = denoisePipelineResult.value();
    temporalPipeline_ = temporalPipelineResult.value();
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

    auto edgeResult = gpu_.createTexture(
        makeStorageSampledTextureDesc(Format::R32_FLOAT, width, height),
        "gtao_edges_" + std::to_string(slot));
    if (edgeResult.hasError()) {
      destroyScratchTextures();
      return Result<bool, std::string>::makeError(edgeResult.error());
    }
    textures.edges = edgeResult.value();

    auto rawResult = gpu_.createTexture(
        makeStorageSampledTextureDesc(kFrameCompositionAmbientOcclusionFormat,
                                      width, height),
        "gtao_raw_ao_" + std::to_string(slot));
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
  if (nuri::isValid(edgePipeline_)) {
    gpu_.destroyComputePipeline(edgePipeline_);
    edgePipeline_ = {};
  }
  if (nuri::isValid(mainPipeline_)) {
    gpu_.destroyComputePipeline(mainPipeline_);
    mainPipeline_ = {};
  }
  if (nuri::isValid(denoisePipeline_)) {
    gpu_.destroyComputePipeline(denoisePipeline_);
    denoisePipeline_ = {};
  }
  if (nuri::isValid(temporalPipeline_)) {
    gpu_.destroyComputePipeline(temporalPipeline_);
    temporalPipeline_ = {};
  }
  if (nuri::isValid(pointClampSampler_)) {
    gpu_.destroySampler(pointClampSampler_);
    pointClampSampler_ = {};
  }
  if (nuri::isValid(linearClampSampler_)) {
    gpu_.destroySampler(linearClampSampler_);
    linearClampSampler_ = {};
  }
  depthPrefilterShader_.reset();
  edgeShader_.reset();
  mainShader_.reset();
  denoiseShader_.reset();
  temporalShader_.reset();
  depthPrefilterShaderHandle_ = {};
  edgeShaderHandle_ = {};
  mainShaderHandle_ = {};
  denoiseShaderHandle_ = {};
  temporalShaderHandle_ = {};
}

void GTAOPass::destroyScratchTextures() {
  for (FrameScratchTextures &textures : scratchTextures_) {
    for (TextureHandle &texture : textures.viewDepthMips) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
        texture = {};
      }
    }
    if (nuri::isValid(textures.edges)) {
      gpu_.destroyTexture(textures.edges);
      textures.edges = {};
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
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  const bool taaSelected =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
      AntiAliasingMode::TAA;
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
  metrics.edgeTextureBytes = textureBytes(Format::R32_FLOAT, width, height);
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
  const uint32_t edgeTexId = gpu_.getTextureBindlessIndex(scratch->edges);
  const uint32_t rawTexId =
      gpu_.getTextureBindlessIndex(scratch->rawAmbientOcclusion);
  const uint32_t scratchTexId =
      gpu_.getTextureBindlessIndex(scratch->denoiseScratch);
  const uint32_t finalTexId =
      gpu_.getTextureBindlessIndex(ctx.shared.ambientOcclusionTexture);
  if (sourceDepthTexId == kInvalidTextureBindlessIndex ||
      normalTexId == kInvalidTextureBindlessIndex ||
      edgeTexId == kInvalidTextureBindlessIndex ||
      rawTexId == kInvalidTextureBindlessIndex ||
      scratchTexId == kInvalidTextureBindlessIndex ||
      finalTexId == kInvalidTextureBindlessIndex) {
    metrics.active = false;
    metrics.disabledReason = AmbientOcclusionDisabledReason::MissingResources;
    return Result<bool, std::string>::makeResult(false);
  }

  bool temporalActive =
      ao.temporalAccumulation && taaSelected && ctx.frame.camera.historyValid &&
      ctx.frame.camera.temporalDataValid &&
      nuri::isValid(ctx.shared.previousAmbientOcclusionTexture) &&
      nuri::isValid(ctx.shared.motionVectorTexture) &&
      nuri::isValid(ctx.shared.motionVectorGraphTexture);
  uint32_t previousAoTexId = kInvalidTextureBindlessIndex;
  uint32_t motionVectorTexId = kInvalidTextureBindlessIndex;
  uint32_t pointSamplerId = kInvalidTextureBindlessIndex;
  uint32_t linearSamplerId = kInvalidTextureBindlessIndex;
  if (temporalActive) {
    previousAoTexId = gpu_.getTextureBindlessIndex(
        ctx.shared.previousAmbientOcclusionTexture);
    motionVectorTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.motionVectorTexture);
    pointSamplerId = gpu_.getSamplerBindlessIndex(pointClampSampler_);
    linearSamplerId = gpu_.getSamplerBindlessIndex(linearClampSampler_);
    temporalActive = previousAoTexId != kInvalidTextureBindlessIndex &&
                     motionVectorTexId != kInvalidTextureBindlessIndex &&
                     pointSamplerId != kInvalidTextureBindlessIndex &&
                     linearSamplerId != kInvalidTextureBindlessIndex;
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
    depthTexIds[level] =
        gpu_.getTextureBindlessIndex(scratch->viewDepthMips[level]);
    if (depthTexIds[level] == kInvalidTextureBindlessIndex) {
      metrics.active = false;
      metrics.disabledReason = AmbientOcclusionDisabledReason::MissingResources;
      return Result<bool, std::string>::makeResult(false);
    }
    depthPc.outputTexIds[level] = depthTexIds[level];
  }

  depthPrefilterDependencies_[0] = ctx.shared.sceneDepthTexture;
  depthPrefilterAccessModes_[0] = RenderGraphAccessMode::Read;
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    depthPrefilterDependencies_[level + 1u] = scratch->viewDepthMips[level];
    depthPrefilterAccessModes_[level + 1u] = RenderGraphAccessMode::Write;
  }
  depthPrefilterDispatches_[0] = ComputeDispatchItem{
      .pipeline = depthPrefilterPipeline_,
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
  auto addDepth = ctx.graph.addGraphicsPass(depthPass);
  if (addDepth.hasError()) {
    return Result<bool, std::string>::makeError(addDepth.error());
  }

  const EdgePushConstants edgePc{
      .depthTexId = depthTexIds[0],
      .outputTexId = edgeTexId,
      .width = width,
      .height = height,
  };
  edgeDependencies_[0] = scratch->viewDepthMips[0];
  edgeAccessModes_[0] = RenderGraphAccessMode::Read;
  edgeDependencies_[1] = scratch->edges;
  edgeAccessModes_[1] = RenderGraphAccessMode::Write;
  edgeDispatches_[0] = ComputeDispatchItem{
      .pipeline = edgePipeline_,
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
  auto addEdge = ctx.graph.addGraphicsPass(edgePass);
  if (addEdge.hasError()) {
    return Result<bool, std::string>::makeError(addEdge.error());
  }

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
    mainDependencies_[level] = scratch->viewDepthMips[level];
    mainAccessModes_[level] = RenderGraphAccessMode::Read;
  }
  mainDependencies_[kViewDepthMipCount] = ctx.shared.normalTexture;
  mainAccessModes_[kViewDepthMipCount] = RenderGraphAccessMode::Read;
  mainDependencies_[kViewDepthMipCount + 1u] = scratch->edges;
  mainAccessModes_[kViewDepthMipCount + 1u] = RenderGraphAccessMode::Read;
  mainDependencies_[kViewDepthMipCount + 2u] = scratch->rawAmbientOcclusion;
  mainAccessModes_[kViewDepthMipCount + 2u] = RenderGraphAccessMode::Write;
  mainDispatches_[0] = ComputeDispatchItem{
      .pipeline = mainPipeline_,
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
  auto addMain = ctx.graph.addGraphicsPass(mainPass);
  if (addMain.hasError()) {
    return Result<bool, std::string>::makeError(addMain.error());
  }

  TextureHandle denoiseSource = scratch->rawAmbientOcclusion;
  uint32_t denoiseSourceTexId = rawTexId;
  for (uint32_t passIndex = 0u; passIndex < denoisePassCount; ++passIndex) {
    const bool finalPass =
        !temporalActive && passIndex + 1u == denoisePassCount;
    TextureHandle outputTexture =
        finalPass ? ctx.shared.ambientOcclusionTexture
                  : (passIndex % 2u == 0u ? scratch->denoiseScratch
                                          : scratch->rawAmbientOcclusion);
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
    denoiseDependencies_[1] = scratch->edges;
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

  if (temporalActive) {
    const TemporalPushConstants temporalPc{
        .currentTexId = denoiseSourceTexId,
        .historyTexId = previousAoTexId,
        .motionVectorTexId = motionVectorTexId,
        .sceneDepthTexId = sourceDepthTexId,
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
    };
    temporalDependencies_[0] = denoiseSource;
    temporalDependencies_[1] = ctx.shared.previousAmbientOcclusionTexture;
    temporalDependencies_[2] = ctx.shared.motionVectorTexture;
    temporalDependencies_[3] = ctx.shared.sceneDepthTexture;
    temporalDependencies_[4] = scratch->edges;
    temporalDependencies_[5] = ctx.shared.ambientOcclusionTexture;
    for (uint32_t i = 0u; i < 5u; ++i) {
      temporalAccessModes_[i] = RenderGraphAccessMode::Read;
    }
    temporalAccessModes_[5] = RenderGraphAccessMode::Write;
    temporalDispatches_[0] = ComputeDispatchItem{
        .pipeline = temporalPipeline_,
        .dispatch = dispatch,
        .pushConstants = copyPushConstants(temporalPushBytes_, temporalPc),
        .dependencyTextures =
            std::span<const TextureHandle>(temporalDependencies_.data(), 6u),
        .debugLabel = "GTAO Temporal",
        .debugColor = kGTAODebugColor,
    };

    RenderGraphGraphicsPassDesc temporalPass{};
    temporalPass.executionMode = RenderPassExecutionMode::ComputeOnly;
    temporalPass.hasColorAttachment = false;
    temporalPass.preDispatches = std::span<const ComputeDispatchItem>(
        temporalDispatches_.data(), temporalDispatches_.size());
    temporalPass.dependencyTextures =
        std::span<const TextureHandle>(temporalDependencies_.data(), 6u);
    temporalPass.dependencyTextureAccessModes =
        std::span<const RenderGraphAccessMode>(temporalAccessModes_.data(), 6u);
    temporalPass.gpuTimingScope = GpuTimingScope::GTAO;
    temporalPass.debugLabel = "GTAO Temporal Pass";
    temporalPass.debugColor = kGTAODebugColor;
    auto addTemporal = ctx.graph.addGraphicsPass(temporalPass);
    if (addTemporal.hasError()) {
      return Result<bool, std::string>::makeError(addTemporal.error());
    }
  }

  auto importFinal = ctx.graph.importTexture(ctx.shared.ambientOcclusionTexture,
                                             "gtao_final_ambient_occlusion");
  if (importFinal.hasError()) {
    return Result<bool, std::string>::makeError(importFinal.error());
  }
  ctx.shared.ambientOcclusionGraphTexture = importFinal.value();
  ctx.frame.sharedResources.ambientOcclusionGraphTexture = importFinal.value();
  ctx.frame.metrics.ambientOcclusion.ambientOcclusionGraphPublished = true;
  publishRequestedCapture(ctx.frame, gpu_, "ambient_occlusion",
                          ctx.shared.ambientOcclusionTexture,
                          RenderCaptureValueKind::Scalar,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "linear_scalar", "scalar", "GTAO Denoise Final Pass");
  return Result<bool, std::string>::makeResult(true);
}

GTAOFeature::GTAOFeature(GPUDevice &gpu, RuntimeOpaqueShaderConfig config)
    : pass_(gpu, std::move(config)) {}

Result<bool, std::string>
GTAOFeature::publishFrameData(FrameBuildContext &ctx) {
  const RenderSettings::AmbientOcclusionSettings ao =
      resolvedAmbientOcclusionSettings(ctx.frame);
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  const bool taaSelected =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
      AntiAliasingMode::TAA;
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
  pass_.observeTemporalPolicy(ao);
  if (ao.active) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::Normals |
        FrameTextureRequirementFlags::AmbientOcclusion;
    if (ao.temporalAccumulation && taaSelected &&
        ctx.frame.camera.historyValid && ctx.frame.camera.temporalDataValid) {
      ctx.shared.textureRequirements |=
          FrameTextureRequirementFlags::MotionVectors;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

std::span<RenderFeaturePass *const> GTAOFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
