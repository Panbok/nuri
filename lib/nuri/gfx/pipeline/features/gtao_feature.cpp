#include "nuri/gfx/pipeline/features/gtao_feature.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/render_capture.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include <bit>
#include <cmath>
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
constexpr size_t kEdges = GTAOPass::kViewDepthMipCount;
constexpr size_t kRawAmbientOcclusion = kEdges + 1u;
constexpr size_t kDenoise = kEdges + 2u;
constexpr size_t kTransientTextureCount = kEdges + 3u;
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
template <typename T> std::span<const std::byte> pushConstants(const T &value) {
  return std::as_bytes(std::span(&value, 1u));
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
Result<bool, std::string> ensureCaptureTexture(GPUDevice &gpu,
                                               TextureHandle &texture,
                                               const TextureDesc &desc,
                                               std::string_view debugName) {
  if (nuri::isValid(texture) && gpu.getTextureFormat(texture) == desc.format &&
      gpu.getTextureDimensions(texture).width == desc.dimensions.width &&
      gpu.getTextureDimensions(texture).height == desc.dimensions.height) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (nuri::isValid(texture))
    gpu.destroyTexture(texture);
  texture = {};
  auto result = gpu.createTexture(desc, debugName);
  if (result.hasError())
    return Result<bool, std::string>::makeError(result.error());
  texture = result.value();
  return Result<bool, std::string>::makeResult(true);
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
  switch (preset) {
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
    auto compiled =
        compileShaderFile(gpu_, spec.name, (shaderDir / spec.file).string(),
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

Result<bool, std::string> GTAOPass::ensureResources(FrameBuildContext &ctx) {
  if (!initializationError_.empty())
    return Result<bool, std::string>::makeError(initializationError_);
  const TextureDimensions sceneDimensions = gpu_.getTextureDimensions(
      ctx.shared[FrameTextureSlot::SceneDepth].texture);
  const uint32_t width = std::max(sceneDimensions.width, 1u);
  const uint32_t height = std::max(sceneDimensions.height, 1u);
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

void GTAOPass::destroyResources() {
  if (nuri::isValid(reconstructedNormalDebugTexture_))
    gpu_.destroyTexture(reconstructedNormalDebugTexture_);
  reconstructedNormalDebugTexture_ = {};
  for (TextureHandle &texture : captureTextures_) {
    if (nuri::isValid(texture))
      gpu_.destroyTexture(texture);
    texture = {};
  }
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

Result<bool, std::string> GTAOPass::build(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  auto resourceResult = ensureResources(ctx);
  if (resourceResult.hasError()) {
    ctx.frame.metrics.ambientOcclusion.active = false;
    ctx.frame.metrics.ambientOcclusion.disabledReason =
        AmbientOcclusionDisabledReason::Unsupported;
    return resourceResult;
  }
  const AmbientOcclusionExecutionPlan &plan = ctx.frame.ambientOcclusion;
  const RenderSettings::AmbientOcclusionSettings &ao =
      ctx.frame.settings.ambientOcclusion;
  const TextureDimensions dimensions = gpu_.getTextureDimensions(
      ctx.shared[FrameTextureSlot::SceneDepth].texture);
  const uint32_t width = std::max(dimensions.width, 1u);
  const uint32_t height = std::max(dimensions.height, 1u);
  const bool halfResolution =
      plan.workingResolution == AmbientOcclusionWorkingResolution::Half;
  const uint32_t workingWidth = plan.workingWidth;
  const uint32_t workingHeight = plan.workingHeight;
  std::array<RenderGraphTextureId, kTransientTextureCount> transientTextures{};
  const auto createTransient =
      [&](size_t index, const TextureDesc &desc,
          std::string_view name) -> Result<bool, std::string> {
    auto result = ctx.graph.createTransientTexture(desc, name);
    if (result.hasError())
      return Result<bool, std::string>::makeError(result.error());
    transientTextures[index] = result.value();
    return Result<bool, std::string>::makeResult(true);
  };
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    auto result =
        createTransient(level,
                        makeStorageSampledTextureDesc(
                            Format::R32_FLOAT, levelDimension(width, level),
                            levelDimension(height, level)),
                        "gtao_view_depth");
    if (result.hasError())
      return result;
  }
  constexpr std::array transientFormats{
      Format::R8_UNORM, kFrameCompositionAmbientOcclusionFormat,
      kFrameCompositionAmbientOcclusionFormat};
  constexpr std::array<std::string_view, 3> transientNames{
      "gtao_edges", "gtao_raw_ao", "gtao_denoise"};
  for (size_t index = 0u; index < transientFormats.size(); ++index) {
    const bool fullResolution = index == 0u;
    auto result = createTransient(
        kEdges + index,
        makeStorageSampledTextureDesc(transientFormats[index],
                                      fullResolution ? width : workingWidth,
                                      fullResolution ? height : workingHeight),
        transientNames[index]);
    if (result.hasError())
      return result;
  }
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
    metrics.gpuTimeMs = timingReport[GpuTimingScope::GTAO].timeMs;
    metrics.gpuTimingSourceFrameIndex =
        timingReport[GpuTimingScope::GTAO].sourceFrameIndex;
    metrics.gpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAOPrefilterEdges)) {
    metrics.prefilterEdgesGpuTimeMs =
        timingReport[GpuTimingScope::GTAOPrefilterEdges].timeMs;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAOMain)) {
    metrics.mainGpuTimeMs = timingReport[GpuTimingScope::GTAOMain].timeMs;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAODenoise)) {
    metrics.denoiseGpuTimeMs = timingReport[GpuTimingScope::GTAODenoise].timeMs;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAOUpscale)) {
    metrics.upscaleGpuTimeMs = timingReport[GpuTimingScope::GTAOUpscale].timeMs;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::GTAOTemporal)) {
    metrics.temporalGpuTimeMs =
        timingReport[GpuTimingScope::GTAOTemporal].timeMs;
    metrics.upscaleGpuTimeMs =
        timingReport[GpuTimingScope::GTAOTemporal].timeMs;
  }
  if (hasGpuTimingScope(timingReport, GpuTimingScope::OpaqueNormal)) {
    metrics.inputGpuTimeMs = timingReport[GpuTimingScope::OpaqueNormal].timeMs;
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
  constexpr uint64_t allocationCount = 1u;
  metrics.depthPrefilterTextureBytes = depthBytes;
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
  const uint32_t sourceDepthTexId = gpu_.getTextureBindlessIndex(
      ctx.shared[FrameTextureSlot::SceneDepth].texture);
  const uint32_t normalTexId =
      nuri::isValid(ctx.shared[FrameTextureSlot::Normal].texture)
          ? gpu_.getTextureBindlessIndex(
                ctx.shared[FrameTextureSlot::Normal].texture)
          : sourceDepthTexId;
  bool temporalActive =
      plan.temporal && hasTemporalCameraContinuity(ctx.frame.camera) &&
      ctx.frame.camera.temporalDataValid &&
      nuri::isValid(
          ctx.shared[FrameHistoryTextureSlot::PreviousAmbientOcclusion]
              .texture) &&
      nuri::isValid(ctx.shared[FrameTextureSlot::MotionVector].texture) &&
      nuri::isValid(ctx.shared[FrameTextureSlot::MotionVector].graph) &&
      nuri::isValid(ctx.shared[FrameTextureSlot::ReactiveMask].texture) &&
      nuri::isValid(ctx.shared[FrameTextureSlot::ReactiveMask].graph) &&
      nuri::isValid(
          ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture);
  uint32_t previousAoTexId = kInvalidTextureBindlessIndex;
  uint32_t motionVectorTexId = kInvalidTextureBindlessIndex;
  uint32_t reactiveMaskTexId = kInvalidTextureBindlessIndex;
  uint32_t previousSceneDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t pointSamplerId = kInvalidTextureBindlessIndex;
  uint32_t linearSamplerId = kInvalidTextureBindlessIndex;
  if (temporalActive) {
    previousAoTexId = gpu_.getTextureBindlessIndex(
        ctx.shared[FrameHistoryTextureSlot::PreviousAmbientOcclusion].texture);
    motionVectorTexId = gpu_.getTextureBindlessIndex(
        ctx.shared[FrameTextureSlot::MotionVector].texture);
    reactiveMaskTexId = gpu_.getTextureBindlessIndex(
        ctx.shared[FrameTextureSlot::ReactiveMask].texture);
    previousSceneDepthTexId = gpu_.getTextureBindlessIndex(
        ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture);
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
  metrics.temporalHistoryValid = nuri::isValid(
      ctx.shared[FrameHistoryTextureSlot::PreviousAmbientOcclusion].texture);
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
  std::array<PushConstantTextureBinding, kViewDepthMipCount> depthBindings{};
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    depthBindings[level] = PushConstantTextureBinding{
        .byteOffset = static_cast<uint32_t>(
            offsetof(DepthPrefilterPushConstants, outputTexIds) +
            level * sizeof(uint32_t)),
        .graphTextureResourceIndex = transientTextures[level].value,
        .access = RenderGraphAccessMode::Write};
  }
  const std::array depthDependencies{
      ctx.shared[FrameTextureSlot::SceneDepth].texture};
  const ComputeDispatchItem depthDispatch{
      .pipeline = pipelines_[DepthPrefilter],
      .dispatch = DispatchSize{.x = divRoundUp(width, 16u),
                               .y = divRoundUp(height, 16u),
                               .z = 1u},
      .pushConstants = pushConstants(depthPc),
      .dependencyTextures = depthDependencies,
      .pushConstantTextureBindings = depthBindings,
      .debugLabel = "GTAO Depth Prefilter",
      .debugColor = kGTAODebugColor,
  };
  RenderGraphGraphicsPassDesc depthPass{};
  depthPass.executionMode = RenderPassExecutionMode::ComputeOnly;
  depthPass.hasColorAttachment = false;
  depthPass.preDispatches = std::span(&depthDispatch, 1u);
  depthPass.dependencyTextures = depthDependencies;
  depthPass.gpuTimingScope = GpuTimingScope::GTAOPrefilterEdges;
  depthPass.debugLabel = "GTAO Depth Prefilter Pass";
  depthPass.debugColor = kGTAODebugColor;
  [[maybe_unused]] const RenderGraphPassId depthPassId =
      ctx.graph.addGraphicsPass(depthPass).value();
  const EdgePushConstants edgePc{
      .depthTexId = 0u,
      .outputTexId = 0u,
      .width = width,
      .height = height,
  };
  const std::array edgeBindings{
      PushConstantTextureBinding{
          .byteOffset = offsetof(EdgePushConstants, depthTexId),
          .graphTextureResourceIndex = transientTextures[0].value},
      PushConstantTextureBinding{
          .byteOffset = offsetof(EdgePushConstants, outputTexId),
          .graphTextureResourceIndex = transientTextures[kEdges].value,
          .access = RenderGraphAccessMode::Write}};
  const ComputeDispatchItem edgeDispatch{
      .pipeline = pipelines_[Edge],
      .dispatch = fullDispatch,
      .pushConstants = pushConstants(edgePc),
      .pushConstantTextureBindings = edgeBindings,
      .debugLabel = "GTAO Edges",
      .debugColor = kGTAODebugColor,
  };
  RenderGraphGraphicsPassDesc edgePass{};
  edgePass.executionMode = RenderPassExecutionMode::ComputeOnly;
  edgePass.hasColorAttachment = false;
  edgePass.preDispatches = std::span(&edgeDispatch, 1u);
  edgePass.gpuTimingScope = GpuTimingScope::GTAOPrefilterEdges;
  edgePass.debugLabel = "GTAO Edge Pass";
  edgePass.debugColor = kGTAODebugColor;
  [[maybe_unused]] const RenderGraphPassId edgePassId =
      ctx.graph.addGraphicsPass(edgePass).value();
  if (nuri::isValid(reconstructedNormalDebugTexture_)) {
    const ReconstructNormalsPushConstants reconstructPc{
        .depthTexId = 0u,
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
    const std::array reconstructDependencies{reconstructedNormalDebugTexture_};
    constexpr std::array reconstructAccessModes{RenderGraphAccessMode::Write};
    const std::array reconstructBindings{PushConstantTextureBinding{
        .byteOffset = offsetof(ReconstructNormalsPushConstants, depthTexId),
        .graphTextureResourceIndex = transientTextures[0].value}};
    const ComputeDispatchItem reconstructDispatch{
        .pipeline = pipelines_[ReconstructNormals],
        .dispatch = fullDispatch,
        .pushConstants = pushConstants(reconstructPc),
        .dependencyTextures = reconstructDependencies,
        .pushConstantTextureBindings = reconstructBindings,
        .debugLabel = "GTAO Reconstructed Normals Debug",
        .debugColor = kGTAODebugColor,
    };
    RenderGraphGraphicsPassDesc reconstructPass{};
    reconstructPass.executionMode = RenderPassExecutionMode::ComputeOnly;
    reconstructPass.hasColorAttachment = false;
    reconstructPass.preDispatches = std::span(&reconstructDispatch, 1u);
    reconstructPass.dependencyTextures = reconstructDependencies;
    reconstructPass.dependencyTextureAccessModes = reconstructAccessModes;
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
      .edgeTexId = 0u,
      .outputTexId = 0u,
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
  std::array<PushConstantTextureBinding, kViewDepthMipCount + 2u>
      mainBindings{};
  for (uint32_t level = 0u; level < kViewDepthMipCount; ++level) {
    mainBindings[level] = PushConstantTextureBinding{
        .byteOffset =
            static_cast<uint32_t>(offsetof(MainPushConstants, depthTexIds) +
                                  level * sizeof(uint32_t)),
        .graphTextureResourceIndex = transientTextures[level].value};
  }
  mainBindings[kViewDepthMipCount] = PushConstantTextureBinding{
      .byteOffset = offsetof(MainPushConstants, edgeTexId),
      .graphTextureResourceIndex = transientTextures[kEdges].value};
  mainBindings[kViewDepthMipCount + 1u] = PushConstantTextureBinding{
      .byteOffset = offsetof(MainPushConstants, outputTexId),
      .graphTextureResourceIndex =
          transientTextures[kRawAmbientOcclusion].value,
      .access = RenderGraphAccessMode::Write};
  std::array<TextureHandle, 1> mainDependencies{};
  uint32_t mainDependencyCount = 0u;
  if (plan.inputMode == AmbientOcclusionInputMode::MaterialNormalAndDepth) {
    mainDependencies[mainDependencyCount] =
        ctx.shared[FrameTextureSlot::Normal].texture;
    ++mainDependencyCount;
  }
  const ComputeDispatchItem mainDispatch{
      .pipeline = pipelines_[Main],
      .dispatch = workingDispatch,
      .pushConstants = pushConstants(mainPc),
      .dependencyTextures = std::span<const TextureHandle>(
          mainDependencies.data(), mainDependencyCount),
      .pushConstantTextureBindings = mainBindings,
      .debugLabel = "GTAO Main",
      .debugColor = kGTAODebugColor,
  };
  RenderGraphGraphicsPassDesc mainPass{};
  mainPass.executionMode = RenderPassExecutionMode::ComputeOnly;
  mainPass.hasColorAttachment = false;
  mainPass.preDispatches = std::span(&mainDispatch, 1u);
  mainPass.dependencyTextures = std::span<const TextureHandle>(
      mainDependencies.data(), mainDependencyCount);
  mainPass.gpuTimingScope = GpuTimingScope::GTAOMain;
  mainPass.debugLabel = "GTAO Main Pass";
  mainPass.debugColor = kGTAODebugColor;
  [[maybe_unused]] const RenderGraphPassId mainPassId =
      ctx.graph.addGraphicsPass(mainPass).value();
  auto finalTextureResult = ctx.graph.importTexture(
      ctx.shared[FrameTextureSlot::AmbientOcclusion].texture,
      "gtao_final_ambient_occlusion");
  if (finalTextureResult.hasError())
    return Result<bool, std::string>::makeError(finalTextureResult.error());
  const RenderGraphTextureId finalTexture = finalTextureResult.value();
  RenderGraphTextureId denoiseSource = transientTextures[kRawAmbientOcclusion];
  uint32_t denoiseSourceTexId = 0u;
  for (uint32_t passIndex = 0u; passIndex < denoisePassCount; ++passIndex) {
    const bool finalPass = !halfResolution && !temporalActive &&
                           passIndex + 1u == denoisePassCount;
    const RenderGraphTextureId outputTexture =
        finalPass
            ? finalTexture
            : (passIndex % 2u == 0u ? transientTextures[kDenoise]
                                    : transientTextures[kRawAmbientOcclusion]);
    constexpr uint32_t outputTexId = 0u;
    DenoisePushConstants denoisePc{
        .sourceTexId = denoiseSourceTexId,
        .outputTexId = outputTexId,
        .edgeTexId = 0u,
        .outputWidth = width,
        .outputHeight = height,
        .workingWidth = workingWidth,
        .workingHeight = workingHeight,
    };
    const std::array denoiseBindings{
        PushConstantTextureBinding{
            .byteOffset = offsetof(DenoisePushConstants, sourceTexId),
            .graphTextureResourceIndex = denoiseSource.value},
        PushConstantTextureBinding{
            .byteOffset = offsetof(DenoisePushConstants, outputTexId),
            .graphTextureResourceIndex = outputTexture.value,
            .access = RenderGraphAccessMode::Write},
        PushConstantTextureBinding{
            .byteOffset = offsetof(DenoisePushConstants, edgeTexId),
            .graphTextureResourceIndex = transientTextures[kEdges].value}};
    const ComputeDispatchItem denoiseDispatch{
        .pipeline = pipelines_[Denoise],
        .dispatch = workingDispatch,
        .pushConstants = pushConstants(denoisePc),
        .pushConstantTextureBindings = denoiseBindings,
        .debugLabel = finalPass ? "GTAO Denoise Final" : "GTAO Denoise",
        .debugColor = kGTAODebugColor,
    };
    RenderGraphGraphicsPassDesc denoisePass{};
    denoisePass.executionMode = RenderPassExecutionMode::ComputeOnly;
    denoisePass.hasColorAttachment = false;
    denoisePass.preDispatches = std::span(&denoiseDispatch, 1u);
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
        .currentTexId = 0u,
        .historyTexId = previousAoTexId,
        .motionVectorTexId = motionVectorTexId,
        .reactiveMaskTexId = reactiveMaskTexId,
        .sceneDepthTexId = sourceDepthTexId,
        .currentViewDepthTexId = 0u,
        .previousSceneDepthTexId = previousSceneDepthTexId,
        .edgeTexId = 0u,
        .outputTexId = 0u,
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
    std::array<TextureHandle, 5> temporalDependencies{};
    uint32_t temporalDependencyCount = 0u;
    if (temporalActive) {
      temporalDependencies[0] =
          ctx.shared[FrameHistoryTextureSlot::PreviousAmbientOcclusion].texture;
      temporalDependencies[1] =
          ctx.shared[FrameTextureSlot::MotionVector].texture;
      temporalDependencies[2] =
          ctx.shared[FrameTextureSlot::ReactiveMask].texture;
      temporalDependencies[3] =
          ctx.shared[FrameTextureSlot::SceneDepth].texture;
      temporalDependencies[4] =
          ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture;
      temporalDependencyCount = 5u;
    }
    const std::array temporalBindings{
        PushConstantTextureBinding{
            .byteOffset = offsetof(TemporalPushConstants, currentTexId),
            .graphTextureResourceIndex = denoiseSource.value},
        PushConstantTextureBinding{
            .byteOffset =
                offsetof(TemporalPushConstants, currentViewDepthTexId),
            .graphTextureResourceIndex = transientTextures[0].value},
        PushConstantTextureBinding{
            .byteOffset = offsetof(TemporalPushConstants, edgeTexId),
            .graphTextureResourceIndex = transientTextures[kEdges].value},
        PushConstantTextureBinding{
            .byteOffset = offsetof(TemporalPushConstants, outputTexId),
            .graphTextureResourceIndex = finalTexture.value,
            .access = RenderGraphAccessMode::Write}};
    const ComputeDispatchItem temporalDispatch{
        .pipeline = pipelines_[Temporal],
        .dispatch = fullDispatch,
        .pushConstants = pushConstants(temporalPc),
        .dependencyTextures = std::span<const TextureHandle>(
            temporalDependencies.data(), temporalDependencyCount),
        .pushConstantTextureBindings = temporalBindings,
        .debugLabel = temporalActive ? "GTAO Upscale Temporal" : "GTAO Upscale",
        .debugColor = kGTAODebugColor,
    };
    RenderGraphGraphicsPassDesc temporalPass{};
    temporalPass.executionMode = RenderPassExecutionMode::ComputeOnly;
    temporalPass.hasColorAttachment = false;
    temporalPass.preDispatches = std::span(&temporalDispatch, 1u);
    temporalPass.dependencyTextures = std::span<const TextureHandle>(
        temporalDependencies.data(), temporalDependencyCount);
    temporalPass.gpuTimingScope = temporalActive ? GpuTimingScope::GTAOTemporal
                                                 : GpuTimingScope::GTAOUpscale;
    temporalPass.debugLabel =
        temporalActive ? "GTAO Upscale Temporal Pass" : "GTAO Upscale Pass";
    temporalPass.debugColor = kGTAODebugColor;
    [[maybe_unused]] const RenderGraphPassId temporalPassId =
        ctx.graph.addGraphicsPass(temporalPass).value();
  }
  const auto publishTransientCapture =
      [&](size_t captureIndex, std::string_view name,
          RenderGraphTextureId source, const TextureDesc &desc,
          RenderCaptureValueKind kind, std::string_view colorSpace,
          std::string_view compareProfile,
          std::string_view producerLabel) -> Result<bool, std::string> {
    if (!isRenderCaptureRequested(ctx.frame, name))
      return Result<bool, std::string>::makeResult(true);
    auto ensure =
        ensureCaptureTexture(gpu_, captureTextures_[captureIndex], desc, name);
    if (ensure.hasError())
      return ensure;
    auto destination =
        ctx.graph.importTexture(captureTextures_[captureIndex], name);
    if (destination.hasError())
      return Result<bool, std::string>::makeError(destination.error());
    const RenderGraphTextureCopyItem copy{.sourceTexture = source,
                                          .destinationTexture =
                                              destination.value(),
                                          .width = desc.dimensions.width,
                                          .height = desc.dimensions.height};
    auto copyPass = ctx.graph.addTextureCopyPass(
        RenderGraphTextureCopyPassDesc{.copies = std::span(&copy, 1u),
                                       .debugLabel = producerLabel,
                                       .debugColor = kGTAODebugColor});
    if (copyPass.hasError())
      return Result<bool, std::string>::makeError(copyPass.error());
    publishRequestedCapture(ctx.frame, gpu_, name,
                            captureTextures_[captureIndex], kind,
                            RenderCaptureLifetimeClass::CaptureCopyTexture,
                            colorSpace, compareProfile, producerLabel);
    return Result<bool, std::string>::makeResult(true);
  };
  auto edgeCapture = publishTransientCapture(
      0u, "gtao_edges", transientTextures[kEdges],
      makeStorageSampledTextureDesc(Format::R8_UNORM, width, height),
      RenderCaptureValueKind::Mask, "packed_edge_mask_u8", "mask",
      "GTAO Edge Capture Copy");
  if (edgeCapture.hasError())
    return edgeCapture;
  const bool currentIsFullResolution = !halfResolution && !temporalActive;
  auto currentCapture = publishTransientCapture(
      1u, "gtao_current", denoiseSource,
      makeStorageSampledTextureDesc(
          kFrameCompositionAmbientOcclusionFormat,
          currentIsFullResolution ? width : workingWidth,
          currentIsFullResolution ? height : workingHeight),
      RenderCaptureValueKind::Scalar, "linear_scalar", "scalar",
      "GTAO Current Capture Copy");
  if (currentCapture.hasError())
    return currentCapture;
  publishRequestedCapture(
      ctx.frame, gpu_, "gtao_history",
      ctx.shared[FrameHistoryTextureSlot::PreviousAmbientOcclusion].texture,
      RenderCaptureValueKind::Scalar,
      RenderCaptureLifetimeClass::FrameSharedRingTexture, "linear_scalar",
      "scalar", "GTAO Temporal Pass");
  publishRequestedCapture(
      ctx.frame, gpu_, "gtao_previous_depth",
      ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture,
      RenderCaptureValueKind::Depth,
      RenderCaptureLifetimeClass::FrameSharedRingTexture, "device_depth",
      "depth", "GTAO Temporal Pass");
  ctx.shared[FrameTextureSlot::AmbientOcclusion].graph = finalTexture;
  ctx.frame.sharedResources[FrameTextureSlot::AmbientOcclusion].graph =
      finalTexture;
  ctx.shared.historyWriteRequirements |=
      FrameTextureRequirementFlags::AmbientOcclusion;
  ctx.frame.metrics.ambientOcclusion.ambientOcclusionGraphPublished = true;
  publishRequestedCapture(
      ctx.frame, gpu_, "ambient_occlusion",
      ctx.shared[FrameTextureSlot::AmbientOcclusion].texture,
      RenderCaptureValueKind::Scalar,
      RenderCaptureLifetimeClass::FrameSharedRingTexture, "linear_scalar",
      "scalar", "GTAO Denoise Final Pass");
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GTAOPass::publishFrameData(FrameBuildContext &ctx) {
  const RenderSettings::AmbientOcclusionSettings &ao =
      ctx.frame.settings.ambientOcclusion;
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  ctx.frame.ambientOcclusion = resolveAmbientOcclusionExecutionPlan(
      ctx.frame.settings, ctx.frame.presentationAA.coverage,
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
    if (ctx.frame.presentationAA.gtaoTemporal) {
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
