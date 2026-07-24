#include "nuri/gfx/pipeline/features/spatial_aa_feature.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/fullscreen.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/smaa_lut_contract.h"
#include "nuri/pch.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
namespace nuri {
namespace {
enum SpatialAAStage : size_t {
  EdgeStage,
  BlendStage,
  NeighborhoodStage,
  StageCount
};
enum SpatialAASampler : size_t { LinearSampler, PointSampler, SamplerCount };
enum SpatialAALut : size_t { AreaLut, SearchLut, LutCount };
enum SpatialAAScratch : size_t {
  EdgeScratch,
  BlendScratch,
  OutputScratch,
  ScratchCount
};
struct SpatialAAStageSpec {
  std::string_view name;
  std::string_view fragment;
  Format format;
};
constexpr std::array<SpatialAAStageSpec, StageCount> kSpatialAAStages{{
    {"spatial_aa_edge", "spatial_aa_edge.frag", Format::RGBA8_UNORM},
    {"spatial_aa_blend", "spatial_aa_blend.frag", Format::RGBA8_UNORM},
    {"spatial_aa_neighborhood", "spatial_aa_neighborhood.frag",
     kFrameCompositionFrameColorFormat},
}};
constexpr uint32_t kSpatialAAModeNeighborhood = 0u;
constexpr uint32_t kSpatialAAModeCopy = 1u;
constexpr uint32_t kSpatialAAModeCleanupMask = 2u;
constexpr uint32_t kSpatialAAModeSplitCompare = 3u;
constexpr uint32_t kSpatialAAPassDebugColor = 0xff55aaffu;
constexpr uint32_t kSpatialAADebugPassDebugColor = 0xff77bbffu;
constexpr uint32_t kSpatialAACopyBackPassDebugColor = 0xff4499ffu;
constexpr uint32_t kSpatialAADrawDebugColor = 0xff66aaffu;
constexpr float kSpatialAAEdgeThreshold = 0.12f;
constexpr float kSpatialAAFallbackResolveStrength = 0.30f;
constexpr float kSpatialAAMsaaCleanupResolveStrength = 0.16f;
constexpr float kSpatialAATaaCleanupEdgeThreshold = 0.14f;
constexpr float kSpatialAATaaCleanupResolveStrength = 0.10f;
constexpr float kSpatialAAFallbackLocalContrastFactor = 2.0f;
constexpr float kSpatialAATaaCleanupLocalContrastFactor = 1.45f;
constexpr float kSpatialAAFallbackCornerRounding = 0.25f;
constexpr float kSpatialAATaaCleanupCornerRounding = 0.40f;
constexpr uint32_t kSpatialAAMaxSearchSteps = 16u;
struct SpatialAAPushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t edgeTexId = 0u;
  uint32_t blendTexId = 0u;
  uint32_t areaTexId = 0u;
  uint32_t searchTexId = 0u;
  uint32_t linearSamplerId = 0u;
  uint32_t pointSamplerId = 0u;
  uint32_t mode = 0u;
  uint32_t inverseWidthBits = 0u;
  uint32_t inverseHeightBits = 0u;
  uint32_t edgeThresholdBits = 0u;
  uint32_t maxSearchSteps = 0u;
  uint32_t resolveStrengthBits = 0u;
  uint32_t localContrastFactorBits = 0u;
  uint32_t cornerRoundingBits = 0u;
};
static_assert(sizeof(SpatialAAPushConstants) <= 128);
struct SpatialAAProfile {
  float edgeThreshold = kSpatialAAEdgeThreshold;
  float resolveStrength = kSpatialAAFallbackResolveStrength;
  float localContrastFactor = kSpatialAAFallbackLocalContrastFactor;
  float cornerRounding = kSpatialAAFallbackCornerRounding;
  uint32_t maxSearchSteps = kSpatialAAMaxSearchSteps;
};
[[nodiscard]] inline bool
isSpatialAADebugView(AntiAliasingDebugView view) noexcept {
  return view == AntiAliasingDebugView::SpatialAAEdges ||
         view == AntiAliasingDebugView::SpatialAABlendWeights ||
         view == AntiAliasingDebugView::SpatialAACleanupMask ||
         view == AntiAliasingDebugView::SpatialAASplitCompare;
}
[[nodiscard]] inline bool isTaaDebugView(AntiAliasingDebugView view) noexcept {
  return view != AntiAliasingDebugView::None &&
         view != AntiAliasingDebugView::Settings && !isSpatialAADebugView(view);
}
[[nodiscard]] inline bool
isAADebugOutputView(AntiAliasingDebugView view) noexcept {
  return view != AntiAliasingDebugView::None &&
         view != AntiAliasingDebugView::Settings;
}
[[nodiscard]] inline bool
shouldRunSpatialAA(const RenderFrameContext &frame) noexcept {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  const AntiAliasingMode mode =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode);
  const AntiAliasingDebugView rawDebugView =
      sanitizeAntiAliasingDebugView(settings.antiAliasing.debug.view);
  if (isSpatialAADebugView(rawDebugView)) {
    return true;
  }
  const RenderSettings::AntiAliasingDebugSettings aaDebug =
      effectiveTemporalAADebugSettings(settings.antiAliasing);
  const AntiAliasingDebugView debugView =
      sanitizeAntiAliasingDebugView(aaDebug.view);
  if (isSpatialAADebugView(debugView)) {
    return true;
  }
  if (mode == AntiAliasingMode::SpatialFallback) {
    return true;
  }
  if (presentationAAPlanForFrame(frame).coverage != CoverageMode::Sample1) {
    return false;
  }
  if (mode != AntiAliasingMode::TAA || isTaaDebugView(debugView)) {
    return false;
  }
  if (presentationAAPlanForFrame(frame).reconstruction ==
      ColorReconstruction::ReferenceTAA) {
    return false;
  }
  const bool fallbackWindow =
      !frame.camera.historyValid ||
      frame.camera.framesSinceHistoryReset < frame.camera.jitterSequenceLength;
  return fallbackWindow ||
         (!frame.camera.jitterEnabled && aaDebug.spatialPostTaaCleanup);
}
[[nodiscard]] inline bool
shouldRunPostTransparentSpatialAA(const RenderFrameContext &frame) noexcept {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  const AntiAliasingMode mode =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode);
  const RenderSettings::AntiAliasingDebugSettings aaDebug =
      effectiveTemporalAADebugSettings(settings.antiAliasing);
  const AntiAliasingDebugView debugView =
      sanitizeAntiAliasingDebugView(aaDebug.view);
  const PresentationAAPlan presentationAA = presentationAAPlanForFrame(frame);
  if (presentationAA.coverage != CoverageMode::Sample1) {
    return presentationAA.postAA.active &&
           presentationAA.postAA.spatial == PostAASpatialAlgorithm::Smaa1x &&
           presentationAA.postAA.debugView == AntiAliasingDebugView::None &&
           !isAADebugOutputView(debugView);
  }
  const bool jitteredTaaFrame = frame.camera.jitterEnabled;
  return mode == AntiAliasingMode::TAA && settings.transparent.enabled &&
         aaDebug.transparentPostTaaSpatialCleanup && !jitteredTaaFrame &&
         !isAADebugOutputView(debugView) &&
         frame.metrics.antiAliasing.taaResolvedSceneColorPublished &&
         frame.metrics.antiAliasing.taaTransparentPostTaaDrawCount > 0u;
}
[[nodiscard]] inline bool
isSpatialFallbackActive(const RenderFrameContext &frame) noexcept {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  const AntiAliasingMode mode =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode);
  if (mode == AntiAliasingMode::SpatialFallback) {
    return true;
  }
  if (presentationAAPlanForFrame(frame).reconstruction ==
      ColorReconstruction::ReferenceTAA) {
    return false;
  }
  return mode == AntiAliasingMode::TAA &&
         (!frame.camera.historyValid || frame.camera.framesSinceHistoryReset <
                                            frame.camera.jitterSequenceLength);
}
[[nodiscard]] inline bool
isSpatialCleanupActive(const RenderFrameContext &frame) noexcept {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  const AntiAliasingMode mode =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode);
  const RenderSettings::AntiAliasingDebugSettings aaDebug =
      effectiveTemporalAADebugSettings(settings.antiAliasing);
  if (presentationAAPlanForFrame(frame).coverage != CoverageMode::Sample1) {
    const PostAAPlan &postAA = presentationAAPlanForFrame(frame).postAA;
    return postAA.active && postAA.spatial == PostAASpatialAlgorithm::Smaa1x;
  }
  return mode == AntiAliasingMode::TAA && aaDebug.spatialPostTaaCleanup &&
         !frame.camera.jitterEnabled && !isSpatialFallbackActive(frame);
}
[[nodiscard]] inline bool
isTaaPostSpatialCleanupActive(const RenderFrameContext &frame) noexcept {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  return sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
             AntiAliasingMode::TAA &&
         isSpatialCleanupActive(frame);
}
[[nodiscard]] inline SpatialAAProfile
spatialAAProfile(const RenderFrameContext &frame) noexcept {
  if (isTaaPostSpatialCleanupActive(frame)) {
    return SpatialAAProfile{
        .edgeThreshold = kSpatialAATaaCleanupEdgeThreshold,
        .resolveStrength = kSpatialAATaaCleanupResolveStrength,
        .localContrastFactor = kSpatialAATaaCleanupLocalContrastFactor,
        .cornerRounding = kSpatialAATaaCleanupCornerRounding,
        .maxSearchSteps = kSpatialAAMaxSearchSteps,
    };
  }
  if (isSpatialCleanupActive(frame)) {
    return SpatialAAProfile{
        .edgeThreshold = kSpatialAAEdgeThreshold,
        .resolveStrength = kSpatialAAMsaaCleanupResolveStrength,
        .localContrastFactor = kSpatialAAFallbackLocalContrastFactor,
        .cornerRounding = kSpatialAAFallbackCornerRounding,
        .maxSearchSteps = kSpatialAAMaxSearchSteps,
    };
  }
  return SpatialAAProfile{};
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
[[nodiscard]] uint64_t textureStorageBytes(GPUDevice &gpu,
                                           TextureHandle texture) {
  if (!nuri::isValid(texture)) {
    return 0u;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  return static_cast<uint64_t>(dimensions.width) *
         static_cast<uint64_t>(dimensions.height) *
         static_cast<uint64_t>(std::max(dimensions.depth, 1u)) *
         formatTexelBytes(gpu.getTextureFormat(texture));
}
[[nodiscard]] Result<TextureHandle, std::string>
createSmaaLut(GPUDevice &gpu, const std::filesystem::path &path, uint32_t width,
              uint32_t height, std::string_view label) {
  const size_t byteCount = static_cast<size_t>(width) * height * 4u;
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() != static_cast<std::streamoff>(byteCount)) {
    return Result<TextureHandle, std::string>::makeError("Invalid SMAA LUT: " +
                                                         path.string());
  }
  std::vector<std::byte> bytes(byteCount);
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(bytes.data()),
                  static_cast<std::streamsize>(byteCount))) {
    return Result<TextureHandle, std::string>::makeError(
        "Failed to read SMAA LUT: " + path.string());
  }
  return gpu.createTexture(
      TextureDesc{.type = TextureType::Texture2D,
                  .format = Format::RGBA8_UNORM,
                  .dimensions = TextureDimensions{width, height, 1u},
                  .usage = TextureUsage::Sampled,
                  .storage = Storage::Device,
                  .numLayers = 1u,
                  .numSamples = 1u,
                  .numMipLevels = 1u,
                  .data = bytes,
                  .dataNumMipLevels = 1u},
      label);
}
void addPostAADegradation(AntiAliasingFrameMetrics &metrics,
                          PostAADegradation degradation) {
  metrics.postAA.degradation = static_cast<PostAADegradation>(
      static_cast<uint32_t>(metrics.postAA.degradation) |
      static_cast<uint32_t>(degradation));
}
} // namespace

SpatialAAPass::SpatialAAPass(GPUDevice &gpu, RuntimeCompositeConfig config,
                             SpatialAAPlacement placement)
    : gpu_(gpu), config_(std::move(config)), placement_(placement) {
  auto result = initialize();
  if (result.hasError()) {
    initializationError_ = result.error();
    destroyResources();
    return;
  }
  (void)ensureLuts();
}

SpatialAAPass::~SpatialAAPass() { destroyResources(); }

SpatialAALifecycleSnapshot SpatialAAPass::lifecycleSnapshot() const noexcept {
  SpatialAALifecycleSnapshot snapshot{
      .scratchSlotCount = static_cast<uint32_t>(scratchSlots_.size()),
      .retiredScratchGroupCount = static_cast<uint32_t>(retiredScratch_.size()),
      .submittedPostAALedgerCount =
          static_cast<uint32_t>(submittedPostAA_.size()),
  };
  for (const ScratchSlot &slot : scratchSlots_) {
    snapshot.recordingLeaseCount += slot.leased ? 1u : 0u;
    snapshot.submittedScratchCount += nuri::isValid(slot.submission) ? 1u : 0u;
  }
  return snapshot;
}

void SpatialAAPass::collectCompletedScratch() {
  for (auto it = retiredScratch_.begin(); it != retiredScratch_.end();) {
    if (nuri::isValid(it->submission) &&
        !gpu_.isSubmissionComplete(it->submission)) {
      ++it;
      continue;
    }
    for (TextureHandle texture : it->textures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
      }
    }
    it = retiredScratch_.erase(it);
  }
}

void SpatialAAPass::retireCurrentScratch() {
  for (size_t slotIndex = 0u; slotIndex < scratchSlots_.size(); ++slotIndex) {
    RetiredScratch retired{
        .submission = scratchSlots_[slotIndex].submission,
    };
    for (size_t kind = 0u; kind < scratchTextures_.size(); ++kind) {
      if (slotIndex < scratchTextures_[kind].size()) {
        retired.textures[kind] = scratchTextures_[kind][slotIndex];
      }
    }
    if (nuri::isValid(retired.submission) &&
        !gpu_.isSubmissionComplete(retired.submission)) {
      retiredScratch_.push_back(retired);
    } else {
      for (TextureHandle texture : retired.textures) {
        if (nuri::isValid(texture)) {
          gpu_.destroyTexture(texture);
        }
      }
    }
  }
  for (auto &textures : scratchTextures_) {
    textures.clear();
  }
  scratchSlots_.clear();
}

void SpatialAAPass::destroyResources() {
  for (RenderPipelineHandle &pipeline : pipelines_) {
    if (nuri::isValid(pipeline))
      gpu_.destroyRenderPipeline(pipeline);
    pipeline = {};
  }
  if (nuri::isValid(vertexShader_)) {
    gpu_.destroyShaderModule(vertexShader_);
  }
  vertexShader_ = {};
  for (ShaderHandle &shader : fragmentShaders_) {
    if (nuri::isValid(shader))
      gpu_.destroyShaderModule(shader);
    shader = {};
  }
  for (auto &textures : scratchTextures_) {
    for (TextureHandle texture : textures)
      if (nuri::isValid(texture))
        gpu_.destroyTexture(texture);
    textures.clear();
  }
  for (RetiredScratch &retired : retiredScratch_) {
    for (TextureHandle texture : retired.textures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
      }
    }
  }
  retiredScratch_.clear();
  scratchSlots_.clear();
  submittedPostAA_.clear();
  pendingLease_.reset();
  scratchWidth_ = 0u;
  scratchHeight_ = 0u;
  scratchRingCount_ = 0u;
  outputScratchFormat_ = Format::Count;
  for (TextureHandle &lut : luts_) {
    if (nuri::isValid(lut))
      gpu_.destroyTexture(lut);
    lut = {};
  }
  for (SamplerHandle &sampler : samplers_) {
    if (nuri::isValid(sampler))
      gpu_.destroySampler(sampler);
    sampler = {};
  }
}

Result<bool, std::string> SpatialAAPass::initialize() {
  const std::filesystem::path basePath = resolveShaderBasePath(config_);
  const std::filesystem::path vertexPath =
      config_.fullscreenVertex.empty() ? basePath / "fullscreen_copy.vert"
                                       : config_.fullscreenVertex;
  auto vertexCompiler = Shader::create("spatial_aa_vertex", gpu_);
  if (!vertexCompiler)
    return Result<bool, std::string>::makeError(
        "SpatialAA: shader creation failed");
  auto vertex =
      vertexCompiler->compileFromFile(vertexPath.string(), ShaderStage::Vertex);
  if (vertex.hasError())
    return Result<bool, std::string>::makeError(vertex.error());
  vertexShader_ = vertex.value();
  for (size_t index = 0; index < kSpatialAAStages.size(); ++index) {
    const SpatialAAStageSpec &spec = kSpatialAAStages[index];
    auto compiler = Shader::create(spec.name, gpu_);
    if (!compiler)
      return Result<bool, std::string>::makeError(
          "SpatialAA: shader creation failed");
    auto fragment = compiler->compileFromFile(
        (basePath / spec.fragment).string(), ShaderStage::Fragment);
    if (fragment.hasError())
      return Result<bool, std::string>::makeError(fragment.error());
    fragmentShaders_[index] = fragment.value();
    auto pipeline = gpu_.createRenderPipeline(
        fullscreenPipelineDesc(spec.format, vertexShader_,
                               fragmentShaders_[index]),
        spec.name);
    if (pipeline.hasError())
      return Result<bool, std::string>::makeError(pipeline.error());
    pipelines_[index] = pipeline.value();
  }
  for (size_t index = 0; index < samplers_.size(); ++index) {
    const SamplerFilter filter =
        index == LinearSampler ? SamplerFilter::Linear : SamplerFilter::Nearest;
    auto sampler =
        gpu_.createSampler(SamplerDesc{.minFilter = filter,
                                       .magFilter = filter,
                                       .mipMode = SamplerMipMode::Disabled,
                                       .wrapU = SamplerWrapMode::Clamp,
                                       .wrapV = SamplerWrapMode::Clamp,
                                       .wrapW = SamplerWrapMode::Clamp},
                           index == LinearSampler ? "spatial_aa_linear_clamp"
                                                  : "spatial_aa_point_clamp");
    if (sampler.hasError())
      return Result<bool, std::string>::makeError(sampler.error());
    samplers_[index] = sampler.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> SpatialAAPass::ensureLuts() {
  const std::filesystem::path basePath = resolveShaderBasePath(config_);
  constexpr std::array filenames{std::string_view{smaa_lut::kAreaFilename},
                                 std::string_view{smaa_lut::kSearchFilename}};
  constexpr std::array widths{smaa_lut::kAreaWidth, smaa_lut::kSearchWidth};
  constexpr std::array heights{smaa_lut::kAreaHeight, smaa_lut::kSearchHeight};
  constexpr std::array names{std::string_view{"spatial_aa_smaa_area_lut"},
                             std::string_view{"spatial_aa_smaa_search_lut"}};
  for (size_t index = 0; index < luts_.size(); ++index) {
    if (nuri::isValid(luts_[index])) {
      continue;
    }
    auto result = createSmaaLut(gpu_, basePath / filenames[index],
                                widths[index], heights[index], names[index]);
    if (result.hasError()) {
      return Result<bool, std::string>::makeError(result.error());
    }
    luts_[index] = result.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SpatialAAPass::ensureScratchTextures(FrameBuildContext &ctx) {
  collectCompletedScratch();
  if (!initializationError_.empty()) {
    return Result<bool, std::string>::makeError(initializationError_);
  }
  auto luts = ensureLuts();
  if (luts.hasError()) {
    return luts;
  }
  const TextureHandle sourceTexture =
      placement_ == SpatialAAPlacement::PostTransparent
          ? ctx.shared.frameColorTexture
          : ctx.shared.sceneColorTexture;
  const TextureDimensions dimensions = gpu_.getTextureDimensions(sourceTexture);
  const uint32_t ringCount = std::max(2u, gpu_.getSwapchainImageCount());
  const bool needsOutputScratch =
      placement_ == SpatialAAPlacement::PostTransparent;
  const Format outputScratchFormat =
      needsOutputScratch ? gpu_.getTextureFormat(sourceTexture) : Format::Count;
  const bool recreateScratch = scratchWidth_ != dimensions.width ||
                               scratchHeight_ != dimensions.height ||
                               scratchRingCount_ != ringCount ||
                               outputScratchFormat_ != outputScratchFormat;
  if (!recreateScratch) {
    return Result<bool, std::string>::makeResult(true);
  }
  const bool hadScratch = scratchRingCount_ != 0u;
  std::array<std::vector<TextureHandle>, ScratchCount> newTextures{};
  newTextures[EdgeScratch].reserve(ringCount);
  newTextures[BlendScratch].reserve(ringCount);
  if (needsOutputScratch) {
    newTextures[OutputScratch].reserve(ringCount);
  }
  const auto abandonNewTextures = [&]() {
    for (auto &textures : newTextures) {
      for (TextureHandle texture : textures) {
        if (nuri::isValid(texture)) {
          gpu_.destroyTexture(texture);
        }
      }
    }
  };
  const TextureDesc scratchDesc{
      .type = TextureType::Texture2D,
      .format = Format::RGBA8_UNORM,
      .dimensions = TextureDimensions{std::max(dimensions.width, 1u),
                                      std::max(dimensions.height, 1u), 1u},
      .usage = TextureUsage::AttachmentSampled,
      .storage = Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = 1u,
      .data = {},
      .dataNumMipLevels = 1u,
      .generateMipmaps = false};
  for (uint32_t i = 0u; i < ringCount; ++i) {
    constexpr std::array names{"spatial_aa_edges", "spatial_aa_blend_weights"};
    for (size_t kind = EdgeScratch; kind <= BlendScratch; ++kind) {
      auto texture = gpu_.createTexture(scratchDesc, names[kind]);
      if (texture.hasError()) {
        abandonNewTextures();
        return Result<bool, std::string>::makeError(texture.error());
      }
      newTextures[kind].push_back(texture.value());
    }
    if (needsOutputScratch) {
      TextureDesc outputDesc = scratchDesc;
      outputDesc.format = outputScratchFormat;
      auto outputTexture =
          gpu_.createTexture(outputDesc, "transparent_spatial_aa_output");
      if (outputTexture.hasError()) {
        abandonNewTextures();
        return Result<bool, std::string>::makeError(outputTexture.error());
      }
      newTextures[OutputScratch].push_back(outputTexture.value());
    }
  }
  retireCurrentScratch();
  scratchTextures_.swap(newTextures);
  scratchSlots_.assign(ringCount, ScratchSlot{});
  scratchWidth_ = dimensions.width;
  scratchHeight_ = dimensions.height;
  scratchRingCount_ = ringCount;
  outputScratchFormat_ = outputScratchFormat;
  const uint32_t createdTextureCount =
      ringCount * (needsOutputScratch ? 3u : 2u);
  AntiAliasingFrameMetrics &metrics = ctx.frame.metrics.antiAliasing;
  if (hadScratch) {
    metrics.spatialAAReallocationCount += createdTextureCount;
  } else {
    metrics.spatialAAAllocationCount += createdTextureCount;
  }
  return Result<bool, std::string>::makeResult(true);
}

bool SpatialAAPass::isEnabled(const FrameBuildContext &ctx) const {
  if (placement_ == SpatialAAPlacement::PostTransparent) {
    return nuri::isValid(ctx.shared.frameColorTexture) &&
           shouldRunPostTransparentSpatialAA(ctx.frame);
  }
  return nuri::isValid(ctx.shared.sceneColorTexture) &&
         nuri::isValid(ctx.shared.frameColorTexture) &&
         shouldRunSpatialAA(ctx.frame);
}

Result<bool, std::string> SpatialAAPass::prepare(FrameBuildContext &ctx) {
  const bool postAA =
      placement_ == SpatialAAPlacement::PostTransparent &&
      ctx.frame.presentationAA.coverage != CoverageMode::Sample1 &&
      ctx.frame.presentationAA.postAA.active &&
      ctx.frame.presentationAA.postAA.spatial == PostAASpatialAlgorithm::Smaa1x;
  AntiAliasingFrameMetrics &metrics = ctx.frame.metrics.antiAliasing;
  const uint64_t timingSource = ctx.frame.gpuTiming.spatialAASourceFrameIndex;
  for (auto it = submittedPostAA_.begin(); it != submittedPostAA_.end();) {
    const bool timingMatches =
        hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::SpatialAA) &&
        timingSource == it->sourceFrameIndex;
    if (!timingMatches && nuri::isValid(it->submission) &&
        !gpu_.isSubmissionComplete(it->submission)) {
      ++it;
      continue;
    }
    metrics.postAA.smaaCompleted = true;
    metrics.postAA.smaaCompletedSourceFrameIndex = it->sourceFrameIndex;
    it = submittedPostAA_.erase(it);
  }
  if (placement_ == SpatialAAPlacement::PostTransparent) {
    const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
    const RenderSettings::AntiAliasingDebugSettings aaDebug =
        effectiveTemporalAADebugSettings(settings.antiAliasing);
    if (presentationAAPlanForFrame(ctx.frame).coverage !=
        CoverageMode::Sample1) {
      ctx.frame.metrics.antiAliasing.msaaSpatialCleanupEnabled = postAA;
    } else {
      ctx.frame.metrics.antiAliasing.taaTransparentPostSpatialCleanupEnabled =
          aaDebug.transparentPostTaaSpatialCleanup;
    }
  }
  if (!initializationError_.empty()) {
    if (postAA) {
      addPostAADegradation(metrics,
                           PostAADegradation::SmaaDependenciesUnavailable);
      return Result<bool, std::string>::makeResult(false);
    }
    return Result<bool, std::string>::makeError(initializationError_);
  }
  auto luts = ensureLuts();
  if (luts.hasError()) {
    if (postAA) {
      addPostAADegradation(metrics,
                           PostAADegradation::SmaaDependenciesUnavailable);
      return Result<bool, std::string>::makeResult(false);
    }
    return luts;
  }
  auto scratch = ensureScratchTextures(ctx);
  if (scratch.hasError()) {
    if (postAA) {
      addPostAADegradation(metrics,
                           PostAADegradation::SmaaScratchAllocationFailed);
      return Result<bool, std::string>::makeResult(false);
    }
    return scratch;
  }
  for (uint32_t slotIndex = 0u; slotIndex < scratchSlots_.size(); ++slotIndex) {
    ScratchSlot &slot = scratchSlots_[slotIndex];
    if (slot.leased) {
      continue;
    }
    if (nuri::isValid(slot.submission) &&
        !gpu_.isSubmissionComplete(slot.submission)) {
      continue;
    }
    slot.submission = {};
    slot.leased = true;
    pendingLease_ = PendingLease{
        .frameIndex = ctx.frame.frameIndex,
        .slot = slotIndex,
        .postAA = postAA,
    };
    return Result<bool, std::string>::makeResult(true);
  }
  if (postAA) {
    addPostAADegradation(metrics, PostAADegradation::SmaaScratchRingSaturated);
    return Result<bool, std::string>::makeResult(false);
  }
  return Result<bool, std::string>::makeError(
      "SpatialAA: every scratch slot is still in flight");
}

Result<bool, std::string> SpatialAAPass::build(FrameBuildContext &ctx) {
  if (!pendingLease_.has_value() ||
      pendingLease_->frameIndex != ctx.frame.frameIndex) {
    return Result<bool, std::string>::makeResult(false);
  }
  const bool postTransparent =
      placement_ == SpatialAAPlacement::PostTransparent;
  const uint32_t ringIndex = pendingLease_->slot;
  const TextureHandle edgeTexture = scratchTextures_[EdgeScratch][ringIndex];
  const TextureHandle blendTexture = scratchTextures_[BlendScratch][ringIndex];
  const TextureHandle sourceTexture = postTransparent
                                          ? ctx.shared.frameColorTexture
                                          : ctx.shared.sceneColorTexture;
  const TextureHandle outputTexture =
      postTransparent ? scratchTextures_[OutputScratch][ringIndex]
                      : ctx.shared.frameColorTexture;
  const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(sourceTexture);
  const uint32_t edgeTexId = gpu_.getTextureBindlessIndex(edgeTexture);
  const uint32_t blendTexId = gpu_.getTextureBindlessIndex(blendTexture);
  const uint32_t areaTexId = gpu_.getTextureBindlessIndex(luts_[AreaLut]);
  const uint32_t searchTexId = gpu_.getTextureBindlessIndex(luts_[SearchLut]);
  const uint32_t outputTexId = gpu_.getTextureBindlessIndex(outputTexture);
  const uint32_t linearSamplerId =
      gpu_.getSamplerBindlessIndex(samplers_[LinearSampler]);
  const uint32_t pointSamplerId =
      gpu_.getSamplerBindlessIndex(samplers_[PointSampler]);
  auto importSource = ctx.graph.importTexture(
      sourceTexture, postTransparent ? "transparent_spatial_aa_frame_color"
                                     : "spatial_aa_scene");
  if (importSource.hasError()) {
    return Result<bool, std::string>::makeError(importSource.error());
  }
  auto importOutput = ctx.graph.importTexture(
      outputTexture,
      postTransparent ? "transparent_spatial_aa_output" : "spatial_aa_output");
  if (importOutput.hasError()) {
    return Result<bool, std::string>::makeError(importOutput.error());
  }
  auto importEdges = ctx.graph.importTexture(edgeTexture, "spatial_aa_edges");
  if (importEdges.hasError()) {
    return Result<bool, std::string>::makeError(importEdges.error());
  }
  auto importBlend =
      ctx.graph.importTexture(blendTexture, "spatial_aa_blend_weights");
  if (importBlend.hasError()) {
    return Result<bool, std::string>::makeError(importBlend.error());
  }
  const TextureDimensions dimensions = gpu_.getTextureDimensions(sourceTexture);
  const float inverseWidth =
      1.0f / static_cast<float>(std::max(dimensions.width, 1u));
  const float inverseHeight =
      1.0f / static_cast<float>(std::max(dimensions.height, 1u));
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  const RenderSettings::AntiAliasingDebugSettings aaDebug =
      effectiveTemporalAADebugSettings(settings.antiAliasing);
  const AntiAliasingDebugView debugView =
      sanitizeAntiAliasingDebugView(aaDebug.view);
  const bool multisampled =
      presentationAAPlanForFrame(ctx.frame).coverage != CoverageMode::Sample1;
  const bool fallbackActive =
      postTransparent ? false : isSpatialFallbackActive(ctx.frame);
  const bool cleanupActive =
      postTransparent ? true : isSpatialCleanupActive(ctx.frame);
  const SpatialAAProfile profile =
      postTransparent && !multisampled
          ? SpatialAAProfile{
                .edgeThreshold = kSpatialAATaaCleanupEdgeThreshold,
                .resolveStrength = kSpatialAATaaCleanupResolveStrength,
                .localContrastFactor = kSpatialAATaaCleanupLocalContrastFactor,
                .cornerRounding = kSpatialAATaaCleanupCornerRounding,
                .maxSearchSteps = kSpatialAAMaxSearchSteps,
            }
          : spatialAAProfile(ctx.frame);
  const SpatialAAPushConstants baseConstants{
      .sourceTexId = sourceTexId,
      .edgeTexId = edgeTexId,
      .blendTexId = blendTexId,
      .areaTexId = areaTexId,
      .searchTexId = searchTexId,
      .linearSamplerId = linearSamplerId,
      .pointSamplerId = pointSamplerId,
      .mode = kSpatialAAModeNeighborhood,
      .inverseWidthBits = std::bit_cast<uint32_t>(inverseWidth),
      .inverseHeightBits = std::bit_cast<uint32_t>(inverseHeight),
      .edgeThresholdBits = std::bit_cast<uint32_t>(profile.edgeThreshold),
      .maxSearchSteps = profile.maxSearchSteps,
      .resolveStrengthBits = std::bit_cast<uint32_t>(profile.resolveStrength),
      .localContrastFactorBits =
          std::bit_cast<uint32_t>(profile.localContrastFactor),
      .cornerRoundingBits = std::bit_cast<uint32_t>(profile.cornerRounding),
  };
  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  const GpuTimingReport &timingReport = ctx.frame.gpuTiming;
  if (hasGpuTimingScope(timingReport, GpuTimingScope::SpatialAA)) {
    aaMetrics.spatialAAGpuTimeMs = timingReport.spatialAATimeMs;
    aaMetrics.spatialAAGpuTimingSourceFrameIndex =
        timingReport.spatialAASourceFrameIndex;
    aaMetrics.spatialAAGpuTimingAvailable = 1u;
  }
  aaMetrics.spatialAATextureBytes =
      textureStorageBytes(gpu_, edgeTexture) +
      textureStorageBytes(gpu_, blendTexture) +
      (postTransparent ? textureStorageBytes(gpu_, outputTexture) : 0u);
  aaMetrics.spatialAATotalBytes =
      aaMetrics.spatialAATextureBytes *
      static_cast<uint64_t>(scratchTextures_[EdgeScratch].size());
  aaMetrics.spatialAALutTextureBytes =
      textureStorageBytes(gpu_, luts_[AreaLut]) +
      textureStorageBytes(gpu_, luts_[SearchLut]);
  if (!postTransparent) {
    aaMetrics.spatialAAEnabled = true;
    aaMetrics.spatialAAFallbackActive = fallbackActive;
    aaMetrics.spatialAACleanupActive = cleanupActive;
    aaMetrics.msaaSpatialCleanupEnabled =
        multisampled && ctx.frame.presentationAA.postAA.active &&
        ctx.frame.presentationAA.postAA.spatial ==
            PostAASpatialAlgorithm::Smaa1x;
    aaMetrics.msaaSpatialCleanupActive = multisampled && cleanupActive;
    aaMetrics.spatialAAWidth = dimensions.width;
    aaMetrics.spatialAAHeight = dimensions.height;
    aaMetrics.spatialAAEdgePixelEstimate = 0.0f;
    aaMetrics.spatialAAModifiedPixelEstimate = 0.0f;
    if (cleanupActive) {
      ++aaMetrics.spatialAACleanupFrameCount;
    }
  } else if (multisampled) {
    aaMetrics.spatialAAEnabled = true;
    aaMetrics.spatialAACleanupActive = true;
    aaMetrics.msaaSpatialCleanupEnabled = true;
    aaMetrics.msaaSpatialCleanupActive = true;
    aaMetrics.spatialAAWidth = dimensions.width;
    aaMetrics.spatialAAHeight = dimensions.height;
    ++aaMetrics.spatialAACleanupFrameCount;
  } else {
    aaMetrics.taaTransparentPostSpatialCleanupActive = true;
  }
  const DrawItem edgeDraw = makeFullscreenDraw(
      pipelines_[EdgeStage],
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&baseConstants),
          sizeof(baseConstants)),
      "SpatialAAEdge", kSpatialAADrawDebugColor);
  const std::array<TextureHandle, 1> edgeReads{sourceTexture};
  RenderGraphGraphicsPassDesc edgePass{};
  edgePass.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  edgePass.colorTexture = importEdges.value();
  edgePass.dependencyTextures =
      std::span<const TextureHandle>(edgeReads.data(), edgeReads.size());
  edgePass.draws = std::span<const DrawItem>(&edgeDraw, 1u);
  edgePass.gpuTimingScope = GpuTimingScope::SpatialAA;
  edgePass.debugLabel = postTransparent ? "Transparent SpatialAA Edge Pass"
                                        : "SMAA Edge Detection Pass";
  edgePass.debugColor = kSpatialAAPassDebugColor;
  auto addEdge = ctx.graph.addGraphicsPass(edgePass);
  if (addEdge.hasError()) {
    return Result<bool, std::string>::makeError(addEdge.error());
  }
  if (postTransparent) {
    if (multisampled) {
      ++aaMetrics.spatialAAPassCount;
      ++aaMetrics.spatialAAEdgePassCount;
    } else {
      ++aaMetrics.taaTransparentPostSpatialAAPassCount;
    }
  } else {
    ++aaMetrics.spatialAAPassCount;
    ++aaMetrics.spatialAAEdgePassCount;
  }
  aaMetrics.spatialAABandwidthEstimateBytes +=
      textureStorageBytes(gpu_, sourceTexture) +
      textureStorageBytes(gpu_, edgeTexture);
  const bool edgeDebug = debugView == AntiAliasingDebugView::SpatialAAEdges;
  const bool blendDebug =
      debugView == AntiAliasingDebugView::SpatialAABlendWeights;
  if (!edgeDebug) {
    const DrawItem blendDraw = makeFullscreenDraw(
        pipelines_[BlendStage],
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&baseConstants),
            sizeof(baseConstants)),
        "SpatialAABlend", kSpatialAADrawDebugColor);
    const std::array<TextureHandle, 4> blendReads{
        edgeTexture, sourceTexture, luts_[AreaLut], luts_[SearchLut]};
    RenderGraphGraphicsPassDesc blendPass{};
    blendPass.color = {.loadOp = LoadOp::Clear,
                       .storeOp = StoreOp::Store,
                       .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    blendPass.colorTexture = importBlend.value();
    blendPass.dependencyTextures =
        std::span<const TextureHandle>(blendReads.data(), blendReads.size());
    blendPass.draws = std::span<const DrawItem>(&blendDraw, 1u);
    blendPass.gpuTimingScope = GpuTimingScope::SpatialAA;
    blendPass.debugLabel = postTransparent ? "Transparent SpatialAA Blend Pass"
                                           : "SMAA Blend Weight Pass";
    blendPass.debugColor = kSpatialAAPassDebugColor;
    auto addBlend = ctx.graph.addGraphicsPass(blendPass);
    if (addBlend.hasError()) {
      return Result<bool, std::string>::makeError(addBlend.error());
    }
    if (postTransparent) {
      if (multisampled) {
        ++aaMetrics.spatialAAPassCount;
        ++aaMetrics.spatialAABlendPassCount;
      } else {
        ++aaMetrics.taaTransparentPostSpatialAAPassCount;
      }
    } else {
      ++aaMetrics.spatialAAPassCount;
      ++aaMetrics.spatialAABlendPassCount;
    }
    aaMetrics.spatialAABandwidthEstimateBytes +=
        textureStorageBytes(gpu_, edgeTexture) +
        textureStorageBytes(gpu_, sourceTexture) +
        textureStorageBytes(gpu_, luts_[AreaLut]) +
        textureStorageBytes(gpu_, luts_[SearchLut]) +
        textureStorageBytes(gpu_, blendTexture);
  }
  SpatialAAPushConstants outputConstants = baseConstants;
  std::string_view outputLabel = postTransparent
                                     ? "Transparent SpatialAA Neighborhood Pass"
                                     : "SMAA Neighborhood Pass";
  TextureHandle outputSource = sourceTexture;
  std::array<TextureHandle, 2> outputReads{sourceTexture, blendTexture};
  size_t outputReadCount = 2u;
  if (edgeDebug || blendDebug) {
    outputConstants.sourceTexId = edgeDebug ? edgeTexId : blendTexId;
    outputConstants.mode = kSpatialAAModeCopy;
    outputLabel =
        edgeDebug ? "SMAA Edge Debug Pass" : "SMAA Blend Weight Debug Pass";
    outputSource = edgeDebug ? edgeTexture : blendTexture;
    outputReads[0] = outputSource;
    outputReadCount = 1u;
    ++aaMetrics.spatialAADebugPassCount;
    aaMetrics.spatialAAEdgesDebugViewRendered = edgeDebug;
    aaMetrics.spatialAABlendWeightsDebugViewRendered = blendDebug;
  } else if (debugView == AntiAliasingDebugView::SpatialAACleanupMask) {
    outputConstants.mode = kSpatialAAModeCleanupMask;
    outputLabel = "SMAA Cleanup Mask Debug Pass";
    ++aaMetrics.spatialAADebugPassCount;
    aaMetrics.spatialAACleanupMaskDebugViewRendered = true;
  } else if (debugView == AntiAliasingDebugView::SpatialAASplitCompare) {
    outputConstants.mode = kSpatialAAModeSplitCompare;
    outputLabel = "SMAA Split Compare Debug Pass";
    ++aaMetrics.spatialAADebugPassCount;
    aaMetrics.spatialAASplitCompareDebugViewRendered = true;
  }
  const DrawItem outputDraw = makeFullscreenDraw(
      pipelines_[NeighborhoodStage],
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&outputConstants),
          sizeof(outputConstants)),
      "SpatialAAOutput", kSpatialAADrawDebugColor);
  RenderGraphGraphicsPassDesc outputPass{};
  outputPass.color = {.loadOp = LoadOp::Clear,
                      .storeOp = StoreOp::Store,
                      .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  outputPass.colorTexture = importOutput.value();
  outputPass.dependencyTextures =
      std::span<const TextureHandle>(outputReads.data(), outputReadCount);
  outputPass.draws = std::span<const DrawItem>(&outputDraw, 1u);
  outputPass.gpuTimingScope = GpuTimingScope::SpatialAA;
  outputPass.debugLabel = outputLabel;
  outputPass.debugColor = isSpatialAADebugView(debugView)
                              ? kSpatialAADebugPassDebugColor
                              : kSpatialAAPassDebugColor;
  auto addOutput = ctx.graph.addGraphicsPass(outputPass);
  if (addOutput.hasError()) {
    return Result<bool, std::string>::makeError(addOutput.error());
  }
  if (postTransparent) {
    if (multisampled) {
      ++aaMetrics.spatialAAPassCount;
      ++aaMetrics.spatialAANeighborhoodPassCount;
    } else {
      ++aaMetrics.taaTransparentPostSpatialAAPassCount;
    }
  } else {
    ++aaMetrics.spatialAAPassCount;
    ++aaMetrics.spatialAANeighborhoodPassCount;
  }
  aaMetrics.spatialAABandwidthEstimateBytes +=
      textureStorageBytes(gpu_, outputSource) +
      (outputReadCount > 1u ? textureStorageBytes(gpu_, blendTexture) : 0u) +
      textureStorageBytes(gpu_, outputTexture);
  SpatialAAPushConstants copyConstants = baseConstants;
  copyConstants.sourceTexId = outputTexId;
  copyConstants.mode = kSpatialAAModeCopy;
  const DrawItem copyDraw = makeFullscreenDraw(
      pipelines_[NeighborhoodStage],
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&copyConstants),
          sizeof(copyConstants)),
      "SpatialAACopyBack", kSpatialAADrawDebugColor);
  const std::array<TextureHandle, 1> copyReads{outputTexture};
  RenderGraphGraphicsPassDesc copyPass{};
  copyPass.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  copyPass.colorTexture = importSource.value();
  copyPass.dependencyTextures =
      std::span<const TextureHandle>(copyReads.data(), copyReads.size());
  copyPass.draws = std::span<const DrawItem>(&copyDraw, 1u);
  copyPass.gpuTimingScope = GpuTimingScope::SpatialAA;
  copyPass.debugLabel = postTransparent ? "Transparent SpatialAA Copy Back Pass"
                                        : "SMAA Copy Back Pass";
  copyPass.debugColor = kSpatialAACopyBackPassDebugColor;
  auto addCopy = ctx.graph.addGraphicsPass(copyPass);
  if (addCopy.hasError()) {
    return Result<bool, std::string>::makeError(addCopy.error());
  }
  if (postTransparent) {
    if (multisampled) {
      ++aaMetrics.spatialAAPassCount;
      ++aaMetrics.spatialAACopyBackPassCount;
    } else {
      ++aaMetrics.taaTransparentPostSpatialAAPassCount;
    }
  } else {
    ++aaMetrics.spatialAAPassCount;
    ++aaMetrics.spatialAACopyBackPassCount;
  }
  aaMetrics.spatialAABandwidthEstimateBytes +=
      textureStorageBytes(gpu_, outputTexture) +
      textureStorageBytes(gpu_, sourceTexture);
  if (pendingLease_->postAA) {
    pendingLease_->passCount = 4u;
    aaMetrics.postAA.smaaPlanned = true;
  }
  if (postTransparent) {
    ctx.shared.frameColorGraphTexture = importSource.value();
    ctx.frame.sharedResources.frameColorGraphTexture = importSource.value();
  } else {
    ctx.shared.sceneColorGraphTexture = importSource.value();
    ctx.frame.sharedResources.sceneColorGraphTexture = importSource.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

void SpatialAAPass::onFrameSubmitted(const RenderFrameContext &frame) noexcept {
  if (!pendingLease_.has_value() ||
      pendingLease_->frameIndex != frame.frameIndex) {
    return;
  }
  ScratchSlot &slot = scratchSlots_[pendingLease_->slot];
  slot.leased = false;
  slot.submission = frame.submission;
  if (pendingLease_->postAA) {
    auto &metrics =
        const_cast<RenderFrameContext &>(frame).metrics.antiAliasing.postAA;
    metrics.smaaSubmitted = true;
    metrics.smaaSubmittedPassCount = pendingLease_->passCount;
    submittedPostAA_.push_back(SubmittedPostAA{
        .sourceFrameIndex = frame.frameIndex,
        .submission = frame.submission,
        .passCount = pendingLease_->passCount,
    });
  }
  pendingLease_.reset();
}

void SpatialAAPass::onFrameAbandoned(const RenderFrameContext &frame) noexcept {
  if (!pendingLease_.has_value() ||
      pendingLease_->frameIndex != frame.frameIndex) {
    return;
  }
  ScratchSlot &slot = scratchSlots_[pendingLease_->slot];
  slot.leased = false;
  slot.submission = {};
  pendingLease_.reset();
}

void registerSpatialAAStage(RenderPipeline &pipeline, GPUDevice &gpu,
                            RuntimeCompositeConfig config,
                            SpatialAAPlacement placement) {
  pipeline.addStage(
      std::make_unique<SpatialAAPass>(gpu, std::move(config), placement),
      "SpatialAAFeature", "SpatialAAPass", false,
      PipelineComponentDesc{
          .submitted =
              [](void *state, const RenderFrameContext &frame) noexcept {
                static_cast<SpatialAAPass *>(state)->onFrameSubmitted(frame);
              },
          .abandoned =
              [](void *state, const RenderFrameContext &frame) noexcept {
                static_cast<SpatialAAPass *>(state)->onFrameAbandoned(frame);
              },
      });
}

} // namespace nuri
