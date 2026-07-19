#include "nuri/gfx/pipeline/features/spatial_aa_feature.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/fullscreen.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
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
constexpr uint32_t kAreaLutWidth = 160u;
constexpr uint32_t kAreaLutHeight = 560u;
constexpr uint32_t kSearchLutWidth = 64u;
constexpr uint32_t kSearchLutHeight = 16u;
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
  if (mode == AntiAliasingMode::MSAA4x) {
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
  if (presentationAA.coverage == CoverageMode::Sample4) {
    return presentationAA.spatialCleanup ==
               SpatialCleanupPoint::PostTransparency &&
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
  if (mode == AntiAliasingMode::MSAA4x) {
    return aaDebug.spatialPostMsaaCleanup;
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
} // namespace

SpatialAAPass::SpatialAAPass(GPUDevice &gpu, RuntimeCompositeConfig config,
                             SpatialAAPlacement placement)
    : gpu_(gpu), config_(std::move(config)), placement_(placement) {
  auto result = initialize();
  if (result.hasError())
    initializationError_ = result.error();
}

SpatialAAPass::~SpatialAAPass() { destroyResources(); }

void SpatialAAPass::destroyResources() {
  for (RenderPipelineHandle pipeline : pipelines_)
    if (nuri::isValid(pipeline))
      gpu_.destroyRenderPipeline(pipeline);
  if (nuri::isValid(vertexShader_))
    gpu_.destroyShaderModule(vertexShader_);
  for (ShaderHandle shader : fragmentShaders_)
    if (nuri::isValid(shader))
      gpu_.destroyShaderModule(shader);
  for (auto &textures : scratchTextures_) {
    for (TextureHandle texture : textures)
      if (nuri::isValid(texture))
        gpu_.destroyTexture(texture);
    textures.clear();
  }
  scratchWidth_ = 0u;
  scratchHeight_ = 0u;
  scratchRingCount_ = 0u;
  outputScratchFormat_ = Format::Count;
  for (TextureHandle lut : luts_)
    if (nuri::isValid(lut))
      gpu_.destroyTexture(lut);
  for (SamplerHandle sampler : samplers_)
    if (nuri::isValid(sampler))
      gpu_.destroySampler(sampler);
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
  auto area =
      createSmaaLut(gpu_, basePath / "smaa_area_rgba8.bin", kAreaLutWidth,
                    kAreaLutHeight, "spatial_aa_smaa_area_lut");
  if (area.hasError())
    return Result<bool, std::string>::makeError(area.error());
  luts_[AreaLut] = area.value();
  auto search =
      createSmaaLut(gpu_, basePath / "smaa_search_rgba8.bin", kSearchLutWidth,
                    kSearchLutHeight, "spatial_aa_smaa_search_lut");
  if (search.hasError())
    return Result<bool, std::string>::makeError(search.error());
  luts_[SearchLut] = search.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SpatialAAPass::ensureScratchTextures(FrameBuildContext &ctx) {
  if (!initializationError_.empty())
    return Result<bool, std::string>::makeError(initializationError_);
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
  for (auto &textures : scratchTextures_) {
    for (TextureHandle texture : textures)
      if (nuri::isValid(texture))
        gpu_.destroyTexture(texture);
    textures.clear();
  }
  scratchTextures_[EdgeScratch].reserve(ringCount);
  scratchTextures_[BlendScratch].reserve(ringCount);
  if (needsOutputScratch) {
    scratchTextures_[OutputScratch].reserve(ringCount);
  }
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
      if (texture.hasError())
        return Result<bool, std::string>::makeError(texture.error());
      scratchTextures_[kind].push_back(texture.value());
    }
    if (needsOutputScratch) {
      TextureDesc outputDesc = scratchDesc;
      outputDesc.format = outputScratchFormat;
      auto outputTexture =
          gpu_.createTexture(outputDesc, "transparent_spatial_aa_output");
      if (outputTexture.hasError()) {
        return Result<bool, std::string>::makeError(outputTexture.error());
      }
      scratchTextures_[OutputScratch].push_back(outputTexture.value());
    }
  }
  scratchWidth_ = dimensions.width;
  scratchHeight_ = dimensions.height;
  scratchRingCount_ = ringCount;
  outputScratchFormat_ = outputScratchFormat;
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
  if (placement_ == SpatialAAPlacement::PostTransparent) {
    const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
    const RenderSettings::AntiAliasingDebugSettings aaDebug =
        effectiveTemporalAADebugSettings(settings.antiAliasing);
    if (presentationAAPlanForFrame(ctx.frame).coverage ==
        CoverageMode::Sample4) {
      ctx.frame.metrics.antiAliasing.msaaSpatialCleanupEnabled =
          presentationAAPlanForFrame(ctx.frame).spatialCleanup ==
          SpatialCleanupPoint::PostTransparency;
    } else {
      ctx.frame.metrics.antiAliasing.taaTransparentPostSpatialCleanupEnabled =
          aaDebug.transparentPostTaaSpatialCleanup;
    }
  }
  return ensureScratchTextures(ctx);
}

Result<bool, std::string> SpatialAAPass::build(FrameBuildContext &ctx) {
  const bool postTransparent =
      placement_ == SpatialAAPlacement::PostTransparent;
  const uint32_t ringIndex = static_cast<uint32_t>(
      ctx.frame.frameIndex % scratchTextures_[EdgeScratch].size());
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
  const AntiAliasingMode aaMode =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode);
  const bool fallbackActive =
      postTransparent ? false : isSpatialFallbackActive(ctx.frame);
  const bool cleanupActive =
      postTransparent ? true : isSpatialCleanupActive(ctx.frame);
  const SpatialAAProfile profile =
      postTransparent && aaMode != AntiAliasingMode::MSAA4x
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
  if (!postTransparent) {
    aaMetrics.spatialAAEnabled = true;
    aaMetrics.spatialAAFallbackActive = fallbackActive;
    aaMetrics.spatialAACleanupActive = cleanupActive;
    aaMetrics.msaaSpatialCleanupEnabled =
        aaMode == AntiAliasingMode::MSAA4x && aaDebug.spatialPostMsaaCleanup;
    aaMetrics.msaaSpatialCleanupActive =
        aaMode == AntiAliasingMode::MSAA4x && cleanupActive;
    aaMetrics.spatialAAWidth = dimensions.width;
    aaMetrics.spatialAAHeight = dimensions.height;
    aaMetrics.spatialAATextureBytes = textureStorageBytes(gpu_, edgeTexture) +
                                      textureStorageBytes(gpu_, blendTexture);
    aaMetrics.spatialAATotalBytes =
        aaMetrics.spatialAATextureBytes *
        static_cast<uint64_t>(scratchTextures_[EdgeScratch].size());
    aaMetrics.spatialAALutTextureBytes =
        textureStorageBytes(gpu_, luts_[AreaLut]) +
        textureStorageBytes(gpu_, luts_[SearchLut]);
    aaMetrics.spatialAAEdgePixelEstimate = 0.0f;
    aaMetrics.spatialAAModifiedPixelEstimate = 0.0f;
    if (cleanupActive) {
      ++aaMetrics.spatialAACleanupFrameCount;
    }
  } else if (aaMode == AntiAliasingMode::MSAA4x) {
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
    if (aaMode == AntiAliasingMode::MSAA4x) {
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
      if (aaMode == AntiAliasingMode::MSAA4x) {
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
    if (aaMode == AntiAliasingMode::MSAA4x) {
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
    if (aaMode == AntiAliasingMode::MSAA4x) {
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
  if (postTransparent) {
    ctx.shared.frameColorGraphTexture = importSource.value();
    ctx.frame.sharedResources.frameColorGraphTexture = importSource.value();
  } else {
    ctx.shared.sceneColorGraphTexture = importSource.value();
    ctx.frame.sharedResources.sceneColorGraphTexture = importSource.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

void registerSpatialAAStage(RenderPipeline &pipeline, GPUDevice &gpu,
                            RuntimeCompositeConfig config,
                            SpatialAAPlacement placement) {
  pipeline.addStage(
      std::make_unique<SpatialAAPass>(gpu, std::move(config), placement),
      "SpatialAAFeature", "SpatialAAPass");
}

} // namespace nuri
