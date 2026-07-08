#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/spatial_aa_feature.h"

#include "nuri/gfx/frame/render_frame_context.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <filesystem>
#include <span>

#include "detail/smaa_area_tex.h"
#include "detail/smaa_search_tex.h"

namespace nuri {
namespace {

constexpr uint32_t kInvalidTextureBindlessIndex = 0xFFFFFFFFu;
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
  if (mode == AntiAliasingMode::MSAA4x) {
    return aaDebug.spatialPostMsaaCleanup && !isTaaDebugView(debugView);
  }
  if (mode != AntiAliasingMode::TAA || isTaaDebugView(debugView)) {
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

[[nodiscard]] RenderPipelineDesc
fullscreenPipelineDesc(Format colorFormat, ShaderHandle vertexShader,
                       ShaderHandle fragmentShader) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {colorFormat},
      .depthFormat = Format::Count,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
  };
}

[[nodiscard]] DrawItem makeFullscreenDraw(RenderPipelineHandle pipeline,
                                          std::span<const std::byte> constants,
                                          std::string_view label) {
  DrawItem draw{};
  draw.pipeline = pipeline;
  draw.vertexCount = 3u;
  draw.instanceCount = 1u;
  draw.pushConstants = constants;
  draw.debugLabel = label;
  draw.debugColor = kSpatialAADrawDebugColor;
  return draw;
}

[[nodiscard]] uint64_t textureFormatBytesPerPixel(Format format) noexcept {
  switch (format) {
  case Format::R8_UNORM:
    return 1u;
  case Format::R16_UNORM:
    return 2u;
  case Format::R32_UINT:
  case Format::R32_FLOAT:
  case Format::RG16_FLOAT:
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return 4u;
  case Format::RG32_FLOAT:
  case Format::RGBA16_FLOAT:
    return 8u;
  case Format::RGBA32_FLOAT:
    return 16u;
  case Format::D16_UNORM:
    return 2u;
  case Format::D32_FLOAT:
    return 4u;
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
  return static_cast<uint64_t>(dimensions.width) *
         static_cast<uint64_t>(dimensions.height) *
         static_cast<uint64_t>(std::max(dimensions.depth, 1u)) *
         textureFormatBytesPerPixel(gpu.getTextureFormat(texture));
}

[[nodiscard]] std::vector<std::byte> expandAreaTexToRgba8() {
  std::vector<std::byte> bytes;
  bytes.resize(static_cast<size_t>(AREATEX_WIDTH) *
               static_cast<size_t>(AREATEX_HEIGHT) * 4u);
  for (size_t i = 0u; i < static_cast<size_t>(AREATEX_WIDTH) *
                              static_cast<size_t>(AREATEX_HEIGHT);
       ++i) {
    bytes[i * 4u + 0u] = static_cast<std::byte>(areaTexBytes[i * 2u + 0u]);
    bytes[i * 4u + 1u] = static_cast<std::byte>(areaTexBytes[i * 2u + 1u]);
    bytes[i * 4u + 2u] = std::byte{0};
    bytes[i * 4u + 3u] = std::byte{255};
  }
  return bytes;
}

[[nodiscard]] std::vector<std::byte> expandSearchTexToRgba8() {
  std::vector<std::byte> bytes;
  bytes.resize(static_cast<size_t>(SEARCHTEX_WIDTH) *
               static_cast<size_t>(SEARCHTEX_HEIGHT) * 4u);
  for (size_t i = 0u; i < static_cast<size_t>(SEARCHTEX_WIDTH) *
                              static_cast<size_t>(SEARCHTEX_HEIGHT);
       ++i) {
    bytes[i * 4u + 0u] = static_cast<std::byte>(searchTexBytes[i]);
    bytes[i * 4u + 1u] = std::byte{0};
    bytes[i * 4u + 2u] = std::byte{0};
    bytes[i * 4u + 3u] = std::byte{255};
  }
  return bytes;
}

} // namespace

SpatialAAPass::SpatialAAPass(GPUDevice &gpu, RuntimeCompositeConfig config,
                             SpatialAAPlacement placement)
    : gpu_(gpu), config_(std::move(config)), placement_(placement) {
  const std::filesystem::path basePath = resolveShaderBasePath(config_);
  const std::filesystem::path vertexPath =
      config_.fullscreenVertex.empty() ? basePath / "fullscreen_copy.vert"
                                       : config_.fullscreenVertex;
  edgeResources_.vertexPath = vertexPath;
  edgeResources_.fragmentPath = basePath / "spatial_aa_edge.frag";
  blendResources_.vertexPath = vertexPath;
  blendResources_.fragmentPath = basePath / "spatial_aa_blend.frag";
  neighborhoodResources_.vertexPath = vertexPath;
  neighborhoodResources_.fragmentPath =
      basePath / "spatial_aa_neighborhood.frag";
}

SpatialAAPass::~SpatialAAPass() { destroyResources(); }

void SpatialAAPass::destroyFullscreenResources(FullscreenResources &resources) {
  if (nuri::isValid(resources.pipeline)) {
    gpu_.destroyRenderPipeline(resources.pipeline);
  }
  if (nuri::isValid(resources.vertexShader)) {
    gpu_.destroyShaderModule(resources.vertexShader);
  }
  if (nuri::isValid(resources.fragmentShader)) {
    gpu_.destroyShaderModule(resources.fragmentShader);
  }
  resources.pipeline = {};
  resources.pipelineColorFormat = Format::Count;
  resources.vertexShader = {};
  resources.fragmentShader = {};
  resources.shader.reset();
  resources.initialized = false;
}

void SpatialAAPass::destroyResources() {
  destroyFullscreenResources(edgeResources_);
  destroyFullscreenResources(blendResources_);
  destroyFullscreenResources(neighborhoodResources_);
  for (TextureHandle texture : edgeTextures_) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
    }
  }
  for (TextureHandle texture : blendTextures_) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
    }
  }
  for (TextureHandle texture : outputTextures_) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
    }
  }
  edgeTextures_.clear();
  blendTextures_.clear();
  outputTextures_.clear();
  scratchWidth_ = 0u;
  scratchHeight_ = 0u;
  scratchRingCount_ = 0u;
  outputScratchFormat_ = Format::Count;
  if (nuri::isValid(areaLutTexture_)) {
    gpu_.destroyTexture(areaLutTexture_);
  }
  if (nuri::isValid(searchLutTexture_)) {
    gpu_.destroyTexture(searchLutTexture_);
  }
  if (nuri::isValid(linearClampSampler_)) {
    gpu_.destroySampler(linearClampSampler_);
  }
  if (nuri::isValid(pointClampSampler_)) {
    gpu_.destroySampler(pointClampSampler_);
  }
  areaLutTexture_ = {};
  searchLutTexture_ = {};
  linearClampSampler_ = {};
  pointClampSampler_ = {};
}

Result<bool, std::string>
SpatialAAPass::ensureResources(FrameBuildContext &ctx) {
  const auto ensureFullscreen =
      [this](FullscreenResources &resources, std::string_view shaderName,
             Format colorFormat,
             std::string_view pipelineName) -> Result<bool, std::string> {
    if (!resources.initialized) {
      std::unique_ptr<Shader> shader = Shader::create(shaderName, gpu_);
      if (!shader) {
        return Result<bool, std::string>::makeError(
            std::string(shaderName) + ": failed to create shader");
      }
      auto vertexResult = shader->compileFromFile(resources.vertexPath.string(),
                                                  ShaderStage::Vertex);
      if (vertexResult.hasError()) {
        return Result<bool, std::string>::makeError(vertexResult.error());
      }
      auto fragmentResult = shader->compileFromFile(
          resources.fragmentPath.string(), ShaderStage::Fragment);
      if (fragmentResult.hasError()) {
        if (nuri::isValid(vertexResult.value())) {
          gpu_.destroyShaderModule(vertexResult.value());
        }
        return Result<bool, std::string>::makeError(fragmentResult.error());
      }
      resources.shader = std::move(shader);
      resources.vertexShader = vertexResult.value();
      resources.fragmentShader = fragmentResult.value();
      resources.initialized = true;
    }
    if (nuri::isValid(resources.pipeline) &&
        resources.pipelineColorFormat == colorFormat) {
      return Result<bool, std::string>::makeResult(true);
    }
    if (nuri::isValid(resources.pipeline)) {
      gpu_.destroyRenderPipeline(resources.pipeline);
      resources.pipeline = {};
    }
    auto pipelineResult = gpu_.createRenderPipeline(
        fullscreenPipelineDesc(colorFormat, resources.vertexShader,
                               resources.fragmentShader),
        pipelineName);
    if (pipelineResult.hasError()) {
      return Result<bool, std::string>::makeError(pipelineResult.error());
    }
    resources.pipeline = pipelineResult.value();
    resources.pipelineColorFormat = colorFormat;
    return Result<bool, std::string>::makeResult(true);
  };

  auto edgeResult = ensureFullscreen(edgeResources_, "spatial_aa_edge",
                                     Format::RGBA8_UNORM, "spatial_aa_edge");
  if (edgeResult.hasError()) {
    return edgeResult;
  }
  auto blendResult = ensureFullscreen(blendResources_, "spatial_aa_blend",
                                      Format::RGBA8_UNORM, "spatial_aa_blend");
  if (blendResult.hasError()) {
    return blendResult;
  }
  auto neighborhoodResult = ensureFullscreen(
      neighborhoodResources_, "spatial_aa_neighborhood",
      kFrameCompositionFrameColorFormat, "spatial_aa_neighborhood");
  if (neighborhoodResult.hasError()) {
    return neighborhoodResult;
  }

  if (!nuri::isValid(linearClampSampler_)) {
    auto samplerResult =
        gpu_.createSampler(SamplerDesc{.minFilter = SamplerFilter::Linear,
                                       .magFilter = SamplerFilter::Linear,
                                       .mipMode = SamplerMipMode::Disabled,
                                       .wrapU = SamplerWrapMode::Clamp,
                                       .wrapV = SamplerWrapMode::Clamp,
                                       .wrapW = SamplerWrapMode::Clamp},
                           "spatial_aa_linear_clamp");
    if (samplerResult.hasError()) {
      return Result<bool, std::string>::makeError(samplerResult.error());
    }
    linearClampSampler_ = samplerResult.value();
  }
  if (!nuri::isValid(pointClampSampler_)) {
    auto samplerResult =
        gpu_.createSampler(SamplerDesc{.minFilter = SamplerFilter::Nearest,
                                       .magFilter = SamplerFilter::Nearest,
                                       .mipMode = SamplerMipMode::Disabled,
                                       .wrapU = SamplerWrapMode::Clamp,
                                       .wrapV = SamplerWrapMode::Clamp,
                                       .wrapW = SamplerWrapMode::Clamp},
                           "spatial_aa_point_clamp");
    if (samplerResult.hasError()) {
      return Result<bool, std::string>::makeError(samplerResult.error());
    }
    pointClampSampler_ = samplerResult.value();
  }

  if (!nuri::isValid(areaLutTexture_)) {
    std::vector<std::byte> areaBytes = expandAreaTexToRgba8();
    auto textureResult = gpu_.createTexture(
        TextureDesc{
            .type = TextureType::Texture2D,
            .format = Format::RGBA8_UNORM,
            .dimensions = TextureDimensions{AREATEX_WIDTH, AREATEX_HEIGHT, 1u},
            .usage = TextureUsage::Sampled,
            .storage = Storage::Device,
            .numLayers = 1u,
            .numSamples = 1u,
            .numMipLevels = 1u,
            .data =
                std::span<const std::byte>(areaBytes.data(), areaBytes.size()),
            .dataNumMipLevels = 1u,
            .generateMipmaps = false},
        "spatial_aa_smaa_area_lut");
    if (textureResult.hasError()) {
      return Result<bool, std::string>::makeError(textureResult.error());
    }
    areaLutTexture_ = textureResult.value();
  }
  if (!nuri::isValid(searchLutTexture_)) {
    std::vector<std::byte> searchBytes = expandSearchTexToRgba8();
    auto textureResult = gpu_.createTexture(
        TextureDesc{.type = TextureType::Texture2D,
                    .format = Format::RGBA8_UNORM,
                    .dimensions = TextureDimensions{SEARCHTEX_WIDTH,
                                                    SEARCHTEX_HEIGHT, 1u},
                    .usage = TextureUsage::Sampled,
                    .storage = Storage::Device,
                    .numLayers = 1u,
                    .numSamples = 1u,
                    .numMipLevels = 1u,
                    .data = std::span<const std::byte>(searchBytes.data(),
                                                       searchBytes.size()),
                    .dataNumMipLevels = 1u,
                    .generateMipmaps = false},
        "spatial_aa_smaa_search_lut");
    if (textureResult.hasError()) {
      return Result<bool, std::string>::makeError(textureResult.error());
    }
    searchLutTexture_ = textureResult.value();
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
  const bool recreateScratch =
      scratchWidth_ != dimensions.width ||
      scratchHeight_ != dimensions.height || scratchRingCount_ != ringCount ||
      edgeTextures_.size() != ringCount || blendTextures_.size() != ringCount ||
      (needsOutputScratch && (outputTextures_.size() != ringCount ||
                              outputScratchFormat_ != outputScratchFormat)) ||
      (!needsOutputScratch && !outputTextures_.empty());
  if (!recreateScratch) {
    return Result<bool, std::string>::makeResult(true);
  }

  for (TextureHandle texture : edgeTextures_) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
    }
  }
  for (TextureHandle texture : blendTextures_) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
    }
  }
  for (TextureHandle texture : outputTextures_) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
    }
  }
  edgeTextures_.clear();
  blendTextures_.clear();
  outputTextures_.clear();
  edgeTextures_.reserve(ringCount);
  blendTextures_.reserve(ringCount);
  if (needsOutputScratch) {
    NURI_ASSERT(outputScratchFormat != Format::Count,
                "SpatialAAPass output scratch format must be resolved");
    outputTextures_.reserve(ringCount);
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
    auto edgeTexture = gpu_.createTexture(scratchDesc, "spatial_aa_edges");
    if (edgeTexture.hasError()) {
      return Result<bool, std::string>::makeError(edgeTexture.error());
    }
    auto blendTexture =
        gpu_.createTexture(scratchDesc, "spatial_aa_blend_weights");
    if (blendTexture.hasError()) {
      gpu_.destroyTexture(edgeTexture.value());
      return Result<bool, std::string>::makeError(blendTexture.error());
    }
    edgeTextures_.push_back(edgeTexture.value());
    blendTextures_.push_back(blendTexture.value());
    if (needsOutputScratch) {
      const TextureDesc outputDesc{
          .type = TextureType::Texture2D,
          .format = outputScratchFormat,
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
      auto outputTexture =
          gpu_.createTexture(outputDesc, "transparent_spatial_aa_output");
      if (outputTexture.hasError()) {
        return Result<bool, std::string>::makeError(outputTexture.error());
      }
      outputTextures_.push_back(outputTexture.value());
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
    ctx.frame.metrics.antiAliasing.taaTransparentPostSpatialCleanupEnabled =
        aaDebug.transparentPostTaaSpatialCleanup;
  }
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }
  return ensureResources(ctx);
}

Result<bool, std::string> SpatialAAPass::build(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(false);
  }
  auto resourcesResult = ensureResources(ctx);
  if (resourcesResult.hasError()) {
    return resourcesResult;
  }
  const bool postTransparent =
      placement_ == SpatialAAPlacement::PostTransparent;
  if (edgeTextures_.empty() || blendTextures_.empty() ||
      (postTransparent && outputTextures_.empty())) {
    return Result<bool, std::string>::makeResult(false);
  }

  const uint32_t ringIndex =
      static_cast<uint32_t>(ctx.frame.frameIndex % edgeTextures_.size());
  const TextureHandle edgeTexture = edgeTextures_[ringIndex];
  const TextureHandle blendTexture = blendTextures_[ringIndex];
  const TextureHandle sourceTexture = postTransparent
                                          ? ctx.shared.frameColorTexture
                                          : ctx.shared.sceneColorTexture;
  const TextureHandle outputTexture = postTransparent
                                          ? outputTextures_[ringIndex]
                                          : ctx.shared.frameColorTexture;
  const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(sourceTexture);
  const uint32_t edgeTexId = gpu_.getTextureBindlessIndex(edgeTexture);
  const uint32_t blendTexId = gpu_.getTextureBindlessIndex(blendTexture);
  const uint32_t areaTexId = gpu_.getTextureBindlessIndex(areaLutTexture_);
  const uint32_t searchTexId = gpu_.getTextureBindlessIndex(searchLutTexture_);
  const uint32_t outputTexId = gpu_.getTextureBindlessIndex(outputTexture);
  const uint32_t linearSamplerId =
      gpu_.getSamplerBindlessIndex(linearClampSampler_);
  const uint32_t pointSamplerId =
      gpu_.getSamplerBindlessIndex(pointClampSampler_);
  if (sourceTexId == kInvalidTextureBindlessIndex ||
      edgeTexId == kInvalidTextureBindlessIndex ||
      blendTexId == kInvalidTextureBindlessIndex ||
      areaTexId == kInvalidTextureBindlessIndex ||
      searchTexId == kInvalidTextureBindlessIndex ||
      outputTexId == kInvalidTextureBindlessIndex ||
      linearSamplerId == kInvalidTextureBindlessIndex ||
      pointSamplerId == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "SpatialAAPass::build: invalid bindless texture or sampler");
  }

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
      postTransparent
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
  const GpuTimingReport timingReport = gpu_.getLatestCompletedGpuTimingReport();
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
    aaMetrics.spatialAATextureCount =
        static_cast<uint32_t>(edgeTextures_.size() + blendTextures_.size());
    aaMetrics.spatialAALutTextureCount =
        nuri::isValid(areaLutTexture_) && nuri::isValid(searchLutTexture_) ? 2u
                                                                           : 0u;
    aaMetrics.spatialAATextureBytes = textureStorageBytes(gpu_, edgeTexture) +
                                      textureStorageBytes(gpu_, blendTexture);
    aaMetrics.spatialAATotalBytes = aaMetrics.spatialAATextureBytes *
                                    static_cast<uint64_t>(edgeTextures_.size());
    aaMetrics.spatialAALutTextureBytes =
        textureStorageBytes(gpu_, areaLutTexture_) +
        textureStorageBytes(gpu_, searchLutTexture_);
    aaMetrics.spatialAAEdgePixelEstimate = 0.0f;
    aaMetrics.spatialAAModifiedPixelEstimate = 0.0f;
    if (fallbackActive) {
      ++aaMetrics.spatialAAFallbackFrameCount;
    }
    if (cleanupActive) {
      ++aaMetrics.spatialAACleanupFrameCount;
    }
  } else {
    aaMetrics.taaTransparentPostSpatialCleanupActive = true;
  }

  const DrawItem edgeDraw = makeFullscreenDraw(
      edgeResources_.pipeline,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&baseConstants),
          sizeof(baseConstants)),
      "SpatialAAEdge");
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
    ++aaMetrics.taaTransparentPostSpatialAAPassCount;
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
        blendResources_.pipeline,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&baseConstants),
            sizeof(baseConstants)),
        "SpatialAABlend");
    const std::array<TextureHandle, 4> blendReads{
        edgeTexture, sourceTexture, areaLutTexture_, searchLutTexture_};
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
      ++aaMetrics.taaTransparentPostSpatialAAPassCount;
    } else {
      ++aaMetrics.spatialAAPassCount;
      ++aaMetrics.spatialAABlendPassCount;
    }
    aaMetrics.spatialAABandwidthEstimateBytes +=
        textureStorageBytes(gpu_, edgeTexture) +
        textureStorageBytes(gpu_, sourceTexture) +
        textureStorageBytes(gpu_, areaLutTexture_) +
        textureStorageBytes(gpu_, searchLutTexture_) +
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
      neighborhoodResources_.pipeline,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&outputConstants),
          sizeof(outputConstants)),
      "SpatialAAOutput");
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
    ++aaMetrics.taaTransparentPostSpatialAAPassCount;
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
      neighborhoodResources_.pipeline,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&copyConstants),
          sizeof(copyConstants)),
      "SpatialAACopyBack");
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
    ++aaMetrics.taaTransparentPostSpatialAAPassCount;
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

SpatialAAFeature::SpatialAAFeature(GPUDevice &gpu,
                                   RuntimeCompositeConfig config,
                                   SpatialAAPlacement placement)
    : spatialPass_(gpu, std::move(config), placement) {}

std::span<RenderFeaturePass *const> SpatialAAFeature::passes() noexcept {
  return std::span<RenderFeaturePass *const>(passes_.data(), passes_.size());
}

} // namespace nuri
