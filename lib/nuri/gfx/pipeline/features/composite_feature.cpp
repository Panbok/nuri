#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/composite_feature.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"

namespace nuri {
namespace {

constexpr uint32_t kSceneCopyFlagDownsample = 1u << 0u;
constexpr uint32_t kPresentFlagManualSrgbEncode = 1u << 0u;
constexpr uint32_t kPresentFlagPrimaryUseAgx = 1u << 1u;
constexpr uint32_t kPresentFlagCompareEnabled = 1u << 2u;
constexpr uint32_t kPresentFlagGrayCardDebug = 1u << 3u;
constexpr uint32_t kPresentFlagAcesLutAvailable = 1u << 4u;
constexpr uint32_t kPresentFlagAgxLutAvailable = 1u << 5u;
constexpr uint32_t kPresentFlagPrimaryLegacyFallback = 1u << 6u;
constexpr uint32_t kPresentFlagCompareLegacyFallback = 1u << 7u;
constexpr uint32_t kInvalidTextureBindlessIndex = 0xFFFFFFFFu;
constexpr uint32_t kDownsamplePassDebugColor = 0xff33aa88u;
constexpr uint32_t kResolvePassDebugColor = 0xff33cc88u;
constexpr uint32_t kPresentPassDebugColor = 0xff55cc88u;
constexpr uint32_t kHDRPassDebugColor = 0xff66dd99u;
constexpr uint32_t kDrawDebugColor = 0xff2299ddu;
constexpr uint64_t kInitialDebugFrames = 4u;
constexpr uint32_t kHDRPostFlagBloomEnabled = 1u << 0u;
constexpr uint32_t kHDRPostFlagAdaptationEnabled = 1u << 1u;
constexpr uint32_t kHDRPostFlagExposureHistoryValid = 1u << 2u;
constexpr uint32_t kHDRPostFlagReducedLuminanceSource = 1u << 3u;
constexpr uint32_t kHDRBloomModePrefilterDownsample = 0u;
constexpr uint32_t kHDRBloomModeDownsample = 1u;
constexpr uint32_t kHDRBloomModeUpsample = 2u;
constexpr uint32_t kHDRBloomModeCopy = 3u;
constexpr float kHDRBloomUpsampleScatter = 0.65f;

struct PresentToneMapState {
  RenderSettings::ToneMapSettings settings{};
  bool primaryUseAgx = false;
  bool compareEnabled = false;
  bool grayCardDebug = false;
  bool acesLutAvailable = false;
  bool agxLutAvailable = false;
  bool primaryHasLut = false;
  bool compareHasLut = false;
};

struct HDRExposurePushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t previousExposureTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t flags = 0u;
  float targetGray = kDefaultHDRAdaptationTargetGray;
  float speed = kDefaultHDRAdaptationSpeed;
  float minEv = kDefaultHDRAdaptationMinEv;
  float maxEv = kDefaultHDRAdaptationMaxEv;
  float deltaSeconds = 1.0f / 60.0f;
  float reserved0 = 0.0f;
  float reserved1 = 0.0f;
  float reserved2 = 0.0f;
};
static_assert(sizeof(HDRExposurePushConstants) <= 128u);

struct HDRLuminanceReducePushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t mode = 0u;
  uint32_t reserved0 = 0u;
  float texelSizeX = 1.0f;
  float texelSizeY = 1.0f;
  float reserved1 = 0.0f;
  float reserved2 = 0.0f;
};
static_assert(sizeof(HDRLuminanceReducePushConstants) <= 128u);

struct HDRBloomCompositePushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t bloomTexId = kInvalidTextureBindlessIndex;
  uint32_t exposureTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t flags = 0u;
  uint32_t debugView = 0u;
  uint32_t reserved0 = 0u;
  uint32_t reserved1 = 0u;
  float bloomStrength = kDefaultHDRBloomStrength;
  float fallbackExposureEv = 0.0f;
  float adaptationTargetGray = kDefaultHDRAdaptationTargetGray;
  float adaptationMinEv = kDefaultHDRAdaptationMinEv;
  float adaptationMaxEv = kDefaultHDRAdaptationMaxEv;
};
static_assert(sizeof(HDRBloomCompositePushConstants) <= 128u);

struct HDRBloomPushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t secondaryTexId = kInvalidTextureBindlessIndex;
  uint32_t exposureTexId = kInvalidTextureBindlessIndex;
  uint32_t sourceSamplerId = 0u;
  uint32_t mode = kHDRBloomModeDownsample;
  uint32_t flags = 0u;
  float threshold = kDefaultHDRBloomThreshold;
  float softKnee = kDefaultHDRBloomSoftKnee;
  float scatter = kHDRBloomUpsampleScatter;
  float manualExposureEv = 0.0f;
  float adaptationTargetGray = kDefaultHDRAdaptationTargetGray;
  float adaptationMinEv = kDefaultHDRAdaptationMinEv;
  float adaptationMaxEv = kDefaultHDRAdaptationMaxEv;
};
static_assert(sizeof(HDRBloomPushConstants) <= 128u);

RenderPipelineDesc fullscreenPipelineDesc(Format colorFormat,
                                          ShaderHandle vertexShader,
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

std::filesystem::path
resolveShaderBasePath(const FrameCompositionFeatureConfig &config) {
  if (!config.shaderBasePath.empty()) {
    return config.shaderBasePath;
  }
  if (!config.fullscreenVertex.empty()) {
    return config.fullscreenVertex.parent_path();
  }
  if (!config.sceneCopyFragment.empty()) {
    return config.sceneCopyFragment.parent_path();
  }
  return config.presentFragment.parent_path();
}

DrawItem makeFullscreenDraw(RenderPipelineHandle pipeline,
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

void publishRequestedCapture(RenderFrameContext &frame, GPUDevice &gpu,
                             std::string_view name, TextureHandle texture,
                             RenderCaptureValueKind kind,
                             RenderCaptureLifetimeClass lifetime,
                             std::string_view colorSpace,
                             std::string_view compareProfile,
                             std::string_view producerPassLabel,
                             std::string_view debugLabel) {
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
      .debugLabel = debugLabel,
  });
}

void destroyFullscreenPassResources(GPUDevice &gpu,
                                    FullscreenPassResources &resources) {
  if (nuri::isValid(resources.pipelineHandle)) {
    gpu.destroyRenderPipeline(resources.pipelineHandle);
  }
  if (nuri::isValid(resources.vertexShader)) {
    gpu.destroyShaderModule(resources.vertexShader);
  }
  if (nuri::isValid(resources.fragmentShader)) {
    gpu.destroyShaderModule(resources.fragmentShader);
  }
  resources.pipelineHandle = {};
  resources.pipelineColorFormat = Format::Count;
  resources.vertexShader = {};
  resources.fragmentShader = {};
  resources.shader.reset();
  resources.initialized = false;
}

void destroyFullscreenPipeline(GPUDevice &gpu,
                               FullscreenPassResources &resources) {
  if (nuri::isValid(resources.pipelineHandle)) {
    gpu.destroyRenderPipeline(resources.pipelineHandle);
  }
  resources.pipelineHandle = {};
  resources.pipelineColorFormat = Format::Count;
}

void destroyFullscreenShaders(GPUDevice &gpu,
                              FullscreenPassResources &resources) {
  if (nuri::isValid(resources.vertexShader)) {
    gpu.destroyShaderModule(resources.vertexShader);
  }
  if (nuri::isValid(resources.fragmentShader)) {
    gpu.destroyShaderModule(resources.fragmentShader);
  }
  resources.vertexShader = {};
  resources.fragmentShader = {};
  resources.shader.reset();
}

Result<bool, std::string>
createFullscreenPassShaders(GPUDevice &gpu, FullscreenPassResources &resources,
                            std::string_view shaderName,
                            std::string_view errorContext) {
  std::unique_ptr<Shader> shader = Shader::create(shaderName, gpu);
  if (!shader) {
    return Result<bool, std::string>::makeError(std::string(errorContext) +
                                                ": failed to create shader");
  }
  auto vertexResult = shader->compileFromFile(resources.vertexPath.string(),
                                              ShaderStage::Vertex);
  if (vertexResult.hasError()) {
    return Result<bool, std::string>::makeError(vertexResult.error());
  }
  const ShaderHandle vertexShader = vertexResult.value();
  auto fragmentResult = shader->compileFromFile(resources.fragmentPath.string(),
                                                ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    if (nuri::isValid(vertexShader)) {
      gpu.destroyShaderModule(vertexShader);
    }
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }
  destroyFullscreenShaders(gpu, resources);
  resources.shader = std::move(shader);
  resources.vertexShader = vertexShader;
  resources.fragmentShader = fragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ensureFullscreenPassInitialized(
    GPUDevice &gpu, FullscreenPassResources &resources,
    std::string_view shaderName, std::string_view errorContext) {
  if (resources.initialized) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaderResult =
      createFullscreenPassShaders(gpu, resources, shaderName, errorContext);
  if (shaderResult.hasError()) {
    return shaderResult;
  }
  resources.initialized = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ensureFullscreenPassPipeline(GPUDevice &gpu, FullscreenPassResources &resources,
                             Format colorFormat, std::string_view debugName,
                             std::string_view errorContext) {
  if (nuri::isValid(resources.pipelineHandle) &&
      resources.pipelineColorFormat == colorFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!nuri::isValid(resources.vertexShader) ||
      !nuri::isValid(resources.fragmentShader)) {
    return Result<bool, std::string>::makeError(std::string(errorContext) +
                                                ": invalid shader handle");
  }
  if (nuri::isValid(resources.pipelineHandle)) {
    gpu.destroyRenderPipeline(resources.pipelineHandle);
    resources.pipelineHandle = {};
  }
  auto pipelineResult = gpu.createRenderPipeline(
      fullscreenPipelineDesc(colorFormat, resources.vertexShader,
                             resources.fragmentShader),
      debugName);
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  resources.pipelineHandle = pipelineResult.value();
  resources.pipelineColorFormat = colorFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> addFullscreenTexturePass(
    RenderGraphBuilder &graph, RenderGraphTextureId colorTexture,
    std::span<const DrawItem> draws,
    std::span<const TextureHandle> textureReads, std::string_view debugLabel,
    uint32_t debugColor, bool markColorAsFrameOutput = false,
    GpuTimingScope gpuTimingScope = GpuTimingScope::None) {
  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  passDesc.colorTexture = colorTexture;
  passDesc.draws = draws;
  passDesc.dependencyTextures = textureReads;
  passDesc.debugLabel = debugLabel;
  passDesc.debugColor = debugColor;
  passDesc.markColorAsFrameOutput = markColorAsFrameOutput;
  passDesc.gpuTimingScope = gpuTimingScope;
  auto addResult = graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

TextureDesc makePostProcessTextureDesc(uint32_t width, uint32_t height) {
  return TextureDesc{
      .type = TextureType::Texture2D,
      .format = kFrameCompositionFrameColorFormat,
      .dimensions = {.width = width, .height = height, .depth = 1u},
      .usage = TextureUsage::AttachmentSampled,
      .storage = Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = 1u,
      .data = {},
      .dataNumMipLevels = 1u,
      .generateMipmaps = false,
  };
}

TextureDesc makeLuminanceTextureDesc(uint32_t width, uint32_t height) {
  TextureDesc desc = makePostProcessTextureDesc(width, height);
  desc.format = kFrameCompositionExposureFormat;
  return desc;
}

TextureDimensions bloomMipDimensions(uint32_t width, uint32_t height,
                                     uint32_t mipIndex) {
  uint32_t mipWidth = std::max(width, 1u);
  uint32_t mipHeight = std::max(height, 1u);
  const uint32_t downsampleCount = mipIndex + 1u;
  for (uint32_t i = 0u; i < downsampleCount; ++i) {
    mipWidth = std::max(mipWidth / 2u, 1u);
    mipHeight = std::max(mipHeight / 2u, 1u);
  }
  return TextureDimensions{.width = mipWidth, .height = mipHeight, .depth = 1u};
}

TextureDimensions luminanceMipDimensions(uint32_t width, uint32_t height,
                                         uint32_t mipIndex) {
  uint32_t mipWidth = std::max(width, 1u);
  uint32_t mipHeight = std::max(height, 1u);
  const uint32_t downsampleCount = mipIndex + 1u;
  for (uint32_t i = 0u; i < downsampleCount; ++i) {
    mipWidth = std::max((mipWidth + 1u) / 2u, 1u);
    mipHeight = std::max((mipHeight + 1u) / 2u, 1u);
  }
  return TextureDimensions{.width = mipWidth, .height = mipHeight, .depth = 1u};
}

uint32_t effectiveBloomMipCount(uint32_t width, uint32_t height,
                                uint32_t requestedMipCount) {
  if (requestedMipCount == 0u) {
    return 0u;
  }

  uint32_t mipCount = 0u;
  uint32_t mipWidth = std::max(width / 2u, 1u);
  uint32_t mipHeight = std::max(height / 2u, 1u);
  while (mipCount < requestedMipCount) {
    ++mipCount;
    if (mipWidth == 1u && mipHeight == 1u) {
      break;
    }
    mipWidth = std::max(mipWidth / 2u, 1u);
    mipHeight = std::max(mipHeight / 2u, 1u);
  }
  return mipCount;
}

uint32_t effectiveLuminanceMipCount(uint32_t width, uint32_t height) {
  uint32_t mipCount = 0u;
  uint32_t mipWidth = std::max(width, 1u);
  uint32_t mipHeight = std::max(height, 1u);
  while (mipWidth > 1u || mipHeight > 1u) {
    ++mipCount;
    mipWidth = std::max((mipWidth + 1u) / 2u, 1u);
    mipHeight = std::max((mipHeight + 1u) / 2u, 1u);
  }
  return std::max(mipCount, 1u);
}

uint32_t bytesPerPixel(Format format) {
  switch (format) {
  case Format::R8_UNORM:
    return 1u;
  case Format::R16_UNORM:
    return 2u;
  case Format::R32_FLOAT:
  case Format::R32_UINT:
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return 4u;
  case Format::RGBA16_FLOAT:
    return 8u;
  case Format::RGBA32_FLOAT:
    return 16u;
  default:
    break;
  }
  NURI_ASSERT(
      false, "HDR postprocess texture storage estimate missing Format value %u",
      static_cast<uint32_t>(format));
  return 0u;
}

[[nodiscard]] float
exposureEvFromLuminance(float luminance,
                        const RenderSettings::HDRPostProcessSettings &hdr) {
  const float safeLuminance = std::max(luminance, 1.0e-4f);
  const float safeTargetGray = std::max(hdr.adaptationTargetGray, 1.0e-4f);
  return std::clamp(std::log2(safeTargetGray / safeLuminance),
                    hdr.adaptationMinEv, hdr.adaptationMaxEv);
}

[[nodiscard]] std::optional<float>
readExposureLuminance(GPUDevice &gpu, TextureHandle texture) {
  if (!nuri::isValid(texture)) {
    return std::nullopt;
  }
  std::array<std::byte, sizeof(float)> bytes{};
  const TextureReadbackRegion region{
      .x = 0u, .y = 0u, .width = 1u, .height = 1u, .mipLevel = 0u, .layer = 0u};
  auto readResult = gpu.readTexture(texture, region, bytes);
  if (readResult.hasError()) {
    NURI_LOG_DEBUG("HDR postprocess exposure readback unavailable: %s",
                   readResult.error().c_str());
    return std::nullopt;
  }

  float luminance = 0.0f;
  std::memcpy(&luminance, bytes.data(), sizeof(luminance));
  return std::isfinite(luminance) ? std::optional<float>(luminance)
                                  : std::nullopt;
}

uint64_t textureStorageBytes(GPUDevice &gpu, TextureHandle texture) {
  if (!nuri::isValid(texture)) {
    return 0u;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  return static_cast<uint64_t>(std::max(dimensions.width, 1u)) *
         static_cast<uint64_t>(std::max(dimensions.height, 1u)) *
         static_cast<uint64_t>(bytesPerPixel(gpu.getTextureFormat(texture)));
}

bool shouldRunHDRPostProcess(const RenderFrameContext &frame) {
  RenderSettings::HDRPostProcessSettings hdr =
      renderSettingsOrDefault(frame).hdrPostProcess;
  sanitizeHDRPostProcessSettings(hdr);
  return hdr.bloomEnabled || hdr.adaptationEnabled ||
         hdr.debugView != HDRPostProcessDebugView::None;
}

bool shouldBuildBloomChain(const RenderSettings::HDRPostProcessSettings &hdr) {
  return hdr.bloomEnabled ||
         hdr.debugView == HDRPostProcessDebugView::BloomPrefilter ||
         hdr.debugView == HDRPostProcessDebugView::BloomFinal;
}

PresentToneMapState
buildPresentToneMapState(const RenderSettings::ToneMapSettings &settings,
                         bool acesLutAvailable, bool agxLutAvailable) {
  PresentToneMapState state{};
  state.settings = settings;
  sanitizeToneMapSettings(state.settings);
  state.primaryUseAgx = state.settings.operator_ == ToneMapper::AgX;
  state.compareEnabled = state.settings.sideBySideCompare;
  state.grayCardDebug = state.settings.grayCardDebug;
  state.acesLutAvailable = acesLutAvailable;
  state.agxLutAvailable = agxLutAvailable;
  state.primaryHasLut =
      state.primaryUseAgx ? state.agxLutAvailable : state.acesLutAvailable;
  state.compareHasLut =
      state.primaryUseAgx ? state.acesLutAvailable : state.agxLutAvailable;
  return state;
}

uint32_t buildPresentFlags(const PresentToneMapState &state,
                           bool manualSrgbEncode) {
  uint32_t flags = manualSrgbEncode ? kPresentFlagManualSrgbEncode : 0u;
  if (state.primaryUseAgx) {
    flags |= kPresentFlagPrimaryUseAgx;
  }
  if (state.compareEnabled) {
    flags |= kPresentFlagCompareEnabled;
  }
  if (state.grayCardDebug) {
    flags |= kPresentFlagGrayCardDebug;
  }
  if (state.acesLutAvailable) {
    flags |= kPresentFlagAcesLutAvailable;
  }
  if (state.agxLutAvailable) {
    flags |= kPresentFlagAgxLutAvailable;
  }
  if (!state.primaryHasLut) {
    flags |= kPresentFlagPrimaryLegacyFallback;
  }
  if (state.compareEnabled && !state.compareHasLut) {
    flags |= kPresentFlagCompareLegacyFallback;
  }
  return flags;
}

struct SceneResolveSource {
  TextureHandle texture{};
  AntiAliasingDebugView debugView = AntiAliasingDebugView::None;
};

[[nodiscard]] SceneResolveSource
resolveSceneResolveSource(const FrameBuildContext &ctx) {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  const RenderSettings::AntiAliasingDebugSettings aaDebug =
      effectiveTemporalAADebugSettings(settings.antiAliasing);
  const AntiAliasingDebugView debugView =
      sanitizeAntiAliasingDebugView(aaDebug.view);
  if (debugView == AntiAliasingDebugView::TAASceneColorHalfRes &&
      nuri::isValid(ctx.shared.sceneColorHalfResTexture)) {
    return {.texture = ctx.shared.sceneColorHalfResTexture,
            .debugView = debugView};
  }
  if (debugView == AntiAliasingDebugView::TAASceneColorQuarterRes &&
      nuri::isValid(ctx.shared.sceneColorQuarterResTexture)) {
    return {.texture = ctx.shared.sceneColorQuarterResTexture,
            .debugView = debugView};
  }
  return {.texture = ctx.shared.sceneColorTexture, .debugView = debugView};
}

[[nodiscard]] bool
isPostTaaSceneColorMipFrame(const RenderFrameContext &frame) noexcept {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  return isTemporalAAResolvedSceneColorOutput(settings.antiAliasing);
}

Result<uint32_t, std::string> resolveLutBindlessIndex(GPUDevice &gpu,
                                                      TextureHandle texture,
                                                      std::string_view name) {
  const uint32_t texId = gpu.getTextureBindlessIndex(texture);
  if (texId == kInvalidTextureBindlessIndex) {
    return Result<uint32_t, std::string>::makeError(
        std::string("PresentToneMapPass::build: invalid ") + std::string(name) +
        " tone-map LUT bindless index");
  }
  return Result<uint32_t, std::string>::makeResult(texId);
}

} // namespace

FullscreenRenderPass::FullscreenRenderPass(GPUDevice &gpu) : gpu_(gpu) {}

FullscreenRenderPass::~FullscreenRenderPass() {
  destroyFullscreenPassResources(gpu_, resources_);
}

Result<bool, std::string>
FullscreenRenderPass::ensureInitialized(std::string_view shaderName,
                                        std::string_view errorContext) {
  return ensureFullscreenPassInitialized(gpu_, resources_, shaderName,
                                         errorContext);
}

Result<bool, std::string>
FullscreenRenderPass::ensurePipeline(Format colorFormat,
                                     std::string_view debugName,
                                     std::string_view errorContext) {
  return ensureFullscreenPassPipeline(gpu_, resources_, colorFormat, debugName,
                                      errorContext);
}

Result<bool, std::string>
FullscreenRenderPass::createShaders(std::string_view shaderName,
                                    std::string_view errorContext) {
  return createFullscreenPassShaders(gpu_, resources_, shaderName,
                                     errorContext);
}

void FullscreenRenderPass::destroyPipeline() {
  destroyFullscreenPipeline(gpu_, resources_);
}

void FullscreenRenderPass::destroyShaders() {
  destroyFullscreenShaders(gpu_, resources_);
}

SceneColorDownsamplePass::SceneColorDownsamplePass(
    GPUDevice &gpu, FrameCompositionFeatureConfig config)
    : FullscreenRenderPass(gpu) {
  const std::filesystem::path basePath = resolveShaderBasePath(config);
  resources_.vertexPath = config.fullscreenVertex.empty()
                              ? basePath / "fullscreen_copy.vert"
                              : config.fullscreenVertex;
  resources_.fragmentPath = config.sceneCopyFragment.empty()
                                ? basePath / "scene_copy.frag"
                                : config.sceneCopyFragment;
}

bool SceneColorDownsamplePass::isEnabled(const FrameBuildContext &ctx) const {
  return nuri::isValid(resolveSceneColorMipTexture(ctx.frame, 0u)) &&
         nuri::isValid(resolveSceneColorMipTexture(ctx.frame, 1u)) &&
         nuri::isValid(resolveSceneColorMipTexture(ctx.frame, 2u));
}

Result<bool, std::string>
SceneColorDownsamplePass::prepare(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto initResult = ensureInitialized(
      "scene_color_downsample", "SceneColorDownsamplePass::createShaders");
  if (initResult.hasError()) {
    return initResult;
  }
  return ensurePipeline(kFrameCompositionSceneColorFormat,
                        "scene_color_downsample",
                        "SceneColorDownsamplePass::ensurePipeline");
}

Result<bool, std::string>
SceneColorDownsamplePass::build(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }

  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  const GpuTimingReport &timingReport = ctx.frame.gpuTiming;
  if (hasGpuTimingScope(timingReport, GpuTimingScope::SceneColorDownsample)) {
    aaMetrics.taaSceneColorDownsampleGpuTimeMs =
        timingReport.sceneColorDownsampleTimeMs;
    aaMetrics.taaSceneColorDownsampleGpuTimingSourceFrameIndex =
        timingReport.sceneColorDownsampleSourceFrameIndex;
    aaMetrics.taaSceneColorDownsampleGpuTimingAvailable = 1u;
  }

  const bool isPostTaaSceneColorMip = isPostTaaSceneColorMipFrame(ctx.frame);
  for (uint32_t mipLevel = 1u; mipLevel < kFrameCompositionSceneColorMipCount;
       ++mipLevel) {
    const TextureHandle source =
        resolveSceneColorMipTexture(ctx.frame, mipLevel - 1u);
    const TextureHandle destination =
        resolveSceneColorMipTexture(ctx.frame, mipLevel);
    NURI_ASSERT(nuri::isValid(source) && nuri::isValid(destination),
                "SceneColorDownsamplePass::build: scene color mip chain is "
                "incomplete");
    const std::string_view debugLabel = mipLevel == 1u
                                            ? "Scene Color Downsample Half"
                                            : "Scene Color Downsample Quarter";
    const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(source);
    if (sourceTexId == kInvalidTextureBindlessIndex) {
      return Result<bool, std::string>::makeError(
          "SceneColorDownsamplePass::build: invalid source texture bindless "
          "index");
    }
    const CopyPushConstants pushConstants{
        .sourceTexId = sourceTexId,
        .sourceSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u),
        .flags = kSceneCopyFlagDownsample,
        .reserved0 = 0u,
    };
    const DrawItem draw = makeFullscreenDraw(
        resources_.pipelineHandle,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&pushConstants),
            sizeof(pushConstants)),
        debugLabel);
    auto colorImportResult =
        ctx.graph.importTexture(destination, "scene_color_mip");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    const std::span<const TextureHandle> textureReads(&source, 1u);
    auto addResult = addFullscreenTexturePass(
        ctx.graph, colorImportResult.value(),
        std::span<const DrawItem>(&draw, 1u), textureReads, debugLabel,
        kDownsamplePassDebugColor, false, GpuTimingScope::SceneColorDownsample);
    if (addResult.hasError()) {
      return addResult;
    }
    if (aaMetrics.taaResolvedSceneColorPublished || isPostTaaSceneColorMip) {
      ++aaMetrics.taaPostResolveSceneColorMipPassCount;
      aaMetrics.taaPostResolveSceneColorMipChainGenerated =
          aaMetrics.taaPostResolveSceneColorMipPassCount >=
          kFrameCompositionSceneColorMipCount - 1u;
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

SceneResolvePass::SceneResolvePass(GPUDevice &gpu,
                                   FrameCompositionFeatureConfig config)
    : FullscreenRenderPass(gpu) {
  const std::filesystem::path basePath = resolveShaderBasePath(config);
  resources_.vertexPath = config.fullscreenVertex.empty()
                              ? basePath / "fullscreen_copy.vert"
                              : config.fullscreenVertex;
  resources_.fragmentPath = config.sceneCopyFragment.empty()
                                ? basePath / "scene_copy.frag"
                                : config.sceneCopyFragment;
}

bool SceneResolvePass::isEnabled(const FrameBuildContext &ctx) const {
  return nuri::isValid(ctx.shared.sceneColorTexture) &&
         nuri::isValid(ctx.shared.frameColorTexture);
}

Result<bool, std::string> SceneResolvePass::prepare(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto initResult =
      ensureInitialized("scene_resolve", "SceneResolvePass::createShaders");
  if (initResult.hasError()) {
    return initResult;
  }
  return ensurePipeline(kFrameCompositionFrameColorFormat, "scene_resolve",
                        "SceneResolvePass::ensurePipeline");
}

Result<bool, std::string> SceneResolvePass::build(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const SceneResolveSource source = resolveSceneResolveSource(ctx);
  const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(source.texture);
  if (sourceTexId == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "SceneResolvePass::build: invalid scene color bindless index");
  }
  const bool mipDebugView =
      source.debugView == AntiAliasingDebugView::TAASceneColorHalfRes ||
      source.debugView == AntiAliasingDebugView::TAASceneColorQuarterRes;
  if (mipDebugView) {
    ctx.frame.metrics.antiAliasing.taaSceneColorMipDebugViewRendered = true;
  }

  const CopyPushConstants pushConstants{
      .sourceTexId = sourceTexId,
      .sourceSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u),
      .flags = 0u,
      .reserved0 = 0u,
  };
  const DrawItem draw = makeFullscreenDraw(
      resources_.pipelineHandle,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pushConstants),
          sizeof(pushConstants)),
      "Scene Resolve");
  if (ctx.frame.frameIndex < kInitialDebugFrames) {
    NURI_LOG_DEBUG("SceneResolvePass::build: frame=%" PRIu64
                   " sourceHandle=%u:%u sourceTexId=%u targetHandle=%u:%u",
                   ctx.frame.frameIndex, source.texture.index,
                   source.texture.generation, sourceTexId,
                   ctx.shared.frameColorTexture.index,
                   ctx.shared.frameColorTexture.generation);
  }

  auto colorImportResult =
      ctx.graph.importTexture(ctx.shared.frameColorTexture, "frame_color");
  if (colorImportResult.hasError()) {
    return Result<bool, std::string>::makeError(colorImportResult.error());
  }
  const std::span<const TextureHandle> textureReads(&source.texture, 1u);
  auto addResult = addFullscreenTexturePass(
      ctx.graph, colorImportResult.value(),
      std::span<const DrawItem>(&draw, 1u), textureReads, "Scene Resolve Pass",
      kResolvePassDebugColor);
  if (addResult.hasError()) {
    return addResult;
  }

  ctx.shared.frameColorGraphTexture = colorImportResult.value();
  return Result<bool, std::string>::makeResult(true);
}

HDRExposureAdaptPass::HDRExposureAdaptPass(GPUDevice &gpu,
                                           FrameCompositionFeatureConfig config)
    : FullscreenRenderPass(gpu) {
  const std::filesystem::path basePath = resolveShaderBasePath(config);
  resources_.vertexPath = config.fullscreenVertex.empty()
                              ? basePath / "fullscreen_copy.vert"
                              : config.fullscreenVertex;
  resources_.fragmentPath = config.hdrExposureAdaptFragment.empty()
                                ? basePath / "hdr_exposure_adapt.frag"
                                : config.hdrExposureAdaptFragment;
  luminanceResources_.vertexPath = resources_.vertexPath;
  luminanceResources_.fragmentPath =
      config.hdrLuminanceReduceFragment.empty()
          ? basePath / "hdr_luminance_reduce.frag"
          : config.hdrLuminanceReduceFragment;
}

HDRExposureAdaptPass::~HDRExposureAdaptPass() {
  destroyLuminanceTextures();
  destroyFullscreenPassResources(gpu_, luminanceResources_);
}

void HDRExposureAdaptPass::destroyLuminanceTextures() {
  for (std::vector<TextureHandle> &mipTextures : luminanceTextures_) {
    for (TextureHandle texture : mipTextures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
      }
    }
  }
  luminanceTextures_.clear();
  luminanceWidth_ = 0u;
  luminanceHeight_ = 0u;
  luminanceRingCount_ = 0u;
  luminanceMipCount_ = 0u;
}

Result<bool, std::string> HDRExposureAdaptPass::ensureLuminanceTextures() {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t width = static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t height = static_cast<uint32_t>(std::max(framebufferHeight, 1));
  const uint32_t ringCount = std::max(1u, gpu_.getSwapchainImageCount());
  const uint32_t mipCount = effectiveLuminanceMipCount(width, height);

  if (luminanceWidth_ == width && luminanceHeight_ == height &&
      luminanceRingCount_ == ringCount && luminanceMipCount_ == mipCount &&
      luminanceTextures_.size() == mipCount) {
    return Result<bool, std::string>::makeResult(true);
  }

  destroyLuminanceTextures();
  luminanceTextures_.resize(mipCount);
  for (uint32_t mip = 0u; mip < mipCount; ++mip) {
    const TextureDimensions dimensions =
        luminanceMipDimensions(width, height, mip);
    const TextureDesc desc =
        makeLuminanceTextureDesc(dimensions.width, dimensions.height);
    luminanceTextures_[mip].resize(ringCount);
    for (uint32_t ring = 0u; ring < ringCount; ++ring) {
      const std::string debugName = "hdr_luminance_reduce_mip_" +
                                    std::to_string(mip) + "_slot_" +
                                    std::to_string(ring);
      auto createResult = gpu_.createTexture(desc, debugName);
      if (createResult.hasError()) {
        destroyLuminanceTextures();
        return Result<bool, std::string>::makeError(createResult.error());
      }
      luminanceTextures_[mip][ring] = createResult.value();
    }
  }

  luminanceWidth_ = width;
  luminanceHeight_ = height;
  luminanceRingCount_ = ringCount;
  luminanceMipCount_ = mipCount;
  return Result<bool, std::string>::makeResult(true);
}

bool HDRExposureAdaptPass::isEnabled(const FrameBuildContext &ctx) const {
  RenderSettings::HDRPostProcessSettings hdr =
      renderSettingsOrDefault(ctx.frame).hdrPostProcess;
  sanitizeHDRPostProcessSettings(hdr);
  return hdr.adaptationEnabled && nuri::isValid(ctx.shared.frameColorTexture) &&
         nuri::isValid(ctx.shared.exposureWriteTexture);
}

Result<bool, std::string>
HDRExposureAdaptPass::prepare(FrameBuildContext &ctx) {
  RenderSettings::HDRPostProcessSettings hdr =
      renderSettingsOrDefault(ctx.frame).hdrPostProcess;
  sanitizeHDRPostProcessSettings(hdr);
  ctx.frame.metrics.hdrPostProcess.adaptationEnabled = hdr.adaptationEnabled;
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto initResult = ensureInitialized("hdr_exposure_adapt",
                                      "HDRExposureAdaptPass::createShaders");
  if (initResult.hasError()) {
    return initResult;
  }
  auto luminanceInitResult = ensureFullscreenPassInitialized(
      gpu_, luminanceResources_, "hdr_luminance_reduce",
      "HDRExposureAdaptPass::createLuminanceShaders");
  if (luminanceInitResult.hasError()) {
    return luminanceInitResult;
  }
  auto exposurePipelineResult =
      ensurePipeline(kFrameCompositionExposureFormat, "hdr_exposure_adapt",
                     "HDRExposureAdaptPass::ensurePipeline");
  if (exposurePipelineResult.hasError()) {
    return exposurePipelineResult;
  }
  auto luminancePipelineResult = ensureFullscreenPassPipeline(
      gpu_, luminanceResources_, kFrameCompositionExposureFormat,
      "hdr_luminance_reduce", "HDRExposureAdaptPass::ensureLuminancePipeline");
  if (luminancePipelineResult.hasError()) {
    return luminancePipelineResult;
  }
  return ensureLuminanceTextures();
}

Result<bool, std::string> HDRExposureAdaptPass::build(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }

  RenderSettings::HDRPostProcessSettings hdr =
      renderSettingsOrDefault(ctx.frame).hdrPostProcess;
  sanitizeHDRPostProcessSettings(hdr);
  const TextureHandle source = ctx.shared.frameColorTexture;
  const TextureHandle exposureWrite = ctx.shared.exposureWriteTexture;
  const TextureHandle exposureRead = ctx.shared.exposureReadTexture;
  const uint32_t frameColorTexId = gpu_.getTextureBindlessIndex(source);
  const uint32_t previousExposureTexId =
      ctx.shared.exposureHistoryValid && nuri::isValid(exposureRead)
          ? gpu_.getTextureBindlessIndex(exposureRead)
          : kInvalidTextureBindlessIndex;
  if (frameColorTexId == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "HDRExposureAdaptPass::build: invalid frame color bindless index");
  }

  auto sourceImport =
      nuri::isValid(ctx.shared.frameColorGraphTexture)
          ? Result<RenderGraphTextureId, std::string>::makeResult(
                ctx.shared.frameColorGraphTexture)
          : ctx.graph.importTexture(source, "hdr_source_color");
  if (sourceImport.hasError()) {
    return Result<bool, std::string>::makeError(sourceImport.error());
  }
  const uint32_t ringIndex =
      luminanceTextures_.empty()
          ? 0u
          : static_cast<uint32_t>(ctx.frame.frameIndex %
                                  luminanceTextures_[0u].size());
  const uint32_t samplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u);
  TextureHandle meteringSource = source;
  RenderGraphTextureId meteringSourceImport = sourceImport.value();
  uint32_t sourceTexId = frameColorTexId;
  bool reducedLuminanceSource = false;
  if (!luminanceTextures_.empty()) {
    RenderGraphTextureId previousImport = sourceImport.value();
    TextureHandle previousTexture = source;
    for (uint32_t mip = 0u; mip < luminanceMipCount_; ++mip) {
      const TextureHandle target = luminanceTextures_[mip][ringIndex];
      const uint32_t reduceSourceTexId =
          gpu_.getTextureBindlessIndex(previousTexture);
      if (reduceSourceTexId == kInvalidTextureBindlessIndex) {
        return Result<bool, std::string>::makeError(
            "HDRExposureAdaptPass::build: invalid luminance source bindless "
            "index");
      }
      auto targetImport =
          ctx.graph.importTexture(target, "hdr_luminance_reduce");
      if (targetImport.hasError()) {
        return Result<bool, std::string>::makeError(targetImport.error());
      }

      const TextureDimensions sourceDimensions =
          gpu_.getTextureDimensions(previousTexture);
      const HDRLuminanceReducePushConstants reduceConstants{
          .sourceTexId = reduceSourceTexId,
          .sourceSamplerId = samplerId,
          .mode = mip == 0u ? 0u : 1u,
          .reserved0 = 0u,
          .texelSizeX =
              1.0f / static_cast<float>(std::max(sourceDimensions.width, 1u)),
          .texelSizeY =
              1.0f / static_cast<float>(std::max(sourceDimensions.height, 1u)),
      };
      const DrawItem reduceDraw = makeFullscreenDraw(
          luminanceResources_.pipelineHandle,
          std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(&reduceConstants),
              sizeof(reduceConstants)),
          "HDRLuminanceReduce");
      RenderGraphGraphicsPassDesc reducePass{};
      reducePass.color = {.loadOp = LoadOp::Clear,
                          .storeOp = StoreOp::Store,
                          .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
      reducePass.colorTexture = targetImport.value();
      reducePass.dependencyTextures =
          std::span<const TextureHandle>(&previousTexture, 1u);
      reducePass.draws = std::span<const DrawItem>(&reduceDraw, 1u);
      reducePass.gpuTimingScope = GpuTimingScope::HDRPostProcess;
      reducePass.debugLabel = "HDR Luminance Reduce Pass";
      reducePass.debugColor = kHDRPassDebugColor;
      auto addReduce = ctx.graph.addGraphicsPass(reducePass);
      if (addReduce.hasError()) {
        return Result<bool, std::string>::makeError(addReduce.error());
      }
      auto sourceRead =
          ctx.graph.addTextureRead(addReduce.value(), previousImport);
      if (sourceRead.hasError()) {
        return Result<bool, std::string>::makeError(sourceRead.error());
      }

      previousTexture = target;
      previousImport = targetImport.value();
    }
    meteringSource = previousTexture;
    meteringSourceImport = previousImport;
    sourceTexId = gpu_.getTextureBindlessIndex(meteringSource);
    if (sourceTexId == kInvalidTextureBindlessIndex) {
      return Result<bool, std::string>::makeError(
          "HDRExposureAdaptPass::build: invalid reduced luminance bindless "
          "index");
    }
    reducedLuminanceSource = true;
  }

  const uint32_t flags =
      kHDRPostFlagAdaptationEnabled |
      (previousExposureTexId != kInvalidTextureBindlessIndex
           ? kHDRPostFlagExposureHistoryValid
           : 0u) |
      (reducedLuminanceSource ? kHDRPostFlagReducedLuminanceSource : 0u);
  const HDRExposurePushConstants pushConstants{
      .sourceTexId = sourceTexId,
      .previousExposureTexId = previousExposureTexId,
      .sourceSamplerId = samplerId,
      .flags = flags,
      .targetGray = hdr.adaptationTargetGray,
      .speed = hdr.adaptationSpeed,
      .minEv = hdr.adaptationMinEv,
      .maxEv = hdr.adaptationMaxEv,
      .deltaSeconds = static_cast<float>(std::max(ctx.frame.deltaSeconds, 0.0)),
  };

  auto exposureImport =
      ctx.graph.importTexture(exposureWrite, "hdr_exposure_write");
  if (exposureImport.hasError()) {
    return Result<bool, std::string>::makeError(exposureImport.error());
  }
  std::array<TextureHandle, 2> reads{meteringSource, {}};
  size_t readCount = 1u;
  RenderGraphTextureId previousExposureImport{};
  if (previousExposureTexId != kInvalidTextureBindlessIndex) {
    auto importPrevious =
        ctx.graph.importTexture(exposureRead, "hdr_exposure_read");
    if (importPrevious.hasError()) {
      return Result<bool, std::string>::makeError(importPrevious.error());
    }
    previousExposureImport = importPrevious.value();
    reads[readCount++] = exposureRead;
  }

  const DrawItem draw = makeFullscreenDraw(
      resources_.pipelineHandle,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pushConstants),
          sizeof(pushConstants)),
      "HDRExposureAdapt");
  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  passDesc.colorTexture = exposureImport.value();
  passDesc.dependencyTextures =
      std::span<const TextureHandle>(reads.data(), readCount);
  passDesc.draws = std::span<const DrawItem>(&draw, 1u);
  passDesc.gpuTimingScope = GpuTimingScope::HDRPostProcess;
  passDesc.debugLabel = "HDR Exposure Adapt Pass";
  passDesc.debugColor = kHDRPassDebugColor;
  auto addResult = ctx.graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  auto sourceRead =
      ctx.graph.addTextureRead(addResult.value(), meteringSourceImport);
  if (sourceRead.hasError()) {
    return Result<bool, std::string>::makeError(sourceRead.error());
  }
  if (nuri::isValid(previousExposureImport)) {
    auto previousRead =
        ctx.graph.addTextureRead(addResult.value(), previousExposureImport);
    if (previousRead.hasError()) {
      return Result<bool, std::string>::makeError(previousRead.error());
    }
  }
  ctx.shared.historyWriteRequirements |= FrameTextureRequirementFlags::Exposure;

  HDRPostProcessFrameMetrics &metrics = ctx.frame.metrics.hdrPostProcess;
  metrics.adaptationActive = true;
  metrics.adaptationPassCount += 1u;
  metrics.luminancePassCount +=
      reducedLuminanceSource ? luminanceMipCount_ : 0u;
  metrics.exposureHistoryValid = ctx.shared.exposureHistoryValid;
  metrics.textureCount += 1u;
  metrics.textureBytes += textureStorageBytes(gpu_, exposureWrite);
  if (reducedLuminanceSource) {
    metrics.textureCount += luminanceMipCount_;
    for (uint32_t mip = 0u; mip < luminanceMipCount_; ++mip) {
      metrics.textureBytes +=
          textureStorageBytes(gpu_, luminanceTextures_[mip][ringIndex]);
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

HDRBloomCompositePass::HDRBloomCompositePass(
    GPUDevice &gpu, FrameCompositionFeatureConfig config)
    : FullscreenRenderPass(gpu) {
  const std::filesystem::path basePath = resolveShaderBasePath(config);
  resources_.vertexPath = config.fullscreenVertex.empty()
                              ? basePath / "fullscreen_copy.vert"
                              : config.fullscreenVertex;
  resources_.fragmentPath = config.hdrBloomCompositeFragment.empty()
                                ? basePath / "hdr_bloom_composite.frag"
                                : config.hdrBloomCompositeFragment;
  bloomResources_.vertexPath = config.fullscreenVertex.empty()
                                   ? basePath / "fullscreen_copy.vert"
                                   : config.fullscreenVertex;
  bloomResources_.fragmentPath = config.hdrBloomFragment.empty()
                                     ? basePath / "hdr_bloom.frag"
                                     : config.hdrBloomFragment;
}

HDRBloomCompositePass::~HDRBloomCompositePass() {
  destroyBloomTextures();
  destroyOutputTextures();
  destroyFullscreenPassResources(gpu_, bloomResources_);
}

void HDRBloomCompositePass::destroyOutputTextures() {
  for (TextureHandle texture : outputTextures_) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
    }
  }
  outputTextures_.clear();
  outputWidth_ = 0u;
  outputHeight_ = 0u;
  outputRingCount_ = 0u;
}

void HDRBloomCompositePass::destroyBloomTextures() {
  for (std::vector<TextureHandle> &mipTextures : bloomDownsampleTextures_) {
    for (TextureHandle texture : mipTextures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
      }
    }
  }
  for (std::vector<TextureHandle> &mipTextures : bloomUpsampleTextures_) {
    for (TextureHandle texture : mipTextures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
      }
    }
  }
  bloomDownsampleTextures_.clear();
  bloomUpsampleTextures_.clear();
  bloomWidth_ = 0u;
  bloomHeight_ = 0u;
  bloomRingCount_ = 0u;
  bloomMipCount_ = 0u;
}

Result<bool, std::string> HDRBloomCompositePass::ensureOutputTextures() {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t width = static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t height = static_cast<uint32_t>(std::max(framebufferHeight, 1));
  const uint32_t ringCount = std::max(1u, gpu_.getSwapchainImageCount());
  if (outputWidth_ == width && outputHeight_ == height &&
      outputRingCount_ == ringCount && outputTextures_.size() == ringCount) {
    return Result<bool, std::string>::makeResult(true);
  }
  destroyOutputTextures();
  outputTextures_.resize(ringCount);
  const TextureDesc desc = makePostProcessTextureDesc(width, height);
  for (uint32_t i = 0u; i < ringCount; ++i) {
    const std::string debugName = "hdr_postprocess_output_" + std::to_string(i);
    auto createResult = gpu_.createTexture(desc, debugName);
    if (createResult.hasError()) {
      destroyOutputTextures();
      return Result<bool, std::string>::makeError(createResult.error());
    }
    outputTextures_[i] = createResult.value();
  }
  outputWidth_ = width;
  outputHeight_ = height;
  outputRingCount_ = ringCount;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
HDRBloomCompositePass::ensureBloomTextures(uint32_t requestedMipCount) {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t width = static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t height = static_cast<uint32_t>(std::max(framebufferHeight, 1));
  const uint32_t ringCount = std::max(1u, gpu_.getSwapchainImageCount());
  const uint32_t mipCount =
      effectiveBloomMipCount(width, height, requestedMipCount);

  if (mipCount == 0u) {
    destroyBloomTextures();
    return Result<bool, std::string>::makeResult(true);
  }

  if (bloomWidth_ == width && bloomHeight_ == height &&
      bloomRingCount_ == ringCount && bloomMipCount_ == mipCount &&
      bloomDownsampleTextures_.size() == mipCount &&
      bloomUpsampleTextures_.size() == mipCount) {
    return Result<bool, std::string>::makeResult(true);
  }

  destroyBloomTextures();
  bloomDownsampleTextures_.resize(mipCount);
  bloomUpsampleTextures_.resize(mipCount);
  for (uint32_t mip = 0u; mip < mipCount; ++mip) {
    const TextureDimensions dimensions = bloomMipDimensions(width, height, mip);
    const TextureDesc desc =
        makePostProcessTextureDesc(dimensions.width, dimensions.height);
    bloomDownsampleTextures_[mip].resize(ringCount);
    bloomUpsampleTextures_[mip].resize(ringCount);
    for (uint32_t ring = 0u; ring < ringCount; ++ring) {
      const std::string downsampleName = "hdr_bloom_downsample_mip_" +
                                         std::to_string(mip) + "_slot_" +
                                         std::to_string(ring);
      auto downsampleResult = gpu_.createTexture(desc, downsampleName);
      if (downsampleResult.hasError()) {
        destroyBloomTextures();
        return Result<bool, std::string>::makeError(downsampleResult.error());
      }
      bloomDownsampleTextures_[mip][ring] = downsampleResult.value();

      const std::string upsampleName = "hdr_bloom_upsample_mip_" +
                                       std::to_string(mip) + "_slot_" +
                                       std::to_string(ring);
      auto upsampleResult = gpu_.createTexture(desc, upsampleName);
      if (upsampleResult.hasError()) {
        destroyBloomTextures();
        return Result<bool, std::string>::makeError(upsampleResult.error());
      }
      bloomUpsampleTextures_[mip][ring] = upsampleResult.value();
    }
  }
  bloomWidth_ = width;
  bloomHeight_ = height;
  bloomRingCount_ = ringCount;
  bloomMipCount_ = mipCount;
  return Result<bool, std::string>::makeResult(true);
}

bool HDRBloomCompositePass::isEnabled(const FrameBuildContext &ctx) const {
  return shouldRunHDRPostProcess(ctx.frame) &&
         nuri::isValid(ctx.shared.frameColorTexture);
}

Result<bool, std::string>
HDRBloomCompositePass::prepare(FrameBuildContext &ctx) {
  RenderSettings::HDRPostProcessSettings hdr =
      renderSettingsOrDefault(ctx.frame).hdrPostProcess;
  sanitizeHDRPostProcessSettings(hdr);
  HDRPostProcessFrameMetrics &metrics = ctx.frame.metrics.hdrPostProcess;
  metrics.bloomEnabled = hdr.bloomEnabled;
  metrics.adaptationEnabled = hdr.adaptationEnabled;
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }
  const bool needsBloomChain = shouldBuildBloomChain(hdr);
  auto initResult = ensureInitialized("hdr_bloom_composite",
                                      "HDRBloomCompositePass::createShaders");
  if (initResult.hasError()) {
    return initResult;
  }
  if (needsBloomChain) {
    auto bloomInitResult = ensureFullscreenPassInitialized(
        gpu_, bloomResources_, "hdr_bloom",
        "HDRBloomCompositePass::createBloomShaders");
    if (bloomInitResult.hasError()) {
      return bloomInitResult;
    }
  }
  auto pipelineResult =
      ensurePipeline(kFrameCompositionFrameColorFormat, "hdr_bloom_composite",
                     "HDRBloomCompositePass::ensurePipeline");
  if (pipelineResult.hasError()) {
    return pipelineResult;
  }
  if (needsBloomChain) {
    auto bloomPipelineResult = ensureFullscreenPassPipeline(
        gpu_, bloomResources_, kFrameCompositionFrameColorFormat, "hdr_bloom",
        "HDRBloomCompositePass::ensureBloomPipeline");
    if (bloomPipelineResult.hasError()) {
      return bloomPipelineResult;
    }
  }
  auto outputResult = ensureOutputTextures();
  if (outputResult.hasError()) {
    return outputResult;
  }
  return ensureBloomTextures(needsBloomChain ? hdr.bloomMaxMipCount : 0u);
}

Result<bool, std::string> HDRBloomCompositePass::build(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }
  RenderSettings::HDRPostProcessSettings hdr =
      renderSettingsOrDefault(ctx.frame).hdrPostProcess;
  sanitizeHDRPostProcessSettings(hdr);
  const bool needsBloomChain = shouldBuildBloomChain(hdr);

  auto outputResult = ensureOutputTextures();
  if (outputResult.hasError()) {
    return outputResult;
  }
  auto bloomTextureResult =
      ensureBloomTextures(needsBloomChain ? hdr.bloomMaxMipCount : 0u);
  if (bloomTextureResult.hasError()) {
    return bloomTextureResult;
  }
  if (outputTextures_.empty()) {
    return Result<bool, std::string>::makeError(
        "HDRBloomCompositePass::build: output texture ring is empty");
  }

  const uint32_t ringIndex =
      static_cast<uint32_t>(ctx.frame.frameIndex % outputTextures_.size());
  const TextureHandle source = ctx.shared.frameColorTexture;
  const TextureHandle output = outputTextures_[ringIndex];
  const TextureHandle exposure =
      hdr.adaptationEnabled ? ctx.shared.exposureWriteTexture : TextureHandle{};
  const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(source);
  const uint32_t outputTexId = gpu_.getTextureBindlessIndex(output);
  const uint32_t exposureTexId = nuri::isValid(exposure)
                                     ? gpu_.getTextureBindlessIndex(exposure)
                                     : kInvalidTextureBindlessIndex;
  if (sourceTexId == kInvalidTextureBindlessIndex ||
      outputTexId == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "HDRBloomCompositePass::build: invalid bindless texture index");
  }

  auto sourceImport =
      nuri::isValid(ctx.shared.frameColorGraphTexture)
          ? Result<RenderGraphTextureId, std::string>::makeResult(
                ctx.shared.frameColorGraphTexture)
          : ctx.graph.importTexture(source, "hdr_frame_color");
  if (sourceImport.hasError()) {
    return Result<bool, std::string>::makeError(sourceImport.error());
  }
  auto outputImport = ctx.graph.importTexture(output, "hdr_postprocess_output");
  if (outputImport.hasError()) {
    return Result<bool, std::string>::makeError(outputImport.error());
  }
  RenderGraphTextureId exposureImport{};
  if (hdr.adaptationEnabled && exposureTexId != kInvalidTextureBindlessIndex) {
    auto importExposure = ctx.graph.importTexture(exposure, "hdr_exposure");
    if (importExposure.hasError()) {
      return Result<bool, std::string>::makeError(importExposure.error());
    }
    exposureImport = importExposure.value();
  }

  const uint32_t samplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u);
  const float manualExposureEv =
      renderSettingsOrDefault(ctx.frame).toneMap.exposureEv;
  const bool bloomAvailable = needsBloomChain && bloomMipCount_ != 0u &&
                              !bloomDownsampleTextures_.empty() &&
                              !bloomUpsampleTextures_.empty();
  std::vector<RenderGraphTextureId> bloomDownsampleImports(bloomMipCount_);
  std::vector<RenderGraphTextureId> bloomUpsampleImports(bloomMipCount_);
  if (bloomAvailable) {
    for (uint32_t mip = 0u; mip < bloomMipCount_; ++mip) {
      const TextureHandle downsampleTarget =
          bloomDownsampleTextures_[mip][ringIndex];
      const TextureHandle downsampleSource =
          mip == 0u ? source : bloomDownsampleTextures_[mip - 1u][ringIndex];
      const uint32_t downsampleSourceTexId =
          gpu_.getTextureBindlessIndex(downsampleSource);
      if (downsampleSourceTexId == kInvalidTextureBindlessIndex) {
        return Result<bool, std::string>::makeError(
            "HDRBloomCompositePass::build: invalid bloom downsample source "
            "bindless index");
      }

      auto targetImport =
          ctx.graph.importTexture(downsampleTarget, "hdr_bloom_downsample");
      if (targetImport.hasError()) {
        return Result<bool, std::string>::makeError(targetImport.error());
      }
      bloomDownsampleImports[mip] = targetImport.value();
      const HDRBloomPushConstants downsampleConstants{
          .sourceTexId = downsampleSourceTexId,
          .secondaryTexId = kInvalidTextureBindlessIndex,
          .exposureTexId = exposureTexId,
          .sourceSamplerId = samplerId,
          .mode = mip == 0u ? kHDRBloomModePrefilterDownsample
                            : kHDRBloomModeDownsample,
          .flags = mip == 0u && exposureTexId != kInvalidTextureBindlessIndex
                       ? kHDRPostFlagAdaptationEnabled
                       : 0u,
          .threshold = hdr.bloomThreshold,
          .softKnee = hdr.bloomSoftKnee,
          .scatter = kHDRBloomUpsampleScatter,
          .manualExposureEv = manualExposureEv,
          .adaptationTargetGray = hdr.adaptationTargetGray,
          .adaptationMinEv = hdr.adaptationMinEv,
          .adaptationMaxEv = hdr.adaptationMaxEv,
      };
      const DrawItem downsampleDraw = makeFullscreenDraw(
          bloomResources_.pipelineHandle,
          std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(&downsampleConstants),
              sizeof(downsampleConstants)),
          "HDRBloomDownsample");
      RenderGraphGraphicsPassDesc downsamplePass{};
      downsamplePass.color = {.loadOp = LoadOp::Clear,
                              .storeOp = StoreOp::Store,
                              .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
      downsamplePass.colorTexture = targetImport.value();
      std::array<TextureHandle, 2> downsampleReads{downsampleSource, exposure};
      const size_t downsampleReadCount =
          mip == 0u && exposureTexId != kInvalidTextureBindlessIndex ? 2u : 1u;
      downsamplePass.dependencyTextures = std::span<const TextureHandle>(
          downsampleReads.data(), downsampleReadCount);
      downsamplePass.draws = std::span<const DrawItem>(&downsampleDraw, 1u);
      downsamplePass.gpuTimingScope = GpuTimingScope::HDRPostProcess;
      downsamplePass.debugLabel = "HDR Bloom Downsample Pass";
      downsamplePass.debugColor = kHDRPassDebugColor;
      auto addDownsample = ctx.graph.addGraphicsPass(downsamplePass);
      if (addDownsample.hasError()) {
        return Result<bool, std::string>::makeError(addDownsample.error());
      }
      const RenderGraphTextureId downsampleSourceImport =
          mip == 0u ? sourceImport.value() : bloomDownsampleImports[mip - 1u];
      auto sourceRead = ctx.graph.addTextureRead(addDownsample.value(),
                                                 downsampleSourceImport);
      if (sourceRead.hasError()) {
        return Result<bool, std::string>::makeError(sourceRead.error());
      }
      if (mip == 0u && nuri::isValid(exposureImport)) {
        auto exposureRead =
            ctx.graph.addTextureRead(addDownsample.value(), exposureImport);
        if (exposureRead.hasError()) {
          return Result<bool, std::string>::makeError(exposureRead.error());
        }
      }
    }

    for (uint32_t offset = 0u; offset < bloomMipCount_; ++offset) {
      const uint32_t mip = bloomMipCount_ - 1u - offset;
      const bool smallestMip = mip + 1u == bloomMipCount_;
      const TextureHandle upsampleTarget =
          bloomUpsampleTextures_[mip][ringIndex];
      const TextureHandle lowSource =
          smallestMip ? bloomDownsampleTextures_[mip][ringIndex]
                      : bloomUpsampleTextures_[mip + 1u][ringIndex];
      const TextureHandle highSource =
          smallestMip ? TextureHandle{}
                      : bloomDownsampleTextures_[mip][ringIndex];
      const uint32_t lowSourceTexId = gpu_.getTextureBindlessIndex(lowSource);
      const uint32_t highSourceTexId =
          nuri::isValid(highSource) ? gpu_.getTextureBindlessIndex(highSource)
                                    : kInvalidTextureBindlessIndex;
      if (lowSourceTexId == kInvalidTextureBindlessIndex ||
          (!smallestMip && highSourceTexId == kInvalidTextureBindlessIndex)) {
        return Result<bool, std::string>::makeError(
            "HDRBloomCompositePass::build: invalid bloom upsample bindless "
            "index");
      }

      auto targetImport =
          ctx.graph.importTexture(upsampleTarget, "hdr_bloom_upsample");
      if (targetImport.hasError()) {
        return Result<bool, std::string>::makeError(targetImport.error());
      }
      bloomUpsampleImports[mip] = targetImport.value();
      const HDRBloomPushConstants upsampleConstants{
          .sourceTexId = lowSourceTexId,
          .secondaryTexId = highSourceTexId,
          .exposureTexId = kInvalidTextureBindlessIndex,
          .sourceSamplerId = samplerId,
          .mode = smallestMip ? kHDRBloomModeCopy : kHDRBloomModeUpsample,
          .flags = 0u,
          .threshold = hdr.bloomThreshold,
          .softKnee = hdr.bloomSoftKnee,
          .scatter = kHDRBloomUpsampleScatter,
          .manualExposureEv = manualExposureEv,
          .adaptationTargetGray = hdr.adaptationTargetGray,
          .adaptationMinEv = hdr.adaptationMinEv,
          .adaptationMaxEv = hdr.adaptationMaxEv,
      };
      const DrawItem upsampleDraw = makeFullscreenDraw(
          bloomResources_.pipelineHandle,
          std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(&upsampleConstants),
              sizeof(upsampleConstants)),
          "HDRBloomUpsample");
      std::array<TextureHandle, 2> upsampleReads{lowSource, highSource};
      const size_t upsampleReadCount = smallestMip ? 1u : 2u;
      RenderGraphGraphicsPassDesc upsamplePass{};
      upsamplePass.color = {.loadOp = LoadOp::Clear,
                            .storeOp = StoreOp::Store,
                            .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
      upsamplePass.colorTexture = targetImport.value();
      upsamplePass.dependencyTextures = std::span<const TextureHandle>(
          upsampleReads.data(), upsampleReadCount);
      upsamplePass.draws = std::span<const DrawItem>(&upsampleDraw, 1u);
      upsamplePass.gpuTimingScope = GpuTimingScope::HDRPostProcess;
      upsamplePass.debugLabel = "HDR Bloom Upsample Pass";
      upsamplePass.debugColor = kHDRPassDebugColor;
      auto addUpsample = ctx.graph.addGraphicsPass(upsamplePass);
      if (addUpsample.hasError()) {
        return Result<bool, std::string>::makeError(addUpsample.error());
      }
      const RenderGraphTextureId lowSourceImport =
          smallestMip ? bloomDownsampleImports[mip]
                      : bloomUpsampleImports[mip + 1u];
      auto lowRead =
          ctx.graph.addTextureRead(addUpsample.value(), lowSourceImport);
      if (lowRead.hasError()) {
        return Result<bool, std::string>::makeError(lowRead.error());
      }
      if (!smallestMip) {
        auto highRead = ctx.graph.addTextureRead(addUpsample.value(),
                                                 bloomDownsampleImports[mip]);
        if (highRead.hasError()) {
          return Result<bool, std::string>::makeError(highRead.error());
        }
      }
    }
  }

  std::array<TextureHandle, 3> compositeReads{source, {}, {}};
  size_t compositeReadCount = 1u;
  if (nuri::isValid(exposureImport)) {
    compositeReads[compositeReadCount++] = exposure;
  }

  TextureHandle bloomCompositeTexture{};
  RenderGraphTextureId bloomCompositeImport{};
  uint32_t bloomTexId = kInvalidTextureBindlessIndex;
  if (bloomAvailable) {
    if (hdr.debugView == HDRPostProcessDebugView::BloomPrefilter) {
      bloomCompositeTexture = bloomDownsampleTextures_[0u][ringIndex];
      bloomCompositeImport = bloomDownsampleImports[0u];
    } else {
      bloomCompositeTexture = bloomUpsampleTextures_[0u][ringIndex];
      bloomCompositeImport = bloomUpsampleImports[0u];
    }
    bloomTexId = gpu_.getTextureBindlessIndex(bloomCompositeTexture);
    if (bloomTexId == kInvalidTextureBindlessIndex) {
      return Result<bool, std::string>::makeError(
          "HDRBloomCompositePass::build: invalid final bloom bindless index");
    }
    compositeReads[compositeReadCount++] = bloomCompositeTexture;
  }

  const float fallbackEv = 0.0f;
  const uint32_t flags =
      (hdr.bloomEnabled && bloomTexId != kInvalidTextureBindlessIndex
           ? kHDRPostFlagBloomEnabled
           : 0u) |
      (hdr.adaptationEnabled && exposureTexId != kInvalidTextureBindlessIndex
           ? kHDRPostFlagAdaptationEnabled
           : 0u);
  const HDRBloomCompositePushConstants compositeConstants{
      .sourceTexId = sourceTexId,
      .bloomTexId = bloomTexId,
      .exposureTexId = exposureTexId,
      .sourceSamplerId = samplerId,
      .flags = flags,
      .debugView = static_cast<uint32_t>(hdr.debugView),
      .reserved0 = 0u,
      .reserved1 = 0u,
      .bloomStrength = hdr.bloomStrength,
      .fallbackExposureEv = fallbackEv,
      .adaptationTargetGray = hdr.adaptationTargetGray,
      .adaptationMinEv = hdr.adaptationMinEv,
      .adaptationMaxEv = hdr.adaptationMaxEv,
  };

  const DrawItem compositeDraw = makeFullscreenDraw(
      resources_.pipelineHandle,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&compositeConstants),
          sizeof(compositeConstants)),
      "HDRBloomComposite");
  RenderGraphGraphicsPassDesc compositePass{};
  compositePass.color = {.loadOp = LoadOp::Clear,
                         .storeOp = StoreOp::Store,
                         .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  compositePass.colorTexture = outputImport.value();
  compositePass.dependencyTextures =
      std::span<const TextureHandle>(compositeReads.data(), compositeReadCount);
  compositePass.draws = std::span<const DrawItem>(&compositeDraw, 1u);
  compositePass.gpuTimingScope = GpuTimingScope::HDRPostProcess;
  compositePass.debugLabel = "HDR Bloom Composite Pass";
  compositePass.debugColor = kHDRPassDebugColor;
  auto addComposite = ctx.graph.addGraphicsPass(compositePass);
  if (addComposite.hasError()) {
    return Result<bool, std::string>::makeError(addComposite.error());
  }
  auto sourceRead =
      ctx.graph.addTextureRead(addComposite.value(), sourceImport.value());
  if (sourceRead.hasError()) {
    return Result<bool, std::string>::makeError(sourceRead.error());
  }
  if (nuri::isValid(exposureImport)) {
    auto exposureRead =
        ctx.graph.addTextureRead(addComposite.value(), exposureImport);
    if (exposureRead.hasError()) {
      return Result<bool, std::string>::makeError(exposureRead.error());
    }
  }
  if (nuri::isValid(bloomCompositeImport)) {
    auto bloomRead =
        ctx.graph.addTextureRead(addComposite.value(), bloomCompositeImport);
    if (bloomRead.hasError()) {
      return Result<bool, std::string>::makeError(bloomRead.error());
    }
  }

  HDRBloomCompositePushConstants copyConstants = compositeConstants;
  copyConstants.sourceTexId = outputTexId;
  copyConstants.bloomTexId = kInvalidTextureBindlessIndex;
  copyConstants.exposureTexId = kInvalidTextureBindlessIndex;
  copyConstants.flags = 0u;
  copyConstants.debugView = 0u;
  copyConstants.bloomStrength = 0.0f;
  copyConstants.fallbackExposureEv = 0.0f;
  const DrawItem copyDraw = makeFullscreenDraw(
      resources_.pipelineHandle,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&copyConstants),
          sizeof(copyConstants)),
      "HDRPostProcessCopyBack");
  RenderGraphGraphicsPassDesc copyPass{};
  copyPass.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  copyPass.colorTexture = sourceImport.value();
  copyPass.dependencyTextures = std::span<const TextureHandle>(&output, 1u);
  copyPass.draws = std::span<const DrawItem>(&copyDraw, 1u);
  copyPass.gpuTimingScope = GpuTimingScope::HDRPostProcess;
  copyPass.debugLabel = "HDR Postprocess Copy Back Pass";
  copyPass.debugColor = kHDRPassDebugColor;
  auto addCopy = ctx.graph.addGraphicsPass(copyPass);
  if (addCopy.hasError()) {
    return Result<bool, std::string>::makeError(addCopy.error());
  }
  auto outputRead =
      ctx.graph.addTextureRead(addCopy.value(), outputImport.value());
  if (outputRead.hasError()) {
    return Result<bool, std::string>::makeError(outputRead.error());
  }

  HDRPostProcessFrameMetrics &metrics = ctx.frame.metrics.hdrPostProcess;
  const TextureDimensions dimensions = gpu_.getTextureDimensions(source);
  metrics.width = dimensions.width;
  metrics.height = dimensions.height;
  metrics.bloomActive = hdr.bloomEnabled && bloomAvailable;
  metrics.adaptationActive =
      hdr.adaptationEnabled && exposureTexId != kInvalidTextureBindlessIndex;
  metrics.bloomMipCount = bloomAvailable ? bloomMipCount_ : 0u;
  metrics.bloomPassCount +=
      bloomAvailable ? (bloomMipCount_ * 2u + (hdr.bloomEnabled ? 1u : 0u))
                     : 0u;
  metrics.textureCount += 1u;
  metrics.textureBytes += textureStorageBytes(gpu_, output);
  if (bloomAvailable) {
    metrics.textureCount += bloomMipCount_ * 2u;
    for (uint32_t mip = 0u; mip < bloomMipCount_; ++mip) {
      metrics.textureBytes +=
          textureStorageBytes(gpu_, bloomDownsampleTextures_[mip][ringIndex]);
      metrics.textureBytes +=
          textureStorageBytes(gpu_, bloomUpsampleTextures_[mip][ringIndex]);
    }
  }
  metrics.effectiveExposureEv = 0.0f;
  metrics.adaptedExposureEv = 0.0f;
  if (hdr.adaptationEnabled && ctx.shared.exposureHistoryValid) {
    const std::optional<float> adaptedLuminance =
        readExposureLuminance(gpu_, ctx.shared.exposureReadTexture);
    if (adaptedLuminance.has_value()) {
      metrics.adaptedExposureEv =
          exposureEvFromLuminance(*adaptedLuminance, hdr);
      metrics.effectiveExposureEv =
          metrics.adaptedExposureEv +
          renderSettingsOrDefault(ctx.frame).toneMap.exposureEv;
      metrics.measuredLogAverageLuminance =
          std::log2(std::max(*adaptedLuminance, 1.0e-4f));
    }
  }
  const GpuTimingReport &timingReport = ctx.frame.gpuTiming;
  if (hasGpuTimingScope(timingReport, GpuTimingScope::HDRPostProcess)) {
    metrics.gpuTimeMs = timingReport.hdrPostProcessTimeMs;
    metrics.gpuTimingSourceFrameIndex =
        timingReport.hdrPostProcessSourceFrameIndex;
    metrics.gpuTimingAvailable = 1u;
  }
  ctx.shared.frameColorGraphTexture = sourceImport.value();
  ctx.frame.sharedResources.frameColorGraphTexture = sourceImport.value();
  return Result<bool, std::string>::makeResult(true);
}

PresentToneMapPass::PresentToneMapPass(GPUDevice &gpu,
                                       FrameCompositionFeatureConfig config)
    : FullscreenRenderPass(gpu) {
  const std::filesystem::path basePath = resolveShaderBasePath(config);
  resources_.vertexPath = config.fullscreenVertex.empty()
                              ? basePath / "fullscreen_copy.vert"
                              : config.fullscreenVertex;
  resources_.fragmentPath = config.presentFragment.empty()
                                ? basePath / "tonemap_present.frag"
                                : config.presentFragment;
  captureResources_.vertexPath = resources_.vertexPath;
  captureResources_.fragmentPath = resources_.fragmentPath;
  aces2SdrLut_.path = config.aces2SdrLut;
  agxLut_.path = config.agxLut;
}

PresentToneMapPass::~PresentToneMapPass() {
  destroyFullscreenPassResources(gpu_, captureResources_);
  destroyToneMapAssets();
}

bool PresentToneMapPass::isEnabled(const FrameBuildContext &ctx) const {
  return nuri::isValid(ctx.shared.frameColorTexture);
}

Result<bool, std::string> PresentToneMapPass::prepare(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto initResult =
      ensureInitialized("present_tonemap", "PresentToneMapPass::createShaders");
  if (initResult.hasError()) {
    return initResult;
  }
  auto assetsResult = ensureToneMapAssetsLoaded();
  if (assetsResult.hasError()) {
    return assetsResult;
  }
  auto pipelineResult =
      ensurePipeline(gpu_.getSwapchainFormat(), "present_tonemap",
                     "PresentToneMapPass::ensurePipeline");
  if (pipelineResult.hasError()) {
    return pipelineResult;
  }
  if (isRenderCaptureRequested(ctx.frame, "final_color") &&
      nuri::isValid(ctx.shared.presentCaptureTexture)) {
    auto captureInit = ensureFullscreenPassInitialized(
        gpu_, captureResources_, "present_tonemap_capture",
        "PresentToneMapPass::createCaptureShaders");
    if (captureInit.hasError()) {
      return captureInit;
    }
    return ensureFullscreenPassPipeline(
        gpu_, captureResources_, Format::RGBA8_UNORM, "present_tonemap_capture",
        "PresentToneMapPass::ensureCapturePipeline");
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> PresentToneMapPass::build(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const TextureHandle source = ctx.shared.frameColorTexture;
  const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(source);
  if (sourceTexId == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "PresentToneMapPass::build: invalid frame color bindless index");
  }

  const bool acesLutAvailable =
      aces2SdrLut_.texture != nullptr && aces2SdrLut_.texture->valid();
  const bool agxLutAvailable =
      agxLut_.texture != nullptr && agxLut_.texture->valid();
  const PresentToneMapState toneMapState =
      buildPresentToneMapState(renderSettingsOrDefault(ctx.frame).toneMap,
                               acesLutAvailable, agxLutAvailable);
  const Format swapchainFormat = gpu_.getSwapchainFormat();
  const bool manualSrgbEncode = swapchainFormat != Format::RGBA8_SRGB;
  const uint32_t flags = buildPresentFlags(toneMapState, manualSrgbEncode);

  uint32_t acesLutTexId = 0u;
  uint32_t agxLutTexId = 0u;
  if (acesLutAvailable) {
    auto texIdResult =
        resolveLutBindlessIndex(gpu_, aces2SdrLut_.texture->handle(), "ACES");
    if (texIdResult.hasError()) {
      return Result<bool, std::string>::makeError(texIdResult.error());
    }
    acesLutTexId = texIdResult.value();
  }
  if (agxLutAvailable) {
    auto texIdResult =
        resolveLutBindlessIndex(gpu_, agxLut_.texture->handle(), "AgX");
    if (texIdResult.hasError()) {
      return Result<bool, std::string>::makeError(texIdResult.error());
    }
    agxLutTexId = texIdResult.value();
  }
  const uint32_t lutSamplerId = gpu_.getSamplerBindlessIndex(lutSampler_);
  if ((acesLutAvailable || agxLutAvailable) &&
      lutSamplerId == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "PresentToneMapPass::build: invalid tone-map LUT sampler bindless "
        "index");
  }

  const PushConstants pushConstants{
      .sourceTexId = sourceTexId,
      .sourceSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u),
      .acesLutTexId = acesLutTexId,
      .agxLutTexId = agxLutTexId,
      .lutSamplerId = lutSamplerId,
      .flags = flags,
      .acesExposureScale =
          std::exp2(toneMapState.settings.exposureEv +
                    toneMapState.settings.acesExposureOffsetEv),
      .agxExposureScale = std::exp2(toneMapState.settings.exposureEv +
                                    toneMapState.settings.agxExposureOffsetEv),
      .compareSplit = toneMapState.settings.compareSplit,
      .shaperMinLog2 = kShaperMinLog2,
      .shaperInvRange = kShaperInvRange,
  };
  const DrawItem draw = makeFullscreenDraw(
      resources_.pipelineHandle,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pushConstants),
          sizeof(pushConstants)),
      "Present ToneMap");
  const DrawItem captureDraw = makeFullscreenDraw(
      captureResources_.pipelineHandle,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pushConstants),
          sizeof(pushConstants)),
      "Present ToneMap Capture");
  if (ctx.frame.frameIndex < kInitialDebugFrames) {
    NURI_LOG_DEBUG(
        "PresentToneMapPass::build: frame=%" PRIu64
        " sourceHandle=%u:%u sourceTexId=%u acesLutTexId=%u agxLutTexId=%u",
        ctx.frame.frameIndex, source.index, source.generation, sourceTexId,
        acesLutTexId, agxLutTexId);
  }

  auto sourceImportResult =
      ctx.graph.importTexture(source, "present_frame_color");
  if (sourceImportResult.hasError()) {
    return Result<bool, std::string>::makeError(sourceImportResult.error());
  }
  std::array<RenderGraphTextureId, 2> lutImports{};
  std::array<TextureHandle, 3> dependencies{source, {}, {}};
  size_t dependencyCount = 1u;
  if (acesLutAvailable) {
    auto lutImportResult = ctx.graph.importTexture(
        aces2SdrLut_.texture->handle(), "present_aces_lut");
    if (lutImportResult.hasError()) {
      return Result<bool, std::string>::makeError(lutImportResult.error());
    }
    lutImports[0] = lutImportResult.value();
    dependencies[dependencyCount++] = aces2SdrLut_.texture->handle();
  }
  if (agxLutAvailable) {
    auto lutImportResult =
        ctx.graph.importTexture(agxLut_.texture->handle(), "present_agx_lut");
    if (lutImportResult.hasError()) {
      return Result<bool, std::string>::makeError(lutImportResult.error());
    }
    lutImports[1] = lutImportResult.value();
    dependencies[dependencyCount++] = agxLut_.texture->handle();
  }

  const auto addToneMapReads =
      [&](RenderGraphPassId passId) -> Result<bool, std::string> {
    auto readResult =
        ctx.graph.addTextureRead(passId, sourceImportResult.value());
    if (readResult.hasError()) {
      return Result<bool, std::string>::makeError(readResult.error());
    }
    if (acesLutAvailable) {
      auto lutReadResult = ctx.graph.addTextureRead(passId, lutImports[0]);
      if (lutReadResult.hasError()) {
        return Result<bool, std::string>::makeError(lutReadResult.error());
      }
    }
    if (agxLutAvailable) {
      auto lutReadResult = ctx.graph.addTextureRead(passId, lutImports[1]);
      if (lutReadResult.hasError()) {
        return Result<bool, std::string>::makeError(lutReadResult.error());
      }
    }
    return Result<bool, std::string>::makeResult(true);
  };

  publishRequestedCapture(ctx.frame, gpu_, "frame_color_hdr", source,
                          RenderCaptureValueKind::LinearHdrColor,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "linear_hdr", "hdr_color", "PresentToneMapPass",
                          "frame_color_hdr");

  if (isRenderCaptureRequested(ctx.frame, "final_color") &&
      nuri::isValid(ctx.shared.presentCaptureTexture)) {
    if (!nuri::isValid(captureResources_.pipelineHandle)) {
      return Result<bool, std::string>::makeError(
          "PresentToneMapPass::build: invalid capture pipeline");
    }
    auto captureImport = ctx.graph.importTexture(
        ctx.shared.presentCaptureTexture, "present_capture_color");
    if (captureImport.hasError()) {
      return Result<bool, std::string>::makeError(captureImport.error());
    }
    RenderGraphGraphicsPassDesc captureDesc{};
    captureDesc.color = {.loadOp = LoadOp::Clear,
                         .storeOp = StoreOp::Store,
                         .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    captureDesc.colorTexture = captureImport.value();
    captureDesc.draws = std::span<const DrawItem>(&captureDraw, 1u);
    captureDesc.debugLabel = "Present ToneMap Capture Pass";
    captureDesc.debugColor = kPresentPassDebugColor;
    captureDesc.dependencyTextures =
        std::span<const TextureHandle>(dependencies.data(), dependencyCount);
    auto captureAdd = ctx.graph.addGraphicsPass(captureDesc);
    if (captureAdd.hasError()) {
      return Result<bool, std::string>::makeError(captureAdd.error());
    }
    auto captureReads = addToneMapReads(captureAdd.value());
    if (captureReads.hasError()) {
      return captureReads;
    }
    ctx.shared.presentCaptureGraphTexture = captureImport.value();
    ctx.frame.sharedResources.presentCaptureGraphTexture =
        captureImport.value();
    publishRequestedCapture(
        ctx.frame, gpu_, "final_color", ctx.shared.presentCaptureTexture,
        RenderCaptureValueKind::Color,
        RenderCaptureLifetimeClass::ToolCaptureTexture, "display_sdr",
        "ldr_color", "Present ToneMap Capture Pass", "final_color");
  }

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  passDesc.draws = std::span<const DrawItem>(&draw, 1u);
  passDesc.debugLabel = "Present ToneMap Pass";
  passDesc.debugColor = kPresentPassDebugColor;
  passDesc.markColorAsFrameOutput = true;
  passDesc.dependencyTextures =
      std::span<const TextureHandle>(dependencies.data(), dependencyCount);
  auto addResult = ctx.graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  auto presentReads = addToneMapReads(addResult.value());
  if (presentReads.hasError()) {
    return presentReads;
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> PresentToneMapPass::ensureToneMapAssetsLoaded() {
  auto samplerResult = ensureToneMapSampler();
  if (samplerResult.hasError()) {
    return samplerResult;
  }
  auto acesResult = ensureToneMapLutLoaded(aces2SdrLut_, "tonemap_aces2_sdr");
  if (acesResult.hasError()) {
    return acesResult;
  }
  auto agxResult = ensureToneMapLutLoaded(agxLut_, "tonemap_agx_sdr");
  if (agxResult.hasError()) {
    return agxResult;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> PresentToneMapPass::ensureToneMapSampler() {
  if (nuri::isValid(lutSampler_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto samplerResult = gpu_.createSampler(
      SamplerDesc{
          .minFilter = SamplerFilter::Linear,
          .magFilter = SamplerFilter::Linear,
          .mipMode = SamplerMipMode::Disabled,
          .wrapU = SamplerWrapMode::Clamp,
          .wrapV = SamplerWrapMode::Clamp,
          .wrapW = SamplerWrapMode::Clamp,
          .mipLodMin = 0.0f,
          .mipLodMax = 0.0f,
          .maxAnisotropy = 1u,
      },
      "present_tonemap_lut_sampler");
  if (samplerResult.hasError()) {
    return Result<bool, std::string>::makeError(samplerResult.error());
  }
  lutSampler_ = samplerResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
PresentToneMapPass::ensureToneMapLutLoaded(ToneMapLutResource &resource,
                                           std::string_view debugName) {
  if (resource.loadAttempted) {
    return Result<bool, std::string>::makeResult(true);
  }
  resource.loadAttempted = true;
  if (resource.path.empty()) {
    if (!resource.warned) {
      NURI_LOG_WARNING(
          "PresentToneMapPass: tone-map LUT '%s' has no configured path; "
          "falling back to legacy fit",
          std::string(debugName).c_str());
      resource.warned = true;
    }
    return Result<bool, std::string>::makeResult(true);
  }

  auto textureResult =
      Texture::loadTextureKtx2(gpu_, resource.path.string(), debugName);
  if (textureResult.hasError()) {
    if (!resource.warned) {
      NURI_LOG_WARNING("PresentToneMapPass: failed to load LUT '%s' from '%s': "
                       "%s. Falling back to legacy fit.",
                       std::string(debugName).c_str(),
                       resource.path.string().c_str(),
                       textureResult.error().c_str());
      resource.warned = true;
    }
    return Result<bool, std::string>::makeResult(true);
  }

  resource.texture = std::move(textureResult.value());
  return Result<bool, std::string>::makeResult(true);
}

void PresentToneMapPass::destroyToneMapAssets() {
  aces2SdrLut_.texture.reset();
  agxLut_.texture.reset();
  if (nuri::isValid(lutSampler_)) {
    gpu_.destroySampler(lutSampler_);
    lutSampler_ = {};
  }
}

FrameCompositionFeature::FrameCompositionFeature(
    GPUDevice &gpu, FrameCompositionFeatureConfig config)
    : downsamplePass_(gpu, config), resolvePass_(gpu, std::move(config)) {}

std::span<RenderFeaturePass *const> FrameCompositionFeature::passes() noexcept {
  return passes_;
}

HDRPostProcessFeature::HDRPostProcessFeature(
    GPUDevice &gpu, FrameCompositionFeatureConfig config)
    : exposurePass_(gpu, config), compositePass_(gpu, std::move(config)) {}

Result<bool, std::string>
HDRPostProcessFeature::publishFrameData(FrameBuildContext &ctx) {
  RenderSettings::HDRPostProcessSettings hdr =
      renderSettingsOrDefault(ctx.frame).hdrPostProcess;
  sanitizeHDRPostProcessSettings(hdr);
  if (hdr.adaptationEnabled) {
    ctx.shared.textureRequirements |= FrameTextureRequirementFlags::Exposure;
  }
  return Result<bool, std::string>::makeResult(true);
}

std::span<RenderFeaturePass *const> HDRPostProcessFeature::passes() noexcept {
  return passes_;
}

FramePresentFeature::FramePresentFeature(GPUDevice &gpu,
                                         FrameCompositionFeatureConfig config)
    : presentPass_(gpu, std::move(config)) {}

std::span<RenderFeaturePass *const> FramePresentFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
