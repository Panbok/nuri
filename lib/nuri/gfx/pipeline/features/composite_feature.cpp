#include "nuri/gfx/pipeline/features/composite_feature.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/render_capture.h"
#include "nuri/gfx/fullscreen.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/pch.h"
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
constexpr uint32_t kDownsamplePassDebugColor = 0xff33aa88u;
constexpr uint32_t kResolvePassDebugColor = 0xff33cc88u;
constexpr uint32_t kPresentPassDebugColor = 0xff55cc88u;
constexpr uint32_t kHDRPassDebugColor = 0xff66dd99u;
constexpr uint32_t kDrawDebugColor = 0xff2299ddu;
constexpr uint32_t kHDRPostFlagBloomEnabled = 1u << 0u;
constexpr uint32_t kHDRPostFlagAdaptationEnabled = 1u << 1u;
constexpr uint32_t kHDRPostFlagExposureHistoryValid = 1u << 2u;
constexpr uint32_t kHDRPostFlagReducedLuminanceSource = 1u << 3u;
constexpr uint32_t kHDRBloomModePrefilterDownsample = 0u;
constexpr uint32_t kHDRBloomModeDownsample = 1u;
constexpr uint32_t kHDRBloomModeUpsample = 2u;
constexpr uint32_t kHDRBloomModeCopy = 3u;
constexpr float kHDRBloomUpsampleScatter = 0.65f;
constexpr std::array<std::string_view, 2> kToneMapLutNames{"tonemap_aces2_sdr",
                                                           "tonemap_agx_sdr"};
constexpr std::array<std::string_view, 2> kToneMapLutImportNames{
    "present_aces_lut", "present_agx_lut"};
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
};
static_assert(sizeof(HDRExposurePushConstants) <= 128u);
struct HDRLuminanceReducePushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t mode = 0u;
  float texelSizeX = 1.0f;
  float texelSizeY = 1.0f;
};
static_assert(sizeof(HDRLuminanceReducePushConstants) <= 128u);
struct HDRBloomCompositePushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t bloomTexId = kInvalidTextureBindlessIndex;
  uint32_t exposureTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t flags = 0u;
  uint32_t debugView = 0u;
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
}
template <typename Range>
void destroyTextureHandles(GPUDevice &gpu, Range &values) {
  if constexpr (std::is_same_v<std::remove_cvref_t<Range>, TextureHandle>) {
    if (nuri::isValid(values))
      gpu.destroyTexture(values);
  } else {
    for (auto &value : values)
      destroyTextureHandles(gpu, value);
    values.clear();
  }
}
template <typename Step, typename... Rest>
Result<bool, std::string> runSteps(Step &&step, Rest &&...rest) {
  auto result = step();
  if constexpr (sizeof...(Rest) == 0u) {
    return result;
  } else {
    return result.hasError() ? result : runSteps(std::forward<Rest>(rest)...);
  }
}
Result<bool, std::string>
createFullscreenPassShaders(GPUDevice &gpu, FullscreenPassResources &resources,
                            std::string_view shaderName) {
  std::unique_ptr<Shader> shader = Shader::create(shaderName, gpu);
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
  resources.vertexShader = vertexShader;
  resources.fragmentShader = fragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}
Result<bool, std::string>
ensureFullscreenPassInitialized(GPUDevice &gpu,
                                FullscreenPassResources &resources,
                                std::string_view shaderName) {
  if (nuri::isValid(resources.vertexShader) &&
      nuri::isValid(resources.fragmentShader)) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaderResult = createFullscreenPassShaders(gpu, resources, shaderName);
  if (shaderResult.hasError()) {
    return shaderResult;
  }
  return Result<bool, std::string>::makeResult(true);
}
Result<bool, std::string>
ensureFullscreenPassPipeline(GPUDevice &gpu, FullscreenPassResources &resources,
                             Format colorFormat, std::string_view debugName) {
  if (nuri::isValid(resources.pipelineHandle) &&
      resources.pipelineColorFormat == colorFormat) {
    return Result<bool, std::string>::makeResult(true);
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
void addFullscreenTexturePass(
    RenderGraphBuilder &graph, RenderGraphTextureId colorTexture,
    std::span<const DrawItem> draws,
    std::span<const TextureHandle> textureReads, std::string_view debugLabel,
    uint32_t debugColor, bool markColorAsFrameOutput = false,
    GpuTimingScope gpuTimingScope = GpuTimingScope::None,
    std::span<const RenderGraphTextureId> graphReads = {}) {
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
  const RenderGraphPassId pass = graph.addGraphicsPass(passDesc).value();
  for (RenderGraphTextureId read : graphReads) {
    (void)graph.addTextureRead(pass, read).value();
  }
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
TextureDimensions mipDimensions(uint32_t width, uint32_t height,
                                uint32_t mipIndex, bool ceil) {
  uint32_t mipWidth = std::max(width, 1u);
  uint32_t mipHeight = std::max(height, 1u);
  for (uint32_t i = 0u; i <= mipIndex; ++i) {
    mipWidth = std::max((mipWidth + static_cast<uint32_t>(ceil)) / 2u, 1u);
    mipHeight = std::max((mipHeight + static_cast<uint32_t>(ceil)) / 2u, 1u);
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
         static_cast<uint64_t>(formatTexelBytes(gpu.getTextureFormat(texture)));
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
  return (manualSrgbEncode ? kPresentFlagManualSrgbEncode : 0u) |
         (state.primaryUseAgx ? kPresentFlagPrimaryUseAgx : 0u) |
         (state.compareEnabled ? kPresentFlagCompareEnabled : 0u) |
         (state.grayCardDebug ? kPresentFlagGrayCardDebug : 0u) |
         (state.acesLutAvailable ? kPresentFlagAcesLutAvailable : 0u) |
         (state.agxLutAvailable ? kPresentFlagAgxLutAvailable : 0u) |
         (!state.primaryHasLut ? kPresentFlagPrimaryLegacyFallback : 0u) |
         (state.compareEnabled && !state.compareHasLut
              ? kPresentFlagCompareLegacyFallback
              : 0u);
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
} // namespace

FullscreenRenderPass::FullscreenRenderPass(GPUDevice &gpu) : gpu_(gpu) {}

FullscreenRenderPass::~FullscreenRenderPass() {
  destroyFullscreenPassResources(gpu_, resources_);
}

Result<bool, std::string>
FullscreenRenderPass::ensureInitialized(std::string_view shaderName) {
  return ensureFullscreenPassInitialized(gpu_, resources_, shaderName);
}

Result<bool, std::string>
FullscreenRenderPass::ensurePipeline(Format colorFormat,
                                     std::string_view debugName) {
  return ensureFullscreenPassPipeline(gpu_, resources_, colorFormat, debugName);
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
SceneColorDownsamplePass::prepare(FrameBuildContext &) {
  return runSteps([&] { return ensureInitialized("scene_color_downsample"); },
                  [&] {
                    return ensurePipeline(kFrameCompositionSceneColorFormat,
                                          "scene_color_downsample");
                  });
}

Result<bool, std::string>
SceneColorDownsamplePass::build(FrameBuildContext &ctx) {
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
    const std::string_view debugLabel = mipLevel == 1u
                                            ? "Scene Color Downsample Half"
                                            : "Scene Color Downsample Quarter";
    const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(source);
    const CopyPushConstants pushConstants{
        .sourceTexId = sourceTexId,
        .sourceSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u),
        .flags = kSceneCopyFlagDownsample,
    };
    const DrawItem draw = makeFullscreenDraw(
        resources_.pipelineHandle,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&pushConstants),
            sizeof(pushConstants)),
        debugLabel, kDrawDebugColor);
    const RenderGraphTextureId color =
        ctx.graph.importTexture(destination, "scene_color_mip").value();
    const std::span<const TextureHandle> textureReads(&source, 1u);
    addFullscreenTexturePass(ctx.graph, color,
                             std::span<const DrawItem>(&draw, 1u), textureReads,
                             debugLabel, kDownsamplePassDebugColor, false,
                             GpuTimingScope::SceneColorDownsample);
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

Result<bool, std::string> SceneResolvePass::prepare(FrameBuildContext &) {
  return runSteps([&] { return ensureInitialized("scene_resolve"); },
                  [&] {
                    return ensurePipeline(kFrameCompositionFrameColorFormat,
                                          "scene_resolve");
                  });
}

Result<bool, std::string> SceneResolvePass::build(FrameBuildContext &ctx) {
  const SceneResolveSource source = resolveSceneResolveSource(ctx);
  const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(source.texture);
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
  };
  const DrawItem draw = makeFullscreenDraw(
      resources_.pipelineHandle,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pushConstants),
          sizeof(pushConstants)),
      "Scene Resolve", kDrawDebugColor);
  const RenderGraphTextureId color =
      ctx.graph.importTexture(ctx.shared.frameColorTexture, "frame_color")
          .value();
  const std::span<const TextureHandle> textureReads(&source.texture, 1u);
  addFullscreenTexturePass(ctx.graph, color,
                           std::span<const DrawItem>(&draw, 1u), textureReads,
                           "Scene Resolve Pass", kResolvePassDebugColor);
  ctx.shared.frameColorGraphTexture = color;
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
  destroyTextureHandles(gpu_, luminanceTextures_);
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
        mipDimensions(width, height, mip, true);
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
  return runSteps(
      [&] { return ensureInitialized("hdr_exposure_adapt"); },
      [&] {
        return ensureFullscreenPassInitialized(gpu_, luminanceResources_,
                                               "hdr_luminance_reduce");
      },
      [&] {
        return ensurePipeline(kFrameCompositionExposureFormat,
                              "hdr_exposure_adapt");
      },
      [&] {
        return ensureFullscreenPassPipeline(gpu_, luminanceResources_,
                                            kFrameCompositionExposureFormat,
                                            "hdr_luminance_reduce");
      },
      [&] { return ensureLuminanceTextures(); });
}

Result<bool, std::string> HDRExposureAdaptPass::build(FrameBuildContext &ctx) {
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
  const RenderGraphTextureId sourceImport =
      nuri::isValid(ctx.shared.frameColorGraphTexture)
          ? ctx.shared.frameColorGraphTexture
          : ctx.graph.importTexture(source, "hdr_source_color").value();
  const uint32_t ringIndex =
      luminanceTextures_.empty()
          ? 0u
          : static_cast<uint32_t>(ctx.frame.frameIndex %
                                  luminanceTextures_[0u].size());
  const uint32_t samplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u);
  TextureHandle meteringSource = source;
  RenderGraphTextureId meteringSourceImport = sourceImport;
  uint32_t sourceTexId = frameColorTexId;
  bool reducedLuminanceSource = false;
  if (!luminanceTextures_.empty()) {
    RenderGraphTextureId previousImport = sourceImport;
    TextureHandle previousTexture = source;
    for (uint32_t mip = 0u; mip < luminanceMipCount_; ++mip) {
      const TextureHandle target = luminanceTextures_[mip][ringIndex];
      const uint32_t reduceSourceTexId =
          gpu_.getTextureBindlessIndex(previousTexture);
      const RenderGraphTextureId targetImport =
          ctx.graph.importTexture(target, "hdr_luminance_reduce").value();
      const TextureDimensions sourceDimensions =
          gpu_.getTextureDimensions(previousTexture);
      const HDRLuminanceReducePushConstants reduceConstants{
          .sourceTexId = reduceSourceTexId,
          .sourceSamplerId = samplerId,
          .mode = mip == 0u ? 0u : 1u,
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
          "HDRLuminanceReduce", kDrawDebugColor);
      addFullscreenTexturePass(
          ctx.graph, targetImport, std::span<const DrawItem>(&reduceDraw, 1u),
          std::span<const TextureHandle>(&previousTexture, 1u),
          "HDR Luminance Reduce Pass", kHDRPassDebugColor, false,
          GpuTimingScope::HDRPostProcess,
          std::span<const RenderGraphTextureId>(&previousImport, 1u));
      previousTexture = target;
      previousImport = targetImport;
    }
    meteringSource = previousTexture;
    meteringSourceImport = previousImport;
    sourceTexId = gpu_.getTextureBindlessIndex(meteringSource);
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
  const RenderGraphTextureId exposureImport =
      ctx.graph.importTexture(exposureWrite, "hdr_exposure_write").value();
  std::array<TextureHandle, 2> reads{meteringSource, {}};
  size_t readCount = 1u;
  RenderGraphTextureId previousExposureImport{};
  if (previousExposureTexId != kInvalidTextureBindlessIndex) {
    previousExposureImport =
        ctx.graph.importTexture(exposureRead, "hdr_exposure_read").value();
    reads[readCount++] = exposureRead;
  }
  const DrawItem draw = makeFullscreenDraw(
      resources_.pipelineHandle,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pushConstants),
          sizeof(pushConstants)),
      "HDRExposureAdapt", kDrawDebugColor);
  std::array<RenderGraphTextureId, 2> graphReads{meteringSourceImport,
                                                 previousExposureImport};
  size_t graphReadCount = nuri::isValid(previousExposureImport) ? 2u : 1u;
  addFullscreenTexturePass(
      ctx.graph, exposureImport, std::span<const DrawItem>(&draw, 1u),
      std::span<const TextureHandle>(reads.data(), readCount),
      "HDR Exposure Adapt Pass", kHDRPassDebugColor, false,
      GpuTimingScope::HDRPostProcess,
      std::span<const RenderGraphTextureId>(graphReads.data(), graphReadCount));
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
  destroyTextureHandles(gpu_, outputTextures_);
  outputWidth_ = 0u;
  outputHeight_ = 0u;
  outputRingCount_ = 0u;
}

void HDRBloomCompositePass::destroyBloomTextures() {
  destroyTextureHandles(gpu_, bloomDownsampleTextures_);
  destroyTextureHandles(gpu_, bloomUpsampleTextures_);
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
    const TextureDimensions dimensions =
        mipDimensions(width, height, mip, false);
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
  const bool needsBloomChain = shouldBuildBloomChain(hdr);
  const auto success = [] {
    return Result<bool, std::string>::makeResult(true);
  };
  return runSteps(
      [&] { return ensureInitialized("hdr_bloom_composite"); },
      [&] {
        return needsBloomChain ? ensureFullscreenPassInitialized(
                                     gpu_, bloomResources_, "hdr_bloom")
                               : success();
      },
      [&] {
        return ensurePipeline(kFrameCompositionFrameColorFormat,
                              "hdr_bloom_composite");
      },
      [&] {
        return needsBloomChain
                   ? ensureFullscreenPassPipeline(
                         gpu_, bloomResources_,
                         kFrameCompositionFrameColorFormat, "hdr_bloom")
                   : success();
      },
      [&] { return ensureOutputTextures(); },
      [&] {
        return ensureBloomTextures(needsBloomChain ? hdr.bloomMaxMipCount : 0u);
      });
}

Result<bool, std::string> HDRBloomCompositePass::build(FrameBuildContext &ctx) {
  RenderSettings::HDRPostProcessSettings hdr =
      renderSettingsOrDefault(ctx.frame).hdrPostProcess;
  sanitizeHDRPostProcessSettings(hdr);
  const bool needsBloomChain = shouldBuildBloomChain(hdr);
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
  const RenderGraphTextureId sourceImport =
      nuri::isValid(ctx.shared.frameColorGraphTexture)
          ? ctx.shared.frameColorGraphTexture
          : ctx.graph.importTexture(source, "hdr_frame_color").value();
  const RenderGraphTextureId outputImport =
      ctx.graph.importTexture(output, "hdr_postprocess_output").value();
  RenderGraphTextureId exposureImport{};
  if (hdr.adaptationEnabled && exposureTexId != kInvalidTextureBindlessIndex) {
    exposureImport = ctx.graph.importTexture(exposure, "hdr_exposure").value();
  }
  const uint32_t samplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u);
  const float manualExposureEv =
      renderSettingsOrDefault(ctx.frame).toneMap.exposureEv;
  const bool bloomAvailable = needsBloomChain && bloomMipCount_ != 0u;
  std::array<RenderGraphTextureId, kMaxSceneDepthPyramidLevels>
      bloomDownsampleImports{};
  std::array<RenderGraphTextureId, kMaxSceneDepthPyramidLevels>
      bloomUpsampleImports{};
  if (bloomAvailable) {
    for (uint32_t mip = 0u; mip < bloomMipCount_; ++mip) {
      const TextureHandle downsampleTarget =
          bloomDownsampleTextures_[mip][ringIndex];
      const TextureHandle downsampleSource =
          mip == 0u ? source : bloomDownsampleTextures_[mip - 1u][ringIndex];
      const uint32_t downsampleSourceTexId =
          gpu_.getTextureBindlessIndex(downsampleSource);
      bloomDownsampleImports[mip] =
          ctx.graph.importTexture(downsampleTarget, "hdr_bloom_downsample")
              .value();
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
          "HDRBloomDownsample", kDrawDebugColor);
      std::array<TextureHandle, 2> downsampleReads{downsampleSource, exposure};
      const size_t downsampleReadCount =
          mip == 0u && exposureTexId != kInvalidTextureBindlessIndex ? 2u : 1u;
      const RenderGraphTextureId downsampleSourceImport =
          mip == 0u ? sourceImport : bloomDownsampleImports[mip - 1u];
      std::array<RenderGraphTextureId, 2> downsampleGraphReads{
          downsampleSourceImport, exposureImport};
      addFullscreenTexturePass(
          ctx.graph, bloomDownsampleImports[mip],
          std::span<const DrawItem>(&downsampleDraw, 1u),
          std::span<const TextureHandle>(downsampleReads.data(),
                                         downsampleReadCount),
          "HDR Bloom Downsample Pass", kHDRPassDebugColor, false,
          GpuTimingScope::HDRPostProcess,
          std::span<const RenderGraphTextureId>(downsampleGraphReads.data(),
                                                downsampleReadCount));
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
      bloomUpsampleImports[mip] =
          ctx.graph.importTexture(upsampleTarget, "hdr_bloom_upsample").value();
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
          "HDRBloomUpsample", kDrawDebugColor);
      std::array<TextureHandle, 2> upsampleReads{lowSource, highSource};
      const size_t upsampleReadCount = smallestMip ? 1u : 2u;
      const RenderGraphTextureId lowSourceImport =
          smallestMip ? bloomDownsampleImports[mip]
                      : bloomUpsampleImports[mip + 1u];
      std::array<RenderGraphTextureId, 2> upsampleGraphReads{
          lowSourceImport, bloomDownsampleImports[mip]};
      addFullscreenTexturePass(
          ctx.graph, bloomUpsampleImports[mip],
          std::span<const DrawItem>(&upsampleDraw, 1u),
          std::span<const TextureHandle>(upsampleReads.data(),
                                         upsampleReadCount),
          "HDR Bloom Upsample Pass", kHDRPassDebugColor, false,
          GpuTimingScope::HDRPostProcess,
          std::span<const RenderGraphTextureId>(upsampleGraphReads.data(),
                                                upsampleReadCount));
    }
  }
  std::array<TextureHandle, 3> compositeReads{source, {}, {}};
  std::array<RenderGraphTextureId, 3> compositeGraphReads{sourceImport, {}, {}};
  size_t compositeReadCount = 1u;
  if (nuri::isValid(exposureImport)) {
    compositeReads[compositeReadCount++] = exposure;
    compositeGraphReads[1] = exposureImport;
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
    compositeReads[compositeReadCount] = bloomCompositeTexture;
    compositeGraphReads[compositeReadCount++] = bloomCompositeImport;
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
      "HDRBloomComposite", kDrawDebugColor);
  addFullscreenTexturePass(
      ctx.graph, outputImport, std::span<const DrawItem>(&compositeDraw, 1u),
      std::span<const TextureHandle>(compositeReads.data(), compositeReadCount),
      "HDR Bloom Composite Pass", kHDRPassDebugColor, false,
      GpuTimingScope::HDRPostProcess,
      std::span<const RenderGraphTextureId>(compositeGraphReads.data(),
                                            compositeReadCount));
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
      "HDRPostProcessCopyBack", kDrawDebugColor);
  addFullscreenTexturePass(
      ctx.graph, sourceImport, std::span<const DrawItem>(&copyDraw, 1u),
      std::span<const TextureHandle>(&output, 1u),
      "HDR Postprocess Copy Back Pass", kHDRPassDebugColor, false,
      GpuTimingScope::HDRPostProcess,
      std::span<const RenderGraphTextureId>(&outputImport, 1u));
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
    }
  }
  const GpuTimingReport &timingReport = ctx.frame.gpuTiming;
  if (hasGpuTimingScope(timingReport, GpuTimingScope::HDRPostProcess)) {
    metrics.gpuTimeMs = timingReport.hdrPostProcessTimeMs;
    metrics.gpuTimingSourceFrameIndex =
        timingReport.hdrPostProcessSourceFrameIndex;
    metrics.gpuTimingAvailable = 1u;
  }
  ctx.shared.frameColorGraphTexture = sourceImport;
  ctx.frame.sharedResources.frameColorGraphTexture = sourceImport;
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
  toneMapLuts_[0].path = config.aces2SdrLut;
  toneMapLuts_[1].path = config.agxLut;
}

PresentToneMapPass::~PresentToneMapPass() {
  destroyFullscreenPassResources(gpu_, captureResources_);
  if (nuri::isValid(lutSampler_))
    gpu_.destroySampler(lutSampler_);
}

bool PresentToneMapPass::isEnabled(const FrameBuildContext &ctx) const {
  return nuri::isValid(ctx.shared.frameColorTexture);
}

Result<bool, std::string> PresentToneMapPass::prepare(FrameBuildContext &ctx) {
  auto initResult = ensureInitialized("present_tonemap");
  if (initResult.hasError()) {
    return initResult;
  }
  auto samplerResult = ensureToneMapSampler();
  if (samplerResult.hasError())
    return samplerResult;
  for (size_t i = 0; i < toneMapLuts_.size(); ++i)
    ensureToneMapLutLoaded(toneMapLuts_[i], kToneMapLutNames[i]);
  auto pipelineResult =
      ensurePipeline(gpu_.getSwapchainFormat(), "present_tonemap");
  if (pipelineResult.hasError()) {
    return pipelineResult;
  }
  if (isRenderCaptureRequested(ctx.frame, "final_color") &&
      nuri::isValid(ctx.shared.presentCaptureTexture)) {
    auto captureInit = ensureFullscreenPassInitialized(
        gpu_, captureResources_, "present_tonemap_capture");
    if (captureInit.hasError()) {
      return captureInit;
    }
    return ensureFullscreenPassPipeline(gpu_, captureResources_,
                                        Format::RGBA8_UNORM,
                                        "present_tonemap_capture");
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> PresentToneMapPass::build(FrameBuildContext &ctx) {
  const TextureHandle source = ctx.shared.frameColorTexture;
  const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(source);
  const auto available = [](const ToneMapLutResource &lut) {
    return lut.texture && lut.texture->valid();
  };
  const bool acesLutAvailable = available(toneMapLuts_[0]);
  const bool agxLutAvailable = available(toneMapLuts_[1]);
  const PresentToneMapState toneMapState =
      buildPresentToneMapState(renderSettingsOrDefault(ctx.frame).toneMap,
                               acesLutAvailable, agxLutAvailable);
  const Format swapchainFormat = gpu_.getSwapchainFormat();
  const bool manualSrgbEncode = swapchainFormat != Format::RGBA8_SRGB;
  const uint32_t flags = buildPresentFlags(toneMapState, manualSrgbEncode);
  std::array<uint32_t, 2> lutTexIds{};
  for (size_t i = 0; i < toneMapLuts_.size(); ++i)
    if (available(toneMapLuts_[i]))
      lutTexIds[i] =
          gpu_.getTextureBindlessIndex(toneMapLuts_[i].texture->handle());
  const uint32_t lutSamplerId = gpu_.getSamplerBindlessIndex(lutSampler_);
  const PushConstants pushConstants{
      .sourceTexId = sourceTexId,
      .sourceSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u),
      .acesLutTexId = lutTexIds[0],
      .agxLutTexId = lutTexIds[1],
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
      "Present ToneMap", kDrawDebugColor);
  const DrawItem captureDraw = makeFullscreenDraw(
      captureResources_.pipelineHandle,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pushConstants),
          sizeof(pushConstants)),
      "Present ToneMap Capture", kDrawDebugColor);
  const RenderGraphTextureId sourceImport =
      ctx.graph.importTexture(source, "present_frame_color").value();
  std::array<RenderGraphTextureId, 2> lutImports{};
  std::array<TextureHandle, 3> dependencies{source, {}, {}};
  size_t dependencyCount = 1u;
  for (size_t i = 0; i < toneMapLuts_.size(); ++i) {
    if (!available(toneMapLuts_[i]))
      continue;
    const TextureHandle texture = toneMapLuts_[i].texture->handle();
    lutImports[i] =
        ctx.graph.importTexture(texture, kToneMapLutImportNames[i]).value();
    dependencies[dependencyCount++] = texture;
  }
  const auto addToneMapReads = [&](RenderGraphPassId passId) {
    (void)ctx.graph.addTextureRead(passId, sourceImport).value();
    for (RenderGraphTextureId lut : lutImports)
      if (nuri::isValid(lut))
        (void)ctx.graph.addTextureRead(passId, lut).value();
  };
  publishRequestedCapture(ctx.frame, gpu_, "frame_color_hdr", source,
                          RenderCaptureValueKind::LinearHdrColor,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "linear_hdr", "hdr_color", "PresentToneMapPass",
                          "frame_color_hdr");
  if (renderSettingsOrDefault(ctx.frame).ddgi.debugView !=
      DDGIDebugView::None) {
    publishRequestedCapture(ctx.frame, gpu_, "ddgi_debug_preview", source,
                            RenderCaptureValueKind::DebugPreview,
                            RenderCaptureLifetimeClass::FrameSharedRingTexture,
                            "linear_hdr", "debug_preview", "PresentToneMapPass",
                            "ddgi_debug_preview");
    const DDGIDebugView ddgiView =
        renderSettingsOrDefault(ctx.frame).ddgi.debugView;
    std::string_view semanticCapture{};
    if (ddgiView == DDGIDebugView::VolumeId ||
        ddgiView == DDGIDebugView::ProbeWeights ||
        ddgiView == DDGIDebugView::Confidence) {
      semanticCapture = "ddgi_coverage_debug_preview";
    } else if (ddgiView == DDGIDebugView::Classification) {
      semanticCapture = "ddgi_classification_debug_preview";
    } else if (ddgiView == DDGIDebugView::UpdateAge) {
      // Update age is the production dirty-response visualization: affected
      // probes return to a low submitted age as their localized work commits.
      semanticCapture = "ddgi_dirty_region_debug_preview";
    }
    if (!semanticCapture.empty() &&
        isRenderCaptureRequested(ctx.frame, semanticCapture)) {
      DDGICaptureMetadata metadata{};
      if (ctx.shared.ddgiFrameGpuData.has_value()) {
        metadata = ctx.shared.ddgiFrameGpuData->captureMetadata[0u];
      }
      ctx.frame.captureRegistry.publish(RenderCapturePoint{
          .name = semanticCapture,
          .version = kDDGICaptureSemanticsVersion,
          .texture = source,
          .format = gpu_.getTextureFormat(source),
          .dimensions = gpu_.getTextureDimensions(source),
          .frameIndex = ctx.frame.frameIndex,
          .kind = RenderCaptureValueKind::DebugPreview,
          .lifetime = RenderCaptureLifetimeClass::FrameSharedRingTexture,
          .colorSpace = "linear_hdr",
          .defaultCompareProfile = "debug_preview",
          .producerPassLabel = "PresentToneMapPass",
          .debugLabel = semanticCapture,
          .ddgiMetadata = metadata,
      });
    }
  }
  if (isRenderCaptureRequested(ctx.frame, "final_color") &&
      nuri::isValid(ctx.shared.presentCaptureTexture)) {
    const RenderGraphTextureId capture =
        ctx.graph
            .importTexture(ctx.shared.presentCaptureTexture,
                           "present_capture_color")
            .value();
    RenderGraphGraphicsPassDesc captureDesc{};
    captureDesc.color = {.loadOp = LoadOp::Clear,
                         .storeOp = StoreOp::Store,
                         .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    captureDesc.colorTexture = capture;
    captureDesc.draws = std::span<const DrawItem>(&captureDraw, 1u);
    captureDesc.debugLabel = "Present ToneMap Capture Pass";
    captureDesc.debugColor = kPresentPassDebugColor;
    captureDesc.dependencyTextures =
        std::span<const TextureHandle>(dependencies.data(), dependencyCount);
    addToneMapReads(ctx.graph.addGraphicsPass(captureDesc).value());
    ctx.shared.presentCaptureGraphTexture = capture;
    ctx.frame.sharedResources.presentCaptureGraphTexture = capture;
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
  addToneMapReads(ctx.graph.addGraphicsPass(passDesc).value());
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

void PresentToneMapPass::ensureToneMapLutLoaded(ToneMapLutResource &resource,
                                                std::string_view debugName) {
  if (resource.loadAttempted) {
    return;
  }
  resource.loadAttempted = true;
  if (resource.path.empty()) {
    NURI_LOG_WARNING(
        "PresentToneMapPass: tone-map LUT '%s' has no configured path; "
        "falling back to legacy fit",
        std::string(debugName).c_str());
    return;
  }
  auto textureResult =
      Texture::loadTextureKtx2(gpu_, resource.path.string(), debugName);
  if (textureResult.hasError()) {
    NURI_LOG_WARNING("PresentToneMapPass: failed to load LUT '%s' from '%s': "
                     "%s. Falling back to legacy fit.",
                     std::string(debugName).c_str(),
                     resource.path.string().c_str(),
                     textureResult.error().c_str());
    return;
  }
  resource.texture = std::move(textureResult.value());
}

Result<bool, std::string>
HDRExposureAdaptPass::publishFrameData(FrameBuildContext &ctx) {
  RenderSettings::HDRPostProcessSettings hdr =
      renderSettingsOrDefault(ctx.frame).hdrPostProcess;
  sanitizeHDRPostProcessSettings(hdr);
  if (hdr.adaptationEnabled) {
    ctx.shared.textureRequirements |= FrameTextureRequirementFlags::Exposure;
  }
  return Result<bool, std::string>::makeResult(true);
}

void registerFrameCompositionStages(RenderPipeline &pipeline, GPUDevice &gpu,
                                    FrameCompositionFeatureConfig config) {
  pipeline.addStage(std::make_unique<SceneColorDownsamplePass>(gpu, config),
                    "FrameCompositionFeature", "SceneColorDownsamplePass");
  pipeline.addStage(std::make_unique<SceneResolvePass>(gpu, std::move(config)),
                    "FrameCompositionFeature", "SceneResolvePass");
}

void registerHDRPostProcessStages(RenderPipeline &pipeline, GPUDevice &gpu,
                                  FrameCompositionFeatureConfig config) {
  pipeline.addStage(
      std::make_unique<HDRExposureAdaptPass>(gpu, config),
      "HDRPostProcessFeature", "HDRExposureAdaptPass", false,
      PipelineComponentDesc{.publish = [](void *state, FrameBuildContext &ctx) {
        return static_cast<HDRExposureAdaptPass *>(state)->publishFrameData(
            ctx);
      }});
  pipeline.addStage(
      std::make_unique<HDRBloomCompositePass>(gpu, std::move(config)),
      "HDRPostProcessFeature", "HDRBloomCompositePass");
}

void registerFramePresentStage(RenderPipeline &pipeline, GPUDevice &gpu,
                               FrameCompositionFeatureConfig config) {
  pipeline.addStage(
      std::make_unique<PresentToneMapPass>(gpu, std::move(config)),
      "FramePresentFeature", "PresentToneMapPass", true);
}

} // namespace nuri
