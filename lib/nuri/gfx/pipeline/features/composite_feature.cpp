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
constexpr uint32_t kDrawDebugColor = 0xff2299ddu;
constexpr uint64_t kInitialDebugFrames = 4u;

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
  const AntiAliasingDebugView debugView = sanitizeAntiAliasingDebugView(
      renderSettingsOrDefault(ctx.frame).antiAliasing.debug.view);
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
  const GpuTimingReport timingReport = gpu_.getLatestCompletedGpuTimingReport();
  if (hasGpuTimingScope(timingReport, GpuTimingScope::SceneColorDownsample)) {
    aaMetrics.taaSceneColorDownsampleGpuTimeMs =
        timingReport.sceneColorDownsampleTimeMs;
    aaMetrics.taaSceneColorDownsampleGpuTimingSourceFrameIndex =
        timingReport.sceneColorDownsampleSourceFrameIndex;
    aaMetrics.taaSceneColorDownsampleGpuTimingAvailable = 1u;
  }

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
    if (aaMetrics.taaResolvedSceneColorPublished) {
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
  aces2SdrLut_.path = config.aces2SdrLut;
  agxLut_.path = config.agxLut;
}

PresentToneMapPass::~PresentToneMapPass() { destroyToneMapAssets(); }

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
  return ensurePipeline(gpu_.getSwapchainFormat(), "present_tonemap",
                        "PresentToneMapPass::ensurePipeline");
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
  auto readResult =
      ctx.graph.addTextureRead(addResult.value(), sourceImportResult.value());
  if (readResult.hasError()) {
    return Result<bool, std::string>::makeError(readResult.error());
  }
  if (acesLutAvailable) {
    auto lutReadResult =
        ctx.graph.addTextureRead(addResult.value(), lutImports[0]);
    if (lutReadResult.hasError()) {
      return Result<bool, std::string>::makeError(lutReadResult.error());
    }
  }
  if (agxLutAvailable) {
    auto lutReadResult =
        ctx.graph.addTextureRead(addResult.value(), lutImports[1]);
    if (lutReadResult.hasError()) {
      return Result<bool, std::string>::makeError(lutReadResult.error());
    }
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
  if (aces2SdrLut_.texture != nullptr && aces2SdrLut_.texture->valid()) {
    gpu_.destroyTexture(aces2SdrLut_.texture->handle());
  }
  if (agxLut_.texture != nullptr && agxLut_.texture->valid()) {
    gpu_.destroyTexture(agxLut_.texture->handle());
  }
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

FramePresentFeature::FramePresentFeature(GPUDevice &gpu,
                                         FrameCompositionFeatureConfig config)
    : presentPass_(gpu, std::move(config)) {}

std::span<RenderFeaturePass *const> FramePresentFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
