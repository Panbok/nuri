#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/composite_feature.h"

#include "nuri/core/profiling.h"

namespace nuri {
namespace {

constexpr uint32_t kSceneCopyFlagDownsample = 1u << 0u;
constexpr uint32_t kInvalidTextureBindlessIndex = 0xFFFFFFFFu;
constexpr uint32_t kDownsamplePassDebugColor = 0xff33aa88u;
constexpr uint32_t kResolvePassDebugColor = 0xff33cc88u;
constexpr uint32_t kPresentPassDebugColor = 0xff55cc88u;
constexpr uint32_t kDrawDebugColor = 0xff2299ddu;

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
  return config.shaderBasePath.empty() ? config.meshFragment.parent_path()
                                       : config.shaderBasePath;
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
    uint32_t debugColor, bool markColorAsFrameOutput = false) {
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
  auto addResult = graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
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
  resources_.vertexPath = basePath / "fullscreen_copy.vert";
  resources_.fragmentPath = basePath / "scene_copy.frag";
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
        kDownsamplePassDebugColor);
    if (addResult.hasError()) {
      return addResult;
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

SceneResolvePass::SceneResolvePass(GPUDevice &gpu,
                                   FrameCompositionFeatureConfig config)
    : FullscreenRenderPass(gpu) {
  const std::filesystem::path basePath = resolveShaderBasePath(config);
  resources_.vertexPath = basePath / "fullscreen_copy.vert";
  resources_.fragmentPath = basePath / "scene_copy.frag";
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

  const TextureHandle source = ctx.shared.sceneColorTexture;
  const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(source);
  if (sourceTexId == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "SceneResolvePass::build: invalid scene color bindless index");
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

  auto colorImportResult =
      ctx.graph.importTexture(ctx.shared.frameColorTexture, "frame_color");
  if (colorImportResult.hasError()) {
    return Result<bool, std::string>::makeError(colorImportResult.error());
  }
  const std::span<const TextureHandle> textureReads(&source, 1u);
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
  resources_.vertexPath = basePath / "fullscreen_copy.vert";
  resources_.fragmentPath = basePath / "tonemap_present.frag";
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

  const PushConstants pushConstants{
      .sourceTexId = sourceTexId,
      .sourceSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u),
      .exposure = 1.0f,
      .reserved0 = 0u,
  };
  const DrawItem draw = makeFullscreenDraw(
      resources_.pipelineHandle,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pushConstants),
          sizeof(pushConstants)),
      "Present ToneMap");

  auto sourceImportResult =
      ctx.graph.importTexture(source, "present_frame_color");
  if (sourceImportResult.hasError()) {
    return Result<bool, std::string>::makeError(sourceImportResult.error());
  }

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  passDesc.draws = std::span<const DrawItem>(&draw, 1u);
  passDesc.debugLabel = "Present ToneMap Pass";
  passDesc.debugColor = kPresentPassDebugColor;
  passDesc.markColorAsFrameOutput = true;
  auto addResult = ctx.graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  auto readResult =
      ctx.graph.addTextureRead(addResult.value(), sourceImportResult.value());
  if (readResult.hasError()) {
    return Result<bool, std::string>::makeError(readResult.error());
  }

  return Result<bool, std::string>::makeResult(true);
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
