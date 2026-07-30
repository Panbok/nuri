#include "nuri/gfx/pipeline/features/skybox_feature.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"
namespace nuri {
namespace {
constexpr uint32_t kSkyboxVertexCount = 36;
[[nodiscard]] CoverageMode
selectedCoverage(const RenderFrameContext &frame) noexcept {
  return frame.presentationAA.coverage;
}
} // namespace

SkyboxPass::SkyboxPass(GPUDevice &gpu, const SkyboxFeatureConfig &config)
    : gpu_(gpu), config_(config) {}

SkyboxPass::~SkyboxPass() {
  destroyFrameBuffers();
  for (OwnedRenderPipelineHandle &pipeline : skyboxDepthPipelines_) {
    pipeline.reset();
  }
  for (OwnedRenderPipelineHandle &pipeline : skyboxPipelines_) {
    pipeline.reset();
  }
  skyboxVertexShader_ = {};
  skyboxFragmentShader_ = {};
}

bool SkyboxPass::isEnabled(const FrameBuildContext &ctx) const {
  return ctx.frame.settings->skybox.enabled;
}

Result<bool, std::string> SkyboxPass::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  return prepareSkyboxDraw(ctx);
}

Result<bool, std::string> SkyboxPass::build(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  if (!hasPreparedDraw_) {
    return Result<bool, std::string>::makeResult(true);
  }
  RenderGraphGraphicsPassDesc passDesc{};
  const CoverageMode coverage = selectedCoverage(ctx.frame);
  const bool msaaSelected = coverage != CoverageMode::Sample1;
  const bool usedMsaa =
      msaaSelected &&
      nuri::isValid(ctx.shared[FrameTextureSlot::MsaaSceneColor].texture) &&
      nuri::isValid(ctx.shared[FrameTextureSlot::MsaaSceneColor].graph) &&
      nuri::isValid(ctx.shared[FrameTextureSlot::MsaaSceneDepth].texture) &&
      nuri::isValid(ctx.shared[FrameTextureSlot::MsaaSceneDepth].graph);
  const TextureHandle sceneColorTexture =
      usedMsaa ? ctx.shared[FrameTextureSlot::MsaaSceneColor].texture
               : ctx.shared[FrameTextureSlot::SceneColor].texture;
  RenderGraphTextureId sceneColorGraphTexture =
      usedMsaa ? ctx.shared[FrameTextureSlot::MsaaSceneColor].graph
               : ctx.shared[FrameTextureSlot::SceneColor].graph;
  const RenderGraphTextureId sceneDepthGraphTexture =
      usedMsaa ? ctx.shared[FrameTextureSlot::MsaaSceneDepth].graph
               : ctx.shared[FrameTextureSlot::SceneDepth].graph;
  const bool useDepthTest = nuri::isValid(sceneDepthGraphTexture);
  passDesc.color = {.loadOp = useDepthTest ? LoadOp::Load : LoadOp::DontCare,
                    .storeOp = StoreOp::Store,
                    .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}};
  if (!nuri::isValid(sceneColorGraphTexture) &&
      nuri::isValid(sceneColorTexture)) {
    auto sceneColorImport = ctx.graph.importTexture(
        sceneColorTexture,
        usedMsaa ? "skybox_msaa_scene_color" : "skybox_scene_color");
    if (sceneColorImport.hasError()) {
      return Result<bool, std::string>::makeError(sceneColorImport.error());
    }
    sceneColorGraphTexture = sceneColorImport.value();
  }
  if (!nuri::isValid(sceneColorGraphTexture)) {
    return Result<bool, std::string>::makeResult(true);
  }
  passDesc.colorTexture = sceneColorGraphTexture;
  if (useDepthTest) {
    passDesc.depth = {.loadOp = LoadOp::Load,
                      .storeOp = StoreOp::Store,
                      .clearDepth = 1.0f,
                      .clearStencil = 0u};
    passDesc.depthTexture = sceneDepthGraphTexture;
    drawItem_.pipeline =
        skyboxDepthPipelines_[coverageModeIndex(
                                  usedMsaa ? coverage : CoverageMode::Sample1)]
            .get();
    drawItem_.useDepthState = true;
    drawItem_.depthState = {.compareOp = CompareOp::LessEqual,
                            .isDepthWriteEnabled = false};
  } else {
    drawItem_.pipeline =
        skyboxPipelines_[coverageModeIndex(usedMsaa ? coverage
                                                    : CoverageMode::Sample1)]
            .get();
    drawItem_.useDepthState = false;
    drawItem_.depthState = {};
  }
  passDesc.draws = std::span<const DrawItem>(&drawItem_, 1u);
  passDesc.gpuTimingScope = GpuTimingScope::Skybox;
  passDesc.debugLabel = "Skybox Pass";
  passDesc.debugColor = 0xff3366ff;
  auto addResult = ctx.graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();
  if (usedMsaa) {
    ctx.shared[FrameTextureSlot::MsaaSceneColor].graph = sceneColorGraphTexture;
    ctx.frame.metrics.antiAliasing.msaaColorGraphPublished = true;
  } else {
    ctx.shared[FrameTextureSlot::SceneColor].graph = sceneColorGraphTexture;
  }
  if (nuri::isValid(preparedFrameBuffer_)) {
    auto frameBufferResult = ctx.graph.importBuffer(preparedFrameBuffer_,
                                                    "skybox_frame_data_buffer");
    if (frameBufferResult.hasError()) {
      return Result<bool, std::string>::makeError(frameBufferResult.error());
    }
    auto accessResult =
        ctx.graph.addBufferRead(passId, frameBufferResult.value());
    if (accessResult.hasError()) {
      return Result<bool, std::string>::makeError(accessResult.error());
    }
  }
  if (ctx.frame.scene != nullptr) {
    const TextureRecord *cubemap =
        ctx.resources.tryGet(ctx.frame.scene->environment().cubemap);
    if (cubemap != nullptr && nuri::isValid(cubemap->texture)) {
      auto cubemapResult = ctx.graph.importTexture(
          cubemap->texture, "skybox_environment_cubemap");
      if (cubemapResult.hasError()) {
        return Result<bool, std::string>::makeError(cubemapResult.error());
      }
      auto accessResult =
          ctx.graph.addTextureRead(passId, cubemapResult.value());
      if (accessResult.hasError()) {
        return Result<bool, std::string>::makeError(accessResult.error());
      }
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> SkyboxPass::ensureInitialized() {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaderResult = createShaders();
  if (shaderResult.hasError()) {
    return shaderResult;
  }
  auto pipelineResult = createPipeline();
  if (pipelineResult.hasError()) {
    return pipelineResult;
  }
  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

void SkyboxPass::syncFrameBufferSlots() {
  const size_t requiredSlotCount =
      static_cast<size_t>(std::max(1u, gpu_.getSwapchainImageCount()));
  if (frameBufferSlots_.size() == requiredSlotCount) {
    return;
  }
  destroyFrameBuffers();
  frameBufferSlots_.resize(requiredSlotCount);
}

Result<bool, std::string>
SkyboxPass::ensureFrameBufferCapacity(FrameBufferSlot &slot,
                                      size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(FrameData));
  if (nuri::isValid(slot.buffer) && slot.capacityBytes >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (nuri::isValid(slot.buffer)) {
    gpu_.destroyBuffer(slot.buffer);
    slot = {};
  }
  const BufferDesc frameBufferDesc{
      .usage = BufferUsage::Storage,
      .storage = Storage::HostVisible,
      .size = requested,
  };
  auto bufferResult = gpu_.createBuffer(frameBufferDesc, "skybox_frame_buffer");
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }
  slot.buffer = bufferResult.value();
  slot.capacityBytes = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> SkyboxPass::createShaders() {
  struct ShaderSpec {
    const std::filesystem::path *path = nullptr;
    ShaderStage stage = ShaderStage::Vertex;
    ShaderHandle *outHandle = nullptr;
  };
  const std::array<ShaderSpec, 2> shaderSpecs = {
      ShaderSpec{&config_.vertex, ShaderStage::Vertex, &skyboxVertexShader_},
      ShaderSpec{&config_.fragment, ShaderStage::Fragment,
                 &skyboxFragmentShader_},
  };
  for (const ShaderSpec &spec : shaderSpecs) {
    if (!spec.outHandle || !spec.path || spec.path->string().empty()) {
      return Result<bool, std::string>::makeError(
          "SkyboxPass::createShaders: empty shader path");
    }
    const std::string shaderPath = spec.path->string();
    auto compileResult =
        compileShaderFile(gpu_, "skybox", shaderPath, spec.stage);
    if (compileResult.hasError()) {
      return Result<bool, std::string>::makeError(compileResult.error());
    }
    *spec.outHandle = compileResult.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> SkyboxPass::createPipeline() {
  const RenderPipelineDesc skyboxDesc{
      .vertexInput = {},
      .vertexShader = skyboxVertexShader_,
      .fragmentShader = skyboxFragmentShader_,
      .colorFormats = {kFrameCompositionSceneColorFormat},
      .depthFormat = Format::Count,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
  };
  RenderPipelineDesc skyboxDepthDesc = skyboxDesc;
  skyboxDepthDesc.depthFormat = kFrameCompositionDepthFormat;
  skyboxDepthDesc.rasterState = makeRasterPipelineState(DepthState{
      .compareOp = CompareOp::LessEqual,
      .isDepthWriteEnabled = false,
  });
  const GpuMultisampleCapabilities multisampleCapabilities =
      gpu_.getMultisampleCapabilities();
  const bool supports8x = multisampleCapabilities.sample8Color &&
                          multisampleCapabilities.sample8Depth;
  constexpr std::array colorNames{"skybox", "skybox_msaa4x", "skybox_msaa8x"};
  constexpr std::array depthNames{"skybox_depth_tested",
                                  "skybox_msaa4x_depth_tested",
                                  "skybox_msaa8x_depth_tested"};
  const std::array enabled{true, true, supports8x};
  for (size_t index = 0; index < kCoverageModeCount; ++index) {
    if (!enabled[index]) {
      continue;
    }
    const CoverageMode coverage = static_cast<CoverageMode>(index);
    RenderPipelineDesc colorDesc = skyboxDesc;
    colorDesc.numSamples = coverageSampleCount(coverage);
    auto colorResult = gpu_.createRenderPipeline(colorDesc, colorNames[index]);
    if (colorResult.hasError()) {
      return Result<bool, std::string>::makeError(colorResult.error());
    }
    skyboxPipelines_[index].reset(gpu_, colorResult.value());

    RenderPipelineDesc depthDesc = skyboxDepthDesc;
    depthDesc.numSamples = colorDesc.numSamples;
    auto depthResult = gpu_.createRenderPipeline(depthDesc, depthNames[index]);
    if (depthResult.hasError()) {
      return Result<bool, std::string>::makeError(depthResult.error());
    }
    skyboxDepthPipelines_[index].reset(gpu_, depthResult.value());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SkyboxPass::prepareSkyboxDraw(FrameBuildContext &ctx) {
  RenderFrameContext &frame = ctx.frame;
  hasPreparedDraw_ = false;
  preparedFrameBuffer_ = {};
  drawItem_ = DrawItem{};
  if (frame.scene == nullptr) {
    return Result<bool, std::string>::makeError(
        "SkyboxPass::prepareSkyboxDraw: frame scene is null");
  }
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return Result<bool, std::string>::makeError(initResult.error());
  }
  syncFrameBufferSlots();
  FrameBufferSlot &frameBuffer = frameBufferSlots_[static_cast<size_t>(
      frame.frameIndex % frameBufferSlots_.size())];
  auto bufferResult =
      ensureFrameBufferCapacity(frameBuffer, sizeof(frameData_));
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }
  const TextureRecord *cubemap =
      ctx.resources.tryGet(frame.scene->environment().cubemap);
  if (cubemap == nullptr || !nuri::isValid(cubemap->texture) ||
      cubemap->bindlessIndex == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeResult(true);
  }
  uint32_t frameFlags = 0u;
  uint32_t sceneColorTexId = 0u;
  uint32_t sceneColorSamplerId = 0u;
  if (nuri::isValid(ctx.shared[FrameTextureSlot::SceneColor].texture)) {
    sceneColorTexId = gpu_.getTextureBindlessIndex(
        ctx.shared[FrameTextureSlot::SceneColor].texture);
    sceneColorSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u);
    if (sceneColorTexId != kInvalidTextureBindlessIndex) {
      frameFlags |= HasSceneColor;
    }
  }
  frameData_ = FrameData{
      .view = frame.camera.view,
      .proj = frame.camera.proj,
      .cameraPos = frame.camera.cameraPos,
      .cubemapTexId = cubemap->bindlessIndex,
      .hasCubemap = 1,
      .irradianceTexId = 0,
      .prefilteredGgxTexId = 0,
      .prefilteredCharlieTexId = 0,
      .brdfLutTexId = 0,
      .flags = frameFlags,
      .cubemapSamplerId = gpu_.getCubemapSamplerBindlessIndex(),
      .materialSamplerId = 0u,
      .sceneColorTexId = sceneColorTexId,
      .sceneColorSamplerId = sceneColorSamplerId,
      .sceneColorHalfResTexId = 0,
      .sceneColorQuarterResTexId = 0,
      .directionalLightBufferAddress = 0u,
      .localLightBufferAddress = 0u,
      .materialHeaderBufferAddress = 0u,
      .materialClearcoatBufferAddress = 0u,
      .materialSheenBufferAddress = 0u,
      .materialTransmissionBufferAddress = 0u,
      .materialSpecularBufferAddress = 0u,
      .directionalLightCount = 0u,
      .localLightCount = 0u,
      .shadowFrameBufferAddress = 0u,
      .shadowFlags = 0u,
      .materialCoverageSamplerId = kInvalidSamplerBindlessIndex,
      .materialDataSamplerId = 0u,
  };
  const std::span<const std::byte> frameBytes{
      reinterpret_cast<const std::byte *>(&frameData_), sizeof(frameData_)};
  auto updateResult = gpu_.updateBuffer(frameBuffer.buffer, frameBytes, 0u);
  if (updateResult.hasError()) {
    return Result<bool, std::string>::makeError(updateResult.error());
  }
  const uint64_t baseAddress = gpu_.getBufferDeviceAddress(frameBuffer.buffer);
  if (baseAddress == 0) {
    return Result<bool, std::string>::makeError(
        "SkyboxPass::prepareSkyboxDraw: invalid frame buffer address");
  }
  pushConstants_ = PushConstants{
      .frameDataAddress = baseAddress,
  };
  drawItem_ = DrawItem{};
  drawItem_.pipeline =
      skyboxPipelines_[coverageModeIndex(selectedCoverage(frame))].get();
  drawItem_.vertexCount = kSkyboxVertexCount;
  drawItem_.pushConstants = std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(&pushConstants_),
      sizeof(pushConstants_));
  drawItem_.debugLabel = "Skybox";
  drawItem_.debugColor = 0xff3366ff;
  preparedFrameBuffer_ = frameBuffer.buffer;
  hasPreparedDraw_ = true;
  return Result<bool, std::string>::makeResult(true);
}

void SkyboxPass::destroyFrameBuffers() {
  for (FrameBufferSlot &slot : frameBufferSlots_) {
    if (nuri::isValid(slot.buffer)) {
      gpu_.destroyBuffer(slot.buffer);
    }
    slot = {};
  }
  frameBufferSlots_.clear();
  preparedFrameBuffer_ = {};
}

void registerSkyboxStage(RenderPipeline &pipeline, GPUDevice &gpu,
                         SkyboxFeatureConfig config) {
  pipeline.addStage(std::make_unique<SkyboxPass>(gpu, std::move(config)),
                    "SkyboxFeature", "SkyboxPass");
}

} // namespace nuri
