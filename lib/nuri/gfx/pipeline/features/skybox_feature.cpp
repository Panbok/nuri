#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/skybox_feature.h"

#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"

namespace nuri {
namespace {

constexpr uint32_t kSkyboxVertexCount = 36;

[[nodiscard]] bool isMsaa4xSelected(const RenderFrameContext &frame) noexcept {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  return sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
         AntiAliasingMode::MSAA4x;
}

} // namespace

SkyboxPass::SkyboxPass(GPUDevice &gpu, const SkyboxFeatureConfig &config)
    : gpu_(gpu), config_(config) {}

SkyboxPass::~SkyboxPass() {
  destroyFrameBuffer();
  skyboxShader_.reset();
  skyboxMsaaPipeline_.reset();
  skyboxPipeline_.reset();
  skyboxVertexShader_ = {};
  skyboxFragmentShader_ = {};
  skyboxPipelineHandle_ = {};
  skyboxMsaaPipelineHandle_ = {};
}

bool SkyboxPass::isEnabled(const FrameBuildContext &ctx) const {
  return ctx.frame.settings == nullptr || ctx.frame.settings->skybox.enabled;
}

Result<bool, std::string> SkyboxPass::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  return prepareSkyboxDraw(ctx);
}

Result<bool, std::string> SkyboxPass::build(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();

  RenderGraphGraphicsPassDesc passDesc{};
  const bool msaaSelected = isMsaa4xSelected(ctx.frame);
  const bool usedMsaa =
      msaaSelected && nuri::isValid(ctx.shared.msaaSceneColorTexture);
  const TextureHandle sceneColorTexture = usedMsaa
                                              ? ctx.shared.msaaSceneColorTexture
                                              : ctx.shared.sceneColorTexture;
  passDesc.color = {.loadOp = LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}};

  if (nuri::isValid(sceneColorTexture)) {
    auto sceneColorImport = ctx.graph.importTexture(
        sceneColorTexture,
        msaaSelected ? "skybox_msaa_scene_color" : "skybox_scene_color");
    if (sceneColorImport.hasError()) {
      return Result<bool, std::string>::makeError(sceneColorImport.error());
    }
    passDesc.colorTexture = sceneColorImport.value();
    if (usedMsaa) {
      ctx.shared.msaaSceneColorGraphTexture = sceneColorImport.value();
      ctx.frame.metrics.antiAliasing.msaaColorGraphPublished = true;
    } else {
      ctx.shared.sceneColorGraphTexture = sceneColorImport.value();
    }
  }

  passDesc.draws = hasPreparedDraw_ ? std::span<const DrawItem>(&drawItem_, 1u)
                                    : std::span<const DrawItem>{};
  passDesc.debugLabel = "Skybox Pass";
  passDesc.debugColor = 0xff3366ff;

  auto addResult = ctx.graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();

  if (hasPreparedDraw_ && frameBuffer_ && frameBuffer_->valid()) {
    auto frameBufferResult = ctx.graph.importBuffer(frameBuffer_->handle(),
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

  if (hasPreparedDraw_ && ctx.frame.scene != nullptr) {
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

  auto bufferResult = ensureFrameBufferCapacity(sizeof(FrameData));
  if (bufferResult.hasError()) {
    return bufferResult;
  }

  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SkyboxPass::ensureFrameBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(FrameData));
  if (frameBuffer_ && frameBuffer_->valid() &&
      frameBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }

  destroyFrameBuffer();

  const BufferDesc frameBufferDesc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = requested,
  };
  auto bufferResult =
      Buffer::create(gpu_, frameBufferDesc, "skybox_frame_buffer");
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }

  frameBuffer_ = std::move(bufferResult.value());
  frameBufferCapacityBytes_ = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> SkyboxPass::createShaders() {
  skyboxShader_ = Shader::create("skybox", gpu_);
  struct ShaderSpec {
    Shader *shader = nullptr;
    const std::filesystem::path *path = nullptr;
    ShaderStage stage = ShaderStage::Vertex;
    ShaderHandle *outHandle = nullptr;
  };
  const std::array<ShaderSpec, 2> shaderSpecs = {
      ShaderSpec{skyboxShader_.get(), &config_.vertex, ShaderStage::Vertex,
                 &skyboxVertexShader_},
      ShaderSpec{skyboxShader_.get(), &config_.fragment, ShaderStage::Fragment,
                 &skyboxFragmentShader_},
  };

  for (const ShaderSpec &spec : shaderSpecs) {
    if (!spec.shader || !spec.outHandle || !spec.path ||
        spec.path->string().empty()) {
      return Result<bool, std::string>::makeError(
          "SkyboxPass::createShaders: empty shader path");
    }
    const std::string shaderPath = spec.path->string();
    auto compileResult = spec.shader->compileFromFile(shaderPath, spec.stage);
    if (compileResult.hasError()) {
      return Result<bool, std::string>::makeError(compileResult.error());
    }
    *spec.outHandle = compileResult.value();
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> SkyboxPass::createPipeline() {
  skyboxPipeline_ = Pipeline::create(gpu_);
  skyboxMsaaPipeline_ = Pipeline::create(gpu_);
  if (!skyboxPipeline_ || !skyboxMsaaPipeline_) {
    return Result<bool, std::string>::makeError(
        "SkyboxPass::createPipeline: failed to create skybox pipeline");
  }

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

  auto pipelineResult =
      skyboxPipeline_->createRenderPipeline(skyboxDesc, "skybox");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  skyboxPipelineHandle_ = skyboxPipeline_->getRenderPipeline();

  RenderPipelineDesc skyboxMsaaDesc = skyboxDesc;
  skyboxMsaaDesc.numSamples = kMsaa4xSampleCount;
  auto msaaPipelineResult = skyboxMsaaPipeline_->createRenderPipeline(
      skyboxMsaaDesc, "skybox_msaa4x");
  if (msaaPipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(msaaPipelineResult.error());
  }
  skyboxMsaaPipelineHandle_ = skyboxMsaaPipeline_->getRenderPipeline();

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SkyboxPass::prepareSkyboxDraw(FrameBuildContext &ctx) {
  RenderFrameContext &frame = ctx.frame;
  hasPreparedDraw_ = false;
  drawItem_ = DrawItem{};

  if (frame.scene == nullptr) {
    return Result<bool, std::string>::makeError(
        "SkyboxPass::prepareSkyboxDraw: frame scene is null");
  }

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return Result<bool, std::string>::makeError(initResult.error());
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
  if (nuri::isValid(ctx.shared.sceneColorTexture)) {
    sceneColorTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.sceneColorTexture);
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

  auto bufferResult = ensureFrameBufferCapacity(sizeof(frameData_));
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }

  const std::span<const std::byte> frameBytes{
      reinterpret_cast<const std::byte *>(&frameData_), sizeof(frameData_)};
  auto updateResult = gpu_.updateBuffer(frameBuffer_->handle(), frameBytes, 0u);
  if (updateResult.hasError()) {
    return Result<bool, std::string>::makeError(updateResult.error());
  }

  const uint64_t baseAddress =
      gpu_.getBufferDeviceAddress(frameBuffer_->handle());
  if (baseAddress == 0) {
    return Result<bool, std::string>::makeError(
        "SkyboxPass::prepareSkyboxDraw: invalid frame buffer address");
  }

  pushConstants_ = PushConstants{
      .frameDataAddress = baseAddress,
  };
  drawItem_ = DrawItem{};
  drawItem_.pipeline = isMsaa4xSelected(frame) ? skyboxMsaaPipelineHandle_
                                               : skyboxPipelineHandle_;
  drawItem_.vertexCount = kSkyboxVertexCount;
  drawItem_.pushConstants = std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(&pushConstants_),
      sizeof(pushConstants_));
  drawItem_.debugLabel = "Skybox";
  drawItem_.debugColor = 0xff3366ff;
  hasPreparedDraw_ = true;

  return Result<bool, std::string>::makeResult(true);
}

void SkyboxPass::destroyFrameBuffer() {
  if (frameBuffer_ && frameBuffer_->valid()) {
    gpu_.destroyBuffer(frameBuffer_->handle());
  }
  frameBuffer_.reset();
  frameBufferCapacityBytes_ = 0;
}

SkyboxFeature::SkyboxFeature(GPUDevice &gpu, SkyboxFeatureConfig config)
    : pass_(gpu, std::move(config)) {}

std::span<RenderFeaturePass *const> SkyboxFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
