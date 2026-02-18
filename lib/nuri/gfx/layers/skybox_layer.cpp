#include "nuri/pch.h"

#include "nuri/gfx/layers/skybox_layer.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/scene/render_scene.h"

namespace nuri {
namespace {

constexpr uint32_t kSkyboxVertexCount = 36;

} // namespace

SkyboxLayer::SkyboxLayer(GPUDevice &gpu) : gpu_(gpu) {}

SkyboxLayer::~SkyboxLayer() { onDetach(); }

void SkyboxLayer::onAttach() {
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    NURI_LOG_WARNING("SkyboxLayer::onAttach: %s", initResult.error().c_str());
  }
}

void SkyboxLayer::onDetach() {
  destroyPerFrameBuffer();
  skyboxShader_.reset();
  skyboxPipeline_.reset();
  skyboxVertexShader_ = {};
  skyboxFragmentShader_ = {};
  skyboxPipelineHandle_ = {};
  initialized_ = false;
}

void SkyboxLayer::onResize(int32_t, int32_t) {}

Result<bool, std::string>
SkyboxLayer::buildRenderPasses(RenderFrameContext &frame, RenderPassList &out) {
  NURI_PROFILER_FUNCTION();

  if (frame.settings && !frame.settings->skybox.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (!frame.scene) {
    return Result<bool, std::string>::makeError(
        "SkyboxLayer::buildRenderPasses: frame scene is null");
  }

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return Result<bool, std::string>::makeError(initResult.error());
  }

  RenderPass pass{};
  pass.color = {.loadOp = LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}};
  pass.debugLabel = "Skybox Pass";
  pass.debugColor = 0xff3366ff;

  const Texture *cubemap = frame.scene->environmentCubemap();
  if (cubemap && cubemap->valid()) {
    frameData_ = FrameData{
        .view = frame.camera.view,
        .proj = frame.camera.proj,
        .cameraPos = frame.camera.cameraPos,
        .cubemapTexId = gpu_.getTextureBindlessIndex(cubemap->handle()),
        .hasCubemap = 1,
        ._padding0 = 0,
        ._padding1 = 0,
    };

    const size_t requiredBytes = sizeof(frameData_);
    auto bufferResult = ensurePerFrameBufferCapacity(requiredBytes);
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }

    const std::span<const std::byte> perFrameBytes{
        reinterpret_cast<const std::byte *>(&frameData_), requiredBytes};
    auto updateResult =
        gpu_.updateBuffer(perFrameBuffer_->handle(), perFrameBytes, 0);
    if (updateResult.hasError()) {
      return Result<bool, std::string>::makeError(updateResult.error());
    }

    const uint64_t baseAddress =
        gpu_.getBufferDeviceAddress(perFrameBuffer_->handle());
    if (baseAddress == 0) {
      return Result<bool, std::string>::makeError(
          "SkyboxLayer::buildRenderPasses: invalid per-frame buffer address");
    }

    pushConstants_ = PushConstants{.frameDataAddress = baseAddress};

    drawItem_ = DrawItem{};
    drawItem_.pipeline = skyboxPipelineHandle_;
    drawItem_.vertexCount = kSkyboxVertexCount;
    drawItem_.pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&pushConstants_),
        sizeof(pushConstants_));
    drawItem_.debugLabel = "Skybox";
    drawItem_.debugColor = 0xff3366ff;

    pass.draws = std::span<const DrawItem>(&drawItem_, 1);
  }

  out.push_back(pass);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> SkyboxLayer::ensureInitialized() {
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

  auto bufferResult = ensurePerFrameBufferCapacity(sizeof(FrameData));
  if (bufferResult.hasError()) {
    return bufferResult;
  }

  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SkyboxLayer::ensurePerFrameBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(FrameData));
  if (perFrameBuffer_ && perFrameBuffer_->valid() &&
      perFrameBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }

  destroyPerFrameBuffer();

  const BufferDesc perFrameDesc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = requested,
  };
  auto perFrameResult =
      Buffer::create(gpu_, perFrameDesc, "skybox_per_frame_buffer");
  if (perFrameResult.hasError()) {
    return Result<bool, std::string>::makeError(perFrameResult.error());
  }

  perFrameBuffer_ = std::move(perFrameResult.value());
  perFrameBufferCapacityBytes_ = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> SkyboxLayer::createShaders() {
  skyboxShader_ = Shader::create("skybox", gpu_);
  struct ShaderSpec {
    Shader *shader = nullptr;
    std::string_view path{};
    ShaderStage stage = ShaderStage::Vertex;
    ShaderHandle *outHandle = nullptr;
  };
  const std::array<ShaderSpec, 2> shaderSpecs = {
      ShaderSpec{skyboxShader_.get(), "assets/shaders/skybox.vert",
                 ShaderStage::Vertex, &skyboxVertexShader_},
      ShaderSpec{skyboxShader_.get(), "assets/shaders/skybox.frag",
                 ShaderStage::Fragment, &skyboxFragmentShader_},
  };

  for (const ShaderSpec &spec : shaderSpecs) {
    if (!spec.shader || !spec.outHandle) {
      return Result<bool, std::string>::makeError(
          "SkyboxLayer::createShaders: invalid shader spec");
    }
    auto compileResult = spec.shader->compileFromFile(spec.path, spec.stage);
    if (compileResult.hasError()) {
      return Result<bool, std::string>::makeError(compileResult.error());
    }
    *spec.outHandle = compileResult.value();
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> SkyboxLayer::createPipeline() {
  skyboxPipeline_ = Pipeline::create(gpu_);
  if (!skyboxPipeline_) {
    return Result<bool, std::string>::makeError(
        "SkyboxLayer::createPipeline: failed to create skybox pipeline");
  }

  const RenderPipelineDesc skyboxDesc{
      .vertexInput = {},
      .vertexShader = skyboxVertexShader_,
      .fragmentShader = skyboxFragmentShader_,
      .colorFormats = {gpu_.getSwapchainFormat()},
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

  return Result<bool, std::string>::makeResult(true);
}

void SkyboxLayer::destroyPerFrameBuffer() {
  if (perFrameBuffer_ && perFrameBuffer_->valid()) {
    gpu_.destroyBuffer(perFrameBuffer_->handle());
  }
  perFrameBuffer_.reset();
  perFrameBufferCapacityBytes_ = 0;
}

} // namespace nuri
