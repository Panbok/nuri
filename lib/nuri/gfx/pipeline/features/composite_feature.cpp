#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/composite_feature.h"

#include "nuri/core/profiling.h"
#include "nuri/gfx/shader.h"

namespace nuri {
namespace {

constexpr uint32_t kCompositePassDebugColor = 0xff55cc88u;
constexpr uint32_t kCompositeDrawDebugColor = 0xff55cc88u;
constexpr uint32_t kInvalidTextureBindlessIndex = 0xFFFFFFFFu;
constexpr std::string_view kCompositePassLabel = "Composite Pass";
constexpr std::string_view kCompositeDrawLabel = "Composite";

RenderPipelineDesc compositePipelineDesc(Format colorFormat,
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

} // namespace

CompositePass::CompositePass(GPUDevice &gpu, CompositeFeatureConfig config)
    : gpu_(gpu) {
  const std::filesystem::path basePath = config.shaderBasePath.empty()
                                             ? config.meshFragment.parent_path()
                                             : config.shaderBasePath;
  vertexPath_ = basePath / "fullscreen_copy.vert";
  fragmentPath_ = basePath / "fullscreen_copy.frag";
}

CompositePass::~CompositePass() {
  destroyBuffers();
  destroyPipelineState();
  destroyShaders();
  shader_.reset();
}

bool CompositePass::isEnabled(const FrameBuildContext &ctx) const {
  return nuri::isValid(ctx.shared.frameColorTexture);
}

Result<bool, std::string> CompositePass::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  hasPreparedDraw_ = false;
  sourceFrameColor_ = ctx.shared.frameColorTexture;
  if (!nuri::isValid(sourceFrameColor_)) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return initResult;
  }
  auto bufferResult = ensureFrameBufferCapacity(sizeof(ForwardSceneFrameData));
  if (bufferResult.hasError()) {
    return bufferResult;
  }
  auto pipelineResult = ensurePipeline();
  if (pipelineResult.hasError()) {
    return pipelineResult;
  }

  const uint32_t sceneColorTexId =
      gpu_.getTextureBindlessIndex(sourceFrameColor_);
  if (sceneColorTexId == kInvalidTextureBindlessIndex) {
    return Result<bool, std::string>::makeError(
        "CompositePass::prepare: invalid frame color bindless index");
  }

  frameData_ = ForwardSceneFrameData{
      .view = glm::mat4(1.0f),
      .proj = glm::mat4(1.0f),
      .cameraPos = glm::vec4(0.0f),
      .cubemapTexId = 0u,
      .hasCubemap = 0u,
      .irradianceTexId = 0u,
      .prefilteredGgxTexId = 0u,
      .prefilteredCharlieTexId = 0u,
      .brdfLutTexId = 0u,
      .flags = 0u,
      .cubemapSamplerId = 0u,
      .materialSamplerId = 0u,
      .sceneColorTexId = sceneColorTexId,
      .sceneColorSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u),
      .sceneColorHalfResTexId = 0u,
      .sceneColorQuarterResTexId = 0u,
      .directionalLightBufferAddress = 0u,
      .localLightBufferAddress = 0u,
      .directionalLightCount = 0u,
      .localLightCount = 0u,
  };

  if (!frameDataUploadValid_ || uploadedFrameData_ != frameData_) {
    Result<bool, std::string> updateResult =
        [&]() -> Result<bool, std::string> {
      std::optional<Result<bool, std::string>> result;
      NURI_PROFILER_ZONE("CompositePass.frame_data_upload",
                         NURI_PROFILER_COLOR_CMD_COPY);
      const std::span<const std::byte> frameBytes{
          reinterpret_cast<const std::byte *>(&frameData_), sizeof(frameData_)};
      result.emplace(gpu_.updateBuffer(frameBuffer_->handle(), frameBytes, 0u));
      NURI_PROFILER_ZONE_END();
      return std::move(*result);
    }();
    if (updateResult.hasError()) {
      return updateResult;
    }
    uploadedFrameData_ = frameData_;
    frameDataUploadValid_ = true;
  }

  pushConstants_.frameDataAddress =
      gpu_.getBufferDeviceAddress(frameBuffer_->handle());
  if (pushConstants_.frameDataAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "CompositePass::prepare: invalid frame data address");
  }

  drawItem_ = DrawItem{};
  drawItem_.pipeline = pipelineHandle_;
  drawItem_.vertexCount = 3u;
  drawItem_.instanceCount = 1u;
  drawItem_.pushConstants = std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(&pushConstants_),
      sizeof(pushConstants_));
  drawItem_.debugLabel = kCompositeDrawLabel;
  drawItem_.debugColor = kCompositeDrawDebugColor;
  hasPreparedDraw_ = true;

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> CompositePass::build(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  if (!hasPreparedDraw_ || !nuri::isValid(sourceFrameColor_)) {
    return Result<bool, std::string>::makeResult(true);
  }

  Result<bool, std::string> passBuildResult =
      [&]() -> Result<bool, std::string> {
    std::optional<Result<bool, std::string>> result;
    NURI_PROFILER_ZONE("CompositePass.pass_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    do {
      RenderGraphGraphicsPassDesc passDesc{};
      passDesc.color = {.loadOp = LoadOp::Clear,
                        .storeOp = StoreOp::Store,
                        .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
      passDesc.draws = std::span<const DrawItem>(&drawItem_, 1u);
      passDesc.debugLabel = kCompositePassLabel;
      passDesc.debugColor = kCompositePassDebugColor;

      auto addResult = ctx.graph.addGraphicsPass(passDesc);
      if (addResult.hasError()) {
        result.emplace(Result<bool, std::string>::makeError(addResult.error()));
        break;
      }

      auto bufferImportResult = ctx.graph.importBuffer(
          frameBuffer_->handle(), "composite_frame_data_buffer");
      if (bufferImportResult.hasError()) {
        result.emplace(
            Result<bool, std::string>::makeError(bufferImportResult.error()));
        break;
      }
      auto bufferReadResult = ctx.graph.addBufferRead(
          addResult.value(), bufferImportResult.value());
      if (bufferReadResult.hasError()) {
        result.emplace(
            Result<bool, std::string>::makeError(bufferReadResult.error()));
        break;
      }

      auto textureImportResult = ctx.graph.importTexture(
          sourceFrameColor_, "composite_frame_color_read");
      if (textureImportResult.hasError()) {
        result.emplace(
            Result<bool, std::string>::makeError(textureImportResult.error()));
        break;
      }
      auto readResult = ctx.graph.addTextureRead(addResult.value(),
                                                 textureImportResult.value());
      if (readResult.hasError()) {
        result.emplace(
            Result<bool, std::string>::makeError(readResult.error()));
        break;
      }

      result.emplace(Result<bool, std::string>::makeResult(true));
    } while (false);
    NURI_PROFILER_ZONE_END();
    return std::move(*result);
  }();

  return passBuildResult;
}

Result<bool, std::string> CompositePass::ensureInitialized() {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto shaderResult = createShaders();
  if (shaderResult.hasError()) {
    return shaderResult;
  }
  auto bufferResult = ensureFrameBufferCapacity(sizeof(ForwardSceneFrameData));
  if (bufferResult.hasError()) {
    return bufferResult;
  }

  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> CompositePass::createShaders() {
  destroyShaders();
  shader_ = Shader::create("frame_composite", gpu_);
  if (!shader_) {
    return Result<bool, std::string>::makeError(
        "CompositePass::createShaders: failed to create shader wrapper");
  }

  auto vertexResult =
      shader_->compileFromFile(vertexPath_.string(), ShaderStage::Vertex);
  if (vertexResult.hasError()) {
    return Result<bool, std::string>::makeError(vertexResult.error());
  }
  auto fragmentResult =
      shader_->compileFromFile(fragmentPath_.string(), ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }

  vertexShader_ = vertexResult.value();
  fragmentShader_ = fragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> CompositePass::ensurePipeline() {
  const Format colorFormat = gpu_.getSwapchainFormat();
  if (nuri::isValid(pipelineHandle_) && pipelineColorFormat_ == colorFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!nuri::isValid(vertexShader_) || !nuri::isValid(fragmentShader_)) {
    return Result<bool, std::string>::makeError(
        "CompositePass::ensurePipeline: vertex or fragment shader handle is "
        "invalid");
  }

  destroyPipelineState();
  auto pipelineResult = gpu_.createRenderPipeline(
      compositePipelineDesc(colorFormat, vertexShader_, fragmentShader_),
      "frame_composite");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }

  pipelineHandle_ = pipelineResult.value();
  pipelineColorFormat_ = colorFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
CompositePass::ensureFrameBufferCapacity(size_t requiredBytes) {
  const size_t requested =
      std::max(requiredBytes, sizeof(ForwardSceneFrameData));
  if (frameBuffer_ && frameBuffer_->valid() &&
      frameBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (frameBuffer_ && frameBuffer_->valid()) {
    gpu_.destroyBuffer(frameBuffer_->handle());
  }
  frameBuffer_.reset();
  auto bufferResult = Buffer::create(gpu_,
                                     BufferDesc{.usage = BufferUsage::Storage,
                                                .storage = Storage::Device,
                                                .size = requested},
                                     "frame_composite_data");
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }
  frameBuffer_ = std::move(bufferResult.value());
  frameBufferCapacityBytes_ = requested;
  frameDataUploadValid_ = false;
  return Result<bool, std::string>::makeResult(true);
}

void CompositePass::destroyPipelineState() {
  if (nuri::isValid(pipelineHandle_)) {
    gpu_.destroyRenderPipeline(pipelineHandle_);
  }
  pipelineHandle_ = {};
  pipelineColorFormat_ = Format::Count;
}

void CompositePass::destroyShaders() {
  if (nuri::isValid(vertexShader_)) {
    gpu_.destroyShaderModule(vertexShader_);
  }
  if (nuri::isValid(fragmentShader_)) {
    gpu_.destroyShaderModule(fragmentShader_);
  }
  vertexShader_ = {};
  fragmentShader_ = {};
}

void CompositePass::destroyBuffers() {
  if (frameBuffer_ && frameBuffer_->valid()) {
    gpu_.destroyBuffer(frameBuffer_->handle());
  }
  frameBuffer_.reset();
  frameBufferCapacityBytes_ = 0;
  frameDataUploadValid_ = false;
  frameData_ = {};
  uploadedFrameData_ = {};
  pushConstants_ = {};
  drawItem_ = {};
  sourceFrameColor_ = {};
}

CompositeFeature::CompositeFeature(GPUDevice &gpu,
                                   CompositeFeatureConfig config)
    : pass_(gpu, std::move(config)) {}

std::span<RenderFeaturePass *const> CompositeFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
