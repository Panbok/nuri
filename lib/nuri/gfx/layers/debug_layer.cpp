#include "nuri/gfx/layers/debug_layer.h"

#include "nuri/core/profiling.h"
#include "nuri/gfx/debug_draw_3d.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/scene/render_scene.h"

namespace nuri {
namespace {

constexpr uint32_t kGridPassDebugColor = 0xff66aaff;
constexpr uint32_t kGridDrawDebugColor = 0xff66aaff;
constexpr uint32_t kGridVertexCount = 6;
constexpr std::string_view kGridPipelineName = "debug_grid";
constexpr std::string_view kGridPassLabel = "DebugGrid Pass";
constexpr std::string_view kGridDrawLabel = "DebugGrid Draw";

} // namespace

DebugLayer::DebugLayer(GPUDevice &gpu, DebugLayerConfig config,
                       std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      debugDraw3D_(std::make_unique<DebugDraw3D>(
          gpu, memory != nullptr ? memory : std::pmr::get_default_resource())) {
}

DebugLayer::~DebugLayer() { onDetach(); }

void DebugLayer::onDetach() {
  debugDraw3D_.reset();
  resetGridState();
}

Result<bool, std::string> DebugLayer::ensureGridInitialized() {
  return createGridShaders();
}

Result<bool, std::string> DebugLayer::createGridShaders() {
  if (gridShader_ && nuri::isValid(gridVertexShader_) &&
      nuri::isValid(gridFragmentShader_)) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (config_.vertex.empty() || config_.fragment.empty()) {
    return Result<bool, std::string>::makeError(
        "DebugLayer::createGridShaders: vertex or fragment shader path is "
        "empty");
  }

  gridShader_ = Shader::create("debug_grid", gpu_);
  if (!gridShader_) {
    return Result<bool, std::string>::makeError(
        "DebugLayer::createGridShaders: failed to create grid shader wrapper");
  }

  const std::string vertexShaderPath = config_.vertex.string();
  auto vertexResult =
      gridShader_->compileFromFile(vertexShaderPath, ShaderStage::Vertex);
  if (vertexResult.hasError()) {
    gridVertexShader_ = {};
    gridFragmentShader_ = {};
    gridShader_.reset();
    return Result<bool, std::string>::makeError(vertexResult.error());
  }
  const std::string fragmentShaderPath = config_.fragment.string();
  auto fragmentResult =
      gridShader_->compileFromFile(fragmentShaderPath, ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    gridVertexShader_ = {};
    gridFragmentShader_ = {};
    gridShader_.reset();
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }

  gridVertexShader_ = vertexResult.value();
  gridFragmentShader_ = fragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> DebugLayer::ensureGridPipeline(Format colorFormat,
                                                         Format depthFormat) {
  auto shaderResult = ensureGridInitialized();
  if (shaderResult.hasError()) {
    return shaderResult;
  }

  if (nuri::isValid(gridPipelineHandle_) &&
      gridPipelineColorFormat_ == colorFormat &&
      gridPipelineDepthFormat_ == depthFormat) {
    return Result<bool, std::string>::makeResult(true);
  }

  gridPipeline_.reset();
  gridPipelineHandle_ = {};

  gridPipeline_ = Pipeline::create(gpu_);
  if (!gridPipeline_) {
    return Result<bool, std::string>::makeError(
        "DebugLayer::ensureGridPipeline: failed to create grid pipeline "
        "wrapper");
  }

  const RenderPipelineDesc desc{
      .vertexInput = {},
      .vertexShader = gridVertexShader_,
      .fragmentShader = gridFragmentShader_,
      .colorFormats = {colorFormat},
      .depthFormat = depthFormat,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = true,
  };

  auto pipelineResult =
      gridPipeline_->createRenderPipeline(desc, kGridPipelineName);
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }

  gridPipelineHandle_ = pipelineResult.value();
  gridPipelineColorFormat_ = colorFormat;
  gridPipelineDepthFormat_ = depthFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<RenderPass, std::string>
DebugLayer::buildGridPass(const RenderFrameContext &frame,
                          bool hasPriorColorPass) {
  const TextureHandle depthTexture = frame.sharedDepthTexture;
  const bool hasDepth = nuri::isValid(depthTexture);
  const Format depthFormat =
      hasDepth ? gpu_.getTextureFormat(depthTexture) : Format::Count;

  auto pipelineResult =
      ensureGridPipeline(gpu_.getSwapchainFormat(), depthFormat);
  if (pipelineResult.hasError()) {
    return Result<RenderPass, std::string>::makeError(pipelineResult.error());
  }

  gridPushConstants_ = GridPushConstants{
      .mvp = frame.camera.proj * frame.camera.view,
      .cameraPos = frame.camera.cameraPos,
      .origin = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f),
  };

  gridDrawItem_ = DrawItem{};
  gridDrawItem_.pipeline = gridPipelineHandle_;
  gridDrawItem_.vertexCount = kGridVertexCount;
  gridDrawItem_.instanceCount = 1;
  gridDrawItem_.pushConstants = std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(&gridPushConstants_),
      sizeof(gridPushConstants_));
  gridDrawItem_.debugLabel = kGridDrawLabel;
  gridDrawItem_.debugColor = kGridDrawDebugColor;
  if (hasDepth) {
    gridDrawItem_.useDepthState = true;
    gridDrawItem_.depthState = {
        .compareOp = CompareOp::LessEqual,
        .isDepthWriteEnabled = false,
    };
  }

  RenderPass pass{};
  pass.color = {.loadOp = hasPriorColorPass ? LoadOp::Load : LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}};
  pass.debugLabel = kGridPassLabel;
  pass.debugColor = kGridPassDebugColor;
  if (hasDepth) {
    pass.depthTexture = depthTexture;
    pass.depth = {.loadOp = LoadOp::Load,
                  .storeOp = StoreOp::Store,
                  .clearDepth = 1.0f,
                  .clearStencil = 0};
  }
  pass.draws = std::span<const DrawItem>(&gridDrawItem_, 1);

  return Result<RenderPass, std::string>::makeResult(pass);
}

void DebugLayer::resetGridState() {
  gridPipeline_.reset();
  gridShader_.reset();

  gridVertexShader_ = {};
  gridFragmentShader_ = {};
  gridPipelineHandle_ = {};
  gridPipelineColorFormat_ = Format::Count;
  gridPipelineDepthFormat_ = Format::Count;

  gridPushConstants_ = GridPushConstants{};
  gridDrawItem_ = DrawItem{};
}

Result<bool, std::string>
DebugLayer::appendModelBoundsPass(const RenderFrameContext &frame,
                                  RenderPassList &out) {
  if (!debugDraw3D_ || !frame.scene ||
      !nuri::isValid(frame.sharedDepthTexture)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const std::span<const OpaqueRenderable> renderables =
      frame.scene->opaqueRenderables();
  if (renderables.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  debugDraw3D_->clear();
  debugDraw3D_->setMatrix(frame.camera.proj * frame.camera.view);
  for (const OpaqueRenderable &renderable : renderables) {
    if (!renderable.model) {
      continue;
    }
    debugDraw3D_->box(renderable.modelMatrix, renderable.model->bounds(),
                      glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
  }

  auto linePassResult = debugDraw3D_->buildRenderPass(frame.sharedDepthTexture);
  if (linePassResult.hasError()) {
    return Result<bool, std::string>::makeError(linePassResult.error());
  }

  out.push_back(linePassResult.value());
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugLayer::buildRenderPasses(RenderFrameContext &frame, RenderPassList &out) {
  NURI_PROFILER_FUNCTION();

  if (!frame.settings || !frame.settings->debug.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (frame.settings->debug.grid) {
    const bool hasPriorColorPass = !out.empty();
    auto gridPassResult = buildGridPass(frame, hasPriorColorPass);
    if (gridPassResult.hasError()) {
      return Result<bool, std::string>::makeError(gridPassResult.error());
    }
    out.push_back(gridPassResult.value());
  }

  if (frame.settings->debug.modelBounds) {
    auto boundsPassResult = appendModelBoundsPass(frame, out);
    if (boundsPassResult.hasError()) {
      return boundsPassResult;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
