#include "nuri/gfx/layers/debug_layer.h"

#include "nuri/core/profiling.h"
#include "nuri/gfx/debug_draw_3d.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"

namespace nuri {
namespace {

constexpr uint32_t kGridPassDebugColor = 0xff66aaff;
constexpr uint32_t kGridDrawDebugColor = 0xff66aaff;
constexpr uint32_t kGridVertexCount = 6;
constexpr std::string_view kGridPipelineName = "debug_grid";
constexpr std::string_view kGridPassLabel = "DebugGrid Pass";
constexpr std::string_view kGridDrawLabel = "DebugGrid Draw";

[[nodiscard]] bool isSameTextureHandle(TextureHandle a, TextureHandle b) {
  return a.index == b.index && a.generation == b.generation;
}

} // namespace

DebugLayer::DebugLayer(GPUDevice &gpu, DebugLayerConfig config,
                       std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      debugDraw3D_(std::make_unique<DebugDraw3D>(gpu, memory_)),
      transparentSortableDraws_(memory_), transparentFixedDraws_(memory_),
      transparentDependencyBuffers_(memory_) {}

DebugLayer::~DebugLayer() { onDetach(); }

void DebugLayer::onDetach() {
  debugDraw3D_.reset();
  resetGridState();
  transparentSortableDraws_.clear();
  transparentFixedDraws_.clear();
  transparentDependencyBuffers_.clear();
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

Result<bool, std::string> DebugLayer::appendModelBoundsGraphPass(
    const RenderFrameContext &frame, RenderGraphBuilder &graph,
    TextureHandle sceneDepthTexture,
    RenderGraphTextureId sceneDepthGraphTexture) {
  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  if (!debugDraw3D_ || !frame.scene || !nuri::isValid(depthTexture) ||
      !frame.resources) {
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
    const ModelRecord *modelRecord = frame.resources->tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      continue;
    }
    debugDraw3D_->box(renderable.modelMatrix, modelRecord->model->bounds(),
                      glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
  }

  auto linePassResult = debugDraw3D_->buildGraphPass(depthTexture);
  if (linePassResult.hasError()) {
    return Result<bool, std::string>::makeError(linePassResult.error());
  }

  DebugDraw3D::PreparedGraphPass pass = linePassResult.value();
  if (nuri::isValid(pass.colorTextureHandle)) {
    auto colorImportResult = graph.importTexture(pass.colorTextureHandle,
                                                 "debug_pass_color_texture");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    pass.desc.colorTexture = colorImportResult.value();
  }
  if (nuri::isValid(pass.depthTextureHandle)) {
    const bool useDepthOverride =
        nuri::isValid(sceneDepthTexture) &&
        nuri::isValid(sceneDepthGraphTexture) &&
        isSameTextureHandle(pass.depthTextureHandle, sceneDepthTexture);
    if (useDepthOverride) {
      pass.desc.depthTexture = sceneDepthGraphTexture;
    } else {
      auto depthImportResult = graph.importTexture(pass.depthTextureHandle,
                                                   "debug_pass_depth_texture");
      if (depthImportResult.hasError()) {
        return Result<bool, std::string>::makeError(depthImportResult.error());
      }
      pass.desc.depthTexture = depthImportResult.value();
    }
  }

  auto addResult = graph.addGraphicsPass(pass.desc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugLayer::buildRenderGraph(RenderFrameContext &frame,
                             RenderGraphBuilder &graph) {
  NURI_PROFILER_FUNCTION();

  if (const bool *transparentStageEnabled =
          frame.channels.tryGet<bool>(kFrameChannelTransparentStageEnabled);
      transparentStageEnabled != nullptr && *transparentStageEnabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (!frame.settings || !frame.settings->debug.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  const TextureHandle sceneDepthTexture = resolveFrameDepthTexture(frame);
  RenderGraphTextureId sceneDepthGraphTexture{};
  if (const RenderGraphTextureId *publishedSceneDepth =
          frame.channels.tryGet<RenderGraphTextureId>(
              kFrameChannelSceneDepthGraphTexture);
      publishedSceneDepth != nullptr) {
    sceneDepthGraphTexture = *publishedSceneDepth;
  }

  if (frame.settings->debug.grid) {
    const bool hasPriorColorPass = graph.passCount() > 0;
    const bool hasDepth = nuri::isValid(sceneDepthTexture);
    const Format depthFormat =
        hasDepth ? gpu_.getTextureFormat(sceneDepthTexture) : Format::Count;
    auto pipelineResult =
        ensureGridPipeline(gpu_.getSwapchainFormat(), depthFormat);
    if (pipelineResult.hasError()) {
      return Result<bool, std::string>::makeError(pipelineResult.error());
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

    RenderGraphTextureId depthTextureId{};
    if (hasDepth) {
      if (nuri::isValid(sceneDepthGraphTexture)) {
        depthTextureId = sceneDepthGraphTexture;
      } else {
        auto importResult =
            graph.importTexture(sceneDepthTexture, "debug_depth_texture");
        if (importResult.hasError()) {
          return Result<bool, std::string>::makeError(importResult.error());
        }
        depthTextureId = importResult.value();
      }
    }

    RenderGraphGraphicsPassDesc gridPass{};
    gridPass.color = {.loadOp =
                          hasPriorColorPass ? LoadOp::Load : LoadOp::Clear,
                      .storeOp = StoreOp::Store,
                      .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}};
    if (hasDepth) {
      gridPass.depth = {.loadOp = LoadOp::Load,
                        .storeOp = StoreOp::Store,
                        .clearDepth = 1.0f,
                        .clearStencil = 0};
      gridPass.depthTexture = depthTextureId;
    }
    gridPass.draws = std::span<const DrawItem>(&gridDrawItem_, 1u);
    gridPass.debugLabel = kGridPassLabel;
    gridPass.debugColor = kGridPassDebugColor;

    auto addResult = graph.addGraphicsPass(gridPass);
    if (addResult.hasError()) {
      return Result<bool, std::string>::makeError(addResult.error());
    }
  }

  if (frame.settings->debug.modelBounds) {
    auto boundsPassResult = appendModelBoundsGraphPass(
        frame, graph, sceneDepthTexture, sceneDepthGraphTexture);
    if (boundsPassResult.hasError()) {
      return boundsPassResult;
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> DebugLayer::buildTransparentStageContribution(
    RenderFrameContext &frame, TransparentStageContribution &out) {
  NURI_PROFILER_FUNCTION();
  out = {};
  transparentSortableDraws_.clear();
  transparentFixedDraws_.clear();
  transparentDependencyBuffers_.clear();

  if (!frame.settings || !frame.settings->debug.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  const bool hasDepth = nuri::isValid(depthTexture);
  const glm::mat4 view = frame.camera.view;

  if (frame.settings->debug.modelBounds && debugDraw3D_ && frame.scene &&
      frame.resources && hasDepth) {
    const std::span<const Renderable> renderables = frame.scene->renderables();
    if (!renderables.empty()) {
      debugDraw3D_->clear();
      debugDraw3D_->setMatrix(frame.camera.proj * frame.camera.view);
      float farthestDepth = 0.0f;
      for (const Renderable &renderable : renderables) {
        const ModelRecord *modelRecord =
            frame.resources->tryGet(renderable.model);
        if (!modelRecord || !modelRecord->model) {
          continue;
        }
        debugDraw3D_->box(renderable.modelMatrix, modelRecord->model->bounds(),
                          glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
        const glm::vec3 center = glm::vec3(
            renderable.modelMatrix *
            glm::vec4(modelRecord->model->bounds().getCenter(), 1.0f));
        farthestDepth =
            std::max(farthestDepth, -(view * glm::vec4(center, 1.0f)).z);
      }

      auto linePassResult = debugDraw3D_->buildGraphPass(depthTexture);
      if (linePassResult.hasError()) {
        return Result<bool, std::string>::makeError(linePassResult.error());
      }
      const DebugDraw3D::PreparedGraphPass pass = linePassResult.value();
      if (!pass.desc.draws.empty()) {
        transparentSortableDraws_.push_back(TransparentStageSortableDraw{
            .draw = pass.desc.draws.front(),
            .sortDepth = farthestDepth,
            .stableOrder = 0u,
        });
        for (const BufferHandle buffer : pass.desc.dependencyBuffers) {
          if (nuri::isValid(buffer)) {
            transparentDependencyBuffers_.push_back(buffer);
          }
        }
      }
    }
  }

  if (frame.settings->debug.grid) {
    const Format depthFormat =
        hasDepth ? gpu_.getTextureFormat(depthTexture) : Format::Count;
    auto pipelineResult =
        ensureGridPipeline(gpu_.getSwapchainFormat(), depthFormat);
    if (pipelineResult.hasError()) {
      return Result<bool, std::string>::makeError(pipelineResult.error());
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
      gridDrawItem_.depthState = {.compareOp = CompareOp::LessEqual,
                                  .isDepthWriteEnabled = false};
    }
    transparentFixedDraws_.push_back(gridDrawItem_);
  }

  out.sortableDraws = std::span<const TransparentStageSortableDraw>(
      transparentSortableDraws_.data(), transparentSortableDraws_.size());
  out.fixedDraws = std::span<const DrawItem>(transparentFixedDraws_.data(),
                                             transparentFixedDraws_.size());
  out.dependencyBuffers =
      std::span<const BufferHandle>(transparentDependencyBuffers_.data(),
                                    transparentDependencyBuffers_.size());
  out.textureReads = {};
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
