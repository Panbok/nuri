#include "nuri/gfx/renderers/debug_renderer.h"

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
constexpr uint32_t kLightIconPassDebugColor = 0xfff0c040;
constexpr uint32_t kGridVertexCount = 6;
constexpr std::string_view kGridPipelineName = "debug_grid";
constexpr std::string_view kGridPassLabel = "DebugGrid Pass";
constexpr std::string_view kGridDrawLabel = "DebugGrid Draw";
const glm::vec4 kDirectionalLightIconColor(1.0f, 0.88f, 0.25f, 1.0f);
const glm::vec4 kPointLightIconColor(0.35f, 0.82f, 1.0f, 1.0f);
const glm::vec4 kSpotLightIconColor(1.0f, 0.55f, 0.24f, 1.0f);
const glm::vec4 kSelectedLightIconColor(1.0f, 0.28f, 0.16f, 1.0f);

[[nodiscard]] bool isSameTextureHandle(TextureHandle a, TextureHandle b) {
  return a.index == b.index && a.generation == b.generation;
}

[[nodiscard]] glm::vec3 safeNormalize(const glm::vec3 &value,
                                      const glm::vec3 &fallback) {
  const float length = glm::length(value);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return fallback;
  }
  return value / length;
}

[[nodiscard]] float lightIconScale(const CameraFrameState &camera,
                                   const glm::vec3 &position) {
  const float distance = glm::length(glm::vec3(camera.cameraPos) - position);
  return std::clamp(distance * 0.08f, 0.2f, 3.0f);
}

void buildLightBasis(const glm::vec3 &direction, glm::vec3 &outRight,
                     glm::vec3 &outUp) {
  const glm::vec3 dir = safeNormalize(direction, glm::vec3(0.0f, 0.0f, -1.0f));
  const glm::vec3 fallbackUp = std::abs(dir.y) > 0.95f
                                   ? glm::vec3(1.0f, 0.0f, 0.0f)
                                   : glm::vec3(0.0f, 1.0f, 0.0f);
  outRight =
      safeNormalize(glm::cross(fallbackUp, dir), glm::vec3(1.0f, 0.0f, 0.0f));
  outUp = safeNormalize(glm::cross(dir, outRight), glm::vec3(0.0f, 1.0f, 0.0f));
}

void drawDirectionalLightIcon(DebugDraw3D &debugDraw, const glm::vec3 &position,
                              const glm::vec3 &direction, float scale,
                              const glm::vec4 &color) {
  const glm::vec3 dir = safeNormalize(direction, glm::vec3(0.0f, 0.0f, -1.0f));
  glm::vec3 right(1.0f, 0.0f, 0.0f);
  glm::vec3 up(0.0f, 1.0f, 0.0f);
  buildLightBasis(dir, right, up);

  const glm::vec3 end = position + dir * (scale * 2.4f);
  const glm::vec3 headBase = end - dir * (scale * 0.8f);
  debugDraw.line(position, end, color);
  debugDraw.line(end, headBase + right * (scale * 0.45f), color);
  debugDraw.line(end, headBase - right * (scale * 0.45f), color);
  debugDraw.line(end, headBase + up * (scale * 0.45f), color);
  debugDraw.line(end, headBase - up * (scale * 0.45f), color);
}

void drawPointLightIcon(DebugDraw3D &debugDraw, const glm::vec3 &position,
                        float scale, const glm::vec4 &color) {
  const glm::vec3 x(scale, 0.0f, 0.0f);
  const glm::vec3 y(0.0f, scale, 0.0f);
  const glm::vec3 z(0.0f, 0.0f, scale);
  debugDraw.line(position - x, position + x, color);
  debugDraw.line(position - y, position + y, color);
  debugDraw.line(position - z, position + z, color);
  debugDraw.line(position + x, position + y, color);
  debugDraw.line(position + y, position - x, color);
  debugDraw.line(position - x, position - y, color);
  debugDraw.line(position - y, position + x, color);
}

void drawSpotLightIcon(DebugDraw3D &debugDraw, const glm::vec3 &position,
                       const glm::vec3 &direction, float scale,
                       const glm::vec4 &color) {
  const glm::vec3 dir = safeNormalize(direction, glm::vec3(0.0f, 0.0f, -1.0f));
  glm::vec3 right(1.0f, 0.0f, 0.0f);
  glm::vec3 up(0.0f, 1.0f, 0.0f);
  buildLightBasis(dir, right, up);

  const glm::vec3 baseCenter = position + dir * (scale * 2.2f);
  const float radius = scale * 0.9f;
  const glm::vec3 p0 = baseCenter + right * radius;
  const glm::vec3 p1 = baseCenter + up * radius;
  const glm::vec3 p2 = baseCenter - right * radius;
  const glm::vec3 p3 = baseCenter - up * radius;
  debugDraw.line(position, p0, color);
  debugDraw.line(position, p1, color);
  debugDraw.line(position, p2, color);
  debugDraw.line(position, p3, color);
  debugDraw.line(p0, p1, color);
  debugDraw.line(p1, p2, color);
  debugDraw.line(p2, p3, color);
  debugDraw.line(p3, p0, color);
}

} // namespace

DebugRenderer::DebugRenderer(GPUDevice &gpu, DebugRendererConfig config,
                             std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      debugDraw3D_(std::make_unique<DebugDraw3D>(gpu, memory_)),
      transparentSortableDraws_(memory_), transparentFixedDraws_(memory_),
      transparentDependencyBuffers_(memory_) {}

DebugRenderer::~DebugRenderer() { onDetach(); }

void DebugRenderer::onDetach() {
  debugDraw3D_.reset();
  resetGridState();
  preparedSceneDepthTexture_ = {};
  preparedFrameColorTexture_ = {};
  preparedSceneDepthGraphTexture_ = {};
  preparedHasPriorColorPass_ = false;
  preparedGridPass_ = false;
  preparedSceneOverlayPass_ = false;
  transparentSortableDraws_.clear();
  transparentFixedDraws_.clear();
  transparentDependencyBuffers_.clear();
}

Result<bool, std::string> DebugRenderer::ensureGridInitialized() {
  return createGridShaders();
}

Result<bool, std::string> DebugRenderer::createGridShaders() {
  if (gridShader_ && nuri::isValid(gridVertexShader_) &&
      nuri::isValid(gridFragmentShader_)) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (config_.vertex.empty() || config_.fragment.empty()) {
    return Result<bool, std::string>::makeError(
        "DebugRenderer::createGridShaders: vertex or fragment shader path is "
        "empty");
  }

  gridShader_ = Shader::create("debug_grid", gpu_);
  if (!gridShader_) {
    return Result<bool, std::string>::makeError(
        "DebugRenderer::createGridShaders: failed to create grid shader "
        "wrapper");
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

Result<bool, std::string>
DebugRenderer::ensureGridPipeline(Format colorFormat, Format depthFormat) {
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
        "DebugRenderer::ensureGridPipeline: failed to create grid pipeline "
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

Result<bool, std::string>
DebugRenderer::prepareGridDraw(const RenderFrameContext &frame,
                               TextureHandle depthTexture) {
  const bool hasDepth = nuri::isValid(depthTexture);
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

  return Result<bool, std::string>::makeResult(true);
}

void DebugRenderer::resetGridState() {
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

Result<bool, std::string> DebugRenderer::appendModelBoundsGraphPass(
    const RenderFrameContext &frame, RenderGraphBuilder &graph,
    TextureHandle sceneDepthTexture,
    RenderGraphTextureId sceneDepthGraphTexture) {
  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  if (!debugDraw3D_ || !frame.scene || !nuri::isValid(depthTexture) ||
      !frame.resources) {
    return Result<bool, std::string>::makeResult(true);
  }

  const std::span<const Renderable> renderables = frame.scene->renderables();
  if (renderables.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  debugDraw3D_->clear();
  debugDraw3D_->setMatrix(frame.camera.proj * frame.camera.view);
  for (const Renderable &renderable : renderables) {
    const ModelRecord *modelRecord = frame.resources->tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      continue;
    }
    debugDraw3D_->box(renderable.modelMatrix, modelRecord->model->bounds(),
                      glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
  }

  auto linePassResult =
      debugDraw3D_->buildGraphPass(frame.frameIndex, depthTexture);
  if (linePassResult.hasError()) {
    return Result<bool, std::string>::makeError(linePassResult.error());
  }

  DebugDraw3D::PreparedGraphPass pass = linePassResult.value();
  TextureHandle colorTexture = resolveFrameColorTexture(frame);
  if (!nuri::isValid(colorTexture)) {
    colorTexture = pass.colorTextureHandle;
  }
  if (nuri::isValid(colorTexture)) {
    auto colorImportResult =
        graph.importTexture(colorTexture, "debug_pass_color_texture");
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

bool DebugRenderer::hasDebugWork(const RenderFrameContext &frame) const {
  if (frame.settings == nullptr) {
    return false;
  }
  const RenderSettings::DebugSettings &debug = frame.settings->debug;
  return debug.enabled || debug.modelBounds || debug.grid || debug.lightIcons;
}

Result<bool, std::string>
DebugRenderer::buildSceneDebugLines(const RenderFrameContext &frame,
                                    TextureHandle depthTexture,
                                    float &outSortDepth) {
  outSortDepth = 0.0f;
  if (!debugDraw3D_ || !frame.scene || !nuri::isValid(depthTexture)) {
    static uint32_t loggedSceneDebugEarlyOuts = 0u;
    if (loggedSceneDebugEarlyOuts < 16u) {
      NURI_LOG_WARNING(
          "DebugRenderer::buildSceneDebugLines early-out: frame=%llu "
          "debugDraw=%u scene=%u depthValid=%u",
          static_cast<unsigned long long>(frame.frameIndex),
          debugDraw3D_ != nullptr ? 1u : 0u, frame.scene != nullptr ? 1u : 0u,
          nuri::isValid(depthTexture) ? 1u : 0u);
      ++loggedSceneDebugEarlyOuts;
    }
    return Result<bool, std::string>::makeResult(false);
  }

  debugDraw3D_->clear();
  debugDraw3D_->setMatrix(frame.camera.proj * frame.camera.view);

  const LightId selectedLightId = resolveSelectedLightId(frame);
  bool hasLines = false;
  const glm::mat4 view = frame.camera.view;

  if (frame.settings != nullptr && frame.settings->debug.modelBounds &&
      frame.resources != nullptr) {
    const std::span<const Renderable> renderables = frame.scene->renderables();
    for (const Renderable &renderable : renderables) {
      const ModelRecord *modelRecord =
          frame.resources->tryGet(renderable.model);
      if (!modelRecord || !modelRecord->model) {
        continue;
      }
      debugDraw3D_->box(renderable.modelMatrix, modelRecord->model->bounds(),
                        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
      const glm::vec3 center =
          glm::vec3(renderable.modelMatrix *
                    glm::vec4(modelRecord->model->bounds().getCenter(), 1.0f));
      outSortDepth =
          std::max(outSortDepth, -(view * glm::vec4(center, 1.0f)).z);
      hasLines = true;
    }
  }

  if (frame.settings != nullptr && frame.settings->debug.lightIcons) {
    frame.scene->forEachLightId([&](LightId lightId) {
      LightDesc light{};
      if (!frame.scene->graph().getCachedLightWorldDesc(lightId, light)) {
        return;
      }

      const float scale = lightIconScale(frame.camera, light.position);
      const bool selected =
          isValid(selectedLightId) && selectedLightId == lightId;
      const glm::vec4 color =
          selected
              ? kSelectedLightIconColor
              : (light.type == LightType::Directional
                     ? kDirectionalLightIconColor
                     : (light.type == LightType::Point ? kPointLightIconColor
                                                       : kSpotLightIconColor));
      const glm::vec3 direction = light.rotation * glm::vec3(0.0f, 0.0f, -1.0f);
      switch (light.type) {
      case LightType::Directional:
        drawDirectionalLightIcon(*debugDraw3D_, light.position, direction,
                                 scale, color);
        break;
      case LightType::Point:
        drawPointLightIcon(*debugDraw3D_, light.position, scale, color);
        break;
      case LightType::Spot:
        drawSpotLightIcon(*debugDraw3D_, light.position, direction, scale,
                          color);
        break;
      }
      outSortDepth =
          std::max(outSortDepth, -(view * glm::vec4(light.position, 1.0f)).z);
      hasLines = true;
    });
  }

  return Result<bool, std::string>::makeResult(hasLines);
}

Result<bool, std::string>
DebugRenderer::prepareDebugPasses(RenderFrameContext &frame) {
  preparedSceneDepthTexture_ = {};
  preparedFrameColorTexture_ = {};
  preparedSceneDepthGraphTexture_ = {};
  preparedHasPriorColorPass_ = false;
  preparedGridPass_ = false;
  preparedSceneOverlayPass_ = false;

  if (frame.sharedResources.transparentStageEnabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (!hasDebugWork(frame)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const TextureHandle sceneDepthTexture = resolveFrameDepthTexture(frame);
  preparedSceneDepthGraphTexture_ = resolveSceneDepthGraphTexture(frame);
  TextureHandle frameColorTexture = resolveFrameColorTexture(frame);
  preparedSceneDepthTexture_ = sceneDepthTexture;
  preparedFrameColorTexture_ = frameColorTexture;
  preparedHasPriorColorPass_ = nuri::isValid(frameColorTexture);

  if (frame.settings->debug.grid) {
    auto gridResult = prepareGridDraw(frame, sceneDepthTexture);
    if (gridResult.hasError()) {
      return gridResult;
    }
    preparedGridPass_ = true;
  }

  float debugSortDepth = 0.0f;
  auto buildLinesResult =
      buildSceneDebugLines(frame, sceneDepthTexture, debugSortDepth);
  if (buildLinesResult.hasError()) {
    return buildLinesResult;
  }
  preparedSceneOverlayPass_ = buildLinesResult.value();

  return Result<bool, std::string>::makeResult(true);
}

bool DebugRenderer::hasPreparedDebugGridPass() const noexcept {
  return preparedGridPass_;
}

bool DebugRenderer::hasPreparedDebugSceneOverlayPass() const noexcept {
  return preparedSceneOverlayPass_;
}

Result<bool, std::string>
DebugRenderer::appendDebugGridPass(RenderFrameContext &frame,
                                   RenderGraphBuilder &graph) {
  if (!preparedGridPass_) {
    return Result<bool, std::string>::makeResult(true);
  }

  const bool hasDepth = nuri::isValid(preparedSceneDepthTexture_);
  RenderGraphTextureId depthTextureId{};
  if (hasDepth) {
    if (nuri::isValid(preparedSceneDepthGraphTexture_)) {
      depthTextureId = preparedSceneDepthGraphTexture_;
    } else {
      auto importResult = graph.importTexture(preparedSceneDepthTexture_,
                                              "debug_depth_texture");
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }
      depthTextureId = importResult.value();
    }
  }

  RenderGraphGraphicsPassDesc gridPass{};
  gridPass.color = {.loadOp = preparedHasPriorColorPass_ ? LoadOp::Load
                                                         : LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}};
  if (nuri::isValid(preparedFrameColorTexture_)) {
    auto colorImportResult = graph.importTexture(preparedFrameColorTexture_,
                                                 "debug_grid_color_texture");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    gridPass.colorTexture = colorImportResult.value();
  }
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
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugRenderer::appendDebugSceneOverlayPass(RenderFrameContext &frame,
                                           RenderGraphBuilder &graph) {
  if (!preparedSceneOverlayPass_) {
    return Result<bool, std::string>::makeResult(true);
  }

  const Format overlayColorFormat =
      nuri::isValid(preparedFrameColorTexture_)
          ? gpu_.getTextureFormat(preparedFrameColorTexture_)
          : Format::Count;
  auto linePassResult = debugDraw3D_->buildGraphPass(
      frame.frameIndex, preparedSceneDepthTexture_, overlayColorFormat);
  if (linePassResult.hasError()) {
    return Result<bool, std::string>::makeError(linePassResult.error());
  }

  DebugDraw3D::PreparedGraphPass pass = linePassResult.value();
  TextureHandle colorTexture = preparedFrameColorTexture_;
  if (!nuri::isValid(colorTexture)) {
    colorTexture = pass.colorTextureHandle;
  }
  if (nuri::isValid(colorTexture)) {
    auto colorImportResult =
        graph.importTexture(colorTexture, "debug_pass_color_texture");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    pass.desc.colorTexture = colorImportResult.value();
  }
  if (nuri::isValid(pass.depthTextureHandle)) {
    const bool useDepthOverride =
        nuri::isValid(preparedSceneDepthTexture_) &&
        nuri::isValid(preparedSceneDepthGraphTexture_) &&
        isSameTextureHandle(pass.depthTextureHandle,
                            preparedSceneDepthTexture_);
    if (useDepthOverride) {
      pass.desc.depthTexture = preparedSceneDepthGraphTexture_;
    } else {
      auto depthImportResult = graph.importTexture(pass.depthTextureHandle,
                                                   "debug_pass_depth_texture");
      if (depthImportResult.hasError()) {
        return Result<bool, std::string>::makeError(depthImportResult.error());
      }
      pass.desc.depthTexture = depthImportResult.value();
    }
  }
  pass.desc.debugColor = kLightIconPassDebugColor;

  auto addResult = graph.addGraphicsPass(pass.desc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> DebugRenderer::buildTransparentStageContribution(
    RenderFrameContext &frame, TransparentStageContribution &out) {
  NURI_PROFILER_FUNCTION();
  out = {};
  transparentSortableDraws_.clear();
  transparentFixedDraws_.clear();
  transparentDependencyBuffers_.clear();

  static uint32_t loggedTransparentDebugBuilds = 0u;
  if (loggedTransparentDebugBuilds < 16u) {
    NURI_LOG_WARNING(
        "DebugRenderer::buildTransparentStageContribution: frame=%llu "
        "hasDebugWork=%u transparentStageEnabled=%u",
        static_cast<unsigned long long>(frame.frameIndex),
        hasDebugWork(frame) ? 1u : 0u,
        frame.sharedResources.transparentStageEnabled ? 1u : 0u);
  }

  if (!hasDebugWork(frame)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  if (loggedTransparentDebugBuilds < 16u) {
    NURI_LOG_WARNING(
        "DebugRenderer::buildTransparentStageContribution depth: frame=%llu "
        "depthValid=%u",
        static_cast<unsigned long long>(frame.frameIndex),
        nuri::isValid(depthTexture) ? 1u : 0u);
  }
  if (nuri::isValid(depthTexture)) {
    float debugSortDepth = 0.0f;
    auto buildLinesResult =
        buildSceneDebugLines(frame, depthTexture, debugSortDepth);
    if (buildLinesResult.hasError()) {
      return Result<bool, std::string>::makeError(buildLinesResult.error());
    }
    if (loggedTransparentDebugBuilds < 16u) {
      NURI_LOG_WARNING(
          "DebugRenderer::buildTransparentStageContribution lines: frame=%llu "
          "built=%u sortDepth=%.5f",
          static_cast<unsigned long long>(frame.frameIndex),
          buildLinesResult.value() ? 1u : 0u, debugSortDepth);
    }
    if (buildLinesResult.value()) {
      const TextureHandle frameColor = resolveFrameColorTexture(frame);
      const Format targetColorFormat = nuri::isValid(frameColor)
                                           ? gpu_.getTextureFormat(frameColor)
                                           : gpu_.getSwapchainFormat();
      auto linePassResult = debugDraw3D_->buildGraphPass(
          frame.frameIndex, depthTexture, targetColorFormat);
      if (linePassResult.hasError()) {
        return Result<bool, std::string>::makeError(linePassResult.error());
      }
      const DebugDraw3D::PreparedGraphPass pass = linePassResult.value();
      if (loggedTransparentDebugBuilds < 16u) {
        NURI_LOG_WARNING(
            "DebugRenderer::buildTransparentStageContribution pass: frame=%llu "
            "draws=%zu deps=%zu depthHandleValid=%u",
            static_cast<unsigned long long>(frame.frameIndex),
            pass.desc.draws.size(), pass.desc.dependencyBuffers.size(),
            nuri::isValid(pass.depthTextureHandle) ? 1u : 0u);
      }
      if (!pass.desc.draws.empty()) {
        for (size_t i = 0; i < pass.desc.draws.size(); ++i) {
          transparentSortableDraws_.push_back(TransparentStageSortableDraw{
              .draw = pass.desc.draws[i],
              .sortDepth = debugSortDepth,
              .stableOrder = static_cast<uint32_t>(i),
          });
        }
      }
      for (const BufferHandle buffer : pass.desc.dependencyBuffers) {
        if (nuri::isValid(buffer)) {
          transparentDependencyBuffers_.push_back(buffer);
        }
      }
    }
  }

  if (frame.settings->debug.grid) {
    auto gridResult = prepareGridDraw(frame, depthTexture);
    if (gridResult.hasError()) {
      return gridResult;
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
  if (loggedTransparentDebugBuilds < 16u) {
    NURI_LOG_WARNING(
        "DebugRenderer::buildTransparentStageContribution out: frame=%llu "
        "sortable=%zu fixed=%zu deps=%zu",
        static_cast<unsigned long long>(frame.frameIndex),
        out.sortableDraws.size(), out.fixedDraws.size(),
        out.dependencyBuffers.size());
    ++loggedTransparentDebugBuilds;
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
