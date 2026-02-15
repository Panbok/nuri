#include "nuri/pch.h"
#include <optional>

#include "nuri/gfx/layers/opaque_layer.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/scene/render_scene.h"

namespace nuri {
namespace {
constexpr float kMinLodRadius = 1.0e-4f;
constexpr uint32_t kMeshDebugColor = 0xffcc5500;

const std::array<VertexAttribute, 3> kMeshVertexAttributes = {
    VertexAttribute{.location = 0,
                    .binding = 0,
                    .offset = offsetof(Vertex, position),
                    .format = VertexFormat::Float3},
    VertexAttribute{.location = 1,
                    .binding = 0,
                    .offset = offsetof(Vertex, normal),
                    .format = VertexFormat::Float3},
    VertexAttribute{.location = 2,
                    .binding = 0,
                    .offset = offsetof(Vertex, uv),
                    .format = VertexFormat::Float2},
};

const std::array<VertexBinding, 1> kMeshVertexBindings = {
    VertexBinding{.stride = sizeof(Vertex)},
};

const RenderSettings &settingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? *frame.settings : kDefaultSettings;
}

VertexInput meshVertexInput() {
  return VertexInput{
      .attributes = kMeshVertexAttributes,
      .bindings = kMeshVertexBindings,
  };
}

RenderPipelineDesc meshPipelineDesc(Format swapchainFormat, Format depthFormat,
                                    ShaderHandle vertexShader,
                                    ShaderHandle fragmentShader,
                                    PolygonMode polygonMode) {
  return RenderPipelineDesc{
      .vertexInput = meshVertexInput(),
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {swapchainFormat},
      .depthFormat = depthFormat,
      .cullMode = CullMode::Back,
      .polygonMode = polygonMode,
      .topology = Topology::Triangle,
      .blendEnabled = false,
  };
}

DrawItem makeBaseMeshDraw(RenderPipelineHandle pipeline,
                          std::string_view debugLabel) {
  DrawItem draw{};
  draw.pipeline = pipeline;
  draw.indexFormat = IndexFormat::U32;
  draw.useDepthState = true;
  draw.depthState = {.compareOp = CompareOp::Less, .isDepthWriteEnabled = true};
  draw.debugLabel = debugLabel;
  draw.debugColor = kMeshDebugColor;
  return draw;
}

float maxAxisScale(const glm::mat4 &transform) {
  const float sx = glm::length(glm::vec3(transform[0]));
  const float sy = glm::length(glm::vec3(transform[1]));
  const float sz = glm::length(glm::vec3(transform[2]));
  return std::max({sx, sy, sz});
}

uint32_t clampRequestedLod(int32_t requestedLod, uint32_t lodCount) {
  if (lodCount == 0) {
    return 0;
  }
  const int32_t maxLod = static_cast<int32_t>(lodCount - 1);
  return static_cast<uint32_t>(std::clamp(requestedLod, 0, maxLod));
}

std::optional<uint32_t> resolveAvailableLod(const Submesh &submesh,
                                            uint32_t desiredLod) {
  const uint32_t lodCount =
      std::clamp(submesh.lodCount, 1u, Submesh::kMaxLodCount);

  uint32_t candidate = std::min(desiredLod, lodCount - 1);
  while (candidate > 0 && submesh.lods[candidate].indexCount == 0) {
    --candidate;
  }
  if (submesh.lods[candidate].indexCount == 0) {
    return std::nullopt;
  }
  return candidate;
}

std::optional<uint32_t> selectAutoLod(const Submesh &submesh,
                                      const glm::mat4 &modelMatrix,
                                      const glm::vec3 &cameraPosition,
                                      std::array<float, 3> thresholds) {
  std::sort(thresholds.begin(), thresholds.end());

  const glm::vec3 localCenter = submesh.bounds.getCenter();
  const float localRadius = 0.5f * glm::length(submesh.bounds.getSize());
  const glm::vec3 worldCenter =
      glm::vec3(modelMatrix * glm::vec4(localCenter, 1.0f));
  const float scale = maxAxisScale(modelMatrix);
  const float worldRadius = std::max(localRadius * scale, kMinLodRadius);

  const float normalizedDistance =
      glm::distance(cameraPosition, worldCenter) / worldRadius;

  uint32_t lodIndex = 0;
  if (normalizedDistance >= thresholds[2]) {
    lodIndex = 3;
  } else if (normalizedDistance >= thresholds[1]) {
    lodIndex = 2;
  } else if (normalizedDistance >= thresholds[0]) {
    lodIndex = 1;
  }

  return resolveAvailableLod(submesh, lodIndex);
}

const SubmeshLod *chooseLodRange(const Submesh *submeshPtr,
                                 const OpaqueRenderable *renderable,
                                 const RenderFrameContext &frame) {
  if (!submeshPtr || !renderable) {
    return nullptr;
  }

  const Submesh &submesh = *submeshPtr;
  const uint32_t lodCount =
      std::clamp(submesh.lodCount, 1u, Submesh::kMaxLodCount);
  const RenderSettings &settings = settingsOrDefault(frame);
  const RenderSettings::OpaqueSettings &opaqueSettings = settings.opaque;

  std::optional<uint32_t> lodIndex;
  if (!opaqueSettings.enableMeshLod) {
    lodIndex = resolveAvailableLod(submesh, 0);
  } else if (opaqueSettings.forcedMeshLod >= 0) {
    const uint32_t requested =
        clampRequestedLod(opaqueSettings.forcedMeshLod, lodCount);
    lodIndex = resolveAvailableLod(submesh, requested);
  } else {
    lodIndex = selectAutoLod(submesh, renderable->modelMatrix,
                             glm::vec3(frame.camera.cameraPos),
                             {opaqueSettings.meshLodDistanceThresholds.x,
                              opaqueSettings.meshLodDistanceThresholds.y,
                              opaqueSettings.meshLodDistanceThresholds.z});
  }

  if (!lodIndex) {
    return nullptr;
  }
  return &submesh.lods[*lodIndex];
}

} // namespace

OpaqueLayer::OpaqueLayer(GPUDevice &gpu)
    : gpu_(gpu), renderableTemplates_(std::pmr::get_default_resource()),
      meshDrawTemplates_(std::pmr::get_default_resource()),
      perFrameEntries_(std::pmr::get_default_resource()),
      drawPushConstants_(std::pmr::get_default_resource()),
      drawItems_(std::pmr::get_default_resource()) {}

OpaqueLayer::~OpaqueLayer() { onDetach(); }

void OpaqueLayer::onAttach() {
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    NURI_LOG_WARNING("OpaqueLayer::onAttach: %s", initResult.error().c_str());
  }
}

void OpaqueLayer::onDetach() {
  destroyPerFrameBuffer();
  destroyDepthTexture();
  resetWireframePipelineState();
  meshPipeline_.reset();
  meshShader_.reset();
  meshVertexShader_ = {};
  meshFragmentShader_ = {};
  meshFillPipelineHandle_ = {};
  baseMeshFillDraw_ = {};
  renderableTemplates_.clear();
  meshDrawTemplates_.clear();
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  initialized_ = false;
}

void OpaqueLayer::onResize(int32_t, int32_t) { destroyDepthTexture(); }

Result<bool, std::string>
OpaqueLayer::buildRenderPasses(RenderFrameContext &frame, RenderPassList &out) {
  NURI_PROFILER_FUNCTION();

  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.opaque.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (!frame.scene) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildRenderPasses: frame scene is null");
  }

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return Result<bool, std::string>::makeError(initResult.error());
  }
  if (!nuri::isValid(depthTexture_)) {
    auto depthResult = recreateDepthTexture();
    if (depthResult.hasError()) {
      return depthResult;
    }
  }
  if (cachedScene_ != frame.scene ||
      cachedTopologyVersion_ != frame.scene->topologyVersion()) {
    auto cacheResult = rebuildSceneCache(*frame.scene);
    if (cacheResult.hasError()) {
      return cacheResult;
    }
  }

  uint32_t cubemapTexId = 0;
  uint32_t hasCubemap = 0;
  if (const Texture *cubemap = frame.scene->environmentCubemap();
      cubemap && cubemap->valid()) {
    cubemapTexId = gpu_.getTextureBindlessIndex(cubemap->handle());
    hasCubemap = 1;
  }

  perFrameEntries_.clear();
  perFrameEntries_.reserve(renderableTemplates_.size());
  for (const RenderableTemplate &templateEntry : renderableTemplates_) {
    const OpaqueRenderable *renderable = templateEntry.renderable;
    if (!renderable || !renderable->albedoTexture ||
        !renderable->albedoTexture->valid()) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildRenderPasses: invalid opaque renderable");
    }

    perFrameEntries_.push_back(PerFrameData{
        .model = renderable->modelMatrix,
        .view = frame.camera.view,
        .proj = frame.camera.proj,
        .cameraPos = frame.camera.cameraPos,
        .albedoTexId =
            gpu_.getTextureBindlessIndex(renderable->albedoTexture->handle()),
        .cubemapTexId = cubemapTexId,
        .hasCubemap = hasCubemap,
    });
  }

  const size_t requiredPerFrameBytes =
      perFrameEntries_.size() * sizeof(PerFrameData);
  auto bufferResult = ensurePerFrameBufferCapacity(requiredPerFrameBytes);
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }

  if (requiredPerFrameBytes > 0) {
    const std::span<const std::byte> perFrameBytes{
        reinterpret_cast<const std::byte *>(perFrameEntries_.data()),
        requiredPerFrameBytes};
    auto updateResult =
        gpu_.updateBuffer(perFrameBuffer_->handle(), perFrameBytes, 0);
    if (updateResult.hasError()) {
      return Result<bool, std::string>::makeError(updateResult.error());
    }
  }

  drawItems_.clear();
  drawPushConstants_.clear();
  DrawItem baseDraw = baseMeshFillDraw_;
  const bool requestedWireframe = settings.opaque.wireframe;
  if (requestedWireframe) {
    auto wireframeResult = ensureWireframePipeline();
    if (wireframeResult.hasError()) {
      return Result<bool, std::string>::makeError(wireframeResult.error());
    }
    if (wireframeResult.value()) {
      baseDraw = baseMeshWireframeDraw_;
    }
  }

  const size_t drawCount = meshDrawTemplates_.size();
  drawItems_.reserve(drawCount);
  drawPushConstants_.reserve(drawCount);
  if (drawCount > 0) {
    const uint64_t baseAddress =
        gpu_.getBufferDeviceAddress(perFrameBuffer_->handle());
    if (baseAddress == 0) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildRenderPasses: invalid per-frame buffer address");
    }

    const uint64_t frameStrideBytes = sizeof(PerFrameData);
    const auto pushConstantBytes = [](const PushConstants &constants) {
      return std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&constants), sizeof(constants));
    };

    NURI_PROFILER_ZONE("OpaqueLayer.select_lod", NURI_PROFILER_COLOR_CMD_DRAW);
    for (const MeshDrawTemplate &templateEntry : meshDrawTemplates_) {
      const SubmeshLod *lodRange = chooseLodRange(
          templateEntry.submesh, templateEntry.renderable, frame);
      if (!lodRange) {
        continue;
      }

      const uint64_t perFrameAddress =
          baseAddress + frameStrideBytes * templateEntry.perFrameIndex;
      drawPushConstants_.push_back(
          PushConstants{.perFrameAddress = perFrameAddress});

      DrawItem meshDraw = baseDraw;
      meshDraw.vertexBuffer = templateEntry.vertexBuffer;
      meshDraw.indexBuffer = templateEntry.indexBuffer;
      meshDraw.vertexCount = templateEntry.vertexCount;
      meshDraw.indexCount = lodRange->indexCount;
      meshDraw.firstIndex = lodRange->indexOffset;
      meshDraw.pushConstants = pushConstantBytes(drawPushConstants_.back());
      drawItems_.push_back(meshDraw);
    }
    NURI_PROFILER_ZONE_END();
  }

  const bool shouldLoadColor = settings.skybox.enabled;

  RenderPass pass{};
  pass.color = {.loadOp = shouldLoadColor ? LoadOp::Load : LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}};
  pass.depth = {.loadOp = LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearDepth = 1.0f,
                .clearStencil = 0};
  pass.depthTexture = depthTexture_;
  pass.draws = std::span<const DrawItem>(drawItems_.data(), drawItems_.size());
  pass.debugLabel = "Opaque Pass";
  pass.debugColor = 0xff0000ff;

  frame.sharedDepthTexture = depthTexture_;
  out.push_back(pass);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::ensureInitialized() {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto shaderResult = createShaders();
  if (shaderResult.hasError()) {
    return shaderResult;
  }

  auto depthResult = recreateDepthTexture();
  if (depthResult.hasError()) {
    meshShader_.reset();
    meshVertexShader_ = {};
    meshFragmentShader_ = {};
    return depthResult;
  }

  auto pipelineResult = createPipelines();
  if (pipelineResult.hasError()) {
    meshShader_.reset();
    meshVertexShader_ = {};
    meshFragmentShader_ = {};
    meshFillPipelineHandle_ = {};
    destroyDepthTexture();
    return pipelineResult;
  }

  auto bufferResult = ensurePerFrameBufferCapacity(sizeof(PerFrameData));
  if (bufferResult.hasError()) {
    meshShader_.reset();
    meshPipeline_.reset();
    meshVertexShader_ = {};
    meshFragmentShader_ = {};
    meshFillPipelineHandle_ = {};
    baseMeshFillDraw_ = {};
    baseMeshWireframeDraw_ = {};
    destroyDepthTexture();
    return bufferResult;
  }

  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::recreateDepthTexture() {
  if (nuri::isValid(depthTexture_)) {
    gpu_.destroyTexture(depthTexture_);
    depthTexture_ = TextureHandle{};
  }

  auto depthResult = gpu_.createDepthBuffer();
  if (depthResult.hasError()) {
    return Result<bool, std::string>::makeError(depthResult.error());
  }
  depthTexture_ = depthResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueLayer::ensurePerFrameBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(PerFrameData));
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
      Buffer::create(gpu_, perFrameDesc, "opaque_per_frame_buffer");
  if (perFrameResult.hasError()) {
    return Result<bool, std::string>::makeError(perFrameResult.error());
  }

  perFrameBuffer_ = std::move(perFrameResult.value());
  perFrameBufferCapacityBytes_ = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueLayer::rebuildSceneCache(const RenderScene &scene) {
  renderableTemplates_.clear();
  meshDrawTemplates_.clear();

  const std::span<const OpaqueRenderable> renderables =
      scene.opaqueRenderables();
  if (renderables.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::rebuildSceneCache: renderables count exceeds UINT32_MAX");
  }
  renderableTemplates_.reserve(renderables.size());

  size_t totalMeshDraws = 0;
  for (const OpaqueRenderable &renderable : renderables) {
    if (!renderable.model) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::rebuildSceneCache: renderable model is null");
    }
    totalMeshDraws += renderable.model->submeshes().size();
  }
  meshDrawTemplates_.reserve(totalMeshDraws);

  for (uint32_t index = 0; index < static_cast<uint32_t>(renderables.size());
       ++index) {
    const OpaqueRenderable &renderable = renderables[index];
    const uint32_t perFrameIndex = index;
    renderableTemplates_.push_back(
        RenderableTemplate{.renderable = &renderable});

    const std::span<const Submesh> submeshes = renderable.model->submeshes();
    for (size_t submeshIndex = 0; submeshIndex < submeshes.size();
         ++submeshIndex) {
      meshDrawTemplates_.push_back(MeshDrawTemplate{
          .renderable = &renderable,
          .submesh = &submeshes[submeshIndex],
          .perFrameIndex = perFrameIndex,
          .vertexBuffer = renderable.model->vertexBuffer()->handle(),
          .indexBuffer = renderable.model->indexBuffer()->handle(),
          .vertexCount = renderable.model->vertexCount(),
      });
    }
  }

  cachedScene_ = &scene;
  cachedTopologyVersion_ = scene.topologyVersion();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::createShaders() {
  meshShader_ = Shader::create("main", gpu_);
  if (!meshShader_) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::createShaders: failed to create meshShader_");
  }
  struct ShaderSpec {
    Shader *shader = nullptr;
    std::string_view path{};
    ShaderStage stage = ShaderStage::Vertex;
    ShaderHandle *outHandle = nullptr;
  };
  const std::array<ShaderSpec, 2> shaderSpecs = {
      ShaderSpec{meshShader_.get(), "assets/shaders/main.vert",
                 ShaderStage::Vertex, &meshVertexShader_},
      ShaderSpec{meshShader_.get(), "assets/shaders/main.frag",
                 ShaderStage::Fragment, &meshFragmentShader_},
  };

  for (const ShaderSpec &spec : shaderSpecs) {
    if (!spec.shader || !spec.outHandle) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::createShaders: invalid shader spec");
    }
    auto compileResult = spec.shader->compileFromFile(spec.path, spec.stage);
    if (compileResult.hasError()) {
      return Result<bool, std::string>::makeError(compileResult.error());
    }
    *spec.outHandle = compileResult.value();
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::createPipelines() {
  meshPipeline_ = Pipeline::create(gpu_);
  const Format depthFormat = nuri::isValid(depthTexture_)
                                 ? gpu_.getTextureFormat(depthTexture_)
                                 : Format::D32_FLOAT;
  const RenderPipelineDesc meshDesc = meshPipelineDesc(
      gpu_.getSwapchainFormat(), depthFormat, meshVertexShader_,
      meshFragmentShader_, PolygonMode::Fill);
  if (!meshPipeline_) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::createPipelines: failed to create mesh pipeline");
  }

  auto pipelineResult =
      meshPipeline_->createRenderPipeline(meshDesc, "opaque_mesh");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  meshFillPipelineHandle_ = meshPipeline_->getRenderPipeline();

  baseMeshFillDraw_ = makeBaseMeshDraw(meshFillPipelineHandle_, "OpaqueMesh");
  resetWireframePipelineState();

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::ensureWireframePipeline() {
  if (wireframePipelineInitialized_ &&
      nuri::isValid(meshWireframePipelineHandle_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (wireframePipelineUnsupported_) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(meshFillPipelineHandle_)) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::ensureWireframePipeline: fill pipeline is invalid");
  }

  const Format depthFormat = nuri::isValid(depthTexture_)
                                 ? gpu_.getTextureFormat(depthTexture_)
                                 : Format::D32_FLOAT;
  const RenderPipelineDesc wireframeDesc = meshPipelineDesc(
      gpu_.getSwapchainFormat(), depthFormat, meshVertexShader_,
      meshFragmentShader_, PolygonMode::Line);

  auto pipelineResult =
      gpu_.createRenderPipeline(wireframeDesc, "opaque_mesh_wireframe");
  if (pipelineResult.hasError()) {
    wireframePipelineUnsupported_ = true;
    NURI_LOG_WARNING("OpaqueLayer::ensureWireframePipeline: %s",
                     pipelineResult.error().c_str());
    return Result<bool, std::string>::makeResult(false);
  }

  meshWireframePipelineHandle_ = pipelineResult.value();
  wireframePipelineInitialized_ = true;

  baseMeshWireframeDraw_ = baseMeshFillDraw_;
  baseMeshWireframeDraw_.pipeline = meshWireframePipelineHandle_;
  baseMeshWireframeDraw_.debugLabel = "OpaqueMeshWireframe";

  return Result<bool, std::string>::makeResult(true);
}

void OpaqueLayer::resetWireframePipelineState() {
  if (nuri::isValid(meshWireframePipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshWireframePipelineHandle_);
  }
  meshWireframePipelineHandle_ = {};
  wireframePipelineInitialized_ = false;
  wireframePipelineUnsupported_ = false;
  baseMeshWireframeDraw_ = {};
}

void OpaqueLayer::destroyDepthTexture() {
  if (nuri::isValid(depthTexture_)) {
    gpu_.destroyTexture(depthTexture_);
    depthTexture_ = TextureHandle{};
  }
}

void OpaqueLayer::destroyPerFrameBuffer() {
  if (perFrameBuffer_ && perFrameBuffer_->valid()) {
    gpu_.destroyBuffer(perFrameBuffer_->handle());
  }
  perFrameBuffer_.reset();
  perFrameBufferCapacityBytes_ = 0;
}

} // namespace nuri
