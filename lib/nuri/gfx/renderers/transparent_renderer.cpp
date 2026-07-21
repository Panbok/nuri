#include "nuri/gfx/renderers/transparent_renderer.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderers/detail/animation_rendering.h"
#include "nuri/gfx/renderers/detail/forward_rendering.h"
#include "nuri/gfx/shader.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_manager.h"
namespace nuri {
namespace {
constexpr uint32_t kTransparentPassDebugColor = 0x66aaffffu;
constexpr uint32_t kTransparentPickPassDebugColor = 0x66ff88ffu;
constexpr uint32_t kTransparentMeshDebugColor = 0x66aaffffu;
constexpr std::string_view kTransparentPassLabel = "Transparent Pass";
constexpr std::string_view kTransparentTransmissionPassLabel =
    "Transparent Transmission Pass";
constexpr std::string_view kTransparentPickPassLabel = "Transparent Pick Pass";
constexpr std::string_view kTransparentMeshLabel = "TransparentMesh";
constexpr std::string_view kTransparentMeshPickLabel = "TransparentMeshPick";
[[nodiscard]] std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}
[[nodiscard]] float transparentSortDepth(const glm::mat4 &view,
                                         const glm::mat4 &model,
                                         const BoundingBox &bounds) {
  const glm::vec3 min = bounds.min_;
  const glm::vec3 max = bounds.max_;
  const glm::vec3 corners[] = {
      glm::vec3(min.x, min.y, min.z), glm::vec3(min.x, max.y, min.z),
      glm::vec3(min.x, min.y, max.z), glm::vec3(min.x, max.y, max.z),
      glm::vec3(max.x, min.y, min.z), glm::vec3(max.x, max.y, min.z),
      glm::vec3(max.x, min.y, max.z), glm::vec3(max.x, max.y, max.z),
  };
  float depth = std::numeric_limits<float>::lowest();
  for (const glm::vec3 &corner : corners) {
    const glm::vec4 viewPos = view * model * glm::vec4(corner, 1.0f);
    depth = std::max(depth, -viewPos.z);
  }
  return std::isfinite(depth) ? depth : 0.0f;
}
void appendPreResolvedDrawBuffers(std::pmr::vector<BufferHandle> &handles,
                                  std::span<const DrawItem> draws) {
  for (const DrawItem &draw : draws) {
    appendUniqueForwardHandle(handles, draw.vertexBuffer);
    appendUniqueForwardHandle(handles, draw.indexBuffer);
    appendUniqueForwardHandle(handles, draw.indirectBuffer);
    appendUniqueForwardHandle(handles, draw.indirectCountBuffer);
  }
}
RenderPipelineDesc meshPipelineDesc(Format colorFormat, Format depthFormat,
                                    ShaderHandle vertexShader,
                                    ShaderHandle fragmentShader,
                                    bool blendEnabled, CullMode cullMode) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {colorFormat},
      .depthFormat = depthFormat,
      .cullMode = cullMode,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = blendEnabled,
      .rasterState = depthFormat != Format::Count
                         ? makeRasterPipelineState(
                               DepthState{.compareOp = CompareOp::LessEqual,
                                          .isDepthWriteEnabled = false})
                         : RasterPipelineState{},
  };
}
[[nodiscard]] const RenderSettings &
settingsOrDefault(const RenderFrameContext &frame) {
  return renderSettingsOrDefault(frame);
}
[[nodiscard]] bool
isAntiAliasingDebugOutputView(AntiAliasingDebugView view) noexcept {
  return view != AntiAliasingDebugView::None &&
         view != AntiAliasingDebugView::Settings;
}
void applyContributorDependencies(DrawItem &draw,
                                  std::span<const BufferHandle> dependencies) {
  if (!nuri::isValid(draw.vertexBuffer) && !dependencies.empty()) {
    draw.vertexBuffer = dependencies[0];
    draw.vertexBufferOffset = 0;
  }
  if (draw.command == DrawCommandType::Direct) {
    if (!nuri::isValid(draw.indirectBuffer) && dependencies.size() > 1u) {
      draw.indirectBuffer = dependencies[1];
      draw.indirectBufferOffset = 0;
    }
    if (!nuri::isValid(draw.indirectCountBuffer) && dependencies.size() > 2u) {
      draw.indirectCountBuffer = dependencies[2];
      draw.indirectCountBufferOffset = 0;
    }
  }
}
} // namespace

TransparentRenderer::TransparentRenderer(GPUDevice &gpu,
                                         TransparentRendererConfig config,
                                         std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(resolveMemoryResource(memory)), instanceMatricesRing_(memory_),
      instanceRemapRing_(memory_), meshDrawTemplates_(memory_),
      instanceMatrices_(memory_), instanceRemap_(memory_),
      instanceDataRingUploadVersions_(memory_),
      materialTextureAccessHandles_(memory_),
      environmentTextureAccessHandles_(memory_),
      contributorSortableDraws_(memory_), contributorFixedDraws_(memory_),
      contributorTextureReads_(memory_), contributorDependencyBuffers_(memory_),
      drawPushConstants_(memory_), pickPushConstants_(memory_),
      meshSortableDraws_(memory_), sortableDraws_(memory_),
      fixedDraws_(memory_), passDrawItems_(memory_),
      transparentRunDrawItems_(memory_),
      transparentRunDependencyBuffers_(memory_),
      transparentCandidateDependencyBuffers_(memory_), pickDrawItems_(memory_),
      passTextureReads_(memory_), passDependencyBuffers_(memory_) {
  const std::filesystem::path basePath =
      !config_.pickFragment.empty() ? config_.pickFragment.parent_path()
                                    : config_.meshFragment.parent_path();
  alphaPickFragmentPath_ = basePath / "main_id_alpha.frag";
}

TransparentRenderer::~TransparentRenderer() { onDetach(); }

void TransparentRenderer::onAttach() {
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    NURI_LOG_WARNING("TransparentRenderer::onAttach: %s",
                     initResult.error().c_str());
  }
}

void TransparentRenderer::onDetach() {
  destroyBuffers();
  destroyPipelineState();
  destroyShaders();
  resetFrameBuildState();
  resetCachedState();
  initialized_ = false;
}

void TransparentRenderer::publishFrameData(RenderFrameContext &frame) {
  frame.sharedResources.transparentStageEnabled = true;
}

Result<bool, std::string>
TransparentRenderer::prepareTransparentPasses(RenderFrameContext &frame) {
  NURI_PROFILER_FUNCTION();
  frame.metrics.transparent = {};
  resetFrameBuildState();
  const RenderSettings &settings = settingsOrDefault(frame);
  auto contributorResult = collectContributorDraws(frame);
  if (contributorResult.hasError())
    return contributorResult;
  const auto publishContributors = [&] {
    sortableDraws_.assign(contributorSortableDraws_.begin(),
                          contributorSortableDraws_.end());
    fixedDraws_.assign(contributorFixedDraws_.begin(),
                       contributorFixedDraws_.end());
    passTextureReads_.clear();
    passDependencyBuffers_.clear();
    for (TextureHandle handle : contributorTextureReads_)
      appendUniqueForwardHandle(passTextureReads_, handle);
    sortTransparentDraws(sortableDraws_);
    frame.metrics.transparent.meshDraws = 0u;
    frame.metrics.transparent.pickDraws = 0u;
    return Result<bool, std::string>::makeResult(true);
  };
  if (!settings.transparent.enabled)
    return publishContributors();
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return initResult;
  }
  const MaterialTableSnapshot materialSnapshot =
      frame.resources->materialSnapshot();
  const bool topologyDirty =
      cachedScene_ != frame.scene ||
      cachedTopologyVersion_ != frame.scene->topologyVersion();
  const uint64_t modelMaterialBindingVersion =
      frame.resources->modelMaterialBindingVersion();
  const bool materialDirty =
      topologyDirty || cachedMaterialVersion_ != materialSnapshot.version ||
      cachedModelMaterialBindingVersion_ != modelMaterialBindingVersion;
  const bool transformDirty =
      topologyDirty ||
      cachedTransformVersion_ != frame.scene->transformVersion();
  const bool excludeTransmissionBlend =
      frame.sharedResources.transparentTransmissionStageEnabled;
  const bool transmissionBlendPolicyDirty =
      cachedExcludeTransmissionBlend_ != excludeTransmissionBlend;
  const uint64_t geometryMutationVersion = gpu_.geometryMutationVersion();
  const bool geometryDirty =
      geometryMutationVersion != 0u &&
      geometryMutationVersion != cachedGeometryMutationVersion_;
  const bool needsGeometryRebuild =
      geometryDirty && !meshDrawTemplates_.empty();
  if (topologyDirty || materialDirty || needsGeometryRebuild ||
      transmissionBlendPolicyDirty) {
    rebuildSceneCache(*frame.sharedResources.sceneDrawDatabase, *frame.scene,
                      excludeTransmissionBlend);
    cachedMaterialVersion_ = materialSnapshot.version;
    cachedModelMaterialBindingVersion_ = modelMaterialBindingVersion;
  } else if (geometryDirty) {
    cachedGeometryMutationVersion_ = geometryMutationVersion;
  }
  if (meshDrawTemplates_.empty()) {
    return publishContributors();
  }
  const std::array rings{&instanceMatricesRing_, &instanceRemapRing_};
  const uint32_t ringCount =
      growDynamicBufferRings(gpu_.getSwapchainImageCount(), rings);
  instanceDataRingUploadVersions_.resize(ringCount,
                                         std::numeric_limits<uint64_t>::max());
  const uint32_t frameSlot =
      static_cast<uint32_t>(frame.frameIndex % instanceMatricesRing_.size());
  const std::span<const Renderable> renderables = frame.scene->renderables();
  if (transformDirty) {
    rebuildForwardInstances(renderables, instanceMatrices_, instanceRemap_,
                            instanceDataRingUploadVersions_);
    cachedTransformVersion_ = frame.scene->transformVersion();
  }
  const ForwardSceneGpuData *sceneGpu =
      &*frame.sharedResources.forwardSceneGpuData;
  const MaterialTableGpuData *materialGpu =
      &*frame.sharedResources.materialTableGpuData;
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);
  if (animationSceneData == nullptr) {
    auto matricesBufferResult = ensureInstanceMatricesRingCapacity(std::max(
        instanceMatrices_.size() * sizeof(InstanceData), sizeof(InstanceData)));
    if (matricesBufferResult.hasError()) {
      return matricesBufferResult;
    }
  }
  auto remapBufferResult = ensureInstanceRemapRingCapacity(
      std::max(instanceRemap_.size() * sizeof(uint32_t), sizeof(uint32_t)));
  if (remapBufferResult.hasError()) {
    return remapBufferResult;
  }
  collectForwardEnvironmentTextures(*frame.scene, *frame.resources,
                                    environmentTextureAccessHandles_);
  if (materialDirty || materialTextureAccessHandles_.empty()) {
    collectForwardMaterialTextures(*frame.resources, meshDrawTemplates_,
                                   materialTextureAccessHandles_);
  }
  if (materialDirty) {
    cachedMaterialVersion_ = materialSnapshot.version;
  }
  auto instanceUpload = uploadForwardInstances(
      gpu_, instanceMatricesRing_, instanceRemapRing_, instanceMatrices_,
      instanceRemap_, instanceDataRingUploadVersions_, frameSlot,
      cachedTransformVersion_, animationSceneData != nullptr);
  if (instanceUpload.hasError()) {
    return instanceUpload;
  }
  const uint64_t frameDataAddress = sceneGpu->postTaaFrameDataAddress != 0u
                                        ? sceneGpu->postTaaFrameDataAddress
                                        : sceneGpu->frameDataAddress;
  transparentUsesJitteredProjection_ =
      frameDataAddress == sceneGpu->frameDataAddress;
  const BufferHandle instanceMatricesBufferHandle =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesBuffer
          : instanceMatricesRing_[frameSlot].buffer->handle();
  const uint64_t instanceMatricesAddress =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesAddress
          : gpu_.getBufferDeviceAddress(instanceMatricesBufferHandle);
  const uint64_t instanceRemapAddress = gpu_.getBufferDeviceAddress(
      instanceRemapRing_[frameSlot].buffer->handle());
  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  const TextureHandle colorTexture = frame.sharedResources.frameColorTexture;
  const Format depthFormat = nuri::isValid(depthTexture)
                                 ? gpu_.getTextureFormat(depthTexture)
                                 : Format::Count;
  const Format colorFormat = nuri::isValid(colorTexture)
                                 ? gpu_.getTextureFormat(colorTexture)
                                 : gpu_.getSwapchainFormat();
  auto pipelineResult = ensurePipelines(colorFormat, depthFormat);
  if (pipelineResult.hasError()) {
    return pipelineResult;
  }
  meshSortableDraws_.clear();
  drawPushConstants_.clear();
  pickDrawItems_.clear();
  pickPushConstants_.clear();
  meshSortableDraws_.reserve(meshDrawTemplates_.size());
  drawPushConstants_.reserve(meshDrawTemplates_.size());
  pickDrawItems_.reserve(meshDrawTemplates_.size());
  pickPushConstants_.reserve(meshDrawTemplates_.size());
  const uint32_t renderableCount = saturateToU32(renderables.size());
  for (size_t i = 0; i < meshDrawTemplates_.size(); ++i) {
    const MeshDrawTemplate &entry = meshDrawTemplates_[i];
    const std::optional<SubmeshLod> lod =
        resolveForwardLod(*entry.submesh, settings.opaque.forcedMeshLod);
    if (!lod.has_value()) {
      continue;
    }
    const float sortDepth =
        transparentSortDepth(frame.camera.view, entry.renderable->modelMatrix,
                             entry.submesh->bounds);
    BufferHandle vertexBuffer = entry.baseVertexBuffer;
    uint64_t vertexBufferAddress = gpu_.getBufferDeviceAddress(
        entry.baseVertexBuffer, entry.vertexBufferByteOffset);
    uint64_t vertexDecodeBufferAddress =
        nuri::isValid(entry.baseVertexDecodeBuffer)
            ? gpu_.getBufferDeviceAddress(entry.baseVertexDecodeBuffer)
            : 0u;
    uint32_t vertexDecodeIndex = entry.vertexDecodeIndex;
    uint32_t packedVertexFormat = entry.packedVertexFormat;
    if (animationSceneData != nullptr) {
      const AnimatedRenderableGeometryOverride &geometryOverride =
          animationSceneData
              ->geometryOverridesByRenderable[entry.instanceIndex];
      if (geometryOverride.enabled &&
          nuri::isValid(geometryOverride.vertexBuffer) &&
          animationOverrideCoversSubmesh(geometryOverride, *entry.submesh)) {
        const uint64_t overrideVertexAddress = gpu_.getBufferDeviceAddress(
            geometryOverride.vertexBuffer, geometryOverride.vertexByteOffset);
        vertexBuffer = geometryOverride.vertexBuffer;
        vertexBufferAddress = overrideVertexAddress;
        vertexDecodeBufferAddress = 0u;
        vertexDecodeIndex = 0u;
        packedVertexFormat =
            static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32);
      }
    }
    drawPushConstants_.push_back(PushConstants{
        .frameDataAddress = frameDataAddress,
        .vertexBufferAddress = vertexBufferAddress,
        .vertexDecodeBufferAddress = vertexDecodeBufferAddress,
        .instanceMatricesAddress = instanceMatricesAddress,
        .instanceRemapAddress = instanceRemapAddress,
        .instanceCentersPhaseAddress = 0u,
        .instanceBaseMatricesAddress = 0u,
        .instanceCount = renderableCount,
        .materialIndex = entry.materialIndex,
        .vertexDecodeIndex = vertexDecodeIndex,
        .packedVertexFormat = packedVertexFormat,
        .timeSeconds = static_cast<float>(frame.timeSeconds),
        .tessNearDistance = 1.0f,
        .tessFarDistance = 8.0f,
        .tessMinFactor = 1.0f,
        .tessMaxFactor = 1.0f,
        .debugVisualizationMode = 0u,
    });
    const PushConstants &pc = drawPushConstants_.back();
    DrawItem draw{};
    draw.pipeline = selectMeshPipeline(entry.doubleSided);
    draw.vertexBuffer = vertexBuffer;
    draw.indexBuffer = entry.indexBuffer;
    draw.indexBufferOffset = entry.indexBufferOffset;
    draw.indexFormat = IndexFormat::U32;
    draw.indexCount = lod->indexCount;
    draw.instanceCount = 1u;
    draw.firstIndex = lod->indexOffset;
    draw.firstInstance = entry.instanceIndex;
    if (nuri::isValid(depthTexture)) {
      draw.useDepthState = true;
      draw.depthState = {.compareOp = CompareOp::LessEqual,
                         .isDepthWriteEnabled = false};
    }
    draw.pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&pc), sizeof(PushConstants));
    draw.debugLabel = kTransparentMeshLabel;
    draw.debugColor = kTransparentMeshDebugColor;
    meshSortableDraws_.push_back(TransparentStageSortableDraw{
        .draw = draw,
        .sortDepth = sortDepth,
        .stableOrder = static_cast<uint32_t>(i),
    });
    pickPushConstants_.push_back(pc);
    const PushConstants &pickPc = pickPushConstants_.back();
    DrawItem pickDraw = draw;
    pickDraw.pipeline = selectPickPipeline(entry.doubleSided);
    pickDraw.pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&pickPc), sizeof(PushConstants));
    pickDraw.debugLabel = kTransparentMeshPickLabel;
    pickDrawItems_.push_back(pickDraw);
  }
  sortableDraws_.clear();
  fixedDraws_.clear();
  passTextureReads_.clear();
  sortableDraws_.reserve(meshSortableDraws_.size() + 16u);
  fixedDraws_.reserve(8u);
  for (const TransparentStageSortableDraw &draw : meshSortableDraws_) {
    sortableDraws_.push_back(draw);
  }
  for (const TextureHandle handle : environmentTextureAccessHandles_) {
    appendUniqueForwardHandle(passTextureReads_, handle);
  }
  for (const TextureHandle handle : materialTextureAccessHandles_) {
    appendUniqueForwardHandle(passTextureReads_, handle);
  }
  for (const TextureHandle handle : sceneGpu->indirectDependencyTextures) {
    appendUniqueForwardHandle(passTextureReads_, handle);
  }
  const uint32_t stableOrderBase = saturateToU32(sortableDraws_.size());
  for (const TransparentStageSortableDraw &source : contributorSortableDraws_) {
    sortableDraws_.push_back(source);
    sortableDraws_.back().stableOrder += stableOrderBase;
  }
  for (const FixedDrawEntry &draw : contributorFixedDraws_) {
    fixedDraws_.push_back(draw);
  }
  for (const TextureHandle handle : contributorTextureReads_) {
    appendUniqueForwardHandle(passTextureReads_, handle);
  }
  sortTransparentDraws(std::span<TransparentStageSortableDraw>(
      sortableDraws_.data(), sortableDraws_.size()));
  passDependencyBuffers_.clear();
  appendUniqueForwardHandle(passDependencyBuffers_, sceneGpu->buffer);
  appendUniqueForwardHandle(passDependencyBuffers_, materialGpu->headerBuffer);
  appendUniqueForwardHandle(passDependencyBuffers_,
                            materialGpu->clearcoatBuffer);
  appendUniqueForwardHandle(passDependencyBuffers_, materialGpu->sheenBuffer);
  appendUniqueForwardHandle(passDependencyBuffers_,
                            materialGpu->transmissionBuffer);
  appendUniqueForwardHandle(passDependencyBuffers_,
                            materialGpu->specularBuffer);
  appendUniqueForwardHandle(passDependencyBuffers_,
                            instanceMatricesBufferHandle);
  appendUniqueForwardHandle(passDependencyBuffers_,
                            instanceRemapRing_[frameSlot].buffer->handle());
  for (const BufferHandle handle : sceneGpu->indirectDependencyBuffers) {
    appendUniqueForwardHandle(passDependencyBuffers_, handle);
  }
  if ((sceneGpu->shadowFlags & kShadowFrameFlagEnabled) != 0u) {
    const auto &shadowFrameGpuData = frame.sharedResources.shadowFrameGpuData;
    appendUniqueForwardHandle(passDependencyBuffers_,
                              shadowFrameGpuData->buffer);
    for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
         ++cascadeIndex) {
      const TextureHandle texture =
          frame.sharedResources.shadowCascadeTextures[cascadeIndex];
      if (nuri::isValid(texture)) {
        appendUniqueForwardHandle(passTextureReads_, texture);
      }
    }
  }
  frame.metrics.transparent.meshDraws =
      saturateToU32(meshSortableDraws_.size());
  frame.metrics.transparent.pickDraws = saturateToU32(pickDrawItems_.size());
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentRenderer::appendTransparentMainPass(RenderFrameContext &frame,
                                               RenderGraphBuilder &graph) {
  const AntiAliasingDebugView debugView = sanitizeAntiAliasingDebugView(
      settingsOrDefault(frame).antiAliasing.debug.view);
  if (isAntiAliasingDebugOutputView(debugView)) {
    return Result<bool, std::string>::makeResult(true);
  }
  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  const TextureHandle colorTexture = frame.sharedResources.frameColorTexture;
  const RenderGraphTextureId sceneDepthGraphTexture =
      frame.sharedResources.sceneDepthGraphTexture;
  const uint32_t transparentDrawCount =
      saturateToU32(sortableDraws_.size() + fixedDraws_.size());
  AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
  if (transparentDrawCount > 0u && aaMetrics.taaResolvedSceneColorPublished) {
    aaMetrics.taaTransparentPostTaaDrawCount = transparentDrawCount;
    aaMetrics.taaTransparentPostTaaMeshDrawCount =
        saturateToU32(meshSortableDraws_.size());
    aaMetrics.taaTransparentPostTaaContributorDrawCount = saturateToU32(
        contributorSortableDraws_.size() + contributorFixedDraws_.size());
    aaMetrics.taaTransparentPostTaaFixedDrawCount =
        saturateToU32(fixedDraws_.size());
    aaMetrics.taaTransparentEdgeJitterEstimate =
        frame.camera.jitterEnabled && transparentUsesJitteredProjection_ ? 1.0f
                                                                         : 0.0f;
  }
  return appendTransparentPass(
      frame, graph, colorTexture, depthTexture, sceneDepthGraphTexture,
      std::span<const TransparentStageSortableDraw>(sortableDraws_.data(),
                                                    sortableDraws_.size()),
      std::span<const FixedDrawEntry>(fixedDraws_.data(), fixedDraws_.size()),
      std::span<const TextureHandle>(passTextureReads_.data(),
                                     passTextureReads_.size()),
      std::span<const BufferHandle>(passDependencyBuffers_.data(),
                                    passDependencyBuffers_.size()));
}

Result<bool, std::string> TransparentRenderer::ensureInitialized() {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaderResult = createShaders();
  if (shaderResult.hasError()) {
    return shaderResult;
  }
  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransparentRenderer::createShaders() {
  destroyShaders();
  const std::filesystem::path pickVertexPath =
      alphaPickFragmentPath_.parent_path() / "main_id.vert";
  struct ShaderSpec {
    std::string_view name;
    const std::filesystem::path *path;
    ShaderStage stage;
  };
  const std::array specs{
      ShaderSpec{"transparent_main", &config_.meshVertex, ShaderStage::Vertex},
      ShaderSpec{"transparent_main", &config_.meshFragment,
                 ShaderStage::Fragment},
      ShaderSpec{"transparent_main_pick", &pickVertexPath, ShaderStage::Vertex},
      ShaderSpec{"transparent_main_pick", &alphaPickFragmentPath_,
                 ShaderStage::Fragment},
  };
  for (size_t index = 0; index < specs.size(); ++index) {
    auto compiler = Shader::create(specs[index].name, gpu_);
    auto result = compiler->compileFromFile(specs[index].path->string(),
                                            specs[index].stage);
    if (result.hasError()) {
      destroyShaders();
      return Result<bool, std::string>::makeError(result.error());
    }
    shaders_[index] = result.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentRenderer::ensurePipelines(Format colorFormat, Format depthFormat) {
  const bool meshPipelinesValid =
      nuri::isValid(meshPipelineHandle_) &&
      nuri::isValid(meshDoubleSidedPipelineHandle_) &&
      meshPipelineColorFormat_ == colorFormat &&
      meshPipelineDepthFormat_ == depthFormat;
  const bool pickPipelinesValid =
      nuri::isValid(meshPickPipelineHandle_) &&
      nuri::isValid(meshPickDoubleSidedPipelineHandle_) &&
      pickPipelineDepthFormat_ == depthFormat;
  if (meshPipelinesValid && pickPipelinesValid) {
    return Result<bool, std::string>::makeResult(true);
  }
  destroyPipelineState();
  auto meshResult = gpu_.createRenderPipeline(
      meshPipelineDesc(colorFormat, depthFormat, shaders_[0], shaders_[1], true,
                       CullMode::Back),
      "transparent_mesh");
  if (meshResult.hasError()) {
    return Result<bool, std::string>::makeError(meshResult.error());
  }
  meshPipelineHandle_ = meshResult.value();
  auto meshDoubleResult = gpu_.createRenderPipeline(
      meshPipelineDesc(colorFormat, depthFormat, shaders_[0], shaders_[1], true,
                       CullMode::None),
      "transparent_mesh_double_sided");
  if (meshDoubleResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(meshDoubleResult.error());
  }
  meshDoubleSidedPipelineHandle_ = meshDoubleResult.value();
  auto pickResult = gpu_.createRenderPipeline(
      meshPipelineDesc(Format::R32_UINT, depthFormat, shaders_[2], shaders_[3],
                       false, CullMode::Back),
      "transparent_mesh_pick");
  if (pickResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(pickResult.error());
  }
  meshPickPipelineHandle_ = pickResult.value();
  auto pickDoubleResult = gpu_.createRenderPipeline(
      meshPipelineDesc(Format::R32_UINT, depthFormat, shaders_[2], shaders_[3],
                       false, CullMode::None),
      "transparent_mesh_pick_double_sided");
  if (pickDoubleResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(pickDoubleResult.error());
  }
  meshPickDoubleSidedPipelineHandle_ = pickDoubleResult.value();
  meshPipelineColorFormat_ = colorFormat;
  meshPipelineDepthFormat_ = depthFormat;
  pickPipelineDepthFormat_ = depthFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentRenderer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
  return ensureDynamicBufferRingCapacity(
      gpu_, instanceMatricesRing_,
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = Storage::Device,
                 .size = requiredBytes},
      "transparent_instance_matrices", [this](size_t i) {
        instanceDataRingUploadVersions_[i] =
            std::numeric_limits<uint64_t>::max();
      });
}

Result<bool, std::string>
TransparentRenderer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
  return ensureDynamicBufferRingCapacity(
      gpu_, instanceRemapRing_,
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = Storage::Device,
                 .size = requiredBytes},
      "transparent_instance_remap", [this](size_t i) {
        instanceDataRingUploadVersions_[i] =
            std::numeric_limits<uint64_t>::max();
      });
}

void TransparentRenderer::rebuildSceneCache(const SceneDrawDatabase &database,
                                            const RenderScene &scene,
                                            bool excludeTransmissionBlend) {
  meshDrawTemplates_.clear();
  for (const SceneDrawRecord &draw : database.draws()) {
    if (draw.alphaBlended && !(draw.transmission && excludeTransmissionBlend)) {
      meshDrawTemplates_.push_back(draw);
    }
  }
  cachedScene_ = &scene;
  cachedTopologyVersion_ = scene.topologyVersion();
  cachedGeometryMutationVersion_ = gpu_.geometryMutationVersion();
  cachedExcludeTransmissionBlend_ = excludeTransmissionBlend;
}

Result<bool, std::string>
TransparentRenderer::collectContributorDraws(RenderFrameContext &frame) {
  contributorSortableDraws_.reserve(16u);
  contributorFixedDraws_.reserve(8u);
  contributorTextureReads_.reserve(8u);
  contributorDependencyBuffers_.reserve(16u);
  for (const TransparentContributionCollector &collector :
       frame.transparentContributors.collectors()) {
    TransparentStageContribution contribution{};
    auto contributionResult =
        collector.collect(collector.user, frame, contribution);
    if (contributionResult.hasError()) {
      return contributionResult;
    }
    const bool requiresFrameColorFeedback =
        contribution.feedbackRefresh.appendRefresh != nullptr;
    if (requiresFrameColorFeedback) {
      if (contribution.feedbackRefresh.user == nullptr) {
        return Result<bool, std::string>::makeError(
            "TransparentRenderer::collectContributorDraws: feedback refresh "
            "owner is missing");
      }
      if (feedbackRefresh_.appendRefresh != nullptr) {
        return Result<bool, std::string>::makeError(
            "TransparentRenderer::collectContributorDraws: multiple feedback "
            "refresh owners are unsupported in one sorted stream");
      }
      feedbackRefresh_ = contribution.feedbackRefresh;
    }
    const uint32_t dependencyOffset =
        saturateToU32(contributorDependencyBuffers_.size());
    for (const BufferHandle handle : contribution.dependencyBuffers) {
      contributorDependencyBuffers_.push_back(handle);
    }
    const uint32_t dependencyCount =
        saturateToU32(contributorDependencyBuffers_.size() - dependencyOffset);
    const uint32_t stableOrderBase =
        saturateToU32(contributorSortableDraws_.size());
    for (const TransparentStageSortableDraw &source :
         contribution.sortableDraws) {
      DrawItem draw = source.draw;
      applyContributorDependencies(draw, contribution.dependencyBuffers);
      contributorSortableDraws_.push_back(source);
      TransparentStageSortableDraw &copy = contributorSortableDraws_.back();
      copy.draw = draw;
      copy.stableOrder += stableOrderBase;
      copy.requiresFrameColorFeedback = requiresFrameColorFeedback;
      copy.dependencyOffset = dependencyOffset;
      copy.dependencyCount = dependencyCount;
    }
    for (const DrawItem &source : contribution.fixedDraws) {
      DrawItem draw = source;
      applyContributorDependencies(draw, contribution.dependencyBuffers);
      contributorFixedDraws_.push_back(FixedDrawEntry{
          .draw = draw,
          .dependencyOffset = dependencyOffset,
          .dependencyCount = dependencyCount,
      });
    }
    for (const TextureHandle handle : contribution.textureReads) {
      appendUniqueForwardHandle(contributorTextureReads_, handle);
    }
    frame.metrics.transparent.contributorSortableDraws +=
        saturateToU32(contribution.sortableDraws.size());
    frame.metrics.transparent.contributorFixedDraws +=
        saturateToU32(contribution.fixedDraws.size());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransparentRenderer::appendTransparentPass(
    RenderFrameContext &frame, RenderGraphBuilder &graph,
    TextureHandle colorTexture, TextureHandle depthTexture,
    RenderGraphTextureId sceneDepthGraphTexture,
    std::span<const TransparentStageSortableDraw> sortableDraws,
    std::span<const FixedDrawEntry> fixedDraws,
    std::span<const TextureHandle> textureReads,
    std::span<const BufferHandle> dependencyBuffers) {
  if (sortableDraws.empty() && fixedDraws.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  transparentRunDrawItems_.clear();
  transparentRunDrawItems_.reserve(sortableDraws.size() + fixedDraws.size());
  transparentRunDependencyBuffers_.clear();
  transparentCandidateDependencyBuffers_.clear();
  bool feedbackFallbackSourceReady = false;
  const auto resetRunDependencies = [this, dependencyBuffers]() {
    transparentRunDependencyBuffers_.clear();
    for (const BufferHandle handle : dependencyBuffers) {
      appendUniqueForwardHandle(transparentRunDependencyBuffers_, handle);
    }
  };
  const auto appendDrawDependencyRange =
      [this](std::pmr::vector<BufferHandle> &target, uint32_t offset,
             uint32_t count) -> Result<bool, std::string> {
    const size_t begin = static_cast<size_t>(offset);
    const size_t end = begin + static_cast<size_t>(count);
    if (begin > contributorDependencyBuffers_.size() ||
        end > contributorDependencyBuffers_.size()) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::appendTransparentPass: contributor "
          "dependency range is out of bounds");
    }
    for (size_t index = begin; index < end; ++index) {
      appendUniqueForwardHandle(target, contributorDependencyBuffers_[index]);
    }
    return Result<bool, std::string>::makeResult(true);
  };
  resetRunDependencies();
  const auto flushTransparentRun =
      [this, &graph, colorTexture, depthTexture, sceneDepthGraphTexture,
       textureReads, &resetRunDependencies, &run = transparentRunDrawItems_](
          std::string_view debugLabel) -> Result<bool, std::string> {
    if (run.empty()) {
      return Result<bool, std::string>::makeResult(true);
    }
    auto runResult = appendTransparentDrawRun(
        graph, colorTexture, depthTexture, sceneDepthGraphTexture,
        std::span<const DrawItem>(run.data(), run.size()), textureReads,
        std::span<const BufferHandle>(transparentRunDependencyBuffers_.data(),
                                      transparentRunDependencyBuffers_.size()),
        debugLabel);
    run.clear();
    resetRunDependencies();
    return runResult;
  };
  const auto prepareRunDependenciesForDraw =
      [this, &appendDrawDependencyRange, &flushTransparentRun,
       &resetRunDependencies](uint32_t offset,
                              uint32_t count) -> Result<bool, std::string> {
    transparentCandidateDependencyBuffers_.assign(
        transparentRunDependencyBuffers_.begin(),
        transparentRunDependencyBuffers_.end());
    auto appendResult = appendDrawDependencyRange(
        transparentCandidateDependencyBuffers_, offset, count);
    if (appendResult.hasError()) {
      return appendResult;
    }
    if (transparentCandidateDependencyBuffers_.size() <=
        kMaxDependencyResources) {
      transparentRunDependencyBuffers_.assign(
          transparentCandidateDependencyBuffers_.begin(),
          transparentCandidateDependencyBuffers_.end());
      return Result<bool, std::string>::makeResult(true);
    }
    if (transparentRunDrawItems_.empty()) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::appendTransparentPass: single transparent "
          "draw dependency buffer count exceeds kMaxDependencyResources");
    }
    auto flushResult = flushTransparentRun(kTransparentPassLabel);
    if (flushResult.hasError()) {
      return flushResult;
    }
    resetRunDependencies();
    transparentCandidateDependencyBuffers_.assign(
        transparentRunDependencyBuffers_.begin(),
        transparentRunDependencyBuffers_.end());
    appendResult = appendDrawDependencyRange(
        transparentCandidateDependencyBuffers_, offset, count);
    if (appendResult.hasError()) {
      return appendResult;
    }
    if (transparentCandidateDependencyBuffers_.size() >
        kMaxDependencyResources) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::appendTransparentPass: single transparent "
          "draw dependency buffer count exceeds kMaxDependencyResources");
    }
    transparentRunDependencyBuffers_.assign(
        transparentCandidateDependencyBuffers_.begin(),
        transparentCandidateDependencyBuffers_.end());
    return Result<bool, std::string>::makeResult(true);
  };
  for (const TransparentStageSortableDraw &draw : sortableDraws) {
    const bool requiresFeedback = draw.requiresFrameColorFeedback;
    if (!requiresFeedback) {
      auto dependencyResult = prepareRunDependenciesForDraw(
          draw.dependencyOffset, draw.dependencyCount);
      if (dependencyResult.hasError()) {
        return dependencyResult;
      }
      transparentRunDrawItems_.push_back(draw.draw);
      continue;
    }
    if (feedbackRefresh_.mode ==
        TransparentStageFeedbackRefreshMode::OnceBeforeFirstDraw) {
      if (!feedbackFallbackSourceReady) {
        auto flushResult = flushTransparentRun(kTransparentPassLabel);
        if (flushResult.hasError()) {
          return flushResult;
        }
        auto feedbackResult =
            feedbackRefresh_.appendRefresh(feedbackRefresh_.user, frame, graph);
        if (feedbackResult.hasError()) {
          return feedbackResult;
        }
        feedbackFallbackSourceReady = true;
      }
      auto dependencyResult = prepareRunDependenciesForDraw(
          draw.dependencyOffset, draw.dependencyCount);
      if (dependencyResult.hasError()) {
        return dependencyResult;
      }
      transparentRunDrawItems_.push_back(draw.draw);
      continue;
    }
    auto flushResult = flushTransparentRun(kTransparentPassLabel);
    if (flushResult.hasError()) {
      return flushResult;
    }
    auto feedbackResult =
        feedbackRefresh_.appendRefresh(feedbackRefresh_.user, frame, graph);
    if (feedbackResult.hasError()) {
      return feedbackResult;
    }
    auto dependencyResult = prepareRunDependenciesForDraw(draw.dependencyOffset,
                                                          draw.dependencyCount);
    if (dependencyResult.hasError()) {
      return dependencyResult;
    }
    auto drawResult = appendTransparentDrawRun(
        graph, colorTexture, depthTexture, sceneDepthGraphTexture,
        std::span<const DrawItem>(&draw.draw, 1u), textureReads,
        std::span<const BufferHandle>(transparentRunDependencyBuffers_.data(),
                                      transparentRunDependencyBuffers_.size()),
        kTransparentTransmissionPassLabel);
    if (drawResult.hasError()) {
      return drawResult;
    }
    resetRunDependencies();
  }
  for (const FixedDrawEntry &draw : fixedDraws) {
    auto dependencyResult = prepareRunDependenciesForDraw(draw.dependencyOffset,
                                                          draw.dependencyCount);
    if (dependencyResult.hasError()) {
      return dependencyResult;
    }
    transparentRunDrawItems_.push_back(draw.draw);
  }
  return flushTransparentRun(kTransparentPassLabel);
}

Result<bool, std::string> TransparentRenderer::appendTransparentDrawRun(
    RenderGraphBuilder &graph, TextureHandle colorTexture,
    TextureHandle depthTexture, RenderGraphTextureId sceneDepthGraphTexture,
    std::span<const DrawItem> draws,
    std::span<const TextureHandle> textureReads,
    std::span<const BufferHandle> dependencyBuffers,
    std::string_view debugLabel) {
  if (draws.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const bool hasPriorColorPass =
      nuri::isValid(colorTexture) || graph.passCount() > 0u;
  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color = {.loadOp = hasPriorColorPass ? LoadOp::Load : LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  if (nuri::isValid(colorTexture)) {
    auto colorImportResult =
        graph.importTexture(colorTexture, "transparent_frame_color");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    passDesc.colorTexture = colorImportResult.value();
  }
  if (nuri::isValid(depthTexture)) {
    passDesc.depth = {.loadOp = LoadOp::Load,
                      .storeOp = StoreOp::Store,
                      .clearDepth = 1.0f,
                      .clearStencil = 0};
    if (nuri::isValid(sceneDepthGraphTexture)) {
      passDesc.depthTexture = sceneDepthGraphTexture;
    } else {
      auto importResult =
          graph.importTexture(depthTexture, "transparent_scene_depth");
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }
      passDesc.depthTexture = importResult.value();
    }
  }
  passDesc.draws = draws;
  std::pmr::vector<BufferHandle> preResolvedDrawBuffers(memory_);
  preResolvedDrawBuffers.reserve(meshDrawTemplates_.size() * 3u +
                                 draws.size() * 2u + 8u);
  for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
    appendUniqueForwardHandle(preResolvedDrawBuffers, entry.baseVertexBuffer);
    appendUniqueForwardHandle(preResolvedDrawBuffers,
                              entry.baseVertexDecodeBuffer);
    appendUniqueForwardHandle(preResolvedDrawBuffers, entry.indexBuffer);
  }
  appendPreResolvedDrawBuffers(preResolvedDrawBuffers, draws);
  passDesc.drawBuffersPreResolved = true;
  passDesc.preResolvedDrawBuffers = std::span<const BufferHandle>(
      preResolvedDrawBuffers.data(), preResolvedDrawBuffers.size());
  passDesc.dependencyBuffers = dependencyBuffers;
  passDesc.debugLabel = debugLabel;
  passDesc.debugColor = kTransparentPassDebugColor;
  auto addResult = graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  for (const TextureHandle handle : textureReads) {
    auto importResult = graph.importTexture(handle, "transparent_texture_read");
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    auto readResult =
        graph.addTextureRead(addResult.value(), importResult.value());
    if (readResult.hasError()) {
      return Result<bool, std::string>::makeError(readResult.error());
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentRenderer::appendTransparentPickPass(RenderFrameContext &frame,
                                               RenderGraphBuilder &graph) {
  if (pickDrawItems_.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const RenderGraphTextureId resolvedPickColor =
      nuri::isValid(frame.sharedResources.opaquePickGraphTexture)
          ? frame.sharedResources.opaquePickGraphTexture
          : RenderGraphTextureId{};
  const RenderGraphTextureId resolvedPickDepth =
      nuri::isValid(frame.sharedResources.opaquePickDepthGraphTexture)
          ? frame.sharedResources.opaquePickDepthGraphTexture
          : RenderGraphTextureId{};
  if (!nuri::isValid(resolvedPickColor) || !nuri::isValid(resolvedPickDepth)) {
    return Result<bool, std::string>::makeResult(true);
  }
  RenderGraphGraphicsPassDesc pickDesc{};
  pickDesc.color = {.loadOp = LoadOp::Load,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}};
  pickDesc.colorTexture = resolvedPickColor;
  pickDesc.depth = {.loadOp = LoadOp::Load,
                    .storeOp = StoreOp::Store,
                    .clearDepth = 1.0f,
                    .clearStencil = 0};
  pickDesc.depthTexture = resolvedPickDepth;
  pickDesc.draws =
      std::span<const DrawItem>(pickDrawItems_.data(), pickDrawItems_.size());
  std::pmr::vector<BufferHandle> preResolvedDrawBuffers(memory_);
  preResolvedDrawBuffers.reserve(meshDrawTemplates_.size() * 3u +
                                 pickDrawItems_.size() * 2u + 8u);
  for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
    appendUniqueForwardHandle(preResolvedDrawBuffers, entry.baseVertexBuffer);
    appendUniqueForwardHandle(preResolvedDrawBuffers,
                              entry.baseVertexDecodeBuffer);
    appendUniqueForwardHandle(preResolvedDrawBuffers, entry.indexBuffer);
  }
  appendPreResolvedDrawBuffers(
      preResolvedDrawBuffers,
      std::span<const DrawItem>(pickDrawItems_.data(), pickDrawItems_.size()));
  pickDesc.drawBuffersPreResolved = true;
  pickDesc.preResolvedDrawBuffers = std::span<const BufferHandle>(
      preResolvedDrawBuffers.data(), preResolvedDrawBuffers.size());
  pickDesc.dependencyBuffers = std::span<const BufferHandle>(
      passDependencyBuffers_.data(), passDependencyBuffers_.size());
  pickDesc.debugLabel = kTransparentPickPassLabel;
  pickDesc.debugColor = kTransparentPickPassDebugColor;
  auto addResult = graph.addGraphicsPass(pickDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  for (const TextureHandle handle : materialTextureAccessHandles_) {
    auto importResult =
        graph.importTexture(handle, "transparent_pick_texture_read");
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    auto readResult =
        graph.addTextureRead(addResult.value(), importResult.value());
    if (readResult.hasError()) {
      return Result<bool, std::string>::makeError(readResult.error());
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

void TransparentRenderer::resetCachedState() {
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedModelMaterialBindingVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  cachedExcludeTransmissionBlend_ = true;
  meshDrawTemplates_.clear();
  instanceMatrices_.clear();
  instanceRemap_.clear();
  instanceDataRingUploadVersions_.clear();
  materialTextureAccessHandles_.clear();
  environmentTextureAccessHandles_.clear();
}

void TransparentRenderer::resetFrameBuildState() {
  contributorSortableDraws_.clear();
  contributorFixedDraws_.clear();
  contributorTextureReads_.clear();
  contributorDependencyBuffers_.clear();
  drawPushConstants_.clear();
  pickPushConstants_.clear();
  meshSortableDraws_.clear();
  sortableDraws_.clear();
  fixedDraws_.clear();
  passDrawItems_.clear();
  transparentRunDrawItems_.clear();
  transparentRunDependencyBuffers_.clear();
  transparentCandidateDependencyBuffers_.clear();
  pickDrawItems_.clear();
  passTextureReads_.clear();
  passDependencyBuffers_.clear();
  feedbackRefresh_ = {};
  transparentUsesJitteredProjection_ = true;
}

void TransparentRenderer::destroyPipelineState() {
  if (nuri::isValid(meshPickDoubleSidedPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshPickDoubleSidedPipelineHandle_);
  }
  if (nuri::isValid(meshPickPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshPickPipelineHandle_);
  }
  if (nuri::isValid(meshDoubleSidedPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshDoubleSidedPipelineHandle_);
  }
  if (nuri::isValid(meshPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshPipelineHandle_);
  }
  meshPipelineHandle_ = {};
  meshDoubleSidedPipelineHandle_ = {};
  meshPickPipelineHandle_ = {};
  meshPickDoubleSidedPipelineHandle_ = {};
  meshPipelineColorFormat_ = Format::Count;
  meshPipelineDepthFormat_ = Format::Count;
  pickPipelineDepthFormat_ = Format::Count;
}

void TransparentRenderer::destroyShaders() {
  for (ShaderHandle &shader : shaders_) {
    if (nuri::isValid(shader)) {
      gpu_.destroyShaderModule(shader);
    }
    shader = {};
  }
}

void TransparentRenderer::destroyBuffers() {
  retireDynamicBufferRing(instanceMatricesRing_);
  retireDynamicBufferRing(instanceRemapRing_);
  instanceMatricesRing_.clear();
  instanceRemapRing_.clear();
  instanceDataRingUploadVersions_.clear();
}

void TransparentRenderer::sortTransparentDraws(
    std::span<TransparentStageSortableDraw> draws) {
  std::sort(draws.begin(), draws.end(),
            [](const TransparentStageSortableDraw &lhs,
               const TransparentStageSortableDraw &rhs) {
              if (lhs.sortDepth != rhs.sortDepth) {
                return lhs.sortDepth > rhs.sortDepth;
              }
              return lhs.stableOrder < rhs.stableOrder;
            });
}

RenderPipelineHandle
TransparentRenderer::selectMeshPipeline(bool doubleSided) const {
  return doubleSided ? meshDoubleSidedPipelineHandle_ : meshPipelineHandle_;
}

RenderPipelineHandle
TransparentRenderer::selectPickPipeline(bool doubleSided) const {
  return doubleSided ? meshPickDoubleSidedPipelineHandle_
                     : meshPickPipelineHandle_;
}

void registerTransparentStages(RenderPipeline &pipeline, GPUDevice &gpu,
                               RuntimeOpaqueShaderConfig config,
                               std::pmr::memory_resource *memory,
                               SceneDrawDatabase *database) {
  if (!database) {
    pipeline.addProvider(std::make_unique<SceneDrawDatabase>(gpu, memory));
  }
  auto owner =
      std::make_unique<TransparentRenderer>(gpu, std::move(config), memory);
  owner->onAttach();
  auto *renderer = pipeline.addComponent(
      std::move(owner),
      PipelineComponentDesc{
          .publish =
              [](void *state, FrameBuildContext &ctx) {
                static_cast<TransparentRenderer *>(state)->publishFrameData(
                    ctx.frame);
                return Result<bool, std::string>::makeResult(true);
              },
          .prepare =
              [](void *state, FrameBuildContext &ctx) {
                return static_cast<TransparentRenderer *>(state)
                    ->prepareTransparentPasses(ctx.frame);
              },
      });
  const auto meshEnabled = [](const void *, const FrameBuildContext &ctx) {
    return !ctx.frame.settings ||
           renderSettingsOrDefault(ctx.frame).transparent.enabled;
  };
  pipeline.addStage(PipelineStageDesc{
      .componentName = "TransparentFeature",
      .name = "TransparentMainPass",
      .state = renderer,
      .enabled = [](const void *, const FrameBuildContext &) { return true; },
      .build =
          [](void *state, FrameBuildContext &ctx) {
            return static_cast<TransparentRenderer *>(state)
                ->appendTransparentMainPass(ctx.frame, ctx.graph);
          },
  });
  pipeline.addStage(PipelineStageDesc{
      .componentName = "TransparentFeature",
      .name = "TransparentPickPass",
      .state = renderer,
      .enabled = meshEnabled,
      .build =
          [](void *state, FrameBuildContext &ctx) {
            return static_cast<TransparentRenderer *>(state)
                ->appendTransparentPickPass(ctx.frame, ctx.graph);
          },
  });
}

} // namespace nuri
