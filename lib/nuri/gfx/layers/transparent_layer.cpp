#include "nuri/pch.h"

#include "nuri/gfx/layers/transparent_layer.h"

#include "nuri/core/layer_stack.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

constexpr uint32_t kTransparentPassDebugColor = 0x66aaffffu;
constexpr uint32_t kTransparentPickPassDebugColor = 0x66ff88ffu;
constexpr uint32_t kTransparentMeshDebugColor = 0x66aaffffu;
constexpr std::string_view kTransparentPassLabel = "Transparent Pass";
constexpr std::string_view kTransparentPickPassLabel = "Transparent Pick Pass";
constexpr std::string_view kTransparentMeshLabel = "TransparentMesh";
constexpr std::string_view kTransparentMeshPickLabel = "TransparentMeshPick";

[[nodiscard]] std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}

[[nodiscard]] bool isSameTextureHandle(TextureHandle lhs, TextureHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

[[nodiscard]] bool isSameBufferHandle(BufferHandle lhs, BufferHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

void appendUniqueTexture(std::pmr::vector<TextureHandle> &handles,
                         TextureHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const TextureHandle existing : handles) {
    if (isSameTextureHandle(existing, handle)) {
      return;
    }
  }
  handles.push_back(handle);
}

void appendUniqueBuffer(std::pmr::vector<BufferHandle> &handles,
                        BufferHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const BufferHandle existing : handles) {
    if (isSameBufferHandle(existing, handle)) {
      return;
    }
  }
  handles.push_back(handle);
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
  };
}

uint32_t saturateToU32(size_t value) {
  return static_cast<uint32_t>(
      std::min(value, size_t(std::numeric_limits<uint32_t>::max())));
}

[[nodiscard]] const RenderSettings &
settingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? *frame.settings : kDefaultSettings;
}

[[nodiscard]] std::optional<SubmeshLod>
resolveTransparentLod(const Submesh &submesh, const RenderSettings &settings) {
  // Transparent meshes intentionally reuse the opaque forced-LOD override so a
  // single debug knob forces the same mesh LOD across both queues.
  if (settings.opaque.forcedMeshLod < 0) {
    if (submesh.indexCount > 0) {
      return SubmeshLod{.indexOffset = submesh.indexOffset,
                        .indexCount = submesh.indexCount,
                        .error = 0.0f};
    }
    for (uint32_t lod = 0; lod < std::max(submesh.lodCount, 1u); ++lod) {
      if (submesh.lods[lod].indexCount > 0) {
        return submesh.lods[lod];
      }
    }
    return std::nullopt;
  }

  const uint32_t lodCount =
      std::clamp(submesh.lodCount, 1u, Submesh::kMaxLodCount);
  uint32_t candidate = std::min(
      static_cast<uint32_t>(settings.opaque.forcedMeshLod), lodCount - 1u);
  while (candidate > 0u && submesh.lods[candidate].indexCount == 0u) {
    --candidate;
  }
  if (submesh.lods[candidate].indexCount > 0u) {
    return submesh.lods[candidate];
  }
  if (submesh.indexCount > 0u) {
    return SubmeshLod{.indexOffset = submesh.indexOffset,
                      .indexCount = submesh.indexCount,
                      .error = 0.0f};
  }
  return std::nullopt;
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

TransparentLayer::TransparentLayer(GPUDevice &gpu,
                                   TransparentLayerConfig config,
                                   std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(resolveMemoryResource(memory)), instanceMatricesRing_(memory_),
      instanceRemapRing_(memory_), meshDrawTemplates_(memory_),
      instanceMatrices_(memory_), instanceRemap_(memory_),
      instanceDataRingUploadVersions_(memory_), materialGpuDataCache_(memory_),
      materialTextureAccessHandles_(memory_),
      environmentTextureAccessHandles_(memory_),
      contributorSortableDraws_(memory_), contributorFixedDraws_(memory_),
      contributorTextureReads_(memory_), drawPushConstants_(memory_),
      pickPushConstants_(memory_), meshSortableDraws_(memory_),
      sortableDraws_(memory_), fixedDraws_(memory_), passDrawItems_(memory_),
      pickDrawItems_(memory_), passTextureReads_(memory_),
      passDependencyBuffers_(memory_) {
  const std::filesystem::path basePath =
      !config_.pickFragment.empty() ? config_.pickFragment.parent_path()
                                    : config_.meshFragment.parent_path();
  alphaPickFragmentPath_ = basePath / "main_id_alpha.frag";
}

TransparentLayer::~TransparentLayer() { onDetach(); }

void TransparentLayer::onAttach() {
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    NURI_LOG_WARNING("TransparentLayer::onAttach: %s",
                     initResult.error().c_str());
  }
}

void TransparentLayer::onDetach() {
  destroyBuffers();
  destroyPipelineState();
  destroyShaders();
  meshShader_.reset();
  meshPickShader_.reset();
  resetFrameBuildState();
  resetCachedState();
  initialized_ = false;
}

void TransparentLayer::publishFrameData(RenderFrameContext &frame) {
  const RenderSettings &settings = settingsOrDefault(frame);
  if (settings.transparent.enabled) {
    frame.channels.publish<bool>(kFrameChannelTransparentStageEnabled, true);
  }
}

Result<bool, std::string>
TransparentLayer::buildRenderGraph(RenderFrameContext &frame,
                                   RenderGraphBuilder &graph) {
  NURI_PROFILER_FUNCTION();
  frame.metrics.transparent = {};
  resetFrameBuildState();

  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.transparent.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!frame.scene) {
    return Result<bool, std::string>::makeError(
        "TransparentLayer::buildRenderGraph: frame scene is null");
  }
  if (!frame.resources) {
    return Result<bool, std::string>::makeError(
        "TransparentLayer::buildRenderGraph: frame resources are null");
  }

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return initResult;
  }

  const MaterialTableSnapshot materialSnapshot =
      frame.resources->materialSnapshot();
  const bool topologyDirty =
      cachedScene_ != frame.scene ||
      cachedTopologyVersion_ != frame.scene->topologyVersion();
  const bool materialDirty =
      topologyDirty || cachedMaterialVersion_ != materialSnapshot.version;
  const bool transformDirty =
      topologyDirty ||
      cachedTransformVersion_ != frame.scene->transformVersion();
  const uint64_t geometryMutationVersion = gpu_.geometryMutationVersion();
  const bool geometryDirty =
      geometryMutationVersion != 0u &&
      geometryMutationVersion != cachedGeometryMutationVersion_;
  const bool needsGeometryRebuild =
      geometryDirty && !meshDrawTemplates_.empty();
  if (topologyDirty || materialDirty || needsGeometryRebuild) {
    auto rebuildResult = rebuildSceneCache(
        *frame.scene, *frame.resources,
        static_cast<uint32_t>(materialSnapshot.gpuData.size()));
    if (rebuildResult.hasError()) {
      return rebuildResult;
    }
    cachedMaterialVersion_ = materialSnapshot.version;
  } else if (geometryDirty) {
    // Geometry compaction can invalidate addresses for transparent mesh draws,
    // but if we currently have no transparent meshes cached there is no scene
    // work to rebuild. Advance the observed version to avoid periodic rescans.
    cachedGeometryMutationVersion_ = geometryMutationVersion;
  }

  auto contributorResult = collectContributorDraws(frame);
  if (contributorResult.hasError()) {
    return contributorResult;
  }

  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  RenderGraphTextureId sceneDepthGraphTexture{};
  if (const RenderGraphTextureId *published =
          frame.channels.tryGet<RenderGraphTextureId>(
              kFrameChannelSceneDepthGraphTexture);
      published != nullptr) {
    sceneDepthGraphTexture = *published;
  }

  if (meshDrawTemplates_.empty()) {
    cachedTransformVersion_ = frame.scene->transformVersion();
    sortTransparentDraws(std::span<TransparentStageSortableDraw>(
        contributorSortableDraws_.data(), contributorSortableDraws_.size()));
    return appendTransparentPass(
        graph, depthTexture, sceneDepthGraphTexture,
        std::span<const TransparentStageSortableDraw>(
            contributorSortableDraws_.data(), contributorSortableDraws_.size()),
        std::span<const DrawItem>(contributorFixedDraws_.data(),
                                  contributorFixedDraws_.size()),
        std::span<const TextureHandle>(contributorTextureReads_.data(),
                                       contributorTextureReads_.size()),
        {});
  }

  auto ringResult =
      ensureRingBufferCount(std::max(1u, gpu_.getSwapchainImageCount()));
  if (ringResult.hasError()) {
    return ringResult;
  }
  const uint32_t frameSlot =
      static_cast<uint32_t>(frame.frameIndex % instanceMatricesRing_.size());
  const std::span<const Renderable> renderables = frame.scene->renderables();

  if (transformDirty) {
    instanceMatrices_.clear();
    instanceRemap_.clear();
    instanceMatrices_.reserve(renderables.size());
    instanceRemap_.reserve(renderables.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(renderables.size()); ++i) {
      instanceMatrices_.push_back(renderables[i].modelMatrix);
      instanceRemap_.push_back(i);
    }
    cachedTransformVersion_ = frame.scene->transformVersion();
    std::fill(instanceDataRingUploadVersions_.begin(),
              instanceDataRingUploadVersions_.end(),
              std::numeric_limits<uint64_t>::max());
  }

  auto frameDataBufferResult = ensureFrameDataBufferCapacity(sizeof(FrameData));
  if (frameDataBufferResult.hasError()) {
    return frameDataBufferResult;
  }
  auto materialBufferResult = ensureMaterialBufferCapacity(
      std::max<size_t>(materialSnapshot.gpuData.size(), 1u) *
      sizeof(MaterialGpuData));
  if (materialBufferResult.hasError()) {
    return materialBufferResult;
  }
  auto matricesBufferResult = ensureInstanceMatricesRingCapacity(std::max(
      instanceMatrices_.size() * sizeof(glm::mat4), sizeof(glm::mat4)));
  if (matricesBufferResult.hasError()) {
    return matricesBufferResult;
  }
  auto remapBufferResult = ensureInstanceRemapRingCapacity(
      std::max(instanceRemap_.size() * sizeof(uint32_t), sizeof(uint32_t)));
  if (remapBufferResult.hasError()) {
    return remapBufferResult;
  }

  collectEnvironmentTextureReads(*frame.scene, *frame.resources);
  if (materialDirty || materialTextureAccessHandles_.empty()) {
    auto cacheResult = rebuildMaterialTextureAccessCache(*frame.resources);
    if (cacheResult.hasError()) {
      return cacheResult;
    }
  }

  uint32_t cubemapTexId = kInvalidTextureBindlessIndex;
  uint32_t hasCubemap = 0u;
  uint32_t irradianceTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredGgxTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredCharlieTexId = kInvalidTextureBindlessIndex;
  uint32_t brdfLutTexId = kInvalidTextureBindlessIndex;
  uint32_t frameFlags = 0u;
  const uint32_t cubemapSamplerId = gpu_.getCubemapSamplerBindlessIndex();
  const EnvironmentHandles environment = frame.scene->environment();

  if (const TextureRecord *cubemap =
          frame.resources->tryGet(environment.cubemap);
      cubemap != nullptr && nuri::isValid(cubemap->texture)) {
    cubemapTexId = cubemap->bindlessIndex;
    hasCubemap = 1u;
  }
  if (const TextureRecord *irradiance =
          frame.resources->tryGet(environment.irradiance);
      irradiance != nullptr && nuri::isValid(irradiance->texture)) {
    irradianceTexId = irradiance->bindlessIndex;
    frameFlags |= 1u << 0u;
  }
  if (const TextureRecord *prefilteredGgx =
          frame.resources->tryGet(environment.prefilteredGgx);
      prefilteredGgx != nullptr && nuri::isValid(prefilteredGgx->texture)) {
    prefilteredGgxTexId = prefilteredGgx->bindlessIndex;
    frameFlags |= 1u << 1u;
  }
  if (const TextureRecord *prefilteredCharlie =
          frame.resources->tryGet(environment.prefilteredCharlie);
      prefilteredCharlie != nullptr &&
      nuri::isValid(prefilteredCharlie->texture)) {
    prefilteredCharlieTexId = prefilteredCharlie->bindlessIndex;
    frameFlags |= 1u << 2u;
  } else if ((frameFlags & (1u << 1u)) != 0u) {
    prefilteredCharlieTexId = prefilteredGgxTexId;
    frameFlags |= 1u << 2u;
  }
  if (const TextureRecord *brdfLut =
          frame.resources->tryGet(environment.brdfLut);
      brdfLut != nullptr && nuri::isValid(brdfLut->texture)) {
    brdfLutTexId = brdfLut->bindlessIndex;
    frameFlags |= 1u << 3u;
  }
  if (gpu_.getSwapchainFormat() != Format::RGBA8_SRGB) {
    frameFlags |= 1u << 4u;
  }

  frameData_ = FrameData{
      .view = frame.camera.view,
      .proj = frame.camera.proj,
      .cameraPos = frame.camera.cameraPos,
      .cubemapTexId = cubemapTexId,
      .hasCubemap = hasCubemap,
      .irradianceTexId = irradianceTexId,
      .prefilteredGgxTexId = prefilteredGgxTexId,
      .prefilteredCharlieTexId = prefilteredCharlieTexId,
      .brdfLutTexId = brdfLutTexId,
      .flags = frameFlags,
      .cubemapSamplerId = cubemapSamplerId,
  };

  if (!frameDataUploadValid_ || !(uploadedFrameData_ == frameData_)) {
    const std::span<const std::byte> frameDataBytes{
        reinterpret_cast<const std::byte *>(&frameData_), sizeof(frameData_)};
    auto updateResult =
        gpu_.updateBuffer(frameDataBuffer_->handle(), frameDataBytes, 0);
    if (updateResult.hasError()) {
      return updateResult;
    }
    uploadedFrameData_ = frameData_;
    frameDataUploadValid_ = true;
  }

  if (materialDirty || materialGpuDataCache_.empty()) {
    materialGpuDataCache_.clear();
    materialGpuDataCache_.insert(materialGpuDataCache_.end(),
                                 materialSnapshot.gpuData.begin(),
                                 materialSnapshot.gpuData.end());
    if (materialGpuDataCache_.empty()) {
      materialGpuDataCache_.push_back(MaterialGpuData{});
    }
    const std::span<const std::byte> materialBytes{
        reinterpret_cast<const std::byte *>(materialGpuDataCache_.data()),
        materialGpuDataCache_.size() * sizeof(MaterialGpuData)};
    auto updateResult =
        gpu_.updateBuffer(materialBuffer_->handle(), materialBytes, 0);
    if (updateResult.hasError()) {
      return updateResult;
    }
    cachedMaterialVersion_ = materialSnapshot.version;
  }

  const bool needsInstanceDataUpload =
      instanceDataRingUploadVersions_[frameSlot] != cachedTransformVersion_;
  if (needsInstanceDataUpload && !instanceMatrices_.empty()) {
    const std::span<const std::byte> matrixBytes{
        reinterpret_cast<const std::byte *>(instanceMatrices_.data()),
        instanceMatrices_.size() * sizeof(glm::mat4)};
    auto updateResult = gpu_.updateBuffer(
        instanceMatricesRing_[frameSlot].buffer->handle(), matrixBytes, 0);
    if (updateResult.hasError()) {
      return updateResult;
    }
  }
  if (needsInstanceDataUpload && !instanceRemap_.empty()) {
    const std::span<const std::byte> remapBytes{
        reinterpret_cast<const std::byte *>(instanceRemap_.data()),
        instanceRemap_.size() * sizeof(uint32_t)};
    auto updateResult = gpu_.updateBuffer(
        instanceRemapRing_[frameSlot].buffer->handle(), remapBytes, 0);
    if (updateResult.hasError()) {
      return updateResult;
    }
  }
  if (needsInstanceDataUpload) {
    instanceDataRingUploadVersions_[frameSlot] = cachedTransformVersion_;
  }

  const uint64_t frameDataAddress =
      gpu_.getBufferDeviceAddress(frameDataBuffer_->handle());
  const uint64_t materialBufferAddress =
      gpu_.getBufferDeviceAddress(materialBuffer_->handle());
  const uint64_t instanceMatricesAddress = gpu_.getBufferDeviceAddress(
      instanceMatricesRing_[frameSlot].buffer->handle());
  const uint64_t instanceRemapAddress = gpu_.getBufferDeviceAddress(
      instanceRemapRing_[frameSlot].buffer->handle());
  if (frameDataAddress == 0u || materialBufferAddress == 0u ||
      instanceMatricesAddress == 0u || instanceRemapAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "TransparentLayer::buildRenderGraph: invalid GPU buffer address");
  }

  const Format depthFormat = nuri::isValid(depthTexture)
                                 ? gpu_.getTextureFormat(depthTexture)
                                 : Format::Count;
  auto pipelineResult = ensurePipelines(gpu_.getSwapchainFormat(), depthFormat);
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
    if (entry.renderable == nullptr || entry.submesh == nullptr) {
      continue;
    }
    const std::optional<SubmeshLod> lod =
        resolveTransparentLod(*entry.submesh, settings);
    if (!lod.has_value()) {
      continue;
    }

    const glm::vec3 worldCenter =
        glm::vec3(entry.renderable->modelMatrix *
                  glm::vec4(entry.submesh->bounds.getCenter(), 1.0f));
    const float sortDepth =
        -(frame.camera.view * glm::vec4(worldCenter, 1.0f)).z;

    drawPushConstants_.push_back(PushConstants{
        .frameDataAddress = frameDataAddress,
        .vertexBufferAddress = entry.vertexBufferAddress,
        .instanceMatricesAddress = instanceMatricesAddress,
        .instanceRemapAddress = instanceRemapAddress,
        .materialBufferAddress = materialBufferAddress,
        .instanceCentersPhaseAddress = 0u,
        .instanceBaseMatricesAddress = 0u,
        .instanceCount = renderableCount,
        .materialIndex = entry.materialIndex,
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
    appendUniqueTexture(passTextureReads_, handle);
  }
  for (const TextureHandle handle : materialTextureAccessHandles_) {
    appendUniqueTexture(passTextureReads_, handle);
  }
  const uint32_t stableOrderBase = saturateToU32(sortableDraws_.size());
  for (const TransparentStageSortableDraw &source : contributorSortableDraws_) {
    sortableDraws_.push_back(TransparentStageSortableDraw{
        .draw = source.draw,
        .sortDepth = source.sortDepth,
        .stableOrder = stableOrderBase + source.stableOrder,
    });
  }
  for (const DrawItem &draw : contributorFixedDraws_) {
    fixedDraws_.push_back(draw);
  }
  for (const TextureHandle handle : contributorTextureReads_) {
    appendUniqueTexture(passTextureReads_, handle);
  }

  sortTransparentDraws(std::span<TransparentStageSortableDraw>(
      sortableDraws_.data(), sortableDraws_.size()));

  passDependencyBuffers_.clear();
  appendUniqueBuffer(passDependencyBuffers_, frameDataBuffer_->handle());
  appendUniqueBuffer(passDependencyBuffers_, materialBuffer_->handle());
  appendUniqueBuffer(passDependencyBuffers_,
                     instanceMatricesRing_[frameSlot].buffer->handle());
  appendUniqueBuffer(passDependencyBuffers_,
                     instanceRemapRing_[frameSlot].buffer->handle());
  auto passResult = appendTransparentPass(
      graph, depthTexture, sceneDepthGraphTexture,
      std::span<const TransparentStageSortableDraw>(sortableDraws_.data(),
                                                    sortableDraws_.size()),
      std::span<const DrawItem>(fixedDraws_.data(), fixedDraws_.size()),
      std::span<const TextureHandle>(passTextureReads_.data(),
                                     passTextureReads_.size()),
      std::span<const BufferHandle>(passDependencyBuffers_.data(),
                                    passDependencyBuffers_.size()));
  if (passResult.hasError()) {
    return passResult;
  }

  auto pickResult = appendTransparentPickPass(frame, graph);
  if (pickResult.hasError()) {
    return pickResult;
  }

  frame.metrics.transparent.meshDraws =
      saturateToU32(meshSortableDraws_.size());
  frame.metrics.transparent.pickDraws = saturateToU32(pickDrawItems_.size());
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransparentLayer::ensureInitialized() {
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

Result<bool, std::string> TransparentLayer::createShaders() {
  destroyShaders();
  meshShader_ = Shader::create("transparent_main", gpu_);
  meshPickShader_ = Shader::create("transparent_main_pick", gpu_);
  if (!meshShader_ || !meshPickShader_) {
    return Result<bool, std::string>::makeError(
        "TransparentLayer::createShaders: failed to create shader wrappers");
  }

  auto vertexResult = meshShader_->compileFromFile(config_.meshVertex.string(),
                                                   ShaderStage::Vertex);
  if (vertexResult.hasError()) {
    return Result<bool, std::string>::makeError(vertexResult.error());
  }
  auto fragmentResult = meshShader_->compileFromFile(
      config_.meshFragment.string(), ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }
  auto pickResult = meshPickShader_->compileFromFile(
      alphaPickFragmentPath_.string(), ShaderStage::Fragment);
  if (pickResult.hasError()) {
    return Result<bool, std::string>::makeError(pickResult.error());
  }

  meshVertexShader_ = vertexResult.value();
  meshFragmentShader_ = fragmentResult.value();
  meshPickFragmentShader_ = pickResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentLayer::ensurePipelines(Format colorFormat, Format depthFormat) {
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
      meshPipelineDesc(colorFormat, depthFormat, meshVertexShader_,
                       meshFragmentShader_, true, CullMode::Back),
      "transparent_mesh");
  if (meshResult.hasError()) {
    return Result<bool, std::string>::makeError(meshResult.error());
  }
  meshPipelineHandle_ = meshResult.value();

  auto meshDoubleResult = gpu_.createRenderPipeline(
      meshPipelineDesc(colorFormat, depthFormat, meshVertexShader_,
                       meshFragmentShader_, true, CullMode::None),
      "transparent_mesh_double_sided");
  if (meshDoubleResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(meshDoubleResult.error());
  }
  meshDoubleSidedPipelineHandle_ = meshDoubleResult.value();

  auto pickResult = gpu_.createRenderPipeline(
      meshPipelineDesc(Format::R32_UINT, depthFormat, meshVertexShader_,
                       meshPickFragmentShader_, false, CullMode::Back),
      "transparent_mesh_pick");
  if (pickResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(pickResult.error());
  }
  meshPickPipelineHandle_ = pickResult.value();

  auto pickDoubleResult = gpu_.createRenderPipeline(
      meshPipelineDesc(Format::R32_UINT, depthFormat, meshVertexShader_,
                       meshPickFragmentShader_, false, CullMode::None),
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
TransparentLayer::ensureFrameDataBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(FrameData));
  if (frameDataBuffer_ && frameDataBuffer_->valid() &&
      frameDataBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (frameDataBuffer_ && frameDataBuffer_->valid()) {
    gpu_.destroyBuffer(frameDataBuffer_->handle());
  }
  frameDataBuffer_.reset();
  auto bufferResult = Buffer::create(gpu_,
                                     BufferDesc{.usage = BufferUsage::Storage,
                                                .storage = Storage::Device,
                                                .size = requested},
                                     "transparent_frame_data");
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }
  frameDataBuffer_ = std::move(bufferResult.value());
  frameDataBufferCapacityBytes_ = requested;
  frameDataUploadValid_ = false;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentLayer::ensureMaterialBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(MaterialGpuData));
  if (materialBuffer_ && materialBuffer_->valid() &&
      materialBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (materialBuffer_ && materialBuffer_->valid()) {
    gpu_.destroyBuffer(materialBuffer_->handle());
  }
  materialBuffer_.reset();
  auto bufferResult = Buffer::create(gpu_,
                                     BufferDesc{.usage = BufferUsage::Storage,
                                                .storage = Storage::Device,
                                                .size = requested},
                                     "transparent_material_data");
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }
  materialBuffer_ = std::move(bufferResult.value());
  materialBufferCapacityBytes_ = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentLayer::ensureRingBufferCount(uint32_t requiredCount) {
  const uint32_t count = std::max(requiredCount, 1u);
  while (instanceMatricesRing_.size() < count) {
    instanceMatricesRing_.push_back(DynamicBufferSlot{});
  }
  while (instanceRemapRing_.size() < count) {
    instanceRemapRing_.push_back(DynamicBufferSlot{});
  }
  instanceDataRingUploadVersions_.resize(count,
                                         std::numeric_limits<uint64_t>::max());
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentLayer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
  for (size_t i = 0; i < instanceMatricesRing_.size(); ++i) {
    DynamicBufferSlot &slot = instanceMatricesRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requiredBytes) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    auto bufferResult = Buffer::create(gpu_,
                                       BufferDesc{.usage = BufferUsage::Storage,
                                                  .storage = Storage::Device,
                                                  .size = requiredBytes},
                                       "transparent_instance_matrices");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;
    instanceDataRingUploadVersions_[i] = std::numeric_limits<uint64_t>::max();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentLayer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
  for (size_t i = 0; i < instanceRemapRing_.size(); ++i) {
    DynamicBufferSlot &slot = instanceRemapRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requiredBytes) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    auto bufferResult = Buffer::create(gpu_,
                                       BufferDesc{.usage = BufferUsage::Storage,
                                                  .storage = Storage::Device,
                                                  .size = requiredBytes},
                                       "transparent_instance_remap");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;
    instanceDataRingUploadVersions_[i] = std::numeric_limits<uint64_t>::max();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentLayer::rebuildSceneCache(const RenderScene &scene,
                                    const ResourceManager &resources,
                                    uint32_t materialCount) {
  meshDrawTemplates_.clear();

  const std::span<const Renderable> renderables = scene.renderables();
  if (renderables.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<bool, std::string>::makeError(
        "TransparentLayer::rebuildSceneCache: renderables count exceeds "
        "UINT32_MAX");
  }

  size_t invalidMaterialFallbackCount = 0u;
  for (uint32_t renderableIndex = 0;
       renderableIndex < static_cast<uint32_t>(renderables.size());
       ++renderableIndex) {
    const Renderable &renderable = renderables[renderableIndex];
    const ModelRecord *modelRecord = resources.tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      return Result<bool, std::string>::makeError(
          "TransparentLayer::rebuildSceneCache: failed to resolve model");
    }
    GeometryAllocationView geometry{};
    if (!gpu_.resolveGeometry(modelRecord->model->geometryHandle(), geometry)) {
      return Result<bool, std::string>::makeError(
          "TransparentLayer::rebuildSceneCache: failed to resolve geometry");
    }
    const uint64_t vertexBufferAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (vertexBufferAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "TransparentLayer::rebuildSceneCache: invalid vertex buffer address");
    }

    const std::span<const Submesh> submeshes = modelRecord->model->submeshes();
    for (size_t submeshIndex = 0; submeshIndex < submeshes.size();
         ++submeshIndex) {
      const MaterialRef modelMaterial =
          modelRecord->materialForSubmesh(static_cast<uint32_t>(submeshIndex));
      const MaterialRef resolvedMaterial =
          nuri::isValid(modelMaterial) ? modelMaterial : renderable.material;
      const MaterialRecord *materialRecord = resources.tryGet(resolvedMaterial);
      if (materialRecord == nullptr ||
          materialRecord->desc.alphaMode != MaterialAlphaMode::Blend) {
        continue;
      }

      uint32_t materialIndex = resources.materialTableIndex(resolvedMaterial);
      if (materialCount == 0u || materialIndex >= materialCount) {
        materialIndex = 0u;
        ++invalidMaterialFallbackCount;
      }

      // Cached pointers stay valid only while scene.topologyVersion() is
      // unchanged; any topology mutation must bump that version so this cache
      // is rebuilt before meshDrawTemplates_ is reused.
      meshDrawTemplates_.push_back(MeshDrawTemplate{
          .renderable = &renderable,
          .submesh = &submeshes[submeshIndex],
          .submeshIndex = static_cast<uint32_t>(submeshIndex),
          .indexBuffer = geometry.indexBuffer,
          .indexBufferOffset = geometry.indexByteOffset,
          .vertexBufferAddress = vertexBufferAddress,
          .materialIndex = materialIndex,
          .instanceIndex = renderableIndex,
          .doubleSided = materialRecord->desc.doubleSided,
      });
    }
  }

  if (invalidMaterialFallbackCount > 0u) {
    if (!loggedMaterialFallbackWarning_) {
      NURI_LOG_WARNING(
          "TransparentLayer::rebuildSceneCache: %zu submesh draw(s) used "
          "fallback material index 0 due to missing/out-of-range material "
          "mapping",
          invalidMaterialFallbackCount);
      loggedMaterialFallbackWarning_ = true;
    }
  } else {
    loggedMaterialFallbackWarning_ = false;
  }

  cachedScene_ = &scene;
  cachedTopologyVersion_ = scene.topologyVersion();
  cachedGeometryMutationVersion_ = gpu_.geometryMutationVersion();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransparentLayer::rebuildMaterialTextureAccessCache(
    const ResourceManager &resources) {
  materialTextureAccessHandles_.clear();
  for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
    if (entry.renderable == nullptr || entry.submesh == nullptr) {
      continue;
    }
    const ModelRecord *modelRecord = resources.tryGet(entry.renderable->model);
    if (!modelRecord || !modelRecord->model) {
      continue;
    }
    const std::span<const Submesh> submeshes = modelRecord->model->submeshes();
    if (entry.submeshIndex >= submeshes.size()) {
      continue;
    }
    const MaterialRef modelMaterial =
        modelRecord->materialForSubmesh(entry.submeshIndex);
    const MaterialRef resolvedMaterial = nuri::isValid(modelMaterial)
                                             ? modelMaterial
                                             : entry.renderable->material;
    const MaterialRecord *materialRecord = resources.tryGet(resolvedMaterial);
    if (materialRecord == nullptr) {
      continue;
    }

    const std::array<TextureRef, 8> refs = {
        materialRecord->textureRefs.baseColor,
        materialRecord->textureRefs.metallicRoughness,
        materialRecord->textureRefs.normal,
        materialRecord->textureRefs.occlusion,
        materialRecord->textureRefs.emissive,
        materialRecord->textureRefs.clearcoat,
        materialRecord->textureRefs.clearcoatRoughness,
        materialRecord->textureRefs.clearcoatNormal,
    };
    for (const TextureRef ref : refs) {
      const TextureRecord *record = resources.tryGet(ref);
      if (record == nullptr || !nuri::isValid(record->texture)) {
        continue;
      }
      appendUniqueTexture(materialTextureAccessHandles_, record->texture);
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentLayer::collectContributorDraws(RenderFrameContext &frame) {
  contributorSortableDraws_.clear();
  contributorFixedDraws_.clear();
  contributorTextureReads_.clear();
  contributorSortableDraws_.reserve(16u);
  contributorFixedDraws_.reserve(8u);
  contributorTextureReads_.reserve(8u);

  if (frame.layerStack == nullptr) {
    return Result<bool, std::string>::makeError(
        "TransparentLayer::collectContributorDraws: frame layer stack is null");
  }

  auto collectResult =
      frame.layerStack->forEachLayerReverseResult([&](Layer &contributor) {
        TransparentStageContribution contribution{};
        auto contributionResult =
            contributor.buildTransparentStageContribution(frame, contribution);
        if (contributionResult.hasError()) {
          return contributionResult;
        }

        const uint32_t stableOrderBase =
            saturateToU32(contributorSortableDraws_.size());
        for (const TransparentStageSortableDraw &source :
             contribution.sortableDraws) {
          DrawItem draw = source.draw;
          applyContributorDependencies(draw, contribution.dependencyBuffers);
          contributorSortableDraws_.push_back(TransparentStageSortableDraw{
              .draw = draw,
              .sortDepth = source.sortDepth,
              .stableOrder = stableOrderBase + source.stableOrder,
          });
        }
        for (const DrawItem &source : contribution.fixedDraws) {
          DrawItem draw = source;
          applyContributorDependencies(draw, contribution.dependencyBuffers);
          contributorFixedDraws_.push_back(draw);
        }
        for (const TextureHandle handle : contribution.textureReads) {
          appendUniqueTexture(contributorTextureReads_, handle);
        }

        frame.metrics.transparent.contributorSortableDraws +=
            saturateToU32(contribution.sortableDraws.size());
        frame.metrics.transparent.contributorFixedDraws +=
            saturateToU32(contribution.fixedDraws.size());
        return Result<bool, std::string>::makeResult(true);
      });
  if (collectResult.hasError()) {
    return collectResult;
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransparentLayer::appendTransparentPass(
    RenderGraphBuilder &graph, TextureHandle depthTexture,
    RenderGraphTextureId sceneDepthGraphTexture,
    std::span<const TransparentStageSortableDraw> sortableDraws,
    std::span<const DrawItem> fixedDraws,
    std::span<const TextureHandle> textureReads,
    std::span<const BufferHandle> dependencyBuffers) {
  if (sortableDraws.empty() && fixedDraws.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  passDrawItems_.clear();
  passDrawItems_.reserve(sortableDraws.size() + fixedDraws.size());
  for (const TransparentStageSortableDraw &draw : sortableDraws) {
    passDrawItems_.push_back(draw.draw);
  }
  for (const DrawItem &draw : fixedDraws) {
    passDrawItems_.push_back(draw);
  }

  const bool hasPriorColorPass = graph.passCount() > 0u;
  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color = {.loadOp = hasPriorColorPass ? LoadOp::Load : LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
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
  passDesc.draws =
      std::span<const DrawItem>(passDrawItems_.data(), passDrawItems_.size());
  passDesc.dependencyBuffers = dependencyBuffers;
  passDesc.debugLabel = kTransparentPassLabel;
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
TransparentLayer::appendTransparentPickPass(RenderFrameContext &frame,
                                            RenderGraphBuilder &graph) {
  if (pickDrawItems_.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  const RenderGraphTextureId *pickColor =
      frame.channels.tryGet<RenderGraphTextureId>(
          kFrameChannelOpaquePickGraphTexture);
  const RenderGraphTextureId *pickDepth =
      frame.channels.tryGet<RenderGraphTextureId>(
          kFrameChannelOpaquePickDepthGraphTexture);
  if (pickColor == nullptr || pickDepth == nullptr) {
    return Result<bool, std::string>::makeResult(true);
  }

  RenderGraphGraphicsPassDesc pickDesc{};
  pickDesc.color = {.loadOp = LoadOp::Load,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}};
  pickDesc.colorTexture = *pickColor;
  pickDesc.depth = {.loadOp = LoadOp::Load,
                    .storeOp = StoreOp::Store,
                    .clearDepth = 1.0f,
                    .clearStencil = 0};
  pickDesc.depthTexture = *pickDepth;
  pickDesc.draws =
      std::span<const DrawItem>(pickDrawItems_.data(), pickDrawItems_.size());
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

void TransparentLayer::collectEnvironmentTextureReads(
    const RenderScene &scene, const ResourceManager &resources) {
  environmentTextureAccessHandles_.clear();
  const EnvironmentHandles environment = scene.environment();
  const std::array<TextureRef, 5> refs = {
      environment.cubemap, environment.irradiance, environment.prefilteredGgx,
      environment.prefilteredCharlie, environment.brdfLut};
  for (const TextureRef ref : refs) {
    const TextureRecord *record = resources.tryGet(ref);
    if (record == nullptr || !nuri::isValid(record->texture)) {
      continue;
    }
    appendUniqueTexture(environmentTextureAccessHandles_, record->texture);
  }
}

void TransparentLayer::resetCachedState() {
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  loggedMaterialFallbackWarning_ = false;

  meshDrawTemplates_.clear();
  instanceMatrices_.clear();
  instanceRemap_.clear();
  instanceDataRingUploadVersions_.clear();
  materialGpuDataCache_.clear();
  materialTextureAccessHandles_.clear();
  environmentTextureAccessHandles_.clear();
  frameData_ = {};
  uploadedFrameData_ = {};
  frameDataUploadValid_ = false;
}

void TransparentLayer::resetFrameBuildState() {
  contributorSortableDraws_.clear();
  contributorFixedDraws_.clear();
  contributorTextureReads_.clear();
  drawPushConstants_.clear();
  pickPushConstants_.clear();
  meshSortableDraws_.clear();
  sortableDraws_.clear();
  fixedDraws_.clear();
  passDrawItems_.clear();
  pickDrawItems_.clear();
  passTextureReads_.clear();
  passDependencyBuffers_.clear();
}

void TransparentLayer::destroyPipelineState() {
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

void TransparentLayer::destroyShaders() {
  if (nuri::isValid(meshVertexShader_)) {
    gpu_.destroyShaderModule(meshVertexShader_);
  }
  if (nuri::isValid(meshFragmentShader_)) {
    gpu_.destroyShaderModule(meshFragmentShader_);
  }
  if (nuri::isValid(meshPickFragmentShader_)) {
    gpu_.destroyShaderModule(meshPickFragmentShader_);
  }
  meshVertexShader_ = {};
  meshFragmentShader_ = {};
  meshPickFragmentShader_ = {};
}

void TransparentLayer::destroyBuffers() {
  if (frameDataBuffer_ && frameDataBuffer_->valid()) {
    gpu_.destroyBuffer(frameDataBuffer_->handle());
  }
  frameDataBuffer_.reset();
  frameDataBufferCapacityBytes_ = 0;
  frameDataUploadValid_ = false;

  if (materialBuffer_ && materialBuffer_->valid()) {
    gpu_.destroyBuffer(materialBuffer_->handle());
  }
  materialBuffer_.reset();
  materialBufferCapacityBytes_ = 0;

  for (DynamicBufferSlot &slot : instanceMatricesRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : instanceRemapRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  instanceMatricesRing_.clear();
  instanceRemapRing_.clear();
  instanceDataRingUploadVersions_.clear();
}

void TransparentLayer::sortTransparentDraws(
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
TransparentLayer::selectMeshPipeline(bool doubleSided) const {
  if (doubleSided && nuri::isValid(meshDoubleSidedPipelineHandle_)) {
    return meshDoubleSidedPipelineHandle_;
  }
  return meshPipelineHandle_;
}

RenderPipelineHandle
TransparentLayer::selectPickPipeline(bool doubleSided) const {
  if (doubleSided && nuri::isValid(meshPickDoubleSidedPipelineHandle_)) {
    return meshPickDoubleSidedPipelineHandle_;
  }
  return meshPickPipelineHandle_;
}

} // namespace nuri
