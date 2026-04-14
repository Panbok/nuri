#include "nuri/pch.h"

#include "nuri/gfx/renderers/transparent_renderer.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/renderers/detail/renderable_material_resolution.h"
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

[[nodiscard]] bool isTransmissionMaterial(const MaterialRecord &material) {
  return (material.desc.featureMask & kMaterialFeatureTransmission) != 0u;
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

void appendPreResolvedDrawBuffers(std::pmr::vector<BufferHandle> &handles,
                                  std::span<const DrawItem> draws) {
  for (const DrawItem &draw : draws) {
    appendUniqueBuffer(handles, draw.vertexBuffer);
    appendUniqueBuffer(handles, draw.indexBuffer);
    appendUniqueBuffer(handles, draw.indirectBuffer);
    appendUniqueBuffer(handles, draw.indirectCountBuffer);
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

const AnimationSceneFrameData *
resolveAnimationSceneFrameData(const RenderFrameContext &frame) {
  if (!frame.sharedResources.animationSceneGpuData.has_value()) {
    return nullptr;
  }
  const AnimationSceneFrameData &data =
      *frame.sharedResources.animationSceneGpuData;
  if (!nuri::isValid(data.instanceMatricesBuffer) ||
      data.instanceMatricesAddress == 0u) {
    return nullptr;
  }
  if (frame.scene == nullptr || data.scene != frame.scene ||
      data.sceneTopologyVersion != frame.scene->topologyVersion() ||
      data.renderableCount != frame.scene->renderables().size() ||
      data.geometryOverridesByRenderable.size() != data.renderableCount) {
    return nullptr;
  }
  return &data;
}

bool animationOverrideCoversSubmesh(
    const AnimatedRenderableGeometryOverride &geometryOverride,
    const Submesh &submesh) noexcept {
  const uint64_t requiredVertexCount =
      static_cast<uint64_t>(submesh.vertexOffset) + submesh.vertexCount;
  return static_cast<uint64_t>(geometryOverride.vertexCount) >=
         requiredVertexCount;
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
  meshShader_.reset();
  meshPickShader_.reset();
  resetFrameBuildState();
  resetCachedState();
  initialized_ = false;
}

void TransparentRenderer::publishFrameData(RenderFrameContext &frame) {
  const RenderSettings &settings = settingsOrDefault(frame);
  if (settings.transparent.enabled) {
    frame.sharedResources.transparentStageEnabled = true;
  }
}

Result<bool, std::string>
TransparentRenderer::prepareTransparentPasses(RenderFrameContext &frame) {
  NURI_PROFILER_FUNCTION();
  frame.metrics.transparent = {};
  resetFrameBuildState();

  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.transparent.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!frame.scene) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::prepareTransparentPasses: frame scene is null");
  }
  if (!frame.resources) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::prepareTransparentPasses: frame resources are "
        "null");
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
        static_cast<uint32_t>(materialSnapshot.headers.size()));
    if (rebuildResult.hasError()) {
      return rebuildResult;
    }
    cachedMaterialVersion_ = materialSnapshot.version;
  } else if (geometryDirty) {
    cachedGeometryMutationVersion_ = geometryMutationVersion;
  }

  auto contributorResult = collectContributorDraws(frame);
  if (contributorResult.hasError()) {
    return contributorResult;
  }

  if (meshDrawTemplates_.empty()) {
    cachedTransformVersion_ = frame.scene->transformVersion();
    sortableDraws_.clear();
    fixedDraws_.clear();
    passTextureReads_.clear();
    passDependencyBuffers_.clear();
    for (const TransparentStageSortableDraw &draw : contributorSortableDraws_) {
      sortableDraws_.push_back(draw);
    }
    for (const DrawItem &draw : contributorFixedDraws_) {
      fixedDraws_.push_back(draw);
    }
    for (const TextureHandle handle : contributorTextureReads_) {
      appendUniqueTexture(passTextureReads_, handle);
    }
    sortTransparentDraws(std::span<TransparentStageSortableDraw>(
        sortableDraws_.data(), sortableDraws_.size()));
    frame.metrics.transparent.meshDraws = 0u;
    frame.metrics.transparent.pickDraws = 0u;
    return Result<bool, std::string>::makeResult(true);
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
      instanceMatrices_.push_back(makeInstanceData(renderables[i].modelMatrix));
      instanceRemap_.push_back(i);
    }
    cachedTransformVersion_ = frame.scene->transformVersion();
    std::fill(instanceDataRingUploadVersions_.begin(),
              instanceDataRingUploadVersions_.end(),
              std::numeric_limits<uint64_t>::max());
  }

  if (!frame.sharedResources.forwardSceneGpuData.has_value() ||
      !nuri::isValid(frame.sharedResources.forwardSceneGpuData->buffer) ||
      frame.sharedResources.forwardSceneGpuData->frameDataAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::prepareTransparentPasses: forward scene GPU data "
        "is "
        "unavailable");
  }
  if (!frame.sharedResources.materialTableGpuData.has_value()) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::prepareTransparentPasses: material table GPU "
        "data is unavailable");
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

  collectEnvironmentTextureReads(*frame.scene, *frame.resources);
  if (materialDirty || materialTextureAccessHandles_.empty()) {
    auto cacheResult = rebuildMaterialTextureAccessCache(*frame.resources);
    if (cacheResult.hasError()) {
      return cacheResult;
    }
  }
  if (materialDirty) {
    cachedMaterialVersion_ = materialSnapshot.version;
  }

  const bool needsInstanceDataUpload =
      animationSceneData == nullptr &&
      instanceDataRingUploadVersions_[frameSlot] != cachedTransformVersion_;
  if (needsInstanceDataUpload && !instanceMatrices_.empty()) {
    const std::span<const std::byte> matrixBytes{
        reinterpret_cast<const std::byte *>(instanceMatrices_.data()),
        instanceMatrices_.size() * sizeof(InstanceData)};
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

  const uint64_t frameDataAddress = sceneGpu->frameDataAddress;
  const uint64_t directionalLightBufferAddress =
      sceneGpu->directionalLightBufferAddress;
  const uint64_t localLightBufferAddress = sceneGpu->localLightBufferAddress;
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
  if (materialGpu->headerBufferAddress == 0u ||
      materialGpu->clearcoatBufferAddress == 0u ||
      materialGpu->sheenBufferAddress == 0u ||
      materialGpu->transmissionBufferAddress == 0u ||
      materialGpu->specularBufferAddress == 0u ||
      instanceMatricesAddress == 0u || instanceRemapAddress == 0u ||
      (sceneGpu->directionalLightCount > 0u &&
       directionalLightBufferAddress == 0u) ||
      (sceneGpu->localLightCount > 0u && localLightBufferAddress == 0u)) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::prepareTransparentPasses: invalid GPU buffer "
        "address");
  }

  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  const TextureHandle colorTexture = resolveFrameColorTexture(frame);
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
  if (instanceRemap_.size() < renderables.size()) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::prepareTransparentPasses: instance remap "
        "buffer is smaller than the renderable set");
  }
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
    if (entry.instanceIndex >= renderableCount ||
        entry.instanceIndex >= instanceRemap_.size()) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::prepareTransparentPasses: draw instance index "
          "is out of range");
    }
    if (entry.materialIndex >= materialSnapshot.headers.size()) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::prepareTransparentPasses: draw material index "
          "is out of range");
    }
    if (!nuri::isValid(entry.indexBuffer)) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::prepareTransparentPasses: draw index buffer "
          "is invalid");
    }

    const glm::vec3 worldCenter =
        glm::vec3(entry.renderable->modelMatrix *
                  glm::vec4(entry.submesh->bounds.getCenter(), 1.0f));
    BufferHandle vertexBuffer = entry.baseVertexBuffer;
    uint64_t vertexBufferAddress = gpu_.getBufferDeviceAddress(
        entry.baseVertexBuffer, entry.vertexBufferByteOffset);
    uint64_t vertexDecodeBufferAddress =
        nuri::isValid(entry.baseVertexDecodeBuffer)
            ? gpu_.getBufferDeviceAddress(entry.baseVertexDecodeBuffer)
            : 0u;
    uint32_t vertexDecodeIndex = entry.vertexDecodeIndex;
    uint32_t packedVertexFormat = entry.packedVertexFormat;
    if (vertexBufferAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::prepareTransparentPasses: invalid live "
          "vertex buffer address");
    }
    if (packedVertexFormat ==
            static_cast<uint32_t>(PackedVertexFormat::StaticQuantized20) &&
        vertexDecodeBufferAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::prepareTransparentPasses: invalid live "
          "vertex decode buffer address");
    }
    if (animationSceneData != nullptr &&
        entry.instanceIndex <
            animationSceneData->geometryOverridesByRenderable.size()) {
      const AnimatedRenderableGeometryOverride &geometryOverride =
          animationSceneData
              ->geometryOverridesByRenderable[entry.instanceIndex];
      if (geometryOverride.enabled &&
          nuri::isValid(geometryOverride.vertexBuffer) &&
          animationOverrideCoversSubmesh(geometryOverride, *entry.submesh)) {
        const uint64_t overrideVertexAddress = gpu_.getBufferDeviceAddress(
            geometryOverride.vertexBuffer, geometryOverride.vertexByteOffset);
        if (overrideVertexAddress != 0u) {
          vertexBuffer = geometryOverride.vertexBuffer;
          vertexBufferAddress = overrideVertexAddress;
          vertexDecodeBufferAddress = 0u;
          vertexDecodeIndex = 0u;
          packedVertexFormat =
              static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32);
        }
      }
    }

    const float sortDepth =
        -(frame.camera.view * glm::vec4(worldCenter, 1.0f)).z;

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
  appendUniqueBuffer(passDependencyBuffers_, sceneGpu->buffer);
  appendUniqueBuffer(passDependencyBuffers_, materialGpu->headerBuffer);
  appendUniqueBuffer(passDependencyBuffers_, materialGpu->clearcoatBuffer);
  appendUniqueBuffer(passDependencyBuffers_, materialGpu->sheenBuffer);
  appendUniqueBuffer(passDependencyBuffers_, materialGpu->transmissionBuffer);
  appendUniqueBuffer(passDependencyBuffers_, materialGpu->specularBuffer);
  appendUniqueBuffer(passDependencyBuffers_, instanceMatricesBufferHandle);
  appendUniqueBuffer(passDependencyBuffers_,
                     instanceRemapRing_[frameSlot].buffer->handle());
  if (loggedAddressProbeTopologyVersion_ != frame.scene->topologyVersion() &&
      !drawPushConstants_.empty()) {
    loggedAddressProbeTopologyVersion_ = frame.scene->topologyVersion();
    const PushConstants &probe = drawPushConstants_.front();
    NURI_LOG_WARNING(
        "TransparentRenderer::prepareTransparentPasses probe: "
        "frameData=0x%llx vertex=0x%llx vertexDecode=0x%llx "
        "instanceMatrices=0x%llx instanceRemap=0x%llx "
        "dirLights=0x%llx localLights=0x%llx materialHeader=0x%llx "
        "materialTransmission=0x%llx materialIndex=%u decodeIndex=%u "
        "packedFormat=%u draws=%zu deps=%zu",
        static_cast<unsigned long long>(probe.frameDataAddress),
        static_cast<unsigned long long>(probe.vertexBufferAddress),
        static_cast<unsigned long long>(probe.vertexDecodeBufferAddress),
        static_cast<unsigned long long>(probe.instanceMatricesAddress),
        static_cast<unsigned long long>(probe.instanceRemapAddress),
        static_cast<unsigned long long>(
            sceneGpu->directionalLightBufferAddress),
        static_cast<unsigned long long>(sceneGpu->localLightBufferAddress),
        static_cast<unsigned long long>(materialGpu->headerBufferAddress),
        static_cast<unsigned long long>(materialGpu->transmissionBufferAddress),
        probe.materialIndex, probe.vertexDecodeIndex, probe.packedVertexFormat,
        meshSortableDraws_.size(), passDependencyBuffers_.size());
  }

  frame.metrics.transparent.meshDraws =
      saturateToU32(meshSortableDraws_.size());
  frame.metrics.transparent.pickDraws = saturateToU32(pickDrawItems_.size());
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentRenderer::appendTransparentMainPass(RenderFrameContext &frame,
                                               RenderGraphBuilder &graph) {
  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  const TextureHandle colorTexture = resolveFrameColorTexture(frame);
  const RenderGraphTextureId sceneDepthGraphTexture =
      resolveSceneDepthGraphTexture(frame);

  return appendTransparentPass(
      graph, colorTexture, depthTexture, sceneDepthGraphTexture,
      std::span<const TransparentStageSortableDraw>(sortableDraws_.data(),
                                                    sortableDraws_.size()),
      std::span<const DrawItem>(fixedDraws_.data(), fixedDraws_.size()),
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
  meshShader_ = Shader::create("transparent_main", gpu_);
  meshPickShader_ = Shader::create("transparent_main_pick", gpu_);
  if (!meshShader_ || !meshPickShader_) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::createShaders: failed to create shader wrappers");
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
TransparentRenderer::ensureRingBufferCount(uint32_t requiredCount) {
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
TransparentRenderer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
  bool needsResize = false;
  bool hasLiveBuffers = false;
  for (const DynamicBufferSlot &slot : instanceMatricesRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      hasLiveBuffers = true;
      if (slot.capacityBytes < requiredBytes) {
        needsResize = true;
        break;
      }
    }
  }
  if (needsResize && hasLiveBuffers) {
    gpu_.waitIdle();
  }
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
TransparentRenderer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
  bool needsResize = false;
  bool hasLiveBuffers = false;
  for (const DynamicBufferSlot &slot : instanceRemapRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      hasLiveBuffers = true;
      if (slot.capacityBytes < requiredBytes) {
        needsResize = true;
        break;
      }
    }
  }
  if (needsResize && hasLiveBuffers) {
    gpu_.waitIdle();
  }
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
TransparentRenderer::rebuildSceneCache(const RenderScene &scene,
                                       const ResourceManager &resources,
                                       uint32_t materialCount) {
  meshDrawTemplates_.clear();

  const std::span<const Renderable> renderables = scene.renderables();
  if (renderables.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::rebuildSceneCache: renderables count exceeds "
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
          "TransparentRenderer::rebuildSceneCache: failed to resolve model");
    }
    GeometryAllocationView geometry{};
    if (!gpu_.resolveGeometry(modelRecord->model->geometryHandle(), geometry)) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::rebuildSceneCache: failed to resolve geometry");
    }
    const uint64_t vertexBufferAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (vertexBufferAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::rebuildSceneCache: invalid vertex buffer "
          "address");
    }

    const std::span<const Submesh> submeshes = modelRecord->model->submeshes();
    for (size_t submeshIndex = 0; submeshIndex < submeshes.size();
         ++submeshIndex) {
      const MaterialRef resolvedMaterial = resolveRenderableMaterial(
          renderable, *modelRecord, static_cast<uint32_t>(submeshIndex));
      const MaterialRecord *materialRecord = resources.tryGet(resolvedMaterial);
      if (materialRecord == nullptr ||
          materialRecord->desc.alphaMode != MaterialAlphaMode::Blend ||
          isTransmissionMaterial(*materialRecord)) {
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
          .baseVertexBuffer = geometry.vertexBuffer,
          .vertexBufferByteOffset = geometry.vertexByteOffset,
          .vertexBuffer = geometry.vertexBuffer,
          .baseVertexDecodeBuffer = modelRecord->model->vertexDecodeBuffer(),
          .vertexBufferAddress = vertexBufferAddress,
          .vertexDecodeBufferAddress =
              modelRecord->model->vertexDecodeBufferAddress(),
          .vertexDecodeIndex = static_cast<uint32_t>(submeshIndex),
          .packedVertexFormat =
              static_cast<uint32_t>(modelRecord->model->drawVertexFormat()),
          .materialIndex = materialIndex,
          .instanceIndex = renderableIndex,
          .doubleSided = materialRecord->desc.doubleSided,
      });
    }
  }

  if (invalidMaterialFallbackCount > 0u) {
    if (!loggedMaterialFallbackWarning_) {
      NURI_LOG_WARNING(
          "TransparentRenderer::rebuildSceneCache: %zu submesh draw(s) used "
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

Result<bool, std::string>
TransparentRenderer::rebuildMaterialTextureAccessCache(
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
    const MaterialRef resolvedMaterial = resolveRenderableMaterial(
        *entry.renderable, *modelRecord, entry.submeshIndex);
    const MaterialRecord *materialRecord = resources.tryGet(resolvedMaterial);
    if (materialRecord == nullptr) {
      continue;
    }

    forEachMaterialTextureRef(materialRecord->textureRefs, [&](TextureRef ref) {
      const TextureRecord *record = resources.tryGet(ref);
      if (record == nullptr || !nuri::isValid(record->texture)) {
        return;
      }
      appendUniqueTexture(materialTextureAccessHandles_, record->texture);
    });
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransparentRenderer::collectContributorDraws(RenderFrameContext &frame) {
  contributorSortableDraws_.clear();
  contributorFixedDraws_.clear();
  contributorTextureReads_.clear();
  contributorSortableDraws_.reserve(16u);
  contributorFixedDraws_.reserve(8u);
  contributorTextureReads_.reserve(8u);

  if (frame.transparentContributors.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  static uint32_t loggedContributorCollections = 0u;
  if (loggedContributorCollections < 16u) {
    NURI_LOG_WARNING("TransparentRenderer::collectContributorDraws: frame=%llu "
                     "collectorCount=%zu",
                     static_cast<unsigned long long>(frame.frameIndex),
                     frame.transparentContributors.collectors().size());
  }

  for (const TransparentContributionCollector &collector :
       frame.transparentContributors.collectors()) {
    if (collector.user == nullptr || collector.collect == nullptr) {
      continue;
    }

    TransparentStageContribution contribution{};
    auto contributionResult =
        collector.collect(collector.user, frame, contribution);
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
  }

  if (loggedContributorCollections < 16u) {
    NURI_LOG_WARNING(
        "TransparentRenderer::collectContributorDraws result: frame=%llu "
        "sortable=%zu fixed=%zu textures=%zu",
        static_cast<unsigned long long>(frame.frameIndex),
        contributorSortableDraws_.size(), contributorFixedDraws_.size(),
        contributorTextureReads_.size());
    ++loggedContributorCollections;
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransparentRenderer::appendTransparentPass(
    RenderGraphBuilder &graph, TextureHandle colorTexture,
    TextureHandle depthTexture, RenderGraphTextureId sceneDepthGraphTexture,
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
  passDesc.draws =
      std::span<const DrawItem>(passDrawItems_.data(), passDrawItems_.size());
  std::pmr::vector<BufferHandle> preResolvedDrawBuffers(memory_);
  preResolvedDrawBuffers.reserve(meshDrawTemplates_.size() * 3u +
                                 passDrawItems_.size() * 2u + 8u);
  for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
    appendUniqueBuffer(preResolvedDrawBuffers, entry.baseVertexBuffer);
    appendUniqueBuffer(preResolvedDrawBuffers, entry.baseVertexDecodeBuffer);
    appendUniqueBuffer(preResolvedDrawBuffers, entry.indexBuffer);
  }
  appendPreResolvedDrawBuffers(
      preResolvedDrawBuffers,
      std::span<const DrawItem>(passDrawItems_.data(), passDrawItems_.size()));
  passDesc.drawBuffersPreResolved = true;
  passDesc.preResolvedDrawBuffers = std::span<const BufferHandle>(
      preResolvedDrawBuffers.data(), preResolvedDrawBuffers.size());
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
    appendUniqueBuffer(preResolvedDrawBuffers, entry.baseVertexBuffer);
    appendUniqueBuffer(preResolvedDrawBuffers, entry.baseVertexDecodeBuffer);
    appendUniqueBuffer(preResolvedDrawBuffers, entry.indexBuffer);
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

void TransparentRenderer::collectEnvironmentTextureReads(
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

void TransparentRenderer::resetCachedState() {
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  loggedMaterialFallbackWarning_ = false;
  loggedAddressProbeTopologyVersion_ = std::numeric_limits<uint64_t>::max();

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

void TransparentRenderer::destroyBuffers() {
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
  if (doubleSided && nuri::isValid(meshDoubleSidedPipelineHandle_)) {
    return meshDoubleSidedPipelineHandle_;
  }
  return meshPipelineHandle_;
}

RenderPipelineHandle
TransparentRenderer::selectPickPipeline(bool doubleSided) const {
  if (doubleSided && nuri::isValid(meshPickDoubleSidedPipelineHandle_)) {
    return meshPickDoubleSidedPipelineHandle_;
  }
  return meshPickPipelineHandle_;
}

} // namespace nuri
