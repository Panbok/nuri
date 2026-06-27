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
constexpr std::string_view kTransparentTransmissionPassLabel =
    "Transparent Transmission Pass";
constexpr std::string_view kTransparentTransmissionFeedbackCopyLabel =
    "Transparent Transmission Feedback Copy";
constexpr std::string_view kTransparentTransmissionFeedbackHalfLabel =
    "Transparent Transmission Feedback Half";
constexpr std::string_view kTransparentTransmissionFeedbackQuarterLabel =
    "Transparent Transmission Feedback Quarter";
constexpr std::string_view kTransparentPickPassLabel = "Transparent Pick Pass";
constexpr std::string_view kTransparentMeshLabel = "TransparentMesh";
constexpr std::string_view kTransparentMeshPickLabel = "TransparentMeshPick";
constexpr uint32_t kSceneCopyFlagDownsample = 1u << 0u;
constexpr uint32_t kMaxExactTransmissionFeedbackDraws = 64u;

struct CopyPushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t flags = 0u;
  uint32_t reserved0 = 0u;
};
static_assert(sizeof(CopyPushConstants) <= 128);

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

void publishRequestedCapture(RenderFrameContext &frame, GPUDevice &gpu,
                             std::string_view name, TextureHandle texture,
                             RenderCaptureValueKind kind,
                             RenderCaptureLifetimeClass lifetime,
                             std::string_view colorSpace,
                             std::string_view compareProfile,
                             std::string_view producerPassLabel) {
  if (!isRenderCaptureRequested(frame, name) || !nuri::isValid(texture)) {
    return;
  }
  frame.captureRegistry.publish(RenderCapturePoint{
      .name = name,
      .version = 1u,
      .texture = texture,
      .format = gpu.getTextureFormat(texture),
      .dimensions = gpu.getTextureDimensions(texture),
      .frameIndex = frame.frameIndex,
      .mip = 0u,
      .layer = 0u,
      .kind = kind,
      .lifetime = lifetime,
      .colorSpace = colorSpace,
      .defaultCompareProfile = compareProfile,
      .producerPassLabel = producerPassLabel,
      .debugLabel = name,
  });
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

RenderPipelineDesc fullscreenPipelineDesc(Format colorFormat,
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

uint32_t saturateToU32(size_t value) {
  return static_cast<uint32_t>(
      std::min(value, size_t(std::numeric_limits<uint32_t>::max())));
}

[[nodiscard]] const RenderSettings &
settingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? *frame.settings : kDefaultSettings;
}

[[nodiscard]] bool
isAntiAliasingDebugOutputView(AntiAliasingDebugView view) noexcept {
  return view != AntiAliasingDebugView::None &&
         view != AntiAliasingDebugView::Settings;
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
  feedbackCopyVertexPath_ = basePath / "fullscreen_copy.vert";
  feedbackCopyFragmentPath_ = basePath / "scene_copy.frag";
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
    auto rebuildResult = rebuildSceneCache(
        *frame.scene, *frame.resources,
        static_cast<uint32_t>(materialSnapshot.headers.size()),
        excludeTransmissionBlend);
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
    for (const FixedDrawEntry &draw : contributorFixedDraws_) {
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

  const uint64_t frameDataAddress = sceneGpu->postTaaFrameDataAddress != 0u
                                        ? sceneGpu->postTaaFrameDataAddress
                                        : sceneGpu->frameDataAddress;
  transparentUsesJitteredProjection_ =
      frameDataAddress == sceneGpu->frameDataAddress;
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
      (sceneGpu->localLightCount > 0u && localLightBufferAddress == 0u) ||
      ((sceneGpu->shadowFlags & kShadowFrameFlagEnabled) != 0u &&
       sceneGpu->shadowFrameBufferAddress == 0u)) {
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
        .flags = source.flags,
        .dependencyOffset = source.dependencyOffset,
        .dependencyCount = source.dependencyCount,
    });
  }
  for (const FixedDrawEntry &draw : contributorFixedDraws_) {
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
  if ((sceneGpu->shadowFlags & kShadowFrameFlagEnabled) != 0u) {
    const auto &shadowFrameGpuData = frame.sharedResources.shadowFrameGpuData;
    const bool hasShadowResources =
        shadowFrameGpuData.has_value() &&
        nuri::isValid(shadowFrameGpuData->buffer) &&
        shadowFrameGpuData->bufferAddress != 0u &&
        shadowFrameGpuData->bufferAddress == sceneGpu->shadowFrameBufferAddress;
    bool hasShadowTexture = false;
    for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
         ++cascadeIndex) {
      hasShadowTexture =
          hasShadowTexture ||
          nuri::isValid(
              frame.sharedResources.shadowCascadeTextures[cascadeIndex]);
    }
    if (!hasShadowResources) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::prepareTransparentPasses: missing shadow "
          "resources");
    }
    if (!hasShadowTexture) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::prepareTransparentPasses: missing shadow "
          "cascade textures");
    }
    appendUniqueBuffer(passDependencyBuffers_, shadowFrameGpuData->buffer);
    for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
         ++cascadeIndex) {
      const TextureHandle texture =
          frame.sharedResources.shadowCascadeTextures[cascadeIndex];
      if (nuri::isValid(texture)) {
        appendUniqueTexture(passTextureReads_, texture);
      }
    }
  }
  if (loggedAddressProbeTopologyVersion_ != frame.scene->topologyVersion() &&
      !drawPushConstants_.empty()) {
    loggedAddressProbeTopologyVersion_ = frame.scene->topologyVersion();
    const PushConstants &probe = drawPushConstants_.front();
    NURI_LOG_TRACE(
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

  const bool hasFeedbackDraw = std::any_of(
      sortableDraws_.begin(), sortableDraws_.end(),
      [](const TransparentStageSortableDraw &draw) {
        return (draw.flags &
                kTransparentStageDrawFlagRequiresFrameColorFeedback) != 0u;
      });
  if (hasFeedbackDraw) {
    const TextureHandle feedbackTexture =
        frame.sharedResources.transparentTransmissionFeedbackTexture;
    if (!nuri::isValid(feedbackTexture)) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::prepareTransparentPasses: transparent "
          "transmission feedback texture is unavailable");
    }
    auto feedbackPipelineResult =
        ensureFeedbackCopyPipeline(gpu_.getTextureFormat(feedbackTexture));
    if (feedbackPipelineResult.hasError()) {
      return feedbackPipelineResult;
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
  const TextureHandle colorTexture = resolveFrameColorTexture(frame);
  const RenderGraphTextureId sceneDepthGraphTexture =
      resolveSceneDepthGraphTexture(frame);
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
    aaMetrics.taaTransparentEdgeJitterTracked = true;
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
  meshShader_ = Shader::create("transparent_main", gpu_);
  meshPickShader_ = Shader::create("transparent_main_pick", gpu_);
  feedbackCopyShader_ =
      Shader::create("transparent_transmission_feedback", gpu_);
  if (!meshShader_ || !meshPickShader_ || !feedbackCopyShader_) {
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
  auto pickVertexResult = meshPickShader_->compileFromFile(
      (alphaPickFragmentPath_.parent_path() / "main_id.vert").string(),
      ShaderStage::Vertex);
  if (pickVertexResult.hasError()) {
    return Result<bool, std::string>::makeError(pickVertexResult.error());
  }
  auto copyVertexResult = feedbackCopyShader_->compileFromFile(
      feedbackCopyVertexPath_.string(), ShaderStage::Vertex);
  if (copyVertexResult.hasError()) {
    return Result<bool, std::string>::makeError(copyVertexResult.error());
  }
  auto copyFragmentResult = feedbackCopyShader_->compileFromFile(
      feedbackCopyFragmentPath_.string(), ShaderStage::Fragment);
  if (copyFragmentResult.hasError()) {
    return Result<bool, std::string>::makeError(copyFragmentResult.error());
  }

  meshVertexShader_ = vertexResult.value();
  meshFragmentShader_ = fragmentResult.value();
  meshPickVertexShader_ = pickVertexResult.value();
  meshPickFragmentShader_ = pickResult.value();
  feedbackCopyVertexShader_ = copyVertexResult.value();
  feedbackCopyFragmentShader_ = copyFragmentResult.value();
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
      meshPipelineDesc(Format::R32_UINT, depthFormat, meshPickVertexShader_,
                       meshPickFragmentShader_, false, CullMode::Back),
      "transparent_mesh_pick");
  if (pickResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(pickResult.error());
  }
  meshPickPipelineHandle_ = pickResult.value();

  auto pickDoubleResult = gpu_.createRenderPipeline(
      meshPipelineDesc(Format::R32_UINT, depthFormat, meshPickVertexShader_,
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
TransparentRenderer::ensureFeedbackCopyPipeline(Format colorFormat) {
  if (nuri::isValid(feedbackCopyPipelineHandle_) &&
      feedbackCopyPipelineColorFormat_ == colorFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!nuri::isValid(feedbackCopyVertexShader_) ||
      !nuri::isValid(feedbackCopyFragmentShader_)) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::ensureFeedbackCopyPipeline: invalid shader "
        "handle");
  }
  if (nuri::isValid(feedbackCopyPipelineHandle_)) {
    gpu_.destroyRenderPipeline(feedbackCopyPipelineHandle_);
    feedbackCopyPipelineHandle_ = {};
  }
  auto pipelineResult = gpu_.createRenderPipeline(
      fullscreenPipelineDesc(colorFormat, feedbackCopyVertexShader_,
                             feedbackCopyFragmentShader_),
      "transparent_transmission_feedback_copy");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  feedbackCopyPipelineHandle_ = pipelineResult.value();
  feedbackCopyPipelineColorFormat_ = colorFormat;
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

Result<bool, std::string> TransparentRenderer::rebuildSceneCache(
    const RenderScene &scene, const ResourceManager &resources,
    uint32_t materialCount, bool excludeTransmissionBlend) {
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
      const bool isTransmission =
          materialRecord != nullptr && isTransmissionMaterial(*materialRecord);
      if (materialRecord == nullptr ||
          materialRecord->desc.alphaMode != MaterialAlphaMode::Blend ||
          (isTransmission && excludeTransmissionBlend)) {
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
  cachedExcludeTransmissionBlend_ = excludeTransmissionBlend;
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
  contributorDependencyBuffers_.clear();
  contributorSortableDraws_.reserve(16u);
  contributorFixedDraws_.reserve(8u);
  contributorTextureReads_.reserve(8u);
  contributorDependencyBuffers_.reserve(16u);

  if (frame.transparentContributors.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (loggedContributorCollections_ < 16u) {
    NURI_LOG_DEBUG("TransparentRenderer::collectContributorDraws: frame=%llu "
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
      contributorSortableDraws_.push_back(TransparentStageSortableDraw{
          .draw = draw,
          .sortDepth = source.sortDepth,
          .stableOrder = stableOrderBase + source.stableOrder,
          .flags = source.flags,
          .dependencyOffset = dependencyOffset,
          .dependencyCount = dependencyCount,
      });
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
      appendUniqueTexture(contributorTextureReads_, handle);
    }

    frame.metrics.transparent.contributorSortableDraws +=
        saturateToU32(contribution.sortableDraws.size());
    frame.metrics.transparent.contributorFixedDraws +=
        saturateToU32(contribution.fixedDraws.size());
  }

  if (loggedContributorCollections_ < 16u) {
    NURI_LOG_DEBUG(
        "TransparentRenderer::collectContributorDraws result: frame=%llu "
        "sortable=%zu fixed=%zu textures=%zu",
        static_cast<unsigned long long>(frame.frameIndex),
        contributorSortableDraws_.size(), contributorFixedDraws_.size(),
        contributorTextureReads_.size());
    ++loggedContributorCollections_;
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
  const uint32_t feedbackDrawCount = static_cast<uint32_t>(std::count_if(
      sortableDraws.begin(), sortableDraws.end(),
      [](const TransparentStageSortableDraw &draw) {
        return (draw.flags &
                kTransparentStageDrawFlagRequiresFrameColorFeedback) != 0u;
      }));
  const bool exactFeedbackPerDraw =
      feedbackDrawCount <= kMaxExactTransmissionFeedbackDraws;
  bool feedbackFallbackSourceReady = false;
  if (!exactFeedbackPerDraw && !loggedTransmissionFeedbackFallbackWarning_) {
    loggedTransmissionFeedbackFallbackWarning_ = true;
    NURI_LOG_WARNING(
        "TransparentRenderer::appendTransparentPass: %u blended "
        "transmission draw(s) exceed exact feedback budget %u; using one "
        "shared feedback refresh for this frame",
        feedbackDrawCount, kMaxExactTransmissionFeedbackDraws);
  }
  const auto resetRunDependencies = [this, dependencyBuffers]() {
    transparentRunDependencyBuffers_.clear();
    for (const BufferHandle handle : dependencyBuffers) {
      appendUniqueBuffer(transparentRunDependencyBuffers_, handle);
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
      appendUniqueBuffer(target, contributorDependencyBuffers_[index]);
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
    const bool requiresFeedback =
        (draw.flags & kTransparentStageDrawFlagRequiresFrameColorFeedback) !=
        0u;
    if (!requiresFeedback) {
      auto dependencyResult = prepareRunDependenciesForDraw(
          draw.dependencyOffset, draw.dependencyCount);
      if (dependencyResult.hasError()) {
        return dependencyResult;
      }
      transparentRunDrawItems_.push_back(draw.draw);
      continue;
    }

    if (!exactFeedbackPerDraw) {
      if (!feedbackFallbackSourceReady) {
        auto flushResult = flushTransparentRun(kTransparentPassLabel);
        if (flushResult.hasError()) {
          return flushResult;
        }
        auto feedbackResult =
            appendTransparentTransmissionFeedbackRefresh(frame, graph);
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
        appendTransparentTransmissionFeedbackRefresh(frame, graph);
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
    appendUniqueBuffer(preResolvedDrawBuffers, entry.baseVertexBuffer);
    appendUniqueBuffer(preResolvedDrawBuffers, entry.baseVertexDecodeBuffer);
    appendUniqueBuffer(preResolvedDrawBuffers, entry.indexBuffer);
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
TransparentRenderer::appendTransparentTransmissionFeedbackRefresh(
    RenderFrameContext &frame, RenderGraphBuilder &graph) {
  const TextureHandle frameColorTexture = resolveFrameColorTexture(frame);
  const TextureHandle feedbackFull =
      frame.sharedResources.transparentTransmissionFeedbackTexture;
  const TextureHandle feedbackHalf =
      frame.sharedResources.transparentTransmissionFeedbackHalfResTexture;
  const TextureHandle feedbackQuarter =
      frame.sharedResources.transparentTransmissionFeedbackQuarterResTexture;
  if (!nuri::isValid(frameColorTexture) || !nuri::isValid(feedbackFull) ||
      !nuri::isValid(feedbackHalf) || !nuri::isValid(feedbackQuarter)) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::appendTransparentTransmissionFeedbackRefresh: "
        "transparent transmission feedback pyramid is incomplete");
  }
  auto pipelineResult =
      ensureFeedbackCopyPipeline(gpu_.getTextureFormat(feedbackFull));
  if (pipelineResult.hasError()) {
    return pipelineResult;
  }
  if (!nuri::isValid(feedbackCopyPipelineHandle_)) {
    return Result<bool, std::string>::makeError(
        "TransparentRenderer::appendTransparentTransmissionFeedbackRefresh: "
        "feedback copy pipeline is invalid");
  }

  const auto appendCopyPass =
      [this, &graph](TextureHandle source, TextureHandle destination,
                     std::string_view debugLabel,
                     bool downsample) -> Result<bool, std::string> {
    const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(source);
    if (sourceTexId == kInvalidTextureBindlessIndex) {
      return Result<bool, std::string>::makeError(
          "TransparentRenderer::appendTransparentTransmissionFeedbackRefresh: "
          "invalid source texture bindless index");
    }

    const CopyPushConstants pushConstants{
        .sourceTexId = sourceTexId,
        .sourceSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u),
        .flags = downsample ? kSceneCopyFlagDownsample : 0u,
        .reserved0 = 0u,
    };
    const DrawItem draw{
        .command = DrawCommandType::Direct,
        .pipeline = feedbackCopyPipelineHandle_,
        .vertexCount = 3u,
        .instanceCount = 1u,
        .pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&pushConstants),
            sizeof(pushConstants)),
        .debugLabel = debugLabel,
        .debugColor = kTransparentPassDebugColor,
    };

    auto colorImportResult =
        graph.importTexture(destination, "transparent_transmission_feedback");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    RenderGraphGraphicsPassDesc passDesc{};
    passDesc.color = {.loadOp = LoadOp::Clear,
                      .storeOp = StoreOp::Store,
                      .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    passDesc.colorTexture = colorImportResult.value();
    passDesc.draws = std::span<const DrawItem>(&draw, 1u);
    passDesc.dependencyTextures = std::span<const TextureHandle>(&source, 1u);
    passDesc.debugLabel = debugLabel;
    passDesc.debugColor = kTransparentPassDebugColor;

    auto addResult = graph.addGraphicsPass(passDesc);
    if (addResult.hasError()) {
      return Result<bool, std::string>::makeError(addResult.error());
    }

    return Result<bool, std::string>::makeResult(true);
  };

  auto fullResult =
      appendCopyPass(frameColorTexture, feedbackFull,
                     kTransparentTransmissionFeedbackCopyLabel, false);
  if (fullResult.hasError()) {
    return fullResult;
  }
  auto halfResult =
      appendCopyPass(feedbackFull, feedbackHalf,
                     kTransparentTransmissionFeedbackHalfLabel, true);
  if (halfResult.hasError()) {
    return halfResult;
  }
  auto quarterResult =
      appendCopyPass(feedbackHalf, feedbackQuarter,
                     kTransparentTransmissionFeedbackQuarterLabel, true);
  if (quarterResult.hasError()) {
    return quarterResult;
  }

  publishRequestedCapture(frame, gpu_, "transmission_feedback", feedbackFull,
                          RenderCaptureValueKind::LinearHdrColor,
                          RenderCaptureLifetimeClass::FeaturePersistentTexture,
                          "linear_hdr", "hdr_color",
                          kTransparentTransmissionFeedbackCopyLabel);
  publishRequestedCapture(frame, gpu_, "transmission_feedback_half",
                          feedbackHalf, RenderCaptureValueKind::LinearHdrColor,
                          RenderCaptureLifetimeClass::FeaturePersistentTexture,
                          "linear_hdr", "hdr_color",
                          kTransparentTransmissionFeedbackHalfLabel);
  publishRequestedCapture(
      frame, gpu_, "transmission_feedback_quarter", feedbackQuarter,
      RenderCaptureValueKind::LinearHdrColor,
      RenderCaptureLifetimeClass::FeaturePersistentTexture, "linear_hdr",
      "hdr_color", kTransparentTransmissionFeedbackQuarterLabel);
  ++frame.metrics.antiAliasing.transparentTransmissionFeedbackRefreshCount;
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
  cachedExcludeTransmissionBlend_ = true;
  loggedMaterialFallbackWarning_ = false;
  loggedTransmissionFeedbackFallbackWarning_ = false;
  loggedContributorCollections_ = 0u;
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
  transparentUsesJitteredProjection_ = true;
}

void TransparentRenderer::destroyPipelineState() {
  if (nuri::isValid(feedbackCopyPipelineHandle_)) {
    gpu_.destroyRenderPipeline(feedbackCopyPipelineHandle_);
  }
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
  feedbackCopyPipelineHandle_ = {};
  meshPipelineColorFormat_ = Format::Count;
  meshPipelineDepthFormat_ = Format::Count;
  pickPipelineDepthFormat_ = Format::Count;
  feedbackCopyPipelineColorFormat_ = Format::Count;
}

void TransparentRenderer::destroyShaders() {
  if (nuri::isValid(feedbackCopyVertexShader_)) {
    gpu_.destroyShaderModule(feedbackCopyVertexShader_);
  }
  if (nuri::isValid(feedbackCopyFragmentShader_)) {
    gpu_.destroyShaderModule(feedbackCopyFragmentShader_);
  }
  if (nuri::isValid(meshVertexShader_)) {
    gpu_.destroyShaderModule(meshVertexShader_);
  }
  if (nuri::isValid(meshFragmentShader_)) {
    gpu_.destroyShaderModule(meshFragmentShader_);
  }
  if (nuri::isValid(meshPickVertexShader_)) {
    gpu_.destroyShaderModule(meshPickVertexShader_);
  }
  if (nuri::isValid(meshPickFragmentShader_)) {
    gpu_.destroyShaderModule(meshPickFragmentShader_);
  }
  meshVertexShader_ = {};
  meshFragmentShader_ = {};
  meshPickVertexShader_ = {};
  meshPickFragmentShader_ = {};
  feedbackCopyVertexShader_ = {};
  feedbackCopyFragmentShader_ = {};
  feedbackCopyShader_.reset();
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
