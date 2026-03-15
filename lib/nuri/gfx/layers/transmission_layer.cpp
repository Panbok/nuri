#include "nuri/pch.h"

#include "nuri/gfx/layers/transmission_layer.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

constexpr uint32_t kTransmissionPassDebugColor = 0x33ffaaeeu;
constexpr uint32_t kTransmissionMeshDebugColor = 0x33ffaaeeu;
constexpr uint32_t kTransmissionCopyDebugColor = 0x3388ffffu;
constexpr uint32_t kTransmissionDownsampleDebugColor = 0x3388ffaau;
constexpr std::string_view kTransmissionPassLabel = "Transmission Pass";
constexpr std::string_view kTransmissionCopyPassLabel =
    "Transmission Copy Pass";
constexpr std::string_view kTransmissionDownsamplePassLabel =
    "Transmission Downsample Pass";
constexpr std::string_view kTransmissionMeshLabel = "TransmissionMesh";
constexpr std::string_view kTransmissionCopyLabel = "TransmissionCopy";
constexpr std::string_view kTransmissionDownsampleLabel =
    "TransmissionDownsample";
constexpr uint32_t kTransmissionSceneColorLevelCount = 3u;
constexpr uint32_t kTransmissionCopyFlagDownsample = 1u << 0u;

[[nodiscard]] std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}

[[nodiscard]] const RenderSettings &
settingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? *frame.settings : kDefaultSettings;
}

[[nodiscard]] bool isTransmissionMaterial(const MaterialRecord &material) {
  return material.desc.alphaMode != MaterialAlphaMode::Blend &&
         (material.desc.featureMask & kMaterialFeatureTransmission) != 0u;
}

void appendUniqueTexture(std::pmr::vector<TextureHandle> &handles,
                         TextureHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  if (std::find_if(handles.begin(), handles.end(),
                   [handle](const TextureHandle existing) {
                     return existing.index == handle.index &&
                            existing.generation == handle.generation;
                   }) != handles.end()) {
    return;
  }
  handles.push_back(handle);
}

void appendUniqueBuffer(std::pmr::vector<BufferHandle> &handles,
                        BufferHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  if (std::find_if(handles.begin(), handles.end(),
                   [handle](const BufferHandle existing) {
                     return existing.index == handle.index &&
                            existing.generation == handle.generation;
                   }) != handles.end()) {
    return;
  }
  handles.push_back(handle);
}

[[nodiscard]] std::optional<SubmeshLod>
resolveTransmissionLod(const Submesh &submesh, const RenderSettings &settings) {
  if (settings.opaque.forcedMeshLod < 0) {
    if (submesh.indexCount > 0) {
      return SubmeshLod{.indexOffset = submesh.indexOffset,
                        .indexCount = submesh.indexCount,
                        .error = 0.0f};
    }
    for (uint32_t lod = 0; lod < std::max(submesh.lodCount, 1u); ++lod) {
      if (submesh.lods[lod].indexCount > 0u) {
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

RenderPipelineDesc meshPipelineDesc(Format colorFormat, Format depthFormat,
                                    ShaderHandle vertexShader,
                                    ShaderHandle fragmentShader,
                                    CullMode cullMode) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {colorFormat},
      .depthFormat = depthFormat,
      .cullMode = cullMode,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
  };
}

RenderPipelineDesc copyPipelineDesc(Format colorFormat, Format depthFormat,
                                    ShaderHandle vertexShader,
                                    ShaderHandle fragmentShader) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {colorFormat},
      .depthFormat = depthFormat,
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

uint32_t levelDimensions(uint32_t base, uint32_t level) {
  return std::max(1u, base >> std::min(level, 31u));
}

[[nodiscard]] bool allTexturesValid(std::span<const TextureHandle> textures) {
  for (const TextureHandle texture : textures) {
    if (!nuri::isValid(texture)) {
      return false;
    }
  }
  return true;
}

void destroyTextures(GPUDevice &gpu,
                     std::pmr::vector<TextureHandle> &textures) {
  for (TextureHandle &texture : textures) {
    if (!nuri::isValid(texture)) {
      continue;
    }
    gpu.destroyTexture(texture);
    texture = {};
  }
  textures.clear();
}

[[nodiscard]] size_t ringSlot(uint64_t frameIndex, size_t slotCount) {
  NURI_ASSERT(slotCount > 0u, "TransmissionLayer ring slot count must be > 0");
  return static_cast<size_t>(frameIndex % slotCount);
}

} // namespace

TransmissionLayer::TransmissionLayer(GPUDevice &gpu,
                                     TransmissionLayerConfig config,
                                     std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(resolveMemoryResource(memory)), instanceMatricesRing_(memory_),
      instanceRemapRing_(memory_), meshDrawTemplates_(memory_),
      instanceMatrices_(memory_), instanceRemap_(memory_),
      instanceDataRingUploadVersions_(memory_), materialGpuDataCache_(memory_),
      materialTextureAccessHandles_(memory_),
      environmentTextureAccessHandles_(memory_), meshPushConstants_(memory_),
      passDrawItems_(memory_), passTextureReads_(memory_),
      passDependencyBuffers_(memory_), copyPushConstantsRing_(memory_),
      sceneColorTextures_(memory_), sceneColorMipTextures_(memory_) {
  const std::filesystem::path basePath = config_.meshFragment.parent_path();
  transmissionFragmentPath_ = basePath / "transmission.frag";
  fullscreenCopyVertexPath_ = basePath / "fullscreen_copy.vert";
  sceneCopyFragmentPath_ = basePath / "scene_copy.frag";
}

TransmissionLayer::~TransmissionLayer() { onDetach(); }

void TransmissionLayer::onAttach() {
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    NURI_LOG_WARNING("TransmissionLayer::onAttach: %s",
                     initResult.error().c_str());
  }
}

void TransmissionLayer::onDetach() {
  destroyBuffers();
  destroyPipelineState();
  destroyShaders();
  meshShader_.reset();
  copyShader_.reset();
  resetFrameBuildState();
  resetCachedState();
  destroyTextures(gpu_, sceneColorTextures_);
  destroyTextures(gpu_, sceneColorMipTextures_);
  sceneColorTextureFormat_ = Format::Count;
  sceneColorTextureWidth_ = 0;
  sceneColorTextureHeight_ = 0;
  initialized_ = false;
}

void TransmissionLayer::publishFrameData(RenderFrameContext &frame) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.transmission.enabled || !hasTransmissionContent(frame)) {
    return;
  }
  auto sceneColorResult = ensureSceneColorTexture();
  if (sceneColorResult.hasError()) {
    NURI_LOG_WARNING("TransmissionLayer::publishFrameData: %s",
                     sceneColorResult.error().c_str());
    return;
  }
  const TextureHandle sceneColorTexture =
      currentSceneColorTexture(frame.frameIndex);
  if (!nuri::isValid(sceneColorTexture)) {
    NURI_LOG_WARNING("TransmissionLayer::publishFrameData: scene color ring "
                     "slot is unavailable");
    return;
  }
  frame.channels.publish<bool>(kFrameChannelTransmissionStageEnabled, true);
  frame.channels.publish<TextureHandle>(kFrameChannelSceneColorTexture,
                                        sceneColorTexture);
}

Result<bool, std::string>
TransmissionLayer::buildRenderGraph(RenderFrameContext &frame,
                                    RenderGraphBuilder &graph) {
  NURI_PROFILER_FUNCTION();
  resetFrameBuildState();

  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.transmission.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!frame.scene) {
    return Result<bool, std::string>::makeError(
        "TransmissionLayer::buildRenderGraph: frame scene is null");
  }
  if (!frame.resources) {
    return Result<bool, std::string>::makeError(
        "TransmissionLayer::buildRenderGraph: frame resources are null");
  }

  const bool *transmissionStageEnabled =
      frame.channels.tryGet<bool>(kFrameChannelTransmissionStageEnabled);
  if (transmissionStageEnabled == nullptr || !*transmissionStageEnabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return initResult;
  }
  auto sceneColorResult = ensureSceneColorTexture();
  if (sceneColorResult.hasError()) {
    return sceneColorResult;
  }

  const TextureHandle *sceneColorTexture =
      frame.channels.tryGet<TextureHandle>(kFrameChannelSceneColorTexture);
  if (sceneColorTexture == nullptr || !nuri::isValid(*sceneColorTexture)) {
    return Result<bool, std::string>::makeError(
        "TransmissionLayer::buildRenderGraph: scene color texture is "
        "unavailable");
  }
  const TextureHandle depthTexture = resolveFrameDepthTexture(frame);
  RenderGraphTextureId sceneDepthGraphTexture{};
  if (const RenderGraphTextureId *published =
          frame.channels.tryGet<RenderGraphTextureId>(
              kFrameChannelSceneDepthGraphTexture);
      published != nullptr) {
    sceneDepthGraphTexture = *published;
  }
  const bool hasSceneColorInput =
      frame.channels.tryGet<RenderGraphTextureId>(
          kFrameChannelSceneColorGraphTexture) != nullptr;

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
    Result<bool, std::string> rebuildResult = [&]() -> Result<bool, std::string> {
      std::optional<Result<bool, std::string>> result;
      NURI_PROFILER_ZONE("TransmissionLayer.cache_rebuild",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      result.emplace(rebuildSceneCache(
          *frame.scene, *frame.resources,
          static_cast<uint32_t>(materialSnapshot.gpuData.size())));
      NURI_PROFILER_ZONE_END();
      return std::move(*result);
    }();
    if (rebuildResult.hasError()) {
      return rebuildResult;
    }
    cachedMaterialVersion_ = materialSnapshot.version;
  } else if (geometryDirty) {
    cachedGeometryMutationVersion_ = geometryMutationVersion;
  }

  {
    NURI_PROFILER_ZONE("TransmissionLayer.env_texture_collect",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    collectEnvironmentTextureReads(*frame.scene, *frame.resources);
    NURI_PROFILER_ZONE_END();
  }
  if (materialDirty || materialTextureAccessHandles_.empty()) {
    Result<bool, std::string> cacheResult = [&]() -> Result<bool, std::string> {
      std::optional<Result<bool, std::string>> result;
      NURI_PROFILER_ZONE("TransmissionLayer.material_texture_cache",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      result.emplace(rebuildMaterialTextureAccessCache(*frame.resources));
      NURI_PROFILER_ZONE_END();
      return std::move(*result);
    }();
    if (cacheResult.hasError()) {
      return cacheResult;
    }
  }

  auto frameDataBufferResult = ensureFrameDataBufferCapacity(sizeof(FrameData));
  if (frameDataBufferResult.hasError()) {
    return frameDataBufferResult;
  }

  const std::span<const Renderable> renderables = frame.scene->renderables();
  if (!meshDrawTemplates_.empty()) {
    auto ringResult =
        ensureRingBufferCount(std::max(1u, gpu_.getSwapchainImageCount()));
    if (ringResult.hasError()) {
      return ringResult;
    }
  }

  const uint32_t frameSlot =
      instanceMatricesRing_.empty()
          ? 0u
          : static_cast<uint32_t>(frame.frameIndex %
                                  instanceMatricesRing_.size());

  if (!meshDrawTemplates_.empty() && transformDirty) {
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
  } else if (meshDrawTemplates_.empty()) {
    cachedTransformVersion_ = frame.scene->transformVersion();
  }

  if (!meshDrawTemplates_.empty()) {
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
  }

  uint32_t cubemapTexId = kInvalidTextureBindlessIndex;
  uint32_t hasCubemap = 0u;
  uint32_t irradianceTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredGgxTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredCharlieTexId = kInvalidTextureBindlessIndex;
  uint32_t brdfLutTexId = kInvalidTextureBindlessIndex;
  uint32_t frameFlags = 0u;
  const uint32_t cubemapSamplerId = gpu_.getCubemapSamplerBindlessIndex();
  const uint32_t sceneColorTexId =
      hasSceneColorInput ? gpu_.getTextureBindlessIndex(*sceneColorTexture)
                         : kInvalidTextureBindlessIndex;
  const uint32_t sceneColorMip1TexId =
      hasSceneColorInput
          ? gpu_.getTextureBindlessIndex(
                currentSceneColorTextureMip(frame.frameIndex, 1u))
          : kInvalidTextureBindlessIndex;
  const uint32_t sceneColorMip2TexId =
      hasSceneColorInput
          ? gpu_.getTextureBindlessIndex(
                currentSceneColorTextureMip(frame.frameIndex, 2u))
          : kInvalidTextureBindlessIndex;
  const uint32_t sceneColorSamplerId = gpu_.getDefaultSamplerBindlessIndex();
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
    frameFlags |= HasIblDiffuse;
  }
  if (const TextureRecord *prefilteredGgx =
          frame.resources->tryGet(environment.prefilteredGgx);
      prefilteredGgx != nullptr && nuri::isValid(prefilteredGgx->texture)) {
    prefilteredGgxTexId = prefilteredGgx->bindlessIndex;
    frameFlags |= HasIblSpecular;
  }
  if (const TextureRecord *prefilteredCharlie =
          frame.resources->tryGet(environment.prefilteredCharlie);
      prefilteredCharlie != nullptr &&
      nuri::isValid(prefilteredCharlie->texture)) {
    prefilteredCharlieTexId = prefilteredCharlie->bindlessIndex;
    frameFlags |= HasIblSheen;
  } else if ((frameFlags & HasIblSpecular) != 0u) {
    prefilteredCharlieTexId = prefilteredGgxTexId;
    frameFlags |= HasIblSheen;
  }
  if (const TextureRecord *brdfLut =
          frame.resources->tryGet(environment.brdfLut);
      brdfLut != nullptr && nuri::isValid(brdfLut->texture)) {
    brdfLutTexId = brdfLut->bindlessIndex;
    frameFlags |= HasBrdfLut;
  }
  if (hasSceneColorInput && sceneColorTexId != kInvalidTextureBindlessIndex) {
    frameFlags |= HasSceneColor;
  }
  if (gpu_.getSwapchainFormat() != Format::RGBA8_SRGB) {
    frameFlags |= OutputLinearToSrgb;
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
      .sceneColorTexId = sceneColorTexId,
      .sceneColorSamplerId = sceneColorSamplerId,
      .reserved0 = sceneColorMip1TexId,
      .reserved1 = sceneColorMip2TexId,
  };

  if (!frameDataUploadValid_ || !(uploadedFrameData_ == frameData_)) {
    Result<bool, std::string> updateResult =
        Result<bool, std::string>::makeResult(true);
    NURI_PROFILER_ZONE("TransmissionLayer.frame_data_upload",
                       NURI_PROFILER_COLOR_CMD_COPY);
    const std::span<const std::byte> frameDataBytes{
        reinterpret_cast<const std::byte *>(&frameData_), sizeof(frameData_)};
    updateResult =
        gpu_.updateBuffer(frameDataBuffer_->handle(), frameDataBytes, 0);
    NURI_PROFILER_ZONE_END();
    if (updateResult.hasError()) {
      return updateResult;
    }
    uploadedFrameData_ = frameData_;
    frameDataUploadValid_ = true;
  }

  const Format depthFormat = nuri::isValid(depthTexture)
                                 ? gpu_.getTextureFormat(depthTexture)
                                 : Format::Count;
  auto pipelineResult = ensurePipelines(gpu_.getSwapchainFormat(), depthFormat);
  if (pipelineResult.hasError()) {
    return pipelineResult;
  }

  passDrawItems_.clear();
  passTextureReads_.clear();
  passDependencyBuffers_.clear();
  meshPushConstants_.clear();
  copyPushConstantsRing_.clear();

  if (!meshDrawTemplates_.empty()) {
    NURI_PROFILER_ZONE("TransmissionLayer.material_instance_uploads",
                       NURI_PROFILER_COLOR_CMD_COPY);
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
    NURI_PROFILER_ZONE_END();

    NURI_PROFILER_ZONE("TransmissionLayer.mesh_draw_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
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
          "TransmissionLayer::buildRenderGraph: invalid GPU buffer address");
    }

    meshPushConstants_.reserve(meshDrawTemplates_.size());
    const uint32_t renderableCount = saturateToU32(renderables.size());
    const bool hasDepthAttachment =
        nuri::isValid(depthTexture) || nuri::isValid(sceneDepthGraphTexture);
    for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
      if (entry.renderable == nullptr || entry.submesh == nullptr) {
        continue;
      }

      const std::optional<SubmeshLod> lod =
          resolveTransmissionLod(*entry.submesh, settings);
      if (!lod.has_value()) {
        continue;
      }

      meshPushConstants_.push_back(MeshPushConstants{
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
          // Transmission reuses these unused tessellation slots to pass the
          // imported local-space authored scale into the fragment shader.
          .tessNearDistance = entry.submesh->authoredScale.x,
          .tessFarDistance = entry.submesh->authoredScale.y,
          .tessMinFactor = entry.submesh->authoredScale.z,
          .tessMaxFactor = 1.0f,
          .debugVisualizationMode = 0u,
      });
      const MeshPushConstants &pc = meshPushConstants_.back();

      DrawItem draw{};
      draw.pipeline = selectMeshPipeline(entry.doubleSided);
      draw.indexBuffer = entry.indexBuffer;
      draw.indexBufferOffset = entry.indexBufferOffset;
      draw.indexFormat = IndexFormat::U32;
      draw.indexCount = lod->indexCount;
      draw.instanceCount = 1u;
      draw.firstIndex = lod->indexOffset;
      draw.firstInstance = entry.instanceIndex;
      if (hasDepthAttachment) {
        draw.useDepthState = true;
        draw.depthState = {.compareOp = CompareOp::LessEqual,
                           .isDepthWriteEnabled = true};
      }
      draw.pushConstants = std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pc), sizeof(MeshPushConstants));
      draw.debugLabel = kTransmissionMeshLabel;
      draw.debugColor = kTransmissionMeshDebugColor;
      passDrawItems_.push_back(draw);
    }

    appendUniqueBuffer(passDependencyBuffers_, materialBuffer_->handle());
    appendUniqueBuffer(passDependencyBuffers_,
                       instanceMatricesRing_[frameSlot].buffer->handle());
    appendUniqueBuffer(passDependencyBuffers_,
                       instanceRemapRing_[frameSlot].buffer->handle());
    for (const TextureHandle handle : environmentTextureAccessHandles_) {
      appendUniqueTexture(passTextureReads_, handle);
    }
    for (const TextureHandle handle : materialTextureAccessHandles_) {
      appendUniqueTexture(passTextureReads_, handle);
    }
    NURI_PROFILER_ZONE_END();
  }

  const bool copySceneColorToSwapchain =
      hasSceneColorInput && sceneColorTexId != kInvalidTextureBindlessIndex;
  const size_t expectedCopyDrawCount =
      static_cast<size_t>(kTransmissionSceneColorLevelCount - 1u) +
      (copySceneColorToSwapchain ? 1u : 0u);
  copyPushConstantsRing_.reserve(expectedCopyDrawCount);
  auto buildCopyDraw = [this](uint32_t sourceTexId, uint32_t sourceSamplerId,
                              uint32_t flags) -> DrawItem {
    copyPushConstantsRing_.push_back(CopyPushConstants{
        .sourceTexId = sourceTexId,
        .sourceSamplerId = sourceSamplerId,
        .flags = flags,
        .reserved0 = 0u,
    });
    const CopyPushConstants &pc = copyPushConstantsRing_.back();

    DrawItem draw{};
    draw.pipeline = copyPipelineHandle_;
    draw.vertexCount = 3u;
    draw.instanceCount = 1u;
    draw.pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&pc), sizeof(CopyPushConstants));
    draw.debugLabel = (flags & kTransmissionCopyFlagDownsample) != 0u
                          ? kTransmissionDownsampleLabel
                          : kTransmissionCopyLabel;
    draw.debugColor = (flags & kTransmissionCopyFlagDownsample) != 0u
                          ? kTransmissionDownsampleDebugColor
                          : kTransmissionCopyDebugColor;
    return draw;
  };

  TextureHandle previousSceneColorLevel = *sceneColorTexture;
  for (uint32_t level = 1u; level < kTransmissionSceneColorLevelCount;
       ++level) {
    NURI_PROFILER_ZONE("TransmissionLayer.downsample_pass_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    const TextureHandle destinationTexture =
        currentSceneColorTextureMip(frame.frameIndex, level);
    if (!nuri::isValid(destinationTexture)) {
      return Result<bool, std::string>::makeError(
          "TransmissionLayer::buildRenderGraph: scene color downsample "
          "texture is unavailable");
    }

    const uint32_t sourceTexId =
        gpu_.getTextureBindlessIndex(previousSceneColorLevel);
    if (sourceTexId == kInvalidTextureBindlessIndex) {
      return Result<bool, std::string>::makeError(
          "TransmissionLayer::buildRenderGraph: invalid scene color "
          "downsample source bindless index");
    }

    DrawItem downsampleDraw = buildCopyDraw(sourceTexId, sceneColorSamplerId,
                                            kTransmissionCopyFlagDownsample);
    RenderGraphGraphicsPassDesc downsamplePassDesc{};
    downsamplePassDesc.color = {.loadOp = LoadOp::Clear,
                                .storeOp = StoreOp::Store,
                                .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    auto destinationImportResult =
        graph.importTexture(destinationTexture, "transmission_scene_color_mip");
    if (destinationImportResult.hasError()) {
      return Result<bool, std::string>::makeError(
          destinationImportResult.error());
    }
    downsamplePassDesc.colorTexture = destinationImportResult.value();
    downsamplePassDesc.draws = std::span<const DrawItem>(&downsampleDraw, 1u);
    downsamplePassDesc.debugLabel = kTransmissionDownsamplePassLabel;
    downsamplePassDesc.debugColor = kTransmissionDownsampleDebugColor;

    auto addDownsampleResult = graph.addGraphicsPass(downsamplePassDesc);
    if (addDownsampleResult.hasError()) {
      return Result<bool, std::string>::makeError(addDownsampleResult.error());
    }

    auto sourceImportResult = graph.importTexture(
        previousSceneColorLevel, "transmission_scene_color_downsample_read");
    if (sourceImportResult.hasError()) {
      return Result<bool, std::string>::makeError(sourceImportResult.error());
    }
    auto sourceReadResult = graph.addTextureRead(addDownsampleResult.value(),
                                                 sourceImportResult.value());
    if (sourceReadResult.hasError()) {
      return Result<bool, std::string>::makeError(sourceReadResult.error());
    }

    previousSceneColorLevel = destinationTexture;
    NURI_PROFILER_ZONE_END();
  }

  if (copySceneColorToSwapchain) {
    NURI_PROFILER_ZONE("TransmissionLayer.copy_pass_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    DrawItem copyDraw = buildCopyDraw(sceneColorTexId, sceneColorSamplerId, 0u);

    RenderGraphGraphicsPassDesc copyPassDesc{};
    copyPassDesc.color = {.loadOp = LoadOp::Clear,
                          .storeOp = StoreOp::Store,
                          .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    copyPassDesc.draws = std::span<const DrawItem>(&copyDraw, 1u);
    copyPassDesc.debugLabel = kTransmissionCopyPassLabel;
    copyPassDesc.debugColor = kTransmissionCopyDebugColor;

    auto addCopyResult = graph.addGraphicsPass(copyPassDesc);
    if (addCopyResult.hasError()) {
      return Result<bool, std::string>::makeError(addCopyResult.error());
    }

    auto copyTextureImportResult = graph.importTexture(
        *sceneColorTexture, "transmission_scene_color_copy");
    if (copyTextureImportResult.hasError()) {
      return Result<bool, std::string>::makeError(
          copyTextureImportResult.error());
    }
    auto copyTextureReadResult = graph.addTextureRead(
        addCopyResult.value(), copyTextureImportResult.value());
    if (copyTextureReadResult.hasError()) {
      return Result<bool, std::string>::makeError(
          copyTextureReadResult.error());
    }
    NURI_PROFILER_ZONE_END();
  }
  if (hasSceneColorInput && sceneColorTexId != kInvalidTextureBindlessIndex) {
    appendUniqueTexture(passTextureReads_, *sceneColorTexture);
    for (uint32_t level = 1u; level < kTransmissionSceneColorLevelCount;
         ++level) {
      appendUniqueTexture(passTextureReads_,
                          currentSceneColorTextureMip(frame.frameIndex, level));
    }
  }

  RenderGraphGraphicsPassDesc passDesc{};
  NURI_PROFILER_ZONE("TransmissionLayer.main_pass_build",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  passDesc.color = {.loadOp = copySceneColorToSwapchain ? LoadOp::Load
                                                        : LoadOp::Clear,
                    .storeOp = StoreOp::Store,
                    .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  if (nuri::isValid(depthTexture) || nuri::isValid(sceneDepthGraphTexture)) {
    passDesc.depth = {.loadOp = LoadOp::Load,
                      .storeOp = StoreOp::Store,
                      .clearDepth = 1.0f,
                      .clearStencil = 0u};
    if (nuri::isValid(sceneDepthGraphTexture)) {
      passDesc.depthTexture = sceneDepthGraphTexture;
    } else {
      auto depthImportResult =
          graph.importTexture(depthTexture, "transmission_scene_depth");
      if (depthImportResult.hasError()) {
        return Result<bool, std::string>::makeError(depthImportResult.error());
      }
      passDesc.depthTexture = depthImportResult.value();
    }
  }
  passDesc.draws =
      std::span<const DrawItem>(passDrawItems_.data(), passDrawItems_.size());
  passDesc.dependencyBuffers = std::span<const BufferHandle>(
      passDependencyBuffers_.data(), passDependencyBuffers_.size());
  passDesc.debugLabel = kTransmissionPassLabel;
  passDesc.debugColor = kTransmissionPassDebugColor;

  auto addResult = graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }

  for (const TextureHandle handle : passTextureReads_) {
    auto importResult =
        graph.importTexture(handle, "transmission_texture_read");
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    auto readResult =
        graph.addTextureRead(addResult.value(), importResult.value());
    if (readResult.hasError()) {
      return Result<bool, std::string>::makeError(readResult.error());
    }
  }
  NURI_PROFILER_ZONE_END();

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransmissionLayer::ensureInitialized() {
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

Result<bool, std::string> TransmissionLayer::createShaders() {
  destroyShaders();
  meshShader_ = Shader::create("transmission_main", gpu_);
  copyShader_ = Shader::create("transmission_copy", gpu_);
  if (!meshShader_ || !copyShader_) {
    return Result<bool, std::string>::makeError(
        "TransmissionLayer::createShaders: failed to create shader wrappers");
  }

  auto meshVertexResult = meshShader_->compileFromFile(
      config_.meshVertex.string(), ShaderStage::Vertex);
  if (meshVertexResult.hasError()) {
    return Result<bool, std::string>::makeError(meshVertexResult.error());
  }
  auto meshFragmentResult = meshShader_->compileFromFile(
      transmissionFragmentPath_.string(), ShaderStage::Fragment);
  if (meshFragmentResult.hasError()) {
    return Result<bool, std::string>::makeError(meshFragmentResult.error());
  }
  auto copyVertexResult = copyShader_->compileFromFile(
      fullscreenCopyVertexPath_.string(), ShaderStage::Vertex);
  if (copyVertexResult.hasError()) {
    return Result<bool, std::string>::makeError(copyVertexResult.error());
  }
  auto copyFragmentResult = copyShader_->compileFromFile(
      sceneCopyFragmentPath_.string(), ShaderStage::Fragment);
  if (copyFragmentResult.hasError()) {
    return Result<bool, std::string>::makeError(copyFragmentResult.error());
  }

  meshVertexShader_ = meshVertexResult.value();
  meshFragmentShader_ = meshFragmentResult.value();
  copyVertexShader_ = copyVertexResult.value();
  copyFragmentShader_ = copyFragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionLayer::ensurePipelines(Format colorFormat, Format depthFormat) {
  const bool meshPipelinesValid =
      nuri::isValid(meshPipelineHandle_) &&
      nuri::isValid(meshDoubleSidedPipelineHandle_) &&
      meshPipelineColorFormat_ == colorFormat &&
      meshPipelineDepthFormat_ == depthFormat;
  const bool copyPipelineValid = nuri::isValid(copyPipelineHandle_) &&
                                 copyPipelineColorFormat_ == colorFormat &&
                                 copyPipelineDepthFormat_ == Format::Count;
  if (meshPipelinesValid && copyPipelineValid) {
    return Result<bool, std::string>::makeResult(true);
  }

  destroyPipelineState();

  auto meshResult = gpu_.createRenderPipeline(
      meshPipelineDesc(colorFormat, depthFormat, meshVertexShader_,
                       meshFragmentShader_, CullMode::Back),
      "transmission_mesh");
  if (meshResult.hasError()) {
    return Result<bool, std::string>::makeError(meshResult.error());
  }
  meshPipelineHandle_ = meshResult.value();

  auto meshDoubleResult = gpu_.createRenderPipeline(
      meshPipelineDesc(colorFormat, depthFormat, meshVertexShader_,
                       meshFragmentShader_, CullMode::None),
      "transmission_mesh_double_sided");
  if (meshDoubleResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(meshDoubleResult.error());
  }
  meshDoubleSidedPipelineHandle_ = meshDoubleResult.value();

  auto copyResult = gpu_.createRenderPipeline(
      copyPipelineDesc(colorFormat, Format::Count, copyVertexShader_,
                       copyFragmentShader_),
      "transmission_copy");
  if (copyResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(copyResult.error());
  }
  copyPipelineHandle_ = copyResult.value();

  meshPipelineColorFormat_ = colorFormat;
  meshPipelineDepthFormat_ = depthFormat;
  copyPipelineColorFormat_ = colorFormat;
  copyPipelineDepthFormat_ = Format::Count;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionLayer::ensureFrameDataBufferCapacity(size_t requiredBytes) {
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
                                     "transmission_frame_data");
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }
  frameDataBuffer_ = std::move(bufferResult.value());
  frameDataBufferCapacityBytes_ = requested;
  frameDataUploadValid_ = false;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionLayer::ensureMaterialBufferCapacity(size_t requiredBytes) {
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
                                     "transmission_material_data");
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }
  materialBuffer_ = std::move(bufferResult.value());
  materialBufferCapacityBytes_ = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionLayer::ensureRingBufferCount(uint32_t requiredCount) {
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
TransmissionLayer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
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
                                       "transmission_instance_matrices");
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
TransmissionLayer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
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
                                       "transmission_instance_remap");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;
    instanceDataRingUploadVersions_[i] = std::numeric_limits<uint64_t>::max();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransmissionLayer::ensureSceneColorTexture() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));
  const Format targetFormat = gpu_.getSwapchainFormat();
  const uint32_t textureCount = std::max(1u, gpu_.getSwapchainImageCount());

  const bool matchesExisting =
      sceneColorTextures_.size() == textureCount &&
      sceneColorMipTextures_.size() ==
          textureCount * (kTransmissionSceneColorLevelCount - 1u) &&
      sceneColorTextureFormat_ == targetFormat &&
      sceneColorTextureWidth_ == safeWidth &&
      sceneColorTextureHeight_ == safeHeight;
  if (matchesExisting) {
    if (allTexturesValid(sceneColorTextures_) &&
        allTexturesValid(sceneColorMipTextures_)) {
      return Result<bool, std::string>::makeResult(true);
    }
  }

  destroyTextures(gpu_, sceneColorTextures_);
  sceneColorTextures_.resize(textureCount);
  destroyTextures(gpu_, sceneColorMipTextures_);
  sceneColorMipTextures_.resize(textureCount *
                                (kTransmissionSceneColorLevelCount - 1u));

  const TextureDesc sceneColorDesc{
      .type = TextureType::Texture2D,
      .format = targetFormat,
      .dimensions = {safeWidth, safeHeight, 1},
      .usage = TextureUsage::AttachmentSampled,
      .storage = Storage::Device,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .data = {},
      .dataNumMipLevels = 1,
      .generateMipmaps = false,
  };
  for (uint32_t i = 0; i < textureCount; ++i) {
    const std::string debugName =
        "transmission_scene_color_" + std::to_string(i);
    auto textureResult =
        gpu_.createFramebufferTexture(sceneColorDesc, debugName);
    if (textureResult.hasError()) {
      return Result<bool, std::string>::makeError(textureResult.error());
    }
    sceneColorTextures_[i] = textureResult.value();
  }

  for (uint32_t level = 1u; level < kTransmissionSceneColorLevelCount;
       ++level) {
    TextureDesc mipDesc = sceneColorDesc;
    mipDesc.dimensions.width = levelDimensions(safeWidth, level);
    mipDesc.dimensions.height = levelDimensions(safeHeight, level);
    for (uint32_t i = 0; i < textureCount; ++i) {
      const std::string debugName = "transmission_scene_color_mip_" +
                                    std::to_string(level) + "_" +
                                    std::to_string(i);
      auto textureResult = gpu_.createTexture(mipDesc, debugName);
      if (textureResult.hasError()) {
        return Result<bool, std::string>::makeError(textureResult.error());
      }
      sceneColorMipTextures_[(level - 1u) * textureCount + i] =
          textureResult.value();
    }
  }

  sceneColorTextureFormat_ = targetFormat;
  sceneColorTextureWidth_ = safeWidth;
  sceneColorTextureHeight_ = safeHeight;
  return Result<bool, std::string>::makeResult(true);
}

TextureHandle
TransmissionLayer::currentSceneColorTexture(uint64_t frameIndex) const {
  if (sceneColorTextures_.empty()) {
    return {};
  }
  const size_t slot = ringSlot(frameIndex, sceneColorTextures_.size());
  return sceneColorTextures_[slot];
}

TextureHandle
TransmissionLayer::currentSceneColorTextureMip(uint64_t frameIndex,
                                               uint32_t level) const {
  if (level == 0u) {
    return currentSceneColorTexture(frameIndex);
  }
  if (level >= kTransmissionSceneColorLevelCount ||
      sceneColorMipTextures_.empty()) {
    return {};
  }
  const size_t textureCount = sceneColorTextures_.size();
  if (textureCount == 0u) {
    return {};
  }
  const size_t slot = ringSlot(frameIndex, textureCount);
  const size_t index = static_cast<size_t>(level - 1u) * textureCount + slot;
  if (index >= sceneColorMipTextures_.size()) {
    return {};
  }
  return sceneColorMipTextures_[index];
}

Result<bool, std::string>
TransmissionLayer::rebuildSceneCache(const RenderScene &scene,
                                     const ResourceManager &resources,
                                     uint32_t materialCount) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  meshDrawTemplates_.clear();

  const std::span<const Renderable> renderables = scene.renderables();
  if (renderables.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<bool, std::string>::makeError(
        "TransmissionLayer::rebuildSceneCache: renderables count exceeds "
        "UINT32_MAX");
  }

  size_t invalidMaterialFallbackCount = 0u;
  bool hasTransmissionContent = false;
  for (uint32_t renderableIndex = 0;
       renderableIndex < static_cast<uint32_t>(renderables.size());
       ++renderableIndex) {
    const Renderable &renderable = renderables[renderableIndex];
    const ModelRecord *modelRecord = resources.tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      return Result<bool, std::string>::makeError(
          "TransmissionLayer::rebuildSceneCache: failed to resolve model");
    }

    GeometryAllocationView geometry{};
    if (!gpu_.resolveGeometry(modelRecord->model->geometryHandle(), geometry)) {
      return Result<bool, std::string>::makeError(
          "TransmissionLayer::rebuildSceneCache: failed to resolve geometry");
    }
    const uint64_t vertexBufferAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (vertexBufferAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "TransmissionLayer::rebuildSceneCache: invalid vertex buffer "
          "address");
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
          !isTransmissionMaterial(*materialRecord)) {
        continue;
      }
      hasTransmissionContent = true;

      uint32_t materialIndex = resources.materialTableIndex(resolvedMaterial);
      if (materialCount == 0u || materialIndex >= materialCount) {
        materialIndex = 0u;
        ++invalidMaterialFallbackCount;
      }

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
          "TransmissionLayer::rebuildSceneCache: %zu submesh draw(s) used "
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
  cachedMaterialVersion_ = resources.materialSnapshot().version;
  cachedTransmissionContentScene_ = &scene;
  cachedTransmissionContentTopologyVersion_ = scene.topologyVersion();
  cachedTransmissionContentMaterialVersion_ = resources.materialSnapshot().version;
  cachedTransmissionContentValid_ = true;
  cachedTransmissionContent_ = hasTransmissionContent;
  cachedGeometryMutationVersion_ = gpu_.geometryMutationVersion();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransmissionLayer::rebuildMaterialTextureAccessCache(
    const ResourceManager &resources) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
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

void TransmissionLayer::collectEnvironmentTextureReads(
    const RenderScene &scene, const ResourceManager &resources) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
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

void TransmissionLayer::resetCachedState() {
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransmissionContentScene_ = nullptr;
  cachedTransmissionContentTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();
  cachedTransmissionContentMaterialVersion_ =
      std::numeric_limits<uint64_t>::max();
  cachedTransmissionContentValid_ = false;
  cachedTransmissionContent_ = false;
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

void TransmissionLayer::resetFrameBuildState() {
  meshPushConstants_.clear();
  passDrawItems_.clear();
  passTextureReads_.clear();
  passDependencyBuffers_.clear();
  copyPushConstantsRing_.clear();
}

void TransmissionLayer::destroyPipelineState() {
  if (nuri::isValid(copyPipelineHandle_)) {
    gpu_.destroyRenderPipeline(copyPipelineHandle_);
  }
  if (nuri::isValid(meshDoubleSidedPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshDoubleSidedPipelineHandle_);
  }
  if (nuri::isValid(meshPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshPipelineHandle_);
  }
  meshPipelineHandle_ = {};
  meshDoubleSidedPipelineHandle_ = {};
  copyPipelineHandle_ = {};
  meshPipelineColorFormat_ = Format::Count;
  meshPipelineDepthFormat_ = Format::Count;
  copyPipelineColorFormat_ = Format::Count;
  copyPipelineDepthFormat_ = Format::Count;
}

void TransmissionLayer::destroyShaders() {
  if (nuri::isValid(meshVertexShader_)) {
    gpu_.destroyShaderModule(meshVertexShader_);
  }
  if (nuri::isValid(meshFragmentShader_)) {
    gpu_.destroyShaderModule(meshFragmentShader_);
  }
  if (nuri::isValid(copyVertexShader_)) {
    gpu_.destroyShaderModule(copyVertexShader_);
  }
  if (nuri::isValid(copyFragmentShader_)) {
    gpu_.destroyShaderModule(copyFragmentShader_);
  }
  meshVertexShader_ = {};
  meshFragmentShader_ = {};
  copyVertexShader_ = {};
  copyFragmentShader_ = {};
}

void TransmissionLayer::destroyBuffers() {
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

bool TransmissionLayer::hasTransmissionContent(
    const RenderFrameContext &frame) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  if (frame.scene == nullptr || frame.resources == nullptr) {
    return false;
  }

  const MaterialTableSnapshot materialSnapshot =
      frame.resources->materialSnapshot();
  const uint64_t topologyVersion = frame.scene->topologyVersion();
  if (cachedTransmissionContentValid_ &&
      cachedTransmissionContentScene_ == frame.scene &&
      cachedTransmissionContentTopologyVersion_ == topologyVersion &&
      cachedTransmissionContentMaterialVersion_ == materialSnapshot.version) {
    return cachedTransmissionContent_;
  }
  auto rebuildResult = rebuildSceneCache(
      *frame.scene, *frame.resources,
      static_cast<uint32_t>(materialSnapshot.gpuData.size()));
  if (rebuildResult.hasError()) {
    NURI_LOG_WARNING("TransmissionLayer::hasTransmissionContent: %s",
                     rebuildResult.error().c_str());
    return false;
  }
  return cachedTransmissionContentValid_ &&
         cachedTransmissionContentScene_ == frame.scene &&
         cachedTransmissionContentTopologyVersion_ == topologyVersion &&
         cachedTransmissionContentMaterialVersion_ == materialSnapshot.version &&
         cachedTransmissionContent_;
}

RenderPipelineHandle
TransmissionLayer::selectMeshPipeline(bool doubleSided) const {
  if (doubleSided && nuri::isValid(meshDoubleSidedPipelineHandle_)) {
    return meshDoubleSidedPipelineHandle_;
  }
  return meshPipelineHandle_;
}

} // namespace nuri
