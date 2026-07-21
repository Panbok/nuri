#include "nuri/gfx/ddgi/ddgi_feature.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"
#include <atomic>

namespace nuri {
namespace {

std::atomic_uint64_t nextDDGIFeatureEpoch{1u};

constexpr uint32_t kDDGIResourceReady = 1u << 0u;
constexpr uint32_t kDDGIFrameEnabled = 1u << 0u;

template <typename T>
[[nodiscard]] std::span<const std::byte> bytesOf(const T &value) {
  return {reinterpret_cast<const std::byte *>(&value), sizeof(T)};
}

template <typename T>
[[nodiscard]] std::span<const std::byte> bytesOf(std::span<T> values) {
  return std::as_bytes(values);
}

[[nodiscard]] uint32_t divRoundUp(uint32_t value, uint32_t divisor) noexcept {
  return value / divisor + (value % divisor != 0u ? 1u : 0u);
}

void appendUnique(std::pmr::vector<BufferHandle> &handles,
                  std::pmr::vector<RenderGraphAccessMode> &modes,
                  BufferHandle handle, RenderGraphAccessMode mode) {
  if (!nuri::isValid(handle)) {
    return;
  }
  const auto found = std::ranges::find(handles, handle);
  if (found == handles.end()) {
    handles.push_back(handle);
    modes.push_back(mode);
    return;
  }
  const size_t index = static_cast<size_t>(found - handles.begin());
  modes[index] = modes[index] | mode;
}

void appendUnique(std::pmr::vector<TextureHandle> &handles,
                  std::pmr::vector<RenderGraphAccessMode> &modes,
                  TextureHandle handle, RenderGraphAccessMode mode) {
  if (!nuri::isValid(handle)) {
    return;
  }
  const auto found = std::ranges::find(handles, handle);
  if (found == handles.end()) {
    handles.push_back(handle);
    modes.push_back(mode);
    return;
  }
  const size_t index = static_cast<size_t>(found - handles.begin());
  modes[index] = modes[index] | mode;
}

[[nodiscard]] DDGIVolumeDesc toDesc(const RenderDDGIVolume &volume) {
  return DDGIVolumeDesc{
      .name = std::string(volume.name),
      .probeCounts = volume.probeCounts,
      .probeSpacing = volume.probeSpacing,
      .blendDistance = volume.blendDistance,
      .maxRayDistance = volume.maxRayDistance,
      .priority = volume.priority,
      .mode = volume.mode,
      .enabled = true,
  };
}

[[nodiscard]] uint32_t low32(uint64_t value) noexcept {
  return static_cast<uint32_t>(value);
}

[[nodiscard]] uint32_t high32(uint64_t value) noexcept {
  return static_cast<uint32_t>(value >> 32u);
}

[[nodiscard]] uint64_t join32(uint32_t low, uint32_t high) noexcept {
  return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u);
}

} // namespace

DDGIFeature::DDGIFeature(GPUDevice &gpu, RuntimeDDGIShaderConfig config,
                         std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      scratch_(memory_), volumes_(memory_), pendingVolumes_(memory_),
      frameSlots_(memory_), scheduledEntries_(memory_),
      scrollInvalidations_(memory_), pendingScrollLayouts_(memory_),
      dispatches_(memory_), blendPushConstants_(memory_),
      dependencyBuffers_(memory_), dependencyBufferModes_(memory_),
      dependencyTextures_(memory_), dependencyTextureModes_(memory_),
      forwardDependencyBuffers_(memory_), forwardDependencyTextures_(memory_),
      deviceEpoch_(
          nextDDGIFeatureEpoch.fetch_add(1u, std::memory_order_relaxed)) {}

DDGIFeature::~DDGIFeature() {
  clearVolumes();
  clearFrameSlots();
}

Result<bool, std::string> DDGIFeature::initialize() {
  auto sampler =
      gpu_.createSampler(SamplerDesc{.minFilter = SamplerFilter::Linear,
                                     .magFilter = SamplerFilter::Linear,
                                     .mipMode = SamplerMipMode::Disabled,
                                     .wrapU = SamplerWrapMode::Clamp,
                                     .wrapV = SamplerWrapMode::Clamp,
                                     .wrapW = SamplerWrapMode::Clamp},
                         "ddgi_atlas_sampler");
  if (sampler.hasError()) {
    return Result<bool, std::string>::makeError(sampler.error());
  }
  sampler_.reset(gpu_, sampler.value());
  const std::array paths{config_.trace, config_.traceInspect,
                         config_.blendIrradiance, config_.blendDistance,
                         config_.updateProbeState};
  const std::array names{"ddgi_trace", "ddgi_trace_inspect",
                         "ddgi_blend_irradiance", "ddgi_blend_distance",
                         "ddgi_update_probe_state"};
  for (size_t index = 0u; index < paths.size(); ++index) {
    shaders_[index] = Shader::create(names[index], gpu_);
    auto shader = shaders_[index]->compileFromFile(paths[index].string(),
                                                   ShaderStage::Compute);
    if (shader.hasError()) {
      return Result<bool, std::string>::makeError(shader.error());
    }
    auto pipeline = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = shader.value()}, names[index]);
    if (pipeline.hasError()) {
      return Result<bool, std::string>::makeError(pipeline.error());
    }
    pipelines_[index].reset(gpu_, pipeline.value());
  }
  return Result<bool, std::string>::makeResult(true);
}

void DDGIFeature::clearVolumes() noexcept {
  volumes_.clear();
  clearPendingVolumes();
  frameData_ = {};
  traceBinding_.reset();
  inspectBinding_.reset();
  boundTlas_ = {};
  inspectBoundTlas_ = {};
  latestInspectionResult_.reset();
  failedVolumeCount_ = 0u;
  volumeFailureReason_ = DDGIVolumeFailureReason::None;
  sceneId_ = 0u;
  volumeTopologyVersion_ = UINT64_MAX;
  volumeTransformVersion_ = UINT64_MAX;
  volumeSettingsVersion_ = UINT64_MAX;
  sceneTopologyVersion_ = UINT64_MAX;
  sceneTransformVersion_ = UINT64_MAX;
  sceneDeformationVersion_ = UINT64_MAX;
  lightTopologyVersion_ = UINT64_MAX;
  lightTransformVersion_ = UINT64_MAX;
  materialVersion_ = UINT64_MAX;
  environmentVersion_ = UINT64_MAX;
}

void DDGIFeature::clearPendingVolumes() noexcept {
  pendingVolumes_.clear();
  replacementPending_ = false;
  pendingSceneId_ = 0u;
  pendingVolumeTopologyVersion_ = UINT64_MAX;
  pendingVolumeTransformVersion_ = UINT64_MAX;
  pendingVolumeSettingsVersion_ = UINT64_MAX;
  pendingFailedVolumeCount_ = 0u;
  pendingVolumeFailureReason_ = DDGIVolumeFailureReason::None;
}

void DDGIFeature::clearFrameSlots() noexcept { frameSlots_.clear(); }

Result<bool, std::string> DDGIFeature::rebuildVolumes(FrameBuildContext &ctx) {
  clearPendingVolumes();
  const auto recordFailure = [this](DDGIVolumeFailureReason reason) {
    ++pendingFailedVolumeCount_;
    if (pendingVolumeFailureReason_ == DDGIVolumeFailureReason::None) {
      pendingVolumeFailureReason_ = reason;
    }
  };
  const std::span<const RenderDDGIVolume> source =
      ctx.frame.scene->ddgiVolumes();
  const uint32_t count =
      std::min<uint32_t>(static_cast<uint32_t>(source.size()), kMaxDDGIVolumes);
  const uint32_t maximumTextureDimension =
      std::max(gpu_.getDeviceCaps().maxTextureDimension2D, 1u);
  uint64_t aggregateBytes = 0u;
  uint64_t activeBytes = 0u;
  for (const VolumeResource &active : volumes_) {
    activeBytes += active.persistentBytes;
  }
  const RenderSettings::DDGISettings &settings =
      renderSettingsOrDefault(ctx.frame).ddgi;
  const uint64_t frameBatchBytes =
      static_cast<uint64_t>(settings.maxProbeUpdatesPerFrame) *
          sizeof(DDGIProbeUpdateEntry) +
      static_cast<uint64_t>(settings.maxRayQueriesPerFrame) *
          sizeof(DDGIRayResultGpuData) +
      sizeof(DDGIFrameGpuData);
  ScopedScratch scratch(scratch_);
  for (uint32_t slot = 0u; slot < count; ++slot) {
    const RenderDDGIVolume &volume = source[slot];
    const DDGIVolumeDesc desc = toDesc(volume);
    const uint32_t probeCount = ddgiProbeCount(desc.probeCounts);
    auto irradiancePacking =
        packDDGIAtlas(probeCount, kDDGIIrradianceTileExtent,
                      glm::uvec2(maximumTextureDimension));
    auto distancePacking = packDDGIAtlas(probeCount, kDDGIDistanceTileExtent,
                                         glm::uvec2(maximumTextureDimension));
    if (irradiancePacking.hasError() || distancePacking.hasError()) {
      recordFailure(DDGIVolumeFailureReason::AtlasPacking);
      continue;
    }
    auto memory = estimateDDGIMemory(probeCount, irradiancePacking.value(),
                                     distancePacking.value());
    if (memory.hasError()) {
      recordFailure(DDGIVolumeFailureReason::AtlasPacking);
      continue;
    }
    if (aggregateBytes + memory.value().persistentBytes >
        config_.persistentMemoryLimitBytes) {
      recordFailure(DDGIVolumeFailureReason::PersistentMemoryLimit);
      continue;
    }
    if (activeBytes + aggregateBytes + memory.value().persistentBytes +
            frameBatchBytes >
        config_.peakMemoryLimitBytes) {
      recordFailure(DDGIVolumeFailureReason::PeakMemoryLimit);
      continue;
    }
    auto layout = makeDDGIVolumeLayout(
        volume.id, desc, volume.worldFromLocal, irradiancePacking.value(),
        distancePacking.value(), ctx.frame.scene->ddgiVolumeSettingsVersion());
    if (layout.hasError()) {
      recordFailure(DDGIVolumeFailureReason::InvalidLayout);
      continue;
    }
    if (desc.mode == DDGIVolumeMode::CameraTracked) {
      const glm::vec3 cameraLocal =
          glm::vec3(layout.value().localFromWorld * ctx.frame.camera.cameraPos);
      layout.value().cameraCell =
          ddgiCameraCell(cameraLocal, layout.value().probeSpacing);
    }
    auto irradiance = gpu_.createTexture(
        TextureDesc{.type = TextureType::Texture2D,
                    .format = Format::RGBA16_FLOAT,
                    .dimensions = {irradiancePacking.value().textureExtent.x,
                                   irradiancePacking.value().textureExtent.y,
                                   1u},
                    .usage = TextureUsage::StorageSampled,
                    .storage = Storage::Device},
        "ddgi_irradiance_atlas");
    if (irradiance.hasError()) {
      recordFailure(DDGIVolumeFailureReason::IrradianceAllocation);
      continue;
    }
    auto distance = gpu_.createTexture(
        TextureDesc{.type = TextureType::Texture2D,
                    .format = Format::RG16_FLOAT,
                    .dimensions = {distancePacking.value().textureExtent.x,
                                   distancePacking.value().textureExtent.y, 1u},
                    .usage = TextureUsage::StorageSampled,
                    .storage = Storage::Device},
        "ddgi_distance_atlas");
    if (distance.hasError()) {
      gpu_.destroyTexture(irradiance.value());
      recordFailure(DDGIVolumeFailureReason::DistanceAllocation);
      continue;
    }
    std::pmr::vector<DDGIProbeStateGpuData> initialStates(scratch.resource());
    initialStates.resize(probeCount);
    const DDGIProbeState initialState = settings.classification
                                            ? DDGIProbeState::Uninitialized
                                            : DDGIProbeState::NewlyVigilant;
    for (DDGIProbeStateGpuData &state : initialStates) {
      state.stateAgeFlags.x = static_cast<uint32_t>(initialState);
    }
    auto state = gpu_.createBuffer(
        BufferDesc{.usage = BufferUsage::Storage,
                   .storage = Storage::Device,
                   .size = initialStates.size() * sizeof(DDGIProbeStateGpuData),
                   .data = bytesOf(std::span(initialStates))},
        "ddgi_probe_state");
    if (state.hasError()) {
      gpu_.destroyTexture(irradiance.value());
      gpu_.destroyTexture(distance.value());
      recordFailure(DDGIVolumeFailureReason::ProbeStateAllocation);
      continue;
    }
    pendingVolumes_.emplace_back(memory_);
    VolumeResource &resource = pendingVolumes_.back();
    resource.id = volume.id;
    resource.layout = layout.value();
    resource.desc = desc;
    resource.irradiance.reset(gpu_, irradiance.value());
    resource.distance.reset(gpu_, distance.value());
    resource.probeState.reset(gpu_, state.value());
    resource.lastSubmittedUpdates.resize(probeCount, 0u);
    resource.persistentBytes = memory.value().persistentBytes;
    resource.resourceGeneration = ++nextResourceGeneration_;
    aggregateBytes += memory.value().persistentBytes;
  }
  pendingSceneId_ = ctx.frame.scene->id();
  pendingVolumeTopologyVersion_ = ctx.frame.scene->ddgiVolumeTopologyVersion();
  pendingVolumeTransformVersion_ =
      ctx.frame.scene->ddgiVolumeTransformVersion();
  pendingVolumeSettingsVersion_ = ctx.frame.scene->ddgiVolumeSettingsVersion();
  pendingRelocationEnabled_ = settings.relocation;
  pendingClassificationEnabled_ = settings.classification;
  replacementPending_ = true;
  return Result<bool, std::string>::makeResult(true);
}

void DDGIFeature::buildScrollPlan(const RenderFrameContext &frame) {
  pendingScrollLayouts_.clear();
  scrollInvalidations_.clear();
  pendingScrollLayouts_.reserve(volumes_.size());
  for (const VolumeResource &volume : volumes_) {
    pendingScrollLayouts_.push_back(volume.layout);
  }
  for (uint32_t slot = 0u; slot < static_cast<uint32_t>(volumes_.size());
       ++slot) {
    const VolumeResource &volume = volumes_[slot];
    if (volume.desc.mode != DDGIVolumeMode::CameraTracked) {
      continue;
    }
    const glm::vec3 cameraLocal =
        glm::vec3(volume.layout.localFromWorld * frame.camera.cameraPos);
    const glm::ivec3 targetCell =
        ddgiCameraCell(cameraLocal, volume.layout.probeSpacing);
    const DDGIScrollPlan plan =
        makeDDGIScrollPlan(volume.layout.cameraCell, volume.layout.ringOrigin,
                           targetCell, volume.layout.probeCounts);
    if (!plan.changed) {
      continue;
    }
    DDGIVolumeLayout &pending = pendingScrollLayouts_[slot];
    pending.cameraCell = plan.cameraCell;
    pending.ringOrigin = plan.ringOrigin;
    const uint32_t probeCount = ddgiProbeCount(volume.layout.probeCounts);
    for (uint32_t logicalIndex = 0u; logicalIndex < probeCount;
         ++logicalIndex) {
      const glm::uvec3 logical =
          ddgiProbeCoordinate(logicalIndex, volume.layout.probeCounts);
      if (!isDDGINewlyExposedCoordinate(logical, plan,
                                        volume.layout.probeCounts)) {
        continue;
      }
      const glm::uvec3 physical = ddgiPhysicalProbeCoordinate(
          logical, plan.ringOrigin, volume.layout.probeCounts);
      scrollInvalidations_.push_back(DDGIProbeUpdateEntry{
          .volumeStableId = slot,
          .probeId = ddgiProbeIndex(physical, volume.layout.probeCounts),
      });
    }
  }
  std::ranges::sort(scrollInvalidations_, {}, [](const auto &entry) {
    return std::pair(entry.volumeStableId, entry.probeId);
  });
}

Result<bool, std::string>
DDGIFeature::ensureFrameSlots(const RenderSettings::DDGISettings &settings,
                              size_t localLightCount,
                              size_t invalidationCapacity) {
  const uint32_t ringCount = std::max(1u, gpu_.getSwapchainImageCount());
  if (frameSlots_.size() != ringCount) {
    clearFrameSlots();
    frameSlots_.resize(ringCount);
  }
  for (FrameSlot &slot : frameSlots_) {
    const auto ensure =
        [this](OwnedBufferHandle &buffer, size_t size,
               std::string_view name) -> Result<bool, std::string> {
      if (buffer.valid()) {
        return Result<bool, std::string>::makeResult(true);
      }
      auto created =
          gpu_.createBuffer(BufferDesc{.usage = BufferUsage::Storage,
                                       .storage = Storage::Device,
                                       .size = std::max<size_t>(size, 16u)},
                            name);
      if (created.hasError()) {
        return Result<bool, std::string>::makeError(created.error());
      }
      buffer.reset(gpu_, created.value());
      return Result<bool, std::string>::makeResult(true);
    };
    if (slot.updateCapacity < settings.maxProbeUpdatesPerFrame) {
      slot.updates.reset();
      slot.updateCapacity = settings.maxProbeUpdatesPerFrame;
    }
    if (slot.rayCapacity < settings.maxRayQueriesPerFrame) {
      slot.rayResults.reset();
      slot.rayCapacity = settings.maxRayQueriesPerFrame;
    }
    if (slot.invalidationCapacity < invalidationCapacity) {
      slot.invalidations.reset();
      slot.invalidationCapacity = invalidationCapacity;
    }
    if (slot.localLightCapacity < localLightCount) {
      slot.localLights.reset();
      slot.localLightCapacity = localLightCount;
    }
    for (auto result :
         {ensure(slot.frameData, sizeof(DDGIFrameGpuData), "ddgi_frame_data"),
          ensure(slot.updates,
                 slot.updateCapacity * sizeof(DDGIProbeUpdateEntry),
                 "ddgi_probe_updates"),
          ensure(slot.invalidations,
                 slot.invalidationCapacity * sizeof(DDGIProbeUpdateEntry),
                 "ddgi_scroll_invalidations"),
          ensure(slot.rayResults,
                 slot.rayCapacity * sizeof(DDGIRayResultGpuData),
                 "ddgi_ray_results"),
          ensure(slot.localLights,
                 sizeof(DDGITraceCountersGpuData) +
                     slot.localLightCapacity * sizeof(LocalLightGpuData),
                 "ddgi_local_lights")}) {
      if (result.hasError()) {
        return result;
      }
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> DDGIFeature::updateFrameData(FrameBuildContext &ctx,
                                                       FrameSlot &slot) {
  frameData_ = {};
  frameData_.activeCountDebugFlagsSampler = glm::uvec4(
      static_cast<uint32_t>(volumes_.size()),
      static_cast<uint32_t>(renderSettingsOrDefault(ctx.frame).ddgi.debugView),
      0u, gpu_.getSamplerBindlessIndex(sampler_.get()));
  for (size_t index = 0u; index < volumes_.size(); ++index) {
    const VolumeResource &resource = volumes_[index];
    const DDGIVolumeLayout &layout =
        pendingScrollLayouts_.size() == volumes_.size()
            ? pendingScrollLayouts_[index]
            : resource.layout;
    DDGIVolumeGpuData &gpuVolume = frameData_.volumes[index];
    gpuVolume.worldFromLocal = layout.worldFromLocal;
    gpuVolume.localFromWorld = layout.localFromWorld;
    gpuVolume.probeStateBufferAddress =
        gpu_.getBufferDeviceAddress(resource.probeState.get());
    gpuVolume.resourceFlags = resource.ready ? kDDGIResourceReady : 0u;
    gpuVolume.probeSpacingAndBias =
        glm::vec4(layout.probeSpacing,
                  renderSettingsOrDefault(ctx.frame).ddgi.selfShadowBias);
    gpuVolume.centerHalfExtentsAndMaxDistance =
        glm::vec4(layout.probeCenterHalfExtents, resource.desc.maxRayDistance);
    gpuVolume.probeCountsAndCount =
        glm::uvec4(layout.probeCounts, ddgiProbeCount(layout.probeCounts));
    gpuVolume.irradianceAtlas =
        glm::uvec4(gpu_.getTextureBindlessIndex(resource.irradiance.get()),
                   layout.irradianceAtlas.tileExtent.x,
                   layout.irradianceAtlas.columns, layout.irradianceAtlas.rows);
    gpuVolume.distanceAtlas =
        glm::uvec4(gpu_.getTextureBindlessIndex(resource.distance.get()),
                   layout.distanceAtlas.tileExtent.x,
                   layout.distanceAtlas.columns, layout.distanceAtlas.rows);
    gpuVolume.ringOriginAndFlags =
        glm::uvec4(layout.ringOrigin,
                   std::bit_cast<uint32_t>(resource.desc.blendDistance));
    gpuVolume.generations = glm::uvec4(
        std::bit_cast<uint32_t>(layout.cameraCell.x),
        std::bit_cast<uint32_t>(layout.cameraCell.y), low32(submittedSequence_),
        std::bit_cast<uint32_t>(layout.cameraCell.z));
  }
  auto upload = gpu_.updateBuffer(slot.frameData.get(), bytesOf(frameData_));
  if (upload.hasError()) {
    return upload;
  }
  const std::span<const LocalLightGpuData> localLights =
      ctx.frame.scene->packedLocalLights();
  if (!localLights.empty()) {
    auto lightUpload =
        gpu_.updateBuffer(slot.localLights.get(), bytesOf(localLights),
                          sizeof(DDGITraceCountersGpuData));
    if (lightUpload.hasError()) {
      return lightUpload;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DDGIFeature::appendInitializationPass(FrameBuildContext &ctx, FrameSlot &slot) {
  blendPushConstants_.clear();
  dispatches_.clear();
  dependencyTextures_.clear();
  dependencyTextureModes_.clear();
  blendPushConstants_.reserve(pendingVolumes_.size() * 2u);
  dispatches_.reserve(pendingVolumes_.size() * 2u);
  for (uint32_t volumeSlot = 0u;
       volumeSlot < static_cast<uint32_t>(pendingVolumes_.size());
       ++volumeSlot) {
    const VolumeResource &volume = pendingVolumes_[volumeSlot];
    const std::array outputs{volume.irradiance.get(), volume.distance.get()};
    const std::array extents{volume.layout.irradianceAtlas.textureExtent,
                             volume.layout.distanceAtlas.textureExtent};
    const std::array pipelineIndices{BlendIrradiance, BlendDistance};
    for (size_t type = 0u; type < outputs.size(); ++type) {
      blendPushConstants_.push_back(BlendPushConstants{
          .frame = gpu_.getBufferDeviceAddress(slot.frameData.get()),
          .volumeSlot = volumeSlot,
          .outputTextureId = gpu_.getTextureBindlessIndex(outputs[type]),
          .clearMode = 1u,
      });
      dispatches_.push_back(ComputeDispatchItem{
          .pipeline = pipelines_[pipelineIndices[type]].get(),
          .dispatch = {.x = divRoundUp(extents[type].x, 8u),
                       .y = divRoundUp(extents[type].y, 8u),
                       .z = 1u},
          .pushConstants = bytesOf(blendPushConstants_.back()),
          .debugLabel =
              type == 0u ? "DDGI Clear Irradiance" : "DDGI Clear Distance",
          .debugColor = 0xff55cc88u,
      });
      appendUnique(dependencyTextures_, dependencyTextureModes_, outputs[type],
                   RenderGraphAccessMode::Write);
    }
  }
  if (!dispatches_.empty()) {
    auto pass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
        .executionMode = RenderPassExecutionMode::ComputeOnly,
        .hasColorAttachment = false,
        .preDispatches = dispatches_,
        .dependencyTextures = dependencyTextures_,
        .dependencyTextureAccessModes = dependencyTextureModes_,
        .gpuTimingScope = GpuTimingScope::DDGIUpdate,
        .debugLabel = "DDGI Initialize Volumes",
        .debugColor = 0xff55cc88u,
    });
    if (pass.hasError()) {
      return Result<bool, std::string>::makeError(pass.error());
    }
  }
  initializationScheduled_ = true;
  scheduledFrameIndex_ = ctx.frame.frameIndex;
  pendingResetEpoch_ =
      renderSettingsOrDefault(ctx.frame).ddgi.requestedEpochs.resetHistory;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> DDGIFeature::appendScrollPass(FrameBuildContext &ctx,
                                                        FrameSlot &slot) {
  if (scrollInvalidations_.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto upload = gpu_.updateBuffer(slot.invalidations.get(),
                                  bytesOf(std::span(scrollInvalidations_)));
  if (upload.hasError()) {
    return upload;
  }
  blendPushConstants_.clear();
  dispatches_.clear();
  dependencyBuffers_.clear();
  dependencyBufferModes_.clear();
  dependencyTextures_.clear();
  dependencyTextureModes_.clear();
  blendPushConstants_.reserve(volumes_.size() * 2u);
  dispatches_.reserve(volumes_.size() * 2u + 1u);
  appendUnique(dependencyBuffers_, dependencyBufferModes_, slot.frameData.get(),
               RenderGraphAccessMode::Read);
  appendUnique(dependencyBuffers_, dependencyBufferModes_,
               slot.invalidations.get(), RenderGraphAccessMode::Read);
  for (uint32_t volumeSlot = 0u;
       volumeSlot < static_cast<uint32_t>(volumes_.size()); ++volumeSlot) {
    const VolumeResource &volume = volumes_[volumeSlot];
    const std::array outputs{volume.irradiance.get(), volume.distance.get()};
    const std::array pipelineIndices{BlendIrradiance, BlendDistance};
    for (size_t type = 0u; type < outputs.size(); ++type) {
      blendPushConstants_.push_back(BlendPushConstants{
          .frame = gpu_.getBufferDeviceAddress(slot.frameData.get()),
          .updates = gpu_.getBufferDeviceAddress(slot.invalidations.get()),
          .updateCount = static_cast<uint32_t>(scrollInvalidations_.size()),
          .volumeSlot = volumeSlot,
          .outputTextureId = gpu_.getTextureBindlessIndex(outputs[type]),
          .clearMode = 2u,
      });
      dispatches_.push_back(ComputeDispatchItem{
          .pipeline = pipelines_[pipelineIndices[type]].get(),
          .dispatch = {.x = static_cast<uint32_t>(scrollInvalidations_.size()),
                       .y = 1u,
                       .z = 1u},
          .pushConstants = bytesOf(blendPushConstants_.back()),
          .debugLabel =
              type == 0u ? "DDGI Scroll Irradiance" : "DDGI Scroll Distance",
          .debugColor = 0xffcc9955u,
      });
      appendUnique(dependencyTextures_, dependencyTextureModes_, outputs[type],
                   RenderGraphAccessMode::Write);
    }
  }

  statePushConstants_ = {};
  statePushConstants_.updates =
      gpu_.getBufferDeviceAddress(slot.invalidations.get());
  statePushConstants_.updateCount =
      static_cast<uint32_t>(scrollInvalidations_.size());
  statePushConstants_.clearMode = 1u;
  statePushConstants_.classificationEnabled =
      renderSettingsOrDefault(ctx.frame).ddgi.classification ? 1u : 0u;
  for (size_t index = 0u; index < volumes_.size(); ++index) {
    statePushConstants_.states[index] =
        gpu_.getBufferDeviceAddress(volumes_[index].probeState.get());
    appendUnique(dependencyBuffers_, dependencyBufferModes_,
                 volumes_[index].probeState.get(),
                 RenderGraphAccessMode::Write);
  }
  dispatches_.push_back(ComputeDispatchItem{
      .pipeline = pipelines_[UpdateProbeState].get(),
      .dispatch = {.x = divRoundUp(
                       static_cast<uint32_t>(scrollInvalidations_.size()), 64u),
                   .y = 1u,
                   .z = 1u},
      .pushConstants = bytesOf(statePushConstants_),
      .debugLabel = "DDGI Scroll Probe State",
      .debugColor = 0xffcc9955u,
  });
  auto pass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
      .executionMode = RenderPassExecutionMode::ComputeOnly,
      .hasColorAttachment = false,
      .preDispatches = dispatches_,
      .dependencyBuffers = dependencyBuffers_,
      .dependencyBufferAccessModes = dependencyBufferModes_,
      .dependencyTextures = dependencyTextures_,
      .dependencyTextureAccessModes = dependencyTextureModes_,
      .gpuTimingScope = GpuTimingScope::DDGIUpdate,
      .debugLabel = "DDGI Scroll Volumes",
      .debugColor = 0xffcc9955u,
  });
  if (pass.hasError()) {
    return Result<bool, std::string>::makeError(pass.error());
  }
  scrollScheduled_ = true;
  scheduledFrameIndex_ = ctx.frame.frameIndex;
  return Result<bool, std::string>::makeResult(true);
}

Result<DDGIScheduleResult, std::string>
DDGIFeature::buildSchedule(const RenderSettings::DDGISettings &settings) {
  ScopedScratch scoped(scratch_);
  size_t candidateCount = 0u;
  for (const VolumeResource &volume : volumes_) {
    candidateCount += volume.lastSubmittedUpdates.size();
  }
  std::pmr::vector<DDGIProbeScheduleCandidate> candidates(scoped.resource());
  std::pmr::vector<DDGIProbeScheduleCandidate> workspace(scoped.resource());
  std::pmr::vector<DDGIProbeUpdateEntry> output(scoped.resource());
  candidates.reserve(candidateCount);
  for (uint32_t slot = 0u; slot < static_cast<uint32_t>(volumes_.size());
       ++slot) {
    const VolumeResource &volume = volumes_[slot];
    for (uint32_t probe = 0u;
         probe < static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
         ++probe) {
      const bool scrolled = std::ranges::binary_search(
          scrollInvalidations_, std::pair(slot, probe), {},
          [](const auto &entry) {
            return std::pair(entry.volumeStableId, entry.probeId);
          });
      candidates.push_back(DDGIProbeScheduleCandidate{
          .volumeStableId = slot,
          .probeId = probe,
          .state = scrolled ? DDGIProbeState::Uninitialized
                            : DDGIProbeState::Vigilant,
          .lastSubmittedUpdate = volume.lastSubmittedUpdates[probe],
          .invalidated = volume.lastSubmittedUpdates[probe] == 0u || scrolled,
      });
    }
  }
  workspace.resize(candidates.size());
  output.resize(settings.maxProbeUpdatesPerFrame);
  auto schedule = scheduleDDGIProbeUpdates(
      candidates,
      DDGISchedulerLimits{
          .raysPerProbe = settings.raysPerProbe,
          .maxProbeUpdates = settings.maxProbeUpdatesPerFrame,
          .maxRayQueries = settings.maxRayQueriesPerFrame,
          .forceFullUpdate = ddgiEpochIsPending(
              settings.requestedEpochs.forceFullUpdate, consumedForceEpoch_),
      },
      workspace, output);
  if (schedule.hasError()) {
    return Result<DDGIScheduleResult, std::string>::makeError(
        "DDGIFeature: deterministic scheduler rejected its frame workspace");
  }
  scheduledEntries_.assign(output.begin(),
                           output.begin() + schedule.value().updatedProbes);
  return Result<DDGIScheduleResult, std::string>::makeResult(schedule.value());
}

Result<bool, std::string>
DDGIFeature::appendUpdatePasses(FrameBuildContext &ctx, FrameSlot &slot,
                                const DDGIScheduleResult &schedule) {
  DDGITraceCountersGpuData initialCounters{};
  initialCounters.reserved = schedule.secondaryQueriesReserved;
  auto counterUpload =
      gpu_.updateBuffer(slot.localLights.get(), bytesOf(initialCounters));
  if (counterUpload.hasError()) {
    return counterUpload;
  }
  const RayTracingSceneFrameView &rt = *ctx.shared.rayTracingScene;
  if (!traceBinding_.valid() ||
      boundTlas_ != rt.topLevelAccelerationStructure) {
    traceBinding_.reset();
    auto binding = gpu_.createRayQueryBinding(pipelines_[Trace].get(),
                                              rt.topLevelAccelerationStructure,
                                              "ddgi_trace_tlas_binding");
    if (binding.hasError()) {
      return Result<bool, std::string>::makeError(binding.error());
    }
    traceBinding_.reset(gpu_, binding.value());
    boundTlas_ = rt.topLevelAccelerationStructure;
  }
  auto upload = gpu_.updateBuffer(slot.updates.get(),
                                  bytesOf(std::span(scheduledEntries_)));
  if (upload.hasError()) {
    return upload;
  }
  const RenderScene &scene = *ctx.frame.scene;
  uint32_t skyTextureId = kInvalidTextureBindlessIndex;
  TextureHandle skyTexture{};
  if (const TextureRecord *sky =
          ctx.resources.tryGet(scene.environment().cubemap);
      sky != nullptr) {
    skyTexture = sky->texture;
    skyTextureId = sky->bindlessIndex;
  }
  const MaterialTableGpuData &materials = *ctx.shared.materialTableGpuData;
  const std::span<const DirectionalLightGpuData> directionalLights =
      scene.packedDirectionalLights();
  const std::span<const LightId> directionalLightIds =
      scene.packedDirectionalLightIds();
  uint32_t directionalLightIndex = 0u;
  if (ctx.shared.selectedShadowLightId.has_value()) {
    const auto selected = std::ranges::find(directionalLightIds,
                                            *ctx.shared.selectedShadowLightId);
    if (selected != directionalLightIds.end()) {
      directionalLightIndex =
          static_cast<uint32_t>(selected - directionalLightIds.begin());
    }
  }
  const DirectionalLightGpuData *directionalLight =
      directionalLightIndex < directionalLights.size()
          ? &directionalLights[directionalLightIndex]
          : nullptr;
  const std::span<const LocalLightGpuData> localLights =
      scene.packedLocalLights();
  tracePushConstants_ = TracePushConstants{
      .frame = gpu_.getBufferDeviceAddress(slot.frameData.get()),
      .updates = gpu_.getBufferDeviceAddress(slot.updates.get()),
      .results = gpu_.getBufferDeviceAddress(slot.rayResults.get()),
      .instances = rt.instanceTableAddress,
      .geometries = rt.geometryTableAddress,
      .materials = materials.headerBufferAddress,
      .directionalDirectionIlluminance =
          directionalLight != nullptr ? directionalLight->directionIlluminance
                                      : glm::vec4(0.0f),
      .directionalColor =
          directionalLight == nullptr
              ? glm::vec4(0.0f, 0.0f, 0.0f,
                          renderSettingsOrDefault(ctx.frame)
                              .ddgi.multiBounceLuminanceClamp)
              : glm::vec4(glm::vec3(directionalLight->colorReserved),
                          renderSettingsOrDefault(ctx.frame)
                              .ddgi.multiBounceLuminanceClamp),
      .localLights = gpu_.getBufferDeviceAddress(slot.localLights.get()),
      .rayCount = schedule.primaryQueries,
      .raysPerProbe = renderSettingsOrDefault(ctx.frame).ddgi.raysPerProbe,
      .skyTextureId = skyTextureId,
      .skySamplerId = gpu_.getCubemapSamplerBindlessIndex(),
      .maxCandidates = renderSettingsOrDefault(ctx.frame)
                           .ddgi.maxCandidateIntersectionsPerRay,
      .frameSeed = static_cast<uint32_t>(submittedSequence_),
      .materialSamplerId = gpu_.getDefaultSamplerBindlessIndex(),
      .directionalLightCount = directionalLight != nullptr ? 1u : 0u,
      .localLightCount = static_cast<uint32_t>(localLights.size()),
      .maxLocalLights =
          renderSettingsOrDefault(ctx.frame).ddgi.maxLocalLightsPerHit |
          (renderSettingsOrDefault(ctx.frame).ddgi.multiBounce ? 0x80000000u
                                                               : 0u),
  };
  dependencyBuffers_.clear();
  dependencyBufferModes_.clear();
  appendUnique(dependencyBuffers_, dependencyBufferModes_, slot.frameData.get(),
               RenderGraphAccessMode::Read);
  appendUnique(dependencyBuffers_, dependencyBufferModes_, slot.updates.get(),
               RenderGraphAccessMode::Read);
  appendUnique(dependencyBuffers_, dependencyBufferModes_,
               slot.rayResults.get(), RenderGraphAccessMode::Write);
  appendUnique(dependencyBuffers_, dependencyBufferModes_,
               slot.localLights.get(),
               RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  for (const VolumeResource &volume : volumes_) {
    appendUnique(dependencyBuffers_, dependencyBufferModes_,
                 volume.probeState.get(), RenderGraphAccessMode::Read);
  }
  for (BufferHandle reference : rt.indirectSubmissionReferences) {
    appendUnique(dependencyBuffers_, dependencyBufferModes_, reference,
                 RenderGraphAccessMode::Read);
  }
  dependencyTextures_.clear();
  dependencyTextureModes_.clear();
  appendUnique(dependencyTextures_, dependencyTextureModes_, skyTexture,
               RenderGraphAccessMode::Read);
  for (const VolumeResource &volume : volumes_) {
    appendUnique(dependencyTextures_, dependencyTextureModes_,
                 volume.irradiance.get(), RenderGraphAccessMode::Read);
    appendUnique(dependencyTextures_, dependencyTextureModes_,
                 volume.distance.get(), RenderGraphAccessMode::Read);
  }
  for (TextureHandle reference : rt.indirectSubmissionTextureReferences) {
    appendUnique(dependencyTextures_, dependencyTextureModes_, reference,
                 RenderGraphAccessMode::Read);
  }
  const std::array traceDispatches{ComputeDispatchItem{
      .pipeline = pipelines_[Trace].get(),
      .rayQueryBinding = traceBinding_.get(),
      .dispatch = {.x = divRoundUp(schedule.primaryQueries, 64u),
                   .y = 1u,
                   .z = 1u},
      .pushConstants = bytesOf(tracePushConstants_),
      .debugLabel = "DDGI Trace",
      .debugColor = 0xff44aaffu,
  }};
  auto tracePass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
      .executionMode = RenderPassExecutionMode::ComputeOnly,
      .hasColorAttachment = false,
      .preDispatches = traceDispatches,
      .dependencyBuffers = dependencyBuffers_,
      .dependencyBufferAccessModes = dependencyBufferModes_,
      .dependencyTextures = dependencyTextures_,
      .dependencyTextureAccessModes = dependencyTextureModes_,
      .gpuTimingScope = GpuTimingScope::DDGITrace,
      .debugLabel = "DDGI Trace",
      .debugColor = 0xff44aaffu,
  });
  if (tracePass.hasError()) {
    return Result<bool, std::string>::makeError(tracePass.error());
  }
  auto tlasAccess = ctx.graph.addAccelerationStructureAccess(
      tracePass.value(), rt.graphTopLevelAccelerationStructure,
      RenderGraphAccelerationStructureAccess::RayQueryRead);
  if (tlasAccess.hasError()) {
    return tlasAccess;
  }

  blendPushConstants_.clear();
  dispatches_.clear();
  dependencyBuffers_.clear();
  dependencyBufferModes_.clear();
  dependencyTextures_.clear();
  dependencyTextureModes_.clear();
  appendUnique(dependencyBuffers_, dependencyBufferModes_, slot.frameData.get(),
               RenderGraphAccessMode::Read);
  appendUnique(dependencyBuffers_, dependencyBufferModes_, slot.updates.get(),
               RenderGraphAccessMode::Read);
  appendUnique(dependencyBuffers_, dependencyBufferModes_,
               slot.rayResults.get(), RenderGraphAccessMode::Read);
  blendPushConstants_.reserve(volumes_.size() * 2u);
  dispatches_.reserve(volumes_.size() * 2u);
  for (uint32_t volumeSlot = 0u;
       volumeSlot < static_cast<uint32_t>(volumes_.size()); ++volumeSlot) {
    const VolumeResource &volume = volumes_[volumeSlot];
    const std::array outputs{volume.irradiance.get(), volume.distance.get()};
    const std::array pipelineIndices{BlendIrradiance, BlendDistance};
    const RenderSettings::DDGISettings &volumeSettings =
        renderSettingsOrDefault(ctx.frame).ddgi;
    const bool irradianceResponse = radiometricResponseScheduled_ ||
                                    geometryResponseScheduled_ ||
                                    volume.irradianceResponseRemaining != 0u;
    const bool distanceResponse =
        geometryResponseScheduled_ || volume.distanceResponseRemaining != 0u;
    const std::array hysteresis{
        volumeSettings.irradianceHysteresis *
            (irradianceResponse ? volumeSettings.changeIrradianceHysteresisScale
                                : 1.0f),
        volumeSettings.distanceHysteresis *
            (distanceResponse ? volumeSettings.changeDistanceHysteresisScale
                              : 1.0f)};
    for (size_t type = 0u; type < outputs.size(); ++type) {
      blendPushConstants_.push_back(BlendPushConstants{
          .frame = gpu_.getBufferDeviceAddress(slot.frameData.get()),
          .updates = gpu_.getBufferDeviceAddress(slot.updates.get()),
          .results = gpu_.getBufferDeviceAddress(slot.rayResults.get()),
          .updateCount = schedule.updatedProbes,
          .volumeSlot = volumeSlot,
          .outputTextureId = gpu_.getTextureBindlessIndex(outputs[type]),
          .raysPerProbe = renderSettingsOrDefault(ctx.frame).ddgi.raysPerProbe,
          .hysteresis = hysteresis[type],
          .historyValid = submittedSequence_ != 0u ? 1u : 0u,
          .frameSeed = static_cast<uint32_t>(submittedSequence_),
      });
      dispatches_.push_back(ComputeDispatchItem{
          .pipeline = pipelines_[pipelineIndices[type]].get(),
          .dispatch = {.x = schedule.updatedProbes, .y = 1u, .z = 1u},
          .pushConstants = bytesOf(blendPushConstants_.back()),
          .dependencyBuffers = dependencyBuffers_,
          .dependencyBufferAccessModes = dependencyBufferModes_,
          .debugLabel =
              type == 0u ? "DDGI Blend Irradiance" : "DDGI Blend Distance",
          .debugColor = 0xff55dd88u,
      });
      appendUnique(dependencyTextures_, dependencyTextureModes_, outputs[type],
                   RenderGraphAccessMode::Write);
    }
  }
  auto updatePass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
      .executionMode = RenderPassExecutionMode::ComputeOnly,
      .hasColorAttachment = false,
      .preDispatches = dispatches_,
      .dependencyBuffers = dependencyBuffers_,
      .dependencyBufferAccessModes = dependencyBufferModes_,
      .dependencyTextures = dependencyTextures_,
      .dependencyTextureAccessModes = dependencyTextureModes_,
      .gpuTimingScope = GpuTimingScope::DDGIUpdate,
      .debugLabel = "DDGI Update Atlases",
      .debugColor = 0xff55dd88u,
  });
  if (updatePass.hasError()) {
    return Result<bool, std::string>::makeError(updatePass.error());
  }

  statePushConstants_ = {};
  statePushConstants_.updates = gpu_.getBufferDeviceAddress(slot.updates.get());
  statePushConstants_.results =
      gpu_.getBufferDeviceAddress(slot.rayResults.get());
  statePushConstants_.frame = gpu_.getBufferDeviceAddress(slot.frameData.get());
  statePushConstants_.updateCount = schedule.updatedProbes;
  statePushConstants_.submittedSequence =
      static_cast<uint32_t>(submittedSequence_ + 1u);
  const RenderSettings::DDGISettings &settings =
      renderSettingsOrDefault(ctx.frame).ddgi;
  statePushConstants_.raysPerProbe = settings.raysPerProbe;
  statePushConstants_.frameSeed = static_cast<uint32_t>(submittedSequence_);
  statePushConstants_.relocationEnabled = settings.relocation ? 1u : 0u;
  statePushConstants_.classificationEnabled = settings.classification ? 1u : 0u;
  dependencyBuffers_.clear();
  dependencyBufferModes_.clear();
  appendUnique(dependencyBuffers_, dependencyBufferModes_, slot.updates.get(),
               RenderGraphAccessMode::Read);
  appendUnique(dependencyBuffers_, dependencyBufferModes_,
               slot.rayResults.get(), RenderGraphAccessMode::Read);
  appendUnique(dependencyBuffers_, dependencyBufferModes_, slot.frameData.get(),
               RenderGraphAccessMode::Read);
  for (size_t index = 0u; index < volumes_.size(); ++index) {
    statePushConstants_.states[index] =
        gpu_.getBufferDeviceAddress(volumes_[index].probeState.get());
    appendUnique(dependencyBuffers_, dependencyBufferModes_,
                 volumes_[index].probeState.get(),
                 RenderGraphAccessMode::Write);
  }
  const std::array stateDispatches{ComputeDispatchItem{
      .pipeline = pipelines_[UpdateProbeState].get(),
      .dispatch = {.x = divRoundUp(schedule.updatedProbes, 64u),
                   .y = 1u,
                   .z = 1u},
      .pushConstants = bytesOf(statePushConstants_),
      .dependencyBuffers = dependencyBuffers_,
      .dependencyBufferAccessModes = dependencyBufferModes_,
      .debugLabel = "DDGI Update Probe State",
      .debugColor = 0xff55bb77u,
  }};
  auto statePass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
      .executionMode = RenderPassExecutionMode::ComputeOnly,
      .hasColorAttachment = false,
      .preDispatches = stateDispatches,
      .dependencyBuffers = dependencyBuffers_,
      .dependencyBufferAccessModes = dependencyBufferModes_,
      .gpuTimingScope = GpuTimingScope::DDGIRelocateClassify,
      .debugLabel = "DDGI Update Probe State",
      .debugColor = 0xff55bb77u,
  });
  if (statePass.hasError()) {
    return Result<bool, std::string>::makeError(statePass.error());
  }
  updatesScheduled_ = true;
  scheduledFrameIndex_ = ctx.frame.frameIndex;
  pendingForceEpoch_ =
      renderSettingsOrDefault(ctx.frame).ddgi.requestedEpochs.forceFullUpdate;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DDGIFeature::appendInspectionPass(FrameBuildContext &ctx, FrameSlot &slot) {
  if (!ctx.frame.ddgiProbeInspectRequest.has_value() ||
      !ctx.shared.rayTracingScene.has_value() ||
      !ctx.shared.rayTracingScene->ready) {
    return Result<bool, std::string>::makeResult(true);
  }
  const DDGIProbeInspectRequest request = *ctx.frame.ddgiProbeInspectRequest;
  ctx.frame.ddgiProbeInspectRequest.reset();
  latestInspectionRequestId_ =
      std::max(latestInspectionRequestId_, request.requestId);
  const auto found =
      std::ranges::find(volumes_, request.volume, &VolumeResource::id);
  const uint32_t volumeSlot =
      found == volumes_.end() ? std::numeric_limits<uint32_t>::max()
                              : static_cast<uint32_t>(found - volumes_.begin());
  if (found == volumes_.end() ||
      request.probeId >= found->lastSubmittedUpdates.size()) {
    latestInspectionResult_ = DDGIProbeInspectResult{
        .requestId = request.requestId,
        .sceneId = ctx.frame.scene != nullptr ? ctx.frame.scene->id() : 0u,
        .volume = request.volume,
        .probeId = request.probeId,
        .valid = false,
    };
    ctx.frame.ddgiProbeInspectResult = latestInspectionResult_;
    return Result<bool, std::string>::makeResult(true);
  }
  if (!slot.diagnostic.valid()) {
    auto buffer =
        gpu_.createBuffer(BufferDesc{.usage = BufferUsage::Storage,
                                     .storage = Storage::Device,
                                     .size = kDDGIDiagnosticBufferBytes},
                          "ddgi_probe_diagnostic");
    if (buffer.hasError()) {
      return Result<bool, std::string>::makeError(buffer.error());
    }
    slot.diagnostic.reset(gpu_, buffer.value());
  }
  const DDGIDiagnosticHeaderGpuData emptyHeader{};
  auto clear = gpu_.updateBuffer(slot.diagnostic.get(), bytesOf(emptyHeader));
  if (clear.hasError()) {
    return clear;
  }
  const RayTracingSceneFrameView &rt = *ctx.shared.rayTracingScene;
  if (!inspectBinding_.valid() ||
      inspectBoundTlas_ != rt.topLevelAccelerationStructure) {
    inspectBinding_.reset();
    auto binding = gpu_.createRayQueryBinding(
        pipelines_[TraceInspect].get(), rt.topLevelAccelerationStructure,
        "ddgi_trace_inspect_tlas_binding");
    if (binding.hasError()) {
      return Result<bool, std::string>::makeError(binding.error());
    }
    inspectBinding_.reset(gpu_, binding.value());
    inspectBoundTlas_ = rt.topLevelAccelerationStructure;
  }
  const uint32_t rayCount =
      std::clamp(request.rayCount, 1u, kDDGIMaxDiagnosticRays);
  const uint64_t diagnosticAddress =
      gpu_.getBufferDeviceAddress(slot.diagnostic.get());
  const MaterialTableGpuData &materials = *ctx.shared.materialTableGpuData;
  inspectPushConstants_ = InspectPushConstants{
      .frame = gpu_.getBufferDeviceAddress(slot.frameData.get()),
      .header = diagnosticAddress,
      .rays = diagnosticAddress + sizeof(DDGIDiagnosticHeaderGpuData),
      .events = diagnosticAddress + sizeof(DDGIDiagnosticHeaderGpuData) +
                kDDGIMaxDiagnosticRays * sizeof(DDGIDiagnosticRayGpuData),
      .instances = rt.instanceTableAddress,
      .geometries = rt.geometryTableAddress,
      .materials = materials.headerBufferAddress,
      .requestId = request.requestId,
      .sceneId = ctx.frame.scene->id(),
      .layoutGeneration = found->layout.generation,
      .resourceGeneration = found->resourceGeneration,
      .deviceEpoch = deviceEpoch_,
      .volumeValue = request.volume.value,
      .volumeSlot = volumeSlot,
      .probeId = request.probeId,
      .rayCount = rayCount,
      .maxCandidates = renderSettingsOrDefault(ctx.frame)
                           .ddgi.maxCandidateIntersectionsPerRay,
      .frameSeed = static_cast<uint32_t>(submittedSequence_),
      .materialSamplerId = gpu_.getDefaultSamplerBindlessIndex(),
      .submissionSequence = static_cast<uint32_t>(
          submittedSequence_ +
          ((initializationScheduled_ || scrollScheduled_ || updatesScheduled_)
               ? 1u
               : 0u)),
  };
  dependencyBuffers_.clear();
  dependencyBufferModes_.clear();
  dependencyTextures_.clear();
  dependencyTextureModes_.clear();
  appendUnique(dependencyBuffers_, dependencyBufferModes_, slot.frameData.get(),
               RenderGraphAccessMode::Read);
  appendUnique(dependencyBuffers_, dependencyBufferModes_,
               slot.diagnostic.get(), RenderGraphAccessMode::Write);
  for (const VolumeResource &volume : volumes_) {
    appendUnique(dependencyBuffers_, dependencyBufferModes_,
                 volume.probeState.get(), RenderGraphAccessMode::Read);
    appendUnique(dependencyTextures_, dependencyTextureModes_,
                 volume.irradiance.get(), RenderGraphAccessMode::Read);
    appendUnique(dependencyTextures_, dependencyTextureModes_,
                 volume.distance.get(), RenderGraphAccessMode::Read);
  }
  for (BufferHandle reference : rt.indirectSubmissionReferences) {
    appendUnique(dependencyBuffers_, dependencyBufferModes_, reference,
                 RenderGraphAccessMode::Read);
  }
  for (TextureHandle reference : rt.indirectSubmissionTextureReferences) {
    appendUnique(dependencyTextures_, dependencyTextureModes_, reference,
                 RenderGraphAccessMode::Read);
  }
  for (BufferHandle reference :
       {materials.headerBuffer, materials.clearcoatBuffer,
        materials.sheenBuffer, materials.transmissionBuffer,
        materials.specularBuffer}) {
    appendUnique(dependencyBuffers_, dependencyBufferModes_, reference,
                 RenderGraphAccessMode::Read);
  }
  const std::array inspectDispatches{ComputeDispatchItem{
      .pipeline = pipelines_[TraceInspect].get(),
      .rayQueryBinding = inspectBinding_.get(),
      .dispatch = {.x = divRoundUp(rayCount, 64u), .y = 1u, .z = 1u},
      .pushConstants = bytesOf(inspectPushConstants_),
      .debugLabel = "DDGI Probe Diagnostic Trace",
      .debugColor = 0xffff55aau,
  }};
  auto pass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
      .executionMode = RenderPassExecutionMode::ComputeOnly,
      .hasColorAttachment = false,
      .preDispatches = inspectDispatches,
      .dependencyBuffers = dependencyBuffers_,
      .dependencyBufferAccessModes = dependencyBufferModes_,
      .dependencyTextures = dependencyTextures_,
      .dependencyTextureAccessModes = dependencyTextureModes_,
      .gpuTimingScope = GpuTimingScope::DDGITrace,
      .debugLabel = "DDGI Probe Inspection",
      .debugColor = 0xffff55aau,
  });
  if (pass.hasError()) {
    return Result<bool, std::string>::makeError(pass.error());
  }
  auto access = ctx.graph.addAccelerationStructureAccess(
      pass.value(), rt.graphTopLevelAccelerationStructure,
      RenderGraphAccelerationStructureAccess::RayQueryRead);
  if (access.hasError()) {
    return access;
  }
  slot.diagnosticRequest = request;
  slot.diagnosticRequest->rayCount = rayCount;
  slot.diagnosticValid = false;
  inspectionScheduled_ = true;
  scheduledFrameIndex_ = ctx.frame.frameIndex;
  return Result<bool, std::string>::makeResult(true);
}

void DDGIFeature::publishCapturePoints(RenderFrameContext &frame) const {
  static constexpr std::array irradianceNames{
      "ddgi_volume0_irradiance_atlas", "ddgi_volume1_irradiance_atlas",
      "ddgi_volume2_irradiance_atlas", "ddgi_volume3_irradiance_atlas"};
  static constexpr std::array distanceNames{
      "ddgi_volume0_distance_atlas", "ddgi_volume1_distance_atlas",
      "ddgi_volume2_distance_atlas", "ddgi_volume3_distance_atlas"};
  for (size_t slot = 0u; slot < volumes_.size(); ++slot) {
    if (!volumes_[slot].ready) {
      continue;
    }
    for (const auto point :
         {RenderCapturePoint{
              .name = irradianceNames[slot],
              .version = kDDGILayoutVersion,
              .texture = volumes_[slot].irradiance.get(),
              .format = Format::RGBA16_FLOAT,
              .dimensions =
                  gpu_.getTextureDimensions(volumes_[slot].irradiance.get()),
              .frameIndex = frame.frameIndex,
              .kind = RenderCaptureValueKind::LinearHdrColor,
              .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
              .colorSpace = "ddgi_perception_gamma5",
              .defaultCompareProfile = "ddgi_irradiance_v1",
              .producerPassLabel = "DDGI Update Atlases",
              .debugLabel = "DDGI Irradiance Atlas"},
          RenderCapturePoint{
              .name = distanceNames[slot],
              .version = kDDGILayoutVersion,
              .texture = volumes_[slot].distance.get(),
              .format = Format::RG16_FLOAT,
              .dimensions =
                  gpu_.getTextureDimensions(volumes_[slot].distance.get()),
              .frameIndex = frame.frameIndex,
              .kind = RenderCaptureValueKind::Scalar,
              .lifetime = RenderCaptureLifetimeClass::FeaturePersistentTexture,
              .colorSpace = "linear_normalized_distance",
              .defaultCompareProfile = "ddgi_distance_v1",
              .producerPassLabel = "DDGI Update Atlases",
              .debugLabel = "DDGI Distance Atlas"}}) {
      frame.captureRegistry.publish(point);
    }
  }
}

void DDGIFeature::collectDebugProbeStateMetrics(FrameBuildContext &ctx) {
  const RenderSettings::DDGISettings &settings =
      renderSettingsOrDefault(ctx.frame).ddgi;
  if (!settings.showProbes && !settings.showSelectedProbeRays &&
      settings.debugView != DDGIDebugView::Classification &&
      settings.debugView != DDGIDebugView::RelocationOffset) {
    return;
  }
  ScopedScratch scoped(scratch_);
  uint32_t vigilant = 0u;
  uint32_t uninitialized = 0u;
  uint32_t off = 0u;
  uint32_t sleeping = 0u;
  uint32_t newlyAwake = 0u;
  uint32_t awake = 0u;
  uint32_t newlyVigilant = 0u;
  uint32_t relocated = 0u;
  float maxRelocation = 0.0f;
  for (const VolumeResource &volume : volumes_) {
    std::pmr::vector<DDGIProbeStateGpuData> states(scoped.resource());
    states.resize(volume.lastSubmittedUpdates.size());
    auto read = gpu_.readBuffer(
        volume.probeState.get(), 0u,
        std::as_writable_bytes(std::span(states.data(), states.size())));
    if (read.hasError()) {
      return;
    }
    for (const DDGIProbeStateGpuData &state : states) {
      const auto probeState =
          static_cast<DDGIProbeState>(state.stateAgeFlags.x);
      uninitialized += probeState == DDGIProbeState::Uninitialized ? 1u : 0u;
      off += probeState == DDGIProbeState::Off ? 1u : 0u;
      sleeping += probeState == DDGIProbeState::Sleeping ? 1u : 0u;
      newlyAwake += probeState == DDGIProbeState::NewlyAwake ? 1u : 0u;
      awake += probeState == DDGIProbeState::Awake ? 1u : 0u;
      newlyVigilant += probeState == DDGIProbeState::NewlyVigilant ? 1u : 0u;
      vigilant += probeState == DDGIProbeState::Vigilant ? 1u : 0u;
      const float relocation = glm::length(glm::vec3(state.relocation));
      relocated += relocation > 1.0e-5f ? 1u : 0u;
      maxRelocation = std::max(maxRelocation, relocation);
    }
  }
  DDGIFrameMetrics &metrics = ctx.frame.metrics.ddgi;
  metrics.vigilantProbes = vigilant;
  metrics.uninitializedProbes = uninitialized;
  metrics.offProbes = off;
  metrics.sleepingProbes = sleeping;
  metrics.newlyAwakeProbes = newlyAwake;
  metrics.awakeProbes = awake;
  metrics.newlyVigilantProbes = newlyVigilant;
  metrics.relocatedProbes = relocated;
  metrics.maxRelocation = maxRelocation;
  metrics.probeStateReadbackAvailable = 1u;
}

void DDGIFeature::collectCompletedTraceMetrics(FrameSlot &slot,
                                               DDGIFrameMetrics &metrics) {
  if (slot.traceCountersValid) {
    DDGITraceCountersGpuData counters{};
    auto read = gpu_.readBuffer(
        slot.localLights.get(), 0u,
        std::as_writable_bytes(std::span(&counters, static_cast<size_t>(1u))));
    if (!read.hasError()) {
      metrics.secondaryQueriesReserved = counters.reserved;
      metrics.secondaryQueries = counters.secondaryQueries;
      metrics.secondaryQueriesUnused =
          counters.reserved -
          std::min(counters.reserved, counters.secondaryQueries);
      metrics.primaryCandidateIntersections =
          counters.primaryCandidateIntersections;
      metrics.secondaryCandidateIntersections =
          counters.secondaryCandidateIntersections;
      metrics.alphaCandidateRejections = counters.alphaCandidateRejections;
      metrics.backfaceCandidateRejections =
          counters.backfaceCandidateRejections;
      metrics.candidateOverflows = counters.candidateOverflows;
      metrics.localLightTruncations = counters.localLightTruncations;
    }
  }
  slot.traceCountersValid = false;
}

void DDGIFeature::collectCompletedInspection(FrameBuildContext &ctx,
                                             FrameSlot &slot) {
  if (!slot.diagnosticValid || !slot.diagnosticRequest.has_value() ||
      slot.diagnosticRequest->requestId != latestInspectionRequestId_) {
    return;
  }
  DDGIDiagnosticHeaderGpuData header{};
  auto read = gpu_.readBuffer(
      slot.diagnostic.get(), 0u,
      std::as_writable_bytes(std::span(&header, static_cast<size_t>(1u))));
  const DDGIProbeInspectRequest request = *slot.diagnosticRequest;
  slot.diagnosticValid = false;
  slot.diagnosticRequest.reset();
  if (read.hasError()) {
    return;
  }
  DDGIProbeInspectResult result{
      .requestId =
          join32(header.requestAndSelection.x, header.requestAndSelection.y),
      .sceneId = join32(header.identity0.x, header.identity0.y),
      .volume = DDGIVolumeId{header.identity0.z},
      .probeId = header.requestAndSelection.w,
      .volumeSlot = header.requestAndSelection.z,
      .probeState = header.counts1.z,
      .lastSuccessfulUpdate = header.counts1.w,
      .hitCount = header.counts0.x,
      .missCount = header.counts0.y,
      .rejectedAlphaCount = header.counts0.z,
      .rejectedBackfaceCount = header.counts0.w,
      .candidateOverflowCount = header.counts1.x,
      .diagnosticEventOverflowCount = header.counts1.y,
      .rayCount = header.identity0.w,
      .nominalWorldPosition = glm::vec3(header.nominalWorldPosition),
      .relocatedWorldPosition = glm::vec3(header.relocatedWorldPosition),
      .irradiance = glm::vec3(header.irradianceDistanceMean),
      .distanceMoments = glm::vec2(header.irradianceDistanceMean.w,
                                   header.distanceSecondReserved.x),
      .layoutGeneration = join32(header.identity1.x, header.identity1.y),
      .resourceGeneration = join32(header.identity1.z, header.identity1.w),
      .deviceEpoch = join32(header.identity2.x, header.identity2.y),
      .submissionSequence = header.identity2.z,
  };
  result.updateAge = static_cast<uint32_t>(result.submissionSequence) -
                     std::min(static_cast<uint32_t>(result.submissionSequence),
                              result.lastSuccessfulUpdate);
  result.active =
      result.probeState == static_cast<uint32_t>(DDGIProbeState::Awake) ||
      result.probeState == static_cast<uint32_t>(DDGIProbeState::Vigilant);
  result.inside =
      result.probeState == static_cast<uint32_t>(DDGIProbeState::Off);
  const auto volume =
      std::ranges::find(volumes_, request.volume, &VolumeResource::id);
  if (volume != volumes_.end()) {
    result.volumeCoordinate =
        ddgiProbeCoordinate(result.probeId, volume->layout.probeCounts);
    result.irradianceAtlasCoordinate =
        ddgiAtlasTileCoordinate(result.probeId, volume->layout.irradianceAtlas);
    result.distanceAtlasCoordinate =
        ddgiAtlasTileCoordinate(result.probeId, volume->layout.distanceAtlas);
  }
  result.valid =
      result.requestId == request.requestId &&
      result.requestId == latestInspectionRequestId_ &&
      ctx.frame.scene != nullptr && result.sceneId == ctx.frame.scene->id() &&
      result.volume == request.volume && result.probeId == request.probeId &&
      volume != volumes_.end() &&
      result.layoutGeneration == volume->layout.generation &&
      result.resourceGeneration == volume->resourceGeneration &&
      result.deviceEpoch == deviceEpoch_;
  latestInspectionResult_ = result;
  ctx.frame.ddgiProbeInspectResult = result;
}

void DDGIFeature::publishFrameData(FrameBuildContext &ctx, FrameSlot &slot,
                                   bool rtReady) {
  forwardDependencyBuffers_.clear();
  forwardDependencyTextures_.clear();
  uint32_t readyCount = 0u;
  for (const VolumeResource &volume : volumes_) {
    if (!volume.ready) {
      continue;
    }
    ++readyCount;
    forwardDependencyBuffers_.push_back(volume.probeState.get());
    forwardDependencyTextures_.push_back(volume.irradiance.get());
    forwardDependencyTextures_.push_back(volume.distance.get());
  }
  if (rtReady && readyCount == volumes_.size() && readyCount != 0u) {
    std::array<DDGIVolumeId, kMaxDDGIVolumes> volumeIds{};
    std::array<uint32_t, kMaxDDGIVolumes> probeCounts{};
    std::array<float, kMaxDDGIVolumes> minimumProbeSpacing{};
    for (uint32_t slotIndex = 0u; slotIndex < readyCount; ++slotIndex) {
      const VolumeResource &volume = volumes_[slotIndex];
      volumeIds[slotIndex] = volume.id;
      probeCounts[slotIndex] =
          static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
      minimumProbeSpacing[slotIndex] =
          std::min({volume.layout.probeSpacing.x, volume.layout.probeSpacing.y,
                    volume.layout.probeSpacing.z});
    }
    forwardDependencyBuffers_.push_back(slot.frameData.get());
    if (inspectionScheduled_ && slot.diagnostic.valid()) {
      forwardDependencyBuffers_.push_back(slot.diagnostic.get());
    }
    ctx.shared.ddgiFrameGpuData = DDGIFrameGpuDataHandle{
        .buffer = slot.frameData.get(),
        .bufferAddress = gpu_.getBufferDeviceAddress(slot.frameData.get()),
        .dependencyBuffers = forwardDependencyBuffers_,
        .dependencyTextures = forwardDependencyTextures_,
        .activeVolumeCount = readyCount,
        .flags = kDDGIFrameEnabled,
        .debugView = renderSettingsOrDefault(ctx.frame).ddgi.debugView,
        .volumeIds = volumeIds,
        .probeCounts = probeCounts,
        .minimumProbeSpacing = minimumProbeSpacing,
        .diagnosticBuffer =
            inspectionScheduled_ ? slot.diagnostic.get() : BufferHandle{},
        .diagnosticRayAddress =
            inspectionScheduled_
                ? gpu_.getBufferDeviceAddress(slot.diagnostic.get()) +
                      sizeof(DDGIDiagnosticHeaderGpuData)
                : 0u,
        .diagnosticRayCount =
            inspectionScheduled_ && slot.diagnosticRequest.has_value()
                ? slot.diagnosticRequest->rayCount
                : 0u,
    };
  }
  publishCapturePoints(ctx.frame);
  DDGIFrameMetrics &metrics = ctx.frame.metrics.ddgi;
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::DDGI)) {
    metrics.gpuTimeMs = ctx.frame.gpuTiming.ddgiTimeMs;
    metrics.gpuTimingSourceFrameIndex =
        ctx.frame.gpuTiming.ddgiSourceFrameIndex;
    metrics.gpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::DDGITrace)) {
    metrics.traceGpuTimeMs = ctx.frame.gpuTiming.ddgiTraceTimeMs;
    metrics.traceGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::DDGIUpdate)) {
    metrics.updateGpuTimeMs = ctx.frame.gpuTiming.ddgiUpdateTimeMs;
    metrics.updateGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming,
                        GpuTimingScope::DDGIRelocateClassify)) {
    metrics.relocateClassifyGpuTimeMs =
        ctx.frame.gpuTiming.ddgiRelocateClassifyTimeMs;
    metrics.relocateClassifyGpuTimingAvailable = 1u;
  }
  metrics.activeVolumes = static_cast<uint32_t>(volumes_.size());
  metrics.readyVolumes = readyCount;
  metrics.failedVolumes =
      replacementPending_ ? pendingFailedVolumeCount_ : failedVolumeCount_;
  metrics.volumeFailureReason =
      replacementPending_ ? pendingVolumeFailureReason_ : volumeFailureReason_;
  metrics.historyReady =
      rtReady && readyCount == volumes_.size() && readyCount != 0u;
  metrics.skyFallbackActive = metrics.historyReady ? 0u : 1u;
  metrics.debugView = renderSettingsOrDefault(ctx.frame).ddgi.debugView;
  metrics.submittedSequence = submittedSequence_;
  metrics.deviceEpoch = deviceEpoch_;
  metrics.consumedResetEpoch = consumedResetEpoch_;
  metrics.consumedForceUpdateEpoch = consumedForceEpoch_;
  if (latestInspectionResult_.has_value()) {
    const DDGIProbeInspectResult &inspection = *latestInspectionResult_;
    metrics.inspectionAvailable = 1u;
    metrics.inspectionValid = inspection.valid ? 1u : 0u;
    metrics.inspectionRayCount = inspection.rayCount;
    metrics.inspectionHitCount = inspection.hitCount;
    metrics.inspectionMissCount = inspection.missCount;
    metrics.inspectionCandidateOverflows = inspection.candidateOverflowCount;
    metrics.inspectionEventOverflows = inspection.diagnosticEventOverflowCount;
  }
  for (const VolumeResource &volume : volumes_) {
    metrics.layoutGeneration =
        std::max(metrics.layoutGeneration, volume.layout.generation);
    metrics.resourceGeneration =
        std::max(metrics.resourceGeneration, volume.resourceGeneration);
    metrics.totalProbes +=
        static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
    metrics.vigilantProbes +=
        static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
    metrics.persistentBytes += volume.persistentBytes;
    metrics.irradianceResponseRemaining =
        std::max(metrics.irradianceResponseRemaining,
                 volume.irradianceResponseRemaining);
    metrics.distanceResponseRemaining = std::max(
        metrics.distanceResponseRemaining, volume.distanceResponseRemaining);
  }
  metrics.frameBatchBytes =
      slot.updateCapacity * sizeof(DDGIProbeUpdateEntry) +
      slot.invalidationCapacity * sizeof(DDGIProbeUpdateEntry) +
      slot.rayCapacity * sizeof(DDGIRayResultGpuData) +
      sizeof(DDGITraceCountersGpuData) +
      slot.localLightCapacity * sizeof(LocalLightGpuData) +
      sizeof(DDGIFrameGpuData) +
      (slot.diagnostic.valid() ? kDDGIDiagnosticBufferBytes : 0u);
  collectDebugProbeStateMetrics(ctx);
}

Result<bool, std::string> DDGIFeature::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  ctx.shared.ddgiFrameGpuData.reset();
  ctx.frame.ddgiProbeInspectResult.reset();
  ctx.frame.metrics.ddgi = {};
  ctx.frame.metrics.ddgi.deviceEpoch = deviceEpoch_;
  ctx.frame.metrics.ddgi.consumedResetEpoch = consumedResetEpoch_;
  ctx.frame.metrics.ddgi.consumedForceUpdateEpoch = consumedForceEpoch_;
  ctx.frame.metrics.ddgi.debugView =
      renderSettingsOrDefault(ctx.frame).ddgi.debugView;
  initializationScheduled_ = false;
  scrollScheduled_ = false;
  updatesScheduled_ = false;
  radiometricResponseScheduled_ = false;
  geometryResponseScheduled_ = false;
  inspectionScheduled_ = false;
  if (ctx.frame.ddgiProbeInspectRequest.has_value()) {
    if (ctx.frame.ddgiProbeInspectRequest->requestId >
        latestInspectionRequestId_) {
      latestInspectionResult_.reset();
    }
    latestInspectionRequestId_ =
        std::max(latestInspectionRequestId_,
                 ctx.frame.ddgiProbeInspectRequest->requestId);
  }
  const RenderSettings::DDGISettings &settings =
      renderSettingsOrDefault(ctx.frame).ddgi;
  if (!settings.enabled || ctx.frame.scene == nullptr) {
    clearVolumes();
    clearFrameSlots();
    ctx.frame.metrics.ddgi.fallbackReason = DDGIFallbackReason::Disabled;
    return Result<bool, std::string>::makeResult(true);
  }
  const RayTracingCapabilities &caps = gpu_.getDeviceCaps().rayTracing;
  if (!caps.accelerationStructure || !caps.rayQuery ||
      !caps.bufferDeviceAddress) {
    clearVolumes();
    clearFrameSlots();
    ctx.frame.metrics.ddgi.fallbackReason = DDGIFallbackReason::Unsupported;
    return Result<bool, std::string>::makeResult(true);
  }
  if (!initialized_ && initializationError_.empty()) {
    auto initialization = initialize();
    if (initialization.hasError()) {
      initializationError_ = initialization.error();
      NURI_LOG_ERROR(
          "DDGIFeature: optional DDGI shader initialization failed: %s",
          initializationError_.c_str());
    } else {
      initialized_ = true;
    }
  }
  if (!initializationError_.empty()) {
    ctx.frame.metrics.ddgi.fallbackReason =
        DDGIFallbackReason::ShaderUnavailable;
    return Result<bool, std::string>::makeResult(true);
  }
  const bool resetRequested = ddgiEpochIsPending(
      settings.requestedEpochs.resetHistory, consumedResetEpoch_);
  const RenderScene &scene = *ctx.frame.scene;
  const bool volumesChanged =
      sceneId_ != scene.id() ||
      sceneTopologyVersion_ != scene.topologyVersion() ||
      volumeTopologyVersion_ != scene.ddgiVolumeTopologyVersion() ||
      volumeTransformVersion_ != scene.ddgiVolumeTransformVersion() ||
      volumeSettingsVersion_ != scene.ddgiVolumeSettingsVersion() ||
      (!volumes_.empty() &&
       (relocationEnabled_ != settings.relocation ||
        classificationEnabled_ != settings.classification));
  if (volumesChanged || resetRequested) {
    auto rebuild = rebuildVolumes(ctx);
    if (rebuild.hasError()) {
      ctx.frame.metrics.ddgi.fallbackReason =
          DDGIFallbackReason::AllocationFailed;
      return Result<bool, std::string>::makeResult(true);
    }
  }
  if (!replacementPending_ && !settings.freezeUpdates) {
    buildScrollPlan(ctx.frame);
  } else {
    pendingScrollLayouts_.clear();
    scrollInvalidations_.clear();
  }
  auto slots = ensureFrameSlots(settings, scene.packedLocalLights().size(),
                                scrollInvalidations_.size());
  if (slots.hasError()) {
    ctx.frame.metrics.ddgi.fallbackReason =
        DDGIFallbackReason::AllocationFailed;
    return Result<bool, std::string>::makeResult(true);
  }
  FrameSlot &slot = frameSlots_[static_cast<size_t>(ctx.frame.frameIndex %
                                                    frameSlots_.size())];
  collectCompletedTraceMetrics(slot, ctx.frame.metrics.ddgi);
  collectCompletedInspection(ctx, slot);
  auto frameUpload = updateFrameData(ctx, slot);
  if (frameUpload.hasError()) {
    return frameUpload;
  }
  if (replacementPending_) {
    auto initialization = appendInitializationPass(ctx, slot);
    if (initialization.hasError()) {
      return initialization;
    }
    const bool rtReady = ctx.shared.rayTracingScene.has_value() &&
                         ctx.shared.rayTracingScene->ready;
    publishFrameData(ctx, slot, rtReady);
    ctx.frame.metrics.ddgi.fallbackReason =
        DDGIFallbackReason::VolumeResourcesWarming;
    ++ctx.frame.metrics.ddgi.resetCount;
    return Result<bool, std::string>::makeResult(true);
  }
  if (volumes_.empty()) {
    ctx.frame.metrics.ddgi.failedVolumes = failedVolumeCount_;
    ctx.frame.metrics.ddgi.volumeFailureReason = volumeFailureReason_;
    ctx.frame.metrics.ddgi.fallbackReason =
        failedVolumeCount_ != 0u ? DDGIFallbackReason::AllocationFailed
                                 : DDGIFallbackReason::NoVolumes;
    return Result<bool, std::string>::makeResult(true);
  }
  if (!scrollInvalidations_.empty()) {
    auto scroll = appendScrollPass(ctx, slot);
    if (scroll.hasError()) {
      return scroll;
    }
    ++ctx.frame.metrics.ddgi.scrollCount;
    ctx.frame.metrics.ddgi.invalidatedProbes =
        static_cast<uint32_t>(scrollInvalidations_.size());
  }
  const bool rtReady = ctx.shared.rayTracingScene.has_value() &&
                       ctx.shared.rayTracingScene->ready;
  if (!settings.freezeUpdates && rtReady) {
    radiometricResponseScheduled_ =
        lightTopologyVersion_ != scene.lightTopologyVersion() ||
        lightTransformVersion_ != scene.lightTransformVersion() ||
        materialVersion_ != ctx.resources.materialVersion() ||
        environmentVersion_ != scene.environmentVersion();
    const uint64_t currentDeformationVersion =
        ctx.shared.rayTracingScene.has_value()
            ? ctx.shared.rayTracingScene->deformationVersion
            : scene.deformationVersion();
    geometryResponseScheduled_ =
        sceneTransformVersion_ != scene.transformVersion() ||
        sceneDeformationVersion_ != currentDeformationVersion;
    auto schedule = buildSchedule(settings);
    if (schedule.hasError()) {
      return Result<bool, std::string>::makeError(schedule.error());
    }
    if (schedule.value().updatedProbes != 0u) {
      auto update = appendUpdatePasses(ctx, slot, schedule.value());
      if (update.hasError()) {
        return update;
      }
    }
    ctx.frame.metrics.ddgi.updatedProbes = schedule.value().updatedProbes;
    ctx.frame.metrics.ddgi.primaryQueries = schedule.value().primaryQueries;
  }
  if (rtReady && ctx.frame.ddgiProbeInspectRequest.has_value()) {
    auto inspection = appendInspectionPass(ctx, slot);
    if (inspection.hasError()) {
      return inspection;
    }
  }
  ctx.frame.metrics.ddgi.probeUpdateCapacity = settings.maxProbeUpdatesPerFrame;
  ctx.frame.metrics.ddgi.rayQueryCapacity = settings.maxRayQueriesPerFrame;
  publishFrameData(ctx, slot, rtReady);
  ctx.frame.metrics.ddgi.fallbackReason =
      rtReady ? DDGIFallbackReason::None
              : DDGIFallbackReason::RayTracingSceneWarming;
  return Result<bool, std::string>::makeResult(true);
}

void DDGIFeature::onFrameSubmitted(const RenderFrameContext &frame) noexcept {
  if (scheduledFrameIndex_ != frame.frameIndex) {
    return;
  }
  const bool submittedWork =
      initializationScheduled_ || scrollScheduled_ || updatesScheduled_;
  const uint64_t committedSequence =
      submittedSequence_ + (submittedWork ? 1u : 0u);
  if (initializationScheduled_) {
    for (VolumeResource &volume : pendingVolumes_) {
      volume.ready = true;
    }
    volumes_.swap(pendingVolumes_);
    pendingVolumes_.clear();
    replacementPending_ = false;
    sceneId_ = pendingSceneId_;
    volumeTopologyVersion_ = pendingVolumeTopologyVersion_;
    volumeTransformVersion_ = pendingVolumeTransformVersion_;
    volumeSettingsVersion_ = pendingVolumeSettingsVersion_;
    relocationEnabled_ = pendingRelocationEnabled_;
    classificationEnabled_ = pendingClassificationEnabled_;
    failedVolumeCount_ = pendingFailedVolumeCount_;
    volumeFailureReason_ = pendingVolumeFailureReason_;
    consumedResetEpoch_ = std::max(consumedResetEpoch_, pendingResetEpoch_);
    if (frame.scene != nullptr && frame.resources != nullptr) {
      sceneTopologyVersion_ = frame.scene->topologyVersion();
      sceneTransformVersion_ = frame.scene->transformVersion();
      sceneDeformationVersion_ =
          frame.sharedResources.rayTracingScene.has_value()
              ? frame.sharedResources.rayTracingScene->deformationVersion
              : frame.scene->deformationVersion();
      lightTopologyVersion_ = frame.scene->lightTopologyVersion();
      lightTransformVersion_ = frame.scene->lightTransformVersion();
      materialVersion_ = frame.resources->materialVersion();
      environmentVersion_ = frame.scene->environmentVersion();
    }
  }
  if (scrollScheduled_) {
    for (const DDGIProbeUpdateEntry &entry : scrollInvalidations_) {
      if (entry.volumeStableId < volumes_.size() &&
          entry.probeId <
              volumes_[entry.volumeStableId].lastSubmittedUpdates.size()) {
        volumes_[entry.volumeStableId].lastSubmittedUpdates[entry.probeId] = 0u;
      }
    }
    if (pendingScrollLayouts_.size() == volumes_.size()) {
      for (size_t index = 0u; index < volumes_.size(); ++index) {
        volumes_[index].layout = pendingScrollLayouts_[index];
      }
    }
  }
  if (updatesScheduled_) {
    if (radiometricResponseScheduled_) {
      for (VolumeResource &volume : volumes_) {
        volume.irradianceResponseRemaining =
            std::max(volume.irradianceResponseRemaining, 10u);
      }
    }
    if (geometryResponseScheduled_) {
      for (VolumeResource &volume : volumes_) {
        volume.irradianceResponseRemaining =
            std::max(volume.irradianceResponseRemaining, 10u);
        volume.distanceResponseRemaining =
            std::max(volume.distanceResponseRemaining, 7u);
      }
    }
    std::array<bool, kMaxDDGIVolumes> volumeUpdated{};
    for (const DDGIProbeUpdateEntry &entry : scheduledEntries_) {
      if (entry.volumeStableId < volumes_.size() &&
          entry.probeId <
              volumes_[entry.volumeStableId].lastSubmittedUpdates.size()) {
        volumes_[entry.volumeStableId].lastSubmittedUpdates[entry.probeId] =
            committedSequence;
        volumeUpdated[entry.volumeStableId] = true;
      }
    }
    for (size_t slot = 0u; slot < volumes_.size(); ++slot) {
      if (!volumeUpdated[slot]) {
        continue;
      }
      VolumeResource &volume = volumes_[slot];
      if (volume.irradianceResponseRemaining != 0u) {
        --volume.irradianceResponseRemaining;
      }
      if (volume.distanceResponseRemaining != 0u) {
        --volume.distanceResponseRemaining;
      }
    }
    consumedForceEpoch_ = std::max(consumedForceEpoch_, pendingForceEpoch_);
    if (frame.scene != nullptr && frame.resources != nullptr) {
      sceneTopologyVersion_ = frame.scene->topologyVersion();
      sceneTransformVersion_ = frame.scene->transformVersion();
      sceneDeformationVersion_ =
          frame.sharedResources.rayTracingScene.has_value()
              ? frame.sharedResources.rayTracingScene->deformationVersion
              : frame.scene->deformationVersion();
      lightTopologyVersion_ = frame.scene->lightTopologyVersion();
      lightTransformVersion_ = frame.scene->lightTransformVersion();
      materialVersion_ = frame.resources->materialVersion();
      environmentVersion_ = frame.scene->environmentVersion();
    }
    if (!frameSlots_.empty()) {
      frameSlots_[static_cast<size_t>(frame.frameIndex % frameSlots_.size())]
          .traceCountersValid = true;
    }
  }
  if (inspectionScheduled_ && !frameSlots_.empty()) {
    FrameSlot &slot =
        frameSlots_[static_cast<size_t>(frame.frameIndex % frameSlots_.size())];
    slot.diagnosticValid = slot.diagnosticRequest.has_value();
  }
  submittedSequence_ = committedSequence;
  initializationScheduled_ = false;
  scrollScheduled_ = false;
  updatesScheduled_ = false;
  radiometricResponseScheduled_ = false;
  geometryResponseScheduled_ = false;
  inspectionScheduled_ = false;
  scheduledFrameIndex_ = UINT64_MAX;
  scrollInvalidations_.clear();
  pendingScrollLayouts_.clear();
}

void DDGIFeature::onFrameAbandoned(const RenderFrameContext &frame) noexcept {
  if (scheduledFrameIndex_ != frame.frameIndex) {
    return;
  }
  if (!frameSlots_.empty()) {
    FrameSlot &slot =
        frameSlots_[static_cast<size_t>(frame.frameIndex % frameSlots_.size())];
    slot.traceCountersValid = false;
    if (inspectionScheduled_) {
      slot.diagnosticValid = false;
      slot.diagnosticRequest.reset();
    }
  }
  initializationScheduled_ = false;
  scrollScheduled_ = false;
  updatesScheduled_ = false;
  radiometricResponseScheduled_ = false;
  geometryResponseScheduled_ = false;
  inspectionScheduled_ = false;
  scheduledFrameIndex_ = UINT64_MAX;
  scrollInvalidations_.clear();
  pendingScrollLayouts_.clear();
  clearPendingVolumes();
}

} // namespace nuri
