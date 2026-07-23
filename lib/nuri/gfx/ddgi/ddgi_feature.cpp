#include "nuri/gfx/ddgi/ddgi_feature.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"
#include <atomic>
#include <chrono>
#include <cstring>

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

[[nodiscard]] DDGIVolumeDesc toDesc(const DDGIEffectiveVolume &volume) {
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

[[nodiscard]] DDGIVolumeId
legacyVolumeId(const DDGIEffectiveVolume &volume) noexcept {
  if (isValid(volume.authoredId)) {
    return volume.authoredId;
  }
  const uint64_t hash = ddgiEffectiveVolumeKeyHash(volume.key);
  const uint32_t index = static_cast<uint32_t>(hash) & kResourceHandleIndexMask;
  uint32_t generation =
      static_cast<uint32_t>(hash >> kResourceHandleIndexBits) &
      kResourceHandleGenerationMask;
  generation = std::max(generation, 1u);
  return makeDDGIVolumeId(index, generation);
}

[[nodiscard]] bool sameMatrix(const glm::mat4 &left,
                              const glm::mat4 &right) noexcept {
  for (uint32_t column = 0u; column < 4u; ++column) {
    if (glm::any(glm::notEqual(left[column], right[column]))) {
      return false;
    }
  }
  return true;
}

template <typename Resource>
[[nodiscard]] bool resourceCompatible(const Resource &resource,
                                      const DDGIEffectiveVolume &volume) {
  return resource.effective.key == volume.key &&
         glm::all(
             glm::equal(resource.effective.probeCounts, volume.probeCounts)) &&
         glm::all(glm::equal(resource.effective.probeSpacing,
                             volume.probeSpacing)) &&
         sameMatrix(resource.effective.worldFromLocal, volume.worldFromLocal) &&
         resource.effective.irradianceAtlas == volume.irradianceAtlas &&
         resource.effective.distanceAtlas == volume.distanceAtlas &&
         resource.effective.maxRayDistance == volume.maxRayDistance &&
         resource.effective.blendDistance == volume.blendDistance &&
         resource.effective.mode == volume.mode;
}

[[nodiscard]] DDGISceneCoverageBounds
selectedSceneBounds(const RenderScene &scene,
                    const DDGICoverageSettings &settings,
                    const RayTracingSceneFrameView *rayTracing) noexcept {
  switch (settings.sceneBoundsSource) {
  case DDGISceneBoundsSource::ActivationSnapshot:
    return scene.ddgiActivationCoverageBounds();
  case DDGISceneBoundsSource::StaticRayTracingGeometry:
    if (!settings.autoRefitOnTopologyChange || rayTracing == nullptr) {
      return scene.ddgiStaticCoverageBounds();
    }
    {
      const DDGISceneCoverageBounds &base =
          scene.ddgiPendingStaticCoverageBounds();
      const DDGISceneCoverageBounds &current = rayTracing->staticCoverageBounds;
      DDGISceneCoverageBounds merged = base;
      merged.complete = current.complete;
      if (current.valid) {
        if (!merged.valid) {
          merged.bounds = current.bounds;
          merged.valid = true;
        } else {
          merged.bounds.combinePoint(current.bounds.min_);
          merged.bounds.combinePoint(current.bounds.max_);
        }
      }
      const bool changed =
          merged.valid != base.valid || merged.complete != base.complete ||
          (merged.valid &&
           (glm::any(glm::notEqual(merged.bounds.min_, base.bounds.min_)) ||
            glm::any(glm::notEqual(merged.bounds.max_, base.bounds.max_))));
      merged.generation =
          changed && base.generation != std::numeric_limits<uint64_t>::max()
              ? base.generation + 1u
              : base.generation;
      return merged;
    }
  case DDGISceneBoundsSource::Authored:
    return settings.authoredBounds;
  }
  return {};
}

[[nodiscard]] bool coverageUsesSceneBounds(DDGICoverageMode mode) noexcept {
  return mode == DDGICoverageMode::SceneFit || mode == DDGICoverageMode::Hybrid;
}

[[nodiscard]] uint32_t tierWeight(DDGIEffectiveTier tier) noexcept {
  switch (tier) {
  case DDGIEffectiveTier::AuthoredOverride:
  case DDGIEffectiveTier::Clipmap0:
    return 8u;
  case DDGIEffectiveTier::Clipmap1:
    return 4u;
  case DDGIEffectiveTier::Clipmap2:
    return 2u;
  case DDGIEffectiveTier::Clipmap3:
  case DDGIEffectiveTier::SceneFitCoarse:
    return 1u;
  }
  return 1u;
}

[[nodiscard]] bool historyReadyState(uint32_t state) noexcept {
  return state != static_cast<uint32_t>(DDGIProbeState::Uninitialized) &&
         state != static_cast<uint32_t>(DDGIProbeState::NewlyAwake) &&
         state != static_cast<uint32_t>(DDGIProbeState::NewlyVigilant);
}

[[nodiscard]] DDGICaptureMetadata
makeCaptureMetadata(const DDGIEffectiveVolume &effective,
                    const DDGIVolumeLayout &layout, uint64_t resourceGeneration,
                    uint64_t coverageGeneration, uint64_t sceneBoundsGeneration,
                    glm::vec3 requestedHalfExtents,
                    glm::vec3 achievedHalfExtents) noexcept {
  return DDGICaptureMetadata{
      .effectiveKeyHash = ddgiEffectiveVolumeKeyHash(effective.key),
      .coverageGeneration = coverageGeneration,
      .layoutGeneration = layout.generation,
      .resourceGeneration = resourceGeneration,
      .sceneBoundsGeneration = sceneBoundsGeneration,
      .effectiveKind = static_cast<uint32_t>(effective.key.kind),
      .cascadeIndex = effective.cascadeIndex,
      .ringOrigin = layout.ringOrigin,
      .cameraCell = layout.cameraCell,
      .requestedHalfExtents = requestedHalfExtents,
      .achievedHalfExtents = achievedHalfExtents,
      .fadeStartHalfExtents = effective.fadeStartHalfExtents,
      .fadeEndHalfExtents = effective.fadeEndHalfExtents,
      .transitionCells = effective.transitionCells,
      .valid = 1u,
  };
}

[[nodiscard]] uint32_t saturatingAge(uint64_t submittedSequence,
                                     uint64_t lastUpdate) noexcept {
  const uint64_t age =
      lastUpdate == 0u
          ? submittedSequence + 1u
          : submittedSequence - std::min(submittedSequence, lastUpdate);
  return static_cast<uint32_t>(
      std::min<uint64_t>(age, std::numeric_limits<uint32_t>::max()));
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

[[nodiscard]] uint64_t saturatingAdd(uint64_t left, uint64_t right) noexcept {
  return left > std::numeric_limits<uint64_t>::max() - right
             ? std::numeric_limits<uint64_t>::max()
             : left + right;
}

[[nodiscard]] uint64_t stableLightKey(LightId id) noexcept {
  return (static_cast<uint64_t>(id.type) << 32u) | id.value;
}

template <typename T>
[[nodiscard]] bool samePackedValue(const T &left, const T &right) noexcept {
  return std::memcmp(&left, &right, sizeof(T)) == 0;
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
      submittedLocalLights_(memory_), submittedDirectionalLights_(memory_),
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
  dirtyRegions_.clear();
  submittedLocalLights_.clear();
  submittedDirectionalLights_.clear();
  failedVolumeCount_ = 0u;
  volumeFailureReason_ = DDGIVolumeFailureReason::None;
  sceneId_ = 0u;
  volumeTopologyVersion_ = UINT64_MAX;
  volumeTransformVersion_ = UINT64_MAX;
  volumeSettingsVersion_ = UINT64_MAX;
  coverageSettings_ = {};
  coverageGeneration_ = 0u;
  sceneBoundsGeneration_ = 0u;
  sceneTopologyVersion_ = UINT64_MAX;
  sceneTransformVersion_ = UINT64_MAX;
  sceneDeformationVersion_ = UINT64_MAX;
  lightTopologyVersion_ = UINT64_MAX;
  lightTransformVersion_ = UINT64_MAX;
  materialVersion_ = UINT64_MAX;
  environmentVersion_ = UINT64_MAX;
  probeStateMirrorSourceFrame_ = 0u;
  probeStateMirrorAvailable_ = false;
}

void DDGIFeature::clearPendingVolumes() noexcept {
  pendingVolumes_.clear();
  replacementPending_ = false;
  compatiblePlanPending_ = false;
  pendingSceneId_ = 0u;
  pendingVolumeTopologyVersion_ = UINT64_MAX;
  pendingVolumeTransformVersion_ = UINT64_MAX;
  pendingVolumeSettingsVersion_ = UINT64_MAX;
  pendingCoverageSettings_ = {};
  pendingCoverageGeneration_ = 0u;
  pendingSceneBoundsGeneration_ = 0u;
  pendingFailedVolumeCount_ = 0u;
  pendingVolumeFailureReason_ = DDGIVolumeFailureReason::None;
  pendingEffectiveVolumeCount_ = 0u;
}

void DDGIFeature::clearFrameSlots() noexcept { frameSlots_.clear(); }

Result<bool, std::string>
DDGIFeature::rebuildVolumes(FrameBuildContext &ctx,
                            const DDGIEffectiveVolumePlan &plan,
                            const DDGICoverageSettings &coverageSettings) {
  clearPendingVolumes();
  const auto recordFailure = [this](DDGIVolumeFailureReason reason) {
    ++pendingFailedVolumeCount_;
    if (pendingVolumeFailureReason_ == DDGIVolumeFailureReason::None) {
      pendingVolumeFailureReason_ = reason;
    }
  };
  pendingFailedVolumeCount_ = static_cast<uint32_t>(
      std::min<size_t>(plan.failedKeys.size() + plan.omittedKeys.size(),
                       std::numeric_limits<uint32_t>::max()));
  const std::span<const DDGIEffectiveVolume> source = plan.activeVolumes();
  const uint32_t maximumTextureDimension =
      std::max(gpu_.getDeviceCaps().maxTextureDimension2D, 1u);
  uint64_t aggregateBytes = 0u;
  uint64_t activeBytes = 0u;
  for (const VolumeResource &active : volumes_) {
    activeBytes = saturatingAdd(activeBytes, active.persistentBytes);
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
  for (uint32_t slot = 0u; slot < static_cast<uint32_t>(source.size());
       ++slot) {
    const DDGIEffectiveVolume &volume = source[slot];
    const DDGIVolumeDesc desc = toDesc(volume);
    const uint32_t probeCount = volume.probeCount;
    const DDGIAtlasLayout &irradiancePacking = volume.irradianceAtlas;
    const DDGIAtlasLayout &distancePacking = volume.distanceAtlas;
    pendingVolumes_.emplace_back(memory_);
    VolumeResource &resource = pendingVolumes_.back();
    resource.id = legacyVolumeId(volume);
    resource.effective = volume;
    resource.desc = desc;
    if (volume.key.kind == DDGIEffectiveVolumeKind::SceneFit) {
      resource.requestedCoverageHalfExtents =
          0.5f * (plan.sceneFit.requestedBounds.max_ -
                  plan.sceneFit.requestedBounds.min_);
      resource.achievedCoverageHalfExtents =
          0.5f * (plan.sceneFit.achievedInteriorBounds.max_ -
                  plan.sceneFit.achievedInteriorBounds.min_);
    } else if (volume.key.kind == DDGIEffectiveVolumeKind::ClipmapCascade) {
      resource.requestedCoverageHalfExtents =
          plan.clipmaps.requestedCoverageHalfExtents;
      resource.achievedCoverageHalfExtents =
          plan.clipmaps.achievedCoverageHalfExtents;
    } else {
      resource.requestedCoverageHalfExtents = volume.probeCenterHalfExtents;
      resource.achievedCoverageHalfExtents = volume.probeCenterHalfExtents;
    }
    if (irradiancePacking.textureExtent.x > maximumTextureDimension ||
        irradiancePacking.textureExtent.y > maximumTextureDimension ||
        distancePacking.textureExtent.x > maximumTextureDimension ||
        distancePacking.textureExtent.y > maximumTextureDimension) {
      recordFailure(DDGIVolumeFailureReason::AtlasPacking);
      continue;
    }
    const DDGIMemoryEstimate &memoryEstimate = volume.memory;
    if (memoryEstimate.persistentBytes >
        config_.persistentMemoryLimitBytes -
            std::min(aggregateBytes, config_.persistentMemoryLimitBytes)) {
      recordFailure(DDGIVolumeFailureReason::PersistentMemoryLimit);
      continue;
    }
    const uint64_t replacementBytes =
        saturatingAdd(activeBytes, aggregateBytes);
    const uint64_t replacementWithVolume =
        saturatingAdd(replacementBytes, memoryEstimate.persistentBytes);
    const uint64_t replacementPeak =
        saturatingAdd(replacementWithVolume, frameBatchBytes);
    if (replacementPeak > config_.peakMemoryLimitBytes) {
      recordFailure(DDGIVolumeFailureReason::PeakMemoryLimit);
      continue;
    }
    const uint64_t resourceGeneration = ++nextResourceGeneration_;
    auto layout = makeDDGIVolumeLayout(
        legacyVolumeId(volume), desc, volume.worldFromLocal, irradiancePacking,
        distancePacking, resourceGeneration, volume.cameraCell);
    if (layout.hasError()) {
      recordFailure(DDGIVolumeFailureReason::InvalidLayout);
      continue;
    }
    resource.layout = layout.value();
    auto irradiance = gpu_.createTexture(
        TextureDesc{.type = TextureType::Texture2D,
                    .format = Format::RGBA16_FLOAT,
                    .dimensions = {irradiancePacking.textureExtent.x,
                                   irradiancePacking.textureExtent.y, 1u},
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
                    .dimensions = {distancePacking.textureExtent.x,
                                   distancePacking.textureExtent.y, 1u},
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
    resource.irradiance.reset(gpu_, irradiance.value());
    resource.distance.reset(gpu_, distance.value());
    resource.probeState.reset(gpu_, state.value());
    resource.lastSubmittedUpdates.resize(probeCount, 0u);
    resource.submittedProbeStates.assign(probeCount, {});
    for (DDGIProbeStateGpuData &probeState : resource.submittedProbeStates) {
      probeState.stateAgeFlags.x = static_cast<uint32_t>(initialState);
    }
    resource.pendingDirtyFlags.resize(probeCount, 0u);
    resource.irradianceResponseFrames.resize(probeCount, 0u);
    resource.distanceResponseFrames.resize(probeCount, 0u);
    resource.persistentBytes = memoryEstimate.persistentBytes;
    resource.resourceGeneration = resourceGeneration;
    resource.allocated = true;
    aggregateBytes += memoryEstimate.persistentBytes;
  }
  pendingSceneId_ = ctx.frame.scene->id();
  pendingVolumeTopologyVersion_ = ctx.frame.scene->ddgiVolumeTopologyVersion();
  pendingVolumeTransformVersion_ =
      ctx.frame.scene->ddgiVolumeTransformVersion();
  pendingVolumeSettingsVersion_ = ctx.frame.scene->ddgiVolumeSettingsVersion();
  pendingCoverageSettings_ = coverageSettings;
  pendingCoverageGeneration_ = plan.coverageGeneration;
  pendingSceneBoundsGeneration_ = plan.sceneBoundsGeneration;
  pendingRelocationEnabled_ = settings.relocation;
  pendingClassificationEnabled_ = settings.classification;
  replacementPending_ = true;
  return Result<bool, std::string>::makeResult(true);
}

void DDGIFeature::stageCompatiblePlan(
    const DDGIEffectiveVolumePlan &plan,
    const DDGICoverageSettings &coverageSettings,
    const RenderScene &scene) noexcept {
  pendingEffectiveVolumeCount_ = plan.volumeCount;
  for (uint32_t index = 0u; index < plan.volumeCount; ++index) {
    const DDGIEffectiveVolume &volume = plan.volumes[index];
    pendingEffectiveVolumes_[index] = volume;
    if (volume.key.kind == DDGIEffectiveVolumeKind::SceneFit) {
      pendingRequestedCoverageHalfExtents_[index] =
          0.5f * (plan.sceneFit.requestedBounds.max_ -
                  plan.sceneFit.requestedBounds.min_);
      pendingAchievedCoverageHalfExtents_[index] =
          0.5f * (plan.sceneFit.achievedInteriorBounds.max_ -
                  plan.sceneFit.achievedInteriorBounds.min_);
    } else if (volume.key.kind == DDGIEffectiveVolumeKind::ClipmapCascade) {
      pendingRequestedCoverageHalfExtents_[index] =
          plan.clipmaps.requestedCoverageHalfExtents;
      pendingAchievedCoverageHalfExtents_[index] =
          plan.clipmaps.achievedCoverageHalfExtents;
    } else {
      pendingRequestedCoverageHalfExtents_[index] =
          volume.probeCenterHalfExtents;
      pendingAchievedCoverageHalfExtents_[index] =
          volume.probeCenterHalfExtents;
    }
  }
  pendingSceneId_ = scene.id();
  pendingVolumeTopologyVersion_ = scene.ddgiVolumeTopologyVersion();
  pendingVolumeTransformVersion_ = scene.ddgiVolumeTransformVersion();
  pendingVolumeSettingsVersion_ = scene.ddgiVolumeSettingsVersion();
  pendingCoverageSettings_ = coverageSettings;
  pendingCoverageGeneration_ = plan.coverageGeneration;
  pendingSceneBoundsGeneration_ = plan.sceneBoundsGeneration;
  compatiblePlanPending_ = true;
}

uint32_t DDGIFeature::dirtyRegionFlagsForProbe(uint32_t slot,
                                               uint32_t probe) const noexcept {
  if (slot >= volumes_.size()) {
    return 0u;
  }
  const VolumeResource &volume = volumes_[slot];
  const DDGIVolumeLayout &layout =
      pendingScrollLayouts_.size() == volumes_.size()
          ? pendingScrollLayouts_[slot]
          : volume.layout;
  if (probe >= volume.lastSubmittedUpdates.size()) {
    return 0u;
  }
  const glm::uvec3 physical = ddgiProbeCoordinate(probe, layout.probeCounts);
  const glm::uvec3 logical =
      (physical + layout.probeCounts - layout.ringOrigin) % layout.probeCounts;
  uint32_t flags = 0u;
  for (const DDGIDirtyRegion &region : dirtyRegions_.pendingRegions()) {
    if ((region.affectedVolumeMask & (1u << slot)) == 0u) {
      continue;
    }
    const DDGIDirtyProbeRange &range = region.probeRanges[slot];
    if (!range.valid || glm::any(glm::lessThan(logical, range.minimum)) ||
        glm::any(glm::greaterThan(logical, range.maximum))) {
      continue;
    }
    const uint8_t response = static_cast<uint8_t>(region.response);
    if ((response & static_cast<uint8_t>(
                        DDGIDirtyResponseFlags::RelocateClassify)) != 0u) {
      flags |= kDDGIProbeUpdateReclassify;
    }
    if ((response & static_cast<uint8_t>(DDGIDirtyResponseFlags::Wake)) != 0u) {
      flags |= kDDGIProbeUpdateWake;
    }
    if ((response & static_cast<uint8_t>(DDGIDirtyResponseFlags::Irradiance)) !=
        0u) {
      flags |= kDDGIProbeUpdateIrradianceResponse;
    }
    if ((response & static_cast<uint8_t>(DDGIDirtyResponseFlags::Distance)) !=
        0u) {
      flags |= kDDGIProbeUpdateDistanceResponse;
    }
  }
  return flags;
}

uint32_t DDGIFeature::dirtyFlagsForProbe(uint32_t slot,
                                         uint32_t probe) const noexcept {
  if (slot >= volumes_.size() ||
      probe >= volumes_[slot].lastSubmittedUpdates.size()) {
    return 0u;
  }
  const VolumeResource &volume = volumes_[slot];
  uint32_t flags = dirtyRegionFlagsForProbe(slot, probe);
  if (probe < volume.pendingDirtyFlags.size()) {
    flags |= volume.pendingDirtyFlags[probe];
  }
  if (probe < volume.irradianceResponseFrames.size() &&
      volume.irradianceResponseFrames[probe] != 0u) {
    flags |= kDDGIProbeUpdateIrradianceResponse;
  }
  if (probe < volume.distanceResponseFrames.size() &&
      volume.distanceResponseFrames[probe] != 0u) {
    flags |= kDDGIProbeUpdateDistanceResponse;
  }
  return flags;
}

void DDGIFeature::commitDirtyResponses() noexcept {
  constexpr uint8_t kIrradianceResponseUpdates = 10u;
  constexpr uint8_t kDistanceResponseUpdates = 7u;
  for (uint32_t slot = 0u; slot < static_cast<uint32_t>(volumes_.size());
       ++slot) {
    VolumeResource &volume = volumes_[slot];
    for (uint32_t probe = 0u;
         probe < static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
         ++probe) {
      const uint32_t flags = dirtyRegionFlagsForProbe(slot, probe);
      volume.pendingDirtyFlags[probe] |=
          flags & (kDDGIProbeUpdateReclassify | kDDGIProbeUpdateWake);
      if ((flags & kDDGIProbeUpdateIrradianceResponse) != 0u) {
        volume.irradianceResponseFrames[probe] = std::max(
            volume.irradianceResponseFrames[probe], kIrradianceResponseUpdates);
      }
      if ((flags & kDDGIProbeUpdateDistanceResponse) != 0u) {
        volume.distanceResponseFrames[probe] = std::max(
            volume.distanceResponseFrames[probe], kDistanceResponseUpdates);
      }
    }
  }
  for (const DDGIProbeUpdateEntry &entry : scheduledEntries_) {
    if (entry.volumeStableId >= volumes_.size()) {
      continue;
    }
    VolumeResource &volume = volumes_[entry.volumeStableId];
    if (entry.probeId >= volume.lastSubmittedUpdates.size()) {
      continue;
    }
    volume.pendingDirtyFlags[entry.probeId] &=
        ~(entry.flags & (kDDGIProbeUpdateReclassify | kDDGIProbeUpdateWake));
    if ((entry.flags & kDDGIProbeUpdateIrradianceResponse) != 0u &&
        volume.irradianceResponseFrames[entry.probeId] != 0u) {
      --volume.irradianceResponseFrames[entry.probeId];
    }
    if ((entry.flags & kDDGIProbeUpdateDistanceResponse) != 0u &&
        volume.distanceResponseFrames[entry.probeId] != 0u) {
      --volume.distanceResponseFrames[entry.probeId];
    }
  }
  for (VolumeResource &volume : volumes_) {
    volume.irradianceResponseRemaining =
        volume.irradianceResponseFrames.empty()
            ? 0u
            : *std::ranges::max_element(volume.irradianceResponseFrames);
    volume.distanceResponseRemaining =
        volume.distanceResponseFrames.empty()
            ? 0u
            : *std::ranges::max_element(volume.distanceResponseFrames);
  }
}

void DDGIFeature::commitRadiometricSnapshot(const RenderScene &scene) noexcept {
  submittedLocalLights_.clear();
  const std::span<const LightId> ids = scene.packedLocalLightIds();
  const std::span<const LocalLightGpuData> lights = scene.packedLocalLights();
  submittedLocalLights_.reserve(std::min(ids.size(), lights.size()));
  for (size_t index = 0u; index < std::min(ids.size(), lights.size());
       ++index) {
    submittedLocalLights_.push_back({.id = ids[index], .data = lights[index]});
  }
  submittedDirectionalLights_.assign(scene.packedDirectionalLights().begin(),
                                     scene.packedDirectionalLights().end());
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
    if (!volume.ready || volume.desc.mode != DDGIVolumeMode::CameraTracked) {
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
        [this](OwnedBufferHandle &buffer, size_t size, Storage storage,
               std::string_view name) -> Result<bool, std::string> {
      if (buffer.valid()) {
        return Result<bool, std::string>::makeResult(true);
      }
      auto created =
          gpu_.createBuffer(BufferDesc{.usage = storage == Storage::Readback
                                                    ? BufferUsage::Copy
                                                    : BufferUsage::Storage,
                                       .storage = storage,
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
      slot.updatesReadback.reset();
      slot.updateCapacity = settings.maxProbeUpdatesPerFrame;
      slot.probeStateResultsValid = false;
      slot.probeStateResultCount = 0u;
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
      slot.traceCountersValid = false;
    }
    for (auto result :
         {ensure(slot.frameData, sizeof(DDGIFrameGpuData), Storage::Device,
                 "ddgi_frame_data"),
          ensure(slot.updates,
                 slot.updateCapacity * sizeof(DDGIProbeUpdateEntry),
                 Storage::Device, "ddgi_probe_updates"),
          ensure(slot.updatesReadback,
                 slot.updateCapacity * sizeof(DDGIProbeUpdateEntry),
                 Storage::Readback, "ddgi_probe_updates_readback"),
          ensure(slot.invalidations,
                 slot.invalidationCapacity * sizeof(DDGIProbeUpdateEntry),
                 Storage::Device, "ddgi_scroll_invalidations"),
          ensure(slot.rayResults,
                 slot.rayCapacity * sizeof(DDGIRayResultGpuData),
                 Storage::Device, "ddgi_ray_results"),
          ensure(slot.localLights,
                 sizeof(DDGITraceCountersGpuData) +
                     slot.localLightCapacity * sizeof(LocalLightGpuData),
                 Storage::Device, "ddgi_local_lights"),
          ensure(slot.traceCountersReadback, sizeof(DDGITraceCountersGpuData),
                 Storage::Readback, "ddgi_trace_counters_readback")}) {
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
      kDDGIFrameGpuDataVersion, gpu_.getSamplerBindlessIndex(sampler_.get()));
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
        resource.ready ? gpu_.getBufferDeviceAddress(resource.probeState.get())
                       : 0u;
    gpuVolume.resourceFlags = resource.ready ? kDDGIResourceReady : 0u;
    gpuVolume.probeSpacingAndBias =
        glm::vec4(layout.probeSpacing,
                  renderSettingsOrDefault(ctx.frame).ddgi.selfShadowBias);
    gpuVolume.centerHalfExtentsAndMaxDistance =
        glm::vec4(layout.probeCenterHalfExtents, resource.desc.maxRayDistance);
    gpuVolume.probeCountsAndCount =
        glm::uvec4(layout.probeCounts, ddgiProbeCount(layout.probeCounts));
    gpuVolume.irradianceAtlas = glm::uvec4(
        resource.ready ? gpu_.getTextureBindlessIndex(resource.irradiance.get())
                       : kInvalidTextureBindlessIndex,
        layout.irradianceAtlas.tileExtent.x, layout.irradianceAtlas.columns,
        layout.irradianceAtlas.rows);
    gpuVolume.distanceAtlas = glm::uvec4(
        resource.ready ? gpu_.getTextureBindlessIndex(resource.distance.get())
                       : kInvalidTextureBindlessIndex,
        layout.distanceAtlas.tileExtent.x, layout.distanceAtlas.columns,
        layout.distanceAtlas.rows);
    gpuVolume.ringOriginAndFlags =
        glm::uvec4(layout.ringOrigin,
                   std::bit_cast<uint32_t>(resource.desc.blendDistance));
    gpuVolume.generations = glm::uvec4(
        std::bit_cast<uint32_t>(layout.cameraCell.x),
        std::bit_cast<uint32_t>(layout.cameraCell.y), low32(submittedSequence_),
        std::bit_cast<uint32_t>(layout.cameraCell.z));
    const uint64_t effectiveHash =
        ddgiEffectiveVolumeKeyHash(resource.effective.key);
    gpuVolume.effectiveIdentity =
        glm::uvec4(static_cast<uint32_t>(resource.effective.key.kind),
                   low32(effectiveHash), high32(effectiveHash),
                   resource.effective.cascadeIndex);
    gpuVolume.tierTransitionCoverageFlags =
        glm::uvec4(static_cast<uint32_t>(resource.effective.tier),
                   resource.effective.transitionCells,
                   resource.effective.requestedCoverageAchieved ? 1u : 0u,
                   resource.ready ? 1u : 0u);
    gpuVolume.continuousCameraLocal =
        glm::vec4(resource.effective.continuousCameraLocal, 0.0f);
    gpuVolume.fadeStartHalfExtents =
        glm::vec4(resource.effective.fadeStartHalfExtents, 0.0f);
    gpuVolume.fadeEndHalfExtents =
        glm::vec4(resource.effective.fadeEndHalfExtents, 0.0f);
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
    if (!volume.allocated) {
      continue;
    }
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
    if (!volume.ready) {
      continue;
    }
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
        volumes_[index].ready
            ? gpu_.getBufferDeviceAddress(volumes_[index].probeState.get())
            : 0u;
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
  const bool sceneFitBootstrap =
      coverageSettings_.mode == DDGICoverageMode::Hybrid &&
      std::ranges::any_of(volumes_, [](const VolumeResource &volume) {
        return volume.effective.key.kind == DDGIEffectiveVolumeKind::SceneFit &&
               std::ranges::any_of(volume.submittedProbeStates,
                                   [](const DDGIProbeStateGpuData &state) {
                                     return !historyReadyState(
                                         state.stateAgeFlags.x);
                                   });
      });
  size_t candidateCount = 0u;
  for (const VolumeResource &volume : volumes_) {
    candidateCount += volume.lastSubmittedUpdates.size();
  }
  std::pmr::vector<DDGITieredProbeScheduleCandidate> candidates(
      scoped.resource());
  std::pmr::vector<DDGITieredProbeScheduleCandidate> workspace(
      scoped.resource());
  std::pmr::vector<DDGITierScheduleInput> tiers(scoped.resource());
  std::pmr::vector<DDGIProbeUpdateEntry> output(scoped.resource());
  candidates.reserve(candidateCount);
  tiers.reserve(volumes_.size());
  for (uint32_t slot = 0u; slot < static_cast<uint32_t>(volumes_.size());
       ++slot) {
    const VolumeResource &volume = volumes_[slot];
    const uint64_t stableKey = ddgiEffectiveVolumeKeyHash(volume.effective.key);
    const bool bootstrapTier =
        sceneFitBootstrap &&
        volume.effective.key.kind == DDGIEffectiveVolumeKind::SceneFit;
    tiers.push_back(DDGITierScheduleInput{
        .stableKey = stableKey,
        .submittedDeficit = volume.schedulerDeficit,
        .submittedStarvationFrames = volume.schedulerStarvationFrames,
        .effectiveOrder = bootstrapTier ? 0u : slot + 1u,
        .weight = bootstrapTier ? std::max(settings.maxProbeUpdatesPerFrame,
                                           tierWeight(volume.effective.tier))
                                : tierWeight(volume.effective.tier),
        .ready = volume.ready,
    });
    for (uint32_t probe = 0u;
         probe < static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
         ++probe) {
      const bool scrolled = std::ranges::binary_search(
          scrollInvalidations_, std::pair(slot, probe), {},
          [](const auto &entry) {
            return std::pair(entry.volumeStableId, entry.probeId);
          });
      const uint32_t dirtyFlags = dirtyFlagsForProbe(slot, probe);
      candidates.push_back(DDGITieredProbeScheduleCandidate{
          .tierStableKey = stableKey,
          .volumeStableId = slot,
          .probeId = probe,
          .state =
              scrolled
                  ? DDGIProbeState::Uninitialized
                  : static_cast<DDGIProbeState>(
                        volume.submittedProbeStates[probe].stateAgeFlags.x),
          .lastSubmittedUpdate = volume.lastSubmittedUpdates[probe],
          .invalidated = volume.lastSubmittedUpdates[probe] == 0u || scrolled ||
                         dirtyFlags != 0u,
      });
    }
  }
  workspace.resize(candidates.size());
  output.resize(settings.maxProbeUpdatesPerFrame);
  auto schedule = scheduleDDGITieredProbeUpdates(
      candidates, tiers,
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
  pendingTierSchedule_ = schedule.value();
  for (uint32_t index = 0u; index < schedule.value().schedule.updatedProbes;
       ++index) {
    output[index].flags =
        dirtyFlagsForProbe(output[index].volumeStableId, output[index].probeId);
  }
  std::span<DDGIProbeUpdateEntry> selected(
      output.data(), schedule.value().schedule.updatedProbes);
  std::ranges::sort(selected, {}, [](const DDGIProbeUpdateEntry &entry) {
    return std::pair(entry.volumeStableId, entry.rayBase);
  });
  for (uint32_t index = 0u; index < selected.size(); ++index) {
    selected[index].rayBase = index * settings.raysPerProbe;
  }
  scheduledEntries_.assign(
      output.begin(), output.begin() + schedule.value().schedule.updatedProbes);
  return Result<DDGIScheduleResult, std::string>::makeResult(
      schedule.value().schedule);
}

Result<bool, std::string>
DDGIFeature::appendUpdatePasses(FrameBuildContext &ctx, FrameSlot &slot,
                                const DDGIScheduleResult &schedule) {
  DDGITraceCountersGpuData initialCounters{};
  initialCounters.secondaryQueriesReserved = schedule.secondaryQueriesReserved;
  initialCounters.sourceFrame = static_cast<uint32_t>(ctx.frame.frameIndex);
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
  uint32_t volumeUpdateOffset = 0u;
  for (uint32_t volumeSlot = 0u;
       volumeSlot < static_cast<uint32_t>(volumes_.size()); ++volumeSlot) {
    const VolumeResource &volume = volumes_[volumeSlot];
    const uint32_t firstUpdate = volumeUpdateOffset;
    while (volumeUpdateOffset < scheduledEntries_.size() &&
           scheduledEntries_[volumeUpdateOffset].volumeStableId == volumeSlot) {
      ++volumeUpdateOffset;
    }
    const uint32_t volumeUpdateCount = volumeUpdateOffset - firstUpdate;
    if (!volume.ready || volumeUpdateCount == 0u) {
      continue;
    }
    const std::array outputs{volume.irradiance.get(), volume.distance.get()};
    const std::array pipelineIndices{BlendIrradiance, BlendDistance};
    const RenderSettings::DDGISettings &volumeSettings =
        renderSettingsOrDefault(ctx.frame).ddgi;
    const std::array hysteresis{volumeSettings.irradianceHysteresis,
                                volumeSettings.distanceHysteresis};
    const std::array responseScales{
        volumeSettings.changeIrradianceHysteresisScale,
        volumeSettings.changeDistanceHysteresisScale};
    for (size_t type = 0u; type < outputs.size(); ++type) {
      blendPushConstants_.push_back(BlendPushConstants{
          .frame = gpu_.getBufferDeviceAddress(slot.frameData.get()),
          .updates = gpu_.getBufferDeviceAddress(slot.updates.get()),
          .results = gpu_.getBufferDeviceAddress(slot.rayResults.get()),
          .updateCount = volumeUpdateCount,
          .updateOffset = firstUpdate,
          .volumeSlot = volumeSlot,
          .outputTextureId = gpu_.getTextureBindlessIndex(outputs[type]),
          .raysPerProbe = renderSettingsOrDefault(ctx.frame).ddgi.raysPerProbe,
          .hysteresis = hysteresis[type],
          .historyValid = submittedSequence_ != 0u ? 1u : 0u,
          .frameSeed = static_cast<uint32_t>(submittedSequence_),
          .responseHysteresisScale = responseScales[type],
      });
      dispatches_.push_back(ComputeDispatchItem{
          .pipeline = pipelines_[pipelineIndices[type]].get(),
          .dispatch = {.x = volumeUpdateCount, .y = 1u, .z = 1u},
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
  statePushConstants_.surfaceBounds = rt.surfaceBoundsAddress;
  statePushConstants_.surfaceBoundsCountsFlags =
      std::min(rt.staticSurfaceBoundsCount, 0x1fffu) |
      (std::min(rt.dynamicSurfaceBoundsCount, 0x1fffu) << 13u) |
      (rt.staticSurfaceBoundsAvailable ? (1u << 30u) : 0u) |
      (rt.dynamicSurfaceBoundsAvailable ? (1u << 31u) : 0u);
  dependencyBuffers_.clear();
  dependencyBufferModes_.clear();
  appendUnique(dependencyBuffers_, dependencyBufferModes_, slot.updates.get(),
               RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  appendUnique(dependencyBuffers_, dependencyBufferModes_,
               slot.rayResults.get(), RenderGraphAccessMode::Read);
  appendUnique(dependencyBuffers_, dependencyBufferModes_, slot.frameData.get(),
               RenderGraphAccessMode::Read);
  appendUnique(dependencyBuffers_, dependencyBufferModes_, rt.surfaceBounds,
               RenderGraphAccessMode::Read);
  for (size_t index = 0u; index < volumes_.size(); ++index) {
    statePushConstants_.states[index] =
        volumes_[index].ready
            ? gpu_.getBufferDeviceAddress(volumes_[index].probeState.get())
            : 0u;
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
  auto updatesSource = ctx.graph.importBuffer(slot.updates.get(),
                                              "ddgi_updates_readback_source");
  auto updatesDestination = ctx.graph.importBuffer(
      slot.updatesReadback.get(), "ddgi_updates_readback_destination");
  auto countersSource = ctx.graph.importBuffer(
      slot.localLights.get(), "ddgi_trace_counters_readback_source");
  auto countersDestination =
      ctx.graph.importBuffer(slot.traceCountersReadback.get(),
                             "ddgi_trace_counters_readback_destination");
  if (updatesSource.hasError() || updatesDestination.hasError() ||
      countersSource.hasError() || countersDestination.hasError()) {
    return Result<bool, std::string>::makeError(
        "DDGI failed to import non-blocking readback buffers");
  }
  const std::array readbackCopies{
      RenderGraphBufferCopyItem{
          .sourceBuffer = updatesSource.value(),
          .destinationBuffer = updatesDestination.value(),
          .size = schedule.updatedProbes * sizeof(DDGIProbeUpdateEntry),
      },
      RenderGraphBufferCopyItem{
          .sourceBuffer = countersSource.value(),
          .destinationBuffer = countersDestination.value(),
          .size = sizeof(DDGITraceCountersGpuData),
      }};
  auto readbackPass = ctx.graph.addBufferCopyPass(RenderGraphBufferCopyPassDesc{
      .copies = readbackCopies,
      .debugLabel = "DDGI Readback Copies",
      .debugColor = 0xff55bb77u,
  });
  if (readbackPass.hasError()) {
    return Result<bool, std::string>::makeError(readbackPass.error());
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
    auto readback =
        gpu_.createBuffer(BufferDesc{.usage = BufferUsage::Copy,
                                     .storage = Storage::Readback,
                                     .size = kDDGIDiagnosticBufferBytes},
                          "ddgi_probe_diagnostic_readback");
    if (readback.hasError()) {
      slot.diagnostic.reset();
      return Result<bool, std::string>::makeError(readback.error());
    }
    slot.diagnosticReadback.reset(gpu_, readback.value());
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
  auto diagnosticSource = ctx.graph.importBuffer(
      slot.diagnostic.get(), "ddgi_diagnostic_readback_source");
  auto diagnosticDestination = ctx.graph.importBuffer(
      slot.diagnosticReadback.get(), "ddgi_diagnostic_readback_destination");
  if (diagnosticSource.hasError() || diagnosticDestination.hasError()) {
    return Result<bool, std::string>::makeError(
        "DDGI failed to import diagnostic readback buffers");
  }
  const std::array diagnosticCopies{RenderGraphBufferCopyItem{
      .sourceBuffer = diagnosticSource.value(),
      .destinationBuffer = diagnosticDestination.value(),
      .size = kDDGIDiagnosticBufferBytes,
  }};
  auto diagnosticReadbackPass =
      ctx.graph.addBufferCopyPass(RenderGraphBufferCopyPassDesc{
          .copies = diagnosticCopies,
          .debugLabel = "DDGI Diagnostic Readback Copy",
          .debugColor = 0xffff55aau,
      });
  if (diagnosticReadbackPass.hasError()) {
    return Result<bool, std::string>::makeError(diagnosticReadbackPass.error());
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
      "ddgi_volume2_irradiance_atlas", "ddgi_volume3_irradiance_atlas",
      "ddgi_volume4_irradiance_atlas", "ddgi_volume5_irradiance_atlas",
      "ddgi_volume6_irradiance_atlas", "ddgi_volume7_irradiance_atlas"};
  static constexpr std::array distanceNames{
      "ddgi_volume0_distance_atlas", "ddgi_volume1_distance_atlas",
      "ddgi_volume2_distance_atlas", "ddgi_volume3_distance_atlas",
      "ddgi_volume4_distance_atlas", "ddgi_volume5_distance_atlas",
      "ddgi_volume6_distance_atlas", "ddgi_volume7_distance_atlas"};
  for (size_t slot = 0u; slot < volumes_.size(); ++slot) {
    if (!volumes_[slot].ready) {
      continue;
    }
    const DDGICaptureMetadata metadata = makeCaptureMetadata(
        volumes_[slot].effective, volumes_[slot].layout,
        volumes_[slot].resourceGeneration, coverageGeneration_,
        sceneBoundsGeneration_, volumes_[slot].requestedCoverageHalfExtents,
        volumes_[slot].achievedCoverageHalfExtents);
    for (const auto point :
         {RenderCapturePoint{
              .name = irradianceNames[slot],
              .version = kDDGICaptureSemanticsVersion,
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
              .debugLabel = "DDGI Irradiance Atlas",
              .ddgiMetadata = metadata},
          RenderCapturePoint{
              .name = distanceNames[slot],
              .version = kDDGICaptureSemanticsVersion,
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
              .debugLabel = "DDGI Distance Atlas",
              .ddgiMetadata = metadata}}) {
      frame.captureRegistry.publish(point);
    }
  }
}

void DDGIFeature::collectDebugProbeStateMetrics(FrameBuildContext &ctx) {
  uint32_t vigilant = 0u;
  uint32_t uninitialized = 0u;
  uint32_t off = 0u;
  uint32_t sleeping = 0u;
  uint32_t newlyAwake = 0u;
  uint32_t awake = 0u;
  uint32_t newlyVigilant = 0u;
  uint32_t relocated = 0u;
  float maxRelocation = 0.0f;
  for (size_t volumeIndex = 0u; volumeIndex < volumes_.size(); ++volumeIndex) {
    const VolumeResource &volume = volumes_[volumeIndex];
    DDGIVolumeFrameMetrics &volumeMetrics =
        ctx.frame.metrics.ddgi.volumes[volumeIndex];
    volumeMetrics.initializedProbes = 0u;
    volumeMetrics.shadingEnabledProbes = 0u;
    volumeMetrics.invalidProbes = 0u;
    uint32_t historyReadyProbes = 0u;
    if (!volume.ready) {
      continue;
    }
    for (const DDGIProbeStateGpuData &state : volume.submittedProbeStates) {
      const auto probeState =
          static_cast<DDGIProbeState>(state.stateAgeFlags.x);
      uninitialized += probeState == DDGIProbeState::Uninitialized ? 1u : 0u;
      off += probeState == DDGIProbeState::Off ? 1u : 0u;
      sleeping += probeState == DDGIProbeState::Sleeping ? 1u : 0u;
      newlyAwake += probeState == DDGIProbeState::NewlyAwake ? 1u : 0u;
      awake += probeState == DDGIProbeState::Awake ? 1u : 0u;
      newlyVigilant += probeState == DDGIProbeState::NewlyVigilant ? 1u : 0u;
      vigilant += probeState == DDGIProbeState::Vigilant ? 1u : 0u;
      volumeMetrics.initializedProbes +=
          probeState != DDGIProbeState::Uninitialized ? 1u : 0u;
      volumeMetrics.shadingEnabledProbes +=
          probeState == DDGIProbeState::Awake ||
                  probeState == DDGIProbeState::Vigilant
              ? 1u
              : 0u;
      volumeMetrics.invalidProbes +=
          probeState == DDGIProbeState::Uninitialized ||
                  probeState == DDGIProbeState::Off
              ? 1u
              : 0u;
      historyReadyProbes += historyReadyState(state.stateAgeFlags.x) ? 1u : 0u;
      const float relocation = glm::length(glm::vec3(state.relocation));
      relocated += relocation > 1.0e-5f ? 1u : 0u;
      maxRelocation = std::max(maxRelocation, relocation);
    }
    const float readyRatio =
        volumeMetrics.totalProbes == 0u
            ? 0.0f
            : static_cast<float>(historyReadyProbes) /
                  static_cast<float>(volumeMetrics.totalProbes);
    volumeMetrics.historyReadyPercentage = readyRatio * 100.0f;
    const float coverageReadyRatio =
        volumeMetrics.totalProbes == 0u
            ? 0.0f
            : static_cast<float>(volumeMetrics.shadingEnabledProbes) /
                  static_cast<float>(volumeMetrics.totalProbes);
    volumeMetrics.coverageReadyPercentage =
        volume.ready ? coverageReadyRatio * 100.0f : 0.0f;
    volumeMetrics.confidence = volume.ready ? coverageReadyRatio : 0.0f;
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
  metrics.probeStateReadbackAvailable = probeStateMirrorAvailable_ ? 1u : 0u;
  metrics.probeStateReadbackSourceFrame =
      static_cast<uint32_t>(probeStateMirrorSourceFrame_);
  metrics.probeStateReadbackStaleFrames = static_cast<uint32_t>(
      std::min(ctx.frame.frameIndex >= probeStateMirrorSourceFrame_
                   ? ctx.frame.frameIndex - probeStateMirrorSourceFrame_
                   : 0u,
               static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
}

void DDGIFeature::collectCompletedProbeStates(FrameSlot &slot) {
  if (!slot.probeStateResultsValid || slot.probeStateResultCount == 0u) {
    slot.probeStateResultsValid = false;
    return;
  }
  ScopedScratch scoped(scratch_);
  std::pmr::vector<DDGIProbeUpdateEntry> completed(scoped.resource());
  completed.resize(slot.probeStateResultCount);
  auto read = gpu_.readBuffer(
      slot.updatesReadback.get(), 0u,
      std::as_writable_bytes(std::span(completed.data(), completed.size())));
  if (read.hasError()) {
    slot.probeStateResultsValid = false;
    slot.probeStateResultCount = 0u;
    return;
  }
  for (const DDGIProbeUpdateEntry &entry : completed) {
    if (entry.volumeStableId >= volumes_.size() ||
        slot.probeStateResourceGenerations[entry.volumeStableId] !=
            volumes_[entry.volumeStableId].resourceGeneration ||
        entry.probeId >=
            volumes_[entry.volumeStableId].submittedProbeStates.size() ||
        entry.resultState > static_cast<uint32_t>(DDGIProbeState::Vigilant)) {
      continue;
    }
    DDGIProbeStateGpuData &state =
        volumes_[entry.volumeStableId].submittedProbeStates[entry.probeId];
    state.relocation = entry.resultRelocation;
    state.stateAgeFlags =
        glm::uvec4(entry.resultState, entry.resultSubmittedSequence,
                   entry.resultIteration, 0u);
  }
  probeStateMirrorSourceFrame_ =
      std::max(probeStateMirrorSourceFrame_, slot.probeStateSourceFrame);
  probeStateMirrorAvailable_ = true;
  slot.probeStateResultsValid = false;
  slot.probeStateResultCount = 0u;
}

void DDGIFeature::collectCompletedTraceMetrics(FrameSlot &slot,
                                               DDGIFrameMetrics &metrics) {
  if (slot.traceCountersValid) {
    DDGITraceCountersGpuData counters{};
    auto read = gpu_.readBuffer(
        slot.traceCountersReadback.get(), 0u,
        std::as_writable_bytes(std::span(&counters, static_cast<size_t>(1u))));
    if (!read.hasError()) {
      metrics.primaryQueriesIssued = counters.primaryQueriesIssued;
      metrics.traceCounterSourceFrame = counters.sourceFrame;
      metrics.secondaryQueriesReserved = counters.secondaryQueriesReserved;
      metrics.secondaryQueries = counters.secondaryQueries;
      metrics.localSecondaryQueries = counters.localSecondaryQueries;
      metrics.directionalSecondaryQueries =
          counters.secondaryQueries -
          std::min(counters.secondaryQueries, counters.localSecondaryQueries);
      metrics.secondaryQueriesUnused =
          counters.secondaryQueriesReserved -
          std::min(counters.secondaryQueriesReserved,
                   counters.secondaryQueries);
      metrics.primaryCandidateIntersections =
          counters.primaryCandidateIntersections;
      metrics.secondaryCandidateIntersections =
          counters.secondaryCandidateIntersections;
      metrics.alphaCandidateRejections = counters.alphaCandidateRejections;
      metrics.backfaceCandidateRejections =
          counters.backfaceCandidateRejections;
      metrics.candidateOverflows = counters.candidateOverflows;
      metrics.localLightTruncations = counters.localLightTruncations;
      for (size_t volumeIndex = 0u; volumeIndex < volumes_.size();
           ++volumeIndex) {
        if (slot.probeStateResourceGenerations[volumeIndex] !=
            volumes_[volumeIndex].resourceGeneration) {
          continue;
        }
        metrics.volumes[volumeIndex].primaryQueriesIssued =
            counters.primaryQueriesIssuedByVolume[volumeIndex];
        metrics.volumes[volumeIndex].secondaryQueries =
            counters.secondaryQueriesByVolume[volumeIndex];
      }
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
      slot.diagnosticReadback.get(), 0u,
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
  if (rtReady && readyCount != 0u) {
    std::array<DDGIEffectiveVolumeKey, kMaxDDGIVolumes> volumeKeys{};
    std::array<DDGIVolumeId, kMaxDDGIVolumes> volumeIds{};
    std::array<uint32_t, kMaxDDGIVolumes> probeCounts{};
    std::array<float, kMaxDDGIVolumes> minimumProbeSpacing{};
    std::array<DDGICaptureMetadata, kMaxDDGIVolumes> captureMetadata{};
    for (uint32_t slotIndex = 0u;
         slotIndex < static_cast<uint32_t>(volumes_.size()); ++slotIndex) {
      const VolumeResource &volume = volumes_[slotIndex];
      volumeKeys[slotIndex] = volume.effective.key;
      volumeIds[slotIndex] = volume.id;
      probeCounts[slotIndex] =
          static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
      minimumProbeSpacing[slotIndex] =
          std::min({volume.layout.probeSpacing.x, volume.layout.probeSpacing.y,
                    volume.layout.probeSpacing.z});
      captureMetadata[slotIndex] = makeCaptureMetadata(
          volume.effective, volume.layout, volume.resourceGeneration,
          coverageGeneration_, sceneBoundsGeneration_,
          volume.requestedCoverageHalfExtents,
          volume.achievedCoverageHalfExtents);
      captureMetadata[slotIndex].valid = volume.ready ? 1u : 0u;
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
        .activeVolumeCount = static_cast<uint32_t>(volumes_.size()),
        .flags = kDDGIFrameEnabled,
        .debugView = renderSettingsOrDefault(ctx.frame).ddgi.debugView,
        .volumeKeys = volumeKeys,
        .volumeIds = volumeIds,
        .probeCounts = probeCounts,
        .minimumProbeSpacing = minimumProbeSpacing,
        .captureMetadata = captureMetadata,
        .coverageGeneration = coverageGeneration_,
        .sceneBoundsGeneration = sceneBoundsGeneration_,
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
  uint64_t committedAtlasBytes = 0u;
  for (size_t volumeIndex = 0u; volumeIndex < volumes_.size(); ++volumeIndex) {
    const VolumeResource &volume = volumes_[volumeIndex];
    DDGIVolumeFrameMetrics &volumeMetrics = metrics.volumes[volumeIndex];
    volumeMetrics.active = 1u;
    volumeMetrics.effectiveKeyHash =
        ddgiEffectiveVolumeKeyHash(volume.effective.key);
    volumeMetrics.layoutGeneration = volume.layout.generation;
    volumeMetrics.resourceGeneration = volume.resourceGeneration;
    volumeMetrics.effectiveKind =
        static_cast<uint32_t>(volume.effective.key.kind);
    volumeMetrics.tier = static_cast<uint32_t>(volume.effective.tier);
    volumeMetrics.cascadeIndex = volume.effective.cascadeIndex;
    volumeMetrics.totalProbes =
        static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
    volumeMetrics.interiorHalfExtents = volume.effective.fadeEndHalfExtents;
    volumeMetrics.fadeStartHalfExtents = volume.effective.fadeStartHalfExtents;
    volumeMetrics.fadeEndHalfExtents = volume.effective.fadeEndHalfExtents;
    volumeMetrics.cameraCell = volume.layout.cameraCell;
    volumeMetrics.deficit = volume.schedulerDeficit;
    volumeMetrics.starvationFrames = volume.schedulerStarvationFrames;
    for (const uint64_t lastUpdate : volume.lastSubmittedUpdates) {
      volumeMetrics.initializedProbes += lastUpdate != 0u ? 1u : 0u;
    }
    volumeMetrics.invalidProbes =
        volumeMetrics.totalProbes - volumeMetrics.initializedProbes;
    volumeMetrics.shadingEnabledProbes = volumeMetrics.initializedProbes;
    for (const DDGIProbeUpdateEntry &entry : scrollInvalidations_) {
      volumeMetrics.newlyExposedProbes +=
          entry.volumeStableId == volumeIndex ? 1u : 0u;
    }
    for (const DDGIProbeUpdateEntry &entry : scheduledEntries_) {
      if (entry.volumeStableId != volumeIndex) {
        continue;
      }
      ++volumeMetrics.updates;
      volumeMetrics.primaryQueries += entry.rayCount;
      metrics.dirtyProbesAffected += entry.flags != 0u ? 1u : 0u;
    }
    for (uint32_t tierIndex = 0u; tierIndex < pendingTierSchedule_.tierCount;
         ++tierIndex) {
      const DDGITierScheduleResult &tier =
          pendingTierSchedule_.tiers[tierIndex];
      if (tier.stableKey != volumeMetrics.effectiveKeyHash) {
        continue;
      }
      volumeMetrics.scheduledQuota = tier.scheduledQuota;
      volumeMetrics.usedQuota = tier.usedQuota;
      volumeMetrics.deficit = tier.pendingDeficit;
      volumeMetrics.starvationFrames = tier.pendingStarvationFrames;
      break;
    }
    if (!volume.lastSubmittedUpdates.empty()) {
      ScopedScratch scoped(scratch_);
      std::pmr::vector<uint32_t> ages(scoped.resource());
      ages.reserve(volume.lastSubmittedUpdates.size());
      for (const uint64_t lastUpdate : volume.lastSubmittedUpdates) {
        ages.push_back(saturatingAge(submittedSequence_, lastUpdate));
      }
      std::ranges::sort(ages);
      volumeMetrics.updateAgeMedian = ages[(ages.size() - 1u) / 2u];
      const size_t p95Index =
          std::min((ages.size() * 95u + 99u) / 100u, ages.size()) - 1u;
      volumeMetrics.updateAgeP95 = ages[p95Index];
      volumeMetrics.updateAgeMaximum = ages.back();
    }
    volumeMetrics.estimatedFullRefreshFrames =
        volumeMetrics.usedQuota == 0u
            ? 0u
            : divRoundUp(volumeMetrics.totalProbes, volumeMetrics.usedQuota);
    const float initializedRatio =
        volumeMetrics.totalProbes == 0u
            ? 0.0f
            : static_cast<float>(volumeMetrics.initializedProbes) /
                  static_cast<float>(volumeMetrics.totalProbes);
    volumeMetrics.historyReadyPercentage = initializedRatio * 100.0f;
    volumeMetrics.coverageReadyPercentage =
        volume.ready ? initializedRatio * 100.0f : 0.0f;
    volumeMetrics.confidence = volume.ready ? initializedRatio : 0.0f;
    metrics.layoutGeneration =
        std::max(metrics.layoutGeneration, volume.layout.generation);
    metrics.resourceGeneration =
        std::max(metrics.resourceGeneration, volume.resourceGeneration);
    metrics.totalProbes +=
        static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
    metrics.persistentBytes += volume.persistentBytes;
    if (volume.allocated) {
      committedAtlasBytes += volume.effective.memory.irradianceBytes +
                             volume.effective.memory.distanceBytes;
    }
    metrics.irradianceResponseRemaining =
        std::max(metrics.irradianceResponseRemaining,
                 volume.irradianceResponseRemaining);
    metrics.distanceResponseRemaining = std::max(
        metrics.distanceResponseRemaining, volume.distanceResponseRemaining);
  }
  metrics.committedAtlasBytes = committedAtlasBytes;
  metrics.pendingAtlasBytes = 0u;
  for (const VolumeResource &volume : pendingVolumes_) {
    if (volume.allocated) {
      metrics.pendingAtlasBytes += volume.effective.memory.irradianceBytes +
                                   volume.effective.memory.distanceBytes;
    }
  }
  metrics.peakAtlasBytes =
      metrics.committedAtlasBytes + metrics.pendingAtlasBytes;
  collectDebugProbeStateMetrics(ctx);
  bool allHistoriesReady = !volumes_.empty();
  bool sceneFitHistoryReady = false;
  bool hasSceneFit = false;
  for (size_t volumeIndex = 0u; volumeIndex < volumes_.size(); ++volumeIndex) {
    const DDGIVolumeFrameMetrics &volumeMetrics = metrics.volumes[volumeIndex];
    const bool volumeHistoryReady =
        volumes_[volumeIndex].ready && volumeMetrics.totalProbes != 0u &&
        std::ranges::all_of(volumes_[volumeIndex].submittedProbeStates,
                            [](const DDGIProbeStateGpuData &state) {
                              return historyReadyState(state.stateAgeFlags.x);
                            });
    allHistoriesReady = allHistoriesReady && volumeHistoryReady;
    if (volumes_[volumeIndex].effective.key.kind ==
        DDGIEffectiveVolumeKind::SceneFit) {
      hasSceneFit = true;
      sceneFitHistoryReady = sceneFitHistoryReady || volumeHistoryReady;
    }
  }
  const bool hybridCoverage =
      coverageSettings_.mode == DDGICoverageMode::Hybrid;
  const bool requiredHistoryReady =
      hybridCoverage ? hasSceneFit && sceneFitHistoryReady : allHistoriesReady;
  metrics.historyReady = rtReady && requiredHistoryReady ? 1u : 0u;
  metrics.skyFallbackActive = metrics.historyReady != 0u ? 0u : 1u;
  const DDGIDirtyRegionMetrics &dirtyMetrics = dirtyRegions_.metrics();
  metrics.dirtyRegionsProduced = dirtyMetrics.produced;
  metrics.dirtyRegionsMerged = dirtyMetrics.merged;
  metrics.dirtyRegionsOverflowed = dirtyMetrics.overflowed;
  metrics.dirtyRegionsPending = dirtyRegions_.unconsumedCount();
  if (ctx.shared.rayTracingScene.has_value()) {
    metrics.classificationFallbacks =
        (ctx.shared.rayTracingScene->staticSurfaceBoundsAvailable ? 0u : 1u) +
        (ctx.shared.rayTracingScene->dynamicSurfaceBoundsAvailable ? 0u : 1u);
  }
  const bool diagnosticMetricsRequested =
      ctx.frame.captureRequests.contains("ddgi_coverage_debug_preview") ||
      ctx.frame.captureRequests.contains("ddgi_classification_debug_preview") ||
      ctx.frame.captureRequests.contains("ddgi_dirty_region_debug_preview");
  if (diagnosticMetricsRequested) {
    constexpr glm::uvec3 kLatticeDimensions{5u, 3u, 5u};
    const glm::vec3 latticeHalfExtents =
        glm::max(metrics.requestedCoverageHalfExtents, glm::vec3(1.0f));
    const glm::vec3 latticeCenter = glm::vec3(ctx.frame.camera.cameraPos);
    for (uint32_t z = 0u; z < kLatticeDimensions.z; ++z) {
      for (uint32_t y = 0u; y < kLatticeDimensions.y; ++y) {
        for (uint32_t x = 0u; x < kLatticeDimensions.x; ++x) {
          const glm::vec3 unit(
              2.0f * static_cast<float>(x) /
                      static_cast<float>(kLatticeDimensions.x - 1u) -
                  1.0f,
              2.0f * static_cast<float>(y) /
                      static_cast<float>(kLatticeDimensions.y - 1u) -
                  1.0f,
              2.0f * static_cast<float>(z) /
                      static_cast<float>(kLatticeDimensions.z - 1u) -
                  1.0f);
          const glm::vec3 worldPoint =
              latticeCenter + unit * latticeHalfExtents;
          float skyRemainder = 1.0f;
          bool covered = false;
          for (size_t volumeIndex = 0u; volumeIndex < volumes_.size();
               ++volumeIndex) {
            const VolumeResource &volume = volumes_[volumeIndex];
            const glm::vec3 localPoint = glm::vec3(
                volume.layout.localFromWorld * glm::vec4(worldPoint, 1.0f));
            const glm::vec3 halfExtents =
                glm::max(volume.effective.fadeEndHalfExtents,
                         volume.effective.probeCenterHalfExtents -
                             volume.effective.probeSpacing);
            if (glm::any(glm::greaterThan(glm::abs(localPoint), halfExtents))) {
              continue;
            }
            covered = true;
            skyRemainder *=
                1.0f -
                glm::clamp(metrics.volumes[volumeIndex].confidence, 0.0f, 1.0f);
          }
          ++metrics.diagnosticSampleCount;
          metrics.uncoveredDiagnosticSamples += covered ? 0u : 1u;
          metrics.skyRemainderSamples += skyRemainder > 0.01f ? 1u : 0u;
        }
      }
    }
    metrics.diagnosticSamplesAvailable = 1u;
  }
  const bool automaticCoverage =
      metrics.coverageMode != static_cast<uint32_t>(DDGICoverageMode::Manual);
  if (metrics.sceneCoverageRatio >= 1.0f && allHistoriesReady) {
    metrics.coverageStatus = DDGICoverageStatus::FullCoverageReady;
  } else if (metrics.sceneCoverageRatio >= 1.0f &&
             ((hybridCoverage && requiredHistoryReady) ||
              (!hybridCoverage && automaticCoverage))) {
    metrics.coverageStatus = DDGICoverageStatus::FullCoverageWarmingDetail;
  } else if (metrics.activeVolumes != 0u) {
    metrics.coverageStatus = DDGICoverageStatus::PartialCoverage;
  } else {
    metrics.coverageStatus = DDGICoverageStatus::SkyFallbackOnly;
  }
  metrics.frameBatchBytes =
      slot.updateCapacity * sizeof(DDGIProbeUpdateEntry) +
      slot.invalidationCapacity * sizeof(DDGIProbeUpdateEntry) +
      slot.rayCapacity * sizeof(DDGIRayResultGpuData) +
      sizeof(DDGITraceCountersGpuData) +
      slot.localLightCapacity * sizeof(LocalLightGpuData) +
      sizeof(DDGIFrameGpuData) +
      (slot.diagnostic.valid() ? kDDGIDiagnosticBufferBytes : 0u);
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
  pendingTierSchedule_ = {};
  dirtyConsumptionScheduled_ = false;
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
  ctx.frame.metrics.ddgi.coverageMode =
      static_cast<uint32_t>(settings.coverage.mode);
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
  DDGICoverageSettings coverageSettings = settings.coverage;
  sanitizeDDGICoverageSettings(coverageSettings);
  const DDGISceneCoverageBounds sceneBounds =
      selectedSceneBounds(scene, coverageSettings,
                          ctx.shared.rayTracingScene.has_value()
                              ? std::addressof(*ctx.shared.rayTracingScene)
                              : nullptr);
  const bool boundsGenerationChanged =
      coverageUsesSceneBounds(coverageSettings.mode) &&
      sceneBoundsGeneration_ != sceneBounds.generation;
  const bool coverageProfileChanged =
      coverageGeneration_ == 0u || sceneId_ != scene.id() ||
      coverageSettings_ != coverageSettings || boundsGenerationChanged;
  const bool pendingInputsMatch =
      replacementPending_ && pendingSceneId_ == scene.id() &&
      pendingCoverageSettings_ == coverageSettings &&
      (!coverageUsesSceneBounds(coverageSettings.mode) ||
       pendingSceneBoundsGeneration_ == sceneBounds.generation) &&
      pendingVolumeTopologyVersion_ == scene.ddgiVolumeTopologyVersion() &&
      pendingVolumeTransformVersion_ == scene.ddgiVolumeTransformVersion() &&
      pendingVolumeSettingsVersion_ == scene.ddgiVolumeSettingsVersion() &&
      pendingRelocationEnabled_ == settings.relocation &&
      pendingClassificationEnabled_ == settings.classification;
  const uint64_t desiredCoverageGeneration =
      pendingInputsMatch ? pendingCoverageGeneration_
                         : (coverageProfileChanged
                                ? coverageGeneration_ + 1u
                                : std::max(coverageGeneration_, uint64_t{1u}));

  uint64_t retainedBytes = 0u;
  for (const VolumeResource &volume : volumes_) {
    retainedBytes = saturatingAdd(retainedBytes, volume.persistentBytes);
  }
  const uint64_t frameBatchBytes =
      static_cast<uint64_t>(settings.maxProbeUpdatesPerFrame) *
          sizeof(DDGIProbeUpdateEntry) +
      static_cast<uint64_t>(settings.maxRayQueriesPerFrame) *
          sizeof(DDGIRayResultGpuData) +
      sizeof(DDGIFrameGpuData);
  DDGIEffectiveVolumePlan plan(memory_);
  const DDGICoverageResolveInput resolveInput{
      .sceneId = scene.id(),
      .coverageGeneration = desiredCoverageGeneration,
      .sceneBounds = sceneBounds,
      .authoredVolumes = scene.ddgiVolumes(),
      .cameraWorldPosition = glm::vec3(ctx.frame.camera.cameraPos),
      .settings = coverageSettings,
      .limits = {.maxTextureExtent = glm::uvec2(
                     std::max(gpu_.getDeviceCaps().maxTextureDimension2D, 1u)),
                 .maxPersistentBytes = config_.persistentMemoryLimitBytes,
                 .maxReplacementPeakBytes = config_.peakMemoryLimitBytes,
                 .retainedReplacementBytes =
                     saturatingAdd(retainedBytes, frameBatchBytes),
                 .maxProbeUpdatesPerFrame = settings.maxProbeUpdatesPerFrame},
      .scratch = memory_,
  };
  const auto coverageResolveStart = std::chrono::steady_clock::now();
  auto resolved = resolveDDGIEffectiveVolumePlan(resolveInput, plan);
  DDGIFrameMetrics &coverageMetrics = ctx.frame.metrics.ddgi;
  coverageMetrics.coverageResolveCpuTimeMs =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - coverageResolveStart)
          .count();
  coverageMetrics.coverageMode = static_cast<uint32_t>(coverageSettings.mode);
  coverageMetrics.coverageError = plan.error.limit;
  coverageMetrics.limitingConstraint = plan.error.limit;
  coverageMetrics.effectiveVolumes = plan.volumeCount;
  coverageMetrics.failedVolumes = static_cast<uint32_t>(
      std::min<size_t>(plan.failedKeys.size() + plan.omittedKeys.size(),
                       std::numeric_limits<uint32_t>::max()));
  for (const DDGIEffectiveVolume &volume : plan.activeVolumes()) {
    coverageMetrics.authoredVolumes +=
        volume.key.kind == DDGIEffectiveVolumeKind::Authored ? 1u : 0u;
    coverageMetrics.generatedVolumes +=
        volume.key.kind != DDGIEffectiveVolumeKind::Authored ? 1u : 0u;
  }
  if (coverageSettings.mode == DDGICoverageMode::SceneFit) {
    coverageMetrics.requestedCoverageHalfExtents =
        0.5f * (plan.sceneFit.requestedBounds.max_ -
                plan.sceneFit.requestedBounds.min_);
    coverageMetrics.achievedCoverageHalfExtents =
        0.5f * (plan.sceneFit.achievedInteriorBounds.max_ -
                plan.sceneFit.achievedInteriorBounds.min_);
  } else if (coverageSettings.mode == DDGICoverageMode::CameraClipmaps ||
             coverageSettings.mode == DDGICoverageMode::Hybrid) {
    coverageMetrics.requestedCoverageHalfExtents =
        plan.clipmaps.requestedCoverageHalfExtents;
    coverageMetrics.achievedCoverageHalfExtents =
        plan.clipmaps.achievedCoverageHalfExtents;
  }
  if (coverageSettings.mode == DDGICoverageMode::SceneFit ||
      coverageSettings.mode == DDGICoverageMode::Hybrid) {
    coverageMetrics.sceneCoverageRatio =
        glm::clamp(plan.sceneFit.requestedVolumeCoverage, 0.0f, 1.0f);
    coverageMetrics.limitingConstraint = plan.sceneFit.limitingConstraint;
  } else if (coverageSettings.mode == DDGICoverageMode::CameraClipmaps) {
    const glm::vec3 requested =
        glm::max(plan.clipmaps.requestedCoverageHalfExtents,
                 glm::vec3(std::numeric_limits<float>::epsilon()));
    const glm::vec3 achieved =
        glm::min(plan.clipmaps.achievedCoverageHalfExtents, requested);
    coverageMetrics.sceneCoverageRatio =
        glm::clamp((achieved.x * achieved.y * achieved.z) /
                       (requested.x * requested.y * requested.z),
                   0.0f, 1.0f);
    coverageMetrics.limitingConstraint = plan.clipmaps.limitingConstraint;
  }
  coverageMetrics.pendingAtlasBytes = 0u;
  for (const DDGIEffectiveVolume &volume : plan.activeVolumes()) {
    coverageMetrics.pendingAtlasBytes +=
        volume.memory.irradianceBytes + volume.memory.distanceBytes;
  }
  coverageMetrics.peakAtlasBytes = coverageMetrics.pendingAtlasBytes;
  const bool coverageSolveFailed = resolved.hasError();
  if (coverageSolveFailed) {
    if (replacementPending_ || compatiblePlanPending_) {
      clearPendingVolumes();
    }
    coverageMetrics.failedVolumes = static_cast<uint32_t>(
        std::min<size_t>(plan.failedKeys.size() + plan.omittedKeys.size() + 1u,
                         std::numeric_limits<uint32_t>::max()));
    coverageMetrics.coverageError = resolved.error().limit;
    coverageMetrics.limitingConstraint = resolved.error().limit;
    coverageMetrics.coverageStatus = DDGICoverageStatus::SkyFallbackOnly;
    coverageMetrics.fallbackReason = DDGIFallbackReason::AllocationFailed;
  }

  const auto compatibleWith = [&plan](const auto &resources) {
    if (resources.size() != plan.volumeCount) {
      return false;
    }
    for (size_t index = 0u; index < resources.size(); ++index) {
      if (!resourceCompatible(resources[index], plan.volumes[index])) {
        return false;
      }
    }
    return true;
  };
  const bool activeCompatible =
      !coverageSolveFailed && compatibleWith(volumes_);
  const bool pendingCompatible = !coverageSolveFailed && pendingInputsMatch &&
                                 compatibleWith(pendingVolumes_);
  const bool runtimeModeChanged =
      !volumes_.empty() && (relocationEnabled_ != settings.relocation ||
                            classificationEnabled_ != settings.classification);
  if (!coverageSolveFailed &&
      (!activeCompatible || resetRequested || runtimeModeChanged) &&
      !pendingCompatible) {
    auto rebuild = rebuildVolumes(ctx, plan, coverageSettings);
    if (rebuild.hasError()) {
      ctx.frame.metrics.ddgi.fallbackReason =
          DDGIFallbackReason::AllocationFailed;
      return Result<bool, std::string>::makeResult(true);
    }
  } else if (!coverageSolveFailed && activeCompatible && !resetRequested &&
             !runtimeModeChanged) {
    if (replacementPending_) {
      clearPendingVolumes();
    }
    stageCompatiblePlan(plan, coverageSettings, scene);
    pendingRelocationEnabled_ = settings.relocation;
    pendingClassificationEnabled_ = settings.classification;
    scheduledFrameIndex_ = ctx.frame.frameIndex;
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
  const bool readbackComplete =
      !isValid(slot.submission) || gpu_.isSubmissionComplete(slot.submission);
  if (readbackComplete) {
    collectCompletedProbeStates(slot);
    collectCompletedTraceMetrics(slot, ctx.frame.metrics.ddgi);
    collectCompletedInspection(ctx, slot);
  }
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
        coverageSolveFailed || failedVolumeCount_ != 0u
            ? DDGIFallbackReason::AllocationFailed
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
    const uint64_t currentDeformationVersion =
        ctx.shared.rayTracingScene.has_value()
            ? ctx.shared.rayTracingScene->deformationVersion
            : scene.deformationVersion();
    const bool geometryChanged =
        sceneTopologyVersion_ != scene.topologyVersion() ||
        sceneTransformVersion_ != scene.transformVersion() ||
        sceneDeformationVersion_ != currentDeformationVersion;
    const std::span<const DirectionalLightGpuData> directionalLights =
        scene.packedDirectionalLights();
    const bool directionalChanged =
        directionalLights.size() != submittedDirectionalLights_.size() ||
        !std::ranges::equal(directionalLights, submittedDirectionalLights_,
                            samePackedValue<DirectionalLightGpuData>);
    const bool localLightsChanged =
        lightTopologyVersion_ != scene.lightTopologyVersion() ||
        lightTransformVersion_ != scene.lightTransformVersion();
    const bool globalRadiometricChanged =
        directionalChanged ||
        materialVersion_ != ctx.resources.materialVersion() ||
        environmentVersion_ != scene.environmentVersion();
    if (dirtyRegions_.unconsumedCount() == 0u &&
        (geometryChanged || localLightsChanged || globalRadiometricChanged)) {
      std::array<DDGIDirtyVolume, kMaxDDGIEffectiveVolumes> dirtyVolumes{};
      uint32_t dirtyVolumeCount = 0u;
      for (uint32_t index = 0u; index < static_cast<uint32_t>(volumes_.size());
           ++index) {
        const VolumeResource &volume = volumes_[index];
        if (!volume.ready) {
          continue;
        }
        const DDGIVolumeLayout &layout =
            pendingScrollLayouts_.size() == volumes_.size()
                ? pendingScrollLayouts_[index]
                : volume.layout;
        const float minimumSpacing =
            std::min({layout.probeSpacing.x, layout.probeSpacing.y,
                      layout.probeSpacing.z});
        dirtyVolumes[dirtyVolumeCount++] = DDGIDirtyVolume{
            .localFromWorld = layout.localFromWorld,
            .probeCounts = layout.probeCounts,
            .probeSpacing = layout.probeSpacing,
            .cameraCell = layout.cameraCell,
            .queryBias = 0.75f * minimumSpacing * settings.selfShadowBias,
            .tier = volume.effective.tier,
            .effectiveIndex = index,
        };
      }
      const std::span<const DDGIDirtyVolume> activeDirtyVolumes(
          dirtyVolumes.data(), dirtyVolumeCount);
      if (geometryChanged) {
        const std::span<const DDGISceneChangeRegion> geometryChanges =
            ctx.shared.rayTracingScene->geometryChangeRegions;
        if (!geometryChanges.empty()) {
          for (DDGISceneChangeRegion change : geometryChanges) {
            change.submissionSequence = submittedSequence_;
            (void)dirtyRegions_.publish(change, activeDirtyVolumes);
          }
        } else {
          DDGISceneChangeKind kind = DDGISceneChangeKind::StaticTransform;
          uint64_t version = scene.transformVersion();
          if (sceneTopologyVersion_ != scene.topologyVersion()) {
            kind = DDGISceneChangeKind::StaticTopology;
            version = scene.topologyVersion();
          } else if (sceneDeformationVersion_ != currentDeformationVersion) {
            kind = DDGISceneChangeKind::Deformation;
            version = currentDeformationVersion;
          }
          (void)dirtyRegions_.publish(
              DDGISceneChangeRegion{
                  .kind = kind,
                  .sourceId = scene.id(),
                  .sourceVersion = version,
                  .submissionSequence = submittedSequence_,
                  .boundsKnown = false,
              },
              activeDirtyVolumes);
        }
        geometryResponseScheduled_ = true;
      }
      if (localLightsChanged) {
        const std::span<const LightId> currentIds = scene.packedLocalLightIds();
        const std::span<const LocalLightGpuData> currentLights =
            scene.packedLocalLights();
        const size_t currentCount =
            std::min(currentIds.size(), currentLights.size());
        const uint64_t lightVersion = std::max(scene.lightTopologyVersion(),
                                               scene.lightTransformVersion());
        for (size_t index = 0u; index < currentCount; ++index) {
          const auto previous =
              std::ranges::find(submittedLocalLights_, currentIds[index],
                                &LocalLightSnapshot::id);
          if (previous != submittedLocalLights_.end() &&
              samePackedValue(previous->data, currentLights[index])) {
            continue;
          }
          DDGISceneChangeRegion change = makeDDGILocalLightChangeRegion(
              previous != submittedLocalLights_.end() ? &previous->data
                                                      : nullptr,
              &currentLights[index], stableLightKey(currentIds[index]),
              lightVersion);
          change.submissionSequence = submittedSequence_;
          (void)dirtyRegions_.publish(change, activeDirtyVolumes);
        }
        for (const LocalLightSnapshot &previous : submittedLocalLights_) {
          if (std::ranges::find(currentIds.first(currentCount), previous.id) !=
              currentIds.first(currentCount).end()) {
            continue;
          }
          DDGISceneChangeRegion change = makeDDGILocalLightChangeRegion(
              &previous.data, nullptr, stableLightKey(previous.id),
              lightVersion);
          change.submissionSequence = submittedSequence_;
          (void)dirtyRegions_.publish(change, activeDirtyVolumes);
        }
        radiometricResponseScheduled_ = true;
      }
      if (globalRadiometricChanged) {
        const uint64_t radiometricVersion = std::max(
            {scene.lightTopologyVersion(), scene.lightTransformVersion(),
             ctx.resources.materialVersion(), scene.environmentVersion()});
        (void)dirtyRegions_.publish(
            DDGISceneChangeRegion{
                .kind = DDGISceneChangeKind::GlobalRadiometric,
                .sourceId = scene.id(),
                .sourceVersion = radiometricVersion,
                .submissionSequence = submittedSequence_,
                .boundsKnown = false,
            },
            activeDirtyVolumes);
        radiometricResponseScheduled_ = true;
      }
    }
    (void)dirtyRegions_.prepareConsumption(ctx.frame.frameIndex);
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
    dirtyConsumptionScheduled_ =
        !dirtyRegions_.pendingRegions().empty() && updatesScheduled_;
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
  if (!dirtyConsumptionScheduled_ && dirtyRegions_.hasPendingConsumption()) {
    dirtyRegions_.abandonConsumption(ctx.frame.frameIndex);
  }
  ctx.frame.metrics.ddgi.fallbackReason =
      coverageSolveFailed || failedVolumeCount_ != 0u
          ? DDGIFallbackReason::AllocationFailed
          : (rtReady ? DDGIFallbackReason::None
                     : DDGIFallbackReason::RayTracingSceneWarming);
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
  if ((updatesScheduled_ || inspectionScheduled_) && !frameSlots_.empty()) {
    FrameSlot &slot =
        frameSlots_[static_cast<size_t>(frame.frameIndex % frameSlots_.size())];
    slot.submission = frame.submission;
  }
  if (initializationScheduled_) {
    for (VolumeResource &volume : pendingVolumes_) {
      volume.ready = volume.allocated;
    }
    volumes_.swap(pendingVolumes_);
    pendingVolumes_.clear();
    replacementPending_ = false;
    sceneId_ = pendingSceneId_;
    volumeTopologyVersion_ = pendingVolumeTopologyVersion_;
    volumeTransformVersion_ = pendingVolumeTransformVersion_;
    volumeSettingsVersion_ = pendingVolumeSettingsVersion_;
    coverageSettings_ = pendingCoverageSettings_;
    coverageGeneration_ = pendingCoverageGeneration_;
    sceneBoundsGeneration_ = pendingSceneBoundsGeneration_;
    relocationEnabled_ = pendingRelocationEnabled_;
    classificationEnabled_ = pendingClassificationEnabled_;
    failedVolumeCount_ = pendingFailedVolumeCount_;
    volumeFailureReason_ = pendingVolumeFailureReason_;
    probeStateMirrorSourceFrame_ = frame.frameIndex;
    probeStateMirrorAvailable_ = !volumes_.empty();
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
      commitRadiometricSnapshot(*frame.scene);
    }
  } else if (compatiblePlanPending_ &&
             pendingEffectiveVolumeCount_ == volumes_.size()) {
    for (size_t index = 0u; index < volumes_.size(); ++index) {
      volumes_[index].effective = pendingEffectiveVolumes_[index];
      volumes_[index].desc = toDesc(pendingEffectiveVolumes_[index]);
      volumes_[index].requestedCoverageHalfExtents =
          pendingRequestedCoverageHalfExtents_[index];
      volumes_[index].achievedCoverageHalfExtents =
          pendingAchievedCoverageHalfExtents_[index];
    }
    sceneId_ = pendingSceneId_;
    volumeTopologyVersion_ = pendingVolumeTopologyVersion_;
    volumeTransformVersion_ = pendingVolumeTransformVersion_;
    volumeSettingsVersion_ = pendingVolumeSettingsVersion_;
    coverageSettings_ = pendingCoverageSettings_;
    coverageGeneration_ = pendingCoverageGeneration_;
    sceneBoundsGeneration_ = pendingSceneBoundsGeneration_;
    relocationEnabled_ = pendingRelocationEnabled_;
    classificationEnabled_ = pendingClassificationEnabled_;
    compatiblePlanPending_ = false;
    pendingEffectiveVolumeCount_ = 0u;
  }
  if (scrollScheduled_) {
    for (const DDGIProbeUpdateEntry &entry : scrollInvalidations_) {
      if (entry.volumeStableId < volumes_.size() &&
          entry.probeId <
              volumes_[entry.volumeStableId].lastSubmittedUpdates.size()) {
        volumes_[entry.volumeStableId].lastSubmittedUpdates[entry.probeId] = 0u;
        volumes_[entry.volumeStableId].pendingDirtyFlags[entry.probeId] = 0u;
        volumes_[entry.volumeStableId].irradianceResponseFrames[entry.probeId] =
            0u;
        volumes_[entry.volumeStableId].distanceResponseFrames[entry.probeId] =
            0u;
        DDGIProbeStateGpuData &state =
            volumes_[entry.volumeStableId].submittedProbeStates[entry.probeId];
        state = {};
        state.stateAgeFlags.x = static_cast<uint32_t>(
            classificationEnabled_ ? DDGIProbeState::Uninitialized
                                   : DDGIProbeState::NewlyVigilant);
      }
    }
    if (pendingScrollLayouts_.size() == volumes_.size()) {
      for (size_t index = 0u; index < volumes_.size(); ++index) {
        volumes_[index].layout = pendingScrollLayouts_[index];
      }
    }
    probeStateMirrorSourceFrame_ = frame.frameIndex;
    probeStateMirrorAvailable_ = true;
  }
  if (updatesScheduled_) {
    for (const DDGIProbeUpdateEntry &entry : scheduledEntries_) {
      if (entry.volumeStableId < volumes_.size() &&
          entry.probeId <
              volumes_[entry.volumeStableId].lastSubmittedUpdates.size()) {
        volumes_[entry.volumeStableId].lastSubmittedUpdates[entry.probeId] =
            committedSequence;
      }
    }
    for (uint32_t tierIndex = 0u; tierIndex < pendingTierSchedule_.tierCount;
         ++tierIndex) {
      const DDGITierScheduleResult &tier =
          pendingTierSchedule_.tiers[tierIndex];
      for (VolumeResource &volume : volumes_) {
        if (ddgiEffectiveVolumeKeyHash(volume.effective.key) !=
            tier.stableKey) {
          continue;
        }
        volume.schedulerDeficit = tier.pendingDeficit;
        volume.schedulerStarvationFrames = tier.pendingStarvationFrames;
        break;
      }
    }
    consumedForceEpoch_ = std::max(consumedForceEpoch_, pendingForceEpoch_);
    if (!frameSlots_.empty()) {
      FrameSlot &slot = frameSlots_[static_cast<size_t>(frame.frameIndex %
                                                        frameSlots_.size())];
      slot.traceCountersValid = true;
      slot.probeStateResultCount =
          static_cast<uint32_t>(scheduledEntries_.size());
      slot.probeStateSourceFrame = frame.frameIndex;
      slot.probeStateResourceGenerations = {};
      for (size_t volumeIndex = 0u; volumeIndex < volumes_.size();
           ++volumeIndex) {
        slot.probeStateResourceGenerations[volumeIndex] =
            volumes_[volumeIndex].resourceGeneration;
      }
      slot.probeStateResultsValid = slot.probeStateResultCount != 0u;
    }
    commitDirtyResponses();
  }
  if (frame.scene != nullptr && frame.resources != nullptr) {
    if (geometryResponseScheduled_) {
      sceneTopologyVersion_ = frame.scene->topologyVersion();
      sceneTransformVersion_ = frame.scene->transformVersion();
      sceneDeformationVersion_ =
          frame.sharedResources.rayTracingScene.has_value()
              ? frame.sharedResources.rayTracingScene->deformationVersion
              : frame.scene->deformationVersion();
    }
    if (radiometricResponseScheduled_) {
      lightTopologyVersion_ = frame.scene->lightTopologyVersion();
      lightTransformVersion_ = frame.scene->lightTransformVersion();
      materialVersion_ = frame.resources->materialVersion();
      environmentVersion_ = frame.scene->environmentVersion();
      commitRadiometricSnapshot(*frame.scene);
    }
  }
  if (inspectionScheduled_ && !frameSlots_.empty()) {
    FrameSlot &slot =
        frameSlots_[static_cast<size_t>(frame.frameIndex % frameSlots_.size())];
    slot.diagnosticValid = slot.diagnosticRequest.has_value();
  }
  if (dirtyConsumptionScheduled_) {
    (void)dirtyRegions_.commitConsumption(frame.frameIndex);
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
  pendingTierSchedule_ = {};
  dirtyConsumptionScheduled_ = false;
}

void DDGIFeature::onFrameAbandoned(const RenderFrameContext &frame) noexcept {
  if (scheduledFrameIndex_ != frame.frameIndex) {
    return;
  }
  if (!frameSlots_.empty()) {
    FrameSlot &slot =
        frameSlots_[static_cast<size_t>(frame.frameIndex % frameSlots_.size())];
    slot.traceCountersValid = false;
    slot.probeStateResultsValid = false;
    slot.probeStateResultCount = 0u;
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
  pendingTierSchedule_ = {};
  if (dirtyRegions_.hasPendingConsumption()) {
    dirtyRegions_.abandonConsumption(frame.frameIndex);
  }
  dirtyConsumptionScheduled_ = false;
  clearPendingVolumes();
}

} // namespace nuri
