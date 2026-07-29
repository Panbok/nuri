#include "nuri/gfx/ddgi/ddgi_feature.h"
#include "nuri/gfx/shader.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/render_capture.h"
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

[[nodiscard]] constexpr DDGIFallbackReason
coverageFallbackReason(DDGICoverageLimit limit) noexcept {
  switch (limit) {
  case DDGICoverageLimit::SceneBoundsUnavailable:
  case DDGICoverageLimit::SceneBoundsIncomplete:
    return DDGIFallbackReason::CoverageBoundsUnavailable;
  case DDGICoverageLimit::None:
    return DDGIFallbackReason::None;
  default:
    return DDGIFallbackReason::CoverageUnsatisfied;
  }
}

[[nodiscard]] constexpr uint64_t
ddgiPresetPersistentBudget(DDGIQualityPreset preset,
                           uint64_t customBudget) noexcept {
  constexpr uint64_t mib = 1024ull * 1024ull;
  switch (preset) {
  case DDGIQualityPreset::Low:
    return std::min(customBudget, 32ull * mib);
  case DDGIQualityPreset::Balanced:
    return std::min(customBudget, 64ull * mib);
  case DDGIQualityPreset::High:
    return std::min(customBudget, 128ull * mib);
  case DDGIQualityPreset::Custom:
    return customBudget;
  }
  return customBudget;
}

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

void appendUnique(std::pmr::vector<RenderGraphImportedBufferUse> &uses,
                  BufferHandle handle, RenderGraphAccessMode mode) {
  if (!nuri::isValid(handle)) {
    return;
  }
  const auto found = std::ranges::find_if(
      uses, [handle](const RenderGraphImportedBufferUse use) {
        return use.buffer == handle;
      });
  if (found == uses.end()) {
    uses.push_back({.buffer = handle, .access = mode});
    return;
  }
  found->access = found->access | mode;
}

void appendUnique(std::pmr::vector<RenderGraphImportedTextureUse> &uses,
                  TextureHandle handle, RenderGraphAccessMode mode) {
  if (!nuri::isValid(handle)) {
    return;
  }
  const auto found = std::ranges::find_if(
      uses, [handle](const RenderGraphImportedTextureUse use) {
        return use.texture == handle;
      });
  if (found == uses.end()) {
    uses.push_back({.texture = handle, .access = mode});
    return;
  }
  found->access = found->access | mode;
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
  return DDGIVolumeId::fromParts(index, generation);
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
      scratch_(memory_), volumes_(memory_), frameSlots_(memory_),
      scheduledEntries_(memory_), dispatchEntries_(memory_),
      scrollInvalidations_(memory_), dispatches_(memory_),
      irradianceDispatches_(memory_), distanceDispatches_(memory_),
      blendPushConstants_(memory_), bufferUses_(memory_), textureUses_(memory_),
      irradianceTextureUses_(memory_), distanceTextureUses_(memory_),
      forwardDependencyBuffers_(memory_), forwardDependencyTextures_(memory_),
      selectedLocalLights_(memory_), submittedLocalLights_(memory_),
      submittedDirectionalLights_(memory_), coveragePlan_(memory_),
      pending_(memory_), deviceEpoch_(nextDDGIFeatureEpoch.fetch_add(
                             1u, std::memory_order_relaxed)) {}

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
    auto shader = compileShaderFile(gpu_, names[index], paths[index].string(),
                                    ShaderStage::Compute);
    if (shader.hasError()) {
      return Result<bool, std::string>::makeError(shader.error());
    }
    ComputePipelineDesc pipelineDesc{.computeShader = shader.value()};
    auto pipeline = gpu_.createComputePipeline(pipelineDesc, names[index]);
    if (pipeline.hasError()) {
      return Result<bool, std::string>::makeError(pipeline.error());
    }
    pipelines_[index].reset(gpu_, pipeline.value());
  }
  // The single-sample compute cache cannot preserve surface identity at MSAA
  // silhouettes. Keep the implementation dormant until it has per-sample inputs
  // and outputs (or an equivalent same-surface validation contract).
  return Result<bool, std::string>::makeResult(true);
}

bool DDGIFeature::opaqueSurfaceCacheActive(
    const FrameBuildContext &ctx) const noexcept {
  const RenderSettings::DDGISettings &settings = ctx.frame.settings.ddgi;
  return pipelines_[OpaqueSurfaceCache].valid() && settings.enabled &&
         settings.opaqueGatherVariant == DDGISurfaceGatherVariant::Product &&
         settings.debugView == DDGIDebugView::None &&
         ctx.frame.presentationAA.coverage != CoverageMode::Sample1 &&
         ctx.shared.ddgiFrameGpuData.has_value() &&
         ctx.shared.forwardSceneGpuData.has_value() &&
         nuri::isValid(ctx.shared[FrameTextureSlot::SceneDepth].texture) &&
         nuri::isValid(ctx.shared[FrameTextureSlot::Normal].texture) &&
         nuri::isValid(
             ctx.shared[FrameTextureSlot::DdgiOpaqueSurfaceCache].texture);
}

Result<bool, std::string>
DDGIFeature::buildOpaqueSurfaceCache(FrameBuildContext &ctx) {
  if (!opaqueSurfaceCacheActive(ctx)) {
    return Result<bool, std::string>::makeResult(true);
  }
  const DDGIFrameGpuDataHandle &ddgi = *ctx.shared.ddgiFrameGpuData;
  const ForwardSceneGpuData &scene = *ctx.shared.forwardSceneGpuData;
  const TextureHandle cache =
      ctx.shared[FrameTextureSlot::DdgiOpaqueSurfaceCache].texture;
  const TextureDimensions dimensions = gpu_.getTextureDimensions(cache);
  if (dimensions.width == 0u || dimensions.height == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto graphTexture =
      ctx.graph.importTexture(cache, "ddgi_opaque_surface_cache");
  if (graphTexture.hasError()) {
    return Result<bool, std::string>::makeError(graphTexture.error());
  }
  ctx.shared[FrameTextureSlot::DdgiOpaqueSurfaceCache].graph =
      graphTexture.value();
  opaqueSurfaceCachePushConstants_ = OpaqueSurfaceCachePushConstants{
      .inverseView = glm::inverse(ctx.frame.camera.view),
      .ddgiFrame = ddgi.bufferAddress,
      .sceneFrame = scene.frameDataAddress,
      .depthTextureId = gpu_.getTextureBindlessIndex(
          ctx.shared[FrameTextureSlot::SceneDepth].texture),
      .normalTextureId = gpu_.getTextureBindlessIndex(
          ctx.shared[FrameTextureSlot::Normal].texture),
      .outputTextureId = gpu_.getTextureBindlessIndex(cache),
      .width = dimensions.width,
      .height = dimensions.height,
      .projectionType = static_cast<uint32_t>(ctx.frame.camera.projectionType),
      .nearPlane = ctx.frame.camera.nearPlane,
      .farPlane = ctx.frame.camera.farPlane,
      .tanHalfFovY = std::tan(ctx.frame.camera.fovYRadians * 0.5f),
      .aspectRatio = ctx.frame.camera.aspectRatio,
      .orthoHeight = ctx.frame.camera.orthoHeight,
  };
  bufferUses_.clear();
  appendUnique(bufferUses_, scene.buffer, RenderGraphAccessMode::Read);
  for (BufferHandle dependency : ddgi.dependencyBuffers) {
    appendUnique(bufferUses_, dependency, RenderGraphAccessMode::Read);
  }
  textureUses_.clear();
  for (TextureHandle dependency : ddgi.dependencyTextures) {
    appendUnique(textureUses_, dependency, RenderGraphAccessMode::Read);
  }
  appendUnique(textureUses_, ctx.shared[FrameTextureSlot::SceneDepth].texture,
               RenderGraphAccessMode::Read);
  appendUnique(textureUses_, ctx.shared[FrameTextureSlot::Normal].texture,
               RenderGraphAccessMode::Read);
  appendUnique(textureUses_, cache, RenderGraphAccessMode::Write);
  const std::array dispatches{ComputeDispatchItem{
      .pipeline = pipelines_[OpaqueSurfaceCache].get(),
      .dispatch = {.x = divRoundUp(dimensions.width, 8u),
                   .y = divRoundUp(dimensions.height, 8u),
                   .z = 1u},
      .pushConstants = bytesOf(opaqueSurfaceCachePushConstants_),
      .debugLabel = "DDGI Opaque Surface Cache",
      .debugColor = 0xff3f9fddu,
  }};
  auto pass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
      .executionMode = RenderPassExecutionMode::ComputeOnly,
      .hasColorAttachment = false,
      .preDispatches = dispatches,
      .importedBufferUses = bufferUses_,
      .importedTextureUses = textureUses_,
      .gpuTimingScope = GpuTimingScope::DDGIOpaqueSurfaceCache,
      .debugLabel = "DDGI Opaque Surface Cache",
      .debugColor = 0xff3f9fddu,
  });
  if (pass.hasError()) {
    return Result<bool, std::string>::makeError(pass.error());
  }
  DDGIFrameMetrics &metrics = ctx.frame.metrics.ddgi;
  metrics.opaqueGatherArchitecture =
      static_cast<uint32_t>(DDGISurfaceGatherArchitecture::ComputeSurfaceCache);
  metrics.surfaceGatherWidth = dimensions.width;
  metrics.surfaceGatherHeight = dimensions.height;
  metrics.surfaceCacheFormat =
      static_cast<uint32_t>(kFrameCompositionDDGIOpaqueSurfaceCacheFormat);
  metrics.surfaceCacheBytes =
      static_cast<uint64_t>(dimensions.width) *
      static_cast<uint64_t>(dimensions.height) *
      static_cast<uint64_t>(
          formatTexelBytes(kFrameCompositionDDGIOpaqueSurfaceCacheFormat));
  publishRequestedCapture(ctx.frame, gpu_, "ddgi_opaque_surface_cache", cache,
                          RenderCaptureValueKind::LinearHdrColor,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "linear_hdr", "hdr_color",
                          "DDGI Opaque Surface Cache");
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
  clearPendingDirtySourceFacts();
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
  latestTraceCounters_ = {};
  latestTraceCounterResourceGenerations_ = {};
  latestTraceCounterSceneId_ = 0u;
  latestTraceCounterDeviceEpoch_ = 0u;
  latestTraceCounterFeatureGeneration_ = 0u;
  latestTraceCountersAvailable_ = false;
  coveragePlan_.clear();
  coveragePlanValid_ = false;
  coveragePlanSucceeded_ = false;
}

void DDGIFeature::clearPendingVolumes() noexcept {
  pending_.volumes.clear();
  pending_.retainedSourceIndices.fill(UINT32_MAX);
  pending_.replacement = false;
  pending_.compatiblePlan = false;
  pending_.sources.sceneId = 0u;
  pending_.volumeTopologyVersion = UINT64_MAX;
  pending_.volumeTransformVersion = UINT64_MAX;
  pending_.volumeSettingsVersion = UINT64_MAX;
  pending_.coverageSettings = {};
  pending_.coverageGeneration = 0u;
  pending_.sceneBoundsGeneration = 0u;
  pending_.failedVolumeCount = 0u;
  pending_.volumeFailureReason = DDGIVolumeFailureReason::None;
  pending_.effectiveVolumeCount = 0u;
}

void DDGIFeature::clearFrameSlots() noexcept {
  frameSlots_.clear();
  activeFrameSlotIndex_ = std::numeric_limits<size_t>::max();
}

Result<bool, std::string>
DDGIFeature::rebuildVolumes(FrameBuildContext &ctx,
                            const DDGIEffectiveVolumePlan &plan,
                            const DDGICoverageSettings &coverageSettings,
                            bool preserveCompatibleResources) {
  clearPendingVolumes();
  const auto recordFailure = [this](DDGIVolumeFailureReason reason) {
    ++pending_.failedVolumeCount;
    if (pending_.volumeFailureReason == DDGIVolumeFailureReason::None) {
      pending_.volumeFailureReason = reason;
    }
  };
  pending_.failedVolumeCount = static_cast<uint32_t>(
      std::min<size_t>(plan.failedKeys.size() + plan.omittedKeys.size(),
                       std::numeric_limits<uint32_t>::max()));
  const std::span<const DDGIEffectiveVolume> source = plan.activeVolumes();
  const uint32_t maximumTextureDimension =
      std::max(gpu_.getDeviceCaps().maxTextureDimension2D, 1u);
  uint64_t finalPersistentBytes = 0u;
  uint64_t newPendingBytes = 0u;
  uint64_t activeBytes = 0u;
  for (const VolumeResource &active : volumes_) {
    activeBytes = saturatingAdd(activeBytes, active.persistentBytes);
  }
  const RenderSettings::DDGISettings &settings = ctx.frame.settings.ddgi;
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
    pending_.volumes.emplace_back(memory_);
    VolumeResource &resource = pending_.volumes.back();
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
    if (preserveCompatibleResources) {
      const auto retained =
          std::ranges::find_if(volumes_, [&](const VolumeResource &candidate) {
            const size_t sourceIndex = static_cast<size_t>(
                std::addressof(candidate) - volumes_.data());
            return pending_.retainedSourceIndices[slot] == UINT32_MAX &&
                   std::ranges::find(pending_.retainedSourceIndices.begin(),
                                     pending_.retainedSourceIndices.begin() +
                                         slot,
                                     static_cast<uint32_t>(sourceIndex)) ==
                       pending_.retainedSourceIndices.begin() + slot &&
                   glm::all(glm::equal(candidate.effective.cameraCell,
                                       volume.cameraCell)) &&
                   resourceCompatible(candidate, volume);
          });
      if (retained != volumes_.end()) {
        const uint32_t sourceIndex =
            static_cast<uint32_t>(retained - volumes_.begin());
        pending_.retainedSourceIndices[slot] = sourceIndex;
        resource.persistentBytes = retained->persistentBytes;
        resource.resourceGeneration = retained->resourceGeneration;
        finalPersistentBytes =
            saturatingAdd(finalPersistentBytes, retained->persistentBytes);
        continue;
      }
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
            std::min(finalPersistentBytes,
                     config_.persistentMemoryLimitBytes)) {
      recordFailure(DDGIVolumeFailureReason::PersistentMemoryLimit);
      continue;
    }
    const uint64_t replacementBytes =
        saturatingAdd(activeBytes, newPendingBytes);
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
    // Generated coverage must remain spatially complete as its clipmaps move.
    // Classification-only traces do not populate atlas history, so using them
    // here exposes the camera-relative probe-state mask in shaded pixels.
    const bool classifyVolume =
        settings.classification &&
        volume.key.kind == DDGIEffectiveVolumeKind::Authored;
    const DDGIProbeState initialState = classifyVolume
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
    finalPersistentBytes =
        saturatingAdd(finalPersistentBytes, memoryEstimate.persistentBytes);
    newPendingBytes =
        saturatingAdd(newPendingBytes, memoryEstimate.persistentBytes);
  }
  pending_.sources.sceneId = ctx.frame.scene->id();
  pending_.volumeTopologyVersion = ctx.frame.scene->ddgiVolumeTopologyVersion();
  pending_.volumeTransformVersion =
      ctx.frame.scene->ddgiVolumeTransformVersion();
  pending_.volumeSettingsVersion = ctx.frame.scene->ddgiVolumeSettingsVersion();
  pending_.coverageSettings = coverageSettings;
  pending_.coverageGeneration = plan.coverageGeneration;
  pending_.sceneBoundsGeneration = plan.sceneBoundsGeneration;
  pending_.relocationEnabled = settings.relocation;
  pending_.classificationEnabled = settings.classification;
  pending_.replacement = true;
  return Result<bool, std::string>::makeResult(true);
}

void DDGIFeature::stageCompatiblePlan(
    const DDGIEffectiveVolumePlan &plan,
    const DDGICoverageSettings &coverageSettings,
    const RenderScene &scene) noexcept {
  pending_.effectiveVolumeCount = plan.volumeCount;
  for (uint32_t index = 0u; index < plan.volumeCount; ++index) {
    const DDGIEffectiveVolume &volume = plan.volumes[index];
    pending_.effectiveVolumes[index] = volume;
    if (volume.key.kind == DDGIEffectiveVolumeKind::SceneFit) {
      pending_.requestedCoverageHalfExtents[index] =
          0.5f * (plan.sceneFit.requestedBounds.max_ -
                  plan.sceneFit.requestedBounds.min_);
      pending_.achievedCoverageHalfExtents[index] =
          0.5f * (plan.sceneFit.achievedInteriorBounds.max_ -
                  plan.sceneFit.achievedInteriorBounds.min_);
    } else if (volume.key.kind == DDGIEffectiveVolumeKind::ClipmapCascade) {
      pending_.requestedCoverageHalfExtents[index] =
          plan.clipmaps.requestedCoverageHalfExtents;
      pending_.achievedCoverageHalfExtents[index] =
          plan.clipmaps.achievedCoverageHalfExtents;
    } else {
      pending_.requestedCoverageHalfExtents[index] =
          volume.probeCenterHalfExtents;
      pending_.achievedCoverageHalfExtents[index] =
          volume.probeCenterHalfExtents;
    }
  }
  pending_.sources.sceneId = scene.id();
  pending_.volumeTopologyVersion = scene.ddgiVolumeTopologyVersion();
  pending_.volumeTransformVersion = scene.ddgiVolumeTransformVersion();
  pending_.volumeSettingsVersion = scene.ddgiVolumeSettingsVersion();
  pending_.coverageSettings = coverageSettings;
  pending_.coverageGeneration = plan.coverageGeneration;
  pending_.sceneBoundsGeneration = plan.sceneBoundsGeneration;
  pending_.compatiblePlan = true;
}

const DDGIEffectiveVolume &
DDGIFeature::frameEffectiveVolume(size_t index) const noexcept {
  if (!pending_.replacement && pending_.compatiblePlan &&
      index < pending_.effectiveVolumeCount && index < volumes_.size() &&
      resourceCompatible(volumes_[index], pending_.effectiveVolumes[index])) {
    return pending_.effectiveVolumes[index];
  }
  return volumes_[index].effective;
}

const DDGICoverageSettings &
DDGIFeature::frameCoverageSettings() const noexcept {
  if (!pending_.replacement && pending_.compatiblePlan) {
    return pending_.coverageSettings;
  }
  return coverageSettings_;
}

uint32_t DDGIFeature::dirtyRegionFlagsForProbe(uint32_t slot,
                                               uint32_t probe) const noexcept {
  const std::span<const DDGIDirtyRegion> pendingRegions =
      dirtyRegions_.pendingRegions();
  if (pendingRegions.empty() || slot >= volumes_.size()) {
    return 0u;
  }
  const VolumeResource &volume = volumes_[slot];
  const DDGIVolumeLayout &layout =
      pending_.scrollLayouts.size() == volumes_.size()
          ? pending_.scrollLayouts[slot]
          : volume.layout;
  if (probe >= volume.lastSubmittedUpdates.size()) {
    return 0u;
  }
  const glm::uvec3 physical = ddgiProbeCoordinate(probe, layout.probeCounts);
  const glm::uvec3 logical =
      (physical + layout.probeCounts - layout.ringOrigin) % layout.probeCounts;
  uint32_t flags = 0u;
  for (const DDGIDirtyRegion &region : pendingRegions) {
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

void DDGIFeature::stagePendingRadiometricSnapshot(
    const RenderScene &scene) noexcept {
  pending_.localLights.clear();
  const std::span<const LightId> ids = scene.packedLocalLightIds();
  const std::span<const LocalLightGpuData> lights = scene.packedLocalLights();
  pending_.localLights.reserve(std::min(ids.size(), lights.size()));
  for (size_t index = 0u; index < std::min(ids.size(), lights.size());
       ++index) {
    pending_.localLights.push_back({.id = ids[index], .data = lights[index]});
  }
  pending_.directionalLights.assign(scene.packedDirectionalLights().begin(),
                                    scene.packedDirectionalLights().end());
}

void DDGIFeature::clearPendingDirtySourceFacts() noexcept {
  pending_.sources.geometryTopology = UINT64_MAX;
  pending_.sources.geometryTransform = UINT64_MAX;
  pending_.sources.geometryDeformation = UINT64_MAX;
  pending_.sources.lightTopology = UINT64_MAX;
  pending_.sources.lightTransform = UINT64_MAX;
  pending_.sources.material = UINT64_MAX;
  pending_.sources.environment = UINT64_MAX;
  pending_.localLights.clear();
  pending_.directionalLights.clear();
  pending_.sources.geometry = false;
  pending_.sources.radiometric = false;
}

void DDGIFeature::buildScrollPlan(const RenderFrameContext &frame) {
  pending_.scrollLayouts.clear();
  scrollInvalidations_.clear();
  pending_.scrollLayouts.reserve(volumes_.size());
  for (const VolumeResource &volume : volumes_) {
    pending_.scrollLayouts.push_back(volume.layout);
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
    DDGIVolumeLayout &pending = pending_.scrollLayouts[slot];
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
  // DDGI owns this ring independently from presentation. Two slots per
  // swapchain image keep ordinary delayed readback from coupling feature
  // progress to the presentation resource cadence.
  const uint32_t desiredRingCount =
      std::max(2u, 2u * std::max(1u, gpu_.getSwapchainImageCount()));
  const size_t primaryResultCapacity = static_cast<size_t>(std::min<uint64_t>(
      settings.maxRayQueriesPerFrame,
      static_cast<uint64_t>(settings.maxProbeUpdatesPerFrame) *
          settings.raysPerProbe));
  if (frameSlots_.size() != desiredRingCount) {
    const bool hasInFlight =
        std::ranges::any_of(frameSlots_, [](const FrameSlot &slot) {
          return slot.state == DDGIReadbackSlotState::Recording ||
                 slot.state == DDGIReadbackSlotState::Pending;
        });
    if (!hasInFlight) {
      clearFrameSlots();
      frameSlots_.resize(desiredRingCount);
    }
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
    const bool needsResize =
        slot.updateCapacity < settings.maxProbeUpdatesPerFrame ||
        slot.rayCapacity < primaryResultCapacity ||
        slot.invalidationCapacity < invalidationCapacity ||
        slot.localLightCapacity < localLightCount;
    if (needsResize && (slot.state == DDGIReadbackSlotState::Recording ||
                        slot.state == DDGIReadbackSlotState::Pending)) {
      continue;
    }
    if (slot.updateCapacity < settings.maxProbeUpdatesPerFrame) {
      slot.updates.reset();
      slot.updatesReadback.reset();
      slot.updateCapacity = settings.maxProbeUpdatesPerFrame;
      slot.probeStateResultsValid = false;
      slot.probeStateResultCount = 0u;
      slot.state = DDGIReadbackSlotState::Dropped;
    }
    if (slot.rayCapacity < primaryResultCapacity) {
      slot.rayResults.reset();
      slot.rayCapacity = primaryResultCapacity;
    }
    if (slot.invalidationCapacity < invalidationCapacity) {
      slot.invalidations.reset();
      slot.invalidationCapacity = invalidationCapacity;
    }
    if (slot.localLightCapacity < localLightCount) {
      slot.localLights.reset();
      slot.localLightCapacity = localLightCount;
      slot.traceCountersValid = false;
      slot.state = DDGIReadbackSlotState::Dropped;
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
  const RenderSettings::DDGISettings &settings = ctx.frame.settings.ddgi;
  const std::span<const LocalLightGpuData> localLights =
      ctx.frame.scene->packedLocalLights();
  totalLocalLightCount_ = static_cast<uint32_t>(std::min<size_t>(
      localLights.size(), std::numeric_limits<uint16_t>::max()));
  secondaryLightingPossible_ =
      !ctx.frame.scene->packedDirectionalLights().empty() ||
      (totalLocalLightCount_ != 0u && settings.maxLocalLightsPerHit != 0u);
  selectedLocalLights_.clear();
  const uint32_t candidateCap =
      std::min(settings.maxLocalLightsPerHit, totalLocalLightCount_);
  selectedLocalLights_.reserve(candidateCap);
  if (candidateCap != 0u) {
    for (uint32_t candidate = 0u; candidate < candidateCap; ++candidate) {
      const uint32_t lightIndex = ddgiUniformSubsetIndex(
          totalLocalLightCount_, candidateCap, submittedSequence_, candidate);
      selectedLocalLights_.push_back(localLights[lightIndex]);
    }
  }
  selectedLocalLightCount_ = static_cast<uint32_t>(selectedLocalLights_.size());
  const uint32_t packedGatherVariants =
      static_cast<uint32_t>(settings.opaqueGatherVariant) |
      (static_cast<uint32_t>(settings.transmissionGatherVariant) << 8u) |
      (static_cast<uint32_t>(settings.traceMultiBounceGatherVariant) << 16u);
  frameData_.activeCountDebugFlagsSampler = glm::uvec4(
      static_cast<uint32_t>(volumes_.size()), packedGatherVariants,
      kDDGIFrameGpuDataVersion, gpu_.getSamplerBindlessIndex(sampler_.get()));
  for (size_t index = 0u; index < volumes_.size(); ++index) {
    const VolumeResource &resource = volumes_[index];
    const DDGIEffectiveVolume &effective = frameEffectiveVolume(index);
    const DDGIVolumeLayout &layout =
        pending_.scrollLayouts.size() == volumes_.size()
            ? pending_.scrollLayouts[index]
            : resource.layout;
    DDGIVolumeGpuData &gpuVolume = frameData_.volumes[index];
    gpuVolume.worldFromLocal = layout.worldFromLocal;
    gpuVolume.localFromWorld = layout.localFromWorld;
    gpuVolume.probeStateBufferAddress =
        resource.ready ? gpu_.getBufferDeviceAddress(resource.probeState.get())
                       : 0u;
    gpuVolume.resourceFlags = resource.ready ? kDDGIResourceReady : 0u;
    gpuVolume.localLightSubsetOffsetCount =
        std::min(selectedLocalLightCount_, uint32_t{0xffffu});
    gpuVolume.probeSpacingAndBias =
        glm::vec4(layout.probeSpacing, ctx.frame.settings.ddgi.selfShadowBias);
    const RenderSettings::DDGISettings &ddgiSettings = ctx.frame.settings.ddgi;
    gpuVolume.rayBiases = glm::vec4(
        ddgiSettings.primaryProbeBias, ddgiSettings.localShadowBias,
        ddgiSettings.directionalShadowBias, ddgiSettings.classificationBias);
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
    const uint64_t effectiveHash = ddgiEffectiveVolumeKeyHash(effective.key);
    gpuVolume.effectiveIdentity = glm::uvec4(
        static_cast<uint32_t>(effective.key.kind), low32(effectiveHash),
        high32(effectiveHash), effective.cascadeIndex);
    gpuVolume.tierTransitionCoverageFlags = glm::uvec4(
        static_cast<uint32_t>(effective.tier), effective.transitionCells,
        effective.requestedCoverageAchieved ? 1u : 0u,
        resource.ready ? 1u : 0u);
    gpuVolume.continuousCameraLocal =
        glm::vec4(effective.continuousCameraLocal, 0.0f);
    gpuVolume.fadeStartHalfExtents =
        glm::vec4(effective.fadeStartHalfExtents, 0.0f);
    gpuVolume.fadeEndHalfExtents =
        glm::vec4(effective.fadeEndHalfExtents, 0.0f);
  }
  auto upload = gpu_.updateBuffer(slot.frameData.get(), bytesOf(frameData_));
  if (upload.hasError()) {
    return upload;
  }
  ++ctx.frame.metrics.ddgi.uploadSubmissionCount;
  if (!selectedLocalLights_.empty()) {
    auto lightUpload = gpu_.updateBuffer(
        slot.localLights.get(), bytesOf(std::span(selectedLocalLights_)),
        sizeof(DDGITraceCountersGpuData));
    if (lightUpload.hasError()) {
      return lightUpload;
    }
    ++ctx.frame.metrics.ddgi.uploadSubmissionCount;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DDGIFeature::appendInitializationPass(FrameBuildContext &ctx, FrameSlot &slot) {
  blendPushConstants_.clear();
  dispatches_.clear();
  textureUses_.clear();
  blendPushConstants_.reserve(pending_.volumes.size() * 2u);
  dispatches_.reserve(pending_.volumes.size() * 2u);
  for (uint32_t volumeSlot = 0u;
       volumeSlot < static_cast<uint32_t>(pending_.volumes.size());
       ++volumeSlot) {
    const VolumeResource &volume = pending_.volumes[volumeSlot];
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
      appendUnique(textureUses_, outputs[type], RenderGraphAccessMode::Write);
    }
  }
  if (!dispatches_.empty()) {
    auto pass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
        .executionMode = RenderPassExecutionMode::ComputeOnly,
        .hasColorAttachment = false,
        .preDispatches = dispatches_,
        .importedTextureUses = textureUses_,
        .gpuTimingScope = GpuTimingScope::DDGIUpdate,
        .debugLabel = "DDGI Initialize Volumes",
        .debugColor = 0xff55cc88u,
    });
    if (pass.hasError()) {
      return Result<bool, std::string>::makeError(pass.error());
    }
  }
  pending_.initializationScheduled = true;
  pending_.scheduledFrameIndex = ctx.frame.frameIndex;
  pending_.resetEpoch = ctx.frame.settings.ddgi.requestedEpochs.resetHistory;
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
  ++ctx.frame.metrics.ddgi.uploadSubmissionCount;
  blendPushConstants_.clear();
  dispatches_.clear();
  irradianceDispatches_.clear();
  distanceDispatches_.clear();
  bufferUses_.clear();
  textureUses_.clear();
  irradianceTextureUses_.clear();
  distanceTextureUses_.clear();
  blendPushConstants_.reserve(volumes_.size() * 2u);
  dispatches_.reserve(volumes_.size() * 2u + 1u);
  appendUnique(bufferUses_, slot.frameData.get(), RenderGraphAccessMode::Read);
  appendUnique(bufferUses_, slot.invalidations.get(),
               RenderGraphAccessMode::Read);
  uint32_t invalidationOffset = 0u;
  for (uint32_t volumeSlot = 0u;
       volumeSlot < static_cast<uint32_t>(volumes_.size()); ++volumeSlot) {
    const VolumeResource &volume = volumes_[volumeSlot];
    const uint32_t firstInvalidation = invalidationOffset;
    while (invalidationOffset < scrollInvalidations_.size() &&
           scrollInvalidations_[invalidationOffset].volumeStableId ==
               volumeSlot) {
      ++invalidationOffset;
    }
    const uint32_t volumeInvalidationCount =
        invalidationOffset - firstInvalidation;
    if (!volume.ready || volumeInvalidationCount == 0u) {
      continue;
    }
    const std::array outputs{volume.irradiance.get(), volume.distance.get()};
    const std::array pipelineIndices{BlendIrradiance, BlendDistance};
    for (size_t type = 0u; type < outputs.size(); ++type) {
      blendPushConstants_.push_back(BlendPushConstants{
          .frame = gpu_.getBufferDeviceAddress(slot.frameData.get()),
          .updates = gpu_.getBufferDeviceAddress(slot.invalidations.get()),
          .updateCount = volumeInvalidationCount,
          .updateOffset = firstInvalidation,
          .volumeSlot = volumeSlot,
          .outputTextureId = gpu_.getTextureBindlessIndex(outputs[type]),
          .clearMode = 2u,
      });
      dispatches_.push_back(ComputeDispatchItem{
          .pipeline = pipelines_[pipelineIndices[type]].get(),
          .dispatch = {.x = volumeInvalidationCount, .y = 1u, .z = 1u},
          .pushConstants = bytesOf(blendPushConstants_.back()),
          .debugLabel =
              type == 0u ? "DDGI Scroll Irradiance" : "DDGI Scroll Distance",
          .debugColor = 0xffcc9955u,
      });
      appendUnique(textureUses_, outputs[type],
                   RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
    }
  }

  statePushConstants_ = {};
  statePushConstants_.updates =
      gpu_.getBufferDeviceAddress(slot.invalidations.get());
  statePushConstants_.frame = gpu_.getBufferDeviceAddress(slot.frameData.get());
  statePushConstants_.updateCount =
      static_cast<uint32_t>(scrollInvalidations_.size());
  statePushConstants_.clearMode = 1u;
  statePushConstants_.classificationEnabled =
      ctx.frame.settings.ddgi.classification ? 1u : 0u;
  for (size_t index = 0u; index < volumes_.size(); ++index) {
    statePushConstants_.states[index] =
        volumes_[index].ready
            ? gpu_.getBufferDeviceAddress(volumes_[index].probeState.get())
            : 0u;
    appendUnique(bufferUses_, volumes_[index].probeState.get(),
                 RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
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
      .importedBufferUses = bufferUses_,
      .importedTextureUses = textureUses_,
      .gpuTimingScope = GpuTimingScope::DDGIUpdate,
      .debugLabel = "DDGI Scroll Volumes",
      .debugColor = 0xffcc9955u,
  });
  if (pass.hasError()) {
    return Result<bool, std::string>::makeError(pass.error());
  }
  pending_.scrollScheduled = true;
  pending_.scheduledFrameIndex = ctx.frame.frameIndex;
  return Result<bool, std::string>::makeResult(true);
}

Result<DDGIScheduleResult, std::string>
DDGIFeature::buildSchedule(const RenderSettings::DDGISettings &settings) {
  ScopedScratch scoped(scratch_);
  secondaryQueriesPer1024Primary_ = secondaryLightingPossible_ ? 1024u : 0u;
  bool sceneFitBootstrap = false;
  if (frameCoverageSettings().mode == DDGICoverageMode::Hybrid) {
    for (size_t index = 0u; index < volumes_.size(); ++index) {
      if (frameEffectiveVolume(index).key.kind ==
              DDGIEffectiveVolumeKind::SceneFit &&
          std::ranges::any_of(volumes_[index].submittedProbeStates,
                              [](const DDGIProbeStateGpuData &state) {
                                return !historyReadyState(
                                    state.stateAgeFlags.x);
                              })) {
        sceneFitBootstrap = true;
        break;
      }
    }
  }
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
    const DDGIEffectiveVolume &effective = frameEffectiveVolume(slot);
    const uint64_t stableKey = ddgiEffectiveVolumeKeyHash(effective.key);
    const bool classifyVolume =
        settings.classification &&
        effective.key.kind == DDGIEffectiveVolumeKind::Authored;
    const bool bootstrapTier =
        sceneFitBootstrap &&
        effective.key.kind == DDGIEffectiveVolumeKind::SceneFit;
    tiers.push_back(DDGITierScheduleInput{
        .stableKey = stableKey,
        .submittedDeficit = volume.schedulerDeficit,
        .submittedStarvationFrames = volume.schedulerStarvationFrames,
        .effectiveOrder = bootstrapTier ? 0u : slot + 1u,
        .weight = bootstrapTier ? std::max(settings.maxProbeUpdatesPerFrame,
                                           tierWeight(effective.tier))
                                : tierWeight(effective.tier),
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
          .probe =
              DDGIProbeScheduleCandidate{
                  .volumeStableId = slot,
                  .probeId = probe,
                  .state = scrolled ? (classifyVolume
                                           ? DDGIProbeState::Uninitialized
                                           : DDGIProbeState::NewlyVigilant)
                                    : static_cast<DDGIProbeState>(
                                          volume.submittedProbeStates[probe]
                                              .stateAgeFlags.x),
                  .lastSubmittedUpdate = volume.lastSubmittedUpdates[probe],
                  .invalidated = volume.lastSubmittedUpdates[probe] == 0u ||
                                 scrolled || dirtyFlags != 0u,
                  .classificationIteration =
                      scrolled
                          ? 0u
                          : volume.submittedProbeStates[probe].stateAgeFlags.z,
                  .radianceRayCount =
                      !classifyVolume &&
                              (volume.lastSubmittedUpdates[probe] == 0u ||
                               scrolled)
                          ? settings.raysPerProbe
                          : 0u,
              },
      });
    }
  }
  workspace.resize(candidates.size());
  output.resize(settings.maxProbeUpdatesPerFrame);
  auto schedule = scheduleDDGITieredProbeUpdates(
      candidates, tiers,
      DDGISchedulerLimits{
          .raysPerProbe = settings.raysPerProbe,
          .classificationRaysPerProbe = settings.classificationRaysPerProbe,
          .maxProbeUpdates = settings.maxProbeUpdatesPerFrame,
          .maxRadianceProbeUpdates = settings.maxRadianceProbeUpdatesPerFrame,
          .maxMaintenanceProbeUpdates =
              settings.maxMaintenanceProbeUpdatesPerFrame,
          .maxRayQueries = settings.maxRayQueriesPerFrame,
          .secondaryQueriesPer1024Primary = secondaryQueriesPer1024Primary_,
          .forceFullUpdate = ddgiEpochIsPending(
              settings.requestedEpochs.forceFullUpdate, consumedForceEpoch_),
      },
      workspace, output);
  if (schedule.hasError()) {
    return Result<DDGIScheduleResult, std::string>::makeError(
        "DDGIFeature: deterministic scheduler rejected its frame workspace");
  }
  pending_.tierSchedule = schedule.value();
  const bool forceRequested = ddgiEpochIsPending(
      settings.requestedEpochs.forceFullUpdate, consumedForceEpoch_);
  for (uint32_t index = 0u; index < schedule.value().schedule.updatedProbes;
       ++index) {
    DDGIProbeUpdateEntry &entry = output[index];
    entry.flags |= dirtyFlagsForProbe(entry.volumeStableId, entry.probeId);
    const bool scrolled = std::ranges::binary_search(
        scrollInvalidations_, std::pair(entry.volumeStableId, entry.probeId),
        {}, [](const auto &item) {
          return std::pair(item.volumeStableId, item.probeId);
        });
    const VolumeResource &volume = volumes_[entry.volumeStableId];
    const bool bootstrap =
        !scrolled && volume.lastSubmittedUpdates[entry.probeId] == 0u;
    if (bootstrap) {
      entry.flags |= kDDGIProbeUpdateReasonBootstrap;
    }
    if (scrolled) {
      entry.flags |= kDDGIProbeUpdateReasonScroll;
    }
    if ((entry.flags & (kDDGIProbeUpdateReclassify | kDDGIProbeUpdateWake)) !=
        0u) {
      entry.flags |= kDDGIProbeUpdateReasonDirtyGeometry;
    }
    if ((entry.flags & (kDDGIProbeUpdateIrradianceResponse |
                        kDDGIProbeUpdateDistanceResponse)) != 0u) {
      entry.flags |= kDDGIProbeUpdateReasonRadiometric;
    }
    if ((entry.flags & kDDGIProbeUpdateWake) != 0u) {
      entry.flags |= kDDGIProbeUpdateReasonWake;
    }
    if ((entry.flags & kDDGIProbeUpdateReclassify) != 0u) {
      entry.flags |= kDDGIProbeUpdateReasonReclassification;
    }
    if (forceRequested) {
      entry.flags |= kDDGIProbeUpdateReasonForce;
    }
    if ((entry.flags & kDDGIProbeUpdateReasonMask) == 0u) {
      entry.flags |= kDDGIProbeUpdateReasonMaintenance;
    }
  }
  scheduledEntries_.assign(
      output.begin(), output.begin() + schedule.value().schedule.updatedProbes);
  dispatchEntries_ = scheduledEntries_;
  std::ranges::sort(
      dispatchEntries_, {}, [](const DDGIProbeUpdateEntry &entry) {
        const uint32_t workClass =
            (entry.flags & kDDGIProbeUpdateClassificationGeometry) != 0u ? 0u
                                                                         : 1u;
        return std::tuple(workClass, entry.volumeStableId, entry.rayBase);
      });
  uint32_t rayBase = 0u;
  for (DDGIProbeUpdateEntry &entry : dispatchEntries_) {
    entry.rayBase = rayBase;
    rayBase += entry.rayCount;
  }
  return Result<DDGIScheduleResult, std::string>::makeResult(
      schedule.value().schedule);
}

Result<bool, std::string>
DDGIFeature::appendUpdatePasses(FrameBuildContext &ctx, FrameSlot &slot,
                                const DDGIScheduleResult &schedule) {
  DDGITraceCountersGpuData initialCounters{};
  // Primary dispatch cardinality is known exactly from the CPU schedule. Seed
  // it here instead of paying one diagnostic atomic per primary ray.
  initialCounters.primaryQueriesIssued = schedule.primaryQueries;
  for (const DDGIProbeUpdateEntry &entry : dispatchEntries_) {
    if (entry.volumeStableId <
        initialCounters.primaryQueriesIssuedByVolume.size()) {
      initialCounters.primaryQueriesIssuedByVolume[entry.volumeStableId] +=
          entry.rayCount;
    }
  }
  initialCounters.secondaryQueriesReserved = schedule.secondaryQueriesReserved;
  initialCounters.sourceFrame = static_cast<uint32_t>(ctx.frame.frameIndex);
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
  const std::array uploads{
      BufferUpdate{.buffer = slot.localLights.get(),
                   .data = bytesOf(initialCounters)},
      BufferUpdate{.buffer = slot.updates.get(),
                   .data = bytesOf(std::span(dispatchEntries_))},
  };
  auto upload = gpu_.updateBuffers(uploads);
  if (upload.hasError()) {
    return upload;
  }
  ++ctx.frame.metrics.ddgi.uploadSubmissionCount;
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
  const uint32_t directionalLightCount = static_cast<uint32_t>(std::min<size_t>(
      directionalLights.size(), std::numeric_limits<uint16_t>::max()));
  const uint32_t directionalLightIndex =
      directionalLightCount == 0u
          ? 0u
          : ddgiUniformSubsetIndex(directionalLightCount, 1u,
                                   submittedSequence_, 0u);
  const DirectionalLightGpuData *directionalLight =
      directionalLightIndex < directionalLights.size()
          ? &directionalLights[directionalLightIndex]
          : nullptr;
  float directionalTraceExtent = 1.0f;
  if (rt.staticCoverageBounds.valid) {
    directionalTraceExtent =
        std::max(glm::length(rt.staticCoverageBounds.bounds.max_ -
                             rt.staticCoverageBounds.bounds.min_),
                 1.0f);
  } else {
    for (const VolumeResource &volume : volumes_) {
      directionalTraceExtent =
          std::max(directionalTraceExtent, 4.0f * volume.desc.maxRayDistance);
    }
  }
  const uint32_t packedTraceLimits =
      (std::min(static_cast<uint32_t>(std::ceil(directionalTraceExtent)),
                static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()))
       << 16u) |
      std::min(ctx.frame.settings.ddgi.maxCandidateIntersectionsPerRay,
               static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()));
  const uint32_t packedLocalLightCounts =
      (std::min(totalLocalLightCount_,
                static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()))
       << 16u) |
      std::min(selectedLocalLightCount_,
               static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()));
  TracePushConstants traceBase{
      .frame = gpu_.getBufferDeviceAddress(slot.frameData.get()),
      .updates = gpu_.getBufferDeviceAddress(slot.updates.get()),
      .results = gpu_.getBufferDeviceAddress(slot.rayResults.get()),
      .instances = rt.instanceTableAddress,
      .geometries = rt.geometryTableAddress,
      .materials = materials[MaterialTableRegion::Header].address,
      .directionalDirectionIlluminance =
          directionalLight != nullptr ? directionalLight->directionIlluminance
                                      : glm::vec4(0.0f),
      .directionalColor =
          directionalLight == nullptr
              ? glm::vec4(0.0f, 0.0f, 0.0f,
                          ctx.frame.settings.ddgi.multiBounceLuminanceClamp)
              : glm::vec4(glm::vec3(directionalLight->colorReserved),
                          ctx.frame.settings.ddgi.multiBounceLuminanceClamp),
      .localLights = gpu_.getBufferDeviceAddress(slot.localLights.get()),
      .updateCount = 0u,
      .raysPerProbe = ctx.frame.settings.ddgi.raysPerProbe,
      .skyTextureId = skyTextureId,
      .skySamplerId = gpu_.getCubemapSamplerBindlessIndex(),
      .maxCandidates = packedTraceLimits,
      .frameSeed = static_cast<uint32_t>(submittedSequence_),
      .materialSamplerId = gpu_.getDefaultSamplerBindlessIndex(),
      .directionalLightCount =
          (directionalLightCount << 16u) |
          static_cast<uint32_t>(directionalLight != nullptr),
      .localLightCount = packedLocalLightCounts,
      .maxLocalLights =
          ctx.frame.settings.ddgi.maxLocalLightsPerHit |
          (ctx.frame.settings.ddgi.multiBounce ? 0x80000000u : 0u) |
          (ctx.frame.settings.ddgi.diagnosticCounters ? 0x40000000u : 0u),
  };
  bufferUses_.clear();
  appendUnique(bufferUses_, slot.frameData.get(), RenderGraphAccessMode::Read);
  appendUnique(bufferUses_, slot.updates.get(), RenderGraphAccessMode::Read);
  appendUnique(bufferUses_, slot.rayResults.get(),
               RenderGraphAccessMode::Write);
  appendUnique(bufferUses_, slot.localLights.get(),
               RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  for (const VolumeResource &volume : volumes_) {
    appendUnique(bufferUses_, volume.probeState.get(),
                 RenderGraphAccessMode::Read);
  }
  for (BufferHandle reference : rt.indirectSubmissionReferences) {
    appendUnique(bufferUses_, reference, RenderGraphAccessMode::Read);
  }
  textureUses_.clear();
  appendUnique(textureUses_, skyTexture, RenderGraphAccessMode::Read);
  for (const VolumeResource &volume : volumes_) {
    appendUnique(textureUses_, volume.irradiance.get(),
                 RenderGraphAccessMode::Read);
    appendUnique(textureUses_, volume.distance.get(),
                 RenderGraphAccessMode::Read);
  }
  for (TextureHandle reference : rt.indirectSubmissionTextureReferences) {
    appendUnique(textureUses_, reference, RenderGraphAccessMode::Read);
  }
  const uint32_t classificationCount = static_cast<uint32_t>(
      std::ranges::count_if(dispatchEntries_, [](const auto &entry) {
        return (entry.flags & kDDGIProbeUpdateClassificationGeometry) != 0u;
      }));
  const uint32_t radianceCount = schedule.updatedProbes - classificationCount;
  std::array<ComputeDispatchItem, 2> traceDispatches{};
  uint32_t traceDispatchCount = 0u;
  const auto appendTraceDispatch = [&](uint32_t offset, uint32_t count,
                                       uint32_t raysPerProbe,
                                       std::string_view label) {
    if (count == 0u) {
      return;
    }
    TracePushConstants &push = tracePushConstants_[traceDispatchCount];
    push = traceBase;
    push.updateCount = count;
    // Low 12 bits keep the preset ray count used by material LOD; the
    // remaining bits carry the compact update-buffer base for this work
    // class without growing the 128-byte push-constant contract.
    push.raysPerProbe = (offset << 12u) | (raysPerProbe & 0xfffu);
    traceDispatches[traceDispatchCount++] = ComputeDispatchItem{
        .pipeline = pipelines_[Trace].get(),
        .rayQueryBinding = traceBinding_.get(),
        .dispatch = {.x = count, .y = divRoundUp(raysPerProbe, 64u), .z = 1u},
        .pushConstants = bytesOf(push),
        .debugLabel = label,
        .debugColor = 0xff44aaffu,
    };
  };
  appendTraceDispatch(0u, classificationCount,
                      ctx.frame.settings.ddgi.classificationRaysPerProbe,
                      "DDGI Trace Classification");
  appendTraceDispatch(classificationCount, radianceCount,
                      ctx.frame.settings.ddgi.raysPerProbe,
                      "DDGI Trace Full Radiance");
  auto tracePass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
      .executionMode = RenderPassExecutionMode::ComputeOnly,
      .hasColorAttachment = false,
      .preDispatches = std::span(traceDispatches).first(traceDispatchCount),
      .importedBufferUses = bufferUses_,
      .importedTextureUses = textureUses_,
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
  bufferUses_.clear();
  textureUses_.clear();
  appendUnique(bufferUses_, slot.frameData.get(), RenderGraphAccessMode::Read);
  appendUnique(bufferUses_, slot.updates.get(), RenderGraphAccessMode::Read);
  appendUnique(bufferUses_, slot.rayResults.get(), RenderGraphAccessMode::Read);
  for (const VolumeResource &volume : volumes_) {
    if (volume.ready) {
      appendUnique(bufferUses_, volume.probeState.get(),
                   RenderGraphAccessMode::Read);
    }
  }
  blendPushConstants_.reserve(volumes_.size() * 2u);
  irradianceDispatches_.reserve(volumes_.size());
  distanceDispatches_.reserve(volumes_.size());
  irradianceTextureUses_.clear();
  distanceTextureUses_.clear();
  irradianceTextureUses_.reserve(volumes_.size());
  distanceTextureUses_.reserve(volumes_.size());
  textureUses_.reserve(volumes_.size());
  for (uint32_t volumeSlot = 0u;
       volumeSlot < static_cast<uint32_t>(volumes_.size()); ++volumeSlot) {
    const VolumeResource &volume = volumes_[volumeSlot];
    const auto first = std::ranges::find_if(
        dispatchEntries_, [volumeSlot](const DDGIProbeUpdateEntry &entry) {
          return entry.volumeStableId == volumeSlot &&
                 (entry.flags & kDDGIProbeUpdateClassificationGeometry) == 0u;
        });
    const auto last = std::find_if(
        first, dispatchEntries_.end(),
        [volumeSlot](const DDGIProbeUpdateEntry &entry) {
          return entry.volumeStableId != volumeSlot ||
                 (entry.flags & kDDGIProbeUpdateClassificationGeometry) != 0u;
        });
    const uint32_t firstUpdate =
        static_cast<uint32_t>(first - dispatchEntries_.begin());
    const uint32_t volumeUpdateCount = static_cast<uint32_t>(last - first);
    if (!volume.ready || volumeUpdateCount == 0u) {
      continue;
    }
    const std::array outputs{volume.irradiance.get(), volume.distance.get()};
    const std::array pipelineIndices{BlendIrradiance, BlendDistance};
    const RenderSettings::DDGISettings &volumeSettings =
        ctx.frame.settings.ddgi;
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
          .raysPerProbe = ctx.frame.settings.ddgi.raysPerProbe,
          .hysteresis = hysteresis[type],
          .historyValid = submittedSequence_ != 0u ? 1u : 0u,
          .frameSeed = static_cast<uint32_t>(submittedSequence_),
          .responseHysteresisScale = responseScales[type],
      });
      ComputeDispatchItem dispatch{
          .pipeline = pipelines_[pipelineIndices[type]].get(),
          .dispatch = {.x = volumeUpdateCount, .y = 1u, .z = 1u},
          .pushConstants = bytesOf(blendPushConstants_.back()),
          .debugLabel =
              type == 0u ? "DDGI Blend Irradiance" : "DDGI Blend Distance",
          .debugColor = 0xff55dd88u,
      };
      if (type == 0u) {
        irradianceDispatches_.push_back(dispatch);
        irradianceTextureUses_.push_back(
            {.texture = outputs[type],
             .access =
                 RenderGraphAccessMode::Read | RenderGraphAccessMode::Write});
      } else {
        distanceDispatches_.push_back(dispatch);
        distanceTextureUses_.push_back(
            {.texture = outputs[type],
             .access =
                 RenderGraphAccessMode::Read | RenderGraphAccessMode::Write});
      }
      dispatches_.push_back(dispatch);
      appendUnique(textureUses_, outputs[type],
                   RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
    }
  }
  const bool diagnosticTiming = ctx.frame.settings.ddgi.diagnosticCounters;
  if (!diagnosticTiming && !dispatches_.empty()) {
    auto updatePass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
        .executionMode = RenderPassExecutionMode::ComputeOnly,
        .hasColorAttachment = false,
        .preDispatches = dispatches_,
        .importedBufferUses = bufferUses_,
        .importedTextureUses = textureUses_,
        .gpuTimingScope = GpuTimingScope::DDGIUpdate,
        .debugLabel = "DDGI Update Atlases",
        .debugColor = 0xff55dd88u,
    });
    if (updatePass.hasError()) {
      return Result<bool, std::string>::makeError(updatePass.error());
    }
  }
  if (diagnosticTiming && !irradianceDispatches_.empty()) {
    auto updatePass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
        .executionMode = RenderPassExecutionMode::ComputeOnly,
        .hasColorAttachment = false,
        .preDispatches = irradianceDispatches_,
        .importedBufferUses = bufferUses_,
        .importedTextureUses = irradianceTextureUses_,
        .gpuTimingScope = GpuTimingScope::DDGIIrradianceUpdate,
        .debugLabel = "DDGI Update Irradiance Atlases",
        .debugColor = 0xff55dd88u,
    });
    if (updatePass.hasError()) {
      return Result<bool, std::string>::makeError(updatePass.error());
    }
  }
  if (diagnosticTiming && !distanceDispatches_.empty()) {
    auto updatePass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
        .executionMode = RenderPassExecutionMode::ComputeOnly,
        .hasColorAttachment = false,
        .preDispatches = distanceDispatches_,
        .importedBufferUses = bufferUses_,
        .importedTextureUses = distanceTextureUses_,
        .gpuTimingScope = GpuTimingScope::DDGIDistanceUpdate,
        .debugLabel = "DDGI Update Distance Atlases",
        .debugColor = 0xff55dd88u,
    });
    if (updatePass.hasError()) {
      return Result<bool, std::string>::makeError(updatePass.error());
    }
  }

  statePushConstants_ = {};
  statePushConstants_.updates = gpu_.getBufferDeviceAddress(slot.updates.get());
  statePushConstants_.results =
      gpu_.getBufferDeviceAddress(slot.rayResults.get());
  statePushConstants_.frame = gpu_.getBufferDeviceAddress(slot.frameData.get());
  statePushConstants_.updateCount = schedule.updatedProbes;
  statePushConstants_.submittedSequence =
      static_cast<uint32_t>(submittedSequence_ + 1u);
  const RenderSettings::DDGISettings &settings = ctx.frame.settings.ddgi;
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
  bufferUses_.clear();
  appendUnique(bufferUses_, slot.updates.get(),
               RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  appendUnique(bufferUses_, slot.rayResults.get(), RenderGraphAccessMode::Read);
  appendUnique(bufferUses_, slot.frameData.get(), RenderGraphAccessMode::Read);
  appendUnique(bufferUses_, rt.surfaceBounds, RenderGraphAccessMode::Read);
  for (size_t index = 0u; index < volumes_.size(); ++index) {
    statePushConstants_.states[index] =
        volumes_[index].ready
            ? gpu_.getBufferDeviceAddress(volumes_[index].probeState.get())
            : 0u;
    appendUnique(bufferUses_, volumes_[index].probeState.get(),
                 RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  }
  const std::array stateDispatches{ComputeDispatchItem{
      .pipeline = pipelines_[UpdateProbeState].get(),
      .dispatch = {.x = divRoundUp(schedule.updatedProbes, 64u),
                   .y = 1u,
                   .z = 1u},
      .pushConstants = bytesOf(statePushConstants_),
      .debugLabel = "DDGI Update Probe State",
      .debugColor = 0xff55bb77u,
  }};
  auto statePass = ctx.graph.addGraphicsPass(RenderGraphGraphicsPassDesc{
      .executionMode = RenderPassExecutionMode::ComputeOnly,
      .hasColorAttachment = false,
      .preDispatches = stateDispatches,
      .importedBufferUses = bufferUses_,
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
      .gpuTimingScope = GpuTimingScope::DDGIReadback,
      .debugLabel = "DDGI Readback Copies",
      .debugColor = 0xff55bb77u,
  });
  if (readbackPass.hasError()) {
    return Result<bool, std::string>::makeError(readbackPass.error());
  }
  slot.byteCount = schedule.updatedProbes * sizeof(DDGIProbeUpdateEntry) +
                   sizeof(DDGITraceCountersGpuData);
  scheduledReadbackBytes_ = slot.byteCount;
  pending_.updatesScheduled = true;
  pending_.scheduledFrameIndex = ctx.frame.frameIndex;
  pending_.forceEpoch = ctx.frame.settings.ddgi.requestedEpochs.forceFullUpdate;
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
      .materials = materials[MaterialTableRegion::Header].address,
      .requestId = request.requestId,
      .sceneId = ctx.frame.scene->id(),
      .layoutGeneration = found->layout.generation,
      .resourceGeneration = found->resourceGeneration,
      .deviceEpoch = deviceEpoch_,
      .volumeValue = request.volume.value,
      .volumeSlot = volumeSlot,
      .probeId = request.probeId,
      .rayCount = rayCount,
      .maxCandidates = ctx.frame.settings.ddgi.maxCandidateIntersectionsPerRay,
      .frameSeed = static_cast<uint32_t>(submittedSequence_),
      .materialSamplerId = gpu_.getDefaultSamplerBindlessIndex(),
      .submissionSequence = static_cast<uint32_t>(
          submittedSequence_ +
          ((pending_.initializationScheduled || pending_.scrollScheduled ||
            pending_.updatesScheduled)
               ? 1u
               : 0u)),
  };
  bufferUses_.clear();
  textureUses_.clear();
  appendUnique(bufferUses_, slot.frameData.get(), RenderGraphAccessMode::Read);
  appendUnique(bufferUses_, slot.diagnostic.get(),
               RenderGraphAccessMode::Write);
  for (const VolumeResource &volume : volumes_) {
    appendUnique(bufferUses_, volume.probeState.get(),
                 RenderGraphAccessMode::Read);
    appendUnique(textureUses_, volume.irradiance.get(),
                 RenderGraphAccessMode::Read);
    appendUnique(textureUses_, volume.distance.get(),
                 RenderGraphAccessMode::Read);
  }
  for (BufferHandle reference : rt.indirectSubmissionReferences) {
    appendUnique(bufferUses_, reference, RenderGraphAccessMode::Read);
  }
  for (TextureHandle reference : rt.indirectSubmissionTextureReferences) {
    appendUnique(textureUses_, reference, RenderGraphAccessMode::Read);
  }
  for (const MaterialTableGpuRegion &region : materials.regions) {
    appendUnique(bufferUses_, region.buffer, RenderGraphAccessMode::Read);
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
      .importedBufferUses = bufferUses_,
      .importedTextureUses = textureUses_,
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
  slot.requestId = request.requestId;
  slot.byteCount += kDDGIDiagnosticBufferBytes;
  scheduledReadbackBytes_ += kDDGIDiagnosticBufferBytes;
  slot.diagnosticValid = false;
  pending_.inspectionScheduled = true;
  pending_.scheduledFrameIndex = ctx.frame.frameIndex;
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
    const bool pendingMetadata = !pending_.replacement &&
                                 pending_.compatiblePlan &&
                                 slot < pending_.effectiveVolumeCount;
    const DDGICaptureMetadata metadata = makeCaptureMetadata(
        frameEffectiveVolume(slot), volumes_[slot].layout,
        volumes_[slot].resourceGeneration,
        pendingMetadata ? pending_.coverageGeneration : coverageGeneration_,
        pendingMetadata ? pending_.sceneBoundsGeneration
                        : sceneBoundsGeneration_,
        pendingMetadata ? pending_.requestedCoverageHalfExtents[slot]
                        : volumes_[slot].requestedCoverageHalfExtents,
        pendingMetadata ? pending_.achievedCoverageHalfExtents[slot]
                        : volumes_[slot].achievedCoverageHalfExtents);
    for (const auto &point :
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
    ctx.frame.metrics.ddgi.stateHistoryScanCount +=
        volume.submittedProbeStates.size();
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
      latestTraceCounters_ = counters;
      latestTraceCounterSceneId_ = slot.sceneId;
      latestTraceCounterDeviceEpoch_ = slot.deviceEpoch;
      latestTraceCounterFeatureGeneration_ = slot.featureGeneration;
      latestTraceCounterResourceGenerations_ =
          slot.probeStateResourceGenerations;
      latestTraceCountersAvailable_ = true;
      metrics.primaryQueriesIssued = counters.primaryQueriesIssued;
      metrics.traceCounterSourceFrame = counters.sourceFrame;
      metrics.secondaryQueriesReserved = counters.secondaryQueriesReserved;
      metrics.secondaryQueries = counters.secondaryQueries;
      metrics.secondaryQueryBudgetOverflows =
          counters.secondaryQueryBudgetOverflows;
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
      metrics.nonFiniteRadianceRejects = counters.nonFiniteRadianceRejects;
      metrics.emissiveRadianceClamps = counters.emissiveRadianceClamps;
      metrics.directRadianceClamps = counters.directRadianceClamps;
      metrics.skyRadianceClamps = counters.skyRadianceClamps;
      metrics.multiBounceRadianceClamps = counters.multiBounceRadianceClamps;
      metrics.finalRadianceClamps = counters.finalRadianceClamps;
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

void DDGIFeature::collectCompletedReadbacks(FrameBuildContext &ctx) {
  for (FrameSlot &slot : frameSlots_) {
    if (slot.state != DDGIReadbackSlotState::Pending ||
        !isValid(slot.submission) ||
        !gpu_.isSubmissionComplete(slot.submission)) {
      continue;
    }
    slot.state = DDGIReadbackSlotState::Completed;
    const bool compatible = ctx.frame.scene != nullptr &&
                            slot.sceneId == ctx.frame.scene->id() &&
                            slot.deviceEpoch == deviceEpoch_ &&
                            slot.featureGeneration == coverageGeneration_ &&
                            slot.payloadSchema == kDDGIFrameMetricsVersion;
    if (!compatible) {
      ++readbackGenerationMismatches_;
      slot.probeStateResultsValid = false;
      slot.traceCountersValid = false;
      slot.diagnosticValid = false;
      slot.diagnosticRequest.reset();
      slot.state = DDGIReadbackSlotState::Dropped;
      continue;
    }
  }

  // Submission completion is ordered, but physical ring indices are not.
  // Consume oldest-to-newest so a recycled low-index slot can never publish
  // newer probe state and then be overwritten by an older high-index slot.
  for (;;) {
    FrameSlot *oldest = nullptr;
    for (FrameSlot &slot : frameSlots_) {
      if (slot.state != DDGIReadbackSlotState::Completed ||
          (oldest != nullptr && slot.sourceFrame >= oldest->sourceFrame)) {
        continue;
      }
      oldest = &slot;
    }
    if (oldest == nullptr) {
      break;
    }
    collectCompletedProbeStates(*oldest);
    collectCompletedTraceMetrics(*oldest, ctx.frame.metrics.ddgi);
    collectCompletedInspection(ctx, *oldest);
    oldest->state = DDGIReadbackSlotState::Consumed;
  }
}

DDGIFeature::FrameSlot *
DDGIFeature::acquireFrameSlot(uint64_t frameIndex) noexcept {
  for (FrameSlot &slot : frameSlots_) {
    if (slot.state == DDGIReadbackSlotState::Recording &&
        !isValid(slot.submission)) {
      slot.state = DDGIReadbackSlotState::Dropped;
    }
    if (slot.state == DDGIReadbackSlotState::Free ||
        slot.state == DDGIReadbackSlotState::Consumed ||
        slot.state == DDGIReadbackSlotState::Dropped) {
      slot.state = DDGIReadbackSlotState::Recording;
      slot.submission = {};
      slot.sourceFrame = frameIndex;
      slot.sceneId = sceneId_;
      slot.deviceEpoch = deviceEpoch_;
      slot.featureGeneration = coverageGeneration_;
      slot.payloadSchema = kDDGIFrameMetricsVersion;
      slot.requestId = 0u;
      slot.byteCount = 0u;
      return &slot;
    }
  }
  ++readbackDroppedSamples_;
  return nullptr;
}

void DDGIFeature::publishReadbackMetrics(DDGIFrameMetrics &metrics,
                                         uint64_t frameIndex) const noexcept {
  uint64_t oldestPendingAge = 0u;
  uint32_t pendingCount = 0u;
  uint64_t perSlotBytes = 0u;
  uint64_t ringDeviceBytes = 0u;
  uint64_t ringReadbackBytes = 0u;
  const auto allocatedBytes = [](bool valid, uint64_t requested) {
    return valid ? std::max<uint64_t>(requested, 16u) : 0u;
  };
  for (const FrameSlot &slot : frameSlots_) {
    const uint64_t slotReadbackBytes =
        allocatedBytes(slot.updatesReadback.valid(),
                       static_cast<uint64_t>(slot.updateCapacity) *
                           sizeof(DDGIProbeUpdateEntry)) +
        allocatedBytes(slot.traceCountersReadback.valid(),
                       sizeof(DDGITraceCountersGpuData)) +
        allocatedBytes(slot.diagnosticReadback.valid(),
                       kDDGIDiagnosticBufferBytes);
    const uint64_t slotDeviceBytes =
        allocatedBytes(slot.frameData.valid(), sizeof(DDGIFrameGpuData)) +
        allocatedBytes(slot.updates.valid(),
                       static_cast<uint64_t>(slot.updateCapacity) *
                           sizeof(DDGIProbeUpdateEntry)) +
        allocatedBytes(slot.invalidations.valid(),
                       static_cast<uint64_t>(slot.invalidationCapacity) *
                           sizeof(DDGIProbeUpdateEntry)) +
        allocatedBytes(slot.rayResults.valid(),
                       static_cast<uint64_t>(slot.rayCapacity) *
                           sizeof(DDGIRayResultGpuData)) +
        allocatedBytes(slot.localLights.valid(),
                       sizeof(DDGITraceCountersGpuData) +
                           static_cast<uint64_t>(slot.localLightCapacity) *
                               sizeof(LocalLightGpuData)) +
        allocatedBytes(slot.diagnostic.valid(), kDDGIDiagnosticBufferBytes);
    perSlotBytes = std::max(perSlotBytes, slotReadbackBytes);
    ringDeviceBytes += slotDeviceBytes;
    ringReadbackBytes += slotReadbackBytes;
    if (slot.state != DDGIReadbackSlotState::Pending) {
      continue;
    }
    ++pendingCount;
    oldestPendingAge = std::max(
        oldestPendingAge,
        frameIndex >= slot.sourceFrame ? frameIndex - slot.sourceFrame : 0u);
  }
  metrics.readbackPendingSlots = pendingCount;
  metrics.readbackDroppedSamples = static_cast<uint32_t>(
      std::min(readbackDroppedSamples_,
               static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
  metrics.readbackGenerationMismatches = static_cast<uint32_t>(
      std::min(readbackGenerationMismatches_,
               static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
  metrics.readbackEarlyReuseAttempts = static_cast<uint32_t>(
      std::min(readbackEarlyReuseAttempts_,
               static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
  metrics.readbackOldestPendingAge = static_cast<uint32_t>(
      std::min(oldestPendingAge,
               static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
  metrics.readbackCopyBytes = scheduledReadbackBytes_;
  metrics.readbackPerSlotBytes = perSlotBytes;
  metrics.readbackRingBytes = ringReadbackBytes;
  metrics.frameSlotCount = static_cast<uint32_t>(frameSlots_.size());
  metrics.frameRingDeviceBytes = ringDeviceBytes;
  metrics.frameRingReadbackBytes = ringReadbackBytes;
  metrics.frameRingBytes = ringDeviceBytes + ringReadbackBytes;
  metrics.readbackWaits = 0u;
  metrics.readbackBlockingFallbacks = 0u;

  if (!latestTraceCountersAvailable_ ||
      latestTraceCounterSceneId_ != sceneId_ ||
      latestTraceCounterDeviceEpoch_ != deviceEpoch_ ||
      latestTraceCounterFeatureGeneration_ != coverageGeneration_) {
    return;
  }
  const DDGITraceCountersGpuData &counters = latestTraceCounters_;
  metrics.traceCountersAvailable = 1u;
  metrics.primaryQueriesIssued = counters.primaryQueriesIssued;
  metrics.traceCounterSourceFrame = counters.sourceFrame;
  metrics.traceCounterStaleFrames = static_cast<uint32_t>(
      std::min(frameIndex >= counters.sourceFrame
                   ? frameIndex - static_cast<uint64_t>(counters.sourceFrame)
                   : 0u,
               static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
  metrics.secondaryQueriesReserved = counters.secondaryQueriesReserved;
  metrics.secondaryQueries = counters.secondaryQueries;
  metrics.secondaryQueryBudgetOverflows =
      counters.secondaryQueryBudgetOverflows;
  metrics.localSecondaryQueries = counters.localSecondaryQueries;
  metrics.directionalSecondaryQueries =
      counters.secondaryQueries -
      std::min(counters.secondaryQueries, counters.localSecondaryQueries);
  metrics.secondaryQueriesUnused =
      counters.secondaryQueriesReserved -
      std::min(counters.secondaryQueriesReserved, counters.secondaryQueries);
  metrics.primaryCandidateIntersections =
      counters.primaryCandidateIntersections;
  metrics.secondaryCandidateIntersections =
      counters.secondaryCandidateIntersections;
  metrics.alphaCandidateRejections = counters.alphaCandidateRejections;
  metrics.backfaceCandidateRejections = counters.backfaceCandidateRejections;
  metrics.candidateOverflows = counters.candidateOverflows;
  metrics.localLightTruncations = counters.localLightTruncations;
  metrics.nonFiniteRadianceRejects = counters.nonFiniteRadianceRejects;
  metrics.emissiveRadianceClamps = counters.emissiveRadianceClamps;
  metrics.directRadianceClamps = counters.directRadianceClamps;
  metrics.skyRadianceClamps = counters.skyRadianceClamps;
  metrics.multiBounceRadianceClamps = counters.multiBounceRadianceClamps;
  metrics.finalRadianceClamps = counters.finalRadianceClamps;
  for (size_t volumeIndex = 0u; volumeIndex < volumes_.size(); ++volumeIndex) {
    if (latestTraceCounterResourceGenerations_[volumeIndex] !=
        volumes_[volumeIndex].resourceGeneration) {
      continue;
    }
    metrics.volumes[volumeIndex].primaryQueriesIssued =
        counters.primaryQueriesIssuedByVolume[volumeIndex];
    metrics.volumes[volumeIndex].secondaryQueries =
        counters.secondaryQueriesByVolume[volumeIndex];
  }
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
      const DDGIEffectiveVolume &effective = frameEffectiveVolume(slotIndex);
      volumeKeys[slotIndex] = effective.key;
      volumeIds[slotIndex] = volume.id;
      probeCounts[slotIndex] =
          static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
      minimumProbeSpacing[slotIndex] =
          std::min({volume.layout.probeSpacing.x, volume.layout.probeSpacing.y,
                    volume.layout.probeSpacing.z});
      captureMetadata[slotIndex] = makeCaptureMetadata(
          effective, volume.layout, volume.resourceGeneration,
          pending_.compatiblePlan ? pending_.coverageGeneration
                                  : coverageGeneration_,
          pending_.compatiblePlan ? pending_.sceneBoundsGeneration
                                  : sceneBoundsGeneration_,
          pending_.compatiblePlan
              ? pending_.requestedCoverageHalfExtents[slotIndex]
              : volume.requestedCoverageHalfExtents,
          pending_.compatiblePlan
              ? pending_.achievedCoverageHalfExtents[slotIndex]
              : volume.achievedCoverageHalfExtents);
      captureMetadata[slotIndex].valid = volume.ready ? 1u : 0u;
    }
    forwardDependencyBuffers_.push_back(slot.frameData.get());
    if (pending_.inspectionScheduled && slot.diagnostic.valid()) {
      forwardDependencyBuffers_.push_back(slot.diagnostic.get());
    }
    ctx.shared.ddgiFrameGpuData = DDGIFrameGpuDataHandle{
        .buffer = slot.frameData.get(),
        .bufferAddress = gpu_.getBufferDeviceAddress(slot.frameData.get()),
        .dependencyBuffers = forwardDependencyBuffers_,
        .dependencyTextures = forwardDependencyTextures_,
        .activeVolumeCount = static_cast<uint32_t>(volumes_.size()),
        .flags = kDDGIFrameEnabled,
        .debugView = ctx.frame.settings.ddgi.debugView,
        .volumeKeys = volumeKeys,
        .volumeIds = volumeIds,
        .probeCounts = probeCounts,
        .minimumProbeSpacing = minimumProbeSpacing,
        .captureMetadata = captureMetadata,
        .coverageGeneration = coverageGeneration_,
        .sceneBoundsGeneration = sceneBoundsGeneration_,
        .diagnosticBuffer = pending_.inspectionScheduled ? slot.diagnostic.get()
                                                         : BufferHandle{},
        .diagnosticRayAddress =
            pending_.inspectionScheduled
                ? gpu_.getBufferDeviceAddress(slot.diagnostic.get()) +
                      sizeof(DDGIDiagnosticHeaderGpuData)
                : 0u,
        .diagnosticRayCount =
            pending_.inspectionScheduled && slot.diagnosticRequest.has_value()
                ? slot.diagnosticRequest->rayCount
                : 0u,
    };
  }
  publishCapturePoints(ctx.frame);
  DDGIFrameMetrics &metrics = ctx.frame.metrics.ddgi;
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::DDGI)) {
    metrics.gpuTimeMs = ctx.frame.gpuTiming[GpuTimingScope::DDGI].timeMs;
    metrics.gpuTimingSourceFrameIndex =
        ctx.frame.gpuTiming[GpuTimingScope::DDGI].sourceFrameIndex;
    metrics.gpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::DDGITrace)) {
    metrics.traceGpuTimeMs =
        ctx.frame.gpuTiming[GpuTimingScope::DDGITrace].timeMs;
    metrics.traceGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::DDGIUpdate)) {
    metrics.updateGpuTimeMs =
        ctx.frame.gpuTiming[GpuTimingScope::DDGIUpdate].timeMs;
    metrics.updateGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming,
                        GpuTimingScope::DDGIIrradianceUpdate)) {
    metrics.irradianceUpdateGpuTimeMs =
        ctx.frame.gpuTiming[GpuTimingScope::DDGIIrradianceUpdate].timeMs;
    metrics.irradianceUpdateGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming,
                        GpuTimingScope::DDGIDistanceUpdate)) {
    metrics.distanceUpdateGpuTimeMs =
        ctx.frame.gpuTiming[GpuTimingScope::DDGIDistanceUpdate].timeMs;
    metrics.distanceUpdateGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming,
                        GpuTimingScope::DDGIRelocateClassify)) {
    metrics.relocateClassifyGpuTimeMs =
        ctx.frame.gpuTiming[GpuTimingScope::DDGIRelocateClassify].timeMs;
    metrics.relocateClassifyGpuTimingAvailable = 1u;
  }
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::DDGIReadback)) {
    metrics.readbackGpuTimeMs =
        ctx.frame.gpuTiming[GpuTimingScope::DDGIReadback].timeMs;
    metrics.readbackGpuTimingAvailable = 1u;
  }
  const RenderSettings::DDGISettings &settings = ctx.frame.settings.ddgi;
  metrics.activeVolumes = static_cast<uint32_t>(volumes_.size());
  metrics.active = metrics.activeVolumes != 0u ? 1u : 0u;
  metrics.diagnosticCountersEnabled = settings.diagnosticCounters ? 1u : 0u;
  metrics.qualitySchema = kDDGIQualitySchemaVersion;
  metrics.requestedQualityPreset =
      static_cast<uint32_t>(settings.requestedPreset);
  metrics.qualityPreset = static_cast<uint32_t>(settings.preset);
  metrics.coveragePresetSchema = kDDGICoveragePresetSchemaVersion;
  metrics.requestedCoveragePreset =
      static_cast<uint32_t>(settings.requestedCoveragePreset);
  metrics.coveragePreset = static_cast<uint32_t>(settings.coveragePreset);
  metrics.productProfileSchema = kDDGIProductProfileSchemaVersion;
  metrics.productProfileFingerprint = ddgiProductProfileFingerprint(settings);
  metrics.opaqueGatherArchitecture =
      static_cast<uint32_t>(DDGISurfaceGatherArchitecture::ForwardFragment);
  metrics.opaqueGatherVariant =
      static_cast<uint32_t>(settings.opaqueGatherVariant);
  metrics.transmissionGatherArchitecture =
      static_cast<uint32_t>(DDGISurfaceGatherArchitecture::ForwardFragment);
  metrics.transmissionGatherVariant =
      static_cast<uint32_t>(settings.transmissionGatherVariant);
  metrics.traceMultiBounceGatherArchitecture =
      static_cast<uint32_t>(DDGISurfaceGatherArchitecture::ForwardFragment);
  metrics.traceMultiBounceGatherVariant =
      static_cast<uint32_t>(settings.traceMultiBounceGatherVariant);
  if (nuri::isValid(ctx.shared[FrameTextureSlot::SceneColor].texture)) {
    const TextureDimensions dimensions = gpu_.getTextureDimensions(
        ctx.shared[FrameTextureSlot::SceneColor].texture);
    metrics.surfaceGatherWidth = dimensions.width;
    metrics.surfaceGatherHeight = dimensions.height;
  }
  metrics.surfaceGatherMaxCandidateVolumes = metrics.activeVolumes;
  metrics.surfaceGatherMaxSampledVolumes =
      std::min(metrics.activeVolumes, kMaxDDGIVolumesSampledPerSurface);
  metrics.surfaceGatherMaxStateLoadsPerPixel =
      8u * (metrics.surfaceGatherMaxCandidateVolumes +
            metrics.surfaceGatherMaxSampledVolumes);
  metrics.surfaceGatherMaxAtlasSamplesPerPixel =
      16u * metrics.surfaceGatherMaxSampledVolumes;
  for (const DDGIProbeUpdateEntry &entry : scheduledEntries_) {
    const bool classification =
        (entry.flags & kDDGIProbeUpdateClassificationGeometry) != 0u;
    const uint32_t laneCount = divRoundUp(entry.rayCount, 64u) * 64u;
    metrics.traceUsefulLanes += entry.rayCount;
    metrics.traceLaunchedLanes += laneCount;
    if (classification) {
      metrics.classificationUsefulLanes += entry.rayCount;
      metrics.classificationLaunchedLanes += laneCount;
    } else {
      ++metrics.irradianceAtlasDispatches;
      ++metrics.distanceAtlasDispatches;
      metrics.irradianceResultVisits += 64ull * entry.rayCount;
      metrics.distanceResultVisits += 256ull * entry.rayCount;
      metrics.irradianceTexelWrites += 100u;
      metrics.distanceTexelWrites += 324u;
    }
    const uint32_t reasons = (entry.flags & kDDGIProbeUpdateReasonMask) >> 8u;
    metrics.updateReasonBits |= reasons;
    metrics.bootstrapUpdates +=
        (reasons & static_cast<uint32_t>(DDGIUpdateReason::Bootstrap)) != 0u;
    metrics.scrollUpdates +=
        (reasons & static_cast<uint32_t>(DDGIUpdateReason::Scroll)) != 0u;
    metrics.dirtyGeometryUpdates +=
        (reasons & static_cast<uint32_t>(DDGIUpdateReason::DirtyGeometry)) !=
        0u;
    metrics.radiometricResponseUpdates +=
        (reasons &
         static_cast<uint32_t>(DDGIUpdateReason::RadiometricResponse)) != 0u;
    metrics.maintenanceUpdates +=
        (reasons & static_cast<uint32_t>(DDGIUpdateReason::Maintenance)) != 0u;
    metrics.forceUpdates +=
        (reasons & static_cast<uint32_t>(DDGIUpdateReason::Force)) != 0u;
    metrics.wakeUpdates +=
        (reasons & static_cast<uint32_t>(DDGIUpdateReason::Wake)) != 0u;
    metrics.reclassificationUpdates +=
        (reasons & static_cast<uint32_t>(DDGIUpdateReason::Reclassification)) !=
        0u;
  }
  metrics.traceDispatches =
      static_cast<uint32_t>(metrics.classificationUsefulLanes != 0u) +
      static_cast<uint32_t>(metrics.traceUsefulLanes !=
                            metrics.classificationUsefulLanes);
  metrics.readyVolumes = readyCount;
  metrics.failedVolumes =
      pending_.replacement ? pending_.failedVolumeCount : failedVolumeCount_;
  metrics.volumeFailureReason = pending_.replacement
                                    ? pending_.volumeFailureReason
                                    : volumeFailureReason_;
  metrics.historyReady =
      rtReady && readyCount == volumes_.size() && readyCount != 0u;
  metrics.skyFallbackActive = metrics.historyReady ? 0u : 1u;
  metrics.debugView = ctx.frame.settings.ddgi.debugView;
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
  const DDGIEffectiveVolume *sceneFitEffective = nullptr;
  for (size_t volumeIndex = 0u; volumeIndex < volumes_.size(); ++volumeIndex) {
    const DDGIEffectiveVolume &effective = frameEffectiveVolume(volumeIndex);
    if (effective.key.kind == DDGIEffectiveVolumeKind::SceneFit) {
      sceneFitEffective = &effective;
      break;
    }
  }
  for (size_t volumeIndex = 0u; volumeIndex < volumes_.size(); ++volumeIndex) {
    const VolumeResource &volume = volumes_[volumeIndex];
    const DDGIEffectiveVolume &effective = frameEffectiveVolume(volumeIndex);
    DDGIVolumeFrameMetrics &volumeMetrics = metrics.volumes[volumeIndex];
    volumeMetrics.active = 1u;
    volumeMetrics.effectiveKeyHash = ddgiEffectiveVolumeKeyHash(effective.key);
    volumeMetrics.layoutGeneration = volume.layout.generation;
    volumeMetrics.resourceGeneration = volume.resourceGeneration;
    volumeMetrics.effectiveKind = static_cast<uint32_t>(effective.key.kind);
    volumeMetrics.tier = static_cast<uint32_t>(effective.tier);
    volumeMetrics.cascadeIndex = effective.cascadeIndex;
    volumeMetrics.totalProbes =
        static_cast<uint32_t>(volume.lastSubmittedUpdates.size());
    volumeMetrics.interiorHalfExtents = effective.fadeEndHalfExtents;
    volumeMetrics.fadeStartHalfExtents = effective.fadeStartHalfExtents;
    volumeMetrics.fadeEndHalfExtents = effective.fadeEndHalfExtents;
    volumeMetrics.cameraCell = volume.layout.cameraCell;
    volumeMetrics.deficit = volume.schedulerDeficit;
    volumeMetrics.starvationFrames = volume.schedulerStarvationFrames;
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
    for (uint32_t tierIndex = 0u; tierIndex < pending_.tierSchedule.tierCount;
         ++tierIndex) {
      const DDGITierScheduleResult &tier =
          pending_.tierSchedule.tiers[tierIndex];
      if (tier.stableKey != volumeMetrics.effectiveKeyHash) {
        continue;
      }
      volumeMetrics.scheduledQuota = tier.scheduledQuota;
      volumeMetrics.usedQuota = tier.usedQuota;
      volumeMetrics.deficit = tier.pendingDeficit;
      volumeMetrics.starvationFrames = tier.pendingStarvationFrames;
      break;
    }
    if (ctx.frame.settings.ddgi.diagnosticCounters &&
        !volume.lastSubmittedUpdates.empty()) {
      ScopedScratch scoped(scratch_);
      std::pmr::vector<uint32_t> ages(scoped.resource());
      ages.reserve(volume.lastSubmittedUpdates.size());
      metrics.ageSampleCount += volume.lastSubmittedUpdates.size();
      for (const uint64_t lastUpdate : volume.lastSubmittedUpdates) {
        ages.push_back(saturatingAge(submittedSequence_, lastUpdate));
      }
      const size_t medianIndex = (ages.size() - 1u) / 2u;
      const size_t p95Index =
          std::min((ages.size() * 95u + 99u) / 100u, ages.size()) - 1u;
      volumeMetrics.updateAgeMaximum = *std::ranges::max_element(ages);
      std::ranges::nth_element(ages, ages.begin() + medianIndex);
      ++metrics.ageSelectionCount;
      volumeMetrics.updateAgeMedian = ages[medianIndex];
      std::ranges::nth_element(ages, ages.begin() + p95Index);
      ++metrics.ageSelectionCount;
      volumeMetrics.updateAgeP95 = ages[p95Index];
    }
    volumeMetrics.estimatedFullRefreshFrames =
        volumeMetrics.usedQuota == 0u
            ? 0u
            : divRoundUp(volumeMetrics.totalProbes, volumeMetrics.usedQuota);
    volumeMetrics.persistentBytes = volume.persistentBytes;
    const bool redundant =
        sceneFitEffective != nullptr &&
        analyzeDDGIVolumeRedundancy(effective, *sceneFitEffective)
            .fullyRedundant;
    volumeMetrics.redundantCoverage = redundant ? 1u : 0u;
    volumeMetrics.uniqueCoveragePercentage = redundant ? 0.0f : 100.0f;
    if (redundant) {
      ++metrics.redundantAuthoredVolumes;
      metrics.redundantAuthoredProbes += volumeMetrics.totalProbes;
      metrics.redundantAuthoredBytes += volume.persistentBytes;
    }
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
  for (const VolumeResource &volume : pending_.volumes) {
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
        volumeMetrics.historyReadyPercentage >= 100.0f;
    allHistoriesReady = allHistoriesReady && volumeHistoryReady;
    if (frameEffectiveVolume(volumeIndex).key.kind ==
        DDGIEffectiveVolumeKind::SceneFit) {
      hasSceneFit = true;
      sceneFitHistoryReady = sceneFitHistoryReady || volumeHistoryReady;
    }
  }
  const bool hybridCoverage =
      frameCoverageSettings().mode == DDGICoverageMode::Hybrid;
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
  if (ctx.frame.settings.ddgi.diagnosticCounters) {
    constexpr glm::uvec3 kLatticeDimensions{5u, 3u, 5u};
    constexpr uint32_t kMaxSampledVolumes = 2u;
    const auto spatialCoverageAndHistory =
        [](const VolumeResource &volume, const DDGIEffectiveVolume &effective,
           const glm::vec3 &worldPoint) noexcept -> glm::vec2 {
      const glm::uvec3 counts = volume.layout.probeCounts;
      const uint32_t probeCount = ddgiProbeCount(counts);
      if (!volume.ready || glm::any(glm::lessThan(counts, glm::uvec3(2u))) ||
          probeCount == 0u || volume.submittedProbeStates.size() < probeCount) {
        return glm::vec2(0.0f);
      }

      const glm::vec3 localPoint =
          glm::vec3(volume.layout.localFromWorld * glm::vec4(worldPoint, 1.0f));
      const glm::vec3 trackedCenter =
          glm::vec3(volume.layout.cameraCell) * volume.layout.probeSpacing;
      float coverage = 0.0f;
      if (effective.key.kind == DDGIEffectiveVolumeKind::ClipmapCascade) {
        const glm::vec3 fadeWidth = glm::max(effective.fadeEndHalfExtents -
                                                 effective.fadeStartHalfExtents,
                                             glm::vec3(1.0e-6f));
        const glm::vec3 axisCoverage =
            glm::vec3(1.0f) -
            glm::clamp((glm::abs(localPoint - effective.continuousCameraLocal) -
                        effective.fadeStartHalfExtents) /
                           fadeWidth,
                       glm::vec3(0.0f), glm::vec3(1.0f));
        coverage = std::min({axisCoverage.x, axisCoverage.y, axisCoverage.z});
      } else {
        const glm::vec3 distanceToFace = volume.layout.probeCenterHalfExtents -
                                         glm::abs(localPoint - trackedCenter);
        if (!glm::any(glm::lessThan(distanceToFace, glm::vec3(0.0f)))) {
          coverage =
              volume.desc.blendDistance <= 1.0e-6f
                  ? 1.0f
                  : glm::clamp(std::min({distanceToFace.x, distanceToFace.y,
                                         distanceToFace.z}) /
                                   volume.desc.blendDistance,
                               0.0f, 1.0f);
        }
      }
      if (coverage <= 0.0f) {
        return glm::vec2(0.0f);
      }

      const glm::vec3 grid =
          glm::clamp((localPoint - trackedCenter) / volume.layout.probeSpacing +
                         0.5f * glm::vec3(counts - glm::uvec3(1u)),
                     glm::vec3(0.0f), glm::vec3(counts - glm::uvec3(1u)));
      const glm::uvec3 base = glm::uvec3(
          glm::min(glm::floor(grid), glm::vec3(counts - glm::uvec3(2u))));
      const glm::vec3 fraction = grid - glm::vec3(base);
      float history = 0.0f;
      for (uint32_t neighbor = 0u; neighbor < 8u; ++neighbor) {
        const glm::uvec3 offset(neighbor & 1u, (neighbor >> 1u) & 1u,
                                (neighbor >> 2u) & 1u);
        const glm::vec3 trilinearAxis(
            offset.x != 0u ? fraction.x : 1.0f - fraction.x,
            offset.y != 0u ? fraction.y : 1.0f - fraction.y,
            offset.z != 0u ? fraction.z : 1.0f - fraction.z);
        const glm::uvec3 physical = ddgiPhysicalProbeCoordinate(
            base + offset, volume.layout.ringOrigin, counts);
        const uint32_t probe = ddgiProbeIndex(physical, counts);
        const uint32_t state =
            probe < volume.submittedProbeStates.size()
                ? volume.submittedProbeStates[probe].stateAgeFlags.x
                : static_cast<uint32_t>(DDGIProbeState::Uninitialized);
        if (state == static_cast<uint32_t>(DDGIProbeState::Awake) ||
            state == static_cast<uint32_t>(DDGIProbeState::Vigilant)) {
          history += trilinearAxis.x * trilinearAxis.y * trilinearAxis.z;
        }
      }
      return glm::vec2(coverage, glm::clamp(history, 0.0f, 1.0f));
    };
    glm::vec3 latticeHalfExtents =
        glm::max(metrics.requestedCoverageHalfExtents, glm::vec3(1.0f));
    glm::vec3 latticeCenter{0.0f};
    const DDGISceneCoverageBounds &activationBounds =
        ctx.frame.scene->ddgiActivationCoverageBounds();
    if (activationBounds.valid) {
      latticeCenter =
          0.5f * (activationBounds.bounds.min_ + activationBounds.bounds.max_);
      latticeHalfExtents = glm::max(
          0.5f * (activationBounds.bounds.max_ - activationBounds.bounds.min_),
          glm::vec3(1.0f));
    } else {
      for (size_t volumeIndex = 0u; volumeIndex < volumes_.size();
           ++volumeIndex) {
        const DDGIEffectiveVolume &effective =
            frameEffectiveVolume(volumeIndex);
        if (effective.key.kind != DDGIEffectiveVolumeKind::SceneFit) {
          continue;
        }
        latticeCenter =
            glm::vec3(volumes_[volumeIndex].layout.worldFromLocal[3]);
        latticeHalfExtents =
            glm::max(effective.fadeEndHalfExtents, glm::vec3(1.0f));
        break;
      }
    }
    for (uint32_t z = 0u; z < kLatticeDimensions.z; ++z) {
      for (uint32_t y = 0u; y < kLatticeDimensions.y; ++y) {
        for (uint32_t x = 0u; x < kLatticeDimensions.x; ++x) {
          // Sample cell centers, not the exact sealed-bounds faces where a
          // volume's intentional blend-to-sky reaches zero measure.
          const glm::vec3 unit(
              2.0f * (static_cast<float>(x) + 0.5f) /
                      static_cast<float>(kLatticeDimensions.x) -
                  1.0f,
              2.0f * (static_cast<float>(y) + 0.5f) /
                      static_cast<float>(kLatticeDimensions.y) -
                  1.0f,
              2.0f * (static_cast<float>(z) + 0.5f) /
                      static_cast<float>(kLatticeDimensions.z) -
                  1.0f);
          const glm::vec3 worldPoint =
              latticeCenter + unit * latticeHalfExtents;
          float skyRemainder = 1.0f;
          bool covered = false;
          uint32_t sampledVolumes = 0u;
          for (size_t volumeIndex = 0u; volumeIndex < volumes_.size();
               ++volumeIndex) {
            const VolumeResource &volume = volumes_[volumeIndex];
            const DDGIEffectiveVolume &effective =
                frameEffectiveVolume(volumeIndex);
            const glm::vec2 coverageAndHistory =
                spatialCoverageAndHistory(volume, effective, worldPoint);
            const float confidence = glm::clamp(
                coverageAndHistory.x * coverageAndHistory.y, 0.0f, 1.0f);
            if (confidence <= 0.0f) {
              continue;
            }
            covered = true;
            skyRemainder *= 1.0f - confidence;
            if (++sampledVolumes == kMaxSampledVolumes) {
              break;
            }
          }
          ++metrics.diagnosticSampleCount;
          ++metrics.coverageLatticeEvaluations;
          metrics.uncoveredDiagnosticSamples += covered ? 0u : 1u;
          metrics.skyRemainderSamples += skyRemainder > 0.01f ? 1u : 0u;
        }
      }
    }
    metrics.diagnosticSamplesAvailable = 1u;
  }
  const bool automaticCoverage =
      metrics.coverageMode != static_cast<uint32_t>(DDGICoverageMode::Manual);
  const bool coarseCoverageReady =
      metrics.diagnosticSamplesAvailable != 0u
          ? metrics.diagnosticSampleCount != 0u &&
                metrics.uncoveredDiagnosticSamples == 0u &&
                metrics.skyRemainderSamples == 0u
          : hasSceneFit && sceneFitHistoryReady;
  if (metrics.sceneCoverageRatio >= 1.0f && allHistoriesReady) {
    metrics.coverageStatus = DDGICoverageStatus::FullCoverageReady;
  } else if (metrics.sceneCoverageRatio >= 1.0f &&
             ((hybridCoverage && coarseCoverageReady) ||
              (!hybridCoverage && automaticCoverage))) {
    metrics.coverageStatus = DDGICoverageStatus::FullCoverageWarmingDetail;
  } else if (metrics.activeVolumes != 0u) {
    metrics.coverageStatus = DDGICoverageStatus::PartialCoverage;
  } else {
    metrics.coverageStatus = DDGICoverageStatus::SkyFallbackOnly;
  }
  metrics.skyRemainderOverThresholdPercentage =
      metrics.diagnosticSampleCount == 0u
          ? 1.0f
          : static_cast<float>(metrics.skyRemainderSamples) /
                static_cast<float>(metrics.diagnosticSampleCount);
  const bool anyIrradianceReady = std::ranges::any_of(
      std::span(metrics.volumes).first(metrics.activeVolumes),
      [](const DDGIVolumeFrameMetrics &volume) {
        return volume.initializedProbes != 0u &&
               volume.shadingEnabledProbes != 0u;
      });
  if (metrics.failedVolumes != 0u) {
    metrics.startupPhase = metrics.readyVolumes != 0u
                               ? DDGIStartupPhase::Degraded
                               : DDGIStartupPhase::Failed;
  } else if (metrics.coverageStatus == DDGICoverageStatus::FullCoverageReady) {
    metrics.startupPhase = DDGIStartupPhase::FullCoverageReady;
  } else if (metrics.coverageStatus ==
             DDGICoverageStatus::FullCoverageWarmingDetail) {
    metrics.startupPhase = DDGIStartupPhase::FullCoverageWarmingDetail;
  } else if (hybridCoverage && coarseCoverageReady) {
    metrics.startupPhase = DDGIStartupPhase::CoarseCoverageReady;
  } else if (anyIrradianceReady) {
    metrics.startupPhase = DDGIStartupPhase::FirstIrradianceReady;
  } else if (metrics.readyVolumes != 0u) {
    metrics.startupPhase = DDGIStartupPhase::ResourcesReadyNoHistory;
  } else {
    metrics.startupPhase = DDGIStartupPhase::ResourcesPending;
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
  struct PrepareCpuTimer {
    DDGIFrameMetrics &metrics;
    std::chrono::steady_clock::time_point start;
    ~PrepareCpuTimer() {
      metrics.prepareCpuTimeMs = std::chrono::duration<float, std::milli>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
    }
  } prepareCpuTimer{ctx.frame.metrics.ddgi, std::chrono::steady_clock::now()};
  ctx.frame.metrics.ddgi.deviceEpoch = deviceEpoch_;
  ctx.frame.metrics.ddgi.consumedResetEpoch = consumedResetEpoch_;
  ctx.frame.metrics.ddgi.consumedForceUpdateEpoch = consumedForceEpoch_;
  ctx.frame.metrics.ddgi.debugView = ctx.frame.settings.ddgi.debugView;
  activeFrameSlotIndex_ = std::numeric_limits<size_t>::max();
  scheduledReadbackBytes_ = 0u;
  pending_.initializationScheduled = false;
  pending_.scrollScheduled = false;
  pending_.updatesScheduled = false;
  pending_.radiometricResponseScheduled = false;
  pending_.geometryResponseScheduled = false;
  pending_.inspectionScheduled = false;
  pending_.tierSchedule = {};
  pending_.dirtyConsumptionScheduled = false;
  scheduledEntries_.clear();
  dispatchEntries_.clear();
  if (ctx.frame.ddgiProbeInspectRequest.has_value()) {
    if (ctx.frame.ddgiProbeInspectRequest->requestId >
        latestInspectionRequestId_) {
      latestInspectionResult_.reset();
    }
    latestInspectionRequestId_ =
        std::max(latestInspectionRequestId_,
                 ctx.frame.ddgiProbeInspectRequest->requestId);
  }
  const RenderSettings::DDGISettings &settings = ctx.frame.settings.ddgi;
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
  const DDGICoverageSettings &coverageSettings = settings.coverage;
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
      pending_.replacement && pending_.sources.sceneId == scene.id() &&
      pending_.coverageSettings == coverageSettings &&
      (!coverageUsesSceneBounds(coverageSettings.mode) ||
       pending_.sceneBoundsGeneration == sceneBounds.generation) &&
      pending_.volumeTopologyVersion == scene.ddgiVolumeTopologyVersion() &&
      pending_.volumeTransformVersion == scene.ddgiVolumeTransformVersion() &&
      pending_.volumeSettingsVersion == scene.ddgiVolumeSettingsVersion() &&
      pending_.relocationEnabled == settings.relocation &&
      pending_.classificationEnabled == settings.classification;
  const uint64_t desiredCoverageGeneration =
      pendingInputsMatch ? pending_.coverageGeneration
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
      std::min<uint64_t>(
          settings.maxRayQueriesPerFrame,
          static_cast<uint64_t>(settings.maxProbeUpdatesPerFrame) *
              settings.raysPerProbe) *
          sizeof(DDGIRayResultGpuData) +
      sizeof(DDGIFrameGpuData);
  const DDGICoverageSolveLimits solveLimits{
      .maxTextureExtent =
          glm::uvec2(std::max(gpu_.getDeviceCaps().maxTextureDimension2D, 1u)),
      .maxPersistentBytes = ddgiPresetPersistentBudget(
          settings.preset, config_.persistentMemoryLimitBytes),
      .maxReplacementPeakBytes = config_.peakMemoryLimitBytes,
      .retainedReplacementBytes = saturatingAdd(retainedBytes, frameBatchBytes),
      .maxProbeUpdatesPerFrame = settings.maxProbeUpdatesPerFrame};
  DDGIEffectiveVolumePlan &plan = coveragePlan_;
  const DDGICoverageResolveInput resolveInput{
      .sceneId = scene.id(),
      .coverageGeneration = desiredCoverageGeneration,
      .sceneBounds = sceneBounds,
      .authoredVolumes = scene.ddgiVolumes(),
      .cameraWorldPosition = glm::vec3(ctx.frame.camera.cameraPos),
      .settings = coverageSettings,
      .limits = solveLimits,
      .scratch = nullptr,
  };
  const auto cameraCellsMatch = [&]() {
    for (const DDGIEffectiveVolume &volume : plan.activeVolumes()) {
      if (volume.mode != DDGIVolumeMode::CameraTracked) {
        continue;
      }
      const glm::vec3 cameraLocal =
          glm::vec3(glm::inverse(volume.worldFromLocal) *
                    glm::vec4(resolveInput.cameraWorldPosition, 1.0f));
      if (ddgiCameraCell(cameraLocal, volume.probeSpacing) !=
          volume.cameraCell) {
        return false;
      }
    }
    return true;
  };
  const bool reuseCoveragePlan =
      coveragePlanValid_ && coveragePlanSceneId_ == scene.id() &&
      coveragePlanSettings_ == coverageSettings &&
      coveragePlanLimits_ == solveLimits &&
      coveragePlanBoundsGeneration_ ==
          (coverageUsesSceneBounds(coverageSettings.mode)
               ? sceneBounds.generation
               : 0u) &&
      coveragePlanVolumeTopologyVersion_ == scene.ddgiVolumeTopologyVersion() &&
      coveragePlanVolumeTransformVersion_ ==
          scene.ddgiVolumeTransformVersion() &&
      coveragePlanVolumeSettingsVersion_ == scene.ddgiVolumeSettingsVersion() &&
      plan.coverageGeneration == desiredCoverageGeneration &&
      cameraCellsMatch();
  const auto coverageResolveStart = std::chrono::steady_clock::now();
  using CoverageResolveResult = Result<bool, DDGICoverageSolveError>;
  auto resolveCoveragePlan = [&]() -> CoverageResolveResult {
    if (reuseCoveragePlan) {
      ctx.frame.metrics.ddgi.coveragePlanCacheHits = 1u;
      return coveragePlanSucceeded_
                 ? CoverageResolveResult::makeResult(true)
                 : CoverageResolveResult::makeError(plan.error);
    }
    ScopedScratch scoped(scratch_);
    DDGICoverageResolveInput uncachedInput = resolveInput;
    uncachedInput.scratch = scoped.resource();
    auto result = resolveDDGIEffectiveVolumePlan(uncachedInput, plan);
    ctx.frame.metrics.ddgi.coverageSolveExecutions = 1u;
    coveragePlanValid_ = true;
    coveragePlanSucceeded_ = !result.hasError();
    coveragePlanSceneId_ = scene.id();
    coveragePlanSettings_ = coverageSettings;
    coveragePlanLimits_ = solveLimits;
    coveragePlanBoundsGeneration_ =
        coverageUsesSceneBounds(coverageSettings.mode) ? sceneBounds.generation
                                                       : 0u;
    coveragePlanVolumeTopologyVersion_ = scene.ddgiVolumeTopologyVersion();
    coveragePlanVolumeTransformVersion_ = scene.ddgiVolumeTransformVersion();
    coveragePlanVolumeSettingsVersion_ = scene.ddgiVolumeSettingsVersion();
    return result;
  };
  auto resolved = resolveCoveragePlan();
  for (DDGIEffectiveVolume &volume :
       std::span(plan.volumes).first(plan.volumeCount)) {
    volume.continuousCameraLocal =
        glm::vec3(glm::inverse(volume.worldFromLocal) *
                  glm::vec4(resolveInput.cameraWorldPosition, 1.0f));
  }
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
    if (pending_.replacement || pending_.compatiblePlan) {
      clearPendingVolumes();
    }
    coverageMetrics.failedVolumes = static_cast<uint32_t>(
        std::min<size_t>(plan.failedKeys.size() + plan.omittedKeys.size() + 1u,
                         std::numeric_limits<uint32_t>::max()));
    coverageMetrics.coverageError = resolved.error().limit;
    coverageMetrics.limitingConstraint = resolved.error().limit;
    coverageMetrics.coverageStatus = DDGICoverageStatus::SkyFallbackOnly;
    coverageMetrics.fallbackReason =
        coverageFallbackReason(resolved.error().limit);
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
                                 compatibleWith(pending_.volumes);
  const bool runtimeModeChanged =
      !volumes_.empty() && (relocationEnabled_ != settings.relocation ||
                            classificationEnabled_ != settings.classification);
  if (!coverageSolveFailed &&
      (!activeCompatible || resetRequested || runtimeModeChanged) &&
      !pendingCompatible) {
    auto rebuild = rebuildVolumes(ctx, plan, coverageSettings,
                                  !resetRequested && !runtimeModeChanged &&
                                      sceneId_ == scene.id());
    if (rebuild.hasError()) {
      ctx.frame.metrics.ddgi.fallbackReason =
          DDGIFallbackReason::AllocationFailed;
      return Result<bool, std::string>::makeResult(true);
    }
  } else if (!coverageSolveFailed && activeCompatible && !resetRequested &&
             !runtimeModeChanged) {
    if (pending_.replacement) {
      clearPendingVolumes();
    }
    stageCompatiblePlan(plan, coverageSettings, scene);
    pending_.relocationEnabled = settings.relocation;
    pending_.classificationEnabled = settings.classification;
    pending_.scheduledFrameIndex = ctx.frame.frameIndex;
  }
  if (!pending_.replacement && !settings.freezeUpdates) {
    buildScrollPlan(ctx.frame);
  } else {
    pending_.scrollLayouts.clear();
    scrollInvalidations_.clear();
  }
  const auto readbackPollStart = std::chrono::steady_clock::now();
  collectCompletedReadbacks(ctx);
  ctx.frame.metrics.ddgi.readbackPollCpuTimeMs =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - readbackPollStart)
          .count();
  const size_t boundedLocalLightCount = std::min<size_t>(
      scene.packedLocalLights().size(), settings.maxLocalLightsPerHit);
  auto slots = ensureFrameSlots(settings, boundedLocalLightCount,
                                scrollInvalidations_.size());
  if (slots.hasError()) {
    ctx.frame.metrics.ddgi.fallbackReason =
        DDGIFallbackReason::AllocationFailed;
    return Result<bool, std::string>::makeResult(true);
  }
  FrameSlot *acquiredSlot = acquireFrameSlot(ctx.frame.frameIndex);
  if (acquiredSlot == nullptr) {
    ctx.frame.metrics.ddgi.fallbackReason =
        DDGIFallbackReason::ReadbackRingSaturated;
    ctx.frame.metrics.ddgi.startupPhase = DDGIStartupPhase::Degraded;
    publishReadbackMetrics(ctx.frame.metrics.ddgi, ctx.frame.frameIndex);
    return Result<bool, std::string>::makeResult(true);
  }
  FrameSlot &slot = *acquiredSlot;
  activeFrameSlotIndex_ =
      static_cast<size_t>(acquiredSlot - frameSlots_.data());
  slot.sceneId = scene.id();
  slot.featureGeneration =
      pending_.replacement ? pending_.coverageGeneration : coverageGeneration_;
  const auto graphBuildElapsed = [](auto start) {
    return std::chrono::duration<float, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
  };
  auto graphBuildStart = std::chrono::steady_clock::now();
  auto frameUpload = updateFrameData(ctx, slot);
  ctx.frame.metrics.ddgi.graphBuildCpuTimeMs +=
      graphBuildElapsed(graphBuildStart);
  if (frameUpload.hasError()) {
    return frameUpload;
  }
  if (pending_.replacement) {
    graphBuildStart = std::chrono::steady_clock::now();
    auto initialization = appendInitializationPass(ctx, slot);
    ctx.frame.metrics.ddgi.graphBuildCpuTimeMs +=
        graphBuildElapsed(graphBuildStart);
    if (initialization.hasError()) {
      return initialization;
    }
    const bool rtReady = ctx.shared.rayTracingScene.has_value() &&
                         ctx.shared.rayTracingScene->ready;
    publishFrameData(ctx, slot, rtReady);
    ctx.frame.metrics.ddgi.fallbackReason =
        DDGIFallbackReason::VolumeResourcesWarming;
    ctx.frame.metrics.ddgi.startupPhase = DDGIStartupPhase::ResourcesPending;
    ++ctx.frame.metrics.ddgi.resetCount;
    publishReadbackMetrics(ctx.frame.metrics.ddgi, ctx.frame.frameIndex);
    return Result<bool, std::string>::makeResult(true);
  }
  if (volumes_.empty()) {
    ctx.frame.metrics.ddgi.failedVolumes = failedVolumeCount_;
    ctx.frame.metrics.ddgi.volumeFailureReason = volumeFailureReason_;
    ctx.frame.metrics.ddgi.fallbackReason =
        coverageSolveFailed ? coverageFallbackReason(resolved.error().limit)
        : failedVolumeCount_ != 0u ? DDGIFallbackReason::AllocationFailed
                                   : DDGIFallbackReason::NoVolumes;
    ctx.frame.metrics.ddgi.startupPhase =
        coverageSolveFailed || failedVolumeCount_ != 0u
            ? DDGIStartupPhase::Failed
            : DDGIStartupPhase::ResourcesPending;
    publishReadbackMetrics(ctx.frame.metrics.ddgi, ctx.frame.frameIndex);
    return Result<bool, std::string>::makeResult(true);
  }
  if (!scrollInvalidations_.empty()) {
    graphBuildStart = std::chrono::steady_clock::now();
    auto scroll = appendScrollPass(ctx, slot);
    ctx.frame.metrics.ddgi.graphBuildCpuTimeMs +=
        graphBuildElapsed(graphBuildStart);
    if (scroll.hasError()) {
      return scroll;
    }
    ++ctx.frame.metrics.ddgi.scrollCount;
    ctx.frame.metrics.ddgi.invalidatedProbes =
        static_cast<uint32_t>(scrollInvalidations_.size());
  }
  const bool rtReady = ctx.shared.rayTracingScene.has_value() &&
                       ctx.shared.rayTracingScene->ready;
  const auto scheduleStart = std::chrono::steady_clock::now();
  if (!settings.freezeUpdates && rtReady) {
    const uint64_t currentDeformationVersion =
        ctx.shared.rayTracingScene.has_value()
            ? ctx.shared.rayTracingScene->deformationVersion
            : scene.deformationVersion();
    const uint64_t comparedTopologyVersion =
        pending_.sources.geometry ? pending_.sources.geometryTopology
                                  : sceneTopologyVersion_;
    const uint64_t comparedTransformVersion =
        pending_.sources.geometry ? pending_.sources.geometryTransform
                                  : sceneTransformVersion_;
    const uint64_t comparedDeformationVersion =
        pending_.sources.geometry ? pending_.sources.geometryDeformation
                                  : sceneDeformationVersion_;
    const bool geometryChanged =
        comparedTopologyVersion != scene.topologyVersion() ||
        comparedTransformVersion != scene.transformVersion() ||
        comparedDeformationVersion != currentDeformationVersion;
    const std::span<const DirectionalLightGpuData> directionalLights =
        scene.packedDirectionalLights();
    const std::span<const DirectionalLightGpuData> comparedDirectionalLights =
        pending_.sources.radiometric ? std::span<const DirectionalLightGpuData>(
                                           pending_.directionalLights)
                                     : std::span<const DirectionalLightGpuData>(
                                           submittedDirectionalLights_);
    const bool directionalChanged =
        directionalLights.size() != comparedDirectionalLights.size() ||
        !std::ranges::equal(directionalLights, comparedDirectionalLights,
                            samePackedValue<DirectionalLightGpuData>);
    const uint64_t comparedLightTopologyVersion =
        pending_.sources.radiometric ? pending_.sources.lightTopology
                                     : lightTopologyVersion_;
    const uint64_t comparedLightTransformVersion =
        pending_.sources.radiometric ? pending_.sources.lightTransform
                                     : lightTransformVersion_;
    const uint64_t comparedMaterialVersion = pending_.sources.radiometric
                                                 ? pending_.sources.material
                                                 : materialVersion_;
    const uint64_t comparedEnvironmentVersion =
        pending_.sources.radiometric ? pending_.sources.environment
                                     : environmentVersion_;
    const bool localLightsChanged =
        comparedLightTopologyVersion != scene.lightTopologyVersion() ||
        comparedLightTransformVersion != scene.lightTransformVersion();
    const bool globalRadiometricChanged =
        directionalChanged ||
        comparedMaterialVersion != ctx.resources.materialVersion() ||
        comparedEnvironmentVersion != scene.environmentVersion();
    if (geometryChanged || localLightsChanged || globalRadiometricChanged) {
      std::array<DDGIDirtyVolume, kMaxDDGIEffectiveVolumes> dirtyVolumes{};
      uint32_t dirtyVolumeCount = 0u;
      for (uint32_t index = 0u; index < static_cast<uint32_t>(volumes_.size());
           ++index) {
        const VolumeResource &volume = volumes_[index];
        if (!volume.ready) {
          continue;
        }
        const DDGIVolumeLayout &layout =
            pending_.scrollLayouts.size() == volumes_.size()
                ? pending_.scrollLayouts[index]
                : volume.layout;
        const float minimumSpacing =
            std::min({layout.probeSpacing.x, layout.probeSpacing.y,
                      layout.probeSpacing.z});
        dirtyVolumes[dirtyVolumeCount++] = DDGIDirtyVolume{
            .localFromWorld = layout.localFromWorld,
            .probeCounts = layout.probeCounts,
            .probeSpacing = layout.probeSpacing,
            .cameraCell = layout.cameraCell,
            .queryBias = 0.75f * minimumSpacing * settings.classificationBias,
            .tier = frameEffectiveVolume(index).tier,
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
          if (comparedTopologyVersion != scene.topologyVersion()) {
            kind = DDGISceneChangeKind::StaticTopology;
            version = scene.topologyVersion();
          } else if (comparedDeformationVersion != currentDeformationVersion) {
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
        pending_.sources.geometryTopology = scene.topologyVersion();
        pending_.sources.geometryTransform = scene.transformVersion();
        pending_.sources.geometryDeformation = currentDeformationVersion;
        pending_.sources.geometry = true;
      }
      if (localLightsChanged) {
        const std::span<const LightId> currentIds = scene.packedLocalLightIds();
        const std::span<const LocalLightGpuData> currentLights =
            scene.packedLocalLights();
        const size_t currentCount =
            std::min(currentIds.size(), currentLights.size());
        const uint64_t lightVersion = std::max(scene.lightTopologyVersion(),
                                               scene.lightTransformVersion());
        const bool countLightComparisons = settings.diagnosticCounters;
        for (size_t index = 0u; index < currentCount; ++index) {
          const auto previous = std::ranges::find_if(
              pending_.sources.radiometric ? pending_.localLights
                                           : submittedLocalLights_,
              [&](const LocalLightSnapshot &candidate) {
                ctx.frame.metrics.ddgi.lightDifferenceComparisons +=
                    countLightComparisons ? 1u : 0u;
                return candidate.id == currentIds[index];
              });
          const auto &comparedLocalLights = pending_.sources.radiometric
                                                ? pending_.localLights
                                                : submittedLocalLights_;
          if (previous != comparedLocalLights.end() &&
              samePackedValue(previous->data, currentLights[index])) {
            continue;
          }
          DDGISceneChangeRegion change = makeDDGILocalLightChangeRegion(
              previous != comparedLocalLights.end() ? &previous->data : nullptr,
              &currentLights[index], stableLightKey(currentIds[index]),
              lightVersion);
          change.submissionSequence = submittedSequence_;
          (void)dirtyRegions_.publish(change, activeDirtyVolumes);
        }
        const auto &comparedLocalLights = pending_.sources.radiometric
                                              ? pending_.localLights
                                              : submittedLocalLights_;
        for (const LocalLightSnapshot &previous : comparedLocalLights) {
          const auto current = std::ranges::find_if(
              currentIds.first(currentCount), [&](LightId candidate) {
                ctx.frame.metrics.ddgi.lightDifferenceComparisons +=
                    countLightComparisons ? 1u : 0u;
                return candidate == previous.id;
              });
          if (current != currentIds.first(currentCount).end()) {
            continue;
          }
          DDGISceneChangeRegion change = makeDDGILocalLightChangeRegion(
              &previous.data, nullptr, stableLightKey(previous.id),
              lightVersion);
          change.submissionSequence = submittedSequence_;
          (void)dirtyRegions_.publish(change, activeDirtyVolumes);
        }
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
      }
      if (localLightsChanged || globalRadiometricChanged) {
        pending_.sources.lightTopology = scene.lightTopologyVersion();
        pending_.sources.lightTransform = scene.lightTransformVersion();
        pending_.sources.material = ctx.resources.materialVersion();
        pending_.sources.environment = scene.environmentVersion();
        stagePendingRadiometricSnapshot(scene);
        pending_.sources.radiometric = true;
      }
    }
    (void)dirtyRegions_.prepareConsumption(ctx.frame.frameIndex);
    auto schedule = buildSchedule(settings);
    if (schedule.hasError()) {
      return Result<bool, std::string>::makeError(schedule.error());
    }
    ctx.frame.metrics.ddgi.scheduleCpuTimeMs =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - scheduleStart)
            .count();
    if (schedule.value().updatedProbes != 0u) {
      graphBuildStart = std::chrono::steady_clock::now();
      auto update = appendUpdatePasses(ctx, slot, schedule.value());
      ctx.frame.metrics.ddgi.graphBuildCpuTimeMs +=
          graphBuildElapsed(graphBuildStart);
      if (update.hasError()) {
        return update;
      }
    }
    ctx.frame.metrics.ddgi.updatedProbes = schedule.value().updatedProbes;
    ctx.frame.metrics.ddgi.primaryQueries = schedule.value().primaryQueries;
    ctx.frame.metrics.ddgi.classificationProbeUpdates =
        schedule.value().classificationProbeUpdates;
    ctx.frame.metrics.ddgi.classificationPrimaryQueries =
        schedule.value().classificationPrimaryQueries;
    ctx.frame.metrics.ddgi.irradiancePrimaryQueries =
        schedule.value().irradiancePrimaryQueries;
    pending_.dirtyConsumptionScheduled =
        !dirtyRegions_.pendingRegions().empty() && pending_.updatesScheduled;
  } else {
    ctx.frame.metrics.ddgi.scheduleCpuTimeMs =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - scheduleStart)
            .count();
  }
  if (pending_.dirtyConsumptionScheduled ||
      dirtyRegions_.unconsumedCount() == 0u) {
    pending_.geometryResponseScheduled = pending_.sources.geometry;
    pending_.radiometricResponseScheduled = pending_.sources.radiometric;
  }
  if (rtReady && ctx.frame.ddgiProbeInspectRequest.has_value()) {
    graphBuildStart = std::chrono::steady_clock::now();
    auto inspection = appendInspectionPass(ctx, slot);
    ctx.frame.metrics.ddgi.graphBuildCpuTimeMs +=
        graphBuildElapsed(graphBuildStart);
    if (inspection.hasError()) {
      return inspection;
    }
  }
  ctx.frame.metrics.ddgi.probeUpdateCapacity = settings.maxProbeUpdatesPerFrame;
  ctx.frame.metrics.ddgi.rayQueryCapacity = settings.maxRayQueriesPerFrame;
  ctx.frame.metrics.ddgi.primaryResultCapacity =
      static_cast<uint32_t>(slot.rayCapacity);
  ctx.frame.metrics.ddgi.requestedProbeUpdateCapacity =
      settings.maxProbeUpdatesPerFrame;
  uint32_t effectiveProbeCapacity = 0u;
  const uint32_t requestedRadianceProbeCapacity =
      std::min(settings.maxProbeUpdatesPerFrame,
               settings.maxRadianceProbeUpdatesPerFrame);
  while (effectiveProbeCapacity < requestedRadianceProbeCapacity) {
    const uint64_t primary =
        static_cast<uint64_t>(effectiveProbeCapacity + 1u) *
        settings.raysPerProbe;
    const uint64_t secondary =
        (primary * secondaryQueriesPer1024Primary_ + 1023u) / 1024u;
    if (primary + secondary > settings.maxRayQueriesPerFrame) {
      break;
    }
    ++effectiveProbeCapacity;
  }
  ctx.frame.metrics.ddgi.effectiveProbeUpdateCapacity = effectiveProbeCapacity;
  ctx.frame.metrics.ddgi.requestedMaintenanceProbeUpdateCapacity =
      settings.maxMaintenanceProbeUpdatesPerFrame;
  ctx.frame.metrics.ddgi.effectiveMaintenanceProbeUpdateCapacity =
      pending_.tierSchedule.tierCount != 0u
          ? pending_.tierSchedule.schedule.effectiveMaintenanceProbeCapacity
          : settings.maxMaintenanceProbeUpdatesPerFrame;
  ctx.frame.metrics.ddgi.maintenanceProbeUpdates =
      pending_.tierSchedule.tierCount != 0u
          ? pending_.tierSchedule.schedule.maintenanceProbeUpdates
          : 0u;
  publishFrameData(ctx, slot, rtReady);
  if (!pending_.dirtyConsumptionScheduled &&
      dirtyRegions_.hasPendingConsumption()) {
    dirtyRegions_.abandonConsumption(ctx.frame.frameIndex);
  }
  ctx.frame.metrics.ddgi.fallbackReason =
      coverageSolveFailed ? coverageFallbackReason(resolved.error().limit)
      : failedVolumeCount_ != 0u
          ? DDGIFallbackReason::AllocationFailed
          : (rtReady ? DDGIFallbackReason::None
                     : DDGIFallbackReason::RayTracingSceneWarming);
  publishReadbackMetrics(ctx.frame.metrics.ddgi, ctx.frame.frameIndex);
  return Result<bool, std::string>::makeResult(true);
}

void DDGIFeature::onFrameSubmitted(const RenderFrameContext &frame) noexcept {
  if (pending_.scheduledFrameIndex != frame.frameIndex) {
    return;
  }
  const bool submittedWork = pending_.initializationScheduled ||
                             pending_.scrollScheduled ||
                             pending_.updatesScheduled;
  const uint64_t committedSequence =
      submittedSequence_ + (submittedWork ? 1u : 0u);
  if ((pending_.updatesScheduled || pending_.inspectionScheduled) &&
      activeFrameSlotIndex_ < frameSlots_.size()) {
    FrameSlot &slot = frameSlots_[activeFrameSlotIndex_];
    slot.submission = frame.submission;
    slot.state = DDGIReadbackSlotState::Pending;
  }
  if (pending_.initializationScheduled) {
    for (size_t index = 0u; index < pending_.volumes.size(); ++index) {
      const uint32_t sourceIndex = pending_.retainedSourceIndices[index];
      if (sourceIndex == UINT32_MAX || sourceIndex >= volumes_.size()) {
        continue;
      }
      const DDGIEffectiveVolume effective = pending_.volumes[index].effective;
      const glm::vec3 requested =
          pending_.volumes[index].requestedCoverageHalfExtents;
      const glm::vec3 achieved =
          pending_.volumes[index].achievedCoverageHalfExtents;
      pending_.volumes[index] = std::move(volumes_[sourceIndex]);
      pending_.volumes[index].id = legacyVolumeId(effective);
      pending_.volumes[index].effective = effective;
      pending_.volumes[index].requestedCoverageHalfExtents = requested;
      pending_.volumes[index].achievedCoverageHalfExtents = achieved;
    }
    for (VolumeResource &volume : pending_.volumes) {
      volume.ready = volume.allocated;
    }
    volumes_.swap(pending_.volumes);
    pending_.volumes.clear();
    pending_.replacement = false;
    sceneId_ = pending_.sources.sceneId;
    volumeTopologyVersion_ = pending_.volumeTopologyVersion;
    volumeTransformVersion_ = pending_.volumeTransformVersion;
    volumeSettingsVersion_ = pending_.volumeSettingsVersion;
    coverageSettings_ = pending_.coverageSettings;
    coverageGeneration_ = pending_.coverageGeneration;
    sceneBoundsGeneration_ = pending_.sceneBoundsGeneration;
    relocationEnabled_ = pending_.relocationEnabled;
    classificationEnabled_ = pending_.classificationEnabled;
    failedVolumeCount_ = pending_.failedVolumeCount;
    volumeFailureReason_ = pending_.volumeFailureReason;
    probeStateMirrorSourceFrame_ = frame.frameIndex;
    probeStateMirrorAvailable_ = !volumes_.empty();
    consumedResetEpoch_ = std::max(consumedResetEpoch_, pending_.resetEpoch);
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
      dirtyRegions_.clear();
      clearPendingDirtySourceFacts();
    }
  } else if (pending_.compatiblePlan &&
             pending_.effectiveVolumeCount == volumes_.size()) {
    for (size_t index = 0u; index < volumes_.size(); ++index) {
      volumes_[index].effective = pending_.effectiveVolumes[index];
      volumes_[index].requestedCoverageHalfExtents =
          pending_.requestedCoverageHalfExtents[index];
      volumes_[index].achievedCoverageHalfExtents =
          pending_.achievedCoverageHalfExtents[index];
    }
    sceneId_ = pending_.sources.sceneId;
    volumeTopologyVersion_ = pending_.volumeTopologyVersion;
    volumeTransformVersion_ = pending_.volumeTransformVersion;
    volumeSettingsVersion_ = pending_.volumeSettingsVersion;
    coverageSettings_ = pending_.coverageSettings;
    coverageGeneration_ = pending_.coverageGeneration;
    sceneBoundsGeneration_ = pending_.sceneBoundsGeneration;
    relocationEnabled_ = pending_.relocationEnabled;
    classificationEnabled_ = pending_.classificationEnabled;
    pending_.compatiblePlan = false;
    pending_.effectiveVolumeCount = 0u;
  }
  if (pending_.scrollScheduled) {
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
        const VolumeResource &volume = volumes_[entry.volumeStableId];
        const bool classifyVolume =
            classificationEnabled_ &&
            volume.effective.key.kind == DDGIEffectiveVolumeKind::Authored;
        state.stateAgeFlags.x = static_cast<uint32_t>(
            classifyVolume ? DDGIProbeState::Uninitialized
                           : DDGIProbeState::NewlyVigilant);
      }
    }
    if (pending_.scrollLayouts.size() == volumes_.size()) {
      for (size_t index = 0u; index < volumes_.size(); ++index) {
        volumes_[index].layout = pending_.scrollLayouts[index];
      }
    }
    probeStateMirrorSourceFrame_ = frame.frameIndex;
    probeStateMirrorAvailable_ = true;
  }
  if (pending_.updatesScheduled) {
    for (const DDGIProbeUpdateEntry &entry : scheduledEntries_) {
      if (entry.volumeStableId < volumes_.size() &&
          entry.probeId <
              volumes_[entry.volumeStableId].lastSubmittedUpdates.size()) {
        volumes_[entry.volumeStableId].lastSubmittedUpdates[entry.probeId] =
            committedSequence;
      }
    }
    for (uint32_t tierIndex = 0u; tierIndex < pending_.tierSchedule.tierCount;
         ++tierIndex) {
      const DDGITierScheduleResult &tier =
          pending_.tierSchedule.tiers[tierIndex];
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
    consumedForceEpoch_ = std::max(consumedForceEpoch_, pending_.forceEpoch);
    if (activeFrameSlotIndex_ < frameSlots_.size()) {
      FrameSlot &slot = frameSlots_[activeFrameSlotIndex_];
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
  if (pending_.geometryResponseScheduled) {
    sceneTopologyVersion_ = pending_.sources.geometryTopology;
    sceneTransformVersion_ = pending_.sources.geometryTransform;
    sceneDeformationVersion_ = pending_.sources.geometryDeformation;
    pending_.sources.geometryTopology = UINT64_MAX;
    pending_.sources.geometryTransform = UINT64_MAX;
    pending_.sources.geometryDeformation = UINT64_MAX;
    pending_.sources.geometry = false;
  }
  if (pending_.radiometricResponseScheduled) {
    lightTopologyVersion_ = pending_.sources.lightTopology;
    lightTransformVersion_ = pending_.sources.lightTransform;
    materialVersion_ = pending_.sources.material;
    environmentVersion_ = pending_.sources.environment;
    submittedLocalLights_.assign(pending_.localLights.begin(),
                                 pending_.localLights.end());
    submittedDirectionalLights_.assign(pending_.directionalLights.begin(),
                                       pending_.directionalLights.end());
    pending_.sources.lightTopology = UINT64_MAX;
    pending_.sources.lightTransform = UINT64_MAX;
    pending_.sources.material = UINT64_MAX;
    pending_.sources.environment = UINT64_MAX;
    pending_.localLights.clear();
    pending_.directionalLights.clear();
    pending_.sources.radiometric = false;
  }
  if (pending_.inspectionScheduled &&
      activeFrameSlotIndex_ < frameSlots_.size()) {
    FrameSlot &slot = frameSlots_[activeFrameSlotIndex_];
    slot.diagnosticValid = slot.diagnosticRequest.has_value();
  }
  if (!pending_.updatesScheduled && !pending_.inspectionScheduled &&
      activeFrameSlotIndex_ < frameSlots_.size()) {
    frameSlots_[activeFrameSlotIndex_].state = DDGIReadbackSlotState::Consumed;
  }
  if (pending_.dirtyConsumptionScheduled) {
    (void)dirtyRegions_.commitConsumption(frame.frameIndex);
  }
  submittedSequence_ = committedSequence;
  pending_.initializationScheduled = false;
  pending_.scrollScheduled = false;
  pending_.updatesScheduled = false;
  pending_.radiometricResponseScheduled = false;
  pending_.geometryResponseScheduled = false;
  pending_.inspectionScheduled = false;
  pending_.scheduledFrameIndex = UINT64_MAX;
  scrollInvalidations_.clear();
  pending_.scrollLayouts.clear();
  pending_.tierSchedule = {};
  pending_.dirtyConsumptionScheduled = false;
  activeFrameSlotIndex_ = std::numeric_limits<size_t>::max();
}

void DDGIFeature::onFrameAbandoned(const RenderFrameContext &frame) noexcept {
  if (pending_.scheduledFrameIndex != frame.frameIndex) {
    return;
  }
  if (activeFrameSlotIndex_ < frameSlots_.size()) {
    FrameSlot &slot = frameSlots_[activeFrameSlotIndex_];
    slot.traceCountersValid = false;
    slot.probeStateResultsValid = false;
    slot.probeStateResultCount = 0u;
    if (pending_.inspectionScheduled) {
      slot.diagnosticValid = false;
      slot.diagnosticRequest.reset();
    }
    slot.state = DDGIReadbackSlotState::Dropped;
  }
  pending_.initializationScheduled = false;
  pending_.scrollScheduled = false;
  pending_.updatesScheduled = false;
  pending_.radiometricResponseScheduled = false;
  pending_.geometryResponseScheduled = false;
  pending_.inspectionScheduled = false;
  pending_.scheduledFrameIndex = UINT64_MAX;
  scrollInvalidations_.clear();
  pending_.scrollLayouts.clear();
  pending_.tierSchedule = {};
  if (dirtyRegions_.hasPendingConsumption()) {
    dirtyRegions_.abandonConsumption(frame.frameIndex);
  }
  pending_.dirtyConsumptionScheduled = false;
  clearPendingVolumes();
  activeFrameSlotIndex_ = std::numeric_limits<size_t>::max();
}

} // namespace nuri
