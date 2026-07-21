#pragma once

#include "nuri/core/pmr_scratch.h"
#include "nuri/core/runtime_config.h"
#include "nuri/gfx/ddgi/ddgi_atlas.h"
#include "nuri/gfx/ddgi/ddgi_scheduler.h"
#include "nuri/gfx/owned_gpu_resource.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include <array>
#include <memory>
#include <memory_resource>
#include <optional>
#include <string>
#include <vector>

namespace nuri {

class GPUDevice;

class NURI_API DDGIFeature final {
public:
  DDGIFeature(
      GPUDevice &gpu, RuntimeDDGIShaderConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~DDGIFeature();
  DDGIFeature(const DDGIFeature &) = delete;
  DDGIFeature &operator=(const DDGIFeature &) = delete;
  [[nodiscard]] Result<bool, std::string> prepare(FrameBuildContext &ctx);
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept;

private:
  enum PipelineIndex : size_t {
    Trace = 0u,
    TraceInspect,
    BlendIrradiance,
    BlendDistance,
    UpdateProbeState,
    PipelineCount,
  };
  struct VolumeResource {
    explicit VolumeResource(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : lastSubmittedUpdates(memory) {}
    DDGIVolumeId id = kInvalidDDGIVolumeId;
    DDGIVolumeLayout layout{};
    DDGIVolumeDesc desc{};
    OwnedTextureHandle irradiance{};
    OwnedTextureHandle distance{};
    OwnedBufferHandle probeState{};
    std::pmr::vector<uint64_t> lastSubmittedUpdates;
    uint64_t persistentBytes = 0u;
    uint64_t resourceGeneration = 0u;
    uint32_t irradianceResponseRemaining = 0u;
    uint32_t distanceResponseRemaining = 0u;
    bool ready = false;
  };
  struct FrameSlot {
    OwnedBufferHandle frameData{};
    OwnedBufferHandle updates{};
    OwnedBufferHandle invalidations{};
    OwnedBufferHandle rayResults{};
    OwnedBufferHandle localLights{};
    OwnedBufferHandle diagnostic{};
    size_t updateCapacity = 0u;
    size_t invalidationCapacity = 0u;
    size_t rayCapacity = 0u;
    size_t localLightCapacity = 0u;
    bool traceCountersValid = false;
    std::optional<DDGIProbeInspectRequest> diagnosticRequest{};
    bool diagnosticValid = false;
  };
  struct TracePushConstants {
    uint64_t frame = 0u;
    uint64_t updates = 0u;
    uint64_t results = 0u;
    uint64_t instances = 0u;
    uint64_t geometries = 0u;
    uint64_t materials = 0u;
    glm::vec4 directionalDirectionIlluminance{0.0f};
    glm::vec4 directionalColor{0.0f};
    uint64_t localLights = 0u;
    uint32_t rayCount = 0u;
    uint32_t raysPerProbe = 0u;
    uint32_t skyTextureId = kInvalidTextureBindlessIndex;
    uint32_t skySamplerId = 0u;
    uint32_t maxCandidates = 0u;
    uint32_t frameSeed = 0u;
    uint32_t materialSamplerId = 0u;
    uint32_t directionalLightCount = 0u;
    uint32_t localLightCount = 0u;
    uint32_t maxLocalLights = 0u;
  };
  static_assert(sizeof(TracePushConstants) == 128u);
  struct BlendPushConstants {
    uint64_t frame = 0u;
    uint64_t updates = 0u;
    uint64_t results = 0u;
    uint32_t updateCount = 0u;
    uint32_t volumeSlot = 0u;
    uint32_t outputTextureId = kInvalidTextureBindlessIndex;
    uint32_t raysPerProbe = 0u;
    float hysteresis = 0.0f;
    uint32_t clearMode = 0u;
    uint32_t historyValid = 0u;
    uint32_t frameSeed = 0u;
  };
  static_assert(sizeof(BlendPushConstants) == 56u);
  struct ProbeStatePushConstants {
    uint64_t updates = 0u;
    uint64_t results = 0u;
    uint64_t frame = 0u;
    std::array<uint64_t, kMaxDDGIVolumes> states{};
    uint32_t updateCount = 0u;
    uint32_t submittedSequence = 0u;
    uint32_t clearMode = 0u;
    uint32_t raysPerProbe = 0u;
    uint32_t frameSeed = 0u;
    uint32_t relocationEnabled = 0u;
    uint32_t classificationEnabled = 0u;
    uint32_t reserved = 0u;
  };
  static_assert(sizeof(ProbeStatePushConstants) == 88u);
  struct InspectPushConstants {
    uint64_t frame = 0u;
    uint64_t header = 0u;
    uint64_t rays = 0u;
    uint64_t events = 0u;
    uint64_t instances = 0u;
    uint64_t geometries = 0u;
    uint64_t materials = 0u;
    uint64_t requestId = 0u;
    uint64_t sceneId = 0u;
    uint64_t layoutGeneration = 0u;
    uint64_t resourceGeneration = 0u;
    uint64_t deviceEpoch = 0u;
    uint32_t volumeValue = 0u;
    uint32_t volumeSlot = 0u;
    uint32_t probeId = 0u;
    uint32_t rayCount = 0u;
    uint32_t maxCandidates = 0u;
    uint32_t frameSeed = 0u;
    uint32_t materialSamplerId = 0u;
    uint32_t submissionSequence = 0u;
  };
  static_assert(sizeof(InspectPushConstants) == 128u);

  [[nodiscard]] Result<bool, std::string> initialize();
  [[nodiscard]] Result<bool, std::string>
  rebuildVolumes(FrameBuildContext &ctx);
  [[nodiscard]] Result<bool, std::string>
  ensureFrameSlots(const RenderSettings::DDGISettings &settings,
                   size_t localLightCount, size_t invalidationCapacity);
  void buildScrollPlan(const RenderFrameContext &frame);
  [[nodiscard]] Result<bool, std::string>
  updateFrameData(FrameBuildContext &ctx, FrameSlot &slot);
  [[nodiscard]] Result<bool, std::string>
  appendInitializationPass(FrameBuildContext &ctx, FrameSlot &slot);
  [[nodiscard]] Result<bool, std::string>
  appendScrollPass(FrameBuildContext &ctx, FrameSlot &slot);
  [[nodiscard]] Result<bool, std::string>
  appendUpdatePasses(FrameBuildContext &ctx, FrameSlot &slot,
                     const DDGIScheduleResult &schedule);
  [[nodiscard]] Result<bool, std::string>
  appendInspectionPass(FrameBuildContext &ctx, FrameSlot &slot);
  [[nodiscard]] Result<DDGIScheduleResult, std::string>
  buildSchedule(const RenderSettings::DDGISettings &settings);
  void publishFrameData(FrameBuildContext &ctx, FrameSlot &slot, bool rtReady);
  void collectCompletedTraceMetrics(FrameSlot &slot, DDGIFrameMetrics &metrics);
  void collectCompletedInspection(FrameBuildContext &ctx, FrameSlot &slot);
  void collectDebugProbeStateMetrics(FrameBuildContext &ctx);
  void publishCapturePoints(RenderFrameContext &frame) const;
  void clearVolumes() noexcept;
  void clearPendingVolumes() noexcept;
  void clearFrameSlots() noexcept;

  GPUDevice &gpu_;
  RuntimeDDGIShaderConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  ScratchArena scratch_;
  std::array<std::unique_ptr<Shader>, PipelineCount> shaders_{};
  std::array<OwnedComputePipelineHandle, PipelineCount> pipelines_{};
  OwnedRayQueryBinding traceBinding_{};
  OwnedRayQueryBinding inspectBinding_{};
  AccelerationStructureHandle boundTlas_{};
  AccelerationStructureHandle inspectBoundTlas_{};
  OwnedSamplerHandle sampler_{};
  std::string initializationError_{};
  std::pmr::vector<VolumeResource> volumes_;
  std::pmr::vector<VolumeResource> pendingVolumes_;
  std::pmr::vector<FrameSlot> frameSlots_;
  std::pmr::vector<DDGIProbeUpdateEntry> scheduledEntries_;
  std::pmr::vector<DDGIProbeUpdateEntry> scrollInvalidations_;
  std::pmr::vector<DDGIVolumeLayout> pendingScrollLayouts_;
  std::pmr::vector<ComputeDispatchItem> dispatches_;
  std::pmr::vector<BlendPushConstants> blendPushConstants_;
  std::pmr::vector<BufferHandle> dependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode> dependencyBufferModes_;
  std::pmr::vector<TextureHandle> dependencyTextures_;
  std::pmr::vector<RenderGraphAccessMode> dependencyTextureModes_;
  std::pmr::vector<BufferHandle> forwardDependencyBuffers_;
  std::pmr::vector<TextureHandle> forwardDependencyTextures_;
  DDGIFrameGpuData frameData_{};
  TracePushConstants tracePushConstants_{};
  ProbeStatePushConstants statePushConstants_{};
  InspectPushConstants inspectPushConstants_{};
  std::optional<DDGIProbeInspectResult> latestInspectionResult_{};
  uint64_t sceneId_ = 0u;
  uint64_t volumeTopologyVersion_ = UINT64_MAX;
  uint64_t volumeTransformVersion_ = UINT64_MAX;
  uint64_t volumeSettingsVersion_ = UINT64_MAX;
  uint64_t sceneTopologyVersion_ = UINT64_MAX;
  uint64_t sceneTransformVersion_ = UINT64_MAX;
  uint64_t sceneDeformationVersion_ = UINT64_MAX;
  uint64_t lightTopologyVersion_ = UINT64_MAX;
  uint64_t lightTransformVersion_ = UINT64_MAX;
  uint64_t materialVersion_ = UINT64_MAX;
  uint64_t environmentVersion_ = UINT64_MAX;
  uint64_t pendingSceneId_ = 0u;
  uint64_t pendingVolumeTopologyVersion_ = UINT64_MAX;
  uint64_t pendingVolumeTransformVersion_ = UINT64_MAX;
  uint64_t pendingVolumeSettingsVersion_ = UINT64_MAX;
  uint64_t submittedSequence_ = 0u;
  uint64_t nextResourceGeneration_ = 0u;
  uint64_t deviceEpoch_ = 0u;
  uint64_t latestInspectionRequestId_ = 0u;
  uint64_t consumedResetEpoch_ = 0u;
  uint64_t consumedForceEpoch_ = 0u;
  uint64_t pendingResetEpoch_ = 0u;
  uint64_t pendingForceEpoch_ = 0u;
  uint64_t scheduledFrameIndex_ = UINT64_MAX;
  uint32_t failedVolumeCount_ = 0u;
  uint32_t pendingFailedVolumeCount_ = 0u;
  DDGIVolumeFailureReason volumeFailureReason_ = DDGIVolumeFailureReason::None;
  DDGIVolumeFailureReason pendingVolumeFailureReason_ =
      DDGIVolumeFailureReason::None;
  bool relocationEnabled_ = true;
  bool classificationEnabled_ = true;
  bool pendingRelocationEnabled_ = true;
  bool pendingClassificationEnabled_ = true;
  bool initialized_ = false;
  bool replacementPending_ = false;
  bool initializationScheduled_ = false;
  bool scrollScheduled_ = false;
  bool updatesScheduled_ = false;
  bool radiometricResponseScheduled_ = false;
  bool geometryResponseScheduled_ = false;
  bool inspectionScheduled_ = false;
};

} // namespace nuri
