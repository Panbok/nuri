#include "nuri/pch.h"

#include "nuri/scene_runtime/scene_runtime_host.h"

#include "nuri/core/profiling.h"
#include "nuri/sim/backends/animation_pose_simulation_backend.h"
#include "nuri/sim/backends/simulation_backend.h"

namespace nuri {
namespace {

[[nodiscard]] bool hasBindingTableData(const SceneRuntimeBindings &bindings) {
  return !bindings.nodes().empty() || !bindings.renderables().empty();
}

} // namespace

SceneRuntimeHost::SceneRuntimeHost(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      registry_(memory_), bindings_(memory_), writebacks_(memory_),
      animationPoseBackend_(
          std::make_unique<AnimationPoseSimulationBackend>(memory_)),
      controller_(this) {}

SceneRuntimeHost::~SceneRuntimeHost() = default;

SimulationTickResult SceneRuntimeHost::tick(const SimulationTickInput &input) {
  NURI_PROFILER_FUNCTION();
  if (scene_ == nullptr) {
    lastTickResult_ = {};
    return lastTickResult_;
  }

  refreshSceneBindingsIfNeeded();
  if (registry_.liveCount() == 0u) {
    lastTickResult_ = {};
    return lastTickResult_;
  }

  gpuContext_.beginFrame(input.frameIndex);
  SimulationTickResult result = scheduler_.tick(*this, input);
  lastTickResult_ = result;
  flushWritebacks();
  return result;
}

void SceneRuntimeHost::bindScene(RenderScene *scene) {
  if (scene_ == scene) {
    return;
  }
  const bool hadBindings = hasBindingTableData(bindings_);
  scene_ = scene;
  bindings_.clear();
  topologyVersion_ = scene_ != nullptr ? scene_->graph().topologyVersion() : 0u;
  transformVersion_ =
      scene_ != nullptr ? scene_->graph().transformVersion() : 0u;
  const bool bindingsChanged =
      hadBindings || (scene_ != nullptr && bindings_.rebuild(scene_));
  if (bindingsChanged) {
    noteBindingMutation();
  }
  if (scene_ != nullptr) {
    validateAllSimulationBindings();
  }
}

void SceneRuntimeHost::attachAnimationGpuServices(
    AnimationGpuServices *services) noexcept {
  if (animationPoseBackend_) {
    animationPoseBackend_->attachGpuServices(services);
  }
}

Result<bool, std::string>
SceneRuntimeHost::prepareAnimationSceneFrame(uint64_t frameIndex) {
  if (animationPoseBackend_ == nullptr) {
    return Result<bool, std::string>::makeResult(true);
  }
  return animationPoseBackend_->prepareSceneFrame(*this, frameIndex);
}

void SceneRuntimeHost::commitAnimationSceneFrame(uint64_t frameIndex) noexcept {
  if (animationPoseBackend_) {
    animationPoseBackend_->commitSceneFrame(frameIndex);
  }
}

void SceneRuntimeHost::abandonAnimationSceneFrame(
    uint64_t frameIndex) noexcept {
  if (animationPoseBackend_) {
    animationPoseBackend_->abandonSceneFrame(frameIndex);
  }
}

Result<SimulationHandle, std::string>
SceneRuntimeHost::createAnimationPoseSimulation(
    const AnimationPoseSimulationCreateInfo &createInfo) {
  auto descResult = makeAnimationPoseSimulationDesc(createInfo, memory_);
  if (descResult.hasError()) {
    return Result<SimulationHandle, std::string>::makeError(descResult.error());
  }
  SimulationDesc desc = std::move(descResult.value());
  pendingAnimationPoseCreatePayload_ = AnimationPoseSimulationCreatePayload{
      .prefab = createInfo.prefab,
      .instantiationMap = createInfo.instantiationMap,
      .controlledPrefabNodeIndices = createInfo.controlledPrefabNodeIndices,
      .params = createInfo.params,
  };
  struct PendingPayloadResetGuard {
    std::optional<AnimationPoseSimulationCreatePayload> &payload;

    ~PendingPayloadResetGuard() { payload.reset(); }
  } guard{pendingAnimationPoseCreatePayload_};
  return controller_.createSimulation(desc);
}

bool SceneRuntimeHost::destroyAnimationPoseSimulation(
    SimulationHandle handle) noexcept {
  return controller_.destroySimulation(handle);
}

const AnimationSceneFrameData *
SceneRuntimeHost::animationSceneFrameData() const noexcept {
  return animationPoseBackend_ != nullptr
             ? animationPoseBackend_->currentSceneFrameData()
             : nullptr;
}

void SceneRuntimeHost::reset() {
  registry_.clear();
  bindings_.clear();
  gpuContext_.reset();
  writebacks_.clear();
  scheduler_.reset();
  if (animationPoseBackend_) {
    animationPoseBackend_->reset();
  }
  simulationVersion_ = 0u;
  bindingVersion_ = 0u;
  deformationVersion_ = 0u;
  topologyVersion_ = scene_ != nullptr ? scene_->graph().topologyVersion() : 0u;
  transformVersion_ =
      scene_ != nullptr ? scene_->graph().transformVersion() : 0u;
  simulationControlVersion_ = 0u;
  lastTickResult_ = {};
}

Result<SimulationHandle, std::string>
SceneRuntimeHost::createSimulationInternal(const SimulationDesc &desc) {
  auto handleResult = registry_.create(desc);
  if (handleResult.hasError()) {
    return handleResult;
  }
  const SimulationHandle handle = handleResult.value();
  SimulationRegistry::Record *record = registry_.tryGet(handle);
  if (record == nullptr) {
    return Result<SimulationHandle, std::string>::makeError(
        "SceneRuntimeHost::createSimulationInternal: registry returned null "
        "record");
  }

  SimulationDesc backendDesc(memory_);
  (void)registry_.getDesc(handle, backendDesc);
  auto backendResult =
      backendFor(*record).createInstance(*this, handle, backendDesc);
  if (backendResult.hasError()) {
    (void)registry_.destroy(handle);
    return Result<SimulationHandle, std::string>::makeError(
        backendResult.error());
  }

  if (scene_ != nullptr && !validateBindingDesc(record->binding)) {
    faultSimulation(handle, "Simulation binding target is invalid");
  }
  noteSimulationMutation();
  return Result<SimulationHandle, std::string>::makeResult(handle);
}

bool SceneRuntimeHost::destroySimulationInternal(SimulationHandle handle) {
  SimulationRegistry::Record *record = registry_.tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  ISimulationBackend &backend = backendFor(*record);
  if (!registry_.destroy(handle)) {
    return false;
  }
  auto destroyResult = backend.destroyInstance(*this, handle);
  if (destroyResult.hasError()) {
    NURI_LOG_WARNING(
        "SceneRuntimeHost::destroySimulationInternal: backend cleanup failed "
        "after registry removal for simulation #%u: %s",
        handle.value, destroyResult.error().c_str());
  } else if (!destroyResult.value()) {
    NURI_LOG_WARNING(
        "SceneRuntimeHost::destroySimulationInternal: backend cleanup failed "
        "after registry removal for simulation #%u",
        handle.value);
  }
  noteSimulationMutation();
  return true;
}

bool SceneRuntimeHost::setSimulationEnabledInternal(SimulationHandle handle,
                                                    bool enabled) {
  if (!registry_.setEnabled(handle, enabled)) {
    return false;
  }
  noteSimulationMutation();
  return true;
}

bool SceneRuntimeHost::pauseSimulationInternal(SimulationHandle handle) {
  if (!registry_.pause(handle)) {
    return false;
  }
  noteSimulationMutation();
  return true;
}

bool SceneRuntimeHost::resumeSimulationInternal(SimulationHandle handle) {
  if (!registry_.resume(handle)) {
    return false;
  }
  noteSimulationMutation();
  return true;
}

bool SceneRuntimeHost::requestSimulationSingleStepInternal(
    SimulationHandle handle) {
  const SimulationRegistry::Record *record = registry_.tryGet(handle);
  const bool alreadyRequested =
      record != nullptr && record->singleStepRequested;
  if (!registry_.requestSingleStep(handle)) {
    return false;
  }
  if (!alreadyRequested) {
    noteSimulationMutation();
  }
  return true;
}

bool SceneRuntimeHost::setSimulationTimeScaleInternal(SimulationHandle handle,
                                                      float timeScale) {
  if (!registry_.setTimeScale(handle, timeScale)) {
    return false;
  }
  noteSimulationMutation();
  return true;
}

bool SceneRuntimeHost::setSimulationSubstepCountInternal(
    SimulationHandle handle, uint32_t count) {
  if (!registry_.setSubstepCount(handle, count)) {
    return false;
  }
  noteSimulationMutation();
  return true;
}

bool SceneRuntimeHost::setSimulationSolverIterationCountInternal(
    SimulationHandle handle, uint32_t count) {
  if (!registry_.setSolverIterationCount(handle, count)) {
    return false;
  }
  noteSimulationMutation();
  return true;
}

bool SceneRuntimeHost::setSimulationParamsInternal(
    SimulationHandle handle, std::span<const std::byte> params) {
  SimulationRegistry::Record *record = registry_.tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  if (!registry_.setParams(handle, params)) {
    return false;
  }
  auto updateResult = backendFor(*record).updateParams(*this, handle, params);
  if (updateResult.hasError()) {
    faultSimulation(handle, updateResult.error());
    return false;
  }
  noteSimulationMutation();
  return true;
}

bool SceneRuntimeHost::getSimulationStateInternal(SimulationHandle handle,
                                                  SimulationState &out) const {
  return registry_.getState(handle, out);
}

bool SceneRuntimeHost::getSimulationDescInternal(SimulationHandle handle,
                                                 SimulationDesc &out) const {
  return registry_.getDesc(handle, out);
}

bool SceneRuntimeHost::getSimulationStatsInternal(SimulationHandle handle,
                                                  SimulationStats &out) const {
  return registry_.getStats(handle, out);
}

ISimulationBackend &SceneRuntimeHost::backendFor(
    const SimulationRegistry::Record &record) noexcept {
  if (record.kind == SimulationKind::AnimationPose &&
      animationPoseBackend_ != nullptr) {
    return *animationPoseBackend_;
  }
  return nullBackend_;
}

bool SceneRuntimeHost::validateBindingTarget(
    SimulationBindingTarget &target) noexcept {
  if (scene_ == nullptr) {
    target.runtimeBindingIndex = kInvalidSimulationBindingIndex;
    return true;
  }

  switch (target.type) {
  case SimulationBindingTargetType::None:
    target.runtimeBindingIndex = kInvalidSimulationBindingIndex;
    return true;
  case SimulationBindingTargetType::Node:
    target.runtimeBindingIndex = bindings_.runtimeNodeIndex(target.node);
    return target.runtimeBindingIndex != kInvalidSimulationBindingIndex;
  case SimulationBindingTargetType::Renderable:
    target.runtimeBindingIndex =
        bindings_.runtimeRenderableIndex(target.renderable);
    return target.runtimeBindingIndex != kInvalidSimulationBindingIndex;
  case SimulationBindingTargetType::PrefabRoot:
    target.runtimeBindingIndex = bindings_.runtimeNodeIndex(target.prefabRoot);
    return target.runtimeBindingIndex != kInvalidSimulationBindingIndex;
  default:
    target.runtimeBindingIndex = kInvalidSimulationBindingIndex;
    return false;
  }
}

bool SceneRuntimeHost::validateBindingDesc(
    SimulationBindingDesc &binding) noexcept {
  if (!validateBindingTarget(binding.primaryTarget)) {
    return false;
  }
  for (SimulationBindingTarget &target : binding.secondaryTargets) {
    if (!validateBindingTarget(target)) {
      return false;
    }
  }
  for (SimulationAttachmentBinding &attachment : binding.attachmentSlots) {
    if (!validateBindingTarget(attachment.target)) {
      return false;
    }
  }
  return true;
}

void SceneRuntimeHost::validateAllSimulationBindings() {
  registry_.forEachLive([this](SimulationHandle handle,
                               SimulationRegistry::Record &record) {
    if (record.faulted || scene_ == nullptr ||
        !hasSimulationBindingFlag(record.binding.flags,
                                  SimulationBindingFlags::ValidateTargets)) {
      return;
    }
    if (!validateBindingDesc(record.binding)) {
      if (hasSimulationBindingFlag(
              record.binding.flags,
              SimulationBindingFlags::FaultOnInvalidTarget)) {
        faultSimulation(handle, "Simulation binding target is invalid");
      }
    }
  });
}

void SceneRuntimeHost::faultSimulation(SimulationHandle handle,
                                       std::string_view reason) {
  if (registry_.markFaulted(handle, reason)) {
    NURI_LOG_WARNING(
        "SceneRuntimeHost::faultSimulation: simulation #%u faulted: %s",
        handle.value, std::string(reason).c_str());
    noteSimulationMutation();
  }
}

void SceneRuntimeHost::noteSimulationMutation() noexcept {
  ++simulationVersion_;
  ++simulationControlVersion_;
}

void SceneRuntimeHost::noteBindingMutation() noexcept { ++bindingVersion_; }

void SceneRuntimeHost::refreshSceneBindingsIfNeeded() {
  NURI_ASSERT(scene_ != nullptr,
              "SceneRuntimeHost::refreshSceneBindingsIfNeeded: scene_ must be "
              "initialized before calling refreshSceneBindingsIfNeeded");
  const uint64_t currentTopologyVersion = scene_->graph().topologyVersion();
  const uint64_t currentTransformVersion = scene_->graph().transformVersion();
  if (currentTopologyVersion == topologyVersion_) {
    transformVersion_ = currentTransformVersion;
    return;
  }

  topologyVersion_ = currentTopologyVersion;
  transformVersion_ = currentTransformVersion;
  if (bindings_.rebuild(scene_)) {
    noteBindingMutation();
    validateAllSimulationBindings();
  }
}

void SceneRuntimeHost::flushWritebacks() {
  if (writebacks_.empty()) {
    return;
  }

  ++simulationVersion_;

  if (!writebacks_.nodeLocalTransforms().empty()) {
    ++transformVersion_;
  }
  if (!writebacks_.renderableDeformations().empty()) {
    ++deformationVersion_;
  }
  writebacks_.clear();
}

} // namespace nuri
