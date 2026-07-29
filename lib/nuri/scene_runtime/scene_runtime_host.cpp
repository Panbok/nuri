#include "nuri/scene_runtime/scene_runtime_host.h"
#include "nuri/core/profiling.h"
#include "nuri/sim/backends/animation_pose_simulation_backend.h"
namespace nuri {
namespace {
[[nodiscard]] bool hasBindingTableData(const SceneRuntimeBindings &bindings) {
  return !bindings.nodes().empty() || !bindings.renderables().empty();
}
template <typename T>
std::shared_ptr<const T>
shareImmutable(const T *source,
               std::unordered_map<const T *, std::weak_ptr<const T>> &owners) {
  if (auto existing = owners[source].lock()) {
    return existing;
  }
  auto owned = std::make_shared<const T>(*source);
  owners[source] = owned;
  return owned;
}
} // namespace

SceneRuntimeHost::SceneRuntimeHost(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      registry_(memory_), bindings_(memory_),
      animationPoseBackend_(
          std::make_unique<AnimationPoseSimulationBackend>(memory_)) {}

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
  SimulationTickResult result = scheduler_.tick(*this, input);
  lastTickResult_ = result;
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
  const bool rebuilt = scene_ != nullptr && bindings_.rebuild(scene_);
  const bool bindingsChanged = hadBindings || rebuilt;
  if (bindingsChanged) {
    noteBindingMutation();
  }
  if (scene_ != nullptr) {
    validateAllSimulationBindings();
  }
}

void SceneRuntimeHost::attachAnimationGpuServices(
    AnimationGpuServices *services) noexcept {
  animationPoseBackend_->attachGpuServices(services);
}

Result<bool, std::string>
SceneRuntimeHost::prepareAnimationSceneFrame(uint64_t frameIndex) {
  return animationPoseBackend_->prepareSceneFrame(*this, frameIndex);
}

void SceneRuntimeHost::commitAnimationSceneFrame(uint64_t frameIndex) noexcept {
  animationPoseBackend_->commitSceneFrame(frameIndex);
}

void SceneRuntimeHost::abandonAnimationSceneFrame(
    uint64_t frameIndex) noexcept {
  animationPoseBackend_->abandonSceneFrame(frameIndex);
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
      .prefab = shareImmutable(createInfo.prefab, animationPrefabOwners_),
      .instantiationMap = shareImmutable(createInfo.instantiationMap,
                                         animationInstantiationOwners_),
      .controlledPrefabNodeIndices = createInfo.controlledPrefabNodeIndices,
      .params = createInfo.params,
  };
  struct PendingPayloadResetGuard {
    std::optional<AnimationPoseSimulationCreatePayload> &payload;
    ~PendingPayloadResetGuard() { payload.reset(); }
  } guard{pendingAnimationPoseCreatePayload_};
  return createSimulation(desc);
}

bool SceneRuntimeHost::destroyAnimationPoseSimulation(
    SimulationHandle handle) noexcept {
  return destroySimulation(handle);
}

const AnimationSceneFrameData *
SceneRuntimeHost::animationSceneFrameData() const noexcept {
  return animationPoseBackend_->currentSceneFrameData();
}

void SceneRuntimeHost::reset() {
  registry_.clear();
  bindings_.clear();
  scheduler_.reset();
  animationPoseBackend_->reset();
  animationPrefabOwners_.clear();
  animationInstantiationOwners_.clear();
  bindingVersion_ = 0u;
  topologyVersion_ = scene_ != nullptr ? scene_->graph().topologyVersion() : 0u;
  lastTickResult_ = {};
}

Result<SimulationHandle, std::string>
SceneRuntimeHost::createSimulation(const SimulationDesc &desc) {
  auto handleResult = registry_.create(desc);
  if (handleResult.hasError()) {
    return handleResult;
  }
  const SimulationHandle handle = handleResult.value();
  SimulationRegistry::Record &record = *registry_.tryGet(handle);
  if (record.kind == SimulationKind::AnimationPose) {
    SimulationDesc backendDesc(memory_);
    (void)registry_.getDesc(handle, backendDesc);
    auto backendResult =
        animationPoseBackend_->createInstance(*this, handle, backendDesc);
    if (backendResult.hasError()) {
      (void)registry_.destroy(handle);
      return Result<SimulationHandle, std::string>::makeError(
          backendResult.error());
    }
  }
  if (scene_ != nullptr && !validateBindingDesc(record.binding)) {
    faultSimulation(handle, "Simulation binding target is invalid");
  }
  return Result<SimulationHandle, std::string>::makeResult(handle);
}

bool SceneRuntimeHost::destroySimulation(SimulationHandle handle) {
  SimulationRegistry::Record *record = registry_.tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  const SimulationKind kind = record->kind;
  if (!registry_.destroy(handle)) {
    return false;
  }
  auto destroyResult =
      kind == SimulationKind::AnimationPose
          ? animationPoseBackend_->destroyInstance(*this, handle)
          : Result<bool, std::string>::makeResult(true);
  if (destroyResult.hasError() || !destroyResult.value()) {
    NURI_LOG_WARNING(
        "SceneRuntimeHost::destroySimulation: backend cleanup failed "
        "after registry removal for simulation #%u",
        handle.value);
  }
  return true;
}

bool SceneRuntimeHost::setEnabled(SimulationHandle handle, bool enabled) {
  if (!registry_.setEnabled(handle, enabled)) {
    return false;
  }
  return true;
}

bool SceneRuntimeHost::pause(SimulationHandle handle) {
  if (!registry_.pause(handle)) {
    return false;
  }
  return true;
}

bool SceneRuntimeHost::resume(SimulationHandle handle) {
  if (!registry_.resume(handle)) {
    return false;
  }
  return true;
}

bool SceneRuntimeHost::requestSingleStep(SimulationHandle handle) {
  const SimulationRegistry::Record *record = registry_.tryGet(handle);
  const bool alreadyRequested =
      record != nullptr && record->singleStepRequested;
  if (!registry_.requestSingleStep(handle)) {
    return false;
  }
  if (!alreadyRequested) {
  }
  return true;
}

bool SceneRuntimeHost::setTimeScale(SimulationHandle handle, float timeScale) {
  if (!registry_.setTimeScale(handle, timeScale)) {
    return false;
  }
  return true;
}

bool SceneRuntimeHost::setSubstepCount(SimulationHandle handle,
                                       uint32_t count) {
  if (!registry_.setSubstepCount(handle, count)) {
    return false;
  }
  return true;
}

bool SceneRuntimeHost::setSolverIterationCount(SimulationHandle handle,
                                               uint32_t count) {
  if (!registry_.setSolverIterationCount(handle, count)) {
    return false;
  }
  return true;
}

bool SceneRuntimeHost::setParams(SimulationHandle handle,
                                 std::span<const std::byte> params) {
  SimulationRegistry::Record *record = registry_.tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  if (!registry_.setParams(handle, params)) {
    return false;
  }
  auto updateResult =
      record->kind == SimulationKind::AnimationPose
          ? animationPoseBackend_->updateParams(*this, handle, params)
          : Result<bool, std::string>::makeResult(true);
  if (updateResult.hasError()) {
    faultSimulation(handle, updateResult.error());
    return false;
  }
  return true;
}

bool SceneRuntimeHost::getState(SimulationHandle handle,
                                SimulationState &out) const {
  return registry_.getState(handle, out);
}

bool SceneRuntimeHost::getDesc(SimulationHandle handle,
                               SimulationDesc &out) const {
  return registry_.getDesc(handle, out);
}

bool SceneRuntimeHost::getStats(SimulationHandle handle,
                                SimulationStats &out) const {
  return registry_.getStats(handle, out);
}

Result<bool, std::string> SceneRuntimeHost::executeSimulationPhase(
    SimulationKind kind, SimulationHandle handle, SimulationPhase phase,
    const SimulationExecutionContext &context) {
  return kind == SimulationKind::AnimationPose
             ? animationPoseBackend_->executePhase(*this, handle, phase,
                                                   context)
             : Result<bool, std::string>::makeResult(true);
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
    if (record.stats.faulted || scene_ == nullptr ||
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
  }
}

void SceneRuntimeHost::noteBindingMutation() noexcept { ++bindingVersion_; }

void SceneRuntimeHost::refreshSceneBindingsIfNeeded() {
  const uint64_t currentTopologyVersion = scene_->graph().topologyVersion();
  if (currentTopologyVersion == topologyVersion_) {
    return;
  }
  topologyVersion_ = currentTopologyVersion;
  if (bindings_.rebuild(scene_)) {
    noteBindingMutation();
    validateAllSimulationBindings();
  }
}

} // namespace nuri
