#include "nuri/pch.h"

#include "nuri/sim/simulation_registry.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nuri {
namespace {

void assignBindingTargetRuntimeIndices(SimulationBindingDesc &binding) {
  binding.primaryTarget.runtimeBindingIndex = kInvalidSimulationBindingIndex;
  for (SimulationBindingTarget &target : binding.secondaryTargets) {
    target.runtimeBindingIndex = kInvalidSimulationBindingIndex;
  }
  for (SimulationAttachmentBinding &attachment : binding.attachmentSlots) {
    attachment.target.runtimeBindingIndex = kInvalidSimulationBindingIndex;
  }
}

void copyBindingDesc(SimulationBindingDesc &dst,
                     const SimulationBindingDesc &src) {
  dst.primaryTarget = src.primaryTarget;
  dst.secondaryTargets.assign(src.secondaryTargets.begin(),
                              src.secondaryTargets.end());
  dst.debugName.assign(src.debugName.data(), src.debugName.size());
  dst.flags = src.flags;
  dst.attachmentSlots.assign(src.attachmentSlots.begin(),
                             src.attachmentSlots.end());
}

void initializeStats(SimulationRegistry::Record &record) {
  record.stats.creationOrder = record.creationOrder;
  record.stats.controlVersion = 1u;
  record.stats.executedStepCount = 0u;
  record.stats.executedSubstepCount = 0u;
  record.stats.phaseExecutionCounts.fill(0u);
  record.stats.lastExecutedFrameIndex = std::numeric_limits<uint64_t>::max();
  record.stats.paramsSizeBytes = record.params.size();
  record.stats.enabled = record.enabled;
  record.stats.paused = record.state == SimulationState::Paused;
  record.stats.faulted = false;
  record.stats.gpuEligible = record.allowGpuExecution;
  record.stats.lastFaultReason.clear();
}

void noteControlMutation(SimulationRegistry::Record &record) {
  ++record.stats.controlVersion;
  record.stats.enabled = record.enabled;
  record.stats.paused = record.state == SimulationState::Paused;
  record.stats.gpuEligible = record.allowGpuExecution;
}

} // namespace

SimulationRegistry::SimulationRegistry(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      slots_(memory_), records_(memory_) {}

void SimulationRegistry::clear() {
  slots_.clear();
  records_.clear();
  nextCreationOrder_ = 1u;
}

Result<bool, std::string>
SimulationRegistry::validateCreateDesc(const SimulationDesc &desc) {
  if (desc.timeScale < 0.0f || !std::isfinite(desc.timeScale)) {
    return Result<bool, std::string>::makeError(
        "SimulationRegistry::create: timeScale must be finite and >= 0");
  }
  if (desc.substepCount == 0u) {
    return Result<bool, std::string>::makeError(
        "SimulationRegistry::create: substepCount must be greater than zero");
  }
  if (desc.solverIterationCount == 0u) {
    return Result<bool, std::string>::makeError(
        "SimulationRegistry::create: solverIterationCount must be greater than "
        "zero");
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<SimulationHandle, std::string>
SimulationRegistry::create(const SimulationDesc &desc) {
  auto validation = validateCreateDesc(desc);
  if (validation.hasError()) {
    return Result<SimulationHandle, std::string>::makeError(validation.error());
  }
  if (slots_.slotCount() > kResourceHandleIndexMask ||
      (slots_.slotCount() == kResourceHandleIndexMask + 1u &&
       slots_.liveCount() == slots_.slotCount())) {
    return Result<SimulationHandle, std::string>::makeError(
        "SimulationRegistry::create: slot pool exhausted");
  }

  const SlotReservation slot = slots_.acquire();
  if (slot.appended) {
    records_.emplace_back(memory_);
  }

  Record &record = records_[slot.index];
  record.kind = desc.kind;
  record.backendPreference = desc.backendPreference;
  record.state = !desc.enabled ? SimulationState::Stopped
                               : (desc.startPaused ? SimulationState::Paused
                                                   : SimulationState::Running);
  record.enabled = desc.enabled;
  record.allowGpuExecution = desc.allowGpuExecution;
  record.singleStepRequested = false;
  record.faulted = false;
  record.timeScale = desc.timeScale;
  record.priority = desc.priority;
  record.substepCount = desc.substepCount;
  record.solverIterationCount = desc.solverIterationCount;
  record.creationOrder = nextCreationOrder_++;
  record.debugName.assign(desc.debugName.data(), desc.debugName.size());
  copyBindingDesc(record.binding, desc.binding);
  assignBindingTargetRuntimeIndices(record.binding);
  record.params.assign(desc.initialParams.begin(), desc.initialParams.end());
  record.faultReason.clear();
  initializeStats(record);

  return Result<SimulationHandle, std::string>::makeResult(
      makeSimulationHandle(slot.index, slot.generation));
}

bool SimulationRegistry::destroy(SimulationHandle handle) {
  if (!slotValid(handle)) {
    return false;
  }
  slots_.release(indexOf(handle));
  records_[indexOf(handle)] = Record(memory_);
  return true;
}

SimulationRegistry::Record *
SimulationRegistry::tryGet(SimulationHandle handle) noexcept {
  if (!slotValid(handle)) {
    return nullptr;
  }
  return &records_[indexOf(handle)];
}

const SimulationRegistry::Record *
SimulationRegistry::tryGet(SimulationHandle handle) const noexcept {
  if (!slotValid(handle)) {
    return nullptr;
  }
  return &records_[indexOf(handle)];
}

bool SimulationRegistry::slotValid(SimulationHandle handle) const noexcept {
  return isValid(handle) &&
         slots_.isValid(indexOf(handle), generationOf(handle));
}

bool SimulationRegistry::setEnabled(SimulationHandle handle, bool enabled) {
  Record *record = tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  if (record->enabled == enabled &&
      ((enabled && record->state != SimulationState::Stopped) ||
       (!enabled && record->state == SimulationState::Stopped))) {
    return true;
  }
  record->enabled = enabled;
  if (!enabled) {
    record->state = SimulationState::Stopped;
  } else if (!record->faulted) {
    record->state = SimulationState::Running;
  }
  noteControlMutation(*record);
  return true;
}

bool SimulationRegistry::pause(SimulationHandle handle) {
  Record *record = tryGet(handle);
  if (record == nullptr || !record->enabled || record->faulted) {
    return false;
  }
  if (record->state == SimulationState::Paused) {
    return true;
  }
  record->state = SimulationState::Paused;
  noteControlMutation(*record);
  return true;
}

bool SimulationRegistry::resume(SimulationHandle handle) {
  Record *record = tryGet(handle);
  if (record == nullptr || !record->enabled || record->faulted) {
    return false;
  }
  if (record->state == SimulationState::Running) {
    return true;
  }
  record->state = SimulationState::Running;
  noteControlMutation(*record);
  return true;
}

bool SimulationRegistry::requestSingleStep(SimulationHandle handle) {
  Record *record = tryGet(handle);
  if (record == nullptr || !record->enabled || record->faulted) {
    return false;
  }
  record->singleStepRequested = true;
  noteControlMutation(*record);
  return true;
}

bool SimulationRegistry::setTimeScale(SimulationHandle handle,
                                      float timeScale) {
  Record *record = tryGet(handle);
  if (record == nullptr || timeScale < 0.0f || !std::isfinite(timeScale)) {
    return false;
  }
  if (record->timeScale == timeScale) {
    return true;
  }
  record->timeScale = timeScale;
  noteControlMutation(*record);
  return true;
}

bool SimulationRegistry::setSubstepCount(SimulationHandle handle,
                                         uint32_t count) {
  Record *record = tryGet(handle);
  if (record == nullptr || count == 0u) {
    return false;
  }
  if (record->substepCount == count) {
    return true;
  }
  record->substepCount = count;
  noteControlMutation(*record);
  return true;
}

bool SimulationRegistry::setSolverIterationCount(SimulationHandle handle,
                                                 uint32_t count) {
  Record *record = tryGet(handle);
  if (record == nullptr || count == 0u) {
    return false;
  }
  if (record->solverIterationCount == count) {
    return true;
  }
  record->solverIterationCount = count;
  noteControlMutation(*record);
  return true;
}

bool SimulationRegistry::setParams(SimulationHandle handle,
                                   std::span<const std::byte> params) {
  Record *record = tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  if (record->params.size() == params.size() &&
      std::equal(record->params.begin(), record->params.end(), params.begin(),
                 params.end())) {
    return true;
  }
  record->params.assign(params.begin(), params.end());
  noteControlMutation(*record);
  record->stats.paramsSizeBytes = record->params.size();
  return true;
}

bool SimulationRegistry::getState(SimulationHandle handle,
                                  SimulationState &out) const noexcept {
  const Record *record = tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  out = record->state;
  return true;
}

bool SimulationRegistry::getDesc(SimulationHandle handle,
                                 SimulationDesc &out) const {
  const Record *record = tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  out.kind = record->kind;
  out.debugName.assign(record->debugName.data(), record->debugName.size());
  copyBindingDesc(out.binding, record->binding);
  out.backendPreference = record->backendPreference;
  out.timeScale = record->timeScale;
  out.priority = record->priority;
  out.substepCount = record->substepCount;
  out.solverIterationCount = record->solverIterationCount;
  out.enabled = record->enabled;
  out.startPaused = record->state == SimulationState::Paused;
  out.allowGpuExecution = record->allowGpuExecution;
  out.initialParams =
      std::span<const std::byte>(record->params.data(), record->params.size());
  return true;
}

bool SimulationRegistry::getStats(SimulationHandle handle,
                                  SimulationStats &out) const {
  const Record *record = tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  out = record->stats;
  return true;
}

bool SimulationRegistry::markFaulted(SimulationHandle handle,
                                     std::string_view reason) {
  Record *record = tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  const bool changed = !record->faulted ||
                       std::string_view(record->faultReason) != reason ||
                       record->state != SimulationState::Stopped;
  record->faulted = true;
  record->state = SimulationState::Stopped;
  record->singleStepRequested = false;
  record->faultReason.assign(reason.data(), reason.size());
  record->stats.faulted = true;
  record->stats.lastFaultReason.assign(reason.data(), reason.size());
  if (changed) {
    noteControlMutation(*record);
  }
  return changed;
}

bool SimulationRegistry::clearSingleStepRequest(SimulationHandle handle) {
  Record *record = tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  if (!record->singleStepRequested) {
    return true;
  }
  record->singleStepRequested = false;
  noteControlMutation(*record);
  return true;
}

bool SimulationRegistry::notePhaseExecution(SimulationHandle handle,
                                            SimulationPhase phase,
                                            uint64_t frameIndex) {
  Record *record = tryGet(handle);
  if (record == nullptr) {
    return false;
  }
  const size_t phaseIndex = static_cast<size_t>(phase);
  if (phaseIndex >= record->stats.phaseExecutionCounts.size()) {
    return false;
  }
  ++record->stats.phaseExecutionCounts[phaseIndex];
  record->stats.lastExecutedFrameIndex = frameIndex;
  if (phase == SimulationPhase::Finalize) {
    ++record->stats.executedSubstepCount;
  } else if (phase == SimulationPhase::PostSceneWrite) {
    ++record->stats.executedStepCount;
  }
  return true;
}

} // namespace nuri
