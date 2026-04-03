#include "nuri/pch.h"

#include "nuri/sim/simulation_controller.h"

#include "nuri/scene_runtime/scene_runtime_host.h"

namespace nuri {

Result<SimulationHandle, std::string>
SimulationController::createSimulation(const SimulationDesc &desc) {
  if (owner_ == nullptr) {
    return Result<SimulationHandle, std::string>::makeError(
        "SimulationController::createSimulation: owner is null");
  }
  return owner_->createSimulationInternal(desc);
}

bool SimulationController::destroySimulation(SimulationHandle handle) {
  return owner_ != nullptr && owner_->destroySimulationInternal(handle);
}

bool SimulationController::setEnabled(SimulationHandle handle, bool enabled) {
  return owner_ != nullptr &&
         owner_->setSimulationEnabledInternal(handle, enabled);
}

bool SimulationController::pause(SimulationHandle handle) {
  return owner_ != nullptr && owner_->pauseSimulationInternal(handle);
}

bool SimulationController::resume(SimulationHandle handle) {
  return owner_ != nullptr && owner_->resumeSimulationInternal(handle);
}

bool SimulationController::requestSingleStep(SimulationHandle handle) {
  return owner_ != nullptr &&
         owner_->requestSimulationSingleStepInternal(handle);
}

bool SimulationController::setTimeScale(SimulationHandle handle,
                                        float timeScale) {
  return owner_ != nullptr &&
         owner_->setSimulationTimeScaleInternal(handle, timeScale);
}

bool SimulationController::setSubstepCount(SimulationHandle handle,
                                           uint32_t count) {
  return owner_ != nullptr &&
         owner_->setSimulationSubstepCountInternal(handle, count);
}

bool SimulationController::setSolverIterationCount(SimulationHandle handle,
                                                   uint32_t count) {
  return owner_ != nullptr &&
         owner_->setSimulationSolverIterationCountInternal(handle, count);
}

bool SimulationController::setParams(SimulationHandle handle,
                                     std::span<const std::byte> params) {
  return owner_ != nullptr &&
         owner_->setSimulationParamsInternal(handle, params);
}

bool SimulationController::getState(SimulationHandle handle,
                                    SimulationState &out) const {
  return owner_ != nullptr && owner_->getSimulationStateInternal(handle, out);
}

bool SimulationController::getDesc(SimulationHandle handle,
                                   SimulationDesc &out) const {
  return owner_ != nullptr && owner_->getSimulationDescInternal(handle, out);
}

bool SimulationController::getStats(SimulationHandle handle,
                                    SimulationStats &out) const {
  return owner_ != nullptr && owner_->getSimulationStatsInternal(handle, out);
}

} // namespace nuri
