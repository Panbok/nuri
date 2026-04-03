#include "nuri/pch.h"

#include "nuri/sim/simulation_scheduler.h"

#include "nuri/scene_runtime/scene_runtime_host.h"
#include "nuri/sim/backends/simulation_backend.h"
#include "nuri/sim/simulation_execution_context.h"
#include "nuri/sim/simulation_registry.h"

#include <algorithm>
#include <ranges>

namespace nuri {
namespace {

[[nodiscard]] bool
isSimulationRunnable(const SimulationRegistry::Record &record) noexcept {
  return record.enabled && !record.faulted &&
         (record.state == SimulationState::Running ||
          record.singleStepRequested);
}

[[nodiscard]] bool simulationLess(const SimulationRegistry &registry,
                                  SimulationHandle lhs,
                                  SimulationHandle rhs) noexcept {
  const auto *lhsRecord = registry.tryGet(lhs);
  const auto *rhsRecord = registry.tryGet(rhs);
  if (lhsRecord == nullptr || rhsRecord == nullptr) {
    return lhs.value < rhs.value;
  }
  if (lhsRecord->priority != rhsRecord->priority) {
    return lhsRecord->priority < rhsRecord->priority;
  }
  return lhsRecord->creationOrder < rhsRecord->creationOrder;
}

} // namespace

void SimulationScheduler::reset() noexcept { clock_.reset(); }

void SimulationScheduler::setConfig(
    const SimulationSchedulerConfig &config) noexcept {
  config_ = config;
}

SimulationTickResult
SimulationScheduler::tick(SceneRuntimeHost &host,
                          const SimulationTickInput &input) {
  SimulationTickResult result{};
  if (host.scene() == nullptr) {
    return result;
  }

  const FixedStepAdvanceResult advance =
      clock_.advance(input.frameDeltaSeconds, config_.fixedDeltaSeconds,
                     config_.maxStepsPerFrame, config_.maxAccumulatedSeconds,
                     config_.allowFrameDropping);
  result.executedSteps = advance.stepCount;
  result.consumedSeconds = advance.consumedSeconds;
  result.remainingAccumulatorSeconds = advance.remainingAccumulatorSeconds;
  result.clamped = advance.clamped;
  if (advance.stepCount == 0u) {
    return result;
  }

  std::pmr::vector<SimulationHandle> active(host.memoryResource());
  active.reserve(host.registry().liveCount());
  host.registry().forEachLive(
      [&](SimulationHandle handle, const SimulationRegistry::Record &record) {
        if (isSimulationRunnable(record)) {
          active.push_back(handle);
        }
      });

  std::ranges::sort(active, [&](SimulationHandle lhs, SimulationHandle rhs) {
    return simulationLess(host.registry(), lhs, rhs);
  });

  for (uint32_t stepIndex = 0; stepIndex < advance.stepCount; ++stepIndex) {
    const double stepTime =
        input.absoluteTimeSeconds -
        (advance.stepCount - static_cast<double>(stepIndex + 1u)) *
            config_.fixedDeltaSeconds;
    for (const SimulationHandle handle : active) {
      SimulationRegistry::Record *record = host.registry().tryGet(handle);
      if (record == nullptr || !isSimulationRunnable(*record)) {
        continue;
      }

      ISimulationBackend &backend = host.backendFor(*record);
      SimulationExecutionContext context{};
      context.fixedDeltaSeconds = config_.fixedDeltaSeconds;
      context.absoluteStepTimeSeconds = stepTime;
      context.frameIndex = input.frameIndex;
      context.stepIndex = stepIndex;
      context.schedulerConfig = &config_;
      context.bindings = &host.bindings();
      context.gpuContext = &host.gpuContext();
      context.writebacks = &host.writebacks();

      auto phaseResult = backend.executePhase(
          host, handle, SimulationPhase::PreSceneWrite, context);
      if (phaseResult.hasError()) {
        host.faultSimulation(handle, phaseResult.error());
        continue;
      }
      (void)host.registry().notePhaseExecution(
          handle, SimulationPhase::PreSceneWrite, input.frameIndex);

      const uint32_t substepCount = std::max(1u, record->substepCount);
      const uint32_t solverIterations =
          std::max(1u, record->solverIterationCount);
      for (uint32_t substepIndex = 0; substepIndex < substepCount;
           ++substepIndex) {
        context.substepIndex = substepIndex;
        context.effectiveDeltaSeconds =
            (config_.fixedDeltaSeconds * record->timeScale) / substepCount;

        phaseResult = backend.executePhase(host, handle,
                                           SimulationPhase::Predict, context);
        if (phaseResult.hasError()) {
          host.faultSimulation(handle, phaseResult.error());
          break;
        }
        result.anySimulationRan = true;
        (void)host.registry().notePhaseExecution(
            handle, SimulationPhase::Predict, input.frameIndex);

        for (uint32_t iteration = 0; iteration < solverIterations;
             ++iteration) {
          phaseResult = backend.executePhase(host, handle,
                                             SimulationPhase::Project, context);
          if (phaseResult.hasError()) {
            host.faultSimulation(handle, phaseResult.error());
            break;
          }
          result.anySimulationRan = true;
          (void)host.registry().notePhaseExecution(
              handle, SimulationPhase::Project, input.frameIndex);
        }
        if (record->faulted) {
          break;
        }

        phaseResult = backend.executePhase(host, handle,
                                           SimulationPhase::Finalize, context);
        if (phaseResult.hasError()) {
          host.faultSimulation(handle, phaseResult.error());
          break;
        }
        result.anySimulationRan = true;
        (void)host.registry().notePhaseExecution(
            handle, SimulationPhase::Finalize, input.frameIndex);
      }

      record = host.registry().tryGet(handle);
      if (record == nullptr || record->faulted) {
        continue;
      }
      phaseResult = backend.executePhase(
          host, handle, SimulationPhase::PostSceneWrite, context);
      if (phaseResult.hasError()) {
        host.faultSimulation(handle, phaseResult.error());
        continue;
      }
      (void)host.registry().notePhaseExecution(
          handle, SimulationPhase::PostSceneWrite, input.frameIndex);

      if (record->singleStepRequested) {
        (void)host.registry().clearSingleStepRequest(handle);
        if (record->state != SimulationState::Stopped) {
          record->state = SimulationState::Paused;
          record->stats.paused = true;
        }
      }
    }
  }

  return result;
}

} // namespace nuri
