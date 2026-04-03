#pragma once

#include "nuri/defines.h"

#include <cstdint>

namespace nuri {

class SceneRuntimeBindings;
class SimulationGpuContext;
class SimulationWritebackState;
struct SimulationSchedulerConfig;

// Carries per-step timing/indexing plus borrowed scheduler, binding, GPU, and
// writeback context for a single simulation phase call. Pointer members are
// non-owning observers managed by SceneRuntimeHost and must outlive this
// SimulationExecutionContext instance.
struct NURI_API SimulationExecutionContext {
  // Nominal scheduler timestep for the current advance.
  double fixedDeltaSeconds = 0.0;
  // Effective delta applied by the simulation after substep/time-scale logic.
  double effectiveDeltaSeconds = 0.0;
  // Absolute world time for the current fixed step.
  double absoluteStepTimeSeconds = 0.0;
  // Frame that scheduled this execution.
  uint64_t frameIndex = 0u;
  // Fixed-step index within the frame.
  uint32_t stepIndex = 0u;
  // Substep index within the current fixed step.
  uint32_t substepIndex = 0u;
  // Borrowed scheduler config used for this tick.
  const SimulationSchedulerConfig *schedulerConfig = nullptr;
  // Borrowed runtime binding tables for scene objects.
  const SceneRuntimeBindings *bindings = nullptr;
  // Borrowed GPU helper context.
  SimulationGpuContext *gpuContext = nullptr;
  // Borrowed writeback queues for scene mutations.
  SimulationWritebackState *writebacks = nullptr;
};

} // namespace nuri
