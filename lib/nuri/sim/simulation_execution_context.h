#pragma once
#include "nuri/defines.h"
#include <cstdint>
namespace nuri {

class SceneRuntimeBindings;
class SimulationGpuContext;
struct SimulationSchedulerConfig;

struct NURI_API SimulationExecutionContext {
  double fixedDeltaSeconds = 0.0;
  double effectiveDeltaSeconds = 0.0;
  double absoluteStepTimeSeconds = 0.0;
  uint64_t frameIndex = 0u;
  uint32_t stepIndex = 0u;
  uint32_t substepIndex = 0u;
  const SimulationSchedulerConfig *schedulerConfig = nullptr;
  const SceneRuntimeBindings *bindings = nullptr;
  SimulationGpuContext *gpuContext = nullptr;
};

} // namespace nuri
