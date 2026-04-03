#pragma once

#include "nuri/defines.h"
#include "nuri/sim/fixed_step_clock.h"

#include <cstdint>

namespace nuri {

class SceneRuntimeHost;

struct NURI_API SimulationSchedulerConfig {
  double fixedDeltaSeconds = 1.0 / 60.0;
  uint32_t maxStepsPerFrame = 4u;
  double maxAccumulatedSeconds = 0.25;
  bool allowFrameDropping = true;
};

struct NURI_API SimulationTickInput {
  double frameDeltaSeconds = 0.0;
  double absoluteTimeSeconds = 0.0;
  uint64_t frameIndex = 0u;
};

struct NURI_API SimulationTickResult {
  uint32_t executedSteps = 0u;
  double consumedSeconds = 0.0;
  double remainingAccumulatorSeconds = 0.0;
  bool clamped = false;
  bool anySimulationRan = false;
};

class NURI_API SimulationScheduler {
public:
  void reset() noexcept;
  void setConfig(const SimulationSchedulerConfig &config) noexcept;
  [[nodiscard]] const SimulationSchedulerConfig &config() const noexcept {
    return config_;
  }

  [[nodiscard]] SimulationTickResult tick(SceneRuntimeHost &host,
                                          const SimulationTickInput &input);

private:
  SimulationSchedulerConfig config_{};
  FixedStepClock clock_{};
};

} // namespace nuri
