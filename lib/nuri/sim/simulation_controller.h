#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/sim/simulation_desc.h"
#include "nuri/sim/simulation_stats.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace nuri {

class SceneRuntimeHost;

class NURI_API SimulationController {
public:
  SimulationController() = delete;
  SimulationController(const SimulationController &) = delete;
  SimulationController &operator=(const SimulationController &) = delete;
  SimulationController(SimulationController &&) = delete;
  SimulationController &operator=(SimulationController &&) = delete;

  [[nodiscard]] Result<SimulationHandle, std::string>
  createSimulation(const SimulationDesc &desc);
  [[nodiscard]] bool destroySimulation(SimulationHandle handle);
  [[nodiscard]] bool setEnabled(SimulationHandle handle, bool enabled);
  [[nodiscard]] bool pause(SimulationHandle handle);
  [[nodiscard]] bool resume(SimulationHandle handle);
  [[nodiscard]] bool requestSingleStep(SimulationHandle handle);
  [[nodiscard]] bool setTimeScale(SimulationHandle handle, float timeScale);
  [[nodiscard]] bool setSubstepCount(SimulationHandle handle, uint32_t count);
  [[nodiscard]] bool setSolverIterationCount(SimulationHandle handle,
                                             uint32_t count);
  [[nodiscard]] bool setParams(SimulationHandle handle,
                               std::span<const std::byte> params);
  [[nodiscard]] bool getState(SimulationHandle handle,
                              SimulationState &out) const;
  [[nodiscard]] bool getDesc(SimulationHandle handle,
                             SimulationDesc &out) const;
  [[nodiscard]] bool getStats(SimulationHandle handle,
                              SimulationStats &out) const;

private:
  friend class SceneRuntimeHost;
  explicit SimulationController(SceneRuntimeHost *owner) : owner_(owner) {}

  SceneRuntimeHost *owner_ = nullptr;
};

} // namespace nuri
