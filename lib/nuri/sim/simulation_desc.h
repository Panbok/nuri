#pragma once

#include "nuri/defines.h"
#include "nuri/sim/simulation_bindings.h"
#include "nuri/sim/simulation_handles.h"

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <string>

namespace nuri {

struct NURI_API SimulationDesc {
  explicit SimulationDesc(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : debugName(memory), binding(memory) {}

  SimulationKind kind = SimulationKind::Unknown;
  std::pmr::string debugName;
  SimulationBindingDesc binding;
  SimulationBackendPreference backendPreference =
      SimulationBackendPreference::Auto;
  float timeScale = 1.0f;
  uint32_t priority = 0u;
  // Must be >= 1.
  uint32_t substepCount = 1u;
  // Must be >= 1.
  uint32_t solverIterationCount = 1u;
  bool enabled = true;
  bool startPaused = false;
  bool allowGpuExecution = true;
  std::span<const std::byte> initialParams{};
};

} // namespace nuri
