#pragma once

#include "nuri/defines.h"
#include "nuri/sim/simulation_handles.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <string>

namespace nuri {

struct NURI_API SimulationStats {
  // Copy construction keeps the default std::pmr copy behavior for
  // lastFaultReason. Construct the destination with an explicit memory_resource
  // before assignment if allocator locality matters.
  explicit SimulationStats(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : lastFaultReason(memory) {}

  uint64_t creationOrder = 0u;
  uint64_t controlVersion = 0u;
  uint64_t executedStepCount = 0u;
  uint64_t executedSubstepCount = 0u;
  std::array<uint64_t, static_cast<size_t>(SimulationPhase::Count)>
      phaseExecutionCounts{};
  uint64_t lastExecutedFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t paramsSizeBytes = 0u;
  bool enabled = false;
  bool paused = false;
  bool faulted = false;
  bool gpuEligible = false;
  // Uses the destination string allocator on assignment; copy construction uses
  // the default std::pmr allocator semantics.
  std::pmr::string lastFaultReason;
};

} // namespace nuri
