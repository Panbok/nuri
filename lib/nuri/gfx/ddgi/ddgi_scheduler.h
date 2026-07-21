#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/ddgi/ddgi_types.h"
#include <cstdint>
#include <span>

namespace nuri {

struct DDGIProbeScheduleCandidate {
  uint32_t volumeStableId = 0u;
  uint32_t probeId = 0u;
  DDGIProbeState state = DDGIProbeState::Uninitialized;
  uint64_t lastSubmittedUpdate = 0u;
  bool invalidated = false;
};

struct DDGIProbeUpdateEntry {
  uint32_t volumeStableId = 0u;
  uint32_t probeId = 0u;
  uint32_t rayBase = 0u;
  uint32_t rayCount = 0u;
};

struct DDGISchedulerLimits {
  uint32_t raysPerProbe = 128u;
  uint32_t maxProbeUpdates = 512u;
  uint32_t maxRayQueries = 65'536u;
  bool forceFullUpdate = false;
};

enum class DDGISchedulerError : uint8_t {
  InvalidLimits = 0,
  WorkspaceTooSmall,
};

struct DDGIScheduleResult {
  uint32_t updatedProbes = 0u;
  uint32_t primaryQueries = 0u;
  uint32_t secondaryQueriesReserved = 0u;
  uint32_t unusedProbeCapacity = 0u;
  uint32_t unusedQueryCapacity = 0u;
  bool truncated = false;
};

[[nodiscard]] NURI_API Result<DDGIScheduleResult, DDGISchedulerError>
scheduleDDGIProbeUpdates(
    std::span<const DDGIProbeScheduleCandidate> candidates,
    const DDGISchedulerLimits &limits,
    std::span<DDGIProbeScheduleCandidate> workspace,
    std::span<DDGIProbeUpdateEntry> output) noexcept;

} // namespace nuri
