#include "nuri/gfx/ddgi/ddgi_scheduler.h"

#include <algorithm>
#include <limits>

namespace nuri {
namespace {

[[nodiscard]] uint32_t
priorityClass(const DDGIProbeScheduleCandidate &candidate,
              bool forceFullUpdate) noexcept {
  if (candidate.invalidated ||
      candidate.state == DDGIProbeState::Uninitialized) {
    return 0u;
  }
  if (candidate.state == DDGIProbeState::NewlyVigilant ||
      candidate.state == DDGIProbeState::NewlyAwake) {
    return 1u;
  }
  if (forceFullUpdate &&
      (candidate.state == DDGIProbeState::Vigilant ||
       candidate.state == DDGIProbeState::Awake)) {
    return 2u;
  }
  if (candidate.state == DDGIProbeState::Vigilant) {
    return 3u;
  }
  if (candidate.state == DDGIProbeState::Awake) {
    return 4u;
  }
  return std::numeric_limits<uint32_t>::max();
}

} // namespace

Result<DDGIScheduleResult, DDGISchedulerError> scheduleDDGIProbeUpdates(
    std::span<const DDGIProbeScheduleCandidate> candidates,
    const DDGISchedulerLimits &limits,
    std::span<DDGIProbeScheduleCandidate> workspace,
    std::span<DDGIProbeUpdateEntry> output) noexcept {
  using ScheduleResult = Result<DDGIScheduleResult, DDGISchedulerError>;
  if (limits.raysPerProbe == 0u || limits.maxProbeUpdates == 0u ||
      limits.maxRayQueries < 2u * static_cast<uint64_t>(limits.raysPerProbe)) {
    return ScheduleResult::makeError(DDGISchedulerError::InvalidLimits);
  }
  if (workspace.size() < candidates.size()) {
    return ScheduleResult::makeError(DDGISchedulerError::WorkspaceTooSmall);
  }

  size_t eligibleCount = 0u;
  for (const DDGIProbeScheduleCandidate &candidate : candidates) {
    if (priorityClass(candidate, limits.forceFullUpdate) ==
        std::numeric_limits<uint32_t>::max()) {
      continue;
    }
    workspace[eligibleCount++] = candidate;
  }
  std::span<DDGIProbeScheduleCandidate> eligible =
      workspace.first(eligibleCount);
  std::sort(eligible.begin(), eligible.end(), [&](const auto &left,
                                                   const auto &right) {
    const uint32_t leftPriority =
        priorityClass(left, limits.forceFullUpdate);
    const uint32_t rightPriority =
        priorityClass(right, limits.forceFullUpdate);
    if (leftPriority != rightPriority) {
      return leftPriority < rightPriority;
    }
    if (leftPriority >= 2u &&
        left.lastSubmittedUpdate != right.lastSubmittedUpdate) {
      return left.lastSubmittedUpdate < right.lastSubmittedUpdate;
    }
    if (left.volumeStableId != right.volumeStableId) {
      return left.volumeStableId < right.volumeStableId;
    }
    return left.probeId < right.probeId;
  });

  DDGIScheduleResult result{};
  const uint64_t reservedQueriesPerProbe =
      2u * static_cast<uint64_t>(limits.raysPerProbe);
  const uint32_t probeCapacity = std::min<uint32_t>(
      limits.maxProbeUpdates, static_cast<uint32_t>(output.size()));
  uint64_t reservedQueries = 0u;
  for (const DDGIProbeScheduleCandidate &candidate : eligible) {
    if (result.updatedProbes >= probeCapacity ||
        reservedQueries + reservedQueriesPerProbe > limits.maxRayQueries) {
      break;
    }
    output[result.updatedProbes] = DDGIProbeUpdateEntry{
        .volumeStableId = candidate.volumeStableId,
        .probeId = candidate.probeId,
        .rayBase = result.primaryQueries,
        .rayCount = limits.raysPerProbe,
    };
    ++result.updatedProbes;
    result.primaryQueries += limits.raysPerProbe;
    result.secondaryQueriesReserved += limits.raysPerProbe;
    reservedQueries += reservedQueriesPerProbe;
  }
  result.unusedProbeCapacity = limits.maxProbeUpdates - result.updatedProbes;
  result.unusedQueryCapacity =
      limits.maxRayQueries - static_cast<uint32_t>(reservedQueries);
  result.truncated = result.updatedProbes < eligibleCount;
  return ScheduleResult::makeResult(result);
}

} // namespace nuri
