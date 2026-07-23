#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/ddgi/ddgi_types.h"
#include <array>
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

struct alignas(16) DDGIProbeUpdateEntry {
  uint32_t volumeStableId = 0u;
  uint32_t probeId = 0u;
  uint32_t rayBase = 0u;
  uint32_t rayCount = 0u;
  uint32_t flags = 0u;
  // Written by the probe-state pass and consumed only after the owning frame
  // slot is safe to reuse. These fields keep the CPU scheduler's submitted
  // state mirror bounded to the probes that actually ran.
  uint32_t resultState = static_cast<uint32_t>(DDGIProbeState::Uninitialized);
  uint32_t resultSubmittedSequence = 0u;
  uint32_t resultIteration = 0u;
  glm::vec4 resultRelocation{0.0f};
};

static_assert(sizeof(DDGIProbeUpdateEntry) == 48u);

inline constexpr uint32_t kDDGIProbeUpdateReclassify = 1u << 0u;
inline constexpr uint32_t kDDGIProbeUpdateWake = 1u << 1u;
inline constexpr uint32_t kDDGIProbeUpdateIrradianceResponse = 1u << 2u;
inline constexpr uint32_t kDDGIProbeUpdateDistanceResponse = 1u << 3u;

struct DDGISchedulerLimits {
  uint32_t raysPerProbe = 128u;
  // Zero preserves the legacy single-work-class contract for direct callers.
  // The renderer always supplies its separately sanitized low-ray count.
  uint32_t classificationRaysPerProbe = 0u;
  uint32_t maxProbeUpdates = 512u;
  uint32_t maxRayQueries = 65'536u;
  // Fixed-point upper bound for secondary visibility reservations. 1024 means
  // one secondary query per primary ray; lower values safely lend the
  // remainder to primary probe work.
  uint32_t secondaryQueriesPer1024Primary = 1024u;
  bool forceFullUpdate = false;
};

enum class DDGISchedulerError : uint8_t {
  InvalidLimits = 0,
  WorkspaceTooSmall,
  InvalidTierInput,
};

struct DDGIScheduleResult {
  uint32_t updatedProbes = 0u;
  uint32_t primaryQueries = 0u;
  uint32_t classificationProbeUpdates = 0u;
  uint32_t classificationPrimaryQueries = 0u;
  uint32_t irradiancePrimaryQueries = 0u;
  uint32_t secondaryQueriesReserved = 0u;
  uint32_t unusedProbeCapacity = 0u;
  uint32_t unusedQueryCapacity = 0u;
  bool truncated = false;
};

[[nodiscard]] NURI_API Result<DDGIScheduleResult, DDGISchedulerError>
scheduleDDGIProbeUpdates(std::span<const DDGIProbeScheduleCandidate> candidates,
                         const DDGISchedulerLimits &limits,
                         std::span<DDGIProbeScheduleCandidate> workspace,
                         std::span<DDGIProbeUpdateEntry> output) noexcept;

inline constexpr uint32_t kMaxDDGISchedulerTiers = 8u;

struct DDGITierScheduleInput {
  uint64_t stableKey = 0u;
  // Committed state from the last submitted frame.
  int64_t submittedDeficit = 0;
  uint32_t submittedStarvationFrames = 0u;
  uint32_t effectiveOrder = 0u;
  uint32_t weight = 1u;
  bool ready = true;
};

struct DDGITieredProbeScheduleCandidate {
  uint64_t tierStableKey = 0u;
  uint32_t volumeStableId = 0u;
  uint32_t probeId = 0u;
  DDGIProbeState state = DDGIProbeState::Uninitialized;
  uint64_t lastSubmittedUpdate = 0u;
  bool invalidated = false;
};

struct DDGITierScheduleResult {
  uint64_t stableKey = 0u;
  // Pending state; the caller commits it only when the owning frame submits.
  int64_t pendingDeficit = 0;
  uint32_t pendingStarvationFrames = 0u;
  // Rollover can make usedQuota exceed the tier's initial entitlement.
  uint32_t scheduledQuota = 0u;
  uint32_t usedQuota = 0u;
  uint32_t eligibleProbes = 0u;
  uint32_t urgentProbes = 0u;
};

struct DDGITieredScheduleResult {
  DDGIScheduleResult schedule{};
  std::array<DDGITierScheduleResult, kMaxDDGISchedulerTiers> tiers{};
  uint32_t tierCount = 0u;
  uint32_t urgentReservation = 0u;
  uint32_t urgentReservationUsed = 0u;
};

[[nodiscard]] NURI_API Result<DDGITieredScheduleResult, DDGISchedulerError>
scheduleDDGITieredProbeUpdates(
    std::span<const DDGITieredProbeScheduleCandidate> candidates,
    std::span<const DDGITierScheduleInput> tiers,
    const DDGISchedulerLimits &limits,
    std::span<DDGITieredProbeScheduleCandidate> workspace,
    std::span<DDGIProbeUpdateEntry> output) noexcept;

} // namespace nuri
