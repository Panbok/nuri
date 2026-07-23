#include "nuri/gfx/ddgi/ddgi_scheduler.h"

#include <algorithm>
#include <array>
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
  if (forceFullUpdate && (candidate.state == DDGIProbeState::Vigilant ||
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

[[nodiscard]] uint32_t
priorityClass(const DDGITieredProbeScheduleCandidate &candidate,
              bool forceFullUpdate) noexcept {
  if (candidate.invalidated ||
      candidate.state == DDGIProbeState::Uninitialized) {
    return 0u;
  }
  if (candidate.state == DDGIProbeState::NewlyVigilant ||
      candidate.state == DDGIProbeState::NewlyAwake) {
    return 1u;
  }
  if (forceFullUpdate && (candidate.state == DDGIProbeState::Vigilant ||
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

[[nodiscard]] int64_t saturatingAdd(int64_t left, int64_t right) noexcept {
  if (right > 0 && left > std::numeric_limits<int64_t>::max() - right) {
    return std::numeric_limits<int64_t>::max();
  }
  if (right < 0 && left < std::numeric_limits<int64_t>::min() - right) {
    return std::numeric_limits<int64_t>::min();
  }
  return left + right;
}

[[nodiscard]] int64_t saturatingProduct(uint32_t left,
                                        uint32_t right) noexcept {
  if (left != 0u &&
      static_cast<uint64_t>(right) >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / left) {
    return std::numeric_limits<int64_t>::max();
  }
  return static_cast<int64_t>(static_cast<uint64_t>(left) * right);
}

[[nodiscard]] uint32_t saturatingIncrement(uint32_t value) noexcept {
  return value == std::numeric_limits<uint32_t>::max() ? value : value + 1u;
}

[[nodiscard]] uint64_t secondaryReservation(uint64_t primaryQueries,
                                            uint32_t per1024Primary) noexcept {
  const uint64_t scaled = primaryQueries * std::min(per1024Primary, 1024u);
  return (scaled + 1023u) / 1024u;
}

template <typename Candidate>
[[nodiscard]] uint32_t
primaryRayCount(const Candidate &candidate,
                const DDGISchedulerLimits &limits) noexcept {
  const uint32_t classificationRays = limits.classificationRaysPerProbe == 0u
                                          ? limits.raysPerProbe
                                          : limits.classificationRaysPerProbe;
  return candidate.state == DDGIProbeState::Uninitialized ? classificationRays
                                                          : limits.raysPerProbe;
}

void accountPrimaryWork(DDGIScheduleResult &result, uint32_t rayCount,
                        bool classification) noexcept {
  result.primaryQueries += rayCount;
  if (classification) {
    ++result.classificationProbeUpdates;
    result.classificationPrimaryQueries += rayCount;
  } else {
    result.irradiancePrimaryQueries += rayCount;
  }
}

} // namespace

Result<DDGIScheduleResult, DDGISchedulerError>
scheduleDDGIProbeUpdates(std::span<const DDGIProbeScheduleCandidate> candidates,
                         const DDGISchedulerLimits &limits,
                         std::span<DDGIProbeScheduleCandidate> workspace,
                         std::span<DDGIProbeUpdateEntry> output) noexcept {
  using ScheduleResult = Result<DDGIScheduleResult, DDGISchedulerError>;
  if (limits.raysPerProbe == 0u ||
      limits.classificationRaysPerProbe > limits.raysPerProbe ||
      limits.maxProbeUpdates == 0u ||
      limits.secondaryQueriesPer1024Primary > 1024u ||
      limits.maxRayQueries <
          static_cast<uint64_t>(limits.raysPerProbe) +
              secondaryReservation(limits.raysPerProbe,
                                   limits.secondaryQueriesPer1024Primary)) {
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
  std::sort(eligible.begin(), eligible.end(),
            [&](const auto &left, const auto &right) {
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
  const uint32_t probeCapacity = std::min<uint32_t>(
      limits.maxProbeUpdates, static_cast<uint32_t>(output.size()));
  for (const DDGIProbeScheduleCandidate &candidate : eligible) {
    const uint32_t rayCount = primaryRayCount(candidate, limits);
    const uint64_t nextPrimary =
        static_cast<uint64_t>(result.primaryQueries) + rayCount;
    const uint64_t nextSecondary = secondaryReservation(
        nextPrimary, limits.secondaryQueriesPer1024Primary);
    if (result.updatedProbes >= probeCapacity ||
        nextPrimary + nextSecondary > limits.maxRayQueries) {
      break;
    }
    output[result.updatedProbes] = DDGIProbeUpdateEntry{
        .volumeStableId = candidate.volumeStableId,
        .probeId = candidate.probeId,
        .rayBase = result.primaryQueries,
        .rayCount = rayCount,
    };
    ++result.updatedProbes;
    accountPrimaryWork(result, rayCount,
                       candidate.state == DDGIProbeState::Uninitialized);
    result.secondaryQueriesReserved = static_cast<uint32_t>(nextSecondary);
  }
  result.unusedProbeCapacity = limits.maxProbeUpdates - result.updatedProbes;
  result.unusedQueryCapacity = limits.maxRayQueries - result.primaryQueries -
                               result.secondaryQueriesReserved;
  result.truncated = result.updatedProbes < eligibleCount;
  return ScheduleResult::makeResult(result);
}

Result<DDGITieredScheduleResult, DDGISchedulerError>
scheduleDDGITieredProbeUpdates(
    std::span<const DDGITieredProbeScheduleCandidate> candidates,
    std::span<const DDGITierScheduleInput> tiers,
    const DDGISchedulerLimits &limits,
    std::span<DDGITieredProbeScheduleCandidate> workspace,
    std::span<DDGIProbeUpdateEntry> output) noexcept {
  using TieredResult = Result<DDGITieredScheduleResult, DDGISchedulerError>;
  if (limits.raysPerProbe == 0u ||
      limits.classificationRaysPerProbe > limits.raysPerProbe ||
      limits.maxProbeUpdates == 0u ||
      limits.secondaryQueriesPer1024Primary > 1024u ||
      limits.maxRayQueries <
          static_cast<uint64_t>(limits.raysPerProbe) +
              secondaryReservation(limits.raysPerProbe,
                                   limits.secondaryQueriesPer1024Primary)) {
    return TieredResult::makeError(DDGISchedulerError::InvalidLimits);
  }
  if (workspace.size() < candidates.size()) {
    return TieredResult::makeError(DDGISchedulerError::WorkspaceTooSmall);
  }
  if (tiers.empty() || tiers.size() > kMaxDDGISchedulerTiers) {
    return TieredResult::makeError(DDGISchedulerError::InvalidTierInput);
  }

  std::array<uint32_t, kMaxDDGISchedulerTiers> tierOrder{};
  for (uint32_t tier = 0u; tier < tiers.size(); ++tier) {
    if (tiers[tier].weight == 0u) {
      return TieredResult::makeError(DDGISchedulerError::InvalidTierInput);
    }
    for (uint32_t other = 0u; other < tier; ++other) {
      if (tiers[other].stableKey == tiers[tier].stableKey) {
        return TieredResult::makeError(DDGISchedulerError::InvalidTierInput);
      }
    }
    tierOrder[tier] = tier;
  }
  std::sort(tierOrder.begin(), tierOrder.begin() + tiers.size(),
            [&](uint32_t left, uint32_t right) {
              if (tiers[left].effectiveOrder != tiers[right].effectiveOrder) {
                return tiers[left].effectiveOrder < tiers[right].effectiveOrder;
              }
              return tiers[left].stableKey < tiers[right].stableKey;
            });

  const auto tierIndex = [&](uint64_t stableKey) noexcept {
    for (uint32_t tier = 0u; tier < tiers.size(); ++tier) {
      if (tiers[tier].stableKey == stableKey) {
        return tier;
      }
    }
    return static_cast<uint32_t>(tiers.size());
  };

  size_t eligibleCount = 0u;
  for (const DDGITieredProbeScheduleCandidate &candidate : candidates) {
    const uint32_t tier = tierIndex(candidate.tierStableKey);
    if (tier == tiers.size()) {
      return TieredResult::makeError(DDGISchedulerError::InvalidTierInput);
    }
    if (!tiers[tier].ready ||
        priorityClass(candidate, limits.forceFullUpdate) ==
            std::numeric_limits<uint32_t>::max()) {
      continue;
    }
    workspace[eligibleCount++] = candidate;
  }
  std::span<DDGITieredProbeScheduleCandidate> eligible =
      workspace.first(eligibleCount);
  std::sort(eligible.begin(), eligible.end(),
            [&](const auto &left, const auto &right) {
              const uint32_t leftTier = tierIndex(left.tierStableKey);
              const uint32_t rightTier = tierIndex(right.tierStableKey);
              if (leftTier != rightTier) {
                return leftTier < rightTier;
              }
              const uint32_t leftPriority =
                  priorityClass(left, limits.forceFullUpdate);
              const uint32_t rightPriority =
                  priorityClass(right, limits.forceFullUpdate);
              if (leftPriority != rightPriority) {
                return leftPriority < rightPriority;
              }
              if (left.lastSubmittedUpdate != right.lastSubmittedUpdate) {
                return left.lastSubmittedUpdate < right.lastSubmittedUpdate;
              }
              if (left.volumeStableId != right.volumeStableId) {
                return left.volumeStableId < right.volumeStableId;
              }
              return left.probeId < right.probeId;
            });

  std::array<uint32_t, kMaxDDGISchedulerTiers> begins{};
  std::array<uint32_t, kMaxDDGISchedulerTiers> ends{};
  std::array<uint32_t, kMaxDDGISchedulerTiers> cursors{};
  begins.fill(static_cast<uint32_t>(eligibleCount));
  ends.fill(static_cast<uint32_t>(eligibleCount));
  for (uint32_t index = 0u; index < eligibleCount; ++index) {
    const uint32_t tier = tierIndex(eligible[index].tierStableKey);
    begins[tier] = std::min(begins[tier], index);
    ends[tier] = index + 1u;
  }
  cursors = begins;

  DDGITieredScheduleResult result{};
  result.tierCount = static_cast<uint32_t>(tiers.size());
  uint32_t nonemptyTierCount = 0u;
  uint64_t totalWeight64 = 0u;
  for (uint32_t tier = 0u; tier < tiers.size(); ++tier) {
    DDGITierScheduleResult &tierResult = result.tiers[tier];
    tierResult.stableKey = tiers[tier].stableKey;
    tierResult.eligibleProbes = ends[tier] - begins[tier];
    for (uint32_t index = begins[tier]; index < ends[tier]; ++index) {
      tierResult.urgentProbes +=
          priorityClass(eligible[index], limits.forceFullUpdate) <= 1u ? 1u
                                                                       : 0u;
    }
    if (tierResult.eligibleProbes != 0u) {
      ++nonemptyTierCount;
      totalWeight64 = std::min<uint64_t>(
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
          totalWeight64 + tiers[tier].weight);
    }
  }

  const uint32_t outputCapacity = static_cast<uint32_t>(
      std::min<size_t>(output.size(), std::numeric_limits<uint32_t>::max()));
  uint32_t queryCapacity = 0u;
  const uint32_t boundedCapacity =
      std::min(limits.maxProbeUpdates, outputCapacity);
  while (queryCapacity < boundedCapacity) {
    const uint64_t primary =
        static_cast<uint64_t>(queryCapacity + 1u) * limits.raysPerProbe;
    if (primary + secondaryReservation(primary,
                                       limits.secondaryQueriesPer1024Primary) >
        limits.maxRayQueries) {
      break;
    }
    ++queryCapacity;
  }
  const uint32_t capacity = queryCapacity;
  const uint32_t serviceCapacity =
      std::min(capacity, static_cast<uint32_t>(eligibleCount));
  result.urgentReservation = capacity / 2u;

  std::array<int64_t, kMaxDDGISchedulerTiers> allocationDeficit{};
  for (uint32_t tier = 0u; tier < tiers.size(); ++tier) {
    allocationDeficit[tier] = tiers[tier].submittedDeficit;
    if (result.tiers[tier].eligibleProbes != 0u) {
      allocationDeficit[tier] =
          saturatingAdd(allocationDeficit[tier],
                        saturatingProduct(serviceCapacity, tiers[tier].weight));
    }
  }
  const int64_t totalWeight = static_cast<int64_t>(totalWeight64);
  std::array<bool, kMaxDDGISchedulerTiers> served{};

  const auto appendFromTier = [&](uint32_t tier) noexcept {
    if (cursors[tier] >= ends[tier] ||
        result.schedule.updatedProbes >= serviceCapacity) {
      return false;
    }
    const DDGITieredProbeScheduleCandidate &candidate =
        eligible[cursors[tier]++];
    const uint32_t rayCount = primaryRayCount(candidate, limits);
    output[result.schedule.updatedProbes++] = DDGIProbeUpdateEntry{
        .volumeStableId = candidate.volumeStableId,
        .probeId = candidate.probeId,
        .rayBase = result.schedule.primaryQueries,
        .rayCount = rayCount,
    };
    accountPrimaryWork(result.schedule, rayCount,
                       candidate.state == DDGIProbeState::Uninitialized);
    result.schedule.secondaryQueriesReserved = static_cast<uint32_t>(
        secondaryReservation(result.schedule.primaryQueries,
                             limits.secondaryQueriesPer1024Primary));
    ++result.tiers[tier].usedQuota;
    allocationDeficit[tier] =
        saturatingAdd(allocationDeficit[tier], -totalWeight);
    served[tier] = true;
    return true;
  };

  while (result.urgentReservationUsed < result.urgentReservation &&
         result.schedule.updatedProbes < serviceCapacity) {
    uint32_t selected = static_cast<uint32_t>(tiers.size());
    for (uint32_t ordered = 0u; ordered < tiers.size(); ++ordered) {
      const uint32_t tier = tierOrder[ordered];
      if (cursors[tier] >= ends[tier] ||
          priorityClass(eligible[cursors[tier]], limits.forceFullUpdate) > 1u) {
        continue;
      }
      uint32_t unserved = 0u;
      for (uint32_t candidateTier = 0u; candidateTier < tiers.size();
           ++candidateTier) {
        unserved += result.tiers[candidateTier].eligibleProbes != 0u &&
                            !served[candidateTier]
                        ? 1u
                        : 0u;
      }
      const uint32_t remaining =
          serviceCapacity - result.schedule.updatedProbes;
      if (serviceCapacity >= nonemptyTierCount && served[tier] &&
          remaining <= unserved) {
        continue;
      }
      if (selected == tiers.size()) {
        selected = tier;
        continue;
      }
      const auto &candidate = eligible[cursors[tier]];
      const auto &chosen = eligible[cursors[selected]];
      const uint32_t candidatePriority =
          priorityClass(candidate, limits.forceFullUpdate);
      const uint32_t chosenPriority =
          priorityClass(chosen, limits.forceFullUpdate);
      if (candidatePriority < chosenPriority ||
          (candidatePriority == chosenPriority &&
           allocationDeficit[tier] > allocationDeficit[selected])) {
        selected = tier;
      }
    }
    if (selected == tiers.size()) {
      break;
    }
    ++result.tiers[selected].scheduledQuota;
    appendFromTier(selected);
    ++result.urgentReservationUsed;
  }

  if (serviceCapacity >= nonemptyTierCount) {
    for (uint32_t ordered = 0u; ordered < tiers.size(); ++ordered) {
      const uint32_t tier = tierOrder[ordered];
      if (result.tiers[tier].eligibleProbes == 0u || served[tier]) {
        continue;
      }
      ++result.tiers[tier].scheduledQuota;
      appendFromTier(tier);
    }
  }

  std::array<uint32_t, kMaxDDGISchedulerTiers> weightedQuota{};
  std::array<int64_t, kMaxDDGISchedulerTiers> quotaDeficit = allocationDeficit;
  const uint32_t quotaCapacity =
      serviceCapacity - result.schedule.updatedProbes;
  for (uint32_t slot = 0u; slot < quotaCapacity; ++slot) {
    uint32_t selected = static_cast<uint32_t>(tiers.size());
    for (uint32_t ordered = 0u; ordered < tiers.size(); ++ordered) {
      const uint32_t tier = tierOrder[ordered];
      if (result.tiers[tier].eligibleProbes == 0u) {
        continue;
      }
      if (selected == tiers.size() ||
          quotaDeficit[tier] > quotaDeficit[selected]) {
        selected = tier;
      }
    }
    if (selected == tiers.size()) {
      break;
    }
    ++weightedQuota[selected];
    ++result.tiers[selected].scheduledQuota;
    quotaDeficit[selected] =
        saturatingAdd(quotaDeficit[selected], -totalWeight);
  }

  uint32_t unusedQuota = 0u;
  for (uint32_t ordered = 0u; ordered < tiers.size(); ++ordered) {
    const uint32_t tier = tierOrder[ordered];
    for (uint32_t quota = 0u; quota < weightedQuota[tier]; ++quota) {
      if (!appendFromTier(tier)) {
        ++unusedQuota;
      }
    }
  }
  for (uint32_t ordered = 0u; ordered < tiers.size() && unusedQuota != 0u;
       ++ordered) {
    const uint32_t tier = tierOrder[ordered];
    while (unusedQuota != 0u && appendFromTier(tier)) {
      --unusedQuota;
    }
  }

  result.schedule.unusedProbeCapacity =
      limits.maxProbeUpdates - result.schedule.updatedProbes;
  result.schedule.unusedQueryCapacity =
      limits.maxRayQueries - result.schedule.primaryQueries -
      result.schedule.secondaryQueriesReserved;
  result.schedule.truncated = result.schedule.updatedProbes < eligibleCount;

  for (uint32_t tier = 0u; tier < tiers.size(); ++tier) {
    DDGITierScheduleResult &tierResult = result.tiers[tier];
    tierResult.pendingDeficit = tiers[tier].submittedDeficit;
    if (tierResult.eligibleProbes != 0u) {
      tierResult.pendingDeficit =
          saturatingAdd(tierResult.pendingDeficit,
                        saturatingProduct(serviceCapacity, tiers[tier].weight));
      for (uint32_t used = 0u; used < tierResult.usedQuota; ++used) {
        tierResult.pendingDeficit =
            saturatingAdd(tierResult.pendingDeficit, -totalWeight);
      }
      tierResult.pendingStarvationFrames =
          tierResult.usedQuota == 0u
              ? saturatingIncrement(tiers[tier].submittedStarvationFrames)
              : 0u;
    }
  }
  return TieredResult::makeResult(result);
}

} // namespace nuri
