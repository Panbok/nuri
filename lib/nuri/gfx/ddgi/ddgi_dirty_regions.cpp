#include "nuri/gfx/ddgi/ddgi_dirty_regions.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_inverse.hpp>
#include <limits>

namespace nuri {
namespace {

[[nodiscard]] bool finiteVector(const glm::vec3 &value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool finiteMatrix(const glm::mat4 &value) noexcept {
  for (uint32_t column = 0u; column < 4u; ++column) {
    for (uint32_t row = 0u; row < 4u; ++row) {
      if (!std::isfinite(value[column][row])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool validBounds(const BoundingBox &bounds) noexcept {
  return finiteVector(bounds.min_) && finiteVector(bounds.max_) &&
         glm::all(glm::lessThanEqual(bounds.min_, bounds.max_));
}

[[nodiscard]] bool validVolume(const DDGIDirtyVolume &volume) noexcept {
  return volume.effectiveIndex < kMaxDDGIEffectiveVolumes &&
         glm::all(glm::greaterThan(volume.probeCounts, glm::uvec3(0u))) &&
         finiteVector(volume.probeSpacing) &&
         glm::all(glm::greaterThan(volume.probeSpacing, glm::vec3(0.0f))) &&
         std::isfinite(volume.queryBias) && volume.queryBias >= 0.0f &&
         finiteMatrix(volume.localFromWorld);
}

[[nodiscard]] uint32_t tierBit(DDGIEffectiveTier tier) noexcept {
  const uint32_t index = static_cast<uint32_t>(tier);
  return index < 32u ? 1u << index : 0u;
}

[[nodiscard]] DDGIDirtyProbeRange
affectedProbeRange(const BoundingBox &worldBounds,
                   const DDGIDirtyVolume &volume) noexcept {
  const BoundingBox localBounds =
      worldBounds.getTransformed(volume.localFromWorld);
  if (!validBounds(localBounds)) {
    return {};
  }

  const glm::vec3 trackedCenter =
      glm::vec3(volume.cameraCell) * volume.probeSpacing;
  const glm::vec3 gridMinimum =
      trackedCenter - 0.5f * glm::vec3(volume.probeCounts - glm::uvec3(1u)) *
                          volume.probeSpacing;
  const glm::vec3 influence = volume.probeSpacing + glm::vec3(volume.queryBias);
  constexpr float kCoordinateTolerance = 1.0e-4f;
  const glm::vec3 rawMinimum = glm::ceil(
      (localBounds.min_ - influence - gridMinimum) / volume.probeSpacing -
      glm::vec3(kCoordinateTolerance));
  const glm::vec3 rawMaximum = glm::floor(
      (localBounds.max_ + influence - gridMinimum) / volume.probeSpacing +
      glm::vec3(kCoordinateTolerance));
  const glm::vec3 maximumCoordinate =
      glm::vec3(volume.probeCounts - glm::uvec3(1u));
  if (glm::any(glm::lessThan(rawMaximum, glm::vec3(0.0f))) ||
      glm::any(glm::greaterThan(rawMinimum, maximumCoordinate))) {
    return {};
  }

  return {
      .minimum = glm::uvec3(
          glm::clamp(rawMinimum, glm::vec3(0.0f), maximumCoordinate)),
      .maximum = glm::uvec3(
          glm::clamp(rawMaximum, glm::vec3(0.0f), maximumCoordinate)),
      .valid = true,
  };
}

[[nodiscard]] DDGIDirtyRegion
makeGlobalRegion(const DDGISceneChangeRegion &change,
                 std::span<const DDGIDirtyVolume> volumes, uint64_t generation,
                 DDGIDirtyResponseFlags response) noexcept {
  DDGIDirtyRegion result{
      .worldBounds = change.worldBounds,
      .kind = change.kind,
      .response = response,
      .sourceId = change.sourceId,
      .sourceVersion = change.sourceVersion,
      .generation = generation,
      .submissionSequence = change.submissionSequence,
      .affectedTierMask = std::numeric_limits<uint32_t>::max(),
      .affectedVolumeMask = (1u << kMaxDDGIEffectiveVolumes) - 1u,
      .global = true,
  };
  for (const DDGIDirtyVolume &volume : volumes) {
    if (volume.effectiveIndex >= kMaxDDGIEffectiveVolumes ||
        glm::any(glm::equal(volume.probeCounts, glm::uvec3(0u)))) {
      continue;
    }
    result.probeRanges[volume.effectiveIndex] = {
        .minimum = glm::uvec3(0u),
        .maximum = volume.probeCounts - glm::uvec3(1u),
        .valid = true,
    };
  }
  return result;
}

} // namespace

DDGIDirtyResponseFlags
ddgiDirtyResponseForChange(DDGISceneChangeKind kind) noexcept {
  switch (kind) {
  case DDGISceneChangeKind::StaticTopology:
  case DDGISceneChangeKind::StaticTransform:
    return DDGIDirtyResponseFlags::RelocateClassify |
           DDGIDirtyResponseFlags::Irradiance |
           DDGIDirtyResponseFlags::Distance;
  case DDGISceneChangeKind::DynamicTransform:
  case DDGISceneChangeKind::Deformation:
    return DDGIDirtyResponseFlags::Wake | DDGIDirtyResponseFlags::Irradiance |
           DDGIDirtyResponseFlags::Distance;
  case DDGISceneChangeKind::LocalLight:
  case DDGISceneChangeKind::GlobalRadiometric:
    return DDGIDirtyResponseFlags::Irradiance;
  }
  return DDGIDirtyResponseFlags::All;
}

bool ddgiLocalLightInfluenceBounds(const LocalLightGpuData &light,
                                   BoundingBox &out) noexcept {
  if (light.innerCosTypeEnabledReserved.z == 0u) {
    return false;
  }
  const glm::vec3 position(light.positionRange);
  const float range = light.positionRange.w;
  if (!finiteVector(position) || !std::isfinite(range) || range <= 0.0f) {
    return false;
  }
  out = BoundingBox(position - glm::vec3(range), position + glm::vec3(range));
  return true;
}

DDGISceneChangeRegion makeDDGILocalLightChangeRegion(
    const LocalLightGpuData *previous, const LocalLightGpuData *current,
    uint64_t sourceId, uint64_t sourceVersion) noexcept {
  BoundingBox previousBounds{};
  BoundingBox currentBounds{};
  const bool previousKnown =
      previous != nullptr &&
      ddgiLocalLightInfluenceBounds(*previous, previousBounds);
  const bool currentKnown = current != nullptr && ddgiLocalLightInfluenceBounds(
                                                      *current, currentBounds);
  DDGISceneChangeRegion change{
      .kind = DDGISceneChangeKind::LocalLight,
      .sourceId = sourceId,
      .sourceVersion = sourceVersion,
  };
  if (previousKnown && currentKnown) {
    change.worldBounds =
        BoundingBox(glm::min(previousBounds.min_, currentBounds.min_),
                    glm::max(previousBounds.max_, currentBounds.max_));
    change.boundsKnown = true;
  } else if (previousKnown) {
    change.worldBounds = previousBounds;
    change.boundsKnown = true;
  } else if (currentKnown) {
    change.worldBounds = currentBounds;
    change.boundsKnown = true;
  }
  return change;
}

DDGIDirtyVolume makeDDGIDirtyVolume(const DDGIEffectiveVolume &volume,
                                    uint32_t effectiveIndex,
                                    float queryBias) noexcept {
  return {
      .localFromWorld = glm::inverse(volume.worldFromLocal),
      .probeCounts = volume.probeCounts,
      .probeSpacing = volume.probeSpacing,
      .cameraCell = volume.cameraCell,
      .queryBias = queryBias,
      .tier = volume.tier,
      .effectiveIndex = effectiveIndex,
  };
}

DDGIDirtyPublishResult DDGIDirtyRegionRing::publish(
    const DDGISceneChangeRegion &change,
    std::span<const DDGIDirtyVolume> volumes) noexcept {
  ++metrics_.produced;
  const bool global = change.kind == DDGISceneChangeKind::GlobalRadiometric ||
                      !change.boundsKnown || !validBounds(change.worldBounds);
  DDGIDirtyRegion region{};
  if (global) {
    region = makeGlobalRegion(change, volumes, nextGeneration_,
                              ddgiDirtyResponseForChange(change.kind));
  } else {
    region.worldBounds = change.worldBounds;
    region.kind = change.kind;
    region.response = ddgiDirtyResponseForChange(change.kind);
    region.sourceId = change.sourceId;
    region.sourceVersion = change.sourceVersion;
    region.generation = nextGeneration_;
    region.submissionSequence = change.submissionSequence;
    for (const DDGIDirtyVolume &volume : volumes) {
      if (!validVolume(volume)) {
        region = makeGlobalRegion(change, volumes, nextGeneration_,
                                  DDGIDirtyResponseFlags::All);
        break;
      }
      const DDGIDirtyProbeRange range =
          affectedProbeRange(change.worldBounds, volume);
      if (!range.valid) {
        continue;
      }
      region.probeRanges[volume.effectiveIndex] = range;
      region.affectedVolumeMask |= 1u << volume.effectiveIndex;
      region.affectedTierMask |= tierBit(volume.tier);
    }
    if (!region.global && region.affectedVolumeMask == 0u) {
      return DDGIDirtyPublishResult::OutsideCoverage;
    }
  }
  ++nextGeneration_;

  if (size_ == kMaxDDGIDirtyRegions) {
    const uint64_t collapsedCount = static_cast<uint64_t>(size_) + 1u;
    DDGISceneChangeRegion overflowChange{
        .kind = DDGISceneChangeKind::StaticTopology,
        .submissionSequence = change.submissionSequence,
    };
    region = makeGlobalRegion(overflowChange, volumes, region.generation,
                              DDGIDirtyResponseFlags::All);
    records_ = {};
    head_ = 0u;
    size_ = 1u;
    records_[0] = region;
    metrics_.merged += collapsedCount;
    ++metrics_.overflowed;
    return DDGIDirtyPublishResult::CollapsedToGlobal;
  }

  records_[(head_ + size_) % kMaxDDGIDirtyRegions] = region;
  ++size_;
  return DDGIDirtyPublishResult::Published;
}

bool DDGIDirtyRegionRing::prepareConsumption(uint64_t frameIndex) noexcept {
  if (pending_.active) {
    return pending_.frameIndex == frameIndex;
  }
  pending_ = {};
  pending_.frameIndex = frameIndex;
  pending_.count = size_;
  pending_.active = true;
  for (uint32_t index = 0u; index < size_; ++index) {
    pending_.records[index] = records_[(head_ + index) % kMaxDDGIDirtyRegions];
  }
  if (pending_.count > 0u) {
    pending_.consumeThroughGeneration =
        pending_.records[pending_.count - 1u].generation;
  }
  return true;
}

std::span<const DDGIDirtyRegion>
DDGIDirtyRegionRing::pendingRegions() const noexcept {
  return pending_.active ? std::span<const DDGIDirtyRegion>(
                               pending_.records.data(), pending_.count)
                         : std::span<const DDGIDirtyRegion>();
}

bool DDGIDirtyRegionRing::commitConsumption(uint64_t frameIndex) noexcept {
  if (!pending_.active || pending_.frameIndex != frameIndex) {
    return false;
  }
  while (size_ > 0u &&
         records_[head_].generation <= pending_.consumeThroughGeneration) {
    records_[head_] = {};
    head_ = (head_ + 1u) % kMaxDDGIDirtyRegions;
    --size_;
  }
  pending_ = {};
  return true;
}

void DDGIDirtyRegionRing::abandonConsumption(uint64_t frameIndex) noexcept {
  if (pending_.active && pending_.frameIndex == frameIndex) {
    pending_ = {};
  }
}

void DDGIDirtyRegionRing::clear() noexcept {
  records_ = {};
  pending_ = {};
  metrics_ = {};
  nextGeneration_ = 1u;
  head_ = 0u;
  size_ = 0u;
}

} // namespace nuri
