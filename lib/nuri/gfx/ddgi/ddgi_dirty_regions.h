#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/ddgi/ddgi_coverage.h"
#include "nuri/gfx/ray_tracing/ray_tracing_types.h"
#include "nuri/math/types.h"
#include "nuri/scene/light.h"

#include <array>
#include <cstdint>
#include <span>

namespace nuri {

inline constexpr uint32_t kMaxDDGIDirtyRegions = 32u;

enum class DDGIDirtyResponseFlags : uint8_t {
  None = 0u,
  RelocateClassify = 1u << 0u,
  Wake = 1u << 1u,
  Irradiance = 1u << 2u,
  Distance = 1u << 3u,
  All = (1u << 4u) - 1u,
};

[[nodiscard]] constexpr DDGIDirtyResponseFlags
operator|(DDGIDirtyResponseFlags left, DDGIDirtyResponseFlags right) noexcept {
  return static_cast<DDGIDirtyResponseFlags>(static_cast<uint8_t>(left) |
                                             static_cast<uint8_t>(right));
}

struct DDGIDirtyVolume {
  glm::mat4 localFromWorld{1.0f};
  glm::uvec3 probeCounts{2u};
  glm::vec3 probeSpacing{1.0f};
  glm::ivec3 cameraCell{0};
  // Local-space scalar expansion already derived from the shading query bias.
  float queryBias = 0.0f;
  DDGIEffectiveTier tier = DDGIEffectiveTier::AuthoredOverride;
  uint32_t effectiveIndex = 0u;
};

struct DDGIDirtyProbeRange {
  glm::uvec3 minimum{0u};
  glm::uvec3 maximum{0u};
  bool valid = false;
  constexpr bool operator==(const DDGIDirtyProbeRange &) const = default;
};

struct DDGIDirtyRegion {
  BoundingBox worldBounds{};
  std::array<DDGIDirtyProbeRange, kMaxDDGIEffectiveVolumes> probeRanges{};
  DDGISceneChangeKind kind = DDGISceneChangeKind::StaticTopology;
  DDGIDirtyResponseFlags response = DDGIDirtyResponseFlags::None;
  uint64_t sourceId = 0u;
  uint64_t sourceVersion = 0u;
  uint64_t generation = 0u;
  uint64_t submissionSequence = 0u;
  uint32_t affectedTierMask = 0u;
  uint32_t affectedVolumeMask = 0u;
  bool global = false;
};

struct DDGIDirtyRegionMetrics {
  uint64_t produced = 0u;
  uint64_t merged = 0u;
  uint64_t overflowed = 0u;
};

enum class DDGIDirtyPublishResult : uint8_t {
  Published = 0,
  OutsideCoverage,
  CollapsedToGlobal,
};

class NURI_API DDGIDirtyRegionRing final {
public:
  [[nodiscard]] DDGIDirtyPublishResult
  publish(const DDGISceneChangeRegion &change,
          std::span<const DDGIDirtyVolume> volumes) noexcept;

  // Copies a stable view of all currently unconsumed records. Only one owning
  // frame may have a prepared consumption batch at a time.
  [[nodiscard]] bool prepareConsumption(uint64_t frameIndex) noexcept;
  [[nodiscard]] std::span<const DDGIDirtyRegion>
  pendingRegions() const noexcept;
  [[nodiscard]] bool commitConsumption(uint64_t frameIndex) noexcept;
  void abandonConsumption(uint64_t frameIndex) noexcept;

  [[nodiscard]] uint32_t unconsumedCount() const noexcept { return size_; }
  [[nodiscard]] bool hasPendingConsumption() const noexcept {
    return pending_.active;
  }
  [[nodiscard]] const DDGIDirtyRegionMetrics &metrics() const noexcept {
    return metrics_;
  }
  void clear() noexcept;

private:
  struct PendingConsumption {
    std::array<DDGIDirtyRegion, kMaxDDGIDirtyRegions> records{};
    uint64_t frameIndex = 0u;
    uint64_t consumeThroughGeneration = 0u;
    uint32_t count = 0u;
    bool active = false;
  };

  std::array<DDGIDirtyRegion, kMaxDDGIDirtyRegions> records_{};
  PendingConsumption pending_{};
  DDGIDirtyRegionMetrics metrics_{};
  uint64_t nextGeneration_ = 1u;
  uint32_t head_ = 0u;
  uint32_t size_ = 0u;
};

[[nodiscard]] NURI_API DDGIDirtyVolume
makeDDGIDirtyVolume(const DDGIEffectiveVolume &volume, uint32_t effectiveIndex,
                    float queryBias) noexcept;

[[nodiscard]] NURI_API DDGIDirtyResponseFlags
ddgiDirtyResponseForChange(DDGISceneChangeKind kind) noexcept;

[[nodiscard]] NURI_API bool
ddgiLocalLightInfluenceBounds(const LocalLightGpuData &light,
                              BoundingBox &out) noexcept;

[[nodiscard]] NURI_API DDGISceneChangeRegion makeDDGILocalLightChangeRegion(
    const LocalLightGpuData *previous, const LocalLightGpuData *current,
    uint64_t sourceId, uint64_t sourceVersion) noexcept;

} // namespace nuri
