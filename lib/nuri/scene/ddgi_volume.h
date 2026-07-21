#pragma once

#include "nuri/defines.h"
#include <cstdint>
#include <glm/glm.hpp>
#include <string>

namespace nuri {

inline constexpr uint32_t kDDGIMinProbeCountPerAxis = 2u;
inline constexpr uint32_t kDDGIMaxProbeCountPerAxis = 64u;
inline constexpr uint64_t kDDGIMaxProbeCount = 65'536u;
inline constexpr uint32_t kDDGIMaxActiveVolumes = 4u;
inline constexpr float kDDGIMinProbeSpacing = 0.1f;
inline constexpr float kDDGIMaxProbeSpacing = 100.0f;
inline constexpr float kDDGIMaxRayDistance = 10'000.0f;

enum class DDGIVolumeMode : uint8_t {
  Authored = 0,
  CameraTracked = 1,
};

struct NURI_API DDGIVolumeDesc {
  std::string name{};
  glm::uvec3 probeCounts{16u, 8u, 16u};
  glm::vec3 probeSpacing{2.0f};
  float blendDistance = 2.0f;
  float maxRayDistance = 20.0f;
  int32_t priority = 0;
  DDGIVolumeMode mode = DDGIVolumeMode::Authored;
  bool enabled = true;
  constexpr bool operator==(const DDGIVolumeDesc &) const = default;
};

enum class DDGIVolumeValidationReason : uint8_t {
  None = 0,
  ProbeCountAxisOutOfRange,
  ProbeCountLimitExceeded,
  ProbeSpacingOutOfRange,
  BlendDistanceOutOfRange,
  MaxRayDistanceOutOfRange,
  NonRigidTransform,
  AtlasPackingUnavailable,
  PersistentMemoryLimitExceeded,
  PeakMemoryLimitExceeded,
};

struct DDGIVolumeValidationError {
  DDGIVolumeValidationReason reason = DDGIVolumeValidationReason::None;
  uint32_t axis = 0u;
};

[[nodiscard]] NURI_API DDGIVolumeValidationError
validateDDGIVolumeDesc(const DDGIVolumeDesc &desc) noexcept;

} // namespace nuri
