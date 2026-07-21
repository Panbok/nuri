#include "nuri/scene/ddgi_volume.h"

#include <algorithm>
#include <cmath>

namespace nuri {

DDGIVolumeValidationError
validateDDGIVolumeDesc(const DDGIVolumeDesc &desc) noexcept {
  uint64_t total = 1u;
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    if (desc.probeCounts[axis] < kDDGIMinProbeCountPerAxis ||
        desc.probeCounts[axis] > kDDGIMaxProbeCountPerAxis) {
      return {DDGIVolumeValidationReason::ProbeCountAxisOutOfRange, axis};
    }
    total *= desc.probeCounts[axis];
    if (!std::isfinite(desc.probeSpacing[axis]) ||
        desc.probeSpacing[axis] < kDDGIMinProbeSpacing ||
        desc.probeSpacing[axis] > kDDGIMaxProbeSpacing) {
      return {DDGIVolumeValidationReason::ProbeSpacingOutOfRange, axis};
    }
  }
  if (total > kDDGIMaxProbeCount) {
    return {DDGIVolumeValidationReason::ProbeCountLimitExceeded, 0u};
  }

  const glm::vec3 halfExtents =
      0.5f * glm::vec3(desc.probeCounts - glm::uvec3(1u)) * desc.probeSpacing;
  const float smallestHalfExtent =
      std::min({halfExtents.x, halfExtents.y, halfExtents.z});
  if (!std::isfinite(desc.blendDistance) || desc.blendDistance < 0.0f ||
      desc.blendDistance > smallestHalfExtent) {
    return {DDGIVolumeValidationReason::BlendDistanceOutOfRange, 0u};
  }
  const float smallestSpacing =
      std::min({desc.probeSpacing.x, desc.probeSpacing.y, desc.probeSpacing.z});
  if (!std::isfinite(desc.maxRayDistance) ||
      desc.maxRayDistance < 0.5f * smallestSpacing ||
      desc.maxRayDistance > kDDGIMaxRayDistance) {
    return {DDGIVolumeValidationReason::MaxRayDistanceOutOfRange, 0u};
  }
  return {};
}

} // namespace nuri
