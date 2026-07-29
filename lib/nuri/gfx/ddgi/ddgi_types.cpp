#include "nuri/gfx/ddgi/ddgi_types.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_inverse.hpp>
#include <limits>

namespace nuri {

uint32_t ddgiProbeCount(const glm::uvec3 &counts) noexcept {
  const uint64_t count = static_cast<uint64_t>(counts.x) * counts.y * counts.z;
  return count <= std::numeric_limits<uint32_t>::max()
             ? static_cast<uint32_t>(count)
             : 0u;
}

std::array<DDGIVolumeMetricValue, 26>
ddgiVolumeMetricValues(const DDGIVolumeFrameMetrics &volume) noexcept {
  return {{{"active", volume.active},
           {"effective_kind", volume.effectiveKind},
           {"tier", volume.tier},
           {"cascade_index", volume.cascadeIndex},
           {"total_probes", volume.totalProbes},
           {"initialized_probes", volume.initializedProbes},
           {"shading_enabled_probes", volume.shadingEnabledProbes},
           {"invalid_probes", volume.invalidProbes},
           {"newly_exposed_probes", volume.newlyExposedProbes},
           {"updates", volume.updates},
           {"primary_queries", volume.primaryQueries},
           {"primary_queries_issued", volume.primaryQueriesIssued},
           {"secondary_queries", volume.secondaryQueries},
           {"update_age_median", volume.updateAgeMedian},
           {"update_age_p95", volume.updateAgeP95},
           {"update_age_maximum", volume.updateAgeMaximum},
           {"scheduled_quota", volume.scheduledQuota},
           {"used_quota", volume.usedQuota},
           {"deficit", static_cast<double>(volume.deficit)},
           {"starvation_frames", volume.starvationFrames},
           {"estimated_full_refresh_frames", volume.estimatedFullRefreshFrames},
           {"unique_coverage_percentage", volume.uniqueCoveragePercentage},
           {"redundant_coverage", volume.redundantCoverage},
           {"history_ready_percentage", volume.historyReadyPercentage},
           {"coverage_ready_percentage", volume.coverageReadyPercentage},
           {"confidence", volume.confidence}}};
}

uint32_t ddgiProbeIndex(const glm::uvec3 &coordinate,
                        const glm::uvec3 &counts) noexcept {
  return coordinate.x + counts.x * (coordinate.y + counts.y * coordinate.z);
}

glm::uvec3 ddgiProbeCoordinate(uint32_t probeIndex,
                               const glm::uvec3 &counts) noexcept {
  if (counts.x == 0u || counts.y == 0u) {
    return glm::uvec3(0u);
  }
  const uint32_t xy = counts.x * counts.y;
  const uint32_t z = probeIndex / xy;
  const uint32_t remainder = probeIndex % xy;
  return glm::uvec3(remainder % counts.x, remainder / counts.x, z);
}

glm::uvec3 ddgiPhysicalProbeCoordinate(const glm::uvec3 &logicalCoordinate,
                                       const glm::uvec3 &ringOrigin,
                                       const glm::uvec3 &counts) noexcept {
  glm::uvec3 physical(0u);
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    physical[axis] =
        counts[axis] == 0u
            ? 0u
            : (logicalCoordinate[axis] + ringOrigin[axis]) % counts[axis];
  }
  return physical;
}

glm::vec3 ddgiLocalProbePosition(const DDGIVolumeLayout &layout,
                                 const glm::uvec3 &coordinate) noexcept {
  return (glm::vec3(coordinate) -
          0.5f * glm::vec3(layout.probeCounts - glm::uvec3(1u))) *
             layout.probeSpacing +
         glm::vec3(layout.cameraCell) * layout.probeSpacing;
}

glm::ivec3 ddgiCameraCell(const glm::vec3 &cameraLocal,
                          const glm::vec3 &spacing) noexcept {
  return glm::ivec3(glm::floor(cameraLocal / spacing));
}

DDGIScrollPlan makeDDGIScrollPlan(const glm::ivec3 &cameraCell,
                                  const glm::uvec3 &ringOrigin,
                                  const glm::ivec3 &targetCameraCell,
                                  const glm::uvec3 &probeCounts) noexcept {
  DDGIScrollPlan plan{
      .cameraCell = targetCameraCell,
      .ringOrigin = ringOrigin,
      .cellDelta = targetCameraCell - cameraCell,
      .changed = targetCameraCell != cameraCell,
  };
  if (!plan.changed) {
    return plan;
  }
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    const int64_t count = static_cast<int64_t>(probeCounts[axis]);
    if (count <= 0) {
      plan.fullInvalidation = true;
      plan.ringOrigin[axis] = 0u;
      continue;
    }
    const int64_t delta = static_cast<int64_t>(plan.cellDelta[axis]);
    int64_t origin = (static_cast<int64_t>(ringOrigin[axis]) + delta) % count;
    if (origin < 0) {
      origin += count;
    }
    plan.ringOrigin[axis] = static_cast<uint32_t>(origin);
    plan.fullInvalidation |= std::abs(delta) >= count;
  }
  return plan;
}

bool isDDGINewlyExposedCoordinate(const glm::uvec3 &logicalCoordinate,
                                  const DDGIScrollPlan &plan,
                                  const glm::uvec3 &probeCounts) noexcept {
  if (!plan.changed) {
    return false;
  }
  if (plan.fullInvalidation) {
    return true;
  }
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    const int64_t delta = static_cast<int64_t>(plan.cellDelta[axis]);
    const uint32_t magnitude = static_cast<uint32_t>(std::abs(delta));
    if ((delta > 0 &&
         logicalCoordinate[axis] >= probeCounts[axis] - magnitude) ||
        (delta < 0 && logicalCoordinate[axis] < magnitude)) {
      return true;
    }
  }
  return false;
}

bool isRigidDDGITransform(const glm::mat4 &worldFromLocal,
                          float epsilon) noexcept {
  if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
    return false;
  }
  for (uint32_t column = 0u; column < 4u; ++column) {
    for (uint32_t row = 0u; row < 4u; ++row) {
      if (!std::isfinite(worldFromLocal[column][row])) {
        return false;
      }
    }
  }
  if (std::abs(worldFromLocal[0][3]) > epsilon ||
      std::abs(worldFromLocal[1][3]) > epsilon ||
      std::abs(worldFromLocal[2][3]) > epsilon ||
      std::abs(worldFromLocal[3][3] - 1.0f) > epsilon) {
    return false;
  }
  const glm::vec3 x(worldFromLocal[0]);
  const glm::vec3 y(worldFromLocal[1]);
  const glm::vec3 z(worldFromLocal[2]);
  return std::abs(glm::length(x) - 1.0f) <= epsilon &&
         std::abs(glm::length(y) - 1.0f) <= epsilon &&
         std::abs(glm::length(z) - 1.0f) <= epsilon &&
         std::abs(glm::dot(x, y)) <= epsilon &&
         std::abs(glm::dot(x, z)) <= epsilon &&
         std::abs(glm::dot(y, z)) <= epsilon &&
         glm::determinant(glm::mat3(worldFromLocal)) > 0.0f;
}

Result<DDGIVolumeLayout, DDGIVolumeValidationError>
makeDDGIVolumeLayout(DDGIVolumeId id, const DDGIVolumeDesc &desc,
                     const glm::mat4 &worldFromLocal,
                     DDGIAtlasLayout irradianceAtlas,
                     DDGIAtlasLayout distanceAtlas, uint64_t generation,
                     glm::ivec3 cameraCell, glm::uvec3 ringOrigin) {
  using LayoutResult = Result<DDGIVolumeLayout, DDGIVolumeValidationError>;
  const DDGIVolumeValidationError validation = validateDDGIVolumeDesc(desc);
  if (validation.reason != DDGIVolumeValidationReason::None) {
    return LayoutResult::makeError(validation);
  }
  if (!isRigidDDGITransform(worldFromLocal)) {
    return LayoutResult::makeError(
        {DDGIVolumeValidationReason::NonRigidTransform, 0u});
  }
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    if (ringOrigin[axis] >= desc.probeCounts[axis]) {
      ringOrigin[axis] %= desc.probeCounts[axis];
    }
  }
  return LayoutResult::makeResult(DDGIVolumeLayout{
      .id = id,
      .probeCounts = desc.probeCounts,
      .probeSpacing = desc.probeSpacing,
      .probeCenterHalfExtents = 0.5f *
                                glm::vec3(desc.probeCounts - glm::uvec3(1u)) *
                                desc.probeSpacing,
      .worldFromLocal = worldFromLocal,
      .localFromWorld = glm::affineInverse(worldFromLocal),
      .cameraCell = cameraCell,
      .ringOrigin = ringOrigin,
      .irradianceAtlas = irradianceAtlas,
      .distanceAtlas = distanceAtlas,
      .generation = generation,
  });
}

DDGIVolumeBlendWeights ddgiPriorityBlendWeights(float firstCoverage,
                                                float secondCoverage) noexcept {
  const float first = std::clamp(firstCoverage, 0.0f, 1.0f);
  const float secondCoverageClamped = std::clamp(secondCoverage, 0.0f, 1.0f);
  const float second = (1.0f - first) * secondCoverageClamped;
  return {.first = first,
          .second = second,
          .sky = (1.0f - first) * (1.0f - secondCoverageClamped)};
}

} // namespace nuri
