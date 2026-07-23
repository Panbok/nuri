#include "nuri/gfx/ddgi/ddgi_coverage.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <tuple>

namespace nuri {
namespace {

constexpr glm::vec3 kDefaultNearSpacing{1.5f};
constexpr glm::vec3 kDefaultCoverageHalfExtents{55.0f, 30.0f, 55.0f};

template <typename Enum>
[[nodiscard]] constexpr bool enumInRange(Enum value, Enum maximum) noexcept {
  return static_cast<uint8_t>(value) <= static_cast<uint8_t>(maximum);
}

[[nodiscard]] bool finiteVector(const glm::vec3 &value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool boundsFacts(const BoundingBox &bounds, glm::vec3 &size,
                               glm::vec3 &center) noexcept {
  if (!finiteVector(bounds.min_) || !finiteVector(bounds.max_) ||
      glm::any(glm::greaterThan(bounds.min_, bounds.max_))) {
    return false;
  }
  constexpr double kFloatMaximum = std::numeric_limits<float>::max();
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    const double minimum = bounds.min_[axis];
    const double maximum = bounds.max_[axis];
    const double length = maximum - minimum;
    const double midpoint = minimum + 0.5 * length;
    if (!std::isfinite(length) || length > kFloatMaximum ||
        !std::isfinite(midpoint) || std::abs(midpoint) > kFloatMaximum) {
      return false;
    }
    size[axis] = static_cast<float>(length);
    center[axis] = static_cast<float>(midpoint);
  }
  return finiteVector(size) && finiteVector(center);
}

[[nodiscard]] bool validBounds(const BoundingBox &bounds) noexcept {
  glm::vec3 size(0.0f);
  glm::vec3 center(0.0f);
  return boundsFacts(bounds, size, center);
}

[[nodiscard]] bool checkedAdd(uint64_t left, uint64_t right,
                              uint64_t &out) noexcept {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    return false;
  }
  out = left + right;
  return true;
}

[[nodiscard]] uint64_t requiredInteriorCells(float length,
                                             float spacing) noexcept {
  const double cells =
      std::ceil(static_cast<double>(length) / static_cast<double>(spacing));
  if (cells >= static_cast<double>(std::numeric_limits<uint64_t>::max())) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(std::max(cells, 0.0));
}

struct EvaluatedLayout {
  glm::uvec3 counts{0u};
  glm::vec3 spacing{0.0f};
  DDGIAtlasLayout irradianceAtlas{};
  DDGIAtlasLayout distanceAtlas{};
  DDGIMemoryEstimate memory{};
  uint32_t probeCount = 0u;
};

using EvaluationResult = Result<EvaluatedLayout, DDGICoverageSolveError>;

[[nodiscard]] EvaluationResult
evaluateLayout(const glm::uvec3 &counts, const glm::vec3 &spacing,
               const DDGICoverageSolveLimits &limits) noexcept {
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    if (counts[axis] < kDDGIMinProbeCountPerAxis ||
        counts[axis] > kDDGIMaxProbeCountPerAxis) {
      return EvaluationResult::makeError(
          {DDGICoverageLimit::ProbeCountAxis, axis});
    }
    if (!std::isfinite(spacing[axis]) || spacing[axis] < kDDGIMinProbeSpacing ||
        spacing[axis] > kDDGIMaxProbeSpacing) {
      return EvaluationResult::makeError(
          {DDGICoverageLimit::ProbeSpacing, axis});
    }
  }

  const uint32_t probeCount = ddgiProbeCount(counts);
  if (probeCount == 0u || probeCount > kDDGIMaxProbeCount) {
    return EvaluationResult::makeError(
        {DDGICoverageLimit::TotalProbeCount, 0u});
  }
  auto irradiance = packDDGIAtlas(probeCount, kDDGIIrradianceTileExtent,
                                  limits.maxTextureExtent);
  auto distance = packDDGIAtlas(probeCount, kDDGIDistanceTileExtent,
                                limits.maxTextureExtent);
  if (irradiance.hasError() || distance.hasError()) {
    return EvaluationResult::makeError(
        {DDGICoverageLimit::AtlasDimensions, 0u});
  }
  auto memory = estimateDDGIMemory(probeCount, *irradiance, *distance);
  if (memory.hasError()) {
    return EvaluationResult::makeError(
        {DDGICoverageLimit::PersistentMemory, 0u});
  }

  uint64_t committedBytes = 0u;
  if (!checkedAdd(limits.otherPersistentBytes, memory->persistentBytes,
                  committedBytes) ||
      committedBytes > limits.maxPersistentBytes) {
    return EvaluationResult::makeError(
        {DDGICoverageLimit::PersistentMemory, 0u});
  }
  uint64_t peakBytes = 0u;
  if (!checkedAdd(committedBytes, limits.retainedReplacementBytes, peakBytes) ||
      peakBytes > limits.maxReplacementPeakBytes) {
    return EvaluationResult::makeError(
        {DDGICoverageLimit::ReplacementPeakMemory, 0u});
  }

  return EvaluationResult::makeResult(EvaluatedLayout{
      .counts = counts,
      .spacing = spacing,
      .irradianceAtlas = *irradiance,
      .distanceAtlas = *distance,
      .memory = *memory,
      .probeCount = probeCount,
  });
}

[[nodiscard]] Result<glm::uvec3, DDGICoverageSolveError>
fullCoverageCounts(const glm::vec3 &size, const glm::vec3 &spacing,
                   uint32_t paddingCells) noexcept {
  using CountsResult = Result<glm::uvec3, DDGICoverageSolveError>;
  glm::uvec3 counts(0u);
  const uint64_t maximumInteriorCells =
      kDDGIMaxProbeCountPerAxis - 1u - 2u * paddingCells;
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    const uint64_t cells = requiredInteriorCells(size[axis], spacing[axis]);
    if (cells > maximumInteriorCells) {
      return CountsResult::makeError({DDGICoverageLimit::ProbeCountAxis, axis});
    }
    counts[axis] = static_cast<uint32_t>(cells + 1u + 2u * paddingCells);
  }
  return CountsResult::makeResult(counts);
}

[[nodiscard]] EvaluationResult
evaluateFullCoverage(const glm::vec3 &size, const glm::vec3 &requestedSpacing,
                     float spacingScale, uint32_t paddingCells,
                     const DDGICoverageSolveLimits &limits) noexcept {
  const glm::vec3 spacing = requestedSpacing * spacingScale;
  auto counts = fullCoverageCounts(size, spacing, paddingCells);
  if (counts.hasError()) {
    return EvaluationResult::makeError(counts.error());
  }
  return evaluateLayout(*counts, spacing, limits);
}

[[nodiscard]] glm::uvec3
maximumNearSpacingCounts(const glm::vec3 &size, const glm::vec3 &spacing,
                         uint32_t paddingCells) noexcept {
  const uint32_t maximumInteriorCells =
      kDDGIMaxProbeCountPerAxis - 1u - 2u * paddingCells;
  glm::uvec3 counts(0u);
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    const uint64_t requestedCells =
        requiredInteriorCells(size[axis], spacing[axis]);
    const uint32_t retainedCells = static_cast<uint32_t>(
        std::min<uint64_t>(requestedCells, maximumInteriorCells));
    counts[axis] = retainedCells + 1u + 2u * paddingCells;
  }
  return counts;
}

[[nodiscard]] glm::uvec3 scaledInteriorCounts(const glm::uvec3 &maximumCounts,
                                              uint32_t paddingCells,
                                              float scale) noexcept {
  glm::uvec3 counts(0u);
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    const uint32_t maximumInteriorCells =
        maximumCounts[axis] - 1u - 2u * paddingCells;
    const uint32_t retainedCells =
        maximumInteriorCells == 0u
            ? 0u
            : std::max(1u, static_cast<uint32_t>(
                               std::floor(maximumInteriorCells * scale)));
    counts[axis] = retainedCells + 1u + 2u * paddingCells;
  }
  return counts;
}

[[nodiscard]] float coverageRatio(const BoundingBox &requested,
                                  const BoundingBox &achieved) noexcept {
  const glm::vec3 requestedSize = requested.getSize();
  if (glm::all(glm::lessThanEqual(requestedSize, glm::vec3(0.0f)))) {
    return ddgiBoundsContain(achieved, requested) ? 1.0f : 0.0f;
  }
  const glm::vec3 intersectionMin = glm::max(requested.min_, achieved.min_);
  const glm::vec3 intersectionMax = glm::min(requested.max_, achieved.max_);
  const glm::vec3 intersectionSize =
      glm::max(intersectionMax - intersectionMin, glm::vec3(0.0f));
  const double requestedVolume =
      static_cast<double>(requestedSize.x) * requestedSize.y * requestedSize.z;
  if (requestedVolume <= 0.0) {
    return ddgiBoundsContain(achieved, requested) ? 1.0f : 0.0f;
  }
  const double intersectionVolume = static_cast<double>(intersectionSize.x) *
                                    intersectionSize.y * intersectionSize.z;
  return static_cast<float>(
      std::clamp(intersectionVolume / requestedVolume, 0.0, 1.0));
}

using SolutionResult = Result<DDGISceneFitSolution, DDGICoverageSolveError>;

[[nodiscard]] SolutionResult makeSolution(
    const DDGISceneCoverageBounds &sceneBounds,
    const DDGICoverageSettings &settings, const DDGICoverageSolveLimits &limits,
    const EvaluatedLayout &layout, DDGICoverageLimit limitingConstraint,
    const glm::vec3 &center) noexcept {
  const glm::uvec3 interiorCellCounts =
      layout.counts - glm::uvec3(1u + 2u * settings.scenePaddingCells);
  const glm::vec3 interiorHalfExtents =
      0.5f * glm::vec3(interiorCellCounts) * layout.spacing;
  const glm::vec3 probeCenterHalfExtents =
      0.5f * glm::vec3(layout.counts - glm::uvec3(1u)) * layout.spacing;
  const glm::vec3 achievedMinimum = center - interiorHalfExtents;
  const glm::vec3 achievedMaximum = center + interiorHalfExtents;
  const glm::vec3 probeMinimum = center - probeCenterHalfExtents;
  const glm::vec3 probeMaximum = center + probeCenterHalfExtents;
  if (!finiteVector(achievedMinimum) || !finiteVector(achievedMaximum) ||
      !finiteVector(probeMinimum) || !finiteVector(probeMaximum)) {
    return SolutionResult::makeError(
        {DDGICoverageLimit::ArithmeticOverflow, 0u});
  }
  const BoundingBox achieved(achievedMinimum, achievedMaximum);
  const uint64_t updateCapacity =
      std::max<uint64_t>(limits.maxProbeUpdatesPerFrame, 1u);
  const uint32_t refreshFrames = static_cast<uint32_t>(
      (static_cast<uint64_t>(layout.probeCount) + updateCapacity - 1u) /
      updateCapacity);
  return SolutionResult::makeResult(DDGISceneFitSolution{
      .requestedBounds = sceneBounds.bounds,
      .achievedInteriorBounds = achieved,
      .worldCenter = center,
      .probeCounts = layout.counts,
      .requestedSpacing = settings.requestedNearSpacing,
      .achievedSpacing = layout.spacing,
      .probeCenterHalfExtents = probeCenterHalfExtents,
      .interiorHalfExtents = interiorHalfExtents,
      .irradianceAtlas = layout.irradianceAtlas,
      .distanceAtlas = layout.distanceAtlas,
      .memory = layout.memory,
      .probeCount = layout.probeCount,
      .estimatedMinimumRefreshFrames = refreshFrames,
      .requestedVolumeCoverage = coverageRatio(sceneBounds.bounds, achieved),
      .fullCoverage = ddgiBoundsContain(achieved, sceneBounds.bounds),
      .limitingConstraint = limitingConstraint,
  });
}

} // namespace

void DDGIEffectiveVolumePlan::clear() noexcept {
  volumes = {};
  omittedKeys.clear();
  failedKeys.clear();
  sceneFit = {};
  clipmaps = {};
  error = {};
  mode = DDGICoverageMode::Manual;
  sceneId = 0u;
  coverageGeneration = 0u;
  sceneBoundsGeneration = 0u;
  persistentBytes = 0u;
  replacementPeakBytes = 0u;
  volumeCount = 0u;
  candidateCount = 0u;
  ready = false;
  overflowed = false;
  fullSceneCoverage = false;
}

bool ddgiEffectiveVolumeKeyLess(const DDGIEffectiveVolumeKey &left,
                                const DDGIEffectiveVolumeKey &right) noexcept {
  return std::tuple(static_cast<uint8_t>(left.kind), left.authoredId.value,
                    left.sceneId, left.coverageGeneration,
                    left.generatedIndex) <
         std::tuple(static_cast<uint8_t>(right.kind), right.authoredId.value,
                    right.sceneId, right.coverageGeneration,
                    right.generatedIndex);
}

uint64_t
ddgiEffectiveVolumeKeyHash(const DDGIEffectiveVolumeKey &key) noexcept {
  uint64_t hash = 1469598103934665603ull;
  const auto mix = [&hash](uint64_t value) {
    for (uint32_t byte = 0u; byte < 8u; ++byte) {
      hash ^= (value >> (byte * 8u)) & 0xffu;
      hash *= 1099511628211ull;
    }
  };
  mix(static_cast<uint8_t>(key.kind));
  mix(key.authoredId.value);
  mix(key.sceneId);
  mix(key.coverageGeneration);
  mix(key.generatedIndex);
  return hash;
}

DDGIRedundancyAnalysis
analyzeDDGIVolumeRedundancy(const DDGIEffectiveVolume &authored,
                            const DDGIEffectiveVolume &generated) noexcept {
  DDGIRedundancyAnalysis result{};
  if (authored.key.kind != DDGIEffectiveVolumeKind::Authored ||
      generated.key.kind == DDGIEffectiveVolumeKind::Authored) {
    return result;
  }
  const auto worldCellVolume = [](const DDGIEffectiveVolume &volume) {
    const float localCellVolume =
        volume.probeSpacing.x * volume.probeSpacing.y * volume.probeSpacing.z;
    return std::abs(glm::determinant(glm::mat3(volume.worldFromLocal))) *
           localCellVolume;
  };
  result.densityRedundant =
      worldCellVolume(authored) + 1.0e-5f >= worldCellVolume(generated);

  const glm::mat4 generatedLocalFromWorld =
      glm::inverse(generated.worldFromLocal);
  const glm::vec3 authoredHalfExtents =
      glm::max(authored.fadeEndHalfExtents, authored.probeCenterHalfExtents);
  const glm::vec3 generatedHalfExtents =
      glm::max(generated.fadeEndHalfExtents, generated.probeCenterHalfExtents);
  result.fullyCovered = true;
  for (uint32_t corner = 0u; corner < 8u; ++corner) {
    const glm::vec3 sign((corner & 1u) != 0u ? 1.0f : -1.0f,
                         (corner & 2u) != 0u ? 1.0f : -1.0f,
                         (corner & 4u) != 0u ? 1.0f : -1.0f);
    const glm::vec3 worldCorner = glm::vec3(
        authored.worldFromLocal * glm::vec4(sign * authoredHalfExtents, 1.0f));
    const glm::vec3 generatedCorner =
        glm::vec3(generatedLocalFromWorld * glm::vec4(worldCorner, 1.0f));
    if (glm::any(glm::greaterThan(glm::abs(generatedCorner),
                                  generatedHalfExtents + glm::vec3(1.0e-4f)))) {
      result.fullyCovered = false;
      break;
    }
  }
  result.fullyRedundant = result.fullyCovered && result.densityRedundant;
  return result;
}

void sanitizeDDGICoverageSettings(DDGICoverageSettings &settings) noexcept {
  if (!enumInRange(settings.mode, DDGICoverageMode::Hybrid)) {
    settings.mode = DDGICoverageMode::Manual;
  }
  if (!enumInRange(settings.constraintPolicy,
                   DDGICoverageConstraintPolicy::RejectUnsatisfied)) {
    settings.constraintPolicy = DDGICoverageConstraintPolicy::PreserveCoverage;
  }
  if (!enumInRange(settings.sceneBoundsSource,
                   DDGISceneBoundsSource::Authored)) {
    settings.sceneBoundsSource = DDGISceneBoundsSource::ActivationSnapshot;
  }
  settings.cascadeCount =
      std::clamp(settings.cascadeCount, 1u, kMaxDDGIClipmapCascades);
  settings.transitionCells = std::clamp(settings.transitionCells, 1u, 3u);
  const uint32_t minimumCascadeCount =
      std::max(kDDGIMinProbeCountPerAxis, 2u * settings.transitionCells + 1u);
  settings.cascadeProbeCounts =
      glm::clamp(settings.cascadeProbeCounts, glm::uvec3(minimumCascadeCount),
                 glm::uvec3(kDDGIMaxProbeCountPerAxis));
  while (ddgiProbeCount(settings.cascadeProbeCounts) > kDDGIMaxProbeCount) {
    uint32_t largestAxis = 0u;
    for (uint32_t axis = 1u; axis < 3u; ++axis) {
      if (settings.cascadeProbeCounts[axis] >
          settings.cascadeProbeCounts[largestAxis]) {
        largestAxis = axis;
      }
    }
    if (settings.cascadeProbeCounts[largestAxis] <= minimumCascadeCount) {
      break;
    }
    --settings.cascadeProbeCounts[largestAxis];
  }
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    settings.requestedNearSpacing[axis] =
        std::clamp(std::isfinite(settings.requestedNearSpacing[axis])
                       ? settings.requestedNearSpacing[axis]
                       : kDefaultNearSpacing[axis],
                   kDDGIMinProbeSpacing, kDDGIMaxProbeSpacing);
    if (!std::isfinite(settings.requestedCoverageHalfExtents[axis]) ||
        settings.requestedCoverageHalfExtents[axis] <= 0.0f) {
      settings.requestedCoverageHalfExtents[axis] =
          kDefaultCoverageHalfExtents[axis];
    }
  }
  settings.spacingRatio = std::clamp(
      std::isfinite(settings.spacingRatio) ? settings.spacingRatio : 2.0f,
      1.25f, 4.0f);
  settings.scenePaddingCells = std::clamp(settings.scenePaddingCells, 1u, 8u);
  if (settings.authoredBounds.valid &&
      !validBounds(settings.authoredBounds.bounds)) {
    settings.authoredBounds.valid = false;
    settings.authoredBounds.complete = false;
  }
}

Result<DDGISceneFitSolution, DDGICoverageSolveError>
solveDDGISceneFit(const DDGISceneCoverageBounds &sceneBounds,
                  const DDGICoverageSettings &settings,
                  const DDGICoverageSolveLimits &limits) noexcept {
  using SolveResult = Result<DDGISceneFitSolution, DDGICoverageSolveError>;
  glm::vec3 size(0.0f);
  glm::vec3 center(0.0f);
  if (!sceneBounds.valid || !boundsFacts(sceneBounds.bounds, size, center)) {
    return SolveResult::makeError(
        {DDGICoverageLimit::SceneBoundsUnavailable, 0u});
  }
  if (!sceneBounds.complete) {
    return SolveResult::makeError(
        {DDGICoverageLimit::SceneBoundsIncomplete, 0u});
  }
  if (!enumInRange(settings.constraintPolicy,
                   DDGICoverageConstraintPolicy::RejectUnsatisfied) ||
      !finiteVector(settings.requestedNearSpacing) ||
      glm::any(
          glm::lessThanEqual(settings.requestedNearSpacing, glm::vec3(0.0f))) ||
      settings.scenePaddingCells < 1u || settings.scenePaddingCells > 8u ||
      1u + 2u * settings.scenePaddingCells >= kDDGIMaxProbeCountPerAxis ||
      limits.maxTextureExtent.x == 0u || limits.maxTextureExtent.y == 0u) {
    return SolveResult::makeError({DDGICoverageLimit::InvalidSettings, 0u});
  }
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    if (settings.requestedNearSpacing[axis] < kDDGIMinProbeSpacing ||
        settings.requestedNearSpacing[axis] > kDDGIMaxProbeSpacing) {
      return SolveResult::makeError({DDGICoverageLimit::InvalidSettings, axis});
    }
  }

  auto requestedCounts = fullCoverageCounts(size, settings.requestedNearSpacing,
                                            settings.scenePaddingCells);
  EvaluationResult requested =
      requestedCounts.hasValue()
          ? evaluateLayout(*requestedCounts, settings.requestedNearSpacing,
                           limits)
          : EvaluationResult::makeError(requestedCounts.error());
  if (requested.hasValue()) {
    return makeSolution(sceneBounds, settings, limits, *requested,
                        DDGICoverageLimit::None, center);
  }
  const DDGICoverageSolveError requestedFailure = requested.error();
  if (settings.constraintPolicy ==
      DDGICoverageConstraintPolicy::RejectUnsatisfied) {
    return SolveResult::makeError(requestedFailure);
  }

  if (settings.constraintPolicy ==
      DDGICoverageConstraintPolicy::PreserveCoverage) {
    float maximumScale = std::numeric_limits<float>::max();
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
      maximumScale =
          std::min(maximumScale,
                   kDDGIMaxProbeSpacing / settings.requestedNearSpacing[axis]);
    }
    EvaluationResult maximum =
        evaluateFullCoverage(size, settings.requestedNearSpacing, maximumScale,
                             settings.scenePaddingCells, limits);
    if (maximum.hasError()) {
      return SolveResult::makeError(maximum.error());
    }

    float failingScale = 1.0f;
    float passingScale = maximumScale;
    EvaluatedLayout best = *maximum;
    DDGICoverageSolveError bindingFailure = requestedFailure;
    for (uint32_t iteration = 0u; iteration < 48u; ++iteration) {
      const float scale = 0.5f * (failingScale + passingScale);
      EvaluationResult candidate =
          evaluateFullCoverage(size, settings.requestedNearSpacing, scale,
                               settings.scenePaddingCells, limits);
      if (candidate.hasValue()) {
        passingScale = scale;
        best = *candidate;
      } else {
        failingScale = scale;
        bindingFailure = candidate.error();
      }
    }
    return makeSolution(sceneBounds, settings, limits, best,
                        bindingFailure.limit, center);
  }

  const glm::uvec3 maximumCounts = maximumNearSpacingCounts(
      size, settings.requestedNearSpacing, settings.scenePaddingCells);
  EvaluationResult maximum =
      evaluateLayout(maximumCounts, settings.requestedNearSpacing, limits);
  if (maximum.hasValue()) {
    return makeSolution(sceneBounds, settings, limits, *maximum,
                        requestedFailure.limit, center);
  }
  EvaluationResult minimum = evaluateLayout(
      scaledInteriorCounts(maximumCounts, settings.scenePaddingCells, 0.0f),
      settings.requestedNearSpacing, limits);
  if (minimum.hasError()) {
    return SolveResult::makeError(minimum.error());
  }

  float passingScale = 0.0f;
  float failingScale = 1.0f;
  EvaluatedLayout best = *minimum;
  DDGICoverageSolveError bindingFailure = maximum.error();
  for (uint32_t iteration = 0u; iteration < 32u; ++iteration) {
    const float scale = 0.5f * (passingScale + failingScale);
    EvaluationResult candidate = evaluateLayout(
        scaledInteriorCounts(maximumCounts, settings.scenePaddingCells, scale),
        settings.requestedNearSpacing, limits);
    if (candidate.hasValue()) {
      passingScale = scale;
      best = *candidate;
    } else {
      failingScale = scale;
      bindingFailure = candidate.error();
    }
  }
  return makeSolution(sceneBounds, settings, limits, best, bindingFailure.limit,
                      center);
}

Result<DDGIClipmapSolution, DDGICoverageSolveError>
solveDDGIClipmaps(const DDGICoverageSettings &settings,
                  const DDGICoverageSolveLimits &limits) noexcept {
  using ClipmapResult = Result<DDGIClipmapSolution, DDGICoverageSolveError>;
  if (settings.cascadeCount == 0u ||
      settings.cascadeCount > kMaxDDGIClipmapCascades ||
      !finiteVector(settings.requestedNearSpacing) ||
      !finiteVector(settings.requestedCoverageHalfExtents) ||
      glm::any(
          glm::lessThanEqual(settings.requestedNearSpacing, glm::vec3(0.0f))) ||
      glm::any(glm::lessThanEqual(settings.requestedCoverageHalfExtents,
                                  glm::vec3(0.0f))) ||
      !std::isfinite(settings.spacingRatio) || settings.spacingRatio < 1.25f ||
      settings.spacingRatio > 4.0f || settings.transitionCells < 1u ||
      settings.transitionCells > 3u) {
    return ClipmapResult::makeError({DDGICoverageLimit::InvalidSettings, 0u});
  }
  const uint32_t probeCount = ddgiProbeCount(settings.cascadeProbeCounts);
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    if (settings.cascadeProbeCounts[axis] < kDDGIMinProbeCountPerAxis ||
        settings.cascadeProbeCounts[axis] > kDDGIMaxProbeCountPerAxis ||
        2u * settings.transitionCells >= settings.cascadeProbeCounts[axis]) {
      return ClipmapResult::makeError(
          {DDGICoverageLimit::ProbeCountAxis, axis});
    }
    if (settings.requestedNearSpacing[axis] < kDDGIMinProbeSpacing ||
        settings.requestedNearSpacing[axis] > kDDGIMaxProbeSpacing) {
      return ClipmapResult::makeError({DDGICoverageLimit::ProbeSpacing, axis});
    }
  }
  if (probeCount == 0u || probeCount > kDDGIMaxProbeCount) {
    return ClipmapResult::makeError({DDGICoverageLimit::TotalProbeCount, 0u});
  }

  const double outerRatio =
      std::pow(static_cast<double>(settings.spacingRatio),
               static_cast<double>(settings.cascadeCount - 1u));
  if (!std::isfinite(outerRatio)) {
    return ClipmapResult::makeError(
        {DDGICoverageLimit::ArithmeticOverflow, 0u});
  }
  const glm::vec3 baseOuterSpacing =
      settings.requestedNearSpacing * static_cast<float>(outerRatio);
  const glm::vec3 baseOuterFade =
      0.5f * glm::vec3(settings.cascadeProbeCounts - glm::uvec3(3u)) *
      baseOuterSpacing;
  float spacingScale = 1.0f;
  bool requestedCoverageFits = true;
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    if (!std::isfinite(baseOuterFade[axis]) || baseOuterFade[axis] <= 0.0f) {
      requestedCoverageFits = false;
      spacingScale = std::numeric_limits<float>::infinity();
      break;
    }
    const float requiredScale =
        settings.requestedCoverageHalfExtents[axis] / baseOuterFade[axis];
    requestedCoverageFits &= requiredScale <= 1.0f;
    spacingScale = std::max(spacingScale, requiredScale);
  }
  DDGICoverageLimit limitingConstraint = DDGICoverageLimit::None;
  if (!requestedCoverageFits) {
    limitingConstraint = DDGICoverageLimit::CoverageExtents;
    if (settings.constraintPolicy ==
        DDGICoverageConstraintPolicy::RejectUnsatisfied) {
      return ClipmapResult::makeError({DDGICoverageLimit::CoverageExtents, 0u});
    }
    if (settings.constraintPolicy ==
        DDGICoverageConstraintPolicy::PreserveNearSpacing) {
      spacingScale = 1.0f;
    }
  }
  if (!std::isfinite(spacingScale)) {
    return ClipmapResult::makeError({DDGICoverageLimit::CoverageExtents, 0u});
  }
  const glm::vec3 achievedNearSpacing =
      settings.requestedNearSpacing * spacingScale;
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    if (!std::isfinite(achievedNearSpacing[axis]) ||
        achievedNearSpacing[axis] > kDDGIMaxProbeSpacing) {
      return ClipmapResult::makeError({DDGICoverageLimit::ProbeSpacing, axis});
    }
  }

  DDGIClipmapSolution solution{
      .requestedNearSpacing = settings.requestedNearSpacing,
      .achievedNearSpacing = achievedNearSpacing,
      .requestedCoverageHalfExtents = settings.requestedCoverageHalfExtents,
      .cascadeCount = settings.cascadeCount,
      .limitingConstraint = limitingConstraint,
  };
  DDGICoverageSolveLimits aggregateLimits = limits;
  for (uint32_t cascade = 0u; cascade < settings.cascadeCount; ++cascade) {
    const double ratio =
        std::pow(static_cast<double>(settings.spacingRatio), cascade);
    if (!std::isfinite(ratio)) {
      return ClipmapResult::makeError(
          {DDGICoverageLimit::ArithmeticOverflow, 0u});
    }
    const glm::vec3 spacing = achievedNearSpacing * static_cast<float>(ratio);
    auto evaluated =
        evaluateLayout(settings.cascadeProbeCounts, spacing, aggregateLimits);
    if (evaluated.hasError()) {
      return ClipmapResult::makeError(evaluated.error());
    }
    const glm::vec3 halfExtents =
        0.5f * glm::vec3(settings.cascadeProbeCounts - glm::uvec3(1u)) *
        spacing;
    const glm::vec3 fadeEnd = glm::max(halfExtents - spacing, glm::vec3(0.0f));
    const glm::vec3 fadeStart = glm::max(
        fadeEnd - static_cast<float>(settings.transitionCells) * spacing,
        glm::vec3(0.0f));
    const uint64_t updateCapacity =
        std::max<uint64_t>(limits.maxProbeUpdatesPerFrame, 1u);
    solution.cascades[cascade] = DDGIClipmapCascadeSolution{
        .probeCounts = settings.cascadeProbeCounts,
        .spacing = spacing,
        .probeCenterHalfExtents = halfExtents,
        .fadeStartHalfExtents = fadeStart,
        .fadeEndHalfExtents = fadeEnd,
        .irradianceAtlas = evaluated->irradianceAtlas,
        .distanceAtlas = evaluated->distanceAtlas,
        .memory = evaluated->memory,
        .probeCount = evaluated->probeCount,
        .estimatedMinimumRefreshFrames = static_cast<uint32_t>(
            (static_cast<uint64_t>(evaluated->probeCount) + updateCapacity -
             1u) /
            updateCapacity),
    };
    solution.persistentBytes += evaluated->memory.persistentBytes;
    if (!checkedAdd(aggregateLimits.otherPersistentBytes,
                    evaluated->memory.persistentBytes,
                    aggregateLimits.otherPersistentBytes)) {
      return ClipmapResult::makeError(
          {DDGICoverageLimit::ArithmeticOverflow, 0u});
    }
  }
  solution.achievedCoverageHalfExtents =
      solution.cascades[settings.cascadeCount - 1u].fadeEndHalfExtents;
  solution.fullCoverage =
      glm::all(glm::greaterThanEqual(solution.achievedCoverageHalfExtents,
                                     solution.requestedCoverageHalfExtents));
  return ClipmapResult::makeResult(solution);
}

Result<bool, DDGICoverageSolveError>
resolveDDGIEffectiveVolumePlan(const DDGICoverageResolveInput &input,
                               DDGIEffectiveVolumePlan &out) {
  using ResolveResult = Result<bool, DDGICoverageSolveError>;
  out.clear();
  DDGICoverageSettings settings = input.settings;
  sanitizeDDGICoverageSettings(settings);
  out.mode = settings.mode;
  out.sceneId = input.sceneId;
  out.coverageGeneration = input.coverageGeneration;

  std::pmr::memory_resource *scratch = input.scratch != nullptr
                                           ? input.scratch
                                           : std::pmr::get_default_resource();
  std::pmr::vector<DDGIEffectiveVolume> candidates(scratch);
  candidates.reserve(input.authoredVolumes.size() + kMaxDDGIClipmapCascades +
                     1u);
  DDGICoverageSolveError firstFailure{};
  const auto noteFailure = [&out,
                            &firstFailure](const DDGIEffectiveVolumeKey &key,
                                           DDGICoverageSolveError error) {
    out.failedKeys.push_back(key);
    if (firstFailure.limit == DDGICoverageLimit::None) {
      firstFailure = error;
    }
  };

  const bool includeAuthored = settings.mode == DDGICoverageMode::Manual ||
                               settings.includeAuthoredVolumes;
  if (includeAuthored) {
    for (const RenderDDGIVolume &volume : input.authoredVolumes) {
      const DDGIEffectiveVolumeKey key{
          .kind = DDGIEffectiveVolumeKind::Authored,
          .authoredId = volume.id,
          .sceneId = input.sceneId,
      };
      if (!isRigidDDGITransform(volume.worldFromLocal)) {
        noteFailure(key, {DDGICoverageLimit::InvalidSettings, 0u});
        continue;
      }
      auto evaluated =
          evaluateLayout(volume.probeCounts, volume.probeSpacing, input.limits);
      if (evaluated.hasError()) {
        noteFailure(key, evaluated.error());
        continue;
      }
      candidates.push_back(DDGIEffectiveVolume{
          .key = key,
          .authoredId = volume.id,
          .name = volume.name,
          .probeCounts = volume.probeCounts,
          .probeSpacing = volume.probeSpacing,
          .worldFromLocal = volume.worldFromLocal,
          .cameraCell =
              volume.mode == DDGIVolumeMode::CameraTracked
                  ? ddgiCameraCell(
                        glm::vec3(glm::inverse(volume.worldFromLocal) *
                                  glm::vec4(input.cameraWorldPosition, 1.0f)),
                        volume.probeSpacing)
                  : glm::ivec3(0),
          .continuousCameraLocal =
              glm::vec3(glm::inverse(volume.worldFromLocal) *
                        glm::vec4(input.cameraWorldPosition, 1.0f)),
          .probeCenterHalfExtents =
              0.5f * glm::vec3(volume.probeCounts - glm::uvec3(1u)) *
              volume.probeSpacing,
          .fadeStartHalfExtents = glm::vec3(0.0f),
          .fadeEndHalfExtents = 0.5f *
                                glm::vec3(volume.probeCounts - glm::uvec3(1u)) *
                                volume.probeSpacing,
          .irradianceAtlas = evaluated->irradianceAtlas,
          .distanceAtlas = evaluated->distanceAtlas,
          .memory = evaluated->memory,
          .blendDistance = volume.blendDistance,
          .maxRayDistance = volume.maxRayDistance,
          .priority = volume.priority,
          .mode = volume.mode,
          .tier = DDGIEffectiveTier::AuthoredOverride,
          .probeCount = evaluated->probeCount,
          .requestedCoverageAchieved = true,
      });
    }
  }

  const bool needsClipmaps =
      settings.mode == DDGICoverageMode::CameraClipmaps ||
      settings.mode == DDGICoverageMode::Hybrid;
  if (needsClipmaps) {
    auto clipmaps = solveDDGIClipmaps(settings, input.limits);
    if (clipmaps.hasError()) {
      out.error = clipmaps.error();
      return ResolveResult::makeError(out.error);
    }
    out.clipmaps = *clipmaps;
    static constexpr std::array<std::string_view, kMaxDDGIClipmapCascades>
        kNames{"DDGI Clipmap 0", "DDGI Clipmap 1", "DDGI Clipmap 2",
               "DDGI Clipmap 3"};
    for (uint32_t cascade = 0u; cascade < clipmaps->cascadeCount; ++cascade) {
      const DDGIClipmapCascadeSolution &solved = clipmaps->cascades[cascade];
      candidates.push_back(DDGIEffectiveVolume{
          .key = {.kind = DDGIEffectiveVolumeKind::ClipmapCascade,
                  .sceneId = input.sceneId,
                  .coverageGeneration = input.coverageGeneration,
                  .generatedIndex = cascade},
          .name = kNames[cascade],
          .probeCounts = solved.probeCounts,
          .probeSpacing = solved.spacing,
          .worldFromLocal = glm::mat4(1.0f),
          .cameraCell =
              ddgiCameraCell(input.cameraWorldPosition, solved.spacing),
          .continuousCameraLocal = input.cameraWorldPosition,
          .probeCenterHalfExtents = solved.probeCenterHalfExtents,
          .fadeStartHalfExtents = solved.fadeStartHalfExtents,
          .fadeEndHalfExtents = solved.fadeEndHalfExtents,
          .irradianceAtlas = solved.irradianceAtlas,
          .distanceAtlas = solved.distanceAtlas,
          .memory = solved.memory,
          .blendDistance = 0.0f,
          .maxRayDistance = 20.0f,
          .priority =
              settings.generatedPriority - static_cast<int32_t>(cascade),
          .mode = DDGIVolumeMode::CameraTracked,
          .tier = static_cast<DDGIEffectiveTier>(
              static_cast<uint8_t>(DDGIEffectiveTier::Clipmap0) + cascade),
          .cascadeIndex = cascade,
          .transitionCells = settings.transitionCells,
          .probeCount = solved.probeCount,
          .requestedCoverageAchieved = clipmaps->fullCoverage,
      });
    }
  }

  const bool needsSceneFit = settings.mode == DDGICoverageMode::SceneFit ||
                             settings.mode == DDGICoverageMode::Hybrid;
  if (needsSceneFit) {
    const DDGISceneCoverageBounds selectedBounds =
        settings.sceneBoundsSource == DDGISceneBoundsSource::Authored
            ? settings.authoredBounds
            : input.sceneBounds;
    out.sceneBoundsGeneration = selectedBounds.generation;
    DDGICoverageSettings sceneFitSettings = settings;
    if (settings.mode == DDGICoverageMode::Hybrid &&
        out.clipmaps.cascadeCount != 0u) {
      sceneFitSettings.requestedNearSpacing =
          out.clipmaps.cascades[out.clipmaps.cascadeCount - 1u].spacing;
    }
    auto sceneFit =
        solveDDGISceneFit(selectedBounds, sceneFitSettings, input.limits);
    if (sceneFit.hasError()) {
      out.error = sceneFit.error();
      return ResolveResult::makeError(out.error);
    }
    out.sceneFit = *sceneFit;
    candidates.push_back(DDGIEffectiveVolume{
        .key = {.kind = DDGIEffectiveVolumeKind::SceneFit,
                .sceneId = input.sceneId,
                .coverageGeneration = input.coverageGeneration,
                .generatedIndex = 0u},
        .name = "DDGI Scene Fit",
        .probeCounts = sceneFit->probeCounts,
        .probeSpacing = sceneFit->achievedSpacing,
        .worldFromLocal =
            glm::translate(glm::mat4(1.0f), sceneFit->worldCenter),
        .probeCenterHalfExtents = sceneFit->probeCenterHalfExtents,
        .fadeStartHalfExtents = sceneFit->interiorHalfExtents,
        .fadeEndHalfExtents = sceneFit->interiorHalfExtents,
        .irradianceAtlas = sceneFit->irradianceAtlas,
        .distanceAtlas = sceneFit->distanceAtlas,
        .memory = sceneFit->memory,
        .blendDistance =
            static_cast<float>(settings.scenePaddingCells) *
            std::min({sceneFit->achievedSpacing.x, sceneFit->achievedSpacing.y,
                      sceneFit->achievedSpacing.z}),
        .maxRayDistance = 20.0f,
        .priority =
            settings.generatedPriority -
            static_cast<int32_t>(needsClipmaps ? settings.cascadeCount : 0u),
        .tier = DDGIEffectiveTier::SceneFitCoarse,
        .probeCount = sceneFit->probeCount,
        .requestedCoverageAchieved = sceneFit->fullCoverage,
    });
  }

  std::ranges::sort(candidates, [](const DDGIEffectiveVolume &left,
                                   const DDGIEffectiveVolume &right) {
    if (left.priority != right.priority) {
      return left.priority > right.priority;
    }
    const uint32_t leftDensity =
        left.key.kind == DDGIEffectiveVolumeKind::ClipmapCascade
            ? left.cascadeIndex
            : UINT32_MAX;
    const uint32_t rightDensity =
        right.key.kind == DDGIEffectiveVolumeKind::ClipmapCascade
            ? right.cascadeIndex
            : UINT32_MAX;
    if (leftDensity != rightDensity) {
      return leftDensity < rightDensity;
    }
    return ddgiEffectiveVolumeKeyLess(left.key, right.key);
  });
  out.candidateCount = static_cast<uint32_t>(std::min<size_t>(
      candidates.size(), std::numeric_limits<uint32_t>::max()));
  for (const DDGIEffectiveVolume &candidate : candidates) {
    if (out.volumeCount == kMaxDDGIEffectiveVolumes) {
      out.omittedKeys.push_back(candidate.key);
      continue;
    }
    uint64_t committedBytes = 0u;
    uint64_t peakBytes = 0u;
    if (!checkedAdd(input.limits.otherPersistentBytes, out.persistentBytes,
                    committedBytes) ||
        !checkedAdd(committedBytes, candidate.memory.persistentBytes,
                    committedBytes)) {
      noteFailure(candidate.key, {DDGICoverageLimit::ArithmeticOverflow, 0u});
      continue;
    }
    if (committedBytes > input.limits.maxPersistentBytes) {
      noteFailure(candidate.key, {DDGICoverageLimit::PersistentMemory, 0u});
      continue;
    }
    if (!checkedAdd(committedBytes, input.limits.retainedReplacementBytes,
                    peakBytes)) {
      noteFailure(candidate.key, {DDGICoverageLimit::ArithmeticOverflow, 0u});
      continue;
    }
    if (peakBytes > input.limits.maxReplacementPeakBytes) {
      noteFailure(candidate.key,
                  {DDGICoverageLimit::ReplacementPeakMemory, 0u});
      continue;
    }
    out.volumes[out.volumeCount++] = candidate;
    out.persistentBytes += candidate.memory.persistentBytes;
    if (candidate.key.kind == DDGIEffectiveVolumeKind::SceneFit) {
      out.fullSceneCoverage = candidate.requestedCoverageAchieved;
    }
  }
  out.overflowed = !out.omittedKeys.empty();
  uint64_t replacementPeak = 0u;
  if (!checkedAdd(input.limits.otherPersistentBytes, out.persistentBytes,
                  replacementPeak) ||
      !checkedAdd(replacementPeak, input.limits.retainedReplacementBytes,
                  replacementPeak)) {
    out.error = {DDGICoverageLimit::ArithmeticOverflow, 0u};
    return ResolveResult::makeError(out.error);
  }
  out.replacementPeakBytes = replacementPeak;
  if (settings.constraintPolicy ==
          DDGICoverageConstraintPolicy::RejectUnsatisfied &&
      (out.overflowed || !out.failedKeys.empty())) {
    out.error =
        out.overflowed
            ? DDGICoverageSolveError{DDGICoverageLimit::EffectiveVolumeCapacity,
                                     0u}
            : firstFailure;
    return ResolveResult::makeError(out.error);
  }
  out.error = firstFailure;
  out.ready = true;
  return ResolveResult::makeResult(true);
}

bool ddgiBoundsContain(const BoundingBox &outer,
                       const BoundingBox &inner) noexcept {
  if (!validBounds(outer) || !validBounds(inner)) {
    return false;
  }
  constexpr float kTolerance = 1.0e-4f;
  return glm::all(glm::lessThanEqual(outer.min_ - glm::vec3(kTolerance),
                                     inner.min_)) &&
         glm::all(glm::greaterThanEqual(outer.max_ + glm::vec3(kTolerance),
                                        inner.max_));
}

} // namespace nuri
