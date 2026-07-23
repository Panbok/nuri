#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/ddgi/ddgi_atlas.h"
#include "nuri/scene/ddgi_coverage_bounds.h"
#include "nuri/scene/render_scene.h"
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory_resource>
#include <span>
#include <string_view>
#include <vector>

namespace nuri {

inline constexpr uint32_t kDDGICoverageSchemaVersion = 1u;

enum class DDGICoverageMode : uint8_t {
  Manual = 0,
  SceneFit,
  CameraClipmaps,
  Hybrid,
};

enum class DDGICoverageConstraintPolicy : uint8_t {
  PreserveCoverage = 0,
  PreserveNearSpacing,
  RejectUnsatisfied,
};

enum class DDGISceneBoundsSource : uint8_t {
  ActivationSnapshot = 0,
  StaticRayTracingGeometry,
  Authored,
};

struct DDGICoverageSettings {
  DDGICoverageMode mode = DDGICoverageMode::Manual;
  DDGICoverageConstraintPolicy constraintPolicy =
      DDGICoverageConstraintPolicy::PreserveCoverage;
  DDGISceneBoundsSource sceneBoundsSource =
      DDGISceneBoundsSource::ActivationSnapshot;
  uint32_t cascadeCount = 3u;
  glm::uvec3 cascadeProbeCounts{20u, 12u, 20u};
  glm::vec3 requestedNearSpacing{1.5f};
  float spacingRatio = 2.0f;
  glm::vec3 requestedCoverageHalfExtents{55.0f, 30.0f, 55.0f};
  uint32_t scenePaddingCells = 2u;
  uint32_t transitionCells = 1u;
  int32_t generatedPriority = -1024;
  bool includeAuthoredVolumes = true;
  bool autoRefitOnTopologyChange = false;
  DDGISceneCoverageBounds authoredBounds{};
  constexpr bool operator==(const DDGICoverageSettings &) const = default;
};

struct DDGICoverageSolveError {
  DDGICoverageLimit limit = DDGICoverageLimit::None;
  uint32_t axis = 0u;
  constexpr bool operator==(const DDGICoverageSolveError &) const = default;
};

struct DDGICoverageSolveLimits {
  glm::uvec2 maxTextureExtent{16'384u};
  uint64_t maxPersistentBytes = 256ull * 1024ull * 1024ull;
  uint64_t maxReplacementPeakBytes = 512ull * 1024ull * 1024ull;
  uint64_t otherPersistentBytes = 0u;
  uint64_t retainedReplacementBytes = 0u;
  uint32_t maxProbeUpdatesPerFrame = 512u;
};

struct DDGISceneFitSolution {
  BoundingBox requestedBounds{};
  BoundingBox achievedInteriorBounds{};
  glm::vec3 worldCenter{0.0f};
  glm::uvec3 probeCounts{2u};
  glm::vec3 requestedSpacing{1.0f};
  glm::vec3 achievedSpacing{1.0f};
  glm::vec3 probeCenterHalfExtents{0.5f};
  glm::vec3 interiorHalfExtents{0.0f};
  DDGIAtlasLayout irradianceAtlas{};
  DDGIAtlasLayout distanceAtlas{};
  DDGIMemoryEstimate memory{};
  uint32_t probeCount = 0u;
  uint32_t estimatedMinimumRefreshFrames = 0u;
  float requestedVolumeCoverage = 0.0f;
  bool fullCoverage = false;
  DDGICoverageLimit limitingConstraint = DDGICoverageLimit::None;
};

enum class DDGIEffectiveVolumeKind : uint8_t {
  Authored = 0,
  SceneFit,
  ClipmapCascade,
};

enum class DDGIEffectiveTier : uint8_t {
  AuthoredOverride = 0,
  Clipmap0,
  Clipmap1,
  Clipmap2,
  Clipmap3,
  SceneFitCoarse,
};

struct DDGIEffectiveVolumeKey {
  DDGIEffectiveVolumeKind kind = DDGIEffectiveVolumeKind::Authored;
  DDGIVolumeId authoredId = kInvalidDDGIVolumeId;
  uint64_t sceneId = 0u;
  uint64_t coverageGeneration = 0u;
  uint32_t generatedIndex = 0u;
  constexpr bool operator==(const DDGIEffectiveVolumeKey &) const = default;
};

struct DDGIClipmapCascadeSolution {
  glm::uvec3 probeCounts{2u};
  glm::vec3 spacing{1.0f};
  glm::vec3 probeCenterHalfExtents{0.5f};
  glm::vec3 fadeStartHalfExtents{0.0f};
  glm::vec3 fadeEndHalfExtents{0.0f};
  DDGIAtlasLayout irradianceAtlas{};
  DDGIAtlasLayout distanceAtlas{};
  DDGIMemoryEstimate memory{};
  uint32_t probeCount = 0u;
  uint32_t estimatedMinimumRefreshFrames = 0u;
};

struct DDGIClipmapSolution {
  std::array<DDGIClipmapCascadeSolution, kMaxDDGIClipmapCascades> cascades{};
  glm::vec3 requestedNearSpacing{1.0f};
  glm::vec3 achievedNearSpacing{1.0f};
  glm::vec3 requestedCoverageHalfExtents{0.0f};
  glm::vec3 achievedCoverageHalfExtents{0.0f};
  uint32_t cascadeCount = 0u;
  uint64_t persistentBytes = 0u;
  bool fullCoverage = false;
  DDGICoverageLimit limitingConstraint = DDGICoverageLimit::None;
};

struct DDGIEffectiveVolume {
  DDGIEffectiveVolumeKey key{};
  DDGIVolumeId authoredId = kInvalidDDGIVolumeId;
  std::string_view name{};
  glm::uvec3 probeCounts{2u};
  glm::vec3 probeSpacing{1.0f};
  glm::mat4 worldFromLocal{1.0f};
  glm::ivec3 cameraCell{0};
  glm::vec3 continuousCameraLocal{0.0f};
  glm::vec3 probeCenterHalfExtents{0.5f};
  glm::vec3 fadeStartHalfExtents{0.0f};
  glm::vec3 fadeEndHalfExtents{0.0f};
  DDGIAtlasLayout irradianceAtlas{};
  DDGIAtlasLayout distanceAtlas{};
  DDGIMemoryEstimate memory{};
  float blendDistance = 0.0f;
  float maxRayDistance = 20.0f;
  int32_t priority = 0;
  DDGIVolumeMode mode = DDGIVolumeMode::Authored;
  DDGIEffectiveTier tier = DDGIEffectiveTier::AuthoredOverride;
  uint32_t cascadeIndex = UINT32_MAX;
  uint32_t transitionCells = 0u;
  uint32_t probeCount = 0u;
  bool requestedCoverageAchieved = false;
};

struct DDGIRedundancyAnalysis {
  bool fullyCovered = false;
  bool densityRedundant = false;
  bool fullyRedundant = false;
};

struct DDGIEffectiveVolumePlan {
  explicit DDGIEffectiveVolumePlan(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : omittedKeys(memory != nullptr ? memory
                                      : std::pmr::get_default_resource()),
        failedKeys(memory != nullptr ? memory
                                     : std::pmr::get_default_resource()) {}

  void clear() noexcept;
  [[nodiscard]] std::span<const DDGIEffectiveVolume> activeVolumes() const {
    return std::span<const DDGIEffectiveVolume>(volumes.data(), volumeCount);
  }

  std::array<DDGIEffectiveVolume, kMaxDDGIEffectiveVolumes> volumes{};
  std::pmr::vector<DDGIEffectiveVolumeKey> omittedKeys;
  std::pmr::vector<DDGIEffectiveVolumeKey> failedKeys;
  DDGISceneFitSolution sceneFit{};
  DDGIClipmapSolution clipmaps{};
  DDGICoverageSolveError error{};
  DDGICoverageMode mode = DDGICoverageMode::Manual;
  uint64_t sceneId = 0u;
  uint64_t coverageGeneration = 0u;
  uint64_t sceneBoundsGeneration = 0u;
  uint64_t persistentBytes = 0u;
  uint64_t replacementPeakBytes = 0u;
  uint32_t volumeCount = 0u;
  uint32_t candidateCount = 0u;
  bool ready = false;
  bool overflowed = false;
  bool fullSceneCoverage = false;
};

struct DDGICoverageResolveInput {
  uint64_t sceneId = 0u;
  uint64_t coverageGeneration = 0u;
  DDGISceneCoverageBounds sceneBounds{};
  std::span<const RenderDDGIVolume> authoredVolumes{};
  glm::vec3 cameraWorldPosition{0.0f};
  DDGICoverageSettings settings{};
  DDGICoverageSolveLimits limits{};
  std::pmr::memory_resource *scratch = std::pmr::get_default_resource();
};

NURI_API void
sanitizeDDGICoverageSettings(DDGICoverageSettings &settings) noexcept;

[[nodiscard]] NURI_API Result<DDGISceneFitSolution, DDGICoverageSolveError>
solveDDGISceneFit(const DDGISceneCoverageBounds &sceneBounds,
                  const DDGICoverageSettings &settings,
                  const DDGICoverageSolveLimits &limits) noexcept;

[[nodiscard]] NURI_API Result<DDGIClipmapSolution, DDGICoverageSolveError>
solveDDGIClipmaps(const DDGICoverageSettings &settings,
                  const DDGICoverageSolveLimits &limits) noexcept;

[[nodiscard]] NURI_API Result<bool, DDGICoverageSolveError>
resolveDDGIEffectiveVolumePlan(const DDGICoverageResolveInput &input,
                               DDGIEffectiveVolumePlan &out);

[[nodiscard]] NURI_API bool
ddgiEffectiveVolumeKeyLess(const DDGIEffectiveVolumeKey &left,
                           const DDGIEffectiveVolumeKey &right) noexcept;

[[nodiscard]] NURI_API uint64_t
ddgiEffectiveVolumeKeyHash(const DDGIEffectiveVolumeKey &key) noexcept;
[[nodiscard]] NURI_API DDGIRedundancyAnalysis
analyzeDDGIVolumeRedundancy(const DDGIEffectiveVolume &authored,
                            const DDGIEffectiveVolume &generated) noexcept;

[[nodiscard]] NURI_API bool
ddgiBoundsContain(const BoundingBox &outer, const BoundingBox &inner) noexcept;

} // namespace nuri
