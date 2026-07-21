#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/scene/ddgi_volume.h"
#include "nuri/scene/scene_handles.h"
#include "nuri/gfx/gpu_types.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>

namespace nuri {

inline constexpr uint32_t kDDGILayoutVersion = 1u;
inline constexpr uint32_t kMaxDDGIVolumes = 4u;
inline constexpr float kDDGIIrradianceGamma = 5.0f;
inline constexpr float kDDGIVisibilityFloor = 0.05f;
inline constexpr float kDDGIChebyshevExponent = 3.0f;
inline constexpr float kDDGIProbeWeightCrushThreshold = 0.2f;
inline constexpr uint32_t kDDGIMaxDiagnosticRays = 1024u;
inline constexpr uint32_t kDDGIMaxDiagnosticCandidateEventsPerRay = 8u;

enum class DDGIDebugView : uint8_t {
  None = 0,
  DiffuseIndirect,
  VolumeId,
  ProbeWeights,
  Confidence,
  Visibility,
  Irradiance,
  DistanceMean,
  DistanceVariance,
  Classification,
  RelocationOffset,
  UpdateAge,
  LeakRisk,
};

struct DDGIAtlasLayout {
  glm::uvec2 tileExtent{0u};
  uint32_t columns = 0u;
  uint32_t rows = 0u;
  glm::uvec2 textureExtent{0u};
  constexpr bool operator==(const DDGIAtlasLayout &) const = default;
};

struct DDGIVolumeLayout {
  uint32_t version = kDDGILayoutVersion;
  DDGIVolumeId id = kInvalidDDGIVolumeId;
  glm::uvec3 probeCounts{2u};
  glm::vec3 probeSpacing{1.0f};
  glm::vec3 probeCenterHalfExtents{0.5f};
  glm::mat4 worldFromLocal{1.0f};
  glm::mat4 localFromWorld{1.0f};
  glm::ivec3 cameraCell{0};
  glm::uvec3 ringOrigin{0u};
  DDGIAtlasLayout irradianceAtlas{};
  DDGIAtlasLayout distanceAtlas{};
  uint64_t generation = 0u;
};

struct DDGIScrollPlan {
  glm::ivec3 cameraCell{0};
  glm::uvec3 ringOrigin{0u};
  glm::ivec3 cellDelta{0};
  bool changed = false;
  bool fullInvalidation = false;
};

enum class DDGIProbeState : uint8_t {
  Uninitialized = 0,
  Off,
  Sleeping,
  NewlyAwake,
  Awake,
  NewlyVigilant,
  Vigilant,
};

enum class DDGIRayResultKind : uint8_t {
  Miss = 0,
  FrontHit,
  BackfaceContainment,
  CandidateOverflow,
};

enum class DDGIFallbackReason : uint8_t {
  None = 0,
  Disabled,
  Unsupported,
  NoVolumes,
  RayTracingSceneWarming,
  VolumeResourcesWarming,
  ShaderUnavailable,
  AllocationFailed,
};

enum class DDGIVolumeFailureReason : uint8_t {
  None = 0,
  AtlasPacking,
  InvalidLayout,
  PersistentMemoryLimit,
  PeakMemoryLimit,
  IrradianceAllocation,
  DistanceAllocation,
  ProbeStateAllocation,
};

struct DDGIFrameMetrics {
  uint32_t activeVolumes = 0u;
  uint32_t readyVolumes = 0u;
  uint32_t totalProbes = 0u;
  uint32_t vigilantProbes = 0u;
  uint32_t uninitializedProbes = 0u;
  uint32_t offProbes = 0u;
  uint32_t sleepingProbes = 0u;
  uint32_t newlyAwakeProbes = 0u;
  uint32_t awakeProbes = 0u;
  uint32_t newlyVigilantProbes = 0u;
  uint32_t relocatedProbes = 0u;
  uint32_t probeStateReadbackAvailable = 0u;
  uint32_t updatedProbes = 0u;
  uint32_t primaryQueries = 0u;
  uint32_t secondaryQueriesReserved = 0u;
  uint32_t secondaryQueriesUnused = 0u;
  uint32_t secondaryQueries = 0u;
  uint32_t primaryCandidateIntersections = 0u;
  uint32_t secondaryCandidateIntersections = 0u;
  uint32_t alphaCandidateRejections = 0u;
  uint32_t backfaceCandidateRejections = 0u;
  uint32_t candidateOverflows = 0u;
  uint32_t localLightTruncations = 0u;
  uint32_t rayQueryCapacity = 0u;
  uint32_t probeUpdateCapacity = 0u;
  uint32_t resetCount = 0u;
  uint32_t scrollCount = 0u;
  uint32_t invalidatedProbes = 0u;
  uint32_t failedVolumes = 0u;
  uint32_t historyReady = 0u;
  uint32_t irradianceResponseRemaining = 0u;
  uint32_t distanceResponseRemaining = 0u;
  uint32_t inspectionAvailable = 0u;
  uint32_t inspectionValid = 0u;
  uint32_t inspectionRayCount = 0u;
  uint32_t inspectionHitCount = 0u;
  uint32_t inspectionMissCount = 0u;
  uint32_t inspectionCandidateOverflows = 0u;
  uint32_t inspectionEventOverflows = 0u;
  uint32_t skyFallbackActive = 1u;
  uint64_t persistentBytes = 0u;
  uint64_t frameBatchBytes = 0u;
  uint64_t submittedSequence = 0u;
  uint64_t layoutGeneration = 0u;
  uint64_t resourceGeneration = 0u;
  uint64_t deviceEpoch = 0u;
  uint64_t consumedResetEpoch = 0u;
  uint64_t consumedForceUpdateEpoch = 0u;
  float gpuTimeMs = 0.0f;
  float traceGpuTimeMs = 0.0f;
  float updateGpuTimeMs = 0.0f;
  float relocateClassifyGpuTimeMs = 0.0f;
  uint64_t gpuTimingSourceFrameIndex = UINT64_MAX;
  uint32_t gpuTimingAvailable = 0u;
  uint32_t traceGpuTimingAvailable = 0u;
  uint32_t updateGpuTimingAvailable = 0u;
  uint32_t relocateClassifyGpuTimingAvailable = 0u;
  float maxRelocation = 0.0f;
  DDGIFallbackReason fallbackReason = DDGIFallbackReason::Disabled;
  DDGIVolumeFailureReason volumeFailureReason =
      DDGIVolumeFailureReason::None;
  DDGIDebugView debugView = DDGIDebugView::None;
};

struct DDGIVolumeBlendWeights {
  float first = 0.0f;
  float second = 0.0f;
  float sky = 1.0f;
};

struct alignas(16) DDGIProbeStateGpuData {
  glm::vec4 relocation{0.0f};
  glm::uvec4 stateAgeFlags{static_cast<uint32_t>(DDGIProbeState::Vigilant),
                           0u, 0u, 0u};
};

struct alignas(16) DDGIVolumeGpuData {
  glm::mat4 worldFromLocal{1.0f};
  glm::mat4 localFromWorld{1.0f};
  uint64_t probeStateBufferAddress = 0u;
  uint32_t resourceFlags = 0u;
  uint32_t reserved0 = 0u;
  glm::vec4 probeSpacingAndBias{1.0f, 1.0f, 1.0f, 0.3f};
  glm::vec4 centerHalfExtentsAndMaxDistance{0.5f, 0.5f, 0.5f, 20.0f};
  glm::uvec4 probeCountsAndCount{2u, 2u, 2u, 8u};
  glm::uvec4 irradianceAtlas{kInvalidTextureBindlessIndex, 0u, 0u, 0u};
  glm::uvec4 distanceAtlas{kInvalidTextureBindlessIndex, 0u, 0u, 0u};
  glm::uvec4 ringOriginAndFlags{0u};
  glm::uvec4 generations{0u};
};

struct alignas(16) DDGIFrameGpuData {
  glm::uvec4 activeCountDebugFlagsSampler{0u};
  std::array<DDGIVolumeGpuData, kMaxDDGIVolumes> volumes{};
};

struct alignas(16) DDGIRayResultGpuData {
  glm::vec4 radianceAndDistance{0.0f};
  glm::uvec4 metadata{0u};
};

struct alignas(16) DDGITraceCountersGpuData {
  uint32_t primaryCandidateIntersections = 0u;
  uint32_t secondaryQueries = 0u;
  uint32_t secondaryCandidateIntersections = 0u;
  uint32_t alphaCandidateRejections = 0u;
  uint32_t backfaceCandidateRejections = 0u;
  uint32_t candidateOverflows = 0u;
  uint32_t localLightTruncations = 0u;
  uint32_t reserved = 0u;
};

struct alignas(16) DDGIDiagnosticHeaderGpuData {
  glm::uvec4 requestAndSelection{0u};
  glm::uvec4 counts0{0u};
  glm::uvec4 counts1{0u};
  glm::uvec4 identity0{0u};
  glm::uvec4 identity1{0u};
  glm::uvec4 identity2{0u};
  glm::vec4 nominalWorldPosition{0.0f};
  glm::vec4 relocatedWorldPosition{0.0f};
  glm::vec4 irradianceDistanceMean{0.0f};
  glm::vec4 distanceSecondReserved{0.0f};
};

struct alignas(16) DDGIDiagnosticRayGpuData {
  glm::vec4 originAndDistance{0.0f};
  glm::vec4 directionAndKind{0.0f};
};

struct alignas(16) DDGIDiagnosticEventGpuData {
  glm::uvec4 metadata{0u};
  glm::vec4 distanceBarycentrics{0.0f};
};

inline constexpr size_t kDDGIDiagnosticBufferBytes =
    sizeof(DDGIDiagnosticHeaderGpuData) +
    kDDGIMaxDiagnosticRays * sizeof(DDGIDiagnosticRayGpuData) +
    kDDGIMaxDiagnosticRays *
        (kDDGIMaxDiagnosticCandidateEventsPerRay + 1u) *
        sizeof(DDGIDiagnosticEventGpuData);

static_assert(sizeof(DDGIProbeStateGpuData) == 32u);
static_assert(sizeof(DDGIVolumeGpuData) == 256u);
static_assert(sizeof(DDGIFrameGpuData) == 1040u);
static_assert(sizeof(DDGIRayResultGpuData) == 32u);
static_assert(sizeof(DDGITraceCountersGpuData) == 32u);
static_assert(sizeof(DDGIDiagnosticHeaderGpuData) == 160u);
static_assert(sizeof(DDGIDiagnosticRayGpuData) == 32u);
static_assert(sizeof(DDGIDiagnosticEventGpuData) == 32u);

[[nodiscard]] NURI_API uint32_t
ddgiProbeCount(const glm::uvec3 &counts) noexcept;
[[nodiscard]] NURI_API uint32_t
ddgiProbeIndex(const glm::uvec3 &coordinate,
               const glm::uvec3 &counts) noexcept;
[[nodiscard]] NURI_API glm::uvec3
ddgiProbeCoordinate(uint32_t probeIndex,
                    const glm::uvec3 &counts) noexcept;
[[nodiscard]] NURI_API glm::uvec3
ddgiPhysicalProbeCoordinate(const glm::uvec3 &logicalCoordinate,
                            const glm::uvec3 &ringOrigin,
                            const glm::uvec3 &counts) noexcept;
[[nodiscard]] NURI_API glm::vec3
ddgiLocalProbePosition(const DDGIVolumeLayout &layout,
                       const glm::uvec3 &coordinate) noexcept;
[[nodiscard]] NURI_API glm::ivec3
ddgiCameraCell(const glm::vec3 &cameraLocal,
               const glm::vec3 &spacing) noexcept;
[[nodiscard]] NURI_API DDGIScrollPlan
makeDDGIScrollPlan(const glm::ivec3 &cameraCell,
                   const glm::uvec3 &ringOrigin,
                   const glm::ivec3 &targetCameraCell,
                   const glm::uvec3 &probeCounts) noexcept;
[[nodiscard]] NURI_API bool
isDDGINewlyExposedCoordinate(const glm::uvec3 &logicalCoordinate,
                             const DDGIScrollPlan &plan,
                             const glm::uvec3 &probeCounts) noexcept;
[[nodiscard]] NURI_API bool
isRigidDDGITransform(const glm::mat4 &worldFromLocal,
                     float epsilon = 1.0e-4f) noexcept;
[[nodiscard]] NURI_API Result<DDGIVolumeLayout, DDGIVolumeValidationError>
makeDDGIVolumeLayout(DDGIVolumeId id, const DDGIVolumeDesc &desc,
                     const glm::mat4 &worldFromLocal,
                     DDGIAtlasLayout irradianceAtlas,
                     DDGIAtlasLayout distanceAtlas, uint64_t generation,
                     glm::ivec3 cameraCell = glm::ivec3(0),
                     glm::uvec3 ringOrigin = glm::uvec3(0u));
[[nodiscard]] NURI_API DDGIVolumeBlendWeights
ddgiPriorityBlendWeights(float firstCoverage, float secondCoverage) noexcept;

} // namespace nuri
