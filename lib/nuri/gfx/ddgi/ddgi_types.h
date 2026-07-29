#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/scene/ddgi_volume.h"
#include "nuri/scene/scene_handles.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <string_view>

namespace nuri {

inline constexpr uint32_t kDDGILayoutVersion = 1u;
inline constexpr uint32_t kDDGIFrameGpuDataVersion = 5u;
inline constexpr uint32_t kDDGICaptureSemanticsVersion = 4u;
inline constexpr uint32_t kDDGIFrameMetricsVersion = 9u;
inline constexpr uint32_t kDDGIGatherIdentitySchemaVersion = 1u;
inline constexpr uint32_t kMaxDDGIEffectiveVolumes = 8u;
inline constexpr uint32_t kMaxDDGIClipmapCascades = 4u;
inline constexpr uint32_t kMaxDDGIVolumesSampledPerSurface = 2u;
inline constexpr uint32_t kMaxDDGIVolumes = kMaxDDGIEffectiveVolumes;
inline constexpr float kDDGIIrradianceGamma = 5.0f;
inline constexpr float kDDGIVisibilityFloor = 0.05f;
inline constexpr float kDDGIChebyshevExponent = 3.0f;
inline constexpr float kDDGIProbeWeightCrushThreshold = 0.2f;
inline constexpr uint32_t kDDGIMaxDiagnosticRays = 1024u;
inline constexpr uint32_t kDDGIMaxDiagnosticCandidateEventsPerRay = 8u;

[[nodiscard]] constexpr uint32_t
ddgiUniformSubsetIndex(uint32_t totalCount, uint32_t sampleCount,
                       uint64_t submittedSequence,
                       uint32_t sampleIndex) noexcept {
  if (totalCount == 0u || sampleCount == 0u) {
    return 0u;
  }
  const uint32_t boundedSampleCount =
      sampleCount < totalCount ? sampleCount : totalCount;
  const uint32_t stride = totalCount / boundedSampleCount;
  return static_cast<uint32_t>(
      (submittedSequence + static_cast<uint64_t>(sampleIndex) * stride) %
      totalCount);
}

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

enum class DDGISurfaceGatherArchitecture : uint8_t {
  ForwardFragment = 0,
  ComputeSurfaceCache,
};

enum class DDGISurfaceGatherVariant : uint8_t {
  Product = 0,
  Bypass,
  Candidates,
  ProbeVisibility,
  Atlas,
};

enum class DDGIUpdateReason : uint32_t {
  Bootstrap = 1u << 0u,
  Scroll = 1u << 1u,
  DirtyGeometry = 1u << 2u,
  RadiometricResponse = 1u << 3u,
  Maintenance = 1u << 4u,
  Force = 1u << 5u,
  Wake = 1u << 6u,
  Reclassification = 1u << 7u,
};

enum class DDGIFallbackReason : uint8_t {
  None = 0,
  Disabled,
  Unsupported,
  NoVolumes,
  CoverageBoundsUnavailable,
  CoverageUnsatisfied,
  RayTracingSceneWarming,
  VolumeResourcesWarming,
  ShaderUnavailable,
  AllocationFailed,
  ReadbackRingSaturated,
  OptionalFeatureFailure,
};

enum class DDGIStartupPhase : uint8_t {
  ResourcesPending = 0,
  ResourcesReadyNoHistory,
  FirstIrradianceReady,
  CoarseCoverageReady,
  FullCoverageWarmingDetail,
  FullCoverageReady,
  Degraded,
  Failed,
};

enum class DDGIReadbackSlotState : uint8_t {
  Free = 0,
  Recording,
  Pending,
  Completed,
  Consumed,
  Dropped,
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

enum class DDGICoverageLimit : uint8_t {
  None = 0,
  InvalidSettings,
  SceneBoundsUnavailable,
  SceneBoundsIncomplete,
  ProbeCountAxis,
  TotalProbeCount,
  AtlasDimensions,
  PersistentMemory,
  ReplacementPeakMemory,
  ProbeSpacing,
  CoverageExtents,
  EffectiveVolumeCapacity,
  ArithmeticOverflow,
};

enum class DDGICoverageStatus : uint8_t {
  SkyFallbackOnly = 0,
  PartialCoverage,
  FullCoverageWarmingDetail,
  FullCoverageReady,
};

struct DDGIVolumeFrameMetrics {
  uint64_t effectiveKeyHash = 0u;
  uint64_t layoutGeneration = 0u;
  uint64_t resourceGeneration = 0u;
  uint32_t effectiveKind = 0u;
  uint32_t tier = 0u;
  uint32_t cascadeIndex = UINT32_MAX;
  uint32_t totalProbes = 0u;
  uint32_t initializedProbes = 0u;
  uint32_t shadingEnabledProbes = 0u;
  uint32_t invalidProbes = 0u;
  uint32_t newlyExposedProbes = 0u;
  uint32_t updates = 0u;
  uint32_t primaryQueries = 0u;
  uint32_t primaryQueriesIssued = 0u;
  uint32_t secondaryQueries = 0u;
  uint32_t updateAgeMedian = 0u;
  uint32_t updateAgeP95 = 0u;
  uint32_t updateAgeMaximum = 0u;
  uint32_t scheduledQuota = 0u;
  uint32_t usedQuota = 0u;
  int64_t deficit = 0;
  uint32_t starvationFrames = 0u;
  uint32_t estimatedFullRefreshFrames = 0u;
  uint64_t persistentBytes = 0u;
  glm::vec3 interiorHalfExtents{0.0f};
  glm::vec3 fadeStartHalfExtents{0.0f};
  glm::vec3 fadeEndHalfExtents{0.0f};
  glm::ivec3 cameraCell{0};
  float historyReadyPercentage = 0.0f;
  float coverageReadyPercentage = 0.0f;
  float uniqueCoveragePercentage = 100.0f;
  float confidence = 0.0f;
  uint32_t active = 0u;
  uint32_t redundantCoverage = 0u;
};

inline constexpr uint32_t kDDGIVolumeMetricSchemaVersion = 1u;
struct DDGIVolumeMetricValue {
  template <typename T>
  constexpr DDGIVolumeMetricValue(std::string_view metricSuffix,
                                  T metricValue) noexcept
      : suffix(metricSuffix), value(static_cast<double>(metricValue)) {}
  std::string_view suffix;
  double value;
};
[[nodiscard]] NURI_API std::array<DDGIVolumeMetricValue, 26>
ddgiVolumeMetricValues(const DDGIVolumeFrameMetrics &volume) noexcept;

struct DDGICaptureMetadata {
  uint64_t effectiveKeyHash = 0u;
  uint64_t coverageGeneration = 0u;
  uint64_t layoutGeneration = 0u;
  uint64_t resourceGeneration = 0u;
  uint64_t sceneBoundsGeneration = 0u;
  uint32_t effectiveKind = 0u;
  uint32_t cascadeIndex = UINT32_MAX;
  glm::uvec3 ringOrigin{0u};
  glm::ivec3 cameraCell{0};
  glm::vec3 requestedHalfExtents{0.0f};
  glm::vec3 achievedHalfExtents{0.0f};
  glm::vec3 fadeStartHalfExtents{0.0f};
  glm::vec3 fadeEndHalfExtents{0.0f};
  uint32_t transitionCells = 0u;
  uint32_t valid = 0u;
};

struct DDGIFrameMetrics {
  uint32_t version = kDDGIFrameMetricsVersion;
  uint32_t requested = 0u;
  uint32_t active = 0u;
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
  uint32_t probeStateReadbackSourceFrame = 0u;
  uint32_t probeStateReadbackStaleFrames = 0u;
  uint32_t updatedProbes = 0u;
  uint32_t primaryQueries = 0u;
  uint32_t classificationProbeUpdates = 0u;
  uint32_t classificationPrimaryQueries = 0u;
  uint32_t irradiancePrimaryQueries = 0u;
  uint32_t primaryQueriesIssued = 0u;
  uint32_t traceCountersAvailable = 0u;
  uint32_t traceCounterSourceFrame = 0u;
  uint32_t traceCounterStaleFrames = 0u;
  uint32_t secondaryQueriesReserved = 0u;
  uint32_t secondaryQueriesUnused = 0u;
  uint32_t secondaryQueries = 0u;
  uint32_t secondaryQueryBudgetOverflows = 0u;
  uint32_t directionalSecondaryQueries = 0u;
  uint32_t localSecondaryQueries = 0u;
  uint32_t primaryCandidateIntersections = 0u;
  uint32_t secondaryCandidateIntersections = 0u;
  uint32_t alphaCandidateRejections = 0u;
  uint32_t backfaceCandidateRejections = 0u;
  uint32_t candidateOverflows = 0u;
  uint32_t localLightTruncations = 0u;
  uint32_t nonFiniteRadianceRejects = 0u;
  uint32_t emissiveRadianceClamps = 0u;
  uint32_t directRadianceClamps = 0u;
  uint32_t skyRadianceClamps = 0u;
  uint32_t multiBounceRadianceClamps = 0u;
  uint32_t finalRadianceClamps = 0u;
  uint32_t diagnosticCountersEnabled = 0u;
  uint32_t qualitySchema = 0u;
  uint32_t requestedQualityPreset = 0u;
  uint32_t qualityPreset = 0u;
  uint32_t coveragePresetSchema = 0u;
  uint32_t requestedCoveragePreset = 0u;
  uint32_t coveragePreset = 0u;
  uint32_t productProfileSchema = 0u;
  uint64_t productProfileFingerprint = 0u;
  uint32_t opaqueGatherArchitecture = 0u;
  uint32_t opaqueGatherVariant = 0u;
  uint32_t transmissionGatherArchitecture = 0u;
  uint32_t transmissionGatherVariant = 0u;
  uint32_t traceMultiBounceGatherArchitecture = 0u;
  uint32_t traceMultiBounceGatherVariant = 0u;
  uint32_t gatherIdentitySchema = kDDGIGatherIdentitySchemaVersion;
  uint32_t surfaceGatherWidth = 0u;
  uint32_t surfaceGatherHeight = 0u;
  uint32_t surfaceGatherMaxCandidateVolumes = 0u;
  uint32_t surfaceGatherMaxSampledVolumes = 0u;
  uint32_t surfaceGatherMaxStateLoadsPerPixel = 0u;
  uint32_t surfaceGatherMaxAtlasSamplesPerPixel = 0u;
  uint32_t surfaceCacheFormat = 0u;
  uint64_t surfaceCacheBytes = 0u;
  uint32_t rayQueryCapacity = 0u;
  uint32_t probeUpdateCapacity = 0u;
  uint32_t requestedProbeUpdateCapacity = 0u;
  uint32_t effectiveProbeUpdateCapacity = 0u;
  uint32_t requestedMaintenanceProbeUpdateCapacity = 0u;
  uint32_t effectiveMaintenanceProbeUpdateCapacity = 0u;
  uint32_t maintenanceProbeUpdates = 0u;
  uint32_t primaryResultCapacity = 0u;
  uint32_t traceDispatches = 0u;
  uint32_t traceLaunchedLanes = 0u;
  uint32_t traceUsefulLanes = 0u;
  uint32_t classificationLaunchedLanes = 0u;
  uint32_t classificationUsefulLanes = 0u;
  uint32_t irradianceAtlasDispatches = 0u;
  uint32_t distanceAtlasDispatches = 0u;
  uint64_t irradianceResultVisits = 0u;
  uint64_t distanceResultVisits = 0u;
  uint64_t irradianceTexelWrites = 0u;
  uint64_t distanceTexelWrites = 0u;
  uint32_t updateReasonBits = 0u;
  uint32_t bootstrapUpdates = 0u;
  uint32_t scrollUpdates = 0u;
  uint32_t dirtyGeometryUpdates = 0u;
  uint32_t radiometricResponseUpdates = 0u;
  uint32_t maintenanceUpdates = 0u;
  uint32_t forceUpdates = 0u;
  uint32_t wakeUpdates = 0u;
  uint32_t reclassificationUpdates = 0u;
  uint32_t readbackWaits = 0u;
  uint32_t readbackPendingSlots = 0u;
  uint32_t readbackDroppedSamples = 0u;
  uint32_t readbackOldestPendingAge = 0u;
  uint32_t readbackBlockingFallbacks = 0u;
  uint32_t readbackGenerationMismatches = 0u;
  uint32_t readbackEarlyReuseAttempts = 0u;
  uint32_t resetCount = 0u;
  uint32_t scrollCount = 0u;
  uint32_t invalidatedProbes = 0u;
  uint32_t failedVolumes = 0u;
  uint32_t effectiveVolumes = 0u;
  uint32_t authoredVolumes = 0u;
  uint32_t generatedVolumes = 0u;
  uint32_t redundantAuthoredVolumes = 0u;
  uint32_t redundantAuthoredProbes = 0u;
  uint32_t coverageMode = 0u;
  uint32_t coverageSolveExecutions = 0u;
  uint32_t coveragePlanCacheHits = 0u;
  uint64_t stateHistoryScanCount = 0u;
  uint64_t ageSampleCount = 0u;
  uint64_t ageSelectionCount = 0u;
  uint64_t coverageLatticeEvaluations = 0u;
  uint32_t uploadSubmissionCount = 0u;
  uint64_t lightDifferenceComparisons = 0u;
  DDGICoverageStatus coverageStatus = DDGICoverageStatus::SkyFallbackOnly;
  DDGICoverageLimit coverageError = DDGICoverageLimit::None;
  DDGICoverageLimit limitingConstraint = DDGICoverageLimit::None;
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
  uint32_t diagnosticSampleCount = 0u;
  uint32_t uncoveredDiagnosticSamples = 0u;
  uint32_t skyRemainderSamples = 0u;
  uint32_t diagnosticSamplesAvailable = 0u;
  uint32_t dirtyRegionsPending = 0u;
  uint32_t dirtyProbesAffected = 0u;
  uint32_t classificationFallbacks = 0u;
  uint32_t classificationOverflows = 0u;
  uint64_t dirtyRegionsProduced = 0u;
  uint64_t dirtyRegionsMerged = 0u;
  uint64_t dirtyRegionsOverflowed = 0u;
  uint64_t persistentBytes = 0u;
  uint64_t redundantAuthoredBytes = 0u;
  uint64_t frameBatchBytes = 0u;
  uint64_t readbackCopyBytes = 0u;
  uint64_t readbackPerSlotBytes = 0u;
  uint64_t readbackRingBytes = 0u;
  uint32_t frameSlotCount = 0u;
  uint64_t frameRingDeviceBytes = 0u;
  uint64_t frameRingReadbackBytes = 0u;
  uint64_t frameRingBytes = 0u;
  uint64_t committedAtlasBytes = 0u;
  uint64_t pendingAtlasBytes = 0u;
  uint64_t peakAtlasBytes = 0u;
  uint64_t submittedSequence = 0u;
  uint64_t layoutGeneration = 0u;
  uint64_t resourceGeneration = 0u;
  uint64_t deviceEpoch = 0u;
  uint64_t consumedResetEpoch = 0u;
  uint64_t consumedForceUpdateEpoch = 0u;
  float gpuTimeMs = 0.0f;
  float traceGpuTimeMs = 0.0f;
  float updateGpuTimeMs = 0.0f;
  float irradianceUpdateGpuTimeMs = 0.0f;
  float distanceUpdateGpuTimeMs = 0.0f;
  float relocateClassifyGpuTimeMs = 0.0f;
  float readbackGpuTimeMs = 0.0f;
  uint64_t gpuTimingSourceFrameIndex = UINT64_MAX;
  uint32_t gpuTimingAvailable = 0u;
  uint32_t traceGpuTimingAvailable = 0u;
  uint32_t updateGpuTimingAvailable = 0u;
  uint32_t irradianceUpdateGpuTimingAvailable = 0u;
  uint32_t distanceUpdateGpuTimingAvailable = 0u;
  uint32_t relocateClassifyGpuTimingAvailable = 0u;
  uint32_t readbackGpuTimingAvailable = 0u;
  float maxRelocation = 0.0f;
  glm::vec3 requestedCoverageHalfExtents{0.0f};
  glm::vec3 achievedCoverageHalfExtents{0.0f};
  float sceneCoverageRatio = 0.0f;
  float coverageResolveCpuTimeMs = 0.0f;
  float prepareCpuTimeMs = 0.0f;
  float scheduleCpuTimeMs = 0.0f;
  float graphBuildCpuTimeMs = 0.0f;
  float readbackPollCpuTimeMs = 0.0f;
  float skyRemainderOverThresholdPercentage = 1.0f;
  DDGIStartupPhase startupPhase = DDGIStartupPhase::ResourcesPending;
  DDGIFallbackReason fallbackReason = DDGIFallbackReason::Disabled;
  DDGIVolumeFailureReason volumeFailureReason = DDGIVolumeFailureReason::None;
  DDGIDebugView debugView = DDGIDebugView::None;
  std::array<DDGIVolumeFrameMetrics, kMaxDDGIEffectiveVolumes> volumes{};
};

struct DDGIVolumeBlendWeights {
  float first = 0.0f;
  float second = 0.0f;
  float sky = 1.0f;
};

struct alignas(16) DDGIProbeStateGpuData {
  glm::vec4 relocation{0.0f};
  // x: state, y: last submitted sequence, z: classification iteration,
  // w: reserved.
  glm::uvec4 stateAgeFlags{static_cast<uint32_t>(DDGIProbeState::Vigilant), 0u,
                           0u, 0u};
};

struct alignas(16) DDGIVolumeGpuData {
  glm::mat4 worldFromLocal{1.0f};
  glm::mat4 localFromWorld{1.0f};
  uint64_t probeStateBufferAddress = 0u;
  uint32_t resourceFlags = 0u;
  // High 16 bits: trace-light subset offset; low 16 bits: subset count.
  uint32_t localLightSubsetOffsetCount = 0u;
  // xyz: probe spacing; w: absolute world-space receiver bias.
  glm::vec4 probeSpacingAndBias{1.0f, 1.0f, 1.0f, 0.1f};
  // x: primary probe ray, y: local shadow, z: directional shadow,
  // w: classification/relocation. Shadow and classification values are
  // spacing-scaled; primary probe ray bias is an absolute world-space distance.
  glm::vec4 rayBiases{0.0001f, 0.30f, 0.30f, 0.30f};
  glm::vec4 centerHalfExtentsAndMaxDistance{0.5f, 0.5f, 0.5f, 20.0f};
  glm::uvec4 probeCountsAndCount{2u, 2u, 2u, 8u};
  glm::uvec4 irradianceAtlas{kInvalidTextureBindlessIndex, 0u, 0u, 0u};
  glm::uvec4 distanceAtlas{kInvalidTextureBindlessIndex, 0u, 0u, 0u};
  glm::uvec4 ringOriginAndFlags{0u};
  glm::uvec4 generations{0u};
  glm::uvec4 effectiveIdentity{0u};
  glm::uvec4 tierTransitionCoverageFlags{0u};
  glm::vec4 continuousCameraLocal{0.0f};
  glm::vec4 fadeStartHalfExtents{0.0f};
  glm::vec4 fadeEndHalfExtents{0.0f};
};

struct alignas(16) DDGIFrameGpuData {
  // x: active count; y: opaque/transmission/trace gather variants packed in
  // successive bytes; z: record version; w: sampler index.
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
  uint32_t primaryQueriesIssued = 0u;
  uint32_t secondaryQueriesReserved = 0u;
  uint32_t sourceFrame = 0u;
  uint32_t localSecondaryQueries = 0u;
  uint32_t secondaryQueryBudgetOverflows = 0u;
  uint32_t nonFiniteRadianceRejects = 0u;
  uint32_t emissiveRadianceClamps = 0u;
  uint32_t directRadianceClamps = 0u;
  uint32_t skyRadianceClamps = 0u;
  uint32_t multiBounceRadianceClamps = 0u;
  uint32_t finalRadianceClamps = 0u;
  uint32_t reserved1 = 0u;
  uint32_t reserved2 = 0u;
  std::array<uint32_t, kMaxDDGIVolumes> primaryQueriesIssuedByVolume{};
  std::array<uint32_t, kMaxDDGIVolumes> secondaryQueriesByVolume{};
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
    kDDGIMaxDiagnosticRays * (kDDGIMaxDiagnosticCandidateEventsPerRay + 1u) *
        sizeof(DDGIDiagnosticEventGpuData);

static_assert(sizeof(DDGIProbeStateGpuData) == 32u);
static_assert(sizeof(DDGIVolumeGpuData) == 352u);
static_assert(sizeof(DDGIFrameGpuData) == 2832u);
static_assert(sizeof(DDGIRayResultGpuData) == 32u);
static_assert(sizeof(DDGITraceCountersGpuData) == 144u);
static_assert(sizeof(DDGIDiagnosticHeaderGpuData) == 160u);
static_assert(sizeof(DDGIDiagnosticRayGpuData) == 32u);
static_assert(sizeof(DDGIDiagnosticEventGpuData) == 32u);

[[nodiscard]] NURI_API uint32_t
ddgiProbeCount(const glm::uvec3 &counts) noexcept;
[[nodiscard]] NURI_API uint32_t
ddgiProbeIndex(const glm::uvec3 &coordinate, const glm::uvec3 &counts) noexcept;
[[nodiscard]] NURI_API glm::uvec3
ddgiProbeCoordinate(uint32_t probeIndex, const glm::uvec3 &counts) noexcept;
[[nodiscard]] NURI_API glm::uvec3
ddgiPhysicalProbeCoordinate(const glm::uvec3 &logicalCoordinate,
                            const glm::uvec3 &ringOrigin,
                            const glm::uvec3 &counts) noexcept;
[[nodiscard]] NURI_API glm::vec3
ddgiLocalProbePosition(const DDGIVolumeLayout &layout,
                       const glm::uvec3 &coordinate) noexcept;
[[nodiscard]] NURI_API glm::ivec3
ddgiCameraCell(const glm::vec3 &cameraLocal, const glm::vec3 &spacing) noexcept;
[[nodiscard]] NURI_API DDGIScrollPlan makeDDGIScrollPlan(
    const glm::ivec3 &cameraCell, const glm::uvec3 &ringOrigin,
    const glm::ivec3 &targetCameraCell, const glm::uvec3 &probeCounts) noexcept;
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
