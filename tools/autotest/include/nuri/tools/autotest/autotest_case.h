#pragma once

#include "nuri/gfx/frame/render_frame_context.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri::tools::autotest {

enum class AutotestExitCode : int {
  Success = 0,
  ScenarioFailure = 1,
  InvalidInput = 2,
  EnvironmentUnavailable = 3,
  RuntimeError = 4,
  MissingBaseline = 5,
};

struct AutotestRenderGraphConfig {
  uint32_t workerCount = 1u;
  bool parallelCompile = false;
  bool parallelRecording = false;
};

struct AutotestCameraConfig {
  glm::vec3 position{0.0f, 1.0f, 4.0f};
  glm::vec3 direction{0.0f, 0.0f, -1.0f};
  glm::vec3 target{0.0f, 1.0f, 3.0f};
  bool hasTarget = false;
  float verticalFovDegrees = 60.0f;
  float nearPlane = 0.05f;
  float farPlane = 500.0f;
};

struct AutotestSceneConfig {
  std::string kind = "procedural";
  std::string pathBase{};
  std::filesystem::path path{};
  bool flipUVs = false;
  bool generateMeshlets = false;
  uint32_t meshletMaxVertices = 64u;
  uint32_t meshletMaxPrimitives = 124u;
  float meshletConeWeight = 0.0f;
  std::string baseModelKind{};
  double baseModelTargetRadius = 1.0;
  double baseModelMinScale = 0.0001;
  double baseModelMaxScale = 1000.0;
  std::string generator = "nuri.procedural.v1";
  uint32_t seed = 1u;
  std::string contentHash = "procedural-empty-v1";
};

struct AutotestEnvironmentTextureConfig {
  bool enabled = false;
  bool required = false;
  std::string pathBase{};
  std::filesystem::path path{};
  std::string kind = "Texture2D";
  std::string debugName{};
};

struct AutotestEnvironmentConfig {
  AutotestEnvironmentTextureConfig cubemap{};
  AutotestEnvironmentTextureConfig irradiance{};
  AutotestEnvironmentTextureConfig prefilteredGgx{};
  AutotestEnvironmentTextureConfig prefilteredCharlie{};
  AutotestEnvironmentTextureConfig brdfLut{};
};

struct AutotestRequirements {
  std::vector<std::string> assets{};
  std::vector<std::string> backends{};
  bool allowVisibleWindow = true;
  std::optional<uint32_t> msaaSamples{};
  bool accelerationStructure = false;
  bool rayQuery = false;
};

struct AutotestCaptureTarget {
  std::string target{};
  std::string profile{};
  bool required = true;
  bool compare = true;
};

struct AutotestMetricAssertion {
  std::string id{};
  std::string metric{};
  std::string severity = "fail";
  bool optional = false;
  bool hasEquals = false;
  double equals = 0.0;
  bool hasMin = false;
  double min = 0.0;
  bool hasMax = false;
  double max = 0.0;
  bool hasLessThan = false;
  double lessThan = 0.0;
  bool hasLessOrEqual = false;
  double lessOrEqual = 0.0;
  bool hasGreaterThan = false;
  double greaterThan = 0.0;
  bool hasGreaterOrEqual = false;
  double greaterOrEqual = 0.0;
};

struct AutotestReadoutRequest {
  std::string id{};
  std::string type{};
  uint32_t x = 0u;
  uint32_t y = 0u;
  bool required = true;
  std::vector<AutotestMetricAssertion> assertions{};
};

struct AutotestMetricWindowAssertion {
  std::string id{};
  std::string metric{};
  std::string severity = "fail";
  bool optional = false;
  bool hasEquals = false;
  double equals = 0.0;
  bool hasMin = false;
  double min = 0.0;
  bool hasMax = false;
  double max = 0.0;
  bool hasMedianMin = false;
  double medianMin = 0.0;
  bool hasMedianMax = false;
  double medianMax = 0.0;
  bool hasP95Min = false;
  double p95Min = 0.0;
  bool hasP95Max = false;
  double p95Max = 0.0;
  bool hasVarianceMax = false;
  double varianceMax = 0.0;
};

struct AutotestMetricWindow {
  std::string id{};
  uint32_t startFrame = 0u;
  uint32_t endFrame = 0u;
  std::vector<AutotestMetricWindowAssertion> assertions{};
};

struct AutotestPixelRoi {
  uint32_t x = 0u;
  uint32_t y = 0u;
  uint32_t width = 0u;
  uint32_t height = 0u;
};

struct AutotestCoverageRange {
  bool hasMin = false;
  double min = 0.0;
  bool hasMax = false;
  double max = 0.0;
};

struct AutotestMotionClassCoverage {
  bool configured = false;
  AutotestCoverageRange invalid{};
  AutotestCoverageRange staticCameraOnly{};
  AutotestCoverageRange full{};
};

struct AutotestMotionOracle {
  std::string motionTarget = "motion_vectors";
  std::string motionClassTarget{};
  AutotestPixelRoi roi{};
  bool hasMask = false;
  std::vector<std::array<uint32_t, 2>> mask{};
  glm::vec2 expectedVelocityPixels{0.0f};
  double p95ErrorMaxPixels = 0.0;
  double maxErrorMaxPixels = 0.0;
  AutotestMotionClassCoverage classCoverage{};
};

struct AutotestQualityOracleFile {
  uint32_t version = 0u;
  std::string pathBase = "repoRoot";
  std::filesystem::path path{};
  bool available = true;
  std::string unavailableReason{};
};

struct AutotestQualityOracleMask {
  uint32_t version = 0u;
  std::string pathBase = "repoRoot";
  std::filesystem::path path{};
};

struct AutotestQualityOracleBudgets {
  double normalizedMaeMax = 0.0;
  double normalizedRmseMax = 0.0;
  double lumaSsimMin = 0.0;
  double darkCollapsePercentMax = 0.0;
  uint32_t darkCollapseComponentMaxPixels = 0u;
  double relativeLumaEnergyDriftMax = 0.0;
  double edgeWidthRatioMin = 0.0;
  double edgeWidthRatioMax = 0.0;
  double edgeOvershootMax = 0.0;
  double edgeUndershootMax = 0.0;
  double temporalErrorMax = 0.0;
  double ghostEnergyMax = 0.0;
  double recoveryRmseMax = 0.0;
};

struct AutotestQualityOracleTemporal {
  std::string previousCheckpoint{};
  std::string previousOutputTarget{};
  AutotestQualityOracleFile previousReference{};
  std::string motionTarget = "motion_vectors";
  AutotestQualityOracleMask revealMask{};
};

struct AutotestQualityOracle {
  uint32_t schemaVersion = 1u;
  std::string outputTarget = "frame_color_hdr";
  AutotestQualityOracleFile reference{};
  std::optional<AutotestQualityOracleMask> mask{};
  double lscale = 0.0;
  AutotestQualityOracleBudgets budgets{};
  std::optional<AutotestQualityOracleTemporal> temporal{};
};

struct AutotestCheckpoint {
  std::string id{};
  uint32_t frame = 0u;
  std::vector<AutotestCaptureTarget> captures{};
  std::vector<AutotestReadoutRequest> readouts{};
  std::vector<AutotestMetricAssertion> assertions{};
  std::optional<AutotestMotionOracle> motionOracle{};
  std::optional<AutotestQualityOracle> qualityOracle{};
};

struct AutotestCameraKeyframe {
  uint32_t frame = 0u;
  glm::vec3 position{0.0f};
  glm::vec3 target{0.0f};
  bool hasTarget = false;
};

struct AutotestCameraPath {
  std::string id{};
  uint32_t startFrame = 0u;
  uint32_t endFrame = 0u;
  std::string interpolation = "linear";
  std::vector<AutotestCameraKeyframe> keyframes{};
};

struct AutotestTimelineEvent {
  uint32_t frame = 0u;
  std::string type{};
  std::string eventReason{};
  std::string target{};
  AutotestCameraConfig camera{};
  RenderSettings settings{};
  glm::vec3 translation{0.0f};
  glm::uvec3 probeCounts{0u};
  std::array<uint32_t, 2> resolution{0u, 0u};
  float intensity = 0.0f;
  uint32_t volumeIndex = 0u;
  bool hasSettings = false;
  bool hasResolution = false;
  bool preserveHistory = true;
};

struct AutotestTimeline {
  std::vector<AutotestCameraPath> cameraPaths{};
  std::vector<AutotestTimelineEvent> events{};
};

struct AutotestDDGIProbeInspection {
  uint32_t frame = 0u;
  uint32_t volumeIndex = 0u;
  uint32_t probeId = 0u;
  uint32_t rayCount = 128u;
};

struct AutotestCase {
  uint32_t schemaVersion = 1u;
  std::string id{};
  std::string suite{};
  std::string description{};
  AutotestSceneConfig scene{};
  std::string backend = "default";
  std::array<uint32_t, 2> resolution{1280u, 720u};
  double fixedDeltaSeconds = 1.0 / 60.0;
  uint32_t warmupFrames = 2u;
  uint32_t endFrame = 4u;
  bool authoritative = false;
  std::string presentMode = "immediate";
  std::string windowMode = "visible";
  AutotestRenderGraphConfig renderGraph{};
  AutotestCameraConfig camera{};
  AutotestEnvironmentConfig environment{};
  RenderSettings settings{};
  AutotestRequirements requirements{};
  AutotestTimeline timeline{};
  std::optional<AutotestDDGIProbeInspection> ddgiProbeInspection{};
  std::vector<AutotestCheckpoint> checkpoints{};
  std::vector<AutotestMetricWindow> metricWindows{};
  std::filesystem::path manifestPath{};
};

[[nodiscard]] std::string autotestExitCodeName(AutotestExitCode code);
void sanitizeAutotestRenderSettings(RenderSettings &settings);

} // namespace nuri::tools::autotest
