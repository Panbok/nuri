#pragma once

#include "nuri/gfx/frame/render_frame_context.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri::tools::benchmark {

enum class BenchmarkExitCode : int {
  Success = 0,
  Regression = 1,
  InvalidInput = 2,
  EnvironmentUnavailable = 3,
  RuntimeError = 4,
  MissingBaseline = 5,
};

struct BenchmarkThresholds {
  double failPercent = 10.0;
  double failAbsoluteMs = 0.2;
  double warnPercent = 5.0;
  double warnAbsoluteMs = 0.1;
};

struct BenchmarkRenderGraphConfig {
  uint32_t workerCount = 1u;
  bool parallelCompile = false;
  bool parallelRecording = false;
};

struct BenchmarkCameraConfig {
  glm::vec3 position{0.0f, 1.0f, 4.0f};
  glm::vec3 direction{0.0f, 0.0f, -1.0f};
  glm::vec3 target{0.0f, 1.0f, 3.0f};
  bool hasTarget = false;
  float verticalFovDegrees = 60.0f;
  float nearPlane = 0.05f;
  float farPlane = 500.0f;
};

struct BenchmarkSceneConfig {
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

struct BenchmarkCameraKeyframe {
  uint32_t frame = 0u;
  glm::vec3 position{0.0f};
  glm::vec3 target{0.0f};
  bool hasTarget = false;
};

struct BenchmarkCameraPath {
  std::string id{};
  uint32_t startFrame = 0u;
  uint32_t endFrame = 0u;
  std::string interpolation = "linear";
  std::vector<BenchmarkCameraKeyframe> keyframes{};
};

struct BenchmarkTimeline {
  std::vector<BenchmarkCameraPath> cameraPaths{};
};

struct BenchmarkRequirements {
  std::vector<std::string> assets{};
  std::vector<std::string> backends{};
  bool allowVisibleWindow = true;
  bool msaa4x = false;
};

struct BenchmarkCase {
  uint32_t schemaVersion = 1u;
  std::string id{};
  std::string suite{};
  std::string comparisonGroup{};
  std::string variant{};
  std::string description{};
  BenchmarkSceneConfig scene{};
  std::string backend = "default";
  std::array<uint32_t, 2> resolution{1280u, 720u};
  double fixedDeltaSeconds = 1.0 / 60.0;
  uint32_t warmupFrames = 5u;
  uint32_t measurementFrames = 10u;
  uint32_t cooldownFrames = 0u;
  uint32_t maxDrainFrames = 30u;
  uint32_t drainTimeoutMs = 5000u;
  uint32_t samples = 1u;
  bool authoritative = false;
  std::string presentMode = "immediate";
  BenchmarkRenderGraphConfig renderGraph{};
  BenchmarkCameraConfig camera{};
  BenchmarkTimeline timeline{};
  RenderSettings settings{};
  std::string settingsSignature{};
  std::string configSignature{};
  BenchmarkRequirements requirements{};
  BenchmarkThresholds thresholds{};
  std::vector<std::string> requiredMetrics{};
  std::filesystem::path manifestPath{};
};

[[nodiscard]] std::string benchmarkExitCodeName(BenchmarkExitCode code);
void sanitizeBenchmarkRenderSettings(RenderSettings &settings);

} // namespace nuri::tools::benchmark
