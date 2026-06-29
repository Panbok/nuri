#pragma once

#include "nuri/gfx/frame/render_frame_context.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri::tools::snapshot {

enum class SnapshotExitCode : int {
  Success = 0,
  VisualMismatch = 1,
  InvalidInput = 2,
  EnvironmentUnavailable = 3,
  RuntimeError = 4,
  MissingBaseline = 5,
};

struct SnapshotRenderGraphConfig {
  uint32_t workerCount = 1u;
  bool parallelCompile = false;
  bool parallelRecording = false;
};

struct SnapshotCameraConfig {
  glm::vec3 position{0.0f, 1.0f, 4.0f};
  glm::vec3 direction{0.0f, 0.0f, -1.0f};
  glm::vec3 positionDeltaPerFrame{0.0f};
  float verticalFovDegrees = 60.0f;
  float nearPlane = 0.05f;
  float farPlane = 500.0f;
};

struct SnapshotSceneConfig {
  std::string kind = "procedural";
  std::string pathBase{};
  std::filesystem::path path{};
  bool flipUVs = false;
  bool generateMeshlets = false;
  uint32_t meshletMaxVertices = 64u;
  uint32_t meshletMaxPrimitives = 124u;
  float meshletConeWeight = 0.0f;
  std::string generator = "nuri.procedural.v1";
  uint32_t seed = 1u;
  std::string contentHash = "procedural-empty-v1";
};

struct SnapshotRequirements {
  std::vector<std::string> assets{};
  std::vector<std::string> backends{};
  bool allowVisibleWindow = true;
};

struct SnapshotCaptureTarget {
  std::string name{};
  std::string profile{};
  bool required = true;
};

struct SnapshotCase {
  uint32_t schemaVersion = 1u;
  std::string id{};
  std::string suite{};
  std::string description{};
  SnapshotSceneConfig scene{};
  std::string backend = "default";
  std::array<uint32_t, 2> resolution{1280u, 720u};
  double fixedDeltaSeconds = 1.0 / 60.0;
  uint32_t warmupFrames = 2u;
  uint32_t captureFrame = 4u;
  bool authoritative = false;
  std::string presentMode = "immediate";
  std::string windowMode = "visible";
  SnapshotRenderGraphConfig renderGraph{};
  SnapshotCameraConfig camera{};
  RenderSettings settings{};
  SnapshotRequirements requirements{};
  std::vector<SnapshotCaptureTarget> captures{};
  std::filesystem::path manifestPath{};
};

[[nodiscard]] std::string snapshotExitCodeName(SnapshotExitCode code);
void sanitizeSnapshotRenderSettings(RenderSettings &settings);

} // namespace nuri::tools::snapshot
