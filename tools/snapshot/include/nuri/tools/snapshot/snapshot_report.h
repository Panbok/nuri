#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/tools/snapshot/snapshot_case.h"
#include "nuri/tools/snapshot/snapshot_compare.h"
#include "nuri/tools/snapshot/snapshot_environment.h"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace nuri::tools::snapshot {

struct SnapshotCaptureReport {
  std::string target{};
  std::string artifactStem{};
  std::string profile{};
  bool required = true;
  bool available = false;
  uint32_t capturePointVersion = 0u;
  uint64_t captureFrameIndex = 0u;
  std::string kind{};
  std::string lifetime{};
  std::string format{};
  std::string colorSpace{};
  std::string origin = "top_left";
  uint32_t width = 0u;
  uint32_t height = 0u;
  uint32_t mip = 0u;
  uint32_t layer = 0u;
  std::string actualHash{};
  std::string expectedHash{};
  std::filesystem::path actual{};
  std::filesystem::path actualMetadata{};
  std::filesystem::path preview{};
  std::filesystem::path expected{};
  std::filesystem::path diff{};
  SnapshotCompareMetrics metrics{};
  SnapshotSemanticMetrics semanticMetrics{};
  std::vector<std::string> failedThresholds{};
  std::string producerPassLabel{};
  std::string readbackError{};
  std::string status = "captured";
  std::string statusReason = "capture_written";
};

struct SnapshotReportArtifacts {
  std::filesystem::path artifactDir{};
  std::filesystem::path rootHtml{};
  std::filesystem::path caseDir{};
  std::filesystem::path caseHtml{};
};

struct SnapshotReport {
  uint32_t schemaVersion = 1u;
  std::string kind = "nuri.snapshot.report";
  std::string generatedAtUtc{};
  std::string command{};
  std::string baselineProfile = "local-nvrhi-visible";
  bool baselineProfileCompatible = false;
  std::vector<std::string> baselineProfileIncompatibilityReasons{};
  SnapshotEnvironment environment{};
  SnapshotCase snapshotCase{};
  SnapshotReportArtifacts artifacts{};
  std::vector<SnapshotCaptureReport> captures{};
  std::vector<std::string> availableCapturePoints{};
  RenderFrameMetrics rendererMetrics{};
  std::map<std::string, double> rendererMetricValues{};
  std::string captureSynchronization = "wait_idle";
  std::string reproduceCommand{};
  std::vector<std::string> warnings{};
  std::vector<std::string> errors{};
};

[[nodiscard]] std::string snapshotFormatName(Format format);
[[nodiscard]] Result<std::string, std::string>
writeSnapshotReportJson(const SnapshotReport &report);
[[nodiscard]] Result<bool, std::string>
writeSnapshotReportFile(const SnapshotReport &report,
                        const std::filesystem::path &path);
[[nodiscard]] Result<SnapshotReport, std::string>
readSnapshotReportFile(const std::filesystem::path &path);

} // namespace nuri::tools::snapshot
