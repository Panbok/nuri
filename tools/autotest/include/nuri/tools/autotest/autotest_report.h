#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/autotest/autotest_assertion.h"
#include "nuri/tools/autotest/autotest_case.h"
#include "nuri/tools/autotest/autotest_environment.h"
#include "nuri/tools/snapshot/snapshot_report.h"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace nuri::tools::autotest {

struct AutotestCaptureReport {
  std::string checkpointId{};
  uint32_t checkpointFrame = 0u;
  std::string target{};
  std::string profile{};
  bool required = true;
  bool compare = true;
  nuri::tools::snapshot::SnapshotCaptureReport snapshot{};
};

struct AutotestReadoutReport {
  std::string checkpointId{};
  std::string id{};
  std::string type{};
  uint64_t requestId = 0u;
  uint32_t requestFrame = 0u;
  uint32_t resultFrame = 0u;
  bool required = true;
  std::string status = "pending";
  std::string statusReason = "pending";
  std::map<std::string, double> values{};
  std::vector<AutotestAssertionResult> assertions{};
};

struct AutotestCheckpointReport {
  std::string id{};
  uint32_t frame = 0u;
  std::vector<AutotestCaptureReport> captures{};
  std::vector<AutotestReadoutReport> readouts{};
  std::map<std::string, double> measurements{};
  std::vector<AutotestAssertionResult> assertions{};
  std::vector<std::string> warnings{};
  std::vector<std::string> errors{};
};

struct AutotestMetricWindowReport {
  std::string id{};
  uint32_t startFrame = 0u;
  uint32_t endFrame = 0u;
  std::vector<AutotestAssertionResult> assertions{};
  std::vector<std::string> warnings{};
  std::vector<std::string> errors{};
};

struct AutotestRunMetadata {
  double fixedDeltaSeconds = 1.0 / 60.0;
  uint32_t warmupFrames = 0u;
  uint32_t endFrame = 0u;
  uint32_t renderedFrames = 0u;
  uint32_t readoutDrainFrames = 0u;
  uint32_t readoutDrainTimeoutMs = 1000u;
  std::string captureSynchronization = "wait_idle";
  bool validForComparison = true;
};

struct AutotestFrameReport {
  uint64_t frameIndex = 0u;
  std::map<std::string, double> measurements{};
  std::vector<std::string> unavailableMetrics{};
};

struct AutotestReportArtifacts {
  std::filesystem::path artifactDir{};
  std::filesystem::path rootHtml{};
  std::filesystem::path caseDir{};
  std::filesystem::path caseHtml{};
};

struct AutotestReport {
  uint32_t schemaVersion = 1u;
  std::string kind = "nuri.autotest.report";
  std::string generatedAtUtc{};
  std::string command{};
  AutotestEnvironment environment{};
  AutotestCase testCase{};
  AutotestRunMetadata run{};
  AutotestReportArtifacts artifacts{};
  std::vector<AutotestCheckpointReport> checkpoints{};
  std::vector<AutotestMetricWindowReport> metricWindows{};
  std::vector<AutotestFrameReport> frames{};
  std::vector<std::string> unavailableMetrics{};
  std::vector<std::string> warnings{};
  std::vector<std::string> errors{};
  std::string reproduceCommand{};
};

[[nodiscard]] Result<std::string, std::string>
writeAutotestReportJson(const AutotestReport &report);
[[nodiscard]] Result<bool, std::string>
writeAutotestReportFile(const AutotestReport &report,
                        const std::filesystem::path &path);
[[nodiscard]] Result<AutotestReport, std::string>
readAutotestReportFile(const std::filesystem::path &path);
[[nodiscard]] Result<std::string, std::string>
writeAutotestHtmlReport(const AutotestReport &report);
[[nodiscard]] Result<bool, std::string>
writeAutotestHtmlReportFile(const AutotestReport &report,
                            const std::filesystem::path &path);
[[nodiscard]] Result<std::string, std::string>
writeAutotestSuiteHtml(const std::vector<AutotestReport> &reports,
                       std::string_view suite);
[[nodiscard]] Result<bool, std::string>
writeAutotestSuiteHtmlFile(const std::vector<AutotestReport> &reports,
                           std::string_view suite,
                           const std::filesystem::path &path);

} // namespace nuri::tools::autotest
