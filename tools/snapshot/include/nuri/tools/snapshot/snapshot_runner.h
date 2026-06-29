#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/snapshot/snapshot_report.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nuri::tools::snapshot {

struct SnapshotRunOptions {
  std::filesystem::path jsonOut{};
  std::filesystem::path htmlOut{};
  std::filesystem::path artifactDir{};
  std::string baselineProfile = "local-lvk-visible";
  std::string windowMode = "visible";
  bool dryRun = false;
  bool printEffectiveConfig = false;
  bool force = false;
  std::string command{};
};

struct SnapshotRunResult {
  SnapshotExitCode exitCode = SnapshotExitCode::Success;
  SnapshotReport report{};
  std::filesystem::path reportPath{};
  std::filesystem::path htmlPath{};
  std::string message{};
};

struct SnapshotSuiteRunResult {
  SnapshotExitCode exitCode = SnapshotExitCode::Success;
  std::vector<SnapshotRunResult> caseResults{};
  std::filesystem::path reportPath{};
  std::filesystem::path htmlPath{};
  std::string message{};
};

[[nodiscard]] Result<std::string, std::string>
formatSnapshotCaseListJson(const std::vector<SnapshotCase> &cases,
                           std::string_view suite = {});
[[nodiscard]] std::string
formatSnapshotCaseListText(const std::vector<SnapshotCase> &cases,
                           std::string_view suite = {});
[[nodiscard]] Result<std::string, std::string>
formatSnapshotCaseExplanationJson(const SnapshotCase &snapshotCase);
[[nodiscard]] std::string
formatSnapshotCaseExplanationText(const SnapshotCase &snapshotCase);
[[nodiscard]] Result<std::string, std::string>
formatSnapshotEffectiveConfigJson(const SnapshotCase &snapshotCase,
                                  const SnapshotRunOptions &options);

[[nodiscard]] SnapshotRunResult
captureSnapshotCase(SnapshotCase snapshotCase,
                    const SnapshotRunOptions &options = {});
[[nodiscard]] SnapshotRunResult
compareSnapshotCase(SnapshotCase snapshotCase,
                    const SnapshotRunOptions &options = {});
[[nodiscard]] SnapshotRunResult
runSnapshotCase(SnapshotCase snapshotCase,
                const SnapshotRunOptions &options = {});
[[nodiscard]] SnapshotSuiteRunResult
runSnapshotSuite(std::vector<SnapshotCase> snapshotCases,
                 std::string_view suite,
                 const SnapshotRunOptions &options = {});

} // namespace nuri::tools::snapshot

