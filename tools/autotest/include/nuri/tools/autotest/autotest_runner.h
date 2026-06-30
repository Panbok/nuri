#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/autotest/autotest_report.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::autotest {

struct AutotestRunOptions {
  std::filesystem::path jsonOut{};
  std::filesystem::path htmlOut{};
  std::filesystem::path artifactDir{};
  std::string baselineProfile = "local-lvk-visible";
  std::string windowMode = "visible";
  bool dryRun = false;
  bool printEffectiveConfig = false;
  bool verboseFrames = false;
  std::string command{};
};

struct AutotestRunResult {
  AutotestExitCode exitCode = AutotestExitCode::Success;
  AutotestReport report{};
  std::filesystem::path reportPath{};
  std::filesystem::path htmlPath{};
  std::string message{};
};

struct AutotestSuiteRunResult {
  AutotestExitCode exitCode = AutotestExitCode::Success;
  std::vector<AutotestRunResult> caseResults{};
  std::filesystem::path reportPath{};
  std::filesystem::path htmlPath{};
  std::string message{};
};

[[nodiscard]] Result<std::string, std::string>
formatAutotestCaseListJson(const std::vector<AutotestCase> &cases,
                           std::string_view suite = {});
[[nodiscard]] std::string
formatAutotestCaseListText(const std::vector<AutotestCase> &cases,
                           std::string_view suite = {});
[[nodiscard]] Result<std::string, std::string>
formatAutotestCaseExplanationJson(const AutotestCase &testCase);
[[nodiscard]] std::string
formatAutotestCaseExplanationText(const AutotestCase &testCase);
[[nodiscard]] Result<std::string, std::string>
formatAutotestEffectiveConfigJson(const AutotestCase &testCase,
                                  const AutotestRunOptions &options);

[[nodiscard]] AutotestRunResult
runAutotestCase(AutotestCase testCase, const AutotestRunOptions &options = {});
[[nodiscard]] AutotestSuiteRunResult
runAutotestSuite(std::vector<AutotestCase> testCases, std::string_view suite,
                 const AutotestRunOptions &options = {});

} // namespace nuri::tools::autotest
