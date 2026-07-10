#pragma once

#include "nuri/tools/autotest/autotest_runner.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::autotest {

struct AutotestBaselineMetadataCompatibility {
  bool compatible = true;
  std::vector<std::string> errors{};
};

struct AutotestBaselinePlanEntry {
  std::string path{};
  std::string operation{};
  std::string previousDigest{};
  std::string sourceDigest{};
};

struct AutotestBaselinePlan {
  std::string caseId{};
  std::string suite{};
  std::string profileId{};
  std::string reason{};
  std::string actor{};
  std::string sourceCommit{};
  std::string sourceReportDigest{};
  std::string historyDigest{};
  uint32_t historyFileCount = 0u;
  std::string digest{};
  std::vector<AutotestBaselinePlanEntry> entries{};
};

struct AutotestBaselineApprovalOptions {
  std::filesystem::path baselineRoot{};
  bool failAfterBackupForTesting = false;
};

[[nodiscard]] std::filesystem::path defaultAutotestBaselineRoot();
[[nodiscard]] Result<std::string, std::string>
inspectAutotestBaseline(const AutotestCase &expectedCase,
                        std::string_view baselineProfile,
                        const std::filesystem::path &baselineRoot = {});
[[nodiscard]] Result<bool, std::string>
verifyAutotestBaseline(const AutotestCase &expectedCase,
                       std::string_view baselineProfile,
                       const std::filesystem::path &baselineRoot = {});
[[nodiscard]] Result<AutotestBaselinePlan, std::string>
planAutotestBaselines(const AutotestCase &expectedCase,
                      const AutotestReport &report,
                      std::string_view baselineProfile, std::string_view reason,
                      std::string_view actor,
                      const AutotestBaselineApprovalOptions &options = {});
[[nodiscard]] Result<std::string, std::string>
writeAutotestBaselinePlanJson(const AutotestBaselinePlan &plan);
[[nodiscard]] Result<bool, std::string> approveAutotestBaselines(
    const AutotestCase &expectedCase, const AutotestReport &report,
    std::string_view baselineProfile, std::string_view reason,
    std::string_view confirmPlanDigest, std::string_view actor,
    const AutotestBaselineApprovalOptions &options = {});

[[nodiscard]] Result<bool, std::string>
writeAutotestRecordMetadataFile(const AutotestReport &report,
                                const std::filesystem::path &caseDir,
                                std::string_view baselineProfile);
[[nodiscard]] Result<AutotestBaselineMetadataCompatibility, std::string>
validateAutotestBaselineMetadataFile(const AutotestCase &testCase,
                                     const AutotestEnvironment &environment,
                                     const std::filesystem::path &caseDir,
                                     std::string_view baselineProfile);
[[nodiscard]] AutotestRunResult
recordAutotestCase(AutotestCase testCase, const AutotestRunOptions &options);

} // namespace nuri::tools::autotest
