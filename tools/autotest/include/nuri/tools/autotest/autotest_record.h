#pragma once

#include "nuri/tools/autotest/autotest_runner.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::autotest {

struct AutotestBaselineMetadataCompatibility {
  bool compatible = true;
  std::vector<std::string> errors{};
};

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
