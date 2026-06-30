#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/autotest/autotest_case.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::autotest {

struct AutotestManifestLoadOptions {
  std::filesystem::path caseRoot{};
  std::filesystem::path repoRoot{};
};

[[nodiscard]] std::filesystem::path defaultAutotestCaseRoot();
[[nodiscard]] Result<AutotestCase, std::string>
loadAutotestCaseManifest(const std::filesystem::path &path);
[[nodiscard]] Result<std::vector<AutotestCase>, std::string>
discoverAutotestCases(const AutotestManifestLoadOptions &options = {});
[[nodiscard]] const AutotestCase *
findAutotestCaseById(const std::vector<AutotestCase> &cases,
                     std::string_view id);
[[nodiscard]] std::vector<const AutotestCase *>
filterAutotestCasesBySuite(const std::vector<AutotestCase> &cases,
                           std::string_view suite);
[[nodiscard]] Result<std::filesystem::path, std::string>
resolveAutotestPath(std::string_view base, const std::filesystem::path &path);

} // namespace nuri::tools::autotest
