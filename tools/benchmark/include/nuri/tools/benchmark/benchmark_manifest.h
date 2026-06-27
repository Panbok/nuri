#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/benchmark/benchmark_case.h"

#include <filesystem>
#include <string>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkManifestLoadOptions {
  std::filesystem::path caseRoot{};
  std::filesystem::path repoRoot{};
};

[[nodiscard]] std::filesystem::path defaultBenchmarkCaseRoot();
[[nodiscard]] Result<BenchmarkCase, std::string>
loadBenchmarkCaseManifest(const std::filesystem::path &path);
[[nodiscard]] Result<std::vector<BenchmarkCase>, std::string>
discoverBenchmarkCases(const BenchmarkManifestLoadOptions &options = {});
[[nodiscard]] const BenchmarkCase *
findBenchmarkCaseById(const std::vector<BenchmarkCase> &cases,
                      std::string_view id);
[[nodiscard]] std::vector<const BenchmarkCase *>
filterBenchmarkCasesBySuite(const std::vector<BenchmarkCase> &cases,
                            std::string_view suite);
[[nodiscard]] Result<std::filesystem::path, std::string>
resolveBenchmarkPath(std::string_view base, const std::filesystem::path &path);

} // namespace nuri::tools::benchmark
