#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/benchmark/benchmark_compare.h"
#include "nuri/tools/benchmark/benchmark_report.h"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkGraphOptions {
  std::vector<std::string> metrics{};
  std::vector<std::string> statistics{};
  std::string title = "Nuri Benchmark Results";
};

[[nodiscard]] Result<std::string, std::string>
writeBenchmarkGraphHtml(std::span<const BenchmarkReport> reports,
                        const BenchmarkGraphOptions &options = {});
[[nodiscard]] Result<bool, std::string>
writeBenchmarkGraphHtmlFile(std::span<const BenchmarkReport> reports,
                            const BenchmarkGraphOptions &options,
                            const std::filesystem::path &path);
[[nodiscard]] Result<std::string, std::string>
writeBenchmarkComparisonHtml(const BenchmarkReport &baseline,
                             const BenchmarkReport &current,
                             const BenchmarkComparisonReport &comparison,
                             const BenchmarkGraphOptions &options = {});
[[nodiscard]] Result<bool, std::string> writeBenchmarkComparisonHtmlFile(
    const BenchmarkReport &baseline, const BenchmarkReport &current,
    const BenchmarkComparisonReport &comparison,
    const BenchmarkGraphOptions &options, const std::filesystem::path &path);

} // namespace nuri::tools::benchmark
