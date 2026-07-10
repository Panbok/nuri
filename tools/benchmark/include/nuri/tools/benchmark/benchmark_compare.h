#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/benchmark/benchmark_case.h"
#include "nuri/tools/benchmark/benchmark_report.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkCompareOptions {
  bool force = false;
};

struct BenchmarkMetricComparison {
  std::string metricId{};
  std::string statistic{};
  double baseline = 0.0;
  double current = 0.0;
  double delta = 0.0;
  double deltaPercent = 0.0;
  bool deltaPercentAvailable = true;
  std::string status = "pass";
  bool required = false;
  std::optional<RepeatComparisonStats> repeatComparison{};
  std::string repeatObservationUnit{};
  bool repeatObservationsIndependent = false;
};

struct BenchmarkComparisonReport {
  uint32_t schemaVersion = 1u;
  std::string kind = "nuri.benchmark.comparison";
  bool valid = true;
  bool authoritative = false;
  bool regression = false;
  std::vector<std::string> errors{};
  std::vector<std::string> warnings{};
  std::vector<BenchmarkMetricComparison> metrics{};
};

[[nodiscard]] BenchmarkComparisonReport
compareBenchmarkReports(const BenchmarkReport &current,
                        const BenchmarkReport &baseline,
                        const BenchmarkCompareOptions &options = {});
[[nodiscard]] Result<std::string, std::string>
writeBenchmarkComparisonJson(const BenchmarkComparisonReport &report);
[[nodiscard]] Result<bool, std::string>
writeBenchmarkComparisonFile(const BenchmarkComparisonReport &report,
                             const std::filesystem::path &path);
[[nodiscard]] Result<BenchmarkComparisonReport, std::string>
readBenchmarkComparisonFile(const std::filesystem::path &path);

} // namespace nuri::tools::benchmark
