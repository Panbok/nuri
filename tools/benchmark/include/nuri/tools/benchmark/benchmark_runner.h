#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/benchmark/benchmark_case.h"
#include "nuri/tools/benchmark/benchmark_report.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkRunOptions {
  std::optional<uint32_t> samplesOverride{};
  std::filesystem::path jsonOut{};
  std::filesystem::path artifactDir{};
  bool dryRun = false;
  bool printEffectiveConfig = false;
  bool tracyDiagnostic = false;
  bool verboseFrames = false;
  std::string command{};
};

struct BenchmarkRunResult {
  BenchmarkExitCode exitCode = BenchmarkExitCode::Success;
  BenchmarkReport report{};
  std::filesystem::path reportPath{};
  std::string message{};
};

struct BenchmarkSuiteRunResult {
  BenchmarkExitCode exitCode = BenchmarkExitCode::Success;
  std::vector<BenchmarkRunResult> caseResults{};
  std::filesystem::path reportPath{};
  std::string message{};
};

[[nodiscard]] Result<std::string, std::string>
formatBenchmarkCaseListJson(const std::vector<BenchmarkCase> &cases,
                            std::string_view suite = {});
[[nodiscard]] std::string
formatBenchmarkCaseListText(const std::vector<BenchmarkCase> &cases,
                            std::string_view suite = {});
[[nodiscard]] Result<std::string, std::string>
formatBenchmarkCaseExplanationJson(const BenchmarkCase &benchmarkCase);
[[nodiscard]] std::string
formatBenchmarkCaseExplanationText(const BenchmarkCase &benchmarkCase);
[[nodiscard]] Result<std::string, std::string>
formatEffectiveConfigJson(const BenchmarkCase &benchmarkCase,
                          const BenchmarkRunOptions &options);

[[nodiscard]] BenchmarkRunResult
runBenchmarkCase(BenchmarkCase benchmarkCase,
                 const BenchmarkRunOptions &options = {});
[[nodiscard]] BenchmarkSuiteRunResult
runBenchmarkSuite(std::vector<BenchmarkCase> benchmarkCases,
                  std::string_view suite,
                  const BenchmarkRunOptions &options = {});

} // namespace nuri::tools::benchmark
