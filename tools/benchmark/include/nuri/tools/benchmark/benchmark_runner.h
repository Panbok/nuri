#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/benchmark/benchmark_case.h"
#include "nuri/tools/benchmark/benchmark_report.h"
#include "nuri/tools/core/baseline_profile.h"

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkRunOptions {
  std::optional<uint32_t> samplesOverride{};
  std::optional<uint32_t> isolatedRepetitions{};
  std::filesystem::path processExecutable{};
  std::chrono::milliseconds repetitionTimeout = std::chrono::minutes(5);
  std::filesystem::path jsonOut{};
  // Internal/embedding override for the shared v2 result envelope. The
  // detailed benchmark report remains at jsonOut for v1 reader compatibility.
  std::filesystem::path envelopeOut{};
  std::filesystem::path artifactDir{};
  bool dryRun = false;
  bool printEffectiveConfig = false;
  bool tracyDiagnostic = false;
  bool verboseFrames = false;
  bool internalIsolatedChild = false;
  std::string baselineProfileId{};
  bool baselineProfileAuthoritative = false;
  uint32_t baselineProfileMinimumRepetitions = 0u;
  std::string baselineProfileWarmupStability = "unknown";
  uint32_t baselineProfileWarmupWindowFrames = 0u;
  double baselineProfileWarmupMaxDriftPercent = 0.0;
  std::vector<std::string> baselineProfileRequiredMetrics{};
  std::optional<nuri::tools::core::BaselineProfile> baselineProfile{};
  std::string baselineProfileWarning{};
  std::string command{};
};

struct BenchmarkRepetitionResult {
  uint32_t index = 0u;
  BenchmarkExitCode exitCode = BenchmarkExitCode::RuntimeError;
  bool completed = false;
  bool timedOut = false;
  std::filesystem::path workspace{};
  std::filesystem::path reportPath{};
  std::filesystem::path envelopePath{};
  std::filesystem::path stdoutLogPath{};
  std::filesystem::path stderrLogPath{};
  std::string message{};
};

struct BenchmarkRunResult {
  BenchmarkExitCode exitCode = BenchmarkExitCode::Success;
  BenchmarkReport report{};
  std::filesystem::path reportPath{};
  std::filesystem::path envelopePath{};
  std::vector<BenchmarkRepetitionResult> repetitions{};
  std::string message{};
};

struct BenchmarkSuiteRunResult {
  BenchmarkExitCode exitCode = BenchmarkExitCode::Success;
  std::vector<BenchmarkRunResult> caseResults{};
  std::filesystem::path reportPath{};
  std::string message{};
};

[[nodiscard]] Result<bool, BenchmarkExitCode>
checkBenchmarkGpuRequirements(const BenchmarkRequirements &requirements,
                              const GpuMultisampleCapabilities &capabilities,
                              std::string &message);

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
[[nodiscard]] BenchmarkRunResult
runBenchmarkCaseIsolated(BenchmarkCase benchmarkCase,
                         const BenchmarkRunOptions &options);
[[nodiscard]] BenchmarkSuiteRunResult
runBenchmarkSuite(std::vector<BenchmarkCase> benchmarkCases,
                  std::string_view suite,
                  const BenchmarkRunOptions &options = {});

} // namespace nuri::tools::benchmark
