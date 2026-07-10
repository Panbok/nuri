#pragma once

#include "nuri/tools/benchmark/benchmark_baseline.h"
#include "nuri/tools/benchmark/benchmark_compare.h"
#include "nuri/tools/benchmark/benchmark_runner.h"
#include "nuri/tools/core/baseline_profile.h"

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkCheckOptions {
  std::filesystem::path processExecutable{};
  std::filesystem::path artifactRoot{};
  std::filesystem::path baselineRoot{};
  std::chrono::milliseconds repetitionTimeout = std::chrono::minutes(5);
  bool force = false;
  std::string requestedSelection{};
  std::string command{};
};

struct BenchmarkCheckCaseResult {
  std::string caseId{};
  BenchmarkExitCode exitCode = BenchmarkExitCode::RuntimeError;
  BenchmarkBaselineVerification baselineVerification{};
  BenchmarkRunResult run{};
  BenchmarkComparisonReport comparison{};
  std::filesystem::path baselineVerificationPath{};
  std::filesystem::path comparisonPath{};
};

struct BenchmarkCheckResult {
  BenchmarkExitCode exitCode = BenchmarkExitCode::RuntimeError;
  std::filesystem::path workspace{};
  std::filesystem::path envelopePath{};
  std::vector<BenchmarkCheckCaseResult> cases{};
  std::string message{};
};

// Runs a governed gate over an already-selected, deterministic case list.
// Baselines are verified for every case before any renderer process starts.
[[nodiscard]] BenchmarkCheckResult
checkBenchmarkCases(std::vector<BenchmarkCase> selectedCases,
                    const nuri::tools::core::BaselineProfile &profile,
                    const BenchmarkCheckOptions &options = {});

} // namespace nuri::tools::benchmark
