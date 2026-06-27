#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/tools/benchmark/benchmark_case.h"
#include "nuri/tools/benchmark/benchmark_environment.h"
#include "nuri/tools/benchmark/benchmark_stats.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkFrameRecord {
  uint64_t frameIndex = 0u;
  uint32_t sampleIndex = 0u;
  bool measured = false;
  std::map<std::string, double> measurements{};
  RenderFrameMetrics metrics{};
};

struct BenchmarkSampleStats {
  uint32_t sampleIndex = 0u;
  bool warmupStable = true;
  uint64_t measuredFrameStart = 0u;
  uint32_t measuredFrameCount = 0u;
  std::map<std::string, MetricStats> stats{};
  std::vector<std::string> warnings{};
};

struct BenchmarkTimingDrain {
  bool drainComplete = true;
  uint32_t drainFrames = 0u;
  uint32_t drainTimeoutMs = 0u;
  uint32_t missingGpuTimingFrames = 0u;
  uint64_t droppedGpuTimingReports = 0u;
};

struct BenchmarkRunInfo {
  uint32_t samples = 1u;
  uint32_t warmupFrames = 0u;
  uint32_t measurementFrames = 0u;
  uint32_t cooldownFrames = 0u;
  uint32_t maxDrainFrames = 0u;
  uint32_t drainTimeoutMs = 0u;
  bool validForComparison = true;
  double fixedDeltaSeconds = 1.0 / 60.0;
};

struct BenchmarkArtifactInfo {
  std::filesystem::path artifactDir{};
  std::vector<std::filesystem::path> caseReports{};
  std::vector<std::filesystem::path> tracyArtifacts{};
};

struct BenchmarkReport {
  uint32_t schemaVersion = 1u;
  std::string kind = "nuri.benchmark.report";
  std::string generatedAtUtc{};
  std::string command{};
  BenchmarkEnvironment environment{};
  BenchmarkCase benchmarkCase{};
  BenchmarkRunInfo run{};
  BenchmarkArtifactInfo artifacts{};
  std::vector<BenchmarkFrameRecord> frames{};
  std::vector<BenchmarkSampleStats> sampleStats{};
  std::map<std::string, MetricStats> stats{};
  BenchmarkTimingDrain timingDrain{};
  std::vector<std::string> unavailableMetrics{};
  std::vector<std::string> warnings{};
};

[[nodiscard]] Result<std::string, std::string>
writeBenchmarkReportJson(const BenchmarkReport &report, bool verboseFrames);
[[nodiscard]] Result<bool, std::string>
writeBenchmarkReportFile(const BenchmarkReport &report,
                         const std::filesystem::path &path,
                         bool verboseFrames);
[[nodiscard]] Result<BenchmarkReport, std::string>
readBenchmarkReportFile(const std::filesystem::path &path);
void computeBenchmarkReportStats(BenchmarkReport &report);

} // namespace nuri::tools::benchmark
