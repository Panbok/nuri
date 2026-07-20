#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/tools/benchmark/benchmark_case.h"
#include "nuri/tools/benchmark/benchmark_environment.h"
#include "nuri/tools/benchmark/benchmark_measurements.h"
#include "nuri/tools/benchmark/benchmark_stats.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkFrameRecord {
  uint64_t frameIndex = 0u;
  uint32_t sampleIndex = 0u;
  bool measured = false;
  BenchmarkFrameMeasurements measurements{};
  RenderFrameMetrics metrics{};
};

struct BenchmarkSampleStats {
  uint32_t sampleIndex = 0u;
  // Absence means the configured warmup window could not be evaluated.
  std::optional<bool> warmupStable{};
  uint64_t measuredFrameStart = 0u;
  uint32_t measuredFrameCount = 0u;
  std::map<std::string, MetricStats> stats{};
  std::vector<std::string> warnings{};
};

struct BenchmarkProfileInfo {
  std::string id{};
  bool profileAuthoritative = false;
  bool authoritative = false;
  uint32_t minimumRepetitions = 0u;
  uint32_t completedRepetitions = 0u;
  bool repetitionRequirementSatisfied = false;
  std::string repetitionUnit = "not-collected";
  std::string warmupStabilityPolicy = "unknown";
  std::string warmupStabilityStatus = "unknown";
  uint32_t warmupWindowFrames = 0u;
  double warmupMaxDriftPercent = 0.0;
  std::vector<std::string> requiredMetrics{};
  std::vector<std::string> authorityBlockers{};
};

struct BenchmarkRepeatObservationInfo {
  // Sample windows share one renderer process and are useful investigative
  // observations, but are not independent profile repetitions.
  std::string unit = "in-process-sample-window";
  bool independent = false;
  uint32_t count = 0u;
};

struct BenchmarkTimingDrain {
  bool drainComplete = true;
  uint32_t drainFrames = 0u;
  uint32_t drainTimeoutMs = 0u;
  uint32_t missingGpuTimingFrames = 0u;
  uint32_t scopeContainmentViolations = 0u;
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

struct BenchmarkTracyZoneStats {
  std::string name{};
  std::filesystem::path sourceFile{};
  uint32_t sourceLine = 0u;
  uint64_t totalNs = 0u;
  double totalPercent = 0.0;
  uint64_t count = 0u;
  double meanNs = 0.0;
  uint64_t medianNs = 0u;
  uint64_t p95Ns = 0u;
  uint64_t minNs = 0u;
  uint64_t maxNs = 0u;
  double stddevNs = 0.0;
};

struct BenchmarkTracyFlameNode {
  std::string name{};
  std::string thread{};
  std::filesystem::path sourceFile{};
  uint32_t sourceLine = 0u;
  uint64_t totalNs = 0u;
  uint64_t selfNs = 0u;
  uint64_t count = 0u;
  std::vector<BenchmarkTracyFlameNode> children{};
};

struct BenchmarkTracyFlameGraph {
  std::filesystem::path eventsCsvPath{};
  std::string eventsExportCommand{};
  bool frameScoped = false;
  uint64_t eventCount = 0u;
  uint64_t retainedNodeCount = 0u;
  uint32_t maxDepth = 0u;
  BenchmarkTracyFlameNode root{};
};

struct BenchmarkTracyReport {
  bool available = false;
  std::filesystem::path tracePath{};
  std::filesystem::path captureLogPath{};
  std::filesystem::path zonesCsvPath{};
  std::filesystem::path selfZonesCsvPath{};
  std::filesystem::path gpuEventsCsvPath{};
  std::filesystem::path exportLogPath{};
  std::string captureCommand{};
  std::string zonesExportCommand{};
  std::string selfZonesExportCommand{};
  std::string gpuEventsExportCommand{};
  bool gpuEventsExportSupported = false;
  uint64_t captureFrameCount = 0u;
  double captureTimeSpanSeconds = 0.0;
  uint64_t captureZoneEventCount = 0u;
  uint64_t gpuZoneEventCount = 0u;
  std::vector<BenchmarkTracyZoneStats> zones{};
  std::vector<BenchmarkTracyZoneStats> selfZones{};
  std::vector<BenchmarkTracyZoneStats> gpuZones{};
  BenchmarkTracyFlameGraph flameGraph{};
};

struct BenchmarkReport {
  uint32_t schemaVersion = 1u;
  std::string kind = "nuri.benchmark.report";
  std::string generatedAtUtc{};
  std::string command{};
  BenchmarkEnvironment environment{};
  BenchmarkCase benchmarkCase{};
  BenchmarkProfileInfo profile{};
  BenchmarkRepeatObservationInfo repeatObservations{};
  BenchmarkRunInfo run{};
  BenchmarkArtifactInfo artifacts{};
  BenchmarkTracyReport tracy{};
  std::vector<BenchmarkFrameRecord> frames{};
  std::vector<BenchmarkSampleStats> sampleStats{};
  std::map<std::string, MetricStats> stats{};
  BenchmarkTimingDrain timingDrain{};
  std::vector<std::string> unavailableMetrics{};
  std::vector<std::string> unregisteredObservedMetrics{};
  std::vector<std::string> warnings{};
};

[[nodiscard]] Result<std::string, std::string>
writeBenchmarkReportJson(const BenchmarkReport &report, bool verboseFrames);
[[nodiscard]] Result<bool, std::string>
writeBenchmarkReportFile(const BenchmarkReport &report,
                         const std::filesystem::path &path, bool verboseFrames);
[[nodiscard]] Result<BenchmarkReport, std::string>
readBenchmarkReportFile(const std::filesystem::path &path);
void computeBenchmarkReportStats(BenchmarkReport &report);

} // namespace nuri::tools::benchmark
