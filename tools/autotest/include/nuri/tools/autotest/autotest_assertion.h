#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/tools/autotest/autotest_case.h"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::autotest {

enum class AutotestAssertionStatus {
  Pass,
  Warn,
  Fail,
  Unavailable,
  Invalid,
};

struct AutotestAssertionResult {
  std::string id{};
  std::string metric{};
  std::string statistic{};
  AutotestAssertionStatus status = AutotestAssertionStatus::Pass;
  std::string statusReason = "passed";
  double actual = 0.0;
  bool hasActual = false;
  uint32_t sampleCount = 0u;
};

struct AutotestMetricStats {
  uint32_t count = 0u;
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double median = 0.0;
  double p95 = 0.0;
  double variance = 0.0;
};

[[nodiscard]] std::string
autotestAssertionStatusName(AutotestAssertionStatus status);
void flattenAutotestRendererMetrics(std::map<std::string, double> &out,
                                    const RenderFrameMetrics &metrics);
void applyAutotestGpuTimingReport(
    std::map<uint64_t, std::map<std::string, double>> &frames,
    const GpuTimingReport &report);
[[nodiscard]] AutotestAssertionResult
evaluateAutotestAssertion(const AutotestMetricAssertion &assertion,
                          const std::map<std::string, double> &measurements);
[[nodiscard]] std::vector<AutotestAssertionResult> evaluateAutotestAssertions(
    const std::vector<AutotestMetricAssertion> &assertions,
    const std::map<std::string, double> &measurements);
[[nodiscard]] Result<AutotestMetricStats, std::string>
computeAutotestMetricStats(std::vector<double> values);
[[nodiscard]] AutotestAssertionResult evaluateAutotestMetricWindowAssertion(
    const AutotestMetricWindowAssertion &assertion,
    const std::map<uint64_t, std::map<std::string, double>> &frames,
    uint32_t startFrame, uint32_t endFrame);
[[nodiscard]] std::vector<AutotestAssertionResult>
evaluateAutotestMetricWindowAssertions(
    const AutotestMetricWindow &window,
    const std::map<uint64_t, std::map<std::string, double>> &frames);
[[nodiscard]] bool autotestAssertionFailuresAreFatal(
    const std::vector<AutotestAssertionResult> &results);

} // namespace nuri::tools::autotest
