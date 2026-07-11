#pragma once

#include "nuri/core/result.h"

#include <cstddef>
#include <string>
#include <vector>

namespace nuri::tools::benchmark {

struct MetricStats {
  size_t count = 0u;
  double min = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double stddev = 0.0;
  double mad = 0.0;
  double iqr = 0.0;
  double coefficientOfVariation = 0.0;
};

struct RepeatComparisonStats {
  size_t baselineRepetitions = 0u;
  size_t currentRepetitions = 0u;
  double baselineMedian = 0.0;
  double currentMedian = 0.0;
  double absoluteDelta = 0.0;
  double percentDelta = 0.0;
  bool percentDeltaDefined = false;
  double robustEffect = 0.0;
  double confidenceLow = 0.0;
  double confidenceHigh = 0.0;
  double noiseScore = 0.0;
  bool lowConfidence = true;
};

[[nodiscard]] double percentileR7(std::vector<double> sortedValues,
                                  double percentile);
[[nodiscard]] Result<MetricStats, std::string>
computeMetricStats(std::vector<double> values);
[[nodiscard]] Result<RepeatComparisonStats, std::string>
computeRepeatComparison(std::vector<double> baselineRepetitions,
                        std::vector<double> currentRepetitions,
                        double nearZeroEpsilon = 1.0e-6);

} // namespace nuri::tools::benchmark
