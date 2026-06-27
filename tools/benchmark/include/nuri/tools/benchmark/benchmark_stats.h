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
  double max = 0.0;
  double mean = 0.0;
  double stddev = 0.0;
  double mad = 0.0;
  double iqr = 0.0;
  double coefficientOfVariation = 0.0;
};

[[nodiscard]] double percentileR7(std::vector<double> sortedValues,
                                  double percentile);
[[nodiscard]] Result<MetricStats, std::string>
computeMetricStats(std::vector<double> values);

} // namespace nuri::tools::benchmark
