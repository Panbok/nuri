#include "nuri/tools/benchmark/benchmark_stats.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nuri::tools::benchmark {

double percentileR7(std::vector<double> sortedValues, double percentile) {
  if (sortedValues.empty()) {
    return 0.0;
  }
  std::sort(sortedValues.begin(), sortedValues.end());
  const double p = std::clamp(percentile, 0.0, 1.0);
  const double h = 1.0 + (static_cast<double>(sortedValues.size()) - 1.0) * p;
  const double floorH = std::floor(h);
  const size_t lowerIndex = static_cast<size_t>(floorH) - 1u;
  const size_t upperIndex =
      std::min(lowerIndex + 1u, sortedValues.size() - 1u);
  const double fraction = h - floorH;
  return sortedValues[lowerIndex] +
         fraction * (sortedValues[upperIndex] - sortedValues[lowerIndex]);
}

Result<MetricStats, std::string> computeMetricStats(std::vector<double> values) {
  if (values.empty()) {
    return Result<MetricStats, std::string>::makeError(
        "computeMetricStats: no values");
  }
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return Result<MetricStats, std::string>::makeError(
          "computeMetricStats: non-finite value");
    }
  }

  std::sort(values.begin(), values.end());
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());

  MetricStats stats{};
  stats.count = values.size();
  stats.min = values.front();
  stats.max = values.back();
  stats.mean = mean;
  stats.median = percentileR7(values, 0.50);
  stats.p90 = percentileR7(values, 0.90);
  stats.p95 = percentileR7(values, 0.95);

  if (values.size() >= 2u) {
    double variance = 0.0;
    for (const double value : values) {
      const double diff = value - mean;
      variance += diff * diff;
    }
    variance /= static_cast<double>(values.size() - 1u);
    stats.stddev = std::sqrt(variance);
    stats.coefficientOfVariation =
        std::abs(mean) > 1.0e-12 ? stats.stddev / std::abs(mean) : 0.0;

    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (const double value : values) {
      deviations.push_back(std::abs(value - stats.median));
    }
    stats.mad = percentileR7(std::move(deviations), 0.50);
    stats.iqr = percentileR7(values, 0.75) - percentileR7(values, 0.25);
  }

  return Result<MetricStats, std::string>::makeResult(stats);
}

} // namespace nuri::tools::benchmark
