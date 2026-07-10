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
  const size_t upperIndex = std::min(lowerIndex + 1u, sortedValues.size() - 1u);
  const double fraction = h - floorH;
  return sortedValues[lowerIndex] +
         fraction * (sortedValues[upperIndex] - sortedValues[lowerIndex]);
}

Result<MetricStats, std::string>
computeMetricStats(std::vector<double> values) {
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

Result<RepeatComparisonStats, std::string>
computeRepeatComparison(std::vector<double> baselineRepetitions,
                        std::vector<double> currentRepetitions,
                        double nearZeroEpsilon) {
  if (!std::isfinite(nearZeroEpsilon) || nearZeroEpsilon <= 0.0) {
    return Result<RepeatComparisonStats, std::string>::makeError(
        "computeRepeatComparison: near-zero epsilon must be finite and "
        "positive");
  }
  auto baseline = computeMetricStats(baselineRepetitions);
  if (baseline.hasError()) {
    return Result<RepeatComparisonStats, std::string>::makeError(
        "computeRepeatComparison: invalid baseline repetitions: " +
        baseline.error());
  }
  auto current = computeMetricStats(currentRepetitions);
  if (current.hasError()) {
    return Result<RepeatComparisonStats, std::string>::makeError(
        "computeRepeatComparison: invalid current repetitions: " +
        current.error());
  }

  RepeatComparisonStats comparison{};
  comparison.baselineRepetitions = baseline.value().count;
  comparison.currentRepetitions = current.value().count;
  comparison.baselineMedian = baseline.value().median;
  comparison.currentMedian = current.value().median;
  comparison.absoluteDelta =
      comparison.currentMedian - comparison.baselineMedian;
  comparison.percentDeltaDefined =
      std::abs(comparison.baselineMedian) > nearZeroEpsilon;
  if (comparison.percentDeltaDefined) {
    comparison.percentDelta =
        comparison.absoluteDelta / std::abs(comparison.baselineMedian) * 100.0;
  }

  const double baselineScale = 1.4826 * baseline.value().mad;
  const double currentScale = 1.4826 * current.value().mad;
  const double pooledRobustScale =
      std::max(nearZeroEpsilon, 0.5 * (baselineScale + currentScale));
  comparison.robustEffect = comparison.absoluteDelta / pooledRobustScale;
  comparison.noiseScore = std::max(baseline.value().coefficientOfVariation,
                                   current.value().coefficientOfVariation);

  const double baselineVariance =
      baseline.value().stddev * baseline.value().stddev;
  const double currentVariance =
      current.value().stddev * current.value().stddev;
  const double standardError =
      std::sqrt(baselineVariance / static_cast<double>(baseline.value().count) +
                currentVariance / static_cast<double>(current.value().count));
  const double margin = 1.96 * standardError;
  comparison.confidenceLow = comparison.absoluteDelta - margin;
  comparison.confidenceHigh = comparison.absoluteDelta + margin;
  comparison.lowConfidence =
      baseline.value().count < 3u || current.value().count < 3u;
  return Result<RepeatComparisonStats, std::string>::makeResult(comparison);
}

} // namespace nuri::tools::benchmark
