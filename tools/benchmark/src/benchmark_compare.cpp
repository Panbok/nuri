#include "nuri/tools/benchmark/benchmark_compare.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>

#include <yyjson.h>

namespace nuri::tools::benchmark {
namespace {

[[nodiscard]] double statValue(const MetricStats &stats,
                               std::string_view statistic) {
  if (statistic == "median") {
    return stats.median;
  }
  if (statistic == "p95") {
    return stats.p95;
  }
  if (statistic == "p90") {
    return stats.p90;
  }
  if (statistic == "mean") {
    return stats.mean;
  }
  return 0.0;
}

void addCompatibilityError(BenchmarkComparisonReport &out, bool force,
                           bool condition, std::string message) {
  if (!condition) {
    return;
  }
  if (force) {
    out.warnings.push_back(std::move(message));
  } else {
    out.valid = false;
    out.errors.push_back(std::move(message));
  }
}

[[nodiscard]] bool containsMetric(const std::vector<std::string> &metrics,
                                  std::string_view metric) {
  return std::find(metrics.begin(), metrics.end(), metric) != metrics.end();
}

void compareMetricStatistic(BenchmarkComparisonReport &out,
                            const BenchmarkReport &current,
                            const BenchmarkReport &baseline,
                            std::string_view metricId,
                            std::string_view statistic, bool required) {
  const auto currentIt = current.stats.find(std::string(metricId));
  const auto baselineIt = baseline.stats.find(std::string(metricId));
  if (currentIt == current.stats.end() || baselineIt == baseline.stats.end()) {
    if (required) {
      out.valid = false;
      out.errors.push_back("missing required metric '" + std::string(metricId) +
                           "'");
    } else if (baselineIt != baseline.stats.end()) {
      out.warnings.push_back("optional metric '" + std::string(metricId) +
                             "' unavailable in current report");
    }
    return;
  }

  const double currentValue = statValue(currentIt->second, statistic);
  const double baselineValue = statValue(baselineIt->second, statistic);
  const double delta = currentValue - baselineValue;
  const double deltaPercent =
      baselineValue != 0.0 ? (delta / baselineValue) * 100.0 : 0.0;
  const BenchmarkThresholds &thresholds = current.benchmarkCase.thresholds;

  BenchmarkMetricComparison metric{};
  metric.metricId = std::string(metricId);
  metric.statistic = std::string(statistic);
  metric.baseline = baselineValue;
  metric.current = currentValue;
  metric.delta = delta;
  metric.deltaPercent = deltaPercent;
  metric.required = required;

  if (delta > thresholds.failAbsoluteMs &&
      deltaPercent > thresholds.failPercent) {
    metric.status = required ? "fail" : "warn";
    if (required) {
      out.regression = true;
    }
  } else if (delta > thresholds.warnAbsoluteMs &&
             deltaPercent > thresholds.warnPercent) {
    metric.status = "warn";
  } else {
    metric.status = "pass";
  }
  out.metrics.push_back(std::move(metric));
}

} // namespace

BenchmarkComparisonReport
compareBenchmarkReports(const BenchmarkReport &current,
                        const BenchmarkReport &baseline,
                        const BenchmarkCompareOptions &options) {
  BenchmarkComparisonReport out{};
  addCompatibilityError(out, options.force,
                        current.benchmarkCase.id != baseline.benchmarkCase.id,
                        "case id mismatch");
  addCompatibilityError(
      out, options.force,
      current.benchmarkCase.resolution != baseline.benchmarkCase.resolution,
      "resolution mismatch");
  addCompatibilityError(out, options.force,
                        current.environment.gpuBackend !=
                            baseline.environment.gpuBackend,
                        "backend mismatch");
  addCompatibilityError(out, options.force,
                        current.environment.buildType !=
                            baseline.environment.buildType,
                        "build type mismatch");
  addCompatibilityError(out, options.force,
                        current.environment.cmakeToolProfile !=
                            baseline.environment.cmakeToolProfile,
                        "tool profile mismatch");
  addCompatibilityError(out, options.force,
                        current.environment.tracyEnabled !=
                                baseline.environment.tracyEnabled ||
                            current.environment.tracyGpuEnabled !=
                                baseline.environment.tracyGpuEnabled ||
                            current.environment.devChecks !=
                                baseline.environment.devChecks,
                        "Tracy/dev-check flags mismatch");
  addCompatibilityError(out, options.force, !current.run.validForComparison,
                        "current report is not valid for comparison");
  addCompatibilityError(out, options.force, !baseline.run.validForComparison,
                        "baseline report is not valid for comparison");
  if (!out.valid && !options.force) {
    return out;
  }

  std::set<std::string> requiredMetrics{
      "cpu.render_submit_ms",
      "gpu.scopes_sum_ms",
  };
  for (const std::string &metric : current.benchmarkCase.requiredMetrics) {
    requiredMetrics.insert(metric);
  }
  for (const std::string &metric : baseline.benchmarkCase.requiredMetrics) {
    requiredMetrics.insert(metric);
  }

  for (const std::string &metric : requiredMetrics) {
    compareMetricStatistic(out, current, baseline, metric, "median", true);
    compareMetricStatistic(out, current, baseline, metric, "p95", true);
  }

  std::set<std::string> optionalMetrics;
  for (const auto &[metricId, stats] : baseline.stats) {
    (void)stats;
    if (metricId.rfind("gpu.scopes.", 0) == 0 &&
        requiredMetrics.find(metricId) == requiredMetrics.end()) {
      optionalMetrics.insert(metricId);
    }
  }
  for (const auto &[metricId, stats] : current.stats) {
    (void)stats;
    if (metricId.rfind("gpu.scopes.", 0) == 0 &&
        requiredMetrics.find(metricId) == requiredMetrics.end()) {
      optionalMetrics.insert(metricId);
    }
  }
  for (const std::string &metric : optionalMetrics) {
    compareMetricStatistic(out, current, baseline, metric, "median", false);
    compareMetricStatistic(out, current, baseline, metric, "p95", false);
  }

  if (options.force && !out.errors.empty()) {
    out.warnings.insert(out.warnings.end(), out.errors.begin(),
                        out.errors.end());
    out.errors.clear();
    out.valid = true;
  }
  return out;
}

Result<std::string, std::string>
writeBenchmarkComparisonJson(const BenchmarkComparisonReport &report) {
  yyjson_mut_doc *rawDoc = yyjson_mut_doc_new(nullptr);
  if (rawDoc == nullptr) {
    return Result<std::string, std::string>::makeError(
        "writeBenchmarkComparisonJson: failed to allocate document");
  }
  std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)> doc(
      rawDoc, &yyjson_mut_doc_free);
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion",
                          report.schemaVersion);
  yyjson_mut_obj_add_strcpy(doc.get(), root, "kind", report.kind.c_str());
  yyjson_mut_obj_add_bool(doc.get(), root, "valid", report.valid);
  yyjson_mut_obj_add_bool(doc.get(), root, "regression", report.regression);

  yyjson_mut_val *errors = yyjson_mut_arr(doc.get());
  for (const std::string &error : report.errors) {
    yyjson_mut_arr_add_strcpy(doc.get(), errors, error.c_str());
  }
  yyjson_mut_obj_add_val(doc.get(), root, "errors", errors);
  yyjson_mut_val *warnings = yyjson_mut_arr(doc.get());
  for (const std::string &warning : report.warnings) {
    yyjson_mut_arr_add_strcpy(doc.get(), warnings, warning.c_str());
  }
  yyjson_mut_obj_add_val(doc.get(), root, "warnings", warnings);

  yyjson_mut_val *metrics = yyjson_mut_arr(doc.get());
  for (const BenchmarkMetricComparison &metric : report.metrics) {
    yyjson_mut_val *object = yyjson_mut_obj(doc.get());
    yyjson_mut_obj_add_strcpy(doc.get(), object, "metricId",
                              metric.metricId.c_str());
    yyjson_mut_obj_add_strcpy(doc.get(), object, "statistic",
                              metric.statistic.c_str());
    yyjson_mut_obj_add_real(doc.get(), object, "baseline", metric.baseline);
    yyjson_mut_obj_add_real(doc.get(), object, "current", metric.current);
    yyjson_mut_obj_add_real(doc.get(), object, "delta", metric.delta);
    yyjson_mut_obj_add_real(doc.get(), object, "deltaPercent",
                            metric.deltaPercent);
    yyjson_mut_obj_add_strcpy(doc.get(), object, "status",
                              metric.status.c_str());
    yyjson_mut_obj_add_bool(doc.get(), object, "required", metric.required);
    yyjson_mut_arr_add_val(metrics, object);
  }
  yyjson_mut_obj_add_val(doc.get(), root, "metrics", metrics);

  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "writeBenchmarkComparisonJson: failed to write JSON");
  }
  std::string out(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(out));
}

Result<bool, std::string>
writeBenchmarkComparisonFile(const BenchmarkComparisonReport &report,
                             const std::filesystem::path &path) {
  auto json = writeBenchmarkComparisonJson(report);
  if (json.hasError()) {
    return Result<bool, std::string>::makeError(json.error());
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return Result<bool, std::string>::makeError(
        "writeBenchmarkComparisonFile: failed to open " + path.string());
  }
  file << json.value();
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri::tools::benchmark
