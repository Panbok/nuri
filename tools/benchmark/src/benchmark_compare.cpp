#include "nuri/tools/benchmark/benchmark_compare.h"

#include "nuri/tools/core/json_contract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>

#include <yyjson.h>

namespace nuri::tools::benchmark {
namespace {

using JsonField = nuri::tools::core::JsonFieldContract;
using JsonType = nuri::tools::core::JsonFieldType;

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

[[nodiscard]] bool stringMismatchWhenPresent(const std::string &lhs,
                                             const std::string &rhs) {
  return (!lhs.empty() || !rhs.empty()) && lhs != rhs;
}

[[nodiscard]] bool signatureMismatch(const std::string &lhs,
                                     const std::string &rhs) {
  return (!lhs.empty() || !rhs.empty()) && lhs != rhs;
}

[[nodiscard]] bool runConfigMismatch(const BenchmarkRunInfo &lhs,
                                     const BenchmarkRunInfo &rhs) {
  return lhs.samples != rhs.samples || lhs.warmupFrames != rhs.warmupFrames ||
         lhs.measurementFrames != rhs.measurementFrames ||
         lhs.cooldownFrames != rhs.cooldownFrames ||
         lhs.maxDrainFrames != rhs.maxDrainFrames ||
         lhs.drainTimeoutMs != rhs.drainTimeoutMs ||
         std::abs(lhs.fixedDeltaSeconds - rhs.fixedDeltaSeconds) > 1.0e-9;
}

[[nodiscard]] bool
renderGraphConfigMismatch(const BenchmarkRenderGraphConfig &lhs,
                          const BenchmarkRenderGraphConfig &rhs) {
  return lhs.workerCount != rhs.workerCount ||
         lhs.parallelCompile != rhs.parallelCompile ||
         lhs.parallelRecording != rhs.parallelRecording;
}

[[nodiscard]] bool
invalidGateThresholds(const BenchmarkThresholds &thresholds) {
  return !std::isfinite(thresholds.failPercent) ||
         !std::isfinite(thresholds.failAbsoluteMs) ||
         !std::isfinite(thresholds.warnPercent) ||
         !std::isfinite(thresholds.warnAbsoluteMs) ||
         thresholds.failPercent < 0.0 || thresholds.failAbsoluteMs < 0.0 ||
         thresholds.warnPercent < 0.0 || thresholds.warnAbsoluteMs < 0.0 ||
         thresholds.warnPercent > thresholds.failPercent ||
         thresholds.warnAbsoluteMs > thresholds.failAbsoluteMs;
}

[[nodiscard]] bool profilePolicyMismatch(const BenchmarkProfileInfo &lhs,
                                         const BenchmarkProfileInfo &rhs) {
  if (lhs.id.empty() && rhs.id.empty()) {
    return false;
  }
  return lhs.profileAuthoritative != rhs.profileAuthoritative ||
         lhs.minimumRepetitions != rhs.minimumRepetitions ||
         lhs.warmupStabilityPolicy != rhs.warmupStabilityPolicy ||
         lhs.warmupWindowFrames != rhs.warmupWindowFrames ||
         lhs.warmupMaxDriftPercent != rhs.warmupMaxDriftPercent ||
         lhs.requiredMetrics != rhs.requiredMetrics;
}

[[nodiscard]] bool
profileAuthoritySatisfied(const BenchmarkProfileInfo &profile) {
  return !profile.id.empty() && profile.profileAuthoritative &&
         profile.authoritative && profile.minimumRepetitions > 0u &&
         profile.completedRepetitions >= profile.minimumRepetitions &&
         profile.repetitionRequirementSatisfied &&
         profile.repetitionUnit == "isolated-process" &&
         profile.warmupStabilityStatus == "stable";
}

[[nodiscard]] std::vector<double>
sampleStatisticValues(const BenchmarkReport &report, std::string_view metricId,
                      std::string_view statistic) {
  std::vector<double> values;
  values.reserve(report.sampleStats.size());
  for (const BenchmarkSampleStats &sample : report.sampleStats) {
    const auto metric = sample.stats.find(std::string(metricId));
    if (metric != sample.stats.end()) {
      values.push_back(statValue(metric->second, statistic));
    }
  }
  return values;
}

void compareMetricStatistic(BenchmarkComparisonReport &out,
                            const BenchmarkReport &current,
                            const BenchmarkReport &baseline,
                            std::string_view metricId,
                            std::string_view statistic, bool required,
                            bool requireStatisticalAgreement) {
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
  const bool relativeDeltaAvailable = std::abs(baselineValue) > 1.0e-12;
  const double deltaPercent =
      relativeDeltaAvailable ? (delta / baselineValue) * 100.0 : 0.0;
  const BenchmarkThresholds &thresholds = baseline.benchmarkCase.thresholds;

  BenchmarkMetricComparison metric{};
  metric.metricId = std::string(metricId);
  metric.statistic = std::string(statistic);
  metric.baseline = baselineValue;
  metric.current = currentValue;
  metric.delta = delta;
  metric.deltaPercent = deltaPercent;
  metric.deltaPercentAvailable = relativeDeltaAvailable;
  metric.required = required;
  std::vector<double> baselineObservations =
      sampleStatisticValues(baseline, metricId, statistic);
  std::vector<double> currentObservations =
      sampleStatisticValues(current, metricId, statistic);
  if (!baselineObservations.empty() && !currentObservations.empty()) {
    auto repeatComparison = computeRepeatComparison(
        std::move(baselineObservations), std::move(currentObservations));
    if (!repeatComparison.hasError()) {
      metric.repeatComparison = repeatComparison.value();
      metric.repeatObservationUnit =
          current.repeatObservations.unit == baseline.repeatObservations.unit
              ? current.repeatObservations.unit
              : "mixed";
      metric.repeatObservationsIndependent =
          current.repeatObservations.independent &&
          baseline.repeatObservations.independent;
      if (required && metric.repeatComparison->lowConfidence) {
        out.warnings.push_back(
            "low-confidence repeat comparison for required metric '" +
            std::string(metricId) + "' statistic '" + std::string(statistic) +
            "'");
      }
    }
  }

  const bool practicalFailure =
      delta > thresholds.failAbsoluteMs &&
      (!relativeDeltaAvailable || deltaPercent > thresholds.failPercent);
  if (practicalFailure) {
    const bool statisticalAgreement =
        metric.repeatComparison.has_value() &&
        metric.repeatObservationsIndependent &&
        !metric.repeatComparison->lowConfidence &&
        metric.repeatComparison->confidenceLow > 0.0;
    if (required && (!requireStatisticalAgreement || statisticalAgreement)) {
      metric.status = "fail";
      out.regression = true;
    } else {
      metric.status = "warn";
      if (required && requireStatisticalAgreement) {
        out.warnings.push_back(
            "practical regression was not confirmed by independent "
            "repeat confidence for required metric '" +
            std::string(metricId) + "' statistic '" + std::string(statistic) +
            "'");
      }
    }
  } else if (delta > thresholds.warnAbsoluteMs &&
             (!relativeDeltaAvailable ||
              deltaPercent > thresholds.warnPercent)) {
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
      stringMismatchWhenPresent(current.profile.id, baseline.profile.id),
      "baseline profile mismatch");
  addCompatibilityError(
      out, options.force,
      profilePolicyMismatch(current.profile, baseline.profile),
      "baseline profile policy mismatch");
  addCompatibilityError(
      out, options.force,
      (!current.profile.id.empty() &&
       !current.profile.repetitionRequirementSatisfied) ||
          (!baseline.profile.id.empty() &&
           !baseline.profile.repetitionRequirementSatisfied),
      "baseline profile minimum independent repetitions were not satisfied");
  addCompatibilityError(
      out, options.force,
      (current.profile.profileAuthoritative &&
       !profileAuthoritySatisfied(current.profile)) ||
          (baseline.profile.profileAuthoritative &&
           !profileAuthoritySatisfied(baseline.profile)),
      "authoritative baseline profile requirements were not satisfied");
  addCompatibilityError(
      out, options.force,
      invalidGateThresholds(baseline.benchmarkCase.thresholds),
      "baseline gate thresholds are invalid");
  addCompatibilityError(out, options.force,
                        current.benchmarkCase.resolution !=
                            baseline.benchmarkCase.resolution,
                        "resolution mismatch");
  addCompatibilityError(out, options.force,
                        current.benchmarkCase.presentMode !=
                            baseline.benchmarkCase.presentMode,
                        "case present mode mismatch");
  addCompatibilityError(out, options.force,
                        runConfigMismatch(current.run, baseline.run),
                        "run frame-count/fixed-delta configuration mismatch");
  addCompatibilityError(
      out, options.force,
      renderGraphConfigMismatch(current.benchmarkCase.renderGraph,
                                baseline.benchmarkCase.renderGraph),
      "case render graph configuration mismatch");
  addCompatibilityError(
      out, options.force,
      signatureMismatch(current.benchmarkCase.settingsSignature,
                        baseline.benchmarkCase.settingsSignature),
      "case settings signature mismatch");
  addCompatibilityError(
      out, options.force,
      signatureMismatch(current.benchmarkCase.configSignature,
                        baseline.benchmarkCase.configSignature),
      "case configuration signature mismatch");
  addCompatibilityError(out, options.force,
                        current.environment.gpuBackend !=
                            baseline.environment.gpuBackend,
                        "backend mismatch");
  const auto adapterKnown = [](const BenchmarkEnvironment &environment) {
    return !environment.gpuDeviceName.empty() &&
           environment.gpuDeviceName != "unknown" &&
           environment.gpuVendorId != 0u && environment.gpuDeviceId != 0u &&
           !environment.gpuDriverVersion.empty() &&
           environment.gpuDriverVersion != "unknown";
  };
  addCompatibilityError(
      out, options.force,
      current.environment.gpuDeviceName != baseline.environment.gpuDeviceName ||
          current.environment.gpuVendorId != baseline.environment.gpuVendorId ||
          current.environment.gpuDeviceId != baseline.environment.gpuDeviceId ||
          current.environment.gpuDriverVersion !=
              baseline.environment.gpuDriverVersion,
      "GPU adapter/driver mismatch");
  addCompatibilityError(
      out, options.force,
      (current.profile.profileAuthoritative &&
       !adapterKnown(current.environment)) ||
          (baseline.profile.profileAuthoritative &&
           !adapterKnown(baseline.environment)),
      "authoritative profile requires known GPU adapter and driver identity");
  addCompatibilityError(
      out, options.force,
      current.environment.osName != baseline.environment.osName ||
          current.environment.osVersion != baseline.environment.osVersion,
      "operating system mismatch");
  addCompatibilityError(out, options.force,
                        current.environment.cpuName !=
                                baseline.environment.cpuName ||
                            current.environment.cpuLogicalThreadCount !=
                                baseline.environment.cpuLogicalThreadCount,
                        "CPU identity mismatch");
  addCompatibilityError(out, options.force,
                        current.environment.dirty || baseline.environment.dirty,
                        "dirty source trees are not valid comparison evidence");
  addCompatibilityError(
      out, options.force,
      stringMismatchWhenPresent(current.environment.resolvedPresentMode,
                                baseline.environment.resolvedPresentMode),
      "resolved present mode mismatch");
  addCompatibilityError(
      out, options.force,
      stringMismatchWhenPresent(current.environment.windowMode,
                                baseline.environment.windowMode) ||
          current.environment.windowVisible !=
              baseline.environment.windowVisible,
      "window mode/visibility mismatch");
  addCompatibilityError(
      out, options.force,
      current.environment.renderGraphWorkerCount !=
              baseline.environment.renderGraphWorkerCount ||
          current.environment.renderGraphParallelCompile !=
              baseline.environment.renderGraphParallelCompile ||
          current.environment.renderGraphParallelRecording !=
              baseline.environment.renderGraphParallelRecording,
      "resolved render graph environment mismatch");
  addCompatibilityError(out, options.force,
                        current.environment.buildType !=
                            baseline.environment.buildType,
                        "build type mismatch");
  addCompatibilityError(out, options.force,
                        current.environment.cmakeToolProfile !=
                            baseline.environment.cmakeToolProfile,
                        "tool profile mismatch");
  addCompatibilityError(out, options.force,
                        current.environment.buildShared !=
                                baseline.environment.buildShared ||
                            current.environment.loggingEnabled !=
                                baseline.environment.loggingEnabled ||
                            current.environment.assertsEnabled !=
                                baseline.environment.assertsEnabled,
                        "build feature flags mismatch");
  addCompatibilityError(
      out, options.force,
      current.environment.tracyEnabled != baseline.environment.tracyEnabled ||
          current.environment.tracyGpuEnabled !=
              baseline.environment.tracyGpuEnabled ||
          current.environment.tracyGpuDrawZonesEnabled !=
              baseline.environment.tracyGpuDrawZonesEnabled ||
          current.environment.devChecks != baseline.environment.devChecks,
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
  for (const std::string &metric : current.profile.requiredMetrics) {
    requiredMetrics.insert(metric);
  }
  for (const std::string &metric : baseline.profile.requiredMetrics) {
    requiredMetrics.insert(metric);
  }

  const bool requireStatisticalAgreement =
      !options.force && profileAuthoritySatisfied(current.profile) &&
      profileAuthoritySatisfied(baseline.profile);

  for (const std::string &metric : requiredMetrics) {
    compareMetricStatistic(out, current, baseline, metric, "median", true,
                           requireStatisticalAgreement);
    compareMetricStatistic(out, current, baseline, metric, "p95", true,
                           requireStatisticalAgreement);
  }

  std::set<std::string> optionalMetrics;
  for (const auto &[metricId, stats] : baseline.stats) {
    (void)stats;
    if ((metricId.rfind("gpu.scopes.", 0) == 0 ||
         metricId.rfind("rendergraph.pass.", 0) == 0) &&
        requiredMetrics.find(metricId) == requiredMetrics.end()) {
      optionalMetrics.insert(metricId);
    }
  }
  for (const auto &[metricId, stats] : current.stats) {
    (void)stats;
    if ((metricId.rfind("gpu.scopes.", 0) == 0 ||
         metricId.rfind("rendergraph.pass.", 0) == 0) &&
        requiredMetrics.find(metricId) == requiredMetrics.end()) {
      optionalMetrics.insert(metricId);
    }
  }
  for (const std::string &metric : optionalMetrics) {
    compareMetricStatistic(out, current, baseline, metric, "median", false,
                           requireStatisticalAgreement);
    compareMetricStatistic(out, current, baseline, metric, "p95", false,
                           requireStatisticalAgreement);
  }

  const bool hasDependentRepeatObservations =
      std::any_of(out.metrics.begin(), out.metrics.end(),
                  [](const BenchmarkMetricComparison &metric) {
                    return metric.repeatComparison.has_value() &&
                           !metric.repeatObservationsIndependent;
                  });
  if (hasDependentRepeatObservations) {
    out.warnings.push_back(
        "repeat-level statistics use in-process sample windows; they are "
        "investigative observations, not independent repetitions");
  }

  if (options.force && !out.errors.empty()) {
    out.warnings.insert(out.warnings.end(), out.errors.begin(),
                        out.errors.end());
    out.errors.clear();
    out.valid = true;
  }
  out.authoritative = out.valid && !options.force &&
                      profileAuthoritySatisfied(current.profile) &&
                      profileAuthoritySatisfied(baseline.profile);
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
  yyjson_mut_obj_add_bool(doc.get(), root, "authoritative",
                          report.authoritative);
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
    if (metric.deltaPercentAvailable) {
      yyjson_mut_obj_add_real(doc.get(), object, "deltaPercent",
                              metric.deltaPercent);
    } else {
      yyjson_mut_obj_add_null(doc.get(), object, "deltaPercent");
    }
    yyjson_mut_obj_add_bool(doc.get(), object, "deltaPercentAvailable",
                            metric.deltaPercentAvailable);
    yyjson_mut_obj_add_strcpy(doc.get(), object, "status",
                              metric.status.c_str());
    yyjson_mut_obj_add_bool(doc.get(), object, "required", metric.required);
    if (metric.repeatComparison.has_value()) {
      const RepeatComparisonStats &repeat = *metric.repeatComparison;
      yyjson_mut_val *repeatObject = yyjson_mut_obj(doc.get());
      yyjson_mut_obj_add_strcpy(doc.get(), repeatObject, "observationUnit",
                                metric.repeatObservationUnit.c_str());
      yyjson_mut_obj_add_bool(doc.get(), repeatObject, "independent",
                              metric.repeatObservationsIndependent);
      yyjson_mut_obj_add_uint(doc.get(), repeatObject, "baselineObservations",
                              repeat.baselineRepetitions);
      yyjson_mut_obj_add_uint(doc.get(), repeatObject, "currentObservations",
                              repeat.currentRepetitions);
      yyjson_mut_obj_add_real(doc.get(), repeatObject, "baselineMedian",
                              repeat.baselineMedian);
      yyjson_mut_obj_add_real(doc.get(), repeatObject, "currentMedian",
                              repeat.currentMedian);
      yyjson_mut_obj_add_real(doc.get(), repeatObject, "absoluteDelta",
                              repeat.absoluteDelta);
      if (repeat.percentDeltaDefined) {
        yyjson_mut_obj_add_real(doc.get(), repeatObject, "percentDelta",
                                repeat.percentDelta);
      } else {
        yyjson_mut_obj_add_null(doc.get(), repeatObject, "percentDelta");
      }
      yyjson_mut_obj_add_bool(doc.get(), repeatObject, "percentDeltaDefined",
                              repeat.percentDeltaDefined);
      yyjson_mut_obj_add_real(doc.get(), repeatObject, "robustEffect",
                              repeat.robustEffect);
      yyjson_mut_obj_add_real(doc.get(), repeatObject, "confidenceLow",
                              repeat.confidenceLow);
      yyjson_mut_obj_add_real(doc.get(), repeatObject, "confidenceHigh",
                              repeat.confidenceHigh);
      yyjson_mut_obj_add_real(doc.get(), repeatObject, "noiseScore",
                              repeat.noiseScore);
      yyjson_mut_obj_add_bool(doc.get(), repeatObject, "lowConfidence",
                              repeat.lowConfidence);
      yyjson_mut_obj_add_val(doc.get(), object, "repeatComparison",
                             repeatObject);
    } else {
      yyjson_mut_obj_add_null(doc.get(), object, "repeatComparison");
    }
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

Result<BenchmarkComparisonReport, std::string>
readBenchmarkComparisonFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<BenchmarkComparisonReport, std::string>::makeError(
        "readBenchmarkComparisonFile: failed to open " + path.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
      yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
      &yyjson_doc_free);
  yyjson_val *root = doc ? yyjson_doc_get_root(doc.get()) : nullptr;
  if (!yyjson_is_obj(root)) {
    return Result<BenchmarkComparisonReport, std::string>::makeError(
        "readBenchmarkComparisonFile: JSON root must be an object");
  }
  auto valid = nuri::tools::core::rejectDuplicateJsonFieldsRecursively(root);
  if (valid.hasError()) {
    return Result<BenchmarkComparisonReport, std::string>::makeError(
        "readBenchmarkComparisonFile: " + valid.error());
  }
  static constexpr std::array rootFields{
      JsonField{"schemaVersion", JsonType::Unsigned},
      JsonField{"kind", JsonType::String},
      JsonField{"valid", JsonType::Boolean},
      JsonField{"authoritative", JsonType::Boolean},
      JsonField{"regression", JsonType::Boolean},
      JsonField{"errors", JsonType::Array},
      JsonField{"warnings", JsonType::Array},
      JsonField{"metrics", JsonType::Array},
  };
  valid = nuri::tools::core::validateJsonObject(root, rootFields, "$");
  if (valid.hasError()) {
    return Result<BenchmarkComparisonReport, std::string>::makeError(
        "readBenchmarkComparisonFile: " + valid.error());
  }
  yyjson_val *schema = yyjson_obj_get(root, "schemaVersion");
  yyjson_val *kind = yyjson_obj_get(root, "kind");
  if (yyjson_get_uint(schema) != 1u ||
      std::string_view(yyjson_get_str(kind), yyjson_get_len(kind)) !=
          "nuri.benchmark.comparison") {
    return Result<BenchmarkComparisonReport, std::string>::makeError(
        "readBenchmarkComparisonFile: unsupported schema or kind");
  }
  const auto readString = [](yyjson_val *object, const char *name) {
    yyjson_val *value = yyjson_obj_get(object, name);
    return std::string(yyjson_get_str(value), yyjson_get_len(value));
  };
  const auto readStrings = [](yyjson_val *array, std::string_view path)
      -> Result<std::vector<std::string>, std::string> {
    std::vector<std::string> values;
    yyjson_arr_iter iterator{};
    yyjson_arr_iter_init(array, &iterator);
    yyjson_val *entry = nullptr;
    size_t index = 0u;
    while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
      if (!yyjson_is_str(entry)) {
        return Result<std::vector<std::string>, std::string>::makeError(
            std::string(path) + "[" + std::to_string(index) +
            "] must be a string");
      }
      values.emplace_back(yyjson_get_str(entry), yyjson_get_len(entry));
      ++index;
    }
    return Result<std::vector<std::string>, std::string>::makeResult(
        std::move(values));
  };
  auto errors = readStrings(yyjson_obj_get(root, "errors"), "$.errors");
  auto warnings = readStrings(yyjson_obj_get(root, "warnings"), "$.warnings");
  if (errors.hasError() || warnings.hasError()) {
    return Result<BenchmarkComparisonReport, std::string>::makeError(
        "readBenchmarkComparisonFile: " +
        (errors.hasError() ? errors.error() : warnings.error()));
  }
  BenchmarkComparisonReport report{};
  report.valid = yyjson_get_bool(yyjson_obj_get(root, "valid"));
  report.authoritative = yyjson_get_bool(yyjson_obj_get(root, "authoritative"));
  report.regression = yyjson_get_bool(yyjson_obj_get(root, "regression"));
  report.errors = std::move(errors.value());
  report.warnings = std::move(warnings.value());

  static constexpr std::array metricFields{
      JsonField{"metricId", JsonType::String},
      JsonField{"statistic", JsonType::String},
      JsonField{"baseline", JsonType::Number},
      JsonField{"current", JsonType::Number},
      JsonField{"delta", JsonType::Number},
      JsonField{"deltaPercent", JsonType::NullOrNumber},
      JsonField{"deltaPercentAvailable", JsonType::Boolean},
      JsonField{"status", JsonType::String},
      JsonField{"required", JsonType::Boolean},
      JsonField{"repeatComparison", JsonType::Any},
  };
  static constexpr std::array repeatFields{
      JsonField{"observationUnit", JsonType::String},
      JsonField{"independent", JsonType::Boolean},
      JsonField{"baselineObservations", JsonType::Unsigned},
      JsonField{"currentObservations", JsonType::Unsigned},
      JsonField{"baselineMedian", JsonType::Number},
      JsonField{"currentMedian", JsonType::Number},
      JsonField{"absoluteDelta", JsonType::Number},
      JsonField{"percentDelta", JsonType::NullOrNumber},
      JsonField{"percentDeltaDefined", JsonType::Boolean},
      JsonField{"robustEffect", JsonType::Number},
      JsonField{"confidenceLow", JsonType::Number},
      JsonField{"confidenceHigh", JsonType::Number},
      JsonField{"noiseScore", JsonType::Number},
      JsonField{"lowConfidence", JsonType::Boolean},
  };
  yyjson_arr_iter iterator{};
  yyjson_arr_iter_init(yyjson_obj_get(root, "metrics"), &iterator);
  yyjson_val *entry = nullptr;
  size_t index = 0u;
  while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
    const std::string metricPath = "$.metrics[" + std::to_string(index++) + "]";
    valid =
        nuri::tools::core::validateJsonObject(entry, metricFields, metricPath);
    if (valid.hasError()) {
      return Result<BenchmarkComparisonReport, std::string>::makeError(
          "readBenchmarkComparisonFile: " + valid.error());
    }
    BenchmarkMetricComparison metric{};
    metric.metricId = readString(entry, "metricId");
    metric.statistic = readString(entry, "statistic");
    metric.baseline = yyjson_get_num(yyjson_obj_get(entry, "baseline"));
    metric.current = yyjson_get_num(yyjson_obj_get(entry, "current"));
    metric.delta = yyjson_get_num(yyjson_obj_get(entry, "delta"));
    metric.deltaPercentAvailable =
        yyjson_get_bool(yyjson_obj_get(entry, "deltaPercentAvailable"));
    yyjson_val *deltaPercent = yyjson_obj_get(entry, "deltaPercent");
    if (yyjson_is_num(deltaPercent)) {
      metric.deltaPercent = yyjson_get_num(deltaPercent);
    }
    metric.status = readString(entry, "status");
    metric.required = yyjson_get_bool(yyjson_obj_get(entry, "required"));
    yyjson_val *repeat = yyjson_obj_get(entry, "repeatComparison");
    if (!yyjson_is_null(repeat)) {
      valid = nuri::tools::core::validateJsonObject(
          repeat, repeatFields, metricPath + ".repeatComparison");
      if (valid.hasError()) {
        return Result<BenchmarkComparisonReport, std::string>::makeError(
            "readBenchmarkComparisonFile: " + valid.error());
      }
      RepeatComparisonStats stats{};
      stats.baselineRepetitions = static_cast<uint32_t>(
          yyjson_get_uint(yyjson_obj_get(repeat, "baselineObservations")));
      stats.currentRepetitions = static_cast<uint32_t>(
          yyjson_get_uint(yyjson_obj_get(repeat, "currentObservations")));
      stats.baselineMedian =
          yyjson_get_num(yyjson_obj_get(repeat, "baselineMedian"));
      stats.currentMedian =
          yyjson_get_num(yyjson_obj_get(repeat, "currentMedian"));
      stats.absoluteDelta =
          yyjson_get_num(yyjson_obj_get(repeat, "absoluteDelta"));
      stats.percentDeltaDefined =
          yyjson_get_bool(yyjson_obj_get(repeat, "percentDeltaDefined"));
      yyjson_val *percentDelta = yyjson_obj_get(repeat, "percentDelta");
      if (yyjson_is_num(percentDelta)) {
        stats.percentDelta = yyjson_get_num(percentDelta);
      }
      stats.robustEffect =
          yyjson_get_num(yyjson_obj_get(repeat, "robustEffect"));
      stats.confidenceLow =
          yyjson_get_num(yyjson_obj_get(repeat, "confidenceLow"));
      stats.confidenceHigh =
          yyjson_get_num(yyjson_obj_get(repeat, "confidenceHigh"));
      stats.noiseScore = yyjson_get_num(yyjson_obj_get(repeat, "noiseScore"));
      stats.lowConfidence =
          yyjson_get_bool(yyjson_obj_get(repeat, "lowConfidence"));
      metric.repeatComparison = stats;
      metric.repeatObservationUnit = readString(repeat, "observationUnit");
      metric.repeatObservationsIndependent =
          yyjson_get_bool(yyjson_obj_get(repeat, "independent"));
    } else if (!yyjson_is_null(repeat)) {
      return Result<BenchmarkComparisonReport, std::string>::makeError(
          "readBenchmarkComparisonFile: repeatComparison must be object or "
          "null");
    }
    report.metrics.push_back(std::move(metric));
  }
  return Result<BenchmarkComparisonReport, std::string>::makeResult(
      std::move(report));
}

} // namespace nuri::tools::benchmark
