#include "nuri/tools/benchmark/benchmark_report.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>

#include <yyjson.h>

namespace nuri::tools::benchmark {
namespace {

using JsonMutDocPtr =
    std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;
using JsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;

void addString(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
               const std::string &value) {
  yyjson_mut_obj_add_strcpy(doc, object, key, value.c_str());
}

void addPath(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
             const std::filesystem::path &value) {
  addString(doc, object, key, value.generic_string());
}

yyjson_mut_val *makeStringArray(yyjson_mut_doc *doc,
                                const std::vector<std::string> &values) {
  yyjson_mut_val *array = yyjson_mut_arr(doc);
  for (const std::string &value : values) {
    yyjson_mut_arr_add_strcpy(doc, array, value.c_str());
  }
  return array;
}

yyjson_mut_val *
makePathArray(yyjson_mut_doc *doc,
              const std::vector<std::filesystem::path> &values) {
  yyjson_mut_val *array = yyjson_mut_arr(doc);
  for (const std::filesystem::path &value : values) {
    yyjson_mut_arr_add_strcpy(doc, array, value.generic_string().c_str());
  }
  return array;
}

yyjson_mut_val *makeStatsObject(yyjson_mut_doc *doc, const MetricStats &stats) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "count", stats.count);
  yyjson_mut_obj_add_real(doc, object, "min", stats.min);
  yyjson_mut_obj_add_real(doc, object, "median", stats.median);
  yyjson_mut_obj_add_real(doc, object, "p90", stats.p90);
  yyjson_mut_obj_add_real(doc, object, "p95", stats.p95);
  yyjson_mut_obj_add_real(doc, object, "max", stats.max);
  yyjson_mut_obj_add_real(doc, object, "mean", stats.mean);
  yyjson_mut_obj_add_real(doc, object, "stddev", stats.stddev);
  yyjson_mut_obj_add_real(doc, object, "mad", stats.mad);
  yyjson_mut_obj_add_real(doc, object, "iqr", stats.iqr);
  yyjson_mut_obj_add_real(doc, object, "coefficientOfVariation",
                          stats.coefficientOfVariation);
  return object;
}

yyjson_mut_val *
makeStatsMapObject(yyjson_mut_doc *doc,
                   const std::map<std::string, MetricStats> &stats) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  for (const auto &[metricId, metricStats] : stats) {
    yyjson_mut_obj_add_val(doc, object, metricId.c_str(),
                           makeStatsObject(doc, metricStats));
  }
  return object;
}

yyjson_mut_val *makeEnvironmentObject(yyjson_mut_doc *doc,
                                      const BenchmarkEnvironment &env) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addPath(doc, object, "repoRoot", env.repoRoot);
  addString(doc, object, "commitHash", env.commitHash);
  addString(doc, object, "branchName", env.branchName);
  yyjson_mut_obj_add_bool(doc, object, "dirty", env.dirty);
  addString(doc, object, "osName", env.osName);
  addString(doc, object, "osVersion", env.osVersion);
  addString(doc, object, "cpuName", env.cpuName);
  yyjson_mut_obj_add_uint(doc, object, "cpuLogicalThreadCount",
                          env.cpuLogicalThreadCount);
  addString(doc, object, "gpuBackend", env.gpuBackend);
  addString(doc, object, "gpuBackendSource", env.gpuBackendSource);
  addString(doc, object, "gpuDeviceName", env.gpuDeviceName);
  yyjson_mut_obj_add_uint(doc, object, "swapchainImageCount",
                          env.swapchainImageCount);
  addString(doc, object, "requestedPresentMode", env.requestedPresentMode);
  addString(doc, object, "resolvedPresentMode", env.resolvedPresentMode);
  addString(doc, object, "presentModeSource", env.presentModeSource);
  addString(doc, object, "windowMode", env.windowMode);
  yyjson_mut_obj_add_bool(doc, object, "windowVisible", env.windowVisible);
  yyjson_mut_obj_add_uint(doc, object, "renderGraphWorkerCount",
                          env.renderGraphWorkerCount);
  yyjson_mut_obj_add_bool(doc, object, "renderGraphParallelCompile",
                          env.renderGraphParallelCompile);
  yyjson_mut_obj_add_bool(doc, object, "renderGraphParallelRecording",
                          env.renderGraphParallelRecording);
  addString(doc, object, "renderGraphWorkerCountSource",
            env.renderGraphWorkerCountSource);
  addString(doc, object, "renderGraphParallelCompileSource",
            env.renderGraphParallelCompileSource);
  addString(doc, object, "renderGraphParallelRecordingSource",
            env.renderGraphParallelRecordingSource);
  addString(doc, object, "buildType", env.buildType);
  addString(doc, object, "cmakeToolProfile", env.cmakeToolProfile);
  addString(doc, object, "vcpkgManifestFeatures", env.vcpkgManifestFeatures);
  yyjson_mut_obj_add_bool(doc, object, "NURI_BUILD_SHARED", env.buildShared);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_LOGGING", env.loggingEnabled);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_ASSERTS", env.assertsEnabled);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_TRACY", env.tracyEnabled);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_TRACY_GPU",
                          env.tracyGpuEnabled);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_TRACY_GPU_DRAW_ZONES",
                          env.tracyGpuDrawZonesEnabled);
  yyjson_mut_obj_add_bool(doc, object, "tracyDiagnostic", env.tracyDiagnostic);
  yyjson_mut_obj_add_bool(doc, object, "devChecks", env.devChecks);
  return object;
}

yyjson_mut_val *makeCaseObject(yyjson_mut_doc *doc,
                               const BenchmarkCase &benchmarkCase) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "schemaVersion",
                          benchmarkCase.schemaVersion);
  addString(doc, object, "id", benchmarkCase.id);
  addString(doc, object, "suite", benchmarkCase.suite);
  addString(doc, object, "description", benchmarkCase.description);
  addString(doc, object, "backend", benchmarkCase.backend);
  yyjson_mut_val *resolution = yyjson_mut_arr(doc);
  yyjson_mut_arr_add_uint(doc, resolution, benchmarkCase.resolution[0]);
  yyjson_mut_arr_add_uint(doc, resolution, benchmarkCase.resolution[1]);
  yyjson_mut_obj_add_val(doc, object, "resolution", resolution);
  addString(doc, object, "presentMode", benchmarkCase.presentMode);
  yyjson_mut_obj_add_bool(doc, object, "authoritative",
                          benchmarkCase.authoritative);
  yyjson_mut_obj_add_val(doc, object, "requiredMetrics",
                         makeStringArray(doc, benchmarkCase.requiredMetrics));

  yyjson_mut_val *thresholds = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_real(doc, thresholds, "failPercent",
                          benchmarkCase.thresholds.failPercent);
  yyjson_mut_obj_add_real(doc, thresholds, "failAbsoluteMs",
                          benchmarkCase.thresholds.failAbsoluteMs);
  yyjson_mut_obj_add_real(doc, thresholds, "warnPercent",
                          benchmarkCase.thresholds.warnPercent);
  yyjson_mut_obj_add_real(doc, thresholds, "warnAbsoluteMs",
                          benchmarkCase.thresholds.warnAbsoluteMs);
  yyjson_mut_obj_add_val(doc, object, "thresholds", thresholds);

  yyjson_mut_val *scene = yyjson_mut_obj(doc);
  addString(doc, scene, "kind", benchmarkCase.scene.kind);
  addString(doc, scene, "pathBase", benchmarkCase.scene.pathBase);
  addPath(doc, scene, "path", benchmarkCase.scene.path);
  addString(doc, scene, "generator", benchmarkCase.scene.generator);
  yyjson_mut_obj_add_uint(doc, scene, "seed", benchmarkCase.scene.seed);
  addString(doc, scene, "contentHash", benchmarkCase.scene.contentHash);
  yyjson_mut_obj_add_val(doc, object, "scene", scene);
  return object;
}

yyjson_mut_val *makeFrameObject(yyjson_mut_doc *doc,
                                const BenchmarkFrameRecord &frame,
                                bool verboseMetrics) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "frameIndex", frame.frameIndex);
  yyjson_mut_obj_add_uint(doc, object, "sampleIndex", frame.sampleIndex);
  yyjson_mut_obj_add_bool(doc, object, "measured", frame.measured);
  yyjson_mut_val *measurements = yyjson_mut_obj(doc);
  for (const auto &[metricId, value] : frame.measurements) {
    yyjson_mut_obj_add_real(doc, measurements, metricId.c_str(), value);
  }
  yyjson_mut_obj_add_val(doc, object, "measurements", measurements);
  if (verboseMetrics) {
    yyjson_mut_val *metrics = yyjson_mut_obj(doc);
    yyjson_mut_val *opaque = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_uint(doc, opaque, "totalInstances",
                            frame.metrics.opaque.totalInstances);
    yyjson_mut_obj_add_uint(doc, opaque, "visibleInstances",
                            frame.metrics.opaque.visibleInstances);
    yyjson_mut_obj_add_val(doc, metrics, "opaque", opaque);
    yyjson_mut_val *shadow = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_uint(doc, shadow, "totalDraws",
                            frame.metrics.shadow.totalDraws);
    yyjson_mut_obj_add_val(doc, metrics, "shadow", shadow);
    yyjson_mut_obj_add_val(doc, object, "metrics", metrics);
  }
  return object;
}

yyjson_mut_val *makeSampleStatsObject(yyjson_mut_doc *doc,
                                      const BenchmarkSampleStats &sample) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "sampleIndex", sample.sampleIndex);
  yyjson_mut_obj_add_bool(doc, object, "warmupStable", sample.warmupStable);
  yyjson_mut_obj_add_uint(doc, object, "measuredFrameStart",
                          sample.measuredFrameStart);
  yyjson_mut_obj_add_uint(doc, object, "measuredFrameCount",
                          sample.measuredFrameCount);
  yyjson_mut_obj_add_val(doc, object, "stats",
                         makeStatsMapObject(doc, sample.stats));
  yyjson_mut_obj_add_val(doc, object, "warnings",
                         makeStringArray(doc, sample.warnings));
  return object;
}

[[nodiscard]] std::string readString(yyjson_val *object, const char *key,
                                     std::string defaultValue = {}) {
  yyjson_val *value = yyjson_obj_get(object, key);
  if (!yyjson_is_str(value)) {
    return defaultValue;
  }
  return std::string(yyjson_get_str(value), yyjson_get_len(value));
}

[[nodiscard]] bool readBool(yyjson_val *object, const char *key,
                            bool defaultValue = false) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_bool(value) ? yyjson_get_bool(value) : defaultValue;
}

[[nodiscard]] uint32_t readU32(yyjson_val *object, const char *key,
                               uint32_t defaultValue = 0u) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_uint(value) ? static_cast<uint32_t>(yyjson_get_uint(value))
                               : defaultValue;
}

[[nodiscard]] double readReal(yyjson_val *object, const char *key,
                              double defaultValue = 0.0) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_num(value) ? yyjson_get_num(value) : defaultValue;
}

[[nodiscard]] MetricStats readStats(yyjson_val *object) {
  MetricStats stats{};
  if (!yyjson_is_obj(object)) {
    return stats;
  }
  stats.count = static_cast<size_t>(readU32(object, "count"));
  stats.min = readReal(object, "min");
  stats.median = readReal(object, "median");
  stats.p90 = readReal(object, "p90");
  stats.p95 = readReal(object, "p95");
  stats.max = readReal(object, "max");
  stats.mean = readReal(object, "mean");
  stats.stddev = readReal(object, "stddev");
  stats.mad = readReal(object, "mad");
  stats.iqr = readReal(object, "iqr");
  stats.coefficientOfVariation = readReal(object, "coefficientOfVariation");
  return stats;
}

} // namespace

void computeBenchmarkReportStats(BenchmarkReport &report) {
  report.stats.clear();
  std::map<std::string, std::vector<double>> valuesByMetric;
  for (const BenchmarkFrameRecord &frame : report.frames) {
    if (!frame.measured) {
      continue;
    }
    for (const auto &[metricId, value] : frame.measurements) {
      valuesByMetric[metricId].push_back(value);
    }
  }
  for (auto &[metricId, values] : valuesByMetric) {
    auto stats = computeMetricStats(std::move(values));
    if (!stats.hasError()) {
      report.stats.emplace(metricId, stats.value());
    }
  }

  report.sampleStats.clear();
  const uint32_t samples = report.run.samples;
  for (uint32_t sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
    BenchmarkSampleStats sample{};
    sample.sampleIndex = sampleIndex;
    sample.measuredFrameCount = report.run.measurementFrames;
    sample.measuredFrameStart =
        static_cast<uint64_t>(sampleIndex) *
            (report.run.warmupFrames + report.run.measurementFrames +
             report.run.cooldownFrames) +
        report.run.warmupFrames;
    std::map<std::string, std::vector<double>> sampleValues;
    for (const BenchmarkFrameRecord &frame : report.frames) {
      if (!frame.measured || frame.sampleIndex != sampleIndex) {
        continue;
      }
      for (const auto &[metricId, value] : frame.measurements) {
        sampleValues[metricId].push_back(value);
      }
    }
    for (auto &[metricId, values] : sampleValues) {
      auto stats = computeMetricStats(std::move(values));
      if (!stats.hasError()) {
        sample.stats.emplace(metricId, stats.value());
      }
    }
    report.sampleStats.push_back(std::move(sample));
  }

  std::set<std::string> unavailable(report.unavailableMetrics.begin(),
                                    report.unavailableMetrics.end());
  for (const std::string &metricId : report.benchmarkCase.requiredMetrics) {
    if (report.stats.find(metricId) == report.stats.end()) {
      unavailable.insert(metricId);
    }
  }
  report.unavailableMetrics.assign(unavailable.begin(), unavailable.end());
}

Result<std::string, std::string>
writeBenchmarkReportJson(const BenchmarkReport &report, bool verboseFrames) {
  JsonMutDocPtr doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "writeBenchmarkReportJson: failed to allocate JSON document");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion",
                          report.schemaVersion);
  addString(doc.get(), root, "kind", report.kind);
  addString(doc.get(), root, "generatedAtUtc", report.generatedAtUtc);
  addString(doc.get(), root, "command", report.command);
  yyjson_mut_obj_add_val(doc.get(), root, "environment",
                         makeEnvironmentObject(doc.get(), report.environment));
  yyjson_mut_obj_add_val(doc.get(), root, "case",
                         makeCaseObject(doc.get(), report.benchmarkCase));

  yyjson_mut_val *run = yyjson_mut_obj(doc.get());
  yyjson_mut_obj_add_uint(doc.get(), run, "samples", report.run.samples);
  yyjson_mut_obj_add_uint(doc.get(), run, "warmupFrames",
                          report.run.warmupFrames);
  yyjson_mut_obj_add_uint(doc.get(), run, "measurementFrames",
                          report.run.measurementFrames);
  yyjson_mut_obj_add_uint(doc.get(), run, "cooldownFrames",
                          report.run.cooldownFrames);
  yyjson_mut_obj_add_uint(doc.get(), run, "maxDrainFrames",
                          report.run.maxDrainFrames);
  yyjson_mut_obj_add_uint(doc.get(), run, "drainTimeoutMs",
                          report.run.drainTimeoutMs);
  yyjson_mut_obj_add_bool(doc.get(), run, "validForComparison",
                          report.run.validForComparison);
  yyjson_mut_obj_add_real(doc.get(), run, "fixedDeltaSeconds",
                          report.run.fixedDeltaSeconds);
  yyjson_mut_obj_add_val(doc.get(), root, "run", run);

  yyjson_mut_val *artifacts = yyjson_mut_obj(doc.get());
  addPath(doc.get(), artifacts, "artifactDir", report.artifacts.artifactDir);
  yyjson_mut_obj_add_val(
      doc.get(), artifacts, "caseReports",
      makePathArray(doc.get(), report.artifacts.caseReports));
  yyjson_mut_obj_add_val(
      doc.get(), artifacts, "tracy",
      makePathArray(doc.get(), report.artifacts.tracyArtifacts));
  yyjson_mut_obj_add_val(doc.get(), root, "artifacts", artifacts);

  yyjson_mut_val *frames = yyjson_mut_arr(doc.get());
  for (const BenchmarkFrameRecord &frame : report.frames) {
    yyjson_mut_arr_add_val(frames,
                           makeFrameObject(doc.get(), frame, verboseFrames));
  }
  yyjson_mut_obj_add_val(doc.get(), root, "frames", frames);

  yyjson_mut_val *samples = yyjson_mut_arr(doc.get());
  for (const BenchmarkSampleStats &sample : report.sampleStats) {
    yyjson_mut_arr_add_val(samples, makeSampleStatsObject(doc.get(), sample));
  }
  yyjson_mut_obj_add_val(doc.get(), root, "sampleStats", samples);
  yyjson_mut_obj_add_val(doc.get(), root, "stats",
                         makeStatsMapObject(doc.get(), report.stats));

  yyjson_mut_obj_add_val(doc.get(), root, "renderGraph",
                         yyjson_mut_obj(doc.get()));
  yyjson_mut_obj_add_val(doc.get(), root, "resourceStats",
                         yyjson_mut_obj(doc.get()));

  yyjson_mut_val *drain = yyjson_mut_obj(doc.get());
  yyjson_mut_obj_add_bool(doc.get(), drain, "drainComplete",
                          report.timingDrain.drainComplete);
  yyjson_mut_obj_add_uint(doc.get(), drain, "drainFrames",
                          report.timingDrain.drainFrames);
  yyjson_mut_obj_add_uint(doc.get(), drain, "drainTimeoutMs",
                          report.timingDrain.drainTimeoutMs);
  yyjson_mut_obj_add_uint(doc.get(), drain, "missingGpuTimingFrames",
                          report.timingDrain.missingGpuTimingFrames);
  yyjson_mut_obj_add_uint(doc.get(), drain, "droppedGpuTimingReports",
                          report.timingDrain.droppedGpuTimingReports);
  yyjson_mut_obj_add_val(doc.get(), root, "timingDrain", drain);
  yyjson_mut_obj_add_val(doc.get(), root, "unavailableMetrics",
                         makeStringArray(doc.get(), report.unavailableMetrics));
  yyjson_mut_obj_add_val(doc.get(), root, "warnings",
                         makeStringArray(doc.get(), report.warnings));

  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "writeBenchmarkReportJson: failed to serialize JSON");
  }
  std::string out(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(out));
}

Result<bool, std::string>
writeBenchmarkReportFile(const BenchmarkReport &report,
                         const std::filesystem::path &path,
                         bool verboseFrames) {
  auto json = writeBenchmarkReportJson(report, verboseFrames);
  if (json.hasError()) {
    return Result<bool, std::string>::makeError(json.error());
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return Result<bool, std::string>::makeError(
        "writeBenchmarkReportFile: failed to open " + path.string());
  }
  file << json.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<BenchmarkReport, std::string>
readBenchmarkReportFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: failed to open " + path.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
                 &yyjson_doc_free);
  if (!doc) {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: JSON parse failed at byte " +
        std::to_string(error.pos) + ": " + error.msg);
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  if (!yyjson_is_obj(root)) {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: root must be an object");
  }
  BenchmarkReport report{};
  report.schemaVersion = readU32(root, "schemaVersion", 1u);
  report.kind = readString(root, "kind", report.kind);
  report.generatedAtUtc = readString(root, "generatedAtUtc");
  report.command = readString(root, "command");

  yyjson_val *caseObject = yyjson_obj_get(root, "case");
  if (yyjson_is_obj(caseObject)) {
    report.benchmarkCase.id = readString(caseObject, "id");
    report.benchmarkCase.suite = readString(caseObject, "suite");
    report.benchmarkCase.backend = readString(caseObject, "backend", "default");
    yyjson_val *resolution = yyjson_obj_get(caseObject, "resolution");
    if (yyjson_is_arr(resolution) && yyjson_arr_size(resolution) == 2u) {
      report.benchmarkCase.resolution[0] =
          static_cast<uint32_t>(yyjson_get_uint(yyjson_arr_get(resolution, 0)));
      report.benchmarkCase.resolution[1] =
          static_cast<uint32_t>(yyjson_get_uint(yyjson_arr_get(resolution, 1)));
    }
    yyjson_val *required = yyjson_obj_get(caseObject, "requiredMetrics");
    if (yyjson_is_arr(required)) {
      yyjson_arr_iter iter;
      yyjson_arr_iter_init(required, &iter);
      yyjson_val *entry = nullptr;
      while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
        if (yyjson_is_str(entry)) {
          report.benchmarkCase.requiredMetrics.emplace_back(
              yyjson_get_str(entry), yyjson_get_len(entry));
        }
      }
    }
    yyjson_val *thresholds = yyjson_obj_get(caseObject, "thresholds");
    if (yyjson_is_obj(thresholds)) {
      report.benchmarkCase.thresholds.failPercent =
          readReal(thresholds, "failPercent", 10.0);
      report.benchmarkCase.thresholds.failAbsoluteMs =
          readReal(thresholds, "failAbsoluteMs", 0.2);
      report.benchmarkCase.thresholds.warnPercent =
          readReal(thresholds, "warnPercent", 5.0);
      report.benchmarkCase.thresholds.warnAbsoluteMs =
          readReal(thresholds, "warnAbsoluteMs", 0.1);
    }
  }

  yyjson_val *environment = yyjson_obj_get(root, "environment");
  if (yyjson_is_obj(environment)) {
    report.environment.gpuBackend = readString(environment, "gpuBackend");
    report.environment.buildType = readString(environment, "buildType");
    report.environment.cmakeToolProfile =
        readString(environment, "cmakeToolProfile");
    report.environment.tracyEnabled = readBool(environment, "NURI_WITH_TRACY");
    report.environment.tracyGpuEnabled =
        readBool(environment, "NURI_WITH_TRACY_GPU");
    report.environment.devChecks = readBool(environment, "devChecks");
  }

  yyjson_val *run = yyjson_obj_get(root, "run");
  if (yyjson_is_obj(run)) {
    report.run.samples = readU32(run, "samples", 1u);
    report.run.validForComparison = readBool(run, "validForComparison", true);
  }

  yyjson_val *stats = yyjson_obj_get(root, "stats");
  if (yyjson_is_obj(stats)) {
    yyjson_obj_iter iter;
    yyjson_obj_iter_init(stats, &iter);
    yyjson_val *key = nullptr;
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
      yyjson_val *value = yyjson_obj_iter_get_val(key);
      report.stats.emplace(
          std::string(yyjson_get_str(key), yyjson_get_len(key)),
          readStats(value));
    }
  }
  return Result<BenchmarkReport, std::string>::makeResult(std::move(report));
}

} // namespace nuri::tools::benchmark
