#include "nuri/tools/autotest/autotest_report.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string_view>

#include <yyjson.h>

namespace nuri::tools::autotest {
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

yyjson_mut_val *makeEnvironmentObject(yyjson_mut_doc *doc,
                                      const AutotestEnvironment &env) {
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
  addString(doc, object, "requestedWindowMode", env.requestedWindowMode);
  addString(doc, object, "resolvedWindowMode", env.resolvedWindowMode);
  yyjson_mut_obj_add_bool(doc, object, "windowVisible", env.windowVisible);
  yyjson_mut_obj_add_uint(doc, object, "renderGraphWorkerCount",
                          env.renderGraphWorkerCount);
  yyjson_mut_obj_add_bool(doc, object, "renderGraphParallelCompile",
                          env.renderGraphParallelCompile);
  yyjson_mut_obj_add_bool(doc, object, "renderGraphParallelRecording",
                          env.renderGraphParallelRecording);
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
  yyjson_mut_obj_add_bool(doc, object, "devChecks", env.devChecks);
  return object;
}

yyjson_mut_val *makeCaseObject(yyjson_mut_doc *doc,
                               const AutotestCase &testCase) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "schemaVersion", testCase.schemaVersion);
  addString(doc, object, "id", testCase.id);
  addString(doc, object, "suite", testCase.suite);
  addString(doc, object, "description", testCase.description);
  addString(doc, object, "backend", testCase.backend);
  yyjson_mut_val *resolution = yyjson_mut_arr(doc);
  yyjson_mut_arr_add_uint(doc, resolution, testCase.resolution[0]);
  yyjson_mut_arr_add_uint(doc, resolution, testCase.resolution[1]);
  yyjson_mut_obj_add_val(doc, object, "resolution", resolution);
  yyjson_mut_obj_add_uint(doc, object, "warmupFrames", testCase.warmupFrames);
  yyjson_mut_obj_add_uint(doc, object, "endFrame", testCase.endFrame);
  yyjson_mut_obj_add_real(doc, object, "fixedDeltaSeconds",
                          testCase.fixedDeltaSeconds);
  addString(doc, object, "presentMode", testCase.presentMode);
  addString(doc, object, "windowMode", testCase.windowMode);
  yyjson_mut_obj_add_bool(doc, object, "authoritative", testCase.authoritative);
  return object;
}

yyjson_mut_val *makeRunObject(yyjson_mut_doc *doc,
                              const AutotestRunMetadata &run) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_real(doc, object, "fixedDeltaSeconds",
                          run.fixedDeltaSeconds);
  yyjson_mut_obj_add_uint(doc, object, "warmupFrames", run.warmupFrames);
  yyjson_mut_obj_add_uint(doc, object, "endFrame", run.endFrame);
  yyjson_mut_obj_add_uint(doc, object, "renderedFrames", run.renderedFrames);
  yyjson_mut_obj_add_uint(doc, object, "readoutDrainFrames",
                          run.readoutDrainFrames);
  yyjson_mut_obj_add_uint(doc, object, "readoutDrainTimeoutMs",
                          run.readoutDrainTimeoutMs);
  addString(doc, object, "captureSynchronization", run.captureSynchronization);
  yyjson_mut_obj_add_bool(doc, object, "validForComparison",
                          run.validForComparison);
  return object;
}

yyjson_mut_val *
makeMeasurementsObject(yyjson_mut_doc *doc,
                       const std::map<std::string, double> &measurements) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  for (const auto &[key, value] : measurements) {
    yyjson_mut_obj_add_real(doc, object, key.c_str(), value);
  }
  return object;
}

yyjson_mut_val *makeSnapshotCaptureObject(
    yyjson_mut_doc *doc,
    const nuri::tools::snapshot::SnapshotCaptureReport &capture) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "target", capture.target);
  addString(doc, object, "artifactStem", capture.artifactStem);
  addString(doc, object, "profile", capture.profile);
  yyjson_mut_obj_add_bool(doc, object, "required", capture.required);
  yyjson_mut_obj_add_bool(doc, object, "available", capture.available);
  yyjson_mut_obj_add_uint(doc, object, "capturePointVersion",
                          capture.capturePointVersion);
  yyjson_mut_obj_add_uint(doc, object, "captureFrameIndex",
                          capture.captureFrameIndex);
  addString(doc, object, "lifetime", capture.lifetime);
  addString(doc, object, "format", capture.format);
  addString(doc, object, "colorSpace", capture.colorSpace);
  yyjson_mut_obj_add_uint(doc, object, "width", capture.width);
  yyjson_mut_obj_add_uint(doc, object, "height", capture.height);
  addString(doc, object, "actualHash", capture.actualHash);
  addString(doc, object, "expectedHash", capture.expectedHash);
  addPath(doc, object, "actual", capture.actual);
  addPath(doc, object, "actualMetadata", capture.actualMetadata);
  addPath(doc, object, "preview", capture.preview);
  addPath(doc, object, "expected", capture.expected);
  addPath(doc, object, "diff", capture.diff);
  yyjson_mut_val *metrics = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_real(doc, metrics, "meanAbsError",
                          capture.metrics.meanAbsError);
  yyjson_mut_obj_add_real(doc, metrics, "rmse", capture.metrics.rmse);
  yyjson_mut_obj_add_real(doc, metrics, "maxAbsError",
                          capture.metrics.maxAbsError);
  yyjson_mut_obj_add_real(doc, metrics, "p99AbsError",
                          capture.metrics.p99AbsError);
  yyjson_mut_obj_add_uint(doc, metrics, "failingValues",
                          capture.metrics.failingValues);
  yyjson_mut_obj_add_uint(doc, metrics, "comparedValues",
                          capture.metrics.comparedValues);
  yyjson_mut_obj_add_val(doc, object, "metrics", metrics);
  yyjson_mut_obj_add_val(doc, object, "failedThresholds",
                         makeStringArray(doc, capture.failedThresholds));
  addString(doc, object, "producerPassLabel", capture.producerPassLabel);
  addString(doc, object, "readbackError", capture.readbackError);
  addString(doc, object, "status", capture.status);
  addString(doc, object, "statusReason", capture.statusReason);
  return object;
}

yyjson_mut_val *makeCaptureObject(yyjson_mut_doc *doc,
                                  const AutotestCaptureReport &capture) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "checkpointId", capture.checkpointId);
  yyjson_mut_obj_add_uint(doc, object, "checkpointFrame",
                          capture.checkpointFrame);
  addString(doc, object, "target", capture.target);
  addString(doc, object, "profile", capture.profile);
  yyjson_mut_obj_add_bool(doc, object, "required", capture.required);
  yyjson_mut_obj_add_bool(doc, object, "compare", capture.compare);
  yyjson_mut_obj_add_val(doc, object, "snapshot",
                         makeSnapshotCaptureObject(doc, capture.snapshot));
  return object;
}

yyjson_mut_val *makeReadoutObject(yyjson_mut_doc *doc,
                                  const AutotestReadoutReport &readout);

yyjson_mut_val *makeAssertionObject(yyjson_mut_doc *doc,
                                    const AutotestAssertionResult &assertion) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "id", assertion.id);
  addString(doc, object, "metric", assertion.metric);
  addString(doc, object, "statistic", assertion.statistic);
  addString(doc, object, "status",
            autotestAssertionStatusName(assertion.status));
  addString(doc, object, "statusReason", assertion.statusReason);
  yyjson_mut_obj_add_bool(doc, object, "hasActual", assertion.hasActual);
  yyjson_mut_obj_add_real(doc, object, "actual", assertion.actual);
  yyjson_mut_obj_add_uint(doc, object, "sampleCount", assertion.sampleCount);
  return object;
}

yyjson_mut_val *makeReadoutObject(yyjson_mut_doc *doc,
                                  const AutotestReadoutReport &readout) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "checkpointId", readout.checkpointId);
  addString(doc, object, "id", readout.id);
  addString(doc, object, "type", readout.type);
  yyjson_mut_obj_add_uint(doc, object, "requestId", readout.requestId);
  yyjson_mut_obj_add_uint(doc, object, "requestFrame", readout.requestFrame);
  yyjson_mut_obj_add_uint(doc, object, "resultFrame", readout.resultFrame);
  yyjson_mut_obj_add_bool(doc, object, "required", readout.required);
  addString(doc, object, "status", readout.status);
  addString(doc, object, "statusReason", readout.statusReason);
  yyjson_mut_obj_add_val(doc, object, "values",
                         makeMeasurementsObject(doc, readout.values));
  yyjson_mut_val *assertions = yyjson_mut_arr(doc);
  for (const AutotestAssertionResult &assertion : readout.assertions) {
    yyjson_mut_arr_add_val(assertions, makeAssertionObject(doc, assertion));
  }
  yyjson_mut_obj_add_val(doc, object, "assertions", assertions);
  return object;
}

[[nodiscard]] std::string htmlEscape(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (char ch : text) {
    switch (ch) {
    case '&':
      out += "&amp;";
      break;
    case '<':
      out += "&lt;";
      break;
    case '>':
      out += "&gt;";
      break;
    case '"':
      out += "&quot;";
      break;
    default:
      out += ch;
      break;
    }
  }
  return out;
}

[[nodiscard]] std::string readString(yyjson_val *object, const char *key,
                                     std::string defaultValue = {}) {
  yyjson_val *value = yyjson_obj_get(object, key);
  if (!yyjson_is_str(value)) {
    return defaultValue;
  }
  return std::string(yyjson_get_str(value), yyjson_get_len(value));
}

[[nodiscard]] uint32_t readU32(yyjson_val *object, const char *key,
                               uint32_t defaultValue = 0u) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_uint(value) ? static_cast<uint32_t>(yyjson_get_uint(value))
                               : defaultValue;
}

[[nodiscard]] uint64_t readU64(yyjson_val *object, const char *key,
                               uint64_t defaultValue = 0u) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_uint(value) ? static_cast<uint64_t>(yyjson_get_uint(value))
                               : defaultValue;
}

[[nodiscard]] double readDouble(yyjson_val *object, const char *key,
                                double defaultValue = 0.0) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_num(value) ? yyjson_get_num(value) : defaultValue;
}

[[nodiscard]] bool readBool(yyjson_val *object, const char *key,
                            bool defaultValue = false) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_bool(value) ? yyjson_get_bool(value) : defaultValue;
}

[[nodiscard]] AutotestAssertionStatus parseStatus(std::string_view value) {
  if (value == "warn") {
    return AutotestAssertionStatus::Warn;
  }
  if (value == "fail") {
    return AutotestAssertionStatus::Fail;
  }
  if (value == "unavailable") {
    return AutotestAssertionStatus::Unavailable;
  }
  if (value == "invalid") {
    return AutotestAssertionStatus::Invalid;
  }
  return AutotestAssertionStatus::Pass;
}

[[nodiscard]] AutotestAssertionResult
readAssertionResult(yyjson_val *assertionValue) {
  AutotestAssertionResult assertion{};
  assertion.id = readString(assertionValue, "id");
  assertion.metric = readString(assertionValue, "metric");
  assertion.statistic = readString(assertionValue, "statistic");
  assertion.status = parseStatus(readString(assertionValue, "status", "pass"));
  assertion.statusReason = readString(assertionValue, "statusReason");
  assertion.hasActual = readBool(assertionValue, "hasActual");
  assertion.actual = readDouble(assertionValue, "actual");
  assertion.sampleCount =
      readU32(assertionValue, "sampleCount", assertion.hasActual ? 1u : 0u);
  return assertion;
}

} // namespace

Result<std::string, std::string>
writeAutotestReportJson(const AutotestReport &report) {
  JsonMutDocPtr doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "writeAutotestReportJson: failed to allocate JSON document");
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
                         makeCaseObject(doc.get(), report.testCase));
  yyjson_mut_obj_add_val(doc.get(), root, "run",
                         makeRunObject(doc.get(), report.run));

  yyjson_mut_val *artifacts = yyjson_mut_obj(doc.get());
  addPath(doc.get(), artifacts, "artifactDir", report.artifacts.artifactDir);
  addPath(doc.get(), artifacts, "rootHtml", report.artifacts.rootHtml);
  addPath(doc.get(), artifacts, "caseDir", report.artifacts.caseDir);
  addPath(doc.get(), artifacts, "caseHtml", report.artifacts.caseHtml);
  yyjson_mut_obj_add_val(doc.get(), root, "artifacts", artifacts);

  yyjson_mut_val *checkpoints = yyjson_mut_arr(doc.get());
  for (const AutotestCheckpointReport &checkpoint : report.checkpoints) {
    yyjson_mut_val *object = yyjson_mut_obj(doc.get());
    addString(doc.get(), object, "id", checkpoint.id);
    yyjson_mut_obj_add_uint(doc.get(), object, "frame", checkpoint.frame);
    yyjson_mut_obj_add_val(
        doc.get(), object, "measurements",
        makeMeasurementsObject(doc.get(), checkpoint.measurements));
    yyjson_mut_val *captures = yyjson_mut_arr(doc.get());
    for (const AutotestCaptureReport &capture : checkpoint.captures) {
      yyjson_mut_arr_add_val(captures, makeCaptureObject(doc.get(), capture));
    }
    yyjson_mut_obj_add_val(doc.get(), object, "captures", captures);
    yyjson_mut_val *readouts = yyjson_mut_arr(doc.get());
    for (const AutotestReadoutReport &readout : checkpoint.readouts) {
      yyjson_mut_arr_add_val(readouts, makeReadoutObject(doc.get(), readout));
    }
    yyjson_mut_obj_add_val(doc.get(), object, "readouts", readouts);
    yyjson_mut_val *assertions = yyjson_mut_arr(doc.get());
    for (const AutotestAssertionResult &assertion : checkpoint.assertions) {
      yyjson_mut_arr_add_val(assertions,
                             makeAssertionObject(doc.get(), assertion));
    }
    yyjson_mut_obj_add_val(doc.get(), object, "assertions", assertions);
    yyjson_mut_obj_add_val(doc.get(), object, "warnings",
                           makeStringArray(doc.get(), checkpoint.warnings));
    yyjson_mut_obj_add_val(doc.get(), object, "errors",
                           makeStringArray(doc.get(), checkpoint.errors));
    yyjson_mut_arr_add_val(checkpoints, object);
  }
  yyjson_mut_obj_add_val(doc.get(), root, "checkpoints", checkpoints);
  yyjson_mut_val *metricWindows = yyjson_mut_arr(doc.get());
  for (const AutotestMetricWindowReport &window : report.metricWindows) {
    yyjson_mut_val *object = yyjson_mut_obj(doc.get());
    addString(doc.get(), object, "id", window.id);
    yyjson_mut_obj_add_uint(doc.get(), object, "startFrame", window.startFrame);
    yyjson_mut_obj_add_uint(doc.get(), object, "endFrame", window.endFrame);
    yyjson_mut_val *assertions = yyjson_mut_arr(doc.get());
    for (const AutotestAssertionResult &assertion : window.assertions) {
      yyjson_mut_arr_add_val(assertions,
                             makeAssertionObject(doc.get(), assertion));
    }
    yyjson_mut_obj_add_val(doc.get(), object, "assertions", assertions);
    yyjson_mut_obj_add_val(doc.get(), object, "warnings",
                           makeStringArray(doc.get(), window.warnings));
    yyjson_mut_obj_add_val(doc.get(), object, "errors",
                           makeStringArray(doc.get(), window.errors));
    yyjson_mut_arr_add_val(metricWindows, object);
  }
  yyjson_mut_obj_add_val(doc.get(), root, "metricWindows", metricWindows);
  yyjson_mut_val *frames = yyjson_mut_arr(doc.get());
  for (const AutotestFrameReport &frame : report.frames) {
    yyjson_mut_val *object = yyjson_mut_obj(doc.get());
    yyjson_mut_obj_add_uint(doc.get(), object, "frameIndex", frame.frameIndex);
    yyjson_mut_obj_add_val(
        doc.get(), object, "measurements",
        makeMeasurementsObject(doc.get(), frame.measurements));
    yyjson_mut_obj_add_val(
        doc.get(), object, "unavailableMetrics",
        makeStringArray(doc.get(), frame.unavailableMetrics));
    yyjson_mut_arr_add_val(frames, object);
  }
  yyjson_mut_obj_add_val(doc.get(), root, "frames", frames);
  yyjson_mut_obj_add_val(doc.get(), root, "unavailableMetrics",
                         makeStringArray(doc.get(), report.unavailableMetrics));
  yyjson_mut_obj_add_val(doc.get(), root, "warnings",
                         makeStringArray(doc.get(), report.warnings));
  yyjson_mut_obj_add_val(doc.get(), root, "errors",
                         makeStringArray(doc.get(), report.errors));
  addString(doc.get(), root, "reproduceCommand", report.reproduceCommand);

  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "writeAutotestReportJson: failed to serialize JSON");
  }
  std::string out(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(out));
}

Result<bool, std::string>
writeAutotestReportFile(const AutotestReport &report,
                        const std::filesystem::path &path) {
  auto json = writeAutotestReportJson(report);
  if (json.hasError()) {
    return Result<bool, std::string>::makeError(json.error());
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return Result<bool, std::string>::makeError(
        "writeAutotestReportFile: failed to open " + path.string());
  }
  file << json.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<AutotestReport, std::string>
readAutotestReportFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: failed to open " + path.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
                 &yyjson_doc_free);
  if (!doc) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: JSON parse failed at byte " +
        std::to_string(error.pos) + ": " + error.msg);
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  if (!yyjson_is_obj(root)) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: root must be an object");
  }
  AutotestReport report{};
  report.kind = readString(root, "kind", report.kind);
  report.generatedAtUtc = readString(root, "generatedAtUtc");
  report.command = readString(root, "command");
  report.reproduceCommand = readString(root, "reproduceCommand");
  yyjson_val *caseObject = yyjson_obj_get(root, "case");
  if (yyjson_is_obj(caseObject)) {
    report.testCase.id = readString(caseObject, "id");
    report.testCase.suite = readString(caseObject, "suite");
    report.testCase.description = readString(caseObject, "description");
    report.testCase.endFrame = readU32(caseObject, "endFrame");
  }
  yyjson_val *run = yyjson_obj_get(root, "run");
  if (yyjson_is_obj(run)) {
    report.run.fixedDeltaSeconds =
        readDouble(run, "fixedDeltaSeconds", report.run.fixedDeltaSeconds);
    report.run.warmupFrames = readU32(run, "warmupFrames");
    report.run.endFrame = readU32(run, "endFrame");
    report.run.renderedFrames = readU32(run, "renderedFrames");
    report.run.readoutDrainFrames = readU32(run, "readoutDrainFrames");
    report.run.readoutDrainTimeoutMs =
        readU32(run, "readoutDrainTimeoutMs", report.run.readoutDrainTimeoutMs);
    report.run.captureSynchronization = readString(
        run, "captureSynchronization", report.run.captureSynchronization);
    report.run.validForComparison =
        readBool(run, "validForComparison", report.run.validForComparison);
  }
  yyjson_val *artifacts = yyjson_obj_get(root, "artifacts");
  if (yyjson_is_obj(artifacts)) {
    report.artifacts.artifactDir = readString(artifacts, "artifactDir");
    report.artifacts.rootHtml = readString(artifacts, "rootHtml");
    report.artifacts.caseDir = readString(artifacts, "caseDir");
    report.artifacts.caseHtml = readString(artifacts, "caseHtml");
  }
  yyjson_val *checkpoints = yyjson_obj_get(root, "checkpoints");
  if (yyjson_is_arr(checkpoints)) {
    yyjson_arr_iter checkpointIter;
    yyjson_arr_iter_init(checkpoints, &checkpointIter);
    yyjson_val *checkpointValue = nullptr;
    while ((checkpointValue = yyjson_arr_iter_next(&checkpointIter)) !=
           nullptr) {
      if (!yyjson_is_obj(checkpointValue)) {
        continue;
      }
      AutotestCheckpointReport checkpoint{};
      checkpoint.id = readString(checkpointValue, "id");
      checkpoint.frame = readU32(checkpointValue, "frame");
      yyjson_val *measurements =
          yyjson_obj_get(checkpointValue, "measurements");
      if (yyjson_is_obj(measurements)) {
        yyjson_obj_iter iter;
        yyjson_obj_iter_init(measurements, &iter);
        yyjson_val *key = nullptr;
        while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
          yyjson_val *value = yyjson_obj_iter_get_val(key);
          if (yyjson_is_num(value)) {
            checkpoint.measurements.emplace(
                std::string(yyjson_get_str(key), yyjson_get_len(key)),
                yyjson_get_num(value));
          }
        }
      }
      yyjson_val *assertions = yyjson_obj_get(checkpointValue, "assertions");
      if (yyjson_is_arr(assertions)) {
        yyjson_arr_iter assertionIter;
        yyjson_arr_iter_init(assertions, &assertionIter);
        yyjson_val *assertionValue = nullptr;
        while ((assertionValue = yyjson_arr_iter_next(&assertionIter)) !=
               nullptr) {
          if (!yyjson_is_obj(assertionValue)) {
            continue;
          }
          checkpoint.assertions.push_back(readAssertionResult(assertionValue));
        }
      }
      yyjson_val *readouts = yyjson_obj_get(checkpointValue, "readouts");
      if (yyjson_is_arr(readouts)) {
        yyjson_arr_iter readoutIter;
        yyjson_arr_iter_init(readouts, &readoutIter);
        yyjson_val *readoutValue = nullptr;
        while ((readoutValue = yyjson_arr_iter_next(&readoutIter)) != nullptr) {
          if (!yyjson_is_obj(readoutValue)) {
            continue;
          }
          AutotestReadoutReport readout{};
          readout.checkpointId = readString(readoutValue, "checkpointId");
          readout.id = readString(readoutValue, "id");
          readout.type = readString(readoutValue, "type");
          readout.requestId = readU64(readoutValue, "requestId");
          readout.requestFrame = readU32(readoutValue, "requestFrame");
          readout.resultFrame = readU32(readoutValue, "resultFrame");
          readout.required = readBool(readoutValue, "required", true);
          readout.status = readString(readoutValue, "status", "pending");
          readout.statusReason =
              readString(readoutValue, "statusReason", "pending");
          yyjson_val *values = yyjson_obj_get(readoutValue, "values");
          if (yyjson_is_obj(values)) {
            yyjson_obj_iter iter;
            yyjson_obj_iter_init(values, &iter);
            yyjson_val *key = nullptr;
            while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
              yyjson_val *value = yyjson_obj_iter_get_val(key);
              if (yyjson_is_num(value)) {
                readout.values.emplace(
                    std::string(yyjson_get_str(key), yyjson_get_len(key)),
                    yyjson_get_num(value));
              }
            }
          }
          yyjson_val *readoutAssertions =
              yyjson_obj_get(readoutValue, "assertions");
          if (yyjson_is_arr(readoutAssertions)) {
            yyjson_arr_iter assertionIter;
            yyjson_arr_iter_init(readoutAssertions, &assertionIter);
            yyjson_val *assertionValue = nullptr;
            while ((assertionValue = yyjson_arr_iter_next(&assertionIter)) !=
                   nullptr) {
              if (yyjson_is_obj(assertionValue)) {
                readout.assertions.push_back(
                    readAssertionResult(assertionValue));
              }
            }
          }
          checkpoint.readouts.push_back(std::move(readout));
        }
      }
      report.checkpoints.push_back(std::move(checkpoint));
    }
  }
  yyjson_val *metricWindows = yyjson_obj_get(root, "metricWindows");
  if (yyjson_is_arr(metricWindows)) {
    yyjson_arr_iter windowIter;
    yyjson_arr_iter_init(metricWindows, &windowIter);
    yyjson_val *windowValue = nullptr;
    while ((windowValue = yyjson_arr_iter_next(&windowIter)) != nullptr) {
      if (!yyjson_is_obj(windowValue)) {
        continue;
      }
      AutotestMetricWindowReport window{};
      window.id = readString(windowValue, "id");
      window.startFrame = readU32(windowValue, "startFrame");
      window.endFrame = readU32(windowValue, "endFrame");
      yyjson_val *assertions = yyjson_obj_get(windowValue, "assertions");
      if (yyjson_is_arr(assertions)) {
        yyjson_arr_iter assertionIter;
        yyjson_arr_iter_init(assertions, &assertionIter);
        yyjson_val *assertionValue = nullptr;
        while ((assertionValue = yyjson_arr_iter_next(&assertionIter)) !=
               nullptr) {
          if (!yyjson_is_obj(assertionValue)) {
            continue;
          }
          window.assertions.push_back(readAssertionResult(assertionValue));
        }
      }
      report.metricWindows.push_back(std::move(window));
    }
  }
  yyjson_val *frames = yyjson_obj_get(root, "frames");
  if (yyjson_is_arr(frames)) {
    yyjson_arr_iter frameIter;
    yyjson_arr_iter_init(frames, &frameIter);
    yyjson_val *frameValue = nullptr;
    while ((frameValue = yyjson_arr_iter_next(&frameIter)) != nullptr) {
      if (!yyjson_is_obj(frameValue)) {
        continue;
      }
      AutotestFrameReport frame{};
      frame.frameIndex = readU64(frameValue, "frameIndex");
      yyjson_val *measurements = yyjson_obj_get(frameValue, "measurements");
      if (yyjson_is_obj(measurements)) {
        yyjson_obj_iter iter;
        yyjson_obj_iter_init(measurements, &iter);
        yyjson_val *key = nullptr;
        while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
          yyjson_val *value = yyjson_obj_iter_get_val(key);
          if (yyjson_is_num(value)) {
            frame.measurements.emplace(
                std::string(yyjson_get_str(key), yyjson_get_len(key)),
                yyjson_get_num(value));
          }
        }
      }
      yyjson_val *unavailable =
          yyjson_obj_get(frameValue, "unavailableMetrics");
      if (yyjson_is_arr(unavailable)) {
        yyjson_arr_iter unavailableIter;
        yyjson_arr_iter_init(unavailable, &unavailableIter);
        yyjson_val *value = nullptr;
        while ((value = yyjson_arr_iter_next(&unavailableIter)) != nullptr) {
          if (yyjson_is_str(value)) {
            frame.unavailableMetrics.emplace_back(yyjson_get_str(value),
                                                  yyjson_get_len(value));
          }
        }
      }
      report.frames.push_back(std::move(frame));
    }
  }
  yyjson_val *unavailable = yyjson_obj_get(root, "unavailableMetrics");
  if (yyjson_is_arr(unavailable)) {
    yyjson_arr_iter unavailableIter;
    yyjson_arr_iter_init(unavailable, &unavailableIter);
    yyjson_val *value = nullptr;
    while ((value = yyjson_arr_iter_next(&unavailableIter)) != nullptr) {
      if (yyjson_is_str(value)) {
        report.unavailableMetrics.emplace_back(yyjson_get_str(value),
                                               yyjson_get_len(value));
      }
    }
  }
  return Result<AutotestReport, std::string>::makeResult(std::move(report));
}

Result<std::string, std::string>
writeAutotestHtmlReport(const AutotestReport &report) {
  std::ostringstream out;
  out << "<!doctype html><html><head><meta charset=\"utf-8\">";
  out << "<title>Nuri Autotest " << htmlEscape(report.testCase.id)
      << "</title>";
  out << "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;"
         "color:#202124;background:#f7f8fa}table{border-collapse:collapse;"
         "width:100%;background:#fff}td,th{border:1px solid #d8dce3;"
         "padding:6px 8px;text-align:left}.status-fail,.status-invalid{"
         "color:#a40000;font-weight:600}.status-warn{color:#8a5a00;"
         "font-weight:600}.status-pass{color:#126b2e;font-weight:600}"
         ".checkpoint{margin:18px 0;padding:14px;background:#fff;"
         "border:1px solid #d8dce3;border-radius:6px}"
         "img{max-width:320px;border:1px solid #d8dce3}</style></head><body>";
  out << "<h1>" << htmlEscape(report.testCase.id) << "</h1>";
  out << "<p>suite: " << htmlEscape(report.testCase.suite)
      << " generated: " << htmlEscape(report.generatedAtUtc) << "</p>";
  if (!report.warnings.empty()) {
    out << "<h2>Warnings</h2><ul>";
    for (const std::string &warning : report.warnings) {
      out << "<li>" << htmlEscape(warning) << "</li>";
    }
    out << "</ul>";
  }
  if (!report.errors.empty()) {
    out << "<h2>Errors</h2><ul>";
    for (const std::string &error : report.errors) {
      out << "<li>" << htmlEscape(error) << "</li>";
    }
    out << "</ul>";
  }
  for (const AutotestCheckpointReport &checkpoint : report.checkpoints) {
    out << "<section class=\"checkpoint\"><h2>" << htmlEscape(checkpoint.id)
        << " frame " << checkpoint.frame << "</h2>";
    out << "<h3>Captures</h3><table><tr><th>Target</th><th>Status</th>"
           "<th>Preview</th><th>Diff</th></tr>";
    for (const AutotestCaptureReport &capture : checkpoint.captures) {
      out << "<tr><td>" << htmlEscape(capture.target) << "</td><td class=\""
          << "status-" << htmlEscape(capture.snapshot.status) << "\">"
          << htmlEscape(capture.snapshot.status) << " "
          << htmlEscape(capture.snapshot.statusReason) << "</td><td>";
      if (!capture.snapshot.preview.empty()) {
        out << "<img src=\""
            << htmlEscape(capture.snapshot.preview.generic_string())
            << "\" alt=\"" << htmlEscape(capture.target) << "\">";
      }
      out << "</td><td>";
      if (!capture.snapshot.diff.empty()) {
        out << "<img src=\""
            << htmlEscape(capture.snapshot.diff.generic_string())
            << "\" alt=\"diff\">";
      }
      out << "</td></tr>";
    }
    out << "</table>";
    if (!checkpoint.readouts.empty()) {
      out << "<h3>Readouts</h3><table><tr><th>ID</th><th>Type</th>"
             "<th>Status</th><th>Request</th><th>Result</th>"
             "<th>Values</th><th>Assertions</th></tr>";
      for (const AutotestReadoutReport &readout : checkpoint.readouts) {
        out << "<tr><td>" << htmlEscape(readout.id) << "</td><td>"
            << htmlEscape(readout.type) << "</td><td class=\"status-"
            << htmlEscape(readout.status) << "\">" << htmlEscape(readout.status)
            << " " << htmlEscape(readout.statusReason) << "</td><td>"
            << readout.requestFrame << "</td><td>";
        if (readout.resultFrame != 0u || readout.status == "pass" ||
            readout.status == "fail" || readout.status == "warn") {
          out << readout.resultFrame;
        }
        out << "</td><td>";
        bool firstValue = true;
        for (const auto &[key, value] : readout.values) {
          if (!firstValue) {
            out << ", ";
          }
          firstValue = false;
          out << htmlEscape(key) << "=" << value;
        }
        out << "</td><td>";
        bool firstAssertion = true;
        for (const AutotestAssertionResult &assertion : readout.assertions) {
          const std::string status =
              autotestAssertionStatusName(assertion.status);
          if (!firstAssertion) {
            out << "<br>";
          }
          firstAssertion = false;
          out << htmlEscape(assertion.id) << ": "
              << "<span class=\"status-" << htmlEscape(status) << "\">"
              << htmlEscape(status) << "</span>";
          if (assertion.hasActual) {
            out << " " << assertion.actual;
          }
        }
        out << "</td></tr>";
      }
      out << "</table>";
    }
    out << "<h3>Assertions</h3><table><tr><th>ID</th><th>Metric</th>"
           "<th>Status</th><th>Actual</th></tr>";
    for (const AutotestAssertionResult &assertion : checkpoint.assertions) {
      const std::string status = autotestAssertionStatusName(assertion.status);
      out << "<tr><td>" << htmlEscape(assertion.id) << "</td><td>"
          << htmlEscape(assertion.metric) << "</td><td class=\"status-"
          << htmlEscape(status) << "\">" << htmlEscape(status) << " "
          << htmlEscape(assertion.statusReason) << "</td><td>";
      if (assertion.hasActual) {
        out << assertion.actual;
      }
      out << "</td></tr>";
    }
    out << "</table></section>";
  }
  if (!report.metricWindows.empty()) {
    out << "<h2>Metric Windows</h2>";
    for (const AutotestMetricWindowReport &window : report.metricWindows) {
      out << "<section class=\"checkpoint\"><h3>" << htmlEscape(window.id)
          << " frames " << window.startFrame << "-" << window.endFrame
          << "</h3><table><tr><th>ID</th><th>Metric</th><th>Statistic</th>"
             "<th>Status</th><th>Actual</th><th>Samples</th></tr>";
      for (const AutotestAssertionResult &assertion : window.assertions) {
        const std::string status =
            autotestAssertionStatusName(assertion.status);
        out << "<tr><td>" << htmlEscape(assertion.id) << "</td><td>"
            << htmlEscape(assertion.metric) << "</td><td>"
            << htmlEscape(assertion.statistic) << "</td><td class=\"status-"
            << htmlEscape(status) << "\">" << htmlEscape(status) << " "
            << htmlEscape(assertion.statusReason) << "</td><td>";
        if (assertion.hasActual) {
          out << assertion.actual;
        }
        out << "</td><td>" << assertion.sampleCount << "</td></tr>";
      }
      out << "</table></section>";
    }
  }
  out << "</body></html>";
  return Result<std::string, std::string>::makeResult(out.str());
}

Result<bool, std::string>
writeAutotestHtmlReportFile(const AutotestReport &report,
                            const std::filesystem::path &path) {
  auto html = writeAutotestHtmlReport(report);
  if (html.hasError()) {
    return Result<bool, std::string>::makeError(html.error());
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return Result<bool, std::string>::makeError(
        "writeAutotestHtmlReportFile: failed to open " + path.string());
  }
  file << html.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<std::string, std::string>
writeAutotestSuiteHtml(const std::vector<AutotestReport> &reports,
                       std::string_view suite) {
  std::ostringstream out;
  out << "<!doctype html><html><head><meta charset=\"utf-8\"><title>Nuri "
         "Autotest Suite</title></head><body><h1>Autotest suite "
      << htmlEscape(suite) << "</h1><ul>";
  for (const AutotestReport &report : reports) {
    out << "<li>" << htmlEscape(report.testCase.id)
        << " checkpoints=" << report.checkpoints.size()
        << " warnings=" << report.warnings.size()
        << " errors=" << report.errors.size() << "</li>";
  }
  out << "</ul></body></html>";
  return Result<std::string, std::string>::makeResult(out.str());
}

Result<bool, std::string>
writeAutotestSuiteHtmlFile(const std::vector<AutotestReport> &reports,
                           std::string_view suite,
                           const std::filesystem::path &path) {
  auto html = writeAutotestSuiteHtml(reports, suite);
  if (html.hasError()) {
    return Result<bool, std::string>::makeError(html.error());
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return Result<bool, std::string>::makeError(
        "writeAutotestSuiteHtmlFile: failed to open " + path.string());
  }
  file << html.value();
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri::tools::autotest
