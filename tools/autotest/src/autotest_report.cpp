#include "nuri/tools/autotest/autotest_report.h"

#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/fingerprint.h"
#include "nuri/tools/core/html_report.h"
#include "nuri/tools/core/json_contract.h"
#include "nuri/tools/core/result_envelope_v2.h"
#include "nuri/tools/core/sha256.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string_view>
#include <utility>

#include <yyjson.h>

namespace nuri::tools::autotest {
namespace {

using JsonMutDocPtr =
    std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;
using JsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
using JsonField = nuri::tools::core::JsonFieldContract;
using JsonType = nuri::tools::core::JsonFieldType;

template <size_t Size>
[[nodiscard]] Result<void, std::string>
validateObject(yyjson_val *object, const std::array<JsonField, Size> &fields,
               std::string_view path) {
  return nuri::tools::core::validateJsonObject(object, fields, path);
}

[[nodiscard]] Result<void, std::string>
validateStringArrayValue(yyjson_val *array, std::string_view path) {
  if (!yyjson_is_arr(array)) {
    return Result<void, std::string>::makeError(std::string(path) +
                                                " must be an array");
  }
  yyjson_arr_iter iterator{};
  yyjson_arr_iter_init(array, &iterator);
  yyjson_val *entry = nullptr;
  size_t index = 0u;
  while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
    if (!yyjson_is_str(entry)) {
      return Result<void, std::string>::makeError(std::string(path) + "[" +
                                                  std::to_string(index) +
                                                  "] must be a string");
    }
    ++index;
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<void, std::string>
validateNumberMap(yyjson_val *object, std::string_view path) {
  if (!yyjson_is_obj(object)) {
    return Result<void, std::string>::makeError(std::string(path) +
                                                " must be an object");
  }
  yyjson_obj_iter iterator{};
  yyjson_obj_iter_init(object, &iterator);
  yyjson_val *key = nullptr;
  while ((key = yyjson_obj_iter_next(&iterator)) != nullptr) {
    yyjson_val *value = yyjson_obj_iter_get_val(key);
    if (!yyjson_is_num(value) && !yyjson_is_null(value)) {
      return Result<void, std::string>::makeError(std::string(path) + "." +
                                                  yyjson_get_str(key) +
                                                  " must be a number or null");
    }
  }
  return Result<void, std::string>::makeResult();
}

void addFiniteReal(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
                   double value) {
  if (std::isfinite(value)) {
    yyjson_mut_obj_add_real(doc, object, key, value);
  } else {
    yyjson_mut_obj_add_null(doc, object, key);
  }
}

void addString(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
               const std::string &value) {
  yyjson_mut_obj_add_strcpy(doc, object, key, value.c_str());
}

void addPath(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
             const std::filesystem::path &value) {
  const std::u8string utf8 = value.generic_u8string();
  addString(
      doc, object, key,
      std::string(reinterpret_cast<const char *>(utf8.data()), utf8.size()));
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
  yyjson_mut_obj_add_uint(doc, object, "gpuVendorId", env.gpuVendorId);
  yyjson_mut_obj_add_uint(doc, object, "gpuDeviceId", env.gpuDeviceId);
  addString(doc, object, "gpuDriverVersion", env.gpuDriverVersion);
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
  addFiniteReal(doc, object, "fixedDeltaSeconds", testCase.fixedDeltaSeconds);
  addString(doc, object, "presentMode", testCase.presentMode);
  addString(doc, object, "windowMode", testCase.windowMode);
  yyjson_mut_obj_add_bool(doc, object, "authoritative", testCase.authoritative);
  return object;
}

yyjson_mut_val *makeRunObject(yyjson_mut_doc *doc,
                              const AutotestRunMetadata &run) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addFiniteReal(doc, object, "fixedDeltaSeconds", run.fixedDeltaSeconds);
  yyjson_mut_obj_add_uint(doc, object, "warmupFrames", run.warmupFrames);
  yyjson_mut_obj_add_uint(doc, object, "endFrame", run.endFrame);
  yyjson_mut_obj_add_uint(doc, object, "renderedFrames", run.renderedFrames);
  yyjson_mut_obj_add_uint(doc, object, "readoutDrainFrames",
                          run.readoutDrainFrames);
  yyjson_mut_obj_add_uint(doc, object, "readoutDrainFrameLimit",
                          run.readoutDrainFrameLimit);
  yyjson_mut_obj_add_uint(doc, object, "readoutDrainTimeoutMs",
                          run.readoutDrainTimeoutMs);
  yyjson_mut_obj_add_uint(doc, object, "readoutDrainElapsedMs",
                          run.readoutDrainElapsedMs);
  addString(doc, object, "requestedWindowMode", run.requestedWindowMode);
  addString(doc, object, "resolvedWindowMode", run.resolvedWindowMode);
  addString(doc, object, "windowModeSource", run.windowModeSource);
  addString(doc, object, "captureSynchronization", run.captureSynchronization);
  yyjson_mut_obj_add_bool(doc, object, "validForComparison",
                          run.validForComparison);
  return object;
}

yyjson_mut_val *makeSelectionObject(yyjson_mut_doc *doc,
                                    const AutotestSelectionSummary &selection) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "requested", selection.requested);
  yyjson_mut_obj_add_uint(doc, object, "selected", selection.selected);
  yyjson_mut_obj_add_uint(doc, object, "attempted", selection.attempted);
  yyjson_mut_obj_add_uint(doc, object, "completed", selection.completed);
  yyjson_mut_obj_add_uint(doc, object, "passed", selection.passed);
  yyjson_mut_obj_add_uint(doc, object, "failed", selection.failed);
  yyjson_mut_obj_add_uint(doc, object, "unavailable", selection.unavailable);
  yyjson_mut_obj_add_uint(doc, object, "notRun", selection.notRun);
  return object;
}

yyjson_mut_val *
makeMeasurementsObject(yyjson_mut_doc *doc,
                       const std::map<std::string, double> &measurements) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  for (const auto &[key, value] : measurements) {
    addFiniteReal(doc, object, key.c_str(), value);
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
  addString(doc, object, "kind", capture.kind);
  addString(doc, object, "lifetime", capture.lifetime);
  addString(doc, object, "format", capture.format);
  addString(doc, object, "colorSpace", capture.colorSpace);
  addString(doc, object, "origin", capture.origin);
  yyjson_mut_obj_add_uint(doc, object, "width", capture.width);
  yyjson_mut_obj_add_uint(doc, object, "height", capture.height);
  yyjson_mut_obj_add_uint(doc, object, "mip", capture.mip);
  yyjson_mut_obj_add_uint(doc, object, "layer", capture.layer);
  addString(doc, object, "actualHash", capture.actualHash);
  addString(doc, object, "expectedHash", capture.expectedHash);
  addPath(doc, object, "actual", capture.actual);
  addPath(doc, object, "actualMetadata", capture.actualMetadata);
  addPath(doc, object, "preview", capture.preview);
  addPath(doc, object, "expected", capture.expected);
  addPath(doc, object, "diff", capture.diff);
  yyjson_mut_val *metrics = yyjson_mut_obj(doc);
  addFiniteReal(doc, metrics, "meanAbsError", capture.metrics.meanAbsError);
  addFiniteReal(doc, metrics, "rmse", capture.metrics.rmse);
  addFiniteReal(doc, metrics, "maxAbsError", capture.metrics.maxAbsError);
  addFiniteReal(doc, metrics, "p99AbsError", capture.metrics.p99AbsError);
  yyjson_mut_obj_add_uint(doc, metrics, "failingValues",
                          capture.metrics.failingValues);
  yyjson_mut_obj_add_uint(doc, metrics, "comparedValues",
                          capture.metrics.comparedValues);
  yyjson_mut_obj_add_val(doc, object, "metrics", metrics);
  yyjson_mut_val *semantic = yyjson_mut_obj(doc);
  addString(doc, semantic, "unit", capture.semanticMetrics.unit);
  addFiniteReal(doc, semantic, "meanError", capture.semanticMetrics.meanError);
  addFiniteReal(doc, semantic, "maxError", capture.semanticMetrics.maxError);
  addFiniteReal(doc, semantic, "rmse", capture.semanticMetrics.rmse);
  addFiniteReal(doc, semantic, "p99Error", capture.semanticMetrics.p99Error);
  yyjson_mut_obj_add_uint(doc, semantic, "failingPixels",
                          capture.semanticMetrics.failingPixels);
  yyjson_mut_obj_add_uint(doc, semantic, "validPixels",
                          capture.semanticMetrics.validPixels);
  yyjson_mut_obj_add_uint(doc, semantic, "ignoredPixels",
                          capture.semanticMetrics.ignoredPixels);
  yyjson_mut_obj_add_uint(doc, semantic, "changedPixels",
                          capture.semanticMetrics.changedPixels);
  yyjson_mut_obj_add_bool(doc, semantic, "changedBoundsValid",
                          capture.semanticMetrics.changedBoundsValid);
  yyjson_mut_obj_add_uint(doc, semantic, "minChangedX",
                          capture.semanticMetrics.minChangedX);
  yyjson_mut_obj_add_uint(doc, semantic, "minChangedY",
                          capture.semanticMetrics.minChangedY);
  yyjson_mut_obj_add_uint(doc, semantic, "maxChangedX",
                          capture.semanticMetrics.maxChangedX);
  yyjson_mut_obj_add_uint(doc, semantic, "maxChangedY",
                          capture.semanticMetrics.maxChangedY);
  yyjson_mut_obj_add_uint(doc, semantic, "maxErrorX",
                          capture.semanticMetrics.maxErrorX);
  yyjson_mut_obj_add_uint(doc, semantic, "maxErrorY",
                          capture.semanticMetrics.maxErrorY);
  addString(doc, semantic, "secondaryUnit",
            capture.semanticMetrics.secondaryUnit);
  addFiniteReal(doc, semantic, "meanSecondaryError",
                capture.semanticMetrics.meanSecondaryError);
  addFiniteReal(doc, semantic, "maxSecondaryError",
                capture.semanticMetrics.maxSecondaryError);
  addFiniteReal(doc, semantic, "secondaryRmse",
                capture.semanticMetrics.secondaryRmse);
  addFiniteReal(doc, semantic, "p99SecondaryError",
                capture.semanticMetrics.p99SecondaryError);
  yyjson_mut_obj_add_uint(doc, semantic, "secondaryFailingPixels",
                          capture.semanticMetrics.secondaryFailingPixels);
  yyjson_mut_obj_add_uint(doc, semantic, "truePositivePixels",
                          capture.semanticMetrics.truePositivePixels);
  yyjson_mut_obj_add_uint(doc, semantic, "trueNegativePixels",
                          capture.semanticMetrics.trueNegativePixels);
  yyjson_mut_obj_add_uint(doc, semantic, "falsePositivePixels",
                          capture.semanticMetrics.falsePositivePixels);
  yyjson_mut_obj_add_uint(doc, semantic, "falseNegativePixels",
                          capture.semanticMetrics.falseNegativePixels);
  addFiniteReal(doc, semantic, "intersectionOverUnion",
                capture.semanticMetrics.intersectionOverUnion);
  yyjson_mut_obj_add_val(doc, object, "semanticMetrics", semantic);
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

yyjson_mut_val *
makeMotionOracleObject(yyjson_mut_doc *doc,
                       const AutotestMotionOracleReport &motionOracle) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "status", motionOracle.status);
  addString(doc, object, "statusReason", motionOracle.statusReason);
  addString(doc, object, "motionTarget", motionOracle.motionTarget);
  addString(doc, object, "motionClassTarget", motionOracle.motionClassTarget);
  yyjson_mut_val *roi = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, roi, "x", motionOracle.roi.x);
  yyjson_mut_obj_add_uint(doc, roi, "y", motionOracle.roi.y);
  yyjson_mut_obj_add_uint(doc, roi, "width", motionOracle.roi.width);
  yyjson_mut_obj_add_uint(doc, roi, "height", motionOracle.roi.height);
  yyjson_mut_obj_add_val(doc, object, "roi", roi);
  yyjson_mut_obj_add_uint(doc, object, "selectedPixelCount",
                          motionOracle.selectedPixelCount);
  yyjson_mut_val *expected = yyjson_mut_arr(doc);
  yyjson_mut_arr_add_real(doc, expected,
                          motionOracle.expectedVelocityPixels[0]);
  yyjson_mut_arr_add_real(doc, expected,
                          motionOracle.expectedVelocityPixels[1]);
  yyjson_mut_obj_add_val(doc, object, "expectedVelocityPixels", expected);
  yyjson_mut_val *mean = yyjson_mut_arr(doc);
  yyjson_mut_arr_add_real(doc, mean, motionOracle.meanVelocityPixels[0]);
  yyjson_mut_arr_add_real(doc, mean, motionOracle.meanVelocityPixels[1]);
  yyjson_mut_obj_add_val(doc, object, "meanVelocityPixels", mean);
  addFiniteReal(doc, object, "meanErrorPixels", motionOracle.meanErrorPixels);
  addFiniteReal(doc, object, "p95ErrorPixels", motionOracle.p95ErrorPixels);
  addFiniteReal(doc, object, "maxErrorPixels", motionOracle.maxErrorPixels);
  addFiniteReal(doc, object, "p95ScaleErrorPixels",
                motionOracle.p95ScaleErrorPixels);
  addFiniteReal(doc, object, "maxScaleErrorPixels",
                motionOracle.maxScaleErrorPixels);
  yyjson_mut_obj_add_uint(doc, object, "wrongSignPixelCount",
                          motionOracle.wrongSignPixelCount);
  yyjson_mut_obj_add_bool(doc, object, "classCoverageAvailable",
                          motionOracle.classCoverageAvailable);
  yyjson_mut_obj_add_uint(doc, object, "classSampleCount",
                          motionOracle.classSampleCount);
  addFiniteReal(doc, object, "invalidClassCoverage",
                motionOracle.invalidClassCoverage);
  addFiniteReal(doc, object, "staticClassCoverage",
                motionOracle.staticClassCoverage);
  addFiniteReal(doc, object, "fullClassCoverage",
                motionOracle.fullClassCoverage);
  addFiniteReal(doc, object, "p95ErrorMaxPixels",
                motionOracle.p95ErrorMaxPixels);
  addFiniteReal(doc, object, "maxErrorMaxPixels",
                motionOracle.maxErrorMaxPixels);
  yyjson_mut_obj_add_val(doc, object, "failedThresholds",
                         makeStringArray(doc, motionOracle.failedThresholds));
  return object;
}

yyjson_mut_val *
makeQualityOracleObject(yyjson_mut_doc *doc,
                        const AutotestQualityOracleReport &quality) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "status", quality.status);
  addString(doc, object, "statusReason", quality.statusReason);
  addString(doc, object, "outputTarget", quality.outputTarget);
  addString(doc, object, "referencePath", quality.referencePath);
  yyjson_mut_obj_add_uint(doc, object, "schemaVersion", quality.schemaVersion);
  yyjson_mut_obj_add_uint(doc, object, "referenceVersion",
                          quality.referenceVersion);
  yyjson_mut_obj_add_uint(doc, object, "maskVersion", quality.maskVersion);
  addFiniteReal(doc, object, "lscale", quality.lscale);
  yyjson_mut_obj_add_uint(doc, object, "selectedPixelCount",
                          quality.selectedPixelCount);
  yyjson_mut_obj_add_uint(doc, object, "finitePixelCount",
                          quality.finitePixelCount);
  yyjson_mut_obj_add_uint(doc, object, "nonFiniteValueCount",
                          quality.nonFiniteValueCount);
  addFiniteReal(doc, object, "normalizedHdrMae", quality.normalizedHdrMae);
  addFiniteReal(doc, object, "normalizedHdrRmse", quality.normalizedHdrRmse);
  addFiniteReal(doc, object, "lumaSsim", quality.lumaSsim);
  yyjson_mut_obj_add_uint(doc, object, "darkCollapsePixelCount",
                          quality.darkCollapsePixelCount);
  addFiniteReal(doc, object, "darkCollapsePercent",
                quality.darkCollapsePercent);
  yyjson_mut_obj_add_uint(doc, object, "darkCollapseMaxComponentPixels",
                          quality.darkCollapseMaxComponentPixels);
  addFiniteReal(doc, object, "relativeLumaEnergyDrift",
                quality.relativeLumaEnergyDrift);
  yyjson_mut_obj_add_bool(doc, object, "edgeAvailable", quality.edgeAvailable);
  addString(doc, object, "edgeAxis", quality.edgeAxis);
  yyjson_mut_obj_add_uint(doc, object, "edgeProfileCount",
                          quality.edgeProfileCount);
  yyjson_mut_obj_add_uint(doc, object, "edgeUnresolvedProfileCount",
                          quality.edgeUnresolvedProfileCount);
  addFiniteReal(doc, object, "referenceEdgeWidth10To90",
                quality.referenceEdgeWidth10To90);
  addFiniteReal(doc, object, "outputEdgeWidth10To90",
                quality.outputEdgeWidth10To90);
  addFiniteReal(doc, object, "edgeWidthRatio", quality.edgeWidthRatio);
  addFiniteReal(doc, object, "edgeOvershoot", quality.edgeOvershoot);
  addFiniteReal(doc, object, "edgeUndershoot", quality.edgeUndershoot);
  yyjson_mut_obj_add_bool(doc, object, "temporalAvailable",
                          quality.temporalAvailable);
  yyjson_mut_obj_add_uint(doc, object, "temporalSampleCount",
                          quality.temporalSampleCount);
  addFiniteReal(doc, object, "temporalError", quality.temporalError);
  yyjson_mut_obj_add_bool(doc, object, "revealAvailable",
                          quality.revealAvailable);
  yyjson_mut_obj_add_uint(doc, object, "revealPixelCount",
                          quality.revealPixelCount);
  addFiniteReal(doc, object, "ghostEnergy", quality.ghostEnergy);
  addFiniteReal(doc, object, "recoveryRmse", quality.recoveryRmse);
  addFiniteReal(doc, object, "normalizedMaeMax",
                quality.budgets.normalizedMaeMax);
  addFiniteReal(doc, object, "normalizedRmseMax",
                quality.budgets.normalizedRmseMax);
  addFiniteReal(doc, object, "lumaSsimMin", quality.budgets.lumaSsimMin);
  addFiniteReal(doc, object, "darkCollapsePercentMax",
                quality.budgets.darkCollapsePercentMax);
  yyjson_mut_obj_add_uint(doc, object, "darkCollapseComponentMaxPixels",
                          quality.budgets.darkCollapseComponentMaxPixels);
  addFiniteReal(doc, object, "relativeLumaEnergyDriftMax",
                quality.budgets.relativeLumaEnergyDriftMax);
  addFiniteReal(doc, object, "edgeWidthRatioMin",
                quality.budgets.edgeWidthRatioMin);
  addFiniteReal(doc, object, "edgeWidthRatioMax",
                quality.budgets.edgeWidthRatioMax);
  addFiniteReal(doc, object, "edgeOvershootMax",
                quality.budgets.edgeOvershootMax);
  addFiniteReal(doc, object, "edgeUndershootMax",
                quality.budgets.edgeUndershootMax);
  addFiniteReal(doc, object, "temporalErrorMax",
                quality.budgets.temporalErrorMax);
  addFiniteReal(doc, object, "ghostEnergyMax", quality.budgets.ghostEnergyMax);
  addFiniteReal(doc, object, "recoveryRmseMax",
                quality.budgets.recoveryRmseMax);
  yyjson_mut_obj_add_val(doc, object, "failedThresholds",
                         makeStringArray(doc, quality.failedThresholds));
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
  if (assertion.hasActual) {
    addFiniteReal(doc, object, "actual", assertion.actual);
  } else {
    yyjson_mut_obj_add_null(doc, object, "actual");
  }
  yyjson_mut_obj_add_uint(doc, object, "sampleCount", assertion.sampleCount);
  yyjson_mut_obj_add_uint(doc, object, "expectedSampleCount",
                          assertion.expectedSampleCount);
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
  return nuri::tools::core::htmlEscape(text);
}

[[nodiscard]] std::string pathToUtf8(const std::filesystem::path &path) {
  const std::u8string encoded = path.generic_u8string();
  return {reinterpret_cast<const char *>(encoded.data()), encoded.size()};
}

[[nodiscard]] std::string
autotestArtifactHref(const AutotestReport &report,
                     const std::filesystem::path &path) {
  if (path.empty()) {
    return {};
  }
  const std::filesystem::path artifact =
      path.is_absolute() ? path : report.artifacts.caseDir / path;
  if (report.artifacts.caseHtml.empty()) {
    return pathToUtf8(artifact);
  }
  std::error_code error;
  const std::filesystem::path relative = std::filesystem::relative(
      artifact, report.artifacts.caseHtml.parent_path(), error);
  return pathToUtf8(error ? artifact : relative);
}

[[nodiscard]] std::string readableStatus(std::string_view status) {
  std::string label(status);
  std::replace(label.begin(), label.end(), '_', ' ');
  return label;
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

[[nodiscard]] int readInt(yyjson_val *object, const char *key,
                          int defaultValue = 0) {
  yyjson_val *value = yyjson_obj_get(object, key);
  if (yyjson_is_sint(value)) {
    return static_cast<int>(yyjson_get_sint(value));
  }
  if (yyjson_is_uint(value)) {
    return static_cast<int>(yyjson_get_uint(value));
  }
  return defaultValue;
}

void readStringArray(yyjson_val *object, const char *key,
                     std::vector<std::string> &out) {
  yyjson_val *array = yyjson_obj_get(object, key);
  if (!yyjson_is_arr(array)) {
    return;
  }
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(array, &iter);
  yyjson_val *value = nullptr;
  while ((value = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (yyjson_is_str(value)) {
      out.emplace_back(yyjson_get_str(value), yyjson_get_len(value));
    }
  }
}

void readMeasurements(yyjson_val *object, const char *key,
                      std::map<std::string, double> &out) {
  yyjson_val *measurements = yyjson_obj_get(object, key);
  if (!yyjson_is_obj(measurements)) {
    return;
  }
  yyjson_obj_iter iter;
  yyjson_obj_iter_init(measurements, &iter);
  yyjson_val *name = nullptr;
  while ((name = yyjson_obj_iter_next(&iter)) != nullptr) {
    yyjson_val *value = yyjson_obj_iter_get_val(name);
    if (yyjson_is_num(value)) {
      out.emplace(std::string(yyjson_get_str(name), yyjson_get_len(name)),
                  yyjson_get_num(value));
    }
  }
}

void readSelection(yyjson_val *object, AutotestSelectionSummary &selection) {
  if (!yyjson_is_obj(object)) {
    return;
  }
  selection.requested = readString(object, "requested");
  selection.selected = readU32(object, "selected");
  selection.attempted = readU32(object, "attempted");
  selection.completed = readU32(object, "completed");
  selection.passed = readU32(object, "passed");
  selection.failed = readU32(object, "failed");
  selection.unavailable = readU32(object, "unavailable");
  selection.notRun = readU32(object, "notRun");
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
  assertion.expectedSampleCount =
      readU32(assertionValue, "expectedSampleCount", assertion.sampleCount);
  return assertion;
}

[[nodiscard]] nuri::tools::snapshot::SnapshotCaptureReport
readSnapshotCapture(yyjson_val *object) {
  nuri::tools::snapshot::SnapshotCaptureReport capture{};
  capture.target = readString(object, "target");
  capture.artifactStem = readString(object, "artifactStem");
  capture.profile = readString(object, "profile");
  capture.required = readBool(object, "required", true);
  capture.available = readBool(object, "available");
  capture.capturePointVersion = readU32(object, "capturePointVersion");
  capture.captureFrameIndex = readU64(object, "captureFrameIndex");
  capture.kind = readString(object, "kind");
  capture.lifetime = readString(object, "lifetime");
  capture.format = readString(object, "format");
  capture.colorSpace = readString(object, "colorSpace");
  capture.origin = readString(object, "origin", capture.origin);
  capture.width = readU32(object, "width");
  capture.height = readU32(object, "height");
  capture.mip = readU32(object, "mip");
  capture.layer = readU32(object, "layer");
  capture.actualHash = readString(object, "actualHash");
  capture.expectedHash = readString(object, "expectedHash");
  capture.actual = readString(object, "actual");
  capture.actualMetadata = readString(object, "actualMetadata");
  capture.preview = readString(object, "preview");
  capture.expected = readString(object, "expected");
  capture.diff = readString(object, "diff");
  yyjson_val *metrics = yyjson_obj_get(object, "metrics");
  if (yyjson_is_obj(metrics)) {
    capture.metrics.meanAbsError = readDouble(metrics, "meanAbsError");
    capture.metrics.rmse = readDouble(metrics, "rmse");
    capture.metrics.maxAbsError = readDouble(metrics, "maxAbsError");
    capture.metrics.p99AbsError = readDouble(metrics, "p99AbsError");
    capture.metrics.failingValues = readU64(metrics, "failingValues");
    capture.metrics.comparedValues = readU64(metrics, "comparedValues");
  }
  yyjson_val *semantic = yyjson_obj_get(object, "semanticMetrics");
  if (yyjson_is_obj(semantic)) {
    capture.semanticMetrics.unit = readString(semantic, "unit");
    capture.semanticMetrics.meanError = readDouble(semantic, "meanError");
    capture.semanticMetrics.maxError = readDouble(semantic, "maxError");
    capture.semanticMetrics.rmse = readDouble(semantic, "rmse");
    capture.semanticMetrics.p99Error = readDouble(semantic, "p99Error");
    capture.semanticMetrics.failingPixels = readU64(semantic, "failingPixels");
    capture.semanticMetrics.validPixels = readU64(semantic, "validPixels");
    capture.semanticMetrics.ignoredPixels = readU64(semantic, "ignoredPixels");
    capture.semanticMetrics.changedPixels = readU64(semantic, "changedPixels");
    capture.semanticMetrics.changedBoundsValid =
        readBool(semantic, "changedBoundsValid");
    capture.semanticMetrics.minChangedX = readU32(semantic, "minChangedX");
    capture.semanticMetrics.minChangedY = readU32(semantic, "minChangedY");
    capture.semanticMetrics.maxChangedX = readU32(semantic, "maxChangedX");
    capture.semanticMetrics.maxChangedY = readU32(semantic, "maxChangedY");
    capture.semanticMetrics.maxErrorX = readU32(semantic, "maxErrorX");
    capture.semanticMetrics.maxErrorY = readU32(semantic, "maxErrorY");
    capture.semanticMetrics.secondaryUnit =
        readString(semantic, "secondaryUnit");
    capture.semanticMetrics.meanSecondaryError =
        readDouble(semantic, "meanSecondaryError");
    capture.semanticMetrics.maxSecondaryError =
        readDouble(semantic, "maxSecondaryError");
    capture.semanticMetrics.secondaryRmse =
        readDouble(semantic, "secondaryRmse");
    capture.semanticMetrics.p99SecondaryError =
        readDouble(semantic, "p99SecondaryError");
    capture.semanticMetrics.secondaryFailingPixels =
        readU64(semantic, "secondaryFailingPixels");
    capture.semanticMetrics.truePositivePixels =
        readU64(semantic, "truePositivePixels");
    capture.semanticMetrics.trueNegativePixels =
        readU64(semantic, "trueNegativePixels");
    capture.semanticMetrics.falsePositivePixels =
        readU64(semantic, "falsePositivePixels");
    capture.semanticMetrics.falseNegativePixels =
        readU64(semantic, "falseNegativePixels");
    capture.semanticMetrics.intersectionOverUnion =
        readDouble(semantic, "intersectionOverUnion");
  }
  readStringArray(object, "failedThresholds", capture.failedThresholds);
  capture.producerPassLabel = readString(object, "producerPassLabel");
  capture.readbackError = readString(object, "readbackError");
  capture.status = readString(object, "status", capture.status);
  capture.statusReason =
      readString(object, "statusReason", capture.statusReason);
  return capture;
}

[[nodiscard]] Result<void, std::string>
validateAssertionArray(yyjson_val *array, std::string_view path) {
  if (!yyjson_is_arr(array)) {
    return Result<void, std::string>::makeError(std::string(path) +
                                                " must be an array");
  }
  static constexpr std::array fields{
      JsonField{"id", JsonType::String},
      JsonField{"metric", JsonType::String},
      JsonField{"statistic", JsonType::String},
      JsonField{"status", JsonType::String},
      JsonField{"statusReason", JsonType::String},
      JsonField{"hasActual", JsonType::Boolean},
      JsonField{"actual", JsonType::NullOrNumber},
      JsonField{"sampleCount", JsonType::Unsigned},
      JsonField{"expectedSampleCount", JsonType::Unsigned},
  };
  yyjson_arr_iter iterator{};
  yyjson_arr_iter_init(array, &iterator);
  yyjson_val *entry = nullptr;
  size_t index = 0u;
  while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
    auto valid = validateObject(
        entry, fields, std::string(path) + "[" + std::to_string(index++) + "]");
    if (valid.hasError()) {
      return valid;
    }
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<void, std::string>
validateSnapshotCaptureV1(yyjson_val *snapshot, std::string_view path) {
  static constexpr std::array fields{
      JsonField{"target", JsonType::String},
      JsonField{"artifactStem", JsonType::String},
      JsonField{"profile", JsonType::String},
      JsonField{"required", JsonType::Boolean},
      JsonField{"available", JsonType::Boolean},
      JsonField{"capturePointVersion", JsonType::Unsigned},
      JsonField{"captureFrameIndex", JsonType::Unsigned},
      JsonField{"kind", JsonType::String, false},
      JsonField{"lifetime", JsonType::String},
      JsonField{"format", JsonType::String},
      JsonField{"colorSpace", JsonType::String},
      JsonField{"origin", JsonType::String},
      JsonField{"width", JsonType::Unsigned},
      JsonField{"height", JsonType::Unsigned},
      JsonField{"mip", JsonType::Unsigned},
      JsonField{"layer", JsonType::Unsigned},
      JsonField{"actualHash", JsonType::String},
      JsonField{"expectedHash", JsonType::String},
      JsonField{"actual", JsonType::String},
      JsonField{"actualMetadata", JsonType::String},
      JsonField{"preview", JsonType::String},
      JsonField{"expected", JsonType::String},
      JsonField{"diff", JsonType::String},
      JsonField{"metrics", JsonType::Object},
      JsonField{"semanticMetrics", JsonType::Object, false},
      JsonField{"failedThresholds", JsonType::Array},
      JsonField{"producerPassLabel", JsonType::String},
      JsonField{"readbackError", JsonType::String},
      JsonField{"status", JsonType::String},
      JsonField{"statusReason", JsonType::String},
  };
  auto valid = validateObject(snapshot, fields, path);
  if (valid.hasError()) {
    return valid;
  }
  for (std::string_view field :
       {"actual", "actualMetadata", "preview", "expected", "diff"}) {
    valid = nuri::tools::core::validateJsonArtifactPath(snapshot, field, path);
    if (valid.hasError()) {
      return valid;
    }
  }
  static constexpr std::array metricFields{
      JsonField{"meanAbsError", JsonType::NullOrNumber},
      JsonField{"rmse", JsonType::NullOrNumber},
      JsonField{"maxAbsError", JsonType::NullOrNumber},
      JsonField{"p99AbsError", JsonType::NullOrNumber},
      JsonField{"failingValues", JsonType::Unsigned},
      JsonField{"comparedValues", JsonType::Unsigned},
  };
  valid = validateObject(yyjson_obj_get(snapshot, "metrics"), metricFields,
                         std::string(path) + ".metrics");
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array semanticFields{
      JsonField{"unit", JsonType::String},
      JsonField{"meanError", JsonType::NullOrNumber},
      JsonField{"maxError", JsonType::NullOrNumber},
      JsonField{"rmse", JsonType::NullOrNumber},
      JsonField{"p99Error", JsonType::NullOrNumber},
      JsonField{"failingPixels", JsonType::Unsigned},
      JsonField{"validPixels", JsonType::Unsigned},
      JsonField{"ignoredPixels", JsonType::Unsigned},
      JsonField{"changedPixels", JsonType::Unsigned},
      JsonField{"changedBoundsValid", JsonType::Boolean},
      JsonField{"minChangedX", JsonType::Unsigned},
      JsonField{"minChangedY", JsonType::Unsigned},
      JsonField{"maxChangedX", JsonType::Unsigned},
      JsonField{"maxChangedY", JsonType::Unsigned},
      JsonField{"maxErrorX", JsonType::Unsigned},
      JsonField{"maxErrorY", JsonType::Unsigned},
      JsonField{"secondaryUnit", JsonType::String},
      JsonField{"meanSecondaryError", JsonType::NullOrNumber},
      JsonField{"maxSecondaryError", JsonType::NullOrNumber},
      JsonField{"secondaryRmse", JsonType::NullOrNumber},
      JsonField{"p99SecondaryError", JsonType::NullOrNumber},
      JsonField{"secondaryFailingPixels", JsonType::Unsigned},
      JsonField{"truePositivePixels", JsonType::Unsigned},
      JsonField{"trueNegativePixels", JsonType::Unsigned},
      JsonField{"falsePositivePixels", JsonType::Unsigned},
      JsonField{"falseNegativePixels", JsonType::Unsigned},
      JsonField{"intersectionOverUnion", JsonType::NullOrNumber},
  };
  if (yyjson_val *semantic = yyjson_obj_get(snapshot, "semanticMetrics")) {
    valid = validateObject(semantic, semanticFields,
                           std::string(path) + ".semanticMetrics");
    if (valid.hasError()) {
      return valid;
    }
  }
  return validateStringArrayValue(yyjson_obj_get(snapshot, "failedThresholds"),
                                  std::string(path) + ".failedThresholds");
}

[[nodiscard]] Result<void, std::string>
validateAutotestReportV1(yyjson_val *root) {
  auto valid = nuri::tools::core::rejectDuplicateJsonFieldsRecursively(root);
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array rootFields{
      JsonField{"schemaVersion", JsonType::Unsigned},
      JsonField{"kind", JsonType::String},
      JsonField{"baselineProfile", JsonType::String, false},
      JsonField{"generatedAtUtc", JsonType::String},
      JsonField{"command", JsonType::String},
      JsonField{"status", JsonType::String},
      JsonField{"exitCode", JsonType::Number},
      JsonField{"selection", JsonType::Object},
      JsonField{"environment", JsonType::Object},
      JsonField{"case", JsonType::Object},
      JsonField{"run", JsonType::Object},
      JsonField{"artifacts", JsonType::Object},
      JsonField{"checkpoints", JsonType::Array},
      JsonField{"metricWindows", JsonType::Array},
      JsonField{"frames", JsonType::Array},
      JsonField{"unavailableMetrics", JsonType::Array},
      JsonField{"warnings", JsonType::Array},
      JsonField{"errors", JsonType::Array},
      JsonField{"reproduceCommand", JsonType::String},
  };
  valid = validateObject(root, rootFields, "$");
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array selectionFields{
      JsonField{"requested", JsonType::String},
      JsonField{"selected", JsonType::Unsigned},
      JsonField{"attempted", JsonType::Unsigned},
      JsonField{"completed", JsonType::Unsigned},
      JsonField{"passed", JsonType::Unsigned},
      JsonField{"failed", JsonType::Unsigned},
      JsonField{"unavailable", JsonType::Unsigned},
      JsonField{"notRun", JsonType::Unsigned},
  };
  valid = validateObject(yyjson_obj_get(root, "selection"), selectionFields,
                         "$.selection");
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array environmentFields{
      JsonField{"repoRoot", JsonType::String},
      JsonField{"commitHash", JsonType::String},
      JsonField{"branchName", JsonType::String},
      JsonField{"dirty", JsonType::Boolean},
      JsonField{"osName", JsonType::String},
      JsonField{"osVersion", JsonType::String},
      JsonField{"cpuName", JsonType::String},
      JsonField{"cpuLogicalThreadCount", JsonType::Unsigned},
      JsonField{"gpuBackend", JsonType::String},
      JsonField{"gpuBackendSource", JsonType::String},
      JsonField{"gpuDeviceName", JsonType::String},
      JsonField{"gpuVendorId", JsonType::Unsigned, false},
      JsonField{"gpuDeviceId", JsonType::Unsigned, false},
      JsonField{"gpuDriverVersion", JsonType::String, false},
      JsonField{"swapchainImageCount", JsonType::Unsigned},
      JsonField{"requestedPresentMode", JsonType::String},
      JsonField{"resolvedPresentMode", JsonType::String},
      JsonField{"presentModeSource", JsonType::String},
      JsonField{"requestedWindowMode", JsonType::String},
      JsonField{"resolvedWindowMode", JsonType::String},
      JsonField{"windowVisible", JsonType::Boolean},
      JsonField{"renderGraphWorkerCount", JsonType::Unsigned},
      JsonField{"renderGraphParallelCompile", JsonType::Boolean},
      JsonField{"renderGraphParallelRecording", JsonType::Boolean},
      JsonField{"buildType", JsonType::String},
      JsonField{"cmakeToolProfile", JsonType::String},
      JsonField{"vcpkgManifestFeatures", JsonType::String},
      JsonField{"NURI_BUILD_SHARED", JsonType::Boolean},
      JsonField{"NURI_WITH_LOGGING", JsonType::Boolean},
      JsonField{"NURI_WITH_ASSERTS", JsonType::Boolean},
      JsonField{"NURI_WITH_TRACY", JsonType::Boolean},
      JsonField{"NURI_WITH_TRACY_GPU", JsonType::Boolean},
      JsonField{"NURI_WITH_TRACY_GPU_DRAW_ZONES", JsonType::Boolean},
      JsonField{"devChecks", JsonType::Boolean},
  };
  valid = validateObject(yyjson_obj_get(root, "environment"), environmentFields,
                         "$.environment");
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array caseFields{
      JsonField{"schemaVersion", JsonType::Unsigned},
      JsonField{"id", JsonType::String},
      JsonField{"suite", JsonType::String},
      JsonField{"description", JsonType::String},
      JsonField{"backend", JsonType::String},
      JsonField{"resolution", JsonType::Array},
      JsonField{"warmupFrames", JsonType::Unsigned},
      JsonField{"endFrame", JsonType::Unsigned},
      JsonField{"fixedDeltaSeconds", JsonType::NullOrNumber},
      JsonField{"presentMode", JsonType::String},
      JsonField{"windowMode", JsonType::String},
      JsonField{"authoritative", JsonType::Boolean},
  };
  yyjson_val *caseObject = yyjson_obj_get(root, "case");
  valid = validateObject(caseObject, caseFields, "$.case");
  if (valid.hasError()) {
    return valid;
  }
  yyjson_val *resolution = yyjson_obj_get(caseObject, "resolution");
  if (yyjson_arr_size(resolution) != 2u ||
      !yyjson_is_uint(yyjson_arr_get(resolution, 0u)) ||
      !yyjson_is_uint(yyjson_arr_get(resolution, 1u))) {
    return Result<void, std::string>::makeError(
        "$.case.resolution must contain two unsigned integers");
  }
  static constexpr std::array runFields{
      JsonField{"fixedDeltaSeconds", JsonType::NullOrNumber},
      JsonField{"warmupFrames", JsonType::Unsigned},
      JsonField{"endFrame", JsonType::Unsigned},
      JsonField{"renderedFrames", JsonType::Unsigned},
      JsonField{"readoutDrainFrames", JsonType::Unsigned},
      JsonField{"readoutDrainFrameLimit", JsonType::Unsigned},
      JsonField{"readoutDrainTimeoutMs", JsonType::Unsigned},
      JsonField{"readoutDrainElapsedMs", JsonType::Unsigned},
      JsonField{"requestedWindowMode", JsonType::String},
      JsonField{"resolvedWindowMode", JsonType::String},
      JsonField{"windowModeSource", JsonType::String},
      JsonField{"captureSynchronization", JsonType::String},
      JsonField{"validForComparison", JsonType::Boolean},
  };
  valid = validateObject(yyjson_obj_get(root, "run"), runFields, "$.run");
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array artifactFields{
      JsonField{"artifactDir", JsonType::String},
      JsonField{"rootHtml", JsonType::String},
      JsonField{"caseDir", JsonType::String},
      JsonField{"caseHtml", JsonType::String},
  };
  yyjson_val *artifacts = yyjson_obj_get(root, "artifacts");
  valid = validateObject(artifacts, artifactFields, "$.artifacts");
  if (valid.hasError()) {
    return valid;
  }
  for (std::string_view field :
       {"artifactDir", "rootHtml", "caseDir", "caseHtml"}) {
    valid = nuri::tools::core::validateJsonArtifactPath(artifacts, field,
                                                        "$.artifacts");
    if (valid.hasError()) {
      return valid;
    }
  }
  static constexpr std::array checkpointFields{
      JsonField{"id", JsonType::String},
      JsonField{"frame", JsonType::Unsigned},
      JsonField{"measurements", JsonType::Object},
      JsonField{"captures", JsonType::Array},
      JsonField{"readouts", JsonType::Array},
      JsonField{"assertions", JsonType::Array},
      JsonField{"motionOracle", JsonType::Object, false},
      JsonField{"qualityOracle", JsonType::Object, false},
      JsonField{"warnings", JsonType::Array},
      JsonField{"errors", JsonType::Array},
  };
  static constexpr std::array captureFields{
      JsonField{"checkpointId", JsonType::String},
      JsonField{"checkpointFrame", JsonType::Unsigned},
      JsonField{"target", JsonType::String},
      JsonField{"profile", JsonType::String},
      JsonField{"required", JsonType::Boolean},
      JsonField{"compare", JsonType::Boolean},
      JsonField{"snapshot", JsonType::Object},
  };
  static constexpr std::array readoutFields{
      JsonField{"checkpointId", JsonType::String},
      JsonField{"id", JsonType::String},
      JsonField{"type", JsonType::String},
      JsonField{"requestId", JsonType::Unsigned},
      JsonField{"requestFrame", JsonType::Unsigned},
      JsonField{"resultFrame", JsonType::Unsigned},
      JsonField{"required", JsonType::Boolean},
      JsonField{"status", JsonType::String},
      JsonField{"statusReason", JsonType::String},
      JsonField{"values", JsonType::Object},
      JsonField{"assertions", JsonType::Array},
  };
  static constexpr std::array motionOracleFields{
      JsonField{"status", JsonType::String},
      JsonField{"statusReason", JsonType::String},
      JsonField{"motionTarget", JsonType::String},
      JsonField{"motionClassTarget", JsonType::String},
      JsonField{"roi", JsonType::Object},
      JsonField{"selectedPixelCount", JsonType::Unsigned},
      JsonField{"expectedVelocityPixels", JsonType::Array},
      JsonField{"meanVelocityPixels", JsonType::Array},
      JsonField{"meanErrorPixels", JsonType::NullOrNumber},
      JsonField{"p95ErrorPixels", JsonType::NullOrNumber},
      JsonField{"maxErrorPixels", JsonType::NullOrNumber},
      JsonField{"p95ScaleErrorPixels", JsonType::NullOrNumber},
      JsonField{"maxScaleErrorPixels", JsonType::NullOrNumber},
      JsonField{"wrongSignPixelCount", JsonType::Unsigned},
      JsonField{"classCoverageAvailable", JsonType::Boolean},
      JsonField{"classSampleCount", JsonType::Unsigned},
      JsonField{"invalidClassCoverage", JsonType::NullOrNumber},
      JsonField{"staticClassCoverage", JsonType::NullOrNumber},
      JsonField{"fullClassCoverage", JsonType::NullOrNumber},
      JsonField{"p95ErrorMaxPixels", JsonType::NullOrNumber},
      JsonField{"maxErrorMaxPixels", JsonType::NullOrNumber},
      JsonField{"failedThresholds", JsonType::Array},
  };
  static constexpr std::array motionRoiFields{
      JsonField{"x", JsonType::Unsigned},
      JsonField{"y", JsonType::Unsigned},
      JsonField{"width", JsonType::Unsigned},
      JsonField{"height", JsonType::Unsigned},
  };
  static constexpr std::array qualityOracleFields{
      JsonField{"status", JsonType::String},
      JsonField{"statusReason", JsonType::String},
      JsonField{"outputTarget", JsonType::String},
      JsonField{"referencePath", JsonType::String},
      JsonField{"schemaVersion", JsonType::Unsigned},
      JsonField{"referenceVersion", JsonType::Unsigned},
      JsonField{"maskVersion", JsonType::Unsigned},
      JsonField{"lscale", JsonType::NullOrNumber},
      JsonField{"selectedPixelCount", JsonType::Unsigned},
      JsonField{"finitePixelCount", JsonType::Unsigned},
      JsonField{"nonFiniteValueCount", JsonType::Unsigned},
      JsonField{"normalizedHdrMae", JsonType::NullOrNumber},
      JsonField{"normalizedHdrRmse", JsonType::NullOrNumber},
      JsonField{"lumaSsim", JsonType::NullOrNumber},
      JsonField{"darkCollapsePixelCount", JsonType::Unsigned},
      JsonField{"darkCollapsePercent", JsonType::NullOrNumber},
      JsonField{"darkCollapseMaxComponentPixels", JsonType::Unsigned},
      JsonField{"relativeLumaEnergyDrift", JsonType::NullOrNumber},
      JsonField{"edgeAvailable", JsonType::Boolean},
      JsonField{"edgeAxis", JsonType::String},
      JsonField{"edgeProfileCount", JsonType::Unsigned},
      JsonField{"edgeUnresolvedProfileCount", JsonType::Unsigned},
      JsonField{"referenceEdgeWidth10To90", JsonType::NullOrNumber},
      JsonField{"outputEdgeWidth10To90", JsonType::NullOrNumber},
      JsonField{"edgeWidthRatio", JsonType::NullOrNumber},
      JsonField{"edgeOvershoot", JsonType::NullOrNumber},
      JsonField{"edgeUndershoot", JsonType::NullOrNumber},
      JsonField{"temporalAvailable", JsonType::Boolean},
      JsonField{"temporalSampleCount", JsonType::Unsigned},
      JsonField{"temporalError", JsonType::NullOrNumber},
      JsonField{"revealAvailable", JsonType::Boolean},
      JsonField{"revealPixelCount", JsonType::Unsigned},
      JsonField{"ghostEnergy", JsonType::NullOrNumber},
      JsonField{"recoveryRmse", JsonType::NullOrNumber},
      JsonField{"normalizedMaeMax", JsonType::NullOrNumber},
      JsonField{"normalizedRmseMax", JsonType::NullOrNumber},
      JsonField{"lumaSsimMin", JsonType::NullOrNumber},
      JsonField{"darkCollapsePercentMax", JsonType::NullOrNumber},
      JsonField{"darkCollapseComponentMaxPixels", JsonType::Unsigned},
      JsonField{"relativeLumaEnergyDriftMax", JsonType::NullOrNumber},
      JsonField{"edgeWidthRatioMin", JsonType::NullOrNumber},
      JsonField{"edgeWidthRatioMax", JsonType::NullOrNumber},
      JsonField{"edgeOvershootMax", JsonType::NullOrNumber},
      JsonField{"edgeUndershootMax", JsonType::NullOrNumber},
      JsonField{"temporalErrorMax", JsonType::NullOrNumber},
      JsonField{"ghostEnergyMax", JsonType::NullOrNumber},
      JsonField{"recoveryRmseMax", JsonType::NullOrNumber},
      JsonField{"failedThresholds", JsonType::Array},
  };
  yyjson_arr_iter checkpointIterator{};
  yyjson_arr_iter_init(yyjson_obj_get(root, "checkpoints"),
                       &checkpointIterator);
  yyjson_val *checkpoint = nullptr;
  size_t checkpointIndex = 0u;
  while ((checkpoint = yyjson_arr_iter_next(&checkpointIterator)) != nullptr) {
    const std::string path =
        "$.checkpoints[" + std::to_string(checkpointIndex++) + "]";
    valid = validateObject(checkpoint, checkpointFields, path);
    if (valid.hasError()) {
      return valid;
    }
    valid = validateNumberMap(yyjson_obj_get(checkpoint, "measurements"),
                              path + ".measurements");
    if (valid.hasError()) {
      return valid;
    }
    yyjson_arr_iter captureIterator{};
    yyjson_arr_iter_init(yyjson_obj_get(checkpoint, "captures"),
                         &captureIterator);
    yyjson_val *capture = nullptr;
    size_t captureIndex = 0u;
    while ((capture = yyjson_arr_iter_next(&captureIterator)) != nullptr) {
      const std::string capturePath =
          path + ".captures[" + std::to_string(captureIndex++) + "]";
      valid = validateObject(capture, captureFields, capturePath);
      if (valid.hasError()) {
        return valid;
      }
      valid = validateSnapshotCaptureV1(yyjson_obj_get(capture, "snapshot"),
                                        capturePath + ".snapshot");
      if (valid.hasError()) {
        return valid;
      }
    }
    yyjson_arr_iter readoutIterator{};
    yyjson_arr_iter_init(yyjson_obj_get(checkpoint, "readouts"),
                         &readoutIterator);
    yyjson_val *readout = nullptr;
    size_t readoutIndex = 0u;
    while ((readout = yyjson_arr_iter_next(&readoutIterator)) != nullptr) {
      const std::string readoutPath =
          path + ".readouts[" + std::to_string(readoutIndex++) + "]";
      valid = validateObject(readout, readoutFields, readoutPath);
      if (valid.hasError()) {
        return valid;
      }
      valid = validateNumberMap(yyjson_obj_get(readout, "values"),
                                readoutPath + ".values");
      if (valid.hasError()) {
        return valid;
      }
      valid = validateAssertionArray(yyjson_obj_get(readout, "assertions"),
                                     readoutPath + ".assertions");
      if (valid.hasError()) {
        return valid;
      }
    }
    valid = validateAssertionArray(yyjson_obj_get(checkpoint, "assertions"),
                                   path + ".assertions");
    if (valid.hasError()) {
      return valid;
    }
    if (yyjson_val *motionOracle = yyjson_obj_get(checkpoint, "motionOracle")) {
      valid = validateObject(motionOracle, motionOracleFields,
                             path + ".motionOracle");
      if (valid.hasError()) {
        return valid;
      }
      valid = validateObject(yyjson_obj_get(motionOracle, "roi"),
                             motionRoiFields, path + ".motionOracle.roi");
      if (valid.hasError()) {
        return valid;
      }
      for (std::string_view field : {std::string_view("expectedVelocityPixels"),
                                     std::string_view("meanVelocityPixels")}) {
        yyjson_val *values =
            yyjson_obj_getn(motionOracle, field.data(), field.size());
        if (!yyjson_is_arr(values) || yyjson_arr_size(values) != 2u ||
            !yyjson_is_num(yyjson_arr_get(values, 0u)) ||
            !yyjson_is_num(yyjson_arr_get(values, 1u))) {
          return Result<void, std::string>::makeError(
              path + ".motionOracle." + std::string(field) +
              " must contain two numbers");
        }
      }
      valid = validateStringArrayValue(
          yyjson_obj_get(motionOracle, "failedThresholds"),
          path + ".motionOracle.failedThresholds");
      if (valid.hasError()) {
        return valid;
      }
    }
    if (yyjson_val *qualityOracle =
            yyjson_obj_get(checkpoint, "qualityOracle")) {
      valid = validateObject(qualityOracle, qualityOracleFields,
                             path + ".qualityOracle");
      if (valid.hasError()) {
        return valid;
      }
      valid = validateStringArrayValue(
          yyjson_obj_get(qualityOracle, "failedThresholds"),
          path + ".qualityOracle.failedThresholds");
      if (valid.hasError()) {
        return valid;
      }
    }
    for (std::string_view field : {"warnings", "errors"}) {
      valid = validateStringArrayValue(
          yyjson_obj_getn(checkpoint, field.data(), field.size()),
          path + "." + std::string(field));
      if (valid.hasError()) {
        return valid;
      }
    }
  }
  static constexpr std::array windowFields{
      JsonField{"id", JsonType::String},
      JsonField{"startFrame", JsonType::Unsigned},
      JsonField{"endFrame", JsonType::Unsigned},
      JsonField{"assertions", JsonType::Array},
      JsonField{"warnings", JsonType::Array},
      JsonField{"errors", JsonType::Array},
  };
  yyjson_arr_iter windowIterator{};
  yyjson_arr_iter_init(yyjson_obj_get(root, "metricWindows"), &windowIterator);
  yyjson_val *window = nullptr;
  size_t windowIndex = 0u;
  while ((window = yyjson_arr_iter_next(&windowIterator)) != nullptr) {
    const std::string path =
        "$.metricWindows[" + std::to_string(windowIndex++) + "]";
    valid = validateObject(window, windowFields, path);
    if (valid.hasError()) {
      return valid;
    }
    valid = validateAssertionArray(yyjson_obj_get(window, "assertions"),
                                   path + ".assertions");
    if (valid.hasError()) {
      return valid;
    }
    for (std::string_view field : {"warnings", "errors"}) {
      valid = validateStringArrayValue(
          yyjson_obj_getn(window, field.data(), field.size()),
          path + "." + std::string(field));
      if (valid.hasError()) {
        return valid;
      }
    }
  }
  static constexpr std::array frameFields{
      JsonField{"frameIndex", JsonType::Unsigned},
      JsonField{"measurements", JsonType::Object},
      JsonField{"unavailableMetrics", JsonType::Array},
  };
  yyjson_arr_iter frameIterator{};
  yyjson_arr_iter_init(yyjson_obj_get(root, "frames"), &frameIterator);
  yyjson_val *frame = nullptr;
  size_t frameIndex = 0u;
  while ((frame = yyjson_arr_iter_next(&frameIterator)) != nullptr) {
    const std::string path = "$.frames[" + std::to_string(frameIndex++) + "]";
    valid = validateObject(frame, frameFields, path);
    if (valid.hasError()) {
      return valid;
    }
    valid = validateNumberMap(yyjson_obj_get(frame, "measurements"),
                              path + ".measurements");
    if (valid.hasError()) {
      return valid;
    }
    valid =
        validateStringArrayValue(yyjson_obj_get(frame, "unavailableMetrics"),
                                 path + ".unavailableMetrics");
    if (valid.hasError()) {
      return valid;
    }
  }
  for (std::string_view field : {"unavailableMetrics", "warnings", "errors"}) {
    valid = validateStringArrayValue(
        yyjson_obj_getn(root, field.data(), field.size()),
        "$." + std::string(field));
    if (valid.hasError()) {
      return valid;
    }
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<void, std::string>
validateAutotestSuiteReportV1(yyjson_val *root) {
  auto valid = nuri::tools::core::rejectDuplicateJsonFieldsRecursively(root);
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array rootFields{
      JsonField{"schemaVersion", JsonType::Unsigned},
      JsonField{"kind", JsonType::String},
      JsonField{"baselineProfile", JsonType::String},
      JsonField{"investigative", JsonType::Boolean},
      JsonField{"generatedAtUtc", JsonType::String},
      JsonField{"command", JsonType::String},
      JsonField{"suite", JsonType::String},
      JsonField{"status", JsonType::String},
      JsonField{"exitCode", JsonType::Number},
      JsonField{"selection", JsonType::Object},
      JsonField{"artifactDir", JsonType::String},
      JsonField{"children", JsonType::Array},
      JsonField{"diagnostics", JsonType::Array},
  };
  valid = validateObject(root, rootFields, "$");
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array selectionFields{
      JsonField{"requested", JsonType::String},
      JsonField{"selected", JsonType::Unsigned},
      JsonField{"attempted", JsonType::Unsigned},
      JsonField{"completed", JsonType::Unsigned},
      JsonField{"passed", JsonType::Unsigned},
      JsonField{"failed", JsonType::Unsigned},
      JsonField{"unavailable", JsonType::Unsigned},
      JsonField{"notRun", JsonType::Unsigned},
  };
  valid = validateObject(yyjson_obj_get(root, "selection"), selectionFields,
                         "$.selection");
  if (valid.hasError()) {
    return valid;
  }
  valid = nuri::tools::core::validateJsonArtifactPath(root, "artifactDir", "$");
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array childFields{
      JsonField{"id", JsonType::String},
      JsonField{"status", JsonType::String},
      JsonField{"exitCode", JsonType::Number},
      JsonField{"report", JsonType::String},
      JsonField{"html", JsonType::String},
  };
  yyjson_arr_iter iterator{};
  yyjson_arr_iter_init(yyjson_obj_get(root, "children"), &iterator);
  yyjson_val *child = nullptr;
  size_t index = 0u;
  while ((child = yyjson_arr_iter_next(&iterator)) != nullptr) {
    const std::string path = "$.children[" + std::to_string(index++) + "]";
    valid = validateObject(child, childFields, path);
    if (valid.hasError()) {
      return valid;
    }
    for (std::string_view field : {"report", "html"}) {
      valid = nuri::tools::core::validateJsonArtifactPath(child, field, path);
      if (valid.hasError()) {
        return valid;
      }
    }
  }
  return validateStringArrayValue(yyjson_obj_get(root, "diagnostics"),
                                  "$.diagnostics");
}

[[nodiscard]] nuri::tools::core::ToolOutcome
autotestToolOutcome(AutotestExitCode exitCode,
                    bool investigative = false) noexcept {
  using nuri::tools::core::ToolOutcome;
  if (investigative && exitCode == AutotestExitCode::Success) {
    return ToolOutcome::Investigative;
  }
  switch (exitCode) {
  case AutotestExitCode::Success:
    return ToolOutcome::Pass;
  case AutotestExitCode::ScenarioFailure:
    return ToolOutcome::Failure;
  case AutotestExitCode::InvalidInput:
    return ToolOutcome::Invalid;
  case AutotestExitCode::EnvironmentUnavailable:
    return ToolOutcome::EnvironmentUnavailable;
  case AutotestExitCode::RuntimeError:
    return ToolOutcome::RuntimeError;
  case AutotestExitCode::MissingBaseline:
    return ToolOutcome::MissingBaseline;
  }
  return ToolOutcome::RuntimeError;
}

[[nodiscard]] std::string resultRunId(std::string_view generatedAtUtc,
                                      std::string_view identity) {
  std::string timestamp = "19700101T000000.000Z";
  if (generatedAtUtc.size() >= 19u && generatedAtUtc[4] == '-' &&
      generatedAtUtc[7] == '-' && generatedAtUtc[10] == 'T' &&
      generatedAtUtc[13] == ':' && generatedAtUtc[16] == ':') {
    timestamp = std::string(generatedAtUtc.substr(0u, 4u)) +
                std::string(generatedAtUtc.substr(5u, 2u)) +
                std::string(generatedAtUtc.substr(8u, 2u)) + "T" +
                std::string(generatedAtUtc.substr(11u, 2u)) +
                std::string(generatedAtUtc.substr(14u, 2u)) +
                std::string(generatedAtUtc.substr(17u, 2u)) + ".000Z";
  }
  const std::string source =
      std::string(generatedAtUtc) + "\n" + std::string(identity);
  const std::string suffix = nuri::tools::core::sha256Hex(
      std::as_bytes(std::span(source.data(), source.size())));
  return timestamp + "-" + suffix.substr(0u, 16u);
}

[[nodiscard]] nuri::tools::core::ResultSelectionV2
resultSelection(const AutotestSelectionSummary &selection,
                nuri::tools::core::ToolOutcome outcome) {
  const bool investigative =
      outcome == nuri::tools::core::ToolOutcome::Investigative;
  return {.requested = selection.requested,
          .selected = selection.selected,
          .attempted = selection.attempted,
          .completed = selection.completed,
          .passed = investigative ? 0u : selection.passed,
          .warned = investigative ? selection.passed : 0u,
          .failed = selection.failed,
          .unavailable = selection.unavailable,
          .notRun = selection.notRun};
}

[[nodiscard]] std::optional<std::string>
autotestEnvironmentFingerprint(const AutotestEnvironment &environment) {
  using nuri::tools::core::FingerprintField;
  auto fingerprint = nuri::tools::core::makeSha256Fingerprint({
      FingerprintField{"os.name", environment.osName},
      FingerprintField{"os.version", environment.osVersion},
      FingerprintField{"cpu.name", environment.cpuName},
      FingerprintField{"cpu.threads",
                       std::to_string(environment.cpuLogicalThreadCount)},
      FingerprintField{"gpu.backend", environment.gpuBackend},
      FingerprintField{"gpu.backendSource", environment.gpuBackendSource},
      FingerprintField{"gpu.name", environment.gpuDeviceName},
      FingerprintField{"gpu.vendor", std::to_string(environment.gpuVendorId)},
      FingerprintField{"gpu.device", std::to_string(environment.gpuDeviceId)},
      FingerprintField{"gpu.driver", environment.gpuDriverVersion},
      FingerprintField{"present.mode", environment.resolvedPresentMode},
      FingerprintField{"window.mode", environment.resolvedWindowMode},
      FingerprintField{"window.visible",
                       environment.windowVisible ? "true" : "false"},
      FingerprintField{"build.type", environment.buildType},
      FingerprintField{"build.profile", environment.cmakeToolProfile},
      FingerprintField{"build.features", environment.vcpkgManifestFeatures},
      FingerprintField{"build.shared",
                       environment.buildShared ? "true" : "false"},
      FingerprintField{"build.asserts",
                       environment.assertsEnabled ? "true" : "false"},
      FingerprintField{"build.devChecks",
                       environment.devChecks ? "true" : "false"},
      FingerprintField{"profiling.cpu",
                       environment.tracyEnabled ? "true" : "false"},
      FingerprintField{"profiling.gpu",
                       environment.tracyGpuEnabled ? "true" : "false"},
  });
  return fingerprint.hasError()
             ? std::nullopt
             : std::optional<std::string>{std::move(fingerprint.value())};
}

[[nodiscard]] std::optional<std::string>
autotestWorkloadFingerprint(const AutotestCase &testCase) {
  using nuri::tools::core::FingerprintField;
  std::string manifestDigest;
  if (!testCase.manifestPath.empty() &&
      std::filesystem::is_regular_file(testCase.manifestPath)) {
    auto digest =
        nuri::tools::core::makeSha256FileFingerprint(testCase.manifestPath);
    if (!digest.hasError()) {
      manifestDigest = std::move(digest.value());
    }
  }
  auto fingerprint = nuri::tools::core::makeSha256Fingerprint({
      FingerprintField{"case.id", testCase.id},
      FingerprintField{"case.suite", testCase.suite},
      FingerprintField{"manifest", std::move(manifestDigest)},
      FingerprintField{"scene.kind", testCase.scene.kind},
      FingerprintField{"scene.content", testCase.scene.contentHash},
      FingerprintField{"resolution.width",
                       std::to_string(testCase.resolution[0])},
      FingerprintField{"resolution.height",
                       std::to_string(testCase.resolution[1])},
      FingerprintField{"fixedDelta",
                       std::format("{:.17g}", testCase.fixedDeltaSeconds)},
      FingerprintField{"frames.warmup", std::to_string(testCase.warmupFrames)},
      FingerprintField{"frames.end", std::to_string(testCase.endFrame)},
      FingerprintField{"checkpoint.count",
                       std::to_string(testCase.checkpoints.size())},
      FingerprintField{"metricWindow.count",
                       std::to_string(testCase.metricWindows.size())},
  });
  return fingerprint.hasError()
             ? std::nullopt
             : std::optional<std::string>{std::move(fingerprint.value())};
}

[[nodiscard]] Result<std::string, std::string>
compactJsonObject(std::string_view json) {
  std::string mutableJson(json);
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(mutableJson.data(), mutableJson.size(), 0u,
                                  nullptr, &error),
                 &yyjson_doc_free);
  yyjson_val *root = doc ? yyjson_doc_get_root(doc.get()) : nullptr;
  if (!yyjson_is_obj(root)) {
    return Result<std::string, std::string>::makeError(
        "autotest payload must be a JSON object");
  }
  size_t length = 0u;
  char *text = yyjson_val_write(root, 0u, &length);
  if (text == nullptr) {
    return Result<std::string, std::string>::makeError(
        "failed to normalize autotest payload JSON");
  }
  std::string normalized(text, length);
  std::free(text);
  return Result<std::string, std::string>::makeResult(std::move(normalized));
}

} // namespace

std::optional<std::string>
makeAutotestEnvironmentFingerprint(const AutotestEnvironment &environment) {
  return autotestEnvironmentFingerprint(environment);
}

std::optional<std::string>
makeAutotestWorkloadFingerprint(const AutotestCase &testCase) {
  return autotestWorkloadFingerprint(testCase);
}

static Result<std::string, std::string>
writeAutotestReportPayloadV1(const AutotestReport &report) {
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
  addString(doc.get(), root, "baselineProfile", report.baselineProfile);
  addString(doc.get(), root, "generatedAtUtc", report.generatedAtUtc);
  addString(doc.get(), root, "command", report.command);
  addString(doc.get(), root, "status", report.status);
  yyjson_mut_obj_add_int(doc.get(), root, "exitCode",
                         static_cast<int>(report.exitCode));
  yyjson_mut_obj_add_val(doc.get(), root, "selection",
                         makeSelectionObject(doc.get(), report.selection));
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
    if (checkpoint.motionOracle.has_value()) {
      yyjson_mut_obj_add_val(
          doc.get(), object, "motionOracle",
          makeMotionOracleObject(doc.get(), *checkpoint.motionOracle));
    }
    if (checkpoint.qualityOracle.has_value()) {
      yyjson_mut_obj_add_val(
          doc.get(), object, "qualityOracle",
          makeQualityOracleObject(doc.get(), *checkpoint.qualityOracle));
    }
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
  yyjson_write_err writeError{};
  char *json = yyjson_mut_write_opts(
      doc.get(), YYJSON_WRITE_PRETTY | YYJSON_WRITE_INF_AND_NAN_AS_NULL,
      nullptr, &length, &writeError);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "writeAutotestReportJson: failed to serialize JSON: " +
        std::string(writeError.msg != nullptr ? writeError.msg
                                              : "unknown error"));
  }
  std::string out(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(out));
}

Result<std::string, std::string>
writeAutotestReportJson(const AutotestReport &report) {
  if (report.schemaVersion != 1u || report.kind != "nuri.autotest.report") {
    return Result<std::string, std::string>::makeError(
        "writeAutotestReportJson: unsupported payload schema or kind");
  }
  if (report.status != nuri::tools::core::toolOutcomeName(
                           autotestToolOutcome(report.exitCode))) {
    return Result<std::string, std::string>::makeError(
        "writeAutotestReportJson: status and exitCode disagree");
  }
  auto payload = writeAutotestReportPayloadV1(report);
  if (payload.hasError()) {
    return Result<std::string, std::string>::makeError(payload.error());
  }
  const auto outcome = autotestToolOutcome(
      report.exitCode, report.exitCode == AutotestExitCode::Success &&
                           (!report.run.validForComparison ||
                            !report.baselineProfileCompatible));
  nuri::tools::core::ResultEnvelopeV2 envelope{};
  envelope.tool = nuri::tools::core::ResultToolV2::Autotest;
  envelope.runId = resultRunId(report.generatedAtUtc, report.testCase.id);
  envelope.status = outcome;
  envelope.exitCode = static_cast<int>(report.exitCode);
  envelope.authoritative = false;
  envelope.environmentFingerprint =
      report.environmentFingerprint.empty()
          ? autotestEnvironmentFingerprint(report.environment)
          : std::optional<std::string>{report.environmentFingerprint};
  envelope.workloadFingerprint =
      report.workloadFingerprint.empty()
          ? autotestWorkloadFingerprint(report.testCase)
          : std::optional<std::string>{report.workloadFingerprint};
  if (!report.generatedAtUtc.empty()) {
    envelope.startedAtUtc = report.generatedAtUtc;
  }
  if (!report.command.empty()) {
    envelope.command = std::vector<std::string>{report.command};
  }
  if (!report.reproduceCommand.empty()) {
    envelope.reproduceCommand = report.reproduceCommand;
  }
  envelope.selection = resultSelection(report.selection, outcome);
  envelope.profile = nuri::tools::core::ResultProfileV2{
      .id = report.baselineProfile,
      .compatible = report.baselineProfileCompatible,
      .incompatibilityReasons = report.baselineProfileIncompatibilityReasons};
  for (const std::string &warning : report.warnings) {
    envelope.diagnostics.push_back(
        {.code = "autotest.warning",
         .severity = nuri::tools::core::ResultDiagnosticSeverityV2::Warning,
         .message = warning});
  }
  for (const std::string &error : report.errors) {
    envelope.diagnostics.push_back(
        {.code = "autotest.error",
         .severity = nuri::tools::core::ResultDiagnosticSeverityV2::Error,
         .message = error});
  }
  envelope.payloadJson = std::move(payload.value());
  auto serialized = nuri::tools::core::serializeResultEnvelopeV2(envelope);
  if (serialized.hasError()) {
    return Result<std::string, std::string>::makeError(
        "writeAutotestReportJson: " + serialized.error());
  }
  return serialized;
}

Result<bool, std::string>
writeAutotestReportFile(const AutotestReport &report,
                        const std::filesystem::path &path) {
  auto json = writeAutotestReportJson(report);
  if (json.hasError()) {
    return Result<bool, std::string>::makeError(json.error());
  }
  const auto written =
      nuri::tools::core::atomicWriteTextFile(path, json.value());
  if (written.hasError()) {
    return Result<bool, std::string>::makeError("writeAutotestReportFile: " +
                                                written.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

static Result<AutotestReport, std::string>
readAutotestReportPayloadV1(std::string json) {
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
  yyjson_val *schemaVersion = yyjson_obj_get(root, "schemaVersion");
  if (!yyjson_is_uint(schemaVersion) || yyjson_get_uint(schemaVersion) != 1u) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: unsupported or missing schemaVersion");
  }
  report.schemaVersion = 1u;
  report.kind = readString(root, "kind");
  if (report.kind != "nuri.autotest.report") {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: unexpected report kind '" + report.kind + "'");
  }
  auto contract = validateAutotestReportV1(root);
  if (contract.hasError()) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: invalid v1 report: " + contract.error());
  }
  report.baselineProfile =
      readString(root, "baselineProfile", report.baselineProfile);
  report.generatedAtUtc = readString(root, "generatedAtUtc");
  report.command = readString(root, "command");
  report.status = readString(root, "status", report.status);
  const int exitCode =
      readInt(root, "exitCode", static_cast<int>(report.exitCode));
  if (exitCode < static_cast<int>(AutotestExitCode::Success) ||
      exitCode > static_cast<int>(AutotestExitCode::MissingBaseline)) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: invalid exitCode");
  }
  report.exitCode = static_cast<AutotestExitCode>(exitCode);
  readSelection(yyjson_obj_get(root, "selection"), report.selection);
  report.reproduceCommand = readString(root, "reproduceCommand");
  yyjson_val *environment = yyjson_obj_get(root, "environment");
  if (yyjson_is_obj(environment)) {
    report.environment.repoRoot = readString(environment, "repoRoot");
    report.environment.commitHash = readString(environment, "commitHash");
    report.environment.branchName = readString(environment, "branchName");
    report.environment.dirty = readBool(environment, "dirty");
    report.environment.osName = readString(environment, "osName");
    report.environment.osVersion = readString(environment, "osVersion");
    report.environment.cpuName = readString(environment, "cpuName");
    report.environment.cpuLogicalThreadCount =
        readU32(environment, "cpuLogicalThreadCount");
    report.environment.gpuBackend = readString(environment, "gpuBackend");
    report.environment.gpuBackendSource =
        readString(environment, "gpuBackendSource");
    report.environment.gpuDeviceName = readString(environment, "gpuDeviceName");
    report.environment.gpuVendorId = readU32(environment, "gpuVendorId");
    report.environment.gpuDeviceId = readU32(environment, "gpuDeviceId");
    report.environment.gpuDriverVersion =
        readString(environment, "gpuDriverVersion");
    report.environment.swapchainImageCount =
        readU32(environment, "swapchainImageCount");
    report.environment.requestedPresentMode =
        readString(environment, "requestedPresentMode");
    report.environment.resolvedPresentMode =
        readString(environment, "resolvedPresentMode");
    report.environment.presentModeSource =
        readString(environment, "presentModeSource");
    report.environment.requestedWindowMode =
        readString(environment, "requestedWindowMode");
    report.environment.resolvedWindowMode =
        readString(environment, "resolvedWindowMode");
    report.environment.windowVisible = readBool(environment, "windowVisible");
    report.environment.renderGraphWorkerCount =
        readU32(environment, "renderGraphWorkerCount");
    report.environment.renderGraphParallelCompile =
        readBool(environment, "renderGraphParallelCompile");
    report.environment.renderGraphParallelRecording =
        readBool(environment, "renderGraphParallelRecording");
    report.environment.buildType = readString(environment, "buildType");
    report.environment.cmakeToolProfile =
        readString(environment, "cmakeToolProfile");
    report.environment.vcpkgManifestFeatures =
        readString(environment, "vcpkgManifestFeatures");
    report.environment.buildShared = readBool(environment, "NURI_BUILD_SHARED");
    report.environment.loggingEnabled =
        readBool(environment, "NURI_WITH_LOGGING");
    report.environment.assertsEnabled =
        readBool(environment, "NURI_WITH_ASSERTS");
    report.environment.tracyEnabled = readBool(environment, "NURI_WITH_TRACY");
    report.environment.tracyGpuEnabled =
        readBool(environment, "NURI_WITH_TRACY_GPU");
    report.environment.tracyGpuDrawZonesEnabled =
        readBool(environment, "NURI_WITH_TRACY_GPU_DRAW_ZONES");
    report.environment.devChecks = readBool(environment, "devChecks");
  }
  yyjson_val *caseObject = yyjson_obj_get(root, "case");
  if (yyjson_is_obj(caseObject)) {
    report.testCase.schemaVersion = readU32(caseObject, "schemaVersion", 1u);
    report.testCase.id = readString(caseObject, "id");
    report.testCase.suite = readString(caseObject, "suite");
    report.testCase.description = readString(caseObject, "description");
    report.testCase.backend = readString(caseObject, "backend", "default");
    yyjson_val *resolution = yyjson_obj_get(caseObject, "resolution");
    if (yyjson_is_arr(resolution) && yyjson_arr_size(resolution) == 2u &&
        yyjson_is_uint(yyjson_arr_get(resolution, 0u)) &&
        yyjson_is_uint(yyjson_arr_get(resolution, 1u))) {
      report.testCase.resolution[0] = static_cast<uint32_t>(
          yyjson_get_uint(yyjson_arr_get(resolution, 0u)));
      report.testCase.resolution[1] = static_cast<uint32_t>(
          yyjson_get_uint(yyjson_arr_get(resolution, 1u)));
    }
    report.testCase.warmupFrames = readU32(caseObject, "warmupFrames");
    report.testCase.endFrame = readU32(caseObject, "endFrame");
    report.testCase.fixedDeltaSeconds = readDouble(
        caseObject, "fixedDeltaSeconds", report.testCase.fixedDeltaSeconds);
    report.testCase.presentMode =
        readString(caseObject, "presentMode", "immediate");
    report.testCase.windowMode =
        readString(caseObject, "windowMode", "visible");
    report.testCase.authoritative = readBool(caseObject, "authoritative");
  }
  yyjson_val *run = yyjson_obj_get(root, "run");
  if (yyjson_is_obj(run)) {
    report.run.fixedDeltaSeconds =
        readDouble(run, "fixedDeltaSeconds", report.run.fixedDeltaSeconds);
    report.run.warmupFrames = readU32(run, "warmupFrames");
    report.run.endFrame = readU32(run, "endFrame");
    report.run.renderedFrames = readU32(run, "renderedFrames");
    report.run.readoutDrainFrames = readU32(run, "readoutDrainFrames");
    report.run.readoutDrainFrameLimit = readU32(run, "readoutDrainFrameLimit");
    report.run.readoutDrainTimeoutMs =
        readU32(run, "readoutDrainTimeoutMs", report.run.readoutDrainTimeoutMs);
    report.run.readoutDrainElapsedMs = readU32(run, "readoutDrainElapsedMs");
    report.run.requestedWindowMode = readString(run, "requestedWindowMode");
    report.run.resolvedWindowMode =
        readString(run, "resolvedWindowMode", report.run.resolvedWindowMode);
    report.run.windowModeSource =
        readString(run, "windowModeSource", report.run.windowModeSource);
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
      yyjson_val *captures = yyjson_obj_get(checkpointValue, "captures");
      if (yyjson_is_arr(captures)) {
        yyjson_arr_iter captureIter;
        yyjson_arr_iter_init(captures, &captureIter);
        yyjson_val *captureValue = nullptr;
        while ((captureValue = yyjson_arr_iter_next(&captureIter)) != nullptr) {
          if (!yyjson_is_obj(captureValue)) {
            continue;
          }
          AutotestCaptureReport capture{};
          capture.checkpointId = readString(captureValue, "checkpointId");
          capture.checkpointFrame = readU32(captureValue, "checkpointFrame");
          capture.target = readString(captureValue, "target");
          capture.profile = readString(captureValue, "profile");
          capture.required = readBool(captureValue, "required", true);
          capture.compare = readBool(captureValue, "compare", true);
          yyjson_val *snapshot = yyjson_obj_get(captureValue, "snapshot");
          if (yyjson_is_obj(snapshot)) {
            capture.snapshot = readSnapshotCapture(snapshot);
          }
          checkpoint.captures.push_back(std::move(capture));
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
      yyjson_val *motionOracleValue =
          yyjson_obj_get(checkpointValue, "motionOracle");
      if (yyjson_is_obj(motionOracleValue)) {
        AutotestMotionOracleReport motionOracle{};
        motionOracle.status = readString(motionOracleValue, "status");
        motionOracle.statusReason =
            readString(motionOracleValue, "statusReason");
        motionOracle.motionTarget =
            readString(motionOracleValue, "motionTarget");
        motionOracle.motionClassTarget =
            readString(motionOracleValue, "motionClassTarget");
        yyjson_val *roi = yyjson_obj_get(motionOracleValue, "roi");
        if (yyjson_is_obj(roi)) {
          motionOracle.roi.x = readU32(roi, "x");
          motionOracle.roi.y = readU32(roi, "y");
          motionOracle.roi.width = readU32(roi, "width");
          motionOracle.roi.height = readU32(roi, "height");
        }
        motionOracle.selectedPixelCount =
            readU32(motionOracleValue, "selectedPixelCount");
        const auto readVec2Array = [](yyjson_val *object, const char *key,
                                      std::array<double, 2> &out) {
          yyjson_val *values = yyjson_obj_get(object, key);
          if (yyjson_is_arr(values) && yyjson_arr_size(values) == 2u &&
              yyjson_is_num(yyjson_arr_get(values, 0u)) &&
              yyjson_is_num(yyjson_arr_get(values, 1u))) {
            out[0] = yyjson_get_num(yyjson_arr_get(values, 0u));
            out[1] = yyjson_get_num(yyjson_arr_get(values, 1u));
          }
        };
        readVec2Array(motionOracleValue, "expectedVelocityPixels",
                      motionOracle.expectedVelocityPixels);
        readVec2Array(motionOracleValue, "meanVelocityPixels",
                      motionOracle.meanVelocityPixels);
        motionOracle.meanErrorPixels =
            readDouble(motionOracleValue, "meanErrorPixels");
        motionOracle.p95ErrorPixels =
            readDouble(motionOracleValue, "p95ErrorPixels");
        motionOracle.maxErrorPixels =
            readDouble(motionOracleValue, "maxErrorPixels");
        motionOracle.p95ScaleErrorPixels =
            readDouble(motionOracleValue, "p95ScaleErrorPixels");
        motionOracle.maxScaleErrorPixels =
            readDouble(motionOracleValue, "maxScaleErrorPixels");
        motionOracle.wrongSignPixelCount =
            readU32(motionOracleValue, "wrongSignPixelCount");
        motionOracle.classCoverageAvailable =
            readBool(motionOracleValue, "classCoverageAvailable");
        motionOracle.classSampleCount =
            readU32(motionOracleValue, "classSampleCount");
        motionOracle.invalidClassCoverage =
            readDouble(motionOracleValue, "invalidClassCoverage");
        motionOracle.staticClassCoverage =
            readDouble(motionOracleValue, "staticClassCoverage");
        motionOracle.fullClassCoverage =
            readDouble(motionOracleValue, "fullClassCoverage");
        motionOracle.p95ErrorMaxPixels =
            readDouble(motionOracleValue, "p95ErrorMaxPixels");
        motionOracle.maxErrorMaxPixels =
            readDouble(motionOracleValue, "maxErrorMaxPixels");
        readStringArray(motionOracleValue, "failedThresholds",
                        motionOracle.failedThresholds);
        checkpoint.motionOracle = std::move(motionOracle);
      }
      yyjson_val *qualityOracleValue =
          yyjson_obj_get(checkpointValue, "qualityOracle");
      if (yyjson_is_obj(qualityOracleValue)) {
        AutotestQualityOracleReport quality{};
        quality.status = readString(qualityOracleValue, "status");
        quality.statusReason = readString(qualityOracleValue, "statusReason");
        quality.outputTarget = readString(qualityOracleValue, "outputTarget");
        quality.referencePath = readString(qualityOracleValue, "referencePath");
        quality.schemaVersion =
            readU32(qualityOracleValue, "schemaVersion", 1u);
        quality.referenceVersion =
            readU32(qualityOracleValue, "referenceVersion");
        quality.maskVersion = readU32(qualityOracleValue, "maskVersion");
        quality.lscale = readDouble(qualityOracleValue, "lscale");
        quality.selectedPixelCount =
            readU32(qualityOracleValue, "selectedPixelCount");
        quality.finitePixelCount =
            readU32(qualityOracleValue, "finitePixelCount");
        quality.nonFiniteValueCount =
            readU64(qualityOracleValue, "nonFiniteValueCount");
        quality.normalizedHdrMae =
            readDouble(qualityOracleValue, "normalizedHdrMae");
        quality.normalizedHdrRmse =
            readDouble(qualityOracleValue, "normalizedHdrRmse");
        quality.lumaSsim = readDouble(qualityOracleValue, "lumaSsim");
        quality.darkCollapsePixelCount =
            readU32(qualityOracleValue, "darkCollapsePixelCount");
        quality.darkCollapsePercent =
            readDouble(qualityOracleValue, "darkCollapsePercent");
        quality.darkCollapseMaxComponentPixels =
            readU32(qualityOracleValue, "darkCollapseMaxComponentPixels");
        quality.relativeLumaEnergyDrift =
            readDouble(qualityOracleValue, "relativeLumaEnergyDrift");
        quality.edgeAvailable = readBool(qualityOracleValue, "edgeAvailable");
        quality.edgeAxis = readString(qualityOracleValue, "edgeAxis");
        quality.edgeProfileCount =
            readU32(qualityOracleValue, "edgeProfileCount");
        quality.edgeUnresolvedProfileCount =
            readU32(qualityOracleValue, "edgeUnresolvedProfileCount");
        quality.referenceEdgeWidth10To90 =
            readDouble(qualityOracleValue, "referenceEdgeWidth10To90");
        quality.outputEdgeWidth10To90 =
            readDouble(qualityOracleValue, "outputEdgeWidth10To90");
        quality.edgeWidthRatio =
            readDouble(qualityOracleValue, "edgeWidthRatio");
        quality.edgeOvershoot = readDouble(qualityOracleValue, "edgeOvershoot");
        quality.edgeUndershoot =
            readDouble(qualityOracleValue, "edgeUndershoot");
        quality.temporalAvailable =
            readBool(qualityOracleValue, "temporalAvailable");
        quality.temporalSampleCount =
            readU32(qualityOracleValue, "temporalSampleCount");
        quality.temporalError = readDouble(qualityOracleValue, "temporalError");
        quality.revealAvailable =
            readBool(qualityOracleValue, "revealAvailable");
        quality.revealPixelCount =
            readU32(qualityOracleValue, "revealPixelCount");
        quality.ghostEnergy = readDouble(qualityOracleValue, "ghostEnergy");
        quality.recoveryRmse = readDouble(qualityOracleValue, "recoveryRmse");
        quality.budgets.normalizedMaeMax =
            readDouble(qualityOracleValue, "normalizedMaeMax");
        quality.budgets.normalizedRmseMax =
            readDouble(qualityOracleValue, "normalizedRmseMax");
        quality.budgets.lumaSsimMin =
            readDouble(qualityOracleValue, "lumaSsimMin");
        quality.budgets.darkCollapsePercentMax =
            readDouble(qualityOracleValue, "darkCollapsePercentMax");
        quality.budgets.darkCollapseComponentMaxPixels =
            readU32(qualityOracleValue, "darkCollapseComponentMaxPixels");
        quality.budgets.relativeLumaEnergyDriftMax =
            readDouble(qualityOracleValue, "relativeLumaEnergyDriftMax");
        quality.budgets.edgeWidthRatioMin =
            readDouble(qualityOracleValue, "edgeWidthRatioMin");
        quality.budgets.edgeWidthRatioMax =
            readDouble(qualityOracleValue, "edgeWidthRatioMax");
        quality.budgets.edgeOvershootMax =
            readDouble(qualityOracleValue, "edgeOvershootMax");
        quality.budgets.edgeUndershootMax =
            readDouble(qualityOracleValue, "edgeUndershootMax");
        quality.budgets.temporalErrorMax =
            readDouble(qualityOracleValue, "temporalErrorMax");
        quality.budgets.ghostEnergyMax =
            readDouble(qualityOracleValue, "ghostEnergyMax");
        quality.budgets.recoveryRmseMax =
            readDouble(qualityOracleValue, "recoveryRmseMax");
        readStringArray(qualityOracleValue, "failedThresholds",
                        quality.failedThresholds);
        checkpoint.qualityOracle = std::move(quality);
      }
      readStringArray(checkpointValue, "warnings", checkpoint.warnings);
      readStringArray(checkpointValue, "errors", checkpoint.errors);
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
      readStringArray(windowValue, "warnings", window.warnings);
      readStringArray(windowValue, "errors", window.errors);
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
  readStringArray(root, "unavailableMetrics", report.unavailableMetrics);
  readStringArray(root, "warnings", report.warnings);
  readStringArray(root, "errors", report.errors);
  return Result<AutotestReport, std::string>::makeResult(std::move(report));
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
  std::string probeText = json;
  yyjson_read_err error{};
  JsonDocPtr probe(
      yyjson_read_opts(probeText.data(), probeText.size(), 0u, nullptr, &error),
      &yyjson_doc_free);
  yyjson_val *root = probe ? yyjson_doc_get_root(probe.get()) : nullptr;
  yyjson_val *schema =
      yyjson_is_obj(root) ? yyjson_obj_get(root, "schemaVersion") : nullptr;
  if (!yyjson_is_uint(schema)) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: unsupported or missing schemaVersion");
  }
  if (yyjson_get_uint(schema) == 1u) {
    return readAutotestReportPayloadV1(std::move(json));
  }
  if (yyjson_get_uint(schema) != 2u) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: unsupported schemaVersion");
  }
  auto envelope = nuri::tools::core::readResultEnvelopeV2(json);
  if (envelope.hasError() ||
      envelope.value().tool != nuri::tools::core::ResultToolV2::Autotest) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: " +
        (envelope.hasError() ? envelope.error()
                             : "v2 envelope tool must be autotest"));
  }
  auto report = readAutotestReportPayloadV1(envelope.value().payloadJson);
  if (report.hasError()) {
    return report;
  }
  if (!envelope.value().profile.has_value()) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: v2 envelope profile is missing");
  }
  report.value().baselineProfileCompatible =
      envelope.value().profile->compatible;
  report.value().baselineProfileIncompatibilityReasons =
      envelope.value().profile->incompatibilityReasons;
  report.value().environmentFingerprint =
      envelope.value().environmentFingerprint.value_or("");
  report.value().workloadFingerprint =
      envelope.value().workloadFingerprint.value_or("");
  const auto expectedOutcome = autotestToolOutcome(
      report.value().exitCode,
      report.value().exitCode == AutotestExitCode::Success &&
          (!report.value().run.validForComparison ||
           !report.value().baselineProfileCompatible));
  const auto expectedSelection =
      resultSelection(report.value().selection, expectedOutcome);
  const auto &actualSelection = envelope.value().selection;
  const bool selectionMatches =
      actualSelection.requested == expectedSelection.requested &&
      actualSelection.selected == expectedSelection.selected &&
      actualSelection.attempted == expectedSelection.attempted &&
      actualSelection.completed == expectedSelection.completed &&
      actualSelection.passed == expectedSelection.passed &&
      actualSelection.warned == expectedSelection.warned &&
      actualSelection.failed == expectedSelection.failed &&
      actualSelection.skipped == expectedSelection.skipped &&
      actualSelection.unavailable == expectedSelection.unavailable &&
      actualSelection.notRun == expectedSelection.notRun;
  auto canonicalPayload = writeAutotestReportPayloadV1(report.value());
  auto actualPayload = compactJsonObject(envelope.value().payloadJson);
  auto expectedPayload = canonicalPayload.hasError()
                             ? Result<std::string, std::string>::makeError(
                                   canonicalPayload.error())
                             : compactJsonObject(canonicalPayload.value());
  auto canonicalEnvelope = writeAutotestReportJson(report.value());
  auto actualEnvelope = compactJsonObject(json);
  auto expectedEnvelope = canonicalEnvelope.hasError()
                              ? Result<std::string, std::string>::makeError(
                                    canonicalEnvelope.error())
                              : compactJsonObject(canonicalEnvelope.value());
  const std::optional<std::string> expectedStartedAt =
      report.value().generatedAtUtc.empty()
          ? std::optional<std::string>{}
          : std::optional<std::string>{report.value().generatedAtUtc};
  const std::optional<std::string> expectedReproduce =
      report.value().reproduceCommand.empty()
          ? std::optional<std::string>{}
          : std::optional<std::string>{report.value().reproduceCommand};
  const std::optional<std::vector<std::string>> expectedCommand =
      report.value().command.empty()
          ? std::optional<std::vector<std::string>>{}
          : std::optional<std::vector<std::string>>{
                std::vector<std::string>{report.value().command}};
  const bool profileMatches =
      envelope.value().profile.has_value() &&
      envelope.value().profile->id == report.value().baselineProfile &&
      envelope.value().profile->compatible ==
          report.value().baselineProfileCompatible &&
      envelope.value().profile->incompatibilityReasons ==
          report.value().baselineProfileIncompatibilityReasons;
  if (report.value().status !=
          nuri::tools::core::toolOutcomeName(
              autotestToolOutcome(report.value().exitCode)) ||
      envelope.value().status != expectedOutcome ||
      envelope.value().exitCode != static_cast<int>(report.value().exitCode) ||
      envelope.value().authoritative || !profileMatches || !selectionMatches ||
      envelope.value().startedAtUtc != expectedStartedAt ||
      envelope.value().reproduceCommand != expectedReproduce ||
      envelope.value().command != expectedCommand ||
      canonicalPayload.hasError() || actualPayload.hasError() ||
      expectedPayload.hasError() ||
      actualPayload.value() != expectedPayload.value() ||
      canonicalEnvelope.hasError() || actualEnvelope.hasError() ||
      expectedEnvelope.hasError() ||
      actualEnvelope.value() != expectedEnvelope.value()) {
    return Result<AutotestReport, std::string>::makeError(
        "readAutotestReportFile: v2 envelope and autotest payload disagree");
  }
  return report;
}

static Result<std::string, std::string>
writeAutotestSuiteReportPayloadV1(const AutotestSuiteReport &report) {
  JsonMutDocPtr doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "writeAutotestSuiteReportJson: failed to allocate JSON document");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion",
                          report.schemaVersion);
  addString(doc.get(), root, "kind", report.kind);
  addString(doc.get(), root, "baselineProfile", report.baselineProfile);
  yyjson_mut_obj_add_bool(doc.get(), root, "investigative",
                          report.investigative);
  addString(doc.get(), root, "generatedAtUtc", report.generatedAtUtc);
  addString(doc.get(), root, "command", report.command);
  addString(doc.get(), root, "suite", report.suite);
  addString(doc.get(), root, "status", report.status);
  yyjson_mut_obj_add_int(doc.get(), root, "exitCode",
                         static_cast<int>(report.exitCode));
  yyjson_mut_obj_add_val(doc.get(), root, "selection",
                         makeSelectionObject(doc.get(), report.selection));
  addPath(doc.get(), root, "artifactDir", report.artifactDir);
  yyjson_mut_val *children = yyjson_mut_arr(doc.get());
  for (const AutotestSuiteChildReport &child : report.children) {
    yyjson_mut_val *object = yyjson_mut_obj(doc.get());
    addString(doc.get(), object, "id", child.id);
    addString(doc.get(), object, "status", child.status);
    yyjson_mut_obj_add_int(doc.get(), object, "exitCode",
                           static_cast<int>(child.exitCode));
    addPath(doc.get(), object, "report", child.report);
    addPath(doc.get(), object, "html", child.html);
    yyjson_mut_arr_add_val(children, object);
  }
  yyjson_mut_obj_add_val(doc.get(), root, "children", children);
  yyjson_mut_obj_add_val(doc.get(), root, "diagnostics",
                         makeStringArray(doc.get(), report.diagnostics));

  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "writeAutotestSuiteReportJson: failed to serialize JSON");
  }
  std::string out(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(out));
}

static Result<AutotestSuiteReport, std::string>
readAutotestSuiteReportPayloadV1(std::string json, bool directLegacy) {
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
                 &yyjson_doc_free);
  if (!doc) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: JSON parse failed at byte " +
        std::to_string(error.pos) + ": " + error.msg);
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  if (!yyjson_is_obj(root)) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: root must be an object");
  }
  auto contract = validateAutotestSuiteReportV1(root);
  if (contract.hasError()) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: invalid v1 suite report: " +
        contract.error());
  }
  if (readU32(root, "schemaVersion") != 1u ||
      readString(root, "kind") != "nuri.autotest.suite_report") {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: unsupported payload schema or kind");
  }
  AutotestSuiteReport report{};
  report.baselineProfile =
      readString(root, "baselineProfile", report.baselineProfile);
  report.investigative = readBool(root, "investigative") || directLegacy;
  report.generatedAtUtc = readString(root, "generatedAtUtc");
  report.command = readString(root, "command");
  report.suite = readString(root, "suite");
  report.status = readString(root, "status");
  const int exitCode = readInt(root, "exitCode", -1);
  if (exitCode < static_cast<int>(AutotestExitCode::Success) ||
      exitCode > static_cast<int>(AutotestExitCode::MissingBaseline)) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: invalid exitCode");
  }
  report.exitCode = static_cast<AutotestExitCode>(exitCode);
  readSelection(yyjson_obj_get(root, "selection"), report.selection);
  report.artifactDir = readString(root, "artifactDir");
  yyjson_arr_iter iterator{};
  yyjson_arr_iter_init(yyjson_obj_get(root, "children"), &iterator);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
    const int childExit = readInt(entry, "exitCode", -1);
    if (childExit < static_cast<int>(AutotestExitCode::Success) ||
        childExit > static_cast<int>(AutotestExitCode::MissingBaseline)) {
      return Result<AutotestSuiteReport, std::string>::makeError(
          "readAutotestSuiteReportFile: invalid child exitCode");
    }
    report.children.push_back(
        {.id = readString(entry, "id"),
         .status = readString(entry, "status"),
         .exitCode = static_cast<AutotestExitCode>(childExit),
         .report = readString(entry, "report"),
         .html = readString(entry, "html")});
  }
  readStringArray(root, "diagnostics", report.diagnostics);
  if (report.status != nuri::tools::core::toolOutcomeName(
                           autotestToolOutcome(report.exitCode))) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: status and exitCode disagree");
  }
  for (const AutotestSuiteChildReport &child : report.children) {
    if (child.status != nuri::tools::core::toolOutcomeName(
                            autotestToolOutcome(child.exitCode))) {
      return Result<AutotestSuiteReport, std::string>::makeError(
          "readAutotestSuiteReportFile: child status and exitCode disagree");
    }
  }
  return Result<AutotestSuiteReport, std::string>::makeResult(
      std::move(report));
}

Result<std::string, std::string>
writeAutotestSuiteReportJson(const AutotestSuiteReport &report) {
  if (report.schemaVersion != 1u ||
      report.kind != "nuri.autotest.suite_report") {
    return Result<std::string, std::string>::makeError(
        "writeAutotestSuiteReportJson: unsupported payload schema or kind");
  }
  if (report.status != nuri::tools::core::toolOutcomeName(
                           autotestToolOutcome(report.exitCode))) {
    return Result<std::string, std::string>::makeError(
        "writeAutotestSuiteReportJson: status and exitCode disagree");
  }
  for (const AutotestSuiteChildReport &child : report.children) {
    if (child.status != nuri::tools::core::toolOutcomeName(
                            autotestToolOutcome(child.exitCode))) {
      return Result<std::string, std::string>::makeError(
          "writeAutotestSuiteReportJson: child status and exitCode disagree");
    }
  }
  auto payload = writeAutotestSuiteReportPayloadV1(report);
  if (payload.hasError()) {
    return Result<std::string, std::string>::makeError(payload.error());
  }
  const auto outcome = autotestToolOutcome(
      report.exitCode,
      report.investigative || !report.baselineProfileCompatible);
  nuri::tools::core::ResultEnvelopeV2 envelope{};
  envelope.tool = nuri::tools::core::ResultToolV2::Autotest;
  envelope.runId = resultRunId(report.generatedAtUtc, report.suite);
  envelope.status = outcome;
  envelope.exitCode = static_cast<int>(report.exitCode);
  envelope.authoritative = false;
  if (!report.environmentFingerprint.empty()) {
    envelope.environmentFingerprint = report.environmentFingerprint;
  }
  if (!report.workloadFingerprint.empty()) {
    envelope.workloadFingerprint = report.workloadFingerprint;
  }
  if (!report.generatedAtUtc.empty()) {
    envelope.startedAtUtc = report.generatedAtUtc;
  }
  if (!report.command.empty()) {
    envelope.command = std::vector<std::string>{report.command};
  }
  envelope.selection = resultSelection(report.selection, outcome);
  envelope.profile = nuri::tools::core::ResultProfileV2{
      .id = report.baselineProfile,
      .compatible = report.baselineProfileCompatible,
      .incompatibilityReasons = report.baselineProfileIncompatibilityReasons};
  for (const std::string &diagnostic : report.diagnostics) {
    envelope.diagnostics.push_back(
        {.code = "autotest.suite.summary",
         .severity = report.exitCode == AutotestExitCode::Success
                         ? nuri::tools::core::ResultDiagnosticSeverityV2::Info
                         : nuri::tools::core::ResultDiagnosticSeverityV2::Error,
         .message = diagnostic});
  }
  for (const AutotestSuiteChildReport &child : report.children) {
    nuri::tools::core::ResultChildV2 resultChild{
        .id = child.id,
        .status = child.status,
        .exitCode = static_cast<int>(child.exitCode)};
    if (!child.report.empty()) {
      resultChild.result = child.report;
    }
    envelope.children.push_back(std::move(resultChild));
  }
  envelope.payloadJson = std::move(payload.value());
  auto serialized = nuri::tools::core::serializeResultEnvelopeV2(envelope);
  if (serialized.hasError()) {
    return Result<std::string, std::string>::makeError(
        "writeAutotestSuiteReportJson: " + serialized.error());
  }
  return serialized;
}

Result<bool, std::string>
writeAutotestSuiteReportFile(const AutotestSuiteReport &report,
                             const std::filesystem::path &path) {
  auto json = writeAutotestSuiteReportJson(report);
  if (json.hasError()) {
    return Result<bool, std::string>::makeError(json.error());
  }
  auto written = nuri::tools::core::atomicWriteTextFile(path, json.value());
  if (written.hasError()) {
    return Result<bool, std::string>::makeError(
        "writeAutotestSuiteReportFile: " + written.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<AutotestSuiteReport, std::string>
readAutotestSuiteReportFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: failed to open " + path.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
                 &yyjson_doc_free);
  yyjson_val *root = doc ? yyjson_doc_get_root(doc.get()) : nullptr;
  yyjson_val *schema =
      yyjson_is_obj(root) ? yyjson_obj_get(root, "schemaVersion") : nullptr;
  if (!yyjson_is_uint(schema)) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: unsupported or missing schemaVersion");
  }
  if (yyjson_get_uint(schema) == 1u) {
    return readAutotestSuiteReportPayloadV1(std::move(json), true);
  }
  if (yyjson_get_uint(schema) != 2u) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: unsupported schemaVersion");
  }
  auto envelope = nuri::tools::core::readResultEnvelopeV2(json);
  if (envelope.hasError() ||
      envelope.value().tool != nuri::tools::core::ResultToolV2::Autotest) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: " +
        (envelope.hasError() ? envelope.error()
                             : "v2 envelope tool must be autotest"));
  }
  auto report =
      readAutotestSuiteReportPayloadV1(envelope.value().payloadJson, false);
  if (report.hasError()) {
    return report;
  }
  if (!envelope.value().profile.has_value()) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: v2 envelope profile is missing");
  }
  report.value().baselineProfileCompatible =
      envelope.value().profile->compatible;
  report.value().baselineProfileIncompatibilityReasons =
      envelope.value().profile->incompatibilityReasons;
  report.value().environmentFingerprint =
      envelope.value().environmentFingerprint.value_or("");
  report.value().workloadFingerprint =
      envelope.value().workloadFingerprint.value_or("");
  const auto expectedOutcome = autotestToolOutcome(
      report.value().exitCode, report.value().investigative ||
                                   !report.value().baselineProfileCompatible);
  const auto expectedSelection =
      resultSelection(report.value().selection, expectedOutcome);
  const auto &actualSelection = envelope.value().selection;
  const bool selectionMatches =
      actualSelection.requested == expectedSelection.requested &&
      actualSelection.selected == expectedSelection.selected &&
      actualSelection.attempted == expectedSelection.attempted &&
      actualSelection.completed == expectedSelection.completed &&
      actualSelection.passed == expectedSelection.passed &&
      actualSelection.warned == expectedSelection.warned &&
      actualSelection.failed == expectedSelection.failed &&
      actualSelection.unavailable == expectedSelection.unavailable &&
      actualSelection.notRun == expectedSelection.notRun;
  bool childrenMatch =
      envelope.value().children.size() == report.value().children.size();
  for (size_t index = 0u;
       childrenMatch && index < report.value().children.size(); ++index) {
    const auto &actual = envelope.value().children[index];
    const auto &expected = report.value().children[index];
    childrenMatch =
        actual.id == expected.id && actual.status == expected.status &&
        actual.exitCode == static_cast<int>(expected.exitCode) &&
        actual.result ==
            (expected.report.empty()
                 ? std::optional<std::filesystem::path>{}
                 : std::optional<std::filesystem::path>{expected.report});
  }
  if (envelope.value().status != expectedOutcome ||
      envelope.value().exitCode != static_cast<int>(report.value().exitCode) ||
      envelope.value().authoritative ||
      envelope.value().profile->id != report.value().baselineProfile ||
      !selectionMatches || !childrenMatch) {
    return Result<AutotestSuiteReport, std::string>::makeError(
        "readAutotestSuiteReportFile: v2 envelope and suite payload disagree");
  }
  return report;
}

namespace {

struct AutotestHtmlCounts {
  size_t captures = 0u;
  size_t readouts = 0u;
  size_t assertions = 0u;
  size_t passed = 0u;
  size_t warned = 0u;
  size_t failed = 0u;
  size_t unavailable = 0u;
};

void countAssertion(const AutotestAssertionResult &assertion,
                    AutotestHtmlCounts &counts) {
  ++counts.assertions;
  const std::string status = autotestAssertionStatusName(assertion.status);
  if (status == "fail" || status == "invalid") {
    ++counts.failed;
  } else if (status == "warn") {
    ++counts.warned;
  } else if (status == "unavailable") {
    ++counts.unavailable;
  } else {
    ++counts.passed;
  }
}

[[nodiscard]] AutotestHtmlCounts
autotestHtmlCounts(const AutotestReport &report) {
  AutotestHtmlCounts counts{};
  for (const AutotestCheckpointReport &checkpoint : report.checkpoints) {
    counts.captures += checkpoint.captures.size();
    counts.readouts += checkpoint.readouts.size();
    for (const AutotestAssertionResult &assertion : checkpoint.assertions) {
      countAssertion(assertion, counts);
    }
    for (const AutotestReadoutReport &readout : checkpoint.readouts) {
      for (const AutotestAssertionResult &assertion : readout.assertions) {
        countAssertion(assertion, counts);
      }
    }
  }
  for (const AutotestMetricWindowReport &window : report.metricWindows) {
    for (const AutotestAssertionResult &assertion : window.assertions) {
      countAssertion(assertion, counts);
    }
  }
  return counts;
}

[[nodiscard]] std::string_view
checkpointStatus(const AutotestCheckpointReport &checkpoint) {
  if (!checkpoint.errors.empty()) {
    return "error";
  }
  bool warned = !checkpoint.warnings.empty();
  if (checkpoint.motionOracle.has_value()) {
    if (checkpoint.motionOracle->status == "error") {
      return "error";
    }
    if (checkpoint.motionOracle->status == "fail") {
      return "fail";
    }
    warned = warned || checkpoint.motionOracle->status == "unavailable" ||
             checkpoint.motionOracle->status == "not_run";
  }
  if (checkpoint.qualityOracle.has_value()) {
    if (checkpoint.qualityOracle->status == "error") {
      return "error";
    }
    if (checkpoint.qualityOracle->status == "fail") {
      return "fail";
    }
    warned = warned || checkpoint.qualityOracle->status == "unavailable" ||
             checkpoint.qualityOracle->status == "not_run";
  }
  for (const AutotestAssertionResult &assertion : checkpoint.assertions) {
    const std::string status = autotestAssertionStatusName(assertion.status);
    if (status == "fail" || status == "invalid") {
      return "fail";
    }
    warned = warned || status == "warn";
  }
  for (const AutotestCaptureReport &capture : checkpoint.captures) {
    if (capture.snapshot.status == "fail" ||
        capture.snapshot.status == "runtime_error" ||
        capture.snapshot.status == "invalid") {
      return "fail";
    }
    warned = warned || capture.snapshot.status == "investigative" ||
             capture.snapshot.status == "missing_baseline" ||
             capture.snapshot.status == "missing_capture_point" ||
             capture.snapshot.status == "environment_unavailable";
  }
  for (const AutotestReadoutReport &readout : checkpoint.readouts) {
    if (readout.status == "fail" || readout.status == "invalid" ||
        readout.status == "error") {
      return "fail";
    }
    warned = warned || readout.status == "warn";
  }
  return warned ? "warn" : "pass";
}

void writeStatusBadge(std::ostringstream &out, std::string_view status) {
  out << "<span class=\"status-badge status-" << htmlEscape(status) << "\">"
      << htmlEscape(readableStatus(status)) << "</span>";
}

void writeMessages(std::ostringstream &out, std::string_view title,
                   const std::vector<std::string> &messages,
                   std::string_view tone, std::string_view role) {
  if (messages.empty()) {
    return;
  }
  out << "<section class=\"diagnostics tone-" << tone << "\" role=\"" << role
      << "\"><h3>" << htmlEscape(title) << "</h3><ul>";
  for (const std::string &message : messages) {
    out << "<li>" << htmlEscape(message) << "</li>";
  }
  out << "</ul></section>";
}

} // namespace

Result<std::string, std::string>
writeAutotestHtmlReport(const AutotestReport &report) {
  const AutotestHtmlCounts counts = autotestHtmlCounts(report);
  const std::string_view overallStatus =
      report.status.empty() ? std::string_view{"invalid"} : report.status;
  std::ostringstream out;
  nuri::tools::core::beginHtmlReport(out, "Nuri Autotest " + report.testCase.id,
                                     R"CSS(
.checkpoint{border-left-width:5px}.checkpoint.status-pass{border-left-color:var(--pass)}.checkpoint.status-warn{border-left-color:var(--warn)}.checkpoint.status-fail,.checkpoint.status-error{border-left-color:var(--fail)}
.checkpoint-meta{display:flex;flex-wrap:wrap;gap:.45rem .8rem;margin:.45rem 0 0;color:var(--muted);font-size:.82rem}
.checkpoint-content{display:grid;gap:1rem}.subsection{min-width:0;margin-top:1rem}.subsection-heading{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;margin-bottom:.55rem}
.evidence-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,20rem),1fr));gap:.75rem}
.evidence-card{min-width:0;padding:.85rem;border:1px solid var(--line);border-radius:var(--radius-small);background:var(--surface-soft)}
.evidence-card .card-heading{align-items:flex-start}.evidence-card p{margin:.45rem 0;color:var(--muted);overflow-wrap:anywhere}
.evidence-images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.5rem;margin-top:.7rem}.evidence-images figure{min-width:0;margin:0}.evidence-images figcaption{margin-bottom:.25rem;color:var(--muted);font-size:.75rem;font-weight:750}.evidence-images a{display:block;overflow:hidden;border:1px solid var(--line);border-radius:7px;background:#05080a}.evidence-images img{display:block;width:100%;height:auto}
.readout-table{min-width:62rem}.assertion-table{min-width:48rem}.metric-table{min-width:55rem}.measurement-table{min-width:32rem}
.status-cell{min-width:10rem}.status-cell .status-badge{margin-bottom:.3rem}.status-reason{display:block;color:var(--muted);font-size:.78rem;overflow-wrap:anywhere}
.compact-list{margin:0;padding-left:1rem}.compact-list li{margin:.2rem 0}.value-list{margin:0;padding:0;list-style:none;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.8rem}
.window-grid{display:grid;gap:.8rem}.metric-window{margin:0}.metric-window>summary{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.2rem;font-weight:800}.metric-window[open]>summary{margin-bottom:.7rem}
@media(max-width:760px){.evidence-images{grid-template-columns:1fr}.subsection-heading{align-items:flex-start;flex-direction:column}}
)CSS");
  out << "<header class=\"report-header\"><p class=\"report-kicker\">"
         "Deterministic renderer autotest</p><div class=\"title-row\"><h1>"
      << htmlEscape(report.testCase.id) << "</h1>";
  writeStatusBadge(out, overallStatus);
  out << "</div><p class=\"lede\">Checkpoint captures, readouts, assertions, "
         "and metric-window evidence in one failure-first report.</p><ul "
         "class=\"meta-list\"><li><strong>Suite:</strong> "
      << htmlEscape(report.testCase.suite)
      << "</li><li><strong>Generated:</strong> "
      << htmlEscape(report.generatedAtUtc)
      << "</li><li><strong>Backend:</strong> "
      << htmlEscape(report.environment.gpuBackend)
      << "</li><li><strong>GPU:</strong> "
      << htmlEscape(report.environment.gpuDeviceName)
      << "</li><li><strong>Commit:</strong> <code>"
      << htmlEscape(report.environment.commitHash)
      << "</code></li></ul><nav class=\"report-nav\" aria-label=\"Report "
         "sections\"><a href=\"#overview\">Overview</a>"
         "<a href=\"#checkpoints\">Checkpoints</a><a href=\"#metric-windows\">"
         "Metric windows</a><a href=\"#environment\">Environment</a>"
         "<button type=\"button\" data-action=\"theme\" aria-pressed=\"false\">"
         "Light theme</button></nav>";
  if (!report.reproduceCommand.empty()) {
    out << "<div class=\"command-box\"><code id=\"reproduce-command\">"
        << htmlEscape(report.reproduceCommand)
        << "</code><button type=\"button\" "
           "data-copy-target=\"#reproduce-command\">"
           "Copy command</button></div>";
  }
  out << "</header><main id=\"main-content\" tabindex=\"-1\"><section "
         "id=\"overview\" aria-labelledby=\"overview-title\"><div "
         "class=\"section-heading\"><div><h2 id=\"overview-title\">Run overview"
         "</h2><p>Evidence totals and assertion outcomes.</p></div></div>"
         "<div class=\"summary-grid\"><div class=\"summary-card\"><strong>"
      << report.checkpoints.size()
      << "</strong><span>checkpoints</span></div><div class=\"summary-card\">"
         "<strong>"
      << counts.captures
      << "</strong><span>captures</span></div><div class=\"summary-card\">"
         "<strong>"
      << counts.readouts
      << "</strong><span>readouts</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-pass\">"
      << counts.passed
      << "</strong><span>assertions passed</span></div><div "
         "class=\"summary-card\">"
         "<strong class=\"tone-warn\">"
      << counts.warned
      << "</strong><span>assertions warned</span></div><div "
         "class=\"summary-card\">"
         "<strong class=\"tone-fail\">"
      << counts.failed
      << "</strong><span>assertions failed</span></div><div "
         "class=\"summary-card\"><strong class=\"tone-warn\">"
      << counts.unavailable
      << "</strong><span>assertions unavailable</span></div></div></section>";
  writeMessages(out, "Run errors", report.errors, "fail", "alert");
  writeMessages(out, "Run warnings", report.warnings, "warn", "status");
  out << "<details class=\"panel\" id=\"environment\"><summary><strong>Run "
         "configuration, environment, and authority</strong></summary><div "
         "class=\"detail-grid\"><dl><dt>Baseline profile</dt><dd>"
      << htmlEscape(report.baselineProfile)
      << "</dd><dt>Profile compatible</dt><dd>"
      << (report.baselineProfileCompatible ? "yes" : "no")
      << "</dd><dt>Resolution</dt><dd>" << report.testCase.resolution[0] << "×"
      << report.testCase.resolution[1]
      << "</dd><dt>Fixed delta</dt><dd class=\"mono\">"
      << report.run.fixedDeltaSeconds
      << " s</dd></dl><dl><dt>Rendered frames</dt><dd>"
      << report.run.renderedFrames << "</dd><dt>Warmup frames</dt><dd>"
      << report.run.warmupFrames << "</dd><dt>Readout drain</dt><dd>"
      << report.run.readoutDrainFrames << " / "
      << report.run.readoutDrainFrameLimit
      << " frames</dd><dt>Window mode</dt><dd>"
      << htmlEscape(report.run.resolvedWindowMode)
      << "</dd></dl><dl><dt>Build</dt><dd>"
      << htmlEscape(report.environment.buildType) << "</dd><dt>Driver</dt><dd>"
      << htmlEscape(report.environment.gpuDriverVersion)
      << "</dd><dt>Present mode</dt><dd>"
      << htmlEscape(report.environment.resolvedPresentMode)
      << "</dd><dt>Capture sync</dt><dd>"
      << htmlEscape(report.run.captureSynchronization) << "</dd></dl></div>";
  if (!report.baselineProfileIncompatibilityReasons.empty()) {
    writeMessages(out, "Profile incompatibility reasons",
                  report.baselineProfileIncompatibilityReasons, "warn",
                  "status");
  }
  out << "</details><section id=\"checkpoints\" "
         "aria-labelledby=\"checkpoints-title\"><div class=\"section-heading\">"
         "<div><h2 id=\"checkpoints-title\">Checkpoints</h2><p>Captures and "
         "assertions at deterministic frames, ordered by execution.</p></div>"
         "<span class=\"section-count\">"
      << report.checkpoints.size() << " total</span></div>";
  if (!report.checkpoints.empty()) {
    out << "<div class=\"toolbar\" role=\"search\" aria-label=\"Filter "
           "checkpoints\"><div class=\"control-group\"><label "
           "for=\"checkpoint-search\">Search checkpoint</label><input "
           "id=\"checkpoint-search\" type=\"search\" placeholder=\"Checkpoint "
           "id\"></div><div class=\"control-group\"><label "
           "for=\"checkpoint-status\">Outcome</label><select "
           "id=\"checkpoint-status\"><option value=\"\">All outcomes</option>"
           "<option>error</option><option>fail</option><option>warn</option>"
           "<option>pass</option></select></div><p id=\"checkpoint-results\" "
           "class=\"results-count\" aria-live=\"polite\"></p></div>";
  }
  for (size_t checkpointIndex = 0u; checkpointIndex < report.checkpoints.size();
       ++checkpointIndex) {
    const AutotestCheckpointReport &checkpoint =
        report.checkpoints[checkpointIndex];
    const std::string_view outcome = checkpointStatus(checkpoint);
    out << "<article class=\"checkpoint status-" << outcome
        << "\" data-status=\"" << outcome << "\" data-checkpoint=\""
        << htmlEscape(checkpoint.id) << "\" id=\"checkpoint-" << checkpointIndex
        << "\"><div class=\"card-heading\"><div>"
           "<p class=\"report-kicker\">Checkpoint "
        << checkpointIndex + 1u << "</p><h2>" << htmlEscape(checkpoint.id)
        << "</h2><p class=\"checkpoint-meta\"><span>Frame " << checkpoint.frame
        << "</span><span>" << checkpoint.captures.size()
        << " captures</span><span>" << checkpoint.readouts.size()
        << " readouts</span><span>" << checkpoint.assertions.size()
        << " direct assertions</span></p></div>";
    writeStatusBadge(out, outcome);
    out << "</div>";
    writeMessages(out, "Checkpoint errors", checkpoint.errors, "fail", "alert");
    writeMessages(out, "Checkpoint warnings", checkpoint.warnings, "warn",
                  "status");
    out << "<div class=\"checkpoint-content\"><section class=\"subsection\" "
           "aria-labelledby=\"checkpoint-captures-"
        << checkpointIndex
        << "\"><div class=\"subsection-heading\"><h3 "
           "id=\"checkpoint-captures-"
        << checkpointIndex << "\">Captures</h3><span class=\"section-count\">"
        << checkpoint.captures.size() << "</span></div>";
    if (checkpoint.captures.empty()) {
      out << "<p class=\"empty-state\">No captures requested at this "
             "checkpoint.</p>";
    } else {
      out << "<div class=\"evidence-grid\">";
    }
    for (const AutotestCaptureReport &capture : checkpoint.captures) {
      out << "<article class=\"evidence-card\"><div class=\"card-heading\">"
             "<h3>"
          << htmlEscape(capture.target) << "</h3>";
      writeStatusBadge(out, capture.snapshot.status);
      out << "</div><p>"
          << htmlEscape(readableStatus(capture.snapshot.statusReason))
          << "</p><dl><dt>Profile</dt><dd>" << htmlEscape(capture.profile)
          << "</dd><dt>Required</dt><dd>" << (capture.required ? "yes" : "no")
          << "</dd><dt>Compare</dt><dd>" << (capture.compare ? "yes" : "no")
          << "</dd></dl>";
      if (!capture.snapshot.preview.empty() || !capture.snapshot.diff.empty()) {
        out << "<div class=\"evidence-images\">";
        if (!capture.snapshot.preview.empty()) {
          const std::string href =
              autotestArtifactHref(report, capture.snapshot.preview);
          out << "<figure><figcaption>Actual preview</figcaption><a href=\""
              << htmlEscape(href) << "\" aria-label=\"Open actual preview for "
              << htmlEscape(capture.target)
              << "\"><img decoding=\"async\" src=\"" << htmlEscape(href)
              << "\" alt=\"" << htmlEscape(capture.target)
              << " actual preview\"></a></figure>";
        }
        if (!capture.snapshot.diff.empty()) {
          const std::string href =
              autotestArtifactHref(report, capture.snapshot.diff);
          out << "<figure><figcaption>Semantic diff</figcaption><a href=\""
              << htmlEscape(href) << "\" aria-label=\"Open semantic diff for "
              << htmlEscape(capture.target)
              << "\"><img decoding=\"async\" src=\"" << htmlEscape(href)
              << "\" alt=\"" << htmlEscape(capture.target)
              << " semantic difference\"></a></figure>";
        }
        out << "</div>";
      }
      out << "</article>";
    }
    if (!checkpoint.captures.empty()) {
      out << "</div>";
    }
    out << "</section>";
    if (checkpoint.motionOracle.has_value()) {
      const AutotestMotionOracleReport &motion = *checkpoint.motionOracle;
      out << "<section class=\"subsection\"><div class=\"subsection-heading\">"
             "<h3>Motion endpoint oracle</h3>";
      writeStatusBadge(out, motion.status);
      out << "</div><p>" << htmlEscape(readableStatus(motion.statusReason))
          << "</p><div class=\"table-wrap\"><table>"
             "<caption>Analytic current-to-previous motion-vector endpoint "
             "metrics in pixels</caption><thead><tr>"
             "<th scope=\"col\">Samples</th><th scope=\"col\">Expected</th>"
             "<th scope=\"col\">Mean</th><th scope=\"col\">P95 error</th>"
             "<th scope=\"col\">Max error</th>"
             "<th scope=\"col\">P95 scale error</th>"
             "<th scope=\"col\">Wrong sign</th></tr></thead><tbody><tr>"
          << "<td class=\"numeric\">" << motion.selectedPixelCount
          << "</td><td class=\"numeric\">(" << motion.expectedVelocityPixels[0]
          << ", " << motion.expectedVelocityPixels[1]
          << ")</td><td class=\"numeric\">(" << motion.meanVelocityPixels[0]
          << ", " << motion.meanVelocityPixels[1]
          << ")</td><td class=\"numeric\">" << motion.p95ErrorPixels << " / "
          << motion.p95ErrorMaxPixels << "</td><td class=\"numeric\">"
          << motion.maxErrorPixels << " / " << motion.maxErrorMaxPixels
          << "</td><td class=\"numeric\">" << motion.p95ScaleErrorPixels
          << "</td><td class=\"numeric\">" << motion.wrongSignPixelCount
          << "</td></tr></tbody></table></div>";
      if (motion.classCoverageAvailable) {
        out << "<p>Motion classes: invalid=" << motion.invalidClassCoverage
            << ", static=" << motion.staticClassCoverage
            << ", full=" << motion.fullClassCoverage << ".</p>";
      }
      if (!motion.failedThresholds.empty()) {
        out << "<p>Failed thresholds: ";
        for (size_t i = 0u; i < motion.failedThresholds.size(); ++i) {
          if (i != 0u) {
            out << ", ";
          }
          out << htmlEscape(motion.failedThresholds[i]);
        }
        out << ".</p>";
      }
      out << "</section>";
    }
    if (checkpoint.qualityOracle.has_value()) {
      const AutotestQualityOracleReport &quality = *checkpoint.qualityOracle;
      out << "<section class=\"subsection\"><div class=\"subsection-heading\">"
             "<h3>Deterministic AA quality oracle</h3>";
      writeStatusBadge(out, quality.status);
      out << "</div><p>" << htmlEscape(readableStatus(quality.statusReason))
          << "</p><div class=\"table-wrap\"><table>"
             "<caption>Scene-referred linear HDR and temporal quality "
             "metrics</caption><thead><tr>"
             "<th scope=\"col\">Metric</th><th scope=\"col\">Actual</th>"
             "<th scope=\"col\">Budget</th></tr></thead><tbody>"
          << "<tr><td>Normalized HDR MAE</td><td class=\"numeric\">"
          << quality.normalizedHdrMae << "</td><td class=\"numeric\">&le; "
          << quality.budgets.normalizedMaeMax << "</td></tr>"
          << "<tr><td>Normalized HDR RMSE</td><td class=\"numeric\">"
          << quality.normalizedHdrRmse << "</td><td class=\"numeric\">&le; "
          << quality.budgets.normalizedRmseMax << "</td></tr>"
          << "<tr><td>Rec.709 luma SSIM</td><td class=\"numeric\">"
          << quality.lumaSsim << "</td><td class=\"numeric\">&ge; "
          << quality.budgets.lumaSsimMin << "</td></tr>"
          << "<tr><td>Dark-collapse pixels</td><td class=\"numeric\">"
          << quality.darkCollapsePercent << "% / component "
          << quality.darkCollapseMaxComponentPixels
          << "</td><td class=\"numeric\">&le; "
          << quality.budgets.darkCollapsePercentMax << "% / component "
          << quality.budgets.darkCollapseComponentMaxPixels << "</td></tr>"
          << "<tr><td>Relative luma energy drift</td><td class=\"numeric\">"
          << quality.relativeLumaEnergyDrift
          << "</td><td class=\"numeric\">&le; "
          << quality.budgets.relativeLumaEnergyDriftMax << "</td></tr>"
          << "<tr><td>Non-finite values</td><td class=\"numeric\">"
          << quality.nonFiniteValueCount
          << "</td><td class=\"numeric\">0</td></tr>";
      if (quality.edgeAvailable) {
        out << "<tr><td>Edge 10-90% width ratio ("
            << htmlEscape(quality.edgeAxis) << ")</td><td class=\"numeric\">"
            << quality.edgeWidthRatio << " (" << quality.outputEdgeWidth10To90
            << " / " << quality.referenceEdgeWidth10To90
            << ")</td><td class=\"numeric\">"
            << quality.budgets.edgeWidthRatioMin << " - "
            << quality.budgets.edgeWidthRatioMax << "</td></tr>"
            << "<tr><td>Edge overshoot / undershoot</td>"
               "<td class=\"numeric\">"
            << quality.edgeOvershoot << " / " << quality.edgeUndershoot
            << "</td><td class=\"numeric\">&le; "
            << quality.budgets.edgeOvershootMax << " / &le; "
            << quality.budgets.edgeUndershootMax << "</td></tr>";
      }
      if (quality.temporalAvailable) {
        out << "<tr><td>Motion-compensated temporal error</td>"
               "<td class=\"numeric\">"
            << quality.temporalError << "</td><td class=\"numeric\">&le; "
            << quality.budgets.temporalErrorMax << "</td></tr>";
      }
      if (quality.revealAvailable) {
        out << "<tr><td>Reveal ghost energy</td><td class=\"numeric\">"
            << quality.ghostEnergy << "</td><td class=\"numeric\">&le; "
            << quality.budgets.ghostEnergyMax << "</td></tr>"
            << "<tr><td>Reveal recovery RMSE</td><td class=\"numeric\">"
            << quality.recoveryRmse << "</td><td class=\"numeric\">&le; "
            << quality.budgets.recoveryRmseMax << "</td></tr>";
      }
      out << "</tbody></table></div><p>Selected pixels: "
          << quality.selectedPixelCount
          << "; finite pixels: " << quality.finitePixelCount
          << "; Lscale: " << quality.lscale << "; reference v"
          << quality.referenceVersion;
      if (quality.maskVersion != 0u) {
        out << "; mask v" << quality.maskVersion;
      }
      out << ".</p>";
      if (!quality.failedThresholds.empty()) {
        out << "<p>Failed thresholds: ";
        for (size_t i = 0u; i < quality.failedThresholds.size(); ++i) {
          if (i != 0u) {
            out << ", ";
          }
          out << htmlEscape(quality.failedThresholds[i]);
        }
        out << ".</p>";
      }
      out << "</section>";
    }
    if (!checkpoint.readouts.empty()) {
      out << "<section class=\"subsection\"><div class=\"subsection-heading\">"
             "<h3>Readouts</h3><span class=\"section-count\">"
          << checkpoint.readouts.size()
          << "</span></div><div class=\"table-wrap\"><table "
             "class=\"readout-table\"><caption>Asynchronous readout requests, "
             "results, values, and nested assertions</caption><thead><tr>"
             "<th scope=\"col\">ID</th><th scope=\"col\">Type</th>"
             "<th scope=\"col\">Status</th><th scope=\"col\">Request frame</th>"
             "<th scope=\"col\">Result frame</th><th scope=\"col\">Values</th>"
             "<th scope=\"col\">Assertions</th></tr></thead><tbody>";
      for (const AutotestReadoutReport &readout : checkpoint.readouts) {
        out << "<tr><td>" << htmlEscape(readout.id) << "</td><td>"
            << htmlEscape(readout.type) << "</td><td class=\"status-cell\">";
        writeStatusBadge(out, readout.status);
        out << "<span class=\"status-reason\">"
            << htmlEscape(readableStatus(readout.statusReason))
            << "</span></td><td class=\"numeric\">" << readout.requestFrame
            << "</td><td class=\"numeric\">";
        if (readout.resultFrame != 0u || readout.status == "pass" ||
            readout.status == "fail" || readout.status == "warn") {
          out << readout.resultFrame;
        }
        out << "</td><td><ul class=\"value-list\">";
        for (const auto &[key, value] : readout.values) {
          out << "<li>" << htmlEscape(key) << " = " << value << "</li>";
        }
        out << "</ul></td><td><ul class=\"compact-list\">";
        for (const AutotestAssertionResult &assertion : readout.assertions) {
          const std::string status =
              autotestAssertionStatusName(assertion.status);
          out << "<li>" << htmlEscape(assertion.id)
              << " — <span class=\"status-" << htmlEscape(status) << "\">"
              << htmlEscape(status) << "</span>";
          if (assertion.hasActual) {
            out << " " << assertion.actual;
          }
          out << "</li>";
        }
        out << "</ul></td></tr>";
      }
      out << "</tbody></table></div></section>";
    }
    out << "<section class=\"subsection\"><div class=\"subsection-heading\">"
           "<h3>Direct assertions</h3><span class=\"section-count\">"
        << checkpoint.assertions.size() << "</span></div>";
    if (checkpoint.assertions.empty()) {
      out << "<p class=\"empty-state\">No direct assertions at this checkpoint."
             "</p>";
    } else {
      out << "<div class=\"table-wrap\"><table class=\"assertion-table\">"
             "<caption>Checkpoint assertion outcomes</caption><thead><tr>"
             "<th scope=\"col\">ID</th><th scope=\"col\">Metric</th>"
             "<th scope=\"col\">Status</th><th scope=\"col\">Actual</th>"
             "<th scope=\"col\">Samples</th></tr></thead><tbody>";
      for (const AutotestAssertionResult &assertion : checkpoint.assertions) {
        const std::string status =
            autotestAssertionStatusName(assertion.status);
        out << "<tr><td>" << htmlEscape(assertion.id) << "</td><td><code>"
            << htmlEscape(assertion.metric)
            << "</code></td><td class=\"status-cell\">";
        writeStatusBadge(out, status);
        out << "<span class=\"status-reason\">"
            << htmlEscape(readableStatus(assertion.statusReason))
            << "</span></td><td class=\"numeric\">";
        if (assertion.hasActual) {
          out << assertion.actual;
        }
        out << "</td><td class=\"numeric\">" << assertion.sampleCount;
        if (assertion.expectedSampleCount != 0u) {
          out << " / " << assertion.expectedSampleCount;
        }
        out << "</td></tr>";
      }
      out << "</tbody></table></div>";
    }
    out << "</section>";
    if (!checkpoint.measurements.empty()) {
      out << "<details class=\"subsection\"><summary><strong>Raw checkpoint "
             "measurements ("
          << checkpoint.measurements.size()
          << ")</strong></summary><div class=\"table-wrap\"><table "
             "class=\"measurement-table\"><caption>Recorded renderer metric "
             "values</caption><thead><tr><th scope=\"col\">Metric</th>"
             "<th scope=\"col\">Value</th></tr></thead><tbody>";
      for (const auto &[metric, value] : checkpoint.measurements) {
        out << "<tr><td><code>" << htmlEscape(metric)
            << "</code></td><td class=\"numeric\">" << value << "</td></tr>";
      }
      out << "</tbody></table></div></details>";
    }
    out << "</div></article>";
  }
  if (report.checkpoints.empty()) {
    out << "<p class=\"empty-state\">No checkpoints were completed.</p>";
  }
  out << "</section>";
  if (!report.metricWindows.empty()) {
    out << "<section id=\"metric-windows\" "
           "aria-labelledby=\"metric-windows-title\">"
           "<div class=\"section-heading\"><div><h2 "
           "id=\"metric-windows-title\">"
           "Metric windows</h2><p>Aggregate assertions across frame ranges.</p>"
           "</div><span class=\"section-count\">"
        << report.metricWindows.size()
        << " total</span></div><div class=\"window-grid\">";
    for (const AutotestMetricWindowReport &window : report.metricWindows) {
      out << "<details class=\"panel metric-window\" open><summary><span>"
          << htmlEscape(window.id) << "</span><span class=\"muted\">frames "
          << window.startFrame << "–" << window.endFrame << "</span></summary>";
      writeMessages(out, "Window errors", window.errors, "fail", "alert");
      writeMessages(out, "Window warnings", window.warnings, "warn", "status");
      out << "<div class=\"table-wrap\"><table class=\"metric-table\">"
             "<caption>Metric-window assertion outcomes</caption><thead><tr>"
             "<th scope=\"col\">ID</th><th scope=\"col\">Metric</th>"
             "<th scope=\"col\">Statistic</th><th scope=\"col\">Status</th>"
             "<th scope=\"col\">Actual</th><th scope=\"col\">Samples</th>"
             "</tr></thead><tbody>";
      for (const AutotestAssertionResult &assertion : window.assertions) {
        const std::string status =
            autotestAssertionStatusName(assertion.status);
        out << "<tr><td>" << htmlEscape(assertion.id) << "</td><td>"
            << "<code>" << htmlEscape(assertion.metric) << "</code></td><td>"
            << htmlEscape(assertion.statistic)
            << "</td><td class=\"status-cell\">";
        writeStatusBadge(out, status);
        out << "<span class=\"status-reason\">"
            << htmlEscape(readableStatus(assertion.statusReason))
            << "</span></td><td class=\"numeric\">";
        if (assertion.hasActual) {
          out << assertion.actual;
        }
        out << "</td><td class=\"numeric\">" << assertion.sampleCount
            << "</td></tr>";
      }
      out << "</tbody></table></div></details>";
    }
    out << "</div></section>";
  } else {
    out << "<section id=\"metric-windows\" "
           "aria-labelledby=\"metric-windows-title\">"
           "<h2 id=\"metric-windows-title\">Metric windows</h2>"
           "<p class=\"empty-state\">No metric windows were "
           "evaluated.</p></section>";
  }
  out << "</main>";
  nuri::tools::core::endHtmlReport(out, R"JS(
(() => {
  const search = document.querySelector('#checkpoint-search');
  const status = document.querySelector('#checkpoint-status');
  const checkpoints = [...document.querySelectorAll('.checkpoint')];
  const results = document.querySelector('#checkpoint-results');
  const filter = () => {
    const query = (search?.value || '').trim().toLowerCase();
    const selected = status?.value || '';
    let visible = 0;
    checkpoints.forEach((checkpoint) => {
      const show = (!query || checkpoint.dataset.checkpoint.toLowerCase().includes(query)) &&
        (!selected || checkpoint.dataset.status === selected);
      checkpoint.hidden = !show;
      if (show) visible += 1;
    });
    if (results) results.textContent = `${visible} of ${checkpoints.length} checkpoints shown`;
  };
  search?.addEventListener('input', filter);
  status?.addEventListener('change', filter);
  filter();
})();
)JS");
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
  size_t passed = 0u;
  size_t warned = 0u;
  size_t failed = 0u;
  size_t invalid = 0u;
  for (const AutotestReport &report : reports) {
    if (report.status == "pass") {
      ++passed;
    } else if (report.status == "warn" || report.status == "investigative" ||
               report.status == "unavailable") {
      ++warned;
    } else if (report.status == "fail" || report.status == "error") {
      ++failed;
    } else {
      ++invalid;
    }
  }
  const std::string_view overallStatus =
      invalid != 0u  ? std::string_view{"invalid"}
      : failed != 0u ? std::string_view{"fail"}
      : warned != 0u ? std::string_view{"warn"}
                     : std::string_view{"pass"};
  std::ostringstream out;
  nuri::tools::core::beginHtmlReport(
      out, "Nuri Autotest Suite " + std::string(suite),
      R"CSS(
.case-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,22rem),1fr));gap:.8rem;margin:0;padding:0;list-style:none}
.case-card{min-width:0;padding:1rem;border:1px solid var(--line);border-left-width:5px;border-radius:var(--radius);background:var(--surface)}
.case-card.status-pass{border-left-color:var(--pass)}.case-card.status-warn,.case-card.status-investigative,.case-card.status-unavailable{border-left-color:var(--warn)}.case-card.status-fail,.case-card.status-error,.case-card.status-invalid{border-left-color:var(--fail)}
.case-card h3{margin:.6rem 0 .35rem}.case-meta{display:flex;flex-wrap:wrap;gap:.35rem .75rem;margin:0;color:var(--muted);font-size:.82rem}
)CSS");
  out << "<header class=\"report-header\"><p class=\"report-kicker\">"
         "Deterministic renderer autotest suite</p><div "
         "class=\"title-row\"><h1>"
      << htmlEscape(suite) << "</h1>";
  writeStatusBadge(out, overallStatus);
  out << "</div><p class=\"lede\">Suite-wide outcomes ranked for triage, with "
         "checkpoint and diagnostic totals per case.</p><nav "
         "class=\"report-nav\" "
         "aria-label=\"Report actions\"><a href=\"#cases\">Cases</a>"
         "<button type=\"button\" data-action=\"theme\" aria-pressed=\"false\">"
         "Light theme</button></nav></header><main id=\"main-content\" "
         "tabindex=\"-1\"><section aria-labelledby=\"suite-overview\">"
         "<div class=\"section-heading\"><div><h2 id=\"suite-overview\">Suite "
         "overview</h2><p>Selected autotest case outcomes.</p></div></div>"
         "<div class=\"summary-grid\"><div class=\"summary-card\"><strong>"
      << reports.size()
      << "</strong><span>cases</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-pass\">"
      << passed
      << "</strong><span>passed</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-warn\">"
      << warned
      << "</strong><span>warn / investigative</span></div><div "
         "class=\"summary-card\"><strong class=\"tone-fail\">"
      << failed
      << "</strong><span>failed</span></div><div class=\"summary-card\">"
         "<strong class=\"tone-fail\">"
      << invalid
      << "</strong><span>invalid</span></div></div></section><section "
         "id=\"cases\" aria-labelledby=\"cases-title\"><div "
         "class=\"section-heading\"><div><h2 id=\"cases-title\">Cases</h2>"
         "<p>Filter cases by identifier or outcome.</p></div><span "
         "class=\"section-count\">"
      << reports.size()
      << " total</span></div><div class=\"toolbar\" role=\"search\" "
         "aria-label=\"Filter autotest cases\"><div class=\"control-group\">"
         "<label for=\"case-search\">Search cases</label><input "
         "id=\"case-search\" "
         "type=\"search\" placeholder=\"Case id\"></div><div "
         "class=\"control-group\">"
         "<label for=\"case-status\">Outcome</label><select id=\"case-status\">"
         "<option value=\"\">All outcomes</option><option>invalid</option>"
         "<option>fail</option><option>error</option><option>warn</option>"
         "<option>investigative</option><option>unavailable</"
         "option><option>pass</option>"
         "</select></div><p id=\"case-results\" class=\"results-count\" "
         "aria-live=\"polite\"></p></div><ul class=\"case-grid\">";
  for (const AutotestReport &report : reports) {
    const std::string_view status =
        report.status.empty() ? std::string_view{"invalid"} : report.status;
    const AutotestHtmlCounts counts = autotestHtmlCounts(report);
    out << "<li class=\"case-card status-" << htmlEscape(status)
        << "\" data-case=\"" << htmlEscape(report.testCase.id)
        << "\" data-status=\"" << htmlEscape(status) << "\">";
    writeStatusBadge(out, status);
    out << "<h3>" << htmlEscape(report.testCase.id)
        << "</h3><p class=\"case-meta\"><span>" << report.checkpoints.size()
        << " checkpoints</span><span>" << counts.captures
        << " captures</span><span>" << counts.assertions
        << " assertions</span><span>" << report.warnings.size()
        << " warnings</span><span>" << report.errors.size()
        << " errors</span></p></li>";
  }
  if (reports.empty()) {
    out << "<li class=\"empty-state\">No autotest cases were selected.</li>";
  }
  out << "</ul></section></main>";
  nuri::tools::core::endHtmlReport(out, R"JS(
(() => {
  const search = document.querySelector('#case-search');
  const status = document.querySelector('#case-status');
  const cases = [...document.querySelectorAll('.case-card')];
  const results = document.querySelector('#case-results');
  const filter = () => {
    const query = (search?.value || '').trim().toLowerCase();
    const selected = status?.value || '';
    let visible = 0;
    cases.forEach((item) => {
      const show = (!query || item.dataset.case.toLowerCase().includes(query)) &&
        (!selected || item.dataset.status === selected);
      item.hidden = !show;
      if (show) visible += 1;
    });
    if (results) results.textContent = `${visible} of ${cases.length} cases shown`;
  };
  search?.addEventListener('input', filter);
  status?.addEventListener('change', filter);
  filter();
})();
)JS");
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
