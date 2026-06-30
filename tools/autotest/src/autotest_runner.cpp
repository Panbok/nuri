#include "nuri/tools/autotest/autotest_runner.h"

#include "nuri/tools/autotest/autotest_manifest.h"
#include "nuri/tools/autotest/autotest_record.h"
#include "nuri/tools/autotest/autotest_timeline.h"
#include "nuri/tools/runtime/render_tool_runtime.h"
#include "nuri/tools/snapshot/snapshot_baseline.h"
#include "nuri/tools/snapshot/snapshot_capture_artifacts.h"
#include "nuri/tools/snapshot/snapshot_compare.h"
#include "nuri/tools/snapshot/snapshot_image.h"

#include "nuri/core/profiling.h"

#include <algorithm>
#include <array>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>

namespace nuri::tools::autotest {
namespace {

constexpr uint32_t kReadoutDrainFrameLimit = 4u;

struct PendingReadout {
  uint64_t requestId = 0u;
  const AutotestReadoutRequest *request = nullptr;
  size_t checkpointReportIndex = 0u;
  size_t readoutReportIndex = 0u;
  uint32_t requestFrame = 0u;
};

struct CheckpointFrameWork {
  const AutotestCheckpoint *checkpoint = nullptr;
  size_t reportIndex = 0u;
};

[[nodiscard]] std::string jsonEscape(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    default:
      out += ch;
      break;
    }
  }
  return out;
}

[[nodiscard]] std::filesystem::path
relativeToCaseDir(const std::filesystem::path &caseDir,
                  const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path relative = std::filesystem::relative(path, caseDir, ec);
  return ec ? path : relative;
}

[[nodiscard]] std::string
checkpointDirName(const AutotestCheckpoint &checkpoint) {
  return checkpoint.id + "_frame_" + std::to_string(checkpoint.frame);
}

[[nodiscard]] std::string resolveBackendName(const AutotestCase &testCase,
                                             std::string &source) {
  const std::string envBackend = readProcessEnvironment("NURI_GPU_BACKEND");
  if (testCase.backend != "default") {
    source = "manifest";
    return testCase.backend;
  }
  if (!envBackend.empty()) {
    source = "NURI_GPU_BACKEND";
    return envBackend;
  }
  source = "default";
  return "nvrhi";
}

[[nodiscard]] std::string resolvePresentMode(const AutotestCase &testCase,
                                             std::string &source) {
  const std::string envPresent = readProcessEnvironment("NURI_PRESENT_MODE");
  if (testCase.presentMode != "default") {
    source = "manifest";
    return testCase.presentMode;
  }
  if (!envPresent.empty()) {
    source = "NURI_PRESENT_MODE";
    return envPresent;
  }
  source = "default";
  return "default";
}

[[nodiscard]] Result<bool, AutotestExitCode>
checkRequirements(const AutotestCase &testCase, std::string_view backend,
                  std::string_view windowMode,
                  std::vector<std::string> &warnings, std::string &message) {
  if (windowMode != "visible") {
    message = "only visible window mode is available";
    return Result<bool, AutotestExitCode>::makeError(
        AutotestExitCode::EnvironmentUnavailable);
  }
  if (!testCase.requirements.allowVisibleWindow) {
    message = "case requires hidden/headless execution, which is unavailable";
    return Result<bool, AutotestExitCode>::makeError(
        AutotestExitCode::EnvironmentUnavailable);
  }
  if (!testCase.requirements.backends.empty()) {
    bool supported = false;
    for (const std::string &allowed : testCase.requirements.backends) {
      supported = supported || allowed == backend || allowed == "default";
    }
    if (!supported) {
      message = "backend '" + std::string(backend) +
                "' is not allowed by case requirements";
      return Result<bool, AutotestExitCode>::makeError(
          AutotestExitCode::EnvironmentUnavailable);
    }
  }
  for (const std::string &asset : testCase.requirements.assets) {
    const size_t colon = asset.find(':');
    if (colon == std::string::npos) {
      message = "invalid asset requirement '" + asset + "'";
      return Result<bool, AutotestExitCode>::makeError(
          AutotestExitCode::InvalidInput);
    }
    auto path =
        resolveAutotestPath(asset.substr(0, colon), asset.substr(colon + 1u));
    if (path.hasError()) {
      message = path.error();
      return Result<bool, AutotestExitCode>::makeError(
          AutotestExitCode::EnvironmentUnavailable);
    }
    if (!std::filesystem::exists(path.value())) {
      message = "missing required asset: " + path.value().string();
      return Result<bool, AutotestExitCode>::makeError(
          AutotestExitCode::EnvironmentUnavailable);
    }
  }
  (void)warnings;
  return Result<bool, AutotestExitCode>::makeResult(true);
}

[[nodiscard]] nuri::tools::runtime::ToolSceneDesc
makeToolSceneDesc(const AutotestSceneConfig &scene) {
  return nuri::tools::runtime::ToolSceneDesc{
      .kind = scene.kind,
      .pathBase = scene.pathBase,
      .path = scene.path,
      .flipUVs = scene.flipUVs,
      .generateMeshlets = scene.generateMeshlets,
      .meshletMaxVertices = scene.meshletMaxVertices,
      .meshletMaxPrimitives = scene.meshletMaxPrimitives,
      .meshletConeWeight = scene.meshletConeWeight,
      .generator = scene.generator,
      .seed = scene.seed,
      .contentHash = scene.contentHash,
  };
}

[[nodiscard]] nuri::tools::runtime::ToolCameraDesc
makeToolCameraDesc(const AutotestCameraConfig &camera) {
  return nuri::tools::runtime::ToolCameraDesc{
      .position = camera.position,
      .direction = camera.direction,
      .target = camera.target,
      .hasTarget = camera.hasTarget,
      .verticalFovDegrees = camera.verticalFovDegrees,
      .nearPlane = camera.nearPlane,
      .farPlane = camera.farPlane,
  };
}

[[nodiscard]] nuri::tools::runtime::ToolRuntimeDesc
makeToolRuntimeDesc(const AutotestCase &testCase, std::string_view backend,
                    std::string_view presentMode) {
  nuri::tools::runtime::ToolRuntimeDesc desc{};
  desc.title = "nuri-autotest " + testCase.id;
  desc.backend = std::string(backend);
  desc.presentMode = std::string(presentMode);
  desc.resolution = testCase.resolution;
  desc.renderGraph.workerCount = testCase.renderGraph.workerCount;
  desc.renderGraph.parallelCompile = testCase.renderGraph.parallelCompile;
  desc.renderGraph.parallelRecording = testCase.renderGraph.parallelRecording;
  desc.scene = makeToolSceneDesc(testCase.scene);
  desc.resolvePath = resolveAutotestPath;
  return desc;
}

[[nodiscard]] std::vector<nuri::tools::snapshot::SnapshotCaptureTarget>
makeSnapshotCaptureTargets(const AutotestCheckpoint &checkpoint) {
  std::vector<nuri::tools::snapshot::SnapshotCaptureTarget> targets;
  targets.reserve(checkpoint.captures.size());
  for (const AutotestCaptureTarget &capture : checkpoint.captures) {
    targets.push_back(nuri::tools::snapshot::SnapshotCaptureTarget{
        .name = capture.target,
        .profile = capture.profile,
        .required = capture.required,
    });
  }
  return targets;
}

[[nodiscard]] const AutotestCaptureTarget *
findAutotestCaptureTarget(const AutotestCheckpoint &checkpoint,
                          std::string_view target) {
  for (const AutotestCaptureTarget &capture : checkpoint.captures) {
    if (capture.target == target) {
      return &capture;
    }
  }
  return nullptr;
}

[[nodiscard]] bool caseHasReadouts(const AutotestCase &testCase) {
  for (const AutotestCheckpoint &checkpoint : testCase.checkpoints) {
    if (!checkpoint.readouts.empty()) {
      return true;
    }
  }
  return false;
}

void drainAutotestGpuTimings(
    GPUDevice &gpu,
    std::map<uint64_t, std::map<std::string, double>> &frameMeasurements) {
  std::array<GpuTimingReport, 64u> reports{};
  for (;;) {
    const size_t count = gpu.drainCompletedGpuTimingReports(reports);
    for (size_t i = 0u; i < count; ++i) {
      applyAutotestGpuTimingReport(frameMeasurements, reports[i]);
    }
    if (count < reports.size()) {
      break;
    }
  }
}

void applyAssertionExitStatus(const std::vector<AutotestAssertionResult> &items,
                              AutotestRunResult &result,
                              std::string_view invalidMessage,
                              std::string_view failMessage) {
  for (const AutotestAssertionResult &assertion : items) {
    if (assertion.status == AutotestAssertionStatus::Invalid &&
        result.exitCode == AutotestExitCode::Success) {
      result.exitCode = AutotestExitCode::InvalidInput;
      result.message = invalidMessage;
    } else if (assertion.status == AutotestAssertionStatus::Fail &&
               result.exitCode == AutotestExitCode::Success) {
      result.exitCode = AutotestExitCode::ScenarioFailure;
      result.message = failMessage;
    }
  }
}

[[nodiscard]] std::map<std::string, double>
readoutValuesFromOpaquePick(const OpaquePickResult &readout) {
  return {{"hit", readout.hit ? 1.0 : 0.0},
          {"renderableIndex", static_cast<double>(readout.renderableIndex)}};
}

[[nodiscard]] std::map<std::string, double>
readoutValuesFromShadowInspect(const ShadowInspectResult &readout) {
  return {{"valid", readout.valid ? 1.0 : 0.0},
          {"receiverDepth", readout.receiverDepth},
          {"receiverCompareDepth", readout.receiverCompareDepth},
          {"sampledDepth", readout.sampledDepth},
          {"cascadeIndex", static_cast<double>(readout.cascadeIndex)},
          {"cascadeBlendFactor", readout.cascadeBlendFactor}};
}

void applyReadoutValuesToReport(AutotestReadoutReport &readoutReport,
                                const AutotestReadoutRequest &request,
                                uint32_t resultFrame,
                                std::map<std::string, double> values,
                                AutotestRunResult &result) {
  readoutReport.resultFrame = resultFrame;
  readoutReport.values = std::move(values);
  readoutReport.assertions =
      evaluateAutotestAssertions(request.assertions, readoutReport.values);
  readoutReport.status = "pass";
  readoutReport.statusReason = "readout_resolved";
  for (const AutotestAssertionResult &assertion : readoutReport.assertions) {
    if (assertion.status == AutotestAssertionStatus::Invalid) {
      readoutReport.status = "invalid_precondition";
      readoutReport.statusReason = assertion.statusReason;
      break;
    }
    if (assertion.status == AutotestAssertionStatus::Fail) {
      readoutReport.status = "fail";
      readoutReport.statusReason = assertion.statusReason;
      break;
    }
    if (assertion.status == AutotestAssertionStatus::Warn &&
        readoutReport.status == "pass") {
      readoutReport.status = "warn";
      readoutReport.statusReason = assertion.statusReason;
    }
  }
  applyAssertionExitStatus(readoutReport.assertions, result,
                           "autotest readout input unavailable",
                           "autotest readout assertion failed");
}

void resolvePendingReadoutsForFrame(
    RenderFrameContext &frameContext, uint32_t resultFrame,
    std::vector<PendingReadout> &pendingReadouts, AutotestReport &report,
    AutotestRunResult &result) {
  auto resolveById = [&](uint64_t requestId, std::string_view type,
                         std::map<std::string, double> values) {
    for (PendingReadout &pending : pendingReadouts) {
      if (pending.requestId != requestId || pending.request == nullptr ||
          pending.request->type != type) {
        continue;
      }
      AutotestReadoutReport &readoutReport =
          report.checkpoints[pending.checkpointReportIndex]
              .readouts[pending.readoutReportIndex];
      applyReadoutValuesToReport(readoutReport, *pending.request, resultFrame,
                                 std::move(values), result);
      pending.request = nullptr;
      return;
    }
  };
  if (frameContext.opaquePickResult.has_value()) {
    resolveById(frameContext.opaquePickResult->requestId, "opaquePick",
                readoutValuesFromOpaquePick(*frameContext.opaquePickResult));
  }
  if (frameContext.shadowInspectResult.has_value()) {
    resolveById(
        frameContext.shadowInspectResult->requestId, "shadowInspect",
        readoutValuesFromShadowInspect(*frameContext.shadowInspectResult));
  }
  pendingReadouts.erase(std::remove_if(pendingReadouts.begin(),
                                       pendingReadouts.end(),
                                       [](const PendingReadout &pending) {
                                         return pending.request == nullptr;
                                       }),
                        pendingReadouts.end());
}

void markPendingReadoutsMissing(std::vector<PendingReadout> &pendingReadouts,
                                AutotestReport &report,
                                AutotestRunResult &result) {
  for (const PendingReadout &pending : pendingReadouts) {
    AutotestReadoutReport &readoutReport =
        report.checkpoints[pending.checkpointReportIndex]
            .readouts[pending.readoutReportIndex];
    readoutReport.status = "missing_readout";
    readoutReport.statusReason = readoutReport.required
                                     ? "readout_drain_limit_exceeded"
                                     : "optional_readout_unavailable";
    if (readoutReport.required &&
        result.exitCode == AutotestExitCode::Success) {
      result.exitCode = AutotestExitCode::ScenarioFailure;
      result.message = "autotest readout missing";
    }
  }
  pendingReadouts.clear();
}

void collectUnavailableAssertionMetric(
    const AutotestAssertionResult &assertion, std::string_view prefix,
    std::set<std::string> &unavailableMetrics) {
  if (assertion.status != AutotestAssertionStatus::Invalid &&
      assertion.status != AutotestAssertionStatus::Unavailable) {
    return;
  }
  if (!assertion.metric.empty()) {
    unavailableMetrics.insert(std::string(prefix) + assertion.metric);
  }
}

void populateUnavailableMetrics(AutotestReport &report) {
  std::set<std::string> unavailableMetrics;
  for (const AutotestCheckpointReport &checkpoint : report.checkpoints) {
    for (const AutotestAssertionResult &assertion : checkpoint.assertions) {
      collectUnavailableAssertionMetric(assertion, {}, unavailableMetrics);
    }
    for (const AutotestReadoutReport &readout : checkpoint.readouts) {
      const std::string prefix = "readout." + readout.id + ".";
      for (const AutotestAssertionResult &assertion : readout.assertions) {
        collectUnavailableAssertionMetric(assertion, prefix,
                                          unavailableMetrics);
      }
    }
  }
  for (const AutotestMetricWindowReport &window : report.metricWindows) {
    for (const AutotestAssertionResult &assertion : window.assertions) {
      collectUnavailableAssertionMetric(assertion, {}, unavailableMetrics);
    }
  }
  report.unavailableMetrics.assign(unavailableMetrics.begin(),
                                   unavailableMetrics.end());
}

void populateVerboseFrames(
    AutotestReport &report,
    const std::map<uint64_t, std::map<std::string, double>>
        &frameMeasurements) {
  report.frames.clear();
  report.frames.reserve(frameMeasurements.size());
  for (const auto &[frameIndex, measurements] : frameMeasurements) {
    report.frames.push_back(AutotestFrameReport{
        .frameIndex = frameIndex,
        .measurements = measurements,
    });
  }
}

[[nodiscard]] std::vector<CheckpointFrameWork> beginCheckpointReportsForFrame(
    RenderFrameContext &frameContext, const AutotestFramePlan &frame,
    AutotestReport &report, std::vector<PendingReadout> &pendingReadouts,
    uint64_t &nextReadoutRequestId, AutotestRunResult &result) {
  std::vector<CheckpointFrameWork> work;
  work.reserve(frame.checkpoints.size());
  bool shadowInspectScheduled = false;
  bool opaquePickScheduled = false;
  for (const AutotestCheckpoint *checkpoint : frame.checkpoints) {
    AutotestCheckpointReport checkpointReport{};
    checkpointReport.id = checkpoint->id;
    checkpointReport.frame = checkpoint->frame;
    const size_t checkpointReportIndex = report.checkpoints.size();
    for (const AutotestReadoutRequest &readout : checkpoint->readouts) {
      const uint64_t requestId = nextReadoutRequestId++;
      AutotestReadoutReport readoutReport{};
      readoutReport.checkpointId = checkpoint->id;
      readoutReport.id = readout.id;
      readoutReport.type = readout.type;
      readoutReport.requestId = requestId;
      readoutReport.requestFrame = checkpoint->frame;
      readoutReport.required = readout.required;
      readoutReport.status = "pending";
      readoutReport.statusReason = "awaiting_result";
      bool scheduled = false;
      if (readout.type == "shadowInspect") {
        if (!shadowInspectScheduled) {
          frameContext.shadowInspectRequest = ShadowInspectRequest{
              .x = readout.x,
              .y = readout.y,
              .requestId = requestId,
          };
          shadowInspectScheduled = true;
          scheduled = true;
        }
      } else if (readout.type == "opaquePick") {
        if (!opaquePickScheduled) {
          frameContext.opaquePickRequest = OpaquePickRequest{
              .x = readout.x,
              .y = readout.y,
              .requestId = requestId,
          };
          opaquePickScheduled = true;
          scheduled = true;
        }
      }
      if (!scheduled) {
        readoutReport.status = "invalid_precondition";
        readoutReport.statusReason =
            "readout_channel_already_scheduled_for_frame";
        if (readout.required && result.exitCode == AutotestExitCode::Success) {
          result.exitCode = AutotestExitCode::InvalidInput;
          result.message = "autotest readout channel conflict";
        }
      }
      checkpointReport.readouts.push_back(std::move(readoutReport));
      if (scheduled) {
        pendingReadouts.push_back(PendingReadout{
            .requestId = requestId,
            .request = &readout,
            .checkpointReportIndex = checkpointReportIndex,
            .readoutReportIndex = checkpointReport.readouts.size() - 1u,
            .requestFrame = checkpoint->frame,
        });
      }
    }
    report.checkpoints.push_back(std::move(checkpointReport));
    work.push_back(CheckpointFrameWork{.checkpoint = checkpoint,
                                       .reportIndex = checkpointReportIndex});
  }
  return work;
}

[[nodiscard]] AutotestReport makeInitialReport(
    const AutotestCase &testCase, const AutotestRunOptions &options,
    const std::filesystem::path &artifactDir,
    const std::filesystem::path &caseDir, const std::filesystem::path &htmlPath,
    std::string_view backend, std::string_view backendSource,
    std::string_view presentMode, std::string_view presentSource) {
  AutotestReport report{};
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = options.command;
  report.testCase = testCase;
  report.run.fixedDeltaSeconds = testCase.fixedDeltaSeconds;
  report.run.warmupFrames = testCase.warmupFrames;
  report.run.endFrame = testCase.endFrame;
  report.run.readoutDrainTimeoutMs = 1000u;
  report.run.captureSynchronization = "wait_idle";
  report.artifacts.artifactDir = artifactDir;
  report.artifacts.caseDir = caseDir;
  report.artifacts.caseHtml = htmlPath;
  report.environment = collectAutotestEnvironment(
      backend, backendSource, presentMode, presentSource, options.windowMode,
      options.windowMode);
  report.environment.renderGraphWorkerCount = testCase.renderGraph.workerCount;
  report.environment.renderGraphParallelCompile =
      testCase.renderGraph.parallelCompile;
  report.environment.renderGraphParallelRecording =
      testCase.renderGraph.parallelRecording;
  report.reproduceCommand = "nuri-autotest run --case " + testCase.id +
                            " --baseline-profile " + options.baselineProfile;
  return report;
}

void initializeDryRunCheckpoints(AutotestReport &report) {
  report.checkpoints.clear();
  report.metricWindows.clear();
  for (const AutotestCheckpoint &checkpoint : report.testCase.checkpoints) {
    AutotestCheckpointReport checkpointReport{};
    checkpointReport.id = checkpoint.id;
    checkpointReport.frame = checkpoint.frame;
    checkpointReport.warnings.push_back(
        "dry run: renderer was not initialized");
    for (const AutotestCaptureTarget &target : checkpoint.captures) {
      nuri::tools::snapshot::SnapshotCaptureReport snapshot{};
      snapshot.target = target.target;
      snapshot.artifactStem = target.target;
      snapshot.profile = target.profile;
      snapshot.required = target.required;
      snapshot.status = "environment_unavailable";
      snapshot.statusReason = "dry_run";
      checkpointReport.captures.push_back(AutotestCaptureReport{
          .checkpointId = checkpoint.id,
          .checkpointFrame = checkpoint.frame,
          .target = target.target,
          .profile = target.profile,
          .required = target.required,
          .compare = target.compare,
          .snapshot = std::move(snapshot),
      });
    }
    for (const AutotestReadoutRequest &readout : checkpoint.readouts) {
      checkpointReport.readouts.push_back(AutotestReadoutReport{
          .checkpointId = checkpoint.id,
          .id = readout.id,
          .type = readout.type,
          .requestId = 0u,
          .requestFrame = checkpoint.frame,
          .resultFrame = 0u,
          .required = readout.required,
          .status = "environment_unavailable",
          .statusReason = "dry_run",
      });
    }
    report.checkpoints.push_back(std::move(checkpointReport));
  }
  for (const AutotestMetricWindow &window : report.testCase.metricWindows) {
    AutotestMetricWindowReport windowReport{};
    windowReport.id = window.id;
    windowReport.startFrame = window.startFrame;
    windowReport.endFrame = window.endFrame;
    windowReport.warnings.push_back("dry run: metrics were not collected");
    report.metricWindows.push_back(std::move(windowReport));
  }
}

void writeReports(AutotestRunResult &result, AutotestReport &report,
                  const std::filesystem::path &jsonPath,
                  const std::filesystem::path &htmlPath) {
  result.reportPath = jsonPath;
  result.htmlPath = htmlPath;
  report.artifacts.caseHtml = htmlPath;
  auto writeJson = writeAutotestReportFile(report, jsonPath);
  if (writeJson.hasError() && result.exitCode == AutotestExitCode::Success) {
    result.exitCode = AutotestExitCode::RuntimeError;
    result.message = writeJson.error();
  }
  auto writeHtml = writeAutotestHtmlReportFile(report, htmlPath);
  if (writeHtml.hasError() && result.exitCode == AutotestExitCode::Success) {
    result.exitCode = AutotestExitCode::RuntimeError;
    result.message = writeHtml.error();
  }
}

[[nodiscard]] std::filesystem::path
baselineCheckpointDir(const AutotestCase &testCase,
                      const AutotestCheckpoint &checkpoint,
                      std::string_view baselineProfile) {
  return nuri::tools::snapshot::defaultSnapshotBaselineRoot() /
         baselineProfile / "autotests" / testCase.suite / testCase.id /
         "checkpoints" / checkpointDirName(checkpoint);
}

[[nodiscard]] AutotestExitCode compareCheckpointCapture(
    const AutotestCase &testCase, const AutotestCheckpoint &checkpoint,
    const AutotestEnvironment &environment,
    nuri::tools::snapshot::SnapshotCaptureReport &capture,
    const std::filesystem::path &autotestCaseDir,
    const std::filesystem::path &checkpointDir,
    std::string_view baselineProfile, std::string &message) {
  if (capture.actual.empty() || capture.status == "missing_capture_point" ||
      capture.status == "unsupported_format" ||
      capture.status == "readback_error") {
    return AutotestExitCode::Success;
  }
  const std::filesystem::path actual = autotestCaseDir / capture.actual;
  const bool usePreview =
      actual.extension() == ".nuri_tex" || actual.extension().empty();
  const std::filesystem::path expected =
      baselineCheckpointDir(testCase, checkpoint, baselineProfile) /
      (usePreview ? capture.target + "_preview.png"
                  : capture.target + actual.extension().string());
  capture.expected = expected;
  if (!std::filesystem::exists(expected)) {
    capture.status = "missing_baseline";
    capture.statusReason = "baseline_artifact_missing";
    message = "autotest baseline missing";
    return AutotestExitCode::MissingBaseline;
  }
  const std::filesystem::path baselineCaseDir =
      nuri::tools::snapshot::defaultSnapshotBaselineRoot() / baselineProfile /
      "autotests" / testCase.suite / testCase.id;
  auto metadata = validateAutotestBaselineMetadataFile(
      testCase, environment, baselineCaseDir, baselineProfile);
  if (metadata.hasError()) {
    capture.status = "invalid_precondition";
    capture.statusReason = "baseline_metadata_invalid";
    message = metadata.error();
    return AutotestExitCode::InvalidInput;
  }
  if (!metadata.value().compatible) {
    capture.status = "invalid_precondition";
    capture.statusReason = metadata.value().errors.empty()
                               ? "baseline_metadata_incompatible"
                               : metadata.value().errors.front();
    message = "autotest baseline metadata incompatible";
    return AutotestExitCode::InvalidInput;
  }
  auto actualImage = nuri::tools::snapshot::readSnapshotImageFile(
      usePreview ? autotestCaseDir / capture.preview : actual);
  auto expectedImage = nuri::tools::snapshot::readSnapshotImageFile(expected);
  if (actualImage.hasError() || expectedImage.hasError()) {
    capture.status = "runtime_error";
    capture.statusReason = "failed_to_load_compare_images";
    message =
        actualImage.hasError() ? actualImage.error() : expectedImage.error();
    return AutotestExitCode::RuntimeError;
  }
  const nuri::tools::snapshot::SnapshotCompareProfile profile =
      nuri::tools::snapshot::builtinSnapshotCompareProfile(capture.profile);
  nuri::tools::snapshot::SnapshotCompareResult comparison =
      nuri::tools::snapshot::compareSnapshotImages(
          actualImage.value(), expectedImage.value(), profile);
  capture.metrics = comparison.metrics;
  capture.failedThresholds = comparison.failedThresholds;
  if (!comparison.compatible) {
    capture.status = "runtime_error";
    capture.statusReason = "comparison_incompatible";
    message = "autotest comparison incompatible";
    return AutotestExitCode::InvalidInput;
  }
  if (comparison.passed) {
    capture.status = "pass";
    capture.statusReason = "within_thresholds";
    return AutotestExitCode::Success;
  }
  capture.status = "fail";
  capture.statusReason = "thresholds_failed";
  const std::filesystem::path diffPath =
      checkpointDir / (capture.target + "_diff.png");
  auto diff = nuri::tools::snapshot::writeSnapshotDiffPng(
      actualImage.value(), expectedImage.value(), diffPath);
  if (!diff.hasError()) {
    capture.diff = relativeToCaseDir(autotestCaseDir, diffPath);
  }
  message = "autotest mismatch";
  return AutotestExitCode::ScenarioFailure;
}

} // namespace

Result<std::string, std::string>
formatAutotestCaseListJson(const std::vector<AutotestCase> &cases,
                           std::string_view suite) {
  std::ostringstream out;
  out << "{\n  \"cases\": [\n";
  bool first = true;
  for (const AutotestCase &testCase : cases) {
    if (!suite.empty() && testCase.suite != suite) {
      continue;
    }
    if (!first) {
      out << ",\n";
    }
    first = false;
    out << "    {\"id\": \"" << jsonEscape(testCase.id) << "\", \"suite\": \""
        << jsonEscape(testCase.suite) << "\", \"description\": \""
        << jsonEscape(testCase.description)
        << "\", \"checkpoints\": " << testCase.checkpoints.size() << "}";
  }
  out << "\n  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string formatAutotestCaseListText(const std::vector<AutotestCase> &cases,
                                       std::string_view suite) {
  std::ostringstream out;
  for (const AutotestCase &testCase : cases) {
    if (!suite.empty() && testCase.suite != suite) {
      continue;
    }
    out << testCase.id << " [" << testCase.suite << "] " << testCase.description
        << "\n";
  }
  return out.str();
}

Result<std::string, std::string>
formatAutotestCaseExplanationJson(const AutotestCase &testCase) {
  std::ostringstream out;
  out << "{\n"
      << "  \"id\": \"" << jsonEscape(testCase.id) << "\",\n"
      << "  \"suite\": \"" << jsonEscape(testCase.suite) << "\",\n"
      << "  \"description\": \"" << jsonEscape(testCase.description) << "\",\n"
      << "  \"sceneKind\": \"" << jsonEscape(testCase.scene.kind) << "\",\n"
      << "  \"backend\": \"" << jsonEscape(testCase.backend) << "\",\n"
      << "  \"endFrame\": " << testCase.endFrame << ",\n"
      << "  \"checkpoints\": [\n";
  for (size_t i = 0u; i < testCase.checkpoints.size(); ++i) {
    const AutotestCheckpoint &checkpoint = testCase.checkpoints[i];
    if (i != 0u) {
      out << ",\n";
    }
    out << "    {\"id\": \"" << jsonEscape(checkpoint.id)
        << "\", \"frame\": " << checkpoint.frame
        << ", \"captures\": " << checkpoint.captures.size()
        << ", \"readouts\": " << checkpoint.readouts.size()
        << ", \"assertions\": " << checkpoint.assertions.size() << "}";
  }
  out << "\n  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string formatAutotestCaseExplanationText(const AutotestCase &testCase) {
  std::ostringstream out;
  out << testCase.id << "\n"
      << "suite: " << testCase.suite << "\n"
      << "description: " << testCase.description << "\n"
      << "scene: " << testCase.scene.kind << "\n"
      << "backend: " << testCase.backend << "\n"
      << "resolution: " << testCase.resolution[0] << "x"
      << testCase.resolution[1] << "\n"
      << "frames: warmup=" << testCase.warmupFrames
      << " end=" << testCase.endFrame << "\n"
      << "checkpoints: " << testCase.checkpoints.size() << "\n";
  for (const AutotestCheckpoint &checkpoint : testCase.checkpoints) {
    out << "  " << checkpoint.id << " frame=" << checkpoint.frame
        << " captures=" << checkpoint.captures.size()
        << " readouts=" << checkpoint.readouts.size()
        << " assertions=" << checkpoint.assertions.size() << "\n";
  }
  return out.str();
}

Result<std::string, std::string>
formatAutotestEffectiveConfigJson(const AutotestCase &testCase,
                                  const AutotestRunOptions &options) {
  std::string backendSource;
  const std::string backend = resolveBackendName(testCase, backendSource);
  std::string presentSource;
  const std::string present = resolvePresentMode(testCase, presentSource);
  std::ostringstream out;
  out << "{\n"
      << "  \"case\": \"" << jsonEscape(testCase.id) << "\",\n"
      << "  \"backend\": \"" << jsonEscape(backend) << "\",\n"
      << "  \"backendSource\": \"" << jsonEscape(backendSource) << "\",\n"
      << "  \"presentMode\": \"" << jsonEscape(present) << "\",\n"
      << "  \"presentModeSource\": \"" << jsonEscape(presentSource) << "\",\n"
      << "  \"windowMode\": \"" << jsonEscape(options.windowMode) << "\",\n"
      << "  \"artifactDir\": \""
      << jsonEscape(options.artifactDir.generic_string()) << "\"\n"
      << "}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

AutotestRunResult runAutotestCase(AutotestCase testCase,
                                  const AutotestRunOptions &options) {
  AutotestRunResult result{};
  std::string backendSource;
  const std::string backend = resolveBackendName(testCase, backendSource);
  std::string presentSource;
  const std::string presentMode = resolvePresentMode(testCase, presentSource);
  const std::filesystem::path artifactDir =
      options.artifactDir.empty() ? autotestRepoRoot() / "artifacts" /
                                        "autotests" / utcTimestampForPath()
                                  : options.artifactDir;
  const std::filesystem::path caseDir = artifactDir / "cases" / testCase.id;
  const std::filesystem::path reportPath =
      options.jsonOut.empty() ? caseDir / "report.json" : options.jsonOut;
  const std::filesystem::path htmlPath =
      options.htmlOut.empty() ? caseDir / "report.html" : options.htmlOut;
  AutotestReport report =
      makeInitialReport(testCase, options, artifactDir, caseDir, htmlPath,
                        backend, backendSource, presentMode, presentSource);

  auto plan = compileAutotestTimeline(testCase);
  if (plan.hasError()) {
    result.exitCode = AutotestExitCode::InvalidInput;
    result.message = plan.error();
    report.errors.push_back(result.message);
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }

  std::string requirementMessage;
  auto requirements = checkRequirements(testCase, backend, options.windowMode,
                                        report.warnings, requirementMessage);
  if (requirements.hasError()) {
    result.exitCode = requirements.error();
    result.message = requirementMessage;
    report.run.validForComparison = false;
    report.warnings.push_back(requirementMessage);
    initializeDryRunCheckpoints(report);
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }

  if (options.dryRun) {
    result.exitCode = AutotestExitCode::Success;
    result.message = "dry run succeeded";
    report.run.validForComparison = false;
    initializeDryRunCheckpoints(report);
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }

  std::error_code cleanupError;
  std::filesystem::remove_all(caseDir / "checkpoints", cleanupError);
  if (cleanupError) {
    result.exitCode = AutotestExitCode::RuntimeError;
    result.message =
        "failed to clean autotest artifacts: " + cleanupError.message();
    report.errors.push_back(result.message);
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }

  std::map<uint64_t, std::map<std::string, double>> frameMeasurements;
  std::vector<AutotestFramePlan> frames = std::move(plan.value());
  if (caseHasReadouts(testCase) && !frames.empty()) {
    const AutotestFramePlan lastFrame = frames.back();
    for (uint32_t i = 1u; i <= kReadoutDrainFrameLimit; ++i) {
      frames.push_back(AutotestFramePlan{
          .frame = lastFrame.frame + i,
          .camera = lastFrame.camera,
      });
    }
  }
  std::vector<PendingReadout> pendingReadouts;
  uint64_t nextReadoutRequestId = 1u;
  try {
    auto runtimeResult = nuri::tools::runtime::createToolRendererRuntime(
        makeToolRuntimeDesc(testCase, backend, presentMode));
    if (runtimeResult.hasError()) {
      result.exitCode = AutotestExitCode::EnvironmentUnavailable;
      result.message = runtimeResult.error();
      report.warnings.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    std::unique_ptr<nuri::tools::runtime::ToolRendererRuntime> runtime =
        std::move(runtimeResult.value());
    report.environment.swapchainImageCount = runtime->swapchainImageCount();

    double timeSeconds = 0.0;
    for (const AutotestFramePlan &frame : frames) {
      NURI_PROFILER_FRAME("AutotestFrame");
      if (frame.frame > testCase.endFrame && pendingReadouts.empty()) {
        break;
      }
      runtime->window().pollEvents();
      auto commitResult = runtime->commitScene();
      if (commitResult.hasError()) {
        result.exitCode = AutotestExitCode::RuntimeError;
        result.message = commitResult.error();
        report.errors.push_back(result.message);
        break;
      }
      RenderSettings settings = frame.settings;
      sanitizeAutotestRenderSettings(settings);
      if (frame.resetTemporalHistory) {
        settings.antiAliasing.debug.resetHistoryRequested = true;
      }
      const Camera camera = nuri::tools::runtime::makeToolCamera(
          makeToolCameraDesc(frame.camera));
      nuri::tools::runtime::buildToolFrameContext(
          runtime->frameContext(), runtime->scene(), runtime->renderer(),
          settings, runtime->cameraHistory(), camera,
          nuri::tools::runtime::ToolFrameDesc{
              .frameIndex = frame.frame,
              .timeSeconds = timeSeconds,
              .deltaSeconds = testCase.fixedDeltaSeconds,
              .width = testCase.resolution[0],
              .height = testCase.resolution[1],
              .cameraCutRequested = frame.cameraCut,
          });
      runtime->frameContext().captureRequests.clear();
      runtime->frameContext().opaquePickRequest.reset();
      runtime->frameContext().shadowInspectRequest.reset();
      for (const AutotestCheckpoint *checkpoint : frame.checkpoints) {
        for (const AutotestCaptureTarget &capture : checkpoint->captures) {
          runtime->frameContext().captureRequests.request(capture.target);
        }
      }
      std::vector<CheckpointFrameWork> checkpointWork =
          beginCheckpointReportsForFrame(runtime->frameContext(), frame, report,
                                         pendingReadouts, nextReadoutRequestId,
                                         result);

      auto renderResult = runtime->renderer().render(runtime->pipeline(),
                                                     runtime->frameContext());
      if (renderResult.hasError()) {
        result.exitCode = AutotestExitCode::RuntimeError;
        result.message = renderResult.error();
        report.errors.push_back(result.message);
        break;
      }
      ++report.run.renderedFrames;
      if (frame.frame > testCase.endFrame) {
        ++report.run.readoutDrainFrames;
      }

      std::map<std::string, double> measurements;
      flattenAutotestRendererMetrics(measurements,
                                     runtime->frameContext().metrics);
      frameMeasurements[frame.frame] = std::move(measurements);
      drainAutotestGpuTimings(runtime->gpu(), frameMeasurements);
      resolvePendingReadoutsForFrame(runtime->frameContext(), frame.frame,
                                     pendingReadouts, report, result);

      if (!frame.checkpoints.empty()) {
        runtime->gpu().waitIdle();
        drainAutotestGpuTimings(runtime->gpu(), frameMeasurements);
      }
      for (const CheckpointFrameWork &work : checkpointWork) {
        const AutotestCheckpoint *checkpoint = work.checkpoint;
        AutotestCheckpointReport &checkpointReport =
            report.checkpoints[work.reportIndex];
        checkpointReport.measurements = frameMeasurements[frame.frame];
        const std::filesystem::path checkpointDir =
            caseDir / "checkpoints" / checkpointDirName(*checkpoint);
        const std::vector<nuri::tools::snapshot::SnapshotCaptureTarget>
            targets = makeSnapshotCaptureTargets(*checkpoint);
        auto captures = nuri::tools::snapshot::writeSnapshotCaptureArtifacts(
            runtime->gpu(), runtime->frameContext(), targets, caseDir,
            checkpointDir);
        if (captures.hasError()) {
          result.exitCode = AutotestExitCode::RuntimeError;
          result.message = captures.error();
          checkpointReport.errors.push_back(result.message);
          continue;
        }
        nuri::tools::snapshot::SnapshotCaptureArtifactResult captureArtifacts =
            std::move(captures.value());
        if (result.exitCode == AutotestExitCode::Success) {
          if (captureArtifacts.missingRequiredCapture) {
            result.exitCode = AutotestExitCode::EnvironmentUnavailable;
            result.message = "required capture point missing";
          } else if (captureArtifacts.unsupportedRequiredCapture) {
            result.exitCode = AutotestExitCode::EnvironmentUnavailable;
            result.message = "required capture format unsupported";
          } else if (captureArtifacts.readbackFailedRequiredCapture) {
            result.exitCode = AutotestExitCode::RuntimeError;
            result.message = "required capture readback failed";
          }
        }
        for (nuri::tools::snapshot::SnapshotCaptureReport &capture :
             captureArtifacts.captures) {
          const AutotestCaptureTarget *target =
              findAutotestCaptureTarget(*checkpoint, capture.target);
          const bool compare = target != nullptr && target->compare;
          if (compare && result.exitCode != AutotestExitCode::RuntimeError) {
            std::string compareMessage;
            const AutotestExitCode compareCode = compareCheckpointCapture(
                testCase, *checkpoint, report.environment, capture, caseDir,
                checkpointDir, options.baselineProfile, compareMessage);
            if (static_cast<int>(compareCode) >
                static_cast<int>(result.exitCode)) {
              result.exitCode = compareCode;
              result.message = compareMessage;
            }
          }
          checkpointReport.captures.push_back(AutotestCaptureReport{
              .checkpointId = checkpoint->id,
              .checkpointFrame = checkpoint->frame,
              .target = capture.target,
              .profile = capture.profile,
              .required = capture.required,
              .compare = compare,
              .snapshot = std::move(capture),
          });
        }
        checkpointReport.assertions = evaluateAutotestAssertions(
            checkpoint->assertions, checkpointReport.measurements);
        applyAssertionExitStatus(checkpointReport.assertions, result,
                                 "autotest assertion input unavailable",
                                 "autotest assertion failed");
      }
      timeSeconds += testCase.fixedDeltaSeconds;
      if (result.exitCode == AutotestExitCode::RuntimeError) {
        break;
      }
    }
    runtime->gpu().waitIdle();
    drainAutotestGpuTimings(runtime->gpu(), frameMeasurements);
  } catch (const std::exception &ex) {
    result.exitCode = AutotestExitCode::RuntimeError;
    result.message = ex.what();
    report.errors.push_back(result.message);
  }
  markPendingReadoutsMissing(pendingReadouts, report, result);

  for (const AutotestMetricWindow &window : testCase.metricWindows) {
    AutotestMetricWindowReport windowReport{};
    windowReport.id = window.id;
    windowReport.startFrame = window.startFrame;
    windowReport.endFrame = window.endFrame;
    windowReport.assertions =
        evaluateAutotestMetricWindowAssertions(window, frameMeasurements);
    applyAssertionExitStatus(windowReport.assertions, result,
                             "autotest metric window input unavailable",
                             "autotest metric window assertion failed");
    report.metricWindows.push_back(std::move(windowReport));
  }
  populateUnavailableMetrics(report);
  if (options.verboseFrames) {
    populateVerboseFrames(report, frameMeasurements);
  }

  auto metadataWritten =
      writeAutotestRecordMetadataFile(report, caseDir, options.baselineProfile);
  if (metadataWritten.hasError() &&
      result.exitCode == AutotestExitCode::Success) {
    result.exitCode = AutotestExitCode::RuntimeError;
    result.message = metadataWritten.error();
    report.errors.push_back(result.message);
  }
  if (result.exitCode == AutotestExitCode::Success) {
    result.message = "autotest run complete";
  } else if (result.message.empty()) {
    result.message = autotestExitCodeName(result.exitCode);
  }
  writeReports(result, report, reportPath, htmlPath);
  result.report = std::move(report);
  return result;
}

AutotestSuiteRunResult runAutotestSuite(std::vector<AutotestCase> testCases,
                                        std::string_view suite,
                                        const AutotestRunOptions &options) {
  AutotestSuiteRunResult suiteResult{};
  const std::filesystem::path artifactDir =
      options.artifactDir.empty() ? autotestRepoRoot() / "artifacts" /
                                        "autotests" / utcTimestampForPath()
                                  : options.artifactDir;
  for (AutotestCase &testCase : testCases) {
    if (testCase.suite != suite) {
      continue;
    }
    AutotestRunOptions caseOptions = options;
    caseOptions.artifactDir = artifactDir;
    caseOptions.jsonOut.clear();
    caseOptions.htmlOut.clear();
    AutotestRunResult result =
        runAutotestCase(std::move(testCase), caseOptions);
    if (static_cast<int>(result.exitCode) >
        static_cast<int>(suiteResult.exitCode)) {
      suiteResult.exitCode = result.exitCode;
    }
    suiteResult.caseResults.push_back(std::move(result));
  }
  suiteResult.reportPath =
      options.jsonOut.empty() ? artifactDir / "run.json" : options.jsonOut;
  suiteResult.htmlPath =
      options.htmlOut.empty() ? artifactDir / "index.html" : options.htmlOut;
  if (!suiteResult.reportPath.parent_path().empty()) {
    std::filesystem::create_directories(suiteResult.reportPath.parent_path());
  }
  std::ofstream json(suiteResult.reportPath, std::ios::binary);
  json << "{\n  \"kind\": \"nuri.autotest.suite_report\",\n"
          "  \"caseReports\": [\n";
  for (size_t i = 0u; i < suiteResult.caseResults.size(); ++i) {
    if (i != 0u) {
      json << ",\n";
    }
    json << "    \""
         << jsonEscape(suiteResult.caseResults[i].reportPath.generic_string())
         << "\"";
  }
  json << "\n  ]\n}\n";
  std::vector<AutotestReport> reports;
  reports.reserve(suiteResult.caseResults.size());
  for (const AutotestRunResult &caseResult : suiteResult.caseResults) {
    reports.push_back(caseResult.report);
  }
  (void)writeAutotestSuiteHtmlFile(reports, suite, suiteResult.htmlPath);
  suiteResult.message = "suite run complete";
  return suiteResult;
}

} // namespace nuri::tools::autotest
