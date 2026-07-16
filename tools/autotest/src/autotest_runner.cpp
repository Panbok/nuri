#include "nuri/tools/autotest/autotest_runner.h"

#include "nuri/tools/autotest/autotest_manifest.h"
#include "nuri/tools/autotest/autotest_motion_oracle.h"
#include "nuri/tools/autotest/autotest_quality_oracle.h"
#include "nuri/tools/autotest/autotest_record.h"
#include "nuri/tools/autotest/autotest_timeline.h"
#include "nuri/tools/core/baseline_profile.h"
#include "nuri/tools/core/case_catalog.h"
#include "nuri/tools/core/fingerprint.h"
#include "nuri/tools/core/run_workspace.h"
#include "nuri/tools/runtime/render_tool_runtime.h"
#include "nuri/tools/snapshot/snapshot_baseline.h"
#include "nuri/tools/snapshot/snapshot_capture_artifacts.h"
#include "nuri/tools/snapshot/snapshot_compare.h"
#include "nuri/tools/snapshot/snapshot_image.h"

#include "nuri/core/profiling.h"
#include "nuri/resources/gpu/resource_manager.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <thread>

namespace nuri::tools::autotest {
namespace {

constexpr uint32_t kMaxPreFrameSleepMs = 60000u;

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

struct ResolvedWindowMode {
  std::string value{};
  std::string source{};
};

void evaluateAutotestBaselineProfile(AutotestReport &report) {
  auto profile = nuri::tools::core::loadBaselineProfile(
      autotestRepoRoot() / "tools" / "profiles", report.baselineProfile);
  if (profile.hasError()) {
    report.baselineProfileCompatible = false;
    report.baselineProfileIncompatibilityReasons = {profile.error()};
    return;
  }
  const std::string profiling = report.environment.tracyEnabled ? "cpu" : "off";
  const auto compatibility = nuri::tools::core::evaluateBaselineProfile(
      profile.value(),
      nuri::tools::core::BaselineProfileObservedEnvironment{
          .os = report.environment.osName,
          .backend = report.environment.gpuBackend,
          .backendSource = report.environment.gpuBackendSource,
          .windowMode = report.environment.resolvedWindowMode,
          .windowVisible = report.environment.windowVisible,
          .gpuVendorId = report.environment.gpuVendorId,
          .gpuDeviceId = report.environment.gpuDeviceId,
          .driver = report.environment.gpuDriverVersion,
          .presentMode = report.environment.resolvedPresentMode,
          .profiling = profiling,
          .devChecks = report.environment.devChecks,
          .dirtyTree = report.environment.dirty,
      });
  report.baselineProfileCompatible = compatibility.compatible;
  report.baselineProfileIncompatibilityReasons = compatibility.reasons;
}

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
  if (testCase.backend != "default") {
    source = "manifest";
    return testCase.backend;
  }
  source = "default";
  return "nvrhi";
}

[[nodiscard]] uint32_t autotestPreFrameSleepMs() {
  const std::string value =
      readProcessEnvironment("NURI_AUTOTEST_PRE_FRAME_SLEEP_MS");
  if (value.empty()) {
    return 0u;
  }

  uint32_t milliseconds = 0u;
  const char *begin = value.data();
  const char *end = value.data() + value.size();
  const std::from_chars_result parsed =
      std::from_chars(begin, end, milliseconds);
  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    return 0u;
  }
  return std::min(milliseconds, kMaxPreFrameSleepMs);
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

[[nodiscard]] ResolvedWindowMode
resolveWindowMode(const AutotestCase &testCase,
                  const AutotestRunOptions &options) {
  if (!options.windowMode.empty()) {
    return ResolvedWindowMode{.value = options.windowMode, .source = "cli"};
  }
  return ResolvedWindowMode{.value = testCase.windowMode, .source = "manifest"};
}

[[nodiscard]] std::string outcomeForExitCode(AutotestExitCode code) {
  switch (code) {
  case AutotestExitCode::Success:
    return "pass";
  case AutotestExitCode::ScenarioFailure:
    return "fail";
  case AutotestExitCode::InvalidInput:
    return "invalid";
  case AutotestExitCode::EnvironmentUnavailable:
    return "unavailable";
  case AutotestExitCode::RuntimeError:
    return "error";
  case AutotestExitCode::MissingBaseline:
    return "missing_baseline";
  }
  return "error";
}

[[nodiscard]] int aggregatePrecedence(AutotestExitCode code) {
  switch (code) {
  case AutotestExitCode::RuntimeError:
    return 5;
  case AutotestExitCode::InvalidInput:
    return 4;
  case AutotestExitCode::EnvironmentUnavailable:
    return 3;
  case AutotestExitCode::MissingBaseline:
    return 2;
  case AutotestExitCode::ScenarioFailure:
    return 1;
  case AutotestExitCode::Success:
    return 0;
  }
  return 5;
}

[[nodiscard]] Result<std::filesystem::path, std::string>
resolveOwnedPath(const std::filesystem::path &root,
                 const std::filesystem::path &relative) {
  if (root.empty() || relative.empty() || relative.is_absolute()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "autotest owned path must be relative to a non-empty root");
  }
  std::error_code ec;
  const std::filesystem::path canonicalRoot =
      std::filesystem::weakly_canonical(root, ec);
  if (ec) {
    return Result<std::filesystem::path, std::string>::makeError(
        "failed to resolve autotest artifact root: " + ec.message());
  }
  const std::filesystem::path candidate =
      std::filesystem::weakly_canonical(canonicalRoot / relative, ec);
  if (ec) {
    return Result<std::filesystem::path, std::string>::makeError(
        "failed to resolve autotest artifact path: " + ec.message());
  }
  auto rootIt = canonicalRoot.begin();
  auto candidateIt = candidate.begin();
  for (; rootIt != canonicalRoot.end() && candidateIt != candidate.end();
       ++rootIt, ++candidateIt) {
    if (*rootIt != *candidateIt) {
      return Result<std::filesystem::path, std::string>::makeError(
          "autotest artifact path escapes its root");
    }
  }
  if (rootIt != canonicalRoot.end() || candidate == canonicalRoot) {
    return Result<std::filesystem::path, std::string>::makeError(
        "autotest artifact path escapes its root");
  }
  return Result<std::filesystem::path, std::string>::makeResult(candidate);
}

[[nodiscard]] Result<bool, AutotestExitCode>
checkRequirements(const AutotestCase &testCase, std::string_view backend,
                  std::string_view windowMode,
                  std::vector<std::string> &warnings, std::string &message) {
  if (backend != "nvrhi") {
    message = "unsupported backend '" + std::string(backend) +
              "'; nvrhi is the only available backend";
    return Result<bool, AutotestExitCode>::makeError(
        AutotestExitCode::InvalidInput);
  }
  if (windowMode == "headless") {
    message = "true offscreen/headless mode is unavailable";
    return Result<bool, AutotestExitCode>::makeError(
        AutotestExitCode::EnvironmentUnavailable);
  }
  if (windowMode == "visible" && !testCase.requirements.allowVisibleWindow) {
    message = "case does not permit visible-window execution";
    return Result<bool, AutotestExitCode>::makeError(
        AutotestExitCode::EnvironmentUnavailable);
  }
  if (!testCase.requirements.backends.empty()) {
    bool supported = false;
    for (const std::string &allowed : testCase.requirements.backends) {
      if (allowed != "default" && allowed != "nvrhi") {
        message = "unsupported backend requirement '" + allowed + "'";
        return Result<bool, AutotestExitCode>::makeError(
            AutotestExitCode::InvalidInput);
      }
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

[[nodiscard]] nuri::tools::runtime::ToolEnvironmentTextureDesc
makeToolEnvironmentTextureDesc(
    const AutotestEnvironmentTextureConfig &texture) {
  return nuri::tools::runtime::ToolEnvironmentTextureDesc{
      .enabled = texture.enabled,
      .required = texture.required,
      .pathBase = texture.pathBase,
      .path = texture.path,
      .kind = texture.kind,
      .debugName = texture.debugName,
  };
}

[[nodiscard]] nuri::tools::runtime::ToolEnvironmentDesc
makeToolEnvironmentDesc(const AutotestEnvironmentConfig &environment) {
  return nuri::tools::runtime::ToolEnvironmentDesc{
      .cubemap = makeToolEnvironmentTextureDesc(environment.cubemap),
      .irradiance = makeToolEnvironmentTextureDesc(environment.irradiance),
      .prefilteredGgx =
          makeToolEnvironmentTextureDesc(environment.prefilteredGgx),
      .prefilteredCharlie =
          makeToolEnvironmentTextureDesc(environment.prefilteredCharlie),
      .brdfLut = makeToolEnvironmentTextureDesc(environment.brdfLut),
  };
}

[[nodiscard]] nuri::tools::runtime::ToolRuntimeDesc
makeToolRuntimeDesc(const AutotestCase &testCase, std::string_view presentMode,
                    std::string_view windowMode) {
  nuri::tools::runtime::ToolRuntimeDesc desc{};
  desc.title = "nuri-autotest " + testCase.id;
  desc.presentMode = std::string(presentMode);
  desc.windowVisible = windowMode == "visible";
  desc.resolution = testCase.resolution;
  desc.renderGraph.workerCount = testCase.renderGraph.workerCount;
  desc.renderGraph.parallelCompile = testCase.renderGraph.parallelCompile;
  desc.renderGraph.parallelRecording = testCase.renderGraph.parallelRecording;
  desc.scene = makeToolSceneDesc(testCase.scene);
  desc.environment = makeToolEnvironmentDesc(testCase.environment);
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
readoutValuesFromOpaquePick(const OpaquePickResult &readout,
                            const RenderFrameContext &frameContext) {
  std::map<std::string, double> values{
      {"hit", readout.hit ? 1.0 : 0.0},
      {"renderableIndex", static_cast<double>(readout.renderableIndex)}};
  if (!readout.hit || frameContext.scene == nullptr ||
      frameContext.resources == nullptr) {
    return values;
  }

  const Renderable *renderable =
      frameContext.scene->renderable(readout.renderableIndex);
  if (renderable == nullptr) {
    return values;
  }
  values.emplace("nodeIndex", static_cast<double>(indexOf(renderable->node)));

  const ModelRecord *modelRecord =
      frameContext.resources->tryGet(renderable->model);
  if (modelRecord == nullptr || !modelRecord->model) {
    return values;
  }
  const BoundingBox worldBounds =
      modelRecord->model->bounds().getTransformed(renderable->modelMatrix);
  const glm::vec3 worldCenter = worldBounds.getCenter();
  values.emplace("worldCenterX", worldCenter.x);
  values.emplace("worldCenterY", worldCenter.y);
  values.emplace("worldCenterZ", worldCenter.z);
  return values;
}

[[nodiscard]] std::map<std::string, double>
readoutValuesFromShadowInspect(const ShadowInspectResult &readout) {
  return {{"valid", readout.valid ? 1.0 : 0.0},
          {"shadowed",
           readout.valid && readout.receiverCompareDepth > readout.sampledDepth
               ? 1.0
               : 0.0},
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
                readoutValuesFromOpaquePick(*frameContext.opaquePickResult,
                                            frameContext));
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
                                     ? "readout_drain_frame_limit_exceeded"
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
    std::string_view presentMode, std::string_view presentSource,
    const ResolvedWindowMode &windowMode) {
  AutotestReport report{};
  report.baselineProfile = options.baselineProfile;
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = options.command;
  report.testCase = testCase;
  report.run.fixedDeltaSeconds = testCase.fixedDeltaSeconds;
  report.run.warmupFrames = testCase.warmupFrames;
  report.run.endFrame = testCase.endFrame;
  report.run.readoutDrainFrameLimit = kAutotestReadoutDrainFrameLimit;
  report.run.readoutDrainTimeoutMs = 0u;
  report.run.requestedWindowMode = windowMode.value;
  report.run.resolvedWindowMode = windowMode.value;
  report.run.windowModeSource = windowMode.source;
  report.run.captureSynchronization = "wait_idle";
  report.artifacts.artifactDir = artifactDir;
  report.artifacts.caseDir = caseDir;
  report.artifacts.caseHtml = htmlPath;
  report.environment = collectAutotestEnvironment(
      backend, backendSource, presentMode, presentSource, windowMode.value,
      windowMode.value);
  report.environment.renderGraphWorkerCount = testCase.renderGraph.workerCount;
  report.environment.renderGraphParallelCompile =
      testCase.renderGraph.parallelCompile;
  report.environment.renderGraphParallelRecording =
      testCase.renderGraph.parallelRecording;
  report.reproduceCommand = "nuri-autotest run --case " + testCase.id +
                            " --baseline-profile " + options.baselineProfile;
  report.selection.requested = testCase.id;
  report.selection.selected = 1u;
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
    if (checkpoint.motionOracle.has_value()) {
      const AutotestMotionOracle &oracle = *checkpoint.motionOracle;
      checkpointReport.motionOracle = AutotestMotionOracleReport{
          .status = "unavailable",
          .statusReason = "dry_run",
          .motionTarget = oracle.motionTarget,
          .motionClassTarget = oracle.motionClassTarget,
          .roi = oracle.roi,
          .expectedVelocityPixels = {oracle.expectedVelocityPixels.x,
                                     oracle.expectedVelocityPixels.y},
          .p95ErrorMaxPixels = oracle.p95ErrorMaxPixels,
          .maxErrorMaxPixels = oracle.maxErrorMaxPixels,
      };
    }
    if (checkpoint.qualityOracle.has_value()) {
      const AutotestQualityOracle &oracle = *checkpoint.qualityOracle;
      checkpointReport.qualityOracle = AutotestQualityOracleReport{
          .status = "unavailable",
          .statusReason = oracle.reference.available
                              ? "dry_run"
                              : oracle.reference.unavailableReason,
          .outputTarget = oracle.outputTarget,
          .referencePath = oracle.reference.path.generic_string(),
          .schemaVersion = oracle.schemaVersion,
          .referenceVersion = oracle.reference.version,
          .maskVersion = oracle.mask.has_value() ? oracle.mask->version : 0u,
          .lscale = oracle.lscale,
          .budgets = oracle.budgets,
      };
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
  evaluateAutotestBaselineProfile(report);
  const auto updateOutcome = [&]() {
    report.status = outcomeForExitCode(result.exitCode);
    report.exitCode = result.exitCode;
    report.selection.attempted =
        result.exitCode == AutotestExitCode::InvalidInput ? 0u : 1u;
    report.selection.completed =
        result.exitCode == AutotestExitCode::InvalidInput ? 0u : 1u;
    report.selection.passed =
        result.exitCode == AutotestExitCode::Success ? 1u : 0u;
    report.selection.failed =
        result.exitCode == AutotestExitCode::ScenarioFailure ||
                result.exitCode == AutotestExitCode::RuntimeError ||
                result.exitCode == AutotestExitCode::MissingBaseline
            ? 1u
            : 0u;
    report.selection.unavailable =
        result.exitCode == AutotestExitCode::EnvironmentUnavailable ? 1u : 0u;
    report.selection.notRun =
        result.exitCode == AutotestExitCode::InvalidInput ? 1u : 0u;
  };
  updateOutcome();
  auto writeHtml = writeAutotestHtmlReportFile(report, htmlPath);
  if (writeHtml.hasError()) {
    result.exitCode = AutotestExitCode::RuntimeError;
    result.message = writeHtml.error();
    report.errors.push_back(result.message);
  }
  updateOutcome();
  auto writeJson = writeAutotestReportFile(report, jsonPath);
  if (writeJson.hasError()) {
    result.exitCode = AutotestExitCode::RuntimeError;
    result.message = writeJson.error();
    report.errors.push_back(result.message);
    updateOutcome();
  }
}

[[nodiscard]] std::filesystem::path
baselineCheckpointDir(const AutotestCase &testCase,
                      const AutotestCheckpoint &checkpoint,
                      std::string_view baselineProfile,
                      const std::filesystem::path &baselineRoot) {
  return baselineRoot / baselineProfile / "autotests" / testCase.suite /
         testCase.id / "checkpoints" / checkpointDirName(checkpoint);
}

[[nodiscard]] AutotestExitCode compareCheckpointCapture(
    const AutotestCase &testCase, const AutotestCheckpoint &checkpoint,
    const AutotestEnvironment &environment,
    nuri::tools::snapshot::SnapshotCaptureReport &capture,
    const std::filesystem::path &autotestCaseDir,
    const std::filesystem::path &checkpointDir,
    std::string_view baselineProfile, const std::filesystem::path &baselineRoot,
    std::string &message) {
  if (capture.actual.empty() || capture.status == "missing_capture_point" ||
      capture.status == "unsupported_format" ||
      capture.status == "readback_error") {
    return AutotestExitCode::Success;
  }
  const std::filesystem::path actual = autotestCaseDir / capture.actual;
  const bool usePreview =
      actual.extension() == ".nuri_tex" || actual.extension().empty();
  const std::filesystem::path expected =
      baselineCheckpointDir(testCase, checkpoint, baselineProfile,
                            baselineRoot) /
      (usePreview ? capture.target + "_preview.png"
                  : capture.target + actual.extension().string());
  capture.expected = expected;
  if (!std::filesystem::exists(expected)) {
    capture.status = "missing_baseline";
    capture.statusReason = "baseline_artifact_missing";
    message = "autotest baseline missing";
    return AutotestExitCode::MissingBaseline;
  }
  const std::filesystem::path baselineCaseDir = baselineRoot / baselineProfile /
                                                "autotests" / testCase.suite /
                                                testCase.id;
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

[[nodiscard]] AutotestMotionOracleReport
makeMotionOracleReport(const AutotestMotionOracle &oracle) {
  AutotestMotionOracleReport report{};
  report.motionTarget = oracle.motionTarget;
  report.motionClassTarget = oracle.motionClassTarget;
  report.roi = oracle.roi;
  report.expectedVelocityPixels = {oracle.expectedVelocityPixels.x,
                                   oracle.expectedVelocityPixels.y};
  report.p95ErrorMaxPixels = oracle.p95ErrorMaxPixels;
  report.maxErrorMaxPixels = oracle.maxErrorMaxPixels;
  return report;
}

[[nodiscard]] AutotestQualityOracleReport
makeQualityOracleReport(const AutotestQualityOracle &oracle) {
  AutotestQualityOracleReport report{};
  report.outputTarget = oracle.outputTarget;
  report.referencePath = oracle.reference.path.generic_string();
  report.schemaVersion = oracle.schemaVersion;
  report.referenceVersion = oracle.reference.version;
  report.maskVersion = oracle.mask.has_value() ? oracle.mask->version : 0u;
  report.lscale = oracle.lscale;
  report.budgets = oracle.budgets;
  return report;
}

[[nodiscard]] const AutotestCaptureReport *
findAutotestCaptureReport(const AutotestCheckpointReport &checkpoint,
                          std::string_view target) {
  const auto found =
      std::find_if(checkpoint.captures.begin(), checkpoint.captures.end(),
                   [&](const AutotestCaptureReport &capture) {
                     return capture.target == target;
                   });
  return found == checkpoint.captures.end() ? nullptr : &*found;
}

[[nodiscard]] AutotestExitCode evaluateCheckpointMotionOracle(
    const AutotestCase &testCase, const AutotestCheckpoint &checkpoint,
    const std::filesystem::path &caseDir,
    AutotestCheckpointReport &checkpointReport, std::string &message) {
  if (!checkpoint.motionOracle.has_value()) {
    return AutotestExitCode::Success;
  }
  const AutotestMotionOracle &oracle = *checkpoint.motionOracle;
  AutotestMotionOracleReport oracleReport = makeMotionOracleReport(oracle);
  const AutotestCaptureReport *motionCapture =
      findAutotestCaptureReport(checkpointReport, oracle.motionTarget);
  if (motionCapture == nullptr || motionCapture->snapshot.actual.empty()) {
    oracleReport.status = "unavailable";
    oracleReport.statusReason = "motion_capture_unavailable";
    checkpointReport.motionOracle = std::move(oracleReport);
    message = "motion oracle capture unavailable";
    return AutotestExitCode::EnvironmentUnavailable;
  }
  const std::filesystem::path motionPath =
      caseDir / motionCapture->snapshot.actual;
  if (motionPath.extension() != ".exr") {
    oracleReport.status = "error";
    oracleReport.statusReason = "motion_capture_not_exr";
    checkpointReport.motionOracle = std::move(oracleReport);
    message = "motion oracle requires a motion_vectors EXR artifact";
    return AutotestExitCode::RuntimeError;
  }
  auto motionImage = nuri::tools::snapshot::readSnapshotImageFile(motionPath);
  if (motionImage.hasError()) {
    oracleReport.status = "error";
    oracleReport.statusReason = "motion_capture_decode_failed";
    checkpointReport.motionOracle = std::move(oracleReport);
    message = motionImage.error();
    return AutotestExitCode::RuntimeError;
  }
  if (motionImage.value().width != testCase.resolution[0] ||
      motionImage.value().height != testCase.resolution[1]) {
    oracleReport.status = "error";
    oracleReport.statusReason = "motion_capture_resolution_mismatch";
    checkpointReport.motionOracle = std::move(oracleReport);
    message = "motion oracle capture resolution differs from the manifest";
    return AutotestExitCode::RuntimeError;
  }

  std::optional<nuri::tools::snapshot::SnapshotImage> motionClassImage;
  if (!oracle.motionClassTarget.empty()) {
    const AutotestCaptureReport *classCapture =
        findAutotestCaptureReport(checkpointReport, oracle.motionClassTarget);
    if (classCapture == nullptr || classCapture->snapshot.actual.empty()) {
      oracleReport.status = "unavailable";
      oracleReport.statusReason = "motion_class_capture_unavailable";
      checkpointReport.motionOracle = std::move(oracleReport);
      message = "motion oracle class capture unavailable";
      return AutotestExitCode::EnvironmentUnavailable;
    }
    auto loaded = nuri::tools::snapshot::readSnapshotImageFile(
        caseDir / classCapture->snapshot.actual);
    if (loaded.hasError()) {
      oracleReport.status = "error";
      oracleReport.statusReason = "motion_class_decode_failed";
      checkpointReport.motionOracle = std::move(oracleReport);
      message = loaded.error();
      return AutotestExitCode::RuntimeError;
    }
    motionClassImage = std::move(loaded.value());
  }

  auto evaluated = evaluateAutotestMotionOracle(
      oracle, motionImage.value(),
      motionClassImage.has_value() ? &*motionClassImage : nullptr);
  if (evaluated.hasError()) {
    oracleReport.status = "error";
    oracleReport.statusReason = "motion_oracle_evaluation_failed";
    checkpointReport.motionOracle = std::move(oracleReport);
    message = evaluated.error();
    return AutotestExitCode::RuntimeError;
  }
  checkpointReport.motionOracle = std::move(evaluated.value());
  if (checkpointReport.motionOracle->status == "fail") {
    message = "motion oracle threshold failure";
    return AutotestExitCode::ScenarioFailure;
  }
  return AutotestExitCode::Success;
}

[[nodiscard]] const AutotestCheckpointReport *
findAutotestCheckpointReport(const AutotestReport &report,
                             std::string_view id) {
  const auto found =
      std::find_if(report.checkpoints.begin(), report.checkpoints.end(),
                   [&](const AutotestCheckpointReport &checkpoint) {
                     return checkpoint.id == id;
                   });
  return found == report.checkpoints.end() ? nullptr : &*found;
}

[[nodiscard]] AutotestExitCode evaluateCheckpointQualityOracle(
    const AutotestCheckpoint &checkpoint, const std::filesystem::path &caseDir,
    const AutotestReport &report, AutotestCheckpointReport &checkpointReport,
    std::string &message) {
  if (!checkpoint.qualityOracle.has_value()) {
    return AutotestExitCode::Success;
  }
  const AutotestQualityOracle &oracle = *checkpoint.qualityOracle;
  AutotestQualityOracleReport oracleReport = makeQualityOracleReport(oracle);
  const auto unavailable = [&](std::string reason, std::string description) {
    oracleReport.status = "unavailable";
    oracleReport.statusReason = std::move(reason);
    checkpointReport.qualityOracle = std::move(oracleReport);
    message = std::move(description);
    return AutotestExitCode::EnvironmentUnavailable;
  };
  const auto runtimeError = [&](std::string reason, std::string description) {
    oracleReport.status = "error";
    oracleReport.statusReason = std::move(reason);
    checkpointReport.qualityOracle = std::move(oracleReport);
    message = std::move(description);
    return AutotestExitCode::RuntimeError;
  };
  if (!oracle.reference.available) {
    return unavailable(oracle.reference.unavailableReason,
                       "quality oracle reference is not generated");
  }
  const AutotestCaptureReport *outputCapture =
      findAutotestCaptureReport(checkpointReport, oracle.outputTarget);
  if (outputCapture == nullptr || outputCapture->snapshot.actual.empty()) {
    return unavailable("output_capture_unavailable",
                       "quality oracle output capture unavailable");
  }
  const std::filesystem::path outputPath =
      caseDir / outputCapture->snapshot.actual;
  if (outputPath.extension() != ".exr") {
    return runtimeError("output_capture_not_exr",
                        "quality oracle requires a linear HDR EXR output");
  }
  auto referencePath =
      resolveAutotestPath(oracle.reference.pathBase, oracle.reference.path);
  if (referencePath.hasError()) {
    return unavailable("reference_path_unavailable", referencePath.error());
  }
  if (!std::filesystem::exists(referencePath.value())) {
    oracleReport.status = "unavailable";
    oracleReport.statusReason = "reference_missing";
    checkpointReport.qualityOracle = std::move(oracleReport);
    message = "quality oracle immutable reference is missing";
    return AutotestExitCode::MissingBaseline;
  }
  auto outputImage = nuri::tools::snapshot::readSnapshotImageFile(outputPath);
  auto referenceImage =
      nuri::tools::snapshot::readSnapshotImageFile(referencePath.value());
  if (outputImage.hasError() || referenceImage.hasError()) {
    return runtimeError("hdr_decode_failed", outputImage.hasError()
                                                 ? outputImage.error()
                                                 : referenceImage.error());
  }

  std::optional<nuri::tools::snapshot::SnapshotImage> maskImage;
  if (oracle.mask.has_value()) {
    auto maskPath =
        resolveAutotestPath(oracle.mask->pathBase, oracle.mask->path);
    if (maskPath.hasError() || !std::filesystem::exists(maskPath.value())) {
      return unavailable("mask_unavailable",
                         maskPath.hasError() ? maskPath.error()
                                             : "quality oracle mask missing");
    }
    auto loaded =
        nuri::tools::snapshot::readSnapshotImageFile(maskPath.value());
    if (loaded.hasError()) {
      return runtimeError("mask_decode_failed", loaded.error());
    }
    maskImage = std::move(loaded.value());
  }

  std::optional<nuri::tools::snapshot::SnapshotImage> previousOutputImage;
  std::optional<nuri::tools::snapshot::SnapshotImage> previousReferenceImage;
  std::optional<nuri::tools::snapshot::SnapshotImage> analyticMotionImage;
  std::optional<nuri::tools::snapshot::SnapshotImage> revealMaskImage;
  if (oracle.temporal.has_value()) {
    const AutotestQualityOracleTemporal &temporal = *oracle.temporal;
    if (!temporal.previousReference.available) {
      return unavailable(temporal.previousReference.unavailableReason,
                         "quality oracle previous reference is not generated");
    }
    const AutotestCheckpointReport *previousCheckpoint =
        findAutotestCheckpointReport(report, temporal.previousCheckpoint);
    const AutotestCaptureReport *previousOutput =
        previousCheckpoint != nullptr
            ? findAutotestCaptureReport(*previousCheckpoint,
                                        temporal.previousOutputTarget)
            : nullptr;
    if (previousOutput == nullptr || previousOutput->snapshot.actual.empty()) {
      return unavailable("previous_output_unavailable",
                         "quality oracle previous output unavailable");
    }
    const AutotestCaptureReport *motionCapture =
        findAutotestCaptureReport(checkpointReport, temporal.motionTarget);
    if (motionCapture == nullptr || motionCapture->snapshot.actual.empty()) {
      return unavailable("analytic_motion_unavailable",
                         "quality oracle analytic motion unavailable");
    }
    auto previousReferencePath = resolveAutotestPath(
        temporal.previousReference.pathBase, temporal.previousReference.path);
    auto revealPath = resolveAutotestPath(temporal.revealMask.pathBase,
                                          temporal.revealMask.path);
    if (previousReferencePath.hasError() || revealPath.hasError()) {
      return unavailable("temporal_reference_path_unavailable",
                         previousReferencePath.hasError()
                             ? previousReferencePath.error()
                             : revealPath.error());
    }
    if (!std::filesystem::exists(previousReferencePath.value())) {
      oracleReport.status = "unavailable";
      oracleReport.statusReason = "previous_reference_missing";
      checkpointReport.qualityOracle = std::move(oracleReport);
      message = "quality oracle previous immutable reference is missing";
      return AutotestExitCode::MissingBaseline;
    }
    if (!std::filesystem::exists(revealPath.value())) {
      return unavailable("reveal_mask_unavailable",
                         "quality oracle reveal mask missing");
    }
    auto previousOutputLoaded = nuri::tools::snapshot::readSnapshotImageFile(
        caseDir / previousOutput->snapshot.actual);
    auto previousReferenceLoaded = nuri::tools::snapshot::readSnapshotImageFile(
        previousReferencePath.value());
    auto motionLoaded = nuri::tools::snapshot::readSnapshotImageFile(
        caseDir / motionCapture->snapshot.actual);
    auto revealLoaded =
        nuri::tools::snapshot::readSnapshotImageFile(revealPath.value());
    if (previousOutputLoaded.hasError() || previousReferenceLoaded.hasError() ||
        motionLoaded.hasError() || revealLoaded.hasError()) {
      const std::string error =
          previousOutputLoaded.hasError()
              ? previousOutputLoaded.error()
              : (previousReferenceLoaded.hasError()
                     ? previousReferenceLoaded.error()
                     : (motionLoaded.hasError() ? motionLoaded.error()
                                                : revealLoaded.error()));
      return runtimeError("temporal_input_decode_failed", error);
    }
    previousOutputImage = std::move(previousOutputLoaded.value());
    previousReferenceImage = std::move(previousReferenceLoaded.value());
    analyticMotionImage = std::move(motionLoaded.value());
    revealMaskImage = std::move(revealLoaded.value());
  }

  const AutotestQualityOracleInputs inputs{
      .output = &outputImage.value(),
      .reference = &referenceImage.value(),
      .mask = maskImage.has_value() ? &*maskImage : nullptr,
      .previousOutput =
          previousOutputImage.has_value() ? &*previousOutputImage : nullptr,
      .previousReference = previousReferenceImage.has_value()
                               ? &*previousReferenceImage
                               : nullptr,
      .analyticMotion =
          analyticMotionImage.has_value() ? &*analyticMotionImage : nullptr,
      .revealMask = revealMaskImage.has_value() ? &*revealMaskImage : nullptr,
  };
  auto evaluated = evaluateAutotestQualityOracle(oracle, inputs);
  if (evaluated.hasError()) {
    return runtimeError("quality_oracle_evaluation_failed", evaluated.error());
  }
  checkpointReport.qualityOracle = std::move(evaluated.value());
  if (checkpointReport.qualityOracle->status == "fail") {
    message = "quality oracle threshold failure";
    return AutotestExitCode::ScenarioFailure;
  }
  return AutotestExitCode::Success;
}

} // namespace

Result<std::string, std::string>
formatAutotestCaseListJson(const std::vector<AutotestCase> &cases,
                           std::string_view suite) {
  std::ostringstream out;
  out << "{\n  \"cases\": [\n";
  bool first = true;
  for (const AutotestCase *testCase :
       filterAutotestCasesBySuite(cases, suite)) {
    if (!first) {
      out << ",\n";
    }
    first = false;
    out << "    {\"id\": \"" << jsonEscape(testCase->id) << "\", \"suite\": \""
        << jsonEscape(testCase->suite) << "\", \"description\": \""
        << jsonEscape(testCase->description)
        << "\", \"checkpoints\": " << testCase->checkpoints.size() << "}";
  }
  out << "\n  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string formatAutotestCaseListText(const std::vector<AutotestCase> &cases,
                                       std::string_view suite) {
  std::ostringstream out;
  for (const AutotestCase *testCase :
       filterAutotestCasesBySuite(cases, suite)) {
    out << testCase->id << " [" << testCase->suite << "] "
        << testCase->description << "\n";
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
        << ", \"assertions\": " << checkpoint.assertions.size()
        << ", \"motionOracle\": "
        << (checkpoint.motionOracle.has_value() ? "true" : "false") << "}";
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
        << " assertions=" << checkpoint.assertions.size() << " motionOracle="
        << (checkpoint.motionOracle.has_value() ? "yes" : "no") << "\n";
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
  const ResolvedWindowMode windowMode = resolveWindowMode(testCase, options);
  std::ostringstream out;
  out << "{\n"
      << "  \"case\": \"" << jsonEscape(testCase.id) << "\",\n"
      << "  \"backend\": \"" << jsonEscape(backend) << "\",\n"
      << "  \"backendSource\": \"" << jsonEscape(backendSource) << "\",\n"
      << "  \"presentMode\": \"" << jsonEscape(present) << "\",\n"
      << "  \"presentModeSource\": \"" << jsonEscape(presentSource) << "\",\n"
      << "  \"windowMode\": \"" << jsonEscape(windowMode.value) << "\",\n"
      << "  \"windowModeSource\": \"" << jsonEscape(windowMode.source)
      << "\",\n"
      << "  \"artifactDir\": \""
      << jsonEscape(options.artifactDir.generic_string()) << "\"\n"
      << "}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

AutotestRunResult runAutotestCase(AutotestCase testCase,
                                  const AutotestRunOptions &options) {
  AutotestRunResult result{};
  auto validation = validateAutotestCase(testCase);
  if (validation.hasError()) {
    result.exitCode = AutotestExitCode::InvalidInput;
    result.message = validation.error();
    return result;
  }
  validation = validateAutotestIdentifier(options.baselineProfile,
                                          "baselineProfile", false);
  if (validation.hasError()) {
    result.exitCode = AutotestExitCode::InvalidInput;
    result.message = validation.error();
    return result;
  }
  auto configuredProfile = nuri::tools::core::loadBaselineProfile(
      autotestRepoRoot() / "tools" / "profiles", options.baselineProfile);
  if (configuredProfile.hasError()) {
    result.exitCode = AutotestExitCode::InvalidInput;
    result.message = configuredProfile.error();
    return result;
  }
  const ResolvedWindowMode windowMode = resolveWindowMode(testCase, options);
  if (windowMode.value != "visible" && windowMode.value != "hidden" &&
      windowMode.value != "headless") {
    result.exitCode = AutotestExitCode::InvalidInput;
    result.message = "windowMode must be visible, hidden, or headless";
    return result;
  }
  std::string backendSource;
  const std::string backend = resolveBackendName(testCase, backendSource);
  std::string presentSource;
  const std::string presentMode = resolvePresentMode(testCase, presentSource);
  std::filesystem::path artifactDir = options.artifactDir;
  if (artifactDir.empty()) {
    auto workspace = nuri::tools::core::createRunWorkspace(
        autotestRepoRoot() / "artifacts" / "autotests");
    if (workspace.hasError()) {
      result.exitCode = AutotestExitCode::RuntimeError;
      result.message = workspace.error();
      return result;
    }
    artifactDir = workspace.value().root;
  }
  auto ownedCaseDir = resolveOwnedPath(
      artifactDir, std::filesystem::path("cases") / testCase.id);
  if (ownedCaseDir.hasError()) {
    result.exitCode = AutotestExitCode::InvalidInput;
    result.message = ownedCaseDir.error();
    return result;
  }
  const std::filesystem::path caseDir = ownedCaseDir.value();
  const std::filesystem::path reportPath =
      options.jsonOut.empty() ? caseDir / "report.json" : options.jsonOut;
  const std::filesystem::path htmlPath =
      options.htmlOut.empty() ? caseDir / "report.html" : options.htmlOut;
  AutotestReport report = makeInitialReport(
      testCase, options, artifactDir, caseDir, htmlPath, backend, backendSource,
      presentMode, presentSource, windowMode);

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
  auto requirements = checkRequirements(testCase, backend, windowMode.value,
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

  const bool comparesBaseline =
      std::any_of(testCase.checkpoints.begin(), testCase.checkpoints.end(),
                  [](const AutotestCheckpoint &checkpoint) {
                    return std::any_of(
                        checkpoint.captures.begin(), checkpoint.captures.end(),
                        [](const AutotestCaptureTarget &capture) {
                          return capture.compare;
                        });
                  });
  const std::filesystem::path baselineRoot = options.baselineRoot.empty()
                                                 ? defaultAutotestBaselineRoot()
                                                 : options.baselineRoot;
  if (comparesBaseline) {
    auto governedBaseline =
        verifyAutotestBaseline(testCase, options.baselineProfile, baselineRoot);
    if (governedBaseline.hasError()) {
      const std::filesystem::path baselineCase =
          baselineRoot / options.baselineProfile / "autotests" /
          testCase.suite / testCase.id;
      result.exitCode = std::filesystem::is_directory(baselineCase)
                            ? AutotestExitCode::InvalidInput
                            : AutotestExitCode::MissingBaseline;
      result.message = governedBaseline.error();
      report.errors.push_back(result.message);
      initializeDryRunCheckpoints(report);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
  }

  std::error_code cleanupError;
  auto checkpointsDir =
      resolveOwnedPath(artifactDir, std::filesystem::path("cases") /
                                        testCase.id / "checkpoints");
  if (checkpointsDir.hasError()) {
    result.exitCode = AutotestExitCode::RuntimeError;
    result.message = checkpointsDir.error();
    report.errors.push_back(result.message);
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }
  std::filesystem::remove_all(checkpointsDir.value(), cleanupError);
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
  std::vector<PendingReadout> pendingReadouts;
  uint64_t nextReadoutRequestId = 1u;
  try {
    auto runtimeResult = nuri::tools::runtime::createToolRendererRuntime(
        makeToolRuntimeDesc(testCase, presentMode, windowMode.value));
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
    const GPUAdapterInfo adapter = runtime->gpu().getAdapterInfo();
    report.environment.gpuDeviceName = adapter.name;
    report.environment.gpuVendorId = adapter.vendorId;
    report.environment.gpuDeviceId = adapter.deviceId;
    report.environment.gpuDriverVersion = adapter.driverVersion;
    evaluateAutotestBaselineProfile(report);
    if (comparesBaseline && !report.baselineProfileCompatible) {
      result.exitCode = AutotestExitCode::InvalidInput;
      result.message = "autotest runtime does not match baseline profile";
      report.errors.insert(report.errors.end(),
                           report.baselineProfileIncompatibilityReasons.begin(),
                           report.baselineProfileIncompatibilityReasons.end());
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }

    if (const uint32_t preFrameSleepMs = autotestPreFrameSleepMs();
        preFrameSleepMs > 0u) {
      std::this_thread::sleep_for(std::chrono::milliseconds(preFrameSleepMs));
    }

    double timeSeconds = 0.0;
    bool drainStarted = false;
    std::chrono::steady_clock::time_point drainStartedAt{};
    for (const AutotestFramePlan &frame : frames) {
      NURI_PROFILER_FRAME("AutotestFrame");
      if (frame.drainOnly && pendingReadouts.empty()) {
        break;
      }
      if (frame.drainOnly && !drainStarted) {
        drainStarted = true;
        drainStartedAt = std::chrono::steady_clock::now();
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
          settings, runtime->temporalFrameService(), camera,
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
      if (frame.drainOnly) {
        ++report.run.readoutDrainFrames;
        report.run.readoutDrainElapsedMs = static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - drainStartedAt)
                .count());
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
                checkpointDir, options.baselineProfile, baselineRoot,
                compareMessage);
            if (aggregatePrecedence(compareCode) >
                aggregatePrecedence(result.exitCode)) {
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
        std::string motionOracleMessage;
        const AutotestExitCode motionOracleCode =
            evaluateCheckpointMotionOracle(testCase, *checkpoint, caseDir,
                                           checkpointReport,
                                           motionOracleMessage);
        if (aggregatePrecedence(motionOracleCode) >
            aggregatePrecedence(result.exitCode)) {
          result.exitCode = motionOracleCode;
          result.message = std::move(motionOracleMessage);
        }
        std::string qualityOracleMessage;
        const AutotestExitCode qualityOracleCode =
            evaluateCheckpointQualityOracle(*checkpoint, caseDir, report,
                                            checkpointReport,
                                            qualityOracleMessage);
        if (aggregatePrecedence(qualityOracleCode) >
            aggregatePrecedence(result.exitCode)) {
          result.exitCode = qualityOracleCode;
          result.message = std::move(qualityOracleMessage);
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
  std::filesystem::path artifactDir = options.artifactDir;
  if (artifactDir.empty()) {
    auto workspace = nuri::tools::core::createRunWorkspace(
        autotestRepoRoot() / "artifacts" / "autotests");
    if (workspace.hasError()) {
      suiteResult.exitCode = AutotestExitCode::RuntimeError;
      suiteResult.message = workspace.error();
      return suiteResult;
    }
    artifactDir = workspace.value().root;
  }
  suiteResult.reportPath =
      options.jsonOut.empty() ? artifactDir / "run.json" : options.jsonOut;
  suiteResult.htmlPath =
      options.htmlOut.empty() ? artifactDir / "index.html" : options.htmlOut;
  AutotestSuiteReport suiteReport{};
  suiteReport.baselineProfile = options.baselineProfile;
  suiteReport.investigative = options.dryRun;
  suiteReport.generatedAtUtc = utcTimestampIso8601();
  suiteReport.command = options.command;
  suiteReport.suite = std::string(suite);
  suiteReport.selection.requested = std::string(suite);
  suiteReport.artifactDir = ".";

  auto suiteId = validateAutotestIdentifier(suite, "suite", false);
  auto baselineProfile = validateAutotestIdentifier(options.baselineProfile,
                                                    "baselineProfile", false);
  const bool invalidWindowOverride =
      !options.windowMode.empty() && options.windowMode != "visible" &&
      options.windowMode != "hidden" && options.windowMode != "headless";
  if (suiteId.hasError() || baselineProfile.hasError() ||
      invalidWindowOverride) {
    suiteResult.exitCode = AutotestExitCode::InvalidInput;
    suiteResult.message =
        suiteId.hasError() ? suiteId.error()
        : baselineProfile.hasError()
            ? baselineProfile.error()
            : "windowMode must be visible, hidden, or headless";
    suiteReport.diagnostics.push_back(suiteResult.message);
  }

  std::vector<size_t> selectedIndices;
  if (suiteResult.exitCode == AutotestExitCode::Success) {
    std::vector<nuri::tools::core::CaseCatalogEntry> catalog;
    catalog.reserve(testCases.size());
    for (const AutotestCase &testCase : testCases) {
      catalog.push_back({.id = testCase.id,
                         .suite = testCase.suite,
                         .manifestPath = testCase.manifestPath});
    }
    auto selected = nuri::tools::core::selectCaseCatalog(
        catalog,
        nuri::tools::core::CaseCatalogSelector{.suite = std::string(suite)},
        nuri::tools::core::CaseCatalogZeroMatchPolicy::Reject, "autotest");
    if (selected.hasError()) {
      suiteResult.exitCode = AutotestExitCode::InvalidInput;
      suiteResult.message = selected.error();
      suiteReport.diagnostics.push_back(suiteResult.message);
    } else {
      selectedIndices = std::move(selected.value());
      suiteReport.selection.selected = selectedIndices.size();
    }
  }

  if (suiteResult.exitCode == AutotestExitCode::Success) {
    for (const size_t index : selectedIndices) {
      AutotestCase &testCase = testCases[index];
      AutotestRunOptions caseOptions = options;
      caseOptions.artifactDir = artifactDir;
      caseOptions.jsonOut.clear();
      caseOptions.htmlOut.clear();
      AutotestRunResult result =
          runAutotestCase(std::move(testCase), caseOptions);
      if (aggregatePrecedence(result.exitCode) >
          aggregatePrecedence(suiteResult.exitCode)) {
        suiteResult.exitCode = result.exitCode;
      }
      suiteResult.caseResults.push_back(std::move(result));
    }
  }

  std::vector<AutotestReport> reports;
  reports.reserve(suiteResult.caseResults.size());
  for (const AutotestRunResult &caseResult : suiteResult.caseResults) {
    reports.push_back(caseResult.report);
    suiteReport.selection.attempted += caseResult.report.selection.attempted;
    suiteReport.selection.completed += caseResult.report.selection.completed;
    suiteReport.selection.passed += caseResult.report.selection.passed;
    suiteReport.selection.failed += caseResult.report.selection.failed;
    suiteReport.selection.unavailable +=
        caseResult.report.selection.unavailable;
    suiteReport.selection.notRun += caseResult.report.selection.notRun;
    suiteReport.children.push_back(AutotestSuiteChildReport{
        .id = caseResult.report.testCase.id,
        .status = caseResult.report.status,
        .exitCode = caseResult.exitCode,
        .report = relativeToCaseDir(artifactDir, caseResult.reportPath),
        .html = relativeToCaseDir(artifactDir, caseResult.htmlPath),
    });
  }
  {
    std::vector<nuri::tools::core::FingerprintField> environmentFields{
        {"scope", "autotest.suite.environment"}};
    std::vector<nuri::tools::core::FingerprintField> workloadFields{
        {"scope", "autotest.suite.workload"}};
    for (const AutotestRunResult &child : suiteResult.caseResults) {
      if (auto fingerprint =
              makeAutotestEnvironmentFingerprint(child.report.environment)) {
        environmentFields.push_back(
            {"child." + child.report.testCase.id, std::move(*fingerprint)});
      }
      if (auto fingerprint =
              makeAutotestWorkloadFingerprint(child.report.testCase)) {
        workloadFields.push_back(
            {"child." + child.report.testCase.id, std::move(*fingerprint)});
      }
    }
    auto environment =
        nuri::tools::core::makeSha256Fingerprint(std::move(environmentFields));
    auto workload =
        nuri::tools::core::makeSha256Fingerprint(std::move(workloadFields));
    if (!environment.hasError()) {
      suiteReport.environmentFingerprint = std::move(environment.value());
    }
    if (!workload.hasError()) {
      suiteReport.workloadFingerprint = std::move(workload.value());
    }
  }
  suiteReport.baselineProfileCompatible = std::all_of(
      suiteResult.caseResults.begin(), suiteResult.caseResults.end(),
      [](const AutotestRunResult &child) {
        return child.report.baselineProfileCompatible;
      });
  for (const AutotestRunResult &child : suiteResult.caseResults) {
    suiteReport.baselineProfileIncompatibilityReasons.insert(
        suiteReport.baselineProfileIncompatibilityReasons.end(),
        child.report.baselineProfileIncompatibilityReasons.begin(),
        child.report.baselineProfileIncompatibilityReasons.end());
  }
  std::sort(suiteReport.baselineProfileIncompatibilityReasons.begin(),
            suiteReport.baselineProfileIncompatibilityReasons.end());
  suiteReport.baselineProfileIncompatibilityReasons.erase(
      std::unique(suiteReport.baselineProfileIncompatibilityReasons.begin(),
                  suiteReport.baselineProfileIncompatibilityReasons.end()),
      suiteReport.baselineProfileIncompatibilityReasons.end());
  suiteReport.status = outcomeForExitCode(suiteResult.exitCode);
  suiteReport.exitCode = suiteResult.exitCode;
  auto htmlWritten =
      writeAutotestSuiteHtmlFile(reports, suite, suiteResult.htmlPath);
  if (htmlWritten.hasError()) {
    suiteResult.exitCode = AutotestExitCode::RuntimeError;
    suiteResult.message = htmlWritten.error();
    suiteReport.diagnostics.push_back(suiteResult.message);
  }
  suiteReport.status = outcomeForExitCode(suiteResult.exitCode);
  suiteReport.exitCode = suiteResult.exitCode;
  auto jsonWritten =
      writeAutotestSuiteReportFile(suiteReport, suiteResult.reportPath);
  if (jsonWritten.hasError()) {
    suiteResult.exitCode = AutotestExitCode::RuntimeError;
    suiteResult.message = jsonWritten.error();
    suiteReport.status = outcomeForExitCode(suiteResult.exitCode);
    suiteReport.exitCode = suiteResult.exitCode;
    suiteReport.diagnostics.push_back(suiteResult.message);
  }
  if (suiteResult.message.empty()) {
    suiteResult.message = suiteResult.exitCode == AutotestExitCode::Success
                              ? "suite run complete"
                              : autotestExitCodeName(suiteResult.exitCode);
  }
  suiteResult.report = std::move(suiteReport);
  return suiteResult;
}

} // namespace nuri::tools::autotest
