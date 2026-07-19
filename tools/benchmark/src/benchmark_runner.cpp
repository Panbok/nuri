#include "nuri/tools/benchmark/benchmark_runner.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/gfx/frame/temporal_frame_service.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/default_render_pipeline.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderer.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/scene_importer.h"
#include "nuri/scene/camera.h"
#include "nuri/scene/render_scene.h"
#include "nuri/tools/benchmark/benchmark_environment.h"
#include "nuri/tools/benchmark/benchmark_manifest.h"
#include "nuri/tools/benchmark/benchmark_metric_registry.h"
#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/case_catalog.h"
#include "nuri/tools/core/fingerprint.h"
#include "nuri/tools/core/process.h"
#include "nuri/tools/core/result_envelope_v2.h"
#include "nuri/tools/core/run_workspace.h"
#include "nuri/tools/core/safe_path.h"
#include "nuri/tools/core/sha256.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory_resource>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <yyjson.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
// clang-format off
#include <stdlib.h>
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <sys/resource.h>
#endif

namespace nuri::tools::benchmark {
namespace {

constexpr double kBytesPerMiB = 1024.0 * 1024.0;

using JsonMutDocPtr =
    std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;

[[nodiscard]] std::string pathToUtf8(const std::filesystem::path &path) {
  const std::u8string encoded = path.generic_u8string();
  return {reinterpret_cast<const char *>(encoded.data()), encoded.size()};
}

[[nodiscard]] std::optional<std::string>
benchmarkEnvironmentFingerprint(const BenchmarkEnvironment &environment) {
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
      FingerprintField{"window.mode", environment.windowMode},
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
  });
  return fingerprint.hasError()
             ? std::nullopt
             : std::optional<std::string>{std::move(fingerprint.value())};
}

[[nodiscard]] std::optional<std::string>
benchmarkWorkloadFingerprint(const BenchmarkCase &benchmarkCase) {
  using nuri::tools::core::FingerprintField;
  std::string manifestDigest;
  if (!benchmarkCase.manifestPath.empty() &&
      std::filesystem::is_regular_file(benchmarkCase.manifestPath)) {
    auto digest = nuri::tools::core::makeSha256FileFingerprint(
        benchmarkCase.manifestPath);
    if (!digest.hasError()) {
      manifestDigest = std::move(digest.value());
    }
  }
  auto fingerprint = nuri::tools::core::makeSha256Fingerprint({
      FingerprintField{"case.id", benchmarkCase.id},
      FingerprintField{"case.suite", benchmarkCase.suite},
      FingerprintField{"case.comparisonGroup", benchmarkCase.comparisonGroup},
      FingerprintField{"case.variant", benchmarkCase.variant},
      FingerprintField{"manifest", std::move(manifestDigest)},
      FingerprintField{"config", benchmarkCase.configSignature},
      FingerprintField{"settings", benchmarkCase.settingsSignature},
      FingerprintField{"scene.kind", benchmarkCase.scene.kind},
      FingerprintField{"scene.content", benchmarkCase.scene.contentHash},
      FingerprintField{"resolution.width",
                       std::to_string(benchmarkCase.resolution[0])},
      FingerprintField{"resolution.height",
                       std::to_string(benchmarkCase.resolution[1])},
      FingerprintField{"fixedDelta",
                       std::format("{:.17g}", benchmarkCase.fixedDeltaSeconds)},
      FingerprintField{"frames.warmup",
                       std::to_string(benchmarkCase.warmupFrames)},
      FingerprintField{"frames.measurement",
                       std::to_string(benchmarkCase.measurementFrames)},
      FingerprintField{"frames.cooldown",
                       std::to_string(benchmarkCase.cooldownFrames)},
      FingerprintField{"samples", std::to_string(benchmarkCase.samples)},
  });
  return fingerprint.hasError()
             ? std::nullopt
             : std::optional<std::string>{std::move(fingerprint.value())};
}

[[nodiscard]] std::optional<std::string> aggregateFingerprint(
    std::string_view scope,
    const std::vector<std::pair<std::string, std::string>> &children) {
  std::vector<nuri::tools::core::FingerprintField> fields;
  fields.reserve(children.size() + 1u);
  fields.push_back({"scope", std::string(scope)});
  for (const auto &[id, fingerprint] : children) {
    fields.push_back({"child." + id, fingerprint});
  }
  auto aggregate = nuri::tools::core::makeSha256Fingerprint(std::move(fields));
  return aggregate.hasError()
             ? std::nullopt
             : std::optional<std::string>{std::move(aggregate.value())};
}

void addJsonString(yyjson_mut_doc *document, yyjson_mut_val *object,
                   const char *key, std::string_view value) {
  yyjson_mut_obj_add_strncpy(document, object, key, value.data(), value.size());
}

[[nodiscard]] std::optional<std::filesystem::path>
portableRelativePath(const std::filesystem::path &path,
                     const std::filesystem::path &runRoot) {
  std::error_code error;
  std::filesystem::path relative =
      std::filesystem::relative(path, runRoot, error);
  if (error || relative.empty() || relative.is_absolute()) {
    return std::nullopt;
  }
  for (const std::filesystem::path &component : relative) {
    if (component == "..") {
      return std::nullopt;
    }
  }
  return relative;
}

[[nodiscard]] nuri::tools::core::ToolOutcome
benchmarkOutcome(BenchmarkExitCode exitCode, bool authoritative = false) {
  using nuri::tools::core::ToolOutcome;
  switch (exitCode) {
  case BenchmarkExitCode::Success:
    return authoritative ? ToolOutcome::Pass : ToolOutcome::Investigative;
  case BenchmarkExitCode::Regression:
    return ToolOutcome::Failure;
  case BenchmarkExitCode::InvalidInput:
    return ToolOutcome::Invalid;
  case BenchmarkExitCode::EnvironmentUnavailable:
    return ToolOutcome::EnvironmentUnavailable;
  case BenchmarkExitCode::RuntimeError:
    return ToolOutcome::RuntimeError;
  case BenchmarkExitCode::MissingBaseline:
    return ToolOutcome::MissingBaseline;
  }
  return ToolOutcome::RuntimeError;
}

[[nodiscard]] Result<std::string, std::string>
makeBenchmarkSuitePayload(const BenchmarkSuiteRunResult &result,
                          std::string_view suite,
                          const BenchmarkRunOptions &options,
                          const std::filesystem::path &artifactDir) {
  JsonMutDocPtr document(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!document) {
    return Result<std::string, std::string>::makeError(
        "failed to allocate benchmark suite payload");
  }
  yyjson_mut_val *root = yyjson_mut_obj(document.get());
  const bool authoritative =
      !result.caseResults.empty() &&
      std::all_of(result.caseResults.begin(), result.caseResults.end(),
                  [](const BenchmarkRunResult &child) {
                    return child.report.profile.authoritative;
                  });
  uint32_t completedRepetitions = 0u;
  bool allWarmupStable = !result.caseResults.empty();
  bool anyWarmupUnstable = false;
  if (!result.caseResults.empty()) {
    completedRepetitions = std::numeric_limits<uint32_t>::max();
    for (const BenchmarkRunResult &child : result.caseResults) {
      completedRepetitions = std::min(
          completedRepetitions, child.report.profile.completedRepetitions);
      allWarmupStable = allWarmupStable &&
                        child.report.profile.warmupStabilityStatus == "stable";
      anyWarmupUnstable =
          anyWarmupUnstable ||
          child.report.profile.warmupStabilityStatus == "unstable";
    }
  }
  yyjson_mut_doc_set_root(document.get(), root);
  yyjson_mut_obj_add_uint(document.get(), root, "schemaVersion", 1u);
  addJsonString(document.get(), root, "kind", "nuri.benchmark.suite_report");
  addJsonString(document.get(), root, "suite", suite);
  addJsonString(document.get(), root, "outcome",
                nuri::tools::core::toolOutcomeName(
                    benchmarkOutcome(result.exitCode, authoritative)));
  yyjson_mut_obj_add_int(document.get(), root, "exitCode",
                         static_cast<int>(result.exitCode));
  addJsonString(document.get(), root, "baselineProfile",
                options.baselineProfileId);
  yyjson_mut_obj_add_bool(document.get(), root, "profileAuthoritative",
                          options.baselineProfileAuthoritative);
  yyjson_mut_obj_add_bool(document.get(), root, "authoritative", authoritative);
  yyjson_mut_obj_add_uint(document.get(), root, "minimumIndependentRepetitions",
                          options.baselineProfileMinimumRepetitions);
  yyjson_mut_obj_add_uint(document.get(), root,
                          "completedIndependentRepetitions",
                          completedRepetitions);
  addJsonString(document.get(), root, "repetitionUnit",
                options.isolatedRepetitions.has_value() ? "isolated-process"
                                                        : "not-collected");
  addJsonString(document.get(), root, "warmupStabilityStatus",
                allWarmupStable ? "stable"
                                : (anyWarmupUnstable ? "unstable" : "unknown"));
  yyjson_mut_obj_add_uint(document.get(), root, "selectedCount",
                          result.caseResults.size());
  yyjson_mut_obj_add_uint(document.get(), root, "executedCount",
                          result.caseResults.size());

  std::map<std::string, std::vector<const BenchmarkRunResult *>>
      experimentGroups;
  for (const BenchmarkRunResult &child : result.caseResults) {
    if (!child.report.benchmarkCase.comparisonGroup.empty()) {
      experimentGroups[child.report.benchmarkCase.comparisonGroup].push_back(
          &child);
    }
  }
  yyjson_mut_val *groups = yyjson_mut_arr(document.get());
  for (const auto &[groupName, variants] : experimentGroups) {
    yyjson_mut_val *group = yyjson_mut_obj(document.get());
    addJsonString(document.get(), group, "comparisonGroup", groupName);
    yyjson_mut_obj_add_uint(document.get(), group, "variantCount",
                            variants.size());
    const bool complete =
        std::all_of(variants.begin(), variants.end(), [](const auto *child) {
          return child->exitCode == BenchmarkExitCode::Success;
        });
    yyjson_mut_obj_add_bool(document.get(), group, "complete", complete);
    yyjson_mut_val *variantRows = yyjson_mut_arr(document.get());
    for (const BenchmarkRunResult *child : variants) {
      yyjson_mut_val *row = yyjson_mut_obj(document.get());
      addJsonString(document.get(), row, "variant",
                    child->report.benchmarkCase.variant);
      addJsonString(document.get(), row, "caseId",
                    child->report.benchmarkCase.id);
      addJsonString(document.get(), row, "outcome",
                    nuri::tools::core::toolOutcomeName(benchmarkOutcome(
                        child->exitCode, child->report.profile.authoritative)));
      yyjson_mut_obj_add_int(document.get(), row, "exitCode",
                             static_cast<int>(child->exitCode));
      const auto relative =
          portableRelativePath(child->reportPath, artifactDir);
      if (relative.has_value()) {
        addJsonString(document.get(), row, "report", pathToUtf8(*relative));
      } else {
        yyjson_mut_obj_add_null(document.get(), row, "report");
      }
      yyjson_mut_arr_add_val(variantRows, row);
    }
    yyjson_mut_obj_add_val(document.get(), group, "variants", variantRows);
    yyjson_mut_arr_add_val(groups, group);
  }
  yyjson_mut_obj_add_val(document.get(), root, "experimentGroups", groups);

  yyjson_mut_val *caseReports = yyjson_mut_arr(document.get());
  for (const BenchmarkRunResult &child : result.caseResults) {
    const auto relative = portableRelativePath(child.reportPath, artifactDir);
    if (!relative.has_value()) {
      continue;
    }
    const std::string path = pathToUtf8(*relative);
    yyjson_mut_arr_add_strncpy(document.get(), caseReports, path.data(),
                               path.size());
  }
  yyjson_mut_obj_add_val(document.get(), root, "caseReports", caseReports);

  size_t length = 0u;
  char *json = yyjson_mut_write_opts(document.get(), YYJSON_WRITE_PRETTY,
                                     nullptr, &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "failed to serialize benchmark suite payload");
  }
  std::string payload(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(payload));
}

void addBenchmarkProfile(nuri::tools::core::ResultEnvelopeV2 &envelope,
                         const BenchmarkProfileInfo &profile) {
  if (profile.id.empty()) {
    return;
  }
  std::vector<std::string> incompatibilityReasons = profile.authorityBlockers;
  if (!profile.authoritative && incompatibilityReasons.empty()) {
    incompatibilityReasons.push_back("benchmark result is investigative");
  }
  envelope.profile = nuri::tools::core::ResultProfileV2{
      .id = profile.id,
      .compatible = profile.authoritative,
      .incompatibilityReasons = std::move(incompatibilityReasons)};
}

[[nodiscard]] Result<void, std::string> writeBenchmarkCaseEnvelope(
    const BenchmarkRunResult &result, const BenchmarkReport &report,
    const std::filesystem::path &artifactDir,
    const std::filesystem::path &envelopePath, std::string_view runId,
    bool verboseFrames, bool detailedReportComplete) {
  auto payload = writeBenchmarkReportJson(report, verboseFrames);
  if (payload.hasError()) {
    return Result<void, std::string>::makeError(payload.error());
  }

  const nuri::tools::core::ToolOutcome outcome =
      benchmarkOutcome(result.exitCode, report.profile.authoritative);
  nuri::tools::core::ResultEnvelopeV2 envelope{};
  envelope.tool = nuri::tools::core::ResultToolV2::Benchmark;
  envelope.runId = std::string(runId);
  envelope.status = outcome;
  envelope.exitCode = static_cast<int>(result.exitCode);
  envelope.authoritative = report.profile.authoritative;
  envelope.environmentFingerprint =
      benchmarkEnvironmentFingerprint(report.environment);
  envelope.workloadFingerprint =
      benchmarkWorkloadFingerprint(report.benchmarkCase);
  if (!report.generatedAtUtc.empty()) {
    envelope.startedAtUtc = report.generatedAtUtc;
  }
  if (!report.command.empty()) {
    envelope.reproduceCommand = report.command;
  }
  envelope.selection.requested = report.benchmarkCase.id;
  if (result.repetitions.empty()) {
    envelope.selection.selected = 1u;
    envelope.selection.attempted = 1u;
    envelope.selection.completed = 1u;
    envelope.selection.passed =
        outcome == nuri::tools::core::ToolOutcome::Pass ? 1u : 0u;
    envelope.selection.warned =
        outcome == nuri::tools::core::ToolOutcome::Investigative ? 1u : 0u;
    envelope.selection.failed =
        outcome == nuri::tools::core::ToolOutcome::Failure ||
                outcome == nuri::tools::core::ToolOutcome::Invalid ||
                outcome == nuri::tools::core::ToolOutcome::RuntimeError ||
                outcome == nuri::tools::core::ToolOutcome::MissingBaseline
            ? 1u
            : 0u;
    envelope.selection.unavailable =
        outcome == nuri::tools::core::ToolOutcome::EnvironmentUnavailable ? 1u
                                                                          : 0u;
  } else {
    envelope.selection.selected = result.repetitions.size();
    envelope.selection.attempted = result.repetitions.size();
    envelope.selection.completed = result.repetitions.size();
    for (const BenchmarkRepetitionResult &repetition : result.repetitions) {
      const nuri::tools::core::ToolOutcome childOutcome =
          benchmarkOutcome(repetition.exitCode, false);
      switch (childOutcome) {
      case nuri::tools::core::ToolOutcome::Pass:
        ++envelope.selection.passed;
        break;
      case nuri::tools::core::ToolOutcome::Investigative:
      case nuri::tools::core::ToolOutcome::Warn:
        ++envelope.selection.warned;
        break;
      case nuri::tools::core::ToolOutcome::EnvironmentUnavailable:
        ++envelope.selection.unavailable;
        break;
      default:
        ++envelope.selection.failed;
        break;
      }
    }
  }
  addBenchmarkProfile(envelope, report.profile);
  for (const std::string &warning : report.warnings) {
    envelope.diagnostics.push_back(
        {.code = "benchmark.warning",
         .severity = nuri::tools::core::ResultDiagnosticSeverityV2::Warning,
         .message = warning});
  }
  if (result.exitCode != BenchmarkExitCode::Success &&
      !result.message.empty()) {
    envelope.diagnostics.push_back(
        {.code = "benchmark.result",
         .severity = nuri::tools::core::ResultDiagnosticSeverityV2::Error,
         .message = result.message});
  }
  const auto addArtifact = [&](std::string role,
                               const std::filesystem::path &path,
                               std::string mediaType) {
    const auto relative = portableRelativePath(path, artifactDir);
    if (!relative.has_value() || !std::filesystem::is_regular_file(path)) {
      return;
    }
    auto digest = nuri::tools::core::sha256File(path);
    if (digest.hasError()) {
      return;
    }
    envelope.artifacts.push_back(
        {.role = std::move(role),
         .path = *relative,
         .mediaType = std::move(mediaType),
         .digest = "sha256:" + digest.value(),
         .status = nuri::tools::core::ResultArtifactStatusV2::Complete});
  };
  if (const auto relative =
          portableRelativePath(result.reportPath, artifactDir);
      result.repetitions.empty() && detailedReportComplete &&
      relative.has_value()) {
    nuri::tools::core::ResultChildV2 child{
        .id = report.benchmarkCase.id,
        .status = std::string(nuri::tools::core::toolOutcomeName(outcome)),
        .exitCode = static_cast<int>(result.exitCode),
        .result = *relative};
    envelope.children.push_back(std::move(child));
    addArtifact("benchmark.case.report", result.reportPath, "application/json");
  }
  if (!result.repetitions.empty()) {
    if (detailedReportComplete) {
      addArtifact("benchmark.aggregate.report", result.reportPath,
                  "application/json");
    }
    for (const BenchmarkRepetitionResult &repetition : result.repetitions) {
      const nuri::tools::core::ToolOutcome childOutcome =
          benchmarkOutcome(repetition.exitCode, false);
      nuri::tools::core::ResultChildV2 child{
          .id = "repetition-" + std::to_string(repetition.index + 1u),
          .status =
              std::string(nuri::tools::core::toolOutcomeName(childOutcome)),
          .exitCode = static_cast<int>(repetition.exitCode)};
      const std::filesystem::path childResult =
          std::filesystem::is_regular_file(repetition.envelopePath)
              ? repetition.envelopePath
              : repetition.reportPath;
      if (const auto relative = portableRelativePath(childResult, artifactDir);
          relative.has_value() &&
          std::filesystem::is_regular_file(childResult)) {
        child.result = *relative;
      }
      envelope.children.push_back(std::move(child));
      addArtifact("benchmark.repetition.report", repetition.reportPath,
                  "application/json");
      addArtifact("benchmark.repetition.result", repetition.envelopePath,
                  "application/json");
      addArtifact("benchmark.repetition.stdout", repetition.stdoutLogPath,
                  "text/plain; charset=utf-8");
      addArtifact("benchmark.repetition.stderr", repetition.stderrLogPath,
                  "text/plain; charset=utf-8");
    }
  }
  envelope.payloadJson = std::move(payload.value());
  return nuri::tools::core::writeResultEnvelopeV2(envelopePath, envelope);
}

[[nodiscard]] Result<void, std::string> writeBenchmarkSuiteEnvelope(
    const BenchmarkSuiteRunResult &result, std::string_view suite,
    const BenchmarkRunOptions &options,
    const std::filesystem::path &artifactDir,
    const std::filesystem::path &envelopePath, std::string_view runId) {
  auto payload = makeBenchmarkSuitePayload(result, suite, options, artifactDir);
  if (payload.hasError()) {
    return Result<void, std::string>::makeError(payload.error());
  }
  const bool authoritative =
      !result.caseResults.empty() &&
      std::all_of(result.caseResults.begin(), result.caseResults.end(),
                  [](const BenchmarkRunResult &child) {
                    return child.report.profile.authoritative;
                  });
  const nuri::tools::core::ToolOutcome outcome =
      benchmarkOutcome(result.exitCode, authoritative);
  nuri::tools::core::ResultEnvelopeV2 envelope{};
  envelope.tool = nuri::tools::core::ResultToolV2::Benchmark;
  envelope.runId = std::string(runId);
  envelope.status = outcome;
  envelope.exitCode = static_cast<int>(result.exitCode);
  envelope.authoritative = authoritative;
  std::vector<std::pair<std::string, std::string>> environmentFingerprints;
  std::vector<std::pair<std::string, std::string>> workloadFingerprints;
  for (const BenchmarkRunResult &child : result.caseResults) {
    if (auto fingerprint =
            benchmarkEnvironmentFingerprint(child.report.environment)) {
      environmentFingerprints.emplace_back(child.report.benchmarkCase.id,
                                           std::move(*fingerprint));
    }
    if (auto fingerprint =
            benchmarkWorkloadFingerprint(child.report.benchmarkCase)) {
      workloadFingerprints.emplace_back(child.report.benchmarkCase.id,
                                        std::move(*fingerprint));
    }
  }
  envelope.environmentFingerprint = aggregateFingerprint(
      "benchmark.suite.environment", environmentFingerprints);
  envelope.workloadFingerprint =
      aggregateFingerprint("benchmark.suite.workload", workloadFingerprints);
  envelope.reproduceCommand = options.command.empty()
                                  ? std::optional<std::string>{}
                                  : std::optional<std::string>{options.command};
  envelope.selection.requested = std::string(suite);
  envelope.selection.selected = result.caseResults.size();
  envelope.selection.attempted = result.caseResults.size();
  envelope.selection.completed = result.caseResults.size();
  if (!options.baselineProfileId.empty()) {
    std::vector<std::string> blockers;
    for (const BenchmarkRunResult &child : result.caseResults) {
      blockers.insert(blockers.end(),
                      child.report.profile.authorityBlockers.begin(),
                      child.report.profile.authorityBlockers.end());
    }
    std::sort(blockers.begin(), blockers.end());
    blockers.erase(std::unique(blockers.begin(), blockers.end()),
                   blockers.end());
    if (!authoritative && blockers.empty()) {
      blockers.push_back("suite contains investigative benchmark results");
    }
    envelope.profile = nuri::tools::core::ResultProfileV2{
        .id = options.baselineProfileId,
        .compatible = authoritative,
        .incompatibilityReasons = std::move(blockers)};
  }
  for (const BenchmarkRunResult &child : result.caseResults) {
    const nuri::tools::core::ToolOutcome childOutcome =
        benchmarkOutcome(child.exitCode, child.report.profile.authoritative);
    switch (childOutcome) {
    case nuri::tools::core::ToolOutcome::Pass:
      ++envelope.selection.passed;
      break;
    case nuri::tools::core::ToolOutcome::Investigative:
    case nuri::tools::core::ToolOutcome::Warn:
      ++envelope.selection.warned;
      break;
    case nuri::tools::core::ToolOutcome::EnvironmentUnavailable:
      ++envelope.selection.unavailable;
      break;
    default:
      ++envelope.selection.failed;
      break;
    }
    nuri::tools::core::ResultChildV2 summary{
        .id = child.report.benchmarkCase.id,
        .status = std::string(nuri::tools::core::toolOutcomeName(childOutcome)),
        .exitCode = static_cast<int>(child.exitCode)};
    if (const auto relative =
            portableRelativePath(child.envelopePath, artifactDir);
        relative.has_value()) {
      summary.result = *relative;
      auto digest = nuri::tools::core::sha256File(child.envelopePath);
      if (!digest.hasError()) {
        envelope.artifacts.push_back(
            {.role = "benchmark.case.result",
             .path = *relative,
             .mediaType = "application/json",
             .digest = "sha256:" + digest.value(),
             .status = nuri::tools::core::ResultArtifactStatusV2::Complete});
      }
    }
    envelope.children.push_back(std::move(summary));
  }
  envelope.diagnostics.push_back(
      {.code = "benchmark.suite.summary",
       .severity = result.exitCode == BenchmarkExitCode::Success
                       ? nuri::tools::core::ResultDiagnosticSeverityV2::Info
                       : nuri::tools::core::ResultDiagnosticSeverityV2::Error,
       .message = result.message});
  envelope.payloadJson = std::move(payload.value());
  return nuri::tools::core::writeResultEnvelopeV2(envelopePath, envelope);
}

#if defined(_WIN32)
constexpr char kPathListSeparator = ';';
constexpr std::string_view kExeSuffix = ".exe";
#else
constexpr char kPathListSeparator = ':';
constexpr std::string_view kExeSuffix = "";
#endif

[[nodiscard]] double bytesToMiB(uint64_t bytes) {
  return static_cast<double>(bytes) / kBytesPerMiB;
}

[[nodiscard]] std::string metricSlug(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  bool previousUnderscore = false;
  for (const char c : text) {
    const unsigned char uc = static_cast<unsigned char>(c);
    const bool alnum = (uc >= '0' && uc <= '9') || (uc >= 'a' && uc <= 'z') ||
                       (uc >= 'A' && uc <= 'Z');
    if (alnum) {
      out.push_back(static_cast<char>(std::tolower(uc)));
      previousUnderscore = false;
      continue;
    }
    if (!previousUnderscore && !out.empty()) {
      out.push_back('_');
      previousUnderscore = true;
    }
  }
  while (!out.empty() && out.back() == '_') {
    out.pop_back();
  }
  if (out.empty()) {
    return "unnamed_pass";
  }
  return out;
}

[[nodiscard]] std::string quoteCommandArg(const std::filesystem::path &path) {
  std::string text = path.string();
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"') {
      out += "\\\"";
    } else {
      out.push_back(c);
    }
  }
  out += "\"";
  return out;
}

[[nodiscard]] std::filesystem::path
executablePathInDir(const std::filesystem::path &dir, std::string_view name) {
  std::filesystem::path candidate = dir / std::string(name);
  if (!kExeSuffix.empty() && candidate.extension() != kExeSuffix) {
    candidate += std::string(kExeSuffix);
  }
  return candidate;
}

[[nodiscard]] std::optional<std::filesystem::path>
findExecutableInPath(std::string_view name) {
  const std::string pathValue = readProcessEnvironment("PATH");
  size_t begin = 0u;
  while (begin <= pathValue.size()) {
    const size_t end = pathValue.find(kPathListSeparator, begin);
    const std::string_view entry =
        end == std::string::npos
            ? std::string_view(pathValue).substr(begin)
            : std::string_view(pathValue).substr(begin, end - begin);
    if (!entry.empty()) {
      std::filesystem::path candidate =
          executablePathInDir(std::filesystem::path(entry), name);
      if (std::filesystem::exists(candidate)) {
        return candidate;
      }
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1u;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::filesystem::path>
findTracyToolInRoot(const std::filesystem::path &root, std::string_view name) {
  if (root.empty() || !std::filesystem::exists(root)) {
    return std::nullopt;
  }
  const std::string fileName = std::string(name) + std::string(kExeSuffix);
  std::error_code ec;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(
           root, std::filesystem::directory_options::skip_permission_denied,
           ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file(ec) || ec) {
      ec.clear();
      continue;
    }
    if (entry.path().filename() == fileName &&
        entry.path().generic_string().find("/tools/tracy/") !=
            std::string::npos) {
      return entry.path();
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::filesystem::path>
findTracyTool(std::string_view name) {
  if (auto fromPath = findExecutableInPath(name); fromPath.has_value()) {
    return fromPath;
  }
  if (const std::string vcpkgRoot = readProcessEnvironment("VCPKG_ROOT");
      !vcpkgRoot.empty()) {
    if (auto fromVcpkg =
            findTracyToolInRoot(std::filesystem::path(vcpkgRoot), name);
        fromVcpkg.has_value()) {
      return fromVcpkg;
    }
  }
  return findTracyToolInRoot(benchmarkRepoRoot() / "build", name);
}

[[nodiscard]] int
processExitCode(const nuri::tools::core::ProcessResult &result) {
  if (result.status != nuri::tools::core::ProcessStatus::Exited ||
      !result.exitCode.has_value()) {
    return -1;
  }
  return *result.exitCode;
}

void writeProcessLog(std::ostream &log, std::string_view displayCommand,
                     const nuri::tools::core::ProcessResult &result) {
  log << displayCommand << "\n\n";
  if (!result.standardOutput.empty()) {
    log << "stdout:\n" << result.standardOutput;
    if (result.standardOutput.back() != '\n') {
      log << '\n';
    }
  }
  if (!result.standardError.empty()) {
    log << "stderr:\n" << result.standardError;
    if (result.standardError.back() != '\n') {
      log << '\n';
    }
  }
  if (!result.errorMessage.empty()) {
    log << "process error: " << result.errorMessage << '\n';
  }
  if (result.exitCode.has_value()) {
    log << "exit code: " << *result.exitCode << "\n\n";
  } else {
    log << "process did not exit normally\n\n";
  }
}

[[nodiscard]] int
runCommandToLog(const nuri::tools::core::ProcessCommand &command,
                std::string_view displayCommand,
                const std::filesystem::path &logPath) {
  if (!logPath.parent_path().empty()) {
    std::filesystem::create_directories(logPath.parent_path());
  }
  std::ofstream log(logPath, std::ios::binary);
  if (!log) {
    return -1;
  }
  const nuri::tools::core::ProcessResult result =
      nuri::tools::core::runProcess(command);
  writeProcessLog(log, displayCommand, result);
  return processExitCode(result);
}

[[nodiscard]] int
runCommandToFile(const nuri::tools::core::ProcessCommand &command,
                 std::string_view displayCommand,
                 const std::filesystem::path &outputPath,
                 const std::filesystem::path &logPath) {
  if (!outputPath.parent_path().empty()) {
    std::filesystem::create_directories(outputPath.parent_path());
  }
  if (!logPath.parent_path().empty()) {
    std::filesystem::create_directories(logPath.parent_path());
  }
  std::ofstream output(outputPath, std::ios::binary);
  std::ofstream log(logPath, std::ios::binary | std::ios::app);
  if (!output || !log) {
    return -1;
  }

  const nuri::tools::core::ProcessResult result =
      nuri::tools::core::runProcess(command);
  output << result.standardOutput;
  writeProcessLog(log, displayCommand, result);
  return processExitCode(result);
}

void addArtifactOnce(std::vector<std::filesystem::path> &artifacts,
                     const std::filesystem::path &path) {
  if (path.empty()) {
    return;
  }
  if (std::find(artifacts.begin(), artifacts.end(), path) == artifacts.end()) {
    artifacts.push_back(path);
  }
}

void trimLineEnd(std::string &line) {
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
    line.pop_back();
  }
}

[[nodiscard]] std::vector<std::string> splitCsvLine(std::string_view line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (size_t i = 0u; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (quoted && i + 1u < line.size() && line[i + 1u] == '"') {
        field.push_back('"');
        ++i;
      } else {
        quoted = !quoted;
      }
      continue;
    }
    if (c == ',' && !quoted) {
      fields.push_back(std::move(field));
      field.clear();
      continue;
    }
    field.push_back(c);
  }
  fields.push_back(std::move(field));
  return fields;
}

[[nodiscard]] std::string cleanNumericField(std::string_view text) {
  std::string cleaned;
  cleaned.reserve(text.size());
  for (const char c : text) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (c != ',' && !std::isspace(uc)) {
      cleaned.push_back(c);
    }
  }
  return cleaned;
}

[[nodiscard]] uint64_t parseU64Field(std::string_view text) {
  std::string cleaned = cleanNumericField(text);
  if (cleaned.empty()) {
    return 0u;
  }
  char *end = nullptr;
  const unsigned long long value = std::strtoull(cleaned.c_str(), &end, 10);
  return end == cleaned.c_str() ? 0u : static_cast<uint64_t>(value);
}

[[nodiscard]] double parseDoubleField(std::string_view text) {
  std::string cleaned = cleanNumericField(text);
  if (cleaned.empty()) {
    return 0.0;
  }
  char *end = nullptr;
  const double value = std::strtod(cleaned.c_str(), &end);
  return end == cleaned.c_str() ? 0.0 : value;
}

[[nodiscard]] std::vector<BenchmarkTracyZoneStats>
readTracyZoneCsv(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return {};
  }

  std::string line;
  if (!std::getline(file, line)) {
    return {};
  }

  std::vector<BenchmarkTracyZoneStats> zones;
  while (std::getline(file, line)) {
    trimLineEnd(line);
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields = splitCsvLine(line);
    if (fields.size() < 10u) {
      continue;
    }
    BenchmarkTracyZoneStats zone{};
    zone.name = std::move(fields[0]);
    zone.sourceFile = std::filesystem::path(fields[1]);
    zone.sourceLine = static_cast<uint32_t>(parseU64Field(fields[2]));
    zone.totalNs = parseU64Field(fields[3]);
    zone.totalPercent = parseDoubleField(fields[4]);
    zone.count = parseU64Field(fields[5]);
    zone.meanNs = parseDoubleField(fields[6]);
    zone.minNs = parseU64Field(fields[7]);
    zone.maxNs = parseU64Field(fields[8]);
    zone.stddevNs = parseDoubleField(fields[9]);
    zones.push_back(std::move(zone));
  }

  std::sort(zones.begin(), zones.end(),
            [](const BenchmarkTracyZoneStats &lhs,
               const BenchmarkTracyZoneStats &rhs) {
              if (lhs.totalNs != rhs.totalNs) {
                return lhs.totalNs > rhs.totalNs;
              }
              return lhs.name < rhs.name;
            });
  constexpr size_t kMaxTracyZoneRows = 80u;
  if (zones.size() > kMaxTracyZoneRows) {
    zones.resize(kMaxTracyZoneRows);
  }
  return zones;
}

struct TracyZoneEvent {
  std::string name{};
  std::filesystem::path sourceFile{};
  uint32_t sourceLine = 0u;
  uint64_t startNs = 0u;
  uint64_t durationNs = 0u;
  std::string thread{};
};

[[nodiscard]] std::vector<TracyZoneEvent>
readTracyEventCsv(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return {};
  }

  std::string line;
  if (!std::getline(file, line)) {
    return {};
  }

  std::vector<TracyZoneEvent> events;
  while (std::getline(file, line)) {
    trimLineEnd(line);
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields = splitCsvLine(line);
    if (fields.size() < 6u) {
      continue;
    }
    TracyZoneEvent event{};
    event.name = std::move(fields[0]);
    event.sourceFile = std::filesystem::path(fields[1]);
    event.sourceLine = static_cast<uint32_t>(parseU64Field(fields[2]));
    event.startNs = parseU64Field(fields[3]);
    event.durationNs = parseU64Field(fields[4]);
    event.thread = std::move(fields[5]);
    if (!event.name.empty() && event.durationNs > 0u) {
      events.push_back(std::move(event));
    }
  }
  return events;
}

struct TracyFlameBuildNode {
  BenchmarkTracyFlameNode node{};
  std::map<std::string, uint32_t> childrenByKey{};
  std::vector<uint32_t> children{};
};

struct TracyFlameStackEntry {
  uint64_t endNs = 0u;
  uint32_t nodeIndex = UINT32_MAX;
  bool inFrame = false;
};

[[nodiscard]] uint64_t saturatingAdd(uint64_t lhs, uint64_t rhs) {
  if (std::numeric_limits<uint64_t>::max() - lhs < rhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs + rhs;
}

[[nodiscard]] std::string tracyFlameKey(std::string_view name,
                                        const std::filesystem::path &sourceFile,
                                        uint32_t sourceLine) {
  return std::string(name) + '\x1f' + sourceFile.generic_string() + '\x1f' +
         std::to_string(sourceLine);
}

[[nodiscard]] uint32_t
findNearestFlameParent(const std::vector<TracyFlameStackEntry> &stack,
                       uint32_t fallback) {
  for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
    if (it->nodeIndex != UINT32_MAX) {
      return it->nodeIndex;
    }
  }
  return fallback;
}

[[nodiscard]] uint32_t
flameChildIndex(std::vector<TracyFlameBuildNode> &nodes, uint32_t parentIndex,
                std::string_view name, std::string_view thread,
                const std::filesystem::path &sourceFile, uint32_t sourceLine) {
  TracyFlameBuildNode &parent = nodes[parentIndex];
  const std::string key = tracyFlameKey(name, sourceFile, sourceLine);
  if (auto it = parent.childrenByKey.find(key);
      it != parent.childrenByKey.end()) {
    return it->second;
  }

  const uint32_t childIndex = static_cast<uint32_t>(nodes.size());
  parent.childrenByKey.emplace(key, childIndex);
  parent.children.push_back(childIndex);
  TracyFlameBuildNode child{};
  child.node.name = std::string(name);
  child.node.thread = std::string(thread);
  child.node.sourceFile = sourceFile;
  child.node.sourceLine = sourceLine;
  nodes.push_back(std::move(child));
  return childIndex;
}

void sortFlameChildren(std::vector<TracyFlameBuildNode> &nodes,
                       uint32_t nodeIndex) {
  std::vector<uint32_t> &children = nodes[nodeIndex].children;
  std::sort(children.begin(), children.end(), [&](uint32_t lhs, uint32_t rhs) {
    const BenchmarkTracyFlameNode &lhsNode = nodes[lhs].node;
    const BenchmarkTracyFlameNode &rhsNode = nodes[rhs].node;
    if (lhsNode.totalNs != rhsNode.totalNs) {
      return lhsNode.totalNs > rhsNode.totalNs;
    }
    return lhsNode.name < rhsNode.name;
  });
  for (const uint32_t child : children) {
    sortFlameChildren(nodes, child);
  }
}

uint64_t finalizeFlameNode(std::vector<TracyFlameBuildNode> &nodes,
                           uint32_t nodeIndex) {
  uint64_t childTotalNs = 0u;
  for (const uint32_t child : nodes[nodeIndex].children) {
    childTotalNs = saturatingAdd(childTotalNs, finalizeFlameNode(nodes, child));
  }
  BenchmarkTracyFlameNode &node = nodes[nodeIndex].node;
  if (node.count == 0u) {
    node.totalNs = childTotalNs;
  }
  node.selfNs = node.totalNs > childTotalNs ? node.totalNs - childTotalNs : 0u;
  return node.totalNs;
}

BenchmarkTracyFlameNode
copyFlameNode(const std::vector<TracyFlameBuildNode> &nodes, uint32_t nodeIndex,
              uint32_t depth, uint64_t minTotalNs, uint64_t &retainedNodeCount,
              uint32_t &maxDepth) {
  BenchmarkTracyFlameNode out = nodes[nodeIndex].node;
  out.children.clear();
  ++retainedNodeCount;
  maxDepth = std::max(maxDepth, depth);

  constexpr uint64_t kMaxRetainedFlameNodes = 900u;
  constexpr uint64_t kAlwaysKeepFlameNodes = 96u;
  for (const uint32_t child : nodes[nodeIndex].children) {
    if (retainedNodeCount >= kMaxRetainedFlameNodes) {
      break;
    }
    if (depth > 0u && retainedNodeCount >= kAlwaysKeepFlameNodes &&
        nodes[child].node.totalNs < minTotalNs) {
      continue;
    }
    out.children.push_back(copyFlameNode(nodes, child, depth + 1u, minTotalNs,
                                         retainedNodeCount, maxDepth));
  }
  return out;
}

[[nodiscard]] BenchmarkTracyFlameGraph
buildTracyFlameGraph(std::vector<TracyZoneEvent> events) {
  BenchmarkTracyFlameGraph flameGraph{};
  flameGraph.eventCount = events.size();
  if (events.empty()) {
    return flameGraph;
  }

  std::sort(events.begin(), events.end(),
            [](const TracyZoneEvent &lhs, const TracyZoneEvent &rhs) {
              if (lhs.thread != rhs.thread) {
                return lhs.thread < rhs.thread;
              }
              if (lhs.startNs != rhs.startNs) {
                return lhs.startNs < rhs.startNs;
              }
              if (lhs.durationNs != rhs.durationNs) {
                return lhs.durationNs > rhs.durationNs;
              }
              return lhs.name < rhs.name;
            });

  const bool frameScoped = std::find_if(events.begin(), events.end(),
                                        [](const TracyZoneEvent &event) {
                                          return event.name == "BenchmarkFrame";
                                        }) != events.end();

  std::vector<TracyFlameBuildNode> nodes;
  nodes.reserve(std::min<size_t>(events.size(), 4096u));
  nodes.push_back(TracyFlameBuildNode{});
  nodes[0].node.name =
      frameScoped ? "BenchmarkFrame stacks" : "All Tracy CPU zones";

  std::string activeThread;
  uint32_t threadIndex = UINT32_MAX;
  std::vector<TracyFlameStackEntry> stack;
  for (const TracyZoneEvent &event : events) {
    if (event.thread != activeThread) {
      activeThread = event.thread;
      stack.clear();
      const std::string threadName =
          activeThread.empty() ? std::string("unnamed thread") : activeThread;
      threadIndex =
          flameChildIndex(nodes, 0u, threadName, activeThread, {}, 0u);
    }

    const uint64_t endNs = saturatingAdd(event.startNs, event.durationNs);
    while (!stack.empty() && event.startNs >= stack.back().endNs) {
      stack.pop_back();
    }
    while (!stack.empty() && endNs > stack.back().endNs) {
      stack.pop_back();
    }

    const bool parentInFrame = !stack.empty() && stack.back().inFrame;
    const bool isFrame = event.name == "BenchmarkFrame";
    const bool include = !frameScoped || isFrame || parentInFrame;
    uint32_t nodeIndex = UINT32_MAX;
    if (include) {
      const uint32_t parentIndex =
          isFrame ? threadIndex : findNearestFlameParent(stack, threadIndex);
      nodeIndex = flameChildIndex(nodes, parentIndex, event.name, event.thread,
                                  event.sourceFile, event.sourceLine);
      BenchmarkTracyFlameNode &node = nodes[nodeIndex].node;
      node.totalNs = saturatingAdd(node.totalNs, event.durationNs);
      ++node.count;
    }
    stack.push_back(TracyFlameStackEntry{
        .endNs = endNs,
        .nodeIndex = nodeIndex,
        .inFrame = parentInFrame || isFrame,
    });
  }

  finalizeFlameNode(nodes, 0u);
  sortFlameChildren(nodes, 0u);
  const uint64_t minTotalNs =
      std::max<uint64_t>(nodes[0].node.totalNs / 2000u, 1'000u);
  flameGraph.frameScoped = frameScoped;
  flameGraph.root =
      copyFlameNode(nodes, 0u, 0u, minTotalNs, flameGraph.retainedNodeCount,
                    flameGraph.maxDepth);
  return flameGraph;
}

void readTracyCaptureSummary(BenchmarkTracyReport &tracy) {
  std::ifstream file(tracy.captureLogPath, std::ios::binary);
  if (!file) {
    return;
  }
  std::string line;
  while (std::getline(file, line)) {
    trimLineEnd(line);
    if (line.starts_with("Frames:")) {
      tracy.captureFrameCount = parseU64Field(line.substr(7u));
    } else if (line.starts_with("Time span:")) {
      tracy.captureTimeSpanSeconds = parseDoubleField(line.substr(10u));
    } else if (line.starts_with("Zones:")) {
      tracy.captureZoneEventCount = parseU64Field(line.substr(6u));
    }
  }
}

struct TracyCaptureSession {
  std::filesystem::path tracePath{};
  std::filesystem::path logPath{};
  std::filesystem::path zonesCsvPath{};
  std::filesystem::path selfZonesCsvPath{};
  std::filesystem::path eventsCsvPath{};
  std::filesystem::path exportLogPath{};
  std::string command{};
  std::future<int> exitCode{};
  bool started = false;

  void finish(BenchmarkReport &report) {
    if (!started) {
      return;
    }
    started = false;
    const int code = exitCode.valid() ? exitCode.get() : -1;
    addArtifactOnce(report.artifacts.tracyArtifacts, tracePath);
    addArtifactOnce(report.artifacts.tracyArtifacts, logPath);
    report.tracy.tracePath = tracePath;
    report.tracy.captureLogPath = logPath;
    report.tracy.captureCommand = command;
    readTracyCaptureSummary(report.tracy);
    if (code != 0) {
      report.warnings.push_back("tracy-capture exited with code " +
                                std::to_string(code));
      return;
    }
    if (!std::filesystem::exists(tracePath)) {
      report.warnings.push_back("tracy-capture did not produce a trace file");
      return;
    }

    report.tracy.available = true;
    const std::optional<std::filesystem::path> exportTool =
        findTracyTool("tracy-csvexport");
    if (!exportTool.has_value()) {
      report.warnings.push_back(
          "Tracy trace captured but tracy-csvexport was not found");
      return;
    }

    zonesCsvPath =
        tracePath.parent_path() / (tracePath.stem().string() + ".zones.csv");
    selfZonesCsvPath = tracePath.parent_path() /
                       (tracePath.stem().string() + ".zones_self.csv");
    eventsCsvPath =
        tracePath.parent_path() / (tracePath.stem().string() + ".events.csv");
    exportLogPath =
        tracePath.parent_path() / (tracePath.stem().string() + ".export.log");

    const std::filesystem::path absoluteTracePath =
        std::filesystem::absolute(tracePath);
    report.tracy.zonesCsvPath = zonesCsvPath;
    report.tracy.selfZonesCsvPath = selfZonesCsvPath;
    report.tracy.exportLogPath = exportLogPath;
    report.tracy.flameGraph.eventsCsvPath = eventsCsvPath;
    report.tracy.zonesExportCommand =
        quoteCommandArg(*exportTool) + " " + quoteCommandArg(absoluteTracePath);
    report.tracy.selfZonesExportCommand = quoteCommandArg(*exportTool) +
                                          " -e " +
                                          quoteCommandArg(absoluteTracePath);
    report.tracy.flameGraph.eventsExportCommand =
        quoteCommandArg(*exportTool) + " -u " +
        quoteCommandArg(absoluteTracePath);

    const nuri::tools::core::ProcessCommand zonesCommand{
        .executable = *exportTool,
        .arguments = {pathToUtf8(absoluteTracePath)}};
    const int zonesCode =
        runCommandToFile(zonesCommand, report.tracy.zonesExportCommand,
                         zonesCsvPath, exportLogPath);
    if (zonesCode == 0) {
      report.tracy.zones = readTracyZoneCsv(zonesCsvPath);
      addArtifactOnce(report.artifacts.tracyArtifacts, zonesCsvPath);
    } else {
      report.warnings.push_back("tracy-csvexport aggregate export exited with "
                                "code " +
                                std::to_string(zonesCode));
    }

    const nuri::tools::core::ProcessCommand selfZonesCommand{
        .executable = *exportTool,
        .arguments = {"-e", pathToUtf8(absoluteTracePath)}};
    const int selfZonesCode =
        runCommandToFile(selfZonesCommand, report.tracy.selfZonesExportCommand,
                         selfZonesCsvPath, exportLogPath);
    if (selfZonesCode == 0) {
      report.tracy.selfZones = readTracyZoneCsv(selfZonesCsvPath);
      addArtifactOnce(report.artifacts.tracyArtifacts, selfZonesCsvPath);
    } else {
      report.warnings.push_back(
          "tracy-csvexport self-time export exited with code " +
          std::to_string(selfZonesCode));
    }

    const nuri::tools::core::ProcessCommand eventsCommand{
        .executable = *exportTool,
        .arguments = {"-u", pathToUtf8(absoluteTracePath)}};
    const int eventsCode = runCommandToFile(
        eventsCommand, report.tracy.flameGraph.eventsExportCommand,
        eventsCsvPath, exportLogPath);
    if (eventsCode == 0) {
      report.tracy.flameGraph =
          buildTracyFlameGraph(readTracyEventCsv(eventsCsvPath));
      report.tracy.flameGraph.eventsCsvPath = eventsCsvPath;
      report.tracy.flameGraph.eventsExportCommand =
          quoteCommandArg(*exportTool) + " -u " +
          quoteCommandArg(absoluteTracePath);
      addArtifactOnce(report.artifacts.tracyArtifacts, eventsCsvPath);
    } else {
      report.warnings.push_back(
          "tracy-csvexport event export exited with code " +
          std::to_string(eventsCode));
    }

    addArtifactOnce(report.artifacts.tracyArtifacts, exportLogPath);
    if (report.tracy.zones.empty() && report.tracy.selfZones.empty()) {
      report.warnings.push_back(
          "tracy-csvexport produced no aggregate zone stats");
    }
  }
};

[[nodiscard]] uint32_t tracyCaptureSeconds(const BenchmarkCase &benchmarkCase) {
  const double frameSeconds =
      static_cast<double>(benchmarkCase.warmupFrames +
                          benchmarkCase.measurementFrames +
                          benchmarkCase.cooldownFrames) *
      benchmarkCase.fixedDeltaSeconds;
  return static_cast<uint32_t>(
      std::clamp(std::ceil(frameSeconds) + 5.0, 5.0, 60.0));
}

[[nodiscard]] TracyCaptureSession
startTracyCaptureIfRequested(const BenchmarkCase &benchmarkCase,
                             const BenchmarkRunOptions &options,
                             BenchmarkReport &report) {
  TracyCaptureSession session{};
  if (!options.tracyDiagnostic) {
    return session;
  }
  if (!report.environment.tracyEnabled) {
    report.warnings.push_back(
        "Tracy diagnostic requested but NURI_WITH_TRACY is disabled");
    return session;
  }
  const std::optional<std::filesystem::path> captureTool =
      findTracyTool("tracy-capture");
  if (!captureTool.has_value()) {
    report.warnings.push_back(
        "Tracy diagnostic requested but tracy-capture was not found");
    return session;
  }

  const std::filesystem::path tracyDir = report.artifacts.artifactDir / "tracy";
  const std::string baseName =
      benchmarkCase.id + "_" + utcTimestampForPath() + "_tracy";
  session.tracePath = tracyDir / (baseName + ".tracy");
  session.logPath = tracyDir / (baseName + ".log");
  std::filesystem::create_directories(tracyDir);
  const uint32_t seconds = tracyCaptureSeconds(benchmarkCase);
  const std::filesystem::path traceOutputPath =
      std::filesystem::absolute(session.tracePath);
  session.command = quoteCommandArg(*captureTool) + " -o " +
                    quoteCommandArg(traceOutputPath) +
                    " -a 127.0.0.1 -p 8086 -s " + std::to_string(seconds) +
                    " -f";
  nuri::tools::core::ProcessCommand captureCommand{
      .executable = *captureTool,
      .arguments = {"-o", pathToUtf8(traceOutputPath), "-a", "127.0.0.1", "-p",
                    "8086", "-s", std::to_string(seconds), "-f"}};
  session.exitCode =
      std::async(std::launch::async,
                 [command = std::move(captureCommand),
                  displayCommand = session.command, logPath = session.logPath] {
                   return runCommandToLog(command, displayCommand, logPath);
                 });
  session.started = true;
  report.warnings.push_back("Tracy diagnostic capture is active; run is not "
                            "valid for authoritative comparison");
  return session;
}

[[nodiscard]] std::string passTimingMetricId(uint32_t orderedPassIndex,
                                             std::string_view passName,
                                             std::string_view suffix) {
  return std::format("rendergraph.pass.{:03}.{}.{}", orderedPassIndex,
                     metricSlug(passName), suffix);
}

[[nodiscard]] std::string_view
resolveTelemetryPassName(const RenderGraphTelemetrySnapshot &snapshot,
                         uint32_t passIndex) {
  if (passIndex >= snapshot.compile.passDebugNames.size()) {
    return "unnamed_pass";
  }
  const std::pmr::string &name = snapshot.compile.passDebugNames[passIndex];
  return name.empty() ? std::string_view("unnamed_pass")
                      : std::string_view(name.data(), name.size());
}

[[nodiscard]] bool
shouldIncludeGpuScopeInSum(const GpuTimingReport &report, GpuTimingScope scope,
                           uint64_t sourceFrameIndex) noexcept {
  if (scope == GpuTimingScope::WholeFrame) {
    return false;
  }
  const GpuTimingScope parent = gpuTimingParentScope(scope);
  return parent == GpuTimingScope::None || !hasGpuTimingScope(report, parent) ||
         gpuTimingScopeSourceFrame(report, parent) != sourceFrameIndex;
}

class TrackingMemoryResource final : public std::pmr::memory_resource {
public:
  explicit TrackingMemoryResource(
      std::pmr::memory_resource *upstream = std::pmr::get_default_resource())
      : upstream_(upstream) {}

  [[nodiscard]] uint64_t currentBytes() const noexcept { return currentBytes_; }
  [[nodiscard]] uint64_t peakBytes() const noexcept { return peakBytes_; }

private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    void *ptr = upstream_->allocate(bytes, alignment);
    currentBytes_ += bytes;
    peakBytes_ = std::max(peakBytes_, currentBytes_);
    return ptr;
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    upstream_->deallocate(ptr, bytes, alignment);
    currentBytes_ -= bytes;
  }

  bool
  do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }

  std::pmr::memory_resource *upstream_ = std::pmr::get_default_resource();
  uint64_t currentBytes_ = 0u;
  uint64_t peakBytes_ = 0u;
};

struct ProcessMemorySnapshot {
  bool available = false;
  uint64_t workingSetBytes = 0u;
  uint64_t peakWorkingSetBytes = 0u;
  uint64_t privateUsageBytes = 0u;
  uint64_t pagefileUsageBytes = 0u;
  uint64_t peakPagefileUsageBytes = 0u;
};

[[nodiscard]] ProcessMemorySnapshot collectProcessMemorySnapshot() {
  ProcessMemorySnapshot snapshot{};
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (K32GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
          sizeof(counters)) == FALSE) {
    return snapshot;
  }
  snapshot.available = true;
  snapshot.workingSetBytes = static_cast<uint64_t>(counters.WorkingSetSize);
  snapshot.peakWorkingSetBytes =
      static_cast<uint64_t>(counters.PeakWorkingSetSize);
  snapshot.privateUsageBytes = static_cast<uint64_t>(counters.PrivateUsage);
  snapshot.pagefileUsageBytes = static_cast<uint64_t>(counters.PagefileUsage);
  snapshot.peakPagefileUsageBytes =
      static_cast<uint64_t>(counters.PeakPagefileUsage);
#else
  struct rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return snapshot;
  }
  snapshot.available = true;
#if defined(__APPLE__)
  snapshot.peakWorkingSetBytes = static_cast<uint64_t>(usage.ru_maxrss);
#else
  snapshot.peakWorkingSetBytes = static_cast<uint64_t>(usage.ru_maxrss) * 1024u;
#endif
#endif
  return snapshot;
}

[[nodiscard]] BenchmarkMetricIndex
registeredMetricIndex(std::string_view id) noexcept {
  const std::optional<BenchmarkMetricIndex> index =
      findExactBenchmarkMetricIndex(id);
  NURI_ASSERT(index.has_value(), "Unknown exact benchmark metric: {}", id);
  return index.value_or(BenchmarkMetricIndex{});
}

#define NURI_BENCHMARK_METRIC(id)                                              \
  ([]() noexcept {                                                             \
    static const BenchmarkMetricIndex index = registeredMetricIndex(id);       \
    return index;                                                              \
  }())

void appendCounter(BenchmarkFrameMeasurements &measurements,
                   BenchmarkMetricIndex index, uint64_t value) {
  // Renderer counters are published descriptors: zero is a valid observation,
  // while absence means the metric was unavailable.
  measurements.appendRegistered(index, static_cast<double>(value));
}

void appendValue(BenchmarkFrameMeasurements &measurements,
                 BenchmarkMetricIndex index, double value) {
  measurements.appendRegistered(index, value);
}

void appendBytesAsMiB(BenchmarkFrameMeasurements &measurements,
                      BenchmarkMetricIndex index, uint64_t bytes) {
  measurements.appendRegistered(index, bytesToMiB(bytes));
}

void addTextureResourceMetrics(BenchmarkFrameMeasurements &measurements,
                               GPUDevice &gpu) {
  const TextureCacheTelemetry cache = Texture::cacheTelemetry();
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_hits"),
                cache.nativeHits);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_misses"),
                cache.nativeMisses);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_stale"),
                cache.nativeStale);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_corrupt"),
                cache.nativeCorrupt);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_writes"),
                cache.nativeWrites);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_write_failures"),
                cache.nativeWriteFailures);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.artifact_builds"),
                cache.artifactBuilds);
  appendBytesAsMiB(measurements,
                   NURI_BENCHMARK_METRIC("texture.io.authored_source_read_mb"),
                   cache.authoredSourceBytesRead);
  appendBytesAsMiB(measurements,
                   NURI_BENCHMARK_METRIC("texture.io.native_artifact_read_mb"),
                   cache.nativeArtifactBytesRead);
  appendBytesAsMiB(measurements,
                   NURI_BENCHMARK_METRIC("texture.io.dds_source_read_mb"),
                   cache.ddsSourceBytesRead);
  measurements.appendRegistered(
      NURI_BENCHMARK_METRIC("texture.artifact_build_ms"),
      static_cast<double>(cache.artifactBuildTimeNs) / 1'000'000.0);
  measurements.appendRegistered(NURI_BENCHMARK_METRIC("texture.dds_read_ms"),
                                static_cast<double>(cache.ddsReadTimeNs) /
                                    1'000'000.0);

  const DdsTexturePackTelemetry pack = ddsTexturePackTelemetry();
  appendCounter(measurements, NURI_BENCHMARK_METRIC("texture.pack.hits"),
                pack.hits);
  appendCounter(measurements, NURI_BENCHMARK_METRIC("texture.pack.misses"),
                pack.misses);
  appendCounter(measurements, NURI_BENCHMARK_METRIC("texture.pack.stale"),
                pack.stale);
  appendCounter(measurements, NURI_BENCHMARK_METRIC("texture.pack.corrupt"),
                pack.corrupt);
  appendCounter(measurements, NURI_BENCHMARK_METRIC("texture.pack.builds"),
                pack.builds);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.pack.build_failures"),
                pack.buildFailures);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.pack.read_failures"),
                pack.readFailures);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.pack.entries_served"),
                pack.entriesServed);
  appendBytesAsMiB(measurements,
                   NURI_BENCHMARK_METRIC("texture.io.pack_bytes_served_mb"),
                   pack.bytesServed);
  appendBytesAsMiB(
      measurements,
      NURI_BENCHMARK_METRIC("texture.io.pack_build_source_read_mb"),
      pack.buildSourceBytesRead);
  measurements.appendRegistered(NURI_BENCHMARK_METRIC("texture.pack.build_ms"),
                                static_cast<double>(pack.buildTimeNs) /
                                    1'000'000.0);
  measurements.appendRegistered(NURI_BENCHMARK_METRIC("texture.pack.open_ms"),
                                static_cast<double>(pack.openTimeNs) /
                                    1'000'000.0);
  measurements.appendRegistered(NURI_BENCHMARK_METRIC("texture.pack.read_ms"),
                                static_cast<double>(pack.readTimeNs) /
                                    1'000'000.0);

  const TextureUploadTelemetry uploads = gpu.getTextureUploadTelemetry();
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.upload.textures_recorded"),
                uploads.texturesRecorded);
  appendBytesAsMiB(measurements,
                   NURI_BENCHMARK_METRIC("texture.upload.recorded_mb"),
                   uploads.bytesRecorded);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.upload.batches_submitted"),
                uploads.batchesSubmitted);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.upload.bounded_batch_flushes"),
                uploads.boundedBatchFlushes);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.upload.completion_waits"),
                uploads.completionWaits);
}

#define addIfNonzero(measurements, id, value)                                  \
  appendCounter(measurements, NURI_BENCHMARK_METRIC(id), value)
#define addBytesAsMiB(measurements, id, value)                                 \
  appendBytesAsMiB(measurements, NURI_BENCHMARK_METRIC(id), value)

void addProcessMemoryMetrics(BenchmarkFrameMeasurements &measurements) {
  const ProcessMemorySnapshot snapshot = collectProcessMemorySnapshot();
  if (!snapshot.available) {
    return;
  }
  addBytesAsMiB(measurements, "memory.process.working_set_mb",
                snapshot.workingSetBytes);
  addBytesAsMiB(measurements, "memory.process.peak_working_set_mb",
                snapshot.peakWorkingSetBytes);
  addBytesAsMiB(measurements, "memory.process.private_usage_mb",
                snapshot.privateUsageBytes);
  addBytesAsMiB(measurements, "memory.process.pagefile_usage_mb",
                snapshot.pagefileUsageBytes);
  addBytesAsMiB(measurements, "memory.process.peak_pagefile_usage_mb",
                snapshot.peakPagefileUsageBytes);
}

void addPmrMemoryMetrics(BenchmarkFrameMeasurements &measurements,
                         const TrackingMemoryResource &rendererMemory,
                         const TrackingMemoryResource &pipelineMemory,
                         const TrackingMemoryResource &sceneMemory) {
  addBytesAsMiB(measurements, "memory.pmr.renderer_current_mb",
                rendererMemory.currentBytes());
  addBytesAsMiB(measurements, "memory.pmr.renderer_peak_mb",
                rendererMemory.peakBytes());
  addBytesAsMiB(measurements, "memory.pmr.pipeline_current_mb",
                pipelineMemory.currentBytes());
  addBytesAsMiB(measurements, "memory.pmr.pipeline_peak_mb",
                pipelineMemory.peakBytes());
  addBytesAsMiB(measurements, "memory.pmr.scene_current_mb",
                sceneMemory.currentBytes());
  addBytesAsMiB(measurements, "memory.pmr.scene_peak_mb",
                sceneMemory.peakBytes());
}

void addRendererFrameMetrics(BenchmarkFrameMeasurements &measurements,
                             const RenderFrameMetrics &metrics) {
  const RenderFrameMetrics::AssetStreamingFrameMetrics &assets = metrics.assets;
  addIfNonzero(measurements, "renderer.assets.cpu_completions",
               assets.cpuCompletions);
  addIfNonzero(measurements, "renderer.assets.cpu_workers", assets.cpuWorkers);
  addIfNonzero(measurements, "renderer.assets.cpu_active_worker_limit",
               assets.cpuActiveWorkerLimit);
  addIfNonzero(measurements, "renderer.assets.cpu_interactive_mode",
               assets.cpuInteractiveMode);
  addIfNonzero(measurements, "renderer.assets.cpu_queued_jobs",
               assets.cpuQueuedJobs);
  addIfNonzero(measurements, "renderer.assets.cpu_running_jobs",
               assets.cpuRunningJobs);
  addIfNonzero(measurements, "renderer.assets.cpu_running_io",
               assets.cpuRunningIo);
  addIfNonzero(measurements, "renderer.assets.cpu_running_decode",
               assets.cpuRunningDecode);
  addIfNonzero(measurements, "renderer.assets.cpu_running_cook",
               assets.cpuRunningCook);
  addIfNonzero(measurements, "renderer.assets.cpu_running_transcode",
               assets.cpuRunningTranscode);
  addIfNonzero(measurements, "renderer.assets.cpu_running_metadata",
               assets.cpuRunningMetadata);
  addIfNonzero(measurements, "renderer.assets.dedicated_copy_queue",
               assets.dedicatedCopyQueue);
  addIfNonzero(measurements, "renderer.assets.gpu_materialized",
               assets.gpuMaterialized);
  addIfNonzero(measurements, "renderer.assets.published", assets.published);
  addIfNonzero(measurements, "renderer.assets.cancelled", assets.cancelled);
  addIfNonzero(measurements, "renderer.assets.failed", assets.failed);
  addIfNonzero(measurements, "renderer.assets.scene_patches",
               assets.scenePatches);
  addIfNonzero(measurements, "renderer.assets.scene_commits",
               assets.sceneCommits);
  addIfNonzero(measurements, "renderer.assets.deferred_cpu_completions",
               assets.deferredCpuCompletions);
  addIfNonzero(measurements, "renderer.assets.publication_deadline_exceeded",
               assets.publicationDeadlineExceeded);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.assets.publication_main_thread_ms"),
      assets.publicationMainThreadMilliseconds);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.assets.publication_max_operation_ms"),
      assets.publicationMaxOperationMilliseconds);
  addIfNonzero(measurements, "renderer.assets.cpu_in_flight_bytes",
               assets.cpuInFlightBytes);
  addIfNonzero(measurements, "renderer.assets.upload_bytes",
               assets.uploadBytes);
  addIfNonzero(measurements, "renderer.assets.submitted_jobs",
               assets.submittedJobs);
  addIfNonzero(measurements, "renderer.assets.completed_jobs",
               assets.completedJobs);
  addIfNonzero(measurements, "renderer.assets.cancelled_jobs",
               assets.cancelledJobs);
  addIfNonzero(measurements, "renderer.assets.rejected_jobs",
               assets.rejectedJobs);

  const OpaqueFrameMetrics &opaque = metrics.opaque;
  addIfNonzero(measurements, "renderer.opaque.total_instances",
               opaque.totalInstances);
  addIfNonzero(measurements, "renderer.opaque.visible_instances",
               opaque.visibleInstances);
  addIfNonzero(measurements, "renderer.opaque.instanced_draws",
               opaque.instancedDraws);
  addIfNonzero(measurements, "renderer.opaque.indirect_draw_calls",
               opaque.indirectDrawCalls);
  addIfNonzero(measurements, "renderer.opaque.indirect_commands",
               opaque.indirectCommands);
  addIfNonzero(measurements, "renderer.opaque.compute_dispatches",
               opaque.computeDispatches);
  addIfNonzero(measurements, "renderer.opaque.depth_prepass_draws",
               opaque.depthPrepassDraws);
  addIfNonzero(measurements, "renderer.opaque.tessellated_draws",
               opaque.tessellatedDraws);
  addIfNonzero(measurements, "renderer.opaque.meshlet_dispatches",
               opaque.meshletDispatches);
  addIfNonzero(measurements, "renderer.opaque.meshlet_task_groups",
               opaque.meshletTaskGroups);
  addIfNonzero(measurements, "renderer.opaque.meshlet_candidates",
               opaque.meshletCandidateCount);
  addIfNonzero(measurements, "renderer.opaque.meshlet_mode_required",
               opaque.meshletModeRequired);
  addIfNonzero(measurements, "renderer.opaque.meshlet_mode_active",
               opaque.meshletModeActive);
  addIfNonzero(measurements, "renderer.opaque.meshlet_rejected_missing_feature",
               opaque.meshletRejectedMissingFeature);
  addIfNonzero(measurements,
               "renderer.opaque.meshlet_rejected_missing_asset_data",
               opaque.meshletRejectedMissingAssetData);
  addIfNonzero(measurements,
               "renderer.opaque.meshlet_rejected_incompatible_frame",
               opaque.meshletRejectedIncompatibleFrame);
  addIfNonzero(measurements, "renderer.opaque.meshlet_hybrid_active",
               opaque.meshletHybridActive);
  addIfNonzero(measurements, "renderer.opaque.meshlet_hybrid_classic_batches",
               opaque.meshletHybridClassicBatches);
  addIfNonzero(measurements, "renderer.opaque.meshlet_hybrid_classic_instances",
               opaque.meshletHybridClassicInstances);
  addIfNonzero(measurements,
               "renderer.opaque.meshlet_hybrid_coverage_classic_batches",
               opaque.meshletHybridCoverageClassicBatches);
  addIfNonzero(measurements,
               "renderer.opaque.meshlet_hybrid_coverage_classic_instances",
               opaque.meshletHybridCoverageClassicInstances);
  addIfNonzero(measurements, "renderer.opaque.meshlet_hybrid_meshlet_batches",
               opaque.meshletHybridMeshletBatches);
  addIfNonzero(measurements, "renderer.opaque.meshlet_hybrid_meshlet_instances",
               opaque.meshletHybridMeshletInstances);
  addIfNonzero(measurements, "renderer.opaque.auto_lod_active",
               opaque.autoLodActive);
  addIfNonzero(measurements, "renderer.opaque.auto_lod_history_reset",
               opaque.autoLodHistoryReset);
  addIfNonzero(measurements, "renderer.opaque.auto_lod_transitions",
               opaque.autoLodTransitions);
  addIfNonzero(measurements, "renderer.opaque.auto_lod_lod0_instances",
               opaque.autoLodLod0Instances);
  addIfNonzero(measurements, "renderer.opaque.auto_lod_lod1_instances",
               opaque.autoLodLod1Instances);

  const VisibilityFrameMetrics &visibility = metrics.visibility;
  addIfNonzero(measurements, "renderer.visibility.cpu_main_candidates",
               visibility.cpuMainCandidates);
  addIfNonzero(measurements, "renderer.visibility.cpu_main_visible_candidates",
               visibility.cpuMainVisibleCandidates);
  addIfNonzero(measurements, "renderer.visibility.cpu_main_rejected",
               visibility.cpuMainRejected);
  addIfNonzero(measurements, "renderer.visibility.gpu_main_candidates",
               visibility.gpuMainCandidates);
  addIfNonzero(measurements, "renderer.visibility.gpu_main_visible_candidates",
               visibility.gpuMainVisibleCandidates);
  addIfNonzero(measurements, "renderer.visibility.gpu_main_rejected_frustum",
               visibility.gpuMainRejectedFrustum);
  addIfNonzero(measurements, "renderer.visibility.gpu_main_rejected_occlusion",
               visibility.gpuMainRejectedOcclusion);
  addIfNonzero(measurements, "renderer.visibility.gpu_output_overflow_count",
               visibility.gpuOutputOverflowCount);
  addIfNonzero(measurements, "renderer.visibility.gpu_main_readback_available",
               visibility.gpuMainReadbackAvailable);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_main_readback_source_frame",
               visibility.gpuMainReadbackSourceFrame);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_main_readback_stale_frame_count",
               visibility.gpuMainReadbackStaleFrameCount);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_main_readback_error_count",
               visibility.gpuMainReadbackErrorCount);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_main_readback_visible_candidates",
               visibility.gpuMainReadbackVisibleCandidates);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_main_visible_list_mismatches",
               visibility.gpuMainVisibleListMismatches);
  addIfNonzero(measurements, "renderer.visibility.gpu_indirect_draw_used",
               visibility.gpuIndirectDrawUsed);
  addIfNonzero(measurements, "renderer.visibility.gpu_indirect_draw_fallback",
               visibility.gpuIndirectDrawFallback);
  addIfNonzero(measurements, "renderer.visibility.gpu_indirect_draw_commands",
               visibility.gpuIndirectDrawCommands);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_indirect_draw_readback_commands",
               visibility.gpuIndirectDrawReadbackCommands);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_indirect_draw_readback_tombstoned",
               visibility.gpuIndirectDrawReadbackTombstoned);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_indirect_draw_readback_visible",
               visibility.gpuIndirectDrawReadbackVisible);
  addIfNonzero(measurements, "renderer.visibility.indirect_mesh_dispatch_count",
               visibility.indirectMeshDispatchCount);
  addIfNonzero(measurements, "renderer.visibility.meshlet_rejected_frustum",
               visibility.meshletRejectedFrustum);
  addIfNonzero(measurements, "renderer.visibility.meshlet_rejected_cone",
               visibility.meshletRejectedCone);
  addIfNonzero(measurements, "renderer.visibility.meshlet_rejected_occlusion",
               visibility.meshletRejectedOcclusion);
  addIfNonzero(measurements, "renderer.visibility.meshlet_occlusion_available",
               visibility.meshletOcclusionAvailable);
  addIfNonzero(measurements, "renderer.visibility.meshlet_occlusion_mode",
               visibility.meshletOcclusionMode);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_occlusion_source_frame",
               visibility.meshletOcclusionSourceFrame);
  addIfNonzero(measurements, "renderer.visibility.meshlet_occlusion_source_age",
               visibility.meshletOcclusionSourceAge);
  addIfNonzero(measurements, "renderer.visibility.current_frame_hiz_active",
               visibility.currentFrameHiZActive);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_compaction_active",
               visibility.meshletPreTaskCompactionActive);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_candidates_input",
               visibility.meshletPreTaskCandidatesInput);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_candidates_output",
               visibility.meshletPreTaskCandidatesOutput);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_task_groups_input",
               visibility.meshletPreTaskTaskGroupsInput);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_task_groups_output",
               visibility.meshletPreTaskTaskGroupsOutput);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_task_groups_saved",
               visibility.meshletPreTaskTaskGroupsSaved);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_overflow_count",
               visibility.meshletPreTaskOverflowCount);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_mismatch_count",
               visibility.meshletPreTaskMismatchCount);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_payload_overflow_count",
               visibility.meshletPayloadOverflowCount);
  addIfNonzero(measurements, "renderer.visibility.meshlet_readback_available",
               visibility.meshletReadbackAvailable);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_readback_source_frame",
               visibility.meshletReadbackSourceFrame);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_readback_stale_frame_count",
               visibility.meshletReadbackStaleFrameCount);
  addIfNonzero(measurements, "renderer.visibility.meshlet_readback_error_count",
               visibility.meshletReadbackErrorCount);
  addIfNonzero(measurements, "renderer.visibility.meshlet_emitted",
               visibility.meshletEmitted);
  addIfNonzero(measurements, "renderer.visibility.meshlet_task_groups_executed",
               visibility.meshletTaskGroupsExecuted);
  addIfNonzero(measurements, "renderer.visibility.uncertain_visible",
               visibility.uncertainVisible);
  addIfNonzero(measurements, "renderer.visibility.shadow_cpu_candidates",
               visibility.shadowCpuCandidates);
  addIfNonzero(measurements, "renderer.visibility.shadow_cpu_rejected",
               visibility.shadowCpuRejected);
  addIfNonzero(measurements, "renderer.visibility.occlusion_available",
               visibility.occlusionAvailable);

  const ShadowFrameMetrics &shadow = metrics.shadow;
  addIfNonzero(measurements, "renderer.shadow.cascades", shadow.cascadeCount);
  addIfNonzero(measurements, "renderer.shadow.total_draws", shadow.totalDraws);
  addIfNonzero(measurements, "renderer.shadow.total_culled_draws",
               shadow.totalCulledDraws);
  addIfNonzero(measurements, "renderer.shadow.static_caster_entries",
               shadow.staticCasterEntries);
  addIfNonzero(measurements, "renderer.shadow.dynamic_caster_entries",
               shadow.dynamicCasterEntries);
  addIfNonzero(measurements, "renderer.shadow.static_batch_templates",
               shadow.staticBatchTemplateCount);
  addIfNonzero(measurements, "renderer.shadow.batch_entries",
               shadow.shadowBatchEntryCount);
  addIfNonzero(measurements, "renderer.shadow.instance_remaps",
               shadow.shadowInstanceRemapCount);
  addIfNonzero(measurements, "renderer.shadow.total_index_count_estimate",
               shadow.totalIndexCountEstimate);
  addIfNonzero(measurements, "renderer.shadow.submitted_draw_items",
               shadow.submittedDrawItemCount);
  addIfNonzero(measurements, "renderer.shadow.indirect_commands",
               shadow.indirectCommandCount);
  addIfNonzero(measurements, "renderer.shadow.draw_packet_bytes",
               shadow.drawPacketBytes);
  addIfNonzero(measurements, "renderer.shadow.filter_sample_budget",
               shadow.filterSampleBudget);
  addIfNonzero(measurements, "renderer.shadow.frame_gpu_bytes",
               shadow.frameGpuBytes);
  addIfNonzero(measurements, "renderer.shadow.sdsm_compute_passes",
               shadow.sdsmComputePassCount);
  addBytesAsMiB(measurements, "gpu.memory.shadow.cascade_texture_mb",
                shadow.cascadeTextureBytes);

  const AntiAliasingFrameMetrics &aa = metrics.antiAliasing;
  addIfNonzero(measurements, "renderer.aa.motion_vector_textures",
               aa.motionVectorTextureCount);
  addIfNonzero(measurements, "renderer.aa.motion_class_textures",
               aa.motionClassTextureCount);
  addIfNonzero(measurements, "renderer.aa.history_color_textures",
               aa.historyColorTextureCount);
  addIfNonzero(measurements, "renderer.aa.motion_vector_allocations",
               aa.motionVectorAllocationCount);
  addIfNonzero(measurements, "renderer.aa.motion_vector_reallocations",
               aa.motionVectorReallocationCount);
  addIfNonzero(measurements,
               "renderer.aa.motion_vector_depth_reprojection_passes",
               aa.motionVectorDepthReprojectionPassCount);
  addIfNonzero(measurements, "renderer.aa.velocity_passes",
               aa.velocityPassCount);
  addIfNonzero(measurements, "renderer.aa.velocity_draws",
               aa.velocityDrawCount);
  addIfNonzero(measurements, "renderer.aa.velocity_instances",
               aa.velocityInstanceCount);
  addIfNonzero(measurements, "renderer.aa.reactive_mask_passes",
               aa.reactiveMaskPassCount);
  addIfNonzero(measurements, "renderer.aa.reactive_mask_draws",
               aa.reactiveMaskDrawCount);
  addIfNonzero(measurements, "renderer.aa.taa_resolve_passes",
               aa.taaResolvePassCount);
  addIfNonzero(measurements, "renderer.aa.taa_copy_back_passes",
               aa.taaCopyBackPassCount);
  addIfNonzero(measurements, "renderer.aa.spatial_aa_passes",
               aa.spatialAAPassCount);
  addIfNonzero(measurements, "renderer.aa.msaa_resolve_passes",
               aa.msaaResolvePassCount);
  addIfNonzero(measurements, "renderer.aa.msaa_color_textures",
               aa.msaaColorTextureCount);
  addIfNonzero(measurements, "renderer.aa.msaa_depth_textures",
               aa.msaaDepthTextureCount);
  addIfNonzero(measurements, "renderer.aa.msaa_sample4_color_supported",
               aa.msaaSample4ColorSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_sample4_depth_supported",
               aa.msaaSample4DepthSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_depth_resolve_min_supported",
               aa.msaaDepthResolveMinSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_alpha_to_coverage_supported",
               aa.msaaAlphaToCoverageSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_sample_rate_shading_supported",
               aa.msaaSampleRateShadingSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_alpha_to_coverage_active",
               aa.msaaAlphaToCoverageEnabled);
  addIfNonzero(measurements, "renderer.aa.msaa_sample_shading_active",
               aa.msaaSampleShadingEnabled);
  addIfNonzero(measurements, "renderer.aa.msaa_unsupported_reason",
               static_cast<uint32_t>(aa.msaaUnsupportedReason));
  addIfNonzero(measurements, "renderer.aa.msaa_alpha_coverage_policy",
               static_cast<uint32_t>(aa.msaaAlphaCoveragePolicy));
  addIfNonzero(measurements, "renderer.aa.msaa_transparency_policy",
               static_cast<uint32_t>(aa.msaaTransparencyPolicy));
  addBytesAsMiB(measurements, "gpu.memory.aa.motion_vector_total_mb",
                aa.motionVectorTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.reactive_mask_total_mb",
                aa.reactiveMaskTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.motion_class_total_mb",
                aa.motionClassTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.history_color_total_mb",
                aa.historyColorTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.spatial_aa_total_mb",
                aa.spatialAATotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.msaa_total_mb", aa.msaaTotalBytes);

  const AmbientOcclusionFrameMetrics &ao = metrics.ambientOcclusion;
  addIfNonzero(measurements, "renderer.ao.normal_prepass_draws",
               ao.normalPrepassDraws);
  addIfNonzero(measurements, "renderer.ao.depth_prefilter_passes",
               ao.depthPrefilterPassCount);
  addIfNonzero(measurements, "renderer.ao.main_passes", ao.mainPassCount);
  addIfNonzero(measurements, "renderer.ao.temporal_passes",
               ao.temporalPassCount);
  addIfNonzero(measurements, "renderer.ao.temporal_motion_class_consumed",
               ao.temporalMotionClassConsumed);
  addIfNonzero(measurements, "renderer.ao.temporal_previous_depth_consumed",
               ao.temporalPreviousDepthConsumed);
  addIfNonzero(measurements, "renderer.ao.texture_count", ao.textureCount);
  addIfNonzero(measurements, "renderer.ao.normal_texture_count",
               ao.normalTextureCount);
  addIfNonzero(measurements, "renderer.ao.history_texture_count",
               ao.ambientOcclusionTextureCount);
  addBytesAsMiB(measurements, "gpu.memory.ao.total_texture_mb",
                ao.totalTextureBytes);

  const HDRPostProcessFrameMetrics &hdr = metrics.hdrPostProcess;
  addIfNonzero(measurements, "renderer.hdr.bloom_passes", hdr.bloomPassCount);
  addIfNonzero(measurements, "renderer.hdr.luminance_passes",
               hdr.luminancePassCount);
  addIfNonzero(measurements, "renderer.hdr.adaptation_passes",
               hdr.adaptationPassCount);
  addIfNonzero(measurements, "renderer.hdr.texture_count", hdr.textureCount);
  addBytesAsMiB(measurements, "gpu.memory.hdr.texture_mb", hdr.textureBytes);

  addIfNonzero(measurements, "renderer.transparent.mesh_draws",
               metrics.transparent.meshDraws);
  addIfNonzero(measurements, "renderer.transparent.contributor_sortable_draws",
               metrics.transparent.contributorSortableDraws);
  addIfNonzero(measurements, "renderer.transparent.contributor_fixed_draws",
               metrics.transparent.contributorFixedDraws);
  addIfNonzero(measurements, "renderer.transparent.pick_draws",
               metrics.transparent.pickDraws);

  const uint64_t estimatedFrameTextureBytes =
      shadow.cascadeTextureBytes + aa.motionVectorTotalBytes +
      aa.reactiveMaskTotalBytes + aa.motionClassTotalBytes +
      aa.historyColorTotalBytes + aa.spatialAATotalBytes + aa.msaaTotalBytes +
      ao.totalTextureBytes + hdr.textureBytes;
  addBytesAsMiB(measurements, "gpu.memory.frame_textures_estimated_mb",
                estimatedFrameTextureBytes);
}

#undef addBytesAsMiB
#undef addIfNonzero

void addRenderGraphTelemetryMetrics(BenchmarkFrameMeasurements &measurements,
                                    const Renderer &renderer) {
  const RenderGraphTelemetrySnapshot *snapshot =
      renderer.renderGraphTelemetry().latestSnapshot();
  if (snapshot == nullptr) {
    return;
  }

  const RenderGraphTelemetrySnapshot::Summary &summary = snapshot->summary;
#define addCount(id, value)                                                    \
  measurements.appendRegistered(NURI_BENCHMARK_METRIC(id),                     \
                                static_cast<double>(value))
  addCount("rendergraph.summary.declared_pass_count",
           summary.declaredPassCount);
  addCount("rendergraph.summary.culled_pass_count", summary.culledPassCount);
  addCount("rendergraph.summary.root_pass_count", summary.rootPassCount);
  addCount("rendergraph.summary.pass_count", summary.passCount);
  addCount("rendergraph.summary.edge_count", summary.edgeCount);
  addCount("rendergraph.summary.recorded_graphics_pass_count",
           summary.recordedGraphicsPassCount);
  addCount("rendergraph.summary.pass_barrier_plan_count",
           summary.passBarrierPlanCount);
  addCount("rendergraph.summary.final_barrier_record_count",
           summary.finalBarrierRecordCount);
  addCount("rendergraph.summary.pass_barrier_record_count",
           summary.passBarrierRecordCount);
  addCount("rendergraph.summary.recorded_command_buffer_count",
           summary.recordedCommandBufferCount);
  addCount("rendergraph.summary.submit_batch_count", summary.submitBatchCount);
  addCount("rendergraph.summary.pass_range_count", summary.passRangeCount);
  addCount("rendergraph.summary.pass_timing_count", summary.passTimingCount);
  addCount("rendergraph.summary.imported_texture_count",
           summary.importedTextures);
  addCount("rendergraph.summary.transient_texture_count",
           summary.transientTextures);
  addCount("rendergraph.summary.imported_buffer_count",
           summary.importedBuffers);
  addCount("rendergraph.summary.transient_buffer_count",
           summary.transientBuffers);
  addCount("rendergraph.summary.transient_texture_lifetime_count",
           summary.transientTextureLifetimeCount);
  addCount("rendergraph.summary.transient_buffer_lifetime_count",
           summary.transientBufferLifetimeCount);
  addCount("rendergraph.summary.transient_texture_physical_count",
           summary.transientTexturePhysicalCount);
  addCount("rendergraph.summary.transient_buffer_physical_count",
           summary.transientBufferPhysicalCount);
  addCount("rendergraph.summary.transient_texture_allocation_map_size",
           summary.transientTextureAllocationMapSize);
  addCount("rendergraph.summary.transient_buffer_allocation_map_size",
           summary.transientBufferAllocationMapSize);
  addCount("rendergraph.summary.transient_texture_physical_allocation_count",
           summary.transientTexturePhysicalAllocationCount);
  addCount("rendergraph.summary.transient_buffer_physical_allocation_count",
           summary.transientBufferPhysicalAllocationCount);
  addCount("rendergraph.summary.unresolved_texture_binding_count",
           summary.unresolvedTextureBindingCount);
  addCount("rendergraph.summary.resolved_dependency_buffer_slot_count",
           summary.resolvedDependencyBufferSlotCount);
  addCount("rendergraph.summary.unresolved_dependency_buffer_binding_count",
           summary.unresolvedDependencyBufferBindingCount);
  addCount("rendergraph.summary.owned_pre_dispatch_count",
           summary.ownedPreDispatchCount);
  addCount("rendergraph.summary.owned_draw_item_count",
           summary.ownedDrawItemCount);
  addCount("rendergraph.summary.owned_mesh_dispatch_item_count",
           summary.ownedMeshDispatchItemCount);

  for (const RenderGraphPassExecutionTiming &timing :
       snapshot->execution.passTimings) {
    const uint32_t orderedPassIndex = timing.orderedPassIndex;
    if (orderedPassIndex >= snapshot->compile.orderedPassIndices.size()) {
      continue;
    }
    const uint32_t declaredPassIndex =
        snapshot->compile.orderedPassIndices[orderedPassIndex];
    const std::string_view passName =
        resolveTelemetryPassName(*snapshot, declaredPassIndex);
    measurements.appendOwned(
        passTimingMetricId(orderedPassIndex, passName, "cpu_ms"),
        static_cast<double>(timing.cpuTimeMs));
  }
#undef addCount
}

class ScopedEnvVar final {
public:
  ScopedEnvVar(std::string name, std::string value)
      : name_(std::move(name)), oldValue_(readProcessEnvironment(name_)),
        hadOldValue_(!oldValue_.empty()) {
    set(value);
  }
  ~ScopedEnvVar() {
    if (hadOldValue_) {
      set(oldValue_);
    } else {
      unset();
    }
  }
  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

private:
  void set(const std::string &value) {
#if defined(_WIN32)
    _putenv_s(name_.c_str(), value.c_str());
#else
    setenv(name_.c_str(), value.c_str(), 1);
#endif
  }
  void unset() {
#if defined(_WIN32)
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }
  std::string name_;
  std::string oldValue_;
  bool hadOldValue_ = false;
};

class BenchmarkLogGuard final {
public:
  explicit BenchmarkLogGuard(const std::filesystem::path &path) {
    std::filesystem::create_directories(path.parent_path());
    LogConfig config{};
    config.filePath = path.string();
    config.logLevel = LogLevel::Info;
    config.consoleLevel = LogLevel::Warning;
    Log::initialize(config);
  }
  ~BenchmarkLogGuard() { Log::shutdown(); }
  BenchmarkLogGuard(const BenchmarkLogGuard &) = delete;
  BenchmarkLogGuard &operator=(const BenchmarkLogGuard &) = delete;
};

[[nodiscard]] std::string resolveBackendName(const BenchmarkCase &benchmarkCase,
                                             std::string &source) {
  if (benchmarkCase.backend != "default") {
    source = "manifest";
    return benchmarkCase.backend;
  }
  source = "default";
  return "nvrhi";
}

[[nodiscard]] std::string resolvePresentMode(const BenchmarkCase &benchmarkCase,
                                             std::string &source) {
  const std::string envPresent = readProcessEnvironment("NURI_PRESENT_MODE");
  if (benchmarkCase.presentMode != "default") {
    source = "manifest";
    return benchmarkCase.presentMode;
  }
  if (!envPresent.empty()) {
    source = "NURI_PRESENT_MODE";
    return envPresent;
  }
  source = "default";
  return "default";
}

[[nodiscard]] Result<bool, BenchmarkExitCode>
checkRequirements(const BenchmarkCase &benchmarkCase, std::string_view backend,
                  std::vector<std::string> &warnings, std::string &message) {
  if (backend != "nvrhi") {
    message = "unsupported backend '" + std::string(backend) +
              "'; nvrhi is the only available backend";
    return Result<bool, BenchmarkExitCode>::makeError(
        BenchmarkExitCode::InvalidInput);
  }
  if (!benchmarkCase.requirements.allowVisibleWindow) {
    message = "case requires hidden/headless execution, which is unavailable";
    return Result<bool, BenchmarkExitCode>::makeError(
        BenchmarkExitCode::EnvironmentUnavailable);
  }
  if (!benchmarkCase.requirements.backends.empty()) {
    bool supported = false;
    for (const std::string &allowed : benchmarkCase.requirements.backends) {
      if (allowed != "default" && allowed != "nvrhi") {
        message = "unsupported backend requirement '" + allowed + "'";
        return Result<bool, BenchmarkExitCode>::makeError(
            BenchmarkExitCode::InvalidInput);
      }
      supported = supported || allowed == backend || allowed == "default";
    }
    if (!supported) {
      message = "backend '" + std::string(backend) +
                "' is not allowed by case requirements";
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::EnvironmentUnavailable);
    }
  }
  for (const std::string &asset : benchmarkCase.requirements.assets) {
    const size_t colon = asset.find(':');
    if (colon == std::string::npos) {
      message = "invalid asset requirement '" + asset + "'";
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::InvalidInput);
    }
    auto path =
        resolveBenchmarkPath(asset.substr(0, colon), asset.substr(colon + 1u));
    if (path.hasError()) {
      message = path.error();
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::EnvironmentUnavailable);
    }
    if (!std::filesystem::exists(path.value())) {
      message = "missing required asset: " + path.value().string();
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::EnvironmentUnavailable);
    }
  }
  if (benchmarkCase.scene.kind == "prefab" &&
      benchmarkCase.scene.baseModelKind == "fitRadius") {
    warnings.push_back(
        "prefab fitRadius transforms are parsed but not enabled for benchmark "
        "comparison in this slice");
  }
  return Result<bool, BenchmarkExitCode>::makeResult(true);
}

void addGpuTimingMetric(BenchmarkFrameMeasurements &measurements,
                        BenchmarkMetricIndex metricIndex,
                        const GpuTimingReport &report, GpuTimingScope scope,
                        float timeMs) {
  if (hasGpuTimingScope(report, scope)) {
    measurements.appendRegistered(metricIndex, static_cast<double>(timeMs));
  }
}

void applyGpuTimingReport(BenchmarkReport &report,
                          const GpuTimingReport &timingReport,
                          const std::map<uint64_t, size_t> &frameByIndex) {
  std::map<uint64_t, double> scopeSumsByFrame;
  const auto add = [&](BenchmarkMetricIndex index, GpuTimingScope scope,
                       uint64_t sourceFrameIndex, float ms) {
    if (!hasGpuTimingScope(timingReport, scope)) {
      return;
    }
    const auto frameIt = frameByIndex.find(sourceFrameIndex);
    if (frameIt == frameByIndex.end()) {
      return;
    }
    report.frames[frameIt->second].measurements.appendRegistered(
        index, static_cast<double>(ms));
    if (shouldIncludeGpuScopeInSum(timingReport, scope, sourceFrameIndex)) {
      scopeSumsByFrame[sourceFrameIndex] += static_cast<double>(ms);
    }
  };
  add(NURI_BENCHMARK_METRIC("gpu.frame_ms"), GpuTimingScope::WholeFrame,
      timingReport.wholeFrameSourceFrameIndex, timingReport.wholeFrameTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.shadow_ms"), GpuTimingScope::Shadow,
      timingReport.shadowSourceFrameIndex, timingReport.shadowTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.shadow_depth_ms"),
      GpuTimingScope::ShadowDepth, timingReport.shadowDepthSourceFrameIndex,
      timingReport.shadowDepthTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.shadow_sdsm_ms"),
      GpuTimingScope::ShadowSdsm, timingReport.shadowSdsmSourceFrameIndex,
      timingReport.shadowSdsmTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.opaque_ms"), GpuTimingScope::Opaque,
      timingReport.opaqueSourceFrameIndex, timingReport.opaqueTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.gtao_ms"), GpuTimingScope::GTAO,
      timingReport.gtaoSourceFrameIndex, timingReport.gtaoTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.msaa_resolve_ms"),
      GpuTimingScope::MsaaResolve, timingReport.msaaResolveSourceFrameIndex,
      timingReport.msaaResolveTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.scene_color_downsample_ms"),
      GpuTimingScope::SceneColorDownsample,
      timingReport.sceneColorDownsampleSourceFrameIndex,
      timingReport.sceneColorDownsampleTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.taa_resolve_ms"),
      GpuTimingScope::TemporalAAResolve,
      timingReport.temporalAAResolveSourceFrameIndex,
      timingReport.temporalAAResolveTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.taa_debug_ms"),
      GpuTimingScope::TemporalAADebug,
      timingReport.temporalAADebugSourceFrameIndex,
      timingReport.temporalAADebugTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.spatial_aa_ms"),
      GpuTimingScope::SpatialAA, timingReport.spatialAASourceFrameIndex,
      timingReport.spatialAATimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.transmission_ms"),
      GpuTimingScope::Transmission, timingReport.transmissionSourceFrameIndex,
      timingReport.transmissionTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.hdr_postprocess_ms"),
      GpuTimingScope::HDRPostProcess,
      timingReport.hdrPostProcessSourceFrameIndex,
      timingReport.hdrPostProcessTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.skybox_ms"), GpuTimingScope::Skybox,
      timingReport.skyboxSourceFrameIndex, timingReport.skyboxTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.velocity_ms"), GpuTimingScope::Velocity,
      timingReport.velocitySourceFrameIndex, timingReport.velocityTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.reactive_mask_ms"),
      GpuTimingScope::ReactiveMask, timingReport.reactiveMaskSourceFrameIndex,
      timingReport.reactiveMaskTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.taa_copy_back_ms"),
      GpuTimingScope::TemporalAACopyBack,
      timingReport.temporalAACopyBackSourceFrameIndex,
      timingReport.temporalAACopyBackTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.gtao_temporal_ms"),
      GpuTimingScope::GTAOTemporal, timingReport.gtaoTemporalSourceFrameIndex,
      timingReport.gtaoTemporalTimeMs);
  for (const auto &[sourceFrameIndex, sum] : scopeSumsByFrame) {
    const auto frameIt = frameByIndex.find(sourceFrameIndex);
    if (frameIt != frameByIndex.end()) {
      report.frames[frameIt->second].measurements.appendRegistered(
          NURI_BENCHMARK_METRIC("gpu.scopes_sum_ms"), sum);
    }
  }
  for (uint32_t orderedPassIndex = 0u;
       orderedPassIndex < timingReport.passTimings.size(); ++orderedPassIndex) {
    const GpuTimingReport::PassTiming &timing =
        timingReport.passTimings[orderedPassIndex];
    const auto frameIt = frameByIndex.find(timing.sourceFrameIndex);
    if (frameIt == frameByIndex.end()) {
      continue;
    }
    report.frames[frameIt->second].measurements.appendOwned(
        passTimingMetricId(orderedPassIndex, timing.debugName, "gpu_ms"),
        static_cast<double>(timing.timeMs));
  }
}

void drainGpuTimings(GPUDevice &gpu, BenchmarkReport &report,
                     const std::map<uint64_t, size_t> &frameByIndex) {
  std::array<GpuTimingReport, 32> reports{};
  size_t drained = 0u;
  do {
    drained = gpu.drainCompletedGpuTimingReports(reports);
    for (size_t i = 0u; i < drained; ++i) {
      applyGpuTimingReport(report, reports[i], frameByIndex);
    }
  } while (drained == reports.size());
}

[[nodiscard]] Result<bool, std::string>
populateScene(const BenchmarkCase &benchmarkCase, Renderer &renderer,
              RenderScene &scene, std::pmr::memory_resource *memory,
              SceneLoadHandle &sceneLoad) {
  (void)memory;
  scene.bindResources(&renderer.resources());
  auto lightResult = scene.graph().addLight(scene.graph().rootNode(),
                                            LightDesc{
                                                .type = LightType::Directional,
                                                .name = "benchmark_key",
                                                .color = glm::vec3(1.0f),
                                                .intensity = 4.0f,
                                                .enabled = true,
                                            });
  if (lightResult.hasError()) {
    return Result<bool, std::string>::makeError(lightResult.error());
  }

  if (benchmarkCase.scene.kind == "prefab") {
    if (benchmarkCase.scene.pathBase.empty() ||
        benchmarkCase.scene.path.empty()) {
      return Result<bool, std::string>::makeError(
          "prefab scene requires pathBase and path");
    }
    auto path = resolveBenchmarkPath(benchmarkCase.scene.pathBase,
                                     benchmarkCase.scene.path);
    if (path.hasError()) {
      return Result<bool, std::string>::makeError(path.error());
    }
    if (!std::filesystem::exists(path.value())) {
      return Result<bool, std::string>::makeError("missing scene asset: " +
                                                  path.value().string());
    }
    SceneImportOptions importOptions{};
    importOptions.assetBuildOptions.flipUVs = benchmarkCase.scene.flipUVs;
    importOptions.assetBuildOptions.generateMeshlets =
        benchmarkCase.scene.generateMeshlets;
    importOptions.assetBuildOptions.meshletMaxVertices =
        benchmarkCase.scene.meshletMaxVertices;
    importOptions.assetBuildOptions.meshletMaxPrimitives =
        benchmarkCase.scene.meshletMaxPrimitives;
    importOptions.assetBuildOptions.meshletConeWeight =
        benchmarkCase.scene.meshletConeWeight;
    auto requested = renderer.assets().requestScene(SceneLoadRequest{
        .path = path.value().string(),
        .importOptions = importOptions,
        .priority = AssetPriority::Critical,
        .publication = ScenePublicationPolicy::CompleteOnly,
        .failurePolicy = SceneFailurePolicy::BestEffort,
        .debugName = benchmarkCase.scene.path.string(),
    });
    if (requested.hasError()) {
      return Result<bool, std::string>::makeError(requested.error());
    }
    sceneLoad = requested.value();
  }

  auto syncResult = scene.graph().syncWorldTransforms();
  (void)syncResult;
  auto commitResult = scene.commit();
  if (commitResult.hasError()) {
    return Result<bool, std::string>::makeError(commitResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
waitForBenchmarkAssets(Renderer &renderer, RenderScene &scene,
                       SceneLoadHandle sceneLoad) {
  if (!isValid(sceneLoad)) {
    return Result<bool, std::string>::makeResult(true);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(120);
  for (;;) {
    const SceneLoadSnapshot status = renderer.assets().query(sceneLoad);
    if (status.terminal()) {
      if (status.state == SceneLoadState::Complete ||
          status.state == SceneLoadState::CompleteWithErrors) {
        return Result<bool, std::string>::makeResult(true);
      }
      std::ostringstream message;
      message << "benchmark async scene load failed: state="
              << static_cast<uint32_t>(status.state)
              << " progress=" << status.progress;
      if (!status.error.empty()) {
        message << " error=" << status.error;
      }
      return Result<bool, std::string>::makeError(message.str());
    }
    auto pumped = renderer.assets().prepareFrame(AssetPublicationContext{
        .scene = &scene,
    });
    if (pumped.hasError()) {
      return Result<bool, std::string>::makeError(pumped.error());
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      const SceneLoadSnapshot timedOut = renderer.assets().query(sceneLoad);
      std::ostringstream message;
      message << "benchmark async scene load timed out: state="
              << static_cast<uint32_t>(timedOut.state)
              << " progress=" << timedOut.progress
              << " models=" << timedOut.models.published << "/"
              << timedOut.models.total
              << " materials=" << timedOut.materials.published << "/"
              << timedOut.materials.total
              << " textures=" << timedOut.textures.published << "/"
              << timedOut.textures.total;
      return Result<bool, std::string>::makeError(message.str());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

[[nodiscard]] glm::vec3 normalizedOrDefault(const glm::vec3 &value,
                                            const glm::vec3 &fallback) {
  return glm::length(value) > 1.0e-6f ? glm::normalize(value) : fallback;
}

[[nodiscard]] BenchmarkCameraConfig
cameraFromKeyframe(const BenchmarkCase &benchmarkCase,
                   const BenchmarkCameraKeyframe &keyframe) {
  BenchmarkCameraConfig camera = benchmarkCase.camera;
  camera.position = keyframe.position;
  if (keyframe.hasTarget) {
    camera.target = keyframe.target;
    camera.hasTarget = true;
    camera.direction = normalizedOrDefault(keyframe.target - keyframe.position,
                                           camera.direction);
  }
  return camera;
}

[[nodiscard]] bool frameInPath(const BenchmarkCameraPath &path,
                               uint32_t frame) {
  return frame >= path.startFrame && frame <= path.endFrame &&
         !path.keyframes.empty();
}

[[nodiscard]] float smoothstep(float t) { return t * t * (3.0f - 2.0f * t); }

[[nodiscard]] Result<BenchmarkCameraConfig, std::string>
evaluatePath(const BenchmarkCase &benchmarkCase,
             const BenchmarkCameraPath &path, uint32_t frame) {
  if (path.keyframes.empty()) {
    return Result<BenchmarkCameraConfig, std::string>::makeError(
        "camera path '" + path.id + "' has no keyframes");
  }

  const auto upper =
      std::lower_bound(path.keyframes.begin(), path.keyframes.end(), frame,
                       [](const BenchmarkCameraKeyframe &keyframe,
                          uint32_t value) { return keyframe.frame < value; });
  if (upper == path.keyframes.begin()) {
    return Result<BenchmarkCameraConfig, std::string>::makeResult(
        cameraFromKeyframe(benchmarkCase, *upper));
  }
  if (upper == path.keyframes.end()) {
    return Result<BenchmarkCameraConfig, std::string>::makeResult(
        cameraFromKeyframe(benchmarkCase, path.keyframes.back()));
  }

  const BenchmarkCameraKeyframe &rhs = *upper;
  const BenchmarkCameraKeyframe &lhs = *(upper - 1);
  const uint32_t span = rhs.frame - lhs.frame;
  float t = span == 0u ? 0.0f
                       : static_cast<float>(frame - lhs.frame) /
                             static_cast<float>(span);
  if (path.interpolation == "smoothstep") {
    t = smoothstep(t);
  } else if (path.interpolation != "linear") {
    return Result<BenchmarkCameraConfig, std::string>::makeError(
        "camera path '" + path.id + "' has unsupported interpolation '" +
        path.interpolation + "'");
  }

  BenchmarkCameraConfig camera = benchmarkCase.camera;
  camera.position = lhs.position + (rhs.position - lhs.position) * t;
  if (lhs.hasTarget && rhs.hasTarget) {
    camera.target = lhs.target + (rhs.target - lhs.target) * t;
    camera.hasTarget = true;
    camera.direction =
        normalizedOrDefault(camera.target - camera.position, camera.direction);
  }
  return Result<BenchmarkCameraConfig, std::string>::makeResult(camera);
}

[[nodiscard]] Result<BenchmarkCameraConfig, std::string>
evaluateBenchmarkCameraAtFrame(const BenchmarkCase &benchmarkCase,
                               uint32_t frame) {
  BenchmarkCameraConfig camera = benchmarkCase.camera;
  for (const BenchmarkCameraPath &path : benchmarkCase.timeline.cameraPaths) {
    if (!frameInPath(path, frame)) {
      continue;
    }
    auto evaluated = evaluatePath(benchmarkCase, path, frame);
    if (evaluated.hasError()) {
      return evaluated;
    }
    camera = evaluated.value();
  }
  return Result<BenchmarkCameraConfig, std::string>::makeResult(camera);
}

void applyBenchmarkCamera(Camera &camera,
                          const BenchmarkCameraConfig &cameraConfig) {
  camera.setPerspective(PerspectiveParams{
      .fovYRadians = glm::radians(cameraConfig.verticalFovDegrees),
      .nearPlane = cameraConfig.nearPlane,
      .farPlane = cameraConfig.farPlane,
  });
  const glm::vec3 direction =
      normalizedOrDefault(cameraConfig.direction, glm::vec3(0.0f, 0.0f, -1.0f));
  const glm::vec3 target = cameraConfig.hasTarget
                               ? cameraConfig.target
                               : cameraConfig.position + direction;
  camera.setLookAt(cameraConfig.position, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

[[nodiscard]] Camera makeBenchmarkCamera(const BenchmarkCase &benchmarkCase) {
  Camera camera;
  applyBenchmarkCamera(camera, benchmarkCase.camera);
  return camera;
}

void buildFrameContext(RenderFrameContext &frameContext, RenderScene &scene,
                       Renderer &renderer, RenderSettings &settings,
                       TemporalFrameService &temporalFrameService,
                       const Camera &camera, uint64_t frameIndex,
                       double timeSeconds, double deltaSeconds, uint32_t width,
                       uint32_t height) {
  sanitizeBenchmarkRenderSettings(settings);
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.frameIndex = frameIndex;
  const MaterialTableSnapshot materialSnapshot =
      renderer.resources().materialSnapshot();
  const TemporalSceneContentState sceneContent{
      .lightTopologyVersion = scene.lightTopologyVersion(),
      .lightTransformVersion = scene.lightTransformVersion(),
      .materialTableVersion = materialSnapshot.version,
      .environmentVersion = scene.environmentVersion(),
  };
  auto planResult = buildPresentationAAPlan(
      settings, {}, renderer.resources().gpuMultisampleCapabilities());
  NURI_ASSERT(!planResult.hasError(), "Invalid presentation AA plan: %s",
              planResult.error().c_str());
  frameContext.presentationAA = planResult.value();
  auto cameraResult = temporalFrameService.prepareFrame(
      camera, static_cast<float>(width) / static_cast<float>(height),
      settings.antiAliasing, frameContext.presentationAA,
      TemporalCameraFrameDesc{
          .renderExtent = glm::uvec2(width, height),
          .sceneContent = sceneContent,
      },
      frameIndex, timeSeconds, deltaSeconds);
  NURI_ASSERT(!cameraResult.hasError(), "Temporal frame prepare failed: %s",
              cameraResult.error().c_str());
  frameContext.camera = cameraResult.value();
  frameContext.temporalFrameService = &temporalFrameService;
  settings.antiAliasing.debug.resetHistoryRequested = false;
  frameContext.settings = &settings;
  frameContext.metrics = {};
  frameContext.metrics.frameIndex = frameContext.frameIndex;
  frameContext.metrics.antiAliasing =
      makeAntiAliasingFrameMetrics(frameContext.camera);
  frameContext.sharedDepthTexture = {};
  frameContext.timeSeconds = timeSeconds;
  frameContext.deltaSeconds = deltaSeconds;
}

[[nodiscard]] double elapsedMs(std::chrono::steady_clock::time_point begin,
                               std::chrono::steady_clock::time_point end =
                                   std::chrono::steady_clock::now()) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

} // namespace

Result<bool, BenchmarkExitCode>
checkBenchmarkGpuRequirements(const BenchmarkRequirements &requirements,
                              const GpuMultisampleCapabilities &capabilities,
                              std::string &message) {
  if (!requirements.msaa4x) {
    return Result<bool, BenchmarkExitCode>::makeResult(true);
  }
  std::vector<std::string_view> missing;
  if (!capabilities.sample4Color) {
    missing.push_back("sample4_color");
  }
  if (!capabilities.sample4Depth) {
    missing.push_back("sample4_depth");
  }
  if (!capabilities.depthResolveMin) {
    missing.push_back("depth_resolve_min");
  }
  if (!capabilities.alphaToCoverage) {
    missing.push_back("alpha_to_coverage");
  }
  if (missing.empty()) {
    return Result<bool, BenchmarkExitCode>::makeResult(true);
  }
  message = "required MSAA4x capability unavailable:";
  for (const std::string_view capability : missing) {
    message += " ";
    message += capability;
  }
  return Result<bool, BenchmarkExitCode>::makeError(
      BenchmarkExitCode::EnvironmentUnavailable);
}

Result<std::string, std::string>
formatBenchmarkCaseListJson(const std::vector<BenchmarkCase> &cases,
                            std::string_view suite) {
  BenchmarkReport dummy{};
  (void)dummy;
  std::ostringstream out;
  out << "{\n  \"cases\": [\n";
  bool first = true;
  for (const BenchmarkCase *benchmarkCase :
       filterBenchmarkCasesBySuite(cases, suite)) {
    if (!first) {
      out << ",\n";
    }
    first = false;
    out << "    {\"id\": \"" << benchmarkCase->id << "\", \"suite\": \""
        << benchmarkCase->suite << "\", \"comparisonGroup\": \""
        << benchmarkCase->comparisonGroup << "\", \"variant\": \""
        << benchmarkCase->variant << "\", \"description\": \""
        << benchmarkCase->description << "\"}";
  }
  out << "\n  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string formatBenchmarkCaseListText(const std::vector<BenchmarkCase> &cases,
                                        std::string_view suite) {
  std::ostringstream out;
  for (const BenchmarkCase *benchmarkCase :
       filterBenchmarkCasesBySuite(cases, suite)) {
    out << benchmarkCase->id << " [" << benchmarkCase->suite << "] "
        << benchmarkCase->description << "\n";
  }
  return out.str();
}

Result<std::string, std::string>
formatBenchmarkCaseExplanationJson(const BenchmarkCase &benchmarkCase) {
  std::ostringstream out;
  out << "{\n"
      << "  \"id\": \"" << benchmarkCase.id << "\",\n"
      << "  \"suite\": \"" << benchmarkCase.suite << "\",\n"
      << "  \"comparisonGroup\": \"" << benchmarkCase.comparisonGroup << "\",\n"
      << "  \"variant\": \"" << benchmarkCase.variant << "\",\n"
      << "  \"description\": \"" << benchmarkCase.description << "\",\n"
      << "  \"sceneKind\": \"" << benchmarkCase.scene.kind << "\",\n"
      << "  \"backend\": \"" << benchmarkCase.backend << "\",\n"
      << "  \"samples\": " << benchmarkCase.samples << ",\n"
      << "  \"measurementFrames\": " << benchmarkCase.measurementFrames << "\n"
      << "}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string
formatBenchmarkCaseExplanationText(const BenchmarkCase &benchmarkCase) {
  std::ostringstream out;
  out << benchmarkCase.id << "\n"
      << "suite: " << benchmarkCase.suite << "\n"
      << "comparison group: " << benchmarkCase.comparisonGroup << "\n"
      << "variant: " << benchmarkCase.variant << "\n"
      << "description: " << benchmarkCase.description << "\n"
      << "scene: " << benchmarkCase.scene.kind << "\n"
      << "backend: " << benchmarkCase.backend << "\n"
      << "resolution: " << benchmarkCase.resolution[0] << "x"
      << benchmarkCase.resolution[1] << "\n"
      << "frames: warmup=" << benchmarkCase.warmupFrames
      << " measured=" << benchmarkCase.measurementFrames << "\n";
  return out.str();
}

Result<std::string, std::string>
formatEffectiveConfigJson(const BenchmarkCase &benchmarkCase,
                          const BenchmarkRunOptions &options) {
  std::string backendSource;
  const std::string backend = resolveBackendName(benchmarkCase, backendSource);
  std::string presentSource;
  const std::string present = resolvePresentMode(benchmarkCase, presentSource);
  std::ostringstream out;
  out << "{\n"
      << "  \"case\": \"" << benchmarkCase.id << "\",\n"
      << "  \"comparisonGroup\": \"" << benchmarkCase.comparisonGroup << "\",\n"
      << "  \"variant\": \"" << benchmarkCase.variant << "\",\n"
      << "  \"baselineProfile\": \"" << options.baselineProfileId << "\",\n"
      << "  \"profileAuthoritative\": "
      << (options.baselineProfileAuthoritative ? "true" : "false") << ",\n"
      << "  \"authoritative\": false,\n"
      << "  \"minimumIndependentRepetitions\": "
      << options.baselineProfileMinimumRepetitions << ",\n"
      << "  \"completedIndependentRepetitions\": 0,\n"
      << "  \"requestedIndependentRepetitions\": "
      << options.isolatedRepetitions.value_or(0u) << ",\n"
      << "  \"repetitionTimeoutMs\": " << options.repetitionTimeout.count()
      << ",\n"
      << "  \"warmupStabilityPolicy\": \""
      << options.baselineProfileWarmupStability << "\",\n"
      << "  \"warmupStabilityStatus\": \"unknown\",\n"
      << "  \"warmupWindowFrames\": "
      << options.baselineProfileWarmupWindowFrames << ",\n"
      << "  \"warmupMaxDriftPercent\": "
      << options.baselineProfileWarmupMaxDriftPercent << ",\n"
      << "  \"sampleWindows\": "
      << options.samplesOverride.value_or(benchmarkCase.samples) << ",\n"
      << "  \"backend\": \"" << backend << "\",\n"
      << "  \"backendSource\": \"" << backendSource << "\",\n"
      << "  \"presentMode\": \"" << present << "\",\n"
      << "  \"presentModeSource\": \"" << presentSource << "\",\n"
      << "  \"requiresMsaa4x\": "
      << (benchmarkCase.requirements.msaa4x ? "true" : "false") << ",\n"
      << "  \"artifactDir\": \"" << options.artifactDir.generic_string()
      << "\"\n"
      << "}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

namespace {

[[nodiscard]] BenchmarkExitCode
benchmarkExitFromProcess(const nuri::tools::core::ProcessResult &process) {
  if (process.status != nuri::tools::core::ProcessStatus::Exited ||
      !process.exitCode.has_value() || *process.exitCode < 0 ||
      *process.exitCode >
          static_cast<int>(BenchmarkExitCode::MissingBaseline)) {
    return BenchmarkExitCode::RuntimeError;
  }
  return static_cast<BenchmarkExitCode>(*process.exitCode);
}

[[nodiscard]] BenchmarkExitCode
benchmarkExitFromOutcome(nuri::tools::core::ToolOutcome outcome) {
  using nuri::tools::core::ToolOutcome;
  switch (outcome) {
  case ToolOutcome::Pass:
  case ToolOutcome::Warn:
  case ToolOutcome::Investigative:
    return BenchmarkExitCode::Success;
  case ToolOutcome::Failure:
    return BenchmarkExitCode::Regression;
  case ToolOutcome::Invalid:
    return BenchmarkExitCode::InvalidInput;
  case ToolOutcome::EnvironmentUnavailable:
    return BenchmarkExitCode::EnvironmentUnavailable;
  case ToolOutcome::RuntimeError:
    return BenchmarkExitCode::RuntimeError;
  case ToolOutcome::MissingBaseline:
    return BenchmarkExitCode::MissingBaseline;
  case ToolOutcome::Cancelled:
  case ToolOutcome::Incomplete:
    return BenchmarkExitCode::RuntimeError;
  }
  return BenchmarkExitCode::RuntimeError;
}

[[nodiscard]] bool sameIsolatedEnvironment(const BenchmarkEnvironment &lhs,
                                           const BenchmarkEnvironment &rhs) {
  return lhs.commitHash == rhs.commitHash && lhs.dirty == rhs.dirty &&
         lhs.osName == rhs.osName && lhs.osVersion == rhs.osVersion &&
         lhs.cpuName == rhs.cpuName &&
         lhs.cpuLogicalThreadCount == rhs.cpuLogicalThreadCount &&
         lhs.gpuBackend == rhs.gpuBackend &&
         lhs.gpuDeviceName == rhs.gpuDeviceName &&
         lhs.gpuVendorId == rhs.gpuVendorId &&
         lhs.gpuDeviceId == rhs.gpuDeviceId &&
         lhs.gpuDriverVersion == rhs.gpuDriverVersion &&
         lhs.swapchainImageCount == rhs.swapchainImageCount &&
         lhs.resolvedPresentMode == rhs.resolvedPresentMode &&
         lhs.windowMode == rhs.windowMode &&
         lhs.windowVisible == rhs.windowVisible &&
         lhs.renderGraphWorkerCount == rhs.renderGraphWorkerCount &&
         lhs.renderGraphParallelCompile == rhs.renderGraphParallelCompile &&
         lhs.renderGraphParallelRecording == rhs.renderGraphParallelRecording &&
         lhs.buildType == rhs.buildType &&
         lhs.cmakeToolProfile == rhs.cmakeToolProfile &&
         lhs.vcpkgManifestFeatures == rhs.vcpkgManifestFeatures &&
         lhs.buildShared == rhs.buildShared &&
         lhs.loggingEnabled == rhs.loggingEnabled &&
         lhs.assertsEnabled == rhs.assertsEnabled &&
         lhs.tracyEnabled == rhs.tracyEnabled &&
         lhs.tracyDiagnostic == rhs.tracyDiagnostic &&
         lhs.devChecks == rhs.devChecks;
}

[[nodiscard]] std::optional<bool>
repetitionWarmupStatus(const BenchmarkReport &report) {
  if (report.sampleStats.empty()) {
    return std::nullopt;
  }
  bool allStable = true;
  for (const BenchmarkSampleStats &sample : report.sampleStats) {
    if (!sample.warmupStable.has_value()) {
      return std::nullopt;
    }
    if (!*sample.warmupStable) {
      allStable = false;
    }
  }
  return allStable;
}

[[nodiscard]] nuri::tools::core::BaselineProfileObservedEnvironment
observedProfileEnvironment(const BenchmarkEnvironment &environment) {
  std::string profiling = "off";
  if (environment.tracyEnabled) {
    profiling = "cpu";
  }
  return {
      .os = environment.osName,
      .backend = environment.gpuBackend,
      .backendSource = environment.gpuBackendSource,
      .windowMode = environment.windowMode,
      .windowVisible = environment.windowVisible,
      .gpuVendorId = environment.gpuVendorId,
      .gpuDeviceId = environment.gpuDeviceId,
      .driver = environment.gpuDriverVersion,
      .presentMode = environment.resolvedPresentMode,
      .profiling = std::move(profiling),
      .devChecks = environment.devChecks,
      .dirtyTree = environment.dirty,
  };
}

[[nodiscard]] std::string
processFailureMessage(const nuri::tools::core::ProcessResult &process) {
  using nuri::tools::core::ProcessStatus;
  switch (process.status) {
  case ProcessStatus::TimedOut:
    return "repetition timed out";
  case ProcessStatus::SpawnFailed:
    return "failed to spawn repetition: " + process.errorMessage;
  case ProcessStatus::InternalError:
    return "repetition process error: " + process.errorMessage;
  case ProcessStatus::Exited:
    if (process.exitCode.has_value()) {
      return "repetition exited with code " + std::to_string(*process.exitCode);
    }
    return "repetition exited without an exit code";
  }
  return "unknown repetition process error";
}

} // namespace

BenchmarkRunResult
runBenchmarkCaseIsolated(BenchmarkCase benchmarkCase,
                         const BenchmarkRunOptions &options) {
  BenchmarkRunResult result{};
  const uint32_t repetitions = options.isolatedRepetitions.value_or(0u);
  std::string runId = nuri::tools::core::createRunId();
  std::filesystem::path artifactDir = options.artifactDir;
  if (artifactDir.empty()) {
    auto workspace = nuri::tools::core::createRunWorkspace(
        benchmarkRepoRoot() / "artifacts" / "bench");
    if (workspace.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = workspace.error();
      return result;
    }
    runId = workspace.value().runId;
    artifactDir = workspace.value().root;
  } else {
    std::error_code createError;
    std::filesystem::create_directories(artifactDir, createError);
    if (createError) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = "failed to create benchmark artifact directory: " +
                       createError.message();
      return result;
    }
  }

  result.reportPath = options.jsonOut.empty()
                          ? artifactDir / "cases" / (benchmarkCase.id + ".json")
                          : options.jsonOut;
  result.envelopePath = options.envelopeOut.empty() ? artifactDir / "run.json"
                                                    : options.envelopeOut;
  if (result.envelopePath.lexically_normal() ==
      result.reportPath.lexically_normal()) {
    result.envelopePath = artifactDir / "result.json";
  }

  BenchmarkReport report{};
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = options.command;
  report.benchmarkCase = benchmarkCase;
  report.run.samples = options.samplesOverride.value_or(benchmarkCase.samples);
  report.run.warmupFrames = benchmarkCase.warmupFrames;
  report.run.measurementFrames = benchmarkCase.measurementFrames;
  report.run.cooldownFrames = benchmarkCase.cooldownFrames;
  report.run.maxDrainFrames = benchmarkCase.maxDrainFrames;
  report.run.drainTimeoutMs = benchmarkCase.drainTimeoutMs;
  report.run.fixedDeltaSeconds = benchmarkCase.fixedDeltaSeconds;
  report.run.validForComparison = false;
  report.artifacts.artifactDir = artifactDir;
  report.profile.id = options.baselineProfileId;
  report.profile.profileAuthoritative = options.baselineProfileAuthoritative;
  report.profile.minimumRepetitions = options.baselineProfileMinimumRepetitions;
  report.profile.completedRepetitions = 0u;
  report.profile.repetitionRequirementSatisfied = false;
  report.profile.repetitionUnit = "isolated-process";
  report.profile.warmupStabilityPolicy = options.baselineProfileWarmupStability;
  report.profile.warmupStabilityStatus = "unknown";
  report.profile.warmupWindowFrames = options.baselineProfileWarmupWindowFrames;
  report.profile.warmupMaxDriftPercent =
      options.baselineProfileWarmupMaxDriftPercent;
  report.profile.requiredMetrics = options.baselineProfileRequiredMetrics;
  report.repeatObservations.unit = "isolated-process";
  report.repeatObservations.independent = true;
  report.repeatObservations.count = 0u;

  const auto finalize = [&]() -> BenchmarkRunResult {
    auto reportWrite = writeBenchmarkReportFile(report, result.reportPath,
                                                options.verboseFrames);
    if (reportWrite.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = reportWrite.error();
      report.run.validForComparison = false;
      report.profile.authoritative = false;
      report.warnings.push_back(reportWrite.error());
    }
    auto envelopeWrite = writeBenchmarkCaseEnvelope(
        result, report, artifactDir, result.envelopePath, runId,
        options.verboseFrames, !reportWrite.hasError());
    if (envelopeWrite.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = envelopeWrite.error();
      report.run.validForComparison = false;
      report.profile.authoritative = false;
      report.warnings.push_back(envelopeWrite.error());
      if (!reportWrite.hasError()) {
        (void)writeBenchmarkReportFile(report, result.reportPath,
                                       options.verboseFrames);
      }
    }
    result.report = std::move(report);
    return std::move(result);
  };

  if (repetitions == 0u) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "repetitions must be greater than zero";
    report.warnings.push_back(result.message);
    return finalize();
  }
  if (options.processExecutable.empty()) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "isolated repetitions require a process executable";
    report.warnings.push_back(result.message);
    return finalize();
  }
  if (options.repetitionTimeout <= std::chrono::milliseconds::zero()) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "repetition timeout must be greater than zero";
    report.warnings.push_back(result.message);
    return finalize();
  }
  if (options.baselineProfileAuthoritative &&
      (options.baselineProfileMinimumRepetitions == 0u ||
       repetitions < options.baselineProfileMinimumRepetitions)) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "authoritative profile requires at least " +
                     std::to_string(options.baselineProfileMinimumRepetitions) +
                     " isolated repetitions";
    report.profile.authorityBlockers.push_back(result.message);
    report.warnings.push_back(result.message);
    return finalize();
  }

  std::vector<std::pair<uint32_t, BenchmarkReport>> completedReports;
  completedReports.reserve(repetitions);
  nuri::tools::core::ToolOutcome aggregateOutcome =
      nuri::tools::core::ToolOutcome::Investigative;
  for (uint32_t index = 0u; index < repetitions; ++index) {
    BenchmarkRepetitionResult repetition{};
    repetition.index = index;
    const std::filesystem::path relativeWorkspace =
        std::filesystem::path("repetitions") / std::format("{:04}", index + 1u);
    auto confinedWorkspace =
        nuri::tools::core::resolvePathUnder(artifactDir, relativeWorkspace);
    if (confinedWorkspace.hasError()) {
      repetition.message = confinedWorkspace.error();
      result.repetitions.push_back(std::move(repetition));
      aggregateOutcome = nuri::tools::core::aggregateOutcome(
          aggregateOutcome, nuri::tools::core::ToolOutcome::RuntimeError);
      continue;
    }
    repetition.workspace = confinedWorkspace.value();
    std::error_code createError;
    std::filesystem::create_directories(repetition.workspace.parent_path(),
                                        createError);
    if (!createError &&
        !std::filesystem::create_directory(repetition.workspace, createError)) {
      if (!createError) {
        createError = std::make_error_code(std::errc::file_exists);
      }
    }
    if (createError) {
      repetition.message =
          "failed to create repetition workspace: " + createError.message();
      result.repetitions.push_back(std::move(repetition));
      aggregateOutcome = nuri::tools::core::aggregateOutcome(
          aggregateOutcome, nuri::tools::core::ToolOutcome::RuntimeError);
      continue;
    }

    repetition.reportPath = repetition.workspace / "report.json";
    repetition.envelopePath = repetition.workspace / "run.json";
    repetition.stdoutLogPath = repetition.workspace / "stdout.log";
    repetition.stderrLogPath = repetition.workspace / "stderr.log";
    nuri::tools::core::ProcessCommand command{
        .executable = options.processExecutable,
        .arguments = {"__run-child", "--case", benchmarkCase.id,
                      "--artifact-dir", pathToUtf8(repetition.workspace),
                      "--repetition-index", std::to_string(index)}};
    if (!options.baselineProfileId.empty()) {
      command.arguments.push_back("--profile");
      command.arguments.push_back(options.baselineProfileId);
    }
    if (options.samplesOverride.has_value()) {
      command.arguments.push_back("--samples");
      command.arguments.push_back(std::to_string(*options.samplesOverride));
    }
    if (options.dryRun) {
      command.arguments.push_back("--dry-run");
    }
    if (options.tracyDiagnostic) {
      command.arguments.push_back("--tracy-diagnostic");
    }
    if (options.verboseFrames) {
      command.arguments.push_back("--verbose-frames");
    }

    const nuri::tools::core::ProcessResult process =
        nuri::tools::core::runProcess(command,
                                      {.workingDirectory = benchmarkRepoRoot(),
                                       .timeout = options.repetitionTimeout});
    auto stdoutWrite = nuri::tools::core::atomicWriteTextFile(
        repetition.stdoutLogPath, process.standardOutput);
    auto stderrWrite = nuri::tools::core::atomicWriteTextFile(
        repetition.stderrLogPath, process.standardError);
    repetition.exitCode = benchmarkExitFromProcess(process);
    repetition.timedOut =
        process.status == nuri::tools::core::ProcessStatus::TimedOut;
    if (stdoutWrite.hasError() || stderrWrite.hasError()) {
      repetition.exitCode = BenchmarkExitCode::RuntimeError;
      repetition.message =
          stdoutWrite.hasError() ? stdoutWrite.error() : stderrWrite.error();
    } else if (process.status != nuri::tools::core::ProcessStatus::Exited ||
               repetition.exitCode != BenchmarkExitCode::Success) {
      repetition.message = processFailureMessage(process);
    } else {
      auto childReport = readBenchmarkReportFile(repetition.reportPath);
      if (childReport.hasError()) {
        repetition.exitCode = BenchmarkExitCode::RuntimeError;
        repetition.message = childReport.error();
      } else if (childReport.value().kind != "nuri.benchmark.report" ||
                 childReport.value().benchmarkCase.id != benchmarkCase.id) {
        repetition.exitCode = BenchmarkExitCode::RuntimeError;
        repetition.message = "child report identity does not match request";
      } else {
        repetition.completed = true;
        completedReports.emplace_back(index, std::move(childReport.value()));
      }
    }
    if (!repetition.completed) {
      report.warnings.push_back("repetition " + std::to_string(index + 1u) +
                                ": " + repetition.message);
    }
    aggregateOutcome = nuri::tools::core::aggregateOutcome(
        aggregateOutcome, benchmarkOutcome(repetition.exitCode, false));
    result.repetitions.push_back(std::move(repetition));
  }

  const std::string parentGeneratedAt = report.generatedAtUtc;
  std::vector<std::string> orchestrationWarnings = std::move(report.warnings);
  if (!completedReports.empty()) {
    const BenchmarkReport &first = completedReports.front().second;
    report = first;
    report.generatedAtUtc = parentGeneratedAt;
    report.command = options.command;
    report.benchmarkCase = benchmarkCase;
    report.artifacts.artifactDir = artifactDir;
    report.frames.clear();
    report.sampleStats.clear();
    report.stats.clear();
    report.warnings = std::move(orchestrationWarnings);
    report.unavailableMetrics.clear();
    report.unregisteredObservedMetrics.clear();
  } else {
    report.warnings = std::move(orchestrationWarnings);
  }

  std::map<std::string, std::vector<double>> repetitionValues;
  std::set<std::string> unavailableMetrics;
  std::set<std::string> unregisteredMetrics;
  bool environmentsMatch = true;
  bool childrenComparable = completedReports.size() == repetitions;
  bool profileCompatible =
      options.baselineProfileId.empty() || options.baselineProfile.has_value();
  std::set<std::string> profileCompatibilityReasons;
  bool anyWarmupUnstable = false;
  bool allWarmupStable = !completedReports.empty();
  report.timingDrain = {};
  for (const auto &[index, child] : completedReports) {
    BenchmarkSampleStats sample{};
    sample.sampleIndex = index;
    sample.measuredFrameCount = static_cast<uint32_t>(std::count_if(
        child.frames.begin(), child.frames.end(),
        [](const BenchmarkFrameRecord &frame) { return frame.measured; }));
    sample.stats = child.stats;
    sample.warmupStable = repetitionWarmupStatus(child);
    if (!sample.warmupStable.has_value()) {
      allWarmupStable = false;
    } else if (!*sample.warmupStable) {
      allWarmupStable = false;
      anyWarmupUnstable = true;
    }
    for (const auto &[metricId, stats] : child.stats) {
      repetitionValues[metricId].push_back(stats.median);
    }
    unavailableMetrics.insert(child.unavailableMetrics.begin(),
                              child.unavailableMetrics.end());
    unregisteredMetrics.insert(child.unregisteredObservedMetrics.begin(),
                               child.unregisteredObservedMetrics.end());
    if (&child != &completedReports.front().second &&
        !sameIsolatedEnvironment(completedReports.front().second.environment,
                                 child.environment)) {
      environmentsMatch = false;
    }
    childrenComparable = childrenComparable && child.run.validForComparison;
    if (options.baselineProfile.has_value()) {
      const auto compatibility = nuri::tools::core::evaluateBaselineProfile(
          *options.baselineProfile,
          observedProfileEnvironment(child.environment));
      profileCompatible = profileCompatible && compatibility.compatible;
      if (!compatibility.compatible) {
        profileCompatibilityReasons.insert(compatibility.reasons.begin(),
                                           compatibility.reasons.end());
      }
    } else if (!options.baselineProfileId.empty()) {
      profileCompatibilityReasons.insert(
          "baseline profile compatibility was not evaluated");
    }
    report.timingDrain.drainComplete =
        report.timingDrain.drainComplete && child.timingDrain.drainComplete;
    report.timingDrain.drainFrames += child.timingDrain.drainFrames;
    report.timingDrain.drainTimeoutMs = std::max(
        report.timingDrain.drainTimeoutMs, child.timingDrain.drainTimeoutMs);
    report.timingDrain.missingGpuTimingFrames +=
        child.timingDrain.missingGpuTimingFrames;
    report.timingDrain.droppedGpuTimingReports +=
        child.timingDrain.droppedGpuTimingReports;
    report.sampleStats.push_back(std::move(sample));
  }
  for (auto &[metricId, values] : repetitionValues) {
    auto stats = computeMetricStats(values);
    if (!stats.hasError()) {
      report.stats.emplace(metricId, std::move(stats.value()));
    }
  }
  report.unregisteredObservedMetrics.assign(unregisteredMetrics.begin(),
                                            unregisteredMetrics.end());
  report.repeatObservations = {
      .unit = "isolated-process",
      .independent = true,
      .count = static_cast<uint32_t>(completedReports.size())};
  report.profile.id = options.baselineProfileId;
  report.profile.profileAuthoritative = options.baselineProfileAuthoritative;
  report.profile.authoritative = false;
  report.profile.minimumRepetitions = options.baselineProfileMinimumRepetitions;
  report.profile.completedRepetitions =
      static_cast<uint32_t>(completedReports.size());
  report.profile.repetitionRequirementSatisfied =
      completedReports.size() >= options.baselineProfileMinimumRepetitions;
  report.profile.repetitionUnit = "isolated-process";
  report.profile.warmupStabilityPolicy = options.baselineProfileWarmupStability;
  report.profile.warmupWindowFrames = options.baselineProfileWarmupWindowFrames;
  report.profile.warmupMaxDriftPercent =
      options.baselineProfileWarmupMaxDriftPercent;
  report.profile.warmupStabilityStatus =
      allWarmupStable ? "stable" : (anyWarmupUnstable ? "unstable" : "unknown");
  report.profile.requiredMetrics = options.baselineProfileRequiredMetrics;
  report.profile.authorityBlockers.clear();

  std::set<std::string> requiredMetrics(benchmarkCase.requiredMetrics.begin(),
                                        benchmarkCase.requiredMetrics.end());
  requiredMetrics.insert(options.baselineProfileRequiredMetrics.begin(),
                         options.baselineProfileRequiredMetrics.end());
  bool requiredMetricsComplete = true;
  for (const std::string &metricId : requiredMetrics) {
    const auto values = repetitionValues.find(metricId);
    if (values == repetitionValues.end() ||
        values->second.size() != completedReports.size() ||
        completedReports.size() != repetitions) {
      requiredMetricsComplete = false;
      unavailableMetrics.insert(metricId);
    }
  }
  report.unavailableMetrics.assign(unavailableMetrics.begin(),
                                   unavailableMetrics.end());
  if (!options.baselineProfileAuthoritative &&
      !options.baselineProfileId.empty()) {
    report.profile.authorityBlockers.push_back(
        "baseline profile is investigative");
  }
  if (completedReports.size() != repetitions) {
    report.profile.authorityBlockers.push_back(
        "not every requested isolated repetition completed");
  }
  if (!report.profile.repetitionRequirementSatisfied) {
    report.profile.authorityBlockers.push_back(
        "minimum isolated repetition requirement was not satisfied");
  }
  if (report.profile.warmupStabilityStatus != "stable") {
    report.profile.authorityBlockers.push_back(
        "warmup stability was not measured stable for every repetition");
  }
  if (!requiredMetricsComplete) {
    report.profile.authorityBlockers.push_back(
        "required metrics were not complete in every repetition");
  }
  if (!environmentsMatch) {
    report.profile.authorityBlockers.push_back(
        "isolated repetition environments did not match");
  }
  if (!profileCompatible) {
    report.profile.authorityBlockers.insert(
        report.profile.authorityBlockers.end(),
        profileCompatibilityReasons.begin(), profileCompatibilityReasons.end());
  }
  if (!childrenComparable) {
    report.profile.authorityBlockers.push_back(
        "one or more repetitions were not valid for comparison");
  }
  if (options.tracyDiagnostic) {
    report.profile.authorityBlockers.push_back(
        "Tracy diagnostic execution is not authoritative");
  }
  if (options.dryRun) {
    report.profile.authorityBlockers.push_back(
        "dry-run execution is not authoritative");
  }
  report.run.validForComparison =
      completedReports.size() == repetitions && requiredMetricsComplete &&
      environmentsMatch && profileCompatible && childrenComparable &&
      !options.tracyDiagnostic && !options.dryRun;
  report.profile.authoritative = options.baselineProfileAuthoritative &&
                                 report.profile.authorityBlockers.empty() &&
                                 report.run.validForComparison;
  report.benchmarkCase.authoritative = report.profile.authoritative;

  result.exitCode = benchmarkExitFromOutcome(aggregateOutcome);
  if (result.exitCode == BenchmarkExitCode::Success &&
      !requiredMetricsComplete) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
  }
  if (result.exitCode == BenchmarkExitCode::Success &&
      options.baselineProfileAuthoritative && !report.profile.authoritative) {
    result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
  }
  result.message =
      result.exitCode == BenchmarkExitCode::Success
          ? "isolated benchmark repetitions complete"
          : "isolated benchmark repetitions completed without authority";
  return finalize();
}

BenchmarkRunResult runBenchmarkCase(BenchmarkCase benchmarkCase,
                                    const BenchmarkRunOptions &options) {
  BenchmarkRunResult result{};
  if (!options.baselineProfileId.empty()) {
    benchmarkCase.authoritative = false;
    benchmarkCase.configSignature.clear();
  }
  const uint32_t samples =
      options.samplesOverride.value_or(benchmarkCase.samples);
  benchmarkCase.samples = samples;
  std::string backendSource;
  const std::string backend = resolveBackendName(benchmarkCase, backendSource);
  std::string presentSource;
  const std::string presentMode =
      resolvePresentMode(benchmarkCase, presentSource);
  std::string runId = nuri::tools::core::createRunId();
  std::filesystem::path artifactDir = options.artifactDir;
  if (artifactDir.empty()) {
    auto workspace = nuri::tools::core::createRunWorkspace(
        benchmarkRepoRoot() / "artifacts" / "bench");
    if (workspace.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = workspace.error();
      result.report.benchmarkCase = std::move(benchmarkCase);
      result.report.run.validForComparison = false;
      return result;
    }
    runId = workspace.value().runId;
    artifactDir = workspace.value().root;
  }
  const std::filesystem::path reportPath =
      options.jsonOut.empty()
          ? artifactDir / "cases" / (benchmarkCase.id + ".json")
          : options.jsonOut;
  std::filesystem::path envelopePath = options.envelopeOut.empty()
                                           ? artifactDir / "run.json"
                                           : options.envelopeOut;
  if (envelopePath.lexically_normal() == reportPath.lexically_normal()) {
    envelopePath = artifactDir / "result.json";
  }
  result.reportPath = reportPath;
  result.envelopePath = envelopePath;

  BenchmarkReport report{};
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = options.command;
  report.benchmarkCase = benchmarkCase;
  if (!options.baselineProfileId.empty()) {
    report.profile.id = options.baselineProfileId;
    report.profile.profileAuthoritative = options.baselineProfileAuthoritative;
    report.profile.authoritative = false;
    report.profile.minimumRepetitions =
        options.baselineProfileMinimumRepetitions;
    report.profile.completedRepetitions = 0u;
    report.profile.repetitionRequirementSatisfied =
        options.baselineProfileMinimumRepetitions == 0u;
    report.profile.repetitionUnit = "not-collected";
    report.profile.warmupStabilityPolicy =
        options.baselineProfileWarmupStability;
    report.profile.warmupStabilityStatus = "unknown";
    report.profile.warmupWindowFrames =
        options.baselineProfileWarmupWindowFrames;
    report.profile.warmupMaxDriftPercent =
        options.baselineProfileWarmupMaxDriftPercent;
    report.profile.requiredMetrics = options.baselineProfileRequiredMetrics;
    if (!options.baselineProfileAuthoritative) {
      report.profile.authorityBlockers.push_back(
          "baseline profile is investigative");
    }
    if (options.baselineProfileMinimumRepetitions > 0u) {
      report.profile.authorityBlockers.push_back(
          "independent repetitions are not collected; in-process sample "
          "windows are not profile repetitions");
    }
  }
  report.run.samples = samples;
  report.run.warmupFrames = benchmarkCase.warmupFrames;
  report.run.measurementFrames = benchmarkCase.measurementFrames;
  report.run.cooldownFrames = benchmarkCase.cooldownFrames;
  report.run.maxDrainFrames = benchmarkCase.maxDrainFrames;
  report.run.drainTimeoutMs = benchmarkCase.drainTimeoutMs;
  report.run.fixedDeltaSeconds = benchmarkCase.fixedDeltaSeconds;
  report.artifacts.artifactDir = artifactDir;
  report.timingDrain.drainTimeoutMs = benchmarkCase.drainTimeoutMs;
  report.environment =
      collectBenchmarkEnvironment(backend, backendSource, presentMode,
                                  presentSource, options.tracyDiagnostic);
  report.environment.renderGraphWorkerCount =
      benchmarkCase.renderGraph.workerCount;
  report.environment.renderGraphParallelCompile =
      benchmarkCase.renderGraph.parallelCompile;
  report.environment.renderGraphParallelRecording =
      benchmarkCase.renderGraph.parallelRecording;
  report.warnings.push_back(
      "first-slice benchmark uses the swapchain-present renderer path");
  if (!options.baselineProfileId.empty()) {
    report.warnings.push_back(options.baselineProfileWarning.empty()
                                  ? "baseline profile '" +
                                        options.baselineProfileId +
                                        "' is investigative; run is not "
                                        "authoritative"
                                  : options.baselineProfileWarning);
  }
  if (options.tracyDiagnostic) {
    report.run.validForComparison = false;
    report.warnings.push_back(
        "Tracy diagnostic mode is not valid for authoritative comparison");
  }

  const auto finalizeResult = [&]() -> BenchmarkRunResult {
    auto reportWrite =
        writeBenchmarkReportFile(report, reportPath, options.verboseFrames);
    if (reportWrite.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = reportWrite.error();
      report.run.validForComparison = false;
      report.warnings.push_back(reportWrite.error());
    }
    auto envelopeWrite = writeBenchmarkCaseEnvelope(
        result, report, artifactDir, envelopePath, runId, options.verboseFrames,
        !reportWrite.hasError());
    if (envelopeWrite.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = envelopeWrite.error();
      report.run.validForComparison = false;
      report.warnings.push_back(envelopeWrite.error());
      if (!reportWrite.hasError()) {
        (void)writeBenchmarkReportFile(report, reportPath,
                                       options.verboseFrames);
      }
    }
    result.report = std::move(report);
    return std::move(result);
  };

  const auto validateRequiredMetrics =
      [&result, &report](const std::vector<std::string> &metricIds,
                         std::string_view field) {
        for (const std::string &metricId : metricIds) {
          auto descriptor = requireBenchmarkMetricDescriptor(metricId, field);
          if (descriptor.hasError()) {
            result.exitCode = BenchmarkExitCode::InvalidInput;
            result.message = descriptor.error();
            report.run.validForComparison = false;
            report.warnings.push_back(descriptor.error());
            return false;
          }
        }
        return true;
      };
  if (options.baselineProfileAuthoritative && !options.internalIsolatedChild) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message =
        "authoritative benchmark profiles require isolated repetitions";
    report.run.validForComparison = false;
    report.warnings.push_back(result.message);
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }
  if (!validateRequiredMetrics(benchmarkCase.requiredMetrics,
                               "case requiredMetrics") ||
      !validateRequiredMetrics(options.baselineProfileRequiredMetrics,
                               "profile requiredMetrics")) {
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }
  if (samples == 0u) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "samples must be greater than zero";
    report.run.validForComparison = false;
    report.warnings.push_back(result.message);
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }

  std::string requirementMessage;
  auto requirements = checkRequirements(benchmarkCase, backend, report.warnings,
                                        requirementMessage);
  if (requirements.hasError()) {
    report.run.validForComparison = false;
    report.warnings.push_back(requirementMessage);
    result.exitCode = requirements.error();
    result.message = requirementMessage;
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }

  if (options.dryRun) {
    result.exitCode = BenchmarkExitCode::Success;
    result.message = "dry run succeeded";
    report.warnings.push_back("dry run: renderer was not initialized");
    report.run.validForComparison = false;
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }

  TracyCaptureSession tracySession{};
  try {
    BenchmarkLogGuard logGuard(artifactDir / "logs" /
                               (benchmarkCase.id + ".log"));
    std::vector<std::unique_ptr<ScopedEnvVar>> env;
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_WORKER_COUNT",
        std::to_string(benchmarkCase.renderGraph.workerCount)));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_COMPILE",
        benchmarkCase.renderGraph.parallelCompile ? "0" : "1"));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_RECORDING",
        benchmarkCase.renderGraph.parallelRecording ? "0" : "1"));
    if (presentSource == "manifest" && presentMode != "default") {
      env.push_back(
          std::make_unique<ScopedEnvVar>("NURI_PRESENT_MODE", presentMode));
    }

    auto configResult = loadRuntimeConfigFromEnvOrDefault();
    if (configResult.hasError()) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = configResult.error();
      report.run.validForComparison = false;
      report.warnings.push_back(configResult.error());
      computeBenchmarkReportStats(report);
      return finalizeResult();
    }
    RuntimeConfig config = std::move(configResult.value());
    config.window.title = "nuri-bench " + benchmarkCase.id;
    config.window.width = static_cast<int32_t>(benchmarkCase.resolution[0]);
    config.window.height = static_cast<int32_t>(benchmarkCase.resolution[1]);
    config.window.mode = WindowMode::Windowed;

    std::unique_ptr<Window> window =
        Window::create(config.window.title, config.window.width,
                       config.window.height, config.window.mode);
    if (!window) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = "failed to create benchmark window";
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      computeBenchmarkReportStats(report);
      return finalizeResult();
    }
    std::unique_ptr<GPUDevice> gpu = GPUDevice::create(*window);
    if (!gpu) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = "failed to create GPU device";
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      computeBenchmarkReportStats(report);
      return finalizeResult();
    }
    report.environment.swapchainImageCount = gpu->getSwapchainImageCount();
    const GPUAdapterInfo adapter = gpu->getAdapterInfo();
    report.environment.gpuDeviceName = adapter.name;
    report.environment.gpuVendorId = adapter.vendorId;
    report.environment.gpuDeviceId = adapter.deviceId;
    report.environment.gpuDriverVersion = adapter.driverVersion;
    std::string gpuRequirementMessage;
    auto gpuRequirements = checkBenchmarkGpuRequirements(
        benchmarkCase.requirements, gpu->getMultisampleCapabilities(),
        gpuRequirementMessage);
    if (gpuRequirements.hasError()) {
      result.exitCode = gpuRequirements.error();
      result.message = gpuRequirementMessage;
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      computeBenchmarkReportStats(report);
      return finalizeResult();
    }

    TrackingMemoryResource rendererMemoryTracker;
    TrackingMemoryResource pipelineMemoryTracker;
    TrackingMemoryResource sceneMemoryTracker;
    std::pmr::unsynchronized_pool_resource rendererMemory(
        &rendererMemoryTracker);
    std::pmr::unsynchronized_pool_resource pipelineMemory(
        &pipelineMemoryTracker);
    std::pmr::unsynchronized_pool_resource sceneMemory(&sceneMemoryTracker);
    std::unique_ptr<Renderer> renderer = Renderer::create(*gpu, rendererMemory);
    RenderPipeline pipeline(&pipelineMemory);
    auto pipelineResult = registerDefaultRenderPipeline(
        pipeline, *gpu, config.shaders, &pipelineMemory);
    if (pipelineResult.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = pipelineResult.error();
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      computeBenchmarkReportStats(report);
      return finalizeResult();
    }

    RenderScene scene(&sceneMemory);
    SceneLoadHandle sceneLoad{};
    const auto sceneResourcePrepareBegin = std::chrono::steady_clock::now();
    auto sceneResult =
        populateScene(benchmarkCase, *renderer, scene, &sceneMemory, sceneLoad);
    if (sceneResult.hasError()) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = sceneResult.error();
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      computeBenchmarkReportStats(report);
      return finalizeResult();
    }
    auto assetsReady = waitForBenchmarkAssets(*renderer, scene, sceneLoad);
    const double sceneResourcePrepareMs = elapsedMs(sceneResourcePrepareBegin);
    if (assetsReady.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = assetsReady.error();
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      computeBenchmarkReportStats(report);
      return finalizeResult();
    }

    RenderSettings settings = benchmarkCase.settings;
    Camera camera = makeBenchmarkCamera(benchmarkCase);
    TemporalFrameService temporalFrameService{};
    RenderFrameContext frameContext{};
    uint64_t frameIndex = 0u;
    double timeSeconds = 0.0;
    std::map<uint64_t, size_t> measuredFrameByIndex;

    const auto renderOneFrame =
        [&](uint32_t sampleIndex, uint32_t sampleFrame,
            bool measured) -> Result<bool, std::string> {
      NURI_PROFILER_FRAME(nullptr);
      NURI_PROFILER_FRAME("BenchmarkFrame");
      NURI_PROFILER_ZONE_STATIC("BenchmarkFrame", NURI_PROFILER_COLOR_SUBMIT);
      window->pollEvents();
      auto evaluatedCamera =
          evaluateBenchmarkCameraAtFrame(benchmarkCase, sampleFrame);
      if (evaluatedCamera.hasError()) {
        return Result<bool, std::string>::makeError(evaluatedCamera.error());
      }
      double cameraPositionDelta = 0.0;
      double cameraDirectionDelta = 0.0;
      if (sampleFrame > 0u) {
        auto previousCamera =
            evaluateBenchmarkCameraAtFrame(benchmarkCase, sampleFrame - 1u);
        if (previousCamera.hasError()) {
          return Result<bool, std::string>::makeError(previousCamera.error());
        }
        cameraPositionDelta =
            static_cast<double>(glm::length(evaluatedCamera.value().position -
                                            previousCamera.value().position));
        cameraDirectionDelta = static_cast<double>(
            glm::length(normalizedOrDefault(evaluatedCamera.value().direction,
                                            glm::vec3(0.0f, 0.0f, -1.0f)) -
                        normalizedOrDefault(previousCamera.value().direction,
                                            glm::vec3(0.0f, 0.0f, -1.0f))));
      }
      applyBenchmarkCamera(camera, evaluatedCamera.value());

      BenchmarkFrameRecord frame{};
      frame.measurements.reserve(exactBenchmarkMetricCount() + 16u);
      frame.frameIndex = frameIndex;
      frame.sampleIndex = sampleIndex;
      frame.measured = measured;
      const auto totalBegin = std::chrono::steady_clock::now();
      const auto commitBegin = std::chrono::steady_clock::now();
      auto commitResult = scene.commit();
      if (commitResult.hasError()) {
        return Result<bool, std::string>::makeError(commitResult.error());
      }
      const double sceneCommitMs = elapsedMs(commitBegin);
      buildFrameContext(
          frameContext, scene, *renderer, settings, temporalFrameService,
          camera, frameIndex, timeSeconds, benchmarkCase.fixedDeltaSeconds,
          benchmarkCase.resolution[0], benchmarkCase.resolution[1]);
      if (measured) {
        renderer->renderGraphTelemetry().requestCapture(
            RenderGraphTelemetryLevel::PassTimings);
      }
      const auto renderBegin = std::chrono::steady_clock::now();
      auto renderResult = renderer->render(pipeline, frameContext);
      const double renderSubmitMs = elapsedMs(renderBegin);
      if (renderResult.hasError()) {
        return Result<bool, std::string>::makeError(renderResult.error());
      }
      const double totalMs = elapsedMs(totalBegin);
      frame.metrics = frameContext.metrics;
      if (!measured) {
        // Warmup stability uses the same submission timer as the measured
        // phase. Cooldown records are ignored by the report's phase bounds.
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("cpu.render_submit_ms"), renderSubmitMs);
      } else {
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("cpu.total_ms"), totalMs);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("cpu.scene_commit_ms"), sceneCommitMs);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("cpu.render_submit_ms"), renderSubmitMs);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("benchmark.camera.position_delta"),
            cameraPositionDelta);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("benchmark.camera.direction_delta"),
            cameraDirectionDelta);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("cpu.scene_resource_prepare_ms"),
            sceneResourcePrepareMs);
        addRendererFrameMetrics(frame.measurements, frame.metrics);
        addRenderGraphTelemetryMetrics(frame.measurements, *renderer);
        addTextureResourceMetrics(frame.measurements, *gpu);
        addProcessMemoryMetrics(frame.measurements);
        addPmrMemoryMetrics(frame.measurements, rendererMemoryTracker,
                            pipelineMemoryTracker, sceneMemoryTracker);
        measuredFrameByIndex.emplace(frame.frameIndex, report.frames.size());
      }
      report.frames.push_back(std::move(frame));
      ++frameIndex;
      timeSeconds += benchmarkCase.fixedDeltaSeconds;
      drainGpuTimings(*gpu, report, measuredFrameByIndex);
      NURI_PROFILER_ZONE_END();
      return Result<bool, std::string>::makeResult(true);
    };

    tracySession = startTracyCaptureIfRequested(benchmarkCase, options, report);
    if (tracySession.started) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    for (uint32_t sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
      temporalFrameService.reset();
      settings.antiAliasing.debug.resetHistoryRequested = true;
      for (uint32_t i = 0u; i < benchmarkCase.warmupFrames; ++i) {
        auto frameResult = renderOneFrame(sampleIndex, i, false);
        if (frameResult.hasError()) {
          result.exitCode = BenchmarkExitCode::RuntimeError;
          result.message = frameResult.error();
          report.warnings.push_back(result.message);
          report.run.validForComparison = false;
          tracySession.finish(report);
          computeBenchmarkReportStats(report);
          return finalizeResult();
        }
      }
      for (uint32_t i = 0u; i < benchmarkCase.measurementFrames; ++i) {
        auto frameResult =
            renderOneFrame(sampleIndex, benchmarkCase.warmupFrames + i, true);
        if (frameResult.hasError()) {
          result.exitCode = BenchmarkExitCode::RuntimeError;
          result.message = frameResult.error();
          report.warnings.push_back(result.message);
          report.run.validForComparison = false;
          tracySession.finish(report);
          computeBenchmarkReportStats(report);
          return finalizeResult();
        }
      }
      for (uint32_t i = 0u; i < benchmarkCase.cooldownFrames; ++i) {
        auto frameResult = renderOneFrame(
            sampleIndex,
            benchmarkCase.warmupFrames + benchmarkCase.measurementFrames + i,
            false);
        if (frameResult.hasError()) {
          break;
        }
      }
    }

    const auto drainBegin = std::chrono::steady_clock::now();
    for (uint32_t drainFrame = 0u; drainFrame < benchmarkCase.maxDrainFrames;
         ++drainFrame) {
      drainGpuTimings(*gpu, report, measuredFrameByIndex);
      report.timingDrain.drainFrames = drainFrame;
      if (elapsedMs(drainBegin) >
          static_cast<double>(benchmarkCase.drainTimeoutMs)) {
        report.timingDrain.drainComplete = false;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    report.timingDrain.droppedGpuTimingReports =
        gpu->droppedGpuTimingReportCount();
    gpu->waitIdle();
    drainGpuTimings(*gpu, report, measuredFrameByIndex);
    for (const BenchmarkFrameRecord &frame : report.frames) {
      if (!frame.measured) {
        continue;
      }
      const auto scopes = frame.measurements.find("gpu.scopes_sum_ms");
      if (scopes == frame.measurements.end()) {
        ++report.timingDrain.missingGpuTimingFrames;
        continue;
      }
      const auto whole = frame.measurements.find("gpu.frame_ms");
      const auto commandBuffers = frame.measurements.find(
          "rendergraph.summary.recorded_command_buffer_count");
      const auto submitBatches =
          frame.measurements.find("rendergraph.summary.submit_batch_count");
      if (whole == frame.measurements.end() ||
          commandBuffers == frame.measurements.end() ||
          submitBatches == frame.measurements.end() ||
          commandBuffers->second != 1.0 || submitBatches->second != 1.0) {
        continue;
      }
      constexpr double kGpuTimingContainmentToleranceMs = 0.05;
      if (scopes->second > whole->second + kGpuTimingContainmentToleranceMs) {
        ++report.timingDrain.scopeContainmentViolations;
      }
    }
    if (report.timingDrain.droppedGpuTimingReports > 0u) {
      report.run.validForComparison = false;
      report.warnings.push_back(
          "GPU timing reports were dropped; run is not comparable");
    }
    if (report.timingDrain.missingGpuTimingFrames > 0u) {
      report.run.validForComparison = false;
      report.warnings.push_back(
          "one or more measured frames are missing GPU timing reports");
    }
    if (report.timingDrain.scopeContainmentViolations > 0u) {
      report.run.validForComparison = false;
      report.warnings.push_back(
          std::to_string(report.timingDrain.scopeContainmentViolations) +
          " single-command-buffer frames report scoped GPU work outside the "
          "same frame's whole-GPU interval");
    }
  } catch (const std::exception &ex) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message = ex.what();
    report.run.validForComparison = false;
    report.warnings.push_back(result.message);
  }
  tracySession.finish(report);

  computeBenchmarkReportStats(report);
  const auto markUnavailable = [&](std::string_view metric) {
    if (report.stats.find(std::string(metric)) == report.stats.end() &&
        std::find(report.unavailableMetrics.begin(),
                  report.unavailableMetrics.end(),
                  metric) == report.unavailableMetrics.end()) {
      report.unavailableMetrics.emplace_back(metric);
    }
  };
  markUnavailable("gpu.frame_ms");
  markUnavailable("gpu.scopes_sum_ms");
  markUnavailable("renderer.aa.motion_class_invalid_pixels");
  markUnavailable("renderer.aa.motion_class_static_camera_only_pixels");
  markUnavailable("renderer.aa.motion_class_full_pixels");
  markUnavailable("renderer.aa.motion_class_background_rotation_pixels");
  markUnavailable("renderer.aa.motion_class_invalid_ratio");
  markUnavailable("renderer.aa.motion_class_static_camera_only_ratio");
  markUnavailable("renderer.aa.motion_class_full_ratio");
  markUnavailable("renderer.aa.motion_class_background_rotation_ratio");
  if (result.exitCode == BenchmarkExitCode::Success) {
    result.message = "benchmark run complete";
  }
  return finalizeResult();
}

BenchmarkSuiteRunResult
runBenchmarkSuite(std::vector<BenchmarkCase> benchmarkCases,
                  std::string_view suite, const BenchmarkRunOptions &options) {
  BenchmarkSuiteRunResult suiteResult{};
  std::string runId = nuri::tools::core::createRunId();
  std::filesystem::path artifactDir = options.artifactDir;
  if (artifactDir.empty()) {
    auto workspace = nuri::tools::core::createRunWorkspace(
        benchmarkRepoRoot() / "artifacts" / "bench");
    if (workspace.hasError()) {
      suiteResult.exitCode = BenchmarkExitCode::RuntimeError;
      suiteResult.message = workspace.error();
      return suiteResult;
    }
    runId = workspace.value().runId;
    artifactDir = workspace.value().root;
  }
  suiteResult.reportPath =
      options.jsonOut.empty() ? artifactDir / "run.json" : options.jsonOut;

  std::vector<nuri::tools::core::CaseCatalogEntry> catalog;
  catalog.reserve(benchmarkCases.size());
  for (const BenchmarkCase &benchmarkCase : benchmarkCases) {
    catalog.push_back({.id = benchmarkCase.id,
                       .suite = benchmarkCase.suite,
                       .manifestPath = benchmarkCase.manifestPath});
  }
  auto selected = nuri::tools::core::selectCaseCatalog(
      catalog,
      nuri::tools::core::CaseCatalogSelector{.suite = std::string(suite)},
      nuri::tools::core::CaseCatalogZeroMatchPolicy::Reject, "benchmark");
  if (selected.hasError()) {
    suiteResult.exitCode = BenchmarkExitCode::InvalidInput;
    suiteResult.message = selected.error();
    auto written =
        writeBenchmarkSuiteEnvelope(suiteResult, suite, options, artifactDir,
                                    suiteResult.reportPath, runId);
    if (written.hasError()) {
      suiteResult.exitCode = BenchmarkExitCode::RuntimeError;
      suiteResult.message = written.error();
    }
    return suiteResult;
  }
  for (const size_t index : selected.value()) {
    BenchmarkCase &benchmarkCase = benchmarkCases[index];
    BenchmarkRunOptions caseOptions = options;
    caseOptions.artifactDir = artifactDir;
    caseOptions.jsonOut.clear();
    caseOptions.envelopeOut =
        artifactDir / "cases" / (benchmarkCase.id + ".result.json");
    auto result =
        options.isolatedRepetitions.has_value()
            ? runBenchmarkCaseIsolated(std::move(benchmarkCase), caseOptions)
            : runBenchmarkCase(std::move(benchmarkCase), caseOptions);
    const auto aggregate = nuri::tools::core::aggregateOutcome(
        benchmarkOutcome(suiteResult.exitCode),
        benchmarkOutcome(result.exitCode));
    if (aggregate == benchmarkOutcome(result.exitCode)) {
      suiteResult.exitCode = result.exitCode;
    }
    suiteResult.caseResults.push_back(std::move(result));
  }
  suiteResult.message = suiteResult.exitCode == BenchmarkExitCode::Success
                            ? "suite run complete"
                            : "suite run completed with failures";
  auto written = writeBenchmarkSuiteEnvelope(
      suiteResult, suite, options, artifactDir, suiteResult.reportPath, runId);
  if (written.hasError()) {
    suiteResult.exitCode = BenchmarkExitCode::RuntimeError;
    suiteResult.message = written.error();
  }
  return suiteResult;
}

} // namespace nuri::tools::benchmark
