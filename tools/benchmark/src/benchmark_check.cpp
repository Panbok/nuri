#include "nuri/tools/benchmark/benchmark_check.h"

#include "nuri/tools/benchmark/benchmark_environment.h"
#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/result_envelope_v2.h"
#include "nuri/tools/core/result_protocol.h"
#include "nuri/tools/core/run_workspace.h"
#include "nuri/tools/core/safe_path.h"
#include "nuri/tools/core/sha256.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <set>
#include <utility>

#include <yyjson.h>

namespace nuri::tools::benchmark {
namespace {

using nuri::tools::core::ResultArtifactStatusV2;
using nuri::tools::core::ResultArtifactV2;
using nuri::tools::core::ResultDiagnosticSeverityV2;
using nuri::tools::core::ResultDiagnosticV2;
using nuri::tools::core::ResultEnvelopeV2;
using nuri::tools::core::ResultProfileV2;
using nuri::tools::core::ResultToolV2;
using nuri::tools::core::ToolOutcome;

[[nodiscard]] ToolOutcome benchmarkOutcome(BenchmarkExitCode code,
                                           bool authoritative = false) {
  switch (code) {
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

[[nodiscard]] BenchmarkExitCode benchmarkExit(ToolOutcome outcome) {
  return static_cast<BenchmarkExitCode>(
      nuri::tools::core::toolOutcomeExitCode(outcome));
}

[[nodiscard]] std::string pathToUtf8(const std::filesystem::path &path) {
  const std::u8string encoded = path.generic_u8string();
  return {reinterpret_cast<const char *>(encoded.data()), encoded.size()};
}

[[nodiscard]] std::filesystem::path
artifactRoot(const BenchmarkCheckOptions &options) {
  return options.artifactRoot.empty()
             ? benchmarkRepoRoot() / "artifacts" / "bench"
             : options.artifactRoot;
}

[[nodiscard]] ResultArtifactV2
artifactFor(std::string role, const std::filesystem::path &path,
            const std::filesystem::path &workspace) {
  std::error_code relativeError;
  const std::filesystem::path relative =
      std::filesystem::relative(path, workspace, relativeError);
  auto digest = nuri::tools::core::sha256File(path);
  if (relativeError || digest.hasError()) {
    return {.role = std::move(role),
            .path = relativeError ? path.filename() : relative,
            .mediaType = "application/json",
            .digest = std::string("sha256:") + std::string(64u, '0'),
            .status = ResultArtifactStatusV2::Invalid};
  }
  return {.role = std::move(role),
          .path = relative,
          .mediaType = "application/json",
          .digest = "sha256:" + digest.value(),
          .status = ResultArtifactStatusV2::Complete};
}

[[nodiscard]] Result<std::string, std::string>
makePayload(const BenchmarkCheckResult &result,
            const nuri::tools::core::BaselineProfile &profile,
            const BenchmarkCheckOptions &options) {
  using Doc = std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;
  Doc document(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!document) {
    return Result<std::string, std::string>::makeError(
        "benchmark check payload allocation failed");
  }
  yyjson_mut_val *root = yyjson_mut_obj(document.get());
  yyjson_mut_doc_set_root(document.get(), root);
  yyjson_mut_obj_add_uint(document.get(), root, "schemaVersion", 1u);
  yyjson_mut_obj_add_strcpy(document.get(), root, "kind",
                            "nuri.benchmark.check");
  yyjson_mut_obj_add_strcpy(document.get(), root, "profileId",
                            profile.id.c_str());
  yyjson_mut_obj_add_bool(document.get(), root, "force", options.force);
  yyjson_mut_val *cases = yyjson_mut_arr(document.get());
  for (const BenchmarkCheckCaseResult &caseResult : result.cases) {
    yyjson_mut_val *object = yyjson_mut_obj(document.get());
    yyjson_mut_obj_add_strcpy(document.get(), object, "id",
                              caseResult.caseId.c_str());
    yyjson_mut_obj_add_int(document.get(), object, "exitCode",
                           static_cast<int>(caseResult.exitCode));
    yyjson_mut_obj_add_bool(document.get(), object, "baselineValid",
                            caseResult.baselineVerification.valid);
    const bool comparisonProduced = !caseResult.comparisonPath.empty();
    yyjson_mut_obj_add_bool(document.get(), object, "comparisonProduced",
                            comparisonProduced);
    yyjson_mut_obj_add_bool(document.get(), object, "comparisonValid",
                            comparisonProduced && caseResult.comparison.valid);
    yyjson_mut_obj_add_bool(document.get(), object, "authoritative",
                            caseResult.comparison.authoritative);
    yyjson_mut_obj_add_bool(document.get(), object, "regression",
                            caseResult.comparison.regression);
    if (!caseResult.run.reportPath.empty()) {
      const std::filesystem::path relative = std::filesystem::relative(
          caseResult.run.reportPath, result.workspace);
      const std::string text = pathToUtf8(relative);
      yyjson_mut_obj_add_strncpy(document.get(), object, "report", text.data(),
                                 text.size());
    } else {
      yyjson_mut_obj_add_null(document.get(), object, "report");
    }
    if (!caseResult.comparisonPath.empty()) {
      const std::filesystem::path relative = std::filesystem::relative(
          caseResult.comparisonPath, result.workspace);
      const std::string text = pathToUtf8(relative);
      yyjson_mut_obj_add_strncpy(document.get(), object, "comparison",
                                 text.data(), text.size());
    } else {
      yyjson_mut_obj_add_null(document.get(), object, "comparison");
    }
    yyjson_mut_arr_add_val(cases, object);
  }
  yyjson_mut_obj_add_val(document.get(), root, "cases", cases);
  size_t length = 0u;
  char *json = yyjson_mut_write_opts(document.get(), YYJSON_WRITE_PRETTY,
                                     nullptr, &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "benchmark check payload serialization failed");
  }
  std::string text(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(text));
}

void addDiagnostic(ResultEnvelopeV2 &envelope, std::string code,
                   ResultDiagnosticSeverityV2 severity, std::string message) {
  envelope.diagnostics.push_back({.code = std::move(code),
                                  .severity = severity,
                                  .message = std::move(message)});
}

[[nodiscard]] Result<void, std::string>
writeEnvelope(BenchmarkCheckResult &result,
              const nuri::tools::core::BaselineProfile &profile,
              const BenchmarkCheckOptions &options, std::string_view runId,
              ToolOutcome outcome, const std::vector<std::string> &reasons) {
  ResultEnvelopeV2 envelope{};
  envelope.tool = ResultToolV2::Benchmark;
  envelope.runId = std::string(runId);
  envelope.status = outcome;
  envelope.exitCode = nuri::tools::core::toolOutcomeExitCode(outcome);
  envelope.authoritative = outcome == ToolOutcome::Pass &&
                           profile.authority.authoritative && !options.force;
  envelope.reproduceCommand = options.command;
  envelope.selection.requested = options.requestedSelection;
  envelope.selection.selected = result.cases.size();
  envelope.profile =
      ResultProfileV2{.id = profile.id,
                      .compatible = reasons.empty() && !options.force,
                      .incompatibilityReasons = reasons};
  for (const BenchmarkCheckCaseResult &caseResult : result.cases) {
    const bool attempted = !caseResult.run.reportPath.empty();
    if (attempted) {
      ++envelope.selection.attempted;
    }
    switch (caseResult.exitCode) {
    case BenchmarkExitCode::Success:
      ++envelope.selection.completed;
      if (caseResult.comparison.authoritative) {
        ++envelope.selection.passed;
      } else {
        ++envelope.selection.warned;
      }
      break;
    case BenchmarkExitCode::Regression:
      ++envelope.selection.completed;
      ++envelope.selection.failed;
      break;
    case BenchmarkExitCode::EnvironmentUnavailable:
      ++envelope.selection.unavailable;
      break;
    case BenchmarkExitCode::InvalidInput:
    case BenchmarkExitCode::RuntimeError:
    case BenchmarkExitCode::MissingBaseline:
      ++envelope.selection.notRun;
      break;
    }
    if (!caseResult.baselineVerificationPath.empty()) {
      envelope.artifacts.push_back(
          artifactFor("benchmark.check.baseline_verification",
                      caseResult.baselineVerificationPath, result.workspace));
    }
    if (!caseResult.run.reportPath.empty() &&
        std::filesystem::is_regular_file(caseResult.run.reportPath)) {
      envelope.artifacts.push_back(artifactFor("benchmark.check.current_report",
                                               caseResult.run.reportPath,
                                               result.workspace));
    }
    if (!caseResult.run.envelopePath.empty() &&
        std::filesystem::is_regular_file(caseResult.run.envelopePath)) {
      envelope.artifacts.push_back(artifactFor("benchmark.check.case_result",
                                               caseResult.run.envelopePath,
                                               result.workspace));
    }
    if (!caseResult.comparisonPath.empty()) {
      envelope.artifacts.push_back(artifactFor("benchmark.check.comparison",
                                               caseResult.comparisonPath,
                                               result.workspace));
    }
    envelope.children.push_back(
        {.id = caseResult.caseId,
         .status =
             std::string(nuri::tools::core::toolOutcomeName(benchmarkOutcome(
                 caseResult.exitCode, caseResult.comparison.authoritative))),
         .exitCode = static_cast<int>(caseResult.exitCode),
         .result =
             caseResult.comparisonPath.empty()
                 ? std::optional<std::filesystem::path>{}
                 : std::optional<std::filesystem::path>{
                       std::filesystem::relative(caseResult.comparisonPath,
                                                 result.workspace)}});
  }
  for (const std::string &reason : reasons) {
    addDiagnostic(envelope, "benchmark.check.incompatible",
                  options.force ? ResultDiagnosticSeverityV2::Warning
                                : ResultDiagnosticSeverityV2::Error,
                  reason);
  }
  auto payload = makePayload(result, profile, options);
  if (payload.hasError()) {
    return Result<void, std::string>::makeError(payload.error());
  }
  envelope.payloadJson = std::move(payload.value());
  return nuri::tools::core::writeResultEnvelopeV2(result.envelopePath,
                                                  envelope);
}

[[nodiscard]] std::vector<std::string>
preflightProfile(const std::vector<BenchmarkCase> &cases,
                 const nuri::tools::core::BaselineProfile &profile) {
  std::set<std::string> reasons;
  for (const BenchmarkCase &benchmarkCase : cases) {
    std::string backendSource = "manifest";
    std::string backend = benchmarkCase.backend;
    if (backend == "default") {
      backend = readProcessEnvironment("NURI_GPU_BACKEND");
      backendSource = backend.empty() ? "default" : "NURI_GPU_BACKEND";
      if (backend.empty()) {
        backend = "nvrhi";
      }
    }
    std::string presentMode = benchmarkCase.presentMode;
    if (presentMode == "default") {
      presentMode = readProcessEnvironment("NURI_PRESENT_MODE");
      if (presentMode.empty()) {
        presentMode = "default";
      }
    }
    BenchmarkEnvironment environment = collectBenchmarkEnvironment(
        backend, backendSource, presentMode, "preflight", false);
    const auto compatibility = nuri::tools::core::evaluateBaselineProfile(
        profile,
        {.os = environment.osName,
         .backend = backend,
         .backendSource = backendSource,
         .windowMode = "windowed",
         .windowVisible = true,
         // GPU identity is runtime-owned. Mirror pinned values so preflight
         // evaluates only facts available without renderer initialization.
         .gpuVendorId = profile.environment.gpuVendorId.value_or(0u),
         .gpuDeviceId = profile.environment.gpuDeviceId.value_or(0u),
         .driver = profile.environment.driver.value_or("unknown"),
         .presentMode = presentMode,
         .profiling = profile.execution.profiling,
         .devChecks = profile.execution.devChecks,
         .dirtyTree = environment.dirty});
    for (const std::string &reason : compatibility.reasons) {
      reasons.insert(benchmarkCase.id + ": " + reason);
    }
  }
  return {reasons.begin(), reasons.end()};
}

} // namespace

BenchmarkCheckResult
checkBenchmarkCases(std::vector<BenchmarkCase> selectedCases,
                    const nuri::tools::core::BaselineProfile &profile,
                    const BenchmarkCheckOptions &options) {
  BenchmarkCheckResult result{};
  auto workspace = nuri::tools::core::createRunWorkspace(artifactRoot(options));
  if (workspace.hasError()) {
    result.message = workspace.error();
    return result;
  }
  result.workspace = workspace.value().root;
  result.envelopePath = result.workspace / "run.json";
  result.cases.reserve(selectedCases.size());
  for (const BenchmarkCase &benchmarkCase : selectedCases) {
    result.cases.push_back({.caseId = benchmarkCase.id});
  }

  ToolOutcome outcome = ToolOutcome::Pass;
  std::vector<std::string> incompatibilityReasons;
  if (selectedCases.empty()) {
    outcome = ToolOutcome::Invalid;
    incompatibilityReasons.push_back(
        "benchmark check selector matched no cases");
  }
  if (options.processExecutable.empty()) {
    outcome =
        nuri::tools::core::aggregateOutcome(outcome, ToolOutcome::Invalid);
    incompatibilityReasons.push_back(
        "benchmark check requires a process executable");
  }
  if (profile.benchmarkPolicy.minimumRepetitions == 0u) {
    outcome =
        nuri::tools::core::aggregateOutcome(outcome, ToolOutcome::Invalid);
    incompatibilityReasons.push_back(
        "benchmark profile must own at least one isolated repetition");
  }

  // Verify every governed baseline before starting any renderer process.
  for (size_t index = 0u; index < selectedCases.size(); ++index) {
    const BenchmarkCase &benchmarkCase = selectedCases[index];
    BenchmarkCheckCaseResult &caseResult = result.cases[index];
    auto verification = verifyBenchmarkBaseline(
        benchmarkCase.id, benchmarkCase.suite, profile, options.baselineRoot);
    if (verification.hasError()) {
      caseResult.exitCode = BenchmarkExitCode::InvalidInput;
      incompatibilityReasons.push_back(benchmarkCase.id + ": " +
                                       verification.error());
      outcome =
          nuri::tools::core::aggregateOutcome(outcome, ToolOutcome::Invalid);
      continue;
    }
    caseResult.baselineVerification = std::move(verification.value());
    caseResult.baselineVerificationPath =
        result.workspace / "baseline" / (benchmarkCase.id + ".json");
    auto verificationJson = writeBenchmarkBaselineVerificationJson(
        caseResult.baselineVerification,
        "nuri.benchmark.baseline_verification");
    auto verificationWrite =
        verificationJson.hasError()
            ? Result<void, std::string>::makeError(verificationJson.error())
            : nuri::tools::core::atomicWriteTextFile(
                  caseResult.baselineVerificationPath,
                  verificationJson.value());
    if (verificationWrite.hasError()) {
      caseResult.exitCode = BenchmarkExitCode::RuntimeError;
      incompatibilityReasons.push_back(verificationWrite.error());
      outcome = nuri::tools::core::aggregateOutcome(outcome,
                                                    ToolOutcome::RuntimeError);
    } else if (!caseResult.baselineVerification.exists) {
      caseResult.exitCode = BenchmarkExitCode::MissingBaseline;
      outcome = nuri::tools::core::aggregateOutcome(
          outcome, ToolOutcome::MissingBaseline);
    } else if (!caseResult.baselineVerification.valid) {
      caseResult.exitCode = BenchmarkExitCode::InvalidInput;
      for (const std::string &error : caseResult.baselineVerification.errors) {
        incompatibilityReasons.push_back(benchmarkCase.id + ": " + error);
      }
      outcome =
          nuri::tools::core::aggregateOutcome(outcome, ToolOutcome::Invalid);
    }
  }

  const bool baselinePreflightPassed =
      outcome == ToolOutcome::Pass && !selectedCases.empty();
  if (baselinePreflightPassed) {
    std::vector<std::string> profilePreflight =
        preflightProfile(selectedCases, profile);
    if (!profilePreflight.empty()) {
      incompatibilityReasons.insert(incompatibilityReasons.end(),
                                    profilePreflight.begin(),
                                    profilePreflight.end());
      if (!options.force) {
        outcome = ToolOutcome::EnvironmentUnavailable;
        for (BenchmarkCheckCaseResult &caseResult : result.cases) {
          caseResult.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
        }
      }
    }
  }

  if (outcome == ToolOutcome::Pass ||
      (options.force && baselinePreflightPassed)) {
    outcome = ToolOutcome::Pass;
    for (size_t index = 0u; index < selectedCases.size(); ++index) {
      BenchmarkCase &benchmarkCase = selectedCases[index];
      BenchmarkCheckCaseResult &caseResult = result.cases[index];
      auto governed = loadVerifiedBenchmarkBaseline(
          benchmarkCase.id, benchmarkCase.suite, profile, options.baselineRoot);
      if (governed.hasError()) {
        caseResult.exitCode = BenchmarkExitCode::InvalidInput;
        incompatibilityReasons.push_back(benchmarkCase.id + ": " +
                                         governed.error());
        outcome =
            nuri::tools::core::aggregateOutcome(outcome, ToolOutcome::Invalid);
        continue;
      }
      auto caseWorkspace = nuri::tools::core::resolvePathUnder(
          result.workspace, std::filesystem::path("cases") / benchmarkCase.id);
      if (caseWorkspace.hasError()) {
        caseResult.exitCode = BenchmarkExitCode::RuntimeError;
        incompatibilityReasons.push_back(caseWorkspace.error());
        outcome = nuri::tools::core::aggregateOutcome(
            outcome, ToolOutcome::RuntimeError);
        continue;
      }
      BenchmarkRunOptions runOptions{
          .isolatedRepetitions = profile.benchmarkPolicy.minimumRepetitions,
          .processExecutable = options.processExecutable,
          .repetitionTimeout = options.repetitionTimeout,
          .jsonOut = caseWorkspace.value() / "report.json",
          .envelopeOut = caseWorkspace.value() / "run.json",
          .artifactDir = caseWorkspace.value(),
          .baselineProfileId = profile.id,
          .baselineProfileAuthoritative =
              profile.authority.authoritative && !options.force,
          .baselineProfileMinimumRepetitions =
              profile.benchmarkPolicy.minimumRepetitions,
          .baselineProfileWarmupStability =
              profile.benchmarkPolicy.warmupStability,
          .baselineProfileWarmupWindowFrames =
              profile.benchmarkPolicy.warmupWindowFrames,
          .baselineProfileWarmupMaxDriftPercent =
              profile.benchmarkPolicy.warmupMaxDriftPercent,
          .baselineProfileRequiredMetrics =
              profile.benchmarkPolicy.requiredMetrics,
          .baselineProfile = profile,
          .baselineProfileWarning =
              options.force
                  ? "--force demoted the governed check to investigative"
                  : std::string{},
          .command = options.command,
      };
      caseResult.run = runBenchmarkCaseIsolated(benchmarkCase, runOptions);
      if (caseResult.run.exitCode != BenchmarkExitCode::Success) {
        caseResult.exitCode = caseResult.run.exitCode;
        outcome = nuri::tools::core::aggregateOutcome(
            outcome, benchmarkOutcome(caseResult.exitCode));
        if (caseResult.exitCode == BenchmarkExitCode::EnvironmentUnavailable) {
          incompatibilityReasons.push_back(benchmarkCase.id + ": " +
                                           caseResult.run.message);
        }
        continue;
      }
      const bool investigativeComparison =
          options.force || !profile.authority.authoritative;
      caseResult.comparison = compareBenchmarkReports(
          caseResult.run.report, governed.value().report,
          BenchmarkCompareOptions{.force = investigativeComparison});
      caseResult.comparisonPath =
          result.workspace / "comparisons" / (benchmarkCase.id + ".json");
      auto comparisonJson = writeBenchmarkComparisonJson(caseResult.comparison);
      auto comparisonWrite =
          comparisonJson.hasError()
              ? Result<void, std::string>::makeError(comparisonJson.error())
              : nuri::tools::core::atomicWriteTextFile(
                    caseResult.comparisonPath, comparisonJson.value());
      if (comparisonWrite.hasError()) {
        caseResult.exitCode = BenchmarkExitCode::RuntimeError;
        incompatibilityReasons.push_back(comparisonWrite.error());
        outcome = nuri::tools::core::aggregateOutcome(
            outcome, ToolOutcome::RuntimeError);
      } else if (!caseResult.comparison.valid) {
        caseResult.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
        for (const std::string &error : caseResult.comparison.errors) {
          incompatibilityReasons.push_back(benchmarkCase.id + ": " + error);
        }
        outcome = nuri::tools::core::aggregateOutcome(
            outcome, ToolOutcome::EnvironmentUnavailable);
      } else if (caseResult.comparison.regression) {
        caseResult.exitCode = BenchmarkExitCode::Regression;
        outcome =
            nuri::tools::core::aggregateOutcome(outcome, ToolOutcome::Failure);
      } else {
        caseResult.exitCode = BenchmarkExitCode::Success;
      }
    }
    if (outcome == ToolOutcome::Pass &&
        (options.force || !profile.authority.authoritative)) {
      outcome = ToolOutcome::Investigative;
    }
  }

  result.exitCode = benchmarkExit(outcome);
  result.message = outcome == ToolOutcome::Pass
                       ? "governed benchmark check passed"
                   : outcome == ToolOutcome::Investigative
                       ? "governed benchmark check completed investigatively"
                       : "governed benchmark check did not pass";
  auto envelope =
      writeEnvelope(result, profile, options, workspace.value().runId, outcome,
                    incompatibilityReasons);
  if (envelope.hasError()) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message = envelope.error();
  }
  return result;
}

} // namespace nuri::tools::benchmark
