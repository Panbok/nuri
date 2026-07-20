#include "nuri/tools/benchmark/benchmark_baseline.h"
#include "nuri/tools/benchmark/benchmark_check.h"
#include "nuri/tools/benchmark/benchmark_compare.h"
#include "nuri/tools/benchmark/benchmark_graph.h"
#include "nuri/tools/benchmark/benchmark_manifest.h"
#include "nuri/tools/benchmark/benchmark_report.h"
#include "nuri/tools/benchmark/benchmark_runner.h"
#include "nuri/tools/core/baseline_profile.h"

#include <CLI/CLI.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace nuri::tools::benchmark;
using nuri::Result;
using nuri::tools::core::BaselineProfile;
using nuri::tools::core::loadBaselineProfile;

namespace {

[[nodiscard]] int exitCode(BenchmarkExitCode code) {
  return static_cast<int>(code);
}

[[nodiscard]] std::vector<BenchmarkCase> loadCasesOrExit() {
  auto cases = discoverBenchmarkCases();
  if (cases.hasError()) {
    std::cerr << cases.error() << "\n";
    std::exit(exitCode(BenchmarkExitCode::InvalidInput));
  }
  return std::move(cases.value());
}

[[nodiscard]] const BenchmarkCase &
requireCase(const std::vector<BenchmarkCase> &cases, std::string_view id) {
  const BenchmarkCase *benchmarkCase = findBenchmarkCaseById(cases, id);
  if (benchmarkCase == nullptr) {
    std::cerr << "unknown benchmark case: " << id << "\n";
    std::exit(exitCode(BenchmarkExitCode::InvalidInput));
  }
  return *benchmarkCase;
}

void writeTextFile(const std::filesystem::path &path, const std::string &text) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  file << text;
}

[[nodiscard]] BenchmarkGraphOptions
makeGraphOptions(const std::vector<std::string> &metrics,
                 const std::vector<std::string> &statistics,
                 std::string title = "Nuri Benchmark Results") {
  BenchmarkGraphOptions options{};
  options.metrics = metrics;
  options.statistics = statistics;
  options.title = std::move(title);
  return options;
}

void writeGraphOrExit(const std::vector<BenchmarkReport> &reports,
                      const std::filesystem::path &htmlOut,
                      const BenchmarkGraphOptions &options) {
  if (htmlOut.empty()) {
    return;
  }
  auto written = writeBenchmarkGraphHtmlFile(reports, options, htmlOut);
  if (written.hasError()) {
    std::cerr << written.error() << "\n";
    std::exit(exitCode(BenchmarkExitCode::RuntimeError));
  }
  std::cout << "html: " << htmlOut.generic_string() << "\n";
}

[[nodiscard]] Result<bool, std::string>
appendBenchmarkReportFile(std::vector<BenchmarkReport> &reports,
                          const std::filesystem::path &path) {
  auto report = readBenchmarkReportFile(path);
  if (report.hasError()) {
    return Result<bool, std::string>::makeError(report.error());
  }
  if (report.value().kind != "nuri.benchmark.report") {
    return Result<bool, std::string>::makeResult(false);
  }
  reports.push_back(std::move(report.value()));
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<std::vector<BenchmarkReport>, std::string>
loadBenchmarkReports(const std::vector<std::filesystem::path> &reportPaths,
                     const std::vector<std::filesystem::path> &reportDirs) {
  std::vector<BenchmarkReport> reports;
  for (const std::filesystem::path &path : reportPaths) {
    auto appended = appendBenchmarkReportFile(reports, path);
    if (appended.hasError()) {
      return Result<std::vector<BenchmarkReport>, std::string>::makeError(
          appended.error());
    }
    if (!appended.value()) {
      return Result<std::vector<BenchmarkReport>, std::string>::makeError(
          "not a benchmark report: " + path.string());
    }
  }
  for (const std::filesystem::path &dir : reportDirs) {
    if (!std::filesystem::exists(dir)) {
      return Result<std::vector<BenchmarkReport>, std::string>::makeError(
          "reports directory does not exist: " + dir.string());
    }
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(dir)) {
      if (!entry.is_regular_file() || entry.path().extension() != ".json") {
        continue;
      }
      auto appended = appendBenchmarkReportFile(reports, entry.path());
      if (appended.hasError()) {
        continue;
      }
    }
  }
  return Result<std::vector<BenchmarkReport>, std::string>::makeResult(
      std::move(reports));
}

} // namespace

int main(int argc, char **argv) {
  CLI::App app{"Nuri renderer benchmark tool"};
  app.require_subcommand(1);

  std::string command = joinCommandLine(argc, argv);

  std::string listSuite;
  bool listJson = false;
  auto *list = app.add_subcommand("list", "List benchmark cases");
  list->add_option("--suite", listSuite, "Filter by suite");
  list->add_flag("--json", listJson, "Emit JSON");
  list->callback([&]() {
    const std::vector<BenchmarkCase> cases = loadCasesOrExit();
    if (listJson) {
      auto json = formatBenchmarkCaseListJson(cases, listSuite);
      if (json.hasError()) {
        std::cerr << json.error() << "\n";
        std::exit(exitCode(BenchmarkExitCode::RuntimeError));
      }
      std::cout << json.value();
    } else {
      std::cout << formatBenchmarkCaseListText(cases, listSuite);
    }
  });

  std::string explainCase;
  bool explainJson = false;
  auto *explain = app.add_subcommand("explain", "Explain a benchmark case");
  explain->add_option("--case", explainCase, "Case id")->required();
  explain->add_flag("--json", explainJson, "Emit JSON");
  explain->callback([&]() {
    const std::vector<BenchmarkCase> cases = loadCasesOrExit();
    const BenchmarkCase &benchmarkCase = requireCase(cases, explainCase);
    if (explainJson) {
      auto json = formatBenchmarkCaseExplanationJson(benchmarkCase);
      if (json.hasError()) {
        std::cerr << json.error() << "\n";
        std::exit(exitCode(BenchmarkExitCode::RuntimeError));
      }
      std::cout << json.value();
    } else {
      std::cout << formatBenchmarkCaseExplanationText(benchmarkCase);
    }
  });

  std::string runCase;
  std::string runSuite;
  std::string runProfile = "local-nvrhi-visible";
  std::optional<uint32_t> samples;
  std::optional<uint32_t> repetitions;
  uint32_t repetitionTimeoutMs = 300000u;
  std::filesystem::path jsonOut;
  std::filesystem::path artifactDir;
  std::filesystem::path runHtmlOut;
  std::vector<std::string> runHtmlMetrics;
  std::vector<std::string> runHtmlStats;
  bool dryRun = false;
  bool printEffectiveConfig = false;
  bool tracyDiagnostic = false;
  bool rgpShaderDiagnostic = false;
  std::filesystem::path rgpToolPath;
  uint32_t rgpCaptureFrame = 30u;
  uint32_t rgpTimeoutMs = 60000u;
  bool renderDocDiagnostic = false;
  std::filesystem::path renderDocToolPath;
  uint32_t renderDocCaptureFrame = 30u;
  uint32_t renderDocTimeoutMs = 60000u;
  bool verboseFrames = false;
  auto *run = app.add_subcommand("run", "Run benchmark case or suite");
  run->add_option("--case", runCase, "Case id");
  run->add_option("--suite", runSuite, "Suite name");
  run->add_option("--profile", runProfile, "Baseline execution profile");
  run->add_option("--samples", samples, "Override sample windows")
      ->check(CLI::PositiveNumber);
  run->add_option("--repetitions", repetitions,
                  "Run independent isolated-process repetitions")
      ->check(CLI::PositiveNumber);
  run->add_option("--repetition-timeout-ms", repetitionTimeoutMs,
                  "Timeout for each isolated repetition")
      ->check(CLI::PositiveNumber);
  run->add_option("--json-out", jsonOut, "Report output path");
  run->add_option("--artifact-dir", artifactDir, "Artifact directory");
  run->add_option("--html-out", runHtmlOut, "Graph HTML output path");
  run->add_option("--html-metric", runHtmlMetrics, "Metric to include in graph")
      ->expected(1, -1);
  run->add_option("--html-stat", runHtmlStats, "Statistic to include in graph")
      ->expected(1, -1);
  run->add_flag("--dry-run", dryRun, "Resolve config without renderer init");
  run->add_flag("--print-effective-config", printEffectiveConfig,
                "Print resolved config before running");
  run->add_flag("--tracy-diagnostic", tracyDiagnostic,
                "Capture benchmark-owned Tracy CPU/GPU diagnostics");
  run->add_flag("--rgp-shader-diagnostic", rgpShaderDiagnostic,
                "Capture AMD RGP shader diagnostics; never performance data");
  auto *rgpToolOption = run->add_option(
      "--rgp-tool", rgpToolPath,
      "Path to RadeonDeveloperPanelCLI; defaults to PATH lookup");
  auto *rgpFrameOption =
      run->add_option("--rgp-capture-frame", rgpCaptureFrame,
                      "Zero-based RGP diagnostic capture frame");
  auto *rgpTimeoutOption = run->add_option("--rgp-timeout-ms", rgpTimeoutMs,
                                           "RGP diagnostic process timeout")
                               ->check(CLI::PositiveNumber);
  run->add_flag("--renderdoc-diagnostic", renderDocDiagnostic,
                "Capture RenderDoc frame forensics; never performance data");
  auto *renderDocToolOption = run->add_option(
      "--renderdoc-tool", renderDocToolPath,
      "Path to renderdoccmd; defaults to PATH or Program Files lookup");
  auto *renderDocFrameOption =
      run->add_option("--renderdoc-capture-frame", renderDocCaptureFrame,
                      "Zero-based RenderDoc diagnostic capture frame");
  auto *renderDocTimeoutOption =
      run->add_option("--renderdoc-timeout-ms", renderDocTimeoutMs,
                      "RenderDoc diagnostic process timeout")
          ->check(CLI::PositiveNumber);
  run->add_flag("--verbose-frames", verboseFrames,
                "Include verbose frame renderer metrics");
  run->callback([&]() {
    const uint32_t diagnosticCount =
        static_cast<uint32_t>(tracyDiagnostic) +
        static_cast<uint32_t>(rgpShaderDiagnostic) +
        static_cast<uint32_t>(renderDocDiagnostic);
    const bool gpuDiagnostic = rgpShaderDiagnostic || renderDocDiagnostic;
    const std::string_view gpuDiagnosticFlag = rgpShaderDiagnostic
                                                   ? "--rgp-shader-diagnostic"
                                                   : "--renderdoc-diagnostic";
    if (runCase.empty() == runSuite.empty()) {
      std::cerr << "run requires exactly one of --case or --suite\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    if (diagnosticCount > 1u) {
      std::cerr << "Tracy, RGP, and RenderDoc diagnostics must be collected "
                   "in separate runs\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    if (gpuDiagnostic && runCase.empty()) {
      std::cerr << gpuDiagnosticFlag << " requires one --case\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    if (gpuDiagnostic && (repetitions.has_value() || dryRun)) {
      std::cerr << gpuDiagnosticFlag
                << " cannot be combined with --repetitions or --dry-run\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    if (!rgpShaderDiagnostic &&
        (rgpToolOption->count() > 0u || rgpFrameOption->count() > 0u ||
         rgpTimeoutOption->count() > 0u)) {
      std::cerr << "--rgp-tool, --rgp-capture-frame, and --rgp-timeout-ms "
                   "require --rgp-shader-diagnostic\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    if (!renderDocDiagnostic && (renderDocToolOption->count() > 0u ||
                                 renderDocFrameOption->count() > 0u ||
                                 renderDocTimeoutOption->count() > 0u)) {
      std::cerr << "--renderdoc-tool, --renderdoc-capture-frame, and "
                   "--renderdoc-timeout-ms require --renderdoc-diagnostic\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto loadedProfile = loadBaselineProfile(
        benchmarkRepoRoot() / "tools" / "profiles", runProfile);
    if (loadedProfile.hasError()) {
      std::cerr << loadedProfile.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    const BaselineProfile &profile = loadedProfile.value();
    if (gpuDiagnostic && profile.authority.authoritative) {
      std::cerr << gpuDiagnosticFlag
                << " requires an investigative benchmark profile\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    if (profile.authority.authoritative &&
        (!repetitions.has_value() ||
         *repetitions < profile.benchmarkPolicy.minimumRepetitions)) {
      std::cerr << "authoritative benchmark profile '" << profile.id
                << "' requires --repetitions >= "
                << profile.benchmarkPolicy.minimumRepetitions << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    std::string profileWarning;
    if (!profile.authority.authoritative) {
      profileWarning = "baseline profile '" + profile.id +
                       "' is investigative; run is not authoritative";
      if (!profile.authority.reason.empty()) {
        profileWarning += ": " + profile.authority.reason;
      }
      std::cerr << "warning: " << profileWarning << "\n";
    }
    std::error_code executableError;
    std::filesystem::path processExecutable = std::filesystem::absolute(
        std::filesystem::path(argv[0]), executableError);
    if (executableError) {
      processExecutable = std::filesystem::path(argv[0]);
    }
    const BenchmarkGpuDiagnosticOptions gpuDiagnosticOptions{
        .kind = rgpShaderDiagnostic ? BenchmarkGpuDiagnosticKind::RgpShader
                : renderDocDiagnostic
                    ? BenchmarkGpuDiagnosticKind::RenderDocFrame
                    : BenchmarkGpuDiagnosticKind::None,
        .toolPath = rgpShaderDiagnostic ? rgpToolPath : renderDocToolPath,
        .captureFrame =
            rgpShaderDiagnostic ? rgpCaptureFrame : renderDocCaptureFrame,
        .timeout = std::chrono::milliseconds(
            rgpShaderDiagnostic ? rgpTimeoutMs : renderDocTimeoutMs)};
    std::vector<BenchmarkCase> cases = loadCasesOrExit();
    BenchmarkRunOptions options{
        .samplesOverride = samples,
        .isolatedRepetitions = repetitions,
        .processExecutable = processExecutable,
        .repetitionTimeout = std::chrono::milliseconds(repetitionTimeoutMs),
        .jsonOut = jsonOut,
        .artifactDir = artifactDir,
        .dryRun = dryRun,
        .printEffectiveConfig = printEffectiveConfig,
        .tracyDiagnostic = tracyDiagnostic,
        .gpuDiagnostic = gpuDiagnosticOptions,
        .verboseFrames = verboseFrames,
        .baselineProfileId = profile.id,
        .baselineProfileAuthoritative = profile.authority.authoritative,
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
        .baselineProfileWarning = profileWarning,
        .command = command,
    };
    if (!runCase.empty()) {
      BenchmarkCase benchmarkCase = requireCase(cases, runCase);
      if (printEffectiveConfig) {
        auto effective = formatEffectiveConfigJson(benchmarkCase, options);
        if (!effective.hasError()) {
          std::cout << effective.value();
        }
      }
      BenchmarkRunResult result{};
      if (gpuDiagnosticOptions.kind ==
          BenchmarkGpuDiagnosticKind::RenderDocFrame) {
        result = runBenchmarkCaseRenderDoc(std::move(benchmarkCase), options);
      } else if (repetitions.has_value()) {
        result = runBenchmarkCaseIsolated(std::move(benchmarkCase), options);
      } else {
        result = runBenchmarkCase(std::move(benchmarkCase), options);
      }
      std::cout << result.message
                << "\nreport: " << result.reportPath.generic_string() << "\n";
      writeGraphOrExit(
          {result.report}, runHtmlOut,
          makeGraphOptions(runHtmlMetrics, runHtmlStats, "Nuri Benchmark Run"));
      std::exit(exitCode(result.exitCode));
    }

    BenchmarkSuiteRunResult suiteResult =
        runBenchmarkSuite(std::move(cases), runSuite, options);
    std::cout << suiteResult.message
              << "\nreport: " << suiteResult.reportPath.generic_string()
              << "\n";
    if (!runHtmlOut.empty()) {
      std::vector<BenchmarkReport> reports;
      reports.reserve(suiteResult.caseResults.size());
      for (const BenchmarkRunResult &caseResult : suiteResult.caseResults) {
        reports.push_back(caseResult.report);
      }
      writeGraphOrExit(reports, runHtmlOut,
                       makeGraphOptions(runHtmlMetrics, runHtmlStats,
                                        "Nuri Benchmark Suite"));
    }
    std::exit(exitCode(suiteResult.exitCode));
  });

  std::string checkCase;
  std::string checkSuite;
  std::string checkProfile;
  std::filesystem::path checkArtifactRoot;
  std::filesystem::path checkBaselineRoot;
  uint32_t checkRepetitionTimeoutMs = 300000u;
  bool forceCheck = false;
  auto *check = app.add_subcommand(
      "check", "Run and compare against verified governed baselines");
  check->add_option("--case", checkCase, "Case id");
  check->add_option("--suite", checkSuite, "Suite name");
  check->add_option("--profile", checkProfile, "Baseline execution profile")
      ->required();
  check->add_option("--artifact-root", checkArtifactRoot,
                    "Root under which one invocation workspace is created");
  check->add_option("--baseline-root", checkBaselineRoot,
                    "Override governed baseline root");
  check
      ->add_option("--repetition-timeout-ms", checkRepetitionTimeoutMs,
                   "Timeout for each isolated repetition")
      ->check(CLI::PositiveNumber);
  check->add_flag(
      "--force", forceCheck,
      "Demote profile/environment incompatibility to investigative evidence");
  check->callback([&]() {
    if (checkCase.empty() == checkSuite.empty()) {
      std::cerr << "check requires exactly one of --case or --suite\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto loadedProfile = loadBaselineProfile(
        benchmarkRepoRoot() / "tools" / "profiles", checkProfile);
    if (loadedProfile.hasError()) {
      std::cerr << loadedProfile.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    std::vector<BenchmarkCase> cases = loadCasesOrExit();
    std::vector<BenchmarkCase> selected;
    if (!checkCase.empty()) {
      const BenchmarkCase *benchmarkCase =
          findBenchmarkCaseById(cases, checkCase);
      if (benchmarkCase != nullptr) {
        selected.push_back(*benchmarkCase);
      }
    } else {
      for (const BenchmarkCase &benchmarkCase : cases) {
        if (benchmarkCase.suite == checkSuite) {
          selected.push_back(benchmarkCase);
        }
      }
    }
    std::error_code executableError;
    std::filesystem::path processExecutable = std::filesystem::absolute(
        std::filesystem::path(argv[0]), executableError);
    if (executableError) {
      processExecutable = std::filesystem::path(argv[0]);
    }
    const std::string requested =
        !checkCase.empty() ? "case:" + checkCase : "suite:" + checkSuite;
    BenchmarkCheckResult result =
        checkBenchmarkCases(std::move(selected), loadedProfile.value(),
                            {.processExecutable = processExecutable,
                             .artifactRoot = checkArtifactRoot,
                             .baselineRoot = checkBaselineRoot,
                             .repetitionTimeout = std::chrono::milliseconds(
                                 checkRepetitionTimeoutMs),
                             .force = forceCheck,
                             .requestedSelection = requested,
                             .command = command});
    std::cout << result.message
              << "\nworkspace: " << result.workspace.generic_string()
              << "\nreport: " << result.envelopePath.generic_string() << "\n";
    std::exit(exitCode(result.exitCode));
  });

  std::string childCase;
  std::string childProfile;
  std::optional<uint32_t> childSamples;
  std::filesystem::path childArtifactDir;
  uint32_t childRepetitionIndex = 0u;
  bool childDryRun = false;
  bool childTracyDiagnostic = false;
  bool childRenderDocDiagnostic = false;
  std::filesystem::path childRenderDocToolPath;
  uint32_t childRenderDocCaptureFrame = 30u;
  bool childVerboseFrames = false;
  auto *child = app.add_subcommand(
      "__run-child", "Internal isolated benchmark repetition worker");
  child->add_option("--case", childCase, "Case id")->required();
  child->add_option("--profile", childProfile, "Baseline execution profile");
  child->add_option("--samples", childSamples, "Override sample windows")
      ->check(CLI::PositiveNumber);
  child
      ->add_option("--artifact-dir", childArtifactDir,
                   "Confined repetition workspace")
      ->required();
  child
      ->add_option("--repetition-index", childRepetitionIndex,
                   "Zero-based parent repetition index")
      ->required();
  child->add_flag("--dry-run", childDryRun,
                  "Resolve config without renderer init");
  child->add_flag("--tracy-diagnostic", childTracyDiagnostic,
                  "Capture benchmark-owned Tracy CPU/GPU diagnostics");
  child->add_flag("--renderdoc-diagnostic", childRenderDocDiagnostic,
                  "Internal RenderDoc frame-forensics worker");
  child->add_option("--renderdoc-tool", childRenderDocToolPath,
                    "RenderDoc launcher path");
  child->add_option("--renderdoc-capture-frame", childRenderDocCaptureFrame,
                    "Zero-based RenderDoc capture frame");
  child->add_flag("--verbose-frames", childVerboseFrames,
                  "Include verbose frame renderer metrics");
  child->callback([&]() {
    (void)childRepetitionIndex;
    std::optional<BaselineProfile> profile;
    std::string profileWarning;
    if (!childProfile.empty()) {
      auto loadedProfile = loadBaselineProfile(
          benchmarkRepoRoot() / "tools" / "profiles", childProfile);
      if (loadedProfile.hasError()) {
        std::cerr << loadedProfile.error() << "\n";
        std::exit(exitCode(BenchmarkExitCode::InvalidInput));
      }
      profile = std::move(loadedProfile.value());
      if (!profile->authority.authoritative) {
        profileWarning =
            "baseline profile '" + profile->id + "' is investigative; " +
            (childRenderDocDiagnostic
                 ? "RenderDoc diagnostic is not authoritative"
                 : "child repetition is not independently authoritative");
      }
    }
    std::vector<BenchmarkCase> cases = loadCasesOrExit();
    BenchmarkCase benchmarkCase = requireCase(cases, childCase);
    BenchmarkRunOptions options{
        .samplesOverride = childSamples,
        .jsonOut = childArtifactDir / "report.json",
        .envelopeOut = childArtifactDir / "run.json",
        .artifactDir = childArtifactDir,
        .dryRun = childDryRun,
        .tracyDiagnostic = childTracyDiagnostic,
        .gpuDiagnostic = {.kind =
                              childRenderDocDiagnostic
                                  ? BenchmarkGpuDiagnosticKind::RenderDocFrame
                                  : BenchmarkGpuDiagnosticKind::None,
                          .toolPath = childRenderDocToolPath,
                          .captureFrame = childRenderDocCaptureFrame},
        .verboseFrames = childVerboseFrames,
        .internalIsolatedChild = true,
        .baselineProfileId = profile.has_value() ? profile->id : std::string{},
        .baselineProfileAuthoritative =
            profile.has_value() && profile->authority.authoritative,
        .baselineProfileMinimumRepetitions =
            profile.has_value() ? profile->benchmarkPolicy.minimumRepetitions
                                : 0u,
        .baselineProfileWarmupStability =
            profile.has_value() ? profile->benchmarkPolicy.warmupStability
                                : "unknown",
        .baselineProfileWarmupWindowFrames =
            profile.has_value() ? profile->benchmarkPolicy.warmupWindowFrames
                                : 0u,
        .baselineProfileWarmupMaxDriftPercent =
            profile.has_value() ? profile->benchmarkPolicy.warmupMaxDriftPercent
                                : 0.0,
        .baselineProfileRequiredMetrics =
            profile.has_value() ? profile->benchmarkPolicy.requiredMetrics
                                : std::vector<std::string>{},
        .baselineProfile = profile,
        .baselineProfileWarning = profileWarning,
        .command = command,
    };
    BenchmarkRunResult result =
        runBenchmarkCase(std::move(benchmarkCase), options);
    std::cout << result.message
              << "\nreport: " << result.reportPath.generic_string() << "\n";
    std::exit(exitCode(result.exitCode));
  });

  auto *baseline = app.add_subcommand(
      "baseline", "Inspect, accept, or verify governed benchmark baselines");
  baseline->require_subcommand(1);

  std::string baselineInspectCase;
  std::string baselineInspectProfile = "local-nvrhi-visible";
  std::filesystem::path baselineInspectRoot;
  auto *baselineInspect =
      baseline->add_subcommand("inspect", "Inspect baseline evidence");
  baselineInspect->add_option("--case", baselineInspectCase, "Case id")
      ->required();
  baselineInspect->add_option("--profile", baselineInspectProfile,
                              "Baseline profile");
  baselineInspect->add_option("--baseline-root", baselineInspectRoot,
                              "Override baseline root");
  baselineInspect->callback([&]() {
    const std::vector<BenchmarkCase> cases = loadCasesOrExit();
    const BenchmarkCase &benchmarkCase =
        requireCase(cases, baselineInspectCase);
    auto loadedProfile = loadBaselineProfile(
        benchmarkRepoRoot() / "tools" / "profiles", baselineInspectProfile);
    if (loadedProfile.hasError()) {
      std::cerr << loadedProfile.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto verification =
        verifyBenchmarkBaseline(benchmarkCase.id, benchmarkCase.suite,
                                loadedProfile.value(), baselineInspectRoot);
    if (verification.hasError()) {
      std::cerr << verification.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto json = writeBenchmarkBaselineVerificationJson(
        verification.value(), "nuri.benchmark.baseline_inspection");
    if (json.hasError()) {
      std::cerr << json.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::RuntimeError));
    }
    std::cout << json.value() << "\n";
    if (!verification.value().exists) {
      std::exit(exitCode(BenchmarkExitCode::MissingBaseline));
    }
  });

  std::string baselineAcceptCase;
  std::string baselineAcceptProfile = "local-nvrhi-visible";
  std::string baselineAcceptReason;
  std::string baselineAcceptActor;
  std::string baselineAcceptConfirmPlan;
  std::filesystem::path baselineAcceptFrom;
  std::filesystem::path baselineAcceptRoot;
  bool baselineAcceptDryRun = false;
  auto *baselineAccept =
      baseline->add_subcommand("accept", "Accept reviewed benchmark evidence");
  baselineAccept->add_option("--from", baselineAcceptFrom, "Source run root")
      ->required();
  baselineAccept->add_option("--case", baselineAcceptCase, "Case id")
      ->required();
  baselineAccept->add_option("--profile", baselineAcceptProfile,
                             "Baseline profile");
  baselineAccept
      ->add_option("--reason", baselineAcceptReason, "Acceptance reason")
      ->required();
  baselineAccept->add_option("--actor", baselineAcceptActor, "Acceptance actor")
      ->required();
  baselineAccept->add_flag(
      "--dry-run", baselineAcceptDryRun,
      "Print the digest-bound plan without mutating baselines");
  baselineAccept->add_option("--confirm-plan", baselineAcceptConfirmPlan,
                             "Digest emitted by the reviewed dry-run plan");
  baselineAccept->add_option("--baseline-root", baselineAcceptRoot,
                             "Override baseline root");
  baselineAccept->callback([&]() {
    const std::vector<BenchmarkCase> cases = loadCasesOrExit();
    const BenchmarkCase &benchmarkCase = requireCase(cases, baselineAcceptCase);
    auto loadedProfile = loadBaselineProfile(
        benchmarkRepoRoot() / "tools" / "profiles", baselineAcceptProfile);
    if (loadedProfile.hasError()) {
      std::cerr << loadedProfile.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto source =
        loadBenchmarkBaselineSource(baselineAcceptFrom, baselineAcceptCase);
    if (source.hasError() ||
        source.value().report.benchmarkCase.suite != benchmarkCase.suite) {
      std::cerr << (source.hasError()
                        ? source.error()
                        : "source report suite does not match requested case")
                << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto plan = planBenchmarkBaseline(source.value(), loadedProfile.value(),
                                      baselineAcceptReason, baselineAcceptActor,
                                      baselineAcceptRoot);
    if (plan.hasError()) {
      std::cerr << plan.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto planJson = writeBenchmarkBaselinePlanJson(plan.value());
    if (planJson.hasError()) {
      std::cerr << planJson.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::RuntimeError));
    }
    if (baselineAcceptDryRun) {
      std::cout << planJson.value() << "\n";
      return;
    }
    if (baselineAcceptConfirmPlan.empty()) {
      std::cerr << "baseline acceptance requires --confirm-plan "
                << plan.value().digest << " after reviewing --dry-run\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto accepted = acceptBenchmarkBaseline(
        source.value(), loadedProfile.value(), baselineAcceptReason,
        baselineAcceptActor, baselineAcceptConfirmPlan,
        BenchmarkBaselineAcceptOptions{.baselineRoot = baselineAcceptRoot});
    if (accepted.hasError()) {
      std::cerr << accepted.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto verification =
        verifyBenchmarkBaseline(benchmarkCase.id, benchmarkCase.suite,
                                loadedProfile.value(), baselineAcceptRoot);
    if (verification.hasError() || !verification.value().valid) {
      std::cerr << (verification.hasError()
                        ? verification.error()
                        : "accepted baseline failed verification")
                << "\n";
      std::exit(exitCode(BenchmarkExitCode::RuntimeError));
    }
    auto json = writeBenchmarkBaselineVerificationJson(verification.value());
    if (json.hasError()) {
      std::cerr << json.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::RuntimeError));
    }
    std::cout << json.value() << "\n";
  });

  std::string baselineVerifyCase;
  std::string baselineVerifyProfile = "local-nvrhi-visible";
  std::filesystem::path baselineVerifyRoot;
  auto *baselineVerify =
      baseline->add_subcommand("verify", "Verify baseline digests and history");
  baselineVerify->add_option("--case", baselineVerifyCase, "Case id")
      ->required();
  baselineVerify->add_option("--profile", baselineVerifyProfile,
                             "Baseline profile");
  baselineVerify->add_option("--baseline-root", baselineVerifyRoot,
                             "Override baseline root");
  baselineVerify->callback([&]() {
    const std::vector<BenchmarkCase> cases = loadCasesOrExit();
    const BenchmarkCase &benchmarkCase = requireCase(cases, baselineVerifyCase);
    auto loadedProfile = loadBaselineProfile(
        benchmarkRepoRoot() / "tools" / "profiles", baselineVerifyProfile);
    if (loadedProfile.hasError()) {
      std::cerr << loadedProfile.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto verification =
        verifyBenchmarkBaseline(benchmarkCase.id, benchmarkCase.suite,
                                loadedProfile.value(), baselineVerifyRoot);
    if (verification.hasError()) {
      std::cerr << verification.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto json = writeBenchmarkBaselineVerificationJson(verification.value());
    if (json.hasError()) {
      std::cerr << json.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::RuntimeError));
    }
    std::cout << json.value() << "\n";
    if (!verification.value().exists) {
      std::exit(exitCode(BenchmarkExitCode::MissingBaseline));
    }
    if (!verification.value().valid) {
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
  });

  std::filesystem::path currentReport;
  std::filesystem::path baselineReport;
  std::string compareProfile;
  std::filesystem::path compareBaselineRoot;
  std::filesystem::path compareOut;
  std::filesystem::path compareHtmlOut;
  std::vector<std::string> compareHtmlMetrics;
  std::vector<std::string> compareHtmlStats;
  bool forceCompare = false;
  auto *compare = app.add_subcommand("compare", "Compare reports");
  compare->add_option("--current", currentReport, "Current report")->required();
  compare
      ->add_option("--profile", compareProfile,
                   "Named baseline profile whose governed evidence is used")
      ->required();
  compare->add_option(
      "--baseline", baselineReport,
      "Arbitrary baseline report (investigative; requires --force)");
  compare->add_option("--baseline-root", compareBaselineRoot,
                      "Override governed baseline root");
  compare->add_option("--json-out", compareOut, "Comparison JSON path");
  compare->add_option("--html-out", compareHtmlOut, "Graph HTML output path");
  compare
      ->add_option("--html-metric", compareHtmlMetrics,
                   "Metric to include in graph")
      ->expected(1, -1);
  compare
      ->add_option("--html-stat", compareHtmlStats,
                   "Statistic to include in graph")
      ->expected(1, -1);
  compare->add_flag(
      "--force", forceCompare,
      "Demote invalid preconditions and allow arbitrary baseline paths");
  compare->callback([&]() {
    auto current = readBenchmarkReportFile(currentReport);
    if (current.hasError()) {
      std::cerr << current.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::MissingBaseline));
    }
    auto profile = loadBaselineProfile(
        benchmarkRepoRoot() / "tools" / "profiles", compareProfile);
    if (profile.hasError()) {
      std::cerr << profile.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    BenchmarkReport currentValue = std::move(current.value());
    BenchmarkReport baselineValue{};
    if (!baselineReport.empty()) {
      if (!forceCompare) {
        std::cerr << "--baseline accepts an arbitrary path only with --force; "
                     "omit it to load the governed --profile baseline\n";
        std::exit(exitCode(BenchmarkExitCode::InvalidInput));
      }
      auto arbitrary = readBenchmarkReportFile(baselineReport);
      if (arbitrary.hasError()) {
        std::cerr << arbitrary.error() << "\n";
        std::exit(exitCode(BenchmarkExitCode::MissingBaseline));
      }
      std::cerr << "warning: arbitrary --baseline evidence is investigative\n";
      baselineValue = std::move(arbitrary.value());
    } else {
      auto governed = loadVerifiedBenchmarkBaseline(
          currentValue.benchmarkCase.id, currentValue.benchmarkCase.suite,
          profile.value(), compareBaselineRoot);
      if (governed.hasError()) {
        std::cerr << governed.error() << "\n";
        std::exit(
            exitCode(governed.error().find("is missing") != std::string::npos
                         ? BenchmarkExitCode::MissingBaseline
                         : BenchmarkExitCode::InvalidInput));
      }
      baselineValue = std::move(governed.value().report);
    }
    BenchmarkComparisonReport comparison =
        compareBenchmarkReports(currentValue, baselineValue,
                                BenchmarkCompareOptions{.force = forceCompare});
    if (!compareOut.empty()) {
      auto writeResult = writeBenchmarkComparisonFile(comparison, compareOut);
      if (writeResult.hasError()) {
        std::cerr << writeResult.error() << "\n";
        std::exit(exitCode(BenchmarkExitCode::RuntimeError));
      }
    }
    if (!compareHtmlOut.empty()) {
      auto written = writeBenchmarkComparisonHtmlFile(
          baselineValue, currentValue, comparison,
          makeGraphOptions(compareHtmlMetrics, compareHtmlStats,
                           "Nuri Benchmark Comparison"),
          compareHtmlOut);
      if (written.hasError()) {
        std::cerr << written.error() << "\n";
        std::exit(exitCode(BenchmarkExitCode::RuntimeError));
      }
      std::cout << "html: " << compareHtmlOut.generic_string() << "\n";
    }
    if (!comparison.valid) {
      std::cout << "comparison invalid\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    if (comparison.regression) {
      std::cout << "benchmark regression\n";
      std::exit(exitCode(BenchmarkExitCode::Regression));
    }
    std::cout << "no material regression\n";
  });

  std::filesystem::path reportsDir;
  std::filesystem::path summaryOut;
  std::filesystem::path summaryHtmlOut;
  std::vector<std::string> summaryHtmlMetrics;
  std::vector<std::string> summaryHtmlStats;
  auto *summarize = app.add_subcommand("summarize", "Summarize report dir");
  summarize->add_option("--reports", reportsDir, "Report directory")
      ->required();
  summarize->add_option("--json-out", summaryOut, "Summary JSON path");
  summarize->add_option("--html-out", summaryHtmlOut, "Graph HTML output path");
  summarize
      ->add_option("--html-metric", summaryHtmlMetrics,
                   "Metric to include in graph")
      ->expected(1, -1);
  summarize
      ->add_option("--html-stat", summaryHtmlStats,
                   "Statistic to include in graph")
      ->expected(1, -1);
  summarize->callback([&]() {
    if (!std::filesystem::exists(reportsDir)) {
      std::cerr << "reports directory does not exist\n";
      std::exit(exitCode(BenchmarkExitCode::MissingBaseline));
    }
    uint32_t reportCount = 0u;
    std::string json = "{\n  \"kind\": \"nuri.benchmark.summary\",\n"
                       "  \"reports\": [\n";
    bool first = true;
    std::vector<BenchmarkReport> reports;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(reportsDir)) {
      if (!entry.is_regular_file() || entry.path().extension() != ".json") {
        continue;
      }
      auto report = readBenchmarkReportFile(entry.path());
      if (report.hasError() || report.value().kind != "nuri.benchmark.report") {
        continue;
      }
      BenchmarkReport reportValue = std::move(report.value());
      if (!first) {
        json += ",\n";
      }
      first = false;
      ++reportCount;
      json += "    {\"path\": \"" + entry.path().generic_string() +
              "\", \"case\": \"" + reportValue.benchmarkCase.id + "\"}";
      reports.push_back(std::move(reportValue));
    }
    json +=
        "\n  ],\n  \"reportCount\": " + std::to_string(reportCount) + "\n}\n";
    if (!summaryOut.empty()) {
      writeTextFile(summaryOut, json);
    } else {
      std::cout << json;
    }
    writeGraphOrExit(reports, summaryHtmlOut,
                     makeGraphOptions(summaryHtmlMetrics, summaryHtmlStats,
                                      "Nuri Benchmark Summary"));
  });

  std::vector<std::filesystem::path> graphReports;
  std::vector<std::filesystem::path> graphReportDirs;
  std::filesystem::path graphHtmlOut;
  std::vector<std::string> graphMetrics;
  std::vector<std::string> graphStats;
  std::string graphTitle = "Nuri Benchmark Results";
  auto *graph = app.add_subcommand("graph", "Write benchmark graph HTML");
  graph->add_option("--report", graphReports, "Benchmark report JSON file")
      ->expected(1, -1);
  graph->add_option("--reports", graphReportDirs, "Benchmark report directory")
      ->expected(1, -1);
  graph->add_option("--metric", graphMetrics, "Metric to include")
      ->expected(1, -1);
  graph->add_option("--stat", graphStats, "Statistic to include")
      ->expected(1, -1);
  graph->add_option("--title", graphTitle, "HTML title");
  graph->add_option("--html-out", graphHtmlOut, "Graph HTML output path")
      ->required();
  graph->callback([&]() {
    if (graphReports.empty() && graphReportDirs.empty()) {
      std::cerr << "graph requires --report or --reports\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    auto reports = loadBenchmarkReports(graphReports, graphReportDirs);
    if (reports.hasError()) {
      std::cerr << reports.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::MissingBaseline));
    }
    if (reports.value().empty()) {
      std::cerr << "no benchmark reports found\n";
      std::exit(exitCode(BenchmarkExitCode::MissingBaseline));
    }
    writeGraphOrExit(reports.value(), graphHtmlOut,
                     makeGraphOptions(graphMetrics, graphStats, graphTitle));
  });

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    const int cliExit = app.exit(e);
    return cliExit == 0 ? 0 : exitCode(BenchmarkExitCode::InvalidInput);
  }
  return 0;
}
