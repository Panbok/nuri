#include "nuri/tools/snapshot/snapshot_baseline.h"
#include "nuri/tools/snapshot/snapshot_compare.h"
#include "nuri/tools/snapshot/snapshot_environment.h"
#include "nuri/tools/snapshot/snapshot_manifest.h"
#include "nuri/tools/snapshot/snapshot_runner.h"

#include <CLI/CLI.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace nuri::tools::snapshot;

namespace {

[[nodiscard]] int exitCode(SnapshotExitCode code) {
  return static_cast<int>(code);
}

[[nodiscard]] std::vector<SnapshotCase> loadCasesOrExit() {
  auto cases = discoverSnapshotCases();
  if (cases.hasError()) {
    std::cerr << cases.error() << "\n";
    std::exit(exitCode(SnapshotExitCode::InvalidInput));
  }
  return std::move(cases.value());
}

[[nodiscard]] const SnapshotCase &
requireCase(const std::vector<SnapshotCase> &cases, std::string_view id) {
  const SnapshotCase *snapshotCase = findSnapshotCaseById(cases, id);
  if (snapshotCase == nullptr) {
    std::cerr << "unknown snapshot case: " << id << "\n";
    std::exit(exitCode(SnapshotExitCode::InvalidInput));
  }
  return *snapshotCase;
}

[[nodiscard]] SnapshotRunOptions makeOptions(
    const std::filesystem::path &artifactDir,
    const std::filesystem::path &jsonOut,
    const std::filesystem::path &htmlOut, const std::string &baselineProfile,
    const std::string &windowMode, bool dryRun, bool printEffectiveConfig,
    bool force, const std::string &command) {
  return SnapshotRunOptions{
      .jsonOut = jsonOut,
      .htmlOut = htmlOut,
      .artifactDir = artifactDir,
      .baselineProfile = baselineProfile,
      .windowMode = windowMode,
      .dryRun = dryRun,
      .printEffectiveConfig = printEffectiveConfig,
      .force = force,
      .command = command,
  };
}

} // namespace

int main(int argc, char **argv) {
  CLI::App app{"Nuri renderer visual snapshot tool"};
  app.require_subcommand(1);
  const std::string command = joinCommandLine(argc, argv);

  std::string listSuite;
  bool listJson = false;
  auto *list = app.add_subcommand("list", "List snapshot cases");
  list->add_option("--suite", listSuite, "Filter by suite");
  list->add_flag("--json", listJson, "Emit JSON");
  list->callback([&]() {
    const std::vector<SnapshotCase> cases = loadCasesOrExit();
    if (listJson) {
      auto json = formatSnapshotCaseListJson(cases, listSuite);
      if (json.hasError()) {
        std::cerr << json.error() << "\n";
        std::exit(exitCode(SnapshotExitCode::RuntimeError));
      }
      std::cout << json.value();
    } else {
      std::cout << formatSnapshotCaseListText(cases, listSuite);
    }
  });

  std::string explainCase;
  bool explainJson = false;
  auto *explain = app.add_subcommand("explain", "Explain a snapshot case");
  explain->add_option("--case", explainCase, "Case id")->required();
  explain->add_flag("--json", explainJson, "Emit JSON");
  explain->callback([&]() {
    const std::vector<SnapshotCase> cases = loadCasesOrExit();
    const SnapshotCase &snapshotCase = requireCase(cases, explainCase);
    if (explainJson) {
      auto json = formatSnapshotCaseExplanationJson(snapshotCase);
      if (json.hasError()) {
        std::cerr << json.error() << "\n";
        std::exit(exitCode(SnapshotExitCode::RuntimeError));
      }
      std::cout << json.value();
    } else {
      std::cout << formatSnapshotCaseExplanationText(snapshotCase);
    }
  });

  std::string caseId;
  std::filesystem::path artifactDir;
  std::filesystem::path jsonOut;
  std::filesystem::path htmlOut;
  std::string baselineProfile = "local-lvk-visible";
  std::string windowMode = "visible";
  bool dryRun = false;
  bool printEffectiveConfig = false;
  auto *capture = app.add_subcommand("capture", "Capture a snapshot case");
  capture->add_option("--case", caseId, "Case id")->required();
  capture->add_option("--artifact-dir", artifactDir, "Artifact directory");
  capture->add_option("--json-out", jsonOut, "Report JSON path");
  capture->add_option("--html-out", htmlOut, "Report HTML path");
  capture->add_option("--window-mode", windowMode,
                      "visible, hidden, or headless");
  capture->add_flag("--dry-run", dryRun, "Resolve config without renderer init");
  capture->add_flag("--print-effective-config", printEffectiveConfig,
                    "Print resolved config before running");
  capture->callback([&]() {
    std::vector<SnapshotCase> cases = loadCasesOrExit();
    SnapshotCase snapshotCase = requireCase(cases, caseId);
    SnapshotRunOptions options =
        makeOptions(artifactDir, jsonOut, htmlOut, baselineProfile, windowMode,
                    dryRun, printEffectiveConfig, false, command);
    if (printEffectiveConfig) {
      auto effective = formatSnapshotEffectiveConfigJson(snapshotCase, options);
      if (!effective.hasError()) {
        std::cout << effective.value();
      }
    }
    SnapshotRunResult result = captureSnapshotCase(std::move(snapshotCase),
                                                   options);
    std::cout << result.message << "\nreport: "
              << result.reportPath.generic_string() << "\nhtml: "
              << result.htmlPath.generic_string() << "\n";
    std::exit(exitCode(result.exitCode));
  });

  std::string compareCase;
  std::filesystem::path compareArtifactDir;
  std::filesystem::path compareJsonOut;
  std::filesystem::path compareHtmlOut;
  std::string compareProfile = "local-lvk-visible";
  bool forceCompare = false;
  auto *compare = app.add_subcommand("compare", "Compare captured artifacts");
  compare->add_option("--case", compareCase, "Case id")->required();
  compare->add_option("--artifact-dir", compareArtifactDir,
                      "Artifact directory")->required();
  compare->add_option("--baseline-profile", compareProfile,
                      "Baseline profile");
  compare->add_option("--json-out", compareJsonOut, "Report JSON path");
  compare->add_option("--html-out", compareHtmlOut, "Report HTML path");
  compare->add_flag("--force", forceCompare, "Demote invalid preconditions");
  compare->callback([&]() {
    std::vector<SnapshotCase> cases = loadCasesOrExit();
    SnapshotCase snapshotCase = requireCase(cases, compareCase);
    SnapshotRunOptions options =
        makeOptions(compareArtifactDir, compareJsonOut, compareHtmlOut,
                    compareProfile, "visible", false, false, forceCompare,
                    command);
    SnapshotRunResult result =
        compareSnapshotCase(std::move(snapshotCase), options);
    std::cout << result.message << "\nreport: "
              << result.reportPath.generic_string() << "\n";
    std::exit(exitCode(result.exitCode));
  });

  std::string runCase;
  std::string runSuite;
  std::filesystem::path runArtifactDir;
  std::filesystem::path runJsonOut;
  std::filesystem::path runHtmlOut;
  std::string runBaselineProfile = "local-lvk-visible";
  std::string runWindowMode = "visible";
  bool runDry = false;
  bool runEffective = false;
  auto *run = app.add_subcommand("run", "Capture and compare case or suite");
  run->add_option("--case", runCase, "Case id");
  run->add_option("--suite", runSuite, "Suite name");
  run->add_option("--artifact-dir", runArtifactDir, "Artifact directory");
  run->add_option("--baseline-profile", runBaselineProfile,
                  "Baseline profile");
  run->add_option("--json-out", runJsonOut, "Report JSON path");
  run->add_option("--html-out", runHtmlOut, "Report HTML path");
  run->add_option("--window-mode", runWindowMode,
                  "visible, hidden, or headless");
  run->add_flag("--dry-run", runDry, "Resolve config without renderer init");
  run->add_flag("--print-effective-config", runEffective,
                "Print resolved config before running");
  run->callback([&]() {
    if (runCase.empty() == runSuite.empty()) {
      std::cerr << "run requires exactly one of --case or --suite\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    std::vector<SnapshotCase> cases = loadCasesOrExit();
    SnapshotRunOptions options =
        makeOptions(runArtifactDir, runJsonOut, runHtmlOut,
                    runBaselineProfile, runWindowMode, runDry, runEffective,
                    false, command);
    if (!runCase.empty()) {
      SnapshotCase snapshotCase = requireCase(cases, runCase);
      if (runEffective) {
        auto effective = formatSnapshotEffectiveConfigJson(snapshotCase, options);
        if (!effective.hasError()) {
          std::cout << effective.value();
        }
      }
      SnapshotRunResult result =
          runSnapshotCase(std::move(snapshotCase), options);
      std::cout << result.message << "\nreport: "
                << result.reportPath.generic_string() << "\nhtml: "
                << result.htmlPath.generic_string() << "\n";
      std::exit(exitCode(result.exitCode));
    }
    SnapshotSuiteRunResult suiteResult =
        runSnapshotSuite(std::move(cases), runSuite, options);
    std::cout << suiteResult.message << "\nreport: "
              << suiteResult.reportPath.generic_string() << "\nhtml: "
              << suiteResult.htmlPath.generic_string() << "\n";
    std::exit(exitCode(suiteResult.exitCode));
  });

  std::string approveCase;
  std::string approveReason;
  std::filesystem::path approveArtifacts;
  std::string approveProfile = "local-lvk-visible";
  auto *approve = app.add_subcommand("approve", "Approve captured baselines");
  approve->add_option("--case", approveCase, "Case id")->required();
  approve->add_option("--reason", approveReason, "Approval reason")->required();
  approve->add_option("--from-artifacts", approveArtifacts,
                      "Artifact directory")->required();
  approve->add_option("--baseline-profile", approveProfile,
                      "Baseline profile");
  approve->callback([&]() {
    std::vector<SnapshotCase> cases = loadCasesOrExit();
    SnapshotCase snapshotCase = requireCase(cases, approveCase);
    const std::filesystem::path reportPath =
        approveArtifacts / "cases" / snapshotCase.id / "report.json";
    auto report = readSnapshotReportFile(reportPath);
    if (report.hasError()) {
      std::cerr << report.error() << "\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    report.value().snapshotCase = snapshotCase;
    auto approved =
        approveSnapshotBaselines(report.value(), approveProfile, approveReason);
    if (approved.hasError()) {
      std::cerr << approved.error() << "\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    std::cout << "baselines approved\n";
  });

  std::filesystem::path diffActual;
  std::filesystem::path diffExpected;
  std::filesystem::path diffOut;
  std::filesystem::path diffJsonOut;
  std::string diffProfile = "exact";
  auto *diff = app.add_subcommand("diff", "Diff two image files");
  diff->add_option("--actual", diffActual, "Actual image")->required();
  diff->add_option("--expected", diffExpected, "Expected image")->required();
  diff->add_option("--profile", diffProfile, "Compare profile")->required();
  diff->add_option("--diff-out", diffOut, "Diff PNG output");
  diff->add_option("--json-out", diffJsonOut, "Comparison JSON output");
  diff->callback([&]() {
    auto actual = readSnapshotImageFile(diffActual);
    auto expected = readSnapshotImageFile(diffExpected);
    if (actual.hasError() || expected.hasError()) {
      std::cerr << (actual.hasError() ? actual.error() : expected.error())
                << "\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    SnapshotCompareResult comparison = compareSnapshotImages(
        actual.value(), expected.value(),
        builtinSnapshotCompareProfile(diffProfile));
    if (!diffOut.empty()) {
      auto written =
          writeSnapshotDiffPng(actual.value(), expected.value(), diffOut);
      if (written.hasError()) {
        std::cerr << written.error() << "\n";
        std::exit(exitCode(SnapshotExitCode::RuntimeError));
      }
    }
    (void)diffJsonOut;
    if (!comparison.compatible) {
      std::cout << "comparison invalid\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    if (!comparison.passed) {
      std::cout << "visual mismatch\n";
      std::exit(exitCode(SnapshotExitCode::VisualMismatch));
    }
    std::cout << "snapshots matched\n";
  });

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }
  return 0;
}
