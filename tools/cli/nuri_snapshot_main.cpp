#include "nuri/tools/snapshot/snapshot_baseline.h"
#include "nuri/tools/snapshot/snapshot_compare.h"
#include "nuri/tools/snapshot/snapshot_environment.h"
#include "nuri/tools/snapshot/snapshot_manifest.h"
#include "nuri/tools/snapshot/snapshot_runner.h"

#include <CLI/CLI.hpp>

#include <algorithm>
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

[[nodiscard]] SnapshotRunOptions
makeOptions(const std::filesystem::path &artifactDir,
            const std::filesystem::path &jsonOut,
            const std::filesystem::path &htmlOut,
            const std::string &baselineProfile, const std::string &windowMode,
            bool dryRun, bool printEffectiveConfig, bool force,
            const std::string &command) {
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
  std::string baselineProfile = "local-nvrhi-visible";
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
  capture->add_flag("--dry-run", dryRun,
                    "Resolve config without renderer init");
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
    SnapshotRunResult result =
        captureSnapshotCase(std::move(snapshotCase), options);
    std::cout << result.message
              << "\nreport: " << result.reportPath.generic_string()
              << "\nhtml: " << result.htmlPath.generic_string() << "\n";
    std::exit(exitCode(result.exitCode));
  });

  std::string compareCase;
  std::filesystem::path compareArtifactDir;
  std::filesystem::path compareJsonOut;
  std::filesystem::path compareHtmlOut;
  std::string compareProfile = "local-nvrhi-visible";
  bool forceCompare = false;
  auto *compare = app.add_subcommand("compare", "Compare captured artifacts");
  compare->add_option("--case", compareCase, "Case id")->required();
  compare
      ->add_option("--artifact-dir", compareArtifactDir, "Artifact directory")
      ->required();
  compare->add_option("--baseline-profile", compareProfile, "Baseline profile");
  compare->add_option("--json-out", compareJsonOut, "Report JSON path");
  compare->add_option("--html-out", compareHtmlOut, "Report HTML path");
  compare->add_flag("--force", forceCompare, "Demote invalid preconditions");
  compare->callback([&]() {
    std::vector<SnapshotCase> cases = loadCasesOrExit();
    SnapshotCase snapshotCase = requireCase(cases, compareCase);
    SnapshotRunOptions options = makeOptions(
        compareArtifactDir, compareJsonOut, compareHtmlOut, compareProfile,
        "visible", false, false, forceCompare, command);
    SnapshotRunResult result =
        compareSnapshotCase(std::move(snapshotCase), options);
    std::cout << result.message
              << "\nreport: " << result.reportPath.generic_string() << "\n";
    std::exit(exitCode(result.exitCode));
  });

  std::string runCase;
  std::string runSuite;
  std::filesystem::path runArtifactDir;
  std::filesystem::path runJsonOut;
  std::filesystem::path runHtmlOut;
  std::string runBaselineProfile = "local-nvrhi-visible";
  std::string runWindowMode = "visible";
  bool runDry = false;
  bool runEffective = false;
  bool runForce = false;
  auto *run = app.add_subcommand("run", "Capture and compare case or suite");
  run->add_option("--case", runCase, "Case id");
  run->add_option("--suite", runSuite, "Suite name");
  run->add_option("--artifact-dir", runArtifactDir, "Artifact directory");
  run->add_option("--baseline-profile", runBaselineProfile, "Baseline profile");
  run->add_option("--json-out", runJsonOut, "Report JSON path");
  run->add_option("--html-out", runHtmlOut, "Report HTML path");
  run->add_option("--window-mode", runWindowMode,
                  "visible, hidden, or headless");
  run->add_flag("--dry-run", runDry, "Resolve config without renderer init");
  run->add_flag("--print-effective-config", runEffective,
                "Print resolved config before running");
  run->add_flag("--force", runForce,
                "Run incompatible comparisons as investigative");
  run->callback([&]() {
    if (runCase.empty() == runSuite.empty()) {
      std::cerr << "run requires exactly one of --case or --suite\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    std::vector<SnapshotCase> cases = loadCasesOrExit();
    SnapshotRunOptions options =
        makeOptions(runArtifactDir, runJsonOut, runHtmlOut, runBaselineProfile,
                    runWindowMode, runDry, runEffective, runForce, command);
    if (!runCase.empty()) {
      SnapshotCase snapshotCase = requireCase(cases, runCase);
      if (runEffective) {
        auto effective =
            formatSnapshotEffectiveConfigJson(snapshotCase, options);
        if (!effective.hasError()) {
          std::cout << effective.value();
        }
      }
      SnapshotRunResult result =
          runSnapshotCase(std::move(snapshotCase), options);
      std::cout << result.message
                << "\nreport: " << result.reportPath.generic_string()
                << "\nhtml: " << result.htmlPath.generic_string() << "\n";
      std::exit(exitCode(result.exitCode));
    }
    SnapshotSuiteRunResult suiteResult =
        runSnapshotSuite(std::move(cases), runSuite, options);
    std::cout << suiteResult.message
              << "\nreport: " << suiteResult.reportPath.generic_string()
              << "\nhtml: " << suiteResult.htmlPath.generic_string() << "\n";
    std::exit(exitCode(suiteResult.exitCode));
  });

  auto *baseline =
      app.add_subcommand("baseline", "Inspect and verify snapshot baselines");
  baseline->require_subcommand(1);
  std::string baselineInspectCase;
  std::string baselineInspectProfile = "local-nvrhi-visible";
  auto *baselineInspect =
      baseline->add_subcommand("inspect", "Inspect baseline files and hashes");
  baselineInspect->add_option("--case", baselineInspectCase, "Case id")
      ->required();
  baselineInspect->add_option("--profile", baselineInspectProfile,
                              "Baseline profile");
  baselineInspect->callback([&]() {
    std::vector<SnapshotCase> cases = loadCasesOrExit();
    const SnapshotCase &snapshotCase = requireCase(cases, baselineInspectCase);
    auto inspection =
        inspectSnapshotBaseline(snapshotCase, baselineInspectProfile);
    if (inspection.hasError()) {
      std::cerr << inspection.error() << "\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    std::cout << inspection.value() << "\n";
  });

  std::string baselineVerifyCase;
  std::string baselineVerifyProfile = "local-nvrhi-visible";
  auto *baselineVerify = baseline->add_subcommand(
      "verify", "Verify a governed baseline against reviewed SHA-256 hashes");
  baselineVerify->add_option("--case", baselineVerifyCase, "Case id")
      ->required();
  baselineVerify->add_option("--profile", baselineVerifyProfile,
                             "Baseline profile");
  baselineVerify->callback([&]() {
    std::vector<SnapshotCase> cases = loadCasesOrExit();
    const SnapshotCase &snapshotCase = requireCase(cases, baselineVerifyCase);
    auto verified = verifySnapshotBaseline(snapshotCase, baselineVerifyProfile);
    if (verified.hasError()) {
      std::cerr << verified.error() << "\n";
      const bool missing =
          verified.error().find("missing") != std::string::npos;
      std::exit(exitCode(missing ? SnapshotExitCode::MissingBaseline
                                 : SnapshotExitCode::InvalidInput));
    }
    auto inspection =
        inspectSnapshotBaseline(snapshotCase, baselineVerifyProfile);
    if (inspection.hasError()) {
      std::cerr << inspection.error() << "\n";
      std::exit(exitCode(SnapshotExitCode::RuntimeError));
    }
    std::cout << inspection.value() << "\n";
  });

  std::string approveCase;
  std::string approveReason;
  std::string approveActor;
  std::string approveConfirmPlan;
  bool approveDryRun = false;
  std::filesystem::path approveArtifacts;
  std::string approveProfile = "local-nvrhi-visible";
  auto *approve = app.add_subcommand("approve", "Approve captured baselines");
  approve->add_option("--case", approveCase, "Case id")->required();
  approve->add_option("--reason", approveReason, "Approval reason")->required();
  approve->add_option("--actor", approveActor, "Approval actor")->required();
  approve
      ->add_option("--from,--from-artifacts", approveArtifacts,
                   "Artifact directory")
      ->required();
  approve->add_option("--profile,--baseline-profile", approveProfile,
                      "Baseline profile");
  approve->add_flag("--dry-run", approveDryRun,
                    "Print the reviewed promotion plan without mutation");
  approve->add_option("--confirm-plan", approveConfirmPlan,
                      "Digest emitted by the reviewed dry-run plan");
  const auto approveCallback = [&]() {
    std::vector<SnapshotCase> cases = loadCasesOrExit();
    SnapshotCase snapshotCase = requireCase(cases, approveCase);
    auto caseDir = resolveSnapshotPathUnder(
        approveArtifacts, std::filesystem::path("cases") / snapshotCase.id);
    if (caseDir.hasError()) {
      std::cerr << caseDir.error() << "\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    const std::filesystem::path reportPath = caseDir.value() / "report.json";
    auto report = readSnapshotReportFile(reportPath);
    if (report.hasError()) {
      std::cerr << report.error() << "\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    if (report.value().snapshotCase.id != snapshotCase.id ||
        report.value().snapshotCase.suite != snapshotCase.suite) {
      std::cerr
          << "approval source report identity does not match requested case\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    bool capturesMatch = report.value().snapshotCase.captures.size() ==
                         snapshotCase.captures.size();
    for (const SnapshotCaptureTarget &expected : snapshotCase.captures) {
      const auto found =
          std::find_if(report.value().snapshotCase.captures.begin(),
                       report.value().snapshotCase.captures.end(),
                       [&](const SnapshotCaptureTarget &actual) {
                         return actual.name == expected.name &&
                                actual.profile == expected.profile &&
                                actual.required == expected.required;
                       });
      capturesMatch =
          capturesMatch && found != report.value().snapshotCase.captures.end();
    }
    if (!capturesMatch) {
      std::cerr << "approval source capture set does not match case manifest\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    report.value().artifacts.caseDir = caseDir.value();
    auto plan = planSnapshotBaselines(report.value(), approveProfile,
                                      approveReason, approveActor);
    if (plan.hasError()) {
      std::cerr << plan.error() << "\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    auto planJson = writeSnapshotBaselinePlanJson(plan.value());
    if (planJson.hasError()) {
      std::cerr << planJson.error() << "\n";
      std::exit(exitCode(SnapshotExitCode::RuntimeError));
    }
    if (approveDryRun) {
      std::cout << planJson.value() << "\n";
      return;
    }
    if (approveConfirmPlan.empty()) {
      std::cerr << "approval requires --confirm-plan " << plan.value().digest
                << " after reviewing --dry-run\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    auto approved =
        approveSnapshotBaselines(report.value(), approveProfile, approveReason,
                                 approveConfirmPlan, approveActor);
    if (approved.hasError()) {
      std::cerr << approved.error() << "\n";
      std::exit(exitCode(SnapshotExitCode::InvalidInput));
    }
    std::cout << "baselines approved\n";
  };
  approve->callback(approveCallback);

  auto *baselineAccept = baseline->add_subcommand(
      "accept", "Review and atomically accept a snapshot baseline");
  baselineAccept->add_option("--case", approveCase, "Case id")->required();
  baselineAccept->add_option("--reason", approveReason, "Approval reason")
      ->required();
  baselineAccept->add_option("--actor", approveActor, "Approval actor")
      ->required();
  baselineAccept
      ->add_option("--from,--from-artifacts", approveArtifacts,
                   "Artifact directory")
      ->required();
  baselineAccept->add_option("--profile,--baseline-profile", approveProfile,
                             "Baseline profile");
  baselineAccept->add_flag(
      "--dry-run", approveDryRun,
      "Print the reviewed promotion plan without mutation");
  baselineAccept->add_option("--confirm-plan", approveConfirmPlan,
                             "Digest emitted by the reviewed dry-run plan");
  baselineAccept->callback(approveCallback);

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
    SnapshotCompareResult comparison =
        compareSnapshotImages(actual.value(), expected.value(),
                              builtinSnapshotCompareProfile(diffProfile));
    if (!diffOut.empty() && comparison.compatible) {
      auto written =
          writeSnapshotDiffPng(actual.value(), expected.value(), diffOut);
      if (written.hasError()) {
        std::cerr << written.error() << "\n";
        std::exit(exitCode(SnapshotExitCode::RuntimeError));
      }
    }
    if (!diffJsonOut.empty()) {
      auto written =
          writeSnapshotComparisonFile(comparison, diffProfile, diffJsonOut);
      if (written.hasError()) {
        std::cerr << written.error() << "\n";
        std::exit(exitCode(SnapshotExitCode::RuntimeError));
      }
    }
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
    const int parserExit = app.exit(e);
    return parserExit == 0 ? 0 : exitCode(SnapshotExitCode::InvalidInput);
  }
  return 0;
}
