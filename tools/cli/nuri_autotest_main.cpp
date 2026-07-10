#include "nuri/tools/autotest/autotest_manifest.h"
#include "nuri/tools/autotest/autotest_record.h"
#include "nuri/tools/autotest/autotest_runner.h"
#include "nuri/tools/core/safe_path.h"

#include <CLI/CLI.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace nuri::tools::autotest;

namespace {

[[nodiscard]] int exitCode(AutotestExitCode code) {
  return static_cast<int>(code);
}

[[nodiscard]] std::vector<AutotestCase> loadCasesOrExit() {
  auto cases = discoverAutotestCases();
  if (cases.hasError()) {
    std::cerr << cases.error() << "\n";
    std::exit(exitCode(AutotestExitCode::InvalidInput));
  }
  return std::move(cases.value());
}

[[nodiscard]] const AutotestCase &
requireCase(const std::vector<AutotestCase> &cases, std::string_view id) {
  const AutotestCase *testCase = findAutotestCaseById(cases, id);
  if (testCase == nullptr) {
    std::cerr << "unknown autotest case: " << id << "\n";
    std::exit(exitCode(AutotestExitCode::InvalidInput));
  }
  return *testCase;
}

[[nodiscard]] AutotestRunOptions
makeOptions(const std::filesystem::path &artifactDir,
            const std::filesystem::path &jsonOut,
            const std::filesystem::path &htmlOut,
            const std::string &baselineProfile, const std::string &windowMode,
            bool dryRun, bool printEffectiveConfig, bool verboseFrames,
            const std::string &command) {
  return AutotestRunOptions{
      .jsonOut = jsonOut,
      .htmlOut = htmlOut,
      .artifactDir = artifactDir,
      .baselineProfile = baselineProfile,
      .windowMode = windowMode,
      .dryRun = dryRun,
      .printEffectiveConfig = printEffectiveConfig,
      .verboseFrames = verboseFrames,
      .command = command,
  };
}

} // namespace

int main(int argc, char **argv) {
  CLI::App app{"Nuri renderer autotest tool"};
  app.require_subcommand(1);
  const std::string command = joinCommandLine(argc, argv);

  std::string listSuite;
  bool listJson = false;
  auto *list = app.add_subcommand("list", "List autotest cases");
  list->add_option("--suite", listSuite, "Filter by suite");
  list->add_flag("--json", listJson, "Emit JSON");
  list->callback([&]() {
    const std::vector<AutotestCase> cases = loadCasesOrExit();
    if (listJson) {
      auto json = formatAutotestCaseListJson(cases, listSuite);
      if (json.hasError()) {
        std::cerr << json.error() << "\n";
        std::exit(exitCode(AutotestExitCode::RuntimeError));
      }
      std::cout << json.value();
    } else {
      std::cout << formatAutotestCaseListText(cases, listSuite);
    }
  });

  std::string explainCase;
  bool explainJson = false;
  auto *explain = app.add_subcommand("explain", "Explain an autotest case");
  explain->add_option("--case", explainCase, "Case id")->required();
  explain->add_flag("--json", explainJson, "Emit JSON");
  explain->callback([&]() {
    const std::vector<AutotestCase> cases = loadCasesOrExit();
    const AutotestCase &testCase = requireCase(cases, explainCase);
    if (explainJson) {
      auto json = formatAutotestCaseExplanationJson(testCase);
      if (json.hasError()) {
        std::cerr << json.error() << "\n";
        std::exit(exitCode(AutotestExitCode::RuntimeError));
      }
      std::cout << json.value();
    } else {
      std::cout << formatAutotestCaseExplanationText(testCase);
    }
  });

  std::string runCase;
  std::string runSuite;
  std::filesystem::path runArtifactDir;
  std::filesystem::path runJsonOut;
  std::filesystem::path runHtmlOut;
  std::string runBaselineProfile = "local-nvrhi-visible";
  std::string runWindowMode;
  bool runDry = false;
  bool runEffective = false;
  bool runVerboseFrames = false;
  auto *run = app.add_subcommand("run", "Run autotest case or suite");
  run->add_option("--case", runCase, "Case id");
  run->add_option("--suite", runSuite, "Suite name");
  run->add_option("--artifact-dir", runArtifactDir, "Artifact directory");
  run->add_option("--baseline-profile", runBaselineProfile, "Baseline profile");
  run->add_option("--json-out", runJsonOut, "Report JSON path");
  run->add_option("--html-out", runHtmlOut, "Report HTML path");
  run->add_option("--window-mode", runWindowMode,
                  "Override manifest mode: visible, hidden, or headless");
  run->add_flag("--dry-run", runDry, "Resolve config without renderer init");
  run->add_flag("--print-effective-config", runEffective,
                "Print resolved config before running");
  run->add_flag("--verbose-frames", runVerboseFrames,
                "Include verbose frame data when available");
  run->callback([&]() {
    if (runCase.empty() == runSuite.empty()) {
      std::cerr << "run requires exactly one of --case or --suite\n";
      std::exit(exitCode(AutotestExitCode::InvalidInput));
    }
    std::vector<AutotestCase> cases = loadCasesOrExit();
    AutotestRunOptions options = makeOptions(
        runArtifactDir, runJsonOut, runHtmlOut, runBaselineProfile,
        runWindowMode, runDry, runEffective, runVerboseFrames, command);
    if (!runCase.empty()) {
      AutotestCase testCase = requireCase(cases, runCase);
      if (runEffective) {
        auto effective = formatAutotestEffectiveConfigJson(testCase, options);
        if (!effective.hasError()) {
          std::cout << effective.value();
        }
      }
      AutotestRunResult result = runAutotestCase(std::move(testCase), options);
      std::cout << result.message
                << "\nreport: " << result.reportPath.generic_string()
                << "\nhtml: " << result.htmlPath.generic_string() << "\n";
      std::exit(exitCode(result.exitCode));
    }
    AutotestSuiteRunResult suiteResult =
        runAutotestSuite(std::move(cases), runSuite, options);
    std::cout << suiteResult.message
              << "\nreport: " << suiteResult.reportPath.generic_string()
              << "\nhtml: " << suiteResult.htmlPath.generic_string() << "\n";
    std::exit(exitCode(suiteResult.exitCode));
  });

  std::string recordCase;
  std::filesystem::path recordOut;
  std::string recordWindowMode;
  std::string recordBaselineProfile = "local-nvrhi-visible";
  bool recordEffective = false;
  auto *record =
      app.add_subcommand("record", "Record candidate autotest artifacts");
  record->add_option("--case", recordCase, "Case id")->required();
  record->add_option("--out", recordOut, "Output package directory")
      ->required();
  record->add_option("--window-mode", recordWindowMode,
                     "Override manifest mode: visible, hidden, or headless");
  record->add_option("--baseline-profile", recordBaselineProfile,
                     "Candidate baseline profile");
  record->add_flag("--print-effective-config", recordEffective,
                   "Print resolved config before running");
  record->callback([&]() {
    std::vector<AutotestCase> cases = loadCasesOrExit();
    AutotestCase testCase = requireCase(cases, recordCase);
    AutotestRunOptions options =
        makeOptions(recordOut, {}, {}, recordBaselineProfile, recordWindowMode,
                    false, recordEffective, false, command);
    if (recordEffective) {
      auto effective = formatAutotestEffectiveConfigJson(testCase, options);
      if (!effective.hasError()) {
        std::cout << effective.value();
      }
    }
    AutotestRunResult result = recordAutotestCase(std::move(testCase), options);
    std::cout << result.message
              << "\nreport: " << result.reportPath.generic_string()
              << "\nhtml: " << result.htmlPath.generic_string() << "\n";
    std::exit(exitCode(result.exitCode));
  });

  auto *baseline = app.add_subcommand(
      "baseline", "Inspect, verify, and accept governed autotest baselines");
  baseline->require_subcommand(1);

  std::string baselineInspectCase;
  std::string baselineInspectProfile = "local-nvrhi-visible";
  auto *baselineInspect = baseline->add_subcommand(
      "inspect", "Inspect baseline identity, files, and SHA-256 digests");
  baselineInspect->add_option("--case", baselineInspectCase, "Case id")
      ->required();
  baselineInspect->add_option("--profile,--baseline-profile",
                              baselineInspectProfile, "Baseline profile");
  baselineInspect->callback([&]() {
    std::vector<AutotestCase> cases = loadCasesOrExit();
    const AutotestCase expectedCase = requireCase(cases, baselineInspectCase);
    auto inspection =
        inspectAutotestBaseline(expectedCase, baselineInspectProfile);
    if (inspection.hasError()) {
      std::cerr << inspection.error() << "\n";
      std::exit(exitCode(AutotestExitCode::InvalidInput));
    }
    std::cout << inspection.value() << "\n";
  });

  std::string baselineVerifyCase;
  std::string baselineVerifyProfile = "local-nvrhi-visible";
  auto *baselineVerify = baseline->add_subcommand(
      "verify", "Verify the reviewed plan, approval history, and file digests");
  baselineVerify->add_option("--case", baselineVerifyCase, "Case id")
      ->required();
  baselineVerify->add_option("--profile,--baseline-profile",
                             baselineVerifyProfile, "Baseline profile");
  baselineVerify->callback([&]() {
    std::vector<AutotestCase> cases = loadCasesOrExit();
    const AutotestCase expectedCase = requireCase(cases, baselineVerifyCase);
    auto verified = verifyAutotestBaseline(expectedCase, baselineVerifyProfile);
    auto inspection =
        inspectAutotestBaseline(expectedCase, baselineVerifyProfile);
    if (inspection.hasError()) {
      std::cerr << inspection.error() << "\n";
      std::exit(exitCode(AutotestExitCode::RuntimeError));
    }
    std::cout << inspection.value() << "\n";
    if (verified.hasError()) {
      std::cerr << verified.error() << "\n";
      const bool missing =
          verified.error().find("missing") != std::string::npos;
      std::exit(exitCode(missing ? AutotestExitCode::MissingBaseline
                                 : AutotestExitCode::InvalidInput));
    }
  });

  std::string approveCase;
  std::string approveReason;
  std::string approveActor;
  std::string approveConfirmPlan;
  std::string approveProfile = "local-nvrhi-visible";
  std::filesystem::path approveArtifacts;
  bool approveDryRun = false;
  auto *approve =
      app.add_subcommand("approve", "Compatibility alias for baseline accept");
  approve->add_option("--case", approveCase, "Case id")->required();
  approve->add_option("--reason", approveReason, "Approval reason")->required();
  approve->add_option("--actor", approveActor, "Approval actor")->required();
  approve
      ->add_option("--from,--from-artifacts", approveArtifacts,
                   "Candidate package root containing cases/")
      ->required();
  approve->add_option("--profile,--baseline-profile", approveProfile,
                      "Baseline profile");
  approve->add_flag("--dry-run", approveDryRun,
                    "Print the reviewed promotion plan without mutation");
  approve->add_option("--confirm-plan", approveConfirmPlan,
                      "Digest emitted by the reviewed dry-run plan");

  auto *baselineAccept = baseline->add_subcommand(
      "accept", "Review and atomically accept an autotest baseline");
  baselineAccept->add_option("--case", approveCase, "Case id")->required();
  baselineAccept->add_option("--reason", approveReason, "Approval reason")
      ->required();
  baselineAccept->add_option("--actor", approveActor, "Approval actor")
      ->required();
  baselineAccept
      ->add_option("--from,--from-artifacts", approveArtifacts,
                   "Candidate package root containing cases/")
      ->required();
  baselineAccept->add_option("--profile,--baseline-profile", approveProfile,
                             "Baseline profile");
  baselineAccept->add_flag(
      "--dry-run", approveDryRun,
      "Print the reviewed promotion plan without mutation");
  baselineAccept->add_option("--confirm-plan", approveConfirmPlan,
                             "Digest emitted by the reviewed dry-run plan");

  const auto approveCallback = [&]() {
    std::vector<AutotestCase> cases = loadCasesOrExit();
    const AutotestCase expectedCase = requireCase(cases, approveCase);
    auto caseDir = nuri::tools::core::resolvePathUnder(
        approveArtifacts, std::filesystem::path("cases") / expectedCase.id);
    if (caseDir.hasError()) {
      std::cerr << caseDir.error() << "\n";
      std::exit(exitCode(AutotestExitCode::InvalidInput));
    }
    auto report = readAutotestReportFile(caseDir.value() / "report.json");
    if (report.hasError()) {
      std::cerr << report.error() << "\n";
      std::exit(exitCode(AutotestExitCode::InvalidInput));
    }
    report.value().artifacts.caseDir = caseDir.value();
    auto plan =
        planAutotestBaselines(expectedCase, report.value(), approveProfile,
                              approveReason, approveActor);
    if (plan.hasError()) {
      std::cerr << plan.error() << "\n";
      std::exit(exitCode(AutotestExitCode::InvalidInput));
    }
    auto planJson = writeAutotestBaselinePlanJson(plan.value());
    if (planJson.hasError()) {
      std::cerr << planJson.error() << "\n";
      std::exit(exitCode(AutotestExitCode::RuntimeError));
    }
    if (approveDryRun) {
      std::cout << planJson.value();
      return;
    }
    if (approveConfirmPlan.empty()) {
      std::cerr << "approval requires --confirm-plan " << plan.value().digest
                << " after reviewing --dry-run\n";
      std::exit(exitCode(AutotestExitCode::InvalidInput));
    }
    auto approved = approveAutotestBaselines(expectedCase, report.value(),
                                             approveProfile, approveReason,
                                             approveConfirmPlan, approveActor);
    if (approved.hasError()) {
      std::cerr << approved.error() << "\n";
      std::exit(exitCode(AutotestExitCode::InvalidInput));
    }
    std::cout << "autotest baseline approved\n";
  };
  approve->callback(approveCallback);
  baselineAccept->callback(approveCallback);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    const int parserExit = app.exit(e);
    return parserExit == 0 ? 0 : exitCode(AutotestExitCode::InvalidInput);
  }
  return 0;
}
