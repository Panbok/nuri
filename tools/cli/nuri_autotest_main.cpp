#include "nuri/tools/autotest/autotest_manifest.h"
#include "nuri/tools/autotest/autotest_record.h"
#include "nuri/tools/autotest/autotest_runner.h"

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

[[nodiscard]] AutotestRunOptions makeOptions(
    const std::filesystem::path &artifactDir,
    const std::filesystem::path &jsonOut,
    const std::filesystem::path &htmlOut, const std::string &baselineProfile,
    const std::string &windowMode, bool dryRun, bool printEffectiveConfig,
    bool verboseFrames, const std::string &command) {
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
  std::string runBaselineProfile = "local-lvk-visible";
  std::string runWindowMode = "visible";
  bool runDry = false;
  bool runEffective = false;
  bool runVerboseFrames = false;
  auto *run = app.add_subcommand("run", "Run autotest case or suite");
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
  run->add_flag("--verbose-frames", runVerboseFrames,
                "Include verbose frame data when available");
  run->callback([&]() {
    if (runCase.empty() == runSuite.empty()) {
      std::cerr << "run requires exactly one of --case or --suite\n";
      std::exit(exitCode(AutotestExitCode::InvalidInput));
    }
    std::vector<AutotestCase> cases = loadCasesOrExit();
    AutotestRunOptions options =
        makeOptions(runArtifactDir, runJsonOut, runHtmlOut,
                    runBaselineProfile, runWindowMode, runDry, runEffective,
                    runVerboseFrames, command);
    if (!runCase.empty()) {
      AutotestCase testCase = requireCase(cases, runCase);
      if (runEffective) {
        auto effective = formatAutotestEffectiveConfigJson(testCase, options);
        if (!effective.hasError()) {
          std::cout << effective.value();
        }
      }
      AutotestRunResult result = runAutotestCase(std::move(testCase), options);
      std::cout << result.message << "\nreport: "
                << result.reportPath.generic_string() << "\nhtml: "
                << result.htmlPath.generic_string() << "\n";
      std::exit(exitCode(result.exitCode));
    }
    AutotestSuiteRunResult suiteResult =
        runAutotestSuite(std::move(cases), runSuite, options);
    std::cout << suiteResult.message << "\nreport: "
              << suiteResult.reportPath.generic_string() << "\nhtml: "
              << suiteResult.htmlPath.generic_string() << "\n";
    std::exit(exitCode(suiteResult.exitCode));
  });

  std::string recordCase;
  std::filesystem::path recordOut;
  std::string recordWindowMode = "visible";
  bool recordEffective = false;
  auto *record = app.add_subcommand("record", "Record autotest artifacts");
  record->add_option("--case", recordCase, "Case id")->required();
  record->add_option("--out", recordOut, "Output package directory")
      ->required();
  record->add_option("--window-mode", recordWindowMode,
                     "visible, hidden, or headless");
  record->add_flag("--print-effective-config", recordEffective,
                   "Print resolved config before running");
  record->callback([&]() {
    std::vector<AutotestCase> cases = loadCasesOrExit();
    AutotestCase testCase = requireCase(cases, recordCase);
    AutotestRunOptions options =
        makeOptions(recordOut, {}, {}, "local-lvk-visible", recordWindowMode,
                    false, recordEffective, false, command);
    if (recordEffective) {
      auto effective = formatAutotestEffectiveConfigJson(testCase, options);
      if (!effective.hasError()) {
        std::cout << effective.value();
      }
    }
    AutotestRunResult result = recordAutotestCase(std::move(testCase), options);
    std::cout << result.message << "\nreport: "
              << result.reportPath.generic_string() << "\nhtml: "
              << result.htmlPath.generic_string() << "\n";
    std::exit(exitCode(result.exitCode));
  });

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }
  return 0;
}
