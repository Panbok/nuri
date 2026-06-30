#include "nuri/tools/benchmark/benchmark_compare.h"
#include "nuri/tools/benchmark/benchmark_graph.h"
#include "nuri/tools/benchmark/benchmark_manifest.h"
#include "nuri/tools/benchmark/benchmark_report.h"
#include "nuri/tools/benchmark/benchmark_runner.h"

#include <CLI/CLI.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <cstdlib>

using namespace nuri::tools::benchmark;
using nuri::Result;

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
    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir)) {
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
  std::optional<uint32_t> samples;
  std::filesystem::path jsonOut;
  std::filesystem::path artifactDir;
  std::filesystem::path runHtmlOut;
  std::vector<std::string> runHtmlMetrics;
  std::vector<std::string> runHtmlStats;
  bool dryRun = false;
  bool printEffectiveConfig = false;
  bool tracyDiagnostic = false;
  bool verboseFrames = false;
  auto *run = app.add_subcommand("run", "Run benchmark case or suite");
  run->add_option("--case", runCase, "Case id");
  run->add_option("--suite", runSuite, "Suite name");
  run->add_option("--samples", samples, "Override sample windows");
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
  run->add_flag("--tracy-diagnostic", tracyDiagnostic, "Diagnostic run");
  run->add_flag("--verbose-frames", verboseFrames,
                "Include verbose frame renderer metrics");
  run->callback([&]() {
    if (runCase.empty() == runSuite.empty()) {
      std::cerr << "run requires exactly one of --case or --suite\n";
      std::exit(exitCode(BenchmarkExitCode::InvalidInput));
    }
    std::vector<BenchmarkCase> cases = loadCasesOrExit();
    BenchmarkRunOptions options{
        .samplesOverride = samples,
        .jsonOut = jsonOut,
        .artifactDir = artifactDir,
        .dryRun = dryRun,
        .printEffectiveConfig = printEffectiveConfig,
        .tracyDiagnostic = tracyDiagnostic,
        .verboseFrames = verboseFrames,
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
      BenchmarkRunResult result = runBenchmarkCase(std::move(benchmarkCase), options);
      std::cout << result.message << "\nreport: "
                << result.reportPath.generic_string() << "\n";
      writeGraphOrExit({result.report}, runHtmlOut,
                       makeGraphOptions(runHtmlMetrics, runHtmlStats,
                                        "Nuri Benchmark Run"));
      std::exit(exitCode(result.exitCode));
    }

    BenchmarkSuiteRunResult suiteResult =
        runBenchmarkSuite(std::move(cases), runSuite, options);
    const std::filesystem::path suiteOut =
        jsonOut.empty()
            ? (artifactDir.empty()
                   ? benchmarkRepoRoot() / "artifacts" / "bench" /
                         utcTimestampForPath() / "run.json"
                   : artifactDir / "run.json")
            : jsonOut;
    std::string json = "{\n  \"kind\": \"nuri.benchmark.suite_report\",\n"
                       "  \"caseReports\": [\n";
    bool first = true;
    for (const BenchmarkRunResult &caseResult : suiteResult.caseResults) {
      if (!first) {
        json += ",\n";
      }
      first = false;
      json += "    \"" + caseResult.reportPath.generic_string() + "\"";
    }
    json += "\n  ]\n}\n";
    writeTextFile(suiteOut, json);
    std::cout << suiteResult.message << "\nreport: "
              << suiteOut.generic_string() << "\n";
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

  std::filesystem::path currentReport;
  std::filesystem::path baselineReport;
  std::filesystem::path compareOut;
  std::filesystem::path compareHtmlOut;
  std::vector<std::string> compareHtmlMetrics;
  std::vector<std::string> compareHtmlStats;
  bool forceCompare = false;
  auto *compare = app.add_subcommand("compare", "Compare reports");
  compare->add_option("--current", currentReport, "Current report")->required();
  compare->add_option("--baseline", baselineReport, "Baseline report")
      ->required();
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
  compare->add_flag("--force", forceCompare, "Demote invalid preconditions");
  compare->callback([&]() {
    auto current = readBenchmarkReportFile(currentReport);
    if (current.hasError()) {
      std::cerr << current.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::MissingBaseline));
    }
    auto baseline = readBenchmarkReportFile(baselineReport);
    if (baseline.hasError()) {
      std::cerr << baseline.error() << "\n";
      std::exit(exitCode(BenchmarkExitCode::MissingBaseline));
    }
    BenchmarkReport currentValue = std::move(current.value());
    BenchmarkReport baselineValue = std::move(baseline.value());
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
      std::vector<BenchmarkReport> reports;
      reports.push_back(std::move(baselineValue));
      reports.push_back(std::move(currentValue));
      writeGraphOrExit(reports, compareHtmlOut,
                       makeGraphOptions(compareHtmlMetrics, compareHtmlStats,
                                        "Nuri Benchmark Comparison"));
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
  summarize->add_option("--reports", reportsDir, "Report directory")->required();
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
    json += "\n  ],\n  \"reportCount\": " + std::to_string(reportCount) +
            "\n}\n";
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
    return app.exit(e);
  }
  return 0;
}
