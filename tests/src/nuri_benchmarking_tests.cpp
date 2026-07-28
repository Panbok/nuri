#include "nuri_benchmark_child_fixture_path.h"
#include "tests_pch.h"

#include "render_graph_test_support.h"

#include "nuri/tools/benchmark/benchmark_baseline.h"
#include "nuri/tools/benchmark/benchmark_check.h"
#include "nuri/tools/benchmark/benchmark_compare.h"
#include "nuri/tools/benchmark/benchmark_graph.h"
#include "nuri/tools/benchmark/benchmark_manifest.h"
#include "nuri/tools/benchmark/benchmark_report.h"
#include "nuri/tools/benchmark/benchmark_runner.h"
#include "nuri/tools/benchmark/benchmark_stats.h"
#include "nuri/tools/core/result_envelope_v2.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <string>

namespace {

using namespace nuri;
using namespace nuri::test_support;
using namespace nuri::tools::benchmark;

std::filesystem::path makeTempPath(std::string_view stem,
                                   std::string_view extension) {
  const auto tick =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("nuri_" + std::string(stem) + "_" + std::to_string(tick) +
          std::string(extension));
}

void writeFile(const std::filesystem::path &path, std::string_view text) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file << text;
}

std::string replaceFirst(std::string text, std::string_view needle,
                         std::string_view replacement) {
  const size_t position = text.find(needle);
  EXPECT_NE(position, std::string::npos);
  if (position != std::string::npos) {
    text.replace(position, needle.size(), replacement);
  }
  return text;
}

MetricStats makeStats(double median, double p95) {
  MetricStats stats{};
  stats.count = 3u;
  stats.min = median;
  stats.median = median;
  stats.p90 = p95;
  stats.p95 = p95;
  stats.p99 = p95;
  stats.max = p95;
  stats.mean = median;
  return stats;
}

BenchmarkReport makeComparableReport(double cpuMedian, double cpuP95,
                                     double gpuMedian, double gpuP95) {
  BenchmarkReport report{};
  report.kind = "nuri.benchmark.report";
  report.benchmarkCase.id = "case.default";
  report.benchmarkCase.suite = "case";
  report.benchmarkCase.backend = "default";
  report.benchmarkCase.resolution = {640u, 360u};
  report.benchmarkCase.presentMode = "immediate";
  report.benchmarkCase.requiredMetrics = {"cpu.render_submit_ms",
                                          "gpu.scopes_sum_ms"};
  report.benchmarkCase.thresholds = BenchmarkThresholds{
      .failPercent = 10.0,
      .failAbsoluteMs = 0.2,
      .warnPercent = 5.0,
      .warnAbsoluteMs = 0.1,
  };
  report.environment.gpuBackend = "nvrhi";
  report.environment.gpuDeviceName = "test GPU";
  report.environment.gpuVendorId = 0x10deu;
  report.environment.gpuDeviceId = 0x2684u;
  report.environment.gpuDriverVersion = "test-driver";
  report.environment.osName = "Windows";
  report.environment.osVersion = "test-os";
  report.environment.cpuName = "test CPU";
  report.environment.cpuLogicalThreadCount = 8u;
  report.environment.requestedPresentMode = "immediate";
  report.environment.resolvedPresentMode = "immediate";
  report.environment.windowMode = "windowed";
  report.environment.windowVisible = true;
  report.environment.renderGraphWorkerCount = 1u;
  report.environment.buildType = "Release";
  report.environment.cmakeToolProfile = "bench";
  report.environment.tracyEnabled = false;
  report.environment.devChecks = false;
  report.run.samples = 1u;
  report.run.warmupFrames = 1u;
  report.run.measurementFrames = 3u;
  report.run.maxDrainFrames = 30u;
  report.run.drainTimeoutMs = 5000u;
  report.run.fixedDeltaSeconds = 1.0 / 60.0;
  report.run.validForComparison = true;
  report.stats.emplace("cpu.render_submit_ms", makeStats(cpuMedian, cpuP95));
  report.stats.emplace("gpu.scopes_sum_ms", makeStats(gpuMedian, gpuP95));
  return report;
}

void addSampleObservations(BenchmarkReport &report, std::string metricId,
                           std::initializer_list<double> values) {
  report.run.samples = static_cast<uint32_t>(values.size());
  report.repeatObservations.unit = "in-process-sample-window";
  report.repeatObservations.independent = false;
  report.repeatObservations.count = static_cast<uint32_t>(values.size());
  uint32_t sampleIndex = 0u;
  for (double value : values) {
    if (report.sampleStats.size() <= sampleIndex) {
      report.sampleStats.push_back(
          BenchmarkSampleStats{.sampleIndex = sampleIndex});
    }
    report.sampleStats[sampleIndex].stats[metricId] = makeStats(value, value);
    ++sampleIndex;
  }
}

nuri::tools::core::BaselineProfile makeInvestigativeBaselineProfile() {
  nuri::tools::core::BaselineProfile profile{};
  profile.id = "local-nvrhi-visible";
  profile.description = "test investigative profile";
  profile.authority.authoritative = false;
  profile.authority.allowDirtyTree = true;
  profile.authority.reason = "test profile";
  profile.benchmarkPolicy.minimumRepetitions = 3u;
  profile.benchmarkPolicy.warmupStability = "investigative";
  profile.benchmarkPolicy.warmupWindowFrames = 2u;
  profile.benchmarkPolicy.warmupMaxDriftPercent = 15.0;
  profile.benchmarkPolicy.requiredMetrics = {"cpu.render_submit_ms"};
  profile.benchmarkPolicy.thresholdOwnership = "baseline";
  return profile;
}

nuri::tools::core::BaselineProfile
makeFixtureBaselineProfile(bool authoritative) {
  nuri::tools::core::BaselineProfile profile{};
  profile.id = authoritative ? "fixture-authoritative" : "fixture-local";
  profile.authority.authoritative = authoritative;
  profile.authority.allowDirtyTree = !authoritative;
  profile.environment.os = "TestOS";
  profile.environment.backend = "nvrhi-default";
  profile.environment.windowMode = "visible";
  profile.environment.gpuVendorId = 0x10deu;
  profile.environment.gpuDeviceId = 0x1234u;
  profile.environment.driver = "fixture-driver";
  profile.execution.presentMode = "immediate";
  profile.execution.profiling = "off";
  profile.execution.devChecks = false;
  profile.benchmarkPolicy.minimumRepetitions = 3u;
  profile.benchmarkPolicy.warmupStability =
      authoritative ? "required" : "investigative";
  profile.benchmarkPolicy.warmupWindowFrames = 2u;
  profile.benchmarkPolicy.warmupMaxDriftPercent = 15.0;
  profile.benchmarkPolicy.requiredMetrics = {"cpu.render_submit_ms"};
  profile.benchmarkPolicy.thresholdOwnership = "baseline";
  return profile;
}

BenchmarkRunOptions
makeIsolatedFixtureOptions(const std::filesystem::path &artifactDir,
                           const nuri::tools::core::BaselineProfile &profile,
                           uint32_t repetitions = 3u) {
  return BenchmarkRunOptions{
      .isolatedRepetitions = repetitions,
      .processExecutable =
          std::filesystem::path(NURI_BENCHMARK_CHILD_FIXTURE_PATH),
      .repetitionTimeout = std::chrono::seconds(2),
      .artifactDir = artifactDir,
      .baselineProfileId = profile.id,
      .baselineProfileAuthoritative = profile.authority.authoritative,
      .baselineProfileMinimumRepetitions =
          profile.benchmarkPolicy.minimumRepetitions,
      .baselineProfileWarmupStability = profile.benchmarkPolicy.warmupStability,
      .baselineProfileWarmupWindowFrames =
          profile.benchmarkPolicy.warmupWindowFrames,
      .baselineProfileWarmupMaxDriftPercent =
          profile.benchmarkPolicy.warmupMaxDriftPercent,
      .baselineProfileRequiredMetrics = profile.benchmarkPolicy.requiredMetrics,
      .baselineProfile = profile,
      .command = "nuri-bench run --case test.isolated --repetitions 3",
  };
}

BenchmarkBaselineSource
writeGovernedBenchmarkSource(const std::filesystem::path &runRoot,
                             BenchmarkThresholds thresholds = {}) {
  BenchmarkReport report = makeComparableReport(10.0, 10.5, 8.0, 8.5);
  report.generatedAtUtc = "2026-07-10T00:00:00Z";
  report.command = "nuri-bench run --case case.default";
  report.environment.commitHash = "test-commit";
  report.benchmarkCase.thresholds = thresholds;
  report.profile.id = "local-nvrhi-visible";
  report.profile.profileAuthoritative = false;
  report.profile.authoritative = false;
  report.profile.minimumRepetitions = 3u;
  report.profile.completedRepetitions = 0u;
  report.profile.repetitionRequirementSatisfied = false;
  report.profile.repetitionUnit = "not-collected";
  report.profile.warmupStabilityPolicy = "investigative";
  report.profile.warmupStabilityStatus = "unknown";
  report.profile.warmupWindowFrames = 2u;
  report.profile.warmupMaxDriftPercent = 15.0;
  report.profile.requiredMetrics = {"cpu.render_submit_ms"};
  report.profile.authorityBlockers = {"baseline profile is investigative"};
  report.run.validForComparison = false;
  report.frames.clear();
  for (uint64_t frameIndex = 0u; frameIndex < 3u; ++frameIndex) {
    report.frames.push_back(BenchmarkFrameRecord{
        .frameIndex = frameIndex,
        .sampleIndex = 0u,
        .measured = true,
        .measurements = {{"cpu.render_submit_ms", 10.0 + 0.1 * frameIndex},
                         {"gpu.scopes_sum_ms", 8.0 + 0.1 * frameIndex}},
    });
  }
  report.frames[0].metrics.opaque.totalInstances = 42u;
  computeBenchmarkReportStats(report);
  const std::filesystem::path reportPath =
      runRoot / "cases" / "case.default.json";
  std::filesystem::create_directories(reportPath.parent_path());
  auto written = writeBenchmarkReportFile(report, reportPath, true);
  EXPECT_FALSE(written.hasError()) << written.error();
  auto source = loadBenchmarkBaselineSource(runRoot, "case.default");
  EXPECT_FALSE(source.hasError()) << source.error();
  return source.hasError() ? BenchmarkBaselineSource{}
                           : std::move(source.value());
}

struct DiagnosticReportCase {
  std::string_view label;
  void (*mark)(BenchmarkReport &);
};

const std::array<DiagnosticReportCase, 3> kDiagnosticReportCases{{
    {"Tracy",
     [](BenchmarkReport &report) {
       report.environment.tracyDiagnostic = true;
     }},
    {"RGP shader",
     [](BenchmarkReport &report) { report.rgp.requested = true; }},
    {"RenderDoc",
     [](BenchmarkReport &report) { report.renderDoc.requested = true; }},
}};

BenchmarkCase makeCheckFixtureCase() {
  BenchmarkCase benchmarkCase{};
  benchmarkCase.id = "test.check";
  benchmarkCase.suite = "test";
  benchmarkCase.backend = "default";
  benchmarkCase.resolution = {1280u, 720u};
  benchmarkCase.presentMode = "immediate";
  benchmarkCase.warmupFrames = 1u;
  benchmarkCase.measurementFrames = 2u;
  benchmarkCase.maxDrainFrames = 0u;
  benchmarkCase.drainTimeoutMs = 0u;
  benchmarkCase.requiredMetrics = {"cpu.render_submit_ms"};
  return benchmarkCase;
}

nuri::tools::core::BaselineProfile makeCheckFixtureProfile() {
  auto profile = makeFixtureBaselineProfile(false);
  profile.environment.os = "any";
  return profile;
}

BenchmarkReport makeCheckFixtureBaseline(double valueScale = 1.0) {
  const auto metricStats = [](std::initializer_list<double> values) {
    auto computed = computeMetricStats(std::vector<double>(values));
    EXPECT_FALSE(computed.hasError()) << computed.error();
    return computed.hasError() ? MetricStats{} : computed.value();
  };
  BenchmarkReport report{};
  report.generatedAtUtc = "2026-07-10T00:00:00Z";
  report.command = "deterministic benchmark check fixture";
  report.benchmarkCase = makeCheckFixtureCase();
  report.environment.osName = "TestOS";
  report.environment.osVersion = "1.0";
  report.environment.cpuName = "Fixture CPU";
  report.environment.cpuLogicalThreadCount = 8u;
  report.environment.gpuBackend = "nvrhi";
  report.environment.gpuBackendSource = "default";
  report.environment.gpuDeviceName = "Fixture GPU";
  report.environment.gpuVendorId = 0x10deu;
  report.environment.gpuDeviceId = 0x1234u;
  report.environment.gpuDriverVersion = "fixture-driver";
  report.environment.swapchainImageCount = 3u;
  report.environment.requestedPresentMode = "immediate";
  report.environment.resolvedPresentMode = "immediate";
  report.environment.windowMode = "windowed";
  report.environment.windowVisible = true;
  report.environment.renderGraphWorkerCount = 1u;
  report.environment.buildType = "Release";
  report.environment.cmakeToolProfile = "tools";
  report.run.samples = 1u;
  report.run.warmupFrames = 1u;
  report.run.measurementFrames = 2u;
  report.run.maxDrainFrames = 0u;
  report.run.drainTimeoutMs = 0u;
  report.run.fixedDeltaSeconds = 1.0 / 60.0;
  report.run.validForComparison = true;
  report.profile.id = "fixture-local";
  report.profile.profileAuthoritative = false;
  report.profile.authoritative = false;
  report.profile.minimumRepetitions = 3u;
  report.profile.completedRepetitions = 3u;
  report.profile.repetitionRequirementSatisfied = true;
  report.profile.repetitionUnit = "isolated-process";
  report.profile.warmupStabilityPolicy = "investigative";
  report.profile.warmupStabilityStatus = "stable";
  report.profile.warmupWindowFrames = 2u;
  report.profile.warmupMaxDriftPercent = 15.0;
  report.profile.requiredMetrics = {"cpu.render_submit_ms"};
  report.profile.authorityBlockers = {"baseline profile is investigative"};
  report.repeatObservations = {
      .unit = "isolated-process", .independent = true, .count = 3u};
  for (uint32_t index = 0u; index < 3u; ++index) {
    const double cpu = (10.0 + static_cast<double>(index)) * valueScale;
    const double gpu = (8.0 + static_cast<double>(index) * 0.8) * valueScale;
    report.sampleStats.push_back(
        {.sampleIndex = index,
         .warmupStable = true,
         .measuredFrameCount = 2u,
         .stats = {
             {"cpu.render_submit_ms",
              metricStats({cpu - 0.5 * valueScale, cpu + 0.5 * valueScale})},
             {"gpu.scopes_sum_ms",
              metricStats({gpu - 0.4 * valueScale, gpu + 0.4 * valueScale})}}});
  }
  report.stats = {
      {"cpu.render_submit_ms",
       metricStats({10.0 * valueScale, 11.0 * valueScale, 12.0 * valueScale})},
      {"gpu.scopes_sum_ms",
       metricStats({8.0 * valueScale, 8.8 * valueScale, 9.6 * valueScale})}};
  return report;
}

bool acceptCheckFixtureBaseline(const std::filesystem::path &tempRoot,
                                const std::filesystem::path &baselineRoot,
                                double valueScale = 1.0) {
  const BenchmarkReport report = makeCheckFixtureBaseline(valueScale);
  const std::filesystem::path reportPath =
      tempRoot / "source" / "cases" / "test.check.json";
  auto written = writeBenchmarkReportFile(report, reportPath, false);
  if (written.hasError()) {
    ADD_FAILURE() << written.error();
    return false;
  }
  auto source = loadBenchmarkBaselineSource(tempRoot / "source", "test.check");
  if (source.hasError()) {
    ADD_FAILURE() << source.error();
    return false;
  }
  const auto profile = makeCheckFixtureProfile();
  auto plan = planBenchmarkBaseline(source.value(), profile, "fixture review",
                                    "test-agent", baselineRoot);
  if (plan.hasError()) {
    ADD_FAILURE() << plan.error();
    return false;
  }
  auto accepted = acceptBenchmarkBaseline(
      source.value(), profile, "fixture review", "test-agent",
      plan.value().digest,
      BenchmarkBaselineAcceptOptions{.baselineRoot = baselineRoot});
  if (accepted.hasError()) {
    ADD_FAILURE() << accepted.error();
    return false;
  }
  return true;
}

std::string joinCheckDiagnostics(const std::vector<std::string> &messages) {
  std::string joined;
  for (const std::string &message : messages) {
    joined += joined.empty() ? message : "; " + message;
  }
  return joined;
}

GpuTimingReport makeOpaqueTimingReport(uint64_t frameIndex, float timeMs) {
  GpuTimingReport report{};
  report.availableScopeMask = gpuTimingScopeToBit(GpuTimingScope::Opaque);
  report.opaqueSourceFrameIndex = frameIndex;
  report.opaqueTimeMs = timeMs;
  return report;
}

TEST(NuriBenchmarkingTest,
     BenchmarkCheckDistinguishesMissingAndUnverifiedBaselines) {
  const std::filesystem::path tempRoot =
      makeTempPath("benchmark_check_baseline_preflight", "");
  const BenchmarkCase benchmarkCase = makeCheckFixtureCase();
  const auto profile = makeCheckFixtureProfile();
  const BenchmarkCheckOptions options{
      .processExecutable =
          std::filesystem::path(NURI_BENCHMARK_CHILD_FIXTURE_PATH),
      .artifactRoot = tempRoot / "artifacts",
      .baselineRoot = tempRoot / "baselines",
      .repetitionTimeout = std::chrono::seconds(2),
      .requestedSelection = "case:test.check",
      .command = "nuri-bench check --case test.check --profile fixture-local"};

  const BenchmarkCheckResult missing =
      checkBenchmarkCases({benchmarkCase}, profile, options);
  EXPECT_EQ(missing.exitCode, BenchmarkExitCode::MissingBaseline);
  ASSERT_EQ(missing.cases.size(), 1u);
  EXPECT_TRUE(missing.cases[0].run.reportPath.empty());

  const std::filesystem::path unverified = options.baselineRoot / profile.id /
                                           benchmarkCase.suite /
                                           (benchmarkCase.id + ".json");
  auto written =
      writeBenchmarkReportFile(makeCheckFixtureBaseline(), unverified, false);
  ASSERT_FALSE(written.hasError()) << written.error();
  const BenchmarkCheckResult rejected =
      checkBenchmarkCases({benchmarkCase}, profile, options);
  ASSERT_EQ(rejected.cases.size(), 1u);
  std::ifstream rejectedEnvelopeFile(rejected.envelopePath, std::ios::binary);
  const std::string rejectedEnvelopeJson{
      std::istreambuf_iterator<char>(rejectedEnvelopeFile),
      std::istreambuf_iterator<char>()};
  auto rejectedEnvelope =
      nuri::tools::core::readResultEnvelopeV2(rejectedEnvelopeJson);
  ASSERT_FALSE(rejectedEnvelope.hasError()) << rejectedEnvelope.error();
  std::string rejectedDiagnostics;
  for (const auto &diagnostic : rejectedEnvelope.value().diagnostics) {
    rejectedDiagnostics += rejectedDiagnostics.empty()
                               ? diagnostic.message
                               : "; " + diagnostic.message;
  }
  EXPECT_EQ(rejected.exitCode, BenchmarkExitCode::InvalidInput)
      << rejected.message << ": envelope="
      << std::string(nuri::tools::core::toolOutcomeName(
             rejectedEnvelope.value().status))
      << ", caseExit=" << static_cast<int>(rejected.cases[0].exitCode)
      << ", diagnostics=" << rejectedDiagnostics;
  EXPECT_EQ(rejectedEnvelope.value().status,
            nuri::tools::core::ToolOutcome::Invalid);
  EXPECT_FALSE(rejected.cases[0].baselineVerification.valid);
  EXPECT_TRUE(rejected.cases[0].run.reportPath.empty());

  std::error_code error;
  std::filesystem::remove_all(tempRoot, error);
}

TEST(NuriBenchmarkingTest, BenchmarkCheckReturnsRegressionForGateFailure) {
  const std::filesystem::path tempRoot =
      makeTempPath("benchmark_check_regression", "");
  const std::filesystem::path baselineRoot = tempRoot / "baselines";
  ASSERT_TRUE(acceptCheckFixtureBaseline(tempRoot, baselineRoot, 0.5));
  const BenchmarkCheckResult result = checkBenchmarkCases(
      {makeCheckFixtureCase()}, makeCheckFixtureProfile(),
      {.processExecutable =
           std::filesystem::path(NURI_BENCHMARK_CHILD_FIXTURE_PATH),
       .artifactRoot = tempRoot / "artifacts",
       .baselineRoot = baselineRoot,
       .repetitionTimeout = std::chrono::seconds(2),
       .requestedSelection = "case:test.check",
       .command =
           "nuri-bench check --case test.check --profile fixture-local"});

  EXPECT_EQ(result.exitCode, BenchmarkExitCode::Regression)
      << result.message << ": "
      << joinCheckDiagnostics(result.cases[0].comparison.errors);
  ASSERT_EQ(result.cases.size(), 1u);
  EXPECT_TRUE(result.cases[0].comparison.valid)
      << joinCheckDiagnostics(result.cases[0].comparison.errors);
  EXPECT_TRUE(result.cases[0].comparison.regression);
  EXPECT_TRUE(std::filesystem::is_regular_file(result.cases[0].comparisonPath));

  std::error_code error;
  std::filesystem::remove_all(tempRoot, error);
}

TEST(NuriBenchmarkingTest,
     BenchmarkCheckCompletesInvestigativeFlowWithPortableSummary) {
  const std::filesystem::path tempRoot =
      makeTempPath("benchmark_check_investigative", "");
  const std::filesystem::path baselineRoot = tempRoot / "baselines";
  ASSERT_TRUE(acceptCheckFixtureBaseline(tempRoot, baselineRoot));
  const BenchmarkCheckResult result = checkBenchmarkCases(
      {makeCheckFixtureCase()}, makeCheckFixtureProfile(),
      {.processExecutable =
           std::filesystem::path(NURI_BENCHMARK_CHILD_FIXTURE_PATH),
       .artifactRoot = tempRoot / "artifacts",
       .baselineRoot = baselineRoot,
       .repetitionTimeout = std::chrono::seconds(2),
       .requestedSelection = "case:test.check",
       .command =
           "nuri-bench check --case test.check --profile fixture-local"});

  EXPECT_EQ(result.exitCode, BenchmarkExitCode::Success)
      << result.message << ": "
      << joinCheckDiagnostics(result.cases[0].comparison.errors);
  ASSERT_EQ(result.cases.size(), 1u);
  EXPECT_TRUE(result.cases[0].comparison.valid)
      << joinCheckDiagnostics(result.cases[0].comparison.errors);
  EXPECT_FALSE(result.cases[0].comparison.regression);
  EXPECT_FALSE(result.cases[0].comparison.authoritative);
  ASSERT_TRUE(std::filesystem::is_regular_file(result.envelopePath));
  std::ifstream file(result.envelopePath, std::ios::binary);
  const std::string json{std::istreambuf_iterator<char>(file),
                         std::istreambuf_iterator<char>()};
  auto envelope = nuri::tools::core::readResultEnvelopeV2(json);
  ASSERT_FALSE(envelope.hasError()) << envelope.error();
  EXPECT_EQ(envelope.value().status,
            nuri::tools::core::ToolOutcome::Investigative);
  EXPECT_FALSE(envelope.value().authoritative);
  EXPECT_EQ(envelope.value().selection.selected, 1u);
  EXPECT_EQ(envelope.value().selection.completed, 1u);
  ASSERT_FALSE(envelope.value().children.empty());
  EXPECT_TRUE(envelope.value().children[0].result.has_value());
  EXPECT_FALSE(envelope.value().children[0].result->is_absolute());

  std::error_code error;
  std::filesystem::remove_all(tempRoot, error);
}

TEST(NuriBenchmarkingTest, ManifestRejectsUnknownKeys) {
  const std::filesystem::path path =
      makeTempPath("benchmark_manifest", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "bad.case",
              "suite": "bad",
              "unexpected": true
            })json");

  auto loaded = loadBenchmarkCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("unknown key"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriBenchmarkingTest, PostAASignatureUsesResolvedCanonicalPlan) {
  const std::filesystem::path path = makeTempPath("benchmark_post_aa", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "aa.post_aa.msaa4x_specular",
              "suite": "aa",
              "comparisonGroup": "aa.post_aa",
              "variant": "msaa4x_specular",
              "settings": {"antiAliasing": {
                "mode": "MSAA4x",
                "debugView": "SpecularAAVariance",
                "specularAAOverride": "None",
                "spatialPostMsaaCleanup": true,
                "postAA": {
                  "enabled": true,
                  "specular": "BakedClean",
                  "spatial": "Off",
                  "materialVarianceScale": 0.75,
                  "geometricVarianceScale": 0.25,
                  "maxSlopeVariance": 0.2
                }
              }}
            })json");

  auto loaded = loadBenchmarkCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  const auto &aa = loaded.value().settings.antiAliasing;
  EXPECT_FALSE(aa.debug.spatialPostMsaaCleanup);
  EXPECT_EQ(aa.debug.view, AntiAliasingDebugView::SpecularAAVariance);
  EXPECT_TRUE(aa.postAA.enabled);
  EXPECT_EQ(aa.postAA.specular, PostAASpecularAlgorithm::BakedClean);
  EXPECT_EQ(aa.postAA.spatial, PostAASpatialAlgorithm::Off);

  BenchmarkReport report{};
  report.benchmarkCase = loaded.value();
  auto json = writeBenchmarkReportJson(report, false);
  ASSERT_FALSE(json.hasError()) << json.error();
  EXPECT_NE(json.value().find("aa.postAA.requested=1"), std::string::npos);
  EXPECT_NE(json.value().find("aa.postAA.active=1"), std::string::npos);
  EXPECT_NE(json.value().find("aa.postAA.specular=1"), std::string::npos);
  EXPECT_NE(json.value().find("aa.postAA.spatial=0"), std::string::npos);
  EXPECT_NE(json.value().find("aa.postAA.materialVarianceScale=0.75"),
            std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriBenchmarkingTest, ManifestParsesCanonicalMsaaSampleRequirement) {
  const std::filesystem::path path =
      makeTempPath("benchmark_msaa_requirement", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "aa.requirement.msaa8x",
              "suite": "aa",
              "comparisonGroup": "aa.requirement",
              "variant": "msaa8x",
              "requirements": {"msaaSamples": 8}
            })json");

  auto loaded = loadBenchmarkCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  ASSERT_TRUE(loaded.value().requirements.msaaSamples.has_value());
  EXPECT_EQ(*loaded.value().requirements.msaaSamples, 8u);

  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "aa.requirement.invalid",
              "suite": "aa",
              "comparisonGroup": "aa.requirement",
              "variant": "invalid",
              "requirements": {"msaaSamples": 2}
            })json");
  loaded = loadBenchmarkCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("must be 1, 4, or 8"), std::string::npos);

  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "aa.requirement.conflict",
              "suite": "aa",
              "comparisonGroup": "aa.requirement",
              "variant": "conflict",
              "requirements": {"msaaSamples": 8, "msaa4x": true}
            })json");
  loaded = loadBenchmarkCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("conflicts"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriBenchmarkingTest, MsaaRequirementPreflightCoversEightSamples) {
  BenchmarkRequirements requirements{};
  requirements.msaaSamples = 8u;
  GpuMultisampleCapabilities capabilities{
      .sample4Color = true,
      .sample4Depth = true,
      .sample8Color = true,
      .sample8Depth = false,
      .depthResolveMin = true,
      .alphaToCoverage = true,
  };
  std::string message;
  auto unavailable =
      checkBenchmarkGpuRequirements(requirements, capabilities, message);
  ASSERT_TRUE(unavailable.hasError());
  EXPECT_EQ(unavailable.error(), BenchmarkExitCode::EnvironmentUnavailable);
  EXPECT_NE(message.find("sample8_depth"), std::string::npos);

  capabilities.sample8Depth = true;
  auto available =
      checkBenchmarkGpuRequirements(requirements, capabilities, message);
  ASSERT_FALSE(available.hasError()) << message;
}

TEST(NuriBenchmarkingTest, ManifestParsesStrictVersionedDDGICoverage) {
  const std::filesystem::path path =
      makeTempPath("benchmark_ddgi_coverage", ".json");
  const std::string manifest = R"json({
    "schemaVersion": 1,
    "id": "ddgi.coverage",
    "suite": "ddgi",
    "settings": {"ddgi": {"coverage": {
      "schemaVersion": 1,
      "mode": "Hybrid",
      "constraintPolicy": "PreserveNearSpacing",
      "sceneBoundsSource": "Authored",
      "authoredBounds": {"min": [-4, -2, -6], "max": [8, 10, 12]},
      "cascadeCount": 4,
      "cascadeProbeCounts": [24, 14, 20],
      "requestedNearSpacing": [1, 1.25, 1.5],
      "spacingRatio": 2.5,
      "requestedCoverageHalfExtents": [60, 35, 55],
      "scenePaddingCells": 3,
      "transitionCells": 2,
      "includeAuthoredVolumes": false
    }}}
  })json";
  writeFile(path, manifest);

  auto loaded = loadBenchmarkCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  EXPECT_EQ(loaded.value().settings.ddgi.preset, DDGIQualityPreset::Balanced);
  EXPECT_EQ(loaded.value().settings.ddgi.requestedCoveragePreset,
            DDGICoveragePreset::Authored);
  EXPECT_EQ(loaded.value().settings.ddgi.coveragePreset,
            DDGICoveragePreset::Custom);
  const DDGICoverageSettings &coverage = loaded.value().settings.ddgi.coverage;
  EXPECT_EQ(coverage.mode, DDGICoverageMode::Hybrid);
  EXPECT_EQ(coverage.constraintPolicy,
            DDGICoverageConstraintPolicy::PreserveNearSpacing);
  EXPECT_EQ(coverage.sceneBoundsSource, DDGISceneBoundsSource::Authored);
  EXPECT_TRUE(coverage.authoredBounds.valid);
  EXPECT_TRUE(coverage.authoredBounds.complete);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.min_.x, -4.0f);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.min_.y, -2.0f);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.min_.z, -6.0f);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.max_.x, 8.0f);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.max_.y, 10.0f);
  EXPECT_FLOAT_EQ(coverage.authoredBounds.bounds.max_.z, 12.0f);
  EXPECT_EQ(coverage.cascadeCount, 4u);
  EXPECT_EQ(coverage.cascadeProbeCounts.x, 24u);
  EXPECT_EQ(coverage.cascadeProbeCounts.y, 14u);
  EXPECT_EQ(coverage.cascadeProbeCounts.z, 20u);
  EXPECT_FLOAT_EQ(coverage.requestedNearSpacing.x, 1.0f);
  EXPECT_FLOAT_EQ(coverage.requestedNearSpacing.y, 1.25f);
  EXPECT_FLOAT_EQ(coverage.requestedNearSpacing.z, 1.5f);
  EXPECT_FLOAT_EQ(coverage.spacingRatio, 2.5f);
  EXPECT_FLOAT_EQ(coverage.requestedCoverageHalfExtents.x, 60.0f);
  EXPECT_FLOAT_EQ(coverage.requestedCoverageHalfExtents.y, 35.0f);
  EXPECT_FLOAT_EQ(coverage.requestedCoverageHalfExtents.z, 55.0f);
  EXPECT_EQ(coverage.scenePaddingCells, 3u);
  EXPECT_EQ(coverage.transitionCells, 2u);
  EXPECT_FALSE(coverage.includeAuthoredVolumes);

  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "ddgi.coverage.default",
              "suite": "ddgi",
              "settings": {"ddgi": {"enabled": true}}
            })json");
  loaded = loadBenchmarkCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  EXPECT_EQ(loaded.value().settings.ddgi.coverage.mode,
            DDGICoverageMode::Manual);
  EXPECT_FALSE(loaded.value().settings.ddgi.coverage.authoredBounds.valid);

  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "ddgi.independent-presets",
              "suite": "ddgi",
              "settings": {"ddgi": {
                "enabled": true,
                "qualityPreset": "High",
                "coveragePreset": "Automatic",
                "gatherVariants": {
                  "opaque": "Candidates",
                  "transmission": "ProbeVisibility",
                  "traceMultiBounce": "Atlas"
                }
              }}
            })json");
  loaded = loadBenchmarkCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  EXPECT_EQ(loaded.value().settings.ddgi.preset, DDGIQualityPreset::High);
  EXPECT_EQ(loaded.value().settings.ddgi.requestedPreset,
            DDGIQualityPreset::High);
  EXPECT_EQ(loaded.value().settings.ddgi.raysPerProbe, 256u);
  EXPECT_EQ(loaded.value().settings.ddgi.coveragePreset,
            DDGICoveragePreset::Automatic);
  EXPECT_EQ(loaded.value().settings.ddgi.requestedCoveragePreset,
            DDGICoveragePreset::Automatic);
  EXPECT_EQ(loaded.value().settings.ddgi.coverage.mode,
            DDGICoverageMode::SceneFit);
  EXPECT_EQ(loaded.value().settings.ddgi.coverage.cascadeCount, 3u);
  EXPECT_EQ(loaded.value().settings.ddgi.opaqueGatherVariant,
            DDGISurfaceGatherVariant::Candidates);
  EXPECT_EQ(loaded.value().settings.ddgi.transmissionGatherVariant,
            DDGISurfaceGatherVariant::ProbeVisibility);
  EXPECT_EQ(loaded.value().settings.ddgi.traceMultiBounceGatherVariant,
            DDGISurfaceGatherVariant::Atlas);

  const std::array invalidManifests{
      replaceFirst(manifest, "\"schemaVersion\": 1,\n      \"mode\"",
                   "\"schemaVersion\": 2,\n      \"mode\""),
      replaceFirst(manifest, "\"includeAuthoredVolumes\": false",
                   "\"includeAuthoredVolumes\": false, \"unexpected\": true"),
      replaceFirst(
          manifest,
          "\"authoredBounds\": {\"min\": [-4, -2, -6], \"max\": [8, 10, 12]}",
          "\"authoredBounds\": {\"min\": [-4, -2, -6]}")};
  for (const std::string &invalidManifest : invalidManifests) {
    writeFile(path, invalidManifest);
    auto invalid = loadBenchmarkCaseManifest(path);
    EXPECT_TRUE(invalid.hasError());
    if (invalid.hasError()) {
      EXPECT_NE(invalid.error().find("settings.ddgi.coverage"),
                std::string::npos);
    }
  }

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriBenchmarkingTest, ManifestRejectsRemovedBackends) {
  const std::filesystem::path path =
      makeTempPath("benchmark_removed_backend", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "backend.removed",
              "suite": "backend",
              "backend": "unsupported"
            })json");
  auto loaded = loadBenchmarkCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("default or nvrhi"), std::string::npos);

  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "backend.requirement_removed",
              "suite": "backend",
              "backend": "nvrhi",
              "requirements": {"backends": ["unsupported"]}
            })json");
  loaded = loadBenchmarkCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("unsupported backend"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriBenchmarkingTest, ManifestLoadsOneSharedTimelineAuthority) {
  const std::filesystem::path source =
      makeTempPath("benchmark_timeline_source", ".json");
  const std::filesystem::path manifest =
      makeTempPath("benchmark_timeline_consumer", ".json");
  writeFile(source,
            R"json({
              "timeline": {
                "cameraPaths": [{
                  "id": "shared_route",
                  "startFrame": 0,
                  "endFrame": 2,
                  "interpolation": "linear",
                  "keyframes": [
                    {"frame": 0, "position": [0, 1, 2], "target": [0, 0, 0]},
                    {"frame": 2, "position": [2, 1, 0], "target": [0, 0, 0]}
                  ]
                }],
                "events": [
                  {"frame": 1, "type": "resetTemporalHistory"},
                  {"frame": 2, "type": "setCamera",
                   "camera": {"position": [3, 2, 1],
                              "direction": [0, 0, -1]},
                   "preserveHistory": false}
                ]
              }
            })json");
  writeFile(manifest, std::string(R"json({
              "schemaVersion": 1,
              "id": "renderer.shared.timeline",
              "suite": "renderer",
              "warmupFrames": 1,
              "measurementFrames": 2,
              "timelineSource": {"pathBase": "repoRoot", "path": ")json") +
                          source.generic_string() + R"json("}
            })json");

  auto loaded = loadBenchmarkCaseManifest(manifest);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  EXPECT_EQ(loaded.value().timeline.source, source.generic_string());
  ASSERT_EQ(loaded.value().timeline.cameraPaths.size(), 1u);
  EXPECT_EQ(loaded.value().timeline.cameraPaths[0].id, "shared_route");
  EXPECT_EQ(loaded.value().timeline.cameraPaths[0].keyframes.size(), 2u);
  ASSERT_EQ(loaded.value().timeline.events.size(), 2u);
  EXPECT_EQ(loaded.value().timeline.events[0].type,
            BenchmarkTimelineEventType::ResetTemporalHistory);
  EXPECT_EQ(loaded.value().timeline.events[1].type,
            BenchmarkTimelineEventType::SetCamera);
  EXPECT_TRUE(loaded.value().timeline.events[1].hasCamera);
  EXPECT_FALSE(loaded.value().timeline.events[1].preserveHistory);
  EXPECT_EQ(loaded.value().timeline.events[1].camera.position,
            glm::vec3(3.0f, 2.0f, 1.0f));

  std::error_code ec;
  std::filesystem::remove(source, ec);
  std::filesystem::remove(manifest, ec);
}

TEST(NuriBenchmarkingTest, CanonicalBistroStressUsesWideRapidUltraRoute) {
  auto loaded = loadBenchmarkCaseManifest(
      defaultBenchmarkCaseRoot() / "stress" /
      "niagara_bistro_full_pipeline_rapid_full_route_720p.json");
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  const BenchmarkCase &benchmarkCase = loaded.value();
  EXPECT_EQ(benchmarkCase.backend, "nvrhi");
  ASSERT_EQ(benchmarkCase.requirements.backends.size(), 1u);
  EXPECT_EQ(benchmarkCase.requirements.backends[0], "nvrhi");
  EXPECT_EQ(benchmarkCase.settings.antiAliasing.mode, AntiAliasingMode::TAA);
  EXPECT_EQ(benchmarkCase.settings.antiAliasing.temporalProvider,
            TemporalReconstructionProvider::Reference);
  EXPECT_EQ(benchmarkCase.settings.antiAliasing.qualityPreset,
            TemporalAAQualityPreset::Ultra);
  EXPECT_EQ(benchmarkCase.settings.ambientOcclusion.mode,
            AmbientOcclusionMode::GTAO);
  EXPECT_EQ(benchmarkCase.settings.ambientOcclusion.preset,
            AmbientOcclusionPreset::Ultra);
  EXPECT_TRUE(benchmarkCase.settings.shadow.enabled);
  EXPECT_EQ(benchmarkCase.settings.shadow.qualityPreset,
            ShadowQualityPreset::Ultra);

  ASSERT_EQ(benchmarkCase.timeline.cameraPaths.size(), 1u);
  const auto &keyframes = benchmarkCase.timeline.cameraPaths[0].keyframes;
  ASSERT_EQ(keyframes.size(), 27u);
  float minX = keyframes[0].position.x;
  float maxX = minX;
  float minZ = keyframes[0].position.z;
  float maxZ = minZ;
  for (const BenchmarkCameraKeyframe &keyframe : keyframes) {
    minX = std::min(minX, keyframe.position.x);
    maxX = std::max(maxX, keyframe.position.x);
    minZ = std::min(minZ, keyframe.position.z);
    maxZ = std::max(maxZ, keyframe.position.z);
  }
  EXPECT_LE(minX, -42.0f);
  EXPECT_GE(maxX, 22.0f);
  EXPECT_LE(minZ, -10.0f);
  EXPECT_GE(maxZ, 31.0f);
  EXPECT_EQ(keyframes[7].frame, 120u);
  EXPECT_EQ(keyframes[8].frame, 124u);
  EXPECT_EQ(keyframes[9].frame, 128u);
  EXPECT_EQ(keyframes[7].position.x, keyframes[8].position.x);
  EXPECT_EQ(keyframes[8].position.x, keyframes[9].position.x);
  EXPECT_NE(keyframes[7].target.x, keyframes[8].target.x);
  EXPECT_NE(keyframes[8].target.x, keyframes[9].target.x);
}

TEST(NuriBenchmarkingTest, ManifestAppliesShadowPresetBeforeExplicitOverrides) {
  const std::filesystem::path path =
      makeTempPath("benchmark_shadow_overrides", ".json");
  const std::string manifest = R"json({
    "schemaVersion": 1,
    "id": "shadow.overrides",
    "suite": "shadow",
    "settings": {"shadow": {
      "qualityPreset": "Ultra",
      "depthFormat": "D32_FLOAT",
      "maxDistance": 73.0,
      "maxDistanceFadeFraction": 0.25,
      "splitLambda": 0.2,
      "cascadeBlendFraction": 0.03,
      "pcfSampleCount": 7,
      "sdsmTemporalBlend": 0.4,
      "enableCascadeCasterCulling": false
    }}
  })json";
  writeFile(path, manifest);

  auto loaded = loadBenchmarkCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  const RenderSettings::ShadowSettings &shadow = loaded.value().settings.shadow;
  EXPECT_EQ(shadow.qualityPreset, ShadowQualityPreset::Ultra);
  EXPECT_EQ(shadow.depthFormat, Format::D32_FLOAT);
  EXPECT_FLOAT_EQ(shadow.maxDistance, 73.0f);
  EXPECT_FLOAT_EQ(shadow.maxDistanceFadeFraction, 0.25f);
  EXPECT_FLOAT_EQ(shadow.splitLambda, 0.2f);
  EXPECT_FLOAT_EQ(shadow.cascadeBlendFraction, 0.03f);
  EXPECT_EQ(shadow.pcfSampleCount, 7u);
  EXPECT_FLOAT_EQ(shadow.sdsmTemporalBlend, 0.4f);
  EXPECT_FALSE(shadow.debug.enableCascadeCasterCulling);

  writeFile(path, replaceFirst(manifest, "D32_FLOAT", "invalid"));
  auto invalid = loadBenchmarkCaseManifest(path);
  EXPECT_TRUE(invalid.hasError());
  EXPECT_NE(invalid.error().find("depthFormat"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriBenchmarkingTest, ManifestRejectsUnsafeIdentifiers) {
  const std::filesystem::path path =
      makeTempPath("benchmark_unsafe_id", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "../outside",
              "suite": "bad"
            })json");

  auto loaded = loadBenchmarkCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("id"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriBenchmarkingTest, ManifestRequiresCompleteExperimentIdentity) {
  const std::filesystem::path path =
      makeTempPath("benchmark_incomplete_experiment", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "aa.incomplete.none",
              "suite": "aa",
              "comparisonGroup": "aa.incomplete"
            })json");
  auto loaded = loadBenchmarkCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("provided together"), std::string::npos);
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriBenchmarkingTest, DiscoveryRejectsDuplicateExperimentVariants) {
  const std::filesystem::path root =
      makeTempPath("benchmark_duplicate_experiment_variant", "");
  std::filesystem::create_directories(root);
  writeFile(
      root / "first.json",
      R"json({"schemaVersion":1,"id":"aa.first.none","suite":"aa","comparisonGroup":"aa.group","variant":"none"})json");
  writeFile(
      root / "second.json",
      R"json({"schemaVersion":1,"id":"aa.second.none","suite":"aa","comparisonGroup":"aa.group","variant":"none"})json");
  auto discovered =
      discoverBenchmarkCases(BenchmarkManifestLoadOptions{.caseRoot = root});
  EXPECT_TRUE(discovered.hasError());
  EXPECT_NE(discovered.error().find("duplicate benchmark experiment variant"),
            std::string::npos);
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST(NuriBenchmarkingTest, ManifestRejectsInvalidExecutionRanges) {
  const std::array<std::string_view, 6u> manifests{
      R"json({"schemaVersion":1,"id":"bad.resolution","suite":"bad","resolution":[0,360]})json",
      R"json({"schemaVersion":1,"id":"bad.delta","suite":"bad","fixedDeltaSeconds":0})json",
      R"json({"schemaVersion":1,"id":"bad.frames","suite":"bad","measurementFrames":0})json",
      R"json({"schemaVersion":1,"id":"bad.workers","suite":"bad","renderGraph":{"workerCount":0}})json",
      R"json({"schemaVersion":1,"id":"bad.camera","suite":"bad","camera":{"nearPlane":10,"farPlane":1}})json",
      R"json({"schemaVersion":1,"id":"bad.threshold","suite":"bad","thresholds":{"failPercent":-1}})json"};

  for (size_t index = 0u; index < manifests.size(); ++index) {
    const std::filesystem::path path = makeTempPath(
        "benchmark_invalid_range_" + std::to_string(index), ".json");
    writeFile(path, manifests[index]);
    auto loaded = loadBenchmarkCaseManifest(path);
    EXPECT_TRUE(loaded.hasError()) << "manifest " << index;
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
}

TEST(NuriBenchmarkingTest, ManifestRejectsDuplicateRequiredMetrics) {
  const std::filesystem::path path =
      makeTempPath("benchmark_duplicate_metrics", ".json");
  writeFile(path,
            R"json({
              "schemaVersion": 1,
              "id": "duplicate.metrics",
              "suite": "test",
              "requiredMetrics": ["cpu.render_submit_ms", "cpu.render_submit_ms"]
            })json");

  auto loaded = loadBenchmarkCaseManifest(path);
  EXPECT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("duplicate"), std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriBenchmarkingTest,
     DryRunCaseWritesStrictV2EnvelopeAndKeepsDetailedV1ReportReadable) {
  const std::filesystem::path artifactDir =
      makeTempPath("benchmark_case_envelope", "");
  BenchmarkCase benchmarkCase{};
  benchmarkCase.id = "contract.case_envelope";
  benchmarkCase.suite = "contract";
  BenchmarkRunOptions options{};
  options.artifactDir = artifactDir;
  options.dryRun = true;
  options.baselineProfileId = "local-nvrhi-visible";
  options.command = "nuri-bench run --case contract.case_envelope --dry-run";

  const BenchmarkRunResult result =
      runBenchmarkCase(std::move(benchmarkCase), options);

  ASSERT_EQ(result.exitCode, BenchmarkExitCode::Success) << result.message;
  EXPECT_EQ(result.reportPath,
            artifactDir / "cases" / "contract.case_envelope.json");
  EXPECT_EQ(result.envelopePath, artifactDir / "run.json");
  auto detailed = readBenchmarkReportFile(result.reportPath);
  ASSERT_FALSE(detailed.hasError()) << detailed.error();
  EXPECT_EQ(detailed.value().schemaVersion, 1u);
  EXPECT_EQ(detailed.value().benchmarkCase.id, "contract.case_envelope");

  std::ifstream file(result.envelopePath, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  const std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  auto envelope = nuri::tools::core::readResultEnvelopeV2(json);
  ASSERT_FALSE(envelope.hasError()) << envelope.error();
  EXPECT_EQ(envelope.value().status,
            nuri::tools::core::ToolOutcome::Investigative);
  EXPECT_EQ(envelope.value().selection.selected, 1u);
  EXPECT_EQ(envelope.value().selection.warned, 1u);
  ASSERT_TRUE(envelope.value().environmentFingerprint.has_value());
  EXPECT_EQ(envelope.value().environmentFingerprint->size(), 71u);
  ASSERT_TRUE(envelope.value().workloadFingerprint.has_value());
  EXPECT_EQ(envelope.value().workloadFingerprint->size(), 71u);
  ASSERT_EQ(envelope.value().children.size(), 1u);
  ASSERT_TRUE(envelope.value().children[0].result.has_value());
  EXPECT_EQ(*envelope.value().children[0].result,
            std::filesystem::path("cases/contract.case_envelope.json"));
  ASSERT_EQ(envelope.value().artifacts.size(), 1u);
  EXPECT_EQ(envelope.value().artifacts[0].path,
            std::filesystem::path("cases/contract.case_envelope.json"));
  EXPECT_EQ(envelope.value().artifacts[0].digest.rfind("sha256:", 0u), 0u);

  auto fromEnvelope = readBenchmarkReportFile(result.envelopePath);
  ASSERT_FALSE(fromEnvelope.hasError()) << fromEnvelope.error();
  EXPECT_EQ(fromEnvelope.value().benchmarkCase.id, "contract.case_envelope");

  file.close();
  std::error_code error;
  std::filesystem::remove_all(artifactDir, error);
}

TEST(NuriBenchmarkingTest, MissingRequiredMetricInvalidatesRun) {
  BenchmarkReport report{};
  report.run.samples = 1u;
  report.run.measurementFrames = 1u;
  report.run.validForComparison = true;
  report.benchmarkCase.requiredMetrics = {"custom.required_ms"};
  report.frames.push_back(BenchmarkFrameRecord{
      .frameIndex = 0u,
      .sampleIndex = 0u,
      .measured = true,
      .measurements = {{"cpu.render_submit_ms", 1.0}},
  });

  computeBenchmarkReportStats(report);

  EXPECT_FALSE(report.run.validForComparison);
  EXPECT_NE(std::find(report.unavailableMetrics.begin(),
                      report.unavailableMetrics.end(), "custom.required_ms"),
            report.unavailableMetrics.end());
}

TEST(NuriBenchmarkingTest, WarmupStabilityUsesProfileOwnedMedianDriftPolicy) {
  BenchmarkReport report{};
  report.run.samples = 2u;
  report.run.warmupFrames = 4u;
  report.run.measurementFrames = 1u;
  report.profile.warmupWindowFrames = 2u;
  report.profile.warmupMaxDriftPercent = 10.0;
  const auto addSample = [&report](uint32_t sampleIndex,
                                   std::initializer_list<double> warmup,
                                   double measured) {
    uint64_t frameIndex = report.frames.size();
    for (const double value : warmup) {
      report.frames.push_back(
          {.frameIndex = frameIndex++,
           .sampleIndex = sampleIndex,
           .measured = false,
           .measurements = {{"cpu.render_submit_ms", value}}});
    }
    report.frames.push_back(
        {.frameIndex = frameIndex,
         .sampleIndex = sampleIndex,
         .measured = true,
         .measurements = {{"cpu.render_submit_ms", measured}}});
  };
  addSample(0u, {10.0, 10.0, 10.5, 10.5}, 10.0);
  addSample(1u, {10.0, 10.0, 12.0, 12.0}, 10.0);

  computeBenchmarkReportStats(report);

  ASSERT_EQ(report.sampleStats.size(), 2u);
  ASSERT_TRUE(report.sampleStats[0].warmupStable.has_value());
  EXPECT_TRUE(*report.sampleStats[0].warmupStable);
  ASSERT_TRUE(report.sampleStats[1].warmupStable.has_value());
  EXPECT_FALSE(*report.sampleStats[1].warmupStable);
  EXPECT_EQ(report.profile.warmupStabilityStatus, "unstable");
}

TEST(NuriBenchmarkingTest,
     IsolatedRepetitionsAggregateIndependentReportsAndPortableArtifacts) {
  const std::filesystem::path artifactDir =
      makeTempPath("benchmark_isolated_repetitions", "");
  const auto profile = makeFixtureBaselineProfile(false);
  BenchmarkCase benchmarkCase{};
  benchmarkCase.id = "test.isolated";
  benchmarkCase.suite = "test";
  benchmarkCase.samples = 1u;
  benchmarkCase.warmupFrames = 1u;
  benchmarkCase.measurementFrames = 2u;
  benchmarkCase.requiredMetrics = {"cpu.render_submit_ms"};

  const BenchmarkRunResult result = runBenchmarkCaseIsolated(
      benchmarkCase, makeIsolatedFixtureOptions(artifactDir, profile));

  ASSERT_EQ(result.exitCode, BenchmarkExitCode::Success) << result.message;
  ASSERT_EQ(result.repetitions.size(), 3u);
  EXPECT_TRUE(std::all_of(result.repetitions.begin(), result.repetitions.end(),
                          [](const BenchmarkRepetitionResult &repetition) {
                            return repetition.completed && !repetition.timedOut;
                          }));
  EXPECT_EQ(result.report.repeatObservations.unit, "isolated-process");
  EXPECT_TRUE(result.report.repeatObservations.independent);
  EXPECT_EQ(result.report.repeatObservations.count, 3u);
  EXPECT_EQ(result.report.sampleStats.size(), 3u);
  EXPECT_EQ(result.report.profile.completedRepetitions, 3u);
  EXPECT_TRUE(result.report.profile.repetitionRequirementSatisfied);
  EXPECT_EQ(result.report.profile.repetitionUnit, "isolated-process");
  EXPECT_EQ(result.report.profile.warmupStabilityStatus, "stable");
  EXPECT_FALSE(result.report.profile.authoritative);
  EXPECT_TRUE(result.report.run.validForComparison);
  ASSERT_EQ(result.report.stats.count("cpu.render_submit_ms"), 1u);
  EXPECT_EQ(result.report.stats.at("cpu.render_submit_ms").count, 3u);
  EXPECT_DOUBLE_EQ(result.report.stats.at("cpu.render_submit_ms").median, 11.0);
  for (const BenchmarkRepetitionResult &repetition : result.repetitions) {
    EXPECT_TRUE(std::filesystem::is_regular_file(repetition.reportPath));
    EXPECT_TRUE(std::filesystem::is_regular_file(repetition.envelopePath));
    EXPECT_TRUE(std::filesystem::is_regular_file(repetition.stdoutLogPath));
    EXPECT_TRUE(std::filesystem::is_regular_file(repetition.stderrLogPath));
    EXPECT_EQ(repetition.workspace.parent_path(),
              std::filesystem::weakly_canonical(artifactDir) / "repetitions");
  }

  std::ifstream envelopeFile(result.envelopePath, std::ios::binary);
  ASSERT_TRUE(envelopeFile.is_open());
  const std::string envelopeJson((std::istreambuf_iterator<char>(envelopeFile)),
                                 std::istreambuf_iterator<char>());
  auto envelope = nuri::tools::core::readResultEnvelopeV2(envelopeJson);
  ASSERT_FALSE(envelope.hasError()) << envelope.error();
  EXPECT_EQ(envelope.value().children.size(), 3u);
  EXPECT_GE(envelope.value().artifacts.size(), 13u);

  std::error_code error;
  std::filesystem::remove_all(artifactDir, error);
}

TEST(NuriBenchmarkingTest,
     AuthoritativeIsolatedRepetitionsRequireMinimumAndStableWarmup) {
  const auto profile = makeFixtureBaselineProfile(true);
  BenchmarkCase benchmarkCase{};
  benchmarkCase.id = "test.too_few";
  benchmarkCase.suite = "test";
  benchmarkCase.samples = 1u;
  benchmarkCase.measurementFrames = 2u;
  benchmarkCase.requiredMetrics = {"cpu.render_submit_ms"};
  const std::filesystem::path tooFewDir =
      makeTempPath("benchmark_too_few_repetitions", "");
  BenchmarkRunOptions tooFew =
      makeIsolatedFixtureOptions(tooFewDir, profile, 2u);

  const BenchmarkRunResult rejected =
      runBenchmarkCaseIsolated(benchmarkCase, tooFew);

  EXPECT_EQ(rejected.exitCode, BenchmarkExitCode::InvalidInput);
  EXPECT_TRUE(rejected.repetitions.empty());
  EXPECT_FALSE(rejected.report.profile.authoritative);
  EXPECT_FALSE(std::filesystem::exists(tooFewDir / "repetitions"));

  benchmarkCase.id = "test.unknown_warmup";
  const std::filesystem::path unknownDir =
      makeTempPath("benchmark_unknown_warmup", "");
  const BenchmarkRunResult unknown = runBenchmarkCaseIsolated(
      benchmarkCase, makeIsolatedFixtureOptions(unknownDir, profile));
  EXPECT_EQ(unknown.exitCode, BenchmarkExitCode::EnvironmentUnavailable);
  EXPECT_EQ(unknown.report.profile.completedRepetitions, 3u);
  EXPECT_EQ(unknown.report.profile.warmupStabilityStatus, "unknown");
  EXPECT_FALSE(unknown.report.profile.authoritative);

  std::error_code error;
  std::filesystem::remove_all(tooFewDir, error);
  std::filesystem::remove_all(unknownDir, error);
}

TEST(NuriBenchmarkingTest,
     AuthoritativeIsolatedRepetitionsClaimAuthorityOnlyWhenAllPolicyMatches) {
  const auto profile = makeFixtureBaselineProfile(true);
  BenchmarkCase benchmarkCase{};
  benchmarkCase.id = "test.authoritative";
  benchmarkCase.suite = "test";
  benchmarkCase.samples = 1u;
  benchmarkCase.warmupFrames = 1u;
  benchmarkCase.measurementFrames = 2u;
  benchmarkCase.requiredMetrics = {"cpu.render_submit_ms"};
  const std::filesystem::path artifactDir =
      makeTempPath("benchmark_authoritative_repetitions", "");

  const BenchmarkRunResult result = runBenchmarkCaseIsolated(
      benchmarkCase, makeIsolatedFixtureOptions(artifactDir, profile));

  ASSERT_EQ(result.exitCode, BenchmarkExitCode::Success) << result.message;
  EXPECT_TRUE(result.report.profile.authoritative);
  EXPECT_TRUE(result.report.profile.authorityBlockers.empty());
  EXPECT_TRUE(result.report.run.validForComparison);

  std::error_code error;
  std::filesystem::remove_all(artifactDir, error);
}

TEST(NuriBenchmarkingTest,
     BenchmarkBaselineDryRunPlanIsDigestBoundAndDoesNotMutate) {
  const std::filesystem::path tempRoot =
      makeTempPath("benchmark_baseline_plan", "");
  const std::filesystem::path runRoot = tempRoot / "run";
  const std::filesystem::path baselineRoot = tempRoot / "baselines";
  std::filesystem::create_directories(runRoot);
  BenchmarkBaselineSource source = writeGovernedBenchmarkSource(runRoot);
  const auto profile = makeInvestigativeBaselineProfile();

  auto first = planBenchmarkBaseline(source, profile, "reviewed evidence",
                                     "test-agent", baselineRoot);
  ASSERT_FALSE(first.hasError()) << first.error();
  auto second = planBenchmarkBaseline(source, profile, "reviewed evidence",
                                      "test-agent", baselineRoot);
  ASSERT_FALSE(second.hasError()) << second.error();

  EXPECT_EQ(first.value().digest, second.value().digest);
  EXPECT_TRUE(first.value().digest.starts_with("sha256:"));
  EXPECT_TRUE(first.value().sourceReportDigest.starts_with("sha256:"));
  EXPECT_TRUE(first.value().acceptedReportDigest.starts_with("sha256:"));
  EXPECT_FALSE(first.value().profileAuthoritative);
  EXPECT_FALSE(first.value().authoritative);
  EXPECT_EQ(first.value().gatePolicy.source, "source-report-initial");
  EXPECT_FALSE(std::filesystem::exists(baselineRoot));
  auto planJson = writeBenchmarkBaselinePlanJson(first.value());
  ASSERT_FALSE(planJson.hasError()) << planJson.error();
  EXPECT_NE(planJson.value().find("nuri.benchmark.baseline_plan"),
            std::string::npos);
  EXPECT_NE(planJson.value().find("\"authoritative\": false"),
            std::string::npos);

  auto wrongConfirmation = acceptBenchmarkBaseline(
      source, profile, "reviewed evidence", "test-agent", "sha256:wrong",
      BenchmarkBaselineAcceptOptions{.baselineRoot = baselineRoot});
  EXPECT_TRUE(wrongConfirmation.hasError());
  EXPECT_FALSE(std::filesystem::exists(baselineRoot));

  std::error_code error;
  std::filesystem::remove_all(tempRoot, error);
}

TEST(NuriBenchmarkingTest,
     BenchmarkBaselineAcceptancePreservesPolicyHistoryAndRollsBack) {
  const std::filesystem::path tempRoot =
      makeTempPath("benchmark_baseline_accept", "");
  const std::filesystem::path runRoot = tempRoot / "run";
  const std::filesystem::path baselineRoot = tempRoot / "baselines";
  std::filesystem::create_directories(runRoot);
  const auto profile = makeInvestigativeBaselineProfile();
  BenchmarkBaselineSource source = writeGovernedBenchmarkSource(runRoot);
  auto firstPlan = planBenchmarkBaseline(source, profile, "initial evidence",
                                         "test-agent", baselineRoot);
  ASSERT_FALSE(firstPlan.hasError()) << firstPlan.error();
  auto accepted = acceptBenchmarkBaseline(
      source, profile, "initial evidence", "test-agent",
      firstPlan.value().digest,
      BenchmarkBaselineAcceptOptions{.baselineRoot = baselineRoot});
  ASSERT_FALSE(accepted.hasError()) << accepted.error();

  const std::filesystem::path acceptedReport =
      baselineRoot / "local-nvrhi-visible" / "case" / "case.default.json";
  const std::filesystem::path approval = baselineRoot / "local-nvrhi-visible" /
                                         "case" / "case.default.approval.json";
  ASSERT_TRUE(std::filesystem::is_regular_file(acceptedReport));
  ASSERT_TRUE(std::filesystem::is_regular_file(approval));
  auto verification =
      verifyBenchmarkBaseline("case.default", "case", profile, baselineRoot);
  ASSERT_FALSE(verification.hasError()) << verification.error();
  EXPECT_TRUE(verification.value().valid);
  EXPECT_FALSE(verification.value().authoritative);
  EXPECT_FALSE(verification.value().historyKey.empty());
  EXPECT_FALSE(verification.value().warnings.empty());

  std::ifstream originalFile(acceptedReport, std::ios::binary);
  const std::string originalBytes(
      (std::istreambuf_iterator<char>(originalFile)),
      std::istreambuf_iterator<char>());
  originalFile.close();
  EXPECT_EQ(originalBytes.find("\"metrics\":"), std::string::npos);

  const BenchmarkThresholds relaxed{
      .failPercent = 999.0,
      .failAbsoluteMs = 999.0,
      .warnPercent = 998.0,
      .warnAbsoluteMs = 998.0,
  };
  source = writeGovernedBenchmarkSource(runRoot, relaxed);
  source.report.frames[0].measurements["cpu.render_submit_ms"] = 12.0;
  computeBenchmarkReportStats(source.report);
  auto sourceWritten =
      writeBenchmarkReportFile(source.report, source.reportPath, true);
  ASSERT_FALSE(sourceWritten.hasError()) << sourceWritten.error();
  auto reloadedSource = loadBenchmarkBaselineSource(runRoot, "case.default");
  ASSERT_FALSE(reloadedSource.hasError()) << reloadedSource.error();
  source = std::move(reloadedSource.value());

  auto replacementPlan = planBenchmarkBaseline(
      source, profile, "replacement evidence", "test-agent", baselineRoot);
  ASSERT_FALSE(replacementPlan.hasError()) << replacementPlan.error();
  EXPECT_EQ(replacementPlan.value().gatePolicy.source, "previous-baseline");
  EXPECT_DOUBLE_EQ(replacementPlan.value().gatePolicy.thresholds.failPercent,
                   firstPlan.value().gatePolicy.thresholds.failPercent);

  auto injectedFailure = acceptBenchmarkBaseline(
      source, profile, "replacement evidence", "test-agent",
      replacementPlan.value().digest,
      BenchmarkBaselineAcceptOptions{
          .baselineRoot = baselineRoot,
          .promotionFault =
              BenchmarkBaselinePromotionFault::AfterBackupRenameForTesting});
  EXPECT_TRUE(injectedFailure.hasError());
  std::ifstream rolledBackFile(acceptedReport, std::ios::binary);
  const std::string rolledBackBytes(
      (std::istreambuf_iterator<char>(rolledBackFile)),
      std::istreambuf_iterator<char>());
  rolledBackFile.close();
  EXPECT_EQ(rolledBackBytes, originalBytes);
  verification =
      verifyBenchmarkBaseline("case.default", "case", profile, baselineRoot);
  ASSERT_FALSE(verification.hasError()) << verification.error();
  EXPECT_TRUE(verification.value().valid);

  accepted = acceptBenchmarkBaseline(
      source, profile, "replacement evidence", "test-agent",
      replacementPlan.value().digest,
      BenchmarkBaselineAcceptOptions{.baselineRoot = baselineRoot});
  ASSERT_FALSE(accepted.hasError()) << accepted.error();
  auto acceptedReplacement = readBenchmarkReportFile(acceptedReport);
  ASSERT_FALSE(acceptedReplacement.hasError()) << acceptedReplacement.error();
  EXPECT_DOUBLE_EQ(
      acceptedReplacement.value().benchmarkCase.thresholds.failPercent,
      firstPlan.value().gatePolicy.thresholds.failPercent);
  EXPECT_FALSE(acceptedReplacement.value().profile.authoritative);
  verification =
      verifyBenchmarkBaseline("case.default", "case", profile, baselineRoot);
  ASSERT_FALSE(verification.hasError()) << verification.error();
  EXPECT_TRUE(verification.value().valid);

  const std::filesystem::path historyDir = baselineRoot /
                                           "local-nvrhi-visible" / "case" /
                                           "history" / "case.default";
  size_t historyFiles = 0u;
  for (const auto &entry : std::filesystem::directory_iterator(historyDir)) {
    historyFiles += entry.is_regular_file() ? 1u : 0u;
  }
  EXPECT_GE(historyFiles, 7u);

  std::ofstream tamper(acceptedReport, std::ios::binary | std::ios::app);
  tamper << "\n";
  tamper.close();
  verification =
      verifyBenchmarkBaseline("case.default", "case", profile, baselineRoot);
  ASSERT_FALSE(verification.hasError()) << verification.error();
  EXPECT_FALSE(verification.value().valid);
  EXPECT_NE(std::find(verification.value().errors.begin(),
                      verification.value().errors.end(),
                      "accepted report digest mismatch"),
            verification.value().errors.end());

  std::error_code error;
  std::filesystem::remove_all(tempRoot, error);
}

TEST(NuriBenchmarkingTest,
     BenchmarkVerifiedBaselineLoaderRejectsTamperedAndLegacyEvidence) {
  const std::filesystem::path tempRoot =
      makeTempPath("benchmark_untrusted_baseline", "");
  const std::filesystem::path runRoot = tempRoot / "run";
  const std::filesystem::path governedRoot = tempRoot / "governed";
  const std::filesystem::path legacyRoot = tempRoot / "legacy";
  std::filesystem::create_directories(runRoot);
  const auto profile = makeInvestigativeBaselineProfile();
  BenchmarkBaselineSource source = writeGovernedBenchmarkSource(runRoot);

  const std::filesystem::path legacyReport =
      legacyRoot / profile.id / "case" / "case.default.json";
  std::filesystem::create_directories(legacyReport.parent_path());
  std::filesystem::copy_file(source.reportPath, legacyReport);
  auto legacy = loadVerifiedBenchmarkBaseline("case.default", "case", profile,
                                              legacyRoot);
  EXPECT_TRUE(legacy.hasError());

  auto plan = planBenchmarkBaseline(source, profile, "reviewed evidence",
                                    "test-agent", governedRoot);
  ASSERT_FALSE(plan.hasError()) << plan.error();
  auto accepted = acceptBenchmarkBaseline(
      source, profile, "reviewed evidence", "test-agent", plan.value().digest,
      BenchmarkBaselineAcceptOptions{.baselineRoot = governedRoot});
  ASSERT_FALSE(accepted.hasError()) << accepted.error();
  const std::filesystem::path acceptedReport =
      governedRoot / profile.id / "case" / "case.default.json";
  std::ofstream tamper(acceptedReport, std::ios::binary | std::ios::app);
  tamper << '\n';
  tamper.close();
  auto tampered = loadVerifiedBenchmarkBaseline("case.default", "case", profile,
                                                governedRoot);
  EXPECT_TRUE(tampered.hasError());

  std::error_code error;
  std::filesystem::remove_all(tempRoot, error);
}

TEST(NuriBenchmarkingTest, BenchmarkBaselineSourceRejectsTraversal) {
  auto source = loadBenchmarkBaselineSource(
      makeTempPath("benchmark_baseline_unsafe", ""), "../outside");
  EXPECT_TRUE(source.hasError());
}

TEST(NuriBenchmarkingTest, ReportWritesReadsAndComputesMeasuredStats) {
  BenchmarkReport report{};
  report.generatedAtUtc = "2026-06-26T00:00:00Z";
  report.command = "nuri-bench run --case smoke.procedural.default";
  report.environment.gpuBackend = "nvrhi";
  report.environment.gpuBackendSource = "manifest";
  report.environment.gpuDeviceName = "Test GPU";
  report.environment.gpuVendorId = 0x10deu;
  report.environment.gpuDeviceId = 0x2684u;
  report.environment.gpuDriverVersion = "test-driver";
  report.environment.requestedPresentMode = "immediate";
  report.environment.resolvedPresentMode = "immediate";
  report.environment.presentModeSource = "manifest";
  report.environment.buildType = "Release";
  report.environment.cmakeToolProfile = "bench";
  report.environment.tracyEnabled = true;
  report.environment.tracyDiagnostic = true;
  report.benchmarkCase.id = "smoke.procedural.default";
  report.benchmarkCase.suite = "smoke";
  report.benchmarkCase.comparisonGroup = "aa.procedural.720p";
  report.benchmarkCase.variant = "reference_taa";
  report.benchmarkCase.description = "Smoke benchmark";
  report.benchmarkCase.resolution = {640u, 360u};
  report.benchmarkCase.presentMode = "immediate";
  report.benchmarkCase.fixedDeltaSeconds = 1.0 / 120.0;
  report.benchmarkCase.warmupFrames = 1u;
  report.benchmarkCase.measurementFrames = 2u;
  report.benchmarkCase.cooldownFrames = 3u;
  report.benchmarkCase.maxDrainFrames = 4u;
  report.benchmarkCase.drainTimeoutMs = 500u;
  report.benchmarkCase.samples = 1u;
  report.benchmarkCase.renderGraph.workerCount = 2u;
  report.benchmarkCase.renderGraph.parallelCompile = true;
  report.benchmarkCase.camera.position = {1.0f, 2.0f, 3.0f};
  report.benchmarkCase.camera.direction = {0.0f, -0.5f, -1.0f};
  report.benchmarkCase.camera.target = {1.0f, 1.0f, 1.0f};
  report.benchmarkCase.camera.hasTarget = true;
  report.benchmarkCase.scene.kind = "procedural";
  report.benchmarkCase.scene.generateMeshlets = true;
  report.benchmarkCase.scene.meshletMaxVertices = 64u;
  report.benchmarkCase.scene.meshletMaxPrimitives = 124u;
  report.benchmarkCase.scene.meshletConeWeight = 0.25f;
  report.benchmarkCase.scene.generator = "nuri.test";
  report.benchmarkCase.scene.seed = 7u;
  report.benchmarkCase.scene.contentHash = "test-scene";
  report.benchmarkCase.timeline.cameraPaths.push_back(
      BenchmarkCameraPath{.id = "test_path",
                          .startFrame = 0u,
                          .endFrame = 2u,
                          .interpolation = "linear",
                          .keyframes = {BenchmarkCameraKeyframe{
                                            .frame = 0u,
                                            .position = {1.0f, 2.0f, 3.0f},
                                            .target = {1.0f, 1.0f, 1.0f},
                                            .hasTarget = true,
                                        },
                                        BenchmarkCameraKeyframe{
                                            .frame = 2u,
                                            .position = {2.0f, 2.0f, 3.0f},
                                            .target = {2.0f, 1.0f, 1.0f},
                                            .hasTarget = true,
                                        }}});
  report.benchmarkCase.timeline.events.push_back(BenchmarkTimelineEvent{
      .frame = 1u,
      .type = BenchmarkTimelineEventType::SetCamera,
      .camera =
          BenchmarkCameraConfig{
              .position = {3.0f, 2.0f, 1.0f},
              .direction = {0.0f, 0.0f, -1.0f},
              .target = {3.0f, 2.0f, 0.0f},
              .hasTarget = true,
          },
      .hasCamera = true,
      .preserveHistory = false,
  });
  report.benchmarkCase.requirements.assets = {"modelsRoot:Test/Test.gltf"};
  report.benchmarkCase.requirements.msaaSamples = 4u;
  report.benchmarkCase.settings.opaque.meshletMode =
      MeshletRenderMode::Required;
  report.benchmarkCase.settings.opaque.enableInstanceAnimation = false;
  report.benchmarkCase.settings.antiAliasing.mode = AntiAliasingMode::TAA;
  report.benchmarkCase.settings.antiAliasing.qualityPreset =
      TemporalAAQualityPreset::Ultra;
  report.benchmarkCase.settings.antiAliasing.debug.spatialPostMsaaCleanup =
      true;
  report.benchmarkCase.settings.ambientOcclusion.mode =
      AmbientOcclusionMode::GTAO;
  report.benchmarkCase.settings.ambientOcclusion.preset =
      AmbientOcclusionPreset::Ultra;
  report.benchmarkCase.settings.ambientOcclusion.temporalAccumulation = false;
  report.benchmarkCase.settings.ambientOcclusion.workingResolutionOverride =
      AmbientOcclusionWorkingResolution::Half;
  report.benchmarkCase.settings.ambientOcclusion.inputModeOverride =
      AmbientOcclusionInputMode::DepthOnlyReconstructedNormal;
  report.benchmarkCase.settings.shadow.qualityPreset =
      ShadowQualityPreset::Ultra;
  report.benchmarkCase.settings.ddgi.enabled = true;
  report.benchmarkCase.settings.ddgi.preset = DDGIQualityPreset::Custom;
  report.benchmarkCase.settings.ddgi.raysPerProbe = 96u;
  report.benchmarkCase.settings.ddgi.maxProbeUpdatesPerFrame = 321u;
  report.benchmarkCase.settings.ddgi.maxRadianceProbeUpdatesPerFrame = 222u;
  report.benchmarkCase.settings.ddgi.maxMaintenanceProbeUpdatesPerFrame = 123u;
  report.benchmarkCase.settings.ddgi.maxRayQueriesPerFrame = 30'000u;
  report.benchmarkCase.settings.ddgi.classification = false;
  report.benchmarkCase.settings.ddgi.debugView = DDGIDebugView::Confidence;
  report.benchmarkCase.settings.ddgi.coverage.mode = DDGICoverageMode::Hybrid;
  report.benchmarkCase.settings.ddgi.coverage.constraintPolicy =
      DDGICoverageConstraintPolicy::PreserveNearSpacing;
  report.benchmarkCase.settings.ddgi.coverage.sceneBoundsSource =
      DDGISceneBoundsSource::Authored;
  report.benchmarkCase.settings.ddgi.coverage.cascadeCount = 4u;
  report.benchmarkCase.settings.ddgi.coverage.cascadeProbeCounts = {24u, 14u,
                                                                    20u};
  report.benchmarkCase.settings.ddgi.coverage.requestedNearSpacing = {
      1.0f, 1.25f, 1.5f};
  report.benchmarkCase.settings.ddgi.coverage.authoredBounds.valid = true;
  report.benchmarkCase.settings.ddgi.coverage.authoredBounds.complete = true;
  report.benchmarkCase.settings.ddgi.coverage.authoredBounds.bounds.min_ = {
      -4.0f, -2.0f, -6.0f};
  report.benchmarkCase.settings.ddgi.coverage.authoredBounds.bounds.max_ = {
      8.0f, 10.0f, 12.0f};
  report.benchmarkCase.requiredMetrics = {"cpu.render_submit_ms"};
  report.profile.id = "local-nvrhi-visible";
  report.profile.profileAuthoritative = false;
  report.profile.authoritative = false;
  report.profile.minimumRepetitions = 3u;
  report.profile.completedRepetitions = 0u;
  report.profile.repetitionRequirementSatisfied = false;
  report.profile.repetitionUnit = "not-collected";
  report.profile.warmupStabilityPolicy = "investigative";
  report.profile.warmupStabilityStatus = "unknown";
  report.profile.warmupWindowFrames = 2u;
  report.profile.warmupMaxDriftPercent = 15.0;
  report.profile.requiredMetrics = {"cpu.render_submit_ms"};
  report.profile.authorityBlockers = {
      "independent repetitions are not collected"};
  report.run.samples = 1u;
  report.run.warmupFrames = 1u;
  report.run.measurementFrames = 2u;
  report.run.cooldownFrames = 3u;
  report.run.maxDrainFrames = 4u;
  report.run.drainTimeoutMs = 500u;
  report.run.fixedDeltaSeconds = 1.0 / 120.0;
  report.run.validForComparison = false;
  report.artifacts.artifactDir = "artifacts/bench/test";
  report.artifacts.caseReports = {"artifacts/bench/test/case.json"};
  report.artifacts.tracyArtifacts = {"artifacts/bench/test/tracy/case.tracy",
                                     "artifacts/bench/test/tracy/case.log"};
  report.artifacts.rgpArtifacts = {"artifacts/bench/test/rgp/case.rgp",
                                   "artifacts/bench/test/rgp/case.log"};
  report.artifacts.renderDocArtifacts = {
      "artifacts/bench/test/renderdoc/case.rdc",
      "artifacts/bench/test/renderdoc/case.chrome.json"};
  report.tracy.available = true;
  report.tracy.tracePath = "artifacts/bench/test/tracy/case.tracy";
  report.tracy.captureLogPath = "artifacts/bench/test/tracy/case.log";
  report.tracy.zonesCsvPath = "artifacts/bench/test/tracy/case.zones.csv";
  report.tracy.selfZonesCsvPath =
      "artifacts/bench/test/tracy/case.zones_self.csv";
  report.tracy.gpuEventsCsvPath =
      "artifacts/bench/test/tracy/case.gpu_events.csv";
  report.tracy.exportLogPath = "artifacts/bench/test/tracy/case.export.log";
  report.tracy.captureCommand = "tracy-capture -o case.tracy";
  report.tracy.zonesExportCommand = "tracy-csvexport case.tracy";
  report.tracy.selfZonesExportCommand = "tracy-csvexport -e case.tracy";
  report.tracy.gpuEventsExportCommand = "tracy-csvexport -g case.tracy";
  report.tracy.gpuEventsExportSupported = true;
  report.tracy.captureFrameCount = 8u;
  report.tracy.captureTimeSpanSeconds = 1.25;
  report.tracy.captureZoneEventCount = 128u;
  report.tracy.gpuZoneEventCount = 6u;
  report.tracy.zones.push_back(BenchmarkTracyZoneStats{
      .name = "nuri::RenderPipeline::buildRenderGraph",
      .sourceFile = "lib/nuri/gfx/pipeline/render_pipeline.cpp",
      .sourceLine = 132u,
      .totalNs = 1'500'000u,
      .totalPercent = 12.5,
      .count = 3u,
      .meanNs = 500'000.0,
      .medianNs = 500'000u,
      .p95Ns = 750'000u,
      .minNs = 250'000u,
      .maxNs = 750'000u,
      .stddevNs = 125'000.0,
  });
  report.tracy.selfZones.push_back(BenchmarkTracyZoneStats{
      .name = "nuri::Renderer::render",
      .sourceFile = "lib/nuri/gfx/renderer.cpp",
      .sourceLine = 88u,
      .totalNs = 900'000u,
      .totalPercent = 7.5,
      .count = 3u,
      .meanNs = 300'000.0,
      .minNs = 200'000u,
      .maxNs = 450'000u,
      .stddevNs = 80'000.0,
  });
  report.tracy.gpuZones.push_back(BenchmarkTracyZoneStats{
      .name = "Opaque Visibility",
      .sourceFile = "lib/nuri/platform/nvrhi_gpu_device.cpp",
      .totalNs = 4'500'000u,
      .totalPercent = 62.5,
      .count = 3u,
      .meanNs = 1'500'000.0,
      .medianNs = 1'450'000u,
      .p95Ns = 1'700'000u,
      .minNs = 1'350'000u,
      .maxNs = 1'700'000u,
      .stddevNs = 180'000.0,
  });
  report.tracy.flameGraph.eventsCsvPath =
      "artifacts/bench/test/tracy/case.events.csv";
  report.tracy.flameGraph.eventsExportCommand = "tracy-csvexport -u case.tracy";
  report.tracy.flameGraph.frameScoped = true;
  report.tracy.flameGraph.eventCount = 4u;
  report.tracy.flameGraph.retainedNodeCount = 4u;
  report.tracy.flameGraph.maxDepth = 3u;
  report.tracy.flameGraph.root = BenchmarkTracyFlameNode{
      .name = "BenchmarkFrame stacks",
      .totalNs = 1'500'000u,
      .children = {BenchmarkTracyFlameNode{
          .name = "main",
          .thread = "main",
          .totalNs = 1'500'000u,
          .children = {BenchmarkTracyFlameNode{
              .name = "BenchmarkFrame",
              .thread = "main",
              .sourceFile = "tools/benchmark/src/benchmark_runner.cpp",
              .sourceLine = 1748u,
              .totalNs = 1'500'000u,
              .selfNs = 600'000u,
              .count = 3u,
              .children = {BenchmarkTracyFlameNode{
                  .name = "nuri::RenderPipeline::buildRenderGraph",
                  .thread = "main",
                  .sourceFile = "lib/nuri/gfx/pipeline/render_pipeline.cpp",
                  .sourceLine = 132u,
                  .totalNs = 900'000u,
                  .selfNs = 900'000u,
                  .count = 3u,
              }},
          }},
      }},
  };
  report.rgp = BenchmarkRgpReport{
      .requested = true,
      .available = true,
      .toolPath = "tools/RadeonDeveloperPanelCLI.exe",
      .tracePath = "artifacts/bench/test/rgp/case.rgp",
      .captureLogPath = "artifacts/bench/test/rgp/case.log",
      .captureCommand = "RadeonDeveloperPanelCLI --rgp-counter-collection",
      .captureFrame = 30u,
      .counterCollectionRequested = true,
      .derivedCounterCount = 9u,
      .captureExitCode = 0,
      .traceSizeBytes = 4096u,
  };
  report.renderDoc = BenchmarkRenderDocReport{
      .requested = true,
      .available = true,
      .captureTriggered = true,
      .apiVersion = "1.7.0",
      .toolPath = "tools/renderdoccmd.exe",
      .capturePath = "artifacts/bench/test/renderdoc/case.rdc",
      .captureLogPath = "artifacts/bench/test/renderdoc/case.log",
      .chromeTracePath = "artifacts/bench/test/renderdoc/case.chrome.json",
      .conversionLogPath = "artifacts/bench/test/renderdoc/case-convert.log",
      .thumbnailPath = "artifacts/bench/test/renderdoc/case.png",
      .captureCommand = "renderdoccmd capture nuri-bench.exe",
      .captureFrame = 30u,
      .launcherExitCode = 0,
      .conversionExitCode = 0,
      .captureSizeBytes = 8192u,
      .chromeTraceSizeBytes = 2048u,
      .chromeEventCount = 128u,
      .drawCallCount = 12u,
      .dispatchCallCount = 6u,
      .barrierCallCount = 10u,
      .copyCallCount = 2u,
  };
  report.timingDrain.drainComplete = false;
  report.timingDrain.drainFrames = 4u;
  report.timingDrain.drainTimeoutMs = 500u;
  report.timingDrain.missingGpuTimingFrames = 1u;
  report.timingDrain.scopeContainmentViolations = 3u;
  report.timingDrain.droppedGpuTimingReports = 2u;
  report.unavailableMetrics = {"gpu.scopes_sum_ms"};
  report.warnings = {"timing incomplete"};
  report.frames.push_back(BenchmarkFrameRecord{
      .frameIndex = 0u,
      .sampleIndex = 0u,
      .measured = true,
      .measurements = {{"cpu.render_submit_ms", 1.0}},
  });
  report.frames.push_back(BenchmarkFrameRecord{
      .frameIndex = 1u,
      .sampleIndex = 0u,
      .measured = true,
      .measurements = {{"cpu.render_submit_ms", 3.0}},
  });
  report.frames[0].metrics.opaque.totalInstances = 7u;
  report.frames[0].metrics.opaque.visibleInstances = 5u;
  report.frames[0].metrics.shadow.totalDraws = 3u;
  computeBenchmarkReportStats(report);
  ASSERT_EQ(report.stats.count("cpu.render_submit_ms"), 1u);
  EXPECT_DOUBLE_EQ(report.stats["cpu.render_submit_ms"].median, 2.0);
  ASSERT_EQ(report.sampleStats.size(), 1u);
  EXPECT_FALSE(report.sampleStats[0].warmupStable.has_value());
  report.sampleStats[0].warmupStable = false;
  report.sampleStats[0].warnings = {"warmup unstable"};

  const std::filesystem::path path = makeTempPath("benchmark_report", ".json");
  auto written = writeBenchmarkReportFile(report, path, true);
  ASSERT_FALSE(written.hasError()) << written.error();

  std::ifstream reportFile(path, std::ios::binary);
  ASSERT_TRUE(reportFile.is_open());
  const std::string reportJson((std::istreambuf_iterator<char>(reportFile)),
                               std::istreambuf_iterator<char>());
  EXPECT_NE(reportJson.find("\"settingsSignature\""), std::string::npos);
  EXPECT_NE(reportJson.find("\"configSignature\""), std::string::npos);
  EXPECT_NE(reportJson.find("\"qualityPreset\": \"Ultra\""), std::string::npos);
  EXPECT_NE(reportJson.find("\"comparisonGroup\": \"aa.procedural.720p\""),
            std::string::npos);
  EXPECT_NE(reportJson.find("\"variant\": \"reference_taa\""),
            std::string::npos);
  EXPECT_NE(reportJson.find("\"msaaSamples\": 4"), std::string::npos);
  EXPECT_NE(reportJson.find("\"meshletMode\": \"Required\""),
            std::string::npos);
  EXPECT_NE(reportJson.find("\"ddgi\""), std::string::npos);
  EXPECT_NE(reportJson.find("\"mode\": \"Hybrid\""), std::string::npos);
  EXPECT_NE(reportJson.find("\"cameraPaths\""), std::string::npos);
  EXPECT_NE(reportJson.find("\"events\""), std::string::npos);

  auto loaded = readBenchmarkReportFile(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  EXPECT_EQ(loaded.value().kind, "nuri.benchmark.report");
  EXPECT_EQ(loaded.value().benchmarkCase.id, "smoke.procedural.default");
  EXPECT_EQ(loaded.value().benchmarkCase.comparisonGroup, "aa.procedural.720p");
  EXPECT_EQ(loaded.value().benchmarkCase.variant, "reference_taa");
  EXPECT_EQ(loaded.value().benchmarkCase.description, "Smoke benchmark");
  EXPECT_EQ(loaded.value().benchmarkCase.presentMode, "immediate");
  EXPECT_EQ(loaded.value().benchmarkCase.renderGraph.workerCount, 2u);
  EXPECT_TRUE(loaded.value().benchmarkCase.renderGraph.parallelCompile);
  EXPECT_TRUE(loaded.value().benchmarkCase.camera.hasTarget);
  EXPECT_EQ(loaded.value().benchmarkCase.scene.generator, "nuri.test");
  EXPECT_TRUE(loaded.value().benchmarkCase.scene.generateMeshlets);
  EXPECT_EQ(loaded.value().benchmarkCase.scene.meshletMaxPrimitives, 124u);
  EXPECT_EQ(loaded.value().benchmarkCase.scene.contentHash, "test-scene");
  ASSERT_EQ(loaded.value().benchmarkCase.timeline.cameraPaths.size(), 1u);
  EXPECT_EQ(loaded.value().benchmarkCase.timeline.cameraPaths[0].id,
            "test_path");
  ASSERT_EQ(
      loaded.value().benchmarkCase.timeline.cameraPaths[0].keyframes.size(),
      2u);
  ASSERT_EQ(loaded.value().benchmarkCase.timeline.events.size(), 1u);
  EXPECT_EQ(loaded.value().benchmarkCase.timeline.events[0].type,
            BenchmarkTimelineEventType::SetCamera);
  EXPECT_TRUE(loaded.value().benchmarkCase.timeline.events[0].hasCamera);
  EXPECT_FALSE(loaded.value().benchmarkCase.timeline.events[0].preserveHistory);
  EXPECT_EQ(loaded.value().benchmarkCase.timeline.events[0].camera.position,
            glm::vec3(3.0f, 2.0f, 1.0f));
  ASSERT_EQ(loaded.value().benchmarkCase.requirements.assets.size(), 1u);
  EXPECT_EQ(loaded.value().benchmarkCase.requirements.assets[0],
            "modelsRoot:Test/Test.gltf");
  ASSERT_TRUE(
      loaded.value().benchmarkCase.requirements.msaaSamples.has_value());
  EXPECT_EQ(*loaded.value().benchmarkCase.requirements.msaaSamples, 4u);
  EXPECT_FALSE(
      loaded.value()
          .benchmarkCase.settings.antiAliasing.debug.spatialPostMsaaCleanup);
  EXPECT_FALSE(
      loaded.value().benchmarkCase.settings.antiAliasing.postAA.enabled);
  const RenderSettings::DDGISettings &loadedDDGI =
      loaded.value().benchmarkCase.settings.ddgi;
  EXPECT_TRUE(loadedDDGI.enabled);
  EXPECT_EQ(loadedDDGI.preset, DDGIQualityPreset::Custom);
  EXPECT_EQ(loadedDDGI.raysPerProbe, 96u);
  EXPECT_EQ(loadedDDGI.maxProbeUpdatesPerFrame, 321u);
  EXPECT_EQ(loadedDDGI.maxRadianceProbeUpdatesPerFrame, 222u);
  EXPECT_EQ(loadedDDGI.maxMaintenanceProbeUpdatesPerFrame, 123u);
  EXPECT_EQ(loadedDDGI.maxRayQueriesPerFrame, 30'000u);
  EXPECT_FALSE(loadedDDGI.classification);
  EXPECT_EQ(loadedDDGI.debugView, DDGIDebugView::Confidence);
  EXPECT_EQ(loadedDDGI.coverage.mode, DDGICoverageMode::Hybrid);
  EXPECT_EQ(loadedDDGI.coverage.constraintPolicy,
            DDGICoverageConstraintPolicy::PreserveNearSpacing);
  EXPECT_EQ(loadedDDGI.coverage.sceneBoundsSource,
            DDGISceneBoundsSource::Authored);
  EXPECT_TRUE(loadedDDGI.coverage.authoredBounds.valid);
  EXPECT_EQ(loadedDDGI.coverage.authoredBounds.bounds.max_.z, 12.0f);
  EXPECT_NE(loaded.value().benchmarkCase.settingsSignature.find("aa.quality=3"),
            std::string::npos);
  EXPECT_NE(loaded.value().benchmarkCase.settingsSignature.find(
                "opaque.meshletMode=2"),
            std::string::npos);
  EXPECT_NE(loaded.value().benchmarkCase.settingsSignature.find("ao.preset=3"),
            std::string::npos);
  EXPECT_NE(loaded.value().benchmarkCase.settingsSignature.find(
                "ddgi.maxRayQueriesPerFrame=30000"),
            std::string::npos);
  EXPECT_NE(loaded.value().benchmarkCase.settingsSignature.find(
                "ddgi.coverage.mode=3"),
            std::string::npos);
  EXPECT_NE(loaded.value().benchmarkCase.configSignature.find(
                "renderGraph.workerCount=2"),
            std::string::npos);
  EXPECT_NE(loaded.value().benchmarkCase.configSignature.find(
                "scene.generateMeshlets=1"),
            std::string::npos);
  EXPECT_NE(loaded.value().benchmarkCase.configSignature.find(
                "timeline.cameraPath.0.id=test_path"),
            std::string::npos);
  EXPECT_NE(loaded.value().benchmarkCase.configSignature.find(
                "timeline.event.0.type=setCamera"),
            std::string::npos);
  ASSERT_EQ(loaded.value().stats.count("cpu.render_submit_ms"), 1u);
  EXPECT_DOUBLE_EQ(loaded.value().stats["cpu.render_submit_ms"].median, 2.0);
  EXPECT_EQ(loaded.value().profile.id, "local-nvrhi-visible");
  EXPECT_FALSE(loaded.value().profile.profileAuthoritative);
  EXPECT_FALSE(loaded.value().profile.authoritative);
  EXPECT_EQ(loaded.value().profile.minimumRepetitions, 3u);
  EXPECT_EQ(loaded.value().profile.completedRepetitions, 0u);
  EXPECT_FALSE(loaded.value().profile.repetitionRequirementSatisfied);
  EXPECT_EQ(loaded.value().profile.repetitionUnit, "not-collected");
  EXPECT_EQ(loaded.value().profile.warmupStabilityStatus, "unknown");
  ASSERT_EQ(loaded.value().profile.requiredMetrics.size(), 1u);
  ASSERT_EQ(loaded.value().profile.authorityBlockers.size(), 1u);
  EXPECT_EQ(loaded.value().repeatObservations.count, 1u);
  EXPECT_FALSE(loaded.value().repeatObservations.independent);
  EXPECT_EQ(loaded.value().environment.gpuBackendSource, "manifest");
  EXPECT_EQ(loaded.value().environment.gpuDeviceName, "Test GPU");
  EXPECT_EQ(loaded.value().environment.gpuVendorId, 0x10deu);
  EXPECT_EQ(loaded.value().environment.gpuDeviceId, 0x2684u);
  EXPECT_EQ(loaded.value().environment.gpuDriverVersion, "test-driver");
  EXPECT_TRUE(loaded.value().environment.tracyEnabled);
  EXPECT_TRUE(loaded.value().environment.tracyDiagnostic);
  EXPECT_FALSE(loaded.value().run.validForComparison);
  EXPECT_EQ(loaded.value().run.warmupFrames, 1u);
  EXPECT_EQ(loaded.value().run.cooldownFrames, 3u);
  EXPECT_EQ(loaded.value().run.maxDrainFrames, 4u);
  EXPECT_EQ(loaded.value().run.drainTimeoutMs, 500u);
  EXPECT_DOUBLE_EQ(loaded.value().run.fixedDeltaSeconds, 1.0 / 120.0);
  EXPECT_EQ(loaded.value().artifacts.artifactDir,
            std::filesystem::path("artifacts/bench/test"));
  ASSERT_EQ(loaded.value().artifacts.caseReports.size(), 1u);
  EXPECT_EQ(loaded.value().artifacts.caseReports[0],
            std::filesystem::path("artifacts/bench/test/case.json"));
  ASSERT_EQ(loaded.value().artifacts.tracyArtifacts.size(), 2u);
  ASSERT_EQ(loaded.value().artifacts.rgpArtifacts.size(), 2u);
  ASSERT_EQ(loaded.value().artifacts.renderDocArtifacts.size(), 2u);
  EXPECT_TRUE(loaded.value().tracy.available);
  EXPECT_EQ(loaded.value().tracy.tracePath,
            std::filesystem::path("artifacts/bench/test/tracy/case.tracy"));
  EXPECT_EQ(loaded.value().tracy.captureLogPath,
            std::filesystem::path("artifacts/bench/test/tracy/case.log"));
  EXPECT_EQ(loaded.value().tracy.zonesCsvPath,
            std::filesystem::path("artifacts/bench/test/tracy/case.zones.csv"));
  EXPECT_EQ(
      loaded.value().tracy.selfZonesCsvPath,
      std::filesystem::path("artifacts/bench/test/tracy/case.zones_self.csv"));
  EXPECT_EQ(
      loaded.value().tracy.gpuEventsCsvPath,
      std::filesystem::path("artifacts/bench/test/tracy/case.gpu_events.csv"));
  EXPECT_EQ(
      loaded.value().tracy.exportLogPath,
      std::filesystem::path("artifacts/bench/test/tracy/case.export.log"));
  EXPECT_EQ(loaded.value().tracy.captureCommand, "tracy-capture -o case.tracy");
  EXPECT_EQ(loaded.value().tracy.zonesExportCommand,
            "tracy-csvexport case.tracy");
  EXPECT_EQ(loaded.value().tracy.selfZonesExportCommand,
            "tracy-csvexport -e case.tracy");
  EXPECT_EQ(loaded.value().tracy.gpuEventsExportCommand,
            "tracy-csvexport -g case.tracy");
  EXPECT_TRUE(loaded.value().tracy.gpuEventsExportSupported);
  EXPECT_EQ(loaded.value().tracy.captureFrameCount, 8u);
  EXPECT_DOUBLE_EQ(loaded.value().tracy.captureTimeSpanSeconds, 1.25);
  EXPECT_EQ(loaded.value().tracy.captureZoneEventCount, 128u);
  EXPECT_EQ(loaded.value().tracy.gpuZoneEventCount, 6u);
  ASSERT_EQ(loaded.value().tracy.zones.size(), 1u);
  EXPECT_EQ(loaded.value().tracy.zones[0].name,
            "nuri::RenderPipeline::buildRenderGraph");
  EXPECT_EQ(loaded.value().tracy.zones[0].sourceFile,
            std::filesystem::path("lib/nuri/gfx/pipeline/render_pipeline.cpp"));
  EXPECT_EQ(loaded.value().tracy.zones[0].sourceLine, 132u);
  EXPECT_EQ(loaded.value().tracy.zones[0].totalNs, 1'500'000u);
  EXPECT_DOUBLE_EQ(loaded.value().tracy.zones[0].totalPercent, 12.5);
  EXPECT_EQ(loaded.value().tracy.zones[0].medianNs, 500'000u);
  EXPECT_EQ(loaded.value().tracy.zones[0].p95Ns, 750'000u);
  ASSERT_EQ(loaded.value().tracy.selfZones.size(), 1u);
  EXPECT_EQ(loaded.value().tracy.selfZones[0].name, "nuri::Renderer::render");
  ASSERT_EQ(loaded.value().tracy.gpuZones.size(), 1u);
  EXPECT_EQ(loaded.value().tracy.gpuZones[0].name, "Opaque Visibility");
  EXPECT_EQ(loaded.value().tracy.gpuZones[0].medianNs, 1'450'000u);
  EXPECT_EQ(loaded.value().tracy.gpuZones[0].p95Ns, 1'700'000u);
  EXPECT_EQ(
      loaded.value().tracy.flameGraph.eventsCsvPath,
      std::filesystem::path("artifacts/bench/test/tracy/case.events.csv"));
  EXPECT_EQ(loaded.value().tracy.flameGraph.eventsExportCommand,
            "tracy-csvexport -u case.tracy");
  EXPECT_TRUE(loaded.value().tracy.flameGraph.frameScoped);
  EXPECT_EQ(loaded.value().tracy.flameGraph.eventCount, 4u);
  EXPECT_EQ(loaded.value().tracy.flameGraph.retainedNodeCount, 4u);
  EXPECT_EQ(loaded.value().tracy.flameGraph.maxDepth, 3u);
  ASSERT_EQ(loaded.value().tracy.flameGraph.root.children.size(), 1u);
  EXPECT_EQ(loaded.value().tracy.flameGraph.root.children[0].name, "main");
  ASSERT_EQ(loaded.value().tracy.flameGraph.root.children[0].children.size(),
            1u);
  const BenchmarkTracyFlameNode &frameNode =
      loaded.value().tracy.flameGraph.root.children[0].children[0];
  EXPECT_EQ(frameNode.name, "BenchmarkFrame");
  EXPECT_EQ(frameNode.selfNs, 600'000u);
  ASSERT_EQ(frameNode.children.size(), 1u);
  EXPECT_EQ(frameNode.children[0].name,
            "nuri::RenderPipeline::buildRenderGraph");
  EXPECT_TRUE(loaded.value().rgp.requested);
  EXPECT_TRUE(loaded.value().rgp.available);
  EXPECT_EQ(loaded.value().rgp.purpose, "shader-diagnostic-only");
  EXPECT_EQ(loaded.value().rgp.tracePath,
            std::filesystem::path("artifacts/bench/test/rgp/case.rgp"));
  EXPECT_EQ(loaded.value().rgp.captureFrame, 30u);
  EXPECT_TRUE(loaded.value().rgp.counterCollectionRequested);
  EXPECT_EQ(loaded.value().rgp.derivedCounterCount, 9u);
  EXPECT_EQ(loaded.value().rgp.captureExitCode, 0);
  EXPECT_EQ(loaded.value().rgp.traceSizeBytes, 4096u);
  EXPECT_TRUE(loaded.value().renderDoc.requested);
  EXPECT_TRUE(loaded.value().renderDoc.available);
  EXPECT_TRUE(loaded.value().renderDoc.captureTriggered);
  EXPECT_EQ(loaded.value().renderDoc.purpose, "frame-forensics-only");
  EXPECT_EQ(loaded.value().renderDoc.apiVersion, "1.7.0");
  EXPECT_EQ(loaded.value().renderDoc.captureFrame, 30u);
  EXPECT_EQ(loaded.value().renderDoc.launcherExitCode, 0);
  EXPECT_EQ(loaded.value().renderDoc.conversionExitCode, 0);
  EXPECT_EQ(loaded.value().renderDoc.captureSizeBytes, 8192u);
  EXPECT_EQ(loaded.value().renderDoc.chromeTraceSizeBytes, 2048u);
  EXPECT_EQ(loaded.value().renderDoc.chromeEventCount, 128u);
  EXPECT_EQ(loaded.value().renderDoc.drawCallCount, 12u);
  EXPECT_EQ(loaded.value().renderDoc.dispatchCallCount, 6u);
  EXPECT_EQ(loaded.value().renderDoc.barrierCallCount, 10u);
  EXPECT_EQ(loaded.value().renderDoc.copyCallCount, 2u);
  EXPECT_EQ(loaded.value().timingDrain.drainComplete, false);
  EXPECT_EQ(loaded.value().timingDrain.drainFrames, 4u);
  EXPECT_EQ(loaded.value().timingDrain.missingGpuTimingFrames, 1u);
  EXPECT_EQ(loaded.value().timingDrain.scopeContainmentViolations, 3u);
  EXPECT_EQ(loaded.value().timingDrain.droppedGpuTimingReports, 2u);
  ASSERT_EQ(loaded.value().unavailableMetrics.size(), 1u);
  EXPECT_EQ(loaded.value().unavailableMetrics[0], "gpu.scopes_sum_ms");
  ASSERT_EQ(loaded.value().warnings.size(), 1u);
  EXPECT_EQ(loaded.value().warnings[0], "timing incomplete");
  ASSERT_EQ(loaded.value().frames.size(), 2u);
  EXPECT_EQ(loaded.value().frames[0].frameIndex, 0u);
  EXPECT_TRUE(loaded.value().frames[0].measured);
  ASSERT_EQ(loaded.value().frames[0].measurements.count("cpu.render_submit_ms"),
            1u);
  EXPECT_DOUBLE_EQ(
      loaded.value().frames[0].measurements["cpu.render_submit_ms"], 1.0);
  EXPECT_EQ(loaded.value().frames[0].metrics.opaque.totalInstances, 7u);
  EXPECT_EQ(loaded.value().frames[0].metrics.opaque.visibleInstances, 5u);
  EXPECT_EQ(loaded.value().frames[0].metrics.shadow.totalDraws, 3u);
  ASSERT_EQ(loaded.value().sampleStats.size(), 1u);
  EXPECT_EQ(loaded.value().sampleStats[0].sampleIndex, 0u);
  EXPECT_EQ(loaded.value().sampleStats[0].measuredFrameCount, 2u);
  ASSERT_TRUE(loaded.value().sampleStats[0].warmupStable.has_value());
  EXPECT_FALSE(*loaded.value().sampleStats[0].warmupStable);
  ASSERT_EQ(loaded.value().sampleStats[0].warnings.size(), 1u);
  EXPECT_EQ(loaded.value().sampleStats[0].warnings[0], "warmup unstable");
  ASSERT_EQ(loaded.value().sampleStats[0].stats.count("cpu.render_submit_ms"),
            1u);
  EXPECT_DOUBLE_EQ(
      loaded.value().sampleStats[0].stats["cpu.render_submit_ms"].median, 2.0);
  EXPECT_EQ(loaded.value().benchmarkCase.settings.opaque.meshletMode,
            MeshletRenderMode::Required);
  EXPECT_FALSE(
      loaded.value().benchmarkCase.settings.opaque.enableInstanceAnimation);
  EXPECT_EQ(loaded.value().benchmarkCase.settings.antiAliasing.mode,
            AntiAliasingMode::TAA);
  EXPECT_EQ(loaded.value().benchmarkCase.settings.ambientOcclusion.preset,
            AmbientOcclusionPreset::Ultra);
  EXPECT_FALSE(
      loaded.value()
          .benchmarkCase.settings.ambientOcclusion.temporalAccumulation);
  EXPECT_EQ(
      loaded.value()
          .benchmarkCase.settings.ambientOcclusion.workingResolutionOverride,
      AmbientOcclusionWorkingResolution::Half);
  EXPECT_EQ(
      loaded.value().benchmarkCase.settings.ambientOcclusion.inputModeOverride,
      AmbientOcclusionInputMode::DepthOnlyReconstructedNormal);
  EXPECT_EQ(loaded.value().benchmarkCase.settings.shadow.qualityPreset,
            ShadowQualityPreset::Ultra);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST(NuriBenchmarkingTest, DetailedReportReaderRejectsAdversarialJson) {
  BenchmarkReport report = makeComparableReport(1.0, 1.1, 0.5, 0.6);
  report.generatedAtUtc = "2026-07-10T00:00:00Z";
  report.artifacts.artifactDir = "artifacts/bench/contract";
  computeBenchmarkReportStats(report);
  report.run.validForComparison = true;
  auto json = writeBenchmarkReportJson(report, true);
  ASSERT_FALSE(json.hasError()) << json.error();
  const std::vector<std::string> invalid{
      replaceFirst(json.value(), "\"profile\": {",
                   "\"profile\": {\"surprise\": 1,"),
      replaceFirst(json.value(), "\"validForComparison\": true",
                   "\"validForComparison\": \"true\""),
      replaceFirst(
          json.value(), "\"validForComparison\": true",
          "\"validForComparison\": true, \"validForComparison\": true"),
      replaceFirst(json.value(), "artifacts/bench/contract",
                   "artifacts/../escape"),
      json.value().substr(0u, json.value().size() / 2u)};
  for (size_t index = 0u; index < invalid.size(); ++index) {
    const auto path = makeTempPath(
        "benchmark_adversarial_report_" + std::to_string(index), ".json");
    writeFile(path, invalid[index]);
    EXPECT_TRUE(readBenchmarkReportFile(path).hasError()) << index;
    std::filesystem::remove(path);
  }
}

TEST(NuriBenchmarkingTest, CompareReportsPassWarnFailAndInvalidPreconditions) {
  const BenchmarkReport baseline = makeComparableReport(10.0, 12.0, 8.0, 9.0);

  BenchmarkComparisonReport pass = compareBenchmarkReports(
      makeComparableReport(10.1, 12.1, 8.1, 9.1), baseline);
  EXPECT_TRUE(pass.valid);
  EXPECT_FALSE(pass.regression);

  BenchmarkComparisonReport warn = compareBenchmarkReports(
      makeComparableReport(10.7, 12.8, 8.1, 9.1), baseline);
  EXPECT_TRUE(warn.valid);
  EXPECT_FALSE(warn.regression);
  bool sawWarn = false;
  for (const BenchmarkMetricComparison &metric : warn.metrics) {
    sawWarn = sawWarn || metric.status == "warn";
  }
  EXPECT_TRUE(sawWarn);

  BenchmarkComparisonReport fail = compareBenchmarkReports(
      makeComparableReport(11.2, 13.4, 8.1, 9.1), baseline);
  EXPECT_TRUE(fail.valid);
  EXPECT_TRUE(fail.regression);

  BenchmarkReport missingRequired = makeComparableReport(10.1, 12.1, 8.1, 9.1);
  missingRequired.stats.erase("gpu.scopes_sum_ms");
  BenchmarkComparisonReport missing =
      compareBenchmarkReports(missingRequired, baseline);
  EXPECT_FALSE(missing.valid);
  EXPECT_FALSE(missing.errors.empty());

  BenchmarkReport mismatch = makeComparableReport(10.1, 12.1, 8.1, 9.1);
  mismatch.environment.buildType = "Debug";
  BenchmarkComparisonReport invalid =
      compareBenchmarkReports(mismatch, baseline);
  EXPECT_FALSE(invalid.valid);
  EXPECT_FALSE(invalid.errors.empty());

  BenchmarkReport runMismatch = makeComparableReport(10.1, 12.1, 8.1, 9.1);
  runMismatch.run.measurementFrames = 4u;
  invalid = compareBenchmarkReports(runMismatch, baseline);
  EXPECT_FALSE(invalid.valid);
  EXPECT_FALSE(invalid.errors.empty());

  BenchmarkReport settingsBaseline = makeComparableReport(10.0, 12.0, 8.0, 9.0);
  settingsBaseline.benchmarkCase.settingsSignature = "settings.v1|aa.quality=3";
  BenchmarkReport settingsMismatch = makeComparableReport(10.1, 12.1, 8.1, 9.1);
  settingsMismatch.benchmarkCase.settingsSignature = "settings.v1|aa.quality=2";
  invalid = compareBenchmarkReports(settingsMismatch, settingsBaseline);
  EXPECT_FALSE(invalid.valid);
  EXPECT_FALSE(invalid.errors.empty());

  BenchmarkReport presentMismatch = makeComparableReport(10.1, 12.1, 8.1, 9.1);
  presentMismatch.environment.resolvedPresentMode = "fifo";
  invalid = compareBenchmarkReports(presentMismatch, baseline);
  EXPECT_FALSE(invalid.valid);
  EXPECT_FALSE(invalid.errors.empty());

  BenchmarkReport adapterMismatch = makeComparableReport(10.1, 12.1, 8.1, 9.1);
  adapterMismatch.environment.gpuDriverVersion = "other-driver";
  invalid = compareBenchmarkReports(adapterMismatch, baseline);
  EXPECT_FALSE(invalid.valid);
  EXPECT_NE(std::find(invalid.errors.begin(), invalid.errors.end(),
                      "GPU adapter/driver mismatch"),
            invalid.errors.end());

  BenchmarkReport dirty = makeComparableReport(10.1, 12.1, 8.1, 9.1);
  dirty.environment.dirty = true;
  invalid = compareBenchmarkReports(dirty, baseline);
  EXPECT_FALSE(invalid.valid);
  EXPECT_NE(std::find(invalid.errors.begin(), invalid.errors.end(),
                      "dirty source trees are not valid comparison evidence"),
            invalid.errors.end());
}

TEST(NuriBenchmarkingTest, DiagnosticReportsCannotBeComparedEvenWhenForced) {
  const BenchmarkReport baseline = makeComparableReport(10.0, 12.0, 8.0, 9.0);
  for (const DiagnosticReportCase &testCase : kDiagnosticReportCases) {
    SCOPED_TRACE(testCase.label);
    BenchmarkReport diagnostic = makeComparableReport(10.1, 12.1, 8.1, 9.1);
    testCase.mark(diagnostic);
    diagnostic.run.validForComparison = false;

    const BenchmarkComparisonReport comparison = compareBenchmarkReports(
        diagnostic, baseline, BenchmarkCompareOptions{.force = true});

    EXPECT_FALSE(comparison.valid);
    EXPECT_TRUE(comparison.metrics.empty());
    ASSERT_EQ(comparison.errors.size(), 1u);
    EXPECT_NE(comparison.errors.front().find(testCase.label),
              std::string::npos);
  }
}

TEST(NuriBenchmarkingTest, DiagnosticReportsCannotBecomeBenchmarkBaselines) {
  for (const DiagnosticReportCase &testCase : kDiagnosticReportCases) {
    SCOPED_TRACE(testCase.label);
    const std::filesystem::path tempRoot =
        makeTempPath("benchmark_diagnostic_baseline_rejection", "");
    const std::filesystem::path runRoot = tempRoot / "run";
    std::filesystem::create_directories(runRoot);
    BenchmarkBaselineSource source = writeGovernedBenchmarkSource(runRoot);
    testCase.mark(source.report);
    ASSERT_FALSE(
        writeBenchmarkReportFile(source.report, source.reportPath, true)
            .hasError());

    const auto plan = planBenchmarkBaseline(
        source, makeInvestigativeBaselineProfile(), "diagnostic evidence",
        "test-agent", tempRoot / "baselines");

    ASSERT_TRUE(plan.hasError());
    EXPECT_NE(plan.error().find(testCase.label), std::string::npos);
    std::error_code error;
    std::filesystem::remove_all(tempRoot, error);
  }
}

TEST(NuriBenchmarkingTest, ComparisonUsesBaselineOwnedThresholds) {
  BenchmarkReport baseline = makeComparableReport(10.0, 12.0, 8.0, 9.0);
  baseline.benchmarkCase.thresholds = BenchmarkThresholds{
      .failPercent = 5.0,
      .failAbsoluteMs = 0.1,
      .warnPercent = 2.0,
      .warnAbsoluteMs = 0.05,
  };
  BenchmarkReport current = makeComparableReport(11.0, 13.0, 8.0, 9.0);
  current.benchmarkCase.thresholds = BenchmarkThresholds{
      .failPercent = 1000.0,
      .failAbsoluteMs = 1000.0,
      .warnPercent = 1000.0,
      .warnAbsoluteMs = 1000.0,
  };

  const BenchmarkComparisonReport comparison =
      compareBenchmarkReports(current, baseline);

  EXPECT_TRUE(comparison.valid);
  EXPECT_TRUE(comparison.regression);
}

TEST(NuriBenchmarkingTest,
     AuthoritativeRegressionRequiresIndependentConfidenceAgreement) {
  const auto makeAuthoritative = [](BenchmarkReport report) {
    report.profile.id = "pinned-profile";
    report.profile.profileAuthoritative = true;
    report.profile.authoritative = true;
    report.profile.minimumRepetitions = 3u;
    report.profile.completedRepetitions = 3u;
    report.profile.repetitionRequirementSatisfied = true;
    report.profile.repetitionUnit = "isolated-process";
    report.profile.warmupStabilityPolicy = "required";
    report.profile.warmupStabilityStatus = "stable";
    report.repeatObservations.unit = "isolated-process";
    report.repeatObservations.independent = true;
    report.repeatObservations.count = 3u;
    return report;
  };
  BenchmarkReport baseline =
      makeAuthoritative(makeComparableReport(10.0, 10.0, 8.0, 8.0));
  BenchmarkReport noisy =
      makeAuthoritative(makeComparableReport(12.0, 12.0, 8.0, 8.0));
  addSampleObservations(baseline, "cpu.render_submit_ms", {9.0, 10.0, 11.0});
  addSampleObservations(noisy, "cpu.render_submit_ms", {8.0, 12.0, 16.0});
  addSampleObservations(baseline, "gpu.scopes_sum_ms", {7.9, 8.0, 8.1});
  addSampleObservations(noisy, "gpu.scopes_sum_ms", {7.9, 8.0, 8.1});
  for (BenchmarkReport *report : {&baseline, &noisy}) {
    report->repeatObservations.unit = "isolated-process";
    report->repeatObservations.independent = true;
    report->repeatObservations.count = 3u;
  }

  const BenchmarkComparisonReport inconclusive =
      compareBenchmarkReports(noisy, baseline);

  EXPECT_TRUE(inconclusive.valid);
  EXPECT_TRUE(inconclusive.authoritative);
  EXPECT_FALSE(inconclusive.regression);
  const auto noisyMedian =
      std::find_if(inconclusive.metrics.begin(), inconclusive.metrics.end(),
                   [](const BenchmarkMetricComparison &row) {
                     return row.metricId == "cpu.render_submit_ms" &&
                            row.statistic == "median";
                   });
  ASSERT_NE(noisyMedian, inconclusive.metrics.end());
  EXPECT_EQ(noisyMedian->status, "warn");

  BenchmarkReport clear =
      makeAuthoritative(makeComparableReport(12.0, 12.0, 8.0, 8.0));
  addSampleObservations(clear, "cpu.render_submit_ms", {11.9, 12.0, 12.1});
  addSampleObservations(clear, "gpu.scopes_sum_ms", {7.9, 8.0, 8.1});
  clear.repeatObservations.unit = "isolated-process";
  clear.repeatObservations.independent = true;
  clear.repeatObservations.count = 3u;

  const BenchmarkComparisonReport confirmed =
      compareBenchmarkReports(clear, baseline);
  EXPECT_TRUE(confirmed.valid);
  EXPECT_TRUE(confirmed.authoritative);
  EXPECT_TRUE(confirmed.regression);
}

TEST(NuriBenchmarkingTest, FakeGpuTimingQueueDrainsAndTracksOverflow) {
  FakeRendererGPUDevice gpu;
  gpu.enqueueCompletedGpuTimingReport(makeOpaqueTimingReport(1u, 1.0f));
  gpu.enqueueCompletedGpuTimingReport(makeOpaqueTimingReport(2u, 2.0f));

  std::array<GpuTimingReport, 1u> one{};
  ASSERT_EQ(gpu.drainCompletedGpuTimingReports(one), 1u);
  EXPECT_EQ(one[0].opaqueSourceFrameIndex, 1u);
  EXPECT_FLOAT_EQ(one[0].opaqueTimeMs, 1.0f);

  ASSERT_EQ(gpu.drainCompletedGpuTimingReports(one), 1u);
  EXPECT_EQ(one[0].opaqueSourceFrameIndex, 2u);
  EXPECT_EQ(gpu.drainCompletedGpuTimingReports(one), 0u);

  gpu.enqueueCompletedGpuTimingReport(makeOpaqueTimingReport(3u, 3.0f), 2u);
  gpu.enqueueCompletedGpuTimingReport(makeOpaqueTimingReport(4u, 4.0f), 2u);
  gpu.enqueueCompletedGpuTimingReport(makeOpaqueTimingReport(5u, 5.0f), 2u);
  EXPECT_EQ(gpu.droppedGpuTimingReportCount(), 1u);

  std::array<GpuTimingReport, 2u> two{};
  ASSERT_EQ(gpu.drainCompletedGpuTimingReports(two), 2u);
  EXPECT_EQ(two[0].opaqueSourceFrameIndex, 4u);
  EXPECT_EQ(two[1].opaqueSourceFrameIndex, 5u);
  EXPECT_EQ(gpu.getLatestCompletedGpuTimingReport().opaqueSourceFrameIndex, 5u);
}

} // namespace
