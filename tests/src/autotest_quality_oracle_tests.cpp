#include "tests_pch.h"

#include <gtest/gtest.h>

#include "nuri/tools/autotest/autotest_manifest.h"
#include "nuri/tools/autotest/autotest_quality_oracle.h"
#include "nuri/tools/autotest/autotest_report.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string_view>

namespace {

using namespace nuri::tools::autotest;
using nuri::tools::snapshot::SnapshotImage;

[[nodiscard]] SnapshotImage rgbImage(uint32_t width, uint32_t height,
                                     float value) {
  SnapshotImage image{};
  image.width = width;
  image.height = height;
  image.channelCount = 3u;
  image.values.resize(static_cast<size_t>(width) * height * 3u, value);
  return image;
}

[[nodiscard]] SnapshotImage scalarImage(uint32_t width, uint32_t height,
                                        float value) {
  SnapshotImage image{};
  image.width = width;
  image.height = height;
  image.channelCount = 1u;
  image.values.resize(static_cast<size_t>(width) * height, value);
  return image;
}

[[nodiscard]] SnapshotImage zeroMotion(uint32_t width, uint32_t height) {
  SnapshotImage image{};
  image.width = width;
  image.height = height;
  image.channelCount = 2u;
  image.values.resize(static_cast<size_t>(width) * height * 2u, 0.0f);
  return image;
}

[[nodiscard]] AutotestQualityOracle qualityOracle(bool temporal = false) {
  AutotestQualityOracle oracle{};
  oracle.reference = {
      .version = 1u,
      .pathBase = "repoRoot",
      .path = "tools/references/aa/v1/test.exr",
  };
  oracle.lscale = 1.0;
  oracle.budgets = {
      .normalizedMaeMax = 0.01,
      .normalizedRmseMax = 0.01,
      .lumaSsimMin = 0.99,
      .darkCollapsePercentMax = 0.1,
      .darkCollapseComponentMaxPixels = 4u,
      .relativeLumaEnergyDriftMax = 0.01,
      .edgeWidthRatioMin = 0.9,
      .edgeWidthRatioMax = 1.1,
      .edgeOvershootMax = 0.05,
      .edgeUndershootMax = 0.05,
      .temporalErrorMax = 0.01,
      .ghostEnergyMax = 0.01,
      .recoveryRmseMax = 0.01,
  };
  if (temporal) {
    oracle.temporal = AutotestQualityOracleTemporal{
        .previousCheckpoint = "previous",
        .previousOutputTarget = "frame_color_hdr",
        .previousReference =
            AutotestQualityOracleFile{
                .version = 1u,
                .pathBase = "repoRoot",
                .path = "tools/references/aa/v1/previous.exr",
            },
        .motionTarget = "motion_vectors",
        .revealMask =
            AutotestQualityOracleMask{
                .version = 1u,
                .pathBase = "repoRoot",
                .path = "tools/references/aa/v1/reveal.png",
            },
    };
  }
  return oracle;
}

[[nodiscard]] bool hasFailure(const AutotestQualityOracleReport &report,
                              std::string_view failure) {
  return std::find(report.failedThresholds.begin(),
                   report.failedThresholds.end(), failure) !=
         report.failedThresholds.end();
}

[[nodiscard]] std::filesystem::path tempJsonPath(std::string_view stem) {
  const auto tick =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("nuri_" + std::string(stem) + "_" + std::to_string(tick) +
          ".json");
}

void writeText(const std::filesystem::path &path, std::string_view text) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file << text;
}

TEST(AutotestQualityOracleTest, IdenticalHdrAndTemporalInputsPass) {
  const AutotestQualityOracle oracle = qualityOracle(true);
  const SnapshotImage output = rgbImage(4u, 4u, 0.75f);
  const SnapshotImage reference = output;
  const SnapshotImage previousOutput = rgbImage(4u, 4u, 0.25f);
  const SnapshotImage previousReference = previousOutput;
  const SnapshotImage motion = zeroMotion(4u, 4u);
  const SnapshotImage reveal = scalarImage(4u, 4u, 1.0f);
  const AutotestQualityOracleInputs inputs{
      .output = &output,
      .reference = &reference,
      .previousOutput = &previousOutput,
      .previousReference = &previousReference,
      .analyticMotion = &motion,
      .revealMask = &reveal,
  };

  auto evaluated = evaluateAutotestQualityOracle(oracle, inputs);
  ASSERT_FALSE(evaluated.hasError()) << evaluated.error();
  EXPECT_EQ(evaluated.value().status, "pass");
  EXPECT_DOUBLE_EQ(evaluated.value().normalizedHdrMae, 0.0);
  EXPECT_DOUBLE_EQ(evaluated.value().normalizedHdrRmse, 0.0);
  EXPECT_NEAR(evaluated.value().lumaSsim, 1.0, 1.0e-12);
  EXPECT_TRUE(evaluated.value().temporalAvailable);
  EXPECT_GT(evaluated.value().temporalSampleCount, 0u);
  EXPECT_DOUBLE_EQ(evaluated.value().temporalError, 0.0);
  EXPECT_DOUBLE_EQ(evaluated.value().ghostEnergy, 0.0);
  EXPECT_DOUBLE_EQ(evaluated.value().recoveryRmse, 0.0);
}

TEST(AutotestQualityOracleTest,
     DarkCollapseNonFiniteAndEnergyFailuresAreObjective) {
  const AutotestQualityOracle oracle = qualityOracle();
  SnapshotImage output = rgbImage(4u, 4u, 0.0f);
  const SnapshotImage reference = rgbImage(4u, 4u, 1.0f);
  output.values[0] = std::numeric_limits<float>::quiet_NaN();
  const AutotestQualityOracleInputs inputs{
      .output = &output,
      .reference = &reference,
  };

  auto evaluated = evaluateAutotestQualityOracle(oracle, inputs);
  ASSERT_FALSE(evaluated.hasError()) << evaluated.error();
  const AutotestQualityOracleReport &report = evaluated.value();
  EXPECT_EQ(report.status, "fail");
  EXPECT_GT(report.nonFiniteValueCount, 0u);
  EXPECT_GT(report.darkCollapsePercent, 90.0);
  EXPECT_GT(report.darkCollapseMaxComponentPixels, 4u);
  EXPECT_GT(report.relativeLumaEnergyDrift, 0.9);
  EXPECT_TRUE(hasFailure(report, "non_finite_values"));
  EXPECT_TRUE(hasFailure(report, "normalized_hdr_rmse"));
  EXPECT_TRUE(hasFailure(report, "dark_collapse_percent"));
  EXPECT_TRUE(hasFailure(report, "dark_collapse_component_pixels"));
  EXPECT_TRUE(hasFailure(report, "relative_luma_energy_drift"));
}

TEST(AutotestQualityOracleTest,
     TemporalResidualGhostAndRecoveryFailuresUseAnalyticMotion) {
  AutotestQualityOracle oracle = qualityOracle(true);
  oracle.budgets.normalizedMaeMax = 1.0;
  oracle.budgets.normalizedRmseMax = 1.0;
  oracle.budgets.lumaSsimMin = 0.0;
  oracle.budgets.darkCollapsePercentMax = 100.0;
  oracle.budgets.darkCollapseComponentMaxPixels = 16u;
  oracle.budgets.relativeLumaEnergyDriftMax = 1.0;
  const SnapshotImage output = rgbImage(4u, 4u, 0.5f);
  const SnapshotImage reference = rgbImage(4u, 4u, 1.0f);
  const SnapshotImage previousOutput = rgbImage(4u, 4u, 0.0f);
  const SnapshotImage previousReference = previousOutput;
  const SnapshotImage motion = zeroMotion(4u, 4u);
  const SnapshotImage reveal = scalarImage(4u, 4u, 1.0f);
  const AutotestQualityOracleInputs inputs{
      .output = &output,
      .reference = &reference,
      .previousOutput = &previousOutput,
      .previousReference = &previousReference,
      .analyticMotion = &motion,
      .revealMask = &reveal,
  };

  auto evaluated = evaluateAutotestQualityOracle(oracle, inputs);
  ASSERT_FALSE(evaluated.hasError()) << evaluated.error();
  EXPECT_EQ(evaluated.value().status, "fail");
  EXPECT_NEAR(evaluated.value().temporalError, 0.5, 1.0e-6);
  EXPECT_NEAR(evaluated.value().ghostEnergy, 0.5, 1.0e-6);
  EXPECT_NEAR(evaluated.value().recoveryRmse, 0.5, 1.0e-6);
  EXPECT_TRUE(hasFailure(evaluated.value(), "temporal_error"));
  EXPECT_TRUE(hasFailure(evaluated.value(), "ghost_energy"));
  EXPECT_TRUE(hasFailure(evaluated.value(), "recovery_rmse"));
}

TEST(AutotestQualityOracleTest, SlantedEdgeWidthAndExcursionGatesAreMeasured) {
  AutotestQualityOracle oracle = qualityOracle();
  oracle.budgets.normalizedMaeMax = 1.0;
  oracle.budgets.normalizedRmseMax = 1.0;
  oracle.budgets.lumaSsimMin = 0.0;
  oracle.budgets.relativeLumaEnergyDriftMax = 1.0;
  const std::array<float, 8> referenceProfile{0.0f, 0.0f, 0.0f, 0.2f,
                                               0.8f, 1.0f, 1.0f, 1.0f};
  const std::array<float, 8> blurredProfile{0.0f, 0.05f, 0.2f, 0.35f,
                                             0.65f, 0.8f, 0.95f, 1.0f};
  SnapshotImage reference = rgbImage(8u, 4u, 0.0f);
  SnapshotImage identical = reference;
  SnapshotImage blurred = reference;
  for (uint32_t y = 0u; y < 4u; ++y) {
    for (uint32_t x = 0u; x < 8u; ++x) {
      for (uint32_t channel = 0u; channel < 3u; ++channel) {
        const size_t index = (static_cast<size_t>(y) * 8u + x) * 3u + channel;
        reference.values[index] = referenceProfile[x];
        identical.values[index] = referenceProfile[x];
        blurred.values[index] = blurredProfile[x];
      }
    }
  }
  AutotestQualityOracleInputs inputs{.output = &identical,
                                     .reference = &reference};
  auto evaluated = evaluateAutotestQualityOracle(oracle, inputs);
  ASSERT_FALSE(evaluated.hasError()) << evaluated.error();
  EXPECT_EQ(evaluated.value().status, "pass");
  EXPECT_TRUE(evaluated.value().edgeAvailable);
  EXPECT_EQ(evaluated.value().edgeAxis, "horizontal");
  EXPECT_NEAR(evaluated.value().edgeWidthRatio, 1.0, 1.0e-6);

  inputs.output = &blurred;
  evaluated = evaluateAutotestQualityOracle(oracle, inputs);
  ASSERT_FALSE(evaluated.hasError()) << evaluated.error();
  EXPECT_EQ(evaluated.value().status, "fail");
  EXPECT_GT(evaluated.value().edgeWidthRatio, 1.1);
  EXPECT_TRUE(hasFailure(evaluated.value(), "edge_width_ratio"));
}

TEST(AutotestQualityOracleTest,
     ManifestParsesExplicitUnavailableReferenceAndRejectsInvalidLscale) {
  const std::filesystem::path manifestPath =
      std::filesystem::path(PROJECT_SOURCE_DIR) / "tools" / "cases" /
      "autotests" / "correctness" / "procedural_aa_quality_oracle.json";
  auto loaded = loadAutotestCaseManifest(manifestPath);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  ASSERT_EQ(loaded.value().checkpoints.size(), 1u);
  ASSERT_TRUE(loaded.value().checkpoints[0].qualityOracle.has_value());
  const AutotestQualityOracle &oracle =
      *loaded.value().checkpoints[0].qualityOracle;
  EXPECT_FALSE(oracle.reference.available);
  EXPECT_EQ(oracle.reference.unavailableReason, "reference_not_generated");
  EXPECT_EQ(oracle.reference.version, 1u);
  EXPECT_DOUBLE_EQ(oracle.lscale, 0.0);
  EXPECT_DOUBLE_EQ(oracle.budgets.normalizedRmseMax, 0.02);

  const std::filesystem::path invalidPath =
      tempJsonPath("quality_oracle_invalid_lscale");
  std::ifstream manifestFile(manifestPath, std::ios::binary);
  ASSERT_TRUE(manifestFile.is_open());
  std::string invalid((std::istreambuf_iterator<char>(manifestFile)),
                      std::istreambuf_iterator<char>());
  const size_t available = invalid.find("\"available\": false");
  ASSERT_NE(available, std::string::npos);
  invalid.replace(available, std::string("\"available\": false").size(),
                  "\"available\": true");
  writeText(invalidPath, invalid);
  loaded = loadAutotestCaseManifest(invalidPath);
  ASSERT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("Lscale/budgets are invalid"),
            std::string::npos);
  std::error_code error;
  std::filesystem::remove(invalidPath, error);
}

TEST(AutotestQualityOracleTest, ReportJsonAndHtmlContainCheckpointMetrics) {
  AutotestReport report{};
  report.generatedAtUtc = "2026-07-10T00:00:00Z";
  report.command = "nuri-autotest run";
  report.status = "fail";
  report.exitCode = AutotestExitCode::ScenarioFailure;
  report.selection = {.requested = "correctness.quality.report",
                      .selected = 1u,
                      .attempted = 1u,
                      .completed = 1u,
                      .failed = 1u};
  report.testCase.id = "correctness.quality.report";
  report.testCase.suite = "correctness";
  report.testCase.resolution = {4u, 4u};
  report.reproduceCommand =
      "nuri-autotest run --case correctness.quality.report";
  AutotestCheckpointReport checkpoint{};
  checkpoint.id = "quality";
  checkpoint.frame = 4u;
  checkpoint.qualityOracle = AutotestQualityOracleReport{
      .status = "fail",
      .statusReason = "quality_oracle_thresholds_failed",
      .outputTarget = "frame_color_hdr",
      .referencePath = "tools/references/aa/v1/reference.exr",
      .schemaVersion = 1u,
      .referenceVersion = 3u,
      .maskVersion = 2u,
      .lscale = 1.0,
      .selectedPixelCount = 16u,
      .finitePixelCount = 16u,
      .normalizedHdrMae = 0.03,
      .normalizedHdrRmse = 0.04,
      .lumaSsim = 0.9,
      .budgets = qualityOracle().budgets,
      .failedThresholds = {"normalized_hdr_rmse"},
  };
  report.checkpoints.push_back(std::move(checkpoint));

  auto json = writeAutotestReportJson(report);
  ASSERT_FALSE(json.hasError()) << json.error();
  EXPECT_NE(json.value().find("\"qualityOracle\""), std::string::npos);
  EXPECT_NE(json.value().find("\"normalizedHdrRmse\""),
            std::string::npos);
  auto html = writeAutotestHtmlReport(report);
  ASSERT_FALSE(html.hasError()) << html.error();
  EXPECT_NE(html.value().find("Deterministic AA quality oracle"),
            std::string::npos);
  EXPECT_NE(html.value().find("Normalized HDR RMSE"), std::string::npos);
}

} // namespace
