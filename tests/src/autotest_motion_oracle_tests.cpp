#include "tests_pch.h"

#include <gtest/gtest.h>

#include "nuri/tools/autotest/autotest_manifest.h"
#include "nuri/tools/autotest/autotest_motion_oracle.h"
#include "nuri/tools/autotest/autotest_report.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string_view>

namespace {

using namespace nuri::tools::autotest;
using nuri::tools::snapshot::SnapshotImage;

[[nodiscard]] std::filesystem::path tempJsonPath(std::string_view stem) {
  const auto tick =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("nuri_" + std::string(stem) + "_" + std::to_string(tick) + ".json");
}

void writeText(const std::filesystem::path &path, std::string_view text) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file << text;
}

[[nodiscard]] SnapshotImage constantMotionImage(float xPixels, float yPixels) {
  SnapshotImage image{};
  image.width = 4u;
  image.height = 4u;
  image.channelCount = 2u;
  image.values.resize(4u * 4u * 2u);
  for (size_t i = 0u; i < image.values.size(); i += 2u) {
    image.values[i] = xPixels / static_cast<float>(image.width);
    image.values[i + 1u] = yPixels / static_cast<float>(image.height);
  }
  return image;
}

[[nodiscard]] AutotestMotionOracle makeOracle() {
  AutotestMotionOracle oracle{};
  oracle.motionClassTarget = "motion_class";
  oracle.roi = {.x = 0u, .y = 0u, .width = 4u, .height = 4u};
  oracle.expectedVelocityPixels = {2.0f, -1.0f};
  oracle.p95ErrorMaxPixels = 0.01;
  oracle.maxErrorMaxPixels = 0.02;
  oracle.classCoverage.configured = true;
  oracle.classCoverage.invalid = {
      .hasMin = true, .min = 0.25, .hasMax = true, .max = 0.25};
  oracle.classCoverage.staticCameraOnly = {
      .hasMin = true, .min = 0.25, .hasMax = true, .max = 0.25};
  oracle.classCoverage.full = {
      .hasMin = true, .min = 0.5, .hasMax = true, .max = 0.5};
  return oracle;
}

TEST(AutotestMotionOracleTest, ProceduralManifestParsesAnalyticEndpoint) {
  const std::filesystem::path path =
      std::filesystem::path(PROJECT_SOURCE_DIR) / "tools" / "cases" /
      "autotests" / "correctness" / "procedural_motion_endpoint.json";
  auto loaded = loadAutotestCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  ASSERT_EQ(loaded.value().checkpoints.size(), 1u);
  const AutotestCheckpoint &checkpoint = loaded.value().checkpoints[0];
  ASSERT_TRUE(checkpoint.motionOracle.has_value());
  EXPECT_EQ(checkpoint.motionOracle->motionTarget, "motion_vectors");
  EXPECT_EQ(checkpoint.motionOracle->roi.x, 440u);
  EXPECT_EQ(checkpoint.motionOracle->mask.size(), 16u);
  EXPECT_NEAR(checkpoint.motionOracle->expectedVelocityPixels.x, 7.79422863f,
              1.0e-6f);
  EXPECT_EQ(checkpoint.motionOracle->motionClassTarget, "motion_class");
  EXPECT_TRUE(checkpoint.motionOracle->classCoverage.configured);
  EXPECT_TRUE(checkpoint.motionOracle->classCoverage.staticCameraOnly.hasMin);
  EXPECT_DOUBLE_EQ(checkpoint.motionOracle->classCoverage.staticCameraOnly.min,
                   1.0);
}

TEST(AutotestMotionOracleTest, ManifestParsesClassCoverageAndRejectsBadMask) {
  const std::filesystem::path path = tempJsonPath("motion_oracle_manifest");
  const std::string manifest = R"json({
    "schemaVersion": 1,
    "id": "correctness.motion.class_contract",
    "suite": "correctness",
    "resolution": [4, 4],
    "warmupFrames": 0,
    "endFrame": 1,
    "checkpoints": [{
      "id": "endpoint",
      "frame": 1,
      "captures": [
        {"target": "motion_vectors", "profile": "velocity", "required": true, "compare": false},
        {"target": "motion_class", "profile": "mask", "required": true, "compare": false}
      ],
      "motionOracle": {
        "motionClassTarget": "motion_class",
        "roi": {"x": 0, "y": 0, "width": 2, "height": 2},
        "mask": [[0, 0], [1, 1]],
        "expectedVelocityPixels": [2, -1],
        "p95ErrorMaxPixels": 0.1,
        "maxErrorMaxPixels": 0.2,
        "classCoverage": {
          "invalid": {"max": 0.1},
          "static": {"max": 0.2},
          "full": {"min": 0.7}
        }
      }
    }]
  })json";
  writeText(path, manifest);
  auto loaded = loadAutotestCaseManifest(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  ASSERT_TRUE(loaded.value().checkpoints[0].motionOracle.has_value());
  const AutotestMotionOracle &oracle =
      *loaded.value().checkpoints[0].motionOracle;
  EXPECT_EQ(oracle.motionClassTarget, "motion_class");
  EXPECT_TRUE(oracle.classCoverage.invalid.hasMax);
  EXPECT_DOUBLE_EQ(oracle.classCoverage.full.min, 0.7);

  const size_t secondMaskPixel = manifest.find("[1, 1]");
  ASSERT_NE(secondMaskPixel, std::string::npos);
  std::string duplicateMask = manifest;
  duplicateMask.replace(secondMaskPixel, 6u, "[0, 0]");
  writeText(path, duplicateMask);
  loaded = loadAutotestCaseManifest(path);
  ASSERT_TRUE(loaded.hasError());
  EXPECT_NE(loaded.error().find("duplicate pixel"), std::string::npos);

  std::error_code error;
  std::filesystem::remove(path, error);
}

TEST(AutotestMotionOracleTest, IdealMotionAndClassCoveragePass) {
  const AutotestMotionOracle oracle = makeOracle();
  const SnapshotImage motion = constantMotionImage(2.0f, -1.0f);
  SnapshotImage motionClass{};
  motionClass.width = 4u;
  motionClass.height = 4u;
  motionClass.channelCount = 1u;
  motionClass.values.resize(16u, 2.0f / 255.0f);
  std::fill_n(motionClass.values.begin(), 4u, 0.0f);
  std::fill_n(motionClass.values.begin() + 4u, 4u, 1.0f / 255.0f);

  auto evaluated = evaluateAutotestMotionOracle(oracle, motion, &motionClass);
  ASSERT_FALSE(evaluated.hasError()) << evaluated.error();
  EXPECT_EQ(evaluated.value().status, "pass");
  EXPECT_EQ(evaluated.value().selectedPixelCount, 16u);
  EXPECT_DOUBLE_EQ(evaluated.value().p95ErrorPixels, 0.0);
  EXPECT_EQ(evaluated.value().wrongSignPixelCount, 0u);
  EXPECT_DOUBLE_EQ(evaluated.value().invalidClassCoverage, 0.25);
  EXPECT_DOUBLE_EQ(evaluated.value().staticClassCoverage, 0.25);
  EXPECT_DOUBLE_EQ(evaluated.value().fullClassCoverage, 0.5);
}

TEST(AutotestMotionOracleTest, ReversedSignAndWrongScaleFailObjectively) {
  AutotestMotionOracle oracle = makeOracle();
  oracle.motionClassTarget.clear();
  oracle.classCoverage = {};

  auto reversed =
      evaluateAutotestMotionOracle(oracle, constantMotionImage(-2.0f, 1.0f));
  ASSERT_FALSE(reversed.hasError()) << reversed.error();
  EXPECT_EQ(reversed.value().status, "fail");
  EXPECT_EQ(reversed.value().wrongSignPixelCount, 16u);
  EXPECT_NE(std::find(reversed.value().failedThresholds.begin(),
                      reversed.value().failedThresholds.end(), "wrong_sign"),
            reversed.value().failedThresholds.end());

  auto wrongScale =
      evaluateAutotestMotionOracle(oracle, constantMotionImage(1.0f, -0.5f));
  ASSERT_FALSE(wrongScale.hasError()) << wrongScale.error();
  EXPECT_EQ(wrongScale.value().status, "fail");
  EXPECT_EQ(wrongScale.value().wrongSignPixelCount, 0u);
  EXPECT_GT(wrongScale.value().p95ScaleErrorPixels, 1.0);
  EXPECT_NE(std::find(wrongScale.value().failedThresholds.begin(),
                      wrongScale.value().failedThresholds.end(),
                      "p95_scale_error_pixels"),
            wrongScale.value().failedThresholds.end());
}

TEST(AutotestMotionOracleTest, ExactStaticMotionDoesNotReportWrongSign) {
  AutotestMotionOracle oracle{};
  oracle.roi = {.x = 0u, .y = 0u, .width = 4u, .height = 4u};
  oracle.expectedVelocityPixels = {0.0f, 0.0f};
  oracle.p95ErrorMaxPixels = 0.01;
  oracle.maxErrorMaxPixels = 0.01;

  auto evaluated =
      evaluateAutotestMotionOracle(oracle, constantMotionImage(0.0f, 0.0f));
  ASSERT_FALSE(evaluated.hasError()) << evaluated.error();
  EXPECT_EQ(evaluated.value().status, "pass");
  EXPECT_EQ(evaluated.value().wrongSignPixelCount, 0u);
  EXPECT_DOUBLE_EQ(evaluated.value().p95ErrorPixels, 0.0);
}

TEST(AutotestMotionOracleTest, ReportRoundTripsMotionMetrics) {
  AutotestReport report{};
  report.generatedAtUtc = "2026-07-10T00:00:00Z";
  report.command = "nuri-autotest run";
  report.status = "fail";
  report.exitCode = AutotestExitCode::ScenarioFailure;
  report.selection = {.requested = "correctness.motion.report",
                      .selected = 1u,
                      .attempted = 1u,
                      .completed = 1u,
                      .failed = 1u};
  report.testCase.id = "correctness.motion.report";
  report.testCase.suite = "correctness";
  report.testCase.resolution = {4u, 4u};
  report.reproduceCommand =
      "nuri-autotest run --case correctness.motion.report";
  AutotestCheckpointReport checkpoint{};
  checkpoint.id = "endpoint";
  checkpoint.frame = 1u;
  checkpoint.motionOracle = AutotestMotionOracleReport{
      .status = "fail",
      .statusReason = "motion_oracle_thresholds_failed",
      .motionTarget = "motion_vectors",
      .roi = {.x = 0u, .y = 0u, .width = 4u, .height = 4u},
      .selectedPixelCount = 16u,
      .expectedVelocityPixels = {2.0, -1.0},
      .meanVelocityPixels = {-2.0, 1.0},
      .p95ErrorPixels = 4.47213595,
      .maxErrorPixels = 4.47213595,
      .p95ScaleErrorPixels = 0.0,
      .maxScaleErrorPixels = 0.0,
      .wrongSignPixelCount = 16u,
      .p95ErrorMaxPixels = 0.01,
      .maxErrorMaxPixels = 0.02,
      .failedThresholds = {"wrong_sign", "p95_error_pixels"},
  };
  report.checkpoints.push_back(std::move(checkpoint));

  const std::filesystem::path path = tempJsonPath("motion_oracle_report");
  auto written = writeAutotestReportFile(report, path);
  ASSERT_FALSE(written.hasError()) << written.error();
  auto loaded = readAutotestReportFile(path);
  ASSERT_FALSE(loaded.hasError()) << loaded.error();
  ASSERT_EQ(loaded.value().checkpoints.size(), 1u);
  ASSERT_TRUE(loaded.value().checkpoints[0].motionOracle.has_value());
  EXPECT_EQ(loaded.value().checkpoints[0].motionOracle->wrongSignPixelCount,
            16u);
  EXPECT_EQ(loaded.value().checkpoints[0].motionOracle->failedThresholds.size(),
            2u);

  std::error_code error;
  std::filesystem::remove(path, error);
}

} // namespace
