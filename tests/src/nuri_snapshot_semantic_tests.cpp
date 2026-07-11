#include "nuri/tools/snapshot/snapshot_compare.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>

namespace {

nuri::tools::snapshot::SnapshotImage image(uint32_t width, uint32_t height,
                                           uint32_t channels,
                                           std::vector<float> values) {
  return {.width = width,
          .height = height,
          .channelCount = channels,
          .values = std::move(values)};
}

nuri::tools::snapshot::SnapshotCompareProfile
permissiveProfile(std::string_view id) {
  auto profile = nuri::tools::snapshot::builtinSnapshotCompareProfile(id);
  profile.maxAbsError = std::numeric_limits<double>::max();
  profile.meanAbsError = std::numeric_limits<double>::max();
  profile.rmse = std::numeric_limits<double>::max();
  profile.p99AbsError = std::numeric_limits<double>::max();
  profile.maxFailingValues = std::numeric_limits<uint64_t>::max();
  return profile;
}

void expectSemanticVerdictBoundary(
    std::string_view id, const nuri::tools::snapshot::SnapshotImage &actual,
    const nuri::tools::snapshot::SnapshotImage &expected) {
  using namespace nuri::tools::snapshot;
  const auto measured =
      compareSnapshotImages(actual, expected, permissiveProfile(id));
  ASSERT_TRUE(measured.compatible) << id;
  const double boundary =
      std::max(measured.semantic.maxError, measured.semantic.maxSecondaryError);
  ASSERT_GT(boundary, 0.0) << id;

  auto profile = builtinSnapshotCompareProfile(id);
  profile.maxAbsError = boundary;
  profile.meanAbsError = boundary;
  profile.rmse = boundary;
  profile.p99AbsError = boundary;
  profile.maxFailingValues = 0u;
  EXPECT_TRUE(compareSnapshotImages(actual, expected, profile).passed) << id;

  profile.maxAbsError = std::nextafter(boundary, 0.0);
  profile.meanAbsError = profile.maxAbsError;
  profile.rmse = profile.maxAbsError;
  profile.p99AbsError = profile.maxAbsError;
  const auto failed = compareSnapshotImages(actual, expected, profile);
  EXPECT_FALSE(failed.passed) << id;
  EXPECT_FALSE(failed.failedThresholds.empty()) << id;
  EXPECT_EQ(failed.failedThresholds.front().find("semantic"), 0u) << id;
}

TEST(NuriSnapshotSemanticTest, NormalsReportAngularError) {
  using namespace nuri::tools::snapshot;
  const auto result =
      compareSnapshotImages(image(1u, 1u, 3u, {1.0f, 0.0f, 0.0f}),
                            image(1u, 1u, 3u, {0.0f, 1.0f, 0.0f}),
                            builtinSnapshotCompareProfile("normal"));
  EXPECT_TRUE(result.compatible);
  EXPECT_EQ(result.semantic.unit, "degrees");
  EXPECT_NEAR(result.semantic.maxError, 90.0, 1.0e-6);
}

TEST(NuriSnapshotSemanticTest, VelocityReportsVectorMagnitude) {
  using namespace nuri::tools::snapshot;
  const auto result = compareSnapshotImages(
      image(1u, 1u, 2u, {3.0f, 4.0f}), image(1u, 1u, 2u, {0.0f, 0.0f}),
      builtinSnapshotCompareProfile("velocity"));
  EXPECT_EQ(result.semantic.unit, "vector_magnitude");
  EXPECT_DOUBLE_EQ(result.semantic.maxError, 5.0);
}

TEST(NuriSnapshotSemanticTest, MaskReportsChangedPixelsAndBounds) {
  using namespace nuri::tools::snapshot;
  const auto result =
      compareSnapshotImages(image(2u, 2u, 1u, {0.0f, 1.0f, 0.0f, 1.0f}),
                            image(2u, 2u, 1u, {0.0f, 0.0f, 0.0f, 0.0f}),
                            builtinSnapshotCompareProfile("mask"));
  EXPECT_EQ(result.semantic.changedPixels, 2u);
  EXPECT_TRUE(result.semantic.changedBoundsValid);
  EXPECT_EQ(result.semantic.minChangedX, 1u);
  EXPECT_EQ(result.semantic.minChangedY, 0u);
  EXPECT_EQ(result.semantic.maxChangedX, 1u);
  EXPECT_EQ(result.semantic.maxChangedY, 1u);
}

TEST(NuriSnapshotSemanticTest, InvalidSemanticChannelCountIsIncompatible) {
  using namespace nuri::tools::snapshot;
  const auto result = compareSnapshotImages(
      image(1u, 1u, 2u, {1.0f, 0.0f}), image(1u, 1u, 2u, {1.0f, 0.0f}),
      builtinSnapshotCompareProfile("normal"));
  EXPECT_FALSE(result.compatible);
  EXPECT_FALSE(result.passed);
}

TEST(NuriSnapshotSemanticTest, EveryCaptureKindUsesSemanticVerdictBoundaries) {
  expectSemanticVerdictBoundary("ldr_color",
                                image(1u, 1u, 3u, {0.25f, 0.0f, 0.0f}),
                                image(1u, 1u, 3u, {0.0f, 0.0f, 0.0f}));
  expectSemanticVerdictBoundary("hdr_color",
                                image(1u, 1u, 3u, {4.0f, 2.0f, 1.0f}),
                                image(1u, 1u, 3u, {2.0f, 1.0f, 0.5f}));
  expectSemanticVerdictBoundary("depth", image(1u, 1u, 1u, {0.75f}),
                                image(1u, 1u, 1u, {0.5f}));
  expectSemanticVerdictBoundary("shadow_depth", image(1u, 1u, 1u, {0.75f}),
                                image(1u, 1u, 1u, {0.5f}));
  expectSemanticVerdictBoundary("normal", image(1u, 1u, 3u, {1.0f, 0.0f, 0.0f}),
                                image(1u, 1u, 3u, {0.0f, 1.0f, 0.0f}));
  expectSemanticVerdictBoundary("velocity", image(1u, 1u, 2u, {3.0f, 4.0f}),
                                image(1u, 1u, 2u, {0.0f, 0.0f}));
  expectSemanticVerdictBoundary("mask", image(1u, 1u, 1u, {1.0f}),
                                image(1u, 1u, 1u, {0.0f}));
  expectSemanticVerdictBoundary("scalar", image(1u, 1u, 1u, {0.75f}),
                                image(1u, 1u, 1u, {0.5f}));
  expectSemanticVerdictBoundary("debug_preview",
                                image(1u, 1u, 3u, {0.25f, 0.0f, 0.0f}),
                                image(1u, 1u, 3u, {0.0f, 0.0f, 0.0f}));
}

TEST(NuriSnapshotSemanticTest, NormalGateIgnoresVectorLength) {
  using namespace nuri::tools::snapshot;
  const auto result =
      compareSnapshotImages(image(1u, 1u, 3u, {100.0f, 0.0f, 0.0f}),
                            image(1u, 1u, 3u, {1.0f, 0.0f, 0.0f}),
                            builtinSnapshotCompareProfile("normal"));
  EXPECT_TRUE(result.passed);
  EXPECT_DOUBLE_EQ(result.semantic.maxError, 0.0);
  EXPECT_GT(result.metrics.maxAbsError, 1.0);
}

TEST(NuriSnapshotSemanticTest, VelocityGateUsesVectorMagnitude) {
  using namespace nuri::tools::snapshot;
  auto profile = builtinSnapshotCompareProfile("velocity");
  profile.maxAbsError = 0.1;
  profile.meanAbsError = 0.1;
  profile.rmse = 0.1;
  profile.p99AbsError = 0.1;
  const auto result =
      compareSnapshotImages(image(1u, 1u, 2u, {0.08f, 0.08f}),
                            image(1u, 1u, 2u, {0.0f, 0.0f}), profile);
  EXPECT_FALSE(result.passed);
  EXPECT_LT(result.metrics.maxAbsError, profile.maxAbsError);
  EXPECT_GT(result.semantic.maxError, profile.maxAbsError);
}

TEST(NuriSnapshotSemanticTest, HdrReportsLogAndRelativeLuminance) {
  using namespace nuri::tools::snapshot;
  const auto result = compareSnapshotImages(
      image(1u, 1u, 3u, {2.0f, 2.0f, 2.0f}),
      image(1u, 1u, 3u, {1.0f, 1.0f, 1.0f}), permissiveProfile("hdr_color"));
  EXPECT_EQ(result.semantic.unit, "log_luminance");
  EXPECT_EQ(result.semantic.secondaryUnit, "relative_luminance");
  EXPECT_NEAR(result.semantic.maxError, std::log(3.0) - std::log(2.0), 1.0e-6);
  EXPECT_NEAR(result.semantic.maxSecondaryError, 1.0, 1.0e-6);
}

TEST(NuriSnapshotSemanticTest, DepthReportsRelativeAndAbsoluteError) {
  using namespace nuri::tools::snapshot;
  const auto result = compareSnapshotImages(image(1u, 1u, 1u, {101.0f}),
                                            image(1u, 1u, 1u, {100.0f}),
                                            permissiveProfile("depth"));
  EXPECT_EQ(result.semantic.unit, "relative");
  EXPECT_EQ(result.semantic.secondaryUnit, "absolute");
  EXPECT_NEAR(result.semantic.maxError, 0.01, 1.0e-6);
  EXPECT_NEAR(result.semantic.maxSecondaryError, 1.0, 1.0e-6);
}

TEST(NuriSnapshotSemanticTest, MaskReportsConfusionAndExactChangedPixels) {
  using namespace nuri::tools::snapshot;
  const auto result = compareSnapshotImages(
      image(4u, 1u, 1u, {0.0f, 1.0f, 0.0f, 1.0f}),
      image(4u, 1u, 1u, {0.0f, 1.0f, 1.0f, 0.0f}), permissiveProfile("mask"));
  EXPECT_EQ(result.semantic.changedPixels, 2u);
  EXPECT_EQ(result.semantic.truePositivePixels, 1u);
  EXPECT_EQ(result.semantic.trueNegativePixels, 1u);
  EXPECT_EQ(result.semantic.falsePositivePixels, 1u);
  EXPECT_EQ(result.semantic.falseNegativePixels, 1u);
  EXPECT_NEAR(result.semantic.intersectionOverUnion, 1.0 / 3.0, 1.0e-9);
  EXPECT_EQ(result.semantic.minChangedX, 2u);
  EXPECT_EQ(result.semantic.maxChangedX, 3u);
}

TEST(NuriSnapshotSemanticTest, RoiAndValidityMaskExcludePixelsFromVerdict) {
  using namespace nuri::tools::snapshot;
  const SnapshotImage actual = image(3u, 1u, 1u, {1.0f, 0.0f, 1.0f});
  const SnapshotImage expected = image(3u, 1u, 1u, {0.0f, 0.0f, 0.0f});
  SnapshotCompareOptions roiOptions{
      .roi = {.x = 1u, .y = 0u, .width = 1u, .height = 1u}};
  const auto roi = compareSnapshotImages(
      actual, expected, builtinSnapshotCompareProfile("scalar"), roiOptions);
  EXPECT_TRUE(roi.passed);
  EXPECT_EQ(roi.semantic.validPixels, 1u);
  EXPECT_EQ(roi.semantic.ignoredPixels, 2u);

  const std::array<uint8_t, 3u> validity{0u, 1u, 0u};
  SnapshotCompareOptions maskOptions{.validityMask = validity};
  const auto masked = compareSnapshotImages(
      actual, expected, builtinSnapshotCompareProfile("scalar"), maskOptions);
  EXPECT_TRUE(masked.passed);
  EXPECT_EQ(masked.semantic.validPixels, 1u);
  EXPECT_EQ(masked.semantic.ignoredPixels, 2u);

  const std::array<uint8_t, 2u> invalidMask{1u, 1u};
  const auto invalid = compareSnapshotImages(
      actual, expected, builtinSnapshotCompareProfile("scalar"),
      SnapshotCompareOptions{.validityMask = invalidMask});
  EXPECT_FALSE(invalid.compatible);
}

TEST(NuriSnapshotSemanticTest, ComparisonPreviewsShareOneScale) {
  using namespace nuri::tools::snapshot;
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "nuri_snapshot_shared_scale";
  std::filesystem::create_directories(root);
  const std::filesystem::path actualPath = root / "actual.png";
  const std::filesystem::path expectedPath = root / "expected.png";
  const auto written = writeSnapshotComparisonPreviews(
      image(3u, 1u, 1u, {0.0f, 1.0f, 1.5f}),
      image(3u, 1u, 1u, {0.0f, 1.0f, 2.0f}), "depth", actualPath, expectedPath);
  ASSERT_FALSE(written.hasError()) << written.error();
  const auto actual = readSnapshotImageFile(actualPath);
  const auto expected = readSnapshotImageFile(expectedPath);
  ASSERT_FALSE(actual.hasError()) << actual.error();
  ASSERT_FALSE(expected.hasError()) << expected.error();
  ASSERT_EQ(actual.value().values.size(), expected.value().values.size());
  for (uint32_t channel = 0u; channel < 4u; ++channel) {
    EXPECT_FLOAT_EQ(actual.value().values[4u + channel],
                    expected.value().values[4u + channel]);
  }
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

} // namespace
