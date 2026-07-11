#include "nuri/tools/autotest/autotest_motion_oracle.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string_view>

namespace nuri::tools::autotest {
namespace {

[[nodiscard]] bool
imagePayloadValid(const nuri::tools::snapshot::SnapshotImage &image,
                  uint32_t requiredChannels) noexcept {
  return image.width != 0u && image.height != 0u &&
         image.channelCount >= requiredChannels &&
         image.values.size() >= static_cast<size_t>(image.width) *
                                    image.height * image.channelCount;
}

[[nodiscard]] bool roiFits(const AutotestPixelRoi &roi, uint32_t width,
                           uint32_t height) noexcept {
  return roi.width != 0u && roi.height != 0u && roi.x < width &&
         roi.y < height && roi.width <= width - roi.x &&
         roi.height <= height - roi.y;
}

[[nodiscard]] double nearestRankP95(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t rank = std::max<size_t>(
      1u, static_cast<size_t>(std::ceil(values.size() * 0.95)));
  return values[rank - 1u];
}

void checkCoverageRange(double value, const AutotestCoverageRange &range,
                        std::string_view name,
                        std::vector<std::string> &failures) {
  if (range.hasMin && value < range.min) {
    failures.emplace_back(std::string(name) + "_coverage_min");
  }
  if (range.hasMax && value > range.max) {
    failures.emplace_back(std::string(name) + "_coverage_max");
  }
}

} // namespace

Result<AutotestMotionOracleReport, std::string> evaluateAutotestMotionOracle(
    const AutotestMotionOracle &oracle,
    const nuri::tools::snapshot::SnapshotImage &motionImage,
    const nuri::tools::snapshot::SnapshotImage *motionClassImage) {
  if (!imagePayloadValid(motionImage, 2u)) {
    return Result<AutotestMotionOracleReport, std::string>::makeError(
        "motion oracle requires a valid two-channel motion image");
  }
  if (!roiFits(oracle.roi, motionImage.width, motionImage.height)) {
    return Result<AutotestMotionOracleReport, std::string>::makeError(
        "motion oracle ROI is outside the motion image");
  }
  const double expectedX = oracle.expectedVelocityPixels.x;
  const double expectedY = oracle.expectedVelocityPixels.y;
  const double expectedMagnitude = std::hypot(expectedX, expectedY);
  if (!std::isfinite(expectedMagnitude) ||
      !std::isfinite(oracle.p95ErrorMaxPixels) ||
      !std::isfinite(oracle.maxErrorMaxPixels) ||
      oracle.p95ErrorMaxPixels < 0.0 || oracle.maxErrorMaxPixels < 0.0) {
    return Result<AutotestMotionOracleReport, std::string>::makeError(
        "motion oracle configuration is invalid");
  }
  if (!oracle.motionClassTarget.empty()) {
    if (motionClassImage == nullptr ||
        !imagePayloadValid(*motionClassImage, 1u) ||
        motionClassImage->width != motionImage.width ||
        motionClassImage->height != motionImage.height) {
      return Result<AutotestMotionOracleReport, std::string>::makeError(
          "motion oracle class image is missing or incompatible");
    }
  }

  std::vector<std::array<uint32_t, 2>> pixels;
  if (oracle.hasMask) {
    if (oracle.mask.empty()) {
      return Result<AutotestMotionOracleReport, std::string>::makeError(
          "motion oracle mask must select at least one pixel");
    }
    pixels.reserve(oracle.mask.size());
    for (const std::array<uint32_t, 2> &local : oracle.mask) {
      if (local[0] >= oracle.roi.width || local[1] >= oracle.roi.height) {
        return Result<AutotestMotionOracleReport, std::string>::makeError(
            "motion oracle mask pixel is outside the ROI");
      }
      pixels.push_back({oracle.roi.x + local[0], oracle.roi.y + local[1]});
    }
  } else {
    pixels.reserve(static_cast<size_t>(oracle.roi.width) * oracle.roi.height);
    for (uint32_t y = 0u; y < oracle.roi.height; ++y) {
      for (uint32_t x = 0u; x < oracle.roi.width; ++x) {
        pixels.push_back({oracle.roi.x + x, oracle.roi.y + y});
      }
    }
  }

  AutotestMotionOracleReport report{};
  report.motionTarget = oracle.motionTarget;
  report.motionClassTarget = oracle.motionClassTarget;
  report.roi = oracle.roi;
  report.selectedPixelCount = static_cast<uint32_t>(pixels.size());
  report.expectedVelocityPixels = {expectedX, expectedY};
  report.p95ErrorMaxPixels = oracle.p95ErrorMaxPixels;
  report.maxErrorMaxPixels = oracle.maxErrorMaxPixels;

  std::vector<double> vectorErrors;
  std::vector<double> scaleErrors;
  vectorErrors.reserve(pixels.size());
  scaleErrors.reserve(pixels.size());
  double velocityXSum = 0.0;
  double velocityYSum = 0.0;
  uint32_t invalidClassCount = 0u;
  uint32_t staticClassCount = 0u;
  uint32_t fullClassCount = 0u;
  for (const std::array<uint32_t, 2> &pixel : pixels) {
    const size_t motionIndex =
        (static_cast<size_t>(pixel[1]) * motionImage.width + pixel[0]) *
        motionImage.channelCount;
    const double actualX =
        static_cast<double>(motionImage.values[motionIndex]) *
        motionImage.width;
    const double actualY =
        static_cast<double>(motionImage.values[motionIndex + 1u]) *
        motionImage.height;
    if (!std::isfinite(actualX) || !std::isfinite(actualY)) {
      return Result<AutotestMotionOracleReport, std::string>::makeError(
          "motion oracle encountered a non-finite motion sample");
    }
    velocityXSum += actualX;
    velocityYSum += actualY;
    vectorErrors.push_back(
        std::hypot(actualX - expectedX, actualY - expectedY));
    scaleErrors.push_back(
        std::abs(std::hypot(actualX, actualY) - expectedMagnitude));
    if (expectedMagnitude > 1.0e-9 &&
        actualX * expectedX + actualY * expectedY <= 0.0) {
      ++report.wrongSignPixelCount;
    }

    if (motionClassImage != nullptr && !oracle.motionClassTarget.empty()) {
      const size_t classIndex =
          (static_cast<size_t>(pixel[1]) * motionClassImage->width + pixel[0]) *
          motionClassImage->channelCount;
      const double value = motionClassImage->values[classIndex];
      if (!std::isfinite(value)) {
        return Result<AutotestMotionOracleReport, std::string>::makeError(
            "motion oracle encountered a non-finite motion class sample");
      }
      switch (static_cast<int>(std::lround(value * 255.0))) {
      case 0:
        ++invalidClassCount;
        break;
      case 1:
        ++staticClassCount;
        break;
      case 2:
        ++fullClassCount;
        break;
      default:
        break;
      }
    }
  }

  const double sampleCount = static_cast<double>(pixels.size());
  report.meanVelocityPixels = {velocityXSum / sampleCount,
                               velocityYSum / sampleCount};
  report.meanErrorPixels =
      std::accumulate(vectorErrors.begin(), vectorErrors.end(), 0.0) /
      sampleCount;
  report.p95ErrorPixels = nearestRankP95(vectorErrors);
  report.maxErrorPixels =
      *std::max_element(vectorErrors.begin(), vectorErrors.end());
  report.p95ScaleErrorPixels = nearestRankP95(scaleErrors);
  report.maxScaleErrorPixels =
      *std::max_element(scaleErrors.begin(), scaleErrors.end());

  if (report.wrongSignPixelCount != 0u) {
    report.failedThresholds.emplace_back("wrong_sign");
  }
  if (report.p95ErrorPixels > oracle.p95ErrorMaxPixels) {
    report.failedThresholds.emplace_back("p95_error_pixels");
  }
  if (report.maxErrorPixels > oracle.maxErrorMaxPixels) {
    report.failedThresholds.emplace_back("max_error_pixels");
  }
  if (report.p95ScaleErrorPixels > oracle.p95ErrorMaxPixels) {
    report.failedThresholds.emplace_back("p95_scale_error_pixels");
  }
  if (report.maxScaleErrorPixels > oracle.maxErrorMaxPixels) {
    report.failedThresholds.emplace_back("max_scale_error_pixels");
  }

  if (motionClassImage != nullptr && !oracle.motionClassTarget.empty()) {
    report.classCoverageAvailable = true;
    report.classSampleCount = report.selectedPixelCount;
    report.invalidClassCoverage = invalidClassCount / sampleCount;
    report.staticClassCoverage = staticClassCount / sampleCount;
    report.fullClassCoverage = fullClassCount / sampleCount;
    if (oracle.classCoverage.configured) {
      checkCoverageRange(report.invalidClassCoverage,
                         oracle.classCoverage.invalid, "invalid_class",
                         report.failedThresholds);
      checkCoverageRange(report.staticClassCoverage,
                         oracle.classCoverage.staticCameraOnly, "static_class",
                         report.failedThresholds);
      checkCoverageRange(report.fullClassCoverage, oracle.classCoverage.full,
                         "full_class", report.failedThresholds);
    }
  }

  report.status = report.failedThresholds.empty() ? "pass" : "fail";
  report.statusReason = report.failedThresholds.empty()
                            ? "motion_oracle_within_thresholds"
                            : "motion_oracle_thresholds_failed";
  return Result<AutotestMotionOracleReport, std::string>::makeResult(
      std::move(report));
}

} // namespace nuri::tools::autotest
