#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/autotest/autotest_case.h"
#include "nuri/tools/snapshot/snapshot_image.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace nuri::tools::autotest {

struct AutotestMotionOracleReport {
  std::string status = "not_run";
  std::string statusReason = "not_run";
  std::string motionTarget{};
  std::string motionClassTarget{};
  AutotestPixelRoi roi{};
  uint32_t selectedPixelCount = 0u;
  std::array<double, 2> expectedVelocityPixels{};
  std::array<double, 2> meanVelocityPixels{};
  double meanErrorPixels = 0.0;
  double p95ErrorPixels = 0.0;
  double maxErrorPixels = 0.0;
  double p95ScaleErrorPixels = 0.0;
  double maxScaleErrorPixels = 0.0;
  uint32_t wrongSignPixelCount = 0u;
  bool classCoverageAvailable = false;
  uint32_t classSampleCount = 0u;
  double invalidClassCoverage = 0.0;
  double staticClassCoverage = 0.0;
  double fullClassCoverage = 0.0;
  double p95ErrorMaxPixels = 0.0;
  double maxErrorMaxPixels = 0.0;
  std::vector<std::string> failedThresholds{};
};

[[nodiscard]] Result<AutotestMotionOracleReport, std::string>
evaluateAutotestMotionOracle(
    const AutotestMotionOracle &oracle,
    const nuri::tools::snapshot::SnapshotImage &motionImage,
    const nuri::tools::snapshot::SnapshotImage *motionClassImage = nullptr);

} // namespace nuri::tools::autotest
