#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/snapshot/snapshot_image.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::snapshot {

struct SnapshotCompareProfile {
  std::string id = "exact";
  bool valid = true;
  double maxAbsError = 0.0;
  double meanAbsError = 0.0;
  double rmse = 0.0;
  double p99AbsError = 0.0;
  uint64_t maxFailingValues = 0u;
};

struct SnapshotCompareMetrics {
  double meanAbsError = 0.0;
  double rmse = 0.0;
  double maxAbsError = 0.0;
  double p99AbsError = 0.0;
  uint64_t failingValues = 0u;
  uint64_t comparedValues = 0u;
};

struct SnapshotSemanticMetrics {
  std::string unit = "absolute";
  double meanError = 0.0;
  double maxError = 0.0;
  uint64_t validPixels = 0u;
  uint64_t ignoredPixels = 0u;
  uint64_t changedPixels = 0u;
  bool changedBoundsValid = false;
  uint32_t minChangedX = 0u;
  uint32_t minChangedY = 0u;
  uint32_t maxChangedX = 0u;
  uint32_t maxChangedY = 0u;
  uint32_t maxErrorX = 0u;
  uint32_t maxErrorY = 0u;
  double rmse = 0.0;
  double p99Error = 0.0;
  uint64_t failingPixels = 0u;
  std::string secondaryUnit{};
  double meanSecondaryError = 0.0;
  double maxSecondaryError = 0.0;
  double secondaryRmse = 0.0;
  double p99SecondaryError = 0.0;
  uint64_t secondaryFailingPixels = 0u;
  uint64_t truePositivePixels = 0u;
  uint64_t trueNegativePixels = 0u;
  uint64_t falsePositivePixels = 0u;
  uint64_t falseNegativePixels = 0u;
  double intersectionOverUnion = 1.0;
};

struct SnapshotCompareRegion {
  uint32_t x = 0u;
  uint32_t y = 0u;
  uint32_t width = 0u;
  uint32_t height = 0u;
};

struct SnapshotCompareOptions {
  SnapshotCompareRegion roi{};
  std::span<const uint8_t> validityMask{};
};

struct SnapshotCompareResult {
  bool compatible = true;
  bool passed = true;
  SnapshotCompareMetrics metrics{};
  SnapshotSemanticMetrics semantic{};
  std::vector<std::string> failedThresholds{};
  std::vector<std::string> errors{};
};

[[nodiscard]] SnapshotCompareProfile
builtinSnapshotCompareProfile(std::string_view id);
[[nodiscard]] bool
isBuiltinSnapshotCompareProfile(std::string_view id) noexcept;
[[nodiscard]] bool
snapshotCompareProfileSupportsKind(std::string_view profile,
                                   RenderCaptureValueKind kind) noexcept;
[[nodiscard]] SnapshotCompareResult
compareSnapshotImages(const SnapshotImage &actual,
                      const SnapshotImage &expected,
                      const SnapshotCompareProfile &profile,
                      const SnapshotCompareOptions &options = {});
[[nodiscard]] Result<bool, std::string>
writeSnapshotDiffPng(const SnapshotImage &actual, const SnapshotImage &expected,
                     const std::filesystem::path &path);
[[nodiscard]] Result<bool, std::string>
writeSnapshotComparisonFile(const SnapshotCompareResult &comparison,
                            std::string_view profile,
                            const std::filesystem::path &path);

} // namespace nuri::tools::snapshot
