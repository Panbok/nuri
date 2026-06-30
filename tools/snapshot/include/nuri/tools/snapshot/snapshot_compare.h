#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/snapshot/snapshot_image.h"

#include <filesystem>
#include <string>
#include <vector>

namespace nuri::tools::snapshot {

struct SnapshotCompareProfile {
  std::string id = "exact";
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

struct SnapshotCompareResult {
  bool compatible = true;
  bool passed = true;
  SnapshotCompareMetrics metrics{};
  std::vector<std::string> failedThresholds{};
  std::vector<std::string> errors{};
};

[[nodiscard]] SnapshotCompareProfile
builtinSnapshotCompareProfile(std::string_view id);
[[nodiscard]] SnapshotCompareResult
compareSnapshotImages(const SnapshotImage &actual,
                      const SnapshotImage &expected,
                      const SnapshotCompareProfile &profile);
[[nodiscard]] Result<bool, std::string>
writeSnapshotDiffPng(const SnapshotImage &actual, const SnapshotImage &expected,
                     const std::filesystem::path &path);

} // namespace nuri::tools::snapshot
