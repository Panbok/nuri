#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/snapshot/snapshot_report.h"

#include <filesystem>
#include <string>
#include <vector>

namespace nuri::tools::snapshot {

struct SnapshotBaselineLookup {
  std::filesystem::path profileDir{};
  std::filesystem::path caseDir{};
};

struct SnapshotBaselinePlanEntry {
  std::string path{};
  std::string operation{};
  std::string previousDigest{};
  std::string sourceDigest{};
};

struct SnapshotBaselinePlan {
  std::string caseId{};
  std::string suite{};
  std::string profileId{};
  std::string reason{};
  std::string actor{};
  std::string sourceCommit{};
  std::string sourceReportDigest{};
  std::string digest{};
  std::vector<SnapshotBaselinePlanEntry> entries{};
};

[[nodiscard]] std::filesystem::path defaultSnapshotBaselineRoot();
[[nodiscard]] SnapshotBaselineLookup
snapshotBaselineLookup(const SnapshotCase &snapshotCase,
                       std::string_view baselineProfile);
[[nodiscard]] Result<std::string, std::string>
inspectSnapshotBaseline(const SnapshotCase &snapshotCase,
                        std::string_view baselineProfile,
                        const std::filesystem::path &baselineRoot = {});
[[nodiscard]] Result<bool, std::string>
verifySnapshotBaseline(const SnapshotCase &snapshotCase,
                       std::string_view baselineProfile,
                       const std::filesystem::path &baselineRoot = {});
[[nodiscard]] Result<SnapshotBaselinePlan, std::string>
planSnapshotBaselines(const SnapshotReport &report,
                      std::string_view baselineProfile, std::string_view reason,
                      std::string_view actor,
                      const std::filesystem::path &baselineRoot = {});
[[nodiscard]] Result<std::string, std::string>
writeSnapshotBaselinePlanJson(const SnapshotBaselinePlan &plan);
[[nodiscard]] Result<bool, std::string> approveSnapshotBaselines(
    const SnapshotReport &report, std::string_view baselineProfile,
    std::string_view reason, std::string_view confirmPlanDigest = {},
    std::string_view actor = {},
    const std::filesystem::path &baselineRoot = {});

} // namespace nuri::tools::snapshot
