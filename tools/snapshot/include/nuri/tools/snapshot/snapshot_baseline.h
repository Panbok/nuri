#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/snapshot/snapshot_report.h"

#include <filesystem>
#include <string>

namespace nuri::tools::snapshot {

struct SnapshotBaselineLookup {
  std::filesystem::path profileDir{};
  std::filesystem::path caseDir{};
};

[[nodiscard]] std::filesystem::path defaultSnapshotBaselineRoot();
[[nodiscard]] SnapshotBaselineLookup
snapshotBaselineLookup(const SnapshotCase &snapshotCase,
                       std::string_view baselineProfile);
[[nodiscard]] Result<bool, std::string>
approveSnapshotBaselines(const SnapshotReport &report,
                         std::string_view baselineProfile,
                         std::string_view reason);

} // namespace nuri::tools::snapshot

