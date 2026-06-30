#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/snapshot/snapshot_report.h"

#include <filesystem>
#include <span>
#include <string>

namespace nuri::tools::snapshot {

[[nodiscard]] Result<std::string, std::string>
writeSnapshotHtmlReport(const SnapshotReport &report);
[[nodiscard]] Result<bool, std::string>
writeSnapshotHtmlReportFile(const SnapshotReport &report,
                            const std::filesystem::path &path);
[[nodiscard]] Result<std::string, std::string>
writeSnapshotSuiteHtml(std::span<const SnapshotReport> reports,
                       std::string_view suite);
[[nodiscard]] Result<bool, std::string>
writeSnapshotSuiteHtmlFile(std::span<const SnapshotReport> reports,
                           std::string_view suite,
                           const std::filesystem::path &path);

} // namespace nuri::tools::snapshot
