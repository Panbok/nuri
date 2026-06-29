#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/snapshot/snapshot_case.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::snapshot {

struct SnapshotManifestLoadOptions {
  std::filesystem::path caseRoot{};
  std::filesystem::path repoRoot{};
};

[[nodiscard]] std::filesystem::path defaultSnapshotCaseRoot();
[[nodiscard]] Result<SnapshotCase, std::string>
loadSnapshotCaseManifest(const std::filesystem::path &path);
[[nodiscard]] Result<std::vector<SnapshotCase>, std::string>
discoverSnapshotCases(const SnapshotManifestLoadOptions &options = {});
[[nodiscard]] const SnapshotCase *
findSnapshotCaseById(const std::vector<SnapshotCase> &cases,
                     std::string_view id);
[[nodiscard]] std::vector<const SnapshotCase *>
filterSnapshotCasesBySuite(const std::vector<SnapshotCase> &cases,
                           std::string_view suite);
[[nodiscard]] Result<std::filesystem::path, std::string>
resolveSnapshotPath(std::string_view base, const std::filesystem::path &path);

} // namespace nuri::tools::snapshot

