#pragma once

#include "nuri/tools/snapshot/snapshot_environment.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace nuri::tools::autotest {

using AutotestEnvironment = nuri::tools::snapshot::SnapshotEnvironment;

[[nodiscard]] std::filesystem::path autotestRepoRoot();
[[nodiscard]] std::string utcTimestampForPath();
[[nodiscard]] std::string utcTimestampIso8601();
[[nodiscard]] std::string readProcessEnvironment(std::string_view name);
[[nodiscard]] AutotestEnvironment collectAutotestEnvironment(
    std::string_view backend, std::string_view backendSource,
    std::string_view presentMode, std::string_view presentSource,
    std::string_view requestedWindowMode, std::string_view resolvedWindowMode);
[[nodiscard]] std::string joinCommandLine(int argc, char **argv);

} // namespace nuri::tools::autotest
