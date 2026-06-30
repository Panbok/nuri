#include "nuri/tools/autotest/autotest_environment.h"

namespace nuri::tools::autotest {

std::filesystem::path autotestRepoRoot() {
  return nuri::tools::snapshot::snapshotRepoRoot();
}

std::string utcTimestampForPath() {
  return nuri::tools::snapshot::utcTimestampForPath();
}

std::string utcTimestampIso8601() {
  return nuri::tools::snapshot::utcTimestampIso8601();
}

std::string readProcessEnvironment(std::string_view name) {
  return nuri::tools::snapshot::readProcessEnvironment(name);
}

AutotestEnvironment collectAutotestEnvironment(
    std::string_view backend, std::string_view backendSource,
    std::string_view presentMode, std::string_view presentSource,
    std::string_view requestedWindowMode, std::string_view resolvedWindowMode) {
  return nuri::tools::snapshot::collectSnapshotEnvironment(
      backend, backendSource, presentMode, presentSource, requestedWindowMode,
      resolvedWindowMode);
}

std::string joinCommandLine(int argc, char **argv) {
  return nuri::tools::snapshot::joinCommandLine(argc, argv);
}

} // namespace nuri::tools::autotest
