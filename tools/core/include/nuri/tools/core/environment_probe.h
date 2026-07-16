#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace nuri::tools::core {

struct EnvironmentHostFacts {
  std::filesystem::path repoRoot{};
  std::string commitHash = "unknown";
  std::string branchName = "unknown";
  bool dirty = false;
  std::string osName = "unknown";
  std::string osVersion = "unknown";
  std::string cpuName = "unknown";
  uint32_t cpuLogicalThreadCount = 0u;
};

struct EnvironmentBuildFacts {
  std::string buildType = "unknown";
  std::string cmakeToolProfile = "unknown";
  std::string vcpkgManifestFeatures{};
  bool buildShared = false;
  bool loggingEnabled = false;
  bool assertsEnabled = false;
  bool tracyEnabled = false;
  bool devChecks = false;
};

struct EnvironmentProbeResult {
  EnvironmentHostFacts host{};
  EnvironmentBuildFacts build{};
};

// Injectable platform operations keep policy deterministic in tests while the
// production source below owns all OS-specific probing.
struct EnvironmentProbeSource {
  std::function<std::optional<std::string>(std::string_view,
                                           const std::filesystem::path &)>
      runCommand{};
  std::function<std::optional<std::string>()> osName{};
  std::function<std::optional<std::string>()> osVersion{};
  std::function<std::optional<std::string>()> cpuName{};
  std::function<uint32_t()> cpuLogicalThreadCount{};
};

[[nodiscard]] EnvironmentProbeSource defaultEnvironmentProbeSource();
[[nodiscard]] EnvironmentProbeResult
collectEnvironmentProbe(const std::filesystem::path &repoRoot,
                        EnvironmentBuildFacts build);
[[nodiscard]] EnvironmentProbeResult
collectEnvironmentProbe(const std::filesystem::path &repoRoot,
                        EnvironmentBuildFacts build,
                        const EnvironmentProbeSource &source);

[[nodiscard]] std::string readEnvironmentVariable(std::string_view name);
[[nodiscard]] std::string joinCommandLine(int argc, char **argv);
[[nodiscard]] std::string utcTimestampIso8601();
[[nodiscard]] std::string utcTimestampForPath();

} // namespace nuri::tools::core
