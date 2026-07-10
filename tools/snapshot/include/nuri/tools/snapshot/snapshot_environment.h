#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace nuri::tools::snapshot {

struct SnapshotEnvironment {
  std::filesystem::path repoRoot{};
  std::string commitHash{};
  std::string branchName{};
  bool dirty = false;
  std::string osName{};
  std::string osVersion{};
  std::string cpuName{};
  uint32_t cpuLogicalThreadCount = 0u;
  std::string gpuBackend{};
  std::string gpuBackendSource{};
  std::string gpuDeviceName{};
  uint32_t gpuVendorId = 0u;
  uint32_t gpuDeviceId = 0u;
  std::string gpuDriverVersion{};
  uint32_t swapchainImageCount = 0u;
  std::string requestedPresentMode{};
  std::string resolvedPresentMode{};
  std::string presentModeSource{};
  std::string requestedWindowMode = "visible";
  std::string resolvedWindowMode = "visible";
  bool windowVisible = true;
  uint32_t renderGraphWorkerCount = 1u;
  bool renderGraphParallelCompile = false;
  bool renderGraphParallelRecording = false;
  std::string buildType{};
  std::string cmakeToolProfile{};
  std::string vcpkgManifestFeatures{};
  bool buildShared = false;
  bool loggingEnabled = false;
  bool assertsEnabled = false;
  bool tracyEnabled = false;
  bool tracyGpuEnabled = false;
  bool tracyGpuDrawZonesEnabled = false;
  bool devChecks = false;
};

[[nodiscard]] std::filesystem::path snapshotRepoRoot();
[[nodiscard]] std::string readProcessEnvironment(std::string_view name);
[[nodiscard]] SnapshotEnvironment collectSnapshotEnvironment(
    std::string_view backend, std::string_view backendSource,
    std::string_view requestedPresentMode, std::string_view presentModeSource,
    std::string_view requestedWindowMode, std::string_view resolvedWindowMode);
[[nodiscard]] std::string joinCommandLine(int argc, char **argv);
[[nodiscard]] std::string utcTimestampIso8601();
[[nodiscard]] std::string utcTimestampForPath();

} // namespace nuri::tools::snapshot
