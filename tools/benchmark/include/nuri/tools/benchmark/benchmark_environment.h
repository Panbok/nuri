#pragma once

#include "nuri/core/result.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkEnvironment {
  std::filesystem::path repoRoot{};
  std::string commitHash = "unknown";
  std::string branchName = "unknown";
  bool dirty = false;
  std::string osName = "unknown";
  std::string osVersion = "unknown";
  std::string cpuName = "unknown";
  uint32_t cpuLogicalThreadCount = 0u;
  std::string gpuBackend = "unknown";
  std::string gpuBackendSource = "default";
  std::string gpuDeviceName = "unknown";
  uint32_t gpuVendorId = 0u;
  uint32_t gpuDeviceId = 0u;
  std::string gpuDriverVersion = "unknown";
  uint32_t swapchainImageCount = 0u;
  std::string requestedPresentMode = "default";
  std::string resolvedPresentMode = "unknown";
  std::string presentModeSource = "default";
  std::string windowMode = "windowed";
  bool windowVisible = true;
  uint32_t renderGraphWorkerCount = 1u;
  bool renderGraphParallelCompile = false;
  bool renderGraphParallelRecording = false;
  std::string renderGraphWorkerCountSource = "manifest";
  std::string renderGraphParallelCompileSource = "manifest";
  std::string renderGraphParallelRecordingSource = "manifest";
  std::string buildType = "unknown";
  std::string cmakeToolProfile = "unknown";
  std::string vcpkgManifestFeatures = "";
  bool buildShared = false;
  bool loggingEnabled = false;
  bool assertsEnabled = false;
  bool tracyEnabled = false;
  bool tracyDiagnostic = false;
  bool devChecks = false;
};

[[nodiscard]] std::filesystem::path benchmarkRepoRoot();
[[nodiscard]] BenchmarkEnvironment collectBenchmarkEnvironment(
    std::string_view backend, std::string_view backendSource,
    std::string_view requestedPresentMode, std::string_view presentModeSource,
    bool tracyDiagnostic);
[[nodiscard]] std::string joinCommandLine(int argc, char **argv);
[[nodiscard]] std::string utcTimestampForPath();
[[nodiscard]] std::string utcTimestampIso8601();
[[nodiscard]] std::string readProcessEnvironment(std::string_view name);

} // namespace nuri::tools::benchmark
