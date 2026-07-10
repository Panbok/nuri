#pragma once

#include "nuri/core/result.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::core {

struct BaselineProfileAuthority {
  bool authoritative = false;
  bool allowDirtyTree = false;
  std::string reason{};
};

struct BaselineProfileEnvironment {
  std::string os{};
  std::string backend{};
  std::string windowMode{};
  std::optional<uint32_t> gpuVendorId{};
  std::optional<uint32_t> gpuDeviceId{};
  std::optional<std::string> driver{};
};

struct BaselineProfileExecution {
  std::string presentMode{};
  std::string profiling{};
  bool devChecks = false;
};

struct BaselineProfileBenchmarkPolicy {
  uint32_t minimumRepetitions = 0u;
  std::string warmupStability{};
  uint32_t warmupWindowFrames = 0u;
  double warmupMaxDriftPercent = 0.0;
  std::vector<std::string> requiredMetrics{};
  std::string thresholdOwnership{};
};

struct BaselineProfile {
  uint32_t schemaVersion = 1u;
  std::string kind = "nuri.baseline.profile";
  std::string id{};
  std::string description{};
  BaselineProfileAuthority authority{};
  BaselineProfileEnvironment environment{};
  BaselineProfileExecution execution{};
  BaselineProfileBenchmarkPolicy benchmarkPolicy{};
  std::filesystem::path sourcePath{};
};

struct BaselineProfileObservedEnvironment {
  std::string os{};
  std::string backend{};
  std::string backendSource{};
  std::string windowMode{};
  bool windowVisible = true;
  uint32_t gpuVendorId = 0u;
  uint32_t gpuDeviceId = 0u;
  std::string driver{};
  std::string presentMode{};
  std::string profiling{};
  bool devChecks = false;
  bool dirtyTree = false;
};

struct BaselineProfileCompatibility {
  bool compatible = false;
  std::vector<std::string> reasons{};
};

[[nodiscard]] Result<BaselineProfile, std::string>
loadBaselineProfile(const std::filesystem::path &profilesDirectory,
                    std::string_view profileId);
[[nodiscard]] BaselineProfileCompatibility
evaluateBaselineProfile(const BaselineProfile &profile,
                        const BaselineProfileObservedEnvironment &observed);

} // namespace nuri::tools::core
