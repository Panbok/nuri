#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>

namespace nuri::bakery {

enum class BakeJobKind : uint8_t { BrdfLut, EnvmapPrefilter };

enum class BakeJobState : uint8_t {
  Queued,
  CacheCheck,
  GpuSetup,
  GpuStep,
  WriteQueued,
  WriteInFlight,
  Succeeded,
  Skipped,
  Failed,
  Canceled
};

enum class BakeryExecutionProfile : uint8_t { Interactive, Balanced, Fast };

struct BakeJobId {
  uint64_t value = 0;
};

struct BrdfLutBakeRequest {
  bool forceRebuild = false;
};

struct EnvmapPrefilterBakeRequest {
  std::filesystem::path environmentHdrPath;
  bool forceRebuild = false;
};

using BakeRequest = std::variant<BrdfLutBakeRequest, EnvmapPrefilterBakeRequest>;

struct BakeJobSnapshot {
  BakeJobId id{};
  BakeJobKind kind = BakeJobKind::BrdfLut;
  BakeJobState state = BakeJobState::Queued;
  uint32_t completedSteps = 0;
  uint32_t totalSteps = 0;
  float progress01 = 0.0f;
  std::string summary;
  std::string error;
};

} // namespace nuri::bakery
