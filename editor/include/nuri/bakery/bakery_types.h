#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace nuri::bakery {

enum class BakeJobKind : uint8_t {
  BrdfLut,
  EnvmapPrefilter,
  SceneTextureArtifacts
};

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

enum class SceneTextureArtifactTarget : uint8_t { BC7, ETC2, RGBA8 };

struct SceneTextureArtifactsBakeRequest {
  std::filesystem::path scenePath;
  // Empty means "prebuild no native targets"; only listed targets are baked.
  std::vector<SceneTextureArtifactTarget> prebuildNativeTargets{};
  bool forceRebuild = false;
};

using BakeRequest = std::variant<BrdfLutBakeRequest, EnvmapPrefilterBakeRequest,
                                 SceneTextureArtifactsBakeRequest>;

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
