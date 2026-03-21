#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace nuri {

struct SceneSourceFingerprint {
  bool exists = false;
  uint64_t sizeBytes = 0;
  int64_t mtimeNs = 0;
};

struct SceneMaterialCacheKey {
  std::filesystem::path normalizedSourcePath;
  std::filesystem::path cachePath;
  uint64_t sourcePathHash = 0;
};

[[nodiscard]] NURI_API Result<SceneMaterialCacheKey, std::string>
buildSceneMaterialCacheKey(const std::filesystem::path &sourcePath);

[[nodiscard]] NURI_API SceneSourceFingerprint
querySceneSourceFingerprint(const std::filesystem::path &sourcePath);

[[nodiscard]] NURI_API uint64_t
hashScenePath(const std::filesystem::path &sourcePath);

} // namespace nuri
