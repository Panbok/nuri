#pragma once
#include "nuri/resources/storage/cache_utils.h"
namespace nuri {

struct SceneMaterialCacheKey {
  std::filesystem::path normalizedSourcePath;
  std::filesystem::path cachePath;
  uint64_t sourcePathHash = 0;
};

[[nodiscard]] NURI_API Result<SceneMaterialCacheKey, std::string>
buildSceneMaterialCacheKey(const std::filesystem::path &sourcePath);

} // namespace nuri
