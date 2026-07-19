#include "nuri/resources/storage/material/material_cache_utils.h"
#include "nuri/pch.h"
#include "nuri/resources/storage/material/material_binary_format.h"
namespace nuri {

Result<SceneMaterialCacheKey, std::string>
buildSceneMaterialCacheKey(const std::filesystem::path &sourcePath) {
  SceneMaterialCacheKey key{};
  key.normalizedSourcePath = normalizeSourcePath(sourcePath);
  const std::string normalizedString =
      key.normalizedSourcePath.generic_string();
  if (normalizedString.empty()) {
    return Result<SceneMaterialCacheKey, std::string>::makeError(
        "buildSceneMaterialCacheKey: normalized source path is empty");
  }
  Fnv1a64 hash{14695981039346656037ull};
  hash.add(normalizedString);
  key.sourcePathHash = hash.value();
  std::filesystem::path cacheDir =
      key.normalizedSourcePath.parent_path() / ".nuri_scene_cache";
  std::string stem = key.normalizedSourcePath.stem().string();
  if (stem.empty()) {
    stem = "scene";
  }
  key.cachePath =
      cacheDir / std::format("{}_{}_v{}.nmat", stem, hexU64(key.sourcePathHash),
                             kMaterialBinaryFormatMajorVersion);
  return Result<SceneMaterialCacheKey, std::string>::makeResult(std::move(key));
}
} // namespace nuri
