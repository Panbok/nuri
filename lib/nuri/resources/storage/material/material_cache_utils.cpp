#include "nuri/pch.h"

#include "nuri/resources/storage/material/material_cache_utils.h"

#include "nuri/resources/storage/material/material_binary_format.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"

namespace nuri {
namespace {

constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

void fnv1aAddBytes(uint64_t &hash, std::span<const std::byte> bytes) {
  for (const std::byte value : bytes) {
    hash ^= static_cast<uint8_t>(value);
    hash *= kFnvPrime;
  }
}

std::string hexU64(uint64_t value) { return std::format("{:016x}", value); }

} // namespace

uint64_t hashScenePath(const std::filesystem::path &sourcePath) {
  const std::filesystem::path normalized = normalizeMeshSourcePath(sourcePath);
  const std::string normalizedString = normalized.generic_string();
  uint64_t hash = kFnvOffsetBasis;
  fnv1aAddBytes(hash,
                {reinterpret_cast<const std::byte *>(normalizedString.data()),
                 normalizedString.size()});
  return hash;
}

Result<SceneMaterialCacheKey, std::string>
buildSceneMaterialCacheKey(const std::filesystem::path &sourcePath) {
  SceneMaterialCacheKey key{};
  key.normalizedSourcePath = normalizeMeshSourcePath(sourcePath);
  const std::string normalizedString =
      key.normalizedSourcePath.generic_string();
  if (normalizedString.empty()) {
    return Result<SceneMaterialCacheKey, std::string>::makeError(
        "buildSceneMaterialCacheKey: normalized source path is empty");
  }
  key.sourcePathHash = hashScenePath(key.normalizedSourcePath);
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

SceneSourceFingerprint
querySceneSourceFingerprint(const std::filesystem::path &sourcePath) {
  const MeshSourceFingerprint meshFingerprint =
      queryMeshSourceFingerprint(sourcePath);
  return SceneSourceFingerprint{
      .exists = meshFingerprint.exists,
      .sizeBytes = meshFingerprint.sizeBytes,
      .mtimeNs = meshFingerprint.mtimeNs,
  };
}

} // namespace nuri
