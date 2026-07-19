#pragma once
#include "nuri/defines.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/storage/cache_utils.h"
namespace nuri {

struct MeshCacheKey {
  std::filesystem::path normalizedSourcePath;
  std::filesystem::path cachePath;
  uint64_t sourcePathHash = 0;
  uint64_t optionsHash = 0;
};

[[nodiscard]] NURI_API uint64_t
hashMeshImportOptions(const MeshImportOptions &options);

[[nodiscard]] NURI_API Result<MeshCacheKey, std::string>
buildMeshCacheKey(const std::filesystem::path &sourcePath,
                  const MeshImportOptions &options);

} // namespace nuri
