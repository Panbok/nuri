#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/mesh_importer.h"

namespace nuri {

struct MeshSourceFingerprint {
  bool exists = false;
  uint64_t sizeBytes = 0;
  int64_t mtimeNs = 0;
};

struct MeshCacheKey {
  std::filesystem::path normalizedSourcePath;
  std::filesystem::path cachePath;
  uint64_t sourcePathHash = 0;
  uint64_t optionsHash = 0;
};

[[nodiscard]] NURI_API std::filesystem::path
normalizeMeshSourcePath(const std::filesystem::path &path);

[[nodiscard]] NURI_API uint64_t
hashMeshImportOptions(const MeshImportOptions &options);

[[nodiscard]] NURI_API Result<MeshCacheKey, std::string>
buildMeshCacheKey(const std::filesystem::path &sourcePath,
                  const MeshImportOptions &options);

[[nodiscard]] NURI_API MeshSourceFingerprint
queryMeshSourceFingerprint(const std::filesystem::path &sourcePath);

[[nodiscard]] NURI_API Result<std::vector<std::byte>, std::string>
readBinaryFile(const std::filesystem::path &path);

[[nodiscard]] NURI_API Result<bool, std::string>
writeBinaryFileAtomic(const std::filesystem::path &path,
                      std::span<const std::byte> bytes);

} // namespace nuri
