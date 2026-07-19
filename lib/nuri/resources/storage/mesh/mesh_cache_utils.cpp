#include "nuri/resources/storage/mesh/mesh_cache_utils.h"
#include "nuri/pch.h"
#include "nuri/resources/storage/mesh/mesh_binary_format.h"
namespace nuri {
namespace {
constexpr uint32_t kMeshCacheContentVersion = 14u;
}

uint64_t hashMeshImportOptions(const MeshImportOptions &options) {
  Fnv1a64 hash;
  hash.add(kMeshCacheContentVersion);
  for (const bool flag :
       {options.triangulate, options.genNormals, options.calcTangents,
        options.flipUVs, options.joinIdenticalVertices, options.genUVCoords,
        options.removeRedundantMaterials, options.limitBoneWeights,
        options.optimize, options.generateLods, options.generateMeshlets}) {
    hash.add(static_cast<uint8_t>(flag));
  }
  hash.add(options.lodCount);
  hash.add(options.meshletMaxVertices);
  hash.add(options.meshletMaxPrimitives);
  hash.add(std::bit_cast<uint32_t>(options.meshletConeWeight));
  const uint32_t lodRatioCount =
      static_cast<uint32_t>(options.lodTriangleRatios.size());
  hash.add(lodRatioCount);
  for (const float ratio : options.lodTriangleRatios) {
    hash.add(std::bit_cast<uint32_t>(ratio));
  }
  hash.add(std::bit_cast<uint32_t>(options.lodTargetError));
  return hash.value();
}

Result<MeshCacheKey, std::string>
buildMeshCacheKey(const std::filesystem::path &sourcePath,
                  const MeshImportOptions &options) {
  MeshCacheKey key{};
  key.normalizedSourcePath = normalizeSourcePath(sourcePath);
  const std::string normalizedString =
      key.normalizedSourcePath.generic_string();
  if (normalizedString.empty()) {
    return Result<MeshCacheKey, std::string>::makeError(
        "buildMeshCacheKey: normalized source path is empty");
  }
  Fnv1a64 pathHash;
  pathHash.add(normalizedString);
  key.sourcePathHash = pathHash.value();
  key.optionsHash = hashMeshImportOptions(options);
  std::filesystem::path cacheDir =
      key.normalizedSourcePath.parent_path() / ".nuri_mesh_cache";
  std::string stem = key.normalizedSourcePath.stem().string();
  if (stem.empty()) {
    stem = "mesh";
  }
  const std::string fileName =
      std::format("{}_{}_{}_v{}.nmesh", stem, hexU64(key.sourcePathHash),
                  hexU64(key.optionsHash), kMeshBinaryFormatMajorVersion);
  key.cachePath = cacheDir / fileName;
  return Result<MeshCacheKey, std::string>::makeResult(std::move(key));
}

} // namespace nuri
