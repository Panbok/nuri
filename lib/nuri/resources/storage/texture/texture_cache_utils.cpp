#include "nuri/pch.h"

#include "nuri/resources/storage/texture/texture_cache_utils.h"

#include "nuri/resources/storage/mesh/mesh_cache_utils.h"

namespace nuri {
namespace {

constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
constexpr size_t kBasisSuffixLen = sizeof("_basis") - 1u;
constexpr size_t kUastcSuffixLen = sizeof("_uastc") - 1u;

void fnv1aAddBytes(uint64_t &hash, std::span<const std::byte> bytes) {
  for (const std::byte value : bytes) {
    hash ^= static_cast<uint8_t>(value);
    hash *= kFnvPrime;
  }
}

void fnv1aAddU32(uint64_t &hash, uint32_t value) {
  fnv1aAddBytes(hash,
                {reinterpret_cast<const std::byte *>(&value), sizeof(value)});
}

void fnv1aAddString(uint64_t &hash, std::string_view value) {
  fnv1aAddBytes(
      hash, {reinterpret_cast<const std::byte *>(value.data()), value.size()});
}

std::string hexU64(uint64_t value) { return std::format("{:016x}", value); }

} // namespace

uint64_t hashSceneTextureSourceIdentity(std::string_view sceneCanonicalPath,
                                        const MaterialTextureSlotData &slotData,
                                        bool srgb, uint32_t bakeSettingsTag) {
  uint64_t hash = kFnvOffsetBasis;
  fnv1aAddString(hash, sceneCanonicalPath);
  fnv1aAddU32(hash, kSceneTextureBakeSettingsVersion);
  fnv1aAddU32(hash, static_cast<uint32_t>(slotData.sourceKind));
  fnv1aAddU32(hash, slotData.embeddedIndex);
  fnv1aAddU32(hash, srgb ? 1u : 0u);
  fnv1aAddU32(hash, bakeSettingsTag);
  fnv1aAddString(hash, slotData.path);
  return hash;
}

Result<std::filesystem::path, std::string>
buildPortableTextureCachePath(const std::filesystem::path &sceneSourcePath,
                              uint64_t sourceIdentityHash) {
  const std::filesystem::path normalized =
      normalizeMeshSourcePath(sceneSourcePath);
  if (normalized.empty()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "buildPortableTextureCachePath: normalized source path is empty");
  }
  return Result<std::filesystem::path, std::string>::makeResult(
      normalized.parent_path() / ".nuri_scene_cache" / "textures" /
      (hexU64(sourceIdentityHash) + "_basis.ktx2"));
}

std::string_view textureFormatCacheSuffix(Format format) noexcept {
  switch (format) {
  case Format::BC7_RGBA_UNORM:
    return "bc7";
  case Format::BC7_RGBA_SRGB:
    return "bc7_srgb";
  case Format::ETC2_RGB8_UNORM:
    return "etc2";
  case Format::ETC2_RGB8_SRGB:
    return "etc2_srgb";
  case Format::RGBA8_UNORM:
    return "rgba8";
  case Format::RGBA8_SRGB:
    return "rgba8_srgb";
  default:
    return "native";
  }
}

std::filesystem::path
buildNativeTextureCachePath(const std::filesystem::path &portableTexturePath,
                            Format targetFormat) {
  const std::filesystem::path parent = portableTexturePath.parent_path();
  const std::filesystem::path sceneCacheDir = parent.parent_path();
  std::string stem = portableTexturePath.stem().string();
  if (stem.ends_with("_basis")) {
    stem.resize(stem.size() - kBasisSuffixLen);
  } else if (stem.ends_with("_uastc")) {
    stem.resize(stem.size() - kUastcSuffixLen);
  }
  return sceneCacheDir / "native_textures" /
         std::format("{}_{}_v1.ktx2", stem,
                     textureFormatCacheSuffix(targetFormat));
}

bool isTextureCacheUpToDate(const std::filesystem::path &cachePath,
                            const std::filesystem::path &sourcePath) noexcept {
  std::error_code ec;
  if (!std::filesystem::exists(cachePath, ec) || ec) {
    return false;
  }
  const auto cacheTime = std::filesystem::last_write_time(cachePath, ec);
  if (ec) {
    return false;
  }
  const auto sourceTime = std::filesystem::last_write_time(sourcePath, ec);
  if (ec) {
    return false;
  }
  return cacheTime >= sourceTime;
}

} // namespace nuri
