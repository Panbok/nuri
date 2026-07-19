#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/resources/cpu/material_data.h"
#include <filesystem>
#include <string_view>
namespace nuri {
constexpr uint32_t kSceneTextureArtifactSettingsVersion = 14u;
constexpr uint32_t kNativeTextureArtifactProfileVersion = 3u;
enum class TextureArtifactTranscodeFormat : uint8_t {
  BC7_RGBA,
  ETC1_RGB,
  RGBA32
};
struct TextureSourceFingerprint {
  uint64_t sizeBytes = 0u;
  int64_t writeTimeTicks = 0;
  constexpr bool
  operator==(const TextureSourceFingerprint &) const noexcept = default;
};
struct NativeTextureCacheMetadata {
  uint32_t profileVersion = kNativeTextureArtifactProfileVersion;
  uint64_t sourceIdentityHash = 0u;
  TextureSourceFingerprint source{};
  Format targetFormat = Format::RGBA8_UNORM;
  TextureType textureType = TextureType::Texture2D;
  TextureDimensions dimensions{};
  uint32_t numLayers = 1u;
  uint32_t numFaces = 1u;
  uint32_t numMipLevels = 1u;
  uint64_t payloadSizeBytes = 0u;
  uint64_t artifactSizeBytes = 0u;
};
enum class NativeTextureCacheProbeStatus : uint8_t {
  Hit,
  Missing,
  Stale,
  Corrupt
};
struct NativeTextureCacheProbe {
  NativeTextureCacheProbeStatus status = NativeTextureCacheProbeStatus::Missing;
  NativeTextureCacheMetadata metadata{};
  std::string message{};
};
[[nodiscard]] NURI_API uint64_t
hashSceneTextureSourceIdentity(std::string_view sceneCanonicalPath,
                               const MaterialTextureSlotData &slotData,
                               bool srgb, uint32_t bakeSettingsTag = 0u);
[[nodiscard]] NURI_API uint64_t
hashTextureSourceIdentity(std::string_view sourceCanonicalPath, bool srgb,
                          uint32_t processingTag = 0u);
[[nodiscard]] NURI_API Result<std::filesystem::path, std::string>
buildNativeTextureArtifactPath(const std::filesystem::path &sourcePath,
                               uint64_t sourceIdentityHash,
                               Format targetFormat);
[[nodiscard]] NURI_API std::filesystem::path
buildNativeTextureCacheMetadataPath(
    const std::filesystem::path &nativeTexturePath);
[[nodiscard]] NURI_API Format
selectTextureArtifactTargetFormat(bool bc7Supported, bool etc2Supported,
                                  bool srgb, uint32_t componentCount) noexcept;
[[nodiscard]] NURI_API bool
shouldPersistNativeTextureArtifact(Format format) noexcept;
[[nodiscard]] NURI_API Result<TextureArtifactTranscodeFormat, std::string>
resolveTextureArtifactTranscodeFormat(Format format);
[[nodiscard]] NURI_API bool
textureArtifactTranscodeUsesHighQuality(Format format) noexcept;
[[nodiscard]] NURI_API Result<TextureSourceFingerprint, std::string>
queryTextureSourceFingerprint(const std::filesystem::path &path);
[[nodiscard]] NURI_API Result<bool, std::string>
writeNativeTextureCacheMetadataAtomic(
    const std::filesystem::path &nativeTexturePath,
    const NativeTextureCacheMetadata &metadata);
[[nodiscard]] NURI_API NativeTextureCacheProbe probeNativeTextureCache(
    const std::filesystem::path &nativeTexturePath,
    const std::filesystem::path &sourcePath,
    uint64_t expectedSourceIdentityHash, Format expectedTargetFormat) noexcept;
[[nodiscard]] NURI_API bool
isTextureCacheUpToDate(const std::filesystem::path &cachePath,
                       const std::filesystem::path &sourcePath) noexcept;
[[nodiscard]] NURI_API std::string_view
textureFormatCacheSuffix(Format format) noexcept;
} // namespace nuri
