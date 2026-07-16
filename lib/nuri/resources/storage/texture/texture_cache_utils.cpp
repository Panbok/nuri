#include "nuri/pch.h"

#include "nuri/resources/storage/texture/texture_cache_utils.h"

#include "nuri/resources/storage/mesh/mesh_cache_utils.h"
#include "nuri/utils/env_utils.h"

namespace nuri {
namespace {

constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
constexpr uint32_t kNativeTextureMetadataMagic = 0x4D43544Eu;
constexpr uint32_t kNativeTextureMetadataSchemaVersion = 2u;

#pragma pack(push, 1)
struct NativeTextureCacheMetadataDisk {
  uint32_t magic = kNativeTextureMetadataMagic;
  uint32_t schemaVersion = kNativeTextureMetadataSchemaVersion;
  uint32_t profileVersion = kNativeTextureArtifactProfileVersion;
  uint64_t sourceIdentityHash = 0u;
  uint64_t sourceSizeBytes = 0u;
  int64_t sourceWriteTimeTicks = 0;
  uint8_t targetFormat = 0u;
  uint8_t textureType = 0u;
  uint16_t reserved = 0u;
  uint32_t width = 0u;
  uint32_t height = 0u;
  uint32_t depth = 0u;
  uint32_t numLayers = 1u;
  uint32_t numFaces = 1u;
  uint32_t numMipLevels = 1u;
  uint64_t payloadSizeBytes = 0u;
  uint64_t artifactSizeBytes = 0u;
};
#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<NativeTextureCacheMetadataDisk>);

[[nodiscard]] std::filesystem::path configuredTextureCacheRoot() {
  const std::optional<std::string> value =
      readEnvVar("NURI_TEXTURE_CACHE_ROOT");
  if (!value.has_value()) {
    return {};
  }
  return std::filesystem::path(*value);
}

[[nodiscard]] NativeTextureCacheMetadataDisk
toDiskMetadata(const NativeTextureCacheMetadata &metadata) noexcept {
  return NativeTextureCacheMetadataDisk{
      .magic = kNativeTextureMetadataMagic,
      .schemaVersion = kNativeTextureMetadataSchemaVersion,
      .profileVersion = metadata.profileVersion,
      .sourceIdentityHash = metadata.sourceIdentityHash,
      .sourceSizeBytes = metadata.source.sizeBytes,
      .sourceWriteTimeTicks = metadata.source.writeTimeTicks,
      .targetFormat = static_cast<uint8_t>(metadata.targetFormat),
      .textureType = static_cast<uint8_t>(metadata.textureType),
      .reserved = 0u,
      .width = metadata.dimensions.width,
      .height = metadata.dimensions.height,
      .depth = metadata.dimensions.depth,
      .numLayers = metadata.numLayers,
      .numFaces = metadata.numFaces,
      .numMipLevels = metadata.numMipLevels,
      .payloadSizeBytes = metadata.payloadSizeBytes,
      .artifactSizeBytes = metadata.artifactSizeBytes,
  };
}

[[nodiscard]] NativeTextureCacheMetadata
fromDiskMetadata(const NativeTextureCacheMetadataDisk &metadata) noexcept {
  return NativeTextureCacheMetadata{
      .profileVersion = metadata.profileVersion,
      .sourceIdentityHash = metadata.sourceIdentityHash,
      .source =
          {
              .sizeBytes = metadata.sourceSizeBytes,
              .writeTimeTicks = metadata.sourceWriteTimeTicks,
          },
      .targetFormat = static_cast<Format>(metadata.targetFormat),
      .textureType = static_cast<TextureType>(metadata.textureType),
      .dimensions =
          {
              .width = metadata.width,
              .height = metadata.height,
              .depth = metadata.depth,
          },
      .numLayers = metadata.numLayers,
      .numFaces = metadata.numFaces,
      .numMipLevels = metadata.numMipLevels,
      .payloadSizeBytes = metadata.payloadSizeBytes,
      .artifactSizeBytes = metadata.artifactSizeBytes,
  };
}

[[nodiscard]] bool
metadataEnumsAreValid(const NativeTextureCacheMetadataDisk &metadata) noexcept {
  return metadata.targetFormat < static_cast<uint8_t>(Format::Count) &&
         metadata.textureType < static_cast<uint8_t>(TextureType::Count);
}

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
  fnv1aAddU32(hash, kSceneTextureArtifactSettingsVersion);
  fnv1aAddU32(hash, static_cast<uint32_t>(slotData.sourceKind));
  fnv1aAddU32(hash, slotData.embeddedIndex);
  fnv1aAddU32(hash, srgb ? 1u : 0u);
  fnv1aAddU32(hash, bakeSettingsTag);
  fnv1aAddString(hash, slotData.path);
  return hash;
}

uint64_t hashTextureSourceIdentity(std::string_view sourceCanonicalPath,
                                   bool srgb, uint32_t processingTag) {
  uint64_t hash = kFnvOffsetBasis;
  fnv1aAddString(hash, sourceCanonicalPath);
  fnv1aAddU32(hash, kSceneTextureArtifactSettingsVersion);
  fnv1aAddU32(hash, srgb ? 1u : 0u);
  fnv1aAddU32(hash, processingTag);
  return hash;
}

Result<std::filesystem::path, std::string>
buildNativeTextureArtifactPath(const std::filesystem::path &sourcePath,
                               uint64_t sourceIdentityHash,
                               Format targetFormat) {
  const std::filesystem::path normalized = normalizeMeshSourcePath(sourcePath);
  if (normalized.empty()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "buildNativeTextureArtifactPath: normalized source path is empty");
  }
  const std::filesystem::path cacheRoot = configuredTextureCacheRoot();
  const std::filesystem::path nativeRoot =
      cacheRoot.empty()
          ? normalized.parent_path() / ".nuri_scene_cache" / "native_textures"
          : cacheRoot / "native_textures";
  return Result<std::filesystem::path, std::string>::makeResult(
      nativeRoot / std::format("{}_{}_v{}.ktx2", hexU64(sourceIdentityHash),
                               textureFormatCacheSuffix(targetFormat),
                               kNativeTextureArtifactProfileVersion));
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

std::filesystem::path buildNativeTextureCacheMetadataPath(
    const std::filesystem::path &nativeTexturePath) {
  std::filesystem::path metadataPath = nativeTexturePath;
  metadataPath += ".meta";
  return metadataPath;
}

Format selectTextureArtifactTargetFormat(bool bc7Supported, bool etc2Supported,
                                         bool srgb,
                                         uint32_t componentCount) noexcept {
  if (bc7Supported) {
    return srgb ? Format::BC7_RGBA_SRGB : Format::BC7_RGBA_UNORM;
  }
  if (etc2Supported && componentCount <= 3u) {
    return srgb ? Format::ETC2_RGB8_SRGB : Format::ETC2_RGB8_UNORM;
  }
  return srgb ? Format::RGBA8_SRGB : Format::RGBA8_UNORM;
}

Format resolveTextureArtifactTargetFormat(TextureArtifactTarget target,
                                          bool srgb,
                                          uint32_t componentCount) noexcept {
  if (target == TextureArtifactTarget::ETC2 && componentCount > 3u) {
    target = TextureArtifactTarget::RGBA8;
  }
  switch (target) {
  case TextureArtifactTarget::BC7:
    return srgb ? Format::BC7_RGBA_SRGB : Format::BC7_RGBA_UNORM;
  case TextureArtifactTarget::ETC2:
    return srgb ? Format::ETC2_RGB8_SRGB : Format::ETC2_RGB8_UNORM;
  case TextureArtifactTarget::RGBA8:
    return srgb ? Format::RGBA8_SRGB : Format::RGBA8_UNORM;
  }
  return srgb ? Format::RGBA8_SRGB : Format::RGBA8_UNORM;
}

bool shouldPersistNativeTextureArtifact(Format format) noexcept {
  switch (format) {
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
    return true;
  default:
    return false;
  }
}

Result<TextureArtifactTranscodeFormat, std::string>
resolveTextureArtifactTranscodeFormat(Format format) {
  switch (format) {
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
    return Result<TextureArtifactTranscodeFormat, std::string>::makeResult(
        TextureArtifactTranscodeFormat::BC7_RGBA);
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
    return Result<TextureArtifactTranscodeFormat, std::string>::makeResult(
        TextureArtifactTranscodeFormat::ETC1_RGB);
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
    return Result<TextureArtifactTranscodeFormat, std::string>::makeResult(
        TextureArtifactTranscodeFormat::RGBA32);
  default:
    return Result<TextureArtifactTranscodeFormat, std::string>::makeError(
        "resolveTextureArtifactTranscodeFormat: unsupported target format");
  }
}

bool textureArtifactTranscodeUsesHighQuality(Format format) noexcept {
  return format == Format::ETC2_RGB8_UNORM || format == Format::ETC2_RGB8_SRGB;
}

Result<TextureSourceFingerprint, std::string>
queryTextureSourceFingerprint(const std::filesystem::path &path) {
  std::error_code ec;
  const uint64_t sizeBytes = std::filesystem::file_size(path, ec);
  if (ec) {
    return Result<TextureSourceFingerprint, std::string>::makeError(
        "queryTextureSourceFingerprint: failed to query file size for '" +
        path.string() + "': " + ec.message());
  }
  const std::filesystem::file_time_type writeTime =
      std::filesystem::last_write_time(path, ec);
  if (ec) {
    return Result<TextureSourceFingerprint, std::string>::makeError(
        "queryTextureSourceFingerprint: failed to query write time for '" +
        path.string() + "': " + ec.message());
  }
  return Result<TextureSourceFingerprint, std::string>::makeResult(
      TextureSourceFingerprint{
          .sizeBytes = sizeBytes,
          .writeTimeTicks = writeTime.time_since_epoch().count(),
      });
}

Result<bool, std::string> writeNativeTextureCacheMetadataAtomic(
    const std::filesystem::path &nativeTexturePath,
    const NativeTextureCacheMetadata &metadata) {
  const NativeTextureCacheMetadataDisk disk = toDiskMetadata(metadata);
  const std::span<const std::byte> bytes{
      reinterpret_cast<const std::byte *>(&disk), sizeof(disk)};
  return writeBinaryFileAtomic(
      buildNativeTextureCacheMetadataPath(nativeTexturePath), bytes);
}

NativeTextureCacheProbe
probeNativeTextureCache(const std::filesystem::path &nativeTexturePath,
                        const std::filesystem::path &sourcePath,
                        uint64_t expectedSourceIdentityHash,
                        Format expectedTargetFormat) noexcept {
  NativeTextureCacheProbe probe{};
  std::error_code ec;
  if (!std::filesystem::exists(nativeTexturePath, ec) || ec) {
    probe.status = NativeTextureCacheProbeStatus::Missing;
    return probe;
  }

  const std::filesystem::path metadataPath =
      buildNativeTextureCacheMetadataPath(nativeTexturePath);
  if (!std::filesystem::exists(metadataPath, ec) || ec) {
    probe.status = NativeTextureCacheProbeStatus::Corrupt;
    probe.message = "native texture metadata sidecar is missing";
    return probe;
  }

  auto sourceFingerprint = queryTextureSourceFingerprint(sourcePath);
  if (sourceFingerprint.hasError()) {
    probe.status = NativeTextureCacheProbeStatus::Stale;
    probe.message = sourceFingerprint.error();
    return probe;
  }

  auto bytesResult = readBinaryFile(metadataPath);
  if (bytesResult.hasError()) {
    probe.status = NativeTextureCacheProbeStatus::Corrupt;
    probe.message = bytesResult.error();
    return probe;
  }
  const std::vector<std::byte> &bytes = bytesResult.value();
  if (bytes.size() != sizeof(NativeTextureCacheMetadataDisk)) {
    probe.status = NativeTextureCacheProbeStatus::Corrupt;
    probe.message = "native texture metadata has an invalid size";
    return probe;
  }

  NativeTextureCacheMetadataDisk disk{};
  std::memcpy(&disk, bytes.data(), sizeof(disk));
  if (disk.magic != kNativeTextureMetadataMagic ||
      disk.schemaVersion != kNativeTextureMetadataSchemaVersion ||
      !metadataEnumsAreValid(disk)) {
    probe.status = NativeTextureCacheProbeStatus::Corrupt;
    probe.message = "native texture metadata header is invalid";
    return probe;
  }

  probe.metadata = fromDiskMetadata(disk);
  if (probe.metadata.profileVersion != kNativeTextureArtifactProfileVersion ||
      probe.metadata.sourceIdentityHash != expectedSourceIdentityHash ||
      probe.metadata.source != sourceFingerprint.value() ||
      probe.metadata.targetFormat != expectedTargetFormat) {
    probe.status = NativeTextureCacheProbeStatus::Stale;
    probe.message = "native texture metadata does not match the source/profile";
    return probe;
  }

  const uint64_t artifactSizeBytes =
      std::filesystem::file_size(nativeTexturePath, ec);
  if (ec || artifactSizeBytes == 0u ||
      artifactSizeBytes != probe.metadata.artifactSizeBytes ||
      probe.metadata.payloadSizeBytes == 0u ||
      probe.metadata.dimensions.width == 0u ||
      probe.metadata.dimensions.height == 0u ||
      probe.metadata.dimensions.depth == 0u || probe.metadata.numLayers == 0u ||
      probe.metadata.numFaces == 0u || probe.metadata.numMipLevels == 0u) {
    probe.status = NativeTextureCacheProbeStatus::Corrupt;
    probe.message = "native texture artifact size/topology is invalid";
    return probe;
  }

  probe.status = NativeTextureCacheProbeStatus::Hit;
  return probe;
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
