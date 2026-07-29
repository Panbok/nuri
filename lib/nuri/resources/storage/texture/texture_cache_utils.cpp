#include "nuri/resources/storage/texture/texture_cache_utils.h"
#include "nuri/resources/storage/binary_io.h"
#include "nuri/resources/storage/cache_utils.h"
#include "nuri/utils/env_utils.h"
namespace nuri {
namespace {
constexpr uint32_t kNativeTextureMetadataMagic = 0x4D43544Eu;
constexpr uint32_t kNativeTextureMetadataSchemaVersion = 3u;
[[nodiscard]] std::vector<std::byte>
encodeMetadata(const NativeTextureCacheMetadata &metadata) {
  BinaryWriter writer;
  writer.write(kNativeTextureMetadataMagic);
  writer.write(kNativeTextureMetadataSchemaVersion);
  writer.write(metadata.profileVersion);
  writer.write(metadata.sourceIdentityHash);
  writer.write(metadata.source.sizeBytes);
  writer.write(metadata.source.writeTimeTicks);
  writer.write(metadata.targetFormat);
  writer.write(metadata.textureType);
  writer.write(uint16_t{0});
  writer.write(metadata.dimensions.width);
  writer.write(metadata.dimensions.height);
  writer.write(metadata.dimensions.depth);
  writer.write(metadata.numLayers);
  writer.write(metadata.numFaces);
  writer.write(metadata.numMipLevels);
  writer.write(metadata.payloadSizeBytes);
  writer.write(metadata.artifactSizeBytes);
  writer.write(metadata.contentContract);
  writer.write(metadata.contentEncodingVersion);
  return std::move(writer).take();
}
[[nodiscard]] bool decodeMetadata(std::span<const std::byte> bytes,
                                  NativeTextureCacheMetadata &metadata) {
  BinaryReader reader(bytes);
  const uint32_t magic = reader.read<uint32_t>();
  const uint32_t schema = reader.read<uint32_t>();
  metadata.profileVersion = reader.read<uint32_t>();
  metadata.sourceIdentityHash = reader.read<uint64_t>();
  metadata.source.sizeBytes = reader.read<uint64_t>();
  metadata.source.writeTimeTicks = reader.read<int64_t>();
  metadata.targetFormat = reader.read<Format>();
  metadata.textureType = reader.read<TextureType>();
  static_cast<void>(reader.read<uint16_t>());
  metadata.dimensions.width = reader.read<uint32_t>();
  metadata.dimensions.height = reader.read<uint32_t>();
  metadata.dimensions.depth = reader.read<uint32_t>();
  metadata.numLayers = reader.read<uint32_t>();
  metadata.numFaces = reader.read<uint32_t>();
  metadata.numMipLevels = reader.read<uint32_t>();
  metadata.payloadSizeBytes = reader.read<uint64_t>();
  metadata.artifactSizeBytes = reader.read<uint64_t>();
  metadata.contentContract = reader.read<TextureContentContract>();
  metadata.contentEncodingVersion = reader.read<uint32_t>();
  return reader.empty() && magic == kNativeTextureMetadataMagic &&
         schema == kNativeTextureMetadataSchemaVersion &&
         metadata.targetFormat < Format::Count &&
         metadata.textureType < TextureType::Count &&
         metadata.contentContract <=
             TextureContentContract::NormalRgbCleanVarianceA;
}
struct FormatPolicy {
  Format format;
  std::string_view suffix;
  TextureArtifactTranscodeFormat transcode;
  bool highQuality;
};
constexpr std::array kFormatPolicies{
    FormatPolicy{Format::BC7_RGBA_UNORM, "bc7",
                 TextureArtifactTranscodeFormat::BC7_RGBA, false},
    FormatPolicy{Format::BC7_RGBA_SRGB, "bc7_srgb",
                 TextureArtifactTranscodeFormat::BC7_RGBA, false},
    FormatPolicy{Format::ETC2_RGB8_UNORM, "etc2",
                 TextureArtifactTranscodeFormat::ETC1_RGB, true},
    FormatPolicy{Format::ETC2_RGB8_SRGB, "etc2_srgb",
                 TextureArtifactTranscodeFormat::ETC1_RGB, true},
    FormatPolicy{Format::RGBA8_UNORM, "rgba8",
                 TextureArtifactTranscodeFormat::RGBA32, false},
    FormatPolicy{Format::RGBA8_SRGB, "rgba8_srgb",
                 TextureArtifactTranscodeFormat::RGBA32, false},
};
[[nodiscard]] const FormatPolicy *formatPolicy(Format format) noexcept {
  const auto found =
      std::ranges::find(kFormatPolicies, format, &FormatPolicy::format);
  return found == kFormatPolicies.end() ? nullptr : &*found;
}
} // namespace
uint64_t hashSceneTextureSourceIdentity(std::string_view sceneCanonicalPath,
                                        const MaterialTextureSlotData &slotData,
                                        bool srgb, uint32_t bakeSettingsTag) {
  Fnv1a64 hash;
  hash.addAll(sceneCanonicalPath, kSceneTextureArtifactSettingsVersion,
              static_cast<uint32_t>(slotData.sourceKind),
              slotData.embeddedIndex, static_cast<uint32_t>(srgb),
              bakeSettingsTag, std::string_view(slotData.path));
  return hash.value();
}

uint64_t hashTextureSourceIdentity(std::string_view sourceCanonicalPath,
                                   bool srgb, uint32_t processingTag) {
  Fnv1a64 hash;
  hash.addAll(sourceCanonicalPath, kSceneTextureArtifactSettingsVersion,
              static_cast<uint32_t>(srgb), processingTag);
  return hash.value();
}

Result<std::filesystem::path, std::string>
buildNativeTextureArtifactPath(const std::filesystem::path &sourcePath,
                               uint64_t sourceIdentityHash,
                               Format targetFormat) {
  const std::filesystem::path normalized = normalizeSourcePath(sourcePath);
  if (normalized.empty()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "buildNativeTextureArtifactPath: normalized source path is empty");
  }
  const auto configuredRoot = readEnvVar("NURI_TEXTURE_CACHE_ROOT");
  const std::filesystem::path cacheRoot =
      configuredRoot ? std::filesystem::path(*configuredRoot)
                     : std::filesystem::path{};
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
  const auto *policy = formatPolicy(format);
  return policy ? policy->suffix : "native";
}

std::filesystem::path buildNativeTextureCacheMetadataPath(
    const std::filesystem::path &nativeTexturePath) {
  return std::filesystem::path(nativeTexturePath).concat(".meta");
}

Format selectTextureArtifactTargetFormat(bool bc7Supported, bool etc2Supported,
                                         bool srgb,
                                         uint32_t componentCount) noexcept {
  constexpr std::array<std::array<Format, 2>, 3> formats{{
      {Format::BC7_RGBA_UNORM, Format::BC7_RGBA_SRGB},
      {Format::ETC2_RGB8_UNORM, Format::ETC2_RGB8_SRGB},
      {Format::RGBA8_UNORM, Format::RGBA8_SRGB},
  }};
  const size_t target = bc7Supported                            ? 0u
                        : etc2Supported && componentCount <= 3u ? 1u
                                                                : 2u;
  return formats[target][srgb ? 1u : 0u];
}

bool shouldPersistNativeTextureArtifact(Format format) noexcept {
  return formatPolicy(format) != nullptr;
}

Result<TextureArtifactTranscodeFormat, std::string>
resolveTextureArtifactTranscodeFormat(Format format) {
  const auto *policy = formatPolicy(format);
  if (policy) {
    return Result<TextureArtifactTranscodeFormat, std::string>::makeResult(
        policy->transcode);
  }
  return Result<TextureArtifactTranscodeFormat, std::string>::makeError(
      "resolveTextureArtifactTranscodeFormat: unsupported target format");
}

bool textureArtifactTranscodeUsesHighQuality(Format format) noexcept {
  const auto *policy = formatPolicy(format);
  return policy && policy->highQuality;
}

Result<TextureSourceFingerprint, std::string>
queryTextureSourceFingerprint(const std::filesystem::path &path) {
  std::error_code ec;
  const uint64_t sizeBytes = std::filesystem::file_size(path, ec);
  const std::filesystem::file_time_type writeTime =
      ec ? std::filesystem::file_time_type{}
         : std::filesystem::last_write_time(path, ec);
  if (ec) {
    return Result<TextureSourceFingerprint, std::string>::makeError(
        "failed to fingerprint texture source '" + path.string() + "'");
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
  return writeBinaryFileAtomic(
      buildNativeTextureCacheMetadataPath(nativeTexturePath),
      encodeMetadata(metadata));
}

Result<NativeTextureCacheMetadata, std::string> readNativeTextureCacheMetadata(
    const std::filesystem::path &nativeTexturePath) noexcept {
  auto bytes =
      readBinaryFile(buildNativeTextureCacheMetadataPath(nativeTexturePath));
  if (bytes.hasError()) {
    return Result<NativeTextureCacheMetadata, std::string>::makeError(
        bytes.error());
  }
  NativeTextureCacheMetadata metadata{};
  if (!decodeMetadata(bytes.value(), metadata) ||
      metadata.profileVersion != kNativeTextureArtifactProfileVersion) {
    return Result<NativeTextureCacheMetadata, std::string>::makeError(
        "native texture metadata profile is invalid");
  }
  std::error_code ec;
  const uint64_t artifactSize =
      std::filesystem::file_size(nativeTexturePath, ec);
  if (ec || artifactSize == 0u || artifactSize != metadata.artifactSizeBytes) {
    return Result<NativeTextureCacheMetadata, std::string>::makeError(
        "native texture artifact size is invalid");
  }
  return Result<NativeTextureCacheMetadata, std::string>::makeResult(metadata);
}

NativeTextureCacheProbe
probeNativeTextureCache(const std::filesystem::path &nativeTexturePath,
                        const std::filesystem::path &sourcePath,
                        uint64_t expectedSourceIdentityHash,
                        Format expectedTargetFormat,
                        TextureContentContract expectedContentContract,
                        uint32_t expectedContentEncodingVersion) noexcept {
  NativeTextureCacheProbe probe{};
  const auto failure = [&probe](NativeTextureCacheProbeStatus status,
                                std::string message = {}) {
    probe.status = status;
    probe.message = std::move(message);
    return probe;
  };
  std::error_code ec;
  if (!std::filesystem::exists(nativeTexturePath, ec) || ec) {
    return failure(NativeTextureCacheProbeStatus::Missing);
  }
  const std::filesystem::path metadataPath =
      buildNativeTextureCacheMetadataPath(nativeTexturePath);
  auto sourceFingerprint = queryTextureSourceFingerprint(sourcePath);
  if (sourceFingerprint.hasError()) {
    return failure(NativeTextureCacheProbeStatus::Stale,
                   sourceFingerprint.error());
  }
  auto bytesResult = readBinaryFile(metadataPath);
  if (bytesResult.hasError()) {
    return failure(NativeTextureCacheProbeStatus::Corrupt, bytesResult.error());
  }
  if (!decodeMetadata(bytesResult.value(), probe.metadata)) {
    return failure(NativeTextureCacheProbeStatus::Corrupt,
                   "native texture metadata header is invalid");
  }
  if (probe.metadata.profileVersion != kNativeTextureArtifactProfileVersion ||
      probe.metadata.sourceIdentityHash != expectedSourceIdentityHash ||
      probe.metadata.source != sourceFingerprint.value() ||
      probe.metadata.targetFormat != expectedTargetFormat ||
      probe.metadata.contentContract != expectedContentContract ||
      probe.metadata.contentEncodingVersion != expectedContentEncodingVersion) {
    return failure(NativeTextureCacheProbeStatus::Stale,
                   "native texture metadata does not match the source/profile");
  }
  const uint64_t artifactSizeBytes =
      std::filesystem::file_size(nativeTexturePath, ec);
  if (ec || artifactSizeBytes == 0u ||
      artifactSizeBytes != probe.metadata.artifactSizeBytes) {
    return failure(NativeTextureCacheProbeStatus::Corrupt,
                   "native texture artifact size is invalid");
  }
  probe.status = NativeTextureCacheProbeStatus::Hit;
  return probe;
}

bool isTextureCacheUpToDate(const std::filesystem::path &cachePath,
                            const std::filesystem::path &sourcePath) noexcept {
  std::error_code ec;
  const auto cacheTime = std::filesystem::last_write_time(cachePath, ec);
  if (ec) {
    return false;
  }
  const auto sourceTime = std::filesystem::last_write_time(sourcePath, ec);
  return !ec && cacheTime >= sourceTime;
}

} // namespace nuri
