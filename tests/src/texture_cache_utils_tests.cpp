#include "tests_pch.h"

#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/resources/storage/texture/dds_texture_pack.h"
#include "nuri/resources/storage/texture/texture_artifact_builder.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"
#include "render_graph_test_support.h"

#include <array>
#include <chrono>
#include <fstream>
#include <ktx.h>
#include <thread>

#include <glm/glm.hpp>

namespace {

struct ScopedTempDir {
  explicit ScopedTempDir(std::string_view prefix) {
    const auto uniqueId =
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
    path = std::filesystem::temp_directory_path() /
           (std::string(prefix) + "_" + std::to_string(uniqueId));
    std::error_code ec;
    const bool created = std::filesystem::create_directories(path, ec);
    if (ec) {
      throw std::filesystem::filesystem_error(
          "ScopedTempDir create_directories", path, ec);
    }
    if (!created && !std::filesystem::is_directory(path)) {
      throw std::filesystem::filesystem_error(
          "ScopedTempDir path is not a directory", path,
          std::make_error_code(std::errc::file_exists));
    }
  }

  ScopedTempDir(const ScopedTempDir &) = delete;
  ScopedTempDir &operator=(const ScopedTempDir &) = delete;
  ScopedTempDir(ScopedTempDir &&other) noexcept : path(std::move(other.path)) {
    other.path.clear();
  }
  ScopedTempDir &operator=(ScopedTempDir &&other) noexcept {
    if (this == &other) {
      return *this;
    }

    std::error_code ec;
    if (!path.empty()) {
      std::filesystem::remove_all(path, ec);
    }
    path = std::move(other.path);
    other.path.clear();
    return *this;
  }

  ~ScopedTempDir() {
    std::error_code ec;
    if (path.empty()) {
      return;
    }
    std::filesystem::remove_all(path, ec);
  }

  std::filesystem::path path;
};

void writeTextFile(const std::filesystem::path &path, std::string_view text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << path.string();
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
  ASSERT_TRUE(file.good()) << path.string();
}

std::string readTextFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

std::vector<std::byte> readBinaryBytes(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return {};
  }
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size <= 0) {
    return {};
  }
  file.seekg(0, std::ios::beg);
  std::vector<std::byte> bytes(static_cast<size_t>(size));
  file.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!file.good()) {
    return {};
  }
  return bytes;
}

void writeRgbaTga(const std::filesystem::path &path, uint16_t width,
                  uint16_t height, std::span<const uint8_t> rgba) {
  ASSERT_EQ(rgba.size(), static_cast<size_t>(width) * height * 4u);
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << path.string();

  std::array<uint8_t, 18> header{};
  header[2] = 2u;
  header[12] = static_cast<uint8_t>(width & 0xffu);
  header[13] = static_cast<uint8_t>(width >> 8u);
  header[14] = static_cast<uint8_t>(height & 0xffu);
  header[15] = static_cast<uint8_t>(height >> 8u);
  header[16] = 32u;
  header[17] = 0x28u;
  file.write(reinterpret_cast<const char *>(header.data()),
             static_cast<std::streamsize>(header.size()));

  for (size_t i = 0u; i < rgba.size(); i += 4u) {
    const std::array<uint8_t, 4> bgra{rgba[i + 2u], rgba[i + 1u], rgba[i + 0u],
                                      rgba[i + 3u]};
    file.write(reinterpret_cast<const char *>(bgra.data()),
               static_cast<std::streamsize>(bgra.size()));
  }
  ASSERT_TRUE(file.good()) << path.string();
}

void writeUastcKtx2(const std::filesystem::path &path) {
  const ktxTextureCreateInfo createInfo{
      .glInternalformat = 0u,
      .vkFormat = 37u,
      .pDfd = nullptr,
      .baseWidth = 4u,
      .baseHeight = 4u,
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = 1u,
      .numLayers = 1u,
      .numFaces = 1u,
      .isArray = KTX_FALSE,
      .generateMipmaps = KTX_FALSE,
  };
  ktxTexture2 *rawTexture = nullptr;
  ASSERT_EQ(ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE,
                               &rawTexture),
            KTX_SUCCESS);
  ASSERT_NE(rawTexture, nullptr);
  struct TextureDestroy {
    void operator()(ktxTexture2 *texture) const noexcept {
      if (texture != nullptr) {
        ktxTexture_Destroy(ktxTexture(texture));
      }
    }
  };
  std::unique_ptr<ktxTexture2, TextureDestroy> texture(rawTexture);

  std::array<uint8_t, 4u * 4u * 4u> rgba{};
  for (size_t i = 0u; i < rgba.size(); ++i) {
    rgba[i] = static_cast<uint8_t>((i * 37u) & 0xffu);
  }
  ASSERT_EQ(ktxTexture_SetImageFromMemory(ktxTexture(texture.get()), 0u, 0u, 0u,
                                          rgba.data(), rgba.size()),
            KTX_SUCCESS);

  ktxBasisParams params{};
  params.structSize = sizeof(params);
  params.threadCount = 1u;
  params.uastc = KTX_TRUE;
  params.uastcFlags = KTX_PACK_UASTC_LEVEL_FASTEST;
  ASSERT_EQ(ktxTexture2_CompressBasisEx(texture.get(), &params), KTX_SUCCESS);
  std::filesystem::create_directories(path.parent_path());
  ASSERT_EQ(ktxTexture_WriteToNamedFile(ktxTexture(texture.get()),
                                        path.string().c_str()),
            KTX_SUCCESS);
}

void writeU32Le(std::span<std::byte> bytes, size_t offset, uint32_t value) {
  ASSERT_LE(offset + sizeof(value), bytes.size());
  bytes[offset + 0u] = static_cast<std::byte>(value & 0xffu);
  bytes[offset + 1u] = static_cast<std::byte>((value >> 8u) & 0xffu);
  bytes[offset + 2u] = static_cast<std::byte>((value >> 16u) & 0xffu);
  bytes[offset + 3u] = static_cast<std::byte>((value >> 24u) & 0xffu);
}

std::array<std::byte, 16> writeBc7Dds(const std::filesystem::path &path,
                                      bool srgb) {
  constexpr size_t kDdsHeaderSize = 148u;
  std::array<std::byte, 16> payload{};
  for (size_t i = 0u; i < payload.size(); ++i) {
    payload[i] = static_cast<std::byte>((i * 19u + 7u) & 0xffu);
  }

  std::vector<std::byte> bytes(kDdsHeaderSize + payload.size());
  writeU32Le(bytes, 0u, 0x20534444u); // "DDS "
  writeU32Le(bytes, 4u, 124u);
  writeU32Le(bytes, 12u, 4u);
  writeU32Le(bytes, 16u, 4u);
  writeU32Le(bytes, 28u, 1u);
  writeU32Le(bytes, 76u, 32u);
  writeU32Le(bytes, 80u, 0x4u);
  writeU32Le(bytes, 84u, 0x30315844u); // "DX10"
  writeU32Le(bytes, 108u, 0x1000u);
  writeU32Le(bytes, 128u, srgb ? 99u : 98u);
  writeU32Le(bytes, 132u, 3u);
  writeU32Le(bytes, 140u, 1u);
  std::copy(payload.begin(), payload.end(), bytes.begin() + kDdsHeaderSize);

  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  EXPECT_TRUE(file.is_open()) << path.string();
  file.write(reinterpret_cast<const char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  EXPECT_TRUE(file.good()) << path.string();
  return payload;
}

class Bc7FakeGpu final : public nuri::test_support::FakeGPUDeviceBase {
public:
  nuri::TextureCompressionCaps getTextureCompressionCaps() const override {
    return {.bc7 = true};
  }
};

} // namespace

TEST(TextureCacheUtilsTests, BuildsSourceDerivedNativeArtifactPaths) {
  const ScopedTempDir dir("nuri_texture_cache");
  const std::filesystem::path scenePath = dir.path / "models" / "helmet.glb";

  auto nativePathResult = nuri::buildNativeTextureArtifactPath(
      scenePath, 0x1234567890ABCDEFull, nuri::Format::BC7_RGBA_SRGB);
  ASSERT_FALSE(nativePathResult.hasError()) << nativePathResult.error();
  const std::filesystem::path nativePath = nativePathResult.value();
  EXPECT_EQ(nativePath.filename(), "1234567890abcdef_bc7_srgb_v3.ktx2");
  EXPECT_EQ(nativePath.parent_path().filename(), "native_textures");
  EXPECT_EQ(nativePath.parent_path().parent_path().filename(),
            ".nuri_scene_cache");
  EXPECT_EQ(nuri::buildNativeTextureCacheMetadataPath(nativePath).filename(),
            "1234567890abcdef_bc7_srgb_v3.ktx2.meta");
}

TEST(TextureCacheUtilsTests, SelectsCompressedLinearAndSrgbTargets) {
  EXPECT_EQ(nuri::selectTextureArtifactTargetFormat(true, false, false, 4u),
            nuri::Format::BC7_RGBA_UNORM);
  EXPECT_EQ(nuri::selectTextureArtifactTargetFormat(true, false, true, 4u),
            nuri::Format::BC7_RGBA_SRGB);
  EXPECT_EQ(nuri::selectTextureArtifactTargetFormat(false, true, false, 3u),
            nuri::Format::ETC2_RGB8_UNORM);
  EXPECT_EQ(nuri::selectTextureArtifactTargetFormat(false, true, false, 4u),
            nuri::Format::RGBA8_UNORM);
  EXPECT_TRUE(
      nuri::shouldPersistNativeTextureArtifact(nuri::Format::RGBA8_UNORM));
}

TEST(TextureCacheUtilsTests, NativeMetadataDetectsStaleAndCorruptArtifacts) {
  const ScopedTempDir dir("nuri_texture_metadata");
  const std::filesystem::path sourcePath = dir.path / "source.ktx2";
  const std::filesystem::path nativePath = dir.path / "source_bc7_v3.ktx2";
  constexpr uint64_t kSourceIdentity = 0x123456789abcdef0ull;
  writeTextFile(sourcePath, "authored-source");
  writeTextFile(nativePath, "native-artifact");

  auto fingerprint = nuri::queryTextureSourceFingerprint(sourcePath);
  ASSERT_FALSE(fingerprint.hasError()) << fingerprint.error();
  const nuri::NativeTextureCacheMetadata metadata{
      .profileVersion = nuri::kNativeTextureArtifactProfileVersion,
      .sourceIdentityHash = kSourceIdentity,
      .source = fingerprint.value(),
      .targetFormat = nuri::Format::BC7_RGBA_UNORM,
      .textureType = nuri::TextureType::Texture2D,
      .dimensions = {.width = 64u, .height = 32u, .depth = 1u},
      .numLayers = 1u,
      .numFaces = 1u,
      .numMipLevels = 7u,
      .payloadSizeBytes = 4096u,
      .artifactSizeBytes = std::filesystem::file_size(nativePath),
  };
  auto writeMetadata =
      nuri::writeNativeTextureCacheMetadataAtomic(nativePath, metadata);
  ASSERT_FALSE(writeMetadata.hasError()) << writeMetadata.error();

  const nuri::NativeTextureCacheProbe hit = nuri::probeNativeTextureCache(
      nativePath, sourcePath, kSourceIdentity, nuri::Format::BC7_RGBA_UNORM);
  EXPECT_EQ(hit.status, nuri::NativeTextureCacheProbeStatus::Hit);
  EXPECT_EQ(hit.metadata.numMipLevels, 7u);

  writeTextFile(sourcePath, "authored-source-changed");
  const nuri::NativeTextureCacheProbe stale = nuri::probeNativeTextureCache(
      nativePath, sourcePath, kSourceIdentity, nuri::Format::BC7_RGBA_UNORM);
  EXPECT_EQ(stale.status, nuri::NativeTextureCacheProbeStatus::Stale);

  writeTextFile(sourcePath, "authored-source");
  auto refreshedFingerprint = nuri::queryTextureSourceFingerprint(sourcePath);
  ASSERT_FALSE(refreshedFingerprint.hasError()) << refreshedFingerprint.error();
  nuri::NativeTextureCacheMetadata refreshed = metadata;
  refreshed.source = refreshedFingerprint.value();
  ASSERT_FALSE(
      nuri::writeNativeTextureCacheMetadataAtomic(nativePath, refreshed)
          .hasError());
  writeTextFile(nativePath, "truncated");
  const nuri::NativeTextureCacheProbe corrupt = nuri::probeNativeTextureCache(
      nativePath, sourcePath, kSourceIdentity, nuri::Format::BC7_RGBA_UNORM);
  EXPECT_EQ(corrupt.status, nuri::NativeTextureCacheProbeStatus::Corrupt);
}

TEST(TextureCacheUtilsTests, AuthoredUastcColdBuildAndWarmHitAreEquivalent) {
  const ScopedTempDir dir("nuri_texture_warm_hit");
  const std::filesystem::path sourcePath =
      dir.path / "textures" / "linear_uastc.ktx2";
  writeUastcKtx2(sourcePath);
  const nuri::TextureArtifactBuildOptions options{
      .loadOptions = nuri::TextureLoadOptions{.srgb = false},
      .encoding = nuri::TextureArtifactEncoding::Uastc,
  };
  const uint64_t identity = nuri::hashTextureSourceIdentity(
      nuri::canonicalizeResourcePath(sourcePath.string()), false,
      nuri::textureArtifactProcessingTag(options));

  const nuri::TextureCacheTelemetry before = nuri::Texture::cacheTelemetry();
  auto coldArtifact = nuri::ensureTextureArtifactFromFile(
      sourcePath, identity, nuri::Format::BC7_RGBA_UNORM, options);
  ASSERT_FALSE(coldArtifact.hasError()) << coldArtifact.error();
  EXPECT_TRUE(coldArtifact.value().built);
  Bc7FakeGpu coldGpu;
  auto coldTexture = nuri::Texture::loadTextureKtx2(
      coldGpu, coldArtifact.value().artifactPath.string(), "cold_linear");
  ASSERT_FALSE(coldTexture.hasError()) << coldTexture.error();
  ASSERT_EQ(coldGpu.createdTextureDescs.size(), 1u);
  ASSERT_EQ(coldGpu.createdTextureData.size(), 1u);
  EXPECT_EQ(coldGpu.createdTextureDescs[0].format,
            nuri::Format::BC7_RGBA_UNORM);

  const std::filesystem::path nativePath = coldArtifact.value().artifactPath;
  EXPECT_TRUE(std::filesystem::exists(nativePath));
  EXPECT_TRUE(std::filesystem::exists(
      nuri::buildNativeTextureCacheMetadataPath(nativePath)));

  const nuri::TextureCacheTelemetry afterCold = nuri::Texture::cacheTelemetry();
  EXPECT_EQ(afterCold.artifactBuilds, before.artifactBuilds + 1u);
  EXPECT_EQ(afterCold.nativeWrites, before.nativeWrites + 1u);
  EXPECT_GT(afterCold.authoredSourceBytesRead, before.authoredSourceBytesRead);

  auto warmArtifact = nuri::ensureTextureArtifactFromFile(
      sourcePath, identity, nuri::Format::BC7_RGBA_UNORM, options);
  ASSERT_FALSE(warmArtifact.hasError()) << warmArtifact.error();
  EXPECT_FALSE(warmArtifact.value().built);
  Bc7FakeGpu warmGpu;
  auto warmTexture = nuri::Texture::loadTextureKtx2(
      warmGpu, warmArtifact.value().artifactPath.string(), "warm_linear");
  ASSERT_FALSE(warmTexture.hasError()) << warmTexture.error();
  ASSERT_EQ(warmGpu.createdTextureData.size(), 1u);
  EXPECT_EQ(warmGpu.createdTextureDescs[0].format,
            nuri::Format::BC7_RGBA_UNORM);
  EXPECT_EQ(warmGpu.createdTextureData[0], coldGpu.createdTextureData[0]);

  const nuri::TextureCacheTelemetry afterWarm = nuri::Texture::cacheTelemetry();
  EXPECT_EQ(afterWarm.nativeHits, afterCold.nativeHits + 1u);
  EXPECT_EQ(afterWarm.artifactBuilds, afterCold.artifactBuilds);
  EXPECT_EQ(afterWarm.authoredSourceBytesRead,
            afterCold.authoredSourceBytesRead);
  EXPECT_GT(afterWarm.nativeArtifactBytesRead,
            afterCold.nativeArtifactBytesRead);
}

TEST(TextureCacheUtilsTests, DdsLoadsNativeBc7PayloadWithoutTranscoding) {
  const ScopedTempDir dir("nuri_dds_native_payload");
  const std::filesystem::path ddsPath = dir.path / "bistro_texture.dds";
  const std::array<std::byte, 16> expectedPayload = writeBc7Dds(ddsPath, true);

  const nuri::TextureCacheTelemetry before = nuri::Texture::cacheTelemetry();
  Bc7FakeGpu gpu;
  auto texture = nuri::Texture::loadTexture(
      gpu, ddsPath.string(), nuri::TextureLoadOptions{}, "bistro_dds_test");
  ASSERT_FALSE(texture.hasError()) << texture.error();
  ASSERT_EQ(gpu.createdTextureDescs.size(), 1u);
  ASSERT_EQ(gpu.createdTextureData.size(), 1u);

  const nuri::TextureDesc &desc = gpu.createdTextureDescs.front();
  EXPECT_EQ(desc.format, nuri::Format::BC7_RGBA_SRGB);
  EXPECT_EQ(desc.dimensions.width, 4u);
  EXPECT_EQ(desc.dimensions.height, 4u);
  EXPECT_EQ(desc.numMipLevels, 1u);
  EXPECT_EQ(desc.dataNumMipLevels, 1u);
  EXPECT_FALSE(desc.generateMipmaps);
  EXPECT_EQ(
      gpu.createdTextureData.front(),
      std::vector<std::byte>(expectedPayload.begin(), expectedPayload.end()));

  const nuri::TextureCacheTelemetry after = nuri::Texture::cacheTelemetry();
  EXPECT_EQ(after.ddsSourceBytesRead, before.ddsSourceBytesRead + 164u);
  EXPECT_EQ(after.artifactBuilds, before.artifactBuilds);
  EXPECT_EQ(after.nativeWrites, before.nativeWrites);
}

TEST(TextureCacheUtilsTests, DdsScenePackBuildsHitsAndRebuildsWhenStale) {
  const ScopedTempDir dir("nuri_dds_scene_pack");
  const std::filesystem::path scenePath = dir.path / "scene" / "bistro.gltf";
  const std::filesystem::path firstPath =
      dir.path / "scene" / "textures" / "first.dds";
  const std::filesystem::path secondPath =
      dir.path / "scene" / "textures" / "second.dds";
  writeTextFile(scenePath, "scene");
  writeBc7Dds(firstPath, false);
  writeBc7Dds(secondPath, true);

  const std::array<nuri::DdsTexturePackSource, 2> sources{{
      {.path = firstPath.string()},
      {.path = secondPath.string()},
  }};
  const nuri::DdsTexturePackTelemetry before = nuri::ddsTexturePackTelemetry();
  auto cold = nuri::ensureDdsTexturePack(scenePath, sources);
  ASSERT_FALSE(cold.hasError()) << cold.error();
  ASSERT_TRUE(cold.value().built);
  ASSERT_NE(cold.value().pack, nullptr);
  EXPECT_EQ(cold.value().pack->entryCount(), 2u);
  EXPECT_EQ(cold.value().pack->path().parent_path().filename(), "scene_packs");

  const std::vector<std::byte> firstBytes = readBinaryBytes(firstPath);
  const std::string firstCanonical =
      nuri::canonicalizeResourcePath(firstPath.string());
  auto packedFirst = cold.value().pack->read(firstCanonical);
  ASSERT_FALSE(packedFirst.hasError()) << packedFirst.error();
  EXPECT_TRUE(std::ranges::equal(packedFirst.value(), firstBytes));
  Bc7FakeGpu packedGpu;
  auto packedTexture = nuri::Texture::loadDdsTexture(
      packedGpu, packedFirst.value(), firstCanonical, "packed_dds");
  ASSERT_FALSE(packedTexture.hasError()) << packedTexture.error();
  ASSERT_EQ(packedGpu.createdTextureData.size(), 1u);
  EXPECT_EQ(packedGpu.createdTextureData.front(),
            std::vector<std::byte>(firstBytes.begin() + 148u,
                                   firstBytes.begin() + 164u));

  const std::vector<std::byte> secondBytes = readBinaryBytes(secondPath);
  const std::string secondCanonical =
      nuri::canonicalizeResourcePath(secondPath.string());
  auto firstOwned = std::async(std::launch::async, [&] {
    return cold.value().pack->readOwned(firstCanonical);
  });
  auto secondOwned = std::async(std::launch::async, [&] {
    return cold.value().pack->readOwned(secondCanonical);
  });
  auto firstOwnedResult = firstOwned.get();
  auto secondOwnedResult = secondOwned.get();
  ASSERT_FALSE(firstOwnedResult.hasError()) << firstOwnedResult.error();
  ASSERT_FALSE(secondOwnedResult.hasError()) << secondOwnedResult.error();
  EXPECT_EQ(firstOwnedResult.value(), firstBytes);
  EXPECT_EQ(secondOwnedResult.value(), secondBytes);

  const nuri::DdsTexturePackTelemetry afterCold =
      nuri::ddsTexturePackTelemetry();
  EXPECT_EQ(afterCold.misses, before.misses + 1u);
  EXPECT_EQ(afterCold.builds, before.builds + 1u);
  EXPECT_EQ(afterCold.buildFailures, before.buildFailures);
  EXPECT_EQ(afterCold.buildSourceBytesRead,
            before.buildSourceBytesRead +
                std::filesystem::file_size(firstPath) +
                std::filesystem::file_size(secondPath));

  auto warm = nuri::ensureDdsTexturePack(scenePath, sources);
  ASSERT_FALSE(warm.hasError()) << warm.error();
  EXPECT_FALSE(warm.value().built);
  ASSERT_NE(warm.value().pack, nullptr);
  const nuri::DdsTexturePackTelemetry afterWarm =
      nuri::ddsTexturePackTelemetry();
  EXPECT_EQ(afterWarm.hits, afterCold.hits + 1u);
  EXPECT_EQ(afterWarm.builds, afterCold.builds);

  {
    std::ofstream append(secondPath, std::ios::binary | std::ios::app);
    ASSERT_TRUE(append.is_open());
    const char marker = '\x5a';
    append.write(&marker, 1);
    ASSERT_TRUE(append.good());
  }
  auto rebuilt = nuri::ensureDdsTexturePack(scenePath, sources);
  ASSERT_FALSE(rebuilt.hasError()) << rebuilt.error();
  EXPECT_TRUE(rebuilt.value().built);
  const nuri::DdsTexturePackTelemetry afterRebuild =
      nuri::ddsTexturePackTelemetry();
  EXPECT_EQ(afterRebuild.stale, afterWarm.stale + 1u);
  EXPECT_EQ(afterRebuild.builds, afterWarm.builds + 1u);
  auto packedSecond = rebuilt.value().pack->read(
      nuri::canonicalizeResourcePath(secondPath.string()));
  ASSERT_FALSE(packedSecond.hasError()) << packedSecond.error();
  EXPECT_EQ(packedSecond.value().size(),
            std::filesystem::file_size(secondPath));
}

TEST(TextureCacheUtilsTests, HashChangesForExternalVsEmbeddedSources) {
  constexpr std::string_view kScenePath = "e:/project/scene.glb";

  nuri::MaterialTextureSlotData externalSlot{};
  externalSlot.sourceKind = nuri::MaterialTextureSourceKind::ExternalFile;
  externalSlot.path = "e:/project/textures/basecolor.png";

  nuri::MaterialTextureSlotData embeddedSlot{};
  embeddedSlot.sourceKind =
      nuri::MaterialTextureSourceKind::EmbeddedSceneTexture;
  embeddedSlot.embeddedIndex = 3u;

  const uint64_t externalHash =
      nuri::hashSceneTextureSourceIdentity(kScenePath, externalSlot, true, 1u);
  const uint64_t embeddedHash =
      nuri::hashSceneTextureSourceIdentity(kScenePath, embeddedSlot, true, 1u);

  EXPECT_NE(externalHash, embeddedHash);
}

TEST(TextureCacheUtilsTests, UpToDateCheckTracksWriteTimes) {
  const ScopedTempDir dir("nuri_texture_times");
  const std::filesystem::path sourcePath = dir.path / "source.png";
  const std::filesystem::path cachePath = dir.path / "cache.ktx2";

  writeTextFile(sourcePath, "source");
  writeTextFile(cachePath, "cache");
  const auto baseTime = std::filesystem::file_time_type::clock::now();
  const auto sourceOlderTime = baseTime - std::chrono::seconds(4);
  const auto cacheNewerTime = baseTime - std::chrono::seconds(1);
  std::filesystem::last_write_time(sourcePath, sourceOlderTime);
  std::filesystem::last_write_time(cachePath, cacheNewerTime);
  EXPECT_TRUE(nuri::isTextureCacheUpToDate(cachePath, sourcePath));

  writeTextFile(sourcePath, "source-newer");
  const auto sourceNewerTime = baseTime + std::chrono::seconds(4);
  std::filesystem::last_write_time(sourcePath, sourceNewerTime);
  EXPECT_FALSE(nuri::isTextureCacheUpToDate(cachePath, sourcePath));
}

TEST(TextureCacheUtilsTests, HdrHalfConversionClampsPositiveOverflow) {
  const std::string source =
      readTextFile(std::filesystem::path(PROJECT_SOURCE_DIR) / "lib" / "nuri" /
                   "resources" / "gpu" / "texture.cpp");
  ASSERT_FALSE(source.empty());

  EXPECT_NE(source.find("constexpr float kMaxFiniteHalf = 65504.0f"),
            std::string::npos);
  EXPECT_NE(source.find("std::isnan(value)"), std::string::npos);
  EXPECT_NE(source.find("std::isinf(value)"), std::string::npos);
  EXPECT_NE(source.find("std::signbit(value) ? 0.0f : kMaxFiniteHalf"),
            std::string::npos);
  EXPECT_NE(source.find("std::clamp(value, 0.0f, kMaxFiniteHalf)"),
            std::string::npos);
  EXPECT_NE(source.find("glm::packHalf1x16(value)"), std::string::npos);
}

TEST(TextureCacheUtilsTests, AlphaCoverageMipSemanticUploadsExplicitMips) {
  const ScopedTempDir dir("nuri_alpha_coverage_mips");
  const std::filesystem::path texturePath = dir.path / "mask.tga";
  const std::array<uint8_t, 16> rgba{
      255u, 255u, 255u, 255u, 255u, 255u, 255u, 0u,
      255u, 255u, 255u, 0u,   255u, 255u, 255u, 0u,
  };
  writeRgbaTga(texturePath, 2u, 2u, rgba);

  nuri::test_support::FakeGPUDeviceBase gpu;
  auto texture = nuri::Texture::loadTexture(
      gpu, texturePath.string(),
      nuri::TextureLoadOptions{
          .srgb = true,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::AlphaCoverage,
          .alphaCoverageCutoff = 0.5f,
      },
      "alpha_coverage_test");
  ASSERT_FALSE(texture.hasError()) << texture.error();
  ASSERT_EQ(gpu.createdTextureDescs.size(), 1u);
  ASSERT_EQ(gpu.createdTextureData.size(), 1u);

  const nuri::TextureDesc &desc = gpu.createdTextureDescs.back();
  EXPECT_EQ(desc.numMipLevels, 2u);
  EXPECT_EQ(desc.dataNumMipLevels, 2u);
  EXPECT_FALSE(desc.generateMipmaps);
  ASSERT_EQ(gpu.createdTextureData.back().size(), 20u);
  const uint8_t mipAlpha =
      std::to_integer<uint8_t>(gpu.createdTextureData.back()[16u + 3u]);
  EXPECT_GE(mipAlpha, 128u);
}

TEST(TextureCacheUtilsTests, NormalMipSemanticRenormalizesUploadedMips) {
  const ScopedTempDir dir("nuri_normal_mips");
  const std::filesystem::path texturePath = dir.path / "normal.tga";
  const std::array<uint8_t, 16> rgba{
      255u, 128u, 128u, 255u, 128u, 255u, 128u, 255u,
      128u, 128u, 255u, 255u, 128u, 128u, 255u, 255u,
  };
  writeRgbaTga(texturePath, 2u, 2u, rgba);

  nuri::test_support::FakeGPUDeviceBase gpu;
  auto texture = nuri::Texture::loadTexture(
      gpu, texturePath.string(),
      nuri::TextureLoadOptions{
          .srgb = false,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::NormalMap,
      },
      "normal_mip_test");
  ASSERT_FALSE(texture.hasError()) << texture.error();
  ASSERT_EQ(gpu.createdTextureDescs.size(), 1u);
  ASSERT_EQ(gpu.createdTextureData.size(), 1u);

  const nuri::TextureDesc &desc = gpu.createdTextureDescs.back();
  EXPECT_EQ(desc.numMipLevels, 2u);
  EXPECT_EQ(desc.dataNumMipLevels, 2u);
  EXPECT_FALSE(desc.generateMipmaps);
  ASSERT_EQ(gpu.createdTextureData.back().size(), 20u);

  const std::vector<std::byte> &bytes = gpu.createdTextureData.back();
  const glm::vec3 normal(
      static_cast<float>(std::to_integer<uint8_t>(bytes[16u + 0u])) / 127.5f -
          1.0f,
      static_cast<float>(std::to_integer<uint8_t>(bytes[16u + 1u])) / 127.5f -
          1.0f,
      static_cast<float>(std::to_integer<uint8_t>(bytes[16u + 2u])) / 127.5f -
          1.0f);
  EXPECT_NEAR(glm::length(normal), 1.0f, 0.01f);
  EXPECT_GT(std::to_integer<uint8_t>(bytes[16u + 2u]), 220u);
}

TEST(TextureCacheUtilsTests, RoughnessMipSemanticUsesRmsChannelMips) {
  const ScopedTempDir dir("nuri_roughness_mips");
  const std::filesystem::path texturePath = dir.path / "roughness.tga";
  const std::array<uint8_t, 16> rgba{
      128u, 0u, 0u, 255u, 128u, 0u,   0u, 255u,
      128u, 0u, 0u, 255u, 128u, 255u, 0u, 255u,
  };
  writeRgbaTga(texturePath, 2u, 2u, rgba);

  nuri::test_support::FakeGPUDeviceBase gpu;
  auto texture = nuri::Texture::loadTexture(
      gpu, texturePath.string(),
      nuri::TextureLoadOptions{
          .srgb = false,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::RoughnessG,
      },
      "roughness_test");
  ASSERT_FALSE(texture.hasError()) << texture.error();
  ASSERT_EQ(gpu.createdTextureDescs.size(), 1u);
  ASSERT_EQ(gpu.createdTextureData.size(), 1u);

  const nuri::TextureDesc &desc = gpu.createdTextureDescs.back();
  EXPECT_EQ(desc.numMipLevels, 2u);
  EXPECT_EQ(desc.dataNumMipLevels, 2u);
  EXPECT_FALSE(desc.generateMipmaps);

  const std::vector<std::byte> &uploaded = gpu.createdTextureData.back();
  ASSERT_EQ(uploaded.size(), 20u);
  const uint8_t roughnessMip = std::to_integer<uint8_t>(uploaded[16u + 1u]);
  EXPECT_GT(roughnessMip, 100u);
  EXPECT_LT(roughnessMip, 150u);
}
