#include "tests_pch.h"

#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/gpu/resource_manager.h"
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

std::array<std::byte, 4> encodeNormal(glm::vec3 normal, uint8_t alpha = 255u) {
  normal = glm::normalize(normal);
  const auto encode = [](float value) {
    return static_cast<std::byte>(
        std::lround(std::clamp(value * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f));
  };
  return {encode(normal.x), encode(normal.y), encode(normal.z),
          static_cast<std::byte>(alpha)};
}

std::vector<std::byte> makeNormalImage(std::span<const glm::vec3> normals) {
  std::vector<std::byte> bytes;
  bytes.reserve(normals.size() * 4u);
  for (const glm::vec3 normal : normals) {
    const auto encoded = encodeNormal(normal);
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  }
  return bytes;
}

float decodedVariance(std::byte alpha) {
  return static_cast<float>(std::to_integer<uint8_t>(alpha)) / 255.0f *
         nuri::kEncodedSlopeVarianceMax;
}

} // namespace

TEST(TextureCacheUtilsTests, BuildsSourceDerivedNativeArtifactPaths) {
  const ScopedTempDir dir("nuri_texture_cache");
  const std::filesystem::path scenePath = dir.path / "models" / "helmet.glb";

  auto nativePathResult = nuri::buildNativeTextureArtifactPath(
      scenePath, 0x1234567890ABCDEFull, nuri::Format::BC7_RGBA_SRGB);
  ASSERT_FALSE(nativePathResult.hasError()) << nativePathResult.error();
  const std::filesystem::path nativePath = nativePathResult.value();
  EXPECT_EQ(nativePath.filename(), "1234567890abcdef_bc7_srgb_v4.ktx2");
  EXPECT_EQ(nativePath.parent_path().filename(), "native_textures");
  EXPECT_EQ(nativePath.parent_path().parent_path().filename(),
            ".nuri_scene_cache");
  EXPECT_EQ(nuri::buildNativeTextureCacheMetadataPath(nativePath).filename(),
            "1234567890abcdef_bc7_srgb_v4.ktx2.meta");
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
  const std::filesystem::path nativePath = dir.path / "source_bc7_v4.ktx2";
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
  auto packedFirst = cold.value().pack->readOwned(firstCanonical);
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
  auto packedSecond = rebuilt.value().pack->readOwned(
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

TEST(TextureCacheUtilsTests,
     CleanNormalVarianceFlatAndDegenerateInputsStayFiniteAndFlat) {
  constexpr uint32_t width = 3u;
  constexpr uint32_t height = 5u;
  std::vector<glm::vec3> normals(width * height, glm::vec3(0.0f, 0.0f, 1.0f));
  const std::vector<std::byte> base = makeNormalImage(normals);
  nuri::NormalVarianceBuildStats stats{};
  auto chain = nuri::generateSemanticRgba8MipChain(
      base, width, height, nuri::textureMipLevelCount(width, height),
      nuri::TextureLoadOptions{
          .srgb = false,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::NormalMap,
      },
      nuri::TextureContentContract::NormalRgbCleanVarianceA, &stats);
  ASSERT_FALSE(chain.hasError()) << chain.error();
  ASSERT_EQ(chain.value().size(), 72u);
  for (size_t offset = 0u; offset < chain.value().size(); offset += 4u) {
    EXPECT_NEAR(std::to_integer<uint8_t>(chain.value()[offset + 0u]), 128u, 1u);
    EXPECT_NEAR(std::to_integer<uint8_t>(chain.value()[offset + 1u]), 128u, 1u);
    EXPECT_EQ(std::to_integer<uint8_t>(chain.value()[offset + 2u]), 255u);
    EXPECT_EQ(std::to_integer<uint8_t>(chain.value()[offset + 3u]), 0u);
  }
  EXPECT_EQ(stats.cleanTexels, 18u);
  EXPECT_EQ(stats.toksvigFallbackTexels, 0u);

  const std::array<std::byte, 4> degenerate{std::byte{128}, std::byte{128},
                                            std::byte{128}, std::byte{37}};
  auto degenerateChain = nuri::generateSemanticRgba8MipChain(
      degenerate, 1u, 1u, 1u,
      nuri::TextureLoadOptions{
          .srgb = false,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::NormalMap,
      },
      nuri::TextureContentContract::NormalRgbCleanVarianceA);
  ASSERT_FALSE(degenerateChain.hasError()) << degenerateChain.error();
  EXPECT_EQ(std::to_integer<uint8_t>(degenerateChain.value()[0u]), 128u);
  EXPECT_EQ(std::to_integer<uint8_t>(degenerateChain.value()[1u]), 128u);
  EXPECT_EQ(std::to_integer<uint8_t>(degenerateChain.value()[2u]), 255u);
  EXPECT_EQ(std::to_integer<uint8_t>(degenerateChain.value()[3u]), 0u);
}

TEST(TextureCacheUtilsTests,
     CleanNormalVarianceMatchesSymmetricIsotropicSlopeReference) {
  constexpr float slope = 0.5f;
  const std::array<glm::vec3, 4> cardinal{
      glm::vec3(slope, 0.0f, 1.0f), glm::vec3(-slope, 0.0f, 1.0f),
      glm::vec3(0.0f, slope, 1.0f), glm::vec3(0.0f, -slope, 1.0f)};
  const std::vector<std::byte> cardinalBase = makeNormalImage(cardinal);
  auto cardinalChain = nuri::generateSemanticRgba8MipChain(
      cardinalBase, 2u, 2u, 2u,
      nuri::TextureLoadOptions{
          .srgb = false,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::NormalMap,
      },
      nuri::TextureContentContract::NormalRgbCleanVarianceA);
  ASSERT_FALSE(cardinalChain.hasError()) << cardinalChain.error();
  const size_t finalMipOffset = 2u * 2u * 4u;
  EXPECT_NEAR(decodedVariance(cardinalChain.value()[finalMipOffset + 3u]),
              slope * slope * 0.5f, 1.0f / 255.0f);
  EXPECT_EQ(
      std::to_integer<uint8_t>(cardinalChain.value()[finalMipOffset + 2u]),
      255u);

  constexpr float kInverseSqrtTwo = 0.7071067811865475f;
  const float diagonal = slope * kInverseSqrtTwo;
  const std::array<glm::vec3, 4> rotated{glm::vec3(diagonal, diagonal, 1.0f),
                                         glm::vec3(-diagonal, -diagonal, 1.0f),
                                         glm::vec3(-diagonal, diagonal, 1.0f),
                                         glm::vec3(diagonal, -diagonal, 1.0f)};
  const std::vector<std::byte> rotatedBase = makeNormalImage(rotated);
  auto rotatedChain = nuri::generateSemanticRgba8MipChain(
      rotatedBase, 2u, 2u, 2u,
      nuri::TextureLoadOptions{
          .srgb = false,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::NormalMap,
      },
      nuri::TextureContentContract::NormalRgbCleanVarianceA);
  ASSERT_FALSE(rotatedChain.hasError()) << rotatedChain.error();
  EXPECT_NEAR(decodedVariance(rotatedChain.value()[finalMipOffset + 3u]),
              decodedVariance(cardinalChain.value()[finalMipOffset + 3u]),
              1.0f / 255.0f);
}

TEST(TextureCacheUtilsTests,
     CleanNormalVarianceNpotReductionAndGrazingFallbackAreBounded) {
  constexpr uint32_t width = 3u;
  constexpr uint32_t height = 5u;
  std::array<glm::vec3, width * height> normals{};
  for (size_t i = 0u; i < normals.size(); ++i) {
    const float x = (static_cast<float>(i % width) - 1.0f) * 0.2f;
    const float y = (static_cast<float>(i / width) - 2.0f) * 0.1f;
    normals[i] = glm::vec3(x, y, 1.0f);
  }
  const std::vector<std::byte> base = makeNormalImage(normals);
  auto chain = nuri::generateSemanticRgba8MipChain(
      base, width, height, nuri::textureMipLevelCount(width, height),
      nuri::TextureLoadOptions{
          .srgb = false,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::NormalMap,
      },
      nuri::TextureContentContract::NormalRgbCleanVarianceA);
  ASSERT_FALSE(chain.hasError()) << chain.error();
  const size_t finalOffset = (width * height + 2u) * 4u;
  ASSERT_EQ(finalOffset + 4u, chain.value().size());
  EXPECT_GE(decodedVariance(chain.value()[finalOffset + 3u]), 0.0f);
  EXPECT_LE(decodedVariance(chain.value()[finalOffset + 3u]),
            nuri::kEncodedSlopeVarianceMax);

  const std::array<glm::vec3, 4> grazing{
      glm::vec3(1.0f, 0.0f, 0.1f), glm::vec3(-1.0f, 0.0f, 0.1f),
      glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f)};
  const std::vector<std::byte> grazingBase = makeNormalImage(grazing);
  nuri::NormalVarianceBuildStats stats{};
  auto grazingChain = nuri::generateSemanticRgba8MipChain(
      grazingBase, 2u, 2u, 2u,
      nuri::TextureLoadOptions{
          .srgb = false,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::NormalMap,
      },
      nuri::TextureContentContract::NormalRgbCleanVarianceA, &stats);
  ASSERT_FALSE(grazingChain.hasError()) << grazingChain.error();
  EXPECT_EQ(stats.toksvigFallbackTexels, 1u);
  const float fallbackVariance =
      decodedVariance(grazingChain.value()[2u * 2u * 4u + 3u]);
  EXPECT_TRUE(std::isfinite(fallbackVariance));
  EXPECT_GE(fallbackVariance, 0.0f);
  EXPECT_LE(fallbackVariance, nuri::kEncodedSlopeVarianceMax);
}

TEST(TextureCacheUtilsTests,
     CleanNormalVarianceEstimatorAndMinificationFiltersStayWithinV1Budget) {
  // The artifact is RGBA8, so step far enough to straddle the decoded UNORM
  // threshold rather than testing two source values that quantize identically.
  constexpr float kBoundaryDelta = 0.02f;
  const auto buildBoundary = [](float z,
                                nuri::NormalVarianceBuildStats &stats) {
    const float radial = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const std::array<glm::vec3, 4> normals{
        glm::vec3(radial, 0.0f, z), glm::vec3(-radial, 0.0f, z),
        glm::vec3(0.0f, radial, z), glm::vec3(0.0f, -radial, z)};
    return nuri::generateSemanticRgba8MipChain(
        makeNormalImage(normals), 2u, 2u, 2u,
        nuri::TextureLoadOptions{
            .srgb = false,
            .generateMipmaps = true,
            .mipSemantic = nuri::TextureMipSemantic::NormalMap,
        },
        nuri::TextureContentContract::NormalRgbCleanVarianceA, &stats);
  };

  nuri::NormalVarianceBuildStats belowStats{};
  nuri::NormalVarianceBuildStats aboveStats{};
  auto below =
      buildBoundary(nuri::kCleanNormalZFloor - kBoundaryDelta, belowStats);
  auto above =
      buildBoundary(nuri::kCleanNormalZFloor + kBoundaryDelta, aboveStats);
  ASSERT_FALSE(below.hasError()) << below.error();
  ASSERT_FALSE(above.hasError()) << above.error();
  const size_t finalMipOffset = 2u * 2u * 4u;
  const float belowVariance =
      decodedVariance(below.value()[finalMipOffset + 3u]);
  const float aboveVariance =
      decodedVariance(above.value()[finalMipOffset + 3u]);
  EXPECT_TRUE(std::isfinite(belowVariance));
  EXPECT_TRUE(std::isfinite(aboveVariance));
  EXPECT_GE(belowVariance, 0.0f);
  EXPECT_GE(aboveVariance, 0.0f);
  EXPECT_LE(belowVariance, nuri::kEncodedSlopeVarianceMax);
  EXPECT_LE(aboveVariance, nuri::kEncodedSlopeVarianceMax);
  EXPECT_GT(belowStats.toksvigFallbackTexels, 0u);
  EXPECT_EQ(aboveStats.toksvigFallbackTexels, 0u);
  EXPECT_EQ(aboveStats.cleanTexels, 5u);

  // Hardware filtering interpolates the stored scalar variance, while the
  // exact mixture also contains the between-mean term. This two-lobe case
  // makes that omission explicit and locks the v1 absolute error budget.
  constexpr float kLobeSlope = 0.5f;
  constexpr float kTrilinearBlend = 0.5f;
  constexpr float kV1ScalarInterpolationAbsoluteBudget = 0.033f;
  const std::array<glm::vec3, 4> lobes{
      glm::vec3(kLobeSlope, 0.0f, 1.0f), glm::vec3(-kLobeSlope, 0.0f, 1.0f),
      glm::vec3(kLobeSlope, 0.0f, 1.0f), glm::vec3(-kLobeSlope, 0.0f, 1.0f)};
  auto lobesChain = nuri::generateSemanticRgba8MipChain(
      makeNormalImage(lobes), 2u, 2u, 2u,
      nuri::TextureLoadOptions{
          .srgb = false,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::NormalMap,
      },
      nuri::TextureContentContract::NormalRgbCleanVarianceA);
  ASSERT_FALSE(lobesChain.hasError()) << lobesChain.error();
  const float integerLodVariance =
      decodedVariance(lobesChain.value()[finalMipOffset + 3u]);
  const float filteredHalfLodVariance =
      std::lerp(0.0f, integerLodVariance, kTrilinearBlend);
  const float exactHalfLodVariance =
      kTrilinearBlend * integerLodVariance + kTrilinearBlend *
                                                 (1.0f - kTrilinearBlend) *
                                                 kLobeSlope * kLobeSlope * 0.5f;
  const float omittedBetweenMeanTerm =
      exactHalfLodVariance - filteredHalfLodVariance;
  EXPECT_NEAR(integerLodVariance, kLobeSlope * kLobeSlope * 0.5f,
              1.0f / 255.0f);
  EXPECT_GT(omittedBetweenMeanTerm, 0.0f);
  EXPECT_LE(omittedBetweenMeanTerm, kV1ScalarInterpolationAbsoluteBudget);

  // Treat each hardware-filter tap as a lobe with a stored scalar variance.
  // The moment mixture is the supersampled reference; interpolating only the
  // stored scalar is the contract implemented by ordinary texture filtering.
  const auto referenceMixtureVariance = [](std::span<const glm::vec2> means,
                                           std::span<const float> variances,
                                           std::span<const float> weights) {
    EXPECT_EQ(means.size(), variances.size());
    EXPECT_EQ(means.size(), weights.size());
    glm::vec2 mixedMean{0.0f};
    float secondMoment = 0.0f;
    float weightSum = 0.0f;
    for (size_t i = 0u; i < means.size(); ++i) {
      mixedMean += weights[i] * means[i];
      secondMoment +=
          weights[i] * (variances[i] + 0.5f * glm::dot(means[i], means[i]));
      weightSum += weights[i];
    }
    EXPECT_NEAR(weightSum, 1.0f, 1.0e-6f);
    return secondMoment - 0.5f * glm::dot(mixedMean, mixedMean);
  };
  const auto filteredScalarVariance = [](std::span<const float> variances,
                                         std::span<const float> weights) {
    float result = 0.0f;
    for (size_t i = 0u; i < variances.size(); ++i) {
      result += weights[i] * variances[i];
    }
    return result;
  };

  // Bilinear footprint at (0.7, 0.4).
  const std::array bilinearMeans{
      glm::vec2(0.15f, 0.05f), glm::vec2(0.35f, 0.05f), glm::vec2(0.15f, 0.25f),
      glm::vec2(0.35f, 0.25f)};
  constexpr std::array bilinearVariances{0.08f, 0.08f, 0.08f, 0.08f};
  constexpr std::array bilinearWeights{0.18f, 0.42f, 0.12f, 0.28f};
  const float bilinearReference = referenceMixtureVariance(
      bilinearMeans, bilinearVariances, bilinearWeights);
  const float bilinearFiltered =
      filteredScalarVariance(bilinearVariances, bilinearWeights);
  EXPECT_GT(bilinearReference, bilinearFiltered);
  EXPECT_LE(bilinearReference - bilinearFiltered,
            kV1ScalarInterpolationAbsoluteBudget);

  // Trilinear blends two already-filtered mip lobes.
  const std::array trilinearMeans{glm::vec2(0.2f, -0.1f),
                                  glm::vec2(0.4f, 0.15f)};
  constexpr std::array trilinearVariances{0.07f, 0.11f};
  constexpr std::array trilinearWeights{0.5f, 0.5f};
  const float trilinearReference = referenceMixtureVariance(
      trilinearMeans, trilinearVariances, trilinearWeights);
  const float trilinearFiltered =
      filteredScalarVariance(trilinearVariances, trilinearWeights);
  EXPECT_GT(trilinearReference, trilinearFiltered);
  EXPECT_LE(trilinearReference - trilinearFiltered,
            kV1ScalarInterpolationAbsoluteBudget);

  // An eight-tap anisotropic footprint must stay rotation-invariant because
  // the v1 signal is isotropic scalar slope variance.
  std::array<glm::vec2, 8> anisotropicX{};
  std::array<glm::vec2, 8> anisotropicY{};
  std::array<float, 8> anisotropicVariances{};
  std::array<float, 8> anisotropicWeights{};
  for (size_t i = 0u; i < anisotropicX.size(); ++i) {
    const float offset = (static_cast<float>(i) - 3.5f) * 0.04f;
    anisotropicX[i] = glm::vec2(0.25f + offset, -0.1f);
    anisotropicY[i] = glm::vec2(0.1f, 0.25f + offset);
    anisotropicVariances[i] = 0.04f;
    anisotropicWeights[i] = 1.0f / 8.0f;
  }
  const float anisotropicXReference = referenceMixtureVariance(
      anisotropicX, anisotropicVariances, anisotropicWeights);
  const float anisotropicYReference = referenceMixtureVariance(
      anisotropicY, anisotropicVariances, anisotropicWeights);
  const float anisotropicFiltered =
      filteredScalarVariance(anisotropicVariances, anisotropicWeights);
  EXPECT_NEAR(anisotropicXReference, anisotropicYReference, 1.0e-6f);
  EXPECT_LE(anisotropicXReference - anisotropicFiltered,
            kV1ScalarInterpolationAbsoluteBudget);

  // Animated fractional LOD: sweep the blend continuously and compare each
  // hardware-style scalar interpolation against the exact moment mixture.
  float previousReference = trilinearVariances.front();
  for (uint32_t step = 0u; step <= 32u; ++step) {
    const float t = static_cast<float>(step) / 32.0f;
    const std::array lodWeights{1.0f - t, t};
    const float reference = referenceMixtureVariance(
        trilinearMeans, trilinearVariances, lodWeights);
    const float filtered =
        filteredScalarVariance(trilinearVariances, lodWeights);
    const float approximationError = reference - filtered;
    EXPECT_GE(approximationError, -1.0e-6f);
    EXPECT_LE(approximationError, kV1ScalarInterpolationAbsoluteBudget);
    if (step != 0u) {
      EXPECT_LE(std::abs(reference - previousReference), 0.003f);
    }
    previousReference = reference;
  }
}

TEST(TextureCacheUtilsTests,
     CleanNormalVarianceUastcRoundTripPreservesAlphaSignalAndBounds) {
  const ScopedTempDir dir("nuri_clean_variance_uastc");
  const std::filesystem::path sourcePath = dir.path / "clean_normal.tga";
  constexpr uint16_t width = 4u;
  constexpr uint16_t height = 4u;
  std::vector<uint8_t> rgba;
  rgba.reserve(static_cast<size_t>(width) * height * 4u);
  for (uint32_t y = 0u; y < height; ++y) {
    for (uint32_t x = 0u; x < width; ++x) {
      const float sx = (x & 1u) == 0u ? -0.55f : 0.55f;
      const float sy = (y & 1u) == 0u ? -0.35f : 0.35f;
      const auto encoded = encodeNormal(glm::vec3(sx, sy, 1.0f));
      for (std::byte value : encoded) {
        rgba.push_back(std::to_integer<uint8_t>(value));
      }
    }
  }
  writeRgbaTga(sourcePath, width, height, rgba);

  const nuri::TextureArtifactBuildOptions options{
      .loadOptions =
          nuri::TextureLoadOptions{
              .srgb = false,
              .generateMipmaps = true,
              .mipSemantic = nuri::TextureMipSemantic::NormalMap,
          },
      .encoding = nuri::TextureArtifactEncoding::Uastc,
      .contentContract = nuri::TextureContentContract::NormalRgbCleanVarianceA,
  };
  const uint64_t identity = nuri::hashTextureSourceIdentity(
      nuri::canonicalizeResourcePath(sourcePath.string()), false,
      nuri::textureArtifactProcessingTag(options));
  auto artifact = nuri::ensureTextureArtifactFromFile(
      sourcePath, identity, nuri::Format::RGBA8_UNORM, options, true);
  ASSERT_FALSE(artifact.hasError()) << artifact.error();
  EXPECT_TRUE(artifact.value().built);

  nuri::test_support::FakeGPUDeviceBase gpu;
  auto texture = nuri::Texture::loadTextureKtx2(
      gpu, artifact.value().artifactPath.string(), "clean_variance_uastc");
  ASSERT_FALSE(texture.hasError()) << texture.error();
  ASSERT_EQ(gpu.createdTextureDescs.size(), 1u);
  ASSERT_EQ(gpu.createdTextureData.size(), 1u);
  EXPECT_EQ(gpu.createdTextureDescs[0].format, nuri::Format::RGBA8_UNORM);
  const std::vector<std::byte> &uploaded = gpu.createdTextureData[0];
  ASSERT_EQ(uploaded.size(), 84u);

  uint8_t baseAlphaMax = 0u;
  for (size_t offset = 3u; offset < 64u; offset += 4u) {
    baseAlphaMax =
        std::max(baseAlphaMax, std::to_integer<uint8_t>(uploaded[offset]));
  }
  const uint8_t mipOneAlpha = std::to_integer<uint8_t>(uploaded[64u + 3u]);
  const uint8_t finalAlpha = std::to_integer<uint8_t>(uploaded[80u + 3u]);
  EXPECT_LE(baseAlphaMax, 16u);
  EXPECT_GT(mipOneAlpha, baseAlphaMax);
  EXPECT_GT(finalAlpha, baseAlphaMax);
  EXPECT_LE(finalAlpha, 255u);
}

TEST(TextureCacheUtilsTests,
     NormalVarianceContractParticipatesInIdentityAndMetadataAuthorization) {
  const nuri::TextureKey rgb{
      .canonicalPath = "normal.ktx2",
      .optionsHash = 7u,
      .kind = nuri::TextureRequestKind::Ktx2Texture2D,
      .contentContract = nuri::TextureContentContract::NormalRgb,
  };
  nuri::TextureKey clean = rgb;
  clean.contentContract = nuri::TextureContentContract::NormalRgbCleanVarianceA;
  EXPECT_NE(rgb, clean);
  EXPECT_NE(nuri::TextureKeyHash{}(rgb), nuri::TextureKeyHash{}(clean));

  const ScopedTempDir dir("nuri_texture_contract_metadata");
  const std::filesystem::path sourcePath = dir.path / "normal.tga";
  const std::filesystem::path nativePath = dir.path / "normal_v4.ktx2";
  writeTextFile(sourcePath, "source");
  writeTextFile(nativePath, "artifact");
  auto fingerprint = nuri::queryTextureSourceFingerprint(sourcePath);
  ASSERT_FALSE(fingerprint.hasError()) << fingerprint.error();
  constexpr uint64_t identity = 42u;
  const nuri::NativeTextureCacheMetadata metadata{
      .profileVersion = nuri::kNativeTextureArtifactProfileVersion,
      .sourceIdentityHash = identity,
      .source = fingerprint.value(),
      .targetFormat = nuri::Format::RGBA8_UNORM,
      .textureType = nuri::TextureType::Texture2D,
      .dimensions = {.width = 1u, .height = 1u, .depth = 1u},
      .numLayers = 1u,
      .numFaces = 1u,
      .numMipLevels = 1u,
      .payloadSizeBytes = 4u,
      .artifactSizeBytes = std::filesystem::file_size(nativePath),
      .contentContract = nuri::TextureContentContract::NormalRgbCleanVarianceA,
      .contentEncodingVersion = nuri::kNormalVarianceEncodingVersion,
  };
  ASSERT_FALSE(nuri::writeNativeTextureCacheMetadataAtomic(nativePath, metadata)
                   .hasError());
  EXPECT_EQ(nuri::probeNativeTextureCache(
                nativePath, sourcePath, identity, nuri::Format::RGBA8_UNORM,
                nuri::TextureContentContract::NormalRgbCleanVarianceA,
                nuri::kNormalVarianceEncodingVersion)
                .status,
            nuri::NativeTextureCacheProbeStatus::Hit);
  EXPECT_EQ(nuri::probeNativeTextureCache(
                nativePath, sourcePath, identity, nuri::Format::RGBA8_UNORM,
                nuri::TextureContentContract::NormalRgb, 0u)
                .status,
            nuri::NativeTextureCacheProbeStatus::Stale);

  auto incompatible = nuri::generateSemanticRgba8MipChain(
      std::array<std::byte, 4>{}, 1u, 1u, 1u,
      nuri::TextureLoadOptions{
          .srgb = true,
          .generateMipmaps = true,
          .mipSemantic = nuri::TextureMipSemantic::NormalMap,
      },
      nuri::TextureContentContract::NormalRgbCleanVarianceA);
  EXPECT_TRUE(incompatible.hasError());
}

TEST(TextureCacheUtilsTests,
     MaterialVarianceFlagsAndLiveAggregatesFollowTextureContractsExactly) {
  const ScopedTempDir dir("nuri_material_variance_projection");
  const std::filesystem::path cleanPath = dir.path / "clean_normal_v4.ktx2";
  writeTextFile(cleanPath, "artifact");
  const nuri::NativeTextureCacheMetadata cleanMetadata{
      .profileVersion = nuri::kNativeTextureArtifactProfileVersion,
      .sourceIdentityHash = 9u,
      .targetFormat = nuri::Format::RGBA8_UNORM,
      .textureType = nuri::TextureType::Texture2D,
      .dimensions = {.width = 1u, .height = 1u, .depth = 1u},
      .numLayers = 1u,
      .numFaces = 1u,
      .numMipLevels = 1u,
      .payloadSizeBytes = 4u,
      .artifactSizeBytes = std::filesystem::file_size(cleanPath),
      .contentContract = nuri::TextureContentContract::NormalRgbCleanVarianceA,
      .contentEncodingVersion = nuri::kNormalVarianceEncodingVersion,
  };
  ASSERT_FALSE(
      nuri::writeNativeTextureCacheMetadataAtomic(cleanPath, cleanMetadata)
          .hasError());

  nuri::test_support::FakeGPUDeviceBase gpu;
  nuri::ResourceManager resources(gpu);
  const nuri::TextureDesc textureDesc{
      .type = nuri::TextureType::Texture2D,
      .format = nuri::Format::RGBA8_UNORM,
      .dimensions = {.width = 1u, .height = 1u, .depth = 1u},
      .usage = nuri::TextureUsage::Sampled,
      .storage = nuri::Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = 1u,
  };
  auto cleanHandle = gpu.createTexture(textureDesc, "clean_normal");
  ASSERT_FALSE(cleanHandle.hasError()) << cleanHandle.error();
  auto cleanTexture = nuri::Texture::adoptPrepared(gpu, cleanHandle.value(),
                                                   textureDesc, "clean_normal");
  auto cleanRef = resources.adoptPreparedTexture(
      nuri::TextureRequest{
          .path = cleanPath.string(),
          .loadOptions =
              nuri::TextureLoadOptions{
                  .srgb = false,
                  .generateMipmaps = true,
                  .mipSemantic = nuri::TextureMipSemantic::NormalMap,
              },
          .contentContract =
              nuri::TextureContentContract::NormalRgbCleanVarianceA,
          .kind = nuri::TextureRequestKind::Ktx2Texture2D,
          .debugName = "clean_normal",
      },
      std::move(cleanTexture));
  ASSERT_FALSE(cleanRef.hasError()) << cleanRef.error();

  const std::filesystem::path genericPath = dir.path / "generic_normal.tga";
  auto genericHandle = gpu.createTexture(textureDesc, "generic_normal");
  ASSERT_FALSE(genericHandle.hasError()) << genericHandle.error();
  auto genericTexture = nuri::Texture::adoptPrepared(
      gpu, genericHandle.value(), textureDesc, "generic_normal");
  auto genericRef = resources.adoptPreparedTexture(
      nuri::TextureRequest{
          .path = genericPath.string(),
          .loadOptions =
              nuri::TextureLoadOptions{
                  .srgb = false,
                  .generateMipmaps = true,
                  .mipSemantic = nuri::TextureMipSemantic::NormalMap,
              },
          .contentContract = nuri::TextureContentContract::NormalRgb,
          .kind = nuri::TextureRequestKind::Texture2D,
          .debugName = "generic_normal",
      },
      std::move(genericTexture));
  ASSERT_FALSE(genericRef.hasError()) << genericRef.error();

  nuri::MaterialRequest materialRequest{};
  materialRequest.sourceIdentity = "variance_material";
  materialRequest.textureRefs[nuri::kMaterialTextureSlotNormal] =
      cleanRef.value();
  materialRequest.textureRefs[nuri::kMaterialTextureSlotClearcoatNormal] =
      genericRef.value();
  auto material = resources.acquireMaterial(materialRequest);
  ASSERT_FALSE(material.hasError()) << material.error();
  const uint32_t tableIndex = resources.materialTableIndex(material.value());
  const nuri::MaterialTableSnapshot snapshot = resources.materialSnapshot();
  ASSERT_LT(tableIndex, snapshot.headers.size());
  EXPECT_NE(snapshot.headers[tableIndex].materialFlags &
                nuri::kMaterialFlagsBaseNormalVarianceBit,
            0u);
  EXPECT_EQ(snapshot.headers[tableIndex].materialFlags &
                nuri::kMaterialFlagsClearcoatNormalVarianceBit,
            0u);
  nuri::PoolStats stats = resources.stats();
  EXPECT_EQ(stats.normalVarianceContractTexturesLive, 1u);
  EXPECT_EQ(stats.normalVarianceContractMaterialsLive, 1u);
  EXPECT_EQ(stats.normalVarianceUnavailableSlotsLive, 1u);
  EXPECT_EQ(stats.normalVarianceContractTextureBytesLive, 4u);

  resources.release(material.value());
  resources.collectGarbage();
  nuri::MaterialRequest replacement{};
  replacement.sourceIdentity = "generic_replacement";
  replacement.textureRefs[nuri::kMaterialTextureSlotNormal] =
      genericRef.value();
  auto genericMaterial = resources.acquireMaterial(replacement);
  ASSERT_FALSE(genericMaterial.hasError()) << genericMaterial.error();
  const uint32_t replacementIndex =
      resources.materialTableIndex(genericMaterial.value());
  const nuri::MaterialTableSnapshot replacementSnapshot =
      resources.materialSnapshot();
  ASSERT_LT(replacementIndex, replacementSnapshot.headers.size());
  EXPECT_EQ(replacementSnapshot.headers[replacementIndex].materialFlags &
                nuri::kMaterialFlagsNormalVarianceMask,
            0u);
  stats = resources.stats();
  EXPECT_EQ(stats.normalVarianceContractMaterialsLive, 0u);
  EXPECT_EQ(stats.normalVarianceUnavailableSlotsLive, 1u);
  EXPECT_EQ(stats.normalVarianceContractTextureBytesLive, 4u);

  resources.release(genericMaterial.value());
  resources.release(cleanRef.value());
  resources.release(genericRef.value());
  resources.collectGarbage();
  stats = resources.stats();
  EXPECT_EQ(stats.normalVarianceContractTexturesLive, 0u);
  EXPECT_EQ(stats.normalVarianceContractMaterialsLive, 0u);
  EXPECT_EQ(stats.normalVarianceUnavailableSlotsLive, 0u);
}
