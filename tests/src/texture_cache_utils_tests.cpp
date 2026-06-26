#include "tests_pch.h"

#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"
#include "render_graph_test_support.h"

#include <array>
#include <chrono>
#include <fstream>
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

} // namespace

TEST(TextureCacheUtilsTests, BuildsPortableAndNativePaths) {
  const ScopedTempDir dir("nuri_texture_cache");
  const std::filesystem::path scenePath = dir.path / "models" / "helmet.glb";

  auto portablePathResult =
      nuri::buildPortableTextureCachePath(scenePath, 0x1234567890ABCDEFull);
  ASSERT_FALSE(portablePathResult.hasError()) << portablePathResult.error();

  const std::filesystem::path portablePath = portablePathResult.value();
  EXPECT_EQ(portablePath.filename(), "1234567890abcdef_basis.ktx2");
  EXPECT_EQ(portablePath.parent_path().filename(), "textures");
  EXPECT_EQ(portablePath.parent_path().parent_path().filename(),
            ".nuri_scene_cache");

  const std::filesystem::path nativePath = nuri::buildNativeTextureCachePath(
      portablePath, nuri::Format::BC7_RGBA_SRGB);
  EXPECT_EQ(nativePath.filename(), "1234567890abcdef_bc7_srgb_v1.ktx2");
  EXPECT_EQ(nativePath.parent_path().filename(), "native_textures");
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

TEST(TextureCacheUtilsTests, HashChangesForSrgbVsLinearSources) {
  constexpr std::string_view kScenePath = "e:/project/scene.glb";

  nuri::MaterialTextureSlotData slot{};
  slot.sourceKind = nuri::MaterialTextureSourceKind::ExternalFile;
  slot.path = "e:/project/textures/basecolor.png";

  const uint64_t srgbHash =
      nuri::hashSceneTextureSourceIdentity(kScenePath, slot, true, 1u);
  const uint64_t linearHash =
      nuri::hashSceneTextureSourceIdentity(kScenePath, slot, false, 1u);

  EXPECT_NE(srgbHash, linearHash);
}

TEST(TextureCacheUtilsTests, HashChangesForBakeSettingsTag) {
  constexpr std::string_view kScenePath = "e:/project/scene.glb";

  nuri::MaterialTextureSlotData slot{};
  slot.sourceKind = nuri::MaterialTextureSourceKind::ExternalFile;
  slot.path = "e:/project/textures/shared_linear.png";

  const uint64_t etc1sHash =
      nuri::hashSceneTextureSourceIdentity(kScenePath, slot, false, 1u);
  const uint64_t uastcHash =
      nuri::hashSceneTextureSourceIdentity(kScenePath, slot, false, 2u);

  EXPECT_NE(etc1sHash, uastcHash);
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

TEST(TextureCacheUtilsTests, TextureLoadOptionsHashTracksMipSemantic) {
  nuri::TextureLoadOptions generic{.srgb = true, .generateMipmaps = true};
  nuri::TextureLoadOptions alpha = generic;
  alpha.mipSemantic = nuri::TextureMipSemantic::AlphaCoverage;
  alpha.alphaCoverageCutoff = 0.4f;
  nuri::TextureLoadOptions normal = generic;
  normal.mipSemantic = nuri::TextureMipSemantic::NormalMap;
  nuri::TextureLoadOptions roughness = generic;
  roughness.mipSemantic = nuri::TextureMipSemantic::RoughnessG;

  EXPECT_NE(nuri::hashTextureLoadOptions(generic),
            nuri::hashTextureLoadOptions(alpha));
  EXPECT_NE(nuri::hashTextureLoadOptions(alpha),
            nuri::hashTextureLoadOptions(normal));
  EXPECT_NE(nuri::hashTextureLoadOptions(normal),
            nuri::hashTextureLoadOptions(roughness));
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
