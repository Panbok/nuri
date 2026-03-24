#include "tests_pch.h"

#include "nuri/resources/storage/texture/texture_cache_utils.h"

#include <chrono>
#include <fstream>
#include <thread>

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
    std::filesystem::create_directories(path);
  }

  ~ScopedTempDir() {
    std::error_code ec;
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
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  writeTextFile(cachePath, "cache");
  EXPECT_TRUE(nuri::isTextureCacheUpToDate(cachePath, sourcePath));

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  writeTextFile(sourcePath, "source-newer");
  EXPECT_FALSE(nuri::isTextureCacheUpToDate(cachePath, sourcePath));
}
