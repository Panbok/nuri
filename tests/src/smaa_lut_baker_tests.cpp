#include "tests_pch.h"

#include "nuri/bakery/smaa_lut_baker.h"
#include "nuri/gfx/smaa_lut_contract.h"

namespace {

constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

[[nodiscard]] uint64_t fnv1a(std::span<const std::byte> bytes) {
  uint64_t hash = kFnvOffsetBasis;
  for (const std::byte value : bytes) {
    hash ^= std::to_integer<uint8_t>(value);
    hash *= kFnvPrime;
  }
  return hash;
}

struct ScopedTempDir {
  ScopedTempDir() {
    path = std::filesystem::temp_directory_path() /
           ("nuri_smaa_lut_baker_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    EXPECT_FALSE(ec) << ec.message();
  }

  ~ScopedTempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }

  std::filesystem::path path;
};

TEST(SmaaLutBakerTests, GeneratesCanonicalRendererArtifacts) {
  const nuri::bakery::detail::SmaaLutArtifacts artifacts =
      nuri::bakery::detail::generateSmaaLutArtifacts();

  ASSERT_EQ(artifacts.areaRgba8.size(), nuri::smaa_lut::kAreaByteCount);
  ASSERT_EQ(artifacts.searchRgba8.size(), nuri::smaa_lut::kSearchByteCount);
  EXPECT_EQ(fnv1a(artifacts.areaRgba8), 0x7eea377052d0acedull);
  EXPECT_EQ(fnv1a(artifacts.searchRgba8), 0x00a22028db3f6325ull);
}

TEST(SmaaLutBakerTests, MissingArtifactsBakeThenBecomeUpToDate) {
  ScopedTempDir temp;
  nuri::RuntimeConfig config{};
  config.roots.shaders = temp.path / "shaders";

  auto missingPlan = nuri::bakery::detail::planSmaaLutBake(config, false);
  ASSERT_TRUE(missingPlan.shouldBake);

  auto bakeResult = nuri::bakery::detail::bakeSmaaLutsToDisk(missingPlan);
  ASSERT_FALSE(bakeResult.hasError()) << bakeResult.error();
  EXPECT_EQ(std::filesystem::file_size(missingPlan.areaOutputPath),
            nuri::smaa_lut::kAreaByteCount);
  EXPECT_EQ(std::filesystem::file_size(missingPlan.searchOutputPath),
            nuri::smaa_lut::kSearchByteCount);

  auto currentPlan = nuri::bakery::detail::planSmaaLutBake(config, false);
  EXPECT_FALSE(currentPlan.shouldBake);
}

} // namespace
