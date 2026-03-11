#include "tests_pch.h"

#include "nuri/resources/mesh_importer.h"

namespace {

std::string modelPath(std::string_view relativePath) {
  const std::filesystem::path root(PROJECT_SOURCE_DIR);
  return (root / "assets" / "models" / std::filesystem::path(relativePath))
      .string();
}

TEST(MaterialImportTests, ClearcoatWickerOverlayImportsClearcoatData) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("ClearcoatWicker/ClearcoatWicker.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialSet &set = result.value();
  ASSERT_FALSE(set.materials.empty());

  const auto it = std::find_if(set.materials.begin(), set.materials.end(),
                               [](const nuri::ImportedMaterialInfo &material) {
                                 return material.clearcoatFactor > 0.0f ||
                                        !material.clearcoatNormal.path.empty();
                               });
  ASSERT_NE(it, set.materials.end());

  const nuri::ImportedMaterialInfo &material = *it;
  EXPECT_FLOAT_EQ(material.clearcoatFactor, 1.0f);
  EXPECT_FLOAT_EQ(material.clearcoatRoughnessFactor, 0.1f);
  EXPECT_FLOAT_EQ(material.clearcoatNormalScale, 1.0f);
  EXPECT_TRUE(material.clearcoat.path.empty());
  EXPECT_TRUE(material.clearcoatRoughness.path.empty());
  EXPECT_FALSE(material.clearcoatNormal.path.empty());
  EXPECT_TRUE(
      std::filesystem::path(material.clearcoatNormal.path).is_absolute());
  EXPECT_EQ(std::filesystem::path(material.clearcoatNormal.path).filename(),
            std::filesystem::path("clearcoat_normal.png"));

  EXPECT_FALSE(material.baseColor.path.empty());
  EXPECT_FALSE(material.normal.path.empty());
  EXPECT_FALSE(material.metallicRoughness.path.empty());
}

TEST(MaterialImportTests, DamagedHelmetLeavesClearcoatDisabled) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("DamagedHelmet/DamagedHelmet.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialSet &set = result.value();
  ASSERT_FALSE(set.materials.empty());

  for (size_t i = 0; i < set.materials.size(); ++i) {
    const nuri::ImportedMaterialInfo &material = set.materials[i];
    SCOPED_TRACE(::testing::Message()
                 << "material[" << i << "] name=\"" << material.name << "\"");
    EXPECT_FLOAT_EQ(material.clearcoatFactor, 0.0f);
    EXPECT_FLOAT_EQ(material.clearcoatRoughnessFactor, 0.0f);
    EXPECT_FLOAT_EQ(material.clearcoatNormalScale, 1.0f);
    EXPECT_TRUE(material.clearcoat.path.empty());
    EXPECT_TRUE(material.clearcoatRoughness.path.empty());
    EXPECT_TRUE(material.clearcoatNormal.path.empty());
  }

  const auto texturedIt =
      std::find_if(set.materials.begin(), set.materials.end(),
                   [](const nuri::ImportedMaterialInfo &material) {
                     return !material.baseColor.path.empty();
                   });
  ASSERT_NE(texturedIt, set.materials.end());
  EXPECT_FALSE(texturedIt->normal.path.empty());
  EXPECT_FALSE(texturedIt->metallicRoughness.path.empty());
}

} // namespace
