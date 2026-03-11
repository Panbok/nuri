#include "tests_pch.h"

#include "nuri/resources/mesh_importer.h"

namespace {

std::string modelPath(std::string_view relativePath) {
  const std::filesystem::path root(PROJECT_SOURCE_DIR);
  return (root / "assets" / "models" / std::filesystem::path(relativePath))
      .string();
}

const nuri::ImportedMaterialInfo *findMaterialByName(
    const nuri::ImportedMaterialSet &set, std::string_view name) {
  const auto it = std::find_if(
      set.materials.begin(), set.materials.end(),
      [name](const nuri::ImportedMaterialInfo &material) {
        return material.name == name;
      });
  return (it == set.materials.end()) ? nullptr : &(*it);
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

TEST(MaterialImportTests, SheenChairImportsSheenFactorsAndTextures) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("SheenChair/SheenChair.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialSet &set = result.value();
  ASSERT_FALSE(set.materials.empty());

  const nuri::ImportedMaterialInfo *mangoVelvet =
      findMaterialByName(set, "fabric Mystere Mango Velvet");
  ASSERT_NE(mangoVelvet, nullptr);
  EXPECT_FLOAT_EQ(mangoVelvet->sheenColorFactor.x, 1.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->sheenColorFactor.y, 0.329f);
  EXPECT_FLOAT_EQ(mangoVelvet->sheenColorFactor.z, 0.1f);
  EXPECT_FLOAT_EQ(mangoVelvet->sheenRoughnessFactor, 0.8f);
  EXPECT_FLOAT_EQ(mangoVelvet->sheenWeight, 1.0f);
  EXPECT_TRUE(mangoVelvet->sheenColor.path.empty());
  EXPECT_TRUE(mangoVelvet->sheenRoughness.path.empty());

  const nuri::ImportedMaterialInfo *peacockVelvet =
      findMaterialByName(set, "fabric Mystere Peacock Velvet");
  ASSERT_NE(peacockVelvet, nullptr);
  EXPECT_FLOAT_EQ(peacockVelvet->sheenColorFactor.x, 0.013f);
  EXPECT_FLOAT_EQ(peacockVelvet->sheenColorFactor.y, 0.284f);
  EXPECT_FLOAT_EQ(peacockVelvet->sheenColorFactor.z, 0.298f);
  EXPECT_FLOAT_EQ(peacockVelvet->sheenRoughnessFactor, 0.8f);
  EXPECT_FLOAT_EQ(peacockVelvet->sheenWeight, 1.0f);
}

TEST(MaterialImportTests, SheenChairImportsTextureTransforms) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("SheenChair/SheenChair.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialSet &set = result.value();
  const nuri::ImportedMaterialInfo *mangoVelvet =
      findMaterialByName(set, "fabric Mystere Mango Velvet");
  ASSERT_NE(mangoVelvet, nullptr);

  EXPECT_EQ(mangoVelvet->baseColor.uvSet, 0u);
  EXPECT_FLOAT_EQ(mangoVelvet->baseColor.transform.offset.x, -3.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->baseColor.transform.offset.y, 3.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->baseColor.transform.scale.x, 7.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->baseColor.transform.scale.y, 7.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->baseColor.transform.rotationRadians, 0.0f);

  EXPECT_EQ(mangoVelvet->normal.uvSet, 0u);
  EXPECT_FLOAT_EQ(mangoVelvet->normalScale, 0.6f);
  EXPECT_FLOAT_EQ(mangoVelvet->normal.transform.offset.x, -0.5f);
  EXPECT_FLOAT_EQ(mangoVelvet->normal.transform.offset.y, 0.5f);
  EXPECT_FLOAT_EQ(mangoVelvet->normal.transform.scale.x, 2.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->normal.transform.scale.y, 2.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->normal.transform.rotationRadians, 0.0f);

  EXPECT_EQ(mangoVelvet->occlusion.uvSet, 1u);
  EXPECT_FLOAT_EQ(mangoVelvet->occlusion.transform.offset.x, 0.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->occlusion.transform.offset.y, 0.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->occlusion.transform.scale.x, 1.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->occlusion.transform.scale.y, 1.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->occlusion.transform.rotationRadians, 0.0f);

  const nuri::ImportedMaterialInfo *woodBrown =
      findMaterialByName(set, "wood Brown");
  ASSERT_NE(woodBrown, nullptr);
  EXPECT_EQ(woodBrown->baseColor.uvSet, 0u);
  EXPECT_NEAR(woodBrown->baseColor.transform.offset.x, -0.8635584f, 1.0e-6f);
  EXPECT_NEAR(woodBrown->baseColor.transform.offset.y, 1.12502563f, 1.0e-6f);
  EXPECT_FLOAT_EQ(woodBrown->baseColor.transform.scale.x, 3.0f);
  EXPECT_FLOAT_EQ(woodBrown->baseColor.transform.scale.y, 3.0f);
  EXPECT_NEAR(woodBrown->baseColor.transform.rotationRadians, 0.08726647f,
              1.0e-6f);
}

TEST(MaterialImportTests, SheenChairLeavesVariantsUnapplied) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("SheenChair/SheenChair.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialSet &set = result.value();
  ASSERT_FALSE(set.materials.empty());
  EXPECT_NE(findMaterialByName(set, "fabric Mystere Mango Velvet"), nullptr);
  EXPECT_NE(findMaterialByName(set, "fabric Mystere Peacock Velvet"), nullptr);
}

} // namespace
