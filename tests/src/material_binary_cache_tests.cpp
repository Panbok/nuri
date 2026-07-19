#include "tests_pch.h"

#include "nuri/resources/storage/material/material_binary_serializer.h"

namespace {

nuri::SceneMaterialRecord makeSceneMaterialRecord() {
  nuri::SceneMaterialRecord record{};
  record.sourceMaterialIndex = 7u;
  record.sourceMaterial.name = "artifact_material";
  record.sourceMaterial.workflow = nuri::MaterialWorkflow::SpecularGlossiness;
  record.sourceMaterial.baseColorFactor = {0.1f, 0.2f, 0.3f, 0.4f};
  record.sourceMaterial.emissiveFactor = {1.0f, 2.0f, 3.0f};
  record.sourceMaterial.emissiveStrength = 4.0f;
  record.sourceMaterial.metallicFactor = 0.25f;
  record.sourceMaterial.roughnessFactor = 0.75f;
  record.sourceMaterial.specularColorFactor = {0.6f, 0.7f, 0.8f};
  record.sourceMaterial.specularFactor = 0.5f;
  record.sourceMaterial.glossinessFactor = 0.25f;
  record.sourceMaterial.sheenColorFactor = {0.11f, 0.22f, 0.33f};
  record.sourceMaterial.sheenWeight = 0.44f;
  record.sourceMaterial.sheenRoughnessFactor = 0.55f;
  record.sourceMaterial.clearcoatFactor = 0.66f;
  record.sourceMaterial.clearcoatRoughnessFactor = 0.77f;
  record.sourceMaterial.clearcoatNormalScale = 0.88f;
  record.sourceMaterial.transmissionFactor = 0.15f;
  record.sourceMaterial.thicknessFactor = 0.35f;
  record.sourceMaterial.attenuationColor = {0.9f, 0.8f, 0.7f};
  record.sourceMaterial.attenuationDistance = 12.0f;
  record.sourceMaterial.ior = 1.33f;
  record.sourceMaterial.normalScale = 0.95f;
  record.sourceMaterial.occlusionStrength = 0.65f;
  record.sourceMaterial.alphaCutoff = 0.42f;
  record.sourceMaterial.doubleSided = true;
  record.sourceMaterial.alphaMode = nuri::MaterialAlphaMode::Blend;

  auto &baseColor =
      record.sourceMaterial.textures[nuri::kMaterialTextureSlotBaseColor];
  baseColor.path = "E:/tmp/albedo.png";
  baseColor.sourceKind = nuri::MaterialTextureSourceKind::ExternalFile;
  baseColor.uvSet = 2u;
  baseColor.samplerIndex = 3u;
  baseColor.scale = 1.5f;
  baseColor.transform.offset = {0.25f, 0.5f};
  baseColor.transform.scale = {2.0f, 3.0f};
  baseColor.transform.rotationRadians = 0.75f;

  auto &normal =
      record.sourceMaterial.textures[nuri::kMaterialTextureSlotNormal];
  normal.sourceKind = nuri::MaterialTextureSourceKind::EmbeddedSceneTexture;
  normal.embeddedIndex = 5u;
  normal.uvSet = 1u;
  normal.samplerIndex = 4u;
  normal.scale = 0.8f;
  normal.transform.offset = {-1.0f, 1.5f};
  normal.transform.scale = {4.0f, 5.0f};
  normal.transform.rotationRadians = 0.25f;

  auto &specGloss =
      record.sourceMaterial.textures[nuri::kMaterialTextureSlotSpecularColor];
  specGloss.path = "E:/tmp/specgloss.png";
  specGloss.sourceKind = nuri::MaterialTextureSourceKind::ExternalFile;
  specGloss.uvSet = 1u;
  specGloss.samplerIndex = 2u;
  specGloss.transform.scale = {6.0f, 7.0f};

  record.textureCache[0].artifactIdentityHash = 0x1122334455667788ull;
  record.textureCache[2].artifactIdentityHash = 0x8877665544332211ull;
  return record;
}

} // namespace

TEST(MaterialBinaryCacheTests, RoundTripPreservesMaterialFields) {
  const nuri::SceneMaterialRecord inputRecord = makeSceneMaterialRecord();
  std::array<nuri::SceneMaterialRecord, 1> records = {inputRecord};

  nuri::MaterialBinarySerializeInput input{};
  input.sourcePathHash = 0xABCDEF1234567890ull;
  input.sourceSizeBytes = 123456u;
  input.sourceMtimeNs = 987654321ll;
  input.materials = std::span<const nuri::SceneMaterialRecord>(records.data(),
                                                               records.size());

  auto serializeResult = nuri::materialBinarySerialize(input);
  ASSERT_FALSE(serializeResult.hasError()) << serializeResult.error();

  nuri::MaterialBinaryDeserializeContext context{};
  context.expectedSourcePathHash = input.sourcePathHash;
  context.validateSourceFingerprint = true;
  context.sourceExists = true;
  context.sourceSizeBytes = input.sourceSizeBytes;
  context.sourceMtimeNs = input.sourceMtimeNs;

  auto deserializeResult =
      nuri::materialBinaryDeserialize(serializeResult.value(), context);
  ASSERT_FALSE(deserializeResult.hasError())
      << deserializeResult.error().message;
  ASSERT_EQ(deserializeResult.value().materials.size(), 1u);

  const nuri::SceneMaterialRecord &output =
      deserializeResult.value().materials[0];
  EXPECT_EQ(output.sourceMaterialIndex, inputRecord.sourceMaterialIndex);
  EXPECT_EQ(output.sourceMaterial.name, inputRecord.sourceMaterial.name);
  EXPECT_EQ(output.sourceMaterial.workflow,
            inputRecord.sourceMaterial.workflow);
  EXPECT_EQ(output.sourceMaterial.alphaMode,
            inputRecord.sourceMaterial.alphaMode);
  EXPECT_EQ(output.sourceMaterial.doubleSided,
            inputRecord.sourceMaterial.doubleSided);
  EXPECT_FLOAT_EQ(output.sourceMaterial.baseColorFactor.x,
                  inputRecord.sourceMaterial.baseColorFactor.x);
  EXPECT_FLOAT_EQ(output.sourceMaterial.emissiveFactor.z,
                  inputRecord.sourceMaterial.emissiveFactor.z);
  EXPECT_FLOAT_EQ(output.sourceMaterial.clearcoatFactor,
                  inputRecord.sourceMaterial.clearcoatFactor);
  EXPECT_FLOAT_EQ(output.sourceMaterial.glossinessFactor,
                  inputRecord.sourceMaterial.glossinessFactor);
  EXPECT_FLOAT_EQ(output.sourceMaterial.ior, inputRecord.sourceMaterial.ior);

  EXPECT_EQ(output.sourceMaterial.textures[nuri::kMaterialTextureSlotBaseColor]
                .sourceKind,
            nuri::MaterialTextureSourceKind::ExternalFile);
  EXPECT_EQ(
      output.sourceMaterial.textures[nuri::kMaterialTextureSlotBaseColor].path,
      inputRecord.sourceMaterial.textures[nuri::kMaterialTextureSlotBaseColor]
          .path);
  EXPECT_EQ(
      output.sourceMaterial.textures[nuri::kMaterialTextureSlotBaseColor].uvSet,
      inputRecord.sourceMaterial.textures[nuri::kMaterialTextureSlotBaseColor]
          .uvSet);
  EXPECT_FLOAT_EQ(
      output.sourceMaterial.textures[nuri::kMaterialTextureSlotBaseColor]
          .transform.scale.y,
      inputRecord.sourceMaterial.textures[nuri::kMaterialTextureSlotBaseColor]
          .transform.scale.y);

  EXPECT_EQ(output.sourceMaterial.textures[nuri::kMaterialTextureSlotNormal]
                .sourceKind,
            nuri::MaterialTextureSourceKind::EmbeddedSceneTexture);
  EXPECT_EQ(
      output.sourceMaterial.textures[nuri::kMaterialTextureSlotNormal]
          .embeddedIndex,
      inputRecord.sourceMaterial.textures[nuri::kMaterialTextureSlotNormal]
          .embeddedIndex);
  EXPECT_FLOAT_EQ(
      output.sourceMaterial.textures[nuri::kMaterialTextureSlotNormal]
          .transform.rotationRadians,
      inputRecord.sourceMaterial.textures[nuri::kMaterialTextureSlotNormal]
          .transform.rotationRadians);
  EXPECT_EQ(
      output.sourceMaterial.textures[nuri::kMaterialTextureSlotSpecularColor]
          .path,
      inputRecord.sourceMaterial
          .textures[nuri::kMaterialTextureSlotSpecularColor]
          .path);
  EXPECT_EQ(
      output.sourceMaterial.textures[nuri::kMaterialTextureSlotSpecularColor]
          .uvSet,
      inputRecord.sourceMaterial
          .textures[nuri::kMaterialTextureSlotSpecularColor]
          .uvSet);

  EXPECT_EQ(output.textureCache[0].artifactIdentityHash,
            inputRecord.textureCache[0].artifactIdentityHash);
  EXPECT_EQ(output.textureCache[2].artifactIdentityHash,
            inputRecord.textureCache[2].artifactIdentityHash);
}

TEST(MaterialBinaryCacheTests, DetectsStaleSceneFingerprint) {
  const nuri::SceneMaterialRecord inputRecord = makeSceneMaterialRecord();
  std::array<nuri::SceneMaterialRecord, 1> records = {inputRecord};

  nuri::MaterialBinarySerializeInput input{};
  input.sourcePathHash = 1234u;
  input.sourceSizeBytes = 100u;
  input.sourceMtimeNs = 200u;
  input.materials = std::span<const nuri::SceneMaterialRecord>(records.data(),
                                                               records.size());

  auto serializeResult = nuri::materialBinarySerialize(input);
  ASSERT_FALSE(serializeResult.hasError()) << serializeResult.error();

  nuri::MaterialBinaryDeserializeContext context{};
  context.expectedSourcePathHash = input.sourcePathHash;
  context.validateSourceFingerprint = true;
  context.sourceExists = true;
  context.sourceSizeBytes = input.sourceSizeBytes + 1u;
  context.sourceMtimeNs = input.sourceMtimeNs;

  auto deserializeResult =
      nuri::materialBinaryDeserialize(serializeResult.value(), context);
  ASSERT_TRUE(deserializeResult.hasError());
  EXPECT_TRUE(deserializeResult.error().isStale());
  EXPECT_NE(deserializeResult.error().message.find("stale"), std::string::npos);
}
