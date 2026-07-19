#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <string>
#include <vector>
namespace nuri {

constexpr uint32_t kInvalidEmbeddedSceneTextureIndex =
    std::numeric_limits<uint32_t>::max();
enum MaterialTextureSlot : uint32_t {
  kMaterialTextureSlotBaseColor,
  kMaterialTextureSlotMetallicRoughness,
  kMaterialTextureSlotNormal,
  kMaterialTextureSlotOcclusion,
  kMaterialTextureSlotEmissive,
  kMaterialTextureSlotClearcoat,
  kMaterialTextureSlotClearcoatRoughness,
  kMaterialTextureSlotClearcoatNormal,
  kMaterialTextureSlotSpecular,
  kMaterialTextureSlotSpecularColor,
  kMaterialTextureSlotSheenColor,
  kMaterialTextureSlotSheenRoughness,
  kMaterialTextureSlotTransmission,
  kMaterialTextureSlotThickness,
  kMaterialTextureSlotCount,
};

template <typename T>
using MaterialTextureSlots = std::array<T, kMaterialTextureSlotCount>;

enum class MaterialAlphaMode : uint8_t {
  Opaque = 0,
  Mask = 1,
  Blend = 2,
};

enum class MaterialWorkflow : uint8_t {
  MetallicRoughness = 0,
  SpecularGlossiness = 1,
};

enum class MaterialTextureSourceKind : uint8_t {
  None = 0,
  ExternalFile = 1,
  EmbeddedSceneTexture = 2,
};

struct MaterialTextureTransformData {
  glm::vec2 offset{0.0f};
  glm::vec2 scale{1.0f};
  float rotationRadians = 0.0f;
};

struct MaterialTextureSlotData {
  std::string path{};
  MaterialTextureSourceKind sourceKind = MaterialTextureSourceKind::None;
  uint32_t embeddedIndex = kInvalidEmbeddedSceneTextureIndex;
  uint32_t uvSet = 0;
  uint32_t samplerIndex = 0;
  float scale = 1.0f;
  MaterialTextureTransformData transform{};
};

struct EmbeddedSceneTextureData {
  uint32_t width = 0u;
  uint32_t height = 0u;
  bool compressed = false;
  std::vector<std::byte> bytes{};
};

struct MaterialData {
  std::string name{};
  MaterialWorkflow workflow = MaterialWorkflow::MetallicRoughness;
  glm::vec4 baseColorFactor{1.0f};
  glm::vec3 emissiveFactor{0.0f};
  float emissiveStrength = 1.0f;
  float metallicFactor = 1.0f;
  float roughnessFactor = 1.0f;
  glm::vec3 specularColorFactor{1.0f};
  float specularFactor = 1.0f;
  float glossinessFactor = 1.0f;
  glm::vec3 sheenColorFactor{0.0f};
  float sheenWeight = 0.0f;
  float sheenRoughnessFactor = 0.0f;
  float clearcoatFactor = 0.0f;
  float clearcoatRoughnessFactor = 0.0f;
  float clearcoatNormalScale = 1.0f;
  float transmissionFactor = 0.0f;
  float thicknessFactor = 0.0f;
  glm::vec3 attenuationColor{1.0f};
  float attenuationDistance = 0.0f;
  float ior = 1.5f;
  float normalScale = 1.0f;
  float occlusionStrength = 1.0f;
  float alphaCutoff = 0.5f;
  bool doubleSided = false;
  MaterialAlphaMode alphaMode = MaterialAlphaMode::Opaque;
  MaterialTextureSlots<MaterialTextureSlotData> textures{};
};

struct MaterialDataSet {
  std::vector<MaterialData> materials{};
};

} // namespace nuri
