#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

constexpr uint32_t kInvalidEmbeddedSceneTextureIndex =
    std::numeric_limits<uint32_t>::max();

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

// `sourceKind` controls which source-identifying field is relevant: `path` is
// used for `ExternalFile`, `embeddedIndex` is used for
// `EmbeddedSceneTexture`, and `kInvalidEmbeddedSceneTextureIndex` is the
// sentinel when there is no embedded texture. `uvSet`, `samplerIndex`,
// `scale`, and `transform` apply regardless of source kind.
struct MaterialTextureSlotData {
  // Normalized file path for external textures.
  std::string path{};
  // Selects whether `path` or `embeddedIndex` identifies the source texture.
  MaterialTextureSourceKind sourceKind = MaterialTextureSourceKind::None;
  // Embedded scene texture index, or `kInvalidEmbeddedSceneTextureIndex`.
  uint32_t embeddedIndex = kInvalidEmbeddedSceneTextureIndex;
  uint32_t uvSet = 0;
  uint32_t samplerIndex = 0;
  float scale = 1.0f;
  MaterialTextureTransformData transform{};
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
  // Spec-gloss glossiness factor; not used by metallic-roughness.
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
  float ior = 1.5f; // Valid domain: {0} U [1, +inf); 0 keeps glTF compat mode.
  float normalScale = 1.0f;
  float occlusionStrength = 1.0f;
  float alphaCutoff = 0.5f;
  bool doubleSided = false;
  MaterialAlphaMode alphaMode = MaterialAlphaMode::Opaque;
  // `baseColor` carries `baseColorTexture` for metallic-roughness and
  // `diffuseTexture` for spec-gloss.
  MaterialTextureSlotData baseColor{};
  MaterialTextureSlotData metallicRoughness{};
  MaterialTextureSlotData normal{};
  MaterialTextureSlotData occlusion{};
  MaterialTextureSlotData emissive{};
  MaterialTextureSlotData clearcoat{};
  MaterialTextureSlotData clearcoatRoughness{};
  MaterialTextureSlotData clearcoatNormal{};
  // `specular` is reserved for `KHR_materials_specular::specularTexture` and
  // is not used by the spec-gloss workflow.
  MaterialTextureSlotData specular{};
  // `specularColor` carries `specularColorTexture` for
  // `KHR_materials_specular`, and the combined RGBA
  // `specularGlossinessTexture` for spec-gloss.
  MaterialTextureSlotData specularColor{};
  MaterialTextureSlotData sheenColor{};
  MaterialTextureSlotData sheenRoughness{};
  MaterialTextureSlotData transmission{};
  MaterialTextureSlotData thickness{};
};

struct MaterialDataSet {
  std::vector<MaterialData> materials{};
};

} // namespace nuri
