#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

enum class MaterialAlphaMode : uint8_t {
  Opaque = 0,
  Mask = 1,
  Blend = 2,
};

struct MaterialTextureTransformData {
  glm::vec2 offset{0.0f};
  glm::vec2 scale{1.0f};
  float rotationRadians = 0.0f;
};

struct MaterialTextureSlotData {
  std::string path{};
  uint32_t uvSet = 0;
  uint32_t samplerIndex = 0;
  float scale = 1.0f;
  bool isEmbedded = false;
  MaterialTextureTransformData transform{};
};

struct MaterialData {
  std::string name{};
  glm::vec4 baseColorFactor{1.0f};
  glm::vec3 emissiveFactor{0.0f};
  float metallicFactor = 1.0f;
  float roughnessFactor = 1.0f;
  glm::vec3 sheenColorFactor{0.0f};
  float sheenWeight = 0.0f;
  float sheenRoughnessFactor = 0.0f;
  float clearcoatFactor = 0.0f;
  float clearcoatRoughnessFactor = 0.0f;
  float clearcoatNormalScale = 1.0f;
  float normalScale = 1.0f;
  float occlusionStrength = 1.0f;
  float alphaCutoff = 0.5f;
  bool doubleSided = false;
  MaterialAlphaMode alphaMode = MaterialAlphaMode::Opaque;
  MaterialTextureSlotData baseColor{};
  MaterialTextureSlotData metallicRoughness{};
  MaterialTextureSlotData normal{};
  MaterialTextureSlotData occlusion{};
  MaterialTextureSlotData emissive{};
  MaterialTextureSlotData clearcoat{};
  MaterialTextureSlotData clearcoatRoughness{};
  MaterialTextureSlotData clearcoatNormal{};
  MaterialTextureSlotData sheenColor{};
  MaterialTextureSlotData sheenRoughness{};
};

struct MaterialDataSet {
  std::vector<MaterialData> materials{};
};

} // namespace nuri
