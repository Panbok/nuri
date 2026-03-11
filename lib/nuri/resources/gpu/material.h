#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/cpu/material_data.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include <glm/glm.hpp>

namespace nuri {

inline constexpr uint32_t kInvalidTextureBindlessIndex =
    std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kInvalidMaterialIndex =
    std::numeric_limits<uint32_t>::max();

enum MaterialFeatureBits : uint32_t {
  kMaterialFeatureNone = 0u,
  kMaterialFeatureMetallicRoughness = 1u << 0u,
  kMaterialFeatureSheen = 1u << 1u,
  kMaterialFeatureClearcoat = 1u << 2u,
};

struct MaterialTextureHandles {
  TextureHandle baseColor{};
  TextureHandle metallicRoughness{};
  TextureHandle normal{};
  TextureHandle occlusion{};
  TextureHandle emissive{};
  TextureHandle clearcoat{};
  TextureHandle clearcoatRoughness{};
  TextureHandle clearcoatNormal{};
};

struct MaterialTextureUvSets {
  uint32_t baseColor = 0;
  uint32_t metallicRoughness = 0;
  uint32_t normal = 0;
  uint32_t occlusion = 0;
  uint32_t emissive = 0;
  uint32_t clearcoat = 0;
  uint32_t clearcoatRoughness = 0;
  uint32_t clearcoatNormal = 0;
};

struct MaterialTextureSamplers {
  uint32_t baseColor = 0;
  uint32_t metallicRoughness = 0;
  uint32_t normal = 0;
  uint32_t occlusion = 0;
  uint32_t emissive = 0;
  uint32_t clearcoat = 0;
  uint32_t clearcoatRoughness = 0;
  uint32_t clearcoatNormal = 0;
};

struct MaterialDesc {
  glm::vec4 baseColorFactor{1.0f};
  glm::vec3 emissiveFactor{0.0f};
  float metallicFactor = 1.0f;
  float roughnessFactor = 1.0f;
  glm::vec3 sheenColorFactor{1.0f, 1.0f, 1.0f};
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
  uint32_t featureMask = kMaterialFeatureMetallicRoughness;
  MaterialTextureHandles textures{};
  MaterialTextureUvSets uvSets{};
  MaterialTextureSamplers samplers{};
};

// std430-friendly packed material payload uploaded to GPU storage buffers.
struct alignas(16) MaterialGpuData {
  glm::vec4 baseColorFactor{1.0f};
  glm::vec4 emissiveFactorNormalScale{0.0f, 0.0f, 0.0f, 1.0f};
  glm::vec4 metallicRoughnessOcclusionAlphaCutoff{1.0f, 1.0f, 1.0f, 0.5f};
  glm::vec4 sheenColorFactorWeight{1.0f, 1.0f, 1.0f, 0.0f};
  glm::vec4 sheenRoughnessClearcoatFactors{
      0.0f, 0.0f, 0.0f,
      1.0f}; // (sheenRoughness, clearcoatFactor, clearcoatRoughness, clearcoatNormalScale)
  glm::uvec4 textureIndices0{
      kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex,
      kInvalidTextureBindlessIndex,
      kInvalidTextureBindlessIndex}; // (baseColor, metallicRoughness, normal, occlusion)
  glm::uvec4 textureIndices1{
      kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex,
      kInvalidTextureBindlessIndex,
      kInvalidTextureBindlessIndex}; // (emissive, clearcoat, clearcoatRoughness, clearcoatNormal)
  glm::uvec4 textureUvSets0{0u, 0u, 0u,
                            0u}; // UV sets for (baseColor, metallicRoughness, normal, occlusion)
  glm::uvec4 textureUvSets1{
      0u, 0u, 0u,
      0u}; // UV sets for (emissive, clearcoat, clearcoatRoughness, clearcoatNormal)
  glm::uvec4 textureSamplerIndices0{0u, 0u, 0u, 0u};
  glm::uvec4 textureSamplerIndices1{0u, 0u, 0u, 0u};
  glm::uvec4 materialFlags{
      static_cast<uint32_t>(MaterialAlphaMode::Opaque), 0u,
      kMaterialFeatureMetallicRoughness,
      0u}; // Kept as a full std430 slot: (alphaMode, doubleSided, featureMask, reserved)
};
static_assert(sizeof(MaterialGpuData) % 16u == 0u,
              "MaterialGpuData must be 16-byte aligned for std430");

class NURI_API Material final {
public:
  ~Material() = default;

  Material(const Material &) = delete;
  Material &operator=(const Material &) = delete;
  Material(Material &&) = delete;
  Material &operator=(Material &&) = delete;

  [[nodiscard]] static Result<std::unique_ptr<Material>, std::string>
  create(GPUDevice &gpu, const MaterialDesc &desc,
         std::string_view debugName = {});

  [[nodiscard]] static Result<std::unique_ptr<Material>, std::string>
  createFromImported(GPUDevice &gpu, const MaterialData &materialData,
                     const MaterialTextureHandles &textures,
                     std::string_view debugName = {});

  [[nodiscard]] const MaterialDesc &desc() const noexcept { return desc_; }
  [[nodiscard]] const MaterialGpuData &gpuData() const noexcept {
    return gpuData_;
  }
  [[nodiscard]] std::string_view debugName() const noexcept {
    return debugName_;
  }

private:
  Material(MaterialDesc desc, MaterialGpuData gpuData, std::string debugName)
      : desc_(desc), gpuData_(gpuData), debugName_(std::move(debugName)) {}

  MaterialDesc desc_{};
  MaterialGpuData gpuData_{};
  std::string debugName_{};
};

} // namespace nuri
