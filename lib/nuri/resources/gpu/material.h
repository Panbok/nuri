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

struct MaterialTextureHandles {
  TextureHandle baseColor{};
  TextureHandle metallicRoughness{};
  TextureHandle normal{};
  TextureHandle occlusion{};
  TextureHandle emissive{};
};

struct MaterialTextureUvSets {
  uint32_t baseColor = 0;
  uint32_t metallicRoughness = 0;
  uint32_t normal = 0;
  uint32_t occlusion = 0;
  uint32_t emissive = 0;
};

struct MaterialTextureSamplers {
  uint32_t baseColor = 0;
  uint32_t metallicRoughness = 0;
  uint32_t normal = 0;
  uint32_t occlusion = 0;
  uint32_t emissive = 0;
};

struct MaterialDesc {
  glm::vec4 baseColorFactor{1.0f};
  glm::vec3 emissiveFactor{0.0f};
  float metallicFactor = 1.0f;
  float roughnessFactor = 1.0f;
  glm::vec3 sheenColorFactor{1.0f, 1.0f, 1.0f};
  float sheenWeight = 0.0f;
  float sheenRoughnessFactor = 0.0f;
  float normalScale = 1.0f;
  float occlusionStrength = 1.0f;
  float alphaCutoff = 0.5f;
  bool doubleSided = false;
  MaterialAlphaMode alphaMode = MaterialAlphaMode::Opaque;
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
  glm::vec4 sheenRoughnessReserved{0.0f, 0.0f, 0.0f, 0.0f};
  glm::uvec4 textureIndices0{
      kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex,
      kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex};
  glm::uvec4 textureIndices1{kInvalidTextureBindlessIndex,
                             static_cast<uint32_t>(MaterialAlphaMode::Opaque),
                             0u, 0u};
  glm::uvec4 textureUvSets0{0u, 0u, 0u, 0u};
  glm::uvec4 textureUvSets1{0u, 0u, 0u, 0u};
  glm::uvec4 textureSamplerIndices0{0u, 0u, 0u, 0u};
  glm::uvec4 textureSamplerIndices1{0u, 0u, 0u, 0u};
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
