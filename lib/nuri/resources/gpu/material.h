#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/cpu/material_data.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
namespace nuri {

inline constexpr uint32_t kInvalidMaterialIndex =
    std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kInvalidMaterialExtensionIndex =
    std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kMaterialFlagsAlphaModeMask = 0x3u;
inline constexpr uint32_t kMaterialFlagsDoubleSidedBit = 1u << 2u;
inline constexpr uint32_t kMaterialFlagsBaseNormalVarianceBit = 1u << 3u;
inline constexpr uint32_t kMaterialFlagsClearcoatNormalVarianceBit = 1u << 4u;
inline constexpr uint32_t kMaterialFlagsNormalVarianceMask =
    kMaterialFlagsBaseNormalVarianceBit |
    kMaterialFlagsClearcoatNormalVarianceBit;
inline constexpr uint32_t kMaterialFlagsWorkflowShift = 8u;
inline constexpr uint32_t kMaterialFlagsWorkflowMask = 0xFFu;
inline constexpr uint32_t kMaterialFlagsFeatureShift = 16u;

enum MaterialFeatureBits : uint32_t {
  kMaterialFeatureNone = 0u,
  kMaterialFeatureMetallicRoughness = 1u << 0u,
  kMaterialFeatureSheen = 1u << 1u,
  kMaterialFeatureClearcoat = 1u << 2u,
  kMaterialFeatureTransmission = 1u << 3u,
  kMaterialFeatureVolume = 1u << 4u,
  kMaterialFeatureSpecular = 1u << 5u,
};

using MaterialTextureHandles = MaterialTextureSlots<TextureHandle>;
using MaterialTextureUvSets = MaterialTextureSlots<uint32_t>;
using MaterialTextureTransforms =
    MaterialTextureSlots<MaterialTextureTransformData>;

struct MaterialDesc {
  MaterialWorkflow workflow = MaterialWorkflow::MetallicRoughness;
  glm::vec4 baseColorFactor{1.0f};
  glm::vec3 emissiveFactor{0.0f};
  float emissiveStrength = 1.0f;
  float metallicFactor = 1.0f;
  float roughnessFactor = 1.0f;
  glm::vec3 specularColorFactor{1.0f, 1.0f, 1.0f};
  float specularFactor = 1.0f;
  float glossinessFactor = 1.0f;
  glm::vec3 sheenColorFactor{0.0f, 0.0f, 0.0f};
  float sheenWeight = 0.0f;
  float sheenRoughnessFactor = 0.0f;
  float clearcoatFactor = 0.0f;
  float clearcoatRoughnessFactor = 0.0f;
  float clearcoatNormalScale = 1.0f;
  float transmissionFactor = 0.0f;
  float thicknessFactor = 0.0f;
  glm::vec3 attenuationColor{1.0f, 1.0f, 1.0f};
  float attenuationDistance = 0.0f;
  float ior = 1.5f;
  float normalScale = 1.0f;
  float occlusionStrength = 1.0f;
  float alphaCutoff = 0.5f;
  bool doubleSided = false;
  MaterialAlphaMode alphaMode = MaterialAlphaMode::Opaque;
  uint32_t featureMask = kMaterialFeatureMetallicRoughness;
  MaterialTextureHandles textures{};
  MaterialTextureUvSets uvSets{};
  MaterialTextureTransforms transforms{};
};

struct PackedMaterialTransformGpuData {
  uint32_t offsetXY = 0u;
  uint32_t scaleXY = 0u;
  uint32_t rotCS = 0u;
};
static_assert(sizeof(PackedMaterialTransformGpuData) == 12u);
static_assert(std::is_trivially_copyable_v<PackedMaterialTransformGpuData>);

struct alignas(16) MaterialHeaderGpuData {
  glm::vec4 baseColorFactor{1.0f};
  glm::vec4 emissiveFactorStrength{0.0f, 0.0f, 0.0f, 1.0f};
  glm::vec4 metallicRoughnessOcclusionAlphaCutoff{1.0f, 1.0f, 1.0f, 0.5f};
  glm::vec4 normalScaleIorReserved{1.0f, 1.5f, 0.0f, 0.0f};
  glm::uvec4 commonTextureIndices{
      kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex,
      kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex};
  uint32_t emissiveTextureIndex = kInvalidTextureBindlessIndex;
  uint32_t uvSetBits = 0u;
  uint32_t materialFlags = 0u;
  uint32_t clearcoatExtensionIndex = kInvalidMaterialExtensionIndex;
  uint32_t sheenExtensionIndex = kInvalidMaterialExtensionIndex;
  uint32_t transmissionExtensionIndex = kInvalidMaterialExtensionIndex;
  uint32_t specularExtensionIndex = kInvalidMaterialExtensionIndex;
  uint32_t reserved0[3] = {0u, 0u, 0u};
  std::array<PackedMaterialTransformGpuData, 5u> commonTransforms{};
};
static_assert(sizeof(MaterialHeaderGpuData) % 16u == 0u);
static_assert(std::is_trivially_copyable_v<MaterialHeaderGpuData>);

struct alignas(16) MaterialClearcoatGpuData {
  glm::vec4 clearcoatFactors{0.0f, 0.0f, 1.0f, 0.0f};
  glm::uvec4 textureIndices{kInvalidTextureBindlessIndex,
                            kInvalidTextureBindlessIndex,
                            kInvalidTextureBindlessIndex, 0u};
  std::array<PackedMaterialTransformGpuData, 3u> transforms{};
  uint32_t uvSetBits = 0u;
  uint32_t reserved0[2] = {0u, 0u};
};
static_assert(sizeof(MaterialClearcoatGpuData) % 16u == 0u);
static_assert(std::is_trivially_copyable_v<MaterialClearcoatGpuData>);

struct alignas(16) MaterialSheenGpuData {
  glm::vec4 sheenColorFactorWeight{0.0f, 0.0f, 0.0f, 0.0f};
  glm::vec4 sheenRoughnessReserved{0.0f, 0.0f, 0.0f, 0.0f};
  glm::uvec4 textureIndices{kInvalidTextureBindlessIndex,
                            kInvalidTextureBindlessIndex, 0u, 0u};
  std::array<PackedMaterialTransformGpuData, 2u> transforms{};
  uint32_t uvSetBits = 0u;
  uint32_t reserved0 = 0u;
};
static_assert(sizeof(MaterialSheenGpuData) % 16u == 0u);
static_assert(std::is_trivially_copyable_v<MaterialSheenGpuData>);

struct alignas(16) MaterialTransmissionGpuData {
  glm::vec4 transmissionThicknessDistance{0.0f, 0.0f, 0.0f, 0.0f};
  glm::vec4 attenuationColorReserved{1.0f, 1.0f, 1.0f, 0.0f};
  glm::uvec4 textureIndices{kInvalidTextureBindlessIndex,
                            kInvalidTextureBindlessIndex, 0u, 0u};
  std::array<PackedMaterialTransformGpuData, 2u> transforms{};
  uint32_t uvSetBits = 0u;
  uint32_t reserved0 = 0u;
};
static_assert(sizeof(MaterialTransmissionGpuData) % 16u == 0u);
static_assert(std::is_trivially_copyable_v<MaterialTransmissionGpuData>);

struct alignas(16) MaterialSpecularGpuData {
  glm::vec4 specularColorFactorSpecular{1.0f, 1.0f, 1.0f, 1.0f};
  glm::uvec4 textureIndices{kInvalidTextureBindlessIndex,
                            kInvalidTextureBindlessIndex, 0u, 0u};
  std::array<PackedMaterialTransformGpuData, 2u> transforms{};
  uint32_t uvSetBits = 0u;
  uint32_t reserved0 = 0u;
};
static_assert(sizeof(MaterialSpecularGpuData) % 16u == 0u);
static_assert(std::is_trivially_copyable_v<MaterialSpecularGpuData>);

struct MaterialPackedGpuData {
  MaterialHeaderGpuData header{};
  MaterialClearcoatGpuData clearcoat{};
  MaterialSheenGpuData sheen{};
  MaterialTransmissionGpuData transmission{};
  MaterialSpecularGpuData specular{};
  bool hasClearcoat = false;
  bool hasSheen = false;
  bool hasTransmissionOrVolume = false;
  bool hasSpecular = false;
};
static_assert(std::is_trivially_copyable_v<MaterialPackedGpuData>,
              "MaterialPackedGpuData must remain trivially copyable");

struct MaterialPackedTablesEntry {
  MaterialHeaderGpuData header{};
  const MaterialClearcoatGpuData *clearcoat = nullptr;
  const MaterialSheenGpuData *sheen = nullptr;
  const MaterialTransmissionGpuData *transmission = nullptr;
  const MaterialSpecularGpuData *specular = nullptr;
};

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
  [[nodiscard]] static MaterialDesc
  descFromImported(const MaterialData &materialData,
                   const MaterialTextureHandles &textures = {});
  static void finalizeDesc(MaterialDesc &desc);
  [[nodiscard]] static Result<std::unique_ptr<Material>, std::string>
  createFromImported(GPUDevice &gpu, const MaterialData &materialData,
                     const MaterialTextureHandles &textures,
                     std::string_view debugName = {});
  [[nodiscard]] const MaterialDesc &desc() const noexcept { return desc_; }
  [[nodiscard]] const MaterialPackedGpuData &packedGpuData() const noexcept {
    return packedGpuData_;
  }
  [[nodiscard]] std::string_view debugName() const noexcept {
    return debugName_;
  }

private:
  Material(MaterialDesc desc, MaterialPackedGpuData packedGpuData,
           std::string debugName)
      : desc_(desc), packedGpuData_(std::move(packedGpuData)),
        debugName_(std::move(debugName)) {}
  MaterialDesc desc_{};
  MaterialPackedGpuData packedGpuData_{};
  std::string debugName_{};
};

} // namespace nuri
