#include "nuri/resources/gpu/material.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/pch.h"
namespace nuri {
namespace {
constexpr uint32_t kMaterialFlagsAlphaModeMask = 0x3u;
constexpr uint32_t kMaterialFlagsDoubleSidedBit = 1u << 2u;
constexpr uint32_t kMaterialFlagsWorkflowShift = 8u;
constexpr uint32_t kMaterialFlagsWorkflowMask = 0xFFu;
constexpr uint32_t kMaterialFlagsFeatureShift = 16u;
constexpr std::array<std::string_view, kMaterialTextureSlotCount> kSlotNames{
    "baseColor",          "metallicRoughness", "normal",
    "occlusion",          "emissive",          "clearcoat",
    "clearcoatRoughness", "clearcoatNormal",   "specular",
    "specularColor",      "sheenColor",        "sheenRoughness",
    "transmission",       "thickness"};
constexpr std::array kCommonSlots{
    kMaterialTextureSlotBaseColor, kMaterialTextureSlotMetallicRoughness,
    kMaterialTextureSlotNormal, kMaterialTextureSlotOcclusion,
    kMaterialTextureSlotEmissive};
constexpr std::array kClearcoatSlots{kMaterialTextureSlotClearcoat,
                                     kMaterialTextureSlotClearcoatRoughness,
                                     kMaterialTextureSlotClearcoatNormal};
constexpr std::array kSheenSlots{kMaterialTextureSlotSheenColor,
                                 kMaterialTextureSlotSheenRoughness};
constexpr std::array kTransmissionSlots{kMaterialTextureSlotTransmission,
                                        kMaterialTextureSlotThickness};
constexpr std::array kSpecularSlots{kMaterialTextureSlotSpecular,
                                    kMaterialTextureSlotSpecularColor};
[[nodiscard]] bool hasSheenData(const MaterialDesc &desc) {
  const float sheenMax =
      std::max(desc.sheenColorFactor.x,
               std::max(desc.sheenColorFactor.y, desc.sheenColorFactor.z));
  return sheenMax > 0.0f || desc.sheenRoughnessFactor > 0.0f ||
         nuri::isValid(desc.textures[kMaterialTextureSlotSheenColor]) ||
         nuri::isValid(desc.textures[kMaterialTextureSlotSheenRoughness]);
}
[[nodiscard]] bool hasSpecularData(const MaterialDesc &desc) {
  if (desc.workflow == MaterialWorkflow::SpecularGlossiness) {
    return false;
  }
  return desc.specularFactor != 1.0f || desc.specularColorFactor.x != 1.0f ||
         desc.specularColorFactor.y != 1.0f ||
         desc.specularColorFactor.z != 1.0f ||
         nuri::isValid(desc.textures[kMaterialTextureSlotSpecular]) ||
         nuri::isValid(desc.textures[kMaterialTextureSlotSpecularColor]);
}
[[nodiscard]] bool hasClearcoatData(const MaterialDesc &desc) {
  return desc.clearcoatFactor > 0.0f ||
         nuri::isValid(desc.textures[kMaterialTextureSlotClearcoat]) ||
         nuri::isValid(desc.textures[kMaterialTextureSlotClearcoatRoughness]) ||
         nuri::isValid(desc.textures[kMaterialTextureSlotClearcoatNormal]);
}
[[nodiscard]] bool hasTransmissionData(const MaterialDesc &desc) {
  return desc.transmissionFactor > 0.0f ||
         nuri::isValid(desc.textures[kMaterialTextureSlotTransmission]);
}
[[nodiscard]] bool hasVolumeData(const MaterialDesc &desc) {
  const bool hasAttenuationColor = desc.attenuationColor.x != 1.0f ||
                                   desc.attenuationColor.y != 1.0f ||
                                   desc.attenuationColor.z != 1.0f;
  return desc.thicknessFactor > 0.0f || desc.attenuationDistance > 0.0f ||
         hasAttenuationColor ||
         nuri::isValid(desc.textures[kMaterialTextureSlotThickness]);
}
[[nodiscard]] float sanitizeMaterialDescIor(float ior) {
  if (ior == 0.0f) {
    return 0.0f;
  }
  if (std::isfinite(ior) && ior >= 1.0f) {
    return ior;
  }
  NURI_LOG_WARNING("Material::finalizeDesc: invalid IOR %.6f; clamping to 1.0",
                   static_cast<double>(ior));
  return 1.0f;
}
void copyImportedTextureMetadata(MaterialDesc &desc,
                                 const MaterialData &materialData) {
  for (size_t index = 0; index < materialData.textures.size(); ++index) {
    desc.uvSets[index] = materialData.textures[index].uvSet;
    desc.transforms[index] = materialData.textures[index].transform;
  }
}
Result<uint32_t, std::string> resolveBindlessIndex(GPUDevice &gpu,
                                                   TextureHandle handle,
                                                   std::string_view slotName) {
  if (!nuri::isValid(handle)) {
    return Result<uint32_t, std::string>::makeResult(
        kInvalidTextureBindlessIndex);
  }
  if (!gpu.isValid(handle)) {
    return Result<uint32_t, std::string>::makeError(
        "Material::create: invalid texture handle for slot '" +
        std::string(slotName) + "'");
  }
  return Result<uint32_t, std::string>::makeResult(
      gpu.getTextureBindlessIndex(handle));
}
uint32_t clampUvSet(uint32_t uvSet) { return uvSet == 0u ? 0u : 1u; }
void setPackedUvBit(uint32_t &bits, uint32_t bitIndex, uint32_t uvSet) {
  const uint32_t clampedUvSet = clampUvSet(uvSet);
  bits &= ~(1u << bitIndex);
  bits |= (clampedUvSet << bitIndex);
}
uint32_t packMaterialFlags(const MaterialDesc &desc) {
  uint32_t flags =
      static_cast<uint32_t>(desc.alphaMode) & kMaterialFlagsAlphaModeMask;
  if (desc.doubleSided) {
    flags |= kMaterialFlagsDoubleSidedBit;
  }
  const uint32_t workflowBits =
      static_cast<uint32_t>(desc.workflow) & kMaterialFlagsWorkflowMask;
  flags |= workflowBits << kMaterialFlagsWorkflowShift;
  flags |= (desc.featureMask << kMaterialFlagsFeatureShift);
  return flags;
}
PackedMaterialTransformGpuData
packMaterialTransform(const MaterialTextureTransformData &transform) {
  return PackedMaterialTransformGpuData{
      .offsetXY = glm::packHalf2x16(transform.offset),
      .scaleXY = glm::packHalf2x16(transform.scale),
      .rotCS =
          glm::packSnorm2x16(glm::vec2(std::cos(transform.rotationRadians),
                                       std::sin(transform.rotationRadians))),
  };
}
template <size_t N>
void packTextureMetadata(std::array<PackedMaterialTransformGpuData, N> &out,
                         uint32_t &uvBits, const MaterialDesc &desc,
                         const std::array<MaterialTextureSlot, N> &slots) {
  for (uint32_t i = 0; i < N; ++i) {
    out[i] = packMaterialTransform(desc.transforms[slots[i]]);
    setPackedUvBit(uvBits, i, desc.uvSets[slots[i]]);
  }
}
Result<MaterialPackedGpuData, std::string>
buildPackedGpuData(GPUDevice &gpu, const MaterialDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const float metallic = std::clamp(desc.metallicFactor, 0.0f, 1.0f);
  const float roughness = std::clamp(desc.roughnessFactor, 0.0f, 1.0f);
  const float occlusion = std::clamp(desc.occlusionStrength, 0.0f, 1.0f);
  const float alphaCutoff = std::clamp(desc.alphaCutoff, 0.0f, 1.0f);
  const float specular = desc.workflow == MaterialWorkflow::SpecularGlossiness
                             ? std::clamp(desc.glossinessFactor, 0.0f, 1.0f)
                             : std::clamp(desc.specularFactor, 0.0f, 1.0f);
  const glm::vec3 specularColor = glm::max(desc.specularColorFactor, 0.0f);
  const float sheenWeight = std::clamp(desc.sheenWeight, 0.0f, 1.0f);
  const float sheenRoughness =
      std::clamp(desc.sheenRoughnessFactor, 0.0f, 1.0f);
  const float clearcoat = std::clamp(desc.clearcoatFactor, 0.0f, 1.0f);
  const float clearcoatRoughness =
      std::clamp(desc.clearcoatRoughnessFactor, 0.0f, 1.0f);
  const float transmission = std::clamp(desc.transmissionFactor, 0.0f, 1.0f);
  const float thickness = std::max(desc.thicknessFactor, 0.0f);
  const float emissiveStrength = std::max(desc.emissiveStrength, 0.0f);
  const float ior = desc.ior;
  const glm::vec3 attenuationColor =
      glm::clamp(desc.attenuationColor, 0.0f, 1.0f);
  const float attenuationDistance = std::max(desc.attenuationDistance, 0.0f);
  MaterialPackedGpuData gpuData{};
  gpuData.header.baseColorFactor = desc.baseColorFactor;
  gpuData.header.emissiveFactorStrength =
      glm::vec4(desc.emissiveFactor, emissiveStrength);
  gpuData.header.metallicRoughnessOcclusionAlphaCutoff =
      glm::vec4(metallic, roughness, occlusion, alphaCutoff);
  gpuData.header.normalScaleIorReserved =
      glm::vec4(desc.normalScale, ior, 0.0f, 0.0f);
  gpuData.header.materialFlags = packMaterialFlags(desc);
  MaterialTextureSlots<uint32_t> textureIndices{};
  for (size_t i = 0; i < textureIndices.size(); ++i) {
    auto index = resolveBindlessIndex(gpu, desc.textures[i], kSlotNames[i]);
    if (index.hasError())
      return Result<MaterialPackedGpuData, std::string>::makeError(
          index.error());
    textureIndices[i] = index.value();
  }
  gpuData.header.commonTextureIndices =
      glm::uvec4(textureIndices[0], textureIndices[1], textureIndices[2],
                 textureIndices[3]);
  gpuData.header.emissiveTextureIndex = textureIndices[4];
  packTextureMetadata(gpuData.header.commonTransforms, gpuData.header.uvSetBits,
                      desc, kCommonSlots);
  if ((desc.featureMask & kMaterialFeatureClearcoat) != 0u) {
    gpuData.hasClearcoat = true;
    gpuData.clearcoat.clearcoatFactors = glm::vec4(
        clearcoat, clearcoatRoughness, desc.clearcoatNormalScale, 0.0f);
    gpuData.clearcoat.textureIndices = glm::uvec4(
        textureIndices[kClearcoatSlots[0]], textureIndices[kClearcoatSlots[1]],
        textureIndices[kClearcoatSlots[2]], 0u);
    packTextureMetadata(gpuData.clearcoat.transforms,
                        gpuData.clearcoat.uvSetBits, desc, kClearcoatSlots);
  }
  if ((desc.featureMask & kMaterialFeatureSheen) != 0u) {
    gpuData.hasSheen = true;
    gpuData.sheen.sheenColorFactorWeight =
        glm::vec4(desc.sheenColorFactor, sheenWeight);
    gpuData.sheen.sheenRoughnessReserved =
        glm::vec4(sheenRoughness, 0.0f, 0.0f, 0.0f);
    gpuData.sheen.textureIndices = glm::uvec4(
        textureIndices[kSheenSlots[0]], textureIndices[kSheenSlots[1]], 0u, 0u);
    packTextureMetadata(gpuData.sheen.transforms, gpuData.sheen.uvSetBits, desc,
                        kSheenSlots);
  }
  if ((desc.featureMask &
       (kMaterialFeatureTransmission | kMaterialFeatureVolume)) != 0u) {
    gpuData.hasTransmissionOrVolume = true;
    gpuData.transmission.transmissionThicknessDistance =
        glm::vec4(transmission, thickness, attenuationDistance, 0.0f);
    gpuData.transmission.attenuationColorReserved =
        glm::vec4(attenuationColor, 0.0f);
    gpuData.transmission.textureIndices =
        glm::uvec4(textureIndices[kTransmissionSlots[0]],
                   textureIndices[kTransmissionSlots[1]], 0u, 0u);
    packTextureMetadata(gpuData.transmission.transforms,
                        gpuData.transmission.uvSetBits, desc,
                        kTransmissionSlots);
  }
  if ((desc.featureMask & kMaterialFeatureSpecular) != 0u ||
      desc.workflow == MaterialWorkflow::SpecularGlossiness) {
    gpuData.hasSpecular = true;
    gpuData.specular.specularColorFactorSpecular =
        glm::vec4(specularColor, specular);
    gpuData.specular.textureIndices =
        glm::uvec4(textureIndices[kSpecularSlots[0]],
                   textureIndices[kSpecularSlots[1]], 0u, 0u);
    packTextureMetadata(gpuData.specular.transforms, gpuData.specular.uvSetBits,
                        desc, kSpecularSlots);
  }
  return Result<MaterialPackedGpuData, std::string>::makeResult(
      std::move(gpuData));
}
} // namespace

Result<std::unique_ptr<Material>, std::string>
Material::create(GPUDevice &gpu, const MaterialDesc &desc,
                 std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto gpuDataResult = buildPackedGpuData(gpu, desc);
  if (gpuDataResult.hasError()) {
    return Result<std::unique_ptr<Material>, std::string>::makeError(
        gpuDataResult.error());
  }
  return Result<std::unique_ptr<Material>, std::string>::makeResult(
      std::unique_ptr<Material>(
          new Material(desc, gpuDataResult.value(), std::string(debugName))));
}

MaterialDesc
Material::descFromImported(const MaterialData &materialData,
                           const MaterialTextureHandles &textures) {
  MaterialDesc desc{};
  desc.workflow = materialData.workflow;
  desc.baseColorFactor = materialData.baseColorFactor;
  desc.emissiveFactor = materialData.emissiveFactor;
  desc.emissiveStrength = materialData.emissiveStrength;
  desc.metallicFactor = materialData.metallicFactor;
  desc.roughnessFactor = materialData.roughnessFactor;
  desc.specularColorFactor = materialData.specularColorFactor;
  desc.specularFactor = materialData.specularFactor;
  desc.glossinessFactor = materialData.glossinessFactor;
  desc.sheenColorFactor = materialData.sheenColorFactor;
  desc.sheenWeight = materialData.sheenWeight;
  desc.sheenRoughnessFactor = materialData.sheenRoughnessFactor;
  desc.clearcoatFactor = materialData.clearcoatFactor;
  desc.clearcoatRoughnessFactor = materialData.clearcoatRoughnessFactor;
  desc.clearcoatNormalScale = materialData.clearcoatNormalScale;
  desc.transmissionFactor = materialData.transmissionFactor;
  desc.thicknessFactor = materialData.thicknessFactor;
  desc.attenuationColor = materialData.attenuationColor;
  desc.attenuationDistance = materialData.attenuationDistance;
  desc.ior = materialData.ior;
  desc.normalScale = materialData.normalScale;
  desc.occlusionStrength = materialData.occlusionStrength;
  desc.alphaCutoff = materialData.alphaCutoff;
  desc.doubleSided = materialData.doubleSided;
  desc.alphaMode = materialData.alphaMode;
  desc.textures = textures;
  copyImportedTextureMetadata(desc, materialData);
  finalizeDesc(desc);
  return desc;
}

void Material::finalizeDesc(MaterialDesc &desc) {
  desc.ior = sanitizeMaterialDescIor(desc.ior);
  desc.emissiveStrength = std::max(desc.emissiveStrength, 0.0f);
  desc.glossinessFactor = std::clamp(desc.glossinessFactor, 0.0f, 1.0f);
  desc.featureMask = 0u;
  if (desc.workflow == MaterialWorkflow::MetallicRoughness) {
    desc.featureMask |= kMaterialFeatureMetallicRoughness;
  } else {
    desc.metallicFactor = 0.0f;
  }
  if (hasSpecularData(desc)) {
    desc.featureMask |= kMaterialFeatureSpecular;
  }
  if (hasSheenData(desc)) {
    desc.featureMask |= kMaterialFeatureSheen;
    if (desc.sheenWeight <= 0.0f) {
      desc.sheenWeight = 1.0f;
    }
  }
  if (hasClearcoatData(desc)) {
    desc.featureMask |= kMaterialFeatureClearcoat;
  }
  if (hasTransmissionData(desc)) {
    desc.featureMask |= kMaterialFeatureTransmission;
  }
  if (hasVolumeData(desc)) {
    desc.featureMask |= kMaterialFeatureVolume;
  }
}

Result<std::unique_ptr<Material>, std::string>
Material::createFromImported(GPUDevice &gpu, const MaterialData &materialData,
                             const MaterialTextureHandles &textures,
                             std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  MaterialDesc desc = descFromImported(materialData, textures);
  const std::string_view name =
      debugName.empty() ? std::string_view(materialData.name) : debugName;
  return create(gpu, desc, name);
}

} // namespace nuri
