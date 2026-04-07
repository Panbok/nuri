#include "nuri/pch.h"

#include "nuri/resources/gpu/material.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"

namespace nuri {
namespace {

constexpr uint32_t kMaterialFlagsAlphaModeMask = 0x3u;
constexpr uint32_t kMaterialFlagsDoubleSidedBit = 1u << 2u;
constexpr uint32_t kMaterialFlagsWorkflowShift = 8u;
constexpr uint32_t kMaterialFlagsFeatureShift = 16u;

struct ImportedTextureMapping {
  uint32_t slotIndex = 0u;
  MaterialTextureSlotData MaterialData::*slot = nullptr;
  uint32_t MaterialTextureUvSets::*uvSet = nullptr;
};

constexpr std::array<ImportedTextureMapping, kMaterialTextureSlotCount>
    kImportedTextureMappings{{
        {kMaterialTextureSlotBaseColor, &MaterialData::baseColor,
         &MaterialTextureUvSets::baseColor},
        {kMaterialTextureSlotMetallicRoughness,
         &MaterialData::metallicRoughness,
         &MaterialTextureUvSets::metallicRoughness},
        {kMaterialTextureSlotNormal, &MaterialData::normal,
         &MaterialTextureUvSets::normal},
        {kMaterialTextureSlotOcclusion, &MaterialData::occlusion,
         &MaterialTextureUvSets::occlusion},
        {kMaterialTextureSlotEmissive, &MaterialData::emissive,
         &MaterialTextureUvSets::emissive},
        {kMaterialTextureSlotClearcoat, &MaterialData::clearcoat,
         &MaterialTextureUvSets::clearcoat},
        {kMaterialTextureSlotClearcoatRoughness,
         &MaterialData::clearcoatRoughness,
         &MaterialTextureUvSets::clearcoatRoughness},
        {kMaterialTextureSlotClearcoatNormal, &MaterialData::clearcoatNormal,
         &MaterialTextureUvSets::clearcoatNormal},
        {kMaterialTextureSlotSpecular, &MaterialData::specular,
         &MaterialTextureUvSets::specular},
        {kMaterialTextureSlotSpecularColor, &MaterialData::specularColor,
         &MaterialTextureUvSets::specularColor},
        {kMaterialTextureSlotSheenColor, &MaterialData::sheenColor,
         &MaterialTextureUvSets::sheenColor},
        {kMaterialTextureSlotSheenRoughness, &MaterialData::sheenRoughness,
         &MaterialTextureUvSets::sheenRoughness},
        {kMaterialTextureSlotTransmission, &MaterialData::transmission,
         &MaterialTextureUvSets::transmission},
        {kMaterialTextureSlotThickness, &MaterialData::thickness,
         &MaterialTextureUvSets::thickness},
    }};

[[nodiscard]] bool hasSheenData(const MaterialDesc &desc) {
  const float sheenMax =
      std::max(desc.sheenColorFactor.x,
               std::max(desc.sheenColorFactor.y, desc.sheenColorFactor.z));
  return sheenMax > 0.0f || desc.sheenRoughnessFactor > 0.0f ||
         nuri::isValid(desc.textures.sheenColor) ||
         nuri::isValid(desc.textures.sheenRoughness);
}

[[nodiscard]] bool hasSpecularData(const MaterialDesc &desc) {
  if (desc.workflow == MaterialWorkflow::SpecularGlossiness) {
    return false;
  }
  return desc.specularFactor != 1.0f || desc.specularColorFactor.x != 1.0f ||
         desc.specularColorFactor.y != 1.0f ||
         desc.specularColorFactor.z != 1.0f ||
         nuri::isValid(desc.textures.specular) ||
         nuri::isValid(desc.textures.specularColor);
}

[[nodiscard]] bool hasClearcoatData(const MaterialDesc &desc) {
  return desc.clearcoatFactor > 0.0f ||
         nuri::isValid(desc.textures.clearcoat) ||
         nuri::isValid(desc.textures.clearcoatRoughness) ||
         nuri::isValid(desc.textures.clearcoatNormal);
}

[[nodiscard]] bool hasTransmissionData(const MaterialDesc &desc) {
  return desc.transmissionFactor > 0.0f ||
         nuri::isValid(desc.textures.transmission);
}

[[nodiscard]] bool hasVolumeData(const MaterialDesc &desc) {
  const bool hasAttenuationColor = desc.attenuationColor.x != 1.0f ||
                                   desc.attenuationColor.y != 1.0f ||
                                   desc.attenuationColor.z != 1.0f;
  return desc.thicknessFactor > 0.0f || desc.attenuationDistance > 0.0f ||
         hasAttenuationColor || nuri::isValid(desc.textures.thickness);
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
  for (const ImportedTextureMapping &mapping : kImportedTextureMappings) {
    const MaterialTextureSlotData &slot = materialData.*(mapping.slot);
    desc.uvSets.*(mapping.uvSet) = slot.uvSet;
    desc.transforms.slots[mapping.slotIndex] = slot.transform;
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
  flags |=
      (static_cast<uint32_t>(desc.workflow) << kMaterialFlagsWorkflowShift);
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

  auto baseColorIdx =
      resolveBindlessIndex(gpu, desc.textures.baseColor, "baseColor");
  if (baseColorIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        baseColorIdx.error());
  }
  auto metallicRoughnessIdx = resolveBindlessIndex(
      gpu, desc.textures.metallicRoughness, "metallicRoughness");
  if (metallicRoughnessIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        metallicRoughnessIdx.error());
  }
  auto normalIdx = resolveBindlessIndex(gpu, desc.textures.normal, "normal");
  if (normalIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        normalIdx.error());
  }
  auto occlusionIdx =
      resolveBindlessIndex(gpu, desc.textures.occlusion, "occlusion");
  if (occlusionIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        occlusionIdx.error());
  }
  auto emissiveIdx =
      resolveBindlessIndex(gpu, desc.textures.emissive, "emissive");
  if (emissiveIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        emissiveIdx.error());
  }
  auto clearcoatIdx =
      resolveBindlessIndex(gpu, desc.textures.clearcoat, "clearcoat");
  if (clearcoatIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        clearcoatIdx.error());
  }
  auto clearcoatRoughnessIdx = resolveBindlessIndex(
      gpu, desc.textures.clearcoatRoughness, "clearcoatRoughness");
  if (clearcoatRoughnessIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        clearcoatRoughnessIdx.error());
  }
  auto clearcoatNormalIdx = resolveBindlessIndex(
      gpu, desc.textures.clearcoatNormal, "clearcoatNormal");
  if (clearcoatNormalIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        clearcoatNormalIdx.error());
  }
  auto specularIdx =
      resolveBindlessIndex(gpu, desc.textures.specular, "specular");
  if (specularIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        specularIdx.error());
  }
  auto specularColorIdx =
      resolveBindlessIndex(gpu, desc.textures.specularColor, "specularColor");
  if (specularColorIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        specularColorIdx.error());
  }
  auto sheenColorIdx =
      resolveBindlessIndex(gpu, desc.textures.sheenColor, "sheenColor");
  if (sheenColorIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        sheenColorIdx.error());
  }
  auto sheenRoughnessIdx =
      resolveBindlessIndex(gpu, desc.textures.sheenRoughness, "sheenRoughness");
  if (sheenRoughnessIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        sheenRoughnessIdx.error());
  }
  auto transmissionIdx =
      resolveBindlessIndex(gpu, desc.textures.transmission, "transmission");
  if (transmissionIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        transmissionIdx.error());
  }
  auto thicknessIdx =
      resolveBindlessIndex(gpu, desc.textures.thickness, "thickness");
  if (thicknessIdx.hasError()) {
    return Result<MaterialPackedGpuData, std::string>::makeError(
        thicknessIdx.error());
  }

  gpuData.header.commonTextureIndices =
      glm::uvec4(baseColorIdx.value(), metallicRoughnessIdx.value(),
                 normalIdx.value(), occlusionIdx.value());
  gpuData.header.emissiveTextureIndex = emissiveIdx.value();
  gpuData.header.commonTransforms[0] = packMaterialTransform(
      desc.transforms.slots[kMaterialTextureSlotBaseColor]);
  gpuData.header.commonTransforms[1] = packMaterialTransform(
      desc.transforms.slots[kMaterialTextureSlotMetallicRoughness]);
  gpuData.header.commonTransforms[2] =
      packMaterialTransform(desc.transforms.slots[kMaterialTextureSlotNormal]);
  gpuData.header.commonTransforms[3] = packMaterialTransform(
      desc.transforms.slots[kMaterialTextureSlotOcclusion]);
  gpuData.header.commonTransforms[4] = packMaterialTransform(
      desc.transforms.slots[kMaterialTextureSlotEmissive]);
  setPackedUvBit(gpuData.header.uvSetBits, 0u, desc.uvSets.baseColor);
  setPackedUvBit(gpuData.header.uvSetBits, 1u, desc.uvSets.metallicRoughness);
  setPackedUvBit(gpuData.header.uvSetBits, 2u, desc.uvSets.normal);
  setPackedUvBit(gpuData.header.uvSetBits, 3u, desc.uvSets.occlusion);
  setPackedUvBit(gpuData.header.uvSetBits, 4u, desc.uvSets.emissive);

  if ((desc.featureMask & kMaterialFeatureClearcoat) != 0u) {
    gpuData.hasClearcoat = true;
    gpuData.clearcoat.clearcoatFactors = glm::vec4(
        clearcoat, clearcoatRoughness, desc.clearcoatNormalScale, 0.0f);
    gpuData.clearcoat.textureIndices =
        glm::uvec4(clearcoatIdx.value(), clearcoatRoughnessIdx.value(),
                   clearcoatNormalIdx.value(), 0u);
    gpuData.clearcoat.transforms[0] = packMaterialTransform(
        desc.transforms.slots[kMaterialTextureSlotClearcoat]);
    gpuData.clearcoat.transforms[1] = packMaterialTransform(
        desc.transforms.slots[kMaterialTextureSlotClearcoatRoughness]);
    gpuData.clearcoat.transforms[2] = packMaterialTransform(
        desc.transforms.slots[kMaterialTextureSlotClearcoatNormal]);
    setPackedUvBit(gpuData.clearcoat.uvSetBits, 0u, desc.uvSets.clearcoat);
    setPackedUvBit(gpuData.clearcoat.uvSetBits, 1u,
                   desc.uvSets.clearcoatRoughness);
    setPackedUvBit(gpuData.clearcoat.uvSetBits, 2u,
                   desc.uvSets.clearcoatNormal);
  }

  if ((desc.featureMask & kMaterialFeatureSheen) != 0u) {
    gpuData.hasSheen = true;
    gpuData.sheen.sheenColorFactorWeight =
        glm::vec4(desc.sheenColorFactor, sheenWeight);
    gpuData.sheen.sheenRoughnessReserved =
        glm::vec4(sheenRoughness, 0.0f, 0.0f, 0.0f);
    gpuData.sheen.textureIndices =
        glm::uvec4(sheenColorIdx.value(), sheenRoughnessIdx.value(), 0u, 0u);
    gpuData.sheen.transforms[0] = packMaterialTransform(
        desc.transforms.slots[kMaterialTextureSlotSheenColor]);
    gpuData.sheen.transforms[1] = packMaterialTransform(
        desc.transforms.slots[kMaterialTextureSlotSheenRoughness]);
    setPackedUvBit(gpuData.sheen.uvSetBits, 0u, desc.uvSets.sheenColor);
    setPackedUvBit(gpuData.sheen.uvSetBits, 1u, desc.uvSets.sheenRoughness);
  }

  if ((desc.featureMask &
       (kMaterialFeatureTransmission | kMaterialFeatureVolume)) != 0u) {
    gpuData.hasTransmission = true;
    gpuData.transmission.transmissionThicknessDistance =
        glm::vec4(transmission, thickness, attenuationDistance, 0.0f);
    gpuData.transmission.attenuationColorReserved =
        glm::vec4(attenuationColor, 0.0f);
    gpuData.transmission.textureIndices =
        glm::uvec4(transmissionIdx.value(), thicknessIdx.value(), 0u, 0u);
    gpuData.transmission.transforms[0] = packMaterialTransform(
        desc.transforms.slots[kMaterialTextureSlotTransmission]);
    gpuData.transmission.transforms[1] = packMaterialTransform(
        desc.transforms.slots[kMaterialTextureSlotThickness]);
    setPackedUvBit(gpuData.transmission.uvSetBits, 0u,
                   desc.uvSets.transmission);
    setPackedUvBit(gpuData.transmission.uvSetBits, 1u, desc.uvSets.thickness);
  }

  if ((desc.featureMask & kMaterialFeatureSpecular) != 0u ||
      desc.workflow == MaterialWorkflow::SpecularGlossiness) {
    gpuData.hasSpecular = true;
    gpuData.specular.specularColorFactorSpecular =
        glm::vec4(specularColor, specular);
    gpuData.specular.textureIndices =
        glm::uvec4(specularIdx.value(), specularColorIdx.value(), 0u, 0u);
    gpuData.specular.transforms[0] = packMaterialTransform(
        desc.transforms.slots[kMaterialTextureSlotSpecular]);
    gpuData.specular.transforms[1] = packMaterialTransform(
        desc.transforms.slots[kMaterialTextureSlotSpecularColor]);
    setPackedUvBit(gpuData.specular.uvSetBits, 0u, desc.uvSets.specular);
    setPackedUvBit(gpuData.specular.uvSetBits, 1u, desc.uvSets.specularColor);
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
