#include "nuri/pch.h"

#include "nuri/resources/gpu/material.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"

namespace nuri {
namespace {

struct ImportedTextureMapping {
  uint32_t slotIndex = 0u;
  MaterialTextureSlotData MaterialData::*slot = nullptr;
  uint32_t MaterialTextureUvSets::*uvSet = nullptr;
  uint32_t MaterialTextureSamplers::*sampler = nullptr;
};

constexpr std::array<ImportedTextureMapping, kMaterialTextureSlotCount>
    kImportedTextureMappings{{
        {kMaterialTextureSlotBaseColor, &MaterialData::baseColor,
         &MaterialTextureUvSets::baseColor,
         &MaterialTextureSamplers::baseColor},
        {kMaterialTextureSlotMetallicRoughness,
         &MaterialData::metallicRoughness,
         &MaterialTextureUvSets::metallicRoughness,
         &MaterialTextureSamplers::metallicRoughness},
        {kMaterialTextureSlotNormal, &MaterialData::normal,
         &MaterialTextureUvSets::normal, &MaterialTextureSamplers::normal},
        {kMaterialTextureSlotOcclusion, &MaterialData::occlusion,
         &MaterialTextureUvSets::occlusion,
         &MaterialTextureSamplers::occlusion},
        {kMaterialTextureSlotEmissive, &MaterialData::emissive,
         &MaterialTextureUvSets::emissive, &MaterialTextureSamplers::emissive},
        {kMaterialTextureSlotClearcoat, &MaterialData::clearcoat,
         &MaterialTextureUvSets::clearcoat,
         &MaterialTextureSamplers::clearcoat},
        {kMaterialTextureSlotClearcoatRoughness,
         &MaterialData::clearcoatRoughness,
         &MaterialTextureUvSets::clearcoatRoughness,
         &MaterialTextureSamplers::clearcoatRoughness},
        {kMaterialTextureSlotClearcoatNormal, &MaterialData::clearcoatNormal,
         &MaterialTextureUvSets::clearcoatNormal,
         &MaterialTextureSamplers::clearcoatNormal},
        {kMaterialTextureSlotSpecular, &MaterialData::specular,
         &MaterialTextureUvSets::specular, &MaterialTextureSamplers::specular},
        {kMaterialTextureSlotSpecularColor, &MaterialData::specularColor,
         &MaterialTextureUvSets::specularColor,
         &MaterialTextureSamplers::specularColor},
        {kMaterialTextureSlotSheenColor, &MaterialData::sheenColor,
         &MaterialTextureUvSets::sheenColor,
         &MaterialTextureSamplers::sheenColor},
        {kMaterialTextureSlotSheenRoughness, &MaterialData::sheenRoughness,
         &MaterialTextureUvSets::sheenRoughness,
         &MaterialTextureSamplers::sheenRoughness},
        {kMaterialTextureSlotTransmission, &MaterialData::transmission,
         &MaterialTextureUvSets::transmission,
         &MaterialTextureSamplers::transmission},
        {kMaterialTextureSlotThickness, &MaterialData::thickness,
         &MaterialTextureUvSets::thickness,
         &MaterialTextureSamplers::thickness},
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
    desc.samplers.*(mapping.sampler) = slot.samplerIndex;
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

Result<MaterialGpuData, std::string> buildGpuData(GPUDevice &gpu,
                                                  const MaterialDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const auto clampUvSet = [](uint32_t uvSet) -> uint32_t {
    return (uvSet == 0u) ? 0u : 1u;
  };
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
  const uint32_t featureMask = desc.featureMask;

  MaterialGpuData gpuData{};
  gpuData.baseColorFactor = desc.baseColorFactor;
  gpuData.emissiveFactorStrength =
      glm::vec4(desc.emissiveFactor, emissiveStrength);
  gpuData.metallicRoughnessOcclusionAlphaCutoff =
      glm::vec4(metallic, roughness, occlusion, alphaCutoff);
  // Pack glossiness for `MaterialWorkflow::SpecularGlossiness`
  // (`desc.glossinessFactor`) or specular intensity otherwise
  // (`desc.specularFactor`) into `specularColorFactorSpecular.w`.
  gpuData.specularColorFactorSpecular = glm::vec4(specularColor, specular);
  gpuData.sheenColorFactorWeight =
      glm::vec4(desc.sheenColorFactor, sheenWeight);
  gpuData.sheenRoughnessClearcoatFactors = glm::vec4(
      sheenRoughness, clearcoat, clearcoatRoughness, desc.clearcoatNormalScale);
  gpuData.transmissionThicknessIorPadding =
      glm::vec4(transmission, thickness, ior, desc.normalScale);
  gpuData.attenuationColorDistance =
      glm::vec4(attenuationColor, attenuationDistance);

  auto baseColorIdx =
      resolveBindlessIndex(gpu, desc.textures.baseColor, "baseColor");
  if (baseColorIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        baseColorIdx.error());
  }
  auto metallicRoughnessIdx = resolveBindlessIndex(
      gpu, desc.textures.metallicRoughness, "metallicRoughness");
  if (metallicRoughnessIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        metallicRoughnessIdx.error());
  }
  auto normalIdx = resolveBindlessIndex(gpu, desc.textures.normal, "normal");
  if (normalIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(normalIdx.error());
  }
  auto occlusionIdx =
      resolveBindlessIndex(gpu, desc.textures.occlusion, "occlusion");
  if (occlusionIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        occlusionIdx.error());
  }
  auto emissiveIdx =
      resolveBindlessIndex(gpu, desc.textures.emissive, "emissive");
  if (emissiveIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(emissiveIdx.error());
  }
  auto clearcoatIdx =
      resolveBindlessIndex(gpu, desc.textures.clearcoat, "clearcoat");
  if (clearcoatIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        clearcoatIdx.error());
  }
  auto clearcoatRoughnessIdx = resolveBindlessIndex(
      gpu, desc.textures.clearcoatRoughness, "clearcoatRoughness");
  if (clearcoatRoughnessIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        clearcoatRoughnessIdx.error());
  }
  auto clearcoatNormalIdx = resolveBindlessIndex(
      gpu, desc.textures.clearcoatNormal, "clearcoatNormal");
  if (clearcoatNormalIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        clearcoatNormalIdx.error());
  }
  auto specularIdx =
      resolveBindlessIndex(gpu, desc.textures.specular, "specular");
  if (specularIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(specularIdx.error());
  }
  auto specularColorIdx =
      resolveBindlessIndex(gpu, desc.textures.specularColor, "specularColor");
  if (specularColorIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        specularColorIdx.error());
  }
  auto sheenColorIdx =
      resolveBindlessIndex(gpu, desc.textures.sheenColor, "sheenColor");
  if (sheenColorIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        sheenColorIdx.error());
  }
  auto sheenRoughnessIdx =
      resolveBindlessIndex(gpu, desc.textures.sheenRoughness, "sheenRoughness");
  if (sheenRoughnessIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        sheenRoughnessIdx.error());
  }
  auto transmissionIdx =
      resolveBindlessIndex(gpu, desc.textures.transmission, "transmission");
  if (transmissionIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        transmissionIdx.error());
  }
  auto thicknessIdx =
      resolveBindlessIndex(gpu, desc.textures.thickness, "thickness");
  if (thicknessIdx.hasError()) {
    return Result<MaterialGpuData, std::string>::makeError(
        thicknessIdx.error());
  }

  gpuData.textureIndices0 =
      glm::uvec4(baseColorIdx.value(), metallicRoughnessIdx.value(),
                 normalIdx.value(), occlusionIdx.value());
  gpuData.textureIndices1 =
      glm::uvec4(emissiveIdx.value(), clearcoatIdx.value(),
                 clearcoatRoughnessIdx.value(), clearcoatNormalIdx.value());
  gpuData.textureIndices2 =
      glm::uvec4(specularIdx.value(), specularColorIdx.value(),
                 sheenColorIdx.value(), sheenRoughnessIdx.value());
  gpuData.textureIndices3 =
      glm::uvec4(transmissionIdx.value(), thicknessIdx.value(),
                 kInvalidTextureBindlessIndex, kInvalidTextureBindlessIndex);
  gpuData.textureUvSets0 = glm::uvec4(clampUvSet(desc.uvSets.baseColor),
                                      clampUvSet(desc.uvSets.metallicRoughness),
                                      clampUvSet(desc.uvSets.normal),
                                      clampUvSet(desc.uvSets.occlusion));
  gpuData.textureUvSets1 = glm::uvec4(
      clampUvSet(desc.uvSets.emissive), clampUvSet(desc.uvSets.clearcoat),
      clampUvSet(desc.uvSets.clearcoatRoughness),
      clampUvSet(desc.uvSets.clearcoatNormal));
  gpuData.textureUvSets2 = glm::uvec4(clampUvSet(desc.uvSets.specular),
                                      clampUvSet(desc.uvSets.specularColor),
                                      clampUvSet(desc.uvSets.sheenColor),
                                      clampUvSet(desc.uvSets.sheenRoughness));
  gpuData.textureUvSets3 =
      glm::uvec4(clampUvSet(desc.uvSets.transmission),
                 clampUvSet(desc.uvSets.thickness), 0u, 0u);
  gpuData.textureSamplerIndices0 =
      glm::uvec4(desc.samplers.baseColor, desc.samplers.metallicRoughness,
                 desc.samplers.normal, desc.samplers.occlusion);
  gpuData.textureSamplerIndices1 = glm::uvec4(
      desc.samplers.emissive, desc.samplers.clearcoat,
      desc.samplers.clearcoatRoughness, desc.samplers.clearcoatNormal);
  gpuData.textureSamplerIndices2 =
      glm::uvec4(desc.samplers.specular, desc.samplers.specularColor,
                 desc.samplers.sheenColor, desc.samplers.sheenRoughness);
  gpuData.textureSamplerIndices3 =
      glm::uvec4(desc.samplers.transmission, desc.samplers.thickness, 0u, 0u);
  for (uint32_t slotIndex = 0; slotIndex < kMaterialTextureSlotCount;
       ++slotIndex) {
    const MaterialTextureTransformData &transform =
        desc.transforms.slots[slotIndex];
    gpuData.textureTransformOffsetScale[slotIndex] =
        glm::vec4(transform.offset, transform.scale);
    gpuData.textureTransformRotation[slotIndex] =
        glm::vec4(std::cos(transform.rotationRadians),
                  std::sin(transform.rotationRadians), 0.0f, 0.0f);
  }
  gpuData.materialFlags = glm::uvec4(static_cast<uint32_t>(desc.alphaMode),
                                     desc.doubleSided ? 1u : 0u, featureMask,
                                     static_cast<uint32_t>(desc.workflow));
  return Result<MaterialGpuData, std::string>::makeResult(gpuData);
}

} // namespace

Result<std::unique_ptr<Material>, std::string>
Material::create(GPUDevice &gpu, const MaterialDesc &desc,
                 std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto gpuDataResult = buildGpuData(gpu, desc);
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
