#include "nuri/pch.h"

#include "nuri/resources/gpu/material.h"

#include "nuri/core/profiling.h"

namespace nuri {
namespace {

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
  const float sheenWeight = std::clamp(desc.sheenWeight, 0.0f, 1.0f);
  const float sheenRoughness =
      std::clamp(desc.sheenRoughnessFactor, 0.0f, 1.0f);
  const float clearcoat = std::clamp(desc.clearcoatFactor, 0.0f, 1.0f);
  const float clearcoatRoughness =
      std::clamp(desc.clearcoatRoughnessFactor, 0.0f, 1.0f);
  const uint32_t featureMask = desc.featureMask;

  MaterialGpuData gpuData{};
  gpuData.baseColorFactor = desc.baseColorFactor;
  gpuData.emissiveFactorNormalScale =
      glm::vec4(desc.emissiveFactor, desc.normalScale);
  gpuData.metallicRoughnessOcclusionAlphaCutoff =
      glm::vec4(metallic, roughness, occlusion, alphaCutoff);
  gpuData.sheenColorFactorWeight =
      glm::vec4(desc.sheenColorFactor, sheenWeight);
  gpuData.sheenRoughnessClearcoatFactors = glm::vec4(
      sheenRoughness, clearcoat, clearcoatRoughness, desc.clearcoatNormalScale);

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

  gpuData.textureIndices0 =
      glm::uvec4(baseColorIdx.value(), metallicRoughnessIdx.value(),
                 normalIdx.value(), occlusionIdx.value());
  gpuData.textureIndices1 =
      glm::uvec4(emissiveIdx.value(), clearcoatIdx.value(),
                 clearcoatRoughnessIdx.value(), clearcoatNormalIdx.value());
  gpuData.textureUvSets0 = glm::uvec4(clampUvSet(desc.uvSets.baseColor),
                                      clampUvSet(desc.uvSets.metallicRoughness),
                                      clampUvSet(desc.uvSets.normal),
                                      clampUvSet(desc.uvSets.occlusion));
  gpuData.textureUvSets1 = glm::uvec4(
      clampUvSet(desc.uvSets.emissive), clampUvSet(desc.uvSets.clearcoat),
      clampUvSet(desc.uvSets.clearcoatRoughness),
      clampUvSet(desc.uvSets.clearcoatNormal));
  gpuData.textureSamplerIndices0 =
      glm::uvec4(desc.samplers.baseColor, desc.samplers.metallicRoughness,
                 desc.samplers.normal, desc.samplers.occlusion);
  gpuData.textureSamplerIndices1 = glm::uvec4(
      desc.samplers.emissive, desc.samplers.clearcoat,
      desc.samplers.clearcoatRoughness, desc.samplers.clearcoatNormal);
  gpuData.materialFlags =
      glm::uvec4(static_cast<uint32_t>(desc.alphaMode),
                 desc.doubleSided ? 1u : 0u, featureMask, 0u);
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

Result<std::unique_ptr<Material>, std::string>
Material::createFromImported(GPUDevice &gpu, const MaterialData &materialData,
                             const MaterialTextureHandles &textures,
                             std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  MaterialDesc desc{};
  desc.baseColorFactor = materialData.baseColorFactor;
  desc.emissiveFactor = materialData.emissiveFactor;
  desc.metallicFactor = materialData.metallicFactor;
  desc.roughnessFactor = materialData.roughnessFactor;
  desc.sheenColorFactor = materialData.sheenColorFactor;
  desc.sheenWeight = materialData.sheenWeight;
  desc.sheenRoughnessFactor = materialData.sheenRoughnessFactor;
  desc.clearcoatFactor = materialData.clearcoatFactor;
  desc.clearcoatRoughnessFactor = materialData.clearcoatRoughnessFactor;
  desc.clearcoatNormalScale = materialData.clearcoatNormalScale;
  desc.normalScale = materialData.normalScale;
  desc.occlusionStrength = materialData.occlusionStrength;
  desc.alphaCutoff = materialData.alphaCutoff;
  desc.doubleSided = materialData.doubleSided;
  desc.alphaMode = materialData.alphaMode;
  desc.featureMask = kMaterialFeatureMetallicRoughness;
  if (desc.sheenWeight > 0.0f) {
    desc.featureMask |= kMaterialFeatureSheen;
  }
  if (desc.clearcoatFactor > 0.0f &&
      (nuri::isValid(textures.clearcoat) ||
       nuri::isValid(textures.clearcoatRoughness) ||
       nuri::isValid(textures.clearcoatNormal))) {
    desc.featureMask |= kMaterialFeatureClearcoat;
  }
  desc.textures = textures;
  desc.uvSets = MaterialTextureUvSets{
      .baseColor = materialData.baseColor.uvSet,
      .metallicRoughness = materialData.metallicRoughness.uvSet,
      .normal = materialData.normal.uvSet,
      .occlusion = materialData.occlusion.uvSet,
      .emissive = materialData.emissive.uvSet,
      .clearcoat = materialData.clearcoat.uvSet,
      .clearcoatRoughness = materialData.clearcoatRoughness.uvSet,
      .clearcoatNormal = materialData.clearcoatNormal.uvSet,
  };
  desc.samplers = MaterialTextureSamplers{
      .baseColor = materialData.baseColor.samplerIndex,
      .metallicRoughness = materialData.metallicRoughness.samplerIndex,
      .normal = materialData.normal.samplerIndex,
      .occlusion = materialData.occlusion.samplerIndex,
      .emissive = materialData.emissive.samplerIndex,
      .clearcoat = materialData.clearcoat.samplerIndex,
      .clearcoatRoughness = materialData.clearcoatRoughness.samplerIndex,
      .clearcoatNormal = materialData.clearcoatNormal.samplerIndex,
  };

  const std::string_view name =
      debugName.empty() ? std::string_view(materialData.name) : debugName;
  return create(gpu, desc, name);
}

} // namespace nuri
