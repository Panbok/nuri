#include "nuri/pch.h"

#include "mesh_importer.h"

#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/detail/gltf_json_utils.h"
#include "nuri/resources/detail/scene_asset_build_backend.h"
#include "nuri/resources/scene_importer.h"

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cmath>
#include <cstdint>
#include <limits>
#if __has_include(<assimp/GltfMaterial.h>)
#include <assimp/GltfMaterial.h>
#define NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS 1
#else
#define NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS 0
#endif

namespace nuri {
namespace {
constexpr float kMeshoptOverdrawThreshold = 1.05f;
constexpr size_t kTriangleIndexCount = 3;
constexpr float kDefaultTextureScale = 1.0f;
constexpr float kDefaultAttenuationDistance = 0.0f;
constexpr float kDefaultIor = 1.5f;
constexpr glm::vec3 kDefaultAttenuationColor(1.0f);
constexpr float kDirectionEpsilon = 1.0e-10f;
using YyJsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
using YyJsonDocResult = Result<YyJsonDocPtr, std::string>;

struct ResolvedExternalTexturePath {
  std::string path{};
  bool usedDdsCompanion = false;
};

[[nodiscard]] ResolvedExternalTexturePath
resolveExternalTexturePath(const std::filesystem::path &modelPath,
                           std::string_view rawPath) {
  if (rawPath.empty()) {
    return {};
  }
  std::filesystem::path texturePath{std::string(rawPath)};
  if (!texturePath.is_absolute()) {
    texturePath = modelPath.parent_path() / texturePath;
  }
  texturePath = texturePath.lexically_normal();

  // Prefer supported precompressed companions when present so we can skip
  // image decode + recompression on assets that already ship with KTX.
  const std::filesystem::path ddsPath =
      texturePath.parent_path() / (texturePath.stem().string() + ".dds");
  if (ddsPath != texturePath && std::filesystem::exists(ddsPath)) {
    return ResolvedExternalTexturePath{
        .path = ddsPath.string(),
        .usedDdsCompanion =
            !detail::hasExtensionCaseInsensitive(texturePath, ".dds"),
    };
  }

  const std::filesystem::path ktx2Path =
      texturePath.parent_path() / (texturePath.stem().string() + ".ktx2");
  if (ktx2Path != texturePath && std::filesystem::exists(ktx2Path)) {
    return ResolvedExternalTexturePath{
        .path = ktx2Path.string(),
        .usedDdsCompanion = false,
    };
  }

  const std::filesystem::path ktxPath =
      texturePath.parent_path() / (texturePath.stem().string() + ".ktx");
  if (ktxPath != texturePath && std::filesystem::exists(ktxPath)) {
    return ResolvedExternalTexturePath{
        .path = ktxPath.string(),
        .usedDdsCompanion = false,
    };
  }

  return ResolvedExternalTexturePath{
      .path = texturePath.string(),
      .usedDdsCompanion = false,
  };
}

void applyVerticalImageFlipToTextureTransform(
    MaterialTextureTransformData &transform) {
  // Auto-substituted DDS companions in NiagaraBistro are authored with the
  // opposite vertical image origin to the original glTF PNG references.
  // Compose a post-transform V flip into the existing affine UV transform.
  transform.offset.y = 1.0f - transform.offset.y;
  transform.scale.y = -transform.scale.y;
  transform.rotationRadians = -transform.rotationRadians;
}

ImportedMaterialTexture
readMaterialTextureSlot(const aiMaterial &material, aiTextureType textureType,
                        uint32_t textureIndex,
                        const std::filesystem::path &modelPath) {
  aiString texturePath;
  aiTextureMapping textureMapping = aiTextureMapping_UV;
  uint32_t uvIndex = 0;
  ai_real blendFactor = 1.0f;
  aiTextureOp textureOp = aiTextureOp_Multiply;
  aiTextureMapMode mapMode[2] = {aiTextureMapMode_Wrap, aiTextureMapMode_Wrap};
  const aiReturn result = material.GetTexture(
      textureType, textureIndex, &texturePath, &textureMapping, &uvIndex,
      &blendFactor, &textureOp, mapMode);
  if (result != aiReturn_SUCCESS || texturePath.length == 0) {
    return {};
  }

  ImportedMaterialTexture out{};
  out.uvSet = uvIndex;
  const std::string_view rawPath(texturePath.C_Str(), texturePath.length);
  if (!rawPath.empty() && rawPath.front() == '*') {
    out.sourceKind = MaterialTextureSourceKind::EmbeddedSceneTexture;
    const std::string_view indexView = rawPath.substr(1u);
    const auto parsedIndex =
        std::from_chars(indexView.data(), indexView.data() + indexView.size(),
                        out.embeddedIndex);
    if (parsedIndex.ec != std::errc() ||
        parsedIndex.ptr != indexView.data() + indexView.size()) {
      out.embeddedIndex = kInvalidEmbeddedSceneTextureIndex;
    }
  } else {
    out.sourceKind = MaterialTextureSourceKind::ExternalFile;
    const ResolvedExternalTexturePath resolved =
        resolveExternalTexturePath(modelPath, rawPath);
    out.path = resolved.path;
    if (resolved.usedDdsCompanion) {
      applyVerticalImageFlipToTextureTransform(out.transform);
    }
  }
  return out;
}

ImportedMaterialTexture
firstAvailableTextureSlot(const aiMaterial &material,
                          const std::filesystem::path &modelPath,
                          std::span<const aiTextureType> textureTypes) {
  for (const aiTextureType textureType : textureTypes) {
    ImportedMaterialTexture slot =
        readMaterialTextureSlot(material, textureType, 0u, modelPath);
    if (!slot.path.empty()) {
      return slot;
    }
  }
  return {};
}

bool hasExtension(const std::filesystem::path &path,
                  std::string_view extension) {
  return detail::hasExtensionCaseInsensitive(path, extension);
}

bool hasExtension(std::string_view path, std::string_view extension) {
  return hasExtension(std::filesystem::path(path), extension);
}

[[nodiscard]] bool isGltfJsonAssetPath(const std::filesystem::path &path) {
  return detail::isGltfJsonAssetPath(path);
}

[[nodiscard]] bool isGltfJsonAssetPath(std::string_view path) {
  return detail::isGltfJsonAssetPath(path);
}

bool isUnsupportedGltfImageUri(std::string_view uri) {
  return uri.empty() || uri.starts_with("data:");
}

bool tryReadJsonFloat(yyjson_val *value, float &out) {
  return detail::tryReadJsonFloat(value, out);
}

bool tryReadJsonUint32(yyjson_val *value, uint32_t &out) {
  return detail::tryReadJsonUint32(value, out);
}

bool tryReadJsonVec2(yyjson_val *value, glm::vec2 &out) {
  return detail::tryReadJsonVec2(value, out);
}

bool tryReadJsonVec3(yyjson_val *value, glm::vec3 &out) {
  return detail::tryReadJsonVec3(value, out);
}

bool tryReadJsonVec4(yyjson_val *value, glm::vec4 &out) {
  return detail::tryReadJsonVec4(value, out);
}

bool tryReadJsonBool(yyjson_val *value, bool &out) {
  if (!yyjson_is_bool(value)) {
    return false;
  }
  out = yyjson_get_bool(value);
  return true;
}

[[nodiscard]] std::string_view readJsonStringView(yyjson_val *value) {
  return detail::readJsonStringView(value);
}

[[nodiscard]] std::string_view readJsonStringView(yyjson_val *object,
                                                  const char *key) {
  return detail::readJsonStringView(object, key);
}

void assignMaterialNameFromGltf(ImportedMaterialInfo &material,
                                yyjson_val *materialValue) {
  const std::string_view materialName =
      readJsonStringView(materialValue, "name");
  if (!materialName.empty()) {
    material.name.assign(materialName);
  }
}

[[nodiscard]] std::string makeFallbackMaterialName(size_t materialIndex) {
  return "material_" + std::to_string(materialIndex);
}

[[nodiscard]] ImportedMaterialAlphaMode
parseGltfAlphaMode(std::string_view alphaMode) {
  if (alphaMode == "MASK") {
    return ImportedMaterialAlphaMode::Mask;
  }
  if (alphaMode == "BLEND") {
    return ImportedMaterialAlphaMode::Blend;
  }
  return ImportedMaterialAlphaMode::Opaque;
}

ImportedMaterialTexture parseGltfTextureSlot(yyjson_val *root,
                                             const std::filesystem::path &path,
                                             yyjson_val *textureInfo,
                                             float *scale = nullptr) {
  if (!yyjson_is_obj(textureInfo)) {
    return {};
  }

  yyjson_val *indexValue = yyjson_obj_get(textureInfo, "index");
  if (!yyjson_is_uint(indexValue) && !yyjson_is_sint(indexValue)) {
    return {};
  }
  uint32_t textureIndex = 0;
  if (!tryReadJsonUint32(indexValue, textureIndex)) {
    return {};
  }

  yyjson_val *textures = yyjson_obj_get(root, "textures");
  if (!yyjson_is_arr(textures)) {
    return {};
  }
  yyjson_val *textureValue = yyjson_arr_get(textures, textureIndex);
  if (!yyjson_is_obj(textureValue)) {
    return {};
  }

  yyjson_val *texCoordValue = yyjson_obj_get(textureInfo, "texCoord");
  yyjson_val *samplerValue = yyjson_obj_get(textureValue, "sampler");
  yyjson_val *scaleValue = yyjson_obj_get(textureInfo, "scale");
  yyjson_val *sourceValue = yyjson_obj_get(textureValue, "source");
  if ((!yyjson_is_uint(sourceValue) && !yyjson_is_sint(sourceValue))) {
    return {};
  }
  uint32_t sourceIndex = 0;
  if (!tryReadJsonUint32(sourceValue, sourceIndex)) {
    return {};
  }

  yyjson_val *images = yyjson_obj_get(root, "images");
  if (!yyjson_is_arr(images)) {
    return {};
  }
  yyjson_val *imageValue = yyjson_arr_get(images, sourceIndex);
  if (!yyjson_is_obj(imageValue)) {
    return {};
  }

  yyjson_val *uriValue = yyjson_obj_get(imageValue, "uri");
  if (!yyjson_is_str(uriValue)) {
    return {};
  }
  const char *uriRaw = yyjson_get_str(uriValue);
  if (uriRaw == nullptr) {
    return {};
  }
  const std::string_view uri(uriRaw);
  if (isUnsupportedGltfImageUri(uri)) {
    return {};
  }

  ImportedMaterialTexture texture{};
  texture.sourceKind = MaterialTextureSourceKind::ExternalFile;
  const ResolvedExternalTexturePath resolvedPath =
      resolveExternalTexturePath(path, uri);
  texture.path = resolvedPath.path;
  if (uint32_t uvSet = 0; tryReadJsonUint32(texCoordValue, uvSet)) {
    texture.uvSet = uvSet;
  }
  if (uint32_t samplerIndex = 0;
      tryReadJsonUint32(samplerValue, samplerIndex)) {
    texture.samplerIndex = samplerIndex;
  }
  if (yyjson_val *extensionsValue = yyjson_obj_get(textureInfo, "extensions");
      yyjson_is_obj(extensionsValue)) {
    yyjson_val *transformValue =
        yyjson_obj_get(extensionsValue, "KHR_texture_transform");
    if (yyjson_is_obj(transformValue)) {
      (void)tryReadJsonVec2(yyjson_obj_get(transformValue, "offset"),
                            texture.transform.offset);
      (void)tryReadJsonVec2(yyjson_obj_get(transformValue, "scale"),
                            texture.transform.scale);
      (void)tryReadJsonFloat(yyjson_obj_get(transformValue, "rotation"),
                             texture.transform.rotationRadians);
      uint32_t transformedUvSet = 0;
      if (tryReadJsonUint32(yyjson_obj_get(transformValue, "texCoord"),
                            transformedUvSet)) {
        texture.uvSet = transformedUvSet;
      }
    }
  }
  if (resolvedPath.usedDdsCompanion) {
    applyVerticalImageFlipToTextureTransform(texture.transform);
  }
  if (scale != nullptr) {
    (void)tryReadJsonFloat(scaleValue, *scale);
  }
  return texture;
}

void updateDerivedSheenState(ImportedMaterialInfo &material) {
  const float sheenMax = std::max(
      material.sheenColorFactor.x,
      std::max(material.sheenColorFactor.y, material.sheenColorFactor.z));
  material.sheenWeight =
      (sheenMax > 0.0f || material.sheenRoughnessFactor > 0.0f ||
       !material.sheenColor.path.empty() ||
       !material.sheenRoughness.path.empty())
          ? 1.0f
          : 0.0f;
}

void updateDerivedTransmissionState(ImportedMaterialInfo &material) {
  material.transmissionFactor =
      std::clamp(material.transmissionFactor, 0.0f, 1.0f);
  material.thicknessFactor = std::max(material.thicknessFactor, 0.0f);
  material.attenuationColor = glm::clamp(material.attenuationColor, 0.0f, 1.0f);
  material.attenuationDistance = std::max(material.attenuationDistance, 0.0f);
}

[[nodiscard]] float sanitizeImportedIor(float ior, const char *context,
                                        const char *materialName = nullptr) {
  if (ior == 0.0f) {
    return 0.0f;
  }
  if (std::isfinite(ior) && ior >= 1.0f) {
    return ior;
  }

  if (materialName != nullptr) {
    NURI_LOG_WARNING("%s: invalid IOR %.6f for material '%s'; clamping to 1.0",
                     context, static_cast<double>(ior), materialName);
  } else {
    NURI_LOG_WARNING("%s: invalid IOR %.6f; clamping to 1.0", context,
                     static_cast<double>(ior));
  }
  return 1.0f;
}

[[nodiscard]] float sanitizeImportedEmissiveStrength(float emissiveStrength) {
  return std::max(emissiveStrength, 0.0f);
}

[[nodiscard]] bool
hasTransmissionContent(const ImportedMaterialInfo &material) {
  return material.transmissionFactor > 0.0f ||
         !material.transmission.path.empty();
}

void warnOnTransmissionBlendCombination(const ImportedMaterialInfo &material,
                                        const char *context) {
  if (!hasTransmissionContent(material) ||
      material.alphaMode != ImportedMaterialAlphaMode::Blend) {
    return;
  }

  if (!material.name.empty()) {
    NURI_LOG_WARNING("%s: material '%s' combines transmission with alphaMode "
                     "BLEND; this uses the expensive exact transparent "
                     "transmission path",
                     context, material.name.c_str());
  } else {
    NURI_LOG_WARNING("%s: unnamed material combines transmission with "
                     "alphaMode BLEND; this uses the expensive exact "
                     "transparent transmission path",
                     context);
  }
}

void finalizeImportedMaterialState(ImportedMaterialInfo &material) {
  updateDerivedSheenState(material);
  updateDerivedTransmissionState(material);
  material.emissiveStrength =
      sanitizeImportedEmissiveStrength(material.emissiveStrength);
  material.ior = sanitizeImportedIor(
      material.ior, "MeshImporter::finalizeImportedMaterialState",
      material.name.empty() ? nullptr : material.name.c_str());
  warnOnTransmissionBlendCombination(
      material, "MeshImporter::finalizeImportedMaterialState");
}

void overlayTextureSlot(ImportedMaterialTexture &target, yyjson_val *root,
                        const std::filesystem::path &modelPath,
                        yyjson_val *textureInfo, float *scale = nullptr) {
  if (!yyjson_is_obj(textureInfo)) {
    return;
  }
  target = parseGltfTextureSlot(root, modelPath, textureInfo, scale);
}

void overlayEmissiveStrengthExtension(ImportedMaterialInfo &material,
                                      yyjson_val *emissiveStrengthExt) {
  if (!yyjson_is_obj(emissiveStrengthExt)) {
    return;
  }

  (void)tryReadJsonFloat(
      yyjson_obj_get(emissiveStrengthExt, "emissiveStrength"),
      material.emissiveStrength);
  material.emissiveStrength =
      sanitizeImportedEmissiveStrength(material.emissiveStrength);
}

void overlayClearcoatExtension(ImportedMaterialInfo &material, yyjson_val *root,
                               const std::filesystem::path &modelPath,
                               yyjson_val *clearcoatExt) {
  if (!yyjson_is_obj(clearcoatExt)) {
    return;
  }

  (void)tryReadJsonFloat(yyjson_obj_get(clearcoatExt, "clearcoatFactor"),
                         material.clearcoatFactor);
  material.clearcoatFactor = std::clamp(material.clearcoatFactor, 0.0f, 1.0f);

  (void)tryReadJsonFloat(
      yyjson_obj_get(clearcoatExt, "clearcoatRoughnessFactor"),
      material.clearcoatRoughnessFactor);
  material.clearcoatRoughnessFactor =
      std::clamp(material.clearcoatRoughnessFactor, 0.0f, 1.0f);

  overlayTextureSlot(material.clearcoat, root, modelPath,
                     yyjson_obj_get(clearcoatExt, "clearcoatTexture"));
  overlayTextureSlot(material.clearcoatRoughness, root, modelPath,
                     yyjson_obj_get(clearcoatExt, "clearcoatRoughnessTexture"));
  overlayTextureSlot(material.clearcoatNormal, root, modelPath,
                     yyjson_obj_get(clearcoatExt, "clearcoatNormalTexture"),
                     &material.clearcoatNormalScale);
}

void overlaySheenExtension(ImportedMaterialInfo &material, yyjson_val *root,
                           const std::filesystem::path &modelPath,
                           yyjson_val *sheenExt) {
  if (!yyjson_is_obj(sheenExt)) {
    return;
  }

  (void)tryReadJsonVec3(yyjson_obj_get(sheenExt, "sheenColorFactor"),
                        material.sheenColorFactor);
  material.sheenColorFactor = glm::clamp(material.sheenColorFactor, 0.0f, 1.0f);

  (void)tryReadJsonFloat(yyjson_obj_get(sheenExt, "sheenRoughnessFactor"),
                         material.sheenRoughnessFactor);
  material.sheenRoughnessFactor =
      std::clamp(material.sheenRoughnessFactor, 0.0f, 1.0f);

  overlayTextureSlot(material.sheenColor, root, modelPath,
                     yyjson_obj_get(sheenExt, "sheenColorTexture"));
  overlayTextureSlot(material.sheenRoughness, root, modelPath,
                     yyjson_obj_get(sheenExt, "sheenRoughnessTexture"));
}

void overlaySpecularExtension(ImportedMaterialInfo &material, yyjson_val *root,
                              const std::filesystem::path &modelPath,
                              yyjson_val *specularExt) {
  if (material.workflow == MaterialWorkflow::SpecularGlossiness) {
    if (yyjson_is_obj(specularExt)) {
      NURI_LOG_WARNING(
          "MeshImporter::overlaySpecularExtension: material '%s' combines "
          "KHR_materials_pbrSpecularGlossiness with KHR_materials_specular; "
          "ignoring the specular extension",
          material.name.empty() ? "<unnamed>" : material.name.c_str());
    }
    return;
  }
  if (!yyjson_is_obj(specularExt)) {
    return;
  }

  (void)tryReadJsonFloat(yyjson_obj_get(specularExt, "specularFactor"),
                         material.specularFactor);
  material.specularFactor = std::clamp(material.specularFactor, 0.0f, 1.0f);

  (void)tryReadJsonVec3(yyjson_obj_get(specularExt, "specularColorFactor"),
                        material.specularColorFactor);
  material.specularColorFactor = glm::max(material.specularColorFactor, 0.0f);

  overlayTextureSlot(material.specular, root, modelPath,
                     yyjson_obj_get(specularExt, "specularTexture"));
  overlayTextureSlot(material.specularColor, root, modelPath,
                     yyjson_obj_get(specularExt, "specularColorTexture"));
}

void overlaySpecularGlossinessExtension(ImportedMaterialInfo &material,
                                        yyjson_val *root,
                                        const std::filesystem::path &modelPath,
                                        yyjson_val *specGlossExt) {
  if (!yyjson_is_obj(specGlossExt)) {
    return;
  }

  material.workflow = MaterialWorkflow::SpecularGlossiness;
  (void)tryReadJsonVec4(yyjson_obj_get(specGlossExt, "diffuseFactor"),
                        material.baseColorFactor);
  material.baseColorFactor = glm::clamp(material.baseColorFactor, 0.0f, 1.0f);

  (void)tryReadJsonVec3(yyjson_obj_get(specGlossExt, "specularFactor"),
                        material.specularColorFactor);
  material.specularColorFactor = glm::max(material.specularColorFactor, 0.0f);
  material.metallicFactor = 0.0f;
  material.roughnessFactor = 1.0f;
  material.metallicRoughness = ImportedMaterialTexture{};
  material.specular = ImportedMaterialTexture{};

  (void)tryReadJsonFloat(yyjson_obj_get(specGlossExt, "glossinessFactor"),
                         material.glossinessFactor);
  material.glossinessFactor = std::clamp(material.glossinessFactor, 0.0f, 1.0f);

  overlayTextureSlot(material.baseColor, root, modelPath,
                     yyjson_obj_get(specGlossExt, "diffuseTexture"));
  overlayTextureSlot(material.specularColor, root, modelPath,
                     yyjson_obj_get(specGlossExt, "specularGlossinessTexture"));
}

void overlayTransmissionExtension(ImportedMaterialInfo &material,
                                  yyjson_val *root,
                                  const std::filesystem::path &modelPath,
                                  yyjson_val *transmissionExt) {
  if (!yyjson_is_obj(transmissionExt)) {
    return;
  }

  (void)tryReadJsonFloat(yyjson_obj_get(transmissionExt, "transmissionFactor"),
                         material.transmissionFactor);
  material.transmissionFactor =
      std::clamp(material.transmissionFactor, 0.0f, 1.0f);
  overlayTextureSlot(material.transmission, root, modelPath,
                     yyjson_obj_get(transmissionExt, "transmissionTexture"));
}

void overlayVolumeExtension(ImportedMaterialInfo &material, yyjson_val *root,
                            const std::filesystem::path &modelPath,
                            yyjson_val *volumeExt) {
  if (!yyjson_is_obj(volumeExt)) {
    return;
  }

  (void)tryReadJsonFloat(yyjson_obj_get(volumeExt, "thicknessFactor"),
                         material.thicknessFactor);
  material.thicknessFactor = std::max(material.thicknessFactor, 0.0f);
  (void)tryReadJsonVec3(yyjson_obj_get(volumeExt, "attenuationColor"),
                        material.attenuationColor);
  material.attenuationColor = glm::clamp(material.attenuationColor, 0.0f, 1.0f);
  (void)tryReadJsonFloat(yyjson_obj_get(volumeExt, "attenuationDistance"),
                         material.attenuationDistance);
  material.attenuationDistance = std::max(material.attenuationDistance, 0.0f);
  overlayTextureSlot(material.thickness, root, modelPath,
                     yyjson_obj_get(volumeExt, "thicknessTexture"));
}

void overlayIorExtension(ImportedMaterialInfo &material, yyjson_val *iorExt) {
  if (!yyjson_is_obj(iorExt)) {
    return;
  }

  (void)tryReadJsonFloat(yyjson_obj_get(iorExt, "ior"), material.ior);
  material.ior = sanitizeImportedIor(
      material.ior, "MeshImporter::overlayIorExtension",
      material.name.empty() ? nullptr : material.name.c_str());
}

glm::vec3 extractTransformScale(const aiMatrix4x4 &transform) {
  const aiVector3D origin = transform * aiVector3D(0.0f, 0.0f, 0.0f);
  const aiVector3D xAxis = transform * aiVector3D(1.0f, 0.0f, 0.0f) - origin;
  const aiVector3D yAxis = transform * aiVector3D(0.0f, 1.0f, 0.0f) - origin;
  const aiVector3D zAxis = transform * aiVector3D(0.0f, 0.0f, 1.0f) - origin;
  return glm::max(glm::vec3(xAxis.Length(), yAxis.Length(), zAxis.Length()),
                  glm::vec3(1.0e-6f));
}

YyJsonDocResult loadGltfJsonDocument(const std::filesystem::path &path);

void overlayMaterialInfoFromGltfValue(ImportedMaterialInfo &material,
                                      yyjson_val *root,
                                      const std::filesystem::path &modelPath,
                                      yyjson_val *materialValue) {
  if (!yyjson_is_obj(materialValue)) {
    return;
  }

  assignMaterialNameFromGltf(material, materialValue);

  material.workflow = MaterialWorkflow::MetallicRoughness;
  yyjson_val *pbrMetallicRoughness =
      yyjson_obj_get(materialValue, "pbrMetallicRoughness");
  yyjson_val *extensions = yyjson_obj_get(materialValue, "extensions");
  yyjson_val *specGlossExt =
      yyjson_is_obj(extensions)
          ? yyjson_obj_get(extensions, "KHR_materials_pbrSpecularGlossiness")
          : nullptr;
  const bool hasSpecGloss = yyjson_is_obj(specGlossExt);

  if (!hasSpecGloss && yyjson_is_obj(pbrMetallicRoughness)) {
    (void)tryReadJsonVec4(
        yyjson_obj_get(pbrMetallicRoughness, "baseColorFactor"),
        material.baseColorFactor);
    (void)tryReadJsonFloat(
        yyjson_obj_get(pbrMetallicRoughness, "metallicFactor"),
        material.metallicFactor);
    (void)tryReadJsonFloat(
        yyjson_obj_get(pbrMetallicRoughness, "roughnessFactor"),
        material.roughnessFactor);
    material.metallicFactor = std::clamp(material.metallicFactor, 0.0f, 1.0f);
    material.roughnessFactor = std::clamp(material.roughnessFactor, 0.0f, 1.0f);

    overlayTextureSlot(
        material.baseColor, root, modelPath,
        yyjson_obj_get(pbrMetallicRoughness, "baseColorTexture"));
    overlayTextureSlot(
        material.metallicRoughness, root, modelPath,
        yyjson_obj_get(pbrMetallicRoughness, "metallicRoughnessTexture"));
  }

  (void)tryReadJsonVec3(yyjson_obj_get(materialValue, "emissiveFactor"),
                        material.emissiveFactor);
  (void)tryReadJsonBool(yyjson_obj_get(materialValue, "doubleSided"),
                        material.doubleSided);
  (void)tryReadJsonFloat(yyjson_obj_get(materialValue, "alphaCutoff"),
                         material.alphaCutoff);

  const std::string_view alphaMode =
      readJsonStringView(materialValue, "alphaMode");
  if (!alphaMode.empty()) {
    material.alphaMode = parseGltfAlphaMode(alphaMode);
  }

  overlayTextureSlot(material.normal, root, modelPath,
                     yyjson_obj_get(materialValue, "normalTexture"),
                     &material.normalScale);
  overlayTextureSlot(material.occlusion, root, modelPath,
                     yyjson_obj_get(materialValue, "occlusionTexture"));
  overlayTextureSlot(material.emissive, root, modelPath,
                     yyjson_obj_get(materialValue, "emissiveTexture"));

  yyjson_val *occlusionTexture =
      yyjson_obj_get(materialValue, "occlusionTexture");
  if (yyjson_is_obj(occlusionTexture)) {
    (void)tryReadJsonFloat(yyjson_obj_get(occlusionTexture, "strength"),
                           material.occlusionStrength);
    material.occlusionStrength =
        std::clamp(material.occlusionStrength, 0.0f, 1.0f);
  }

  if (!yyjson_is_obj(extensions)) {
    finalizeImportedMaterialState(material);
    return;
  }

  overlayEmissiveStrengthExtension(
      material, yyjson_obj_get(extensions, "KHR_materials_emissive_strength"));
  if (hasSpecGloss) {
    overlaySpecularGlossinessExtension(material, root, modelPath, specGlossExt);
  }
  overlayClearcoatExtension(
      material, root, modelPath,
      yyjson_obj_get(extensions, "KHR_materials_clearcoat"));
  overlaySpecularExtension(
      material, root, modelPath,
      yyjson_obj_get(extensions, "KHR_materials_specular"));
  overlaySheenExtension(material, root, modelPath,
                        yyjson_obj_get(extensions, "KHR_materials_sheen"));
  overlayTransmissionExtension(
      material, root, modelPath,
      yyjson_obj_get(extensions, "KHR_materials_transmission"));
  overlayVolumeExtension(material, root, modelPath,
                         yyjson_obj_get(extensions, "KHR_materials_volume"));
  overlayIorExtension(material,
                      yyjson_obj_get(extensions, "KHR_materials_ior"));
  finalizeImportedMaterialState(material);
}

YyJsonDocResult loadGltfJsonDocument(const std::filesystem::path &path) {
  return detail::loadGltfJsonDocument(path, "glTF material overlay source");
}

std::optional<detail::GltfPrimitiveMaterialMapping>
loadGltfPrimitiveMaterialMapping(std::string_view path) {
  if (!isGltfJsonAssetPath(path)) {
    return std::nullopt;
  }

  auto docResult =
      loadGltfJsonDocument(std::filesystem::path(std::string(path)));
  if (docResult.hasError()) {
    NURI_LOG_WARNING(
        "MeshImporter: failed to read glTF primitive materials for '%.*s': %s",
        static_cast<int>(path.size()), path.data(), docResult.error().c_str());
    return std::nullopt;
  }

  auto mappingResult = detail::readGltfPrimitiveMaterialMapping(
      yyjson_doc_get_root(docResult.value().get()));
  if (mappingResult.hasError()) {
    NURI_LOG_WARNING(
        "MeshImporter: failed to parse glTF primitive materials for '%.*s': %s",
        static_cast<int>(path.size()), path.data(),
        mappingResult.error().c_str());
    return std::nullopt;
  }
  return std::move(mappingResult.value());
}

void applyGltfPrimitiveMaterialOverride(
    MeshData &mesh, uint32_t sourceSceneMeshIndex,
    const std::optional<detail::GltfPrimitiveMaterialMapping> &mapping) {
  if (!mapping.has_value() || !mapping->sceneMeshIndicesAreFlatPrimitiveOrder ||
      sourceSceneMeshIndex >= mapping->primitiveMaterialIndices.size()) {
    return;
  }
  const uint32_t sourceMaterialIndex =
      mapping->primitiveMaterialIndices[sourceSceneMeshIndex];
  if (sourceMaterialIndex == std::numeric_limits<uint32_t>::max()) {
    return;
  }
  for (Submesh &submesh : mesh.submeshes) {
    submesh.materialIndex = sourceMaterialIndex;
  }
}

Result<bool, std::string>
overlayMaterialInfoFromGltf(std::string_view path, ImportedMaterialSet &set) {
  const std::string pathString(path);
  const std::filesystem::path modelPath(pathString);
  auto docResult = loadGltfJsonDocument(modelPath);
  if (docResult.hasError()) {
    return Result<bool, std::string>::makeError(docResult.error());
  }
  YyJsonDocPtr doc = std::move(docResult.value());
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  if (!yyjson_is_obj(root)) {
    return Result<bool, std::string>::makeError(
        "glTF root is not a JSON object");
  }

  yyjson_val *materials = yyjson_obj_get(root, "materials");
  if (!yyjson_is_arr(materials)) {
    return Result<bool, std::string>::makeError(
        "glTF materials array is missing");
  }

  const size_t materialCount = yyjson_arr_size(materials);
  if (materialCount == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (set.materials.size() > materialCount) {
    NURI_LOG_WARNING(
        "MeshImporter::loadMaterialInfoFromFile: Assimp reported %zu extra "
        "material(s) for '%s'; using glTF material order",
        set.materials.size() - materialCount, pathString.c_str());
  }

  std::vector<ImportedMaterialInfo> gltfMaterials;
  gltfMaterials.reserve(materialCount);

  for (size_t materialIndex = 0; materialIndex < materialCount;
       ++materialIndex) {
    yyjson_val *materialValue = yyjson_arr_get(materials, materialIndex);
    if (!yyjson_is_obj(materialValue)) {
      ImportedMaterialInfo fallback{};
      fallback.name = makeFallbackMaterialName(materialIndex);
      gltfMaterials.push_back(std::move(fallback));
      continue;
    }

    ImportedMaterialInfo material{};
    overlayMaterialInfoFromGltfValue(material, root, modelPath, materialValue);
    if (material.name.empty()) {
      material.name = makeFallbackMaterialName(materialIndex);
    }
    gltfMaterials.push_back(std::move(material));
  }

  set.materials = std::move(gltfMaterials);
  return Result<bool, std::string>::makeResult(true);
}

ImportedMaterialInfo parseMaterial(const aiMaterial &material,
                                   const std::filesystem::path &modelPath) {
  ImportedMaterialInfo parsed{};

  if (material.GetName().length > 0) {
    parsed.name.assign(material.GetName().C_Str(), material.GetName().length);
  }

  aiColor4D baseColor(1.0f, 1.0f, 1.0f, 1.0f);
#if NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS
  if (material.Get(AI_MATKEY_BASE_COLOR, baseColor) == aiReturn_SUCCESS) {
    parsed.baseColorFactor =
        glm::vec4(baseColor.r, baseColor.g, baseColor.b, baseColor.a);
  } else
#endif
      if (material.Get(AI_MATKEY_COLOR_DIFFUSE, baseColor) ==
          aiReturn_SUCCESS) {
    parsed.baseColorFactor =
        glm::vec4(baseColor.r, baseColor.g, baseColor.b, baseColor.a);
  }

  aiColor3D emissiveColor(0.0f, 0.0f, 0.0f);
  if (material.Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor) ==
      aiReturn_SUCCESS) {
    parsed.emissiveFactor =
        glm::vec3(emissiveColor.r, emissiveColor.g, emissiveColor.b);
  }

  ai_real metallicFactor = 1.0f;
  if (material.Get(AI_MATKEY_METALLIC_FACTOR, metallicFactor) ==
      aiReturn_SUCCESS) {
    parsed.metallicFactor = static_cast<float>(metallicFactor);
  }

  ai_real roughnessFactor = 1.0f;
  if (material.Get(AI_MATKEY_ROUGHNESS_FACTOR, roughnessFactor) ==
      aiReturn_SUCCESS) {
    parsed.roughnessFactor = static_cast<float>(roughnessFactor);
  }

  aiColor3D sheenColor(1.0f, 1.0f, 1.0f);
  if (material.Get(AI_MATKEY_SHEEN_COLOR_FACTOR, sheenColor) ==
      aiReturn_SUCCESS) {
    parsed.sheenColorFactor =
        glm::vec3(sheenColor.r, sheenColor.g, sheenColor.b);
    const float sheenMax = std::max(
        parsed.sheenColorFactor.x,
        std::max(parsed.sheenColorFactor.y, parsed.sheenColorFactor.z));
    parsed.sheenWeight = sheenMax > 0.0f ? 1.0f : 0.0f;
  }
  ai_real sheenRoughness = 0.0f;
  if (material.Get(AI_MATKEY_SHEEN_ROUGHNESS_FACTOR, sheenRoughness) ==
      aiReturn_SUCCESS) {
    parsed.sheenRoughnessFactor =
        std::clamp(static_cast<float>(sheenRoughness), 0.0f, 1.0f);
    if (parsed.sheenWeight == 0.0f && parsed.sheenRoughnessFactor > 0.0f) {
      parsed.sheenWeight = 1.0f;
    }
  }

#if NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS
  ai_real transmissionFactor = 0.0f;
  if (material.Get(AI_MATKEY_TRANSMISSION_FACTOR, transmissionFactor) ==
      aiReturn_SUCCESS) {
    parsed.transmissionFactor =
        std::clamp(static_cast<float>(transmissionFactor), 0.0f, 1.0f);
  }
  ai_real thicknessFactor = 0.0f;
  if (material.Get(AI_MATKEY_VOLUME_THICKNESS_FACTOR, thicknessFactor) ==
      aiReturn_SUCCESS) {
    parsed.thicknessFactor =
        std::max(static_cast<float>(thicknessFactor), 0.0f);
  }
  ai_real attenuationDistance = 0.0f;
  if (material.Get(AI_MATKEY_VOLUME_ATTENUATION_DISTANCE,
                   attenuationDistance) == aiReturn_SUCCESS) {
    parsed.attenuationDistance =
        std::max(static_cast<float>(attenuationDistance), 0.0f);
  }
  aiColor4D attenuationColor(1.0f, 1.0f, 1.0f, 1.0f);
  if (material.Get(AI_MATKEY_VOLUME_ATTENUATION_COLOR, attenuationColor) ==
      aiReturn_SUCCESS) {
    parsed.attenuationColor = glm::clamp(
        glm::vec3(attenuationColor.r, attenuationColor.g, attenuationColor.b),
        0.0f, 1.0f);
  }
#endif

  ai_real ior = kDefaultIor;
  if (material.Get(AI_MATKEY_REFRACTI, ior) == aiReturn_SUCCESS) {
    parsed.ior = sanitizeImportedIor(
        static_cast<float>(ior), "MeshImporter::parseMaterial",
        parsed.name.empty() ? nullptr : parsed.name.c_str());
  }

#if NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS
  ai_real normalScale = kDefaultTextureScale;
  if (material.Get(AI_MATKEY_GLTF_TEXTURE_SCALE(aiTextureType_NORMALS, 0),
                   normalScale) == aiReturn_SUCCESS) {
    parsed.normalScale = static_cast<float>(normalScale);
  }
  ai_real occlusionStrength = kDefaultTextureScale;
  if (material.Get(
          AI_MATKEY_GLTF_TEXTURE_SCALE(aiTextureType_AMBIENT_OCCLUSION, 0),
          occlusionStrength) == aiReturn_SUCCESS) {
    parsed.occlusionStrength = static_cast<float>(occlusionStrength);
  }
#endif

  int32_t doubleSided = 0;
  if (material.Get(AI_MATKEY_TWOSIDED, doubleSided) == aiReturn_SUCCESS) {
    parsed.doubleSided = doubleSided != 0;
  }

#if NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS
  aiString alphaMode;
  if (material.Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode) == aiReturn_SUCCESS) {
    const std::string_view alphaModeStr(alphaMode.C_Str(), alphaMode.length);
    parsed.alphaMode = parseGltfAlphaMode(alphaModeStr);
  }

  ai_real alphaCutoff = parsed.alphaCutoff;
  if (material.Get(AI_MATKEY_GLTF_ALPHACUTOFF, alphaCutoff) ==
      aiReturn_SUCCESS) {
    parsed.alphaCutoff = static_cast<float>(alphaCutoff);
  }
#endif

  parsed.baseColor = firstAvailableTextureSlot(
      material, modelPath,
      std::array<aiTextureType, 2>{aiTextureType_BASE_COLOR,
                                   aiTextureType_DIFFUSE});
  parsed.metallicRoughness = firstAvailableTextureSlot(
      material, modelPath,
      std::array<aiTextureType, 2>{aiTextureType_METALNESS,
                                   aiTextureType_DIFFUSE_ROUGHNESS});
  parsed.normal = firstAvailableTextureSlot(
      material, modelPath, std::array<aiTextureType, 1>{aiTextureType_NORMALS});
  parsed.occlusion = firstAvailableTextureSlot(
      material, modelPath,
      std::array<aiTextureType, 2>{aiTextureType_AMBIENT_OCCLUSION,
                                   aiTextureType_LIGHTMAP});
  parsed.emissive = firstAvailableTextureSlot(
      material, modelPath,
      std::array<aiTextureType, 1>{aiTextureType_EMISSIVE});
  parsed.transmission = readMaterialTextureSlot(
      material, aiTextureType_TRANSMISSION, 0u, modelPath);
  parsed.thickness = readMaterialTextureSlot(
      material, aiTextureType_TRANSMISSION, 1u, modelPath);
  updateDerivedSheenState(parsed);
  updateDerivedTransmissionState(parsed);

  return parsed;
}

size_t meshIndexCount(const aiMesh &mesh) {
  size_t count = 0;
  for (unsigned int faceIndex = 0; faceIndex < mesh.mNumFaces; ++faceIndex) {
    const aiFace &face = mesh.mFaces[faceIndex];
    if (face.mNumIndices != kTriangleIndexCount) {
      continue;
    }
    count += kTriangleIndexCount;
  }
  return count;
}

float lodRatioFor(const MeshImportOptions &options, uint32_t lodIndex) {
  if (options.lodTriangleRatios.empty()) {
    return 0.5f;
  }

  const size_t ratioIndex =
      std::min<size_t>(lodIndex - 1, options.lodTriangleRatios.size() - 1);
  return std::max(options.lodTriangleRatios[ratioIndex], 0.0f);
}

BoundingBox computeSubmeshBounds(std::span<const Vertex> vertices) {
  if (vertices.empty()) {
    return BoundingBox(glm::vec3(0.0f), glm::vec3(0.0f));
  }

  glm::vec3 minPos = vertices.front().position;
  glm::vec3 maxPos = vertices.front().position;
  for (size_t i = 1; i < vertices.size(); ++i) {
    minPos = glm::min(minPos, vertices[i].position);
    maxPos = glm::max(maxPos, vertices[i].position);
  }
  return BoundingBox(minPos, maxPos);
}

uint32_t clampLodCount(const MeshImportOptions &options) {
  const uint32_t maxLodCount =
      std::min(MeshImportOptions::kMaxLodCount, Submesh::kMaxLodCount);
  return std::clamp(options.lodCount, 1u, maxLodCount);
}

size_t sanitizeTargetIndexCount(size_t targetIndexCount,
                                size_t sourceIndexCount) {
  if (sourceIndexCount < kTriangleIndexCount) {
    return 0;
  }
  targetIndexCount =
      std::clamp(targetIndexCount, kTriangleIndexCount, sourceIndexCount);
  targetIndexCount -= targetIndexCount % kTriangleIndexCount;
  return std::max(kTriangleIndexCount, targetIndexCount);
}

size_t targetLodIndexCount(const MeshImportOptions &options, uint32_t lodIndex,
                           size_t sourceIndexCount) {
  return sanitizeTargetIndexCount(
      static_cast<size_t>(static_cast<double>(sourceIndexCount) *
                          lodRatioFor(options, lodIndex)),
      sourceIndexCount);
}

void optimizeIndexOrder(std::span<uint32_t> indices,
                        std::span<const Vertex> vertices) {
  if (indices.empty() || vertices.empty()) {
    return;
  }

  meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(),
                              vertices.size());
  meshopt_optimizeOverdraw(indices.data(), indices.data(), indices.size(),
                           &vertices.front().position.x, vertices.size(),
                           sizeof(Vertex), kMeshoptOverdrawThreshold);
}

void remapMeshVertices(std::pmr::vector<Vertex> &vertices,
                       std::pmr::vector<uint32_t> &indices) {
  if (vertices.empty() || indices.empty()) {
    return;
  }

  std::pmr::memory_resource *mem = vertices.get_allocator().resource();
  std::pmr::vector<unsigned int> remap(mem);
  remap.resize(vertices.size());
  const size_t uniqueVertexCount = meshopt_generateVertexRemap(
      remap.data(), indices.data(), indices.size(), vertices.data(),
      vertices.size(), sizeof(Vertex));
  if (uniqueVertexCount == 0 || uniqueVertexCount > vertices.size()) {
    return;
  }

  std::pmr::vector<uint32_t> remappedIndices(mem);
  remappedIndices.resize(indices.size());
  meshopt_remapIndexBuffer(remappedIndices.data(), indices.data(),
                           indices.size(), remap.data());

  std::pmr::vector<Vertex> remappedVertices(mem);
  remappedVertices.resize(uniqueVertexCount);
  meshopt_remapVertexBuffer(remappedVertices.data(), vertices.data(),
                            vertices.size(), sizeof(Vertex), remap.data());

  indices.swap(remappedIndices);
  vertices.swap(remappedVertices);
}

glm::vec3 normalizeTransformedDirection(const aiVector3D &direction) {
  const glm::vec3 value(direction.x, direction.y, direction.z);
  const float length2 = glm::dot(value, value);
  if (length2 <= kDirectionEpsilon) {
    return glm::vec3(0.0f);
  }
  return value * glm::inversesqrt(length2);
}

glm::vec3 orthogonalizeTangent(glm::vec3 tangent, glm::vec3 normal) {
  const float normalLen2 = glm::dot(normal, normal);
  if (normalLen2 > kDirectionEpsilon) {
    normal *= glm::inversesqrt(normalLen2);
    tangent -= normal * glm::dot(tangent, normal);
  }
  const float tangentLen2 = glm::dot(tangent, tangent);
  if (tangentLen2 <= kDirectionEpsilon) {
    return glm::vec3(0.0f);
  }
  return tangent * glm::inversesqrt(tangentLen2);
}

glm::vec4 transformTangent(const aiMatrix3x3 &directionTransform,
                           const aiVector3D &tangent,
                           const aiVector3D &bitangent,
                           glm::vec3 transformedNormal) {
  const float normalLen2 = glm::dot(transformedNormal, transformedNormal);
  if (normalLen2 > kDirectionEpsilon) {
    transformedNormal *= glm::inversesqrt(normalLen2);
  }
  const glm::vec3 transformedTangent = orthogonalizeTangent(
      normalizeTransformedDirection(directionTransform * tangent),
      transformedNormal);
  if (glm::dot(transformedTangent, transformedTangent) <= kDirectionEpsilon) {
    return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
  }

  const glm::vec3 transformedBitangent =
      normalizeTransformedDirection(directionTransform * bitangent);
  const float handedness =
      glm::dot(glm::cross(transformedNormal, transformedTangent),
               transformedBitangent) < 0.0f
          ? -1.0f
          : 1.0f;
  return glm::vec4(transformedTangent, handedness);
}

aiMatrix4x4 glmMat4ToAiMatrix4x4(const glm::mat4 &matrix) {
  aiMatrix4x4 out;
  out.a1 = matrix[0][0];
  out.a2 = matrix[1][0];
  out.a3 = matrix[2][0];
  out.a4 = matrix[3][0];
  out.b1 = matrix[0][1];
  out.b2 = matrix[1][1];
  out.b3 = matrix[2][1];
  out.b4 = matrix[3][1];
  out.c1 = matrix[0][2];
  out.c2 = matrix[1][2];
  out.c3 = matrix[2][2];
  out.c4 = matrix[3][2];
  out.d1 = matrix[0][3];
  out.d2 = matrix[1][3];
  out.d3 = matrix[2][3];
  out.d4 = matrix[3][3];
  return out;
}

glm::mat4 computeImportedSceneWorldMatrix(uint32_t nodeIndex,
                                          const ImportedScene &scene,
                                          std::span<glm::mat4> cache,
                                          std::span<uint8_t> state) {
  if (nodeIndex >= scene.nodes.size()) {
    return glm::mat4(1.0f);
  }
  if (state[nodeIndex] == 2u) {
    return cache[nodeIndex];
  }
  if (state[nodeIndex] == 1u) {
    NURI_LOG_WARNING(
        "MeshImporter::computeImportedSceneWorldMatrix: detected node cycle at "
        "%u ('%s')",
        nodeIndex, scene.nodes[nodeIndex].name.c_str());
    return glm::mat4(1.0f);
  }

  state[nodeIndex] = 1u;
  glm::mat4 world = scene.nodes[nodeIndex].localFromParent;
  if (scene.nodes[nodeIndex].parentIndex != kInvalidScenePrefabIndex) {
    world = computeImportedSceneWorldMatrix(scene.nodes[nodeIndex].parentIndex,
                                            scene, cache, state) *
            world;
  }
  cache[nodeIndex] = world;
  state[nodeIndex] = 2u;
  return world;
}

[[nodiscard]] uint32_t
findMeshAssetIndexBySourceSceneMeshIndex(const ImportedScene &scene,
                                         uint32_t sourceSceneMeshIndex) {
  for (uint32_t assetIndex = 0u; assetIndex < scene.meshAssets.size();
       ++assetIndex) {
    if (scene.meshAssets[assetIndex].sourceSceneMeshIndex ==
        sourceSceneMeshIndex) {
      return assetIndex;
    }
  }
  return kInvalidScenePrefabIndex;
}

void extractMeshGeometry(const aiMesh &mesh, const aiMatrix4x4 &transform,
                         std::pmr::vector<Vertex> &outVertices,
                         std::pmr::vector<uint32_t> &outIndices) {
  outVertices.clear();
  outVertices.reserve(mesh.mNumVertices);
  aiMatrix3x3 normalTransform(transform);
  normalTransform.Inverse().Transpose();
  const aiMatrix3x3 directionTransform(transform);

  for (unsigned int vertexIndex = 0; vertexIndex < mesh.mNumVertices;
       ++vertexIndex) {
    Vertex vertex{};
    const aiVector3D &pos = mesh.mVertices[vertexIndex];
    const aiVector3D transformedPosition = transform * pos;
    vertex.position = {transformedPosition.x, transformedPosition.y,
                       transformedPosition.z};

    if (mesh.HasNormals()) {
      const aiVector3D &normal = mesh.mNormals[vertexIndex];
      vertex.normal = normalizeTransformedDirection(normalTransform * normal);
    }

    if (mesh.HasTangentsAndBitangents()) {
      vertex.tangent =
          transformTangent(directionTransform, mesh.mTangents[vertexIndex],
                           mesh.mBitangents[vertexIndex], vertex.normal);
    }

    if (mesh.HasTextureCoords(0)) {
      const aiVector3D &uv = mesh.mTextureCoords[0][vertexIndex];
      vertex.uv = {uv.x, uv.y};
    }
    if (mesh.HasTextureCoords(1)) {
      const aiVector3D &uv1 = mesh.mTextureCoords[1][vertexIndex];
      vertex.uv1 = {uv1.x, uv1.y};
    }

    outVertices.push_back(vertex);
  }

  outIndices.clear();
  outIndices.reserve(mesh.mNumFaces * kTriangleIndexCount);
  for (unsigned int faceIndex = 0; faceIndex < mesh.mNumFaces; ++faceIndex) {
    const aiFace &face = mesh.mFaces[faceIndex];
    if (face.mNumIndices != kTriangleIndexCount) {
      continue;
    }
    for (unsigned int i = 0; i < kTriangleIndexCount; ++i) {
      outIndices.push_back(face.mIndices[i]);
    }
  }
}

void extractSkinInfluences(
    const aiMesh &mesh, std::pmr::vector<VertexSkinInfluence> &outInfluences) {
  outInfluences.clear();
  if (!mesh.HasBones() || mesh.mNumVertices == 0u) {
    return;
  }

  outInfluences.resize(mesh.mNumVertices);
  struct InfluenceSlot {
    uint16_t joint = 0u;
    float weight = 0.0f;
  };
  std::pmr::memory_resource *mem = outInfluences.get_allocator().resource();
  std::pmr::vector<std::array<InfluenceSlot, 4>> accumulated(mem);
  accumulated.resize(mesh.mNumVertices);
  std::pmr::vector<uint32_t> counts(mem);
  counts.resize(mesh.mNumVertices, 0u);

  for (uint32_t boneIndex = 0u; boneIndex < mesh.mNumBones; ++boneIndex) {
    const aiBone *bone = mesh.mBones[boneIndex];
    if (bone == nullptr) {
      continue;
    }
    const uint16_t jointIndex = static_cast<uint16_t>(
        std::min<uint32_t>(boneIndex, std::numeric_limits<uint16_t>::max()));
    for (uint32_t weightIndex = 0u; weightIndex < bone->mNumWeights;
         ++weightIndex) {
      const aiVertexWeight &weight = bone->mWeights[weightIndex];
      if (weight.mVertexId >= mesh.mNumVertices || weight.mWeight <= 0.0f) {
        continue;
      }
      auto &slots = accumulated[weight.mVertexId];
      uint32_t &count = counts[weight.mVertexId];
      if (count < slots.size()) {
        slots[count++] =
            InfluenceSlot{.joint = jointIndex, .weight = weight.mWeight};
        continue;
      }
      uint32_t minSlot = 0u;
      for (uint32_t i = 1u; i < slots.size(); ++i) {
        if (slots[i].weight < slots[minSlot].weight) {
          minSlot = i;
        }
      }
      if (weight.mWeight > slots[minSlot].weight) {
        slots[minSlot] =
            InfluenceSlot{.joint = jointIndex, .weight = weight.mWeight};
      }
    }
  }

  for (uint32_t vertexIndex = 0u; vertexIndex < mesh.mNumVertices;
       ++vertexIndex) {
    auto &dst = outInfluences[vertexIndex];
    const auto &slots = accumulated[vertexIndex];
    float totalWeight = 0.0f;
    for (const InfluenceSlot &slot : slots) {
      totalWeight += slot.weight;
    }
    const float weightScale = totalWeight > 0.0f ? (1.0f / totalWeight) : 0.0f;
    for (uint32_t i = 0u; i < slots.size(); ++i) {
      dst.joints[i] = slots[i].joint;
      dst.weights[i] = slots[i].weight * weightScale;
    }
  }
}

void extractMorphTargets(const aiMesh &mesh, const aiMatrix4x4 &transform,
                         std::span<const Vertex> baseVertices,
                         std::pmr::vector<MorphTarget> &outMorphTargets) {
  outMorphTargets.clear();
  if (mesh.mNumAnimMeshes == 0u || baseVertices.empty()) {
    return;
  }

  aiMatrix3x3 normalTransform(transform);
  normalTransform.Inverse().Transpose();
  const aiMatrix3x3 directionTransform(transform);
  outMorphTargets.reserve(mesh.mNumAnimMeshes);
  for (uint32_t animMeshIndex = 0u; animMeshIndex < mesh.mNumAnimMeshes;
       ++animMeshIndex) {
    const aiAnimMesh *animMesh = mesh.mAnimMeshes[animMeshIndex];
    if (animMesh == nullptr) {
      continue;
    }
    outMorphTargets.emplace_back(outMorphTargets.get_allocator().resource());
    MorphTarget &target = outMorphTargets.back();
    if (animMesh->mName.length > 0u) {
      target.name.assign(animMesh->mName.C_Str(), animMesh->mName.length);
    } else {
      target.name = "morph_" + std::to_string(animMeshIndex);
    }
    target.positionDeltas.resize(baseVertices.size(), glm::vec3(0.0f));
    if (animMesh->HasNormals()) {
      target.normalDeltas.resize(baseVertices.size(), glm::vec3(0.0f));
    }
    if (animMesh->HasTangentsAndBitangents()) {
      target.tangentDeltas.resize(baseVertices.size(), glm::vec3(0.0f));
    }

    const uint32_t vertexCount =
        std::min<uint32_t>(mesh.mNumVertices, animMesh->mNumVertices);
    for (uint32_t vertexIndex = 0u; vertexIndex < vertexCount; ++vertexIndex) {
      const aiVector3D transformedPosition =
          transform * animMesh->mVertices[vertexIndex];
      target.positionDeltas[vertexIndex] =
          glm::vec3(transformedPosition.x, transformedPosition.y,
                    transformedPosition.z) -
          baseVertices[vertexIndex].position;
      if (!target.normalDeltas.empty() && animMesh->HasNormals()) {
        const glm::vec3 transformedNormal = normalizeTransformedDirection(
            normalTransform * animMesh->mNormals[vertexIndex]);
        target.normalDeltas[vertexIndex] =
            transformedNormal - baseVertices[vertexIndex].normal;
      }
      if (!target.tangentDeltas.empty() &&
          animMesh->HasTangentsAndBitangents()) {
        const glm::vec3 targetNormal =
            animMesh->HasNormals()
                ? normalizeTransformedDirection(normalTransform *
                                                animMesh->mNormals[vertexIndex])
                : baseVertices[vertexIndex].normal;
        const glm::vec4 transformedTangent = transformTangent(
            directionTransform, animMesh->mTangents[vertexIndex],
            animMesh->mBitangents[vertexIndex], targetNormal);
        target.tangentDeltas[vertexIndex] =
            glm::vec3(transformedTangent) -
            glm::vec3(baseVertices[vertexIndex].tangent);
      }
    }
  }
}

std::array<std::pmr::vector<uint32_t>, Submesh::kMaxLodCount>
makeLodIndexBuffers(std::pmr::memory_resource *mem) {
  static_assert(Submesh::kMaxLodCount == 4,
                "Update makeLodIndexBuffers for new max LOD count");
  return {std::pmr::vector<uint32_t>(mem), std::pmr::vector<uint32_t>(mem),
          std::pmr::vector<uint32_t>(mem), std::pmr::vector<uint32_t>(mem)};
}

uint32_t
buildLodIndexBuffers(const MeshImportOptions &options,
                     uint32_t requestedLodCount, uint32_t meshIndex,
                     std::span<const Vertex> vertices, bool optimize,
                     std::span<std::pmr::vector<uint32_t>> lodIndexBuffers,
                     std::array<float, Submesh::kMaxLodCount> &lodErrors) {
  uint32_t generatedLodCount = 1;

  if (lodIndexBuffers.empty()) {
    return generatedLodCount;
  }

  if (!options.generateLods || requestedLodCount <= 1 ||
      lodIndexBuffers[0].size() < 2 * kTriangleIndexCount) {
    return generatedLodCount;
  }

  NURI_PROFILER_ZONE("MeshImporter.meshopt_lod_generation",
                     NURI_PROFILER_COLOR_CREATE);
  std::pmr::memory_resource *mem =
      lodIndexBuffers[0].get_allocator().resource();
  const size_t baseIndexCount = lodIndexBuffers[0].size();
  for (uint32_t lodIndex = 1; lodIndex < requestedLodCount; ++lodIndex) {
    const size_t targetIndexCount =
        targetLodIndexCount(options, lodIndex, baseIndexCount);
    if (targetIndexCount == 0 || targetIndexCount >= baseIndexCount) {
      continue;
    }

    std::pmr::vector<uint32_t> simplifiedIndices(mem);
    simplifiedIndices.resize(baseIndexCount);
    float lodError = 0.0f;
    size_t simplifiedCount = meshopt_simplify(
        simplifiedIndices.data(), lodIndexBuffers[0].data(), baseIndexCount,
        &vertices.front().position.x, vertices.size(), sizeof(Vertex),
        targetIndexCount, options.lodTargetError, 0, &lodError);
    simplifiedCount -= simplifiedCount % kTriangleIndexCount;
    if (simplifiedCount < kTriangleIndexCount) {
      NURI_LOG_WARNING(
          "MeshImporter::loadFromFile: Mesh %u LOD%u simplification failed, "
          "keeping previous LODs",
          meshIndex, lodIndex);
      break;
    }

    simplifiedIndices.resize(simplifiedCount);
    if (optimize) {
      optimizeIndexOrder(simplifiedIndices, vertices);
    }
    lodIndexBuffers[lodIndex] = std::move(simplifiedIndices);
    lodErrors[lodIndex] = lodError;
    generatedLodCount = lodIndex + 1;
  }
  NURI_PROFILER_ZONE_END();
  return generatedLodCount;
}

void optimizeVertexFetchForAllLods(
    std::pmr::vector<Vertex> &vertices, uint32_t lodCount,
    std::span<std::pmr::vector<uint32_t>> lodIndexBuffers,
    std::pmr::vector<VertexSkinInfluence> *skinInfluences = nullptr,
    std::span<MorphTarget> morphTargets = {}) {
  if (vertices.empty() || lodCount == 0 || lodIndexBuffers.empty() ||
      lodIndexBuffers[0].empty()) {
    return;
  }

  NURI_PROFILER_ZONE("MeshImporter.meshopt_vertex_fetch",
                     NURI_PROFILER_COLOR_CREATE);
  std::pmr::memory_resource *mem = vertices.get_allocator().resource();
  std::pmr::vector<unsigned int> vertexFetchRemap(mem);
  vertexFetchRemap.resize(vertices.size());
  const size_t optimizedVertexCount = meshopt_optimizeVertexFetchRemap(
      vertexFetchRemap.data(), lodIndexBuffers[0].data(),
      lodIndexBuffers[0].size(), vertices.size());
  if (optimizedVertexCount > 0 && optimizedVertexCount <= vertices.size()) {
    std::pmr::vector<Vertex> remappedVertices(mem);
    remappedVertices.resize(optimizedVertexCount);
    meshopt_remapVertexBuffer(remappedVertices.data(), vertices.data(),
                              vertices.size(), sizeof(Vertex),
                              vertexFetchRemap.data());
    vertices.swap(remappedVertices);

    if (skinInfluences != nullptr &&
        skinInfluences->size() == vertexFetchRemap.size()) {
      std::pmr::vector<VertexSkinInfluence> remappedSkinInfluences(mem);
      remappedSkinInfluences.resize(optimizedVertexCount);
      for (size_t oldIndex = 0; oldIndex < vertexFetchRemap.size();
           ++oldIndex) {
        const unsigned int newIndex = vertexFetchRemap[oldIndex];
        if (newIndex < remappedSkinInfluences.size()) {
          remappedSkinInfluences[newIndex] = (*skinInfluences)[oldIndex];
        }
      }
      skinInfluences->swap(remappedSkinInfluences);
    } else if (skinInfluences != nullptr && !skinInfluences->empty()) {
      NURI_LOG_DEBUG(
          "MeshImporter::optimizeVertexFetchForAllLods: skipping skin "
          "influence remap because influence count (%zu) does not match "
          "vertex remap size (%zu)",
          skinInfluences->size(), vertexFetchRemap.size());
    }

    const auto remapDeltas = [&](std::pmr::vector<glm::vec3> &deltas) {
      if (deltas.empty()) {
        return;
      }
      std::pmr::vector<glm::vec3> remapped(mem);
      remapped.resize(optimizedVertexCount, glm::vec3(0.0f));
      for (size_t oldIndex = 0; oldIndex < vertexFetchRemap.size();
           ++oldIndex) {
        const unsigned int newIndex = vertexFetchRemap[oldIndex];
        if (newIndex < remapped.size() && oldIndex < deltas.size()) {
          remapped[newIndex] = deltas[oldIndex];
        }
      }
      deltas.swap(remapped);
    };

    for (MorphTarget &morphTarget : morphTargets) {
      remapDeltas(morphTarget.positionDeltas);
      remapDeltas(morphTarget.normalDeltas);
      remapDeltas(morphTarget.tangentDeltas);
    }

    for (uint32_t lodIndex = 0; lodIndex < lodCount; ++lodIndex) {
      std::pmr::vector<uint32_t> remappedIndices(mem);
      remappedIndices.resize(lodIndexBuffers[lodIndex].size());
      meshopt_remapIndexBuffer(
          remappedIndices.data(), lodIndexBuffers[lodIndex].data(),
          lodIndexBuffers[lodIndex].size(), vertexFetchRemap.data());
      lodIndexBuffers[lodIndex].swap(remappedIndices);
    }
  }
  NURI_PROFILER_ZONE_END();
}

void appendSubmeshToMeshData(
    MeshData &data, const aiMesh &mesh, std::span<const Vertex> vertices,
    std::span<const VertexSkinInfluence> skinInfluences,
    std::span<const MorphTarget> morphTargets, const BoundingBox &bounds,
    const glm::vec3 &authoredScale, uint32_t sourceMaterialIndex,
    uint32_t lodCount,
    std::span<const std::pmr::vector<uint32_t>> lodIndexBuffers,
    const std::array<float, Submesh::kMaxLodCount> &lodErrors,
    uint32_t meshIndex) {
  const uint32_t vertexBase = static_cast<uint32_t>(data.vertices.size());
  data.vertices.insert(data.vertices.end(), vertices.begin(), vertices.end());
  if (!data.skinInfluences.empty() || !skinInfluences.empty()) {
    if (data.skinInfluences.size() < vertexBase) {
      data.skinInfluences.resize(vertexBase);
    }
    if (!skinInfluences.empty()) {
      data.skinInfluences.insert(data.skinInfluences.end(),
                                 skinInfluences.begin(), skinInfluences.end());
    } else {
      data.skinInfluences.resize(vertexBase + vertices.size());
    }
  }

  Submesh submesh{};
  submesh.vertexOffset = vertexBase;
  submesh.vertexCount = static_cast<uint32_t>(vertices.size());
  submesh.materialIndex = sourceMaterialIndex;
  submesh.morphTargetFirst = static_cast<uint32_t>(data.morphTargets.size());
  submesh.morphTargetCount = static_cast<uint32_t>(morphTargets.size());
  submesh.bounds = bounds;
  submesh.authoredScale = authoredScale;
  submesh.lodCount = lodCount;

  for (uint32_t lodIndex = 0; lodIndex < lodCount; ++lodIndex) {
    const uint32_t lodOffset = static_cast<uint32_t>(data.indices.size());
    for (uint32_t localIndex : lodIndexBuffers[lodIndex]) {
      data.indices.push_back(vertexBase + localIndex);
    }

    const uint32_t lodIndexCount =
        static_cast<uint32_t>(data.indices.size() - lodOffset);
    submesh.lods[lodIndex] = SubmeshLod{
        .indexOffset = lodOffset,
        .indexCount = lodIndexCount,
        .error = lodErrors[lodIndex],
    };
    if (lodIndex == 0) {
      submesh.indexOffset = lodOffset;
      submesh.indexCount = lodIndexCount;
    }
  }

  if (submesh.indexCount == 0) {
    NURI_LOG_WARNING("MeshImporter::loadFromFile: Mesh %u LOD0 is empty",
                     meshIndex);
    return;
  }

  for (const MorphTarget &morphTarget : morphTargets) {
    data.morphTargets.emplace_back(
        data.morphTargets.get_allocator().resource());
    MorphTarget &dst = data.morphTargets.back();
    dst.name.assign(morphTarget.name.data(), morphTarget.name.size());
    dst.positionDeltas.assign(morphTarget.positionDeltas.begin(),
                              morphTarget.positionDeltas.end());
    dst.normalDeltas.assign(morphTarget.normalDeltas.begin(),
                            morphTarget.normalDeltas.end());
    dst.tangentDeltas.assign(morphTarget.tangentDeltas.begin(),
                             morphTarget.tangentDeltas.end());
  }
  data.submeshes.push_back(submesh);
}

[[nodiscard]] std::string importedSceneName(const aiScene &scene,
                                            std::string_view path) {
  if (scene.mName.length > 0u) {
    return std::string(scene.mName.C_Str(), scene.mName.length);
  }
  return std::filesystem::path(std::string(path)).stem().string();
}

unsigned int buildAssimpFlags(const MeshImportOptions &options,
                              bool preTransformVertices,
                              bool preserveSceneIndices = false);

[[nodiscard]] nuri::Result<const aiScene *, std::string>
loadSceneMeshImportScene(Assimp::Importer &importer, std::string_view path,
                         const nuri::MeshImportOptions &options) {
  const std::string pathStr(path);
  // Scene-mesh indices come from the structural import path, so this second
  // import must not reorder/merge meshes or remap materials underneath those
  // indices.
  const unsigned int flags = buildAssimpFlags(options, false, true);
  const aiScene *scene = importer.ReadFile(pathStr, flags);
  if (!scene || !scene->HasMeshes()) {
    const std::string error =
        scene ? "Assimp scene has no meshes" : importer.GetErrorString();
    return nuri::Result<const aiScene *, std::string>::makeError(error);
  }
  return nuri::Result<const aiScene *, std::string>::makeResult(scene);
}

[[nodiscard]] nuri::Result<const aiMesh *, std::string>
resolveSceneMesh(const aiScene &scene, uint32_t sceneMeshIndex,
                 std::string_view context) {
  if (sceneMeshIndex == std::numeric_limits<uint32_t>::max()) {
    return nuri::Result<const aiMesh *, std::string>::makeError(
        std::string(context) + ": mesh index is invalid");
  }
  if (sceneMeshIndex >= scene.mNumMeshes ||
      scene.mMeshes[sceneMeshIndex] == nullptr) {
    return nuri::Result<const aiMesh *, std::string>::makeError(
        std::string(context) + ": mesh index " +
        std::to_string(sceneMeshIndex) + " is out of range");
  }
  return nuri::Result<const aiMesh *, std::string>::makeResult(
      scene.mMeshes[sceneMeshIndex]);
}

[[nodiscard]] nuri::Result<MeshData, std::string>
buildSceneMeshData(const aiMesh &mesh, uint32_t sceneMeshIndex,
                   std::string_view sceneName,
                   const nuri::MeshImportOptions &options,
                   ScratchArena &scratch, std::pmr::memory_resource *mem) {
  MeshData data(mem);
  data.name.assign(sceneName.data(), sceneName.size());

  const uint32_t requestedLodCount = clampLodCount(options);
  ScopedScratch scopedScratch(scratch);
  std::pmr::vector<Vertex> meshVertices(scopedScratch.resource());
  std::pmr::vector<VertexSkinInfluence> meshSkinInfluences(
      scopedScratch.resource());
  std::pmr::vector<MorphTarget> meshMorphTargets(scopedScratch.resource());
  std::pmr::vector<uint32_t> lod0Indices(scopedScratch.resource());
  std::array<std::pmr::vector<uint32_t>, Submesh::kMaxLodCount>
      lodIndexBuffers = makeLodIndexBuffers(scopedScratch.resource());
  std::array<float, Submesh::kMaxLodCount> lodErrors{};

  extractMeshGeometry(mesh, aiMatrix4x4(), meshVertices, lod0Indices);
  extractSkinInfluences(mesh, meshSkinInfluences);
  extractMorphTargets(mesh, aiMatrix4x4(), meshVertices, meshMorphTargets);
  if (meshVertices.empty() || lod0Indices.size() < kTriangleIndexCount) {
    return nuri::Result<MeshData, std::string>::makeError(
        "mesh has insufficient geometry");
  }

  if (options.optimize) {
    NURI_PROFILER_ZONE("MeshImporter.scene_mesh_optimize",
                       NURI_PROFILER_COLOR_CREATE);
    remapMeshVertices(meshVertices, lod0Indices);
    optimizeIndexOrder(lod0Indices, meshVertices);
    NURI_PROFILER_ZONE_END();
  }

  lodIndexBuffers[0] = std::move(lod0Indices);
  const uint32_t generatedLodCount = buildLodIndexBuffers(
      options, requestedLodCount, sceneMeshIndex, meshVertices,
      options.optimize, lodIndexBuffers, lodErrors);
  if (options.optimize) {
    optimizeVertexFetchForAllLods(
        meshVertices, generatedLodCount, lodIndexBuffers, &meshSkinInfluences,
        std::span<MorphTarget>(meshMorphTargets.data(),
                               meshMorphTargets.size()));
  }

  data.vertices.reserve(meshVertices.size());
  size_t totalIndexCount = 0;
  for (uint32_t lodIndex = 0; lodIndex < generatedLodCount; ++lodIndex) {
    totalIndexCount += lodIndexBuffers[lodIndex].size();
  }
  data.indices.reserve(totalIndexCount);
  const BoundingBox submeshBounds = computeSubmeshBounds(meshVertices);
  appendSubmeshToMeshData(data, mesh, meshVertices, meshSkinInfluences,
                          meshMorphTargets, submeshBounds, glm::vec3(1.0f),
                          mesh.mMaterialIndex, generatedLodCount,
                          lodIndexBuffers, lodErrors, sceneMeshIndex);
  return nuri::Result<MeshData, std::string>::makeResult(std::move(data));
}

[[nodiscard]] nuri::Result<MeshData, std::string>
buildFlattenedSceneDataFromImportedScene(std::string_view path,
                                         const ImportedScene &importedScene,
                                         const MeshImportOptions &options,
                                         std::pmr::memory_resource *mem,
                                         bool includeAnimationData = true) {
  Assimp::Importer importer;
  auto sceneResult = loadSceneMeshImportScene(importer, path, options);
  if (sceneResult.hasError()) {
    return nuri::Result<MeshData, std::string>::makeError(sceneResult.error());
  }
  const aiScene *scene = sceneResult.value();

  MeshData data(mem);
  data.name.assign(importedScene.sourceSceneName.data(),
                   importedScene.sourceSceneName.size());
  if (data.name.empty()) {
    const std::string fallbackName = importedSceneName(*scene, path);
    data.name.assign(fallbackName.data(), fallbackName.size());
  }

  const uint32_t requestedLodCount = clampLodCount(options);
  ScratchArena scratch(mem);
  std::pmr::vector<glm::mat4> worldCache(mem);
  worldCache.resize(importedScene.nodes.size(), glm::mat4(1.0f));
  std::pmr::vector<uint8_t> worldState(mem);
  worldState.resize(importedScene.nodes.size(), 0u);

  size_t totalVertices = 0;
  size_t totalIndices = 0;
  const auto reserveForMesh = [&](uint32_t sourceSceneMeshIndex) {
    auto meshResult = resolveSceneMesh(*scene, sourceSceneMeshIndex,
                                       "MeshImporter::loadFromFile");
    if (meshResult.hasError()) {
      return;
    }
    const aiMesh &mesh = *meshResult.value();
    totalVertices += mesh.mNumVertices;
    const size_t baseIndexCount = meshIndexCount(mesh);
    totalIndices += baseIndexCount;
    if (options.generateLods && requestedLodCount > 1) {
      for (uint32_t lodIndex = 1; lodIndex < requestedLodCount; ++lodIndex) {
        totalIndices += targetLodIndexCount(options, lodIndex, baseIndexCount);
      }
    }
  };
  if (!importedScene.renderables.empty()) {
    for (const ImportedSceneRenderable &renderable :
         importedScene.renderables) {
      if (renderable.meshAssetIndex >= importedScene.meshAssets.size()) {
        continue;
      }
      reserveForMesh(importedScene.meshAssets[renderable.meshAssetIndex]
                         .sourceSceneMeshIndex);
    }
    data.submeshes.reserve(importedScene.renderables.size());
  } else {
    for (const ImportedSceneMeshAsset &meshAsset : importedScene.meshAssets) {
      reserveForMesh(meshAsset.sourceSceneMeshIndex);
    }
    data.submeshes.reserve(importedScene.meshAssets.size());
  }
  data.vertices.reserve(totalVertices);
  data.indices.reserve(totalIndices);

  size_t insufficientGeometryMeshCount = 0;
  std::array<uint32_t, 8> insufficientGeometryMeshSamples{};
  size_t insufficientGeometrySampleCount = 0;

  const auto appendRenderable = [&](uint32_t sourceSceneMeshIndex,
                                    const aiMatrix4x4 &transform,
                                    const glm::vec3 &authoredScale,
                                    uint32_t sourceMaterialIndex) -> void {
    auto meshResult = resolveSceneMesh(*scene, sourceSceneMeshIndex,
                                       "MeshImporter::loadFromFile");
    if (meshResult.hasError()) {
      return;
    }
    const aiMesh &mesh = *meshResult.value();

    ScopedScratch scopedScratch(scratch);
    std::pmr::vector<Vertex> meshVertices(scopedScratch.resource());
    std::pmr::vector<VertexSkinInfluence> meshSkinInfluences(
        scopedScratch.resource());
    std::pmr::vector<MorphTarget> meshMorphTargets(scopedScratch.resource());
    std::pmr::vector<uint32_t> lod0Indices(scopedScratch.resource());
    std::array<std::pmr::vector<uint32_t>, Submesh::kMaxLodCount>
        lodIndexBuffers = makeLodIndexBuffers(scopedScratch.resource());
    std::array<float, Submesh::kMaxLodCount> lodErrors{};

    extractMeshGeometry(mesh, transform, meshVertices, lod0Indices);
    if (includeAnimationData) {
      extractSkinInfluences(mesh, meshSkinInfluences);
      extractMorphTargets(mesh, transform, meshVertices, meshMorphTargets);
    }
    if (meshVertices.empty() || lod0Indices.size() < kTriangleIndexCount) {
      ++insufficientGeometryMeshCount;
      if (insufficientGeometrySampleCount <
          insufficientGeometryMeshSamples.size()) {
        insufficientGeometryMeshSamples[insufficientGeometrySampleCount++] =
            sourceSceneMeshIndex;
      }
      return;
    }

    if (options.optimize) {
      NURI_PROFILER_ZONE("MeshImporter.meshopt_base_optimize",
                         NURI_PROFILER_COLOR_CREATE);
      remapMeshVertices(meshVertices, lod0Indices);
      optimizeIndexOrder(lod0Indices, meshVertices);
      NURI_PROFILER_ZONE_END();
    }

    lodIndexBuffers[0] = std::move(lod0Indices);
    const uint32_t generatedLodCount = buildLodIndexBuffers(
        options, requestedLodCount, sourceSceneMeshIndex, meshVertices,
        options.optimize, lodIndexBuffers, lodErrors);

    if (options.optimize) {
      optimizeVertexFetchForAllLods(
          meshVertices, generatedLodCount, lodIndexBuffers,
          includeAnimationData ? &meshSkinInfluences : nullptr,
          includeAnimationData ? std::span<MorphTarget>(meshMorphTargets.data(),
                                                        meshMorphTargets.size())
                               : std::span<MorphTarget>());
    }

    const BoundingBox submeshBounds = computeSubmeshBounds(meshVertices);
    appendSubmeshToMeshData(data, mesh, meshVertices, meshSkinInfluences,
                            meshMorphTargets, submeshBounds, authoredScale,
                            sourceMaterialIndex, generatedLodCount,
                            lodIndexBuffers, lodErrors, sourceSceneMeshIndex);
  };

  if (!importedScene.renderables.empty()) {
    for (const ImportedSceneRenderable &renderable :
         importedScene.renderables) {
      if (renderable.meshAssetIndex >= importedScene.meshAssets.size() ||
          renderable.materialAssetIndex >=
              importedScene.materialAssets.size()) {
        continue;
      }
      const ImportedSceneMeshAsset &meshAsset =
          importedScene.meshAssets[renderable.meshAssetIndex];
      const ImportedSceneMaterialAsset &materialAsset =
          importedScene.materialAssets[renderable.materialAssetIndex];
      const glm::mat4 worldTransform = computeImportedSceneWorldMatrix(
          renderable.nodeIndex, importedScene, worldCache, worldState);
      const aiMatrix4x4 aiTransform = glmMat4ToAiMatrix4x4(worldTransform);
      const glm::vec3 authoredScale = extractTransformScale(aiTransform);
      appendRenderable(meshAsset.sourceSceneMeshIndex, aiTransform,
                       authoredScale, materialAsset.sourceMaterialIndex);
    }
  } else {
    for (const ImportedSceneMeshAsset &meshAsset : importedScene.meshAssets) {
      auto meshResult = resolveSceneMesh(*scene, meshAsset.sourceSceneMeshIndex,
                                         "MeshImporter::loadFromFile");
      if (meshResult.hasError()) {
        continue;
      }
      appendRenderable(meshAsset.sourceSceneMeshIndex, aiMatrix4x4(),
                       glm::vec3(1.0f), meshResult.value()->mMaterialIndex);
    }
  }

  if (insufficientGeometryMeshCount > 0) {
    std::ostringstream sampleStream;
    for (size_t sampleIndex = 0; sampleIndex < insufficientGeometrySampleCount;
         ++sampleIndex) {
      if (sampleIndex > 0) {
        sampleStream << ", ";
      }
      sampleStream << insufficientGeometryMeshSamples[sampleIndex];
    }
    NURI_LOG_WARNING(
        "MeshImporter::loadFromFile: skipped %zu mesh(es) with insufficient "
        "triangle geometry (sample indices: %s)",
        insufficientGeometryMeshCount, sampleStream.str().c_str());
  }

  return nuri::Result<MeshData, std::string>::makeResult(std::move(data));
}

unsigned int buildAssimpFlags(const MeshImportOptions &options,
                              bool preTransformVertices,
                              bool preserveSceneIndices) {
  unsigned int flags = aiProcess_SortByPType | aiProcess_FindDegenerates |
                       aiProcess_FindInvalidData;
  if (preTransformVertices) {
    // Pre-transform vertices because the runtime model format does not retain
    // source scene-graph transforms.
    flags |= aiProcess_PreTransformVertices;
  }

  if (options.triangulate) {
    flags |= aiProcess_Triangulate;
  }

  if (options.joinIdenticalVertices) {
    flags |= aiProcess_JoinIdenticalVertices;
  }

  if (options.genNormals) {
    flags |= aiProcess_GenSmoothNormals;
  }

  if (options.calcTangents) {
    flags |= aiProcess_CalcTangentSpace;
  }

  if (options.flipUVs) {
    flags |= aiProcess_FlipUVs;
  }

  if (options.genUVCoords) {
    flags |= aiProcess_GenUVCoords;
  }

  if (options.removeRedundantMaterials && !preserveSceneIndices) {
    flags |= aiProcess_RemoveRedundantMaterials;
  }

  if (options.limitBoneWeights) {
    flags |= aiProcess_LimitBoneWeights;
  }

  if (options.optimize && !preserveSceneIndices) {
    flags |= aiProcess_OptimizeMeshes;
  }

  return flags;
}
} // namespace

nuri::Result<MeshData, std::string>
MeshImporter::loadFromFile(std::string_view path,
                           const MeshImportOptions &options,
                           std::pmr::memory_resource *mem) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (path.empty()) {
    NURI_LOG_WARNING("MeshImporter::loadFromFile: Path is empty");
    return nuri::Result<MeshData, std::string>::makeError("Path is empty");
  }

  if (!mem) {
    mem = std::pmr::get_default_resource();
  }
  const SceneImportOptions sceneOptions{.assetBuildOptions = options};
  auto importedSceneResult =
      SceneImporter::loadSceneFromFile(path, sceneOptions, mem);
  if (importedSceneResult.hasError()) {
    return nuri::Result<MeshData, std::string>::makeError(
        "MeshImporter::loadFromFile: " + importedSceneResult.error());
  }
  auto flattenedResult = buildFlattenedSceneDataFromImportedScene(
      path, importedSceneResult.value(), options, mem, false);
  if (flattenedResult.hasError()) {
    return flattenedResult;
  }
  return nuri::Result<MeshData, std::string>::makeResult(
      std::move(flattenedResult.value()));
}

nuri::Result<MeshData, std::string> detail::loadSceneMeshFromSourceIndex(
    std::string_view path, uint32_t sceneMeshIndex,
    const MeshImportOptions &options, std::pmr::memory_resource *mem) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (path.empty()) {
    return nuri::Result<MeshData, std::string>::makeError(
        "detail::loadSceneMeshFromSourceIndex: path is empty");
  }
  if (!mem) {
    mem = std::pmr::get_default_resource();
  }

  Assimp::Importer importer;
  auto sceneResult = loadSceneMeshImportScene(importer, path, options);
  if (sceneResult.hasError()) {
    return nuri::Result<MeshData, std::string>::makeError(sceneResult.error());
  }
  const aiScene *scene = sceneResult.value();

  auto meshResult = resolveSceneMesh(*scene, sceneMeshIndex,
                                     "MeshImporter::loadSceneMeshFromFile");
  if (meshResult.hasError()) {
    return nuri::Result<MeshData, std::string>::makeError(meshResult.error());
  }

  const std::string sceneName = importedSceneName(*scene, path);
  ScratchArena scratch(mem);
  auto meshDataResult = buildSceneMeshData(*meshResult.value(), sceneMeshIndex,
                                           sceneName, options, scratch, mem);
  if (meshDataResult.hasError()) {
    return nuri::Result<MeshData, std::string>::makeError(
        "detail::loadSceneMeshFromSourceIndex: " + meshDataResult.error());
  }
  auto mapping = loadGltfPrimitiveMaterialMapping(path);
  applyGltfPrimitiveMaterialOverride(meshDataResult.value(), sceneMeshIndex,
                                     mapping);
  return meshDataResult;
}

nuri::Result<std::pmr::vector<MeshData>, std::string>
detail::loadSceneMeshesFromSourceIndices(
    std::string_view path, std::span<const uint32_t> sceneMeshIndices,
    const MeshImportOptions &options, std::pmr::memory_resource *mem) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (path.empty()) {
    return nuri::Result<std::pmr::vector<MeshData>, std::string>::makeError(
        "detail::loadSceneMeshesFromSourceIndices: path is empty");
  }
  if (!mem) {
    mem = std::pmr::get_default_resource();
  }

  std::pmr::vector<MeshData> meshData(mem);
  if (sceneMeshIndices.empty()) {
    return nuri::Result<std::pmr::vector<MeshData>, std::string>::makeResult(
        std::move(meshData));
  }

  Assimp::Importer importer;
  auto sceneResult = loadSceneMeshImportScene(importer, path, options);
  if (sceneResult.hasError()) {
    return nuri::Result<std::pmr::vector<MeshData>, std::string>::makeError(
        sceneResult.error());
  }
  const aiScene *scene = sceneResult.value();

  meshData.reserve(sceneMeshIndices.size());
  const std::string sceneName = importedSceneName(*scene, path);
  ScratchArena scratch(mem);
  auto mapping = loadGltfPrimitiveMaterialMapping(path);

  for (const uint32_t sceneMeshIndex : sceneMeshIndices) {
    auto meshResult = resolveSceneMesh(*scene, sceneMeshIndex,
                                       "MeshImporter::loadSceneMeshesFromFile");
    if (meshResult.hasError()) {
      return nuri::Result<std::pmr::vector<MeshData>, std::string>::makeError(
          meshResult.error());
    }

    auto meshDataResult = buildSceneMeshData(
        *meshResult.value(), sceneMeshIndex, sceneName, options, scratch, mem);
    if (meshDataResult.hasError()) {
      return nuri::Result<std::pmr::vector<MeshData>, std::string>::makeError(
          "detail::loadSceneMeshesFromSourceIndices: mesh index " +
          std::to_string(sceneMeshIndex) + ": " + meshDataResult.error());
    }
    applyGltfPrimitiveMaterialOverride(meshDataResult.value(), sceneMeshIndex,
                                       mapping);
    meshData.push_back(std::move(meshDataResult.value()));
  }

  return nuri::Result<std::pmr::vector<MeshData>, std::string>::makeResult(
      std::move(meshData));
}

nuri::Result<ImportedMaterialSet, std::string>
detail::loadMaterialInfoFromSourceFile(std::string_view path) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (path.empty()) {
    return nuri::Result<ImportedMaterialSet, std::string>::makeError(
        "detail::loadMaterialInfoFromSourceFile: path is empty");
  }

  Assimp::Importer importer;
  const std::string pathStr(path);
  constexpr unsigned int kMaterialOnlyFlags =
      aiProcess_SortByPType | aiProcess_FindInvalidData;
  const aiScene *scene = importer.ReadFile(pathStr, kMaterialOnlyFlags);
  if (!scene) {
    return nuri::Result<ImportedMaterialSet, std::string>::makeError(
        std::string("MeshImporter::loadMaterialInfoFromFile: Assimp error: ") +
        importer.GetErrorString());
  }

  ImportedMaterialSet set{};
  const std::filesystem::path modelPath(pathStr);
  if (!scene->HasMaterials()) {
    NURI_LOG_WARNING("MeshImporter::loadMaterialInfoFromFile: scene '%s' has "
                     "no materials",
                     pathStr.c_str());
    return nuri::Result<ImportedMaterialSet, std::string>::makeResult(
        std::move(set));
  }

  set.materials.reserve(scene->mNumMaterials);
  for (uint32_t materialIndex = 0; materialIndex < scene->mNumMaterials;
       ++materialIndex) {
    const aiMaterial *material = scene->mMaterials[materialIndex];
    if (!material) {
      ImportedMaterialInfo fallback{};
      fallback.name = makeFallbackMaterialName(materialIndex);
      set.materials.push_back(std::move(fallback));
      continue;
    }

    ImportedMaterialInfo parsed = parseMaterial(*material, modelPath);
    if (parsed.name.empty()) {
      parsed.name = makeFallbackMaterialName(materialIndex);
    }
    set.materials.push_back(std::move(parsed));
  }

  if (isGltfJsonAssetPath(path)) {
    auto overlayResult = overlayMaterialInfoFromGltf(path, set);
    if (overlayResult.hasError()) {
      NURI_LOG_WARNING(
          "MeshImporter::loadMaterialInfoFromFile: glTF material overlay "
          "skipped for '%s': %s",
          pathStr.c_str(), overlayResult.error().c_str());
    }
  }

  NURI_LOG_DEBUG("MeshImporter::loadMaterialInfoFromFile: extracted %zu "
                 "material(s) from '%s'",
                 set.materials.size(), pathStr.c_str());
  return nuri::Result<ImportedMaterialSet, std::string>::makeResult(
      std::move(set));
}

nuri::Result<MeshData, std::string> MeshImporter::loadSceneMeshFromFile(
    std::string_view path, uint32_t sceneMeshIndex,
    const MeshImportOptions &options, std::pmr::memory_resource *mem) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!mem) {
    mem = std::pmr::get_default_resource();
  }
  return detail::loadSceneMeshFromSourceIndex(path, sceneMeshIndex, options,
                                              mem);
}

nuri::Result<std::pmr::vector<MeshData>, std::string>
MeshImporter::loadSceneMeshesFromFile(
    std::string_view path, std::span<const uint32_t> sceneMeshIndices,
    const MeshImportOptions &options, std::pmr::memory_resource *mem) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!mem) {
    mem = std::pmr::get_default_resource();
  }
  if (sceneMeshIndices.empty()) {
    return nuri::Result<std::pmr::vector<MeshData>, std::string>::makeResult(
        std::pmr::vector<MeshData>(mem));
  }
  return detail::loadSceneMeshesFromSourceIndices(path, sceneMeshIndices,
                                                  options, mem);
}

nuri::Result<ImportedMaterialSet, std::string>
MeshImporter::loadMaterialInfoFromFile(std::string_view path) {
  return detail::loadMaterialInfoFromSourceFile(path);
}

} // namespace nuri
