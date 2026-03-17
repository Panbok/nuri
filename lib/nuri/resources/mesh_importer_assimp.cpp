#include "nuri/pch.h"

#include "mesh_importer.h"

#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"

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
constexpr uint32_t kGlbMagic = 0x46546C67u;
constexpr uint32_t kGlbVersion2 = 2u;
constexpr uint32_t kGlbChunkTypeJson = 0x4E4F534Au;
using YyJsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
using YyJsonDocResult = Result<YyJsonDocPtr, std::string>;

struct AssimpMeshInstance {
  const aiMesh *mesh = nullptr;
  uint32_t meshIndex = 0;
  aiMatrix4x4 transform{};
};

struct GltfMeshInstanceScale {
  uint32_t meshIndex = 0;
  glm::vec3 scale{1.0f};
};

std::string normalizeExternalTexturePath(const std::filesystem::path &modelPath,
                                         std::string_view rawPath) {
  if (rawPath.empty()) {
    return {};
  }
  std::filesystem::path texturePath{std::string(rawPath)};
  if (!texturePath.is_absolute()) {
    texturePath = modelPath.parent_path() / texturePath;
  }
  return texturePath.lexically_normal().string();
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
  out.isEmbedded = !rawPath.empty() && rawPath.front() == '*';
  out.path = out.isEmbedded ? std::string(rawPath)
                            : normalizeExternalTexturePath(modelPath, rawPath);
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
  return path.has_extension() && path.extension().string() == extension;
}

bool hasExtension(std::string_view path, std::string_view extension) {
  return hasExtension(std::filesystem::path(path), extension);
}

[[nodiscard]] bool isGltfJsonAssetPath(const std::filesystem::path &path) {
  return hasExtension(path, ".gltf") || hasExtension(path, ".glb");
}

[[nodiscard]] bool isGltfJsonAssetPath(std::string_view path) {
  return isGltfJsonAssetPath(std::filesystem::path(path));
}

bool isUnsupportedGltfImageUri(std::string_view uri) {
  return uri.empty() || uri.starts_with("data:");
}

bool tryReadJsonFloat(yyjson_val *value, float &out) {
  if (yyjson_is_uint(value)) {
    out = static_cast<float>(yyjson_get_uint(value));
    return true;
  }
  if (yyjson_is_sint(value)) {
    out = static_cast<float>(yyjson_get_sint(value));
    return true;
  }
  if (yyjson_is_real(value) || yyjson_is_num(value)) {
    out = static_cast<float>(yyjson_get_real(value));
    return true;
  }
  return false;
}

bool tryReadJsonUint32(yyjson_val *value, uint32_t &out) {
  if (yyjson_is_uint(value)) {
    const uint64_t raw = yyjson_get_uint(value);
    if (raw > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    out = static_cast<uint32_t>(raw);
    return true;
  }
  if (yyjson_is_sint(value)) {
    const int64_t raw = yyjson_get_sint(value);
    if (raw < 0 ||
        raw > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return false;
    }
    out = static_cast<uint32_t>(raw);
    return true;
  }
  return false;
}

bool tryReadJsonVec2(yyjson_val *value, glm::vec2 &out) {
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) < 2u) {
    return false;
  }
  float x = 0.0f;
  float y = 0.0f;
  if (!tryReadJsonFloat(yyjson_arr_get(value, 0u), x) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 1u), y)) {
    return false;
  }
  out = glm::vec2(x, y);
  return true;
}

bool tryReadJsonVec3(yyjson_val *value, glm::vec3 &out) {
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) < 3u) {
    return false;
  }
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  if (!tryReadJsonFloat(yyjson_arr_get(value, 0u), x) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 1u), y) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 2u), z)) {
    return false;
  }
  out = glm::vec3(x, y, z);
  return true;
}

bool tryReadJsonVec4(yyjson_val *value, glm::vec4 &out) {
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) < 4u) {
    return false;
  }
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;
  if (!tryReadJsonFloat(yyjson_arr_get(value, 0u), x) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 1u), y) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 2u), z) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 3u), w)) {
    return false;
  }
  out = glm::vec4(x, y, z, w);
  return true;
}

bool tryReadJsonBool(yyjson_val *value, bool &out) {
  if (!yyjson_is_bool(value)) {
    return false;
  }
  out = yyjson_get_bool(value);
  return true;
}

constexpr size_t kInvalidMaterialIndex = std::numeric_limits<size_t>::max();

[[nodiscard]] std::string_view readJsonStringView(yyjson_val *value) {
  if (!yyjson_is_str(value)) {
    return {};
  }
  const char *raw = yyjson_get_str(value);
  return raw != nullptr ? std::string_view(raw) : std::string_view{};
}

[[nodiscard]] std::string_view readJsonStringView(yyjson_val *object,
                                                  const char *key) {
  if (!yyjson_is_obj(object)) {
    return {};
  }
  return readJsonStringView(yyjson_obj_get(object, key));
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

[[nodiscard]] size_t
findMaterialIndexByName(std::span<const ImportedMaterialInfo> materials,
                        const std::vector<bool> &matchedExisting,
                        std::string_view materialName) {
  if (materialName.empty()) {
    return kInvalidMaterialIndex;
  }
  if (matchedExisting.size() != materials.size()) {
    NURI_LOG_WARNING(
        "findMaterialIndexByName: matchedExisting size %zu does not match "
        "materials size %zu",
        matchedExisting.size(), materials.size());
    return kInvalidMaterialIndex;
  }

  for (size_t materialIndex = 0; materialIndex < materials.size();
       ++materialIndex) {
    if (matchedExisting[materialIndex]) {
      continue;
    }
    if (materials[materialIndex].name == materialName) {
      return materialIndex;
    }
  }
  return kInvalidMaterialIndex;
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
  texture.path = normalizeExternalTexturePath(path, uri);
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
  if (scale != nullptr) {
    (void)tryReadJsonFloat(scaleValue, *scale);
  }
  return texture;
}

void resetGltfOverlayState(ImportedMaterialInfo &material) {
  material.emissiveStrength = 1.0f;
  material.clearcoatFactor = 0.0f;
  material.clearcoatRoughnessFactor = 0.0f;
  material.clearcoatNormalScale = kDefaultTextureScale;
  material.clearcoat = ImportedMaterialTexture{};
  material.clearcoatRoughness = ImportedMaterialTexture{};
  material.clearcoatNormal = ImportedMaterialTexture{};
  material.specularFactor = 1.0f;
  material.specularColorFactor = glm::vec3(1.0f);
  material.specular = ImportedMaterialTexture{};
  material.specularColor = ImportedMaterialTexture{};
  material.sheenColorFactor = glm::vec3(0.0f);
  material.sheenWeight = 0.0f;
  material.sheenRoughnessFactor = 0.0f;
  material.sheenColor = ImportedMaterialTexture{};
  material.sheenRoughness = ImportedMaterialTexture{};
  material.transmissionFactor = 0.0f;
  material.thicknessFactor = 0.0f;
  material.attenuationColor = kDefaultAttenuationColor;
  material.attenuationDistance = kDefaultAttenuationDistance;
  material.ior = kDefaultIor;
  material.transmission = ImportedMaterialTexture{};
  material.thickness = ImportedMaterialTexture{};
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
                     "BLEND; glTF optical transparency is expected to use "
                     "OPAQUE or MASK",
                     context, material.name.c_str());
  } else {
    NURI_LOG_WARNING("%s: unnamed material combines transmission with "
                     "alphaMode BLEND; glTF optical transparency is expected "
                     "to use OPAQUE or MASK",
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

glm::vec3 readGltfNodeLocalScale(yyjson_val *nodeValue) {
  if (!yyjson_is_obj(nodeValue)) {
    return glm::vec3(1.0f);
  }

  glm::vec3 scale(1.0f);
  if (tryReadJsonVec3(yyjson_obj_get(nodeValue, "scale"), scale)) {
    return scale;
  }

  yyjson_val *matrixValue = yyjson_obj_get(nodeValue, "matrix");
  if (!yyjson_is_arr(matrixValue) || yyjson_arr_size(matrixValue) < 16u) {
    return scale;
  }

  std::array<float, 16> matrix{};
  for (uint32_t i = 0; i < 16u; ++i) {
    if (!tryReadJsonFloat(yyjson_arr_get(matrixValue, i), matrix[i])) {
      return glm::vec3(1.0f);
    }
  }

  const glm::vec3 basisX(matrix[0], matrix[1], matrix[2]);
  const glm::vec3 basisY(matrix[4], matrix[5], matrix[6]);
  const glm::vec3 basisZ(matrix[8], matrix[9], matrix[10]);
  return glm::max(
      glm::vec3(glm::length(basisX), glm::length(basisY), glm::length(basisZ)),
      glm::vec3(1.0e-6f));
}

void collectGltfMeshInstanceScalesRecursive(
    yyjson_val *nodes, yyjson_val *nodeValue, const glm::vec3 &parentScale,
    std::pmr::vector<GltfMeshInstanceScale> &out) {
  if (!yyjson_is_obj(nodeValue)) {
    return;
  }

  const glm::vec3 localScale = readGltfNodeLocalScale(nodeValue);
  const glm::vec3 globalScale = parentScale * localScale;

  uint32_t meshIndex = 0;
  if (tryReadJsonUint32(yyjson_obj_get(nodeValue, "mesh"), meshIndex)) {
    out.push_back(GltfMeshInstanceScale{
        .meshIndex = meshIndex,
        .scale = globalScale,
    });
  }

  yyjson_val *childrenValue = yyjson_obj_get(nodeValue, "children");
  if (!yyjson_is_arr(childrenValue)) {
    return;
  }

  yyjson_arr_iter childIter = yyjson_arr_iter_with(childrenValue);
  yyjson_val *childValue = nullptr;
  while ((childValue = yyjson_arr_iter_next(&childIter)) != nullptr) {
    uint32_t childIndex = 0;
    if (!tryReadJsonUint32(childValue, childIndex)) {
      continue;
    }
    yyjson_val *childNode = yyjson_arr_get(nodes, childIndex);
    collectGltfMeshInstanceScalesRecursive(nodes, childNode, globalScale, out);
  }
}

YyJsonDocResult loadGltfJsonDocument(const std::filesystem::path &path);

Result<std::pmr::vector<GltfMeshInstanceScale>, std::string>
loadGltfMeshInstanceScales(std::string_view path,
                           std::pmr::memory_resource *mem) {
  if (!isGltfJsonAssetPath(path)) {
    return Result<std::pmr::vector<GltfMeshInstanceScale>, std::string>::
        makeResult(std::pmr::vector<GltfMeshInstanceScale>(mem));
  }

  auto docResult = loadGltfJsonDocument(std::filesystem::path(path));
  if (docResult.hasError()) {
    return Result<std::pmr::vector<GltfMeshInstanceScale>,
                  std::string>::makeError(docResult.error());
  }

  yyjson_doc *doc = docResult.value().get();
  yyjson_val *root = yyjson_doc_get_root(doc);
  if (!yyjson_is_obj(root)) {
    return Result<std::pmr::vector<GltfMeshInstanceScale>,
                  std::string>::makeError("glTF root is not an object");
  }

  yyjson_val *nodes = yyjson_obj_get(root, "nodes");
  yyjson_val *scenes = yyjson_obj_get(root, "scenes");
  if (!yyjson_is_arr(nodes) || !yyjson_is_arr(scenes)) {
    return Result<std::pmr::vector<GltfMeshInstanceScale>, std::string>::
        makeResult(std::pmr::vector<GltfMeshInstanceScale>(mem));
  }

  uint32_t sceneIndex = 0u;
  (void)tryReadJsonUint32(yyjson_obj_get(root, "scene"), sceneIndex);
  yyjson_val *sceneValue = yyjson_arr_get(scenes, sceneIndex);
  if (!yyjson_is_obj(sceneValue)) {
    sceneValue = yyjson_arr_get(scenes, 0u);
  }
  if (!yyjson_is_obj(sceneValue)) {
    return Result<std::pmr::vector<GltfMeshInstanceScale>, std::string>::
        makeResult(std::pmr::vector<GltfMeshInstanceScale>(mem));
  }

  std::pmr::vector<GltfMeshInstanceScale> out(mem);
  yyjson_val *sceneNodes = yyjson_obj_get(sceneValue, "nodes");
  if (!yyjson_is_arr(sceneNodes)) {
    return Result<std::pmr::vector<GltfMeshInstanceScale>,
                  std::string>::makeResult(std::move(out));
  }

  out.reserve(yyjson_arr_size(sceneNodes));
  yyjson_arr_iter nodeIter = yyjson_arr_iter_with(sceneNodes);
  yyjson_val *nodeIndexValue = nullptr;
  while ((nodeIndexValue = yyjson_arr_iter_next(&nodeIter)) != nullptr) {
    uint32_t nodeIndex = 0;
    if (!tryReadJsonUint32(nodeIndexValue, nodeIndex)) {
      continue;
    }
    yyjson_val *nodeValue = yyjson_arr_get(nodes, nodeIndex);
    collectGltfMeshInstanceScalesRecursive(nodes, nodeValue, glm::vec3(1.0f),
                                           out);
  }

  return Result<std::pmr::vector<GltfMeshInstanceScale>,
                std::string>::makeResult(std::move(out));
}

glm::vec3
resolveAuthoredScale(const AssimpMeshInstance &instance,
                     std::span<const GltfMeshInstanceScale> gltfInstanceScales,
                     size_t instanceOrdinal) {
  glm::vec3 authoredScale = extractTransformScale(instance.transform);
  if (instanceOrdinal < gltfInstanceScales.size()) {
    const GltfMeshInstanceScale &gltfScale =
        gltfInstanceScales[instanceOrdinal];
    if (gltfScale.meshIndex == instance.meshIndex) {
      return gltfScale.scale;
    }
  }
  return authoredScale;
}

void overlayMaterialInfoFromGltfValue(ImportedMaterialInfo &material,
                                      yyjson_val *root,
                                      const std::filesystem::path &modelPath,
                                      yyjson_val *materialValue) {
  if (!yyjson_is_obj(materialValue)) {
    return;
  }

  assignMaterialNameFromGltf(material, materialValue);

  yyjson_val *pbrMetallicRoughness =
      yyjson_obj_get(materialValue, "pbrMetallicRoughness");
  if (yyjson_is_obj(pbrMetallicRoughness)) {
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

  yyjson_val *extensions = yyjson_obj_get(materialValue, "extensions");
  if (!yyjson_is_obj(extensions)) {
    finalizeImportedMaterialState(material);
    return;
  }

  overlayEmissiveStrengthExtension(
      material, yyjson_obj_get(extensions, "KHR_materials_emissive_strength"));
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
  const auto parseJsonDocument = [](std::string jsonText) -> YyJsonDocResult {
    yyjson_read_err parseError{};
    yyjson_doc *rawDoc = yyjson_read_opts(jsonText.data(), jsonText.size(), 0,
                                          nullptr, &parseError);
    if (rawDoc == nullptr) {
      return YyJsonDocResult::makeError("yyjson parse failed at offset " +
                                        std::to_string(parseError.pos));
    }
    return YyJsonDocResult::makeResult(YyJsonDocPtr(rawDoc, &yyjson_doc_free));
  };
  const auto readU32 = [](std::span<const uint8_t> bytes, size_t offset,
                          uint32_t &out) -> bool {
    if (offset + sizeof(uint32_t) > bytes.size()) {
      return false;
    }
    out = static_cast<uint32_t>(bytes[offset]) |
          (static_cast<uint32_t>(bytes[offset + 1u]) << 8u) |
          (static_cast<uint32_t>(bytes[offset + 2u]) << 16u) |
          (static_cast<uint32_t>(bytes[offset + 3u]) << 24u);
    return true;
  };

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return YyJsonDocResult::makeError(
        "Failed to open glTF material overlay source");
  }

  if (hasExtension(path, ".gltf")) {
    std::ostringstream jsonStream;
    jsonStream << file.rdbuf();
    return parseJsonDocument(jsonStream.str());
  }

  if (!hasExtension(path, ".glb")) {
    return YyJsonDocResult::makeError(
        "Unsupported glTF material overlay file extension");
  }

  file.seekg(0, std::ios::end);
  const std::streamoff fileSize = file.tellg();
  if (fileSize < 0) {
    return YyJsonDocResult::makeError("Failed to determine .glb file size");
  }
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> bytes(static_cast<size_t>(fileSize));
  if (!bytes.empty() &&
      !file.read(reinterpret_cast<char *>(bytes.data()), fileSize)) {
    return YyJsonDocResult::makeError("Failed to read .glb file");
  }
  if (bytes.size() < 20u) {
    return YyJsonDocResult::makeError(".glb file is too small");
  }

  uint32_t magic = 0u;
  uint32_t version = 0u;
  uint32_t declaredLength = 0u;
  if (!readU32(bytes, 0u, magic) || !readU32(bytes, 4u, version) ||
      !readU32(bytes, 8u, declaredLength)) {
    return YyJsonDocResult::makeError("Failed to read .glb header");
  }
  if (magic != kGlbMagic) {
    return YyJsonDocResult::makeError(".glb magic mismatch");
  }
  if (version != kGlbVersion2) {
    return YyJsonDocResult::makeError(".glb version is not 2");
  }
  if (declaredLength != bytes.size()) {
    return YyJsonDocResult::makeError(".glb declared length mismatch");
  }

  size_t chunkOffset = 12u;
  while (chunkOffset + 8u <= bytes.size()) {
    uint32_t chunkLength = 0u;
    uint32_t chunkType = 0u;
    if (!readU32(bytes, chunkOffset, chunkLength) ||
        !readU32(bytes, chunkOffset + 4u, chunkType)) {
      return YyJsonDocResult::makeError("Failed to read .glb chunk header");
    }
    chunkOffset += 8u;
    if (chunkLength > bytes.size() - chunkOffset) {
      return YyJsonDocResult::makeError(".glb chunk exceeds file bounds");
    }
    if (chunkType == kGlbChunkTypeJson) {
      return parseJsonDocument(std::string(
          reinterpret_cast<const char *>(bytes.data() + chunkOffset),
          chunkLength));
    }
    chunkOffset += chunkLength;
  }

  return YyJsonDocResult::makeError(".glb JSON chunk is missing");
}

Result<bool, std::string>
overlayMaterialInfoFromGltf(std::string_view path, ImportedMaterialSet &set) {
  const std::filesystem::path modelPath(path);
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
        "material(s) for '%s'; keeping unmatched Assimp materials intact",
        set.materials.size() - materialCount, std::string(path).c_str());
  }

  std::vector<bool> matchedExisting(set.materials.size(), false);

  for (size_t materialIndex = 0; materialIndex < materialCount;
       ++materialIndex) {
    yyjson_val *materialValue = yyjson_arr_get(materials, materialIndex);
    if (!yyjson_is_obj(materialValue)) {
      continue;
    }

    size_t targetIndex =
        findMaterialIndexByName(set.materials, matchedExisting,
                                readJsonStringView(materialValue, "name"));

    if (targetIndex == kInvalidMaterialIndex &&
        materialIndex < set.materials.size() &&
        !matchedExisting[materialIndex]) {
      targetIndex = materialIndex;
    }

    if (targetIndex != kInvalidMaterialIndex) {
      matchedExisting[targetIndex] = true;
      resetGltfOverlayState(set.materials[targetIndex]);
      overlayMaterialInfoFromGltfValue(set.materials[targetIndex], root,
                                       modelPath, materialValue);
      continue;
    }

    ImportedMaterialInfo material{};
    overlayMaterialInfoFromGltfValue(material, root, modelPath, materialValue);
    if (material.name.empty()) {
      material.name = makeFallbackMaterialName(materialIndex);
    }
    set.materials.push_back(std::move(material));
  }

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
  if (length2 <= std::numeric_limits<float>::epsilon()) {
    return glm::vec3(0.0f);
  }
  return value * glm::inversesqrt(length2);
}

void collectMeshInstancesRecursive(const aiScene &scene, const aiNode &node,
                                   const aiMatrix4x4 &parentTransform,
                                   std::pmr::vector<AssimpMeshInstance> &out) {
  const aiMatrix4x4 globalTransform = parentTransform * node.mTransformation;

  for (unsigned int meshSlot = 0; meshSlot < node.mNumMeshes; ++meshSlot) {
    const unsigned int meshIndex = node.mMeshes[meshSlot];
    if (meshIndex >= scene.mNumMeshes) {
      continue;
    }
    const aiMesh *mesh = scene.mMeshes[meshIndex];
    if (!mesh) {
      continue;
    }
    out.push_back(AssimpMeshInstance{
        .mesh = mesh,
        .meshIndex = meshIndex,
        .transform = globalTransform,
    });
  }

  for (unsigned int childIndex = 0; childIndex < node.mNumChildren;
       ++childIndex) {
    const aiNode *child = node.mChildren[childIndex];
    if (!child) {
      continue;
    }
    collectMeshInstancesRecursive(scene, *child, globalTransform, out);
  }
}

std::pmr::vector<AssimpMeshInstance>
collectMeshInstances(const aiScene &scene, std::pmr::memory_resource *mem) {
  std::pmr::vector<AssimpMeshInstance> instances(mem);
  if (scene.mRootNode != nullptr) {
    collectMeshInstancesRecursive(scene, *scene.mRootNode, aiMatrix4x4(),
                                  instances);
  }

  if (!instances.empty()) {
    return instances;
  }

  instances.reserve(scene.mNumMeshes);
  for (unsigned int meshIndex = 0; meshIndex < scene.mNumMeshes; ++meshIndex) {
    const aiMesh *mesh = scene.mMeshes[meshIndex];
    if (!mesh) {
      continue;
    }
    instances.push_back(AssimpMeshInstance{
        .mesh = mesh,
        .meshIndex = meshIndex,
        .transform = aiMatrix4x4(),
    });
  }
  return instances;
}

void extractMeshGeometry(const aiMesh &mesh, const aiMatrix4x4 &transform,
                         std::pmr::vector<Vertex> &outVertices,
                         std::pmr::vector<uint32_t> &outIndices) {
  outVertices.clear();
  outVertices.reserve(mesh.mNumVertices);
  aiMatrix3x3 normalTransform(transform);
  normalTransform.Inverse().Transpose();
  const float tangentHandednessScale =
      normalTransform.Determinant() < 0.0f ? -1.0f : 1.0f;

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

    if (mesh.HasTextureCoords(0)) {
      const aiVector3D &uv = mesh.mTextureCoords[0][vertexIndex];
      vertex.uv = {uv.x, uv.y};
    }
    if (mesh.HasTextureCoords(1)) {
      const aiVector3D &uv1 = mesh.mTextureCoords[1][vertexIndex];
      vertex.uv1 = {uv1.x, uv1.y};
    }

    if (mesh.HasTangentsAndBitangents()) {
      const aiVector3D &tangent = mesh.mTangents[vertexIndex];
      const aiVector3D &bitangent = mesh.mBitangents[vertexIndex];
      const glm::vec3 n = vertex.normal;
      const glm::vec3 t =
          normalizeTransformedDirection(normalTransform * tangent);
      const glm::vec3 b =
          normalizeTransformedDirection(normalTransform * bitangent);
      const float sign = (glm::dot(glm::cross(n, t), b) < 0.0f) ? -1.0f : 1.0f;
      vertex.tangent = {t.x, t.y, t.z, sign * tangentHandednessScale};
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
    std::span<std::pmr::vector<uint32_t>> lodIndexBuffers) {
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
    const BoundingBox &bounds, const glm::vec3 &authoredScale,
    uint32_t lodCount,
    std::span<const std::pmr::vector<uint32_t>> lodIndexBuffers,
    const std::array<float, Submesh::kMaxLodCount> &lodErrors,
    uint32_t meshIndex) {
  const uint32_t vertexBase = static_cast<uint32_t>(data.vertices.size());
  data.vertices.insert(data.vertices.end(), vertices.begin(), vertices.end());

  Submesh submesh{};
  submesh.materialIndex = mesh.mMaterialIndex;
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

  data.submeshes.push_back(submesh);
}

unsigned int buildAssimpFlags(const MeshImportOptions &options) {
  // Pre-transform vertices because the runtime model format does not retain
  // source scene-graph transforms.
  unsigned int flags = aiProcess_SortByPType | aiProcess_FindDegenerates |
                       aiProcess_FindInvalidData |
                       aiProcess_PreTransformVertices;

  if (options.triangulate) {
    flags |= aiProcess_Triangulate;
  }

  if (options.joinIdenticalVertices) {
    flags |= aiProcess_JoinIdenticalVertices;
  }

  if (options.genNormals) {
    flags |= aiProcess_GenSmoothNormals;
  }

  if (options.genTangents) {
    flags |= aiProcess_CalcTangentSpace;
  }

  if (options.flipUVs) {
    flags |= aiProcess_FlipUVs;
  }

  if (options.genUVCoords) {
    flags |= aiProcess_GenUVCoords;
  }

  if (options.removeRedundantMaterials) {
    flags |= aiProcess_RemoveRedundantMaterials;
  }

  if (options.limitBoneWeights) {
    flags |= aiProcess_LimitBoneWeights;
  }

  if (options.optimize) {
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

  Assimp::Importer importer;
  const std::string pathStr(path);
  const unsigned int flags = buildAssimpFlags(options);

  NURI_LOG_DEBUG(
      "MeshImporter::loadFromFile: Importing mesh '%s' with flags %u",
      pathStr.c_str(), flags);
  const aiScene *scene = importer.ReadFile(pathStr, flags);
  if (!scene || !scene->HasMeshes()) {
    const std::string error =
        scene ? "Assimp scene has no meshes" : importer.GetErrorString();
    NURI_LOG_WARNING(
        "MeshImporter::loadFromFile: Failed to import mesh '%s': %s",
        pathStr.c_str(), error.c_str());
    return nuri::Result<MeshData, std::string>::makeError(error);
  }

  NURI_LOG_DEBUG(
      "MeshImporter::loadFromFile: Imported model '%s' with %u meshes",
      pathStr.c_str(), scene->mNumMeshes);
  std::pmr::vector<AssimpMeshInstance> meshInstances =
      collectMeshInstances(*scene, mem);
  std::pmr::vector<GltfMeshInstanceScale> gltfInstanceScales(mem);
  auto gltfScaleResult = loadGltfMeshInstanceScales(path, mem);
  if (gltfScaleResult.hasError()) {
    NURI_LOG_WARNING("MeshImporter::loadFromFile: Failed to load glTF node "
                     "scales for '%s': %s",
                     pathStr.c_str(), gltfScaleResult.error().c_str());
  } else {
    gltfInstanceScales = std::move(gltfScaleResult.value());
  }
  if (meshInstances.empty()) {
    NURI_LOG_WARNING("MeshImporter::loadFromFile: Scene '%s' has no mesh "
                     "instances after node traversal",
                     pathStr.c_str());
    return nuri::Result<MeshData, std::string>::makeError(
        "Assimp scene has no mesh instances");
  }

  MeshData data(mem);
  const uint32_t requestedLodCount = clampLodCount(options);

  size_t totalVertices = 0;
  size_t totalIndices = 0;
  for (size_t instanceOrdinal = 0; instanceOrdinal < meshInstances.size();
       ++instanceOrdinal) {
    const AssimpMeshInstance &instance = meshInstances[instanceOrdinal];
    const aiMesh *mesh = instance.mesh;
    if (!mesh) {
      continue;
    }

    totalVertices += mesh->mNumVertices;
    const size_t baseIndexCount = meshIndexCount(*mesh);
    totalIndices += baseIndexCount;
    if (options.generateLods && requestedLodCount > 1) {
      for (uint32_t lodIndex = 1; lodIndex < requestedLodCount; ++lodIndex) {
        totalIndices += targetLodIndexCount(options, lodIndex, baseIndexCount);
      }
    }
  }

  data.vertices.reserve(totalVertices);
  data.indices.reserve(totalIndices);
  data.submeshes.reserve(meshInstances.size());

  if (scene->mName.length > 0) {
    data.name.assign(scene->mName.C_Str(), scene->mName.length);
  } else {
    const std::string stem = std::filesystem::path(pathStr).stem().string();
    data.name.assign(stem.data(), stem.size());
  }

  ScratchArena scratch(mem);
  size_t insufficientGeometryMeshCount = 0;
  std::array<uint32_t, 8> insufficientGeometryMeshSamples{};
  size_t insufficientGeometrySampleCount = 0;

  NURI_LOG_DEBUG("MeshImporter::loadFromFile: Mesh optimization processing");
  for (size_t instanceOrdinal = 0; instanceOrdinal < meshInstances.size();
       ++instanceOrdinal) {
    const AssimpMeshInstance &instance = meshInstances[instanceOrdinal];
    const aiMesh *mesh = instance.mesh;
    if (!mesh) {
      continue;
    }

    ScopedScratch scopedScratch(scratch);
    std::pmr::vector<Vertex> meshVertices(scopedScratch.resource());
    std::pmr::vector<uint32_t> lod0Indices(scopedScratch.resource());
    std::array<std::pmr::vector<uint32_t>, Submesh::kMaxLodCount>
        lodIndexBuffers = makeLodIndexBuffers(scopedScratch.resource());
    std::array<float, Submesh::kMaxLodCount> lodErrors{};

    extractMeshGeometry(*mesh, instance.transform, meshVertices, lod0Indices);

    if (meshVertices.empty() || lod0Indices.size() < kTriangleIndexCount) {
      ++insufficientGeometryMeshCount;
      if (insufficientGeometrySampleCount <
          insufficientGeometryMeshSamples.size()) {
        insufficientGeometryMeshSamples[insufficientGeometrySampleCount++] =
            instance.meshIndex;
      }
      continue;
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
        options, requestedLodCount, instance.meshIndex, meshVertices,
        options.optimize, lodIndexBuffers, lodErrors);

    if (options.optimize) {
      optimizeVertexFetchForAllLods(meshVertices, generatedLodCount,
                                    lodIndexBuffers);
    }

    const BoundingBox submeshBounds = computeSubmeshBounds(meshVertices);
    const glm::vec3 authoredScale =
        resolveAuthoredScale(instance, gltfInstanceScales, instanceOrdinal);
    appendSubmeshToMeshData(data, *mesh, meshVertices, submeshBounds,
                            authoredScale, generatedLodCount, lodIndexBuffers,
                            lodErrors, instance.meshIndex);
  }
  NURI_LOG_DEBUG(
      "MeshImporter::loadFromFile: Mesh optimization processing complete");

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

nuri::Result<ImportedMaterialSet, std::string>
MeshImporter::loadMaterialInfoFromFile(std::string_view path) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (path.empty()) {
    return nuri::Result<ImportedMaterialSet, std::string>::makeError(
        "MeshImporter::loadMaterialInfoFromFile: path is empty");
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
      fallback.name = "material_" + std::to_string(materialIndex);
      set.materials.push_back(std::move(fallback));
      continue;
    }

    ImportedMaterialInfo parsed = parseMaterial(*material, modelPath);
    if (parsed.name.empty()) {
      parsed.name = "material_" + std::to_string(materialIndex);
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

} // namespace nuri
