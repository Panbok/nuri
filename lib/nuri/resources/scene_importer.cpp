#include "nuri/pch.h"

#include "nuri/resources/scene_importer.h"

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/math/light.h"
#include "nuri/resources/detail/gltf_buffer_utils.h"
#include "nuri/resources/detail/gltf_json_utils.h"
#include "nuri/resources/detail/scene_asset_build_backend.h"
#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace nuri {
namespace {

struct ParsedLightDef {
  LightDesc desc{};
  std::string name{};
};

struct ParsedNode {
  std::string name{};
  glm::mat4 localMatrix{1.0f};
  std::vector<uint32_t> children{};
  std::vector<float> morphWeights{};
  std::optional<uint32_t> lightIndex{};
  std::optional<uint32_t> meshIndex{};
  std::optional<uint32_t> skinIndex{};
};

struct ParsedAnimationSamplerSource {
  uint32_t inputAccessor = std::numeric_limits<uint32_t>::max();
  uint32_t outputAccessor = std::numeric_limits<uint32_t>::max();
  AnimationInterpolation interpolation = AnimationInterpolation::Linear;
};

[[nodiscard]] Result<uint32_t, std::string>
resolveGltfMeshMorphTargetCount(yyjson_val *root, uint32_t meshIndex);
[[nodiscard]] Result<std::pmr::vector<SkinData>, std::string>
parseSkinDefinitions(yyjson_val *root,
                     std::span<const std::pmr::vector<std::byte>> buffers,
                     std::pmr::memory_resource *memory);
[[nodiscard]] Result<std::pmr::vector<AnimationClipData>, std::string>
parseAnimationDefinitions(yyjson_val *root,
                          std::span<const ParsedNode> parsedNodes,
                          std::span<const std::pmr::vector<std::byte>> buffers,
                          std::pmr::memory_resource *memory);

[[nodiscard]] glm::mat4 aiMatrix4x4ToMat4(const aiMatrix4x4 &matrix) {
  glm::mat4 out(1.0f);
  out[0][0] = matrix.a1;
  out[1][0] = matrix.a2;
  out[2][0] = matrix.a3;
  out[3][0] = matrix.a4;
  out[0][1] = matrix.b1;
  out[1][1] = matrix.b2;
  out[2][1] = matrix.b3;
  out[3][1] = matrix.b4;
  out[0][2] = matrix.c1;
  out[1][2] = matrix.c2;
  out[2][2] = matrix.c3;
  out[3][2] = matrix.c4;
  out[0][3] = matrix.d1;
  out[1][3] = matrix.d2;
  out[2][3] = matrix.d3;
  out[3][3] = matrix.d4;
  return out;
}

[[nodiscard]] std::string makeFallbackLightName(uint32_t lightIndex) {
  return "light_" + std::to_string(lightIndex);
}

[[nodiscard]] std::string makeFallbackMeshName(uint32_t meshIndex) {
  return "mesh_" + std::to_string(meshIndex);
}

[[nodiscard]] std::string makeFallbackMaterialName(uint32_t materialIndex) {
  return "material_" + std::to_string(materialIndex);
}

[[nodiscard]] bool hasRenderableTriangleGeometry(const aiMesh &mesh);

[[nodiscard]] ImportedSceneMeshAsset
makeImportedMeshAsset(const aiScene &scene, uint32_t sourceSceneMeshIndex,
                      std::string_view sourceNameOverride = {}) {
  ImportedSceneMeshAsset meshAsset{};
  meshAsset.sourceSceneMeshIndex = sourceSceneMeshIndex;
  if (!sourceNameOverride.empty()) {
    meshAsset.sourceName.assign(sourceNameOverride);
    return meshAsset;
  }
  if (sourceSceneMeshIndex < scene.mNumMeshes &&
      scene.mMeshes[sourceSceneMeshIndex] != nullptr &&
      scene.mMeshes[sourceSceneMeshIndex]->mName.length > 0u) {
    meshAsset.sourceName.assign(
        scene.mMeshes[sourceSceneMeshIndex]->mName.C_Str(),
        scene.mMeshes[sourceSceneMeshIndex]->mName.length);
    return meshAsset;
  }
  meshAsset.sourceName = makeFallbackMeshName(sourceSceneMeshIndex);
  return meshAsset;
}

[[nodiscard]] ImportedSceneMaterialAsset
makeImportedMaterialAsset(const aiScene &scene, uint32_t sourceMaterialIndex,
                          std::string_view sourceNameOverride = {}) {
  ImportedSceneMaterialAsset materialAsset{};
  materialAsset.sourceMaterialIndex = sourceMaterialIndex;
  if (!sourceNameOverride.empty()) {
    materialAsset.sourceName.assign(sourceNameOverride);
    return materialAsset;
  }
  if (sourceMaterialIndex < scene.mNumMaterials &&
      scene.mMaterials[sourceMaterialIndex] != nullptr) {
    aiString materialName;
    if (scene.mMaterials[sourceMaterialIndex]->Get(
            AI_MATKEY_NAME, materialName) == aiReturn_SUCCESS &&
        materialName.length > 0u) {
      materialAsset.sourceName.assign(materialName.C_Str(),
                                      materialName.length);
    }
  }
  if (materialAsset.sourceName.empty()) {
    materialAsset.sourceName = makeFallbackMaterialName(sourceMaterialIndex);
  }
  return materialAsset;
}

[[nodiscard]] uint32_t
ensureImportedMeshAsset(ImportedScene &imported, const aiScene &scene,
                        HashMap<uint32_t, uint32_t> &meshOrdinalToAsset,
                        uint32_t sourceSceneMeshIndex,
                        std::string_view sourceNameOverride = {}) {
  if (auto it = meshOrdinalToAsset.find(sourceSceneMeshIndex);
      it != meshOrdinalToAsset.end()) {
    return it->second;
  }

  const uint32_t meshAssetIndex =
      static_cast<uint32_t>(imported.meshAssets.size());
  imported.meshAssets.push_back(
      makeImportedMeshAsset(scene, sourceSceneMeshIndex, sourceNameOverride));
  meshOrdinalToAsset.emplace(sourceSceneMeshIndex, meshAssetIndex);
  return meshAssetIndex;
}

[[nodiscard]] uint32_t
ensureImportedMaterialAsset(ImportedScene &imported, const aiScene &scene,
                            HashMap<uint32_t, uint32_t> &materialOrdinalToAsset,
                            uint32_t sourceMaterialIndex,
                            std::string_view sourceNameOverride = {}) {
  if (auto it = materialOrdinalToAsset.find(sourceMaterialIndex);
      it != materialOrdinalToAsset.end()) {
    return it->second;
  }

  const uint32_t materialAssetIndex =
      static_cast<uint32_t>(imported.materialAssets.size());
  imported.materialAssets.push_back(makeImportedMaterialAsset(
      scene, sourceMaterialIndex, sourceNameOverride));
  materialOrdinalToAsset.emplace(sourceMaterialIndex, materialAssetIndex);
  return materialAssetIndex;
}

void appendRemainingSceneAssets(
    ImportedScene &imported, const aiScene &scene,
    HashMap<uint32_t, uint32_t> &meshOrdinalToAsset,
    HashMap<uint32_t, uint32_t> &materialOrdinalToAsset,
    uint32_t materialCountLimit = std::numeric_limits<uint32_t>::max()) {
  for (uint32_t sourceSceneMeshIndex = 0u;
       sourceSceneMeshIndex < scene.mNumMeshes; ++sourceSceneMeshIndex) {
    if (scene.mMeshes[sourceSceneMeshIndex] == nullptr ||
        !hasRenderableTriangleGeometry(*scene.mMeshes[sourceSceneMeshIndex]) ||
        meshOrdinalToAsset.contains(sourceSceneMeshIndex)) {
      continue;
    }
    (void)ensureImportedMeshAsset(imported, scene, meshOrdinalToAsset,
                                  sourceSceneMeshIndex);
  }

  const uint32_t materialEnd =
      std::min(scene.mNumMaterials, materialCountLimit);
  for (uint32_t sourceMaterialIndex = 0u; sourceMaterialIndex < materialEnd;
       ++sourceMaterialIndex) {
    (void)ensureImportedMaterialAsset(imported, scene, materialOrdinalToAsset,
                                      sourceMaterialIndex);
  }
}

[[nodiscard]] uint32_t resolveGltfPrimitiveMaterialIndex(
    const detail::GltfPrimitiveMaterialMapping &mapping,
    uint32_t sourceSceneMeshIndex, uint32_t fallbackMaterialIndex) {
  if (!mapping.sceneMeshIndicesAreFlatPrimitiveOrder ||
      sourceSceneMeshIndex >= mapping.primitiveMaterialIndices.size()) {
    return fallbackMaterialIndex;
  }
  const uint32_t sourceMaterialIndex =
      mapping.primitiveMaterialIndices[sourceSceneMeshIndex];
  return sourceMaterialIndex == std::numeric_limits<uint32_t>::max()
             ? fallbackMaterialIndex
             : sourceMaterialIndex;
}

[[nodiscard]] uint32_t resolveGltfSinglePrimitiveMeshMaterialIndex(
    const detail::GltfPrimitiveMaterialMapping &mapping, uint32_t meshIndex,
    uint32_t fallbackMaterialIndex) {
  if (meshIndex >= mapping.singlePrimitiveMeshMaterialIndices.size()) {
    return fallbackMaterialIndex;
  }
  const uint32_t sourceMaterialIndex =
      mapping.singlePrimitiveMeshMaterialIndices[meshIndex];
  return sourceMaterialIndex == std::numeric_limits<uint32_t>::max()
             ? fallbackMaterialIndex
             : sourceMaterialIndex;
}

[[nodiscard]] bool hasRenderableTriangleGeometry(const aiMesh &mesh) {
  if (mesh.mNumVertices == 0u || mesh.mNumFaces == 0u) {
    return false;
  }
  if ((mesh.mPrimitiveTypes & aiPrimitiveType_TRIANGLE) == 0u) {
    return false;
  }
  for (uint32_t faceIndex = 0u; faceIndex < mesh.mNumFaces; ++faceIndex) {
    if (mesh.mFaces[faceIndex].mNumIndices >= 3u) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] Result<LightType, std::string>
parseLightType(std::string_view typeName) {
  if (typeName == "directional") {
    return Result<LightType, std::string>::makeResult(LightType::Directional);
  }
  if (typeName == "point") {
    return Result<LightType, std::string>::makeResult(LightType::Point);
  }
  if (typeName == "spot") {
    return Result<LightType, std::string>::makeResult(LightType::Spot);
  }
  return Result<LightType, std::string>::makeError(
      "glTF punctual light type is invalid");
}

[[nodiscard]] float sanitizeNonNegative(float value, float fallback = 0.0f) {
  if (!std::isfinite(value)) {
    return fallback;
  }
  return std::max(value, 0.0f);
}

[[nodiscard]] glm::quat sanitizeRotation(const glm::quat &rotation) {
  const float length = glm::length(rotation);
  if (!std::isfinite(length) || length <= kLightBasisMinLength) {
    return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }
  return glm::normalize(rotation);
}

[[nodiscard]] LightDesc sanitizeImportedLightDesc(const LightDesc &desc) {
  LightDesc sanitized = desc;
  sanitized.color = glm::max(desc.color, glm::vec3(0.0f));
  sanitized.intensity = sanitizeNonNegative(desc.intensity, 1.0f);
  sanitized.range = sanitizeNonNegative(desc.range, 0.0f);
  sanitized.innerConeAngleRadians =
      sanitizeNonNegative(desc.innerConeAngleRadians, 0.0f);
  sanitized.outerConeAngleRadians =
      sanitizeNonNegative(desc.outerConeAngleRadians, glm::quarter_pi<float>());

  if (sanitized.type == LightType::Directional) {
    sanitized.range = 0.0f;
    sanitized.innerConeAngleRadians = 0.0f;
    sanitized.outerConeAngleRadians = 0.0f;
  } else if (sanitized.type == LightType::Point) {
    sanitized.innerConeAngleRadians = 0.0f;
    sanitized.outerConeAngleRadians = 0.0f;
  } else {
    sanitized.outerConeAngleRadians = std::clamp(
        sanitized.outerConeAngleRadians, 0.0f, glm::half_pi<float>() - 1.0e-4f);
    sanitized.innerConeAngleRadians = std::clamp(
        sanitized.innerConeAngleRadians, 0.0f, sanitized.outerConeAngleRadians);
  }

  sanitized.position = glm::vec3(0.0f);
  sanitized.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  return sanitized;
}

[[nodiscard]] Result<uint32_t, std::string>
resolveGltfMeshMorphTargetCount(yyjson_val *root, uint32_t meshIndex) {
  yyjson_val *meshesValue = yyjson_obj_get(root, "meshes");
  if (!yyjson_is_arr(meshesValue) ||
      meshIndex >= yyjson_arr_size(meshesValue)) {
    return Result<uint32_t, std::string>::makeError(
        "glTF mesh index is out of range");
  }
  yyjson_val *meshValue = yyjson_arr_get(meshesValue, meshIndex);
  yyjson_val *primitivesValue = yyjson_obj_get(meshValue, "primitives");
  if (!yyjson_is_arr(primitivesValue)) {
    return Result<uint32_t, std::string>::makeResult(0u);
  }

  uint32_t morphTargetCount = 0u;
  bool initialized = false;
  yyjson_arr_iter primitivesIter = yyjson_arr_iter_with(primitivesValue);
  yyjson_val *primitiveValue = nullptr;
  while ((primitiveValue = yyjson_arr_iter_next(&primitivesIter)) != nullptr) {
    yyjson_val *targetsValue = yyjson_obj_get(primitiveValue, "targets");
    const uint32_t primitiveTargetCount =
        yyjson_is_arr(targetsValue)
            ? static_cast<uint32_t>(yyjson_arr_size(targetsValue))
            : 0u;
    if (!initialized) {
      morphTargetCount = primitiveTargetCount;
      initialized = true;
      continue;
    }
    if (primitiveTargetCount != morphTargetCount) {
      return Result<uint32_t, std::string>::makeError(
          "glTF mesh primitives attached to the same node have inconsistent "
          "morph target counts");
    }
  }
  return Result<uint32_t, std::string>::makeResult(morphTargetCount);
}

[[nodiscard]] std::vector<float> readJsonFloatArray(yyjson_val *value) {
  std::vector<float> out;
  if (!yyjson_is_arr(value)) {
    return out;
  }
  out.reserve(yyjson_arr_size(value));
  yyjson_arr_iter iter = yyjson_arr_iter_with(value);
  yyjson_val *element = nullptr;
  while ((element = yyjson_arr_iter_next(&iter)) != nullptr) {
    float component = 0.0f;
    if (!detail::tryReadJsonFloat(element, component)) {
      out.clear();
      return out;
    }
    out.push_back(component);
  }
  return out;
}

template <typename ResolveImportedNodeIndexFn>
[[nodiscard]] Result<bool, std::string> attachParsedLightsToImportedNodes(
    ImportedScene &imported, std::span<const ParsedNode> parsedNodes,
    std::span<const ParsedLightDef> parsedLights,
    ResolveImportedNodeIndexFn &&resolveImportedNodeIndexFn) {
  for (uint32_t parsedNodeIndex = 0u; parsedNodeIndex < parsedNodes.size();
       ++parsedNodeIndex) {
    const ParsedNode &parsedNode = parsedNodes[parsedNodeIndex];
    if (!parsedNode.lightIndex.has_value()) {
      continue;
    }
    if (*parsedNode.lightIndex >= parsedLights.size()) {
      return Result<bool, std::string>::makeError(
          "glTF punctual light index is out of range");
    }

    const uint32_t importedNodeIndex =
        resolveImportedNodeIndexFn(parsedNode, parsedNodeIndex);
    if (importedNodeIndex == kInvalidScenePrefabIndex) {
      continue;
    }

    ImportedSceneLight light{};
    light.nodeIndex = importedNodeIndex;
    light.light = parsedLights[*parsedNode.lightIndex].desc;
    light.sourceName = !parsedNode.name.empty()
                           ? parsedNode.name
                           : parsedLights[*parsedNode.lightIndex].name;
    if (light.light.name.empty()) {
      light.light.name = light.sourceName;
    }
    light.sourceNodeIndex = parsedNodeIndex;
    imported.lights.push_back(std::move(light));
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] glm::mat4 parseNodeLocalMatrix(yyjson_val *nodeValue) {
  glm::mat4 matrix(1.0f);
  if (detail::tryReadJsonMat4(yyjson_obj_get(nodeValue, "matrix"), matrix)) {
    return matrix;
  }

  glm::vec3 translation(0.0f);
  glm::vec4 rotationValues(0.0f, 0.0f, 0.0f, 1.0f);
  glm::vec3 scale(1.0f);
  (void)detail::tryReadJsonVec3(yyjson_obj_get(nodeValue, "translation"),
                                translation);
  (void)detail::tryReadJsonVec4(yyjson_obj_get(nodeValue, "rotation"),
                                rotationValues);
  (void)detail::tryReadJsonVec3(yyjson_obj_get(nodeValue, "scale"), scale);

  const glm::quat rotation = sanitizeRotation(glm::quat(
      rotationValues.w, rotationValues.x, rotationValues.y, rotationValues.z));
  return glm::translate(glm::mat4(1.0f), translation) *
         glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

[[nodiscard]] bool matrixNearlyEqual(const glm::mat4 &lhs, const glm::mat4 &rhs,
                                     float epsilon = 1.0e-4f) {
  for (uint32_t column = 0; column < 4u; ++column) {
    for (uint32_t row = 0; row < 4u; ++row) {
      if (std::abs(lhs[column][row] - rhs[column][row]) > epsilon) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] std::string makeNodePath(std::string_view parentPath,
                                       std::string_view nodeName,
                                       uint32_t siblingOrdinal) {
  std::string component = nodeName.empty()
                              ? ("#" + std::to_string(siblingOrdinal))
                              : std::string(nodeName);
  if (parentPath.empty()) {
    return component;
  }
  return std::string(parentPath) + "/" + component;
}

[[nodiscard]] std::string stripLeadingPathComponent(std::string_view path) {
  const size_t slash = path.find('/');
  if (slash == std::string_view::npos || slash + 1u >= path.size()) {
    return {};
  }
  return std::string(path.substr(slash + 1u));
}

[[nodiscard]] uint32_t resolveImportedNodeIndex(
    const ParsedNode &parsedNode, uint32_t parsedNodeIndex,
    std::span<const std::string> parsedPaths,
    const HashMap<std::string, uint32_t> &importedPathToNode,
    const HashMap<std::string, std::vector<uint32_t>> &importedNameToNodes,
    std::span<const ImportedSceneNode> importedNodes);

void remapSkinAndAnimationNodeIndices(
    std::span<SkinData> skins, std::span<AnimationClipData> animations,
    std::span<const uint32_t> parsedToImportedNodeIndex) {
  for (SkinData &skin : skins) {
    if (skin.skeletonRootNodeIndex < parsedToImportedNodeIndex.size()) {
      skin.skeletonRootNodeIndex =
          parsedToImportedNodeIndex[skin.skeletonRootNodeIndex];
    } else {
      skin.skeletonRootNodeIndex = kInvalidScenePrefabIndex;
    }
    for (uint32_t &jointNodeIndex : skin.jointNodeIndices) {
      if (jointNodeIndex < parsedToImportedNodeIndex.size()) {
        jointNodeIndex = parsedToImportedNodeIndex[jointNodeIndex];
      } else {
        jointNodeIndex = kInvalidScenePrefabIndex;
      }
    }
  }

  for (AnimationClipData &clip : animations) {
    for (AnimationChannelData &channel : clip.channels) {
      if (channel.targetNodeIndex < parsedToImportedNodeIndex.size()) {
        channel.targetNodeIndex =
            parsedToImportedNodeIndex[channel.targetNodeIndex];
      } else {
        channel.targetNodeIndex = kInvalidScenePrefabIndex;
      }
    }
  }
}

Result<std::vector<ParsedLightDef>, std::string>
parseLightDefinitions(yyjson_val *root) {
  yyjson_val *extensionsValue = yyjson_obj_get(root, "extensions");
  if (!yyjson_is_obj(extensionsValue)) {
    return Result<std::vector<ParsedLightDef>, std::string>::makeResult({});
  }

  yyjson_val *lightExtension =
      yyjson_obj_get(extensionsValue, "KHR_lights_punctual");
  if (!yyjson_is_obj(lightExtension)) {
    return Result<std::vector<ParsedLightDef>, std::string>::makeResult({});
  }

  yyjson_val *lightsValue = yyjson_obj_get(lightExtension, "lights");
  if (!yyjson_is_arr(lightsValue)) {
    return Result<std::vector<ParsedLightDef>, std::string>::makeResult({});
  }

  std::vector<ParsedLightDef> lights;
  lights.reserve(yyjson_arr_size(lightsValue));
  yyjson_arr_iter iter = yyjson_arr_iter_with(lightsValue);
  yyjson_val *lightValue = nullptr;
  uint32_t lightIndex = 0u;
  while ((lightValue = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(lightValue)) {
      return Result<std::vector<ParsedLightDef>, std::string>::makeError(
          "glTF light entry is not an object");
    }

    ParsedLightDef parsed{};
    auto lightTypeResult =
        parseLightType(detail::readJsonStringView(lightValue, "type"));
    if (lightTypeResult.hasError()) {
      return Result<std::vector<ParsedLightDef>, std::string>::makeError(
          lightTypeResult.error());
    }

    parsed.desc.type = lightTypeResult.value();
    parsed.desc.color = glm::vec3(1.0f);
    parsed.desc.intensity = 1.0f;
    parsed.desc.range = 0.0f;
    parsed.desc.innerConeAngleRadians = 0.0f;
    parsed.desc.outerConeAngleRadians = glm::quarter_pi<float>();
    parsed.desc.enabled = true;
    const std::string_view lightName =
        detail::readJsonStringView(lightValue, "name");
    parsed.name = lightName.empty() ? makeFallbackLightName(lightIndex)
                                    : std::string(lightName);

    (void)detail::tryReadJsonVec3(yyjson_obj_get(lightValue, "color"),
                                  parsed.desc.color);
    (void)detail::tryReadJsonFloat(yyjson_obj_get(lightValue, "intensity"),
                                   parsed.desc.intensity);
    (void)detail::tryReadJsonFloat(yyjson_obj_get(lightValue, "range"),
                                   parsed.desc.range);

    if (parsed.desc.type == LightType::Spot) {
      yyjson_val *spotValue = yyjson_obj_get(lightValue, "spot");
      if (yyjson_is_obj(spotValue)) {
        (void)detail::tryReadJsonFloat(
            yyjson_obj_get(spotValue, "innerConeAngle"),
            parsed.desc.innerConeAngleRadians);
        (void)detail::tryReadJsonFloat(
            yyjson_obj_get(spotValue, "outerConeAngle"),
            parsed.desc.outerConeAngleRadians);
      }
    }

    parsed.desc = sanitizeImportedLightDesc(parsed.desc);
    lights.push_back(std::move(parsed));
    ++lightIndex;
  }

  return Result<std::vector<ParsedLightDef>, std::string>::makeResult(
      std::move(lights));
}

Result<std::vector<ParsedNode>, std::string> parseNodes(yyjson_val *root) {
  yyjson_val *nodesValue = yyjson_obj_get(root, "nodes");
  if (!yyjson_is_arr(nodesValue)) {
    return Result<std::vector<ParsedNode>, std::string>::makeResult({});
  }

  std::vector<ParsedNode> nodes;
  nodes.reserve(yyjson_arr_size(nodesValue));
  yyjson_arr_iter iter = yyjson_arr_iter_with(nodesValue);
  yyjson_val *nodeValue = nullptr;
  while ((nodeValue = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(nodeValue)) {
      return Result<std::vector<ParsedNode>, std::string>::makeError(
          "glTF node entry is not an object");
    }

    ParsedNode parsed{};
    const std::string_view nodeName =
        detail::readJsonStringView(nodeValue, "name");
    if (!nodeName.empty()) {
      parsed.name.assign(nodeName);
    }
    parsed.localMatrix = parseNodeLocalMatrix(nodeValue);

    uint32_t meshIndex = 0u;
    if (detail::tryReadJsonUint32(yyjson_obj_get(nodeValue, "mesh"),
                                  meshIndex)) {
      auto morphCountResult = resolveGltfMeshMorphTargetCount(root, meshIndex);
      if (morphCountResult.hasError()) {
        return Result<std::vector<ParsedNode>, std::string>::makeError(
            morphCountResult.error());
      }
      parsed.meshIndex = meshIndex;
      parsed.morphWeights =
          readJsonFloatArray(yyjson_obj_get(nodeValue, "weights"));
      if (parsed.morphWeights.empty()) {
        yyjson_val *meshesValue = yyjson_obj_get(root, "meshes");
        if (yyjson_is_arr(meshesValue) &&
            meshIndex < yyjson_arr_size(meshesValue)) {
          parsed.morphWeights = readJsonFloatArray(yyjson_obj_get(
              yyjson_arr_get(meshesValue, meshIndex), "weights"));
        }
      }
      if (parsed.morphWeights.empty() && morphCountResult.value() > 0u) {
        parsed.morphWeights.resize(morphCountResult.value(), 0.0f);
      }
      if (!parsed.morphWeights.empty() &&
          parsed.morphWeights.size() != morphCountResult.value()) {
        return Result<std::vector<ParsedNode>, std::string>::makeError(
            "glTF node morph weight count does not match mesh morph target "
            "count");
      }
    } else {
      parsed.morphWeights =
          readJsonFloatArray(yyjson_obj_get(nodeValue, "weights"));
      if (!parsed.morphWeights.empty()) {
        return Result<std::vector<ParsedNode>, std::string>::makeError(
            "glTF node weights require a mesh");
      }
    }

    uint32_t skinIndex = 0u;
    if (detail::tryReadJsonUint32(yyjson_obj_get(nodeValue, "skin"),
                                  skinIndex)) {
      parsed.skinIndex = skinIndex;
    }

    yyjson_val *childrenValue = yyjson_obj_get(nodeValue, "children");
    if (yyjson_is_arr(childrenValue)) {
      parsed.children.reserve(yyjson_arr_size(childrenValue));
      yyjson_arr_iter childrenIter = yyjson_arr_iter_with(childrenValue);
      yyjson_val *childValue = nullptr;
      while ((childValue = yyjson_arr_iter_next(&childrenIter)) != nullptr) {
        uint32_t childIndex = 0u;
        if (!detail::tryReadJsonUint32(childValue, childIndex)) {
          return Result<std::vector<ParsedNode>, std::string>::makeError(
              "glTF node child index is invalid");
        }
        parsed.children.push_back(childIndex);
      }
    }

    yyjson_val *extensionsValue = yyjson_obj_get(nodeValue, "extensions");
    if (yyjson_is_obj(extensionsValue)) {
      yyjson_val *lightExtension =
          yyjson_obj_get(extensionsValue, "KHR_lights_punctual");
      if (yyjson_is_obj(lightExtension)) {
        uint32_t lightIndex = 0u;
        if (detail::tryReadJsonUint32(yyjson_obj_get(lightExtension, "light"),
                                      lightIndex)) {
          parsed.lightIndex = lightIndex;
        }
      }
    }

    nodes.push_back(std::move(parsed));
  }

  return Result<std::vector<ParsedNode>, std::string>::makeResult(
      std::move(nodes));
}

[[nodiscard]] Result<std::pmr::vector<SkinData>, std::string>
parseSkinDefinitions(yyjson_val *root,
                     std::span<const std::pmr::vector<std::byte>> buffers,
                     std::pmr::memory_resource *memory) {
  std::pmr::vector<SkinData> skins(memory);
  yyjson_val *skinsValue = yyjson_obj_get(root, "skins");
  if (!yyjson_is_arr(skinsValue)) {
    return Result<std::pmr::vector<SkinData>, std::string>::makeResult(
        std::move(skins));
  }

  skins.reserve(yyjson_arr_size(skinsValue));
  yyjson_arr_iter iter = yyjson_arr_iter_with(skinsValue);
  yyjson_val *skinValue = nullptr;
  while ((skinValue = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(skinValue)) {
      return Result<std::pmr::vector<SkinData>, std::string>::makeError(
          "glTF skin entry is invalid");
    }
    skins.emplace_back(memory);
    SkinData &skin = skins.back();
    const std::string_view name = detail::readJsonStringView(skinValue, "name");
    skin.name.assign(name.data(), name.size());

    (void)detail::tryReadJsonUint32(yyjson_obj_get(skinValue, "skeleton"),
                                    skin.skeletonRootNodeIndex);

    yyjson_val *jointsValue = yyjson_obj_get(skinValue, "joints");
    if (!yyjson_is_arr(jointsValue) || yyjson_arr_size(jointsValue) == 0u) {
      return Result<std::pmr::vector<SkinData>, std::string>::makeError(
          "glTF skin joints array is invalid");
    }
    skin.jointNodeIndices.reserve(yyjson_arr_size(jointsValue));
    yyjson_arr_iter jointsIter = yyjson_arr_iter_with(jointsValue);
    yyjson_val *jointValue = nullptr;
    while ((jointValue = yyjson_arr_iter_next(&jointsIter)) != nullptr) {
      uint32_t jointIndex = 0u;
      if (!detail::tryReadJsonUint32(jointValue, jointIndex)) {
        return Result<std::pmr::vector<SkinData>, std::string>::makeError(
            "glTF skin joint index is invalid");
      }
      skin.jointNodeIndices.push_back(jointIndex);
    }

    uint32_t ibmAccessor = std::numeric_limits<uint32_t>::max();
    if (detail::tryReadJsonUint32(
            yyjson_obj_get(skinValue, "inverseBindMatrices"), ibmAccessor)) {
      auto matricesResult = detail::readGltfAccessorAsMat4Array(
          root, buffers, ibmAccessor, memory);
      if (matricesResult.hasError()) {
        return Result<std::pmr::vector<SkinData>, std::string>::makeError(
            matricesResult.error());
      }
      skin.inverseBindMatrices = std::move(matricesResult.value());
    } else {
      skin.inverseBindMatrices.resize(skin.jointNodeIndices.size(),
                                      glm::mat4(1.0f));
    }

    if (skin.inverseBindMatrices.size() != skin.jointNodeIndices.size()) {
      return Result<std::pmr::vector<SkinData>, std::string>::makeError(
          "glTF skin inverse bind matrix count does not match joint count");
    }
  }

  return Result<std::pmr::vector<SkinData>, std::string>::makeResult(
      std::move(skins));
}

[[nodiscard]] Result<AnimationTargetPath, std::string>
parseAnimationTargetPath(std::string_view path) {
  if (path == "translation") {
    return Result<AnimationTargetPath, std::string>::makeResult(
        AnimationTargetPath::Translation);
  }
  if (path == "rotation") {
    return Result<AnimationTargetPath, std::string>::makeResult(
        AnimationTargetPath::Rotation);
  }
  if (path == "scale") {
    return Result<AnimationTargetPath, std::string>::makeResult(
        AnimationTargetPath::Scale);
  }
  if (path == "weights") {
    return Result<AnimationTargetPath, std::string>::makeResult(
        AnimationTargetPath::Weights);
  }
  return Result<AnimationTargetPath, std::string>::makeError(
      "glTF animation target path is unsupported");
}

[[nodiscard]] AnimationInterpolation
parseAnimationInterpolation(std::string_view interpolation) {
  if (interpolation == "STEP") {
    return AnimationInterpolation::Step;
  }
  if (interpolation == "CUBICSPLINE") {
    return AnimationInterpolation::CubicSpline;
  }
  return AnimationInterpolation::Linear;
}

[[nodiscard]] Result<std::pmr::vector<AnimationClipData>, std::string>
parseAnimationDefinitions(yyjson_val *root,
                          std::span<const ParsedNode> parsedNodes,
                          std::span<const std::pmr::vector<std::byte>> buffers,
                          std::pmr::memory_resource *memory) {
  std::pmr::vector<AnimationClipData> clips(memory);
  yyjson_val *animationsValue = yyjson_obj_get(root, "animations");
  if (!yyjson_is_arr(animationsValue)) {
    return Result<std::pmr::vector<AnimationClipData>, std::string>::makeResult(
        std::move(clips));
  }

  clips.reserve(yyjson_arr_size(animationsValue));
  yyjson_arr_iter animationsIter = yyjson_arr_iter_with(animationsValue);
  yyjson_val *animationValue = nullptr;
  while ((animationValue = yyjson_arr_iter_next(&animationsIter)) != nullptr) {
    if (!yyjson_is_obj(animationValue)) {
      return Result<std::pmr::vector<AnimationClipData>,
                    std::string>::makeError("glTF animation entry is invalid");
    }

    std::pmr::vector<ParsedAnimationSamplerSource> samplerSources(memory);
    yyjson_val *samplersValue = yyjson_obj_get(animationValue, "samplers");
    if (!yyjson_is_arr(samplersValue)) {
      return Result<std::pmr::vector<AnimationClipData>, std::string>::
          makeError("glTF animation samplers array is invalid");
    }
    samplerSources.reserve(yyjson_arr_size(samplersValue));
    yyjson_arr_iter samplersIter = yyjson_arr_iter_with(samplersValue);
    yyjson_val *samplerValue = nullptr;
    while ((samplerValue = yyjson_arr_iter_next(&samplersIter)) != nullptr) {
      ParsedAnimationSamplerSource source{};
      if (!detail::tryReadJsonUint32(yyjson_obj_get(samplerValue, "input"),
                                     source.inputAccessor) ||
          !detail::tryReadJsonUint32(yyjson_obj_get(samplerValue, "output"),
                                     source.outputAccessor)) {
        return Result<std::pmr::vector<AnimationClipData>, std::string>::
            makeError("glTF animation sampler accessors are invalid");
      }
      source.interpolation = parseAnimationInterpolation(
          detail::readJsonStringView(samplerValue, "interpolation"));
      samplerSources.push_back(source);
    }

    clips.emplace_back(memory);
    AnimationClipData &clip = clips.back();
    const std::string_view clipName =
        detail::readJsonStringView(animationValue, "name");
    clip.name.assign(clipName.data(), clipName.size());
    clip.samplers.reserve(samplerSources.size());
    for (size_t samplerIndex = 0; samplerIndex < samplerSources.size();
         ++samplerIndex) {
      clip.samplers.emplace_back(memory);
    }

    yyjson_val *channelsValue = yyjson_obj_get(animationValue, "channels");
    if (!yyjson_is_arr(channelsValue)) {
      return Result<std::pmr::vector<AnimationClipData>, std::string>::
          makeError("glTF animation channels array is invalid");
    }
    clip.channels.reserve(yyjson_arr_size(channelsValue));
    yyjson_arr_iter channelsIter = yyjson_arr_iter_with(channelsValue);
    yyjson_val *channelValue = nullptr;
    while ((channelValue = yyjson_arr_iter_next(&channelsIter)) != nullptr) {
      if (!yyjson_is_obj(channelValue)) {
        return Result<std::pmr::vector<AnimationClipData>, std::string>::
            makeError("glTF animation channel entry is invalid");
      }
      AnimationChannelData channel{};
      if (!detail::tryReadJsonUint32(yyjson_obj_get(channelValue, "sampler"),
                                     channel.samplerIndex) ||
          channel.samplerIndex >= samplerSources.size()) {
        return Result<std::pmr::vector<AnimationClipData>, std::string>::
            makeError("glTF animation channel sampler index is invalid");
      }
      yyjson_val *targetValue = yyjson_obj_get(channelValue, "target");
      if (!yyjson_is_obj(targetValue) ||
          !detail::tryReadJsonUint32(yyjson_obj_get(targetValue, "node"),
                                     channel.targetNodeIndex) ||
          channel.targetNodeIndex >= parsedNodes.size()) {
        return Result<std::pmr::vector<AnimationClipData>, std::string>::
            makeError("glTF animation channel target node is invalid");
      }
      auto pathResult = parseAnimationTargetPath(
          detail::readJsonStringView(targetValue, "path"));
      if (pathResult.hasError()) {
        return Result<std::pmr::vector<AnimationClipData>,
                      std::string>::makeError(pathResult.error());
      }
      channel.path = pathResult.value();
      clip.channels.push_back(channel);
    }

    for (const AnimationChannelData &channel : clip.channels) {
      const ParsedAnimationSamplerSource &source =
          samplerSources[channel.samplerIndex];
      auto inputInfoResult =
          detail::describeGltfAccessor(root, source.inputAccessor);
      if (inputInfoResult.hasError()) {
        return Result<std::pmr::vector<AnimationClipData>,
                      std::string>::makeError(inputInfoResult.error());
      }
      auto outputInfoResult =
          detail::describeGltfAccessor(root, source.outputAccessor);
      if (outputInfoResult.hasError()) {
        return Result<std::pmr::vector<AnimationClipData>,
                      std::string>::makeError(outputInfoResult.error());
      }
      const detail::GltfAccessorInfo inputInfo = inputInfoResult.value();
      const detail::GltfAccessorInfo outputInfo = outputInfoResult.value();
      if (inputInfo.componentType != 5126u || inputInfo.componentCount != 1u) {
        return Result<std::pmr::vector<AnimationClipData>, std::string>::
            makeError("glTF animation input accessor must be float scalar");
      }

      uint32_t valueArity = 0u;
      if (channel.path == AnimationTargetPath::Translation ||
          channel.path == AnimationTargetPath::Scale) {
        if (outputInfo.componentType != 5126u ||
            outputInfo.componentCount != 3u) {
          return Result<std::pmr::vector<AnimationClipData>, std::string>::
              makeError(
                  "glTF translation/scale output accessor must be float VEC3");
        }
        valueArity = 3u;
      } else if (channel.path == AnimationTargetPath::Rotation) {
        if (outputInfo.componentType != 5126u ||
            outputInfo.componentCount != 4u) {
          return Result<std::pmr::vector<AnimationClipData>, std::string>::
              makeError("glTF rotation output accessor must be float VEC4");
        }
        valueArity = 4u;
      } else {
        const uint32_t splineFactor =
            source.interpolation == AnimationInterpolation::CubicSpline ? 3u
                                                                        : 1u;
        const uint64_t totalOutputScalars =
            static_cast<uint64_t>(outputInfo.count) * outputInfo.componentCount;
        const uint64_t totalInputFrames =
            static_cast<uint64_t>(inputInfo.count) * splineFactor;
        if (totalInputFrames == 0u ||
            (totalOutputScalars % totalInputFrames) != 0u) {
          return Result<std::pmr::vector<AnimationClipData>, std::string>::
              makeError("glTF morph weight output accessor shape is invalid");
        }
        valueArity =
            static_cast<uint32_t>(totalOutputScalars / totalInputFrames);
        const ParsedNode &targetNode = parsedNodes[channel.targetNodeIndex];
        if (targetNode.morphWeights.empty()) {
          return Result<std::pmr::vector<AnimationClipData>, std::string>::
              makeError(
                  "glTF weight animation targets a node without morph weights");
        }
        if (valueArity != targetNode.morphWeights.size()) {
          return Result<std::pmr::vector<AnimationClipData>, std::string>::
              makeError("glTF weight animation output arity does not match "
                        "node morph target count");
        }
      }

      AnimationSamplerData &sampler = clip.samplers[channel.samplerIndex];
      if (sampler.valueArity != 0u && sampler.valueArity != valueArity) {
        return Result<std::pmr::vector<AnimationClipData>, std::string>::
            makeError("glTF animation sampler is reused with incompatible "
                      "target arity");
      }
      if (sampler.valueArity == 0u) {
        sampler.valueArity = valueArity;
        sampler.interpolation = source.interpolation;
        auto keyTimesResult = detail::readGltfAccessorAsFloatArray(
            root, buffers, source.inputAccessor, memory);
        if (keyTimesResult.hasError()) {
          return Result<std::pmr::vector<AnimationClipData>,
                        std::string>::makeError(keyTimesResult.error());
        }
        sampler.keyTimes = std::move(keyTimesResult.value());
        auto valuesResult = detail::readGltfAccessorAsFloatArray(
            root, buffers, source.outputAccessor, memory);
        if (valuesResult.hasError()) {
          return Result<std::pmr::vector<AnimationClipData>,
                        std::string>::makeError(valuesResult.error());
        }
        sampler.values = std::move(valuesResult.value());
        if (!sampler.keyTimes.empty()) {
          clip.durationSeconds =
              std::max(clip.durationSeconds, sampler.keyTimes.back());
        }
      }
    }
  }

  return Result<std::pmr::vector<AnimationClipData>, std::string>::makeResult(
      std::move(clips));
}

Result<bool, std::string> rebuildImportedSceneFromParsedGltf(
    ImportedScene &imported, const aiScene &scene,
    std::span<const ParsedNode> parsedNodes,
    std::span<const uint32_t> parsedRoots,
    std::span<const std::string> parsedPaths,
    const HashMap<std::string, uint32_t> &importedPathToNode,
    const HashMap<std::string, std::vector<uint32_t>> &importedNameToNodes,
    std::span<const ParsedLightDef> parsedLights,
    const detail::GltfPrimitiveMaterialMapping &materialMapping,
    std::vector<uint32_t> *outParsedToImportedNodeIndices) {
  if (parsedNodes.empty()) {
    return Result<bool, std::string>::makeResult(false);
  }

  std::vector<uint32_t> parentIndices(parsedNodes.size(),
                                      kInvalidScenePrefabIndex);
  for (uint32_t nodeIndex = 0u; nodeIndex < parsedNodes.size(); ++nodeIndex) {
    for (const uint32_t childIndex : parsedNodes[nodeIndex].children) {
      if (childIndex >= parsedNodes.size()) {
        return Result<bool, std::string>::makeError(
            "glTF node child index is out of range");
      }
      if (parentIndices[childIndex] != kInvalidScenePrefabIndex) {
        return Result<bool, std::string>::makeError(
            "glTF node has multiple parents");
      }
      parentIndices[childIndex] = nodeIndex;
    }
  }

  for (const uint32_t rootIndex : parsedRoots) {
    if (rootIndex >= parsedNodes.size()) {
      return Result<bool, std::string>::makeError(
          "glTF scene root node index is out of range");
    }
    parentIndices[rootIndex] = kInvalidScenePrefabIndex;
  }

  std::vector<uint32_t> remappedNodeIndices(parsedNodes.size(),
                                            kInvalidScenePrefabIndex);
  std::vector<uint32_t> orderedParsedNodeIndices;
  orderedParsedNodeIndices.reserve(parsedNodes.size());

  struct AppendFrame {
    uint32_t nodeIndex = kInvalidScenePrefabIndex;
    bool processChildren = false;
  };
  std::vector<uint8_t> appendActive(parsedNodes.size(), 0u);
  for (const uint32_t rootIndex : parsedRoots) {
    std::vector<AppendFrame> stack;
    stack.push_back({rootIndex, false});
    while (!stack.empty()) {
      const AppendFrame frame = stack.back();
      stack.pop_back();
      if (frame.nodeIndex >= parsedNodes.size()) {
        continue;
      }
      if (frame.processChildren) {
        remappedNodeIndices[frame.nodeIndex] =
            static_cast<uint32_t>(orderedParsedNodeIndices.size());
        orderedParsedNodeIndices.push_back(frame.nodeIndex);
        appendActive[frame.nodeIndex] = 0u;
        const auto &children = parsedNodes[frame.nodeIndex].children;
        for (auto it = children.rbegin(); it != children.rend(); ++it) {
          stack.push_back({*it, false});
        }
        continue;
      }
      if (remappedNodeIndices[frame.nodeIndex] != kInvalidScenePrefabIndex ||
          appendActive[frame.nodeIndex] != 0u) {
        continue;
      }
      appendActive[frame.nodeIndex] = 1u;
      stack.push_back({frame.nodeIndex, true});
      const uint32_t parentIndex = parentIndices[frame.nodeIndex];
      if (parentIndex != kInvalidScenePrefabIndex) {
        stack.push_back({parentIndex, false});
      }
    }
  }

  std::pmr::vector<ImportedSceneNode> structuralNodes =
      std::move(imported.nodes);
  std::pmr::vector<ImportedSceneRenderable> structuralRenderables =
      std::move(imported.renderables);
  std::pmr::vector<ImportedSceneMeshAsset> structuralMeshAssets =
      std::move(imported.meshAssets);
  std::pmr::vector<ImportedSceneMaterialAsset> structuralMaterialAssets =
      std::move(imported.materialAssets);
  std::pmr::vector<uint32_t> structuralRootNodes =
      std::move(imported.rootNodes);

  imported.nodes.clear();
  imported.renderables.clear();
  imported.meshAssets.clear();
  imported.materialAssets.clear();
  imported.lights.clear();
  imported.rootNodes.clear();
  imported.nodes.reserve(parsedNodes.size());
  imported.rootNodes.reserve(parsedRoots.size());

  std::vector<std::vector<uint32_t>> structuralChildren(structuralNodes.size());
  for (uint32_t nodeIndex = 0u; nodeIndex < structuralNodes.size();
       ++nodeIndex) {
    const uint32_t parentIndex = structuralNodes[nodeIndex].parentIndex;
    if (parentIndex == kInvalidScenePrefabIndex ||
        parentIndex >= structuralChildren.size()) {
      continue;
    }
    structuralChildren[parentIndex].push_back(nodeIndex);
  }

  std::vector<std::vector<uint32_t>> structuralRenderableIndicesByNode(
      structuralNodes.size());
  for (uint32_t renderableIndex = 0u;
       renderableIndex < structuralRenderables.size(); ++renderableIndex) {
    const ImportedSceneRenderable &renderable =
        structuralRenderables[renderableIndex];
    if (renderable.nodeIndex >= structuralRenderableIndicesByNode.size()) {
      continue;
    }
    structuralRenderableIndicesByNode[renderable.nodeIndex].push_back(
        renderableIndex);
  }

  std::vector<uint32_t> structuralRootCandidates;
  structuralRootCandidates.reserve(structuralRootNodes.size());
  for (const uint32_t rootIndex : structuralRootNodes) {
    if (rootIndex >= structuralNodes.size()) {
      continue;
    }
    const ImportedSceneNode &rootNode = structuralNodes[rootIndex];
    if (rootNode.parentIndex == kInvalidScenePrefabIndex &&
        rootNode.name.empty() && !structuralChildren[rootIndex].empty()) {
      structuralRootCandidates.insert(structuralRootCandidates.end(),
                                      structuralChildren[rootIndex].begin(),
                                      structuralChildren[rootIndex].end());
      continue;
    }
    structuralRootCandidates.push_back(rootIndex);
  }
  if (structuralRootCandidates.empty()) {
    structuralRootCandidates.insert(structuralRootCandidates.end(),
                                    structuralRootNodes.begin(),
                                    structuralRootNodes.end());
  }

  std::vector<uint8_t> usedStructuralNodes(structuralNodes.size(), 0u);
  std::vector<uint32_t> structuralNodeByParsedNode(parsedNodes.size(),
                                                   kInvalidScenePrefabIndex);

  const auto chooseCandidateFromSet =
      [&](const ParsedNode &parsedNode,
          std::span<const uint32_t> candidates) -> uint32_t {
    std::vector<uint32_t> filtered;
    filtered.reserve(candidates.size());
    for (const uint32_t candidateIndex : candidates) {
      if (candidateIndex >= structuralNodes.size() ||
          usedStructuralNodes[candidateIndex] != 0u) {
        continue;
      }
      filtered.push_back(candidateIndex);
    }
    if (filtered.empty()) {
      return kInvalidScenePrefabIndex;
    }

    const auto chooseBest =
        [&](std::span<const uint32_t> localCandidates) -> uint32_t {
      uint32_t matrixMatch = kInvalidScenePrefabIndex;
      uint32_t presenceMatch = kInvalidScenePrefabIndex;
      for (const uint32_t candidateIndex : localCandidates) {
        const ImportedSceneNode &candidate = structuralNodes[candidateIndex];
        if (matrixNearlyEqual(candidate.localFromParent,
                              parsedNode.localMatrix)) {
          if (matrixMatch != kInvalidScenePrefabIndex) {
            matrixMatch = kInvalidScenePrefabIndex;
            break;
          }
          matrixMatch = candidateIndex;
        }
      }
      if (matrixMatch != kInvalidScenePrefabIndex) {
        return matrixMatch;
      }

      const bool parsedHasRenderable = parsedNode.meshIndex.has_value();
      for (const uint32_t candidateIndex : localCandidates) {
        const bool structuralHasRenderable =
            candidateIndex < structuralRenderableIndicesByNode.size() &&
            !structuralRenderableIndicesByNode[candidateIndex].empty();
        if (structuralHasRenderable != parsedHasRenderable) {
          continue;
        }
        if (presenceMatch != kInvalidScenePrefabIndex) {
          return kInvalidScenePrefabIndex;
        }
        presenceMatch = candidateIndex;
      }
      if (presenceMatch != kInvalidScenePrefabIndex) {
        return presenceMatch;
      }
      return localCandidates.size() == 1u ? localCandidates.front()
                                          : kInvalidScenePrefabIndex;
    };

    if (!parsedNode.name.empty()) {
      std::vector<uint32_t> namedCandidates;
      namedCandidates.reserve(filtered.size());
      for (const uint32_t candidateIndex : filtered) {
        if (std::string_view(structuralNodes[candidateIndex].name) ==
            std::string_view(parsedNode.name)) {
          namedCandidates.push_back(candidateIndex);
        }
      }
      if (!namedCandidates.empty()) {
        return chooseBest(namedCandidates);
      }
      return kInvalidScenePrefabIndex;
    }

    std::vector<uint32_t> unnamedCandidates;
    unnamedCandidates.reserve(filtered.size());
    for (const uint32_t candidateIndex : filtered) {
      if (structuralNodes[candidateIndex].name.empty()) {
        unnamedCandidates.push_back(candidateIndex);
      }
    }
    if (!unnamedCandidates.empty()) {
      return chooseBest(unnamedCandidates);
    }
    return chooseBest(filtered);
  };

  struct SubtreeFrame {
    uint32_t parsedNodeIndex = kInvalidScenePrefabIndex;
    uint32_t structuralParentIndex = kInvalidScenePrefabIndex;
  };
  for (const uint32_t rootIndex : parsedRoots) {
    std::vector<SubtreeFrame> stack;
    stack.push_back({rootIndex, kInvalidScenePrefabIndex});
    while (!stack.empty()) {
      const SubtreeFrame frame = stack.back();
      stack.pop_back();
      if (frame.parsedNodeIndex >= parsedNodes.size()) {
        continue;
      }

      const ParsedNode &parsedNode = parsedNodes[frame.parsedNodeIndex];
      const std::vector<uint32_t> &candidateSet =
          frame.structuralParentIndex == kInvalidScenePrefabIndex
              ? structuralRootCandidates
              : structuralChildren[frame.structuralParentIndex];

      uint32_t matchedStructuralNode =
          chooseCandidateFromSet(parsedNode, candidateSet);
      if (matchedStructuralNode == kInvalidScenePrefabIndex) {
        const uint32_t globalMatch = resolveImportedNodeIndex(
            parsedNode, frame.parsedNodeIndex, parsedPaths, importedPathToNode,
            importedNameToNodes, structuralNodes);
        if (globalMatch != kInvalidScenePrefabIndex &&
            globalMatch < usedStructuralNodes.size() &&
            usedStructuralNodes[globalMatch] == 0u) {
          matchedStructuralNode = globalMatch;
        }
      }

      uint32_t nextStructuralParent = frame.structuralParentIndex;
      if (matchedStructuralNode != kInvalidScenePrefabIndex) {
        structuralNodeByParsedNode[frame.parsedNodeIndex] =
            matchedStructuralNode;
        usedStructuralNodes[matchedStructuralNode] = 1u;
        nextStructuralParent = matchedStructuralNode;
      }

      for (auto it = parsedNode.children.rbegin();
           it != parsedNode.children.rend(); ++it) {
        stack.push_back({*it, nextStructuralParent});
      }
    }
  }

  HashMap<uint32_t, uint32_t> meshOrdinalToAsset{};
  HashMap<uint32_t, uint32_t> materialOrdinalToAsset{};

  for (const uint32_t parsedNodeIndex : orderedParsedNodeIndices) {
    ImportedSceneNode importedNode{};
    const uint32_t parsedParentIndex = parentIndices[parsedNodeIndex];
    importedNode.parentIndex = parsedParentIndex == kInvalidScenePrefabIndex
                                   ? kInvalidScenePrefabIndex
                                   : remappedNodeIndices[parsedParentIndex];
    importedNode.localFromParent = parsedNodes[parsedNodeIndex].localMatrix;
    importedNode.name = parsedNodes[parsedNodeIndex].name;
    importedNode.morphWeights.assign(
        parsedNodes[parsedNodeIndex].morphWeights.begin(),
        parsedNodes[parsedNodeIndex].morphWeights.end());
    imported.nodes.push_back(std::move(importedNode));
  }
  for (const uint32_t rootIndex : parsedRoots) {
    imported.rootNodes.push_back(remappedNodeIndices[rootIndex]);
  }

  size_t copiedStructuralRenderableCount = 0u;
  size_t fallbackRenderableCount = 0u;
  size_t unmatchedParsedMeshNodeCount = 0u;
  for (uint32_t parsedNodeIndex = 0u; parsedNodeIndex < parsedNodes.size();
       ++parsedNodeIndex) {
    if (remappedNodeIndices[parsedNodeIndex] == kInvalidScenePrefabIndex) {
      continue;
    }
    const ParsedNode &parsedNode = parsedNodes[parsedNodeIndex];
    bool copiedStructuralRenderables = false;
    const uint32_t structuralNodeIndex =
        structuralNodeByParsedNode[parsedNodeIndex];
    if (structuralNodeIndex != kInvalidScenePrefabIndex &&
        structuralNodeIndex < structuralRenderableIndicesByNode.size()) {
      for (const uint32_t renderableIndex :
           structuralRenderableIndicesByNode[structuralNodeIndex]) {
        if (renderableIndex >= structuralRenderables.size()) {
          continue;
        }
        const ImportedSceneRenderable &structuralRenderable =
            structuralRenderables[renderableIndex];
        if (structuralRenderable.meshAssetIndex >=
                structuralMeshAssets.size() ||
            structuralRenderable.materialAssetIndex >=
                structuralMaterialAssets.size()) {
          continue;
        }

        const ImportedSceneMeshAsset &structuralMeshAsset =
            structuralMeshAssets[structuralRenderable.meshAssetIndex];
        const ImportedSceneMaterialAsset &structuralMaterialAsset =
            structuralMaterialAssets[structuralRenderable.materialAssetIndex];
        uint32_t sourceMaterialIndex =
            structuralMaterialAsset.sourceMaterialIndex;
        if (parsedNode.meshIndex.has_value()) {
          sourceMaterialIndex = resolveGltfSinglePrimitiveMeshMaterialIndex(
              materialMapping, *parsedNode.meshIndex, sourceMaterialIndex);
        }
        sourceMaterialIndex = resolveGltfPrimitiveMaterialIndex(
            materialMapping, structuralMeshAsset.sourceSceneMeshIndex,
            sourceMaterialIndex);
        const std::string_view sourceMaterialName =
            sourceMaterialIndex == structuralMaterialAsset.sourceMaterialIndex
                ? std::string_view(structuralMaterialAsset.sourceName)
                : std::string_view{};
        imported.renderables.push_back(ImportedSceneRenderable{
            .nodeIndex = remappedNodeIndices[parsedNodeIndex],
            .meshAssetIndex = ensureImportedMeshAsset(
                imported, scene, meshOrdinalToAsset,
                structuralMeshAsset.sourceSceneMeshIndex,
                structuralMeshAsset.sourceName),
            .materialAssetIndex = ensureImportedMaterialAsset(
                imported, scene, materialOrdinalToAsset, sourceMaterialIndex,
                sourceMaterialName),
            .skinIndex =
                parsedNode.skinIndex.value_or(kInvalidScenePrefabIndex),
        });
        copiedStructuralRenderables = true;
        ++copiedStructuralRenderableCount;
      }
    }
    if (copiedStructuralRenderables || !parsedNode.meshIndex.has_value()) {
      continue;
    }

    const uint32_t sourceSceneMeshIndex = *parsedNode.meshIndex;
    if (sourceSceneMeshIndex >= scene.mNumMeshes ||
        scene.mMeshes[sourceSceneMeshIndex] == nullptr) {
      ++unmatchedParsedMeshNodeCount;
      continue;
    }
    const aiMesh &mesh = *scene.mMeshes[sourceSceneMeshIndex];
    if (!hasRenderableTriangleGeometry(mesh)) {
      ++unmatchedParsedMeshNodeCount;
      continue;
    }

    uint32_t sourceMaterialIndex = resolveGltfSinglePrimitiveMeshMaterialIndex(
        materialMapping, *parsedNode.meshIndex, mesh.mMaterialIndex);
    sourceMaterialIndex = resolveGltfPrimitiveMaterialIndex(
        materialMapping, sourceSceneMeshIndex, sourceMaterialIndex);
    imported.renderables.push_back(ImportedSceneRenderable{
        .nodeIndex = remappedNodeIndices[parsedNodeIndex],
        .meshAssetIndex = ensureImportedMeshAsset(
            imported, scene, meshOrdinalToAsset, sourceSceneMeshIndex),
        .materialAssetIndex = ensureImportedMaterialAsset(
            imported, scene, materialOrdinalToAsset, sourceMaterialIndex),
        .skinIndex = parsedNode.skinIndex.value_or(kInvalidScenePrefabIndex),
    });
    ++fallbackRenderableCount;
  }

  const uint32_t materialCountLimit =
      materialMapping.materialCount > 0u ? materialMapping.materialCount
                                         : std::numeric_limits<uint32_t>::max();
  appendRemainingSceneAssets(imported, scene, meshOrdinalToAsset,
                             materialOrdinalToAsset, materialCountLimit);

  NURI_LOG_DEBUG(
      "SceneImporter::loadSceneFromFile: glTF rebuild matched %zu/%zu parsed "
      "nodes to structural nodes; copied %zu structural renderable(s), "
      "fell back to %zu raw mesh binding(s), skipped %zu parsed mesh node(s)",
      std::ranges::count_if(
          structuralNodeByParsedNode,
          [](uint32_t index) { return index != kInvalidScenePrefabIndex; }),
      parsedNodes.size(), copiedStructuralRenderableCount,
      fallbackRenderableCount, unmatchedParsedMeshNodeCount);

  auto attachStructuralLightsResult = attachParsedLightsToImportedNodes(
      imported, parsedNodes, parsedLights,
      [&](const ParsedNode &, uint32_t parsedNodeIndex) {
        return parsedNodeIndex < remappedNodeIndices.size()
                   ? remappedNodeIndices[parsedNodeIndex]
                   : kInvalidScenePrefabIndex;
      });
  if (attachStructuralLightsResult.hasError()) {
    return Result<bool, std::string>::makeError(
        attachStructuralLightsResult.error());
  }

  if (outParsedToImportedNodeIndices != nullptr) {
    outParsedToImportedNodeIndices->assign(parsedNodes.size(),
                                           kInvalidScenePrefabIndex);
    for (uint32_t parsedNodeIndex = 0u; parsedNodeIndex < parsedNodes.size();
         ++parsedNodeIndex) {
      if (parsedNodeIndex < remappedNodeIndices.size()) {
        (*outParsedToImportedNodeIndices)[parsedNodeIndex] =
            remappedNodeIndices[parsedNodeIndex];
      }
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<std::vector<uint32_t>, std::string>
resolveSceneRootNodes(yyjson_val *root, size_t nodeCount) {
  if (nodeCount == 0u) {
    return Result<std::vector<uint32_t>, std::string>::makeResult({});
  }

  yyjson_val *scenesValue = yyjson_obj_get(root, "scenes");
  if (yyjson_is_arr(scenesValue) && yyjson_arr_size(scenesValue) > 0u) {
    uint32_t sceneIndex = 0u;
    if (!detail::tryReadJsonUint32(yyjson_obj_get(root, "scene"), sceneIndex)) {
      sceneIndex = 0u;
    }
    if (sceneIndex >= yyjson_arr_size(scenesValue)) {
      return Result<std::vector<uint32_t>, std::string>::makeError(
          "glTF selected scene index is out of range");
    }

    yyjson_val *sceneValue = yyjson_arr_get(scenesValue, sceneIndex);
    yyjson_val *sceneNodesValue = yyjson_obj_get(sceneValue, "nodes");
    if (!yyjson_is_arr(sceneNodesValue)) {
      return Result<std::vector<uint32_t>, std::string>::makeResult({});
    }

    std::vector<uint32_t> roots;
    roots.reserve(yyjson_arr_size(sceneNodesValue));
    yyjson_arr_iter iter = yyjson_arr_iter_with(sceneNodesValue);
    yyjson_val *rootNodeValue = nullptr;
    while ((rootNodeValue = yyjson_arr_iter_next(&iter)) != nullptr) {
      uint32_t rootNodeIndex = 0u;
      if (!detail::tryReadJsonUint32(rootNodeValue, rootNodeIndex)) {
        return Result<std::vector<uint32_t>, std::string>::makeError(
            "glTF scene root node index is invalid");
      }
      roots.push_back(rootNodeIndex);
    }
    return Result<std::vector<uint32_t>, std::string>::makeResult(
        std::move(roots));
  }

  std::vector<uint32_t> roots;
  std::vector<uint8_t> referenced(nodeCount, 0u);
  yyjson_val *nodesValue = yyjson_obj_get(root, "nodes");
  if (yyjson_is_arr(nodesValue)) {
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
      yyjson_val *nodeValue = yyjson_arr_get(nodesValue, nodeIndex);
      yyjson_val *childrenValue = yyjson_obj_get(nodeValue, "children");
      if (!yyjson_is_arr(childrenValue)) {
        continue;
      }

      yyjson_arr_iter iter = yyjson_arr_iter_with(childrenValue);
      yyjson_val *childValue = nullptr;
      while ((childValue = yyjson_arr_iter_next(&iter)) != nullptr) {
        uint32_t childIndex = 0u;
        if (!detail::tryReadJsonUint32(childValue, childIndex) ||
            childIndex >= nodeCount) {
          return Result<std::vector<uint32_t>, std::string>::makeError(
              "glTF node child index is invalid");
        }
        referenced[childIndex] = 1u;
      }
    }
  }

  roots.reserve(nodeCount);
  for (uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex) {
    if (referenced[nodeIndex] == 0u) {
      roots.push_back(nodeIndex);
    }
  }
  if (roots.empty()) {
    NURI_LOG_WARNING(
        "SceneImporter::resolveSceneRootNodes: glTF root fallback: no "
        "unreferenced nodes; treating all %zu nodes as roots",
        nodeCount);
    for (uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex) {
      roots.push_back(nodeIndex);
    }
  }
  return Result<std::vector<uint32_t>, std::string>::makeResult(
      std::move(roots));
}

void buildParsedNodePathsRecursive(uint32_t nodeIndex,
                                   std::span<const ParsedNode> nodes,
                                   std::vector<std::string> &outPaths,
                                   std::vector<uint8_t> &active,
                                   const std::string &parentPath,
                                   uint32_t siblingOrdinal) {
  if (nodeIndex >= nodes.size() || active[nodeIndex] != 0u) {
    return;
  }
  active[nodeIndex] = 1u;
  outPaths[nodeIndex] =
      makeNodePath(parentPath, nodes[nodeIndex].name, siblingOrdinal);
  for (size_t childOrdinal = 0; childOrdinal < nodes[nodeIndex].children.size();
       ++childOrdinal) {
    buildParsedNodePathsRecursive(nodes[nodeIndex].children[childOrdinal],
                                  nodes, outPaths, active, outPaths[nodeIndex],
                                  static_cast<uint32_t>(childOrdinal));
  }
  active[nodeIndex] = 0u;
}

[[nodiscard]] uint32_t resolveImportedNodeIndex(
    const ParsedNode &parsedNode, uint32_t parsedNodeIndex,
    std::span<const std::string> parsedPaths,
    const HashMap<std::string, uint32_t> &importedPathToNode,
    const HashMap<std::string, std::vector<uint32_t>> &importedNameToNodes,
    std::span<const ImportedSceneNode> importedNodes) {
  if (parsedNodeIndex < parsedPaths.size()) {
    auto it = importedPathToNode.find(parsedPaths[parsedNodeIndex]);
    if (it != importedPathToNode.end()) {
      return it->second;
    }
  }

  if (!parsedNode.name.empty()) {
    auto nameIt = importedNameToNodes.find(parsedNode.name);
    if (nameIt != importedNameToNodes.end() && nameIt->second.size() == 1u) {
      return nameIt->second.front();
    }
  }

  uint32_t match = kInvalidScenePrefabIndex;
  for (uint32_t nodeIndex = 0u; nodeIndex < importedNodes.size(); ++nodeIndex) {
    const ImportedSceneNode &candidate = importedNodes[nodeIndex];
    if (matrixNearlyEqual(candidate.localFromParent, parsedNode.localMatrix)) {
      if (match != kInvalidScenePrefabIndex) {
        return kInvalidScenePrefabIndex;
      }
      match = nodeIndex;
    }
  }
  return match;
}

[[nodiscard]] unsigned int structuralSceneAssimpFlags() {
  return aiProcess_SortByPType | aiProcess_FindInvalidData;
}

[[nodiscard]] std::string importedSceneName(const aiScene &scene,
                                            std::string_view path) {
  if (scene.mName.length > 0u) {
    return std::string(scene.mName.C_Str(), scene.mName.length);
  }
  return std::filesystem::path(std::string(path)).stem().string();
}

} // namespace

Result<ImportedScene, std::string>
SceneImporter::loadSceneFromFile(std::string_view path,
                                 const SceneImportOptions &options,
                                 std::pmr::memory_resource *memory) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (path.empty()) {
    return Result<ImportedScene, std::string>::makeError(
        "SceneImporter::loadSceneFromFile: path is empty");
  }
  if (memory == nullptr) {
    memory = std::pmr::get_default_resource();
  }

  ImportedScene imported(memory);
  imported.sourcePath = canonicalizeResourcePath(path);
  imported.importOptions = options.assetBuildOptions;

  Assimp::Importer importer;
  const std::string pathString(path);
  const unsigned int assimpFlags =
      options.adaptAssetSources
          ? detail::sceneMeshImportFlags(options.assetBuildOptions)
          : structuralSceneAssimpFlags();
  const aiScene *scene = importer.ReadFile(pathString, assimpFlags);
  if (scene == nullptr || scene->mRootNode == nullptr) {
    return Result<ImportedScene, std::string>::makeError(
        std::string("SceneImporter::loadSceneFromFile: Assimp error: ") +
        importer.GetErrorString());
  }

  const std::string sceneName = importedSceneName(*scene, path);
  imported.sourceSceneName.assign(sceneName.data(), sceneName.size());
  const bool hasSyntheticRootWrapper = scene->mRootNode->mParent == nullptr &&
                                       scene->mRootNode->mNumMeshes == 0u &&
                                       scene->mRootNode->mName.length == 0u;

  HashMap<uint32_t, uint32_t> meshOrdinalToAsset{};
  HashMap<uint32_t, uint32_t> materialOrdinalToAsset{};
  HashMap<std::string, uint32_t> importedPathToNode{};
  HashMap<std::string, std::vector<uint32_t>> importedNameToNodes{};

  std::function<void(const aiNode *, uint32_t, const std::string &, uint32_t)>
      addNodeRecursive;
  addNodeRecursive = [&](const aiNode *node, uint32_t parentIndex,
                         const std::string &parentPath,
                         uint32_t siblingOrdinal) {
    if (node == nullptr) {
      return;
    }

    const uint32_t importedNodeIndex =
        static_cast<uint32_t>(imported.nodes.size());
    ImportedSceneNode importedNode{};
    importedNode.parentIndex = parentIndex;
    importedNode.localFromParent = aiMatrix4x4ToMat4(node->mTransformation);
    if (node->mName.length > 0u) {
      importedNode.name.assign(node->mName.C_Str(), node->mName.length);
    }

    const std::string pathKey =
        makeNodePath(parentPath, importedNode.name, siblingOrdinal);
    importedPathToNode.emplace(pathKey, importedNodeIndex);
    if (hasSyntheticRootWrapper && importedNodeIndex != 0u) {
      const std::string strippedPath = stripLeadingPathComponent(pathKey);
      if (!strippedPath.empty() && !importedPathToNode.contains(strippedPath)) {
        importedPathToNode.emplace(strippedPath, importedNodeIndex);
      }
    }
    if (!importedNode.name.empty()) {
      importedNameToNodes[std::string(importedNode.name)].push_back(
          importedNodeIndex);
    }

    imported.nodes.push_back(std::move(importedNode));

    for (unsigned int meshSlot = 0; meshSlot < node->mNumMeshes; ++meshSlot) {
      const uint32_t sourceSceneMeshIndex = node->mMeshes[meshSlot];
      if (sourceSceneMeshIndex >= scene->mNumMeshes ||
          scene->mMeshes[sourceSceneMeshIndex] == nullptr) {
        continue;
      }
      const aiMesh *mesh = scene->mMeshes[sourceSceneMeshIndex];
      if (!hasRenderableTriangleGeometry(*mesh)) {
        continue;
      }

      const uint32_t sourceMaterialIndex = mesh->mMaterialIndex;
      imported.renderables.push_back(ImportedSceneRenderable{
          .nodeIndex = importedNodeIndex,
          .meshAssetIndex = ensureImportedMeshAsset(
              imported, *scene, meshOrdinalToAsset, sourceSceneMeshIndex),
          .materialAssetIndex = ensureImportedMaterialAsset(
              imported, *scene, materialOrdinalToAsset, sourceMaterialIndex),
      });
    }

    for (unsigned int childIndex = 0; childIndex < node->mNumChildren;
         ++childIndex) {
      addNodeRecursive(node->mChildren[childIndex], importedNodeIndex, pathKey,
                       childIndex);
    }
  };
  addNodeRecursive(scene->mRootNode, kInvalidScenePrefabIndex, std::string(),
                   0u);
  if (!imported.nodes.empty()) {
    imported.rootNodes.push_back(0u);
  }

  appendRemainingSceneAssets(imported, *scene, meshOrdinalToAsset,
                             materialOrdinalToAsset);

  const auto finalizeImported = [&]() -> Result<ImportedScene, std::string> {
    if (!options.adaptAssetSources) {
      return Result<ImportedScene, std::string>::makeResult(
          std::move(imported));
    }
    std::pmr::vector<uint32_t> sourceMeshIndices(memory);
    sourceMeshIndices.reserve(imported.meshAssets.size());
    for (const ImportedSceneMeshAsset &asset : imported.meshAssets) {
      sourceMeshIndices.push_back(asset.sourceSceneMeshIndex);
    }
    auto adaptedMeshes = detail::adaptSceneMeshes(
        *scene,
        std::span<const uint32_t>(sourceMeshIndices.data(),
                                  sourceMeshIndices.size()),
        imported.sourcePath, options.assetBuildOptions, memory);
    if (adaptedMeshes.hasError()) {
      return Result<ImportedScene, std::string>::makeError(
          "SceneImporter::loadSceneFromFile: failed to adapt mesh sources: " +
          adaptedMeshes.error());
    }
    auto adaptedMaterials =
        detail::adaptMaterialInfo(*scene, imported.sourcePath);
    if (adaptedMaterials.hasError()) {
      return Result<ImportedScene, std::string>::makeError(
          "SceneImporter::loadSceneFromFile: failed to adapt material "
          "sources: " +
          adaptedMaterials.error());
    }
    imported.adaptedMeshes = std::move(adaptedMeshes.value());
    imported.adaptedMaterials = std::move(adaptedMaterials.value());
    imported.embeddedTextures.reserve(scene->mNumTextures);
    for (uint32_t textureIndex = 0u; textureIndex < scene->mNumTextures;
         ++textureIndex) {
      const aiTexture *texture = scene->mTextures[textureIndex];
      if (texture == nullptr) {
        imported.embeddedTextures.emplace_back();
        continue;
      }
      EmbeddedSceneTextureData adaptedTexture{
          .width = texture->mWidth,
          .height = texture->mHeight,
          .compressed = texture->mHeight == 0u,
      };
      if (adaptedTexture.compressed) {
        const std::byte *begin =
            reinterpret_cast<const std::byte *>(texture->pcData);
        adaptedTexture.bytes.assign(begin, begin + texture->mWidth);
      } else {
        adaptedTexture.bytes.resize(static_cast<size_t>(texture->mWidth) *
                                    static_cast<size_t>(texture->mHeight) * 4u);
        for (size_t texelIndex = 0u;
             texelIndex < static_cast<size_t>(texture->mWidth) *
                              static_cast<size_t>(texture->mHeight);
             ++texelIndex) {
          const aiTexel &source = texture->pcData[texelIndex];
          adaptedTexture.bytes[texelIndex * 4u + 0u] =
              static_cast<std::byte>(source.r);
          adaptedTexture.bytes[texelIndex * 4u + 1u] =
              static_cast<std::byte>(source.g);
          adaptedTexture.bytes[texelIndex * 4u + 2u] =
              static_cast<std::byte>(source.b);
          adaptedTexture.bytes[texelIndex * 4u + 3u] =
              static_cast<std::byte>(source.a);
        }
      }
      imported.embeddedTextures.push_back(std::move(adaptedTexture));
    }
    return Result<ImportedScene, std::string>::makeResult(std::move(imported));
  };

  if (!detail::isGltfJsonAssetPath(path)) {
    return finalizeImported();
  }

  const std::filesystem::path scenePath{std::string(path)};
  auto fileBytesResult = readBinaryFile(scenePath);
  if (fileBytesResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(
        fileBytesResult.error());
  }

  auto docResult = detail::loadGltfJsonDocument(
      scenePath,
      std::span<const std::byte>(fileBytesResult.value().data(),
                                 fileBytesResult.value().size()),
      "glTF scene source");
  if (docResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(docResult.error());
  }
  yyjson_val *root = yyjson_doc_get_root(docResult.value().get());
  if (!yyjson_is_obj(root)) {
    return Result<ImportedScene, std::string>::makeError(
        "glTF root is not a JSON object");
  }

  auto buffersResult = detail::loadGltfBuffers(
      scenePath, root,
      std::span<const std::byte>(fileBytesResult.value().data(),
                                 fileBytesResult.value().size()),
      memory);
  if (buffersResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(buffersResult.error());
  }

  auto lightsResult = parseLightDefinitions(root);
  if (lightsResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(lightsResult.error());
  }

  auto parsedNodesResult = parseNodes(root);
  if (parsedNodesResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(
        parsedNodesResult.error());
  }
  std::vector<ParsedNode> parsedNodes = std::move(parsedNodesResult.value());
  auto skinsResult = parseSkinDefinitions(root, buffersResult.value(), memory);
  if (skinsResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(skinsResult.error());
  }
  imported.skins = std::move(skinsResult.value());
  auto animationsResult = parseAnimationDefinitions(
      root, parsedNodes, buffersResult.value(), memory);
  if (animationsResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(
        animationsResult.error());
  }
  imported.animations = std::move(animationsResult.value());
  if (parsedNodes.empty()) {
    return finalizeImported();
  }

  auto rootsResult = resolveSceneRootNodes(root, parsedNodes.size());
  if (rootsResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(rootsResult.error());
  }
  const std::vector<uint32_t> parsedRoots = std::move(rootsResult.value());
  std::vector<std::string> parsedPaths(parsedNodes.size());
  std::vector<uint8_t> parsedActive(parsedNodes.size(), 0u);
  for (size_t rootOrdinal = 0; rootOrdinal < parsedRoots.size();
       ++rootOrdinal) {
    buildParsedNodePathsRecursive(parsedRoots[rootOrdinal], parsedNodes,
                                  parsedPaths, parsedActive, std::string(),
                                  static_cast<uint32_t>(rootOrdinal));
  }

  std::vector<uint32_t> parsedToImportedNodeIndex;
  auto primitiveMaterialMappingResult =
      detail::readGltfPrimitiveMaterialMapping(root);
  if (primitiveMaterialMappingResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(
        primitiveMaterialMappingResult.error());
  }
  const detail::GltfPrimitiveMaterialMapping primitiveMaterialMapping =
      std::move(primitiveMaterialMappingResult.value());

  if (auto rebuildResult = rebuildImportedSceneFromParsedGltf(
          imported, *scene, parsedNodes, parsedRoots, parsedPaths,
          importedPathToNode, importedNameToNodes, lightsResult.value(),
          primitiveMaterialMapping, &parsedToImportedNodeIndex);
      rebuildResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(rebuildResult.error());
  } else if (rebuildResult.value()) {
    remapSkinAndAnimationNodeIndices(imported.skins, imported.animations,
                                     parsedToImportedNodeIndex);
    return finalizeImported();
  }

  parsedToImportedNodeIndex.assign(parsedNodes.size(),
                                   kInvalidScenePrefabIndex);
  std::vector<uint8_t> importedNodeMatched(imported.nodes.size(), 0u);
  for (uint32_t parsedNodeIndex = 0u; parsedNodeIndex < parsedNodes.size();
       ++parsedNodeIndex) {
    const uint32_t importedNodeIndex = resolveImportedNodeIndex(
        parsedNodes[parsedNodeIndex], parsedNodeIndex, parsedPaths,
        importedPathToNode, importedNameToNodes, imported.nodes);
    if (importedNodeIndex == kInvalidScenePrefabIndex ||
        importedNodeIndex >= imported.nodes.size() ||
        importedNodeMatched[importedNodeIndex] != 0u) {
      continue;
    }
    imported.nodes[importedNodeIndex].localFromParent =
        parsedNodes[parsedNodeIndex].localMatrix;
    imported.nodes[importedNodeIndex].morphWeights.assign(
        parsedNodes[parsedNodeIndex].morphWeights.begin(),
        parsedNodes[parsedNodeIndex].morphWeights.end());
    importedNodeMatched[importedNodeIndex] = 1u;
    parsedToImportedNodeIndex[parsedNodeIndex] = importedNodeIndex;
  }

  std::vector<std::vector<uint32_t>> renderableIndicesByNode(
      imported.nodes.size());
  for (uint32_t renderableIndex = 0u;
       renderableIndex < imported.renderables.size(); ++renderableIndex) {
    const uint32_t nodeIndex = imported.renderables[renderableIndex].nodeIndex;
    if (nodeIndex < renderableIndicesByNode.size()) {
      renderableIndicesByNode[nodeIndex].push_back(renderableIndex);
    }
  }

  for (uint32_t parsedNodeIndex = 0u; parsedNodeIndex < parsedNodes.size();
       ++parsedNodeIndex) {
    const uint32_t importedNodeIndex =
        parsedToImportedNodeIndex[parsedNodeIndex];
    if (importedNodeIndex == kInvalidScenePrefabIndex) {
      continue;
    }
    if (parsedNodes[parsedNodeIndex].skinIndex.has_value()) {
      for (const uint32_t renderableIndex :
           renderableIndicesByNode[importedNodeIndex]) {
        imported.renderables[renderableIndex].skinIndex =
            parsedNodes[parsedNodeIndex].skinIndex.value();
      }
    }
  }

  const std::vector<ParsedLightDef> parsedLights =
      std::move(lightsResult.value());
  auto attachImportedLightsResult = attachParsedLightsToImportedNodes(
      imported, parsedNodes, parsedLights,
      [&](const ParsedNode &parsedNode, uint32_t parsedNodeIndex) {
        const uint32_t importedNodeIndex = resolveImportedNodeIndex(
            parsedNode, parsedNodeIndex, parsedPaths, importedPathToNode,
            importedNameToNodes, imported.nodes);
        if (importedNodeIndex == kInvalidScenePrefabIndex) {
          NURI_LOG_WARNING(
              "SceneImporter::loadSceneFromFile: skipping glTF light "
              "attachment for parsed node %u ('%s') because no unique "
              "structural node match was found",
              parsedNodeIndex, parsedNode.name.c_str());
        }
        return importedNodeIndex;
      });
  if (attachImportedLightsResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(
        attachImportedLightsResult.error());
  }

  remapSkinAndAnimationNodeIndices(imported.skins, imported.animations,
                                   parsedToImportedNodeIndex);

  return finalizeImported();
}

Result<ScenePrefab, std::string>
SceneImporter::buildScenePrefab(const ImportedScene &scene,
                                std::pmr::memory_resource *memory) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (memory == nullptr) {
    memory = std::pmr::get_default_resource();
  }

  ScenePrefab prefab(memory);
  prefab.sourcePath = scene.sourcePath;
  prefab.sourceSceneName = scene.sourceSceneName;
  prefab.importOptions = scene.importOptions;

  prefab.nodes.reserve(scene.nodes.size());
  for (const ImportedSceneNode &sceneNode : scene.nodes) {
    prefab.nodes.emplace_back(memory);
    ScenePrefabNode &prefabNode = prefab.nodes.back();
    prefabNode.parentIndex = sceneNode.parentIndex;
    prefabNode.localFromParent = sceneNode.localFromParent;
    prefabNode.name.assign(sceneNode.name.data(), sceneNode.name.size());
    prefabNode.morphWeights.assign(sceneNode.morphWeights.begin(),
                                   sceneNode.morphWeights.end());
  }

  prefab.meshAssets.reserve(scene.meshAssets.size());
  for (const ImportedSceneMeshAsset &meshAsset : scene.meshAssets) {
    prefab.meshAssets.push_back(ScenePrefabMeshAssetRef{
        .sourceSceneMeshIndex = meshAsset.sourceSceneMeshIndex,
        .sourceName = std::string(meshAsset.sourceName),
    });
  }

  prefab.materialAssets.reserve(scene.materialAssets.size());
  for (const ImportedSceneMaterialAsset &materialAsset : scene.materialAssets) {
    prefab.materialAssets.push_back(ScenePrefabMaterialAssetRef{
        .sourceMaterialIndex = materialAsset.sourceMaterialIndex,
        .sourceName = std::string(materialAsset.sourceName),
    });
  }

  prefab.renderables.reserve(scene.renderables.size());
  for (const ImportedSceneRenderable &renderable : scene.renderables) {
    prefab.renderables.push_back(ScenePrefabRenderable{
        .nodeIndex = renderable.nodeIndex,
        .meshIndex = renderable.meshAssetIndex,
        .materialIndex = renderable.materialAssetIndex,
        .skinIndex = renderable.skinIndex,
    });
  }

  prefab.lights.reserve(scene.lights.size());
  for (const ImportedSceneLight &sceneLight : scene.lights) {
    ScenePrefabLight prefabLight{};
    prefabLight.nodeIndex = sceneLight.nodeIndex;
    prefabLight.light = sceneLight.light;
    prefabLight.light.position = glm::vec3(0.0f);
    prefabLight.light.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    if (prefabLight.light.name.empty()) {
      prefabLight.light.name = std::string(sceneLight.sourceName);
    }
    prefab.lights.push_back(std::move(prefabLight));
  }

  prefab.skins.reserve(scene.skins.size());
  for (const SkinData &skin : scene.skins) {
    prefab.skins.emplace_back(memory);
    SkinData &prefabSkin = prefab.skins.back();
    prefabSkin.name.assign(skin.name.data(), skin.name.size());
    prefabSkin.skeletonRootNodeIndex = skin.skeletonRootNodeIndex;
    prefabSkin.jointNodeIndices.assign(skin.jointNodeIndices.begin(),
                                       skin.jointNodeIndices.end());
    prefabSkin.inverseBindMatrices.assign(skin.inverseBindMatrices.begin(),
                                          skin.inverseBindMatrices.end());
  }

  prefab.animations.reserve(scene.animations.size());
  for (const AnimationClipData &clip : scene.animations) {
    prefab.animations.emplace_back(memory);
    AnimationClipData &prefabClip = prefab.animations.back();
    prefabClip.name.assign(clip.name.data(), clip.name.size());
    prefabClip.durationSeconds = clip.durationSeconds;
    prefabClip.samplers.reserve(clip.samplers.size());
    for (const AnimationSamplerData &sampler : clip.samplers) {
      prefabClip.samplers.emplace_back(memory);
      AnimationSamplerData &prefabSampler = prefabClip.samplers.back();
      prefabSampler.interpolation = sampler.interpolation;
      prefabSampler.valueArity = sampler.valueArity;
      prefabSampler.keyTimes.assign(sampler.keyTimes.begin(),
                                    sampler.keyTimes.end());
      prefabSampler.values.assign(sampler.values.begin(), sampler.values.end());
    }
    prefabClip.channels.assign(clip.channels.begin(), clip.channels.end());
  }

  return Result<ScenePrefab, std::string>::makeResult(std::move(prefab));
}

Result<ImportedSceneAssets, std::string>
SceneImporter::buildSceneAssets(const ImportedScene &scene,
                                std::pmr::memory_resource *memory) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (memory == nullptr) {
    memory = std::pmr::get_default_resource();
  }

  ImportedSceneAssets assets(memory);
  std::pmr::vector<uint32_t> sourceMeshIndices(memory);
  sourceMeshIndices.reserve(scene.meshAssets.size());
  for (const ImportedSceneMeshAsset &meshAsset : scene.meshAssets) {
    sourceMeshIndices.push_back(meshAsset.sourceSceneMeshIndex);
  }

  if (!sourceMeshIndices.empty()) {
    auto meshesResult = detail::loadSceneMeshesFromSourceIndices(
        scene.sourcePath,
        std::span<const uint32_t>(sourceMeshIndices.data(),
                                  sourceMeshIndices.size()),
        scene.importOptions, memory);
    if (meshesResult.hasError()) {
      return Result<ImportedSceneAssets, std::string>::makeError(
          "SceneImporter::buildSceneAssets: failed to build scene meshes: " +
          meshesResult.error());
    }
    assets.meshes = std::move(meshesResult.value());
  }

  auto materialsResult =
      detail::loadMaterialInfoFromSourceFile(scene.sourcePath);
  if (materialsResult.hasError()) {
    return Result<ImportedSceneAssets, std::string>::makeError(
        "SceneImporter::buildSceneAssets: failed to build scene materials: " +
        materialsResult.error());
  }
  const ImportedMaterialSet &allMaterials = materialsResult.value();
  assets.materials.materials.reserve(scene.materialAssets.size());
  for (const ImportedSceneMaterialAsset &materialAsset : scene.materialAssets) {
    if (materialAsset.sourceMaterialIndex < allMaterials.materials.size()) {
      assets.materials.materials.push_back(
          allMaterials.materials[materialAsset.sourceMaterialIndex]);
    } else {
      MaterialData fallback{};
      fallback.name = std::string(materialAsset.sourceName);
      assets.materials.materials.push_back(std::move(fallback));
    }
  }

  return Result<ImportedSceneAssets, std::string>::makeResult(
      std::move(assets));
}

Result<ScenePrefab, std::string>
SceneImporter::loadScenePrefabFromFile(std::string_view path,
                                       const SceneImportOptions &options,
                                       std::pmr::memory_resource *memory) {
  auto sceneResult = loadSceneFromFile(path, options, memory);
  if (sceneResult.hasError()) {
    return Result<ScenePrefab, std::string>::makeError(sceneResult.error());
  }
  return buildScenePrefab(sceneResult.value(), memory);
}

Result<ImportedSceneAssets, std::string>
SceneImporter::loadSceneAssetsFromFile(std::string_view path,
                                       const SceneImportOptions &options,
                                       std::pmr::memory_resource *memory) {
  auto sceneResult = loadSceneFromFile(path, options, memory);
  if (sceneResult.hasError()) {
    return Result<ImportedSceneAssets, std::string>::makeError(
        sceneResult.error());
  }
  return buildSceneAssets(sceneResult.value(), memory);
}

} // namespace nuri
