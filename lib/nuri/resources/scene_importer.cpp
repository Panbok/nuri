#include "nuri/pch.h"

#include "nuri/resources/scene_importer.h"

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/math/light.h"
#include "nuri/resources/detail/gltf_json_utils.h"
#include "nuri/resources/detail/scene_asset_build_backend.h"
#include "nuri/resources/gpu/resource_keys.h"

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
  std::optional<uint32_t> lightIndex{};
};

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
  sanitized.rotation = sanitizeRotation(desc.rotation);
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

[[nodiscard]] uint32_t resolveImportedLightNodeIndex(
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
  const aiScene *scene =
      importer.ReadFile(pathString, structuralSceneAssimpFlags());
  if (scene == nullptr || scene->mRootNode == nullptr) {
    return Result<ImportedScene, std::string>::makeError(
        std::string("SceneImporter::loadSceneFromFile: Assimp error: ") +
        importer.GetErrorString());
  }

  const std::string sceneName = importedSceneName(*scene, path);
  imported.sourceSceneName.assign(sceneName.data(), sceneName.size());

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
    if (!importedNode.name.empty()) {
      importedNameToNodes[importedNode.name].push_back(importedNodeIndex);
    }

    imported.nodes.push_back(std::move(importedNode));

    for (unsigned int meshSlot = 0; meshSlot < node->mNumMeshes; ++meshSlot) {
      const uint32_t sourceSceneMeshIndex = node->mMeshes[meshSlot];
      if (sourceSceneMeshIndex >= scene->mNumMeshes ||
          scene->mMeshes[sourceSceneMeshIndex] == nullptr) {
        continue;
      }

      uint32_t meshAssetIndex = kInvalidScenePrefabIndex;
      if (auto it = meshOrdinalToAsset.find(sourceSceneMeshIndex);
          it != meshOrdinalToAsset.end()) {
        meshAssetIndex = it->second;
      } else {
        meshAssetIndex = static_cast<uint32_t>(imported.meshAssets.size());
        const aiMesh *mesh = scene->mMeshes[sourceSceneMeshIndex];
        ImportedSceneMeshAsset meshAsset{};
        meshAsset.sourceSceneMeshIndex = sourceSceneMeshIndex;
        if (mesh->mName.length > 0u) {
          meshAsset.sourceName.assign(mesh->mName.C_Str(), mesh->mName.length);
        } else {
          meshAsset.sourceName = makeFallbackMeshName(sourceSceneMeshIndex);
        }
        imported.meshAssets.push_back(std::move(meshAsset));
        meshOrdinalToAsset.emplace(sourceSceneMeshIndex, meshAssetIndex);
      }

      const aiMesh *mesh = scene->mMeshes[sourceSceneMeshIndex];
      const uint32_t sourceMaterialIndex = mesh->mMaterialIndex;
      uint32_t materialAssetIndex = kInvalidScenePrefabIndex;
      if (auto it = materialOrdinalToAsset.find(sourceMaterialIndex);
          it != materialOrdinalToAsset.end()) {
        materialAssetIndex = it->second;
      } else {
        materialAssetIndex =
            static_cast<uint32_t>(imported.materialAssets.size());
        ImportedSceneMaterialAsset materialAsset{};
        materialAsset.sourceMaterialIndex = sourceMaterialIndex;
        if (sourceMaterialIndex < scene->mNumMaterials &&
            scene->mMaterials[sourceMaterialIndex] != nullptr) {
          aiString materialName;
          if (scene->mMaterials[sourceMaterialIndex]->Get(
                  AI_MATKEY_NAME, materialName) == aiReturn_SUCCESS &&
              materialName.length > 0u) {
            materialAsset.sourceName.assign(materialName.C_Str(),
                                            materialName.length);
          }
        }
        if (materialAsset.sourceName.empty()) {
          materialAsset.sourceName =
              makeFallbackMaterialName(sourceMaterialIndex);
        }
        imported.materialAssets.push_back(std::move(materialAsset));
        materialOrdinalToAsset.emplace(sourceMaterialIndex, materialAssetIndex);
      }

      imported.renderables.push_back(ImportedSceneRenderable{
          .nodeIndex = importedNodeIndex,
          .meshAssetIndex = meshAssetIndex,
          .materialAssetIndex = materialAssetIndex,
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

  for (uint32_t sourceSceneMeshIndex = 0u;
       sourceSceneMeshIndex < scene->mNumMeshes; ++sourceSceneMeshIndex) {
    if (scene->mMeshes[sourceSceneMeshIndex] == nullptr ||
        meshOrdinalToAsset.contains(sourceSceneMeshIndex)) {
      continue;
    }
    ImportedSceneMeshAsset meshAsset{};
    meshAsset.sourceSceneMeshIndex = sourceSceneMeshIndex;
    if (scene->mMeshes[sourceSceneMeshIndex]->mName.length > 0u) {
      meshAsset.sourceName.assign(
          scene->mMeshes[sourceSceneMeshIndex]->mName.C_Str(),
          scene->mMeshes[sourceSceneMeshIndex]->mName.length);
    } else {
      meshAsset.sourceName = makeFallbackMeshName(sourceSceneMeshIndex);
    }
    meshOrdinalToAsset.emplace(
        sourceSceneMeshIndex,
        static_cast<uint32_t>(imported.meshAssets.size()));
    imported.meshAssets.push_back(std::move(meshAsset));
  }

  for (uint32_t sourceMaterialIndex = 0u;
       sourceMaterialIndex < scene->mNumMaterials; ++sourceMaterialIndex) {
    if (materialOrdinalToAsset.contains(sourceMaterialIndex)) {
      continue;
    }
    ImportedSceneMaterialAsset materialAsset{};
    materialAsset.sourceMaterialIndex = sourceMaterialIndex;
    if (scene->mMaterials[sourceMaterialIndex] != nullptr) {
      aiString materialName;
      if (scene->mMaterials[sourceMaterialIndex]->Get(
              AI_MATKEY_NAME, materialName) == aiReturn_SUCCESS &&
          materialName.length > 0u) {
        materialAsset.sourceName.assign(materialName.C_Str(),
                                        materialName.length);
      }
    }
    if (materialAsset.sourceName.empty()) {
      materialAsset.sourceName = makeFallbackMaterialName(sourceMaterialIndex);
    }
    materialOrdinalToAsset.emplace(
        sourceMaterialIndex,
        static_cast<uint32_t>(imported.materialAssets.size()));
    imported.materialAssets.push_back(std::move(materialAsset));
  }

  if (!detail::isGltfJsonAssetPath(path)) {
    return Result<ImportedScene, std::string>::makeResult(std::move(imported));
  }

  auto docResult = detail::loadGltfJsonDocument(
      std::filesystem::path(std::string(path)), "glTF scene source");
  if (docResult.hasError()) {
    return Result<ImportedScene, std::string>::makeError(docResult.error());
  }
  yyjson_val *root = yyjson_doc_get_root(docResult.value().get());
  if (!yyjson_is_obj(root)) {
    return Result<ImportedScene, std::string>::makeError(
        "glTF root is not a JSON object");
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
  if (lightsResult.value().empty() || parsedNodes.empty()) {
    return Result<ImportedScene, std::string>::makeResult(std::move(imported));
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

  const std::vector<ParsedLightDef> parsedLights =
      std::move(lightsResult.value());
  for (uint32_t parsedNodeIndex = 0; parsedNodeIndex < parsedNodes.size();
       ++parsedNodeIndex) {
    const ParsedNode &parsedNode = parsedNodes[parsedNodeIndex];
    if (!parsedNode.lightIndex.has_value()) {
      continue;
    }
    if (*parsedNode.lightIndex >= parsedLights.size()) {
      return Result<ImportedScene, std::string>::makeError(
          "glTF punctual light index is out of range");
    }

    const uint32_t importedNodeIndex = resolveImportedLightNodeIndex(
        parsedNode, parsedNodeIndex, parsedPaths, importedPathToNode,
        importedNameToNodes, imported.nodes);
    if (importedNodeIndex == kInvalidScenePrefabIndex) {
      NURI_LOG_WARNING(
          "SceneImporter::loadSceneFromFile: skipping glTF light attachment "
          "for parsed node %u ('%s') because no unique structural node match "
          "was found",
          parsedNodeIndex, parsedNode.name.c_str());
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
    light.sourceNodeIndex = static_cast<int32_t>(parsedNodeIndex);
    imported.lights.push_back(std::move(light));
  }

  return Result<ImportedScene, std::string>::makeResult(std::move(imported));
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
  }

  prefab.meshAssets.reserve(scene.meshAssets.size());
  for (const ImportedSceneMeshAsset &meshAsset : scene.meshAssets) {
    prefab.meshAssets.push_back(ScenePrefabMeshAssetRef{
        .sourceSceneMeshIndex = meshAsset.sourceSceneMeshIndex,
        .sourceName = meshAsset.sourceName,
    });
  }

  prefab.materialAssets.reserve(scene.materialAssets.size());
  for (const ImportedSceneMaterialAsset &materialAsset : scene.materialAssets) {
    prefab.materialAssets.push_back(ScenePrefabMaterialAssetRef{
        .sourceMaterialIndex = materialAsset.sourceMaterialIndex,
        .sourceName = materialAsset.sourceName,
    });
  }

  prefab.renderables.reserve(scene.renderables.size());
  for (const ImportedSceneRenderable &renderable : scene.renderables) {
    prefab.renderables.push_back(ScenePrefabRenderable{
        .nodeIndex = renderable.nodeIndex,
        .meshIndex = renderable.meshAssetIndex,
        .materialIndex = renderable.materialAssetIndex,
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
      prefabLight.light.name = sceneLight.sourceName;
    }
    prefab.lights.push_back(std::move(prefabLight));
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
      fallback.name = materialAsset.sourceName;
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
