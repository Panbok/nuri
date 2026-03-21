#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_prefab.h"

#include <cstdint>
#include <memory_resource>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

struct NURI_API SceneImportOptions {
  MeshImportOptions assetBuildOptions{};
};

struct NURI_API ImportedSceneNode {
  uint32_t parentIndex = kInvalidScenePrefabIndex;
  glm::mat4 localFromParent{1.0f};
  std::string name{};
};

struct NURI_API ImportedSceneRenderable {
  uint32_t nodeIndex = kInvalidScenePrefabIndex;
  uint32_t meshAssetIndex = kInvalidScenePrefabIndex;
  uint32_t materialAssetIndex = kInvalidScenePrefabIndex;
};

struct NURI_API ImportedSceneMeshAsset {
  uint32_t sourceSceneMeshIndex = kInvalidScenePrefabIndex;
  std::string sourceName{};
};

struct NURI_API ImportedSceneMaterialAsset {
  uint32_t sourceMaterialIndex = kInvalidScenePrefabIndex;
  std::string sourceName{};
};

struct NURI_API ImportedSceneLight {
  uint32_t nodeIndex = kInvalidScenePrefabIndex;
  LightDesc light{};
  std::string sourceName{};
  int32_t sourceNodeIndex = -1;
};

struct NURI_API ImportedScene {
  explicit ImportedScene(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : nodes(memory), renderables(memory), meshAssets(memory),
        materialAssets(memory), lights(memory), rootNodes(memory),
        sourcePath(memory), sourceSceneName(memory) {}

  std::pmr::vector<ImportedSceneNode> nodes;
  std::pmr::vector<ImportedSceneRenderable> renderables;
  std::pmr::vector<ImportedSceneMeshAsset> meshAssets;
  std::pmr::vector<ImportedSceneMaterialAsset> materialAssets;
  std::pmr::vector<ImportedSceneLight> lights;
  std::pmr::vector<uint32_t> rootNodes;
  std::pmr::string sourcePath;
  std::pmr::string sourceSceneName;
  MeshImportOptions importOptions{};
};

struct NURI_API ImportedSceneAssets {
  explicit ImportedSceneAssets(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : meshes(memory) {}

  std::pmr::vector<MeshData> meshes;
  ImportedMaterialSet materials{};
};

class NURI_API SceneImporter {
public:
  [[nodiscard]] static Result<ImportedScene, std::string> loadSceneFromFile(
      std::string_view path, const SceneImportOptions &options = {},
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());

  [[nodiscard]] static Result<ScenePrefab, std::string> buildScenePrefab(
      const ImportedScene &scene,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  [[nodiscard]] static Result<ImportedSceneAssets, std::string>
  buildSceneAssets(
      const ImportedScene &scene,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());

  [[nodiscard]] static Result<ScenePrefab, std::string> loadScenePrefabFromFile(
      std::string_view path, const SceneImportOptions &options = {},
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  [[nodiscard]] static Result<ImportedSceneAssets, std::string>
  loadSceneAssetsFromFile(
      std::string_view path, const SceneImportOptions &options = {},
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
};

} // namespace nuri
