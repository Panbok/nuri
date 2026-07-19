#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/cpu/animation_data.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_prefab.h"
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory_resource>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

struct NURI_API SceneImportOptions {
  MeshImportOptions assetBuildOptions{};
  bool adaptAssetSources = false;
};

struct NURI_API AdaptedSceneMesh {
  explicit AdaptedSceneMesh(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : mesh(memory) {}
  uint32_t sourceSceneMeshIndex = kInvalidScenePrefabIndex;
  uint32_t sourceMaterialIndex = kInvalidScenePrefabIndex;
  MeshData mesh;
};

using ImportedSceneNode = ScenePrefabNode;
using ImportedSceneRenderable = ScenePrefabRenderable;
using ImportedSceneAsset = ScenePrefabAssetRef;

struct NURI_API ImportedSceneLight {
  using allocator_type = std::pmr::polymorphic_allocator<std::byte>;
  explicit ImportedSceneLight(const allocator_type &alloc = {})
      : sourceName(alloc.resource()) {}
  ImportedSceneLight(const ImportedSceneLight &other,
                     const allocator_type &alloc = {})
      : nodeIndex(other.nodeIndex), light(other.light),
        sourceName(other.sourceName, alloc.resource()),
        sourceNodeIndex(other.sourceNodeIndex) {}
  ImportedSceneLight(ImportedSceneLight &&other,
                     const allocator_type &alloc = {}) noexcept
      : nodeIndex(other.nodeIndex), light(std::move(other.light)),
        sourceName(std::move(other.sourceName), alloc.resource()),
        sourceNodeIndex(other.sourceNodeIndex) {}
  ImportedSceneLight &operator=(const ImportedSceneLight &) = default;
  ImportedSceneLight &operator=(ImportedSceneLight &&) noexcept = default;
  uint32_t nodeIndex = kInvalidScenePrefabIndex;
  LightDesc light{};
  std::pmr::string sourceName;
  uint32_t sourceNodeIndex = kInvalidScenePrefabIndex;
};

struct NURI_API ImportedLightInfo {
  LightDesc desc{};
  std::string sourceName{};
  uint32_t sourceNodeIndex = kInvalidScenePrefabIndex;
};

using ImportedLightSet = std::vector<ImportedLightInfo>;

struct NURI_API ImportedScene {
  explicit ImportedScene(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : nodes(memory), renderables(memory), meshAssets(memory),
        materialAssets(memory), lights(memory), skins(memory),
        animations(memory), rootNodes(memory), adaptedMeshes(memory),
        sourcePath(memory), sourceSceneName(memory) {}
  std::pmr::vector<ImportedSceneNode> nodes;
  std::pmr::vector<ImportedSceneRenderable> renderables;
  std::pmr::vector<ImportedSceneAsset> meshAssets;
  std::pmr::vector<ImportedSceneAsset> materialAssets;
  std::pmr::vector<ImportedSceneLight> lights;
  std::pmr::vector<SkinData> skins;
  std::pmr::vector<AnimationClipData> animations;
  std::pmr::vector<uint32_t> rootNodes;
  std::pmr::vector<AdaptedSceneMesh> adaptedMeshes;
  ImportedMaterialSet adaptedMaterials{};
  std::vector<EmbeddedSceneTextureData> embeddedTextures{};
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
  [[nodiscard]] static Result<ImportedLightSet, std::string>
  loadLightsFromFile(std::string_view path);
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
