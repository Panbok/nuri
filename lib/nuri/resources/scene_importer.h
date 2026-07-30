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
#include <memory>
#include <memory_resource>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

struct NURI_API SceneImportOptions {
  MeshImportOptions assetBuildOptions{};
};

struct NURI_API ImportedLightInfo {
  LightDesc desc{};
  std::string sourceName{};
  uint32_t sourceNodeIndex = kInvalidScenePrefabIndex;
};

using ImportedLightSet = std::vector<ImportedLightInfo>;

struct NURI_API ScenePrefabAdaptedMesh {
  explicit ScenePrefabAdaptedMesh(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : mesh(memory) {}
  uint32_t sourceSceneMeshIndex = kInvalidScenePrefabIndex;
  uint32_t sourceMaterialIndex = kInvalidScenePrefabIndex;
  MeshData mesh;
};

struct NURI_API SceneImportProduct {
  ScenePrefab prefab{};
  std::vector<ScenePrefabAdaptedMesh> meshes{};
  std::vector<MaterialData> materials{};
  std::shared_ptr<const std::vector<EmbeddedSceneTextureData>>
      embeddedTextures{};
};

class NURI_API SceneImporter {
public:
  [[nodiscard]] static Result<ImportedLightSet, std::string>
  loadLightsFromFile(std::string_view path);
  [[nodiscard]] static Result<SceneImportProduct, std::string>
  loadSceneFromFile(
      std::string_view path, const SceneImportOptions &options = {},
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
};

} // namespace nuri
