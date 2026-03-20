#pragma once

#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_handles.h"

#include <cstdint>
#include <limits>
#include <memory_resource>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

inline constexpr uint32_t kInvalidScenePrefabIndex =
    std::numeric_limits<uint32_t>::max();

struct NURI_API ScenePrefabNode {
  uint32_t parentIndex = kInvalidScenePrefabIndex;
  glm::mat4 localFromParent{1.0f};
  std::pmr::string name;

  explicit ScenePrefabNode(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : name(memory) {}
};

struct NURI_API ScenePrefabRenderable {
  uint32_t nodeIndex = kInvalidScenePrefabIndex;
  uint32_t meshIndex = kInvalidScenePrefabIndex;
  uint32_t materialIndex = kInvalidScenePrefabIndex;
};

struct NURI_API ScenePrefabLight {
  uint32_t nodeIndex = kInvalidScenePrefabIndex;
  LightDesc light{};
};

struct NURI_API ScenePrefab {
  explicit ScenePrefab(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : nodes(memory), renderables(memory), lights(memory), sourcePath(memory),
        sourceSceneName(memory) {}

  std::pmr::vector<ScenePrefabNode> nodes;
  std::pmr::vector<ScenePrefabRenderable> renderables;
  std::pmr::vector<ScenePrefabLight> lights;
  std::pmr::string sourcePath;
  std::pmr::string sourceSceneName;
  MeshImportOptions importOptions{};
  uint32_t meshCount = 0;
  uint32_t materialCount = 0;
};

struct NURI_API ScenePrefabAssets {
  explicit ScenePrefabAssets(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : models(memory), materials(memory) {}

  std::pmr::vector<ModelRef> models;
  std::pmr::vector<MaterialRef> materials;
};

struct NURI_API SceneInstantiationMap {
  explicit SceneInstantiationMap(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : nodes(memory), renderables(memory), lights(memory) {}

  std::pmr::vector<NodeId> nodes;
  std::pmr::vector<RenderableId> renderables;
  std::pmr::vector<LightId> lights;
};

} // namespace nuri
