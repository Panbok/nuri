#pragma once

#include "nuri/defines.h"
#include "nuri/resources/cpu/animation_data.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_handles.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

inline constexpr uint32_t kInvalidScenePrefabIndex =
    std::numeric_limits<uint32_t>::max();

struct NURI_API ScenePrefabNode {
  using allocator_type = std::pmr::polymorphic_allocator<>;

  uint32_t parentIndex = kInvalidScenePrefabIndex;
  glm::mat4 localFromParent{1.0f};
  std::pmr::string name;
  std::pmr::vector<float> morphWeights;

  explicit ScenePrefabNode(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : name(memory), morphWeights(memory) {}

  template <typename T>
  ScenePrefabNode(std::pmr::memory_resource *memory,
                  const std::pmr::polymorphic_allocator<T> &alloc)
      : name(memory != nullptr ? memory : alloc.resource()),
        morphWeights(memory != nullptr ? memory : alloc.resource()) {}

  template <typename T>
  explicit ScenePrefabNode(const std::pmr::polymorphic_allocator<T> &alloc)
      : name(alloc.resource()), morphWeights(alloc.resource()) {}

  template <typename T>
  ScenePrefabNode(std::allocator_arg_t,
                  const std::pmr::polymorphic_allocator<T> &alloc,
                  std::pmr::memory_resource *memory)
      : name(memory != nullptr ? memory : alloc.resource()),
        morphWeights(memory != nullptr ? memory : alloc.resource()) {}

  template <typename T>
  ScenePrefabNode(std::allocator_arg_t,
                  const std::pmr::polymorphic_allocator<T> &alloc)
      : name(alloc.resource()), morphWeights(alloc.resource()) {}

  template <typename T>
  ScenePrefabNode(const ScenePrefabNode &other,
                  const std::pmr::polymorphic_allocator<T> &alloc)
      : parentIndex(other.parentIndex), localFromParent(other.localFromParent),
        name(other.name, alloc.resource()),
        morphWeights(other.morphWeights, alloc.resource()) {}

  template <typename T>
  ScenePrefabNode(ScenePrefabNode &&other,
                  const std::pmr::polymorphic_allocator<T> &alloc)
      : parentIndex(other.parentIndex), localFromParent(other.localFromParent),
        name(std::move(other.name), alloc.resource()),
        morphWeights(std::move(other.morphWeights), alloc.resource()) {}

  template <typename T>
  ScenePrefabNode(std::allocator_arg_t,
                  const std::pmr::polymorphic_allocator<T> &alloc,
                  const ScenePrefabNode &other)
      : parentIndex(other.parentIndex), localFromParent(other.localFromParent),
        name(other.name, alloc.resource()),
        morphWeights(other.morphWeights, alloc.resource()) {}

  template <typename T>
  ScenePrefabNode(std::allocator_arg_t,
                  const std::pmr::polymorphic_allocator<T> &alloc,
                  ScenePrefabNode &&other)
      : parentIndex(other.parentIndex), localFromParent(other.localFromParent),
        name(std::move(other.name), alloc.resource()),
        morphWeights(std::move(other.morphWeights), alloc.resource()) {}
};

struct NURI_API ScenePrefabRenderable {
  uint32_t nodeIndex = kInvalidScenePrefabIndex;
  // These indices address ScenePrefab::meshAssets/materialAssets rather than
  // raw source-scene ordinals.
  uint32_t meshIndex = kInvalidScenePrefabIndex;
  uint32_t materialIndex = kInvalidScenePrefabIndex;
  uint32_t skinIndex = kInvalidScenePrefabIndex;
};

struct NURI_API ScenePrefabMeshAssetRef {
  uint32_t sourceSceneMeshIndex = kInvalidScenePrefabIndex;
  std::string sourceName{};
};

struct NURI_API ScenePrefabMaterialAssetRef {
  uint32_t sourceMaterialIndex = kInvalidScenePrefabIndex;
  std::string sourceName{};
};

struct NURI_API ScenePrefabLight {
  uint32_t nodeIndex = kInvalidScenePrefabIndex;
  LightDesc light{};
};

struct NURI_API ScenePrefab {
  explicit ScenePrefab(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : nodes(memory), renderables(memory), meshAssets(memory),
        materialAssets(memory), lights(memory), skins(memory),
        animations(memory), sourcePath(memory), sourceSceneName(memory) {}

  std::pmr::vector<ScenePrefabNode> nodes;
  std::pmr::vector<ScenePrefabRenderable> renderables;
  std::pmr::vector<ScenePrefabMeshAssetRef> meshAssets;
  std::pmr::vector<ScenePrefabMaterialAssetRef> materialAssets;
  std::pmr::vector<ScenePrefabLight> lights;
  std::pmr::vector<SkinData> skins;
  std::pmr::vector<AnimationClipData> animations;
  std::pmr::string sourcePath;
  std::pmr::string sourceSceneName;
  MeshImportOptions importOptions{};
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

namespace std {

template <typename T>
struct uses_allocator<nuri::ScenePrefabNode, std::pmr::polymorphic_allocator<T>>
    : true_type {};

} // namespace std
