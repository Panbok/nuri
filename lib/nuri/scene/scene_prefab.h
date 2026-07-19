#pragma once
#include "nuri/defines.h"
#include "nuri/resources/cpu/animation_data.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_handles.h"
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <memory>
#include <memory_resource>
#include <string>
#include <vector>
namespace nuri {

inline constexpr uint32_t kInvalidScenePrefabIndex =
    std::numeric_limits<uint32_t>::max();

struct NURI_API ScenePrefabNode {
  using allocator_type = std::pmr::polymorphic_allocator<std::byte>;
  explicit ScenePrefabNode(const allocator_type &alloc = {})
      : name(alloc.resource()), morphWeights(alloc.resource()) {}
  ScenePrefabNode(const ScenePrefabNode &other,
                  const allocator_type &alloc = {})
      : parentIndex(other.parentIndex), localFromParent(other.localFromParent),
        name(other.name, alloc.resource()),
        morphWeights(other.morphWeights, alloc.resource()) {}
  ScenePrefabNode(ScenePrefabNode &&other,
                  const allocator_type &alloc = {}) noexcept
      : parentIndex(other.parentIndex), localFromParent(other.localFromParent),
        name(std::move(other.name), alloc.resource()),
        morphWeights(std::move(other.morphWeights), alloc.resource()) {}
  ScenePrefabNode &operator=(const ScenePrefabNode &) = default;
  ScenePrefabNode &operator=(ScenePrefabNode &&) noexcept = default;
  uint32_t parentIndex = kInvalidScenePrefabIndex;
  glm::mat4 localFromParent{1.0f};
  std::pmr::string name;
  std::pmr::vector<float> morphWeights;
};

struct NURI_API ScenePrefabRenderable {
  uint32_t nodeIndex = kInvalidScenePrefabIndex;
  uint32_t meshAssetIndex = kInvalidScenePrefabIndex;
  uint32_t materialAssetIndex = kInvalidScenePrefabIndex;
  uint32_t skinIndex = kInvalidScenePrefabIndex;
};

struct NURI_API ScenePrefabAssetRef {
  using allocator_type = std::pmr::polymorphic_allocator<std::byte>;
  explicit ScenePrefabAssetRef(const allocator_type &alloc = {})
      : sourceName(alloc.resource()) {}
  ScenePrefabAssetRef(const ScenePrefabAssetRef &other,
                      const allocator_type &alloc = {})
      : sourceIndex(other.sourceIndex),
        sourceName(other.sourceName, alloc.resource()) {}
  ScenePrefabAssetRef(ScenePrefabAssetRef &&other,
                      const allocator_type &alloc = {}) noexcept
      : sourceIndex(other.sourceIndex),
        sourceName(std::move(other.sourceName), alloc.resource()) {}
  ScenePrefabAssetRef &operator=(const ScenePrefabAssetRef &) = default;
  ScenePrefabAssetRef &operator=(ScenePrefabAssetRef &&) noexcept = default;
  uint32_t sourceIndex = kInvalidScenePrefabIndex;
  std::pmr::string sourceName;
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
  std::pmr::vector<ScenePrefabAssetRef> meshAssets;
  std::pmr::vector<ScenePrefabAssetRef> materialAssets;
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
