#pragma once
#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/scene/ddgi_volume.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_handles.h"
#include "nuri/scene/scene_prefab.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

class RenderScene;

struct ScenePrefabStructureCursor {
  uint32_t nextNode = 0u;
  uint32_t nextLight = 0u;
  NodeId firstRoot = kInvalidNodeId;
  bool initialized = false;
  bool complete = false;
};

struct SceneGraphCapacityReservation {
  static constexpr uint32_t kStageCount = 4u;
  size_t nodeCapacity = 0u;
  size_t renderableCapacity = 0u;
  size_t ddgiVolumeCapacity = 0u;
  uint32_t stage = 0u;
  [[nodiscard]] bool complete() const noexcept { return stage >= kStageCount; }
  [[nodiscard]] float progress() const noexcept {
    return static_cast<float>(std::min(stage, kStageCount)) /
           static_cast<float>(kStageCount);
  }
};

class NURI_API SceneGraph {
public:
  explicit SceneGraph(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  SceneGraph(const SceneGraph &) = delete;
  SceneGraph &operator=(const SceneGraph &) = delete;
  SceneGraph(SceneGraph &&) = delete;
  SceneGraph &operator=(SceneGraph &&) = delete;
  [[nodiscard]] NodeId rootNode() const noexcept { return rootNode_; }
  [[nodiscard]] uint64_t topologyVersion() const noexcept {
    return topologyVersion_;
  }
  [[nodiscard]] uint64_t transformVersion() const noexcept {
    return transformVersion_;
  }
  void clear();
  [[nodiscard]] Result<NodeId, std::string>
  createNode(NodeId parent, std::string_view name = {},
             const glm::mat4 &localFromParent = glm::mat4(1.0f));
  [[nodiscard]] bool destroyNodeSubtree(NodeId node);
  [[nodiscard]] bool setNodeParent(NodeId node, NodeId newParent,
                                   bool preserveWorldTransform = false);
  [[nodiscard]] bool setNodeLocalTransform(NodeId node,
                                           const glm::mat4 &localFromParent);
  [[nodiscard]] bool syncWorldTransforms();
  [[nodiscard]] bool syncWorldTransformsStep(uint32_t maxNodes);
  [[nodiscard]] Result<bool, std::string>
  reserveCapacityStep(SceneGraphCapacityReservation &reservation,
                      uint32_t maxOperations);
  [[nodiscard]] bool getNodeLocalTransform(NodeId node, glm::mat4 &out) const;
  [[nodiscard]] bool getCachedNodeWorldTransform(NodeId node,
                                                 glm::mat4 &out) const;
  [[nodiscard]] bool getNodeParent(NodeId node, NodeId &out) const;
  [[nodiscard]] bool getNodeFirstChild(NodeId node, NodeId &out) const;
  [[nodiscard]] bool getNodeNextSibling(NodeId node, NodeId &out) const;
  [[nodiscard]] bool setNodeName(NodeId node, std::string_view name);
  [[nodiscard]] bool getNodeName(NodeId node, std::string_view &out) const;
  [[nodiscard]] Result<RenderableId, std::string>
  addRenderable(NodeId node, ModelRef model, MaterialRef material);
  [[nodiscard]] Result<uint32_t, std::string>
  addRenderablesInstanced(ModelRef model, MaterialRef material,
                          std::span<const glm::mat4> modelMatrices);
  [[nodiscard]] bool removeRenderable(RenderableId id);
  [[nodiscard]] bool setRenderableModel(RenderableId id, ModelRef model);
  [[nodiscard]] bool setRenderableMaterial(RenderableId id,
                                           MaterialRef material);
  [[nodiscard]] bool getRenderableModel(RenderableId id, ModelRef &out) const;
  [[nodiscard]] bool getRenderableMaterial(RenderableId id,
                                           MaterialRef &out) const;
  [[nodiscard]] bool getRenderableMaterialOverride(RenderableId id,
                                                   MaterialRef &out) const;
  [[nodiscard]] bool setRenderableMaterialOverride(RenderableId id,
                                                   MaterialRef material);
  [[nodiscard]] bool clearRenderableMaterialOverride(RenderableId id);
  [[nodiscard]] bool getRenderableNode(RenderableId id, NodeId &out) const;
  [[nodiscard]] bool setRenderableMorphWeights(RenderableId id,
                                               std::span<const float> weights);
  [[nodiscard]] std::span<const float>
  getRenderableMorphWeights(RenderableId id) const;
  [[nodiscard]] bool
  setRenderableSkinPalette(RenderableId id,
                           std::span<const glm::mat4> matrices);
  [[nodiscard]] std::span<const glm::mat4>
  getRenderableSkinPalette(RenderableId id) const;
  template <typename Fn>
  void forEachRenderableOnNode(NodeId node, Fn &&fn) const {
    if (!nodeSlotValid(node)) {
      return;
    }
    const uint32_t nodeIndex = indexOf(node);
    for (uint32_t index = nodes_.renderableHead[nodeIndex];
         index != kInvalidIndex;
         index = renderableComponents_.nextOnNode[index]) {
      fn(makeRenderableId(index,
                          renderableComponents_.slots.generation(index)));
    }
  }
  [[nodiscard]] Result<LightId, std::string> addLight(NodeId node,
                                                      const LightDesc &desc);
  [[nodiscard]] bool removeLight(LightId id);
  [[nodiscard]] bool getLightDesc(LightId id, LightDesc &outLocal) const;
  [[nodiscard]] bool getCachedLightWorldDesc(LightId id,
                                             LightDesc &outWorld) const;
  [[nodiscard]] bool getLightNode(LightId id, NodeId &out) const;
  [[nodiscard]] bool updateLight(LightId id, const LightDesc &desc);
  [[nodiscard]] bool setLightNode(LightId id, NodeId node,
                                  bool preserveWorldTransform = false);
  template <typename Fn> void forEachLightOnNode(NodeId node, Fn &&fn) const {
    if (!nodeSlotValid(node)) {
      return;
    }
    const uint32_t nodeIndex = indexOf(node);
    for (size_t typeIndex = 0; typeIndex < kLightTypeCount; ++typeIndex) {
      const auto &store = lights_[typeIndex];
      for (uint32_t index = nodes_.lightHead[typeIndex][nodeIndex];
           index != kInvalidIndex; index = store.records[index].nextOnNode) {
        fn(makeLightId(static_cast<LightType>(typeIndex), index,
                       store.slots.generation(index)));
      }
    }
  }
  [[nodiscard]] Result<DDGIVolumeId, std::string>
  addDDGIVolume(NodeId node, const DDGIVolumeDesc &desc);
  [[nodiscard]] bool removeDDGIVolume(DDGIVolumeId id);
  [[nodiscard]] bool updateDDGIVolume(DDGIVolumeId id,
                                      const DDGIVolumeDesc &desc);
  [[nodiscard]] bool getDDGIVolume(DDGIVolumeId id, DDGIVolumeDesc &out) const;
  [[nodiscard]] bool getDDGIVolumeNode(DDGIVolumeId id, NodeId &out) const;
  template <typename Fn>
  void forEachDDGIVolumeOnNode(NodeId node, Fn &&fn) const {
    if (!nodeSlotValid(node)) {
      return;
    }
    for (uint32_t index = nodes_.ddgiVolumeHead[indexOf(node)];
         index != kInvalidIndex;
         index = ddgiVolumes_.records[index].nextOnNode) {
      fn(makeDDGIVolumeId(index, ddgiVolumes_.slots.generation(index)));
    }
  }
  [[nodiscard]] Result<NodeId, std::string>
  instantiatePrefabStructure(const ScenePrefab &prefab, NodeId parent,
                             SceneInstantiationMap *outMap = nullptr);
  [[nodiscard]] Result<bool, std::string> instantiatePrefabStructureStep(
      const ScenePrefab &prefab, NodeId parent, SceneInstantiationMap &outMap,
      ScenePrefabStructureCursor &cursor, uint32_t maxOperations);
  [[nodiscard]] Result<RenderableId, std::string>
  attachPrefabRenderable(const ScenePrefab &prefab,
                         uint32_t prefabRenderableIndex, ModelRef model,
                         MaterialRef material, SceneInstantiationMap &map);
  [[nodiscard]] Result<NodeId, std::string>
  instantiatePrefab(const ScenePrefab &prefab, NodeId parent,
                    const ScenePrefabAssets &assets,
                    SceneInstantiationMap *outMap = nullptr);
  template <typename Fn> void forEachLightId(Fn &&fn) const {
    for (size_t typeIndex = 0; typeIndex < kLightTypeCount; ++typeIndex) {
      const auto &store = lights_[typeIndex];
      for (uint32_t index = 0; index < store.slots.slotCount(); ++index) {
        if (!store.slots.isLive(index)) {
          continue;
        }
        fn(makeLightId(static_cast<LightType>(typeIndex), index,
                       store.slots.generation(index)));
      }
    }
  }
  template <typename Fn> void forEachDDGIVolumeId(Fn &&fn) const {
    for (uint32_t index = 0u; index < ddgiVolumes_.slots.slotCount(); ++index) {
      if (ddgiVolumes_.slots.isLive(index)) {
        fn(makeDDGIVolumeId(index, ddgiVolumes_.slots.generation(index)));
      }
    }
  }
  void clearRenderables();
  void clearLights();
  void clearDDGIVolumes();

private:
  friend class RenderScene;
  static constexpr uint32_t kInvalidIndex =
      std::numeric_limits<uint32_t>::max();
  static constexpr size_t kLightTypeCount = 3u;
  using GenerationPool =
      SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>;
  using IndexArray = std::pmr::vector<uint32_t>;
  using LightIndexArrays = std::array<IndexArray, kLightTypeCount>;
  struct NodeStore {
    explicit NodeStore(std::pmr::memory_resource *memory)
        : slots(memory), parent(memory), firstChild(memory),
          nextSibling(memory), prevSibling(memory), depth(memory),
          localFromParent(memory), worldFromRoot(memory), dirty(memory),
          dirtyRootQueued(memory), names(memory), renderableHead(memory),
          renderableTail(memory), lightHead{std::pmr::vector<uint32_t>{memory},
                                            std::pmr::vector<uint32_t>{memory},
                                            std::pmr::vector<uint32_t>{memory}},
          lightTail{std::pmr::vector<uint32_t>{memory},
                    std::pmr::vector<uint32_t>{memory},
                    std::pmr::vector<uint32_t>{memory}},
          ddgiVolumeHead(memory), ddgiVolumeTail(memory) {}
    GenerationPool slots;
    IndexArray parent;
    IndexArray firstChild;
    IndexArray nextSibling;
    IndexArray prevSibling;
    IndexArray depth;
    std::pmr::vector<glm::mat4> localFromParent;
    std::pmr::vector<glm::mat4> worldFromRoot;
    std::pmr::vector<uint8_t> dirty;
    std::pmr::vector<uint8_t> dirtyRootQueued;
    std::pmr::vector<std::pmr::string> names;
    IndexArray renderableHead;
    IndexArray renderableTail;
    LightIndexArrays lightHead;
    LightIndexArrays lightTail;
    IndexArray ddgiVolumeHead;
    IndexArray ddgiVolumeTail;
  };
  struct RenderableStore {
    explicit RenderableStore(std::pmr::memory_resource *memory)
        : slots(memory), node(memory), models(memory), materials(memory),
          materialOverrides(memory), morphWeights(memory), skinPalette(memory),
          flatRenderableIndex(memory), nextOnNode(memory), prevOnNode(memory) {}
    GenerationPool slots;
    IndexArray node;
    std::pmr::vector<ModelRef> models;
    std::pmr::vector<MaterialRef> materials;
    std::pmr::vector<MaterialRef> materialOverrides;
    std::pmr::vector<std::pmr::vector<float>> morphWeights;
    std::pmr::vector<std::pmr::vector<glm::mat4>> skinPalette;
    IndexArray flatRenderableIndex;
    IndexArray nextOnNode;
    IndexArray prevOnNode;
  };
  struct LightRecord {
    explicit LightRecord(std::pmr::memory_resource *memory) : name(memory) {}
    uint32_t packedIndex = kInvalidIndex;
    uint32_t node = kInvalidIndex;
    std::pmr::string name;
    glm::vec3 localPosition{0.0f};
    glm::quat localRotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 color{1.0f};
    float intensity = 1.0f;
    float range = 0.0f;
    float innerConeAngle = 0.0f;
    float outerConeAngle = glm::quarter_pi<float>();
    float angularRadiusDegrees = 0.27f;
    uint32_t nextOnNode = kInvalidIndex;
    uint32_t prevOnNode = kInvalidIndex;
    bool enabled = false;
  };
  struct LightStore {
    explicit LightStore(std::pmr::memory_resource *memory)
        : slots(memory), records(memory) {}
    GenerationPool slots;
    std::pmr::vector<LightRecord> records;
  };
  struct DDGIVolumeRecord {
    explicit DDGIVolumeRecord(std::pmr::memory_resource *memory)
        : name(memory) {}
    uint32_t node = kInvalidIndex;
    std::pmr::string name;
    glm::uvec3 probeCounts{16u, 8u, 16u};
    glm::vec3 probeSpacing{2.0f};
    float blendDistance = 2.0f;
    float maxRayDistance = 20.0f;
    int32_t priority = 0;
    DDGIVolumeMode mode = DDGIVolumeMode::Authored;
    uint32_t nextOnNode = kInvalidIndex;
    uint32_t prevOnNode = kInvalidIndex;
    bool enabled = true;
  };
  struct DDGIVolumeStore {
    explicit DDGIVolumeStore(std::pmr::memory_resource *memory)
        : slots(memory), records(memory) {}
    GenerationPool slots;
    std::pmr::vector<DDGIVolumeRecord> records;
  };
  [[nodiscard]] bool nodeSlotValid(NodeId id) const noexcept;
  [[nodiscard]] bool renderableSlotValid(RenderableId id) const noexcept;
  [[nodiscard]] bool lightSlotValid(LightId id) const noexcept;
  [[nodiscard]] bool ddgiVolumeSlotValid(DDGIVolumeId id) const noexcept;
  [[nodiscard]] Result<SlotReservation, std::string> allocateNodeSlot();
  [[nodiscard]] Result<SlotReservation, std::string> allocateRenderableSlot();
  [[nodiscard]] Result<SlotReservation, std::string>
  allocateLightSlot(LightType type);
  [[nodiscard]] Result<SlotReservation, std::string> allocateDDGIVolumeSlot();
  void markSubtreeDirty(uint32_t rootIndex);
  void resetWorldTransformSync() noexcept;
  void attachNode(uint32_t childIndex, uint32_t parentIndex);
  void detachNode(uint32_t nodeIndex);
  void updateSubtreeDepth(uint32_t rootIndex);
  [[nodiscard]] bool isDescendantOf(uint32_t candidateIndex,
                                    uint32_t ancestorIndex) const;
  void noteTopologyMutation() noexcept;
  void noteTransformMutation() noexcept;
  void markTransformDependentsDirty() noexcept;
  void markDDGIVolumeTransformsDirty(uint32_t rootIndex) noexcept;
  void recycleRenderableSlot(uint32_t index) noexcept;
  [[nodiscard]] uint32_t localLightCount() const noexcept;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::pmr::vector<uint32_t> worldSyncRoots_;
  std::pmr::vector<uint32_t> worldSyncStack_;
  size_t worldSyncRootCursor_ = 0u;
  bool worldSyncPrepared_ = false;
  std::pmr::vector<uint32_t> dirtyRoots_;
  NodeStore nodes_;
  RenderableStore renderableComponents_;
  std::array<LightStore, kLightTypeCount> lights_;
  DDGIVolumeStore ddgiVolumes_;
  NodeId rootNode_ = kInvalidNodeId;
  uint64_t topologyVersion_ = 0u;
  uint64_t transformVersion_ = 0u;
  bool renderableTopologyDirty_ = false;
  bool renderableTransformsDirty_ = false;
  bool renderableDeformationsDirty_ = false;
  bool lightTopologyDirty_ = false;
  bool lightDataDirty_ = false;
  bool ddgiVolumeTopologyDirty_ = false;
  bool ddgiVolumeTransformsDirty_ = false;
  bool ddgiVolumeSettingsDirty_ = false;
};

} // namespace nuri
