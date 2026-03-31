#pragma once

#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_handles.h"
#include "nuri/scene/scene_prefab.h"

#include <cstdint>
#include <limits>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

class RenderScene;

class NURI_API SceneGraph {
public:
  // SceneGraph owns authored node/component state plus an explicit node
  // world-transform cache. Call syncWorldTransforms() before reading cached
  // world-space queries after local hierarchy edits.
  explicit SceneGraph(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~SceneGraph() = default;

  SceneGraph(const SceneGraph &) = delete;
  SceneGraph &operator=(const SceneGraph &) = delete;
  SceneGraph(SceneGraph &&) = delete;
  SceneGraph &operator=(SceneGraph &&) = delete;

  [[nodiscard]] NodeId rootNode() const noexcept { return rootNode_; }
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
    const auto visitStore = [&](const auto &store, const auto &head,
                                LightType type) {
      for (uint32_t index = head[nodeIndex]; index != kInvalidIndex;
           index = store.nextOnNode[index]) {
        fn(makeLightId(type, index, store.slots.generation(index)));
      }
    };

    visitStore(directionalLights_, nodes_.directionalLightHead,
               LightType::Directional);
    visitStore(pointLights_, nodes_.pointLightHead, LightType::Point);
    visitStore(spotLights_, nodes_.spotLightHead, LightType::Spot);
  }

  [[nodiscard]] Result<NodeId, std::string>
  instantiatePrefab(const ScenePrefab &prefab, NodeId parent,
                    const ScenePrefabAssets &assets,
                    SceneInstantiationMap *outMap = nullptr);

  template <typename Fn> void forEachLightId(Fn &&fn) const {
    const auto visitStore = [&](const auto &store, LightType type) {
      for (uint32_t index = 0; index < store.slots.slotCount(); ++index) {
        if (!store.slots.isLive(index)) {
          continue;
        }
        fn(makeLightId(type, index, store.slots.generation(index)));
      }
    };

    visitStore(directionalLights_, LightType::Directional);
    visitStore(pointLights_, LightType::Point);
    visitStore(spotLights_, LightType::Spot);
  }

  void clearRenderables();
  void clearLights();

private:
  friend class RenderScene;

  static constexpr uint32_t kInvalidIndex =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kInvalidPackedLightIndex =
      std::numeric_limits<uint32_t>::max();

  struct NodeStore {
    explicit NodeStore(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : slots(memory), parent(memory), firstChild(memory),
          nextSibling(memory), prevSibling(memory), depth(memory),
          localFromParent(memory), worldFromRoot(memory), dirty(memory),
          dirtyRootQueued(memory), names(memory), renderableHead(memory),
          renderableTail(memory), directionalLightHead(memory),
          directionalLightTail(memory), pointLightHead(memory),
          pointLightTail(memory), spotLightHead(memory), spotLightTail(memory) {
    }

    SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>
        slots;
    std::pmr::vector<uint32_t> parent;
    std::pmr::vector<uint32_t> firstChild;
    std::pmr::vector<uint32_t> nextSibling;
    std::pmr::vector<uint32_t> prevSibling;
    std::pmr::vector<uint32_t> depth;
    std::pmr::vector<glm::mat4> localFromParent;
    std::pmr::vector<glm::mat4> worldFromRoot;
    std::pmr::vector<uint8_t> dirty;
    std::pmr::vector<uint8_t> dirtyRootQueued;
    std::pmr::vector<std::pmr::string> names;
    std::pmr::vector<uint32_t> renderableHead;
    std::pmr::vector<uint32_t> renderableTail;
    std::pmr::vector<uint32_t> directionalLightHead;
    std::pmr::vector<uint32_t> directionalLightTail;
    std::pmr::vector<uint32_t> pointLightHead;
    std::pmr::vector<uint32_t> pointLightTail;
    std::pmr::vector<uint32_t> spotLightHead;
    std::pmr::vector<uint32_t> spotLightTail;
  };

  struct RenderableStore {
    explicit RenderableStore(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : slots(memory), node(memory), models(memory), materials(memory),
          materialOverrides(memory), morphWeights(memory), skinPalette(memory),
          flatRenderableIndex(memory), nextOnNode(memory), prevOnNode(memory) {}

    SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>
        slots;
    std::pmr::vector<uint32_t> node;
    std::pmr::vector<ModelRef> models;
    std::pmr::vector<MaterialRef> materials;
    std::pmr::vector<MaterialRef> materialOverrides;
    std::pmr::vector<std::pmr::vector<float>> morphWeights;
    std::pmr::vector<std::pmr::vector<glm::mat4>> skinPalette;
    std::pmr::vector<uint32_t> flatRenderableIndex;
    std::pmr::vector<uint32_t> nextOnNode;
    std::pmr::vector<uint32_t> prevOnNode;
  };

  struct DirectionalLightStore {
    explicit DirectionalLightStore(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : slots(memory), packedIndices(memory), node(memory), names(memory),
          localPositions(memory), localRotations(memory), colors(memory),
          intensities(memory), enabled(memory), nextOnNode(memory),
          prevOnNode(memory) {}

    SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>
        slots;
    std::pmr::vector<uint32_t> packedIndices;
    std::pmr::vector<uint32_t> node;
    std::pmr::vector<std::pmr::string> names;
    std::pmr::vector<glm::vec3> localPositions;
    std::pmr::vector<glm::quat> localRotations;
    std::pmr::vector<glm::vec3> colors;
    std::pmr::vector<float> intensities;
    std::pmr::vector<uint8_t> enabled;
    std::pmr::vector<uint32_t> nextOnNode;
    std::pmr::vector<uint32_t> prevOnNode;
  };

  struct PointLightStore {
    explicit PointLightStore(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : slots(memory), packedIndices(memory), node(memory), names(memory),
          localPositions(memory), localRotations(memory), colors(memory),
          intensities(memory), ranges(memory), enabled(memory),
          nextOnNode(memory), prevOnNode(memory) {}

    SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>
        slots;
    std::pmr::vector<uint32_t> packedIndices;
    std::pmr::vector<uint32_t> node;
    std::pmr::vector<std::pmr::string> names;
    std::pmr::vector<glm::vec3> localPositions;
    std::pmr::vector<glm::quat> localRotations;
    std::pmr::vector<glm::vec3> colors;
    std::pmr::vector<float> intensities;
    std::pmr::vector<float> ranges;
    std::pmr::vector<uint8_t> enabled;
    std::pmr::vector<uint32_t> nextOnNode;
    std::pmr::vector<uint32_t> prevOnNode;
  };

  struct SpotLightStore {
    explicit SpotLightStore(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : slots(memory), packedIndices(memory), node(memory), names(memory),
          localPositions(memory), localRotations(memory), colors(memory),
          intensities(memory), ranges(memory), innerConeAngles(memory),
          outerConeAngles(memory), enabled(memory), nextOnNode(memory),
          prevOnNode(memory) {}

    SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>
        slots;
    std::pmr::vector<uint32_t> packedIndices;
    std::pmr::vector<uint32_t> node;
    std::pmr::vector<std::pmr::string> names;
    std::pmr::vector<glm::vec3> localPositions;
    std::pmr::vector<glm::quat> localRotations;
    std::pmr::vector<glm::vec3> colors;
    std::pmr::vector<float> intensities;
    std::pmr::vector<float> ranges;
    std::pmr::vector<float> innerConeAngles;
    std::pmr::vector<float> outerConeAngles;
    std::pmr::vector<uint8_t> enabled;
    std::pmr::vector<uint32_t> nextOnNode;
    std::pmr::vector<uint32_t> prevOnNode;
  };

  [[nodiscard]] bool nodeSlotValid(NodeId id) const noexcept;
  [[nodiscard]] bool renderableSlotValid(RenderableId id) const noexcept;
  [[nodiscard]] bool directionalSlotValid(LightId id) const noexcept;
  [[nodiscard]] bool pointSlotValid(LightId id) const noexcept;
  [[nodiscard]] bool spotSlotValid(LightId id) const noexcept;

  [[nodiscard]] Result<SlotReservation, std::string> allocateNodeSlot();
  [[nodiscard]] Result<SlotReservation, std::string> allocateRenderableSlot();
  [[nodiscard]] Result<SlotReservation, std::string>
  allocateDirectionalLightSlot();
  [[nodiscard]] Result<SlotReservation, std::string> allocatePointLightSlot();
  [[nodiscard]] Result<SlotReservation, std::string> allocateSpotLightSlot();

  void markSubtreeDirty(uint32_t rootIndex);
  void attachNode(uint32_t childIndex, uint32_t parentIndex);
  void detachNode(uint32_t nodeIndex);
  void updateSubtreeDepth(uint32_t rootIndex);
  [[nodiscard]] bool isDescendantOf(uint32_t candidateIndex,
                                    uint32_t ancestorIndex) const;
  void markTransformDependentsDirty() noexcept;
  void recycleRenderableSlot(uint32_t index) noexcept;
  [[nodiscard]] uint32_t localLightCount() const noexcept;
  [[nodiscard]] bool tryGetLightNodeIndex(LightId id,
                                          uint32_t &outNodeIndex) const;

  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::pmr::vector<uint32_t> dirtyRoots_;
  NodeStore nodes_;
  RenderableStore renderableComponents_;
  DirectionalLightStore directionalLights_;
  PointLightStore pointLights_;
  SpotLightStore spotLights_;
  NodeId rootNode_ = kInvalidNodeId;
  bool renderableTopologyDirty_ = false;
  bool renderableTransformsDirty_ = false;
  bool lightTopologyDirty_ = false;
  bool lightDataDirty_ = false;
};

} // namespace nuri
