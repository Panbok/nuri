#include "nuri/scene/scene_graph.h"
#include "nuri/core/profiling.h"
#include "nuri/math/light.h"
#include "nuri/math/utils.h"
#include "nuri/pch.h"
namespace nuri {
namespace {
constexpr uint32_t kMaxDirectionalLightCount = 4u;
constexpr uint32_t kMaxLocalLightCount = 64u;
constexpr uint32_t kInvalidSceneGraphIndex =
    std::numeric_limits<uint32_t>::max();
[[nodiscard]] Result<SlotReservation, std::string>
makePackedSlotOverflowError(std::string_view context) {
  return Result<SlotReservation, std::string>::makeError(
      std::string(context) + ": slot pool exhausted");
}
template <typename Pool>
[[nodiscard]] bool slotPoolExhausted(const Pool &pool,
                                     uint32_t maxIndex) noexcept {
  const uint32_t slotCount = pool.slotCount();
  return slotCount > maxIndex + 1u ||
         (slotCount == maxIndex + 1u && pool.liveCount() == slotCount);
}
template <typename... Containers>
void reserveAll(size_t capacity, Containers &...containers) {
  (containers.reserve(capacity), ...);
}
template <typename Store>
[[nodiscard]] auto &componentNode(Store &store, uint32_t index) {
  if constexpr (requires { store.records; }) {
    return store.records[index].node;
  } else {
    return store.node[index];
  }
}
template <typename Store>
[[nodiscard]] auto &componentNext(Store &store, uint32_t index) {
  if constexpr (requires { store.records; }) {
    return store.records[index].nextOnNode;
  } else {
    return store.nextOnNode[index];
  }
}
template <typename Store>
[[nodiscard]] auto &componentPrevious(Store &store, uint32_t index) {
  if constexpr (requires { store.records; }) {
    return store.records[index].prevOnNode;
  } else {
    return store.prevOnNode[index];
  }
}
template <typename Store>
void attachComponentToNode(Store &store, std::pmr::vector<uint32_t> &head,
                           std::pmr::vector<uint32_t> &tail,
                           uint32_t componentIndex, uint32_t nodeIndex) {
  componentNode(store, componentIndex) = nodeIndex;
  componentPrevious(store, componentIndex) = tail[nodeIndex];
  componentNext(store, componentIndex) = kInvalidSceneGraphIndex;
  if (tail[nodeIndex] != kInvalidSceneGraphIndex) {
    componentNext(store, tail[nodeIndex]) = componentIndex;
  } else {
    head[nodeIndex] = componentIndex;
  }
  tail[nodeIndex] = componentIndex;
}
template <typename Store>
void detachComponentFromNode(Store &store, std::pmr::vector<uint32_t> &head,
                             std::pmr::vector<uint32_t> &tail,
                             uint32_t componentIndex) {
  const uint32_t nodeIndex = componentNode(store, componentIndex);
  if (nodeIndex == kInvalidSceneGraphIndex || nodeIndex >= head.size()) {
    componentNode(store, componentIndex) = kInvalidSceneGraphIndex;
    componentPrevious(store, componentIndex) = kInvalidSceneGraphIndex;
    componentNext(store, componentIndex) = kInvalidSceneGraphIndex;
    return;
  }
  const uint32_t prevIndex = componentPrevious(store, componentIndex);
  const uint32_t nextIndex = componentNext(store, componentIndex);
  if (prevIndex != kInvalidSceneGraphIndex) {
    componentNext(store, prevIndex) = nextIndex;
  } else {
    head[nodeIndex] = nextIndex;
  }
  if (nextIndex != kInvalidSceneGraphIndex) {
    componentPrevious(store, nextIndex) = prevIndex;
  } else {
    tail[nodeIndex] = prevIndex;
  }
  componentNode(store, componentIndex) = kInvalidSceneGraphIndex;
  componentPrevious(store, componentIndex) = kInvalidSceneGraphIndex;
  componentNext(store, componentIndex) = kInvalidSceneGraphIndex;
}
template <typename Store>
void recycleLightSlot(Store &store, std::pmr::vector<uint32_t> &head,
                      std::pmr::vector<uint32_t> &tail, uint32_t lightIndex) {
  detachComponentFromNode(store, head, tail, lightIndex);
  store.slots.release(lightIndex);
}
template <typename Record>
void writeLightRecord(Record &record, const LightDesc &desc) {
  record.name.assign(desc.name.data(), desc.name.size());
  record.localPosition = desc.position;
  record.localRotation = desc.rotation;
  record.color = desc.color;
  record.intensity = desc.intensity;
  record.range = desc.range;
  record.innerConeAngle = desc.innerConeAngleRadians;
  record.outerConeAngle = desc.outerConeAngleRadians;
  record.angularRadiusDegrees = desc.angularRadiusDegrees;
  record.enabled = desc.enabled;
}
template <typename Record>
[[nodiscard]] bool lightRecordDataEqual(const Record &record,
                                        const LightDesc &desc) {
  return nuri::vec3ExactEqual(record.localPosition, desc.position) &&
         nuri::quatExactEqual(record.localRotation, desc.rotation) &&
         nuri::vec3ExactEqual(record.color, desc.color) &&
         record.intensity == desc.intensity && record.range == desc.range &&
         record.innerConeAngle == desc.innerConeAngleRadians &&
         record.outerConeAngle == desc.outerConeAngleRadians &&
         record.angularRadiusDegrees == desc.angularRadiusDegrees;
}
template <typename Record>
void writeDDGIVolumeRecord(Record &record, const DDGIVolumeDesc &desc) {
  record.name.assign(desc.name.data(), desc.name.size());
  record.probeCounts = desc.probeCounts;
  record.probeSpacing = desc.probeSpacing;
  record.blendDistance = desc.blendDistance;
  record.maxRayDistance = desc.maxRayDistance;
  record.priority = desc.priority;
  record.mode = desc.mode;
  record.enabled = desc.enabled;
}
template <typename Record>
[[nodiscard]] DDGIVolumeDesc makeDDGIVolumeDesc(const Record &record) {
  return DDGIVolumeDesc{
      .name = std::string(record.name),
      .probeCounts = record.probeCounts,
      .probeSpacing = record.probeSpacing,
      .blendDistance = record.blendDistance,
      .maxRayDistance = record.maxRayDistance,
      .priority = record.priority,
      .mode = record.mode,
      .enabled = record.enabled,
  };
}
template <typename Record>
[[nodiscard]] bool ddgiVolumeRecordEqual(const Record &record,
                                         const DDGIVolumeDesc &desc) {
  return std::string_view(record.name) == std::string_view(desc.name) &&
         record.probeCounts == desc.probeCounts &&
         nuri::vec3ExactEqual(record.probeSpacing, desc.probeSpacing) &&
         record.blendDistance == desc.blendDistance &&
         record.maxRayDistance == desc.maxRayDistance &&
         record.priority == desc.priority && record.mode == desc.mode &&
         record.enabled == desc.enabled;
}
template <typename Pool, typename Values, typename Id, typename Value>
[[nodiscard]] bool readSlotValue(const Pool &slots, const Values &values, Id id,
                                 Value &out) {
  if (!isValid(id) || !slots.isValid(indexOf(id), generationOf(id))) {
    return false;
  }
  out = values[indexOf(id)];
  return true;
}
template <typename Pool, typename Values, typename Id, typename Value>
[[nodiscard]] bool writeSlotRef(const Pool &slots, Values &values, Id id,
                                Value value, bool &dirty) {
  if (!isValid(id) || !isValid(value) ||
      !slots.isValid(indexOf(id), generationOf(id))) {
    return false;
  }
  auto &current = values[indexOf(id)];
  if (current.value != value.value) {
    current = value;
    dirty = true;
  }
  return true;
}
template <typename Pool, typename Links>
[[nodiscard]] bool readNodeLink(const Pool &slots, const Links &links,
                                NodeId node, NodeId &out) {
  if (!isValid(node) || !slots.isValid(indexOf(node), generationOf(node))) {
    return false;
  }
  const uint32_t linkedIndex = links[indexOf(node)];
  out = linkedIndex == kInvalidSceneGraphIndex
            ? kInvalidNodeId
            : makeNodeId(linkedIndex, slots.generation(linkedIndex));
  return true;
}
template <typename Pool, typename Storage, typename Id, typename Value,
          typename Equal = std::equal_to<>>
[[nodiscard]] bool writeSlotRange(const Pool &slots, Storage &values, Id id,
                                  std::span<const Value> input, bool &dirty,
                                  Equal equal = {}) {
  if (!isValid(id) || !slots.isValid(indexOf(id), generationOf(id))) {
    return false;
  }
  auto &current = values[indexOf(id)];
  if (current.size() != input.size() ||
      !std::equal(current.begin(), current.end(), input.begin(), input.end(),
                  equal)) {
    current.assign(input.begin(), input.end());
    dirty = true;
  }
  return true;
}
template <typename Value, typename Pool, typename Storage, typename Id>
[[nodiscard]] std::span<const Value>
readSlotRange(const Pool &slots, const Storage &values, Id id) {
  if (!isValid(id) || !slots.isValid(indexOf(id), generationOf(id))) {
    return {};
  }
  const auto &current = values[indexOf(id)];
  return {current.data(), current.size()};
}
} // namespace

SceneGraph::SceneGraph(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      worldSyncRoots_(memory_), worldSyncStack_(memory_), dirtyRoots_(memory_),
      nodes_(memory_), renderableComponents_(memory_),
      lights_{LightStore(memory_), LightStore(memory_), LightStore(memory_)},
      ddgiVolumes_(memory_) {
  clear();
}

void SceneGraph::clear() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  dirtyRoots_.clear();
  resetWorldTransformSync();
  nodes_ = NodeStore(memory_);
  renderableComponents_ = RenderableStore(memory_);
  for (auto &store : lights_) {
    store = LightStore(memory_);
  }
  ddgiVolumes_ = DDGIVolumeStore(memory_);
  renderableTopologyDirty_ = true;
  renderableTransformsDirty_ = false;
  renderableDeformationsDirty_ = false;
  lightTopologyDirty_ = true;
  lightDataDirty_ = false;
  ddgiVolumeTopologyDirty_ = true;
  ddgiVolumeTransformsDirty_ = false;
  ddgiVolumeSettingsDirty_ = false;
  topologyVersion_ = 0u;
  transformVersion_ = 0u;
  const auto rootResult = allocateNodeSlot();
  const SlotReservation root = rootResult.value();
  const uint32_t rootIndex = root.index;
  nodes_.names[rootIndex] = "Root";
  rootNode_ = makeNodeId(rootIndex, root.generation);
}

Result<bool, std::string>
SceneGraph::reserveCapacityStep(SceneGraphCapacityReservation &reservation,
                                uint32_t maxOperations) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (reservation.nodeCapacity > kResourceHandleIndexMask ||
      reservation.renderableCapacity > kResourceHandleIndexMask ||
      reservation.ddgiVolumeCapacity > kResourceHandleIndexMask) {
    return Result<bool, std::string>::makeError(
        "SceneGraph::reserveCapacityStep: requested capacity exceeds handle "
        "index range");
  }
  const uint32_t nodeCapacity = static_cast<uint32_t>(reservation.nodeCapacity);
  const uint32_t renderableCapacity =
      static_cast<uint32_t>(reservation.renderableCapacity);
  const uint32_t ddgiVolumeCapacity =
      static_cast<uint32_t>(reservation.ddgiVolumeCapacity);
  const uint32_t operationLimit = std::max(maxOperations, 1u);
  uint32_t operations = 0u;
  try {
    while (reservation.stage < SceneGraphCapacityReservation::kStageCount &&
           operations < operationLimit) {
      switch (reservation.stage) {
      case 0u:
        reserveAll(nodeCapacity, dirtyRoots_, worldSyncRoots_, worldSyncStack_);
        break;
      case 1u:
        reserveAll(nodeCapacity, nodes_.slots, nodes_.parent, nodes_.firstChild,
                   nodes_.nextSibling, nodes_.prevSibling, nodes_.depth,
                   nodes_.localFromParent, nodes_.worldFromRoot, nodes_.dirty,
                   nodes_.dirtyRootQueued, nodes_.names, nodes_.renderableHead,
                   nodes_.renderableTail, nodes_.ddgiVolumeHead,
                   nodes_.ddgiVolumeTail);
        for (size_t typeIndex = 0; typeIndex < kLightTypeCount; ++typeIndex) {
          reserveAll(nodeCapacity, nodes_.lightHead[typeIndex],
                     nodes_.lightTail[typeIndex]);
        }
        break;
      case 2u:
        reserveAll(renderableCapacity, renderableComponents_.slots,
                   renderableComponents_.node, renderableComponents_.models,
                   renderableComponents_.materials,
                   renderableComponents_.materialOverrides,
                   renderableComponents_.morphWeights,
                   renderableComponents_.skinPalette,
                   renderableComponents_.flatRenderableIndex,
                   renderableComponents_.nextOnNode,
                   renderableComponents_.prevOnNode);
        break;
      case 3u:
        reserveAll(ddgiVolumeCapacity, ddgiVolumes_.slots,
                   ddgiVolumes_.records);
        break;
      }
      ++reservation.stage;
      ++operations;
    }
  } catch (const std::bad_alloc &) {
    return Result<bool, std::string>::makeError(
        "SceneGraph::reserveCapacityStep: backing-array allocation failed");
  }
  return Result<bool, std::string>::makeResult(reservation.complete());
}

Result<SlotReservation, std::string> SceneGraph::allocateNodeSlot() {
  if (slotPoolExhausted(nodes_.slots, kResourceHandleIndexMask)) {
    return makePackedSlotOverflowError("SceneGraph::allocateNodeSlot");
  }
  const SlotReservation slot = nodes_.slots.acquire();
  if (slot.appended) {
    nodes_.parent.push_back(kInvalidIndex);
    nodes_.firstChild.push_back(kInvalidIndex);
    nodes_.nextSibling.push_back(kInvalidIndex);
    nodes_.prevSibling.push_back(kInvalidIndex);
    nodes_.depth.push_back(0u);
    nodes_.localFromParent.push_back(glm::mat4(1.0f));
    nodes_.worldFromRoot.push_back(glm::mat4(1.0f));
    nodes_.dirty.push_back(0u);
    nodes_.dirtyRootQueued.push_back(0u);
    nodes_.names.emplace_back();
    nodes_.renderableHead.push_back(kInvalidIndex);
    nodes_.renderableTail.push_back(kInvalidIndex);
    nodes_.ddgiVolumeHead.push_back(kInvalidIndex);
    nodes_.ddgiVolumeTail.push_back(kInvalidIndex);
    for (size_t typeIndex = 0; typeIndex < kLightTypeCount; ++typeIndex) {
      nodes_.lightHead[typeIndex].push_back(kInvalidIndex);
      nodes_.lightTail[typeIndex].push_back(kInvalidIndex);
    }
  }
  return Result<SlotReservation, std::string>::makeResult(slot);
}

Result<SlotReservation, std::string> SceneGraph::allocateRenderableSlot() {
  if (slotPoolExhausted(renderableComponents_.slots,
                        kResourceHandleIndexMask)) {
    return makePackedSlotOverflowError("SceneGraph::allocateRenderableSlot");
  }
  const SlotReservation slot = renderableComponents_.slots.acquire();
  if (slot.appended) {
    renderableComponents_.node.push_back(kInvalidIndex);
    renderableComponents_.models.push_back(kInvalidModelRef);
    renderableComponents_.materials.push_back(kInvalidMaterialRef);
    renderableComponents_.materialOverrides.push_back(kInvalidMaterialRef);
    renderableComponents_.morphWeights.emplace_back();
    renderableComponents_.skinPalette.emplace_back();
    renderableComponents_.flatRenderableIndex.push_back(kInvalidIndex);
    renderableComponents_.nextOnNode.push_back(kInvalidIndex);
    renderableComponents_.prevOnNode.push_back(kInvalidIndex);
  }
  return Result<SlotReservation, std::string>::makeResult(slot);
}

Result<SlotReservation, std::string>
SceneGraph::allocateLightSlot(LightType type) {
  auto &store = lights_[static_cast<size_t>(type)];
  if (slotPoolExhausted(store.slots, kResourceHandleIndexMask)) {
    return makePackedSlotOverflowError("SceneGraph::allocateLightSlot");
  }
  const SlotReservation slot = store.slots.acquire();
  if (slot.appended) {
    store.records.emplace_back(memory_);
  }
  return Result<SlotReservation, std::string>::makeResult(slot);
}

Result<SlotReservation, std::string> SceneGraph::allocateDDGIVolumeSlot() {
  if (slotPoolExhausted(ddgiVolumes_.slots, kResourceHandleIndexMask)) {
    return makePackedSlotOverflowError("SceneGraph::allocateDDGIVolumeSlot");
  }
  const SlotReservation slot = ddgiVolumes_.slots.acquire();
  if (slot.appended) {
    ddgiVolumes_.records.emplace_back(memory_);
  }
  return Result<SlotReservation, std::string>::makeResult(slot);
}

bool SceneGraph::nodeSlotValid(NodeId id) const noexcept {
  if (!isValid(id)) {
    return false;
  }
  return nodes_.slots.isValid(indexOf(id), generationOf(id));
}

bool SceneGraph::renderableSlotValid(RenderableId id) const noexcept {
  if (!isValid(id)) {
    return false;
  }
  return renderableComponents_.slots.isValid(indexOf(id), generationOf(id));
}

bool SceneGraph::lightSlotValid(LightId id) const noexcept {
  const size_t typeIndex = static_cast<size_t>(id.type);
  if (!isValid(id) || typeIndex >= kLightTypeCount) {
    return false;
  }
  return lights_[typeIndex].slots.isValid(indexOf(id), generationOf(id));
}

bool SceneGraph::ddgiVolumeSlotValid(DDGIVolumeId id) const noexcept {
  return isValid(id) &&
         ddgiVolumes_.slots.isValid(indexOf(id), generationOf(id));
}

void SceneGraph::attachNode(uint32_t childIndex, uint32_t parentIndex) {
  nodes_.parent[childIndex] = parentIndex;
  nodes_.prevSibling[childIndex] = kInvalidIndex;
  nodes_.nextSibling[childIndex] = nodes_.firstChild[parentIndex];
  if (nodes_.firstChild[parentIndex] != kInvalidIndex) {
    nodes_.prevSibling[nodes_.firstChild[parentIndex]] = childIndex;
  }
  nodes_.firstChild[parentIndex] = childIndex;
  nodes_.depth[childIndex] = nodes_.depth[parentIndex] + 1u;
}

void SceneGraph::detachNode(uint32_t nodeIndex) {
  const uint32_t parentIndex = nodes_.parent[nodeIndex];
  if (parentIndex == kInvalidIndex) {
    return;
  }
  const uint32_t prevIndex = nodes_.prevSibling[nodeIndex];
  const uint32_t nextIndex = nodes_.nextSibling[nodeIndex];
  if (prevIndex != kInvalidIndex) {
    nodes_.nextSibling[prevIndex] = nextIndex;
  } else {
    nodes_.firstChild[parentIndex] = nextIndex;
  }
  if (nextIndex != kInvalidIndex) {
    nodes_.prevSibling[nextIndex] = prevIndex;
  }
  nodes_.parent[nodeIndex] = kInvalidIndex;
  nodes_.prevSibling[nodeIndex] = kInvalidIndex;
  nodes_.nextSibling[nodeIndex] = kInvalidIndex;
}

void SceneGraph::updateSubtreeDepth(uint32_t rootIndex) {
  std::pmr::vector<uint32_t> stack(memory_);
  stack.push_back(rootIndex);
  while (!stack.empty()) {
    const uint32_t nodeIndex = stack.back();
    stack.pop_back();
    for (uint32_t child = nodes_.firstChild[nodeIndex]; child != kInvalidIndex;
         child = nodes_.nextSibling[child]) {
      nodes_.depth[child] = nodes_.depth[nodeIndex] + 1u;
      stack.push_back(child);
    }
  }
}

bool SceneGraph::isDescendantOf(uint32_t candidateIndex,
                                uint32_t ancestorIndex) const {
  for (uint32_t current = candidateIndex; current != kInvalidIndex;
       current = nodes_.parent[current]) {
    if (current == ancestorIndex) {
      return true;
    }
  }
  return false;
}

void SceneGraph::noteTopologyMutation() noexcept { ++topologyVersion_; }

void SceneGraph::noteTransformMutation() noexcept { ++transformVersion_; }

void SceneGraph::markTransformDependentsDirty() noexcept {
  renderableTransformsDirty_ = true;
  lightDataDirty_ = true;
}

void SceneGraph::markDDGIVolumeTransformsDirty(uint32_t rootIndex) noexcept {
  for (uint32_t index = 0u; index < ddgiVolumes_.slots.slotCount(); ++index) {
    if (!ddgiVolumes_.slots.isLive(index)) {
      continue;
    }
    if (isDescendantOf(ddgiVolumes_.records[index].node, rootIndex)) {
      ddgiVolumeTransformsDirty_ = true;
      return;
    }
  }
}

void SceneGraph::recycleRenderableSlot(uint32_t index) noexcept {
  detachComponentFromNode(renderableComponents_, nodes_.renderableHead,
                          nodes_.renderableTail, index);
  renderableComponents_.morphWeights[index].clear();
  renderableComponents_.skinPalette[index].clear();
  renderableComponents_.slots.release(index);
}

uint32_t SceneGraph::localLightCount() const noexcept {
  return lights_[static_cast<size_t>(LightType::Point)].slots.liveCount() +
         lights_[static_cast<size_t>(LightType::Spot)].slots.liveCount();
}

void SceneGraph::markSubtreeDirty(uint32_t rootIndex) {
  resetWorldTransformSync();
  bool coveredByDirtyAncestor = false;
  for (uint32_t ancestor = nodes_.parent[rootIndex]; ancestor != kInvalidIndex;
       ancestor = nodes_.parent[ancestor]) {
    if (nodes_.dirtyRootQueued[ancestor] != 0u) {
      coveredByDirtyAncestor = true;
      break;
    }
  }
  if (!coveredByDirtyAncestor && nodes_.dirtyRootQueued[rootIndex] == 0u) {
    nodes_.dirtyRootQueued[rootIndex] = 1u;
    dirtyRoots_.push_back(rootIndex);
  }
  nodes_.dirty[rootIndex] = 1u;
  if (dirtyRoots_.size() > 256u && nodeSlotValid(rootNode_)) {
    for (const uint32_t dirtyRoot : dirtyRoots_) {
      if (dirtyRoot < nodes_.dirtyRootQueued.size()) {
        nodes_.dirtyRootQueued[dirtyRoot] = 0u;
      }
    }
    dirtyRoots_.clear();
    const uint32_t graphRoot = indexOf(rootNode_);
    nodes_.dirtyRootQueued[graphRoot] = 1u;
    nodes_.dirty[graphRoot] = 1u;
    dirtyRoots_.push_back(graphRoot);
  }
}

void SceneGraph::resetWorldTransformSync() noexcept {
  worldSyncRoots_.clear();
  worldSyncStack_.clear();
  worldSyncRootCursor_ = 0u;
  worldSyncPrepared_ = false;
}

bool SceneGraph::syncWorldTransformsStep(uint32_t maxNodes) {
  if (dirtyRoots_.empty()) {
    resetWorldTransformSync();
    return true;
  }
  if (!worldSyncPrepared_) {
    worldSyncRoots_.reserve(dirtyRoots_.size());
    for (const uint32_t rootIndex : dirtyRoots_) {
      if (rootIndex < nodes_.slots.slotCount() &&
          nodes_.slots.isLive(rootIndex) &&
          nodes_.dirtyRootQueued[rootIndex] != 0u) {
        worldSyncRoots_.push_back(rootIndex);
      }
    }
    std::sort(worldSyncRoots_.begin(), worldSyncRoots_.end(),
              [this](uint32_t lhs, uint32_t rhs) {
                return nodes_.depth[lhs] < nodes_.depth[rhs];
              });
    size_t writeIndex = 0u;
    for (const uint32_t rootIndex : worldSyncRoots_) {
      bool covered = false;
      for (uint32_t parent = nodes_.parent[rootIndex]; parent != kInvalidIndex;
           parent = nodes_.parent[parent]) {
        if (parent < nodes_.dirtyRootQueued.size() &&
            nodes_.dirtyRootQueued[parent] != 0u) {
          covered = true;
          break;
        }
      }
      if (!covered) {
        worldSyncRoots_[writeIndex++] = rootIndex;
      }
    }
    worldSyncRoots_.resize(writeIndex);
    worldSyncPrepared_ = true;
  }
  uint32_t processed = 0u;
  while (processed < std::max(maxNodes, 1u)) {
    if (worldSyncStack_.empty()) {
      if (worldSyncRootCursor_ >= worldSyncRoots_.size()) {
        for (const uint32_t rootIndex : dirtyRoots_) {
          if (rootIndex < nodes_.dirtyRootQueued.size()) {
            nodes_.dirtyRootQueued[rootIndex] = 0u;
          }
        }
        dirtyRoots_.clear();
        resetWorldTransformSync();
        return true;
      }
      worldSyncStack_.push_back(worldSyncRoots_[worldSyncRootCursor_++]);
    }
    const uint32_t nodeIndex = worldSyncStack_.back();
    worldSyncStack_.pop_back();
    if (nodeIndex >= nodes_.slots.slotCount() ||
        !nodes_.slots.isLive(nodeIndex)) {
      continue;
    }
    const uint32_t parentIndex = nodes_.parent[nodeIndex];
    nodes_.worldFromRoot[nodeIndex] =
        parentIndex == kInvalidIndex ? nodes_.localFromParent[nodeIndex]
                                     : nodes_.worldFromRoot[parentIndex] *
                                           nodes_.localFromParent[nodeIndex];
    nodes_.dirty[nodeIndex] = 0u;
    ++processed;
    for (uint32_t child = nodes_.firstChild[nodeIndex]; child != kInvalidIndex;
         child = nodes_.nextSibling[child]) {
      worldSyncStack_.push_back(child);
    }
  }
  return false;
}

bool SceneGraph::syncWorldTransforms() {
  const bool hadDirtyRoots = !dirtyRoots_.empty();
  while (!syncWorldTransformsStep(std::numeric_limits<uint32_t>::max())) {
  }
  return hadDirtyRoots;
}

Result<NodeId, std::string>
SceneGraph::createNode(NodeId parent, std::string_view name,
                       const glm::mat4 &localFromParent) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(parent)) {
    return Result<NodeId, std::string>::makeError(
        "SceneGraph::createNode: parent node is invalid");
  }
  auto slotResult = allocateNodeSlot();
  if (slotResult.hasError()) {
    return Result<NodeId, std::string>::makeError(slotResult.error());
  }
  const SlotReservation slot = slotResult.value();
  const uint32_t index = slot.index;
  nodes_.firstChild[index] = kInvalidIndex;
  nodes_.localFromParent[index] = localFromParent;
  nodes_.worldFromRoot[index] = glm::mat4(1.0f);
  nodes_.dirtyRootQueued[index] = 0u;
  nodes_.names[index].assign(name.data(), name.size());
  attachNode(index, indexOf(parent));
  markSubtreeDirty(index);
  markTransformDependentsDirty();
  noteTopologyMutation();
  noteTransformMutation();
  return Result<NodeId, std::string>::makeResult(
      makeNodeId(index, slot.generation));
}

bool SceneGraph::destroyNodeSubtree(NodeId node) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(node) || node == rootNode_) {
    return false;
  }
  const uint32_t rootIndex = indexOf(node);
  std::pmr::vector<uint32_t> nodeStack(memory_);
  std::pmr::vector<uint32_t> nodesToDestroy(memory_);
  nodeStack.push_back(rootIndex);
  while (!nodeStack.empty()) {
    const uint32_t nodeIndex = nodeStack.back();
    nodeStack.pop_back();
    nodesToDestroy.push_back(nodeIndex);
    for (uint32_t child = nodes_.firstChild[nodeIndex]; child != kInvalidIndex;
         child = nodes_.nextSibling[child]) {
      nodeStack.push_back(child);
    }
  }
  detachNode(rootIndex);
  for (const uint32_t nodeIndex : nodesToDestroy) {
    while (nodes_.renderableHead[nodeIndex] != kInvalidIndex) {
      recycleRenderableSlot(nodes_.renderableHead[nodeIndex]);
      renderableTopologyDirty_ = true;
    }
    for (size_t typeIndex = 0; typeIndex < kLightTypeCount; ++typeIndex) {
      auto &head = nodes_.lightHead[typeIndex];
      while (head[nodeIndex] != kInvalidIndex) {
        recycleLightSlot(lights_[typeIndex], head, nodes_.lightTail[typeIndex],
                         head[nodeIndex]);
        lightTopologyDirty_ = true;
        lightDataDirty_ = true;
      }
    }
    while (nodes_.ddgiVolumeHead[nodeIndex] != kInvalidIndex) {
      const uint32_t volumeIndex = nodes_.ddgiVolumeHead[nodeIndex];
      detachComponentFromNode(ddgiVolumes_, nodes_.ddgiVolumeHead,
                              nodes_.ddgiVolumeTail, volumeIndex);
      ddgiVolumes_.slots.release(volumeIndex);
      ddgiVolumeTopologyDirty_ = true;
      ddgiVolumeTransformsDirty_ = true;
      ddgiVolumeSettingsDirty_ = true;
    }
    nodes_.slots.release(nodeIndex);
  }
  markTransformDependentsDirty();
  noteTopologyMutation();
  noteTransformMutation();
  return true;
}

bool SceneGraph::setNodeParent(NodeId node, NodeId newParent,
                               bool preserveWorldTransform) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(node) || !nodeSlotValid(newParent) || node == rootNode_) {
    return false;
  }
  const uint32_t nodeIndex = indexOf(node);
  const uint32_t newParentIndex = indexOf(newParent);
  if (nodeIndex == newParentIndex ||
      isDescendantOf(newParentIndex, nodeIndex)) {
    return false;
  }
  const uint32_t oldParentIndex = nodes_.parent[nodeIndex];
  if (oldParentIndex == newParentIndex) {
    return true;
  }
  glm::mat4 preservedWorld(1.0f);
  if (preserveWorldTransform) {
    (void)syncWorldTransforms();
    preservedWorld = nodes_.worldFromRoot[nodeIndex];
  }
  detachNode(nodeIndex);
  attachNode(nodeIndex, newParentIndex);
  updateSubtreeDepth(nodeIndex);
  if (preserveWorldTransform) {
    nodes_.localFromParent[nodeIndex] =
        nuri::safeInverseOrIdentity(nodes_.worldFromRoot[newParentIndex]) *
        preservedWorld;
  }
  markSubtreeDirty(nodeIndex);
  markTransformDependentsDirty();
  if (!preserveWorldTransform) {
    markDDGIVolumeTransformsDirty(nodeIndex);
  }
  noteTopologyMutation();
  noteTransformMutation();
  return true;
}

bool SceneGraph::setNodeLocalTransform(NodeId node,
                                       const glm::mat4 &localFromParent) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(node)) {
    return false;
  }
  const uint32_t nodeIndex = indexOf(node);
  if (nuri::mat4ExactEqual(nodes_.localFromParent[nodeIndex],
                           localFromParent)) {
    return true;
  }
  nodes_.localFromParent[nodeIndex] = localFromParent;
  markSubtreeDirty(nodeIndex);
  markTransformDependentsDirty();
  markDDGIVolumeTransformsDirty(nodeIndex);
  noteTransformMutation();
  return true;
}

bool SceneGraph::getNodeLocalTransform(NodeId node, glm::mat4 &out) const {
  if (!nodeSlotValid(node)) {
    return false;
  }
  out = nodes_.localFromParent[indexOf(node)];
  return true;
}

bool SceneGraph::getCachedNodeWorldTransform(NodeId node,
                                             glm::mat4 &out) const {
  if (!nodeSlotValid(node)) {
    return false;
  }
  out = nodes_.worldFromRoot[indexOf(node)];
  return true;
}

bool SceneGraph::getNodeParent(NodeId node, NodeId &out) const {
  return readNodeLink(nodes_.slots, nodes_.parent, node, out);
}

bool SceneGraph::getNodeFirstChild(NodeId node, NodeId &out) const {
  return readNodeLink(nodes_.slots, nodes_.firstChild, node, out);
}

bool SceneGraph::getNodeNextSibling(NodeId node, NodeId &out) const {
  return readNodeLink(nodes_.slots, nodes_.nextSibling, node, out);
}

bool SceneGraph::setNodeName(NodeId node, std::string_view name) {
  if (!nodeSlotValid(node)) {
    return false;
  }
  nodes_.names[indexOf(node)].assign(name.data(), name.size());
  return true;
}

bool SceneGraph::getNodeName(NodeId node, std::string_view &out) const {
  if (!nodeSlotValid(node)) {
    return false;
  }
  const std::pmr::string &name = nodes_.names[indexOf(node)];
  out = std::string_view(name.data(), name.size());
  return true;
}

Result<RenderableId, std::string>
SceneGraph::addRenderable(NodeId node, ModelRef model, MaterialRef material) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(node) || !isValid(model) || !isValid(material)) {
    return Result<RenderableId, std::string>::makeError(
        "SceneGraph::addRenderable: invalid node or resource");
  }
  auto slotResult = allocateRenderableSlot();
  if (slotResult.hasError()) {
    return Result<RenderableId, std::string>::makeError(slotResult.error());
  }
  const SlotReservation slot = slotResult.value();
  const uint32_t index = slot.index;
  attachComponentToNode(renderableComponents_, nodes_.renderableHead,
                        nodes_.renderableTail, index, indexOf(node));
  renderableComponents_.models[index] = model;
  renderableComponents_.materials[index] = material;
  renderableComponents_.materialOverrides[index] = kInvalidMaterialRef;
  renderableComponents_.flatRenderableIndex[index] = kInvalidIndex;
  renderableTopologyDirty_ = true;
  markTransformDependentsDirty();
  noteTopologyMutation();
  return Result<RenderableId, std::string>::makeResult(
      makeRenderableId(index, slot.generation));
}

Result<uint32_t, std::string>
SceneGraph::addRenderablesInstanced(ModelRef model, MaterialRef material,
                                    std::span<const glm::mat4> modelMatrices) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (modelMatrices.empty()) {
    return Result<uint32_t, std::string>::makeError(
        "SceneGraph::addRenderablesInstanced: modelMatrices is empty");
  }
  constexpr size_t kInlineRollbackCapacity = 64u;
  std::array<NodeId, kInlineRollbackCapacity> inlineCreatedNodes{};
  std::pmr::vector<NodeId> overflowCreatedNodes(memory_);
  std::span<NodeId> createdNodes;
  if (modelMatrices.size() <= inlineCreatedNodes.size()) {
    createdNodes =
        std::span<NodeId>(inlineCreatedNodes.data(), modelMatrices.size());
  } else {
    overflowCreatedNodes.resize(modelMatrices.size());
    createdNodes = overflowCreatedNodes;
  }
  size_t createdNodeCount = 0u;
  const auto rollback = [this, &createdNodes, &createdNodeCount]() {
    while (createdNodeCount != 0u) {
      --createdNodeCount;
      (void)destroyNodeSubtree(createdNodes[createdNodeCount]);
    }
  };
  uint32_t firstIndex = kInvalidIndex;
  for (const glm::mat4 &modelMatrix : modelMatrices) {
    auto nodeResult = createNode(rootNode_, {}, modelMatrix);
    if (nodeResult.hasError()) {
      rollback();
      return Result<uint32_t, std::string>::makeError(nodeResult.error());
    }
    createdNodes[createdNodeCount++] = nodeResult.value();
    auto renderableResult = addRenderable(nodeResult.value(), model, material);
    if (renderableResult.hasError()) {
      rollback();
      return Result<uint32_t, std::string>::makeError(renderableResult.error());
    }
    if (firstIndex == kInvalidIndex) {
      firstIndex = indexOf(renderableResult.value());
    }
  }
  return Result<uint32_t, std::string>::makeResult(firstIndex);
}

bool SceneGraph::removeRenderable(RenderableId id) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!renderableSlotValid(id)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  recycleRenderableSlot(index);
  renderableTopologyDirty_ = true;
  noteTopologyMutation();
  return true;
}

bool SceneGraph::setRenderableModel(RenderableId id, ModelRef model) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  return writeSlotRef(renderableComponents_.slots, renderableComponents_.models,
                      id, model, renderableTopologyDirty_);
}

bool SceneGraph::setRenderableMaterial(RenderableId id, MaterialRef material) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  return writeSlotRef(renderableComponents_.slots,
                      renderableComponents_.materials, id, material,
                      renderableTopologyDirty_);
}

bool SceneGraph::getRenderableModel(RenderableId id, ModelRef &out) const {
  return readSlotValue(renderableComponents_.slots,
                       renderableComponents_.models, id, out);
}

bool SceneGraph::getRenderableMaterial(RenderableId id,
                                       MaterialRef &out) const {
  return readSlotValue(renderableComponents_.slots,
                       renderableComponents_.materials, id, out);
}

bool SceneGraph::getRenderableMaterialOverride(RenderableId id,
                                               MaterialRef &out) const {
  return readSlotValue(renderableComponents_.slots,
                       renderableComponents_.materialOverrides, id, out);
}

bool SceneGraph::setRenderableMorphWeights(RenderableId id,
                                           std::span<const float> weights) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  return writeSlotRange(renderableComponents_.slots,
                        renderableComponents_.morphWeights, id, weights,
                        renderableDeformationsDirty_);
}

std::span<const float>
SceneGraph::getRenderableMorphWeights(RenderableId id) const {
  return readSlotRange<float>(renderableComponents_.slots,
                              renderableComponents_.morphWeights, id);
}

bool SceneGraph::setRenderableSkinPalette(RenderableId id,
                                          std::span<const glm::mat4> matrices) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  return writeSlotRange(renderableComponents_.slots,
                        renderableComponents_.skinPalette, id, matrices,
                        renderableDeformationsDirty_, nuri::mat4ExactEqual);
}

std::span<const glm::mat4>
SceneGraph::getRenderableSkinPalette(RenderableId id) const {
  return readSlotRange<glm::mat4>(renderableComponents_.slots,
                                  renderableComponents_.skinPalette, id);
}

bool SceneGraph::setRenderableMaterialOverride(RenderableId id,
                                               MaterialRef material) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  return writeSlotRef(renderableComponents_.slots,
                      renderableComponents_.materialOverrides, id, material,
                      renderableTopologyDirty_);
}

bool SceneGraph::clearRenderableMaterialOverride(RenderableId id) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!renderableSlotValid(id)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  if (!isValid(renderableComponents_.materialOverrides[index])) {
    return true;
  }
  renderableComponents_.materialOverrides[index] = kInvalidMaterialRef;
  renderableTopologyDirty_ = true;
  return true;
}

bool SceneGraph::getRenderableNode(RenderableId id, NodeId &out) const {
  if (!renderableSlotValid(id)) {
    return false;
  }
  const uint32_t nodeIndex = renderableComponents_.node[indexOf(id)];
  out = makeNodeId(nodeIndex, nodes_.slots.generation(nodeIndex));
  return true;
}

Result<LightId, std::string> SceneGraph::addLight(NodeId node,
                                                  const LightDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const size_t typeIndex = static_cast<size_t>(desc.type);
  if (!nodeSlotValid(node) || typeIndex >= kLightTypeCount) {
    return Result<LightId, std::string>::makeError(
        "SceneGraph::addLight: invalid node or light type");
  }
  const LightDesc sanitized = nuri::sanitizeLightDesc(desc);
  const bool capReached =
      sanitized.type == LightType::Directional
          ? lights_[typeIndex].slots.liveCount() >= kMaxDirectionalLightCount
          : localLightCount() >= kMaxLocalLightCount;
  if (capReached) {
    return Result<LightId, std::string>::makeError(
        "SceneGraph::addLight: light cap reached");
  }
  auto slotResult = allocateLightSlot(sanitized.type);
  if (slotResult.hasError()) {
    return Result<LightId, std::string>::makeError(slotResult.error());
  }
  const SlotReservation slot = slotResult.value();
  auto &store = lights_[typeIndex];
  auto &record = store.records[slot.index];
  record.packedIndex = kInvalidIndex;
  attachComponentToNode(store, nodes_.lightHead[typeIndex],
                        nodes_.lightTail[typeIndex], slot.index, indexOf(node));
  writeLightRecord(record, sanitized);
  lightTopologyDirty_ = true;
  lightDataDirty_ = true;
  return Result<LightId, std::string>::makeResult(
      makeLightId(sanitized.type, slot.index, slot.generation));
}

bool SceneGraph::removeLight(LightId id) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!lightSlotValid(id)) {
    return false;
  }
  const size_t typeIndex = static_cast<size_t>(id.type);
  recycleLightSlot(lights_[typeIndex], nodes_.lightHead[typeIndex],
                   nodes_.lightTail[typeIndex], indexOf(id));
  lightTopologyDirty_ = true;
  lightDataDirty_ = true;
  return true;
}

bool SceneGraph::getLightDesc(LightId id, LightDesc &outLocal) const {
  if (!lightSlotValid(id)) {
    return false;
  }
  outLocal = nuri::makeLocalLightDesc(
      lights_[static_cast<size_t>(id.type)].records[indexOf(id)], id.type);
  return true;
}

bool SceneGraph::getCachedLightWorldDesc(LightId id,
                                         LightDesc &outWorld) const {
  if (!lightSlotValid(id)) {
    return false;
  }
  const auto &record =
      lights_[static_cast<size_t>(id.type)].records[indexOf(id)];
  outWorld = transformLightDesc(nuri::makeLocalLightDesc(record, id.type),
                                nodes_.worldFromRoot[record.node]);
  return true;
}

bool SceneGraph::getLightNode(LightId id, NodeId &out) const {
  if (!lightSlotValid(id)) {
    return false;
  }
  const uint32_t nodeIndex =
      lights_[static_cast<size_t>(id.type)].records[indexOf(id)].node;
  out = makeNodeId(nodeIndex, nodes_.slots.generation(nodeIndex));
  return true;
}

bool SceneGraph::updateLight(LightId id, const LightDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!lightSlotValid(id) || desc.type != id.type) {
    return false;
  }
  const LightDesc sanitized = nuri::sanitizeLightDesc(desc);
  auto &record = lights_[static_cast<size_t>(id.type)].records[indexOf(id)];
  const bool topologyChanged = record.enabled != sanitized.enabled;
  const bool dataChanged = !lightRecordDataEqual(record, sanitized);
  writeLightRecord(record, sanitized);
  lightTopologyDirty_ |= topologyChanged;
  lightDataDirty_ |= topologyChanged || dataChanged;
  return true;
}

bool SceneGraph::setLightNode(LightId id, NodeId node,
                              bool preserveWorldTransform) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!lightSlotValid(id) || !nodeSlotValid(node)) {
    return false;
  }
  const size_t typeIndex = static_cast<size_t>(id.type);
  auto &store = lights_[typeIndex];
  auto &record = store.records[indexOf(id)];
  LightDesc localDesc = nuri::makeLocalLightDesc(record, id.type);
  if (preserveWorldTransform) {
    (void)syncWorldTransforms();
    const LightDesc worldDesc =
        transformLightDesc(localDesc, nodes_.worldFromRoot[record.node]);
    localDesc = nuri::lightLocalFromWorld(worldDesc,
                                          nodes_.worldFromRoot[indexOf(node)]);
  }
  detachComponentFromNode(store, nodes_.lightHead[typeIndex],
                          nodes_.lightTail[typeIndex], indexOf(id));
  attachComponentToNode(store, nodes_.lightHead[typeIndex],
                        nodes_.lightTail[typeIndex], indexOf(id),
                        indexOf(node));
  record.localPosition = localDesc.position;
  record.localRotation = localDesc.rotation;
  lightDataDirty_ = true;
  return true;
}

Result<DDGIVolumeId, std::string>
SceneGraph::addDDGIVolume(NodeId node, const DDGIVolumeDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(node)) {
    return Result<DDGIVolumeId, std::string>::makeError(
        "SceneGraph::addDDGIVolume: node is invalid");
  }
  if (validateDDGIVolumeDesc(desc).reason != DDGIVolumeValidationReason::None) {
    return Result<DDGIVolumeId, std::string>::makeError(
        "SceneGraph::addDDGIVolume: descriptor is invalid");
  }
  auto slotResult = allocateDDGIVolumeSlot();
  if (slotResult.hasError()) {
    return Result<DDGIVolumeId, std::string>::makeError(slotResult.error());
  }
  const SlotReservation slot = slotResult.value();
  attachComponentToNode(ddgiVolumes_, nodes_.ddgiVolumeHead,
                        nodes_.ddgiVolumeTail, slot.index, indexOf(node));
  writeDDGIVolumeRecord(ddgiVolumes_.records[slot.index], desc);
  ddgiVolumeTopologyDirty_ = true;
  ddgiVolumeTransformsDirty_ = true;
  ddgiVolumeSettingsDirty_ = true;
  return Result<DDGIVolumeId, std::string>::makeResult(
      makeDDGIVolumeId(slot.index, slot.generation));
}

bool SceneGraph::removeDDGIVolume(DDGIVolumeId id) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!ddgiVolumeSlotValid(id)) {
    return false;
  }
  detachComponentFromNode(ddgiVolumes_, nodes_.ddgiVolumeHead,
                          nodes_.ddgiVolumeTail, indexOf(id));
  ddgiVolumes_.slots.release(indexOf(id));
  ddgiVolumeTopologyDirty_ = true;
  ddgiVolumeTransformsDirty_ = true;
  ddgiVolumeSettingsDirty_ = true;
  return true;
}

bool SceneGraph::updateDDGIVolume(DDGIVolumeId id, const DDGIVolumeDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!ddgiVolumeSlotValid(id) ||
      validateDDGIVolumeDesc(desc).reason != DDGIVolumeValidationReason::None) {
    return false;
  }
  DDGIVolumeRecord &record = ddgiVolumes_.records[indexOf(id)];
  if (ddgiVolumeRecordEqual(record, desc)) {
    return true;
  }
  const bool enabledChanged = record.enabled != desc.enabled;
  writeDDGIVolumeRecord(record, desc);
  ddgiVolumeTopologyDirty_ |= enabledChanged;
  ddgiVolumeSettingsDirty_ = true;
  return true;
}

bool SceneGraph::getDDGIVolume(DDGIVolumeId id, DDGIVolumeDesc &out) const {
  if (!ddgiVolumeSlotValid(id)) {
    return false;
  }
  out = makeDDGIVolumeDesc(ddgiVolumes_.records[indexOf(id)]);
  return true;
}

bool SceneGraph::getDDGIVolumeNode(DDGIVolumeId id, NodeId &out) const {
  if (!ddgiVolumeSlotValid(id)) {
    return false;
  }
  const uint32_t nodeIndex = ddgiVolumes_.records[indexOf(id)].node;
  out = makeNodeId(nodeIndex, nodes_.slots.generation(nodeIndex));
  return true;
}

Result<NodeId, std::string>
SceneGraph::instantiatePrefabStructure(const ScenePrefab &prefab, NodeId parent,
                                       SceneInstantiationMap *outMap) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  SceneInstantiationMap localMap(memory_);
  SceneInstantiationMap &instantiation = outMap != nullptr ? *outMap : localMap;
  ScenePrefabStructureCursor cursor{};
  auto result =
      instantiatePrefabStructureStep(prefab, parent, instantiation, cursor,
                                     std::numeric_limits<uint32_t>::max());
  if (result.hasError()) {
    return Result<NodeId, std::string>::makeError(result.error());
  }
  return Result<NodeId, std::string>::makeResult(cursor.firstRoot);
}

Result<bool, std::string> SceneGraph::instantiatePrefabStructureStep(
    const ScenePrefab &prefab, NodeId parent, SceneInstantiationMap &outMap,
    ScenePrefabStructureCursor &cursor, uint32_t maxOperations) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(parent) || prefab.nodes.empty()) {
    return Result<bool, std::string>::makeError(
        "SceneGraph::instantiatePrefabStructure: invalid parent or prefab");
  }
  if (cursor.complete) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!cursor.initialized) {
    outMap.nodes.assign(prefab.nodes.size(), kInvalidNodeId);
    outMap.renderables.assign(prefab.renderables.size(), kInvalidRenderableId);
    outMap.lights.clear();
    outMap.lights.reserve(prefab.lights.size());
    cursor = ScenePrefabStructureCursor{.initialized = true};
  }
  const auto rollbackAndReturnError = [&](std::string error) {
    for (uint32_t nodeIndex = cursor.nextNode; nodeIndex > 0u; --nodeIndex) {
      const uint32_t prefabNodeIndex = nodeIndex - 1u;
      if (prefab.nodes[prefabNodeIndex].parentIndex !=
              kInvalidScenePrefabIndex ||
          prefabNodeIndex >= outMap.nodes.size() ||
          !isValid(outMap.nodes[prefabNodeIndex])) {
        continue;
      }
      (void)destroyNodeSubtree(outMap.nodes[prefabNodeIndex]);
    }
    cursor.complete = false;
    return Result<bool, std::string>::makeError(std::move(error));
  };
  uint32_t completedOperations = 0u;
  while (cursor.nextNode < prefab.nodes.size() &&
         completedOperations < maxOperations) {
    const uint32_t prefabNodeIndex = cursor.nextNode;
    const ScenePrefabNode &prefabNode = prefab.nodes[prefabNodeIndex];
    if (prefabNode.parentIndex != kInvalidScenePrefabIndex &&
        prefabNode.parentIndex >= prefabNodeIndex) {
      return rollbackAndReturnError(
          "SceneGraph::instantiatePrefabStructure: prefab parent index must "
          "refer to an earlier node");
    }
    const NodeId parentNode = prefabNode.parentIndex == kInvalidScenePrefabIndex
                                  ? parent
                                  : outMap.nodes[prefabNode.parentIndex];
    auto createResult =
        createNode(parentNode, prefabNode.name, prefabNode.localFromParent);
    if (createResult.hasError()) {
      return rollbackAndReturnError(createResult.error());
    }
    outMap.nodes[prefabNodeIndex] = createResult.value();
    if (!isValid(cursor.firstRoot)) {
      cursor.firstRoot = createResult.value();
    }
    ++cursor.nextNode;
    ++completedOperations;
  }
  while (cursor.nextNode == prefab.nodes.size() &&
         cursor.nextLight < prefab.lights.size() &&
         completedOperations < maxOperations) {
    const ScenePrefabLight &prefabLight = prefab.lights[cursor.nextLight];
    if (prefabLight.nodeIndex >= outMap.nodes.size()) {
      return rollbackAndReturnError(
          "SceneGraph::instantiatePrefabStructure: prefab light node index is "
          "invalid");
    }
    auto lightResult =
        addLight(outMap.nodes[prefabLight.nodeIndex], prefabLight.light);
    if (lightResult.hasError()) {
      return rollbackAndReturnError(lightResult.error());
    }
    outMap.lights.push_back(lightResult.value());
    ++cursor.nextLight;
    ++completedOperations;
  }
  if (cursor.nextNode == prefab.nodes.size() &&
      cursor.nextLight == prefab.lights.size()) {
    cursor.complete = true;
  }
  return Result<bool, std::string>::makeResult(cursor.complete);
}

Result<RenderableId, std::string> SceneGraph::attachPrefabRenderable(
    const ScenePrefab &prefab, uint32_t prefabRenderableIndex, ModelRef model,
    MaterialRef material, SceneInstantiationMap &map) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (prefabRenderableIndex >= prefab.renderables.size() || !isValid(model) ||
      !isValid(material) ||
      map.renderables.size() != prefab.renderables.size()) {
    return Result<RenderableId, std::string>::makeError(
        "SceneGraph::attachPrefabRenderable: invalid prefab or resources");
  }
  const ScenePrefabRenderable &prefabRenderable =
      prefab.renderables[prefabRenderableIndex];
  if (prefabRenderable.nodeIndex >= map.nodes.size() ||
      !nodeSlotValid(map.nodes[prefabRenderable.nodeIndex]) ||
      isValid(map.renderables[prefabRenderableIndex])) {
    return Result<RenderableId, std::string>::makeError(
        "SceneGraph::attachPrefabRenderable: invalid instantiation map");
  }
  auto renderableResult =
      addRenderable(map.nodes[prefabRenderable.nodeIndex], model, material);
  if (renderableResult.hasError()) {
    return renderableResult;
  }
  const ScenePrefabNode &prefabNode = prefab.nodes[prefabRenderable.nodeIndex];
  if (!prefabNode.morphWeights.empty()) {
    (void)setRenderableMorphWeights(
        renderableResult.value(),
        std::span<const float>(prefabNode.morphWeights.data(),
                               prefabNode.morphWeights.size()));
  }
  map.renderables[prefabRenderableIndex] = renderableResult.value();
  return renderableResult;
}

Result<NodeId, std::string>
SceneGraph::instantiatePrefab(const ScenePrefab &prefab, NodeId parent,
                              const ScenePrefabAssets &assets,
                              SceneInstantiationMap *outMap) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  SceneInstantiationMap localMap(memory_);
  auto structureResult = instantiatePrefabStructure(prefab, parent, &localMap);
  if (structureResult.hasError()) {
    if (outMap != nullptr) {
      *outMap = std::move(localMap);
    }
    return structureResult;
  }
  const auto rollbackAndReturnError = [&](std::string error) {
    for (uint32_t prefabNodeIndex = 0u; prefabNodeIndex < prefab.nodes.size();
         ++prefabNodeIndex) {
      if (prefab.nodes[prefabNodeIndex].parentIndex !=
              kInvalidScenePrefabIndex ||
          prefabNodeIndex >= localMap.nodes.size() ||
          !isValid(localMap.nodes[prefabNodeIndex])) {
        continue;
      }
      (void)destroyNodeSubtree(localMap.nodes[prefabNodeIndex]);
    }
    if (outMap != nullptr) {
      *outMap = std::move(localMap);
    }
    return Result<NodeId, std::string>::makeError(std::move(error));
  };
  MaterialRef fallbackMaterial = kInvalidMaterialRef;
  for (const MaterialRef material : assets.materials) {
    if (isValid(material)) {
      fallbackMaterial = material;
      break;
    }
  }
  for (uint32_t renderableIndex = 0u;
       renderableIndex < prefab.renderables.size(); ++renderableIndex) {
    const ScenePrefabRenderable &prefabRenderable =
        prefab.renderables[renderableIndex];
    if (prefabRenderable.meshAssetIndex >= assets.models.size() ||
        !isValid(assets.models[prefabRenderable.meshAssetIndex])) {
      return rollbackAndReturnError(
          "SceneGraph::instantiatePrefab: prefab model asset is unresolved");
    }
    MaterialRef material = fallbackMaterial;
    if (prefabRenderable.materialAssetIndex < assets.materials.size() &&
        isValid(assets.materials[prefabRenderable.materialAssetIndex])) {
      material = assets.materials[prefabRenderable.materialAssetIndex];
    }
    if (!isValid(material)) {
      return rollbackAndReturnError(
          "SceneGraph::instantiatePrefab: prefab material asset is unresolved");
    }
    auto renderableResult = attachPrefabRenderable(
        prefab, renderableIndex, assets.models[prefabRenderable.meshAssetIndex],
        material, localMap);
    if (renderableResult.hasError()) {
      return rollbackAndReturnError(renderableResult.error());
    }
  }
  if (outMap != nullptr) {
    *outMap = std::move(localMap);
  }
  return structureResult;
}

void SceneGraph::clearRenderables() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const bool hadRenderables = renderableComponents_.slots.liveCount() != 0u;
  renderableComponents_ = RenderableStore(memory_);
  std::fill(nodes_.renderableHead.begin(), nodes_.renderableHead.end(),
            kInvalidIndex);
  std::fill(nodes_.renderableTail.begin(), nodes_.renderableTail.end(),
            kInvalidIndex);
  renderableTopologyDirty_ = true;
  renderableTransformsDirty_ = false;
  if (hadRenderables) {
    noteTopologyMutation();
  }
}

void SceneGraph::clearLights() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  for (size_t typeIndex = 0; typeIndex < kLightTypeCount; ++typeIndex) {
    lights_[typeIndex] = LightStore(memory_);
    std::fill(nodes_.lightHead[typeIndex].begin(),
              nodes_.lightHead[typeIndex].end(), kInvalidIndex);
    std::fill(nodes_.lightTail[typeIndex].begin(),
              nodes_.lightTail[typeIndex].end(), kInvalidIndex);
  }
  lightTopologyDirty_ = true;
  lightDataDirty_ = false;
}

void SceneGraph::clearDDGIVolumes() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const bool hadVolumes = ddgiVolumes_.slots.liveCount() != 0u;
  ddgiVolumes_ = DDGIVolumeStore(memory_);
  std::fill(nodes_.ddgiVolumeHead.begin(), nodes_.ddgiVolumeHead.end(),
            kInvalidIndex);
  std::fill(nodes_.ddgiVolumeTail.begin(), nodes_.ddgiVolumeTail.end(),
            kInvalidIndex);
  ddgiVolumeTopologyDirty_ |= hadVolumes;
  ddgiVolumeTransformsDirty_ |= hadVolumes;
  ddgiVolumeSettingsDirty_ |= hadVolumes;
}

} // namespace nuri
