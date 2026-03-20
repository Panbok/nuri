#include "nuri/pch.h"

#include "nuri/scene/scene_graph.h"

#include "nuri/core/profiling.h"
#include "nuri/math/light.h"

namespace nuri {
namespace {

constexpr uint32_t kMaxDirectionalLightCount = 4u;
constexpr uint32_t kMaxLocalLightCount = 64u;
constexpr glm::quat kIdentityRotation(1.0f, 0.0f, 0.0f, 0.0f);

[[nodiscard]] glm::vec3 sanitizeFiniteVec3(const glm::vec3 &value,
                                           const glm::vec3 &fallback) {
  if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
      !std::isfinite(value.z)) {
    return fallback;
  }
  return value;
}

[[nodiscard]] glm::quat sanitizeRotation(const glm::quat &rotation) {
  const float length = glm::length(rotation);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return kIdentityRotation;
  }
  return glm::normalize(rotation);
}

[[nodiscard]] float sanitizeNonNegative(float value, float fallback = 0.0f) {
  if (!std::isfinite(value)) {
    return fallback;
  }
  return std::max(value, 0.0f);
}

[[nodiscard]] LightDesc sanitizeLightDesc(const LightDesc &desc) {
  LightDesc sanitized = desc;
  sanitized.position = sanitizeFiniteVec3(desc.position, glm::vec3(0.0f));
  sanitized.rotation = sanitizeRotation(desc.rotation);
  sanitized.color = glm::max(sanitizeFiniteVec3(desc.color, glm::vec3(1.0f)),
                             glm::vec3(0.0f));
  sanitized.intensity = sanitizeNonNegative(desc.intensity, 1.0f);
  sanitized.enabled = desc.enabled;

  switch (sanitized.type) {
  case LightType::Directional:
    sanitized.range = 0.0f;
    sanitized.innerConeAngleRadians = 0.0f;
    sanitized.outerConeAngleRadians = 0.0f;
    break;
  case LightType::Point:
    sanitized.range = sanitizeNonNegative(desc.range, 0.0f);
    sanitized.innerConeAngleRadians = 0.0f;
    sanitized.outerConeAngleRadians = 0.0f;
    break;
  case LightType::Spot:
    sanitized.range = sanitizeNonNegative(desc.range, 0.0f);
    sanitized.outerConeAngleRadians =
        std::clamp(sanitizeNonNegative(desc.outerConeAngleRadians,
                                       glm::quarter_pi<float>()),
                   0.0f, glm::half_pi<float>() - 1.0e-4f);
    sanitized.innerConeAngleRadians =
        std::clamp(sanitizeNonNegative(desc.innerConeAngleRadians, 0.0f), 0.0f,
                   sanitized.outerConeAngleRadians);
    break;
  }

  return sanitized;
}

[[nodiscard]] uint32_t nextSlotIndex(std::pmr::vector<uint32_t> &freeSlots,
                                     size_t currentSize) {
  if (!freeSlots.empty()) {
    const uint32_t index = freeSlots.back();
    freeSlots.pop_back();
    return index;
  }
  return static_cast<uint32_t>(currentSize);
}

[[nodiscard]] bool vec3ExactEqual(const glm::vec3 &lhs, const glm::vec3 &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

[[nodiscard]] bool quatExactEqual(const glm::quat &lhs, const glm::quat &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
}

[[nodiscard]] bool mat4ExactEqual(const glm::mat4 &lhs, const glm::mat4 &rhs) {
  for (uint32_t column = 0; column < 4u; ++column) {
    for (uint32_t row = 0; row < 4u; ++row) {
      if (lhs[column][row] != rhs[column][row]) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] glm::mat4 safeInverseOrIdentity(const glm::mat4 &matrix) {
  const float determinant = glm::determinant(matrix);
  if (!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-8f) {
    return glm::mat4(1.0f);
  }
  return glm::inverse(matrix);
}

[[nodiscard]] LightDesc lightLocalFromWorld(const LightDesc &worldDesc,
                                            const glm::mat4 &nodeWorld) {
  LightDesc local = worldDesc;
  const glm::mat4 inverseNodeWorld = safeInverseOrIdentity(nodeWorld);
  local.position =
      glm::vec3(inverseNodeWorld * glm::vec4(worldDesc.position, 1.0f));
  const glm::quat nodeRotation = rotationFromMatrixOrIdentity(nodeWorld);
  local.rotation =
      sanitizeRotation(glm::inverse(nodeRotation) * worldDesc.rotation);
  return sanitizeLightDesc(local);
}

template <typename Store>
[[nodiscard]] LightDesc makeLocalLightDesc(const Store &store, uint32_t index,
                                           LightType type) {
  LightDesc out{};
  out.type = type;
  out.name = store.names[index];
  out.position = store.localPositions[index];
  out.rotation = store.localRotations[index];
  out.color = store.colors[index];
  out.intensity = store.intensities[index];
  out.enabled = store.enabled[index] != 0u;
  return out;
}

} // namespace

SceneGraph::SceneGraph(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      dirtyRoots_(memory_), nodes_(memory_), renderableComponents_(memory_),
      directionalLights_(memory_), pointLights_(memory_), spotLights_(memory_) {
  clear();
}

void SceneGraph::clear() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  dirtyRoots_.clear();
  nodes_ = NodeStore(memory_);
  renderableComponents_ = RenderableStore(memory_);
  directionalLights_ = DirectionalLightStore(memory_);
  pointLights_ = PointLightStore(memory_);
  spotLights_ = SpotLightStore(memory_);
  renderableTopologyDirty_ = true;
  renderableTransformsDirty_ = false;
  lightTopologyDirty_ = true;
  lightDataDirty_ = false;

  const auto rootResult = allocateNodeSlot();
  NURI_ASSERT(!rootResult.hasError(),
              "SceneGraph::clear: failed to create root node: %s",
              rootResult.error().c_str());
  const uint32_t rootIndex = rootResult.value();
  nodes_.live[rootIndex] = 1u;
  nodes_.parent[rootIndex] = kInvalidIndex;
  nodes_.firstChild[rootIndex] = kInvalidIndex;
  nodes_.nextSibling[rootIndex] = kInvalidIndex;
  nodes_.prevSibling[rootIndex] = kInvalidIndex;
  nodes_.depth[rootIndex] = 0u;
  nodes_.localFromParent[rootIndex] = glm::mat4(1.0f);
  nodes_.worldFromRoot[rootIndex] = glm::mat4(1.0f);
  nodes_.dirty[rootIndex] = 0u;
  nodes_.dirtyRootQueued[rootIndex] = 0u;
  nodes_.names[rootIndex] = "Root";
  rootNode_ = makeNodeId(rootIndex, nodes_.generations[rootIndex]);
}

Result<uint32_t, std::string> SceneGraph::allocateNodeSlot() {
  const uint32_t index =
      nextSlotIndex(nodes_.freeSlots, nodes_.generations.size());
  if (index == nodes_.generations.size()) {
    nodes_.generations.push_back(1u);
    nodes_.live.push_back(0u);
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
  }
  return Result<uint32_t, std::string>::makeResult(index);
}

uint32_t SceneGraph::allocateRenderableSlot() {
  const uint32_t index =
      nextSlotIndex(renderableComponents_.freeSlots,
                    renderableComponents_.generations.size());
  if (index == renderableComponents_.generations.size()) {
    renderableComponents_.generations.push_back(1u);
    renderableComponents_.live.push_back(0u);
    renderableComponents_.node.push_back(kInvalidIndex);
    renderableComponents_.models.push_back(kInvalidModelRef);
    renderableComponents_.materials.push_back(kInvalidMaterialRef);
    renderableComponents_.materialOverrides.push_back(kInvalidMaterialRef);
    renderableComponents_.flatRenderableIndex.push_back(kInvalidIndex);
  }
  return index;
}

uint32_t SceneGraph::allocateDirectionalLightSlot() {
  const uint32_t index = nextSlotIndex(directionalLights_.freeSlots,
                                       directionalLights_.generations.size());
  if (index == directionalLights_.generations.size()) {
    directionalLights_.generations.push_back(1u);
    directionalLights_.live.push_back(0u);
    directionalLights_.packedIndices.push_back(kInvalidPackedLightIndex);
    directionalLights_.node.push_back(kInvalidIndex);
    directionalLights_.names.emplace_back();
    directionalLights_.localPositions.push_back(glm::vec3(0.0f));
    directionalLights_.localRotations.push_back(kIdentityRotation);
    directionalLights_.colors.push_back(glm::vec3(1.0f));
    directionalLights_.intensities.push_back(1.0f);
    directionalLights_.enabled.push_back(0u);
  }
  return index;
}

uint32_t SceneGraph::allocatePointLightSlot() {
  const uint32_t index =
      nextSlotIndex(pointLights_.freeSlots, pointLights_.generations.size());
  if (index == pointLights_.generations.size()) {
    pointLights_.generations.push_back(1u);
    pointLights_.live.push_back(0u);
    pointLights_.packedIndices.push_back(kInvalidPackedLightIndex);
    pointLights_.node.push_back(kInvalidIndex);
    pointLights_.names.emplace_back();
    pointLights_.localPositions.push_back(glm::vec3(0.0f));
    pointLights_.localRotations.push_back(kIdentityRotation);
    pointLights_.colors.push_back(glm::vec3(1.0f));
    pointLights_.intensities.push_back(1.0f);
    pointLights_.ranges.push_back(0.0f);
    pointLights_.enabled.push_back(0u);
  }
  return index;
}

uint32_t SceneGraph::allocateSpotLightSlot() {
  const uint32_t index =
      nextSlotIndex(spotLights_.freeSlots, spotLights_.generations.size());
  if (index == spotLights_.generations.size()) {
    spotLights_.generations.push_back(1u);
    spotLights_.live.push_back(0u);
    spotLights_.packedIndices.push_back(kInvalidPackedLightIndex);
    spotLights_.node.push_back(kInvalidIndex);
    spotLights_.names.emplace_back();
    spotLights_.localPositions.push_back(glm::vec3(0.0f));
    spotLights_.localRotations.push_back(kIdentityRotation);
    spotLights_.colors.push_back(glm::vec3(1.0f));
    spotLights_.intensities.push_back(1.0f);
    spotLights_.ranges.push_back(0.0f);
    spotLights_.innerConeAngles.push_back(0.0f);
    spotLights_.outerConeAngles.push_back(glm::quarter_pi<float>());
    spotLights_.enabled.push_back(0u);
  }
  return index;
}

bool SceneGraph::nodeSlotValid(NodeId id) const noexcept {
  if (!isValid(id)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  return index < nodes_.generations.size() && nodes_.live[index] != 0u &&
         nodes_.generations[index] == generationOf(id);
}

bool SceneGraph::renderableSlotValid(RenderableId id) const noexcept {
  if (!isValid(id)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  return index < renderableComponents_.generations.size() &&
         renderableComponents_.live[index] != 0u &&
         renderableComponents_.generations[index] == generationOf(id);
}

bool SceneGraph::directionalSlotValid(LightId id) const noexcept {
  if (id.type != LightType::Directional || !isValid(id)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  return index < directionalLights_.generations.size() &&
         directionalLights_.live[index] != 0u &&
         directionalLights_.generations[index] == generationOf(id);
}

bool SceneGraph::pointSlotValid(LightId id) const noexcept {
  if (id.type != LightType::Point || !isValid(id)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  return index < pointLights_.generations.size() &&
         pointLights_.live[index] != 0u &&
         pointLights_.generations[index] == generationOf(id);
}

bool SceneGraph::spotSlotValid(LightId id) const noexcept {
  if (id.type != LightType::Spot || !isValid(id)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  return index < spotLights_.generations.size() &&
         spotLights_.live[index] != 0u &&
         spotLights_.generations[index] == generationOf(id);
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

void SceneGraph::markTransformDependentsDirty() noexcept {
  renderableTransformsDirty_ = true;
  lightDataDirty_ = true;
}

void SceneGraph::recycleRenderableSlot(uint32_t index) noexcept {
  renderableComponents_.live[index] = 0u;
  renderableComponents_.node[index] = kInvalidIndex;
  renderableComponents_.models[index] = kInvalidModelRef;
  renderableComponents_.materials[index] = kInvalidMaterialRef;
  renderableComponents_.materialOverrides[index] = kInvalidMaterialRef;
  renderableComponents_.flatRenderableIndex[index] = kInvalidIndex;
  renderableComponents_.generations[index] =
      nextResourceGeneration(renderableComponents_.generations[index]);
  renderableComponents_.freeSlots.push_back(index);
}

uint32_t SceneGraph::localLightCount() const noexcept {
  return static_cast<uint32_t>(
      pointLights_.generations.size() - pointLights_.freeSlots.size() +
      spotLights_.generations.size() - spotLights_.freeSlots.size());
}

bool SceneGraph::tryGetLightNodeIndex(LightId id,
                                      uint32_t &outNodeIndex) const {
  outNodeIndex = kInvalidIndex;
  if (!isValid(id)) {
    return false;
  }

  switch (id.type) {
  case LightType::Directional:
    if (!directionalSlotValid(id)) {
      return false;
    }
    outNodeIndex = directionalLights_.node[indexOf(id)];
    return true;
  case LightType::Point:
    if (!pointSlotValid(id)) {
      return false;
    }
    outNodeIndex = pointLights_.node[indexOf(id)];
    return true;
  case LightType::Spot:
    if (!spotSlotValid(id)) {
      return false;
    }
    outNodeIndex = spotLights_.node[indexOf(id)];
    return true;
  }

  return false;
}

void SceneGraph::markSubtreeDirty(uint32_t rootIndex) {
  if (rootIndex == kInvalidIndex || rootIndex >= nodes_.live.size() ||
      nodes_.live[rootIndex] == 0u) {
    return;
  }

  if (nodes_.dirtyRootQueued[rootIndex] == 0u) {
    nodes_.dirtyRootQueued[rootIndex] = 1u;
    dirtyRoots_.push_back(rootIndex);
  }

  std::pmr::vector<uint32_t> stack(memory_);
  stack.push_back(rootIndex);
  while (!stack.empty()) {
    const uint32_t nodeIndex = stack.back();
    stack.pop_back();
    if (nodes_.live[nodeIndex] == 0u) {
      continue;
    }
    nodes_.dirty[nodeIndex] = 1u;
    for (uint32_t child = nodes_.firstChild[nodeIndex]; child != kInvalidIndex;
         child = nodes_.nextSibling[child]) {
      stack.push_back(child);
    }
  }
}

bool SceneGraph::syncWorldTransforms() {
  if (dirtyRoots_.empty()) {
    return false;
  }

  std::pmr::vector<uint32_t> roots(memory_);
  roots.reserve(dirtyRoots_.size());
  for (const uint32_t rootIndex : dirtyRoots_) {
    if (rootIndex < nodes_.live.size() && nodes_.live[rootIndex] != 0u &&
        nodes_.dirtyRootQueued[rootIndex] != 0u) {
      roots.push_back(rootIndex);
    }
  }
  std::sort(roots.begin(), roots.end(), [this](uint32_t lhs, uint32_t rhs) {
    return nodes_.depth[lhs] < nodes_.depth[rhs];
  });

  std::pmr::vector<uint32_t> filtered(memory_);
  filtered.reserve(roots.size());
  for (const uint32_t rootIndex : roots) {
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
      filtered.push_back(rootIndex);
    }
  }

  std::pmr::vector<uint32_t> stack(memory_);
  bool updated = false;
  for (const uint32_t rootIndex : filtered) {
    stack.clear();
    stack.push_back(rootIndex);
    while (!stack.empty()) {
      const uint32_t nodeIndex = stack.back();
      stack.pop_back();
      if (nodeIndex >= nodes_.live.size() || nodes_.live[nodeIndex] == 0u) {
        continue;
      }
      const uint32_t parentIndex = nodes_.parent[nodeIndex];
      nodes_.worldFromRoot[nodeIndex] =
          parentIndex == kInvalidIndex ? nodes_.localFromParent[nodeIndex]
                                       : nodes_.worldFromRoot[parentIndex] *
                                             nodes_.localFromParent[nodeIndex];
      nodes_.dirty[nodeIndex] = 0u;
      nodes_.dirtyRootQueued[nodeIndex] = 0u;
      updated = true;
      for (uint32_t child = nodes_.firstChild[nodeIndex];
           child != kInvalidIndex; child = nodes_.nextSibling[child]) {
        stack.push_back(child);
      }
    }
  }

  dirtyRoots_.clear();
  return updated;
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
  const uint32_t index = slotResult.value();
  nodes_.live[index] = 1u;
  nodes_.firstChild[index] = kInvalidIndex;
  nodes_.nextSibling[index] = kInvalidIndex;
  nodes_.prevSibling[index] = kInvalidIndex;
  nodes_.localFromParent[index] = localFromParent;
  nodes_.worldFromRoot[index] = glm::mat4(1.0f);
  nodes_.dirty[index] = 0u;
  nodes_.dirtyRootQueued[index] = 0u;
  nodes_.names[index].assign(name.data(), name.size());
  attachNode(index, indexOf(parent));
  markSubtreeDirty(index);
  markTransformDependentsDirty();
  return Result<NodeId, std::string>::makeResult(
      makeNodeId(index, nodes_.generations[index]));
}

bool SceneGraph::destroyNodeSubtree(NodeId node) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(node) || node == rootNode_) {
    return false;
  }

  const uint32_t rootIndex = indexOf(node);
  std::pmr::vector<uint32_t> nodeStack(memory_);
  std::pmr::vector<uint32_t> nodesToDestroy(memory_);
  std::pmr::vector<uint8_t> killNodes(memory_);
  killNodes.resize(nodes_.live.size(), 0u);

  nodeStack.push_back(rootIndex);
  while (!nodeStack.empty()) {
    const uint32_t nodeIndex = nodeStack.back();
    nodeStack.pop_back();
    if (nodeIndex >= nodes_.live.size() || nodes_.live[nodeIndex] == 0u ||
        killNodes[nodeIndex] != 0u) {
      continue;
    }
    killNodes[nodeIndex] = 1u;
    nodesToDestroy.push_back(nodeIndex);
    for (uint32_t child = nodes_.firstChild[nodeIndex]; child != kInvalidIndex;
         child = nodes_.nextSibling[child]) {
      nodeStack.push_back(child);
    }
  }

  for (uint32_t index = 0; index < renderableComponents_.generations.size();
       ++index) {
    if (renderableComponents_.live[index] == 0u) {
      continue;
    }
    const uint32_t nodeIndex = renderableComponents_.node[index];
    if (nodeIndex < killNodes.size() && killNodes[nodeIndex] != 0u) {
      recycleRenderableSlot(index);
      renderableTopologyDirty_ = true;
    }
  }

  const auto removeDeadLights = [this, &killNodes](auto &store) {
    for (uint32_t index = 0; index < store.generations.size(); ++index) {
      if (store.live[index] == 0u) {
        continue;
      }
      const uint32_t nodeIndex = store.node[index];
      if (nodeIndex >= killNodes.size() || killNodes[nodeIndex] == 0u) {
        continue;
      }
      store.live[index] = 0u;
      store.node[index] = kInvalidIndex;
      store.packedIndices[index] = kInvalidPackedLightIndex;
      store.names[index].clear();
      store.enabled[index] = 0u;
      store.generations[index] =
          nextResourceGeneration(store.generations[index]);
      store.freeSlots.push_back(index);
      lightTopologyDirty_ = true;
      lightDataDirty_ = true;
    }
  };
  removeDeadLights(directionalLights_);
  removeDeadLights(pointLights_);
  removeDeadLights(spotLights_);

  detachNode(rootIndex);
  for (const uint32_t nodeIndex : nodesToDestroy) {
    nodes_.live[nodeIndex] = 0u;
    nodes_.parent[nodeIndex] = kInvalidIndex;
    nodes_.firstChild[nodeIndex] = kInvalidIndex;
    nodes_.nextSibling[nodeIndex] = kInvalidIndex;
    nodes_.prevSibling[nodeIndex] = kInvalidIndex;
    nodes_.depth[nodeIndex] = 0u;
    nodes_.localFromParent[nodeIndex] = glm::mat4(1.0f);
    nodes_.worldFromRoot[nodeIndex] = glm::mat4(1.0f);
    nodes_.dirty[nodeIndex] = 0u;
    nodes_.dirtyRootQueued[nodeIndex] = 0u;
    nodes_.names[nodeIndex].clear();
    nodes_.generations[nodeIndex] =
        nextResourceGeneration(nodes_.generations[nodeIndex]);
    nodes_.freeSlots.push_back(nodeIndex);
  }
  markTransformDependentsDirty();
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
        safeInverseOrIdentity(nodes_.worldFromRoot[newParentIndex]) *
        preservedWorld;
  }
  markSubtreeDirty(nodeIndex);
  markTransformDependentsDirty();
  return true;
}

bool SceneGraph::setNodeLocalTransform(NodeId node,
                                       const glm::mat4 &localFromParent) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(node)) {
    return false;
  }
  const uint32_t nodeIndex = indexOf(node);
  if (mat4ExactEqual(nodes_.localFromParent[nodeIndex], localFromParent)) {
    return true;
  }
  nodes_.localFromParent[nodeIndex] = localFromParent;
  markSubtreeDirty(nodeIndex);
  markTransformDependentsDirty();
  return true;
}

bool SceneGraph::getNodeLocalTransform(NodeId node, glm::mat4 &out) const {
  if (!nodeSlotValid(node)) {
    return false;
  }
  out = nodes_.localFromParent[indexOf(node)];
  return true;
}

bool SceneGraph::getCachedNodeWorldTransform(NodeId node, glm::mat4 &out) const {
  if (!nodeSlotValid(node)) {
    return false;
  }
  out = nodes_.worldFromRoot[indexOf(node)];
  return true;
}

bool SceneGraph::getNodeParent(NodeId node, NodeId &out) const {
  if (!nodeSlotValid(node)) {
    return false;
  }
  const uint32_t parentIndex = nodes_.parent[indexOf(node)];
  if (parentIndex == kInvalidIndex ||
      parentIndex >= nodes_.generations.size() ||
      nodes_.live[parentIndex] == 0u) {
    out = kInvalidNodeId;
    return true;
  }
  out = makeNodeId(parentIndex, nodes_.generations[parentIndex]);
  return true;
}

bool SceneGraph::getNodeFirstChild(NodeId node, NodeId &out) const {
  if (!nodeSlotValid(node)) {
    return false;
  }
  const uint32_t childIndex = nodes_.firstChild[indexOf(node)];
  if (childIndex == kInvalidIndex || childIndex >= nodes_.generations.size() ||
      nodes_.live[childIndex] == 0u) {
    out = kInvalidNodeId;
    return true;
  }
  out = makeNodeId(childIndex, nodes_.generations[childIndex]);
  return true;
}

bool SceneGraph::getNodeNextSibling(NodeId node, NodeId &out) const {
  if (!nodeSlotValid(node)) {
    return false;
  }
  const uint32_t siblingIndex = nodes_.nextSibling[indexOf(node)];
  if (siblingIndex == kInvalidIndex ||
      siblingIndex >= nodes_.generations.size() ||
      nodes_.live[siblingIndex] == 0u) {
    out = kInvalidNodeId;
    return true;
  }
  out = makeNodeId(siblingIndex, nodes_.generations[siblingIndex]);
  return true;
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
  if (!nodeSlotValid(node)) {
    return Result<RenderableId, std::string>::makeError(
        "SceneGraph::addRenderable: node is invalid");
  }
  if (!isValid(model)) {
    return Result<RenderableId, std::string>::makeError(
        "SceneGraph::addRenderable: model handle is invalid");
  }
  if (!isValid(material)) {
    return Result<RenderableId, std::string>::makeError(
        "SceneGraph::addRenderable: material handle is invalid");
  }

  const uint32_t index = allocateRenderableSlot();
  renderableComponents_.live[index] = 1u;
  renderableComponents_.node[index] = indexOf(node);
  renderableComponents_.models[index] = model;
  renderableComponents_.materials[index] = material;
  renderableComponents_.materialOverrides[index] = kInvalidMaterialRef;
  renderableComponents_.flatRenderableIndex[index] = kInvalidIndex;
  renderableTopologyDirty_ = true;
  markTransformDependentsDirty();
  return Result<RenderableId, std::string>::makeResult(
      makeRenderableId(index, renderableComponents_.generations[index]));
}

Result<uint32_t, std::string>
SceneGraph::addRenderablesInstanced(ModelRef model, MaterialRef material,
                                    std::span<const glm::mat4> modelMatrices) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (modelMatrices.empty()) {
    return Result<uint32_t, std::string>::makeError(
        "SceneGraph::addRenderablesInstanced: modelMatrices is empty");
  }

  uint32_t firstIndex = kInvalidIndex;
  for (const glm::mat4 &modelMatrix : modelMatrices) {
    auto nodeResult = createNode(rootNode_, {}, modelMatrix);
    if (nodeResult.hasError()) {
      return Result<uint32_t, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult = addRenderable(nodeResult.value(), model, material);
    if (renderableResult.hasError()) {
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
  return true;
}

bool SceneGraph::setRenderableModel(RenderableId id, ModelRef model) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!renderableSlotValid(id) || !isValid(model)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  if (renderableComponents_.models[index].value == model.value) {
    return true;
  }
  renderableComponents_.models[index] = model;
  renderableTopologyDirty_ = true;
  return true;
}

bool SceneGraph::setRenderableMaterial(RenderableId id, MaterialRef material) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!renderableSlotValid(id) || !isValid(material)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  if (renderableComponents_.materials[index].value == material.value) {
    return true;
  }
  renderableComponents_.materials[index] = material;
  renderableTopologyDirty_ = true;
  return true;
}

bool SceneGraph::getRenderableModel(RenderableId id, ModelRef &out) const {
  if (!renderableSlotValid(id)) {
    return false;
  }
  out = renderableComponents_.models[indexOf(id)];
  return true;
}

bool SceneGraph::getRenderableMaterial(RenderableId id,
                                       MaterialRef &out) const {
  if (!renderableSlotValid(id)) {
    return false;
  }
  out = renderableComponents_.materials[indexOf(id)];
  return true;
}

bool SceneGraph::getRenderableMaterialOverride(RenderableId id,
                                               MaterialRef &out) const {
  if (!renderableSlotValid(id)) {
    return false;
  }
  out = renderableComponents_.materialOverrides[indexOf(id)];
  return true;
}

bool SceneGraph::setRenderableMaterialOverride(RenderableId id,
                                               MaterialRef material) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!renderableSlotValid(id) || !isValid(material)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  if (renderableComponents_.materialOverrides[index].value == material.value) {
    return true;
  }
  renderableComponents_.materialOverrides[index] = material;
  renderableTopologyDirty_ = true;
  return true;
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
  if (nodeIndex >= nodes_.generations.size() || nodes_.live[nodeIndex] == 0u) {
    return false;
  }
  out = makeNodeId(nodeIndex, nodes_.generations[nodeIndex]);
  return true;
}

Result<LightId, std::string> SceneGraph::addLight(NodeId node,
                                                  const LightDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(node)) {
    return Result<LightId, std::string>::makeError(
        "SceneGraph::addLight: node is invalid");
  }

  const LightDesc sanitized = sanitizeLightDesc(desc);
  switch (sanitized.type) {
  case LightType::Directional: {
    const uint32_t liveCount =
        static_cast<uint32_t>(directionalLights_.generations.size() -
                              directionalLights_.freeSlots.size());
    if (liveCount >= kMaxDirectionalLightCount) {
      return Result<LightId, std::string>::makeError(
          "SceneGraph::addLight: directional light cap reached");
    }
    const uint32_t index = allocateDirectionalLightSlot();
    directionalLights_.live[index] = 1u;
    directionalLights_.packedIndices[index] = kInvalidPackedLightIndex;
    directionalLights_.node[index] = indexOf(node);
    directionalLights_.names[index].assign(sanitized.name.data(),
                                           sanitized.name.size());
    directionalLights_.localPositions[index] = sanitized.position;
    directionalLights_.localRotations[index] = sanitized.rotation;
    directionalLights_.colors[index] = sanitized.color;
    directionalLights_.intensities[index] = sanitized.intensity;
    directionalLights_.enabled[index] = sanitized.enabled ? 1u : 0u;
    lightTopologyDirty_ = true;
    lightDataDirty_ = true;
    return Result<LightId, std::string>::makeResult(makeLightId(
        LightType::Directional, index, directionalLights_.generations[index]));
  }
  case LightType::Point: {
    if (localLightCount() >= kMaxLocalLightCount) {
      return Result<LightId, std::string>::makeError(
          "SceneGraph::addLight: local light cap reached");
    }
    const uint32_t index = allocatePointLightSlot();
    pointLights_.live[index] = 1u;
    pointLights_.packedIndices[index] = kInvalidPackedLightIndex;
    pointLights_.node[index] = indexOf(node);
    pointLights_.names[index].assign(sanitized.name.data(),
                                     sanitized.name.size());
    pointLights_.localPositions[index] = sanitized.position;
    pointLights_.localRotations[index] = sanitized.rotation;
    pointLights_.colors[index] = sanitized.color;
    pointLights_.intensities[index] = sanitized.intensity;
    pointLights_.ranges[index] = sanitized.range;
    pointLights_.enabled[index] = sanitized.enabled ? 1u : 0u;
    lightTopologyDirty_ = true;
    lightDataDirty_ = true;
    return Result<LightId, std::string>::makeResult(
        makeLightId(LightType::Point, index, pointLights_.generations[index]));
  }
  case LightType::Spot: {
    if (localLightCount() >= kMaxLocalLightCount) {
      return Result<LightId, std::string>::makeError(
          "SceneGraph::addLight: local light cap reached");
    }
    const uint32_t index = allocateSpotLightSlot();
    spotLights_.live[index] = 1u;
    spotLights_.packedIndices[index] = kInvalidPackedLightIndex;
    spotLights_.node[index] = indexOf(node);
    spotLights_.names[index].assign(sanitized.name.data(),
                                    sanitized.name.size());
    spotLights_.localPositions[index] = sanitized.position;
    spotLights_.localRotations[index] = sanitized.rotation;
    spotLights_.colors[index] = sanitized.color;
    spotLights_.intensities[index] = sanitized.intensity;
    spotLights_.ranges[index] = sanitized.range;
    spotLights_.innerConeAngles[index] = sanitized.innerConeAngleRadians;
    spotLights_.outerConeAngles[index] = sanitized.outerConeAngleRadians;
    spotLights_.enabled[index] = sanitized.enabled ? 1u : 0u;
    lightTopologyDirty_ = true;
    lightDataDirty_ = true;
    return Result<LightId, std::string>::makeResult(
        makeLightId(LightType::Spot, index, spotLights_.generations[index]));
  }
  }

  return Result<LightId, std::string>::makeError(
      "SceneGraph::addLight: unknown light type");
}

bool SceneGraph::removeLight(LightId id) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(id)) {
    return false;
  }

  const auto removeFromStore = [this](auto &store, LightId lightId) {
    const uint32_t index = indexOf(lightId);
    store.live[index] = 0u;
    store.node[index] = kInvalidIndex;
    store.packedIndices[index] = kInvalidPackedLightIndex;
    store.names[index].clear();
    store.enabled[index] = 0u;
    store.generations[index] = nextResourceGeneration(store.generations[index]);
    store.freeSlots.push_back(index);
    lightTopologyDirty_ = true;
    lightDataDirty_ = true;
  };

  switch (id.type) {
  case LightType::Directional:
    if (!directionalSlotValid(id)) {
      return false;
    }
    removeFromStore(directionalLights_, id);
    return true;
  case LightType::Point:
    if (!pointSlotValid(id)) {
      return false;
    }
    removeFromStore(pointLights_, id);
    return true;
  case LightType::Spot:
    if (!spotSlotValid(id)) {
      return false;
    }
    removeFromStore(spotLights_, id);
    return true;
  }
  return false;
}

bool SceneGraph::getLightDesc(LightId id, LightDesc &outLocal) const {
  if (!isValid(id)) {
    return false;
  }

  switch (id.type) {
  case LightType::Directional:
    if (!directionalSlotValid(id)) {
      return false;
    }
    outLocal = makeLocalLightDesc(directionalLights_, indexOf(id), id.type);
    outLocal.range = 0.0f;
    outLocal.innerConeAngleRadians = 0.0f;
    outLocal.outerConeAngleRadians = 0.0f;
    return true;
  case LightType::Point:
    if (!pointSlotValid(id)) {
      return false;
    }
    outLocal = makeLocalLightDesc(pointLights_, indexOf(id), id.type);
    outLocal.range = pointLights_.ranges[indexOf(id)];
    outLocal.innerConeAngleRadians = 0.0f;
    outLocal.outerConeAngleRadians = 0.0f;
    return true;
  case LightType::Spot:
    if (!spotSlotValid(id)) {
      return false;
    }
    outLocal = makeLocalLightDesc(spotLights_, indexOf(id), id.type);
    outLocal.range = spotLights_.ranges[indexOf(id)];
    outLocal.innerConeAngleRadians = spotLights_.innerConeAngles[indexOf(id)];
    outLocal.outerConeAngleRadians = spotLights_.outerConeAngles[indexOf(id)];
    return true;
  }
  return false;
}

bool SceneGraph::getCachedLightWorldDesc(LightId id,
                                         LightDesc &outWorld) const {
  LightDesc local{};
  if (!getLightDesc(id, local)) {
    return false;
  }

  uint32_t nodeIndex = kInvalidIndex;
  if (!tryGetLightNodeIndex(id, nodeIndex)) {
    return false;
  }
  if (nodeIndex >= nodes_.worldFromRoot.size() ||
      nodes_.live[nodeIndex] == 0u) {
    return false;
  }

  outWorld = transformLightDesc(local, nodes_.worldFromRoot[nodeIndex]);
  outWorld.type = local.type;
  outWorld.name = local.name;
  outWorld.color = local.color;
  outWorld.intensity = local.intensity;
  outWorld.range = local.range;
  outWorld.innerConeAngleRadians = local.innerConeAngleRadians;
  outWorld.outerConeAngleRadians = local.outerConeAngleRadians;
  outWorld.enabled = local.enabled;
  return true;
}

bool SceneGraph::getLightNode(LightId id, NodeId &out) const {
  uint32_t nodeIndex = kInvalidIndex;
  if (!tryGetLightNodeIndex(id, nodeIndex)) {
    return false;
  }
  if (nodeIndex >= nodes_.generations.size() || nodes_.live[nodeIndex] == 0u) {
    return false;
  }
  out = makeNodeId(nodeIndex, nodes_.generations[nodeIndex]);
  return true;
}

bool SceneGraph::updateLight(LightId id, const LightDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(id) || desc.type != id.type) {
    return false;
  }
  const LightDesc sanitized = sanitizeLightDesc(desc);

  switch (id.type) {
  case LightType::Directional: {
    if (!directionalSlotValid(id)) {
      return false;
    }
    const uint32_t index = indexOf(id);
    const bool topologyChanged =
        (directionalLights_.enabled[index] != 0u) != sanitized.enabled;
    const bool derivedDataChanged =
        !vec3ExactEqual(directionalLights_.localPositions[index],
                        sanitized.position) ||
        !quatExactEqual(directionalLights_.localRotations[index],
                        sanitized.rotation) ||
        !vec3ExactEqual(directionalLights_.colors[index], sanitized.color) ||
        directionalLights_.intensities[index] != sanitized.intensity;
    directionalLights_.names[index].assign(sanitized.name.data(),
                                           sanitized.name.size());
    directionalLights_.localPositions[index] = sanitized.position;
    directionalLights_.localRotations[index] = sanitized.rotation;
    directionalLights_.colors[index] = sanitized.color;
    directionalLights_.intensities[index] = sanitized.intensity;
    directionalLights_.enabled[index] = sanitized.enabled ? 1u : 0u;
    lightTopologyDirty_ |= topologyChanged;
    lightDataDirty_ |= topologyChanged || derivedDataChanged;
    return true;
  }
  case LightType::Point: {
    if (!pointSlotValid(id)) {
      return false;
    }
    const uint32_t index = indexOf(id);
    const bool topologyChanged =
        (pointLights_.enabled[index] != 0u) != sanitized.enabled;
    const bool derivedDataChanged =
        !vec3ExactEqual(pointLights_.localPositions[index],
                        sanitized.position) ||
        !quatExactEqual(pointLights_.localRotations[index],
                        sanitized.rotation) ||
        !vec3ExactEqual(pointLights_.colors[index], sanitized.color) ||
        pointLights_.intensities[index] != sanitized.intensity ||
        pointLights_.ranges[index] != sanitized.range;
    pointLights_.names[index].assign(sanitized.name.data(),
                                     sanitized.name.size());
    pointLights_.localPositions[index] = sanitized.position;
    pointLights_.localRotations[index] = sanitized.rotation;
    pointLights_.colors[index] = sanitized.color;
    pointLights_.intensities[index] = sanitized.intensity;
    pointLights_.ranges[index] = sanitized.range;
    pointLights_.enabled[index] = sanitized.enabled ? 1u : 0u;
    lightTopologyDirty_ |= topologyChanged;
    lightDataDirty_ |= topologyChanged || derivedDataChanged;
    return true;
  }
  case LightType::Spot: {
    if (!spotSlotValid(id)) {
      return false;
    }
    const uint32_t index = indexOf(id);
    const bool topologyChanged =
        (spotLights_.enabled[index] != 0u) != sanitized.enabled;
    const bool derivedDataChanged =
        !vec3ExactEqual(spotLights_.localPositions[index],
                        sanitized.position) ||
        !quatExactEqual(spotLights_.localRotations[index],
                        sanitized.rotation) ||
        !vec3ExactEqual(spotLights_.colors[index], sanitized.color) ||
        spotLights_.intensities[index] != sanitized.intensity ||
        spotLights_.ranges[index] != sanitized.range ||
        spotLights_.innerConeAngles[index] != sanitized.innerConeAngleRadians ||
        spotLights_.outerConeAngles[index] != sanitized.outerConeAngleRadians;
    spotLights_.names[index].assign(sanitized.name.data(),
                                    sanitized.name.size());
    spotLights_.localPositions[index] = sanitized.position;
    spotLights_.localRotations[index] = sanitized.rotation;
    spotLights_.colors[index] = sanitized.color;
    spotLights_.intensities[index] = sanitized.intensity;
    spotLights_.ranges[index] = sanitized.range;
    spotLights_.innerConeAngles[index] = sanitized.innerConeAngleRadians;
    spotLights_.outerConeAngles[index] = sanitized.outerConeAngleRadians;
    spotLights_.enabled[index] = sanitized.enabled ? 1u : 0u;
    lightTopologyDirty_ |= topologyChanged;
    lightDataDirty_ |= topologyChanged || derivedDataChanged;
    return true;
  }
  }

  return false;
}

bool SceneGraph::setLightNode(LightId id, NodeId node,
                              bool preserveWorldTransform) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(id) || !nodeSlotValid(node)) {
    return false;
  }

  LightDesc localDesc{};
  if (!getLightDesc(id, localDesc)) {
    return false;
  }
  if (preserveWorldTransform) {
    (void)syncWorldTransforms();
    LightDesc worldDesc{};
    if (!getCachedLightWorldDesc(id, worldDesc)) {
      return false;
    }
    localDesc =
        lightLocalFromWorld(worldDesc, nodes_.worldFromRoot[indexOf(node)]);
  }

  switch (id.type) {
  case LightType::Directional:
    if (!directionalSlotValid(id)) {
      return false;
    }
    directionalLights_.node[indexOf(id)] = indexOf(node);
    directionalLights_.localPositions[indexOf(id)] = localDesc.position;
    directionalLights_.localRotations[indexOf(id)] = localDesc.rotation;
    break;
  case LightType::Point:
    if (!pointSlotValid(id)) {
      return false;
    }
    pointLights_.node[indexOf(id)] = indexOf(node);
    pointLights_.localPositions[indexOf(id)] = localDesc.position;
    pointLights_.localRotations[indexOf(id)] = localDesc.rotation;
    break;
  case LightType::Spot:
    if (!spotSlotValid(id)) {
      return false;
    }
    spotLights_.node[indexOf(id)] = indexOf(node);
    spotLights_.localPositions[indexOf(id)] = localDesc.position;
    spotLights_.localRotations[indexOf(id)] = localDesc.rotation;
    break;
  }
  lightDataDirty_ = true;
  return true;
}

Result<NodeId, std::string>
SceneGraph::instantiatePrefab(const ScenePrefab &prefab, NodeId parent,
                              const ScenePrefabAssets &assets,
                              SceneInstantiationMap *outMap) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!nodeSlotValid(parent)) {
    return Result<NodeId, std::string>::makeError(
        "SceneGraph::instantiatePrefab: parent node is invalid");
  }
  if (prefab.nodes.empty()) {
    return Result<NodeId, std::string>::makeError(
        "SceneGraph::instantiatePrefab: prefab has no nodes");
  }

  SceneInstantiationMap localMap(memory_);
  localMap.nodes.resize(prefab.nodes.size(), kInvalidNodeId);
  localMap.renderables.reserve(prefab.renderables.size());
  localMap.lights.reserve(prefab.lights.size());

  NodeId firstRoot = kInvalidNodeId;
  for (uint32_t prefabNodeIndex = 0; prefabNodeIndex < prefab.nodes.size();
       ++prefabNodeIndex) {
    const ScenePrefabNode &prefabNode = prefab.nodes[prefabNodeIndex];
    const NodeId parentNode = prefabNode.parentIndex == kInvalidScenePrefabIndex
                                  ? parent
                                  : localMap.nodes[prefabNode.parentIndex];
    auto createResult =
        createNode(parentNode, prefabNode.name, prefabNode.localFromParent);
    if (createResult.hasError()) {
      return Result<NodeId, std::string>::makeError(createResult.error());
    }
    localMap.nodes[prefabNodeIndex] = createResult.value();
    if (!isValid(firstRoot)) {
      firstRoot = createResult.value();
    }
  }

  MaterialRef fallbackMaterial = kInvalidMaterialRef;
  for (const MaterialRef material : assets.materials) {
    if (isValid(material)) {
      fallbackMaterial = material;
      break;
    }
  }

  for (const ScenePrefabRenderable &prefabRenderable : prefab.renderables) {
    if (prefabRenderable.nodeIndex >= localMap.nodes.size()) {
      return Result<NodeId, std::string>::makeError(
          "SceneGraph::instantiatePrefab: prefab renderable node index is "
          "invalid");
    }
    if (prefabRenderable.meshIndex >= assets.models.size() ||
        !isValid(assets.models[prefabRenderable.meshIndex])) {
      return Result<NodeId, std::string>::makeError(
          "SceneGraph::instantiatePrefab: prefab model asset is unresolved");
    }
    MaterialRef material = fallbackMaterial;
    if (prefabRenderable.materialIndex < assets.materials.size() &&
        isValid(assets.materials[prefabRenderable.materialIndex])) {
      material = assets.materials[prefabRenderable.materialIndex];
    }
    if (!isValid(material)) {
      return Result<NodeId, std::string>::makeError(
          "SceneGraph::instantiatePrefab: prefab material asset is unresolved");
    }
    auto renderableResult =
        addRenderable(localMap.nodes[prefabRenderable.nodeIndex],
                      assets.models[prefabRenderable.meshIndex], material);
    if (renderableResult.hasError()) {
      return Result<NodeId, std::string>::makeError(renderableResult.error());
    }
    localMap.renderables.push_back(renderableResult.value());
  }

  for (const ScenePrefabLight &prefabLight : prefab.lights) {
    if (prefabLight.nodeIndex >= localMap.nodes.size()) {
      return Result<NodeId, std::string>::makeError(
          "SceneGraph::instantiatePrefab: prefab light node index is invalid");
    }
    auto lightResult =
        addLight(localMap.nodes[prefabLight.nodeIndex], prefabLight.light);
    if (lightResult.hasError()) {
      return Result<NodeId, std::string>::makeError(lightResult.error());
    }
    localMap.lights.push_back(lightResult.value());
  }

  if (outMap != nullptr) {
    *outMap = std::move(localMap);
  }
  return Result<NodeId, std::string>::makeResult(firstRoot);
}

void SceneGraph::clearRenderables() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  renderableComponents_ = RenderableStore(memory_);
  renderableTopologyDirty_ = true;
  renderableTransformsDirty_ = false;
}

void SceneGraph::clearLights() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  directionalLights_ = DirectionalLightStore(memory_);
  pointLights_ = PointLightStore(memory_);
  spotLights_ = SpotLightStore(memory_);
  lightTopologyDirty_ = true;
  lightDataDirty_ = false;
}

} // namespace nuri
