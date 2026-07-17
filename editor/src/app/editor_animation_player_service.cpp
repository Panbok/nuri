#include "nuri/pch.h"

#include "nuri/app/editor_animation_player_service.h"

#include "nuri/core/log.h"
#include "nuri/scene/render_scene.h"
#include "nuri/scene_runtime/scene_runtime_host.h"
#include "nuri/ui/editor_services.h"

namespace nuri {
namespace {

[[nodiscard]] std::string_view clipDisplayName(const AnimationClipData &clip) {
  return clip.name.empty() ? std::string_view("Unnamed Clip")
                           : std::string_view(clip.name);
}

[[nodiscard]] std::string nodeDisplayName(const SceneGraph &graph,
                                          NodeId node) {
  std::string_view name{};
  if (graph.getNodeName(node, name) && !name.empty()) {
    return std::string(name);
  }
  return "Node #" + std::to_string(indexOf(node));
}

[[nodiscard]] std::string prefabNodeDisplayName(const ScenePrefab &prefab,
                                                uint32_t nodeIndex) {
  if (nodeIndex < prefab.nodes.size() &&
      !prefab.nodes[nodeIndex].name.empty()) {
    return std::string(prefab.nodes[nodeIndex].name);
  }
  return "Node #" + std::to_string(nodeIndex);
}

[[nodiscard]] float clampToClipDuration(const ScenePrefab &prefab,
                                        uint32_t clipIndex, float timeSeconds) {
  if (clipIndex >= prefab.animations.size()) {
    return 0.0f;
  }
  const float duration =
      std::max(prefab.animations[clipIndex].durationSeconds, 0.0f);
  return std::clamp(timeSeconds, 0.0f, duration);
}

[[nodiscard]] float clipDurationSeconds(const ScenePrefab &prefab,
                                        uint32_t clipIndex) {
  if (clipIndex >= prefab.animations.size()) {
    return 0.0f;
  }
  return std::max(prefab.animations[clipIndex].durationSeconds, 0.0f);
}

[[nodiscard]] float advancePlaybackTime(float timeSeconds,
                                        const AnimationClipData &clip,
                                        double deltaSeconds) {
  const float duration = std::max(clip.durationSeconds, 0.0f);
  if (duration <= 0.0f) {
    return 0.0f;
  }
  const float nextTime =
      std::max(0.0f, timeSeconds + static_cast<float>(deltaSeconds));
  const float wrapped = std::fmod(nextTime, duration);
  return wrapped >= 0.0f ? wrapped : wrapped + duration;
}

[[nodiscard]] bool isDescendantOrSelf(const ScenePrefab &prefab,
                                      uint32_t nodeIndex,
                                      uint32_t ancestorIndex) {
  if (nodeIndex >= prefab.nodes.size() ||
      ancestorIndex >= prefab.nodes.size()) {
    return false;
  }
  uint32_t current = nodeIndex;
  size_t traversed = 0u;
  while (current != kInvalidScenePrefabIndex && current < prefab.nodes.size()) {
    if (current == ancestorIndex) {
      return true;
    }
    if (traversed >= prefab.nodes.size()) {
      break;
    }
    current = prefab.nodes[current].parentIndex;
    ++traversed;
  }
  return false;
}

[[nodiscard]] std::string
makeRecordLabel(std::string_view instanceLabel, const ScenePrefab &prefab,
                std::span<const uint32_t> groupRoots) {
  if (groupRoots.size() != 1u) {
    return std::string(instanceLabel);
  }
  const std::string targetLabel = prefabNodeDisplayName(prefab, groupRoots[0]);
  if (targetLabel.empty() || targetLabel == instanceLabel) {
    return std::string(instanceLabel);
  }
  std::string label(instanceLabel);
  label += " / ";
  label += targetLabel;
  return label;
}

struct DisjointSet {
  explicit DisjointSet(size_t count, std::pmr::memory_resource *memory =
                                         std::pmr::get_default_resource())
      : parent(memory), rank(memory) {
    parent.resize(count);
    rank.resize(count, 0u);
    for (size_t i = 0; i < count; ++i) {
      parent[i] = static_cast<uint32_t>(i);
    }
  }

  [[nodiscard]] uint32_t find(uint32_t value) {
    if (parent[value] == value) {
      return value;
    }
    parent[value] = find(parent[value]);
    return parent[value];
  }

  void unite(uint32_t lhs, uint32_t rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs) {
      return;
    }
    if (rank[lhs] < rank[rhs]) {
      std::swap(lhs, rhs);
    }
    parent[rhs] = lhs;
    if (rank[lhs] == rank[rhs]) {
      ++rank[lhs];
    }
  }

  std::pmr::vector<uint32_t> parent;
  std::pmr::vector<uint8_t> rank;
};

struct GroupBuildState {
  explicit GroupBuildState(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : rootNodes(memory), controlledNodeMask(memory),
        controlledPrefabNodeIndices(memory), mappedNodes(memory),
        mappedRenderables(memory), availableClipMask(memory),
        clipInfos(memory) {}

  std::pmr::vector<uint32_t> rootNodes;
  std::pmr::vector<uint8_t> controlledNodeMask;
  std::pmr::vector<uint32_t> controlledPrefabNodeIndices;
  std::pmr::vector<NodeId> mappedNodes;
  std::pmr::vector<RenderableId> mappedRenderables;
  std::pmr::vector<uint8_t> availableClipMask;
  std::pmr::vector<EditorAnimationClipInfo> clipInfos;
};

[[nodiscard]] bool
renderableControlledByGroup(const ScenePrefab &prefab,
                            const GroupBuildState &group,
                            const ScenePrefabRenderable &renderable) {
  const auto controlsNode = [&group](uint32_t nodeIndex) {
    return nodeIndex < group.controlledNodeMask.size() &&
           group.controlledNodeMask[nodeIndex] != 0u;
  };
  if (controlsNode(renderable.nodeIndex)) {
    return true;
  }
  if (renderable.skinIndex >= prefab.skins.size()) {
    return false;
  }
  return std::any_of(
      prefab.skins[renderable.skinIndex].jointNodeIndices.begin(),
      prefab.skins[renderable.skinIndex].jointNodeIndices.end(), controlsNode);
}

} // namespace

struct EditorAnimationPlayerService::InstanceRecord {
  explicit InstanceRecord(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : label(memory), debugName(memory), instantiationMap(memory),
        clipInfos(memory), availableClipMask(memory),
        controlledPrefabNodeIndices(memory), mappedNodes(memory),
        mappedRenderables(memory) {}

  NodeId instanceRootNode = kInvalidNodeId;
  std::pmr::string label;
  std::pmr::string debugName;
  const ScenePrefab *prefab = nullptr;
  SceneInstantiationMap instantiationMap;
  std::pmr::vector<EditorAnimationClipInfo> clipInfos;
  std::pmr::vector<uint8_t> availableClipMask;
  std::pmr::vector<uint32_t> controlledPrefabNodeIndices;
  std::pmr::vector<NodeId> mappedNodes;
  std::pmr::vector<RenderableId> mappedRenderables;
  SimulationHandle simulation = kInvalidSimulationHandle;
  AnimationPoseSimulationParams params{};
};

EditorAnimationPlayerService::EditorAnimationPlayerService(
    RenderScene &scene, SceneRuntimeHost &sceneRuntime,
    SceneEditorSelectionState &selectionState,
    std::function<double()> currentTimeSeconds,
    std::function<uint64_t()> nextSimulationFrameIndex,
    std::pmr::memory_resource *memory)
    : scene_(&scene), sceneRuntime_(&sceneRuntime),
      selectionState_(selectionState),
      currentTimeSeconds_(std::move(currentTimeSeconds)),
      nextSimulationFrameIndex_(std::move(nextSimulationFrameIndex)),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      records_(memory_), recordIndexByNode_(memory_),
      recordIndexByRenderable_(memory_) {}

EditorAnimationPlayerService::~EditorAnimationPlayerService() = default;

void EditorAnimationPlayerService::onUpdate(double deltaTime) {
  if (!std::isfinite(deltaTime) || deltaTime <= 0.0) {
    return;
  }
  for (InstanceRecord &record : records_) {
    if (!isValid(record.simulation) || record.prefab == nullptr ||
        !recordHasClip(record, record.params.primary.clipIndex)) {
      continue;
    }
    SimulationState state = SimulationState::Stopped;
    if (!simulationState(record, state) || state != SimulationState::Running) {
      continue;
    }
    record.params.primary.timeSeconds = advancePlaybackTime(
        record.params.primary.timeSeconds,
        record.prefab->animations[record.params.primary.clipIndex], deltaTime);
    record.params.secondary.timeSeconds = computeSecondaryTime(record);
  }
}

void EditorAnimationPlayerService::clear() {
  for (InstanceRecord &record : records_) {
    if (isValid(record.simulation)) {
      (void)sceneRuntime_->destroyAnimationPoseSimulation(record.simulation);
      record.simulation = kInvalidSimulationHandle;
    }
  }
  records_.clear();
  recordIndexByNode_.clear();
  recordIndexByRenderable_.clear();
}

void EditorAnimationPlayerService::bindScene(RenderScene &scene,
                                             SceneRuntimeHost &sceneRuntime,
                                             bool destroyExistingSimulations) {
  if (scene_ == &scene && sceneRuntime_ == &sceneRuntime) {
    return;
  }
  if (destroyExistingSimulations) {
    clear();
  } else {
    records_.clear();
    recordIndexByNode_.clear();
    recordIndexByRenderable_.clear();
  }
  scene_ = &scene;
  sceneRuntime_ = &sceneRuntime;
}

void EditorAnimationPlayerService::registerPrefabInstance(
    std::string_view label, const ScenePrefab &prefab,
    const SceneInstantiationMap &instantiationMap, NodeId rootNode) {
  if (!isValid(rootNode) || prefab.animations.empty()) {
    return;
  }

  unregisterPrefabInstance(rootNode);

  const size_t nodeCount = prefab.nodes.size();
  std::pmr::vector<uint8_t> animatedNodeMask(nodeCount, uint8_t{0u}, memory_);
  for (const AnimationClipData &clip : prefab.animations) {
    for (const AnimationChannelData &channel : clip.channels) {
      if (channel.targetNodeIndex < nodeCount) {
        animatedNodeMask[channel.targetNodeIndex] = 1u;
      }
    }
  }

  std::pmr::vector<uint32_t> minimalAnimatedRoots(memory_);
  minimalAnimatedRoots.reserve(nodeCount);
  for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
    if (animatedNodeMask[nodeIndex] == 0u) {
      continue;
    }
    bool hasAnimatedAncestor = false;
    uint32_t current = prefab.nodes[nodeIndex].parentIndex;
    size_t traversed = 0u;
    while (current != kInvalidScenePrefabIndex && current < nodeCount) {
      if (animatedNodeMask[current] != 0u) {
        hasAnimatedAncestor = true;
        break;
      }
      if (traversed >= nodeCount) {
        break;
      }
      current = prefab.nodes[current].parentIndex;
      ++traversed;
    }
    if (!hasAnimatedAncestor) {
      minimalAnimatedRoots.push_back(nodeIndex);
    }
  }
  if (minimalAnimatedRoots.empty()) {
    return;
  }

  std::pmr::vector<uint32_t> owningRootByNode(
      nodeCount, kInvalidScenePrefabIndex, memory_);
  for (uint32_t rootIndex = 0u; rootIndex < minimalAnimatedRoots.size();
       ++rootIndex) {
    const uint32_t rootPrefabNodeIndex = minimalAnimatedRoots[rootIndex];
    for (uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex) {
      if (isDescendantOrSelf(prefab, nodeIndex, rootPrefabNodeIndex)) {
        owningRootByNode[nodeIndex] = rootIndex;
      }
    }
  }

  DisjointSet groups(minimalAnimatedRoots.size(), memory_);
  for (const ScenePrefabRenderable &renderable : prefab.renderables) {
    std::pmr::vector<uint32_t> relatedRoots(memory_);
    const auto appendRoot = [&relatedRoots](uint32_t rootIndex) {
      if (rootIndex == kInvalidScenePrefabIndex) {
        return;
      }
      if (std::find(relatedRoots.begin(), relatedRoots.end(), rootIndex) ==
          relatedRoots.end()) {
        relatedRoots.push_back(rootIndex);
      }
    };

    if (renderable.nodeIndex < owningRootByNode.size()) {
      appendRoot(owningRootByNode[renderable.nodeIndex]);
    }
    if (renderable.skinIndex < prefab.skins.size()) {
      for (const uint32_t jointNodeIndex :
           prefab.skins[renderable.skinIndex].jointNodeIndices) {
        if (jointNodeIndex < owningRootByNode.size()) {
          appendRoot(owningRootByNode[jointNodeIndex]);
        }
      }
    }
    for (size_t i = 1; i < relatedRoots.size(); ++i) {
      groups.unite(relatedRoots[0], relatedRoots[i]);
    }
  }

  PmrHashMap<uint32_t, size_t> componentIndexByRepresentative(memory_);
  std::pmr::vector<GroupBuildState> builtGroups(memory_);
  for (uint32_t rootIndex = 0u; rootIndex < minimalAnimatedRoots.size();
       ++rootIndex) {
    const uint32_t representative = groups.find(rootIndex);
    const auto existing = componentIndexByRepresentative.find(representative);
    if (existing != componentIndexByRepresentative.end()) {
      builtGroups[existing->second].rootNodes.push_back(
          minimalAnimatedRoots[rootIndex]);
      continue;
    }
    const size_t groupIndex = builtGroups.size();
    builtGroups.emplace_back(memory_);
    builtGroups.back().rootNodes.push_back(minimalAnimatedRoots[rootIndex]);
    componentIndexByRepresentative.emplace(representative, groupIndex);
  }

  for (uint32_t nodeIndex = 0u; nodeIndex < owningRootByNode.size();
       ++nodeIndex) {
    const uint32_t rootIndex = owningRootByNode[nodeIndex];
    if (rootIndex == kInvalidScenePrefabIndex) {
      continue;
    }
    const uint32_t representative = groups.find(rootIndex);
    const auto groupIt = componentIndexByRepresentative.find(representative);
    if (groupIt == componentIndexByRepresentative.end()) {
      continue;
    }
    GroupBuildState &group = builtGroups[groupIt->second];
    if (group.controlledNodeMask.empty()) {
      group.controlledNodeMask.assign(nodeCount, uint8_t{0u});
    }
    group.controlledNodeMask[nodeIndex] = 1u;
    group.controlledPrefabNodeIndices.push_back(nodeIndex);
    if (nodeIndex < instantiationMap.nodes.size()) {
      const NodeId runtimeNode = instantiationMap.nodes[nodeIndex];
      if (isValid(runtimeNode)) {
        group.mappedNodes.push_back(runtimeNode);
      }
    }
  }

  for (GroupBuildState &group : builtGroups) {
    group.availableClipMask.assign(prefab.animations.size(), uint8_t{0u});
    group.clipInfos.reserve(prefab.animations.size());
    for (uint32_t clipIndex = 0u; clipIndex < prefab.animations.size();
         ++clipIndex) {
      const AnimationClipData &clip = prefab.animations[clipIndex];
      const bool affectsGroup = std::any_of(
          clip.channels.begin(), clip.channels.end(),
          [&group](const AnimationChannelData &channel) {
            return channel.targetNodeIndex < group.controlledNodeMask.size() &&
                   group.controlledNodeMask[channel.targetNodeIndex] != 0u;
          });
      if (!affectsGroup) {
        continue;
      }
      group.availableClipMask[clipIndex] = 1u;
      group.clipInfos.push_back(EditorAnimationClipInfo{
          .clipIndex = clipIndex,
          .name = clipDisplayName(clip),
          .durationSeconds = std::max(clip.durationSeconds, 0.0f),
      });
    }

    for (uint32_t prefabRenderableIndex = 0u;
         prefabRenderableIndex < prefab.renderables.size();
         ++prefabRenderableIndex) {
      const ScenePrefabRenderable &renderable =
          prefab.renderables[prefabRenderableIndex];
      if (!renderableControlledByGroup(prefab, group, renderable) ||
          prefabRenderableIndex >= instantiationMap.renderables.size()) {
        continue;
      }
      const RenderableId runtimeRenderable =
          instantiationMap.renderables[prefabRenderableIndex];
      if (isValid(runtimeRenderable)) {
        group.mappedRenderables.push_back(runtimeRenderable);
      }
    }
  }

  for (GroupBuildState &group : builtGroups) {
    if (group.clipInfos.empty()) {
      continue;
    }

    records_.emplace_back(memory_);
    InstanceRecord &record = records_.back();
    record.instanceRootNode = rootNode;
    const std::string recordLabel =
        makeRecordLabel(label, prefab, group.rootNodes);
    record.label.assign(recordLabel.data(), recordLabel.size());
    record.debugName = record.label;
    record.prefab = &prefab;
    record.instantiationMap.nodes.assign(instantiationMap.nodes.begin(),
                                         instantiationMap.nodes.end());
    record.instantiationMap.renderables.assign(
        instantiationMap.renderables.begin(),
        instantiationMap.renderables.end());
    record.instantiationMap.lights.assign(instantiationMap.lights.begin(),
                                          instantiationMap.lights.end());
    record.clipInfos.assign(group.clipInfos.begin(), group.clipInfos.end());
    record.availableClipMask.assign(group.availableClipMask.begin(),
                                    group.availableClipMask.end());
    record.controlledPrefabNodeIndices.assign(
        group.controlledPrefabNodeIndices.begin(),
        group.controlledPrefabNodeIndices.end());
    record.mappedNodes.assign(group.mappedNodes.begin(),
                              group.mappedNodes.end());
    record.mappedRenderables.assign(group.mappedRenderables.begin(),
                                    group.mappedRenderables.end());
    record.params.primary.clipIndex = defaultClipIndex(record);
    record.params.primary.playbackMode = AnimationPosePlaybackMode::Loop;
    record.params.primary.playing = true;
    record.params.secondary.clipIndex = kInvalidScenePrefabIndex;
    record.params.secondary.playbackMode = AnimationPosePlaybackMode::Loop;
    record.params.secondary.playing = true;
    record.params.blendWeight = 0.0f;
    record.params.blendMode = AnimationPoseBlendMode::Single;
    record.params.blendSyncMode = AnimationPoseBlendSyncMode::NormalizedTime;
  }

  rebuildSelectionMaps();
}

void EditorAnimationPlayerService::unregisterPrefabInstance(NodeId rootNode) {
  if (!isValid(rootNode)) {
    return;
  }

  auto removeIt =
      std::remove_if(records_.begin(), records_.end(),
                     [this, rootNode](InstanceRecord &record) {
                       if (record.instanceRootNode != rootNode) {
                         return false;
                       }
                       if (isValid(record.simulation)) {
                         (void)sceneRuntime_->destroyAnimationPoseSimulation(
                             record.simulation);
                         record.simulation = kInvalidSimulationHandle;
                       }
                       return true;
                     });
  if (removeIt == records_.end()) {
    return;
  }
  records_.erase(removeIt, records_.end());
  rebuildSelectionMaps();
}

bool EditorAnimationPlayerService::startPrefabInstancePlayback(
    NodeId rootNode, const AnimationPoseSimulationParams &params,
    std::string_view simulationDebugName) {
  bool updatedAny = false;
  for (InstanceRecord &record : records_) {
    if (record.instanceRootNode != rootNode || record.prefab == nullptr ||
        record.clipInfos.empty()) {
      continue;
    }

    AnimationPoseSimulationParams seeded = params;
    sanitizeAnimationPoseSimulationParams(seeded);
    seeded.primary.clipIndex =
        chooseClipIndex(record, seeded.primary.clipIndex);
    seeded.primary.timeSeconds = clampToClipDuration(
        *record.prefab, seeded.primary.clipIndex, seeded.primary.timeSeconds);

    if (recordHasClip(record, seeded.secondary.clipIndex) &&
        seeded.secondary.clipIndex != seeded.primary.clipIndex) {
      seeded.secondary.timeSeconds =
          clampToClipDuration(*record.prefab, seeded.secondary.clipIndex,
                              seeded.secondary.timeSeconds);
    } else if (seeded.blendMode == AnimationPoseBlendMode::Lerp &&
               seeded.blendWeight > 0.0f) {
      const auto secondaryIt =
          std::find_if(record.clipInfos.begin(), record.clipInfos.end(),
                       [&seeded](const EditorAnimationClipInfo &clip) {
                         return clip.clipIndex != seeded.primary.clipIndex;
                       });
      if (secondaryIt != record.clipInfos.end()) {
        seeded.secondary.clipIndex = secondaryIt->clipIndex;
        seeded.secondary.timeSeconds =
            clampToClipDuration(*record.prefab, seeded.secondary.clipIndex,
                                seeded.secondary.timeSeconds);
      } else {
        seeded.secondary.clipIndex = kInvalidScenePrefabIndex;
        seeded.blendMode = AnimationPoseBlendMode::Single;
        seeded.blendWeight = 0.0f;
      }
    } else if (!recordHasClip(record, seeded.secondary.clipIndex) ||
               seeded.secondary.clipIndex == seeded.primary.clipIndex) {
      seeded.secondary.clipIndex = kInvalidScenePrefabIndex;
    }

    if (seeded.secondary.clipIndex == seeded.primary.clipIndex) {
      seeded.secondary.clipIndex = kInvalidScenePrefabIndex;
      seeded.blendMode = AnimationPoseBlendMode::Single;
      seeded.blendWeight = 0.0f;
    }

    if (!simulationDebugName.empty()) {
      record.debugName.assign(simulationDebugName.data(),
                              simulationDebugName.size());
      if (!record.controlledPrefabNodeIndices.empty()) {
        record.debugName += "/";
        record.debugName += prefabNodeDisplayName(
            *record.prefab, record.controlledPrefabNodeIndices.front());
      }
    } else {
      record.debugName = record.label;
    }

    record.params = seeded;
    record.params.secondary.timeSeconds = computeSecondaryTime(record);
    if (!ensureSimulation(record, !record.params.primary.playing) ||
        !pushRuntimeParams(record)) {
      return false;
    }
    if (record.params.primary.playing) {
      if (!sceneRuntime_->simulations().resume(record.simulation)) {
        return false;
      }
    } else if (!sceneRuntime_->simulations().pause(record.simulation)) {
      return false;
    }
    updatedAny = true;
  }
  return updatedAny;
}

bool EditorAnimationPlayerService::hasAnimatedSelection() const {
  return selectedRecord() != nullptr;
}

EditorAnimationPlayerView EditorAnimationPlayerService::selectedView() const {
  const bool hasRenderableSelection = isValid(selectionState_.renderableId);
  const bool hasNodeSelection = isValid(selectionState_.node);
  if (!hasRenderableSelection && !hasNodeSelection) {
    return EditorAnimationPlayerView{
        .availability = EditorAnimationPlayerAvailability::NoSelection,
    };
  }

  const InstanceRecord *record = selectedRecord();
  if (record == nullptr || record->prefab == nullptr) {
    return EditorAnimationPlayerView{
        .availability = EditorAnimationPlayerAvailability::NotAnimated,
        .selectionLabel =
            hasNodeSelection
                ? nodeDisplayName(scene_->graph(), selectionState_.node)
                : std::string{},
    };
  }

  SimulationState state = SimulationState::Stopped;
  const bool hasSimulation =
      isValid(record->simulation) && simulationState(*record, state);
  return EditorAnimationPlayerView{
      .availability = EditorAnimationPlayerAvailability::Animated,
      .instanceLabel = std::string(record->label),
      .selectionLabel = hasNodeSelection ? nodeDisplayName(scene_->graph(),
                                                           selectionState_.node)
                                         : std::string{},
      .clips = std::span<const EditorAnimationClipInfo>(
          record->clipInfos.data(), record->clipInfos.size()),
      .primaryClipIndex = record->params.primary.clipIndex,
      .secondaryClipIndex = record->params.secondary.clipIndex,
      .blendWeight = record->params.blendWeight,
      .timelineTimeSeconds = record->params.primary.timeSeconds,
      .timelineDurationSeconds = clipDurationSeconds(
          *record->prefab, record->params.primary.clipIndex),
      .hasSimulation = hasSimulation,
      .running = hasSimulation && state == SimulationState::Running,
      .paused = hasSimulation && state == SimulationState::Paused,
      .blendSyncMode = record->params.blendSyncMode,
  };
}

bool EditorAnimationPlayerService::startSelectionPlayback() {
  InstanceRecord *record = selectedRecord();
  if (record == nullptr || record->prefab == nullptr ||
      !recordHasClip(*record, record->params.primary.clipIndex)) {
    return false;
  }
  if (!ensureSimulation(*record, false)) {
    return false;
  }
  record->params.primary.timeSeconds =
      clampToClipDuration(*record->prefab, record->params.primary.clipIndex,
                          record->params.primary.timeSeconds);
  record->params.secondary.timeSeconds = computeSecondaryTime(*record);
  if (!pushRuntimeParams(*record)) {
    return false;
  }
  return sceneRuntime_->simulations().resume(record->simulation);
}

bool EditorAnimationPlayerService::pauseSelectionPlayback() {
  InstanceRecord *record = selectedRecord();
  return record != nullptr && isValid(record->simulation) &&
         sceneRuntime_->simulations().pause(record->simulation);
}

bool EditorAnimationPlayerService::restartSelectionPlayback() {
  InstanceRecord *record = selectedRecord();
  if (record == nullptr || record->prefab == nullptr ||
      !recordHasClip(*record, record->params.primary.clipIndex)) {
    return false;
  }
  record->params.primary.timeSeconds = 0.0f;
  record->params.secondary.timeSeconds = computeSecondaryTime(*record);
  if (!ensureSimulation(*record, false) || !pushRuntimeParams(*record)) {
    return false;
  }
  return sceneRuntime_->simulations().resume(record->simulation);
}

bool EditorAnimationPlayerService::seekSelectionPlayback(float timeSeconds) {
  InstanceRecord *record = selectedRecord();
  if (record == nullptr || record->prefab == nullptr ||
      !recordHasClip(*record, record->params.primary.clipIndex)) {
    return false;
  }
  record->params.primary.timeSeconds = clampToClipDuration(
      *record->prefab, record->params.primary.clipIndex, timeSeconds);
  record->params.secondary.timeSeconds = computeSecondaryTime(*record);
  if (!ensureSimulation(*record, true)) {
    return false;
  }
  return pushRuntimeParams(*record);
}

bool EditorAnimationPlayerService::setSelectionPrimaryClip(uint32_t clipIndex) {
  InstanceRecord *record = selectedRecord();
  if (record == nullptr || record->prefab == nullptr ||
      !recordHasClip(*record, clipIndex)) {
    return false;
  }
  record->params.primary.clipIndex = clipIndex;
  record->params.primary.timeSeconds = clampToClipDuration(
      *record->prefab, clipIndex, record->params.primary.timeSeconds);
  if (record->params.secondary.clipIndex == clipIndex ||
      !recordHasClip(*record, record->params.secondary.clipIndex)) {
    record->params.secondary.clipIndex = kInvalidScenePrefabIndex;
    record->params.blendWeight = 0.0f;
    record->params.blendMode = AnimationPoseBlendMode::Single;
  }
  record->params.secondary.timeSeconds = computeSecondaryTime(*record);
  if (!isValid(record->simulation)) {
    return true;
  }
  return pushRuntimeParams(*record);
}

bool EditorAnimationPlayerService::setSelectionSecondaryClip(
    std::optional<uint32_t> clipIndex) {
  InstanceRecord *record = selectedRecord();
  if (record == nullptr || record->prefab == nullptr) {
    return false;
  }
  if (!clipIndex.has_value()) {
    record->params.secondary.clipIndex = kInvalidScenePrefabIndex;
    record->params.blendWeight = 0.0f;
    record->params.blendMode = AnimationPoseBlendMode::Single;
  } else {
    if (!recordHasClip(*record, *clipIndex) ||
        *clipIndex == record->params.primary.clipIndex) {
      return false;
    }
    record->params.secondary.clipIndex = *clipIndex;
    record->params.blendMode = AnimationPoseBlendMode::Lerp;
  }
  record->params.secondary.timeSeconds = computeSecondaryTime(*record);
  if (!isValid(record->simulation)) {
    return true;
  }
  return pushRuntimeParams(*record);
}

bool EditorAnimationPlayerService::setSelectionBlendWeight(float blendWeight) {
  InstanceRecord *record = selectedRecord();
  if (record == nullptr) {
    return false;
  }
  record->params.blendWeight = std::clamp(blendWeight, 0.0f, 1.0f);
  record->params.blendMode =
      record->params.secondary.clipIndex != kInvalidScenePrefabIndex
          ? AnimationPoseBlendMode::Lerp
          : AnimationPoseBlendMode::Single;
  sanitizeAnimationPoseSimulationParams(record->params);
  record->params.secondary.timeSeconds = computeSecondaryTime(*record);
  if (!isValid(record->simulation)) {
    return true;
  }
  return pushRuntimeParams(*record);
}

std::optional<size_t>
EditorAnimationPlayerService::selectedRecordIndex() const {
  if (isValid(selectionState_.renderableId)) {
    if (const auto it =
            recordIndexByRenderable_.find(selectionState_.renderableId);
        it != recordIndexByRenderable_.end()) {
      return it->second;
    }
  }
  if (isValid(selectionState_.node)) {
    if (const auto it = recordIndexByNode_.find(selectionState_.node);
        it != recordIndexByNode_.end()) {
      return it->second;
    }
  }
  return std::nullopt;
}

const EditorAnimationPlayerService::InstanceRecord *
EditorAnimationPlayerService::selectedRecord() const {
  const auto recordIndex = selectedRecordIndex();
  return recordIndex.has_value() ? &records_[*recordIndex] : nullptr;
}

EditorAnimationPlayerService::InstanceRecord *
EditorAnimationPlayerService::selectedRecord() {
  const auto recordIndex = selectedRecordIndex();
  return recordIndex.has_value() ? &records_[*recordIndex] : nullptr;
}

bool EditorAnimationPlayerService::recordHasClip(const InstanceRecord &record,
                                                 uint32_t clipIndex) const {
  return clipIndex < record.availableClipMask.size() &&
         record.availableClipMask[clipIndex] != 0u;
}

uint32_t EditorAnimationPlayerService::defaultClipIndex(
    const InstanceRecord &record) const {
  return !record.clipInfos.empty() ? record.clipInfos.front().clipIndex : 0u;
}

uint32_t EditorAnimationPlayerService::chooseClipIndex(
    const InstanceRecord &record, uint32_t preferredClipIndex) const {
  return recordHasClip(record, preferredClipIndex) ? preferredClipIndex
                                                   : defaultClipIndex(record);
}

void EditorAnimationPlayerService::rebuildSelectionMaps() {
  recordIndexByNode_.clear();
  recordIndexByRenderable_.clear();
  for (size_t recordIndex = 0; recordIndex < records_.size(); ++recordIndex) {
    const InstanceRecord &record = records_[recordIndex];
    for (NodeId node : record.mappedNodes) {
      if (isValid(node)) {
        recordIndexByNode_[node] = recordIndex;
      }
    }
    for (RenderableId renderable : record.mappedRenderables) {
      if (isValid(renderable)) {
        recordIndexByRenderable_[renderable] = recordIndex;
      }
    }
  }
}

bool EditorAnimationPlayerService::ensureSimulation(InstanceRecord &record,
                                                    bool paused) {
  if (record.prefab == nullptr) {
    return false;
  }
  if (isValid(record.simulation)) {
    SimulationState state = SimulationState::Stopped;
    if (simulationState(record, state)) {
      if (paused && state != SimulationState::Paused) {
        (void)sceneRuntime_->simulations().pause(record.simulation);
      }
      return true;
    }
    record.simulation = kInvalidSimulationHandle;
  }

  auto commitResult = scene_->commit();
  if (commitResult.hasError()) {
    NURI_LOG_WARNING(
        "EditorAnimationPlayerService::ensureSimulation: scene commit failed: "
        "%s",
        commitResult.error().c_str());
    return false;
  }
  (void)sceneRuntime_->tick({
      .frameDeltaSeconds = 0.0,
      .absoluteTimeSeconds = currentTimeSeconds_ ? currentTimeSeconds_() : 0.0,
      .frameIndex =
          nextSimulationFrameIndex_ ? nextSimulationFrameIndex_() : 0u,
  });

  AnimationPoseSimulationParams params = record.params;
  params.primary.clipIndex = chooseClipIndex(record, params.primary.clipIndex);
  params.primary.timeSeconds = clampToClipDuration(
      *record.prefab, params.primary.clipIndex, params.primary.timeSeconds);
  record.params.primary.clipIndex = params.primary.clipIndex;
  record.params.primary.timeSeconds = params.primary.timeSeconds;
  record.params.secondary.timeSeconds = computeSecondaryTime(record);
  params.secondary.timeSeconds = record.params.secondary.timeSeconds;
  auto createResult = sceneRuntime_->createAnimationPoseSimulation(
      AnimationPoseSimulationCreateInfo{
          .prefab = record.prefab,
          .instantiationMap = &record.instantiationMap,
          .controlledPrefabNodeIndices = std::span<const uint32_t>(
              record.controlledPrefabNodeIndices.data(),
              record.controlledPrefabNodeIndices.size()),
          .rootNode = record.instanceRootNode,
          .debugName = record.debugName,
          .params = params,
      });
  if (createResult.hasError()) {
    NURI_LOG_WARNING(
        "EditorAnimationPlayerService::ensureSimulation: failed to create "
        "simulation for '%s': %s",
        record.label.c_str(), createResult.error().c_str());
    return false;
  }

  record.simulation = createResult.value();
  if (paused && !sceneRuntime_->simulations().pause(record.simulation)) {
    NURI_LOG_WARNING(
        "EditorAnimationPlayerService::ensureSimulation: failed to pause new "
        "simulation for '%s'",
        record.label.c_str());
  }
  return true;
}

bool EditorAnimationPlayerService::pushRuntimeParams(InstanceRecord &record) {
  if (!isValid(record.simulation) || record.prefab == nullptr) {
    return false;
  }

  AnimationPoseSimulationParams params = record.params;
  params.primary.clipIndex = chooseClipIndex(record, params.primary.clipIndex);
  params.primary.timeSeconds = clampToClipDuration(
      *record.prefab, params.primary.clipIndex, params.primary.timeSeconds);
  record.params.primary.clipIndex = params.primary.clipIndex;
  record.params.primary.timeSeconds = params.primary.timeSeconds;
  if (!recordHasClip(record, params.secondary.clipIndex) ||
      params.secondary.clipIndex == params.primary.clipIndex) {
    params.secondary.clipIndex = kInvalidScenePrefabIndex;
    params.blendMode = AnimationPoseBlendMode::Single;
    params.blendWeight = 0.0f;
  }
  record.params.secondary.clipIndex = params.secondary.clipIndex;
  record.params.secondary.timeSeconds = computeSecondaryTime(record);
  params.secondary.timeSeconds = record.params.secondary.timeSeconds;
  sanitizeAnimationPoseSimulationParams(params);
  auto validateResult =
      validateAnimationPoseSimulationParams(*record.prefab, params);
  if (validateResult.hasError()) {
    NURI_LOG_WARNING(
        "EditorAnimationPlayerService::pushRuntimeParams: invalid params for "
        "'%s': %s",
        record.label.c_str(), validateResult.error().c_str());
    return false;
  }
  if (!sceneRuntime_->simulations().setParams(record.simulation,
                                              asBytes(params))) {
    NURI_LOG_WARNING(
        "EditorAnimationPlayerService::pushRuntimeParams: failed to update "
        "simulation params for '%s'",
        record.label.c_str());
    return false;
  }
  record.params = params;
  return true;
}

bool EditorAnimationPlayerService::simulationState(const InstanceRecord &record,
                                                   SimulationState &out) const {
  return isValid(record.simulation) &&
         sceneRuntime_->simulations().getState(record.simulation, out);
}

float EditorAnimationPlayerService::computeSecondaryTime(
    const InstanceRecord &record) {
  if (record.prefab == nullptr ||
      record.params.secondary.clipIndex == kInvalidScenePrefabIndex ||
      record.params.secondary.clipIndex >= record.prefab->animations.size()) {
    return 0.0f;
  }
  if (record.params.blendSyncMode == AnimationPoseBlendSyncMode::Independent) {
    return clampToClipDuration(*record.prefab,
                               record.params.secondary.clipIndex,
                               record.params.secondary.timeSeconds);
  }

  const uint32_t primaryClipIndex = record.params.primary.clipIndex;
  if (primaryClipIndex >= record.prefab->animations.size()) {
    return 0.0f;
  }
  const float primaryDuration = std::max(
      record.prefab->animations[primaryClipIndex].durationSeconds, 0.0f);
  if (primaryDuration <= 0.0f) {
    return 0.0f;
  }
  const float secondaryDuration =
      std::max(record.prefab->animations[record.params.secondary.clipIndex]
                   .durationSeconds,
               0.0f);
  const float normalized = glm::clamp(
      record.params.primary.timeSeconds / primaryDuration, 0.0f, 1.0f);
  return normalized * secondaryDuration;
}

} // namespace nuri
