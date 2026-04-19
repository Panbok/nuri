#pragma once

#include "nuri/core/containers/hash_map.h"
#include "nuri/defines.h"
#include "nuri/scene/scene_prefab.h"
#include "nuri/sim/animation_pose_simulation.h"
#include "nuri/sim/simulation_handles.h"

#include <cstdint>
#include <functional>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri {

class RenderScene;
class SceneRuntimeHost;
struct SceneEditorSelectionState;

enum class EditorAnimationPlayerAvailability : uint8_t {
  NoSelection = 0,
  NotAnimated = 1,
  Animated = 2,
};

struct EditorAnimationClipInfo {
  uint32_t clipIndex = 0u;
  std::string_view name{};
  float durationSeconds = 0.0f;
};

struct EditorAnimationPlayerView {
  EditorAnimationPlayerAvailability availability =
      EditorAnimationPlayerAvailability::NoSelection;
  std::string instanceLabel{};
  std::string selectionLabel{};
  std::span<const EditorAnimationClipInfo> clips{};
  uint32_t primaryClipIndex = 0u;
  uint32_t secondaryClipIndex = kInvalidScenePrefabIndex;
  float blendWeight = 0.0f;
  float timelineTimeSeconds = 0.0f;
  float timelineDurationSeconds = 0.0f;
  bool hasSimulation = false;
  bool running = false;
  bool paused = false;
  AnimationPoseBlendSyncMode blendSyncMode =
      AnimationPoseBlendSyncMode::NormalizedTime;

  [[nodiscard]] bool hasSecondaryClipSelection() const noexcept {
    return secondaryClipIndex != kInvalidScenePrefabIndex;
  }
};

class EditorAnimationPlayerService final {
public:
  EditorAnimationPlayerService(
      RenderScene &scene, SceneRuntimeHost &sceneRuntime,
      SceneEditorSelectionState &selectionState,
      std::function<double()> currentTimeSeconds,
      std::function<uint64_t()> nextSimulationFrameIndex,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~EditorAnimationPlayerService();

  EditorAnimationPlayerService(const EditorAnimationPlayerService &) = delete;
  EditorAnimationPlayerService &
  operator=(const EditorAnimationPlayerService &) = delete;
  EditorAnimationPlayerService(EditorAnimationPlayerService &&) = delete;
  EditorAnimationPlayerService &
  operator=(EditorAnimationPlayerService &&) = delete;

  void onUpdate(double deltaTime);
  void clear();

  void registerPrefabInstance(std::string_view label, const ScenePrefab &prefab,
                              const SceneInstantiationMap &instantiationMap,
                              NodeId rootNode);
  void unregisterPrefabInstance(NodeId rootNode);
  [[nodiscard]] bool
  startPrefabInstancePlayback(NodeId rootNode,
                              const AnimationPoseSimulationParams &params,
                              std::string_view simulationDebugName);

  [[nodiscard]] bool hasAnimatedSelection() const;
  [[nodiscard]] EditorAnimationPlayerView selectedView() const;

  [[nodiscard]] bool startSelectionPlayback();
  [[nodiscard]] bool pauseSelectionPlayback();
  [[nodiscard]] bool restartSelectionPlayback();
  [[nodiscard]] bool seekSelectionPlayback(float timeSeconds);
  [[nodiscard]] bool setSelectionPrimaryClip(uint32_t clipIndex);
  [[nodiscard]] bool
  setSelectionSecondaryClip(std::optional<uint32_t> clipIndex);
  [[nodiscard]] bool setSelectionBlendWeight(float blendWeight);

private:
  struct InstanceRecord;

  [[nodiscard]] std::optional<size_t> selectedRecordIndex() const;
  [[nodiscard]] const InstanceRecord *selectedRecord() const;
  [[nodiscard]] InstanceRecord *selectedRecord();
  [[nodiscard]] bool recordHasClip(const InstanceRecord &record,
                                   uint32_t clipIndex) const;
  [[nodiscard]] uint32_t defaultClipIndex(const InstanceRecord &record) const;
  [[nodiscard]] uint32_t chooseClipIndex(const InstanceRecord &record,
                                         uint32_t preferredClipIndex) const;
  void rebuildSelectionMaps();
  [[nodiscard]] bool ensureSimulation(InstanceRecord &record, bool paused);
  [[nodiscard]] bool pushRuntimeParams(InstanceRecord &record);
  [[nodiscard]] bool simulationState(const InstanceRecord &record,
                                     SimulationState &out) const;
  [[nodiscard]] static float computeSecondaryTime(const InstanceRecord &record);

  RenderScene &scene_;
  SceneRuntimeHost &sceneRuntime_;
  SceneEditorSelectionState &selectionState_;
  std::function<double()> currentTimeSeconds_;
  std::function<uint64_t()> nextSimulationFrameIndex_;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::pmr::vector<InstanceRecord> records_;
  PmrHashMap<NodeId, size_t> recordIndexByNode_;
  PmrHashMap<RenderableId, size_t> recordIndexByRenderable_;
};

} // namespace nuri
