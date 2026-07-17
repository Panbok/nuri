#pragma once

#include "nuri/app/editor_scene_context.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/math/types.h"
#include "nuri/resources/async/asset_system.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/scene_importer.h"
#include "nuri/scene/scene_graph.h"
#include "nuri/scene/scene_prefab.h"
#include "nuri/sim/animation_pose_simulation.h"
#include "nuri/sim/simulation_handles.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <memory_resource>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

class EditorRuntime;
class ResourceManager;

using EditorSceneId = std::string;

struct EditorSceneInfo {
  EditorSceneId id{};
  std::string label{};
  int32_t sortOrder = 0;
  bool initiallySelected = false;
};

struct EditorSceneEntry {
  EditorSceneInfo info{};
  bool prepared = false;
  bool active = false;
};

struct EditorSceneStagingStatus {
  bool ready = true;
  bool failed = false;
  float progress = 1.0f;
  std::string error{};
};

struct EditorSceneSpec {
  EditorSceneInfo info{};
  std::function<Result<void, std::string>(EditorScenePrepareContext &)>
      prepare{};
  std::function<Result<void, std::string>(EditorSceneActivateContext &)>
      activate{};
  std::function<void(EditorSceneDeactivateContext &)> deactivate{};
  std::function<void(EditorSceneUpdateContext &)> update{};
  // Optional authoritative readiness for scene-local staging work that is not
  // represented by AssetSystem publication requests. When omitted, callbacks
  // are required to finish their small staging work synchronously.
  std::function<EditorSceneStagingStatus()> stagingStatus{};
};

struct ImportedPrefabSceneResources {
  ResourceManager *resources = nullptr;
  AssetSystem *assetSystem = nullptr;
  std::filesystem::path sourcePath{};
  SceneLoadHandle sceneLoad{};
  ScenePrefab prefab{};
  ScenePrefabAssets assets{};
  std::vector<ImportedSceneLight> fallbackLights{};
  bool ready = false;
  bool textureArtifactBakeQueued = false;

  ImportedPrefabSceneResources() = default;
  ~ImportedPrefabSceneResources();

  ImportedPrefabSceneResources(const ImportedPrefabSceneResources &) = delete;
  ImportedPrefabSceneResources &
  operator=(const ImportedPrefabSceneResources &) = delete;
  ImportedPrefabSceneResources(ImportedPrefabSceneResources &&other) noexcept;
  ImportedPrefabSceneResources &
  operator=(ImportedPrefabSceneResources &&other) noexcept;

  void release() noexcept;
};

struct SimpleModelSceneAssets {
  ResourceManager *resources = nullptr;
  std::filesystem::path sourcePath{};
  ModelRef model = kInvalidModelRef;
  MaterialRef material = kInvalidMaterialRef;
  std::vector<ImportedSceneLight> fallbackLights{};
  bool ready = false;
  bool textureArtifactBakeQueued = false;

  SimpleModelSceneAssets() = default;
  ~SimpleModelSceneAssets();

  SimpleModelSceneAssets(const SimpleModelSceneAssets &) = delete;
  SimpleModelSceneAssets &operator=(const SimpleModelSceneAssets &) = delete;
  SimpleModelSceneAssets(SimpleModelSceneAssets &&other) noexcept;
  SimpleModelSceneAssets &operator=(SimpleModelSceneAssets &&other) noexcept;

  void release() noexcept;
};

struct AnimatedPrefabSceneState {
  NodeId rootNode = kInvalidNodeId;
  SceneInstantiationMap instantiationMap;
  SimulationHandle simulation = kInvalidSimulationHandle;

  explicit AnimatedPrefabSceneState(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : instantiationMap(memory) {}

  AnimatedPrefabSceneState(const AnimatedPrefabSceneState &) = delete;
  AnimatedPrefabSceneState &
  operator=(const AnimatedPrefabSceneState &) = delete;
  AnimatedPrefabSceneState(AnimatedPrefabSceneState &&) noexcept = default;
  AnimatedPrefabSceneState &
  operator=(AnimatedPrefabSceneState &&) noexcept = default;
};

struct StreamingSceneState {
  AssetSystem *assets = nullptr;
  std::filesystem::path sourcePath{};
  SceneLoadHandle sceneLoad{};
  ModelRef model = kInvalidModelRef;
  RenderableId renderableId = kInvalidRenderableId;
  glm::mat4 baseModel{1.0f};
  bool configured = false;
  bool baseConfigured = false;
  bool populationComplete = false;
  bool cameraConfigured = false;
  bool lightingConfigured = false;
  bool loadFailed = false;
  std::string loadError{};
  uint32_t populationCursor = 0u;
  float populationProgress = 0.0f;
  SceneGraphCapacityReservation graphCapacityReservation{};
  bool textureArtifactBakeQueued = false;
  double loadStartTimeSeconds = 0.0;
  double lastProgressLogTimeSeconds = 0.0;

  StreamingSceneState() = default;
  ~StreamingSceneState();

  StreamingSceneState(const StreamingSceneState &) = delete;
  StreamingSceneState &operator=(const StreamingSceneState &) = delete;
  StreamingSceneState(StreamingSceneState &&other) noexcept;
  StreamingSceneState &operator=(StreamingSceneState &&other) noexcept;

  void release() noexcept;
};

struct PrefabSceneFactoryDesc {
  EditorSceneInfo info{};
  std::filesystem::path sourcePath{};
  MeshImportOptions importOptions{};
  std::string instanceName{};
  std::string fallbackMaterialDebugName{};
  std::string importedMaterialDebugNamePrefix{};
  glm::mat4 baseModel{1.0f};
  glm::vec3 lodThresholds{8.0f, 16.0f, 32.0f};
  bool requirePrefabInstantiation = false;
  std::function<Result<void, std::string>(EditorRuntime &,
                                          ImportedPrefabSceneResources &)>
      prepareAdditionalAssets{};
  std::function<void(EditorRuntime &)> configureRender{};
  std::function<std::optional<BoundingBox>(
      EditorRuntime &, const ImportedPrefabSceneResources &)>
      computeBounds{};
  std::function<void(EditorRuntime &, const ImportedPrefabSceneResources &,
                     const BoundingBox &)>
      configureCamera{};
  std::function<Result<void, std::string>(EditorRuntime &,
                                          const ImportedPrefabSceneResources &,
                                          const BoundingBox &)>
      populateScene{};
};

struct AnimatedPrefabSceneFactoryDesc {
  PrefabSceneFactoryDesc prefab{};
  std::vector<std::string> preferredClipNames{};
  std::vector<std::string> secondaryPreferredClipNames{};
  // Expected range: [0.0f, 1.0f]. Animated prefab scene creation clamps
  // authored values before creating simulation params.
  float initialBlendWeight = 0.0f;
  AnimationPoseBlendSyncMode blendSyncMode =
      AnimationPoseBlendSyncMode::Independent;
  std::string simulationDebugName{};
};

struct StreamingSceneFactoryDesc {
  EditorSceneInfo info{};
  std::filesystem::path sourcePath{};
  MeshImportOptions importOptions{};
  std::string instanceName{};
  std::string fallbackMaterialDebugName{};
  glm::vec3 lodThresholds{8.0f, 24.0f, 48.0f};
  bool scaleToNiagaraBistro = false;
  // Optional absolute capacities reserved incrementally before custom scene
  // population begins. Use these for large authored/instanced scenes so node
  // and component arrays never grow opportunistically in an editor frame.
  size_t stagingNodeCapacity = 0u;
  size_t stagingRenderableCapacity = 0u;
  std::function<void(EditorRuntime &)> configureRender{};
  std::function<void(EditorRuntime &, StreamingSceneState &)>
      configureLoadedScene{};
  // Called once per editor frame after the imported prefab is resident. Return
  // true when all custom authored staging work is complete.
  std::function<Result<bool, std::string>(EditorRuntime &,
                                          StreamingSceneState &)>
      populateLoadedSceneStep{};
  std::function<void(EditorRuntime &, StreamingSceneState &)> configureCamera{};
};

[[nodiscard]] EditorSceneSpec makeCustomScene(EditorSceneSpec spec);
[[nodiscard]] EditorSceneSpec makePrefabScene(PrefabSceneFactoryDesc desc);
[[nodiscard]] EditorSceneSpec
makeAnimatedPrefabScene(AnimatedPrefabSceneFactoryDesc desc);
[[nodiscard]] EditorSceneSpec makeInstancedModelScene(EditorSceneSpec spec);
[[nodiscard]] EditorSceneSpec
makeStreamingScene(StreamingSceneFactoryDesc desc);

} // namespace nuri
