#pragma once

#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/async/asset_cpu_scheduler.h"
#include "nuri/resources/async/scene_asset_preparation.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nuri {

template <typename Tag> struct AssetHandle {
  uint32_t index = 0u;
  uint32_t generation = 0u;
  constexpr bool operator==(const AssetHandle &) const noexcept = default;
};

struct TextureAssetTag;
struct ModelAssetTag;
struct MaterialAssetTag;
struct EnvironmentAssetTag;
struct SceneLoadTag;
struct ScenePublicationTargetTag;

using TextureAssetHandle = AssetHandle<TextureAssetTag>;
using ModelAssetHandle = AssetHandle<ModelAssetTag>;
using MaterialAssetHandle = AssetHandle<MaterialAssetTag>;
using EnvironmentAssetHandle = AssetHandle<EnvironmentAssetTag>;
using SceneLoadHandle = AssetHandle<SceneLoadTag>;
using ScenePublicationTargetHandle = AssetHandle<ScenePublicationTargetTag>;

template <typename Tag>
[[nodiscard]] constexpr bool
isValidAssetHandle(AssetHandle<Tag> handle) noexcept {
  return handle.generation != 0u;
}

enum class AssetState : uint8_t {
  Queued,
  CpuRunning,
  CpuReady,
  GpuQueued,
  GpuReady,
  GpuSubmitted,
  Resident,
  Published,
  CancelRequested,
  Cancelled,
  Failed,
};

enum class SceneLoadState : uint8_t {
  Requested,
  ManifestReady,
  HierarchyPublished,
  PartiallyResident,
  Complete,
  CompleteWithErrors,
  Failed,
  Cancelling,
  Cancelled,
};

enum class ScenePublicationPolicy : uint8_t {
  Progressive,
  CompleteOnly,
};

enum class SceneFailurePolicy : uint8_t {
  BestEffort,
  FailOnAnyAsset,
};

struct AssetLoadSnapshot {
  AssetState state = AssetState::Failed;
  AssetPriority priority = AssetPriority::Normal;
  float progress = 0.0f;
  uint64_t cpuPayloadBytes = 0u;
  std::string error{};

  [[nodiscard]] bool terminal() const noexcept {
    return state == AssetState::Published || state == AssetState::Cancelled ||
           state == AssetState::Failed;
  }
};

struct SceneLoadRequest {
  std::string path{};
  SceneImportOptions importOptions{};
  AssetPriority priority = AssetPriority::Normal;
  ScenePublicationPolicy publication = ScenePublicationPolicy::Progressive;
  SceneFailurePolicy failurePolicy = SceneFailurePolicy::BestEffort;
  // Optional explicit main-thread publication target. When invalid, the
  // request uses AssetPublicationContext::scene for legacy/tool callers.
  ScenePublicationTargetHandle publicationTarget{};
  std::string debugName{};
};

struct ScenePublicationTargetSnapshot {
  uint32_t requestCount = 0u;
  uint32_t pendingCount = 0u;
  uint32_t failedCount = 0u;
  uint32_t cancelledCount = 0u;
  float progress = 0.0f;

  [[nodiscard]] bool ready() const noexcept {
    return requestCount == 0u ||
           (pendingCount == 0u && failedCount == 0u && cancelledCount == 0u);
  }
};

struct SceneAssetCounts {
  uint32_t total = 0u;
  uint32_t queued = 0u;
  uint32_t cpuReady = 0u;
  uint32_t gpuSubmitted = 0u;
  uint32_t published = 0u;
  uint32_t failed = 0u;
  uint32_t cancelled = 0u;
};

struct SceneLoadSnapshot {
  SceneLoadState state = SceneLoadState::Failed;
  AssetPriority priority = AssetPriority::Normal;
  float progress = 0.0f;
  bool sourceDiscoveryComplete = false;
  bool hierarchyPublished = false;
  bool cancellationPendingGpuRetirement = false;
  SceneAssetCounts models{};
  SceneAssetCounts materials{};
  SceneAssetCounts textures{};
  uint32_t publishedRenderables = 0u;
  uint32_t totalRenderables = 0u;
  uint32_t requiredFailures = 0u;
  uint32_t optionalFailures = 0u;
  uint64_t cpuPayloadBytes = 0u;
  std::string error{};

  [[nodiscard]] bool terminal() const noexcept {
    return state == SceneLoadState::Complete ||
           state == SceneLoadState::CompleteWithErrors ||
           state == SceneLoadState::Failed ||
           state == SceneLoadState::Cancelled;
  }
};

struct AssetPublicationContext {
  RenderScene *scene = nullptr;
  NodeId parent = kInvalidNodeId;
  bool commitScene = true;
};

struct AssetSystemConfig {
  AssetCpuSchedulerConfig cpu{};
  uint32_t maxGpuMaterializationsPerFrame = 16u;
  uint64_t maxGpuUploadBytesPerFrame = 256ull * 1024ull * 1024ull;
  uint32_t maxMaterialPublicationsPerFrame = 256u;
  uint32_t maxScenePatchesPerFrame = 1024u;
  uint32_t maxCpuCompletionsPerFrame = 64u;
  double maxMainThreadMillisecondsPerFrame = 2.0;
};

struct AssetPublicationStats {
  uint32_t cpuCompletions = 0u;
  uint32_t gpuMaterialized = 0u;
  uint32_t published = 0u;
  uint32_t cancelled = 0u;
  uint32_t failed = 0u;
  uint32_t scenePatches = 0u;
  uint32_t sceneCommits = 0u;
  uint64_t uploadBytes = 0u;
  uint32_t deferredCpuCompletions = 0u;
  double mainThreadMilliseconds = 0.0;
  double maxOperationMilliseconds = 0.0;
  bool deadlineExceeded = false;
};

struct MaterialAssetRequest {
  MaterialDesc desc{};
  struct TextureAssets {
    TextureAssetHandle baseColor{};
    TextureAssetHandle metallicRoughness{};
    TextureAssetHandle normal{};
    TextureAssetHandle occlusion{};
    TextureAssetHandle emissive{};
    TextureAssetHandle clearcoat{};
    TextureAssetHandle clearcoatRoughness{};
    TextureAssetHandle clearcoatNormal{};
    TextureAssetHandle specular{};
    TextureAssetHandle specularColor{};
    TextureAssetHandle sheenColor{};
    TextureAssetHandle sheenRoughness{};
    TextureAssetHandle transmission{};
    TextureAssetHandle thickness{};
  } textures{};
  std::string debugName{};
  std::string sourceIdentity{};
};

struct EnvironmentAssetRequest {
  std::optional<TextureRequest> cubemap{};
  std::optional<TextureRequest> irradiance{};
  std::optional<TextureRequest> prefilteredGgx{};
  std::optional<TextureRequest> prefilteredCharlie{};
  std::optional<TextureRequest> brdfLut{};
  AssetPriority priority = AssetPriority::Normal;
  bool cubemapOptional = false;
  bool irradianceOptional = false;
  bool prefilteredGgxOptional = false;
  bool prefilteredCharlieOptional = true;
  bool brdfLutOptional = false;
  std::string debugName{};
};

class NURI_API AssetSystem final {
public:
  AssetSystem(GPUDevice &gpu, ResourceManager &resources,
              AssetSystemConfig config = {});
  ~AssetSystem();

  AssetSystem(const AssetSystem &) = delete;
  AssetSystem &operator=(const AssetSystem &) = delete;
  AssetSystem(AssetSystem &&) = delete;
  AssetSystem &operator=(AssetSystem &&) = delete;

  [[nodiscard]] Result<TextureAssetHandle, std::string>
  requestTexture(const TextureRequest &request,
                 AssetPriority priority = AssetPriority::Normal);
  [[nodiscard]] Result<ModelAssetHandle, std::string>
  requestModel(const ModelRequest &request,
               AssetPriority priority = AssetPriority::Normal);
  [[nodiscard]] Result<MaterialAssetHandle, std::string>
  requestMaterial(const MaterialAssetRequest &request,
                  AssetPriority priority = AssetPriority::Normal);
  [[nodiscard]] Result<EnvironmentAssetHandle, std::string>
  requestEnvironment(const EnvironmentAssetRequest &request);
  [[nodiscard]] Result<SceneLoadHandle, std::string>
  requestScene(const SceneLoadRequest &request);
  void setInteractiveMode(bool enabled);

  // Scene publication targets are registered and resolved only on the
  // render/main thread. A document must keep its target registered until every
  // request that references it is terminal.
  [[nodiscard]] ScenePublicationTargetHandle
  registerScenePublicationTarget(RenderScene &scene,
                                 NodeId parent = kInvalidNodeId);
  [[nodiscard]] bool
  unregisterScenePublicationTarget(ScenePublicationTargetHandle handle);
  [[nodiscard]] ScenePublicationTargetSnapshot
  query(ScenePublicationTargetHandle handle) const;

  [[nodiscard]] AssetLoadSnapshot query(TextureAssetHandle handle) const;
  [[nodiscard]] AssetLoadSnapshot query(ModelAssetHandle handle) const;
  [[nodiscard]] AssetLoadSnapshot query(MaterialAssetHandle handle) const;
  [[nodiscard]] AssetLoadSnapshot query(EnvironmentAssetHandle handle) const;
  [[nodiscard]] SceneLoadSnapshot query(SceneLoadHandle handle) const;
  [[nodiscard]] std::optional<TextureRef>
  tryResolve(TextureAssetHandle handle) const;
  [[nodiscard]] std::optional<ModelRef>
  tryResolve(ModelAssetHandle handle) const;
  [[nodiscard]] std::optional<MaterialRef>
  tryResolve(MaterialAssetHandle handle) const;
  [[nodiscard]] std::optional<EnvironmentHandles>
  tryResolve(EnvironmentAssetHandle handle) const;
  [[nodiscard]] const ScenePrefab *
  tryGetScenePrefab(SceneLoadHandle handle) const;
  [[nodiscard]] std::optional<ScenePrefabAssets>
  tryGetSceneAssets(SceneLoadHandle handle) const;
  [[nodiscard]] std::optional<SceneInstantiationMap>
  tryGetSceneInstantiation(SceneLoadHandle handle) const;

  void setPriority(TextureAssetHandle handle, AssetPriority priority);
  void setPriority(ModelAssetHandle handle, AssetPriority priority);
  void setPriority(MaterialAssetHandle handle, AssetPriority priority);
  void setPriority(EnvironmentAssetHandle handle, AssetPriority priority);
  void setPriority(SceneLoadHandle handle, AssetPriority priority);
  void cancel(TextureAssetHandle handle);
  void cancel(ModelAssetHandle handle);
  void cancel(MaterialAssetHandle handle);
  void cancel(EnvironmentAssetHandle handle);
  void cancel(SceneLoadHandle handle);

  // Single-writer render/main-thread boundary. Drains immutable worker
  // completions, records a bounded GPU upload batch, polls availability, and
  // publishes only resident resources.
  [[nodiscard]] Result<AssetPublicationStats, std::string>
  prepareFrame(AssetPublicationContext context = {});

  [[nodiscard]] AssetCpuSchedulerStats cpuStats() const {
    return scheduler_.stats();
  }

private:
  struct TextureNode {
    TextureAssetHandle handle{};
    AssetState state = AssetState::Queued;
    AssetPriority priority = AssetPriority::Normal;
    TextureRequest request{};
    TextureKey key{};
    AssetCpuTaskHandle cpuTask{};
    AssetCpuTaskHandle gpuTask{};
    std::optional<PreparedTextureData> prepared{};
    std::unique_ptr<PreparedGpuTexture> preparedGpuTexture{};
    TextureDesc preparedGpuDesc{};
    std::string preparedGpuDebugName{};
    std::unique_ptr<Texture> pendingTexture{};
    SubmissionHandle upload{};
    TextureRef published = kInvalidTextureRef;
    uint32_t subscriberCount = 1u;
    uint64_t cpuPayloadBytes = 0u;
    std::string error{};
  };

  struct ModelNode {
    ModelAssetHandle handle{};
    AssetState state = AssetState::Queued;
    AssetPriority priority = AssetPriority::Normal;
    ModelRequest request{};
    ModelKey key{};
    AssetCpuTaskHandle cpuTask{};
    AssetCpuTaskHandle gpuTask{};
    std::optional<PreparedModelData> prepared{};
    std::unique_ptr<PreparedGpuModelData> preparedGpuModel{};
    std::unique_ptr<Model> pendingModel{};
    SubmissionHandle upload{};
    ModelRef published = kInvalidModelRef;
    uint32_t subscriberCount = 1u;
    uint64_t cpuPayloadBytes = 0u;
    std::string error{};
  };

  struct MaterialAssetKey {
    uint64_t descHash = 0u;
    std::array<uint64_t, 14u> textureHandles{};
    std::string sourceIdentity{};

    bool operator==(const MaterialAssetKey &) const noexcept = default;
  };

  struct MaterialAssetKeyHash {
    size_t operator()(const MaterialAssetKey &key) const noexcept;
  };

  struct MaterialNode {
    MaterialAssetHandle handle{};
    AssetState state = AssetState::CpuReady;
    AssetPriority priority = AssetPriority::Normal;
    MaterialAssetRequest request{};
    MaterialAssetKey key{};
    MaterialRef published = kInvalidMaterialRef;
    uint32_t subscriberCount = 1u;
    std::string error{};
  };

  struct EnvironmentNode {
    EnvironmentAssetHandle handle{};
    AssetState state = AssetState::Queued;
    EnvironmentAssetRequest request{};
    std::array<TextureAssetHandle, 5u> textures{};
    EnvironmentHandles published{};
    RenderScene *boundScene = nullptr;
    uint32_t subscriberCount = 1u;
    bool environmentPublished = false;
    bool dependenciesCancelled = false;
    std::string error{};
  };

  struct SceneKey {
    std::string canonicalPath{};
    uint64_t importOptionsHash = 0u;
    ScenePublicationPolicy publication = ScenePublicationPolicy::Progressive;
    SceneFailurePolicy failurePolicy = SceneFailurePolicy::BestEffort;
    ScenePublicationTargetHandle publicationTarget{};

    bool operator==(const SceneKey &) const noexcept = default;
  };

  struct SceneKeyHash {
    size_t operator()(const SceneKey &key) const noexcept;
  };

  struct SceneNode {
    SceneLoadHandle handle{};
    SceneLoadState state = SceneLoadState::Requested;
    SceneLoadRequest request{};
    SceneKey key{};
    AssetCpuTaskHandle manifestTask{};
    std::vector<AssetCpuTaskHandle> materialTasks{};
    std::vector<uint8_t> materialPreparationFinished{};
    std::optional<PreparedSceneManifest> manifest{};
    std::vector<ModelAssetHandle> models{};
    std::vector<MaterialAssetHandle> materials{};
    std::vector<std::vector<TextureAssetHandle>> textureSubscriptions{};
    MaterialAssetHandle fallbackMaterial{};
    SceneInstantiationMap instantiation{};
    NodeId root = kInvalidNodeId;
    RenderScene *boundScene = nullptr;
    ScenePublicationTargetHandle publicationTarget{};
    std::vector<uint8_t> modelFallbackMapped{};
    std::vector<uint8_t> renderableMaterialMapped{};
    ScenePrefabStructureCursor structureCursor{};
    uint32_t modelAdmissionCursor = 0u;
    uint32_t materialAdmissionCursor = 0u;
    uint32_t modelMappingCursor = 0u;
    uint32_t materialMappingCursor = 0u;
    uint32_t renderableCursor = 0u;
    uint32_t cancellationTaskCursor = 0u;
    uint32_t cancellationModelCursor = 0u;
    uint32_t cancellationMaterialCursor = 0u;
    uint32_t cancellationTextureSubscriptionCursor = 0u;
    uint32_t cancellationTextureCursor = 0u;
    uint32_t subscriberCount = 1u;
    uint32_t requiredFailures = 0u;
    uint32_t optionalFailures = 0u;
    uint64_t cpuPayloadBytes = 0u;
    float progress = 0.0f;
    bool hierarchyPublished = false;
    bool fallbackRequested = false;
    bool manifestAdmissionComplete = false;
    bool commitPending = false;
    bool dependenciesCancelled = false;
    bool manifestTaskCancelled = false;
    bool fallbackMaterialCancelled = false;
    std::string error{};
  };

  struct ScenePublicationTargetNode {
    ScenePublicationTargetHandle handle{};
    RenderScene *scene = nullptr;
    NodeId parent = kInvalidNodeId;
  };

  struct TextureCpuCompletion {
    TextureAssetHandle handle{};
    std::optional<PreparedTextureData> prepared{};
    std::string error{};
    bool cancelled = false;
  };

  struct ModelCpuCompletion {
    ModelAssetHandle handle{};
    std::optional<PreparedModelData> prepared{};
    std::string error{};
    bool cancelled = false;
  };

  struct TextureGpuCompletion {
    TextureAssetHandle handle{};
    std::unique_ptr<PreparedGpuTexture> prepared{};
    TextureDesc desc{};
    std::string debugName{};
    std::string error{};
    bool cancelled = false;
  };

  struct ModelGpuCompletion {
    ModelAssetHandle handle{};
    std::unique_ptr<PreparedGpuModelData> prepared{};
    std::string error{};
    bool cancelled = false;
  };

  struct SceneManifestCompletion {
    SceneLoadHandle handle{};
    std::optional<PreparedSceneManifest> manifest{};
    std::string error{};
    bool cancelled = false;
  };

  struct SceneMaterialCompletion {
    SceneLoadHandle scene{};
    uint32_t materialIndex = 0u;
    std::optional<PreparedImportedMaterial> prepared{};
    std::string error{};
    bool cancelled = false;
  };

  using CpuCompletion =
      std::variant<TextureCpuCompletion, ModelCpuCompletion,
                   TextureGpuCompletion, ModelGpuCompletion,
                   SceneManifestCompletion, SceneMaterialCompletion>;

  [[nodiscard]] TextureNode *find(TextureAssetHandle handle);
  [[nodiscard]] const TextureNode *find(TextureAssetHandle handle) const;
  [[nodiscard]] ModelNode *find(ModelAssetHandle handle);
  [[nodiscard]] const ModelNode *find(ModelAssetHandle handle) const;
  [[nodiscard]] MaterialNode *find(MaterialAssetHandle handle);
  [[nodiscard]] const MaterialNode *find(MaterialAssetHandle handle) const;
  [[nodiscard]] EnvironmentNode *find(EnvironmentAssetHandle handle);
  [[nodiscard]] const EnvironmentNode *
  find(EnvironmentAssetHandle handle) const;
  [[nodiscard]] SceneNode *find(SceneLoadHandle handle);
  [[nodiscard]] const SceneNode *find(SceneLoadHandle handle) const;
  [[nodiscard]] ScenePublicationTargetNode *
  find(ScenePublicationTargetHandle handle);
  [[nodiscard]] const ScenePublicationTargetNode *
  find(ScenePublicationTargetHandle handle) const;

  void reclaimReleasedNodes();
  void pushCompletion(CpuCompletion completion);
  [[nodiscard]] std::vector<CpuCompletion>
  takeCompletions(uint32_t maxCompletions);
  void returnCompletions(std::span<CpuCompletion> completions);
  void
  discardPreparedTextureAsync(std::unique_ptr<PreparedGpuTexture> prepared);
  void
  discardPreparedModelAsync(std::unique_ptr<PreparedGpuModelData> prepared);
  void finishTextureNode(TextureNode &node, AssetState terminalState,
                         std::string error = {});
  void finishModelNode(ModelNode &node, AssetState terminalState,
                       std::string error = {});
  void finishMaterialNode(MaterialNode &node, AssetState terminalState,
                          std::string error = {});
  void finishSceneNode(SceneNode &node, SceneLoadState terminalState,
                       std::string error = {});
  [[nodiscard]] bool
  cancelSceneDependencies(SceneNode &node,
                          std::chrono::steady_clock::time_point deadline,
                          uint32_t maxOperations = 64u);
  [[nodiscard]] Result<bool, std::string>
  prepareSceneNode(SceneNode &node, AssetPublicationContext &context,
                   AssetPublicationStats &stats, uint32_t &patchBudget,
                   std::chrono::steady_clock::time_point deadline);
  [[nodiscard]] static SceneAssetCounts
  collectSceneCounts(const AssetSystem &assets,
                     std::span<const ModelAssetHandle> handles);
  [[nodiscard]] static SceneAssetCounts
  collectSceneCounts(const AssetSystem &assets,
                     std::span<const MaterialAssetHandle> handles);
  [[nodiscard]] static SceneAssetCounts
  collectSceneTextureCounts(const AssetSystem &assets, const SceneNode &node);
  [[nodiscard]] static MaterialAssetKey
  makeMaterialAssetKey(const MaterialAssetRequest &request);
  [[nodiscard]] Result<ModelAssetHandle, std::string>
  requestAdaptedModel(const ModelRequest &request, AdaptedSceneMesh source,
                      AssetPriority priority);
  [[nodiscard]] static float progressForState(AssetState state) noexcept;
  [[nodiscard]] static float progressForScene(const SceneNode &node,
                                              const AssetSystem &assets);

  GPUDevice &gpu_;
  ResourceManager &resources_;
  AssetSystemConfig config_{};
  AssetCpuScheduler scheduler_;

  // Slot-indexed nodes must remain at stable addresses. GPU submission
  // tracking and the scene-inspection API retain pointers across unrelated
  // request insertion, so vector growth is not a valid storage contract here.
  std::deque<TextureNode> textureNodes_{};
  std::deque<ModelNode> modelNodes_{};
  std::deque<MaterialNode> materialNodes_{};
  std::deque<EnvironmentNode> environmentNodes_{};
  std::deque<SceneNode> sceneNodes_{};
  std::deque<ScenePublicationTargetNode> scenePublicationTargets_{};
  SlotPool<UnmaskedNonZeroGenerationPolicy> textureSlots_{};
  SlotPool<UnmaskedNonZeroGenerationPolicy> modelSlots_{};
  SlotPool<UnmaskedNonZeroGenerationPolicy> materialSlots_{};
  SlotPool<UnmaskedNonZeroGenerationPolicy> environmentSlots_{};
  SlotPool<UnmaskedNonZeroGenerationPolicy> sceneSlots_{};
  SlotPool<UnmaskedNonZeroGenerationPolicy> scenePublicationTargetSlots_{};
  std::unordered_map<TextureKey, TextureAssetHandle, TextureKeyHash>
      textureInFlight_{};
  std::unordered_map<ModelKey, ModelAssetHandle, ModelKeyHash> modelInFlight_{};
  std::unordered_map<MaterialAssetKey, MaterialAssetHandle,
                     MaterialAssetKeyHash>
      materialInFlight_{};
  std::unordered_map<SceneKey, SceneLoadHandle, SceneKeyHash> sceneInFlight_{};

  mutable std::mutex completionMutex_{};
  std::vector<CpuCompletion> completions_{};
  mutable std::recursive_mutex stateMutex_{};
  size_t scenePrepareCursor_ = 0u;
  bool releasedTerminalNodes_ = false;
};

} // namespace nuri
