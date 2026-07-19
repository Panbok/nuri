#pragma once
#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/pmr_scratch.h"
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
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>
namespace nuri {

template <typename Tag> using AssetHandle = Handle<Tag>;

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
  using TextureAssets = MaterialTextureSlots<TextureAssetHandle>;
  TextureAssets textures{};
  std::string debugName{};
  std::string sourceIdentity{};
};

struct EnvironmentAssetRequest {
  static constexpr size_t kTextureCount = 5u;
  std::array<std::optional<TextureRequest>, kTextureCount> textures{};
  AssetPriority priority = AssetPriority::Normal;
  std::array<bool, kTextureCount> optionalTextures{false, false, false, true,
                                                   false};
  std::string debugName{};
};

class NURI_API AssetSystem final {
public:
  AssetSystem(GPUDevice &gpu, ResourceManager &resources,
              AssetSystemConfig config = {});
  ~AssetSystem();
  [[nodiscard]] Result<TextureAssetHandle, std::string>
  requestTexture(const TextureRequest &request,
                 AssetPriority priority = AssetPriority::Normal);
  [[nodiscard]] Result<ModelAssetHandle, std::string>
  requestModel(const ModelRequest &request,
               AssetPriority priority = AssetPriority::Normal);
  [[nodiscard]] MaterialAssetHandle
  requestMaterial(const MaterialAssetRequest &request,
                  AssetPriority priority = AssetPriority::Normal);
  [[nodiscard]] Result<EnvironmentAssetHandle, std::string>
  requestEnvironment(const EnvironmentAssetRequest &request);
  [[nodiscard]] Result<SceneLoadHandle, std::string>
  requestScene(const SceneLoadRequest &request);
  void setInteractiveMode(bool enabled);
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
  [[nodiscard]] Result<AssetPublicationStats, std::string>
  prepareFrame(AssetPublicationContext context = {});
  [[nodiscard]] AssetCpuSchedulerStats cpuStats() const {
    return scheduler_.stats();
  }

private:
  struct TexturePublicationState {
    TextureDesc desc{};
    std::string debugName{};
  };
  template <typename Handle, typename Request, typename Key, typename Prepared,
            typename PreparedGpu, typename Pending, typename Published,
            typename Publication = std::monostate>
  struct GpuAssetNode {
    Handle handle{};
    AssetState state = AssetState::Queued;
    AssetPriority priority = AssetPriority::Normal;
    Request request{};
    Key key{};
    AssetCpuTaskHandle cpuTask{};
    AssetCpuTaskHandle gpuTask{};
    std::optional<Prepared> prepared{};
    std::unique_ptr<PreparedGpu> preparedGpu{};
    std::unique_ptr<Pending> pending{};
    Publication publication{};
    SubmissionHandle upload{};
    Published published{};
    uint32_t subscriberCount = 1u;
    uint64_t cpuPayloadBytes = 0u;
    std::string error{};
  };
  using TextureNode =
      GpuAssetNode<TextureAssetHandle, TextureRequest, TextureKey,
                   PreparedTextureData, PreparedGpuTexture, Texture, TextureRef,
                   TexturePublicationState>;
  using ModelNode =
      GpuAssetNode<ModelAssetHandle, ModelRequest, ModelKey, PreparedModelData,
                   PreparedGpuModelData, Model, ModelRef>;
  struct MaterialAssetKey {
    uint64_t descHash = 0u;
    MaterialTextureSlots<uint64_t> textureHandles{};
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
    std::array<TextureAssetHandle, EnvironmentAssetRequest::kTextureCount>
        textures{};
    EnvironmentHandles published{};
    RenderScene *boundScene = nullptr;
    uint32_t subscriberCount = 1u;
    bool environmentPublished = false;
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
  using SceneDependency = std::variant<AssetCpuTaskHandle, TextureAssetHandle,
                                       ModelAssetHandle, MaterialAssetHandle>;
  struct SceneNode {
    SceneLoadHandle handle{};
    SceneLoadState state = SceneLoadState::Requested;
    SceneLoadRequest request{};
    SceneKey key{};
    AssetCpuTaskHandle manifestTask{};
    std::vector<uint8_t> materialPreparationFinished{};
    std::optional<PreparedSceneManifest> manifest{};
    std::vector<ModelAssetHandle> models{};
    std::vector<MaterialAssetHandle> materials{};
    MaterialAssetHandle fallbackMaterial{};
    SceneInstantiationMap instantiation{};
    RenderScene *boundScene = nullptr;
    ScenePublicationTargetHandle publicationTarget{};
    std::vector<uint8_t> modelFallbackMapped{};
    std::vector<uint8_t> renderableMaterialMapped{};
    std::vector<SceneDependency> dependencies{};
    ScenePrefabStructureCursor structureCursor{};
    uint32_t modelAdmissionCursor = 0u;
    uint32_t materialAdmissionCursor = 0u;
    uint32_t modelMappingCursor = 0u;
    uint32_t materialMappingCursor = 0u;
    uint32_t renderableCursor = 0u;
    uint32_t cancellationCursor = 0u;
    uint32_t subscriberCount = 1u;
    uint32_t requiredFailures = 0u;
    uint32_t optionalFailures = 0u;
    uint64_t cpuPayloadBytes = 0u;
    float progress = 0.0f;
    bool hierarchyPublished = false;
    bool commitPending = false;
    std::string error{};
  };
  struct ScenePublicationTargetNode {
    ScenePublicationTargetHandle handle{};
    RenderScene *scene = nullptr;
    NodeId parent = kInvalidNodeId;
  };
  template <typename Handle, typename Prepared> struct CpuAssetCompletion {
    Handle handle{};
    std::optional<Prepared> prepared{};
    std::string error{};
    bool cancelled = false;
  };
  using TextureCpuCompletion =
      CpuAssetCompletion<TextureAssetHandle, PreparedTextureData>;
  using ModelCpuCompletion =
      CpuAssetCompletion<ModelAssetHandle, PreparedModelData>;
  template <typename Handle, typename Prepared,
            typename Publication = std::monostate>
  struct GpuAssetCompletion {
    Handle handle{};
    std::unique_ptr<Prepared> prepared{};
    Publication publication{};
    std::string error{};
    bool cancelled = false;
  };
  using TextureGpuCompletion =
      GpuAssetCompletion<TextureAssetHandle, PreparedGpuTexture,
                         TexturePublicationState>;
  using ModelGpuCompletion =
      GpuAssetCompletion<ModelAssetHandle, PreparedGpuModelData>;
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
  template <typename Node, typename Handle> class NodePool {
  public:
    [[nodiscard]] Node *find(Handle handle) {
      return slots_.isValid(handle.index, handle.generation)
                 ? &nodes_[handle.index]
                 : nullptr;
    }
    [[nodiscard]] const Node *find(Handle handle) const {
      return slots_.isValid(handle.index, handle.generation)
                 ? &nodes_[handle.index]
                 : nullptr;
    }
    [[nodiscard]] Node &insert(Node node) {
      const SlotReservation slot = slots_.acquire();
      if (slot.appended) {
        nodes_.emplace_back();
      }
      node.handle = Handle{slot.index, slot.generation};
      return nodes_[slot.index] = std::move(node);
    }
    void release(Handle handle) { slots_.release(handle.index); }
    template <typename Predicate> void reclaim(Predicate isTerminal) {
      for (uint32_t index = 0; index < slots_.slotCount(); ++index) {
        Node &node = nodes_[index];
        if (slots_.isLive(index) && node.subscriberCount == 0u &&
            isTerminal(node.state)) {
          node = {};
          slots_.release(index);
        }
      }
    }
    template <typename Fn> void forEachLive(Fn &&fn) {
      for (uint32_t index = 0; index < slots_.slotCount(); ++index) {
        if (slots_.isLive(index)) {
          fn(nodes_[index]);
        }
      }
    }
    [[nodiscard]] auto begin() noexcept { return nodes_.begin(); }
    [[nodiscard]] auto end() noexcept { return nodes_.end(); }
    [[nodiscard]] auto begin() const noexcept { return nodes_.begin(); }
    [[nodiscard]] auto end() const noexcept { return nodes_.end(); }

  private:
    std::deque<Node> nodes_{};
    SlotPool<UnmaskedNonZeroGenerationPolicy> slots_{};
  };
  template <typename Handle> [[nodiscard]] auto *find(Handle handle) {
    if constexpr (std::is_same_v<Handle, TextureAssetHandle>)
      return textures_.find(handle);
    else if constexpr (std::is_same_v<Handle, ModelAssetHandle>)
      return models_.find(handle);
    else if constexpr (std::is_same_v<Handle, MaterialAssetHandle>)
      return materials_.find(handle);
    else if constexpr (std::is_same_v<Handle, EnvironmentAssetHandle>)
      return environments_.find(handle);
    else if constexpr (std::is_same_v<Handle, SceneLoadHandle>)
      return scenes_.find(handle);
    else
      return sceneTargets_.find(handle);
  }
  template <typename Handle>
  [[nodiscard]] const auto *find(Handle handle) const {
    return const_cast<AssetSystem *>(this)->find(handle);
  }
  template <typename Handle>
  void setGpuAssetPriority(Handle handle, AssetPriority priority);
  template <typename Handle> void cancelGpuAsset(Handle handle);
  void reclaimReleasedNodes();
  void pushCompletion(CpuCompletion completion);
  [[nodiscard]] std::vector<CpuCompletion>
  takeCompletions(uint32_t maxCompletions);
  void returnCompletions(std::span<CpuCompletion> completions);
  template <typename Node>
  void finishNode(Node &node, AssetState terminalState, std::string error = {});
  void finishSceneNode(SceneNode &node, SceneLoadState terminalState,
                       std::string error = {});
  [[nodiscard]] bool
  cancelSceneDependencies(SceneNode &node,
                          std::chrono::steady_clock::time_point deadline,
                          uint32_t maxOperations = 64u);
  [[nodiscard]] bool
  prepareSceneNode(SceneNode &node, AssetPublicationStats &stats,
                   uint32_t &patchBudget,
                   std::chrono::steady_clock::time_point deadline,
                   RenderScene *targetScene, NodeId targetParent);
  template <typename Handle>
  [[nodiscard]] SceneAssetCounts
  collectSceneCounts(std::span<const Handle> handles) const {
    SceneAssetCounts counts{.total = static_cast<uint32_t>(handles.size())};
    for (Handle handle : handles) {
      const auto *node = find(handle);
      countSceneAssetState(counts, node ? node->state : AssetState::Queued);
    }
    return counts;
  }
  static void countSceneAssetState(SceneAssetCounts &counts, AssetState state);
  [[nodiscard]] SceneAssetCounts
  collectSceneTextureCounts(const SceneNode &node) const;
  [[nodiscard]] static MaterialAssetKey
  makeMaterialAssetKey(const MaterialAssetRequest &request);
  [[nodiscard]] Result<ModelAssetHandle, std::string>
  requestModelAsset(const ModelRequest &request,
                    std::shared_ptr<AdaptedSceneMesh> source,
                    AssetPriority priority);
  template <typename Node, typename Handle, typename Completion, typename Pool,
            typename Map, typename Request, typename Key, typename Published,
            typename Prepare>
  [[nodiscard]] Result<Handle, std::string>
  requestGpuAsset(Pool &pool, Map &inFlight, Request request, Key key,
                  std::optional<Published> ready, AssetPriority priority,
                  AssetWorkClass workClass, uint64_t estimatedBytes,
                  Prepare prepare);
  [[nodiscard]] static float progressForScene(const SceneNode &node,
                                              const AssetSystem &assets);
  GPUDevice &gpu_;
  ResourceManager &resources_;
  AssetSystemConfig config_{};
  AssetCpuScheduler scheduler_;
  NodePool<TextureNode, TextureAssetHandle> textures_{};
  NodePool<ModelNode, ModelAssetHandle> models_{};
  NodePool<MaterialNode, MaterialAssetHandle> materials_{};
  NodePool<EnvironmentNode, EnvironmentAssetHandle> environments_{};
  NodePool<SceneNode, SceneLoadHandle> scenes_{};
  NodePool<ScenePublicationTargetNode, ScenePublicationTargetHandle>
      sceneTargets_{};
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
  ScratchArena frameScratch_{};
  size_t scenePrepareCursor_ = 0u;
  bool releasedTerminalNodes_ = false;
};

} // namespace nuri
