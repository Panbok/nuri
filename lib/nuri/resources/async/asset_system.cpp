#include "nuri/resources/async/asset_system.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/detail/scene_asset_build_backend.h"
#include "nuri/resources/storage/texture/dds_texture_pack.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"
#include "nuri/scene/render_scene.h"
#include <algorithm>
#include <filesystem>
#include <limits>
namespace nuri {
namespace {
[[nodiscard]] uint64_t estimateSourceBytes(std::string_view path) noexcept {
  std::error_code error;
  const uintmax_t bytes = std::filesystem::file_size(
      std::filesystem::path(std::string(path)), error);
  if (error) {
    return 4ull * 1024ull * 1024ull;
  }
  return std::min<uint64_t>(static_cast<uint64_t>(std::min<uintmax_t>(
                                bytes, std::numeric_limits<uint64_t>::max())),
                            1ull * 1024ull * 1024ull * 1024ull);
}
[[nodiscard]] uint64_t
estimateAdaptedMeshBytes(const ScenePrefabAdaptedMesh &source) noexcept {
  uint64_t bytes =
      static_cast<uint64_t>(source.mesh.vertices.size()) * sizeof(Vertex) +
      static_cast<uint64_t>(source.mesh.indices.size()) * sizeof(uint32_t) +
      static_cast<uint64_t>(source.mesh.skinInfluences.size()) *
          sizeof(VertexSkinInfluence);
  for (const MorphTarget &morph : source.mesh.morphTargets) {
    bytes += static_cast<uint64_t>(morph.positionDeltas.size() +
                                   morph.normalDeltas.size() +
                                   morph.tangentDeltas.size()) *
             sizeof(glm::vec3);
  }
  return std::max<uint64_t>(bytes, 1u);
}
[[nodiscard]] Result<PreparedTextureData, std::string>
prepareTextureRequest(const TextureRequest &request) {
  switch (request.kind) {
  case TextureRequestKind::Texture2D:
    if (request.ddsPack != nullptr) {
      auto bytes = request.ddsPack->readOwned(request.path);
      if (bytes.hasError()) {
        return Result<PreparedTextureData, std::string>::makeError(
            bytes.error());
      }
      return Texture::prepareDdsTexture(bytes.value(), request.path,
                                        request.debugName);
    }
    return Texture::prepareTexture(request.path, request.loadOptions,
                                   request.debugName);
  case TextureRequestKind::Ktx2Texture2D:
    return Texture::prepareTextureKtx2(request.path, request.debugName);
  case TextureRequestKind::Ktx2Cubemap:
    return Texture::prepareCubemapKtx2(request.path, request.debugName);
  case TextureRequestKind::EquirectHdrCubemap:
    return Texture::prepareCubemapFromEquirectangularHDR(request.path,
                                                         request.debugName);
  }
  return Result<PreparedTextureData, std::string>::makeError(
      "AssetSystem: unknown texture request kind");
}
[[nodiscard]] Result<ModelUploadPlan, std::string>
prepareModelRequest(const ModelRequest &request,
                    ScenePrefabAdaptedMesh *adapted) {
  if (adapted != nullptr) {
    auto cooked =
        detail::cookAdaptedSceneMesh(std::move(*adapted), request.importOptions,
                                     std::pmr::get_default_resource());
    return cooked.hasError()
               ? Result<ModelUploadPlan, std::string>::makeError(cooked.error())
               : Model::prepare(std::move(cooked.value()));
  }
  if (request.sceneMeshIndex == std::numeric_limits<uint32_t>::max()) {
    return Model::prepareFromFile(request.path, request.importOptions);
  }
  return Model::prepareSceneMeshFromFile(request.path, request.sceneMeshIndex,
                                         request.importOptions);
}
[[nodiscard]] bool canSelectMaterialization(uint64_t bytes,
                                            uint32_t selectedCount,
                                            uint64_t selectedBytes,
                                            const AssetSystemConfig &config) {
  if (selectedCount >= config.maxGpuMaterializationsPerFrame) {
    return false;
  }
  if (selectedCount == 0u) {
    return true;
  }
  return selectedBytes < config.maxGpuUploadBytesPerFrame &&
         bytes <= config.maxGpuUploadBytesPerFrame - selectedBytes;
}
void hashCombine(size_t &seed, uint64_t value) noexcept {
  seed ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ull + (seed << 6u) +
          (seed >> 2u);
}
[[nodiscard]] bool supportsBackgroundPreparation(GPUDevice &gpu,
                                                 const PreparedTextureData &) {
  return gpu.supportsBackgroundTexturePreparation();
}
[[nodiscard]] bool supportsBackgroundPreparation(GPUDevice &gpu,
                                                 const ModelUploadPlan &) {
  return gpu.supportsBackgroundBufferPreparation() &&
         gpu.supportsBackgroundBufferBatchPreparation() &&
         gpu.supportsBackgroundGeometryPreparation();
}
[[nodiscard]] std::pair<std::string, std::string>
materializationNames(const TextureRequest &request,
                     const PreparedTextureData &prepared) {
  std::string gpuName = prepared.debugName;
  return {gpuName, gpuName.empty() ? request.debugName : gpuName};
}
[[nodiscard]] std::pair<std::string, std::string>
materializationNames(const ModelRequest &request, const ModelUploadPlan &) {
  return {request.debugName,
          request.debugName.empty() ? request.path : request.debugName};
}
auto prepareGpuAsset(GPUDevice &gpu, PreparedTextureData &prepared,
                     std::string_view debugName) {
  return gpu.prepareTexture(prepared.descriptor(), debugName);
}
auto prepareGpuAsset(GPUDevice &gpu, ModelUploadPlan &prepared,
                     std::string_view debugName) {
  return Model::prepareGpu(gpu, std::move(prepared), debugName);
}
auto createGpuAsset(GPUDevice &gpu, PreparedTextureData prepared,
                    std::string_view) {
  return Texture::createPrepared(gpu, std::move(prepared));
}
auto createGpuAsset(GPUDevice &gpu, ModelUploadPlan prepared,
                    std::string_view debugName) {
  return Model::createPrepared(gpu, std::move(prepared), debugName);
}
auto adoptGpuAsset(ResourceManager &resources, const TextureRequest &request,
                   const TextureKey &key, std::unique_ptr<Texture> pending) {
  return resources.adoptPreparedTexture(
      ResolvedTextureRequest{.request = request, .key = key},
      std::move(pending));
}
auto adoptGpuAsset(ResourceManager &resources, const ModelRequest &request,
                   const ModelKey &key, std::unique_ptr<Model> pending) {
  return resources.adoptPreparedModel(
      ResolvedModelRequest{.request = request, .key = key}, std::move(pending));
}
constexpr std::array<TextureRef EnvironmentHandles::*,
                     EnvironmentAssetRequest::kTextureCount>
    kEnvironmentTextureFields{
        &EnvironmentHandles::cubemap, &EnvironmentHandles::irradiance,
        &EnvironmentHandles::prefilteredGgx,
        &EnvironmentHandles::prefilteredCharlie, &EnvironmentHandles::brdfLut};
[[nodiscard]] bool isAssetTerminalState(AssetState state) noexcept {
  return state == AssetState::Published || state == AssetState::Cancelled ||
         state == AssetState::Failed;
}
[[nodiscard]] bool isSceneTerminalState(SceneLoadState state) noexcept {
  return state == SceneLoadState::Complete ||
         state == SceneLoadState::CompleteWithErrors ||
         state == SceneLoadState::Failed || state == SceneLoadState::Cancelled;
}
constexpr std::array kAssetStateProgress{0.0f, 0.15f, 0.55f, 0.55f, 0.7f, 0.75f,
                                         0.9f, 1.0f,  0.0f,  1.0f,  1.0f};
[[nodiscard]] constexpr float assetStateProgress(AssetState state) noexcept {
  return kAssetStateProgress[static_cast<size_t>(state)];
}
template <typename Node>
[[nodiscard]] AssetLoadSnapshot assetSnapshot(const Node *node,
                                              std::string_view error) {
  if (node == nullptr) {
    return {.state = AssetState::Failed,
            .progress = 1.0f,
            .error = std::string(error)};
  }
  AssetLoadSnapshot snapshot{
      .state = node->state,
      .priority = node->priority,
      .progress = assetStateProgress(node->state),
      .error = node->error,
  };
  if constexpr (requires { node->cpuPayloadBytes; }) {
    snapshot.cpuPayloadBytes = node->cpuPayloadBytes;
  }
  return snapshot;
}
template <typename Node> [[nodiscard]] auto resolvedAsset(const Node *node) {
  using Ref = std::remove_cvref_t<decltype(node->published)>;
  return node != nullptr && node->state == AssetState::Published
             ? std::optional<Ref>(node->published)
             : std::nullopt;
}
template <typename Completion, typename Work, typename Emit>
[[nodiscard]] AssetCpuJob
makeAssetCpuJob(AssetPriority priority, AssetWorkClass workClass,
                uint64_t estimatedBytes, std::string debugName, Completion base,
                Work work, Emit emit) {
  return AssetCpuJob{
      .priority = priority,
      .workClass = workClass,
      .estimatedBytes = estimatedBytes,
      .debugName = std::move(debugName),
      .execute =
          [base, work = std::move(work),
           emit](std::stop_token stopToken) mutable {
            Completion completion = base;
            if (stopToken.stop_requested()) {
              completion.cancelled = true;
            } else {
              auto result = work();
              completion.cancelled = stopToken.stop_requested();
              if (!completion.cancelled) {
                if (result.hasError()) {
                  completion.error = result.error();
                } else if constexpr (requires {
                                       completion.manifest =
                                           std::move(result.value());
                                     }) {
                  completion.manifest = std::move(result.value());
                } else {
                  completion.prepared = std::move(result.value());
                }
              }
            }
            emit(std::move(completion));
          },
      .onCancelled =
          [base, emit]() mutable {
            base.cancelled = true;
            emit(std::move(base));
          },
  };
}
template <typename Node, typename Map>
void finishAssetNode(Node &node, AssetState state, std::string error, Map &map,
                     bool &releasedTerminalNodes) {
  node.state = state;
  if (!error.empty() || node.error.empty())
    node.error = std::move(error);
  if constexpr (requires { node.prepared; }) {
    if (state != AssetState::Published)
      node.prepared.reset();
  }
  if constexpr (requires { node.preparedGpu; }) {
    if (state != AssetState::Published)
      node.preparedGpu.reset();
  }
  if constexpr (requires { node.publication.desc; }) {
    node.publication = {};
  }
  map.erase(node.key);
  if (state == AssetState::Cancelled) {
    node.request = {};
    node.key = {};
    if constexpr (requires { node.cpuPayloadBytes; })
      node.cpuPayloadBytes = 0u;
  }
  releasedTerminalNodes |=
      node.subscriberCount == 0u && isAssetTerminalState(state);
}
template <typename T>
void deferPreparedDestroy(AssetCpuScheduler &scheduler,
                          std::unique_ptr<T> prepared) {
  if (!prepared)
    return;
  auto lifetime = std::shared_ptr<T>(std::move(prepared));
  (void)scheduler.enqueue(AssetCpuJob{
      .priority = AssetPriority::Background,
      .workClass = AssetWorkClass::GpuMaterialization,
      .debugName = "discard prepared GPU asset",
      .execute = [lifetime](std::stop_token) {},
  });
}
template <typename Node, typename Released>
bool claimCancellation(Node *node, bool &releasedTerminalNodes,
                       Released released) {
  if (node == nullptr || node->subscriberCount == 0u)
    return false;
  if (--node->subscriberCount != 0u)
    return false;
  if (released(node->state)) {
    releasedTerminalNodes = true;
    return false;
  }
  return true;
}
struct OperationTimer {
  explicit OperationTimer(double &maximum)
      : maximum(maximum), start(std::chrono::steady_clock::now()) {}
  ~OperationTimer() {
    maximum = std::max(maximum, std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count());
  }
  double &maximum;
  std::chrono::steady_clock::time_point start;
};
} // namespace

template <typename Node>
void AssetSystem::finishNode(Node &node, AssetState state, std::string error) {
  if constexpr (std::is_same_v<Node, TextureNode>)
    finishAssetNode(node, state, std::move(error), textureInFlight_,
                    releasedTerminalNodes_);
  else if constexpr (std::is_same_v<Node, ModelNode>)
    finishAssetNode(node, state, std::move(error), modelInFlight_,
                    releasedTerminalNodes_);
  else
    finishAssetNode(node, state, std::move(error), materialInFlight_,
                    releasedTerminalNodes_);
}

AssetSystem::AssetSystem(GPUDevice &gpu, ResourceManager &resources,
                         AssetSystemConfig config)
    : gpu_(gpu), resources_(resources), config_(config),
      scheduler_(config.cpu) {
  config_.maxGpuMaterializationsPerFrame =
      std::max(config_.maxGpuMaterializationsPerFrame, 1u);
  config_.maxGpuUploadBytesPerFrame =
      std::max(config_.maxGpuUploadBytesPerFrame, 1ull);
  config_.maxMaterialPublicationsPerFrame =
      std::max(config_.maxMaterialPublicationsPerFrame, 1u);
  config_.maxScenePatchesPerFrame =
      std::max(config_.maxScenePatchesPerFrame, 1u);
  config_.maxCpuCompletionsPerFrame =
      std::max(config_.maxCpuCompletionsPerFrame, 1u);
  config_.maxMainThreadMillisecondsPerFrame =
      std::max(config_.maxMainThreadMillisecondsPerFrame, 0.1);
}

AssetSystem::~AssetSystem() {
  scheduler_.requestStop();
  scheduler_.waitIdle();
  const auto release = [this](auto &pool) {
    for (auto &node : pool) {
      if constexpr (requires { node.pending; })
        node.pending.reset();
      if (isValid(node.published) && resources_.owns(node.published)) {
        resources_.release(node.published);
      }
    }
  };
  release(textures_);
  release(models_);
  release(materials_);
}

void AssetSystem::setInteractiveMode(bool enabled) {
  scheduler_.setInteractiveMode(enabled);
}

AssetSystem::MaterialAssetKey
AssetSystem::makeMaterialAssetKey(const MaterialAssetRequest &request) {
  MaterialAssetKey key{};
  key.descHash = hashMaterialDesc(request.desc);
  for (size_t index = 0; index < request.textures.size(); ++index)
    key.textureHandles[index] = handleKey(request.textures[index]);
  key.sourceIdentity = request.sourceIdentity;
  return key;
}

Result<TextureAssetHandle, std::string>
AssetSystem::requestTexture(const TextureRequest &request,
                            AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  reclaimReleasedNodes();
  if (request.path.empty()) {
    return Result<TextureAssetHandle, std::string>::makeError(
        "AssetSystem::requestTexture: path is empty");
  }
  ResolvedTextureRequest resolved = resources_.resolveTextureRequest(request);
  const auto ready = resources_.tryAcquireTexture(resolved);
  return requestGpuAsset<TextureNode, TextureAssetHandle, TextureCpuCompletion>(
      textures_, textureInFlight_, std::move(resolved.request),
      std::move(resolved.key), ready, priority, AssetWorkClass::Decode,
      estimateSourceBytes(request.path), prepareTextureRequest);
}

Result<ModelAssetHandle, std::string>
AssetSystem::requestModel(const ModelRequest &request, AssetPriority priority) {
  return requestModelAsset(request, nullptr, priority);
}

Result<ModelAssetHandle, std::string>
AssetSystem::requestModelAsset(const ModelRequest &request,
                               std::shared_ptr<ScenePrefabAdaptedMesh> source,
                               AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  reclaimReleasedNodes();
  if (request.path.empty()) {
    return Result<ModelAssetHandle, std::string>::makeError(
        "AssetSystem::requestModel: path is empty");
  }
  ResolvedModelRequest resolved = resources_.resolveModelRequest(request);
  const uint64_t estimatedBytes =
      source != nullptr ? estimateAdaptedMeshBytes(*source)
                        : estimateSourceBytes(resolved.key.canonicalPath);
  const auto ready = resources_.tryAcquireModel(resolved);
  return requestGpuAsset<ModelNode, ModelAssetHandle, ModelCpuCompletion>(
      models_, modelInFlight_, std::move(resolved.request),
      std::move(resolved.key), ready, priority, AssetWorkClass::Cook,
      estimatedBytes,
      [source = std::move(source)](const ModelRequest &workerRequest) {
        return prepareModelRequest(workerRequest, source.get());
      });
}

template <typename Node, typename Handle, typename Completion, typename Pool,
          typename Map, typename Request, typename Key, typename Published,
          typename Prepare>
Result<Handle, std::string>
AssetSystem::requestGpuAsset(Pool &pool, Map &inFlight, Request request,
                             Key key, std::optional<Published> ready,
                             AssetPriority priority, AssetWorkClass workClass,
                             uint64_t estimatedBytes, Prepare prepare) {
  if (ready) {
    Node newNode{};
    newNode.state = AssetState::Published;
    newNode.priority = priority;
    newNode.request = std::move(request);
    newNode.key = std::move(key);
    newNode.published = *ready;
    Node &node = pool.insert(std::move(newNode));
    return Result<Handle, std::string>::makeResult(node.handle);
  }
  if (auto it = inFlight.find(key); it != inFlight.end()) {
    Node &node = *pool.find(it->second);
    ++node.subscriberCount;
    if (priority < node.priority)
      setPriority(node.handle, priority);
    return Result<Handle, std::string>::makeResult(node.handle);
  }
  Node newNode{};
  newNode.priority = priority;
  newNode.request = std::move(request);
  newNode.key = std::move(key);
  Node &node = pool.insert(std::move(newNode));
  inFlight.emplace(node.key, node.handle);
  const Handle handle = node.handle;
  const Request workerRequest = node.request;
  auto task = scheduler_.enqueue(makeAssetCpuJob<Completion>(
      priority, workClass, estimatedBytes,
      workerRequest.debugName.empty() ? workerRequest.path
                                      : workerRequest.debugName,
      Completion{.handle = handle},
      [workerRequest, prepare = std::move(prepare)]() mutable {
        return prepare(workerRequest);
      },
      [this](Completion completion) {
        pushCompletion(std::move(completion));
      }));
  if (task.hasError()) {
    node.subscriberCount = 0u;
    finishNode(node, AssetState::Failed, task.error());
    return Result<Handle, std::string>::makeError(task.error());
  }
  node.cpuTask = task.value();
  node.state = AssetState::CpuRunning;
  return Result<Handle, std::string>::makeResult(handle);
}

MaterialAssetHandle
AssetSystem::requestMaterial(const MaterialAssetRequest &request,
                             AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  reclaimReleasedNodes();
  const MaterialAssetKey key = makeMaterialAssetKey(request);
  if (auto it = materialInFlight_.find(key); it != materialInFlight_.end()) {
    MaterialNode &node = *find(it->second);
    ++node.subscriberCount;
    if (priority < node.priority)
      setPriority(node.handle, priority);
    return node.handle;
  }
  MaterialNode newNode{};
  newNode.priority = priority;
  newNode.request = request;
  newNode.key = key;
  MaterialNode &node = materials_.insert(std::move(newNode));
  materialInFlight_.emplace(node.key, node.handle);
  for (TextureAssetHandle texture : node.request.textures)
    setPriority(texture, priority);
  return node.handle;
}

Result<EnvironmentAssetHandle, std::string>
AssetSystem::requestEnvironment(const EnvironmentAssetRequest &request) {
  std::lock_guard stateLock(stateMutex_);
  reclaimReleasedNodes();
  EnvironmentNode newNode{};
  newNode.state = AssetState::CpuReady;
  newNode.priority = request.priority;
  newNode.request = request;
  EnvironmentNode &node = environments_.insert(std::move(newNode));
  std::string requestError;
  for (size_t index = 0; index < node.request.textures.size(); ++index) {
    const auto &texture = node.request.textures[index];
    if (!texture)
      continue;
    auto requested = requestTexture(*texture, node.priority);
    if (requested.hasError()) {
      if (node.request.optionalTextures[index])
        continue;
      requestError = requested.error();
      break;
    }
    node.textures[index] = requested.value();
  }
  if (!requestError.empty()) {
    for (const TextureAssetHandle texture : node.textures)
      cancel(texture);
    node.state = AssetState::Failed;
    node.subscriberCount = 0u;
    releasedTerminalNodes_ = true;
    node.error = requestError;
    return Result<EnvironmentAssetHandle, std::string>::makeError(requestError);
  }
  return Result<EnvironmentAssetHandle, std::string>::makeResult(node.handle);
}

Result<SceneLoadHandle, std::string>
AssetSystem::requestScene(const SceneLoadRequest &request) {
  std::lock_guard stateLock(stateMutex_);
  reclaimReleasedNodes();
  if (request.path.empty()) {
    return Result<SceneLoadHandle, std::string>::makeError(
        "AssetSystem::requestScene: path is empty");
  }
  SceneLoadRequest resolved = request;
  resolved.path = canonicalizeResourcePath(request.path);
  if (isValid(resolved.publicationTarget) &&
      find(resolved.publicationTarget) == nullptr) {
    return Result<SceneLoadHandle, std::string>::makeError(
        "AssetSystem::requestScene: publication target is invalid or stale");
  }
  const SceneKey key{
      .canonicalPath = resolved.path,
      .importOptionsHash =
          hashModelImportOptions(resolved.importOptions.assetBuildOptions),
      .publication = resolved.publication,
      .failurePolicy = resolved.failurePolicy,
      .publicationTarget = resolved.publicationTarget,
  };
  if (auto it = sceneInFlight_.find(key); it != sceneInFlight_.end()) {
    SceneNode &node = *find(it->second);
    ++node.subscriberCount;
    if (resolved.priority < node.priority)
      setPriority(node.handle, resolved.priority);
    return Result<SceneLoadHandle, std::string>::makeResult(node.handle);
  }
  SceneNode newNode{};
  newNode.priority = resolved.priority;
  newNode.request = std::move(resolved);
  newNode.key = key;
  newNode.publicationTarget = key.publicationTarget;
  SceneNode &node = scenes_.insert(std::move(newNode));
  sceneInFlight_.emplace(node.key, node.handle);
  const SceneLoadHandle handle = node.handle;
  const std::string workerPath = node.request.path;
  const SceneImportOptions workerOptions = node.request.importOptions;
  auto task = scheduler_.enqueue(makeAssetCpuJob<SceneManifestCompletion>(
      node.priority, AssetWorkClass::Metadata, estimateSourceBytes(workerPath),
      node.request.debugName.empty() ? workerPath : node.request.debugName,
      SceneManifestCompletion{.handle = handle},
      [workerPath, workerOptions] {
        return SceneImporter::loadSceneFromFile(workerPath, workerOptions);
      },
      [this](SceneManifestCompletion completion) {
        pushCompletion(std::move(completion));
      }));
  if (task.hasError()) {
    node.subscriberCount = 0u;
    finishSceneNode(node, SceneLoadState::Failed, task.error());
    return Result<SceneLoadHandle, std::string>::makeError(task.error());
  }
  node.manifestTask = task.value();
  node.dependencies.emplace_back(node.manifestTask);
  return Result<SceneLoadHandle, std::string>::makeResult(handle);
}

ScenePublicationTargetHandle
AssetSystem::registerScenePublicationTarget(RenderScene &scene, NodeId parent) {
  std::lock_guard stateLock(stateMutex_);
  ScenePublicationTargetNode &target =
      sceneTargets_.insert(ScenePublicationTargetNode{
          .scene = &scene,
          .parent = parent,
      });
  return target.handle;
}

bool AssetSystem::unregisterScenePublicationTarget(
    ScenePublicationTargetHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  ScenePublicationTargetNode *target = find(handle);
  if (target == nullptr) {
    return false;
  }
  for (const SceneNode &scene : scenes_) {
    if (scene.publicationTarget == handle &&
        !isSceneTerminalState(scene.state)) {
      return false;
    }
  }
  target->scene = nullptr;
  sceneTargets_.release(handle);
  return true;
}

ScenePublicationTargetSnapshot
AssetSystem::query(ScenePublicationTargetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  if (find(handle) == nullptr) {
    return ScenePublicationTargetSnapshot{
        .requestCount = 1u,
        .failedCount = 1u,
    };
  }
  ScenePublicationTargetSnapshot snapshot{};
  float progressSum = 0.0f;
  for (const SceneNode &scene : scenes_) {
    if (scene.publicationTarget != handle) {
      continue;
    }
    ++snapshot.requestCount;
    progressSum += scene.progress;
    if (!isSceneTerminalState(scene.state)) {
      ++snapshot.pendingCount;
    } else if (scene.state == SceneLoadState::Failed) {
      ++snapshot.failedCount;
    } else if (scene.state == SceneLoadState::Cancelled) {
      ++snapshot.cancelledCount;
    }
  }
  snapshot.progress =
      snapshot.requestCount == 0u
          ? 1.0f
          : progressSum / static_cast<float>(snapshot.requestCount);
  return snapshot;
}

AssetLoadSnapshot AssetSystem::query(TextureAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  return assetSnapshot(find(handle), "invalid or stale texture asset handle");
}

AssetLoadSnapshot AssetSystem::query(ModelAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  return assetSnapshot(find(handle), "invalid or stale model asset handle");
}

AssetLoadSnapshot AssetSystem::query(MaterialAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  return assetSnapshot(find(handle), "invalid or stale material asset handle");
}

AssetLoadSnapshot AssetSystem::query(EnvironmentAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const EnvironmentNode *node = find(handle);
  if (node == nullptr) {
    return AssetLoadSnapshot{
        .state = AssetState::Failed,
        .error = "invalid or stale environment asset handle",
    };
  }
  float progress = 0.0f;
  uint32_t dependencyCount = 0u;
  for (const TextureAssetHandle texture : node->textures) {
    if (!isValid(texture)) {
      continue;
    }
    ++dependencyCount;
    const TextureNode *dependency = find(texture);
    progress += assetStateProgress(dependency != nullptr ? dependency->state
                                                         : AssetState::Failed);
  }
  if (dependencyCount != 0u) {
    progress /= static_cast<float>(dependencyCount);
  } else {
    progress = assetStateProgress(node->state);
  }
  return AssetLoadSnapshot{
      .state = node->state,
      .priority = node->priority,
      .progress = node->state == AssetState::Published ? 1.0f : progress,
      .error = node->error,
  };
}

SceneLoadSnapshot AssetSystem::query(SceneLoadHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const SceneNode *node = find(handle);
  if (node == nullptr) {
    return SceneLoadSnapshot{
        .state = SceneLoadState::Failed,
        .error = "invalid or stale scene load handle",
    };
  }
  uint32_t publishedRenderables = 0u;
  for (const RenderableId renderable : node->instantiation.renderables) {
    publishedRenderables += isValid(renderable) ? 1u : 0u;
  }
  const bool publishProgress =
      node->request.publication == ScenePublicationPolicy::Progressive ||
      isSceneTerminalState(node->state);
  bool cancellationPendingGpuRetirement = false;
  for (const SceneDependency &dependency : node->dependencies)
    std::visit(
        [this, &cancellationPendingGpuRetirement](auto handleValue) {
          if constexpr (!std::is_same_v<decltype(handleValue),
                                        AssetCpuTaskHandle>)
            cancellationPendingGpuRetirement |=
                find(handleValue)->state == AssetState::CancelRequested;
        },
        dependency);
  const SceneAssetCounts models =
      collectSceneCounts(std::span<const ModelAssetHandle>(node->models));
  const SceneAssetCounts materials =
      collectSceneCounts(std::span<const MaterialAssetHandle>(node->materials));
  const SceneAssetCounts textures = collectSceneTextureCounts(*node);
  return SceneLoadSnapshot{
      .state = node->state,
      .priority = node->priority,
      .progress = node->progress,
      .sourceDiscoveryComplete = node->manifest.has_value(),
      .hierarchyPublished = publishProgress && node->hierarchyPublished,
      .cancellationPendingGpuRetirement = cancellationPendingGpuRetirement,
      .models = models,
      .materials = materials,
      .textures = textures,
      .publishedRenderables = publishProgress ? publishedRenderables : 0u,
      .totalRenderables =
          node->manifest.has_value()
              ? static_cast<uint32_t>(node->manifest->prefab.renderables.size())
              : 0u,
      .requiredFailures = node->requiredFailures,
      .optionalFailures = node->optionalFailures + models.failed +
                          materials.failed + textures.failed,
      .cpuPayloadBytes = node->cpuPayloadBytes,
      .error = node->error,
  };
}

std::optional<TextureRef>
AssetSystem::tryResolve(TextureAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  return resolvedAsset(find(handle));
}

std::optional<ModelRef> AssetSystem::tryResolve(ModelAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  return resolvedAsset(find(handle));
}

std::optional<MaterialRef>
AssetSystem::tryResolve(MaterialAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  return resolvedAsset(find(handle));
}

std::optional<EnvironmentHandles>
AssetSystem::tryResolve(EnvironmentAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  return resolvedAsset(find(handle));
}

const ScenePrefab *
AssetSystem::tryGetScenePrefab(SceneLoadHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const SceneNode *node = find(handle);
  return node != nullptr && node->manifest.has_value() ? &node->manifest->prefab
                                                       : nullptr;
}

std::optional<ScenePrefabAssets>
AssetSystem::tryGetSceneAssets(SceneLoadHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const SceneNode *node = find(handle);
  if (node == nullptr || !node->manifest.has_value()) {
    return std::nullopt;
  }
  ScenePrefabAssets assets;
  assets.models.reserve(node->models.size());
  for (const ModelAssetHandle model : node->models) {
    const ModelNode *dependency = find(model);
    assets.models.push_back(dependency != nullptr &&
                                    dependency->state == AssetState::Published
                                ? dependency->published
                                : kInvalidModelRef);
  }
  assets.materials.reserve(node->materials.size());
  for (const MaterialAssetHandle material : node->materials) {
    const MaterialNode *dependency = find(material);
    assets.materials.push_back(
        dependency != nullptr && dependency->state == AssetState::Published
            ? dependency->published
            : kInvalidMaterialRef);
  }
  return assets;
}

std::optional<SceneInstantiationMap>
AssetSystem::tryGetSceneInstantiation(SceneLoadHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const SceneNode *node = find(handle);
  return node != nullptr && node->hierarchyPublished
             ? std::optional<SceneInstantiationMap>(node->instantiation)
             : std::nullopt;
}

template <typename Handle>
void AssetSystem::setGpuAssetPriority(Handle handle, AssetPriority priority) {
  if (auto *node = find(handle); node != nullptr && priority < node->priority) {
    node->priority = priority;
    (void)scheduler_.setPriority(node->cpuTask, priority);
    (void)scheduler_.setPriority(node->gpuTask, priority);
  }
}

void AssetSystem::setPriority(TextureAssetHandle handle,
                              AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  setGpuAssetPriority(handle, priority);
}

void AssetSystem::setPriority(ModelAssetHandle handle, AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  setGpuAssetPriority(handle, priority);
}

void AssetSystem::setPriority(MaterialAssetHandle handle,
                              AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  if (MaterialNode *node = find(handle)) {
    if (priority >= node->priority) {
      return;
    }
    node->priority = priority;
    for (TextureAssetHandle texture : node->request.textures)
      setPriority(texture, priority);
  }
}

void AssetSystem::setPriority(EnvironmentAssetHandle handle,
                              AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  EnvironmentNode *node = find(handle);
  if (node == nullptr || priority >= node->priority) {
    return;
  }
  node->priority = priority;
  for (const TextureAssetHandle texture : node->textures)
    setPriority(texture, priority);
}

void AssetSystem::setPriority(SceneLoadHandle handle, AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  SceneNode *node = find(handle);
  if (node == nullptr || priority >= node->priority) {
    return;
  }
  node->priority = priority;
  for (const SceneDependency &dependency : node->dependencies)
    std::visit(
        [this, priority](auto dependencyHandle) {
          if constexpr (std::is_same_v<decltype(dependencyHandle),
                                       AssetCpuTaskHandle>)
            (void)scheduler_.setPriority(dependencyHandle, priority);
          else
            setPriority(dependencyHandle, priority);
        },
        dependency);
}

template <typename Handle> void AssetSystem::cancelGpuAsset(Handle handle) {
  auto *node = find(handle);
  if (!claimCancellation(node, releasedTerminalNodes_, [](AssetState state) {
        return state == AssetState::Cancelled || state == AssetState::Failed;
      })) {
    return;
  }
  const auto finish = [this, node] {
    finishNode(*node, AssetState::Cancelled);
  };
  if (node->state == AssetState::Published) {
    resources_.release(node->published);
    node->published = {};
    finish();
    return;
  }
  if (node->state == AssetState::CpuReady) {
    finish();
    return;
  }
  if (node->state == AssetState::GpuReady) {
    deferPreparedDestroy(scheduler_, std::move(node->preparedGpu));
    finish();
    return;
  }
  node->state = AssetState::CancelRequested;
  (void)scheduler_.cancel(node->cpuTask);
  (void)scheduler_.cancel(node->gpuTask);
}

void AssetSystem::cancel(TextureAssetHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  cancelGpuAsset(handle);
}

void AssetSystem::cancel(ModelAssetHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  cancelGpuAsset(handle);
}

void AssetSystem::cancel(MaterialAssetHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  MaterialNode *node = find(handle);
  if (!claimCancellation(node, releasedTerminalNodes_, [](AssetState state) {
        return state == AssetState::Cancelled || state == AssetState::Failed;
      })) {
    return;
  }
  if (node->state == AssetState::Published) {
    resources_.release(node->published);
    node->published = kInvalidMaterialRef;
  }
  finishNode(*node, AssetState::Cancelled);
}

void AssetSystem::cancel(EnvironmentAssetHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  EnvironmentNode *node = find(handle);
  if (!claimCancellation(node, releasedTerminalNodes_, [](AssetState state) {
        return state == AssetState::Cancelled;
      })) {
    return;
  }
  node->state = AssetState::CancelRequested;
  for (const TextureAssetHandle texture : node->textures)
    cancel(texture);
}

void AssetSystem::cancel(SceneLoadHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  SceneNode *node = find(handle);
  if (!claimCancellation(node, releasedTerminalNodes_,
                         [](SceneLoadState state) {
                           return state == SceneLoadState::Cancelled;
                         })) {
    return;
  }
  node->state = SceneLoadState::Cancelling;
}

Result<AssetPublicationStats, std::string>
AssetSystem::prepareFrame(AssetPublicationContext context) {
  std::lock_guard stateLock(stateMutex_);
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  using Clock = std::chrono::steady_clock;
  const Clock::time_point start = Clock::now();
  const Clock::duration frameBudget =
      std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double, std::milli>(
              config_.maxMainThreadMillisecondsPerFrame));
  const Clock::time_point deadline = start + frameBudget;
  const Clock::duration completionReserve = std::min(
      frameBudget / 4, std::chrono::duration_cast<Clock::duration>(
                           std::chrono::duration<double, std::milli>(0.5)));
  const Clock::time_point workDeadline = deadline - completionReserve;
  const auto deadlineReached = [&workDeadline] {
    return Clock::now() >= workDeadline;
  };
  double maxOperationMilliseconds = 0.0;
  const auto measureOperation =
      [&maxOperationMilliseconds](auto &&operation) -> decltype(auto) {
    OperationTimer timer(maxOperationMilliseconds);
    return operation();
  };
  resources_.beginPublicationBatch();
  AssetPublicationStats stats{};
  std::vector<CpuCompletion> completions =
      takeCompletions(config_.maxCpuCompletionsPerFrame);
  const auto ingestCpu = [this, &stats](auto &completion) {
    auto *node = find(completion.handle);
    const auto finish = [this, node](AssetState state, std::string error = {}) {
      finishNode(*node, state, std::move(error));
    };
    if (completion.cancelled || node->state == AssetState::CancelRequested) {
      finish(AssetState::Cancelled);
      ++stats.cancelled;
    } else if (!completion.error.empty()) {
      finish(AssetState::Failed, std::move(completion.error));
      ++stats.failed;
    } else {
      node->cpuPayloadBytes = completion.prepared->uploadBytes();
      node->prepared = std::move(completion.prepared);
      node->state = AssetState::CpuReady;
    }
  };
  const auto ingestGpu = [this, &stats](auto &completion) {
    using Completion = std::decay_t<decltype(completion)>;
    auto *node = find(completion.handle);
    const auto discard = [this, &completion] {
      deferPreparedDestroy(scheduler_, std::move(completion.prepared));
    };
    const auto finish = [this, node](AssetState state, std::string error = {}) {
      finishNode(*node, state, std::move(error));
    };
    if (completion.cancelled || node->state == AssetState::CancelRequested) {
      discard();
      finish(AssetState::Cancelled);
      ++stats.cancelled;
    } else if (!completion.error.empty()) {
      finish(AssetState::Failed, std::move(completion.error));
      ++stats.failed;
    } else {
      if constexpr (std::is_same_v<Completion, TextureGpuCompletion>) {
        node->preparedGpu = std::move(completion.prepared);
        node->publication.desc = completion.publication.desc;
        node->publication.desc.data = {};
        node->publication.debugName =
            std::move(completion.publication.debugName);
      } else {
        node->preparedGpu = std::move(completion.prepared);
      }
      node->state = AssetState::GpuReady;
    }
  };
  size_t processedCompletionCount = 0u;
  for (; processedCompletionCount < completions.size();
       ++processedCompletionCount) {
    if (deadlineReached()) {
      break;
    }
    CpuCompletion &completion = completions[processedCompletionCount];
    measureOperation([&] {
      std::visit(
          [this, &stats, &ingestCpu, &ingestGpu](auto &typed) {
            using Completion = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<Completion, TextureCpuCompletion> ||
                          std::is_same_v<Completion, ModelCpuCompletion>) {
              ingestCpu(typed);
            } else if constexpr (std::is_same_v<Completion,
                                                TextureGpuCompletion> ||
                                 std::is_same_v<Completion,
                                                ModelGpuCompletion>) {
              ingestGpu(typed);
            } else if constexpr (std::is_same_v<Completion,
                                                SceneManifestCompletion>) {
              SceneNode *node = find(typed.handle);
              if (node == nullptr) {
                return;
              }
              if (typed.cancelled ||
                  node->state == SceneLoadState::Cancelling) {
                finishSceneNode(*node, SceneLoadState::Cancelled);
                ++stats.cancelled;
                return;
              }
              if (!typed.error.empty()) {
                ++node->requiredFailures;
                finishSceneNode(*node, SceneLoadState::Failed,
                                std::move(typed.error));
                ++stats.failed;
                return;
              }
              node->manifest = std::move(typed.manifest);
              const ScenePrefab &prefab = node->manifest->prefab;
              node->cpuPayloadBytes = estimateSourceBytes(node->request.path);
              node->models.resize(prefab.meshAssets.size());
              node->materials.resize(prefab.materialAssets.size());
              node->materialPreparationFinished.resize(
                  prefab.materialAssets.size(), 0u);
              node->modelFallbackMapped.resize(prefab.meshAssets.size(), 0u);
              node->renderableMaterialMapped.resize(prefab.renderables.size(),
                                                    0u);
              node->state = SceneLoadState::ManifestReady;
            } else if constexpr (std::is_same_v<Completion,
                                                SceneMaterialCompletion>) {
              SceneNode *node = find(typed.scene);
              if (node == nullptr) {
                return;
              }
              node->materialPreparationFinished[typed.materialIndex] = 1u;
              if (typed.cancelled ||
                  node->state == SceneLoadState::Cancelling) {
                return;
              }
              if (!typed.error.empty()) {
                ++node->optionalFailures;
                return;
              }
              PreparedImportedMaterial &prepared = *typed.prepared;
              MaterialAssetRequest materialRequest{
                  .desc = prepared.desc,
                  .debugName = prepared.debugName,
                  .sourceIdentity = prepared.sourceIdentity,
              };
              node->optionalFailures +=
                  static_cast<uint32_t>(prepared.optionalTextureErrors.size());
              for (size_t slotIndex = 0u; slotIndex < prepared.textures.size();
                   ++slotIndex) {
                if (!prepared.textures[slotIndex].has_value()) {
                  continue;
                }
                auto texture = requestTexture(*prepared.textures[slotIndex],
                                              node->priority);
                if (texture.hasError()) {
                  ++node->optionalFailures;
                  continue;
                }
                materialRequest.textures[slotIndex] = texture.value();
                node->dependencies.emplace_back(texture.value());
              }
              node->materials[typed.materialIndex] =
                  requestMaterial(materialRequest, node->priority);
              node->dependencies.emplace_back(
                  node->materials[typed.materialIndex]);
            }
          },
          completion);
    });
    ++stats.cpuCompletions;
  }
  stats.deferredCpuCompletions =
      static_cast<uint32_t>(completions.size() - processedCompletionCount);
  returnCompletions(
      std::span<CpuCompletion>(completions).subspan(processedCompletionCount));
  ScopedScratch frameScratch(frameScratch_);
  std::pmr::vector<TextureNode *> textureCpuReady(frameScratch.resource());
  std::pmr::vector<TextureNode *> textureGpuReady(frameScratch.resource());
  std::pmr::vector<TextureNode *> textureResidency(frameScratch.resource());
  std::pmr::vector<ModelNode *> modelCpuReady(frameScratch.resource());
  std::pmr::vector<ModelNode *> modelGpuReady(frameScratch.resource());
  std::pmr::vector<ModelNode *> modelResidency(frameScratch.resource());
  std::pmr::vector<MaterialNode *> materialReady(frameScratch.resource());
  std::pmr::vector<EnvironmentNode *> activeEnvironments(
      frameScratch.resource());
  std::pmr::vector<SceneNode *> activeScenes(frameScratch.resource());
  const auto collectGpuPhases = [](auto &pool, auto &cpuReady, auto &gpuReady,
                                   auto &residency) {
    using Node = std::remove_pointer_t<
        typename std::decay_t<decltype(cpuReady)>::value_type>;
    pool.forEachLive([&](Node &node) {
      switch (node.state) {
      case AssetState::CpuReady:
        cpuReady.push_back(&node);
        break;
      case AssetState::GpuReady:
        gpuReady.push_back(&node);
        break;
      case AssetState::GpuSubmitted:
        residency.push_back(&node);
        break;
      case AssetState::CancelRequested:
        if (node.pending) {
          residency.push_back(&node);
        }
        break;
      default:
        break;
      }
    });
  };
  collectGpuPhases(textures_, textureCpuReady, textureGpuReady,
                   textureResidency);
  collectGpuPhases(models_, modelCpuReady, modelGpuReady, modelResidency);
  materials_.forEachLive([&](MaterialNode &node) {
    if (node.state == AssetState::CpuReady) {
      materialReady.push_back(&node);
    }
  });
  environments_.forEachLive([&](EnvironmentNode &node) {
    if (node.state != AssetState::Failed &&
        node.state != AssetState::Cancelled) {
      activeEnvironments.push_back(&node);
    }
  });
  scenes_.forEachLive([&](SceneNode &node) {
    if (node.state != SceneLoadState::Failed &&
        node.state != SceneLoadState::Cancelled) {
      activeScenes.push_back(&node);
    }
  });
  uint32_t selectedCount = 0u;
  uint64_t selectedBytes = 0u;
  uint64_t submittedBytes = 0u;
  std::pmr::vector<TextureNode *> submittedTextures(frameScratch.resource());
  std::pmr::vector<ModelNode *> submittedModels(frameScratch.resource());
  const auto finishGpuAsset = [this, &stats](auto &node, AssetState state,
                                             std::string error = {}) {
    finishNode(node, state, std::move(error));
    state == AssetState::Failed ? ++stats.failed : ++stats.cancelled;
  };
  const auto publishPrepared = [&](auto &candidates, auto &submitted) {
    using Node = std::remove_pointer_t<
        typename std::decay_t<decltype(candidates)>::value_type>;
    for (Node *candidate : candidates) {
      Node &node = *candidate;
      if (deadlineReached()) {
        break;
      }
      if (!canSelectMaterialization(node.cpuPayloadBytes, selectedCount,
                                    selectedBytes, config_)) {
        continue;
      }
      auto published = measureOperation([&] {
        if constexpr (std::is_same_v<Node, TextureNode>) {
          auto result =
              gpu_.publishPreparedTexture(std::move(node.preparedGpu));
          using Published = Result<std::unique_ptr<Texture>, std::string>;
          return result.hasError()
                     ? Published::makeError(result.error())
                     : Published::makeResult(Texture::adoptPrepared(
                           gpu_, result.value(), node.publication.desc,
                           std::move(node.publication.debugName)));
        } else {
          return Model::publishPreparedGpu(gpu_, std::move(node.preparedGpu));
        }
      });
      if (published.hasError()) {
        finishGpuAsset(node, AssetState::Failed, published.error());
        continue;
      }
      node.pending = std::move(published.value());
      submitted.push_back(&node);
      ++selectedCount;
      selectedBytes += node.cpuPayloadBytes;
      submittedBytes += node.cpuPayloadBytes;
    }
  };
  publishPrepared(textureGpuReady, submittedTextures);
  publishPrepared(modelGpuReady, submittedModels);
  const auto materializeCpuReady = [&](auto &candidates, auto &submitted) {
    using Node = std::remove_pointer_t<
        typename std::decay_t<decltype(candidates)>::value_type>;
    using Completion =
        std::conditional_t<std::is_same_v<Node, TextureNode>,
                           TextureGpuCompletion, ModelGpuCompletion>;
    for (Node *candidate : candidates) {
      Node &node = *candidate;
      if (deadlineReached()) {
        break;
      }
      if (!canSelectMaterialization(node.cpuPayloadBytes, selectedCount,
                                    selectedBytes, config_)) {
        continue;
      }
      node.state = AssetState::GpuQueued;
      const uint64_t bytes = node.prepared->uploadBytes();
      const bool background =
          supportsBackgroundPreparation(gpu_, *node.prepared);
      if (background) {
        const auto handle = node.handle;
        auto [gpuDebugName, taskDebugName] =
            materializationNames(node.request, *node.prepared);
        auto prepared = std::move(*node.prepared);
        node.prepared.reset();
        auto task = scheduler_.enqueue(AssetCpuJob{
            .priority = node.priority,
            .workClass = AssetWorkClass::GpuMaterialization,
            .estimatedBytes = bytes,
            .debugName = std::move(taskDebugName),
            .execute =
                [this, handle, debugName = std::move(gpuDebugName),
                 prepared =
                     std::move(prepared)](std::stop_token stopToken) mutable {
                  if (stopToken.stop_requested()) {
                    pushCompletion(
                        Completion{.handle = handle, .cancelled = true});
                    return;
                  }
                  TextureDesc storedDesc{};
                  if constexpr (std::is_same_v<Node, TextureNode>) {
                    storedDesc = prepared.createDesc;
                    storedDesc.data = {};
                  }
                  auto result = prepareGpuAsset(gpu_, prepared, debugName);
                  Completion completion{.handle = handle};
                  completion.cancelled = stopToken.stop_requested();
                  if (completion.cancelled && !result.hasError())
                    result.value().reset();
                  else if (result.hasError())
                    completion.error = result.error();
                  else
                    completion.prepared = std::move(result.value());
                  if constexpr (std::is_same_v<Node, TextureNode>) {
                    completion.publication = {
                        .desc = storedDesc,
                        .debugName = std::move(debugName),
                    };
                  }
                  pushCompletion(std::move(completion));
                },
            .onCancelled =
                [this, handle] {
                  pushCompletion(
                      Completion{.handle = handle, .cancelled = true});
                },
        });
        if (task.hasError()) {
          finishGpuAsset(node, AssetState::Failed, task.error());
          continue;
        }
        node.gpuTask = task.value();
        ++selectedCount;
        selectedBytes += bytes;
        continue;
      }
      auto result = measureOperation([&] {
        return createGpuAsset(gpu_, std::move(*node.prepared),
                              node.request.debugName);
      });
      node.prepared.reset();
      if (result.hasError()) {
        finishGpuAsset(node, AssetState::Failed, result.error());
        continue;
      }
      node.pending = std::move(result.value());
      submitted.push_back(&node);
      ++selectedCount;
      selectedBytes += bytes;
      submittedBytes += bytes;
    }
  };
  materializeCpuReady(textureCpuReady, submittedTextures);
  materializeCpuReady(modelCpuReady, submittedModels);
  if (!submittedTextures.empty() || !submittedModels.empty()) {
    auto uploadResult =
        measureOperation([&] { return gpu_.submitPendingUploads(); });
    if (uploadResult.hasError()) {
      const auto failSubmitted = [&](auto &nodes) {
        for (auto *node : nodes) {
          node->pending.reset();
          finishNode(*node, AssetState::Failed, uploadResult.error());
          ++stats.failed;
        }
      };
      failSubmitted(submittedTextures);
      failSubmitted(submittedModels);
    } else {
      const auto markSubmitted = [&](auto &nodes, auto &residency) {
        for (auto *node : nodes) {
          node->upload = uploadResult.value();
          node->state = AssetState::GpuSubmitted;
          residency.push_back(node);
        }
      };
      markSubmitted(submittedTextures, textureResidency);
      markSubmitted(submittedModels, modelResidency);
      stats.gpuMaterialized = static_cast<uint32_t>(submittedTextures.size() +
                                                    submittedModels.size());
      stats.uploadBytes = submittedBytes;
    }
  }
  const auto publishResident = [&](auto &candidates) {
    using Node = std::remove_pointer_t<
        typename std::decay_t<decltype(candidates)>::value_type>;
    for (Node *candidate : candidates) {
      Node &node = *candidate;
      if (deadlineReached()) {
        break;
      }
      auto &pending = node.pending;
      const auto finish = [&](AssetState state, std::string error = {}) {
        finishNode(node, state, std::move(error));
      };
      if (!gpu_.isSubmissionComplete(node.upload)) {
        continue;
      }
      auto visible = gpu_.makeSubmissionVisibleToGraphics(node.upload);
      if (visible.hasError()) {
        pending.reset();
        finish(AssetState::Failed, visible.error());
        ++stats.failed;
        continue;
      }
      if (!visible.value()) {
        continue;
      }
      if (node.state == AssetState::CancelRequested) {
        pending.reset();
        finish(AssetState::Cancelled);
        ++stats.cancelled;
        continue;
      }
      node.state = AssetState::Resident;
      auto adopted =
          adoptGpuAsset(resources_, node.request, node.key, std::move(pending));
      if (adopted.hasError()) {
        finish(AssetState::Failed, adopted.error());
        ++stats.failed;
        continue;
      }
      node.published = adopted.value();
      finish(AssetState::Published);
      ++stats.published;
    }
  };
  publishResident(textureResidency);
  publishResident(modelResidency);
  uint32_t materialPublications = 0u;
  for (MaterialNode *candidate : materialReady) {
    MaterialNode &node = *candidate;
    if (deadlineReached()) {
      break;
    }
    if (materialPublications >= config_.maxMaterialPublicationsPerFrame) {
      continue;
    }
    MaterialRequest materialRequest{
        .desc = node.request.desc,
        .debugName = node.request.debugName,
        .sourceIdentity = node.request.sourceIdentity,
    };
    bool dependenciesReady = true;
    const auto resolveDependency =
        [this,
         &dependenciesReady](TextureAssetHandle dependency) -> TextureRef {
      if (!isValid(dependency)) {
        return kInvalidTextureRef;
      }
      const TextureNode &texture = *find(dependency);
      if (texture.state == AssetState::Cancelled ||
          texture.state == AssetState::Failed) {
        return kInvalidTextureRef;
      }
      if (texture.state != AssetState::Published) {
        dependenciesReady = false;
        return kInvalidTextureRef;
      }
      return texture.published;
    };
    for (size_t index = 0; index < node.request.textures.size(); ++index) {
      materialRequest.textureRefs[index] =
          resolveDependency(node.request.textures[index]);
    }
    if (!dependenciesReady) {
      continue;
    }
    auto resolved = resources_.resolveMaterialRequest(materialRequest);
    if (resolved.hasError()) {
      finishNode(node, AssetState::Failed, resolved.error());
      ++stats.failed;
      continue;
    }
    auto material = resources_.acquireMaterial(resolved.value());
    if (material.hasError()) {
      finishNode(node, AssetState::Failed, material.error());
      ++stats.failed;
      continue;
    }
    node.published = material.value();
    finishNode(node, AssetState::Published);
    ++materialPublications;
    ++stats.published;
  }
  for (EnvironmentNode *candidate : activeEnvironments) {
    EnvironmentNode &node = *candidate;
    if (deadlineReached()) {
      break;
    }
    if (node.state == AssetState::Published && node.environmentPublished &&
        context.scene == node.boundScene && context.scene != nullptr &&
        context.scene->environment() == node.published) {
      continue;
    }
    bool dependenciesTerminal = true;
    bool requiredDependencyFailed = false;
    EnvironmentHandles resolved{};
    for (size_t index = 0u; index < node.textures.size(); ++index) {
      const TextureAssetHandle textureHandle = node.textures[index];
      if (!isValid(textureHandle)) {
        continue;
      }
      const TextureNode &texture = *find(textureHandle);
      if (!isAssetTerminalState(texture.state)) {
        dependenciesTerminal = false;
        continue;
      }
      if (texture.state == AssetState::Published) {
        resolved.*kEnvironmentTextureFields[index] = texture.published;
      } else if (!node.request.optionalTextures[index]) {
        requiredDependencyFailed = true;
        if (node.error.empty()) {
          node.error = texture.error;
        }
      }
    }
    if (node.state == AssetState::CancelRequested) {
      if (node.environmentPublished && node.boundScene != nullptr &&
          node.boundScene->environment() == node.published) {
        node.boundScene->setEnvironment(EnvironmentHandles{});
      }
      if (dependenciesTerminal) {
        node.published = {};
        node.environmentPublished = false;
        node.state = AssetState::Cancelled;
        if (node.subscriberCount == 0u) {
          releasedTerminalNodes_ = true;
        }
        ++stats.cancelled;
      }
      continue;
    }
    if (!dependenciesTerminal) {
      node.state = AssetState::CpuReady;
      continue;
    }
    if (requiredDependencyFailed) {
      node.state = AssetState::Failed;
      if (node.error.empty()) {
        node.error = "AssetSystem: required environment texture failed";
      }
      ++stats.failed;
      continue;
    }
    if (context.scene == nullptr) {
      continue;
    }
    context.scene->setEnvironment(resolved);
    node.boundScene = context.scene;
    node.published = resolved;
    node.environmentPublished = true;
    node.state = AssetState::Published;
    ++stats.published;
  }
  uint32_t remainingScenePatches = config_.maxScenePatchesPerFrame;
  std::pmr::vector<RenderScene *> dirtyScenes(frameScratch.resource());
  const size_t sceneCount = activeScenes.size();
  const size_t sceneStart =
      sceneCount == 0u ? 0u : scenePrepareCursor_ % sceneCount;
  for (size_t visited = 0u; visited < sceneCount; ++visited) {
    if (deadlineReached()) {
      break;
    }
    const size_t sceneIndex = (sceneStart + visited) % sceneCount;
    SceneNode &node = *activeScenes[sceneIndex];
    scenePrepareCursor_ = (sceneIndex + 1u) % sceneCount;
    const ScenePublicationTargetNode *publicationTarget =
        find(node.publicationTarget);
    RenderScene *candidateScene =
        node.boundScene != nullptr
            ? node.boundScene
            : (publicationTarget != nullptr ? publicationTarget->scene
                                            : context.scene);
    const NodeId candidateParent = publicationTarget != nullptr
                                       ? publicationTarget->parent
                                       : context.parent;
    const bool sceneDirty =
        prepareSceneNode(node, stats, remainingScenePatches, workDeadline,
                         candidateScene, candidateParent);
    node.progress = std::max(node.progress, progressForScene(node, *this));
    node.commitPending |= sceneDirty;
    if (node.commitPending && candidateScene != nullptr &&
        !isValid(node.publicationTarget) &&
        std::ranges::find(dirtyScenes, candidateScene) == dirtyScenes.end()) {
      dirtyScenes.push_back(candidateScene);
    }
  }
  resources_.endPublicationBatch();
  if (context.commitScene) {
    for (RenderScene *scene : dirtyScenes) {
      if (deadlineReached()) {
        break;
      }
      auto commit = measureOperation([&] { return scene->commit(); });
      if (commit.hasError()) {
        return Result<AssetPublicationStats, std::string>::makeError(
            "AssetSystem: progressive scene commit failed: " + commit.error());
      }
      for (SceneNode &node : scenes_) {
        if (node.boundScene == scene) {
          node.commitPending = false;
        }
      }
      ++stats.sceneCommits;
    }
  }
  const auto finishTime = Clock::now();
  stats.mainThreadMilliseconds =
      std::chrono::duration<double, std::milli>(finishTime - start).count();
  stats.maxOperationMilliseconds = maxOperationMilliseconds;
  stats.deadlineExceeded = finishTime > deadline;
  return Result<AssetPublicationStats, std::string>::makeResult(stats);
}

void AssetSystem::pushCompletion(CpuCompletion completion) {
  std::lock_guard lock(completionMutex_);
  completions_.push_back(std::move(completion));
}

std::vector<AssetSystem::CpuCompletion>
AssetSystem::takeCompletions(uint32_t maxCompletions) {
  std::lock_guard lock(completionMutex_);
  const size_t count = std::min<size_t>(completions_.size(), maxCompletions);
  std::vector<CpuCompletion> result(
      std::make_move_iterator(completions_.begin()),
      std::make_move_iterator(completions_.begin() + count));
  completions_.erase(completions_.begin(), completions_.begin() + count);
  return result;
}

void AssetSystem::returnCompletions(std::span<CpuCompletion> completions) {
  std::lock_guard lock(completionMutex_);
  completions_.insert(completions_.begin(),
                      std::make_move_iterator(completions.begin()),
                      std::make_move_iterator(completions.end()));
}

void AssetSystem::reclaimReleasedNodes() {
  if (!releasedTerminalNodes_) {
    return;
  }
  releasedTerminalNodes_ = false;
  textures_.reclaim(isAssetTerminalState);
  models_.reclaim(isAssetTerminalState);
  materials_.reclaim(isAssetTerminalState);
  environments_.reclaim(isAssetTerminalState);
  scenes_.reclaim(isSceneTerminalState);
}

void AssetSystem::finishSceneNode(SceneNode &node, SceneLoadState terminalState,
                                  std::string error) {
  node.state = terminalState;
  if (!error.empty() || node.error.empty()) {
    node.error = std::move(error);
  }
  if (isSceneTerminalState(terminalState)) {
    node.progress = 1.0f;
    sceneInFlight_.erase(node.key);
  }
  if (terminalState == SceneLoadState::Cancelled) {
    const SceneLoadHandle handle = node.handle;
    std::string retainedError = std::move(node.error);
    node = {};
    node.handle = handle;
    node.state = SceneLoadState::Cancelled;
    node.subscriberCount = 0u;
    node.progress = 1.0f;
    node.error = std::move(retainedError);
  }
  if (node.subscriberCount == 0u && isSceneTerminalState(terminalState)) {
    releasedTerminalNodes_ = true;
  }
}

bool AssetSystem::cancelSceneDependencies(
    SceneNode &node, std::chrono::steady_clock::time_point deadline,
    uint32_t maxOperations) {
  uint32_t operations = 0u;
  while (node.cancellationCursor < node.dependencies.size() &&
         operations < maxOperations &&
         std::chrono::steady_clock::now() < deadline) {
    std::visit(
        [this](auto handle) {
          using Handle = decltype(handle);
          if constexpr (std::is_same_v<Handle, AssetCpuTaskHandle>) {
            (void)scheduler_.cancel(handle);
          } else {
            cancel(handle);
          }
        },
        node.dependencies[node.cancellationCursor++]);
    ++operations;
  }
  return node.cancellationCursor == node.dependencies.size();
}

bool AssetSystem::prepareSceneNode(
    SceneNode &node, AssetPublicationStats &stats, uint32_t &patchBudget,
    std::chrono::steady_clock::time_point deadline, RenderScene *targetScene,
    NodeId targetParent) {
  bool sceneDirty = false;
  const auto deadlineReached = [&deadline] {
    return std::chrono::steady_clock::now() >= deadline;
  };
  const bool completeOnly =
      node.request.publication == ScenePublicationPolicy::CompleteOnly;
  const auto dependenciesTerminal = [this, &node] {
    const ScenePrefab &prefab = node.manifest->prefab;
    if ((!isValid(node.fallbackMaterial) &&
         node.state != SceneLoadState::Cancelling) ||
        node.modelAdmissionCursor != prefab.meshAssets.size() ||
        node.materialAdmissionCursor != prefab.materialAssets.size()) {
      return false;
    }
    if (std::ranges::any_of(node.materialPreparationFinished,
                            [](uint8_t finished) { return finished == 0u; })) {
      return false;
    }
    return std::ranges::all_of(node.dependencies, [this](const auto &entry) {
      return std::visit(
          [this](auto handle) {
            if constexpr (std::is_same_v<decltype(handle), AssetCpuTaskHandle>)
              return true;
            else
              return isAssetTerminalState(find(handle)->state);
          },
          entry);
    });
  };
  const auto destroyPublishedHierarchy = [&node, &sceneDirty] {
    if (isValid(node.publicationTarget) || !node.hierarchyPublished) {
      return;
    }
    const ScenePrefab &prefab = node.manifest->prefab;
    for (uint32_t nodeIndex = 0u; nodeIndex < prefab.nodes.size();
         ++nodeIndex) {
      if (prefab.nodes[nodeIndex].parentIndex != kInvalidScenePrefabIndex ||
          !isValid(node.instantiation.nodes[nodeIndex])) {
        continue;
      }
      (void)node.boundScene->graph().destroyNodeSubtree(
          node.instantiation.nodes[nodeIndex]);
    }
    node.hierarchyPublished = false;
    sceneDirty = true;
  };
  if (node.state == SceneLoadState::Cancelling) {
    if (!cancelSceneDependencies(node, deadline)) {
      return false;
    }
    if (!node.manifest.has_value()) {
      return false;
    }
    std::fill(node.materialPreparationFinished.begin() +
                  node.materialAdmissionCursor,
              node.materialPreparationFinished.end(), 1u);
    node.modelAdmissionCursor = static_cast<uint32_t>(node.models.size());
    node.materialAdmissionCursor = static_cast<uint32_t>(node.materials.size());
    if (!dependenciesTerminal()) {
      return false;
    }
    destroyPublishedHierarchy();
    finishSceneNode(node, SceneLoadState::Cancelled);
    ++stats.cancelled;
    return sceneDirty;
  }
  if (!node.manifest.has_value()) {
    return false;
  }
  const ScenePrefab &prefab = node.manifest->prefab;
  if (!isValid(node.fallbackMaterial)) {
    node.fallbackMaterial = requestMaterial(
        MaterialAssetRequest{
            .debugName = "async_scene_fallback_material",
            .sourceIdentity = node.request.path + "#async_fallback",
        },
        node.priority);
    node.dependencies.emplace_back(node.fallbackMaterial);
  }
  while (node.modelAdmissionCursor < prefab.meshAssets.size() &&
         !deadlineReached()) {
    const uint32_t modelIndex = node.modelAdmissionCursor++;
    const ScenePrefabAssetRef &asset = prefab.meshAssets[modelIndex];
    ModelRequest modelRequest{
        .path = node.request.path,
        .importOptions = node.request.importOptions.assetBuildOptions,
        .debugName = "async_scene_mesh_" + std::to_string(asset.sourceIndex),
        .sceneMeshIndex = asset.sourceIndex,
    };
    Result<ModelAssetHandle, std::string> model =
        modelIndex < node.manifest->meshes.size()
            ? requestModelAsset(
                  modelRequest,
                  std::make_shared<ScenePrefabAdaptedMesh>(
                      std::move(node.manifest->meshes[modelIndex])),
                  node.priority)
            : requestModel(modelRequest, node.priority);
    if (model.hasError()) {
      ++node.optionalFailures;
      continue;
    }
    node.models[modelIndex] = model.value();
    node.dependencies.emplace_back(model.value());
  }
  const TextureCompressionCaps compressionCaps =
      resources_.textureCompressionCaps();
  const auto embeddedTextures = node.manifest->embeddedTextures;
  while (node.materialAdmissionCursor < prefab.materialAssets.size() &&
         !deadlineReached()) {
    const uint32_t materialIndex = node.materialAdmissionCursor++;
    const ScenePrefabAssetRef &asset = prefab.materialAssets[materialIndex];
    const MaterialData workerMaterial =
        materialIndex < node.manifest->materials.size()
            ? node.manifest->materials[materialIndex]
            : MaterialData{};
    const SceneLoadHandle sceneHandle = node.handle;
    const std::string scenePath = node.request.path;
    const AssetPriority priority = node.priority;
    auto task = scheduler_.enqueue(makeAssetCpuJob<SceneMaterialCompletion>(
        priority, AssetWorkClass::Transcode,
        std::min<uint64_t>(estimateSourceBytes(scenePath),
                           64ull * 1024ull * 1024ull),
        "prepare_scene_material_" + std::to_string(asset.sourceIndex),
        SceneMaterialCompletion{.scene = sceneHandle,
                                .materialIndex = materialIndex},
        [workerMaterial, scenePath, sourceMaterialIndex = asset.sourceIndex,
         compressionCaps, embeddedTextures] {
          return prepareImportedMaterial(
              workerMaterial, scenePath, sourceMaterialIndex, compressionCaps,
              embeddedTextures != nullptr
                  ? std::span<const EmbeddedSceneTextureData>(
                        embeddedTextures->data(), embeddedTextures->size())
                  : std::span<const EmbeddedSceneTextureData>(),
              "async_scene");
        },
        [this](SceneMaterialCompletion completion) {
          pushCompletion(std::move(completion));
        }));
    if (task.hasError()) {
      node.materialPreparationFinished[materialIndex] = 1u;
      ++node.optionalFailures;
      continue;
    }
    node.dependencies.emplace_back(task.value());
  }
  const bool allDependenciesTerminal = dependenciesTerminal();
  if (!node.hierarchyPublished) {
    if (completeOnly && !allDependenciesTerminal) {
      return false;
    }
    if (patchBudget == 0u) {
      return false;
    }
    if (targetScene == nullptr) {
      return false;
    }
    if (!isValid(node.publicationTarget)) {
      node.boundScene = targetScene;
    }
    const NodeId parent =
        isValid(targetParent) ? targetParent : targetScene->graph().rootNode();
    const uint32_t nodesBefore = node.structureCursor.nextNode;
    const uint32_t lightsBefore = node.structureCursor.nextLight;
    const uint32_t structureBudget = std::min(patchBudget, 128u);
    auto structure = targetScene->graph().instantiatePrefabStructureStep(
        prefab, parent, node.instantiation, node.structureCursor,
        structureBudget);
    if (structure.hasError()) {
      ++node.requiredFailures;
      finishSceneNode(node, SceneLoadState::Failed, structure.error());
      ++stats.failed;
      return false;
    }
    const uint32_t structureOperations =
        (node.structureCursor.nextNode - nodesBefore) +
        (node.structureCursor.nextLight - lightsBefore);
    patchBudget -= std::min(patchBudget, structureOperations);
    stats.scenePatches += structureOperations;
    sceneDirty = structureOperations != 0u;
    if (!structure.value()) {
      return sceneDirty;
    }
    node.hierarchyPublished = true;
    node.state = SceneLoadState::HierarchyPublished;
  }
  const MaterialNode &fallbackNode = *find(node.fallbackMaterial);
  if (fallbackNode.state == AssetState::Failed ||
      fallbackNode.state == AssetState::Cancelled) {
    ++node.requiredFailures;
    destroyPublishedHierarchy();
    finishSceneNode(node, SceneLoadState::Failed,
                    "AssetSystem: scene fallback material failed");
    ++stats.failed;
    return sceneDirty;
  }
  if (fallbackNode.state != AssetState::Published) {
    node.state = SceneLoadState::PartiallyResident;
    return sceneDirty;
  }
  const MaterialRef fallbackMaterial = fallbackNode.published;
  uint32_t modelIndex = completeOnly ? node.modelMappingCursor : 0u;
  for (; modelIndex < node.models.size() && patchBudget > 0u &&
         !deadlineReached();
       ++modelIndex) {
    const std::optional<ModelRef> model =
        resolvedAsset(find(node.models[modelIndex]));
    if (!model.has_value()) {
      if (completeOnly && allDependenciesTerminal) {
        node.modelFallbackMapped[modelIndex] = 1u;
      }
      continue;
    }
    if (node.modelFallbackMapped[modelIndex] == 0u) {
      const ModelRecord &record = *resources_.tryGet(*model);
      for (uint32_t sourceMaterialIndex = 0u;
           sourceMaterialIndex < record.sourceMaterialToRuntime.size();
           ++sourceMaterialIndex) {
        if (!isValid(record.materialForSource(sourceMaterialIndex))) {
          (void)resources_.setModelMaterialForSource(
              *model, sourceMaterialIndex, fallbackMaterial);
        }
      }
      node.modelFallbackMapped[modelIndex] = 1u;
      --patchBudget;
      ++stats.scenePatches;
    }
  }
  if (completeOnly) {
    node.modelMappingCursor = modelIndex;
  }
  bool bindingsSettled = modelIndex == node.models.size();
  uint32_t materialBindingIndex =
      completeOnly ? node.materialMappingCursor : 0u;
  for (; materialBindingIndex < prefab.renderables.size() && patchBudget > 0u &&
         !deadlineReached();
       ++materialBindingIndex) {
    if (node.renderableMaterialMapped[materialBindingIndex] != 0u) {
      continue;
    }
    const ScenePrefabRenderable &binding =
        prefab.renderables[materialBindingIndex];
    const std::optional<ModelRef> model =
        resolvedAsset(find(node.models[binding.meshAssetIndex]));
    const std::optional<MaterialRef> material =
        resolvedAsset(find(node.materials[binding.materialAssetIndex]));
    if (!model.has_value() || !material.has_value()) {
      if (completeOnly && allDependenciesTerminal) {
        node.renderableMaterialMapped[materialBindingIndex] = 1u;
      }
      continue;
    }
    bool mappedModelSubmesh = false;
    const ModelRecord &record = *resources_.tryGet(*model);
    for (const Submesh &submesh : record.model->submeshes()) {
      mappedModelSubmesh = resources_.setModelMaterialForSource(
                               *model, submesh.materialIndex, *material) ||
                           mappedModelSubmesh;
    }
    if (!mappedModelSubmesh) {
      (void)resources_.setModelMaterialForSource(
          *model, prefab.materialAssets[binding.materialAssetIndex].sourceIndex,
          *material);
    }
    node.renderableMaterialMapped[materialBindingIndex] = 1u;
    --patchBudget;
    ++stats.scenePatches;
  }
  if (completeOnly) {
    node.materialMappingCursor = materialBindingIndex;
  }
  bindingsSettled &= materialBindingIndex == prefab.renderables.size();
  uint32_t renderableIndex = completeOnly ? node.renderableCursor : 0u;
  bool renderablesSettled = true;
  for (; renderableIndex < prefab.renderables.size() && patchBudget > 0u &&
         !deadlineReached();
       ++renderableIndex) {
    const ScenePrefabRenderable &prefabRenderable =
        prefab.renderables[renderableIndex];
    const ModelNode *modelNode =
        find(node.models[prefabRenderable.meshAssetIndex]);
    const std::optional<ModelRef> model = resolvedAsset(modelNode);
    if (!model.has_value()) {
      renderablesSettled &= modelNode == nullptr ||
                            modelNode->state == AssetState::Failed ||
                            modelNode->state == AssetState::Cancelled;
      continue;
    }
    MaterialRef material = fallbackMaterial;
    if (const std::optional<MaterialRef> resolved = resolvedAsset(
            find(node.materials[prefabRenderable.materialAssetIndex]));
        resolved.has_value()) {
      material = *resolved;
    }
    if (!isValid(node.instantiation.renderables[renderableIndex])) {
      auto attached = targetScene->graph().attachPrefabRenderable(
          prefab, renderableIndex, *model, material, node.instantiation);
      if (attached.hasError()) {
        ++node.optionalFailures;
        renderablesSettled = false;
        continue;
      }
      sceneDirty = true;
      --patchBudget;
      ++stats.scenePatches;
      continue;
    }
    MaterialRef current = kInvalidMaterialRef;
    if (targetScene->graph().getRenderableMaterial(
            node.instantiation.renderables[renderableIndex], current) &&
        current.value != material.value &&
        targetScene->graph().setRenderableMaterial(
            node.instantiation.renderables[renderableIndex], material)) {
      sceneDirty = true;
      --patchBudget;
      ++stats.scenePatches;
    }
  }
  if (completeOnly) {
    node.renderableCursor = renderableIndex;
  }
  renderablesSettled &= renderableIndex == prefab.renderables.size();
  if (deadlineReached()) {
    return sceneDirty;
  }
  const SceneAssetCounts modelCounts =
      collectSceneCounts(std::span<const ModelAssetHandle>(node.models.data(),
                                                           node.models.size()));
  const SceneAssetCounts materialCounts =
      collectSceneCounts(std::span<const MaterialAssetHandle>(
          node.materials.data(), node.materials.size()));
  const SceneAssetCounts textureCounts = collectSceneTextureCounts(node);
  if (node.error.empty()) {
    for (const ModelAssetHandle modelHandle : node.models) {
      const ModelNode *model = find(modelHandle);
      if (model != nullptr && model->state == AssetState::Failed) {
        node.error = "model dependency failed: " + model->error;
        break;
      }
    }
  }
  if (node.error.empty()) {
    for (const MaterialAssetHandle materialHandle : node.materials) {
      const MaterialNode *material = find(materialHandle);
      if (material != nullptr && material->state == AssetState::Failed) {
        node.error = "material dependency failed: " + material->error;
        break;
      }
    }
  }
  const bool anyFailures =
      node.optionalFailures != 0u || modelCounts.failed != 0u ||
      materialCounts.failed != 0u || textureCounts.failed != 0u;
  if (node.request.failurePolicy == SceneFailurePolicy::FailOnAnyAsset &&
      anyFailures) {
    ++node.requiredFailures;
    destroyPublishedHierarchy();
    finishSceneNode(node, SceneLoadState::Failed,
                    "AssetSystem: scene dependency failed");
    ++stats.failed;
    return sceneDirty;
  }
  if (allDependenciesTerminal && renderablesSettled && bindingsSettled) {
    finishSceneNode(node, anyFailures ? SceneLoadState::CompleteWithErrors
                                      : SceneLoadState::Complete);
  } else {
    node.state = SceneLoadState::PartiallyResident;
  }
  return sceneDirty;
}

SceneAssetCounts
AssetSystem::collectSceneTextureCounts(const SceneNode &node) const {
  SceneAssetCounts counts{};
  for (const SceneDependency &dependency : node.dependencies)
    if (const auto *texture = std::get_if<TextureAssetHandle>(&dependency)) {
      ++counts.total;
      countSceneAssetState(counts, find(*texture)->state);
    }
  return counts;
}

void AssetSystem::countSceneAssetState(SceneAssetCounts &counts,
                                       AssetState state) {
  static constexpr std::array counters{
      &SceneAssetCounts::queued,       &SceneAssetCounts::queued,
      &SceneAssetCounts::cpuReady,     &SceneAssetCounts::cpuReady,
      &SceneAssetCounts::cpuReady,     &SceneAssetCounts::gpuSubmitted,
      &SceneAssetCounts::gpuSubmitted, &SceneAssetCounts::published,
      &SceneAssetCounts::queued,       &SceneAssetCounts::cancelled,
      &SceneAssetCounts::failed};
  ++(counts.*counters[static_cast<size_t>(state)]);
}

size_t AssetSystem::MaterialAssetKeyHash::operator()(
    const MaterialAssetKey &key) const noexcept {
  size_t seed = static_cast<size_t>(key.descHash);
  for (const uint64_t handle : key.textureHandles)
    hashCombine(seed, handle);
  hashCombine(seed, std::hash<std::string>{}(key.sourceIdentity));
  return seed;
}

size_t
AssetSystem::SceneKeyHash::operator()(const SceneKey &key) const noexcept {
  size_t seed = std::hash<std::string>{}(key.canonicalPath);
  hashCombine(seed, key.importOptionsHash);
  hashCombine(seed, static_cast<uint64_t>(key.publication));
  hashCombine(seed, static_cast<uint64_t>(key.failurePolicy));
  hashCombine(seed,
              (static_cast<uint64_t>(key.publicationTarget.generation) << 32u) |
                  key.publicationTarget.index);
  return seed;
}

float AssetSystem::progressForScene(const SceneNode &node,
                                    const AssetSystem &assets) {
  if (isSceneTerminalState(node.state))
    return 1.0f;
  if (!node.manifest.has_value()) {
    return node.state == SceneLoadState::Requested ? 0.05f : 0.0f;
  }
  const SceneAssetCounts models =
      assets.collectSceneCounts(std::span<const ModelAssetHandle>(node.models));
  const SceneAssetCounts materials = assets.collectSceneCounts(
      std::span<const MaterialAssetHandle>(node.materials));
  const SceneAssetCounts textures = assets.collectSceneTextureCounts(node);
  const auto weightedProgress = [](const SceneAssetCounts &counts) {
    if (counts.total == 0u) {
      return 1.0f;
    }
    const float value =
        static_cast<float>(counts.queued) * 0.10f +
        static_cast<float>(counts.cpuReady) * 0.55f +
        static_cast<float>(counts.gpuSubmitted) * 0.75f +
        static_cast<float>(counts.published + counts.failed + counts.cancelled);
    return value / static_cast<float>(counts.total);
  };
  const std::array counts{models, materials, textures};
  uint32_t classCount = 0u;
  float classProgress = 0.0f;
  for (const SceneAssetCounts &entry : counts) {
    if (entry.total != 0u) {
      classProgress += weightedProgress(entry);
      ++classCount;
    }
  }
  const float assetProgress =
      classCount == 0u ? 1.0f : classProgress / static_cast<float>(classCount);
  const float renderableProgress =
      node.manifest->prefab.renderables.empty()
          ? 1.0f
          : static_cast<float>(std::ranges::count_if(
                node.instantiation.renderables,
                [](RenderableId id) { return isValid(id); })) /
                static_cast<float>(node.manifest->prefab.renderables.size());
  return std::min(0.99f, 0.20f + (node.hierarchyPublished ? 0.10f : 0.0f) +
                             assetProgress * 0.55f +
                             renderableProgress * 0.15f);
}

} // namespace nuri
