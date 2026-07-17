#include "nuri/pch.h"

#include "nuri/resources/async/asset_system.h"

#include "nuri/core/log.h"
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
  const uintmax_t bytes =
      std::filesystem::file_size(std::filesystem::path(std::string(path)),
                                 error);
  if (error) {
    return 4ull * 1024ull * 1024ull;
  }
  return std::min<uint64_t>(
      static_cast<uint64_t>(
          std::min<uintmax_t>(bytes, std::numeric_limits<uint64_t>::max())),
      1ull * 1024ull * 1024ull * 1024ull);
}

[[nodiscard]] uint64_t
estimateAdaptedMeshBytes(const AdaptedSceneMesh &source) noexcept {
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

[[nodiscard]] Result<PreparedModelData, std::string>
prepareModelRequest(const ModelRequest &request) {
  if (request.sceneMeshIndex == std::numeric_limits<uint32_t>::max()) {
    return Model::prepareFromFile(request.path, request.importOptions);
  }
  return Model::prepareSceneMeshFromFile(
      request.path, request.sceneMeshIndex, request.importOptions);
}

template <typename Node>
[[nodiscard]] bool canMaterialize(const Node &node, uint32_t selectedCount,
                                  uint64_t selectedBytes,
                                  const AssetSystemConfig &config) {
  if (!node.prepared.has_value()) {
    return false;
  }
  const uint64_t bytes = node.prepared->uploadBytes();
  if (selectedCount >= config.maxGpuMaterializationsPerFrame) {
    return false;
  }
  if (selectedCount == 0u) {
    return true;
  }
  return selectedBytes < config.maxGpuUploadBytesPerFrame &&
         bytes <= config.maxGpuUploadBytesPerFrame - selectedBytes;
}

[[nodiscard]] uint64_t encodeTextureAssetHandle(
    TextureAssetHandle handle) noexcept {
  return (static_cast<uint64_t>(handle.generation) << 32u) |
         static_cast<uint64_t>(handle.index);
}

template <typename Fn>
void forEachMaterialTextureAsset(
    const MaterialAssetRequest::TextureAssets &textures, Fn &&fn) {
  fn(textures.baseColor);
  fn(textures.metallicRoughness);
  fn(textures.normal);
  fn(textures.occlusion);
  fn(textures.emissive);
  fn(textures.clearcoat);
  fn(textures.clearcoatRoughness);
  fn(textures.clearcoatNormal);
  fn(textures.specular);
  fn(textures.specularColor);
  fn(textures.sheenColor);
  fn(textures.sheenRoughness);
  fn(textures.transmission);
  fn(textures.thickness);
}

void setMaterialTextureAsset(
    MaterialAssetRequest::TextureAssets &textures, size_t slotIndex,
    TextureAssetHandle handle) {
  std::array<TextureAssetHandle *, kMaterialTextureSlotCount> slots{
      &textures.baseColor,
      &textures.metallicRoughness,
      &textures.normal,
      &textures.occlusion,
      &textures.emissive,
      &textures.clearcoat,
      &textures.clearcoatRoughness,
      &textures.clearcoatNormal,
      &textures.specular,
      &textures.specularColor,
      &textures.sheenColor,
      &textures.sheenRoughness,
      &textures.transmission,
      &textures.thickness,
  };
  if (slotIndex < slots.size()) {
    *slots[slotIndex] = handle;
  }
}

template <typename Fn>
void forEachEnvironmentTexture(
    const EnvironmentAssetRequest &request, Fn &&fn) {
  fn(0u, request.cubemap, request.cubemapOptional);
  fn(1u, request.irradiance, request.irradianceOptional);
  fn(2u, request.prefilteredGgx, request.prefilteredGgxOptional);
  fn(3u, request.prefilteredCharlie,
     request.prefilteredCharlieOptional);
  fn(4u, request.brdfLut, request.brdfLutOptional);
}

[[nodiscard]] bool sameEnvironment(const EnvironmentHandles &lhs,
                                   const EnvironmentHandles &rhs) noexcept {
  return lhs.cubemap == rhs.cubemap &&
         lhs.irradiance == rhs.irradiance &&
         lhs.prefilteredGgx == rhs.prefilteredGgx &&
         lhs.prefilteredCharlie == rhs.prefilteredCharlie &&
         lhs.brdfLut == rhs.brdfLut;
}

[[nodiscard]] bool isAssetTerminalState(AssetState state) noexcept {
  return state == AssetState::Published || state == AssetState::Cancelled ||
         state == AssetState::Failed;
}

} // namespace

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
}

AssetSystem::~AssetSystem() {
  scheduler_.requestStop();
  scheduler_.waitIdle();
  for (TextureNode &node : textureNodes_) {
    node.pendingTexture.reset();
    if (isValid(node.published) && resources_.owns(node.published)) {
      resources_.release(node.published);
    }
  }
  for (ModelNode &node : modelNodes_) {
    node.pendingModel.reset();
    if (isValid(node.published) && resources_.owns(node.published)) {
      resources_.release(node.published);
    }
  }
  for (MaterialNode &node : materialNodes_) {
    if (isValid(node.published) && resources_.owns(node.published)) {
      resources_.release(node.published);
    }
  }
}

AssetSystem::MaterialAssetKey
AssetSystem::makeMaterialAssetKey(const MaterialAssetRequest &request) {
  MaterialAssetKey key{};
  key.descHash = hashMaterialDesc(request.desc);
  size_t index = 0u;
  forEachMaterialTextureAsset(
      request.textures, [&key, &index](TextureAssetHandle handle) {
        key.textureHandles[index++] = encodeTextureAssetHandle(handle);
      });
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
  TextureRequest resolved = request;
  resolved.path = canonicalizeResourcePath(request.path);
  const TextureKey key{
      .canonicalPath = resolved.path,
      .optionsHash = hashTextureLoadOptions(resolved.loadOptions),
      .kind = resolved.kind,
  };

  if (const auto ready = resources_.tryAcquireTexture(resolved);
      ready.has_value()) {
    const SlotReservation slot = textureSlots_.acquire();
    if (slot.appended) {
      textureNodes_.emplace_back();
    }
    TextureNode &node = textureNodes_[slot.index];
    node = TextureNode{
        .handle = TextureAssetHandle{slot.index, slot.generation},
        .state = AssetState::Published,
        .priority = priority,
        .request = std::move(resolved),
        .key = key,
        .published = *ready,
    };
    return Result<TextureAssetHandle, std::string>::makeResult(node.handle);
  }

  if (auto it = textureInFlight_.find(key); it != textureInFlight_.end()) {
    if (TextureNode *node = find(it->second)) {
      ++node->subscriberCount;
      if (priority < node->priority) {
        setPriority(node->handle, priority);
      }
      return Result<TextureAssetHandle, std::string>::makeResult(node->handle);
    }
    textureInFlight_.erase(it);
  }

  const SlotReservation slot = textureSlots_.acquire();
  if (slot.appended) {
    textureNodes_.emplace_back();
  }
  TextureNode &node = textureNodes_[slot.index];
  node = TextureNode{
      .handle = TextureAssetHandle{slot.index, slot.generation},
      .state = AssetState::Queued,
      .priority = priority,
      .request = std::move(resolved),
      .key = key,
  };
  textureInFlight_.emplace(node.key, node.handle);
  const TextureAssetHandle handle = node.handle;
  const TextureRequest workerRequest = node.request;
  auto task = scheduler_.enqueue(AssetCpuJob{
      .priority = priority,
      .workClass = AssetWorkClass::Decode,
      .estimatedBytes = estimateSourceBytes(workerRequest.path),
      .debugName = workerRequest.debugName.empty()
                       ? workerRequest.path
                       : workerRequest.debugName,
      .execute =
          [this, handle, workerRequest](std::stop_token stopToken) {
            if (stopToken.stop_requested()) {
              pushCompletion(TextureCpuCompletion{
                  .handle = handle,
                  .cancelled = true,
              });
              return;
            }
            auto result = prepareTextureRequest(workerRequest);
            if (stopToken.stop_requested()) {
              pushCompletion(TextureCpuCompletion{
                  .handle = handle,
                  .cancelled = true,
              });
            } else if (result.hasError()) {
              pushCompletion(TextureCpuCompletion{
                  .handle = handle,
                  .error = result.error(),
              });
            } else {
              pushCompletion(TextureCpuCompletion{
                  .handle = handle,
                  .prepared = std::move(result.value()),
              });
            }
          },
      .onCancelled =
          [this, handle] {
            pushCompletion(TextureCpuCompletion{
                .handle = handle,
                .cancelled = true,
            });
          },
  });
  if (task.hasError()) {
    finishTextureNode(node, AssetState::Failed, task.error());
    node.subscriberCount = 0u;
    releasedTerminalNodes_ = true;
    return Result<TextureAssetHandle, std::string>::makeError(task.error());
  }
  node.cpuTask = task.value();
  node.state = AssetState::CpuRunning;
  return Result<TextureAssetHandle, std::string>::makeResult(handle);
}

Result<ModelAssetHandle, std::string>
AssetSystem::requestModel(const ModelRequest &request,
                          AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  reclaimReleasedNodes();
  if (request.path.empty()) {
    return Result<ModelAssetHandle, std::string>::makeError(
        "AssetSystem::requestModel: path is empty");
  }
  ModelRequest resolved = request;
  resolved.path = canonicalizeResourcePath(request.path);
  const ModelKey key{
      .canonicalPath = resolved.path,
      .importOptionsHash = hashModelImportOptions(resolved.importOptions),
      .sceneMeshIndex = resolved.sceneMeshIndex,
  };
  if (const auto ready = resources_.tryAcquireModel(resolved);
      ready.has_value()) {
    const SlotReservation slot = modelSlots_.acquire();
    if (slot.appended) {
      modelNodes_.emplace_back();
    }
    ModelNode &node = modelNodes_[slot.index];
    node = ModelNode{
        .handle = ModelAssetHandle{slot.index, slot.generation},
        .state = AssetState::Published,
        .priority = priority,
        .request = std::move(resolved),
        .key = key,
        .published = *ready,
    };
    return Result<ModelAssetHandle, std::string>::makeResult(node.handle);
  }
  if (auto it = modelInFlight_.find(key); it != modelInFlight_.end()) {
    if (ModelNode *node = find(it->second)) {
      ++node->subscriberCount;
      if (priority < node->priority) {
        setPriority(node->handle, priority);
      }
      return Result<ModelAssetHandle, std::string>::makeResult(node->handle);
    }
    modelInFlight_.erase(it);
  }

  const SlotReservation slot = modelSlots_.acquire();
  if (slot.appended) {
    modelNodes_.emplace_back();
  }
  ModelNode &node = modelNodes_[slot.index];
  node = ModelNode{
      .handle = ModelAssetHandle{slot.index, slot.generation},
      .state = AssetState::Queued,
      .priority = priority,
      .request = std::move(resolved),
      .key = key,
  };
  modelInFlight_.emplace(node.key, node.handle);
  const ModelAssetHandle handle = node.handle;
  const ModelRequest workerRequest = node.request;
  auto task = scheduler_.enqueue(AssetCpuJob{
      .priority = priority,
      .workClass = AssetWorkClass::Cook,
      .estimatedBytes = estimateSourceBytes(workerRequest.path),
      .debugName = workerRequest.debugName.empty()
                       ? workerRequest.path
                       : workerRequest.debugName,
      .execute =
          [this, handle, workerRequest](std::stop_token stopToken) {
            if (stopToken.stop_requested()) {
              pushCompletion(ModelCpuCompletion{
                  .handle = handle,
                  .cancelled = true,
              });
              return;
            }
            auto result = prepareModelRequest(workerRequest);
            if (stopToken.stop_requested()) {
              pushCompletion(ModelCpuCompletion{
                  .handle = handle,
                  .cancelled = true,
              });
            } else if (result.hasError()) {
              pushCompletion(ModelCpuCompletion{
                  .handle = handle,
                  .error = result.error(),
              });
            } else {
              pushCompletion(ModelCpuCompletion{
                  .handle = handle,
                  .prepared = std::move(result.value()),
              });
            }
          },
      .onCancelled =
          [this, handle] {
            pushCompletion(ModelCpuCompletion{
                .handle = handle,
                .cancelled = true,
            });
          },
  });
  if (task.hasError()) {
    finishModelNode(node, AssetState::Failed, task.error());
    node.subscriberCount = 0u;
    releasedTerminalNodes_ = true;
    return Result<ModelAssetHandle, std::string>::makeError(task.error());
  }
  node.cpuTask = task.value();
  node.state = AssetState::CpuRunning;
  return Result<ModelAssetHandle, std::string>::makeResult(handle);
}

Result<ModelAssetHandle, std::string>
AssetSystem::requestAdaptedModel(const ModelRequest &request,
                                 AdaptedSceneMesh source,
                                 AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  reclaimReleasedNodes();
  if (request.path.empty()) {
    return Result<ModelAssetHandle, std::string>::makeError(
        "AssetSystem::requestAdaptedModel: path is empty");
  }
  ModelRequest resolved = request;
  resolved.path = canonicalizeResourcePath(request.path);
  const ModelKey key{
      .canonicalPath = resolved.path,
      .importOptionsHash = hashModelImportOptions(resolved.importOptions),
      .sceneMeshIndex = resolved.sceneMeshIndex,
  };
  if (const auto ready = resources_.tryAcquireModel(resolved);
      ready.has_value()) {
    const SlotReservation slot = modelSlots_.acquire();
    if (slot.appended) {
      modelNodes_.emplace_back();
    }
    ModelNode &node = modelNodes_[slot.index];
    node = ModelNode{
        .handle = ModelAssetHandle{slot.index, slot.generation},
        .state = AssetState::Published,
        .priority = priority,
        .request = std::move(resolved),
        .key = key,
        .published = *ready,
    };
    return Result<ModelAssetHandle, std::string>::makeResult(node.handle);
  }
  if (auto it = modelInFlight_.find(key); it != modelInFlight_.end()) {
    if (ModelNode *node = find(it->second)) {
      ++node->subscriberCount;
      if (priority < node->priority) {
        setPriority(node->handle, priority);
      }
      return Result<ModelAssetHandle, std::string>::makeResult(node->handle);
    }
    modelInFlight_.erase(it);
  }

  const uint64_t estimatedBytes = estimateAdaptedMeshBytes(source);
  auto sourceOwner =
      std::make_shared<AdaptedSceneMesh>(std::move(source));
  const SlotReservation slot = modelSlots_.acquire();
  if (slot.appended) {
    modelNodes_.emplace_back();
  }
  ModelNode &node = modelNodes_[slot.index];
  node = ModelNode{
      .handle = ModelAssetHandle{slot.index, slot.generation},
      .state = AssetState::Queued,
      .priority = priority,
      .request = std::move(resolved),
      .key = key,
  };
  modelInFlight_.emplace(node.key, node.handle);
  const ModelAssetHandle handle = node.handle;
  const ModelRequest workerRequest = node.request;
  auto task = scheduler_.enqueue(AssetCpuJob{
      .priority = priority,
      .workClass = AssetWorkClass::Cook,
      .estimatedBytes = estimatedBytes,
      .debugName = workerRequest.debugName.empty()
                       ? workerRequest.path
                       : workerRequest.debugName,
      .execute =
          [this, handle, workerRequest,
           sourceOwner](std::stop_token stopToken) {
            if (stopToken.stop_requested()) {
              pushCompletion(ModelCpuCompletion{
                  .handle = handle,
                  .cancelled = true,
              });
              return;
            }
            auto cooked = detail::cookAdaptedSceneMesh(
                std::move(*sourceOwner), workerRequest.importOptions,
                std::pmr::get_default_resource());
            if (cooked.hasError()) {
              pushCompletion(ModelCpuCompletion{
                  .handle = handle,
                  .error = cooked.error(),
              });
              return;
            }
            auto prepared = Model::prepare(std::move(cooked.value()));
            if (stopToken.stop_requested()) {
              pushCompletion(ModelCpuCompletion{
                  .handle = handle,
                  .cancelled = true,
              });
            } else if (prepared.hasError()) {
              pushCompletion(ModelCpuCompletion{
                  .handle = handle,
                  .error = prepared.error(),
              });
            } else {
              pushCompletion(ModelCpuCompletion{
                  .handle = handle,
                  .prepared = std::move(prepared.value()),
              });
            }
          },
      .onCancelled =
          [this, handle] {
            pushCompletion(ModelCpuCompletion{
                .handle = handle,
                .cancelled = true,
            });
          },
  });
  if (task.hasError()) {
    finishModelNode(node, AssetState::Failed, task.error());
    node.subscriberCount = 0u;
    releasedTerminalNodes_ = true;
    return Result<ModelAssetHandle, std::string>::makeError(task.error());
  }
  node.cpuTask = task.value();
  node.state = AssetState::CpuRunning;
  return Result<ModelAssetHandle, std::string>::makeResult(handle);
}

Result<MaterialAssetHandle, std::string>
AssetSystem::requestMaterial(const MaterialAssetRequest &request,
                             AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  reclaimReleasedNodes();
  const MaterialAssetKey key = makeMaterialAssetKey(request);
  if (auto it = materialInFlight_.find(key); it != materialInFlight_.end()) {
    if (MaterialNode *node = find(it->second)) {
      ++node->subscriberCount;
      if (priority < node->priority) {
        setPriority(node->handle, priority);
      }
      return Result<MaterialAssetHandle, std::string>::makeResult(node->handle);
    }
    materialInFlight_.erase(it);
  }

  const SlotReservation slot = materialSlots_.acquire();
  if (slot.appended) {
    materialNodes_.emplace_back();
  }
  MaterialNode &node = materialNodes_[slot.index];
  node = MaterialNode{
      .handle = MaterialAssetHandle{slot.index, slot.generation},
      .state = AssetState::CpuReady,
      .priority = priority,
      .request = request,
      .key = key,
  };
  materialInFlight_.emplace(node.key, node.handle);
  forEachMaterialTextureAsset(
      node.request.textures, [this, priority](TextureAssetHandle handle) {
        if (isValidAssetHandle(handle)) {
          setPriority(handle, priority);
        }
      });
  return Result<MaterialAssetHandle, std::string>::makeResult(node.handle);
}

Result<EnvironmentAssetHandle, std::string>
AssetSystem::requestEnvironment(const EnvironmentAssetRequest &request) {
  std::lock_guard stateLock(stateMutex_);
  reclaimReleasedNodes();
  const SlotReservation slot = environmentSlots_.acquire();
  if (slot.appended) {
    environmentNodes_.emplace_back();
  }
  EnvironmentNode &node = environmentNodes_[slot.index];
  node = EnvironmentNode{
      .handle = EnvironmentAssetHandle{slot.index, slot.generation},
      .state = AssetState::CpuReady,
      .request = request,
  };

  std::string requestError;
  forEachEnvironmentTexture(
      node.request,
      [this, &node, &requestError](
          size_t index, const std::optional<TextureRequest> &texture,
          bool optional) {
        if (!requestError.empty() || !texture.has_value()) {
          return;
        }
        auto requested = requestTexture(*texture, node.request.priority);
        if (requested.hasError()) {
          if (!optional) {
            requestError = requested.error();
          }
          return;
        }
        node.textures[index] = requested.value();
      });
  if (!requestError.empty()) {
    for (const TextureAssetHandle texture : node.textures) {
      if (isValidAssetHandle(texture)) {
        cancel(texture);
      }
    }
    node.state = AssetState::Failed;
    node.subscriberCount = 0u;
    releasedTerminalNodes_ = true;
    node.error = requestError;
    return Result<EnvironmentAssetHandle, std::string>::makeError(
        requestError);
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
  const SceneKey key{
      .canonicalPath = resolved.path,
      .importOptionsHash =
          hashModelImportOptions(resolved.importOptions.assetBuildOptions),
      .publication = resolved.publication,
      .failurePolicy = resolved.failurePolicy,
  };
  if (auto it = sceneInFlight_.find(key); it != sceneInFlight_.end()) {
    if (SceneNode *node = find(it->second)) {
      ++node->subscriberCount;
      if (resolved.priority < node->request.priority) {
        setPriority(node->handle, resolved.priority);
      }
      return Result<SceneLoadHandle, std::string>::makeResult(node->handle);
    }
    sceneInFlight_.erase(it);
  }

  const SlotReservation slot = sceneSlots_.acquire();
  if (slot.appended) {
    sceneNodes_.emplace_back();
  }
  SceneNode &node = sceneNodes_[slot.index];
  node = SceneNode{
      .handle = SceneLoadHandle{slot.index, slot.generation},
      .state = SceneLoadState::Requested,
      .request = std::move(resolved),
      .key = key,
  };
  sceneInFlight_.emplace(node.key, node.handle);

  const SceneLoadHandle handle = node.handle;
  const std::string workerPath = node.request.path;
  const SceneImportOptions workerOptions = node.request.importOptions;
  auto task = scheduler_.enqueue(AssetCpuJob{
      .priority = node.request.priority,
      .workClass = AssetWorkClass::Metadata,
      .estimatedBytes = estimateSourceBytes(workerPath),
      .debugName = node.request.debugName.empty() ? workerPath
                                                  : node.request.debugName,
      .execute =
          [this, handle, workerPath,
           workerOptions](std::stop_token stopToken) {
            if (stopToken.stop_requested()) {
              pushCompletion(SceneManifestCompletion{
                  .handle = handle,
                  .cancelled = true,
              });
              return;
            }
            auto result = prepareSceneManifest(workerPath, workerOptions);
            if (stopToken.stop_requested()) {
              pushCompletion(SceneManifestCompletion{
                  .handle = handle,
                  .cancelled = true,
              });
            } else if (result.hasError()) {
              pushCompletion(SceneManifestCompletion{
                  .handle = handle,
                  .error = result.error(),
              });
            } else {
              pushCompletion(SceneManifestCompletion{
                  .handle = handle,
                  .manifest = std::move(result.value()),
              });
            }
          },
      .onCancelled =
          [this, handle] {
            pushCompletion(SceneManifestCompletion{
                .handle = handle,
                .cancelled = true,
            });
          },
  });
  if (task.hasError()) {
    finishSceneNode(node, SceneLoadState::Failed, task.error());
    node.subscriberCount = 0u;
    releasedTerminalNodes_ = true;
    return Result<SceneLoadHandle, std::string>::makeError(task.error());
  }
  node.manifestTask = task.value();
  return Result<SceneLoadHandle, std::string>::makeResult(handle);
}

AssetLoadSnapshot AssetSystem::query(TextureAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const TextureNode *node = find(handle);
  if (node == nullptr) {
    return AssetLoadSnapshot{
        .state = AssetState::Failed,
        .error = "invalid or stale texture asset handle",
    };
  }
  return AssetLoadSnapshot{
      .state = node->state,
      .priority = node->priority,
      .progress = progressForState(node->state),
      .cpuPayloadBytes = node->cpuPayloadBytes,
      .error = node->error,
  };
}

AssetLoadSnapshot AssetSystem::query(ModelAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const ModelNode *node = find(handle);
  if (node == nullptr) {
    return AssetLoadSnapshot{
        .state = AssetState::Failed,
        .error = "invalid or stale model asset handle",
    };
  }
  return AssetLoadSnapshot{
      .state = node->state,
      .priority = node->priority,
      .progress = progressForState(node->state),
      .cpuPayloadBytes = node->cpuPayloadBytes,
      .error = node->error,
  };
}

AssetLoadSnapshot AssetSystem::query(MaterialAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const MaterialNode *node = find(handle);
  if (node == nullptr) {
    return AssetLoadSnapshot{
        .state = AssetState::Failed,
        .error = "invalid or stale material asset handle",
    };
  }
  return AssetLoadSnapshot{
      .state = node->state,
      .priority = node->priority,
      .progress = progressForState(node->state),
      .error = node->error,
  };
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
    if (!isValidAssetHandle(texture)) {
      continue;
    }
    ++dependencyCount;
    progress += progressForState(query(texture).state);
  }
  if (dependencyCount != 0u) {
    progress /= static_cast<float>(dependencyCount);
  } else {
    progress = progressForState(node->state);
  }
  return AssetLoadSnapshot{
      .state = node->state,
      .priority = node->request.priority,
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
  bool cancellationPendingGpuRetirement = false;
  const auto inspectCancellation = [this, &cancellationPendingGpuRetirement](
                                       auto handleValue) {
    const auto *dependency = find(handleValue);
    cancellationPendingGpuRetirement |=
        dependency != nullptr &&
        dependency->state == AssetState::CancelRequested;
  };
  for (const ModelAssetHandle model : node->models) {
    if (isValidAssetHandle(model)) {
      inspectCancellation(model);
    }
  }
  for (const MaterialAssetHandle material : node->materials) {
    if (isValidAssetHandle(material)) {
      inspectCancellation(material);
    }
  }
  for (const auto &subscriptions : node->textureSubscriptions) {
    for (const TextureAssetHandle texture : subscriptions) {
      if (isValidAssetHandle(texture)) {
        inspectCancellation(texture);
      }
    }
  }

  return SceneLoadSnapshot{
      .state = node->state,
      .priority = node->request.priority,
      .progress = node->progress,
      .sourceDiscoveryComplete = node->manifest.has_value(),
      .hierarchyPublished = node->hierarchyPublished,
      .cancellationPendingGpuRetirement = cancellationPendingGpuRetirement,
      .models = collectSceneCounts(
          *this, std::span<const ModelAssetHandle>(node->models.data(),
                                                   node->models.size())),
      .materials = collectSceneCounts(
          *this, std::span<const MaterialAssetHandle>(node->materials.data(),
                                                      node->materials.size())),
      .textures = collectSceneTextureCounts(*this, *node),
      .publishedRenderables = publishedRenderables,
      .totalRenderables =
          node->manifest.has_value()
              ? static_cast<uint32_t>(node->manifest->prefab.renderables.size())
              : 0u,
      .requiredFailures = node->requiredFailures,
      .optionalFailures =
          node->optionalFailures +
          collectSceneCounts(
              *this, std::span<const ModelAssetHandle>(node->models.data(),
                                                       node->models.size()))
              .failed +
          collectSceneCounts(
              *this,
              std::span<const MaterialAssetHandle>(node->materials.data(),
                                                   node->materials.size()))
              .failed +
          collectSceneTextureCounts(*this, *node).failed,
      .cpuPayloadBytes = node->cpuPayloadBytes,
      .error = node->error,
  };
}

std::optional<TextureRef>
AssetSystem::tryResolve(TextureAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const TextureNode *node = find(handle);
  return node != nullptr && node->state == AssetState::Published &&
                 isValid(node->published)
             ? std::optional<TextureRef>(node->published)
             : std::nullopt;
}

std::optional<ModelRef> AssetSystem::tryResolve(ModelAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const ModelNode *node = find(handle);
  return node != nullptr && node->state == AssetState::Published &&
                 isValid(node->published)
             ? std::optional<ModelRef>(node->published)
             : std::nullopt;
}

std::optional<MaterialRef>
AssetSystem::tryResolve(MaterialAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const MaterialNode *node = find(handle);
  return node != nullptr && node->state == AssetState::Published &&
                 isValid(node->published)
             ? std::optional<MaterialRef>(node->published)
             : std::nullopt;
}

std::optional<EnvironmentHandles>
AssetSystem::tryResolve(EnvironmentAssetHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const EnvironmentNode *node = find(handle);
  return node != nullptr && node->state == AssetState::Published
             ? std::optional<EnvironmentHandles>(node->published)
             : std::nullopt;
}

const ScenePrefab *
AssetSystem::tryGetScenePrefab(SceneLoadHandle handle) const {
  std::lock_guard stateLock(stateMutex_);
  const SceneNode *node = find(handle);
  return node != nullptr && node->manifest.has_value()
             ? &node->manifest->prefab
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
    assets.models.push_back(
        dependency != nullptr && dependency->state == AssetState::Published
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

void AssetSystem::setPriority(TextureAssetHandle handle,
                              AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  if (TextureNode *node = find(handle)) {
    if (priority < node->priority) {
      node->priority = priority;
      (void)scheduler_.setPriority(node->cpuTask, priority);
    }
  }
}

void AssetSystem::setPriority(ModelAssetHandle handle,
                              AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  if (ModelNode *node = find(handle)) {
    if (priority < node->priority) {
      node->priority = priority;
      (void)scheduler_.setPriority(node->cpuTask, priority);
    }
  }
}

void AssetSystem::setPriority(MaterialAssetHandle handle,
                              AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  if (MaterialNode *node = find(handle)) {
    if (priority >= node->priority) {
      return;
    }
    node->priority = priority;
    forEachMaterialTextureAsset(
        node->request.textures, [this, priority](TextureAssetHandle texture) {
          if (isValidAssetHandle(texture)) {
            setPriority(texture, priority);
          }
        });
  }
}

void AssetSystem::setPriority(EnvironmentAssetHandle handle,
                              AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  EnvironmentNode *node = find(handle);
  if (node == nullptr || priority >= node->request.priority) {
    return;
  }
  node->request.priority = priority;
  for (const TextureAssetHandle texture : node->textures) {
    if (isValidAssetHandle(texture)) {
      setPriority(texture, priority);
    }
  }
}

void AssetSystem::setPriority(SceneLoadHandle handle, AssetPriority priority) {
  std::lock_guard stateLock(stateMutex_);
  SceneNode *node = find(handle);
  if (node == nullptr || priority >= node->request.priority) {
    return;
  }
  node->request.priority = priority;
  (void)scheduler_.setPriority(node->manifestTask, priority);
  for (const AssetCpuTaskHandle task : node->materialTasks) {
    (void)scheduler_.setPriority(task, priority);
  }
  for (const ModelAssetHandle model : node->models) {
    setPriority(model, priority);
  }
  for (const MaterialAssetHandle material : node->materials) {
    setPriority(material, priority);
  }
  for (const auto &subscriptions : node->textureSubscriptions) {
    for (const TextureAssetHandle texture : subscriptions) {
      setPriority(texture, priority);
    }
  }
  if (isValidAssetHandle(node->fallbackMaterial)) {
    setPriority(node->fallbackMaterial, priority);
  }
}

void AssetSystem::cancel(TextureAssetHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  TextureNode *node = find(handle);
  if (node == nullptr || node->subscriberCount == 0u) {
    return;
  }
  if (node->subscriberCount > 1u) {
    --node->subscriberCount;
    return;
  }
  node->subscriberCount = 0u;
  if (node->state == AssetState::Cancelled ||
      node->state == AssetState::Failed) {
    releasedTerminalNodes_ = true;
    return;
  }
  if (node->state == AssetState::Published) {
    if (isValid(node->published) && resources_.owns(node->published)) {
      resources_.release(node->published);
    }
    node->published = kInvalidTextureRef;
    finishTextureNode(*node, AssetState::Cancelled);
    return;
  }
  if (node->state == AssetState::CpuReady) {
    finishTextureNode(*node, AssetState::Cancelled);
    return;
  }
  node->state = AssetState::CancelRequested;
  (void)scheduler_.cancel(node->cpuTask);
}

void AssetSystem::cancel(ModelAssetHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  ModelNode *node = find(handle);
  if (node == nullptr || node->subscriberCount == 0u) {
    return;
  }
  if (node->subscriberCount > 1u) {
    --node->subscriberCount;
    return;
  }
  node->subscriberCount = 0u;
  if (node->state == AssetState::Cancelled ||
      node->state == AssetState::Failed) {
    releasedTerminalNodes_ = true;
    return;
  }
  if (node->state == AssetState::Published) {
    if (isValid(node->published) && resources_.owns(node->published)) {
      resources_.release(node->published);
    }
    node->published = kInvalidModelRef;
    finishModelNode(*node, AssetState::Cancelled);
    return;
  }
  if (node->state == AssetState::CpuReady) {
    finishModelNode(*node, AssetState::Cancelled);
    return;
  }
  node->state = AssetState::CancelRequested;
  (void)scheduler_.cancel(node->cpuTask);
}

void AssetSystem::cancel(MaterialAssetHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  MaterialNode *node = find(handle);
  if (node == nullptr || node->subscriberCount == 0u) {
    return;
  }
  if (node->subscriberCount > 1u) {
    --node->subscriberCount;
    return;
  }
  node->subscriberCount = 0u;
  if (node->state == AssetState::Cancelled ||
      node->state == AssetState::Failed) {
    releasedTerminalNodes_ = true;
    return;
  }
  if (node->state == AssetState::Published &&
      isValid(node->published) && resources_.owns(node->published)) {
    resources_.release(node->published);
    node->published = kInvalidMaterialRef;
  }
  finishMaterialNode(*node, AssetState::Cancelled);
}

void AssetSystem::cancel(EnvironmentAssetHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  EnvironmentNode *node = find(handle);
  if (node == nullptr || node->subscriberCount == 0u) {
    return;
  }
  if (node->subscriberCount > 1u) {
    --node->subscriberCount;
    return;
  }
  node->subscriberCount = 0u;
  if (node->state == AssetState::Cancelled) {
    releasedTerminalNodes_ = true;
    return;
  }
  node->state = AssetState::CancelRequested;
  if (!node->dependenciesCancelled) {
    node->dependenciesCancelled = true;
    for (const TextureAssetHandle texture : node->textures) {
      if (isValidAssetHandle(texture)) {
        cancel(texture);
      }
    }
  }
}

void AssetSystem::cancel(SceneLoadHandle handle) {
  std::lock_guard stateLock(stateMutex_);
  SceneNode *node = find(handle);
  if (node == nullptr || node->subscriberCount == 0u) {
    return;
  }
  if (node->subscriberCount > 1u) {
    --node->subscriberCount;
    return;
  }
  node->subscriberCount = 0u;
  if (node->state == SceneLoadState::Cancelled) {
    releasedTerminalNodes_ = true;
    return;
  }
  if (node->state == SceneLoadState::Failed) {
    cancelSceneDependencies(*node);
    releasedTerminalNodes_ = true;
    return;
  }
  node->state = SceneLoadState::Cancelling;
  cancelSceneDependencies(*node);
}

Result<AssetPublicationStats, std::string>
AssetSystem::prepareFrame(AssetPublicationContext context) {
  std::lock_guard stateLock(stateMutex_);
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  resources_.beginPublicationBatch();
  AssetPublicationStats stats{};
  std::vector<CpuCompletion> completions = takeCompletions();
  stats.cpuCompletions = static_cast<uint32_t>(completions.size());
  for (CpuCompletion &completion : completions) {
    std::visit(
        [this, &stats](auto &typed) {
          using Completion = std::decay_t<decltype(typed)>;
          if constexpr (std::is_same_v<Completion, TextureCpuCompletion>) {
            TextureNode *node = find(typed.handle);
            if (node == nullptr) {
              return;
            }
            if (typed.cancelled ||
                node->state == AssetState::CancelRequested) {
              finishTextureNode(*node, AssetState::Cancelled);
              ++stats.cancelled;
            } else if (!typed.error.empty()) {
              finishTextureNode(*node, AssetState::Failed,
                                std::move(typed.error));
              ++stats.failed;
            } else if (typed.prepared.has_value()) {
              node->cpuPayloadBytes = typed.prepared->uploadBytes();
              node->prepared = std::move(typed.prepared);
              node->state = AssetState::CpuReady;
            }
          } else if constexpr (std::is_same_v<Completion,
                                               ModelCpuCompletion>) {
            ModelNode *node = find(typed.handle);
            if (node == nullptr) {
              return;
            }
            if (typed.cancelled ||
                node->state == AssetState::CancelRequested) {
              finishModelNode(*node, AssetState::Cancelled);
              ++stats.cancelled;
            } else if (!typed.error.empty()) {
              finishModelNode(*node, AssetState::Failed,
                              std::move(typed.error));
              ++stats.failed;
            } else if (typed.prepared.has_value()) {
              node->cpuPayloadBytes = typed.prepared->uploadBytes();
              node->prepared = std::move(typed.prepared);
              node->state = AssetState::CpuReady;
            }
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
            if (!typed.manifest.has_value()) {
              ++node->requiredFailures;
              finishSceneNode(
                  *node, SceneLoadState::Failed,
                  "AssetSystem: scene manifest completion has no payload");
              ++stats.failed;
              return;
            }

            node->manifest = std::move(typed.manifest);
            const ScenePrefab &prefab = node->manifest->prefab;
            node->cpuPayloadBytes = estimateSourceBytes(node->request.path);
            node->models.resize(prefab.meshAssets.size());
            node->materials.resize(prefab.materialAssets.size());
            node->materialTasks.resize(prefab.materialAssets.size());
            node->materialPreparationFinished.resize(
                prefab.materialAssets.size(), 0u);
            node->textureSubscriptions.resize(prefab.materialAssets.size());
            node->modelFallbackMapped.resize(prefab.meshAssets.size(), 0u);
            node->modelMaterialMapped.resize(prefab.meshAssets.size());
            for (auto &mapped : node->modelMaterialMapped) {
              mapped.resize(prefab.materialAssets.size(), 0u);
            }

            auto fallback = requestMaterial(
                MaterialAssetRequest{
                    .debugName = "async_scene_fallback_material",
                    .sourceIdentity = node->request.path + "#async_fallback",
                },
                node->request.priority);
            if (fallback.hasError()) {
              ++node->requiredFailures;
              finishSceneNode(*node, SceneLoadState::Failed,
                              fallback.error());
              ++stats.failed;
              return;
            }
            node->fallbackMaterial = fallback.value();

            for (uint32_t modelIndex = 0u;
                 modelIndex < prefab.meshAssets.size(); ++modelIndex) {
              const ScenePrefabMeshAssetRef &asset =
                  prefab.meshAssets[modelIndex];
              ModelRequest modelRequest{
                  .path = node->request.path,
                  .importOptions =
                      node->request.importOptions.assetBuildOptions,
                  .debugName =
                      "async_scene_mesh_" +
                      std::to_string(asset.sourceSceneMeshIndex),
                  .sceneMeshIndex = asset.sourceSceneMeshIndex,
              };
              Result<ModelAssetHandle, std::string> model =
                  modelIndex < node->manifest->meshes.size()
                      ? requestAdaptedModel(
                            modelRequest,
                            std::move(node->manifest->meshes[modelIndex]),
                            node->request.priority)
                      : requestModel(modelRequest, node->request.priority);
              if (model.hasError()) {
                ++node->optionalFailures;
                continue;
              }
              node->models[modelIndex] = model.value();
            }

            const TextureCompressionCaps compressionCaps =
                resources_.textureCompressionCaps();
            const auto embeddedTextures =
                node->manifest->embeddedTextures;
            for (uint32_t materialIndex = 0u;
                 materialIndex < prefab.materialAssets.size();
                 ++materialIndex) {
              const ScenePrefabMaterialAssetRef &asset =
                  prefab.materialAssets[materialIndex];
              const MaterialData workerMaterial =
                  materialIndex < node->manifest->materials.size()
                      ? node->manifest->materials[materialIndex]
                      : MaterialData{};
              const SceneLoadHandle sceneHandle = node->handle;
              const std::string scenePath = node->request.path;
              const AssetPriority priority = node->request.priority;
              auto task = scheduler_.enqueue(AssetCpuJob{
                  .priority = priority,
                  .workClass = AssetWorkClass::Transcode,
                  .estimatedBytes = std::min<uint64_t>(
                      estimateSourceBytes(scenePath),
                      64ull * 1024ull * 1024ull),
                  .debugName =
                      "prepare_scene_material_" +
                      std::to_string(asset.sourceMaterialIndex),
                  .execute =
                      [this, sceneHandle, materialIndex, workerMaterial,
                       scenePath, sourceMaterialIndex = asset.sourceMaterialIndex,
                       compressionCaps,
                       embeddedTextures](std::stop_token stopToken) {
                        if (stopToken.stop_requested()) {
                          pushCompletion(SceneMaterialCompletion{
                              .scene = sceneHandle,
                              .materialIndex = materialIndex,
                              .cancelled = true,
                          });
                          return;
                        }
                        auto result = prepareImportedMaterial(
                            workerMaterial, scenePath, sourceMaterialIndex,
                            compressionCaps,
                            embeddedTextures != nullptr
                                ? std::span<const EmbeddedSceneTextureData>(
                                      embeddedTextures->data(),
                                      embeddedTextures->size())
                                : std::span<
                                      const EmbeddedSceneTextureData>(),
                            "async_scene");
                        if (stopToken.stop_requested()) {
                          pushCompletion(SceneMaterialCompletion{
                              .scene = sceneHandle,
                              .materialIndex = materialIndex,
                              .cancelled = true,
                          });
                        } else if (result.hasError()) {
                          pushCompletion(SceneMaterialCompletion{
                              .scene = sceneHandle,
                              .materialIndex = materialIndex,
                              .error = result.error(),
                          });
                        } else {
                          pushCompletion(SceneMaterialCompletion{
                              .scene = sceneHandle,
                              .materialIndex = materialIndex,
                              .prepared = std::move(result.value()),
                          });
                        }
                      },
                  .onCancelled =
                      [this, sceneHandle, materialIndex] {
                        pushCompletion(SceneMaterialCompletion{
                            .scene = sceneHandle,
                            .materialIndex = materialIndex,
                            .cancelled = true,
                        });
                      },
              });
              if (task.hasError()) {
                node->materialPreparationFinished[materialIndex] = 1u;
                ++node->optionalFailures;
                continue;
              }
              node->materialTasks[materialIndex] = task.value();
            }
            node->state = SceneLoadState::ManifestReady;
          } else if constexpr (std::is_same_v<Completion,
                                               SceneMaterialCompletion>) {
            SceneNode *node = find(typed.scene);
            if (node == nullptr ||
                typed.materialIndex >=
                    node->materialPreparationFinished.size()) {
              return;
            }
            node->materialPreparationFinished[typed.materialIndex] = 1u;
            if (typed.cancelled ||
                node->state == SceneLoadState::Cancelling) {
              return;
            }
            if (!typed.error.empty() || !typed.prepared.has_value()) {
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
            for (size_t slotIndex = 0u;
                 slotIndex < prepared.textures.size(); ++slotIndex) {
              if (!prepared.textures[slotIndex].has_value()) {
                continue;
              }
              auto texture = requestTexture(*prepared.textures[slotIndex],
                                            node->request.priority);
              if (texture.hasError()) {
                ++node->optionalFailures;
                continue;
              }
              setMaterialTextureAsset(materialRequest.textures, slotIndex,
                                      texture.value());
              node->textureSubscriptions[typed.materialIndex].push_back(
                  texture.value());
            }

            auto material =
                requestMaterial(materialRequest, node->request.priority);
            if (material.hasError()) {
              ++node->optionalFailures;
              return;
            }
            node->materials[typed.materialIndex] = material.value();
          }
        },
        completion);
  }

  uint32_t selectedCount = 0u;
  uint64_t selectedBytes = 0u;
  std::vector<TextureNode *> submittedTextures{};
  std::vector<ModelNode *> submittedModels{};
  for (TextureNode &node : textureNodes_) {
    if (node.state != AssetState::CpuReady ||
        !canMaterialize(node, selectedCount, selectedBytes, config_)) {
      continue;
    }
    node.state = AssetState::GpuQueued;
    const uint64_t bytes = node.prepared->uploadBytes();
    auto textureResult =
        Texture::createPrepared(gpu_, std::move(*node.prepared));
    node.prepared.reset();
    if (textureResult.hasError()) {
      finishTextureNode(node, AssetState::Failed, textureResult.error());
      ++stats.failed;
      continue;
    }
    node.pendingTexture = std::move(textureResult.value());
    submittedTextures.push_back(&node);
    ++selectedCount;
    selectedBytes += bytes;
  }
  for (ModelNode &node : modelNodes_) {
    if (node.state != AssetState::CpuReady ||
        !canMaterialize(node, selectedCount, selectedBytes, config_)) {
      continue;
    }
    node.state = AssetState::GpuQueued;
    const uint64_t bytes = node.prepared->uploadBytes();
    auto modelResult = Model::createPrepared(
        gpu_, std::move(*node.prepared), node.request.debugName);
    node.prepared.reset();
    if (modelResult.hasError()) {
      finishModelNode(node, AssetState::Failed, modelResult.error());
      ++stats.failed;
      continue;
    }
    node.pendingModel = std::move(modelResult.value());
    submittedModels.push_back(&node);
    ++selectedCount;
    selectedBytes += bytes;
  }

  if (!submittedTextures.empty() || !submittedModels.empty()) {
    auto uploadResult = gpu_.submitPendingUploads();
    if (uploadResult.hasError()) {
      for (TextureNode *node : submittedTextures) {
        node->pendingTexture.reset();
        finishTextureNode(*node, AssetState::Failed, uploadResult.error());
        ++stats.failed;
      }
      for (ModelNode *node : submittedModels) {
        node->pendingModel.reset();
        finishModelNode(*node, AssetState::Failed, uploadResult.error());
        ++stats.failed;
      }
    } else {
      for (TextureNode *node : submittedTextures) {
        node->upload = uploadResult.value();
        node->state = AssetState::GpuSubmitted;
      }
      for (ModelNode *node : submittedModels) {
        node->upload = uploadResult.value();
        node->state = AssetState::GpuSubmitted;
      }
      stats.gpuMaterialized =
          static_cast<uint32_t>(submittedTextures.size() +
                                submittedModels.size());
      stats.uploadBytes = selectedBytes;
    }
  }

  for (TextureNode &node : textureNodes_) {
    const bool cancelAfterSubmit =
        node.state == AssetState::CancelRequested &&
        node.pendingTexture != nullptr;
    if ((node.state != AssetState::GpuSubmitted && !cancelAfterSubmit) ||
        !gpu_.isSubmissionComplete(node.upload)) {
      continue;
    }
    auto graphicsVisibility =
        gpu_.makeSubmissionVisibleToGraphics(node.upload);
    if (graphicsVisibility.hasError()) {
      node.pendingTexture.reset();
      finishTextureNode(node, AssetState::Failed,
                        graphicsVisibility.error());
      ++stats.failed;
      continue;
    }
    if (!graphicsVisibility.value()) {
      continue;
    }
    if (cancelAfterSubmit) {
      node.pendingTexture.reset();
      finishTextureNode(node, AssetState::Cancelled);
      ++stats.cancelled;
      continue;
    }
    node.state = AssetState::Resident;
    if (!node.pendingTexture) {
      node.pendingTexture.reset();
      finishTextureNode(node, AssetState::Cancelled);
      ++stats.cancelled;
      continue;
    }
    auto adopted = resources_.adoptPreparedTexture(
        node.request, std::move(node.pendingTexture));
    if (adopted.hasError()) {
      finishTextureNode(node, AssetState::Failed, adopted.error());
      ++stats.failed;
      continue;
    }
    node.published = adopted.value();
    finishTextureNode(node, AssetState::Published);
    ++stats.published;
  }
  for (ModelNode &node : modelNodes_) {
    const bool cancelAfterSubmit =
        node.state == AssetState::CancelRequested &&
        node.pendingModel != nullptr;
    if ((node.state != AssetState::GpuSubmitted && !cancelAfterSubmit) ||
        !gpu_.isSubmissionComplete(node.upload)) {
      continue;
    }
    auto graphicsVisibility =
        gpu_.makeSubmissionVisibleToGraphics(node.upload);
    if (graphicsVisibility.hasError()) {
      node.pendingModel.reset();
      finishModelNode(node, AssetState::Failed,
                      graphicsVisibility.error());
      ++stats.failed;
      continue;
    }
    if (!graphicsVisibility.value()) {
      continue;
    }
    if (cancelAfterSubmit) {
      node.pendingModel.reset();
      finishModelNode(node, AssetState::Cancelled);
      ++stats.cancelled;
      continue;
    }
    node.state = AssetState::Resident;
    if (!node.pendingModel) {
      node.pendingModel.reset();
      finishModelNode(node, AssetState::Cancelled);
      ++stats.cancelled;
      continue;
    }
    auto adopted =
        resources_.adoptPreparedModel(node.request, std::move(node.pendingModel));
    if (adopted.hasError()) {
      finishModelNode(node, AssetState::Failed, adopted.error());
      ++stats.failed;
      continue;
    }
    node.published = adopted.value();
    finishModelNode(node, AssetState::Published);
    ++stats.published;
  }

  uint32_t materialPublications = 0u;
  for (MaterialNode &node : materialNodes_) {
    if (node.state != AssetState::CpuReady ||
        materialPublications >= config_.maxMaterialPublicationsPerFrame) {
      continue;
    }

    MaterialRequest materialRequest{
        .desc = node.request.desc,
        .debugName = node.request.debugName,
        .sourceIdentity = node.request.sourceIdentity,
    };
    bool dependenciesReady = true;
    size_t dependencyIndex = 0u;
    const auto resolveDependency =
        [this, &dependenciesReady](TextureAssetHandle dependency)
        -> TextureRef {
      if (!isValidAssetHandle(dependency)) {
        return kInvalidTextureRef;
      }
      const TextureNode *texture = find(dependency);
      if (texture == nullptr || texture->state == AssetState::Cancelled ||
          texture->state == AssetState::Failed) {
        return kInvalidTextureRef;
      }
      if (texture->state != AssetState::Published ||
          !isValid(texture->published)) {
        dependenciesReady = false;
        return kInvalidTextureRef;
      }
      return texture->published;
    };
    std::array<TextureRef *, 14u> outputRefs{
        &materialRequest.textureRefs.baseColor,
        &materialRequest.textureRefs.metallicRoughness,
        &materialRequest.textureRefs.normal,
        &materialRequest.textureRefs.occlusion,
        &materialRequest.textureRefs.emissive,
        &materialRequest.textureRefs.clearcoat,
        &materialRequest.textureRefs.clearcoatRoughness,
        &materialRequest.textureRefs.clearcoatNormal,
        &materialRequest.textureRefs.specular,
        &materialRequest.textureRefs.specularColor,
        &materialRequest.textureRefs.sheenColor,
        &materialRequest.textureRefs.sheenRoughness,
        &materialRequest.textureRefs.transmission,
        &materialRequest.textureRefs.thickness,
    };
    forEachMaterialTextureAsset(
        node.request.textures,
        [&resolveDependency, &outputRefs,
         &dependencyIndex](TextureAssetHandle dependency) {
          *outputRefs[dependencyIndex++] = resolveDependency(dependency);
        });
    if (!dependenciesReady) {
      continue;
    }

    auto material = resources_.acquireMaterial(materialRequest);
    if (material.hasError()) {
      finishMaterialNode(node, AssetState::Failed, material.error());
      ++stats.failed;
      continue;
    }
    node.published = material.value();
    finishMaterialNode(node, AssetState::Published);
    ++materialPublications;
    ++stats.published;
  }

  for (EnvironmentNode &node : environmentNodes_) {
    if (node.state == AssetState::Failed ||
        node.state == AssetState::Cancelled) {
      continue;
    }
    const auto dependencyOptional = [&node](size_t index) {
      return (index == 0u && node.request.cubemapOptional) ||
             (index == 1u && node.request.irradianceOptional) ||
             (index == 2u && node.request.prefilteredGgxOptional) ||
             (index == 3u && node.request.prefilteredCharlieOptional) ||
             (index == 4u && node.request.brdfLutOptional);
    };
    bool dependenciesTerminal = true;
    bool requiredDependencyFailed = false;
    EnvironmentHandles resolved{};
    std::array<TextureRef *, 5u> output{
        &resolved.cubemap,
        &resolved.irradiance,
        &resolved.prefilteredGgx,
        &resolved.prefilteredCharlie,
        &resolved.brdfLut,
    };
    for (size_t index = 0u; index < node.textures.size(); ++index) {
      const TextureAssetHandle textureHandle = node.textures[index];
      if (!isValidAssetHandle(textureHandle)) {
        continue;
      }
      const TextureNode *texture = find(textureHandle);
      if (texture == nullptr) {
        if (node.state != AssetState::CancelRequested) {
          dependenciesTerminal = false;
        }
        continue;
      }
      if (!isAssetTerminalState(texture->state)) {
        dependenciesTerminal = false;
        continue;
      }
      if (texture->state == AssetState::Published &&
          isValid(texture->published)) {
        *output[index] = texture->published;
      } else if (!dependencyOptional(index)) {
        requiredDependencyFailed = true;
        if (node.error.empty() && texture != nullptr) {
          node.error = texture->error;
        }
      }
    }

    if (node.state == AssetState::CancelRequested) {
      if (node.environmentPublished && node.boundScene != nullptr &&
          sameEnvironment(node.boundScene->environment(), node.published)) {
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
  std::vector<RenderScene *> dirtyScenes{};
  for (SceneNode &node : sceneNodes_) {
    if (node.state == SceneLoadState::Failed ||
        node.state == SceneLoadState::Cancelled) {
      continue;
    }
    RenderScene *candidateScene =
        node.boundScene != nullptr ? node.boundScene : context.scene;
    auto sceneResult =
        prepareSceneNode(node, context, stats, remainingScenePatches);
    if (sceneResult.hasError()) {
      resources_.endPublicationBatch();
      return Result<AssetPublicationStats, std::string>::makeError(
          sceneResult.error());
    }
    node.progress =
        std::max(node.progress, progressForScene(node, *this));
    if (sceneResult.value() && candidateScene != nullptr &&
        std::ranges::find(dirtyScenes, candidateScene) == dirtyScenes.end()) {
      dirtyScenes.push_back(candidateScene);
    }
  }

  resources_.endPublicationBatch();
  if (context.commitScene) {
    for (RenderScene *scene : dirtyScenes) {
      auto commit = scene->commit();
      if (commit.hasError()) {
        return Result<AssetPublicationStats, std::string>::makeError(
            "AssetSystem: progressive scene commit failed: " +
            commit.error());
      }
      ++stats.sceneCommits;
    }
  }
  return Result<AssetPublicationStats, std::string>::makeResult(stats);
}

AssetSystem::TextureNode *AssetSystem::find(TextureAssetHandle handle) {
  return handle.index < textureNodes_.size() &&
                 textureSlots_.isValid(handle.index, handle.generation)
             ? &textureNodes_[handle.index]
             : nullptr;
}

const AssetSystem::TextureNode *
AssetSystem::find(TextureAssetHandle handle) const {
  return handle.index < textureNodes_.size() &&
                 textureSlots_.isValid(handle.index, handle.generation)
             ? &textureNodes_[handle.index]
             : nullptr;
}

AssetSystem::ModelNode *AssetSystem::find(ModelAssetHandle handle) {
  return handle.index < modelNodes_.size() &&
                 modelSlots_.isValid(handle.index, handle.generation)
             ? &modelNodes_[handle.index]
             : nullptr;
}

const AssetSystem::ModelNode *
AssetSystem::find(ModelAssetHandle handle) const {
  return handle.index < modelNodes_.size() &&
                 modelSlots_.isValid(handle.index, handle.generation)
             ? &modelNodes_[handle.index]
             : nullptr;
}

AssetSystem::MaterialNode *AssetSystem::find(MaterialAssetHandle handle) {
  return handle.index < materialNodes_.size() &&
                 materialSlots_.isValid(handle.index, handle.generation)
             ? &materialNodes_[handle.index]
             : nullptr;
}

const AssetSystem::MaterialNode *
AssetSystem::find(MaterialAssetHandle handle) const {
  return handle.index < materialNodes_.size() &&
                 materialSlots_.isValid(handle.index, handle.generation)
             ? &materialNodes_[handle.index]
             : nullptr;
}

AssetSystem::EnvironmentNode *
AssetSystem::find(EnvironmentAssetHandle handle) {
  return handle.index < environmentNodes_.size() &&
                 environmentSlots_.isValid(handle.index, handle.generation)
             ? &environmentNodes_[handle.index]
             : nullptr;
}

const AssetSystem::EnvironmentNode *
AssetSystem::find(EnvironmentAssetHandle handle) const {
  return handle.index < environmentNodes_.size() &&
                 environmentSlots_.isValid(handle.index, handle.generation)
             ? &environmentNodes_[handle.index]
             : nullptr;
}

AssetSystem::SceneNode *AssetSystem::find(SceneLoadHandle handle) {
  return handle.index < sceneNodes_.size() &&
                 sceneSlots_.isValid(handle.index, handle.generation)
             ? &sceneNodes_[handle.index]
             : nullptr;
}

const AssetSystem::SceneNode *
AssetSystem::find(SceneLoadHandle handle) const {
  return handle.index < sceneNodes_.size() &&
                 sceneSlots_.isValid(handle.index, handle.generation)
             ? &sceneNodes_[handle.index]
             : nullptr;
}

void AssetSystem::pushCompletion(CpuCompletion completion) {
  std::lock_guard lock(completionMutex_);
  completions_.push_back(std::move(completion));
}

std::vector<AssetSystem::CpuCompletion> AssetSystem::takeCompletions() {
  std::lock_guard lock(completionMutex_);
  return std::exchange(completions_, {});
}

void AssetSystem::reclaimReleasedNodes() {
  if (!releasedTerminalNodes_) {
    return;
  }
  releasedTerminalNodes_ = false;
  const auto reclaim = [](auto &nodes, auto &slots, auto isTerminal) {
    for (uint32_t index = 0u; index < slots.slotCount(); ++index) {
      if (!slots.isLive(index)) {
        continue;
      }
      auto &node = nodes[index];
      if (node.subscriberCount != 0u || !isTerminal(node.state)) {
        continue;
      }
      node = {};
      slots.release(index);
    }
  };

  reclaim(textureNodes_, textureSlots_, isAssetTerminalState);
  reclaim(modelNodes_, modelSlots_, isAssetTerminalState);
  reclaim(materialNodes_, materialSlots_, isAssetTerminalState);
  reclaim(environmentNodes_, environmentSlots_, isAssetTerminalState);
  reclaim(sceneNodes_, sceneSlots_, [](SceneLoadState state) {
    return state == SceneLoadState::Complete ||
           state == SceneLoadState::CompleteWithErrors ||
           state == SceneLoadState::Failed ||
           state == SceneLoadState::Cancelled;
  });
}

void AssetSystem::finishTextureNode(TextureNode &node,
                                    AssetState terminalState,
                                    std::string error) {
  node.state = terminalState;
  if (!error.empty() || node.error.empty()) {
    node.error = std::move(error);
  }
  if (terminalState != AssetState::Published) {
    node.prepared.reset();
  }
  textureInFlight_.erase(node.key);
  if (terminalState == AssetState::Cancelled) {
    node.request = {};
    node.key = {};
    node.cpuPayloadBytes = 0u;
  }
  if (node.subscriberCount == 0u && isAssetTerminalState(terminalState)) {
    releasedTerminalNodes_ = true;
  }
}

void AssetSystem::finishModelNode(ModelNode &node, AssetState terminalState,
                                  std::string error) {
  node.state = terminalState;
  node.error = std::move(error);
  if (terminalState != AssetState::Published) {
    node.prepared.reset();
  }
  modelInFlight_.erase(node.key);
  if (terminalState == AssetState::Cancelled) {
    node.request = {};
    node.key = {};
    node.cpuPayloadBytes = 0u;
  }
  if (node.subscriberCount == 0u && isAssetTerminalState(terminalState)) {
    releasedTerminalNodes_ = true;
  }
}

void AssetSystem::finishMaterialNode(MaterialNode &node,
                                     AssetState terminalState,
                                     std::string error) {
  node.state = terminalState;
  node.error = std::move(error);
  materialInFlight_.erase(node.key);
  if (terminalState == AssetState::Cancelled) {
    node.request = {};
    node.key = {};
  }
  if (node.subscriberCount == 0u && isAssetTerminalState(terminalState)) {
    releasedTerminalNodes_ = true;
  }
}

void AssetSystem::finishSceneNode(SceneNode &node,
                                  SceneLoadState terminalState,
                                  std::string error) {
  node.state = terminalState;
  if (!error.empty() || node.error.empty()) {
    node.error = std::move(error);
  }
  if (terminalState == SceneLoadState::Complete ||
      terminalState == SceneLoadState::CompleteWithErrors ||
      terminalState == SceneLoadState::Failed ||
      terminalState == SceneLoadState::Cancelled) {
    node.progress = 1.0f;
    sceneInFlight_.erase(node.key);
  }
  if (terminalState == SceneLoadState::Failed) {
    cancelSceneDependencies(node);
  }
  if (terminalState == SceneLoadState::Cancelled) {
    node.manifest.reset();
    node.materialTasks = {};
    node.materialPreparationFinished = {};
    node.models = {};
    node.materials = {};
    node.textureSubscriptions = {};
    node.fallbackMaterial = {};
    node.instantiation = SceneInstantiationMap{};
    node.root = kInvalidNodeId;
    node.boundScene = nullptr;
    node.modelFallbackMapped = {};
    node.modelMaterialMapped = {};
    node.cpuPayloadBytes = 0u;
    node.hierarchyPublished = false;
  }
  if (node.subscriberCount == 0u &&
      (terminalState == SceneLoadState::Complete ||
       terminalState == SceneLoadState::CompleteWithErrors ||
       terminalState == SceneLoadState::Failed ||
       terminalState == SceneLoadState::Cancelled)) {
    releasedTerminalNodes_ = true;
  }
}

void AssetSystem::cancelSceneDependencies(SceneNode &node) {
  if (node.dependenciesCancelled) {
    return;
  }
  node.dependenciesCancelled = true;
  (void)scheduler_.cancel(node.manifestTask);
  for (const AssetCpuTaskHandle task : node.materialTasks) {
    (void)scheduler_.cancel(task);
  }
  for (const ModelAssetHandle model : node.models) {
    if (isValidAssetHandle(model)) {
      cancel(model);
    }
  }
  for (const MaterialAssetHandle material : node.materials) {
    if (isValidAssetHandle(material)) {
      cancel(material);
    }
  }
  for (const auto &subscriptions : node.textureSubscriptions) {
    for (const TextureAssetHandle texture : subscriptions) {
      if (isValidAssetHandle(texture)) {
        cancel(texture);
      }
    }
  }
  if (isValidAssetHandle(node.fallbackMaterial)) {
    cancel(node.fallbackMaterial);
  }
}

Result<bool, std::string> AssetSystem::prepareSceneNode(
    SceneNode &node, AssetPublicationContext &context,
    AssetPublicationStats &stats, uint32_t &patchBudget) {
  bool sceneDirty = false;
  const auto dependenciesTerminal = [this, &node] {
    if (!node.manifest.has_value()) {
      return false;
    }
    if (std::ranges::any_of(node.materialPreparationFinished,
                            [](uint8_t finished) { return finished == 0u; })) {
      return false;
    }
    const auto handlesTerminal = [this](const auto &handles) {
      for (const auto handle : handles) {
        if (!isValidAssetHandle(handle)) {
          continue;
        }
        const auto *dependency = find(handle);
        if (dependency != nullptr &&
            !isAssetTerminalState(dependency->state)) {
          return false;
        }
      }
      return true;
    };
    if (!handlesTerminal(node.models) || !handlesTerminal(node.materials)) {
      return false;
    }
    for (const auto &subscriptions : node.textureSubscriptions) {
      if (!handlesTerminal(subscriptions)) {
        return false;
      }
    }
    if (isValidAssetHandle(node.fallbackMaterial)) {
      const MaterialNode *fallback = find(node.fallbackMaterial);
      if (fallback != nullptr && !isAssetTerminalState(fallback->state)) {
        return false;
      }
    }
    return true;
  };

  const auto destroyPublishedHierarchy = [&node, &sceneDirty] {
    if (!node.hierarchyPublished || node.boundScene == nullptr ||
        !node.manifest.has_value()) {
      return;
    }
    const ScenePrefab &prefab = node.manifest->prefab;
    for (uint32_t nodeIndex = 0u; nodeIndex < prefab.nodes.size();
         ++nodeIndex) {
      if (prefab.nodes[nodeIndex].parentIndex != kInvalidScenePrefabIndex ||
          nodeIndex >= node.instantiation.nodes.size() ||
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
    cancelSceneDependencies(node);
    if (!node.manifest.has_value()) {
      return Result<bool, std::string>::makeResult(false);
    }
    if (!dependenciesTerminal()) {
      return Result<bool, std::string>::makeResult(false);
    }
    destroyPublishedHierarchy();
    finishSceneNode(node, SceneLoadState::Cancelled);
    ++stats.cancelled;
    return Result<bool, std::string>::makeResult(sceneDirty);
  }
  if (!node.manifest.has_value()) {
    return Result<bool, std::string>::makeResult(false);
  }

  const ScenePrefab &prefab = node.manifest->prefab;
  const bool allDependenciesTerminal = dependenciesTerminal();
  uint32_t completeOnlyPatchBudget = std::numeric_limits<uint32_t>::max();
  uint32_t &scenePatchBudget =
      node.request.publication == ScenePublicationPolicy::CompleteOnly &&
              allDependenciesTerminal
          ? completeOnlyPatchBudget
          : patchBudget;
  if (!node.hierarchyPublished) {
    if (node.request.publication == ScenePublicationPolicy::CompleteOnly &&
        !allDependenciesTerminal) {
      return Result<bool, std::string>::makeResult(false);
    }
    if (scenePatchBudget == 0u) {
      return Result<bool, std::string>::makeResult(false);
    }
    if (context.scene == nullptr) {
      return Result<bool, std::string>::makeResult(false);
    }
    const NodeId parent =
        isValid(context.parent) ? context.parent
                                : context.scene->graph().rootNode();
    auto structure = context.scene->graph().instantiatePrefabStructure(
        prefab, parent, &node.instantiation);
    if (structure.hasError()) {
      ++node.requiredFailures;
      finishSceneNode(node, SceneLoadState::Failed, structure.error());
      ++stats.failed;
      return Result<bool, std::string>::makeResult(false);
    }
    node.root = structure.value();
    node.boundScene = context.scene;
    node.hierarchyPublished = true;
    node.state = SceneLoadState::HierarchyPublished;
    sceneDirty = true;
    ++stats.scenePatches;
    --scenePatchBudget;
  }

  const MaterialNode *fallbackNode = find(node.fallbackMaterial);
  if (fallbackNode == nullptr || fallbackNode->state == AssetState::Failed ||
      fallbackNode->state == AssetState::Cancelled) {
    ++node.requiredFailures;
    destroyPublishedHierarchy();
    finishSceneNode(node, SceneLoadState::Failed,
                    "AssetSystem: scene fallback material failed");
    ++stats.failed;
    return Result<bool, std::string>::makeResult(sceneDirty);
  }
  if (fallbackNode->state != AssetState::Published ||
      !isValid(fallbackNode->published)) {
    node.state = SceneLoadState::PartiallyResident;
    return Result<bool, std::string>::makeResult(sceneDirty);
  }
  const MaterialRef fallbackMaterial = fallbackNode->published;

  for (uint32_t modelIndex = 0u;
       modelIndex < node.models.size() && scenePatchBudget > 0u; ++modelIndex) {
    const std::optional<ModelRef> model = tryResolve(node.models[modelIndex]);
    if (!model.has_value()) {
      continue;
    }
    if (node.modelFallbackMapped[modelIndex] == 0u) {
      resources_.setModelMaterialForAllSources(*model, fallbackMaterial);
      node.modelFallbackMapped[modelIndex] = 1u;
      --scenePatchBudget;
      ++stats.scenePatches;
    }
    for (uint32_t materialIndex = 0u;
         materialIndex < node.materials.size() && scenePatchBudget > 0u;
         ++materialIndex) {
      if (node.modelMaterialMapped[modelIndex][materialIndex] != 0u) {
        continue;
      }
      const std::optional<MaterialRef> material =
          tryResolve(node.materials[materialIndex]);
      if (!material.has_value()) {
        continue;
      }
      (void)resources_.setModelMaterialForSource(
          *model, prefab.materialAssets[materialIndex].sourceMaterialIndex,
          *material);
      node.modelMaterialMapped[modelIndex][materialIndex] = 1u;
      --scenePatchBudget;
      ++stats.scenePatches;
    }
  }

  for (uint32_t renderableIndex = 0u;
       renderableIndex < prefab.renderables.size() &&
       scenePatchBudget > 0u;
       ++renderableIndex) {
    const ScenePrefabRenderable &prefabRenderable =
        prefab.renderables[renderableIndex];
    if (prefabRenderable.meshIndex >= node.models.size()) {
      continue;
    }
    const std::optional<ModelRef> model =
        tryResolve(node.models[prefabRenderable.meshIndex]);
    if (!model.has_value()) {
      continue;
    }
    MaterialRef material = fallbackMaterial;
    if (prefabRenderable.materialIndex < node.materials.size()) {
      if (const std::optional<MaterialRef> resolved =
              tryResolve(node.materials[prefabRenderable.materialIndex]);
          resolved.has_value()) {
        material = *resolved;
      }
    }

    if (renderableIndex >= node.instantiation.renderables.size() ||
        !isValid(node.instantiation.renderables[renderableIndex])) {
      auto attached = node.boundScene->graph().attachPrefabRenderable(
          prefab, renderableIndex, *model, material, node.instantiation);
      if (attached.hasError()) {
        ++node.optionalFailures;
        continue;
      }
      sceneDirty = true;
      --scenePatchBudget;
      ++stats.scenePatches;
      continue;
    }

    MaterialRef current = kInvalidMaterialRef;
    if (node.boundScene->graph().getRenderableMaterial(
            node.instantiation.renderables[renderableIndex], current) &&
        current.value != material.value &&
        node.boundScene->graph().setRenderableMaterial(
            node.instantiation.renderables[renderableIndex], material)) {
      sceneDirty = true;
      --scenePatchBudget;
      ++stats.scenePatches;
    }
  }

  const SceneAssetCounts modelCounts = collectSceneCounts(
      *this, std::span<const ModelAssetHandle>(node.models.data(),
                                               node.models.size()));
  const SceneAssetCounts materialCounts = collectSceneCounts(
      *this, std::span<const MaterialAssetHandle>(node.materials.data(),
                                                  node.materials.size()));
  const SceneAssetCounts textureCounts =
      collectSceneTextureCounts(*this, node);
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
    return Result<bool, std::string>::makeResult(sceneDirty);
  }

  bool renderablesSettled = true;
  for (uint32_t renderableIndex = 0u;
       renderableIndex < prefab.renderables.size(); ++renderableIndex) {
    if (renderableIndex < node.instantiation.renderables.size() &&
        isValid(node.instantiation.renderables[renderableIndex])) {
      continue;
    }
    const ScenePrefabRenderable &renderable =
        prefab.renderables[renderableIndex];
    if (renderable.meshIndex >= node.models.size()) {
      continue;
    }
    const ModelNode *model = find(node.models[renderable.meshIndex]);
    if (model != nullptr && model->state != AssetState::Failed &&
        model->state != AssetState::Cancelled) {
      renderablesSettled = false;
      break;
    }
  }

  bool bindingsSettled = true;
  for (uint32_t modelIndex = 0u; modelIndex < node.models.size();
       ++modelIndex) {
    if (!tryResolve(node.models[modelIndex]).has_value()) {
      continue;
    }
    if (node.modelFallbackMapped[modelIndex] == 0u) {
      bindingsSettled = false;
      break;
    }
    for (uint32_t materialIndex = 0u;
         materialIndex < node.materials.size(); ++materialIndex) {
      if (tryResolve(node.materials[materialIndex]).has_value() &&
          node.modelMaterialMapped[modelIndex][materialIndex] == 0u) {
        bindingsSettled = false;
        break;
      }
    }
    if (!bindingsSettled) {
      break;
    }
  }

  if (allDependenciesTerminal && renderablesSettled && bindingsSettled) {
    finishSceneNode(node,
                    anyFailures ? SceneLoadState::CompleteWithErrors
                                : SceneLoadState::Complete);
  } else {
    node.state = SceneLoadState::PartiallyResident;
  }
  return Result<bool, std::string>::makeResult(sceneDirty);
}

SceneAssetCounts AssetSystem::collectSceneCounts(
    const AssetSystem &assets, std::span<const ModelAssetHandle> handles) {
  SceneAssetCounts counts{.total = static_cast<uint32_t>(handles.size())};
  for (const ModelAssetHandle handle : handles) {
    const ModelNode *node = assets.find(handle);
    if (node == nullptr) {
      ++counts.queued;
      continue;
    }
    switch (node->state) {
    case AssetState::CpuReady:
    case AssetState::GpuQueued:
      ++counts.cpuReady;
      break;
    case AssetState::GpuSubmitted:
    case AssetState::Resident:
      ++counts.gpuSubmitted;
      break;
    case AssetState::Published:
      ++counts.published;
      break;
    case AssetState::Failed:
      ++counts.failed;
      break;
    case AssetState::Cancelled:
      ++counts.cancelled;
      break;
    default:
      ++counts.queued;
      break;
    }
  }
  return counts;
}

SceneAssetCounts AssetSystem::collectSceneCounts(
    const AssetSystem &assets, std::span<const MaterialAssetHandle> handles) {
  SceneAssetCounts counts{.total = static_cast<uint32_t>(handles.size())};
  for (const MaterialAssetHandle handle : handles) {
    const MaterialNode *node = assets.find(handle);
    if (node == nullptr) {
      ++counts.queued;
      continue;
    }
    switch (node->state) {
    case AssetState::CpuReady:
    case AssetState::GpuQueued:
      ++counts.cpuReady;
      break;
    case AssetState::GpuSubmitted:
    case AssetState::Resident:
      ++counts.gpuSubmitted;
      break;
    case AssetState::Published:
      ++counts.published;
      break;
    case AssetState::Failed:
      ++counts.failed;
      break;
    case AssetState::Cancelled:
      ++counts.cancelled;
      break;
    default:
      ++counts.queued;
      break;
    }
  }
  return counts;
}

SceneAssetCounts
AssetSystem::collectSceneTextureCounts(const AssetSystem &assets,
                                       const SceneNode &node) {
  SceneAssetCounts counts{};
  for (const auto &subscriptions : node.textureSubscriptions) {
    const SceneAssetCounts materialTextures = [&assets, &subscriptions] {
      SceneAssetCounts local{
          .total = static_cast<uint32_t>(subscriptions.size())};
      for (const TextureAssetHandle handle : subscriptions) {
        const TextureNode *texture = assets.find(handle);
        if (texture == nullptr) {
          ++local.queued;
          continue;
        }
        switch (texture->state) {
        case AssetState::CpuReady:
        case AssetState::GpuQueued:
          ++local.cpuReady;
          break;
        case AssetState::GpuSubmitted:
        case AssetState::Resident:
          ++local.gpuSubmitted;
          break;
        case AssetState::Published:
          ++local.published;
          break;
        case AssetState::Failed:
          ++local.failed;
          break;
        case AssetState::Cancelled:
          ++local.cancelled;
          break;
        default:
          ++local.queued;
          break;
        }
      }
      return local;
    }();
    counts.total += materialTextures.total;
    counts.queued += materialTextures.queued;
    counts.cpuReady += materialTextures.cpuReady;
    counts.gpuSubmitted += materialTextures.gpuSubmitted;
    counts.published += materialTextures.published;
    counts.failed += materialTextures.failed;
    counts.cancelled += materialTextures.cancelled;
  }
  return counts;
}

size_t AssetSystem::MaterialAssetKeyHash::operator()(
    const MaterialAssetKey &key) const noexcept {
  size_t seed = static_cast<size_t>(key.descHash);
  const auto combine = [&seed](uint64_t value) {
    seed ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ull +
            (seed << 6u) + (seed >> 2u);
  };
  for (const uint64_t handle : key.textureHandles) {
    combine(handle);
  }
  combine(std::hash<std::string>{}(key.sourceIdentity));
  return seed;
}

size_t
AssetSystem::SceneKeyHash::operator()(const SceneKey &key) const noexcept {
  size_t seed = std::hash<std::string>{}(key.canonicalPath);
  const auto combine = [&seed](uint64_t value) {
    seed ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ull +
            (seed << 6u) + (seed >> 2u);
  };
  combine(key.importOptionsHash);
  combine(static_cast<uint64_t>(key.publication));
  combine(static_cast<uint64_t>(key.failurePolicy));
  return seed;
}

float AssetSystem::progressForState(AssetState state) noexcept {
  switch (state) {
  case AssetState::Queued:
    return 0.0f;
  case AssetState::CpuRunning:
    return 0.15f;
  case AssetState::CpuReady:
  case AssetState::GpuQueued:
    return 0.55f;
  case AssetState::GpuSubmitted:
    return 0.75f;
  case AssetState::Resident:
    return 0.9f;
  case AssetState::Published:
  case AssetState::Cancelled:
  case AssetState::Failed:
    return 1.0f;
  case AssetState::CancelRequested:
    return 0.0f;
  }
  return 0.0f;
}

float AssetSystem::progressForScene(const SceneNode &node,
                                    const AssetSystem &assets) {
  if (node.state == SceneLoadState::Complete ||
      node.state == SceneLoadState::CompleteWithErrors ||
      node.state == SceneLoadState::Failed ||
      node.state == SceneLoadState::Cancelled) {
    return 1.0f;
  }
  if (!node.manifest.has_value()) {
    return node.state == SceneLoadState::Requested ? 0.05f : 0.0f;
  }

  const SceneAssetCounts models = collectSceneCounts(
      assets, std::span<const ModelAssetHandle>(node.models.data(),
                                                node.models.size()));
  const SceneAssetCounts materials = collectSceneCounts(
      assets, std::span<const MaterialAssetHandle>(node.materials.data(),
                                                   node.materials.size()));
  const SceneAssetCounts textures = collectSceneTextureCounts(assets, node);
  const auto weightedProgress = [](const SceneAssetCounts &counts) {
    if (counts.total == 0u) {
      return 1.0f;
    }
    const float value =
        static_cast<float>(counts.queued) * 0.10f +
        static_cast<float>(counts.cpuReady) * 0.55f +
        static_cast<float>(counts.gpuSubmitted) * 0.75f +
        static_cast<float>(counts.published + counts.failed +
                           counts.cancelled);
    return value / static_cast<float>(counts.total);
  };
  const uint32_t classCount =
      (models.total != 0u ? 1u : 0u) + (materials.total != 0u ? 1u : 0u) +
      (textures.total != 0u ? 1u : 0u);
  float classProgress = 0.0f;
  if (models.total != 0u) {
    classProgress += weightedProgress(models);
  }
  if (materials.total != 0u) {
    classProgress += weightedProgress(materials);
  }
  if (textures.total != 0u) {
    classProgress += weightedProgress(textures);
  }
  const float assetProgress =
      classCount == 0u
          ? 1.0f
          : classProgress / static_cast<float>(classCount);
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
