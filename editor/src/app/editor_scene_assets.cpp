#include "nuri/editor_pch.h"

#include "editor_scene_assets.h"

#include "nuri/core/log.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/scene_importer.h"

namespace nuri {
namespace {

template <typename Ref>
void releaseResourceRef(ResourceManager *resources, Ref &ref, Ref invalidRef) {
  if (resources != nullptr && isValid(ref)) {
    resources->release(ref);
  }
  ref = invalidRef;
}

} // namespace

ImportedPrefabSceneResources::~ImportedPrefabSceneResources() { release(); }

ImportedPrefabSceneResources::ImportedPrefabSceneResources(
    ImportedPrefabSceneResources &&other) noexcept {
  *this = std::move(other);
}

ImportedPrefabSceneResources &ImportedPrefabSceneResources::operator=(
    ImportedPrefabSceneResources &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  resources = std::exchange(other.resources, nullptr);
  assetSystem = std::exchange(other.assetSystem, nullptr);
  sourcePath = std::move(other.sourcePath);
  sceneLoad = std::exchange(other.sceneLoad, {});
  prefab = std::move(other.prefab);
  assets = std::move(other.assets);
  fallbackLights = std::move(other.fallbackLights);
  ready = std::exchange(other.ready, false);
  textureArtifactBakeQueued =
      std::exchange(other.textureArtifactBakeQueued, false);
  return *this;
}

void ImportedPrefabSceneResources::release() noexcept {
  if (assetSystem != nullptr && isValid(sceneLoad)) {
    assetSystem->cancel(sceneLoad);
  }
  sceneLoad = {};
  assetSystem = nullptr;
  assets = ScenePrefabAssets{};
  prefab = ScenePrefab{};
  sourcePath.clear();
  fallbackLights.clear();
  resources = nullptr;
  ready = false;
  textureArtifactBakeQueued = false;
}

SimpleModelSceneAssets::~SimpleModelSceneAssets() { release(); }

SimpleModelSceneAssets::SimpleModelSceneAssets(
    SimpleModelSceneAssets &&other) noexcept {
  *this = std::move(other);
}

SimpleModelSceneAssets &
SimpleModelSceneAssets::operator=(SimpleModelSceneAssets &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  resources = std::exchange(other.resources, nullptr);
  sourcePath = std::move(other.sourcePath);
  model = std::exchange(other.model, kInvalidModelRef);
  material = std::exchange(other.material, kInvalidMaterialRef);
  fallbackLights = std::move(other.fallbackLights);
  ready = std::exchange(other.ready, false);
  textureArtifactBakeQueued =
      std::exchange(other.textureArtifactBakeQueued, false);
  return *this;
}

void SimpleModelSceneAssets::release() noexcept {
  releaseResourceRef(resources, model, kInvalidModelRef);
  releaseResourceRef(resources, material, kInvalidMaterialRef);
  sourcePath.clear();
  fallbackLights.clear();
  resources = nullptr;
  ready = false;
  textureArtifactBakeQueued = false;
}

StreamingSceneState::~StreamingSceneState() { release(); }

StreamingSceneState::StreamingSceneState(StreamingSceneState &&other) noexcept {
  *this = std::move(other);
}

StreamingSceneState &
StreamingSceneState::operator=(StreamingSceneState &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  assets = std::exchange(other.assets, nullptr);
  sourcePath = std::move(other.sourcePath);
  sceneLoad = std::exchange(other.sceneLoad, {});
  model = std::exchange(other.model, kInvalidModelRef);
  renderableId = std::exchange(other.renderableId, kInvalidRenderableId);
  baseModel = other.baseModel;
  configured = std::exchange(other.configured, false);
  baseConfigured = std::exchange(other.baseConfigured, false);
  populationComplete = std::exchange(other.populationComplete, false);
  cameraConfigured = std::exchange(other.cameraConfigured, false);
  lightingConfigured = std::exchange(other.lightingConfigured, false);
  loadFailed = std::exchange(other.loadFailed, false);
  loadError = std::move(other.loadError);
  populationCursor = std::exchange(other.populationCursor, 0u);
  populationProgress = std::exchange(other.populationProgress, 0.0f);
  graphCapacityReservation = other.graphCapacityReservation;
  other.graphCapacityReservation = {};
  textureArtifactBakeQueued =
      std::exchange(other.textureArtifactBakeQueued, false);
  loadStartTimeSeconds = std::exchange(other.loadStartTimeSeconds, 0.0);
  lastProgressLogTimeSeconds =
      std::exchange(other.lastProgressLogTimeSeconds, 0.0);
  other.baseModel = glm::mat4(1.0f);
  return *this;
}

void StreamingSceneState::release() noexcept {
  if (assets != nullptr && isValid(sceneLoad)) {
    assets->cancel(sceneLoad);
  }
  sceneLoad = {};
  model = kInvalidModelRef;
  sourcePath.clear();
  renderableId = kInvalidRenderableId;
  baseModel = glm::mat4(1.0f);
  configured = false;
  baseConfigured = false;
  populationComplete = false;
  cameraConfigured = false;
  lightingConfigured = false;
  loadFailed = false;
  loadError.clear();
  populationCursor = 0u;
  populationProgress = 0.0f;
  graphCapacityReservation = {};
  assets = nullptr;
  textureArtifactBakeQueued = false;
  loadStartTimeSeconds = 0.0;
  lastProgressLogTimeSeconds = 0.0;
}

Result<void, std::string> prepareImportedPrefabSceneResources(
    EditorRuntime &runtime, std::string_view sceneName,
    const std::filesystem::path &modelPath,
    const MeshImportOptions &importOptions, ImportedPrefabSceneResources &out) {
  if (isValid(out.sceneLoad)) {
    return Result<void, std::string>::makeResult();
  }

  out.release();
  out.resources = &runtime.resources();
  out.assetSystem = &runtime.assets();
  out.sourcePath = modelPath;
  auto requested = runtime.assets().requestScene(SceneLoadRequest{
      .path = modelPath.string(),
      .importOptions = SceneImportOptions{.assetBuildOptions = importOptions},
      .priority = AssetPriority::Critical,
      .publication = ScenePublicationPolicy::CompleteOnly,
      .failurePolicy = SceneFailurePolicy::BestEffort,
      .publicationTarget = runtime.scenePublicationTarget(),
      .debugName = std::string(sceneName),
  });
  if (requested.hasError()) {
    out.release();
    return Result<void, std::string>::makeError(requested.error());
  }
  out.sceneLoad = requested.value();
  return Result<void, std::string>::makeResult();
}

Result<bool, std::string>
refreshImportedPrefabSceneResources(EditorRuntime &runtime,
                                    std::string_view sceneName,
                                    ImportedPrefabSceneResources &out) {
  if (out.ready) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!isValid(out.sceneLoad)) {
    return Result<bool, std::string>::makeError(
        "async prefab scene request is invalid");
  }
  const SceneLoadSnapshot status = runtime.assets().query(out.sceneLoad);
  if (status.state == SceneLoadState::Failed ||
      status.state == SceneLoadState::Cancelled) {
    return Result<bool, std::string>::makeError(
        status.error.empty() ? "async prefab scene load failed" : status.error);
  }
  if (!status.terminal()) {
    return Result<bool, std::string>::makeResult(false);
  }
  const ScenePrefab *prefab = runtime.assets().tryGetScenePrefab(out.sceneLoad);
  auto assets = runtime.assets().tryGetSceneAssets(out.sceneLoad);
  if (prefab == nullptr || !assets.has_value()) {
    return Result<bool, std::string>::makeError(
        "async prefab scene completed without publication data");
  }
  out.prefab = *prefab;
  out.assets = std::move(*assets);
  out.ready = true;
  NURI_LOG_INFO("prepareImportedPrefabSceneResources: %s loaded (nodes=%zu "
                "renderables=%zu lights=%zu meshes=%zu)",
                std::string(sceneName).c_str(), out.prefab.nodes.size(),
                out.prefab.renderables.size(), out.prefab.lights.size(),
                out.prefab.meshAssets.size());
  return Result<bool, std::string>::makeResult(true);
}

void loadImportedLightsForScene(std::string_view sceneName,
                                std::string_view modelPath,
                                std::vector<ImportedSceneLight> &outLights) {
  auto loadResult = SceneImporter::loadSceneFromFile(modelPath);
  if (loadResult.hasError()) {
    NURI_LOG_WARNING("loadImportedLightsForScene: failed to import %s punctual "
                     "lights from '%s': %s",
                     std::string(sceneName).c_str(),
                     std::string(modelPath).c_str(),
                     loadResult.error().c_str());
    outLights.clear();
    return;
  }

  outLights.assign(loadResult.value().lights.begin(),
                   loadResult.value().lights.end());
}

static void
queueSceneTextureArtifactBake(EditorRuntime &runtime,
                              const std::filesystem::path &sourcePath,
                              bool &queuedFlag) {
  if (queuedFlag) {
    return;
  }
  queuedFlag = runtime.deferSceneTextureArtifactBake(sourcePath);
}

void queueSceneTextureArtifactBakeIfNeeded(
    EditorRuntime &runtime, ImportedPrefabSceneResources &assets) {
  queueSceneTextureArtifactBake(runtime, assets.sourcePath,
                                assets.textureArtifactBakeQueued);
}

void queueSceneTextureArtifactBakeIfNeeded(EditorRuntime &runtime,
                                           SimpleModelSceneAssets &assets) {
  queueSceneTextureArtifactBake(runtime, assets.sourcePath,
                                assets.textureArtifactBakeQueued);
}

void queueSceneTextureArtifactBakeIfNeeded(EditorRuntime &runtime,
                                           StreamingSceneState &assets) {
  queueSceneTextureArtifactBake(runtime, assets.sourcePath,
                                assets.textureArtifactBakeQueued);
}

} // namespace nuri
