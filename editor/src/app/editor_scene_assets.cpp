#include "nuri/editor_pch.h"

#include "editor_scene_assets.h"

#include "nuri/core/log.h"
#include "nuri/resources/gpu/material.h"
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
  sourcePath = std::move(other.sourcePath);
  prefab = std::move(other.prefab);
  assets = std::move(other.assets);
  fallbackLights = std::move(other.fallbackLights);
  ready = std::exchange(other.ready, false);
  textureArtifactBakeQueued =
      std::exchange(other.textureArtifactBakeQueued, false);
  return *this;
}

void ImportedPrefabSceneResources::release() noexcept {
  if (resources != nullptr) {
    for (ModelRef model : assets.models) {
      if (isValid(model)) {
        resources->release(model);
      }
    }
    for (MaterialRef material : assets.materials) {
      if (isValid(material)) {
        resources->release(material);
      }
    }
  }
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
  resources = std::exchange(other.resources, nullptr);
  sourcePath = std::move(other.sourcePath);
  prefab = std::move(other.prefab);
  model = std::exchange(other.model, kInvalidModelRef);
  material = std::exchange(other.material, kInvalidMaterialRef);
  asyncLoad = std::move(other.asyncLoad);
  renderableId = std::exchange(other.renderableId, kInvalidRenderableId);
  baseModel = other.baseModel;
  loadFailed = std::exchange(other.loadFailed, false);
  loadError = std::move(other.loadError);
  textureArtifactBakeQueued =
      std::exchange(other.textureArtifactBakeQueued, false);
  loadStartTimeSeconds = std::exchange(other.loadStartTimeSeconds, 0.0);
  lastProgressLogTimeSeconds =
      std::exchange(other.lastProgressLogTimeSeconds, 0.0);
  other.baseModel = glm::mat4(1.0f);
  return *this;
}

void StreamingSceneState::release() noexcept {
  asyncLoad.reset();
  prefab.release();
  releaseResourceRef(resources, model, kInvalidModelRef);
  releaseResourceRef(resources, material, kInvalidMaterialRef);
  sourcePath.clear();
  renderableId = kInvalidRenderableId;
  baseModel = glm::mat4(1.0f);
  loadFailed = false;
  loadError.clear();
  resources = nullptr;
  textureArtifactBakeQueued = false;
  loadStartTimeSeconds = 0.0;
  lastProgressLogTimeSeconds = 0.0;
}

Result<void, std::string> prepareImportedPrefabSceneResources(
    EditorRuntime &runtime, std::string_view sceneName,
    const std::filesystem::path &modelPath,
    const MeshImportOptions &importOptions, ImportedPrefabSceneResources &out) {
  if (out.ready) {
    return Result<void, std::string>::makeResult();
  }

  out.release();
  out.resources = &runtime.resources();
  out.sourcePath = modelPath;

  auto sceneResult = SceneImporter::loadSceneFromFile(
      modelPath.string(),
      SceneImportOptions{.assetBuildOptions = importOptions});
  if (sceneResult.hasError()) {
    return Result<void, std::string>::makeError(sceneResult.error());
  }

  ImportedScene importedScene = std::move(sceneResult.value());
  out.fallbackLights.assign(importedScene.lights.begin(),
                            importedScene.lights.end());

  auto prefabResult = SceneImporter::buildScenePrefab(importedScene);
  if (prefabResult.hasError()) {
    return Result<void, std::string>::makeError(prefabResult.error());
  }

  auto assetsResult =
      runtime.resources().acquireScenePrefabAssets(prefabResult.value());
  if (assetsResult.hasError()) {
    return Result<void, std::string>::makeError(assetsResult.error());
  }

  out.prefab = std::move(prefabResult.value());
  out.assets = std::move(assetsResult.value());
  out.ready = true;
  NURI_LOG_INFO("prepareImportedPrefabSceneResources: %s loaded (nodes=%zu "
                "renderables=%zu lights=%zu meshes=%zu)",
                std::string(sceneName).c_str(), out.prefab.nodes.size(),
                out.prefab.renderables.size(), out.prefab.lights.size(),
                out.prefab.meshAssets.size());
  return Result<void, std::string>::makeResult();
}

Result<void, std::string> prepareSimpleImportedModelSceneAssets(
    EditorRuntime &runtime, std::string_view sceneName,
    const std::filesystem::path &modelPath,
    const MeshImportOptions &importOptions, std::string_view modelDebugName,
    std::string_view importedMaterialPrefix,
    std::string_view fallbackMaterialDebugName, bool loadImportedLights,
    SimpleModelSceneAssets &out) {
  if (out.ready) {
    return Result<void, std::string>::makeResult();
  }

  out.release();
  out.resources = &runtime.resources();
  out.sourcePath = modelPath;

  auto modelResult = runtime.resources().acquireModel(ModelRequest{
      .path = modelPath.string(),
      .importOptions = importOptions,
      .debugName = std::string(modelDebugName),
  });
  if (modelResult.hasError()) {
    return Result<void, std::string>::makeError(modelResult.error());
  }
  out.model = modelResult.value();
  out.material = acquireImportedMaterialOrFallback(
      runtime.resources(), sceneName, modelPath.string(), out.model,
      importedMaterialPrefix, fallbackMaterialDebugName);
  if (!isValid(out.material)) {
    return Result<void, std::string>::makeError(
        std::string("prepareSimpleImportedModelSceneAssets: failed to acquire "
                    "material for '") +
        std::string(sceneName) + "'");
  }
  if (loadImportedLights) {
    loadImportedLightsForScene(sceneName, modelPath.string(),
                               out.fallbackLights);
  }
  out.ready = true;
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] MaterialRef acquireImportedMaterialOrFallback(
    ResourceManager &resources, std::string_view sceneName,
    std::string_view modelPath, ModelRef modelRef,
    std::string_view debugNamePrefix, std::string_view fallbackDebugName) {
  auto loadResult = resources.acquireMaterialsFromModel(ImportedMaterialRequest{
      .modelPath = std::string(modelPath),
      .model = modelRef,
      .debugNamePrefix = std::string(debugNamePrefix),
  });
  if (!loadResult.hasError() && isValid(loadResult.value().firstMaterial)) {
    return loadResult.value().firstMaterial;
  }

  if (loadResult.hasError()) {
    NURI_LOG_WARNING("prepareSimpleImportedModelSceneAssets: failed to import "
                     "%s materials from '%s': %s",
                     std::string(sceneName).c_str(),
                     std::string(modelPath).c_str(),
                     loadResult.error().c_str());
  }

  auto fallbackMaterialResult = resources.acquireMaterial(MaterialRequest{
      .desc = MaterialDesc{},
      .debugName = std::string(fallbackDebugName),
  });
  NURI_ASSERT(!fallbackMaterialResult.hasError(),
              "Failed to acquire %s fallback material: %s",
              std::string(sceneName).c_str(),
              fallbackMaterialResult.error().c_str());
  if (fallbackMaterialResult.hasError()) {
    NURI_LOG_ERROR("prepareSimpleImportedModelSceneAssets: failed to acquire "
                   "fallback material for '%s': %s",
                   std::string(sceneName).c_str(),
                   fallbackMaterialResult.error().c_str());
    return kInvalidMaterialRef;
  }
  const MaterialRef fallbackMaterial = fallbackMaterialResult.value();
  resources.setModelMaterialForAllSources(modelRef, fallbackMaterial);
  return fallbackMaterial;
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
  bakery::BakerySystem *bakery = runtime.bakery();
  if (bakery == nullptr) {
    NURI_LOG_WARNING(
        "queueSceneTextureArtifactBakeIfNeeded: bakery system is unavailable "
        "for '%s'",
        sourcePath.string().c_str());
    return;
  }
  auto enqueueResult = bakery->enqueue(
      bakery::BakeRequest{bakery::SceneTextureArtifactsBakeRequest{
          .scenePath = sourcePath,
          .prebuildNativeTargets =
              {
                  bakery::SceneTextureArtifactTarget::BC7,
              },
          .forceRebuild = false,
      }});
  if (enqueueResult.hasError()) {
    NURI_LOG_WARNING(
        "queueSceneTextureArtifactBakeIfNeeded: failed to queue scene texture "
        "artifact bake for '%s': %s",
        sourcePath.string().c_str(), enqueueResult.error().c_str());
    return;
  }
  queuedFlag = true;
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
