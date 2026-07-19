#include "nuri/editor_pch.h"

#include "editor_scene_assets.h"

#include "nuri/app/editor_scene_spec.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

[[nodiscard]] std::optional<std::pair<ModelRef, MaterialRef>>
firstResolvedPrefabRenderable(const ImportedPrefabSceneResources &assets) {
  if (!assets.ready) {
    return std::nullopt;
  }
  for (const ScenePrefabRenderable &renderable : assets.prefab.renderables) {
    if (renderable.meshAssetIndex < assets.assets.models.size() &&
        renderable.materialAssetIndex < assets.assets.materials.size()) {
      const ModelRef model = assets.assets.models[renderable.meshAssetIndex];
      const MaterialRef material =
          assets.assets.materials[renderable.materialAssetIndex];
      if (isValid(model) && isValid(material)) {
        return std::make_pair(model, material);
      }
    }
  }
  return std::nullopt;
}

[[nodiscard]] BoundingBox
chooseBounds(EditorRuntime &runtime, const ImportedPrefabSceneResources &assets,
             const PrefabSceneFactoryDesc &desc, ModelRef fallbackModel) {
  if (desc.computeBounds) {
    if (const auto bounds = desc.computeBounds(runtime, assets);
        bounds.has_value()) {
      return *bounds;
    }
  }
  if (const auto prefabBounds =
          runtime.computeImportedPrefabBounds(assets, desc.baseModel);
      prefabBounds.has_value()) {
    return *prefabBounds;
  }
  return runtime
      .requireLoadedModel(fallbackModel, "Scene model is not loaded",
                          "Scene model record lookup failed")
      .bounds();
}

[[nodiscard]] float computeNiagaraBistroScale(const BoundingBox &bounds) {
  constexpr float kNiagaraBistroTargetRadius = 120.0f;
  constexpr float kNiagaraBistroMinScale = 0.0005f;
  constexpr float kNiagaraBistroMaxScale = 2.0f;
  const float rawRadius =
      std::max(0.5f * glm::length(bounds.getSize()), 1.0e-3f);
  const float targetScale = kNiagaraBistroTargetRadius / rawRadius;
  return std::clamp(targetScale, kNiagaraBistroMinScale,
                    kNiagaraBistroMaxScale);
}

[[nodiscard]] std::optional<RenderableId>
firstInstantiatedPrefabRenderable(EditorRuntime &runtime,
                                  const ImportedPrefabSceneResources &assets) {
  const auto instantiation =
      runtime.assets().tryGetSceneInstantiation(assets.sceneLoad);
  if (!instantiation.has_value()) {
    return std::nullopt;
  }
  for (const RenderableId renderable : instantiation->renderables) {
    if (isValid(renderable)) {
      return renderable;
    }
  }
  return std::nullopt;
}

void applyPrefabRootTransform(EditorRuntime &runtime,
                              const ImportedPrefabSceneResources &assets,
                              const glm::mat4 &baseModel) {
  const auto instantiation =
      runtime.assets().tryGetSceneInstantiation(assets.sceneLoad);
  if (!instantiation.has_value()) {
    return;
  }
  for (uint32_t nodeIndex = 0u; nodeIndex < assets.prefab.nodes.size();
       ++nodeIndex) {
    if (assets.prefab.nodes[nodeIndex].parentIndex !=
            kInvalidScenePrefabIndex ||
        nodeIndex >= instantiation->nodes.size() ||
        !isValid(instantiation->nodes[nodeIndex])) {
      continue;
    }
    (void)runtime.scene().graph().setNodeLocalTransform(
        instantiation->nodes[nodeIndex],
        baseModel * assets.prefab.nodes[nodeIndex].localFromParent);
  }
}

void applyPendingPrefabRootTransform(EditorRuntime &runtime,
                                     SceneLoadHandle sceneLoad,
                                     const glm::mat4 &baseModel) {
  const ScenePrefab *prefab = runtime.assets().tryGetScenePrefab(sceneLoad);
  const auto instantiation =
      runtime.assets().tryGetSceneInstantiation(sceneLoad);
  if (prefab == nullptr || !instantiation.has_value()) {
    return;
  }
  for (uint32_t nodeIndex = 0u; nodeIndex < prefab->nodes.size(); ++nodeIndex) {
    if (prefab->nodes[nodeIndex].parentIndex != kInvalidScenePrefabIndex ||
        nodeIndex >= instantiation->nodes.size() ||
        !isValid(instantiation->nodes[nodeIndex])) {
      continue;
    }
    (void)runtime.scene().graph().setNodeLocalTransform(
        instantiation->nodes[nodeIndex],
        baseModel * prefab->nodes[nodeIndex].localFromParent);
  }
}

} // namespace

EditorSceneSpec makeCustomScene(EditorSceneSpec spec) { return spec; }

EditorSceneSpec makePrefabScene(PrefabSceneFactoryDesc desc) {
  auto assets = std::make_shared<ImportedPrefabSceneResources>();
  auto configured = std::make_shared<bool>(false);
  auto stagingError = std::make_shared<std::string>();
  return EditorSceneSpec{
      .info = desc.info,
      .prepare = [desc, assets](EditorScenePrepareContext &ctx)
          -> Result<void, std::string> {
        assets->resources = &ctx.runtime.resources();
        assets->assetSystem = &ctx.runtime.assets();
        assets->sourcePath = desc.sourcePath;
        queueSceneTextureArtifactBakeIfNeeded(ctx.runtime, *assets);
        return Result<void, std::string>::makeResult();
      },
      .activate =
          [desc, assets, configured, stagingError](
              EditorSceneActivateContext &ctx) -> Result<void, std::string> {
        if (desc.configureRender) {
          desc.configureRender(ctx.runtime);
        } else {
          ctx.runtime.configureStaticModelOpaqueSettings(desc.lodThresholds);
        }
        *configured = false;
        stagingError->clear();
        return prepareImportedPrefabSceneResources(ctx.runtime, desc.info.label,
                                                   desc.sourcePath,
                                                   desc.importOptions, *assets);
      },
      .deactivate =
          [assets, configured, stagingError](EditorSceneDeactivateContext &) {
            assets->release();
            *configured = false;
            stagingError->clear();
          },
      .update =
          [desc, assets, configured,
           stagingError](EditorSceneUpdateContext &ctx) {
            if (*configured || !isValid(assets->sceneLoad)) {
              return;
            }
            applyPendingPrefabRootTransform(ctx.runtime, assets->sceneLoad,
                                            desc.baseModel);
            auto refreshed = refreshImportedPrefabSceneResources(
                ctx.runtime, desc.info.label, *assets);
            if (refreshed.hasError()) {
              NURI_LOG_ERROR("%s: async prefab load failed: %s",
                             desc.info.label.c_str(),
                             refreshed.error().c_str());
              *stagingError = refreshed.error();
              *configured = true;
              return;
            }
            if (!refreshed.value()) {
              return;
            }
            applyPrefabRootTransform(ctx.runtime, *assets, desc.baseModel);
            if (desc.prepareAdditionalAssets) {
              auto additionalAssetsResult =
                  desc.prepareAdditionalAssets(ctx.runtime, *assets);
              if (additionalAssetsResult.hasError()) {
                NURI_LOG_ERROR("%s: additional asset preparation failed: %s",
                               desc.info.label.c_str(),
                               additionalAssetsResult.error().c_str());
                *stagingError = additionalAssetsResult.error();
                *configured = true;
                return;
              }
            }

            const auto instantiated =
                firstInstantiatedPrefabRenderable(ctx.runtime, *assets);
            if (!instantiated.has_value() && desc.requirePrefabInstantiation) {
              NURI_LOG_ERROR("%s: scene prefab instantiation failed",
                             desc.info.label.c_str());
              *stagingError = "scene prefab instantiation failed";
              *configured = true;
              return;
            }
            std::optional<BoundingBox> sceneBounds;
            const auto fallback = firstResolvedPrefabRenderable(*assets);
            if (fallback.has_value()) {
              sceneBounds =
                  chooseBounds(ctx.runtime, *assets, desc, fallback->first);
              FramedSceneCameraState cameraState{};
              if (desc.configureCamera) {
                desc.configureCamera(ctx.runtime, *assets, *sceneBounds);
              } else {
                cameraState = ctx.runtime.frameSceneCamera(
                    *sceneBounds, desc.baseModel, 2.5f, 2.0f,
                    glm::vec4(0.42f, 0.20f, 1.0f, 0.2f),
                    glm::vec2(0.06f, 0.0f));
                ctx.runtime.logSingleRenderableSceneStats(
                    desc.info.label,
                    ctx.runtime.requireLoadedModel(
                        fallback->first, "Scene model is not loaded",
                        "Scene model record lookup failed"),
                    cameraState);
              }
            } else {
              NURI_LOG_WARNING(
                  "%s: camera framing skipped - no valid renderable in "
                  "prefab",
                  desc.info.label.c_str());
            }
            if (!sceneBounds.has_value() && desc.populateScene) {
              sceneBounds = ctx.runtime.computeImportedPrefabBounds(
                  *assets, desc.baseModel);
            }
            if (desc.populateScene) {
              if (!sceneBounds.has_value()) {
                NURI_LOG_ERROR(
                    "%s: scene bounds are unavailable for prefab population",
                    desc.info.label.c_str());
                *stagingError =
                    "scene bounds are unavailable for prefab population";
                *configured = true;
                return;
              }
              auto populateResult =
                  desc.populateScene(ctx.runtime, *assets, *sceneBounds);
              if (populateResult.hasError()) {
                NURI_LOG_ERROR("%s: scene population failed: %s",
                               desc.info.label.c_str(),
                               populateResult.error().c_str());
                *stagingError = populateResult.error();
                *configured = true;
                return;
              }
            }

            if (assets->prefab.lights.empty()) {
              ctx.runtime.finalizeSceneLighting(assets->fallbackLights,
                                                desc.baseModel);
            }
            *configured = true;
          },
      .stagingStatus =
          [configured, stagingError] {
            return EditorSceneStagingStatus{
                .ready = *configured && stagingError->empty(),
                .failed = !stagingError->empty(),
                .progress = *configured ? 1.0f : 0.0f,
                .error = *stagingError,
            };
          },
  };
}

EditorSceneSpec makeAnimatedPrefabScene(AnimatedPrefabSceneFactoryDesc desc) {
  auto assets = std::make_shared<ImportedPrefabSceneResources>();
  auto animation = std::make_shared<AnimatedPrefabSceneState>();
  auto configured = std::make_shared<bool>(false);
  auto stagingError = std::make_shared<std::string>();
  return EditorSceneSpec{
      .info = desc.prefab.info,
      .prepare = [prefab = desc.prefab, assets](EditorScenePrepareContext &ctx)
          -> Result<void, std::string> {
        assets->resources = &ctx.runtime.resources();
        assets->assetSystem = &ctx.runtime.assets();
        assets->sourcePath = prefab.sourcePath;
        queueSceneTextureArtifactBakeIfNeeded(ctx.runtime, *assets);
        return Result<void, std::string>::makeResult();
      },
      .activate =
          [desc, assets, animation, configured, stagingError](
              EditorSceneActivateContext &ctx) -> Result<void, std::string> {
        if (desc.prefab.configureRender) {
          desc.prefab.configureRender(ctx.runtime);
        } else {
          ctx.runtime.configureStaticModelOpaqueSettings(
              desc.prefab.lodThresholds);
        }
        *configured = false;
        stagingError->clear();
        animation->rootNode = kInvalidNodeId;
        animation->instantiationMap = SceneInstantiationMap{};
        return prepareImportedPrefabSceneResources(
            ctx.runtime, desc.prefab.info.label, desc.prefab.sourcePath,
            desc.prefab.importOptions, *assets);
      },
      .deactivate =
          [assets, animation, configured,
           stagingError](EditorSceneDeactivateContext &ctx) {
            ctx.runtime.destroyAnimatedPrefabSceneInstance(*animation);
            assets->release();
            *configured = false;
            stagingError->clear();
          },
      .update =
          [desc, assets, animation, configured,
           stagingError](EditorSceneUpdateContext &ctx) {
            if (*configured || !isValid(assets->sceneLoad)) {
              return;
            }
            applyPendingPrefabRootTransform(ctx.runtime, assets->sceneLoad,
                                            desc.prefab.baseModel);
            auto refreshed = refreshImportedPrefabSceneResources(
                ctx.runtime, desc.prefab.info.label, *assets);
            if (refreshed.hasError()) {
              NURI_LOG_ERROR("%s: async animated prefab load failed: %s",
                             desc.prefab.info.label.c_str(),
                             refreshed.error().c_str());
              *stagingError = refreshed.error();
              *configured = true;
              return;
            }
            if (!refreshed.value()) {
              return;
            }
            applyPrefabRootTransform(ctx.runtime, *assets,
                                     desc.prefab.baseModel);
            auto instantiation = ctx.runtime.assets().tryGetSceneInstantiation(
                assets->sceneLoad);
            if (!instantiation.has_value() ||
                !firstInstantiatedPrefabRenderable(ctx.runtime, *assets)
                     .has_value()) {
              NURI_LOG_ERROR("%s: animated scene prefab instantiation failed",
                             desc.prefab.info.label.c_str());
              *stagingError = "animated scene prefab instantiation failed";
              *configured = true;
              return;
            }
            animation->instantiationMap = std::move(*instantiation);
            for (uint32_t nodeIndex = 0u;
                 nodeIndex < assets->prefab.nodes.size(); ++nodeIndex) {
              if (assets->prefab.nodes[nodeIndex].parentIndex ==
                      kInvalidScenePrefabIndex &&
                  nodeIndex < animation->instantiationMap.nodes.size() &&
                  isValid(animation->instantiationMap.nodes[nodeIndex])) {
                animation->rootNode =
                    animation->instantiationMap.nodes[nodeIndex];
                break;
              }
            }

            std::vector<std::string_view> preferredNames;
            preferredNames.reserve(desc.preferredClipNames.size());
            for (const std::string &name : desc.preferredClipNames) {
              preferredNames.push_back(name);
            }
            AnimationPoseSimulationParams params{};
            params.primary.clipIndex = ctx.runtime.selectPreferredClipIndex(
                assets->prefab, preferredNames);
            params.primary.timeSeconds = 0.0f;
            params.primary.playbackMode = AnimationPosePlaybackMode::Loop;
            params.primary.playing = true;
            if (desc.initialBlendWeight > 0.0f &&
                !desc.secondaryPreferredClipNames.empty()) {
              std::vector<std::string_view> secondaryPreferredNames;
              secondaryPreferredNames.reserve(
                  desc.secondaryPreferredClipNames.size());
              for (const std::string &name : desc.secondaryPreferredClipNames) {
                secondaryPreferredNames.push_back(name);
              }
              params.secondary.clipIndex = ctx.runtime.selectPreferredClipIndex(
                  assets->prefab, secondaryPreferredNames);
              params.secondary.timeSeconds = 0.0f;
              params.secondary.playbackMode = AnimationPosePlaybackMode::Loop;
              params.secondary.playing = true;
              params.blendWeight =
                  glm::clamp(desc.initialBlendWeight, 0.0f, 1.0f);
              params.blendMode = AnimationPoseBlendMode::Lerp;
              params.blendSyncMode = desc.blendSyncMode;
            }
            ctx.runtime.startAnimatedPrefabSceneSimulation(
                desc.prefab.info.label, *assets, *animation, params,
                desc.simulationDebugName);

            if (const auto fallback = firstResolvedPrefabRenderable(*assets);
                fallback.has_value()) {
              const BoundingBox bounds = chooseBounds(
                  ctx.runtime, *assets, desc.prefab, fallback->first);
              if (desc.prefab.configureCamera) {
                desc.prefab.configureCamera(ctx.runtime, *assets, bounds);
              }
            }
            if (assets->prefab.lights.empty()) {
              ctx.runtime.finalizeSceneLighting(assets->fallbackLights,
                                                desc.prefab.baseModel);
            }
            *configured = true;
          },
      .stagingStatus =
          [configured, stagingError] {
            return EditorSceneStagingStatus{
                .ready = *configured && stagingError->empty(),
                .failed = !stagingError->empty(),
                .progress = *configured ? 1.0f : 0.0f,
                .error = *stagingError,
            };
          },
  };
}

EditorSceneSpec makeInstancedModelScene(EditorSceneSpec spec) { return spec; }

EditorSceneSpec makeStreamingScene(StreamingSceneFactoryDesc desc) {
  auto state = std::make_shared<StreamingSceneState>();
  return EditorSceneSpec{
      .info = desc.info,
      .prepare = [desc, state](EditorScenePrepareContext &ctx)
          -> Result<void, std::string> {
        if (!state->sourcePath.empty()) {
          return Result<void, std::string>::makeResult();
        }
        // StreamingSceneState::release observes AssetSystem through this
        // pointer, and EditorRuntime outlives the shared scene state.
        state->assets = &ctx.runtime.assets();
        state->sourcePath = desc.sourcePath;
        queueSceneTextureArtifactBakeIfNeeded(ctx.runtime, *state);
        return Result<void, std::string>::makeResult();
      },
      .activate = [desc, state](EditorSceneActivateContext &ctx)
          -> Result<void, std::string> {
        state->loadFailed = false;
        state->loadError.clear();
        if (desc.configureRender) {
          desc.configureRender(ctx.runtime);
        }
        auto requested = ctx.runtime.assets().requestScene(SceneLoadRequest{
            .path = state->sourcePath.string(),
            .importOptions =
                SceneImportOptions{.assetBuildOptions = desc.importOptions},
            .priority = AssetPriority::Critical,
            .publication = ScenePublicationPolicy::CompleteOnly,
            .failurePolicy = SceneFailurePolicy::BestEffort,
            .publicationTarget = ctx.runtime.scenePublicationTarget(),
            .debugName = desc.info.label,
        });
        if (requested.hasError()) {
          state->loadFailed = true;
          state->loadError = requested.error();
          return Result<void, std::string>::makeError(state->loadError);
        }
        state->sceneLoad = requested.value();
        state->configured = false;
        state->baseConfigured = false;
        state->populationComplete = !desc.populateLoadedSceneStep;
        state->cameraConfigured = false;
        state->lightingConfigured = false;
        state->model = kInvalidModelRef;
        state->renderableId = kInvalidRenderableId;
        state->baseModel = glm::mat4(1.0f);
        state->populationCursor = 0u;
        state->populationProgress = 0.0f;
        state->graphCapacityReservation = SceneGraphCapacityReservation{
            .nodeCapacity = desc.stagingNodeCapacity,
            .renderableCapacity = desc.stagingRenderableCapacity,
            .stage = desc.stagingNodeCapacity == 0u &&
                             desc.stagingRenderableCapacity == 0u
                         ? SceneGraphCapacityReservation::kStageCount
                         : 0u,
        };
        state->loadStartTimeSeconds = ctx.runtime.timeSeconds();
        state->lastProgressLogTimeSeconds = state->loadStartTimeSeconds;
        return Result<void, std::string>::makeResult();
      },
      .deactivate =
          [state](EditorSceneDeactivateContext &ctx) {
            if (isValid(state->sceneLoad)) {
              ctx.runtime.assets().cancel(state->sceneLoad);
              state->sceneLoad = {};
            }
            state->configured = false;
            state->baseConfigured = false;
            state->populationComplete = false;
            state->cameraConfigured = false;
            state->lightingConfigured = false;
            state->model = kInvalidModelRef;
            state->renderableId = kInvalidRenderableId;
          },
      .update =
          [desc, state](EditorSceneUpdateContext &ctx) {
            if (state->configured || !isValid(state->sceneLoad)) {
              return;
            }
            const SceneLoadSnapshot load =
                ctx.runtime.assets().query(state->sceneLoad);
            if (load.state == SceneLoadState::Failed ||
                load.state == SceneLoadState::Cancelled) {
              state->loadFailed = true;
              state->loadError =
                  load.error.empty() ? "async scene load failed" : load.error;
              NURI_LOG_ERROR("%s: %s", desc.info.label.c_str(),
                             state->loadError.c_str());
              return;
            }
            const ScenePrefab *prefab =
                ctx.runtime.assets().tryGetScenePrefab(state->sceneLoad);
            auto instantiation =
                ctx.runtime.assets().tryGetSceneInstantiation(state->sceneLoad);
            if (prefab == nullptr || !instantiation.has_value()) {
              return;
            }
            RenderableId firstRenderable = kInvalidRenderableId;
            ModelRef firstModel = kInvalidModelRef;
            for (const RenderableId renderable : instantiation->renderables) {
              if (!isValid(renderable) ||
                  !ctx.runtime.scene().graph().getRenderableModel(renderable,
                                                                  firstModel) ||
                  !isValid(firstModel)) {
                continue;
              }
              firstRenderable = renderable;
              break;
            }
            if (!isValid(firstRenderable)) {
              if (load.terminal()) {
                state->loadFailed = true;
                state->loadError = "async scene completed without a renderable";
                NURI_LOG_ERROR("%s: %s", desc.info.label.c_str(),
                               state->loadError.c_str());
              }
              return;
            }

            if (!state->baseConfigured) {
              const std::string sceneInstanceName = desc.instanceName.empty()
                                                        ? desc.info.label
                                                        : desc.instanceName;
              const std::string modelError =
                  "Model for " + sceneInstanceName + " is not loaded";
              const std::string recordError =
                  sceneInstanceName + " model record lookup failed";
              const Model &model = ctx.runtime.requireLoadedModel(
                  firstModel, modelError.c_str(), recordError.c_str());
              state->baseModel = glm::mat4(1.0f);
              if (desc.scaleToNiagaraBistro) {
                const float scale = computeNiagaraBistroScale(model.bounds());
                state->baseModel =
                    glm::scale(glm::mat4(1.0f), glm::vec3(scale));
              }
              for (uint32_t nodeIndex = 0u; nodeIndex < prefab->nodes.size();
                   ++nodeIndex) {
                if (prefab->nodes[nodeIndex].parentIndex !=
                        kInvalidScenePrefabIndex ||
                    nodeIndex >= instantiation->nodes.size() ||
                    !isValid(instantiation->nodes[nodeIndex])) {
                  continue;
                }
                (void)ctx.runtime.scene().graph().setNodeLocalTransform(
                    instantiation->nodes[nodeIndex],
                    state->baseModel *
                        prefab->nodes[nodeIndex].localFromParent);
              }
              state->model = firstModel;
              state->renderableId = firstRenderable;
              if (desc.configureLoadedScene) {
                desc.configureLoadedScene(ctx.runtime, *state);
              }
              state->baseConfigured = true;
            }
            if (!state->populationComplete && desc.populateLoadedSceneStep) {
              if (!state->graphCapacityReservation.complete()) {
                auto reserved = ctx.runtime.scene().graph().reserveCapacityStep(
                    state->graphCapacityReservation, 1u);
                if (reserved.hasError()) {
                  state->loadFailed = true;
                  state->loadError = reserved.error();
                  NURI_LOG_ERROR("%s: %s", desc.info.label.c_str(),
                                 state->loadError.c_str());
                  return;
                }
                state->populationProgress =
                    0.02f * state->graphCapacityReservation.progress();
                if (!reserved.value()) {
                  return;
                }
              }
              auto populate = desc.populateLoadedSceneStep(ctx.runtime, *state);
              if (populate.hasError()) {
                state->loadFailed = true;
                state->loadError = populate.error();
                NURI_LOG_ERROR("%s: %s", desc.info.label.c_str(),
                               state->loadError.c_str());
                return;
              }
              if (!populate.value()) {
                return;
              }
              state->populationComplete = true;
              return;
            }
            if (!state->cameraConfigured) {
              if (desc.configureCamera) {
                desc.configureCamera(ctx.runtime, *state);
              }
              state->cameraConfigured = true;
              return;
            }
            if (!state->lightingConfigured) {
              if (prefab->lights.empty()) {
                ctx.runtime.finalizeSceneLighting({}, state->baseModel);
              }
              state->lightingConfigured = true;
              return;
            }
            state->configured = true;
          },
      .stagingStatus =
          [state] {
            return EditorSceneStagingStatus{
                .ready = state->configured && !state->loadFailed,
                .failed = state->loadFailed,
                .progress = state->configured
                                ? 1.0f
                                : (state->baseConfigured
                                       ? std::clamp(state->populationProgress,
                                                    0.05f, 0.99f)
                                       : 0.0f),
                .error = state->loadError,
            };
          },
  };
}

} // namespace nuri
