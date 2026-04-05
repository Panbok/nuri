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
    if (renderable.meshIndex < assets.assets.models.size() &&
        renderable.materialIndex < assets.assets.materials.size()) {
      const ModelRef model = assets.assets.models[renderable.meshIndex];
      const MaterialRef material =
          assets.assets.materials[renderable.materialIndex];
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

} // namespace

EditorSceneSpec makeCustomScene(EditorSceneSpec spec) { return spec; }

EditorSceneSpec makePrefabScene(PrefabSceneFactoryDesc desc) {
  auto assets = std::make_shared<ImportedPrefabSceneResources>();
  return EditorSceneSpec{
      .info = desc.info,
      .prepare = [desc, assets](EditorScenePrepareContext &ctx)
          -> Result<void, std::string> {
        auto prepareResult = prepareImportedPrefabSceneResources(
            ctx.runtime, desc.info.label, desc.sourcePath, desc.importOptions,
            *assets);
        if (prepareResult.hasError()) {
          return prepareResult;
        }
        queueScenePortableBakeIfNeeded(ctx.runtime, *assets);
        return Result<void, std::string>::makeResult();
      },
      .activate = [desc, assets](EditorSceneActivateContext &ctx)
          -> Result<void, std::string> {
        if (desc.configureRender) {
          desc.configureRender(ctx.runtime);
        } else {
          ctx.runtime.configureStaticModelOpaqueSettings(desc.lodThresholds);
        }

        RenderableId renderableId = ctx.runtime.instantiateImportedPrefabScene(
            desc.instanceName.empty() ? desc.info.label : desc.instanceName,
            *assets, desc.baseModel);
        if (!isValid(renderableId)) {
          if (desc.requirePrefabInstantiation) {
            return Result<void, std::string>::makeError(
                "Scene prefab instantiation failed");
          }
          const auto fallback = firstResolvedPrefabRenderable(*assets);
          if (!fallback.has_value()) {
            return Result<void, std::string>::makeError(
                "Scene prefab did not resolve any renderable fallback");
          }
          renderableId = ctx.runtime.addRequiredRenderable(
              fallback->first, fallback->second, desc.baseModel,
              "Failed to add scene fallback renderable");
        }
        const auto fallback = firstResolvedPrefabRenderable(*assets);
        if (fallback.has_value()) {
          const BoundingBox bounds =
              chooseBounds(ctx.runtime, *assets, desc, fallback->first);
          FramedSceneCameraState cameraState{};
          if (desc.configureCamera) {
            desc.configureCamera(ctx.runtime, *assets, bounds);
          } else {
            cameraState = ctx.runtime.frameSceneCamera(
                bounds, desc.baseModel, 2.5f, 2.0f,
                glm::vec4(0.42f, 0.20f, 1.0f, 0.2f), glm::vec2(0.06f, 0.0f));
            ctx.runtime.logSingleRenderableSceneStats(
                desc.info.label,
                ctx.runtime.requireLoadedModel(
                    fallback->first, "Scene model is not loaded",
                    "Scene model record lookup failed"),
                cameraState);
          }
        }

        ctx.runtime.finalizeSceneLighting(assets->fallbackLights,
                                          desc.baseModel);
        return Result<void, std::string>::makeResult();
      },
  };
}

EditorSceneSpec makeAnimatedPrefabScene(AnimatedPrefabSceneFactoryDesc desc) {
  auto assets = std::make_shared<ImportedPrefabSceneResources>();
  auto animation = std::make_shared<AnimatedPrefabSceneState>();
  return EditorSceneSpec{
      .info = desc.prefab.info,
      .prepare = [prefab = desc.prefab, assets](EditorScenePrepareContext &ctx)
          -> Result<void, std::string> {
        auto prepareResult = prepareImportedPrefabSceneResources(
            ctx.runtime, prefab.info.label, prefab.sourcePath,
            prefab.importOptions, *assets);
        if (prepareResult.hasError()) {
          return prepareResult;
        }
        queueScenePortableBakeIfNeeded(ctx.runtime, *assets);
        return Result<void, std::string>::makeResult();
      },
      .activate = [desc, assets, animation](EditorSceneActivateContext &ctx)
          -> Result<void, std::string> {
        if (desc.prefab.configureRender) {
          desc.prefab.configureRender(ctx.runtime);
        } else {
          ctx.runtime.configureStaticModelOpaqueSettings(
              desc.prefab.lodThresholds);
        }

        RenderableId renderableId = ctx.runtime.instantiateImportedPrefabScene(
            desc.prefab.instanceName.empty() ? desc.prefab.info.label
                                             : desc.prefab.instanceName,
            *assets, desc.prefab.baseModel, &animation->instantiationMap,
            &animation->rootNode);
        if (!isValid(renderableId)) {
          return Result<void, std::string>::makeError(
              "Animated scene prefab instantiation failed");
        }

        std::vector<std::string_view> preferredNames;
        preferredNames.reserve(desc.preferredClipNames.size());
        for (const std::string &name : desc.preferredClipNames) {
          preferredNames.push_back(name);
        }
        const uint32_t clipIndex = ctx.runtime.selectPreferredClipIndex(
            assets->prefab, preferredNames);
        ctx.runtime.startAnimatedPrefabSceneSimulation(
            desc.prefab.info.label, *assets, *animation, clipIndex,
            desc.simulationDebugName);

        if (const auto fallback = firstResolvedPrefabRenderable(*assets);
            fallback.has_value()) {
          const BoundingBox bounds =
              chooseBounds(ctx.runtime, *assets, desc.prefab, fallback->first);
          if (desc.prefab.configureCamera) {
            desc.prefab.configureCamera(ctx.runtime, *assets, bounds);
          }
        }
        ctx.runtime.finalizeSceneLighting(assets->fallbackLights,
                                          desc.prefab.baseModel);
        return Result<void, std::string>::makeResult();
      },
      .deactivate =
          [animation](EditorSceneDeactivateContext &ctx) {
            ctx.runtime.destroyAnimatedPrefabSceneInstance(*animation);
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
        state->resources = &ctx.runtime.resources();
        state->sourcePath = desc.sourcePath;
        auto materialResult = ctx.runtime.resources().acquireMaterial(
            MaterialRequest{.desc = MaterialDesc{},
                            .debugName = desc.fallbackMaterialDebugName});
        if (materialResult.hasError()) {
          return Result<void, std::string>::makeError(materialResult.error());
        }
        state->material = materialResult.value();
        auto prefabResult = prepareImportedPrefabSceneResources(
            ctx.runtime, desc.info.label, desc.sourcePath, MeshImportOptions{},
            state->prefab);
        if (prefabResult.hasError()) {
          return prefabResult;
        }
        queueScenePortableBakeIfNeeded(ctx.runtime, *state);
        return Result<void, std::string>::makeResult();
      },
      .activate = [desc, state](EditorSceneActivateContext &ctx)
          -> Result<void, std::string> {
        if (desc.configureRender) {
          desc.configureRender(ctx.runtime);
        }
        if ((!state->asyncLoad.has_value() || !state->asyncLoad->valid()) &&
            !isValid(state->model) && !state->loadFailed) {
          auto asyncLoadResult = Model::createFromFileAsync(
              state->sourcePath.string(), MeshImportOptions{});
          if (asyncLoadResult.hasError()) {
            state->loadFailed = true;
            state->loadError = asyncLoadResult.error();
            return Result<void, std::string>::makeError(state->loadError);
          }
          state->asyncLoad = std::move(asyncLoadResult.value());
          state->loadStartTimeSeconds = ctx.runtime.timeSeconds();
          state->lastProgressLogTimeSeconds = state->loadStartTimeSeconds;
        }
        return Result<void, std::string>::makeResult();
      },
      .deactivate =
          [state](EditorSceneDeactivateContext &) {
            state->renderableId = kInvalidRenderableId;
          },
      .update =
          [desc, state](EditorSceneUpdateContext &ctx) {
            if (isValid(state->renderableId) || !state->asyncLoad.has_value() ||
                !state->asyncLoad->valid()) {
              return;
            }
            if (!state->asyncLoad->isReady()) {
              return;
            }
            auto warmupResult = state->asyncLoad->resolveWarmup();
            (void)warmupResult;

            auto modelResult =
                ctx.runtime.resources().acquireModel(ModelRequest{
                    .path = state->sourcePath.string(),
                    .debugName = std::string(desc.instanceName),
                });
            if (modelResult.hasError()) {
              state->loadFailed = true;
              state->loadError = modelResult.error();
              state->asyncLoad.reset();
              return;
            }
            state->asyncLoad.reset();
            state->model = modelResult.value();
            ctx.runtime.resources().setModelMaterialForAllSources(
                state->model, state->material);

            const Model &model = ctx.runtime.requireLoadedModel(
                state->model, "Niagara Bistro model is not loaded",
                "Niagara Bistro model record lookup failed");
            const float scale = computeNiagaraBistroScale(model.bounds());
            state->baseModel = glm::scale(glm::mat4(1.0f), glm::vec3(scale));
            state->renderableId = ctx.runtime.instantiateImportedPrefabScene(
                desc.instanceName, state->prefab, state->baseModel);
            if (!isValid(state->renderableId)) {
              state->renderableId = ctx.runtime.addRequiredRenderable(
                  state->model, state->material, state->baseModel,
                  "Failed to add Niagara Bistro renderable");
            }
            if (desc.configureCamera) {
              desc.configureCamera(ctx.runtime, *state);
            }
            ctx.runtime.finalizeSceneLighting(state->prefab.fallbackLights,
                                              state->baseModel);
          },
  };
}

} // namespace nuri
