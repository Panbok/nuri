#include "nuri/editor_pch.h"

#include "editor_scene_assets.h"

#include "nuri/app/editor_runtime.h"
#include "nuri/app/editor_scene_catalog.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/gpu/texture.h"

namespace nuri {
namespace {

constexpr std::string_view kSampleDuckModelRelativePath =
    "rubber_duck/scene.gltf";
constexpr std::string_view kSampleDuckAlbedoRelativePath =
    "rubber_duck/textures/Duck_baseColor.png";
constexpr std::string_view kNiagaraBistroModelRelativePath =
    "NiagaraBistro/bistrox.gltf";
constexpr std::string_view kDamagedHelmetModelRelativePath =
    "DamagedHelmet/DamagedHelmet.gltf";
constexpr std::string_view kLightsPunctualLampModelRelativePath =
    "LightsPunctualLamp/LightsPunctualLamp.gltf";
constexpr std::string_view kClearcoatWickerModelRelativePath =
    "ClearcoatWicker/ClearcoatWicker.gltf";
constexpr std::string_view kSheenChairModelRelativePath =
    "SheenChair/SheenChair.gltf";
constexpr std::string_view kSpecularSilkPoufModelRelativePath =
    "SpecularSilkPouf/SpecularSilkPouf.gltf";
constexpr std::string_view kDragonAttenuationModelRelativePath =
    "DragonAttenuation/DragonAttenuation.gltf";
constexpr std::string_view kDragonIorModelRelativePath =
    "DragonAttenuation/DragonIor.gltf";
constexpr std::string_view kEmissiveStrengthTestModelRelativePath =
    "EmissiveStrengthTest/EmissiveStrengthTest.gltf";
constexpr std::string_view kOrreyModelRelativePath = "Orrey/scene.gltf";
constexpr std::string_view kFoxModelRelativePath = "Fox/Fox.gltf";
constexpr std::string_view kMedievalFantasyBookModelRelativePath =
    "MedievalFantasyBook/scene.gltf";
constexpr uint32_t kDuckGridSide = 32;
constexpr uint32_t kDuckInstanceCount =
    kDuckGridSide * kDuckGridSide * kDuckGridSide;

[[nodiscard]] std::filesystem::path modelPath(const RuntimeConfig &config,
                                              std::string_view relativePath) {
  return config.roots.models / std::filesystem::path(relativePath);
}

[[nodiscard]] glm::mat4 dragonBaseModel() {
  return glm::rotate(glm::mat4(1.0f), glm::radians(180.0f),
                     glm::vec3(0.0f, 1.0f, 0.0f));
}

void configureStaticOpaqueScene(EditorRuntime &runtime,
                                const glm::vec3 &lodThresholds,
                                bool enableDebugGrid = false) {
  runtime.configureStaticModelOpaqueSettings(lodThresholds);
  if (enableDebugGrid) {
    runtime.renderSettings().debug.enabled = true;
    runtime.renderSettings().debug.grid = true;
  }
}

float hashToUnitFloat(uint32_t value) {
  uint32_t x = value;
  x ^= x >> 17u;
  x *= 0xed5ad4bbu;
  x ^= x >> 11u;
  x *= 0xac4c1b51u;
  x ^= x >> 15u;
  x *= 0x31848babu;
  x ^= x >> 14u;
  return static_cast<float>(x & 0x00ffffffu) / 16777215.0f;
}

[[nodiscard]] glm::vec3 instancePositionFromGrid(uint32_t index) {
  constexpr float kDuckSpacing = 18.0f;
  constexpr float kDuckJitter = 3.0f;
  const uint32_t x = index % kDuckGridSide;
  const uint32_t y = (index / kDuckGridSide) % kDuckGridSide;
  const uint32_t z = index / (kDuckGridSide * kDuckGridSide);
  const glm::vec3 centered =
      glm::vec3(static_cast<float>(x), static_cast<float>(y),
                static_cast<float>(z)) -
      glm::vec3((static_cast<float>(kDuckGridSide) - 1.0f) * 0.5f);
  glm::vec3 pos = centered * kDuckSpacing;
  pos +=
      glm::vec3((hashToUnitFloat(index * 3u + 1u) - 0.5f) * 2.0f * kDuckJitter,
                (hashToUnitFloat(index * 3u + 2u) - 0.5f) * 2.0f * kDuckJitter,
                (hashToUnitFloat(index * 3u + 3u) - 0.5f) * 2.0f * kDuckJitter);
  return pos;
}

[[nodiscard]] Result<void, std::string>
appendPrefab(EditorSceneCatalog &catalog, PrefabSceneFactoryDesc desc) {
  return catalog.append(makePrefabScene(std::move(desc)));
}

} // namespace

Result<void, std::string> registerBuiltInScenes(EditorSceneCatalog &catalog,
                                                const RuntimeConfig &config) {
  MeshImportOptions flipUvImport{};
  flipUvImport.flipUVs = true;
  MeshImportOptions dragonImportOptions{};
  dragonImportOptions.flipUVs = false;

  auto duckAssets = std::make_shared<SimpleModelSceneAssets>();
  auto prepareDuckScene =
      [duckAssets,
       config](EditorScenePrepareContext &ctx) -> Result<void, std::string> {
    if (duckAssets->ready) {
      return Result<void, std::string>::makeResult();
    }
    duckAssets->resources = &ctx.runtime.resources();
    duckAssets->sourcePath = modelPath(config, kSampleDuckModelRelativePath);
    auto modelResult = ctx.runtime.resources().acquireModel(ModelRequest{
        .path = duckAssets->sourcePath.string(),
        .debugName = "rubber_duck",
    });
    if (modelResult.hasError()) {
      return Result<void, std::string>::makeError(modelResult.error());
    }
    duckAssets->model = modelResult.value();

    auto albedoResult = ctx.runtime.resources().acquireTexture(TextureRequest{
        .path = (config.roots.models / kSampleDuckAlbedoRelativePath).string(),
        .loadOptions =
            TextureLoadOptions{.srgb = true, .generateMipmaps = true},
        .kind = TextureRequestKind::Texture2D,
        .debugName = "duck_albedo",
    });
    if (albedoResult.hasError()) {
      return Result<void, std::string>::makeError(albedoResult.error());
    }

    const TextureRecord *duckAlbedoRecord =
        ctx.runtime.resources().tryGet(albedoResult.value());
    NURI_ASSERT(duckAlbedoRecord != nullptr,
                "Duck albedo record lookup failed");
    MaterialDesc duckMaterialDesc{};
    duckMaterialDesc.textures.baseColor = duckAlbedoRecord->texture;
    auto materialResult =
        ctx.runtime.resources().acquireMaterial(MaterialRequest{
            .desc = duckMaterialDesc,
            .textureRefs =
                MaterialRequest::TextureRefs{.baseColor = albedoResult.value()},
            .debugName = "duck_material",
        });
    ctx.runtime.resources().release(albedoResult.value());
    if (materialResult.hasError()) {
      return Result<void, std::string>::makeError(materialResult.error());
    }
    duckAssets->material = materialResult.value();
    ctx.runtime.resources().setModelMaterialForAllSources(duckAssets->model,
                                                          duckAssets->material);
    loadImportedLightsForScene("Rubber Duck", duckAssets->sourcePath.string(),
                               duckAssets->fallbackLights);
    duckAssets->ready = true;
    queueScenePortableBakeIfNeeded(ctx.runtime, *duckAssets);
    return Result<void, std::string>::makeResult();
  };

  {
    auto result = catalog.append(makeCustomScene(EditorSceneSpec{
        .info = {.id = "single_duck", .label = "Single Duck"},
        .prepare = prepareDuckScene,
        .activate = [duckAssets](EditorSceneActivateContext &ctx)
            -> Result<void, std::string> {
          ctx.runtime.configureStaticModelOpaqueSettings(
              glm::vec3(8.0f, 16.0f, 32.0f));
          const RenderableId duckRenderable = ctx.runtime.addRequiredRenderable(
              duckAssets->model, duckAssets->material, glm::mat4(1.0f),
              "Failed to add duck renderable");
          if (!isValid(duckRenderable)) {
            return Result<void, std::string>::makeError(
                "Failed to add duck renderable");
          }
          Camera *camera = ctx.runtime.mainCamera();
          NURI_ASSERT(camera != nullptr, "Failed to get main camera");
          camera->setLookAt(glm::vec3(0.0f, 1.0f, -1.5f),
                            glm::vec3(0.0f, 0.5f, 0.0f),
                            glm::vec3(0.0f, 1.0f, 0.0f));
          ctx.runtime.syncEditorCameraWidgetState(*camera);
          ctx.runtime.finalizeSceneLighting(duckAssets->fallbackLights,
                                            glm::mat4(1.0f));
          return Result<void, std::string>::makeResult();
        },
    }));
    if (result.hasError()) {
      return result;
    }
  }

  {
    auto result =
        catalog.append(makeInstancedModelScene(makeCustomScene(EditorSceneSpec{
            .info = {.id = "instanced_duck_32k", .label = "Instanced Duck 32K"},
            .prepare = prepareDuckScene,
            .activate = [duckAssets](EditorSceneActivateContext &ctx)
                -> Result<void, std::string> {
              ctx.runtime.renderSettings().opaque.enableInstanceCompute = true;
              ctx.runtime.renderSettings().opaque.enableDepthPrepass = false;
              ctx.runtime.renderSettings().opaque.enableMeshLod = true;
              ctx.runtime.renderSettings().opaque.forcedMeshLod = -1;
              ctx.runtime.renderSettings().opaque.meshLodDistanceThresholds =
                  glm::vec3(4.0f, 8.0f, 16.0f);
              ctx.runtime.renderSettings().opaque.enableInstanceAnimation =
                  true;
              std::vector<glm::mat4> transforms;
              transforms.reserve(kDuckInstanceCount);
              for (uint32_t i = 0; i < kDuckInstanceCount; ++i) {
                transforms.push_back(glm::translate(
                    glm::mat4(1.0f), instancePositionFromGrid(i)));
              }
              auto addResult =
                  ctx.runtime.scene().graph().addRenderablesInstanced(
                      duckAssets->model, duckAssets->material, transforms);
              if (addResult.hasError()) {
                return Result<void, std::string>::makeError(addResult.error());
              }
              Camera *camera = ctx.runtime.mainCamera();
              NURI_ASSERT(camera != nullptr, "Failed to get main camera");
              camera->setLookAt(glm::vec3(0.0f, 120.0f, -760.0f),
                                glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
              ctx.runtime.syncEditorCameraWidgetState(*camera);
              ctx.runtime.finalizeSceneLighting({}, glm::mat4(1.0f));
              return Result<void, std::string>::makeResult();
            },
        })));
    if (result.hasError()) {
      return result;
    }
  }

  {
    auto result = catalog.append(makeStreamingScene({
        .info = {.id = "niagara_bistro", .label = "Niagara Bistro"},
        .sourcePath = modelPath(config, kNiagaraBistroModelRelativePath),
        .instanceName = "NiagaraBistro",
        .fallbackMaterialDebugName = "niagara_bistro_fallback_material",
        .configureRender =
            [](EditorRuntime &runtime) {
              runtime.renderSettings().opaque.enableInstanceCompute = false;
              runtime.renderSettings().opaque.enableMeshLod = false;
              runtime.renderSettings().opaque.enableTessellation = false;
              runtime.renderSettings().opaque.forcedMeshLod = 0;
              runtime.renderSettings().opaque.meshLodDistanceThresholds =
                  glm::vec3(8.0f, 24.0f, 48.0f);
              runtime.renderSettings().opaque.enableInstanceAnimation = false;
              runtime.renderSettings().textureFiltering.mode =
                  TextureFilterMode::Anisotropic;
              runtime.renderSettings().textureFiltering.anisotropy = 8u;
            },
        .configureCamera =
            [](EditorRuntime &runtime, StreamingSceneState &state) {
              const Model &model = runtime.requireLoadedModel(
                  state.model, "Niagara Bistro model is not loaded",
                  "Niagara Bistro model record lookup failed");
              const BoundingBox bounds = runtime
                                             .computeImportedPrefabBounds(
                                                 state.prefab, state.baseModel)
                                             .value_or(model.bounds());
              (void)runtime.frameSceneCamera(
                  bounds, state.baseModel, 1.8f, 50.0f,
                  glm::vec4(0.32f, 0.14f, 1.0f, 4.0f), glm::vec2(0.03f, 0.0f));
            },
    }));
    if (result.hasError()) {
      return result;
    }
  }

  auto result = appendPrefab(
      catalog,
      {
          .info = {.id = "damaged_helmet", .label = "Damaged Helmet"},
          .sourcePath = modelPath(config, kDamagedHelmetModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "DamagedHelmet",
          .lodThresholds = glm::vec3(8.0f, 16.0f, 32.0f),
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &bounds) {
                (void)runtime.frameSceneCamera(
                    bounds, glm::mat4(1.0f), 2.4f, 2.0f,
                    glm::vec4(0.38f, 0.18f, 1.0f, 0.2f),
                    glm::vec2(0.03f, 0.0f));
              },
      });
  if (result.hasError()) {
    return result;
  }

  result = appendPrefab(
      catalog,
      {
          .info = {.id = "lights_punctual_lamp",
                   .label = "Lights Punctual Lamp"},
          .sourcePath = modelPath(config, kLightsPunctualLampModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "LightsPunctualLamp",
          .configureRender =
              [](EditorRuntime &runtime) {
                configureStaticOpaqueScene(runtime,
                                           glm::vec3(6.0f, 12.0f, 24.0f), true);
              },
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &bounds) {
                (void)runtime.frameSceneCamera(
                    bounds, glm::mat4(1.0f), 2.5f, 2.0f,
                    glm::vec4(0.42f, 0.25f, 1.0f, 0.1f),
                    glm::vec2(0.08f, 0.0f));
              },
      });
  if (result.hasError()) {
    return result;
  }

  result = appendPrefab(
      catalog,
      {
          .info = {.id = "clearcoat_wicker", .label = "Clearcoat Wicker"},
          .sourcePath = modelPath(config, kClearcoatWickerModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "ClearcoatWicker",
          .configureRender =
              [](EditorRuntime &runtime) {
                runtime.configureStaticModelOpaqueSettings(
                    glm::vec3(8.0f, 16.0f, 32.0f));
              },
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &bounds) {
                (void)runtime.frameSceneCamera(
                    bounds, glm::mat4(1.0f), 2.4f, 2.0f,
                    glm::vec4(0.28f, 0.12f, 1.0f, 0.15f),
                    glm::vec2(0.03f, 0.0f));
              },
      });
  if (result.hasError()) {
    return result;
  }

  result = appendPrefab(
      catalog,
      {
          .info = {.id = "sheen_chair", .label = "Sheen Chair"},
          .sourcePath = modelPath(config, kSheenChairModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "SheenChair",
          .configureRender =
              [](EditorRuntime &runtime) {
                runtime.configureStaticModelOpaqueSettings(
                    glm::vec3(6.0f, 12.0f, 24.0f));
              },
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &bounds) {
                (void)runtime.frameSceneCamera(
                    bounds, glm::mat4(1.0f), 2.5f, 2.0f,
                    glm::vec4(0.52f, 0.52f, 1.0f, 0.0f), glm::vec2(0.1f, 0.0f));
              },
      });
  if (result.hasError()) {
    return result;
  }

  result = appendPrefab(
      catalog,
      {
          .info = {.id = "specular_silk_pouf", .label = "Specular Silk Pouf"},
          .sourcePath = modelPath(config, kSpecularSilkPoufModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "SpecularSilkPouf",
          .configureRender =
              [](EditorRuntime &runtime) {
                runtime.configureStaticModelOpaqueSettings(
                    glm::vec3(6.0f, 12.0f, 24.0f));
              },
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &bounds) {
                (void)runtime.frameSceneCamera(
                    bounds, glm::mat4(1.0f), 2.5f, 2.0f,
                    glm::vec4(0.52f, 0.52f, 1.0f, 0.0f), glm::vec2(0.1f, 0.0f));
              },
      });
  if (result.hasError()) {
    return result;
  }

  result = appendPrefab(
      catalog,
      {
          .info = {.id = "dragon_attenuation", .label = "Dragon Attenuation"},
          .sourcePath = modelPath(config, kDragonAttenuationModelRelativePath),
          .importOptions = dragonImportOptions,
          .instanceName = "DragonAttenuation",
          .baseModel = dragonBaseModel(),
          .lodThresholds = glm::vec3(6.0f, 12.0f, 24.0f),
          .requirePrefabInstantiation = true,
          .configureRender =
              [](EditorRuntime &runtime) {
                configureStaticOpaqueScene(runtime,
                                           glm::vec3(6.0f, 12.0f, 24.0f), true);
                runtime.renderSettings().skybox.enabled = false;
                runtime.renderSettings().transparent.enabled = true;
                runtime.renderSettings().transmission.enabled = true;
              },
          .computeBounds = [](EditorRuntime &runtime,
                              const ImportedPrefabSceneResources &assets)
              -> std::optional<BoundingBox> {
            return runtime.computeImportedPrefabNodeBounds(
                assets, dragonBaseModel(), "Dragon");
          },
          .configureCamera =
              [](EditorRuntime &runtime,
                 const ImportedPrefabSceneResources &assets,
                 const BoundingBox &bounds) {
                (void)runtime.configureDragonSampleCamera(
                    assets, dragonBaseModel(), bounds);
              },
      });
  if (result.hasError()) {
    return result;
  }

  result = appendPrefab(
      catalog,
      {
          .info = {.id = "dragon_ior", .label = "Dragon IOR"},
          .sourcePath = modelPath(config, kDragonIorModelRelativePath),
          .importOptions = dragonImportOptions,
          .instanceName = "DragonIor",
          .baseModel = dragonBaseModel(),
          .lodThresholds = glm::vec3(6.0f, 12.0f, 24.0f),
          .requirePrefabInstantiation = true,
          .configureRender =
              [](EditorRuntime &runtime) {
                configureStaticOpaqueScene(runtime,
                                           glm::vec3(6.0f, 12.0f, 24.0f), true);
                runtime.renderSettings().skybox.enabled = false;
                runtime.renderSettings().transparent.enabled = true;
                runtime.renderSettings().transmission.enabled = true;
              },
          .computeBounds = [](EditorRuntime &runtime,
                              const ImportedPrefabSceneResources &assets)
              -> std::optional<BoundingBox> {
            return runtime.computeImportedPrefabNodeBounds(
                assets, dragonBaseModel(), "Dragon");
          },
          .configureCamera =
              [](EditorRuntime &runtime,
                 const ImportedPrefabSceneResources &assets,
                 const BoundingBox &bounds) {
                (void)runtime.configureDragonSampleCamera(
                    assets, dragonBaseModel(), bounds);
              },
      });
  if (result.hasError()) {
    return result;
  }

  result = appendPrefab(
      catalog,
      {
          .info = {.id = "emissive_strength_test",
                   .label = "Emissive Strength Test"},
          .sourcePath =
              modelPath(config, kEmissiveStrengthTestModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "EmissiveStrengthTest",
          .configureRender =
              [](EditorRuntime &runtime) {
                runtime.configureStaticModelOpaqueSettings(
                    glm::vec3(6.0f, 12.0f, 24.0f));
                runtime.renderSettings().skybox.enabled = false;
              },
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &bounds) {
                (void)runtime.frameSceneCamera(
                    bounds, glm::mat4(1.0f), 2.5f, 2.0f,
                    glm::vec4(0.52f, 0.52f, 1.0f, 0.0f), glm::vec2(0.1f, 0.0f));
              },
      });
  if (result.hasError()) {
    return result;
  }

  result = appendPrefab(
      catalog,
      {
          .info = {.id = "orrey", .label = "Orrey"},
          .sourcePath = modelPath(config, kOrreyModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "Orrey",
          .configureRender =
              [](EditorRuntime &runtime) {
                runtime.configureStaticModelOpaqueSettings(
                    glm::vec3(12.0f, 24.0f, 48.0f));
                runtime.renderSettings().debug.enabled = true;
                runtime.renderSettings().debug.grid = true;
              },
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &bounds) {
                (void)runtime.frameSceneCamera(
                    bounds, glm::mat4(1.0f), 2.8f, 4.0f,
                    glm::vec4(0.42f, 0.20f, 1.0f, 0.35f),
                    glm::vec2(0.06f, 0.0f));
              },
      });
  if (result.hasError()) {
    return result;
  }

  result = catalog.append(makeAnimatedPrefabScene({
      .prefab = {
          .info = {.id = "fox",
                   .label = "Fox Skinning",
                   .initiallySelected = true},
          .sourcePath = modelPath(config, kFoxModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "Fox",
          .baseModel = glm::mat4(1.0f),
          .lodThresholds = glm::vec3(8.0f, 16.0f, 32.0f),
          .configureRender =
              [](EditorRuntime &runtime) {
                configureStaticOpaqueScene(runtime,
                                           glm::vec3(8.0f, 16.0f, 32.0f), true);
              },
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &bounds) {
                (void)runtime.frameSceneCamera(
                    bounds, glm::mat4(1.0f), 1.55f, 1.0f,
                    glm::vec4(0.80f, 0.12f, 0.20f, 0.10f),
                    glm::vec2(0.06f, 0.0f));
              },
      },
      .preferredClipNames = {"Run", "Walk"},
      .simulationDebugName = "FoxAnimation",
  }));
  if (result.hasError()) {
    return result;
  }

  result = catalog.append(makeAnimatedPrefabScene({
      .prefab = {
          .info = {.id = "medieval_fantasy_book",
                   .label = "Medieval Fantasy Book"},
          .sourcePath =
              modelPath(config, kMedievalFantasyBookModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "MedievalFantasyBook",
          .lodThresholds = glm::vec3(10.0f, 20.0f, 40.0f),
          .configureRender =
              [](EditorRuntime &runtime) {
                configureStaticOpaqueScene(
                    runtime, glm::vec3(10.0f, 20.0f, 40.0f), true);
                runtime.renderSettings().transparent.enabled = true;
                runtime.renderSettings().transmission.enabled = true;
              },
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &bounds) {
                (void)runtime.frameSceneCamera(
                    bounds, glm::mat4(1.0f), 2.3f, 2.0f,
                    glm::vec4(0.48f, 0.18f, 1.0f, 0.2f),
                    glm::vec2(0.06f, 0.0f));
              },
      },
      .preferredClipNames = {},
      .simulationDebugName = "MedievalFantasyBookAnimation",
  }));
  if (result.hasError()) {
    return result;
  }

  result = catalog.append(makeCustomScene(EditorSceneSpec{
      .info = {.id = "text_3d_test", .label = "Text 3D Test"},
      .activate =
          [](EditorSceneActivateContext &ctx) -> Result<void, std::string> {
        ctx.runtime.setupText3DTestScene();
        ctx.runtime.finalizeSceneLighting({}, glm::mat4(1.0f));
        return Result<void, std::string>::makeResult();
      },
  }));
  if (result.hasError()) {
    return result;
  }

  return Result<void, std::string>::makeResult();
}

} // namespace nuri
