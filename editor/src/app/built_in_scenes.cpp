#include "nuri/app/editor_scene_spec.h"
#include "nuri/editor_pch.h"

#include "editor_scene_assets.h"

#include "nuri/app/editor_runtime.h"
#include "nuri/app/editor_scene_catalog.h"
#include "nuri/core/log.h"
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
constexpr std::string_view kAnimatedMorphCubeModelRelativePath =
    "AnimatedMorphCube/AnimatedMorphCube.gltf";
constexpr std::string_view kSponzaModelRelativePath = "Sponza/Sponza.gltf";
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

[[nodiscard]] glm::mat4 sponzaBaseModel() {
  constexpr glm::vec3 kViewFocus(820.0f, 270.0f, -20.0f);
  constexpr float kScale = 2.2f;
  return glm::translate(glm::mat4(1.0f), kViewFocus) *
         glm::scale(glm::mat4(1.0f), glm::vec3(kScale)) *
         glm::translate(glm::mat4(1.0f), -kViewFocus);
}

[[nodiscard]] glm::quat rotationFromLightDirection(glm::vec3 direction) {
  const float length = glm::length(direction);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }
  direction /= length;
  const glm::vec3 up = std::abs(direction.y) < 0.99f
                           ? glm::vec3(0.0f, 1.0f, 0.0f)
                           : glm::vec3(0.0f, 0.0f, 1.0f);
  const glm::mat4 view = glm::lookAt(glm::vec3(0.0f), direction, up);
  return glm::normalize(glm::quat_cast(glm::inverse(view)));
}

struct GroundPlanePlacement {
  float extent = 1.0f;
  float y = 0.0f;
};

struct AlphaMaskValidationSceneAssets {
  ResourceManager *resources = nullptr;
  ModelRef cardModel = kInvalidModelRef;
  ModelRef groundModel = kInvalidModelRef;
  TextureRef alphaTexture = kInvalidTextureRef;
  MaterialRef cardMaterial = kInvalidMaterialRef;
  MaterialRef groundMaterial = kInvalidMaterialRef;
  std::filesystem::path alphaTexturePath{};
  std::vector<ImportedSceneLight> fallbackLights{};
  bool ready = false;

  ~AlphaMaskValidationSceneAssets() { release(); }

  AlphaMaskValidationSceneAssets() = default;
  AlphaMaskValidationSceneAssets(const AlphaMaskValidationSceneAssets &) =
      delete;
  AlphaMaskValidationSceneAssets &
  operator=(const AlphaMaskValidationSceneAssets &) = delete;

  void release() noexcept {
    if (resources != nullptr) {
      if (isValid(cardModel)) {
        resources->release(cardModel);
      }
      if (isValid(groundModel)) {
        resources->release(groundModel);
      }
      if (isValid(cardMaterial)) {
        resources->release(cardMaterial);
      }
      if (isValid(groundMaterial)) {
        resources->release(groundMaterial);
      }
      if (isValid(alphaTexture)) {
        resources->release(alphaTexture);
      }
    }
    resources = nullptr;
    cardModel = kInvalidModelRef;
    groundModel = kInvalidModelRef;
    alphaTexture = kInvalidTextureRef;
    cardMaterial = kInvalidMaterialRef;
    groundMaterial = kInvalidMaterialRef;
    alphaTexturePath.clear();
    fallbackLights.clear();
    ready = false;
  }
};

[[nodiscard]] GroundPlanePlacement
damagedHelmetGroundPlacement(const BoundingBox &bounds) {
  const glm::vec3 size = bounds.getSize();
  return GroundPlanePlacement{
      .extent = std::max(std::max(size.x, size.z) * 3.0f, 4.0f),
      .y = bounds.min_.y - std::max(size.y * 0.005f, 0.0025f),
  };
}

[[nodiscard]] BoundingBox damagedHelmetCameraBounds(const BoundingBox &bounds) {
  const GroundPlanePlacement plane = damagedHelmetGroundPlacement(bounds);
  const glm::vec3 center = bounds.getCenter();
  const glm::vec3 size = bounds.getSize();
  const float framingExtent = std::max(std::max(size.x, size.z) * 0.9f, 1.0f);

  BoundingBox framedBounds = bounds;
  framedBounds.combinePoint(
      glm::vec3(center.x - framingExtent, plane.y, center.z - framingExtent));
  framedBounds.combinePoint(
      glm::vec3(center.x + framingExtent, plane.y, center.z + framingExtent));
  return framedBounds;
}

[[nodiscard]] MeshData makeGeneratedPlaneMeshData() {
  MeshData data(std::pmr::get_default_resource());
  data.name = "generated_flat_plane";
  data.vertices = {
      Vertex{.position = glm::vec3(-0.5f, 0.0f, -0.5f),
             .normal = glm::vec3(0.0f, 1.0f, 0.0f),
             .tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
             .uv = glm::vec2(0.0f, 0.0f)},
      Vertex{.position = glm::vec3(0.5f, 0.0f, -0.5f),
             .normal = glm::vec3(0.0f, 1.0f, 0.0f),
             .tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
             .uv = glm::vec2(1.0f, 0.0f)},
      Vertex{.position = glm::vec3(0.5f, 0.0f, 0.5f),
             .normal = glm::vec3(0.0f, 1.0f, 0.0f),
             .tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
             .uv = glm::vec2(1.0f, 1.0f)},
      Vertex{.position = glm::vec3(-0.5f, 0.0f, 0.5f),
             .normal = glm::vec3(0.0f, 1.0f, 0.0f),
             .tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
             .uv = glm::vec2(0.0f, 1.0f)},
  };
  data.indices = {0u, 2u, 1u, 0u, 3u, 2u};

  Submesh submesh{};
  submesh.vertexOffset = 0u;
  submesh.vertexCount = 4u;
  submesh.indexOffset = 0u;
  submesh.indexCount = 6u;
  submesh.materialIndex = 0u;
  submesh.bounds =
      BoundingBox(glm::vec3(-0.5f, 0.0f, -0.5f), glm::vec3(0.5f, 0.0f, 0.5f));
  submesh.authoredScale = glm::vec3(1.0f);
  submesh.lodCount = 1u;
  submesh.lods[0].indexOffset = submesh.indexOffset;
  submesh.lods[0].indexCount = submesh.indexCount;
  data.submeshes.push_back(submesh);
  return data;
}

[[nodiscard]] MeshData makeAlphaMaskCardMeshData() {
  MeshData data(std::pmr::get_default_resource());
  data.name = "generated_alpha_mask_card";
  data.vertices = {
      Vertex{.position = glm::vec3(-0.5f, 0.0f, 0.0f),
             .normal = glm::vec3(0.0f, 0.0f, 1.0f),
             .tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
             .uv = glm::vec2(0.0f, 1.0f)},
      Vertex{.position = glm::vec3(0.5f, 0.0f, 0.0f),
             .normal = glm::vec3(0.0f, 0.0f, 1.0f),
             .tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
             .uv = glm::vec2(1.0f, 1.0f)},
      Vertex{.position = glm::vec3(0.5f, 1.0f, 0.0f),
             .normal = glm::vec3(0.0f, 0.0f, 1.0f),
             .tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
             .uv = glm::vec2(1.0f, 0.0f)},
      Vertex{.position = glm::vec3(-0.5f, 1.0f, 0.0f),
             .normal = glm::vec3(0.0f, 0.0f, 1.0f),
             .tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
             .uv = glm::vec2(0.0f, 0.0f)},
  };
  data.indices = {0u, 1u, 2u, 0u, 2u, 3u};

  Submesh submesh{};
  submesh.vertexOffset = 0u;
  submesh.vertexCount = 4u;
  submesh.indexOffset = 0u;
  submesh.indexCount = 6u;
  submesh.materialIndex = 0u;
  submesh.bounds =
      BoundingBox(glm::vec3(-0.5f, 0.0f, 0.0f), glm::vec3(0.5f, 1.0f, 0.0f));
  submesh.authoredScale = glm::vec3(1.0f);
  submesh.lodCount = 1u;
  submesh.lods[0].indexOffset = submesh.indexOffset;
  submesh.lods[0].indexCount = submesh.indexCount;
  data.submeshes.push_back(submesh);
  return data;
}

[[nodiscard]] bool alphaMaskValidationPixelVisible(uint32_t x,
                                                   uint32_t y) noexcept {
  constexpr float kSize = 64.0f;
  const float u = (static_cast<float>(x) + 0.5f) / kSize;
  const float v = (static_cast<float>(y) + 0.5f) / kSize;
  const float stem = std::abs(u - 0.5f);
  const bool verticalStem = stem < 0.045f && v > 0.08f && v < 0.92f;
  const float leftLeafX = (u - 0.35f) / 0.18f;
  const float leftLeafY = (v - 0.62f) / 0.25f;
  const float rightLeafX = (u - 0.65f) / 0.18f;
  const float rightLeafY = (v - 0.42f) / 0.25f;
  const float leftLeaf = leftLeafX * leftLeafX + leftLeafY * leftLeafY;
  const float rightLeaf = rightLeafX * rightLeafX + rightLeafY * rightLeafY;
  const bool leaf = leftLeaf < 1.0f || rightLeaf < 1.0f;
  const bool calibrationBars =
      v < 0.18f && x >= 8u && x < 56u && ((x / 8u + y / 8u) % 2u) == 0u;
  return verticalStem || leaf || calibrationBars;
}

[[nodiscard]] Result<std::filesystem::path, std::string>
writeAlphaMaskValidationTexture(const std::filesystem::path &outputDirectory) {
  constexpr uint32_t kSize = 64u;
  std::error_code ec;
  std::filesystem::create_directories(outputDirectory, ec);
  if (ec) {
    return Result<std::filesystem::path, std::string>::makeError(
        "Failed to create alpha-mask validation texture directory: " +
        outputDirectory.string() + ": " + ec.message());
  }
  const std::filesystem::path outputPath =
      outputDirectory / "nuri_alpha_mask_validation_card.tga";
  const std::filesystem::path normalizedOutputPath =
      outputPath.lexically_normal();

  std::ofstream out(normalizedOutputPath, std::ios::binary);
  if (!out.is_open()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "Failed to open alpha-mask validation texture for writing: " +
        normalizedOutputPath.string());
  }

  std::array<uint8_t, 18> header{};
  header[2] = 2u;
  header[12] = static_cast<uint8_t>(kSize & 0xffu);
  header[13] = static_cast<uint8_t>((kSize >> 8u) & 0xffu);
  header[14] = static_cast<uint8_t>(kSize & 0xffu);
  header[15] = static_cast<uint8_t>((kSize >> 8u) & 0xffu);
  header[16] = 32u;
  header[17] = 0x28u;
  out.write(reinterpret_cast<const char *>(header.data()),
            static_cast<std::streamsize>(header.size()));

  for (uint32_t y = 0u; y < kSize; ++y) {
    for (uint32_t x = 0u; x < kSize; ++x) {
      const bool visible = alphaMaskValidationPixelVisible(x, y);
      const std::array<uint8_t, 4> bgra = {
          visible ? uint8_t(64u) : uint8_t(0u),
          visible ? uint8_t(190u) : uint8_t(0u),
          visible ? uint8_t(80u) : uint8_t(0u),
          visible ? uint8_t(255u) : uint8_t(0u),
      };
      out.write(reinterpret_cast<const char *>(bgra.data()),
                static_cast<std::streamsize>(bgra.size()));
    }
  }
  if (!out.good()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "Failed to write alpha-mask validation texture: " +
        normalizedOutputPath.string());
  }
  return Result<std::filesystem::path, std::string>::makeResult(
      normalizedOutputPath);
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
  MeshImportOptions meshletImport{};
  meshletImport.generateMeshlets = true;
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
              // Instance compute culling plus animation and mesh LOD keep this
              // 32K-instance scene vertex-bound; depth prepass overhead wins
              // over its occlusion benefit here.
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

  auto alphaMaskAssets = std::make_shared<AlphaMaskValidationSceneAssets>();
  {
    auto result = catalog.append(makeCustomScene(EditorSceneSpec{
        .info = {.id = "alpha_mask_shadows", .label = "Alpha Mask Shadows"},
        .prepare = [alphaMaskAssets, config](EditorScenePrepareContext &ctx)
            -> Result<void, std::string> {
          if (alphaMaskAssets->ready) {
            return Result<void, std::string>::makeResult();
          }
          alphaMaskAssets->release();
          alphaMaskAssets->resources = &ctx.runtime.resources();

          auto texturePathResult = writeAlphaMaskValidationTexture(
              config.sourcePath.parent_path() / "build" / "generated");
          if (texturePathResult.hasError()) {
            alphaMaskAssets->release();
            return Result<void, std::string>::makeError(
                texturePathResult.error());
          }
          alphaMaskAssets->alphaTexturePath = texturePathResult.value();

          auto textureResult =
              ctx.runtime.resources().acquireTexture(TextureRequest{
                  .path = alphaMaskAssets->alphaTexturePath.string(),
                  .loadOptions =
                      TextureLoadOptions{.srgb = true, .generateMipmaps = true},
                  .kind = TextureRequestKind::Texture2D,
                  .debugName = "alpha_mask_validation_card"});
          if (textureResult.hasError()) {
            alphaMaskAssets->release();
            return Result<void, std::string>::makeError(textureResult.error());
          }
          alphaMaskAssets->alphaTexture = textureResult.value();

          auto cardModelResult = ctx.runtime.resources().acquireGeneratedModel(
              makeAlphaMaskCardMeshData(), "alpha_mask_validation_card");
          if (cardModelResult.hasError()) {
            alphaMaskAssets->release();
            return Result<void, std::string>::makeError(
                cardModelResult.error());
          }
          alphaMaskAssets->cardModel = cardModelResult.value();

          auto groundModelResult =
              ctx.runtime.resources().acquireGeneratedModel(
                  makeGeneratedPlaneMeshData(), "alpha_mask_validation_ground");
          if (groundModelResult.hasError()) {
            alphaMaskAssets->release();
            return Result<void, std::string>::makeError(
                groundModelResult.error());
          }
          alphaMaskAssets->groundModel = groundModelResult.value();

          MaterialDesc cardMaterialDesc{};
          cardMaterialDesc.baseColorFactor =
              glm::vec4(0.45f, 1.0f, 0.52f, 1.0f);
          cardMaterialDesc.roughnessFactor = 0.85f;
          cardMaterialDesc.metallicFactor = 0.0f;
          cardMaterialDesc.alphaCutoff = 0.5f;
          cardMaterialDesc.alphaMode = MaterialAlphaMode::Mask;
          cardMaterialDesc.doubleSided = true;
          auto cardMaterialResult =
              ctx.runtime.resources().acquireMaterial(MaterialRequest{
                  .desc = cardMaterialDesc,
                  .textureRefs =
                      MaterialRequest::TextureRefs{
                          .baseColor = alphaMaskAssets->alphaTexture},
                  .debugName = "alpha_mask_validation_card_material"});
          if (cardMaterialResult.hasError()) {
            alphaMaskAssets->release();
            return Result<void, std::string>::makeError(
                cardMaterialResult.error());
          }
          alphaMaskAssets->cardMaterial = cardMaterialResult.value();
          ctx.runtime.resources().setModelMaterialForAllSources(
              alphaMaskAssets->cardModel, alphaMaskAssets->cardMaterial);

          MaterialDesc groundMaterialDesc{};
          groundMaterialDesc.baseColorFactor =
              glm::vec4(0.58f, 0.60f, 0.55f, 1.0f);
          groundMaterialDesc.roughnessFactor = 1.0f;
          groundMaterialDesc.metallicFactor = 0.0f;
          groundMaterialDesc.doubleSided = true;
          auto groundMaterialResult =
              ctx.runtime.resources().acquireMaterial(MaterialRequest{
                  .desc = groundMaterialDesc,
                  .debugName = "alpha_mask_validation_ground_material"});
          if (groundMaterialResult.hasError()) {
            alphaMaskAssets->release();
            return Result<void, std::string>::makeError(
                groundMaterialResult.error());
          }
          alphaMaskAssets->groundMaterial = groundMaterialResult.value();
          ctx.runtime.resources().setModelMaterialForAllSources(
              alphaMaskAssets->groundModel, alphaMaskAssets->groundMaterial);

          alphaMaskAssets->fallbackLights.clear();
          ImportedSceneLight light{};
          light.light.type = LightType::Directional;
          light.light.name = "AlphaMaskValidationSun";
          light.light.rotation =
              rotationFromLightDirection(glm::vec3(-0.45f, -0.78f, 0.34f));
          light.light.color = glm::vec3(1.0f);
          light.light.intensity = 4.0f;
          light.light.angularRadiusDegrees = 0.27f;
          light.light.enabled = true;
          alphaMaskAssets->fallbackLights.push_back(std::move(light));
          alphaMaskAssets->ready = true;
          return Result<void, std::string>::makeResult();
        },
        .activate = [alphaMaskAssets](EditorSceneActivateContext &ctx)
            -> Result<void, std::string> {
          configureStaticOpaqueScene(ctx.runtime, glm::vec3(8.0f, 16.0f, 32.0f),
                                     true);
          RenderSettings &settings = ctx.runtime.renderSettings();
          settings.shadow.enabled = true;
          settings.shadow.shadowMapSize = 1024u;
          settings.shadow.maxDistance = 16.0f;
          settings.shadow.filterMode = ShadowFilterMode::PCF3x3;
          settings.shadow.pcfSampleCount = 9u;
          settings.shadow.debug.showShadowMapViewport = true;
          settings.shadow.debug.showLightViewBounds = true;
          settings.shadow.debug.previewDepthMin = 0.0f;
          settings.shadow.debug.previewDepthMax = 1.0f;
          settings.shadow.debug.previewDepthInvert = false;
          settings.shadow.debug.previewDepthLog = false;

          const glm::mat4 groundTransform =
              glm::scale(glm::mat4(1.0f), glm::vec3(8.0f, 1.0f, 6.0f));
          const RenderableId groundRenderable =
              ctx.runtime.addRequiredRenderable(
                  alphaMaskAssets->groundModel, alphaMaskAssets->groundMaterial,
                  groundTransform,
                  "Failed to add alpha-mask validation ground");
          if (!isValid(groundRenderable)) {
            return Result<void, std::string>::makeError(
                "Failed to add alpha-mask validation ground");
          }

          const std::array<glm::vec3, 5> kCardPositions = {
              glm::vec3(-2.4f, 0.0f, -0.8f),  glm::vec3(-1.2f, 0.0f, 0.65f),
              glm::vec3(0.0f, 0.0f, -0.15f),  glm::vec3(1.25f, 0.0f, 0.75f),
              glm::vec3(2.35f, 0.0f, -0.65f),
          };
          const std::array<float, 5> kCardRotations = {
              -18.0f, 12.0f, 0.0f, -10.0f, 20.0f,
          };
          const std::array<float, 5> kCardHeights = {
              2.2f, 1.7f, 2.6f, 1.9f, 2.4f,
          };
          for (size_t i = 0; i < kCardPositions.size(); ++i) {
            const glm::mat4 transform =
                glm::translate(glm::mat4(1.0f), kCardPositions[i]) *
                glm::rotate(glm::mat4(1.0f), glm::radians(kCardRotations[i]),
                            glm::vec3(0.0f, 1.0f, 0.0f)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(kCardHeights[i] * 0.55f,
                                                      kCardHeights[i], 1.0f));
            const RenderableId cardRenderable =
                ctx.runtime.addRequiredRenderable(
                    alphaMaskAssets->cardModel, alphaMaskAssets->cardMaterial,
                    transform, "Failed to add alpha-mask validation card");
            if (!isValid(cardRenderable)) {
              return Result<void, std::string>::makeError(
                  "Failed to add alpha-mask validation card");
            }
          }

          Camera *camera = ctx.runtime.mainCamera();
          NURI_ASSERT(camera != nullptr, "Failed to get main camera");
          PerspectiveParams perspective = camera->perspective();
          perspective.nearPlane = 0.02f;
          perspective.farPlane = 80.0f;
          camera->setProjectionType(ProjectionType::Perspective);
          camera->setPerspective(perspective);
          camera->setLookAt(glm::vec3(0.0f, 2.2f, -7.0f),
                            glm::vec3(0.0f, 1.1f, 0.0f),
                            glm::vec3(0.0f, 1.0f, 0.0f));
          ctx.runtime.syncEditorCameraWidgetState(*camera);
          ctx.runtime.finalizeSceneLighting(alphaMaskAssets->fallbackLights,
                                            glm::mat4(1.0f));
          return Result<void, std::string>::makeResult();
        },
    }));
    if (result.hasError()) {
      return result;
    }
  }

  {
    auto result = catalog.append(makeStreamingScene({
        .info = {.id = "niagara_bistro",
                 .label = "Niagara Bistro",
                 .initiallySelected = true},
        .sourcePath = modelPath(config, kNiagaraBistroModelRelativePath),
        .importOptions = meshletImport,
        .instanceName = "NiagaraBistro",
        .fallbackMaterialDebugName = "niagara_bistro_fallback_material",
        .configureRender =
            [](EditorRuntime &runtime) {
              runtime.renderSettings().opaque.enableInstanceCompute = false;
              runtime.renderSettings().opaque.enableMeshLod = false;
              runtime.renderSettings().opaque.meshletMode =
                  MeshletRenderMode::Opportunistic;
              runtime.renderSettings().opaque.enableTessellation = false;
              runtime.renderSettings().opaque.forcedMeshLod = 0;
              runtime.renderSettings().opaque.meshLodDistanceThresholds =
                  glm::vec3(8.0f, 24.0f, 48.0f);
              runtime.renderSettings().opaque.enableInstanceAnimation = false;
              runtime.renderSettings().textureFiltering.mode =
                  TextureFilterMode::Anisotropic;
              runtime.renderSettings().textureFiltering.anisotropy = 8u;
            },
        .configureLoadedScene =
            [](EditorRuntime &, StreamingSceneState &state) {
              for (ImportedSceneLight &light : state.prefab.fallbackLights) {
                if (light.light.name == "Sun") {
                  light.light.intensity = 10.0f;
                }
              }
              for (ScenePrefabLight &light : state.prefab.prefab.lights) {
                if (light.light.name == "Sun") {
                  light.light.intensity = 10.0f;
                }
              }
              const float scale = glm::length(glm::vec3(state.baseModel[0]));
              NURI_LOG_INFO("Niagara Bistro reference setup scale=%.6f "
                            "sunIntensity=10.000",
                            scale);
            },
        .configureCamera =
            [](EditorRuntime &runtime, StreamingSceneState &state) {
              (void)state;
              Camera *camera = runtime.mainCamera();
              NURI_ASSERT(camera != nullptr, "Failed to get main camera");
              PerspectiveParams perspective = camera->perspective();
              perspective.nearPlane = 0.05f;
              perspective.farPlane = 500.0f;
              camera->setProjectionType(ProjectionType::Perspective);
              camera->setPerspective(perspective);
              const glm::vec3 position(-37.326633f, 8.540223f, 12.257857f);
              const glm::vec3 requestedDirection =
                  glm::normalize(glm::vec3(0.911857f, -0.287630f, 0.292892f));
              const glm::vec3 direction = glm::normalize(
                  glm::vec3(-requestedDirection.x, requestedDirection.y,
                            -requestedDirection.z));
              camera->setLookAt(position, position + direction,
                                glm::vec3(0.0f, 1.0f, 0.0f));
              if (CameraController *controller =
                      runtime.cameraSystem().activeController()) {
                controller->reset();
              }
              const glm::vec3 actualPosition = camera->position();
              const glm::vec3 actualDirection = camera->forward();
              NURI_LOG_INFO(
                  "Niagara Bistro reference camera requestedDirection="
                  "(%.6f, %.6f, %.6f) appliedPosition=(%.6f, %.6f, "
                  "%.6f) appliedDirection=(%.6f, %.6f, %.6f)",
                  requestedDirection.x, requestedDirection.y,
                  requestedDirection.z, actualPosition.x, actualPosition.y,
                  actualPosition.z, actualDirection.x, actualDirection.y,
                  actualDirection.z);
              runtime.syncEditorCameraWidgetState(*camera);
            },
    }));
    if (result.hasError()) {
      return result;
    }
  }

  auto damagedHelmetGroundAssets = std::make_shared<SimpleModelSceneAssets>();
  auto result = appendPrefab(
      catalog,
      {
          .info = {.id = "damaged_helmet", .label = "Damaged Helmet"},
          .sourcePath = modelPath(config, kDamagedHelmetModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "DamagedHelmet",
          .lodThresholds = glm::vec3(8.0f, 16.0f, 32.0f),
          .prepareAdditionalAssets =
              [damagedHelmetGroundAssets,
               config](EditorRuntime &runtime,
                       ImportedPrefabSceneResources &sceneResources)
              -> Result<void, std::string> {
            sceneResources.fallbackLights.clear();
            ImportedSceneLight shadowLight{};
            shadowLight.light.type = LightType::Directional;
            shadowLight.light.name = "DamagedHelmetSun";
            // Directional light rotation encodes the light ray direction.
            shadowLight.light.rotation =
                rotationFromLightDirection(glm::vec3(-0.42f, -0.87f, 0.26f));
            shadowLight.light.color = glm::vec3(1.0f);
            shadowLight.light.intensity = 2.5f;
            shadowLight.light.angularRadiusDegrees = 0.27f;
            shadowLight.light.enabled = true;
            sceneResources.fallbackLights.push_back(std::move(shadowLight));

            if (damagedHelmetGroundAssets->ready) {
              return Result<void, std::string>::makeResult();
            }
            damagedHelmetGroundAssets->release();
            damagedHelmetGroundAssets->resources = &runtime.resources();
            auto modelResult = runtime.resources().acquireGeneratedModel(
                makeGeneratedPlaneMeshData(), "flat_plane_generated");
            if (modelResult.hasError()) {
              damagedHelmetGroundAssets->release();
              return Result<void, std::string>::makeError(modelResult.error());
            }
            damagedHelmetGroundAssets->model = modelResult.value();

            MaterialDesc materialDesc{};
            materialDesc.baseColorFactor = glm::vec4(0.62f, 0.62f, 0.58f, 1.0f);
            materialDesc.metallicFactor = 0.0f;
            materialDesc.roughnessFactor = 1.0f;
            materialDesc.doubleSided = true;
            auto materialResult =
                runtime.resources().acquireMaterial(MaterialRequest{
                    .desc = materialDesc,
                    .debugName = "damaged_helmet_ground",
                });
            if (materialResult.hasError()) {
              damagedHelmetGroundAssets->release();
              return Result<void, std::string>::makeError(
                  materialResult.error());
            }
            damagedHelmetGroundAssets->material = materialResult.value();
            runtime.resources().setModelMaterialForAllSources(
                damagedHelmetGroundAssets->model,
                damagedHelmetGroundAssets->material);
            damagedHelmetGroundAssets->ready = true;
            return Result<void, std::string>::makeResult();
          },
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &bounds) {
                const BoundingBox framedBounds =
                    damagedHelmetCameraBounds(bounds);
                (void)runtime.frameSceneCamera(
                    framedBounds, glm::mat4(1.0f), 2.1f, 2.0f,
                    glm::vec4(0.34f, 0.64f, 0.88f, 0.08f),
                    glm::vec2(-0.2f, 0.0f));
              },
          .populateScene =
              [damagedHelmetGroundAssets](
                  EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                  const BoundingBox &bounds) -> Result<void, std::string> {
            const GroundPlanePlacement plane =
                damagedHelmetGroundPlacement(bounds);
            const glm::vec3 center = bounds.getCenter();
            const glm::mat4 planeTransform =
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(center.x, plane.y, center.z)) *
                glm::scale(glm::mat4(1.0f),
                           glm::vec3(plane.extent, 1.0f, plane.extent));
            const RenderableId planeRenderable = runtime.addRequiredRenderable(
                damagedHelmetGroundAssets->model,
                damagedHelmetGroundAssets->material, planeTransform,
                "Failed to add Damaged Helmet ground plane");
            if (!isValid(planeRenderable)) {
              return Result<void, std::string>::makeError(
                  "Failed to add Damaged Helmet ground plane");
            }
            return Result<void, std::string>::makeResult();
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

  result = appendPrefab(
      catalog,
      {
          .info = {.id = "sponza", .label = "Sponza"},
          .sourcePath = modelPath(config, kSponzaModelRelativePath),
          .importOptions = flipUvImport,
          .instanceName = "Sponza",
          .baseModel = sponzaBaseModel(),
          .configureRender =
              [](EditorRuntime &runtime) {
                runtime.configureStaticModelOpaqueSettings(
                    glm::vec3(12.0f, 24.0f, 48.0f));
              },
          .configureCamera =
              [](EditorRuntime &runtime, const ImportedPrefabSceneResources &,
                 const BoundingBox &) {
                Camera *camera = runtime.mainCamera();
                NURI_ASSERT(camera != nullptr, "Failed to get main camera");
                PerspectiveParams perspective = camera->perspective();
                perspective.nearPlane = 0.5f;
                perspective.farPlane = 2500.0f;
                camera->setProjectionType(ProjectionType::Perspective);
                camera->setPerspective(perspective);
                // Sponza is authored as one node with a large exterior shell;
                // bounds-framing starts outside that shell and makes it look
                // like a tiled box instead of the atrium.
                camera->setLookAt(glm::vec3(-720.0f, 250.0f, -65.0f),
                                  glm::vec3(820.0f, 270.0f, -20.0f),
                                  glm::vec3(0.0f, 1.0f, 0.0f));
                runtime.syncEditorCameraWidgetState(*camera);
              },
      });
  if (result.hasError()) {
    return result;
  }

  result = catalog.append(makeAnimatedPrefabScene({
      .prefab = {
          .info = {.id = "fox",
                   .label = "Fox Survey Walk Blend",
                   .initiallySelected = false},
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
      .preferredClipNames = {"Survey", "Walk"},
      .secondaryPreferredClipNames = {"Walk", "Survey"},
      .initialBlendWeight = 0.5f,
      .blendSyncMode = AnimationPoseBlendSyncMode::NormalizedTime,
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

  result = catalog.append(makeAnimatedPrefabScene(
      {.prefab = {
           .info = {.id = "animated_morph_cube",
                    .label = "Animated Morph Cube"},
           .sourcePath = modelPath(config, kAnimatedMorphCubeModelRelativePath),
           .importOptions = flipUvImport,
           .instanceName = "AnimatedMorphCube",
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
       .simulationDebugName = "AnimatedMorphCubeAnimation"}));
  if (result.hasError()) {
    return result;
  }

  result = catalog.append(makeCustomScene(EditorSceneSpec{
      .info = {.id = "text_3d_test",
               .label = "Text 3D Test",
               .initiallySelected = false},
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
