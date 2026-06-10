#include "nuri/pch.h"

#include "nuri/core/application.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/pipeline/features/text_feature.h"
#include "nuri/math/light.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/resources/scene_importer.h"
#include "nuri/scene/camera_system.h"
#include "nuri/scene/render_scene.h"
#include "nuri/scene_runtime/scene_runtime_host.h"
#include "nuri/text/text_system.h"

#include <glm/gtc/matrix_transform.hpp>

namespace {

constexpr std::string_view kDamagedHelmetModelRelativePath =
    "DamagedHelmet/DamagedHelmet.gltf";
constexpr std::string_view kSampleEnvironmentHdrRelativePath =
    "qwantani_moon_noon_puresky_4k.hdr";
constexpr float kHelmetRotationSpeedRadians = glm::radians(35.0f);

[[nodiscard]] glm::mat4 makeTransformMatrix(const glm::vec3 &position,
                                            const glm::quat &rotation) {
  return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotation);
}

std::filesystem::path pickDefaultNfontPath(const nuri::RuntimeConfig &config) {
  const std::filesystem::path fontsRoot = config.roots.fonts;
  const std::array<std::string_view, 2> preferred = {"default_ui.nfont",
                                                     "generated_ui.nfont"};

  std::error_code ec;
  for (std::string_view name : preferred) {
    const std::filesystem::path candidate =
        (fontsRoot / std::string(name)).lexically_normal();
    if (std::filesystem::exists(candidate, ec) &&
        std::filesystem::is_regular_file(candidate, ec)) {
      return candidate;
    }
    ec.clear();
  }

  std::filesystem::path newest;
  std::filesystem::file_time_type newestWriteTime{};
  bool foundNewest = false;
  if (std::filesystem::exists(fontsRoot, ec) &&
      std::filesystem::is_directory(fontsRoot, ec)) {
    for (const auto &entry :
         std::filesystem::directory_iterator(fontsRoot, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_regular_file(ec)) {
        ec.clear();
        continue;
      }
      const std::filesystem::path candidate = entry.path();
      if (candidate.extension() != ".nfont") {
        continue;
      }
      const auto writeTime = entry.last_write_time(ec);
      if (ec) {
        ec.clear();
        continue;
      }
      if (!foundNewest || writeTime > newestWriteTime) {
        newest = candidate;
        newestWriteTime = writeTime;
        foundNewest = true;
      }
    }
  }
  if (foundNewest) {
    return newest.lexically_normal();
  }

  return (fontsRoot / "default_ui.nfont").lexically_normal();
}

} // namespace

namespace {

[[nodiscard]] nuri::ApplicationConfig
makeSampleApplicationConfig(const nuri::RuntimeConfig &config) {
  nuri::ApplicationConfig appConfig = nuri::makeApplicationConfig(config);
  appConfig.renderComposition = nuri::RenderCompositionMode::PipelineOnly;
  return appConfig;
}

} // namespace

class NuriApplication : public nuri::Application {
public:
  explicit NuriApplication(nuri::RuntimeConfig config)
      : nuri::Application(makeSampleApplicationConfig(config)),
        config_(std::move(config)), cameraSystem_(cameraMemory_),
        scene_(&sceneMemory_), sceneRuntime_(&sceneMemory_) {}

  void onInit() override {
    NURI_PROFILER_FUNCTION();
    scene_.bindResources(&getRenderer().resources());
    sceneRuntime_.bindScene(&scene_);
    initializeCamera();
    initializeTextSystem();
    loadSceneResources();
    initializeTextOverlayFeature();

    NURI_LOG_INFO("Application was initialized");
  }

  void onDraw() override {
    NURI_PROFILER_FUNCTION();

    const nuri::Camera *activeCamera = cameraSystem_.activeCamera();
    NURI_ASSERT(activeCamera != nullptr, "No active camera");
    auto commitResult = scene_.commit();
    NURI_ASSERT(!commitResult.hasError(), "Scene commit failed: %s",
                commitResult.error().c_str());

    buildFrameContext(*activeCamera, getTime());
    queuePerformanceOverlay();
    submitPipelineFrame();
  }

  void onUpdate(double deltaTime) override {
    updatePerformanceMetrics(deltaTime);
    updateHelmetRotation(deltaTime);
    (void)sceneRuntime_.tick({
        .frameDeltaSeconds = std::max(0.0, deltaTime),
        .absoluteTimeSeconds = getTime(),
        .frameIndex = simulationFrameIndex_++,
    });
    cameraSystem_.update(deltaTime, getInput());
  }

  void onResize(std::int32_t, std::int32_t) override {}

  bool onInput(const nuri::InputEvent &event) override {
    if (cameraSystem_.onInput(event, getWindow())) {
      return true;
    }
    return nuri::Application::onInput(event);
  }

  void onShutdown() override {
    sceneRuntime_.reset();
    sceneRuntime_.bindScene(nullptr);
    scene_.graph().clearRenderables();
    scene_.graph().clearLights();
    scene_.setEnvironment(nuri::EnvironmentHandles{});
    releaseOwnedResourceHandles();
    scene_.bindResources(nullptr);
    getWindow().setCursorMode(nuri::CursorMode::Normal);
    NURI_LOG_INFO("Application was shutdown");
  }

private:
  void initializeCamera() {
    nuri::Camera camera{};
    camera.setProjectionType(nuri::ProjectionType::Perspective);

    nuri::CameraController controller = nuri::makeFpsDirectController();
    mainCameraHandle_ = cameraSystem_.addCamera(camera, std::move(controller));
    NURI_ASSERT(mainCameraHandle_.isValid(),
                "Failed to add camera to camera system");

    const bool setActive =
        cameraSystem_.setActiveCamera(mainCameraHandle_, getWindow());
    NURI_ASSERT(setActive, "Failed to activate main camera");
  }

  void initializeTextOverlayFeature() {
    if (textOverlayEnabled_ || textSystem_ == nullptr) {
      return;
    }
    auto *feature = getRenderPipeline().addFeature(
        std::make_unique<nuri::Text2DFeature>(*textSystem_));
    NURI_ASSERT(feature != nullptr, "Failed to register 2D text feature");
    textOverlayEnabled_ = true;
  }

  void initializeTextSystem() {
    const std::filesystem::path defaultFontPath = pickDefaultNfontPath(config_);
    const bool requireDefaultFont =
        std::filesystem::is_regular_file(defaultFontPath);
    NURI_LOG_INFO("NuriApplication::initializeTextSystem: default font '%s'",
                  defaultFontPath.string().c_str());

    auto textSystemResult = nuri::TextSystem::create({
        .gpu = getGPU(),
        .memory = *pipelineMemoryResource(),
        .defaultFontPath = defaultFontPath,
        .requireDefaultFont = requireDefaultFont,
        .shaderPaths =
            {
                .uiVertex = config_.shaders.textMtsdf.uiVertex,
                .uiFragment = config_.shaders.textMtsdf.uiFragment,
                .worldVertex = config_.shaders.textMtsdf.worldVertex,
                .worldFragment = config_.shaders.textMtsdf.worldFragment,
            },
    });
    NURI_ASSERT(!textSystemResult.hasError(),
                "Failed to create text system: %s",
                textSystemResult.error().c_str());
    textSystem_ = std::move(textSystemResult.value());
    NURI_ASSERT(textSystem_ != nullptr, "Text system was not created");
  }

  void queuePerformanceOverlay() {
    if (textSystem_ == nullptr || !textOverlayEnabled_) {
      return;
    }

    auto begin = textSystem_->renderer().beginFrame(frameContext_.frameIndex);
    if (begin.hasError()) {
      NURI_LOG_WARNING("NuriApplication: failed to begin text frame: %s",
                       begin.error().c_str());
      return;
    }

    const nuri::FontHandle defaultFont = textSystem_->defaultFont();
    if (!nuri::isValid(defaultFont)) {
      return;
    }

    const float baseFontSizePx =
        std::clamp(textSystem_->defaultFontSizePx(), 8.0f, 256.0f);
    std::array<char, 64> perfText{};
    std::snprintf(perfText.data(), perfText.size(), "FPS: %.1f\nFT: %.2f ms",
                  currentFps_, static_cast<float>(frameDeltaSeconds_ * 1000.0));

    nuri::ScopedScratch scopedScratch(textScratchArena_);
    std::pmr::memory_resource &scratch = *scopedScratch.resource();

    nuri::Text2DDesc perf{};
    perf.utf8 = perfText.data();
    perf.style.font = defaultFont;
    perf.style.pxSize = baseFontSizePx * 0.55f;
    perf.layout.alignH = nuri::TextAlignH::Left;
    perf.layout.alignV = nuri::TextAlignV::Top;
    perf.fillColor = {.r = 0.95f, .g = 1.0f, .b = 0.95f, .a = 1.0f};
    perf.x = 20.0f;
    perf.y = 20.0f;

    auto enqueue = textSystem_->renderer().enqueue2D(perf, scratch);
    if (enqueue.hasError()) {
      NURI_LOG_WARNING("NuriApplication: failed to enqueue overlay text: %s",
                       enqueue.error().c_str());
    }
  }

  void loadSceneResources() {
    nuri::ResourceManager &resources = getRenderer().resources();
    scene_.graph().clearRenderables();
    scene_.graph().clearLights();
    scene_.setEnvironment(nuri::EnvironmentHandles{});
    releaseOwnedResourceHandles();

    const std::string helmetModelPath =
        (config_.roots.models / kDamagedHelmetModelRelativePath).string();
    const std::string environmentHdrPath =
        (config_.roots.textures / kSampleEnvironmentHdrRelativePath).string();

    nuri::MeshImportOptions helmetImportOptions{};
    helmetImportOptions.flipUVs = true;
    auto helmetPrefabResult = nuri::SceneImporter::loadScenePrefabFromFile(
        helmetModelPath, nuri::SceneImportOptions{
                             .assetBuildOptions = helmetImportOptions,
                         });
    NURI_ASSERT(!helmetPrefabResult.hasError(),
                "Failed to load DamagedHelmet scene prefab: %s",
                helmetPrefabResult.error().c_str());
    helmetPrefab_ = std::move(helmetPrefabResult.value());

    auto helmetAssetsResult = resources.acquireScenePrefabAssets(helmetPrefab_);
    NURI_ASSERT(!helmetAssetsResult.hasError(),
                "Failed to acquire DamagedHelmet scene prefab assets: %s",
                helmetAssetsResult.error().c_str());
    helmetSceneAssets_ = std::move(helmetAssetsResult.value());

    auto cubemapResult = resources.acquireTexture(nuri::TextureRequest{
        .path = environmentHdrPath,
        .loadOptions = nuri::TextureLoadOptions{},
        .kind = nuri::TextureRequestKind::EquirectHdrCubemap,
        .debugName = "cubemap",
    });
    NURI_ASSERT(!cubemapResult.hasError(),
                "Failed to create cubemap texture: %s",
                cubemapResult.error().c_str());

    const std::filesystem::path environmentHdrFile{environmentHdrPath};
    const std::string environmentStem = environmentHdrFile.stem().string();
    const std::array<std::filesystem::path, 2> irradianceCandidates = {
        config_.roots.textures / (environmentStem + "_irradiance.ktx2"),
        config_.roots.textures / (environmentStem + "_irradiance.ktx"),
    };
    const std::array<std::filesystem::path, 2> prefilteredGgxCandidates = {
        config_.roots.textures / (environmentStem + "_prefilter_ggx.ktx2"),
        config_.roots.textures / (environmentStem + "_prefilter_ggx.ktx"),
    };
    const std::array<std::filesystem::path, 4> prefilteredCharlieCandidates = {
        config_.roots.textures / (environmentStem + "_prefilter_charlie.ktx2"),
        config_.roots.textures / (environmentStem + "_prefilter_charlie.ktx"),
        config_.roots.textures / (environmentStem + "_prefilter_charile.ktx"),
        config_.roots.textures / (environmentStem + "_prefilter_charile.ktx2"),
    };
    const std::array<std::filesystem::path, 2> brdfLutCandidates = {
        config_.roots.textures / "brdf_lut.ktx2",
        config_.roots.textures / "brdf_lut.ktx",
    };

    const auto resolveIblAssetPath =
        [this](const std::filesystem::path &preferredPath)
        -> std::filesystem::path {
      std::error_code ec;
      if (std::filesystem::exists(preferredPath, ec) &&
          std::filesystem::is_regular_file(preferredPath, ec)) {
        return preferredPath;
      }

      const std::filesystem::path fileName = preferredPath.filename();
      const std::array<std::filesystem::path, 2> fallbacks = {
          config_.roots.textures / ".hide" / fileName,
          config_.roots.textures / ".hide2" / fileName,
      };
      for (const auto &candidate : fallbacks) {
        ec.clear();
        if (std::filesystem::exists(candidate, ec) &&
            std::filesystem::is_regular_file(candidate, ec)) {
          NURI_LOG_WARNING(
              "NuriApplication::loadSceneResources: using fallback IBL asset "
              "'%s' (preferred '%s' not found)",
              candidate.string().c_str(), preferredPath.string().c_str());
          return candidate;
        }
      }

      return preferredPath;
    };

    const auto resolveFirstExistingIblAssetPath =
        [&resolveIblAssetPath](
            std::span<const std::filesystem::path> candidates)
        -> std::filesystem::path {
      if (candidates.empty()) {
        return {};
      }
      std::error_code ec;
      for (const auto &candidate : candidates) {
        const std::filesystem::path resolved = resolveIblAssetPath(candidate);
        ec.clear();
        if (std::filesystem::exists(resolved, ec) &&
            std::filesystem::is_regular_file(resolved, ec)) {
          return resolved;
        }
      }
      return resolveIblAssetPath(candidates.front());
    };

    const auto tryLoadKtxCubemap =
        [&resources, &resolveFirstExistingIblAssetPath](
            std::span<const std::filesystem::path> candidates,
            std::string_view debugName) -> nuri::TextureRef {
      if (candidates.empty()) {
        return nuri::kInvalidTextureRef;
      }
      const std::filesystem::path resolvedPath =
          resolveFirstExistingIblAssetPath(candidates);
      std::error_code ec;
      if (!std::filesystem::exists(resolvedPath, ec) ||
          !std::filesystem::is_regular_file(resolvedPath, ec)) {
        NURI_LOG_WARNING(
            "NuriApplication::loadSceneResources: missing IBL cubemap '%s'",
            candidates.front().string().c_str());
        return nuri::kInvalidTextureRef;
      }

      auto result = resources.acquireTexture(nuri::TextureRequest{
          .path = resolvedPath.string(),
          .kind = nuri::TextureRequestKind::Ktx2Cubemap,
          .debugName = std::string(debugName),
      });
      if (result.hasError()) {
        NURI_LOG_WARNING("NuriApplication::loadSceneResources: failed to load "
                         "IBL cubemap '%s': %s",
                         resolvedPath.string().c_str(), result.error().c_str());
        return nuri::kInvalidTextureRef;
      }
      return result.value();
    };

    const auto tryLoadKtxTexture2D =
        [&resources, &resolveFirstExistingIblAssetPath](
            std::span<const std::filesystem::path> candidates,
            std::string_view debugName) -> nuri::TextureRef {
      if (candidates.empty()) {
        return nuri::kInvalidTextureRef;
      }
      const std::filesystem::path resolvedPath =
          resolveFirstExistingIblAssetPath(candidates);
      std::error_code ec;
      if (!std::filesystem::exists(resolvedPath, ec) ||
          !std::filesystem::is_regular_file(resolvedPath, ec)) {
        NURI_LOG_WARNING("NuriApplication::loadSceneResources: missing BRDF "
                         "LUT '%s'",
                         candidates.front().string().c_str());
        return nuri::kInvalidTextureRef;
      }

      auto result = resources.acquireTexture(nuri::TextureRequest{
          .path = resolvedPath.string(),
          .kind = nuri::TextureRequestKind::Ktx2Texture2D,
          .debugName = std::string(debugName),
      });
      if (result.hasError()) {
        NURI_LOG_WARNING("NuriApplication::loadSceneResources: failed to load "
                         "BRDF LUT '%s': %s",
                         resolvedPath.string().c_str(), result.error().c_str());
        return nuri::kInvalidTextureRef;
      }
      return result.value();
    };

    nuri::TextureRef irradianceCubemap =
        tryLoadKtxCubemap(irradianceCandidates, "ibl_irradiance");
    nuri::TextureRef prefilteredGgxCubemap =
        tryLoadKtxCubemap(prefilteredGgxCandidates, "ibl_prefilter_ggx");
    nuri::TextureRef prefilteredCharlieCubemap = tryLoadKtxCubemap(
        prefilteredCharlieCandidates, "ibl_prefilter_charlie");
    nuri::TextureRef brdfLutTexture =
        tryLoadKtxTexture2D(brdfLutCandidates, "ibl_brdf_lut");
    scene_.setEnvironment(nuri::EnvironmentHandles{
        .cubemap = cubemapResult.value(),
        .irradiance = irradianceCubemap,
        .prefilteredGgx = prefilteredGgxCubemap,
        .prefilteredCharlie = prefilteredCharlieCubemap,
        .brdfLut = brdfLutTexture,
    });
    resources.release(cubemapResult.value());
    if (nuri::isValid(irradianceCubemap)) {
      resources.release(irradianceCubemap);
    }
    if (nuri::isValid(prefilteredGgxCubemap)) {
      resources.release(prefilteredGgxCubemap);
    }
    if (nuri::isValid(prefilteredCharlieCubemap)) {
      resources.release(prefilteredCharlieCubemap);
    }
    if (nuri::isValid(brdfLutTexture)) {
      resources.release(brdfLutTexture);
    }

    nuri::ModelRef helmetPrimaryModel = nuri::kInvalidModelRef;
    for (nuri::ModelRef model : helmetSceneAssets_.models) {
      if (nuri::isValid(model)) {
        helmetPrimaryModel = model;
        break;
      }
    }
    NURI_ASSERT(nuri::isValid(helmetPrimaryModel),
                "DamagedHelmet prefab did not resolve any model assets");

    const nuri::ModelRecord *helmetRecord =
        resources.tryGet(helmetPrimaryModel);
    NURI_ASSERT(helmetRecord != nullptr && helmetRecord->model != nullptr,
                "DamagedHelmet model record lookup failed");
    const nuri::Model &helmetModel = *helmetRecord->model;

    renderSettings_.opaque.enableInstanceCompute = false;
    renderSettings_.opaque.enableMeshLod = true;
    renderSettings_.opaque.enableTessellation = false;
    renderSettings_.opaque.forcedMeshLod = -1;
    renderSettings_.opaque.meshLodDistanceThresholds =
        glm::vec3(8.0f, 16.0f, 32.0f);
    renderSettings_.opaque.enableInstanceAnimation = false;

    helmetRotationRadians_ = 0.0f;
    auto helmetNodeResult = scene_.graph().createNode(
        scene_.graph().rootNode(), "DamagedHelmet", helmetBaseModel_);
    NURI_ASSERT(!helmetNodeResult.hasError(),
                "Failed to create DamagedHelmet node: %s",
                helmetNodeResult.error().c_str());
    helmetNode_ = helmetNodeResult.value();

    nuri::SceneInstantiationMap instantiated;
    auto instantiateResult = scene_.graph().instantiatePrefab(
        helmetPrefab_, helmetNode_, helmetSceneAssets_, &instantiated);
    NURI_ASSERT(!instantiateResult.hasError(),
                "Failed to instantiate DamagedHelmet prefab: %s",
                instantiateResult.error().c_str());
    for (nuri::RenderableId renderable : instantiated.renderables) {
      if (nuri::isValid(renderable)) {
        helmetRenderableId_ = renderable;
        break;
      }
    }
    NURI_ASSERT(nuri::isValid(helmetRenderableId_),
                "DamagedHelmet prefab did not produce any renderables");

    const nuri::BoundingBox &bounds = helmetModel.bounds();
    const float rawRadius =
        std::max(0.5f * glm::length(bounds.getSize()), 0.25f);
    const glm::vec3 center =
        glm::vec3(helmetBaseModel_ * glm::vec4(bounds.getCenter(), 1.0f));
    const float radius = std::max(0.25f, rawRadius);
    const float cameraDistance = std::max(radius * 2.4f, 2.0f);

    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    NURI_ASSERT(camera != nullptr, "Failed to get main camera");
    nuri::PerspectiveParams perspective = camera->perspective();
    perspective.nearPlane = std::max(0.01f, cameraDistance / 3000.0f);
    perspective.farPlane = std::max(500.0f, cameraDistance + radius * 12.0f);
    camera->setProjectionType(nuri::ProjectionType::Perspective);
    camera->setPerspective(perspective);
    camera->setLookAt(center + glm::vec3(-cameraDistance * 0.38f,
                                         radius * 0.18f + 0.2f,
                                         -cameraDistance),
                      center + glm::vec3(0.0f, radius * 0.03f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f));
    if (instantiated.lights.empty()) {
      addDefaultDirectionalLight();
    }
  }

  void addDefaultDirectionalLight() {
    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    if (camera == nullptr) {
      return;
    }

    nuri::LightDesc light{};
    light.type = nuri::LightType::Directional;
    light.name = "Default Directional";
    light.position = camera->position() + camera->forward() * 2.0f;
    light.rotation = camera->orientation();
    light.color = glm::vec3(1.0f);
    light.intensity = 2.0f;
    light.enabled = true;
    auto nodeResult = scene_.graph().createNode(
        scene_.graph().rootNode(), light.name,
        makeTransformMatrix(light.position, light.rotation));
    NURI_ASSERT(!nodeResult.hasError(),
                "Failed to create default directional light node: %s",
                nodeResult.error().c_str());
    light.position = glm::vec3(0.0f);
    light.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    auto addResult = scene_.graph().addLight(nodeResult.value(), light);
    NURI_ASSERT(!addResult.hasError(),
                "Failed to add default directional light: %s",
                addResult.error().c_str());
  }

  void releaseOwnedResourceHandles() {
    nuri::ResourceManager &resources = getRenderer().resources();
    const auto releaseRef = [&resources](auto &ref, const auto invalidRef) {
      if (nuri::isValid(ref)) {
        resources.release(ref);
      }
      ref = invalidRef;
    };

    for (nuri::ModelRef model : helmetSceneAssets_.models) {
      if (nuri::isValid(model)) {
        resources.release(model);
      }
    }
    for (nuri::MaterialRef material : helmetSceneAssets_.materials) {
      if (nuri::isValid(material)) {
        resources.release(material);
      }
    }
    helmetSceneAssets_ = nuri::ScenePrefabAssets{};
    helmetPrefab_ = nuri::ScenePrefab{};
    helmetRenderableId_ = nuri::kInvalidRenderableId;
    helmetNode_ = nuri::kInvalidNodeId;
  }

  void updatePerformanceMetrics(double deltaTime) {
    frameDeltaSeconds_ =
        (std::isfinite(deltaTime) && deltaTime >= 0.0) ? deltaTime : 0.0;
    fpsFrameCount_++;
    fpsAccumulatorSeconds_ += frameDeltaSeconds_;
    constexpr double kFpsAverageWindowSeconds = 0.5;
    if (fpsAccumulatorSeconds_ >= kFpsAverageWindowSeconds) {
      currentFps_ =
          fpsAccumulatorSeconds_ > 0.0
              ? static_cast<float>(fpsFrameCount_ / fpsAccumulatorSeconds_)
              : 0.0f;
      fpsAccumulatorSeconds_ = 0.0;
      fpsFrameCount_ = 0;
    }
  }

  void updateHelmetRotation(double deltaTime) {
    if (!nuri::isValid(helmetNode_) || !std::isfinite(deltaTime) ||
        deltaTime <= 0.0) {
      return;
    }

    helmetRotationRadians_ +=
        static_cast<float>(deltaTime) * kHelmetRotationSpeedRadians;
    if (helmetRotationRadians_ >= glm::radians(360.0f)) {
      helmetRotationRadians_ -= glm::radians(360.0f);
    }

    const glm::mat4 modelMatrix = glm::rotate(
        helmetBaseModel_, helmetRotationRadians_, glm::vec3(0.0f, 0.0f, 1.0f));
    const bool updated =
        scene_.graph().setNodeLocalTransform(helmetNode_, modelMatrix);
    NURI_ASSERT(updated, "Failed to update DamagedHelmet node transform");
  }

  void buildFrameContext(const nuri::Camera &camera, double timeSeconds) {
    nuri::sanitizeHDRPostProcessSettings(renderSettings_.hdrPostProcess);
    nuri::sanitizeTransmissionSettings(renderSettings_.transmission);
    nuri::sanitizeAntiAliasingSettings(renderSettings_.antiAliasing);
    nuri::sanitizeAmbientOcclusionSettings(renderSettings_.ambientOcclusion,
                                           renderSettings_.opaque,
                                           renderSettings_.antiAliasing);
    frameContext_.scene = &scene_;
    nuri::ResourceManager &resources = getRenderer().resources();
    frameContext_.resources = &resources;
    frameContext_.frameIndex = frameIndex_++;
    const nuri::MaterialTableSnapshot materialSnapshot =
        resources.materialSnapshot();
    const nuri::TemporalSceneContentState sceneContent{
        .lightTopologyVersion = scene_.lightTopologyVersion(),
        .lightTransformVersion = scene_.lightTransformVersion(),
        .materialTableVersion = materialSnapshot.version,
        .environmentVersion = scene_.environmentVersion(),
    };
    frameContext_.camera = nuri::makeTemporalCameraFrameState(
        camera, getAspectRatio(), renderSettings_.antiAliasing,
        nuri::TemporalCameraFrameDesc{
            .renderExtent =
                glm::uvec2(static_cast<uint32_t>(std::max(getWidth(), 0)),
                           static_cast<uint32_t>(std::max(getHeight(), 0))),
            .sceneContent = sceneContent,
        },
        temporalCameraHistory_);
    renderSettings_.antiAliasing.debug.resetHistoryRequested = false;
    frameContext_.settings = &renderSettings_;
    frameContext_.metrics = {};
    frameContext_.metrics.frameIndex = frameContext_.frameIndex;
    frameContext_.metrics.antiAliasing =
        nuri::makeAntiAliasingFrameMetrics(frameContext_.camera);
    frameContext_.sharedDepthTexture = {};
    frameContext_.timeSeconds = timeSeconds;
    frameContext_.deltaSeconds = frameDeltaSeconds_;
  }

  void submitPipelineFrame() {
    auto renderResult =
        getRenderer().render(getRenderPipeline(), frameContext_);
    NURI_ASSERT(!renderResult.hasError(), "Render failed: %s",
                renderResult.error().c_str());
  }

  const nuri::RuntimeConfig config_;
  std::pmr::unsynchronized_pool_resource cameraMemory_;
  std::pmr::unsynchronized_pool_resource sceneMemory_;
  nuri::CameraSystem cameraSystem_;
  nuri::RenderScene scene_;
  nuri::SceneRuntimeHost sceneRuntime_;
  nuri::ScenePrefab helmetPrefab_{};
  nuri::ScenePrefabAssets helmetSceneAssets_{};
  nuri::CameraHandle mainCameraHandle_{};
  nuri::NodeId helmetNode_ = nuri::kInvalidNodeId;
  nuri::RenderableId helmetRenderableId_ = nuri::kInvalidRenderableId;
  glm::mat4 helmetBaseModel_ =
      glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));

  nuri::RenderSettings renderSettings_{};
  nuri::RenderFrameContext frameContext_{};
  nuri::TemporalCameraHistoryState temporalCameraHistory_{};
  uint64_t frameIndex_ = 0;
  uint64_t simulationFrameIndex_ = 0;
  double frameDeltaSeconds_ = 0.0;
  double fpsAccumulatorSeconds_ = 0.0;
  uint32_t fpsFrameCount_ = 0;
  float currentFps_ = 0.0f;
  float helmetRotationRadians_ = 0.0f;
  std::unique_ptr<nuri::TextSystem> textSystem_{};
  nuri::ScratchArena textScratchArena_{};
  bool textOverlayEnabled_ = false;
};

int main() {
  NURI_PROFILER_THREAD("Main");
  auto configResult = nuri::loadRuntimeConfigFromEnvOrDefault();
  NURI_ASSERT(!configResult.hasError(), "Failed to load app config: %s",
              configResult.error().c_str());

  NuriApplication app{std::move(configResult.value())};
  app.run();
  return 0;
}
