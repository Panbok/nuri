#include "nuri/pch.h"

#include "nuri/core/application.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/frame/temporal_frame_service.h"
#include "nuri/math/light.h"
#include "nuri/resources/async/asset_system.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/gpu/texture.h"
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

struct VisibilityReadbackOverlayState {
  uint32_t available = 0;
  uint32_t sourceFrame = 0;
  uint32_t staleFrameCount = 0;
  uint32_t errorCount = 0;
  uint32_t occlusionAvailable = 0;
};

[[nodiscard]] const char *
visibilityReadbackStatus(const VisibilityReadbackOverlayState &state) {
  if (state.errorCount != 0u) {
    return "error";
  }
  return state.available != 0u ? "ready" : "waiting";
}

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
    requestSceneResources();
    initializeTextOverlayFeature();

    NURI_LOG_INFO("Application was initialized");
  }

  void onDraw() override {
    NURI_PROFILER_FUNCTION();

    pollSceneResources();
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
    cancelSceneResourceRequests();
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
    nuri::registerText2DStage(getRenderPipeline(), *textSystem_);
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

    textSystem_->beginFrame(frameContext_.frameIndex);

    const nuri::FontHandle defaultFont = textSystem_->defaultFont();
    if (!nuri::isValid(defaultFont)) {
      return;
    }

    const float baseFontSizePx =
        std::clamp(textSystem_->defaultFontSizePx(), 8.0f, 256.0f);
    const bool meshletsRequested =
        renderSettings_.opaque.meshletMode != nuri::MeshletRenderMode::Disabled;
    const char *meshletStatus = "Off";
    if (meshletActiveLastFrame_) {
      meshletStatus = "Active";
    } else if (meshletFallbackLastFrame_) {
      meshletStatus = "Fallback";
    } else if (meshletsRequested) {
      meshletStatus = meshletMetricsInitialized_ ? "Waiting" : "Pending";
    }

    const char *mainReadbackStatus =
        visibilityReadbackStatus(mainVisibilityLastFrame_);
    const char *meshletReadbackStatus =
        visibilityReadbackStatus(meshletVisibilityLastFrame_);

    std::array<char, 512> perfText{};
    std::snprintf(perfText.data(), perfText.size(),
                  "FPS: %.1f\nFT: %.2f ms\nMeshlets: %s (%u GPU groups)\n"
                  "Vis main: %s src=%u stale=%u err=%u occ=%u\n"
                  "Vis mesh: %s src=%u stale=%u err=%u occ=%u",
                  currentFps_, static_cast<float>(frameDeltaSeconds_ * 1000.0),
                  meshletStatus, meshletTaskGroupsLastFrame_,
                  mainReadbackStatus, mainVisibilityLastFrame_.sourceFrame,
                  mainVisibilityLastFrame_.staleFrameCount,
                  mainVisibilityLastFrame_.errorCount,
                  mainVisibilityLastFrame_.occlusionAvailable,
                  meshletReadbackStatus,
                  meshletVisibilityLastFrame_.sourceFrame,
                  meshletVisibilityLastFrame_.staleFrameCount,
                  meshletVisibilityLastFrame_.errorCount,
                  meshletVisibilityLastFrame_.occlusionAvailable);

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

    auto enqueue = textSystem_->enqueue2D(perf, scratch);
    if (enqueue.hasError()) {
      NURI_LOG_WARNING("NuriApplication: failed to enqueue overlay text: %s",
                       enqueue.error().c_str());
    }
  }

  void requestSceneResources() {
    nuri::AssetSystem &assets = getRenderer().assets();
    scene_.graph().clearRenderables();
    scene_.graph().clearLights();
    scene_.setEnvironment(nuri::EnvironmentHandles{});
    cancelSceneResourceRequests();

    const std::string helmetModelPath =
        (config_.roots.models / kDamagedHelmetModelRelativePath).string();
    const std::string environmentHdrPath =
        (config_.roots.textures / kSampleEnvironmentHdrRelativePath).string();

    nuri::MeshImportOptions helmetImportOptions{};
    helmetImportOptions.flipUVs = true;
    helmetImportOptions.generateMeshlets = true;
    auto helmetRequest = assets.requestScene(nuri::SceneLoadRequest{
        .path = helmetModelPath,
        .importOptions =
            nuri::SceneImportOptions{
                .assetBuildOptions = helmetImportOptions,
            },
        .priority = nuri::AssetPriority::Critical,
        .publication = nuri::ScenePublicationPolicy::Progressive,
        .failurePolicy = nuri::SceneFailurePolicy::BestEffort,
        .debugName = "sample_damaged_helmet",
    });
    NURI_ASSERT(!helmetRequest.hasError(),
                "Failed to request DamagedHelmet scene: %s",
                helmetRequest.error().c_str());
    helmetSceneLoad_ = helmetRequest.value();

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
              "NuriApplication::requestSceneResources: using fallback IBL "
              "asset "
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

    const auto makeOptionalKtxRequest =
        [&resolveFirstExistingIblAssetPath](
            std::span<const std::filesystem::path> candidates,
            nuri::TextureRequestKind kind,
            std::string_view debugName) -> std::optional<nuri::TextureRequest> {
      if (candidates.empty()) {
        return std::nullopt;
      }
      const std::filesystem::path resolvedPath =
          resolveFirstExistingIblAssetPath(candidates);
      std::error_code ec;
      if (!std::filesystem::exists(resolvedPath, ec) ||
          !std::filesystem::is_regular_file(resolvedPath, ec)) {
        NURI_LOG_WARNING(
            "NuriApplication::requestSceneResources: missing optional IBL "
            "asset '%s'",
            candidates.front().string().c_str());
        return std::nullopt;
      }
      return nuri::TextureRequest{
          .path = resolvedPath.string(),
          .kind = kind,
          .debugName = std::string(debugName),
      };
    };

    auto environmentRequest =
        assets.requestEnvironment(nuri::EnvironmentAssetRequest{
            .textures =
                {
                    nuri::TextureRequest{
                        .path = environmentHdrPath,
                        .kind = nuri::TextureRequestKind::EquirectHdrCubemap,
                        .debugName = "cubemap",
                    },
                    makeOptionalKtxRequest(
                        irradianceCandidates,
                        nuri::TextureRequestKind::Ktx2Cubemap,
                        "ibl_irradiance"),
                    makeOptionalKtxRequest(
                        prefilteredGgxCandidates,
                        nuri::TextureRequestKind::Ktx2Cubemap,
                        "ibl_prefilter_ggx"),
                    makeOptionalKtxRequest(
                        prefilteredCharlieCandidates,
                        nuri::TextureRequestKind::Ktx2Cubemap,
                        "ibl_prefilter_charlie"),
                    makeOptionalKtxRequest(
                        brdfLutCandidates,
                        nuri::TextureRequestKind::Ktx2Texture2D,
                        "ibl_brdf_lut"),
                },
            .priority = nuri::AssetPriority::Visible,
            .optionalTextures = {false, true, true, true, true},
            .debugName = "sample_environment",
        });
    NURI_ASSERT(!environmentRequest.hasError(),
                "Failed to request sample environment: %s",
                environmentRequest.error().c_str());
    environmentLoad_ = environmentRequest.value();

    renderSettings_.opaque.enableInstanceCompute = false;
    renderSettings_.opaque.enableMeshLod = true;
    renderSettings_.opaque.meshletMode = nuri::MeshletRenderMode::Opportunistic;
    renderSettings_.opaque.enableTessellation = false;
    renderSettings_.opaque.forcedMeshLod = -1;
    renderSettings_.opaque.meshLodDistanceThresholds =
        glm::vec3(8.0f, 16.0f, 32.0f);
    renderSettings_.opaque.enableInstanceAnimation = false;

    helmetRotationRadians_ = 0.0f;
    helmetNode_ = nuri::kInvalidNodeId;
    helmetRootLocal_ = glm::mat4(1.0f);
    helmetTransformBound_ = false;
    helmetCameraConfigured_ = false;
    defaultLightAdded_ = false;
  }

  void pollSceneResources() {
    nuri::AssetSystem &assets = getRenderer().assets();
    if (nuri::isValid(environmentLoad_)) {
      const nuri::AssetLoadSnapshot environment =
          assets.query(environmentLoad_);
      NURI_ASSERT(environment.state != nuri::AssetState::Failed,
                  "Sample environment load failed: %s",
                  environment.error.c_str());
    }
    if (!nuri::isValid(helmetSceneLoad_)) {
      return;
    }

    const nuri::SceneLoadSnapshot sceneLoad = assets.query(helmetSceneLoad_);
    NURI_ASSERT(sceneLoad.state != nuri::SceneLoadState::Failed,
                "DamagedHelmet scene load failed: %s", sceneLoad.error.c_str());

    const nuri::ScenePrefab *prefab =
        assets.tryGetScenePrefab(helmetSceneLoad_);
    const std::optional<nuri::SceneInstantiationMap> instantiation =
        assets.tryGetSceneInstantiation(helmetSceneLoad_);
    if (!helmetTransformBound_ && prefab != nullptr &&
        instantiation.has_value()) {
      for (uint32_t nodeIndex = 0u; nodeIndex < prefab->nodes.size();
           ++nodeIndex) {
        if (prefab->nodes[nodeIndex].parentIndex !=
                nuri::kInvalidScenePrefabIndex ||
            nodeIndex >= instantiation->nodes.size() ||
            !nuri::isValid(instantiation->nodes[nodeIndex])) {
          continue;
        }
        helmetNode_ = instantiation->nodes[nodeIndex];
        helmetRootLocal_ = prefab->nodes[nodeIndex].localFromParent;
        const bool transformed = scene_.graph().setNodeLocalTransform(
            helmetNode_, helmetBaseModel_ * helmetRootLocal_);
        NURI_ASSERT(transformed,
                    "Failed to apply DamagedHelmet root transform");
        helmetTransformBound_ = true;
        break;
      }
    }

    if (!helmetCameraConfigured_) {
      const std::optional<nuri::ScenePrefabAssets> readyAssets =
          assets.tryGetSceneAssets(helmetSceneLoad_);
      nuri::ModelRef helmetPrimaryModel = nuri::kInvalidModelRef;
      if (readyAssets.has_value()) {
        for (const nuri::ModelRef model : readyAssets->models) {
          if (nuri::isValid(model)) {
            helmetPrimaryModel = model;
            break;
          }
        }
      }
      if (nuri::isValid(helmetPrimaryModel)) {
        const nuri::ModelRecord *helmetRecord =
            getRenderer().resources().tryGet(helmetPrimaryModel);
        NURI_ASSERT(helmetRecord != nullptr && helmetRecord->model != nullptr,
                    "DamagedHelmet model record lookup failed");
        const nuri::BoundingBox &bounds = helmetRecord->model->bounds();
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
        perspective.farPlane =
            std::max(500.0f, cameraDistance + radius * 12.0f);
        camera->setProjectionType(nuri::ProjectionType::Perspective);
        camera->setPerspective(perspective);
        camera->setLookAt(center + glm::vec3(-cameraDistance * 0.38f,
                                             radius * 0.18f + 0.2f,
                                             -cameraDistance),
                          center + glm::vec3(0.0f, radius * 0.03f, 0.0f),
                          glm::vec3(0.0f, 1.0f, 0.0f));
        helmetCameraConfigured_ = true;
      }
    }

    if (sceneLoad.terminal() && !defaultLightAdded_ &&
        instantiation.has_value()) {
      if (instantiation->lights.empty()) {
        addDefaultDirectionalLight();
      }
      defaultLightAdded_ = true;
    }
  }

  void cancelSceneResourceRequests() {
    nuri::AssetSystem &assets = getRenderer().assets();
    if (nuri::isValid(helmetSceneLoad_)) {
      assets.cancel(helmetSceneLoad_);
      helmetSceneLoad_ = {};
    }
    if (nuri::isValid(environmentLoad_)) {
      assets.cancel(environmentLoad_);
      environmentLoad_ = {};
    }
    helmetNode_ = nuri::kInvalidNodeId;
    helmetTransformBound_ = false;
    helmetCameraConfigured_ = false;
    defaultLightAdded_ = false;
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

    const glm::mat4 modelMatrix =
        glm::rotate(helmetBaseModel_, helmetRotationRadians_,
                    glm::vec3(0.0f, 0.0f, 1.0f)) *
        helmetRootLocal_;
    const bool updated =
        scene_.graph().setNodeLocalTransform(helmetNode_, modelMatrix);
    NURI_ASSERT(updated, "Failed to update DamagedHelmet node transform");
  }

  void buildFrameContext(const nuri::Camera &camera, double timeSeconds) {
    nuri::ResolvedRenderSettings resolvedSettings =
        nuri::resolveRenderSettings(renderSettings_);
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
    auto planResult = nuri::buildPresentationAAPlan(
        resolvedSettings, {}, getGPU().getMultisampleCapabilities());
    NURI_ASSERT(!planResult.hasError(), "Invalid presentation AA plan: %s",
                planResult.error().c_str());
    frameContext_.presentationAA = planResult.value();
    auto cameraResult = temporalFrameService_.prepareFrame(
        camera, getAspectRatio(), resolvedSettings.antiAliasing,
        frameContext_.presentationAA,
        nuri::TemporalCameraFrameDesc{
            .renderExtent =
                glm::uvec2(static_cast<uint32_t>(std::max(getWidth(), 0)),
                           static_cast<uint32_t>(std::max(getHeight(), 0))),
            .sceneContent = sceneContent,
        },
        frameContext_.frameIndex, timeSeconds, frameDeltaSeconds_);
    NURI_ASSERT(!cameraResult.hasError(), "Temporal frame prepare failed: %s",
                cameraResult.error().c_str());
    frameContext_.camera = cameraResult.value();
    frameContext_.temporalFrameService = &temporalFrameService_;
    renderSettings_.antiAliasing.debug.resetHistoryRequested = false;
    resolvedSettings.antiAliasing.debug.resetHistoryRequested = false;
    frameContext_.settings = std::move(resolvedSettings);
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

    const nuri::OpaqueFrameMetrics &opaqueMetrics =
        frameContext_.metrics.opaque;
    const nuri::VisibilityFrameMetrics &visibilityMetrics =
        frameContext_.metrics.visibility;
    meshletMetricsInitialized_ = true;
    meshletActiveLastFrame_ = opaqueMetrics.meshletModeActive != 0u;
    meshletFallbackLastFrame_ =
        opaqueMetrics.meshletRejectedMissingFeature != 0u ||
        opaqueMetrics.meshletRejectedMissingAssetData != 0u ||
        opaqueMetrics.meshletRejectedIncompatibleFrame != 0u;
    meshletTaskGroupsLastFrame_ =
        nuri::resolveGeometryWorkMetrics(frameContext_.metrics)
            .executedMeshletTaskGroups;
    mainVisibilityLastFrame_ = {
        .available = visibilityMetrics.gpuMainReadbackAvailable,
        .sourceFrame = visibilityMetrics.gpuMainReadbackSourceFrame,
        .staleFrameCount = visibilityMetrics.gpuMainReadbackStaleFrameCount,
        .errorCount = visibilityMetrics.gpuMainReadbackErrorCount,
        .occlusionAvailable = visibilityMetrics.occlusionAvailable,
    };
    meshletVisibilityLastFrame_ = {
        .available = visibilityMetrics.meshletReadbackAvailable,
        .sourceFrame = visibilityMetrics.meshletReadbackSourceFrame,
        .staleFrameCount = visibilityMetrics.meshletReadbackStaleFrameCount,
        .errorCount = visibilityMetrics.meshletReadbackErrorCount,
        .occlusionAvailable = visibilityMetrics.meshletOcclusionAvailable,
    };
  }

  const nuri::RuntimeConfig config_;
  std::pmr::unsynchronized_pool_resource cameraMemory_;
  std::pmr::unsynchronized_pool_resource sceneMemory_;
  nuri::CameraSystem cameraSystem_;
  nuri::RenderScene scene_;
  nuri::SceneRuntimeHost sceneRuntime_;
  nuri::SceneLoadHandle helmetSceneLoad_{};
  nuri::EnvironmentAssetHandle environmentLoad_{};
  nuri::CameraHandle mainCameraHandle_{};
  nuri::NodeId helmetNode_ = nuri::kInvalidNodeId;
  glm::mat4 helmetBaseModel_ =
      glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
  glm::mat4 helmetRootLocal_{1.0f};

  nuri::RenderSettings renderSettings_{};
  nuri::RenderFrameContext frameContext_{};
  nuri::TemporalFrameService temporalFrameService_{};
  uint64_t frameIndex_ = 0;
  uint64_t simulationFrameIndex_ = 0;
  double frameDeltaSeconds_ = 0.0;
  double fpsAccumulatorSeconds_ = 0.0;
  uint32_t fpsFrameCount_ = 0;
  float currentFps_ = 0.0f;
  float helmetRotationRadians_ = 0.0f;
  uint32_t meshletTaskGroupsLastFrame_ = 0;
  VisibilityReadbackOverlayState mainVisibilityLastFrame_{};
  VisibilityReadbackOverlayState meshletVisibilityLastFrame_{};
  std::unique_ptr<nuri::TextSystem> textSystem_{};
  nuri::ScratchArena textScratchArena_{};
  bool textOverlayEnabled_ = false;
  bool meshletMetricsInitialized_ = false;
  bool meshletActiveLastFrame_ = false;
  bool meshletFallbackLastFrame_ = false;
  bool helmetTransformBound_ = false;
  bool helmetCameraConfigured_ = false;
  bool defaultLightAdded_ = false;
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
