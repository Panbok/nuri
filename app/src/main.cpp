#include "nuri/pch.h"

#include "nuri/core/application.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/gfx/layers/opaque_layer.h"
#include "nuri/gfx/layers/render_frame_context.h"
#include "nuri/gfx/layers/skybox_layer.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/scene/camera_system.h"
#include "nuri/scene/render_scene.h"
#include "nuri/text/text_layer_2d.h"
#include "nuri/text/text_system.h"

#include <glm/gtc/matrix_transform.hpp>

namespace {

constexpr std::string_view kDamagedHelmetModelRelativePath =
    "DamagedHelmet/DamagedHelmet.gltf";
constexpr std::string_view kSampleEnvironmentHdrRelativePath =
    "qwantani_moon_noon_puresky_4k.hdr";
constexpr float kHelmetRotationSpeedRadians = glm::radians(35.0f);

nuri::ApplicationConfig toApplicationConfig(const nuri::RuntimeConfig &config) {
  return nuri::ApplicationConfig{
      .title = config.window.title,
      .width = config.window.width,
      .height = config.window.height,
      .windowMode = config.window.mode,
  };
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

class NuriApplication : public nuri::Application {
public:
  explicit NuriApplication(nuri::RuntimeConfig config)
      : nuri::Application(toApplicationConfig(config)),
        config_(std::move(config)), cameraSystem_(cameraMemory_),
        scene_(&sceneMemory_) {}

  void onInit() override {
    NURI_PROFILER_FUNCTION();
    scene_.bindResources(&getRenderer().resources());
    initializeCamera();
    initializeTextSystem();
    loadSceneResources();
    initializeRenderLayers();
    initializeTextOverlayLayer();

    NURI_LOG_INFO("Application was initialized");
  }

  void onDraw() override {
    NURI_PROFILER_FUNCTION();

    const nuri::Camera *activeCamera = cameraSystem_.activeCamera();
    NURI_ASSERT(activeCamera != nullptr, "No active camera");

    buildFrameContext(*activeCamera, getTime());
    queuePerformanceOverlay();
    submitLayeredFrame();
  }

  void onUpdate(double deltaTime) override {
    updatePerformanceMetrics(deltaTime);
    updateHelmetRotation(deltaTime);
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
    scene_.clearOpaqueRenderables();
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

  void initializeRenderLayers() {
    auto skyboxLayer =
        nuri::SkyboxLayer::create(getGPU(), config_.shaders.skybox);
    NURI_ASSERT(skyboxLayer != nullptr, "Failed to create skybox layer");
    NURI_ASSERT(getLayerStack().pushLayer(std::move(skyboxLayer)) != nullptr,
                "Failed to push skybox layer");

    auto opaqueLayer = nuri::OpaqueLayer::create(
        getGPU(), config_.shaders.opaque, layerMemoryResource());
    NURI_ASSERT(opaqueLayer != nullptr, "Failed to create opaque layer");
    NURI_ASSERT(getLayerStack().pushLayer(std::move(opaqueLayer)) != nullptr,
                "Failed to push opaque layer");
  }

  void initializeTextOverlayLayer() {
    if (textLayer2D_ != nullptr || textSystem_ == nullptr) {
      return;
    }

    auto textLayer2D = nuri::TextLayer2D::create({
        .text = *textSystem_,
    });
    NURI_ASSERT(textLayer2D != nullptr, "Failed to create 2D text layer");
    textLayer2D_ = static_cast<nuri::TextLayer2D *>(
        getLayerStack().pushOverlay(std::move(textLayer2D)));
    NURI_ASSERT(textLayer2D_ != nullptr, "Failed to push 2D text layer");
  }

  void initializeTextSystem() {
    const std::filesystem::path defaultFontPath = pickDefaultNfontPath(config_);
    const bool requireDefaultFont =
        std::filesystem::is_regular_file(defaultFontPath);
    NURI_LOG_INFO("NuriApplication::initializeTextSystem: default font '%s'",
                  defaultFontPath.string().c_str());

    auto textSystemResult = nuri::TextSystem::create({
        .gpu = getGPU(),
        .memory = *layerMemoryResource(),
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
    if (textSystem_ == nullptr || textLayer2D_ == nullptr) {
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
    scene_.clearOpaqueRenderables();
    scene_.setEnvironment(nuri::EnvironmentHandles{});
    releaseOwnedResourceHandles();

    const std::string helmetModelPath =
        (config_.roots.models / kDamagedHelmetModelRelativePath).string();
    const std::string environmentHdrPath =
        (config_.roots.textures / kSampleEnvironmentHdrRelativePath).string();

    nuri::MeshImportOptions helmetImportOptions{};
    helmetImportOptions.flipUVs = true;
    auto helmetModelResult = resources.acquireModel(nuri::ModelRequest{
        .path = helmetModelPath,
        .importOptions = helmetImportOptions,
        .debugName = "damaged_helmet",
    });
    NURI_ASSERT(!helmetModelResult.hasError(),
                "Failed to create DamagedHelmet model: %s",
                helmetModelResult.error().c_str());
    helmetModel_ = helmetModelResult.value();

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

    auto importedMaterialsResult =
        resources.acquireMaterialsFromModel(nuri::ImportedMaterialRequest{
            .modelPath = helmetModelPath,
            .model = helmetModel_,
            .debugNamePrefix = "damaged_helmet",
        });
    if (!importedMaterialsResult.hasError() &&
        nuri::isValid(importedMaterialsResult.value().firstMaterial)) {
      helmetMaterial_ = importedMaterialsResult.value().firstMaterial;
    } else {
      if (importedMaterialsResult.hasError()) {
        NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to "
                         "import DamagedHelmet materials from '%s': %s",
                         helmetModelPath.c_str(),
                         importedMaterialsResult.error().c_str());
      }

      auto fallbackMaterialResult =
          resources.acquireMaterial(nuri::MaterialRequest{
              .desc = nuri::MaterialDesc{},
              .debugName = "damaged_helmet_fallback_material",
          });
      NURI_ASSERT(!fallbackMaterialResult.hasError(),
                  "Failed to acquire DamagedHelmet fallback material: %s",
                  fallbackMaterialResult.error().c_str());
      helmetMaterial_ = fallbackMaterialResult.value();
      resources.setModelMaterialForAllSources(helmetModel_, helmetMaterial_);
    }

    const nuri::ModelRecord *helmetRecord = resources.tryGet(helmetModel_);
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
    auto addResult = scene_.addOpaqueRenderable(helmetModel_, helmetMaterial_,
                                                helmetBaseModel_);
    NURI_ASSERT(!addResult.hasError(),
                "Failed to add DamagedHelmet renderable: %s",
                addResult.error().c_str());
    helmetRenderableIndex_ = addResult.value();

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
  }

  void releaseOwnedResourceHandles() {
    nuri::ResourceManager &resources = getRenderer().resources();
    const auto releaseRef = [&resources](auto &ref, const auto invalidRef) {
      if (nuri::isValid(ref)) {
        resources.release(ref);
      }
      ref = invalidRef;
    };

    releaseRef(helmetModel_, nuri::kInvalidModelRef);
    releaseRef(helmetMaterial_, nuri::kInvalidMaterialRef);
    helmetRenderableIndex_ = std::numeric_limits<uint32_t>::max();
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
    if (helmetRenderableIndex_ == std::numeric_limits<uint32_t>::max() ||
        !std::isfinite(deltaTime) || deltaTime <= 0.0) {
      return;
    }

    helmetRotationRadians_ +=
        static_cast<float>(deltaTime) * kHelmetRotationSpeedRadians;
    if (helmetRotationRadians_ >= glm::radians(360.0f)) {
      helmetRotationRadians_ -= glm::radians(360.0f);
    }

    const glm::mat4 modelMatrix = glm::rotate(
        helmetBaseModel_, helmetRotationRadians_, glm::vec3(0.0f, 0.0f, 1.0f));
    const bool updated = scene_.setOpaqueRenderableTransform(
        helmetRenderableIndex_, modelMatrix);
    NURI_ASSERT(updated, "Failed to update DamagedHelmet transform");
  }

  void buildFrameContext(const nuri::Camera &camera, double timeSeconds) {
    frameContext_.scene = &scene_;
    frameContext_.resources = &getRenderer().resources();
    frameContext_.camera.view = camera.viewMatrix();
    frameContext_.camera.proj = camera.projectionMatrix(getAspectRatio());
    frameContext_.camera.cameraPos = glm::vec4(camera.position(), 1.0f);
    frameContext_.camera.aspectRatio = getAspectRatio();
    frameContext_.settings = &renderSettings_;
    frameContext_.metrics = {};
    frameContext_.channels.clear();
    frameContext_.sharedDepthTexture = {};
    frameContext_.timeSeconds = timeSeconds;
    frameContext_.frameIndex = frameIndex_++;
  }

  void submitLayeredFrame() {
    auto renderResult = getRenderer().render(getLayerStack(), frameContext_);
    NURI_ASSERT(!renderResult.hasError(), "Render failed: %s",
                renderResult.error().c_str());
  }

  const nuri::RuntimeConfig config_;
  std::pmr::unsynchronized_pool_resource cameraMemory_;
  std::pmr::unsynchronized_pool_resource sceneMemory_;
  nuri::CameraSystem cameraSystem_;
  nuri::RenderScene scene_;
  nuri::ModelRef helmetModel_ = nuri::kInvalidModelRef;
  nuri::MaterialRef helmetMaterial_ = nuri::kInvalidMaterialRef;
  nuri::CameraHandle mainCameraHandle_{};
  uint32_t helmetRenderableIndex_ = std::numeric_limits<uint32_t>::max();
  glm::mat4 helmetBaseModel_ =
      glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));

  nuri::RenderSettings renderSettings_{};
  nuri::RenderFrameContext frameContext_{};
  uint64_t frameIndex_ = 0;
  double frameDeltaSeconds_ = 0.0;
  double fpsAccumulatorSeconds_ = 0.0;
  uint32_t fpsFrameCount_ = 0;
  float currentFps_ = 0.0f;
  float helmetRotationRadians_ = 0.0f;
  std::unique_ptr<nuri::TextSystem> textSystem_{};
  nuri::ScratchArena textScratchArena_{};
  nuri::TextLayer2D *textLayer2D_ = nullptr;
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
