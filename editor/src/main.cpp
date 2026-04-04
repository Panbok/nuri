#include "nuri/editor_pch.h"

#include "nuri/bakery/bakery_system.h"
#include "nuri/core/application.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/pipeline/features/text_feature.h"
#include "nuri/gfx/sim/animation_gpu_services.h"
#include "nuri/gfx/sim/animation_scene_frame_provider.h"
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
#include "nuri/ui/camera_controller_widget.h"
#include "nuri/ui/editor_feature.h"
#include "nuri/ui/editor_overlay_controller.h"


#include <ctime>
#include <glm/gtc/matrix_transform.hpp>

namespace {

enum class ScenePreset : uint8_t {
  SingleDuck,
  InstancedDuck32K,
  NiagaraBistro,
  DamagedHelmet,
  LightsPunctualLamp,
  ClearcoatWicker,
  SheenChair,
  SpecularSilkPouf,
  DragonAttenuation,
  DragonIor,
  EmissiveStrengthTest,
  Orrey,
  Fox,
  MedievalFantasyBook,
  Text3DTest,
};

constexpr ScenePreset kScenePreset = ScenePreset::Fox;
constexpr uint32_t kDuckGridSide = 32;
constexpr uint32_t kDuckInstanceCount =
    kDuckGridSide * kDuckGridSide * kDuckGridSide;
constexpr float kDuckSpacing = 18.0f;
constexpr float kDuckJitter = 3.0f;
constexpr std::string_view kSampleDuckModelRelativePath =
    "rubber_duck/scene.gltf";
constexpr std::string_view kSampleDuckAlbedoRelativePath =
    "rubber_duck/textures/Duck_baseColor.png";
constexpr std::string_view kSampleEnvironmentHdrRelativePath =
    "qwantani_moon_noon_puresky_4k.hdr";
constexpr std::string_view kNiagaraBistroModelRelativePath =
    "NiagaraBistro/bistrox.gltf";
constexpr std::string_view kNiagaraBistroAbsolutePath =
    "E:/install/nuri/assets/models/NiagaraBistro/bistrox.gltf";
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
constexpr float kNiagaraBistroTargetRadius = 120.0f;
constexpr float kNiagaraBistroMinScale = 0.0005f;
constexpr float kNiagaraBistroMaxScale = 2.0f;
constexpr const char *kScenePresetNames[] = {
    "Single Duck",    "Instanced Duck 32K",     "Niagara Bistro",
    "Damaged Helmet", "Lights Punctual Lamp",   "Clearcoat Wicker",
    "Sheen Chair",    "Specular Silk Pouf",     "Dragon Attenuation",
    "Dragon IOR",     "Emissive Strength Test", "Orrey",
    "Fox Skinning",   "Medieval Fantasy Book",  "Text 3D Test"};

[[nodiscard]] glm::mat4 makeTransformMatrix(const glm::vec3 &position,
                                            const glm::quat &rotation) {
  return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotation);
}

int scenePresetToIndex(ScenePreset preset) {
  switch (preset) {
  case ScenePreset::SingleDuck:
    return 0;
  case ScenePreset::InstancedDuck32K:
    return 1;
  case ScenePreset::NiagaraBistro:
    return 2;
  case ScenePreset::DamagedHelmet:
    return 3;
  case ScenePreset::LightsPunctualLamp:
    return 4;
  case ScenePreset::ClearcoatWicker:
    return 5;
  case ScenePreset::SheenChair:
    return 6;
  case ScenePreset::SpecularSilkPouf:
    return 7;
  case ScenePreset::DragonAttenuation:
    return 8;
  case ScenePreset::DragonIor:
    return 9;
  case ScenePreset::EmissiveStrengthTest:
    return 10;
  case ScenePreset::Orrey:
    return 11;
  case ScenePreset::Fox:
    return 12;
  case ScenePreset::MedievalFantasyBook:
    return 13;
  case ScenePreset::Text3DTest:
    return 14;
  }
  return 0;
}

ScenePreset scenePresetFromIndex(int index) {
  switch (index) {
  case 0:
    return ScenePreset::SingleDuck;
  case 1:
    return ScenePreset::InstancedDuck32K;
  case 2:
    return ScenePreset::NiagaraBistro;
  case 3:
    return ScenePreset::DamagedHelmet;
  case 4:
    return ScenePreset::LightsPunctualLamp;
  case 5:
    return ScenePreset::ClearcoatWicker;
  case 6:
    return ScenePreset::SheenChair;
  case 7:
    return ScenePreset::SpecularSilkPouf;
  case 8:
    return ScenePreset::DragonAttenuation;
  case 9:
    return ScenePreset::DragonIor;
  case 10:
    return ScenePreset::EmissiveStrengthTest;
  case 11:
    return ScenePreset::Orrey;
  case 12:
    return ScenePreset::Fox;
  case 13:
    return ScenePreset::MedievalFantasyBook;
  case 14:
    return ScenePreset::Text3DTest;
  default:
    return ScenePreset::SingleDuck;
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

glm::vec3 instancePositionFromGrid(uint32_t index) {
  const uint32_t x = index % kDuckGridSide;
  const uint32_t y = (index / kDuckGridSide) % kDuckGridSide;
  const uint32_t z = index / (kDuckGridSide * kDuckGridSide);

  const glm::vec3 centered =
      glm::vec3(static_cast<float>(x), static_cast<float>(y),
                static_cast<float>(z)) -
      glm::vec3((static_cast<float>(kDuckGridSide) - 1.0f) * 0.5f);
  glm::vec3 pos = centered * kDuckSpacing;

  const glm::vec3 jitter(
      (hashToUnitFloat(index * 3u + 1u) - 0.5f) * 2.0f * kDuckJitter,
      (hashToUnitFloat(index * 3u + 2u) - 0.5f) * 2.0f * kDuckJitter,
      (hashToUnitFloat(index * 3u + 3u) - 0.5f) * 2.0f * kDuckJitter);
  pos += jitter;
  return pos;
}

float computeNiagaraBistroScale(const nuri::BoundingBox &bounds) {
  const float rawRadius =
      std::max(0.5f * glm::length(bounds.getSize()), 1.0e-3f);
  const float targetScale = kNiagaraBistroTargetRadius / rawRadius;
  return std::clamp(targetScale, kNiagaraBistroMinScale,
                    kNiagaraBistroMaxScale);
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

[[nodiscard]] std::array<float, 16> encodeWorld(const glm::mat4 &m) {
  std::array<float, 16> out{};
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      out[static_cast<size_t>(c * 4 + r)] = m[c][r];
    }
  }
  return out;
}

[[nodiscard]] std::string formatLocalTimeHhMmSs() {
  const std::time_t now = std::time(nullptr);
  std::tm localTime{};
#if defined(_WIN32)
  if (localtime_s(&localTime, &now) != 0) {
    return "00:00:00";
  }
#else
  if (localtime_r(&now, &localTime) == nullptr) {
    return "00:00:00";
  }
#endif

  std::array<char, 16> buffer{};
  if (std::strftime(buffer.data(), buffer.size(), "%H:%M:%S", &localTime) ==
      0) {
    return "00:00:00";
  }
  return std::string(buffer.data());
}

} // namespace

namespace {

[[nodiscard]] nuri::ApplicationConfig
makeEditorApplicationConfig(const nuri::RuntimeConfig &config) {
  nuri::ApplicationConfig appConfig = nuri::makeApplicationConfig(config);
  appConfig.renderComposition = nuri::RenderCompositionMode::PipelineOnly;
  return appConfig;
}

} // namespace

class NuriApplication : public nuri::Application {
public:
  explicit NuriApplication(nuri::RuntimeConfig config)
      : nuri::Application(makeEditorApplicationConfig(config)),
        config_(std::move(config)), cameraSystem_(cameraMemory_),
        scene_(&sceneMemory_), sceneRuntime_(&sceneMemory_),
        foxAnimation_(&sceneMemory_),
        medievalFantasyBookAnimation_(&sceneMemory_) {}

  void onInit() override {
    NURI_PROFILER_FUNCTION();
    scene_.bindResources(&getRenderer().resources());
    sceneRuntime_.bindScene(&scene_);
    animationGpuServices_ = std::make_unique<nuri::AnimationGpuServices>(
        getGPU(), config_.roots.shaders, pipelineMemoryResource());
    sceneRuntime_.attachAnimationGpuServices(animationGpuServices_.get());
    auto *animationProvider = getRenderPipeline().addProvider(
        std::make_unique<nuri::AnimationSceneFrameProvider>(sceneRuntime_));
    NURI_ASSERT(animationProvider != nullptr,
                "Failed to register animation scene frame provider");
    auto bakeryResult = nuri::bakery::BakerySystem::create({
        .gpu = getGPU(),
        .config = config_,
        .profile = nuri::bakery::BakeryExecutionProfile::Interactive,
    });
    if (bakeryResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::onInit: failed to create bakery "
                       "system: %s",
                       bakeryResult.error().c_str());
    } else {
      bakerySystem_ = std::move(bakeryResult.value());
    }
    initializeCamera();
    initializeTextSystem();
    initializeEditorRenderFeature();
    loadSceneResources();
    initializeEditorOverlay();

    NURI_LOG_INFO("Application was initialized");
  }

  void onDraw() override {
    NURI_PROFILER_FUNCTION();

    const nuri::Camera *activeCamera = cameraSystem_.activeCamera();
    NURI_ASSERT(activeCamera != nullptr, "No active camera");

    applyPendingScenePreset();
    updateNiagaraBistroSceneStreaming();

    const double timeSeconds = getTime();
    auto commitResult = scene_.commit();
    NURI_ASSERT(!commitResult.hasError(), "Scene commit failed: %s",
                commitResult.error().c_str());
    buildFrameContext(*activeCamera, timeSeconds);
    queueTextSamples();
    if (editorOverlay_ != nullptr) {
      const int presetIndex = scenePresetToIndex(scenePreset_);
      const std::span<const char *const> presetNames{
          kScenePresetNames, static_cast<size_t>(sizeof(kScenePresetNames) /
                                                 sizeof(kScenePresetNames[0]))};
      editorOverlay_->setScenePresetUi(presetNames, presetIndex,
                                       "Toggle Editor: F6");
    }
    submitPipelineFrame();
    if (editorOverlay_ != nullptr) {
      if (const auto requestedIndex =
              editorOverlay_->takeScenePresetSelectionRequest()) {
        const ScenePreset selectedPreset =
            scenePresetFromIndex(*requestedIndex);
        if (selectedPreset != scenePreset_) {
          requestScenePreset(selectedPreset);
        }
      }
    }
  }

  void onUpdate(double deltaTime) override {
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
    if (editorOverlay_ != nullptr) {
      editorOverlay_->onUpdate(deltaTime);
    }
    (void)sceneRuntime_.tick({
        .frameDeltaSeconds = std::max(0.0, deltaTime),
        .absoluteTimeSeconds = getTime(),
        .frameIndex = simulationFrameIndex_++,
    });
    cameraSystem_.update(deltaTime, getInput());
    if (bakerySystem_) {
      bakerySystem_->tick();
    }
  }

  void onResize(std::int32_t, std::int32_t) override {}

  bool onInput(const nuri::InputEvent &event) override {
    if (event.type == nuri::InputEventType::Key &&
        event.payload.key.action == nuri::KeyAction::Press &&
        event.payload.key.key == nuri::Key::F6) {
      toggleEditorOverlay();
      return true;
    }

    if (editorOverlay_ != nullptr && editorOverlay_->onInput(event)) {
      return true;
    }
    if (cameraSystem_.onInput(event, getWindow())) {
      return true;
    }
    return nuri::Application::onInput(event);
  }

  void onShutdown() override {
    destroyFoxAnimation();
    destroyMedievalFantasyBookAnimation();
    sceneRuntime_.reset();
    sceneRuntime_.bindScene(nullptr);
    scene_.graph().clear();
    scene_.setEnvironment(nuri::EnvironmentHandles{});
    releaseOwnedResourceHandles();
    scene_.bindResources(nullptr);
    bistroAsyncLoad_.reset();
    getWindow().setCursorMode(nuri::CursorMode::Normal);
    NURI_LOG_INFO("Application was shutdown");
  }

private:
  struct FramedSceneCameraState {
    glm::vec3 center{0.0f};
    float rawRadius = 0.25f;
    float radius = 0.25f;
    float cameraDistance = 2.0f;
    float nearPlane = 0.01f;
    float farPlane = 500.0f;
  };

  struct ImportedPrefabSceneResources {
    nuri::ScenePrefab prefab{};
    nuri::ScenePrefabAssets assets{};
    std::vector<nuri::ImportedSceneLight> fallbackLights{};
    bool ready = false;
  };

  struct AnimatedPrefabSceneInstance {
    explicit AnimatedPrefabSceneInstance(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : instantiationMap(memory) {}

    nuri::NodeId rootNode = nuri::kInvalidNodeId;
    nuri::SceneInstantiationMap instantiationMap;
    nuri::SimulationHandle simulation = nuri::kInvalidSimulationHandle;
  };

  void initializeCamera() {
    nuri::Camera camera{};
    camera.setLookAt(glm::vec3(0.0f, 1.0f, -1.5f), glm::vec3(0.0f, 0.5f, 0.0f),
                     glm::vec3(0.0f, 1.0f, 0.0f));
    camera.setProjectionType(nuri::ProjectionType::Perspective);

    nuri::CameraController controller = nuri::makeFpsDirectController();
    mainCameraHandle_ = cameraSystem_.addCamera(camera, std::move(controller));
    NURI_ASSERT(mainCameraHandle_.isValid(),
                "Failed to add camera to camera system");

    const bool setActive =
        cameraSystem_.setActiveCamera(mainCameraHandle_, getWindow());
    NURI_ASSERT(setActive, "Failed to activate main camera");

    syncEditorCameraWidgetState(camera);
    NURI_LOG_INFO("NuriApplication::initializeCamera: Main camera initialized "
                  "(WASD/QE move, RMB look, P projection toggle, ImGui camera "
                  "controller panel)");
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

    auto *text3DFeature = getRenderPipeline().addFeature(
        std::make_unique<nuri::Text3DFeature>(*textSystem_));
    NURI_ASSERT(text3DFeature != nullptr, "Failed to register 3D text feature");
    text3DEnabled_ = true;

    auto *text2DFeature = getRenderPipeline().addFeature(
        std::make_unique<nuri::Text2DFeature>(*textSystem_));
    NURI_ASSERT(text2DFeature != nullptr, "Failed to register 2D text feature");
  }

  void initializeEditorRenderFeature() {
    if (editorRenderFeature_ != nullptr) {
      return;
    }
    auto feature = std::make_unique<nuri::EditorOverlayFeature>();
    editorRenderFeature_ = feature.get();
    auto *registered = getRenderPipeline().addFeature(std::move(feature));
    NURI_ASSERT(registered != nullptr,
                "Failed to register editor overlay feature");
  }

  void queueTextSamples() {
    if (textSystem_ == nullptr) {
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
    nuri::ScopedScratch scopedScratch(textScratchArena_);
    std::pmr::memory_resource &scratch = *scopedScratch.resource();

    if (textOverlayEnabled_) {
      (void)enqueue2DTextSamples(defaultFont, baseFontSizePx, scratch);
    }

    if (scenePreset_ == ScenePreset::Text3DTest && text3DEnabled_) {
      (void)enqueue3DTextSamples(defaultFont, baseFontSizePx, scratch);
    }
  }

  [[nodiscard]] bool enqueue2DTextSamples(nuri::FontHandle defaultFont,
                                          float baseFontSizePx,
                                          std::pmr::memory_resource &scratch) {
    auto enqueueSample =
        [&](const nuri::Text2DDesc &sample) -> std::optional<nuri::TextBounds> {
      auto enqueue = textSystem_->renderer().enqueue2D(sample, scratch);
      if (enqueue.hasError()) {
        NURI_LOG_WARNING(
            "NuriApplication: failed to enqueue 2D text sample: %s",
            enqueue.error().c_str());
        return std::nullopt;
      }
      return enqueue.value();
    };
    const auto configureAsciiLayout = [](nuri::TextLayoutParams &layout) {
      (void)layout;
    };

    nuri::Text2DDesc headline{};
    headline.utf8 = "MTSDF 2D Raster Test 0123456789 AaBbCc";
    headline.style.font = defaultFont;
    headline.style.pxSize = baseFontSizePx;
    headline.layout.alignH = nuri::TextAlignH::Left;
    headline.layout.alignV = nuri::TextAlignV::Top;
    configureAsciiLayout(headline.layout);
    headline.fillColor = {.r = 1.0f, .g = 1.0f, .b = 1.0f, .a = 1.0f};
    headline.x = 20.0f;
    headline.y = 20.0f;
    const std::optional<nuri::TextBounds> headlineBounds =
        enqueueSample(headline);
    if (!headlineBounds.has_value()) {
      return false;
    }

    nuri::Text2DDesc kerning{};
    kerning.utf8 = "Kerning: AV AVATAR To WA TA YA LT";
    kerning.style.font = defaultFont;
    kerning.style.pxSize = baseFontSizePx * 0.67f;
    kerning.layout.alignH = nuri::TextAlignH::Left;
    kerning.layout.alignV = nuri::TextAlignV::Top;
    configureAsciiLayout(kerning.layout);
    kerning.fillColor = {.r = 0.86f, .g = 0.92f, .b = 1.0f, .a = 1.0f};
    kerning.x = 20.0f;
    kerning.y = headlineBounds->maxY + 14.0f;
    const std::optional<nuri::TextBounds> kerningBounds =
        enqueueSample(kerning);
    if (!kerningBounds.has_value()) {
      return false;
    }

    nuri::Text2DDesc wrap{};
    wrap.utf8 = "Wrap(320px): The quick brown fox jumps over the lazy dog near "
                "the river bank.";
    wrap.style.font = defaultFont;
    wrap.style.pxSize = baseFontSizePx * 0.62f;
    wrap.layout.wrapMode = nuri::TextWrapMode::Word;
    wrap.layout.maxWidthPx = 320.0f;
    wrap.layout.alignH = nuri::TextAlignH::Left;
    wrap.layout.alignV = nuri::TextAlignV::Top;
    configureAsciiLayout(wrap.layout);
    wrap.fillColor = {.r = 0.9f, .g = 1.0f, .b = 0.9f, .a = 1.0f};
    wrap.x = 20.0f;
    wrap.y = kerningBounds->maxY + 14.0f;
    const std::optional<nuri::TextBounds> wrapBounds = enqueueSample(wrap);
    if (!wrapBounds.has_value()) {
      return false;
    }

    nuri::Text2DDesc multiline{};
    multiline.utf8 = "Manual newline:\nLine 1\nLine 2\nLine 3";
    multiline.style.font = defaultFont;
    multiline.style.pxSize = baseFontSizePx * 0.57f;
    multiline.layout.alignH = nuri::TextAlignH::Left;
    multiline.layout.alignV = nuri::TextAlignV::Top;
    configureAsciiLayout(multiline.layout);
    multiline.fillColor = {.r = 1.0f, .g = 0.95f, .b = 0.85f, .a = 1.0f};
    multiline.x = 20.0f;
    multiline.y = wrapBounds->maxY + 18.0f;
    if (!enqueueSample(multiline).has_value()) {
      return false;
    }

    int32_t windowWidth = 0;
    int32_t windowHeight = 0;
    getWindow().getWindowSize(windowWidth, windowHeight);
    if (windowWidth <= 0 || windowHeight <= 0) {
      getWindow().getFramebufferSize(windowWidth, windowHeight);
    }
    const float overlayWidth =
        std::max(static_cast<float>(windowWidth) - 40.0f, 0.0f);
    if (overlayWidth > 0.0f) {
      const float fps = currentFps_;
      const float frameTimeMs = fps > 0.0f ? 1000.0f / fps : 0.0f;
      std::array<char, 96> perfText{};
      std::snprintf(perfText.data(), perfText.size(), "FPS: %.1f\nFT: %.2f ms",
                    fps, frameTimeMs);

      nuri::Text2DDesc perf{};
      perf.utf8 = perfText.data();
      perf.style.font = defaultFont;
      perf.style.pxSize = baseFontSizePx * 0.52f;
      perf.layout.alignH = nuri::TextAlignH::Right;
      perf.layout.alignV = nuri::TextAlignV::Top;
      perf.layout.maxWidthPx = overlayWidth;
      configureAsciiLayout(perf.layout);
      perf.fillColor = {.r = 0.95f, .g = 1.0f, .b = 0.95f, .a = 1.0f};
      perf.x = 20.0f;
      perf.y = 20.0f;
      if (!enqueueSample(perf).has_value()) {
        return false;
      }
    }

    return true;
  }

  [[nodiscard]] bool enqueue3DTextSamples(nuri::FontHandle defaultFont,
                                          float baseFontSizePx,
                                          std::pmr::memory_resource &scratch) {
    auto enqueueSample =
        [&](const nuri::Text3DDesc &sample) -> std::optional<nuri::TextBounds> {
      auto enqueue = textSystem_->renderer().enqueue3D(sample, scratch);
      if (enqueue.hasError()) {
        NURI_LOG_WARNING(
            "NuriApplication: failed to enqueue 3D text sample: %s",
            enqueue.error().c_str());
        return std::nullopt;
      }
      return enqueue.value();
    };
    const auto configureAsciiLayout = [](nuri::TextLayoutParams &layout) {
      (void)layout;
    };

    nuri::Text3DDesc spherical{};
    spherical.utf8 = "MTSDF 3D BILLBOARD";
    spherical.style.font = defaultFont;
    spherical.style.pxSize = baseFontSizePx * 0.75f;
    spherical.layout.alignH = nuri::TextAlignH::Center;
    spherical.layout.alignV = nuri::TextAlignV::Middle;
    configureAsciiLayout(spherical.layout);
    spherical.fillColor = {.r = 1.0f, .g = 1.0f, .b = 1.0f, .a = 1.0f};
    spherical.billboard = nuri::TextBillboardMode::Spherical;
    glm::mat4 sphericalWorld(1.0f);
    sphericalWorld =
        glm::translate(sphericalWorld, glm::vec3(0.0f, 2.2f, 0.0f));
    sphericalWorld = glm::scale(sphericalWorld, glm::vec3(0.025f));
    spherical.worldFromText = encodeWorld(sphericalWorld);
    if (!enqueueSample(spherical).has_value()) {
      return false;
    }

    nuri::Text3DDesc cylindrical{};
    cylindrical.utf8 = "CYLINDRICAL Y";
    cylindrical.style.font = defaultFont;
    cylindrical.style.pxSize = baseFontSizePx * 0.52f;
    cylindrical.layout.alignH = nuri::TextAlignH::Center;
    cylindrical.layout.alignV = nuri::TextAlignV::Middle;
    configureAsciiLayout(cylindrical.layout);
    cylindrical.fillColor = {.r = 0.85f, .g = 1.0f, .b = 0.85f, .a = 1.0f};
    cylindrical.billboard = nuri::TextBillboardMode::CylindricalY;
    glm::mat4 cylindricalWorld(1.0f);
    cylindricalWorld =
        glm::translate(cylindricalWorld, glm::vec3(-2.3f, 1.5f, 0.0f));
    cylindricalWorld = glm::scale(cylindricalWorld, glm::vec3(0.02f));
    cylindrical.worldFromText = encodeWorld(cylindricalWorld);
    if (!enqueueSample(cylindrical).has_value()) {
      return false;
    }

    nuri::Text3DDesc clock{};
    const std::string clockText = formatLocalTimeHhMmSs();
    clock.utf8 = clockText;
    clock.style.font = defaultFont;
    clock.style.pxSize = baseFontSizePx * 0.62f;
    clock.layout.alignH = nuri::TextAlignH::Center;
    clock.layout.alignV = nuri::TextAlignV::Middle;
    configureAsciiLayout(clock.layout);
    clock.fillColor = {.r = 1.0f, .g = 0.95f, .b = 0.82f, .a = 1.0f};
    clock.billboard = nuri::TextBillboardMode::Spherical;
    glm::mat4 clockWorld(1.0f);
    clockWorld = glm::translate(clockWorld, glm::vec3(0.0f, 0.85f, 1.15f));
    clockWorld = glm::scale(clockWorld, glm::vec3(0.02f));
    clock.worldFromText = encodeWorld(clockWorld);
    if (!enqueueSample(clock).has_value()) {
      return false;
    }

    nuri::Text3DDesc fixed{};
    fixed.utf8 = "WORLD FIXED";
    fixed.style.font = defaultFont;
    fixed.style.pxSize = baseFontSizePx * 0.50f;
    fixed.layout.alignH = nuri::TextAlignH::Center;
    fixed.layout.alignV = nuri::TextAlignV::Middle;
    configureAsciiLayout(fixed.layout);
    fixed.fillColor = {.r = 0.85f, .g = 0.9f, .b = 1.0f, .a = 1.0f};
    fixed.billboard = nuri::TextBillboardMode::None;
    glm::mat4 fixedWorld(1.0f);
    fixedWorld = glm::translate(fixedWorld, glm::vec3(2.3f, 1.5f, 0.0f));
    fixedWorld = glm::rotate(fixedWorld, glm::radians(145.0f),
                             glm::vec3(0.0f, 1.0f, 0.0f));
    fixedWorld = glm::scale(fixedWorld, glm::vec3(0.02f, -0.02f, 0.02f));
    fixed.worldFromText = encodeWorld(fixedWorld);
    if (!enqueueSample(fixed).has_value()) {
      return false;
    }

    const float uiTextScale = 0.0085f;
    const float uiAnchorX = -3.2f;
    float uiCursorY = 0.65f;
    auto enqueueUiStress3D = [&](std::string_view text, float pxScale,
                                 const nuri::TextColor &color, float gapAfterPx,
                                 bool wrap, float wrapWidthPx, bool multiline) {
      nuri::Text3DDesc sample{};
      sample.utf8 = text;
      sample.style.font = defaultFont;
      sample.style.pxSize = baseFontSizePx * pxScale;
      sample.layout.alignH = nuri::TextAlignH::Left;
      sample.layout.alignV = nuri::TextAlignV::Top;
      if (wrap) {
        sample.layout.wrapMode = nuri::TextWrapMode::Word;
        sample.layout.maxWidthPx = wrapWidthPx;
      }
      if (multiline) {
        sample.layout.wrapMode = nuri::TextWrapMode::None;
      }
      configureAsciiLayout(sample.layout);
      sample.fillColor = color;
      sample.billboard = nuri::TextBillboardMode::Spherical;
      glm::mat4 world(1.0f);
      world = glm::translate(world, glm::vec3(uiAnchorX, uiCursorY, 0.0f));
      world = glm::scale(world, glm::vec3(uiTextScale));
      sample.worldFromText = encodeWorld(world);
      const std::optional<nuri::TextBounds> bounds = enqueueSample(sample);
      if (!bounds.has_value()) {
        return false;
      }
      const float blockHeightPx = std::max(bounds->maxY - bounds->minY, 0.0f);
      uiCursorY -= (blockHeightPx + gapAfterPx) * uiTextScale;
      return true;
    };

    if (!enqueueUiStress3D("MTSDF 2D Raster Test 0123456789 AaBbCc", 1.0f,
                           {.r = 1.0f, .g = 1.0f, .b = 1.0f, .a = 1.0f}, 14.0f,
                           false, 0.0f, false)) {
      return false;
    }
    if (!enqueueUiStress3D("Kerning: AV AVATAR To WA TA YA LT", 0.67f,
                           {.r = 0.86f, .g = 0.92f, .b = 1.0f, .a = 1.0f},
                           14.0f, false, 0.0f, false)) {
      return false;
    }
    if (!enqueueUiStress3D(
            "Wrap(320px): The quick brown fox jumps over the lazy dog near the "
            "river bank.",
            0.62f, {.r = 0.9f, .g = 1.0f, .b = 0.9f, .a = 1.0f}, 18.0f, true,
            320.0f, false)) {
      return false;
    }
    if (!enqueueUiStress3D("Manual newline:\nLine 1\nLine 2\nLine 3", 0.57f,
                           {.r = 1.0f, .g = 0.95f, .b = 0.85f, .a = 1.0f}, 0.0f,
                           false, 0.0f, true)) {
      return false;
    }
    return true;
  }

  void initializeTextOverlayFeature() {
    if (textOverlayEnabled_ || textSystem_ == nullptr ||
        editorOverlay_ != nullptr) {
      return;
    }
    textOverlayEnabled_ = true;
  }

  void removeTextOverlayFeature() {
    if (!textOverlayEnabled_) {
      return;
    }
    textOverlayEnabled_ = false;
  }

  void initializeEditorOverlay() {
    if (editorOverlay_ != nullptr) {
      return;
    }
    removeTextOverlayFeature();
    const nuri::EditorServices editorServices{
        .scene = &scene_,
        .cameraSystem = &cameraSystem_,
        .gpu = &getGPU(),
        .resources = &getRenderer().resources(),
        .renderPipeline = &getRenderPipeline(),
        .textSystem = textSystem_.get(),
        .bakery = bakerySystem_.get(),
        .renderGraphTelemetry = &getRenderer().renderGraphTelemetry(),
        .selectionState = &sceneEditorSelectionState_,
    };
    auto editorOverlay = nuri::EditorOverlayController::create(
        getWindow(), getGPU(), {}, editorServices);
    NURI_ASSERT(editorOverlay != nullptr,
                "Failed to create editor overlay controller");
    editorOverlay_ = std::move(editorOverlay);
    NURI_ASSERT(editorRenderFeature_ != nullptr,
                "Editor overlay feature is not initialized");
    editorRenderFeature_->setController(editorOverlay_.get());
    if (nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_)) {
      editorOverlay_->syncCameraControllerWidgetStateFromCamera(*camera);
    }
  }

  void removeEditorOverlay() {
    if (editorOverlay_ == nullptr) {
      return;
    }
    if (editorRenderFeature_ != nullptr) {
      editorRenderFeature_->setController(nullptr);
    }
    editorOverlay_.reset();
    initializeTextOverlayFeature();
  }

  void toggleEditorOverlay() {
    if (editorOverlay_ != nullptr) {
      removeEditorOverlay();
      return;
    }
    initializeEditorOverlay();
  }

  void
  releaseImportedPrefabSceneResources(ImportedPrefabSceneResources &scene) {
    nuri::ResourceManager &resources = getRenderer().resources();
    for (nuri::ModelRef model : scene.assets.models) {
      if (nuri::isValid(model)) {
        resources.release(model);
      }
    }
    for (nuri::MaterialRef material : scene.assets.materials) {
      if (nuri::isValid(material)) {
        resources.release(material);
      }
    }
    scene.assets = nuri::ScenePrefabAssets{};
    scene.prefab = nuri::ScenePrefab{};
    scene.fallbackLights.clear();
    scene.ready = false;
  }

  void
  loadImportedPrefabSceneResources(std::string_view sceneName,
                                   std::string_view modelPath,
                                   const nuri::MeshImportOptions &importOptions,
                                   ImportedPrefabSceneResources &out) {
    releaseImportedPrefabSceneResources(out);

    auto sceneResult = nuri::SceneImporter::loadSceneFromFile(
        modelPath, nuri::SceneImportOptions{
                       .assetBuildOptions = importOptions,
                   });
    if (sceneResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "%s imported scene from '%s': %s",
                       std::string(sceneName).c_str(),
                       std::string(modelPath).c_str(),
                       sceneResult.error().c_str());
      return;
    }

    nuri::ImportedScene importedScene = std::move(sceneResult.value());
    out.fallbackLights.assign(importedScene.lights.begin(),
                              importedScene.lights.end());
    auto prefabResult = nuri::SceneImporter::buildScenePrefab(importedScene);
    if (prefabResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to build "
                       "%s scene prefab from '%s': %s",
                       std::string(sceneName).c_str(),
                       std::string(modelPath).c_str(),
                       prefabResult.error().c_str());
      return;
    }

    nuri::ResourceManager &resources = getRenderer().resources();
    auto assetsResult =
        resources.acquireScenePrefabAssets(prefabResult.value());
    if (assetsResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to acquire "
                       "%s prefab assets from '%s': %s",
                       std::string(sceneName).c_str(),
                       std::string(modelPath).c_str(),
                       assetsResult.error().c_str());
      return;
    }

    out.prefab = std::move(prefabResult.value());
    out.assets = std::move(assetsResult.value());
    out.ready = true;
    NURI_LOG_INFO("NuriApplication::loadSceneResources: %s prefab loaded "
                  "(nodes=%zu renderables=%zu lights=%zu meshes=%zu)",
                  std::string(sceneName).c_str(), out.prefab.nodes.size(),
                  out.prefab.renderables.size(), out.prefab.lights.size(),
                  out.prefab.meshAssets.size());

    if (bakerySystem_) {
      auto enqueueResult = bakerySystem_->enqueue(nuri::bakery::BakeRequest{
          nuri::bakery::ScenePortableAssetsBakeRequest{
              .scenePath = std::filesystem::path(std::string(modelPath)),
              .prebuildNativeTargets = {},
              .forceRebuild = false,
          }});
      if (enqueueResult.hasError()) {
        NURI_LOG_WARNING(
            "NuriApplication::loadSceneResources: failed to queue scene "
            "portable assets bake for '%s': %s",
            std::string(modelPath).c_str(), enqueueResult.error().c_str());
      } else {
        NURI_LOG_INFO(
            "NuriApplication::loadSceneResources: queued scene portable "
            "assets bake for '%s' as job #%llu",
            std::string(modelPath).c_str(),
            static_cast<unsigned long long>(enqueueResult.value().value));
      }
    }
  }

  [[nodiscard]] nuri::RenderableId instantiateImportedPrefabScene(
      std::string_view sceneName, const ImportedPrefabSceneResources &resources,
      const glm::mat4 &baseModel,
      nuri::SceneInstantiationMap *outInstantiation = nullptr,
      nuri::NodeId *outRootNode = nullptr) {
    if (!resources.ready) {
      return nuri::kInvalidRenderableId;
    }
    if (outInstantiation != nullptr) {
      outInstantiation->nodes.clear();
      outInstantiation->renderables.clear();
      outInstantiation->lights.clear();
    }
    if (outRootNode != nullptr) {
      *outRootNode = nuri::kInvalidNodeId;
    }

    auto rootNodeResult = scene_.graph().createNode(scene_.graph().rootNode(),
                                                    sceneName, baseModel);
    if (rootNodeResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::instantiateImportedPrefabScene: "
                       "failed to create '%s' root node: %s",
                       std::string(sceneName).c_str(),
                       rootNodeResult.error().c_str());
      return nuri::kInvalidRenderableId;
    }

    nuri::SceneInstantiationMap localInstantiated;
    nuri::SceneInstantiationMap &instantiated =
        outInstantiation != nullptr ? *outInstantiation : localInstantiated;
    auto instantiateResult = scene_.graph().instantiatePrefab(
        resources.prefab, rootNodeResult.value(), resources.assets,
        &instantiated);
    if (instantiateResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::instantiateImportedPrefabScene: "
                       "failed to instantiate '%s': %s",
                       std::string(sceneName).c_str(),
                       instantiateResult.error().c_str());
      (void)scene_.graph().destroyNodeSubtree(rootNodeResult.value());
      return nuri::kInvalidRenderableId;
    }

    if (outRootNode != nullptr) {
      *outRootNode = rootNodeResult.value();
    }
    sceneHasAuthoredLights_ = !instantiated.lights.empty();
    for (nuri::RenderableId renderableId : instantiated.renderables) {
      if (nuri::isValid(renderableId)) {
        return renderableId;
      }
    }
    return nuri::kInvalidRenderableId;
  }

  void loadSceneResources() {
    nuri::ResourceManager &resources = getRenderer().resources();
    scene_.graph().clear();
    scene_.setEnvironment(nuri::EnvironmentHandles{});
    releaseOwnedResourceHandles();

    const std::string duckModelPath =
        (config_.roots.models / kSampleDuckModelRelativePath).string();
    const std::string duckAlbedoPath =
        (config_.roots.models / kSampleDuckAlbedoRelativePath).string();
    const std::string helmetModelPath =
        (config_.roots.models / kDamagedHelmetModelRelativePath).string();
    const std::string lightsPunctualLampModelPath =
        (config_.roots.models / kLightsPunctualLampModelRelativePath).string();
    const std::string clearcoatModelPath =
        (config_.roots.models / kClearcoatWickerModelRelativePath).string();
    const std::string sheenChairModelPath =
        (config_.roots.models / kSheenChairModelRelativePath).string();
    const std::string specularSilkPoufModelPath =
        (config_.roots.models / kSpecularSilkPoufModelRelativePath).string();
    const std::string dragonAttenuationModelPath =
        (config_.roots.models / kDragonAttenuationModelRelativePath).string();
    const std::string dragonIorModelPath =
        (config_.roots.models / kDragonIorModelRelativePath).string();
    const std::string emissiveStrengthTestModelPath =
        (config_.roots.models / kEmissiveStrengthTestModelRelativePath)
            .string();
    const std::string orreyModelPath =
        (config_.roots.models / kOrreyModelRelativePath).string();
    const std::string foxModelPath =
        (config_.roots.models / kFoxModelRelativePath).string();
    const std::string medievalFantasyBookModelPath =
        (config_.roots.models / kMedievalFantasyBookModelRelativePath).string();
    const std::string environmentHdrPath =
        (config_.roots.textures / kSampleEnvironmentHdrRelativePath).string();

    auto duckModelResult = resources.acquireModel(
        nuri::ModelRequest{.path = duckModelPath, .debugName = "rubber_duck"});
    NURI_ASSERT(!duckModelResult.hasError(), "Failed to create model: %s",
                duckModelResult.error().c_str());
    duckModel_ = duckModelResult.value();

    auto duckAlbedoRefResult = resources.acquireTexture(nuri::TextureRequest{
        .path = duckAlbedoPath,
        .loadOptions =
            nuri::TextureLoadOptions{.srgb = true, .generateMipmaps = true},
        .kind = nuri::TextureRequestKind::Texture2D,
        .debugName = "duck_albedo",
    });
    NURI_ASSERT(!duckAlbedoRefResult.hasError(),
                "Failed to load albedo texture: %s",
                duckAlbedoRefResult.error().c_str());

    nuri::MeshImportOptions helmetImportOptions{};
    helmetImportOptions.flipUVs = true;
    nuri::MeshImportOptions dragonImportOptions = helmetImportOptions;
    dragonImportOptions.flipUVs = false;
    auto helmetModelResult = resources.acquireModel(nuri::ModelRequest{
        .path = helmetModelPath,
        .importOptions = helmetImportOptions,
        .debugName = "damaged_helmet",
    });
    if (helmetModelResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "DamagedHelmet model '%s': %s",
                       helmetModelPath.c_str(),
                       helmetModelResult.error().c_str());
    }
    NURI_ASSERT(!helmetModelResult.hasError(),
                "Failed to create DamagedHelmet model: %s",
                helmetModelResult.error().c_str());

    auto clearcoatModelResult = resources.acquireModel(nuri::ModelRequest{
        .path = clearcoatModelPath,
        .importOptions = helmetImportOptions,
        .debugName = "clearcoat_wicker",
    });
    if (clearcoatModelResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "ClearcoatWicker model '%s': %s",
                       clearcoatModelPath.c_str(),
                       clearcoatModelResult.error().c_str());
    }
    NURI_ASSERT(!clearcoatModelResult.hasError(),
                "Failed to create ClearcoatWicker model: %s",
                clearcoatModelResult.error().c_str());

    auto lightsPunctualLampModelResult = resources.acquireModel(
        nuri::ModelRequest{.path = lightsPunctualLampModelPath,
                           .importOptions = helmetImportOptions,
                           .debugName = "lights_punctual_lamp"});
    if (lightsPunctualLampModelResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "LightsPunctualLamp model '%s': %s",
                       lightsPunctualLampModelPath.c_str(),
                       lightsPunctualLampModelResult.error().c_str());
    }
    NURI_ASSERT(!lightsPunctualLampModelResult.hasError(),
                "Failed to create LightsPunctualLamp model: %s",
                lightsPunctualLampModelResult.error().c_str());

    auto sheenChairModelResult = resources.acquireModel(nuri::ModelRequest{
        .path = sheenChairModelPath,
        .importOptions = helmetImportOptions,
        .debugName = "sheen_chair",
    });
    if (sheenChairModelResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "SheenChair model '%s': %s",
                       sheenChairModelPath.c_str(),
                       sheenChairModelResult.error().c_str());
    }
    NURI_ASSERT(!sheenChairModelResult.hasError(),
                "Failed to create SheenChair model: %s",
                sheenChairModelResult.error().c_str());

    auto specularSilkPoufModelResult =
        resources.acquireModel(nuri::ModelRequest{
            .path = specularSilkPoufModelPath,
            .importOptions = helmetImportOptions,
            .debugName = "specular_silk_pouf",
        });
    if (specularSilkPoufModelResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "SpecularSilkPouf model '%s': %s",
                       specularSilkPoufModelPath.c_str(),
                       specularSilkPoufModelResult.error().c_str());
    }
    NURI_ASSERT(!specularSilkPoufModelResult.hasError(),
                "Failed to create SpecularSilkPouf model: %s",
                specularSilkPoufModelResult.error().c_str());

    auto dragonAttenuationModelResult =
        resources.acquireModel(nuri::ModelRequest{
            .path = dragonAttenuationModelPath,
            .importOptions = dragonImportOptions,
            .debugName = "dragon_attenuation",
        });
    if (dragonAttenuationModelResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "DragonAttenuation model '%s': %s",
                       dragonAttenuationModelPath.c_str(),
                       dragonAttenuationModelResult.error().c_str());
    }
    NURI_ASSERT(!dragonAttenuationModelResult.hasError(),
                "Failed to create DragonAttenuation model: %s",
                dragonAttenuationModelResult.error().c_str());

    auto dragonIorModelResult = resources.acquireModel(nuri::ModelRequest{
        .path = dragonIorModelPath,
        .importOptions = dragonImportOptions,
        .debugName = "dragon_ior",
    });
    if (dragonIorModelResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "DragonIor model '%s': %s",
                       dragonIorModelPath.c_str(),
                       dragonIorModelResult.error().c_str());
    }
    NURI_ASSERT(!dragonIorModelResult.hasError(),
                "Failed to create DragonIor model: %s",
                dragonIorModelResult.error().c_str());

    auto emissiveStrengthTestModelResult =
        resources.acquireModel(nuri::ModelRequest{
            .path = emissiveStrengthTestModelPath,
            .importOptions = helmetImportOptions,
            .debugName = "emissive_strength_test",
        });
    if (emissiveStrengthTestModelResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "EmissiveStrengthTest model '%s': %s",
                       emissiveStrengthTestModelPath.c_str(),
                       emissiveStrengthTestModelResult.error().c_str());
    }
    NURI_ASSERT(!emissiveStrengthTestModelResult.hasError(),
                "Failed to create EmissiveStrengthTest model: %s",
                emissiveStrengthTestModelResult.error().c_str());

    auto orreyModelResult = resources.acquireModel(nuri::ModelRequest{
        .path = orreyModelPath,
        .importOptions = helmetImportOptions,
        .debugName = "orrey",
    });
    if (orreyModelResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "Orrey model '%s': %s",
                       orreyModelPath.c_str(),
                       orreyModelResult.error().c_str());
    }
    NURI_ASSERT(!orreyModelResult.hasError(),
                "Failed to create Orrey model: %s",
                orreyModelResult.error().c_str());

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
      NURI_LOG_INFO("NuriApplication::loadSceneResources: loaded IBL cubemap "
                    "'%s'",
                    resolvedPath.string().c_str());
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
      NURI_LOG_INFO("NuriApplication::loadSceneResources: loaded BRDF LUT '%s'",
                    resolvedPath.string().c_str());
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

    helmetModel_ = helmetModelResult.value();
    lightsPunctualLampModel_ = lightsPunctualLampModelResult.value();
    clearcoatModel_ = clearcoatModelResult.value();
    sheenChairModel_ = sheenChairModelResult.value();
    specularSilkPoufModel_ = specularSilkPoufModelResult.value();
    dragonAttenuationModel_ = dragonAttenuationModelResult.value();
    dragonIorModel_ = dragonIorModelResult.value();
    emissiveStrengthTestModel_ = emissiveStrengthTestModelResult.value();
    orreyModel_ = orreyModelResult.value();

    {
      const nuri::TextureRecord *duckAlbedoRecord =
          resources.tryGet(duckAlbedoRefResult.value());
      NURI_ASSERT(duckAlbedoRecord != nullptr,
                  "Duck albedo record lookup failed");
      nuri::MaterialDesc duckMaterialDesc{};
      duckMaterialDesc.textures.baseColor = duckAlbedoRecord->texture;
      auto addDuckMaterialResult =
          resources.acquireMaterial(nuri::MaterialRequest{
              .desc = duckMaterialDesc,
              .textureRefs =
                  nuri::MaterialRequest::TextureRefs{
                      .baseColor = duckAlbedoRefResult.value(),
                  },
              .debugName = "duck_material"});
      NURI_ASSERT(!addDuckMaterialResult.hasError(),
                  "Failed to acquire duck material: %s",
                  addDuckMaterialResult.error().c_str());
      duckMaterialIndex_ = addDuckMaterialResult.value();
      resources.setModelMaterialForAllSources(duckModel_, duckMaterialIndex_);
    }
    resources.release(duckAlbedoRefResult.value());

    {
      nuri::MaterialDesc bistroMaterialDesc{};
      auto addBistroMaterialResult =
          resources.acquireMaterial(nuri::MaterialRequest{
              .desc = bistroMaterialDesc,
              .debugName = "niagara_bistro_fallback_material"});
      NURI_ASSERT(!addBistroMaterialResult.hasError(),
                  "Failed to acquire Niagara Bistro fallback material: %s",
                  addBistroMaterialResult.error().c_str());
      bistroMaterialIndex_ = addBistroMaterialResult.value();
    }

    auto helmetLoadResult =
        resources.acquireMaterialsFromModel(nuri::ImportedMaterialRequest{
            .modelPath = helmetModelPath,
            .model = helmetModel_,
            .debugNamePrefix = "damaged_helmet",
        });
    if (helmetLoadResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to import "
                       "DamagedHelmet materials from '%s': %s",
                       helmetModelPath.c_str(),
                       helmetLoadResult.error().c_str());
    }

    helmetFallbackMaterialIndex_ = helmetLoadResult.hasError()
                                       ? nuri::kInvalidMaterialRef
                                       : helmetLoadResult.value().firstMaterial;
    if (!nuri::isValid(helmetFallbackMaterialIndex_)) {
      helmetFallbackMaterialIndex_ = bistroMaterialIndex_;
    }

    clearcoatMaterial_ = acquireImportedMaterialOrFallback(
        resources, "ClearcoatWicker", clearcoatModelPath, clearcoatModel_,
        "clearcoat_wicker", "clearcoat_wicker_fallback_material");

    lightsPunctualLampMaterial_ = acquireImportedMaterialOrFallback(
        resources, "LightsPunctualLamp", lightsPunctualLampModelPath,
        lightsPunctualLampModel_, "lights_punctual_lamp",
        "lights_punctual_lamp_fallback_material");

    sheenChairMaterial_ = acquireImportedMaterialOrFallback(
        resources, "SheenChair", sheenChairModelPath, sheenChairModel_,
        "sheen_chair", "sheen_chair_fallback_material");

    specularSilkPoufMaterial_ = acquireImportedMaterialOrFallback(
        resources, "SpecularSilkPouf", specularSilkPoufModelPath,
        specularSilkPoufModel_, "specular_silk_pouf",
        "specular_silk_pouf_fallback_material");

    dragonAttenuationMaterial_ = acquireImportedMaterialOrFallback(
        resources, "DragonAttenuation", dragonAttenuationModelPath,
        dragonAttenuationModel_, "dragon_attenuation",
        "dragon_attenuation_fallback_material");

    dragonIorMaterial_ = acquireImportedMaterialOrFallback(
        resources, "DragonIor", dragonIorModelPath, dragonIorModel_,
        "dragon_ior", "dragon_ior_fallback_material");

    emissiveStrengthTestMaterial_ = acquireImportedMaterialOrFallback(
        resources, "EmissiveStrengthTest", emissiveStrengthTestModelPath,
        emissiveStrengthTestModel_, "emissive_strength_test",
        "emissive_strength_test_fallback_material");

    orreyMaterial_ = acquireImportedMaterialOrFallback(
        resources, "Orrey", orreyModelPath, orreyModel_, "orrey",
        "orrey_fallback_material");

    loadImportedLightsForScene("Rubber Duck", duckModelPath,
                               duckImportedLights_);
    loadImportedPrefabSceneResources("DamagedHelmet", helmetModelPath,
                                     helmetImportOptions, helmetPrefabScene_);
    loadImportedPrefabSceneResources(
        "LightsPunctualLamp", lightsPunctualLampModelPath, helmetImportOptions,
        lightsPunctualLampPrefabScene_);
    loadImportedPrefabSceneResources("ClearcoatWicker", clearcoatModelPath,
                                     helmetImportOptions,
                                     clearcoatPrefabScene_);
    loadImportedPrefabSceneResources("SheenChair", sheenChairModelPath,
                                     helmetImportOptions,
                                     sheenChairPrefabScene_);
    loadImportedPrefabSceneResources(
        "SpecularSilkPouf", specularSilkPoufModelPath, helmetImportOptions,
        specularSilkPoufPrefabScene_);
    loadImportedPrefabSceneResources(
        "DragonAttenuation", dragonAttenuationModelPath, dragonImportOptions,
        dragonAttenuationPrefabScene_);
    loadImportedPrefabSceneResources("DragonIor", dragonIorModelPath,
                                     dragonImportOptions,
                                     dragonIorPrefabScene_);
    loadImportedPrefabSceneResources(
        "EmissiveStrengthTest", emissiveStrengthTestModelPath,
        helmetImportOptions, emissiveStrengthTestPrefabScene_);
    loadImportedPrefabSceneResources("Orrey", orreyModelPath,
                                     helmetImportOptions, orreyPrefabScene_);
    loadImportedPrefabSceneResources("Fox", foxModelPath, helmetImportOptions,
                                     foxPrefabScene_);
    loadImportedPrefabSceneResources(
        "MedievalFantasyBook", medievalFantasyBookModelPath,
        helmetImportOptions, medievalFantasyBookPrefabScene_);

    logLoadedModelSummary(resources, helmetModel_, "DamagedHelmet");
    logLoadedModelSummary(resources, lightsPunctualLampModel_,
                          "LightsPunctualLamp");
    logLoadedModelSummary(resources, clearcoatModel_, "ClearcoatWicker");
    logLoadedModelSummary(resources, sheenChairModel_, "SheenChair");
    NURI_LOG_INFO("NuriApplication::loadSceneResources: SheenChair uses "
                  "default glTF material assignments; "
                  "KHR_materials_variants is not supported yet");
    logLoadedModelSummary(resources, specularSilkPoufModel_,
                          "SpecularSilkPouf");
    logLoadedModelSummary(resources, dragonAttenuationModel_,
                          "DragonAttenuation");
    logLoadedModelSummary(resources, dragonIorModel_, "DragonIor");
    logLoadedModelSummary(resources, emissiveStrengthTestModel_,
                          "EmissiveStrengthTest");
    logLoadedModelSummary(resources, orreyModel_, "Orrey");

    pendingScenePreset_ = scenePreset_;
  }

  void loadSingleDuckSceneResources() {
    renderSettings_.opaque.enableInstanceCompute = true;
    renderSettings_.opaque.enableMeshLod = true;
    renderSettings_.opaque.forcedMeshLod = -1;
    renderSettings_.opaque.meshLodDistanceThresholds =
        glm::vec3(8.0f, 16.0f, 32.0f);
    renderSettings_.opaque.enableInstanceAnimation = false;

    duckRenderableId_ =
        addRequiredRenderable(duckModel_, duckMaterialIndex_, duckBaseModel_,
                              "Failed to add duck renderable");

    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    NURI_ASSERT(camera != nullptr, "Failed to get main camera");
    camera->setLookAt(glm::vec3(0.0f, 1.0f, -1.5f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f));
    syncEditorCameraWidgetState(*camera);
  }

  [[nodiscard]] nuri::MaterialRef acquireImportedMaterialOrFallback(
      nuri::ResourceManager &resources, std::string_view sceneName,
      std::string_view modelPath, nuri::ModelRef modelRef,
      std::string_view debugNamePrefix, std::string_view fallbackDebugName) {
    auto loadResult =
        resources.acquireMaterialsFromModel(nuri::ImportedMaterialRequest{
            .modelPath = std::string(modelPath),
            .model = modelRef,
            .debugNamePrefix = std::string(debugNamePrefix),
        });
    if (!loadResult.hasError() &&
        nuri::isValid(loadResult.value().firstMaterial)) {
      return loadResult.value().firstMaterial;
    }

    if (loadResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to import "
                       "%s materials from '%s': %s",
                       std::string(sceneName).c_str(),
                       std::string(modelPath).c_str(),
                       loadResult.error().c_str());
    }

    auto fallbackMaterialResult =
        resources.acquireMaterial(nuri::MaterialRequest{
            .desc = nuri::MaterialDesc{},
            .debugName = std::string(fallbackDebugName),
        });
    NURI_ASSERT(!fallbackMaterialResult.hasError(),
                "Failed to acquire %s fallback material: %s",
                std::string(sceneName).c_str(),
                fallbackMaterialResult.error().c_str());
    const nuri::MaterialRef fallbackMaterial = fallbackMaterialResult.value();
    resources.setModelMaterialForAllSources(modelRef, fallbackMaterial);
    return fallbackMaterial;
  }

  void logLoadedModelSummary(nuri::ResourceManager &resources,
                             nuri::ModelRef modelRef,
                             std::string_view sceneName) {
    const nuri::ModelRecord *record = resources.tryGet(modelRef);
    NURI_ASSERT(record != nullptr && record->model != nullptr,
                "%s model record lookup failed",
                std::string(sceneName).c_str());
    NURI_LOG_INFO("NuriApplication::loadSceneResources: %s loaded "
                  "(submeshes=%zu vertices=%u indices=%u)",
                  std::string(sceneName).c_str(),
                  record->model->submeshes().size(),
                  record->model->vertexCount(), record->model->indexCount());
  }

  void
  loadImportedLightsForScene(std::string_view sceneName,
                             std::string_view modelPath,
                             std::vector<nuri::ImportedSceneLight> &outLights) {
    auto loadResult = nuri::SceneImporter::loadSceneFromFile(modelPath);
    if (loadResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to import "
                       "%s punctual lights from '%s': %s",
                       std::string(sceneName).c_str(),
                       std::string(modelPath).c_str(),
                       loadResult.error().c_str());
      outLights.clear();
      return;
    }

    outLights.assign(loadResult.value().lights.begin(),
                     loadResult.value().lights.end());
    if (!outLights.empty()) {
      NURI_LOG_INFO("NuriApplication::loadSceneResources: imported %zu glTF "
                    "punctual light(s) for %s",
                    outLights.size(), std::string(sceneName).c_str());
    }
  }

  [[nodiscard]] bool
  applyImportedLights(std::span<const nuri::ImportedSceneLight> importedLights,
                      const glm::mat4 &modelMatrix) {
    if (importedLights.empty()) {
      return false;
    }

    auto rootNodeResult =
        scene_.graph().createNode(scene_.graph().rootNode(), {}, modelMatrix);
    if (rootNodeResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::applyImportedLights: failed to create "
                       "scene light root: %s",
                       rootNodeResult.error().c_str());
      return false;
    }

    bool addedAny = false;
    for (const nuri::ImportedSceneLight &importedLight : importedLights) {
      auto lightNodeResult = scene_.graph().createNode(
          rootNodeResult.value(), importedLight.light.name,
          makeTransformMatrix(importedLight.light.position,
                              importedLight.light.rotation));
      if (lightNodeResult.hasError()) {
        NURI_LOG_WARNING("NuriApplication::applyImportedLights: failed to add "
                         "imported light node '%s': %s",
                         importedLight.light.name.c_str(),
                         lightNodeResult.error().c_str());
        continue;
      }

      nuri::LightDesc localLight = importedLight.light;
      localLight.position = glm::vec3(0.0f);
      localLight.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
      auto addResult =
          scene_.graph().addLight(lightNodeResult.value(), localLight);
      if (addResult.hasError()) {
        NURI_LOG_WARNING("NuriApplication::applyImportedLights: failed to add "
                         "imported light '%s': %s",
                         importedLight.light.name.c_str(),
                         addResult.error().c_str());
        continue;
      }
      addedAny = true;
    }
    return addedAny;
  }

  [[nodiscard]] bool applyImportedLightsForCurrentPreset() {
    switch (scenePreset_) {
    case ScenePreset::SingleDuck:
      return applyImportedLights(duckImportedLights_, duckBaseModel_);
    case ScenePreset::DamagedHelmet:
      return applyImportedLights(helmetPrefabScene_.fallbackLights,
                                 helmetBaseModel_);
    case ScenePreset::NiagaraBistro:
      return applyImportedLights(niagaraBistroPrefabScene_.fallbackLights,
                                 niagaraBistroBaseModel_);
    case ScenePreset::LightsPunctualLamp:
      return applyImportedLights(lightsPunctualLampPrefabScene_.fallbackLights,
                                 lightsPunctualLampBaseModel_);
    case ScenePreset::ClearcoatWicker:
      return applyImportedLights(clearcoatPrefabScene_.fallbackLights,
                                 clearcoatBaseModel_);
    case ScenePreset::SheenChair:
      return applyImportedLights(sheenChairPrefabScene_.fallbackLights,
                                 glm::mat4(1.0f));
    case ScenePreset::SpecularSilkPouf:
      return applyImportedLights(specularSilkPoufPrefabScene_.fallbackLights,
                                 glm::mat4(1.0f));
    case ScenePreset::DragonAttenuation:
      return applyImportedLights(dragonAttenuationPrefabScene_.fallbackLights,
                                 glm::mat4(1.0f));
    case ScenePreset::DragonIor:
      return applyImportedLights(dragonIorPrefabScene_.fallbackLights,
                                 glm::mat4(1.0f));
    case ScenePreset::EmissiveStrengthTest:
      return applyImportedLights(
          emissiveStrengthTestPrefabScene_.fallbackLights, glm::mat4(1.0f));
    case ScenePreset::Orrey:
      return applyImportedLights(orreyPrefabScene_.fallbackLights,
                                 orreyBaseModel_);
    case ScenePreset::Fox:
      return applyImportedLights(foxPrefabScene_.fallbackLights, foxBaseModel_);
    case ScenePreset::MedievalFantasyBook:
      return applyImportedLights(medievalFantasyBookPrefabScene_.fallbackLights,
                                 glm::mat4(1.0f));
    case ScenePreset::InstancedDuck32K:
    case ScenePreset::Text3DTest:
      return false;
    }
    return false;
  }

  void setupInstancedDuckScene32k() {
    // Benchmark defaults: keep auto LOD enabled and push far instances to
    // lower LODs earlier to reduce GPU pressure.
    renderSettings_.opaque.enableInstanceCompute = true;
    renderSettings_.opaque.enableMeshLod = true;
    renderSettings_.opaque.forcedMeshLod = -1;
    renderSettings_.opaque.meshLodDistanceThresholds =
        glm::vec3(4.0f, 8.0f, 16.0f);
    renderSettings_.opaque.enableInstanceAnimation = true;

    std::vector<glm::mat4> transforms;
    transforms.reserve(kDuckInstanceCount);
    for (uint32_t i = 0; i < kDuckInstanceCount; ++i) {
      const glm::vec3 position = instancePositionFromGrid(i);
      transforms.push_back(glm::translate(glm::mat4(1.0f), position) *
                           duckBaseModel_);
    }

    auto addResult = scene_.graph().addRenderablesInstanced(
        duckModel_, duckMaterialIndex_, transforms);
    NURI_ASSERT(!addResult.hasError(),
                "Failed to add instanced duck renderables: %s",
                addResult.error().c_str());

    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    NURI_ASSERT(camera != nullptr, "Failed to get main camera");
    camera->setLookAt(glm::vec3(0.0f, 120.0f, -760.0f), glm::vec3(0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f));
    syncEditorCameraWidgetState(*camera);

    NURI_LOG_INFO(
        "NuriApplication::setupInstancedDuckScene32k: spawned %u ducks in a "
        "32x32x32 grid",
        kDuckInstanceCount);
  }

  std::filesystem::path resolveNiagaraBistroPath() const {
    std::filesystem::path preferredPath =
        config_.roots.models / kNiagaraBistroModelRelativePath;
    std::error_code ec;
    if (std::filesystem::exists(preferredPath, ec) &&
        std::filesystem::is_regular_file(preferredPath, ec)) {
      return preferredPath;
    }

    std::filesystem::path fallbackPath{kNiagaraBistroAbsolutePath};
    ec.clear();
    if (std::filesystem::exists(fallbackPath, ec) &&
        std::filesystem::is_regular_file(fallbackPath, ec)) {
      return fallbackPath;
    }
    return preferredPath;
  }

  void beginNiagaraBistroModelLoad() {
    const bool hasAsyncLoadInFlight = bistroAsyncLoad_ &&
                                      bistroAsyncLoad_->valid() &&
                                      !bistroAsyncLoad_->isFinalized();
    if (nuri::isValid(bistroModel_) || hasAsyncLoadInFlight ||
        bistroLoadFailed_) {
      return;
    }
    bistroLoadFailed_ = false;
    bistroLoadError_.clear();
    bistroLoadStartTimeSeconds_ = getTime();
    bistroLastProgressLogTimeSeconds_ = bistroLoadStartTimeSeconds_;

    const std::string path = resolveNiagaraBistroPath().string();

    auto asyncLoadResult =
        nuri::Model::createFromFileAsync(path, nuri::MeshImportOptions{});
    if (asyncLoadResult.hasError()) {
      bistroLoadFailed_ = true;
      bistroLoadError_ = asyncLoadResult.error();
      NURI_LOG_WARNING(
          "NuriApplication: Niagara Bistro async load start failed: %s",
          bistroLoadError_.c_str());
      return;
    }

    bistroAsyncLoad_ = std::move(asyncLoadResult.value());
    NURI_LOG_INFO(
        "NuriApplication: started Niagara Bistro async model load from '%s'",
        path.c_str());
  }

  void updateNiagaraBistroSceneStreaming() {
    nuri::ResourceManager &resources = getRenderer().resources();
    if (!bistroAsyncLoad_ || !bistroAsyncLoad_->valid()) {
      if (scenePreset_ == ScenePreset::NiagaraBistro &&
          nuri::isValid(bistroModel_) && !nuri::isValid(bistroRenderableId_)) {
        setupNiagaraBistroScene();
      }
      return;
    }

    if (!bistroAsyncLoad_->isReady()) {
      const double now = getTime();
      if (now - bistroLastProgressLogTimeSeconds_ >= 5.0) {
        bistroLastProgressLogTimeSeconds_ = now;
        NURI_LOG_INFO(
            "NuriApplication: Niagara Bistro load in progress (%.1f s)",
            now - bistroLoadStartTimeSeconds_);
      }
      return;
    }

    auto warmupResult = bistroAsyncLoad_->resolveWarmup();
    const std::optional<bool> warmupCacheHit =
        warmupResult.hasError() ? std::nullopt
                                : std::optional<bool>(warmupResult.value());
    const std::string warmupError =
        std::string(bistroAsyncLoad_->warmupError());
    const double totalLoadSeconds = getTime() - bistroLoadStartTimeSeconds_;

    auto modelResult = resources.acquireModel(nuri::ModelRequest{
        .path = resolveNiagaraBistroPath().string(),
        .debugName = "niagara_bistro",
    });
    if (modelResult.hasError()) {
      bistroLoadFailed_ = true;
      bistroLoadError_ = modelResult.error();
      if (!warmupError.empty()) {
        NURI_LOG_WARNING(
            "NuriApplication: Niagara Bistro async warmup failed: %s",
            warmupError.c_str());
      }
      NURI_LOG_WARNING(
          "NuriApplication: Niagara Bistro GPU model creation failed: %s",
          bistroLoadError_.c_str());
      bistroAsyncLoad_.reset();
      return;
    }

    if (warmupCacheHit.has_value()) {
      NURI_LOG_INFO("NuriApplication: Niagara Bistro cache %s in %.1f s",
                    warmupCacheHit.value() ? "hit" : "rebuilt",
                    totalLoadSeconds);
    } else if (!warmupError.empty()) {
      NURI_LOG_WARNING(
          "NuriApplication: Niagara Bistro async warmup failed: %s "
          "(load completed via direct path)",
          warmupError.c_str());
    }
    bistroAsyncLoad_.reset();

    bistroModel_ = modelResult.value();
    resources.setModelMaterialForAllSources(bistroModel_, bistroMaterialIndex_);
    if (!niagaraBistroPrefabScene_.ready) {
      loadImportedPrefabSceneResources(
          "Niagara Bistro", resolveNiagaraBistroPath().string(),
          nuri::MeshImportOptions{}, niagaraBistroPrefabScene_);
    }
    NURI_LOG_INFO("NuriApplication: Niagara Bistro model is ready in %.1f s",
                  getTime() - bistroLoadStartTimeSeconds_);

    if (scenePreset_ == ScenePreset::NiagaraBistro &&
        !nuri::isValid(bistroRenderableId_)) {
      setupNiagaraBistroScene();
    }
  }

  void setupNiagaraBistroScene() {
    nuri::ResourceManager &resources = getRenderer().resources();
    if (!nuri::isValid(bistroModel_)) {
      if (bistroLoadFailed_) {
        return;
      }
      beginNiagaraBistroModelLoad();
      return;
    }
    if (nuri::isValid(bistroRenderableId_)) {
      return;
    }
    NURI_ASSERT(nuri::isValid(bistroMaterialIndex_),
                "Niagara Bistro fallback material is not loaded");
    const nuri::ModelRecord *bistroRecord = resources.tryGet(bistroModel_);
    NURI_ASSERT(bistroRecord != nullptr && bistroRecord->model != nullptr,
                "Niagara Bistro model is not available in resource manager");
    const nuri::Model &bistroModel = *bistroRecord->model;

    // Keep Niagara on LOD0 until scene import is stable.
    renderSettings_.opaque.enableInstanceCompute = false;
    renderSettings_.opaque.enableMeshLod = false;
    renderSettings_.opaque.enableTessellation = false;
    renderSettings_.opaque.forcedMeshLod = 0;
    renderSettings_.opaque.meshLodDistanceThresholds =
        glm::vec3(8.0f, 24.0f, 48.0f);
    renderSettings_.opaque.enableInstanceAnimation = false;
    renderSettings_.textureFiltering.mode =
        nuri::TextureFilterMode::Anisotropic;
    renderSettings_.textureFiltering.anisotropy = 8u;
    const nuri::BoundingBox &bounds = bistroModel.bounds();
    const float bistroScale = computeNiagaraBistroScale(bounds);
    const glm::mat4 bistroModelMatrix = glm::mat4(1.0f);
    niagaraBistroBaseModel_ = bistroModelMatrix;

    bistroRenderableId_ = instantiateImportedPrefabScene(
        "NiagaraBistro", niagaraBistroPrefabScene_, niagaraBistroBaseModel_);
    if (!nuri::isValid(bistroRenderableId_)) {
      bistroRenderableId_ = addRequiredRenderable(
          bistroModel_, bistroMaterialIndex_, bistroModelMatrix,
          "Failed to add Niagara Bistro renderable");
    }

    const std::optional<nuri::BoundingBox> prefabBounds =
        computeImportedPrefabBounds(niagaraBistroPrefabScene_,
                                    niagaraBistroBaseModel_);
    const nuri::BoundingBox &framedBounds =
        prefabBounds.has_value() ? *prefabBounds : bounds;
    const FramedSceneCameraState cameraState = frameSceneCamera(
        framedBounds, glm::mat4(1.0f), 1.8f, 50.0f,
        glm::vec4(0.32f, 0.14f, 1.0f, 4.0f), glm::vec2(0.03f, 0.0f));
    NURI_LOG_INFO("NuriApplication: Niagara Bistro scene stats submeshes=%zu "
                  "vertices=%u indices=%u rawRadius=%.2f scale=%.6f "
                  "radius=%.2f near=%.3f far=%.2f",
                  bistroModel.submeshes().size(), bistroModel.vertexCount(),
                  bistroModel.indexCount(), cameraState.rawRadius, bistroScale,
                  cameraState.radius, cameraState.nearPlane,
                  cameraState.farPlane);
  }

  void releaseOwnedResourceHandles() {
    nuri::ResourceManager &resources = getRenderer().resources();
    const auto releaseRef = [&resources](auto &ref, const auto invalidRef) {
      if (nuri::isValid(ref)) {
        resources.release(ref);
      }
      ref = invalidRef;
    };

    releaseRef(duckModel_, nuri::kInvalidModelRef);
    releaseRef(helmetModel_, nuri::kInvalidModelRef);
    releaseRef(lightsPunctualLampModel_, nuri::kInvalidModelRef);
    releaseRef(clearcoatModel_, nuri::kInvalidModelRef);
    releaseRef(sheenChairModel_, nuri::kInvalidModelRef);
    releaseRef(specularSilkPoufModel_, nuri::kInvalidModelRef);
    releaseRef(dragonAttenuationModel_, nuri::kInvalidModelRef);
    releaseRef(dragonIorModel_, nuri::kInvalidModelRef);
    releaseRef(emissiveStrengthTestModel_, nuri::kInvalidModelRef);
    releaseRef(orreyModel_, nuri::kInvalidModelRef);
    releaseRef(bistroModel_, nuri::kInvalidModelRef);
    releaseRef(duckMaterialIndex_, nuri::kInvalidMaterialRef);
    releaseRef(helmetFallbackMaterialIndex_, nuri::kInvalidMaterialRef);
    releaseRef(lightsPunctualLampMaterial_, nuri::kInvalidMaterialRef);
    releaseRef(clearcoatMaterial_, nuri::kInvalidMaterialRef);
    releaseRef(sheenChairMaterial_, nuri::kInvalidMaterialRef);
    releaseRef(specularSilkPoufMaterial_, nuri::kInvalidMaterialRef);
    releaseRef(dragonAttenuationMaterial_, nuri::kInvalidMaterialRef);
    releaseRef(dragonIorMaterial_, nuri::kInvalidMaterialRef);
    releaseRef(emissiveStrengthTestMaterial_, nuri::kInvalidMaterialRef);
    releaseRef(orreyMaterial_, nuri::kInvalidMaterialRef);
    releaseRef(bistroMaterialIndex_, nuri::kInvalidMaterialRef);

    releaseImportedPrefabSceneResources(helmetPrefabScene_);
    releaseImportedPrefabSceneResources(niagaraBistroPrefabScene_);
    releaseImportedPrefabSceneResources(lightsPunctualLampPrefabScene_);
    releaseImportedPrefabSceneResources(clearcoatPrefabScene_);
    releaseImportedPrefabSceneResources(sheenChairPrefabScene_);
    releaseImportedPrefabSceneResources(specularSilkPoufPrefabScene_);
    releaseImportedPrefabSceneResources(dragonAttenuationPrefabScene_);
    releaseImportedPrefabSceneResources(dragonIorPrefabScene_);
    releaseImportedPrefabSceneResources(emissiveStrengthTestPrefabScene_);
    releaseImportedPrefabSceneResources(orreyPrefabScene_);
    releaseImportedPrefabSceneResources(foxPrefabScene_);
    releaseImportedPrefabSceneResources(medievalFantasyBookPrefabScene_);
  }

  void setupText3DTestScene() {
    configureStaticModelOpaqueSettings(glm::vec3(8.0f, 16.0f, 32.0f));

    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    NURI_ASSERT(camera != nullptr, "Failed to get main camera");
    nuri::PerspectiveParams perspective = camera->perspective();
    perspective.nearPlane = 0.01f;
    perspective.farPlane = 500.0f;
    camera->setProjectionType(nuri::ProjectionType::Perspective);
    camera->setPerspective(perspective);
    camera->setLookAt(glm::vec3(0.0f, 1.2f, -4.2f), glm::vec3(0.0f, 1.2f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f));
    syncEditorCameraWidgetState(*camera);
  }

  void configureStaticModelOpaqueSettings(const glm::vec3 &lodThresholds) {
    renderSettings_.opaque.enableInstanceCompute = false;
    renderSettings_.opaque.enableMeshLod = true;
    renderSettings_.opaque.enableTessellation = false;
    renderSettings_.opaque.forcedMeshLod = -1;
    renderSettings_.opaque.meshLodDistanceThresholds = lodThresholds;
    renderSettings_.opaque.enableInstanceAnimation = false;
  }

  [[nodiscard]] const nuri::Model &
  requireLoadedModel(nuri::ModelRef modelRef, std::string_view modelError,
                     std::string_view recordError) {
    nuri::ResourceManager &resources = getRenderer().resources();
    NURI_ASSERT(nuri::isValid(modelRef), "%s", std::string(modelError).c_str());
    const nuri::ModelRecord *record = resources.tryGet(modelRef);
    NURI_ASSERT(record != nullptr && record->model != nullptr, "%s",
                std::string(recordError).c_str());
    return *record->model;
  }

  [[nodiscard]] nuri::RenderableId
  addRequiredRenderable(nuri::ModelRef modelRef, nuri::MaterialRef materialRef,
                        const glm::mat4 &modelMatrix,
                        std::string_view errorMessage) {
    auto nodeResult =
        scene_.graph().createNode(scene_.graph().rootNode(), {}, modelMatrix);
    NURI_ASSERT(!nodeResult.hasError(), "%s: %s",
                std::string(errorMessage).c_str(), nodeResult.error().c_str());
    auto addResult =
        scene_.graph().addRenderable(nodeResult.value(), modelRef, materialRef);
    NURI_ASSERT(!addResult.hasError(), "%s: %s",
                std::string(errorMessage).c_str(), addResult.error().c_str());
    return addResult.value();
  }

  void setRequiredModelSourceMaterial(nuri::ModelRef modelRef,
                                      uint32_t sourceMaterialIndex,
                                      nuri::MaterialRef materialRef,
                                      std::string_view errorMessage) {
    NURI_ASSERT(nuri::isValid(modelRef), "%s: invalid model ref",
                std::string(errorMessage).c_str());
    NURI_ASSERT(nuri::isValid(materialRef), "%s: invalid material ref",
                std::string(errorMessage).c_str());
    nuri::ResourceManager &resources = getRenderer().resources();
    const bool mapped = resources.setModelMaterialForSource(
        modelRef, sourceMaterialIndex, materialRef);
    NURI_ASSERT(mapped, "%s: failed to map source material %u",
                std::string(errorMessage).c_str(), sourceMaterialIndex);
  }

  [[nodiscard]] FramedSceneCameraState
  frameSceneCamera(const nuri::BoundingBox &bounds,
                   const glm::mat4 &modelMatrix, float distanceScale,
                   float minDistance, const glm::vec4 &eyeOffsetParams,
                   const glm::vec2 &targetOffsetParams) {
    FramedSceneCameraState state{};
    state.rawRadius = std::max(0.5f * glm::length(bounds.getSize()), 0.25f);
    state.center = glm::vec3(modelMatrix * glm::vec4(bounds.getCenter(), 1.0f));
    state.radius = std::max(0.25f, state.rawRadius);
    state.cameraDistance = std::max(state.radius * distanceScale, minDistance);
    state.nearPlane = std::max(0.01f, state.cameraDistance / 3000.0f);
    state.farPlane =
        std::max(500.0f, state.cameraDistance + state.radius * 12.0f);

    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    NURI_ASSERT(camera != nullptr, "Failed to get main camera");
    nuri::PerspectiveParams perspective = camera->perspective();
    perspective.nearPlane = state.nearPlane;
    perspective.farPlane = state.farPlane;
    camera->setProjectionType(nuri::ProjectionType::Perspective);
    camera->setPerspective(perspective);
    camera->setLookAt(
        state.center +
            glm::vec3(-state.cameraDistance * eyeOffsetParams.x,
                      state.radius * eyeOffsetParams.y + eyeOffsetParams.w,
                      -state.cameraDistance * eyeOffsetParams.z),
        state.center + glm::vec3(0.0f, state.radius * targetOffsetParams.x,
                                 targetOffsetParams.y),
        glm::vec3(0.0f, 1.0f, 0.0f));
    syncEditorCameraWidgetState(*camera);
    return state;
  }

  [[nodiscard]] std::optional<nuri::BoundingBox>
  computeImportedPrefabBounds(const ImportedPrefabSceneResources &scene,
                              const glm::mat4 &baseModel) {
    if (!scene.ready || scene.prefab.nodes.empty()) {
      return std::nullopt;
    }

    std::vector<glm::mat4> worldMatrices(scene.prefab.nodes.size(),
                                         glm::mat4(1.0f));
    for (size_t nodeIndex = 0; nodeIndex < scene.prefab.nodes.size();
         ++nodeIndex) {
      const nuri::ScenePrefabNode &node = scene.prefab.nodes[nodeIndex];
      if (node.parentIndex == nuri::kInvalidScenePrefabIndex) {
        worldMatrices[nodeIndex] = baseModel * node.localFromParent;
      } else if (node.parentIndex < worldMatrices.size()) {
        worldMatrices[nodeIndex] =
            worldMatrices[node.parentIndex] * node.localFromParent;
      } else {
        worldMatrices[nodeIndex] = baseModel * node.localFromParent;
      }
    }

    nuri::ResourceManager &resources = getRenderer().resources();
    nuri::BoundingBox bounds{};
    bool hasBounds = false;
    for (const nuri::ScenePrefabRenderable &renderable :
         scene.prefab.renderables) {
      if (renderable.nodeIndex >= worldMatrices.size() ||
          renderable.meshIndex >= scene.assets.models.size()) {
        continue;
      }
      const nuri::ModelRef modelRef = scene.assets.models[renderable.meshIndex];
      const nuri::ModelRecord *modelRecord = resources.tryGet(modelRef);
      if (modelRecord == nullptr || modelRecord->model == nullptr) {
        continue;
      }

      const nuri::BoundingBox worldBounds =
          modelRecord->model->bounds().getTransformed(
              worldMatrices[renderable.nodeIndex]);
      if (!hasBounds) {
        bounds = worldBounds;
        hasBounds = true;
      } else {
        bounds.combinePoint(worldBounds.min_);
        bounds.combinePoint(worldBounds.max_);
      }
    }

    if (!hasBounds) {
      return std::nullopt;
    }
    return bounds;
  }

  [[nodiscard]] std::optional<nuri::BoundingBox>
  computeImportedPrefabNodeBounds(const ImportedPrefabSceneResources &scene,
                                  const glm::mat4 &baseModel,
                                  std::string_view nodeName) {
    if (!scene.ready || scene.prefab.nodes.empty() || nodeName.empty()) {
      return std::nullopt;
    }

    std::vector<glm::mat4> worldMatrices(scene.prefab.nodes.size(),
                                         glm::mat4(1.0f));
    for (size_t nodeIndex = 0; nodeIndex < scene.prefab.nodes.size();
         ++nodeIndex) {
      const nuri::ScenePrefabNode &node = scene.prefab.nodes[nodeIndex];
      if (node.parentIndex == nuri::kInvalidScenePrefabIndex) {
        worldMatrices[nodeIndex] = baseModel * node.localFromParent;
      } else if (node.parentIndex < worldMatrices.size()) {
        worldMatrices[nodeIndex] =
            worldMatrices[node.parentIndex] * node.localFromParent;
      } else {
        worldMatrices[nodeIndex] = baseModel * node.localFromParent;
      }
    }

    nuri::ResourceManager &resources = getRenderer().resources();
    nuri::BoundingBox bounds{};
    bool hasBounds = false;
    for (const nuri::ScenePrefabRenderable &renderable :
         scene.prefab.renderables) {
      if (renderable.nodeIndex >= worldMatrices.size() ||
          renderable.meshIndex >= scene.assets.models.size()) {
        continue;
      }
      const nuri::ScenePrefabNode &node =
          scene.prefab.nodes[renderable.nodeIndex];
      if (node.name != nodeName) {
        continue;
      }
      const nuri::ModelRef modelRef = scene.assets.models[renderable.meshIndex];
      const nuri::ModelRecord *modelRecord = resources.tryGet(modelRef);
      if (modelRecord == nullptr || modelRecord->model == nullptr) {
        continue;
      }

      const nuri::BoundingBox worldBounds =
          modelRecord->model->bounds().getTransformed(
              worldMatrices[renderable.nodeIndex]);
      if (!hasBounds) {
        bounds = worldBounds;
        hasBounds = true;
      } else {
        bounds.combinePoint(worldBounds.min_);
        bounds.combinePoint(worldBounds.max_);
      }
    }

    if (!hasBounds) {
      return std::nullopt;
    }
    return bounds;
  }

  void
  destroyAnimatedPrefabSceneInstance(AnimatedPrefabSceneInstance &instance) {
    if (nuri::isValid(instance.simulation)) {
      (void)sceneRuntime_.destroyAnimationPoseSimulation(instance.simulation);
      instance.simulation = nuri::kInvalidSimulationHandle;
    }
    instance.rootNode = nuri::kInvalidNodeId;
    instance.instantiationMap.nodes.clear();
    instance.instantiationMap.renderables.clear();
    instance.instantiationMap.lights.clear();
  }

  void destroyFoxAnimation() {
    destroyAnimatedPrefabSceneInstance(foxAnimation_);
  }

  void destroyMedievalFantasyBookAnimation() {
    destroyAnimatedPrefabSceneInstance(medievalFantasyBookAnimation_);
  }

  void syncSceneRuntimeForCommittedScene() {
    (void)sceneRuntime_.tick({
        .frameDeltaSeconds = 0.0,
        .absoluteTimeSeconds = getTime(),
        .frameIndex = simulationFrameIndex_++,
    });
  }

  [[nodiscard]] static uint32_t
  selectPreferredClipIndex(const nuri::ScenePrefab &prefab,
                           std::span<const std::string_view> preferredNames) {
    if (prefab.animations.empty()) {
      return 0u;
    }
    for (std::string_view preferredName : preferredNames) {
      for (uint32_t i = 0; i < prefab.animations.size(); ++i) {
        if (std::string_view(prefab.animations[i].name) == preferredName) {
          return i;
        }
      }
    }
    return 0u;
  }

  void startAnimatedPrefabSceneSimulation(
      std::string_view sceneName, const ImportedPrefabSceneResources &resources,
      AnimatedPrefabSceneInstance &instance, uint32_t clipIndex,
      std::string_view simulationDebugName) {
    const std::string sceneNameString(sceneName);
    NURI_ASSERT(!resources.prefab.animations.empty(),
                "%s prefab has no animations", sceneNameString.c_str());
    NURI_ASSERT(clipIndex < resources.prefab.animations.size(),
                "%s clip index %u is out of range", sceneNameString.c_str(),
                clipIndex);

    const auto commitResult = scene_.commit();
    NURI_ASSERT(!commitResult.hasError(), "Scene commit failed for %s: %s",
                sceneNameString.c_str(), commitResult.error().c_str());
    syncSceneRuntimeForCommittedScene();

    const nuri::AnimationClipData &clip =
        resources.prefab.animations[clipIndex];
    NURI_LOG_INFO("NuriApplication::setup%sScene: playing clip %u '%s' "
                  "(clips=%zu duration=%.3fs)",
                  sceneNameString.c_str(), clipIndex, clip.name.c_str(),
                  resources.prefab.animations.size(), clip.durationSeconds);

    auto simulationResult = sceneRuntime_.createAnimationPoseSimulation(
        nuri::AnimationPoseSimulationCreateInfo{
            .prefab = &resources.prefab,
            .instantiationMap = &instance.instantiationMap,
            .rootNode = instance.rootNode,
            .debugName = simulationDebugName,
            .params =
                nuri::AnimationPoseSimulationParams{
                    .clipIndex = clipIndex,
                    .timeSeconds = 0.0f,
                    .playbackMode = nuri::AnimationPosePlaybackMode::Loop,
                    .playing = true,
                },
        });
    NURI_ASSERT(!simulationResult.hasError(),
                "Failed to create %s animation simulation: %s",
                sceneNameString.c_str(), simulationResult.error().c_str());
    instance.simulation = simulationResult.value();

    syncSceneRuntimeForCommittedScene();
  }

  void
  logSingleRenderableSceneStats(std::string_view sceneName,
                                const nuri::Model &model,
                                const FramedSceneCameraState &cameraState) {
    NURI_LOG_INFO("NuriApplication: %s scene stats submeshes=%zu vertices=%u "
                  "indices=%u rawRadius=%.2f radius=%.2f near=%.3f far=%.2f",
                  std::string(sceneName).c_str(), model.submeshes().size(),
                  model.vertexCount(), model.indexCount(),
                  cameraState.rawRadius, cameraState.radius,
                  cameraState.nearPlane, cameraState.farPlane);
  }

  [[nodiscard]] FramedSceneCameraState
  configureDragonSampleCamera(const ImportedPrefabSceneResources &scene,
                              const glm::mat4 &baseModel,
                              const nuri::BoundingBox &dragonBounds) {
    FramedSceneCameraState cameraState{};
    cameraState.rawRadius =
        std::max(0.5f * glm::length(dragonBounds.getSize()), 0.25f);
    cameraState.radius = std::max(0.25f, cameraState.rawRadius);
    cameraState.cameraDistance = std::max(cameraState.radius * 2.05f, 4.0f);
    cameraState.nearPlane = 0.01f;
    cameraState.farPlane = 500.0f;
    cameraState.center = dragonBounds.getCenter();

    glm::vec3 viewDirection = glm::vec3(0.0f, 0.0f, 1.0f);
    if (const std::optional<nuri::BoundingBox> backdropBounds =
            computeImportedPrefabNodeBounds(scene, baseModel, "Cloth Backdrop");
        backdropBounds.has_value()) {
      viewDirection = cameraState.center - backdropBounds->getCenter();
      viewDirection.y = 0.0f;
      if (glm::dot(viewDirection, viewDirection) > 1.0e-4f) {
        viewDirection = glm::normalize(viewDirection);
      } else {
        viewDirection = glm::vec3(0.0f, 0.0f, 1.0f);
      }
    }

    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    NURI_ASSERT(camera != nullptr, "Failed to get main camera");
    nuri::PerspectiveParams perspective = camera->perspective();
    perspective.nearPlane = cameraState.nearPlane;
    perspective.farPlane = cameraState.farPlane;
    camera->setProjectionType(nuri::ProjectionType::Perspective);
    camera->setPerspective(perspective);

    glm::vec3 eye =
        cameraState.center + viewDirection * cameraState.cameraDistance;
    eye.y = std::max(cameraState.center.y + cameraState.radius * 0.2f, 1.0f);
    camera->setLookAt(eye, cameraState.center, glm::vec3(0.0f, 1.0f, 0.0f));
    syncEditorCameraWidgetState(*camera);
    return cameraState;
  }

  void resetScenePresetState() {
    destroyFoxAnimation();
    destroyMedievalFantasyBookAnimation();
    scene_.graph().clear();
    if (editorOverlay_ != nullptr) {
      editorOverlay_->resetSceneUiState();
    }
    duckRenderableId_ = nuri::kInvalidRenderableId;
    bistroRenderableId_ = nuri::kInvalidRenderableId;
    helmetRenderableId_ = nuri::kInvalidRenderableId;
    lightsPunctualLampRenderableId_ = nuri::kInvalidRenderableId;
    clearcoatRenderableId_ = nuri::kInvalidRenderableId;
    sheenChairRenderableId_ = nuri::kInvalidRenderableId;
    specularSilkPoufRenderableId_ = nuri::kInvalidRenderableId;
    dragonAttenuationRenderableId_ = nuri::kInvalidRenderableId;
    dragonIorRenderableId_ = nuri::kInvalidRenderableId;
    emissiveStrengthTestRenderableId_ = nuri::kInvalidRenderableId;
    orreyRenderableId_ = nuri::kInvalidRenderableId;
    foxRenderableId_ = nuri::kInvalidRenderableId;
    medievalFantasyBookRenderableId_ = nuri::kInvalidRenderableId;
    sceneHasAuthoredLights_ = false;
    renderSettings_ = nuri::RenderSettings{};
  }

  void setupDefaultSceneLighting() {
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

  void setupDamagedHelmetScene() {
    NURI_ASSERT(nuri::isValid(helmetModel_),
                "DamagedHelmet model is not loaded");
    NURI_ASSERT(nuri::isValid(helmetFallbackMaterialIndex_),
                "DamagedHelmet material is not loaded");
    if (nuri::isValid(helmetRenderableId_)) {
      return;
    }
    const nuri::Model &helmetModel =
        requireLoadedModel(helmetModel_, "DamagedHelmet model is not loaded",
                           "DamagedHelmet model record lookup failed");

    configureStaticModelOpaqueSettings(glm::vec3(8.0f, 16.0f, 32.0f));

    helmetRenderableId_ = instantiateImportedPrefabScene(
        "DamagedHelmet", helmetPrefabScene_, helmetBaseModel_);
    if (!nuri::isValid(helmetRenderableId_)) {
      helmetRenderableId_ = addRequiredRenderable(
          helmetModel_, helmetFallbackMaterialIndex_, helmetBaseModel_,
          "Failed to add DamagedHelmet renderable");
    }
    const FramedSceneCameraState cameraState = frameSceneCamera(
        helmetModel.bounds(), helmetBaseModel_, 2.4f, 2.0f,
        glm::vec4(0.38f, 0.18f, 1.0f, 0.2f), glm::vec2(0.03f, 0.0f));
    logSingleRenderableSceneStats("DamagedHelmet", helmetModel, cameraState);
  }

  void setupLightsPunctualLampScene() {
    NURI_ASSERT(nuri::isValid(lightsPunctualLampModel_),
                "LightsPunctualLamp model is not loaded");
    NURI_ASSERT(nuri::isValid(lightsPunctualLampMaterial_),
                "LightsPunctualLamp material is not loaded");
    if (nuri::isValid(lightsPunctualLampRenderableId_)) {
      return;
    }

    const nuri::Model &lampModel = requireLoadedModel(
        lightsPunctualLampModel_, "LightsPunctualLamp model is not loaded",
        "LightsPunctualLamp model record lookup failed");

    configureStaticModelOpaqueSettings(glm::vec3(6.0f, 12.0f, 24.0f));
    renderSettings_.debug.enabled = true;
    renderSettings_.debug.grid = true;
    renderSettings_.debug.modelBounds = false;

    lightsPunctualLampRenderableId_ = instantiateImportedPrefabScene(
        "LightsPunctualLamp", lightsPunctualLampPrefabScene_,
        lightsPunctualLampBaseModel_);
    if (!nuri::isValid(lightsPunctualLampRenderableId_)) {
      lightsPunctualLampRenderableId_ = addRequiredRenderable(
          lightsPunctualLampModel_, lightsPunctualLampMaterial_,
          lightsPunctualLampBaseModel_,
          "Failed to add LightsPunctualLamp renderable");
    }
    const FramedSceneCameraState cameraState = frameSceneCamera(
        lampModel.bounds(), lightsPunctualLampBaseModel_, 2.5f, 2.0f,
        glm::vec4(0.42f, 0.25f, 1.0f, 0.1f), glm::vec2(0.08f, 0.0f));
    logSingleRenderableSceneStats("LightsPunctualLamp", lampModel, cameraState);
  }

  void setupClearcoatWickerScene() {
    NURI_ASSERT(nuri::isValid(clearcoatModel_),
                "ClearcoatWicker model is not loaded");
    NURI_ASSERT(nuri::isValid(clearcoatMaterial_),
                "ClearcoatWicker material is not loaded");
    if (nuri::isValid(clearcoatRenderableId_)) {
      return;
    }
    const nuri::Model &clearcoatModel = requireLoadedModel(
        clearcoatModel_, "ClearcoatWicker model is not loaded",
        "ClearcoatWicker model record lookup failed");

    configureStaticModelOpaqueSettings(glm::vec3(8.0f, 16.0f, 32.0f));

    clearcoatRenderableId_ = instantiateImportedPrefabScene(
        "ClearcoatWicker", clearcoatPrefabScene_, clearcoatBaseModel_);
    if (!nuri::isValid(clearcoatRenderableId_)) {
      clearcoatRenderableId_ = addRequiredRenderable(
          clearcoatModel_, clearcoatMaterial_, clearcoatBaseModel_,
          "Failed to add ClearcoatWicker renderable");
    }
    const FramedSceneCameraState cameraState = frameSceneCamera(
        clearcoatModel.bounds(), clearcoatBaseModel_, 2.4f, 2.0f,
        glm::vec4(0.28f, 0.12f, 1.0f, 0.15f), glm::vec2(0.03f, 0.0f));
    logSingleRenderableSceneStats("ClearcoatWicker", clearcoatModel,
                                  cameraState);
  }

  void setupSheenChairScene() {
    NURI_ASSERT(nuri::isValid(sheenChairModel_),
                "SheenChair model is not loaded");
    NURI_ASSERT(nuri::isValid(sheenChairMaterial_),
                "SheenChair material is not loaded");
    if (nuri::isValid(sheenChairRenderableId_)) {
      return;
    }
    const nuri::Model &sheenChairModel =
        requireLoadedModel(sheenChairModel_, "SheenChair model is not loaded",
                           "SheenChair model record lookup failed");

    configureStaticModelOpaqueSettings(glm::vec3(6.0f, 12.0f, 24.0f));

    sheenChairRenderableId_ = instantiateImportedPrefabScene(
        "SheenChair", sheenChairPrefabScene_, glm::mat4(1.0f));
    if (!nuri::isValid(sheenChairRenderableId_)) {
      sheenChairRenderableId_ = addRequiredRenderable(
          sheenChairModel_, sheenChairMaterial_, glm::mat4(1.0f),
          "Failed to add SheenChair renderable");
    }
    const FramedSceneCameraState cameraState = frameSceneCamera(
        sheenChairModel.bounds(), glm::mat4(1.0f), 2.5f, 2.0f,
        glm::vec4(0.52f, 0.52f, 1.0f, 0.0f), glm::vec2(0.1f, 0.0f));
    logSingleRenderableSceneStats("SheenChair", sheenChairModel, cameraState);
  }

  void setupSpecularSilkPoufScene() {
    NURI_ASSERT(nuri::isValid(specularSilkPoufModel_),
                "SpecularSilkPouf model is not loaded");
    NURI_ASSERT(nuri::isValid(specularSilkPoufMaterial_),
                "SpecularSilkPouf material is not loaded");
    if (nuri::isValid(specularSilkPoufRenderableId_)) {
      return;
    }
    const nuri::Model &specularSilkPoufModel = requireLoadedModel(
        specularSilkPoufModel_, "SpecularSilkPouf model is not loaded",
        "SpecularSilkPouf model record lookup failed");

    configureStaticModelOpaqueSettings(glm::vec3(6.0f, 12.0f, 24.0f));

    specularSilkPoufRenderableId_ = instantiateImportedPrefabScene(
        "SpecularSilkPouf", specularSilkPoufPrefabScene_, glm::mat4(1.0f));
    if (!nuri::isValid(specularSilkPoufRenderableId_)) {
      specularSilkPoufRenderableId_ = addRequiredRenderable(
          specularSilkPoufModel_, specularSilkPoufMaterial_, glm::mat4(1.0f),
          "Failed to add SpecularSilkPouf renderable");
    }
    const FramedSceneCameraState cameraState = frameSceneCamera(
        specularSilkPoufModel.bounds(), glm::mat4(1.0f), 2.5f, 2.0f,
        glm::vec4(0.52f, 0.52f, 1.0f, 0.0f), glm::vec2(0.1f, 0.0f));
    logSingleRenderableSceneStats("SpecularSilkPouf", specularSilkPoufModel,
                                  cameraState);
  }

  void setupDragonAttenuationScene() {
    if (nuri::isValid(dragonAttenuationRenderableId_)) {
      return;
    }

    const nuri::Model &dragonModel = requireLoadedModel(
        dragonAttenuationModel_, "DragonAttenuation model is not loaded",
        "DragonAttenuation model record lookup failed");

    configureStaticModelOpaqueSettings(glm::vec3(6.0f, 12.0f, 24.0f));
    renderSettings_.skybox.enabled = false;
    renderSettings_.transparent.enabled = true;
    renderSettings_.transmission.enabled = true;
    renderSettings_.debug.enabled = true;
    renderSettings_.debug.grid = true;
    renderSettings_.debug.modelBounds = false;

    const glm::mat4 dragonBaseModel = glm::rotate(
        glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    dragonAttenuationRenderableId_ = instantiateImportedPrefabScene(
        "DragonAttenuation", dragonAttenuationPrefabScene_, dragonBaseModel);
    NURI_ASSERT(nuri::isValid(dragonAttenuationRenderableId_),
                "Failed to instantiate DragonAttenuation prefab scene");
    const nuri::BoundingBox framedBounds =
        computeImportedPrefabNodeBounds(dragonAttenuationPrefabScene_,
                                        dragonBaseModel, "Dragon")
            .value_or(dragonModel.bounds());
    const FramedSceneCameraState cameraState = configureDragonSampleCamera(
        dragonAttenuationPrefabScene_, dragonBaseModel, framedBounds);
    logSingleRenderableSceneStats("DragonAttenuation", dragonModel,
                                  cameraState);
  }

  void setupDragonIorScene() {
    if (nuri::isValid(dragonIorRenderableId_)) {
      return;
    }

    const nuri::Model &dragonModel =
        requireLoadedModel(dragonIorModel_, "DragonIor model is not loaded",
                           "DragonIor model record lookup failed");

    configureStaticModelOpaqueSettings(glm::vec3(6.0f, 12.0f, 24.0f));
    renderSettings_.skybox.enabled = false;
    renderSettings_.transparent.enabled = true;
    renderSettings_.transmission.enabled = true;
    renderSettings_.debug.enabled = true;
    renderSettings_.debug.grid = true;
    renderSettings_.debug.modelBounds = false;

    const glm::mat4 dragonBaseModel = glm::rotate(
        glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    dragonIorRenderableId_ = instantiateImportedPrefabScene(
        "DragonIor", dragonIorPrefabScene_, dragonBaseModel);
    NURI_ASSERT(nuri::isValid(dragonIorRenderableId_),
                "Failed to instantiate DragonIor prefab scene");
    const nuri::BoundingBox framedBounds =
        computeImportedPrefabNodeBounds(dragonIorPrefabScene_, dragonBaseModel,
                                        "Dragon")
            .value_or(dragonModel.bounds());
    const FramedSceneCameraState cameraState = configureDragonSampleCamera(
        dragonIorPrefabScene_, dragonBaseModel, framedBounds);
    logSingleRenderableSceneStats("DragonIor", dragonModel, cameraState);
  }

  void setupEmissiveStrengthTestScene() {
    NURI_ASSERT(nuri::isValid(emissiveStrengthTestModel_),
                "EmissiveStrengthTest model is not loaded");
    NURI_ASSERT(nuri::isValid(emissiveStrengthTestMaterial_),
                "EmissiveStrengthTest material is not loaded");
    if (nuri::isValid(emissiveStrengthTestRenderableId_)) {
      return;
    }

    const nuri::Model &emissiveStrengthTestModel = requireLoadedModel(
        emissiveStrengthTestModel_, "EmissiveStrengthTest model is not loaded",
        "EmissiveStrengthTest model record lookup failed");

    configureStaticModelOpaqueSettings(glm::vec3(6.0f, 12.0f, 24.0f));
    renderSettings_.skybox.enabled = false;

    emissiveStrengthTestRenderableId_ = instantiateImportedPrefabScene(
        "EmissiveStrengthTest", emissiveStrengthTestPrefabScene_,
        glm::mat4(1.0f));
    if (!nuri::isValid(emissiveStrengthTestRenderableId_)) {
      emissiveStrengthTestRenderableId_ = addRequiredRenderable(
          emissiveStrengthTestModel_, emissiveStrengthTestMaterial_,
          glm::mat4(1.0f), "Failed to add EmissiveStrengthTest renderable");
    }
    const FramedSceneCameraState cameraState = frameSceneCamera(
        emissiveStrengthTestModel.bounds(), glm::mat4(1.0f), 2.5f, 2.0f,
        glm::vec4(0.52f, 0.52f, 1.0f, 0.0f), glm::vec2(0.1f, 0.0f));
    logSingleRenderableSceneStats("EmissiveStrengthTest",
                                  emissiveStrengthTestModel, cameraState);
  }

  void setupOrreyScene() {
    NURI_ASSERT(nuri::isValid(orreyModel_), "Orrey model is not loaded");
    NURI_ASSERT(nuri::isValid(orreyMaterial_), "Orrey material is not loaded");
    if (nuri::isValid(orreyRenderableId_)) {
      return;
    }

    const nuri::Model &orreyModel =
        requireLoadedModel(orreyModel_, "Orrey model is not loaded",
                           "Orrey model record lookup failed");

    configureStaticModelOpaqueSettings(glm::vec3(12.0f, 24.0f, 48.0f));
    renderSettings_.debug.enabled = true;
    renderSettings_.debug.grid = true;
    renderSettings_.debug.modelBounds = false;

    orreyRenderableId_ = instantiateImportedPrefabScene(
        "Orrey", orreyPrefabScene_, orreyBaseModel_);
    if (!nuri::isValid(orreyRenderableId_)) {
      orreyRenderableId_ =
          addRequiredRenderable(orreyModel_, orreyMaterial_, orreyBaseModel_,
                                "Failed to add Orrey renderable");
    }
    const FramedSceneCameraState cameraState = frameSceneCamera(
        orreyModel.bounds(), orreyBaseModel_, 2.8f, 4.0f,
        glm::vec4(0.42f, 0.20f, 1.0f, 0.35f), glm::vec2(0.06f, 0.0f));
    logSingleRenderableSceneStats("Orrey", orreyModel, cameraState);
  }

  void setupMedievalFantasyBookScene() {
    NURI_ASSERT(medievalFantasyBookPrefabScene_.ready,
                "MedievalFantasyBook prefab scene is not loaded");
    if (nuri::isValid(medievalFantasyBookRenderableId_)) {
      return;
    }

    configureStaticModelOpaqueSettings(glm::vec3(10.0f, 20.0f, 40.0f));
    renderSettings_.transparent.enabled = true;
    renderSettings_.transmission.enabled = true;
    renderSettings_.debug.enabled = true;
    renderSettings_.debug.grid = true;
    renderSettings_.debug.modelBounds = false;

    medievalFantasyBookRenderableId_ = instantiateImportedPrefabScene(
        "MedievalFantasyBook", medievalFantasyBookPrefabScene_, glm::mat4(1.0f),
        &medievalFantasyBookAnimation_.instantiationMap,
        &medievalFantasyBookAnimation_.rootNode);
    NURI_ASSERT(nuri::isValid(medievalFantasyBookRenderableId_),
                "Failed to instantiate MedievalFantasyBook prefab scene");
    startAnimatedPrefabSceneSimulation(
        "MedievalFantasyBook", medievalFantasyBookPrefabScene_,
        medievalFantasyBookAnimation_, 0u, "MedievalFantasyBookAnimation");

    const nuri::BoundingBox framedBounds =
        computeImportedPrefabBounds(medievalFantasyBookPrefabScene_,
                                    glm::mat4(1.0f))
            .value_or(nuri::BoundingBox{});
    const FramedSceneCameraState cameraState = frameSceneCamera(
        framedBounds, glm::mat4(1.0f), 2.3f, 2.0f,
        glm::vec4(0.48f, 0.18f, 1.0f, 0.2f), glm::vec2(0.06f, 0.0f));
    NURI_LOG_INFO(
        "NuriApplication::setupMedievalFantasyBookScene: renderables=%zu "
        "nodes=%zu radius=%.2f near=%.3f far=%.2f",
        medievalFantasyBookPrefabScene_.prefab.renderables.size(),
        medievalFantasyBookPrefabScene_.prefab.nodes.size(), cameraState.radius,
        cameraState.nearPlane, cameraState.farPlane);
  }

  void setupFoxScene() {
    NURI_ASSERT(foxPrefabScene_.ready, "Fox prefab scene is not loaded");
    if (nuri::isValid(foxRenderableId_)) {
      return;
    }

    configureStaticModelOpaqueSettings(glm::vec3(8.0f, 16.0f, 32.0f));
    renderSettings_.debug.enabled = true;
    renderSettings_.debug.grid = true;
    renderSettings_.debug.modelBounds = false;

    foxRenderableId_ = instantiateImportedPrefabScene(
        "Fox", foxPrefabScene_, foxBaseModel_, &foxAnimation_.instantiationMap,
        &foxAnimation_.rootNode);
    NURI_ASSERT(nuri::isValid(foxRenderableId_),
                "Failed to instantiate Fox prefab scene");
    const uint32_t foxClipIndex = selectPreferredClipIndex(
        foxPrefabScene_.prefab, std::array<std::string_view, 2>{"Run", "Walk"});
    startAnimatedPrefabSceneSimulation("Fox", foxPrefabScene_, foxAnimation_,
                                       foxClipIndex, "FoxAnimation");

    const nuri::BoundingBox framedBounds =
        computeImportedPrefabBounds(foxPrefabScene_, foxBaseModel_)
            .value_or(nuri::BoundingBox{});
    const FramedSceneCameraState cameraState = frameSceneCamera(
        framedBounds, foxBaseModel_, 1.55f, 1.0f,
        glm::vec4(0.80f, 0.12f, 0.20f, 0.10f), glm::vec2(0.06f, 0.0f));
    NURI_LOG_INFO(
        "NuriApplication::setupFoxScene: renderables=%zu nodes=%zu radius=%.2f "
        "near=%.3f far=%.2f",
        foxPrefabScene_.prefab.renderables.size(),
        foxPrefabScene_.prefab.nodes.size(), cameraState.radius,
        cameraState.nearPlane, cameraState.farPlane);
  }

  void applyScenePreset(ScenePreset preset) {
    NURI_ASSERT(nuri::isValid(duckModel_), "Duck model is not loaded");
    NURI_ASSERT(nuri::isValid(duckMaterialIndex_),
                "Duck material is not loaded");

    resetScenePresetState();
    scenePreset_ = preset;

    switch (scenePreset_) {
    case ScenePreset::InstancedDuck32K:
      setupInstancedDuckScene32k();
      break;
    case ScenePreset::NiagaraBistro:
      bistroLoadFailed_ = false;
      bistroLoadError_.clear();
      setupNiagaraBistroScene();
      break;
    case ScenePreset::DamagedHelmet:
      setupDamagedHelmetScene();
      break;
    case ScenePreset::LightsPunctualLamp:
      setupLightsPunctualLampScene();
      break;
    case ScenePreset::ClearcoatWicker:
      setupClearcoatWickerScene();
      break;
    case ScenePreset::SheenChair:
      setupSheenChairScene();
      break;
    case ScenePreset::SpecularSilkPouf:
      setupSpecularSilkPoufScene();
      break;
    case ScenePreset::DragonAttenuation:
      setupDragonAttenuationScene();
      break;
    case ScenePreset::DragonIor:
      setupDragonIorScene();
      break;
    case ScenePreset::EmissiveStrengthTest:
      setupEmissiveStrengthTestScene();
      break;
    case ScenePreset::Orrey:
      setupOrreyScene();
      break;
    case ScenePreset::Fox:
      setupFoxScene();
      break;
    case ScenePreset::MedievalFantasyBook:
      setupMedievalFantasyBookScene();
      break;
    case ScenePreset::Text3DTest:
      setupText3DTestScene();
      break;
    case ScenePreset::SingleDuck:
      loadSingleDuckSceneResources();
      break;
    }

    if (!sceneHasAuthoredLights_ && !applyImportedLightsForCurrentPreset()) {
      setupDefaultSceneLighting();
    }
  }

  void requestScenePreset(ScenePreset preset) {
    if (preset == scenePreset_) {
      return;
    }
    pendingScenePreset_ = preset;
  }

  void applyPendingScenePreset() {
    if (!pendingScenePreset_.has_value()) {
      return;
    }

    const ScenePreset preset = *pendingScenePreset_;
    pendingScenePreset_.reset();
    applyScenePreset(preset);
    NURI_LOG_INFO("NuriApplication::toggleScenePreset: switched to %s",
                  kScenePresetNames[scenePresetToIndex(scenePreset_)]);
  }

  void syncEditorCameraWidgetState(const nuri::Camera &camera) {
    if (editorOverlay_ != nullptr) {
      editorOverlay_->syncCameraControllerWidgetStateFromCamera(camera);
    }
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
    frameContext_.sharedDepthTexture = {};
    frameContext_.timeSeconds = timeSeconds;
    frameContext_.frameIndex = frameIndex_++;
  }

  void submitPipelineFrame() {
    auto renderResult =
        getRenderer().render(getRenderPipeline(), frameContext_);
    NURI_ASSERT(!renderResult.hasError(), "Render failed: %s",
                renderResult.error().c_str());
  }

  const nuri::RuntimeConfig config_;
  ScenePreset scenePreset_ = kScenePreset;
  std::pmr::unsynchronized_pool_resource cameraMemory_;
  std::pmr::unsynchronized_pool_resource sceneMemory_;
  nuri::CameraSystem cameraSystem_;
  nuri::RenderScene scene_;
  nuri::SceneRuntimeHost sceneRuntime_;
  nuri::SceneEditorSelectionState sceneEditorSelectionState_{};
  nuri::ModelRef duckModel_ = nuri::kInvalidModelRef;
  nuri::ModelRef helmetModel_ = nuri::kInvalidModelRef;
  nuri::ModelRef lightsPunctualLampModel_ = nuri::kInvalidModelRef;
  nuri::ModelRef clearcoatModel_ = nuri::kInvalidModelRef;
  nuri::ModelRef sheenChairModel_ = nuri::kInvalidModelRef;
  nuri::ModelRef specularSilkPoufModel_ = nuri::kInvalidModelRef;
  nuri::ModelRef dragonAttenuationModel_ = nuri::kInvalidModelRef;
  nuri::ModelRef dragonIorModel_ = nuri::kInvalidModelRef;
  nuri::ModelRef emissiveStrengthTestModel_ = nuri::kInvalidModelRef;
  nuri::ModelRef orreyModel_ = nuri::kInvalidModelRef;
  nuri::ModelRef bistroModel_ = nuri::kInvalidModelRef;
  std::unique_ptr<nuri::bakery::BakerySystem> bakerySystem_{};
  nuri::MaterialRef duckMaterialIndex_ = nuri::kInvalidMaterialRef;
  nuri::MaterialRef helmetFallbackMaterialIndex_ = nuri::kInvalidMaterialRef;
  nuri::MaterialRef lightsPunctualLampMaterial_ = nuri::kInvalidMaterialRef;
  nuri::MaterialRef clearcoatMaterial_ = nuri::kInvalidMaterialRef;
  nuri::MaterialRef sheenChairMaterial_ = nuri::kInvalidMaterialRef;
  nuri::MaterialRef specularSilkPoufMaterial_ = nuri::kInvalidMaterialRef;
  nuri::MaterialRef dragonAttenuationMaterial_ = nuri::kInvalidMaterialRef;
  nuri::MaterialRef dragonIorMaterial_ = nuri::kInvalidMaterialRef;
  nuri::MaterialRef emissiveStrengthTestMaterial_ = nuri::kInvalidMaterialRef;
  nuri::MaterialRef orreyMaterial_ = nuri::kInvalidMaterialRef;
  nuri::MaterialRef bistroMaterialIndex_ = nuri::kInvalidMaterialRef;
  std::vector<nuri::ImportedSceneLight> duckImportedLights_{};
  ImportedPrefabSceneResources helmetPrefabScene_{};
  ImportedPrefabSceneResources niagaraBistroPrefabScene_{};
  ImportedPrefabSceneResources lightsPunctualLampPrefabScene_{};
  ImportedPrefabSceneResources clearcoatPrefabScene_{};
  ImportedPrefabSceneResources sheenChairPrefabScene_{};
  ImportedPrefabSceneResources specularSilkPoufPrefabScene_{};
  ImportedPrefabSceneResources dragonAttenuationPrefabScene_{};
  ImportedPrefabSceneResources dragonIorPrefabScene_{};
  ImportedPrefabSceneResources emissiveStrengthTestPrefabScene_{};
  ImportedPrefabSceneResources orreyPrefabScene_{};
  ImportedPrefabSceneResources foxPrefabScene_{};
  ImportedPrefabSceneResources medievalFantasyBookPrefabScene_{};
  std::optional<nuri::ModelAsyncLoad> bistroAsyncLoad_{};
  std::unique_ptr<nuri::AnimationGpuServices> animationGpuServices_{};
  bool bistroLoadFailed_ = false;
  std::string bistroLoadError_{};
  double bistroLoadStartTimeSeconds_ = 0.0;
  double bistroLastProgressLogTimeSeconds_ = 0.0;
  glm::mat4 niagaraBistroBaseModel_{1.0f};
  bool sceneHasAuthoredLights_ = false;
  nuri::CameraHandle mainCameraHandle_{};
  nuri::RenderableId duckRenderableId_ = nuri::kInvalidRenderableId;
  nuri::RenderableId helmetRenderableId_ = nuri::kInvalidRenderableId;
  nuri::RenderableId lightsPunctualLampRenderableId_ =
      nuri::kInvalidRenderableId;
  nuri::RenderableId clearcoatRenderableId_ = nuri::kInvalidRenderableId;
  nuri::RenderableId sheenChairRenderableId_ = nuri::kInvalidRenderableId;
  nuri::RenderableId specularSilkPoufRenderableId_ = nuri::kInvalidRenderableId;
  nuri::RenderableId dragonAttenuationRenderableId_ =
      nuri::kInvalidRenderableId;
  nuri::RenderableId dragonIorRenderableId_ = nuri::kInvalidRenderableId;
  nuri::RenderableId emissiveStrengthTestRenderableId_ =
      nuri::kInvalidRenderableId;
  nuri::RenderableId orreyRenderableId_ = nuri::kInvalidRenderableId;
  nuri::RenderableId foxRenderableId_ = nuri::kInvalidRenderableId;
  nuri::RenderableId medievalFantasyBookRenderableId_ =
      nuri::kInvalidRenderableId;
  nuri::RenderableId bistroRenderableId_ = nuri::kInvalidRenderableId;
  glm::mat4 duckBaseModel_ = glm::mat4(1.0f);
  glm::mat4 helmetBaseModel_ = glm::mat4(1.0f);
  glm::mat4 lightsPunctualLampBaseModel_ = glm::mat4(1.0f);
  glm::mat4 clearcoatBaseModel_ = glm::mat4(1.0f);
  glm::mat4 orreyBaseModel_ = glm::mat4(1.0f);
  glm::mat4 foxBaseModel_ = glm::mat4(1.0f);
  AnimatedPrefabSceneInstance foxAnimation_;
  AnimatedPrefabSceneInstance medievalFantasyBookAnimation_;

  nuri::RenderSettings renderSettings_{};
  nuri::RenderFrameContext frameContext_{};
  uint64_t frameIndex_ = 0;
  uint64_t simulationFrameIndex_ = 0;
  double frameDeltaSeconds_ = 0.0;
  double fpsAccumulatorSeconds_ = 0.0;
  uint32_t fpsFrameCount_ = 0;
  float currentFps_ = 0.0f;
  std::unique_ptr<nuri::TextSystem> textSystem_{};
  nuri::ScratchArena textScratchArena_{};
  bool text3DEnabled_ = false;
  bool textOverlayEnabled_ = false;
  std::unique_ptr<nuri::EditorOverlayController> editorOverlay_{};
  nuri::EditorOverlayFeature *editorRenderFeature_ = nullptr;
  std::optional<ScenePreset> pendingScenePreset_{};
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
