#include "nuri/pch.h"

#include "nuri/core/application.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/gfx/layers/debug_layer.h"
#include "nuri/gfx/layers/opaque_layer.h"
#include "nuri/gfx/layers/render_frame_context.h"
#include "nuri/gfx/layers/skybox_layer.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/scene/camera_system.h"
#include "nuri/text/text_layer_2d.h"
#include "nuri/text/text_layer_3d.h"
#include "nuri/text/text_system.h"
#include "nuri/scene/render_scene.h"
#include "nuri/ui/camera_controller_widget.h"
#include "nuri/ui/editor_layer.h"

#include <glm/gtc/matrix_transform.hpp>
#include <ctime>

namespace {

enum class ScenePreset : uint8_t {
  SingleDuck,
  InstancedDuck32K,
  BistroExterior,
  DamagedHelmet,
  Text3DTest,
};

constexpr ScenePreset kScenePreset = ScenePreset::SingleDuck;
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
    "piazza_bologni_1k.hdr";
constexpr std::string_view kBistroExteriorModelRelativePath =
    "bistro/exterior/exterior.obj";
constexpr std::string_view kBistroExteriorAbsolutePath =
    "E:/install/nuri/assets/models/bistro/exterior/exterior.obj";
constexpr std::string_view kDamagedHelmetModelRelativePath =
    "DamagedHelmet/DamagedHelmet.gltf";
constexpr std::string_view kDamagedHelmetAlbedoRelativePath =
    "DamagedHelmet/Default_albedo.jpg";
constexpr float kBistroTargetRadius = 120.0f;
constexpr float kBistroMinScale = 0.0005f;
constexpr float kBistroMaxScale = 2.0f;
constexpr const char *kScenePresetNames[] = {
    "Single Duck",      "Instanced Duck 32K", "Bistro Exterior",
    "Damaged Helmet",   "Text 3D Test"};

int scenePresetToIndex(ScenePreset preset) {
  switch (preset) {
  case ScenePreset::SingleDuck:
    return 0;
  case ScenePreset::InstancedDuck32K:
    return 1;
  case ScenePreset::BistroExterior:
    return 2;
  case ScenePreset::DamagedHelmet:
    return 3;
  case ScenePreset::Text3DTest:
    return 4;
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
    return ScenePreset::BistroExterior;
  case 3:
    return ScenePreset::DamagedHelmet;
  case 4:
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

float computeBistroScale(const nuri::BoundingBox &bounds) {
  const float rawRadius =
      std::max(0.5f * glm::length(bounds.getSize()), 1.0e-3f);
  const float targetScale = kBistroTargetRadius / rawRadius;
  return std::clamp(targetScale, kBistroMinScale, kBistroMaxScale);
}

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
    for (const auto &entry : std::filesystem::directory_iterator(fontsRoot, ec)) {
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

class NuriApplication : public nuri::Application {
public:
  explicit NuriApplication(nuri::RuntimeConfig config)
      : nuri::Application(toApplicationConfig(config)),
        config_(std::move(config)), cameraSystem_(cameraMemory_),
        scene_(&sceneMemory_) {}

  void onInit() override {
    NURI_PROFILER_FUNCTION();
    initializeCamera();
    initializeTextSystem();
    loadSceneResources();
    initializeRenderLayers();
    initializeEditorLayer();

    NURI_LOG_INFO("Application was initialized");
  }

  void onDraw() override {
    NURI_PROFILER_FUNCTION();

    const nuri::Camera *activeCamera = cameraSystem_.activeCamera();
    NURI_ASSERT(activeCamera != nullptr, "No active camera");

    applyPendingScenePreset();
    updateBistroSceneStreaming();

    const double timeSeconds = getTime();
    buildFrameContext(*activeCamera, timeSeconds);
    queueTextSamples();
    submitLayeredFrame();
  }

  void onUpdate(double deltaTime) override {
    frameDeltaSeconds_ = (std::isfinite(deltaTime) && deltaTime >= 0.0)
                             ? deltaTime
                             : 0.0;
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
    cameraSystem_.update(deltaTime, getInput());
  }

  void onResize(std::int32_t, std::int32_t) override {}

  bool onInput(const nuri::InputEvent &event) override {
    if (event.type == nuri::InputEventType::Key &&
        event.payload.key.action == nuri::KeyAction::Press &&
        event.payload.key.key == nuri::Key::F6) {
      toggleEditorLayer();
      return true;
    }

    if (cameraSystem_.onInput(event, getWindow())) {
      return true;
    }
    return nuri::Application::onInput(event);
  }

  void onShutdown() override {
    getWindow().setCursorMode(nuri::CursorMode::Normal);
    NURI_LOG_INFO("Application was shutdown");
  }

private:
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

    nuri::syncCameraControllerWidgetStateFromCamera(camera, cameraWidgetState_);
    NURI_LOG_INFO("NuriApplication::initializeCamera: Main camera initialized "
                  "(WASD/QE move, RMB look, P projection toggle, ImGui camera "
                  "controller panel)");
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

    auto debugLayer = nuri::DebugLayer::create(
        getGPU(), config_.shaders.debugGrid, layerMemoryResource());
    NURI_ASSERT(debugLayer != nullptr, "Failed to create debug layer");
    NURI_ASSERT(getLayerStack().pushLayer(std::move(debugLayer)) != nullptr,
                "Failed to push debug layer");

    if (textSystem_) {
      auto textLayer3D = nuri::TextLayer3D::create({
          .text = *textSystem_,
      });
      NURI_ASSERT(textLayer3D != nullptr, "Failed to create 3D text layer");
      textLayer3D_ = static_cast<nuri::TextLayer3D *>(
          getLayerStack().pushLayer(std::move(textLayer3D)));
      NURI_ASSERT(textLayer3D_ != nullptr, "Failed to push 3D text layer");
    }
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
    NURI_ASSERT(!textSystemResult.hasError(), "Failed to create text system: %s",
                textSystemResult.error().c_str());
    textSystem_ = std::move(textSystemResult.value());
    NURI_ASSERT(textSystem_ != nullptr, "Text system was not created");
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

    if (textLayer2D_ != nullptr) {
      (void)enqueue2DTextSamples(defaultFont, baseFontSizePx, scratch);
    }

    if (scenePreset_ == ScenePreset::Text3DTest && textLayer3D_ != nullptr) {
      (void)enqueue3DTextSamples(defaultFont, baseFontSizePx, scratch);
    }
  }

  [[nodiscard]] bool
  enqueue2DTextSamples(nuri::FontHandle defaultFont, float baseFontSizePx,
                       std::pmr::memory_resource &scratch) {
    auto enqueueSample =
        [&](const nuri::Text2DDesc &sample) -> std::optional<nuri::TextBounds> {
      auto enqueue = textSystem_->renderer().enqueue2D(sample, scratch);
      if (enqueue.hasError()) {
        NURI_LOG_WARNING("NuriApplication: failed to enqueue 2D text sample: %s",
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
    const std::optional<nuri::TextBounds> headlineBounds = enqueueSample(headline);
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
    const std::optional<nuri::TextBounds> kerningBounds = enqueueSample(kerning);
    if (!kerningBounds.has_value()) {
      return false;
    }

    nuri::Text2DDesc wrap{};
    wrap.utf8 =
        "Wrap(320px): The quick brown fox jumps over the lazy dog near the river bank.";
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

  [[nodiscard]] bool
  enqueue3DTextSamples(nuri::FontHandle defaultFont, float baseFontSizePx,
                       std::pmr::memory_resource &scratch) {
    auto enqueueSample =
        [&](const nuri::Text3DDesc &sample) -> std::optional<nuri::TextBounds> {
      auto enqueue = textSystem_->renderer().enqueue3D(sample, scratch);
      if (enqueue.hasError()) {
        NURI_LOG_WARNING("NuriApplication: failed to enqueue 3D text sample: %s",
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
    sphericalWorld = glm::translate(sphericalWorld, glm::vec3(0.0f, 2.2f, 0.0f));
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
                                 const nuri::TextColor &color,
                                 float gapAfterPx, bool wrap,
                                 float wrapWidthPx, bool multiline) {
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
                           {.r = 0.86f, .g = 0.92f, .b = 1.0f, .a = 1.0f}, 14.0f,
                           false, 0.0f, false)) {
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

  void initializeTextOverlayLayer() {
    if (textLayer2D_ != nullptr || textSystem_ == nullptr || editorLayer_ != nullptr) {
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

  void removeTextOverlayLayer() {
    if (textLayer2D_ == nullptr) {
      return;
    }
    const bool removed = getLayerStack().popOverlay(textLayer2D_);
    NURI_ASSERT(removed, "Failed to remove 2D text layer");
    textLayer2D_ = nullptr;
  }

  void initializeEditorLayer() {
    if (editorLayer_ != nullptr) {
      return;
    }
    removeTextOverlayLayer();
    const nuri::EditorServices editorServices{
        .scene = &scene_,
        .cameraSystem = &cameraSystem_,
        .gpu = &getGPU(),
        .textSystem = textSystem_.get(),
    };
    auto editorLayer = nuri::EditorLayer::create(
        getWindow(), getGPU(), getEventManager(),
        nuri::EditorLayer::UiCallback([this]() {
          nuri::drawCameraControllerWidget(cameraSystem_, cameraWidgetState_);
          drawScenePresetPanel();
        }),
        editorServices);
    NURI_ASSERT(editorLayer != nullptr, "Failed to create editor layer");
    editorLayer_ = static_cast<nuri::EditorLayer *>(
        getLayerStack().pushOverlay(std::move(editorLayer)));
    NURI_ASSERT(editorLayer_ != nullptr, "Failed to push editor layer");
  }

  void removeEditorLayer() {
    if (editorLayer_ == nullptr) {
      return;
    }

    const bool removed = getLayerStack().popOverlay(editorLayer_);
    NURI_ASSERT(removed, "Failed to remove editor layer");
    editorLayer_ = nullptr;
    initializeTextOverlayLayer();
  }

  void toggleEditorLayer() {
    if (editorLayer_ != nullptr) {
      removeEditorLayer();
      return;
    }
    initializeEditorLayer();
  }

  void loadSceneResources() {
    const std::string duckModelPath =
        (config_.roots.models / kSampleDuckModelRelativePath).string();
    const std::string duckAlbedoPath =
        (config_.roots.models / kSampleDuckAlbedoRelativePath).string();
    const std::string helmetModelPath =
        (config_.roots.models / kDamagedHelmetModelRelativePath).string();
    const std::string helmetAlbedoPath =
        (config_.roots.models / kDamagedHelmetAlbedoRelativePath).string();
    const std::string environmentHdrPath =
        (config_.roots.textures / kSampleEnvironmentHdrRelativePath).string();

    std::pmr::unsynchronized_pool_resource meshImportMemory;
    auto modelResult = nuri::Model::createFromFile(
        getGPU(), duckModelPath, {}, &meshImportMemory, "rubber_duck");
    NURI_ASSERT(!modelResult.hasError(), "Failed to create model: %s",
                modelResult.error().c_str());

    auto albedoResult =
        nuri::Texture::loadTexture(getGPU(), duckAlbedoPath, "duck_albedo");
    NURI_ASSERT(!albedoResult.hasError(), "Failed to load albedo texture: %s",
                albedoResult.error().c_str());

    auto helmetModelResult = nuri::Model::createFromFile(
        getGPU(), helmetModelPath, {}, &meshImportMemory, "damaged_helmet");
    if (helmetModelResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "DamagedHelmet model '%s': %s",
                       helmetModelPath.c_str(),
                       helmetModelResult.error().c_str());
    }
    NURI_ASSERT(!helmetModelResult.hasError(),
                "Failed to create DamagedHelmet model: %s",
                helmetModelResult.error().c_str());

    auto helmetAlbedoResult = nuri::Texture::loadTexture(
        getGPU(), helmetAlbedoPath, "damaged_helmet_albedo");
    if (helmetAlbedoResult.hasError()) {
      NURI_LOG_WARNING("NuriApplication::loadSceneResources: Failed to load "
                       "DamagedHelmet albedo '%s': %s",
                       helmetAlbedoPath.c_str(),
                       helmetAlbedoResult.error().c_str());
    }
    NURI_ASSERT(!helmetAlbedoResult.hasError(),
                "Failed to load DamagedHelmet albedo: %s",
                helmetAlbedoResult.error().c_str());

    auto cubemapResult = nuri::Texture::loadCubemapFromEquirectangularHDR(
        getGPU(), environmentHdrPath, "duck_cubemap");
    NURI_ASSERT(!cubemapResult.hasError(),
                "Failed to create cubemap texture: %s",
                cubemapResult.error().c_str());

    const std::array<std::byte, 4> whitePixel = {
        std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff}};
    const nuri::TextureDesc whiteTextureDesc{
        .type = nuri::TextureType::Texture2D,
        .format = nuri::Format::RGBA8_UNORM,
        .dimensions = {1, 1, 1},
        .usage = nuri::TextureUsage::Sampled,
        .storage = nuri::Storage::Device,
        .numLayers = 1,
        .numSamples = 1,
        .numMipLevels = 1,
        .data =
            std::span<const std::byte>(whitePixel.data(), whitePixel.size()),
        .dataNumMipLevels = 1,
        .generateMipmaps = false,
    };
    auto whiteAlbedoResult = nuri::Texture::create(getGPU(), whiteTextureDesc,
                                                   "bistro_white_albedo");
    NURI_ASSERT(!whiteAlbedoResult.hasError(),
                "Failed to create fallback white albedo texture: %s",
                whiteAlbedoResult.error().c_str());

    scene_.setEnvironmentCubemap(std::move(cubemapResult.value()));
    duckModel_ = std::shared_ptr<nuri::Model>(std::move(modelResult.value()));
    duckAlbedo_ =
        std::shared_ptr<nuri::Texture>(std::move(albedoResult.value()));
    helmetModel_ =
        std::shared_ptr<nuri::Model>(std::move(helmetModelResult.value()));
    helmetAlbedo_ =
        std::shared_ptr<nuri::Texture>(std::move(helmetAlbedoResult.value()));
    bistroAlbedoFallback_ =
        std::shared_ptr<nuri::Texture>(std::move(whiteAlbedoResult.value()));
    NURI_LOG_INFO(
        "NuriApplication::loadSceneResources: DamagedHelmet loaded "
        "(submeshes=%zu vertices=%u indices=%u)",
        helmetModel_->submeshes().size(), helmetModel_->vertexCount(),
        helmetModel_->indexCount());

    applyScenePreset(scenePreset_);
  }

  void loadSingleDuckSceneResources() {
    renderSettings_.opaque.enableInstanceCompute = true;
    renderSettings_.opaque.enableMeshLod = true;
    renderSettings_.opaque.forcedMeshLod = -1;
    renderSettings_.opaque.meshLodDistanceThresholds =
        glm::vec3(8.0f, 16.0f, 32.0f);
    renderSettings_.opaque.enableInstanceAnimation = false;

    auto addResult =
        scene_.addOpaqueRenderable(duckModel_, duckAlbedo_, duckBaseModel_);
    NURI_ASSERT(!addResult.hasError(), "Failed to add duck renderable: %s",
                addResult.error().c_str());
    duckRenderableIndex_ = addResult.value();

    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    NURI_ASSERT(camera != nullptr, "Failed to get main camera");
    camera->setLookAt(glm::vec3(0.0f, 1.0f, -1.5f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f));
    nuri::syncCameraControllerWidgetStateFromCamera(*camera,
                                                    cameraWidgetState_);
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

    auto addResult = scene_.addOpaqueRenderablesInstanced(
        duckModel_, duckAlbedo_, transforms);
    NURI_ASSERT(!addResult.hasError(),
                "Failed to add instanced duck renderables: %s",
                addResult.error().c_str());

    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    NURI_ASSERT(camera != nullptr, "Failed to get main camera");
    camera->setLookAt(glm::vec3(0.0f, 120.0f, -760.0f), glm::vec3(0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f));
    nuri::syncCameraControllerWidgetStateFromCamera(*camera,
                                                    cameraWidgetState_);

    NURI_LOG_INFO(
        "NuriApplication::setupInstancedDuckScene32k: spawned %u ducks in a "
        "32x32x32 grid",
        kDuckInstanceCount);
  }

  std::filesystem::path resolveBistroExteriorPath() const {
    std::filesystem::path preferredPath =
        config_.roots.models / kBistroExteriorModelRelativePath;
    std::error_code ec;
    if (std::filesystem::exists(preferredPath, ec) &&
        std::filesystem::is_regular_file(preferredPath, ec)) {
      return preferredPath;
    }

    std::filesystem::path fallbackPath{kBistroExteriorAbsolutePath};
    ec.clear();
    if (std::filesystem::exists(fallbackPath, ec) &&
        std::filesystem::is_regular_file(fallbackPath, ec)) {
      return fallbackPath;
    }
    return preferredPath;
  }

  void beginBistroModelLoad() {
    const bool hasAsyncLoadInFlight = bistroAsyncLoad_ &&
                                      bistroAsyncLoad_->valid() &&
                                      !bistroAsyncLoad_->isFinalized();
    if (bistroModel_ != nullptr || hasAsyncLoadInFlight || bistroLoadFailed_) {
      return;
    }
    bistroLoadFailed_ = false;
    bistroLoadError_.clear();
    bistroLoadStartTimeSeconds_ = getTime();
    bistroLastProgressLogTimeSeconds_ = bistroLoadStartTimeSeconds_;

    const std::string path = resolveBistroExteriorPath().string();

    auto asyncLoadResult =
        nuri::Model::createFromFileAsync(path, nuri::MeshImportOptions{});
    if (asyncLoadResult.hasError()) {
      bistroLoadFailed_ = true;
      bistroLoadError_ = asyncLoadResult.error();
      NURI_LOG_WARNING("NuriApplication: Bistro async load start failed: %s",
                       bistroLoadError_.c_str());
      return;
    }

    bistroAsyncLoad_ = std::move(asyncLoadResult.value());
    NURI_LOG_INFO("NuriApplication: started Bistro async model load from '%s'",
                  path.c_str());
  }

  void updateBistroSceneStreaming() {
    if (!bistroAsyncLoad_ || !bistroAsyncLoad_->valid()) {
      if (scenePreset_ == ScenePreset::BistroExterior &&
          bistroModel_ != nullptr &&
          bistroRenderableIndex_ == std::numeric_limits<uint32_t>::max()) {
        setupBistroExteriorScene();
      }
      return;
    }

    if (!bistroAsyncLoad_->isReady()) {
      const double now = getTime();
      if (now - bistroLastProgressLogTimeSeconds_ >= 5.0) {
        bistroLastProgressLogTimeSeconds_ = now;
        NURI_LOG_INFO("NuriApplication: Bistro load in progress (%.1f s)",
                      now - bistroLoadStartTimeSeconds_);
      }
      return;
    }

    auto modelResult = bistroAsyncLoad_->finalize(
        getGPU(), std::pmr::get_default_resource(), "bistro_exterior");
    const std::optional<bool> warmupCacheHit = bistroAsyncLoad_->cacheHit();
    const std::string warmupError =
        std::string(bistroAsyncLoad_->warmupError());
    const double totalLoadSeconds = getTime() - bistroLoadStartTimeSeconds_;
    if (modelResult.hasError()) {
      bistroLoadFailed_ = true;
      bistroLoadError_ = modelResult.error();
      if (!warmupError.empty()) {
        NURI_LOG_WARNING("NuriApplication: Bistro async warmup failed: %s",
                         warmupError.c_str());
      }
      NURI_LOG_WARNING("NuriApplication: Bistro GPU model creation failed: %s",
                       bistroLoadError_.c_str());
      bistroAsyncLoad_.reset();
      return;
    }

    if (warmupCacheHit.has_value()) {
      NURI_LOG_INFO("NuriApplication: Bistro cache %s in %.1f s",
                    warmupCacheHit.value() ? "hit" : "rebuilt",
                    totalLoadSeconds);
    } else if (!warmupError.empty()) {
      NURI_LOG_WARNING("NuriApplication: Bistro async warmup failed: %s "
                       "(load completed via direct path)",
                       warmupError.c_str());
    }
    bistroAsyncLoad_.reset();

    // Ensure geometry uploads are fully visible before first frame that draws
    // the freshly created model.
    getGPU().waitIdle();

    bistroModel_ = std::shared_ptr<nuri::Model>(std::move(modelResult.value()));
    NURI_LOG_INFO("NuriApplication: Bistro model is ready in %.1f s",
                  getTime() - bistroLoadStartTimeSeconds_);

    if (scenePreset_ == ScenePreset::BistroExterior &&
        bistroRenderableIndex_ == std::numeric_limits<uint32_t>::max()) {
      setupBistroExteriorScene();
    }
  }

  void setupBistroExteriorScene() {
    if (bistroModel_ == nullptr) {
      if (bistroLoadFailed_) {
        return;
      }
      beginBistroModelLoad();
      return;
    }
    if (bistroRenderableIndex_ != std::numeric_limits<uint32_t>::max()) {
      return;
    }
    NURI_ASSERT(bistroAlbedoFallback_ != nullptr,
                "Bistro fallback albedo is not loaded");

    // Keep tessellation off for stable performance; use generated mesh LODs.
    renderSettings_.opaque.enableInstanceCompute = false;
    renderSettings_.opaque.enableMeshLod = true;
    renderSettings_.opaque.enableTessellation = false;
    renderSettings_.opaque.forcedMeshLod = -1;
    renderSettings_.opaque.meshLodDistanceThresholds =
        glm::vec3(8.0f, 24.0f, 48.0f);
    renderSettings_.opaque.enableInstanceAnimation = false;
    const nuri::BoundingBox &bounds = bistroModel_->bounds();
    const float bistroScale = computeBistroScale(bounds);
    const glm::mat4 bistroModelMatrix =
        glm::scale(glm::mat4(1.0f), glm::vec3(bistroScale));

    auto addResult = scene_.addOpaqueRenderable(
        bistroModel_, bistroAlbedoFallback_, bistroModelMatrix);
    NURI_ASSERT(!addResult.hasError(), "Failed to add Bistro renderable: %s",
                addResult.error().c_str());
    bistroRenderableIndex_ = addResult.value();

    const float rawRadius =
        std::max(0.5f * glm::length(bounds.getSize()), 1.0f);
    const glm::vec3 center = bounds.getCenter() * bistroScale;
    const float radius = std::max(1.0f, rawRadius * bistroScale);
    const float cameraDistance = std::max(radius * 1.2f, 25.0f);

    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    NURI_ASSERT(camera != nullptr, "Failed to get main camera");
    nuri::PerspectiveParams perspective = camera->perspective();
    perspective.nearPlane = std::max(0.05f, cameraDistance / 5000.0f);
    perspective.farPlane = std::max(2000.0f, cameraDistance + radius * 3.0f);
    camera->setProjectionType(nuri::ProjectionType::Perspective);
    camera->setPerspective(perspective);
    camera->setLookAt(center + glm::vec3(-cameraDistance * 0.32f,
                                         radius * 0.14f + 4.0f,
                                         -cameraDistance),
                      center + glm::vec3(0.0f, radius * 0.03f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f));
    nuri::syncCameraControllerWidgetStateFromCamera(*camera,
                                                    cameraWidgetState_);
    NURI_LOG_INFO("NuriApplication: Bistro scene stats submeshes=%zu "
                  "vertices=%u indices=%u rawRadius=%.2f scale=%.6f "
                  "radius=%.2f near=%.3f far=%.2f",
                  bistroModel_->submeshes().size(), bistroModel_->vertexCount(),
                  bistroModel_->indexCount(), rawRadius, bistroScale, radius,
                  perspective.nearPlane, perspective.farPlane);
  }

  void setupText3DTestScene() {
    renderSettings_.opaque.enableInstanceCompute = false;
    renderSettings_.opaque.enableMeshLod = true;
    renderSettings_.opaque.enableTessellation = false;
    renderSettings_.opaque.forcedMeshLod = -1;
    renderSettings_.opaque.enableInstanceAnimation = false;

    nuri::Camera *camera = cameraSystem_.camera(mainCameraHandle_);
    NURI_ASSERT(camera != nullptr, "Failed to get main camera");
    nuri::PerspectiveParams perspective = camera->perspective();
    perspective.nearPlane = 0.01f;
    perspective.farPlane = 500.0f;
    camera->setProjectionType(nuri::ProjectionType::Perspective);
    camera->setPerspective(perspective);
    camera->setLookAt(glm::vec3(0.0f, 1.2f, -4.2f), glm::vec3(0.0f, 1.2f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f));
    nuri::syncCameraControllerWidgetStateFromCamera(*camera,
                                                    cameraWidgetState_);
  }

  void setupDamagedHelmetScene() {
    NURI_ASSERT(helmetModel_ != nullptr, "DamagedHelmet model is not loaded");
    NURI_ASSERT(helmetAlbedo_ != nullptr,
                "DamagedHelmet albedo texture is not loaded");
    if (helmetRenderableIndex_ != std::numeric_limits<uint32_t>::max()) {
      return;
    }

    renderSettings_.opaque.enableInstanceCompute = false;
    renderSettings_.opaque.enableMeshLod = true;
    renderSettings_.opaque.enableTessellation = false;
    renderSettings_.opaque.forcedMeshLod = -1;
    renderSettings_.opaque.meshLodDistanceThresholds =
        glm::vec3(8.0f, 16.0f, 32.0f);
    renderSettings_.opaque.enableInstanceAnimation = false;

    const nuri::BoundingBox &bounds = helmetModel_->bounds();
    auto addResult = scene_.addOpaqueRenderable(
        helmetModel_, helmetAlbedo_, helmetBaseModel_);
    NURI_ASSERT(!addResult.hasError(),
                "Failed to add DamagedHelmet renderable: %s",
                addResult.error().c_str());
    helmetRenderableIndex_ = addResult.value();

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
    nuri::syncCameraControllerWidgetStateFromCamera(*camera,
                                                    cameraWidgetState_);
    NURI_LOG_INFO("NuriApplication: DamagedHelmet scene stats submeshes=%zu "
                  "vertices=%u indices=%u rawRadius=%.2f radius=%.2f "
                  "near=%.3f far=%.2f",
                  helmetModel_->submeshes().size(), helmetModel_->vertexCount(),
                  helmetModel_->indexCount(), rawRadius, radius,
                  perspective.nearPlane, perspective.farPlane);
  }

  void applyScenePreset(ScenePreset preset) {
    NURI_ASSERT(duckModel_ != nullptr, "Duck model is not loaded");
    NURI_ASSERT(duckAlbedo_ != nullptr, "Duck albedo texture is not loaded");

    scene_.clearOpaqueRenderables();
    if (editorLayer_ != nullptr) {
      editorLayer_->resetControllers();
    }
    duckRenderableIndex_ = std::numeric_limits<uint32_t>::max();
    bistroRenderableIndex_ = std::numeric_limits<uint32_t>::max();
    helmetRenderableIndex_ = std::numeric_limits<uint32_t>::max();
    scenePreset_ = preset;

    if (scenePreset_ == ScenePreset::InstancedDuck32K) {
      setupInstancedDuckScene32k();
    } else if (scenePreset_ == ScenePreset::BistroExterior) {
      bistroLoadFailed_ = false;
      bistroLoadError_.clear();
      setupBistroExteriorScene();
    } else if (scenePreset_ == ScenePreset::DamagedHelmet) {
      setupDamagedHelmetScene();
    } else if (scenePreset_ == ScenePreset::Text3DTest) {
      setupText3DTestScene();
    } else {
      loadSingleDuckSceneResources();
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

  void drawScenePresetPanel() {
    int presetIndex = scenePresetToIndex(scenePreset_);
    const std::span<const char *const> presetNames{
        kScenePresetNames, static_cast<size_t>(sizeof(kScenePresetNames) /
                                               sizeof(kScenePresetNames[0]))};
    if (nuri::drawScenePresetWidget(presetNames, presetIndex,
                                    "Toggle Editor: F6")) {
      const ScenePreset selectedPreset = scenePresetFromIndex(presetIndex);
      if (selectedPreset != scenePreset_) {
        requestScenePreset(selectedPreset);
      }
    }
  }

  void buildFrameContext(const nuri::Camera &camera, double timeSeconds) {
    frameContext_.scene = &scene_;
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

  void submitLayeredFrame() {
    const nuri::RenderFrame frame{};
    auto renderResult =
        getRenderer().render(frame, getLayerStack(), frameContext_);
    NURI_ASSERT(!renderResult.hasError(), "Render failed: %s",
                renderResult.error().c_str());
  }

  const nuri::RuntimeConfig config_;
  ScenePreset scenePreset_ = kScenePreset;
  std::pmr::unsynchronized_pool_resource cameraMemory_;
  std::pmr::unsynchronized_pool_resource sceneMemory_;
  nuri::CameraSystem cameraSystem_;
  nuri::RenderScene scene_;
  std::shared_ptr<nuri::Model> duckModel_{};
  std::shared_ptr<nuri::Texture> duckAlbedo_{};
  std::shared_ptr<nuri::Model> helmetModel_{};
  std::shared_ptr<nuri::Texture> helmetAlbedo_{};
  std::shared_ptr<nuri::Model> bistroModel_{};
  std::shared_ptr<nuri::Texture> bistroAlbedoFallback_{};
  std::optional<nuri::ModelAsyncLoad> bistroAsyncLoad_{};
  bool bistroLoadFailed_ = false;
  std::string bistroLoadError_{};
  double bistroLoadStartTimeSeconds_ = 0.0;
  double bistroLastProgressLogTimeSeconds_ = 0.0;
  nuri::CameraHandle mainCameraHandle_{};
  uint32_t duckRenderableIndex_ = std::numeric_limits<uint32_t>::max();
  uint32_t helmetRenderableIndex_ = std::numeric_limits<uint32_t>::max();
  uint32_t bistroRenderableIndex_ = std::numeric_limits<uint32_t>::max();
  glm::mat4 duckBaseModel_ =
      glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1, 0, 0));
  glm::mat4 helmetBaseModel_ =
      glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));

  nuri::RenderSettings renderSettings_{};
  nuri::RenderFrameContext frameContext_{};
  uint64_t frameIndex_ = 0;
  double frameDeltaSeconds_ = 0.0;
  double fpsAccumulatorSeconds_ = 0.0;
  uint32_t fpsFrameCount_ = 0;
  float currentFps_ = 0.0f;
  std::unique_ptr<nuri::TextSystem> textSystem_{};
  nuri::ScratchArena textScratchArena_{};
  nuri::CameraControllerWidgetState cameraWidgetState_{};
  nuri::TextLayer3D *textLayer3D_ = nullptr;
  nuri::TextLayer2D *textLayer2D_ = nullptr;
  nuri::EditorLayer *editorLayer_ = nullptr;
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
