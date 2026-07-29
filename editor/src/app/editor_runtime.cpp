#include "nuri/editor_pch.h"

#include "nuri/app/editor_runtime.h"

#include "nuri/app/editor_scene_catalog.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/sim/animation_gpu_services.h"
#include "nuri/gfx/sim/animation_scene_frame_provider.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/utils/env_utils.h"

#include <cctype>
#include <charconv>
#include <ctime>
#include <optional>

namespace nuri {
namespace {

constexpr std::string_view kSampleEnvironmentHdrRelativePath =
    "qwantani_moon_noon_puresky_4k.hdr";

[[nodiscard]] glm::mat4 makeTransformMatrix(const glm::vec3 &position,
                                            const glm::quat &rotation) {
  return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotation);
}

[[nodiscard]] std::vector<glm::mat4>
buildPrefabWorldMatrices(const ScenePrefab &prefab,
                         const glm::mat4 &baseModel) {
  std::vector<glm::mat4> worldMatrices(prefab.nodes.size(), glm::mat4(1.0f));
  for (size_t nodeIndex = 0; nodeIndex < prefab.nodes.size(); ++nodeIndex) {
    const ScenePrefabNode &node = prefab.nodes[nodeIndex];
    if (node.parentIndex == kInvalidScenePrefabIndex ||
        node.parentIndex >= worldMatrices.size()) {
      worldMatrices[nodeIndex] = baseModel * node.localFromParent;
      continue;
    }
    worldMatrices[nodeIndex] =
        worldMatrices[node.parentIndex] * node.localFromParent;
  }
  return worldMatrices;
}

std::filesystem::path pickDefaultNfontPath(const RuntimeConfig &config) {
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

void logDebugRenderOverrideOnce(const char *envName, const char *effect,
                                bool &logged) {
  if (!logged) {
    logged = true;
    NURI_LOG_WARNING("EditorRuntime: %s active; %s", envName, effect);
  }
}

[[nodiscard]] bool stringEqualsIgnoreCase(std::string_view lhs,
                                          std::string_view rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    const char left =
        static_cast<char>(std::tolower(static_cast<unsigned char>(lhs[i])));
    const char right =
        static_cast<char>(std::tolower(static_cast<unsigned char>(rhs[i])));
    if (left != right) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::string_view trimAsciiWhitespace(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

[[nodiscard]] std::optional<uint32_t> parseUint32(std::string_view value) {
  value = trimAsciiWhitespace(value);
  if (value.empty()) {
    return std::nullopt;
  }
  uint32_t parsed = 0;
  const char *begin = value.data();
  const char *end = value.data() + value.size();
  const std::from_chars_result result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end) {
    return std::nullopt;
  }
  return parsed;
}

struct DebugShadowInspectProbeConfig {
  bool enabled = false;
  bool explicitPixel = false;
  uint32_t x = 0;
  uint32_t y = 0;
  uint64_t warmupFrames = 120;
  uint64_t timeoutFrames = 240;
};

[[nodiscard]] DebugShadowInspectProbeConfig
readDebugShadowInspectProbeConfig() {
  const std::optional<std::string> value =
      readEnvVar("NURI_DEBUG_SHADOW_INSPECT");
  if (!value.has_value()) {
    return {};
  }

  const std::string_view view = trimAsciiWhitespace(*value);
  if (view.empty() || stringEqualsIgnoreCase(view, "0") ||
      stringEqualsIgnoreCase(view, "false") ||
      stringEqualsIgnoreCase(view, "off") ||
      stringEqualsIgnoreCase(view, "no")) {
    return {};
  }
  if (stringEqualsIgnoreCase(view, "1") ||
      stringEqualsIgnoreCase(view, "true") ||
      stringEqualsIgnoreCase(view, "on") ||
      stringEqualsIgnoreCase(view, "yes")) {
    return {.enabled = true};
  }

  const size_t comma = view.find(',');
  if (comma == std::string_view::npos) {
    NURI_LOG_WARNING(
        "EditorRuntime: ignoring unrecognized NURI_DEBUG_SHADOW_INSPECT=%s",
        value->c_str());
    return {};
  }
  const std::optional<uint32_t> x = parseUint32(view.substr(0, comma));
  const std::optional<uint32_t> y = parseUint32(view.substr(comma + 1));
  if (!x.has_value() || !y.has_value()) {
    NURI_LOG_WARNING(
        "EditorRuntime: ignoring unrecognized NURI_DEBUG_SHADOW_INSPECT=%s",
        value->c_str());
    return {};
  }
  return {.enabled = true, .explicitPixel = true, .x = *x, .y = *y};
}

[[nodiscard]] const DebugShadowInspectProbeConfig &
debugShadowInspectProbeConfig() {
  static const DebugShadowInspectProbeConfig config =
      readDebugShadowInspectProbeConfig();
  return config;
}

[[nodiscard]] std::optional<AntiAliasingMode> readDebugAntiAliasingMode() {
  const std::optional<std::string> value = readEnvVar("NURI_DEBUG_AA_MODE");
  if (!value.has_value()) {
    return std::nullopt;
  }
  const std::string_view view = *value;
  if (stringEqualsIgnoreCase(view, "none")) {
    return AntiAliasingMode::None;
  }
  if (stringEqualsIgnoreCase(view, "taa")) {
    return AntiAliasingMode::TAA;
  }
  if (stringEqualsIgnoreCase(view, "spatial") ||
      stringEqualsIgnoreCase(view, "spatial_fallback")) {
    return AntiAliasingMode::SpatialFallback;
  }
  if (stringEqualsIgnoreCase(view, "msaa") ||
      stringEqualsIgnoreCase(view, "msaa4x")) {
    return AntiAliasingMode::MSAA4x;
  }
  if (stringEqualsIgnoreCase(view, "msaa8x")) {
    return AntiAliasingMode::MSAA8x;
  }
  NURI_LOG_WARNING("EditorRuntime: ignoring unrecognized NURI_DEBUG_AA_MODE=%s",
                   value->c_str());
  return std::nullopt;
}

[[nodiscard]] std::optional<TemporalAAQualityPreset>
readDebugTemporalAAQualityPreset() {
  const std::optional<std::string> value = readEnvVar("NURI_DEBUG_TAA_PRESET");
  if (!value.has_value()) {
    return std::nullopt;
  }
  const std::string_view view = trimAsciiWhitespace(*value);
  if (stringEqualsIgnoreCase(view, "performance") ||
      stringEqualsIgnoreCase(view, "perf")) {
    return TemporalAAQualityPreset::Performance;
  }
  if (stringEqualsIgnoreCase(view, "balanced")) {
    return TemporalAAQualityPreset::Balanced;
  }
  if (stringEqualsIgnoreCase(view, "quality")) {
    return TemporalAAQualityPreset::Quality;
  }
  if (stringEqualsIgnoreCase(view, "ultra")) {
    return TemporalAAQualityPreset::Ultra;
  }
  if (stringEqualsIgnoreCase(view, "custom")) {
    return TemporalAAQualityPreset::Custom;
  }
  NURI_LOG_WARNING(
      "EditorRuntime: ignoring unrecognized NURI_DEBUG_TAA_PRESET=%s",
      value->c_str());
  return std::nullopt;
}

struct DebugRenderEnvOverrides {
  bool disableOpaque = false;
  bool disableTransmission = false;
  bool disableTransparent = false;
  bool disableSkybox = false;
  std::optional<AntiAliasingMode> antiAliasingMode{};
  std::optional<TemporalAAQualityPreset> temporalAAQualityPreset{};
  std::optional<bool> temporalAAJitterEnabled{};
  std::optional<bool> temporalAADiagnostics{};
  bool shadowDiagnostics = false;
};

[[nodiscard]] const DebugRenderEnvOverrides &debugRenderEnvOverrides() {
  static const DebugRenderEnvOverrides overrides{
      .disableOpaque = readEnvFlag("NURI_DEBUG_DISABLE_OPAQUE"),
      .disableTransmission = readEnvFlag("NURI_DEBUG_DISABLE_TRANSMISSION"),
      .disableTransparent = readEnvFlag("NURI_DEBUG_DISABLE_TRANSPARENT"),
      .disableSkybox = readEnvFlag("NURI_DEBUG_DISABLE_SKYBOX"),
      .antiAliasingMode = readDebugAntiAliasingMode(),
      .temporalAAQualityPreset = readDebugTemporalAAQualityPreset(),
      .temporalAAJitterEnabled = readEnvBoolOverride("NURI_DEBUG_TAA_JITTER"),
      .temporalAADiagnostics =
          readEnvBoolOverride("NURI_DEBUG_TAA_DIAGNOSTICS"),
      .shadowDiagnostics = readEnvFlag("NURI_DEBUG_SHADOW_DIAGNOSTICS"),
  };
  return overrides;
}

void applyDebugRenderEnvOverrides(RenderSettings &settings) {
  const DebugRenderEnvOverrides &overrides = debugRenderEnvOverrides();
  if (overrides.disableOpaque) {
    settings.opaque.enabled = false;
    static bool logged = false;
    logDebugRenderOverrideOnce("NURI_DEBUG_DISABLE_OPAQUE",
                               "opaque pass disabled", logged);
  }
  if (overrides.disableTransmission) {
    settings.transmission.enabled = false;
    static bool logged = false;
    logDebugRenderOverrideOnce("NURI_DEBUG_DISABLE_TRANSMISSION",
                               "transmission pass disabled", logged);
  }
  if (overrides.disableTransparent) {
    settings.transparent.enabled = false;
    static bool logged = false;
    logDebugRenderOverrideOnce("NURI_DEBUG_DISABLE_TRANSPARENT",
                               "transparent pass disabled", logged);
  }
  if (overrides.disableSkybox) {
    settings.skybox.enabled = false;
    static bool logged = false;
    logDebugRenderOverrideOnce("NURI_DEBUG_DISABLE_SKYBOX",
                               "skybox pass disabled", logged);
  }
  if (overrides.antiAliasingMode.has_value()) {
    settings.antiAliasing.mode = *overrides.antiAliasingMode;
    static bool logged = false;
    logDebugRenderOverrideOnce("NURI_DEBUG_AA_MODE",
                               "anti-aliasing mode overridden", logged);
  }
  if (overrides.temporalAAQualityPreset.has_value()) {
    settings.antiAliasing.qualityPreset = *overrides.temporalAAQualityPreset;
    static bool logged = false;
    logDebugRenderOverrideOnce("NURI_DEBUG_TAA_PRESET",
                               "TAA quality preset overridden", logged);
  }
  if (overrides.temporalAAJitterEnabled.has_value()) {
    settings.antiAliasing.debug.jitterEnabled =
        *overrides.temporalAAJitterEnabled;
    static bool logged = false;
    logDebugRenderOverrideOnce("NURI_DEBUG_TAA_JITTER", "TAA jitter overridden",
                               logged);
  }
  if (overrides.temporalAADiagnostics.has_value()) {
    settings.antiAliasing.debug.logDiagnostics =
        *overrides.temporalAADiagnostics;
    static bool logged = false;
    logDebugRenderOverrideOnce("NURI_DEBUG_TAA_DIAGNOSTICS",
                               "TAA diagnostics overridden", logged);
  }
  if (overrides.shadowDiagnostics) {
    settings.shadow.debug.logDiagnostics = true;
    settings.shadow.debug.diagnosticLogLevel = LogLevel::Info;
    settings.shadow.debug.diagnosticLogIntervalFrames = 15u;
    static bool logged = false;
    logDebugRenderOverrideOnce("NURI_DEBUG_SHADOW_DIAGNOSTICS",
                               "shadow diagnostics enabled", logged);
  }
}

void persistFrameRenderSettings(RenderSettings &persistent,
                                const RenderSettings &frameSettings) {
  const bool opaqueEnabled = persistent.opaque.enabled;
  const bool transmissionEnabled = persistent.transmission.enabled;
  const bool transparentEnabled = persistent.transparent.enabled;
  const bool skyboxEnabled = persistent.skybox.enabled;

  persistent = frameSettings;

  const DebugRenderEnvOverrides &overrides = debugRenderEnvOverrides();
  if (overrides.disableOpaque) {
    persistent.opaque.enabled = opaqueEnabled;
  }
  if (overrides.disableTransmission) {
    persistent.transmission.enabled = transmissionEnabled;
  }
  if (overrides.disableTransparent) {
    persistent.transparent.enabled = transparentEnabled;
  }
  if (overrides.disableSkybox) {
    persistent.skybox.enabled = skyboxEnabled;
  }
}

EnvironmentAssetHandle loadSharedEnvironment(EditorRuntime &runtime) {
  const RuntimeConfig &config = runtime.config();
  const std::string environmentHdrPath =
      (config.roots.textures / kSampleEnvironmentHdrRelativePath).string();

  const std::filesystem::path environmentHdrFile{environmentHdrPath};
  const std::string environmentStem = environmentHdrFile.stem().string();
  const std::array<std::filesystem::path, 2> irradianceCandidates = {
      config.roots.textures / (environmentStem + "_irradiance.ktx2"),
      config.roots.textures / (environmentStem + "_irradiance.ktx"),
  };
  const std::array<std::filesystem::path, 2> prefilteredGgxCandidates = {
      config.roots.textures / (environmentStem + "_prefilter_ggx.ktx2"),
      config.roots.textures / (environmentStem + "_prefilter_ggx.ktx"),
  };
  // prefilteredCharlieCandidates intentionally includes legacy
  // "_prefilter_charile" filenames under environmentStem in
  // config.roots.textures.
  const std::array<std::filesystem::path, 4> prefilteredCharlieCandidates = {
      config.roots.textures / (environmentStem + "_prefilter_charlie.ktx2"),
      config.roots.textures / (environmentStem + "_prefilter_charlie.ktx"),
      config.roots.textures / (environmentStem + "_prefilter_charile.ktx"),
      config.roots.textures / (environmentStem + "_prefilter_charile.ktx2"),
  };
  const std::array<std::filesystem::path, 2> brdfLutCandidates = {
      config.roots.textures / "brdf_lut.ktx2",
      config.roots.textures / "brdf_lut.ktx",
  };

  const auto resolveFirstExistingIblAssetPath =
      [](std::span<const std::filesystem::path> candidates)
      -> std::filesystem::path {
    std::error_code ec;
    for (const auto &candidate : candidates) {
      if (std::filesystem::exists(candidate, ec) &&
          std::filesystem::is_regular_file(candidate, ec)) {
        return candidate;
      }
      ec.clear();
    }
    return candidates.empty() ? std::filesystem::path{} : candidates.front();
  };

  const auto optionalKtxRequest =
      [&resolveFirstExistingIblAssetPath](
          std::span<const std::filesystem::path> candidates,
          TextureRequestKind kind,
          std::string_view debugName) -> std::optional<TextureRequest> {
    const std::filesystem::path resolvedPath =
        resolveFirstExistingIblAssetPath(candidates);
    std::error_code ec;
    if (resolvedPath.empty() || !std::filesystem::exists(resolvedPath, ec) ||
        !std::filesystem::is_regular_file(resolvedPath, ec)) {
      return std::nullopt;
    }
    return TextureRequest{
        .path = resolvedPath.string(),
        .kind = kind,
        .debugName = std::string(debugName),
    };
  };

  auto requested = runtime.assets().requestEnvironment(EnvironmentAssetRequest{
      .textures =
          {
              TextureRequest{
                  .path = environmentHdrPath,
                  .loadOptions = TextureLoadOptions{},
                  .kind = TextureRequestKind::EquirectHdrCubemap,
                  .debugName = "editor_cubemap",
              },
              optionalKtxRequest(irradianceCandidates,
                                 TextureRequestKind::Ktx2Cubemap,
                                 "ibl_irradiance"),
              optionalKtxRequest(prefilteredGgxCandidates,
                                 TextureRequestKind::Ktx2Cubemap,
                                 "ibl_prefilter_ggx"),
              optionalKtxRequest(prefilteredCharlieCandidates,
                                 TextureRequestKind::Ktx2Cubemap,
                                 "ibl_prefilter_charlie"),
              optionalKtxRequest(brdfLutCandidates,
                                 TextureRequestKind::Ktx2Texture2D,
                                 "ibl_brdf_lut"),
          },
      .priority = AssetPriority::Visible,
      .debugName = "editor_shared_environment",
  });
  if (requested.hasError()) {
    NURI_LOG_ERROR("EditorRuntime: failed to request shared environment: %s",
                   requested.error().c_str());
    return {};
  }
  return requested.value();
}

} // namespace

struct EditorRuntime::EditorSceneDocument {
  struct PendingAnimationActivation {
    std::string sceneName{};
    const ImportedPrefabSceneResources *resources = nullptr;
    AnimatedPrefabSceneState *instance = nullptr;
    AnimationPoseSimulationParams params{};
    std::string simulationDebugName{};
  };

  std::pmr::unsynchronized_pool_resource memory{};
  RenderScene scene{&memory};
  SceneRuntimeHost sceneRuntime;
  RenderSettings renderSettings{};
  Camera camera{};
  ScenePublicationTargetHandle publicationTarget{};
  std::vector<PendingAnimationActivation> pendingAnimations{};
  std::vector<std::filesystem::path> deferredTextureArtifactBakes{};
  bool resetCameraController = false;
  bool sceneHasAuthoredLights = false;
  bool text3DEnabled = false;
  bool publicationTargetUnregistered = false;
  bool renderSceneFinalized = false;

  explicit EditorSceneDocument(std::pmr::memory_resource *runtimeMemory)
      : sceneRuntime(runtimeMemory) {}
};

EditorRuntime::SceneDocumentScope::SceneDocumentScope(
    EditorRuntime &runtime, EditorSceneDocument *document)
    : runtime_(&runtime), previous_(runtime.callbackDocument_) {
  runtime.callbackDocument_ = document;
}

EditorRuntime::SceneDocumentScope::~SceneDocumentScope() {
  if (runtime_ != nullptr) {
    runtime_->callbackDocument_ = previous_;
  }
}

EditorRuntime::SceneDocumentScope::SceneDocumentScope(
    SceneDocumentScope &&other) noexcept
    : runtime_(std::exchange(other.runtime_, nullptr)),
      previous_(other.previous_) {}

EditorRuntime::EditorRuntime(Application &app, RuntimeConfig config)
    : app_(app), config_(std::move(config)), cameraSystem_(cameraMemory_),
      activeDocument_(std::make_unique<EditorSceneDocument>(&pipelineMemory_)) {
}

EditorRuntime::~EditorRuntime() = default;

void EditorRuntime::initialize() {
  NURI_PROFILER_FUNCTION();
  activeDocument_->scene.bindResources(&resources());
  activeDocument_->publicationTarget =
      assets().registerScenePublicationTarget(activeDocument_->scene);
  activeDocument_->sceneRuntime.bindScene(&activeDocument_->scene);
  animationGpuServices_ = std::make_unique<AnimationGpuServices>(
      app_.getGPU(), config_.roots.shaders, &pipelineMemory_);
  activeDocument_->sceneRuntime.attachAnimationGpuServices(
      animationGpuServices_.get());
  animationPlayerService_ = std::make_unique<EditorAnimationPlayerService>(
      activeDocument_->scene, activeDocument_->sceneRuntime,
      sceneEditorSelectionState_, [this]() { return timeSeconds(); },
      [this]() { return advanceSimulationFrameIndex(); }, &pipelineMemory_);
  auto animationFrameProvider = std::make_unique<AnimationSceneFrameProvider>(
      activeDocument_->sceneRuntime);
  animationFrameProvider_ = animationFrameProvider.get();
  app_.getRenderPipeline().addComponent(
      std::move(animationFrameProvider),
      PipelineComponentDesc{
          .publish =
              [](void *state, FrameBuildContext &ctx) {
                return static_cast<AnimationSceneFrameProvider *>(state)
                    ->prepare(ctx);
              },
          .submitted =
              [](void *state, const RenderFrameContext &frame) noexcept {
                static_cast<AnimationSceneFrameProvider *>(state)
                    ->onFrameSubmitted(frame);
              },
          .abandoned =
              [](void *state, const RenderFrameContext &frame) noexcept {
                static_cast<AnimationSceneFrameProvider *>(state)
                    ->onFrameAbandoned(frame);
              },
      });

  auto bakeryResult = bakery::BakerySystem::create({
      .gpu = app_.getGPU(),
      .config = config_,
      .profile = bakery::BakeryExecutionProfile::Interactive,
  });
  if (bakeryResult.hasError()) {
    NURI_LOG_WARNING(
        "EditorRuntime::initialize: failed to create bakery system: %s",
        bakeryResult.error().c_str());
  } else {
    bakerySystem_ = std::move(bakeryResult.value());
  }

  initializeCamera();
  initializeTextSystem();
  initializeEditorRenderFeature();
  if (readEnvFlag("NURI_DEBUG_EDITOR_OVERLAY")) {
    initializeEditorOverlay();
    NURI_LOG_WARNING(
        "EditorRuntime: NURI_DEBUG_EDITOR_OVERLAY active; editor overlay "
        "enabled at startup");
  } else {
    textOverlayEnabled_ = false;
  }
  sharedEnvironmentLoad_ = loadSharedEnvironment(*this);
}

void EditorRuntime::update(double deltaTime) {
  tickRetiringSceneDocuments();
  const bool backgroundWorkAllowed =
      std::chrono::steady_clock::now() >= backgroundWorkResumeTime_;
  if (backgroundWorkAllowed) {
    tickDeferredSceneTextureArtifactBakes();
  }
  updateMetrics(deltaTime);
  if (editorOverlay_ != nullptr) {
    editorOverlay_->onUpdate(deltaTime);
  }
  (void)activeDocument_->sceneRuntime.tick({
      .frameDeltaSeconds = std::max(0.0, deltaTime),
      .absoluteTimeSeconds = timeSeconds(),
      .frameIndex = advanceSimulationFrameIndex(),
  });
  if (animationPlayerService_ != nullptr) {
    animationPlayerService_->onUpdate(deltaTime);
  }
  cameraSystem_.update(deltaTime, app_.getInput());
  if (backgroundWorkAllowed && bakerySystem_ && stagingDocument_ == nullptr) {
    bakerySystem_->tick();
  }
}

void EditorRuntime::draw() {
  const Camera *activeCamera = cameraSystem_.activeCamera();
  NURI_ASSERT(activeCamera != nullptr, "No active camera");
  auto commitResult = activeDocument_->scene.commit();
  NURI_ASSERT(!commitResult.hasError(), "Scene commit failed: %s",
              commitResult.error().c_str());
  buildFrameContext(*activeCamera, timeSeconds());
  queueTextSamples();
  submitPipelineFrame();
}

void EditorRuntime::resize(std::int32_t, std::int32_t) {}

bool EditorRuntime::onInput(const InputEvent &event) {
  if (event.type == InputEventType::Key &&
      event.payload.key.action == KeyAction::Press &&
      event.payload.key.key == Key::F6) {
    toggleEditorOverlay();
    return true;
  }
  if (editorOverlay_ != nullptr && editorOverlay_->onInput(event)) {
    return true;
  }
  return cameraSystem_.onInput(event, app_.getWindow());
}

void EditorRuntime::shutdown() {
  removeEditorOverlay();
  sceneSelectionIds_.clear();
  sceneSelectionLabels_.clear();
  sceneSelectionOptions_.clear();
  sceneSelectionVersion_ = std::numeric_limits<uint64_t>::max();
  if (animationPlayerService_ != nullptr) {
    animationPlayerService_->clear();
  }
  activeDocument_->sceneRuntime.reset();
  activeDocument_->sceneRuntime.bindScene(nullptr);
  activeDocument_->scene.graph().clear();
  if (isValid(sharedEnvironmentLoad_)) {
    assets().cancel(sharedEnvironmentLoad_);
    sharedEnvironmentLoad_ = {};
  }
  activeDocument_->scene.setEnvironment(EnvironmentHandles{});
  activeDocument_->scene.bindResources(nullptr);
  app_.getWindow().setCursorMode(CursorMode::Normal);
}

ResourceManager &EditorRuntime::resources() {
  return app_.getRenderer().resources();
}

AssetSystem &EditorRuntime::assets() { return app_.getRenderer().assets(); }

EditorRuntime::EditorSceneDocument &EditorRuntime::currentDocument() noexcept {
  NURI_ASSERT(activeDocument_ != nullptr,
              "EditorRuntime has no active scene document");
  return callbackDocument_ != nullptr ? *callbackDocument_ : *activeDocument_;
}

const EditorRuntime::EditorSceneDocument &
EditorRuntime::currentDocument() const noexcept {
  NURI_ASSERT(activeDocument_ != nullptr,
              "EditorRuntime has no active scene document");
  return callbackDocument_ != nullptr ? *callbackDocument_ : *activeDocument_;
}

RenderScene &EditorRuntime::scene() noexcept { return currentDocument().scene; }

SceneRuntimeHost &EditorRuntime::sceneRuntime() noexcept {
  return currentDocument().sceneRuntime;
}

RenderSettings &EditorRuntime::renderSettings() noexcept {
  return currentDocument().renderSettings;
}

const RenderSettings &EditorRuntime::renderSettings() const noexcept {
  return currentDocument().renderSettings;
}

ScenePublicationTargetHandle
EditorRuntime::scenePublicationTarget() const noexcept {
  return currentDocument().publicationTarget;
}

EditorRuntime::SceneDocumentScope EditorRuntime::useActiveSceneDocument() {
  return SceneDocumentScope(*this, activeDocument_.get());
}

EditorRuntime::SceneDocumentScope EditorRuntime::useStagingSceneDocument() {
  NURI_ASSERT(stagingDocument_ != nullptr,
              "EditorRuntime has no staging scene document");
  return SceneDocumentScope(*this, stagingDocument_.get());
}

Result<void, std::string> EditorRuntime::beginStagingSceneDocument() {
  if (stagingDocument_ != nullptr) {
    return Result<void, std::string>::makeError(
        "EditorRuntime already has a staging scene document");
  }
  auto document = std::make_unique<EditorSceneDocument>(&pipelineMemory_);
  document->scene.bindResources(&resources());
  document->sceneRuntime.bindScene(&document->scene);
  document->sceneRuntime.attachAnimationGpuServices(
      animationGpuServices_.get());
  document->scene.setEnvironment(activeDocument_->scene.environment());
  if (const Camera *camera = cameraSystem_.camera(mainCameraHandle_)) {
    document->camera = *camera;
  }
  document->publicationTarget =
      assets().registerScenePublicationTarget(document->scene);
  stagingDocument_ = std::move(document);
  assets().setInteractiveMode(true);
  return Result<void, std::string>::makeResult();
}

Result<bool, std::string> EditorRuntime::finalizeStagingSceneDocument() {
  if (stagingDocument_ == nullptr) {
    return Result<bool, std::string>::makeError(
        "EditorRuntime has no staging scene document to finalize");
  }
  if (!stagingDocument_->renderSceneFinalized) {
    auto commitResult = stagingDocument_->scene.commitInactiveStep(512u);
    if (commitResult.hasError()) {
      return commitResult;
    }
    if (!commitResult.value()) {
      return Result<bool, std::string>::makeResult(false);
    }
    stagingDocument_->renderSceneFinalized = true;
  }
  return app_.getRenderPipeline().prepareSceneStep(
      stagingDocument_->scene, resources(), 65536u,
      &stagingDocument_->renderSettings, &stagingDocument_->camera,
      app_.getAspectRatio(),
      static_cast<uint32_t>(std::max(app_.getWidth(), 1)),
      static_cast<uint32_t>(std::max(app_.getHeight(), 1)));
}

bool EditorRuntime::activateStagingSceneDocument() {
  if (stagingDocument_ == nullptr || !stagingDocument_->renderSceneFinalized) {
    return false;
  }
  (void)stagingDocument_->scene.sealDDGIActivationCoverageBounds();

  std::unique_ptr<EditorSceneDocument> previous = std::move(activeDocument_);
  activeDocument_ = std::move(stagingDocument_);
  backgroundWorkResumeTime_ =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  assets().setInteractiveMode(false);
  callbackDocument_ = nullptr;
  retiringDocuments_.push_back(std::move(previous));

  bindActiveSceneServices();
  if (Camera *camera = cameraSystem_.camera(mainCameraHandle_)) {
    *camera = activeDocument_->camera;
    syncEditorCameraWidgetState(*camera);
  }
  if (activeDocument_->resetCameraController) {
    if (CameraController *controller = cameraSystem_.activeController()) {
      controller->reset();
    }
  }
  sceneEditorSelectionState_.clear();
  temporalFrameService_.reset();
  if (editorOverlay_ != nullptr) {
    editorOverlay_->resetSceneUiState();
  }

  auto pendingAnimations = std::move(activeDocument_->pendingAnimations);
  for (std::filesystem::path &path :
       activeDocument_->deferredTextureArtifactBakes) {
    deferredSceneTextureArtifactBakes_.push_back(std::move(path));
  }
  activeDocument_->deferredTextureArtifactBakes.clear();
  auto scope = useActiveSceneDocument();
  for (const EditorSceneDocument::PendingAnimationActivation &pending :
       pendingAnimations) {
    if (pending.resources == nullptr || pending.instance == nullptr) {
      continue;
    }
    startAnimatedPrefabSceneSimulation(pending.sceneName, *pending.resources,
                                       *pending.instance, pending.params,
                                       pending.simulationDebugName);
  }
  return true;
}

void EditorRuntime::retireStagingSceneDocument() {
  if (stagingDocument_ == nullptr) {
    return;
  }
  callbackDocument_ = nullptr;
  retiringDocuments_.push_back(std::move(stagingDocument_));
  assets().setInteractiveMode(false);
}

void EditorRuntime::tickRetiringSceneDocuments() {
  for (auto it = retiringDocuments_.begin(); it != retiringDocuments_.end();
       ++it) {
    EditorSceneDocument &document = **it;
    if (!document.publicationTargetUnregistered) {
      const ScenePublicationTargetSnapshot snapshot =
          assets().query(document.publicationTarget);
      if (snapshot.pendingCount != 0u ||
          !assets().unregisterScenePublicationTarget(
              document.publicationTarget)) {
        continue;
      }
      document.publicationTargetUnregistered = true;
    }
    if (!document.scene.retireInactiveStep(512u)) {
      continue;
    }
    retiringDocuments_.erase(it);
    break;
  }
}

bool EditorRuntime::deferSceneTextureArtifactBake(
    const std::filesystem::path &sourcePath) {
  if (sourcePath.empty()) {
    return false;
  }
  const std::filesystem::path normalized = sourcePath.lexically_normal();
  if (callbackDocument_ == stagingDocument_.get() &&
      stagingDocument_ != nullptr) {
    auto &requests = stagingDocument_->deferredTextureArtifactBakes;
    if (std::ranges::find(requests, normalized) == requests.end()) {
      requests.push_back(normalized);
    }
    return true;
  }
  if (std::ranges::find(deferredSceneTextureArtifactBakes_, normalized) ==
      deferredSceneTextureArtifactBakes_.end()) {
    deferredSceneTextureArtifactBakes_.push_back(normalized);
  }
  return true;
}

void EditorRuntime::tickDeferredSceneTextureArtifactBakes() {
  if (stagingDocument_ != nullptr ||
      deferredSceneTextureArtifactBakes_.empty() || bakerySystem_ == nullptr) {
    return;
  }
  std::filesystem::path sourcePath =
      std::move(deferredSceneTextureArtifactBakes_.front());
  deferredSceneTextureArtifactBakes_.pop_front();
  auto enqueueResult = bakerySystem_->enqueue(
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
        "EditorRuntime: failed to queue deferred scene texture artifact "
        "bake for '%s': %s",
        sourcePath.string().c_str(), enqueueResult.error().c_str());
  }
}

ScenePublicationTargetSnapshot
EditorRuntime::stagingScenePublicationSnapshot() {
  if (stagingDocument_ == nullptr) {
    return ScenePublicationTargetSnapshot{
        .requestCount = 1u,
        .failedCount = 1u,
    };
  }
  return assets().query(stagingDocument_->publicationTarget);
}

void EditorRuntime::bindActiveSceneServices() {
  NURI_ASSERT(activeDocument_ != nullptr,
              "EditorRuntime has no active document to bind");
  if (animationFrameProvider_ != nullptr) {
    animationFrameProvider_->bindRuntime(activeDocument_->sceneRuntime);
  }
  if (animationPlayerService_ != nullptr) {
    animationPlayerService_->bindScene(activeDocument_->scene,
                                       activeDocument_->sceneRuntime, false);
  }
  if (editorOverlay_ != nullptr) {
    editorOverlay_->bindScene(activeDocument_->scene);
  }
}

Camera *EditorRuntime::mainCamera() {
  if (callbackDocument_ != nullptr &&
      callbackDocument_ != activeDocument_.get()) {
    return &callbackDocument_->camera;
  }
  return cameraSystem_.camera(mainCameraHandle_);
}

const Camera *EditorRuntime::mainCamera() const {
  if (callbackDocument_ != nullptr &&
      callbackDocument_ != activeDocument_.get()) {
    return &callbackDocument_->camera;
  }
  return cameraSystem_.camera(mainCameraHandle_);
}

double EditorRuntime::timeSeconds() const { return app_.getTime(); }

void EditorRuntime::syncEditorCameraWidgetState(const Camera &camera) {
  if (callbackDocument_ != nullptr &&
      callbackDocument_ != activeDocument_.get()) {
    return;
  }
  if (editorOverlay_ != nullptr) {
    editorOverlay_->syncCameraControllerWidgetStateFromCamera(camera);
  }
}

void EditorRuntime::syncSceneSelectionUi(const EditorSceneCatalog &catalog) {
  if (editorOverlay_ == nullptr) {
    return;
  }

  if (sceneSelectionVersion_ != catalog.version()) {
    const auto entries = catalog.entries();
    sceneSelectionIds_.clear();
    sceneSelectionLabels_.clear();
    sceneSelectionOptions_.clear();
    sceneSelectionIds_.reserve(entries.size());
    sceneSelectionLabels_.reserve(entries.size());
    sceneSelectionOptions_.reserve(entries.size());
    for (const EditorSceneEntry &entry : entries) {
      const std::string &newId = sceneSelectionIds_.emplace_back(entry.info.id);
      const std::string &newLabel =
          sceneSelectionLabels_.emplace_back(entry.info.label);
      sceneSelectionOptions_.push_back(EditorSceneSelectionOption{
          .id = newId,
          .label = newLabel,
      });
    }
    sceneSelectionVersion_ = catalog.version();
  }
  const EditorSceneTransitionSnapshot transition = catalog.transitionSnapshot();
  std::string_view phase{};
  switch (transition.phase) {
  case EditorSceneTransitionPhase::Idle:
    break;
  case EditorSceneTransitionPhase::Preparing:
    phase = "Preparing";
    break;
  case EditorSceneTransitionPhase::LoadingAssets:
    phase = "Loading";
    break;
  case EditorSceneTransitionPhase::Finalizing:
    phase = "Finalizing";
    break;
  case EditorSceneTransitionPhase::Failed:
    phase = "Failed";
    break;
  }
  editorOverlay_->setSceneSelectionUi(
      sceneSelectionOptions_, catalog.activeSceneId(), catalog.version(),
      "Toggle Editor: F6",
      EditorSceneLoadUiState{
          .pendingSceneId = transition.pendingSceneId,
          .phase = phase,
          .error = transition.error,
          .progress = transition.progress,
          .cancellable = transition.cancellable,
          .failed = transition.phase == EditorSceneTransitionPhase::Failed,
      });
}

std::optional<std::string> EditorRuntime::takeSceneSelectionRequest() {
  return editorOverlay_ != nullptr ? editorOverlay_->takeSceneSelectionRequest()
                                   : std::nullopt;
}

bool EditorRuntime::takeSceneCancelRequest() {
  return editorOverlay_ != nullptr && editorOverlay_->takeSceneCancelRequest();
}

void EditorRuntime::resetSceneState() {
  EditorSceneDocument &document = currentDocument();
  if (&document == activeDocument_.get() &&
      animationPlayerService_ != nullptr) {
    animationPlayerService_->clear();
  }
  document.scene.graph().clear();
  if (&document == activeDocument_.get() && editorOverlay_ != nullptr) {
    editorOverlay_->resetSceneUiState();
  }
  document.renderSettings = RenderSettings{};
  document.sceneHasAuthoredLights = false;
  document.text3DEnabled = false;
  document.pendingAnimations.clear();
}

void EditorRuntime::finalizeSceneLighting(
    std::span<const ScenePrefabLight> fallbackLights,
    const glm::mat4 &baseModel) {
  if (!currentDocument().sceneHasAuthoredLights &&
      !applyImportedLights(fallbackLights, baseModel)) {
    setupDefaultSceneLighting();
  }
}

void EditorRuntime::configureStaticModelOpaqueSettings(
    const glm::vec3 &lodThresholds) {
  RenderSettings &settings = renderSettings();
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableMeshLod = true;
  settings.opaque.enableTessellation = false;
  settings.opaque.forcedMeshLod = -1;
  settings.opaque.meshLodDistanceThresholds = lodThresholds;
  settings.opaque.enableInstanceAnimation = false;
}

const Model &EditorRuntime::requireLoadedModel(ModelRef modelRef,
                                               const char *modelError,
                                               const char *recordError) {
  NURI_ASSERT(isValid(modelRef), "%s", modelError);
  const ModelRecord *record = resources().tryGet(modelRef);
  NURI_ASSERT(record != nullptr && record->model != nullptr, "%s", recordError);
  return *record->model;
}

RenderableId EditorRuntime::addRequiredRenderable(ModelRef modelRef,
                                                  MaterialRef materialRef,
                                                  const glm::mat4 &modelMatrix,
                                                  const char *errorMessage) {
  auto nodeResult =
      scene().graph().createNode(scene().graph().rootNode(), {}, modelMatrix);
  NURI_ASSERT(!nodeResult.hasError(), "%s: %s", errorMessage,
              nodeResult.error().c_str());
  auto addResult =
      scene().graph().addRenderable(nodeResult.value(), modelRef, materialRef);
  NURI_ASSERT(!addResult.hasError(), "%s: %s", errorMessage,
              addResult.error().c_str());
  return addResult.value();
}

RenderableId EditorRuntime::instantiateImportedPrefabScene(
    std::string_view sceneName, const ImportedPrefabSceneResources &resourcesIn,
    const glm::mat4 &baseModel, SceneInstantiationMap *outInstantiation,
    NodeId *outRootNode) {
  if (!resourcesIn.ready) {
    return kInvalidRenderableId;
  }
  if (outInstantiation != nullptr) {
    outInstantiation->nodes.clear();
    outInstantiation->renderables.clear();
    outInstantiation->lights.clear();
  }
  if (outRootNode != nullptr) {
    *outRootNode = kInvalidNodeId;
  }

  auto rootNodeResult = scene().graph().createNode(scene().graph().rootNode(),
                                                   sceneName, baseModel);
  if (rootNodeResult.hasError()) {
    return kInvalidRenderableId;
  }

  SceneInstantiationMap localInstantiated;
  SceneInstantiationMap &instantiated =
      outInstantiation != nullptr ? *outInstantiation : localInstantiated;
  auto instantiateResult = scene().graph().instantiatePrefab(
      resourcesIn.prefab, rootNodeResult.value(), resourcesIn.assets,
      &instantiated);
  if (instantiateResult.hasError()) {
    (void)scene().graph().destroyNodeSubtree(rootNodeResult.value());
    return kInvalidRenderableId;
  }

  if (outRootNode != nullptr) {
    *outRootNode = rootNodeResult.value();
  }
  if (animationPlayerService_ != nullptr &&
      !resourcesIn.prefab.animations.empty()) {
    animationPlayerService_->registerPrefabInstance(
        sceneName, resourcesIn.prefab, instantiated, rootNodeResult.value());
  }
  currentDocument().sceneHasAuthoredLights = !instantiated.lights.empty();
  for (RenderableId renderableId : instantiated.renderables) {
    if (isValid(renderableId)) {
      return renderableId;
    }
  }
  return kInvalidRenderableId;
}

std::optional<BoundingBox> EditorRuntime::computeImportedPrefabBounds(
    const ImportedPrefabSceneResources &sceneIn, const glm::mat4 &baseModel) {
  if (!sceneIn.ready || sceneIn.prefab.nodes.empty()) {
    return std::nullopt;
  }

  const std::vector<glm::mat4> worldMatrices =
      buildPrefabWorldMatrices(sceneIn.prefab, baseModel);

  BoundingBox bounds{};
  bool hasBounds = false;
  for (const ScenePrefabRenderable &renderable : sceneIn.prefab.renderables) {
    if (renderable.nodeIndex >= worldMatrices.size() ||
        renderable.meshAssetIndex >= sceneIn.assets.models.size()) {
      continue;
    }
    const ModelRef modelRef = sceneIn.assets.models[renderable.meshAssetIndex];
    const ModelRecord *modelRecord = resources().tryGet(modelRef);
    if (modelRecord == nullptr || modelRecord->model == nullptr) {
      continue;
    }
    const BoundingBox worldBounds = modelRecord->model->bounds().getTransformed(
        worldMatrices[renderable.nodeIndex]);
    if (!hasBounds) {
      bounds = worldBounds;
      hasBounds = true;
    } else {
      bounds.combinePoint(worldBounds.min_);
      bounds.combinePoint(worldBounds.max_);
    }
  }
  return hasBounds ? std::optional<BoundingBox>(bounds) : std::nullopt;
}

std::optional<BoundingBox> EditorRuntime::computeImportedPrefabNodeBounds(
    const ImportedPrefabSceneResources &sceneIn, const glm::mat4 &baseModel,
    std::string_view nodeName) {
  if (!sceneIn.ready || sceneIn.prefab.nodes.empty() || nodeName.empty()) {
    return std::nullopt;
  }

  const std::vector<glm::mat4> worldMatrices =
      buildPrefabWorldMatrices(sceneIn.prefab, baseModel);

  BoundingBox bounds{};
  bool hasBounds = false;
  for (const ScenePrefabRenderable &renderable : sceneIn.prefab.renderables) {
    if (renderable.nodeIndex >= worldMatrices.size() ||
        renderable.meshAssetIndex >= sceneIn.assets.models.size()) {
      continue;
    }
    const ScenePrefabNode &node = sceneIn.prefab.nodes[renderable.nodeIndex];
    if (node.name != nodeName) {
      continue;
    }
    const ModelRef modelRef = sceneIn.assets.models[renderable.meshAssetIndex];
    const ModelRecord *modelRecord = resources().tryGet(modelRef);
    if (modelRecord == nullptr || modelRecord->model == nullptr) {
      continue;
    }
    const BoundingBox worldBounds = modelRecord->model->bounds().getTransformed(
        worldMatrices[renderable.nodeIndex]);
    if (!hasBounds) {
      bounds = worldBounds;
      hasBounds = true;
    } else {
      bounds.combinePoint(worldBounds.min_);
      bounds.combinePoint(worldBounds.max_);
    }
  }
  return hasBounds ? std::optional<BoundingBox>(bounds) : std::nullopt;
}

FramedSceneCameraState EditorRuntime::frameSceneCamera(
    const BoundingBox &bounds, const glm::mat4 &modelMatrix,
    float distanceScale, float minDistance, const glm::vec4 &eyeOffsetParams,
    const glm::vec2 &targetOffsetParams) {
  FramedSceneCameraState state{};
  state.rawRadius = std::max(0.5f * glm::length(bounds.getSize()), 0.25f);
  state.center = glm::vec3(modelMatrix * glm::vec4(bounds.getCenter(), 1.0f));
  state.radius = std::max(0.25f, state.rawRadius);
  state.cameraDistance = std::max(state.radius * distanceScale, minDistance);
  state.nearPlane = std::max(0.01f, state.cameraDistance / 3000.0f);
  state.farPlane =
      std::max(500.0f, state.cameraDistance + state.radius * 12.0f);

  Camera *camera = mainCamera();
  NURI_ASSERT(camera != nullptr, "Failed to get main camera");
  PerspectiveParams perspective = camera->perspective();
  perspective.nearPlane = state.nearPlane;
  perspective.farPlane = state.farPlane;
  camera->setProjectionType(ProjectionType::Perspective);
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

FramedSceneCameraState EditorRuntime::configureDragonSampleCamera(
    const ImportedPrefabSceneResources &sceneIn, const glm::mat4 &baseModel,
    const BoundingBox &dragonBounds) {
  FramedSceneCameraState cameraState{};
  cameraState.rawRadius =
      std::max(0.5f * glm::length(dragonBounds.getSize()), 0.25f);
  cameraState.radius = std::max(0.25f, cameraState.rawRadius);
  cameraState.cameraDistance = std::max(cameraState.radius * 2.05f, 4.0f);
  cameraState.nearPlane = 0.01f;
  cameraState.farPlane = 500.0f;
  cameraState.center = dragonBounds.getCenter();

  glm::vec3 viewDirection = glm::vec3(0.0f, 0.0f, 1.0f);
  if (const auto backdropBounds =
          computeImportedPrefabNodeBounds(sceneIn, baseModel, "Cloth Backdrop");
      backdropBounds.has_value()) {
    viewDirection = cameraState.center - backdropBounds->getCenter();
    viewDirection.y = 0.0f;
    if (glm::dot(viewDirection, viewDirection) > 1.0e-4f) {
      viewDirection = glm::normalize(viewDirection);
    } else {
      viewDirection = glm::vec3(0.0f, 0.0f, 1.0f);
    }
  }

  Camera *camera = mainCamera();
  NURI_ASSERT(camera != nullptr, "Failed to get main camera");
  PerspectiveParams perspective = camera->perspective();
  perspective.nearPlane = cameraState.nearPlane;
  perspective.farPlane = cameraState.farPlane;
  camera->setProjectionType(ProjectionType::Perspective);
  camera->setPerspective(perspective);

  glm::vec3 eye =
      cameraState.center + viewDirection * cameraState.cameraDistance;
  eye.y = std::max(cameraState.center.y + cameraState.radius * 0.2f, 1.0f);
  camera->setLookAt(eye, cameraState.center, glm::vec3(0.0f, 1.0f, 0.0f));
  syncEditorCameraWidgetState(*camera);
  return cameraState;
}

void EditorRuntime::destroyAnimatedPrefabSceneInstance(
    AnimatedPrefabSceneState &instance) {
  EditorSceneDocument &document = currentDocument();
  if (&document != activeDocument_.get()) {
    std::erase_if(document.pendingAnimations, [&instance](const auto &pending) {
      return pending.instance == &instance;
    });
    instance.simulation = kInvalidSimulationHandle;
    instance.rootNode = kInvalidNodeId;
    instance.instantiationMap.nodes.clear();
    instance.instantiationMap.renderables.clear();
    instance.instantiationMap.lights.clear();
    return;
  }
  if (animationPlayerService_ != nullptr && isValid(instance.rootNode)) {
    animationPlayerService_->unregisterPrefabInstance(instance.rootNode);
    instance.simulation = kInvalidSimulationHandle;
  } else if (isValid(instance.simulation)) {
    (void)currentDocument().sceneRuntime.destroyAnimationPoseSimulation(
        instance.simulation);
    instance.simulation = kInvalidSimulationHandle;
  }
  instance.rootNode = kInvalidNodeId;
  instance.instantiationMap.nodes.clear();
  instance.instantiationMap.renderables.clear();
  instance.instantiationMap.lights.clear();
}

void EditorRuntime::startAnimatedPrefabSceneSimulation(
    std::string_view sceneName, const ImportedPrefabSceneResources &resourcesIn,
    AnimatedPrefabSceneState &instance,
    const AnimationPoseSimulationParams &params,
    std::string_view simulationDebugName) {
  EditorSceneDocument &document = currentDocument();
  if (&document != activeDocument_.get()) {
    document.pendingAnimations.push_back(
        EditorSceneDocument::PendingAnimationActivation{
            .sceneName = std::string(sceneName),
            .resources = &resourcesIn,
            .instance = &instance,
            .params = params,
            .simulationDebugName = std::string(simulationDebugName),
        });
    return;
  }
  const std::string sceneNameString(sceneName);
  NURI_ASSERT(!resourcesIn.prefab.animations.empty(),
              "%s prefab has no animations", sceneNameString.c_str());
  NURI_ASSERT(params.primary.clipIndex < resourcesIn.prefab.animations.size(),
              "%s primary clip index %u is out of range",
              sceneNameString.c_str(), params.primary.clipIndex);
  if (params.blendMode == AnimationPoseBlendMode::Lerp) {
    NURI_ASSERT(params.secondary.clipIndex <
                    resourcesIn.prefab.animations.size(),
                "%s secondary clip index %u is out of range",
                sceneNameString.c_str(), params.secondary.clipIndex);
  }

  const auto commitResult = scene().commit();
  NURI_ASSERT(!commitResult.hasError(), "Scene commit failed for %s: %s",
              sceneNameString.c_str(), commitResult.error().c_str());
  (void)currentDocument().sceneRuntime.tick({
      .frameDeltaSeconds = 0.0,
      .absoluteTimeSeconds = timeSeconds(),
      .frameIndex = simulationFrameIndex_++,
  });

  if (animationPlayerService_ != nullptr) {
    animationPlayerService_->registerPrefabInstance(
        sceneName, resourcesIn.prefab, instance.instantiationMap,
        instance.rootNode);
    const bool started = animationPlayerService_->startPrefabInstancePlayback(
        instance.rootNode, params, simulationDebugName);
    NURI_ASSERT(started,
                "Failed to create %s animation simulation(s) for prefab "
                "instance",
                sceneNameString.c_str());
    instance.simulation = kInvalidSimulationHandle;
    return;
  }
  auto fallbackResult =
      currentDocument().sceneRuntime.createAnimationPoseSimulation(
          AnimationPoseSimulationCreateInfo{
              .prefab = &resourcesIn.prefab,
              .instantiationMap = &instance.instantiationMap,
              .rootNode = instance.rootNode,
              .debugName = simulationDebugName,
              .params = params,
          });
  NURI_ASSERT(!fallbackResult.hasError(),
              "Failed to create %s animation simulation: %s",
              sceneNameString.c_str(), fallbackResult.error().c_str());
  instance.simulation = fallbackResult.value();
}

uint32_t EditorRuntime::selectPreferredClipIndex(
    const ScenePrefab &prefab,
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

void EditorRuntime::setupText3DTestScene() {
  currentDocument().text3DEnabled = true;
  configureStaticModelOpaqueSettings(glm::vec3(8.0f, 16.0f, 32.0f));
  Camera *camera = mainCamera();
  NURI_ASSERT(camera != nullptr, "Failed to get main camera");
  PerspectiveParams perspective = camera->perspective();
  perspective.nearPlane = 0.01f;
  perspective.farPlane = 500.0f;
  camera->setProjectionType(ProjectionType::Perspective);
  camera->setPerspective(perspective);
  camera->setLookAt(glm::vec3(0.0f, 1.2f, -4.2f), glm::vec3(0.0f, 1.2f, 0.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f));
  syncEditorCameraWidgetState(*camera);
}

void EditorRuntime::resetSceneCameraController() {
  EditorSceneDocument &document = currentDocument();
  if (&document != activeDocument_.get()) {
    document.resetCameraController = true;
    return;
  }
  if (CameraController *controller = cameraSystem_.activeController()) {
    controller->reset();
  }
}

void EditorRuntime::logSingleRenderableSceneStats(
    std::string_view sceneName, const Model &model,
    const FramedSceneCameraState &cameraState) {
  NURI_LOG_INFO("EditorRuntime: %s scene stats submeshes=%zu vertices=%u "
                "indices=%u rawRadius=%.2f radius=%.2f near=%.3f far=%.2f",
                std::string(sceneName).c_str(), model.submeshes().size(),
                model.vertexCount(), model.indexCount(), cameraState.rawRadius,
                cameraState.radius, cameraState.nearPlane,
                cameraState.farPlane);
}

void EditorRuntime::initializeCamera() {
  Camera camera{};
  camera.setLookAt(glm::vec3(0.0f, 1.0f, -1.5f), glm::vec3(0.0f, 0.5f, 0.0f),
                   glm::vec3(0.0f, 1.0f, 0.0f));
  camera.setProjectionType(ProjectionType::Perspective);
  CameraController controller = makeFpsDirectController();
  mainCameraHandle_ = cameraSystem_.addCamera(camera, std::move(controller));
  NURI_ASSERT(mainCameraHandle_.isValid(),
              "Failed to add camera to camera system");
  const bool setActive =
      cameraSystem_.setActiveCamera(mainCameraHandle_, app_.getWindow());
  NURI_ASSERT(setActive, "Failed to activate main camera");
}

void EditorRuntime::initializeTextSystem() {
  const std::filesystem::path defaultFontPath = pickDefaultNfontPath(config_);
  const bool requireDefaultFont =
      std::filesystem::is_regular_file(defaultFontPath);
  auto textSystemResult = TextSystem::create({
      .gpu = app_.getGPU(),
      .memory = pipelineMemory_,
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

  registerText3DStage(app_.getRenderPipeline(), *textSystem_);
  registerText2DStage(app_.getRenderPipeline(), *textSystem_);
}

void EditorRuntime::initializeEditorRenderFeature() {
  if (editorRenderFeature_ != nullptr) {
    return;
  }
  editorRenderFeature_ = registerEditorOverlayStage(app_.getRenderPipeline());
}

void EditorRuntime::initializeEditorOverlay() {
  if (editorOverlay_ != nullptr) {
    return;
  }
  textOverlayEnabled_ = false;
  const EditorServices editorServices{
      .application = &app_,
      .scene = &activeDocument_->scene,
      .cameraSystem = &cameraSystem_,
      .gpu = &app_.getGPU(),
      .resources = &resources(),
      .renderPipeline = &app_.getRenderPipeline(),
      .textSystem = textSystem_.get(),
      .bakery = bakerySystem_.get(),
      .renderGraphTelemetry = &app_.getRenderer().renderGraphTelemetry(),
      .selectionState = &sceneEditorSelectionState_,
      .animationPlayer = animationPlayerService_.get(),
  };
  editorOverlay_ = EditorOverlayController::create(
      app_.getWindow(), app_.getGPU(), {}, editorServices);
  NURI_ASSERT(editorOverlay_ != nullptr,
              "Failed to create editor overlay controller");
  NURI_ASSERT(editorRenderFeature_ != nullptr,
              "Editor overlay feature is not initialized");
  editorRenderFeature_->setController(editorOverlay_.get());
  if (Camera *camera = mainCamera()) {
    editorOverlay_->syncCameraControllerWidgetStateFromCamera(*camera);
  }
}

void EditorRuntime::removeEditorOverlay() {
  if (editorOverlay_ == nullptr) {
    return;
  }
  if (editorRenderFeature_ != nullptr) {
    editorRenderFeature_->setController(nullptr);
  }
  editorOverlay_.reset();
  textOverlayEnabled_ = true;
}

void EditorRuntime::toggleEditorOverlay() {
  if (editorOverlay_ != nullptr) {
    removeEditorOverlay();
  } else {
    initializeEditorOverlay();
  }
}

void EditorRuntime::buildFrameContext(const Camera &camera,
                                      double timeSecondsIn) {
  frameContext_.scene = &activeDocument_->scene;
  frameContext_.resources = &resources();
  RenderSettings authoredSettings = activeDocument_->renderSettings;
  applyDebugRenderEnvOverrides(authoredSettings);
  frameRenderSettings_ = resolveRenderSettings(authoredSettings);
  frameContext_.frameIndex = frameIndex_;
  const TemporalSceneContentState sceneContent{
      .lightTopologyVersion = activeDocument_->scene.lightTopologyVersion(),
      .lightTransformVersion = activeDocument_->scene.lightTransformVersion(),
      .materialTableVersion = resources().materialVersion(),
      .environmentVersion = activeDocument_->scene.environmentVersion(),
  };
  auto planResult = buildPresentationAAPlan(
      frameRenderSettings_, {}, resources().gpuMultisampleCapabilities());
  NURI_ASSERT(!planResult.hasError(), "Invalid presentation AA plan: %s",
              planResult.error().c_str());
  frameContext_.presentationAA = planResult.value();
  auto cameraResult = temporalFrameService_.prepareFrame(
      camera, app_.getAspectRatio(), frameRenderSettings_.antiAliasing,
      frameContext_.presentationAA,
      TemporalCameraFrameDesc{
          .renderExtent =
              glm::uvec2(static_cast<uint32_t>(std::max(app_.getWidth(), 0)),
                         static_cast<uint32_t>(std::max(app_.getHeight(), 0))),
          .sceneContent = sceneContent,
      },
      frameContext_.frameIndex, timeSecondsIn, frameDeltaSeconds_);
  NURI_ASSERT(!cameraResult.hasError(), "Temporal frame prepare failed: %s",
              cameraResult.error().c_str());
  frameContext_.camera = cameraResult.value();
  frameContext_.temporalFrameService = &temporalFrameService_;
  frameRenderSettings_.antiAliasing.debug.resetHistoryRequested = false;
  frameContext_.settings = frameRenderSettings_;
  frameContext_.metrics = {};
  frameContext_.metrics.frameIndex = frameContext_.frameIndex;
  frameContext_.metrics.antiAliasing =
      makeAntiAliasingFrameMetrics(frameContext_.camera);
  frameContext_.ddgiProbeInspectRequest.reset();
  frameContext_.ddgiProbeInspectResult.reset();
  frameContext_.sharedDepthTexture = {};
  frameContext_.timeSeconds = timeSecondsIn;
  frameContext_.deltaSeconds = frameDeltaSeconds_;
  enqueueDebugShadowInspectProbe();
}

void EditorRuntime::submitPipelineFrame() {
  if (editorOverlay_ != nullptr) {
    frameContext_.ddgiProbeInspectRequest =
        editorOverlay_->takeDDGIProbeInspectRequest();
  }
  auto renderResult =
      app_.getRenderer().render(app_.getRenderPipeline(), frameContext_);
  NURI_ASSERT(!renderResult.hasError(), "Render failed: %s",
              renderResult.error().c_str());
  lastFrameOutputAvailable_ = renderResult.value();
  if (lastFrameOutputAvailable_) {
    ++frameIndex_;
    recordFrameOutput();
  }
  logDebugShadowInspectProbeResult();
  if (editorOverlay_ != nullptr) {
    if (auto settingsUpdate = editorOverlay_->takeRenderSettingsUpdate()) {
      frameRenderSettings_ = resolveRenderSettings(*settingsUpdate);
    }
  }
  persistFrameRenderSettings(activeDocument_->renderSettings,
                             frameRenderSettings_);
}

void EditorRuntime::enqueueDebugShadowInspectProbe() {
  const DebugShadowInspectProbeConfig &config = debugShadowInspectProbeConfig();
  if (!config.enabled || debugShadowInspectProbe_.submitted ||
      debugShadowInspectProbe_.completed ||
      frameContext_.shadowInspectRequest.has_value() ||
      frameContext_.frameIndex < config.warmupFrames ||
      activeDocument_->scene.renderables().empty()) {
    return;
  }

  const uint32_t framebufferWidth =
      static_cast<uint32_t>(std::max(app_.getWidth(), 1));
  const uint32_t framebufferHeight =
      static_cast<uint32_t>(std::max(app_.getHeight(), 1));
  const uint32_t x = config.explicitPixel
                         ? std::min(config.x, framebufferWidth - 1u)
                         : framebufferWidth / 2u;
  const uint32_t y = config.explicitPixel
                         ? std::min(config.y, framebufferHeight - 1u)
                         : framebufferHeight / 2u;

  const uint64_t requestId = ++debugShadowInspectProbe_.requestId;
  frameContext_.shadowInspectRequest =
      ShadowInspectRequest{.x = x, .y = y, .requestId = requestId};
  debugShadowInspectProbe_.submitted = true;
  debugShadowInspectProbe_.submissionFrame = frameContext_.frameIndex;
  NURI_LOG_INFO("EditorRuntime: NURI_DEBUG_SHADOW_INSPECT queued "
                "request=%llu pixel=(%u,%u) frame=%llu renderables=%zu",
                static_cast<unsigned long long>(requestId), x, y,
                static_cast<unsigned long long>(frameContext_.frameIndex),
                activeDocument_->scene.renderables().size());
}

void EditorRuntime::logDebugShadowInspectProbeResult() {
  const DebugShadowInspectProbeConfig &config = debugShadowInspectProbeConfig();
  if (!config.enabled || !debugShadowInspectProbe_.submitted ||
      debugShadowInspectProbe_.completed) {
    return;
  }

  if (frameContext_.shadowInspectResult.has_value() &&
      frameContext_.shadowInspectResult->requestId ==
          debugShadowInspectProbe_.requestId) {
    const ShadowInspectResult &result = *frameContext_.shadowInspectResult;
    NURI_LOG_INFO("EditorRuntime: NURI_DEBUG_SHADOW_INSPECT result "
                  "request=%llu valid=%s receiverDepth=%.6f "
                  "receiverCompareDepth=%.6f sampledDepth=%.6f cascade=%u "
                  "cascadeBlend=%.6f",
                  static_cast<unsigned long long>(result.requestId),
                  boolToString(result.valid), result.receiverDepth,
                  result.receiverCompareDepth, result.sampledDepth,
                  result.cascadeIndex, result.cascadeBlendFactor);
    debugShadowInspectProbe_.completed = true;
    return;
  }

  if (!debugShadowInspectProbe_.timeoutLogged &&
      frameContext_.frameIndex >
          debugShadowInspectProbe_.submissionFrame + config.timeoutFrames) {
    NURI_LOG_WARNING(
        "EditorRuntime: NURI_DEBUG_SHADOW_INSPECT timed out "
        "waiting for request=%llu after %llu frames",
        static_cast<unsigned long long>(debugShadowInspectProbe_.requestId),
        static_cast<unsigned long long>(
            frameContext_.frameIndex -
            debugShadowInspectProbe_.submissionFrame));
    debugShadowInspectProbe_.timeoutLogged = true;
  }
}

void EditorRuntime::queueTextSamples() {
  if (textSystem_ == nullptr) {
    return;
  }
  textSystem_->beginFrame(frameContext_.frameIndex);

  const FontHandle defaultFont = textSystem_->defaultFont();
  if (!isValid(defaultFont)) {
    return;
  }

  const float baseFontSizePx =
      std::clamp(textSystem_->defaultFontSizePx(), 8.0f, 256.0f);
  ScopedScratch scopedScratch(textScratchArena_);
  std::pmr::memory_resource &scratch = *scopedScratch.resource();
  if (textOverlayEnabled_) {
    (void)enqueue2DTextSamples(defaultFont, baseFontSizePx, scratch);
  }
  if (activeDocument_->text3DEnabled) {
    (void)enqueue3DTextSamples(defaultFont, baseFontSizePx, scratch);
  }
}

bool EditorRuntime::enqueue2DTextSamples(FontHandle defaultFont,
                                         float baseFontSizePx,
                                         std::pmr::memory_resource &scratch) {
  auto enqueueSample =
      [&](const Text2DDesc &sample) -> std::optional<TextBounds> {
    auto enqueue = textSystem_->enqueue2D(sample, scratch);
    return enqueue.hasError() ? std::nullopt
                              : std::optional<TextBounds>(enqueue.value());
  };

  Text2DDesc headline{};
  headline.utf8 = "MTSDF 2D Raster Test 0123456789 AaBbCc";
  headline.style.font = defaultFont;
  headline.style.pxSize = baseFontSizePx;
  headline.layout.alignH = TextAlignH::Left;
  headline.layout.alignV = TextAlignV::Top;
  headline.fillColor = {.r = 1.0f, .g = 1.0f, .b = 1.0f, .a = 1.0f};
  headline.x = 20.0f;
  headline.y = 20.0f;
  if (!enqueueSample(headline).has_value()) {
    return false;
  }

  int32_t windowWidth = 0;
  int32_t windowHeight = 0;
  app_.getWindow().getWindowSize(windowWidth, windowHeight);
  if (windowWidth <= 0 || windowHeight <= 0) {
    app_.getWindow().getFramebufferSize(windowWidth, windowHeight);
  }
  const float overlayWidth =
      std::max(static_cast<float>(windowWidth) - 40.0f, 0.0f);
  if (overlayWidth > 0.0f) {
    const float fps = currentFps_;
    const float frameTimeMs = fps > 0.0f ? 1000.0f / fps : 0.0f;
    std::array<char, 96> perfText{};
    std::snprintf(perfText.data(), perfText.size(), "FPS: %.1f\nFT: %.2f ms",
                  fps, frameTimeMs);

    Text2DDesc perf{};
    perf.utf8 = perfText.data();
    perf.style.font = defaultFont;
    perf.style.pxSize = baseFontSizePx * 0.52f;
    perf.layout.alignH = TextAlignH::Right;
    perf.layout.alignV = TextAlignV::Top;
    perf.layout.maxWidthPx = overlayWidth;
    perf.fillColor = {.r = 0.95f, .g = 1.0f, .b = 0.95f, .a = 1.0f};
    perf.x = 20.0f;
    perf.y = 20.0f;
    if (!enqueueSample(perf).has_value()) {
      return false;
    }
  }
  return true;
}

bool EditorRuntime::enqueue3DTextSamples(FontHandle defaultFont,
                                         float baseFontSizePx,
                                         std::pmr::memory_resource &scratch) {
  auto enqueueSample =
      [&](const Text3DDesc &sample) -> std::optional<TextBounds> {
    auto enqueue = textSystem_->enqueue3D(sample, scratch);
    return enqueue.hasError() ? std::nullopt
                              : std::optional<TextBounds>(enqueue.value());
  };

  Text3DDesc spherical{};
  spherical.utf8 = "MTSDF 3D BILLBOARD";
  spherical.style.font = defaultFont;
  spherical.style.pxSize = baseFontSizePx * 0.75f;
  spherical.layout.alignH = TextAlignH::Center;
  spherical.layout.alignV = TextAlignV::Middle;
  spherical.fillColor = {.r = 1.0f, .g = 1.0f, .b = 1.0f, .a = 1.0f};
  spherical.billboard = TextBillboardMode::Spherical;
  glm::mat4 sphericalWorld =
      glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 2.2f, 0.0f)),
                 glm::vec3(0.025f));
  spherical.worldFromText = encodeWorld(sphericalWorld);
  if (!enqueueSample(spherical).has_value()) {
    return false;
  }

  Text3DDesc clock{};
  clock.utf8 = formatLocalTimeHhMmSs();
  clock.style.font = defaultFont;
  clock.style.pxSize = baseFontSizePx * 0.62f;
  clock.layout.alignH = TextAlignH::Center;
  clock.layout.alignV = TextAlignV::Middle;
  clock.fillColor = {.r = 1.0f, .g = 0.95f, .b = 0.82f, .a = 1.0f};
  clock.billboard = TextBillboardMode::Spherical;
  glm::mat4 clockWorld =
      glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.85f, 1.15f)),
                 glm::vec3(0.02f));
  clock.worldFromText = encodeWorld(clockWorld);
  return enqueueSample(clock).has_value();
}

bool EditorRuntime::applyImportedLights(
    std::span<const ScenePrefabLight> importedLights,
    const glm::mat4 &modelMatrix) {
  if (importedLights.empty()) {
    return false;
  }
  auto rootNodeResult =
      scene().graph().createNode(scene().graph().rootNode(), {}, modelMatrix);
  if (rootNodeResult.hasError()) {
    return false;
  }
  bool addedAny = false;
  for (const ScenePrefabLight &importedLight : importedLights) {
    auto lightNodeResult = scene().graph().createNode(
        rootNodeResult.value(), importedLight.light.name,
        makeTransformMatrix(importedLight.light.position,
                            importedLight.light.rotation));
    if (lightNodeResult.hasError()) {
      continue;
    }

    LightDesc localLight = importedLight.light;
    localLight.position = glm::vec3(0.0f);
    localLight.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    auto addResult =
        scene().graph().addLight(lightNodeResult.value(), localLight);
    if (addResult.hasError()) {
      continue;
    }
    addedAny = true;
  }
  return addedAny;
}

void EditorRuntime::setupDefaultSceneLighting() {
  Camera *camera = mainCamera();
  if (camera == nullptr) {
    return;
  }

  LightDesc light{};
  light.type = LightType::Directional;
  light.name = "Default Directional";
  light.position = camera->position() + camera->forward() * 2.0f;
  light.rotation = camera->orientation();
  light.color = glm::vec3(1.0f);
  light.intensity = 2.0f;
  light.enabled = true;
  auto nodeResult = scene().graph().createNode(
      scene().graph().rootNode(), light.name,
      makeTransformMatrix(light.position, light.rotation));
  NURI_ASSERT(!nodeResult.hasError(),
              "Failed to create default directional light node: %s",
              nodeResult.error().c_str());
  light.position = glm::vec3(0.0f);
  light.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  auto addResult = scene().graph().addLight(nodeResult.value(), light);
  NURI_ASSERT(!addResult.hasError(),
              "Failed to add default directional light: %s",
              addResult.error().c_str());
}

void EditorRuntime::updateMetrics(double deltaTime) {
  frameDeltaSeconds_ =
      (std::isfinite(deltaTime) && deltaTime >= 0.0) ? deltaTime : 0.0;
  fpsAccumulatorSeconds_ += frameDeltaSeconds_;
}

void EditorRuntime::recordFrameOutput() {
  ++fpsFrameCount_;
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

} // namespace nuri
