#pragma once

#include "nuri/app/editor_animation_player_service.h"
#include "nuri/app/editor_scene_spec.h"
#include "nuri/bakery/bakery_system.h"
#include "nuri/core/application.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/frame/temporal_frame_service.h"
#include "nuri/scene/camera_system.h"
#include "nuri/scene/render_scene.h"
#include "nuri/scene_runtime/scene_runtime_host.h"
#include "nuri/text/text_system.h"
#include "nuri/ui/editor_feature.h"
#include "nuri/ui/editor_overlay_controller.h"
#include "nuri/ui/editor_services.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri {

class AnimationGpuServices;
class AnimationSceneFrameProvider;

struct FramedSceneCameraState {
  glm::vec3 center{0.0f};
  float rawRadius = 0.25f;
  float radius = 0.25f;
  float cameraDistance = 2.0f;
  float nearPlane = 0.01f;
  float farPlane = 500.0f;
};

class EditorSceneCatalog;

class EditorRuntime {
public:
  EditorRuntime(Application &app, RuntimeConfig config);
  ~EditorRuntime();

  EditorRuntime(const EditorRuntime &) = delete;
  EditorRuntime &operator=(const EditorRuntime &) = delete;

  void initialize();
  void update(double deltaTime);
  void draw();
  void resize(std::int32_t width, std::int32_t height);
  [[nodiscard]] bool onInput(const InputEvent &event);
  void shutdown();

  [[nodiscard]] const RuntimeConfig &config() const noexcept { return config_; }
  [[nodiscard]] Application &app() noexcept { return app_; }
  [[nodiscard]] const Application &app() const noexcept { return app_; }
  [[nodiscard]] ResourceManager &resources();
  [[nodiscard]] AssetSystem &assets();
  [[nodiscard]] RenderScene &scene() noexcept;
  [[nodiscard]] SceneRuntimeHost &sceneRuntime() noexcept;
  [[nodiscard]] CameraSystem &cameraSystem() noexcept { return cameraSystem_; }
  [[nodiscard]] bakery::BakerySystem *bakery() noexcept {
    return bakerySystem_.get();
  }
  [[nodiscard]] TextSystem *textSystem() noexcept { return textSystem_.get(); }
  [[nodiscard]] RenderSettings &renderSettings() noexcept;
  [[nodiscard]] const RenderSettings &renderSettings() const noexcept;
  [[nodiscard]] ScenePublicationTargetHandle
  scenePublicationTarget() const noexcept;
  [[nodiscard]] CameraHandle mainCameraHandle() const noexcept {
    return mainCameraHandle_;
  }
  [[nodiscard]] Camera *mainCamera();
  [[nodiscard]] const Camera *mainCamera() const;
  [[nodiscard]] uint64_t advanceSimulationFrameIndex() noexcept {
    return simulationFrameIndex_++;
  }
  [[nodiscard]] double timeSeconds() const;
  [[nodiscard]] float currentFps() const noexcept { return currentFps_; }
  [[nodiscard]] RenderFrameContext &frameContext() noexcept {
    return frameContext_;
  }
  [[nodiscard]] bool lastFrameOutputAvailable() const noexcept {
    return lastFrameOutputAvailable_;
  }
  [[nodiscard]] SceneEditorSelectionState &selectionState() noexcept {
    return sceneEditorSelectionState_;
  }
  [[nodiscard]] EditorAnimationPlayerService *animationPlayer() noexcept {
    return animationPlayerService_.get();
  }

  void syncEditorCameraWidgetState(const Camera &camera);
  void syncSceneSelectionUi(const EditorSceneCatalog &catalog);
  [[nodiscard]] std::optional<std::string> takeSceneSelectionRequest();
  [[nodiscard]] bool takeSceneCancelRequest();

  void resetSceneState();
  void finalizeSceneLighting(std::span<const ImportedSceneLight> fallbackLights,
                             const glm::mat4 &baseModel);
  void configureStaticModelOpaqueSettings(const glm::vec3 &lodThresholds);
  [[nodiscard]] const Model &requireLoadedModel(ModelRef modelRef,
                                                const char *modelError,
                                                const char *recordError);
  [[nodiscard]] RenderableId addRequiredRenderable(ModelRef modelRef,
                                                   MaterialRef materialRef,
                                                   const glm::mat4 &modelMatrix,
                                                   const char *errorMessage);
  [[nodiscard]] RenderableId instantiateImportedPrefabScene(
      std::string_view sceneName, const ImportedPrefabSceneResources &resources,
      const glm::mat4 &baseModel,
      SceneInstantiationMap *outInstantiation = nullptr,
      NodeId *outRootNode = nullptr);
  [[nodiscard]] std::optional<BoundingBox>
  computeImportedPrefabBounds(const ImportedPrefabSceneResources &scene,
                              const glm::mat4 &baseModel);
  [[nodiscard]] std::optional<BoundingBox>
  computeImportedPrefabNodeBounds(const ImportedPrefabSceneResources &scene,
                                  const glm::mat4 &baseModel,
                                  std::string_view nodeName);
  [[nodiscard]] FramedSceneCameraState
  frameSceneCamera(const BoundingBox &bounds, const glm::mat4 &modelMatrix,
                   float distanceScale, float minDistance,
                   const glm::vec4 &eyeOffsetParams,
                   const glm::vec2 &targetOffsetParams);
  [[nodiscard]] FramedSceneCameraState
  configureDragonSampleCamera(const ImportedPrefabSceneResources &scene,
                              const glm::mat4 &baseModel,
                              const BoundingBox &dragonBounds);
  void destroyAnimatedPrefabSceneInstance(AnimatedPrefabSceneState &instance);
  void startAnimatedPrefabSceneSimulation(
      std::string_view sceneName, const ImportedPrefabSceneResources &resources,
      AnimatedPrefabSceneState &instance,
      const AnimationPoseSimulationParams &params,
      std::string_view simulationDebugName);
  [[nodiscard]] static uint32_t
  selectPreferredClipIndex(const ScenePrefab &prefab,
                           std::span<const std::string_view> preferredNames);
  void queueTextSamples();
  void setupText3DTestScene();
  void logSingleRenderableSceneStats(std::string_view sceneName,
                                     const Model &model,
                                     const FramedSceneCameraState &cameraState);
  void resetSceneCameraController();
  // Scene factories call this while authoring either the active or staging
  // document. Staging requests are held until after atomic activation so the
  // bakery cannot compete with the critical transition path.
  [[nodiscard]] bool
  deferSceneTextureArtifactBake(const std::filesystem::path &sourcePath);

private:
  friend class EditorSceneCatalog;
  struct EditorSceneDocument;

  class SceneDocumentScope final {
  public:
    ~SceneDocumentScope();
    SceneDocumentScope(const SceneDocumentScope &) = delete;
    SceneDocumentScope &operator=(const SceneDocumentScope &) = delete;
    SceneDocumentScope(SceneDocumentScope &&other) noexcept;
    SceneDocumentScope &operator=(SceneDocumentScope &&) = delete;

  private:
    friend class EditorRuntime;
    SceneDocumentScope(EditorRuntime &runtime, EditorSceneDocument *document);
    EditorRuntime *runtime_ = nullptr;
    EditorSceneDocument *previous_ = nullptr;
  };

  [[nodiscard]] EditorSceneDocument &currentDocument() noexcept;
  [[nodiscard]] const EditorSceneDocument &currentDocument() const noexcept;
  [[nodiscard]] SceneDocumentScope useActiveSceneDocument();
  [[nodiscard]] SceneDocumentScope useStagingSceneDocument();
  [[nodiscard]] Result<void, std::string> beginStagingSceneDocument();
  [[nodiscard]] Result<bool, std::string> finalizeStagingSceneDocument();
  [[nodiscard]] bool activateStagingSceneDocument();
  void retireStagingSceneDocument();
  void tickRetiringSceneDocuments();
  void tickDeferredSceneTextureArtifactBakes();
  [[nodiscard]] ScenePublicationTargetSnapshot
  stagingScenePublicationSnapshot();
  void bindActiveSceneServices();

  void initializeCamera();
  void initializeTextSystem();
  void initializeEditorRenderFeature();
  void initializeEditorOverlay();
  void removeEditorOverlay();
  void toggleEditorOverlay();
  void updateMetrics(double deltaTime);
  void recordFrameOutput();
  void buildFrameContext(const Camera &camera, double timeSeconds);
  void submitPipelineFrame();
  void enqueueDebugShadowInspectProbe();
  void logDebugShadowInspectProbeResult();
  [[nodiscard]] bool enqueue2DTextSamples(FontHandle defaultFont,
                                          float baseFontSizePx,
                                          std::pmr::memory_resource &scratch);
  [[nodiscard]] bool enqueue3DTextSamples(FontHandle defaultFont,
                                          float baseFontSizePx,
                                          std::pmr::memory_resource &scratch);
  [[nodiscard]] bool
  applyImportedLights(std::span<const ImportedSceneLight> importedLights,
                      const glm::mat4 &modelMatrix);
  void setupDefaultSceneLighting();

  Application &app_;
  const RuntimeConfig config_;
  // EditorRuntime keeps cameraMemory_ and pipelineMemory_ as
  // unsynchronized_pool_resource instances; they are main-thread-only.
  std::pmr::unsynchronized_pool_resource cameraMemory_;
  std::pmr::unsynchronized_pool_resource pipelineMemory_;
  CameraSystem cameraSystem_;
  std::unique_ptr<EditorSceneDocument> activeDocument_{};
  std::unique_ptr<EditorSceneDocument> stagingDocument_{};
  std::deque<std::unique_ptr<EditorSceneDocument>> retiringDocuments_{};
  std::deque<std::filesystem::path> deferredSceneTextureArtifactBakes_{};
  std::chrono::steady_clock::time_point backgroundWorkResumeTime_{};
  EditorSceneDocument *callbackDocument_ = nullptr;
  SceneEditorSelectionState sceneEditorSelectionState_{};
  std::unique_ptr<AnimationGpuServices> animationGpuServices_{};
  AnimationSceneFrameProvider *animationFrameProvider_ = nullptr;
  std::unique_ptr<EditorAnimationPlayerService> animationPlayerService_{};
  std::unique_ptr<bakery::BakerySystem> bakerySystem_{};
  std::unique_ptr<TextSystem> textSystem_{};
  ScratchArena textScratchArena_{};
  std::vector<std::string> sceneSelectionIds_{};
  std::vector<std::string> sceneSelectionLabels_{};
  std::vector<EditorSceneSelectionOption> sceneSelectionOptions_{};
  uint64_t sceneSelectionVersion_ = std::numeric_limits<uint64_t>::max();
  std::unique_ptr<EditorOverlayController> editorOverlay_{};
  EditorOverlayPass *editorRenderFeature_ = nullptr;
  CameraHandle mainCameraHandle_{};
  // Transient per-frame render settings after frame-local overrides.
  RenderSettings frameRenderSettings_{};
  RenderFrameContext frameContext_{};
  bool lastFrameOutputAvailable_ = false;
  TemporalFrameService temporalFrameService_{};
  uint64_t frameIndex_ = 0;
  uint64_t simulationFrameIndex_ = 0;
  struct DebugShadowInspectProbeState {
    bool submitted = false;
    bool completed = false;
    bool timeoutLogged = false;
    uint64_t requestId = 0;
    uint64_t submissionFrame = 0;
  } debugShadowInspectProbe_{};
  double frameDeltaSeconds_ = 0.0;
  double fpsAccumulatorSeconds_ = 0.0;
  uint32_t fpsFrameCount_ = 0;
  float currentFps_ = 0.0f;
  bool textOverlayEnabled_ = false;
  EnvironmentAssetHandle sharedEnvironmentLoad_{};
};

} // namespace nuri
