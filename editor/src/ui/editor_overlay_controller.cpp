#include "nuri/editor_pch.h"

#include "nuri/ui/editor_overlay_controller.h"

#include "nuri/core/log.h"
#include "nuri/ui/imgui_editor.h"
#include "nuri/ui/imgui_gizmo_controller.h"
#include "nuri/utils/env_utils.h"

namespace nuri {
namespace {

[[nodiscard]] bool debugShadowInspectOnClickEnabled() {
  static const bool enabled = readEnvFlag("NURI_DEBUG_SHADOW_INSPECT_ON_CLICK");
  return enabled;
}

} // namespace

std::unique_ptr<EditorOverlayController>
EditorOverlayController::create(Window &window, GPUDevice &gpu,
                                std::function<void()> callback,
                                const EditorServices &services) {
  NURI_LOG_DEBUG(
      "EditorOverlayController::create: Creating editor overlay controller");
  return std::unique_ptr<EditorOverlayController>(
      new EditorOverlayController(window, gpu, std::move(callback), services));
}

EditorOverlayController::EditorOverlayController(Window &window, GPUDevice &gpu,
                                                 std::function<void()> callback,
                                                 const EditorServices &services)
    : editor_(ImGuiEditor::create(window, gpu, services)),
      callback_(std::move(callback)) {
  if (services.hasAllDependencies()) {
    gizmoController_ = createImGuizmoController(services);
    if (!gizmoController_) {
      NURI_LOG_WARNING(
          "EditorOverlayController: failed to create gizmo controller");
    }
    return;
  }

  const bool hasAnyGizmoService = services.scene != nullptr ||
                                  services.cameraSystem != nullptr ||
                                  services.gpu != nullptr;
  if (hasAnyGizmoService) {
    NURI_LOG_WARNING(
        "EditorOverlayController: incomplete EditorServices for gizmo "
        "(scene=%d cameraSystem=%d gpu=%d); gizmo disabled",
        services.scene != nullptr, services.cameraSystem != nullptr,
        services.gpu != nullptr);
  }
}

EditorOverlayController::~EditorOverlayController() = default;

void EditorOverlayController::resetControllers() {
  if (gizmoController_) {
    gizmoController_->reset();
  }
}

void EditorOverlayController::resetSceneUiState() {
  resetControllers();
  if (editor_) {
    editor_->resetSceneUiState();
  }
}

void EditorOverlayController::bindScene(RenderScene &scene) {
  if (editor_) {
    editor_->bindScene(scene);
  }
  if (gizmoController_) {
    gizmoController_->bindScene(scene);
  }
}

void EditorOverlayController::syncCameraControllerWidgetStateFromCamera(
    const Camera &camera) {
  if (editor_) {
    editor_->syncCameraControllerWidgetStateFromCamera(camera);
  }
}

void EditorOverlayController::setSceneSelectionUi(
    std::span<const EditorSceneSelectionOption> scenes,
    std::string_view selectedSceneId, uint64_t version,
    std::string_view hotkeyHint, const EditorSceneLoadUiState &load) {
  if (editor_) {
    editor_->setSceneSelectionUi(scenes, selectedSceneId, version, hotkeyHint,
                                 load);
  }
}

std::optional<std::string>
EditorOverlayController::takeSceneSelectionRequest() {
  return editor_ ? editor_->takeSceneSelectionRequest() : std::nullopt;
}

bool EditorOverlayController::takeSceneCancelRequest() {
  return editor_ != nullptr && editor_->takeSceneCancelRequest();
}

std::optional<RenderSettings>
EditorOverlayController::takeRenderSettingsUpdate() {
  return std::exchange(pendingRenderSettingsUpdate_, std::nullopt);
}

bool EditorOverlayController::onInput(const InputEvent &event) {
  if (!editor_) {
    return false;
  }

  if (event.type == InputEventType::Key &&
      event.payload.key.action == KeyAction::Press &&
      event.payload.key.key == Key::F6) {
    // Let application-level hotkeys toggle the editor overlay.
    return false;
  }

  switch (event.type) {
  case InputEventType::Key:
  case InputEventType::Character:
    return editor_->wantsCaptureKeyboard();
  case InputEventType::MouseButton:
  case InputEventType::MouseMove:
  case InputEventType::MouseScroll:
  case InputEventType::CursorEnter:
    if (editor_->wantsCaptureMouse()) {
      if (gizmoController_) {
        gizmoController_->invalidatePendingPicks();
      }
      return true;
    }
    if (gizmoController_) {
      gizmoController_->setShadowInspectRequestsEnabled(
          editor_->isShadowsWindowOpen() || debugShadowInspectOnClickEnabled());
    }
    if (gizmoController_ && gizmoController_->onInput(event)) {
      return true;
    }
    return false;
  case InputEventType::Focus:
    return false;
  default:
    NURI_LOG_WARNING(
        "EditorOverlayController::onInput: Unknown input event type: %d",
        static_cast<int>(event.type));
  }
  return false;
}

void EditorOverlayController::onUpdate(double deltaTime) {
  frameDeltaSeconds_ = deltaTime;
  if (editor_) {
    editor_->applyDeferredUiActions();
  }
}

void EditorOverlayController::prepareOverlayFrameContext(
    RenderFrameContext &frame) {
  if (gizmoController_) {
    gizmoController_->setShadowInspectRequestsEnabled(
        editor_ != nullptr &&
        (editor_->isShadowsWindowOpen() || debugShadowInspectOnClickEnabled()));
    gizmoController_->onFrame(frame);
  }
}

Result<void, std::string>
EditorOverlayController::buildOverlayPass(RenderFrameContext &frame,
                                          RenderGraphBuilder &graph) {
  if (!editor_) {
    return Result<void, std::string>::makeError(
        "EditorOverlayController has no ImGui editor");
  }

  if (frame.settings) {
    editor_->setRenderSettings(*frame.settings);
  }
  editor_->setShadowDebugResources(
      frame.sharedResources.shadowDebugFrameData,
      frame.sharedResources.shadowDebugPreviewTexture);
  editor_->setShadowInspectResult(frame.shadowInspectResult);
  editor_->setFrameIndex(frame.frameIndex);
  editor_->setFrameMetrics(frame.metrics);

  editor_->beginFrame();
  try {
    if (callback_) {
      callback_();
    }
    if (gizmoController_) {
      gizmoController_->drawUi({
          .showControlsWindow = editor_->gizmoControlsWindowOpenState(),
          .controlsWindowTitle = "Gizmo Controls",
          .showLightsWindow = editor_->lightsWindowOpenState(),
          .lightsWindowTitle = "Lights",
      });
    }
  } catch (const std::exception &e) {
    editor_->setFrameDeltaSeconds(frameDeltaSeconds_);
    (void)editor_->endFrame();
    return Result<void, std::string>::makeError(
        std::string("Editor callback threw: ") + e.what());
  } catch (...) {
    editor_->setFrameDeltaSeconds(frameDeltaSeconds_);
    (void)editor_->endFrame();
    return Result<void, std::string>::makeError(
        "Editor callback threw unknown exception");
  }

  editor_->setFrameDeltaSeconds(frameDeltaSeconds_);
  auto passResult = editor_->endFrame();
  if (passResult.hasError()) {
    return Result<void, std::string>::makeError(passResult.error());
  }

  if (frame.settings) {
    pendingRenderSettingsUpdate_ = editor_->renderSettings();
  }

  auto addResult = graph.addGraphicsPass(passResult.value());
  if (addResult.hasError()) {
    return Result<void, std::string>::makeError(addResult.error());
  }

  return Result<void, std::string>::makeResult();
}
} // namespace nuri
