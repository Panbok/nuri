#include "nuri/editor_pch.h"

#include "nuri/ui/editor_overlay_controller.h"

#include "nuri/core/log.h"
#include "nuri/ui/imgui_editor.h"
#include "nuri/ui/imgui_gizmo_controller.h"

namespace nuri {
std::unique_ptr<EditorOverlayController>
EditorOverlayController::create(Window &window, GPUDevice &gpu,
                                EventManager &events, UiCallback callback,
                                const EditorServices &services) {
  NURI_LOG_DEBUG(
      "EditorOverlayController::create: Creating editor overlay controller");
  return std::unique_ptr<EditorOverlayController>(new EditorOverlayController(
      window, gpu, events, std::move(callback), services));
}

EditorOverlayController::EditorOverlayController(Window &window, GPUDevice &gpu,
                                                 EventManager &events,
                                                 UiCallback callback,
                                                 const EditorServices &services)
    : editor_(ImGuiEditor::create(window, gpu, events, services)),
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

void EditorOverlayController::syncCameraControllerWidgetStateFromCamera(
    const Camera &camera) {
  if (editor_) {
    editor_->syncCameraControllerWidgetStateFromCamera(camera);
  }
}

void EditorOverlayController::setScenePresetUi(
    std::span<const char *const> presetNames, int selectedIndex,
    std::string_view hotkeyHint) {
  if (editor_) {
    editor_->setScenePresetUi(presetNames, selectedIndex, hotkeyHint);
  }
}

std::optional<int> EditorOverlayController::takeScenePresetSelectionRequest() {
  return editor_ ? editor_->takeScenePresetSelectionRequest() : std::nullopt;
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
    break;
  }
  return false;
}

void EditorOverlayController::onUpdate(double deltaTime) {
  frameDeltaSeconds_ = deltaTime;
}

void EditorOverlayController::prepareOverlayFrameContext(
    RenderFrameContext &frame) {
  if (gizmoController_) {
    gizmoController_->onFrame(frame);
  }
}

Result<bool, std::string>
EditorOverlayController::buildOverlayPass(RenderFrameContext &frame,
                                          RenderGraphBuilder &graph) {
  if (!editor_) {
    return Result<bool, std::string>::makeError(
        "EditorOverlayController has no ImGui editor");
  }

  if (frame.settings) {
    editor_->setRenderSettings(*frame.settings);
  }
  editor_->setFrameIndex(frame.frameIndex);
  editor_->setFrameMetrics(frame.metrics);

  editor_->beginFrame();
  try {
    if (callback_.callback) {
      callback_.callback();
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
    return Result<bool, std::string>::makeError(
        std::string("Editor callback threw: ") + e.what());
  } catch (...) {
    editor_->setFrameDeltaSeconds(frameDeltaSeconds_);
    (void)editor_->endFrame();
    return Result<bool, std::string>::makeError(
        "Editor callback threw unknown exception");
  }

  editor_->setFrameDeltaSeconds(frameDeltaSeconds_);
  auto passResult = editor_->endFrame();
  if (passResult.hasError()) {
    return Result<bool, std::string>::makeError(passResult.error());
  }

  if (frame.settings) {
    *frame.settings = editor_->renderSettings();
  }

  auto addResult = graph.addGraphicsPass(passResult.value());
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }

  return Result<bool, std::string>::makeResult(true);
}
} // namespace nuri
