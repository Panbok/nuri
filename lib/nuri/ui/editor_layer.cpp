#include "nuri/ui/editor_layer.h"

#include "nuri/core/log.h"
#include "nuri/ui/imgui_editor.h"

#include <exception>
#include <string>

namespace nuri {

std::unique_ptr<EditorLayer> EditorLayer::create(Window &window, GPUDevice &gpu,
                                                 EventManager &events,
                                                 UiCallback callback) {
  NURI_LOG_DEBUG("EditorLayer::create: Creating editor layer");
  return std::unique_ptr<EditorLayer>(
      new EditorLayer(window, gpu, events, callback));
}

EditorLayer::EditorLayer(Window &window, GPUDevice &gpu, EventManager &events,
                         UiCallback callback)
    : editor_(ImGuiEditor::create(window, gpu, events)), callback_(callback) {}

EditorLayer::~EditorLayer() = default;

bool EditorLayer::onInput(const InputEvent &event) {
  if (!editor_) {
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
    return editor_->wantsCaptureMouse();
  case InputEventType::Focus:
    break;
  }
  return false;
}

void EditorLayer::onUpdate(double deltaTime) { frameDeltaSeconds_ = deltaTime; }

Result<bool, std::string>
EditorLayer::buildRenderPasses(RenderFrameContext &frame, RenderPassList &out) {
  if (!editor_) {
    return Result<bool, std::string>::makeError(
        "EditorLayer has no ImGui editor");
  }

  if (frame.settings) {
    editor_->setRenderSettings(*frame.settings);
  }

  editor_->beginFrame();
  try {
    if (callback_.callback) {
      callback_.callback();
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

  out.push_back(passResult.value());
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
