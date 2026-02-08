#include "nuri/ui/editor_layer.h"

#include "nuri/core/log.h"
#include "nuri/ui/imgui_editor.h"

namespace nuri {

std::unique_ptr<EditorLayer> EditorLayer::create(Window &window, GPUDevice &gpu,
                                                 UiCallback callback) {
  NURI_LOG_INFO("EditorLayer::create: Creating editor layer");
  return std::unique_ptr<EditorLayer>(new EditorLayer(window, gpu, callback));
}

EditorLayer::EditorLayer(Window &window, GPUDevice &gpu, UiCallback callback)
    : editor_(ImGuiEditor::create(window, gpu)), callback_(callback) {}

EditorLayer::~EditorLayer() = default;

void EditorLayer::onUpdate(double deltaTime) { frameDeltaSeconds_ = deltaTime; }

Result<bool, std::string> EditorLayer::buildRenderPasses(RenderPassList &out) {
  if (!editor_) {
    return Result<bool, std::string>::makeError(
        "EditorLayer has no ImGui editor");
  }

  editor_->beginFrame();
  if (callback_.callback) {
    callback_.callback();
  }

  editor_->setFrameDeltaSeconds(frameDeltaSeconds_);
  auto passResult = editor_->endFrame();
  if (passResult.hasError()) {
    return Result<bool, std::string>::makeError(passResult.error());
  }

  out.push_back(passResult.value());
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
