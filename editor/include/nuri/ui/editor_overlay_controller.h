#pragma once

#include "nuri/core/input_events.h"
#include "nuri/core/result.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/scene/camera.h"
#include "nuri/ui/editor.h"
#include "nuri/ui/editor_services.h"

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace nuri {

class ImGuiEditor;
class GizmoController;

class EditorOverlayController final {
public:
  static std::unique_ptr<EditorOverlayController>
  create(Window &window, GPUDevice &gpu, std::function<void()> callback = {},
         const EditorServices &services = {});
  ~EditorOverlayController();

  EditorOverlayController(const EditorOverlayController &) = delete;
  EditorOverlayController &operator=(const EditorOverlayController &) = delete;
  EditorOverlayController(EditorOverlayController &&) = delete;
  EditorOverlayController &operator=(EditorOverlayController &&) = delete;

  void setUiCallback(std::function<void()> callback) {
    callback_ = std::move(callback);
  }
  void resetControllers();
  void resetSceneUiState();
  void syncCameraControllerWidgetStateFromCamera(const Camera &camera);
  void setScenePresetUi(std::span<const char *const> presetNames,
                        int selectedIndex,
                        std::string_view hotkeyHint = "Toggle Editor: F6");
  [[nodiscard]] std::optional<int> takeScenePresetSelectionRequest();

  bool onInput(const InputEvent &event);
  void onUpdate(double deltaTime);
  void prepareOverlayFrameContext(RenderFrameContext &frame);
  Result<void, std::string> buildOverlayPass(RenderFrameContext &frame,
                                             RenderGraphBuilder &graph);

private:
  EditorOverlayController(Window &window, GPUDevice &gpu,
                          std::function<void()> callback,
                          const EditorServices &services);

  std::unique_ptr<ImGuiEditor> editor_;
  std::function<void()> callback_{};
  std::shared_ptr<GizmoController> gizmoController_{};
  double frameDeltaSeconds_ = 0.0;
};

} // namespace nuri
