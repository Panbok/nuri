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
#include <string>
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
  void bindScene(RenderScene &scene);
  void syncCameraControllerWidgetStateFromCamera(const Camera &camera);
  void setSceneSelectionUi(std::span<const EditorSceneSelectionOption> scenes,
                           std::string_view selectedSceneId, uint64_t version,
                           std::string_view hotkeyHint = "Toggle Editor: F6",
                           const EditorSceneLoadUiState &load = {});
  [[nodiscard]] std::optional<std::string> takeSceneSelectionRequest();
  [[nodiscard]] bool takeSceneCancelRequest();
  [[nodiscard]] std::optional<RenderSettings> takeRenderSettingsUpdate();

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
  std::optional<RenderSettings> pendingRenderSettingsUpdate_{};
  double frameDeltaSeconds_ = 0.0;
};

} // namespace nuri
