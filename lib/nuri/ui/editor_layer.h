#pragma once

#include "nuri/core/event_manager.h"
#include "nuri/core/layer.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/ui/editor.h"
#include "nuri/ui/editor_services.h"

#include <functional>
#include <memory>

namespace nuri {

class ImGuiEditor;
class GizmoController;

class NURI_API EditorLayer final : public Layer {
public:
  struct UiCallback {
    std::function<void()> callback;

    UiCallback() : callback{} {}
    explicit UiCallback(std::function<void()> cb) : callback(std::move(cb)) {}
  };

  static std::unique_ptr<EditorLayer>
  create(Window &window, GPUDevice &gpu, EventManager &events,
         UiCallback callback = UiCallback{},
         const EditorServices &services = {});
  ~EditorLayer() override;

  EditorLayer(const EditorLayer &) = delete;
  EditorLayer &operator=(const EditorLayer &) = delete;
  EditorLayer(EditorLayer &&) = delete;
  EditorLayer &operator=(EditorLayer &&) = delete;

  void setUiCallback(UiCallback callback) { callback_ = std::move(callback); }
  void resetControllers();

  bool onInput(const InputEvent &event) override;
  void onUpdate(double deltaTime) override;
  void prepareFrameContext(RenderFrameContext &frame) override;
  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  EditorLayer(Window &window, GPUDevice &gpu, EventManager &events,
              UiCallback callback, const EditorServices &services);

  std::unique_ptr<ImGuiEditor> editor_;
  UiCallback callback_{};
  std::shared_ptr<GizmoController> gizmoController_{};
  double frameDeltaSeconds_ = 0.0;
};

} // namespace nuri
