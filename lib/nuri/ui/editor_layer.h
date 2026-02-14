#pragma once

#include "nuri/core/event_manager.h"
#include "nuri/core/layer.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/ui/imgui_editor.h"

#include <functional>
#include <memory>

namespace nuri {

class NURI_API EditorLayer final : public Layer {
public:
  struct UiCallback {
    std::function<void()> callback;

    UiCallback() : callback{} {}
    explicit UiCallback(std::function<void()> cb) : callback(std::move(cb)) {}
  };

  static std::unique_ptr<EditorLayer>
  create(Window &window, GPUDevice &gpu, EventManager &events,
         UiCallback callback = UiCallback{});
  ~EditorLayer() override;

  EditorLayer(const EditorLayer &) = delete;
  EditorLayer &operator=(const EditorLayer &) = delete;
  EditorLayer(EditorLayer &&) = delete;
  EditorLayer &operator=(EditorLayer &&) = delete;

  void setUiCallback(UiCallback callback) { callback_ = std::move(callback); }

  bool onInput(const InputEvent &event) override;
  void onUpdate(double deltaTime) override;
  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  EditorLayer(Window &window, GPUDevice &gpu, EventManager &events,
              UiCallback callback);

  std::unique_ptr<ImGuiEditor> editor_;
  UiCallback callback_{};
  double frameDeltaSeconds_ = 0.0;
};

} // namespace nuri
