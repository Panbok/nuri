#pragma once

#include "nuri/core/layer.h"
#include "nuri/defines.h"

#include <functional>
#include <memory>

namespace nuri {

class GPUDevice;
class ImGuiEditor;
class Window;

class NURI_API EditorLayer final : public Layer {
public:
  struct UiCallback {
    std::function<void()> callback;

    UiCallback() : callback{} {}
    explicit UiCallback(std::function<void()> cb) : callback(std::move(cb)) {}
  };

  static std::unique_ptr<EditorLayer>
  create(Window &window, GPUDevice &gpu,
         UiCallback callback = UiCallback{});
  ~EditorLayer() override;

  EditorLayer(const EditorLayer &) = delete;
  EditorLayer &operator=(const EditorLayer &) = delete;
  EditorLayer(EditorLayer &&) = delete;
  EditorLayer &operator=(EditorLayer &&) = delete;

  void setUiCallback(UiCallback callback) {
    callback_ = std::move(callback);
  }

  void onUpdate(double deltaTime) override;
  Result<bool, std::string> buildRenderPasses(RenderPassList &out) override;

private:
  EditorLayer(Window &window, GPUDevice &gpu, UiCallback callback);

  std::unique_ptr<ImGuiEditor> editor_;
  UiCallback callback_{};
  double frameDeltaSeconds_ = 0.0;
};

} // namespace nuri
