#pragma once

#include "nuri/core/layer.h"
#include "nuri/defines.h"

#include <memory>

namespace nuri {

class GPUDevice;
class ImGuiEditor;
class Window;

class NURI_API EditorLayer final : public Layer {
public:
  struct UiCallback {
    void (*fn)(void *userData);
    void *userData;

    constexpr UiCallback() noexcept : fn(nullptr), userData(nullptr) {}
    constexpr UiCallback(void (*callback)(void *), void *data) noexcept
        : fn(callback), userData(data) {}
  };

  static std::unique_ptr<EditorLayer>
  create(Window &window, GPUDevice &gpu,
         UiCallback callback = UiCallback{});
  ~EditorLayer() override;

  EditorLayer(const EditorLayer &) = delete;
  EditorLayer &operator=(const EditorLayer &) = delete;
  EditorLayer(EditorLayer &&) = delete;
  EditorLayer &operator=(EditorLayer &&) = delete;

  void setUiCallback(UiCallback callback) { callback_ = callback; }

  void onUpdate(double deltaTime) override;
  Result<bool, std::string> buildRenderPasses(RenderPassList &out) override;

private:
  EditorLayer(Window &window, GPUDevice &gpu, UiCallback callback);

  std::unique_ptr<ImGuiEditor> editor_;
  UiCallback callback_{};
  double frameDeltaSeconds_ = 0.0;
};

} // namespace nuri
