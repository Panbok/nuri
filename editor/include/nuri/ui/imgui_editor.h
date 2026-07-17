#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "nuri/core/window.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/scene/camera.h"
#include "nuri/ui/editor.h"
#include "nuri/ui/editor_services.h"

namespace nuri {
class ImGuiEditor final : public Editor {
public:
  static std::unique_ptr<ImGuiEditor>
  create(Window &window, GPUDevice &gpu, const EditorServices &services = {});
  ~ImGuiEditor() override;

  ImGuiEditor(const ImGuiEditor &) = delete;
  ImGuiEditor &operator=(const ImGuiEditor &) = delete;
  ImGuiEditor(ImGuiEditor &&) = delete;
  ImGuiEditor &operator=(ImGuiEditor &&) = delete;

  void setFrameDeltaSeconds(double deltaTime);
  void setFrameIndex(uint64_t frameIndex);
  void setFrameMetrics(const RenderFrameMetrics &metrics);
  void setRenderSettings(const RenderSettings &settings);
  void setShadowDebugResources(const std::optional<ShadowDebugFrameData> &debug,
                               TextureHandle previewTexture);
  void setShadowInspectResult(
      const std::optional<ShadowInspectResult> &inspectResult);
  void syncCameraControllerWidgetStateFromCamera(const Camera &camera);
  void setSceneSelectionUi(std::span<const EditorSceneSelectionOption> scenes,
                           std::string_view selectedSceneId, uint64_t version,
                           std::string_view hotkeyHint = "Toggle Editor: F6",
                           const EditorSceneLoadUiState &load = {});
  void resetSceneUiState();
  void bindScene(RenderScene &scene);
  [[nodiscard]] std::optional<std::string> takeSceneSelectionRequest();
  [[nodiscard]] bool takeSceneCancelRequest();
  [[nodiscard]] bool *gizmoControlsWindowOpenState();
  [[nodiscard]] bool *lightsWindowOpenState();
  [[nodiscard]] bool isGizmoControlsWindowOpen() const;
  [[nodiscard]] bool isLightsWindowOpen() const;
  [[nodiscard]] bool isShadowsWindowOpen() const;
  [[nodiscard]] RenderSettings renderSettings() const;
  bool wantsCaptureKeyboard() const;
  bool wantsCaptureMouse() const;
  void applyDeferredUiActions();
  void beginFrame() override;
  Result<RenderGraphGraphicsPassDesc, std::string> endFrame() override;

private:
  struct Impl;

  ImGuiEditor(Window &window, GPUDevice &gpu, const EditorServices &services);

  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
