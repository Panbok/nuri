#pragma once

#include <memory>

#include "nuri/core/event_manager.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/layers/render_frame_context.h"
#include "nuri/ui/editor.h"
#include "nuri/ui/editor_services.h"

namespace nuri {
class NURI_API ImGuiEditor final : public Editor {
public:
  static std::unique_ptr<ImGuiEditor> create(Window &window, GPUDevice &gpu,
                                             EventManager &events,
                                             const EditorServices &services = {});
  ~ImGuiEditor() override;

  ImGuiEditor(const ImGuiEditor &) = delete;
  ImGuiEditor &operator=(const ImGuiEditor &) = delete;
  ImGuiEditor(ImGuiEditor &&) = delete;
  ImGuiEditor &operator=(ImGuiEditor &&) = delete;

  void setFrameDeltaSeconds(double deltaTime);
  void setFrameIndex(uint64_t frameIndex);
  void setFrameMetrics(const RenderFrameMetrics &metrics);
  void setRenderSettings(const RenderSettings &settings);
  [[nodiscard]] RenderSettings renderSettings() const;
  bool wantsCaptureKeyboard() const;
  bool wantsCaptureMouse() const;
  void beginFrame() override;
  Result<RenderPass, std::string> endFrame() override;

private:
  struct Impl;

  ImGuiEditor(Window &window, GPUDevice &gpu, EventManager &events,
              const EditorServices &services);

  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
