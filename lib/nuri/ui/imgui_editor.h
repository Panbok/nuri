#pragma once

#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/ui/editor.h"

#include <memory>

namespace nuri {
class NURI_API ImGuiEditor final : public Editor {
public:
  static std::unique_ptr<ImGuiEditor> create(Window &window, GPUDevice &gpu);
  ~ImGuiEditor() override;

  ImGuiEditor(const ImGuiEditor &) = delete;
  ImGuiEditor &operator=(const ImGuiEditor &) = delete;
  ImGuiEditor(ImGuiEditor &&) = delete;
  ImGuiEditor &operator=(ImGuiEditor &&) = delete;

  void setFrameDeltaSeconds(double deltaTime);
  void beginFrame() override;
  Result<RenderPass, std::string> endFrame() override;

private:
  struct Impl;

  ImGuiEditor(Window &window, GPUDevice &gpu);

  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
