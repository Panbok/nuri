#pragma once

#include "nuri/defines.h"
#include "nuri/ui/editor.h"
#include "nuri/ui/editor_services.h"

#include <memory>

namespace nuri {

class NURI_API ImGuizmoController final : public GizmoController {
public:
  explicit ImGuizmoController(const EditorServices &services);
  ~ImGuizmoController() override;

  ImGuizmoController(const ImGuizmoController &) = delete;
  ImGuizmoController &operator=(const ImGuizmoController &) = delete;
  ImGuizmoController(ImGuizmoController &&) = delete;
  ImGuizmoController &operator=(ImGuizmoController &&) = delete;

  bool onInput(const InputEvent &event) override;
  void onFrame(RenderFrameContext &frame) override;
  void drawUi() override;
  void reset() override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] NURI_API std::shared_ptr<GizmoController>
createImGuizmoController(const EditorServices &services);

} // namespace nuri
