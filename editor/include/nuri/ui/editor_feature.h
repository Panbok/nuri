#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/ui/editor_overlay_controller.h"

namespace nuri {

class RenderPipeline;

class EditorOverlayPass final {
public:
  explicit EditorOverlayPass(EditorOverlayController *controller = nullptr)
      : controller_(controller) {}
  ~EditorOverlayPass() = default;

  EditorOverlayPass(const EditorOverlayPass &) = delete;
  EditorOverlayPass &operator=(const EditorOverlayPass &) = delete;
  EditorOverlayPass(EditorOverlayPass &&) = delete;
  EditorOverlayPass &operator=(EditorOverlayPass &&) = delete;

  void setController(EditorOverlayController *controller) noexcept {
    controller_ = controller;
  }

  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  EditorOverlayController *controller_ = nullptr;
};

EditorOverlayPass *registerEditorOverlayStage(RenderPipeline &pipeline);

} // namespace nuri
