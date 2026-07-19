#include "nuri/editor_pch.h"

#include "nuri/ui/editor_feature.h"

#include "nuri/gfx/pipeline/render_pipeline.h"

namespace nuri {

bool EditorOverlayPass::isEnabled(const FrameBuildContext &ctx) const {
  (void)ctx;
  return controller_ != nullptr;
}

Result<bool, std::string> EditorOverlayPass::prepare(FrameBuildContext &ctx) {
  controller_->prepareOverlayFrameContext(ctx.frame);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> EditorOverlayPass::build(FrameBuildContext &ctx) {
  auto buildResult = controller_->buildOverlayPass(ctx.frame, ctx.graph);
  if (buildResult.hasError()) {
    return Result<bool, std::string>::makeError(buildResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

EditorOverlayPass *registerEditorOverlayStage(RenderPipeline &pipeline) {
  return pipeline.addStage(std::make_unique<EditorOverlayPass>(),
                           "EditorOverlayFeature", "EditorOverlayPass", true);
}

} // namespace nuri
