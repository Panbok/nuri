#include "nuri/editor_pch.h"

#include "nuri/ui/editor_feature.h"

namespace nuri {

bool EditorOverlayPass::isEnabled(const FrameBuildContext &ctx) const {
  (void)ctx;
  return controller_ != nullptr;
}

Result<bool, std::string> EditorOverlayPass::prepare(FrameBuildContext &ctx) {
  if (controller_ == nullptr) {
    return Result<bool, std::string>::makeResult(true);
  }
  controller_->prepareOverlayFrameContext(ctx.frame);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> EditorOverlayPass::build(FrameBuildContext &ctx) {
  if (controller_ == nullptr) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto buildResult = controller_->buildOverlayPass(ctx.frame, ctx.graph);
  if (buildResult.hasError()) {
    return Result<bool, std::string>::makeError(buildResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
EditorOverlayFeature::prepare(FrameBuildContext &ctx) {
  (void)ctx;
  return Result<bool, std::string>::makeResult(true);
}

std::span<RenderFeaturePass *const> EditorOverlayFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
