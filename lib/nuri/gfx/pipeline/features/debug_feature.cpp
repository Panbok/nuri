#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/debug_feature.h"

namespace nuri {

bool DebugGridPass::isEnabled(const FrameBuildContext &ctx) const {
  return ctx.frame.settings != nullptr && renderer_.hasPreparedDebugGridPass();
}

Result<bool, std::string> DebugGridPass::build(FrameBuildContext &ctx) {
  return renderer_.appendDebugGridPass(ctx.frame, ctx.graph);
}

bool DebugSceneOverlayPass::isEnabled(const FrameBuildContext &ctx) const {
  return ctx.frame.settings != nullptr &&
         renderer_.hasPreparedDebugSceneOverlayPass();
}

Result<bool, std::string> DebugSceneOverlayPass::build(FrameBuildContext &ctx) {
  return renderer_.appendDebugSceneOverlayPass(ctx.frame, ctx.graph);
}

DebugFeature::DebugFeature(GPUDevice &gpu, DebugRendererConfig config,
                           std::pmr::memory_resource *memory)
    : renderer_(
          std::make_unique<DebugRenderer>(gpu, std::move(config), memory)),
      gridPass_(*renderer_), overlayPass_(*renderer_) {}

DebugFeature::~DebugFeature() = default;

Result<bool, std::string>
DebugFeature::publishFrameData(FrameBuildContext &ctx) {
  ctx.frame.transparentContributors.publish(TransparentContributionCollector{
      .user = renderer_.get(),
      .collect =
          [](void *user, RenderFrameContext &frame,
             TransparentStageContribution &out) -> Result<bool, std::string> {
        return static_cast<DebugRenderer *>(user)
            ->buildTransparentStageContribution(frame, out);
      },
  });
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> DebugFeature::prepare(FrameBuildContext &ctx) {
  return renderer_->prepareDebugPasses(ctx.frame);
}

std::span<RenderFeaturePass *const> DebugFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
