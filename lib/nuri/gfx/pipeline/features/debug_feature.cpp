#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/debug_feature.h"

namespace nuri {
namespace {

Result<bool, std::string>
collectDebugTransparentContribution(void *user, RenderFrameContext &frame,
                                    TransparentStageContribution &out) {
  if (user == nullptr) {
    out = {};
    return Result<bool, std::string>::makeResult(true);
  }
  static uint32_t loggedCollectorCalls = 0u;
  if (loggedCollectorCalls < 16u) {
    NURI_LOG_WARNING(
        "DebugFeature::collectDebugTransparentContribution: frame=%llu "
        "transparentStageEnabled=%u",
        static_cast<unsigned long long>(frame.frameIndex),
        frame.sharedResources.transparentStageEnabled ? 1u : 0u);
    ++loggedCollectorCalls;
  }
  return static_cast<DebugRenderer *>(user)->buildTransparentStageContribution(
      frame, out);
}

} // namespace

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
      .collect = &collectDebugTransparentContribution,
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
