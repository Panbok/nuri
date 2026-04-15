#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/shadow_feature.h"

namespace nuri {

bool ShadowDepthPass::isEnabled(const FrameBuildContext &ctx) const {
  return ctx.frame.settings != nullptr && ctx.frame.settings->shadow.enabled &&
         renderer_.hasPreparedShadowDepthPasses();
}

Result<bool, std::string> ShadowDepthPass::build(FrameBuildContext &ctx) {
  return renderer_.appendShadowDepthPasses(ctx.frame, ctx.graph);
}

ShadowFeature::ShadowFeature(GPUDevice &gpu, std::pmr::memory_resource *memory)
    : renderer_(std::make_unique<ShadowRenderer>(gpu, memory)),
      depthPass_(*renderer_) {}

ShadowFeature::ShadowFeature(GPUDevice &gpu, ShadowRendererConfig config,
                             std::pmr::memory_resource *memory)
    : renderer_(
          std::make_unique<ShadowRenderer>(gpu, std::move(config), memory)),
      depthPass_(*renderer_) {}

Result<bool, std::string>
ShadowFeature::publishFrameData(FrameBuildContext &ctx) {
  return renderer_->publishFrameData(ctx.frame);
}

Result<bool, std::string> ShadowFeature::prepare(FrameBuildContext &ctx) {
  return renderer_->prepareShadowGraphPasses(ctx.frame);
}

std::span<RenderFeaturePass *const> ShadowFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
