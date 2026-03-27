#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/transparent_feature.h"

namespace nuri {

bool TransparentMainPass::isEnabled(const FrameBuildContext &ctx) const {
  return ctx.frame.settings == nullptr ||
         ctx.frame.settings->transparent.enabled;
}

Result<bool, std::string> TransparentMainPass::build(FrameBuildContext &ctx) {
  return renderer_.appendTransparentMainPass(ctx.frame, ctx.graph);
}

bool TransparentPickPass::isEnabled(const FrameBuildContext &ctx) const {
  return ctx.frame.settings == nullptr ||
         ctx.frame.settings->transparent.enabled;
}

Result<bool, std::string> TransparentPickPass::build(FrameBuildContext &ctx) {
  return renderer_.appendTransparentPickPass(ctx.frame, ctx.graph);
}

TransparentFeature::TransparentFeature(GPUDevice &gpu,
                                       TransparentRendererConfig config,
                                       std::pmr::memory_resource *memory)
    : renderer_(std::make_unique<TransparentRenderer>(gpu, std::move(config),
                                                      memory)),
      mainPass_(*renderer_), pickPass_(*renderer_) {
  renderer_->onAttach();
}

TransparentFeature::~TransparentFeature() {
  if (renderer_) {
    renderer_->onDetach();
  }
}

Result<bool, std::string>
TransparentFeature::publishFrameData(FrameBuildContext &ctx) {
  renderer_->publishFrameData(ctx.frame);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransparentFeature::prepare(FrameBuildContext &ctx) {
  return renderer_->prepareTransparentPasses(ctx.frame);
}

std::span<RenderFeaturePass *const> TransparentFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
