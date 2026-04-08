#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/opaque_feature.h"

namespace nuri {

bool OpaqueMainPass::isEnabled(const FrameBuildContext &ctx) const {
  return (ctx.frame.settings == nullptr ||
          ctx.frame.settings->opaque.enabled) &&
         renderer_.hasPreparedOpaqueMainPasses();
}

Result<bool, std::string> OpaqueMainPass::build(FrameBuildContext &ctx) {
  return renderer_.appendOpaqueMainPasses(ctx.frame, ctx.graph);
}

bool OpaquePickPass::isEnabled(const FrameBuildContext &ctx) const {
  return (ctx.frame.settings == nullptr ||
          ctx.frame.settings->opaque.enabled) &&
         renderer_.hasPreparedOpaquePickPasses();
}

Result<bool, std::string> OpaquePickPass::build(FrameBuildContext &ctx) {
  return renderer_.appendOpaquePickPasses(ctx.frame, ctx.graph);
}

OpaqueFeature::OpaqueFeature(GPUDevice &gpu, OpaqueRendererConfig config,
                             std::pmr::memory_resource *memory)
    : renderer_(
          std::make_unique<OpaqueRenderer>(gpu, std::move(config), memory)),
      mainPass_(*renderer_), pickPass_(*renderer_),
      passes_{&pickPass_, &mainPass_} {
  renderer_->onAttach();
}

OpaqueFeature::~OpaqueFeature() {
  if (renderer_) {
    renderer_->onDetach();
  }
}

Result<bool, std::string> OpaqueFeature::prepare(FrameBuildContext &ctx) {
  return renderer_->prepareOpaqueGraphPasses(ctx.frame);
}

Result<bool, std::string>
OpaqueFeature::publishFrameData(FrameBuildContext &ctx) {
  renderer_->publishFrameData(ctx.frame);
  return Result<bool, std::string>::makeResult(true);
}

std::span<RenderFeaturePass *const> OpaqueFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
