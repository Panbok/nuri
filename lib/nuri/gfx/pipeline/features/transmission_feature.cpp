#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/transmission_feature.h"

namespace nuri {

bool TransmissionDownsamplePass::isEnabled(const FrameBuildContext &ctx) const {
  return (ctx.frame.settings == nullptr ||
          ctx.frame.settings->transmission.enabled) &&
         renderer_.hasPreparedTransmissionDownsamplePasses();
}

bool TransmissionCopyPass::isEnabled(const FrameBuildContext &ctx) const {
  return (ctx.frame.settings == nullptr ||
          ctx.frame.settings->transmission.enabled) &&
         renderer_.hasPreparedTransmissionCopyPass();
}

bool TransmissionMainPass::isEnabled(const FrameBuildContext &ctx) const {
  return (ctx.frame.settings == nullptr ||
          ctx.frame.settings->transmission.enabled) &&
         renderer_.hasPreparedTransmissionMainPass();
}

Result<bool, std::string>
TransmissionDownsamplePass::build(FrameBuildContext &ctx) {
  return renderer_.appendTransmissionDownsamplePasses(ctx.frame, ctx.graph);
}

Result<bool, std::string> TransmissionCopyPass::build(FrameBuildContext &ctx) {
  return renderer_.appendTransmissionCopyPass(ctx.frame, ctx.graph);
}

Result<bool, std::string> TransmissionMainPass::build(FrameBuildContext &ctx) {
  return renderer_.appendTransmissionMainPass(ctx.frame, ctx.graph);
}

TransmissionFeature::TransmissionFeature(GPUDevice &gpu,
                                         TransmissionRendererConfig config,
                                         std::pmr::memory_resource *memory)
    : renderer_(std::make_unique<TransmissionRenderer>(gpu, std::move(config),
                                                       memory)),
      downsamplePass_(*renderer_), copyPass_(*renderer_),
      mainPass_(*renderer_) {
  renderer_->onAttach();
}

TransmissionFeature::~TransmissionFeature() {
  if (renderer_) {
    renderer_->onDetach();
  }
}

Result<bool, std::string>
TransmissionFeature::publishFrameData(FrameBuildContext &ctx) {
  renderer_->publishFrameData(ctx.frame);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransmissionFeature::prepare(FrameBuildContext &ctx) {
  return renderer_->prepareTransmissionPasses(ctx.frame);
}

std::span<RenderFeaturePass *const> TransmissionFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
