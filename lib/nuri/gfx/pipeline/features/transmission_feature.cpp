#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/transmission_feature.h"

namespace nuri {

bool TransmissionMainPass::isEnabled(const FrameBuildContext &ctx) const {
  return (ctx.frame.settings == nullptr ||
          ctx.frame.settings->transmission.enabled) &&
         renderer_.hasPreparedTransmissionMainPass();
}

Result<bool, std::string> TransmissionMainPass::build(FrameBuildContext &ctx) {
  return renderer_.appendTransmissionMainPass(ctx.frame, ctx.graph);
}

TransmissionFeature::TransmissionFeature(GPUDevice &gpu,
                                         TransmissionRendererConfig config,
                                         std::pmr::memory_resource *memory)
    : renderer_(std::make_unique<TransmissionRenderer>(gpu, std::move(config),
                                                       memory)),
      mainPass_(*renderer_) {
  renderer_->onAttach();
}

TransmissionFeature::~TransmissionFeature() {
  if (renderer_) {
    renderer_->onDetach();
  }
}

Result<bool, std::string> TransmissionFeature::prepare(FrameBuildContext &ctx) {
  return renderer_->prepareTransmissionPasses(ctx.frame);
}

std::span<RenderFeaturePass *const> TransmissionFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
