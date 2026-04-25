#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/temporal_aa_feature.h"

#include "nuri/gfx/frame/render_frame_context.h"

namespace nuri {

bool TemporalAANoopPass::isEnabled(const FrameBuildContext &ctx) const {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  return sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
         AntiAliasingMode::TAA;
}

Result<bool, std::string> TemporalAANoopPass::prepare(FrameBuildContext &ctx) {
  (void)ctx;
  return Result<bool, std::string>::makeResult(false);
}

Result<bool, std::string> TemporalAANoopPass::build(FrameBuildContext &ctx) {
  (void)ctx;
  return Result<bool, std::string>::makeResult(false);
}

std::span<RenderFeaturePass *const> TemporalAAFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
