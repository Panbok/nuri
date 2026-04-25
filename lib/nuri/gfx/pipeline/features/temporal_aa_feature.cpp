#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/temporal_aa_feature.h"

#include "nuri/gfx/frame/render_frame_context.h"

namespace nuri {

bool TemporalAAMotionVectorClearPass::isEnabled(
    const FrameBuildContext &ctx) const {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  return sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
         AntiAliasingMode::TAA;
}

Result<bool, std::string>
TemporalAAMotionVectorClearPass::prepare(FrameBuildContext &ctx) {
  (void)ctx;
  return Result<bool, std::string>::makeResult(false);
}

Result<bool, std::string>
TemporalAAMotionVectorClearPass::build(FrameBuildContext &ctx) {
  if (!nuri::isValid(ctx.shared.motionVectorTexture)) {
    return Result<bool, std::string>::makeResult(false);
  }

  auto motionVectorImport = ctx.graph.importTexture(
      ctx.shared.motionVectorTexture, "taa_motion_vectors");
  if (motionVectorImport.hasError()) {
    return Result<bool, std::string>::makeError(motionVectorImport.error());
  }
  ctx.shared.motionVectorGraphTexture = motionVectorImport.value();
  ctx.frame.metrics.antiAliasing.motionVectorGraphPublished = true;

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color =
      AttachmentColor{.loadOp = LoadOp::Clear,
                      .storeOp = StoreOp::Store,
                      .clearColor = kFrameCompositionMotionVectorClearValue};
  passDesc.colorTexture = ctx.shared.motionVectorGraphTexture;
  passDesc.debugLabel = "Temporal AA Motion Vector Clear";
  passDesc.debugColor = 0xff44aaff;

  auto addResult = ctx.graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  aaMetrics.motionVectorClearPassCount = 1u;
  aaMetrics.motionVectorClearBytes = aaMetrics.motionVectorTextureBytes;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TemporalAAFeature::publishFrameData(FrameBuildContext &ctx) {
  const RenderSettings &settings = renderSettingsOrDefault(ctx.frame);
  if (sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
      AntiAliasingMode::TAA) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::MotionVectors;
  }
  return Result<bool, std::string>::makeResult(true);
}

std::span<RenderFeaturePass *const> TemporalAAFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
