#include "nuri/gfx/pipeline/features/msaa_resolve_feature.h"

#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/frame_build_context.h"

namespace nuri {
namespace {

[[nodiscard]] bool isMsaa4xSelected(const RenderFrameContext &frame) noexcept {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  return sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
         AntiAliasingMode::MSAA4x;
}

[[nodiscard]] bool
hasBoundMsaaResolveTargets(const FrameBuildContext &ctx) noexcept {
  const AntiAliasingFrameMetrics &metrics = ctx.frame.metrics.antiAliasing;
  return metrics.msaaColorResolveTargetBound &&
         metrics.msaaDepthResolveTargetBound &&
         nuri::isValid(resolveMsaaSceneColorTexture(ctx.frame)) &&
         nuri::isValid(resolveMsaaSceneDepthTexture(ctx.frame)) &&
         nuri::isValid(ctx.shared.sceneColorGraphTexture) &&
         nuri::isValid(ctx.shared.sceneDepthGraphTexture);
}

} // namespace

bool MsaaResolvePass::isEnabled(const FrameBuildContext &ctx) const noexcept {
  return isMsaa4xSelected(ctx.frame) && !hasBoundMsaaResolveTargets(ctx) &&
         nuri::isValid(ctx.shared.msaaSceneColorTexture) &&
         nuri::isValid(ctx.shared.msaaSceneDepthTexture) &&
         nuri::isValid(ctx.shared.sceneColorTexture) &&
         nuri::isValid(ctx.shared.sceneDepthTexture);
}

Result<bool, std::string> MsaaResolvePass::prepare(FrameBuildContext &ctx) {
  return Result<bool, std::string>::makeResult(isEnabled(ctx));
}

Result<bool, std::string> MsaaResolvePass::build(FrameBuildContext &ctx) {
  if (!isEnabled(ctx)) {
    return Result<bool, std::string>::makeResult(false);
  }

  auto msaaColorResult =
      nuri::isValid(ctx.shared.msaaSceneColorGraphTexture)
          ? Result<RenderGraphTextureId, std::string>::makeResult(
                ctx.shared.msaaSceneColorGraphTexture)
          : ctx.graph.importTexture(ctx.shared.msaaSceneColorTexture,
                                    "msaa_resolve_scene_color");
  if (msaaColorResult.hasError()) {
    return Result<bool, std::string>::makeError(msaaColorResult.error());
  }
  auto msaaDepthResult =
      nuri::isValid(ctx.shared.msaaSceneDepthGraphTexture)
          ? Result<RenderGraphTextureId, std::string>::makeResult(
                ctx.shared.msaaSceneDepthGraphTexture)
          : ctx.graph.importTexture(ctx.shared.msaaSceneDepthTexture,
                                    "msaa_resolve_scene_depth");
  if (msaaDepthResult.hasError()) {
    return Result<bool, std::string>::makeError(msaaDepthResult.error());
  }
  ctx.shared.msaaSceneColorGraphTexture = msaaColorResult.value();
  ctx.shared.msaaSceneDepthGraphTexture = msaaDepthResult.value();
  auto colorResolveResult = ctx.graph.importTexture(
      ctx.shared.sceneColorTexture, "msaa_resolve_scene_color_target");
  if (colorResolveResult.hasError()) {
    return Result<bool, std::string>::makeError(colorResolveResult.error());
  }
  auto depthResolveResult = ctx.graph.importTexture(
      ctx.shared.sceneDepthTexture, "msaa_resolve_scene_depth_target");
  if (depthResolveResult.hasError()) {
    return Result<bool, std::string>::makeError(depthResolveResult.error());
  }

  RenderGraphGraphicsPassDesc passDesc{};
  passDesc.color = {.loadOp = LoadOp::Load,
                    .storeOp = StoreOp::MsaaResolve,
                    .resolveMode = ResolveMode::Average};
  passDesc.colorTexture = msaaColorResult.value();
  passDesc.colorResolveTexture = colorResolveResult.value();
  passDesc.depth = {.loadOp = LoadOp::Load,
                    .storeOp = StoreOp::MsaaResolve,
                    .resolveMode = ResolveMode::Min};
  passDesc.depthTexture = msaaDepthResult.value();
  passDesc.depthResolveTexture = depthResolveResult.value();
  passDesc.debugLabel = "MSAA Resolve Pass";
  passDesc.debugColor = 0xff44ccff;
  passDesc.gpuTimingScope = GpuTimingScope::MsaaResolve;
  passDesc.markImplicitOutputSideEffect = true;

  auto addResult = ctx.graph.addGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }

  ctx.shared.sceneColorGraphTexture = colorResolveResult.value();
  ctx.shared.sceneDepthGraphTexture = depthResolveResult.value();
  AntiAliasingFrameMetrics &metrics = ctx.frame.metrics.antiAliasing;
  metrics.msaaResolvePassCount = 1u;
  metrics.msaaColorGraphPublished = true;
  metrics.msaaDepthGraphPublished = true;
  metrics.msaaColorResolveTargetBound = true;
  metrics.msaaDepthResolveTargetBound = true;
  metrics.msaaResolveBandwidthEstimateBytes =
      metrics.msaaColorTextureBytes + metrics.msaaDepthTextureBytes;
  const GpuTimingReport &timingReport = ctx.frame.gpuTiming;
  if (hasGpuTimingScope(timingReport, GpuTimingScope::MsaaResolve)) {
    metrics.msaaResolveGpuTimeMs = timingReport.msaaResolveTimeMs;
    metrics.msaaResolveGpuTimingSourceFrameIndex =
        timingReport.msaaResolveSourceFrameIndex;
    metrics.msaaResolveGpuTimingAvailable = 1u;
  }

  return Result<bool, std::string>::makeResult(true);
}

std::span<RenderFeaturePass *const> MsaaResolveFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
