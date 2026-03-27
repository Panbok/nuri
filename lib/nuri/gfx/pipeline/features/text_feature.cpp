#include "nuri/pch.h"

#include "nuri/gfx/pipeline/features/text_feature.h"

namespace nuri {
namespace {

Result<bool, std::string>
collectText3DTransparentContribution(void *user, RenderFrameContext &frame,
                                     TransparentStageContribution &out) {
  if (user == nullptr) {
    out = {};
    return Result<bool, std::string>::makeResult(true);
  }
  return static_cast<TextSystem *>(user)
      ->renderer()
      .buildTransparentStageContribution(frame, out);
}

} // namespace

bool Text3DPass::isEnabled(const FrameBuildContext &ctx) const {
  (void)ctx;
  return true;
}

Result<bool, std::string> Text3DPass::prepare(FrameBuildContext &ctx) {
  auto begin = text_.renderer().beginFrame(ctx.frame.frameIndex);
  if (begin.hasError()) {
    return Result<bool, std::string>::makeError(begin.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> Text3DPass::build(FrameBuildContext &ctx) {
  if (ctx.shared.transparentStageEnabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  const RenderGraphTextureId sceneDepthGraphTexture =
      resolveSceneDepthGraphTexture(ctx.frame);
  const bool hasPriorColorPass = ctx.graph.passCount() > 0u;
  return text_.renderer().append3DGraphPass(
      ctx.frame, ctx.graph, sceneDepthGraphTexture, hasPriorColorPass);
}

Result<bool, std::string>
Text3DFeature::publishFrameData(FrameBuildContext &ctx) {
  ctx.frame.transparentContributors.publish(TransparentContributionCollector{
      .user = &pass_.textSystem(),
      .collect = &collectText3DTransparentContribution,
  });
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> Text3DFeature::prepare(FrameBuildContext &ctx) {
  return pass_.prepare(ctx);
}

std::span<RenderFeaturePass *const> Text3DFeature::passes() noexcept {
  return passes_;
}

bool Text2DPass::isEnabled(const FrameBuildContext &ctx) const {
  (void)ctx;
  return true;
}

Result<bool, std::string> Text2DPass::prepare(FrameBuildContext &ctx) {
  auto begin = text_.renderer().beginFrame(ctx.frame.frameIndex);
  if (begin.hasError()) {
    return Result<bool, std::string>::makeError(begin.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> Text2DPass::build(FrameBuildContext &ctx) {
  const bool hasPriorColorPass = ctx.graph.passCount() > 0u;
  return text_.renderer().append2DGraphPass(ctx.frame, ctx.graph,
                                            hasPriorColorPass);
}

Result<bool, std::string> Text2DFeature::prepare(FrameBuildContext &ctx) {
  return pass_.prepare(ctx);
}

std::span<RenderFeaturePass *const> Text2DFeature::passes() noexcept {
  return passes_;
}

} // namespace nuri
