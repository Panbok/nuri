#include "nuri/pch.h"

#include "nuri/text/text_layer_3d.h"

namespace nuri {

std::unique_ptr<TextLayer3D> TextLayer3D::create(const CreateDesc &desc) {
  return std::unique_ptr<TextLayer3D>(new TextLayer3D(desc));
}

TextLayer3D::TextLayer3D(const CreateDesc &desc) : text_(desc.text) {}

void TextLayer3D::publishFrameData(RenderFrameContext &frame) {
  frame.transparentStage.registerProducer(this);
}

Result<bool, std::string>
TextLayer3D::buildRenderGraph(RenderFrameContext &frame,
                              RenderGraphBuilder &graph) {
  if (const bool *transparentStageEnabled =
          frame.channels.tryGet<bool>(kFrameChannelTransparentStageEnabled);
      transparentStageEnabled != nullptr && *transparentStageEnabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto begin = text_.renderer().beginFrame(frame.frameIndex);
  if (begin.hasError()) {
    return Result<bool, std::string>::makeError(begin.error());
  }

  const bool hasPriorColorPass = graph.passCount() > 0;
  RenderGraphTextureId sceneDepthGraphTexture{};
  if (const RenderGraphTextureId *publishedSceneDepth =
          frame.channels.tryGet<RenderGraphTextureId>(
              kFrameChannelSceneDepthGraphTexture);
      publishedSceneDepth != nullptr) {
    sceneDepthGraphTexture = *publishedSceneDepth;
  }
  return text_.renderer().append3DGraphPass(frame, graph,
                                            sceneDepthGraphTexture,
                                            hasPriorColorPass);
}

Result<bool, std::string>
TextLayer3D::buildTransparentStageContribution(RenderFrameContext &frame,
                                               TransparentStageContribution &out) {
  auto begin = text_.renderer().beginFrame(frame.frameIndex);
  if (begin.hasError()) {
    return Result<bool, std::string>::makeError(begin.error());
  }
  return text_.renderer().buildTransparentStageContribution(frame, out);
}

} // namespace nuri
