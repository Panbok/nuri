#include "nuri/pch.h"

#include "nuri/text/text_layer_2d.h"

namespace nuri {

std::unique_ptr<TextLayer2D> TextLayer2D::create(const CreateDesc &desc) {
  return std::unique_ptr<TextLayer2D>(new TextLayer2D(desc));
}

TextLayer2D::TextLayer2D(const CreateDesc &desc)
    : text_(desc.text) {
}

Result<bool, std::string>
TextLayer2D::buildRenderGraph(RenderFrameContext &frame,
                              RenderGraphBuilder &graph) {
  auto begin = text_.renderer().beginFrame(frame.frameIndex);
  if (begin.hasError()) {
    return Result<bool, std::string>::makeError(begin.error());
  }

  const bool hasPriorColorPass = graph.passCount() > 0;

  return text_.renderer().append2DGraphPass(frame, graph, hasPriorColorPass);
}

} // namespace nuri
