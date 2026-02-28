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
TextLayer2D::buildRenderPasses(RenderFrameContext &frame, RenderPassList &out) {
  auto begin = text_.renderer().beginFrame(frame.frameIndex);
  if (begin.hasError()) {
    return Result<bool, std::string>::makeError(begin.error());
  }
  return text_.renderer().append2DPasses(frame, out);
}

} // namespace nuri
