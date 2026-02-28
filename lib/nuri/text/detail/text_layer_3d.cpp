#include "nuri/pch.h"

#include "nuri/text/text_layer_3d.h"

namespace nuri {

std::unique_ptr<TextLayer3D> TextLayer3D::create(const CreateDesc &desc) {
  return std::unique_ptr<TextLayer3D>(new TextLayer3D(desc));
}

TextLayer3D::TextLayer3D(const CreateDesc &desc) : text_(desc.text) {}

Result<bool, std::string>
TextLayer3D::buildRenderPasses(RenderFrameContext &frame, RenderPassList &out) {
  auto begin = text_.renderer().beginFrame(frame.frameIndex);
  if (begin.hasError()) {
    return Result<bool, std::string>::makeError(begin.error());
  }
  return text_.renderer().append3DPasses(frame, out);
}

} // namespace nuri
