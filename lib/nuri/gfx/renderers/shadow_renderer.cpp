#include "nuri/pch.h"

#include "nuri/gfx/renderers/shadow_renderer.h"

namespace nuri {

ShadowRenderer::ShadowRenderer(GPUDevice &gpu,
                               std::pmr::memory_resource *memory)
    : gpu_(gpu),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()) {}

void ShadowRenderer::publishFrameData(RenderFrameContext &) {}

Result<bool, std::string>
ShadowRenderer::prepareShadowGraphPasses(RenderFrameContext &) {
  hasPreparedShadowDepthPasses_ = false;
  return Result<bool, std::string>::makeResult(true);
}

bool ShadowRenderer::hasPreparedShadowDepthPasses() const noexcept {
  return hasPreparedShadowDepthPasses_;
}

Result<bool, std::string>
ShadowRenderer::appendShadowDepthPasses(RenderFrameContext &,
                                        RenderGraphBuilder &) {
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
