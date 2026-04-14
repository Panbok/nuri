#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"

#include <memory_resource>
#include <string>

namespace nuri {

class NURI_API ShadowRenderer {
public:
  explicit ShadowRenderer(GPUDevice &gpu, std::pmr::memory_resource *memory =
                                              std::pmr::get_default_resource());
  ~ShadowRenderer() = default;

  ShadowRenderer(const ShadowRenderer &) = delete;
  ShadowRenderer &operator=(const ShadowRenderer &) = delete;
  ShadowRenderer(ShadowRenderer &&) = delete;
  ShadowRenderer &operator=(ShadowRenderer &&) = delete;

  void publishFrameData(RenderFrameContext &frame);
  Result<bool, std::string> prepareShadowGraphPasses(RenderFrameContext &frame);
  [[nodiscard]] bool hasPreparedShadowDepthPasses() const noexcept;
  Result<bool, std::string> appendShadowDepthPasses(RenderFrameContext &frame,
                                                    RenderGraphBuilder &graph);

private:
  GPUDevice &gpu_;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  bool hasPreparedShadowDepthPasses_ = false;
};

} // namespace nuri
