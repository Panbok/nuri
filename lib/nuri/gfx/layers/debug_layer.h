#pragma once

#include "nuri/core/layer.h"
#include "nuri/defines.h"
#include "nuri/gfx/debug_draw_3d.h"
#include "nuri/gfx/gpu_device.h"

#include <memory>

namespace nuri {

class NURI_API DebugLayer final : public Layer {
public:
  explicit DebugLayer(GPUDevice &gpu);
  ~DebugLayer() override = default;

  DebugLayer(const DebugLayer &) = delete;
  DebugLayer &operator=(const DebugLayer &) = delete;
  DebugLayer(DebugLayer &&) = delete;
  DebugLayer &operator=(DebugLayer &&) = delete;

  static std::unique_ptr<DebugLayer> create(GPUDevice &gpu) {
    return std::make_unique<DebugLayer>(gpu);
  }

  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  std::unique_ptr<DebugDraw3D> debugDraw3D_;
};

} // namespace nuri
