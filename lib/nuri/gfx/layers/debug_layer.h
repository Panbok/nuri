#pragma once

#include <memory>
#include <memory_resource>

#include "nuri/core/layer.h"
#include "nuri/defines.h"
#include "nuri/gfx/debug_draw_3d.h"
#include "nuri/gfx/gpu_device.h"

namespace nuri {

class NURI_API DebugLayer final : public Layer {
public:
  explicit DebugLayer(
      GPUDevice &gpu,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~DebugLayer() override = default;

  DebugLayer(const DebugLayer &) = delete;
  DebugLayer &operator=(const DebugLayer &) = delete;
  DebugLayer(DebugLayer &&) = delete;
  DebugLayer &operator=(DebugLayer &&) = delete;

  static std::unique_ptr<DebugLayer> create(
      GPUDevice &gpu,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource()) {
    return std::make_unique<DebugLayer>(gpu, memory);
  }

  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  std::unique_ptr<DebugDraw3D> debugDraw3D_;
};

} // namespace nuri
