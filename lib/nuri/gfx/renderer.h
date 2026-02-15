#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/layers/render_frame_context.h"

#include "nuri/core/layer_stack.h"
#include "nuri/gfx/gpu_device.h"

#include <cstdint>
#include <memory>
#include <memory_resource>
#include <string>
#include <vector>

namespace nuri {

class NURI_API Renderer {
public:
  explicit Renderer(GPUDevice &gpu);
  ~Renderer() = default;

  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;
  Renderer(Renderer &&) = delete;
  Renderer &operator=(Renderer &&) = delete;

  static std::unique_ptr<Renderer> create(GPUDevice &gpu) {
    return std::make_unique<Renderer>(gpu);
  }

  Result<bool, std::string> render(const RenderFrame &frame);
  Result<bool, std::string> render(const RenderFrame &frame, LayerStack &layers,
                                   RenderFrameContext &frameContext);

  void onResize(uint32_t width, uint32_t height);

private:
  GPUDevice &gpu_;
  std::pmr::vector<RenderPass> combinedPasses_;
};

} // namespace nuri
