#include "nuri/gfx/renderer.h"
#include "nuri/gfx/gpu_device.h"

namespace nuri {

Renderer::Renderer(GPUDevice &gpu) : gpu_(gpu) {}

Result<bool, std::string> Renderer::render(const RenderFrame &frame) {
  return gpu_.submitFrame(frame);
}

void Renderer::onResize(uint32_t width, uint32_t height) {
  gpu_.resizeSwapchain(static_cast<int32_t>(width),
                       static_cast<int32_t>(height));
}

} // namespace nuri



