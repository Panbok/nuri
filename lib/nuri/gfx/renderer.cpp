#include "nuri/gfx/renderer.h"

#include "nuri/core/profiling.h"
#include "nuri/core/layer_stack.h"
#include "nuri/gfx/gpu_device.h"

namespace nuri {

Renderer::Renderer(GPUDevice &gpu) : gpu_(gpu) {}

Result<bool, std::string> Renderer::render(const RenderFrame &frame) {
  NURI_PROFILER_FUNCTION();
  return gpu_.submitFrame(frame);
}

Result<bool, std::string> Renderer::render(const RenderFrame &frame,
                                           LayerStack &layers) {
  NURI_PROFILER_FUNCTION();
  if (layers.empty()) {
    return render(frame);
  }

  combinedPasses_.clear();
  combinedPasses_.reserve(frame.passes.size() + layers.size());
  if (!frame.passes.empty()) {
    combinedPasses_.insert(combinedPasses_.end(), frame.passes.begin(),
                           frame.passes.end());
  }

  auto layerResult = layers.appendRenderPasses(combinedPasses_);
  if (layerResult.hasError()) {
    return Result<bool, std::string>::makeError(layerResult.error());
  }

  if (combinedPasses_.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  const RenderFrame combinedFrame{
      .passes = std::span<const RenderPass>(combinedPasses_.data(),
                                            combinedPasses_.size()),
  };

  return gpu_.submitFrame(combinedFrame);
}

void Renderer::onResize(uint32_t width, uint32_t height) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  gpu_.resizeSwapchain(static_cast<int32_t>(width),
                       static_cast<int32_t>(height));
}

} // namespace nuri
