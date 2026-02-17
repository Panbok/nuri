#include "nuri/gfx/renderer.h"

#include <cassert>

#include "nuri/core/layer_stack.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_device.h"

namespace nuri {

Renderer::Renderer(GPUDevice &gpu, std::pmr::memory_resource *memory)
    : gpu_(gpu), combinedPasses_(memory) {
  NURI_ASSERT(memory != nullptr, "Memory resource cannot be nullptr");
  NURI_LOG_DEBUG("Renderer::Renderer: Renderer created");
}

Result<bool, std::string> Renderer::render(const RenderFrame &frame) {
  NURI_PROFILER_FUNCTION();
  auto frameResult = gpu_.beginFrame(standaloneFrameIndex_++);
  if (frameResult.hasError()) {
    return frameResult;
  }
  return gpu_.submitFrame(frame);
}

Result<bool, std::string> Renderer::render(const RenderFrame &frame,
                                           LayerStack &layers,
                                           RenderFrameContext &frameContext) {
  NURI_PROFILER_FUNCTION();
  auto frameResult = gpu_.beginFrame(frameContext.frameIndex);
  if (frameResult.hasError()) {
    return frameResult;
  }

  if (layers.empty()) {
    return gpu_.submitFrame(frame);
  }

  combinedPasses_.clear();
  combinedPasses_.reserve(frame.passes.size() + layers.size());
  combinedPasses_.insert(combinedPasses_.end(), frame.passes.begin(),
                         frame.passes.end());

  auto layerResult = layers.appendRenderPasses(frameContext, combinedPasses_);
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
