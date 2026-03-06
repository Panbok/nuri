#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/layers/render_frame_context.h"

#include "nuri/core/layer_stack.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/resources/gpu/resource_manager.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <memory_resource>
#include <string>

namespace nuri {

class NURI_API Renderer {
public:
  explicit Renderer(GPUDevice &gpu, std::pmr::memory_resource &memory);
  ~Renderer() = default;

  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;
  Renderer(Renderer &&) = delete;
  Renderer &operator=(Renderer &&) = delete;

  static std::unique_ptr<Renderer> create(GPUDevice &gpu,
                                          std::pmr::memory_resource &memory) {
    return std::make_unique<Renderer>(gpu, memory);
  }

  Result<bool, std::string> render();
  Result<bool, std::string> render(LayerStack &layers,
                                   RenderFrameContext &frameContext);

  void onResize(uint32_t width, uint32_t height);
  [[nodiscard]] ResourceManager &resources() noexcept { return resources_; }
  [[nodiscard]] const ResourceManager &resources() const noexcept {
    return resources_;
  }

private:
  [[nodiscard]] std::filesystem::path
  makeRenderGraphDumpPath(uint64_t frameIndex) const;

  [[nodiscard]] Result<bool, std::string>
  compileAndExecuteRenderGraph(uint64_t frameIndex);

  GPUDevice &gpu_;
  ResourceManager resources_;
  RenderGraphBuilder renderGraphBuilder_;
  RenderGraphExecutor renderGraphExecutor_;
  std::filesystem::path renderGraphDumpDirectory_;
  bool suppressInferredSideEffects_ = false;
  uint64_t standaloneFrameIndex_ = 0;
};

} // namespace nuri
