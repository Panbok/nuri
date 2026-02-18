#pragma once

#include "nuri/core/pmr_scratch.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nuri {

class NURI_API ImGuiGpuRenderer final {
public:
  static std::unique_ptr<ImGuiGpuRenderer> create(GPUDevice &gpu);
  ~ImGuiGpuRenderer() = default;

  ImGuiGpuRenderer(const ImGuiGpuRenderer &) = delete;
  ImGuiGpuRenderer &operator=(const ImGuiGpuRenderer &) = delete;
  ImGuiGpuRenderer(ImGuiGpuRenderer &&) = delete;
  ImGuiGpuRenderer &operator=(ImGuiGpuRenderer &&) = delete;

  Result<RenderPass, std::string> buildRenderPass(Format swapchainFormat,
                                                  uint64_t frameIndex);

private:
  explicit ImGuiGpuRenderer(GPUDevice &gpu);

  Result<bool, std::string> ensurePipeline(Format swapchainFormat);
  Result<bool, std::string> ensureFontTexture();
  Result<bool, std::string>
  ensureBuffers(uint64_t frameIndex, size_t vertexBytes, size_t indexBytes);

  struct FrameBuffers {
    BufferHandle vb{};
    BufferHandle ib{};
    size_t vbCapacityBytes = 0;
    size_t ibCapacityBytes = 0;
  };

  GPUDevice &gpu_;
  ScratchArena scratch_;

  Format pipelineFormat_ = Format::Count;
  ShaderHandle vs_{};
  ShaderHandle fs_{};
  RenderPipelineHandle pipeline_{};

  TextureHandle fontTexture_{};
  uint32_t fontTextureId_ = 0;

  std::vector<FrameBuffers> frames_;

  std::vector<DrawItem> draws_;
  struct PushConstants {
    float lrtb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t textureId = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
    uint32_t _pad2 = 0;
  };
  std::vector<PushConstants> pushConstants_;
};

} // namespace nuri
