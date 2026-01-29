#pragma once

#include "nuri/pch.h"

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/platform/gpu_descriptors.h"
#include "nuri/platform/gpu_render_types.h"
#include "nuri/platform/gpu_types.h"

namespace nuri {

class Window;

class NURI_API GPUDevice {
public:
  static std::unique_ptr<GPUDevice> create(Window &window);
  ~GPUDevice();

  GPUDevice(const GPUDevice &) = delete;
  GPUDevice &operator=(const GPUDevice &) = delete;
  GPUDevice(GPUDevice &&) = delete;
  GPUDevice &operator=(GPUDevice &&) = delete;

  // Window/Swapchain
  void pollEvents();
  bool shouldClose() const;
  void getFramebufferSize(int32_t &outWidth, int32_t &outHeight) const;
  void resizeSwapchain(int32_t width, int32_t height);
  Format getSwapchainFormat() const;
  double getTime() const;

  // Resource creation
  Result<BufferHandle, std::string>
  createBuffer(const BufferDesc &desc, std::string_view debugName = {});
  Result<TextureHandle, std::string>
  createTexture(const TextureDesc &desc, std::string_view debugName = {});
  Result<ShaderHandle, std::string> createShaderModule(const ShaderDesc &desc);
  Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &desc,
                       std::string_view debugName = {});
  Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &desc,
                        std::string_view debugName = {});

  // Resource queries
  bool isValid(BufferHandle h) const;
  bool isValid(TextureHandle h) const;
  bool isValid(ShaderHandle h) const;
  bool isValid(RenderPipelineHandle h) const;
  bool isValid(ComputePipelineHandle h) const;
  Format getTextureFormat(TextureHandle h) const;

  // Rendering
  Result<bool, std::string> submitFrame(const RenderFrame &frame);

  // Shutdown
  void waitIdle();

private:
  GPUDevice();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
