#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"

namespace nuri {

class Window;

class NURI_API GPUDevice {
public:
  static std::unique_ptr<GPUDevice> create(Window &window);
  virtual ~GPUDevice() = default;

  GPUDevice(const GPUDevice &) = delete;
  GPUDevice &operator=(const GPUDevice &) = delete;
  GPUDevice(GPUDevice &&) = delete;
  GPUDevice &operator=(GPUDevice &&) = delete;

  // Window/Swapchain
  virtual void pollEvents() = 0;
  virtual bool shouldClose() const = 0;
  virtual void getFramebufferSize(int32_t &outWidth,
                                  int32_t &outHeight) const = 0;
  virtual void resizeSwapchain(int32_t width, int32_t height) = 0;
  virtual Format getSwapchainFormat() const = 0;
  virtual double getTime() const = 0;

  // Resource creation
  virtual Result<BufferHandle, std::string>
  createBuffer(const BufferDesc &desc, std::string_view debugName = {}) = 0;
  virtual Result<TextureHandle, std::string>
  createTexture(const TextureDesc &desc, std::string_view debugName = {}) = 0;
  virtual Result<TextureHandle, std::string>
  createFramebufferTexture(const TextureDesc &desc,
                           std::string_view debugName = {}) = 0;
  virtual Result<TextureHandle, std::string> createDepthBuffer() = 0;
  virtual Result<ShaderHandle, std::string>
  createShaderModule(const ShaderDesc &desc) = 0;
  virtual Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &desc,
                       std::string_view debugName = {}) = 0;
  virtual Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &desc,
                        std::string_view debugName = {}) = 0;

  // Resource queries
  virtual bool isValid(BufferHandle h) const = 0;
  virtual bool isValid(TextureHandle h) const = 0;
  virtual bool isValid(ShaderHandle h) const = 0;
  virtual bool isValid(RenderPipelineHandle h) const = 0;
  virtual bool isValid(ComputePipelineHandle h) const = 0;
  virtual Format getTextureFormat(TextureHandle h) const = 0;
  // Bindless index used by LVK shaders (kTextures2D[]).
  virtual uint32_t getTextureBindlessIndex(TextureHandle h) const = 0;

  // Rendering
  virtual Result<bool, std::string> submitFrame(const RenderFrame &frame) = 0;

  // Shutdown
  virtual void waitIdle() = 0;

protected:
  GPUDevice() = default;
};

} // namespace nuri
