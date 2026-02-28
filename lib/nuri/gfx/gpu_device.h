#pragma once

#include <cstddef>
#include <span>
#include <string_view>

#include "nuri/core/result.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"

namespace nuri {

struct TextureReadbackRegion {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t mipLevel = 0;
  uint32_t layer = 0;
};

class NURI_API GPUDevice {
public:
  static std::unique_ptr<GPUDevice>
  create(Window &window, const GPUDeviceCreateDesc &desc = {});
  virtual ~GPUDevice() = default;

  GPUDevice(const GPUDevice &) = delete;
  GPUDevice &operator=(const GPUDevice &) = delete;
  GPUDevice(GPUDevice &&) = delete;
  GPUDevice &operator=(GPUDevice &&) = delete;

  // Window/Swapchain
  virtual bool shouldClose() const = 0;
  virtual void getWindowSize(int32_t &outWidth, int32_t &outHeight) const = 0;
  virtual void getFramebufferSize(int32_t &outWidth,
                                  int32_t &outHeight) const = 0;
  virtual void resizeSwapchain(int32_t width, int32_t height) = 0;
  virtual Format getSwapchainFormat() const = 0;
  virtual uint32_t getSwapchainImageIndex() const = 0;
  virtual uint32_t getSwapchainImageCount() const = 0;
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

  virtual void destroyRenderPipeline(RenderPipelineHandle pipeline) = 0;
  virtual void destroyComputePipeline(ComputePipelineHandle pipeline) = 0;
  virtual void destroyBuffer(BufferHandle buffer) = 0;
  virtual void destroyTexture(TextureHandle texture) = 0;
  virtual void destroyShaderModule(ShaderHandle shader) = 0;

  // Resource queries
  virtual bool isValid(BufferHandle h) const = 0;
  virtual bool isValid(TextureHandle h) const = 0;
  virtual bool isValid(ShaderHandle h) const = 0;
  virtual bool isValid(RenderPipelineHandle h) const = 0;
  virtual bool isValid(ComputePipelineHandle h) const = 0;
  virtual Format getTextureFormat(TextureHandle h) const = 0;
  // Bindless index used by LVK shaders (kTextures2D[]).
  virtual uint32_t getTextureBindlessIndex(TextureHandle h) const = 0;
  // GPU virtual address used by LVK shaders (GL_EXT_buffer_reference).
  virtual uint64_t getBufferDeviceAddress(BufferHandle h,
                                          size_t offset = 0) const = 0;
  virtual bool resolveGeometry(GeometryAllocationHandle h,
                               GeometryAllocationView &out) const = 0;

  // Rendering
  virtual Result<bool, std::string> beginFrame(uint64_t frameIndex) = 0;
  virtual Result<bool, std::string> submitFrame(const RenderFrame &frame) = 0;
  virtual Result<bool, std::string> submitComputeDispatches(
      std::span<const ComputeDispatchItem> dispatches) = 0;
  virtual Result<GeometryAllocationHandle, std::string>
  allocateGeometry(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
                   std::span<const std::byte> indexBytes, uint32_t indexCount,
                   std::string_view debugName = {}) = 0;
  virtual void releaseGeometry(GeometryAllocationHandle h) = 0;
  virtual Result<bool, std::string>
  copyBufferRegions(std::span<const BufferCopyRegion> regions) = 0;

  // Data updates
  virtual Result<bool, std::string>
  updateBuffer(BufferHandle buffer, std::span<const std::byte> data,
               size_t offset = 0) = 0;
  virtual Result<bool, std::string>
  readBuffer(BufferHandle buffer, size_t offset,
             std::span<std::byte> outBytes) = 0;
  // Returns a persistent mapped pointer for host-visible buffers, or nullptr
  // when direct mapping is unavailable for this buffer/backend.
  virtual std::byte *getMappedBufferPtr(BufferHandle buffer) = 0;
  // Flushes a region of a previously mapped buffer to ensure host writes are
  // visible to the GPU. No-op if the buffer is not host-visible or not mapped.
  virtual void flushMappedBuffer(BufferHandle buffer, size_t offset,
                                 size_t size) = 0;
  virtual Result<bool, std::string>
  readTexture(TextureHandle texture, const TextureReadbackRegion &region,
              std::span<std::byte> outBytes) = 0;

  // Shutdown
  virtual void waitIdle() = 0;

protected:
  GPUDevice() = default;
};

} // namespace nuri
