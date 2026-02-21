#pragma once

#include "nuri/gfx/gpu_device.h"

namespace nuri {
class LvkGPUDevice final : public GPUDevice {
public:
  static std::unique_ptr<LvkGPUDevice>
  create(Window &window, const GPUDeviceCreateDesc &desc = {});
  ~LvkGPUDevice() override;

  LvkGPUDevice(const LvkGPUDevice &) = delete;
  LvkGPUDevice &operator=(const LvkGPUDevice &) = delete;
  LvkGPUDevice(LvkGPUDevice &&) = delete;
  LvkGPUDevice &operator=(LvkGPUDevice &&) = delete;

  // Window/Swapchain
  bool shouldClose() const override;
  void getFramebufferSize(int32_t &outWidth, int32_t &outHeight) const override;
  void resizeSwapchain(int32_t width, int32_t height) override;
  Format getSwapchainFormat() const override;
  uint32_t getSwapchainImageIndex() const override;
  uint32_t getSwapchainImageCount() const override;
  double getTime() const override;

  // Resource creation
  Result<BufferHandle, std::string>
  createBuffer(const BufferDesc &desc,
               std::string_view debugName = {}) override;
  Result<TextureHandle, std::string>
  createTexture(const TextureDesc &desc,
                std::string_view debugName = {}) override;
  Result<TextureHandle, std::string>
  createFramebufferTexture(const TextureDesc &desc,
                           std::string_view debugName = {}) override;
  Result<TextureHandle, std::string> createDepthBuffer() override;
  Result<ShaderHandle, std::string>
  createShaderModule(const ShaderDesc &desc) override;
  Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &desc,
                       std::string_view debugName = {}) override;
  Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &desc,
                        std::string_view debugName = {}) override;

  // Resource destruction
  void destroyRenderPipeline(RenderPipelineHandle pipeline) override;
  void destroyComputePipeline(ComputePipelineHandle pipeline) override;
  void destroyBuffer(BufferHandle buffer) override;
  void destroyTexture(TextureHandle texture) override;
  void destroyShaderModule(ShaderHandle shader) override;

  // Resource queries
  bool isValid(BufferHandle h) const override;
  bool isValid(TextureHandle h) const override;
  bool isValid(ShaderHandle h) const override;
  bool isValid(RenderPipelineHandle h) const override;
  bool isValid(ComputePipelineHandle h) const override;
  Format getTextureFormat(TextureHandle h) const override;
  uint32_t getTextureBindlessIndex(TextureHandle h) const override;
  uint64_t getBufferDeviceAddress(BufferHandle h,
                                  size_t offset = 0) const override;
  bool resolveGeometry(GeometryAllocationHandle h,
                       GeometryAllocationView &out) const override;

  // Rendering
  Result<bool, std::string> beginFrame(uint64_t frameIndex) override;
  Result<bool, std::string> submitFrame(const RenderFrame &frame) override;
  Result<GeometryAllocationHandle, std::string>
  allocateGeometry(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
                   std::span<const std::byte> indexBytes, uint32_t indexCount,
                   std::string_view debugName = {}) override;
  void releaseGeometry(GeometryAllocationHandle h) override;
  Result<bool, std::string>
  copyBufferRegions(std::span<const BufferCopyRegion> regions) override;

  // Data updates
  Result<bool, std::string> updateBuffer(BufferHandle buffer,
                                         std::span<const std::byte> data,
                                         size_t offset = 0) override;
  std::byte *getMappedBufferPtr(BufferHandle buffer) override;
  void flushMappedBuffer(BufferHandle buffer, size_t offset,
                         size_t size) override;

  // Shutdown
  void waitIdle() override;

private:
  LvkGPUDevice();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
