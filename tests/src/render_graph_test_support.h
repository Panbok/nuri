#pragma once

#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::test_support {

bool sameHandle(BufferHandle lhs, BufferHandle rhs);
bool sameHandle(TextureHandle lhs, TextureHandle rhs);
bool sameBuffer(BufferHandle lhs, BufferHandle rhs);
bool sameTexture(TextureHandle lhs, TextureHandle rhs);

BufferDesc makeTransientBufferDesc(size_t bytes);
TextureDesc makeTransientTextureDesc(Format format, uint32_t width,
                                     uint32_t height);
RenderPass makeTestPass(std::string_view label, TextureHandle colorTexture = {});

Result<RenderGraphPassId, std::string>
addTestGraphicsPass(RenderGraphBuilder &builder, const RenderPass &pass,
                    std::string_view debugName = {},
                    bool autoBindPassResources = true);

class EnvVarGuard {
public:
  EnvVarGuard(std::string_view name, std::string_view value);
  ~EnvVarGuard();

  EnvVarGuard(const EnvVarGuard &) = delete;
  EnvVarGuard &operator=(const EnvVarGuard &) = delete;
  EnvVarGuard(EnvVarGuard &&) = delete;
  EnvVarGuard &operator=(EnvVarGuard &&) = delete;

private:
  std::string name_;
  std::string oldValue_{};
  bool hadOldValue_ = false;
};

class FakeGPUDeviceBase : public GPUDevice {
public:
  bool shouldClose() const override;
  void getWindowSize(int32_t &outWidth, int32_t &outHeight) const override;
  void getFramebufferSize(int32_t &outWidth, int32_t &outHeight) const override;
  void resizeSwapchain(int32_t width, int32_t height) override;
  Format getSwapchainFormat() const override;
  uint32_t getSwapchainImageIndex() const override;
  uint32_t getSwapchainImageCount() const override;
  double getTime() const override;

  Result<BufferHandle, std::string>
  createBuffer(const BufferDesc &desc, std::string_view debugName) override;
  Result<TextureHandle, std::string>
  createTexture(const TextureDesc &desc, std::string_view debugName) override;
  Result<TextureHandle, std::string>
  createFramebufferTexture(const TextureDesc &desc,
                           std::string_view debugName) override;
  Result<TextureHandle, std::string> createDepthBuffer() override;
  Result<ShaderHandle, std::string>
  createShaderModule(const ShaderDesc &desc) override;
  Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &desc,
                       std::string_view debugName) override;
  Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &desc,
                        std::string_view debugName) override;

  void destroyRenderPipeline(RenderPipelineHandle pipeline) override;
  void destroyComputePipeline(ComputePipelineHandle pipeline) override;
  void destroyBuffer(BufferHandle buffer) override;
  void destroyTexture(TextureHandle texture) override;
  void destroyShaderModule(ShaderHandle shader) override;

  bool isValid(BufferHandle h) const override;
  bool isValid(TextureHandle h) const override;
  bool isValid(ShaderHandle h) const override;
  bool isValid(RenderPipelineHandle h) const override;
  bool isValid(ComputePipelineHandle h) const override;
  Format getTextureFormat(TextureHandle h) const override;
  uint32_t getTextureBindlessIndex(TextureHandle h) const override;
  uint32_t getDefaultSamplerBindlessIndex() const override;
  uint32_t getCubemapSamplerBindlessIndex() const override;
  uint64_t getBufferDeviceAddress(BufferHandle h, size_t offset) const override;
  bool resolveGeometry(GeometryAllocationHandle h,
                       GeometryAllocationView &out) const override;

  Result<bool, std::string> beginFrame(uint64_t frameIndex) override;
  Result<bool, std::string> submitFrame(const RenderFrame &frame) override;
  Result<bool, std::string> submitComputeDispatches(
      std::span<const ComputeDispatchItem> dispatches) override;
  Result<GeometryAllocationHandle, std::string>
  allocateGeometry(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
                   std::span<const std::byte> indexBytes, uint32_t indexCount,
                   std::string_view debugName) override;
  void releaseGeometry(GeometryAllocationHandle h) override;
  Result<bool, std::string>
  copyBufferRegions(std::span<const BufferCopyRegion> regions) override;

  Result<bool, std::string>
  updateBuffer(BufferHandle buffer, std::span<const std::byte> data,
               size_t offset) override;
  Result<bool, std::string>
  readBuffer(BufferHandle buffer, size_t offset,
             std::span<std::byte> outBytes) override;
  std::byte *getMappedBufferPtr(BufferHandle buffer) override;
  void flushMappedBuffer(BufferHandle buffer, size_t offset,
                         size_t size) override;
  Result<bool, std::string>
  readTexture(TextureHandle texture, const TextureReadbackRegion &region,
              std::span<std::byte> outBytes) override;

  void waitIdle() override;

  uint32_t swapchainImageCount = 2u;
  uint32_t createdBufferCount = 0u;
  uint32_t createdTextureCount = 0u;
  uint32_t destroyedBufferCount = 0u;
  uint32_t destroyedTextureCount = 0u;
  uint32_t submitCount = 0u;

protected:
  Result<BufferHandle, std::string> createBufferImpl();
  Result<TextureHandle, std::string> createTextureImpl();
  void destroyBufferImpl(BufferHandle buffer);
  void destroyTextureImpl(TextureHandle texture);
  void recordSubmitFrame(const RenderFrame &frame);

private:
  uint32_t nextBufferIndex_ = 1u;
  uint32_t nextTextureIndex_ = 1u;
};

class FakeExecutorGPUDevice final : public FakeGPUDeviceBase {
public:
  Result<BufferHandle, std::string>
  createBuffer(const BufferDesc &desc, std::string_view debugName) override;
  Result<TextureHandle, std::string>
  createTexture(const TextureDesc &desc, std::string_view debugName) override;
  Result<bool, std::string> submitFrame(const RenderFrame &frame) override;

  size_t lastSubmitPassCount = 0u;
  TextureHandle lastColorTexture{};
  TextureHandle lastDepthTexture{};
  BufferHandle lastDependencyBuffer{};
  BufferHandle lastPreDispatchDependencyBuffer{};
  BufferHandle lastDrawVertexBuffer{};
  uint32_t failCreateBufferAtCall = 0u;
  uint32_t failCreateTextureAtCall = 0u;
  bool failSubmitFrame = false;

private:
  uint32_t createBufferCallCount = 0u;
  uint32_t createTextureCallCount = 0u;
};

class FakeRendererGPUDevice final : public FakeGPUDeviceBase {
public:
  Result<bool, std::string> submitFrame(const RenderFrame &frame) override;
  size_t submittedPassCount = 0u;
  std::vector<std::string> submittedPassLabels{};
};

bool hasPassLabel(const FakeRendererGPUDevice &gpu, std::string_view label);

} // namespace nuri::test_support
