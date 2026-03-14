#pragma once

#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::test_support {

bool sameHandle(BufferHandle lhs, BufferHandle rhs);
bool sameHandle(TextureHandle lhs, TextureHandle rhs);
bool sameHandle(RecordingContextHandle lhs, RecordingContextHandle rhs);
bool sameHandle(RecordedCommandBufferHandle lhs,
                RecordedCommandBufferHandle rhs);
bool sameHandle(SubmissionHandle lhs, SubmissionHandle rhs);
bool sameBuffer(BufferHandle lhs, BufferHandle rhs);
bool sameTexture(TextureHandle lhs, TextureHandle rhs);

BufferDesc makeTransientBufferDesc(size_t bytes);
TextureDesc makeTransientTextureDesc(Format format, uint32_t width,
                                     uint32_t height);
RenderPass makeTestPass(std::string_view label,
                        TextureHandle colorTexture = {});

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
  Result<bool, std::string> prepareFrameOutput() override;
  bool supportsParallelGraphicsRecording() const override;
  uint32_t maxParallelGraphicsRecordingContexts() const override;
  Result<RecordingContextHandle, std::string>
  acquireGraphicsRecordingContext(uint32_t workerIndex) override;
  Result<bool, std::string>
  recordGraphicsBarriers(RecordingContextHandle ctx,
                         const GraphicsBarrierRecord *barriers,
                         uint32_t barrierCount) override;
  Result<bool, std::string> recordGraphicsPass(RecordingContextHandle ctx,
                                               const RenderPass &pass) override;
  Result<RecordedCommandBufferHandle, std::string>
  finishGraphicsRecordingContext(RecordingContextHandle ctx) override;
  Result<bool, std::string>
  discardGraphicsRecordingContext(RecordingContextHandle ctx) override;
  Result<bool, std::string> discardRecordedGraphicsCommandBuffer(
      RecordedCommandBufferHandle commandBuffer) override;
  Result<SubmissionHandle, std::string> submitRecordedGraphicsFrame(
      std::span<const RecordedCommandBufferHandle> commandBuffers,
      std::span<const SubmitBatchMeta> batches) override;
  bool isSubmissionComplete(SubmissionHandle handle) const override;
  Result<bool, std::string> submitComputeDispatches(
      std::span<const ComputeDispatchItem> dispatches) override;
  Result<GeometryAllocationHandle, std::string>
  allocateGeometry(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
                   std::span<const std::byte> indexBytes, uint32_t indexCount,
                   std::string_view debugName) override;
  void releaseGeometry(GeometryAllocationHandle h) override;
  Result<bool, std::string>
  copyBufferRegions(std::span<const BufferCopyRegion> regions) override;

  Result<bool, std::string> updateBuffer(BufferHandle buffer,
                                         std::span<const std::byte> data,
                                         size_t offset) override;
  Result<bool, std::string> readBuffer(BufferHandle buffer, size_t offset,
                                       std::span<std::byte> outBytes) override;
  std::byte *getMappedBufferPtr(BufferHandle buffer) override;
  void flushMappedBuffer(BufferHandle buffer, size_t offset,
                         size_t size) override;
  Result<bool, std::string> readTexture(TextureHandle texture,
                                        const TextureReadbackRegion &region,
                                        std::span<std::byte> outBytes) override;

  void waitIdle() override;

  uint32_t swapchainImageCount = 2u;
  uint32_t createdBufferCount = 0u;
  uint32_t createdTextureCount = 0u;
  uint32_t destroyedBufferCount = 0u;
  uint32_t destroyedTextureCount = 0u;
  uint32_t submitCount = 0u;
  uint32_t waitIdleCallCount = 0u;
  uint32_t discardedRecordingContextCount = 0u;
  uint32_t discardedRecordedCommandBufferCount = 0u;
  uint32_t finishedRecordingContextCount = 0u;
  uint32_t acquiredRecordingContextCount = 0u;
  uint32_t maxRecordingContexts = 8u;
  int32_t failAcquireWorkerIndex = -1;
  std::string failRecordPassLabel{};
  uint32_t failFinishAtCall = 0u;
  std::vector<RenderPass> recordedPasses{};
  std::vector<uint32_t> recordedBarrierBatchCounts{};
  std::vector<RecordedCommandBufferHandle> submittedCommandBuffers{};
  std::vector<SubmitBatchMeta> submittedBatches{};
  SubmissionHandle lastSubmittedFrameHandle{};

protected:
  Result<BufferHandle, std::string> createBufferImpl();
  Result<TextureHandle, std::string> createTextureImpl();
  void destroyBufferImpl(BufferHandle buffer);
  void destroyTextureImpl(TextureHandle texture);
  void recordSubmitFrame(
      std::span<const RenderPass> passes,
      std::span<const RecordedCommandBufferHandle> commandBuffers = {},
      std::span<const SubmitBatchMeta> batches = {});

private:
  struct RecordingContextState {
    RecordingContextHandle handle{};
    uint32_t workerIndex = 0u;
    bool finished = false;
    std::vector<RenderPass> passes{};
  };

  struct RecordedCommandBufferState {
    RecordedCommandBufferHandle handle{};
    std::vector<RenderPass> passes{};
  };

  struct SubmissionState {
    SubmissionHandle handle{};
    uint64_t readyFrameIndex = 0u;
  };

  uint32_t nextBufferIndex_ = 1u;
  uint32_t nextTextureIndex_ = 1u;
  uint32_t nextRecordingContextIndex_ = 1u;
  uint32_t nextRecordedCommandBufferIndex_ = 1u;
  uint32_t nextSubmissionIndex_ = 1u;
  uint32_t finishCallCount_ = 0u;
  uint64_t currentFrameIndex_ = 0u;
  std::vector<RecordingContextState> activeRecordingContexts_{};
  std::vector<RecordedCommandBufferState> finishedCommandBuffers_{};
  std::vector<SubmissionState> submissions_{};
  mutable std::mutex recordingStateMutex_{};
};

class FakeExecutorGPUDevice final : public FakeGPUDeviceBase {
public:
  Result<BufferHandle, std::string>
  createBuffer(const BufferDesc &desc, std::string_view debugName) override;
  Result<TextureHandle, std::string>
  createTexture(const TextureDesc &desc, std::string_view debugName) override;
  Result<SubmissionHandle, std::string> submitRecordedGraphicsFrame(
      std::span<const RecordedCommandBufferHandle> commandBuffers,
      std::span<const SubmitBatchMeta> batches) override;

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
  Result<SubmissionHandle, std::string> submitRecordedGraphicsFrame(
      std::span<const RecordedCommandBufferHandle> commandBuffers,
      std::span<const SubmitBatchMeta> batches) override;
  size_t submittedPassCount = 0u;
  std::vector<std::string> submittedPassLabels{};
};

bool hasPassLabel(const FakeRendererGPUDevice &gpu, std::string_view label);

} // namespace nuri::test_support
