#pragma once
#include "nuri/core/result.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

class NURI_API PreparedGpuBuffer {
public:
  virtual ~PreparedGpuBuffer() = default;
  PreparedGpuBuffer(const PreparedGpuBuffer &) = delete;
  PreparedGpuBuffer &operator=(const PreparedGpuBuffer &) = delete;
  PreparedGpuBuffer(PreparedGpuBuffer &&) = delete;
  PreparedGpuBuffer &operator=(PreparedGpuBuffer &&) = delete;

protected:
  PreparedGpuBuffer() = default;
};

class NURI_API PreparedGpuTexture {
public:
  virtual ~PreparedGpuTexture() = default;
  PreparedGpuTexture(const PreparedGpuTexture &) = delete;
  PreparedGpuTexture &operator=(const PreparedGpuTexture &) = delete;
  PreparedGpuTexture(PreparedGpuTexture &&) = delete;
  PreparedGpuTexture &operator=(PreparedGpuTexture &&) = delete;

protected:
  PreparedGpuTexture() = default;
};

struct PreparedBufferRequest {
  BufferDesc desc{};
  std::string_view debugName{};
};

struct TextureReadbackRegion {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t mipLevel = 0;
  uint32_t layer = 0;
};

struct TextureCompressionCaps {
  bool bc7 = false;
  bool etc2 = false;
  bool astc = false;
};

struct TextureUploadTelemetry {
  uint64_t texturesRecorded = 0u;
  uint64_t bytesRecorded = 0u;
  uint64_t batchesSubmitted = 0u;
  uint64_t boundedBatchFlushes = 0u;
  uint64_t completionWaits = 0u;
  uint64_t pendingBytes = 0u;
  uint32_t pendingTextures = 0u;
};

struct BackendCreationTelemetry {
  uint64_t renderPipelines = 0u;
  uint64_t computePipelines = 0u;
  uint64_t meshletPipelines = 0u;
  uint64_t framebuffers = 0u;
};

struct GPUAdapterInfo {
  std::string name = "unknown";
  uint32_t vendorId = 0u;
  uint32_t deviceId = 0u;
  std::string driverVersion = "unknown";
};

struct GPUDeviceImpl;

class NURI_API GPUDevice {
public:
  static std::unique_ptr<GPUDevice>
  create(Window &window, const GPUDeviceCreateDesc &desc = {});
  virtual ~GPUDevice();
  GPUDevice(const GPUDevice &) = delete;
  GPUDevice &operator=(const GPUDevice &) = delete;
  GPUDevice(GPUDevice &&) = delete;
  GPUDevice &operator=(GPUDevice &&) = delete;
  virtual bool shouldClose() const;
  virtual void getWindowSize(int32_t &outWidth, int32_t &outHeight) const;
  virtual void getFramebufferSize(int32_t &outWidth, int32_t &outHeight) const;
  virtual void resizeSwapchain(int32_t width, int32_t height);
  virtual Format getSwapchainFormat() const;
  virtual uint32_t getSwapchainImageIndex() const;
  virtual uint32_t getSwapchainImageCount() const;
  virtual double getTime() const;
  [[nodiscard]] virtual bool
  supportsSwapchainPresentModeChange() const noexcept;
  [[nodiscard]] virtual SwapchainPresentMode
  getSwapchainPresentMode() const noexcept;
  virtual Result<SwapchainPresentMode, std::string>
  setSwapchainPresentMode(SwapchainPresentMode mode);
  virtual Result<BufferHandle, std::string>
  createBuffer(const BufferDesc &desc, std::string_view debugName = {});
  [[nodiscard]] virtual bool
  supportsBackgroundBufferPreparation() const noexcept {
    return impl_ != nullptr;
  }
  virtual Result<std::unique_ptr<PreparedGpuBuffer>, std::string>
  prepareBuffer(const BufferDesc &desc, std::string_view debugName = {});
  virtual Result<BufferHandle, std::string>
  publishPreparedBuffer(std::unique_ptr<PreparedGpuBuffer> prepared);
  [[nodiscard]] virtual bool
  supportsBackgroundBufferBatchPreparation() const noexcept {
    return impl_ != nullptr;
  }
  virtual Result<std::vector<std::unique_ptr<PreparedGpuBuffer>>, std::string>
  prepareBufferBatch(std::span<const PreparedBufferRequest> requests);
  virtual Result<TextureHandle, std::string>
  createTexture(const TextureDesc &desc, std::string_view debugName = {});
  [[nodiscard]] virtual bool
  supportsBackgroundTexturePreparation() const noexcept {
    return impl_ != nullptr;
  }
  virtual Result<std::unique_ptr<PreparedGpuTexture>, std::string>
  prepareTexture(const TextureDesc &desc, std::string_view debugName = {});
  virtual Result<TextureHandle, std::string>
  publishPreparedTexture(std::unique_ptr<PreparedGpuTexture> prepared);
  virtual Result<TextureHandle, std::string>
  createFramebufferTexture(const TextureDesc &desc,
                           std::string_view debugName = {});
  virtual Result<SamplerHandle, std::string>
  createSampler(const SamplerDesc &desc, std::string_view debugName = {});
  virtual Result<TextureHandle, std::string> createDepthBuffer();
  virtual Result<ShaderHandle, std::string>
  createShaderModule(const ShaderDesc &desc);
  virtual Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &desc,
                       std::string_view debugName = {});
  virtual Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &desc,
                        std::string_view debugName = {});
  virtual Result<RayQueryBindingHandle, std::string> createRayQueryBinding(
      ComputePipelineHandle pipeline,
      AccelerationStructureHandle topLevelAccelerationStructure,
      std::string_view debugName = {});
  virtual Result<MeshletPipelineHandle, std::string>
  createMeshletPipeline(const MeshletPipelineDesc &desc,
                        std::string_view debugName = {});
  virtual void destroyRenderPipeline(RenderPipelineHandle pipeline);
  virtual void destroyComputePipeline(ComputePipelineHandle pipeline);
  virtual void destroyRayQueryBinding(RayQueryBindingHandle binding);
  virtual void destroyMeshletPipeline(MeshletPipelineHandle pipeline);
  virtual void destroyBuffer(BufferHandle buffer);
  virtual void destroyTexture(TextureHandle texture);
  virtual void destroySampler(SamplerHandle sampler);
  virtual void destroyShaderModule(ShaderHandle shader);
  virtual bool isValid(BufferHandle h) const;
  virtual bool isValid(TextureHandle h) const;
  virtual bool isValid(SamplerHandle h) const;
  virtual bool isValid(ShaderHandle h) const;
  virtual bool isValid(RenderPipelineHandle h) const;
  virtual bool isValid(ComputePipelineHandle h) const;
  virtual bool isValid(RayQueryBindingHandle h) const;
  virtual bool isValid(MeshletPipelineHandle h) const;
  virtual Format getTextureFormat(TextureHandle h) const;
  virtual TextureDimensions getTextureDimensions(TextureHandle h) const;
  virtual TextureCompressionCaps getTextureCompressionCaps() const;
  [[nodiscard]] virtual TextureUploadTelemetry
  getTextureUploadTelemetry() const;
  [[nodiscard]] virtual BackendCreationTelemetry
  getBackendCreationTelemetry() const;
  virtual GPUAdapterInfo getAdapterInfo() const;
  virtual GpuMultisampleCapabilities getMultisampleCapabilities() const;
  [[nodiscard]] virtual DeviceCaps getDeviceCaps() const;
  virtual ExternalTemporalProviderBackend *externalTemporalProviderBackend() {
    return nullptr;
  }
  virtual ExternalTemporalProviderCapabilities
  getExternalTemporalProviderCapabilities() const {
    return {};
  }
  virtual bool supportsFeature(GPUFeature feature) const;
  virtual MeshletLimits getMeshletLimits() const;
  virtual bool supportsSampledImageLinearFiltering(Format format) const;
  virtual uint32_t getTextureBindlessIndex(TextureHandle h) const;
  virtual uint32_t getSamplerBindlessIndex(SamplerHandle h) const;
  virtual uint8_t getMaxSamplerAnisotropy() const;
  virtual uint32_t
  getLinearRepeatSamplerBindlessIndex(bool useMipmaps,
                                      uint8_t maxAnisotropy = 1u) const;
  virtual uint32_t getDefaultSamplerBindlessIndex() const;
  virtual uint32_t getCubemapSamplerBindlessIndex() const;
  virtual uint64_t getBufferDeviceAddress(BufferHandle h,
                                          size_t offset = 0) const;
  virtual Result<AccelerationStructureHandle, std::string>
  createBottomLevelAccelerationStructure(const BlasCreateDesc &desc,
                                         std::string_view debugName = {});
  virtual Result<AccelerationStructureHandle, std::string>
  createTopLevelAccelerationStructure(const TlasCreateDesc &desc,
                                      std::string_view debugName = {});
  virtual void destroyAccelerationStructure(
      AccelerationStructureHandle accelerationStructure);
  virtual bool isValid(AccelerationStructureHandle h) const;
  virtual Result<AccelerationStructureFacts, std::string>
  getAccelerationStructureFacts(AccelerationStructureHandle h) const;
  [[nodiscard]] virtual RayTracingBackendTelemetry
  getRayTracingBackendTelemetry() const;
  virtual bool resolveGeometry(GeometryAllocationHandle h,
                               GeometryAllocationView &out) const;
  virtual uint64_t geometryMutationVersion() const;
  virtual GpuTimingReport getLatestCompletedGpuTimingReport() const;
  virtual size_t
  drainCompletedGpuTimingReports(std::span<GpuTimingReport> outReports);
  virtual uint64_t droppedGpuTimingReportCount() const;
  virtual bool supportsParallelGraphicsRecording() const { return false; }
  virtual uint32_t maxParallelGraphicsRecordingContexts() const { return 1u; }
  virtual Result<bool, std::string> beginFrame(uint64_t frameIndex);
  virtual Result<bool, std::string> prepareFrameOutput();
  virtual Result<RecordingContextHandle, std::string>
  acquireGraphicsRecordingContext(uint32_t workerIndex);
  virtual Result<bool, std::string>
  recordGraphicsBarriers(RecordingContextHandle ctx,
                         std::span<const GraphicsBarrierRecord> barriers);
  virtual Result<bool, std::string> retainGraphicsRecordingReferences(
      RecordingContextHandle ctx,
      const GraphicsRecordingReferences &references);
  virtual Result<bool, std::string>
  recordGraphicsPass(RecordingContextHandle ctx, const RenderPass &pass);
  virtual Result<bool, std::string>
  recordGraphicsRange(RecordingContextHandle ctx,
                      std::span<const GraphicsRecordingStep> steps);
  virtual Result<bool, std::string> recordExternalTemporalDispatch(
      RecordingContextHandle ctx, const ExternalTemporalDispatchItem &dispatch,
      GpuTimingScope timingScope, std::string_view debugLabel) {
    (void)ctx;
    (void)dispatch;
    (void)timingScope;
    (void)debugLabel;
    return Result<bool, std::string>::makeError(
        "external temporal dispatch is unsupported by this GPU backend");
  }
  virtual Result<RecordedCommandBufferHandle, std::string>
  finishGraphicsRecordingContext(RecordingContextHandle ctx);
  virtual Result<bool, std::string>
  discardGraphicsRecordingContext(RecordingContextHandle ctx);
  virtual Result<bool, std::string> discardRecordedGraphicsCommandBuffer(
      RecordedCommandBufferHandle commandBuffer);
  virtual Result<SubmittedGraphicsFrame, std::string>
  submitRecordedGraphicsFrame(
      std::span<const RecordedCommandBufferHandle> commandBuffers,
      std::span<const SubmitBatchMeta> batches);
  virtual bool isSubmissionComplete(SubmissionHandle handle) const;
  virtual Result<bool, std::string>
  makeSubmissionVisibleToGraphics(SubmissionHandle handle);
  virtual Result<bool, std::string>
  submitComputeDispatches(std::span<const ComputeDispatchItem> dispatches);
  virtual Result<GeometryAllocationHandle, std::string>
  allocateGeometry(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
                   std::span<const std::byte> indexBytes, uint32_t indexCount,
                   std::string_view debugName = {});
  [[nodiscard]] virtual bool
  supportsBackgroundGeometryPreparation() const noexcept {
    return impl_ != nullptr;
  }
  virtual Result<GeometryAllocationHandle, std::string>
  adoptPreparedGeometry(BufferHandle vertexBuffer, size_t vertexBytes,
                        uint32_t vertexCount, BufferHandle indexBuffer,
                        size_t indexBytes, uint32_t indexCount,
                        std::string_view debugName = {});
  virtual void releaseGeometry(GeometryAllocationHandle h);
  virtual Result<SubmissionHandle, std::string>
  submitBackgroundBufferCopies(std::span<const BufferCopyRegion> regions,
                               std::string_view debugName = {});
  virtual Result<SubmissionHandle, std::string> submitPendingUploads();
  [[nodiscard]] virtual bool assetUploadsUseDedicatedCopyQueue() const noexcept;
  virtual Result<SubmissionHandle, std::string> captureWorkCompletion();
  virtual Result<bool, std::string>
  updateBuffer(BufferHandle buffer, std::span<const std::byte> data,
               size_t offset = 0);
  virtual Result<bool, std::string>
  updateBuffers(std::span<const BufferUpdate> updates);
  virtual Result<bool, std::string>
  readBuffer(BufferHandle buffer, size_t offset, std::span<std::byte> outBytes);
  virtual std::byte *getMappedBufferPtr(BufferHandle buffer);
  virtual void flushMappedBuffer(BufferHandle buffer, size_t offset,
                                 size_t size);
  virtual Result<bool, std::string>
  readTexture(TextureHandle texture, const TextureReadbackRegion &region,
              std::span<std::byte> outBytes);
  virtual void waitIdle();

protected:
  GPUDevice();

private:
  std::unique_ptr<GPUDeviceImpl> impl_;
};

} // namespace nuri
