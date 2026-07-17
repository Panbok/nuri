#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "nuri/core/result.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"

namespace nuri {

// Backend-private buffer allocation produced off the render thread. Prepared
// buffers do not enter public handle tables until the render thread publishes
// them, so driver allocation cannot stall editor event processing.
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

// Backend-private texture allocation produced off the render thread. The
// object deliberately exposes no native API types; only the originating
// GPUDevice may publish or discard it.
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

struct GPUAdapterInfo {
  std::string name = "unknown";
  uint32_t vendorId = 0u;
  uint32_t deviceId = 0u;
  std::string driverVersion = "unknown";
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
  [[nodiscard]] virtual bool
  supportsSwapchainPresentModeChange() const noexcept {
    return false;
  }
  [[nodiscard]] virtual SwapchainPresentMode
  getSwapchainPresentMode() const noexcept {
    return SwapchainPresentMode::Unknown;
  }
  virtual Result<SwapchainPresentMode, std::string>
  setSwapchainPresentMode(SwapchainPresentMode) {
    return Result<SwapchainPresentMode, std::string>::makeError(
        "Runtime swapchain present-mode changes are not supported");
  }

  // Resource creation
  virtual Result<BufferHandle, std::string>
  createBuffer(const BufferDesc &desc, std::string_view debugName = {}) = 0;
  // Optional two-phase path for streaming and inactive renderer preparation.
  // prepareBuffer may run on a resource worker and must not mutate public Nuri
  // handle tables. Publication is render-thread-only and intentionally small.
  [[nodiscard]] virtual bool
  supportsBackgroundBufferPreparation() const noexcept {
    return false;
  }
  virtual Result<std::unique_ptr<PreparedGpuBuffer>, std::string>
  prepareBuffer(const BufferDesc &, std::string_view = {}) {
    return Result<std::unique_ptr<PreparedGpuBuffer>, std::string>::makeError(
        "background buffer preparation is unsupported");
  }
  virtual Result<BufferHandle, std::string>
  publishPreparedBuffer(std::unique_ptr<PreparedGpuBuffer>) {
    return Result<BufferHandle, std::string>::makeError(
        "prepared buffer publication is unsupported");
  }
  // A batch is indivisible with respect to upload submission. This is used by
  // compound assets whose buffers must share one residency fence.
  [[nodiscard]] virtual bool
  supportsBackgroundBufferBatchPreparation() const noexcept {
    return false;
  }
  virtual Result<std::vector<std::unique_ptr<PreparedGpuBuffer>>, std::string>
  prepareBufferBatch(std::span<const PreparedBufferRequest>) {
    return Result<std::vector<std::unique_ptr<PreparedGpuBuffer>>,
                  std::string>::
        makeError("background buffer batch preparation is unsupported");
  }
  virtual Result<TextureHandle, std::string>
  createTexture(const TextureDesc &desc, std::string_view debugName = {}) = 0;
  // Optional two-phase path for asset streaming. prepareTexture may run on a
  // dedicated resource worker and must not mutate public Nuri handle tables.
  // publishPreparedTexture is render-thread-only and performs the bounded
  // handle/descriptor adoption step.
  [[nodiscard]] virtual bool
  supportsBackgroundTexturePreparation() const noexcept {
    return false;
  }
  virtual Result<std::unique_ptr<PreparedGpuTexture>, std::string>
  prepareTexture(const TextureDesc &, std::string_view = {}) {
    return Result<std::unique_ptr<PreparedGpuTexture>, std::string>::makeError(
        "background texture preparation is unsupported");
  }
  virtual Result<TextureHandle, std::string>
  publishPreparedTexture(std::unique_ptr<PreparedGpuTexture>) {
    return Result<TextureHandle, std::string>::makeError(
        "prepared texture publication is unsupported");
  }
  virtual Result<TextureHandle, std::string>
  createFramebufferTexture(const TextureDesc &desc,
                           std::string_view debugName = {}) = 0;
  virtual Result<SamplerHandle, std::string>
  createSampler(const SamplerDesc &desc, std::string_view debugName = {}) = 0;
  virtual Result<TextureHandle, std::string> createDepthBuffer() = 0;
  virtual Result<ShaderHandle, std::string>
  createShaderModule(const ShaderDesc &desc) = 0;
  virtual Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &desc,
                       std::string_view debugName = {}) = 0;
  virtual Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &desc,
                        std::string_view debugName = {}) = 0;
  virtual Result<MeshletPipelineHandle, std::string>
  createMeshletPipeline(const MeshletPipelineDesc &desc,
                        std::string_view debugName = {}) = 0;

  virtual void destroyRenderPipeline(RenderPipelineHandle pipeline) = 0;
  virtual void destroyComputePipeline(ComputePipelineHandle pipeline) = 0;
  virtual void destroyMeshletPipeline(MeshletPipelineHandle pipeline) = 0;
  virtual void destroyBuffer(BufferHandle buffer) = 0;
  virtual void destroyTexture(TextureHandle texture) = 0;
  virtual void destroySampler(SamplerHandle sampler) = 0;
  virtual void destroyShaderModule(ShaderHandle shader) = 0;

  // Resource queries
  virtual bool isValid(BufferHandle h) const = 0;
  virtual bool isValid(TextureHandle h) const = 0;
  virtual bool isValid(SamplerHandle h) const = 0;
  virtual bool isValid(ShaderHandle h) const = 0;
  virtual bool isValid(RenderPipelineHandle h) const = 0;
  virtual bool isValid(ComputePipelineHandle h) const = 0;
  virtual bool isValid(MeshletPipelineHandle h) const = 0;
  virtual Format getTextureFormat(TextureHandle h) const = 0;
  virtual TextureDimensions getTextureDimensions(TextureHandle h) const = 0;
  virtual TextureCompressionCaps getTextureCompressionCaps() const = 0;
  [[nodiscard]] virtual TextureUploadTelemetry
  getTextureUploadTelemetry() const {
    return {};
  }
  virtual GPUAdapterInfo getAdapterInfo() const { return {}; }
  virtual GpuMultisampleCapabilities getMultisampleCapabilities() const {
    return {};
  }
  virtual ExternalTemporalProviderBackend *externalTemporalProviderBackend() {
    return nullptr;
  }
  virtual ExternalTemporalProviderCapabilities
  getExternalTemporalProviderCapabilities() const {
    return {};
  }
  virtual bool supportsFeature(GPUFeature feature) const {
    (void)feature;
    return false;
  }
  virtual MeshletLimits getMeshletLimits() const { return {}; }
  virtual bool supportsSampledImageLinearFiltering(Format format) const {
    (void)format;
    return false;
  }
  // Descriptor-table index used by shader resource arrays.
  virtual uint32_t getTextureBindlessIndex(TextureHandle h) const = 0;
  virtual uint32_t getSamplerBindlessIndex(SamplerHandle h) const = 0;
  virtual uint8_t getMaxSamplerAnisotropy() const = 0;
  virtual uint32_t
  getLinearRepeatSamplerBindlessIndex(bool useMipmaps,
                                      uint8_t maxAnisotropy = 1u) const = 0;
  // Bindless index for the default general-purpose sampler.
  virtual uint32_t getDefaultSamplerBindlessIndex() const = 0;
  // Bindless index for cubemap/IBL sampling (clamp-to-edge).
  virtual uint32_t getCubemapSamplerBindlessIndex() const = 0;
  // GPU virtual address used by shaders.
  virtual uint64_t getBufferDeviceAddress(BufferHandle h,
                                          size_t offset = 0) const = 0;
  virtual bool resolveGeometry(GeometryAllocationHandle h,
                               GeometryAllocationView &out) const = 0;
  virtual uint64_t geometryMutationVersion() const { return 0; }
  // Returns timing data after the GPU/command queue has captured and finalized
  // the submitted ranges, making the report safe to read. A report may become
  // available one or more frames after submission, depending on queue progress.
  virtual GpuTimingReport getLatestCompletedGpuTimingReport() const {
    return {};
  }
  // Drains completed per-submission GPU timing reports into `outReports` in
  // completion order and removes returned reports from the device queue.
  virtual size_t
  drainCompletedGpuTimingReports(std::span<GpuTimingReport> outReports) {
    (void)outReports;
    return 0u;
  }
  virtual uint64_t droppedGpuTimingReportCount() const { return 0u; }

  // Rendering
  virtual bool supportsParallelGraphicsRecording() const { return false; }
  virtual uint32_t maxParallelGraphicsRecordingContexts() const { return 1u; }
  virtual Result<bool, std::string> beginFrame(uint64_t frameIndex) = 0;
  virtual Result<bool, std::string> prepareFrameOutput() {
    return Result<bool, std::string>::makeResult(true);
  }
  virtual Result<RecordingContextHandle, std::string>
  acquireGraphicsRecordingContext(uint32_t workerIndex) = 0;
  virtual Result<bool, std::string>
  recordGraphicsBarriers(RecordingContextHandle ctx,
                         std::span<const GraphicsBarrierRecord> barriers) = 0;
  virtual Result<bool, std::string>
  recordGraphicsPass(RecordingContextHandle ctx, const RenderPass &pass) = 0;
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
  finishGraphicsRecordingContext(RecordingContextHandle ctx) = 0;
  virtual Result<bool, std::string>
  discardGraphicsRecordingContext(RecordingContextHandle ctx) = 0;
  virtual Result<bool, std::string> discardRecordedGraphicsCommandBuffer(
      RecordedCommandBufferHandle commandBuffer) = 0;
  virtual Result<SubmissionHandle, std::string> submitRecordedGraphicsFrame(
      std::span<const RecordedCommandBufferHandle> commandBuffers,
      std::span<const SubmitBatchMeta> batches) = 0;
  virtual bool isSubmissionComplete(SubmissionHandle handle) const = 0;
  // Establishes the queue dependency required before resources written by a
  // completed upload submission may be consumed by graphics. Graphics-queue
  // submissions are already ordered and return immediately; a dedicated copy
  // queue backend queues a non-blocking timeline wait for the next graphics
  // submission.
  virtual Result<bool, std::string>
  makeSubmissionVisibleToGraphics(SubmissionHandle handle) {
    (void)handle;
    return Result<bool, std::string>::makeResult(true);
  }
  virtual Result<bool, std::string>
  submitComputeDispatches(std::span<const ComputeDispatchItem> dispatches) = 0;
  virtual Result<GeometryAllocationHandle, std::string>
  allocateGeometry(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
                   std::span<const std::byte> indexBytes, uint32_t indexCount,
                   std::string_view debugName = {}) = 0;
  // Optional publication seam for geometry whose immutable vertex/index
  // buffers were prepared on a resource worker. Ownership of both buffers is
  // transferred to the geometry pool only when publication succeeds.
  [[nodiscard]] virtual bool
  supportsBackgroundGeometryPreparation() const noexcept {
    return false;
  }
  virtual Result<GeometryAllocationHandle, std::string>
  adoptPreparedGeometry(BufferHandle, size_t, uint32_t, BufferHandle, size_t,
                        uint32_t, std::string_view = {}) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        "prepared geometry publication is unsupported");
  }
  virtual void releaseGeometry(GeometryAllocationHandle h) = 0;
  virtual Result<SubmissionHandle, std::string>
  submitBackgroundBufferCopies(std::span<const BufferCopyRegion> regions,
                               std::string_view debugName = {}) = 0;
  // Submits all resource-initialization and update commands recorded by this
  // device since the previous upload submission. The returned handle becomes
  // complete when those uploads are resident. An invalid handle means no
  // commands were pending.
  virtual Result<SubmissionHandle, std::string> submitPendingUploads() {
    return Result<SubmissionHandle, std::string>::makeResult(
        SubmissionHandle{});
  }
  [[nodiscard]] virtual bool
  assetUploadsUseDedicatedCopyQueue() const noexcept {
    return false;
  }
  // Captures completion of every graphics recording or upload that already
  // exists at this call. Work recorded later is not included. This is used to
  // defer reuse of suballocated memory without stalling the CPU.
  virtual Result<SubmissionHandle, std::string> captureWorkCompletion() {
    return Result<SubmissionHandle, std::string>::makeResult(
        SubmissionHandle{});
  }

  // Data updates
  virtual Result<bool, std::string>
  updateBuffer(BufferHandle buffer, std::span<const std::byte> data,
               size_t offset = 0) = 0;
  // Validates and consumes an ordered batch of source spans before returning.
  // Backends may submit the writes asynchronously on the graphics queue; later
  // graphics submissions observe them through queue FIFO ordering.
  virtual Result<bool, std::string>
  updateBuffers(std::span<const BufferUpdate> updates) {
    for (const BufferUpdate &update : updates) {
      auto result = updateBuffer(update.buffer, update.data, update.offset);
      if (result.hasError()) {
        return result;
      }
    }
    return Result<bool, std::string>::makeResult(true);
  }
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
  // Reads the requested texture region into tightly packed, row-major rows.
  // The first output row is the texture's top row at (region.x, region.y), and
  // each following row starts immediately after width * bytesPerPixel bytes.
  // Callers must synchronize GPU work before reading textures written by
  // previous submissions.
  virtual Result<bool, std::string>
  readTexture(TextureHandle texture, const TextureReadbackRegion &region,
              std::span<std::byte> outBytes) = 0;

  // Shutdown
  virtual void waitIdle() = 0;

protected:
  GPUDevice() = default;
};

} // namespace nuri
