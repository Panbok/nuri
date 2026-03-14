#include "render_graph_test_support.h"

#include <cstdlib>

namespace nuri::test_support {

bool sameHandle(BufferHandle lhs, BufferHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

bool sameHandle(TextureHandle lhs, TextureHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

bool sameHandle(RecordingContextHandle lhs, RecordingContextHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

bool sameHandle(RecordedCommandBufferHandle lhs,
                RecordedCommandBufferHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

bool sameHandle(SubmissionHandle lhs, SubmissionHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

bool sameBuffer(BufferHandle lhs, BufferHandle rhs) {
  return sameHandle(lhs, rhs);
}

bool sameTexture(TextureHandle lhs, TextureHandle rhs) {
  return sameHandle(lhs, rhs);
}

BufferDesc makeTransientBufferDesc(size_t bytes) {
  BufferDesc desc{};
  desc.usage = BufferUsage::Storage;
  desc.storage = Storage::Device;
  desc.size = bytes;
  desc.data = {};
  return desc;
}

TextureDesc makeTransientTextureDesc(Format format, uint32_t width,
                                     uint32_t height) {
  TextureDesc desc{};
  desc.type = TextureType::Texture2D;
  desc.format = format;
  desc.dimensions = TextureDimensions{
      .width = width,
      .height = height,
      .depth = 1u,
  };
  desc.usage = TextureUsage::Attachment;
  desc.storage = Storage::Device;
  desc.numLayers = 1u;
  desc.numSamples = 1u;
  desc.numMipLevels = 1u;
  desc.data = {};
  desc.dataNumMipLevels = 1u;
  desc.generateMipmaps = false;
  return desc;
}

RenderPass makeTestPass(std::string_view label, TextureHandle colorTexture) {
  RenderPass pass{};
  pass.debugLabel = label;
  pass.colorTexture = colorTexture;
  return pass;
}

Result<RenderGraphPassId, std::string>
addTestGraphicsPass(RenderGraphBuilder &builder, const RenderPass &pass,
                    std::string_view debugName, bool autoBindPassResources) {
  RenderGraphGraphicsPassDesc desc{};
  desc.color = pass.color;
  desc.depth = pass.depth;
  desc.useViewport = pass.useViewport;
  desc.viewport = pass.viewport;
  desc.preDispatches = pass.preDispatches;
  desc.dependencyBuffers = pass.dependencyBuffers;
  desc.draws = pass.draws;
  desc.debugLabel = !debugName.empty() ? debugName : pass.debugLabel;
  desc.debugColor = pass.debugColor;
  desc.markColorAsFrameOutput = false;
  desc.markImplicitOutputSideEffect = false;

  if (autoBindPassResources) {
    if (nuri::isValid(pass.colorTexture)) {
      auto colorImportResult =
          builder.importTexture(pass.colorTexture, "test_pass_color");
      if (colorImportResult.hasError()) {
        return Result<RenderGraphPassId, std::string>::makeError(
            colorImportResult.error());
      }
      desc.colorTexture = colorImportResult.value();
    }
    if (nuri::isValid(pass.depthTexture)) {
      auto depthImportResult =
          builder.importTexture(pass.depthTexture, "test_pass_depth");
      if (depthImportResult.hasError()) {
        return Result<RenderGraphPassId, std::string>::makeError(
            depthImportResult.error());
      }
      desc.depthTexture = depthImportResult.value();
    }
  }

  return builder.addGraphicsPass(desc);
}

EnvVarGuard::EnvVarGuard(std::string_view name, std::string_view value)
    : name_(name) {
#if defined(_WIN32)
  char *raw = nullptr;
  size_t len = 0u;
  if (_dupenv_s(&raw, &len, name_.c_str()) == 0 && raw != nullptr) {
    hadOldValue_ = true;
    oldValue_ = raw;
    std::free(raw);
  }
  _putenv_s(name_.c_str(), std::string(value).c_str());
#else
  const char *old = std::getenv(name_.c_str());
  if (old != nullptr) {
    hadOldValue_ = true;
    oldValue_ = old;
  }
  setenv(name_.c_str(), std::string(value).c_str(), 1);
#endif
}

EnvVarGuard::~EnvVarGuard() {
#if defined(_WIN32)
  if (hadOldValue_) {
    _putenv_s(name_.c_str(), oldValue_.c_str());
  } else {
    std::string env = name_ + "=";
    _putenv(env.c_str());
  }
#else
  if (hadOldValue_) {
    setenv(name_.c_str(), oldValue_.c_str(), 1);
  } else {
    unsetenv(name_.c_str());
  }
#endif
}

bool FakeGPUDeviceBase::shouldClose() const { return false; }

void FakeGPUDeviceBase::getWindowSize(int32_t &outWidth,
                                      int32_t &outHeight) const {
  outWidth = 1280;
  outHeight = 720;
}

void FakeGPUDeviceBase::getFramebufferSize(int32_t &outWidth,
                                           int32_t &outHeight) const {
  outWidth = 1280;
  outHeight = 720;
}

void FakeGPUDeviceBase::resizeSwapchain(int32_t, int32_t) {}

Format FakeGPUDeviceBase::getSwapchainFormat() const {
  return Format::RGBA8_UNORM;
}

uint32_t FakeGPUDeviceBase::getSwapchainImageIndex() const { return 0u; }

uint32_t FakeGPUDeviceBase::getSwapchainImageCount() const {
  return swapchainImageCount;
}

double FakeGPUDeviceBase::getTime() const { return 0.0; }

Result<BufferHandle, std::string>
FakeGPUDeviceBase::createBuffer(const BufferDesc &, std::string_view) {
  return createBufferImpl();
}

Result<TextureHandle, std::string>
FakeGPUDeviceBase::createTexture(const TextureDesc &, std::string_view) {
  return createTextureImpl();
}

Result<BufferHandle, std::string> FakeGPUDeviceBase::createBufferImpl() {
  BufferHandle handle{.index = nextBufferIndex_++, .generation = 1u};
  ++createdBufferCount;
  return Result<BufferHandle, std::string>::makeResult(handle);
}

Result<TextureHandle, std::string> FakeGPUDeviceBase::createTextureImpl() {
  TextureHandle handle{.index = nextTextureIndex_++, .generation = 1u};
  ++createdTextureCount;
  return Result<TextureHandle, std::string>::makeResult(handle);
}

Result<BufferHandle, std::string>
FakeExecutorGPUDevice::createBuffer(const BufferDesc &, std::string_view) {
  ++createBufferCallCount;
  if (failCreateBufferAtCall != 0u &&
      createBufferCallCount == failCreateBufferAtCall) {
    return Result<BufferHandle, std::string>::makeError(
        "fake createBuffer failure");
  }
  return createBufferImpl();
}

Result<TextureHandle, std::string>
FakeExecutorGPUDevice::createTexture(const TextureDesc &, std::string_view) {
  ++createTextureCallCount;
  if (failCreateTextureAtCall != 0u &&
      createTextureCallCount == failCreateTextureAtCall) {
    return Result<TextureHandle, std::string>::makeError(
        "fake createTexture failure");
  }
  return createTextureImpl();
}

Result<TextureHandle, std::string>
FakeGPUDeviceBase::createFramebufferTexture(const TextureDesc &,
                                            std::string_view) {
  return Result<TextureHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<TextureHandle, std::string> FakeGPUDeviceBase::createDepthBuffer() {
  return Result<TextureHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<ShaderHandle, std::string>
FakeGPUDeviceBase::createShaderModule(const ShaderDesc &) {
  return Result<ShaderHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<RenderPipelineHandle, std::string>
FakeGPUDeviceBase::createRenderPipeline(const RenderPipelineDesc &,
                                        std::string_view) {
  return Result<RenderPipelineHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<ComputePipelineHandle, std::string>
FakeGPUDeviceBase::createComputePipeline(const ComputePipelineDesc &,
                                         std::string_view) {
  return Result<ComputePipelineHandle, std::string>::makeError(
      "not implemented in fake device");
}

void FakeGPUDeviceBase::destroyRenderPipeline(RenderPipelineHandle) {}

void FakeGPUDeviceBase::destroyComputePipeline(ComputePipelineHandle) {}

void FakeGPUDeviceBase::destroyBuffer(BufferHandle buffer) {
  destroyBufferImpl(buffer);
}

void FakeGPUDeviceBase::destroyTexture(TextureHandle texture) {
  destroyTextureImpl(texture);
}

void FakeGPUDeviceBase::destroyBufferImpl(BufferHandle buffer) {
  if (nuri::isValid(buffer)) {
    ++destroyedBufferCount;
  }
}

void FakeGPUDeviceBase::destroyTextureImpl(TextureHandle texture) {
  if (nuri::isValid(texture)) {
    ++destroyedTextureCount;
  }
}

void FakeGPUDeviceBase::destroyShaderModule(ShaderHandle) {}

bool FakeGPUDeviceBase::isValid(BufferHandle h) const {
  return nuri::isValid(h);
}

bool FakeGPUDeviceBase::isValid(TextureHandle h) const {
  return nuri::isValid(h);
}

bool FakeGPUDeviceBase::isValid(ShaderHandle h) const {
  return nuri::isValid(h);
}

bool FakeGPUDeviceBase::isValid(RenderPipelineHandle h) const {
  return nuri::isValid(h);
}

bool FakeGPUDeviceBase::isValid(ComputePipelineHandle h) const {
  return nuri::isValid(h);
}

Format FakeGPUDeviceBase::getTextureFormat(TextureHandle) const {
  return Format::RGBA8_UNORM;
}

uint32_t FakeGPUDeviceBase::getTextureBindlessIndex(TextureHandle) const {
  return 0u;
}

uint32_t FakeGPUDeviceBase::getDefaultSamplerBindlessIndex() const {
  return 0u;
}

uint32_t FakeGPUDeviceBase::getCubemapSamplerBindlessIndex() const {
  return 0u;
}

uint64_t FakeGPUDeviceBase::getBufferDeviceAddress(BufferHandle, size_t) const {
  return 0ull;
}

bool FakeGPUDeviceBase::resolveGeometry(GeometryAllocationHandle,
                                        GeometryAllocationView &) const {
  return false;
}

Result<bool, std::string> FakeGPUDeviceBase::beginFrame(uint64_t frameIndex) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  currentFrameIndex_ = frameIndex;
  activeRecordingContexts_.clear();
  finishedCommandBuffers_.clear();
  finishCallCount_ = 0u;
  recordedPasses.clear();
  submittedCommandBuffers.clear();
  submittedBatches.clear();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> FakeGPUDeviceBase::prepareFrameOutput() {
  return Result<bool, std::string>::makeResult(true);
}

bool FakeGPUDeviceBase::supportsParallelGraphicsRecording() const {
  return maxRecordingContexts > 1u;
}

uint32_t FakeGPUDeviceBase::maxParallelGraphicsRecordingContexts() const {
  return maxRecordingContexts;
}

Result<RecordingContextHandle, std::string>
FakeGPUDeviceBase::acquireGraphicsRecordingContext(uint32_t workerIndex) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  if (failAcquireWorkerIndex >= 0 &&
      workerIndex == static_cast<uint32_t>(failAcquireWorkerIndex)) {
    return Result<RecordingContextHandle, std::string>::makeError(
        "fake acquireGraphicsRecordingContext injected failure");
  }
  if (activeRecordingContexts_.size() >= maxRecordingContexts) {
    return Result<RecordingContextHandle, std::string>::makeError(
        "fake acquireGraphicsRecordingContext limit exceeded");
  }

  RecordingContextState state{};
  state.handle = RecordingContextHandle{.index = nextRecordingContextIndex_++,
                                        .generation = 1u};
  state.workerIndex = workerIndex;
  activeRecordingContexts_.push_back(state);
  ++acquiredRecordingContextCount;
  return Result<RecordingContextHandle, std::string>::makeResult(
      activeRecordingContexts_.back().handle);
}

Result<bool, std::string>
FakeGPUDeviceBase::recordGraphicsBarriers(RecordingContextHandle ctx,
                                          const GraphicsBarrierRecord *,
                                          uint32_t barrierCount) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  for (auto &state : activeRecordingContexts_) {
    if (sameHandle(state.handle, ctx)) {
      recordedBarrierBatchCounts.push_back(barrierCount);
      return Result<bool, std::string>::makeResult(true);
    }
  }
  return Result<bool, std::string>::makeError(
      "fake recordGraphicsBarriers: unknown recording context");
}

Result<bool, std::string>
FakeGPUDeviceBase::recordGraphicsPass(RecordingContextHandle ctx,
                                      const RenderPass &pass) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  if (!failRecordPassLabel.empty() && pass.debugLabel == failRecordPassLabel) {
    return Result<bool, std::string>::makeError(
        "fake recordGraphicsPass injected failure");
  }
  for (auto &state : activeRecordingContexts_) {
    if (sameHandle(state.handle, ctx)) {
      state.passes.push_back(pass);
      return Result<bool, std::string>::makeResult(true);
    }
  }
  return Result<bool, std::string>::makeError(
      "fake recordGraphicsPass: unknown recording context");
}

Result<RecordedCommandBufferHandle, std::string>
FakeGPUDeviceBase::finishGraphicsRecordingContext(RecordingContextHandle ctx) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  ++finishCallCount_;
  if (failFinishAtCall != 0u && finishCallCount_ == failFinishAtCall) {
    return Result<RecordedCommandBufferHandle, std::string>::makeError(
        "fake finishGraphicsRecordingContext injected failure");
  }
  for (size_t i = 0u; i < activeRecordingContexts_.size(); ++i) {
    if (!sameHandle(activeRecordingContexts_[i].handle, ctx)) {
      continue;
    }

    RecordedCommandBufferState finished{};
    finished.handle = RecordedCommandBufferHandle{
        .index = nextRecordedCommandBufferIndex_++, .generation = 1u};
    finished.passes = activeRecordingContexts_[i].passes;
    finishedCommandBuffers_.push_back(std::move(finished));
    activeRecordingContexts_.erase(activeRecordingContexts_.begin() + i);
    ++finishedRecordingContextCount;
    return Result<RecordedCommandBufferHandle, std::string>::makeResult(
        finishedCommandBuffers_.back().handle);
  }

  return Result<RecordedCommandBufferHandle, std::string>::makeError(
      "fake finishGraphicsRecordingContext: unknown recording context");
}

Result<bool, std::string>
FakeGPUDeviceBase::discardGraphicsRecordingContext(RecordingContextHandle ctx) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  for (size_t i = 0u; i < activeRecordingContexts_.size(); ++i) {
    if (sameHandle(activeRecordingContexts_[i].handle, ctx)) {
      activeRecordingContexts_.erase(activeRecordingContexts_.begin() + i);
      ++discardedRecordingContextCount;
      return Result<bool, std::string>::makeResult(true);
    }
  }
  return Result<bool, std::string>::makeError(
      "fake discardGraphicsRecordingContext: unknown recording context");
}

Result<bool, std::string>
FakeGPUDeviceBase::discardRecordedGraphicsCommandBuffer(
    RecordedCommandBufferHandle commandBuffer) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  for (size_t i = 0u; i < finishedCommandBuffers_.size(); ++i) {
    if (sameHandle(finishedCommandBuffers_[i].handle, commandBuffer)) {
      finishedCommandBuffers_.erase(finishedCommandBuffers_.begin() + i);
      ++discardedRecordedCommandBufferCount;
      return Result<bool, std::string>::makeResult(true);
    }
  }
  return Result<bool, std::string>::makeError(
      "fake discardRecordedGraphicsCommandBuffer: unknown command buffer");
}

void FakeGPUDeviceBase::recordSubmitFrame(
    std::span<const RenderPass> passes,
    std::span<const RecordedCommandBufferHandle> commandBuffers,
    std::span<const SubmitBatchMeta> batches) {
  ++submitCount;
  recordedPasses.assign(passes.begin(), passes.end());
  submittedCommandBuffers.assign(commandBuffers.begin(), commandBuffers.end());
  submittedBatches.assign(batches.begin(), batches.end());
}

Result<SubmissionHandle, std::string>
FakeGPUDeviceBase::submitRecordedGraphicsFrame(
    std::span<const RecordedCommandBufferHandle> commandBuffers,
    std::span<const SubmitBatchMeta> batches) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  std::vector<RenderPass> submittedPasses{};
  for (const RecordedCommandBufferHandle handle : commandBuffers) {
    bool found = false;
    for (const auto &finished : finishedCommandBuffers_) {
      if (!sameHandle(finished.handle, handle)) {
        continue;
      }
      submittedPasses.insert(submittedPasses.end(), finished.passes.begin(),
                             finished.passes.end());
      found = true;
      break;
    }
    if (!found) {
      return Result<SubmissionHandle, std::string>::makeError(
          "fake submitRecordedGraphicsFrame: unknown command buffer");
    }
  }
  recordSubmitFrame(submittedPasses, commandBuffers, batches);
  lastSubmittedFrameHandle =
      SubmissionHandle{.index = nextSubmissionIndex_++, .generation = 1u};
  const uint64_t retireLag =
      static_cast<uint64_t>(std::max(1u, swapchainImageCount)) + 1ull;
  submissions_.push_back(SubmissionState{
      .handle = lastSubmittedFrameHandle,
      .readyFrameIndex = currentFrameIndex_ + retireLag,
  });
  return Result<SubmissionHandle, std::string>::makeResult(
      lastSubmittedFrameHandle);
}

bool FakeGPUDeviceBase::isSubmissionComplete(SubmissionHandle handle) const {
  if (!nuri::isValid(handle)) {
    return true;
  }
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  for (const SubmissionState &submission : submissions_) {
    if (sameHandle(submission.handle, handle)) {
      return currentFrameIndex_ >= submission.readyFrameIndex;
    }
  }
  return false;
}

Result<SubmissionHandle, std::string>
FakeExecutorGPUDevice::submitRecordedGraphicsFrame(
    std::span<const RecordedCommandBufferHandle> commandBuffers,
    std::span<const SubmitBatchMeta> batches) {
  auto baseResult =
      FakeGPUDeviceBase::submitRecordedGraphicsFrame(commandBuffers, batches);
  if (baseResult.hasError()) {
    return baseResult;
  }
  lastSubmitPassCount = recordedPasses.size();

  lastColorTexture = {};
  lastDepthTexture = {};
  lastDependencyBuffer = {};
  lastPreDispatchDependencyBuffer = {};
  lastDrawVertexBuffer = {};

  if (recordedPasses.empty()) {
    return baseResult;
  }

  const RenderPass &pass = recordedPasses[0u];
  lastColorTexture = pass.colorTexture;
  lastDepthTexture = pass.depthTexture;

  if (!pass.dependencyBuffers.empty()) {
    lastDependencyBuffer = pass.dependencyBuffers[0u];
  }
  if (!pass.preDispatches.empty() &&
      !pass.preDispatches[0u].dependencyBuffers.empty()) {
    lastPreDispatchDependencyBuffer =
        pass.preDispatches[0u].dependencyBuffers[0u];
  }
  if (!pass.draws.empty()) {
    lastDrawVertexBuffer = pass.draws[0u].vertexBuffer;
  }

  if (failSubmitFrame) {
    return Result<SubmissionHandle, std::string>::makeError(
        "fake submitFrame failure");
  }

  return baseResult;
}

Result<bool, std::string> FakeGPUDeviceBase::submitComputeDispatches(
    std::span<const ComputeDispatchItem>) {
  return Result<bool, std::string>::makeResult(true);
}

Result<GeometryAllocationHandle, std::string>
FakeGPUDeviceBase::allocateGeometry(std::span<const std::byte>, uint32_t,
                                    std::span<const std::byte>, uint32_t,
                                    std::string_view) {
  return Result<GeometryAllocationHandle, std::string>::makeError(
      "not implemented in fake device");
}

void FakeGPUDeviceBase::releaseGeometry(GeometryAllocationHandle) {}

Result<bool, std::string>
FakeGPUDeviceBase::copyBufferRegions(std::span<const BufferCopyRegion>) {
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeGPUDeviceBase::updateBuffer(BufferHandle, std::span<const std::byte>,
                                size_t) {
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> FakeGPUDeviceBase::readBuffer(BufferHandle, size_t,
                                                        std::span<std::byte>) {
  return Result<bool, std::string>::makeError("not implemented in fake device");
}

std::byte *FakeGPUDeviceBase::getMappedBufferPtr(BufferHandle) {
  return nullptr;
}

void FakeGPUDeviceBase::flushMappedBuffer(BufferHandle, size_t, size_t) {}

Result<bool, std::string>
FakeGPUDeviceBase::readTexture(TextureHandle, const TextureReadbackRegion &,
                               std::span<std::byte>) {
  return Result<bool, std::string>::makeError("not implemented in fake device");
}

void FakeGPUDeviceBase::waitIdle() {
  ++waitIdleCallCount;
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  for (SubmissionState &submission : submissions_) {
    submission.readyFrameIndex = currentFrameIndex_;
  }
}

Result<SubmissionHandle, std::string>
FakeRendererGPUDevice::submitRecordedGraphicsFrame(
    std::span<const RecordedCommandBufferHandle> commandBuffers,
    std::span<const SubmitBatchMeta> batches) {
  auto baseResult =
      FakeGPUDeviceBase::submitRecordedGraphicsFrame(commandBuffers, batches);
  if (baseResult.hasError()) {
    return baseResult;
  }
  submittedPassCount = recordedPasses.size();
  submittedPassLabels.clear();
  submittedPassLabels.reserve(recordedPasses.size());
  for (const RenderPass &pass : recordedPasses) {
    submittedPassLabels.emplace_back(pass.debugLabel);
  }
  return baseResult;
}

bool hasPassLabel(const FakeRendererGPUDevice &gpu, std::string_view label) {
  for (const std::string &entry : gpu.submittedPassLabels) {
    if (entry == label) {
      return true;
    }
  }
  return false;
}

} // namespace nuri::test_support
