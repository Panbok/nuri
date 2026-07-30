#include "render_graph_test_support.h"

#include <algorithm>
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

[[nodiscard]] size_t fakeTextureBytesPerPixel(Format format) {
  switch (format) {
  case Format::R8_UNORM:
    return 1u;
  case Format::R16_UNORM:
    return sizeof(uint16_t);
  case Format::R32_UINT:
    return sizeof(uint32_t);
  case Format::R32_FLOAT:
    return sizeof(float);
  case Format::RG32_FLOAT:
    return sizeof(float) * 2u;
  case Format::RG16_FLOAT:
    return sizeof(uint16_t) * 2u;
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return 4u;
  case Format::RGBA16_FLOAT:
    return 8u;
  case Format::RGBA32_FLOAT:
    return 16u;
  case Format::RGB32_FLOAT:
    return 12u;
  case Format::D16_UNORM:
    return sizeof(uint16_t);
  case Format::D32_FLOAT:
    return sizeof(float);
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::Count:
    break;
  }
  return 0u;
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
  desc.executionMode = pass.executionMode;
  desc.color = pass.color;
  desc.depth = pass.depth;
  desc.useViewport = pass.useViewport;
  desc.viewport = pass.viewport;
  desc.preDispatches = pass.preDispatches;
  desc.recordingSamplers = pass.recordingSamplers;
  desc.draws = pass.draws;
  desc.gpuTimingScope = pass.gpuTimingScope;
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
  outWidth = windowWidth;
  outHeight = windowHeight;
}

void FakeGPUDeviceBase::getFramebufferSize(int32_t &outWidth,
                                           int32_t &outHeight) const {
  outWidth = framebufferWidth;
  outHeight = framebufferHeight;
}

void FakeGPUDeviceBase::resizeSwapchain(int32_t width, int32_t height) {
  windowWidth = width;
  windowHeight = height;
  framebufferWidth = width;
  framebufferHeight = height;
}

Format FakeGPUDeviceBase::getSwapchainFormat() const { return swapchainFormat; }

uint32_t FakeGPUDeviceBase::getSwapchainImageIndex() const { return 0u; }

uint32_t FakeGPUDeviceBase::getSwapchainImageCount() const {
  return swapchainImageCount;
}

double FakeGPUDeviceBase::getTime() const { return 0.0; }

Result<BufferHandle, std::string>
FakeGPUDeviceBase::createBuffer(const BufferDesc &desc, std::string_view) {
  return createBufferImpl(desc);
}

Result<TextureHandle, std::string>
FakeGPUDeviceBase::createTexture(const TextureDesc &desc, std::string_view) {
  return createTextureImpl(desc);
}

Result<SamplerHandle, std::string>
FakeGPUDeviceBase::createSampler(const SamplerDesc &desc, std::string_view) {
  SamplerHandle handle{.index = nextSamplerIndex_++, .generation = 1u};
  ++createdSamplerCount;
  createdSamplerDescs.push_back(desc);
  return Result<SamplerHandle, std::string>::makeResult(handle);
}

Result<BufferHandle, std::string>
FakeGPUDeviceBase::createBufferImpl(const BufferDesc &desc) {
  BufferHandle handle{.index = nextBufferIndex_++, .generation = 1u};
  if (buffers_.size() < handle.index) {
    buffers_.resize(handle.index);
  }
  BufferState &state = buffers_[handle.index - 1u];
  state.handle = handle;
  state.size = desc.size != 0u ? desc.size : desc.data.size();
  state.storage = desc.storage;
  state.bytes.assign(state.size, std::byte{0});
  if (!desc.data.empty()) {
    std::copy(desc.data.begin(), desc.data.end(), state.bytes.begin());
  }
  state.live = true;
  ++createdBufferCount;
  return Result<BufferHandle, std::string>::makeResult(handle);
}

Result<TextureHandle, std::string>
FakeGPUDeviceBase::createTextureImpl(const TextureDesc &desc) {
  TextureHandle handle{.index = nextTextureIndex_++, .generation = 1u};
  if (textures_.size() < handle.index) {
    textures_.resize(handle.index);
  }
  TextureState &state = textures_[handle.index - 1u];
  state.handle = handle;
  state.format = desc.format;
  state.width = desc.dimensions.width;
  state.height = desc.dimensions.height;
  const size_t bytesPerPixel = fakeTextureBytesPerPixel(desc.format);
  const size_t texelCount =
      static_cast<size_t>(std::max(desc.dimensions.width, 1u)) *
      static_cast<size_t>(std::max(desc.dimensions.height, 1u));
  state.bytes.assign(texelCount * bytesPerPixel, std::byte{0});
  state.live = true;
  ++createdTextureCount;
  TextureDesc storedDesc = desc;
  storedDesc.data = {};
  createdTextureDescs.push_back(storedDesc);
  createdTextureData.emplace_back(desc.data.begin(), desc.data.end());
  return Result<TextureHandle, std::string>::makeResult(handle);
}

Result<bool, std::string>
FakeGPUDeviceBase::copyBufferRegion(const BufferCopyRegion &copy) {
  if (!isValid(copy.srcBuffer) || !isValid(copy.dstBuffer) ||
      copy.srcBuffer.index > buffers_.size() ||
      copy.dstBuffer.index > buffers_.size()) {
    return Result<bool, std::string>::makeError(
        "fake buffer copy: invalid buffer");
  }
  BufferState &source = buffers_[copy.srcBuffer.index - 1u];
  BufferState &destination = buffers_[copy.dstBuffer.index - 1u];
  if (!source.live || !destination.live || copy.srcOffset > source.size ||
      copy.size > source.size - copy.srcOffset ||
      copy.dstOffset > destination.size ||
      copy.size > destination.size - copy.dstOffset) {
    return Result<bool, std::string>::makeError(
        "fake buffer copy: range out of bounds");
  }
  std::copy_n(
      source.bytes.begin() + static_cast<std::ptrdiff_t>(copy.srcOffset),
      static_cast<std::ptrdiff_t>(copy.size),
      destination.bytes.begin() + static_cast<std::ptrdiff_t>(copy.dstOffset));
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeGPUDeviceBase::copyTextureRegion(const TextureCopyItem &copy) {
  if (!isValid(copy.sourceTexture) || !isValid(copy.destinationTexture)) {
    return Result<bool, std::string>::makeError(
        "fake texture copy: invalid texture handle");
  }
  if (sameHandle(copy.sourceTexture, copy.destinationTexture)) {
    return Result<bool, std::string>::makeError(
        "fake texture copy: source and destination match");
  }
  if (copy.sourceMipLevel != 0u || copy.destinationMipLevel != 0u ||
      copy.sourceLayer != 0u || copy.destinationLayer != 0u) {
    return Result<bool, std::string>::makeError(
        "fake texture copy: unsupported mip or layer");
  }
  if (copy.width == 0u || copy.height == 0u) {
    return Result<bool, std::string>::makeError(
        "fake texture copy: empty region");
  }

  const TextureState &source = textures_[copy.sourceTexture.index - 1u];
  TextureState &destination = textures_[copy.destinationTexture.index - 1u];
  if (source.format != destination.format) {
    return Result<bool, std::string>::makeError(
        "fake texture copy: format mismatch");
  }
  if (copy.sourceX >= source.width || copy.sourceY >= source.height ||
      copy.width > source.width - copy.sourceX ||
      copy.height > source.height - copy.sourceY ||
      copy.destinationX >= destination.width ||
      copy.destinationY >= destination.height ||
      copy.width > destination.width - copy.destinationX ||
      copy.height > destination.height - copy.destinationY) {
    return Result<bool, std::string>::makeError(
        "fake texture copy: region out of bounds");
  }

  const size_t bytesPerPixel = fakeTextureBytesPerPixel(source.format);
  if (bytesPerPixel == 0u) {
    return Result<bool, std::string>::makeError(
        "fake texture copy: unsupported texture format");
  }
  const size_t rowBytes = static_cast<size_t>(copy.width) * bytesPerPixel;
  for (uint32_t row = 0u; row < copy.height; ++row) {
    const size_t sourceOffset = (static_cast<size_t>(copy.sourceY + row) *
                                     static_cast<size_t>(source.width) +
                                 static_cast<size_t>(copy.sourceX)) *
                                bytesPerPixel;
    const size_t destinationOffset =
        (static_cast<size_t>(copy.destinationY + row) *
             static_cast<size_t>(destination.width) +
         static_cast<size_t>(copy.destinationX)) *
        bytesPerPixel;
    std::copy_n(source.bytes.begin() +
                    static_cast<std::ptrdiff_t>(sourceOffset),
                rowBytes,
                destination.bytes.begin() +
                    static_cast<std::ptrdiff_t>(destinationOffset));
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<BufferHandle, std::string>
FakeExecutorGPUDevice::createBuffer(const BufferDesc &desc, std::string_view) {
  ++createBufferCallCount;
  if (failCreateBufferAtCall != 0u &&
      createBufferCallCount == failCreateBufferAtCall) {
    return Result<BufferHandle, std::string>::makeError(
        "fake createBuffer failure");
  }
  return createBufferImpl(desc);
}

Result<TextureHandle, std::string>
FakeExecutorGPUDevice::createTexture(const TextureDesc &desc,
                                     std::string_view) {
  ++createTextureCallCount;
  if (failCreateTextureAtCall != 0u &&
      createTextureCallCount == failCreateTextureAtCall) {
    return Result<TextureHandle, std::string>::makeError(
        "fake createTexture failure");
  }
  return createTextureImpl(desc);
}

Result<bool, std::string>
FakeExecutorGPUDevice::recordGraphicsRangeWithReferences(
    RecordingContextHandle ctx, std::span<const GraphicsRecordingStep> steps,
    const GraphicsRecordingReferences &references) {
  ++combinedRecordingCallCount;
  retainedBuffers.assign(references.buffers.begin(), references.buffers.end());
  retainedTextures.assign(references.textures.begin(),
                          references.textures.end());
  retainedSamplers.assign(references.samplers.begin(),
                          references.samplers.end());
  retainedAccelerationStructures.assign(
      references.accelerationStructures.begin(),
      references.accelerationStructures.end());
  retainedRenderPipelines.assign(references.renderPipelines.begin(),
                                 references.renderPipelines.end());
  retainedComputePipelines.assign(references.computePipelines.begin(),
                                  references.computePipelines.end());
  retainedMeshletPipelines.assign(references.meshletPipelines.begin(),
                                  references.meshletPipelines.end());
  retainedRayQueryBindings.assign(references.rayQueryBindings.begin(),
                                  references.rayQueryBindings.end());
  return GPUDevice::recordGraphicsRange(ctx, steps);
}

Result<TextureHandle, std::string>
FakeGPUDeviceBase::createFramebufferTexture(const TextureDesc &desc,
                                            std::string_view) {
  TextureDesc resizedDesc = desc;
  resizedDesc.dimensions = {
      .width = static_cast<uint32_t>(std::max(framebufferWidth, 1)),
      .height = static_cast<uint32_t>(std::max(framebufferHeight, 1)),
      .depth = 1u};
  if (resizedDesc.type == TextureType::Count) {
    resizedDesc.type = TextureType::Texture2D;
  }
  if (resizedDesc.format == Format::Count) {
    resizedDesc.format = Format::RGBA8_UNORM;
  }
  if (resizedDesc.usage == TextureUsage::Count) {
    resizedDesc.usage = TextureUsage::AttachmentSampled;
  }
  if (resizedDesc.storage == Storage::Count) {
    resizedDesc.storage = Storage::Device;
  }
  if (resizedDesc.numLayers == 0u) {
    resizedDesc.numLayers = 1u;
  }
  if (resizedDesc.numSamples == 0u) {
    resizedDesc.numSamples = 1u;
  }
  if (resizedDesc.numMipLevels == 0u) {
    resizedDesc.numMipLevels = 1u;
  }
  if (resizedDesc.dataNumMipLevels == 0u) {
    resizedDesc.dataNumMipLevels = 1u;
  }
  return createTextureImpl(resizedDesc);
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

Result<MeshletPipelineHandle, std::string>
FakeGPUDeviceBase::createMeshletPipeline(const MeshletPipelineDesc &,
                                         std::string_view) {
  return Result<MeshletPipelineHandle, std::string>::makeError(
      "meshlet pipelines are unsupported in fake device");
}

void FakeGPUDeviceBase::destroyRenderPipeline(RenderPipelineHandle) {}

void FakeGPUDeviceBase::destroyComputePipeline(ComputePipelineHandle) {}

void FakeGPUDeviceBase::destroyMeshletPipeline(MeshletPipelineHandle) {}

void FakeGPUDeviceBase::destroyBuffer(BufferHandle buffer) {
  destroyBufferImpl(buffer);
}

void FakeGPUDeviceBase::destroyTexture(TextureHandle texture) {
  destroyTextureImpl(texture);
}

void FakeGPUDeviceBase::destroyBufferImpl(BufferHandle buffer) {
  if (!nuri::isValid(buffer) || buffer.index == 0u ||
      buffer.index > buffers_.size()) {
    return;
  }
  BufferState &state = buffers_[buffer.index - 1u];
  if (state.live && sameHandle(state.handle, buffer)) {
    state.live = false;
    state.bytes.clear();
    state.size = 0u;
    ++destroyedBufferCount;
  }
}

void FakeGPUDeviceBase::destroyTextureImpl(TextureHandle texture) {
  if (!nuri::isValid(texture) || texture.index == 0u ||
      texture.index > textures_.size()) {
    return;
  }
  TextureState &state = textures_[texture.index - 1u];
  if (state.live && sameHandle(state.handle, texture)) {
    state.live = false;
    ++destroyedTextureCount;
  }
}

void FakeGPUDeviceBase::destroyShaderModule(ShaderHandle) {}

bool FakeGPUDeviceBase::isValid(BufferHandle h) const {
  if (!nuri::isValid(h) || h.index == 0u || h.index > buffers_.size()) {
    return false;
  }
  const BufferState &state = buffers_[h.index - 1u];
  return state.live && sameHandle(state.handle, h);
}

bool FakeGPUDeviceBase::isValid(TextureHandle h) const {
  if (!nuri::isValid(h) || h.index == 0u || h.index > textures_.size()) {
    return false;
  }
  const TextureState &state = textures_[h.index - 1u];
  return state.live && sameHandle(state.handle, h);
}

bool FakeGPUDeviceBase::isValid(SamplerHandle h) const {
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

bool FakeGPUDeviceBase::isValid(MeshletPipelineHandle h) const {
  return nuri::isValid(h);
}

Format FakeGPUDeviceBase::getTextureFormat(TextureHandle h) const {
  if (!nuri::isValid(h) || h.index == 0u || h.index > textures_.size()) {
    return Format::Count;
  }
  return textures_[h.index - 1u].format;
}

TextureDimensions
FakeGPUDeviceBase::getTextureDimensions(TextureHandle h) const {
  if (!nuri::isValid(h) || h.index == 0u || h.index > textures_.size()) {
    return {};
  }
  const TextureState &state = textures_[h.index - 1u];
  return TextureDimensions{
      .width = state.width,
      .height = state.height,
      .depth = 1u,
  };
}

TextureCompressionCaps FakeGPUDeviceBase::getTextureCompressionCaps() const {
  return {};
}

GpuMultisampleCapabilities
FakeGPUDeviceBase::getMultisampleCapabilities() const {
  return multisampleCapabilities;
}

bool FakeGPUDeviceBase::supportsFeature(GPUFeature feature) const {
  (void)feature;
  return false;
}

MeshletLimits FakeGPUDeviceBase::getMeshletLimits() const { return {}; }

bool FakeGPUDeviceBase::supportsSampledImageLinearFiltering(Format) const {
  return false;
}

uint32_t FakeGPUDeviceBase::getTextureBindlessIndex(TextureHandle) const {
  return 1u;
}

void FakeGPUDeviceBase::destroySampler(SamplerHandle sampler) {
  if (nuri::isValid(sampler)) {
    ++destroyedSamplerCount;
  }
}

uint32_t FakeGPUDeviceBase::getSamplerBindlessIndex(SamplerHandle h) const {
  return h.index;
}

uint8_t FakeGPUDeviceBase::getMaxSamplerAnisotropy() const { return 16u; }

uint32_t FakeGPUDeviceBase::getLinearRepeatSamplerBindlessIndex(
    bool useMipmaps, uint8_t maxAnisotropy) const {
  if (!useMipmaps) {
    return 1u;
  }
  if (maxAnisotropy > 1u) {
    return 2u;
  }
  return 0u;
}

uint32_t FakeGPUDeviceBase::getDefaultSamplerBindlessIndex() const {
  return 0u;
}

uint32_t FakeGPUDeviceBase::getCubemapSamplerBindlessIndex() const {
  return 0u;
}

uint64_t FakeGPUDeviceBase::getBufferDeviceAddress(BufferHandle h,
                                                   size_t offset) const {
  ++bufferDeviceAddressCallCount;
  if (!isValid(h)) {
    return 0ull;
  }
  return 0x100000000ull + (static_cast<uint64_t>(h.generation) << 24u) +
         (static_cast<uint64_t>(h.index) << 12u) + offset;
}

bool FakeGPUDeviceBase::resolveGeometry(GeometryAllocationHandle handle,
                                        GeometryAllocationView &out) const {
  const auto it =
      std::ranges::find_if(geometries_, [handle](const GeometryState &state) {
        return state.live && state.handle.index == handle.index &&
               state.handle.generation == handle.generation;
      });
  if (it == geometries_.end()) {
    return false;
  }
  out = it->view;
  return true;
}

uint64_t FakeGPUDeviceBase::geometryMutationVersion() const {
  return geometryMutationVersion_;
}

GpuTimingReport FakeGPUDeviceBase::getLatestCompletedGpuTimingReport() const {
  ++latestGpuTimingReportFetchCount;
  return latestCompletedGpuTimingReport;
}

size_t FakeGPUDeviceBase::drainCompletedGpuTimingReports(
    std::span<GpuTimingReport> outReports) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  const size_t count =
      std::min(outReports.size(), completedGpuTimingReports.size());
  for (size_t i = 0u; i < count; ++i) {
    outReports[i] = completedGpuTimingReports[i];
  }
  completedGpuTimingReports.erase(completedGpuTimingReports.begin(),
                                  completedGpuTimingReports.begin() +
                                      static_cast<std::ptrdiff_t>(count));
  return count;
}

uint64_t FakeGPUDeviceBase::droppedGpuTimingReportCount() const {
  return droppedGpuTimingReports;
}

void FakeGPUDeviceBase::enqueueCompletedGpuTimingReport(
    const GpuTimingReport &report, size_t queueCapacity) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  latestCompletedGpuTimingReport = report;
  if (completedGpuTimingReports.size() >= queueCapacity) {
    completedGpuTimingReports.erase(completedGpuTimingReports.begin());
    ++droppedGpuTimingReports;
  }
  completedGpuTimingReports.push_back(report);
}

Result<bool, std::string> FakeGPUDeviceBase::beginFrame(uint64_t frameIndex) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  currentFrameIndex_ = frameIndex;
  activeRecordingContexts_.clear();
  finishedCommandBuffers_.clear();
  finishCallCount_ = 0u;
  recordedPasses.clear();
  recordedMeshDispatchStorage_.clear();
  recordedBufferCopyStorage_.clear();
  recordedTextureCopyStorage_.clear();
  submittedCommandBuffers.clear();
  submittedBatches.clear();
  hasPreparedFrameOutput_ = false;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> FakeGPUDeviceBase::prepareFrameOutput() {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  hasPreparedFrameOutput_ = true;
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

Result<bool, std::string> FakeGPUDeviceBase::recordGraphicsBarriers(
    RecordingContextHandle ctx,
    std::span<const GraphicsBarrierRecord> barriers) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  for (auto &state : activeRecordingContexts_) {
    if (sameHandle(state.handle, ctx)) {
      recordedBarrierBatchCounts.push_back(
          static_cast<uint32_t>(barriers.size()));
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
      for (const BufferCopyRegion &copy : pass.bufferCopies) {
        auto copyResult = copyBufferRegion(copy);
        if (copyResult.hasError()) {
          return copyResult;
        }
      }
      for (const TextureCopyItem &copy : pass.textureCopies) {
        auto copyResult = copyTextureRegion(copy);
        if (copyResult.hasError()) {
          return copyResult;
        }
      }
      RenderPass copiedPass = pass;
      state.meshDispatchStorage.emplace_back(pass.meshDispatches.begin(),
                                             pass.meshDispatches.end());
      std::vector<MeshDispatchItem> &meshDispatches =
          state.meshDispatchStorage.back();
      copiedPass.meshDispatches = std::span<const MeshDispatchItem>(
          meshDispatches.data(), meshDispatches.size());
      state.bufferCopyStorage.emplace_back(pass.bufferCopies.begin(),
                                           pass.bufferCopies.end());
      std::vector<BufferCopyRegion> &bufferCopies =
          state.bufferCopyStorage.back();
      copiedPass.bufferCopies = std::span<const BufferCopyRegion>(
          bufferCopies.data(), bufferCopies.size());
      state.textureCopyStorage.emplace_back(pass.textureCopies.begin(),
                                            pass.textureCopies.end());
      std::vector<TextureCopyItem> &textureCopies =
          state.textureCopyStorage.back();
      copiedPass.textureCopies = std::span<const TextureCopyItem>(
          textureCopies.data(), textureCopies.size());
      state.passes.push_back(copiedPass);
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
    finished.meshDispatchStorage =
        std::move(activeRecordingContexts_[i].meshDispatchStorage);
    finished.bufferCopyStorage =
        std::move(activeRecordingContexts_[i].bufferCopyStorage);
    finished.textureCopyStorage =
        std::move(activeRecordingContexts_[i].textureCopyStorage);
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
  recordedMeshDispatchStorage_.clear();
  recordedBufferCopyStorage_.clear();
  recordedTextureCopyStorage_.clear();
  recordedMeshDispatchStorage_.reserve(recordedPasses.size());
  recordedBufferCopyStorage_.reserve(recordedPasses.size());
  recordedTextureCopyStorage_.reserve(recordedPasses.size());
  for (RenderPass &pass : recordedPasses) {
    recordedMeshDispatchStorage_.emplace_back(pass.meshDispatches.begin(),
                                              pass.meshDispatches.end());
    std::vector<MeshDispatchItem> &meshDispatches =
        recordedMeshDispatchStorage_.back();
    pass.meshDispatches = std::span<const MeshDispatchItem>(
        meshDispatches.data(), meshDispatches.size());
    recordedBufferCopyStorage_.emplace_back(pass.bufferCopies.begin(),
                                            pass.bufferCopies.end());
    std::vector<BufferCopyRegion> &bufferCopies =
        recordedBufferCopyStorage_.back();
    pass.bufferCopies = std::span<const BufferCopyRegion>(bufferCopies.data(),
                                                          bufferCopies.size());
    recordedTextureCopyStorage_.emplace_back(pass.textureCopies.begin(),
                                             pass.textureCopies.end());
    std::vector<TextureCopyItem> &textureCopies =
        recordedTextureCopyStorage_.back();
    pass.textureCopies = std::span<const TextureCopyItem>(textureCopies.data(),
                                                          textureCopies.size());
  }
  submittedCommandBuffers.assign(commandBuffers.begin(), commandBuffers.end());
  submittedBatches.assign(batches.begin(), batches.end());
}

Result<SubmittedGraphicsFrame, std::string>
FakeGPUDeviceBase::submitRecordedGraphicsFrame(
    std::span<const RecordedCommandBufferHandle> commandBuffers,
    std::span<const SubmitBatchMeta> batches) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  const bool wantsPresent =
      std::any_of(batches.begin(), batches.end(), [](const SubmitBatchMeta &b) {
        return b.presentsFrameOutput && b.commandBufferCount > 0u;
      });
  if (wantsPresent && !hasPreparedFrameOutput_) {
    return Result<SubmittedGraphicsFrame, std::string>::makeError(
        "fake submitRecordedGraphicsFrame: frame output was not prepared");
  }
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
      return Result<SubmittedGraphicsFrame, std::string>::makeError(
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
  if (wantsPresent) {
    hasPreparedFrameOutput_ = false;
  }
  return Result<SubmittedGraphicsFrame, std::string>::makeResult(
      SubmittedGraphicsFrame{.submission = lastSubmittedFrameHandle});
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

Result<bool, std::string>
FakeGPUDeviceBase::makeSubmissionVisibleToGraphics(SubmissionHandle) {
  return Result<bool, std::string>::makeResult(true);
}

Result<SubmittedGraphicsFrame, std::string>
FakeExecutorGPUDevice::submitRecordedGraphicsFrame(
    std::span<const RecordedCommandBufferHandle> commandBuffers,
    std::span<const SubmitBatchMeta> batches) {
  if (failSubmitFrame) {
    ++submitCount;
    return Result<SubmittedGraphicsFrame, std::string>::makeError(
        "fake submitFrame failure");
  }
  auto baseResult =
      FakeGPUDeviceBase::submitRecordedGraphicsFrame(commandBuffers, batches);
  if (baseResult.hasError()) {
    return baseResult;
  }
  lastSubmitPassCount = recordedPasses.size();

  lastColorTexture = {};
  lastDepthTexture = {};
  lastDrawVertexBuffer = {};

  if (recordedPasses.empty()) {
    return baseResult;
  }

  const RenderPass &pass = recordedPasses[0u];
  lastColorTexture = pass.colorTexture;
  lastDepthTexture = pass.depthTexture;

  if (!pass.draws.empty()) {
    lastDrawVertexBuffer = pass.draws[0u].vertexBuffer;
  }

  if (failPresentFrame) {
    baseResult.value().presentationError = "fake presentFrame failure";
  }

  return baseResult;
}

Result<bool, std::string> FakeGPUDeviceBase::submitComputeDispatches(
    std::span<const ComputeDispatchItem>) {
  return Result<bool, std::string>::makeResult(true);
}

Result<GeometryAllocationHandle, std::string>
FakeGPUDeviceBase::allocateGeometry(std::span<const std::byte> vertexBytes,
                                    uint32_t vertexCount,
                                    std::span<const std::byte> indexBytes,
                                    uint32_t indexCount, std::string_view) {
  auto vertexBuffer = createBufferImpl(BufferDesc{
      .usage = BufferUsage::Vertex,
      .storage = Storage::Device,
      .size = vertexBytes.size(),
      .data = vertexBytes,
  });
  if (vertexBuffer.hasError()) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        vertexBuffer.error());
  }
  auto indexBuffer = createBufferImpl(BufferDesc{
      .usage = BufferUsage::Index,
      .storage = Storage::Device,
      .size = indexBytes.size(),
      .data = indexBytes,
  });
  if (indexBuffer.hasError()) {
    destroyBufferImpl(vertexBuffer.value());
    return Result<GeometryAllocationHandle, std::string>::makeError(
        indexBuffer.error());
  }

  const GeometryAllocationHandle handle{
      .index = nextGeometryIndex_++,
      .generation = 1u,
  };
  geometries_.push_back(GeometryState{
      .handle = handle,
      .view =
          GeometryAllocationView{
              .vertexBuffer = vertexBuffer.value(),
              .vertexByteOffset = 0u,
              .vertexByteSize = vertexBytes.size(),
              .indexBuffer = indexBuffer.value(),
              .indexByteOffset = 0u,
              .indexByteSize = indexBytes.size(),
              .vertexCount = vertexCount,
              .indexCount = indexCount,
          },
      .live = true,
  });
  ++geometryMutationVersion_;
  return Result<GeometryAllocationHandle, std::string>::makeResult(handle);
}

void FakeGPUDeviceBase::releaseGeometry(GeometryAllocationHandle handle) {
  auto it =
      std::ranges::find_if(geometries_, [handle](const GeometryState &state) {
        return state.live && state.handle.index == handle.index &&
               state.handle.generation == handle.generation;
      });
  if (it == geometries_.end()) {
    return;
  }
  destroyBufferImpl(it->view.vertexBuffer);
  destroyBufferImpl(it->view.indexBuffer);
  it->live = false;
  ++geometryMutationVersion_;
}

Result<SubmissionHandle, std::string>
FakeGPUDeviceBase::submitBackgroundBufferCopies(
    std::span<const BufferCopyRegion> regions, std::string_view) {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  ++backgroundCopySubmitCount;
  backgroundCopyBatchSizes.push_back(regions.size());
  if (failBackgroundCopySubmitAtCall != 0u &&
      backgroundCopySubmitCount == failBackgroundCopySubmitAtCall) {
    return Result<SubmissionHandle, std::string>::makeError(
        "fake submitBackgroundBufferCopies failure");
  }

  size_t copiedBytes = 0u;
  for (const BufferCopyRegion &region : regions) {
    if (region.size == 0u) {
      continue;
    }
    if (!isValid(region.srcBuffer) || !isValid(region.dstBuffer)) {
      return Result<SubmissionHandle, std::string>::makeError(
          "fake submitBackgroundBufferCopies: invalid buffer");
    }
    BufferState &src = buffers_[region.srcBuffer.index - 1u];
    BufferState &dst = buffers_[region.dstBuffer.index - 1u];
    if (region.srcOffset > src.size ||
        region.size > src.size - region.srcOffset) {
      return Result<SubmissionHandle, std::string>::makeError(
          "fake submitBackgroundBufferCopies: source range out of bounds");
    }
    if (region.dstOffset > dst.size ||
        region.size > dst.size - region.dstOffset) {
      return Result<SubmissionHandle, std::string>::makeError(
          "fake submitBackgroundBufferCopies: destination range out of bounds");
    }
    std::copy_n(
        src.bytes.begin() + static_cast<std::ptrdiff_t>(region.srcOffset),
        static_cast<std::ptrdiff_t>(region.size),
        dst.bytes.begin() + static_cast<std::ptrdiff_t>(region.dstOffset));
    copiedBytes += static_cast<size_t>(region.size);
  }

  backgroundCopyBatchBytes.push_back(copiedBytes);
  lastBackgroundCopyHandle =
      SubmissionHandle{.index = nextSubmissionIndex_++, .generation = 1u};
  const uint64_t retireLag =
      static_cast<uint64_t>(std::max(1u, swapchainImageCount)) + 1ull;
  submissions_.push_back(SubmissionState{
      .handle = lastBackgroundCopyHandle,
      .readyFrameIndex = currentFrameIndex_ + retireLag,
  });
  return Result<SubmissionHandle, std::string>::makeResult(
      lastBackgroundCopyHandle);
}

Result<SubmissionHandle, std::string>
FakeGPUDeviceBase::captureWorkCompletion() {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  const SubmissionHandle handle{
      .index = nextSubmissionIndex_++,
      .generation = 1u,
  };
  const uint64_t retireLag =
      static_cast<uint64_t>(std::max(1u, swapchainImageCount)) + 1ull;
  submissions_.push_back(SubmissionState{
      .handle = handle,
      .readyFrameIndex = currentFrameIndex_ + retireLag,
  });
  return Result<SubmissionHandle, std::string>::makeResult(handle);
}

Result<SubmissionHandle, std::string>
FakeGPUDeviceBase::submitPendingUploads() {
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  ++pendingUploadSubmitCount;
  const SubmissionHandle handle{
      .index = nextSubmissionIndex_++,
      .generation = 1u,
  };
  const uint64_t retireLag =
      static_cast<uint64_t>(std::max(1u, swapchainImageCount)) + 1ull;
  submissions_.push_back(SubmissionState{
      .handle = handle,
      .readyFrameIndex = currentFrameIndex_ + retireLag,
  });
  return Result<SubmissionHandle, std::string>::makeResult(handle);
}

bool FakeGPUDeviceBase::assetUploadsUseDedicatedCopyQueue() const noexcept {
  return false;
}

Result<bool, std::string> FakeGPUDeviceBase::updateBuffer(
    BufferHandle buffer, std::span<const std::byte> data, size_t offset) {
  if (!isValid(buffer)) {
    return Result<bool, std::string>::makeError(
        "fake updateBuffer: invalid buffer");
  }
  ++updateBufferCallCount;
  BufferState &state = buffers_[buffer.index - 1u];
  if (offset > state.size || data.size() > state.size - offset) {
    return Result<bool, std::string>::makeError(
        "fake updateBuffer: range out of bounds");
  }
  std::copy(data.begin(), data.end(),
            state.bytes.begin() + static_cast<std::ptrdiff_t>(offset));
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeGPUDeviceBase::updateBuffers(std::span<const BufferUpdate> updates) {
  ++updateBufferBatchCallCount;
  updateBufferBatchSizes.push_back(updates.size());
  for (const BufferUpdate &update : updates) {
    auto result = updateBuffer(update.buffer, update.data, update.offset);
    if (result.hasError())
      return result;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeGPUDeviceBase::readBuffer(BufferHandle buffer, size_t offset,
                              std::span<std::byte> outBytes) {
  if (!isValid(buffer)) {
    return Result<bool, std::string>::makeError(
        "fake readBuffer: invalid buffer");
  }
  const BufferState &state = buffers_[buffer.index - 1u];
  if (rejectDeviceLocalReadBuffer && state.storage == Storage::Device) {
    return Result<bool, std::string>::makeError(
        "fake readBuffer: buffer is not host-visible/mapped");
  }
  if (offset > state.size || outBytes.size() > state.size - offset) {
    return Result<bool, std::string>::makeError(
        "fake readBuffer: range out of bounds");
  }
  std::copy(state.bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            state.bytes.begin() +
                static_cast<std::ptrdiff_t>(offset + outBytes.size()),
            outBytes.begin());
  return Result<bool, std::string>::makeResult(true);
}

std::byte *FakeGPUDeviceBase::getMappedBufferPtr(BufferHandle) {
  return nullptr;
}

void FakeGPUDeviceBase::flushMappedBuffer(BufferHandle, size_t, size_t) {}

Result<bool, std::string>
FakeGPUDeviceBase::seedTextureBytes(TextureHandle texture,
                                    std::span<const std::byte> bytes) {
  if (!nuri::isValid(texture) || texture.index == 0u ||
      texture.index > textures_.size()) {
    return Result<bool, std::string>::makeError("invalid texture handle");
  }
  TextureState &state = textures_[texture.index - 1u];
  if (!state.live || !sameHandle(state.handle, texture)) {
    return Result<bool, std::string>::makeError("texture is not live");
  }
  if (state.bytes.size() != bytes.size()) {
    return Result<bool, std::string>::makeError(
        "seedTextureBytes: size mismatch");
  }
  std::copy(bytes.begin(), bytes.end(), state.bytes.begin());
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeGPUDeviceBase::readTexture(TextureHandle texture,
                               const TextureReadbackRegion &region,
                               std::span<std::byte> outBytes) {
  if (!nuri::isValid(texture) || texture.index == 0u ||
      texture.index > textures_.size()) {
    return Result<bool, std::string>::makeError("invalid texture handle");
  }
  const TextureState &state = textures_[texture.index - 1u];
  if (!state.live || !sameHandle(state.handle, texture)) {
    return Result<bool, std::string>::makeError("texture is not live");
  }
  if (region.width == 0u || region.height == 0u || region.mipLevel != 0u ||
      region.layer != 0u) {
    return Result<bool, std::string>::makeError("unsupported readback region");
  }
  const size_t bytesPerPixel = fakeTextureBytesPerPixel(state.format);
  if (bytesPerPixel == 0u) {
    return Result<bool, std::string>::makeError("unsupported texture format");
  }
  if (region.x >= state.width || region.y >= state.height ||
      region.width > state.width - region.x ||
      region.height > state.height - region.y) {
    return Result<bool, std::string>::makeError(
        "readback region out of bounds");
  }
  const size_t expectedSize = static_cast<size_t>(region.width) *
                              static_cast<size_t>(region.height) *
                              bytesPerPixel;
  if (outBytes.size() < expectedSize) {
    return Result<bool, std::string>::makeError("readback buffer too small");
  }
  const size_t rowBytes = static_cast<size_t>(region.width) * bytesPerPixel;
  for (uint32_t row = 0u; row < region.height; ++row) {
    const size_t srcOffset = (static_cast<size_t>(region.y + row) *
                                  static_cast<size_t>(state.width) +
                              static_cast<size_t>(region.x)) *
                             bytesPerPixel;
    const size_t dstOffset = static_cast<size_t>(row) * rowBytes;
    std::copy_n(state.bytes.begin() + static_cast<std::ptrdiff_t>(srcOffset),
                rowBytes,
                outBytes.begin() + static_cast<std::ptrdiff_t>(dstOffset));
  }
  return Result<bool, std::string>::makeResult(true);
}

void FakeGPUDeviceBase::waitIdle() {
  ++waitIdleCallCount;
  const std::lock_guard<std::mutex> lock(recordingStateMutex_);
  for (SubmissionState &submission : submissions_) {
    submission.readyFrameIndex = currentFrameIndex_;
  }
}

Result<SubmittedGraphicsFrame, std::string>
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
