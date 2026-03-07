#include "render_graph_test_support.h"

#include <cstdlib>

namespace nuri::test_support {

bool sameHandle(BufferHandle lhs, BufferHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

bool sameHandle(TextureHandle lhs, TextureHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

bool sameBuffer(BufferHandle lhs, BufferHandle rhs) { return sameHandle(lhs, rhs); }

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
    _putenv_s(name_.c_str(), "");
  }
#else
  if (hadOldValue_) {
    setenv(name_.c_str(), oldValue_.c_str(), 1);
  } else {
    unsetenv(name_.c_str());
  }
#endif
}

bool FakeExecutorGPUDevice::shouldClose() const { return false; }

void FakeExecutorGPUDevice::getWindowSize(int32_t &outWidth,
                                          int32_t &outHeight) const {
  outWidth = 1280;
  outHeight = 720;
}

void FakeExecutorGPUDevice::getFramebufferSize(int32_t &outWidth,
                                               int32_t &outHeight) const {
  outWidth = 1280;
  outHeight = 720;
}

void FakeExecutorGPUDevice::resizeSwapchain(int32_t, int32_t) {}

Format FakeExecutorGPUDevice::getSwapchainFormat() const {
  return Format::RGBA8_UNORM;
}

uint32_t FakeExecutorGPUDevice::getSwapchainImageIndex() const { return 0u; }

uint32_t FakeExecutorGPUDevice::getSwapchainImageCount() const {
  return swapchainImageCount;
}

double FakeExecutorGPUDevice::getTime() const { return 0.0; }

Result<BufferHandle, std::string>
FakeExecutorGPUDevice::createBuffer(const BufferDesc &, std::string_view) {
  ++createBufferCallCount;
  if (failCreateBufferAtCall != 0u &&
      createBufferCallCount == failCreateBufferAtCall) {
    return Result<BufferHandle, std::string>::makeError(
        "fake createBuffer failure");
  }
  BufferHandle handle{.index = nextBufferIndex++, .generation = 1u};
  ++createdBufferCount;
  return Result<BufferHandle, std::string>::makeResult(handle);
}

Result<TextureHandle, std::string>
FakeExecutorGPUDevice::createTexture(const TextureDesc &, std::string_view) {
  ++createTextureCallCount;
  if (failCreateTextureAtCall != 0u &&
      createTextureCallCount == failCreateTextureAtCall) {
    return Result<TextureHandle, std::string>::makeError(
        "fake createTexture failure");
  }
  TextureHandle handle{.index = nextTextureIndex++, .generation = 1u};
  ++createdTextureCount;
  return Result<TextureHandle, std::string>::makeResult(handle);
}

Result<TextureHandle, std::string>
FakeExecutorGPUDevice::createFramebufferTexture(const TextureDesc &,
                                                std::string_view) {
  return Result<TextureHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<TextureHandle, std::string> FakeExecutorGPUDevice::createDepthBuffer() {
  return Result<TextureHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<ShaderHandle, std::string>
FakeExecutorGPUDevice::createShaderModule(const ShaderDesc &) {
  return Result<ShaderHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<RenderPipelineHandle, std::string>
FakeExecutorGPUDevice::createRenderPipeline(const RenderPipelineDesc &,
                                            std::string_view) {
  return Result<RenderPipelineHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<ComputePipelineHandle, std::string>
FakeExecutorGPUDevice::createComputePipeline(const ComputePipelineDesc &,
                                             std::string_view) {
  return Result<ComputePipelineHandle, std::string>::makeError(
      "not implemented in fake device");
}

void FakeExecutorGPUDevice::destroyRenderPipeline(RenderPipelineHandle) {}

void FakeExecutorGPUDevice::destroyComputePipeline(ComputePipelineHandle) {}

void FakeExecutorGPUDevice::destroyBuffer(BufferHandle buffer) {
  if (nuri::isValid(buffer)) {
    ++destroyedBufferCount;
  }
}

void FakeExecutorGPUDevice::destroyTexture(TextureHandle texture) {
  if (nuri::isValid(texture)) {
    ++destroyedTextureCount;
  }
}

void FakeExecutorGPUDevice::destroyShaderModule(ShaderHandle) {}

bool FakeExecutorGPUDevice::isValid(BufferHandle h) const {
  return nuri::isValid(h);
}

bool FakeExecutorGPUDevice::isValid(TextureHandle h) const {
  return nuri::isValid(h);
}

bool FakeExecutorGPUDevice::isValid(ShaderHandle h) const {
  return nuri::isValid(h);
}

bool FakeExecutorGPUDevice::isValid(RenderPipelineHandle h) const {
  return nuri::isValid(h);
}

bool FakeExecutorGPUDevice::isValid(ComputePipelineHandle h) const {
  return nuri::isValid(h);
}

Format FakeExecutorGPUDevice::getTextureFormat(TextureHandle) const {
  return Format::RGBA8_UNORM;
}

uint32_t FakeExecutorGPUDevice::getTextureBindlessIndex(TextureHandle) const {
  return 0u;
}

uint32_t FakeExecutorGPUDevice::getDefaultSamplerBindlessIndex() const {
  return 0u;
}

uint32_t FakeExecutorGPUDevice::getCubemapSamplerBindlessIndex() const {
  return 0u;
}

uint64_t FakeExecutorGPUDevice::getBufferDeviceAddress(BufferHandle,
                                                       size_t) const {
  return 0ull;
}

bool FakeExecutorGPUDevice::resolveGeometry(GeometryAllocationHandle,
                                            GeometryAllocationView &) const {
  return false;
}

Result<bool, std::string>
FakeExecutorGPUDevice::beginFrame(uint64_t) {
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeExecutorGPUDevice::submitFrame(const RenderFrame &frame) {
  ++submitCount;
  lastSubmitPassCount = frame.passes.size();

  lastColorTexture = {};
  lastDepthTexture = {};
  lastDependencyBuffer = {};
  lastPreDispatchDependencyBuffer = {};
  lastDrawVertexBuffer = {};

  if (frame.passes.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  const RenderPass &pass = frame.passes[0u];
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
    return Result<bool, std::string>::makeError("fake submitFrame failure");
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeExecutorGPUDevice::submitComputeDispatches(
    std::span<const ComputeDispatchItem>) {
  return Result<bool, std::string>::makeResult(true);
}

Result<GeometryAllocationHandle, std::string>
FakeExecutorGPUDevice::allocateGeometry(std::span<const std::byte>, uint32_t,
                                        std::span<const std::byte>, uint32_t,
                                        std::string_view) {
  return Result<GeometryAllocationHandle, std::string>::makeError(
      "not implemented in fake device");
}

void FakeExecutorGPUDevice::releaseGeometry(GeometryAllocationHandle) {}

Result<bool, std::string>
FakeExecutorGPUDevice::copyBufferRegions(std::span<const BufferCopyRegion>) {
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeExecutorGPUDevice::updateBuffer(BufferHandle, std::span<const std::byte>,
                                    size_t) {
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeExecutorGPUDevice::readBuffer(BufferHandle, size_t, std::span<std::byte>) {
  return Result<bool, std::string>::makeError("not implemented in fake device");
}

std::byte *FakeExecutorGPUDevice::getMappedBufferPtr(BufferHandle) {
  return nullptr;
}

void FakeExecutorGPUDevice::flushMappedBuffer(BufferHandle, size_t, size_t) {}

Result<bool, std::string>
FakeExecutorGPUDevice::readTexture(TextureHandle,
                                   const TextureReadbackRegion &,
                                   std::span<std::byte>) {
  return Result<bool, std::string>::makeError("not implemented in fake device");
}

void FakeExecutorGPUDevice::waitIdle() {}

bool FakeRendererGPUDevice::shouldClose() const { return false; }

void FakeRendererGPUDevice::getWindowSize(int32_t &outWidth,
                                          int32_t &outHeight) const {
  outWidth = 1280;
  outHeight = 720;
}

void FakeRendererGPUDevice::getFramebufferSize(int32_t &outWidth,
                                               int32_t &outHeight) const {
  outWidth = 1280;
  outHeight = 720;
}

void FakeRendererGPUDevice::resizeSwapchain(int32_t, int32_t) {}

Format FakeRendererGPUDevice::getSwapchainFormat() const {
  return Format::RGBA8_UNORM;
}

uint32_t FakeRendererGPUDevice::getSwapchainImageIndex() const { return 0u; }

uint32_t FakeRendererGPUDevice::getSwapchainImageCount() const { return 2u; }

double FakeRendererGPUDevice::getTime() const { return 0.0; }

Result<BufferHandle, std::string>
FakeRendererGPUDevice::createBuffer(const BufferDesc &, std::string_view) {
  return Result<BufferHandle, std::string>::makeResult(
      BufferHandle{.index = 1u, .generation = 1u});
}

Result<TextureHandle, std::string>
FakeRendererGPUDevice::createTexture(const TextureDesc &, std::string_view) {
  return Result<TextureHandle, std::string>::makeResult(
      TextureHandle{.index = 1u, .generation = 1u});
}

Result<TextureHandle, std::string>
FakeRendererGPUDevice::createFramebufferTexture(const TextureDesc &,
                                                std::string_view) {
  return Result<TextureHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<TextureHandle, std::string> FakeRendererGPUDevice::createDepthBuffer() {
  return Result<TextureHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<ShaderHandle, std::string>
FakeRendererGPUDevice::createShaderModule(const ShaderDesc &) {
  return Result<ShaderHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<RenderPipelineHandle, std::string>
FakeRendererGPUDevice::createRenderPipeline(const RenderPipelineDesc &,
                                            std::string_view) {
  return Result<RenderPipelineHandle, std::string>::makeError(
      "not implemented in fake device");
}

Result<ComputePipelineHandle, std::string>
FakeRendererGPUDevice::createComputePipeline(const ComputePipelineDesc &,
                                             std::string_view) {
  return Result<ComputePipelineHandle, std::string>::makeError(
      "not implemented in fake device");
}

void FakeRendererGPUDevice::destroyRenderPipeline(RenderPipelineHandle) {}

void FakeRendererGPUDevice::destroyComputePipeline(ComputePipelineHandle) {}

void FakeRendererGPUDevice::destroyBuffer(BufferHandle) {}

void FakeRendererGPUDevice::destroyTexture(TextureHandle) {}

void FakeRendererGPUDevice::destroyShaderModule(ShaderHandle) {}

bool FakeRendererGPUDevice::isValid(BufferHandle h) const {
  return nuri::isValid(h);
}

bool FakeRendererGPUDevice::isValid(TextureHandle h) const {
  return nuri::isValid(h);
}

bool FakeRendererGPUDevice::isValid(ShaderHandle h) const {
  return nuri::isValid(h);
}

bool FakeRendererGPUDevice::isValid(RenderPipelineHandle h) const {
  return nuri::isValid(h);
}

bool FakeRendererGPUDevice::isValid(ComputePipelineHandle h) const {
  return nuri::isValid(h);
}

Format FakeRendererGPUDevice::getTextureFormat(TextureHandle) const {
  return Format::RGBA8_UNORM;
}

uint32_t FakeRendererGPUDevice::getTextureBindlessIndex(TextureHandle) const {
  return 0u;
}

uint32_t FakeRendererGPUDevice::getDefaultSamplerBindlessIndex() const {
  return 0u;
}

uint32_t FakeRendererGPUDevice::getCubemapSamplerBindlessIndex() const {
  return 0u;
}

uint64_t FakeRendererGPUDevice::getBufferDeviceAddress(BufferHandle,
                                                       size_t) const {
  return 0ull;
}

bool FakeRendererGPUDevice::resolveGeometry(GeometryAllocationHandle,
                                            GeometryAllocationView &) const {
  return false;
}

Result<bool, std::string>
FakeRendererGPUDevice::beginFrame(uint64_t) {
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeRendererGPUDevice::submitFrame(const RenderFrame &frame) {
  ++submitCount;
  submittedPassCount = frame.passes.size();
  submittedPassLabels.clear();
  submittedPassLabels.reserve(frame.passes.size());
  for (const RenderPass &pass : frame.passes) {
    submittedPassLabels.emplace_back(pass.debugLabel);
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeRendererGPUDevice::submitComputeDispatches(
    std::span<const ComputeDispatchItem>) {
  return Result<bool, std::string>::makeResult(true);
}

Result<GeometryAllocationHandle, std::string>
FakeRendererGPUDevice::allocateGeometry(std::span<const std::byte>, uint32_t,
                                        std::span<const std::byte>, uint32_t,
                                        std::string_view) {
  return Result<GeometryAllocationHandle, std::string>::makeError(
      "not implemented in fake device");
}

void FakeRendererGPUDevice::releaseGeometry(GeometryAllocationHandle) {}

Result<bool, std::string>
FakeRendererGPUDevice::copyBufferRegions(std::span<const BufferCopyRegion>) {
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeRendererGPUDevice::updateBuffer(BufferHandle, std::span<const std::byte>,
                                    size_t) {
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
FakeRendererGPUDevice::readBuffer(BufferHandle, size_t, std::span<std::byte>) {
  return Result<bool, std::string>::makeError("not implemented in fake device");
}

std::byte *FakeRendererGPUDevice::getMappedBufferPtr(BufferHandle) {
  return nullptr;
}

void FakeRendererGPUDevice::flushMappedBuffer(BufferHandle, size_t, size_t) {}

Result<bool, std::string>
FakeRendererGPUDevice::readTexture(TextureHandle,
                                   const TextureReadbackRegion &,
                                   std::span<std::byte>) {
  return Result<bool, std::string>::makeError("not implemented in fake device");
}

void FakeRendererGPUDevice::waitIdle() {}

bool hasPassLabel(const FakeRendererGPUDevice &gpu, std::string_view label) {
  for (const std::string &entry : gpu.submittedPassLabels) {
    if (entry == label) {
      return true;
    }
  }
  return false;
}

} // namespace nuri::test_support
