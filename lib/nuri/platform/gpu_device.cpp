#include "nuri/platform/gpu_device.h"

#include "nuri/platform/window.h"

#include <lvk/LVK.h>

namespace nuri {

namespace {

// Conversion functions from Nuri types to LVK types

lvk::Format toLvkFormat(Format format) {
  switch (format) {
  case Format::RGBA8_UNORM:
    return lvk::Format_RGBA_UN8;
  case Format::RGBA8_SRGB:
    return lvk::Format_RGBA_SRGB8;
  case Format::RGBA8_UINT:
    return lvk::Format_RGBA_UI32;
  case Format::D32_FLOAT:
    return lvk::Format_Z_F32;
  case Format::Count:
    break;
  }
  return lvk::Format_Invalid;
}

Format fromLvkFormat(lvk::Format format) {
  switch (format) {
  case lvk::Format_RGBA_UN8:
    return Format::RGBA8_UNORM;
  case lvk::Format_RGBA_SRGB8:
    return Format::RGBA8_SRGB;
  case lvk::Format_RGBA_UI32:
    return Format::RGBA8_UINT;
  case lvk::Format_Z_F32:
    return Format::D32_FLOAT;
  default:
    break;
  }
  return Format::RGBA8_UNORM;
}

lvk::BufferUsageBits toLvkBufferUsage(BufferUsage usage) {
  switch (usage) {
  case BufferUsage::Vertex:
    return lvk::BufferUsageBits_Vertex;
  case BufferUsage::Index:
    return lvk::BufferUsageBits_Index;
  case BufferUsage::Uniform:
    return lvk::BufferUsageBits_Uniform;
  case BufferUsage::Storage:
    return lvk::BufferUsageBits_Storage;
  case BufferUsage::Count:
    break;
  }
  return lvk::BufferUsageBits_Vertex;
}

lvk::StorageType toLvkStorageType(Storage storage) {
  switch (storage) {
  case Storage::Device:
    return lvk::StorageType_Device;
  case Storage::HostVisible:
    return lvk::StorageType_HostVisible;
  case Storage::Memoryless:
    return lvk::StorageType_Memoryless;
  case Storage::Count:
    break;
  }
  return lvk::StorageType_Device;
}

lvk::TextureType toLvkTextureType(TextureType type) {
  switch (type) {
  case TextureType::Texture2D:
    return lvk::TextureType_2D;
  case TextureType::Texture3D:
    return lvk::TextureType_3D;
  case TextureType::TextureCube:
    return lvk::TextureType_Cube;
  case TextureType::Count:
    break;
  }
  return lvk::TextureType_2D;
}

lvk::TextureUsageBits toLvkTextureUsage(TextureUsage usage) {
  switch (usage) {
  case TextureUsage::Sampled:
    return lvk::TextureUsageBits_Sampled;
  case TextureUsage::Storage:
    return lvk::TextureUsageBits_Storage;
  case TextureUsage::Attachment:
    return lvk::TextureUsageBits_Attachment;
  case TextureUsage::InputAttachment:
    return lvk::TextureUsageBits_InputAttachment;
  case TextureUsage::Count:
    break;
  }
  return lvk::TextureUsageBits_Sampled;
}

lvk::IndexFormat toLvkIndexFormat(IndexFormat format) {
  switch (format) {
  case IndexFormat::U16:
    return lvk::IndexFormat_UI16;
  case IndexFormat::U32:
    return lvk::IndexFormat_UI32;
  case IndexFormat::Count:
    break;
  }
  return lvk::IndexFormat_UI32;
}

lvk::CompareOp toLvkCompareOp(CompareOp op) {
  switch (op) {
  case CompareOp::Less:
    return lvk::CompareOp_Less;
  case CompareOp::LessEqual:
    return lvk::CompareOp_LessEqual;
  case CompareOp::Greater:
    return lvk::CompareOp_Greater;
  case CompareOp::GreaterEqual:
    return lvk::CompareOp_GreaterEqual;
  case CompareOp::Equal:
    return lvk::CompareOp_Equal;
  case CompareOp::NotEqual:
    return lvk::CompareOp_NotEqual;
  case CompareOp::Always:
    return lvk::CompareOp_AlwaysPass;
  case CompareOp::Never:
    return lvk::CompareOp_Never;
  case CompareOp::Count:
    break;
  }
  return lvk::CompareOp_Less;
}

lvk::CullMode toLvkCullMode(CullMode mode) {
  switch (mode) {
  case CullMode::None:
    return lvk::CullMode_None;
  case CullMode::Front:
    return lvk::CullMode_Front;
  case CullMode::Back:
    return lvk::CullMode_Back;
  case CullMode::Count:
    break;
  }
  return lvk::CullMode_Back;
}

lvk::PolygonMode toLvkPolygonMode(PolygonMode mode) {
  switch (mode) {
  case PolygonMode::Fill:
    return lvk::PolygonMode_Fill;
  case PolygonMode::Line:
    return lvk::PolygonMode_Line;
  case PolygonMode::Count:
    break;
  }
  return lvk::PolygonMode_Fill;
}

lvk::LoadOp toLvkLoadOp(LoadOp op) {
  switch (op) {
  case LoadOp::Load:
    return lvk::LoadOp_Load;
  case LoadOp::Clear:
    return lvk::LoadOp_Clear;
  case LoadOp::DontCare:
    return lvk::LoadOp_DontCare;
  }
  return lvk::LoadOp_DontCare;
}

lvk::StoreOp toLvkStoreOp(StoreOp op) {
  switch (op) {
  case StoreOp::Store:
    return lvk::StoreOp_Store;
  case StoreOp::DontCare:
    return lvk::StoreOp_DontCare;
  }
  return lvk::StoreOp_DontCare;
}

lvk::VertexFormat toLvkVertexFormat(VertexFormat format) {
  switch (format) {
  case VertexFormat::Float1:
    return lvk::VertexFormat::Float1;
  case VertexFormat::Float2:
    return lvk::VertexFormat::Float2;
  case VertexFormat::Float3:
    return lvk::VertexFormat::Float3;
  case VertexFormat::Float4:
    return lvk::VertexFormat::Float4;
  case VertexFormat::Int1:
    return lvk::VertexFormat::Int1;
  case VertexFormat::Int2:
    return lvk::VertexFormat::Int2;
  case VertexFormat::Int3:
    return lvk::VertexFormat::Int3;
  case VertexFormat::Int4:
    return lvk::VertexFormat::Int4;
  case VertexFormat::UInt1:
    return lvk::VertexFormat::UInt1;
  case VertexFormat::UInt2:
    return lvk::VertexFormat::UInt2;
  case VertexFormat::UInt3:
    return lvk::VertexFormat::UInt3;
  case VertexFormat::UInt4:
    return lvk::VertexFormat::UInt4;
  case VertexFormat::Byte4_Norm:
    return lvk::VertexFormat::Byte4Norm;
  case VertexFormat::UByte4_Norm:
    return lvk::VertexFormat::UByte4Norm;
  case VertexFormat::Short2:
    return lvk::VertexFormat::Short2;
  case VertexFormat::Short2_Norm:
    return lvk::VertexFormat::Short2Norm;
  case VertexFormat::Count:
    break;
  }
  return lvk::VertexFormat::Float3;
}

lvk::ShaderStage toLvkShaderStage(ShaderStage stage) {
  switch (stage) {
  case ShaderStage::Vertex:
    return lvk::ShaderStage::Stage_Vert;
  case ShaderStage::TessControl:
    return lvk::ShaderStage::Stage_Tesc;
  case ShaderStage::TessEval:
    return lvk::ShaderStage::Stage_Tese;
  case ShaderStage::Geometry:
    return lvk::ShaderStage::Stage_Geom;
  case ShaderStage::Fragment:
    return lvk::ShaderStage::Stage_Frag;
  case ShaderStage::Compute:
    return lvk::ShaderStage::Stage_Comp;
  case ShaderStage::Task:
    return lvk::ShaderStage::Stage_Task;
  case ShaderStage::Mesh:
    return lvk::ShaderStage::Stage_Mesh;
  case ShaderStage::RayGen:
    return lvk::ShaderStage::Stage_RayGen;
  case ShaderStage::AnyHit:
    return lvk::ShaderStage::Stage_AnyHit;
  case ShaderStage::ClosestHit:
    return lvk::ShaderStage::Stage_ClosestHit;
  case ShaderStage::Miss:
    return lvk::ShaderStage::Stage_Miss;
  case ShaderStage::Intersection:
    return lvk::ShaderStage::Stage_Intersection;
  case ShaderStage::Callable:
    return lvk::ShaderStage::Stage_Callable;
  case ShaderStage::Count:
    break;
  }
  return lvk::ShaderStage::Stage_Vert;
}

} // namespace

template <typename LvkHandle> struct ResourceSlot {
  lvk::Holder<LvkHandle> resource;
  uint32_t generation = 0;
  bool live = false;
  std::string debugName;
  Format format = Format::RGBA8_UNORM; // For textures
};

template <typename NuriHandle, typename LvkHandle> class ResourceTable {
public:
  ResourceTable() = default;

  NuriHandle allocate(lvk::Holder<LvkHandle> &&resource, std::string debugName,
                      Format format = Format::RGBA8_UNORM) {
    uint32_t index;
    if (!freeList_.empty()) {
      index = freeList_.back();
      freeList_.pop_back();
    } else {
      index = static_cast<uint32_t>(slots_.size());
      slots_.emplace_back();
    }

    auto &slot = slots_[index];
    slot.resource = std::move(resource);
    slot.generation++;
    slot.live = true;
    slot.debugName = std::move(debugName);
    slot.format = format;

    return NuriHandle{index, slot.generation};
  }

  void deallocate(NuriHandle h) {
    if (!isValid(h))
      return;
    auto &slot = slots_[h.index];
    slot.resource.reset();
    slot.live = false;
    slot.debugName.clear();
    freeList_.push_back(h.index);
  }

  bool isValid(NuriHandle h) const {
    return h.index < slots_.size() &&
           slots_[h.index].generation == h.generation && slots_[h.index].live;
  }

  LvkHandle getLvkHandle(NuriHandle h) const {
    if (!isValid(h))
      return {};
    return slots_[h.index].resource;
  }

  Format getFormat(NuriHandle h) const {
    if (!isValid(h))
      return Format::RGBA8_UNORM;
    return slots_[h.index].format;
  }

private:
  std::vector<ResourceSlot<LvkHandle>> slots_;
  std::vector<uint32_t> freeList_;
};

struct GPUDevice::Impl {
  Window *window = nullptr;
  std::unique_ptr<lvk::IContext> context;

  ResourceTable<BufferHandle, lvk::BufferHandle> buffers;
  ResourceTable<TextureHandle, lvk::TextureHandle> textures;
  ResourceTable<ShaderHandle, lvk::ShaderModuleHandle> shaders;
  ResourceTable<RenderPipelineHandle, lvk::RenderPipelineHandle>
      renderPipelines;
  ResourceTable<ComputePipelineHandle, lvk::ComputePipelineHandle>
      computePipelines;
};

GPUDevice::GPUDevice() : impl_(std::make_unique<Impl>()) {}

GPUDevice::~GPUDevice() {
  if (!impl_) {
    return;
  }

  if (impl_->context) {
    impl_->context->wait(lvk::SubmitHandle{});
  }

  impl_.reset();
  minilog::deinitialize();
}

std::unique_ptr<GPUDevice> GPUDevice::create(Window &window) {
  auto device = std::unique_ptr<GPUDevice>(new GPUDevice());

  minilog::initialize(nullptr, {.threadNames = false});

  device->impl_->window = &window;

  int32_t width = 0;
  int32_t height = 0;
  window.getFramebufferSize(width, height);
  if (!width || !height) {
    width = 1;
    height = 1;
  }

  lvk::ContextConfig config{};
#if defined(NURI_DEBUG)
  config.enableValidation = true;
#else
  config.enableValidation = false;
#endif

  device->impl_->context = lvk::createVulkanContextWithSwapchain(
      static_cast<lvk::LVKwindow *>(window.nativeHandle()),
      static_cast<uint32_t>(width), static_cast<uint32_t>(height), config);

  if (!device->impl_->context) {
    return nullptr;
  }

  return device;
}

void GPUDevice::pollEvents() {
  if (impl_->window) {
    impl_->window->pollEvents();
  }
}

bool GPUDevice::shouldClose() const {
  return impl_->window ? impl_->window->shouldClose() : true;
}

void GPUDevice::getFramebufferSize(int32_t &outWidth,
                                   int32_t &outHeight) const {
  if (!impl_->window) {
    outWidth = 0;
    outHeight = 0;
    return;
  }
  impl_->window->getFramebufferSize(outWidth, outHeight);
}

void GPUDevice::resizeSwapchain(int32_t width, int32_t height) {
  impl_->context->recreateSwapchain(width, height);
}

Format GPUDevice::getSwapchainFormat() const {
  return fromLvkFormat(impl_->context->getSwapchainFormat());
}

double GPUDevice::getTime() const {
  return impl_->window ? impl_->window->getTime() : 0.0;
}

Result<BufferHandle, std::string>
GPUDevice::createBuffer(const BufferDesc &desc, std::string_view debugName) {
  const size_t resolvedSize = desc.size != 0 ? desc.size : desc.data.size();
  if (resolvedSize == 0) {
    return Result<BufferHandle, std::string>::makeError("Buffer size is zero");
  }
  if (!desc.data.empty() && desc.size != 0 && desc.data.size() != desc.size) {
    return Result<BufferHandle, std::string>::makeError(
        "Buffer data size must match buffer size");
  }

  std::string debugNameStorage(debugName);
  const char *debugNameCStr =
      debugNameStorage.empty() ? "" : debugNameStorage.c_str();

  lvk::BufferDesc bufferDesc{
      .usage = static_cast<uint8_t>(toLvkBufferUsage(desc.usage)),
      .storage = toLvkStorageType(desc.storage),
      .size = resolvedSize,
      .data = desc.data.empty() ? nullptr : desc.data.data(),
      .debugName = debugNameCStr,
  };

  lvk::Result res;
  lvk::Holder<lvk::BufferHandle> handle =
      impl_->context->createBuffer(bufferDesc, debugNameCStr, &res);

  if (!res.isOk()) {
    return Result<BufferHandle, std::string>::makeError(
        std::string(res.message));
  }
  if (!handle.valid()) {
    return Result<BufferHandle, std::string>::makeError(
        "Failed to create buffer");
  }

  BufferHandle nuriHandle =
      impl_->buffers.allocate(std::move(handle), std::move(debugNameStorage));
  return Result<BufferHandle, std::string>::makeResult(nuriHandle);
}

Result<TextureHandle, std::string>
GPUDevice::createTexture(const TextureDesc &desc, std::string_view debugName) {
  if (desc.dimensions.width == 0 || desc.dimensions.height == 0) {
    return Result<TextureHandle, std::string>::makeError(
        "Texture dimensions cannot be zero");
  }

  std::string debugNameStorage(debugName);
  const char *debugNameCStr =
      debugNameStorage.empty() ? "" : debugNameStorage.c_str();

  lvk::TextureDesc textureDesc{
      .type = toLvkTextureType(desc.type),
      .format = toLvkFormat(desc.format),
      .dimensions = {desc.dimensions.width, desc.dimensions.height,
                     desc.dimensions.depth},
      .numLayers = desc.numLayers,
      .numSamples = desc.numSamples,
      .usage = static_cast<uint8_t>(toLvkTextureUsage(desc.usage)),
      .numMipLevels = desc.numMipLevels,
      .storage = toLvkStorageType(desc.storage),
      .debugName = debugNameCStr,
  };

  lvk::Result res;
  lvk::Holder<lvk::TextureHandle> handle =
      impl_->context->createTexture(textureDesc, debugNameCStr, &res);

  if (!res.isOk()) {
    return Result<TextureHandle, std::string>::makeError(
        std::string(res.message));
  }
  if (!handle.valid()) {
    return Result<TextureHandle, std::string>::makeError(
        "Failed to create texture");
  }

  TextureHandle nuriHandle = impl_->textures.allocate(
      std::move(handle), std::move(debugNameStorage), desc.format);
  return Result<TextureHandle, std::string>::makeResult(nuriHandle);
}

Result<ShaderHandle, std::string>
GPUDevice::createShaderModule(const ShaderDesc &desc) {
  if (desc.source.empty()) {
    return Result<ShaderHandle, std::string>::makeError(
        "Shader source is empty");
  }

  std::string moduleNameStorage(desc.moduleName);
  std::string sourceStorage(desc.source);

  lvk::ShaderModuleDesc shaderDesc(
      sourceStorage.c_str(), toLvkShaderStage(desc.stage),
      moduleNameStorage.empty() ? "" : moduleNameStorage.c_str());

  lvk::Result res;
  lvk::Holder<lvk::ShaderModuleHandle> handle =
      impl_->context->createShaderModule(shaderDesc, &res);

  if (!res.isOk()) {
    return Result<ShaderHandle, std::string>::makeError(
        std::string(res.message));
  }
  if (!handle.valid()) {
    return Result<ShaderHandle, std::string>::makeError(
        "Failed to create shader module");
  }

  ShaderHandle nuriHandle =
      impl_->shaders.allocate(std::move(handle), std::move(moduleNameStorage));
  return Result<ShaderHandle, std::string>::makeResult(nuriHandle);
}

Result<RenderPipelineHandle, std::string>
GPUDevice::createRenderPipeline(const RenderPipelineDesc &desc,
                                std::string_view debugName) {
  if (!isValid(desc.vertexShader)) {
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Invalid vertex shader handle");
  }
  if (!isValid(desc.fragmentShader)) {
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Invalid fragment shader handle");
  }

  std::string debugNameStorage(debugName);

  lvk::VertexInput vertexInput{};
  const size_t numAttribs = std::min(
      desc.vertexInput.attributes.size(),
      static_cast<size_t>(lvk::VertexInput::LVK_VERTEX_ATTRIBUTES_MAX));
  for (size_t i = 0; i < numAttribs; ++i) {
    const auto &attr = desc.vertexInput.attributes[i];
    vertexInput.attributes[i] = {
        .location = attr.location,
        .binding = attr.binding,
        .format = toLvkVertexFormat(attr.format),
        .offset = attr.offset,
    };
  }

  const size_t numBindings =
      std::min(desc.vertexInput.bindings.size(),
               static_cast<size_t>(lvk::VertexInput::LVK_VERTEX_BUFFER_MAX));
  for (size_t i = 0; i < numBindings; ++i) {
    const auto &binding = desc.vertexInput.bindings[i];
    vertexInput.inputBindings[i] = {.stride = binding.stride};
  }

  lvk::SpecializationConstantDesc specInfo{};
  if (!desc.specInfo.entries.empty()) {
    const size_t numEntries = std::min(
        desc.specInfo.entries.size(),
        static_cast<size_t>(
            lvk::SpecializationConstantDesc::LVK_SPECIALIZATION_CONSTANTS_MAX));
    for (size_t i = 0; i < numEntries; ++i) {
      const auto &entry = desc.specInfo.entries[i];
      if (entry.offset + entry.size > desc.specInfo.dataSize) {
        return Result<RenderPipelineHandle, std::string>::makeError(
            "Specialization entry offset+size exceeds specInfo.dataSize");
      }
      specInfo.entries[i] = {
          .constantId = entry.constantId,
          .offset = entry.offset,
          .size = entry.size,
      };
    }
    specInfo.data = desc.specInfo.data;
    specInfo.dataSize = desc.specInfo.dataSize;
  }

  lvk::RenderPipelineDesc pipelineDesc{
      .vertexInput = vertexInput,
      .smVert = impl_->shaders.getLvkHandle(desc.vertexShader),
      .smFrag = impl_->shaders.getLvkHandle(desc.fragmentShader),
      .specInfo = specInfo,
      .color = {{.format = toLvkFormat(desc.colorFormats[0])}},
      .depthFormat = toLvkFormat(desc.depthFormat),
      .cullMode = toLvkCullMode(desc.cullMode),
      .polygonMode = toLvkPolygonMode(desc.polygonMode),
      .debugName = debugNameStorage.empty() ? "" : debugNameStorage.c_str(),
  };

  lvk::Result res;
  lvk::Holder<lvk::RenderPipelineHandle> handle =
      impl_->context->createRenderPipeline(pipelineDesc, &res);

  if (!res.isOk()) {
    return Result<RenderPipelineHandle, std::string>::makeError(
        std::string(res.message));
  }
  if (!handle.valid()) {
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Failed to create render pipeline");
  }

  RenderPipelineHandle nuriHandle = impl_->renderPipelines.allocate(
      std::move(handle), std::move(debugNameStorage));
  return Result<RenderPipelineHandle, std::string>::makeResult(nuriHandle);
}

Result<ComputePipelineHandle, std::string>
GPUDevice::createComputePipeline(const ComputePipelineDesc &desc,
                                 std::string_view debugName) {
  if (!isValid(desc.computeShader)) {
    return Result<ComputePipelineHandle, std::string>::makeError(
        "Invalid compute shader handle");
  }

  std::string debugNameStorage(debugName);

  // Build specialization info
  lvk::SpecializationConstantDesc specInfo{};
  if (!desc.specInfo.entries.empty()) {
    const size_t numEntries = std::min(
        desc.specInfo.entries.size(),
        static_cast<size_t>(
            lvk::SpecializationConstantDesc::LVK_SPECIALIZATION_CONSTANTS_MAX));
    for (size_t i = 0; i < numEntries; ++i) {
      const auto &entry = desc.specInfo.entries[i];
      if (entry.offset + entry.size > desc.specInfo.dataSize) {
        return Result<ComputePipelineHandle, std::string>::makeError(
            "Specialization entry offset+size exceeds specInfo.dataSize");
      }
      specInfo.entries[i] = {
          .constantId = entry.constantId,
          .offset = entry.offset,
          .size = entry.size,
      };
    }
    specInfo.data = desc.specInfo.data;
    specInfo.dataSize = desc.specInfo.dataSize;
  }

  lvk::ComputePipelineDesc pipelineDesc{
      .smComp = impl_->shaders.getLvkHandle(desc.computeShader),
      .specInfo = specInfo,
      .debugName = debugNameStorage.empty() ? "" : debugNameStorage.c_str(),
  };

  lvk::Result res;
  lvk::Holder<lvk::ComputePipelineHandle> handle =
      impl_->context->createComputePipeline(pipelineDesc, &res);

  if (!res.isOk()) {
    return Result<ComputePipelineHandle, std::string>::makeError(
        std::string(res.message));
  }
  if (!handle.valid()) {
    return Result<ComputePipelineHandle, std::string>::makeError(
        "Failed to create compute pipeline");
  }

  ComputePipelineHandle nuriHandle = impl_->computePipelines.allocate(
      std::move(handle), std::move(debugNameStorage));
  return Result<ComputePipelineHandle, std::string>::makeResult(nuriHandle);
}

bool GPUDevice::isValid(BufferHandle h) const {
  return impl_->buffers.isValid(h);
}

bool GPUDevice::isValid(TextureHandle h) const {
  return impl_->textures.isValid(h);
}

bool GPUDevice::isValid(ShaderHandle h) const {
  return impl_->shaders.isValid(h);
}

bool GPUDevice::isValid(RenderPipelineHandle h) const {
  return impl_->renderPipelines.isValid(h);
}

bool GPUDevice::isValid(ComputePipelineHandle h) const {
  return impl_->computePipelines.isValid(h);
}

Format GPUDevice::getTextureFormat(TextureHandle h) const {
  return impl_->textures.getFormat(h);
}

Result<bool, std::string> GPUDevice::submitFrame(const RenderFrame &frame) {
  if (frame.passes.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  lvk::ICommandBuffer &commandBuffer = impl_->context->acquireCommandBuffer();

  for (const RenderPass &pass : frame.passes) {
    if (!pass.debugLabel.empty()) {
      commandBuffer.cmdPushDebugGroupLabel(pass.debugLabel.data(),
                                           pass.debugColor);
    }

    lvk::RenderPass renderPass{};
    renderPass.color[0] = {
        .loadOp = toLvkLoadOp(pass.color.loadOp),
        .storeOp = toLvkStoreOp(pass.color.storeOp),
        .clearColor = {pass.color.clearColor.r, pass.color.clearColor.g,
                       pass.color.clearColor.b, pass.color.clearColor.a},
    };

    lvk::Framebuffer framebuffer{};
    framebuffer.color[0] = {.texture =
                                impl_->context->getCurrentSwapchainTexture()};

    if (nuri::isValid(pass.depthTexture)) {
      renderPass.depth = {
          .loadOp = toLvkLoadOp(pass.depth.loadOp),
          .storeOp = toLvkStoreOp(pass.depth.storeOp),
          .clearDepth = pass.depth.clearDepth,
          .clearStencil = pass.depth.clearStencil,
      };
      framebuffer.depthStencil = {
          .texture = impl_->textures.getLvkHandle(pass.depthTexture)};
    } else {
      renderPass.depth.loadOp = lvk::LoadOp_Invalid;
    }

    commandBuffer.cmdBeginRendering(renderPass, framebuffer);

    for (const DrawItem &draw : pass.draws) {
      if (!impl_->renderPipelines.isValid(draw.pipeline)) {
        commandBuffer.cmdEndRendering();
        if (!pass.debugLabel.empty()) {
          commandBuffer.cmdPopDebugGroupLabel();
        }
        return Result<bool, std::string>::makeError(
            "Invalid render pipeline handle");
      }

      if (!draw.debugLabel.empty()) {
        commandBuffer.cmdPushDebugGroupLabel(draw.debugLabel.data(),
                                             draw.debugColor);
      }

      commandBuffer.cmdBindRenderPipeline(
          impl_->renderPipelines.getLvkHandle(draw.pipeline));

      if (nuri::isValid(draw.vertexBuffer)) {
        commandBuffer.cmdBindVertexBuffer(
            0, impl_->buffers.getLvkHandle(draw.vertexBuffer),
            draw.vertexBufferOffset);
      }

      if (draw.indexCount > 0) {
        if (!impl_->buffers.isValid(draw.indexBuffer)) {
          commandBuffer.cmdEndRendering();
          if (!pass.debugLabel.empty()) {
            commandBuffer.cmdPopDebugGroupLabel();
          }
          return Result<bool, std::string>::makeError(
              "Index buffer is invalid");
        }
        commandBuffer.cmdBindIndexBuffer(
            impl_->buffers.getLvkHandle(draw.indexBuffer),
            toLvkIndexFormat(draw.indexFormat), draw.indexBufferOffset);
      }

      if (draw.useDepthState) {
        lvk::DepthState depthState{
            .compareOp = toLvkCompareOp(draw.depthState.compareOp),
            .isDepthWriteEnabled = draw.depthState.isDepthWriteEnabled,
        };
        commandBuffer.cmdBindDepthState(depthState);
      }

      commandBuffer.cmdSetDepthBiasEnable(draw.depthBiasEnable);
      if (draw.depthBiasEnable) {
        commandBuffer.cmdSetDepthBias(draw.depthBiasConstant,
                                      draw.depthBiasSlope, draw.depthBiasClamp);
      }

      if (!draw.pushConstants.empty()) {
        commandBuffer.cmdPushConstants(
            static_cast<const void *>(draw.pushConstants.data()),
            draw.pushConstants.size(), 0);
      }

      if (draw.indexCount > 0) {
        commandBuffer.cmdDrawIndexed(draw.indexCount, draw.instanceCount,
                                     draw.firstIndex, draw.vertexOffset,
                                     draw.firstInstance);
      } else {
        commandBuffer.cmdDraw(draw.vertexCount, draw.instanceCount,
                              draw.firstVertex, draw.firstInstance);
      }

      if (!draw.debugLabel.empty()) {
        commandBuffer.cmdPopDebugGroupLabel();
      }
    }

    commandBuffer.cmdEndRendering();

    if (!pass.debugLabel.empty()) {
      commandBuffer.cmdPopDebugGroupLabel();
    }
  }

  impl_->context->submit(commandBuffer,
                         impl_->context->getCurrentSwapchainTexture());
  return Result<bool, std::string>::makeResult(true);
}

void GPUDevice::waitIdle() {
  if (impl_->context) {
    // Empty SubmitHandle results in vkDeviceWaitIdle
    impl_->context->wait(lvk::SubmitHandle{});
  }
}

} // namespace nuri
