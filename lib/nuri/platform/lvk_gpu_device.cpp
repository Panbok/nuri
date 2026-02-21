#include "nuri/pch.h"

#include "nuri/platform/lvk_gpu_device.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/core/window.h"
#include "nuri/resources/gpu/geometry_pool.h"

#include <lvk/LVK.h>
#include <vulkan/VulkanUtils.h>

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
  case Format::RGBA16_FLOAT:
    return lvk::Format_RGBA_F16;
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
  case lvk::Format_RGBA_F16:
    return Format::RGBA16_FLOAT;
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

lvk::Topology toLvkTopology(Topology topology) {
  switch (topology) {
  case Topology::Point:
    return lvk::Topology_Point;
  case Topology::Line:
    return lvk::Topology_Line;
  case Topology::LineStrip:
    return lvk::Topology_LineStrip;
  case Topology::Triangle:
    return lvk::Topology_Triangle;
  case Topology::TriangleStrip:
    return lvk::Topology_TriangleStrip;
  case Topology::Patch:
    return lvk::Topology_Patch;
  case Topology::Count:
    break;
  }
  return lvk::Topology_Triangle;
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

  struct ReservedSlot {
    NuriHandle handle{};
    const char *debugNameCStr = "";
  };

  ReservedSlot reserve(std::string debugName,
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
    slot.resource.reset();
    slot.generation++;
    slot.live = true;
    slot.debugName = std::move(debugName);
    slot.format = format;

    const char *cstr = slot.debugName.empty() ? "" : slot.debugName.c_str();
    return ReservedSlot{NuriHandle{index, slot.generation}, cstr};
  }

  bool setResource(NuriHandle h, lvk::Holder<LvkHandle> &&resource) {
    if (!isValid(h))
      return false;
    slots_[h.index].resource = std::move(resource);
    return true;
  }

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

  bool replace(NuriHandle h, lvk::Holder<LvkHandle> &&resource,
               std::string_view debugName,
               Format format = Format::RGBA8_UNORM) {
    if (!isValid(h))
      return false;
    auto &slot = slots_[h.index];
    slot.resource = std::move(resource);
    slot.debugName = std::string(debugName);
    slot.format = format;
    return true;
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
  // Stable element addresses (LVK stores debugName pointers for pipelines).
  std::deque<ResourceSlot<LvkHandle>> slots_;
  std::vector<uint32_t> freeList_;
};

struct FramebufferTexture {
  TextureHandle handle{};
  TextureDesc desc{};
  std::string debugName;
};

struct LvkGPUDevice::Impl {
  Window *window = nullptr;
  std::unique_ptr<lvk::IContext> context;

  ResourceTable<BufferHandle, lvk::BufferHandle> buffers;
  ResourceTable<TextureHandle, lvk::TextureHandle> textures;
  ResourceTable<ShaderHandle, lvk::ShaderModuleHandle> shaders;
  ResourceTable<RenderPipelineHandle, lvk::RenderPipelineHandle>
      renderPipelines;
  ResourceTable<ComputePipelineHandle, lvk::ComputePipelineHandle>
      computePipelines;
  std::vector<FramebufferTexture> framebufferTextures;
  std::unique_ptr<GeometryPool> geometryPool;
};

LvkGPUDevice::LvkGPUDevice() : impl_(std::make_unique<Impl>()) {}

LvkGPUDevice::~LvkGPUDevice() {
  if (!impl_) {
    return;
  }

  if (impl_->context) {
    impl_->context->wait(lvk::SubmitHandle{});
  }

  impl_.reset();
}

std::unique_ptr<LvkGPUDevice>
LvkGPUDevice::create(Window &window, const GPUDeviceCreateDesc &desc) {
  auto device = std::unique_ptr<LvkGPUDevice>(new LvkGPUDevice());

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
    NURI_LOG_WARNING("LvkGPUDevice::create: Failed to create Vulkan context "
                     "with swapchain (%d x %d)",
                     width, height);
    return nullptr;
  }

  device->impl_->geometryPool =
      std::make_unique<GeometryPool>(*device, desc.geometryPool);

  return device;
}

std::unique_ptr<GPUDevice> GPUDevice::create(Window &window,
                                             const GPUDeviceCreateDesc &desc) {
  NURI_LOG_DEBUG("GPUDevice::create: Creating Vulkan GPU device");
  return LvkGPUDevice::create(window, desc);
}

bool LvkGPUDevice::shouldClose() const {
  return impl_->window ? impl_->window->shouldClose() : true;
}

void LvkGPUDevice::getFramebufferSize(int32_t &outWidth,
                                      int32_t &outHeight) const {
  if (!impl_->window) {
    outWidth = 0;
    outHeight = 0;
    return;
  }
  impl_->window->getFramebufferSize(outWidth, outHeight);
}

void LvkGPUDevice::resizeSwapchain(int32_t width, int32_t height) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  impl_->context->recreateSwapchain(width, height);
  impl_->context->wait(lvk::SubmitHandle{});
  if (!width || !height) {
    return;
  }

  const auto replaceTextureResource =
      [this](TextureHandle handle, const TextureDesc &desc,
             std::string_view debugName) -> Result<bool, std::string> {
    if (desc.dimensions.width == 0 || desc.dimensions.height == 0) {
      return Result<bool, std::string>::makeError(
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
    lvk::Holder<lvk::TextureHandle> newHandle =
        impl_->context->createTexture(textureDesc, debugNameCStr, &res);

    if (!res.isOk()) {
      return Result<bool, std::string>::makeError(std::string(res.message));
    }
    if (!newHandle.valid()) {
      return Result<bool, std::string>::makeError("Failed to create texture");
    }
    if (!impl_->textures.replace(handle, std::move(newHandle), debugNameStorage,
                                 desc.format)) {
      return Result<bool, std::string>::makeError("Invalid texture handle");
    }
    return Result<bool, std::string>::makeResult(true);
  };

  impl_->framebufferTextures.erase(
      std::remove_if(impl_->framebufferTextures.begin(),
                     impl_->framebufferTextures.end(),
                     [this](const FramebufferTexture &entry) {
                       return !impl_->textures.isValid(entry.handle);
                     }),
      impl_->framebufferTextures.end());

  const auto framebufferWidth = static_cast<uint32_t>(width);
  const auto framebufferHeight = static_cast<uint32_t>(height);
  for (const auto &entry : impl_->framebufferTextures) {
    TextureDesc resizedDesc = entry.desc;
    resizedDesc.dimensions.width = framebufferWidth;
    resizedDesc.dimensions.height = framebufferHeight;
    auto result =
        replaceTextureResource(entry.handle, resizedDesc, entry.debugName);
    if (result.hasError()) {
      NURI_LOG_WARNING("LvkGPUDevice::resizeSwapchain: Failed to resize "
                       "framebuffer texture '%s': %s",
                       entry.debugName.c_str(), result.error().c_str());
    }
  }
}

Format LvkGPUDevice::getSwapchainFormat() const {
  return fromLvkFormat(impl_->context->getSwapchainFormat());
}

uint32_t LvkGPUDevice::getSwapchainImageIndex() const {
  return impl_->context ? impl_->context->getSwapchainCurrentImageIndex() : 0u;
}

uint32_t LvkGPUDevice::getSwapchainImageCount() const {
  return impl_->context ? impl_->context->getNumSwapchainImages() : 1u;
}

double LvkGPUDevice::getTime() const {
  return impl_->window ? impl_->window->getTime() : 0.0;
}

Result<bool, std::string> LvkGPUDevice::beginFrame(uint64_t frameIndex) {
  if (!impl_->geometryPool) {
    return Result<bool, std::string>::makeResult(true);
  }
  return impl_->geometryPool->beginFrame(frameIndex);
}

Result<BufferHandle, std::string>
LvkGPUDevice::createBuffer(const BufferDesc &desc, std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const size_t resolvedSize = desc.size != 0 ? desc.size : desc.data.size();
  if (resolvedSize == 0) {
    NURI_LOG_WARNING("LvkGPUDevice::createBuffer: Buffer size is zero");
    return Result<BufferHandle, std::string>::makeError("Buffer size is zero");
  }
  if (!desc.data.empty() && desc.size != 0 && desc.data.size() != desc.size) {
    NURI_LOG_WARNING("LvkGPUDevice::createBuffer: Buffer data size must match "
                     "buffer size");
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
    if (debugNameCStr && debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING(
          "LvkGPUDevice::createBuffer: Failed to create buffer '%s': %s",
          debugNameCStr, res.message);
    } else {
      NURI_LOG_WARNING(
          "LvkGPUDevice::createBuffer: Failed to create buffer: %s",
          res.message);
    }
    return Result<BufferHandle, std::string>::makeError(
        std::string(res.message));
  }
  if (!handle.valid()) {
    if (debugNameCStr && debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING(
          "LvkGPUDevice::createBuffer: Failed to create buffer '%s'",
          debugNameCStr);
    } else {
      NURI_LOG_WARNING("LvkGPUDevice::createBuffer: Failed to create buffer");
    }
    return Result<BufferHandle, std::string>::makeError(
        "Failed to create buffer");
  }

  BufferHandle nuriHandle =
      impl_->buffers.allocate(std::move(handle), std::move(debugNameStorage));
  return Result<BufferHandle, std::string>::makeResult(nuriHandle);
}

Result<TextureHandle, std::string>
LvkGPUDevice::createTexture(const TextureDesc &desc,
                            std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (desc.dimensions.width == 0 || desc.dimensions.height == 0) {
    NURI_LOG_WARNING(
        "LvkGPUDevice::createTexture: Texture dimensions cannot be zero");
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
      .data = desc.data.empty() ? nullptr
                                : static_cast<const void *>(desc.data.data()),
      .dataNumMipLevels = desc.dataNumMipLevels,
      .generateMipmaps = desc.generateMipmaps && !desc.data.empty(),
      .debugName = debugNameCStr,
  };

  lvk::Result res;
  lvk::Holder<lvk::TextureHandle> handle =
      impl_->context->createTexture(textureDesc, debugNameCStr, &res);

  if (!res.isOk()) {
    if (debugNameCStr && debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING(
          "LvkGPUDevice::createTexture: Failed to create texture '%s': %s",
          debugNameCStr, res.message);
    } else {
      NURI_LOG_WARNING(
          "LvkGPUDevice::createTexture: Failed to create texture: %s",
          res.message);
    }
    return Result<TextureHandle, std::string>::makeError(
        std::string(res.message));
  }
  if (!handle.valid()) {
    if (debugNameCStr && debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING(
          "LvkGPUDevice::createTexture: Failed to create texture '%s'",
          debugNameCStr);
    } else {
      NURI_LOG_WARNING("LvkGPUDevice::createTexture: Failed to create texture");
    }
    return Result<TextureHandle, std::string>::makeError(
        "Failed to create texture");
  }

  TextureHandle nuriHandle = impl_->textures.allocate(
      std::move(handle), std::move(debugNameStorage), desc.format);
  return Result<TextureHandle, std::string>::makeResult(nuriHandle);
}

Result<TextureHandle, std::string>
LvkGPUDevice::createFramebufferTexture(const TextureDesc &desc,
                                       std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!impl_->window) {
    NURI_LOG_WARNING("LvkGPUDevice::createFramebufferTexture: No window "
                     "available to get framebuffer size");
    return Result<TextureHandle, std::string>::makeError(
        "No window available to get framebuffer size");
  }

  int32_t width = 0;
  int32_t height = 0;
  impl_->window->getFramebufferSize(width, height);
  if (!width || !height) {
    NURI_LOG_WARNING("LvkGPUDevice::createFramebufferTexture: Failed to get "
                     "framebuffer size");
    return Result<TextureHandle, std::string>::makeError(
        "Failed to get framebuffer size");
  }

  TextureDesc resizedDesc = desc;
  resizedDesc.dimensions.width = static_cast<uint32_t>(width);
  resizedDesc.dimensions.height = static_cast<uint32_t>(height);

  auto result = createTexture(resizedDesc, debugName);
  if (result.hasError()) {
    return result;
  }

  FramebufferTexture entry{
      .handle = result.value(),
      .desc = desc,
      .debugName = std::string(debugName),
  };
  impl_->framebufferTextures.push_back(std::move(entry));

  return result;
}

Result<TextureHandle, std::string> LvkGPUDevice::createDepthBuffer() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  TextureDesc desc{
      .type = TextureType::Texture2D,
      .format = Format::D32_FLOAT,
      .dimensions = {1, 1, 1},
      .usage = TextureUsage::Attachment,
  };
  return createFramebufferTexture(desc, "Depth buffer");
}

Result<ShaderHandle, std::string>
LvkGPUDevice::createShaderModule(const ShaderDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (desc.source.empty()) {
    if (!desc.moduleName.empty()) {
      NURI_LOG_WARNING("LvkGPUDevice::createShaderModule: Shader source is "
                       "empty for module '%.*s'",
                       static_cast<int>(desc.moduleName.size()),
                       desc.moduleName.data());
    } else {
      NURI_LOG_WARNING(
          "LvkGPUDevice::createShaderModule: Shader source is empty");
    }
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
    if (!moduleNameStorage.empty()) {
      NURI_LOG_WARNING("LvkGPUDevice::createShaderModule: Failed to create "
                       "shader module '%s': %s",
                       moduleNameStorage.c_str(), res.message);
    } else {
      NURI_LOG_WARNING("LvkGPUDevice::createShaderModule: Failed to create "
                       "shader module: %s",
                       res.message);
    }
    return Result<ShaderHandle, std::string>::makeError(
        std::string(res.message));
  }
  if (!handle.valid()) {
    if (!moduleNameStorage.empty()) {
      NURI_LOG_WARNING("LvkGPUDevice::createShaderModule: Failed to create "
                       "shader module '%s'",
                       moduleNameStorage.c_str());
    } else {
      NURI_LOG_WARNING(
          "LvkGPUDevice::createShaderModule: Failed to create shader module");
    }
    return Result<ShaderHandle, std::string>::makeError(
        "Failed to create shader module");
  }

  ShaderHandle nuriHandle =
      impl_->shaders.allocate(std::move(handle), std::move(moduleNameStorage));
  return Result<ShaderHandle, std::string>::makeResult(nuriHandle);
}

Result<RenderPipelineHandle, std::string>
LvkGPUDevice::createRenderPipeline(const RenderPipelineDesc &desc,
                                   std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(desc.vertexShader)) {
    NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Invalid vertex "
                     "shader handle for render pipeline");
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Invalid vertex shader handle");
  }
  if (!isValid(desc.fragmentShader)) {
    NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Invalid fragment "
                     "shader handle for render pipeline");
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Invalid fragment shader handle");
  }

  const bool hasTessControlShader = isValid(desc.tessControlShader);
  const bool hasTessEvalShader = isValid(desc.tessEvalShader);
  const bool hasGeometryShader = isValid(desc.geometryShader);
  if (nuri::isValid(desc.tessControlShader) && !hasTessControlShader) {
    NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Invalid tessellation "
                     "control shader handle for render pipeline");
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Invalid tessellation control shader handle");
  }
  if (nuri::isValid(desc.tessEvalShader) && !hasTessEvalShader) {
    NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Invalid tessellation "
                     "evaluation shader handle for render pipeline");
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Invalid tessellation evaluation shader handle");
  }
  if (nuri::isValid(desc.geometryShader) && !hasGeometryShader) {
    NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Invalid geometry "
                     "shader handle for render pipeline");
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Invalid geometry shader handle");
  }
  if (hasTessControlShader != hasTessEvalShader) {
    NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Tessellation "
                     "control/eval shaders must both be valid or both be "
                     "unset");
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Tessellation control/eval shaders must both be valid or both be "
        "unset");
  }

  const bool hasTessellation = hasTessControlShader && hasTessEvalShader;
  if (hasTessellation) {
    if (desc.topology != Topology::Patch) {
      NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Tessellation "
                       "requires patch topology");
      return Result<RenderPipelineHandle, std::string>::makeError(
          "Tessellation requires patch topology");
    }
    if (desc.patchControlPoints == 0) {
      NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Tessellation "
                       "requires patchControlPoints > 0");
      return Result<RenderPipelineHandle, std::string>::makeError(
          "Tessellation requires patchControlPoints > 0");
    }
  } else {
    if (desc.patchControlPoints != 0) {
      NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: "
                       "patchControlPoints must be 0 when tessellation is "
                       "disabled");
      return Result<RenderPipelineHandle, std::string>::makeError(
          "patchControlPoints must be 0 when tessellation is disabled");
    }
    if (desc.topology == Topology::Patch) {
      NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Patch topology "
                       "requires tessellation shaders");
      return Result<RenderPipelineHandle, std::string>::makeError(
          "Patch topology requires tessellation shaders");
    }
  }

  const auto reserved = impl_->renderPipelines.reserve(std::string(debugName),
                                                       Format::RGBA8_UNORM);

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
        NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Render pipeline "
                         "specialization entry offset+size exceeds "
                         "specInfo.dataSize");
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
      .topology = toLvkTopology(desc.topology),
      .vertexInput = vertexInput,
      .smVert = impl_->shaders.getLvkHandle(desc.vertexShader),
      .smTesc = impl_->shaders.getLvkHandle(desc.tessControlShader),
      .smTese = impl_->shaders.getLvkHandle(desc.tessEvalShader),
      .smGeom = impl_->shaders.getLvkHandle(desc.geometryShader),
      .smFrag = impl_->shaders.getLvkHandle(desc.fragmentShader),
      .specInfo = specInfo,
      .color = {{.format = toLvkFormat(desc.colorFormats[0]),
                 .blendEnabled = desc.blendEnabled,
                 .srcRGBBlendFactor = desc.blendEnabled
                                          ? lvk::BlendFactor_SrcAlpha
                                          : lvk::BlendFactor_One,
                 .srcAlphaBlendFactor = lvk::BlendFactor_One,
                 .dstRGBBlendFactor = desc.blendEnabled
                                          ? lvk::BlendFactor_OneMinusSrcAlpha
                                          : lvk::BlendFactor_Zero,
                 .dstAlphaBlendFactor = desc.blendEnabled
                                            ? lvk::BlendFactor_OneMinusSrcAlpha
                                            : lvk::BlendFactor_Zero}},
      .depthFormat = toLvkFormat(desc.depthFormat),
      .cullMode = toLvkCullMode(desc.cullMode),
      .polygonMode = toLvkPolygonMode(desc.polygonMode),
      .patchControlPoints = desc.patchControlPoints,
      .debugName = reserved.debugNameCStr,
  };

  lvk::Result res;
  lvk::Holder<lvk::RenderPipelineHandle> handle =
      impl_->context->createRenderPipeline(pipelineDesc, &res);

  if (!res.isOk()) {
    impl_->renderPipelines.deallocate(reserved.handle);
    if (reserved.debugNameCStr && reserved.debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Failed to create "
                       "render pipeline '%s': %s",
                       reserved.debugNameCStr, res.message);
    } else {
      NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Failed to create "
                       "render pipeline: %s",
                       res.message);
    }
    return Result<RenderPipelineHandle, std::string>::makeError(
        std::string(res.message));
  }
  if (!handle.valid()) {
    impl_->renderPipelines.deallocate(reserved.handle);
    if (reserved.debugNameCStr && reserved.debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Failed to create "
                       "render pipeline '%s'",
                       reserved.debugNameCStr);
    } else {
      NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Failed to create "
                       "render pipeline");
    }
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Failed to create render pipeline");
  }

  if (!impl_->renderPipelines.setResource(reserved.handle, std::move(handle))) {
    impl_->renderPipelines.deallocate(reserved.handle);
    if (reserved.debugNameCStr && reserved.debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Failed to store "
                       "render pipeline resource '%s'",
                       reserved.debugNameCStr);
    } else {
      NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: Failed to store "
                       "render pipeline resource");
    }
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Failed to store render pipeline resource");
  }
  return Result<RenderPipelineHandle, std::string>::makeResult(reserved.handle);
}

Result<ComputePipelineHandle, std::string>
LvkGPUDevice::createComputePipeline(const ComputePipelineDesc &desc,
                                    std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(desc.computeShader)) {
    NURI_LOG_WARNING("LvkGPUDevice::createComputePipeline: Invalid compute "
                     "shader handle for compute pipeline");
    return Result<ComputePipelineHandle, std::string>::makeError(
        "Invalid compute shader handle");
  }

  const auto reserved = impl_->computePipelines.reserve(std::string(debugName),
                                                        Format::RGBA8_UNORM);

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
        NURI_LOG_WARNING("LvkGPUDevice::createComputePipeline: Compute "
                         "pipeline specialization entry offset+size exceeds "
                         "specInfo.dataSize");
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
      .debugName = reserved.debugNameCStr,
  };

  lvk::Result res;
  lvk::Holder<lvk::ComputePipelineHandle> handle =
      impl_->context->createComputePipeline(pipelineDesc, &res);

  if (!res.isOk()) {
    impl_->computePipelines.deallocate(reserved.handle);
    if (reserved.debugNameCStr && reserved.debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING("LvkGPUDevice::createComputePipeline: Failed to create "
                       "compute pipeline '%s': %s",
                       reserved.debugNameCStr, res.message);
    } else {
      NURI_LOG_WARNING("LvkGPUDevice::createComputePipeline: Failed to create "
                       "compute pipeline: %s",
                       res.message);
    }
    return Result<ComputePipelineHandle, std::string>::makeError(
        std::string(res.message));
  }
  if (!handle.valid()) {
    impl_->computePipelines.deallocate(reserved.handle);
    if (reserved.debugNameCStr && reserved.debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING("LvkGPUDevice::createComputePipeline: Failed to create "
                       "compute pipeline '%s'",
                       reserved.debugNameCStr);
    } else {
      NURI_LOG_WARNING("LvkGPUDevice::createComputePipeline: Failed to create "
                       "compute pipeline");
    }
    return Result<ComputePipelineHandle, std::string>::makeError(
        "Failed to create compute pipeline");
  }

  if (!impl_->computePipelines.setResource(reserved.handle,
                                           std::move(handle))) {
    impl_->computePipelines.deallocate(reserved.handle);
    if (reserved.debugNameCStr && reserved.debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING("LvkGPUDevice::createComputePipeline: Failed to store "
                       "compute pipeline resource '%s'",
                       reserved.debugNameCStr);
    } else {
      NURI_LOG_WARNING("LvkGPUDevice::createComputePipeline: Failed to store "
                       "compute pipeline resource");
    }
    return Result<ComputePipelineHandle, std::string>::makeError(
        "Failed to store compute pipeline resource");
  }
  return Result<ComputePipelineHandle, std::string>::makeResult(
      reserved.handle);
}

void LvkGPUDevice::destroyRenderPipeline(RenderPipelineHandle pipeline) {
  if (!impl_) {
    return;
  }
  impl_->renderPipelines.deallocate(pipeline);
}

void LvkGPUDevice::destroyComputePipeline(ComputePipelineHandle pipeline) {
  if (!impl_) {
    return;
  }
  impl_->computePipelines.deallocate(pipeline);
}

void LvkGPUDevice::destroyBuffer(BufferHandle buffer) {
  if (!impl_) {
    return;
  }
  impl_->buffers.deallocate(buffer);
}

void LvkGPUDevice::destroyTexture(TextureHandle texture) {
  if (!impl_) {
    return;
  }
  impl_->framebufferTextures.erase(
      std::remove_if(impl_->framebufferTextures.begin(),
                     impl_->framebufferTextures.end(),
                     [texture](const FramebufferTexture &entry) {
                       return entry.handle.index == texture.index &&
                              entry.handle.generation == texture.generation;
                     }),
      impl_->framebufferTextures.end());
  impl_->textures.deallocate(texture);
}

void LvkGPUDevice::destroyShaderModule(ShaderHandle shader) {
  if (!impl_) {
    return;
  }
  impl_->shaders.deallocate(shader);
}

bool LvkGPUDevice::isValid(BufferHandle h) const {
  return impl_->buffers.isValid(h);
}

bool LvkGPUDevice::isValid(TextureHandle h) const {
  return impl_->textures.isValid(h);
}

bool LvkGPUDevice::isValid(ShaderHandle h) const {
  return impl_->shaders.isValid(h);
}

bool LvkGPUDevice::isValid(RenderPipelineHandle h) const {
  return impl_->renderPipelines.isValid(h);
}

bool LvkGPUDevice::isValid(ComputePipelineHandle h) const {
  return impl_->computePipelines.isValid(h);
}

Format LvkGPUDevice::getTextureFormat(TextureHandle h) const {
  return impl_->textures.getFormat(h);
}

uint32_t LvkGPUDevice::getTextureBindlessIndex(TextureHandle h) const {
  if (!impl_->textures.isValid(h)) {
    return 0;
  }
  return impl_->textures.getLvkHandle(h).index();
}

uint64_t LvkGPUDevice::getBufferDeviceAddress(BufferHandle h,
                                              size_t offset) const {
  if (!impl_->buffers.isValid(h)) {
    return 0;
  }
  if ((offset & 7u) != 0u) {
    NURI_LOG_WARNING("LvkGPUDevice::getBufferDeviceAddress: Offset must be "
                     "8-byte aligned");
    return 0;
  }
  return impl_->context->gpuAddress(impl_->buffers.getLvkHandle(h), offset);
}

bool LvkGPUDevice::resolveGeometry(GeometryAllocationHandle h,
                                   GeometryAllocationView &out) const {
  return impl_->geometryPool ? impl_->geometryPool->resolve(h, out) : false;
}

Result<GeometryAllocationHandle, std::string> LvkGPUDevice::allocateGeometry(
    std::span<const std::byte> vertexBytes, uint32_t vertexCount,
    std::span<const std::byte> indexBytes, uint32_t indexCount,
    std::string_view debugName) {
  if (!impl_->geometryPool) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        "Geometry pool is not initialized");
  }
  return impl_->geometryPool->allocate(vertexBytes, vertexCount, indexBytes,
                                       indexCount, debugName);
}

void LvkGPUDevice::releaseGeometry(GeometryAllocationHandle h) {
  if (impl_->geometryPool) {
    impl_->geometryPool->release(h);
  }
}

Result<bool, std::string> LvkGPUDevice::submitFrame(const RenderFrame &frame) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_SUBMIT);
  if (frame.passes.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  lvk::ICommandBuffer &commandBuffer = impl_->context->acquireCommandBuffer();
  const lvk::TextureHandle swapchainTexture =
      impl_->context->getCurrentSwapchainTexture();
  if (!swapchainTexture.valid()) {
    return Result<bool, std::string>::makeError(
        "LvkGPUDevice::submitFrame: invalid swapchain texture");
  }
  const auto fillDependencies =
      [this](std::span<const BufferHandle> dependencyBuffers,
             lvk::Dependencies &deps,
             std::string_view context) -> Result<bool, std::string> {
    if (dependencyBuffers.size() >
        lvk::Dependencies::LVK_MAX_SUBMIT_DEPENDENCIES) {
      return Result<bool, std::string>::makeError(
          std::string(context) + ": dependency buffer count exceeds " +
          std::to_string(lvk::Dependencies::LVK_MAX_SUBMIT_DEPENDENCIES));
    }

    size_t dstIndex = 0;
    for (const BufferHandle bufferHandle : dependencyBuffers) {
      if (!nuri::isValid(bufferHandle)) {
        continue;
      }
      if (!impl_->buffers.isValid(bufferHandle)) {
        return Result<bool, std::string>::makeError(
            std::string(context) + ": dependency buffer is invalid");
      }
      deps.buffers[dstIndex++] = impl_->buffers.getLvkHandle(bufferHandle);
    }
    return Result<bool, std::string>::makeResult(true);
  };

  for (const RenderPass &pass : frame.passes) {
    if (!pass.debugLabel.empty()) {
      const std::string label(pass.debugLabel);
      commandBuffer.cmdPushDebugGroupLabel(label.c_str(), pass.debugColor);
    }

    lvk::RenderPass renderPass{};
    renderPass.color[0] = {
        .loadOp = toLvkLoadOp(pass.color.loadOp),
        .storeOp = toLvkStoreOp(pass.color.storeOp),
        .clearColor = {pass.color.clearColor.r, pass.color.clearColor.g,
                       pass.color.clearColor.b, pass.color.clearColor.a},
    };

    lvk::Framebuffer framebuffer{};
    framebuffer.color[0] = {.texture = swapchainTexture};

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

    {
      for (const ComputeDispatchItem &dispatch : pass.preDispatches) {
        NURI_PROFILER_ZONE("LvkGPUDevice.compute_dispatch_submission",
                           NURI_PROFILER_COLOR_CMD_DISPATCH);
        if (!impl_->computePipelines.isValid(dispatch.pipeline)) {
          if (!pass.debugLabel.empty()) {
            commandBuffer.cmdPopDebugGroupLabel();
          }
          return Result<bool, std::string>::makeError(
              "Invalid compute pipeline handle");
        }
        if (dispatch.dispatch.x == 0 || dispatch.dispatch.y == 0 ||
            dispatch.dispatch.z == 0) {
          if (!pass.debugLabel.empty()) {
            commandBuffer.cmdPopDebugGroupLabel();
          }
          return Result<bool, std::string>::makeError(
              "Invalid compute dispatch size");
        }

        lvk::Dependencies dispatchDependencies{};
        auto dispatchDepsResult =
            fillDependencies(dispatch.dependencyBuffers, dispatchDependencies,
                             "LvkGPUDevice::submitFrame compute dispatch");
        if (dispatchDepsResult.hasError()) {
          if (!pass.debugLabel.empty()) {
            commandBuffer.cmdPopDebugGroupLabel();
          }
          return dispatchDepsResult;
        }

        if (!dispatch.debugLabel.empty()) {
          const std::string label(dispatch.debugLabel);
          commandBuffer.cmdPushDebugGroupLabel(label.c_str(),
                                               dispatch.debugColor);
        }

        commandBuffer.cmdBindComputePipeline(
            impl_->computePipelines.getLvkHandle(dispatch.pipeline));
        if (!dispatch.pushConstants.empty()) {
          commandBuffer.cmdPushConstants(
              static_cast<const void *>(dispatch.pushConstants.data()),
              dispatch.pushConstants.size(), 0);
        }
        commandBuffer.cmdDispatchThreadGroups({.width = dispatch.dispatch.x,
                                               .height = dispatch.dispatch.y,
                                               .depth = dispatch.dispatch.z},
                                              dispatchDependencies);

        if (!dispatch.debugLabel.empty()) {
          commandBuffer.cmdPopDebugGroupLabel();
        }
      }
      NURI_PROFILER_ZONE_END();
    }

    lvk::Dependencies renderDependencies{};
    auto renderDepsResult =
        fillDependencies(pass.dependencyBuffers, renderDependencies,
                         "LvkGPUDevice::submitFrame render pass");
    if (renderDepsResult.hasError()) {
      if (!pass.debugLabel.empty()) {
        commandBuffer.cmdPopDebugGroupLabel();
      }
      return renderDepsResult;
    }

    commandBuffer.cmdBeginRendering(renderPass, framebuffer,
                                    renderDependencies);

    // Pipelines use dynamic viewport/scissor in LVK, so we must bind them for
    // every pass (even if the pass didn't specify overrides).
    Viewport vp{};
    if (pass.useViewport) {
      vp = pass.viewport;
    } else {
      const lvk::Dimensions dim =
          impl_->context->getDimensions(framebuffer.color[0].texture);
      vp = {
          .x = 0.0f,
          .y = 0.0f,
          .width = static_cast<float>(dim.width),
          .height = static_cast<float>(dim.height),
          .minDepth = 0.0f,
          .maxDepth = 1.0f,
      };
    }

    commandBuffer.cmdBindViewport({
        .x = vp.x,
        .y = vp.y,
        .width = vp.width,
        .height = vp.height,
        .minDepth = vp.minDepth,
        .maxDepth = vp.maxDepth,
    });
    // Default scissor for the pass; individual draws may override.
    commandBuffer.cmdBindScissorRect({
        static_cast<uint32_t>(vp.x),
        static_cast<uint32_t>(vp.y),
        static_cast<uint32_t>(vp.width),
        static_cast<uint32_t>(vp.height),
    });

    bool scissorMatchesViewport = true;

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
        const std::string label(draw.debugLabel);
        commandBuffer.cmdPushDebugGroupLabel(label.c_str(), draw.debugColor);
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

      if (draw.useScissor) {
        commandBuffer.cmdBindScissorRect({draw.scissor.x, draw.scissor.y,
                                          draw.scissor.width,
                                          draw.scissor.height});
        scissorMatchesViewport = false;
      } else if (!scissorMatchesViewport) {
        commandBuffer.cmdBindScissorRect({
            static_cast<uint32_t>(vp.x),
            static_cast<uint32_t>(vp.y),
            static_cast<uint32_t>(vp.width),
            static_cast<uint32_t>(vp.height),
        });
        scissorMatchesViewport = true;
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

  impl_->context->submit(commandBuffer, swapchainTexture);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
LvkGPUDevice::updateBuffer(BufferHandle buffer, std::span<const std::byte> data,
                           size_t offset) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  if (!impl_->buffers.isValid(buffer)) {
    return Result<bool, std::string>::makeError("Invalid buffer handle");
  }
  if (data.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  const lvk::BufferHandle lvkBuf = impl_->buffers.getLvkHandle(buffer);
  const size_t bufferSize =
      static_cast<size_t>(lvk::getBufferSize(impl_->context.get(), lvkBuf));
  if (offset > bufferSize || data.size() > bufferSize - offset) {
    return Result<bool, std::string>::makeError(
        "updateBuffer: offset + data.size() exceeds buffer size");
  }

  if (uint8_t *mapped = impl_->context->getMappedPtr(lvkBuf)) {
    std::memcpy(mapped + offset, data.data(), data.size());
    impl_->context->flushMappedMemory(lvkBuf, offset, data.size());
    return Result<bool, std::string>::makeResult(true);
  }

  lvk::Result res = impl_->context->upload(
      lvkBuf, static_cast<const void *>(data.data()), data.size(), offset);
  if (!res.isOk()) {
    return Result<bool, std::string>::makeError(std::string(res.message));
  }
  return Result<bool, std::string>::makeResult(true);
}

std::byte *LvkGPUDevice::getMappedBufferPtr(BufferHandle buffer) {
  if (!impl_->buffers.isValid(buffer)) {
    return nullptr;
  }
  const lvk::BufferHandle lvkBuf = impl_->buffers.getLvkHandle(buffer);
  return reinterpret_cast<std::byte *>(impl_->context->getMappedPtr(lvkBuf));
}

void LvkGPUDevice::flushMappedBuffer(BufferHandle buffer, size_t offset,
                                     size_t size) {
  if (!impl_->buffers.isValid(buffer) || size == 0) {
    return;
  }
  const lvk::BufferHandle lvkBuf = impl_->buffers.getLvkHandle(buffer);
  impl_->context->flushMappedMemory(lvkBuf, offset, size);
}

Result<bool, std::string>
LvkGPUDevice::copyBufferRegions(std::span<const BufferCopyRegion> regions) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  if (regions.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  // Validate all regions before acquiring a command buffer so we never
  // acquire without submitting.
  size_t copyCount = 0;
  for (const BufferCopyRegion &region : regions) {
    if (region.size == 0) {
      continue;
    }
    if (!impl_->buffers.isValid(region.srcBuffer) ||
        !impl_->buffers.isValid(region.dstBuffer)) {
      return Result<bool, std::string>::makeError(
          "copyBufferRegions: invalid source or destination buffer");
    }

    const lvk::BufferHandle src = impl_->buffers.getLvkHandle(region.srcBuffer);
    const lvk::BufferHandle dst = impl_->buffers.getLvkHandle(region.dstBuffer);
    const size_t srcSize =
        static_cast<size_t>(lvk::getBufferSize(impl_->context.get(), src));
    const size_t dstSize =
        static_cast<size_t>(lvk::getBufferSize(impl_->context.get(), dst));
    if (region.srcOffset > srcSize ||
        region.size > srcSize - region.srcOffset) {
      return Result<bool, std::string>::makeError(
          "copyBufferRegions: source copy range is out of bounds");
    }
    if (region.dstOffset > dstSize ||
        region.size > dstSize - region.dstOffset) {
      return Result<bool, std::string>::makeError(
          "copyBufferRegions: destination copy range is out of bounds");
    }
    ++copyCount;
  }

  if (copyCount == 0) {
    return Result<bool, std::string>::makeResult(true);
  }

  lvk::ICommandBuffer &commandBuffer = impl_->context->acquireCommandBuffer();
  for (const BufferCopyRegion &region : regions) {
    if (region.size == 0) {
      continue;
    }
    const lvk::BufferHandle src = impl_->buffers.getLvkHandle(region.srcBuffer);
    const lvk::BufferHandle dst = impl_->buffers.getLvkHandle(region.dstBuffer);
    commandBuffer.cmdCopyBuffer(src, dst, region.srcOffset, region.dstOffset,
                                region.size);
  }

  // TODO: move this to an async pre-pass copy path once frame graph support
  // exists.
  const lvk::SubmitHandle submitHandle = impl_->context->submit(commandBuffer);
  impl_->context->wait(submitHandle);
  return Result<bool, std::string>::makeResult(true);
}

void LvkGPUDevice::waitIdle() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_WAIT);
  if (impl_->context) {
    // Empty SubmitHandle results in vkDeviceWaitIdle
    impl_->context->wait(lvk::SubmitHandle{});
  }
}

} // namespace nuri
