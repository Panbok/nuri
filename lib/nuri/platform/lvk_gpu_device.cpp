#include "nuri/pch.h"

#include "nuri/platform/lvk_gpu_device.h"

#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/core/window.h"
#include "nuri/resources/gpu/geometry_pool.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/utils/env_utils.h"

#include <lvk/LVK.h>
#if __has_include(<lvk/vulkan/VulkanClasses.h>)
#include <lvk/vulkan/VulkanClasses.h>
#define NURI_LVK_HAS_VULKAN_COMMAND_BUFFER 1
#else
#define NURI_LVK_HAS_VULKAN_COMMAND_BUFFER 0
#endif
#include <vulkan/VulkanUtils.h>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace nuri {

namespace {

[[nodiscard]] bool isRenderDocAttached() {
#if defined(_WIN32)
  if (GetModuleHandleA("renderdoc.dll") != nullptr ||
      GetModuleHandleA("renderdoccmd.dll") != nullptr) {
    return true;
  }
#endif
  return readEnvVar("RENDERDOC_CAPFILE").has_value();
}

[[nodiscard]] bool resolveValidationEnabled(bool renderDocAttached) {
  if (const std::optional<bool> override =
          readEnvBoolOverride("NURI_VK_VALIDATION");
      override.has_value()) {
    return *override;
  }
#if defined(NURI_DEBUG)
  return true;
#else
  return renderDocAttached;
#endif
}

[[nodiscard]] bool resolveVerboseVkDiagnostics(bool renderDocAttached) {
  if (const std::optional<bool> override =
          readEnvBoolOverride("NURI_VK_DIAGNOSTICS");
      override.has_value()) {
    return *override;
  }
  return renderDocAttached;
}

[[nodiscard]] bool resolveTerminateOnValidationError() {
  return readEnvFlag("NURI_VK_TERMINATE_ON_VALIDATION_ERROR");
}

[[nodiscard]] std::string formatVkVersion(uint32_t version) {
  std::array<char, 32> text{};
  const int written =
      std::snprintf(text.data(), text.size(), "%u.%u.%u",
                    static_cast<unsigned>(VK_API_VERSION_MAJOR(version)),
                    static_cast<unsigned>(VK_API_VERSION_MINOR(version)),
                    static_cast<unsigned>(VK_API_VERSION_PATCH(version)));
  if (written <= 0 || static_cast<size_t>(written) >= text.size()) {
    return std::to_string(version);
  }
  return std::string(text.data(), static_cast<size_t>(written));
}

[[nodiscard]] std::string
formatRecordedCommandBufferHandle(RecordedCommandBufferHandle handle) {
  std::array<char, 48> text{};
  const int written = std::snprintf(text.data(), text.size(), "%u:%u",
                                    handle.index, handle.generation);
  if (written <= 0 || static_cast<size_t>(written) >= text.size()) {
    return "<invalid>";
  }
  return std::string(text.data(), static_cast<size_t>(written));
}

struct CommonMeshPushConstantsProbe {
  uint64_t frameDataAddress = 0u;
  uint64_t vertexBufferAddress = 0u;
  uint64_t vertexDecodeBufferAddress = 0u;
  uint64_t instanceMatricesAddress = 0u;
  uint64_t previousInstanceMatricesAddress = 0u;
  uint64_t instanceRemapAddress = 0u;
  uint64_t instanceCentersPhaseAddress = 0u;
  uint64_t instanceBaseMatricesAddress = 0u;
  uint64_t velocityInstanceFlagsAddress = 0u;
  uint64_t velocityFrameDataAddress = 0u;
  uint32_t instanceCount = 0u;
  uint32_t materialIndex = 0u;
  uint32_t vertexDecodeIndex = 0u;
  uint32_t packedVertexFormat = 0u;
  float timeSeconds = 0.0f;
  float tessNearDistance = 0.0f;
  float tessFarDistance = 0.0f;
  float tessMinFactor = 0.0f;
  float tessMaxFactor = 0.0f;
  uint32_t debugVisualizationMode = 0u;
  uint32_t shadowCascadeIndex = 0u;
};
static_assert(sizeof(CommonMeshPushConstantsProbe) == 128u);
static_assert(offsetof(CommonMeshPushConstantsProbe, instanceRemapAddress) ==
              40u);
static_assert(offsetof(CommonMeshPushConstantsProbe,
                       instanceCentersPhaseAddress) == 48u);
static_assert(offsetof(CommonMeshPushConstantsProbe,
                       instanceBaseMatricesAddress) == 56u);
static_assert(offsetof(CommonMeshPushConstantsProbe, instanceCount) == 80u);
static_assert(offsetof(CommonMeshPushConstantsProbe, shadowCascadeIndex) ==
              120u);

constexpr size_t kCommonMeshPushConstantsProbeBytes =
    offsetof(CommonMeshPushConstantsProbe, shadowCascadeIndex) +
    sizeof(uint32_t);
constexpr size_t kDebugDraw3DVertexBufferAddressOffset = sizeof(float) * 16u;

[[nodiscard]] const char *drawCommandTypeName(DrawCommandType command) {
  switch (command) {
  case DrawCommandType::Direct:
    return "Direct";
  case DrawCommandType::IndexedIndirect:
    return "IndexedIndirect";
  case DrawCommandType::IndexedIndirectCount:
    return "IndexedIndirectCount";
  }
  return "Unknown";
}

// Conversion functions from Nuri types to LVK types

lvk::Format toLvkFormat(Format format) {
  switch (format) {
  case Format::R32_UINT:
    return lvk::Format_R_UI32;
  case Format::R32_FLOAT:
    return lvk::Format_R_F32;
  case Format::RG32_FLOAT:
    return lvk::Format_RG_F32;
  case Format::RG16_FLOAT:
    return lvk::Format_RG_F16;
  case Format::RGBA8_UNORM:
    return lvk::Format_RGBA_UN8;
  case Format::RGBA8_SRGB:
    return lvk::Format_RGBA_SRGB8;
  case Format::RGBA8_UINT:
    return lvk::Format_RGBA_UI32;
  case Format::RGBA16_FLOAT:
    return lvk::Format_RGBA_F16;
  case Format::RGBA32_FLOAT:
    return lvk::Format_RGBA_F32;
  case Format::BC7_RGBA_UNORM:
    return lvk::Format_BC7_RGBA;
  case Format::BC7_RGBA_SRGB:
    return lvk::Format_BC7_SRGBA;
  case Format::ETC2_RGB8_UNORM:
    return lvk::Format_ETC2_RGB8;
  case Format::ETC2_RGB8_SRGB:
    return lvk::Format_ETC2_SRGB8;
  case Format::D16_UNORM:
    return lvk::Format_Z_UN16;
  case Format::D32_FLOAT:
    return lvk::Format_Z_F32;
  case Format::Count:
    break;
  }
  return lvk::Format_Invalid;
}

Format fromLvkFormat(lvk::Format format) {
  switch (format) {
  case lvk::Format_R_UI32:
    return Format::R32_UINT;
  case lvk::Format_R_F32:
    return Format::R32_FLOAT;
  case lvk::Format_RG_F32:
    return Format::RG32_FLOAT;
  case lvk::Format_RG_F16:
    return Format::RG16_FLOAT;
  case lvk::Format_RGBA_UN8:
    return Format::RGBA8_UNORM;
  case lvk::Format_RGBA_SRGB8:
    return Format::RGBA8_SRGB;
  case lvk::Format_RGBA_UI32:
    return Format::RGBA8_UINT;
  case lvk::Format_RGBA_F16:
    return Format::RGBA16_FLOAT;
  case lvk::Format_RGBA_F32:
    return Format::RGBA32_FLOAT;
  case lvk::Format_BC7_RGBA:
    return Format::BC7_RGBA_UNORM;
  case lvk::Format_BC7_SRGBA:
    return Format::BC7_RGBA_SRGB;
  case lvk::Format_ETC2_RGB8:
    return Format::ETC2_RGB8_UNORM;
  case lvk::Format_ETC2_SRGB8:
    return Format::ETC2_RGB8_SRGB;
  case lvk::Format_Z_UN16:
    return Format::D16_UNORM;
  case lvk::Format_Z_F32:
    return Format::D32_FLOAT;
  default:
    break;
  }
  return Format::RGBA8_UNORM;
}

uint8_t toLvkBufferUsage(BufferUsage usage) {
  uint8_t bits = 0;
  if (hasBufferUsage(usage, BufferUsage::Vertex)) {
    bits |= lvk::BufferUsageBits_Vertex;
  }
  if (hasBufferUsage(usage, BufferUsage::Index)) {
    bits |= lvk::BufferUsageBits_Index;
  }
  if (hasBufferUsage(usage, BufferUsage::Uniform)) {
    bits |= lvk::BufferUsageBits_Uniform;
  }
  if (hasBufferUsage(usage, BufferUsage::Storage)) {
    bits |= lvk::BufferUsageBits_Storage;
  }
  if (hasBufferUsage(usage, BufferUsage::Indirect)) {
    bits |= lvk::BufferUsageBits_Indirect;
  }
  return bits;
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

lvk::SamplerFilter toLvkSamplerFilter(SamplerFilter filter) {
  switch (filter) {
  case SamplerFilter::Nearest:
    return lvk::SamplerFilter_Nearest;
  case SamplerFilter::Linear:
    return lvk::SamplerFilter_Linear;
  case SamplerFilter::Count:
    break;
  }
  return lvk::SamplerFilter_Linear;
}

lvk::SamplerMip toLvkSamplerMipMode(SamplerMipMode mode) {
  switch (mode) {
  case SamplerMipMode::Disabled:
    return lvk::SamplerMip_Disabled;
  case SamplerMipMode::Nearest:
    return lvk::SamplerMip_Nearest;
  case SamplerMipMode::Linear:
    return lvk::SamplerMip_Linear;
  case SamplerMipMode::Count:
    break;
  }
  return lvk::SamplerMip_Disabled;
}

lvk::SamplerWrap toLvkSamplerWrapMode(SamplerWrapMode mode) {
  switch (mode) {
  case SamplerWrapMode::Repeat:
    return lvk::SamplerWrap_Repeat;
  case SamplerWrapMode::MirrorRepeat:
    return lvk::SamplerWrap_MirrorRepeat;
  case SamplerWrapMode::Clamp:
    return lvk::SamplerWrap_Clamp;
  case SamplerWrapMode::Count:
    break;
  }
  return lvk::SamplerWrap_Repeat;
}

lvk::TextureUsageBits toLvkTextureUsage(TextureUsage usage) {
  switch (usage) {
  case TextureUsage::Sampled:
    return lvk::TextureUsageBits_Sampled;
  case TextureUsage::Storage:
    return lvk::TextureUsageBits_Storage;
  case TextureUsage::Attachment:
    return lvk::TextureUsageBits_Attachment;
  case TextureUsage::AttachmentSampled:
    return static_cast<lvk::TextureUsageBits>(lvk::TextureUsageBits_Attachment |
                                              lvk::TextureUsageBits_Sampled);
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
  case StoreOp::MsaaResolve:
    return lvk::StoreOp_MsaaResolve;
  case StoreOp::DontCare:
    return lvk::StoreOp_DontCare;
  }
  return lvk::StoreOp_DontCare;
}

lvk::ResolveMode toLvkResolveMode(ResolveMode mode) {
  switch (mode) {
  case ResolveMode::None:
    return lvk::ResolveMode_None;
  case ResolveMode::SampleZero:
    return lvk::ResolveMode_SampleZero;
  case ResolveMode::Average:
    return lvk::ResolveMode_Average;
  case ResolveMode::Min:
    return lvk::ResolveMode_Min;
  case ResolveMode::Max:
    return lvk::ResolveMode_Max;
  case ResolveMode::Count:
    break;
  }
  return lvk::ResolveMode_Average;
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

template <typename HandleType>
[[nodiscard]] constexpr bool areSameHandle(HandleType a, HandleType b) {
  return a.index == b.index && a.generation == b.generation;
}

[[nodiscard]] Result<uint32_t, std::string>
allocateMonotonicHandleIndex(uint32_t &nextIndex, std::string_view context) {
  if (nextIndex == std::numeric_limits<uint32_t>::max()) {
    std::string message;
    message.reserve(context.size() + 24u);
    message.append(context);
    message.append(": handle index overflow");
    return Result<uint32_t, std::string>::makeError(std::move(message));
  }

  return Result<uint32_t, std::string>::makeResult(nextIndex++);
}

[[nodiscard]] uint32_t nextHandleGeneration(std::vector<uint32_t> &generations,
                                            uint32_t index) {
  if (index >= generations.size()) {
    generations.resize(static_cast<size_t>(index) + 1u, 0u);
  }

  uint32_t &generation = generations[index];
  ++generation;
  if (generation == 0u) {
    ++generation;
  }
  return generation;
}

[[nodiscard]] Result<bool, std::string>
makeLvkBarrierApiError(std::string_view context) {
  std::string message;
  message.reserve(context.size() + 240u);
  message.append(context);
  message.append(": buffer barriers and explicit texture layout transitions "
                 "require the vendored Vulkan lvk::CommandBuffer "
                 "implementation from the current external/lightweightvk "
                 "snapshot because this LVK ICommandBuffer API only exposes "
                 "shader-read texture transitions");
  return Result<bool, std::string>::makeError(std::move(message));
}

[[nodiscard]] SubmissionHandle
toNuriSubmissionHandle(lvk::SubmitHandle handle) {
  if (handle.empty()) {
    return {};
  }
  return SubmissionHandle{
      .index = handle.bufferIndex_,
      .generation = handle.submitId_,
  };
}

[[nodiscard]] lvk::SubmitHandle toLvkSubmitHandle(SubmissionHandle handle) {
  if (!isValid(handle)) {
    return {};
  }
  const uint64_t packed =
      (static_cast<uint64_t>(handle.generation) << 32u) + handle.index;
  return lvk::SubmitHandle(packed);
}

[[nodiscard]] bool pushDebugLabel(lvk::ICommandBuffer &commandBuffer,
                                  std::string_view label, uint32_t color) {
  if (label.empty()) {
    return false;
  }

  // Avoid heap allocations for common short labels.
  constexpr size_t kInlineDebugLabelCapacity = 128;
  if (label.size() < kInlineDebugLabelCapacity) {
    std::array<char, kInlineDebugLabelCapacity> inlineLabel{};
    std::memcpy(inlineLabel.data(), label.data(), label.size());
    inlineLabel[label.size()] = '\0';
    commandBuffer.cmdPushDebugGroupLabel(inlineLabel.data(), color);
    return true;
  }

  std::string heapLabel(label);
  commandBuffer.cmdPushDebugGroupLabel(heapLabel.c_str(), color);
  return true;
}

constexpr bool kEnablePerDrawDebugLabels = false;
constexpr uint32_t kMaxGraphicsRecordingContexts = 8u;
constexpr uint32_t kGpuTimingIntervalsPerContext = 16u;
constexpr uint32_t kGpuTimingQueriesPerContext =
    kGpuTimingIntervalsPerContext * 2u;
constexpr uint32_t kGpuTimingQueryPoolSize =
    kMaxGraphicsRecordingContexts * kGpuTimingQueriesPerContext;
constexpr uint32_t kInvalidTimingQueryIndex = UINT32_MAX;

[[nodiscard]] Result<bool, std::string>
makeDependencyError(std::string_view context, std::string_view detail) {
  std::string message;
  message.reserve(context.size() + 2 + detail.size());
  message.append(context);
  message.append(": ");
  message.append(detail);
  return Result<bool, std::string>::makeError(std::move(message));
}

[[nodiscard]] Result<bool, std::string>
makeDependencyCountExceededError(std::string_view context) {
  std::array<char, 96> detail{};
  const int written = std::snprintf(
      detail.data(), detail.size(), "dependency resource count exceeds %u",
      static_cast<unsigned>(kMaxDependencyResources));
  if (written > 0 && static_cast<size_t>(written) < detail.size()) {
    return makeDependencyError(
        context, std::string_view(detail.data(), static_cast<size_t>(written)));
  }
  return makeDependencyError(context,
                             "dependency resource count exceeds limit");
}

constexpr std::array<uint8_t, 4> kSupportedAnisotropyLevels = {2u, 4u, 8u, 16u};

[[nodiscard]] SamplerHandle
findBestAnisotropicSampler(std::span<const SamplerHandle> samplers,
                           uint8_t requestedAnisotropy) {
  for (size_t i = samplers.size(); i-- > 0u;) {
    if (requestedAnisotropy < kSupportedAnisotropyLevels[i]) {
      continue;
    }
    if (isValid(samplers[i])) {
      return samplers[i];
    }
  }
  return {};
}

[[nodiscard]] VkPipelineStageFlags2
graphicsBarrierStages(GraphicsBarrierState state, bool isDepthTexture) {
  switch (state) {
  case GraphicsBarrierState::Attachment:
    return isDepthTexture ? (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
                          : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  case GraphicsBarrierState::Present:
    return VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
  case GraphicsBarrierState::Read:
  case GraphicsBarrierState::Write:
  case GraphicsBarrierState::Unknown:
  default:
    return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
           VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
           VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
  }
}

} // namespace

template <typename LvkHandle> struct ResourceSlot {
  lvk::Holder<LvkHandle> resource;
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
    const SlotReservation reservation = slotsMeta_.acquire();
    if (reservation.appended) {
      slots_.emplace_back();
    }

    auto &slot = slots_[reservation.index];
    slot.resource.reset();
    slot.debugName = std::move(debugName);
    slot.format = format;

    const char *cstr = slot.debugName.empty() ? "" : slot.debugName.c_str();
    return ReservedSlot{NuriHandle{reservation.index, reservation.generation},
                        cstr};
  }

  bool setResource(NuriHandle h, lvk::Holder<LvkHandle> &&resource) {
    if (!isValid(h))
      return false;
    slots_[h.index].resource = std::move(resource);
    return true;
  }

  NuriHandle allocate(lvk::Holder<LvkHandle> &&resource, std::string debugName,
                      Format format = Format::RGBA8_UNORM) {
    const SlotReservation reservation = slotsMeta_.acquire();
    if (reservation.appended) {
      slots_.emplace_back();
    }

    auto &slot = slots_[reservation.index];
    slot.resource = std::move(resource);
    slot.debugName = std::move(debugName);
    slot.format = format;

    return NuriHandle{reservation.index, reservation.generation};
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
    slot.debugName.clear();
    slotsMeta_.release(h.index);
  }

  bool isValid(NuriHandle h) const {
    return h.index < slots_.size() && slotsMeta_.isValid(h.index, h.generation);
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
  SlotPool<UnmaskedNonZeroGenerationPolicy> slotsMeta_;
};

struct FramebufferTexture {
  TextureHandle handle{};
  TextureDesc desc{};
  std::string debugName;
};

struct TimingQueryPoolSlot {
  lvk::Holder<lvk::QueryPoolHandle> handle{};
};

struct PendingTimingQueryRange {
  GpuTimingScope scope = GpuTimingScope::None;
  uint32_t firstQuery = 0u;
  uint32_t intervalCount = 0u;
};

struct CurrentFrameTimingCapture {
  TimingQueryPoolSlot pool{};
  uint64_t frameIndex = 0u;
};

struct PendingGpuTimingSubmission {
  SubmissionHandle submission{};
  uint64_t frameIndex = 0u;
  TimingQueryPoolSlot pool{};
  std::vector<PendingTimingQueryRange> timingRanges{};
};

struct ActiveGraphicsRecordingContext {
  RecordingContextHandle handle{};
  lvk::ICommandBuffer *commandBuffer = nullptr;
  uint32_t workerIndex = 0u;
  uint32_t timingQueryBase = 0u;
  uint32_t timingQueryCursor = 0u;
  std::vector<PendingTimingQueryRange> timingRanges{};
  bool hadShadowSdsmPass = false;
  bool timingQuerySliceReset = false;
};

struct RecordedGraphicsCommandBuffer {
  RecordedCommandBufferHandle handle{};
  lvk::ICommandBuffer *commandBuffer = nullptr;
  std::vector<PendingTimingQueryRange> timingRanges{};
  bool hadShadowSdsmPass = false;
};

struct LvkGPUDevice::Impl {
  Window *window = nullptr;
  std::unique_ptr<lvk::IContext> context;
  bool renderDocAttached = false;
  bool validationEnabled = false;
  bool verboseVkDiagnostics = false;
  uint64_t currentFrameIndex = 0u;
  uint64_t submittedFrameCount = 0u;
  uint32_t preparedSwapchainImageIndex = 0u;
  uint32_t preparedSwapchainImageCount = 0u;
  bool hasPreparedSwapchainImage = false;
  TextureCompressionCaps compressionCaps{};
  ResourceTable<SamplerHandle, lvk::SamplerHandle> samplers;
  SamplerHandle cubemapSampler{};
  SamplerHandle bilinearSampler{};
  SamplerHandle trilinearSampler{};
  std::array<SamplerHandle, 4> anisotropicSamplers{};
  uint8_t maxSamplerAnisotropy = 1u;

  ResourceTable<BufferHandle, lvk::BufferHandle> buffers;
  ResourceTable<TextureHandle, lvk::TextureHandle> textures;
  ResourceTable<ShaderHandle, lvk::ShaderModuleHandle> shaders;
  ResourceTable<RenderPipelineHandle, lvk::RenderPipelineHandle>
      renderPipelines;
  ResourceTable<ComputePipelineHandle, lvk::ComputePipelineHandle>
      computePipelines;
  std::vector<FramebufferTexture> framebufferTextures;
  mutable std::mutex contextImmediateMutex;
  std::mutex graphicsContextMutex;
  std::vector<ActiveGraphicsRecordingContext> activeGraphicsContexts;
  std::vector<uint8_t> activeGraphicsContextOccupied;
  std::vector<RecordedGraphicsCommandBuffer> recordedGraphicsCommandBuffers;
  lvk::TextureHandle currentFrameSwapchainTexture{};
  std::vector<uint32_t> recordingContextGenerations;
  std::vector<uint32_t> recordedCommandBufferGenerations;
  uint32_t nextRecordingContextIndex = 1u;
  uint32_t nextRecordedCommandBufferIndex = 1u;
  std::unique_ptr<GeometryPool> geometryPool;
  std::vector<TimingQueryPoolSlot> availableTimingQueryPools;
  std::optional<CurrentFrameTimingCapture> currentFrameTimingCapture;
  std::vector<PendingGpuTimingSubmission> pendingGpuTimingSubmissions;
  GpuTimingReport latestCompletedGpuTimingReport{};
  double gpuTimingTimestampPeriodToMs = 0.0;
  bool gpuTimingQueriesEnabled = false;
  bool loggedGpuTimingQueryWarning = false;
  bool loggedShadowSdsmTimingRecordDiagnostic = false;
  bool loggedShadowSdsmTimingSubmissionDiagnostic = false;
  bool loggedShadowSdsmTimingCollectionDiagnostic = false;
  bool loggedShadowSdsmTimingSubmissionWarning = false;
  bool loggedShadowSdsmTimingCollectionWarning = false;
};

struct PassTimingReservation {
  GpuTimingScope scope = GpuTimingScope::None;
  lvk::QueryPoolHandle pool{};
  uint32_t resetFirstQuery = kInvalidTimingQueryIndex;
  uint32_t resetQueryCount = 0u;
  uint32_t beginQuery = kInvalidTimingQueryIndex;
  uint32_t endQuery = kInvalidTimingQueryIndex;
};

namespace {

template <typename Impl>
using ActiveGraphicsContextSlotPtr =
    std::conditional_t<std::is_const_v<std::remove_reference_t<Impl>>,
                       const ActiveGraphicsRecordingContext *,
                       ActiveGraphicsRecordingContext *>;

template <typename Impl>
[[nodiscard]] ActiveGraphicsContextSlotPtr<Impl>
findActiveGraphicsContextSlot(Impl &impl, RecordingContextHandle handle) {
  if (handle.index >= impl.activeGraphicsContexts.size() ||
      handle.index >= impl.activeGraphicsContextOccupied.size() ||
      impl.activeGraphicsContextOccupied[handle.index] == 0u) {
    return nullptr;
  }
  auto &entry = impl.activeGraphicsContexts[handle.index];
  if (!areSameHandle(entry.handle, handle)) {
    return nullptr;
  }
  return &entry;
}

[[nodiscard]] bool
hasTimingReservation(const PassTimingReservation &reservation) noexcept {
  return reservation.pool.valid() &&
         reservation.beginQuery != kInvalidTimingQueryIndex &&
         reservation.endQuery != kInvalidTimingQueryIndex;
}

template <typename Impl>
void recycleTimingQueryPool(Impl &impl, TimingQueryPoolSlot &&pool) {
  if (pool.handle.valid()) {
    impl.availableTimingQueryPools.push_back(std::move(pool));
  }
}

template <typename Impl>
[[nodiscard]] Result<TimingQueryPoolSlot, std::string>
acquireTimingQueryPool(Impl &impl) {
  if (!impl.availableTimingQueryPools.empty()) {
    TimingQueryPoolSlot pool = std::move(impl.availableTimingQueryPools.back());
    impl.availableTimingQueryPools.pop_back();
    return Result<TimingQueryPoolSlot, std::string>::makeResult(
        std::move(pool));
  }
  if (!impl.context) {
    return Result<TimingQueryPoolSlot, std::string>::makeError(
        "LvkGPUDevice::acquireTimingQueryPool: context is null");
  }

  lvk::Result result;
  lvk::Holder<lvk::QueryPoolHandle> handle = impl.context->createQueryPool(
      kGpuTimingQueryPoolSize, "nuri_gpu_timing_queries", &result);
  if (!result.isOk() || !handle.valid()) {
    return Result<TimingQueryPoolSlot, std::string>::makeError(
        result.message != nullptr ? std::string(result.message)
                                  : std::string("failed to create query pool"));
  }

  return Result<TimingQueryPoolSlot, std::string>::makeResult(
      TimingQueryPoolSlot{.handle = std::move(handle)});
}

template <typename Impl>
PassTimingReservation
reservePassTimingReservationLocked(Impl &impl,
                                   ActiveGraphicsRecordingContext &context,
                                   const RenderPass &pass) {
  if (!impl.gpuTimingQueriesEnabled ||
      pass.gpuTimingScope == GpuTimingScope::None ||
      !impl.currentFrameTimingCapture.has_value() ||
      !impl.currentFrameTimingCapture->pool.handle.valid()) {
    return {};
  }
  if (context.timingQueryCursor + 2u > kGpuTimingQueriesPerContext) {
    if (!impl.loggedGpuTimingQueryWarning) {
      impl.loggedGpuTimingQueryWarning = true;
      NURI_LOG_WARNING("LvkGPUDevice: shadow GPU timing query capacity "
                       "exhausted for a recording context; dropping timing "
                       "for later shadow passes in that context");
    }
    return {};
  }

  PassTimingReservation reservation{};
  reservation.scope = pass.gpuTimingScope;
  reservation.pool = impl.currentFrameTimingCapture->pool.handle;
  if (!context.timingQuerySliceReset) {
    reservation.resetFirstQuery = context.timingQueryBase;
    reservation.resetQueryCount = kGpuTimingQueriesPerContext;
    context.timingQuerySliceReset = true;
  }
  reservation.beginQuery = context.timingQueryBase + context.timingQueryCursor;
  reservation.endQuery = reservation.beginQuery + 1u;
  context.timingQueryCursor += 2u;
  return reservation;
}

template <typename Impl> void collectCompletedGpuTimingSubmissions(Impl &impl) {
  if (!impl.context || impl.pendingGpuTimingSubmissions.empty()) {
    return;
  }

  size_t writeIndex = 0u;
  for (size_t readIndex = 0u;
       readIndex < impl.pendingGpuTimingSubmissions.size(); ++readIndex) {
    PendingGpuTimingSubmission &pending =
        impl.pendingGpuTimingSubmissions[readIndex];
    if (!impl.context->isReady(toLvkSubmitHandle(pending.submission))) {
      if (writeIndex != readIndex) {
        impl.pendingGpuTimingSubmissions[writeIndex] = std::move(pending);
      }
      ++writeIndex;
      continue;
    }

    GpuTimingReport completedReport{};
    completedReport.shadowSourceFrameIndex = pending.frameIndex;
    completedReport.sceneColorDownsampleSourceFrameIndex = pending.frameIndex;
    completedReport.transmissionSourceFrameIndex = pending.frameIndex;
    completedReport.temporalAAResolveSourceFrameIndex = pending.frameIndex;
    completedReport.temporalAADebugSourceFrameIndex = pending.frameIndex;
    completedReport.spatialAASourceFrameIndex = pending.frameIndex;
    completedReport.opaqueSourceFrameIndex = pending.frameIndex;
    completedReport.msaaResolveSourceFrameIndex = pending.frameIndex;
    bool hadShadowSdsmRange = false;
    double shadowTimeMs = 0.0;
    double shadowDepthTimeMs = 0.0;
    double shadowSdsmTimeMs = 0.0;
    double sceneColorDownsampleTimeMs = 0.0;
    double transmissionTimeMs = 0.0;
    double temporalAAResolveTimeMs = 0.0;
    double temporalAADebugTimeMs = 0.0;
    double spatialAATimeMs = 0.0;
    double opaqueTimeMs = 0.0;
    double msaaResolveTimeMs = 0.0;
    bool shadowTimingAvailable = false;
    bool shadowDepthTimingAvailable = false;
    bool shadowSdsmTimingAvailable = false;
    bool sceneColorDownsampleTimingAvailable = false;
    bool transmissionTimingAvailable = false;
    bool temporalAAResolveTimingAvailable = false;
    bool temporalAADebugTimingAvailable = false;
    bool spatialAATimingAvailable = false;
    bool opaqueTimingAvailable = false;
    bool msaaResolveTimingAvailable = false;
    std::vector<uint64_t> queryData;
    for (const PendingTimingQueryRange &range : pending.timingRanges) {
      hadShadowSdsmRange =
          hadShadowSdsmRange || range.scope == GpuTimingScope::ShadowSdsm;
      if (range.intervalCount == 0u || !pending.pool.handle.valid()) {
        continue;
      }
      queryData.assign(range.intervalCount * 2u, 0u);
      const bool queryResultOk = impl.context->getQueryPoolResults(
          pending.pool.handle, range.firstQuery, range.intervalCount * 2u,
          queryData.size() * sizeof(uint64_t), queryData.data(),
          sizeof(uint64_t));
      if (!queryResultOk) {
        continue;
      }
      for (uint32_t intervalIndex = 0u; intervalIndex < range.intervalCount;
           ++intervalIndex) {
        const uint64_t beginTicks = queryData[intervalIndex * 2u];
        const uint64_t endTicks = queryData[intervalIndex * 2u + 1u];
        if (endTicks < beginTicks) {
          continue;
        }
        const double intervalTimeMs =
            static_cast<double>(endTicks - beginTicks) *
            impl.gpuTimingTimestampPeriodToMs;
        switch (range.scope) {
        case GpuTimingScope::Shadow:
          shadowTimeMs += intervalTimeMs;
          shadowTimingAvailable = true;
          break;
        case GpuTimingScope::ShadowDepth:
          shadowDepthTimeMs += intervalTimeMs;
          shadowTimingAvailable = true;
          shadowDepthTimingAvailable = true;
          break;
        case GpuTimingScope::ShadowSdsm:
          shadowSdsmTimeMs += intervalTimeMs;
          shadowTimingAvailable = true;
          shadowSdsmTimingAvailable = true;
          break;
        case GpuTimingScope::SceneColorDownsample:
          sceneColorDownsampleTimeMs += intervalTimeMs;
          sceneColorDownsampleTimingAvailable = true;
          break;
        case GpuTimingScope::Transmission:
          transmissionTimeMs += intervalTimeMs;
          transmissionTimingAvailable = true;
          break;
        case GpuTimingScope::TemporalAAResolve:
          temporalAAResolveTimeMs += intervalTimeMs;
          temporalAAResolveTimingAvailable = true;
          break;
        case GpuTimingScope::TemporalAADebug:
          temporalAADebugTimeMs += intervalTimeMs;
          temporalAADebugTimingAvailable = true;
          break;
        case GpuTimingScope::SpatialAA:
          spatialAATimeMs += intervalTimeMs;
          spatialAATimingAvailable = true;
          break;
        case GpuTimingScope::Opaque:
          opaqueTimeMs += intervalTimeMs;
          opaqueTimingAvailable = true;
          break;
        case GpuTimingScope::MsaaResolve:
          msaaResolveTimeMs += intervalTimeMs;
          msaaResolveTimingAvailable = true;
          break;
        case GpuTimingScope::None:
          break;
        }
      }
    }

    if (shadowTimingAvailable) {
      completedReport.shadowTimeMs = static_cast<float>(shadowTimeMs);
      completedReport.availableScopeMask |=
          gpuTimingScopeToBit(GpuTimingScope::Shadow);
      completedReport.shadowSourceFrameIndex = pending.frameIndex;
    }
    if (shadowDepthTimingAvailable) {
      completedReport.shadowDepthTimeMs = static_cast<float>(shadowDepthTimeMs);
      completedReport.shadowDepthSourceFrameIndex = pending.frameIndex;
      completedReport.availableScopeMask |=
          gpuTimingScopeToBit(GpuTimingScope::ShadowDepth);
    }
    if (shadowSdsmTimingAvailable) {
      completedReport.shadowSdsmTimeMs = static_cast<float>(shadowSdsmTimeMs);
      completedReport.shadowSdsmSourceFrameIndex = pending.frameIndex;
      completedReport.availableScopeMask |=
          gpuTimingScopeToBit(GpuTimingScope::ShadowSdsm);
      if (!impl.loggedShadowSdsmTimingCollectionDiagnostic) {
        impl.loggedShadowSdsmTimingCollectionDiagnostic = true;
        NURI_LOG_INFO(
            "LvkGPUDevice: collected shadow SDSM timing result frame=%llu "
            "timeMs=%.3f",
            static_cast<unsigned long long>(pending.frameIndex),
            static_cast<float>(shadowSdsmTimeMs));
      }
    }
    if (sceneColorDownsampleTimingAvailable) {
      completedReport.sceneColorDownsampleTimeMs =
          static_cast<float>(sceneColorDownsampleTimeMs);
      completedReport.sceneColorDownsampleSourceFrameIndex = pending.frameIndex;
      completedReport.availableScopeMask |=
          gpuTimingScopeToBit(GpuTimingScope::SceneColorDownsample);
    }
    if (transmissionTimingAvailable) {
      completedReport.transmissionTimeMs =
          static_cast<float>(transmissionTimeMs);
      completedReport.transmissionSourceFrameIndex = pending.frameIndex;
      completedReport.availableScopeMask |=
          gpuTimingScopeToBit(GpuTimingScope::Transmission);
    }
    if (temporalAAResolveTimingAvailable) {
      completedReport.temporalAAResolveTimeMs =
          static_cast<float>(temporalAAResolveTimeMs);
      completedReport.temporalAAResolveSourceFrameIndex = pending.frameIndex;
      completedReport.availableScopeMask |=
          gpuTimingScopeToBit(GpuTimingScope::TemporalAAResolve);
    }
    if (temporalAADebugTimingAvailable) {
      completedReport.temporalAADebugTimeMs =
          static_cast<float>(temporalAADebugTimeMs);
      completedReport.temporalAADebugSourceFrameIndex = pending.frameIndex;
      completedReport.availableScopeMask |=
          gpuTimingScopeToBit(GpuTimingScope::TemporalAADebug);
    }
    if (spatialAATimingAvailable) {
      completedReport.spatialAATimeMs = static_cast<float>(spatialAATimeMs);
      completedReport.spatialAASourceFrameIndex = pending.frameIndex;
      completedReport.availableScopeMask |=
          gpuTimingScopeToBit(GpuTimingScope::SpatialAA);
    }
    if (opaqueTimingAvailable) {
      completedReport.opaqueTimeMs = static_cast<float>(opaqueTimeMs);
      completedReport.opaqueSourceFrameIndex = pending.frameIndex;
      completedReport.availableScopeMask |=
          gpuTimingScopeToBit(GpuTimingScope::Opaque);
    }
    if (msaaResolveTimingAvailable) {
      completedReport.msaaResolveTimeMs = static_cast<float>(msaaResolveTimeMs);
      completedReport.msaaResolveSourceFrameIndex = pending.frameIndex;
      completedReport.availableScopeMask |=
          gpuTimingScopeToBit(GpuTimingScope::MsaaResolve);
    }
    if (hadShadowSdsmRange && !shadowSdsmTimingAvailable &&
        !impl.loggedShadowSdsmTimingCollectionWarning) {
      impl.loggedShadowSdsmTimingCollectionWarning = true;
      NURI_LOG_WARNING(
          "LvkGPUDevice: shadow SDSM timing range completed without a "
          "collectable SDSM timing result (frame %llu)",
          static_cast<unsigned long long>(pending.frameIndex));
    }
    mergeGpuTimingReportScopes(impl.latestCompletedGpuTimingReport,
                               completedReport);
    recycleTimingQueryPool(impl, std::move(pending.pool));
  }

  impl.pendingGpuTimingSubmissions.resize(writeIndex);
}

} // namespace

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
  device->impl_->renderDocAttached = isRenderDocAttached();
  device->impl_->validationEnabled =
      resolveValidationEnabled(device->impl_->renderDocAttached);
  device->impl_->verboseVkDiagnostics =
      resolveVerboseVkDiagnostics(device->impl_->renderDocAttached);

  int32_t width = 0;
  int32_t height = 0;
  window.getFramebufferSize(width, height);
  if (!width || !height) {
    width = 1;
    height = 1;
  }

  lvk::ContextConfig config{};
  config.enableValidation = device->impl_->validationEnabled;
  config.terminateOnValidationError = resolveTerminateOnValidationError();

  NURI_LOG_INFO("LvkGPUDevice::create: RenderDoc attached=%s validation=%s "
                "diagnostics=%s terminateOnValidationError=%s",
                boolToString(device->impl_->renderDocAttached),
                boolToString(config.enableValidation),
                boolToString(device->impl_->verboseVkDiagnostics),
                boolToString(config.terminateOnValidationError));
  if (device->impl_->renderDocAttached && config.enableValidation) {
    NURI_LOG_INFO(
        "LvkGPUDevice::create: RenderDoc suppresses validation output when "
        "DebugOutputMute is enabled in capture options");
  }

  device->impl_->context = lvk::createVulkanContextWithSwapchain(
      static_cast<lvk::LVKwindow *>(window.nativeHandle()),
      static_cast<uint32_t>(width), static_cast<uint32_t>(height), config);

  if (!device->impl_->context) {
    NURI_LOG_WARNING("LvkGPUDevice::create: Failed to create Vulkan context "
                     "with swapchain (%d x %d)",
                     width, height);
    return nullptr;
  }

#if NURI_LVK_HAS_VULKAN_COMMAND_BUFFER
  if (auto *vkContext =
          dynamic_cast<lvk::VulkanContext *>(device->impl_->context.get());
      vkContext != nullptr) {
    VkPhysicalDeviceFeatures deviceFeatures{};
    vkGetPhysicalDeviceFeatures(vkContext->getVkPhysicalDevice(),
                                &deviceFeatures);
    const VkPhysicalDeviceProperties &properties =
        vkContext->getVkPhysicalDeviceProperties();
    NURI_LOG_INFO(
        "LvkGPUDevice::create: Vulkan device='%s' vendor=0x%04x device=0x%04x "
        "api=%s driver='%s' driverInfo='%s'",
        properties.deviceName, properties.vendorID, properties.deviceID,
        formatVkVersion(properties.apiVersion).c_str(),
        vkContext->vkPhysicalDeviceDriverProperties_.driverName,
        vkContext->vkPhysicalDeviceDriverProperties_.driverInfo);
    NURI_LOG_INFO("LvkGPUDevice::create: swapchain images=%u",
                  device->impl_->context->getNumSwapchainImages());
    device->impl_->compressionCaps.bc7 =
        deviceFeatures.textureCompressionBC == VK_TRUE;
    device->impl_->compressionCaps.etc2 =
        deviceFeatures.textureCompressionETC2 == VK_TRUE;
    device->impl_->compressionCaps.astc =
        deviceFeatures.textureCompressionASTC_LDR == VK_TRUE;
    if (deviceFeatures.samplerAnisotropy == VK_TRUE &&
        properties.limits.maxSamplerAnisotropy > 1.0f) {
      device->impl_->maxSamplerAnisotropy = static_cast<uint8_t>(std::clamp(
          static_cast<uint32_t>(
              std::floor(properties.limits.maxSamplerAnisotropy)),
          1u, static_cast<uint32_t>(std::numeric_limits<uint8_t>::max())));
    }
  }
#endif

  if (device->impl_->context) {
    device->impl_->gpuTimingTimestampPeriodToMs =
        device->impl_->context->getTimestampPeriodToMs();
    device->impl_->gpuTimingQueriesEnabled =
        device->impl_->gpuTimingTimestampPeriodToMs > 0.0;
  }

  {
    const auto createBuiltinSampler =
        [&device](
            const SamplerDesc &desc,
            std::string_view debugName) -> Result<SamplerHandle, std::string> {
      auto samplerResult = device->createSampler(desc, debugName);
      if (samplerResult.hasError()) {
        NURI_LOG_ERROR(
            "LvkGPUDevice::create: Failed to create sampler '%.*s': %s",
            static_cast<int>(debugName.size()), debugName.data(),
            samplerResult.error().c_str());
        return Result<SamplerHandle, std::string>::makeError(
            samplerResult.error());
      }
      return samplerResult;
    };

    auto bilinearSampler = createBuiltinSampler(
        SamplerDesc{
            .minFilter = SamplerFilter::Linear,
            .magFilter = SamplerFilter::Linear,
            .mipMode = SamplerMipMode::Disabled,
            .wrapU = SamplerWrapMode::Repeat,
            .wrapV = SamplerWrapMode::Repeat,
            .wrapW = SamplerWrapMode::Repeat,
        },
        "nuri_sampler_bilinear");
    if (bilinearSampler.hasError()) {
      return nullptr;
    }
    device->impl_->bilinearSampler = bilinearSampler.value();

    auto trilinearSampler = createBuiltinSampler(
        SamplerDesc{
            .minFilter = SamplerFilter::Linear,
            .magFilter = SamplerFilter::Linear,
            .mipMode = SamplerMipMode::Linear,
            .wrapU = SamplerWrapMode::Repeat,
            .wrapV = SamplerWrapMode::Repeat,
            .wrapW = SamplerWrapMode::Repeat,
        },
        "nuri_sampler_trilinear");
    if (trilinearSampler.hasError()) {
      return nullptr;
    }
    device->impl_->trilinearSampler = trilinearSampler.value();

    for (size_t i = 0; i < kSupportedAnisotropyLevels.size(); ++i) {
      const uint8_t requested = kSupportedAnisotropyLevels[i];
      if (device->impl_->maxSamplerAnisotropy < requested) {
        continue;
      }
      auto anisotropicSampler = createBuiltinSampler(
          SamplerDesc{
              .minFilter = SamplerFilter::Linear,
              .magFilter = SamplerFilter::Linear,
              .mipMode = SamplerMipMode::Linear,
              .wrapU = SamplerWrapMode::Repeat,
              .wrapV = SamplerWrapMode::Repeat,
              .wrapW = SamplerWrapMode::Repeat,
              .maxAnisotropy = requested,
          },
          std::format("nuri_sampler_aniso_{}x", requested));
      if (anisotropicSampler.hasError()) {
        return nullptr;
      }
      device->impl_->anisotropicSamplers[i] = anisotropicSampler.value();
    }

    auto cubemapSampler = createBuiltinSampler(
        SamplerDesc{
            .minFilter = SamplerFilter::Linear,
            .magFilter = SamplerFilter::Linear,
            .mipMode = SamplerMipMode::Linear,
            .wrapU = SamplerWrapMode::Clamp,
            .wrapV = SamplerWrapMode::Clamp,
            .wrapW = SamplerWrapMode::Clamp,
        },
        "nuri_cubemap_sampler");
    if (cubemapSampler.hasError()) {
      return nullptr;
    }
    device->impl_->cubemapSampler = cubemapSampler.value();
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

void LvkGPUDevice::getWindowSize(int32_t &outWidth, int32_t &outHeight) const {
  if (!impl_->window) {
    outWidth = 0;
    outHeight = 0;
    return;
  }
  impl_->window->getWindowSize(outWidth, outHeight);
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
  {
    std::lock_guard immediateLock(impl_->contextImmediateMutex);
    impl_->currentFrameSwapchainTexture = {};
  }
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
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_WAIT);
  {
    std::lock_guard immediateLock(impl_->contextImmediateMutex);
    impl_->currentFrameIndex = frameIndex;
    impl_->currentFrameSwapchainTexture = {};
    impl_->hasPreparedSwapchainImage = false;
    if (impl_->currentFrameTimingCapture.has_value()) {
      recycleTimingQueryPool(*impl_,
                             std::move(impl_->currentFrameTimingCapture->pool));
      impl_->currentFrameTimingCapture.reset();
    }
    collectCompletedGpuTimingSubmissions(*impl_);
    if (impl_->gpuTimingQueriesEnabled) {
      auto timingPoolResult = acquireTimingQueryPool(*impl_);
      if (timingPoolResult.hasError()) {
        if (!impl_->loggedGpuTimingQueryWarning) {
          impl_->loggedGpuTimingQueryWarning = true;
          NURI_LOG_WARNING("LvkGPUDevice::beginFrame: disabling GPU timing "
                           "queries: %s",
                           timingPoolResult.error().c_str());
        }
        impl_->gpuTimingQueriesEnabled = false;
      } else {
        TimingQueryPoolSlot timingPool = std::move(timingPoolResult.value());
        impl_->currentFrameTimingCapture.emplace(CurrentFrameTimingCapture{
            .pool = std::move(timingPool),
            .frameIndex = frameIndex,
        });
      }
    }
  }
  if (!impl_->geometryPool) {
    return Result<bool, std::string>::makeResult(true);
  }
  return impl_->geometryPool->beginFrame(frameIndex);
}

Result<bool, std::string> LvkGPUDevice::prepareFrameOutput() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_WAIT);
  if (!impl_ || !impl_->context) {
    return Result<bool, std::string>::makeError(
        "LvkGPUDevice::prepareFrameOutput: context is null");
  }
  {
    std::lock_guard immediateLock(impl_->contextImmediateMutex);
    if (impl_->currentFrameSwapchainTexture.valid()) {
      return Result<bool, std::string>::makeResult(true);
    }
  }

  lvk::TextureHandle swapchainTexture{};
  {
    NURI_PROFILER_ZONE("LvkGPUDevice.acquire_swapchain_texture",
                       NURI_PROFILER_COLOR_WAIT);
    swapchainTexture = impl_->context->getCurrentSwapchainTexture();
    NURI_PROFILER_ZONE_END();
  }

  std::lock_guard immediateLock(impl_->contextImmediateMutex);
  if (!impl_->currentFrameSwapchainTexture.valid()) {
    impl_->currentFrameSwapchainTexture = swapchainTexture;
    impl_->preparedSwapchainImageIndex =
        impl_->context->getSwapchainCurrentImageIndex();
    impl_->preparedSwapchainImageCount =
        impl_->context->getNumSwapchainImages();
    impl_->hasPreparedSwapchainImage = swapchainTexture.valid();
  }
  if (!impl_->currentFrameSwapchainTexture.valid()) {
    return Result<bool, std::string>::makeError(
        "LvkGPUDevice::prepareFrameOutput: failed to acquire swapchain "
        "texture");
  }
  return Result<bool, std::string>::makeResult(true);
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
  const uint8_t lvkUsage = toLvkBufferUsage(desc.usage);
  if (lvkUsage == 0) {
    NURI_LOG_WARNING("LvkGPUDevice::createBuffer: Buffer usage is empty");
    return Result<BufferHandle, std::string>::makeError(
        "Buffer usage is empty");
  }

  lvk::BufferDesc bufferDesc{
      .usage = lvkUsage,
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

Result<SamplerHandle, std::string>
LvkGPUDevice::createSampler(const SamplerDesc &desc,
                            std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  std::string debugNameStorage(debugName);
  const char *debugNameCStr =
      debugNameStorage.empty() ? "" : debugNameStorage.c_str();

  lvk::SamplerStateDesc samplerDesc{};
  samplerDesc.minFilter = toLvkSamplerFilter(desc.minFilter);
  samplerDesc.magFilter = toLvkSamplerFilter(desc.magFilter);
  samplerDesc.mipMap = toLvkSamplerMipMode(desc.mipMode);
  samplerDesc.wrapU = toLvkSamplerWrapMode(desc.wrapU);
  samplerDesc.wrapV = toLvkSamplerWrapMode(desc.wrapV);
  samplerDesc.wrapW = toLvkSamplerWrapMode(desc.wrapW);
  samplerDesc.depthCompareOp = toLvkCompareOp(desc.depthCompareOp);
  samplerDesc.mipLodMin = desc.mipLodMin;
  samplerDesc.mipLodMax = std::max(desc.mipLodMax, desc.mipLodMin);
  samplerDesc.maxAnisotropic =
      std::min(desc.maxAnisotropy, impl_->maxSamplerAnisotropy);
  if (samplerDesc.maxAnisotropic < 1u) {
    samplerDesc.maxAnisotropic = 1u;
  }
  samplerDesc.depthCompareEnabled = desc.depthCompareEnabled;
  samplerDesc.debugName = debugNameCStr;

  lvk::Result result;
  lvk::Holder<lvk::SamplerHandle> handle =
      impl_->context->createSampler(samplerDesc, &result);
  if (!result.isOk() || !handle.valid()) {
    if (debugNameCStr[0] != '\0') {
      NURI_LOG_WARNING(
          "LvkGPUDevice::createSampler: Failed to create sampler '%s': %s",
          debugNameCStr, result.message ? result.message : "unknown error");
    } else {
      NURI_LOG_WARNING(
          "LvkGPUDevice::createSampler: Failed to create sampler: %s",
          result.message ? result.message : "unknown error");
    }
    return Result<SamplerHandle, std::string>::makeError(
        result.message != nullptr ? std::string(result.message)
                                  : std::string("Failed to create sampler"));
  }

  SamplerHandle nuriHandle =
      impl_->samplers.allocate(std::move(handle), std::move(debugNameStorage));
  return Result<SamplerHandle, std::string>::makeResult(nuriHandle);
}

Result<TextureHandle, std::string> LvkGPUDevice::createDepthBuffer() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  TextureDesc desc{
      .type = TextureType::Texture2D,
      .format = Format::D32_FLOAT,
      .dimensions = {1, 1, 1},
      .usage = TextureUsage::AttachmentSampled,
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
  if (desc.colorAttachmentCount > desc.colorFormats.size()) {
    NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: color attachment "
                     "count exceeds provided color formats");
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Color attachment count exceeds provided color formats");
  }
  if (desc.colorAttachmentCount > lvk::LVK_MAX_COLOR_ATTACHMENTS) {
    NURI_LOG_WARNING("LvkGPUDevice::createRenderPipeline: color attachment "
                     "count exceeds LVK color attachment slots");
    return Result<RenderPipelineHandle, std::string>::makeError(
        "Color attachment count exceeds LVK color attachment slots");
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
      .depthFormat = toLvkFormat(desc.depthFormat),
      .cullMode = toLvkCullMode(desc.cullMode),
      .polygonMode = toLvkPolygonMode(desc.polygonMode),
      .samplesCount = std::max(desc.numSamples, 1u),
      .patchControlPoints = desc.patchControlPoints,
      .minSampleShading = desc.minSampleShading,
      .alphaToCoverageEnabled = desc.alphaToCoverageEnabled,
      .debugName = reserved.debugNameCStr,
  };
  for (uint32_t i = 0u; i < desc.colorAttachmentCount; ++i) {
    pipelineDesc.color[i] = {
        .format = toLvkFormat(desc.colorFormats[i]),
        .blendEnabled = desc.blendEnabled,
        .srcRGBBlendFactor = desc.blendEnabled ? lvk::BlendFactor_SrcAlpha
                                               : lvk::BlendFactor_One,
        .srcAlphaBlendFactor = lvk::BlendFactor_One,
        .dstRGBBlendFactor = desc.blendEnabled
                                 ? lvk::BlendFactor_OneMinusSrcAlpha
                                 : lvk::BlendFactor_Zero,
        .dstAlphaBlendFactor = desc.blendEnabled
                                   ? lvk::BlendFactor_OneMinusSrcAlpha
                                   : lvk::BlendFactor_Zero};
  }

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
                       return areSameHandle(entry.handle, texture);
                     }),
      impl_->framebufferTextures.end());
  impl_->textures.deallocate(texture);
}

void LvkGPUDevice::destroySampler(SamplerHandle sampler) {
  if (!impl_) {
    return;
  }
  impl_->samplers.deallocate(sampler);
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

bool LvkGPUDevice::isValid(SamplerHandle h) const {
  return impl_->samplers.isValid(h);
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

TextureDimensions LvkGPUDevice::getTextureDimensions(TextureHandle h) const {
  if (!impl_ || !impl_->textures.isValid(h)) {
    return TextureDimensions{};
  }
  const lvk::TextureHandle texture = impl_->textures.getLvkHandle(h);
  if (!texture.valid()) {
    return TextureDimensions{};
  }
  const lvk::Dimensions dimensions = impl_->context->getDimensions(texture);
  return TextureDimensions{.width = dimensions.width,
                           .height = dimensions.height,
                           .depth = dimensions.depth};
}

TextureCompressionCaps LvkGPUDevice::getTextureCompressionCaps() const {
  return impl_->compressionCaps;
}

uint32_t LvkGPUDevice::getTextureBindlessIndex(TextureHandle h) const {
  if (!impl_->textures.isValid(h)) {
    // Must not return 0: that is a valid bindless heap slot.  Shaders compare
    // against kInvalidTextureBindlessIndex (UINT32_MAX); returning 0 made
    // invalid handles look "valid" and sample the wrong texture (often black
    // in large scenes where slot 0 is not the intended resource).
    return kInvalidTextureBindlessIndex;
  }
  return impl_->textures.getLvkHandle(h).index();
}

uint32_t LvkGPUDevice::getSamplerBindlessIndex(SamplerHandle h) const {
  if (!impl_->samplers.isValid(h)) {
    return 0u;
  }
  return impl_->samplers.getLvkHandle(h).index();
}

uint8_t LvkGPUDevice::getMaxSamplerAnisotropy() const {
  return impl_->maxSamplerAnisotropy;
}

uint32_t
LvkGPUDevice::getLinearRepeatSamplerBindlessIndex(bool useMipmaps,
                                                  uint8_t maxAnisotropy) const {
  if (!useMipmaps) {
    return getSamplerBindlessIndex(impl_->bilinearSampler);
  }
  if (maxAnisotropy <= 1u || impl_->maxSamplerAnisotropy <= 1u) {
    return getSamplerBindlessIndex(impl_->trilinearSampler);
  }

  const uint8_t clamped = std::min(maxAnisotropy, impl_->maxSamplerAnisotropy);
  if (const SamplerHandle handle =
          findBestAnisotropicSampler(impl_->anisotropicSamplers, clamped);
      isValid(handle)) {
    return getSamplerBindlessIndex(handle);
  }
  return getSamplerBindlessIndex(impl_->trilinearSampler);
}

uint32_t LvkGPUDevice::getDefaultSamplerBindlessIndex() const { return 0u; }

uint32_t LvkGPUDevice::getCubemapSamplerBindlessIndex() const {
  return getSamplerBindlessIndex(impl_->cubemapSampler);
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

uint64_t LvkGPUDevice::geometryMutationVersion() const {
  return impl_->geometryPool ? impl_->geometryPool->mutationVersion() : 0;
}

GpuTimingReport LvkGPUDevice::getLatestCompletedGpuTimingReport() const {
  if (!impl_) {
    return {};
  }
  std::lock_guard immediateLock(impl_->contextImmediateMutex);
  return impl_->latestCompletedGpuTimingReport;
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

Result<bool, std::string> LvkGPUDevice::recordRenderPasses(
    lvk::ICommandBuffer &commandBuffer, std::span<const RenderPass> passes,
    std::span<const PassTimingReservation> passTimings) {
  static_assert(kMaxDependencyResources <=
                lvk::Dependencies::LVK_MAX_SUBMIT_DEPENDENCIES);
  if (passes.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  lvk::TextureHandle swapchainTexture{};
  {
    // beginFrame() and resizeSwapchain() reset currentFrameSwapchainTexture
    // while holding contextImmediateMutex, so copy it under the same lock.
    std::lock_guard immediateLock(impl_->contextImmediateMutex);
    swapchainTexture = impl_->currentFrameSwapchainTexture;
  }
  const auto fillDependencies =
      [this](std::span<const BufferHandle> dependencyBuffers,
             std::span<const TextureHandle> dependencyTextures,
             lvk::Dependencies &deps,
             std::string_view context) -> Result<bool, std::string> {
    if (dependencyBuffers.size() > kMaxDependencyResources ||
        dependencyTextures.size() > kMaxDependencyResources) {
      return makeDependencyCountExceededError(context);
    }

    size_t bufferDstIndex = 0;
    for (const BufferHandle bufferHandle : dependencyBuffers) {
      if (!nuri::isValid(bufferHandle)) {
        continue;
      }
      if (!impl_->buffers.isValid(bufferHandle)) {
        return makeDependencyError(context, "dependency buffer is invalid");
      }
      deps.buffers[bufferDstIndex++] =
          impl_->buffers.getLvkHandle(bufferHandle);
    }

    size_t textureDstIndex = 0;
    for (const TextureHandle textureHandle : dependencyTextures) {
      if (!nuri::isValid(textureHandle)) {
        continue;
      }
      if (!impl_->textures.isValid(textureHandle)) {
        return makeDependencyError(context, "dependency texture is invalid");
      }
      deps.textures[textureDstIndex++] =
          impl_->textures.getLvkHandle(textureHandle);
    }
    return Result<bool, std::string>::makeResult(true);
  };
  const bool supportsIndexedIndirectCount =
      impl_->context->supportsDrawIndexedIndirectCount();

  for (size_t passIndex = 0u; passIndex < passes.size(); ++passIndex) {
    const RenderPass &pass = passes[passIndex];
    const PassTimingReservation timingReservation =
        passIndex < passTimings.size() ? passTimings[passIndex]
                                       : PassTimingReservation{};
    const bool passLabelPushed =
        pushDebugLabel(commandBuffer, pass.debugLabel, pass.debugColor);
    const auto returnPassError =
        [&](std::string_view message) -> Result<bool, std::string> {
      if (passLabelPushed) {
        commandBuffer.cmdPopDebugGroupLabel();
      }
      return Result<bool, std::string>::makeError(std::string(message));
    };
    const auto returnPassErrorResult =
        [&](Result<bool, std::string> result) -> Result<bool, std::string> {
      if (passLabelPushed) {
        commandBuffer.cmdPopDebugGroupLabel();
      }
      return result;
    };

    lvk::RenderPass renderPass{};
    lvk::Framebuffer framebuffer{};
    lvk::TextureHandle colorTexture{};
    lvk::TextureHandle colorResolveTexture{};
    lvk::TextureHandle depthTexture{};
    lvk::TextureHandle depthResolveTexture{};
    lvk::TextureHandle viewportTexture{};
    const bool computeOnly =
        pass.executionMode == RenderPassExecutionMode::ComputeOnly;
    if (computeOnly) {
      if (pass.hasColorAttachment || nuri::isValid(pass.colorTexture) ||
          nuri::isValid(pass.colorResolveTexture) ||
          nuri::isValid(pass.depthTexture) ||
          nuri::isValid(pass.depthResolveTexture) || !pass.draws.empty()) {
        return returnPassError(
            "LvkGPUDevice::recordGraphicsPass: invalid compute-only pass "
            "attachments or draws");
      }
      renderPass.color[0].loadOp = lvk::LoadOp_Invalid;
      renderPass.depth.loadOp = lvk::LoadOp_Invalid;
    } else if (pass.hasColorAttachment) {
      renderPass.color[0] = {
          .loadOp = toLvkLoadOp(pass.color.loadOp),
          .storeOp = toLvkStoreOp(pass.color.storeOp),
          .resolveMode = toLvkResolveMode(pass.color.resolveMode),
          .clearColor = {pass.color.clearColor.r, pass.color.clearColor.g,
                         pass.color.clearColor.b, pass.color.clearColor.a},
      };

      if (nuri::isValid(pass.colorTexture)) {
        if (!impl_->textures.isValid(pass.colorTexture)) {
          return returnPassError(
              "LvkGPUDevice::recordGraphicsPass: invalid pass color texture "
              "handle");
        }
        colorTexture = impl_->textures.getLvkHandle(pass.colorTexture);
        if (!colorTexture.valid()) {
          return returnPassError(
              "LvkGPUDevice::recordGraphicsPass: invalid LVK pass color "
              "texture handle");
        }
      } else {
        colorTexture = swapchainTexture;
        if (!colorTexture.valid()) {
          return returnPassError(
              "LvkGPUDevice::recordGraphicsPass: invalid swapchain texture");
        }
      }

      if (nuri::isValid(pass.colorResolveTexture)) {
        if (!impl_->textures.isValid(pass.colorResolveTexture)) {
          return returnPassError(
              "LvkGPUDevice::recordGraphicsPass: invalid pass color resolve "
              "texture handle");
        }
        colorResolveTexture =
            impl_->textures.getLvkHandle(pass.colorResolveTexture);
        if (!colorResolveTexture.valid()) {
          return returnPassError(
              "LvkGPUDevice::recordGraphicsPass: invalid LVK pass color "
              "resolve texture handle");
        }
      }

      framebuffer.color[0] = {.texture = colorTexture,
                              .resolveTexture = colorResolveTexture};
      viewportTexture = colorTexture;
    } else {
      renderPass.color[0].loadOp = lvk::LoadOp_Invalid;
      if (nuri::isValid(pass.colorTexture)) {
        return returnPassError(
            "LvkGPUDevice::recordGraphicsPass: no-color pass has color "
            "texture");
      }
      if (nuri::isValid(pass.colorResolveTexture)) {
        return returnPassError(
            "LvkGPUDevice::recordGraphicsPass: no-color pass has color "
            "resolve texture");
      }
    }

    if (!computeOnly && nuri::isValid(pass.depthTexture)) {
      if (!impl_->textures.isValid(pass.depthTexture)) {
        return returnPassError(
            "LvkGPUDevice::recordGraphicsPass: invalid pass depth texture "
            "handle");
      }
      depthTexture = impl_->textures.getLvkHandle(pass.depthTexture);
      if (!depthTexture.valid()) {
        return returnPassError(
            "LvkGPUDevice::recordGraphicsPass: invalid LVK pass depth texture "
            "handle");
      }
      renderPass.depth = {
          .loadOp = toLvkLoadOp(pass.depth.loadOp),
          .storeOp = toLvkStoreOp(pass.depth.storeOp),
          .resolveMode = toLvkResolveMode(pass.depth.resolveMode),
          .clearDepth = pass.depth.clearDepth,
          .clearStencil = pass.depth.clearStencil,
      };
      if (nuri::isValid(pass.depthResolveTexture)) {
        if (!impl_->textures.isValid(pass.depthResolveTexture)) {
          return returnPassError(
              "LvkGPUDevice::recordGraphicsPass: invalid pass depth resolve "
              "texture handle");
        }
        depthResolveTexture =
            impl_->textures.getLvkHandle(pass.depthResolveTexture);
        if (!depthResolveTexture.valid()) {
          return returnPassError(
              "LvkGPUDevice::recordGraphicsPass: invalid LVK pass depth "
              "resolve texture handle");
        }
      }
      framebuffer.depthStencil = {.texture = depthTexture,
                                  .resolveTexture = depthResolveTexture};
      if (!viewportTexture.valid()) {
        viewportTexture = depthTexture;
      }
    } else if (!computeOnly) {
      if (nuri::isValid(pass.depthResolveTexture)) {
        return returnPassError(
            "LvkGPUDevice::recordGraphicsPass: depth resolve texture requires "
            "a depth texture");
      }
      renderPass.depth.loadOp = lvk::LoadOp_Invalid;
    }

    if (hasTimingReservation(timingReservation)) {
      if (timingReservation.resetQueryCount > 0u) {
        commandBuffer.cmdResetQueryPool(timingReservation.pool,
                                        timingReservation.resetFirstQuery,
                                        timingReservation.resetQueryCount);
      }
      commandBuffer.cmdWriteTimestamp(timingReservation.pool,
                                      timingReservation.beginQuery);
    }

    {
      bool computePipelineBound = false;
      ComputePipelineHandle boundComputePipeline{};
      for (const ComputeDispatchItem &dispatch : pass.preDispatches) {
        NURI_PROFILER_ZONE("LvkGPUDevice.compute_dispatch_submission",
                           NURI_PROFILER_COLOR_CMD_DISPATCH);
        if (!impl_->computePipelines.isValid(dispatch.pipeline)) {
          return returnPassError("Invalid compute pipeline handle");
        }
        if (dispatch.dispatch.x == 0 || dispatch.dispatch.y == 0 ||
            dispatch.dispatch.z == 0) {
          return returnPassError("Invalid compute dispatch size");
        }

        lvk::Dependencies dispatchDependencies{};
        auto dispatchDepsResult = fillDependencies(
            dispatch.dependencyBuffers, dispatch.dependencyTextures,
            dispatchDependencies,
            "LvkGPUDevice::recordGraphicsPass compute dispatch");
        if (dispatchDepsResult.hasError()) {
          return returnPassErrorResult(dispatchDepsResult);
        }

        const bool dispatchLabelPushed = pushDebugLabel(
            commandBuffer, dispatch.debugLabel, dispatch.debugColor);

        if (!computePipelineBound ||
            !areSameHandle(dispatch.pipeline, boundComputePipeline)) {
          commandBuffer.cmdBindComputePipeline(
              impl_->computePipelines.getLvkHandle(dispatch.pipeline));
          boundComputePipeline = dispatch.pipeline;
          computePipelineBound = true;
        }
        if (!dispatch.pushConstants.empty()) {
          commandBuffer.cmdPushConstants(
              static_cast<const void *>(dispatch.pushConstants.data()),
              dispatch.pushConstants.size(), 0);
        }
        commandBuffer.cmdDispatchThreadGroups({.width = dispatch.dispatch.x,
                                               .height = dispatch.dispatch.y,
                                               .depth = dispatch.dispatch.z},
                                              dispatchDependencies);

        if (dispatchLabelPushed) {
          commandBuffer.cmdPopDebugGroupLabel();
        }
      }
      NURI_PROFILER_ZONE_END();
    }

    if (computeOnly) {
      if (hasTimingReservation(timingReservation)) {
        commandBuffer.cmdWriteTimestamp(timingReservation.pool,
                                        timingReservation.endQuery);
      }
      if (passLabelPushed) {
        commandBuffer.cmdPopDebugGroupLabel();
      }
      continue;
    }

    lvk::Dependencies renderDependencies{};
    auto renderDepsResult =
        fillDependencies(pass.dependencyBuffers, {}, renderDependencies,
                         "LvkGPUDevice::recordGraphicsPass render pass");
    if (renderDepsResult.hasError()) {
      return returnPassErrorResult(renderDepsResult);
    }

    commandBuffer.cmdBeginRendering(renderPass, framebuffer,
                                    renderDependencies);
    const auto returnDrawError =
        [&](std::string_view message,
            bool drawLabelPushed) -> Result<bool, std::string> {
      commandBuffer.cmdEndRendering();
      if (drawLabelPushed) {
        commandBuffer.cmdPopDebugGroupLabel();
      }
      if (passLabelPushed) {
        commandBuffer.cmdPopDebugGroupLabel();
      }
      return Result<bool, std::string>::makeError(std::string(message));
    };

    Viewport vp{};
    if (pass.useViewport) {
      vp = pass.viewport;
    } else {
      if (!viewportTexture.valid()) {
        return returnPassError(
            "LvkGPUDevice::recordGraphicsPass: pass has no framebuffer "
            "attachment for viewport inference");
      }
      const lvk::Dimensions dim =
          impl_->context->getDimensions(viewportTexture);
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
    const lvk::ScissorRect viewportScissor = {
        static_cast<uint32_t>(vp.x),
        static_cast<uint32_t>(vp.y),
        static_cast<uint32_t>(vp.width),
        static_cast<uint32_t>(vp.height),
    };
    commandBuffer.cmdBindScissorRect(viewportScissor);

    bool scissorMatchesViewport = true;
    bool pipelineBound = false;
    RenderPipelineHandle boundPipeline{};
    bool vertexBindingBound = false;
    BufferHandle boundVertexBuffer{};
    uint64_t boundVertexBufferOffset = 0;
    bool indexBindingBound = false;
    BufferHandle boundIndexBuffer{};
    uint64_t boundIndexBufferOffset = 0;
    IndexFormat boundIndexFormat = IndexFormat::U32;
    bool depthStateBound = false;
    DepthState boundDepthState{};
    bool depthBiasEnableKnown = false;
    bool depthBiasEnabled = false;
    bool depthBiasParamsKnown = false;
    float boundDepthBiasConstant = 0.0f;
    float boundDepthBiasSlope = 0.0f;
    float boundDepthBiasClamp = 0.0f;

    for (const DrawItem &draw : pass.draws) {
      const bool drawLabelPushed =
          kEnablePerDrawDebugLabels &&
          pushDebugLabel(commandBuffer, draw.debugLabel, draw.debugColor);

      if (!pipelineBound || !areSameHandle(draw.pipeline, boundPipeline)) {
        if (!impl_->renderPipelines.isValid(draw.pipeline)) {
          return returnDrawError("Invalid render pipeline handle",
                                 drawLabelPushed);
        }
        commandBuffer.cmdBindRenderPipeline(
            impl_->renderPipelines.getLvkHandle(draw.pipeline));
        boundPipeline = draw.pipeline;
        pipelineBound = true;
      }

      if (nuri::isValid(draw.vertexBuffer)) {
        if (!impl_->buffers.isValid(draw.vertexBuffer)) {
          return returnDrawError("Vertex buffer is invalid", drawLabelPushed);
        }
        const bool vertexBindingChanged =
            !vertexBindingBound ||
            !areSameHandle(draw.vertexBuffer, boundVertexBuffer) ||
            boundVertexBufferOffset != draw.vertexBufferOffset;
        if (vertexBindingChanged) {
          commandBuffer.cmdBindVertexBuffer(
              0, impl_->buffers.getLvkHandle(draw.vertexBuffer),
              draw.vertexBufferOffset);
          boundVertexBuffer = draw.vertexBuffer;
          boundVertexBufferOffset = draw.vertexBufferOffset;
          vertexBindingBound = true;
        }
      }

      const bool isDirectDraw = draw.command == DrawCommandType::Direct;
      const bool isIndexedIndirectDraw =
          draw.command == DrawCommandType::IndexedIndirect ||
          draw.command == DrawCommandType::IndexedIndirectCount;
      const bool requiresIndexBuffer =
          isIndexedIndirectDraw || (isDirectDraw && draw.indexCount > 0);
      if (requiresIndexBuffer) {
        const bool indexBindingChanged =
            !indexBindingBound ||
            !areSameHandle(draw.indexBuffer, boundIndexBuffer) ||
            boundIndexBufferOffset != draw.indexBufferOffset ||
            boundIndexFormat != draw.indexFormat;
        if (indexBindingChanged) {
          if (!impl_->buffers.isValid(draw.indexBuffer)) {
            return returnDrawError("Index buffer is invalid", drawLabelPushed);
          }
          commandBuffer.cmdBindIndexBuffer(
              impl_->buffers.getLvkHandle(draw.indexBuffer),
              toLvkIndexFormat(draw.indexFormat), draw.indexBufferOffset);
          boundIndexBuffer = draw.indexBuffer;
          boundIndexBufferOffset = draw.indexBufferOffset;
          boundIndexFormat = draw.indexFormat;
          indexBindingBound = true;
        }
      }

      if (draw.useDepthState) {
        const bool depthStateChanged =
            !depthStateBound ||
            boundDepthState.compareOp != draw.depthState.compareOp ||
            boundDepthState.isDepthWriteEnabled !=
                draw.depthState.isDepthWriteEnabled;
        if (depthStateChanged) {
          lvk::DepthState depthState{
              .compareOp = toLvkCompareOp(draw.depthState.compareOp),
              .isDepthWriteEnabled = draw.depthState.isDepthWriteEnabled,
          };
          commandBuffer.cmdBindDepthState(depthState);
          boundDepthState = draw.depthState;
          depthStateBound = true;
        }
      }

      const bool depthBiasEnableChanged =
          !depthBiasEnableKnown || depthBiasEnabled != draw.depthBiasEnable;
      if (depthBiasEnableChanged) {
        commandBuffer.cmdSetDepthBiasEnable(draw.depthBiasEnable);
        depthBiasEnableKnown = true;
        depthBiasEnabled = draw.depthBiasEnable;
      }
      if (draw.depthBiasEnable) {
        const bool depthBiasParamsChanged =
            !depthBiasParamsKnown || depthBiasEnableChanged ||
            boundDepthBiasConstant != draw.depthBiasConstant ||
            boundDepthBiasSlope != draw.depthBiasSlope ||
            boundDepthBiasClamp != draw.depthBiasClamp;
        if (depthBiasParamsChanged) {
          commandBuffer.cmdSetDepthBias(
              draw.depthBiasConstant, draw.depthBiasSlope, draw.depthBiasClamp);
          boundDepthBiasConstant = draw.depthBiasConstant;
          boundDepthBiasSlope = draw.depthBiasSlope;
          boundDepthBiasClamp = draw.depthBiasClamp;
          depthBiasParamsKnown = true;
        }
      }

      if (draw.useScissor) {
        commandBuffer.cmdBindScissorRect({draw.scissor.x, draw.scissor.y,
                                          draw.scissor.width,
                                          draw.scissor.height});
        scissorMatchesViewport = false;
      } else if (!scissorMatchesViewport) {
        commandBuffer.cmdBindScissorRect(viewportScissor);
        scissorMatchesViewport = true;
      }

      if (!draw.pushConstants.empty()) {
        commandBuffer.cmdPushConstants(
            static_cast<const void *>(draw.pushConstants.data()),
            draw.pushConstants.size(), 0);
      }

      if (draw.command == DrawCommandType::IndexedIndirect) {
        if (!impl_->buffers.isValid(draw.indirectBuffer)) {
          return returnDrawError("Indirect buffer is invalid", drawLabelPushed);
        }
        if (draw.indirectDrawCount > 0) {
          commandBuffer.cmdDrawIndexedIndirect(
              impl_->buffers.getLvkHandle(draw.indirectBuffer),
              draw.indirectBufferOffset, draw.indirectDrawCount,
              draw.indirectStride);
        }
      } else if (draw.command == DrawCommandType::IndexedIndirectCount) {
        if (!impl_->buffers.isValid(draw.indirectBuffer) ||
            !impl_->buffers.isValid(draw.indirectCountBuffer)) {
          return returnDrawError("Indirect or count buffer is invalid",
                                 drawLabelPushed);
        }
        if (draw.indirectDrawCount > 0) {
          if (supportsIndexedIndirectCount) {
            commandBuffer.cmdDrawIndexedIndirectCount(
                impl_->buffers.getLvkHandle(draw.indirectBuffer),
                draw.indirectBufferOffset,
                impl_->buffers.getLvkHandle(draw.indirectCountBuffer),
                draw.indirectCountBufferOffset, draw.indirectDrawCount,
                draw.indirectStride);
          } else {
            commandBuffer.cmdDrawIndexedIndirect(
                impl_->buffers.getLvkHandle(draw.indirectBuffer),
                draw.indirectBufferOffset, draw.indirectDrawCount,
                draw.indirectStride);
          }
        }
      } else if (draw.indexCount > 0) {
        commandBuffer.cmdDrawIndexed(draw.indexCount, draw.instanceCount,
                                     draw.firstIndex, draw.vertexOffset,
                                     draw.firstInstance);
      } else {
        commandBuffer.cmdDraw(draw.vertexCount, draw.instanceCount,
                              draw.firstVertex, draw.firstInstance);
      }

      if (drawLabelPushed) {
        commandBuffer.cmdPopDebugGroupLabel();
      }
    }

    commandBuffer.cmdEndRendering();

    if (hasTimingReservation(timingReservation)) {
      commandBuffer.cmdWriteTimestamp(timingReservation.pool,
                                      timingReservation.endQuery);
    }

    if (passLabelPushed) {
      commandBuffer.cmdPopDebugGroupLabel();
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

bool LvkGPUDevice::supportsParallelGraphicsRecording() const { return true; }

uint32_t LvkGPUDevice::maxParallelGraphicsRecordingContexts() const {
  return kMaxGraphicsRecordingContexts;
}

Result<RecordingContextHandle, std::string>
LvkGPUDevice::acquireGraphicsRecordingContext(uint32_t workerIndex) {
  if (!impl_->context) {
    return Result<RecordingContextHandle, std::string>::makeError(
        "LvkGPUDevice::acquireGraphicsRecordingContext: context is null");
  }
  std::scoped_lock lock(impl_->contextImmediateMutex,
                        impl_->graphicsContextMutex);
  lvk::ICommandBuffer &commandBuffer = impl_->context->acquireCommandBuffer();
  auto indexResult = allocateMonotonicHandleIndex(
      impl_->nextRecordingContextIndex,
      "LvkGPUDevice::acquireGraphicsRecordingContext");
  if (indexResult.hasError()) {
    return Result<RecordingContextHandle, std::string>::makeError(
        indexResult.error());
  }

  const uint32_t index = indexResult.value();
  const RecordingContextHandle handle{
      .index = index,
      .generation =
          nextHandleGeneration(impl_->recordingContextGenerations, index)};
  if (impl_->activeGraphicsContexts.size() <= index) {
    impl_->activeGraphicsContexts.resize(static_cast<size_t>(index) + 1u);
  }
  if (impl_->activeGraphicsContextOccupied.size() <= index) {
    impl_->activeGraphicsContextOccupied.resize(static_cast<size_t>(index) + 1u,
                                                0u);
  }
  impl_->activeGraphicsContexts[index] = ActiveGraphicsRecordingContext{
      .handle = handle,
      .commandBuffer = &commandBuffer,
      .workerIndex = workerIndex,
      .timingQueryBase = workerIndex * kGpuTimingQueriesPerContext,
      .timingQueryCursor = 0u,
      .timingRanges = {},
      .hadShadowSdsmPass = false,
      .timingQuerySliceReset = false,
  };
  impl_->activeGraphicsContextOccupied[index] = 1u;
  return Result<RecordingContextHandle, std::string>::makeResult(handle);
}

Result<bool, std::string> LvkGPUDevice::recordGraphicsBarriers(
    RecordingContextHandle ctx,
    std::span<const GraphicsBarrierRecord> barriers) {
  if (barriers.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  lvk::ICommandBuffer *rawCommandBuffer = nullptr;
  {
    std::lock_guard lock(impl_->graphicsContextMutex);
    if (const ActiveGraphicsRecordingContext *entry =
            findActiveGraphicsContextSlot(*impl_, ctx);
        entry != nullptr) {
      rawCommandBuffer = entry->commandBuffer;
    }
  }
  if (rawCommandBuffer == nullptr) {
    return Result<bool, std::string>::makeError(
        "LvkGPUDevice::recordGraphicsBarriers: unknown recording context");
  }

  const bool needsInternalBarrierApi =
      std::find_if(barriers.begin(), barriers.end(),
                   [](const GraphicsBarrierRecord &barrier) {
                     return barrier.resourceKind ==
                                GraphicsBarrierResourceKind::Buffer ||
                            (barrier.resourceKind ==
                                 GraphicsBarrierResourceKind::Texture &&
                             barrier.afterState != GraphicsBarrierState::Read &&
                             barrier.afterState !=
                                 GraphicsBarrierState::Unknown);
                   }) != barriers.end();
  lvk::ICommandBuffer *commandBuffer = nullptr;
#if NURI_LVK_HAS_VULKAN_COMMAND_BUFFER
  commandBuffer = needsInternalBarrierApi ? rawCommandBuffer : nullptr;
#else
  if (needsInternalBarrierApi) {
    return makeLvkBarrierApiError("LvkGPUDevice::recordGraphicsBarriers");
  }
#endif

  for (const GraphicsBarrierRecord &barrier : barriers) {
    if (barrier.resourceKind == GraphicsBarrierResourceKind::Texture) {
      const TextureHandle textureHandle = barrier.textureHandle();
      if (!nuri::isValid(textureHandle) ||
          !impl_->textures.isValid(textureHandle)) {
        return Result<bool, std::string>::makeError(
            "LvkGPUDevice::recordGraphicsBarriers: texture barrier handle "
            "is invalid");
      }

      const lvk::TextureHandle texture =
          impl_->textures.getLvkHandle(textureHandle);
      switch (barrier.afterState) {
      case GraphicsBarrierState::Read:
        rawCommandBuffer->transitionToShaderReadOnly(texture);
        break;
      case GraphicsBarrierState::Attachment:
#if NURI_LVK_HAS_VULKAN_COMMAND_BUFFER
        if (commandBuffer == nullptr) {
          return makeLvkBarrierApiError("LvkGPUDevice::recordGraphicsBarriers");
        }
        static_cast<lvk::CommandBuffer *>(commandBuffer)
            ->transitionTextureLayout(texture,
                                      VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
#else
        return makeLvkBarrierApiError("LvkGPUDevice::recordGraphicsBarriers");
#endif
        break;
      case GraphicsBarrierState::Write:
#if NURI_LVK_HAS_VULKAN_COMMAND_BUFFER
        if (commandBuffer == nullptr) {
          return makeLvkBarrierApiError("LvkGPUDevice::recordGraphicsBarriers");
        }
        static_cast<lvk::CommandBuffer *>(commandBuffer)
            ->transitionTextureLayout(texture, VK_IMAGE_LAYOUT_GENERAL);
#else
        return makeLvkBarrierApiError("LvkGPUDevice::recordGraphicsBarriers");
#endif
        break;
      case GraphicsBarrierState::Present:
#if NURI_LVK_HAS_VULKAN_COMMAND_BUFFER
        if (commandBuffer == nullptr) {
          return makeLvkBarrierApiError("LvkGPUDevice::recordGraphicsBarriers");
        }
        static_cast<lvk::CommandBuffer *>(commandBuffer)
            ->transitionTextureLayout(texture, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
#else
        return makeLvkBarrierApiError("LvkGPUDevice::recordGraphicsBarriers");
#endif
        break;
      case GraphicsBarrierState::Unknown:
      default:
        break;
      }
      continue;
    }

    const BufferHandle bufferHandle = barrier.bufferHandle();
    if (!nuri::isValid(bufferHandle) || !impl_->buffers.isValid(bufferHandle)) {
      return Result<bool, std::string>::makeError(
          "LvkGPUDevice::recordGraphicsBarriers: buffer barrier handle is "
          "invalid");
    }
    if (commandBuffer == nullptr) {
      return makeLvkBarrierApiError("LvkGPUDevice::recordGraphicsBarriers");
    }

    const VkPipelineStageFlags2 srcStage =
        graphicsBarrierStages(barrier.beforeState, false);
    const VkPipelineStageFlags2 dstStage =
        graphicsBarrierStages(barrier.afterState, false);
#if NURI_LVK_HAS_VULKAN_COMMAND_BUFFER
    static_cast<lvk::CommandBuffer *>(commandBuffer)
        ->bufferBarrier(impl_->buffers.getLvkHandle(bufferHandle), srcStage,
                        dstStage);
#else
    return makeLvkBarrierApiError("LvkGPUDevice::recordGraphicsBarriers");
#endif
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
LvkGPUDevice::recordGraphicsPass(RecordingContextHandle ctx,
                                 const RenderPass &pass) {
  lvk::ICommandBuffer *commandBuffer = nullptr;
  PassTimingReservation timingReservation{};
  {
    std::lock_guard lock(impl_->graphicsContextMutex);
    if (ActiveGraphicsRecordingContext *entry =
            findActiveGraphicsContextSlot(*impl_, ctx);
        entry != nullptr) {
      commandBuffer = entry->commandBuffer;
      timingReservation =
          reservePassTimingReservationLocked(*impl_, *entry, pass);
    }
  }
  if (commandBuffer != nullptr) {
    const std::array<PassTimingReservation, 1u> timingReservations{
        timingReservation};
    auto result = recordRenderPasses(
        *commandBuffer, std::span<const RenderPass>(&pass, 1u),
        std::span<const PassTimingReservation>(timingReservations.data(),
                                               timingReservations.size()));
    if (!result.hasError() && result.value()) {
      if (pass.gpuTimingScope == GpuTimingScope::ShadowSdsm &&
          !impl_->loggedShadowSdsmTimingRecordDiagnostic) {
        impl_->loggedShadowSdsmTimingRecordDiagnostic = true;
        if (hasTimingReservation(timingReservation)) {
          NURI_LOG_INFO(
              "LvkGPUDevice: recorded shadow SDSM pass with timing "
              "reservation frame=%llu beginQuery=%u endQuery=%u",
              static_cast<unsigned long long>(impl_->currentFrameIndex),
              timingReservation.beginQuery, timingReservation.endQuery);
        } else {
          NURI_LOG_WARNING(
              "LvkGPUDevice: recorded shadow SDSM pass without a timing "
              "reservation (frame %llu)",
              static_cast<unsigned long long>(impl_->currentFrameIndex));
        }
      }
      std::lock_guard lock(impl_->graphicsContextMutex);
      if (ActiveGraphicsRecordingContext *entry =
              findActiveGraphicsContextSlot(*impl_, ctx);
          entry != nullptr) {
        entry->hadShadowSdsmPass =
            entry->hadShadowSdsmPass ||
            pass.gpuTimingScope == GpuTimingScope::ShadowSdsm;
        if (timingReservation.scope != GpuTimingScope::None) {
          entry->timingRanges.push_back(PendingTimingQueryRange{
              .scope = timingReservation.scope,
              .firstQuery = timingReservation.beginQuery,
              .intervalCount = 1u,
          });
        }
      }
    }
    return result;
  }
  return Result<bool, std::string>::makeError(
      "LvkGPUDevice::recordGraphicsPass: unknown recording context");
}

Result<RecordedCommandBufferHandle, std::string>
LvkGPUDevice::finishGraphicsRecordingContext(RecordingContextHandle ctx) {
  std::lock_guard lock(impl_->graphicsContextMutex);
  ActiveGraphicsRecordingContext *activeContext =
      findActiveGraphicsContextSlot(*impl_, ctx);
  if (activeContext == nullptr) {
    return Result<RecordedCommandBufferHandle, std::string>::makeError(
        "LvkGPUDevice::finishGraphicsRecordingContext: unknown recording "
        "context");
  }

  auto indexResult = allocateMonotonicHandleIndex(
      impl_->nextRecordedCommandBufferIndex,
      "LvkGPUDevice::finishGraphicsRecordingContext");
  if (indexResult.hasError()) {
    return Result<RecordedCommandBufferHandle, std::string>::makeError(
        indexResult.error());
  }

  const uint32_t index = indexResult.value();
  const RecordedCommandBufferHandle handle{
      .index = index,
      .generation =
          nextHandleGeneration(impl_->recordedCommandBufferGenerations, index)};
  impl_->recordedGraphicsCommandBuffers.push_back(RecordedGraphicsCommandBuffer{
      .handle = handle,
      .commandBuffer = activeContext->commandBuffer,
      .timingRanges = std::move(activeContext->timingRanges),
      .hadShadowSdsmPass = activeContext->hadShadowSdsmPass,
  });
  impl_->activeGraphicsContexts[ctx.index] = ActiveGraphicsRecordingContext{};
  impl_->activeGraphicsContextOccupied[ctx.index] = 0u;
  return Result<RecordedCommandBufferHandle, std::string>::makeResult(handle);
}

Result<bool, std::string>
LvkGPUDevice::discardGraphicsRecordingContext(RecordingContextHandle ctx) {
  std::scoped_lock lock(impl_->contextImmediateMutex,
                        impl_->graphicsContextMutex);
  ActiveGraphicsRecordingContext *activeContext =
      findActiveGraphicsContextSlot(*impl_, ctx);
  if (activeContext == nullptr) {
    return Result<bool, std::string>::makeError(
        "LvkGPUDevice::discardGraphicsRecordingContext: unknown recording "
        "context");
  }

  impl_->context->discard(*activeContext->commandBuffer);
  impl_->activeGraphicsContexts[ctx.index] = ActiveGraphicsRecordingContext{};
  impl_->activeGraphicsContextOccupied[ctx.index] = 0u;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> LvkGPUDevice::discardRecordedGraphicsCommandBuffer(
    RecordedCommandBufferHandle commandBuffer) {
  std::scoped_lock lock(impl_->contextImmediateMutex,
                        impl_->graphicsContextMutex);
  for (size_t i = 0u; i < impl_->recordedGraphicsCommandBuffers.size(); ++i) {
    if (!areSameHandle(impl_->recordedGraphicsCommandBuffers[i].handle,
                       commandBuffer)) {
      continue;
    }

    impl_->context->discard(
        *impl_->recordedGraphicsCommandBuffers[i].commandBuffer);
    if (i + 1u != impl_->recordedGraphicsCommandBuffers.size()) {
      impl_->recordedGraphicsCommandBuffers[i] =
          std::move(impl_->recordedGraphicsCommandBuffers.back());
    }
    impl_->recordedGraphicsCommandBuffers.pop_back();
    return Result<bool, std::string>::makeResult(true);
  }

  return Result<bool, std::string>::makeError(
      "LvkGPUDevice::discardRecordedGraphicsCommandBuffer: unknown command "
      "buffer");
}

Result<SubmissionHandle, std::string> LvkGPUDevice::submitRecordedGraphicsFrame(
    std::span<const RecordedCommandBufferHandle> commandBuffers,
    std::span<const SubmitBatchMeta> batches) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_SUBMIT);
  if (commandBuffers.empty()) {
    std::scoped_lock emptyLock(impl_->contextImmediateMutex,
                               impl_->graphicsContextMutex);
    if (impl_->currentFrameTimingCapture.has_value()) {
      recycleTimingQueryPool(*impl_,
                             std::move(impl_->currentFrameTimingCapture->pool));
      impl_->currentFrameTimingCapture.reset();
    }
    return Result<SubmissionHandle, std::string>::makeResult({});
  }

  std::scoped_lock lock(impl_->contextImmediateMutex,
                        impl_->graphicsContextMutex);
  std::vector<uint8_t> presentFlags(commandBuffers.size(), 0u);
  for (const SubmitBatchMeta &batch : batches) {
    if (batch.commandBufferOffset > commandBuffers.size() ||
        batch.commandBufferCount >
            commandBuffers.size() - batch.commandBufferOffset) {
      return Result<SubmissionHandle, std::string>::makeError(
          "LvkGPUDevice::submitRecordedGraphicsFrame: submit batch range is "
          "out of bounds");
    }
    if (batch.presentsFrameOutput && batch.commandBufferCount > 0u) {
      presentFlags[batch.commandBufferOffset + batch.commandBufferCount - 1u] =
          1u;
    }
  }

  const bool wantsPresent = std::find(presentFlags.begin(), presentFlags.end(),
                                      1u) != presentFlags.end();
  const lvk::TextureHandle swapchainTexture =
      wantsPresent ? impl_->currentFrameSwapchainTexture : lvk::TextureHandle{};
  if (wantsPresent && !swapchainTexture.valid()) {
    return Result<SubmissionHandle, std::string>::makeError(
        "LvkGPUDevice::submitRecordedGraphicsFrame: invalid swapchain "
        "texture");
  }

  const auto foldRecordedHandleKey =
      [](RecordedCommandBufferHandle handle) -> uint64_t {
    return (static_cast<uint64_t>(handle.index) << 32u) | handle.generation;
  };

  std::unordered_map<uint64_t, size_t> recordedIndexByHandle{};
  recordedIndexByHandle.reserve(impl_->recordedGraphicsCommandBuffers.size());
  for (size_t i = 0u; i < impl_->recordedGraphicsCommandBuffers.size(); ++i) {
    recordedIndexByHandle.emplace(
        foldRecordedHandleKey(impl_->recordedGraphicsCommandBuffers[i].handle),
        i);
  }

  std::vector<size_t> matchedRecordedIndices(
      commandBuffers.size(), impl_->recordedGraphicsCommandBuffers.size());
  for (size_t i = 0u; i < commandBuffers.size(); ++i) {
    const auto found =
        recordedIndexByHandle.find(foldRecordedHandleKey(commandBuffers[i]));
    if (found == recordedIndexByHandle.end()) {
      return Result<SubmissionHandle, std::string>::makeError(
          "LvkGPUDevice::submitRecordedGraphicsFrame: unknown recorded "
          "command buffer");
    }
    matchedRecordedIndices[i] = found->second;
  }

  std::vector<PendingTimingQueryRange> timingRanges{};
  bool hadShadowSdsmPass = false;
  if (impl_->currentFrameTimingCapture.has_value()) {
    size_t timingRangeCount = 0u;
    for (const size_t matchedIndex : matchedRecordedIndices) {
      if (matchedIndex >= impl_->recordedGraphicsCommandBuffers.size()) {
        continue;
      }
      const RecordedGraphicsCommandBuffer &recorded =
          impl_->recordedGraphicsCommandBuffers[matchedIndex];
      hadShadowSdsmPass = hadShadowSdsmPass || recorded.hadShadowSdsmPass;
      timingRangeCount += recorded.timingRanges.size();
    }
    timingRanges.reserve(timingRangeCount);
    for (const size_t matchedIndex : matchedRecordedIndices) {
      if (matchedIndex >= impl_->recordedGraphicsCommandBuffers.size()) {
        continue;
      }
      const RecordedGraphicsCommandBuffer &recorded =
          impl_->recordedGraphicsCommandBuffers[matchedIndex];
      if (recorded.timingRanges.empty()) {
        continue;
      }
      timingRanges.insert(timingRanges.end(), recorded.timingRanges.begin(),
                          recorded.timingRanges.end());
    }
  }
  if (hadShadowSdsmPass && !impl_->loggedShadowSdsmTimingSubmissionWarning) {
    bool hasShadowSdsmRange = false;
    for (const PendingTimingQueryRange &range : timingRanges) {
      hasShadowSdsmRange =
          hasShadowSdsmRange || range.scope == GpuTimingScope::ShadowSdsm;
    }
    if (!impl_->loggedShadowSdsmTimingSubmissionDiagnostic) {
      impl_->loggedShadowSdsmTimingSubmissionDiagnostic = true;
      NURI_LOG_INFO("LvkGPUDevice: submitting shadow SDSM pass frame=%llu "
                    "timingRanges=%zu hasShadowSdsmRange=%u",
                    static_cast<unsigned long long>(
                        impl_->currentFrameTimingCapture.has_value()
                            ? impl_->currentFrameTimingCapture->frameIndex
                            : impl_->currentFrameIndex),
                    timingRanges.size(), hasShadowSdsmRange ? 1u : 0u);
    }
    if (!hasShadowSdsmRange) {
      impl_->loggedShadowSdsmTimingSubmissionWarning = true;
      NURI_LOG_WARNING(
          "LvkGPUDevice: shadow SDSM pass was submitted without a matching "
          "SDSM timing range (frame %llu)",
          static_cast<unsigned long long>(
              impl_->currentFrameTimingCapture.has_value()
                  ? impl_->currentFrameTimingCapture->frameIndex
                  : impl_->currentFrameIndex));
    }
  }

  lvk::SubmitHandle lastSubmitHandle{};
  std::vector<uint8_t> consumedRecordedFlags(
      impl_->recordedGraphicsCommandBuffers.size(), 0u);
  for (uint32_t i = 0u; i < commandBuffers.size(); ++i) {
    const size_t matchedIndex = matchedRecordedIndices[i];
    if (matchedIndex >= impl_->recordedGraphicsCommandBuffers.size()) {
      return Result<SubmissionHandle, std::string>::makeError(
          "LvkGPUDevice::submitRecordedGraphicsFrame: unknown recorded "
          "command buffer");
    }
    lvk::ICommandBuffer *commandBuffer =
        impl_->recordedGraphicsCommandBuffers[matchedIndex].commandBuffer;
    if (commandBuffer == nullptr) {
      return Result<SubmissionHandle, std::string>::makeError(
          "LvkGPUDevice::submitRecordedGraphicsFrame: unknown recorded "
          "command buffer");
    }

    lastSubmitHandle = impl_->context->submit(
        *commandBuffer,
        presentFlags[i] != 0u ? swapchainTexture : lvk::TextureHandle{});
    consumedRecordedFlags[matchedIndex] = 1u;
  }

  size_t writeIndex = 0u;
  for (size_t readIndex = 0u;
       readIndex < impl_->recordedGraphicsCommandBuffers.size(); ++readIndex) {
    if (consumedRecordedFlags[readIndex] != 0u) {
      continue;
    }
    if (writeIndex != readIndex) {
      impl_->recordedGraphicsCommandBuffers[writeIndex] =
          std::move(impl_->recordedGraphicsCommandBuffers[readIndex]);
    }
    ++writeIndex;
  }
  impl_->recordedGraphicsCommandBuffers.resize(writeIndex);

  if (wantsPresent) {
    impl_->currentFrameSwapchainTexture = {};
    impl_->hasPreparedSwapchainImage = false;
  }
  if (impl_->currentFrameTimingCapture.has_value()) {
    if (!lastSubmitHandle.empty() && !timingRanges.empty()) {
      PendingGpuTimingSubmission pending{};
      pending.submission = toNuriSubmissionHandle(lastSubmitHandle);
      pending.frameIndex = impl_->currentFrameTimingCapture->frameIndex;
      pending.pool = std::move(impl_->currentFrameTimingCapture->pool);
      pending.timingRanges = std::move(timingRanges);
      impl_->pendingGpuTimingSubmissions.push_back(std::move(pending));
    } else {
      recycleTimingQueryPool(*impl_,
                             std::move(impl_->currentFrameTimingCapture->pool));
    }
    impl_->currentFrameTimingCapture.reset();
  }
  ++impl_->submittedFrameCount;

  return Result<SubmissionHandle, std::string>::makeResult(
      toNuriSubmissionHandle(lastSubmitHandle));
}

bool LvkGPUDevice::isSubmissionComplete(SubmissionHandle handle) const {
  if (!nuri::isValid(handle)) {
    return true;
  }
  if (!impl_ || !impl_->context) {
    return false;
  }
  std::lock_guard immediateLock(impl_->contextImmediateMutex);
  return impl_->context->isReady(toLvkSubmitHandle(handle));
}

Result<bool, std::string> LvkGPUDevice::submitComputeDispatches(
    std::span<const ComputeDispatchItem> dispatches) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DISPATCH);
  static_assert(kMaxDependencyResources <=
                lvk::Dependencies::LVK_MAX_SUBMIT_DEPENDENCIES);
  if (dispatches.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  std::lock_guard immediateLock(impl_->contextImmediateMutex);
  lvk::ICommandBuffer &commandBuffer = impl_->context->acquireCommandBuffer();
  const auto fillDependencies =
      [this](std::span<const BufferHandle> dependencyBuffers,
             lvk::Dependencies &deps,
             std::string_view context) -> Result<bool, std::string> {
    if (dependencyBuffers.size() > kMaxDependencyResources) {
      return makeDependencyCountExceededError(context);
    }

    size_t dstIndex = 0;
    for (const BufferHandle bufferHandle : dependencyBuffers) {
      if (!nuri::isValid(bufferHandle)) {
        continue;
      }
      if (!impl_->buffers.isValid(bufferHandle)) {
        return makeDependencyError(context, "dependency buffer is invalid");
      }
      deps.buffers[dstIndex++] = impl_->buffers.getLvkHandle(bufferHandle);
    }
    return Result<bool, std::string>::makeResult(true);
  };

  bool computePipelineBound = false;
  ComputePipelineHandle boundComputePipeline{};
  for (const ComputeDispatchItem &dispatch : dispatches) {
    if (!impl_->computePipelines.isValid(dispatch.pipeline)) {
      return Result<bool, std::string>::makeError(
          "submitComputeDispatches: invalid compute pipeline handle");
    }
    if (dispatch.dispatch.x == 0 || dispatch.dispatch.y == 0 ||
        dispatch.dispatch.z == 0) {
      return Result<bool, std::string>::makeError(
          "submitComputeDispatches: invalid compute dispatch size");
    }

    lvk::Dependencies dispatchDependencies{};
    auto dispatchDepsResult =
        fillDependencies(dispatch.dependencyBuffers, dispatchDependencies,
                         "LvkGPUDevice::submitComputeDispatches");
    if (dispatchDepsResult.hasError()) {
      return dispatchDepsResult;
    }

    const bool dispatchLabelPushed =
        pushDebugLabel(commandBuffer, dispatch.debugLabel, dispatch.debugColor);

    if (!computePipelineBound ||
        !areSameHandle(dispatch.pipeline, boundComputePipeline)) {
      commandBuffer.cmdBindComputePipeline(
          impl_->computePipelines.getLvkHandle(dispatch.pipeline));
      boundComputePipeline = dispatch.pipeline;
      computePipelineBound = true;
    }
    if (!dispatch.pushConstants.empty()) {
      commandBuffer.cmdPushConstants(
          static_cast<const void *>(dispatch.pushConstants.data()),
          dispatch.pushConstants.size(), 0);
    }
    commandBuffer.cmdDispatchThreadGroups({.width = dispatch.dispatch.x,
                                           .height = dispatch.dispatch.y,
                                           .depth = dispatch.dispatch.z},
                                          dispatchDependencies);

    if (dispatchLabelPushed) {
      commandBuffer.cmdPopDebugGroupLabel();
    }
  }

  const lvk::SubmitHandle submitHandle = impl_->context->submit(commandBuffer);
  impl_->context->wait(submitHandle);
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

Result<bool, std::string>
LvkGPUDevice::readBuffer(BufferHandle buffer, size_t offset,
                         std::span<std::byte> outBytes) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  if (!impl_->buffers.isValid(buffer)) {
    return Result<bool, std::string>::makeError("readBuffer: invalid buffer");
  }
  if (outBytes.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  const lvk::BufferHandle lvkBuf = impl_->buffers.getLvkHandle(buffer);
  const size_t bufferSize =
      static_cast<size_t>(lvk::getBufferSize(impl_->context.get(), lvkBuf));
  if (offset > bufferSize || outBytes.size() > bufferSize - offset) {
    return Result<bool, std::string>::makeError(
        "readBuffer: offset + outBytes.size() exceeds buffer size");
  }

  if (impl_->context->getMappedPtr(lvkBuf) == nullptr) {
    return Result<bool, std::string>::makeError(
        "readBuffer: buffer is not host-visible/mapped");
  }

  // Always route readback through LVK's download path so non-coherent mapped
  // memory gets invalidated correctly before host reads.
  lvk::Result res = impl_->context->download(
      lvkBuf, static_cast<void *>(outBytes.data()), outBytes.size(), offset);
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
  const size_t bufferSize =
      static_cast<size_t>(lvk::getBufferSize(impl_->context.get(), lvkBuf));
  if (offset > bufferSize || size > bufferSize - offset) {
    return;
  }
  impl_->context->flushMappedMemory(lvkBuf, offset, size);
}

Result<bool, std::string>
LvkGPUDevice::readTexture(TextureHandle texture,
                          const TextureReadbackRegion &region,
                          std::span<std::byte> outBytes) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  if (!impl_->textures.isValid(texture)) {
    return Result<bool, std::string>::makeError(
        "readTexture: invalid texture handle");
  }
  if (region.width == 0 || region.height == 0) {
    return Result<bool, std::string>::makeError(
        "readTexture: readback region size must be non-zero");
  }

  const Format format = impl_->textures.getFormat(texture);
  size_t bytesPerPixel = 0;
  switch (format) {
  case Format::R32_UINT:
    bytesPerPixel = sizeof(uint32_t);
    break;
  case Format::R32_FLOAT:
    bytesPerPixel = sizeof(float);
    break;
  case Format::RG32_FLOAT:
    bytesPerPixel = sizeof(float) * 2u;
    break;
  case Format::RG16_FLOAT:
    bytesPerPixel = sizeof(uint16_t) * 2u;
    break;
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    bytesPerPixel = 4;
    break;
  case Format::RGBA16_FLOAT:
    bytesPerPixel = 8;
    break;
  case Format::RGBA32_FLOAT:
    bytesPerPixel = 16;
    break;
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
    // Block-compressed formats need block-aligned regions and specialized
    // readback size calculations, so generic readTexture() rejects them here.
    break;
  case Format::D16_UNORM:
    bytesPerPixel = sizeof(uint16_t);
    break;
  case Format::D32_FLOAT:
    bytesPerPixel = sizeof(float);
    break;
  case Format::Count:
    break;
  }
  if (bytesPerPixel == 0) {
    return Result<bool, std::string>::makeError(
        "readTexture: unsupported texture format for readback");
  }

  const lvk::TextureHandle lvkTexture = impl_->textures.getLvkHandle(texture);
  if (!lvkTexture.valid()) {
    return Result<bool, std::string>::makeError(
        "readTexture: invalid LVK texture handle");
  }
  const lvk::Dimensions dimensions = impl_->context->getDimensions(lvkTexture);
  const uint32_t mipWidth =
      std::max(1u, dimensions.width >> std::min(region.mipLevel, 31u));
  const uint32_t mipHeight =
      std::max(1u, dimensions.height >> std::min(region.mipLevel, 31u));
  if (region.x >= mipWidth || region.y >= mipHeight) {
    return Result<bool, std::string>::makeError(
        "readTexture: readback origin is out of bounds");
  }
  if (region.width > mipWidth - region.x ||
      region.height > mipHeight - region.y) {
    return Result<bool, std::string>::makeError(
        "readTexture: readback region is out of bounds");
  }

  const uint64_t expectedSize = static_cast<uint64_t>(region.width) *
                                static_cast<uint64_t>(region.height) *
                                static_cast<uint64_t>(bytesPerPixel);
  if (expectedSize > std::numeric_limits<size_t>::max()) {
    return Result<bool, std::string>::makeError(
        "readTexture: readback size overflows size_t");
  }
  if (outBytes.size() < static_cast<size_t>(expectedSize)) {
    return Result<bool, std::string>::makeError(
        "readTexture: output buffer is too small");
  }

  const lvk::TextureRangeDesc range{
      .offset = {.x = static_cast<int32_t>(region.x),
                 .y = static_cast<int32_t>(region.y),
                 .z = 0},
      .dimensions = {.width = region.width,
                     .height = region.height,
                     .depth = 1},
      .layer = region.layer,
      .numLayers = 1,
      .mipLevel = region.mipLevel,
      .numMipLevels = 1,
  };
  lvk::Result result = impl_->context->download(
      lvkTexture, range, static_cast<void *>(outBytes.data()));
  if (!result.isOk()) {
    return Result<bool, std::string>::makeError(std::string(result.message));
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<SubmissionHandle, std::string>
LvkGPUDevice::submitBackgroundBufferCopies(
    std::span<const BufferCopyRegion> regions, std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  if (regions.empty()) {
    return Result<SubmissionHandle, std::string>::makeResult({});
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
      return Result<SubmissionHandle, std::string>::makeError(
          "submitBackgroundBufferCopies: invalid source or destination "
          "buffer");
    }

    const lvk::BufferHandle src = impl_->buffers.getLvkHandle(region.srcBuffer);
    const lvk::BufferHandle dst = impl_->buffers.getLvkHandle(region.dstBuffer);
    const size_t srcSize =
        static_cast<size_t>(lvk::getBufferSize(impl_->context.get(), src));
    const size_t dstSize =
        static_cast<size_t>(lvk::getBufferSize(impl_->context.get(), dst));
    if (region.srcOffset > srcSize ||
        region.size > srcSize - region.srcOffset) {
      return Result<SubmissionHandle, std::string>::makeError(
          "submitBackgroundBufferCopies: source copy range is out of bounds");
    }
    if (region.dstOffset > dstSize ||
        region.size > dstSize - region.dstOffset) {
      return Result<SubmissionHandle, std::string>::makeError(
          "submitBackgroundBufferCopies: destination copy range is out of "
          "bounds");
    }
    ++copyCount;
  }

  if (copyCount == 0) {
    return Result<SubmissionHandle, std::string>::makeResult({});
  }

  std::lock_guard immediateLock(impl_->contextImmediateMutex);
  lvk::ICommandBuffer &commandBuffer = impl_->context->acquireCommandBuffer();
  const bool debugLabelPushed =
      pushDebugLabel(commandBuffer, debugName, NURI_PROFILER_COLOR_CMD_COPY);
  for (const BufferCopyRegion &region : regions) {
    if (region.size == 0) {
      continue;
    }
    const lvk::BufferHandle src = impl_->buffers.getLvkHandle(region.srcBuffer);
    const lvk::BufferHandle dst = impl_->buffers.getLvkHandle(region.dstBuffer);
    commandBuffer.cmdCopyBuffer(src, dst, region.srcOffset, region.dstOffset,
                                region.size);
  }
  if (debugLabelPushed) {
    commandBuffer.cmdPopDebugGroupLabel();
  }

  const lvk::SubmitHandle submitHandle = impl_->context->submit(commandBuffer);
  return Result<SubmissionHandle, std::string>::makeResult(
      toNuriSubmissionHandle(submitHandle));
}

void LvkGPUDevice::waitIdle() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_WAIT);
  if (impl_->context) {
    // Empty SubmitHandle results in vkDeviceWaitIdle
    impl_->context->wait(lvk::SubmitHandle{});
    std::lock_guard immediateLock(impl_->contextImmediateMutex);
    collectCompletedGpuTimingSubmissions(*impl_);
  }
}

} // namespace nuri
