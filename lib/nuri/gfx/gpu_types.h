#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <type_traits>
namespace nuri {

inline constexpr uint32_t kInvalidTextureBindlessIndex =
    std::numeric_limits<uint32_t>::max();

enum class SwapchainPresentMode : uint8_t {
  Unknown = 0,
  Immediate,
  Mailbox,
  Fifo,
};

template <typename Tag> struct Handle {
  uint32_t index = 0;
  uint32_t generation = 0;
  constexpr bool operator==(const Handle &) const noexcept = default;
};

struct BufferHandleTag;
struct TextureHandleTag;
struct SamplerHandleTag;
struct ShaderHandleTag;
struct RenderPipelineHandleTag;
struct ComputePipelineHandleTag;
struct MeshletPipelineHandleTag;
struct RecordingContextHandleTag;
struct RecordedCommandBufferHandleTag;
struct SubmissionHandleTag;
struct GeometryAllocationHandleTag;

using BufferHandle = Handle<BufferHandleTag>;
using TextureHandle = Handle<TextureHandleTag>;
using SamplerHandle = Handle<SamplerHandleTag>;
using ShaderHandle = Handle<ShaderHandleTag>;
using RenderPipelineHandle = Handle<RenderPipelineHandleTag>;
using ComputePipelineHandle = Handle<ComputePipelineHandleTag>;
using MeshletPipelineHandle = Handle<MeshletPipelineHandleTag>;
using RecordingContextHandle = Handle<RecordingContextHandleTag>;
using RecordedCommandBufferHandle = Handle<RecordedCommandBufferHandleTag>;
using SubmissionHandle = Handle<SubmissionHandleTag>;
using GeometryAllocationHandle = Handle<GeometryAllocationHandleTag>;

template <typename Tag> constexpr bool isValid(Handle<Tag> handle) noexcept {
  return handle.generation != 0;
}

template <typename Tag>
[[nodiscard]] constexpr uint64_t handleKey(Handle<Tag> handle) noexcept {
  return (static_cast<uint64_t>(handle.generation) << 32u) | handle.index;
}

struct HandleHash {
  template <typename Tag>
  [[nodiscard]] size_t operator()(Handle<Tag> handle) const noexcept {
    return std::hash<uint64_t>{}(handleKey(handle));
  }
};

[[nodiscard]] constexpr uint32_t saturateToU32(size_t value) noexcept {
  return static_cast<uint32_t>(
      std::min(value, size_t(std::numeric_limits<uint32_t>::max())));
}

static_assert(std::is_aggregate_v<BufferHandle>);
static_assert(std::is_standard_layout_v<BufferHandle>);
static_assert(std::is_trivially_copyable_v<BufferHandle>);
static_assert(std::is_trivially_destructible_v<BufferHandle>);
static_assert(sizeof(BufferHandle) == sizeof(uint32_t) * 2u);
static_assert(alignof(BufferHandle) == alignof(uint32_t));
static_assert(!std::is_same_v<BufferHandle, TextureHandle>);
static_assert(BufferHandle{.index = 7u, .generation = 3u} ==
              BufferHandle{7u, 3u});
static_assert(BufferHandle{.index = 7u, .generation = 3u} !=
              BufferHandle{7u, 4u});
static_assert(isValid(BufferHandle{.index = 7u, .generation = 3u}));
static_assert(!isValid(BufferHandle{}));

enum class Format : uint8_t {
  R32_UINT,
  RGBA8_UNORM,
  RGBA8_SRGB,
  RGBA8_UINT,
  RGBA16_FLOAT,
  RGBA32_FLOAT,
  D32_FLOAT,
  BC7_RGBA_UNORM,
  BC7_RGBA_SRGB,
  ETC2_RGB8_UNORM,
  ETC2_RGB8_SRGB,
  R32_FLOAT,
  RG32_FLOAT,
  D16_UNORM,
  RG16_FLOAT,
  R8_UNORM,
  R16_UNORM,
  Count
};

[[nodiscard]] constexpr uint32_t formatTexelBytes(Format format) noexcept {
  switch (format) {
  case Format::R8_UNORM:
    return 1u;
  case Format::R16_UNORM:
  case Format::D16_UNORM:
    return 2u;
  case Format::R32_UINT:
  case Format::R32_FLOAT:
  case Format::D32_FLOAT:
  case Format::RG16_FLOAT:
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return 4u;
  case Format::RG32_FLOAT:
  case Format::RGBA16_FLOAT:
    return 8u;
  case Format::RGBA32_FLOAT:
    return 16u;
  default:
    return 0u;
  }
}

static_assert(static_cast<uint8_t>(Format::D32_FLOAT) == 6u);
static_assert(static_cast<uint8_t>(Format::BC7_RGBA_UNORM) == 7u);
static_assert(static_cast<uint8_t>(Format::R32_FLOAT) == 11u);
static_assert(static_cast<uint8_t>(Format::RG32_FLOAT) == 12u);
static_assert(static_cast<uint8_t>(Format::D16_UNORM) == 13u);
static_assert(static_cast<uint8_t>(Format::RG16_FLOAT) == 14u);
static_assert(static_cast<uint8_t>(Format::R8_UNORM) == 15u);
static_assert(static_cast<uint8_t>(Format::R16_UNORM) == 16u);

enum class BufferUsage : uint8_t {
  None = 0,
  Vertex = 1u << 0u,
  Index = 1u << 1u,
  Uniform = 1u << 2u,
  Storage = 1u << 3u,
  Indirect = 1u << 4u,
};

constexpr BufferUsage operator|(BufferUsage lhs, BufferUsage rhs) {
  return static_cast<BufferUsage>(static_cast<uint8_t>(lhs) |
                                  static_cast<uint8_t>(rhs));
}

constexpr BufferUsage operator&(BufferUsage lhs, BufferUsage rhs) {
  return static_cast<BufferUsage>(static_cast<uint8_t>(lhs) &
                                  static_cast<uint8_t>(rhs));
}

constexpr BufferUsage &operator|=(BufferUsage &lhs, BufferUsage rhs) {
  lhs = lhs | rhs;
  return lhs;
}

constexpr bool hasBufferUsage(BufferUsage usage, BufferUsage flags) {
  return (usage & flags) != BufferUsage::None;
}

enum class TextureUsage : uint8_t {
  Sampled,
  Storage,
  Attachment,
  AttachmentSampled,
  InputAttachment,
  StorageSampled,
  Count
};

enum class Storage : uint8_t { Device, HostVisible, Memoryless, Count };

enum class TextureType : uint8_t { Texture2D, Texture3D, TextureCube, Count };

enum class SamplerFilter : uint8_t { Nearest, Linear, Count };

enum class SamplerMipMode : uint8_t { Disabled, Nearest, Linear, Count };

enum class SamplerWrapMode : uint8_t { Repeat, MirrorRepeat, Clamp, Count };

enum class IndexFormat : uint8_t { U16, U32, Count };

enum class CompareOp : uint8_t {
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  Equal,
  NotEqual,
  Always,
  Never,
  Count
};

enum class CullMode : uint8_t { None, Front, Back, Count };

enum class PolygonMode : uint8_t { Fill, Line, Count };

enum class Topology : uint8_t {
  Point,
  Line,
  LineStrip,
  Triangle,
  TriangleStrip,
  Patch,
  Count
};

enum class LoadOp : uint8_t { DontCare, Load, Clear };

enum class StoreOp : uint8_t { DontCare, Store, MsaaResolve };

enum class ResolveMode : uint8_t { None, SampleZero, Average, Min, Max, Count };

enum class VertexFormat : uint8_t {
  Float1,
  Float2,
  Float3,
  Float4,
  Int1,
  Int2,
  Int3,
  Int4,
  UInt1,
  UInt2,
  UInt3,
  UInt4,
  Byte4_Norm,
  UByte4_Norm,
  Short2,
  Short2_Norm,
  Count
};

enum class ShaderStage : uint8_t {
  Vertex,
  TessControl,
  TessEval,
  Geometry,
  Fragment,
  Compute,
  Task,
  Mesh,
  RayGen,
  AnyHit,
  ClosestHit,
  Miss,
  Intersection,
  Callable,
  Count
};

enum class GPUFeature : uint8_t {
  Meshlets,
  RayTracingClusters,
};

struct GpuMultisampleCapabilities {
  bool sample4Color = false;
  bool sample4Depth = false;
  bool sample8Color = false;
  bool sample8Depth = false;
  bool depthResolveMin = false;
  bool alphaToCoverage = false;
  bool sampleRateShading = false;
};

constexpr const char *ShaderStageToString(ShaderStage stage) {
  switch (stage) {
  case ShaderStage::Vertex:
    return "Vertex";
  case ShaderStage::TessControl:
    return "TessControl";
  case ShaderStage::TessEval:
    return "TessEval";
  case ShaderStage::Geometry:
    return "Geometry";
  case ShaderStage::Fragment:
    return "Fragment";
  case ShaderStage::Compute:
    return "Compute";
  case ShaderStage::Task:
    return "Task";
  case ShaderStage::Mesh:
    return "Mesh";
  case ShaderStage::RayGen:
    return "RayGen";
  case ShaderStage::AnyHit:
    return "AnyHit";
  case ShaderStage::ClosestHit:
    return "ClosestHit";
  case ShaderStage::Miss:
    return "Miss";
  case ShaderStage::Intersection:
    return "Intersection";
  case ShaderStage::Callable:
    return "Callable";
  case ShaderStage::Count:
    return "Invalid";
  default:
    return "Unknown";
  }
}

struct TextureDimensions {
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t depth = 1;
};

struct GeometryAllocationView {
  BufferHandle vertexBuffer{};
  uint64_t vertexByteOffset = 0;
  uint64_t vertexByteSize = 0;
  BufferHandle indexBuffer{};
  uint64_t indexByteOffset = 0;
  uint64_t indexByteSize = 0;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
};

struct GeometryPoolConfig {
  size_t vertexChunkSizeBytes = 64u * 1024u * 1024u;
  size_t indexChunkSizeBytes = 32u * 1024u * 1024u;
  uint64_t compactionCooldownFrames = 300;
  float compactionFragmentationThreshold = 0.3f;
  size_t compactionMinSavingsBytes = 16u * 1024u * 1024u;
  size_t compactionCopyBudgetBytesPerFrame = 32u * 1024u * 1024u;
};

struct MeshletLimits {
  uint32_t maxTaskWorkGroupTotalCount = 0;
  uint32_t maxTaskWorkGroupInvocations = 0;
  uint32_t maxTaskWorkGroupSizeX = 0;
  uint32_t maxTaskWorkGroupSizeY = 0;
  uint32_t maxTaskWorkGroupSizeZ = 0;
  uint32_t maxTaskWorkGroupCountX = 0;
  uint32_t maxTaskWorkGroupCountY = 0;
  uint32_t maxTaskWorkGroupCountZ = 0;
  uint32_t maxTaskPayloadBytes = 0;
  uint32_t maxTaskSharedMemoryBytes = 0;
  uint32_t maxTaskPayloadAndSharedMemoryBytes = 0;
  uint32_t maxMeshWorkGroupTotalCount = 0;
  uint32_t maxMeshWorkGroupInvocations = 0;
  uint32_t maxMeshWorkGroupSizeX = 0;
  uint32_t maxMeshWorkGroupSizeY = 0;
  uint32_t maxMeshWorkGroupSizeZ = 0;
  uint32_t maxMeshWorkGroupCountX = 0;
  uint32_t maxMeshWorkGroupCountY = 0;
  uint32_t maxMeshWorkGroupCountZ = 0;
  uint32_t maxMeshSharedMemoryBytes = 0;
  uint32_t maxMeshPayloadAndSharedMemoryBytes = 0;
  uint32_t maxMeshOutputMemoryBytes = 0;
  uint32_t maxMeshPayloadAndOutputMemoryBytes = 0;
  uint32_t maxMeshOutputComponents = 0;
  uint32_t meshOutputPerVertexGranularity = 0;
  uint32_t meshOutputPerPrimitiveGranularity = 0;
  uint32_t maxMeshOutputVertices = 0;
  uint32_t maxMeshOutputPrimitives = 0;
  uint32_t maxMeshOutputLayers = 0;
  uint32_t maxPreferredTaskWorkGroupInvocations = 0;
  uint32_t maxPreferredMeshWorkGroupInvocations = 0;
  bool prefersLocalInvocationVertexOutput = false;
  bool prefersLocalInvocationPrimitiveOutput = false;
  bool prefersCompactVertexOutput = false;
  bool prefersCompactPrimitiveOutput = false;
  bool supportsMeshDispatchIndirect = false;
  bool supportsMeshDispatchIndirectCount = false;
};

struct GPUDeviceCreateDesc {
  GeometryPoolConfig geometryPool{};
};

struct BufferCopyRegion {
  BufferHandle srcBuffer{};
  BufferHandle dstBuffer{};
  uint64_t srcOffset = 0;
  uint64_t dstOffset = 0;
  uint64_t size = 0;
};

struct BufferUpdate {
  BufferHandle buffer{};
  std::span<const std::byte> data{};
  size_t offset = 0u;
};

struct VertexAttribute {
  uint32_t location = 0;
  uint32_t binding = 0;
  uint32_t offset = 0;
  VertexFormat format = VertexFormat::Float3;
};

struct VertexBinding {
  uint32_t stride = 0;
};

struct VertexInput {
  std::span<const VertexAttribute> attributes{};
  std::span<const VertexBinding> bindings{};
};

struct DepthState {
  CompareOp compareOp = CompareOp::Less;
  bool isDepthWriteEnabled = true;
};

struct RasterPipelineState {
  CompareOp compareOp = CompareOp::Less;
  bool depthWrite = true;
  bool depthBiasEnable = false;
  int32_t depthBiasConstant = 0;
  float depthBiasSlope = 0.0f;
  float depthBiasClamp = 0.0f;
  constexpr bool
  operator==(const RasterPipelineState &) const noexcept = default;
};

[[nodiscard]] inline float canonicalRasterPipelineFloat(float value) noexcept {
  return value == 0.0f || !std::isfinite(value) ? 0.0f : value;
}

[[nodiscard]] inline int32_t
canonicalRasterPipelineDepthBiasConstant(float value) noexcept {
  if (!std::isfinite(value)) {
    return 0;
  }
  const double rounded = std::round(static_cast<double>(value));
  constexpr double kMin =
      static_cast<double>(std::numeric_limits<int32_t>::min());
  constexpr double kMax =
      static_cast<double>(std::numeric_limits<int32_t>::max());
  if (rounded <= kMin) {
    return std::numeric_limits<int32_t>::min();
  }
  if (rounded >= kMax) {
    return std::numeric_limits<int32_t>::max();
  }
  return static_cast<int32_t>(rounded);
}

[[nodiscard]] inline RasterPipelineState
canonicalRasterPipelineState(RasterPipelineState state) noexcept {
  if (!state.depthBiasEnable) {
    state.depthBiasConstant = 0;
    state.depthBiasSlope = 0.0f;
    state.depthBiasClamp = 0.0f;
    return state;
  }
  state.depthBiasSlope = canonicalRasterPipelineFloat(state.depthBiasSlope);
  state.depthBiasClamp = canonicalRasterPipelineFloat(state.depthBiasClamp);
  return state;
}

[[nodiscard]] inline RasterPipelineState
makeRasterPipelineState(DepthState depthState, bool depthBiasEnable = false,
                        float depthBiasConstant = 0.0f,
                        float depthBiasSlope = 0.0f,
                        float depthBiasClamp = 0.0f) noexcept {
  return canonicalRasterPipelineState(RasterPipelineState{
      .compareOp = depthState.compareOp,
      .depthWrite = depthState.isDepthWriteEnabled,
      .depthBiasEnable = depthBiasEnable,
      .depthBiasConstant =
          depthBiasEnable
              ? canonicalRasterPipelineDepthBiasConstant(depthBiasConstant)
              : 0,
      .depthBiasSlope = depthBiasSlope,
      .depthBiasClamp = depthBiasClamp,
  });
}

struct SpecializationEntry {
  uint32_t constantId = 0;
  uint32_t offset = 0;
  uint32_t size = 0;
};

struct SpecializationInfo {
  std::span<const SpecializationEntry> entries{};
  const void *data = nullptr;
  size_t dataSize = 0;
};

} // namespace nuri
