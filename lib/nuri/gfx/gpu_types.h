#pragma once

#include "nuri/pch.h"

namespace nuri {

// Handle types with index + generation for safety
struct BufferHandle {
  uint32_t index = 0;
  uint32_t generation = 0;
};

struct TextureHandle {
  uint32_t index = 0;
  uint32_t generation = 0;
};

struct ShaderHandle {
  uint32_t index = 0;
  uint32_t generation = 0;
};

struct RenderPipelineHandle {
  uint32_t index = 0;
  uint32_t generation = 0;
};

struct ComputePipelineHandle {
  uint32_t index = 0;
  uint32_t generation = 0;
};

constexpr bool isValid(BufferHandle h) { return h.generation != 0; }
constexpr bool isValid(TextureHandle h) { return h.generation != 0; }
constexpr bool isValid(ShaderHandle h) { return h.generation != 0; }
constexpr bool isValid(RenderPipelineHandle h) { return h.generation != 0; }
constexpr bool isValid(ComputePipelineHandle h) { return h.generation != 0; }

// GPU enums (LVK-free)
enum class Format : uint8_t {
  RGBA8_UNORM,
  RGBA8_SRGB,
  RGBA8_UINT,
  D32_FLOAT,
  Count
};

enum class BufferUsage : uint8_t { Vertex, Index, Uniform, Storage, Count };

enum class TextureUsage : uint8_t {
  Sampled,
  Storage,
  Attachment,
  InputAttachment,
  Count
};

enum class Storage : uint8_t { Device, HostVisible, Memoryless, Count };

enum class TextureType : uint8_t { Texture2D, Texture3D, TextureCube, Count };

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

enum class StoreOp : uint8_t { DontCare, Store };

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

struct TextureDimensions {
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t depth = 1;
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
