#pragma once
#include "nuri/defines.h"
#include "nuri/gfx/gpu_types.h"
#include <array>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

inline constexpr std::string_view kRayQueryShaderFeatureToken =
    "NURI_SHADER_FEATURE_RAY_QUERY";

struct BufferDesc {
  BufferUsage usage = BufferUsage::Vertex;
  Storage storage = Storage::Device;
  size_t size = 0;
  std::span<const std::byte> data{};
  bool immutable = false;
};

enum class AccelerationStructureGeometryFlags : uint8_t {
  None = 0,
  Opaque = 1u << 0u,
  NoDuplicateAnyHitInvocation = 1u << 1u,
};

constexpr AccelerationStructureGeometryFlags
operator|(AccelerationStructureGeometryFlags lhs,
          AccelerationStructureGeometryFlags rhs) noexcept {
  return static_cast<AccelerationStructureGeometryFlags>(
      static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

constexpr AccelerationStructureGeometryFlags
operator&(AccelerationStructureGeometryFlags lhs,
          AccelerationStructureGeometryFlags rhs) noexcept {
  return static_cast<AccelerationStructureGeometryFlags>(
      static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

[[nodiscard]] constexpr bool hasAccelerationStructureGeometryFlag(
    AccelerationStructureGeometryFlags flags,
    AccelerationStructureGeometryFlags requested) noexcept {
  return (flags & requested) != AccelerationStructureGeometryFlags::None;
}

struct AccelerationStructureTriangleGeometryDesc {
  BufferHandle vertexBuffer{};
  BufferHandle indexBuffer{};
  BufferHandle transformBuffer{};
  Format vertexFormat = Format::RGB32_FLOAT;
  IndexFormat indexFormat = IndexFormat::U32;
  uint64_t vertexByteOffset = 0u;
  uint64_t indexByteOffset = 0u;
  uint64_t transformByteOffset = 0u;
  uint32_t vertexStrideBytes = 12u;
  uint32_t vertexCount = 0u;
  uint32_t indexCount = 0u;
  AccelerationStructureGeometryFlags flags =
      AccelerationStructureGeometryFlags::None;
};

struct AccelerationStructureTransform {
  std::array<float, 12u> rowMajor3x4{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  constexpr bool
  operator==(const AccelerationStructureTransform &) const = default;
};

enum class AccelerationStructureInstanceFlags : uint8_t {
  None = 0,
  TriangleCullDisable = 1u << 0u,
  TriangleFrontCounterClockwise = 1u << 1u,
  ForceOpaque = 1u << 2u,
  ForceNonOpaque = 1u << 3u,
};

constexpr AccelerationStructureInstanceFlags
operator|(AccelerationStructureInstanceFlags lhs,
          AccelerationStructureInstanceFlags rhs) noexcept {
  return static_cast<AccelerationStructureInstanceFlags>(
      static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

struct AccelerationStructureInstanceDesc {
  AccelerationStructureTransform transform{};
  uint32_t customIndex = 0u;
  uint8_t mask = 0xffu;
  AccelerationStructureInstanceFlags flags =
      AccelerationStructureInstanceFlags::None;
  AccelerationStructureHandle bottomLevel{};
};

struct BlasCreateDesc {
  std::span<const AccelerationStructureTriangleGeometryDesc> geometries{};
  AccelerationStructureBuildFlags buildFlags =
      AccelerationStructureBuildFlags::PreferFastTrace;
};

struct TlasCreateDesc {
  uint32_t maxInstanceCount = 0u;
  AccelerationStructureBuildFlags buildFlags =
      AccelerationStructureBuildFlags::PreferFastTrace |
      AccelerationStructureBuildFlags::AllowUpdate;
};

enum class AccelerationStructureValidationReason : uint8_t {
  None = 0,
  Unsupported,
  ConflictingBuildPreference,
  EmptyGeometry,
  InvalidBuffer,
  UnsupportedVertexFormat,
  InvalidVertexRange,
  InvalidIndexRange,
  UnsupportedTransformBuffer,
  CapacityLimitExceeded,
  InvalidInstance,
};

struct AccelerationStructureValidationError {
  AccelerationStructureValidationReason reason =
      AccelerationStructureValidationReason::None;
  uint32_t itemIndex = 0u;
};

[[nodiscard]] NURI_API AccelerationStructureValidationError
validateBlasCreateDesc(const BlasCreateDesc &desc,
                       const RayTracingCapabilities &caps) noexcept;
[[nodiscard]] NURI_API AccelerationStructureValidationError
validateTlasCreateDesc(const TlasCreateDesc &desc,
                       const RayTracingCapabilities &caps) noexcept;
[[nodiscard]] NURI_API AccelerationStructureValidationError
validateAccelerationStructureInstances(
    std::span<const AccelerationStructureInstanceDesc> instances,
    uint32_t maximumCount) noexcept;

struct TextureDesc {
  TextureType type = TextureType::Texture2D;
  Format format = Format::RGBA8_UNORM;
  TextureDimensions dimensions{1, 1, 1};
  TextureUsage usage = TextureUsage::Sampled;
  Storage storage = Storage::Device;
  uint32_t numLayers = 1;
  uint32_t numSamples = 1;
  uint32_t numMipLevels = 1;
  std::span<const std::byte> data{};
  uint32_t dataNumMipLevels = 1;
  bool generateMipmaps = false;
};

struct SamplerDesc {
  SamplerFilter minFilter = SamplerFilter::Linear;
  SamplerFilter magFilter = SamplerFilter::Linear;
  SamplerMipMode mipMode = SamplerMipMode::Disabled;
  SamplerWrapMode wrapU = SamplerWrapMode::Repeat;
  SamplerWrapMode wrapV = SamplerWrapMode::Repeat;
  SamplerWrapMode wrapW = SamplerWrapMode::Repeat;
  float mipLodMin = 0.0f;
  float mipLodMax = 15.0f;
  float mipLodBias = 0.0f;
  uint8_t maxAnisotropy = 1;
  bool depthCompareEnabled = false;
  CompareOp depthCompareOp = CompareOp::LessEqual;
  constexpr bool operator==(const SamplerDesc &) const noexcept = default;
};

struct ShaderDesc {
  std::string_view moduleName{};
  std::string_view source{};
  ShaderStage stage = ShaderStage::Vertex;
};

struct RenderPipelineDesc {
  VertexInput vertexInput{};
  ShaderHandle vertexShader{};
  ShaderHandle tessControlShader{};
  ShaderHandle tessEvalShader{};
  ShaderHandle geometryShader{};
  ShaderHandle fragmentShader{};
  std::vector<Format> colorFormats{Format::RGBA8_UNORM};
  uint32_t colorAttachmentCount = 1;
  Format depthFormat = Format::Count;
  CullMode cullMode = CullMode::Back;
  PolygonMode polygonMode = PolygonMode::Fill;
  Topology topology = Topology::Triangle;
  uint32_t patchControlPoints = 0;
  uint32_t numSamples = 1u;
  float minSampleShading = 0.0f;
  bool alphaToCoverageEnabled = false;
  bool blendEnabled = false;
  SpecializationInfo specInfo{};
  RasterPipelineState rasterState{};
  std::span<const RasterPipelineState> prewarmRasterStates{};
};

struct ComputePipelineDesc {
  ShaderHandle computeShader{};
  SpecializationInfo specInfo{};
};

struct MeshletPipelineDesc {
  ShaderHandle taskShader{};
  ShaderHandle meshShader{};
  ShaderHandle fragmentShader{};
  std::vector<Format> colorFormats{Format::RGBA8_UNORM};
  uint32_t colorAttachmentCount = 1;
  Format depthFormat = Format::Count;
  CullMode cullMode = CullMode::Back;
  PolygonMode polygonMode = PolygonMode::Fill;
  uint32_t numSamples = 1u;
  float minSampleShading = 0.0f;
  bool alphaToCoverageEnabled = false;
  bool blendEnabled = false;
  SpecializationInfo specInfo{};
  RasterPipelineState rasterState{};
  std::span<const RasterPipelineState> prewarmRasterStates{};
};

} // namespace nuri
