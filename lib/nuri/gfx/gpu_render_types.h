#pragma once
#include "nuri/gfx/frame/external_temporal_provider.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_types.h"
#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
namespace nuri {

enum class RenderGraphAccessMode : uint8_t {
  None = 0,
  Read = 1u << 0u,
  Write = 1u << 1u,
};

struct NURI_API RenderGraphImportedBufferUse {
  BufferHandle buffer{};
  RenderGraphAccessMode access = RenderGraphAccessMode::Read;
};
struct NURI_API RenderGraphImportedTextureUse {
  TextureHandle texture{};
  RenderGraphAccessMode access = RenderGraphAccessMode::Read;
};

[[nodiscard]] constexpr RenderGraphAccessMode
operator|(RenderGraphAccessMode lhs, RenderGraphAccessMode rhs) {
  return static_cast<RenderGraphAccessMode>(static_cast<uint8_t>(lhs) |
                                            static_cast<uint8_t>(rhs));
}

[[nodiscard]] constexpr bool hasAccessFlag(RenderGraphAccessMode mode,
                                           RenderGraphAccessMode flag) {
  return (static_cast<uint8_t>(mode) & static_cast<uint8_t>(flag)) != 0u;
}

constexpr size_t kMaxDependencyResources = 32;
constexpr size_t kMaxDependencyBuffers = kMaxDependencyResources;
constexpr size_t kMaxMeshDispatchDependencyResources = 4096;
constexpr size_t kMaxDependencyTextures = 1024;

struct Viewport {
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float minDepth = 0.0f;
  float maxDepth = 1.0f;
};

struct RectU32 {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t width = 0;
  uint32_t height = 0;
};

struct ClearColor {
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  float a = 1.0f;
};

struct AttachmentColor {
  LoadOp loadOp = LoadOp::Clear;
  StoreOp storeOp = StoreOp::Store;
  ResolveMode resolveMode = ResolveMode::Average;
  ClearColor clearColor{};
};

struct AttachmentDepth {
  LoadOp loadOp = LoadOp::Clear;
  StoreOp storeOp = StoreOp::Store;
  ResolveMode resolveMode = ResolveMode::Min;
  float clearDepth = 1.0f;
  uint32_t clearStencil = 0;
};

struct DispatchSize {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

struct PushConstantTextureBinding {
  uint32_t byteOffset = UINT32_MAX;
  TextureHandle texture{};
  uint32_t graphTextureResourceIndex = UINT32_MAX;
  RenderGraphAccessMode access = RenderGraphAccessMode::Read;
};

struct ComputeDispatchItem {
  ComputePipelineHandle pipeline{};
  RayQueryBindingHandle rayQueryBinding{};
  DispatchSize dispatch{};
  std::span<const std::byte> pushConstants{};
  std::span<const PushConstantTextureBinding> pushConstantTextureBindings{};
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
};

enum class GpuTimingScope : uint8_t {
  None = 0,
  Shadow = 1,
  ShadowDepth = 2,
  ShadowSdsm = 3,
  SceneColorDownsample = 4,
  Transmission = 5,
  TemporalAAResolve = 6,
  TemporalAADebug = 7,
  SpatialAA = 8,
  Opaque = 9,
  MsaaResolve = 10,
  GTAO = 11,
  HDRPostProcess = 12,
  Skybox = 13,
  Velocity = 14,
  ReactiveMask = 15,
  TemporalAACopyBack = 16,
  GTAOTemporal = 17,
  WholeFrame = 18,
  RayTracingScene = 19,
  RayTracingBLAS = 20,
  RayTracingTLAS = 21,
  DDGI = 22,
  DDGITrace = 23,
  DDGIUpdate = 24,
  DDGIRelocateClassify = 25,
  OpaqueDepth = 26,
  OpaqueNormal = 27,
  OpaqueMain = 28,
  GTAOPrefilterEdges = 29,
  GTAOMain = 30,
  GTAODenoise = 31,
  GTAOUpscale = 32,
  DDGIIrradianceUpdate = 33,
  DDGIDistanceUpdate = 34,
  DDGIReadback = 35,
  DDGIOpaqueSurfaceCache = 36,
  Count = 37,
};

[[nodiscard]] constexpr uint64_t
gpuTimingScopeToBit(GpuTimingScope scope) noexcept {
  return scope == GpuTimingScope::None
             ? 0ull
             : (1ull << (static_cast<uint8_t>(scope) - 1u));
}

struct GpuTimingScopeDesc {
  GpuTimingScope scope = GpuTimingScope::None;
  GpuTimingScope parent = GpuTimingScope::None;
  std::string_view name{};
};

inline constexpr auto kGpuTimingScopeDescs = std::to_array<GpuTimingScopeDesc>({
    {GpuTimingScope::Shadow, GpuTimingScope::None, "Shadow"},
    {GpuTimingScope::ShadowDepth, GpuTimingScope::Shadow, "ShadowDepth"},
    {GpuTimingScope::ShadowSdsm, GpuTimingScope::Shadow, "ShadowSdsm"},
    {GpuTimingScope::SceneColorDownsample, GpuTimingScope::None,
     "SceneColorDownsample"},
    {GpuTimingScope::Transmission, GpuTimingScope::None, "Transmission"},
    {GpuTimingScope::TemporalAAResolve, GpuTimingScope::None,
     "TemporalAAResolve"},
    {GpuTimingScope::TemporalAADebug, GpuTimingScope::None, "TemporalAADebug"},
    {GpuTimingScope::SpatialAA, GpuTimingScope::None, "SpatialAA"},
    {GpuTimingScope::Opaque, GpuTimingScope::None, "Opaque"},
    {GpuTimingScope::MsaaResolve, GpuTimingScope::None, "MsaaResolve"},
    {GpuTimingScope::GTAO, GpuTimingScope::None, "GTAO"},
    {GpuTimingScope::HDRPostProcess, GpuTimingScope::None, "HDRPostProcess"},
    {GpuTimingScope::Skybox, GpuTimingScope::None, "Skybox"},
    {GpuTimingScope::Velocity, GpuTimingScope::Opaque, "Velocity"},
    {GpuTimingScope::ReactiveMask, GpuTimingScope::Opaque, "ReactiveMask"},
    {GpuTimingScope::TemporalAACopyBack, GpuTimingScope::TemporalAAResolve,
     "TemporalAACopyBack"},
    {GpuTimingScope::GTAOTemporal, GpuTimingScope::GTAO, "GTAOTemporal"},
    {GpuTimingScope::WholeFrame, GpuTimingScope::None, "WholeFrame"},
    {GpuTimingScope::RayTracingScene, GpuTimingScope::None, "RayTracingScene"},
    {GpuTimingScope::RayTracingBLAS, GpuTimingScope::RayTracingScene,
     "RayTracingBLAS"},
    {GpuTimingScope::RayTracingTLAS, GpuTimingScope::RayTracingScene,
     "RayTracingTLAS"},
    {GpuTimingScope::DDGI, GpuTimingScope::None, "DDGI"},
    {GpuTimingScope::DDGITrace, GpuTimingScope::DDGI, "DDGITrace"},
    {GpuTimingScope::DDGIUpdate, GpuTimingScope::DDGI, "DDGIUpdate"},
    {GpuTimingScope::DDGIRelocateClassify, GpuTimingScope::DDGI,
     "DDGIRelocateClassify"},
    {GpuTimingScope::OpaqueDepth, GpuTimingScope::Opaque, "OpaqueDepth"},
    {GpuTimingScope::OpaqueNormal, GpuTimingScope::Opaque, "OpaqueNormal"},
    {GpuTimingScope::OpaqueMain, GpuTimingScope::Opaque, "OpaqueMain"},
    {GpuTimingScope::GTAOPrefilterEdges, GpuTimingScope::GTAO,
     "GTAOPrefilterEdges"},
    {GpuTimingScope::GTAOMain, GpuTimingScope::GTAO, "GTAOMain"},
    {GpuTimingScope::GTAODenoise, GpuTimingScope::GTAO, "GTAODenoise"},
    {GpuTimingScope::GTAOUpscale, GpuTimingScope::GTAO, "GTAOUpscale"},
    {GpuTimingScope::DDGIIrradianceUpdate, GpuTimingScope::DDGIUpdate,
     "DDGIIrradianceUpdate"},
    {GpuTimingScope::DDGIDistanceUpdate, GpuTimingScope::DDGIUpdate,
     "DDGIDistanceUpdate"},
    {GpuTimingScope::DDGIReadback, GpuTimingScope::None, "DDGIReadback"},
    {GpuTimingScope::DDGIOpaqueSurfaceCache, GpuTimingScope::Opaque,
     "DDGIOpaqueSurfaceCache"},
});

[[nodiscard]] constexpr const GpuTimingScopeDesc *
gpuTimingScopeDesc(GpuTimingScope scope) noexcept {
  const size_t index = static_cast<size_t>(scope);
  return index > 0u && index < static_cast<size_t>(GpuTimingScope::Count)
             ? &kGpuTimingScopeDescs[index - 1u]
             : nullptr;
}

[[nodiscard]] constexpr GpuTimingScope
gpuTimingParentScope(GpuTimingScope scope) noexcept {
  const GpuTimingScopeDesc *desc = gpuTimingScopeDesc(scope);
  return desc != nullptr ? desc->parent : GpuTimingScope::None;
}

struct GpuTimingSample {
  float timeMs = 0.0f;
  uint64_t sourceFrameIndex = std::numeric_limits<uint64_t>::max();
};

struct GpuTimingReport {
  std::array<GpuTimingSample, static_cast<size_t>(GpuTimingScope::Count)>
      samples{};
  bool opaquePipelineStatisticsRequested = false;
  bool opaquePipelineStatisticsAvailable = false;
  uint64_t opaquePipelineStatisticsSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t opaqueInputAssemblyVertices = 0u;
  uint64_t opaqueInputAssemblyPrimitives = 0u;
  uint64_t opaqueClippingInvocations = 0u;
  uint64_t opaqueClippingPrimitives = 0u;
  uint64_t opaqueFragmentShaderInvocations = 0u;
  uint64_t availableScopeMask = 0u;

  struct PassTiming {
    std::string debugName{};
    uint64_t sourceFrameIndex = std::numeric_limits<uint64_t>::max();
    float timeMs = 0.0f;
  };
  std::vector<PassTiming> passTimings{};

  [[nodiscard]] constexpr GpuTimingSample &
  operator[](GpuTimingScope scope) noexcept {
    return samples[static_cast<size_t>(scope)];
  }
  [[nodiscard]] constexpr const GpuTimingSample &
  operator[](GpuTimingScope scope) const noexcept {
    return samples[static_cast<size_t>(scope)];
  }
};

[[nodiscard]] constexpr bool hasGpuTimingScope(const GpuTimingReport &report,
                                               GpuTimingScope scope) noexcept {
  return (report.availableScopeMask & gpuTimingScopeToBit(scope)) != 0u;
}

[[nodiscard]] constexpr bool
gpuTimingScopeContributesToScopedSum(const GpuTimingReport &report,
                                     GpuTimingScope scope) noexcept {
  if (scope == GpuTimingScope::WholeFrame) {
    return false;
  }
  const GpuTimingScope parent = gpuTimingParentScope(scope);
  return parent == GpuTimingScope::None || !hasGpuTimingScope(report, parent);
}

[[nodiscard]] constexpr uint64_t
gpuTimingScopeSourceFrame(const GpuTimingReport &report,
                          GpuTimingScope scope) noexcept {
  return report[scope].sourceFrameIndex;
}

inline void mergeGpuTimingReportScope(GpuTimingReport &dst,
                                      const GpuTimingReport &src,
                                      GpuTimingScope scope) {
  if (!hasGpuTimingScope(src, scope)) {
    return;
  }
  dst[scope] = src[scope];
  dst.availableScopeMask |= gpuTimingScopeToBit(scope);
}

inline void mergeGpuTimingReportScopes(GpuTimingReport &dst,
                                       const GpuTimingReport &src) {
  for (const GpuTimingScopeDesc &desc : kGpuTimingScopeDescs) {
    mergeGpuTimingReportScope(dst, src, desc.scope);
  }
}

enum class GraphicsBarrierResourceKind : uint8_t {
  Texture = 0,
  Buffer = 1,
  AccelerationStructure = 2,
};

using GraphicsBarrierAccessMode = RenderGraphAccessMode;

struct GraphicsTextureSubresourceRange {
  uint32_t firstMip = 0u;
  uint32_t mipCount = UINT32_MAX;
  uint32_t firstLayer = 0u;
  uint32_t layerCount = UINT32_MAX;

  [[nodiscard]] bool
  operator==(const GraphicsTextureSubresourceRange &) const noexcept = default;
};

[[nodiscard]] constexpr bool
hasGraphicsBarrierAccessFlag(GraphicsBarrierAccessMode mode,
                             GraphicsBarrierAccessMode flag) {
  return hasAccessFlag(mode, flag);
}

enum class GraphicsBarrierState : uint8_t {
  Unknown = 0,
  Read = 1,
  Write = 2,
  Attachment = 3,
  Present = 4,
  AccelerationStructureBuildRead = 5,
  AccelerationStructureBuildWrite = 6,
  RayQueryRead = 7,
  AccelerationStructureBuildInput = 8,
};

union GraphicsBarrierResourceStorage {
  constexpr GraphicsBarrierResourceStorage() noexcept : texture{} {}
  TextureHandle texture;
  BufferHandle buffer;
  AccelerationStructureHandle accelerationStructure;
};

struct GraphicsBarrierRecord {
  GraphicsBarrierResourceKind resourceKind =
      GraphicsBarrierResourceKind::Texture;
  GraphicsBarrierResourceStorage resource{};
  GraphicsBarrierAccessMode beforeAccess = GraphicsBarrierAccessMode::None;
  GraphicsBarrierAccessMode afterAccess = GraphicsBarrierAccessMode::None;
  GraphicsBarrierState beforeState = GraphicsBarrierState::Unknown;
  GraphicsBarrierState afterState = GraphicsBarrierState::Unknown;
  GraphicsTextureSubresourceRange subresources{};
  [[nodiscard]] static constexpr GraphicsBarrierRecord ForTexture(
      TextureHandle textureHandle,
      GraphicsBarrierAccessMode beforeAccessMode =
          GraphicsBarrierAccessMode::None,
      GraphicsBarrierAccessMode afterAccessMode =
          GraphicsBarrierAccessMode::None,
      GraphicsBarrierState beforeBarrierState = GraphicsBarrierState::Unknown,
      GraphicsBarrierState afterBarrierState = GraphicsBarrierState::Unknown,
      GraphicsTextureSubresourceRange textureSubresources = {}) noexcept {
    GraphicsBarrierRecord record{};
    record.setTextureHandle(textureHandle);
    record.beforeAccess = beforeAccessMode;
    record.afterAccess = afterAccessMode;
    record.beforeState = beforeBarrierState;
    record.afterState = afterBarrierState;
    record.subresources = textureSubresources;
    return record;
  }
  [[nodiscard]] static constexpr GraphicsBarrierRecord ForBuffer(
      BufferHandle bufferHandle,
      GraphicsBarrierAccessMode beforeAccessMode =
          GraphicsBarrierAccessMode::None,
      GraphicsBarrierAccessMode afterAccessMode =
          GraphicsBarrierAccessMode::None,
      GraphicsBarrierState beforeBarrierState = GraphicsBarrierState::Unknown,
      GraphicsBarrierState afterBarrierState =
          GraphicsBarrierState::Unknown) noexcept {
    GraphicsBarrierRecord record{};
    record.setBufferHandle(bufferHandle);
    record.beforeAccess = beforeAccessMode;
    record.afterAccess = afterAccessMode;
    record.beforeState = beforeBarrierState;
    record.afterState = afterBarrierState;
    return record;
  }
  [[nodiscard]] static constexpr GraphicsBarrierRecord ForAccelerationStructure(
      AccelerationStructureHandle accelerationStructureHandle,
      GraphicsBarrierAccessMode beforeAccessMode =
          GraphicsBarrierAccessMode::None,
      GraphicsBarrierAccessMode afterAccessMode =
          GraphicsBarrierAccessMode::None,
      GraphicsBarrierState beforeBarrierState = GraphicsBarrierState::Unknown,
      GraphicsBarrierState afterBarrierState =
          GraphicsBarrierState::Unknown) noexcept {
    GraphicsBarrierRecord record{};
    record.setAccelerationStructureHandle(accelerationStructureHandle);
    record.beforeAccess = beforeAccessMode;
    record.afterAccess = afterAccessMode;
    record.beforeState = beforeBarrierState;
    record.afterState = afterBarrierState;
    return record;
  }
  constexpr void setTextureHandle(TextureHandle textureHandle) noexcept {
    resourceKind = GraphicsBarrierResourceKind::Texture;
    resource.texture = textureHandle;
  }
  constexpr void setBufferHandle(BufferHandle bufferHandle) noexcept {
    resourceKind = GraphicsBarrierResourceKind::Buffer;
    resource.buffer = bufferHandle;
  }
  constexpr void setAccelerationStructureHandle(
      AccelerationStructureHandle accelerationStructureHandle) noexcept {
    resourceKind = GraphicsBarrierResourceKind::AccelerationStructure;
    resource.accelerationStructure = accelerationStructureHandle;
  }
  [[nodiscard]] constexpr bool isTexture() const noexcept {
    return resourceKind == GraphicsBarrierResourceKind::Texture;
  }
  [[nodiscard]] constexpr TextureHandle textureHandle() const noexcept {
    return isTexture() ? resource.texture : TextureHandle{};
  }
  [[nodiscard]] constexpr BufferHandle bufferHandle() const noexcept {
    return resourceKind == GraphicsBarrierResourceKind::Buffer ? resource.buffer
                                                               : BufferHandle{};
  }
  [[nodiscard]] constexpr AccelerationStructureHandle
  accelerationStructureHandle() const noexcept {
    return resourceKind == GraphicsBarrierResourceKind::AccelerationStructure
               ? resource.accelerationStructure
               : AccelerationStructureHandle{};
  }
};

enum class DrawCommandType : uint8_t {
  Direct,
  IndexedIndirect,
  IndexedIndirectCount,
};

struct DrawItem {
  DrawCommandType command = DrawCommandType::Direct;
  RenderPipelineHandle pipeline{};
  BufferHandle vertexBuffer{};
  uint64_t vertexBufferOffset = 0;
  BufferHandle indexBuffer{};
  uint64_t indexBufferOffset = 0;
  BufferHandle indirectBuffer{};
  uint64_t indirectBufferOffset = 0;
  BufferHandle indirectCountBuffer{};
  uint64_t indirectCountBufferOffset = 0;
  uint32_t indirectDrawCount = 0;
  uint32_t indirectStride = 0;
  IndexFormat indexFormat = IndexFormat::U32;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  uint32_t instanceCount = 1;
  uint32_t firstVertex = 0;
  uint32_t firstIndex = 0;
  int32_t vertexOffset = 0;
  uint32_t firstInstance = 0;
  bool useScissor = false;
  RectU32 scissor{};
  bool useDepthState = false;
  DepthState depthState{};
  bool alphaMasked = false;
  bool depthBiasEnable = false;
  float depthBiasConstant = 0.0f;
  float depthBiasSlope = 0.0f;
  float depthBiasClamp = 0.0f;
  std::span<const std::byte> pushConstants{};
  std::span<const PushConstantTextureBinding> pushConstantTextureBindings{};
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
};

enum class MeshDispatchCommandType : uint8_t {
  Direct,
  Indirect,
  IndirectCount,
};

struct MeshDispatchItem {
  MeshDispatchCommandType command = MeshDispatchCommandType::Direct;
  MeshletPipelineHandle pipeline{};
  BufferHandle indirectBuffer{};
  uint64_t indirectBufferOffset = 0;
  BufferHandle indirectCountBuffer{};
  uint64_t indirectCountBufferOffset = 0;
  uint32_t indirectDispatchCount = 0;
  uint32_t groupsX = 1;
  uint32_t groupsY = 1;
  uint32_t groupsZ = 1;
  bool useScissor = false;
  RectU32 scissor{};
  bool useDepthState = false;
  DepthState depthState{};
  bool depthBiasEnable = false;
  float depthBiasConstant = 0.0f;
  float depthBiasSlope = 0.0f;
  float depthBiasClamp = 0.0f;
  std::span<const std::byte> pushConstants{};
  std::span<const PushConstantTextureBinding> pushConstantTextureBindings{};
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
};

struct TextureCopyItem {
  TextureHandle sourceTexture{};
  TextureHandle destinationTexture{};
  uint32_t sourceX = 0;
  uint32_t sourceY = 0;
  uint32_t destinationX = 0;
  uint32_t destinationY = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t sourceMipLevel = 0;
  uint32_t destinationMipLevel = 0;
  uint32_t sourceLayer = 0;
  uint32_t destinationLayer = 0;
};

struct BuildBlasItem {
  AccelerationStructureHandle destination{};
  std::span<const AccelerationStructureTriangleGeometryDesc> geometries{};
};

struct UpdateBlasItem {
  AccelerationStructureHandle destination{};
  std::span<const AccelerationStructureTriangleGeometryDesc> geometries{};
};

struct BuildTlasItem {
  AccelerationStructureHandle destination{};
  std::span<const AccelerationStructureInstanceDesc> instances{};
};

struct UpdateTlasItem {
  AccelerationStructureHandle destination{};
  std::span<const AccelerationStructureInstanceDesc> instances{};
};

using AccelerationStructureBuildCommand =
    std::variant<BuildBlasItem, UpdateBlasItem, BuildTlasItem, UpdateTlasItem>;

struct AccelerationStructureBuildItem {
  AccelerationStructureBuildCommand command{};
};

enum class RenderPassExecutionMode : uint8_t {
  Graphics = 0,
  ComputeOnly = 1,
  CopyOnly = 2,
  ExternalTemporal = 3,
  AccelerationStructureBuild = 4,
};

struct RenderPass {
  RenderPassExecutionMode executionMode = RenderPassExecutionMode::Graphics;
  AttachmentColor color;
  TextureHandle colorTexture{};
  TextureHandle colorResolveTexture{};
  bool hasColorAttachment = true;
  AttachmentDepth depth;
  TextureHandle depthTexture{};
  TextureHandle depthResolveTexture{};
  bool useViewport = false;
  Viewport viewport{};
  std::span<const ComputeDispatchItem> preDispatches{};
  std::span<const SamplerHandle> recordingSamplers{};
  std::span<const DrawItem> draws{};
  std::span<const MeshDispatchItem> meshDispatches{};
  std::span<const BufferCopyRegion> bufferCopies{};
  std::span<const TextureCopyItem> textureCopies{};
  std::span<const AccelerationStructureBuildItem> accelerationStructureBuilds{};
  ExternalTemporalDispatchItem externalTemporalDispatch{};
  bool drawBuffersPreResolved = false;
  GpuTimingScope gpuTimingScope = GpuTimingScope::None;
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
};

struct GraphicsRecordingStep {
  std::span<const GraphicsBarrierRecord> barriers{};
  const RenderPass *pass = nullptr;
};

struct GraphicsRecordingReferences {
  std::span<const BufferHandle> buffers{};
  std::span<const TextureHandle> textures{};
  std::span<const SamplerHandle> samplers{};
  std::span<const AccelerationStructureHandle> accelerationStructures{};
  std::span<const RenderPipelineHandle> renderPipelines{};
  std::span<const ComputePipelineHandle> computePipelines{};
  std::span<const MeshletPipelineHandle> meshletPipelines{};
  std::span<const RayQueryBindingHandle> rayQueryBindings{};
};

struct SubmitBatchMeta {
  uint32_t commandBufferOffset = 0u;
  uint32_t commandBufferCount = 0u;
  bool presentsFrameOutput = false;
};

struct SubmittedGraphicsFrame {
  SubmissionHandle submission{};
  std::string presentationError{};
};

struct RenderFrame {
  std::span<const RenderPass> passes{};
};

} // namespace nuri
