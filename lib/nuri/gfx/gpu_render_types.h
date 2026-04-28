#pragma once

#include "nuri/gfx/gpu_types.h"

#include <cstddef>
#include <limits>
#include <span>
#include <string_view>

namespace nuri {

// Public render-graph contract: a submit item may depend on up to 12 resources
// of each dependency kind.
// Backends can impose stricter limits, but higher-level layers should respect
// this cap to avoid backend-specific failures.
constexpr size_t kMaxDependencyResources = 12;
constexpr size_t kMaxDependencyBuffers = kMaxDependencyResources;

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
  ClearColor clearColor{};
};

struct AttachmentDepth {
  LoadOp loadOp = LoadOp::Clear;
  StoreOp storeOp = StoreOp::Store;
  float clearDepth = 1.0f;
  uint32_t clearStencil = 0;
};

struct DispatchSize {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

struct ComputeDispatchItem {
  ComputePipelineHandle pipeline{};
  DispatchSize dispatch{};
  std::span<const std::byte> pushConstants{};
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const TextureHandle> dependencyTextures{};
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
};

[[nodiscard]] constexpr uint32_t
gpuTimingScopeToBit(GpuTimingScope scope) noexcept {
  return scope == GpuTimingScope::None
             ? 0u
             : (1u << (static_cast<uint8_t>(scope) - 1u));
}

constexpr uint32_t kGpuTimingScopeShadowBit =
    gpuTimingScopeToBit(GpuTimingScope::Shadow);
constexpr uint32_t kGpuTimingScopeShadowDepthBit =
    gpuTimingScopeToBit(GpuTimingScope::ShadowDepth);
constexpr uint32_t kGpuTimingScopeShadowSdsmBit =
    gpuTimingScopeToBit(GpuTimingScope::ShadowSdsm);
constexpr uint32_t kGpuTimingScopeSceneColorDownsampleBit =
    gpuTimingScopeToBit(GpuTimingScope::SceneColorDownsample);
constexpr uint32_t kGpuTimingScopeTransmissionBit =
    gpuTimingScopeToBit(GpuTimingScope::Transmission);
constexpr uint32_t kGpuTimingScopeTemporalAAResolveBit =
    gpuTimingScopeToBit(GpuTimingScope::TemporalAAResolve);
constexpr uint32_t kGpuTimingScopeTemporalAADebugBit =
    gpuTimingScopeToBit(GpuTimingScope::TemporalAADebug);
constexpr uint32_t kGpuTimingScopeSpatialAABit =
    gpuTimingScopeToBit(GpuTimingScope::SpatialAA);

struct GpuTimingReport {
  uint64_t shadowSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t shadowDepthSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t shadowSdsmSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t sceneColorDownsampleSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t transmissionSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t temporalAAResolveSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t temporalAADebugSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t spatialAASourceFrameIndex = std::numeric_limits<uint64_t>::max();
  float shadowTimeMs = 0.0f;
  float shadowDepthTimeMs = 0.0f;
  float shadowSdsmTimeMs = 0.0f;
  float sceneColorDownsampleTimeMs = 0.0f;
  float transmissionTimeMs = 0.0f;
  float temporalAAResolveTimeMs = 0.0f;
  float temporalAADebugTimeMs = 0.0f;
  float spatialAATimeMs = 0.0f;
  uint32_t availableScopeMask = 0u;
};

[[nodiscard]] constexpr bool hasGpuTimingScope(const GpuTimingReport &report,
                                               GpuTimingScope scope) noexcept {
  return (report.availableScopeMask & gpuTimingScopeToBit(scope)) != 0u;
}

inline void mergeGpuTimingReportScopes(GpuTimingReport &dst,
                                       const GpuTimingReport &src) {
  if (hasGpuTimingScope(src, GpuTimingScope::Shadow)) {
    dst.shadowTimeMs = src.shadowTimeMs;
    dst.shadowSourceFrameIndex = src.shadowSourceFrameIndex;
    dst.availableScopeMask |= gpuTimingScopeToBit(GpuTimingScope::Shadow);
  }
  if (hasGpuTimingScope(src, GpuTimingScope::ShadowDepth)) {
    dst.shadowDepthTimeMs = src.shadowDepthTimeMs;
    dst.shadowDepthSourceFrameIndex = src.shadowDepthSourceFrameIndex;
    dst.availableScopeMask |= gpuTimingScopeToBit(GpuTimingScope::ShadowDepth);
  }
  if (hasGpuTimingScope(src, GpuTimingScope::ShadowSdsm)) {
    dst.shadowSdsmTimeMs = src.shadowSdsmTimeMs;
    dst.shadowSdsmSourceFrameIndex = src.shadowSdsmSourceFrameIndex;
    dst.availableScopeMask |= gpuTimingScopeToBit(GpuTimingScope::ShadowSdsm);
  }
  if (hasGpuTimingScope(src, GpuTimingScope::SceneColorDownsample)) {
    dst.sceneColorDownsampleTimeMs = src.sceneColorDownsampleTimeMs;
    dst.sceneColorDownsampleSourceFrameIndex =
        src.sceneColorDownsampleSourceFrameIndex;
    dst.availableScopeMask |=
        gpuTimingScopeToBit(GpuTimingScope::SceneColorDownsample);
  }
  if (hasGpuTimingScope(src, GpuTimingScope::Transmission)) {
    dst.transmissionTimeMs = src.transmissionTimeMs;
    dst.transmissionSourceFrameIndex = src.transmissionSourceFrameIndex;
    dst.availableScopeMask |= gpuTimingScopeToBit(GpuTimingScope::Transmission);
  }
  if (hasGpuTimingScope(src, GpuTimingScope::TemporalAAResolve)) {
    dst.temporalAAResolveTimeMs = src.temporalAAResolveTimeMs;
    dst.temporalAAResolveSourceFrameIndex =
        src.temporalAAResolveSourceFrameIndex;
    dst.availableScopeMask |=
        gpuTimingScopeToBit(GpuTimingScope::TemporalAAResolve);
  }
  if (hasGpuTimingScope(src, GpuTimingScope::TemporalAADebug)) {
    dst.temporalAADebugTimeMs = src.temporalAADebugTimeMs;
    dst.temporalAADebugSourceFrameIndex = src.temporalAADebugSourceFrameIndex;
    dst.availableScopeMask |=
        gpuTimingScopeToBit(GpuTimingScope::TemporalAADebug);
  }
  if (hasGpuTimingScope(src, GpuTimingScope::SpatialAA)) {
    dst.spatialAATimeMs = src.spatialAATimeMs;
    dst.spatialAASourceFrameIndex = src.spatialAASourceFrameIndex;
    dst.availableScopeMask |= gpuTimingScopeToBit(GpuTimingScope::SpatialAA);
  }
}

enum class GraphicsBarrierResourceKind : uint8_t {
  Texture = 0,
  Buffer = 1,
};

enum class GraphicsBarrierAccessMode : uint8_t {
  None = 0,
  Read = 1u << 0u,
  Write = 1u << 1u,
};

[[nodiscard]] constexpr GraphicsBarrierAccessMode
operator|(GraphicsBarrierAccessMode lhs, GraphicsBarrierAccessMode rhs) {
  return static_cast<GraphicsBarrierAccessMode>(static_cast<uint8_t>(lhs) |
                                                static_cast<uint8_t>(rhs));
}

[[nodiscard]] constexpr GraphicsBarrierAccessMode
operator&(GraphicsBarrierAccessMode lhs, GraphicsBarrierAccessMode rhs) {
  return static_cast<GraphicsBarrierAccessMode>(static_cast<uint8_t>(lhs) &
                                                static_cast<uint8_t>(rhs));
}

constexpr GraphicsBarrierAccessMode &operator|=(GraphicsBarrierAccessMode &lhs,
                                                GraphicsBarrierAccessMode rhs) {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool
hasGraphicsBarrierAccessFlag(GraphicsBarrierAccessMode mode,
                             GraphicsBarrierAccessMode flag) {
  return (static_cast<uint8_t>(mode) & static_cast<uint8_t>(flag)) != 0u;
}

enum class GraphicsBarrierState : uint8_t {
  Unknown = 0,
  Read = 1,
  Write = 2,
  Attachment = 3,
  Present = 4,
};

union GraphicsBarrierResourceStorage {
  constexpr GraphicsBarrierResourceStorage() noexcept : texture{} {}

  TextureHandle texture;
  BufferHandle buffer;
};

struct GraphicsBarrierRecord {
  GraphicsBarrierResourceKind resourceKind =
      GraphicsBarrierResourceKind::Texture;
  GraphicsBarrierResourceStorage resource{};
  GraphicsBarrierAccessMode beforeAccess = GraphicsBarrierAccessMode::None;
  GraphicsBarrierAccessMode afterAccess = GraphicsBarrierAccessMode::None;
  GraphicsBarrierState beforeState = GraphicsBarrierState::Unknown;
  GraphicsBarrierState afterState = GraphicsBarrierState::Unknown;

  [[nodiscard]] static constexpr GraphicsBarrierRecord ForTexture(
      TextureHandle textureHandle,
      GraphicsBarrierAccessMode beforeAccessMode =
          GraphicsBarrierAccessMode::None,
      GraphicsBarrierAccessMode afterAccessMode =
          GraphicsBarrierAccessMode::None,
      GraphicsBarrierState beforeBarrierState = GraphicsBarrierState::Unknown,
      GraphicsBarrierState afterBarrierState =
          GraphicsBarrierState::Unknown) noexcept {
    GraphicsBarrierRecord record{};
    record.setTextureHandle(textureHandle);
    record.beforeAccess = beforeAccessMode;
    record.afterAccess = afterAccessMode;
    record.beforeState = beforeBarrierState;
    record.afterState = afterBarrierState;
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

  constexpr void setTextureHandle(TextureHandle textureHandle) noexcept {
    resourceKind = GraphicsBarrierResourceKind::Texture;
    resource.texture = textureHandle;
  }

  constexpr void setBufferHandle(BufferHandle bufferHandle) noexcept {
    resourceKind = GraphicsBarrierResourceKind::Buffer;
    resource.buffer = bufferHandle;
  }

  [[nodiscard]] constexpr bool isTexture() const noexcept {
    return resourceKind == GraphicsBarrierResourceKind::Texture;
  }

  [[nodiscard]] constexpr TextureHandle textureHandle() const noexcept {
    return isTexture() ? resource.texture : TextureHandle{};
  }

  [[nodiscard]] constexpr BufferHandle bufferHandle() const noexcept {
    return isTexture() ? BufferHandle{} : resource.buffer;
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
  bool depthBiasEnable = false;
  float depthBiasConstant = 0.0f;
  float depthBiasSlope = 0.0f;
  float depthBiasClamp = 0.0f;
  std::span<const std::byte> pushConstants{};
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
};

enum class RenderPassExecutionMode : uint8_t {
  Graphics = 0,
  ComputeOnly = 1,
};

struct RenderPass {
  RenderPassExecutionMode executionMode = RenderPassExecutionMode::Graphics;
  AttachmentColor color;
  TextureHandle colorTexture{};
  bool hasColorAttachment = true;
  AttachmentDepth depth;
  TextureHandle depthTexture{};
  bool useViewport = false;
  Viewport viewport{};
  std::span<const ComputeDispatchItem> preDispatches{};
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const DrawItem> draws{};
  bool drawBuffersPreResolved = false;
  GpuTimingScope gpuTimingScope = GpuTimingScope::None;
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
};

struct SubmitBatchMeta {
  uint32_t commandBufferOffset = 0u;
  uint32_t commandBufferCount = 0u;
  bool presentsFrameOutput = false;
};

struct RenderFrame {
  std::span<const RenderPass> passes{};
};

} // namespace nuri
