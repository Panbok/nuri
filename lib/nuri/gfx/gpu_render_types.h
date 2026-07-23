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

struct ComputeDispatchItem {
  ComputePipelineHandle pipeline{};
  RayQueryBindingHandle rayQueryBinding{};
  DispatchSize dispatch{};
  std::span<const std::byte> pushConstants{};
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const RenderGraphAccessMode> dependencyBufferAccessModes{};
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
};

[[nodiscard]] constexpr uint32_t
gpuTimingScopeToBit(GpuTimingScope scope) noexcept {
  return scope == GpuTimingScope::None
             ? 0u
             : (1u << (static_cast<uint8_t>(scope) - 1u));
}

[[nodiscard]] constexpr GpuTimingScope
gpuTimingParentScope(GpuTimingScope scope) noexcept {
  switch (scope) {
  case GpuTimingScope::ShadowDepth:
  case GpuTimingScope::ShadowSdsm:
    return GpuTimingScope::Shadow;
  case GpuTimingScope::Velocity:
  case GpuTimingScope::ReactiveMask:
  case GpuTimingScope::OpaqueDepth:
  case GpuTimingScope::OpaqueNormal:
  case GpuTimingScope::OpaqueMain:
    return GpuTimingScope::Opaque;
  case GpuTimingScope::TemporalAACopyBack:
    return GpuTimingScope::TemporalAAResolve;
  case GpuTimingScope::GTAOTemporal:
    return GpuTimingScope::GTAO;
  case GpuTimingScope::RayTracingBLAS:
  case GpuTimingScope::RayTracingTLAS:
    return GpuTimingScope::RayTracingScene;
  case GpuTimingScope::DDGITrace:
  case GpuTimingScope::DDGIUpdate:
  case GpuTimingScope::DDGIRelocateClassify:
    return GpuTimingScope::DDGI;
  default:
    return GpuTimingScope::None;
  }
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
constexpr uint32_t kGpuTimingScopeOpaqueBit =
    gpuTimingScopeToBit(GpuTimingScope::Opaque);
constexpr uint32_t kGpuTimingScopeMsaaResolveBit =
    gpuTimingScopeToBit(GpuTimingScope::MsaaResolve);
constexpr uint32_t kGpuTimingScopeGTAOBit =
    gpuTimingScopeToBit(GpuTimingScope::GTAO);
constexpr uint32_t kGpuTimingScopeHDRPostProcessBit =
    gpuTimingScopeToBit(GpuTimingScope::HDRPostProcess);
constexpr uint32_t kGpuTimingScopeSkyboxBit =
    gpuTimingScopeToBit(GpuTimingScope::Skybox);
constexpr uint32_t kGpuTimingScopeVelocityBit =
    gpuTimingScopeToBit(GpuTimingScope::Velocity);
constexpr uint32_t kGpuTimingScopeReactiveMaskBit =
    gpuTimingScopeToBit(GpuTimingScope::ReactiveMask);
constexpr uint32_t kGpuTimingScopeTemporalAACopyBackBit =
    gpuTimingScopeToBit(GpuTimingScope::TemporalAACopyBack);
constexpr uint32_t kGpuTimingScopeGTAOTemporalBit =
    gpuTimingScopeToBit(GpuTimingScope::GTAOTemporal);
constexpr uint32_t kGpuTimingScopeWholeFrameBit =
    gpuTimingScopeToBit(GpuTimingScope::WholeFrame);
constexpr uint32_t kGpuTimingScopeRayTracingSceneBit =
    gpuTimingScopeToBit(GpuTimingScope::RayTracingScene);
constexpr uint32_t kGpuTimingScopeRayTracingBLASBit =
    gpuTimingScopeToBit(GpuTimingScope::RayTracingBLAS);
constexpr uint32_t kGpuTimingScopeRayTracingTLASBit =
    gpuTimingScopeToBit(GpuTimingScope::RayTracingTLAS);
constexpr uint32_t kGpuTimingScopeDDGIBit =
    gpuTimingScopeToBit(GpuTimingScope::DDGI);
constexpr uint32_t kGpuTimingScopeDDGITraceBit =
    gpuTimingScopeToBit(GpuTimingScope::DDGITrace);
constexpr uint32_t kGpuTimingScopeDDGIUpdateBit =
    gpuTimingScopeToBit(GpuTimingScope::DDGIUpdate);
constexpr uint32_t kGpuTimingScopeDDGIRelocateClassifyBit =
    gpuTimingScopeToBit(GpuTimingScope::DDGIRelocateClassify);
constexpr uint32_t kGpuTimingScopeOpaqueDepthBit =
    gpuTimingScopeToBit(GpuTimingScope::OpaqueDepth);
constexpr uint32_t kGpuTimingScopeOpaqueNormalBit =
    gpuTimingScopeToBit(GpuTimingScope::OpaqueNormal);
constexpr uint32_t kGpuTimingScopeOpaqueMainBit =
    gpuTimingScopeToBit(GpuTimingScope::OpaqueMain);

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
  uint64_t opaqueSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t msaaResolveSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t gtaoSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t hdrPostProcessSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t skyboxSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t velocitySourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t reactiveMaskSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t temporalAACopyBackSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t gtaoTemporalSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t wholeFrameSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t rayTracingSceneSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t rayTracingBlasSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t rayTracingTlasSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t ddgiSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t ddgiTraceSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t ddgiUpdateSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t ddgiRelocateClassifySourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t opaqueDepthSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t opaqueNormalSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  uint64_t opaqueMainSourceFrameIndex = std::numeric_limits<uint64_t>::max();
  float shadowTimeMs = 0.0f;
  float shadowDepthTimeMs = 0.0f;
  float shadowSdsmTimeMs = 0.0f;
  float sceneColorDownsampleTimeMs = 0.0f;
  float transmissionTimeMs = 0.0f;
  float temporalAAResolveTimeMs = 0.0f;
  float temporalAADebugTimeMs = 0.0f;
  float spatialAATimeMs = 0.0f;
  float opaqueTimeMs = 0.0f;
  float msaaResolveTimeMs = 0.0f;
  float gtaoTimeMs = 0.0f;
  float hdrPostProcessTimeMs = 0.0f;
  float skyboxTimeMs = 0.0f;
  float velocityTimeMs = 0.0f;
  float reactiveMaskTimeMs = 0.0f;
  float temporalAACopyBackTimeMs = 0.0f;
  float gtaoTemporalTimeMs = 0.0f;
  float wholeFrameTimeMs = 0.0f;
  float rayTracingSceneTimeMs = 0.0f;
  float rayTracingBlasTimeMs = 0.0f;
  float rayTracingTlasTimeMs = 0.0f;
  float ddgiTimeMs = 0.0f;
  float ddgiTraceTimeMs = 0.0f;
  float ddgiUpdateTimeMs = 0.0f;
  float ddgiRelocateClassifyTimeMs = 0.0f;
  float opaqueDepthTimeMs = 0.0f;
  float opaqueNormalTimeMs = 0.0f;
  float opaqueMainTimeMs = 0.0f;
  bool opaquePipelineStatisticsRequested = false;
  bool opaquePipelineStatisticsAvailable = false;
  uint64_t opaquePipelineStatisticsSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  uint64_t opaqueInputAssemblyVertices = 0u;
  uint64_t opaqueInputAssemblyPrimitives = 0u;
  uint64_t opaqueClippingInvocations = 0u;
  uint64_t opaqueClippingPrimitives = 0u;
  uint64_t opaqueFragmentShaderInvocations = 0u;
  uint32_t availableScopeMask = 0u;
  struct PassTiming {
    std::string debugName{};
    uint64_t sourceFrameIndex = std::numeric_limits<uint64_t>::max();
    float timeMs = 0.0f;
  };
  std::vector<PassTiming> passTimings{};
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

struct GpuTimingScopeMergeDesc {
  GpuTimingScope scope = GpuTimingScope::None;
  float GpuTimingReport::*timeMs = nullptr;
  uint64_t GpuTimingReport::*sourceFrameIndex = nullptr;
  uint32_t bit = 0u;
};

inline constexpr auto kGpuTimingScopeDescs =
    std::to_array<GpuTimingScopeMergeDesc>({
        {GpuTimingScope::Shadow, &GpuTimingReport::shadowTimeMs,
         &GpuTimingReport::shadowSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::Shadow)},
        {GpuTimingScope::ShadowDepth, &GpuTimingReport::shadowDepthTimeMs,
         &GpuTimingReport::shadowDepthSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::ShadowDepth)},
        {GpuTimingScope::ShadowSdsm, &GpuTimingReport::shadowSdsmTimeMs,
         &GpuTimingReport::shadowSdsmSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::ShadowSdsm)},
        {GpuTimingScope::SceneColorDownsample,
         &GpuTimingReport::sceneColorDownsampleTimeMs,
         &GpuTimingReport::sceneColorDownsampleSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::SceneColorDownsample)},
        {GpuTimingScope::Transmission, &GpuTimingReport::transmissionTimeMs,
         &GpuTimingReport::transmissionSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::Transmission)},
        {GpuTimingScope::TemporalAAResolve,
         &GpuTimingReport::temporalAAResolveTimeMs,
         &GpuTimingReport::temporalAAResolveSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::TemporalAAResolve)},
        {GpuTimingScope::TemporalAADebug,
         &GpuTimingReport::temporalAADebugTimeMs,
         &GpuTimingReport::temporalAADebugSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::TemporalAADebug)},
        {GpuTimingScope::SpatialAA, &GpuTimingReport::spatialAATimeMs,
         &GpuTimingReport::spatialAASourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::SpatialAA)},
        {GpuTimingScope::Opaque, &GpuTimingReport::opaqueTimeMs,
         &GpuTimingReport::opaqueSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::Opaque)},
        {GpuTimingScope::MsaaResolve, &GpuTimingReport::msaaResolveTimeMs,
         &GpuTimingReport::msaaResolveSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::MsaaResolve)},
        {GpuTimingScope::GTAO, &GpuTimingReport::gtaoTimeMs,
         &GpuTimingReport::gtaoSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::GTAO)},
        {GpuTimingScope::HDRPostProcess, &GpuTimingReport::hdrPostProcessTimeMs,
         &GpuTimingReport::hdrPostProcessSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::HDRPostProcess)},
        {GpuTimingScope::Skybox, &GpuTimingReport::skyboxTimeMs,
         &GpuTimingReport::skyboxSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::Skybox)},
        {GpuTimingScope::Velocity, &GpuTimingReport::velocityTimeMs,
         &GpuTimingReport::velocitySourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::Velocity)},
        {GpuTimingScope::ReactiveMask, &GpuTimingReport::reactiveMaskTimeMs,
         &GpuTimingReport::reactiveMaskSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::ReactiveMask)},
        {GpuTimingScope::TemporalAACopyBack,
         &GpuTimingReport::temporalAACopyBackTimeMs,
         &GpuTimingReport::temporalAACopyBackSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::TemporalAACopyBack)},
        {GpuTimingScope::GTAOTemporal, &GpuTimingReport::gtaoTemporalTimeMs,
         &GpuTimingReport::gtaoTemporalSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::GTAOTemporal)},
        {GpuTimingScope::WholeFrame, &GpuTimingReport::wholeFrameTimeMs,
         &GpuTimingReport::wholeFrameSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::WholeFrame)},
        {GpuTimingScope::RayTracingScene,
         &GpuTimingReport::rayTracingSceneTimeMs,
         &GpuTimingReport::rayTracingSceneSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::RayTracingScene)},
        {GpuTimingScope::RayTracingBLAS, &GpuTimingReport::rayTracingBlasTimeMs,
         &GpuTimingReport::rayTracingBlasSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::RayTracingBLAS)},
        {GpuTimingScope::RayTracingTLAS, &GpuTimingReport::rayTracingTlasTimeMs,
         &GpuTimingReport::rayTracingTlasSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::RayTracingTLAS)},
        {GpuTimingScope::DDGI, &GpuTimingReport::ddgiTimeMs,
         &GpuTimingReport::ddgiSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::DDGI)},
        {GpuTimingScope::DDGITrace, &GpuTimingReport::ddgiTraceTimeMs,
         &GpuTimingReport::ddgiTraceSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::DDGITrace)},
        {GpuTimingScope::DDGIUpdate, &GpuTimingReport::ddgiUpdateTimeMs,
         &GpuTimingReport::ddgiUpdateSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::DDGIUpdate)},
        {GpuTimingScope::DDGIRelocateClassify,
         &GpuTimingReport::ddgiRelocateClassifyTimeMs,
         &GpuTimingReport::ddgiRelocateClassifySourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::DDGIRelocateClassify)},
        {GpuTimingScope::OpaqueDepth, &GpuTimingReport::opaqueDepthTimeMs,
         &GpuTimingReport::opaqueDepthSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::OpaqueDepth)},
        {GpuTimingScope::OpaqueNormal, &GpuTimingReport::opaqueNormalTimeMs,
         &GpuTimingReport::opaqueNormalSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::OpaqueNormal)},
        {GpuTimingScope::OpaqueMain, &GpuTimingReport::opaqueMainTimeMs,
         &GpuTimingReport::opaqueMainSourceFrameIndex,
         gpuTimingScopeToBit(GpuTimingScope::OpaqueMain)},
    });

[[nodiscard]] constexpr const GpuTimingScopeMergeDesc *
gpuTimingScopeDesc(GpuTimingScope scope) noexcept {
  for (const GpuTimingScopeMergeDesc &desc : kGpuTimingScopeDescs) {
    if (desc.scope == scope) {
      return &desc;
    }
  }
  return nullptr;
}

[[nodiscard]] constexpr uint64_t
gpuTimingScopeSourceFrame(const GpuTimingReport &report,
                          GpuTimingScope scope) noexcept {
  const GpuTimingScopeMergeDesc *desc = gpuTimingScopeDesc(scope);
  return desc != nullptr ? report.*desc->sourceFrameIndex
                         : std::numeric_limits<uint64_t>::max();
}

inline void mergeGpuTimingReportScope(GpuTimingReport &dst,
                                      const GpuTimingReport &src,
                                      GpuTimingScopeMergeDesc desc) {
  if (!hasGpuTimingScope(src, desc.scope)) {
    return;
  }
  dst.*desc.timeMs = src.*desc.timeMs;
  dst.*desc.sourceFrameIndex = src.*desc.sourceFrameIndex;
  dst.availableScopeMask |= desc.bit;
}

inline void mergeGpuTimingReportScopes(GpuTimingReport &dst,
                                       const GpuTimingReport &src) {
  for (const GpuTimingScopeMergeDesc desc : kGpuTimingScopeDescs) {
    mergeGpuTimingReportScope(dst, src, desc);
  }
}

enum class GraphicsBarrierResourceKind : uint8_t {
  Texture = 0,
  Buffer = 1,
  AccelerationStructure = 2,
};

using GraphicsBarrierAccessMode = RenderGraphAccessMode;

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
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const TextureHandle> dependencyTextures{};
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
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const TextureHandle> dependencyTextures{};
  std::span<const DrawItem> draws{};
  std::span<const MeshDispatchItem> meshDispatches{};
  std::span<const BufferCopyRegion> bufferCopies{};
  std::span<const TextureCopyItem> textureCopies{};
  std::span<const AccelerationStructureBuildItem> accelerationStructureBuilds{};
  ExternalTemporalDispatchItem externalTemporalDispatch{};
  bool payloadBorrowed = false;
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

struct SubmittedGraphicsFrame {
  SubmissionHandle submission{};
  std::string presentationError{};
};

struct RenderFrame {
  std::span<const RenderPass> passes{};
};

} // namespace nuri
