#pragma once

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/containers/hash_set.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/render_graph/render_graph_runtime.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>

namespace nuri {

struct NURI_API RenderGraphPassId {
  uint32_t value = UINT32_MAX;
};

struct NURI_API RenderGraphTextureId {
  uint32_t value = UINT32_MAX;
};

struct NURI_API RenderGraphBufferId {
  uint32_t value = UINT32_MAX;
};

struct NURI_API PersistentBufferId {
  uint32_t value = UINT32_MAX;
};

struct NURI_API PersistentTextureId {
  uint32_t value = UINT32_MAX;
};

[[nodiscard]] constexpr bool isValid(PersistentBufferId id) {
  return id.value != UINT32_MAX;
}
[[nodiscard]] constexpr bool isValid(PersistentTextureId id) {
  return id.value != UINT32_MAX;
}

[[nodiscard]] constexpr bool isValid(RenderGraphPassId id) {
  return id.value != UINT32_MAX;
}

[[nodiscard]] constexpr bool isValid(RenderGraphTextureId id) {
  return id.value != UINT32_MAX;
}

[[nodiscard]] constexpr bool isValid(RenderGraphBufferId id) {
  return id.value != UINT32_MAX;
}

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

struct NURI_API RenderGraphGraphicsPassDesc {
  RenderPassExecutionMode executionMode = RenderPassExecutionMode::Graphics;
  AttachmentColor color{};
  RenderGraphTextureId colorTexture{};
  RenderGraphTextureId colorResolveTexture{};
  bool hasColorAttachment = true;
  AttachmentDepth depth{};
  RenderGraphTextureId depthTexture{};
  RenderGraphTextureId depthResolveTexture{};
  bool useViewport = false;
  Viewport viewport{};
  std::span<const ComputeDispatchItem> preDispatches{};
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const RenderGraphAccessMode> dependencyBufferAccessModes{};
  std::span<const TextureHandle> dependencyTextures{};
  std::span<const RenderGraphAccessMode> dependencyTextureAccessModes{};
  std::span<const DrawItem> draws{};
  std::span<const MeshDispatchItem> meshDispatches{};
  bool drawBuffersPreResolved = false;
  // Explicit extra draw-buffer dependencies; draw items are not scanned when
  // drawBuffersPreResolved is true.
  std::span<const BufferHandle> preResolvedDrawBuffers{};
  GpuTimingScope gpuTimingScope = GpuTimingScope::None;
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
  bool markColorAsFrameOutput = false;
  bool markImplicitOutputSideEffect = true;
  bool borrowPayload = false;
};

struct NURI_API RenderGraphTextureCopyItem {
  RenderGraphTextureId sourceTexture{};
  RenderGraphTextureId destinationTexture{};
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

struct NURI_API RenderGraphTextureCopyPassDesc {
  std::span<const RenderGraphTextureCopyItem> copies{};
  GpuTimingScope gpuTimingScope = GpuTimingScope::None;
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
  bool markImplicitOutputSideEffect = true;
};

enum class RenderGraphDrawBufferBindingTarget : uint8_t {
  Vertex = 0,
  Index = 1,
  Indirect = 2,
  IndirectCount = 3,
};

enum class RenderGraphMeshDispatchBufferBindingTarget : uint8_t {
  Indirect = 0,
  IndirectCount = 1,
};

struct NURI_API RenderGraphPreparedDependencyBufferBinding {
  uint32_t dependencyIndex = UINT32_MAX;
  RenderGraphBufferId buffer{};
  RenderGraphAccessMode mode =
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write;
};

struct NURI_API RenderGraphPreparedDependencyTextureBinding {
  RenderGraphTextureId texture{};
  RenderGraphAccessMode mode = RenderGraphAccessMode::Read;
};

struct NURI_API RenderGraphPreparedPreDispatchDependencyBinding {
  uint32_t preDispatchIndex = UINT32_MAX;
  uint32_t dependencyIndex = UINT32_MAX;
  RenderGraphBufferId buffer{};
  RenderGraphAccessMode mode =
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write;
};

struct NURI_API RenderGraphPreparedDrawBufferBinding {
  uint32_t drawIndex = UINT32_MAX;
  RenderGraphDrawBufferBindingTarget target =
      RenderGraphDrawBufferBindingTarget::Vertex;
  RenderGraphBufferId buffer{};
  RenderGraphAccessMode mode = RenderGraphAccessMode::Read;
};

struct NURI_API RenderGraphPreparedMeshDispatchBufferBinding {
  uint32_t meshDispatchIndex = UINT32_MAX;
  RenderGraphMeshDispatchBufferBindingTarget target =
      RenderGraphMeshDispatchBufferBindingTarget::Indirect;
  RenderGraphBufferId buffer{};
  RenderGraphAccessMode mode = RenderGraphAccessMode::Read;
};

struct NURI_API RenderGraphPreparedGraphicsPassDesc {
  RenderPassExecutionMode executionMode = RenderPassExecutionMode::Graphics;
  AttachmentColor color{};
  RenderGraphTextureId colorTexture{};
  RenderGraphTextureId colorResolveTexture{};
  bool hasColorAttachment = true;
  AttachmentDepth depth{};
  RenderGraphTextureId depthTexture{};
  RenderGraphTextureId depthResolveTexture{};
  bool useViewport = false;
  Viewport viewport{};
  std::span<const ComputeDispatchItem> preDispatches{};
  // Use dependencyBuffers when the caller already has resolved BufferHandle
  // objects. Use dependencyBufferBindings when the pass refers to graph-owned
  // buffers by RenderGraphBufferId/dependency index and resolution happens
  // during graph compilation; both fields are optional and do not need to be
  // populated together.
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const DrawItem> draws{};
  std::span<const MeshDispatchItem> meshDispatches{};
  std::span<const RenderGraphPreparedDependencyBufferBinding>
      dependencyBufferBindings{};
  std::span<const RenderGraphPreparedDependencyTextureBinding>
      dependencyTextureBindings{};
  std::span<const RenderGraphPreparedPreDispatchDependencyBinding>
      preDispatchDependencyBindings{};
  std::span<const RenderGraphPreparedDrawBufferBinding> drawBufferBindings{};
  std::span<const RenderGraphPreparedMeshDispatchBufferBinding>
      meshDispatchBufferBindings{};
  bool drawBuffersPreResolved = false;
  // Explicit extra draw-buffer dependencies; drawBufferBindings must be empty
  // when drawBuffersPreResolved is true.
  std::span<const BufferHandle> preResolvedDrawBuffers{};
  // Resolved extra draw-buffer dependencies for callers that already imported
  // the shared draw buffer set once for the frame.
  std::span<const RenderGraphBufferId> preResolvedDrawBufferIds{};
  GpuTimingScope gpuTimingScope = GpuTimingScope::None;
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
  bool markColorAsFrameOutput = false;
  bool markImplicitOutputSideEffect = true;
  bool borrowPayload = false;
};

struct NURI_API RecordedGraphicsPassMeta {
  uint32_t orderedPassIndex = UINT32_MAX;
  uint32_t declaredPassIndex = UINT32_MAX;
};

struct NURI_API RenderGraphPassExecutionTiming {
  uint32_t orderedPassIndex = UINT32_MAX;
  float cpuTimeMs = 0.0f;
};

struct NURI_API RecordedCommandBufferMeta {
  uint32_t firstOrderedPassIndex = UINT32_MAX;
  uint32_t passCount = 0u;
};

enum class RenderGraphBarrierResourceKind : uint8_t {
  Texture = 0,
  Buffer = 1,
};

enum class RenderGraphResourceState : uint8_t {
  Unknown = 0,
  Read = 1,
  Write = 2,
  Attachment = 3,
  Present = 4,
};

struct NURI_API RenderGraphBarrierRecord {
  RenderGraphBarrierResourceKind resourceKind =
      RenderGraphBarrierResourceKind::Texture;
  uint32_t resourceIndex = UINT32_MAX;
  RenderGraphAccessMode beforeAccess = RenderGraphAccessMode::None;
  RenderGraphAccessMode afterAccess = RenderGraphAccessMode::None;
  RenderGraphResourceState beforeState = RenderGraphResourceState::Unknown;
  RenderGraphResourceState afterState = RenderGraphResourceState::Unknown;
};

struct NURI_API PassBarrierPlan {
  uint32_t orderedPassIndex = UINT32_MAX;
  uint32_t barrierOffset = 0u;
  uint32_t barrierCount = 0u;
};

struct NURI_API FinalBarrierPlan {
  uint32_t barrierOffset = 0u;
  uint32_t barrierCount = 0u;
};

struct NURI_API RenderGraphPassRange {
  uint32_t workerIndex = UINT32_MAX;
  uint32_t firstOrderedPassIndex = UINT32_MAX;
  uint32_t passCount = 0u;
};

enum class RenderGraphExecutionFailureStage : uint8_t {
  ValidateCompiledMetadata = 0,
  MaterializeTransients,
  BuildExecutablePayload,
  PatchUnresolvedBindings,
  ResolveBarriers,
  AcquireRecordingContext,
  RecordGraphicsBarriers,
  RecordGraphicsPasses,
  FinishRecordingContext,
  SubmitRecordedFrame,
};

[[nodiscard]] NURI_API std::string_view
toString(RenderGraphExecutionFailureStage stage) noexcept;

struct NURI_API RenderGraphExecutionMetadata {
  std::pmr::vector<RecordedCommandBufferMeta> recordedCommandBuffers;
  std::pmr::vector<SubmitBatchMeta> submitBatches;
  std::pmr::vector<RenderGraphPassRange> passRanges;
  std::pmr::vector<RenderGraphPassExecutionTiming> passTimings;
  bool usedParallelCompile = false;
  bool usedParallelRecording = false;

  explicit RenderGraphExecutionMetadata(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : recordedCommandBuffers(ensureMemory(memory)),
        submitBatches(ensureMemory(memory)), passRanges(ensureMemory(memory)),
        passTimings(ensureMemory(memory)) {}

private:
  static std::pmr::memory_resource *ensureMemory(std::pmr::memory_resource *m) {
    return m != nullptr ? m : std::pmr::get_default_resource();
  }
};

struct NURI_API RenderGraphCompileResult {
  struct Edge {
    uint32_t before = UINT32_MAX;
    uint32_t after = UINT32_MAX;
  };

  struct ResourceStats {
    uint32_t importedTextures = 0;
    uint32_t transientTextures = 0;
    uint32_t importedBuffers = 0;
    uint32_t transientBuffers = 0;
  };

  struct TransientLifetime {
    uint32_t resourceIndex = UINT32_MAX;
    uint32_t firstExecutionIndex = UINT32_MAX;
    uint32_t lastExecutionIndex = UINT32_MAX;
  };

  struct TransientAllocation {
    uint32_t resourceIndex = UINT32_MAX;
    uint32_t allocationIndex = UINT32_MAX;
  };

  struct TransientTexturePhysicalAllocation {
    uint32_t allocationIndex = UINT32_MAX;
    uint32_t representativeResourceIndex = UINT32_MAX;
    TextureDesc desc{};
  };

  struct TransientBufferPhysicalAllocation {
    uint32_t allocationIndex = UINT32_MAX;
    uint32_t representativeResourceIndex = UINT32_MAX;
    BufferDesc desc{};
  };

  enum class PassTextureBindingTarget : uint8_t {
    Color = 0,
    Depth = 1,
    ColorResolve = 2,
    DepthResolve = 3,
  };

  struct PassTextureBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t textureResourceIndex = UINT32_MAX;
    PassTextureBindingTarget target = PassTextureBindingTarget::Color;
  };

  struct PassDependencyBufferRange {
    uint32_t offset = 0;
    uint32_t count = 0;
  };

  struct UnresolvedDependencyBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t dependencyBufferIndex = UINT32_MAX;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };

  struct PassDependencyTextureRange {
    uint32_t offset = 0;
    uint32_t count = 0;
  };

  struct UnresolvedDependencyTextureBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t dependencyTextureIndex = UINT32_MAX;
    uint32_t textureResourceIndex = UINT32_MAX;
  };

  struct PassDispatchRange {
    uint32_t offset = 0;
    uint32_t count = 0;
  };

  struct PassDrawRange {
    uint32_t offset = 0;
    uint32_t count = 0;
  };

  struct PassTextureCopyRange {
    uint32_t offset = 0;
    uint32_t count = 0;
  };

  struct DispatchDependencyBufferRange {
    uint32_t offset = 0;
    uint32_t count = 0;
  };

  enum class DrawBufferBindingTarget : uint8_t {
    Vertex = 0,
    Index = 1,
    Indirect = 2,
    IndirectCount = 3,
  };

  enum class MeshDispatchBufferBindingTarget : uint8_t {
    Indirect = 0,
    IndirectCount = 1,
  };

  struct UnresolvedPreDispatchDependencyBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t preDispatchIndex = UINT32_MAX;
    uint32_t dependencyBufferIndex = UINT32_MAX;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };

  struct UnresolvedDrawBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t drawIndex = UINT32_MAX;
    DrawBufferBindingTarget target = DrawBufferBindingTarget::Vertex;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };

  struct UnresolvedMeshDispatchBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t meshDispatchIndex = UINT32_MAX;
    MeshDispatchBufferBindingTarget target =
        MeshDispatchBufferBindingTarget::Indirect;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };

  enum class TextureCopyBindingTarget : uint8_t {
    Source = 0,
    Destination = 1,
  };

  struct UnresolvedTextureCopyBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t textureCopyIndex = UINT32_MAX;
    TextureCopyBindingTarget target = TextureCopyBindingTarget::Source;
    uint32_t textureResourceIndex = UINT32_MAX;
  };

  uint64_t frameIndex = 0;
  uint32_t declaredPassCount = 0;
  uint32_t culledPassCount = 0;
  uint32_t rootPassCount = 0;
  uint32_t transientTexturePhysicalCount = 0;
  uint32_t transientBufferPhysicalCount = 0;
  bool metadataValidated = false;
  bool usedParallelCompile = false;
  bool usedParallelValidation = false;
  bool usedParallelPayloadResolution = false;
  bool usedParallelHazardAnalysis = false;
  bool usedParallelLifetimeAnalysis = false;
  ResourceStats resourceStats{};
  std::pmr::vector<TextureHandle> textureHandlesByResource;
  std::pmr::vector<BufferHandle> bufferHandlesByResource;
  std::pmr::vector<RenderPass> orderedPasses;
  std::pmr::vector<uint32_t> orderedPassIndices;
  std::pmr::vector<RecordedGraphicsPassMeta> recordedGraphicsPasses;
  std::pmr::vector<std::pmr::string> passDebugNames;
  std::pmr::vector<Edge> edges;
  std::pmr::vector<PassBarrierPlan> passBarrierPlans;
  FinalBarrierPlan finalBarrierPlan{};
  std::pmr::vector<RenderGraphBarrierRecord> passBarrierRecords;
  std::pmr::vector<TransientLifetime> transientTextureLifetimes;
  std::pmr::vector<TransientLifetime> transientBufferLifetimes;
  std::pmr::vector<TransientAllocation> transientTextureAllocations;
  std::pmr::vector<TransientAllocation> transientBufferAllocations;
  std::pmr::vector<uint32_t> transientTextureAllocationByResource;
  std::pmr::vector<uint32_t> transientBufferAllocationByResource;
  std::pmr::vector<TransientTexturePhysicalAllocation>
      transientTexturePhysicalAllocations;
  std::pmr::vector<TransientBufferPhysicalAllocation>
      transientBufferPhysicalAllocations;
  std::pmr::vector<PassTextureBinding> unresolvedTextureBindings;
  std::pmr::vector<BufferHandle> resolvedDependencyBuffers;
  // Parallel to resolvedDependencyBuffers: the buffer resource index for each
  // slot.  UINT32_MAX means no tracking (null slot).  Used by
  // refreshHandlesInCompileResult to patch stale handles for imported per-frame
  // buffers (e.g. ring buffer slots) on compile-cache hits.
  std::pmr::vector<uint32_t> resolvedDependencyBufferResourceIndices;
  std::pmr::vector<PassDependencyBufferRange> dependencyBufferRangesByPass;
  std::pmr::vector<UnresolvedDependencyBufferBinding>
      unresolvedDependencyBufferBindings;
  std::pmr::vector<TextureHandle> resolvedDependencyTextures;
  std::pmr::vector<uint32_t> resolvedDependencyTextureResourceIndices;
  std::pmr::vector<PassDependencyTextureRange> dependencyTextureRangesByPass;
  std::pmr::vector<UnresolvedDependencyTextureBinding>
      unresolvedDependencyTextureBindings;
  std::pmr::vector<ComputeDispatchItem> ownedPreDispatches;
  std::pmr::vector<DrawItem> ownedDrawItems;
  std::pmr::vector<MeshDispatchItem> ownedMeshDispatchItems;
  std::pmr::vector<TextureCopyItem> ownedTextureCopyItems;
  std::pmr::vector<std::pmr::string> ownedMeshDispatchDebugLabels;
  std::pmr::vector<std::pmr::vector<std::byte>> ownedMeshDispatchPushConstants;
  std::pmr::vector<std::pmr::vector<BufferHandle>>
      ownedMeshDispatchDependencyBuffers;
  std::pmr::vector<std::pmr::vector<TextureHandle>>
      ownedMeshDispatchDependencyTextures;
  std::pmr::vector<PassDispatchRange> preDispatchRangesByPass;
  std::pmr::vector<PassDrawRange> drawRangesByPass;
  std::pmr::vector<PassDispatchRange> meshDispatchRangesByPass;
  std::pmr::vector<PassTextureCopyRange> textureCopyRangesByPass;
  std::pmr::vector<BufferHandle> resolvedPreDispatchDependencyBuffers;
  // Parallel to resolvedPreDispatchDependencyBuffers: resource index per slot.
  std::pmr::vector<uint32_t> resolvedPreDispatchDependencyBufferResourceIndices;
  std::pmr::vector<DispatchDependencyBufferRange> preDispatchDependencyRanges;
  std::pmr::vector<UnresolvedPreDispatchDependencyBufferBinding>
      unresolvedPreDispatchDependencyBufferBindings;
  std::pmr::vector<UnresolvedDrawBufferBinding> unresolvedDrawBufferBindings;
  std::pmr::vector<UnresolvedMeshDispatchBufferBinding>
      unresolvedMeshDispatchBufferBindings;
  std::pmr::vector<UnresolvedTextureCopyBinding> unresolvedTextureCopyBindings;

  explicit RenderGraphCompileResult(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : textureHandlesByResource(ensureMemory(memory)),
        bufferHandlesByResource(ensureMemory(memory)),
        orderedPasses(ensureMemory(memory)),
        orderedPassIndices(ensureMemory(memory)),
        recordedGraphicsPasses(ensureMemory(memory)),
        passDebugNames(ensureMemory(memory)), edges(ensureMemory(memory)),
        passBarrierPlans(ensureMemory(memory)),
        passBarrierRecords(ensureMemory(memory)),
        transientTextureLifetimes(ensureMemory(memory)),
        transientBufferLifetimes(ensureMemory(memory)),
        transientTextureAllocations(ensureMemory(memory)),
        transientBufferAllocations(ensureMemory(memory)),
        transientTextureAllocationByResource(ensureMemory(memory)),
        transientBufferAllocationByResource(ensureMemory(memory)),
        transientTexturePhysicalAllocations(ensureMemory(memory)),
        transientBufferPhysicalAllocations(ensureMemory(memory)),
        unresolvedTextureBindings(ensureMemory(memory)),
        resolvedDependencyBuffers(ensureMemory(memory)),
        resolvedDependencyBufferResourceIndices(ensureMemory(memory)),
        dependencyBufferRangesByPass(ensureMemory(memory)),
        unresolvedDependencyBufferBindings(ensureMemory(memory)),
        resolvedDependencyTextures(ensureMemory(memory)),
        resolvedDependencyTextureResourceIndices(ensureMemory(memory)),
        dependencyTextureRangesByPass(ensureMemory(memory)),
        unresolvedDependencyTextureBindings(ensureMemory(memory)),
        ownedPreDispatches(ensureMemory(memory)),
        ownedDrawItems(ensureMemory(memory)),
        ownedMeshDispatchItems(ensureMemory(memory)),
        ownedTextureCopyItems(ensureMemory(memory)),
        ownedMeshDispatchDebugLabels(ensureMemory(memory)),
        ownedMeshDispatchPushConstants(ensureMemory(memory)),
        ownedMeshDispatchDependencyBuffers(ensureMemory(memory)),
        ownedMeshDispatchDependencyTextures(ensureMemory(memory)),
        preDispatchRangesByPass(ensureMemory(memory)),
        drawRangesByPass(ensureMemory(memory)),
        meshDispatchRangesByPass(ensureMemory(memory)),
        textureCopyRangesByPass(ensureMemory(memory)),
        resolvedPreDispatchDependencyBuffers(ensureMemory(memory)),
        resolvedPreDispatchDependencyBufferResourceIndices(
            ensureMemory(memory)),
        preDispatchDependencyRanges(ensureMemory(memory)),
        unresolvedPreDispatchDependencyBufferBindings(ensureMemory(memory)),
        unresolvedDrawBufferBindings(ensureMemory(memory)),
        unresolvedMeshDispatchBufferBindings(ensureMemory(memory)),
        unresolvedTextureCopyBindings(ensureMemory(memory)) {}

private:
  static std::pmr::memory_resource *ensureMemory(std::pmr::memory_resource *m) {
    return m != nullptr ? m : std::pmr::get_default_resource();
  }
};

class NURI_API RenderGraphBuilder {
public:
  explicit RenderGraphBuilder(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());

  void beginFrame(uint64_t frameIndex);
  [[nodiscard]] Result<RenderGraphTextureId, std::string>
  importTexture(TextureHandle texture, std::string_view debugName = {});
  [[nodiscard]] Result<RenderGraphBufferId, std::string>
  importBuffer(BufferHandle buffer, std::string_view debugName = {});
  [[nodiscard]] Result<RenderGraphTextureId, std::string>
  createTransientTexture(const TextureDesc &desc,
                         std::string_view debugName = {});
  [[nodiscard]] Result<RenderGraphBufferId, std::string>
  createTransientBuffer(const BufferDesc &desc,
                        std::string_view debugName = {});
  [[nodiscard]] Result<bool, std::string>
  addTextureAccess(RenderGraphPassId pass, RenderGraphTextureId texture,
                   RenderGraphAccessMode mode);
  [[nodiscard]] Result<bool, std::string>
  addBufferAccess(RenderGraphPassId pass, RenderGraphBufferId buffer,
                  RenderGraphAccessMode mode);
  [[nodiscard]] Result<bool, std::string>
  addTextureRead(RenderGraphPassId pass, RenderGraphTextureId texture);
  [[nodiscard]] Result<bool, std::string>
  addTextureWrite(RenderGraphPassId pass, RenderGraphTextureId texture);
  [[nodiscard]] Result<bool, std::string>
  addBufferRead(RenderGraphPassId pass, RenderGraphBufferId buffer);
  [[nodiscard]] Result<bool, std::string>
  addBufferWrite(RenderGraphPassId pass, RenderGraphBufferId buffer);
  [[nodiscard]] Result<RenderGraphPassId, std::string>
  addGraphicsPass(const RenderGraphGraphicsPassDesc &desc);
  [[nodiscard]] Result<RenderGraphPassId, std::string>
  addPreparedGraphicsPass(const RenderGraphPreparedGraphicsPassDesc &desc);
  [[nodiscard]] Result<RenderGraphPassId, std::string>
  addTextureCopyPass(const RenderGraphTextureCopyPassDesc &desc);
  [[nodiscard]] Result<bool, std::string>
  bindPassColorTexture(RenderGraphPassId pass, RenderGraphTextureId texture);
  [[nodiscard]] Result<bool, std::string>
  bindPassColorResolveTexture(RenderGraphPassId pass,
                              RenderGraphTextureId texture);
  [[nodiscard]] Result<bool, std::string>
  bindPassDepthTexture(RenderGraphPassId pass, RenderGraphTextureId texture);
  [[nodiscard]] Result<bool, std::string>
  bindPassDepthResolveTexture(RenderGraphPassId pass,
                              RenderGraphTextureId texture);
  [[nodiscard]] Result<bool, std::string> bindPassDependencyBuffer(
      RenderGraphPassId pass, uint32_t dependencyIndex,
      RenderGraphBufferId buffer,
      RenderGraphAccessMode mode = RenderGraphAccessMode::Read |
                                   RenderGraphAccessMode::Write);
  [[nodiscard]] Result<bool, std::string> bindPreDispatchDependencyBuffer(
      RenderGraphPassId pass, uint32_t preDispatchIndex,
      uint32_t dependencyIndex, RenderGraphBufferId buffer,
      RenderGraphAccessMode mode = RenderGraphAccessMode::Read |
                                   RenderGraphAccessMode::Write);
  [[nodiscard]] Result<bool, std::string>
  bindDrawBuffer(RenderGraphPassId pass, uint32_t drawIndex,
                 RenderGraphCompileResult::DrawBufferBindingTarget target,
                 RenderGraphBufferId buffer,
                 RenderGraphAccessMode mode = RenderGraphAccessMode::Read);
  [[nodiscard]] Result<bool, std::string> bindMeshDispatchBuffer(
      RenderGraphPassId pass, uint32_t meshDispatchIndex,
      RenderGraphCompileResult::MeshDispatchBufferBindingTarget target,
      RenderGraphBufferId buffer,
      RenderGraphAccessMode mode = RenderGraphAccessMode::Read);
  [[nodiscard]] Result<bool, std::string>
  addDependency(RenderGraphPassId before, RenderGraphPassId after);
  [[nodiscard]] Result<bool, std::string>
  markPassSideEffect(RenderGraphPassId pass);
  [[nodiscard]] Result<bool, std::string>
  markTextureAsFrameOutput(RenderGraphTextureId texture);
  void setInferredSideEffectSuppression(bool enabled) noexcept {
    suppressInferredSideEffectsWhenExplicitOutputs_ = enabled;
  }
  [[nodiscard]] Result<RenderGraphCompileResult, std::string>
  compile(RenderGraphRuntime &runtime) const;
  [[nodiscard]] size_t passCount() const noexcept { return passes_.size(); }

  // Compile-result caching support.
  //
  // GraphFingerprint is a compact signature over the graph's structural state
  // (topology, access patterns, transient descriptors).  Two graphs with the
  // same fingerprint will produce structurally identical
  // RenderGraphCompileResult values — only the imported texture/buffer handles
  // will differ.
  //
  // Callers can use this to detect stable frames and avoid re-running the full
  // C0–C7 compile pipeline.  On a fingerprint match, call
  // refreshHandlesInCompileResult() to update imported handles in a previously
  // compiled result, then pass it directly to the executor.
  struct NURI_API GraphFingerprint {
    size_t passCount = 0;
    size_t totalTextureCount = 0;
    size_t totalBufferCount = 0;
    size_t edgeCount = 0;
    size_t passAccessCount = 0;
    size_t frameOutputCount = 0;
    size_t sideEffectMarkCount = 0;
    bool allPassesBorrowPayload = true;
    // Combined hash over pass payload layout metadata that affects compile
    // result shape, such as dependency/pre-dispatch/draw counts.
    uint64_t payloadLayoutHash = 0;
    // Combined hash over transient texture/buffer descriptors recorded by
    // createTransientTexture()/createTransientBuffer() for this frame.
    uint64_t transientResourceDescriptorsHash = 0;
    // Monotonically incremented by updatePersistentBuffer/Texture whenever a
    // registered persistent handle is replaced.  A handle replacement changes
    // the import order on the next frame (the new handle gets a fresh index),
    // so the compile result must be discarded.
    uint64_t persistentHandlesVersion = 0;
    // Optional caller-provided salt for frame-variant payloads that are not
    // represented by structural graph state alone.
    uint64_t dynamicPayloadVersion = 0;

    [[nodiscard]] bool operator==(const GraphFingerprint &o) const noexcept {
      return passCount == o.passCount &&
             totalTextureCount == o.totalTextureCount &&
             totalBufferCount == o.totalBufferCount &&
             edgeCount == o.edgeCount && passAccessCount == o.passAccessCount &&
             frameOutputCount == o.frameOutputCount &&
             sideEffectMarkCount == o.sideEffectMarkCount &&
             allPassesBorrowPayload == o.allPassesBorrowPayload &&
             payloadLayoutHash == o.payloadLayoutHash &&
             transientResourceDescriptorsHash ==
                 o.transientResourceDescriptorsHash &&
             persistentHandlesVersion == o.persistentHandlesVersion &&
             dynamicPayloadVersion == o.dynamicPayloadVersion;
    }
  };

  [[nodiscard]] GraphFingerprint computeGraphFingerprint() const noexcept;
  void mixDynamicPayloadVersion(uint64_t version) noexcept;

  // Updates textureHandlesByResource and bufferHandlesByResource in a cached
  // compile result to reflect the imported handles recorded in the current
  // frame's builder state.  All other structural data (topology, barriers,
  // transient lifetimes) is left unchanged.
  void refreshHandlesInCompileResult(RenderGraphCompileResult &result) const;

  // Persistent import API: PersistentBufferId / PersistentTextureId registers
  // a handle for cross-frame tracking, but beginFrame() does not automatically
  // pre-import persistent resources. Call importBuffer()/importTexture() in
  // the desired BUILD-phase order each frame; use updatePersistentBuffer() /
  // updatePersistentTexture() when the underlying GPU handle changes and
  // unregisterPersistentBuffer() / unregisterPersistentTexture() when the
  // resource is destroyed.
  //
  // The returned PersistentBufferId / PersistentTextureId is valid for the
  // lifetime of this builder and must be passed to updatePersistentBuffer /
  // unregisterPersistentBuffer when the underlying GPU handle is recreated or
  // destroyed.
  [[nodiscard]] PersistentBufferId
  registerPersistentBuffer(BufferHandle handle, std::string_view debugName);
  [[nodiscard]] PersistentTextureId
  registerPersistentTexture(TextureHandle handle, std::string_view debugName);
  void updatePersistentBuffer(PersistentBufferId id, BufferHandle newHandle);
  void updatePersistentTexture(PersistentTextureId id, TextureHandle newHandle);
  void unregisterPersistentBuffer(PersistentBufferId id);
  void unregisterPersistentTexture(PersistentTextureId id);

private:
  struct DependencyEdge {
    uint32_t before = UINT32_MAX;
    uint32_t after = UINT32_MAX;
  };

  enum class AccessResourceKind : uint8_t {
    Texture = 0,
    Buffer = 1,
  };

  struct PassResourceAccess {
    uint32_t passIndex = UINT32_MAX;
    AccessResourceKind resourceKind = AccessResourceKind::Texture;
    uint32_t resourceIndex = UINT32_MAX;
    RenderGraphAccessMode mode = RenderGraphAccessMode::None;
    bool inferred = false;
  };

  struct SideEffectPassMark {
    uint32_t passIndex = UINT32_MAX;
    bool inferred = false;
  };

  struct TextureResource {
    bool imported = false;
    TextureHandle importedHandle{};
    TextureDesc transientDesc{};
    std::pmr::string debugName;

    explicit TextureResource(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : debugName(memory) {}
  };

  struct BufferResource {
    bool imported = false;
    BufferHandle importedHandle{};
    BufferDesc transientDesc{};
    std::pmr::string debugName;

    explicit BufferResource(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : debugName(memory) {}
  };

  struct OwnedPassPayload {
    std::pmr::string debugLabel;
    std::pmr::vector<ComputeDispatchItem> preDispatches;
    std::pmr::vector<std::pmr::string> preDispatchDebugLabels;
    std::pmr::vector<std::pmr::vector<std::byte>> preDispatchPushConstants;
    std::pmr::vector<std::pmr::vector<BufferHandle>>
        preDispatchDependencyBuffers;
    std::pmr::vector<std::pmr::vector<TextureHandle>>
        preDispatchDependencyTextures;
    std::pmr::vector<BufferHandle> dependencyBuffers;
    std::pmr::vector<TextureHandle> dependencyTextures;
    std::pmr::vector<DrawItem> draws;
    std::pmr::vector<std::pmr::string> drawDebugLabels;
    std::pmr::vector<std::pmr::vector<std::byte>> drawPushConstants;
    std::pmr::vector<MeshDispatchItem> meshDispatches;
    std::pmr::vector<TextureCopyItem> textureCopies;
    std::pmr::vector<std::pmr::string> meshDispatchDebugLabels;
    std::pmr::vector<std::pmr::vector<std::byte>> meshDispatchPushConstants;
    std::pmr::vector<std::pmr::vector<BufferHandle>>
        meshDispatchDependencyBuffers;
    std::pmr::vector<std::pmr::vector<TextureHandle>>
        meshDispatchDependencyTextures;

    explicit OwnedPassPayload(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : debugLabel(memory), preDispatches(memory),
          preDispatchDebugLabels(memory), preDispatchPushConstants(memory),
          preDispatchDependencyBuffers(memory),
          preDispatchDependencyTextures(memory), dependencyBuffers(memory),
          dependencyTextures(memory), draws(memory), drawDebugLabels(memory),
          drawPushConstants(memory), meshDispatches(memory),
          textureCopies(memory), meshDispatchDebugLabels(memory),
          meshDispatchPushConstants(memory),
          meshDispatchDependencyBuffers(memory),
          meshDispatchDependencyTextures(memory) {}
  };

  struct CompileWorkState {
    uint32_t passCount = 0u;
    uint32_t activePassCount = 0u;
    bool usedParallelValidation = false;
    bool usedParallelHazardAnalysis = false;
    std::pmr::vector<uint8_t> activePassMask;
    std::pmr::vector<DependencyEdge> scheduledDependencies;
    std::pmr::vector<uint32_t> order;
    std::pmr::vector<PassResourceAccess> compiledAccesses;

    explicit CompileWorkState(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : activePassMask(memory), scheduledDependencies(memory), order(memory),
          compiledAccesses(memory) {}
  };

  [[nodiscard]] bool isValidPassIndex(uint32_t passIndex) const {
    return passIndex < passes_.size();
  }
  [[nodiscard]] bool isValidTextureIndex(uint32_t textureIndex) const {
    return textureIndex < textures_.size();
  }
  [[nodiscard]] bool isValidBufferIndex(uint32_t bufferIndex) const {
    return bufferIndex < buffers_.size();
  }
  [[nodiscard]] Result<bool, std::string>
  addTextureAccessInternal(RenderGraphPassId pass, RenderGraphTextureId texture,
                           RenderGraphAccessMode mode, bool inferred);
  [[nodiscard]] Result<bool, std::string>
  addBufferAccessInternal(RenderGraphPassId pass, RenderGraphBufferId buffer,
                          RenderGraphAccessMode mode, bool inferred);
  [[nodiscard]] Result<bool, std::string>
  markPassSideEffectInternal(RenderGraphPassId pass, bool inferred);
  [[nodiscard]] OwnedPassPayload
  clonePassPayload(const RenderGraphGraphicsPassDesc &desc) const;
  [[nodiscard]] Result<bool, std::string>
  bindImplicitPassResources(RenderGraphPassId pass,
                            const RenderGraphGraphicsPassDesc &desc);
  [[nodiscard]] Result<bool, std::string>
  addPreResolvedDrawBufferAccesses(RenderGraphPassId pass,
                                   std::span<const BufferHandle> buffers,
                                   std::string_view debugLabel);
  [[nodiscard]] Result<bool, std::string> addPreResolvedDrawBufferAccesses(
      RenderGraphPassId pass, std::span<const RenderGraphBufferId> buffers);
  [[nodiscard]] Result<bool, std::string> bindPassDependencyTexture(
      RenderGraphPassId pass, uint32_t dependencyIndex,
      RenderGraphTextureId texture,
      RenderGraphAccessMode mode = RenderGraphAccessMode::Read);
  [[nodiscard]] Result<bool, std::string> appendPassDependencyTexture(
      RenderGraphPassId pass, RenderGraphTextureId texture,
      RenderGraphAccessMode mode = RenderGraphAccessMode::Read);
  [[nodiscard]] Result<bool, std::string>
  applyImplicitPassRoots(RenderGraphPassId pass,
                         const RenderGraphGraphicsPassDesc &desc);
  [[nodiscard]] Result<bool, std::string> applyGraphicsPassRoots(
      RenderGraphPassId pass, RenderGraphTextureId colorTexture,
      bool markColorAsFrameOutput, bool markImplicitOutputSideEffect);
  [[nodiscard]] Result<RenderGraphPassId, std::string>
  addPassRecord(RenderPass pass, std::string_view debugName);
  [[nodiscard]] Result<bool, std::string>
  compileStageC0ValidateInputs(RenderGraphRuntime &runtime,
                               RenderGraphCompileResult &compiled,
                               CompileWorkState &work) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC1C2BuildTopology(RenderGraphRuntime &runtime,
                                RenderGraphCompileResult &compiled,
                                CompileWorkState &work) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC3ResolvePassPayloads(RenderGraphRuntime &runtime,
                                    RenderGraphCompileResult &compiled,
                                    const CompileWorkState &work) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC4PlanBarriers(RenderGraphCompileResult &compiled,
                             const CompileWorkState &work) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC5PlanTransientLifetimes(RenderGraphRuntime &runtime,
                                       RenderGraphCompileResult &compiled,
                                       CompileWorkState &work) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC6PlanTransientAliasing(RenderGraphCompileResult &compiled) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC7ValidateCompiledMetadata(
      const RenderGraphCompileResult &compiled) const;

  std::pmr::memory_resource *memory_ = nullptr;
  uint64_t frameIndex_ = 0;
  std::pmr::vector<TextureResource> textures_;
  std::pmr::vector<BufferResource> buffers_;
  std::pmr::deque<OwnedPassPayload> ownedPassPayloads_;
  std::pmr::vector<RenderPass> passes_;
  std::pmr::vector<std::pmr::string> passDebugNames_;
  std::pmr::vector<uint32_t> passColorTextureBindings_;
  std::pmr::vector<uint32_t> passColorResolveTextureBindings_;
  std::pmr::vector<uint32_t> passDepthTextureBindings_;
  std::pmr::vector<uint32_t> passDepthResolveTextureBindings_;
  std::pmr::vector<uint32_t> passDependencyBufferBindingOffsets_;
  std::pmr::vector<uint32_t> passDependencyBufferBindingCounts_;
  std::pmr::vector<uint32_t> passDependencyBufferBindingResourceIndices_;
  std::pmr::vector<uint32_t> passDependencyTextureBindingOffsets_;
  std::pmr::vector<uint32_t> passDependencyTextureBindingCounts_;
  std::pmr::vector<uint32_t> passDependencyTextureBindingResourceIndices_;
  std::pmr::vector<uint32_t> passPreDispatchBindingOffsets_;
  std::pmr::vector<uint32_t> passPreDispatchBindingCounts_;
  std::pmr::vector<uint32_t> preDispatchDependencyBindingOffsets_;
  std::pmr::vector<uint32_t> preDispatchDependencyBindingCounts_;
  std::pmr::vector<uint32_t> preDispatchDependencyBindingResourceIndices_;
  std::pmr::vector<uint32_t> passDrawBindingOffsets_;
  std::pmr::vector<uint32_t> passDrawBindingCounts_;
  std::pmr::vector<uint32_t> drawVertexBindingResourceIndices_;
  std::pmr::vector<uint32_t> drawIndexBindingResourceIndices_;
  std::pmr::vector<uint32_t> drawIndirectBindingResourceIndices_;
  std::pmr::vector<uint32_t> drawIndirectCountBindingResourceIndices_;
  std::pmr::vector<uint32_t> passMeshDispatchBindingOffsets_;
  std::pmr::vector<uint32_t> passMeshDispatchBindingCounts_;
  std::pmr::vector<uint32_t> meshDispatchIndirectBindingResourceIndices_;
  std::pmr::vector<uint32_t> meshDispatchIndirectCountBindingResourceIndices_;
  std::pmr::vector<uint32_t> passTextureCopyBindingOffsets_;
  std::pmr::vector<uint32_t> passTextureCopyBindingCounts_;
  std::pmr::vector<uint32_t> textureCopySourceBindingResourceIndices_;
  std::pmr::vector<uint32_t> textureCopyDestinationBindingResourceIndices_;
  PmrHashMap<uint64_t, uint32_t> importedTextureIndicesByHandle_;
  PmrHashMap<uint64_t, uint32_t> importedBufferIndicesByHandle_;
  PmrHashMap<uint64_t, uint32_t> explicitTextureAccessIndicesByPassResource_;
  PmrHashMap<uint64_t, uint32_t> inferredTextureAccessIndicesByPassResource_;
  PmrHashMap<uint64_t, uint32_t> explicitBufferAccessIndicesByPassResource_;
  PmrHashMap<uint64_t, uint32_t> inferredBufferAccessIndicesByPassResource_;
  PmrHashSet<uint64_t> dependencyEdgeKeys_;
  std::pmr::vector<DependencyEdge> dependencies_;
  std::pmr::vector<PassResourceAccess> passResourceAccesses_;
  PmrHashSet<uint32_t> frameOutputTextureSet_;
  std::pmr::vector<uint32_t> frameOutputTextureIndices_;
  PmrHashMap<uint32_t, uint32_t> sideEffectMarkIndicesByPass_;
  std::pmr::vector<SideEffectPassMark> sideEffectPassMarks_;
  bool suppressInferredSideEffectsWhenExplicitOutputs_ = false;
  // Tracks whether every recorded pass this frame used borrowPayload = true.
  bool allPassesBorrowPayload_ = true;

  // Cross-frame persistent import tables. These are NOT cleared by
  // beginFrame(). Use std::vector (not pmr) so entries survive PMR arena
  // resets. beginFrame() does not pre-import them; callers choose the
  // importBuffer()/importTexture() order explicitly during BUILD and must pair
  // registerPersistentBuffer()/registerPersistentTexture() with
  // updatePersistentBuffer()/updatePersistentTexture() and
  // unregisterPersistentBuffer()/unregisterPersistentTexture() as handles
  // change across frames.
  struct PersistentBufferEntry {
    bool occupied = false;
    BufferHandle handle{};
    std::string debugName;
  };
  struct PersistentTextureEntry {
    bool occupied = false;
    TextureHandle handle{};
    std::string debugName;
  };
  std::vector<PersistentBufferEntry> persistentBuffers_;
  std::vector<PersistentTextureEntry> persistentTextures_;
  std::vector<uint32_t> persistentBufferFreeIndices_;
  std::vector<uint32_t> persistentTextureFreeIndices_;
  // Monotonically incremented whenever a persistent handle is registered,
  // replaced, or unregistered. Included in the fingerprint so that persistent
  // import-table changes always trigger a full recompile.
  uint64_t persistentHandlesVersion_ = 0;
  uint64_t transientResourceDescriptorsHash_ = 0xcbf29ce484222325ull;
  uint64_t dynamicPayloadVersion_ = 0u;
};

class NURI_API RenderGraphExecutor {
public:
  explicit RenderGraphExecutor(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  [[nodiscard]] Result<RenderGraphExecutionMetadata, std::string>
  execute(RenderGraphRuntime &runtime, GPUDevice &gpu,
          const RenderGraphCompileResult &compiled);

private:
  [[nodiscard]] Result<bool, std::string>
  executeInternal(RenderGraphRuntime *runtime, GPUDevice &gpu,
                  const RenderGraphCompileResult &compiled,
                  RenderGraphExecutionMetadata *metadata);

  struct PendingFrameResources {
    SubmissionHandle submission{};
    std::pmr::vector<TextureHandle> textures;
    std::pmr::vector<BufferHandle> buffers;
    std::pmr::vector<TextureDesc> textureDescs;
    std::pmr::vector<BufferDesc> bufferDescs;

    explicit PendingFrameResources(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : textures(memory), buffers(memory), textureDescs(memory),
          bufferDescs(memory) {}
  };

  struct ReusableTextureResource {
    TextureHandle handle{};
    TextureDesc desc{};
  };

  struct ReusableBufferResource {
    BufferHandle handle{};
    BufferDesc desc{};
  };

  void collectRetiredResources(GPUDevice &gpu);

  std::pmr::memory_resource *memory_ = nullptr;
  std::pmr::vector<PendingFrameResources> pendingFrames_;
  // Transient resource pools keyed by a hash of the descriptor fields checked
  // by isTextureDescAliasCompatible / isBufferDescAliasCompatible.  Each
  // bucket holds resources whose desc produces the same hash; since the hash
  // covers every compatibility field, all entries in a bucket are mutually
  // compatible and a lookup only ever needs to scan a single bucket (O(1)).
  PmrHashMap<uint64_t, std::pmr::vector<ReusableTextureResource>>
      reusableTexturesByHash_;
  PmrHashMap<uint64_t, std::pmr::vector<ReusableBufferResource>>
      reusableBuffersByHash_;
  size_t reusableTexturePoolSize_ = 0;
  size_t reusableBufferPoolSize_ = 0;
};

} // namespace nuri
