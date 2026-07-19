#pragma once
#include "nuri/core/containers/hash_map.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/render_graph/render_graph_runtime.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
namespace nuri {
template <typename Tag> struct RenderGraphId {
  uint32_t value = UINT32_MAX;
};
struct RenderGraphPassTag;
struct RenderGraphTextureTag;
struct RenderGraphBufferTag;
struct PersistentBufferTag;
struct PersistentTextureTag;
using RenderGraphPassId = RenderGraphId<RenderGraphPassTag>;
using RenderGraphTextureId = RenderGraphId<RenderGraphTextureTag>;
using RenderGraphBufferId = RenderGraphId<RenderGraphBufferTag>;
using PersistentBufferId = RenderGraphId<PersistentBufferTag>;
using PersistentTextureId = RenderGraphId<PersistentTextureTag>;
template <typename Tag>
[[nodiscard]] constexpr bool isValid(RenderGraphId<Tag> id) {
  return id.value != UINT32_MAX;
}
enum class RenderGraphDrawBufferBindingTarget : uint8_t {
  Vertex,
  Index,
  Indirect,
  IndirectCount,
};
enum class RenderGraphMeshDispatchBufferBindingTarget : uint8_t {
  Indirect,
  IndirectCount,
};
struct NURI_API RenderGraphPreparedMeshDispatchBufferBinding {
  uint32_t meshDispatchIndex = UINT32_MAX;
  RenderGraphMeshDispatchBufferBindingTarget target =
      RenderGraphMeshDispatchBufferBindingTarget::Indirect;
  RenderGraphBufferId buffer{};
  RenderGraphAccessMode mode = RenderGraphAccessMode::Read;
};
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
  ExternalTemporalDispatchItem externalTemporalDispatch{};
  bool drawBuffersPreResolved = false;
  std::span<const BufferHandle> preResolvedDrawBuffers{};
  std::span<const RenderGraphBufferId> preResolvedDrawBufferIds{};
  std::span<const RenderGraphPreparedMeshDispatchBufferBinding>
      meshDispatchBufferBindings{};
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
struct NURI_API RecordedGraphicsPassMeta {
  uint32_t orderedPassIndex = UINT32_MAX, declaredPassIndex = UINT32_MAX;
};
struct NURI_API RenderGraphPassExecutionTiming {
  uint32_t orderedPassIndex = UINT32_MAX;
  float cpuTimeMs = 0.0f;
};
struct NURI_API RecordedCommandBufferMeta {
  uint32_t firstOrderedPassIndex = UINT32_MAX, passCount = 0;
};
enum class RenderGraphBarrierResourceKind : uint8_t { Texture, Buffer };
using RenderGraphResourceState = GraphicsBarrierState;
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
  uint32_t orderedPassIndex = UINT32_MAX, barrierOffset = 0, barrierCount = 0;
};
struct NURI_API FinalBarrierPlan {
  uint32_t barrierOffset = 0, barrierCount = 0;
};
struct NURI_API RenderGraphPassRange {
  uint32_t workerIndex = UINT32_MAX, firstOrderedPassIndex = UINT32_MAX,
           passCount = 0;
};
enum class RenderGraphExecutionFailureStage : uint8_t {
  MaterializeTransients,
  AcquireRecordingContext,
  RecordGraphicsBarriers,
  RecordGraphicsPasses,
  FinishRecordingContext,
  SubmitRecordedFrame,
  PresentFrameOutput,
};
enum class RenderGraphTelemetryLevel : uint8_t {
  None,
  Metadata,
  PassTimings,
};
struct NURI_API RenderGraphExecutionOptions {
  RenderGraphTelemetryLevel telemetry = RenderGraphTelemetryLevel::None;
};
[[nodiscard]] NURI_API std::string_view
toString(RenderGraphExecutionFailureStage stage) noexcept;
struct NURI_API RenderGraphExecutionMetadata {
  SubmissionHandle submission{};
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
    uint32_t before = UINT32_MAX, after = UINT32_MAX;
  };
  struct ResourceStats {
    uint32_t importedTextures = 0, transientTextures = 0;
    uint32_t importedBuffers = 0, transientBuffers = 0;
  };
  struct TransientLifetime {
    uint32_t resourceIndex = UINT32_MAX, firstExecutionIndex = UINT32_MAX;
    uint32_t lastExecutionIndex = UINT32_MAX;
  };
  struct TransientAllocation {
    uint32_t resourceIndex = UINT32_MAX, allocationIndex = UINT32_MAX;
  };
  struct TransientTexturePhysicalAllocation {
    uint32_t allocationIndex = UINT32_MAX,
             representativeResourceIndex = UINT32_MAX;
    TextureDesc desc{};
  };
  struct TransientBufferPhysicalAllocation {
    uint32_t allocationIndex = UINT32_MAX,
             representativeResourceIndex = UINT32_MAX;
    BufferDesc desc{};
  };
  enum class PassTextureBindingTarget : uint8_t {
    Color,
    Depth,
    ColorResolve,
    DepthResolve,
  };
  struct PassTextureBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t textureResourceIndex = UINT32_MAX;
    PassTextureBindingTarget target = PassTextureBindingTarget::Color;
  };
  struct Range {
    uint32_t offset = 0, count = 0;
  };
  using PassDependencyBufferRange = Range;
  using PassDependencyTextureRange = Range;
  using PassDispatchRange = Range;
  using PassDrawRange = Range;
  using PassTextureCopyRange = Range;
  using DispatchDependencyBufferRange = Range;
  struct UnresolvedDependencyBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX, dependencyBufferIndex = UINT32_MAX;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };
  struct UnresolvedDependencyTextureBinding {
    uint32_t orderedPassIndex = UINT32_MAX, dependencyTextureIndex = UINT32_MAX;
    uint32_t textureResourceIndex = UINT32_MAX;
  };
  using DrawBufferBindingTarget = RenderGraphDrawBufferBindingTarget;
  using MeshDispatchBufferBindingTarget =
      RenderGraphMeshDispatchBufferBindingTarget;
  struct UnresolvedPreDispatchDependencyBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX, preDispatchIndex = UINT32_MAX;
    uint32_t dependencyBufferIndex = UINT32_MAX;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };
  struct UnresolvedDrawBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX, drawIndex = UINT32_MAX;
    DrawBufferBindingTarget target = DrawBufferBindingTarget::Vertex;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };
  struct UnresolvedMeshDispatchBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX, meshDispatchIndex = UINT32_MAX;
    MeshDispatchBufferBindingTarget target =
        MeshDispatchBufferBindingTarget::Indirect;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };
  enum class TextureCopyBindingTarget : uint8_t { Source, Destination };
  struct UnresolvedTextureCopyBinding {
    uint32_t orderedPassIndex = UINT32_MAX, textureCopyIndex = UINT32_MAX;
    TextureCopyBindingTarget target = TextureCopyBindingTarget::Source;
    uint32_t textureResourceIndex = UINT32_MAX;
  };
  uint64_t frameIndex = 0;
  uint32_t declaredPassCount = 0, culledPassCount = 0, rootPassCount = 0;
  uint32_t transientTexturePhysicalCount = 0, transientBufferPhysicalCount = 0;
  bool usedParallelCompile = false, usedParallelPayloadResolution = false;
  bool usedParallelHazardAnalysis = false, usedParallelLifetimeAnalysis = false;
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
  struct NURI_API GraphFingerprint {
    size_t passCount = 0;
    size_t totalTextureCount = 0;
    size_t totalBufferCount = 0;
    size_t edgeCount = 0;
    size_t passAccessCount = 0;
    size_t frameOutputCount = 0;
    size_t sideEffectMarkCount = 0;
    bool allPassesBorrowPayload = true;
    uint64_t payloadLayoutHash = 0;
    uint64_t structuralIdentityHash = 0;
    uint64_t transientResourceDescriptorsHash = 0;
    uint64_t persistentHandlesVersion = 0;
    [[nodiscard]] bool
    operator==(const GraphFingerprint &) const noexcept = default;
  };
  [[nodiscard]] GraphFingerprint computeGraphFingerprint() const noexcept;
  void refreshHandlesInCompileResult(RenderGraphCompileResult &result) const;
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
    uint32_t before = UINT32_MAX, after = UINT32_MAX;
  };
  enum class AccessResourceKind : uint8_t { Texture, Buffer };
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
    std::pmr::vector<std::pmr::vector<RenderGraphAccessMode>>
        preDispatchDependencyBufferAccessModes;
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
          preDispatchDependencyBufferAccessModes(memory),
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
  using BindingRange = RenderGraphCompileResult::Range;
  struct PassBindings {
    uint32_t color = UINT32_MAX, colorResolve = UINT32_MAX;
    uint32_t depth = UINT32_MAX, depthResolve = UINT32_MAX;
    BindingRange dependencyBuffers{}, dependencyTextures{}, preDispatches{};
    BindingRange draws{}, meshDispatches{}, textureCopies{};
  };
  struct DrawBindings {
    uint32_t vertex = UINT32_MAX, index = UINT32_MAX;
    uint32_t indirect = UINT32_MAX, indirectCount = UINT32_MAX;
  };
  struct MeshDispatchBindings {
    uint32_t indirect = UINT32_MAX, indirectCount = UINT32_MAX;
  };
  struct TextureCopyBindings {
    uint32_t source = UINT32_MAX, destination = UINT32_MAX;
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
  [[nodiscard]] Result<bool, std::string>
  applyImplicitPassRoots(RenderGraphPassId pass,
                         const RenderGraphGraphicsPassDesc &desc);
  [[nodiscard]] Result<bool, std::string> applyGraphicsPassRoots(
      RenderGraphPassId pass, RenderGraphTextureId colorTexture,
      bool markColorAsFrameOutput, bool markImplicitOutputSideEffect);
  [[nodiscard]] Result<RenderGraphPassId, std::string>
  addPassRecord(RenderPass pass, std::string_view debugName);
  void compileStageC0BuildResourceTables(RenderGraphCompileResult &compiled,
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
  std::pmr::memory_resource *memory_ = nullptr;
  uint64_t frameIndex_ = 0;
  std::pmr::vector<TextureResource> textures_;
  std::pmr::vector<BufferResource> buffers_;
  std::pmr::deque<OwnedPassPayload> ownedPassPayloads_;
  std::pmr::vector<RenderPass> passes_;
  std::pmr::vector<std::pmr::string> passDebugNames_;
  std::pmr::vector<PassBindings> passBindings_;
  std::pmr::vector<uint32_t> passDependencyBufferBindingResourceIndices_;
  std::pmr::vector<uint32_t> passDependencyTextureBindingResourceIndices_;
  std::pmr::vector<BindingRange> preDispatchDependencyBindings_;
  std::pmr::vector<uint32_t> preDispatchDependencyBindingResourceIndices_;
  std::pmr::vector<DrawBindings> drawBindings_;
  std::pmr::vector<MeshDispatchBindings> meshDispatchBindings_;
  std::pmr::vector<TextureCopyBindings> textureCopyBindings_;
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
  bool allPassesBorrowPayload_ = true;
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
  uint64_t persistentHandlesVersion_ = 0;
  uint64_t transientResourceDescriptorsHash_ = 0xcbf29ce484222325ull;
};
class NURI_API RenderGraphExecutor {
public:
  explicit RenderGraphExecutor(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  [[nodiscard]] Result<RenderGraphExecutionMetadata, std::string>
  execute(RenderGraphRuntime &runtime, GPUDevice &gpu,
          const RenderGraphCompileResult &compiled,
          RenderGraphExecutionOptions options = {});

private:
  [[nodiscard]] Result<bool, std::string>
  executeInternal(RenderGraphRuntime *runtime, GPUDevice &gpu,
                  const RenderGraphCompileResult &compiled,
                  RenderGraphExecutionMetadata &metadata,
                  RenderGraphExecutionOptions options);
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
  PmrHashMap<uint64_t, std::pmr::vector<ReusableTextureResource>>
      reusableTexturesByHash_;
  PmrHashMap<uint64_t, std::pmr::vector<ReusableBufferResource>>
      reusableBuffersByHash_;
  size_t reusableTexturePoolSize_ = 0;
  size_t reusableBufferPoolSize_ = 0;
};
} // namespace nuri
