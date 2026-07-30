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
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <variant>
namespace nuri {
template <typename Tag> struct RenderGraphId {
  uint32_t value = UINT32_MAX;
};
struct RenderGraphPassTag;
struct RenderGraphTextureTag;
struct RenderGraphBufferTag;
struct RenderGraphAccelerationStructureTag;
struct PersistentBufferTag;
struct PersistentTextureTag;
using RenderGraphPassId = RenderGraphId<RenderGraphPassTag>;
using RenderGraphTextureId = RenderGraphId<RenderGraphTextureTag>;
using RenderGraphBufferId = RenderGraphId<RenderGraphBufferTag>;
using RenderGraphAccelerationStructureId =
    RenderGraphId<RenderGraphAccelerationStructureTag>;
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
struct NURI_API RenderGraphBufferUse {
  RenderGraphBufferId buffer{};
  RenderGraphAccessMode access = RenderGraphAccessMode::Read;
};
struct NURI_API RenderGraphTextureUse {
  RenderGraphTextureId texture{};
  RenderGraphAccessMode access = RenderGraphAccessMode::Read;
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
  std::span<const SamplerHandle> recordingSamplers{};
  std::span<const RenderGraphBufferUse> bufferUses{};
  std::span<const RenderGraphTextureUse> textureUses{};
  std::span<const RenderGraphImportedBufferUse> importedBufferUses{};
  std::span<const RenderGraphImportedTextureUse> importedTextureUses{};
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
};
enum class RenderGraphAccelerationStructureAccess : uint8_t {
  BuildRead,
  BuildWrite,
  RayQueryRead,
};
struct NURI_API RenderGraphAccelerationStructureUse {
  RenderGraphAccelerationStructureId accelerationStructure{};
  RenderGraphAccelerationStructureAccess access =
      RenderGraphAccelerationStructureAccess::BuildRead;
};
struct NURI_API RenderGraphAccelerationStructurePassDesc {
  std::span<const AccelerationStructureBuildItem> builds{};
  std::span<const RenderGraphBufferUse> buffers{};
  std::span<const RenderGraphAccelerationStructureUse> accelerationStructures{};
  GpuTimingScope gpuTimingScope = GpuTimingScope::None;
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
  bool markImplicitOutputSideEffect = true;
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
struct NURI_API RenderGraphBufferCopyItem {
  RenderGraphBufferId sourceBuffer{};
  RenderGraphBufferId destinationBuffer{};
  uint64_t sourceOffset = 0u;
  uint64_t destinationOffset = 0u;
  uint64_t size = 0u;
};
struct NURI_API RenderGraphBufferCopyPassDesc {
  std::span<const RenderGraphBufferCopyItem> copies{};
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
enum class RenderGraphBarrierResourceKind : uint8_t {
  Texture,
  Buffer,
  AccelerationStructure,
};
using RenderGraphResourceState = GraphicsBarrierState;
enum class RenderGraphResourceKind : uint8_t {
  Texture,
  Buffer,
  AccelerationStructure,
};
enum class RenderGraphResourceUseProvenance : uint8_t {
  Explicit,
  Inferred,
};
using RenderGraphSubresourceRange = GraphicsTextureSubresourceRange;
struct NURI_API RenderGraphResourceUse {
  uint32_t passIndex = UINT32_MAX;
  RenderGraphResourceKind resourceKind = RenderGraphResourceKind::Texture;
  uint32_t resourceIndex = UINT32_MAX;
  RenderGraphAccessMode access = RenderGraphAccessMode::None;
  RenderGraphResourceState state = RenderGraphResourceState::Unknown;
  RenderPassExecutionMode stage = RenderPassExecutionMode::Graphics;
  RenderGraphSubresourceRange subresources{};
  RenderGraphResourceUseProvenance provenance =
      RenderGraphResourceUseProvenance::Explicit;
};
struct NURI_API RenderGraphBarrierRecord {
  RenderGraphBarrierResourceKind resourceKind =
      RenderGraphBarrierResourceKind::Texture;
  uint32_t resourceIndex = UINT32_MAX;
  RenderGraphAccessMode beforeAccess = RenderGraphAccessMode::None;
  RenderGraphAccessMode afterAccess = RenderGraphAccessMode::None;
  RenderGraphResourceState beforeState = RenderGraphResourceState::Unknown;
  RenderGraphResourceState afterState = RenderGraphResourceState::Unknown;
  RenderGraphSubresourceRange subresources{};
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
struct NURI_API RenderGraphPlan {
  struct Edge {
    uint32_t before = UINT32_MAX, after = UINT32_MAX;
  };
  struct ResourceStats {
    uint32_t importedTextures = 0, transientTextures = 0;
    uint32_t importedBuffers = 0, transientBuffers = 0;
    uint32_t importedAccelerationStructures = 0;
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
  template <typename Tag> struct PassResourcePatch {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t resourceIndex = UINT32_MAX;
  };
  template <typename Tag> struct CommandResourcePatchSite {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t commandIndex = 0u;
    uint32_t resourceIndex = UINT32_MAX;
  };
  struct PassColorTexturePatchTag;
  struct PassDepthTexturePatchTag;
  struct PassColorResolveTexturePatchTag;
  struct PassDepthResolveTexturePatchTag;
  struct DrawVertexBufferPatchTag;
  struct DrawIndexBufferPatchTag;
  struct DrawIndirectBufferPatchTag;
  struct DrawIndirectCountBufferPatchTag;
  struct MeshDispatchIndirectBufferPatchTag;
  struct MeshDispatchIndirectCountBufferPatchTag;
  struct BufferCopySourcePatchTag;
  struct BufferCopyDestinationPatchTag;
  struct TextureCopySourcePatchTag;
  struct TextureCopyDestinationPatchTag;
  using PassColorTexturePatch = PassResourcePatch<PassColorTexturePatchTag>;
  using PassDepthTexturePatch = PassResourcePatch<PassDepthTexturePatchTag>;
  using PassColorResolveTexturePatch =
      PassResourcePatch<PassColorResolveTexturePatchTag>;
  using PassDepthResolveTexturePatch =
      PassResourcePatch<PassDepthResolveTexturePatchTag>;
  using DrawVertexBufferPatch =
      CommandResourcePatchSite<DrawVertexBufferPatchTag>;
  using DrawIndexBufferPatch =
      CommandResourcePatchSite<DrawIndexBufferPatchTag>;
  using DrawIndirectBufferPatch =
      CommandResourcePatchSite<DrawIndirectBufferPatchTag>;
  using DrawIndirectCountBufferPatch =
      CommandResourcePatchSite<DrawIndirectCountBufferPatchTag>;
  using MeshDispatchIndirectBufferPatch =
      CommandResourcePatchSite<MeshDispatchIndirectBufferPatchTag>;
  using MeshDispatchIndirectCountBufferPatch =
      CommandResourcePatchSite<MeshDispatchIndirectCountBufferPatchTag>;
  using BufferCopySourcePatch =
      CommandResourcePatchSite<BufferCopySourcePatchTag>;
  using BufferCopyDestinationPatch =
      CommandResourcePatchSite<BufferCopyDestinationPatchTag>;
  using TextureCopySourcePatch =
      CommandResourcePatchSite<TextureCopySourcePatchTag>;
  using TextureCopyDestinationPatch =
      CommandResourcePatchSite<TextureCopyDestinationPatchTag>;
  using CommandResourcePatch =
      std::variant<PassColorTexturePatch, PassDepthTexturePatch,
                   PassColorResolveTexturePatch, PassDepthResolveTexturePatch,
                   DrawVertexBufferPatch, DrawIndexBufferPatch,
                   DrawIndirectBufferPatch, DrawIndirectCountBufferPatch,
                   MeshDispatchIndirectBufferPatch,
                   MeshDispatchIndirectCountBufferPatch, BufferCopySourcePatch,
                   BufferCopyDestinationPatch, TextureCopySourcePatch,
                   TextureCopyDestinationPatch>;
  struct Range {
    uint32_t offset = 0, count = 0;
  };
  using PassDispatchRange = Range;
  using PassDrawRange = Range;
  using PassBufferCopyRange = Range;
  using PassTextureCopyRange = Range;
  uint32_t declaredPassCount = 0, culledPassCount = 0, rootPassCount = 0;
  uint32_t transientTexturePhysicalCount = 0, transientBufferPhysicalCount = 0;
  bool usedParallelCompile = false;
  bool usedParallelHazardAnalysis = false, usedParallelLifetimeAnalysis = false;
  ResourceStats resourceStats{};
  std::pmr::vector<RenderGraphResourceUse> resourceUses;
  std::pmr::vector<uint32_t> orderedPassIndices;
  std::pmr::vector<RecordedGraphicsPassMeta> recordedGraphicsPasses;
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
  std::pmr::vector<CommandResourcePatch> commandResourcePatches;
  std::pmr::vector<PassDispatchRange> preDispatchRangesByPass;
  std::pmr::vector<PassDrawRange> drawRangesByPass;
  std::pmr::vector<PassDispatchRange> meshDispatchRangesByPass;
  std::pmr::vector<PassBufferCopyRange> bufferCopyRangesByPass;
  std::pmr::vector<PassTextureCopyRange> textureCopyRangesByPass;
  explicit RenderGraphPlan(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : resourceUses(ensureMemory(memory)),
        orderedPassIndices(ensureMemory(memory)),
        recordedGraphicsPasses(ensureMemory(memory)),
        edges(ensureMemory(memory)), passBarrierPlans(ensureMemory(memory)),
        passBarrierRecords(ensureMemory(memory)),
        transientTextureLifetimes(ensureMemory(memory)),
        transientBufferLifetimes(ensureMemory(memory)),
        transientTextureAllocations(ensureMemory(memory)),
        transientBufferAllocations(ensureMemory(memory)),
        transientTextureAllocationByResource(ensureMemory(memory)),
        transientBufferAllocationByResource(ensureMemory(memory)),
        transientTexturePhysicalAllocations(ensureMemory(memory)),
        transientBufferPhysicalAllocations(ensureMemory(memory)),
        commandResourcePatches(ensureMemory(memory)),
        preDispatchRangesByPass(ensureMemory(memory)),
        drawRangesByPass(ensureMemory(memory)),
        meshDispatchRangesByPass(ensureMemory(memory)),
        bufferCopyRangesByPass(ensureMemory(memory)),
        textureCopyRangesByPass(ensureMemory(memory)) {}

private:
  static std::pmr::memory_resource *ensureMemory(std::pmr::memory_resource *m) {
    return m != nullptr ? m : std::pmr::get_default_resource();
  }
};
struct NURI_API FrameCommandArena {
  uint64_t frameIndex = 0;
  std::pmr::vector<TextureHandle> textureHandlesByResource;
  std::pmr::vector<BufferHandle> bufferHandlesByResource;
  std::pmr::vector<AccelerationStructureHandle>
      accelerationStructureHandlesByResource;
  std::pmr::vector<RenderPass> orderedPasses;
  std::pmr::vector<std::pmr::string> passDebugNames;
  std::pmr::vector<std::pmr::vector<SamplerHandle>>
      ownedRecordingSamplersByPass;
  std::pmr::vector<ComputeDispatchItem> ownedPreDispatches;
  std::pmr::vector<std::pmr::string> ownedPreDispatchDebugLabels;
  std::pmr::vector<std::pmr::vector<std::byte>> ownedPreDispatchPushConstants;
  std::pmr::vector<std::pmr::vector<PushConstantTextureBinding>>
      ownedPreDispatchTextureBindings;
  std::pmr::vector<DrawItem> ownedDrawItems;
  std::pmr::vector<std::pmr::string> ownedDrawDebugLabels;
  std::pmr::vector<std::pmr::vector<std::byte>> ownedDrawPushConstants;
  std::pmr::vector<std::pmr::vector<PushConstantTextureBinding>>
      ownedDrawTextureBindings;
  std::pmr::vector<MeshDispatchItem> ownedMeshDispatchItems;
  std::pmr::vector<BufferCopyRegion> ownedBufferCopyItems;
  std::pmr::vector<TextureCopyItem> ownedTextureCopyItems;
  std::pmr::vector<std::pmr::vector<AccelerationStructureBuildItem>>
      ownedAccelerationStructureBuildsByPass;
  std::pmr::vector<std::pmr::vector<
      std::pmr::vector<AccelerationStructureTriangleGeometryDesc>>>
      ownedAccelerationStructureGeometriesByPass;
  std::pmr::vector<
      std::pmr::vector<std::pmr::vector<AccelerationStructureInstanceDesc>>>
      ownedAccelerationStructureInstancesByPass;
  std::pmr::vector<std::pmr::string> ownedMeshDispatchDebugLabels;
  std::pmr::vector<std::pmr::vector<std::byte>> ownedMeshDispatchPushConstants;
  std::pmr::vector<std::pmr::vector<PushConstantTextureBinding>>
      ownedMeshDispatchTextureBindings;

  explicit FrameCommandArena(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : textureHandlesByResource(ensureMemory(memory)),
        bufferHandlesByResource(ensureMemory(memory)),
        accelerationStructureHandlesByResource(ensureMemory(memory)),
        orderedPasses(ensureMemory(memory)),
        passDebugNames(ensureMemory(memory)),
        ownedRecordingSamplersByPass(ensureMemory(memory)),
        ownedPreDispatches(ensureMemory(memory)),
        ownedPreDispatchDebugLabels(ensureMemory(memory)),
        ownedPreDispatchPushConstants(ensureMemory(memory)),
        ownedPreDispatchTextureBindings(ensureMemory(memory)),
        ownedDrawItems(ensureMemory(memory)),
        ownedDrawDebugLabels(ensureMemory(memory)),
        ownedDrawPushConstants(ensureMemory(memory)),
        ownedDrawTextureBindings(ensureMemory(memory)),
        ownedMeshDispatchItems(ensureMemory(memory)),
        ownedBufferCopyItems(ensureMemory(memory)),
        ownedTextureCopyItems(ensureMemory(memory)),
        ownedAccelerationStructureBuildsByPass(ensureMemory(memory)),
        ownedAccelerationStructureGeometriesByPass(ensureMemory(memory)),
        ownedAccelerationStructureInstancesByPass(ensureMemory(memory)),
        ownedMeshDispatchDebugLabels(ensureMemory(memory)),
        ownedMeshDispatchPushConstants(ensureMemory(memory)),
        ownedMeshDispatchTextureBindings(ensureMemory(memory)) {}

private:
  static std::pmr::memory_resource *ensureMemory(std::pmr::memory_resource *m) {
    return m != nullptr ? m : std::pmr::get_default_resource();
  }
};
struct NURI_API CompiledRenderGraph {
  RenderGraphPlan plan;
  FrameCommandArena commands;

  explicit CompiledRenderGraph(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : plan(memory), commands(memory) {}
};
struct NURI_API CompiledRenderGraphView {
  const RenderGraphPlan &plan;
  FrameCommandArena &commands;
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
  [[nodiscard]] Result<RenderGraphAccelerationStructureId, std::string>
  importAccelerationStructure(AccelerationStructureHandle accelerationStructure,
                              std::string_view debugName = {});
  [[nodiscard]] Result<RenderGraphTextureId, std::string>
  createTransientTexture(const TextureDesc &desc,
                         std::string_view debugName = {});
  [[nodiscard]] Result<RenderGraphBufferId, std::string>
  createTransientBuffer(const BufferDesc &desc,
                        std::string_view debugName = {});
  [[nodiscard]] Result<bool, std::string>
  addTextureAccess(RenderGraphPassId pass, RenderGraphTextureId texture,
                   RenderGraphAccessMode mode,
                   RenderGraphSubresourceRange subresources = {});
  [[nodiscard]] Result<bool, std::string>
  addBufferAccess(RenderGraphPassId pass, RenderGraphBufferId buffer,
                  RenderGraphAccessMode mode);
  [[nodiscard]] Result<bool, std::string>
  addImportedTextureAccess(RenderGraphPassId pass, TextureHandle texture,
                           RenderGraphAccessMode mode,
                           std::string_view debugName = {});
  [[nodiscard]] Result<bool, std::string>
  addImportedBufferAccess(RenderGraphPassId pass, BufferHandle buffer,
                          RenderGraphAccessMode mode,
                          std::string_view debugName = {});
  [[nodiscard]] Result<bool, std::string>
  addImportedTextureReads(RenderGraphPassId pass,
                          std::span<const TextureHandle> textures,
                          std::string_view debugName = {});
  [[nodiscard]] Result<bool, std::string>
  addImportedBufferReads(RenderGraphPassId pass,
                         std::span<const BufferHandle> buffers,
                         std::string_view debugName = {});
  [[nodiscard]] Result<bool, std::string> addImportedTextureAccesses(
      RenderGraphPassId pass,
      std::span<const RenderGraphImportedTextureUse> uses,
      std::string_view debugName = {});
  [[nodiscard]] Result<bool, std::string>
  addImportedBufferAccesses(RenderGraphPassId pass,
                            std::span<const RenderGraphImportedBufferUse> uses,
                            std::string_view debugName = {});
  [[nodiscard]] Result<bool, std::string> addAccelerationStructureAccess(
      RenderGraphPassId pass,
      RenderGraphAccelerationStructureId accelerationStructure,
      RenderGraphAccelerationStructureAccess access);
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
  [[nodiscard]] Result<RenderGraphPassId, std::string>
  addBufferCopyPass(const RenderGraphBufferCopyPassDesc &desc);
  [[nodiscard]] Result<RenderGraphPassId, std::string>
  addAccelerationStructurePass(
      const RenderGraphAccelerationStructurePassDesc &desc);
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
  [[nodiscard]] Result<bool, std::string>
  bindDrawBuffer(RenderGraphPassId pass, uint32_t drawIndex,
                 RenderGraphDrawBufferBindingTarget target,
                 RenderGraphBufferId buffer,
                 RenderGraphAccessMode mode = RenderGraphAccessMode::Read);
  [[nodiscard]] Result<bool, std::string> bindMeshDispatchBuffer(
      RenderGraphPassId pass, uint32_t meshDispatchIndex,
      RenderGraphMeshDispatchBufferBindingTarget target,
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
  [[nodiscard]] Result<CompiledRenderGraph, std::string>
  compile(RenderGraphRuntime &runtime);
  [[nodiscard]] size_t passCount() const noexcept { return passes_.size(); }
  struct NURI_API GraphFingerprint {
    size_t passCount = 0;
    size_t totalTextureCount = 0;
    size_t totalBufferCount = 0;
    size_t totalAccelerationStructureCount = 0;
    size_t edgeCount = 0;
    size_t passAccessCount = 0;
    size_t frameOutputCount = 0;
    size_t sideEffectMarkCount = 0;
    uint64_t payloadLayoutHash = 0;
    uint64_t structuralIdentityHash = 0;
    uint64_t transientResourceDescriptorsHash = 0;
    uint64_t persistentHandlesVersion = 0;
    [[nodiscard]] bool
    operator==(const GraphFingerprint &) const noexcept = default;
  };
  [[nodiscard]] GraphFingerprint computeGraphFingerprint() const noexcept;
  [[nodiscard]] FrameCommandArena
  buildFrameCommands(const RenderGraphPlan &plan);
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
  struct AccelerationStructureResource {
    AccelerationStructureHandle importedHandle{};
    std::pmr::string debugName;
    explicit AccelerationStructureResource(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : debugName(memory) {}
  };
  struct CompileWorkState {
    uint32_t passCount = 0u;
    uint32_t activePassCount = 0u;
    bool usedParallelHazardAnalysis = false;
    std::pmr::vector<uint8_t> activePassMask;
    std::pmr::vector<DependencyEdge> scheduledDependencies;
    std::pmr::vector<uint32_t> order;
    std::pmr::vector<RenderGraphResourceUse> compiledResourceUses;
    explicit CompileWorkState(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : activePassMask(memory), scheduledDependencies(memory), order(memory),
          compiledResourceUses(memory) {}
  };
  using BindingRange = RenderGraphPlan::Range;
  struct PassBindings {
    uint32_t color = UINT32_MAX, colorResolve = UINT32_MAX;
    uint32_t depth = UINT32_MAX, depthResolve = UINT32_MAX;
    BindingRange preDispatches{};
    BindingRange drawPayloads{}, draws{}, meshDispatches{}, bufferCopies{},
        textureCopies{};
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
  struct BufferCopyBindings {
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
  [[nodiscard]] bool
  isValidAccelerationStructureIndex(uint32_t accelerationStructureIndex) const {
    return accelerationStructureIndex < accelerationStructures_.size();
  }
  [[nodiscard]] Result<bool, std::string>
  addTextureAccessInternal(RenderGraphPassId pass, RenderGraphTextureId texture,
                           RenderGraphAccessMode mode, bool inferred,
                           RenderGraphSubresourceRange subresources = {});
  [[nodiscard]] Result<bool, std::string>
  addBufferAccessInternal(RenderGraphPassId pass, RenderGraphBufferId buffer,
                          RenderGraphAccessMode mode, bool inferred,
                          RenderGraphResourceState requestedState =
                              RenderGraphResourceState::Unknown);
  [[nodiscard]] Result<bool, std::string>
  addAccelerationStructureAccessInternal(
      RenderGraphPassId pass,
      RenderGraphAccelerationStructureId accelerationStructure,
      RenderGraphAccelerationStructureAccess access, bool inferred);
  [[nodiscard]] Result<bool, std::string>
  markPassSideEffectInternal(RenderGraphPassId pass, bool inferred);
  void appendGraphicsPayload(const RenderGraphGraphicsPassDesc &desc,
                             RenderPass &pass);
  void refreshPassViews();
  [[nodiscard]] Result<bool, std::string>
  bindImplicitPassResources(RenderGraphPassId pass,
                            const RenderGraphGraphicsPassDesc &desc);
  [[nodiscard]] Result<bool, std::string>
  addPreResolvedDrawBufferAccesses(RenderGraphPassId pass,
                                   std::span<const BufferHandle> buffers,
                                   std::string_view debugLabel);
  [[nodiscard]] Result<bool, std::string> addPreResolvedDrawBufferAccesses(
      RenderGraphPassId pass, std::span<const RenderGraphBufferId> buffers);
  [[nodiscard]] Result<bool, std::string>
  applyImplicitPassRoots(RenderGraphPassId pass,
                         const RenderGraphGraphicsPassDesc &desc);
  [[nodiscard]] Result<bool, std::string> applyGraphicsPassRoots(
      RenderGraphPassId pass, RenderGraphTextureId colorTexture,
      bool markColorAsFrameOutput, bool markImplicitOutputSideEffect);
  [[nodiscard]] Result<RenderGraphPassId, std::string>
  addPassRecord(RenderPass pass, std::string_view debugName);
  void compileStageC0BuildResourceTables(CompiledRenderGraph &compiled,
                                         CompileWorkState &work) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC1C2BuildTopology(RenderGraphRuntime &runtime,
                                CompiledRenderGraph &compiled,
                                CompileWorkState &work) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC3ResolvePassPayloads(RenderGraphRuntime &runtime,
                                    CompiledRenderGraph &compiled,
                                    const CompileWorkState &work) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC4PlanBarriers(CompiledRenderGraph &compiled,
                             const CompileWorkState &work) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC5PlanTransientLifetimes(RenderGraphRuntime &runtime,
                                       CompiledRenderGraph &compiled,
                                       CompileWorkState &work) const;
  [[nodiscard]] Result<bool, std::string>
  compileStageC6PlanTransientAliasing(CompiledRenderGraph &compiled) const;
  std::pmr::memory_resource *memory_ = nullptr;
  uint64_t frameIndex_ = 0;
  std::pmr::vector<TextureResource> textures_;
  std::pmr::vector<BufferResource> buffers_;
  std::pmr::vector<AccelerationStructureResource> accelerationStructures_;
  FrameCommandArena currentCommands_;
  std::pmr::vector<RenderPass> passes_;
  std::pmr::vector<std::pmr::string> passDebugNames_;
  std::pmr::vector<PassBindings> passBindings_;
  std::pmr::vector<DrawBindings> drawBindings_;
  std::pmr::vector<MeshDispatchBindings> meshDispatchBindings_;
  std::pmr::vector<BufferCopyBindings> bufferCopyBindings_;
  std::pmr::vector<TextureCopyBindings> textureCopyBindings_;
  PmrHashMap<uint64_t, uint32_t> importedTextureIndicesByHandle_;
  PmrHashMap<uint64_t, uint32_t> importedBufferIndicesByHandle_;
  PmrHashMap<uint64_t, uint32_t> importedAccelerationStructureIndicesByHandle_;
  PmrHashMap<uint64_t, uint32_t> explicitBufferAccessIndicesByPassResource_;
  PmrHashMap<uint64_t, uint32_t> inferredBufferAccessIndicesByPassResource_;
  PmrHashMap<uint64_t, uint32_t>
      explicitAccelerationStructureAccessIndicesByPassResource_;
  PmrHashMap<uint64_t, uint32_t>
      inferredAccelerationStructureAccessIndicesByPassResource_;
  PmrHashSet<uint64_t> dependencyEdgeKeys_;
  std::pmr::vector<DependencyEdge> dependencies_;
  std::pmr::vector<RenderGraphResourceUse> resourceUses_;
  PmrHashSet<uint32_t> frameOutputTextureSet_;
  std::pmr::vector<uint32_t> frameOutputTextureIndices_;
  PmrHashMap<uint32_t, uint32_t> sideEffectMarkIndicesByPass_;
  std::pmr::vector<SideEffectPassMark> sideEffectPassMarks_;
  bool suppressInferredSideEffectsWhenExplicitOutputs_ = false;
  bool commandsTransferred_ = false;
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
          CompiledRenderGraphView compiled,
          RenderGraphExecutionOptions options = {});

private:
  [[nodiscard]] Result<bool, std::string>
  executeInternal(RenderGraphRuntime *runtime, GPUDevice &gpu,
                  CompiledRenderGraphView compiled,
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
