#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/render_graph/render_graph.h"

#include <filesystem>
#include <memory_resource>
#include <string>
#include <string_view>

namespace nuri {

struct NURI_API RenderGraphTelemetrySnapshot {
  struct Summary {
    uint64_t frameIndex = 0;
    uint32_t declaredPassCount = 0;
    uint32_t culledPassCount = 0;
    uint32_t rootPassCount = 0;
    uint32_t passCount = 0;
    uint32_t edgeCount = 0;
    uint32_t recordedGraphicsPassCount = 0;
    uint32_t passBarrierPlanCount = 0;
    uint32_t finalBarrierRecordCount = 0;
    uint32_t passBarrierRecordCount = 0;
    uint32_t recordedCommandBufferCount = 0;
    uint32_t submitBatchCount = 0;
    uint32_t passRangeCount = 0;
    uint32_t importedTextures = 0;
    uint32_t transientTextures = 0;
    uint32_t importedBuffers = 0;
    uint32_t transientBuffers = 0;
    uint32_t transientTextureLifetimeCount = 0;
    uint32_t transientBufferLifetimeCount = 0;
    uint32_t transientTexturePhysicalCount = 0;
    uint32_t transientBufferPhysicalCount = 0;
    uint32_t transientTextureAllocationMapSize = 0;
    uint32_t transientBufferAllocationMapSize = 0;
    uint32_t transientTexturePhysicalAllocationCount = 0;
    uint32_t transientBufferPhysicalAllocationCount = 0;
    uint32_t unresolvedTextureBindingCount = 0;
    uint32_t resolvedDependencyBufferSlotCount = 0;
    uint32_t unresolvedDependencyBufferBindingCount = 0;
    uint32_t ownedPreDispatchCount = 0;
    uint32_t ownedDrawItemCount = 0;
    uint32_t resolvedPreDispatchDependencyBufferSlotCount = 0;
    uint32_t unresolvedPreDispatchDependencyBufferBindingCount = 0;
    uint32_t unresolvedDrawBufferBindingCount = 0;
    bool usedParallelCompile = false;
    bool usedParallelValidation = false;
    bool usedParallelPayloadResolution = false;
    bool usedParallelHazardAnalysis = false;
    bool usedParallelLifetimeAnalysis = false;
    bool usedParallelRecording = false;
    uint64_t compileFingerprint = 0;
    uint64_t barrierFingerprint = 0;
    uint64_t executionFingerprint = 0;
  };

  Summary summary{};
  std::pmr::vector<std::pmr::string> passNames;
  std::pmr::vector<uint32_t> orderedPassIndices;
  std::pmr::vector<RecordedGraphicsPassMeta> recordedGraphicsPasses;
  std::pmr::vector<RenderGraphCompileResult::Edge> edges;
  std::pmr::vector<PassBarrierPlan> passBarrierPlans;
  FinalBarrierPlan finalBarrierPlan{};
  std::pmr::vector<RenderGraphBarrierRecord> passBarrierRecords;
  std::pmr::vector<RecordedCommandBufferMeta> recordedCommandBuffers;
  std::pmr::vector<SubmitBatchMeta> submitBatches;
  std::pmr::vector<RenderGraphPassRange> passRanges;
  std::pmr::vector<RenderGraphCompileResult::TransientLifetime>
      transientTextureLifetimes;
  std::pmr::vector<RenderGraphCompileResult::TransientLifetime>
      transientBufferLifetimes;
  std::pmr::vector<RenderGraphCompileResult::TransientAllocation>
      transientTextureAllocations;
  std::pmr::vector<RenderGraphCompileResult::TransientAllocation>
      transientBufferAllocations;
  std::pmr::vector<uint32_t> transientTextureAllocationByResource;
  std::pmr::vector<uint32_t> transientBufferAllocationByResource;
  std::pmr::vector<RenderGraphCompileResult::TransientTexturePhysicalAllocation>
      transientTexturePhysicalAllocations;
  std::pmr::vector<RenderGraphCompileResult::TransientBufferPhysicalAllocation>
      transientBufferPhysicalAllocations;
  std::pmr::vector<RenderGraphCompileResult::PassTextureBinding>
      unresolvedTextureBindings;
  std::pmr::vector<BufferHandle> resolvedDependencyBuffers;
  std::pmr::vector<RenderGraphCompileResult::PassDependencyBufferRange>
      dependencyBufferRangesByPass;
  std::pmr::vector<RenderGraphCompileResult::UnresolvedDependencyBufferBinding>
      unresolvedDependencyBufferBindings;
  std::pmr::vector<RenderGraphCompileResult::PassDispatchRange>
      preDispatchRangesByPass;
  std::pmr::vector<RenderGraphCompileResult::DispatchDependencyBufferRange>
      preDispatchDependencyRanges;
  std::pmr::vector<
      RenderGraphCompileResult::UnresolvedPreDispatchDependencyBufferBinding>
      unresolvedPreDispatchDependencyBufferBindings;
  std::pmr::vector<RenderGraphCompileResult::PassDrawRange> drawRangesByPass;
  std::pmr::vector<RenderGraphCompileResult::UnresolvedDrawBufferBinding>
      unresolvedDrawBufferBindings;

  explicit RenderGraphTelemetrySnapshot(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());

  void captureFrom(const RenderGraphCompileResult &compiled);
  void captureFrom(const RenderGraphCompileResult &compiled,
                   const RenderGraphExecutionMetadata &execution);
  void reset();
};

class NURI_API RenderGraphTelemetryService {
public:
  explicit RenderGraphTelemetryService(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());

  void capture(const RenderGraphCompileResult &compiled);
  void capture(const RenderGraphCompileResult &compiled,
               const RenderGraphExecutionMetadata &execution);
  [[nodiscard]] bool hasSnapshot() const noexcept { return hasSnapshot_; }
  [[nodiscard]] const RenderGraphTelemetrySnapshot *
  latestSnapshot() const noexcept {
    return hasSnapshot_ ? &snapshot_ : nullptr;
  }
  [[nodiscard]] std::filesystem::path suggestDumpPath() const;
  [[nodiscard]] Result<bool, std::string>
  writeLatestTextDump(std::string_view outputPath) const;

private:
  RenderGraphTelemetrySnapshot snapshot_;
  std::filesystem::path configuredDumpDirectory_;
  bool hasSnapshot_ = false;
};

[[nodiscard]] NURI_API Result<bool, std::string>
writeRenderGraphTelemetryTextDump(const RenderGraphTelemetrySnapshot &snapshot,
                                  std::string_view outputPath);

} // namespace nuri
