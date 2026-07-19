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
    uint32_t declaredPassCount = 0, culledPassCount = 0, rootPassCount = 0;
    uint32_t passCount = 0, edgeCount = 0, recordedGraphicsPassCount = 0;
    uint32_t passBarrierPlanCount = 0, finalBarrierRecordCount = 0;
    uint32_t passBarrierRecordCount = 0, recordedCommandBufferCount = 0;
    uint32_t submitBatchCount = 0, passRangeCount = 0, passTimingCount = 0;
    uint32_t importedTextures = 0, transientTextures = 0;
    uint32_t importedBuffers = 0, transientBuffers = 0;
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
    uint32_t ownedPreDispatchCount = 0, ownedDrawItemCount = 0;
    uint32_t ownedMeshDispatchItemCount = 0;
    uint32_t resolvedPreDispatchDependencyBufferSlotCount = 0;
    uint32_t unresolvedPreDispatchDependencyBufferBindingCount = 0;
    uint32_t unresolvedDrawBufferBindingCount = 0;
    uint32_t unresolvedMeshDispatchBufferBindingCount = 0;
    uint64_t compileFingerprint = 0, barrierFingerprint = 0;
    uint64_t executionFingerprint = 0;
    bool usedParallelCompile = false, usedParallelPayloadResolution = false;
    bool usedParallelHazardAnalysis = false;
    bool usedParallelLifetimeAnalysis = false, usedParallelRecording = false;
  };
  Summary summary{};
  RenderGraphCompileResult compile;
  RenderGraphExecutionMetadata execution;
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
  void requestCapture(RenderGraphTelemetryLevel level =
                          RenderGraphTelemetryLevel::Metadata) noexcept;
  [[nodiscard]] RenderGraphTelemetryLevel
  requestedCaptureLevel() const noexcept;
  [[nodiscard]] bool captureRequested() const noexcept {
    return requestedCaptureLevel() != RenderGraphTelemetryLevel::None;
  }
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
  RenderGraphTelemetryLevel pendingCapture_ = RenderGraphTelemetryLevel::None;
  bool captureEveryFrame_ = false;
  bool hasSnapshot_ = false;
};

[[nodiscard]] NURI_API Result<bool, std::string>
writeRenderGraphTelemetryTextDump(const RenderGraphTelemetrySnapshot &snapshot,
                                  std::string_view outputPath);

} // namespace nuri
