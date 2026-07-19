#include "nuri/gfx/render_graph/render_graph_telemetry.h"
#include "nuri/pch.h"
#include "nuri/utils/env_utils.h"
namespace nuri {
namespace {
constexpr std::string_view kUnnamedPassName = "unnamed_pass";
[[nodiscard]] std::pmr::memory_resource *
ensureMemory(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}
[[nodiscard]] std::filesystem::path defaultRenderGraphDumpDirectory() {
  return "logs/render_graph";
}
[[nodiscard]] bool equalsIgnoreAsciiCase(std::string_view lhs,
                                         std::string_view rhs) {
  return std::ranges::equal(lhs, rhs, [](char a, char b) {
    return std::tolower(static_cast<unsigned char>(a)) ==
           std::tolower(static_cast<unsigned char>(b));
  });
}
[[nodiscard]] std::filesystem::path resolveRenderGraphDumpDirectory() {
  const std::optional<std::string> value = readEnvVar("NURI_RENDER_GRAPH_DUMP");
  if (!value || value->empty() || *value == "0" ||
      equalsIgnoreAsciiCase(*value, "false")) {
    return {};
  }
  return *value == "1" || equalsIgnoreAsciiCase(*value, "true")
             ? defaultRenderGraphDumpDirectory()
             : std::filesystem::path(*value);
}
[[nodiscard]] std::string_view
resolvePassName(const RenderGraphTelemetrySnapshot &snapshot,
                uint32_t passIndex) {
  const auto &names = snapshot.compile.passDebugNames;
  if (passIndex >= names.size() || names[passIndex].empty()) {
    return kUnnamedPassName;
  }
  return names[passIndex];
}
template <typename T> [[nodiscard]] uint32_t count(const T &values) {
  return static_cast<uint32_t>(values.size());
}
template <typename T> void fingerprint(uint64_t &hash, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  for (const uint8_t byte :
       std::span{reinterpret_cast<const uint8_t *>(&value), sizeof(value)}) {
    hash = (hash ^ byte) * 1099511628211ull;
  }
}
template <typename... T>
void fingerprintValues(uint64_t &hash, const T &...values) {
  (fingerprint(hash, values), ...);
}
void fingerprint(uint64_t &hash, std::string_view value) {
  fingerprint(hash, static_cast<uint32_t>(value.size()));
  for (const uint8_t byte : std::span{
           reinterpret_cast<const uint8_t *>(value.data()), value.size()}) {
    hash = (hash ^ byte) * 1099511628211ull;
  }
}
[[nodiscard]] uint64_t
compileFingerprint(const RenderGraphCompileResult &compiled) {
  uint64_t hash = 14695981039346656037ull;
  fingerprintValues(hash, compiled.frameIndex, compiled.declaredPassCount,
                    compiled.culledPassCount, compiled.rootPassCount,
                    compiled.usedParallelCompile,
                    compiled.usedParallelPayloadResolution,
                    compiled.usedParallelHazardAnalysis,
                    compiled.usedParallelLifetimeAnalysis);
  for (const auto &name : compiled.passDebugNames) {
    fingerprint(hash, std::string_view(name));
  }
  for (const uint32_t pass : compiled.orderedPassIndices) {
    fingerprint(hash, pass);
  }
  for (const auto &edge : compiled.edges) {
    fingerprintValues(hash, edge.before, edge.after);
  }
  for (const auto &pass : compiled.recordedGraphicsPasses) {
    fingerprintValues(hash, pass.orderedPassIndex, pass.declaredPassIndex);
  }
  const auto hashLifetimes = [&](const auto &lifetimes) {
    for (const auto &lifetime : lifetimes) {
      fingerprintValues(hash, lifetime.resourceIndex,
                        lifetime.firstExecutionIndex,
                        lifetime.lastExecutionIndex);
    }
  };
  hashLifetimes(compiled.transientTextureLifetimes);
  hashLifetimes(compiled.transientBufferLifetimes);
  const auto hashAllocations = [&](const auto &allocations) {
    for (const auto &allocation : allocations) {
      fingerprintValues(hash, allocation.resourceIndex,
                        allocation.allocationIndex);
    }
  };
  hashAllocations(compiled.transientTextureAllocations);
  hashAllocations(compiled.transientBufferAllocations);
  return hash;
}
[[nodiscard]] uint64_t
barrierFingerprint(const RenderGraphCompileResult &compiled) {
  uint64_t hash = 14695981039346656037ull;
  for (const auto &plan : compiled.passBarrierPlans) {
    fingerprintValues(hash, plan.orderedPassIndex, plan.barrierOffset,
                      plan.barrierCount);
  }
  fingerprintValues(hash, compiled.finalBarrierPlan.barrierOffset,
                    compiled.finalBarrierPlan.barrierCount);
  for (const auto &barrier : compiled.passBarrierRecords) {
    fingerprintValues(hash, barrier.resourceKind, barrier.resourceIndex,
                      barrier.beforeAccess, barrier.afterAccess,
                      barrier.beforeState, barrier.afterState);
  }
  return hash;
}
[[nodiscard]] uint64_t
executionFingerprint(const RenderGraphExecutionMetadata &execution) {
  uint64_t hash = 14695981039346656037ull;
  fingerprintValues(hash, execution.usedParallelCompile,
                    execution.usedParallelRecording);
  for (const auto &buffer : execution.recordedCommandBuffers) {
    fingerprintValues(hash, buffer.firstOrderedPassIndex, buffer.passCount);
  }
  for (const auto &batch : execution.submitBatches) {
    fingerprintValues(hash, batch.commandBufferOffset, batch.commandBufferCount,
                      batch.presentsFrameOutput);
  }
  for (const auto &range : execution.passRanges) {
    fingerprintValues(hash, range.workerIndex, range.firstOrderedPassIndex,
                      range.passCount);
  }
  return hash;
}
[[nodiscard]] RenderGraphTelemetrySnapshot::Summary
buildSummary(const RenderGraphCompileResult &compiled,
             const RenderGraphExecutionMetadata *execution) {
  RenderGraphTelemetrySnapshot::Summary result{};
  result.frameIndex = compiled.frameIndex;
  result.declaredPassCount = compiled.declaredPassCount;
  result.culledPassCount = compiled.culledPassCount;
  result.rootPassCount = compiled.rootPassCount;
  result.passCount = count(compiled.passDebugNames);
  result.edgeCount = count(compiled.edges);
  result.recordedGraphicsPassCount = count(compiled.recordedGraphicsPasses);
  result.passBarrierPlanCount = count(compiled.passBarrierPlans);
  result.finalBarrierRecordCount = compiled.finalBarrierPlan.barrierCount;
  result.passBarrierRecordCount = count(compiled.passBarrierRecords);
  result.importedTextures = compiled.resourceStats.importedTextures;
  result.transientTextures = compiled.resourceStats.transientTextures;
  result.importedBuffers = compiled.resourceStats.importedBuffers;
  result.transientBuffers = compiled.resourceStats.transientBuffers;
  result.transientTextureLifetimeCount =
      count(compiled.transientTextureLifetimes);
  result.transientBufferLifetimeCount =
      count(compiled.transientBufferLifetimes);
  result.transientTexturePhysicalCount = compiled.transientTexturePhysicalCount;
  result.transientBufferPhysicalCount = compiled.transientBufferPhysicalCount;
  result.transientTextureAllocationMapSize =
      count(compiled.transientTextureAllocationByResource);
  result.transientBufferAllocationMapSize =
      count(compiled.transientBufferAllocationByResource);
  result.transientTexturePhysicalAllocationCount =
      count(compiled.transientTexturePhysicalAllocations);
  result.transientBufferPhysicalAllocationCount =
      count(compiled.transientBufferPhysicalAllocations);
  result.unresolvedTextureBindingCount =
      count(compiled.unresolvedTextureBindings);
  result.resolvedDependencyBufferSlotCount =
      count(compiled.resolvedDependencyBuffers);
  result.unresolvedDependencyBufferBindingCount =
      count(compiled.unresolvedDependencyBufferBindings);
  result.ownedPreDispatchCount = count(compiled.ownedPreDispatches);
  result.ownedDrawItemCount = count(compiled.ownedDrawItems);
  result.ownedMeshDispatchItemCount = count(compiled.ownedMeshDispatchItems);
  result.resolvedPreDispatchDependencyBufferSlotCount =
      count(compiled.resolvedPreDispatchDependencyBuffers);
  result.unresolvedPreDispatchDependencyBufferBindingCount =
      count(compiled.unresolvedPreDispatchDependencyBufferBindings);
  result.unresolvedDrawBufferBindingCount =
      count(compiled.unresolvedDrawBufferBindings);
  result.unresolvedMeshDispatchBufferBindingCount =
      count(compiled.unresolvedMeshDispatchBufferBindings);
  result.usedParallelCompile = compiled.usedParallelCompile;
  result.usedParallelPayloadResolution = compiled.usedParallelPayloadResolution;
  result.usedParallelHazardAnalysis = compiled.usedParallelHazardAnalysis;
  result.usedParallelLifetimeAnalysis = compiled.usedParallelLifetimeAnalysis;
  result.compileFingerprint = compileFingerprint(compiled);
  result.barrierFingerprint = barrierFingerprint(compiled);
  if (execution != nullptr) {
    result.recordedCommandBufferCount =
        count(execution->recordedCommandBuffers);
    result.submitBatchCount = count(execution->submitBatches);
    result.passRangeCount = count(execution->passRanges);
    result.passTimingCount = count(execution->passTimings);
    result.usedParallelRecording = execution->usedParallelRecording;
    result.executionFingerprint = executionFingerprint(*execution);
  }
  return result;
}
template <typename Range, typename Writer>
void writeSection(std::ostream &stream, std::string_view title,
                  const Range &items, Writer writer) {
  stream << title << ":\n";
  if (items.empty()) {
    stream << "  <none>\n";
  } else {
    for (const auto &item : items) {
      writer(item);
    }
  }
  stream << "\n";
}
[[nodiscard]] std::string
makeOpenErrorMessage(const std::filesystem::path &path) {
  std::string result = "writeRenderGraphTelemetryTextDump: failed to open '" +
                       path.string() + "'";
  if (errno != 0) {
    result += ": " + std::error_code(errno, std::generic_category()).message();
  }
  return result;
}
} // namespace
RenderGraphTelemetrySnapshot::RenderGraphTelemetrySnapshot(
    std::pmr::memory_resource *memory)
    : compile(ensureMemory(memory)), execution(ensureMemory(memory)) {}

void RenderGraphTelemetrySnapshot::captureFrom(
    const RenderGraphCompileResult &compiled) {
  reset();
  summary = buildSummary(compiled, nullptr);
  compile = compiled;
  compile.orderedPasses.clear();
}

void RenderGraphTelemetrySnapshot::captureFrom(
    const RenderGraphCompileResult &compiled,
    const RenderGraphExecutionMetadata &metadata) {
  captureFrom(compiled);
  summary = buildSummary(compiled, &metadata);
  execution = metadata;
}

void RenderGraphTelemetrySnapshot::reset() {
  auto *memory = compile.passDebugNames.get_allocator().resource();
  summary = {};
  compile = RenderGraphCompileResult(memory);
  execution = RenderGraphExecutionMetadata(memory);
}

RenderGraphTelemetryService::RenderGraphTelemetryService(
    std::pmr::memory_resource *memory)
    : snapshot_(ensureMemory(memory)),
      configuredDumpDirectory_(resolveRenderGraphDumpDirectory()),
      captureEveryFrame_(!configuredDumpDirectory_.empty()) {}

void RenderGraphTelemetryService::requestCapture(
    RenderGraphTelemetryLevel level) noexcept {
  pendingCapture_ =
      static_cast<uint8_t>(level) > static_cast<uint8_t>(pendingCapture_)
          ? level
          : pendingCapture_;
}

RenderGraphTelemetryLevel
RenderGraphTelemetryService::requestedCaptureLevel() const noexcept {
  return captureEveryFrame_ ? RenderGraphTelemetryLevel::PassTimings
                            : pendingCapture_;
}

void RenderGraphTelemetryService::capture(
    const RenderGraphCompileResult &compiled) {
  snapshot_.captureFrom(compiled);
  pendingCapture_ = RenderGraphTelemetryLevel::None;
  hasSnapshot_ = true;
}

void RenderGraphTelemetryService::capture(
    const RenderGraphCompileResult &compiled,
    const RenderGraphExecutionMetadata &execution) {
  snapshot_.captureFrom(compiled, execution);
  pendingCapture_ = RenderGraphTelemetryLevel::None;
  hasSnapshot_ = true;
}

std::filesystem::path RenderGraphTelemetryService::suggestDumpPath() const {
  const auto directory = configuredDumpDirectory_.empty()
                             ? defaultRenderGraphDumpDirectory()
                             : configuredDumpDirectory_;
  return directory /
         ("render_graph_frame_" +
          std::to_string(hasSnapshot_ ? snapshot_.summary.frameIndex : 0ull) +
          ".txt");
}

Result<bool, std::string> RenderGraphTelemetryService::writeLatestTextDump(
    std::string_view outputPath) const {
  if (!hasSnapshot_) {
    return Result<bool, std::string>::makeError(
        "RenderGraphTelemetryService::writeLatestTextDump: no snapshot "
        "captured");
  }
  return writeRenderGraphTelemetryTextDump(snapshot_, outputPath);
}

Result<bool, std::string>
writeRenderGraphTelemetryTextDump(const RenderGraphTelemetrySnapshot &snapshot,
                                  std::string_view outputPath) {
  if (outputPath.empty()) {
    return Result<bool, std::string>::makeError(
        "writeRenderGraphTelemetryTextDump: output path is empty");
  }
  const std::filesystem::path path{std::string(outputPath)};
  if (path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      return Result<bool, std::string>::makeError(
          "writeRenderGraphTelemetryTextDump: failed to create directory '" +
          path.parent_path().string() + "': " + ec.message());
    }
  }
  errno = 0;
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    return Result<bool, std::string>::makeError(makeOpenErrorMessage(path));
  }
  const auto &summary = snapshot.summary;
  const std::pair<std::string_view, uint64_t> values[] = {
      {"frame_index", summary.frameIndex},
      {"declared_pass_count", summary.declaredPassCount},
      {"culled_pass_count", summary.culledPassCount},
      {"root_pass_count", summary.rootPassCount},
      {"pass_count", summary.passCount},
      {"edge_count", summary.edgeCount},
      {"recorded_graphics_pass_count", summary.recordedGraphicsPassCount},
      {"pass_barrier_plan_count", summary.passBarrierPlanCount},
      {"final_barrier_record_count", summary.finalBarrierRecordCount},
      {"pass_barrier_record_count", summary.passBarrierRecordCount},
      {"recorded_command_buffer_count", summary.recordedCommandBufferCount},
      {"submit_batch_count", summary.submitBatchCount},
      {"pass_range_count", summary.passRangeCount},
      {"pass_timing_count", summary.passTimingCount},
      {"imported_textures", summary.importedTextures},
      {"transient_textures", summary.transientTextures},
      {"imported_buffers", summary.importedBuffers},
      {"transient_buffers", summary.transientBuffers},
      {"transient_texture_lifetimes", summary.transientTextureLifetimeCount},
      {"transient_buffer_lifetimes", summary.transientBufferLifetimeCount},
      {"transient_texture_physical_count",
       summary.transientTexturePhysicalCount},
      {"transient_buffer_physical_count", summary.transientBufferPhysicalCount},
      {"owned_pre_dispatches", summary.ownedPreDispatchCount},
      {"owned_draw_items", summary.ownedDrawItemCount},
      {"owned_mesh_dispatch_items", summary.ownedMeshDispatchItemCount},
      {"used_parallel_compile", summary.usedParallelCompile},
      {"used_parallel_payload_resolution",
       summary.usedParallelPayloadResolution},
      {"used_parallel_hazard_analysis", summary.usedParallelHazardAnalysis},
      {"used_parallel_lifetime_analysis", summary.usedParallelLifetimeAnalysis},
      {"used_parallel_recording", summary.usedParallelRecording},
      {"compile_fingerprint", summary.compileFingerprint},
      {"barrier_fingerprint", summary.barrierFingerprint},
      {"execution_fingerprint", summary.executionFingerprint},
  };
  file << "RenderGraph Frame Dump\n";
  for (const auto &[name, value] : values) {
    file << name << ": " << value << "\n";
  }
  file << "\n";
  const auto &compiled = snapshot.compile;
  const auto &execution = snapshot.execution;
  writeSection(file, "Passes", compiled.passDebugNames, [&](const auto &name) {
    const uint32_t index =
        static_cast<uint32_t>(&name - compiled.passDebugNames.data());
    file << "  [" << index << "] " << resolvePassName(snapshot, index) << "\n";
  });
  writeSection(file, "Dependencies", compiled.edges, [&](const auto &edge) {
    file << "  " << edge.before << " -> " << edge.after << "  ("
         << resolvePassName(snapshot, edge.before) << " -> "
         << resolvePassName(snapshot, edge.after) << ")\n";
  });
  writeSection(file, "Execution Order", compiled.orderedPassIndices,
               [&](uint32_t pass) {
                 file << "  [" << pass << "] "
                      << resolvePassName(snapshot, pass) << "\n";
               });
  file << "Final Barrier Plan:\n";
  if (compiled.finalBarrierPlan.barrierCount == 0u) {
    file << "  <none>\n\n";
  } else {
    file << "  offset=" << compiled.finalBarrierPlan.barrierOffset
         << " count=" << compiled.finalBarrierPlan.barrierCount << "\n\n";
  }
  writeSection(file, "Recorded Command Buffers",
               execution.recordedCommandBuffers, [&](const auto &buffer) {
                 file << "  first_pass=" << buffer.firstOrderedPassIndex
                      << " pass_count=" << buffer.passCount << "\n";
               });
  writeSection(file, "Submit Batches", execution.submitBatches,
               [&](const auto &batch) {
                 file << "  offset=" << batch.commandBufferOffset
                      << " count=" << batch.commandBufferCount
                      << " presents=" << batch.presentsFrameOutput << "\n";
               });
  writeSection(file, "Pass CPU Timings", execution.passTimings,
               [&](const auto &timing) {
                 file << "  pass_exec[" << timing.orderedPassIndex
                      << "] cpu_ms=" << timing.cpuTimeMs << "\n";
               });
  file.flush();
  if (file.fail()) {
    return Result<bool, std::string>::makeError(
        "writeRenderGraphTelemetryTextDump: failed while writing '" +
        path.string() + "'");
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
