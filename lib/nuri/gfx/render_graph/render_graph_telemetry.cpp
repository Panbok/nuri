#include "nuri/pch.h"

#include "nuri/gfx/render_graph/render_graph_telemetry.h"

#include "nuri/utils/env_utils.h"

#include <fstream>

namespace nuri {

namespace {

[[nodiscard]] std::pmr::memory_resource *
ensureMemory(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}

[[nodiscard]] std::filesystem::path resolveRenderGraphDumpDirectory() {
  const std::optional<std::string> dumpEnv =
      readEnvVar("NURI_RENDER_GRAPH_DUMP");
  if (!dumpEnv.has_value() || dumpEnv->empty() || dumpEnv->front() == '0') {
    return {};
  }

  const std::string &value = *dumpEnv;
  if (value == "1" || value == "true" || value == "TRUE") {
    return std::filesystem::path("logs/render_graph");
  }
  return std::filesystem::path(value);
}

[[nodiscard]] std::string_view
resolvePassName(const RenderGraphTelemetrySnapshot &snapshot,
                uint32_t passIndex) {
  if (passIndex >= snapshot.passNames.size()) {
    return "unnamed_pass";
  }
  const std::pmr::string &name = snapshot.passNames[passIndex];
  return name.empty() ? std::string_view("unnamed_pass")
                      : std::string_view(name.data(), name.size());
}

template <typename T>
void copyVector(std::pmr::vector<T> &dst, const std::pmr::vector<T> &src) {
  dst.clear();
  dst.reserve(src.size());
  for (const T &value : src) {
    dst.push_back(value);
  }
}

void copyPassNames(std::pmr::vector<std::pmr::string> &dst,
                   const std::pmr::vector<std::pmr::string> &src) {
  dst.clear();
  dst.reserve(src.size());
  for (const std::pmr::string &name : src) {
    dst.emplace_back();
    dst.back().assign(name.data(), name.size());
  }
}

} // namespace

RenderGraphTelemetrySnapshot::RenderGraphTelemetrySnapshot(
    std::pmr::memory_resource *memory)
    : passNames(ensureMemory(memory)),
      orderedPassIndices(ensureMemory(memory)), edges(ensureMemory(memory)),
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
      dependencyBufferRangesByPass(ensureMemory(memory)),
      unresolvedDependencyBufferBindings(ensureMemory(memory)),
      preDispatchRangesByPass(ensureMemory(memory)),
      preDispatchDependencyRanges(ensureMemory(memory)),
      unresolvedPreDispatchDependencyBufferBindings(ensureMemory(memory)),
      drawRangesByPass(ensureMemory(memory)),
      unresolvedDrawBufferBindings(ensureMemory(memory)) {}

void RenderGraphTelemetrySnapshot::reset() {
  summary = Summary{};
  passNames.clear();
  orderedPassIndices.clear();
  edges.clear();
  transientTextureLifetimes.clear();
  transientBufferLifetimes.clear();
  transientTextureAllocations.clear();
  transientBufferAllocations.clear();
  transientTextureAllocationByResource.clear();
  transientBufferAllocationByResource.clear();
  transientTexturePhysicalAllocations.clear();
  transientBufferPhysicalAllocations.clear();
  unresolvedTextureBindings.clear();
  resolvedDependencyBuffers.clear();
  dependencyBufferRangesByPass.clear();
  unresolvedDependencyBufferBindings.clear();
  preDispatchRangesByPass.clear();
  preDispatchDependencyRanges.clear();
  unresolvedPreDispatchDependencyBufferBindings.clear();
  drawRangesByPass.clear();
  unresolvedDrawBufferBindings.clear();
}

RenderGraphTelemetryService::RenderGraphTelemetryService(
    std::pmr::memory_resource *memory)
    : memory_(ensureMemory(memory)),
      snapshot_(ensureMemory(memory)),
      defaultDumpDirectory_(resolveRenderGraphDumpDirectory()) {}

void RenderGraphTelemetryService::capture(
    const RenderGraphCompileResult &compiled) {
  snapshot_.reset();

  RenderGraphTelemetrySnapshot::Summary &summary = snapshot_.summary;
  summary.frameIndex = compiled.frameIndex;
  summary.declaredPassCount = compiled.declaredPassCount;
  summary.culledPassCount = compiled.culledPassCount;
  summary.rootPassCount = compiled.rootPassCount;
  summary.passCount = static_cast<uint32_t>(compiled.passDebugNames.size());
  summary.edgeCount = static_cast<uint32_t>(compiled.edges.size());
  summary.importedTextures = compiled.resourceStats.importedTextures;
  summary.transientTextures = compiled.resourceStats.transientTextures;
  summary.importedBuffers = compiled.resourceStats.importedBuffers;
  summary.transientBuffers = compiled.resourceStats.transientBuffers;
  summary.transientTextureLifetimeCount =
      static_cast<uint32_t>(compiled.transientTextureLifetimes.size());
  summary.transientBufferLifetimeCount =
      static_cast<uint32_t>(compiled.transientBufferLifetimes.size());
  summary.transientTexturePhysicalCount =
      compiled.transientTexturePhysicalCount;
  summary.transientBufferPhysicalCount = compiled.transientBufferPhysicalCount;
  summary.transientTextureAllocationMapSize = static_cast<uint32_t>(
      compiled.transientTextureAllocationByResource.size());
  summary.transientBufferAllocationMapSize = static_cast<uint32_t>(
      compiled.transientBufferAllocationByResource.size());
  summary.transientTexturePhysicalAllocationCount = static_cast<uint32_t>(
      compiled.transientTexturePhysicalAllocations.size());
  summary.transientBufferPhysicalAllocationCount = static_cast<uint32_t>(
      compiled.transientBufferPhysicalAllocations.size());
  summary.unresolvedTextureBindingCount =
      static_cast<uint32_t>(compiled.unresolvedTextureBindings.size());
  summary.resolvedDependencyBufferSlotCount =
      static_cast<uint32_t>(compiled.resolvedDependencyBuffers.size());
  summary.unresolvedDependencyBufferBindingCount = static_cast<uint32_t>(
      compiled.unresolvedDependencyBufferBindings.size());
  summary.ownedPreDispatchCount =
      static_cast<uint32_t>(compiled.ownedPreDispatches.size());
  summary.ownedDrawItemCount =
      static_cast<uint32_t>(compiled.ownedDrawItems.size());
  summary.resolvedPreDispatchDependencyBufferSlotCount = static_cast<uint32_t>(
      compiled.resolvedPreDispatchDependencyBuffers.size());
  summary.unresolvedPreDispatchDependencyBufferBindingCount =
      static_cast<uint32_t>(
          compiled.unresolvedPreDispatchDependencyBufferBindings.size());
  summary.unresolvedDrawBufferBindingCount =
      static_cast<uint32_t>(compiled.unresolvedDrawBufferBindings.size());

  copyPassNames(snapshot_.passNames, compiled.passDebugNames);
  copyVector(snapshot_.orderedPassIndices, compiled.orderedPassIndices);
  copyVector(snapshot_.edges, compiled.edges);
  copyVector(snapshot_.transientTextureLifetimes,
             compiled.transientTextureLifetimes);
  copyVector(snapshot_.transientBufferLifetimes,
             compiled.transientBufferLifetimes);
  copyVector(snapshot_.transientTextureAllocations,
             compiled.transientTextureAllocations);
  copyVector(snapshot_.transientBufferAllocations,
             compiled.transientBufferAllocations);
  copyVector(snapshot_.transientTextureAllocationByResource,
             compiled.transientTextureAllocationByResource);
  copyVector(snapshot_.transientBufferAllocationByResource,
             compiled.transientBufferAllocationByResource);
  copyVector(snapshot_.transientTexturePhysicalAllocations,
             compiled.transientTexturePhysicalAllocations);
  copyVector(snapshot_.transientBufferPhysicalAllocations,
             compiled.transientBufferPhysicalAllocations);
  copyVector(snapshot_.unresolvedTextureBindings,
             compiled.unresolvedTextureBindings);
  copyVector(snapshot_.resolvedDependencyBuffers,
             compiled.resolvedDependencyBuffers);
  copyVector(snapshot_.dependencyBufferRangesByPass,
             compiled.dependencyBufferRangesByPass);
  copyVector(snapshot_.unresolvedDependencyBufferBindings,
             compiled.unresolvedDependencyBufferBindings);
  copyVector(snapshot_.preDispatchRangesByPass, compiled.preDispatchRangesByPass);
  copyVector(snapshot_.preDispatchDependencyRanges,
             compiled.preDispatchDependencyRanges);
  copyVector(snapshot_.unresolvedPreDispatchDependencyBufferBindings,
             compiled.unresolvedPreDispatchDependencyBufferBindings);
  copyVector(snapshot_.drawRangesByPass, compiled.drawRangesByPass);
  copyVector(snapshot_.unresolvedDrawBufferBindings,
             compiled.unresolvedDrawBufferBindings);

  hasSnapshot_ = true;
}

std::filesystem::path RenderGraphTelemetryService::suggestDumpPath() const {
  const std::filesystem::path dumpDirectory =
      defaultDumpDirectory_.empty() ? std::filesystem::path("logs/render_graph")
                                    : defaultDumpDirectory_;
  std::ostringstream fileName;
  fileName << "render_graph_frame_"
           << (hasSnapshot_ ? snapshot_.summary.frameIndex : 0ull) << ".txt";
  return dumpDirectory / fileName.str();
}

Result<bool, std::string>
RenderGraphTelemetryService::writeLatestTextDump(
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

  std::ofstream file(path, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    return Result<bool, std::string>::makeError(
        "writeRenderGraphTelemetryTextDump: failed to open '" + path.string() +
        "'");
  }

  const auto &summary = snapshot.summary;
  file << "RenderGraph Frame Dump\n";
  file << "frame_index: " << summary.frameIndex << "\n";
  file << "declared_pass_count: " << summary.declaredPassCount << "\n";
  file << "culled_pass_count: " << summary.culledPassCount << "\n";
  file << "root_pass_count: " << summary.rootPassCount << "\n";
  file << "pass_count: " << summary.passCount << "\n";
  file << "edge_count: " << summary.edgeCount << "\n";
  file << "imported_textures: " << summary.importedTextures << "\n";
  file << "transient_textures: " << summary.transientTextures << "\n";
  file << "imported_buffers: " << summary.importedBuffers << "\n";
  file << "transient_buffers: " << summary.transientBuffers << "\n";
  file << "transient_texture_lifetimes: "
       << summary.transientTextureLifetimeCount << "\n";
  file << "transient_buffer_lifetimes: "
       << summary.transientBufferLifetimeCount << "\n";
  file << "transient_texture_physical_count: "
       << summary.transientTexturePhysicalCount << "\n";
  file << "transient_buffer_physical_count: "
       << summary.transientBufferPhysicalCount << "\n";
  file << "transient_texture_allocation_map_size: "
       << summary.transientTextureAllocationMapSize << "\n";
  file << "transient_buffer_allocation_map_size: "
       << summary.transientBufferAllocationMapSize << "\n";
  file << "transient_texture_physical_allocations: "
       << summary.transientTexturePhysicalAllocationCount << "\n";
  file << "transient_buffer_physical_allocations: "
       << summary.transientBufferPhysicalAllocationCount << "\n";
  file << "unresolved_texture_bindings: "
       << summary.unresolvedTextureBindingCount << "\n";
  file << "resolved_dependency_buffer_slots: "
       << summary.resolvedDependencyBufferSlotCount << "\n";
  file << "unresolved_dependency_buffer_bindings: "
       << summary.unresolvedDependencyBufferBindingCount << "\n";
  file << "owned_pre_dispatches: " << summary.ownedPreDispatchCount << "\n";
  file << "owned_draw_items: " << summary.ownedDrawItemCount << "\n";
  file << "resolved_pre_dispatch_dependency_buffer_slots: "
       << summary.resolvedPreDispatchDependencyBufferSlotCount << "\n";
  file << "unresolved_pre_dispatch_dependency_buffer_bindings: "
       << summary.unresolvedPreDispatchDependencyBufferBindingCount << "\n";
  file << "unresolved_draw_buffer_bindings: "
       << summary.unresolvedDrawBufferBindingCount << "\n";
  file << "\n";

  file << "Passes:\n";
  for (uint32_t passIndex = 0; passIndex < snapshot.passNames.size();
       ++passIndex) {
    file << "  [" << passIndex << "] " << resolvePassName(snapshot, passIndex)
         << "\n";
  }
  file << "\n";

  file << "Dependencies:\n";
  for (const RenderGraphCompileResult::Edge &edge : snapshot.edges) {
    file << "  " << edge.before << " -> " << edge.after << "  ("
         << resolvePassName(snapshot, edge.before) << " -> "
         << resolvePassName(snapshot, edge.after) << ")\n";
  }
  if (snapshot.edges.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Execution Order:\n";
  for (uint32_t rank = 0; rank < snapshot.orderedPassIndices.size(); ++rank) {
    const uint32_t passIndex = snapshot.orderedPassIndices[rank];
    file << "  #" << rank << ": [" << passIndex << "] "
         << resolvePassName(snapshot, passIndex) << "\n";
  }
  if (snapshot.orderedPassIndices.empty()) {
    file << "  <empty>\n";
  }
  file << "\n";

  file << "Transient Texture Lifetimes:\n";
  for (const auto &lifetime : snapshot.transientTextureLifetimes) {
    file << "  tex[" << lifetime.resourceIndex << "] first_exec="
         << lifetime.firstExecutionIndex << " last_exec="
         << lifetime.lastExecutionIndex << "\n";
  }
  if (snapshot.transientTextureLifetimes.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Transient Buffer Lifetimes:\n";
  for (const auto &lifetime : snapshot.transientBufferLifetimes) {
    file << "  buf[" << lifetime.resourceIndex << "] first_exec="
         << lifetime.firstExecutionIndex << " last_exec="
         << lifetime.lastExecutionIndex << "\n";
  }
  if (snapshot.transientBufferLifetimes.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Transient Texture Allocations:\n";
  for (const auto &allocation : snapshot.transientTextureAllocations) {
    file << "  tex[" << allocation.resourceIndex << "] -> phys["
         << allocation.allocationIndex << "]\n";
  }
  if (snapshot.transientTextureAllocations.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Transient Buffer Allocations:\n";
  for (const auto &allocation : snapshot.transientBufferAllocations) {
    file << "  buf[" << allocation.resourceIndex << "] -> phys["
         << allocation.allocationIndex << "]\n";
  }
  if (snapshot.transientBufferAllocations.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Transient Texture Allocation Map:\n";
  bool wroteTextureAllocationMap = false;
  for (uint32_t resourceIndex = 0;
       resourceIndex < snapshot.transientTextureAllocationByResource.size();
       ++resourceIndex) {
    const uint32_t allocationIndex =
        snapshot.transientTextureAllocationByResource[resourceIndex];
    if (allocationIndex == UINT32_MAX) {
      continue;
    }
    file << "  tex[" << resourceIndex << "] -> phys[" << allocationIndex
         << "]\n";
    wroteTextureAllocationMap = true;
  }
  if (!wroteTextureAllocationMap) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Transient Buffer Allocation Map:\n";
  bool wroteBufferAllocationMap = false;
  for (uint32_t resourceIndex = 0;
       resourceIndex < snapshot.transientBufferAllocationByResource.size();
       ++resourceIndex) {
    const uint32_t allocationIndex =
        snapshot.transientBufferAllocationByResource[resourceIndex];
    if (allocationIndex == UINT32_MAX) {
      continue;
    }
    file << "  buf[" << resourceIndex << "] -> phys[" << allocationIndex
         << "]\n";
    wroteBufferAllocationMap = true;
  }
  if (!wroteBufferAllocationMap) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Transient Texture Physical Allocations:\n";
  for (const auto &physical : snapshot.transientTexturePhysicalAllocations) {
    const TextureDesc &desc = physical.desc;
    file << "  phys[" << physical.allocationIndex
         << "] repr_tex=" << physical.representativeResourceIndex
         << " fmt=" << static_cast<uint32_t>(desc.format)
         << " dims=" << desc.dimensions.width << "x" << desc.dimensions.height
         << "x" << desc.dimensions.depth
         << " layers=" << desc.numLayers
         << " samples=" << desc.numSamples
         << " mips=" << desc.numMipLevels << "\n";
  }
  if (snapshot.transientTexturePhysicalAllocations.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Transient Buffer Physical Allocations:\n";
  for (const auto &physical : snapshot.transientBufferPhysicalAllocations) {
    const BufferDesc &desc = physical.desc;
    file << "  phys[" << physical.allocationIndex
         << "] repr_buf=" << physical.representativeResourceIndex
         << " usage=" << static_cast<uint32_t>(desc.usage)
         << " storage=" << static_cast<uint32_t>(desc.storage)
         << " size=" << desc.size << "\n";
  }
  if (snapshot.transientBufferPhysicalAllocations.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Unresolved Pass Texture Bindings:\n";
  for (const auto &binding : snapshot.unresolvedTextureBindings) {
    const std::string_view target =
        binding.target ==
                RenderGraphCompileResult::PassTextureBindingTarget::Color
            ? "color"
            : "depth";
    file << "  pass_exec[" << binding.orderedPassIndex << "]." << target
         << " <- tex[" << binding.textureResourceIndex << "]\n";
  }
  if (snapshot.unresolvedTextureBindings.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Pass Dependency Buffer Ranges:\n";
  for (uint32_t passIndex = 0;
       passIndex < snapshot.dependencyBufferRangesByPass.size(); ++passIndex) {
    const auto &range = snapshot.dependencyBufferRangesByPass[passIndex];
    file << "  pass_exec[" << passIndex << "] offset=" << range.offset
         << " count=" << range.count << "\n";
  }
  if (snapshot.dependencyBufferRangesByPass.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Resolved Dependency Buffers:\n";
  bool wroteResolvedDependencyBuffer = false;
  for (uint32_t i = 0; i < snapshot.resolvedDependencyBuffers.size(); ++i) {
    const BufferHandle handle = snapshot.resolvedDependencyBuffers[i];
    if (!nuri::isValid(handle)) {
      continue;
    }
    file << "  slot[" << i << "] handle=(" << handle.index << ","
         << handle.generation << ")\n";
    wroteResolvedDependencyBuffer = true;
  }
  if (!wroteResolvedDependencyBuffer) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Unresolved Dependency Buffer Bindings:\n";
  for (const auto &binding : snapshot.unresolvedDependencyBufferBindings) {
    file << "  pass_exec[" << binding.orderedPassIndex << "].dep["
         << binding.dependencyBufferIndex << "] <- buf["
         << binding.bufferResourceIndex << "]\n";
  }
  if (snapshot.unresolvedDependencyBufferBindings.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Pass Pre-Dispatch Ranges:\n";
  for (uint32_t passIndex = 0;
       passIndex < snapshot.preDispatchRangesByPass.size(); ++passIndex) {
    const auto &range = snapshot.preDispatchRangesByPass[passIndex];
    file << "  pass_exec[" << passIndex << "] offset=" << range.offset
         << " count=" << range.count << "\n";
  }
  if (snapshot.preDispatchRangesByPass.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Pre-Dispatch Dependency Ranges:\n";
  for (uint32_t dispatchIndex = 0;
       dispatchIndex < snapshot.preDispatchDependencyRanges.size();
       ++dispatchIndex) {
    const auto &range = snapshot.preDispatchDependencyRanges[dispatchIndex];
    file << "  pre_dispatch[" << dispatchIndex << "] offset=" << range.offset
         << " count=" << range.count << "\n";
  }
  if (snapshot.preDispatchDependencyRanges.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Unresolved Pre-Dispatch Dependency Buffer Bindings:\n";
  for (const auto &binding :
       snapshot.unresolvedPreDispatchDependencyBufferBindings) {
    file << "  pass_exec[" << binding.orderedPassIndex << "].pre_dispatch["
         << binding.preDispatchIndex << "].dep["
         << binding.dependencyBufferIndex << "] <- buf["
         << binding.bufferResourceIndex << "]\n";
  }
  if (snapshot.unresolvedPreDispatchDependencyBufferBindings.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Pass Draw Ranges:\n";
  for (uint32_t passIndex = 0; passIndex < snapshot.drawRangesByPass.size();
       ++passIndex) {
    const auto &range = snapshot.drawRangesByPass[passIndex];
    file << "  pass_exec[" << passIndex << "] offset=" << range.offset
         << " count=" << range.count << "\n";
  }
  if (snapshot.drawRangesByPass.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file << "Unresolved Draw Buffer Bindings:\n";
  for (const auto &binding : snapshot.unresolvedDrawBufferBindings) {
    const std::string_view target = [&]() {
      switch (binding.target) {
      case RenderGraphCompileResult::DrawBufferBindingTarget::Vertex:
        return std::string_view("vertex");
      case RenderGraphCompileResult::DrawBufferBindingTarget::Index:
        return std::string_view("index");
      case RenderGraphCompileResult::DrawBufferBindingTarget::Indirect:
        return std::string_view("indirect");
      case RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount:
        return std::string_view("indirect_count");
      default:
        return std::string_view("unknown");
      }
    }();
    file << "  pass_exec[" << binding.orderedPassIndex << "].draw["
         << binding.drawIndex << "]." << target << " <- buf["
         << binding.bufferResourceIndex << "]\n";
  }
  if (snapshot.unresolvedDrawBufferBindings.empty()) {
    file << "  <none>\n";
  }
  file << "\n";

  file.flush();
  if (file.fail()) {
    return Result<bool, std::string>::makeError(
        "writeRenderGraphTelemetryTextDump: failed while writing '" +
        path.string() + "'");
  }

  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
