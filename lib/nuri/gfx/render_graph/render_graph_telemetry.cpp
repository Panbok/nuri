#include "nuri/pch.h"

#include "nuri/gfx/render_graph/render_graph_telemetry.h"

#include "nuri/utils/env_utils.h"

namespace nuri {

namespace {

constexpr std::string_view kUnnamedPassName = "unnamed_pass";

[[nodiscard]] std::pmr::memory_resource *
ensureMemory(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}

[[nodiscard]] std::filesystem::path defaultRenderGraphDumpDirectory() {
  return std::filesystem::path("logs/render_graph");
}

[[nodiscard]] bool equalsIgnoreAsciiCase(std::string_view lhs,
                                         std::string_view rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (size_t i = 0; i < lhs.size(); ++i) {
    const auto normalize = [](char c) {
      return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    };
    if (normalize(lhs[i]) != normalize(rhs[i])) {
      return false;
    }
  }

  return true;
}

[[nodiscard]] std::filesystem::path resolveRenderGraphDumpDirectory() {
  const std::optional<std::string> dumpEnv =
      readEnvVar("NURI_RENDER_GRAPH_DUMP");
  if (!dumpEnv.has_value()) {
    return {};
  }

  const std::string &value = *dumpEnv;
  if (value.empty() || value == "0") {
    return {};
  }

  if (value == "1" || equalsIgnoreAsciiCase(value, "true")) {
    return defaultRenderGraphDumpDirectory();
  }

  if (equalsIgnoreAsciiCase(value, "false")) {
    return {};
  }

  return std::filesystem::path(value);
}

[[nodiscard]] std::string_view
resolvePassName(const RenderGraphTelemetrySnapshot &snapshot,
                uint32_t passIndex) {
  if (passIndex >= snapshot.passNames.size()) {
    return kUnnamedPassName;
  }
  const std::pmr::string &name = snapshot.passNames[passIndex];
  return name.empty() ? kUnnamedPassName
                      : std::string_view(name.data(), name.size());
}

[[nodiscard]] RenderGraphTelemetrySnapshot::Summary
buildSummary(const RenderGraphCompileResult &compiled) {
  RenderGraphTelemetrySnapshot::Summary summary{};
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
  summary.transientBufferPhysicalAllocationCount =
      static_cast<uint32_t>(compiled.transientBufferPhysicalAllocations.size());
  summary.unresolvedTextureBindingCount =
      static_cast<uint32_t>(compiled.unresolvedTextureBindings.size());
  summary.resolvedDependencyBufferSlotCount =
      static_cast<uint32_t>(compiled.resolvedDependencyBuffers.size());
  summary.unresolvedDependencyBufferBindingCount =
      static_cast<uint32_t>(compiled.unresolvedDependencyBufferBindings.size());
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
  return summary;
}

template <typename T>
void copyVector(std::pmr::vector<T> &destination,
                const std::pmr::vector<T> &source) {
  destination.assign(source.begin(), source.end());
}

template <typename Value>
void writeKeyValue(std::ostream &stream, std::string_view key, Value value) {
  stream << key << ": " << value << "\n";
}

template <typename T, typename Writer>
void writeSection(std::ostream &stream, std::string_view title,
                  std::span<const T> items, Writer &&writer) {
  stream << title << ":\n";
  bool wroteAny = false;
  for (const T &item : items) {
    wroteAny = writer(item) || wroteAny;
  }
  if (!wroteAny) {
    stream << "  <none>\n";
  }
  stream << "\n";
}

template <typename T, typename Writer>
void writeIndexedSection(std::ostream &stream, std::string_view title,
                         std::span<const T> items, Writer &&writer) {
  stream << title << ":\n";
  bool wroteAny = false;
  for (uint32_t index = 0; index < items.size(); ++index) {
    wroteAny = writer(index, items[index]) || wroteAny;
  }
  if (!wroteAny) {
    stream << "  <none>\n";
  }
  stream << "\n";
}

[[nodiscard]] std::string_view resolveTextureBindingTarget(
    RenderGraphCompileResult::PassTextureBindingTarget target) {
  switch (target) {
  case RenderGraphCompileResult::PassTextureBindingTarget::Color:
    return "color";
  case RenderGraphCompileResult::PassTextureBindingTarget::Depth:
    return "depth";
  }

  return "unknown";
}

[[nodiscard]] std::string_view resolveDrawBufferBindingTarget(
    RenderGraphCompileResult::DrawBufferBindingTarget target) {
  switch (target) {
  case RenderGraphCompileResult::DrawBufferBindingTarget::Vertex:
    return "vertex";
  case RenderGraphCompileResult::DrawBufferBindingTarget::Index:
    return "index";
  case RenderGraphCompileResult::DrawBufferBindingTarget::Indirect:
    return "indirect";
  case RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount:
    return "indirect_count";
  }

  return "unknown";
}

[[nodiscard]] std::string
makeOpenErrorMessage(const std::filesystem::path &path, int errorNumber) {
  std::string message = "writeRenderGraphTelemetryTextDump: failed to open '" +
                        path.string() + "'";
  if (errorNumber != 0) {
    message +=
        ": " + std::error_code(errorNumber, std::generic_category()).message();
  }
  return message;
}

} // namespace

RenderGraphTelemetrySnapshot::RenderGraphTelemetrySnapshot(
    std::pmr::memory_resource *memory)
    : passNames(ensureMemory(memory)), orderedPassIndices(ensureMemory(memory)),
      edges(ensureMemory(memory)),
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

void RenderGraphTelemetrySnapshot::captureFrom(
    const RenderGraphCompileResult &compiled) {
  reset();
  summary = buildSummary(compiled);

  copyVector(passNames, compiled.passDebugNames);
  copyVector(orderedPassIndices, compiled.orderedPassIndices);
  copyVector(edges, compiled.edges);
  copyVector(transientTextureLifetimes, compiled.transientTextureLifetimes);
  copyVector(transientBufferLifetimes, compiled.transientBufferLifetimes);
  copyVector(transientTextureAllocations, compiled.transientTextureAllocations);
  copyVector(transientBufferAllocations, compiled.transientBufferAllocations);
  copyVector(transientTextureAllocationByResource,
             compiled.transientTextureAllocationByResource);
  copyVector(transientBufferAllocationByResource,
             compiled.transientBufferAllocationByResource);
  copyVector(transientTexturePhysicalAllocations,
             compiled.transientTexturePhysicalAllocations);
  copyVector(transientBufferPhysicalAllocations,
             compiled.transientBufferPhysicalAllocations);
  copyVector(unresolvedTextureBindings, compiled.unresolvedTextureBindings);
  copyVector(resolvedDependencyBuffers, compiled.resolvedDependencyBuffers);
  copyVector(dependencyBufferRangesByPass,
             compiled.dependencyBufferRangesByPass);
  copyVector(unresolvedDependencyBufferBindings,
             compiled.unresolvedDependencyBufferBindings);
  copyVector(preDispatchRangesByPass, compiled.preDispatchRangesByPass);
  copyVector(preDispatchDependencyRanges, compiled.preDispatchDependencyRanges);
  copyVector(unresolvedPreDispatchDependencyBufferBindings,
             compiled.unresolvedPreDispatchDependencyBufferBindings);
  copyVector(drawRangesByPass, compiled.drawRangesByPass);
  copyVector(unresolvedDrawBufferBindings,
             compiled.unresolvedDrawBufferBindings);
}

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
    : snapshot_(ensureMemory(memory)),
      configuredDumpDirectory_(resolveRenderGraphDumpDirectory()) {}

void RenderGraphTelemetryService::capture(
    const RenderGraphCompileResult &compiled) {
  snapshot_.captureFrom(compiled);
  hasSnapshot_ = true;
}

std::filesystem::path RenderGraphTelemetryService::suggestDumpPath() const {
  const std::filesystem::path dumpDirectory =
      configuredDumpDirectory_.empty() ? defaultRenderGraphDumpDirectory()
                                       : configuredDumpDirectory_;
  std::ostringstream fileName;
  fileName << "render_graph_frame_"
           << (hasSnapshot_ ? snapshot_.summary.frameIndex : 0ull) << ".txt";
  return dumpDirectory / fileName.str();
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
    return Result<bool, std::string>::makeError(
        makeOpenErrorMessage(path, errno));
  }

  const auto &summary = snapshot.summary;
  file << "RenderGraph Frame Dump\n";
  writeKeyValue(file, "frame_index", summary.frameIndex);
  writeKeyValue(file, "declared_pass_count", summary.declaredPassCount);
  writeKeyValue(file, "culled_pass_count", summary.culledPassCount);
  writeKeyValue(file, "root_pass_count", summary.rootPassCount);
  writeKeyValue(file, "pass_count", summary.passCount);
  writeKeyValue(file, "edge_count", summary.edgeCount);
  writeKeyValue(file, "imported_textures", summary.importedTextures);
  writeKeyValue(file, "transient_textures", summary.transientTextures);
  writeKeyValue(file, "imported_buffers", summary.importedBuffers);
  writeKeyValue(file, "transient_buffers", summary.transientBuffers);
  writeKeyValue(file, "transient_texture_lifetimes",
                summary.transientTextureLifetimeCount);
  writeKeyValue(file, "transient_buffer_lifetimes",
                summary.transientBufferLifetimeCount);
  writeKeyValue(file, "transient_texture_physical_count",
                summary.transientTexturePhysicalCount);
  writeKeyValue(file, "transient_buffer_physical_count",
                summary.transientBufferPhysicalCount);
  writeKeyValue(file, "transient_texture_allocation_map_size",
                summary.transientTextureAllocationMapSize);
  writeKeyValue(file, "transient_buffer_allocation_map_size",
                summary.transientBufferAllocationMapSize);
  writeKeyValue(file, "transient_texture_physical_allocations",
                summary.transientTexturePhysicalAllocationCount);
  writeKeyValue(file, "transient_buffer_physical_allocations",
                summary.transientBufferPhysicalAllocationCount);
  writeKeyValue(file, "unresolved_texture_bindings",
                summary.unresolvedTextureBindingCount);
  writeKeyValue(file, "resolved_dependency_buffer_slots",
                summary.resolvedDependencyBufferSlotCount);
  writeKeyValue(file, "unresolved_dependency_buffer_bindings",
                summary.unresolvedDependencyBufferBindingCount);
  writeKeyValue(file, "owned_pre_dispatches", summary.ownedPreDispatchCount);
  writeKeyValue(file, "owned_draw_items", summary.ownedDrawItemCount);
  writeKeyValue(file, "resolved_pre_dispatch_dependency_buffer_slots",
                summary.resolvedPreDispatchDependencyBufferSlotCount);
  writeKeyValue(file, "unresolved_pre_dispatch_dependency_buffer_bindings",
                summary.unresolvedPreDispatchDependencyBufferBindingCount);
  writeKeyValue(file, "unresolved_draw_buffer_bindings",
                summary.unresolvedDrawBufferBindingCount);
  file << "\n";

  writeIndexedSection(file, "Passes", std::span{snapshot.passNames},
                      [&](uint32_t passIndex, const std::pmr::string &) {
                        file << "  [" << passIndex << "] "
                             << resolvePassName(snapshot, passIndex) << "\n";
                        return true;
                      });

  writeSection(file, "Dependencies", std::span{snapshot.edges},
               [&](const RenderGraphCompileResult::Edge &edge) {
                 file << "  " << edge.before << " -> " << edge.after << "  ("
                      << resolvePassName(snapshot, edge.before) << " -> "
                      << resolvePassName(snapshot, edge.after) << ")\n";
                 return true;
               });

  writeIndexedSection(file, "Execution Order",
                      std::span{snapshot.orderedPassIndices},
                      [&](uint32_t rank, uint32_t passIndex) {
                        file << "  #" << rank << ": [" << passIndex << "] "
                             << resolvePassName(snapshot, passIndex) << "\n";
                        return true;
                      });

  writeSection(
      file, "Transient Texture Lifetimes",
      std::span{snapshot.transientTextureLifetimes},
      [&](const RenderGraphCompileResult::TransientLifetime &lifetime) {
        file << "  tex[" << lifetime.resourceIndex
             << "] first_exec=" << lifetime.firstExecutionIndex
             << " last_exec=" << lifetime.lastExecutionIndex << "\n";
        return true;
      });

  writeSection(
      file, "Transient Buffer Lifetimes",
      std::span{snapshot.transientBufferLifetimes},
      [&](const RenderGraphCompileResult::TransientLifetime &lifetime) {
        file << "  buf[" << lifetime.resourceIndex
             << "] first_exec=" << lifetime.firstExecutionIndex
             << " last_exec=" << lifetime.lastExecutionIndex << "\n";
        return true;
      });

  writeSection(
      file, "Transient Texture Allocations",
      std::span{snapshot.transientTextureAllocations},
      [&](const RenderGraphCompileResult::TransientAllocation &allocation) {
        file << "  tex[" << allocation.resourceIndex << "] -> phys["
             << allocation.allocationIndex << "]\n";
        return true;
      });

  writeSection(
      file, "Transient Buffer Allocations",
      std::span{snapshot.transientBufferAllocations},
      [&](const RenderGraphCompileResult::TransientAllocation &allocation) {
        file << "  buf[" << allocation.resourceIndex << "] -> phys["
             << allocation.allocationIndex << "]\n";
        return true;
      });

  writeIndexedSection(file, "Transient Texture Allocation Map",
                      std::span{snapshot.transientTextureAllocationByResource},
                      [&](uint32_t resourceIndex, uint32_t allocationIndex) {
                        if (allocationIndex == UINT32_MAX) {
                          return false;
                        }
                        file << "  tex[" << resourceIndex << "] -> phys["
                             << allocationIndex << "]\n";
                        return true;
                      });

  writeIndexedSection(file, "Transient Buffer Allocation Map",
                      std::span{snapshot.transientBufferAllocationByResource},
                      [&](uint32_t resourceIndex, uint32_t allocationIndex) {
                        if (allocationIndex == UINT32_MAX) {
                          return false;
                        }
                        file << "  buf[" << resourceIndex << "] -> phys["
                             << allocationIndex << "]\n";
                        return true;
                      });

  writeSection(
      file, "Transient Texture Physical Allocations",
      std::span{snapshot.transientTexturePhysicalAllocations},
      [&](const RenderGraphCompileResult::TransientTexturePhysicalAllocation
              &physical) {
        const TextureDesc &desc = physical.desc;
        file << "  phys[" << physical.allocationIndex
             << "] repr_tex=" << physical.representativeResourceIndex
             << " fmt=" << static_cast<uint32_t>(desc.format)
             << " dims=" << desc.dimensions.width << "x"
             << desc.dimensions.height << "x" << desc.dimensions.depth
             << " layers=" << desc.numLayers << " samples=" << desc.numSamples
             << " mips=" << desc.numMipLevels << "\n";
        return true;
      });

  writeSection(
      file, "Transient Buffer Physical Allocations",
      std::span{snapshot.transientBufferPhysicalAllocations},
      [&](const RenderGraphCompileResult::TransientBufferPhysicalAllocation
              &physical) {
        const BufferDesc &desc = physical.desc;
        file << "  phys[" << physical.allocationIndex
             << "] repr_buf=" << physical.representativeResourceIndex
             << " usage=" << static_cast<uint32_t>(desc.usage)
             << " storage=" << static_cast<uint32_t>(desc.storage)
             << " size=" << desc.size << "\n";
        return true;
      });

  writeSection(
      file, "Unresolved Pass Texture Bindings",
      std::span{snapshot.unresolvedTextureBindings},
      [&](const RenderGraphCompileResult::PassTextureBinding &binding) {
        file << "  pass_exec[" << binding.orderedPassIndex << "]."
             << resolveTextureBindingTarget(binding.target) << " <- tex["
             << binding.textureResourceIndex << "]\n";
        return true;
      });

  writeIndexedSection(
      file, "Pass Dependency Buffer Ranges",
      std::span{snapshot.dependencyBufferRangesByPass},
      [&](uint32_t passIndex,
          const RenderGraphCompileResult::PassDependencyBufferRange &range) {
        file << "  pass_exec[" << passIndex << "] offset=" << range.offset
             << " count=" << range.count << "\n";
        return true;
      });

  writeIndexedSection(file, "Resolved Dependency Buffers",
                      std::span{snapshot.resolvedDependencyBuffers},
                      [&](uint32_t index, BufferHandle handle) {
                        if (!nuri::isValid(handle)) {
                          return false;
                        }
                        file << "  slot[" << index << "] handle=("
                             << handle.index << "," << handle.generation
                             << ")\n";
                        return true;
                      });

  writeSection(
      file, "Unresolved Dependency Buffer Bindings",
      std::span{snapshot.unresolvedDependencyBufferBindings},
      [&](const RenderGraphCompileResult::UnresolvedDependencyBufferBinding
              &binding) {
        file << "  pass_exec[" << binding.orderedPassIndex << "].dep["
             << binding.dependencyBufferIndex << "] <- buf["
             << binding.bufferResourceIndex << "]\n";
        return true;
      });

  writeIndexedSection(
      file, "Pass Pre-Dispatch Ranges",
      std::span{snapshot.preDispatchRangesByPass},
      [&](uint32_t passIndex,
          const RenderGraphCompileResult::PassDispatchRange &range) {
        file << "  pass_exec[" << passIndex << "] offset=" << range.offset
             << " count=" << range.count << "\n";
        return true;
      });

  writeIndexedSection(
      file, "Pre-Dispatch Dependency Ranges",
      std::span{snapshot.preDispatchDependencyRanges},
      [&](uint32_t dispatchIndex,
          const RenderGraphCompileResult::DispatchDependencyBufferRange
              &range) {
        file << "  pre_dispatch[" << dispatchIndex
             << "] offset=" << range.offset << " count=" << range.count << "\n";
        return true;
      });

  writeSection(
      file, "Unresolved Pre-Dispatch Dependency Buffer Bindings",
      std::span{snapshot.unresolvedPreDispatchDependencyBufferBindings},
      [&](const RenderGraphCompileResult::
              UnresolvedPreDispatchDependencyBufferBinding &binding) {
        file << "  pass_exec[" << binding.orderedPassIndex << "].pre_dispatch["
             << binding.preDispatchIndex << "].dep["
             << binding.dependencyBufferIndex << "] <- buf["
             << binding.bufferResourceIndex << "]\n";
        return true;
      });

  writeIndexedSection(
      file, "Pass Draw Ranges", std::span{snapshot.drawRangesByPass},
      [&](uint32_t passIndex,
          const RenderGraphCompileResult::PassDrawRange &range) {
        file << "  pass_exec[" << passIndex << "] offset=" << range.offset
             << " count=" << range.count << "\n";
        return true;
      });

  writeSection(file, "Unresolved Draw Buffer Bindings",
               std::span{snapshot.unresolvedDrawBufferBindings},
               [&](const RenderGraphCompileResult::UnresolvedDrawBufferBinding
                       &binding) {
                 file << "  pass_exec[" << binding.orderedPassIndex << "].draw["
                      << binding.drawIndex << "]."
                      << resolveDrawBufferBindingTarget(binding.target)
                      << " <- buf[" << binding.bufferResourceIndex << "]\n";
                 return true;
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
