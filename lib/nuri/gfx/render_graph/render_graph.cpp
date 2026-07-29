#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/core/containers/hash_map.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
namespace nuri {
namespace {
[[nodiscard]] std::string_view
resolvePassDebugName(const std::pmr::vector<std::pmr::string> &passDebugNames,
                     uint32_t passIndex) {
  if (passIndex >= passDebugNames.size()) {
    return "unnamed_pass";
  }
  const std::pmr::string &name = passDebugNames[passIndex];
  if (name.empty()) {
    return "unnamed_pass";
  }
  return std::string_view(name.data(), name.size());
}
[[nodiscard]] std::string_view
resolveResourceDebugName(std::string_view name, std::string_view fallback) {
  return name.empty() ? fallback : name;
}
[[nodiscard]] std::string makePassResourceDebugName(std::string_view passLabel,
                                                    std::string_view suffix) {
  const std::string_view base =
      passLabel.empty() ? std::string_view("graphics_pass") : passLabel;
  std::string out;
  out.reserve(base.size() + 1u + suffix.size());
  out.append(base.data(), base.size());
  out.push_back('_');
  out.append(suffix.data(), suffix.size());
  return out;
}
[[nodiscard]] bool isValidTransientTextureDesc(const TextureDesc &desc) {
  return desc.type != TextureType::Count && desc.format != Format::Count &&
         desc.storage != Storage::Count && desc.usage != TextureUsage::Count &&
         desc.dimensions.width > 0 && desc.dimensions.height > 0 &&
         desc.dimensions.depth > 0 && desc.numLayers > 0 &&
         desc.numSamples > 0 && desc.numMipLevels > 0;
}
[[nodiscard]] bool isValidTransientBufferDesc(const BufferDesc &desc) {
  return desc.usage != BufferUsage::None && desc.storage != Storage::Count &&
         desc.size > 0;
}
[[nodiscard]] uint64_t foldHandleKey(uint32_t index, uint32_t generation) {
  return (static_cast<uint64_t>(generation) << 32u) | index;
}
[[nodiscard]] uint64_t foldDependencyEdgeKey(uint32_t before, uint32_t after) {
  return (static_cast<uint64_t>(before) << 32u) | after;
}
template <typename Ranges>
[[nodiscard]] uint32_t rangeSlotCount(const Ranges &ranges) {
  uint32_t count = 0u;
  for (const auto range : ranges) {
    count = std::max(count, range.offset + range.count);
  }
  return count;
}
[[nodiscard]] bool
isWholeTextureRange(RenderGraphSubresourceRange range) noexcept {
  return range.firstMip == 0u && range.mipCount == UINT32_MAX &&
         range.firstLayer == 0u && range.layerCount == UINT32_MAX;
}
[[nodiscard]] bool textureRangesOverlap(RenderGraphSubresourceRange lhs,
                                        RenderGraphSubresourceRange rhs) {
  const auto intervalOverlaps = [](uint32_t lhsFirst, uint32_t lhsCount,
                                   uint32_t rhsFirst, uint32_t rhsCount) {
    const uint64_t lhsEnd = lhsCount == UINT32_MAX
                                ? std::numeric_limits<uint64_t>::max()
                                : static_cast<uint64_t>(lhsFirst) + lhsCount;
    const uint64_t rhsEnd = rhsCount == UINT32_MAX
                                ? std::numeric_limits<uint64_t>::max()
                                : static_cast<uint64_t>(rhsFirst) + rhsCount;
    return lhsFirst < rhsEnd && rhsFirst < lhsEnd;
  };
  return intervalOverlaps(lhs.firstMip, lhs.mipCount, rhs.firstMip,
                          rhs.mipCount) &&
         intervalOverlaps(lhs.firstLayer, lhs.layerCount, rhs.firstLayer,
                          rhs.layerCount);
}
struct TextureLifetimeTag {};
struct BufferLifetimeTag {};
template <typename Tag> struct TransientLifetimeRanks {
  std::pmr::vector<uint32_t> first;
  std::pmr::vector<uint32_t> last;
  TransientLifetimeRanks(std::pmr::memory_resource *memory, size_t count)
      : first(memory), last(memory) {
    first.resize(count, UINT32_MAX);
    last.resize(count, 0u);
  }
};
void ownPreDispatchPayload(FrameCommandArena &arena, size_t index,
                           const ComputeDispatchItem &source,
                           ComputeDispatchItem &destination) {
  auto &label = arena.ownedPreDispatchDebugLabels[index];
  label.assign(source.debugLabel.begin(), source.debugLabel.end());
  destination.debugLabel = label;
  auto &pushConstants = arena.ownedPreDispatchPushConstants[index];
  pushConstants.assign(source.pushConstants.begin(),
                       source.pushConstants.end());
  destination.pushConstants = pushConstants;
  auto &dependencyTextures = arena.ownedPreDispatchDependencyTextures[index];
  dependencyTextures.assign(source.dependencyTextures.begin(),
                            source.dependencyTextures.end());
  destination.dependencyTextures = dependencyTextures;
  auto &textureBindings = arena.ownedPreDispatchTextureBindings[index];
  textureBindings.assign(source.pushConstantTextureBindings.begin(),
                         source.pushConstantTextureBindings.end());
  destination.pushConstantTextureBindings = textureBindings;
}
void ownDrawPayload(FrameCommandArena &arena, size_t index,
                    const DrawItem &source, DrawItem &destination) {
  auto &label = arena.ownedDrawDebugLabels[index];
  label.assign(source.debugLabel.begin(), source.debugLabel.end());
  destination.debugLabel = label;
  auto &pushConstants = arena.ownedDrawPushConstants[index];
  pushConstants.assign(source.pushConstants.begin(),
                       source.pushConstants.end());
  destination.pushConstants = pushConstants;
  auto &textureBindings = arena.ownedDrawTextureBindings[index];
  textureBindings.assign(source.pushConstantTextureBindings.begin(),
                         source.pushConstantTextureBindings.end());
  destination.pushConstantTextureBindings = textureBindings;
}
void ownMeshDispatchPayload(FrameCommandArena &arena, size_t index,
                            const MeshDispatchItem &source,
                            MeshDispatchItem &destination) {
  auto &label = arena.ownedMeshDispatchDebugLabels[index];
  label.assign(source.debugLabel.begin(), source.debugLabel.end());
  destination.debugLabel = label;
  auto &pushConstants = arena.ownedMeshDispatchPushConstants[index];
  pushConstants.assign(source.pushConstants.begin(),
                       source.pushConstants.end());
  destination.pushConstants = pushConstants;
  auto &dependencyBuffers = arena.ownedMeshDispatchDependencyBuffers[index];
  dependencyBuffers.assign(source.dependencyBuffers.begin(),
                           source.dependencyBuffers.end());
  destination.dependencyBuffers = dependencyBuffers;
  auto &dependencyTextures = arena.ownedMeshDispatchDependencyTextures[index];
  dependencyTextures.assign(source.dependencyTextures.begin(),
                            source.dependencyTextures.end());
  destination.dependencyTextures = dependencyTextures;
  auto &textureBindings = arena.ownedMeshDispatchTextureBindings[index];
  textureBindings.assign(source.pushConstantTextureBindings.begin(),
                         source.pushConstantTextureBindings.end());
  destination.pushConstantTextureBindings = textureBindings;
}
void ownAccelerationStructureBuilds(FrameCommandArena &arena,
                                    size_t orderedPassIndex,
                                    const RenderPass &source,
                                    RenderPass &destination) {
  auto &builds = arena.ownedAccelerationStructureBuildsByPass[orderedPassIndex];
  auto &geometries =
      arena.ownedAccelerationStructureGeometriesByPass[orderedPassIndex];
  auto &instances =
      arena.ownedAccelerationStructureInstancesByPass[orderedPassIndex];
  builds.clear();
  geometries.clear();
  instances.clear();
  builds.reserve(source.accelerationStructureBuilds.size());
  geometries.reserve(source.accelerationStructureBuilds.size());
  instances.reserve(source.accelerationStructureBuilds.size());
  for (const AccelerationStructureBuildItem &sourceBuild :
       source.accelerationStructureBuilds) {
    geometries.emplace_back();
    instances.emplace_back();
    auto &ownedGeometries = geometries.back();
    auto &ownedInstances = instances.back();
    std::visit(
        [&](const auto &command) {
          using Command = std::decay_t<decltype(command)>;
          if constexpr (std::is_same_v<Command, BuildBlasItem> ||
                        std::is_same_v<Command, UpdateBlasItem>) {
            ownedGeometries.assign(command.geometries.begin(),
                                   command.geometries.end());
            builds.push_back(AccelerationStructureBuildItem{
                .command = Command{.destination = command.destination,
                                   .geometries = ownedGeometries}});
          } else {
            ownedInstances.assign(command.instances.begin(),
                                  command.instances.end());
            builds.push_back(AccelerationStructureBuildItem{
                .command = Command{.destination = command.destination,
                                   .instances = ownedInstances}});
          }
        },
        sourceBuild.command);
  }
  destination.accelerationStructureBuilds = builds;
}
struct RecordingReferenceSet {
  std::pmr::vector<BufferHandle> buffers;
  std::pmr::vector<TextureHandle> textures;
  std::pmr::vector<AccelerationStructureHandle> accelerationStructures;
  std::pmr::vector<RenderPipelineHandle> renderPipelines;
  std::pmr::vector<ComputePipelineHandle> computePipelines;
  std::pmr::vector<MeshletPipelineHandle> meshletPipelines;
  std::pmr::vector<RayQueryBindingHandle> rayQueryBindings;

  explicit RecordingReferenceSet(std::pmr::memory_resource *memory)
      : buffers(memory), textures(memory), accelerationStructures(memory),
        renderPipelines(memory), computePipelines(memory),
        meshletPipelines(memory), rayQueryBindings(memory) {}

  template <typename Handle>
  static void append(std::pmr::vector<Handle> &out, Handle handle) {
    if (nuri::isValid(handle)) {
      out.push_back(handle);
    }
  }
  template <typename Handle>
  static void append(std::pmr::vector<Handle> &out,
                     std::span<const Handle> handles) {
    for (const Handle handle : handles) {
      append(out, handle);
    }
  }
  template <typename Handle>
  static void makeUnique(std::pmr::vector<Handle> &handles) {
    std::sort(handles.begin(), handles.end(), [](Handle lhs, Handle rhs) {
      return handleKey(lhs) < handleKey(rhs);
    });
    handles.erase(std::unique(handles.begin(), handles.end()), handles.end());
  }
  void normalize() {
    makeUnique(buffers);
    makeUnique(textures);
    makeUnique(accelerationStructures);
    makeUnique(renderPipelines);
    makeUnique(computePipelines);
    makeUnique(meshletPipelines);
    makeUnique(rayQueryBindings);
  }
  [[nodiscard]] GraphicsRecordingReferences view() const {
    return {
        .buffers = buffers,
        .textures = textures,
        .accelerationStructures = accelerationStructures,
        .renderPipelines = renderPipelines,
        .computePipelines = computePipelines,
        .meshletPipelines = meshletPipelines,
        .rayQueryBindings = rayQueryBindings,
    };
  }
};
void appendRecordingReferences(RecordingReferenceSet &out,
                               const RenderPass &pass) {
  RecordingReferenceSet::append(out.textures, pass.colorTexture);
  RecordingReferenceSet::append(out.textures, pass.colorResolveTexture);
  RecordingReferenceSet::append(out.textures, pass.depthTexture);
  RecordingReferenceSet::append(out.textures, pass.depthResolveTexture);
  RecordingReferenceSet::append(out.buffers, pass.dependencyBuffers);
  RecordingReferenceSet::append(out.textures, pass.dependencyTextures);
  for (const ComputeDispatchItem &dispatch : pass.preDispatches) {
    RecordingReferenceSet::append(out.computePipelines, dispatch.pipeline);
    RecordingReferenceSet::append(out.rayQueryBindings,
                                  dispatch.rayQueryBinding);
    RecordingReferenceSet::append(out.buffers, dispatch.dependencyBuffers);
    RecordingReferenceSet::append(out.textures, dispatch.dependencyTextures);
    for (const PushConstantTextureBinding binding :
         dispatch.pushConstantTextureBindings) {
      RecordingReferenceSet::append(out.textures, binding.texture);
    }
  }
  for (const DrawItem &draw : pass.draws) {
    RecordingReferenceSet::append(out.renderPipelines, draw.pipeline);
    RecordingReferenceSet::append(out.buffers, draw.vertexBuffer);
    RecordingReferenceSet::append(out.buffers, draw.indexBuffer);
    RecordingReferenceSet::append(out.buffers, draw.indirectBuffer);
    RecordingReferenceSet::append(out.buffers, draw.indirectCountBuffer);
    for (const PushConstantTextureBinding binding :
         draw.pushConstantTextureBindings) {
      RecordingReferenceSet::append(out.textures, binding.texture);
    }
  }
  for (const MeshDispatchItem &dispatch : pass.meshDispatches) {
    RecordingReferenceSet::append(out.meshletPipelines, dispatch.pipeline);
    RecordingReferenceSet::append(out.buffers, dispatch.indirectBuffer);
    RecordingReferenceSet::append(out.buffers, dispatch.indirectCountBuffer);
    RecordingReferenceSet::append(out.buffers, dispatch.dependencyBuffers);
    RecordingReferenceSet::append(out.textures, dispatch.dependencyTextures);
    for (const PushConstantTextureBinding binding :
         dispatch.pushConstantTextureBindings) {
      RecordingReferenceSet::append(out.textures, binding.texture);
    }
  }
  for (const BufferCopyRegion &copy : pass.bufferCopies) {
    RecordingReferenceSet::append(out.buffers, copy.srcBuffer);
    RecordingReferenceSet::append(out.buffers, copy.dstBuffer);
  }
  for (const TextureCopyItem &copy : pass.textureCopies) {
    RecordingReferenceSet::append(out.textures, copy.sourceTexture);
    RecordingReferenceSet::append(out.textures, copy.destinationTexture);
  }
  for (const AccelerationStructureBuildItem &build :
       pass.accelerationStructureBuilds) {
    std::visit(
        [&out](const auto &command) {
          RecordingReferenceSet::append(out.accelerationStructures,
                                        command.destination);
          using Command = std::decay_t<decltype(command)>;
          if constexpr (std::is_same_v<Command, BuildBlasItem> ||
                        std::is_same_v<Command, UpdateBlasItem>) {
            for (const auto &geometry : command.geometries) {
              RecordingReferenceSet::append(out.buffers, geometry.vertexBuffer);
              RecordingReferenceSet::append(out.buffers, geometry.indexBuffer);
              RecordingReferenceSet::append(out.buffers,
                                            geometry.transformBuffer);
            }
          } else {
            for (const auto &instance : command.instances) {
              RecordingReferenceSet::append(out.accelerationStructures,
                                            instance.bottomLevel);
            }
          }
        },
        build.command);
  }
  const auto &external = pass.externalTemporalDispatch.execute;
  for (const TextureHandle texture :
       {external.sceneColor, external.sceneDepth, external.motionVectors,
        external.reactiveMask, external.compositionMask, external.exposure,
        external.output}) {
    RecordingReferenceSet::append(out.textures, texture);
  }
}
void appendRecordingReferences(
    RecordingReferenceSet &out,
    std::span<const GraphicsBarrierRecord> barriers) {
  for (const GraphicsBarrierRecord &barrier : barriers) {
    if (barrier.isTexture()) {
      RecordingReferenceSet::append(out.textures, barrier.textureHandle());
    } else if (barrier.resourceKind == GraphicsBarrierResourceKind::Buffer) {
      RecordingReferenceSet::append(out.buffers, barrier.bufferHandle());
    } else {
      RecordingReferenceSet::append(out.accelerationStructures,
                                    barrier.accelerationStructureHandle());
    }
  }
}
[[nodiscard]] uint64_t quantizeToNextPow2(uint64_t value) noexcept {
  if (value <= 1u) {
    return value;
  }
  uint64_t v = value - 1u;
  v |= v >> 1u;
  v |= v >> 2u;
  v |= v >> 4u;
  v |= v >> 8u;
  v |= v >> 16u;
  v |= v >> 32u;
  return v + 1u;
}
template <typename PassBindings>
[[nodiscard]] uint64_t
computePassPayloadLayoutHash(const std::pmr::vector<RenderPass> &passes,
                             const PassBindings &bindings) noexcept {
  uint64_t hash = 0xcbf29ce484222325ull;
  const auto mix = [&hash](uint64_t value) noexcept {
    hash ^= value;
    hash *= 0x100000001b3ull;
  };
  for (size_t passIndex = 0; passIndex < passes.size(); ++passIndex) {
    const RenderPass &pass = passes[passIndex];
    mix(static_cast<uint64_t>(pass.executionMode));
    mix(pass.hasColorAttachment ? 1u : 0u);
    mix(nuri::isValid(pass.colorResolveTexture) ? 1u : 0u);
    mix(nuri::isValid(pass.depthResolveTexture) ? 1u : 0u);
    mix(pass.useViewport ? 1u : 0u);
    mix(pass.payloadBorrowed ? 1u : 0u);
    mix(pass.drawBuffersPreResolved ? 1u : 0u);
    mix(static_cast<uint64_t>(pass.gpuTimingScope));
    mix(static_cast<uint64_t>(pass.dependencyBuffers.size()));
    const size_t dependencyTextureCount =
        passIndex < bindings.size()
            ? bindings[passIndex].dependencyTextures.count
            : pass.dependencyTextures.size();
    mix(static_cast<uint64_t>(dependencyTextureCount));
    mix(static_cast<uint64_t>(pass.preDispatches.size()));
    const bool borrowedPreResolvedDrawPayload =
        pass.payloadBorrowed && pass.drawBuffersPreResolved;
    mix(borrowedPreResolvedDrawPayload
            ? 0u
            : quantizeToNextPow2(static_cast<uint64_t>(pass.draws.size())));
    mix(static_cast<uint64_t>(pass.meshDispatches.size()));
    mix(static_cast<uint64_t>(pass.bufferCopies.size()));
    mix(static_cast<uint64_t>(pass.textureCopies.size()));
    mix(quantizeToNextPow2(
        static_cast<uint64_t>(pass.accelerationStructureBuilds.size())));
    for (const AccelerationStructureBuildItem &build :
         pass.accelerationStructureBuilds) {
      mix(static_cast<uint64_t>(build.command.index()));
      std::visit(
          [&mix](const auto &command) {
            using Command = std::decay_t<decltype(command)>;
            if constexpr (std::is_same_v<Command, BuildBlasItem> ||
                          std::is_same_v<Command, UpdateBlasItem>) {
              mix(static_cast<uint64_t>(command.geometries.size()));
              for (const auto &geometry : command.geometries) {
                mix(static_cast<uint64_t>(geometry.vertexFormat));
                mix(static_cast<uint64_t>(geometry.indexFormat));
                mix(geometry.vertexStrideBytes);
                mix(static_cast<uint64_t>(geometry.flags));
                mix(nuri::isValid(geometry.transformBuffer) ? 1u : 0u);
              }
            } else {
              mix(quantizeToNextPow2(
                  static_cast<uint64_t>(command.instances.size())));
            }
          },
          build.command);
    }
    mix(pass.externalTemporalDispatch.backend != nullptr ? 1u : 0u);
    for (const ComputeDispatchItem &dispatch : pass.preDispatches) {
      mix(static_cast<uint64_t>(dispatch.dependencyBuffers.size()));
      mix(static_cast<uint64_t>(dispatch.dependencyTextures.size()));
      mix(nuri::isValid(dispatch.rayQueryBinding) ? 1u : 0u);
      for (const PushConstantTextureBinding binding :
           dispatch.pushConstantTextureBindings) {
        mix(binding.byteOffset);
        mix(binding.graphTextureResourceIndex);
        mix(static_cast<uint64_t>(binding.access));
      }
    }
    for (const DrawItem &draw : pass.draws) {
      for (const PushConstantTextureBinding binding :
           draw.pushConstantTextureBindings) {
        mix(binding.byteOffset);
        mix(binding.graphTextureResourceIndex);
        mix(static_cast<uint64_t>(binding.access));
      }
    }
    for (const MeshDispatchItem &dispatch : pass.meshDispatches) {
      mix(static_cast<uint64_t>(dispatch.command));
      mix(nuri::isValid(dispatch.indirectBuffer) ? 1u : 0u);
      mix(nuri::isValid(dispatch.indirectCountBuffer) ? 1u : 0u);
      for (const PushConstantTextureBinding binding :
           dispatch.pushConstantTextureBindings) {
        mix(binding.byteOffset);
        mix(binding.graphTextureResourceIndex);
        mix(static_cast<uint64_t>(binding.access));
      }
    }
  }
  return hash;
}
[[nodiscard]] bool
isComputeOnlyExecutionMode(RenderPassExecutionMode mode) noexcept {
  return mode == RenderPassExecutionMode::ComputeOnly;
}
[[nodiscard]] bool
isCopyOnlyExecutionMode(RenderPassExecutionMode mode) noexcept {
  return mode == RenderPassExecutionMode::CopyOnly;
}
[[nodiscard]] bool
isExternalTemporalExecutionMode(RenderPassExecutionMode mode) noexcept {
  return mode == RenderPassExecutionMode::ExternalTemporal;
}
[[nodiscard]] RenderGraphTextureId
graphicsPassRootColorTexture(const RenderGraphGraphicsPassDesc &desc) noexcept {
  return nuri::isValid(desc.colorResolveTexture) ? desc.colorResolveTexture
                                                 : desc.colorTexture;
}
[[nodiscard]] Result<bool, std::string>
validatePassExecutionMode(const RenderGraphGraphicsPassDesc &desc,
                          std::string_view caller) {
  if (isCopyOnlyExecutionMode(desc.executionMode)) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": copy-only pass must use "
                                                "a dedicated copy-pass API");
  }
  const bool computeOnly = isComputeOnlyExecutionMode(desc.executionMode);
  const bool externalTemporal =
      isExternalTemporalExecutionMode(desc.executionMode);
  if (!computeOnly && !externalTemporal) {
    if (desc.externalTemporalDispatch.backend != nullptr) {
      return Result<bool, std::string>::makeError(
          std::string(caller) +
          ": graphics pass cannot contain an external temporal dispatch");
    }
    return Result<bool, std::string>::makeResult(true);
  }
  if (desc.hasColorAttachment) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": compute pass cannot "
                                                "have a color attachment");
  }
  if (nuri::isValid(desc.colorTexture)) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": compute pass cannot "
                                                "bind a color texture");
  }
  if (nuri::isValid(desc.colorResolveTexture)) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": compute pass cannot "
                                                "bind a color resolve texture");
  }
  if (nuri::isValid(desc.depthTexture)) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": compute pass cannot "
                                                "bind a depth texture");
  }
  if (nuri::isValid(desc.depthResolveTexture)) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": compute pass cannot "
                                                "bind a depth resolve texture");
  }
  if (!desc.draws.empty()) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": compute pass cannot "
                                                "contain draws");
  }
  if (!desc.meshDispatches.empty() ||
      !desc.meshDispatchBufferBindings.empty()) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": compute pass cannot "
                                                "contain mesh dispatches");
  }
  if (desc.drawBuffersPreResolved) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": compute pass cannot "
                                                "use pre-resolved draw "
                                                "buffers");
  }
  if (desc.markColorAsFrameOutput) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": compute pass cannot "
                                                "mark frame output");
  }
  if (computeOnly && desc.preDispatches.empty()) {
    return Result<bool, std::string>::makeError(std::string(caller) +
                                                ": compute-only pass requires "
                                                "at least one dispatch");
  }
  if (computeOnly && desc.externalTemporalDispatch.backend != nullptr) {
    return Result<bool, std::string>::makeError(
        std::string(caller) +
        ": compute-only pass cannot contain an external temporal dispatch");
  }
  if (externalTemporal && !desc.preDispatches.empty()) {
    return Result<bool, std::string>::makeError(
        std::string(caller) +
        ": external temporal pass cannot contain native compute dispatches");
  }
  if (externalTemporal && desc.externalTemporalDispatch.backend == nullptr) {
    return Result<bool, std::string>::makeError(
        std::string(caller) +
        ": external temporal pass requires a typed provider dispatch");
  }
  return Result<bool, std::string>::makeResult(true);
}
[[nodiscard]] uint64_t foldPassResourceKey(uint32_t passIndex,
                                           uint32_t resourceIndex) {
  return (static_cast<uint64_t>(passIndex) << 32u) | resourceIndex;
}
[[nodiscard]] bool isTextureDescAliasCompatible(const TextureDesc &a,
                                                const TextureDesc &b) {
  return a.type == b.type && a.format == b.format &&
         a.dimensions.width == b.dimensions.width &&
         a.dimensions.height == b.dimensions.height &&
         a.dimensions.depth == b.dimensions.depth && a.usage == b.usage &&
         a.storage == b.storage && a.numLayers == b.numLayers &&
         a.numSamples == b.numSamples && a.numMipLevels == b.numMipLevels &&
         a.dataNumMipLevels == b.dataNumMipLevels &&
         a.generateMipmaps == b.generateMipmaps;
}
[[nodiscard]] bool isBufferDescAliasCompatible(const BufferDesc &a,
                                               const BufferDesc &b) {
  return a.usage == b.usage && a.storage == b.storage && a.size == b.size;
}
[[nodiscard]] uint64_t hashTextureDescForPool(const TextureDesc &d) noexcept {
  uint64_t h = 0xcbf29ce484222325ull;
  const auto mix = [&h](uint64_t v) noexcept {
    h ^= v;
    h *= 0x100000001b3ull;
  };
  mix(static_cast<uint64_t>(d.type));
  mix(static_cast<uint64_t>(d.format));
  mix(static_cast<uint64_t>(d.dimensions.width));
  mix(static_cast<uint64_t>(d.dimensions.height));
  mix(static_cast<uint64_t>(d.dimensions.depth));
  mix(static_cast<uint64_t>(d.usage));
  mix(static_cast<uint64_t>(d.storage));
  mix(static_cast<uint64_t>(d.numLayers));
  mix(static_cast<uint64_t>(d.numSamples));
  mix(static_cast<uint64_t>(d.numMipLevels));
  mix(static_cast<uint64_t>(d.dataNumMipLevels));
  mix(static_cast<uint64_t>(d.generateMipmaps ? 1u : 0u));
  return h;
}
[[nodiscard]] uint64_t hashBufferDescForPool(const BufferDesc &d) noexcept {
  uint64_t h = 0xcbf29ce484222325ull;
  const auto mix = [&h](uint64_t v) noexcept {
    h ^= v;
    h *= 0x100000001b3ull;
  };
  mix(static_cast<uint64_t>(d.usage));
  mix(static_cast<uint64_t>(d.storage));
  mix(d.size);
  return h;
}
void mixFingerprintValue(uint64_t &hash, uint64_t value) noexcept {
  hash ^= value;
  hash *= 0x100000001b3ull;
}
void appendTransientTextureDescriptorFingerprint(uint64_t &hash,
                                                 const TextureDesc &desc) {
  mixFingerprintValue(hash, 0x746578ull);
  mixFingerprintValue(hash, hashTextureDescForPool(desc));
}
void appendTransientBufferDescriptorFingerprint(uint64_t &hash,
                                                const BufferDesc &desc) {
  mixFingerprintValue(hash, 0x627566ull);
  mixFingerprintValue(hash, hashBufferDescForPool(desc));
}
constexpr size_t kMaxReusableTransientTextures = 32u;
constexpr size_t kMaxReusableTransientBuffers = 64u;
constexpr uint32_t kMinHazardGroupsPerWorker = 128u;
constexpr uint32_t kMinPayloadPassesPerWorker = 24u;
constexpr uint32_t kMinLifetimeItemsPerWorker = 256u;
constexpr uint32_t kMinRecordingPassesPerWorker = 12u;
[[nodiscard]] std::vector<RenderGraphContiguousRange>
makeAdaptiveRanges(uint32_t itemCount, uint32_t maxRangeCount,
                   uint32_t minItemsPerWorker) {
  if (itemCount == 0u || maxRangeCount == 0u) {
    return {};
  }
  uint32_t rangeCount = std::min(itemCount, maxRangeCount);
  if (rangeCount > 1u && minItemsPerWorker > 1u) {
    const uint32_t granularityLimitedCount =
        std::max(1u, itemCount / minItemsPerWorker);
    rangeCount = std::min(rangeCount, granularityLimitedCount);
  }
  return RenderGraphRuntime::makeRanges(itemCount, rangeCount);
}
[[nodiscard]] std::vector<RenderGraphContiguousRange>
makeHazardRanges(uint32_t itemCount, uint32_t workerCount) {
  return makeAdaptiveRanges(itemCount, workerCount, kMinHazardGroupsPerWorker);
}
[[nodiscard]] std::vector<RenderGraphContiguousRange>
makePayloadRanges(uint32_t itemCount, uint32_t workerCount) {
  return makeAdaptiveRanges(itemCount, workerCount, kMinPayloadPassesPerWorker);
}
[[nodiscard]] std::vector<RenderGraphContiguousRange>
makeLifetimeRanges(uint32_t itemCount, uint32_t workerCount) {
  return makeAdaptiveRanges(itemCount, workerCount, kMinLifetimeItemsPerWorker);
}
[[nodiscard]] std::vector<RenderGraphContiguousRange>
makeRecordingRanges(uint32_t itemCount, uint32_t workerCount) {
  return makeAdaptiveRanges(itemCount, workerCount,
                            kMinRecordingPassesPerWorker);
}
[[nodiscard]] RenderGraphAccessMode attachmentAccessMode(LoadOp loadOp) {
  RenderGraphAccessMode mode = RenderGraphAccessMode::None;
  if (loadOp == LoadOp::Load) {
    mode = mode | RenderGraphAccessMode::Read;
  }
  mode = mode | RenderGraphAccessMode::Write;
  return mode;
}
[[nodiscard]] std::string
makeExecutionStageError(RenderGraphExecutionFailureStage stage,
                        std::string_view message) {
  std::string error = "[stage=";
  const std::string_view stageName = toString(stage);
  error.append(stageName.data(), stageName.size());
  error.append("] ");
  error.append(message.data(), message.size());
  return error;
}
} // namespace

std::string_view toString(RenderGraphExecutionFailureStage stage) noexcept {
  switch (stage) {
  case RenderGraphExecutionFailureStage::MaterializeTransients:
    return "materialize_transients";
  case RenderGraphExecutionFailureStage::AcquireRecordingContext:
    return "acquire_recording_context";
  case RenderGraphExecutionFailureStage::RetainRecordingReferences:
    return "retain_recording_references";
  case RenderGraphExecutionFailureStage::RecordGraphicsBarriers:
    return "record_graphics_barriers";
  case RenderGraphExecutionFailureStage::RecordGraphicsPasses:
    return "record_graphics_passes";
  case RenderGraphExecutionFailureStage::FinishRecordingContext:
    return "finish_recording_context";
  case RenderGraphExecutionFailureStage::SubmitRecordedFrame:
    return "submit_recorded_frame";
  case RenderGraphExecutionFailureStage::PresentFrameOutput:
    return "present_frame_output";
  default:
    return "unknown";
  }
}

RenderGraphBuilder::RenderGraphBuilder(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      textures_(memory_), buffers_(memory_), accelerationStructures_(memory_),
      ownedPassPayloads_(memory_), passes_(memory_), passDebugNames_(memory_),
      passBindings_(memory_),
      passDependencyBufferBindingResourceIndices_(memory_),
      passDependencyTextureBindingResourceIndices_(memory_),
      preDispatchDependencyBindings_(memory_),
      preDispatchDependencyBindingResourceIndices_(memory_),
      drawBindings_(memory_), meshDispatchBindings_(memory_),
      bufferCopyBindings_(memory_), textureCopyBindings_(memory_),
      importedTextureIndicesByHandle_(memory_),
      importedBufferIndicesByHandle_(memory_),
      importedAccelerationStructureIndicesByHandle_(memory_),
      explicitBufferAccessIndicesByPassResource_(memory_),
      inferredBufferAccessIndicesByPassResource_(memory_),
      explicitAccelerationStructureAccessIndicesByPassResource_(memory_),
      inferredAccelerationStructureAccessIndicesByPassResource_(memory_),
      dependencyEdgeKeys_(memory_), dependencies_(memory_),
      resourceUses_(memory_), frameOutputTextureSet_(memory_),
      frameOutputTextureIndices_(memory_),
      sideEffectMarkIndicesByPass_(memory_), sideEffectPassMarks_(memory_) {}

void RenderGraphBuilder::beginFrame(uint64_t frameIndex) {
  frameIndex_ = frameIndex;
  textures_.clear();
  buffers_.clear();
  accelerationStructures_.clear();
  ownedPassPayloads_.clear();
  passes_.clear();
  passDebugNames_.clear();
  passBindings_.clear();
  passDependencyBufferBindingResourceIndices_.clear();
  passDependencyTextureBindingResourceIndices_.clear();
  preDispatchDependencyBindings_.clear();
  preDispatchDependencyBindingResourceIndices_.clear();
  drawBindings_.clear();
  meshDispatchBindings_.clear();
  bufferCopyBindings_.clear();
  textureCopyBindings_.clear();
  importedTextureIndicesByHandle_.clear();
  importedBufferIndicesByHandle_.clear();
  importedAccelerationStructureIndicesByHandle_.clear();
  explicitBufferAccessIndicesByPassResource_.clear();
  inferredBufferAccessIndicesByPassResource_.clear();
  explicitAccelerationStructureAccessIndicesByPassResource_.clear();
  inferredAccelerationStructureAccessIndicesByPassResource_.clear();
  dependencyEdgeKeys_.clear();
  dependencies_.clear();
  resourceUses_.clear();
  frameOutputTextureSet_.clear();
  frameOutputTextureIndices_.clear();
  sideEffectMarkIndicesByPass_.clear();
  sideEffectPassMarks_.clear();
  allPassesBorrowPayload_ = true;
  transientResourceDescriptorsHash_ = 0xcbf29ce484222325ull;
}

PersistentBufferId
RenderGraphBuilder::registerPersistentBuffer(BufferHandle handle,
                                             std::string_view debugName) {
  PersistentBufferId id{};
  if (!persistentBufferFreeIndices_.empty()) {
    id.value = persistentBufferFreeIndices_.back();
    persistentBufferFreeIndices_.pop_back();
    auto &entry = persistentBuffers_[id.value];
    entry.occupied = true;
    entry.handle = handle;
    entry.debugName.assign(debugName.data(), debugName.size());
  } else {
    id.value = static_cast<uint32_t>(persistentBuffers_.size());
    persistentBuffers_.push_back({.occupied = true,
                                  .handle = handle,
                                  .debugName = std::string(debugName)});
  }
  ++persistentHandlesVersion_;
  if (nuri::isValid(handle)) {
    (void)importBuffer(handle, debugName);
  }
  return id;
}

PersistentTextureId
RenderGraphBuilder::registerPersistentTexture(TextureHandle handle,
                                              std::string_view debugName) {
  PersistentTextureId id{};
  if (!persistentTextureFreeIndices_.empty()) {
    id.value = persistentTextureFreeIndices_.back();
    persistentTextureFreeIndices_.pop_back();
    auto &entry = persistentTextures_[id.value];
    entry.occupied = true;
    entry.handle = handle;
    entry.debugName.assign(debugName.data(), debugName.size());
  } else {
    id.value = static_cast<uint32_t>(persistentTextures_.size());
    persistentTextures_.push_back({.occupied = true,
                                   .handle = handle,
                                   .debugName = std::string(debugName)});
  }
  ++persistentHandlesVersion_;
  if (nuri::isValid(handle)) {
    (void)importTexture(handle, debugName);
  }
  return id;
}

void RenderGraphBuilder::updatePersistentBuffer(PersistentBufferId id,
                                                BufferHandle newHandle) {
  if (id.value >= persistentBuffers_.size() ||
      !persistentBuffers_[id.value].occupied) {
    return;
  }
  persistentBuffers_[id.value].handle = newHandle;
  ++persistentHandlesVersion_;
  if (nuri::isValid(newHandle)) {
    (void)importBuffer(newHandle, persistentBuffers_[id.value].debugName);
  }
}

void RenderGraphBuilder::updatePersistentTexture(PersistentTextureId id,
                                                 TextureHandle newHandle) {
  if (id.value >= persistentTextures_.size() ||
      !persistentTextures_[id.value].occupied) {
    return;
  }
  persistentTextures_[id.value].handle = newHandle;
  ++persistentHandlesVersion_;
  if (nuri::isValid(newHandle)) {
    (void)importTexture(newHandle, persistentTextures_[id.value].debugName);
  }
}

void RenderGraphBuilder::unregisterPersistentBuffer(PersistentBufferId id) {
  if (id.value >= persistentBuffers_.size()) {
    return;
  }
  auto &entry = persistentBuffers_[id.value];
  if (!entry.occupied) {
    return;
  }
  entry.occupied = false;
  entry.handle = BufferHandle{};
  entry.debugName.clear();
  persistentBufferFreeIndices_.push_back(id.value);
  ++persistentHandlesVersion_;
}

void RenderGraphBuilder::unregisterPersistentTexture(PersistentTextureId id) {
  if (id.value >= persistentTextures_.size()) {
    return;
  }
  auto &entry = persistentTextures_[id.value];
  if (!entry.occupied) {
    return;
  }
  entry.occupied = false;
  entry.handle = TextureHandle{};
  entry.debugName.clear();
  persistentTextureFreeIndices_.push_back(id.value);
  ++persistentHandlesVersion_;
}

RenderGraphBuilder::GraphFingerprint
RenderGraphBuilder::computeGraphFingerprint() const noexcept {
  uint64_t payloadLayoutHash =
      computePassPayloadLayoutHash(passes_, passBindings_);
  const auto mixPayload = [&payloadLayoutHash](uint64_t value) noexcept {
    payloadLayoutHash ^= value;
    payloadLayoutHash *= 0x100000001b3ull;
  };
  for (size_t passIndex = 0; passIndex < passes_.size(); ++passIndex) {
    const auto [bindingOffset, bindingCount] =
        passBindings_[passIndex].meshDispatches;
    mixPayload(bindingCount);
    for (uint32_t bindingIndex = 0; bindingIndex < bindingCount;
         ++bindingIndex) {
      const uint32_t globalBindingIndex = bindingOffset + bindingIndex;
      const bool hasIndirect =
          meshDispatchBindings_[globalBindingIndex].indirect != UINT32_MAX;
      const bool hasIndirectCount =
          meshDispatchBindings_[globalBindingIndex].indirectCount != UINT32_MAX;
      mixPayload(hasIndirect ? 1u : 0u);
      mixPayload(hasIndirectCount ? 1u : 0u);
    }
  }
  uint64_t structuralIdentityHash = 0xcbf29ce484222325ull;
  const auto mixStructure = [&structuralIdentityHash](uint64_t value) noexcept {
    structuralIdentityHash ^= value;
    structuralIdentityHash *= 0x100000001b3ull;
  };
  const auto mixU32Range = [&mixStructure](const auto &values) noexcept {
    mixStructure(static_cast<uint64_t>(values.size()));
    for (const uint32_t value : values) {
      mixStructure(value);
    }
  };
  mixStructure(0x7465787475726573ull);
  mixStructure(static_cast<uint64_t>(textures_.size()));
  for (const TextureResource &texture : textures_) {
    mixStructure(texture.imported ? 1u : 0u);
  }
  mixStructure(0x6275666665727300ull);
  mixStructure(static_cast<uint64_t>(buffers_.size()));
  for (const BufferResource &buffer : buffers_) {
    mixStructure(buffer.imported ? 1u : 0u);
  }
  mixStructure(0x616363656c737472ull);
  mixStructure(static_cast<uint64_t>(accelerationStructures_.size()));
  mixStructure(static_cast<uint64_t>(passes_.size()));
  for (const RenderPass &pass : passes_) {
    mixStructure(static_cast<uint64_t>(pass.executionMode));
    mixStructure(pass.hasColorAttachment ? 1u : 0u);
    mixStructure(pass.drawBuffersPreResolved ? 1u : 0u);
  }
  mixStructure(static_cast<uint64_t>(dependencies_.size()));
  for (const DependencyEdge &dependency : dependencies_) {
    mixStructure(dependency.before);
    mixStructure(dependency.after);
  }
  mixStructure(static_cast<uint64_t>(resourceUses_.size()));
  for (const RenderGraphResourceUse &access : resourceUses_) {
    mixStructure(access.passIndex);
    mixStructure(static_cast<uint64_t>(access.resourceKind));
    mixStructure(access.resourceIndex);
    mixStructure(static_cast<uint64_t>(access.access));
    mixStructure(static_cast<uint64_t>(access.state));
    mixStructure(static_cast<uint64_t>(access.stage));
    mixStructure(access.subresources.firstMip);
    mixStructure(access.subresources.mipCount);
    mixStructure(access.subresources.firstLayer);
    mixStructure(access.subresources.layerCount);
    mixStructure(static_cast<uint64_t>(access.provenance));
  }
  mixU32Range(frameOutputTextureIndices_);
  mixStructure(static_cast<uint64_t>(sideEffectPassMarks_.size()));
  for (const SideEffectPassMark &mark : sideEffectPassMarks_) {
    mixStructure(mark.passIndex);
    mixStructure(mark.inferred ? 1u : 0u);
  }
  mixStructure(suppressInferredSideEffectsWhenExplicitOutputs_ ? 1u : 0u);
  mixStructure(passBindings_.size());
  for (const PassBindings &bindings : passBindings_) {
    mixStructure(bindings.color);
    mixStructure(bindings.colorResolve);
    mixStructure(bindings.depth);
    mixStructure(bindings.depthResolve);
    mixStructure(bindings.dependencyBuffers.offset);
    mixStructure(bindings.dependencyBuffers.count);
    mixStructure(bindings.dependencyTextures.offset);
    mixStructure(bindings.dependencyTextures.count);
    mixStructure(bindings.preDispatches.offset);
    mixStructure(bindings.preDispatches.count);
    mixStructure(bindings.draws.offset);
    mixStructure(bindings.draws.count);
    mixStructure(bindings.meshDispatches.offset);
    mixStructure(bindings.meshDispatches.count);
    mixStructure(bindings.bufferCopies.offset);
    mixStructure(bindings.bufferCopies.count);
    mixStructure(bindings.textureCopies.offset);
    mixStructure(bindings.textureCopies.count);
  }
  mixU32Range(passDependencyBufferBindingResourceIndices_);
  mixU32Range(passDependencyTextureBindingResourceIndices_);
  mixStructure(preDispatchDependencyBindings_.size());
  for (const BindingRange range : preDispatchDependencyBindings_) {
    mixStructure(range.offset);
    mixStructure(range.count);
  }
  mixU32Range(preDispatchDependencyBindingResourceIndices_);
  mixStructure(drawBindings_.size());
  for (const DrawBindings &binding : drawBindings_) {
    mixStructure(binding.vertex);
    mixStructure(binding.index);
    mixStructure(binding.indirect);
    mixStructure(binding.indirectCount);
  }
  mixStructure(meshDispatchBindings_.size());
  for (const MeshDispatchBindings &binding : meshDispatchBindings_) {
    mixStructure(binding.indirect);
    mixStructure(binding.indirectCount);
  }
  mixStructure(bufferCopyBindings_.size());
  for (const BufferCopyBindings &binding : bufferCopyBindings_) {
    mixStructure(binding.source);
    mixStructure(binding.destination);
  }
  mixStructure(textureCopyBindings_.size());
  for (const TextureCopyBindings &binding : textureCopyBindings_) {
    mixStructure(binding.source);
    mixStructure(binding.destination);
  }
  return GraphFingerprint{
      .passCount = passes_.size(),
      .totalTextureCount = textures_.size(),
      .totalBufferCount = buffers_.size(),
      .totalAccelerationStructureCount = accelerationStructures_.size(),
      .edgeCount = dependencies_.size(),
      .passAccessCount = resourceUses_.size(),
      .frameOutputCount = frameOutputTextureIndices_.size(),
      .sideEffectMarkCount = sideEffectPassMarks_.size(),
      .allPassesBorrowPayload = allPassesBorrowPayload_,
      .payloadLayoutHash = payloadLayoutHash,
      .structuralIdentityHash = structuralIdentityHash,
      .transientResourceDescriptorsHash = transientResourceDescriptorsHash_,
      .persistentHandlesVersion = persistentHandlesVersion_,
  };
}

FrameCommandArena
RenderGraphBuilder::buildFrameCommands(const RenderGraphPlan &plan) const {
  FrameCommandArena commands(memory_);
  commands.frameIndex = frameIndex_;
  commands.textureHandlesByResource.resize(textures_.size());
  commands.bufferHandlesByResource.resize(buffers_.size());
  commands.accelerationStructureHandlesByResource.resize(
      accelerationStructures_.size());
  commands.orderedPasses.resize(plan.orderedPassIndices.size());
  commands.passDebugNames.assign(passDebugNames_.begin(),
                                 passDebugNames_.end());
  commands.resolvedDependencyBuffers.resize(
      plan.resolvedDependencyBufferResourceIndices.size());
  commands.resolvedDependencyTextures.resize(
      plan.resolvedDependencyTextureResourceIndices.size());
  commands.resolvedPreDispatchDependencyBuffers.resize(
      plan.resolvedPreDispatchDependencyBufferResourceIndices.size());
  commands.ownedPreDispatches.resize(plan.preDispatchDependencyRanges.size());
  commands.ownedPreDispatchDebugLabels.resize(
      plan.preDispatchDependencyRanges.size());
  commands.ownedPreDispatchPushConstants.resize(
      plan.preDispatchDependencyRanges.size());
  commands.ownedPreDispatchDependencyTextures.resize(
      plan.preDispatchDependencyRanges.size());
  commands.ownedPreDispatchTextureBindings.resize(
      plan.preDispatchDependencyRanges.size());
  const size_t fixedDrawCount = rangeSlotCount(plan.drawRangesByPass);
  size_t dynamicDrawCount = 0u;
  for (size_t i = 0u; i < plan.orderedPassIndices.size(); ++i) {
    if (plan.drawRangesByPass[i].count == 0u) {
      dynamicDrawCount += passes_[plan.orderedPassIndices[i]].draws.size();
    }
  }
  commands.ownedDrawItems.resize(fixedDrawCount + dynamicDrawCount);
  commands.ownedDrawDebugLabels.resize(commands.ownedDrawItems.size());
  commands.ownedDrawPushConstants.resize(commands.ownedDrawItems.size());
  commands.ownedDrawTextureBindings.resize(commands.ownedDrawItems.size());
  const uint32_t meshDispatchCount =
      rangeSlotCount(plan.meshDispatchRangesByPass);
  commands.ownedMeshDispatchItems.resize(meshDispatchCount);
  commands.ownedMeshDispatchDebugLabels.resize(meshDispatchCount);
  commands.ownedMeshDispatchPushConstants.resize(meshDispatchCount);
  commands.ownedMeshDispatchDependencyBuffers.resize(meshDispatchCount);
  commands.ownedMeshDispatchDependencyTextures.resize(meshDispatchCount);
  commands.ownedMeshDispatchTextureBindings.resize(meshDispatchCount);
  commands.ownedBufferCopyItems.resize(
      rangeSlotCount(plan.bufferCopyRangesByPass));
  commands.ownedTextureCopyItems.resize(
      rangeSlotCount(plan.textureCopyRangesByPass));
  commands.ownedAccelerationStructureBuildsByPass.resize(
      plan.orderedPassIndices.size());
  commands.ownedAccelerationStructureGeometriesByPass.resize(
      plan.orderedPassIndices.size());
  commands.ownedAccelerationStructureInstancesByPass.resize(
      plan.orderedPassIndices.size());
  populateFrameCommands(plan, commands);
  return commands;
}

void RenderGraphBuilder::populateFrameCommands(
    const RenderGraphPlan &plan, FrameCommandArena &commands) const {
  for (size_t i = 0;
       i < textures_.size() && i < commands.textureHandlesByResource.size();
       ++i) {
    if (textures_[i].imported) {
      commands.textureHandlesByResource[i] = textures_[i].importedHandle;
    }
  }
  for (size_t i = 0;
       i < buffers_.size() && i < commands.bufferHandlesByResource.size();
       ++i) {
    if (buffers_[i].imported) {
      commands.bufferHandlesByResource[i] = buffers_[i].importedHandle;
    }
  }
  for (size_t i = 0; i < accelerationStructures_.size() &&
                     i < commands.accelerationStructureHandlesByResource.size();
       ++i) {
    commands.accelerationStructureHandlesByResource[i] =
        accelerationStructures_[i].importedHandle;
  }
  for (size_t i = 0; i < plan.resolvedDependencyBufferResourceIndices.size() &&
                     i < commands.resolvedDependencyBuffers.size();
       ++i) {
    const uint32_t resourceIndex =
        plan.resolvedDependencyBufferResourceIndices[i];
    if (resourceIndex == UINT32_MAX || resourceIndex >= buffers_.size()) {
      continue;
    }
    if (buffers_[resourceIndex].imported) {
      commands.resolvedDependencyBuffers[i] =
          buffers_[resourceIndex].importedHandle;
    }
  }
  for (size_t i = 0; i < plan.resolvedDependencyTextureResourceIndices.size() &&
                     i < commands.resolvedDependencyTextures.size();
       ++i) {
    const uint32_t resourceIndex =
        plan.resolvedDependencyTextureResourceIndices[i];
    if (resourceIndex == UINT32_MAX || resourceIndex >= textures_.size()) {
      continue;
    }
    if (textures_[resourceIndex].imported) {
      commands.resolvedDependencyTextures[i] =
          textures_[resourceIndex].importedHandle;
    }
  }
  for (size_t i = 0;
       i < plan.resolvedPreDispatchDependencyBufferResourceIndices.size() &&
       i < commands.resolvedPreDispatchDependencyBuffers.size();
       ++i) {
    const uint32_t resourceIndex =
        plan.resolvedPreDispatchDependencyBufferResourceIndices[i];
    if (resourceIndex == UINT32_MAX || resourceIndex >= buffers_.size()) {
      continue;
    }
    if (buffers_[resourceIndex].imported) {
      commands.resolvedPreDispatchDependencyBuffers[i] =
          buffers_[resourceIndex].importedHandle;
    }
  }
  const size_t passCount = commands.orderedPasses.size();
  size_t nextDynamicDrawIndex = rangeSlotCount(plan.drawRangesByPass);
  constexpr std::array attachmentBindings{
      std::pair{&RenderPass::colorTexture, &PassBindings::color},
      std::pair{&RenderPass::colorResolveTexture, &PassBindings::colorResolve},
      std::pair{&RenderPass::depthTexture, &PassBindings::depth},
      std::pair{&RenderPass::depthResolveTexture, &PassBindings::depthResolve},
  };
  for (size_t i = 0; i < passCount; ++i) {
    const uint32_t passIndex = plan.orderedPassIndices[i];
    for (const auto [target, binding] : attachmentBindings) {
      const uint32_t resource = passBindings_[passIndex].*binding;
      if (resource != UINT32_MAX && textures_[resource].imported) {
        commands.orderedPasses[i].*target = textures_[resource].importedHandle;
      }
    }
  }
  for (size_t i = 0; i < passCount; ++i) {
    const uint32_t passIndex = plan.orderedPassIndices[i];
    const RenderPass &sourcePass = passes_[passIndex];
    RenderPass &refreshedPass = commands.orderedPasses[i];
    refreshedPass.color = sourcePass.color;
    refreshedPass.executionMode = sourcePass.executionMode;
    refreshedPass.hasColorAttachment = sourcePass.hasColorAttachment;
    refreshedPass.depth = sourcePass.depth;
    refreshedPass.useViewport = sourcePass.useViewport;
    refreshedPass.viewport = sourcePass.viewport;
    refreshedPass.payloadBorrowed = sourcePass.payloadBorrowed;
    refreshedPass.drawBuffersPreResolved = sourcePass.drawBuffersPreResolved;
    refreshedPass.gpuTimingScope = sourcePass.gpuTimingScope;
    refreshedPass.debugLabel = sourcePass.debugLabel;
    refreshedPass.debugColor = sourcePass.debugColor;
    refreshedPass.externalTemporalDispatch =
        sourcePass.externalTemporalDispatch;
    ownAccelerationStructureBuilds(commands, i, sourcePass, refreshedPass);
    const auto dependencyRange = plan.dependencyBufferRangesByPass[i];
    if (dependencyRange.count > 0u &&
        dependencyRange.offset <= commands.resolvedDependencyBuffers.size() &&
        dependencyRange.count <= commands.resolvedDependencyBuffers.size() -
                                     dependencyRange.offset) {
      refreshedPass.dependencyBuffers = std::span<const BufferHandle>(
          commands.resolvedDependencyBuffers.data() + dependencyRange.offset,
          dependencyRange.count);
    } else {
      refreshedPass.dependencyBuffers = {};
    }
    const auto dependencyTextureRange = plan.dependencyTextureRangesByPass[i];
    if (dependencyTextureRange.count > 0u &&
        dependencyTextureRange.offset <=
            commands.resolvedDependencyTextures.size() &&
        dependencyTextureRange.count <=
            commands.resolvedDependencyTextures.size() -
                dependencyTextureRange.offset) {
      refreshedPass.dependencyTextures = std::span<const TextureHandle>(
          commands.resolvedDependencyTextures.data() +
              dependencyTextureRange.offset,
          dependencyTextureRange.count);
    } else {
      refreshedPass.dependencyTextures = {};
    }
    const auto preDispatchRange = plan.preDispatchRangesByPass[i];
    if (preDispatchRange.count > 0u) {
      if (sourcePass.preDispatches.size() == preDispatchRange.count &&
          preDispatchRange.offset <= commands.ownedPreDispatches.size() &&
          preDispatchRange.count <=
              commands.ownedPreDispatches.size() - preDispatchRange.offset) {
        for (uint32_t dispatchIndex = 0; dispatchIndex < preDispatchRange.count;
             ++dispatchIndex) {
          const uint32_t globalDispatchIndex =
              preDispatchRange.offset + dispatchIndex;
          if (globalDispatchIndex >= plan.preDispatchDependencyRanges.size()) {
            break;
          }
          ComputeDispatchItem refreshedDispatch =
              sourcePass.preDispatches[dispatchIndex];
          const auto depRange =
              plan.preDispatchDependencyRanges[globalDispatchIndex];
          if (depRange.count > 0u &&
              depRange.offset <=
                  commands.resolvedPreDispatchDependencyBuffers.size() &&
              depRange.count <=
                  commands.resolvedPreDispatchDependencyBuffers.size() -
                      depRange.offset) {
            refreshedDispatch.dependencyBuffers = std::span<const BufferHandle>(
                commands.resolvedPreDispatchDependencyBuffers.data() +
                    depRange.offset,
                depRange.count);
          } else {
            refreshedDispatch.dependencyBuffers = {};
          }
          refreshedDispatch.dependencyBufferAccessModes = {};
          ownPreDispatchPayload(commands, globalDispatchIndex,
                                sourcePass.preDispatches[dispatchIndex],
                                refreshedDispatch);
          commands.ownedPreDispatches[globalDispatchIndex] = refreshedDispatch;
        }
        refreshedPass.preDispatches = std::span<const ComputeDispatchItem>(
            commands.ownedPreDispatches.data() + preDispatchRange.offset,
            preDispatchRange.count);
      } else {
        refreshedPass.preDispatches = {};
      }
    } else {
      refreshedPass.preDispatches = sourcePass.preDispatches;
    }
    const auto drawRange = plan.drawRangesByPass[i];
    if (drawRange.count > 0u) {
      const uint32_t actualDrawCount =
          static_cast<uint32_t>(sourcePass.draws.size());
      if (actualDrawCount <= drawRange.count &&
          drawRange.offset <= commands.ownedDrawItems.size() &&
          drawRange.count <=
              commands.ownedDrawItems.size() - drawRange.offset) {
        for (uint32_t drawIndex = 0; drawIndex < actualDrawCount; ++drawIndex) {
          const size_t outputIndex = drawRange.offset + drawIndex;
          DrawItem draw = sourcePass.draws[drawIndex];
          ownDrawPayload(commands, outputIndex, sourcePass.draws[drawIndex],
                         draw);
          commands.ownedDrawItems[outputIndex] = draw;
        }
        refreshedPass.draws = std::span<const DrawItem>(
            commands.ownedDrawItems.data() + drawRange.offset, actualDrawCount);
      } else {
        refreshedPass.draws = {};
      }
    } else {
      const uint32_t actualDrawCount =
          static_cast<uint32_t>(sourcePass.draws.size());
      for (uint32_t drawIndex = 0u; drawIndex < actualDrawCount; ++drawIndex) {
        const size_t outputIndex = nextDynamicDrawIndex + drawIndex;
        DrawItem draw = sourcePass.draws[drawIndex];
        ownDrawPayload(commands, outputIndex, sourcePass.draws[drawIndex],
                       draw);
        commands.ownedDrawItems[outputIndex] = draw;
      }
      refreshedPass.draws =
          actualDrawCount == 0u
              ? std::span<const DrawItem>{}
              : std::span<const DrawItem>(commands.ownedDrawItems.data() +
                                              nextDynamicDrawIndex,
                                          actualDrawCount);
      nextDynamicDrawIndex += actualDrawCount;
    }
    const auto meshDispatchRange = plan.meshDispatchRangesByPass[i];
    if (meshDispatchRange.count > 0u) {
      const uint32_t actualDispatchCount =
          static_cast<uint32_t>(sourcePass.meshDispatches.size());
      const uint32_t bindingOffset =
          passBindings_[passIndex].meshDispatches.offset;
      for (uint32_t dispatchIndex = 0; dispatchIndex < actualDispatchCount;
           ++dispatchIndex) {
        MeshDispatchItem dispatch = sourcePass.meshDispatches[dispatchIndex];
        const MeshDispatchBindings binding =
            meshDispatchBindings_[bindingOffset + dispatchIndex];
        const auto resolve = [&](uint32_t resource) {
          return resource != UINT32_MAX && buffers_[resource].imported
                     ? buffers_[resource].importedHandle
                     : BufferHandle{};
        };
        dispatch.indirectBuffer = resolve(binding.indirect);
        dispatch.indirectCountBuffer = resolve(binding.indirectCount);
        const size_t outputIndex = meshDispatchRange.offset + dispatchIndex;
        ownMeshDispatchPayload(commands, outputIndex,
                               sourcePass.meshDispatches[dispatchIndex],
                               dispatch);
        commands.ownedMeshDispatchItems[outputIndex] = dispatch;
      }
      refreshedPass.meshDispatches = std::span<const MeshDispatchItem>(
          commands.ownedMeshDispatchItems.data() + meshDispatchRange.offset,
          actualDispatchCount);
    } else {
      refreshedPass.meshDispatches = sourcePass.meshDispatches;
    }
    const auto bufferCopyRange = plan.bufferCopyRangesByPass[i];
    if (bufferCopyRange.count > 0u) {
      const uint32_t actualCopyCount =
          static_cast<uint32_t>(sourcePass.bufferCopies.size());
      const uint32_t bindingOffset =
          passBindings_[passIndex].bufferCopies.offset;
      for (uint32_t copyIndex = 0; copyIndex < actualCopyCount; ++copyIndex) {
        BufferCopyRegion copy = sourcePass.bufferCopies[copyIndex];
        const BufferCopyBindings binding =
            bufferCopyBindings_[bindingOffset + copyIndex];
        copy.srcBuffer = buffers_[binding.source].importedHandle;
        copy.dstBuffer = buffers_[binding.destination].importedHandle;
        commands.ownedBufferCopyItems[bufferCopyRange.offset + copyIndex] =
            copy;
      }
      refreshedPass.bufferCopies = std::span<const BufferCopyRegion>(
          commands.ownedBufferCopyItems.data() + bufferCopyRange.offset,
          actualCopyCount);
    } else {
      refreshedPass.bufferCopies = sourcePass.bufferCopies;
    }
    const auto textureCopyRange = plan.textureCopyRangesByPass[i];
    if (textureCopyRange.count > 0u) {
      const uint32_t actualCopyCount =
          static_cast<uint32_t>(sourcePass.textureCopies.size());
      const uint32_t bindingOffset =
          passBindings_[passIndex].textureCopies.offset;
      for (uint32_t copyIndex = 0; copyIndex < actualCopyCount; ++copyIndex) {
        TextureCopyItem copy = sourcePass.textureCopies[copyIndex];
        const TextureCopyBindings binding =
            textureCopyBindings_[bindingOffset + copyIndex];
        copy.sourceTexture = textures_[binding.source].importedHandle;
        copy.destinationTexture = textures_[binding.destination].importedHandle;
        commands.ownedTextureCopyItems[textureCopyRange.offset + copyIndex] =
            copy;
      }
      refreshedPass.textureCopies = std::span<const TextureCopyItem>(
          commands.ownedTextureCopyItems.data() + textureCopyRange.offset,
          actualCopyCount);
    } else {
      refreshedPass.textureCopies = sourcePass.textureCopies;
    }
    const std::pmr::string &name = commands.passDebugNames[passIndex];
    refreshedPass.debugLabel = std::string_view(name.data(), name.size());
  }
}

Result<RenderGraphTextureId, std::string>
RenderGraphBuilder::importTexture(TextureHandle texture,
                                  std::string_view debugName) {
  if (!nuri::isValid(texture)) {
    return Result<RenderGraphTextureId, std::string>::makeError(
        "RenderGraphBuilder::importTexture: texture handle is invalid");
  }
  const uint64_t textureKey = foldHandleKey(texture.index, texture.generation);
  if (const auto existing = importedTextureIndicesByHandle_.find(textureKey);
      existing != importedTextureIndicesByHandle_.end()) {
    return Result<RenderGraphTextureId, std::string>::makeResult(
        RenderGraphTextureId{.value = existing->second});
  }
  if (textures_.size() >= UINT32_MAX) {
    return Result<RenderGraphTextureId, std::string>::makeError(
        "RenderGraphBuilder::importTexture: texture count exceeds uint32_t");
  }
  const uint32_t textureIndex = static_cast<uint32_t>(textures_.size());
  TextureResource resource(memory_);
  resource.imported = true;
  resource.importedHandle = texture;
  const std::string_view name =
      resolveResourceDebugName(debugName, "imported_texture");
  resource.debugName.assign(name.data(), name.size());
  textures_.push_back(std::move(resource));
  importedTextureIndicesByHandle_.emplace(textureKey, textureIndex);
  return Result<RenderGraphTextureId, std::string>::makeResult(
      RenderGraphTextureId{.value = textureIndex});
}

Result<RenderGraphBufferId, std::string>
RenderGraphBuilder::importBuffer(BufferHandle buffer,
                                 std::string_view debugName) {
  if (!nuri::isValid(buffer)) {
    return Result<RenderGraphBufferId, std::string>::makeError(
        "RenderGraphBuilder::importBuffer: buffer handle is invalid");
  }
  const uint64_t bufferKey = foldHandleKey(buffer.index, buffer.generation);
  if (const auto existing = importedBufferIndicesByHandle_.find(bufferKey);
      existing != importedBufferIndicesByHandle_.end()) {
    return Result<RenderGraphBufferId, std::string>::makeResult(
        RenderGraphBufferId{.value = existing->second});
  }
  if (buffers_.size() >= UINT32_MAX) {
    return Result<RenderGraphBufferId, std::string>::makeError(
        "RenderGraphBuilder::importBuffer: buffer count exceeds uint32_t");
  }
  const uint32_t bufferIndex = static_cast<uint32_t>(buffers_.size());
  BufferResource resource(memory_);
  resource.imported = true;
  resource.importedHandle = buffer;
  const std::string_view name =
      resolveResourceDebugName(debugName, "imported_buffer");
  resource.debugName.assign(name.data(), name.size());
  buffers_.push_back(std::move(resource));
  importedBufferIndicesByHandle_.emplace(bufferKey, bufferIndex);
  return Result<RenderGraphBufferId, std::string>::makeResult(
      RenderGraphBufferId{.value = bufferIndex});
}

Result<RenderGraphAccelerationStructureId, std::string>
RenderGraphBuilder::importAccelerationStructure(
    AccelerationStructureHandle accelerationStructure,
    std::string_view debugName) {
  if (!nuri::isValid(accelerationStructure)) {
    return Result<RenderGraphAccelerationStructureId, std::string>::makeError(
        "RenderGraphBuilder::importAccelerationStructure: handle is invalid");
  }
  const uint64_t key = foldHandleKey(accelerationStructure.index,
                                     accelerationStructure.generation);
  if (const auto existing =
          importedAccelerationStructureIndicesByHandle_.find(key);
      existing != importedAccelerationStructureIndicesByHandle_.end()) {
    return Result<RenderGraphAccelerationStructureId, std::string>::makeResult(
        RenderGraphAccelerationStructureId{.value = existing->second});
  }
  if (accelerationStructures_.size() >= UINT32_MAX) {
    return Result<RenderGraphAccelerationStructureId, std::string>::makeError(
        "RenderGraphBuilder::importAccelerationStructure: resource count "
        "exceeds uint32_t");
  }
  const uint32_t resourceIndex =
      static_cast<uint32_t>(accelerationStructures_.size());
  AccelerationStructureResource resource(memory_);
  resource.importedHandle = accelerationStructure;
  const std::string_view name =
      resolveResourceDebugName(debugName, "imported_acceleration_structure");
  resource.debugName.assign(name.data(), name.size());
  accelerationStructures_.push_back(std::move(resource));
  importedAccelerationStructureIndicesByHandle_.emplace(key, resourceIndex);
  return Result<RenderGraphAccelerationStructureId, std::string>::makeResult(
      RenderGraphAccelerationStructureId{.value = resourceIndex});
}

Result<RenderGraphTextureId, std::string>
RenderGraphBuilder::createTransientTexture(const TextureDesc &desc,
                                           std::string_view debugName) {
  if (!isValidTransientTextureDesc(desc)) {
    return Result<RenderGraphTextureId, std::string>::makeError(
        "RenderGraphBuilder::createTransientTexture: descriptor is invalid");
  }
  if (textures_.size() >= UINT32_MAX) {
    return Result<RenderGraphTextureId, std::string>::makeError(
        "RenderGraphBuilder::createTransientTexture: texture count exceeds "
        "uint32_t");
  }
  const uint32_t textureIndex = static_cast<uint32_t>(textures_.size());
  TextureResource resource(memory_);
  resource.imported = false;
  resource.transientDesc = desc;
  resource.transientDesc.data = {};
  const std::string_view name =
      resolveResourceDebugName(debugName, "transient_texture");
  resource.debugName.assign(name.data(), name.size());
  appendTransientTextureDescriptorFingerprint(transientResourceDescriptorsHash_,
                                              resource.transientDesc);
  textures_.push_back(std::move(resource));
  return Result<RenderGraphTextureId, std::string>::makeResult(
      RenderGraphTextureId{.value = textureIndex});
}

Result<RenderGraphBufferId, std::string>
RenderGraphBuilder::createTransientBuffer(const BufferDesc &desc,
                                          std::string_view debugName) {
  if (!isValidTransientBufferDesc(desc)) {
    return Result<RenderGraphBufferId, std::string>::makeError(
        "RenderGraphBuilder::createTransientBuffer: descriptor is invalid");
  }
  if (buffers_.size() >= UINT32_MAX) {
    return Result<RenderGraphBufferId, std::string>::makeError(
        "RenderGraphBuilder::createTransientBuffer: buffer count exceeds "
        "uint32_t");
  }
  const uint32_t bufferIndex = static_cast<uint32_t>(buffers_.size());
  BufferResource resource(memory_);
  resource.imported = false;
  resource.transientDesc = desc;
  resource.transientDesc.data = {};
  const std::string_view name =
      resolveResourceDebugName(debugName, "transient_buffer");
  resource.debugName.assign(name.data(), name.size());
  appendTransientBufferDescriptorFingerprint(transientResourceDescriptorsHash_,
                                             resource.transientDesc);
  buffers_.push_back(std::move(resource));
  return Result<RenderGraphBufferId, std::string>::makeResult(
      RenderGraphBufferId{.value = bufferIndex});
}

Result<bool, std::string> RenderGraphBuilder::addTextureAccessInternal(
    RenderGraphPassId pass, RenderGraphTextureId texture,
    RenderGraphAccessMode mode, bool inferred,
    RenderGraphSubresourceRange subresources) {
  if (!isValid(pass) || !isValid(texture)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addTextureAccessInternal: id is invalid");
  }
  if (!hasAccessFlag(mode, RenderGraphAccessMode::Read) &&
      !hasAccessFlag(mode, RenderGraphAccessMode::Write)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addTextureAccessInternal: access mode must "
        "contain "
        "read or write");
  }
  if (!isValidPassIndex(pass.value) || !isValidTextureIndex(texture.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addTextureAccessInternal: id is out of range");
  }
  if (subresources.mipCount == 0u || subresources.layerCount == 0u) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addTextureAccessInternal: subresource range is "
        "empty");
  }
  const auto existing = std::ranges::find_if(
      resourceUses_, [&](const RenderGraphResourceUse &use) {
        return use.passIndex == pass.value &&
               use.resourceKind == RenderGraphResourceKind::Texture &&
               use.resourceIndex == texture.value &&
               use.subresources == subresources &&
               use.provenance ==
                   (inferred ? RenderGraphResourceUseProvenance::Inferred
                             : RenderGraphResourceUseProvenance::Explicit);
      });
  if (existing != resourceUses_.end()) {
    RenderGraphResourceUse &merged = *existing;
    merged.access = merged.access | mode;
    return Result<bool, std::string>::makeResult(true);
  }
  const uint32_t accessIndex = static_cast<uint32_t>(resourceUses_.size());
  if (accessIndex == UINT32_MAX) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addTextureAccessInternal: access count exceeds "
        "uint32_t");
  }
  resourceUses_.push_back(RenderGraphResourceUse{
      .passIndex = pass.value,
      .resourceKind = RenderGraphResourceKind::Texture,
      .resourceIndex = texture.value,
      .access = mode,
      .stage = passes_[pass.value].executionMode,
      .subresources = subresources,
      .provenance = inferred ? RenderGraphResourceUseProvenance::Inferred
                             : RenderGraphResourceUseProvenance::Explicit,
  });
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RenderGraphBuilder::addTextureAccess(
    RenderGraphPassId pass, RenderGraphTextureId texture,
    RenderGraphAccessMode mode, RenderGraphSubresourceRange subresources) {
  return addTextureAccessInternal(pass, texture, mode, false, subresources);
}

Result<bool, std::string> RenderGraphBuilder::addBufferAccessInternal(
    RenderGraphPassId pass, RenderGraphBufferId buffer,
    RenderGraphAccessMode mode, bool inferred,
    RenderGraphResourceState requestedState) {
  if (!isValid(pass) || !isValid(buffer)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addBufferAccessInternal: id is invalid");
  }
  if (!hasAccessFlag(mode, RenderGraphAccessMode::Read) &&
      !hasAccessFlag(mode, RenderGraphAccessMode::Write)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addBufferAccessInternal: access mode must contain "
        "read or write");
  }
  if (!isValidPassIndex(pass.value) || !isValidBufferIndex(buffer.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addBufferAccessInternal: id is out of range");
  }
  const uint64_t key = foldPassResourceKey(pass.value, buffer.value);
  auto &indexByKey = inferred ? inferredBufferAccessIndicesByPassResource_
                              : explicitBufferAccessIndicesByPassResource_;
  if (const auto existing = indexByKey.find(key);
      existing != indexByKey.end()) {
    RenderGraphResourceUse &merged = resourceUses_[existing->second];
    if (merged.state != RenderGraphResourceState::Unknown &&
        requestedState != RenderGraphResourceState::Unknown &&
        merged.state != requestedState) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::addBufferAccessInternal: contradictory "
          "resource states");
    }
    merged.access = merged.access | mode;
    if (merged.state == RenderGraphResourceState::Unknown) {
      merged.state = requestedState;
    }
    return Result<bool, std::string>::makeResult(true);
  }
  const uint32_t accessIndex = static_cast<uint32_t>(resourceUses_.size());
  if (accessIndex == UINT32_MAX) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addBufferAccessInternal: access count exceeds "
        "uint32_t");
  }
  resourceUses_.push_back(RenderGraphResourceUse{
      .passIndex = pass.value,
      .resourceKind = RenderGraphResourceKind::Buffer,
      .resourceIndex = buffer.value,
      .access = mode,
      .state = requestedState,
      .stage = passes_[pass.value].executionMode,
      .provenance = inferred ? RenderGraphResourceUseProvenance::Inferred
                             : RenderGraphResourceUseProvenance::Explicit,
  });
  indexByKey.emplace(key, accessIndex);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RenderGraphBuilder::addBufferAccess(RenderGraphPassId pass,
                                    RenderGraphBufferId buffer,
                                    RenderGraphAccessMode mode) {
  return addBufferAccessInternal(pass, buffer, mode, false);
}

Result<bool, std::string>
RenderGraphBuilder::addAccelerationStructureAccessInternal(
    RenderGraphPassId pass,
    RenderGraphAccelerationStructureId accelerationStructure,
    RenderGraphAccelerationStructureAccess access, bool inferred) {
  if (!isValid(pass) || !isValid(accelerationStructure)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addAccelerationStructureAccessInternal: id is "
        "invalid");
  }
  if (!isValidPassIndex(pass.value) ||
      !isValidAccelerationStructureIndex(accelerationStructure.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addAccelerationStructureAccessInternal: id is "
        "out of range");
  }
  const RenderGraphAccessMode mode =
      access == RenderGraphAccelerationStructureAccess::BuildWrite
          ? RenderGraphAccessMode::Write
          : RenderGraphAccessMode::Read;
  const RenderGraphResourceState state = [&]() {
    switch (access) {
    case RenderGraphAccelerationStructureAccess::BuildRead:
      return RenderGraphResourceState::AccelerationStructureBuildRead;
    case RenderGraphAccelerationStructureAccess::BuildWrite:
      return RenderGraphResourceState::AccelerationStructureBuildWrite;
    case RenderGraphAccelerationStructureAccess::RayQueryRead:
      return RenderGraphResourceState::RayQueryRead;
    }
    return RenderGraphResourceState::Unknown;
  }();
  const uint64_t key =
      foldPassResourceKey(pass.value, accelerationStructure.value);
  auto &indexByKey =
      inferred ? inferredAccelerationStructureAccessIndicesByPassResource_
               : explicitAccelerationStructureAccessIndicesByPassResource_;
  if (const auto existing = indexByKey.find(key);
      existing != indexByKey.end()) {
    RenderGraphResourceUse &merged = resourceUses_[existing->second];
    if (merged.state != state) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::addAccelerationStructureAccessInternal: "
          "contradictory access for the same pass and resource");
    }
    merged.access = merged.access | mode;
    return Result<bool, std::string>::makeResult(true);
  }
  if (resourceUses_.size() >= UINT32_MAX) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addAccelerationStructureAccessInternal: access "
        "count exceeds uint32_t");
  }
  const uint32_t accessIndex = static_cast<uint32_t>(resourceUses_.size());
  resourceUses_.push_back(RenderGraphResourceUse{
      .passIndex = pass.value,
      .resourceKind = RenderGraphResourceKind::AccelerationStructure,
      .resourceIndex = accelerationStructure.value,
      .access = mode,
      .state = state,
      .stage = passes_[pass.value].executionMode,
      .provenance = inferred ? RenderGraphResourceUseProvenance::Inferred
                             : RenderGraphResourceUseProvenance::Explicit,
  });
  indexByKey.emplace(key, accessIndex);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RenderGraphBuilder::addAccelerationStructureAccess(
    RenderGraphPassId pass,
    RenderGraphAccelerationStructureId accelerationStructure,
    RenderGraphAccelerationStructureAccess access) {
  return addAccelerationStructureAccessInternal(pass, accelerationStructure,
                                                access, false);
}

Result<bool, std::string>
RenderGraphBuilder::addTextureRead(RenderGraphPassId pass,
                                   RenderGraphTextureId texture) {
  return addTextureAccess(pass, texture, RenderGraphAccessMode::Read);
}

Result<bool, std::string>
RenderGraphBuilder::addTextureWrite(RenderGraphPassId pass,
                                    RenderGraphTextureId texture) {
  return addTextureAccess(pass, texture, RenderGraphAccessMode::Write);
}

Result<bool, std::string>
RenderGraphBuilder::addBufferRead(RenderGraphPassId pass,
                                  RenderGraphBufferId buffer) {
  return addBufferAccess(pass, buffer, RenderGraphAccessMode::Read);
}

Result<bool, std::string>
RenderGraphBuilder::addBufferWrite(RenderGraphPassId pass,
                                   RenderGraphBufferId buffer) {
  return addBufferAccess(pass, buffer, RenderGraphAccessMode::Write);
}

Result<bool, std::string> RenderGraphBuilder::addPreResolvedDrawBufferAccesses(
    RenderGraphPassId pass, std::span<const BufferHandle> buffers,
    std::string_view debugLabel) {
  const std::string drawDebugName =
      makePassResourceDebugName(debugLabel, "draw_buffer");
  for (const BufferHandle buffer : buffers) {
    if (!nuri::isValid(buffer)) {
      continue;
    }
    auto importResult = importBuffer(buffer, drawDebugName);
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    auto accessResult = addBufferAccess(pass, importResult.value(),
                                        RenderGraphAccessMode::Read);
    if (accessResult.hasError()) {
      return accessResult;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RenderGraphBuilder::addPreResolvedDrawBufferAccesses(
    RenderGraphPassId pass, std::span<const RenderGraphBufferId> buffers) {
  for (const RenderGraphBufferId buffer : buffers) {
    if (!isValid(buffer)) {
      continue;
    }
    auto accessResult =
        addBufferAccess(pass, buffer, RenderGraphAccessMode::Read);
    if (accessResult.hasError()) {
      return accessResult;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

RenderGraphBuilder::OwnedPassPayload RenderGraphBuilder::clonePassPayload(
    const RenderGraphGraphicsPassDesc &desc) const {
  OwnedPassPayload ownedPayload(memory_);
  ownedPayload.debugLabel.assign(desc.debugLabel.data(),
                                 desc.debugLabel.size());
  ownedPayload.dependencyBuffers.assign(desc.dependencyBuffers.begin(),
                                        desc.dependencyBuffers.end());
  ownedPayload.dependencyTextures.assign(desc.dependencyTextures.begin(),
                                         desc.dependencyTextures.end());
  ownedPayload.preDispatchDebugLabels.reserve(desc.preDispatches.size());
  ownedPayload.preDispatchPushConstants.reserve(desc.preDispatches.size());
  ownedPayload.preDispatchDependencyBuffers.reserve(desc.preDispatches.size());
  ownedPayload.preDispatchDependencyBufferAccessModes.reserve(
      desc.preDispatches.size());
  ownedPayload.preDispatchDependencyTextures.reserve(desc.preDispatches.size());
  ownedPayload.preDispatchTextureBindings.reserve(desc.preDispatches.size());
  ownedPayload.preDispatches.reserve(desc.preDispatches.size());
  for (const ComputeDispatchItem &sourceDispatch : desc.preDispatches) {
    ownedPayload.preDispatchDebugLabels.push_back(std::pmr::string(memory_));
    auto &label = ownedPayload.preDispatchDebugLabels.back();
    label.assign(sourceDispatch.debugLabel.data(),
                 sourceDispatch.debugLabel.size());
    ownedPayload.preDispatchPushConstants.push_back(
        std::pmr::vector<std::byte>(memory_));
    auto &pushConstants = ownedPayload.preDispatchPushConstants.back();
    pushConstants.assign(sourceDispatch.pushConstants.begin(),
                         sourceDispatch.pushConstants.end());
    ownedPayload.preDispatchDependencyBuffers.push_back(
        std::pmr::vector<BufferHandle>(memory_));
    auto &dependencyBuffers = ownedPayload.preDispatchDependencyBuffers.back();
    dependencyBuffers.assign(sourceDispatch.dependencyBuffers.begin(),
                             sourceDispatch.dependencyBuffers.end());
    ownedPayload.preDispatchDependencyBufferAccessModes.push_back(
        std::pmr::vector<RenderGraphAccessMode>(memory_));
    auto &dependencyBufferAccessModes =
        ownedPayload.preDispatchDependencyBufferAccessModes.back();
    dependencyBufferAccessModes.assign(
        sourceDispatch.dependencyBufferAccessModes.begin(),
        sourceDispatch.dependencyBufferAccessModes.end());
    ownedPayload.preDispatchDependencyTextures.push_back(
        std::pmr::vector<TextureHandle>(memory_));
    auto &dependencyTextures =
        ownedPayload.preDispatchDependencyTextures.back();
    dependencyTextures.assign(sourceDispatch.dependencyTextures.begin(),
                              sourceDispatch.dependencyTextures.end());
    ownedPayload.preDispatchTextureBindings.push_back(
        std::pmr::vector<PushConstantTextureBinding>(memory_));
    ownedPayload.preDispatchTextureBindings.back().assign(
        sourceDispatch.pushConstantTextureBindings.begin(),
        sourceDispatch.pushConstantTextureBindings.end());
  }
  ownedPayload.preDispatches.resize(desc.preDispatches.size());
  for (size_t i = 0; i < desc.preDispatches.size(); ++i) {
    const ComputeDispatchItem &sourceDispatch = desc.preDispatches[i];
    ComputeDispatchItem &dispatch = ownedPayload.preDispatches[i];
    dispatch = sourceDispatch;
    dispatch.pushConstants = std::span<const std::byte>(
        ownedPayload.preDispatchPushConstants[i].data(),
        ownedPayload.preDispatchPushConstants[i].size());
    dispatch.dependencyBuffers = std::span<const BufferHandle>(
        ownedPayload.preDispatchDependencyBuffers[i].data(),
        ownedPayload.preDispatchDependencyBuffers[i].size());
    dispatch.dependencyBufferAccessModes =
        std::span<const RenderGraphAccessMode>(
            ownedPayload.preDispatchDependencyBufferAccessModes[i].data(),
            ownedPayload.preDispatchDependencyBufferAccessModes[i].size());
    dispatch.dependencyTextures = std::span<const TextureHandle>(
        ownedPayload.preDispatchDependencyTextures[i].data(),
        ownedPayload.preDispatchDependencyTextures[i].size());
    dispatch.pushConstantTextureBindings =
        ownedPayload.preDispatchTextureBindings[i];
    dispatch.debugLabel =
        std::string_view(ownedPayload.preDispatchDebugLabels[i].data(),
                         ownedPayload.preDispatchDebugLabels[i].size());
  }
  ownedPayload.drawDebugLabels.reserve(desc.draws.size());
  ownedPayload.drawPushConstants.reserve(desc.draws.size());
  ownedPayload.drawTextureBindings.reserve(desc.draws.size());
  ownedPayload.draws.reserve(desc.draws.size());
  for (const DrawItem &sourceDraw : desc.draws) {
    ownedPayload.drawDebugLabels.push_back(std::pmr::string(memory_));
    auto &label = ownedPayload.drawDebugLabels.back();
    label.assign(sourceDraw.debugLabel.data(), sourceDraw.debugLabel.size());
    ownedPayload.drawPushConstants.push_back(
        std::pmr::vector<std::byte>(memory_));
    auto &pushConstants = ownedPayload.drawPushConstants.back();
    pushConstants.assign(sourceDraw.pushConstants.begin(),
                         sourceDraw.pushConstants.end());
    ownedPayload.drawTextureBindings.push_back(
        std::pmr::vector<PushConstantTextureBinding>(memory_));
    ownedPayload.drawTextureBindings.back().assign(
        sourceDraw.pushConstantTextureBindings.begin(),
        sourceDraw.pushConstantTextureBindings.end());
  }
  ownedPayload.draws.resize(desc.draws.size());
  for (size_t i = 0; i < desc.draws.size(); ++i) {
    const DrawItem &sourceDraw = desc.draws[i];
    DrawItem &draw = ownedPayload.draws[i];
    draw = sourceDraw;
    draw.pushConstants =
        std::span<const std::byte>(ownedPayload.drawPushConstants[i].data(),
                                   ownedPayload.drawPushConstants[i].size());
    draw.pushConstantTextureBindings = ownedPayload.drawTextureBindings[i];
    draw.debugLabel = std::string_view(ownedPayload.drawDebugLabels[i].data(),
                                       ownedPayload.drawDebugLabels[i].size());
  }
  ownedPayload.meshDispatchDebugLabels.reserve(desc.meshDispatches.size());
  ownedPayload.meshDispatchPushConstants.reserve(desc.meshDispatches.size());
  ownedPayload.meshDispatchDependencyBuffers.reserve(
      desc.meshDispatches.size());
  ownedPayload.meshDispatchDependencyTextures.reserve(
      desc.meshDispatches.size());
  ownedPayload.meshDispatchTextureBindings.reserve(desc.meshDispatches.size());
  ownedPayload.meshDispatches.reserve(desc.meshDispatches.size());
  for (const MeshDispatchItem &sourceDispatch : desc.meshDispatches) {
    ownedPayload.meshDispatchDebugLabels.push_back(std::pmr::string(memory_));
    auto &label = ownedPayload.meshDispatchDebugLabels.back();
    label.assign(sourceDispatch.debugLabel.data(),
                 sourceDispatch.debugLabel.size());
    ownedPayload.meshDispatchPushConstants.push_back(
        std::pmr::vector<std::byte>(memory_));
    auto &pushConstants = ownedPayload.meshDispatchPushConstants.back();
    pushConstants.assign(sourceDispatch.pushConstants.begin(),
                         sourceDispatch.pushConstants.end());
    ownedPayload.meshDispatchDependencyBuffers.push_back(
        std::pmr::vector<BufferHandle>(memory_));
    auto &dependencyBuffers = ownedPayload.meshDispatchDependencyBuffers.back();
    dependencyBuffers.assign(sourceDispatch.dependencyBuffers.begin(),
                             sourceDispatch.dependencyBuffers.end());
    ownedPayload.meshDispatchDependencyTextures.push_back(
        std::pmr::vector<TextureHandle>(memory_));
    auto &dependencyTextures =
        ownedPayload.meshDispatchDependencyTextures.back();
    dependencyTextures.assign(sourceDispatch.dependencyTextures.begin(),
                              sourceDispatch.dependencyTextures.end());
    ownedPayload.meshDispatchTextureBindings.push_back(
        std::pmr::vector<PushConstantTextureBinding>(memory_));
    ownedPayload.meshDispatchTextureBindings.back().assign(
        sourceDispatch.pushConstantTextureBindings.begin(),
        sourceDispatch.pushConstantTextureBindings.end());
  }
  ownedPayload.meshDispatches.resize(desc.meshDispatches.size());
  for (size_t i = 0; i < desc.meshDispatches.size(); ++i) {
    const MeshDispatchItem &sourceDispatch = desc.meshDispatches[i];
    MeshDispatchItem &dispatch = ownedPayload.meshDispatches[i];
    dispatch = sourceDispatch;
    dispatch.pushConstants = std::span<const std::byte>(
        ownedPayload.meshDispatchPushConstants[i].data(),
        ownedPayload.meshDispatchPushConstants[i].size());
    dispatch.dependencyBuffers = std::span<const BufferHandle>(
        ownedPayload.meshDispatchDependencyBuffers[i].data(),
        ownedPayload.meshDispatchDependencyBuffers[i].size());
    dispatch.dependencyTextures = std::span<const TextureHandle>(
        ownedPayload.meshDispatchDependencyTextures[i].data(),
        ownedPayload.meshDispatchDependencyTextures[i].size());
    dispatch.pushConstantTextureBindings =
        ownedPayload.meshDispatchTextureBindings[i];
    dispatch.debugLabel =
        std::string_view(ownedPayload.meshDispatchDebugLabels[i].data(),
                         ownedPayload.meshDispatchDebugLabels[i].size());
  }
  return ownedPayload;
}

Result<bool, std::string> RenderGraphBuilder::applyGraphicsPassRoots(
    RenderGraphPassId pass, RenderGraphTextureId colorTexture,
    bool markColorAsFrameOutput, bool markImplicitOutputSideEffect) {
  if (nuri::isValid(colorTexture)) {
    if (markColorAsFrameOutput) {
      return markTextureAsFrameOutput(colorTexture);
    }
    if (markImplicitOutputSideEffect &&
        textures_[colorTexture.value].imported) {
      return markPassSideEffect(pass);
    }
    return Result<bool, std::string>::makeResult(true);
  }
  if (!markImplicitOutputSideEffect) {
    return Result<bool, std::string>::makeResult(true);
  }
  return markPassSideEffect(pass);
}

Result<bool, std::string> RenderGraphBuilder::applyImplicitPassRoots(
    RenderGraphPassId pass, const RenderGraphGraphicsPassDesc &desc) {
  if (isComputeOnlyExecutionMode(desc.executionMode)) {
    return desc.markImplicitOutputSideEffect
               ? markPassSideEffect(pass)
               : Result<bool, std::string>::makeResult(true);
  }
  if (!desc.hasColorAttachment) {
    if (nuri::isValid(desc.colorTexture)) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::applyImplicitPassRoots: no-color pass has a "
          "color texture");
    }
    if (nuri::isValid(desc.colorResolveTexture)) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::applyImplicitPassRoots: no-color pass has a "
          "color resolve texture");
    }
    return Result<bool, std::string>::makeResult(true);
  }
  return applyGraphicsPassRoots(pass, graphicsPassRootColorTexture(desc),
                                desc.markColorAsFrameOutput,
                                desc.markImplicitOutputSideEffect);
}

Result<bool, std::string> RenderGraphBuilder::bindImplicitPassResources(
    RenderGraphPassId pass, const RenderGraphGraphicsPassDesc &desc) {
  if (!desc.dependencyBufferAccessModes.empty() &&
      desc.dependencyBufferAccessModes.size() !=
          desc.dependencyBuffers.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindImplicitPassResources: dependency buffer "
        "access mode count does not match dependency buffer count");
  }
  if (!desc.dependencyTextureAccessModes.empty() &&
      desc.dependencyTextureAccessModes.size() !=
          desc.dependencyTextures.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindImplicitPassResources: dependency texture "
        "access mode count does not match dependency texture count");
  }
  const auto bindPushConstantTextures =
      [this, pass](const auto &commands) -> Result<bool, std::string> {
    for (const auto &command : commands) {
      for (const PushConstantTextureBinding binding :
           command.pushConstantTextureBindings) {
        if (binding.byteOffset == UINT32_MAX ||
            binding.byteOffset % alignof(uint32_t) != 0u ||
            binding.byteOffset > command.pushConstants.size() ||
            sizeof(uint32_t) >
                command.pushConstants.size() - binding.byteOffset) {
          return Result<bool, std::string>::makeError(
              "RenderGraphBuilder::bindImplicitPassResources: push-constant "
              "texture binding offset is invalid");
        }
        if (binding.graphTextureResourceIndex == UINT32_MAX) {
          if (!nuri::isValid(binding.texture)) {
            return Result<bool, std::string>::makeError(
                "RenderGraphBuilder::bindImplicitPassResources: "
                "push-constant texture binding is unresolved");
          }
          continue;
        }
        const RenderGraphTextureId texture{
            .value = binding.graphTextureResourceIndex};
        if (binding.access == RenderGraphAccessMode::None) {
          return Result<bool, std::string>::makeError(
              "RenderGraphBuilder::bindImplicitPassResources: push-constant "
              "texture binding access is empty");
        }
        auto accessResult = addTextureAccess(pass, texture, binding.access);
        if (accessResult.hasError()) {
          return accessResult;
        }
      }
    }
    return Result<bool, std::string>::makeResult(true);
  };
  auto dispatchTextureResult = bindPushConstantTextures(desc.preDispatches);
  if (dispatchTextureResult.hasError()) {
    return dispatchTextureResult;
  }
  auto drawTextureResult = bindPushConstantTextures(desc.draws);
  if (drawTextureResult.hasError()) {
    return drawTextureResult;
  }
  auto meshTextureResult = bindPushConstantTextures(desc.meshDispatches);
  if (meshTextureResult.hasError()) {
    return meshTextureResult;
  }
  if (nuri::isValid(desc.colorTexture)) {
    if (!desc.hasColorAttachment) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::bindImplicitPassResources: no-color pass has a "
          "color texture");
    }
    auto bindResult = bindPassColorTexture(pass, desc.colorTexture);
    if (bindResult.hasError()) {
      return bindResult;
    }
  }
  if (nuri::isValid(desc.colorResolveTexture)) {
    if (!desc.hasColorAttachment) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::bindImplicitPassResources: no-color pass has a "
          "color resolve texture");
    }
    auto bindResult =
        bindPassColorResolveTexture(pass, desc.colorResolveTexture);
    if (bindResult.hasError()) {
      return bindResult;
    }
  }
  if (nuri::isValid(desc.depthTexture)) {
    auto bindResult = bindPassDepthTexture(pass, desc.depthTexture);
    if (bindResult.hasError()) {
      return bindResult;
    }
  }
  if (nuri::isValid(desc.depthResolveTexture)) {
    auto bindResult =
        bindPassDepthResolveTexture(pass, desc.depthResolveTexture);
    if (bindResult.hasError()) {
      return bindResult;
    }
  }
  const std::string dependencyDebugName =
      makePassResourceDebugName(desc.debugLabel, "dependency_buffer");
  for (size_t i = 0; i < desc.dependencyBuffers.size(); ++i) {
    const BufferHandle dependency = desc.dependencyBuffers[i];
    if (!nuri::isValid(dependency)) {
      continue;
    }
    const RenderGraphAccessMode accessMode =
        desc.dependencyBufferAccessModes.empty()
            ? (RenderGraphAccessMode::Read | RenderGraphAccessMode::Write)
            : desc.dependencyBufferAccessModes[i];
    if (!hasAccessFlag(accessMode, RenderGraphAccessMode::Read) &&
        !hasAccessFlag(accessMode, RenderGraphAccessMode::Write)) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::bindImplicitPassResources: dependency buffer "
          "access mode must contain read or write");
    }
    auto importResult = importBuffer(dependency, dependencyDebugName);
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    auto bindResult = bindPassDependencyBuffer(
        pass, static_cast<uint32_t>(i), importResult.value(), accessMode);
    if (bindResult.hasError()) {
      return bindResult;
    }
  }
  const std::string dependencyTextureDebugName =
      makePassResourceDebugName(desc.debugLabel, "dependency_texture");
  for (size_t i = 0; i < desc.dependencyTextures.size(); ++i) {
    const TextureHandle dependency = desc.dependencyTextures[i];
    if (!nuri::isValid(dependency)) {
      continue;
    }
    const RenderGraphAccessMode accessMode =
        desc.dependencyTextureAccessModes.empty()
            ? RenderGraphAccessMode::Read
            : desc.dependencyTextureAccessModes[i];
    if (!hasAccessFlag(accessMode, RenderGraphAccessMode::Read) &&
        !hasAccessFlag(accessMode, RenderGraphAccessMode::Write)) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::bindImplicitPassResources: dependency texture "
          "access mode must contain read or write");
    }
    auto importResult = importTexture(dependency, dependencyTextureDebugName);
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    auto bindResult = bindPassDependencyTexture(
        pass, static_cast<uint32_t>(i), importResult.value(), accessMode);
    if (bindResult.hasError()) {
      return bindResult;
    }
  }
  const std::string preDispatchDependencyDebugName = makePassResourceDebugName(
      desc.debugLabel, "pre_dispatch_dependency_buffer");
  for (size_t dispatchIndex = 0; dispatchIndex < desc.preDispatches.size();
       ++dispatchIndex) {
    const ComputeDispatchItem &dispatch = desc.preDispatches[dispatchIndex];
    if (!dispatch.dependencyBufferAccessModes.empty() &&
        dispatch.dependencyBufferAccessModes.size() !=
            dispatch.dependencyBuffers.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::bindImplicitPassResources: pre-dispatch "
          "dependency buffer access mode count does not match dependency "
          "buffer count");
    }
    for (size_t dependencyIndex = 0;
         dependencyIndex < dispatch.dependencyBuffers.size();
         ++dependencyIndex) {
      const BufferHandle dependency =
          dispatch.dependencyBuffers[dependencyIndex];
      if (!nuri::isValid(dependency)) {
        continue;
      }
      auto importResult =
          importBuffer(dependency, preDispatchDependencyDebugName);
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }
      auto bindResult = bindPreDispatchDependencyBuffer(
          pass, static_cast<uint32_t>(dispatchIndex),
          static_cast<uint32_t>(dependencyIndex), importResult.value(),
          dispatch.dependencyBufferAccessModes.empty()
              ? (RenderGraphAccessMode::Read | RenderGraphAccessMode::Write)
              : dispatch.dependencyBufferAccessModes[dependencyIndex]);
      if (bindResult.hasError()) {
        return bindResult;
      }
    }
  }
  const std::string meshDispatchIndirectDebugName = makePassResourceDebugName(
      desc.debugLabel, "mesh_dispatch_indirect_buffer");
  for (size_t dispatchIndex = 0u; dispatchIndex < desc.meshDispatches.size();
       ++dispatchIndex) {
    const MeshDispatchItem &dispatch = desc.meshDispatches[dispatchIndex];
    if (nuri::isValid(dispatch.indirectBuffer)) {
      auto importResult =
          importBuffer(dispatch.indirectBuffer, meshDispatchIndirectDebugName);
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }
      auto bindResult = bindMeshDispatchBuffer(
          pass, static_cast<uint32_t>(dispatchIndex),
          RenderGraphMeshDispatchBufferBindingTarget::Indirect,
          importResult.value(), RenderGraphAccessMode::Read);
      if (bindResult.hasError()) {
        return bindResult;
      }
    }
    if (nuri::isValid(dispatch.indirectCountBuffer)) {
      auto importResult = importBuffer(dispatch.indirectCountBuffer,
                                       meshDispatchIndirectDebugName);
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }
      auto bindResult = bindMeshDispatchBuffer(
          pass, static_cast<uint32_t>(dispatchIndex),
          RenderGraphMeshDispatchBufferBindingTarget::IndirectCount,
          importResult.value(), RenderGraphAccessMode::Read);
      if (bindResult.hasError()) {
        return bindResult;
      }
    }
  }
  for (const auto &binding : desc.meshDispatchBufferBindings) {
    auto result =
        bindMeshDispatchBuffer(pass, binding.meshDispatchIndex, binding.target,
                               binding.buffer, binding.mode);
    if (result.hasError()) {
      return result;
    }
  }
  if (desc.drawBuffersPreResolved || !desc.preResolvedDrawBufferIds.empty() ||
      !desc.preResolvedDrawBuffers.empty()) {
    auto accessResult =
        !desc.preResolvedDrawBufferIds.empty()
            ? addPreResolvedDrawBufferAccesses(pass,
                                               desc.preResolvedDrawBufferIds)
            : addPreResolvedDrawBufferAccesses(
                  pass, desc.preResolvedDrawBuffers, desc.debugLabel);
    if (accessResult.hasError()) {
      return accessResult;
    }
  } else {
    const std::string drawDebugName =
        makePassResourceDebugName(desc.debugLabel, "draw_buffer");
    for (size_t drawIndex = 0; drawIndex < desc.draws.size(); ++drawIndex) {
      const DrawItem &draw = desc.draws[drawIndex];
      const std::array<
          std::pair<BufferHandle, RenderGraphDrawBufferBindingTarget>, 4>
          bindings = {{
              {draw.vertexBuffer, RenderGraphDrawBufferBindingTarget::Vertex},
              {draw.indexBuffer, RenderGraphDrawBufferBindingTarget::Index},
              {draw.indirectBuffer,
               RenderGraphDrawBufferBindingTarget::Indirect},
              {draw.indirectCountBuffer,
               RenderGraphDrawBufferBindingTarget::IndirectCount},
          }};
      for (const auto &[buffer, target] : bindings) {
        if (!nuri::isValid(buffer)) {
          continue;
        }
        auto importResult = importBuffer(buffer, drawDebugName);
        if (importResult.hasError()) {
          return Result<bool, std::string>::makeError(importResult.error());
        }
        auto bindResult =
            bindDrawBuffer(pass, static_cast<uint32_t>(drawIndex), target,
                           importResult.value(), RenderGraphAccessMode::Read);
        if (bindResult.hasError()) {
          return bindResult;
        }
      }
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<RenderGraphPassId, std::string>
RenderGraphBuilder::addGraphicsPass(const RenderGraphGraphicsPassDesc &desc) {
  auto executionModeResult =
      validatePassExecutionMode(desc, "RenderGraphBuilder::addGraphicsPass");
  if (executionModeResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(
        executionModeResult.error());
  }
  const uint32_t bufferDependencyAuthorities = !desc.dependencyBuffers.empty() +
                                               !desc.bufferUses.empty() +
                                               !desc.importedBufferUses.empty();
  const uint32_t textureDependencyAuthorities =
      !desc.dependencyTextures.empty() + !desc.textureUses.empty() +
      !desc.importedTextureUses.empty();
  if (bufferDependencyAuthorities > 1u || textureDependencyAuthorities > 1u) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addGraphicsPass: graph resource uses and "
        "physical dependencies cannot describe the same resource kind");
  }
  std::pmr::vector<BufferHandle> graphDependencyBuffers(memory_);
  graphDependencyBuffers.reserve(desc.bufferUses.size() +
                                 desc.importedBufferUses.size());
  for (const RenderGraphBufferUse use : desc.bufferUses) {
    if (!isValidBufferIndex(use.buffer.value)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addGraphicsPass: buffer use is out of range");
    }
    graphDependencyBuffers.push_back(
        buffers_[use.buffer.value].imported
            ? buffers_[use.buffer.value].importedHandle
            : BufferHandle{});
  }
  for (const RenderGraphImportedBufferUse use : desc.importedBufferUses) {
    if (!nuri::isValid(use.buffer)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addGraphicsPass: imported buffer use is "
          "invalid");
    }
    graphDependencyBuffers.push_back(use.buffer);
  }
  std::pmr::vector<TextureHandle> graphDependencyTextures(memory_);
  graphDependencyTextures.reserve(desc.textureUses.size() +
                                  desc.importedTextureUses.size());
  for (const RenderGraphTextureUse use : desc.textureUses) {
    if (!isValidTextureIndex(use.texture.value)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addGraphicsPass: texture use is out of range");
    }
    graphDependencyTextures.push_back(
        textures_[use.texture.value].imported
            ? textures_[use.texture.value].importedHandle
            : TextureHandle{});
  }
  for (const RenderGraphImportedTextureUse use : desc.importedTextureUses) {
    if (!nuri::isValid(use.texture)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addGraphicsPass: imported texture use is "
          "invalid");
    }
    graphDependencyTextures.push_back(use.texture);
  }
  RenderGraphGraphicsPassDesc payloadDesc = desc;
  if (!graphDependencyBuffers.empty()) {
    payloadDesc.dependencyBuffers = graphDependencyBuffers;
  }
  if (!graphDependencyTextures.empty()) {
    payloadDesc.dependencyTextures = graphDependencyTextures;
  }
  const bool borrowPayload = desc.borrowPayload &&
                             graphDependencyBuffers.empty() &&
                             graphDependencyTextures.empty();
  if (!borrowPayload) {
    allPassesBorrowPayload_ = false;
  }
  RenderPass pass{};
  pass.executionMode = desc.executionMode;
  pass.color = desc.color;
  pass.colorResolveTexture = {};
  pass.hasColorAttachment = desc.hasColorAttachment;
  pass.depth = desc.depth;
  pass.depthResolveTexture = {};
  pass.useViewport = desc.useViewport;
  pass.viewport = desc.viewport;
  pass.gpuTimingScope = desc.gpuTimingScope;
  pass.externalTemporalDispatch = desc.externalTemporalDispatch;
  pass.debugColor = desc.debugColor;
  pass.payloadBorrowed = borrowPayload;
  pass.drawBuffersPreResolved = desc.drawBuffersPreResolved;
  if (borrowPayload) {
    pass.preDispatches = payloadDesc.preDispatches;
    pass.dependencyBuffers = payloadDesc.dependencyBuffers;
    pass.dependencyTextures = payloadDesc.dependencyTextures;
    pass.draws = payloadDesc.draws;
    pass.meshDispatches = payloadDesc.meshDispatches;
    pass.debugLabel = payloadDesc.debugLabel;
  } else {
    OwnedPassPayload ownedPayload = clonePassPayload(payloadDesc);
    ownedPassPayloads_.push_back(std::move(ownedPayload));
    OwnedPassPayload &storedPayload = ownedPassPayloads_.back();
    pass.preDispatches = std::span<const ComputeDispatchItem>(
        storedPayload.preDispatches.data(), storedPayload.preDispatches.size());
    pass.dependencyBuffers =
        std::span<const BufferHandle>(storedPayload.dependencyBuffers.data(),
                                      storedPayload.dependencyBuffers.size());
    pass.dependencyTextures =
        std::span<const TextureHandle>(storedPayload.dependencyTextures.data(),
                                       storedPayload.dependencyTextures.size());
    pass.draws = std::span<const DrawItem>(storedPayload.draws.data(),
                                           storedPayload.draws.size());
    pass.meshDispatches =
        std::span<const MeshDispatchItem>(storedPayload.meshDispatches.data(),
                                          storedPayload.meshDispatches.size());
    pass.debugLabel = std::string_view(storedPayload.debugLabel.data(),
                                       storedPayload.debugLabel.size());
  }
  auto addResult = addPassRecord(pass, desc.debugLabel);
  if (addResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();
  auto bindResourcesResult = bindImplicitPassResources(passId, desc);
  if (bindResourcesResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(
        bindResourcesResult.error());
  }
  for (uint32_t i = 0u; i < desc.bufferUses.size(); ++i) {
    const RenderGraphBufferUse use = desc.bufferUses[i];
    auto bindResult =
        bindPassDependencyBuffer(passId, i, use.buffer, use.access);
    if (bindResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          bindResult.error());
    }
  }
  for (uint32_t i = 0u; i < desc.textureUses.size(); ++i) {
    const RenderGraphTextureUse use = desc.textureUses[i];
    auto bindResult =
        bindPassDependencyTexture(passId, i, use.texture, use.access);
    if (bindResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          bindResult.error());
    }
  }
  const std::string bufferUseDebugName =
      makePassResourceDebugName(desc.debugLabel, "imported_buffer_use");
  for (uint32_t i = 0u; i < desc.importedBufferUses.size(); ++i) {
    const RenderGraphImportedBufferUse use = desc.importedBufferUses[i];
    auto imported = importBuffer(use.buffer, bufferUseDebugName);
    if (imported.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          imported.error());
    }
    auto bindResult =
        bindPassDependencyBuffer(passId, i, imported.value(), use.access);
    if (bindResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          bindResult.error());
    }
  }
  const std::string textureUseDebugName =
      makePassResourceDebugName(desc.debugLabel, "imported_texture_use");
  for (uint32_t i = 0u; i < desc.importedTextureUses.size(); ++i) {
    const RenderGraphImportedTextureUse use = desc.importedTextureUses[i];
    auto imported = importTexture(use.texture, textureUseDebugName);
    if (imported.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          imported.error());
    }
    auto bindResult =
        bindPassDependencyTexture(passId, i, imported.value(), use.access);
    if (bindResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          bindResult.error());
    }
  }
  auto rootResult = applyImplicitPassRoots(passId, desc);
  if (rootResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(
        rootResult.error());
  }
  return Result<RenderGraphPassId, std::string>::makeResult(passId);
}

Result<RenderGraphPassId, std::string>
RenderGraphBuilder::addAccelerationStructurePass(
    const RenderGraphAccelerationStructurePassDesc &desc) {
  if (desc.builds.empty()) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addAccelerationStructurePass: build list is "
        "empty");
  }
  const auto declaredBufferAccess = [&](BufferHandle handle) {
    for (const RenderGraphBufferUse use : desc.buffers) {
      if (isValid(use.buffer) && isValidBufferIndex(use.buffer.value) &&
          buffers_[use.buffer.value].imported &&
          buffers_[use.buffer.value].importedHandle == handle) {
        return use.access;
      }
    }
    return RenderGraphAccessMode::None;
  };
  const auto hasDeclaredAccelerationStructureAccess =
      [&](AccelerationStructureHandle handle,
          RenderGraphAccelerationStructureAccess expected) {
        for (const RenderGraphAccelerationStructureUse use :
             desc.accelerationStructures) {
          if (use.access == expected && isValid(use.accelerationStructure) &&
              isValidAccelerationStructureIndex(
                  use.accelerationStructure.value) &&
              accelerationStructures_[use.accelerationStructure.value]
                      .importedHandle == handle) {
            return true;
          }
        }
        return false;
      };
  for (const RenderGraphBufferUse use : desc.buffers) {
    if (!isValid(use.buffer) || !isValidBufferIndex(use.buffer.value)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addAccelerationStructurePass: buffer use is "
          "invalid");
    }
    if (use.access != RenderGraphAccessMode::Read) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addAccelerationStructurePass: build-input "
          "buffers must declare read-only access");
    }
  }
  for (const RenderGraphAccelerationStructureUse use :
       desc.accelerationStructures) {
    if (!isValid(use.accelerationStructure) ||
        !isValidAccelerationStructureIndex(use.accelerationStructure.value)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addAccelerationStructurePass: acceleration "
          "structure use is invalid");
    }
    if (use.access == RenderGraphAccelerationStructureAccess::RayQueryRead) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addAccelerationStructurePass: ray-query read "
          "is not legal in a build pass");
    }
  }
  std::string dependencyError;
  for (const AccelerationStructureBuildItem &build : desc.builds) {
    std::visit(
        [&](const auto &command) {
          if (!dependencyError.empty()) {
            return;
          }
          if (!hasDeclaredAccelerationStructureAccess(
                  command.destination,
                  RenderGraphAccelerationStructureAccess::BuildWrite)) {
            dependencyError =
                "destination is missing an AS BuildWrite declaration";
            return;
          }
          using Command = std::decay_t<decltype(command)>;
          if constexpr (std::is_same_v<Command, BuildBlasItem> ||
                        std::is_same_v<Command, UpdateBlasItem>) {
            if (command.geometries.empty()) {
              dependencyError = "BLAS geometry list is empty";
              return;
            }
            for (const auto &geometry : command.geometries) {
              const std::array requiredBuffers{
                  geometry.vertexBuffer,
                  geometry.indexBuffer,
                  geometry.transformBuffer,
              };
              for (const BufferHandle buffer : requiredBuffers) {
                if (!nuri::isValid(buffer)) {
                  continue;
                }
                if (declaredBufferAccess(buffer) !=
                    RenderGraphAccessMode::Read) {
                  dependencyError =
                      "BLAS input is missing a read-only buffer declaration";
                  return;
                }
              }
            }
          } else {
            if (command.instances.empty()) {
              dependencyError = "TLAS instance list is empty";
              return;
            }
            for (const auto &instance : command.instances) {
              if (!hasDeclaredAccelerationStructureAccess(
                      instance.bottomLevel,
                      RenderGraphAccelerationStructureAccess::BuildRead)) {
                dependencyError =
                    "TLAS instance BLAS is missing an AS BuildRead declaration";
                return;
              }
            }
          }
        },
        build.command);
    if (!dependencyError.empty()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addAccelerationStructurePass: " +
          dependencyError);
    }
  }

  OwnedPassPayload ownedPayload(memory_);
  ownedPayload.debugLabel.assign(desc.debugLabel.data(),
                                 desc.debugLabel.size());
  ownedPayload.accelerationStructureBuilds.reserve(desc.builds.size());
  ownedPayload.accelerationStructureGeometries.reserve(desc.builds.size());
  ownedPayload.accelerationStructureInstances.reserve(desc.builds.size());
  for (const AccelerationStructureBuildItem &source : desc.builds) {
    ownedPayload.accelerationStructureGeometries.push_back(
        std::pmr::vector<AccelerationStructureTriangleGeometryDesc>(memory_));
    ownedPayload.accelerationStructureInstances.push_back(
        std::pmr::vector<AccelerationStructureInstanceDesc>(memory_));
    auto &geometries = ownedPayload.accelerationStructureGeometries.back();
    auto &instances = ownedPayload.accelerationStructureInstances.back();
    std::visit(
        [&](const auto &command) {
          using Command = std::decay_t<decltype(command)>;
          if constexpr (std::is_same_v<Command, BuildBlasItem>) {
            geometries.assign(command.geometries.begin(),
                              command.geometries.end());
            ownedPayload.accelerationStructureBuilds.push_back(
                AccelerationStructureBuildItem{
                    .command = BuildBlasItem{
                        .destination = command.destination,
                        .geometries = geometries,
                    }});
          } else if constexpr (std::is_same_v<Command, UpdateBlasItem>) {
            geometries.assign(command.geometries.begin(),
                              command.geometries.end());
            ownedPayload.accelerationStructureBuilds.push_back(
                AccelerationStructureBuildItem{
                    .command = UpdateBlasItem{
                        .destination = command.destination,
                        .geometries = geometries,
                    }});
          } else if constexpr (std::is_same_v<Command, BuildTlasItem>) {
            instances.assign(command.instances.begin(),
                             command.instances.end());
            ownedPayload.accelerationStructureBuilds.push_back(
                AccelerationStructureBuildItem{
                    .command = BuildTlasItem{
                        .destination = command.destination,
                        .instances = instances,
                    }});
          } else {
            instances.assign(command.instances.begin(),
                             command.instances.end());
            ownedPayload.accelerationStructureBuilds.push_back(
                AccelerationStructureBuildItem{
                    .command = UpdateTlasItem{
                        .destination = command.destination,
                        .instances = instances,
                    }});
          }
        },
        source.command);
  }
  allPassesBorrowPayload_ = false;
  ownedPassPayloads_.push_back(std::move(ownedPayload));
  OwnedPassPayload &storedPayload = ownedPassPayloads_.back();
  RenderPass pass{};
  pass.executionMode = RenderPassExecutionMode::AccelerationStructureBuild;
  pass.hasColorAttachment = false;
  pass.payloadBorrowed = false;
  pass.accelerationStructureBuilds = storedPayload.accelerationStructureBuilds;
  pass.gpuTimingScope = desc.gpuTimingScope;
  pass.debugLabel = std::string_view(storedPayload.debugLabel.data(),
                                     storedPayload.debugLabel.size());
  pass.debugColor = desc.debugColor;
  auto addResult = addPassRecord(pass, desc.debugLabel);
  if (addResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();
  for (const RenderGraphBufferUse use : desc.buffers) {
    auto accessResult = addBufferAccessInternal(
        passId, use.buffer, use.access, false,
        RenderGraphResourceState::AccelerationStructureBuildInput);
    if (accessResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          accessResult.error());
    }
  }
  for (const RenderGraphAccelerationStructureUse use :
       desc.accelerationStructures) {
    auto accessResult = addAccelerationStructureAccess(
        passId, use.accelerationStructure, use.access);
    if (accessResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          accessResult.error());
    }
  }
  if (desc.markImplicitOutputSideEffect) {
    auto sideEffectResult = markPassSideEffect(passId);
    if (sideEffectResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          sideEffectResult.error());
    }
  }
  return Result<RenderGraphPassId, std::string>::makeResult(passId);
}

Result<RenderGraphPassId, std::string> RenderGraphBuilder::addTextureCopyPass(
    const RenderGraphTextureCopyPassDesc &desc) {
  if (desc.copies.empty()) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addTextureCopyPass: copy list is empty");
  }
  OwnedPassPayload ownedPayload(memory_);
  ownedPayload.debugLabel.assign(desc.debugLabel.data(),
                                 desc.debugLabel.size());
  ownedPayload.textureCopies.reserve(desc.copies.size());
  bool hasImportedDestination = false;
  for (const RenderGraphTextureCopyItem &copy : desc.copies) {
    if (!isValid(copy.sourceTexture) || !isValid(copy.destinationTexture)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addTextureCopyPass: copy texture id is "
          "invalid");
    }
    if (!isValidTextureIndex(copy.sourceTexture.value) ||
        !isValidTextureIndex(copy.destinationTexture.value)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addTextureCopyPass: copy texture id is out of "
          "range");
    }
    if (copy.sourceTexture.value == copy.destinationTexture.value) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addTextureCopyPass: source and destination "
          "textures must differ");
    }
    if (copy.width == 0u || copy.height == 0u) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addTextureCopyPass: copy region is empty");
    }
    hasImportedDestination = hasImportedDestination ||
                             textures_[copy.destinationTexture.value].imported;
    ownedPayload.textureCopies.push_back(TextureCopyItem{
        .sourceTexture = {},
        .destinationTexture = {},
        .sourceX = copy.sourceX,
        .sourceY = copy.sourceY,
        .destinationX = copy.destinationX,
        .destinationY = copy.destinationY,
        .width = copy.width,
        .height = copy.height,
        .sourceMipLevel = copy.sourceMipLevel,
        .destinationMipLevel = copy.destinationMipLevel,
        .sourceLayer = copy.sourceLayer,
        .destinationLayer = copy.destinationLayer,
    });
  }
  allPassesBorrowPayload_ = false;
  ownedPassPayloads_.push_back(std::move(ownedPayload));
  OwnedPassPayload &storedPayload = ownedPassPayloads_.back();
  RenderPass pass{};
  pass.executionMode = RenderPassExecutionMode::CopyOnly;
  pass.hasColorAttachment = false;
  pass.payloadBorrowed = false;
  pass.textureCopies = std::span<const TextureCopyItem>(
      storedPayload.textureCopies.data(), storedPayload.textureCopies.size());
  pass.gpuTimingScope = desc.gpuTimingScope;
  pass.debugColor = desc.debugColor;
  pass.debugLabel = std::string_view(storedPayload.debugLabel.data(),
                                     storedPayload.debugLabel.size());
  auto addResult = addPassRecord(pass, desc.debugLabel);
  if (addResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();
  const uint32_t bindingOffset =
      passBindings_[passId.value].textureCopies.offset;
  for (uint32_t copyIndex = 0u; copyIndex < desc.copies.size(); ++copyIndex) {
    const RenderGraphTextureCopyItem &copy = desc.copies[copyIndex];
    const uint32_t bindingIndex = bindingOffset + copyIndex;
    textureCopyBindings_[bindingIndex].source = copy.sourceTexture.value;
    textureCopyBindings_[bindingIndex].destination =
        copy.destinationTexture.value;
    auto readResult = addTextureAccess(passId, copy.sourceTexture,
                                       RenderGraphAccessMode::Read,
                                       {.firstMip = copy.sourceMipLevel,
                                        .mipCount = 1u,
                                        .firstLayer = copy.sourceLayer,
                                        .layerCount = 1u});
    if (readResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          readResult.error());
    }
    auto writeResult = addTextureAccess(passId, copy.destinationTexture,
                                        RenderGraphAccessMode::Write,
                                        {.firstMip = copy.destinationMipLevel,
                                         .mipCount = 1u,
                                         .firstLayer = copy.destinationLayer,
                                         .layerCount = 1u});
    if (writeResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          writeResult.error());
    }
  }
  if (desc.markImplicitOutputSideEffect && hasImportedDestination) {
    auto sideEffectResult = markPassSideEffect(passId);
    if (sideEffectResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          sideEffectResult.error());
    }
  }
  return Result<RenderGraphPassId, std::string>::makeResult(passId);
}

Result<RenderGraphPassId, std::string> RenderGraphBuilder::addBufferCopyPass(
    const RenderGraphBufferCopyPassDesc &desc) {
  if (desc.copies.empty()) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addBufferCopyPass: copy list is empty");
  }
  OwnedPassPayload ownedPayload(memory_);
  ownedPayload.debugLabel.assign(desc.debugLabel.data(),
                                 desc.debugLabel.size());
  ownedPayload.bufferCopies.reserve(desc.copies.size());
  bool hasImportedDestination = false;
  for (const RenderGraphBufferCopyItem &copy : desc.copies) {
    if (!isValid(copy.sourceBuffer) || !isValid(copy.destinationBuffer)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addBufferCopyPass: copy buffer id is invalid");
    }
    if (!isValidBufferIndex(copy.sourceBuffer.value) ||
        !isValidBufferIndex(copy.destinationBuffer.value)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addBufferCopyPass: copy buffer id is out of "
          "range");
    }
    if (copy.sourceBuffer.value == copy.destinationBuffer.value) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addBufferCopyPass: source and destination "
          "buffers must differ");
    }
    if (copy.size == 0u) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addBufferCopyPass: copy region is empty");
    }
    hasImportedDestination = hasImportedDestination ||
                             buffers_[copy.destinationBuffer.value].imported;
    ownedPayload.bufferCopies.push_back(BufferCopyRegion{
        .srcBuffer = {},
        .dstBuffer = {},
        .srcOffset = copy.sourceOffset,
        .dstOffset = copy.destinationOffset,
        .size = copy.size,
    });
  }
  allPassesBorrowPayload_ = false;
  ownedPassPayloads_.push_back(std::move(ownedPayload));
  OwnedPassPayload &storedPayload = ownedPassPayloads_.back();
  RenderPass pass{};
  pass.executionMode = RenderPassExecutionMode::CopyOnly;
  pass.hasColorAttachment = false;
  pass.payloadBorrowed = false;
  pass.bufferCopies = std::span<const BufferCopyRegion>(
      storedPayload.bufferCopies.data(), storedPayload.bufferCopies.size());
  pass.gpuTimingScope = desc.gpuTimingScope;
  pass.debugColor = desc.debugColor;
  pass.debugLabel = std::string_view(storedPayload.debugLabel.data(),
                                     storedPayload.debugLabel.size());
  auto addResult = addPassRecord(pass, desc.debugLabel);
  if (addResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();
  const uint32_t bindingOffset =
      passBindings_[passId.value].bufferCopies.offset;
  for (uint32_t copyIndex = 0u; copyIndex < desc.copies.size(); ++copyIndex) {
    const RenderGraphBufferCopyItem &copy = desc.copies[copyIndex];
    BufferCopyBindings &binding =
        bufferCopyBindings_[bindingOffset + copyIndex];
    binding.source = copy.sourceBuffer.value;
    binding.destination = copy.destinationBuffer.value;
    auto readResult =
        addBufferAccess(passId, copy.sourceBuffer, RenderGraphAccessMode::Read);
    if (readResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          readResult.error());
    }
    auto writeResult = addBufferAccess(passId, copy.destinationBuffer,
                                       RenderGraphAccessMode::Write);
    if (writeResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          writeResult.error());
    }
  }
  if (desc.markImplicitOutputSideEffect && hasImportedDestination) {
    auto sideEffectResult = markPassSideEffect(passId);
    if (sideEffectResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          sideEffectResult.error());
    }
  }
  return Result<RenderGraphPassId, std::string>::makeResult(passId);
}

Result<RenderGraphPassId, std::string>
RenderGraphBuilder::addPassRecord(RenderPass pass, std::string_view debugName) {
  const RenderGraphPassId passId{.value =
                                     static_cast<uint32_t>(passes_.size())};
  const uint32_t dependencyOffset =
      static_cast<uint32_t>(passDependencyBufferBindingResourceIndices_.size());
  const uint32_t dependencyTextureOffset = static_cast<uint32_t>(
      passDependencyTextureBindingResourceIndices_.size());
  const uint32_t preDispatchOffset =
      static_cast<uint32_t>(preDispatchDependencyBindings_.size());
  const uint32_t drawOffset = static_cast<uint32_t>(drawBindings_.size());
  const uint32_t meshDispatchOffset =
      static_cast<uint32_t>(meshDispatchBindings_.size());
  const uint32_t bufferCopyOffset =
      static_cast<uint32_t>(bufferCopyBindings_.size());
  const uint32_t textureCopyOffset =
      static_cast<uint32_t>(textureCopyBindings_.size());
  const uint32_t drawCount = pass.drawBuffersPreResolved
                                 ? 0u
                                 : static_cast<uint32_t>(pass.draws.size());
  passBindings_.push_back(PassBindings{
      .dependencyBuffers = {.offset = dependencyOffset,
                            .count = static_cast<uint32_t>(
                                pass.dependencyBuffers.size())},
      .dependencyTextures = {.offset = dependencyTextureOffset,
                             .count = static_cast<uint32_t>(
                                 pass.dependencyTextures.size())},
      .preDispatches = {.offset = preDispatchOffset,
                        .count =
                            static_cast<uint32_t>(pass.preDispatches.size())},
      .draws = {.offset = drawOffset, .count = drawCount},
      .meshDispatches = {.offset = meshDispatchOffset,
                         .count =
                             static_cast<uint32_t>(pass.meshDispatches.size())},
      .bufferCopies = {.offset = bufferCopyOffset,
                       .count =
                           static_cast<uint32_t>(pass.bufferCopies.size())},
      .textureCopies = {.offset = textureCopyOffset,
                        .count =
                            static_cast<uint32_t>(pass.textureCopies.size())},
  });
  passDependencyBufferBindingResourceIndices_.resize(
      dependencyOffset + pass.dependencyBuffers.size(), UINT32_MAX);
  passDependencyTextureBindingResourceIndices_.resize(
      dependencyTextureOffset + pass.dependencyTextures.size(), UINT32_MAX);
  for (const ComputeDispatchItem &dispatch : pass.preDispatches) {
    const uint32_t offset = static_cast<uint32_t>(
        preDispatchDependencyBindingResourceIndices_.size());
    preDispatchDependencyBindings_.push_back(
        {.offset = offset,
         .count = static_cast<uint32_t>(dispatch.dependencyBuffers.size())});
    preDispatchDependencyBindingResourceIndices_.resize(
        offset + dispatch.dependencyBuffers.size(), UINT32_MAX);
  }
  drawBindings_.resize(drawOffset + drawCount);
  meshDispatchBindings_.resize(meshDispatchOffset + pass.meshDispatches.size());
  bufferCopyBindings_.resize(bufferCopyOffset + pass.bufferCopies.size());
  textureCopyBindings_.resize(textureCopyOffset + pass.textureCopies.size());
  passes_.push_back(pass);
  const std::string_view name = debugName.empty() ? pass.debugLabel : debugName;
  passDebugNames_.emplace_back(name);
  return Result<RenderGraphPassId, std::string>::makeResult(passId);
}
Result<bool, std::string>
RenderGraphBuilder::bindPassColorTexture(RenderGraphPassId pass,
                                         RenderGraphTextureId texture) {
  if (!isValidPassIndex(pass.value) || !isValidTextureIndex(texture.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassColorTexture: id is out of range");
  }
  if (!passes_[pass.value].hasColorAttachment) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassColorTexture: no-color pass cannot bind "
        "a color texture");
  }
  passBindings_[pass.value].color = texture.value;
  const RenderPass &targetPass = passes_[pass.value];
  const RenderGraphAccessMode mode =
      attachmentAccessMode(targetPass.color.loadOp);
  if (mode == RenderGraphAccessMode::None) {
    return Result<bool, std::string>::makeResult(true);
  }
  return addTextureAccess(pass, texture, mode);
}

Result<bool, std::string>
RenderGraphBuilder::bindPassColorResolveTexture(RenderGraphPassId pass,
                                                RenderGraphTextureId texture) {
  if (!isValidPassIndex(pass.value) || !isValidTextureIndex(texture.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassColorResolveTexture: id is out of range");
  }
  if (!passes_[pass.value].hasColorAttachment) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassColorResolveTexture: no-color pass cannot "
        "bind a color resolve texture");
  }
  passBindings_[pass.value].colorResolve = texture.value;
  auto accessResult =
      addTextureAccess(pass, texture, RenderGraphAccessMode::Write);
  if (accessResult.hasError()) {
    return accessResult;
  }
  const TextureResource &resource = textures_[texture.value];
  passes_[pass.value].colorResolveTexture =
      resource.imported
          ? resource.importedHandle
          : TextureHandle{.index = texture.value, .generation = 1u};
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RenderGraphBuilder::bindPassDepthTexture(RenderGraphPassId pass,
                                         RenderGraphTextureId texture) {
  if (!isValidPassIndex(pass.value) || !isValidTextureIndex(texture.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDepthTexture: id is out of range");
  }
  passBindings_[pass.value].depth = texture.value;
  const RenderPass &targetPass = passes_[pass.value];
  const RenderGraphAccessMode mode =
      attachmentAccessMode(targetPass.depth.loadOp);
  if (mode == RenderGraphAccessMode::None) {
    return Result<bool, std::string>::makeResult(true);
  }
  return addTextureAccess(pass, texture, mode);
}

Result<bool, std::string>
RenderGraphBuilder::bindPassDepthResolveTexture(RenderGraphPassId pass,
                                                RenderGraphTextureId texture) {
  if (!isValidPassIndex(pass.value) || !isValidTextureIndex(texture.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDepthResolveTexture: id is out of range");
  }
  passBindings_[pass.value].depthResolve = texture.value;
  auto accessResult =
      addTextureAccess(pass, texture, RenderGraphAccessMode::Write);
  if (accessResult.hasError()) {
    return accessResult;
  }
  const TextureResource &resource = textures_[texture.value];
  passes_[pass.value].depthResolveTexture =
      resource.imported
          ? resource.importedHandle
          : TextureHandle{.index = texture.value, .generation = 1u};
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RenderGraphBuilder::bindPassDependencyBuffer(
    RenderGraphPassId pass, uint32_t dependencyIndex,
    RenderGraphBufferId buffer, RenderGraphAccessMode mode) {
  if (!isValidPassIndex(pass.value) || !isValidBufferIndex(buffer.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyBuffer: id is out of range");
  }
  const BindingRange range = passBindings_[pass.value].dependencyBuffers;
  if (dependencyIndex >= range.count) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyBuffer: dependency index is "
        "out of range");
  }
  passDependencyBufferBindingResourceIndices_[range.offset + dependencyIndex] =
      buffer.value;
  return addBufferAccess(pass, buffer, mode);
}

Result<bool, std::string> RenderGraphBuilder::bindPassDependencyTexture(
    RenderGraphPassId pass, uint32_t dependencyIndex,
    RenderGraphTextureId texture, RenderGraphAccessMode mode) {
  if (!isValidPassIndex(pass.value) || !isValidTextureIndex(texture.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyTexture: id is out of range");
  }
  const BindingRange range = passBindings_[pass.value].dependencyTextures;
  if (dependencyIndex >= range.count) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyTexture: dependency index is "
        "out of range");
  }
  passDependencyTextureBindingResourceIndices_[range.offset + dependencyIndex] =
      texture.value;
  return addTextureAccess(pass, texture, mode);
}

Result<bool, std::string> RenderGraphBuilder::bindPreDispatchDependencyBuffer(
    RenderGraphPassId pass, uint32_t preDispatchIndex, uint32_t dependencyIndex,
    RenderGraphBufferId buffer, RenderGraphAccessMode mode) {
  if (!isValidPassIndex(pass.value) || !isValidBufferIndex(buffer.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: id is out of "
        "range");
  }
  const BindingRange dispatches = passBindings_[pass.value].preDispatches;
  if (preDispatchIndex >= dispatches.count) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: pre-dispatch "
        "index is out of range");
  }
  const BindingRange dependencies =
      preDispatchDependencyBindings_[dispatches.offset + preDispatchIndex];
  if (dependencyIndex >= dependencies.count) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: dependency "
        "index is out of range");
  }
  preDispatchDependencyBindingResourceIndices_[dependencies.offset +
                                               dependencyIndex] = buffer.value;
  return addBufferAccess(pass, buffer, mode);
}

Result<bool, std::string>
RenderGraphBuilder::bindDrawBuffer(RenderGraphPassId pass, uint32_t drawIndex,
                                   RenderGraphDrawBufferBindingTarget target,
                                   RenderGraphBufferId buffer,
                                   RenderGraphAccessMode mode) {
  if (!isValidPassIndex(pass.value) || !isValidBufferIndex(buffer.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindDrawBuffer: id is out of range");
  }
  const BindingRange draws = passBindings_[pass.value].draws;
  if (drawIndex >= draws.count) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindDrawBuffer: draw index is out of range");
  }
  static constexpr std::array targets{
      &DrawBindings::vertex, &DrawBindings::index, &DrawBindings::indirect,
      &DrawBindings::indirectCount};
  drawBindings_[draws.offset + drawIndex].*
      targets[static_cast<size_t>(target)] = buffer.value;
  return addBufferAccess(pass, buffer, mode);
}

Result<bool, std::string> RenderGraphBuilder::bindMeshDispatchBuffer(
    RenderGraphPassId pass, uint32_t meshDispatchIndex,
    RenderGraphMeshDispatchBufferBindingTarget target,
    RenderGraphBufferId buffer, RenderGraphAccessMode mode) {
  if (!isValidPassIndex(pass.value) || !isValidBufferIndex(buffer.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindMeshDispatchBuffer: id is out of range");
  }
  const BindingRange dispatches = passBindings_[pass.value].meshDispatches;
  if (meshDispatchIndex >= dispatches.count) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindMeshDispatchBuffer: mesh dispatch index is "
        "out of range");
  }
  static constexpr std::array targets{&MeshDispatchBindings::indirect,
                                      &MeshDispatchBindings::indirectCount};
  meshDispatchBindings_[dispatches.offset + meshDispatchIndex].*
      targets[static_cast<size_t>(target)] = buffer.value;
  return addBufferAccess(pass, buffer, mode);
}

Result<bool, std::string>
RenderGraphBuilder::addDependency(RenderGraphPassId before,
                                  RenderGraphPassId after) {
  if (!isValid(before) || !isValid(after)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addDependency: pass id is invalid");
  }
  if (before.value == after.value) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addDependency: self-dependency is not allowed");
  }
  if (!isValidPassIndex(before.value) || !isValidPassIndex(after.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addDependency: pass index is out of range");
  }
  const uint64_t key = foldDependencyEdgeKey(before.value, after.value);
  if (!dependencyEdgeKeys_.insert(key).second) {
    return Result<bool, std::string>::makeResult(true);
  }
  dependencies_.push_back(
      DependencyEdge{.before = before.value, .after = after.value});
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RenderGraphBuilder::markPassSideEffect(RenderGraphPassId pass) {
  return markPassSideEffectInternal(pass, false);
}

Result<bool, std::string>
RenderGraphBuilder::markPassSideEffectInternal(RenderGraphPassId pass,
                                               bool inferred) {
  if (!isValid(pass) || !isValidPassIndex(pass.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::markPassSideEffectInternal: pass id is invalid");
  }
  if (const auto existing = sideEffectMarkIndicesByPass_.find(pass.value);
      existing != sideEffectMarkIndicesByPass_.end()) {
    SideEffectPassMark &mark = sideEffectPassMarks_[existing->second];
    if (!inferred) {
      mark.inferred = false;
    }
    return Result<bool, std::string>::makeResult(true);
  }
  const uint32_t markIndex = static_cast<uint32_t>(sideEffectPassMarks_.size());
  if (markIndex == UINT32_MAX) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::markPassSideEffectInternal: mark count exceeds "
        "uint32_t");
  }
  sideEffectPassMarks_.push_back(SideEffectPassMark{
      .passIndex = pass.value,
      .inferred = inferred,
  });
  sideEffectMarkIndicesByPass_.emplace(pass.value, markIndex);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RenderGraphBuilder::markTextureAsFrameOutput(RenderGraphTextureId texture) {
  if (!isValid(texture) || !isValidTextureIndex(texture.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::markTextureAsFrameOutput: texture id is invalid");
  }
  if (!frameOutputTextureSet_.insert(texture.value).second) {
    return Result<bool, std::string>::makeResult(true);
  }
  frameOutputTextureIndices_.push_back(texture.value);
  return Result<bool, std::string>::makeResult(true);
}

void RenderGraphBuilder::compileStageC0BuildResourceTables(
    CompiledRenderGraph &compiled,
    RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  compiled.commands.textureHandlesByResource.resize(textures_.size());
  for (uint32_t i = 0; i < textures_.size(); ++i) {
    const TextureResource &texture = textures_[i];
    if (texture.imported) {
      compiled.commands.textureHandlesByResource[i] = texture.importedHandle;
      ++compiled.plan.resourceStats.importedTextures;
    } else {
      ++compiled.plan.resourceStats.transientTextures;
    }
  }
  compiled.commands.bufferHandlesByResource.resize(buffers_.size());
  for (uint32_t i = 0; i < buffers_.size(); ++i) {
    const BufferResource &buffer = buffers_[i];
    if (buffer.imported) {
      compiled.commands.bufferHandlesByResource[i] = buffer.importedHandle;
      ++compiled.plan.resourceStats.importedBuffers;
    } else {
      ++compiled.plan.resourceStats.transientBuffers;
    }
  }
  compiled.commands.accelerationStructureHandlesByResource.resize(
      accelerationStructures_.size());
  for (uint32_t i = 0; i < accelerationStructures_.size(); ++i) {
    compiled.commands.accelerationStructureHandlesByResource[i] =
        accelerationStructures_[i].importedHandle;
    ++compiled.plan.resourceStats.importedAccelerationStructures;
  }
  work.passCount = static_cast<uint32_t>(passes_.size());
  work.activePassCount = work.passCount;
  compiled.plan.declaredPassCount = work.passCount;
}

Result<bool, std::string> RenderGraphBuilder::compileStageC1C2BuildTopology(
    RenderGraphRuntime &runtime, CompiledRenderGraph &compiled,
    RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_BARRIER);
  work.compiledResourceUses = resourceUses_;
  std::sort(
      work.compiledResourceUses.begin(), work.compiledResourceUses.end(),
      [](const RenderGraphResourceUse &lhs, const RenderGraphResourceUse &rhs) {
        const uint8_t lhsKind = static_cast<uint8_t>(lhs.resourceKind);
        const uint8_t rhsKind = static_cast<uint8_t>(rhs.resourceKind);
        if (lhsKind != rhsKind) {
          return lhsKind < rhsKind;
        }
        if (lhs.resourceIndex != rhs.resourceIndex) {
          return lhs.resourceIndex < rhs.resourceIndex;
        }
        return std::tie(lhs.passIndex, lhs.subresources.firstMip,
                        lhs.subresources.mipCount, lhs.subresources.firstLayer,
                        lhs.subresources.layerCount) <
               std::tie(rhs.passIndex, rhs.subresources.firstMip,
                        rhs.subresources.mipCount, rhs.subresources.firstLayer,
                        rhs.subresources.layerCount);
      });
  if (!work.compiledResourceUses.empty()) {
    size_t writeIndex = 0u;
    size_t groupBegin = 0u;
    while (groupBegin < work.compiledResourceUses.size()) {
      size_t groupEnd = groupBegin + 1u;
      while (groupEnd < work.compiledResourceUses.size() &&
             work.compiledResourceUses[groupEnd].resourceKind ==
                 work.compiledResourceUses[groupBegin].resourceKind &&
             work.compiledResourceUses[groupEnd].resourceIndex ==
                 work.compiledResourceUses[groupBegin].resourceIndex &&
             work.compiledResourceUses[groupEnd].passIndex ==
                 work.compiledResourceUses[groupBegin].passIndex &&
             work.compiledResourceUses[groupEnd].subresources ==
                 work.compiledResourceUses[groupBegin].subresources) {
        ++groupEnd;
      }
      RenderGraphAccessMode explicitMode = RenderGraphAccessMode::None;
      RenderGraphAccessMode inferredMode = RenderGraphAccessMode::None;
      RenderGraphResourceState explicitState =
          RenderGraphResourceState::Unknown;
      RenderGraphResourceState inferredState =
          RenderGraphResourceState::Unknown;
      for (size_t i = groupBegin; i < groupEnd; ++i) {
        if (work.compiledResourceUses[i].provenance ==
            RenderGraphResourceUseProvenance::Inferred) {
          inferredMode = inferredMode | work.compiledResourceUses[i].access;
          inferredState = work.compiledResourceUses[i].state;
          continue;
        }
        explicitMode = explicitMode | work.compiledResourceUses[i].access;
        explicitState = work.compiledResourceUses[i].state;
      }
      const bool hasExplicit =
          hasAccessFlag(explicitMode, RenderGraphAccessMode::Read) ||
          hasAccessFlag(explicitMode, RenderGraphAccessMode::Write);
      const RenderGraphAccessMode selectedMode =
          hasExplicit ? explicitMode : inferredMode;
      if (hasAccessFlag(selectedMode, RenderGraphAccessMode::Read) ||
          hasAccessFlag(selectedMode, RenderGraphAccessMode::Write)) {
        work.compiledResourceUses[writeIndex++] = RenderGraphResourceUse{
            .passIndex = work.compiledResourceUses[groupBegin].passIndex,
            .resourceKind = work.compiledResourceUses[groupBegin].resourceKind,
            .resourceIndex =
                work.compiledResourceUses[groupBegin].resourceIndex,
            .access = selectedMode,
            .state = hasExplicit ? explicitState : inferredState,
            .stage = work.compiledResourceUses[groupBegin].stage,
            .subresources = work.compiledResourceUses[groupBegin].subresources,
            .provenance = hasExplicit
                              ? RenderGraphResourceUseProvenance::Explicit
                              : RenderGraphResourceUseProvenance::Inferred,
        };
      }
      groupBegin = groupEnd;
    }
    work.compiledResourceUses.resize(writeIndex);
  }
  compiled.plan.resourceUses = work.compiledResourceUses;
  NURI_PROFILER_ZONE("RenderGraph.compile.build_topology",
                     NURI_PROFILER_COLOR_BARRIER);
  PmrHashSet<uint64_t> dependencyEdgeKeys(memory_);
  dependencyEdgeKeys.reserve(dependencies_.size() +
                             work.compiledResourceUses.size() * 2u);
  std::pmr::vector<DependencyEdge> allDependencies(memory_);
  for (const DependencyEdge edge : dependencies_) {
    const uint64_t key =
        (static_cast<uint64_t>(edge.before) << 32u) | edge.after;
    if (dependencyEdgeKeys.insert(key).second) {
      allDependencies.push_back(edge);
    }
  }
  allDependencies.reserve(dependencies_.size() +
                          work.compiledResourceUses.size() * 2u);
  struct ResourceAccessGroup {
    uint32_t resourceIndex = UINT32_MAX;
    size_t begin = 0u;
    size_t end = 0u;
    bool hasPartialTextureRanges = false;
  };
  const auto mergeHazardEdgeKeys =
      [&allDependencies,
       &dependencyEdgeKeys](const std::vector<uint64_t> &mergedKeys) {
        for (const uint64_t key : mergedKeys) {
          if (!dependencyEdgeKeys.insert(key).second) {
            continue;
          }
          allDependencies.push_back(DependencyEdge{
              .before = static_cast<uint32_t>(key >> 32u),
              .after = static_cast<uint32_t>(key & 0xffffffffu),
          });
        }
      };
  const auto addResourceHazards = [&](RenderGraphResourceKind resourceKind) {
    std::vector<ResourceAccessGroup> groups{};
    groups.reserve(work.compiledResourceUses.size());
    for (size_t i = 0u; i < work.compiledResourceUses.size();) {
      if (work.compiledResourceUses[i].resourceKind != resourceKind) {
        ++i;
        continue;
      }
      const uint32_t resourceIndex = work.compiledResourceUses[i].resourceIndex;
      const size_t begin = i;
      bool hasPartialTextureRanges = false;
      do {
        hasPartialTextureRanges =
            hasPartialTextureRanges ||
            (resourceKind == RenderGraphResourceKind::Texture &&
             !isWholeTextureRange(work.compiledResourceUses[i].subresources));
        ++i;
      } while (i < work.compiledResourceUses.size() &&
               work.compiledResourceUses[i].resourceKind == resourceKind &&
               work.compiledResourceUses[i].resourceIndex == resourceIndex);
      groups.push_back(ResourceAccessGroup{
          .resourceIndex = resourceIndex,
          .begin = begin,
          .end = i,
          .hasPartialTextureRanges = hasPartialTextureRanges,
      });
    }
    if (groups.empty()) {
      return;
    }
    const auto processGroupRange = [&](uint32_t workerIndex,
                                       RenderGraphContiguousRange range,
                                       std::vector<uint64_t> &edgeKeys) {
      edgeKeys.clear();
      ScopedScratch scratch(runtime.workerScratchArena(workerIndex));
      std::pmr::vector<uint32_t> activeReaders(scratch.resource());
      activeReaders.reserve(work.passCount);
      for (uint32_t groupIndex = range.offset;
           groupIndex < range.offset + range.count; ++groupIndex) {
        const ResourceAccessGroup &group = groups[groupIndex];
        (void)group.resourceIndex;
        if (group.hasPartialTextureRanges) {
          for (size_t accessIndex = group.begin; accessIndex < group.end;
               ++accessIndex) {
            const RenderGraphResourceUse &access =
                work.compiledResourceUses[accessIndex];
            for (size_t previousIndex = group.begin;
                 previousIndex < accessIndex; ++previousIndex) {
              const RenderGraphResourceUse &previous =
                  work.compiledResourceUses[previousIndex];
              if (previous.passIndex == access.passIndex ||
                  !textureRangesOverlap(previous.subresources,
                                        access.subresources)) {
                continue;
              }
              if (hasAccessFlag(previous.access,
                                RenderGraphAccessMode::Write) ||
                  hasAccessFlag(access.access, RenderGraphAccessMode::Write)) {
                edgeKeys.push_back(foldDependencyEdgeKey(previous.passIndex,
                                                         access.passIndex));
              }
            }
          }
          continue;
        }
        activeReaders.clear();
        uint32_t lastWriter = UINT32_MAX;
        for (size_t accessIndex = group.begin; accessIndex < group.end;
             ++accessIndex) {
          const RenderGraphResourceUse &access =
              work.compiledResourceUses[accessIndex];
          const bool hasRead =
              hasAccessFlag(access.access, RenderGraphAccessMode::Read);
          const bool hasWrite =
              hasAccessFlag(access.access, RenderGraphAccessMode::Write);
          if (!hasRead && !hasWrite) {
            continue;
          }
          if (hasRead && lastWriter != UINT32_MAX &&
              lastWriter != access.passIndex) {
            edgeKeys.push_back(
                foldDependencyEdgeKey(lastWriter, access.passIndex));
          }
          if (hasWrite) {
            if (lastWriter != UINT32_MAX && lastWriter != access.passIndex) {
              edgeKeys.push_back(
                  foldDependencyEdgeKey(lastWriter, access.passIndex));
            }
            for (const uint32_t reader : activeReaders) {
              if (reader == access.passIndex) {
                continue;
              }
              edgeKeys.push_back(
                  foldDependencyEdgeKey(reader, access.passIndex));
            }
            activeReaders.clear();
            lastWriter = access.passIndex;
          } else {
            activeReaders.push_back(access.passIndex);
          }
        }
      }
      std::sort(edgeKeys.begin(), edgeKeys.end());
      edgeKeys.erase(std::unique(edgeKeys.begin(), edgeKeys.end()),
                     edgeKeys.end());
    };
    const bool canParallelizeHazards =
        runtime.parallelCompileEnabled() && groups.size() > 1u;
    const uint32_t maxRangeCount =
        canParallelizeHazards ? std::min(runtime.workerCount(),
                                         static_cast<uint32_t>(groups.size()))
                              : 1u;
    const std::vector<RenderGraphContiguousRange> stdRanges =
        makeHazardRanges(static_cast<uint32_t>(groups.size()), maxRangeCount);
    std::vector<std::vector<uint64_t>> workerEdgeKeys(stdRanges.size());
    if (canParallelizeHazards && stdRanges.size() > 1u) {
      std::pmr::vector<RenderGraphContiguousRange> ranges(memory_);
      ranges.assign(stdRanges.begin(), stdRanges.end());
      runtime.runRanges(
          std::span<const RenderGraphContiguousRange>(ranges.data(),
                                                      ranges.size()),
          [&](uint32_t workerIndex, RenderGraphContiguousRange range) {
            processGroupRange(workerIndex, range, workerEdgeKeys[workerIndex]);
          });
      work.usedParallelHazardAnalysis = true;
    } else if (!stdRanges.empty()) {
      processGroupRange(0u, stdRanges[0u], workerEdgeKeys[0u]);
    }
    size_t mergedCount = 0u;
    for (const auto &edgeKeys : workerEdgeKeys) {
      mergedCount += edgeKeys.size();
    }
    std::vector<uint64_t> mergedEdgeKeys{};
    mergedEdgeKeys.reserve(mergedCount);
    for (const auto &edgeKeys : workerEdgeKeys) {
      mergedEdgeKeys.insert(mergedEdgeKeys.end(), edgeKeys.begin(),
                            edgeKeys.end());
    }
    std::sort(mergedEdgeKeys.begin(), mergedEdgeKeys.end());
    mergedEdgeKeys.erase(
        std::unique(mergedEdgeKeys.begin(), mergedEdgeKeys.end()),
        mergedEdgeKeys.end());
    mergeHazardEdgeKeys(mergedEdgeKeys);
  };
  addResourceHazards(RenderGraphResourceKind::Texture);
  addResourceHazards(RenderGraphResourceKind::Buffer);
  addResourceHazards(RenderGraphResourceKind::AccelerationStructure);
  work.activePassMask.resize(work.passCount, 1u);
  if (!frameOutputTextureIndices_.empty() || !sideEffectPassMarks_.empty()) {
    std::fill(work.activePassMask.begin(), work.activePassMask.end(), 0u);
    std::pmr::vector<uint32_t> reverseCount(memory_);
    reverseCount.resize(work.passCount, 0u);
    for (const DependencyEdge edge : allDependencies) {
      if (edge.before >= work.passCount || edge.after >= work.passCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: dependency edge references "
            "out-of-range pass");
      }
      ++reverseCount[edge.after];
    }
    std::pmr::vector<uint32_t> reverseOffsets(memory_);
    reverseOffsets.resize(static_cast<size_t>(work.passCount) + 1u, 0u);
    for (uint32_t i = 0; i < work.passCount; ++i) {
      reverseOffsets[i + 1u] = reverseOffsets[i] + reverseCount[i];
    }
    std::pmr::vector<uint32_t> reverseEdges(memory_);
    reverseEdges.resize(allDependencies.size(), 0u);
    std::pmr::vector<uint32_t> reverseCursor(memory_);
    reverseCursor = reverseOffsets;
    for (const DependencyEdge edge : allDependencies) {
      reverseEdges[reverseCursor[edge.after]++] = edge.before;
    }
    std::pmr::vector<uint32_t> stack(memory_);
    stack.reserve(work.passCount);
    const auto pushRoot = [&work, &stack, &compiled](uint32_t passIndex) {
      if (work.activePassMask[passIndex] != 0u) {
        return;
      }
      work.activePassMask[passIndex] = 1u;
      stack.push_back(passIndex);
      ++compiled.plan.rootPassCount;
    };
    const bool hasExplicitFrameOutputRoots =
        !frameOutputTextureIndices_.empty();
    const bool suppressInferredSideEffectRoots =
        suppressInferredSideEffectsWhenExplicitOutputs_ &&
        hasExplicitFrameOutputRoots;
    for (const SideEffectPassMark &mark : sideEffectPassMarks_) {
      const uint32_t passIndex = mark.passIndex;
      if (!isValidPassIndex(passIndex)) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: side-effect pass index is out of "
            "range");
      }
      if (mark.inferred && suppressInferredSideEffectRoots) {
        continue;
      }
      pushRoot(passIndex);
    }
    std::pmr::vector<RenderGraphAccessMode> textureAccessByPass(memory_);
    textureAccessByPass.resize(work.passCount, RenderGraphAccessMode::None);
    for (const uint32_t textureIndex : frameOutputTextureIndices_) {
      if (!isValidTextureIndex(textureIndex)) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: frame-output texture index is out "
            "of range");
      }
      std::fill(textureAccessByPass.begin(), textureAccessByPass.end(),
                RenderGraphAccessMode::None);
      for (const RenderGraphResourceUse &access : work.compiledResourceUses) {
        if (access.resourceKind != RenderGraphResourceKind::Texture ||
            access.resourceIndex != textureIndex) {
          continue;
        }
        textureAccessByPass[access.passIndex] =
            textureAccessByPass[access.passIndex] | access.access;
      }
      for (uint32_t passIndex = 0; passIndex < work.passCount; ++passIndex) {
        if (hasAccessFlag(textureAccessByPass[passIndex],
                          RenderGraphAccessMode::Write)) {
          pushRoot(passIndex);
        }
      }
    }
    while (!stack.empty()) {
      const uint32_t current = stack.back();
      stack.pop_back();
      const uint32_t begin = reverseOffsets[current];
      const uint32_t end = reverseOffsets[current + 1u];
      for (uint32_t edgeIndex = begin; edgeIndex < end; ++edgeIndex) {
        const uint32_t predecessor = reverseEdges[edgeIndex];
        if (work.activePassMask[predecessor] != 0u) {
          continue;
        }
        work.activePassMask[predecessor] = 1u;
        stack.push_back(predecessor);
      }
    }
    work.activePassCount = 0u;
    for (const uint8_t isActive : work.activePassMask) {
      work.activePassCount += static_cast<uint32_t>(isActive != 0u);
    }
    compiled.plan.culledPassCount = work.passCount - work.activePassCount;
  }
  work.scheduledDependencies.reserve(allDependencies.size());
  for (const DependencyEdge edge : allDependencies) {
    if (work.activePassMask[edge.before] == 0u ||
        work.activePassMask[edge.after] == 0u) {
      continue;
    }
    work.scheduledDependencies.push_back(edge);
  }
  std::pmr::vector<uint32_t> indegree(memory_);
  indegree.resize(work.passCount, 0u);
  std::pmr::vector<uint32_t> outgoingCount(memory_);
  outgoingCount.resize(work.passCount, 0u);
  for (const DependencyEdge edge : work.scheduledDependencies) {
    if (!isValidPassIndex(edge.before) || !isValidPassIndex(edge.after)) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency edge references "
          "out-of-range pass");
    }
    if (indegree[edge.after] == UINT32_MAX) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency indegree overflow");
    }
    ++indegree[edge.after];
    if (outgoingCount[edge.before] == UINT32_MAX) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency outgoing edge overflow");
    }
    ++outgoingCount[edge.before];
  }
  std::pmr::vector<uint32_t> outgoingOffsets(memory_);
  outgoingOffsets.resize(static_cast<size_t>(work.passCount) + 1u, 0u);
  for (uint32_t i = 0; i < work.passCount; ++i) {
    outgoingOffsets[i + 1u] = outgoingOffsets[i] + outgoingCount[i];
  }
  std::pmr::vector<uint32_t> outgoingEdges(memory_);
  outgoingEdges.resize(work.scheduledDependencies.size(), 0u);
  std::pmr::vector<uint32_t> outgoingCursor(memory_);
  outgoingCursor = outgoingOffsets;
  for (const DependencyEdge edge : work.scheduledDependencies) {
    const uint32_t cursor = outgoingCursor[edge.before]++;
    outgoingEdges[cursor] = edge.after;
  }
  std::pmr::vector<uint32_t> readyStorage(memory_);
  readyStorage.reserve(work.passCount);
  std::priority_queue<uint32_t, std::pmr::vector<uint32_t>,
                      std::greater<uint32_t>>
      ready(std::greater<uint32_t>{}, std::move(readyStorage));
  for (uint32_t i = 0; i < indegree.size(); ++i) {
    if (work.activePassMask[i] != 0u && indegree[i] == 0u) {
      ready.push(i);
    }
  }
  work.order.reserve(work.activePassCount);
  while (!ready.empty()) {
    const uint32_t current = ready.top();
    ready.pop();
    work.order.push_back(current);
    const uint32_t start = outgoingOffsets[current];
    const uint32_t end = outgoingOffsets[current + 1u];
    for (uint32_t edgeIndex = start; edgeIndex < end; ++edgeIndex) {
      const uint32_t next = outgoingEdges[edgeIndex];
      if (indegree[next] == 0u) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: indegree underflow");
      }
      --indegree[next];
      if (indegree[next] == 0u) {
        ready.push(next);
      }
    }
  }
  if (work.order.size() != work.activePassCount) {
    std::ostringstream message;
    message << "RenderGraphBuilder::compile: dependency cycle detected";
    bool hasCyclePass = false;
    for (uint32_t i = 0; i < work.passCount; ++i) {
      if (work.activePassMask[i] == 0u || indegree[i] == 0u) {
        continue;
      }
      if (!hasCyclePass) {
        message << " among passes ";
      } else {
        message << ", ";
      }
      message << "[" << i << "] " << resolvePassDebugName(passDebugNames_, i);
      hasCyclePass = true;
    }
    return Result<bool, std::string>::makeError(message.str());
  }
  NURI_PROFILER_ZONE_END();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RenderGraphBuilder::compileStageC3ResolvePassPayloads(
    RenderGraphRuntime &runtime, CompiledRenderGraph &compiled,
    const RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  compiled.commands.passDebugNames.reserve(passDebugNames_.size());
  for (const std::pmr::string &name : passDebugNames_) {
    std::pmr::string copiedName(memory_);
    copiedName.assign(name.data(), name.size());
    compiled.commands.passDebugNames.push_back(std::move(copiedName));
  }
  compiled.plan.edges.reserve(work.scheduledDependencies.size());
  for (const DependencyEdge edge : work.scheduledDependencies) {
    compiled.plan.edges.push_back(
        RenderGraphPlan::Edge{.before = edge.before, .after = edge.after});
  }
  struct PassResolvePlan {
    uint32_t passIndex = UINT32_MAX;
    uint32_t colorTextureIndex = UINT32_MAX;
    uint32_t colorResolveTextureIndex = UINT32_MAX;
    uint32_t depthTextureIndex = UINT32_MAX;
    uint32_t depthResolveTextureIndex = UINT32_MAX;
    uint32_t dependencyCount = 0u;
    uint32_t dependencyBindingOffset = 0u;
    uint32_t resolvedDependencyOffset = 0u;
    uint32_t dependencyTextureCount = 0u;
    uint32_t dependencyTextureBindingOffset = 0u;
    uint32_t resolvedDependencyTextureOffset = 0u;
    uint32_t preDispatchCount = 0u;
    uint32_t preDispatchBindingOffset = 0u;
    uint32_t preDispatchOutputOffset = 0u;
    uint32_t preDispatchDependencyOffset = 0u;
    uint32_t preDispatchDependencyCount = 0u;
    uint32_t drawCount = 0u;
    uint32_t drawBindingOffset = 0u;
    uint32_t drawOutputOffset = 0u;
    uint32_t arenaDrawOutputOffset = 0u;
    uint32_t quantizedDrawCount = 0u;
    uint32_t meshDispatchCount = 0u;
    uint32_t meshDispatchBindingOffset = 0u;
    uint32_t meshDispatchOutputOffset = 0u;
    uint32_t bufferCopyCount = 0u;
    uint32_t bufferCopyBindingOffset = 0u;
    uint32_t bufferCopyOutputOffset = 0u;
    uint32_t textureCopyCount = 0u;
    uint32_t textureCopyBindingOffset = 0u;
    uint32_t textureCopyOutputOffset = 0u;
    uint32_t resourcePatchOffset = 0u;
    uint32_t resourcePatchCount = 0u;
    uint32_t drawPatchCount = 0u;
  };
  const auto countTransientBindings = [](const auto &bindings, uint32_t offset,
                                         uint32_t count,
                                         const auto &resources) {
    uint32_t transientCount = 0u;
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t resourceIndex = bindings[offset + i];
      transientCount +=
          resourceIndex != UINT32_MAX && !resources[resourceIndex].imported;
    }
    return transientCount;
  };
  std::pmr::vector<PassResolvePlan> passPlans(memory_);
  passPlans.resize(work.order.size());
  const auto planPassRange = [&](uint32_t, RenderGraphContiguousRange range) {
    for (uint32_t orderedPassIndex = range.offset;
         orderedPassIndex < range.offset + range.count; ++orderedPassIndex) {
      const uint32_t passIndex = work.order[orderedPassIndex];
      const RenderPass &pass = passes_[passIndex];
      PassResolvePlan &plan = passPlans[orderedPassIndex];
      plan.passIndex = passIndex;
      plan.colorTextureIndex = passBindings_[passIndex].color;
      plan.colorResolveTextureIndex = passBindings_[passIndex].colorResolve;
      plan.depthTextureIndex = passBindings_[passIndex].depth;
      plan.depthResolveTextureIndex = passBindings_[passIndex].depthResolve;
      plan.dependencyCount = passBindings_[passIndex].dependencyBuffers.count;
      plan.dependencyBindingOffset =
          passBindings_[passIndex].dependencyBuffers.offset;
      plan.dependencyTextureCount =
          passBindings_[passIndex].dependencyTextures.count;
      plan.dependencyTextureBindingOffset =
          passBindings_[passIndex].dependencyTextures.offset;
      plan.preDispatchCount = passBindings_[passIndex].preDispatches.count;
      plan.preDispatchBindingOffset =
          passBindings_[passIndex].preDispatches.offset;
      plan.drawCount = static_cast<uint32_t>(pass.draws.size());
      plan.drawBindingOffset = passBindings_[passIndex].draws.offset;
      plan.meshDispatchCount =
          static_cast<uint32_t>(pass.meshDispatches.size());
      plan.meshDispatchBindingOffset =
          passBindings_[passIndex].meshDispatches.offset;
      plan.bufferCopyCount = static_cast<uint32_t>(pass.bufferCopies.size());
      plan.bufferCopyBindingOffset =
          passBindings_[passIndex].bufferCopies.offset;
      plan.textureCopyCount = static_cast<uint32_t>(pass.textureCopies.size());
      plan.textureCopyBindingOffset =
          passBindings_[passIndex].textureCopies.offset;
      const auto needsTexturePatch = [&](uint32_t resourceIndex) {
        return resourceIndex != UINT32_MAX &&
               !textures_[resourceIndex].imported;
      };
      plan.resourcePatchCount =
          needsTexturePatch(plan.colorTextureIndex) +
          needsTexturePatch(plan.colorResolveTextureIndex) +
          needsTexturePatch(plan.depthTextureIndex) +
          needsTexturePatch(plan.depthResolveTextureIndex);
      plan.resourcePatchCount += countTransientBindings(
          passDependencyBufferBindingResourceIndices_,
          plan.dependencyBindingOffset, plan.dependencyCount, buffers_);
      plan.resourcePatchCount +=
          countTransientBindings(passDependencyTextureBindingResourceIndices_,
                                 plan.dependencyTextureBindingOffset,
                                 plan.dependencyTextureCount, textures_);
      for (uint32_t i = 0; i < plan.preDispatchCount; ++i) {
        const uint32_t bindingIndex = plan.preDispatchBindingOffset + i;
        const uint32_t offset =
            preDispatchDependencyBindings_[bindingIndex].offset;
        const uint32_t count =
            preDispatchDependencyBindings_[bindingIndex].count;
        plan.preDispatchDependencyCount += count;
        plan.resourcePatchCount +=
            countTransientBindings(preDispatchDependencyBindingResourceIndices_,
                                   offset, count, buffers_);
      }
      if (!pass.drawBuffersPreResolved) {
        for (uint32_t i = 0; i < plan.drawCount; ++i) {
          const DrawBindings binding =
              drawBindings_[plan.drawBindingOffset + i];
          for (const uint32_t resource :
               {binding.vertex, binding.index, binding.indirect,
                binding.indirectCount}) {
            const uint32_t needsPatch =
                resource != UINT32_MAX && !buffers_[resource].imported;
            plan.resourcePatchCount += needsPatch;
            plan.drawPatchCount += needsPatch;
          }
        }
      }
      for (uint32_t i = 0; i < plan.meshDispatchCount; ++i) {
        const MeshDispatchBindings binding =
            meshDispatchBindings_[plan.meshDispatchBindingOffset + i];
        plan.resourcePatchCount += (binding.indirect != UINT32_MAX &&
                                    !buffers_[binding.indirect].imported) +
                                   (binding.indirectCount != UINT32_MAX &&
                                    !buffers_[binding.indirectCount].imported);
      }
      for (uint32_t i = 0; i < plan.bufferCopyCount; ++i) {
        const BufferCopyBindings binding =
            bufferCopyBindings_[plan.bufferCopyBindingOffset + i];
        plan.resourcePatchCount += !buffers_[binding.source].imported;
        plan.resourcePatchCount += !buffers_[binding.destination].imported;
      }
      for (uint32_t i = 0; i < plan.textureCopyCount; ++i) {
        const TextureCopyBindings binding =
            textureCopyBindings_[plan.textureCopyBindingOffset + i];
        plan.resourcePatchCount += !textures_[binding.source].imported;
        plan.resourcePatchCount += !textures_[binding.destination].imported;
      }
    }
  };
  const uint32_t workerCount = std::max(1u, runtime.workerCount());
  const std::vector<RenderGraphContiguousRange> payloadRanges =
      runtime.parallelCompileEnabled() && work.order.size() > 1u
          ? makePayloadRanges(static_cast<uint32_t>(work.order.size()),
                              workerCount)
          : std::vector<RenderGraphContiguousRange>{};
  const bool usedParallelPassResolution = payloadRanges.size() > 1u;
  if (usedParallelPassResolution) {
    runtime.runRanges(payloadRanges, planPassRange);
  } else {
    planPassRange(0u, RenderGraphContiguousRange{
                          .offset = 0u,
                          .count = static_cast<uint32_t>(work.order.size())});
  }
  compiled.commands.usedParallelPayloadResolution = usedParallelPassResolution;
  size_t totalDependencyBufferSlots = 0u;
  size_t totalDependencyTextureSlots = 0u;
  size_t totalPreDispatchItems = 0u;
  size_t totalPreDispatchDependencySlots = 0u;
  size_t totalDrawItems = 0u;
  size_t totalMeshDispatchItems = 0u;
  size_t totalBufferCopyItems = 0u;
  size_t totalTextureCopyItems = 0u;
  size_t totalResourcePatches = 0u;
  for (uint32_t orderedPassIndex = 0u; orderedPassIndex < passPlans.size();
       ++orderedPassIndex) {
    PassResolvePlan &plan = passPlans[orderedPassIndex];
    plan.resolvedDependencyOffset =
        static_cast<uint32_t>(totalDependencyBufferSlots);
    totalDependencyBufferSlots += plan.dependencyCount;
    plan.resolvedDependencyTextureOffset =
        static_cast<uint32_t>(totalDependencyTextureSlots);
    totalDependencyTextureSlots += plan.dependencyTextureCount;
    plan.preDispatchOutputOffset = static_cast<uint32_t>(totalPreDispatchItems);
    totalPreDispatchItems += plan.preDispatchCount;
    plan.preDispatchDependencyOffset =
        static_cast<uint32_t>(totalPreDispatchDependencySlots);
    totalPreDispatchDependencySlots += plan.preDispatchDependencyCount;
    if (plan.drawCount != 0u && (plan.drawPatchCount != 0u ||
                                 !passes_[plan.passIndex].payloadBorrowed)) {
      if (totalDrawItems > UINT32_MAX) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: draw item output offset exceeds "
            "uint32_t range");
      }
      plan.drawOutputOffset = static_cast<uint32_t>(totalDrawItems);
      plan.arenaDrawOutputOffset = plan.drawOutputOffset;
      const uint64_t quantizedCount =
          quantizeToNextPow2(static_cast<uint64_t>(plan.drawCount));
      if (quantizedCount > UINT32_MAX) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: quantized draw count exceeds "
            "uint32_t range");
      }
      if (quantizedCount >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max() -
                                totalDrawItems)) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: total draw item count overflow");
      }
      if (quantizedCount > static_cast<uint64_t>(UINT32_MAX) -
                               static_cast<uint64_t>(totalDrawItems)) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: total draw item count exceeds "
            "uint32_t range");
      }
      plan.quantizedDrawCount = static_cast<uint32_t>(quantizedCount);
      totalDrawItems += static_cast<size_t>(quantizedCount);
    } else {
      plan.drawOutputOffset = 0u;
      plan.arenaDrawOutputOffset = 0u;
      plan.quantizedDrawCount = 0u;
    }
    if (totalMeshDispatchItems > UINT32_MAX) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: mesh dispatch output offset exceeds "
          "uint32_t range");
    }
    plan.meshDispatchOutputOffset =
        static_cast<uint32_t>(totalMeshDispatchItems);
    if (plan.meshDispatchCount >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max() -
                              totalMeshDispatchItems)) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: total mesh dispatch count overflow");
    }
    totalMeshDispatchItems += plan.meshDispatchCount;
    if (totalBufferCopyItems > UINT32_MAX) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: buffer copy output offset exceeds "
          "uint32_t range");
    }
    plan.bufferCopyOutputOffset = static_cast<uint32_t>(totalBufferCopyItems);
    if (plan.bufferCopyCount >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max() -
                              totalBufferCopyItems)) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: total buffer copy count overflow");
    }
    totalBufferCopyItems += plan.bufferCopyCount;
    if (totalTextureCopyItems > UINT32_MAX) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: texture copy output offset exceeds "
          "uint32_t range");
    }
    plan.textureCopyOutputOffset = static_cast<uint32_t>(totalTextureCopyItems);
    if (plan.textureCopyCount >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max() -
                              totalTextureCopyItems)) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: total texture copy count overflow");
    }
    totalTextureCopyItems += plan.textureCopyCount;
    plan.resourcePatchOffset = static_cast<uint32_t>(totalResourcePatches);
    totalResourcePatches += plan.resourcePatchCount;
  }
  size_t totalArenaDrawItems = totalDrawItems;
  for (PassResolvePlan &plan : passPlans) {
    if (plan.drawCount == 0u || plan.quantizedDrawCount != 0u) {
      continue;
    }
    if (plan.drawCount >
        std::numeric_limits<size_t>::max() - totalArenaDrawItems) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: dynamic draw item count overflow");
    }
    if (totalArenaDrawItems > UINT32_MAX ||
        plan.drawCount > UINT32_MAX - totalArenaDrawItems) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: dynamic draw item count exceeds "
          "uint32_t range");
    }
    plan.arenaDrawOutputOffset = static_cast<uint32_t>(totalArenaDrawItems);
    totalArenaDrawItems += plan.drawCount;
  }
  compiled.commands.resolvedDependencyBuffers.resize(
      totalDependencyBufferSlots);
  compiled.plan.resolvedDependencyBufferResourceIndices.assign(
      totalDependencyBufferSlots, UINT32_MAX);
  compiled.plan.dependencyBufferRangesByPass.resize(work.order.size());
  compiled.commands.resolvedDependencyTextures.resize(
      totalDependencyTextureSlots);
  compiled.plan.resolvedDependencyTextureResourceIndices.assign(
      totalDependencyTextureSlots, UINT32_MAX);
  compiled.plan.dependencyTextureRangesByPass.resize(work.order.size());
  compiled.commands.ownedPreDispatches.resize(totalPreDispatchItems);
  compiled.commands.ownedPreDispatchDebugLabels.resize(totalPreDispatchItems);
  compiled.commands.ownedPreDispatchPushConstants.resize(totalPreDispatchItems);
  compiled.commands.ownedPreDispatchDependencyTextures.resize(
      totalPreDispatchItems);
  compiled.commands.ownedPreDispatchTextureBindings.resize(
      totalPreDispatchItems);
  compiled.plan.preDispatchRangesByPass.resize(work.order.size());
  compiled.commands.resolvedPreDispatchDependencyBuffers.resize(
      totalPreDispatchDependencySlots);
  compiled.plan.resolvedPreDispatchDependencyBufferResourceIndices.assign(
      totalPreDispatchDependencySlots, UINT32_MAX);
  compiled.plan.preDispatchDependencyRanges.resize(totalPreDispatchItems);
  compiled.commands.ownedDrawItems.resize(totalArenaDrawItems);
  compiled.commands.ownedDrawDebugLabels.resize(totalArenaDrawItems);
  compiled.commands.ownedDrawPushConstants.resize(totalArenaDrawItems);
  compiled.commands.ownedDrawTextureBindings.resize(totalArenaDrawItems);
  compiled.plan.drawRangesByPass.resize(work.order.size());
  compiled.commands.ownedMeshDispatchItems.resize(totalMeshDispatchItems);
  compiled.commands.ownedBufferCopyItems.resize(totalBufferCopyItems);
  compiled.commands.ownedTextureCopyItems.resize(totalTextureCopyItems);
  compiled.commands.ownedAccelerationStructureBuildsByPass.resize(
      work.order.size());
  compiled.commands.ownedAccelerationStructureGeometriesByPass.resize(
      work.order.size());
  compiled.commands.ownedAccelerationStructureInstancesByPass.resize(
      work.order.size());
  {
    compiled.commands.ownedMeshDispatchDebugLabels.clear();
    compiled.commands.ownedMeshDispatchPushConstants.clear();
    compiled.commands.ownedMeshDispatchDependencyBuffers.clear();
    compiled.commands.ownedMeshDispatchDependencyTextures.clear();
    compiled.commands.ownedMeshDispatchTextureBindings.clear();
    compiled.commands.ownedMeshDispatchDebugLabels.reserve(
        totalMeshDispatchItems);
    compiled.commands.ownedMeshDispatchPushConstants.reserve(
        totalMeshDispatchItems);
    compiled.commands.ownedMeshDispatchDependencyBuffers.reserve(
        totalMeshDispatchItems);
    compiled.commands.ownedMeshDispatchDependencyTextures.reserve(
        totalMeshDispatchItems);
    compiled.commands.ownedMeshDispatchTextureBindings.reserve(
        totalMeshDispatchItems);
    for (size_t i = 0u; i < totalMeshDispatchItems; ++i) {
      compiled.commands.ownedMeshDispatchDebugLabels.emplace_back();
      compiled.commands.ownedMeshDispatchPushConstants.emplace_back();
      compiled.commands.ownedMeshDispatchDependencyBuffers.emplace_back();
      compiled.commands.ownedMeshDispatchDependencyTextures.emplace_back();
      compiled.commands.ownedMeshDispatchTextureBindings.emplace_back();
    }
  }
  compiled.plan.meshDispatchRangesByPass.resize(work.order.size());
  compiled.plan.bufferCopyRangesByPass.resize(work.order.size());
  compiled.plan.textureCopyRangesByPass.resize(work.order.size());
  compiled.commands.orderedPasses.resize(work.order.size());
  compiled.plan.orderedPassIndices.resize(work.order.size());
  compiled.plan.recordedGraphicsPasses.resize(work.order.size());
  compiled.plan.passBarrierPlans.resize(work.order.size());
  compiled.plan.commandResourcePatches.resize(totalResourcePatches);
  const auto fillPassRange = [&](uint32_t, RenderGraphContiguousRange range) {
    for (uint32_t orderedPassIndex = range.offset;
         orderedPassIndex < range.offset + range.count; ++orderedPassIndex) {
      const PassResolvePlan &plan = passPlans[orderedPassIndex];
      const uint32_t passIndex = plan.passIndex;
      const RenderPass &sourcePass = passes_[passIndex];
      RenderPass resolvedPass = sourcePass;
      uint32_t resourcePatchWriteOffset = plan.resourcePatchOffset;
      if (plan.colorTextureIndex != UINT32_MAX) {
        const TextureResource &resource = textures_[plan.colorTextureIndex];
        if (resource.imported) {
          resolvedPass.colorTexture = resource.importedHandle;
        } else {
          resolvedPass.colorTexture = {};
          compiled.plan.commandResourcePatches[resourcePatchWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .resourceIndex = plan.colorTextureIndex,
              .target = RenderGraphPlan::CommandResourcePatchTarget::PassColor};
        }
      }
      if (plan.colorResolveTextureIndex != UINT32_MAX) {
        const TextureResource &resource =
            textures_[plan.colorResolveTextureIndex];
        if (resource.imported) {
          resolvedPass.colorResolveTexture = resource.importedHandle;
        } else {
          resolvedPass.colorResolveTexture = {};
          compiled.plan.commandResourcePatches[resourcePatchWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .resourceIndex = plan.colorResolveTextureIndex,
              .target = RenderGraphPlan::CommandResourcePatchTarget::
                  PassColorResolve};
        }
      }
      if (plan.depthTextureIndex != UINT32_MAX) {
        const TextureResource &resource = textures_[plan.depthTextureIndex];
        if (resource.imported) {
          resolvedPass.depthTexture = resource.importedHandle;
        } else {
          resolvedPass.depthTexture = {};
          compiled.plan.commandResourcePatches[resourcePatchWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .resourceIndex = plan.depthTextureIndex,
              .target = RenderGraphPlan::CommandResourcePatchTarget::PassDepth};
        }
      }
      if (plan.depthResolveTextureIndex != UINT32_MAX) {
        const TextureResource &resource =
            textures_[plan.depthResolveTextureIndex];
        if (resource.imported) {
          resolvedPass.depthResolveTexture = resource.importedHandle;
        } else {
          resolvedPass.depthResolveTexture = {};
          compiled.plan.commandResourcePatches[resourcePatchWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .resourceIndex = plan.depthResolveTextureIndex,
              .target = RenderGraphPlan::CommandResourcePatchTarget::
                  PassDepthResolve};
        }
      }
      compiled.plan.dependencyBufferRangesByPass[orderedPassIndex] = {
          .offset = plan.resolvedDependencyOffset,
          .count = plan.dependencyCount};
      for (uint32_t depIndex = 0; depIndex < plan.dependencyCount; ++depIndex) {
        const uint32_t resourceIndex =
            passDependencyBufferBindingResourceIndices_
                [plan.dependencyBindingOffset + depIndex];
        BufferHandle &resolvedHandle =
            compiled.commands
                .resolvedDependencyBuffers[plan.resolvedDependencyOffset +
                                           depIndex];
        if (resourceIndex == UINT32_MAX) {
          resolvedHandle = {};
          continue;
        }
        const BufferResource &resource = buffers_[resourceIndex];
        if (resource.imported) {
          resolvedHandle = resource.importedHandle;
          compiled.plan.resolvedDependencyBufferResourceIndices
              [plan.resolvedDependencyOffset + depIndex] = resourceIndex;
        } else {
          resolvedHandle = {};
          compiled.plan.commandResourcePatches[resourcePatchWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .dependencyIndex = depIndex,
              .resourceIndex = resourceIndex,
              .resourceKind = RenderGraphResourceKind::Buffer,
              .target = RenderGraphPlan::CommandResourcePatchTarget::
                  PassDependencyBuffer};
        }
      }
      if (plan.dependencyCount > 0u) {
        resolvedPass.dependencyBuffers = std::span<const BufferHandle>(
            compiled.commands.resolvedDependencyBuffers.data() +
                plan.resolvedDependencyOffset,
            plan.dependencyCount);
      } else {
        resolvedPass.dependencyBuffers = {};
      }
      compiled.plan.dependencyTextureRangesByPass[orderedPassIndex] = {
          .offset = plan.resolvedDependencyTextureOffset,
          .count = plan.dependencyTextureCount};
      for (uint32_t depIndex = 0; depIndex < plan.dependencyTextureCount;
           ++depIndex) {
        const uint32_t resourceIndex =
            passDependencyTextureBindingResourceIndices_
                [plan.dependencyTextureBindingOffset + depIndex];
        TextureHandle &resolvedHandle =
            compiled.commands.resolvedDependencyTextures
                [plan.resolvedDependencyTextureOffset + depIndex];
        if (resourceIndex == UINT32_MAX) {
          resolvedHandle = {};
          continue;
        }
        const TextureResource &resource = textures_[resourceIndex];
        if (resource.imported) {
          resolvedHandle = resource.importedHandle;
          compiled.plan.resolvedDependencyTextureResourceIndices
              [plan.resolvedDependencyTextureOffset + depIndex] = resourceIndex;
        } else {
          resolvedHandle = {};
          compiled.plan.commandResourcePatches[resourcePatchWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .dependencyIndex = depIndex,
              .resourceIndex = resourceIndex,
              .target = RenderGraphPlan::CommandResourcePatchTarget::
                  PassDependencyTexture};
        }
      }
      if (plan.dependencyTextureCount > 0u) {
        resolvedPass.dependencyTextures = std::span<const TextureHandle>(
            compiled.commands.resolvedDependencyTextures.data() +
                plan.resolvedDependencyTextureOffset,
            plan.dependencyTextureCount);
      } else {
        resolvedPass.dependencyTextures = {};
      }
      compiled.plan.preDispatchRangesByPass[orderedPassIndex] = {
          .offset = plan.preDispatchOutputOffset,
          .count = plan.preDispatchCount};
      uint32_t nextPreDispatchDependencyOffset =
          plan.preDispatchDependencyOffset;
      for (uint32_t dispatchIndex = 0; dispatchIndex < plan.preDispatchCount;
           ++dispatchIndex) {
        const ComputeDispatchItem &sourceDispatch =
            sourcePass.preDispatches[dispatchIndex];
        ComputeDispatchItem resolvedDispatch = sourceDispatch;
        const uint32_t globalDispatchBindingIndex =
            plan.preDispatchBindingOffset + dispatchIndex;
        const uint32_t dispatchDependencyOffset =
            preDispatchDependencyBindings_[globalDispatchBindingIndex].offset;
        const uint32_t dispatchDependencyCount =
            preDispatchDependencyBindings_[globalDispatchBindingIndex].count;
        compiled.plan.preDispatchDependencyRanges[plan.preDispatchOutputOffset +
                                                  dispatchIndex] = {
            .offset = nextPreDispatchDependencyOffset,
            .count = dispatchDependencyCount};
        for (uint32_t depIndex = 0; depIndex < dispatchDependencyCount;
             ++depIndex) {
          const uint32_t resourceIndex =
              preDispatchDependencyBindingResourceIndices_
                  [dispatchDependencyOffset + depIndex];
          BufferHandle &resolvedHandle =
              compiled.commands.resolvedPreDispatchDependencyBuffers
                  [nextPreDispatchDependencyOffset + depIndex];
          if (resourceIndex == UINT32_MAX) {
            resolvedHandle = {};
            continue;
          }
          const BufferResource &resource = buffers_[resourceIndex];
          if (resource.imported) {
            resolvedHandle = resource.importedHandle;
            compiled.plan.resolvedPreDispatchDependencyBufferResourceIndices
                [nextPreDispatchDependencyOffset + depIndex] = resourceIndex;
          } else {
            resolvedHandle = {};
            compiled.plan.commandResourcePatches[resourcePatchWriteOffset++] = {
                .orderedPassIndex = orderedPassIndex,
                .commandIndex = dispatchIndex,
                .dependencyIndex = depIndex,
                .resourceIndex = resourceIndex,
                .resourceKind = RenderGraphResourceKind::Buffer,
                .target = RenderGraphPlan::CommandResourcePatchTarget::
                    PreDispatchDependencyBuffer};
          }
        }
        if (dispatchDependencyCount > 0u) {
          resolvedDispatch.dependencyBuffers = std::span<const BufferHandle>(
              compiled.commands.resolvedPreDispatchDependencyBuffers.data() +
                  nextPreDispatchDependencyOffset,
              dispatchDependencyCount);
        } else {
          resolvedDispatch.dependencyBuffers = {};
        }
        resolvedDispatch.dependencyBufferAccessModes = {};
        ownPreDispatchPayload(compiled.commands,
                              plan.preDispatchOutputOffset + dispatchIndex,
                              sourceDispatch, resolvedDispatch);
        compiled.commands
            .ownedPreDispatches[plan.preDispatchOutputOffset + dispatchIndex] =
            resolvedDispatch;
        nextPreDispatchDependencyOffset += dispatchDependencyCount;
      }
      if (plan.preDispatchCount > 0u) {
        resolvedPass.preDispatches = std::span<const ComputeDispatchItem>(
            compiled.commands.ownedPreDispatches.data() +
                plan.preDispatchOutputOffset,
            plan.preDispatchCount);
      } else {
        resolvedPass.preDispatches = {};
      }
      if (plan.drawCount > 0u && plan.drawPatchCount == 0u &&
          sourcePass.payloadBorrowed) {
        compiled.plan.drawRangesByPass[orderedPassIndex] = {};
        for (uint32_t drawIndex = 0u; drawIndex < plan.drawCount; ++drawIndex) {
          const size_t outputIndex = plan.arenaDrawOutputOffset + drawIndex;
          DrawItem draw = sourcePass.draws[drawIndex];
          ownDrawPayload(compiled.commands, outputIndex,
                         sourcePass.draws[drawIndex], draw);
          compiled.commands.ownedDrawItems[outputIndex] = draw;
        }
        resolvedPass.draws =
            std::span<const DrawItem>(compiled.commands.ownedDrawItems.data() +
                                          plan.arenaDrawOutputOffset,
                                      plan.drawCount);
      } else if (plan.drawPatchCount == 0u) {
        compiled.plan.drawRangesByPass[orderedPassIndex] = {
            .offset = plan.drawOutputOffset, .count = plan.quantizedDrawCount};
        for (uint32_t drawIndex = 0; drawIndex < plan.drawCount; ++drawIndex) {
          const size_t outputIndex = plan.drawOutputOffset + drawIndex;
          DrawItem draw = sourcePass.draws[drawIndex];
          ownDrawPayload(compiled.commands, outputIndex,
                         sourcePass.draws[drawIndex], draw);
          compiled.commands.ownedDrawItems[outputIndex] = draw;
        }
        resolvedPass.draws = plan.drawCount == 0u
                                 ? std::span<const DrawItem>{}
                                 : std::span<const DrawItem>(
                                       compiled.commands.ownedDrawItems.data() +
                                           plan.drawOutputOffset,
                                       plan.drawCount);
      } else {
        compiled.plan.drawRangesByPass[orderedPassIndex] = {
            .offset = plan.drawOutputOffset, .count = plan.quantizedDrawCount};
        for (uint32_t drawIndex = 0; drawIndex < plan.drawCount; ++drawIndex) {
          const DrawItem &sourceDraw = sourcePass.draws[drawIndex];
          DrawItem resolvedDraw = sourceDraw;
          const uint32_t globalDrawIndex = plan.drawBindingOffset + drawIndex;
          const auto resolveDrawBinding =
              [&](uint32_t resourceIndex,
                  RenderGraphPlan::CommandResourcePatchTarget target,
                  BufferHandle &slotHandle) {
                if (resourceIndex == UINT32_MAX) {
                  slotHandle = {};
                  return;
                }
                const BufferResource &resource = buffers_[resourceIndex];
                if (resource.imported) {
                  slotHandle = resource.importedHandle;
                  return;
                }
                slotHandle = {};
                compiled.plan
                    .commandResourcePatches[resourcePatchWriteOffset++] = {
                    .orderedPassIndex = orderedPassIndex,
                    .commandIndex = drawIndex,
                    .resourceIndex = resourceIndex,
                    .resourceKind = RenderGraphResourceKind::Buffer,
                    .target = target};
              };
          resolveDrawBinding(
              drawBindings_[globalDrawIndex].vertex,
              RenderGraphPlan::CommandResourcePatchTarget::DrawVertexBuffer,
              resolvedDraw.vertexBuffer);
          resolveDrawBinding(
              drawBindings_[globalDrawIndex].index,
              RenderGraphPlan::CommandResourcePatchTarget::DrawIndexBuffer,
              resolvedDraw.indexBuffer);
          resolveDrawBinding(
              drawBindings_[globalDrawIndex].indirect,
              RenderGraphPlan::CommandResourcePatchTarget::DrawIndirectBuffer,
              resolvedDraw.indirectBuffer);
          resolveDrawBinding(drawBindings_[globalDrawIndex].indirectCount,
                             RenderGraphPlan::CommandResourcePatchTarget::
                                 DrawIndirectCountBuffer,
                             resolvedDraw.indirectCountBuffer);
          ownDrawPayload(compiled.commands, plan.drawOutputOffset + drawIndex,
                         sourceDraw, resolvedDraw);
          compiled.commands.ownedDrawItems[plan.drawOutputOffset + drawIndex] =
              resolvedDraw;
        }
        if (plan.drawCount > 0u) {
          resolvedPass.draws = std::span<const DrawItem>(
              compiled.commands.ownedDrawItems.data() + plan.drawOutputOffset,
              plan.drawCount);
        } else {
          resolvedPass.draws = {};
        }
      }
      compiled.plan.meshDispatchRangesByPass[orderedPassIndex] = {
          .offset = plan.meshDispatchOutputOffset,
          .count = plan.meshDispatchCount};
      if (plan.meshDispatchCount > 0u) {
        for (uint32_t dispatchIndex = 0; dispatchIndex < plan.meshDispatchCount;
             ++dispatchIndex) {
          const size_t outputIndex =
              static_cast<size_t>(plan.meshDispatchOutputOffset) +
              dispatchIndex;
          const MeshDispatchItem &sourceDispatch =
              sourcePass.meshDispatches[dispatchIndex];
          MeshDispatchItem resolvedDispatch = sourceDispatch;
          const uint32_t globalDispatchIndex =
              plan.meshDispatchBindingOffset + dispatchIndex;
          const auto resolveMeshDispatchBinding =
              [&](uint32_t resourceIndex,
                  RenderGraphPlan::CommandResourcePatchTarget target,
                  BufferHandle &slotHandle) {
                if (resourceIndex == UINT32_MAX) {
                  slotHandle = {};
                  return;
                }
                const BufferResource &resource = buffers_[resourceIndex];
                if (resource.imported) {
                  slotHandle = resource.importedHandle;
                  return;
                }
                slotHandle = {};
                compiled.plan
                    .commandResourcePatches[resourcePatchWriteOffset++] = {
                    .orderedPassIndex = orderedPassIndex,
                    .commandIndex = dispatchIndex,
                    .resourceIndex = resourceIndex,
                    .resourceKind = RenderGraphResourceKind::Buffer,
                    .target = target,
                };
              };
          resolveMeshDispatchBinding(
              meshDispatchBindings_[globalDispatchIndex].indirect,
              RenderGraphPlan::CommandResourcePatchTarget::
                  MeshDispatchIndirectBuffer,
              resolvedDispatch.indirectBuffer);
          resolveMeshDispatchBinding(
              meshDispatchBindings_[globalDispatchIndex].indirectCount,
              RenderGraphPlan::CommandResourcePatchTarget::
                  MeshDispatchIndirectCountBuffer,
              resolvedDispatch.indirectCountBuffer);
          ownMeshDispatchPayload(compiled.commands, outputIndex, sourceDispatch,
                                 resolvedDispatch);
          compiled.commands.ownedMeshDispatchItems[outputIndex] =
              resolvedDispatch;
        }
        resolvedPass.meshDispatches = std::span<const MeshDispatchItem>(
            compiled.commands.ownedMeshDispatchItems.data() +
                plan.meshDispatchOutputOffset,
            plan.meshDispatchCount);
      } else {
        resolvedPass.meshDispatches = {};
      }
      compiled.plan.bufferCopyRangesByPass[orderedPassIndex] = {
          .offset = plan.bufferCopyOutputOffset, .count = plan.bufferCopyCount};
      if (plan.bufferCopyCount > 0u) {
        for (uint32_t copyIndex = 0; copyIndex < plan.bufferCopyCount;
             ++copyIndex) {
          BufferCopyRegion resolvedCopy = sourcePass.bufferCopies[copyIndex];
          const uint32_t globalCopyIndex =
              plan.bufferCopyBindingOffset + copyIndex;
          const auto resolveBufferCopyBinding =
              [&](uint32_t resourceIndex,
                  RenderGraphPlan::CommandResourcePatchTarget target,
                  BufferHandle &slotHandle) {
                if (resourceIndex == UINT32_MAX) {
                  slotHandle = {};
                  return;
                }
                const BufferResource &resource = buffers_[resourceIndex];
                if (resource.imported) {
                  slotHandle = resource.importedHandle;
                  return;
                }
                slotHandle = {};
                compiled.plan
                    .commandResourcePatches[resourcePatchWriteOffset++] = {
                    .orderedPassIndex = orderedPassIndex,
                    .commandIndex = copyIndex,
                    .resourceIndex = resourceIndex,
                    .resourceKind = RenderGraphResourceKind::Buffer,
                    .target = target,
                };
              };
          resolveBufferCopyBinding(
              bufferCopyBindings_[globalCopyIndex].source,
              RenderGraphPlan::CommandResourcePatchTarget::BufferCopySource,
              resolvedCopy.srcBuffer);
          resolveBufferCopyBinding(
              bufferCopyBindings_[globalCopyIndex].destination,
              RenderGraphPlan::CommandResourcePatchTarget::
                  BufferCopyDestination,
              resolvedCopy.dstBuffer);
          compiled.commands
              .ownedBufferCopyItems[plan.bufferCopyOutputOffset + copyIndex] =
              resolvedCopy;
        }
        resolvedPass.bufferCopies = std::span<const BufferCopyRegion>(
            compiled.commands.ownedBufferCopyItems.data() +
                plan.bufferCopyOutputOffset,
            plan.bufferCopyCount);
      } else {
        resolvedPass.bufferCopies = {};
      }
      compiled.plan.textureCopyRangesByPass[orderedPassIndex] = {
          .offset = plan.textureCopyOutputOffset,
          .count = plan.textureCopyCount};
      if (plan.textureCopyCount > 0u) {
        for (uint32_t copyIndex = 0; copyIndex < plan.textureCopyCount;
             ++copyIndex) {
          TextureCopyItem resolvedCopy = sourcePass.textureCopies[copyIndex];
          const uint32_t globalCopyIndex =
              plan.textureCopyBindingOffset + copyIndex;
          const auto resolveTextureCopyBinding =
              [&](uint32_t resourceIndex,
                  RenderGraphPlan::CommandResourcePatchTarget target,
                  TextureHandle &slotHandle) {
                if (resourceIndex == UINT32_MAX) {
                  slotHandle = {};
                  return;
                }
                const TextureResource &resource = textures_[resourceIndex];
                if (resource.imported) {
                  slotHandle = resource.importedHandle;
                  return;
                }
                slotHandle = {};
                compiled.plan
                    .commandResourcePatches[resourcePatchWriteOffset++] = {
                    .orderedPassIndex = orderedPassIndex,
                    .commandIndex = copyIndex,
                    .resourceIndex = resourceIndex,
                    .target = target,
                };
              };
          resolveTextureCopyBinding(
              textureCopyBindings_[globalCopyIndex].source,
              RenderGraphPlan::CommandResourcePatchTarget::TextureCopySource,
              resolvedCopy.sourceTexture);
          resolveTextureCopyBinding(
              textureCopyBindings_[globalCopyIndex].destination,
              RenderGraphPlan::CommandResourcePatchTarget::
                  TextureCopyDestination,
              resolvedCopy.destinationTexture);
          compiled.commands
              .ownedTextureCopyItems[plan.textureCopyOutputOffset + copyIndex] =
              resolvedCopy;
        }
        resolvedPass.textureCopies = std::span<const TextureCopyItem>(
            compiled.commands.ownedTextureCopyItems.data() +
                plan.textureCopyOutputOffset,
            plan.textureCopyCount);
      } else {
        resolvedPass.textureCopies = {};
      }
      ownAccelerationStructureBuilds(compiled.commands, orderedPassIndex,
                                     sourcePass, resolvedPass);
      const std::pmr::string &compiledName =
          compiled.commands.passDebugNames[passIndex];
      resolvedPass.debugLabel =
          std::string_view(compiledName.data(), compiledName.size());
      compiled.plan.orderedPassIndices[orderedPassIndex] = passIndex;
      compiled.plan.recordedGraphicsPasses[orderedPassIndex] = {
          .orderedPassIndex = orderedPassIndex, .declaredPassIndex = passIndex};
      compiled.plan.passBarrierPlans[orderedPassIndex] = {.orderedPassIndex =
                                                              orderedPassIndex,
                                                          .barrierOffset = 0u,
                                                          .barrierCount = 0u};
      compiled.commands.orderedPasses[orderedPassIndex] = resolvedPass;
      NURI_ASSERT(resourcePatchWriteOffset ==
                      plan.resourcePatchOffset + plan.resourcePatchCount,
                  "render-graph command resource patch count mismatch");
    }
  };
  if (usedParallelPassResolution) {
    runtime.runRanges(payloadRanges, fillPassRange);
  } else {
    fillPassRange(0u, RenderGraphContiguousRange{
                          .offset = 0u,
                          .count = static_cast<uint32_t>(work.order.size())});
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RenderGraphBuilder::compileStageC4PlanBarriers(
    CompiledRenderGraph &compiled,
    const RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_BARRIER);
  compiled.plan.passBarrierRecords.clear();
  compiled.plan.finalBarrierPlan = {};
  std::pmr::vector<uint32_t> executionRankByPass(memory_);
  executionRankByPass.resize(work.passCount, UINT32_MAX);
  for (uint32_t rank = 0; rank < work.order.size(); ++rank) {
    executionRankByPass[work.order[rank]] = rank;
  }
  std::pmr::vector<RenderGraphResourceUse> orderedAccesses(memory_);
  orderedAccesses.reserve(work.compiledResourceUses.size());
  for (const RenderGraphResourceUse &access : work.compiledResourceUses) {
    if (access.passIndex >= work.passCount ||
        work.activePassMask[access.passIndex] == 0u ||
        executionRankByPass[access.passIndex] == UINT32_MAX) {
      continue;
    }
    orderedAccesses.push_back(access);
  }
  std::sort(orderedAccesses.begin(), orderedAccesses.end(),
            [&executionRankByPass](const RenderGraphResourceUse &lhs,
                                   const RenderGraphResourceUse &rhs) {
              const uint8_t lhsKind = static_cast<uint8_t>(lhs.resourceKind);
              const uint8_t rhsKind = static_cast<uint8_t>(rhs.resourceKind);
              if (lhsKind != rhsKind) {
                return lhsKind < rhsKind;
              }
              if (lhs.resourceIndex != rhs.resourceIndex) {
                return lhs.resourceIndex < rhs.resourceIndex;
              }
              const uint32_t lhsRank = executionRankByPass[lhs.passIndex];
              const uint32_t rhsRank = executionRankByPass[rhs.passIndex];
              if (lhsRank != rhsRank) {
                return lhsRank < rhsRank;
              }
              return lhs.passIndex < rhs.passIndex;
            });
  const auto resolveResourceState = [&](const RenderGraphResourceUse &access) {
    if (access.state != RenderGraphResourceState::Unknown) {
      return access.state;
    }
    const bool hasWrite =
        hasAccessFlag(access.access, RenderGraphAccessMode::Write);
    if (access.resourceKind == RenderGraphResourceKind::Texture &&
        (passBindings_[access.passIndex].color == access.resourceIndex ||
         passBindings_[access.passIndex].colorResolve == access.resourceIndex ||
         passBindings_[access.passIndex].depth == access.resourceIndex ||
         passBindings_[access.passIndex].depthResolve ==
             access.resourceIndex)) {
      return RenderGraphResourceState::Attachment;
    }
    return hasWrite ? RenderGraphResourceState::Write
                    : RenderGraphResourceState::Read;
  };
  std::pmr::vector<RenderGraphBarrierRecord> stagedBarrierRecords(memory_);
  std::pmr::vector<uint32_t> stagedBarrierPassIndices(memory_);
  std::pmr::vector<RenderGraphBarrierRecord> stagedFinalBarrierRecords(memory_);
  std::pmr::vector<uint32_t> barrierCounts(memory_);
  barrierCounts.resize(work.order.size(), 0u);
  std::pmr::vector<RenderGraphAccessMode> lastTextureAccessByResource(memory_);
  lastTextureAccessByResource.resize(textures_.size(),
                                     RenderGraphAccessMode::None);
  std::pmr::vector<RenderGraphResourceState> lastTextureStateByResource(
      memory_);
  lastTextureStateByResource.resize(textures_.size(),
                                    RenderGraphResourceState::Unknown);
  std::pmr::vector<uint8_t> hasLastTextureAccess(memory_);
  hasLastTextureAccess.resize(textures_.size(), 0u);
  std::pmr::vector<uint8_t> hasPartialTextureRanges(memory_);
  hasPartialTextureRanges.resize(textures_.size(), 0u);
  for (const RenderGraphResourceUse &access : orderedAccesses) {
    if (access.resourceKind == RenderGraphResourceKind::Texture &&
        access.resourceIndex < textures_.size() &&
        !isWholeTextureRange(access.subresources)) {
      hasPartialTextureRanges[access.resourceIndex] = 1u;
    }
  }
  std::pmr::vector<RenderGraphResourceUse> priorSubresourceAccesses(memory_);
  RenderGraphResourceKind previousKind = RenderGraphResourceKind::Texture;
  uint32_t previousResourceIndex = UINT32_MAX;
  RenderGraphAccessMode previousAccess = RenderGraphAccessMode::None;
  RenderGraphResourceState previousState = RenderGraphResourceState::Unknown;
  bool havePreviousResource = false;
  for (const RenderGraphResourceUse &access : orderedAccesses) {
    const bool sameResource = havePreviousResource &&
                              previousKind == access.resourceKind &&
                              previousResourceIndex == access.resourceIndex;
    if (!sameResource) {
      previousKind = access.resourceKind;
      previousResourceIndex = access.resourceIndex;
      previousAccess = RenderGraphAccessMode::None;
      previousState = RenderGraphResourceState::Unknown;
      havePreviousResource = true;
      priorSubresourceAccesses.clear();
    }
    const RenderGraphResourceState nextState = resolveResourceState(access);
    if (access.resourceKind == RenderGraphResourceKind::Texture &&
        access.resourceIndex < hasPartialTextureRanges.size() &&
        hasPartialTextureRanges[access.resourceIndex] != 0u) {
      RenderGraphAccessMode beforeAccess = RenderGraphAccessMode::None;
      RenderGraphResourceState beforeState = RenderGraphResourceState::Unknown;
      for (auto previous = priorSubresourceAccesses.rbegin();
           previous != priorSubresourceAccesses.rend(); ++previous) {
        if (textureRangesOverlap(previous->subresources, access.subresources)) {
          beforeAccess = previous->access;
          beforeState = resolveResourceState(*previous);
          break;
        }
      }
      const uint32_t orderedPassIndex = executionRankByPass[access.passIndex];
      stagedBarrierRecords.push_back(RenderGraphBarrierRecord{
          .resourceKind = RenderGraphBarrierResourceKind::Texture,
          .resourceIndex = access.resourceIndex,
          .beforeAccess = beforeAccess,
          .afterAccess = access.access,
          .beforeState = beforeState,
          .afterState = nextState,
          .subresources = access.subresources});
      stagedBarrierPassIndices.push_back(orderedPassIndex);
      ++barrierCounts[orderedPassIndex];
      priorSubresourceAccesses.push_back(access);
      lastTextureAccessByResource[access.resourceIndex] = access.access;
      lastTextureStateByResource[access.resourceIndex] = nextState;
      hasLastTextureAccess[access.resourceIndex] = 1u;
      previousAccess = access.access;
      previousState = nextState;
      continue;
    }
    const bool needsBarrier =
        previousState == RenderGraphResourceState::Unknown ||
        previousState != nextState ||
        hasAccessFlag(previousAccess, RenderGraphAccessMode::Write) ||
        hasAccessFlag(access.access, RenderGraphAccessMode::Write);
    if (needsBarrier) {
      const uint32_t orderedPassIndex = executionRankByPass[access.passIndex];
      stagedBarrierRecords.push_back(RenderGraphBarrierRecord{
          .resourceKind =
              [&]() {
                switch (access.resourceKind) {
                case RenderGraphResourceKind::Texture:
                  return RenderGraphBarrierResourceKind::Texture;
                case RenderGraphResourceKind::Buffer:
                  return RenderGraphBarrierResourceKind::Buffer;
                case RenderGraphResourceKind::AccelerationStructure:
                  return RenderGraphBarrierResourceKind::AccelerationStructure;
                }
                return RenderGraphBarrierResourceKind::Texture;
              }(),
          .resourceIndex = access.resourceIndex,
          .beforeAccess = previousAccess,
          .afterAccess = access.access,
          .beforeState = previousState,
          .afterState = nextState,
          .subresources = access.subresources,
      });
      stagedBarrierPassIndices.push_back(orderedPassIndex);
      ++barrierCounts[orderedPassIndex];
    }
    if (access.resourceKind == RenderGraphResourceKind::Texture &&
        access.resourceIndex < textures_.size()) {
      lastTextureAccessByResource[access.resourceIndex] = access.access;
      lastTextureStateByResource[access.resourceIndex] = nextState;
      hasLastTextureAccess[access.resourceIndex] = 1u;
    }
    previousAccess = access.access;
    previousState = nextState;
  }
  if (!frameOutputTextureIndices_.empty()) {
    std::pmr::vector<uint32_t> sortedFrameOutputTextures(memory_);
    sortedFrameOutputTextures.assign(frameOutputTextureIndices_.begin(),
                                     frameOutputTextureIndices_.end());
    std::sort(sortedFrameOutputTextures.begin(),
              sortedFrameOutputTextures.end());
    for (const uint32_t textureIndex : sortedFrameOutputTextures) {
      if (textureIndex >= textures_.size() ||
          hasLastTextureAccess[textureIndex] == 0u) {
        continue;
      }
      const RenderGraphResourceState lastState =
          lastTextureStateByResource[textureIndex];
      if (lastState == RenderGraphResourceState::Present) {
        continue;
      }
      stagedFinalBarrierRecords.push_back(RenderGraphBarrierRecord{
          .resourceKind = RenderGraphBarrierResourceKind::Texture,
          .resourceIndex = textureIndex,
          .beforeAccess = lastTextureAccessByResource[textureIndex],
          .afterAccess = RenderGraphAccessMode::None,
          .beforeState = lastState,
          .afterState = RenderGraphResourceState::Present,
      });
    }
  }
  compiled.plan.passBarrierRecords.resize(stagedBarrierRecords.size() +
                                          stagedFinalBarrierRecords.size());
  std::pmr::vector<uint32_t> nextBarrierOffset(memory_);
  nextBarrierOffset.resize(compiled.plan.passBarrierPlans.size(), 0u);
  uint32_t runningBarrierOffset = 0u;
  for (uint32_t orderedPassIndex = 0u;
       orderedPassIndex < compiled.plan.passBarrierPlans.size();
       ++orderedPassIndex) {
    compiled.plan.passBarrierPlans[orderedPassIndex].barrierOffset =
        runningBarrierOffset;
    compiled.plan.passBarrierPlans[orderedPassIndex].barrierCount =
        barrierCounts[orderedPassIndex];
    nextBarrierOffset[orderedPassIndex] = runningBarrierOffset;
    runningBarrierOffset += barrierCounts[orderedPassIndex];
  }
  compiled.plan.finalBarrierPlan = FinalBarrierPlan{
      .barrierOffset = runningBarrierOffset,
      .barrierCount = static_cast<uint32_t>(stagedFinalBarrierRecords.size()),
  };
  for (uint32_t i = 0u; i < stagedBarrierRecords.size(); ++i) {
    const uint32_t orderedPassIndex = stagedBarrierPassIndices[i];
    compiled.plan.passBarrierRecords[nextBarrierOffset[orderedPassIndex]++] =
        stagedBarrierRecords[i];
  }
  for (uint32_t i = 0u; i < stagedFinalBarrierRecords.size(); ++i) {
    compiled.plan
        .passBarrierRecords[compiled.plan.finalBarrierPlan.barrierOffset + i] =
        stagedFinalBarrierRecords[i];
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RenderGraphBuilder::compileStageC5PlanTransientLifetimes(
    RenderGraphRuntime &runtime, CompiledRenderGraph &compiled,
    RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  std::pmr::vector<uint32_t> executionRankByPass(memory_);
  executionRankByPass.resize(work.passCount, UINT32_MAX);
  for (uint32_t rank = 0; rank < work.order.size(); ++rank) {
    executionRankByPass[work.order[rank]] = rank;
  }
  TransientLifetimeRanks<TextureLifetimeTag> textureRanks(memory_,
                                                          textures_.size());
  TransientLifetimeRanks<BufferLifetimeTag> bufferRanks(memory_,
                                                        buffers_.size());
  const auto updateLifetimeRanks = [](std::span<uint32_t> firstRanks,
                                      std::span<uint32_t> lastRanks,
                                      uint32_t resourceIndex, uint32_t rank) {
    if (firstRanks[resourceIndex] == UINT32_MAX ||
        rank < firstRanks[resourceIndex]) {
      firstRanks[resourceIndex] = rank;
    }
    if (rank > lastRanks[resourceIndex]) {
      lastRanks[resourceIndex] = rank;
    }
  };
  const auto analyzeAccessRange = [&](std::span<uint32_t> textureFirstRanks,
                                      std::span<uint32_t> textureLastRanks,
                                      std::span<uint32_t> bufferFirstRanks,
                                      std::span<uint32_t> bufferLastRanks,
                                      RenderGraphContiguousRange range) {
    for (uint32_t accessIndex = range.offset;
         accessIndex < range.offset + range.count; ++accessIndex) {
      const RenderGraphResourceUse &access =
          work.compiledResourceUses[accessIndex];
      if (access.passIndex >= work.passCount ||
          work.activePassMask[access.passIndex] == 0u) {
        continue;
      }
      const uint32_t rank = executionRankByPass[access.passIndex];
      if (rank == UINT32_MAX) {
        continue;
      }
      if (access.resourceKind == RenderGraphResourceKind::Texture) {
        if (access.resourceIndex >= textures_.size() ||
            textures_[access.resourceIndex].imported) {
          continue;
        }
        updateLifetimeRanks(textureFirstRanks, textureLastRanks,
                            access.resourceIndex, rank);
        continue;
      }
      if (access.resourceKind == RenderGraphResourceKind::Buffer) {
        if (access.resourceIndex >= buffers_.size() ||
            buffers_[access.resourceIndex].imported) {
          continue;
        }
        updateLifetimeRanks(bufferFirstRanks, bufferLastRanks,
                            access.resourceIndex, rank);
      }
    }
  };
  struct WorkerLifetimeRanks {
    TransientLifetimeRanks<TextureLifetimeTag> textures;
    TransientLifetimeRanks<BufferLifetimeTag> buffers;
    WorkerLifetimeRanks(std::pmr::memory_resource *memory, size_t textureCount,
                        size_t bufferCount)
        : textures(memory, textureCount), buffers(memory, bufferCount) {}
  };
  bool usedParallelLifetimeAnalysis = false;
  if (!work.compiledResourceUses.empty()) {
    const std::vector<RenderGraphContiguousRange> stdRanges =
        runtime.parallelCompileEnabled() &&
                work.compiledResourceUses.size() > 1u
            ? makeLifetimeRanges(
                  static_cast<uint32_t>(work.compiledResourceUses.size()),
                  runtime.workerCount())
            : std::vector<RenderGraphContiguousRange>{};
    if (stdRanges.size() > 1u) {
      std::vector<WorkerLifetimeRanks> workerRanks{};
      workerRanks.reserve(stdRanges.size());
      for (size_t i = 0; i < stdRanges.size(); ++i) {
        workerRanks.emplace_back(memory_, textures_.size(), buffers_.size());
      }
      std::pmr::vector<RenderGraphContiguousRange> ranges(memory_);
      ranges.assign(stdRanges.begin(), stdRanges.end());
      runtime.runRanges(
          std::span<const RenderGraphContiguousRange>(ranges.data(),
                                                      ranges.size()),
          [&](uint32_t workerIndex, RenderGraphContiguousRange range) {
            WorkerLifetimeRanks &worker = workerRanks[workerIndex];
            analyzeAccessRange(std::span<uint32_t>(worker.textures.first),
                               std::span<uint32_t>(worker.textures.last),
                               std::span<uint32_t>(worker.buffers.first),
                               std::span<uint32_t>(worker.buffers.last), range);
          });
      usedParallelLifetimeAnalysis = true;
      for (const WorkerLifetimeRanks &worker : workerRanks) {
        for (uint32_t textureIndex = 0; textureIndex < textures_.size();
             ++textureIndex) {
          if (worker.textures.first[textureIndex] == UINT32_MAX) {
            continue;
          }
          updateLifetimeRanks(textureRanks.first, textureRanks.last,
                              textureIndex,
                              worker.textures.first[textureIndex]);
          textureRanks.last[textureIndex] =
              std::max(textureRanks.last[textureIndex],
                       worker.textures.last[textureIndex]);
        }
        for (uint32_t bufferIndex = 0; bufferIndex < buffers_.size();
             ++bufferIndex) {
          if (worker.buffers.first[bufferIndex] == UINT32_MAX) {
            continue;
          }
          updateLifetimeRanks(bufferRanks.first, bufferRanks.last, bufferIndex,
                              worker.buffers.first[bufferIndex]);
          bufferRanks.last[bufferIndex] = std::max(
              bufferRanks.last[bufferIndex], worker.buffers.last[bufferIndex]);
        }
      }
    } else {
      analyzeAccessRange(
          textureRanks.first, textureRanks.last, bufferRanks.first,
          bufferRanks.last,
          RenderGraphContiguousRange{.offset = 0u,
                                     .count = static_cast<uint32_t>(
                                         work.compiledResourceUses.size())});
    }
  }
  compiled.plan.usedParallelHazardAnalysis = work.usedParallelHazardAnalysis;
  compiled.plan.usedParallelLifetimeAnalysis = usedParallelLifetimeAnalysis;
  compiled.plan.usedParallelCompile =
      compiled.plan.usedParallelHazardAnalysis ||
      compiled.plan.usedParallelLifetimeAnalysis;
  for (uint32_t textureIndex = 0; textureIndex < textures_.size();
       ++textureIndex) {
    if (textures_[textureIndex].imported ||
        textureRanks.first[textureIndex] == UINT32_MAX) {
      continue;
    }
    compiled.plan.transientTextureLifetimes.push_back(
        {.resourceIndex = textureIndex,
         .firstExecutionIndex = textureRanks.first[textureIndex],
         .lastExecutionIndex = textureRanks.last[textureIndex]});
  }
  for (uint32_t bufferIndex = 0; bufferIndex < buffers_.size(); ++bufferIndex) {
    if (buffers_[bufferIndex].imported ||
        bufferRanks.first[bufferIndex] == UINT32_MAX) {
      continue;
    }
    compiled.plan.transientBufferLifetimes.push_back(
        {.resourceIndex = bufferIndex,
         .firstExecutionIndex = bufferRanks.first[bufferIndex],
         .lastExecutionIndex = bufferRanks.last[bufferIndex]});
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RenderGraphBuilder::compileStageC6PlanTransientAliasing(
    CompiledRenderGraph &compiled) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const auto planAliasing =
      [this](const auto &resources, const auto &lifetimes,
             auto &allocationByResource, auto &allocations,
             auto &physicalAllocations, uint32_t &physicalCount,
             const auto &isCompatible, const auto &makePhysicalAllocation) {
        std::pmr::vector<uint32_t> orderIndices(memory_);
        orderIndices.resize(lifetimes.size(), 0u);
        std::iota(orderIndices.begin(), orderIndices.end(), 0u);
        std::sort(orderIndices.begin(), orderIndices.end(),
                  [&lifetimes](uint32_t lhs, uint32_t rhs) {
                    const auto &a = lifetimes[lhs];
                    const auto &b = lifetimes[rhs];
                    return std::tie(a.firstExecutionIndex, a.lastExecutionIndex,
                                    a.resourceIndex) <
                           std::tie(b.firstExecutionIndex, b.lastExecutionIndex,
                                    b.resourceIndex);
                  });
        std::pmr::vector<uint32_t> slotLastUse(memory_);
        std::pmr::vector<uint32_t> slotRepresentativeResource(memory_);
        slotLastUse.reserve(lifetimes.size());
        slotRepresentativeResource.reserve(lifetimes.size());
        for (const uint32_t lifetimeIndex : orderIndices) {
          const auto &lifetime = lifetimes[lifetimeIndex];
          uint32_t chosenSlot = UINT32_MAX;
          for (uint32_t slot = 0u; slot < slotLastUse.size(); ++slot) {
            if (slotLastUse[slot] >= lifetime.firstExecutionIndex) {
              continue;
            }
            const uint32_t representative = slotRepresentativeResource[slot];
            if (representative < resources.size() &&
                lifetime.resourceIndex < resources.size() &&
                isCompatible(resources[representative].transientDesc,
                             resources[lifetime.resourceIndex].transientDesc)) {
              chosenSlot = slot;
              break;
            }
          }
          if (chosenSlot == UINT32_MAX) {
            chosenSlot = static_cast<uint32_t>(slotLastUse.size());
            slotLastUse.push_back(lifetime.lastExecutionIndex);
            slotRepresentativeResource.push_back(lifetime.resourceIndex);
            physicalAllocations.push_back(makePhysicalAllocation(
                chosenSlot, lifetime.resourceIndex,
                resources[lifetime.resourceIndex].transientDesc));
          } else {
            slotLastUse[chosenSlot] = lifetime.lastExecutionIndex;
          }
          allocationByResource[lifetime.resourceIndex] = chosenSlot;
          allocations.push_back(RenderGraphPlan::TransientAllocation{
              .resourceIndex = lifetime.resourceIndex,
              .allocationIndex = chosenSlot});
        }
        std::sort(allocations.begin(), allocations.end(),
                  [](const auto &a, const auto &b) {
                    return a.resourceIndex < b.resourceIndex;
                  });
        physicalCount = static_cast<uint32_t>(physicalAllocations.size());
      };
  {
    NURI_PROFILER_ZONE("RenderGraph.compile.plan_texture_aliasing",
                       NURI_PROFILER_COLOR_CREATE);
    planAliasing(
        textures_, compiled.plan.transientTextureLifetimes,
        compiled.plan.transientTextureAllocationByResource,
        compiled.plan.transientTextureAllocations,
        compiled.plan.transientTexturePhysicalAllocations,
        compiled.plan.transientTexturePhysicalCount,
        isTextureDescAliasCompatible,
        [](uint32_t allocationIndex, uint32_t resourceIndex, TextureDesc desc) {
          desc.data = {};
          return RenderGraphPlan::TransientTexturePhysicalAllocation{
              .allocationIndex = allocationIndex,
              .representativeResourceIndex = resourceIndex,
              .desc = desc};
        });
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("RenderGraph.compile.plan_buffer_aliasing",
                       NURI_PROFILER_COLOR_CREATE);
    planAliasing(
        buffers_, compiled.plan.transientBufferLifetimes,
        compiled.plan.transientBufferAllocationByResource,
        compiled.plan.transientBufferAllocations,
        compiled.plan.transientBufferPhysicalAllocations,
        compiled.plan.transientBufferPhysicalCount, isBufferDescAliasCompatible,
        [](uint32_t allocationIndex, uint32_t resourceIndex, BufferDesc desc) {
          desc.data = {};
          return RenderGraphPlan::TransientBufferPhysicalAllocation{
              .allocationIndex = allocationIndex,
              .representativeResourceIndex = resourceIndex,
              .desc = desc};
        });
    NURI_PROFILER_ZONE_END();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<CompiledRenderGraph, std::string>
RenderGraphBuilder::compile(RenderGraphRuntime &runtime) const {
  NURI_PROFILER_FUNCTION();
  CompiledRenderGraph compiled(memory_);
  compiled.commands.frameIndex = frameIndex_;
  CompileWorkState work(memory_);
  compileStageC0BuildResourceTables(compiled, work);
  if (passes_.empty()) {
    compiled.commands = buildFrameCommands(compiled.plan);
    return Result<CompiledRenderGraph, std::string>::makeResult(
        std::move(compiled));
  }
  auto topologyResult = compileStageC1C2BuildTopology(runtime, compiled, work);
  if (topologyResult.hasError()) {
    return Result<CompiledRenderGraph, std::string>::makeError(
        topologyResult.error());
  }
  auto resolveResult =
      compileStageC3ResolvePassPayloads(runtime, compiled, work);
  if (resolveResult.hasError()) {
    return Result<CompiledRenderGraph, std::string>::makeError(
        resolveResult.error());
  }
  auto barrierResult = compileStageC4PlanBarriers(compiled, work);
  if (barrierResult.hasError()) {
    return Result<CompiledRenderGraph, std::string>::makeError(
        barrierResult.error());
  }
  auto lifetimeResult =
      compileStageC5PlanTransientLifetimes(runtime, compiled, work);
  if (lifetimeResult.hasError()) {
    return Result<CompiledRenderGraph, std::string>::makeError(
        lifetimeResult.error());
  }
  compiled.plan.transientTextureAllocationByResource.resize(textures_.size(),
                                                            UINT32_MAX);
  compiled.plan.transientBufferAllocationByResource.resize(buffers_.size(),
                                                           UINT32_MAX);
  auto aliasingResult = compileStageC6PlanTransientAliasing(compiled);
  if (aliasingResult.hasError()) {
    return Result<CompiledRenderGraph, std::string>::makeError(
        aliasingResult.error());
  }
  compiled.plan.usedParallelCompile =
      compiled.plan.usedParallelCompile ||
      compiled.commands.usedParallelPayloadResolution;
  return Result<CompiledRenderGraph, std::string>::makeResult(
      std::move(compiled));
}

RenderGraphExecutor::RenderGraphExecutor(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      pendingFrames_(memory_), reusableTexturesByHash_(memory_),
      reusableBuffersByHash_(memory_) {}

void RenderGraphExecutor::collectRetiredResources(GPUDevice &gpu) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_DESTROY);
  size_t writeIndex = 0u;
  for (size_t readIndex = 0u; readIndex < pendingFrames_.size(); ++readIndex) {
    PendingFrameResources &pending = pendingFrames_[readIndex];
    if (isValid(pending.submission) &&
        !gpu.isSubmissionComplete(pending.submission)) {
      if (writeIndex != readIndex) {
        pendingFrames_[writeIndex] = std::move(pending);
      }
      ++writeIndex;
      continue;
    }
    const bool hasBufferDescs =
        pending.bufferDescs.size() == pending.buffers.size();
    for (size_t bufferIndex = 0u; bufferIndex < pending.buffers.size();
         ++bufferIndex) {
      const BufferHandle buffer = pending.buffers[bufferIndex];
      if (nuri::isValid(buffer)) {
        if (hasBufferDescs &&
            reusableBufferPoolSize_ < kMaxReusableTransientBuffers) {
          ReusableBufferResource entry{};
          entry.handle = buffer;
          entry.desc = pending.bufferDescs[bufferIndex];
          entry.desc.data = {};
          const uint64_t key = hashBufferDescForPool(entry.desc);
          auto it = reusableBuffersByHash_.find(key);
          if (it == reusableBuffersByHash_.end()) {
            auto [inserted, _] = reusableBuffersByHash_.emplace(
                key, std::pmr::vector<ReusableBufferResource>(memory_));
            inserted->second.push_back(entry);
          } else {
            it->second.push_back(entry);
          }
          ++reusableBufferPoolSize_;
        } else {
          gpu.destroyBuffer(buffer);
        }
      }
    }
    const bool hasTextureDescs =
        pending.textureDescs.size() == pending.textures.size();
    for (size_t textureIndex = 0u; textureIndex < pending.textures.size();
         ++textureIndex) {
      const TextureHandle texture = pending.textures[textureIndex];
      if (nuri::isValid(texture)) {
        if (hasTextureDescs &&
            reusableTexturePoolSize_ < kMaxReusableTransientTextures) {
          ReusableTextureResource entry{};
          entry.handle = texture;
          entry.desc = pending.textureDescs[textureIndex];
          entry.desc.data = {};
          const uint64_t key = hashTextureDescForPool(entry.desc);
          auto it = reusableTexturesByHash_.find(key);
          if (it == reusableTexturesByHash_.end()) {
            auto [inserted, _] = reusableTexturesByHash_.emplace(
                key, std::pmr::vector<ReusableTextureResource>(memory_));
            inserted->second.push_back(entry);
          } else {
            it->second.push_back(entry);
          }
          ++reusableTexturePoolSize_;
        } else {
          gpu.destroyTexture(texture);
        }
      }
    }
  }
  pendingFrames_.resize(writeIndex);
}

Result<RenderGraphExecutionMetadata, std::string>
RenderGraphExecutor::execute(RenderGraphRuntime &runtime, GPUDevice &gpu,
                             CompiledRenderGraphView compiled,
                             RenderGraphExecutionOptions options) {
  RenderGraphExecutionMetadata metadata(memory_);
  auto result = executeInternal(&runtime, gpu, compiled, metadata, options);
  if (result.hasError()) {
    return Result<RenderGraphExecutionMetadata, std::string>::makeError(
        result.error());
  }
  return Result<RenderGraphExecutionMetadata, std::string>::makeResult(
      std::move(metadata));
}

Result<bool, std::string> RenderGraphExecutor::executeInternal(
    RenderGraphRuntime *runtime, GPUDevice &gpu,
    CompiledRenderGraphView compiled, RenderGraphExecutionMetadata &metadata,
    RenderGraphExecutionOptions options) {
  NURI_PROFILER_FUNCTION();
  const bool captureTelemetry =
      options.telemetry != RenderGraphTelemetryLevel::None;
  const bool capturePassTimings =
      options.telemetry == RenderGraphTelemetryLevel::PassTimings;
  const auto failWithString = [](RenderGraphExecutionFailureStage stage,
                                 const std::string &message) {
    return Result<bool, std::string>::makeError(
        makeExecutionStageError(stage, message));
  };
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.retire_resources",
                       NURI_PROFILER_COLOR_DESTROY);
    if (!pendingFrames_.empty()) {
      collectRetiredResources(gpu);
    }
    NURI_PROFILER_ZONE_END();
  }
  std::pmr::vector<TextureHandle> transientTextureHandles(memory_);
  transientTextureHandles.resize(compiled.plan.transientTexturePhysicalCount,
                                 TextureHandle{});
  std::pmr::vector<TextureDesc> transientTextureDescs(memory_);
  transientTextureDescs.resize(compiled.plan.transientTexturePhysicalCount,
                               TextureDesc{});
  std::pmr::vector<BufferHandle> transientBufferHandles(memory_);
  transientBufferHandles.resize(compiled.plan.transientBufferPhysicalCount,
                                BufferHandle{});
  std::pmr::vector<BufferDesc> transientBufferDescs(memory_);
  transientBufferDescs.resize(compiled.plan.transientBufferPhysicalCount,
                              BufferDesc{});
  const auto destroyMaterializedResources = [&gpu, &transientTextureHandles,
                                             &transientBufferHandles]() {
    for (const BufferHandle buffer : transientBufferHandles) {
      if (nuri::isValid(buffer)) {
        gpu.destroyBuffer(buffer);
      }
    }
    for (const TextureHandle texture : transientTextureHandles) {
      if (nuri::isValid(texture)) {
        gpu.destroyTexture(texture);
      }
    }
  };
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.materialize_transients",
                       NURI_PROFILER_COLOR_CREATE);
    for (const auto &allocation :
         compiled.plan.transientTexturePhysicalAllocations) {
      TextureDesc desc = allocation.desc;
      desc.data = {};
      transientTextureDescs[allocation.allocationIndex] = desc;
      TextureHandle transientTexture{};
      {
        const uint64_t poolKey = hashTextureDescForPool(desc);
        auto poolIt = reusableTexturesByHash_.find(poolKey);
        if (poolIt != reusableTexturesByHash_.end()) {
          auto &bucket = poolIt->second;
          size_t poolIndex = 0u;
          while (poolIndex < bucket.size()) {
            const ReusableTextureResource &candidate = bucket[poolIndex];
            if (!nuri::isValid(candidate.handle) ||
                !gpu.isValid(candidate.handle)) {
              bucket[poolIndex] = bucket.back();
              bucket.pop_back();
              --reusableTexturePoolSize_;
              continue;
            }
            if (!isTextureDescAliasCompatible(desc, candidate.desc)) {
              ++poolIndex;
              continue;
            }
            transientTexture = candidate.handle;
            bucket[poolIndex] = bucket.back();
            bucket.pop_back();
            --reusableTexturePoolSize_;
            break;
          }
          if (bucket.empty()) {
            reusableTexturesByHash_.erase(poolIt);
          }
        }
      }
      if (!nuri::isValid(transientTexture)) {
        auto createResult = gpu.createTexture(desc, "rg_transient_texture");
        if (createResult.hasError()) {
          destroyMaterializedResources();
          return failWithString(
              RenderGraphExecutionFailureStage::MaterializeTransients,
              "RenderGraphExecutor::execute: failed to create transient "
              "texture: " +
                  createResult.error());
        }
        transientTexture = createResult.value();
      }
      transientTextureHandles[allocation.allocationIndex] = transientTexture;
    }
    for (const auto &allocation :
         compiled.plan.transientBufferPhysicalAllocations) {
      BufferDesc desc = allocation.desc;
      desc.data = {};
      transientBufferDescs[allocation.allocationIndex] = desc;
      BufferHandle transientBuffer{};
      {
        const uint64_t poolKey = hashBufferDescForPool(desc);
        auto poolIt = reusableBuffersByHash_.find(poolKey);
        if (poolIt != reusableBuffersByHash_.end()) {
          auto &bucket = poolIt->second;
          size_t poolIndex = 0u;
          while (poolIndex < bucket.size()) {
            const ReusableBufferResource &candidate = bucket[poolIndex];
            if (!nuri::isValid(candidate.handle) ||
                !gpu.isValid(candidate.handle)) {
              bucket[poolIndex] = bucket.back();
              bucket.pop_back();
              --reusableBufferPoolSize_;
              continue;
            }
            if (!isBufferDescAliasCompatible(desc, candidate.desc)) {
              ++poolIndex;
              continue;
            }
            transientBuffer = candidate.handle;
            bucket[poolIndex] = bucket.back();
            bucket.pop_back();
            --reusableBufferPoolSize_;
            break;
          }
          if (bucket.empty()) {
            reusableBuffersByHash_.erase(poolIt);
          }
        }
      }
      if (!nuri::isValid(transientBuffer)) {
        auto createResult = gpu.createBuffer(desc, "rg_transient_buffer");
        if (createResult.hasError()) {
          destroyMaterializedResources();
          return failWithString(
              RenderGraphExecutionFailureStage::MaterializeTransients,
              "RenderGraphExecutor::execute: failed to create transient "
              "buffer: " +
                  createResult.error());
        }
        transientBuffer = createResult.value();
      }
      transientBufferHandles[allocation.allocationIndex] = transientBuffer;
    }
    NURI_PROFILER_ZONE_END();
  }
  FrameCommandArena &commands = compiled.commands;
  auto &executablePasses = commands.orderedPasses;
  auto &executableDependencyBuffers = commands.resolvedDependencyBuffers;
  auto &executableDependencyTextures = commands.resolvedDependencyTextures;
  auto &executablePreDispatches = commands.ownedPreDispatches;
  auto &executableDrawItems = commands.ownedDrawItems;
  auto &executableMeshDispatches = commands.ownedMeshDispatchItems;
  auto &executableBufferCopies = commands.ownedBufferCopyItems;
  auto &executableTextureCopies = commands.ownedTextureCopyItems;
  auto &executablePreDispatchTextureBindings =
      commands.ownedPreDispatchTextureBindings;
  auto &executableDrawTextureBindings = commands.ownedDrawTextureBindings;
  auto &executableMeshDispatchTextureBindings =
      commands.ownedMeshDispatchTextureBindings;
  auto &executablePreDispatchDependencyBuffers =
      commands.resolvedPreDispatchDependencyBuffers;
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.resolve_command_bindings",
                       NURI_PROFILER_COLOR_CMD_COPY);
    const auto resolveTextureBinding = [&](PushConstantTextureBinding
                                               &binding) {
      if (binding.graphTextureResourceIndex == UINT32_MAX) {
        return;
      }
      const uint32_t resourceIndex = binding.graphTextureResourceIndex;
      binding.texture =
          compiled.commands.textureHandlesByResource[resourceIndex];
      if (!nuri::isValid(binding.texture)) {
        binding.texture = transientTextureHandles
            [compiled.plan.transientTextureAllocationByResource[resourceIndex]];
      }
    };
    for (size_t i = 0u; i < executablePreDispatches.size(); ++i) {
      for (auto &binding : executablePreDispatchTextureBindings[i]) {
        resolveTextureBinding(binding);
      }
      executablePreDispatches[i].pushConstantTextureBindings =
          executablePreDispatchTextureBindings[i];
    }
    for (size_t i = 0u; i < executableDrawItems.size(); ++i) {
      for (auto &binding : executableDrawTextureBindings[i]) {
        resolveTextureBinding(binding);
      }
      executableDrawItems[i].pushConstantTextureBindings =
          executableDrawTextureBindings[i];
    }
    for (size_t i = 0u; i < executableMeshDispatches.size(); ++i) {
      for (auto &binding : executableMeshDispatchTextureBindings[i]) {
        resolveTextureBinding(binding);
      }
      executableMeshDispatches[i].pushConstantTextureBindings =
          executableMeshDispatchTextureBindings[i];
    }
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.patch_graph_bindings",
                       NURI_PROFILER_COLOR_CMD_COPY);
    const auto transientTexture = [&](uint32_t resourceIndex) {
      return transientTextureHandles
          [compiled.plan.transientTextureAllocationByResource[resourceIndex]];
    };
    const auto transientBuffer = [&](uint32_t resourceIndex) {
      return transientBufferHandles
          [compiled.plan.transientBufferAllocationByResource[resourceIndex]];
    };
    using PatchTarget = RenderGraphPlan::CommandResourcePatchTarget;
    for (const RenderGraphPlan::CommandResourcePatch &patch :
         compiled.plan.commandResourcePatches) {
      const auto texture = [&] {
        return transientTexture(patch.resourceIndex);
      };
      const auto buffer = [&] { return transientBuffer(patch.resourceIndex); };
      switch (patch.target) {
      case PatchTarget::PassColor:
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Texture,
                    "texture command patch has non-texture resource kind");
        executablePasses[patch.orderedPassIndex].colorTexture = texture();
        break;
      case PatchTarget::PassDepth:
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Texture,
                    "texture command patch has non-texture resource kind");
        executablePasses[patch.orderedPassIndex].depthTexture = texture();
        break;
      case PatchTarget::PassColorResolve:
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Texture,
                    "texture command patch has non-texture resource kind");
        executablePasses[patch.orderedPassIndex].colorResolveTexture =
            texture();
        break;
      case PatchTarget::PassDepthResolve:
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Texture,
                    "texture command patch has non-texture resource kind");
        executablePasses[patch.orderedPassIndex].depthResolveTexture =
            texture();
        break;
      case PatchTarget::PassDependencyBuffer: {
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Buffer,
                    "buffer command patch has non-buffer resource kind");
        const auto range =
            compiled.plan.dependencyBufferRangesByPass[patch.orderedPassIndex];
        executableDependencyBuffers[range.offset + patch.dependencyIndex] =
            buffer();
        break;
      }
      case PatchTarget::PassDependencyTexture: {
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Texture,
                    "texture command patch has non-texture resource kind");
        const auto range =
            compiled.plan.dependencyTextureRangesByPass[patch.orderedPassIndex];
        executableDependencyTextures[range.offset + patch.dependencyIndex] =
            texture();
        break;
      }
      case PatchTarget::PreDispatchDependencyBuffer: {
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Buffer,
                    "buffer command patch has non-buffer resource kind");
        const auto passRange =
            compiled.plan.preDispatchRangesByPass[patch.orderedPassIndex];
        const auto range =
            compiled.plan.preDispatchDependencyRanges[passRange.offset +
                                                      patch.commandIndex];
        executablePreDispatchDependencyBuffers[range.offset +
                                               patch.dependencyIndex] =
            buffer();
        break;
      }
      case PatchTarget::DrawVertexBuffer:
      case PatchTarget::DrawIndexBuffer:
      case PatchTarget::DrawIndirectBuffer:
      case PatchTarget::DrawIndirectCountBuffer: {
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Buffer,
                    "buffer command patch has non-buffer resource kind");
        const auto range =
            compiled.plan.drawRangesByPass[patch.orderedPassIndex];
        DrawItem &draw = executableDrawItems[range.offset + patch.commandIndex];
        const BufferHandle handle = buffer();
        if (patch.target == PatchTarget::DrawVertexBuffer)
          draw.vertexBuffer = handle;
        else if (patch.target == PatchTarget::DrawIndexBuffer)
          draw.indexBuffer = handle;
        else if (patch.target == PatchTarget::DrawIndirectBuffer)
          draw.indirectBuffer = handle;
        else
          draw.indirectCountBuffer = handle;
        break;
      }
      case PatchTarget::MeshDispatchIndirectBuffer:
      case PatchTarget::MeshDispatchIndirectCountBuffer: {
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Buffer,
                    "buffer command patch has non-buffer resource kind");
        const auto range =
            compiled.plan.meshDispatchRangesByPass[patch.orderedPassIndex];
        MeshDispatchItem &dispatch =
            executableMeshDispatches[range.offset + patch.commandIndex];
        if (patch.target == PatchTarget::MeshDispatchIndirectBuffer)
          dispatch.indirectBuffer = buffer();
        else
          dispatch.indirectCountBuffer = buffer();
        break;
      }
      case PatchTarget::BufferCopySource:
      case PatchTarget::BufferCopyDestination: {
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Buffer,
                    "buffer command patch has non-buffer resource kind");
        const auto range =
            compiled.plan.bufferCopyRangesByPass[patch.orderedPassIndex];
        BufferCopyRegion &copy =
            executableBufferCopies[range.offset + patch.commandIndex];
        if (patch.target == PatchTarget::BufferCopySource)
          copy.srcBuffer = buffer();
        else
          copy.dstBuffer = buffer();
        break;
      }
      case PatchTarget::TextureCopySource:
      case PatchTarget::TextureCopyDestination: {
        NURI_ASSERT(patch.resourceKind == RenderGraphResourceKind::Texture,
                    "texture command patch has non-texture resource kind");
        const auto range =
            compiled.plan.textureCopyRangesByPass[patch.orderedPassIndex];
        TextureCopyItem &copy =
            executableTextureCopies[range.offset + patch.commandIndex];
        if (patch.target == PatchTarget::TextureCopySource)
          copy.sourceTexture = texture();
        else
          copy.destinationTexture = texture();
        break;
      }
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  std::pmr::vector<GraphicsBarrierRecord> executableBarrierRecords(memory_);
  executableBarrierRecords.reserve(compiled.plan.passBarrierRecords.size());
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.resolve_barriers",
                       NURI_PROFILER_COLOR_BARRIER);
    for (const RenderGraphBarrierRecord &barrier :
         compiled.plan.passBarrierRecords) {
      if (barrier.resourceKind == RenderGraphBarrierResourceKind::Texture) {
        TextureHandle texture =
            compiled.commands.textureHandlesByResource[barrier.resourceIndex];
        if (!nuri::isValid(texture)) {
          texture =
              transientTextureHandles[compiled.plan
                                          .transientTextureAllocationByResource
                                              [barrier.resourceIndex]];
        }
        executableBarrierRecords.push_back(GraphicsBarrierRecord::ForTexture(
            texture, barrier.beforeAccess, barrier.afterAccess,
            barrier.beforeState, barrier.afterState, barrier.subresources));
      } else if (barrier.resourceKind ==
                 RenderGraphBarrierResourceKind::Buffer) {
        BufferHandle buffer =
            compiled.commands.bufferHandlesByResource[barrier.resourceIndex];
        if (!nuri::isValid(buffer)) {
          buffer = transientBufferHandles
              [compiled.plan
                   .transientBufferAllocationByResource[barrier.resourceIndex]];
        }
        executableBarrierRecords.push_back(GraphicsBarrierRecord::ForBuffer(
            buffer, barrier.beforeAccess, barrier.afterAccess,
            barrier.beforeState, barrier.afterState));
      } else {
        const AccelerationStructureHandle accelerationStructure =
            compiled.commands
                .accelerationStructureHandlesByResource[barrier.resourceIndex];
        executableBarrierRecords.push_back(
            GraphicsBarrierRecord::ForAccelerationStructure(
                accelerationStructure, barrier.beforeAccess,
                barrier.afterAccess, barrier.beforeState, barrier.afterState));
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  Result<bool, std::string> submitResult =
      Result<bool, std::string>::makeResult(true);
  SubmissionHandle frameSubmission{};
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.submit_frame",
                       NURI_PROFILER_COLOR_SUBMIT);
    if (captureTelemetry) {
      metadata.usedParallelCompile = compiled.plan.usedParallelCompile;
      metadata.usedParallelRecording = false;
      metadata.recordedCommandBuffers.clear();
      metadata.submitBatches.clear();
      metadata.passRanges.clear();
      metadata.passTimings.clear();
    }
    if (!executablePasses.empty()) {
      {
        Result<bool, std::string> prepareFrameOutputResult =
            Result<bool, std::string>::makeResult(true);
        NURI_PROFILER_ZONE("RenderGraph.execute.prepare_frame_output",
                           NURI_PROFILER_COLOR_WAIT);
        prepareFrameOutputResult = gpu.prepareFrameOutput();
        NURI_PROFILER_ZONE_END();
        if (prepareFrameOutputResult.hasError()) {
          destroyMaterializedResources();
          return failWithString(
              RenderGraphExecutionFailureStage::SubmitRecordedFrame,
              "RenderGraphExecutor::execute: failed to prepare frame "
              "output: " +
                  prepareFrameOutputResult.error());
        }
      }
      const bool supportsParallelRecording =
          runtime != nullptr && runtime->parallelGraphicsRecordingEnabled() &&
          gpu.supportsParallelGraphicsRecording() &&
          gpu.maxParallelGraphicsRecordingContexts() > 1u &&
          executablePasses.size() > 1u;
      const uint32_t maxContextCount =
          supportsParallelRecording
              ? std::min(
                    std::min(runtime->workerCount(),
                             static_cast<uint32_t>(executablePasses.size())),
                    gpu.maxParallelGraphicsRecordingContexts())
              : 1u;
      std::pmr::vector<RenderGraphContiguousRange> ranges(memory_);
      {
        NURI_PROFILER_ZONE("RenderGraph.execute.schedule_recording_ranges",
                           NURI_PROFILER_COLOR_CMD_COPY);
        const std::vector<RenderGraphContiguousRange> stdRanges =
            makeRecordingRanges(static_cast<uint32_t>(executablePasses.size()),
                                maxContextCount);
        ranges.assign(stdRanges.begin(), stdRanges.end());
        NURI_PROFILER_ZONE_END();
      }
      if (captureTelemetry) {
        metadata.usedParallelRecording =
            supportsParallelRecording && ranges.size() > 1u;
        metadata.passRanges.reserve(ranges.size());
        if (capturePassTimings) {
          metadata.passTimings.resize(executablePasses.size());
          for (uint32_t orderedPassIndex = 0u;
               orderedPassIndex < metadata.passTimings.size();
               ++orderedPassIndex) {
            metadata.passTimings[orderedPassIndex].orderedPassIndex =
                orderedPassIndex;
          }
        }
        for (uint32_t workerIndex = 0u; workerIndex < ranges.size();
             ++workerIndex) {
          metadata.passRanges.push_back(RenderGraphPassRange{
              .workerIndex = workerIndex,
              .firstOrderedPassIndex = ranges[workerIndex].count > 0u
                                           ? ranges[workerIndex].offset
                                           : UINT32_MAX,
              .passCount = ranges[workerIndex].count,
          });
        }
      }
      std::vector<RecordingReferenceSet> recordingReferences;
      recordingReferences.reserve(ranges.size());
      for (const RenderGraphContiguousRange range : ranges) {
        RecordingReferenceSet &references =
            recordingReferences.emplace_back(memory_);
        for (uint32_t localIndex = 0u; localIndex < range.count; ++localIndex) {
          const uint32_t orderedPassIndex = range.offset + localIndex;
          appendRecordingReferences(references,
                                    executablePasses[orderedPassIndex]);
          const PassBarrierPlan &barrierPlan =
              compiled.plan.passBarrierPlans[orderedPassIndex];
          appendRecordingReferences(
              references,
              std::span<const GraphicsBarrierRecord>(executableBarrierRecords)
                  .subspan(barrierPlan.barrierOffset,
                           barrierPlan.barrierCount));
        }
        if (range.offset + range.count == executablePasses.size()) {
          appendRecordingReferences(
              references,
              std::span<const GraphicsBarrierRecord>(executableBarrierRecords)
                  .subspan(compiled.plan.finalBarrierPlan.barrierOffset,
                           compiled.plan.finalBarrierPlan.barrierCount));
        }
        references.normalize();
      }
      std::pmr::vector<RecordingContextHandle> recordingContexts(memory_);
      recordingContexts.resize(ranges.size());
      std::atomic<bool> recordingFailed = false;
      std::string recordingError{};
      std::mutex recordingFailureMutex{};
      const auto setRecordingFailure = [&](std::string message) {
        bool expected = false;
        if (!recordingFailed.compare_exchange_strong(expected, true)) {
          return;
        }
        std::lock_guard lock(recordingFailureMutex);
        recordingError = std::move(message);
      };
      const auto recordRange = [&](uint32_t workerIndex,
                                   RenderGraphContiguousRange range) {
        NURI_PROFILER_ZONE("RenderGraph.execute.record_graphics_range",
                           NURI_PROFILER_COLOR_CMD_COPY);
        if (range.count == 0u || recordingFailed.load()) {
          return;
        }
        auto contextResult = gpu.acquireGraphicsRecordingContext(workerIndex);
        if (contextResult.hasError()) {
          setRecordingFailure(makeExecutionStageError(
              RenderGraphExecutionFailureStage::AcquireRecordingContext,
              "RenderGraphExecutor::execute: failed to acquire graphics "
              "recording context: " +
                  contextResult.error()));
          return;
        }
        recordingContexts[workerIndex] = contextResult.value();
        auto retainResult = gpu.retainGraphicsRecordingReferences(
            recordingContexts[workerIndex],
            recordingReferences[workerIndex].view());
        if (retainResult.hasError()) {
          setRecordingFailure(makeExecutionStageError(
              RenderGraphExecutionFailureStage::RetainRecordingReferences,
              "RenderGraphExecutor::execute: failed to retain recording "
              "references: " +
                  retainResult.error()));
          return;
        }
        if (!capturePassTimings) {
          std::pmr::vector<GraphicsRecordingStep> steps(memory_);
          steps.reserve(range.count + 1u);
          for (uint32_t localIndex = 0u; localIndex < range.count;
               ++localIndex) {
            const uint32_t orderedPassIndex = range.offset + localIndex;
            const PassBarrierPlan &barrierPlan =
                compiled.plan.passBarrierPlans[orderedPassIndex];
            steps.push_back(GraphicsRecordingStep{
                .barriers = std::span<const GraphicsBarrierRecord>(
                                executableBarrierRecords)
                                .subspan(barrierPlan.barrierOffset,
                                         barrierPlan.barrierCount),
                .pass = &executablePasses[orderedPassIndex]});
          }
          if (range.offset + range.count == executablePasses.size() &&
              compiled.plan.finalBarrierPlan.barrierCount > 0u) {
            steps.push_back(GraphicsRecordingStep{
                .barriers =
                    std::span<const GraphicsBarrierRecord>(
                        executableBarrierRecords)
                        .subspan(compiled.plan.finalBarrierPlan.barrierOffset,
                                 compiled.plan.finalBarrierPlan.barrierCount)});
          }
          auto recordResult =
              gpu.recordGraphicsRange(recordingContexts[workerIndex], steps);
          if (recordResult.hasError()) {
            setRecordingFailure(makeExecutionStageError(
                RenderGraphExecutionFailureStage::RecordGraphicsPasses,
                "RenderGraphExecutor::execute: failed to record graphics "
                "range: " +
                    recordResult.error()));
          }
          return;
        }
        for (uint32_t localIndex = 0u; localIndex < range.count; ++localIndex) {
          if (recordingFailed.load()) {
            return;
          }
          const uint32_t orderedPassIndex = range.offset + localIndex;
          const PassBarrierPlan &barrierPlan =
              compiled.plan.passBarrierPlans[orderedPassIndex];
          if (barrierPlan.barrierCount > 0u) {
            auto barrierResult = gpu.recordGraphicsBarriers(
                recordingContexts[workerIndex],
                std::span<const GraphicsBarrierRecord>(executableBarrierRecords)
                    .subspan(barrierPlan.barrierOffset,
                             barrierPlan.barrierCount));
            if (barrierResult.hasError()) {
              const std::string_view passLabel =
                  executablePasses[orderedPassIndex].debugLabel;
              setRecordingFailure(makeExecutionStageError(
                  RenderGraphExecutionFailureStage::RecordGraphicsBarriers,
                  "RenderGraphExecutor::execute: failed to record graphics "
                  "barriers for ordered pass " +
                      std::to_string(orderedPassIndex) + " ('" +
                      std::string(passLabel) + "'): " + barrierResult.error()));
              return;
            }
          }
          std::chrono::steady_clock::time_point passRecordStart{};
          if (capturePassTimings) {
            passRecordStart = std::chrono::steady_clock::now();
          }
          auto recordResult =
              gpu.recordGraphicsPass(recordingContexts[workerIndex],
                                     executablePasses[orderedPassIndex]);
          if (capturePassTimings) {
            const auto passRecordEnd = std::chrono::steady_clock::now();
            metadata.passTimings[orderedPassIndex].cpuTimeMs =
                std::chrono::duration<float, std::milli>(passRecordEnd -
                                                         passRecordStart)
                    .count();
          }
          if (recordResult.hasError()) {
            setRecordingFailure(makeExecutionStageError(
                RenderGraphExecutionFailureStage::RecordGraphicsPasses,
                "RenderGraphExecutor::execute: failed to record graphics "
                "pass: " +
                    recordResult.error()));
            return;
          }
        }
        if (range.offset + range.count == executablePasses.size() &&
            compiled.plan.finalBarrierPlan.barrierCount > 0u) {
          auto finalBarrierResult = gpu.recordGraphicsBarriers(
              recordingContexts[workerIndex],
              std::span<const GraphicsBarrierRecord>(executableBarrierRecords)
                  .subspan(compiled.plan.finalBarrierPlan.barrierOffset,
                           compiled.plan.finalBarrierPlan.barrierCount));
          if (finalBarrierResult.hasError()) {
            setRecordingFailure(makeExecutionStageError(
                RenderGraphExecutionFailureStage::RecordGraphicsBarriers,
                "RenderGraphExecutor::execute: failed to record final "
                "graphics barriers: " +
                    finalBarrierResult.error()));
            return;
          }
        }
        NURI_PROFILER_ZONE_END();
      };
      {
        NURI_PROFILER_ZONE("RenderGraph.execute.record_graphics_ranges",
                           NURI_PROFILER_COLOR_CMD_COPY);
        if (supportsParallelRecording && ranges.size() > 1u) {
          runtime->runRanges(std::span<const RenderGraphContiguousRange>(
                                 ranges.data(), ranges.size()),
                             recordRange);
        } else {
          for (uint32_t workerIndex = 0u; workerIndex < ranges.size();
               ++workerIndex) {
            recordRange(workerIndex, ranges[workerIndex]);
            if (recordingFailed.load()) {
              break;
            }
          }
        }
        NURI_PROFILER_ZONE_END();
      }
      if (recordingFailed.load()) {
        for (const RecordingContextHandle ctx : recordingContexts) {
          if (!nuri::isValid(ctx)) {
            continue;
          }
          gpu.discardGraphicsRecordingContext(ctx);
        }
        destroyMaterializedResources();
        return Result<bool, std::string>::makeError(recordingError);
      }
      std::pmr::vector<RecordedCommandBufferHandle> recordedCommandBuffers(
          memory_);
      recordedCommandBuffers.reserve(ranges.size());
      {
        NURI_PROFILER_ZONE("RenderGraph.execute.finish_recording_contexts",
                           NURI_PROFILER_COLOR_CMD_COPY);
        for (uint32_t workerIndex = 0u; workerIndex < ranges.size();
             ++workerIndex) {
          const RenderGraphContiguousRange range = ranges[workerIndex];
          if (range.count == 0u) {
            continue;
          }
          auto finishResult = gpu.finishGraphicsRecordingContext(
              recordingContexts[workerIndex]);
          if (finishResult.hasError()) {
            for (const RecordedCommandBufferHandle handle :
                 recordedCommandBuffers) {
              if (nuri::isValid(handle)) {
                gpu.discardRecordedGraphicsCommandBuffer(handle);
              }
            }
            for (uint32_t discardIndex = workerIndex;
                 discardIndex < ranges.size(); ++discardIndex) {
              if (nuri::isValid(recordingContexts[discardIndex])) {
                gpu.discardGraphicsRecordingContext(
                    recordingContexts[discardIndex]);
              }
            }
            destroyMaterializedResources();
            return failWithString(
                RenderGraphExecutionFailureStage::FinishRecordingContext,
                "RenderGraphExecutor::execute: failed to finish graphics "
                "recording context: " +
                    finishResult.error());
          }
          recordingContexts[workerIndex] = {};
          recordedCommandBuffers.push_back(finishResult.value());
          if (captureTelemetry) {
            metadata.recordedCommandBuffers.push_back(RecordedCommandBufferMeta{
                .firstOrderedPassIndex = range.offset,
                .passCount = range.count,
            });
          }
        }
        NURI_PROFILER_ZONE_END();
      }
      std::pmr::vector<SubmitBatchMeta> batches(memory_);
      {
        NURI_PROFILER_ZONE("RenderGraph.execute.build_submit_batches",
                           NURI_PROFILER_COLOR_SUBMIT);
        batches.push_back(SubmitBatchMeta{
            .commandBufferOffset = 0u,
            .commandBufferCount =
                static_cast<uint32_t>(recordedCommandBuffers.size()),
            .presentsFrameOutput = true,
        });
        if (captureTelemetry) {
          metadata.submitBatches.assign(batches.begin(), batches.end());
        }
        NURI_PROFILER_ZONE_END();
      }
      {
        NURI_PROFILER_ZONE("RenderGraph.execute.submit_recorded_frame",
                           NURI_PROFILER_COLOR_SUBMIT);
        auto submitFrameResult = gpu.submitRecordedGraphicsFrame(
            std::span<const RecordedCommandBufferHandle>(
                recordedCommandBuffers.data(), recordedCommandBuffers.size()),
            std::span<const SubmitBatchMeta>(batches.data(), batches.size()));
        if (submitFrameResult.hasError()) {
          submitResult = failWithString(
              RenderGraphExecutionFailureStage::SubmitRecordedFrame,
              "RenderGraphExecutor::execute: failed to submit recorded "
              "graphics "
              "frame: " +
                  submitFrameResult.error());
        } else {
          SubmittedGraphicsFrame submitted =
              std::move(submitFrameResult.value());
          frameSubmission = submitted.submission;
          metadata.submission = frameSubmission;
          if (!submitted.presentationError.empty()) {
            submitResult = failWithString(
                RenderGraphExecutionFailureStage::PresentFrameOutput,
                "RenderGraphExecutor::execute: failed to present frame "
                "output: " +
                    submitted.presentationError);
          }
        }
        NURI_PROFILER_ZONE_END();
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.defer_transient_retire",
                       NURI_PROFILER_COLOR_DESTROY);
    PendingFrameResources pending(memory_);
    pending.submission = frameSubmission;
    for (size_t textureSlot = 0u; textureSlot < transientTextureHandles.size();
         ++textureSlot) {
      const TextureHandle texture = transientTextureHandles[textureSlot];
      if (nuri::isValid(texture)) {
        pending.textures.push_back(texture);
        pending.textureDescs.push_back(transientTextureDescs[textureSlot]);
      }
    }
    for (size_t bufferSlot = 0u; bufferSlot < transientBufferHandles.size();
         ++bufferSlot) {
      const BufferHandle buffer = transientBufferHandles[bufferSlot];
      if (nuri::isValid(buffer)) {
        pending.buffers.push_back(buffer);
        pending.bufferDescs.push_back(transientBufferDescs[bufferSlot]);
      }
    }
    if (!pending.textures.empty() || !pending.buffers.empty()) {
      pendingFrames_.push_back(std::move(pending));
    }
    NURI_PROFILER_ZONE_END();
  }
  return submitResult;
}

} // namespace nuri
