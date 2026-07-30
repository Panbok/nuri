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
  std::pmr::vector<SamplerHandle> samplers;
  std::pmr::vector<AccelerationStructureHandle> accelerationStructures;
  std::pmr::vector<RenderPipelineHandle> renderPipelines;
  std::pmr::vector<ComputePipelineHandle> computePipelines;
  std::pmr::vector<MeshletPipelineHandle> meshletPipelines;
  std::pmr::vector<RayQueryBindingHandle> rayQueryBindings;

  explicit RecordingReferenceSet(std::pmr::memory_resource *memory)
      : buffers(memory), textures(memory), samplers(memory),
        accelerationStructures(memory), renderPipelines(memory),
        computePipelines(memory), meshletPipelines(memory),
        rayQueryBindings(memory) {}

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
    makeUnique(samplers);
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
        .samplers = samplers,
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
  RecordingReferenceSet::append(out.samplers, pass.recordingSamplers);
  for (const ComputeDispatchItem &dispatch : pass.preDispatches) {
    RecordingReferenceSet::append(out.computePipelines, dispatch.pipeline);
    RecordingReferenceSet::append(out.rayQueryBindings,
                                  dispatch.rayQueryBinding);
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
[[nodiscard]] uint64_t computePassPayloadLayoutHash(
    const std::pmr::vector<RenderPass> &passes) noexcept {
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
    mix(pass.drawBuffersPreResolved ? 1u : 0u);
    mix(static_cast<uint64_t>(pass.gpuTimingScope));
    mix(static_cast<uint64_t>(pass.recordingSamplers.size()));
    mix(static_cast<uint64_t>(pass.preDispatches.size()));
    mix(quantizeToNextPow2(static_cast<uint64_t>(pass.draws.size())));
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
      currentCommands_(memory_), passes_(memory_), passDebugNames_(memory_),
      passBindings_(memory_), drawBindings_(memory_),
      meshDispatchBindings_(memory_), bufferCopyBindings_(memory_),
      textureCopyBindings_(memory_), importedTextureIndicesByHandle_(memory_),
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
  currentCommands_ = FrameCommandArena(memory_);
  currentCommands_.frameIndex = frameIndex;
  commandsTransferred_ = false;
  passes_.clear();
  passDebugNames_.clear();
  passBindings_.clear();
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
  uint64_t payloadLayoutHash = computePassPayloadLayoutHash(passes_);
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
      .payloadLayoutHash = payloadLayoutHash,
      .structuralIdentityHash = structuralIdentityHash,
      .transientResourceDescriptorsHash = transientResourceDescriptorsHash_,
      .persistentHandlesVersion = persistentHandlesVersion_,
  };
}

FrameCommandArena
RenderGraphBuilder::buildFrameCommands(const RenderGraphPlan &plan) {
  if (commandsTransferred_) {
    FrameCommandArena commands(memory_);
    commands.frameIndex = frameIndex_;
    return commands;
  }
  refreshPassViews();
  FrameCommandArena commands = std::move(currentCommands_);
  commandsTransferred_ = true;
  commands.frameIndex = frameIndex_;
  commands.textureHandlesByResource.resize(textures_.size());
  commands.bufferHandlesByResource.resize(buffers_.size());
  commands.accelerationStructureHandlesByResource.resize(
      accelerationStructures_.size());
  commands.orderedPasses.resize(plan.orderedPassIndices.size());
  commands.passDebugNames = std::move(passDebugNames_);
  for (size_t i = 0; i < textures_.size(); ++i) {
    if (textures_[i].imported) {
      commands.textureHandlesByResource[i] = textures_[i].importedHandle;
    }
  }
  for (size_t i = 0; i < buffers_.size(); ++i) {
    if (buffers_[i].imported) {
      commands.bufferHandlesByResource[i] = buffers_[i].importedHandle;
    }
  }
  for (size_t i = 0; i < accelerationStructures_.size(); ++i) {
    commands.accelerationStructureHandlesByResource[i] =
        accelerationStructures_[i].importedHandle;
  }
  for (size_t i = 0u; i < commands.ownedPreDispatches.size(); ++i) {
    auto &command = commands.ownedPreDispatches[i];
    command.debugLabel = commands.ownedPreDispatchDebugLabels[i];
    command.pushConstants = commands.ownedPreDispatchPushConstants[i];
    command.pushConstantTextureBindings =
        commands.ownedPreDispatchTextureBindings[i];
  }
  for (size_t i = 0u; i < commands.ownedDrawItems.size(); ++i) {
    auto &command = commands.ownedDrawItems[i];
    command.debugLabel = commands.ownedDrawDebugLabels[i];
    command.pushConstants = commands.ownedDrawPushConstants[i];
    command.pushConstantTextureBindings = commands.ownedDrawTextureBindings[i];
  }
  for (size_t i = 0u; i < commands.ownedMeshDispatchItems.size(); ++i) {
    auto &command = commands.ownedMeshDispatchItems[i];
    command.debugLabel = commands.ownedMeshDispatchDebugLabels[i];
    command.pushConstants = commands.ownedMeshDispatchPushConstants[i];
    command.pushConstantTextureBindings =
        commands.ownedMeshDispatchTextureBindings[i];
  }
  const auto view = [](const auto &storage, BindingRange range) {
    using Element = typename std::decay_t<decltype(storage)>::value_type;
    return range.count == 0u ? std::span<const Element>{}
                             : std::span<const Element>(
                                   storage.data() + range.offset, range.count);
  };
  const auto resolveTexture = [&](uint32_t resource) {
    return resource != UINT32_MAX && textures_[resource].imported
               ? textures_[resource].importedHandle
               : TextureHandle{};
  };
  const auto resolveBuffer = [&](uint32_t resource) {
    return resource != UINT32_MAX && buffers_[resource].imported
               ? buffers_[resource].importedHandle
               : BufferHandle{};
  };
  for (size_t i = 0; i < commands.orderedPasses.size(); ++i) {
    const uint32_t passIndex = plan.orderedPassIndices[i];
    const RenderPass &sourcePass = passes_[passIndex];
    const PassBindings &bindings = passBindings_[passIndex];
    RenderPass &refreshedPass = commands.orderedPasses[i] = sourcePass;
    refreshedPass.colorTexture = resolveTexture(bindings.color);
    refreshedPass.colorResolveTexture = resolveTexture(bindings.colorResolve);
    refreshedPass.depthTexture = resolveTexture(bindings.depth);
    refreshedPass.depthResolveTexture = resolveTexture(bindings.depthResolve);
    refreshedPass.preDispatches =
        view(commands.ownedPreDispatches, bindings.preDispatches);
    refreshedPass.draws = view(commands.ownedDrawItems, bindings.drawPayloads);
    refreshedPass.meshDispatches =
        view(commands.ownedMeshDispatchItems, bindings.meshDispatches);
    refreshedPass.bufferCopies =
        view(commands.ownedBufferCopyItems, bindings.bufferCopies);
    refreshedPass.textureCopies =
        view(commands.ownedTextureCopyItems, bindings.textureCopies);
    refreshedPass.recordingSamplers =
        passIndex < commands.ownedRecordingSamplersByPass.size()
            ? std::span<const SamplerHandle>(
                  commands.ownedRecordingSamplersByPass[passIndex])
            : std::span<const SamplerHandle>{};
    refreshedPass.accelerationStructureBuilds =
        passIndex < commands.ownedAccelerationStructureBuildsByPass.size()
            ? std::span<const AccelerationStructureBuildItem>(
                  commands.ownedAccelerationStructureBuildsByPass[passIndex])
            : std::span<const AccelerationStructureBuildItem>{};
    for (uint32_t drawIndex = 0u; drawIndex < bindings.draws.count;
         ++drawIndex) {
      DrawItem &draw =
          commands.ownedDrawItems[bindings.drawPayloads.offset + drawIndex];
      const DrawBindings binding =
          drawBindings_[bindings.draws.offset + drawIndex];
      draw.vertexBuffer = resolveBuffer(binding.vertex);
      draw.indexBuffer = resolveBuffer(binding.index);
      draw.indirectBuffer = resolveBuffer(binding.indirect);
      draw.indirectCountBuffer = resolveBuffer(binding.indirectCount);
    }
    for (uint32_t commandIndex = 0u;
         commandIndex < bindings.meshDispatches.count; ++commandIndex) {
      MeshDispatchItem &dispatch =
          commands.ownedMeshDispatchItems[bindings.meshDispatches.offset +
                                          commandIndex];
      const MeshDispatchBindings binding =
          meshDispatchBindings_[bindings.meshDispatches.offset + commandIndex];
      dispatch.indirectBuffer = resolveBuffer(binding.indirect);
      dispatch.indirectCountBuffer = resolveBuffer(binding.indirectCount);
    }
    for (uint32_t commandIndex = 0u; commandIndex < bindings.bufferCopies.count;
         ++commandIndex) {
      BufferCopyRegion &copy =
          commands.ownedBufferCopyItems[bindings.bufferCopies.offset +
                                        commandIndex];
      const BufferCopyBindings binding =
          bufferCopyBindings_[bindings.bufferCopies.offset + commandIndex];
      copy.srcBuffer = resolveBuffer(binding.source);
      copy.dstBuffer = resolveBuffer(binding.destination);
    }
    for (uint32_t commandIndex = 0u;
         commandIndex < bindings.textureCopies.count; ++commandIndex) {
      TextureCopyItem &copy =
          commands.ownedTextureCopyItems[bindings.textureCopies.offset +
                                         commandIndex];
      const TextureCopyBindings binding =
          textureCopyBindings_[bindings.textureCopies.offset + commandIndex];
      copy.sourceTexture = resolveTexture(binding.source);
      copy.destinationTexture = resolveTexture(binding.destination);
    }
    const std::pmr::string &name = commands.passDebugNames[passIndex];
    refreshedPass.debugLabel = std::string_view(name.data(), name.size());
  }
  return commands;
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

Result<bool, std::string> RenderGraphBuilder::addImportedTextureAccess(
    RenderGraphPassId pass, TextureHandle texture, RenderGraphAccessMode mode,
    std::string_view debugName) {
  auto imported = importTexture(texture, debugName);
  if (imported.hasError()) {
    return Result<bool, std::string>::makeError(imported.error());
  }
  return addTextureAccessInternal(pass, imported.value(), mode, false);
}

Result<bool, std::string> RenderGraphBuilder::addImportedBufferAccess(
    RenderGraphPassId pass, BufferHandle buffer, RenderGraphAccessMode mode,
    std::string_view debugName) {
  auto imported = importBuffer(buffer, debugName);
  if (imported.hasError()) {
    return Result<bool, std::string>::makeError(imported.error());
  }
  return addBufferAccessInternal(pass, imported.value(), mode, false);
}

Result<bool, std::string> RenderGraphBuilder::addImportedTextureReads(
    RenderGraphPassId pass, std::span<const TextureHandle> textures,
    std::string_view debugName) {
  for (TextureHandle texture : textures) {
    auto result = addImportedTextureAccess(
        pass, texture, RenderGraphAccessMode::Read, debugName);
    if (result.hasError()) {
      return result;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RenderGraphBuilder::addImportedBufferReads(
    RenderGraphPassId pass, std::span<const BufferHandle> buffers,
    std::string_view debugName) {
  for (BufferHandle buffer : buffers) {
    auto result = addImportedBufferAccess(
        pass, buffer, RenderGraphAccessMode::Read, debugName);
    if (result.hasError()) {
      return result;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RenderGraphBuilder::addImportedTextureAccesses(
    RenderGraphPassId pass, std::span<const RenderGraphImportedTextureUse> uses,
    std::string_view debugName) {
  for (const RenderGraphImportedTextureUse use : uses) {
    auto result =
        addImportedTextureAccess(pass, use.texture, use.access, debugName);
    if (result.hasError()) {
      return result;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> RenderGraphBuilder::addImportedBufferAccesses(
    RenderGraphPassId pass, std::span<const RenderGraphImportedBufferUse> uses,
    std::string_view debugName) {
  for (const RenderGraphImportedBufferUse use : uses) {
    auto result =
        addImportedBufferAccess(pass, use.buffer, use.access, debugName);
    if (result.hasError()) {
      return result;
    }
  }
  return Result<bool, std::string>::makeResult(true);
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

void RenderGraphBuilder::appendGraphicsPayload(
    const RenderGraphGraphicsPassDesc &desc, RenderPass &pass) {
  const size_t passIndex = passes_.size();
  currentCommands_.ownedRecordingSamplersByPass.resize(passIndex + 1u);
  auto &samplers = currentCommands_.ownedRecordingSamplersByPass[passIndex];
  samplers.assign(desc.recordingSamplers.begin(), desc.recordingSamplers.end());
  pass.recordingSamplers = samplers;

  const size_t dispatchOffset = currentCommands_.ownedPreDispatches.size();
  const size_t dispatchEnd = dispatchOffset + desc.preDispatches.size();
  currentCommands_.ownedPreDispatches.resize(dispatchEnd);
  currentCommands_.ownedPreDispatchDebugLabels.resize(dispatchEnd);
  currentCommands_.ownedPreDispatchPushConstants.resize(dispatchEnd);
  currentCommands_.ownedPreDispatchTextureBindings.resize(dispatchEnd);
  for (size_t i = 0; i < desc.preDispatches.size(); ++i) {
    ComputeDispatchItem command = desc.preDispatches[i];
    ownPreDispatchPayload(currentCommands_, dispatchOffset + i,
                          desc.preDispatches[i], command);
    currentCommands_.ownedPreDispatches[dispatchOffset + i] = command;
  }
  pass.preDispatches =
      desc.preDispatches.empty()
          ? std::span<const ComputeDispatchItem>{}
          : std::span<const ComputeDispatchItem>(
                currentCommands_.ownedPreDispatches.data() + dispatchOffset,
                desc.preDispatches.size());

  const size_t drawOffset = currentCommands_.ownedDrawItems.size();
  const size_t drawEnd = drawOffset + desc.draws.size();
  currentCommands_.ownedDrawItems.resize(drawEnd);
  currentCommands_.ownedDrawDebugLabels.resize(drawEnd);
  currentCommands_.ownedDrawPushConstants.resize(drawEnd);
  currentCommands_.ownedDrawTextureBindings.resize(drawEnd);
  for (size_t i = 0; i < desc.draws.size(); ++i) {
    DrawItem command = desc.draws[i];
    ownDrawPayload(currentCommands_, drawOffset + i, desc.draws[i], command);
    currentCommands_.ownedDrawItems[drawOffset + i] = command;
  }
  pass.draws = desc.draws.empty()
                   ? std::span<const DrawItem>{}
                   : std::span<const DrawItem>(
                         currentCommands_.ownedDrawItems.data() + drawOffset,
                         desc.draws.size());

  const size_t meshOffset = currentCommands_.ownedMeshDispatchItems.size();
  const size_t meshEnd = meshOffset + desc.meshDispatches.size();
  currentCommands_.ownedMeshDispatchItems.resize(meshEnd);
  currentCommands_.ownedMeshDispatchDebugLabels.resize(meshEnd);
  currentCommands_.ownedMeshDispatchPushConstants.resize(meshEnd);
  currentCommands_.ownedMeshDispatchTextureBindings.resize(meshEnd);
  for (size_t i = 0; i < desc.meshDispatches.size(); ++i) {
    MeshDispatchItem command = desc.meshDispatches[i];
    ownMeshDispatchPayload(currentCommands_, meshOffset + i,
                           desc.meshDispatches[i], command);
    currentCommands_.ownedMeshDispatchItems[meshOffset + i] = command;
  }
  pass.meshDispatches =
      desc.meshDispatches.empty()
          ? std::span<const MeshDispatchItem>{}
          : std::span<const MeshDispatchItem>(
                currentCommands_.ownedMeshDispatchItems.data() + meshOffset,
                desc.meshDispatches.size());
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
  for (const RenderGraphBufferUse use : desc.bufferUses) {
    if (!isValidBufferIndex(use.buffer.value)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addGraphicsPass: buffer use is out of range");
    }
  }
  for (const RenderGraphImportedBufferUse use : desc.importedBufferUses) {
    if (!nuri::isValid(use.buffer)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addGraphicsPass: imported buffer use is "
          "invalid");
    }
  }
  for (const RenderGraphTextureUse use : desc.textureUses) {
    if (!isValidTextureIndex(use.texture.value)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addGraphicsPass: texture use is out of range");
    }
  }
  for (const RenderGraphImportedTextureUse use : desc.importedTextureUses) {
    if (!nuri::isValid(use.texture)) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addGraphicsPass: imported texture use is "
          "invalid");
    }
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
  pass.drawBuffersPreResolved = desc.drawBuffersPreResolved;
  appendGraphicsPayload(desc, pass);
  pass.debugLabel = desc.debugLabel;
  auto addResult = addPassRecord(pass, desc.debugLabel);
  if (addResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();
  refreshPassViews();
  auto bindResourcesResult = bindImplicitPassResources(passId, desc);
  if (bindResourcesResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(
        bindResourcesResult.error());
  }
  for (uint32_t i = 0u; i < desc.bufferUses.size(); ++i) {
    const RenderGraphBufferUse use = desc.bufferUses[i];
    auto bindResult =
        addBufferAccessInternal(passId, use.buffer, use.access, false);
    if (bindResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          bindResult.error());
    }
  }
  for (uint32_t i = 0u; i < desc.textureUses.size(); ++i) {
    const RenderGraphTextureUse use = desc.textureUses[i];
    auto bindResult =
        addTextureAccessInternal(passId, use.texture, use.access, false);
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
        addBufferAccessInternal(passId, imported.value(), use.access, false);
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
        addTextureAccessInternal(passId, imported.value(), use.access, false);
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

  RenderPass pass{};
  pass.executionMode = RenderPassExecutionMode::AccelerationStructureBuild;
  pass.hasColorAttachment = false;
  pass.gpuTimingScope = desc.gpuTimingScope;
  pass.debugLabel = desc.debugLabel;
  pass.debugColor = desc.debugColor;
  const size_t passIndex = passes_.size();
  currentCommands_.ownedAccelerationStructureBuildsByPass.resize(passIndex +
                                                                 1u);
  currentCommands_.ownedAccelerationStructureGeometriesByPass.resize(passIndex +
                                                                     1u);
  currentCommands_.ownedAccelerationStructureInstancesByPass.resize(passIndex +
                                                                    1u);
  RenderPass source = pass;
  source.accelerationStructureBuilds = desc.builds;
  ownAccelerationStructureBuilds(currentCommands_, passIndex, source, pass);
  auto addResult = addPassRecord(pass, desc.debugLabel);
  if (addResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();
  refreshPassViews();
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
  }
  const size_t copyOffset = currentCommands_.ownedTextureCopyItems.size();
  currentCommands_.ownedTextureCopyItems.resize(copyOffset +
                                                desc.copies.size());
  for (size_t i = 0u; i < desc.copies.size(); ++i) {
    const RenderGraphTextureCopyItem &copy = desc.copies[i];
    currentCommands_.ownedTextureCopyItems[copyOffset + i] =
        TextureCopyItem{.sourceTexture = {},
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
                        .destinationLayer = copy.destinationLayer};
  }
  RenderPass pass{};
  pass.executionMode = RenderPassExecutionMode::CopyOnly;
  pass.hasColorAttachment = false;
  pass.textureCopies = std::span<const TextureCopyItem>(
      currentCommands_.ownedTextureCopyItems.data() + copyOffset,
      desc.copies.size());
  pass.gpuTimingScope = desc.gpuTimingScope;
  pass.debugColor = desc.debugColor;
  pass.debugLabel = desc.debugLabel;
  auto addResult = addPassRecord(pass, desc.debugLabel);
  if (addResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();
  refreshPassViews();
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
  }
  const size_t copyOffset = currentCommands_.ownedBufferCopyItems.size();
  currentCommands_.ownedBufferCopyItems.resize(copyOffset + desc.copies.size());
  for (size_t i = 0u; i < desc.copies.size(); ++i) {
    const RenderGraphBufferCopyItem &copy = desc.copies[i];
    currentCommands_.ownedBufferCopyItems[copyOffset + i] =
        BufferCopyRegion{.srcBuffer = {},
                         .dstBuffer = {},
                         .srcOffset = copy.sourceOffset,
                         .dstOffset = copy.destinationOffset,
                         .size = copy.size};
  }
  RenderPass pass{};
  pass.executionMode = RenderPassExecutionMode::CopyOnly;
  pass.hasColorAttachment = false;
  pass.bufferCopies = std::span<const BufferCopyRegion>(
      currentCommands_.ownedBufferCopyItems.data() + copyOffset,
      desc.copies.size());
  pass.gpuTimingScope = desc.gpuTimingScope;
  pass.debugColor = desc.debugColor;
  pass.debugLabel = desc.debugLabel;
  auto addResult = addPassRecord(pass, desc.debugLabel);
  if (addResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();
  refreshPassViews();
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
      .preDispatches = {.offset = static_cast<uint32_t>(
                            currentCommands_.ownedPreDispatches.size() -
                            pass.preDispatches.size()),
                        .count =
                            static_cast<uint32_t>(pass.preDispatches.size())},
      .drawPayloads = {.offset = static_cast<uint32_t>(
                           currentCommands_.ownedDrawItems.size() -
                           pass.draws.size()),
                       .count = static_cast<uint32_t>(pass.draws.size())},
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
  drawBindings_.resize(drawOffset + drawCount);
  meshDispatchBindings_.resize(meshDispatchOffset + pass.meshDispatches.size());
  bufferCopyBindings_.resize(bufferCopyOffset + pass.bufferCopies.size());
  textureCopyBindings_.resize(textureCopyOffset + pass.textureCopies.size());
  passes_.push_back(pass);
  const std::string_view name = debugName.empty() ? pass.debugLabel : debugName;
  passDebugNames_.emplace_back(name);
  return Result<RenderGraphPassId, std::string>::makeResult(passId);
}

void RenderGraphBuilder::refreshPassViews() {
  for (size_t i = 0u; i < currentCommands_.ownedPreDispatches.size(); ++i) {
    auto &command = currentCommands_.ownedPreDispatches[i];
    command.debugLabel = currentCommands_.ownedPreDispatchDebugLabels[i];
    command.pushConstants = currentCommands_.ownedPreDispatchPushConstants[i];
    command.pushConstantTextureBindings =
        currentCommands_.ownedPreDispatchTextureBindings[i];
  }
  for (size_t i = 0u; i < currentCommands_.ownedDrawItems.size(); ++i) {
    auto &command = currentCommands_.ownedDrawItems[i];
    command.debugLabel = currentCommands_.ownedDrawDebugLabels[i];
    command.pushConstants = currentCommands_.ownedDrawPushConstants[i];
    command.pushConstantTextureBindings =
        currentCommands_.ownedDrawTextureBindings[i];
  }
  for (size_t i = 0u; i < currentCommands_.ownedMeshDispatchItems.size(); ++i) {
    auto &command = currentCommands_.ownedMeshDispatchItems[i];
    command.debugLabel = currentCommands_.ownedMeshDispatchDebugLabels[i];
    command.pushConstants = currentCommands_.ownedMeshDispatchPushConstants[i];
    command.pushConstantTextureBindings =
        currentCommands_.ownedMeshDispatchTextureBindings[i];
  }
  for (size_t passIndex = 0u; passIndex < passes_.size(); ++passIndex) {
    RenderPass &pass = passes_[passIndex];
    const PassBindings &bindings = passBindings_[passIndex];
    const auto view = [](const auto &storage, BindingRange range) {
      using Element = typename std::decay_t<decltype(storage)>::value_type;
      return range.count == 0u
                 ? std::span<const Element>{}
                 : std::span<const Element>(storage.data() + range.offset,
                                            range.count);
    };
    pass.preDispatches =
        view(currentCommands_.ownedPreDispatches, bindings.preDispatches);
    pass.draws = view(currentCommands_.ownedDrawItems, bindings.drawPayloads);
    pass.meshDispatches =
        view(currentCommands_.ownedMeshDispatchItems, bindings.meshDispatches);
    pass.bufferCopies =
        view(currentCommands_.ownedBufferCopyItems, bindings.bufferCopies);
    pass.textureCopies =
        view(currentCommands_.ownedTextureCopyItems, bindings.textureCopies);
    pass.recordingSamplers =
        passIndex < currentCommands_.ownedRecordingSamplersByPass.size()
            ? std::span<const SamplerHandle>(
                  currentCommands_.ownedRecordingSamplersByPass[passIndex])
            : std::span<const SamplerHandle>{};
    pass.accelerationStructureBuilds =
        passIndex <
                currentCommands_.ownedAccelerationStructureBuildsByPass.size()
            ? std::span<const AccelerationStructureBuildItem>(
                  currentCommands_
                      .ownedAccelerationStructureBuildsByPass[passIndex])
            : std::span<const AccelerationStructureBuildItem>{};
    const std::pmr::string &name = passDebugNames_[passIndex];
    pass.debugLabel = std::string_view(name.data(), name.size());
  }
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
  (void)runtime;
  auto &plan = compiled.plan;
  plan.edges.reserve(work.scheduledDependencies.size());
  for (const DependencyEdge edge : work.scheduledDependencies)
    plan.edges.push_back({.before = edge.before, .after = edge.after});
  plan.orderedPassIndices.assign(work.order.begin(), work.order.end());
  plan.recordedGraphicsPasses.resize(work.order.size());
  plan.passBarrierPlans.resize(work.order.size());
  plan.preDispatchRangesByPass.resize(work.order.size());
  plan.drawRangesByPass.resize(work.order.size());
  plan.meshDispatchRangesByPass.resize(work.order.size());
  plan.bufferCopyRangesByPass.resize(work.order.size());
  plan.textureCopyRangesByPass.resize(work.order.size());
  plan.commandResourcePatches.clear();
  const auto appendPassTexturePatch =
      [&](uint32_t orderedPassIndex, uint32_t resourceIndex, auto patchTag) {
        if (resourceIndex != UINT32_MAX && !textures_[resourceIndex].imported) {
          using Patch = decltype(patchTag);
          plan.commandResourcePatches.push_back(
              Patch{.orderedPassIndex = orderedPassIndex,
                    .resourceIndex = resourceIndex});
        }
      };
  const auto appendCommandBufferPatch =
      [&](uint32_t orderedPassIndex, uint32_t commandIndex,
          uint32_t resourceIndex, auto patchTag) {
        if (resourceIndex != UINT32_MAX && !buffers_[resourceIndex].imported) {
          using Patch = decltype(patchTag);
          plan.commandResourcePatches.push_back(
              Patch{.orderedPassIndex = orderedPassIndex,
                    .commandIndex = commandIndex,
                    .resourceIndex = resourceIndex});
        }
      };
  const auto appendCommandTexturePatch =
      [&](uint32_t orderedPassIndex, uint32_t commandIndex,
          uint32_t resourceIndex, auto patchTag) {
        if (resourceIndex != UINT32_MAX && !textures_[resourceIndex].imported) {
          using Patch = decltype(patchTag);
          plan.commandResourcePatches.push_back(
              Patch{.orderedPassIndex = orderedPassIndex,
                    .commandIndex = commandIndex,
                    .resourceIndex = resourceIndex});
        }
      };
  for (uint32_t orderedPassIndex = 0u; orderedPassIndex < work.order.size();
       ++orderedPassIndex) {
    const uint32_t passIndex = work.order[orderedPassIndex];
    const PassBindings &bindings = passBindings_[passIndex];
    plan.recordedGraphicsPasses[orderedPassIndex] = {
        .orderedPassIndex = orderedPassIndex, .declaredPassIndex = passIndex};
    plan.passBarrierPlans[orderedPassIndex] = {.orderedPassIndex =
                                                   orderedPassIndex};
    plan.preDispatchRangesByPass[orderedPassIndex] = bindings.preDispatches;
    plan.drawRangesByPass[orderedPassIndex] = bindings.drawPayloads;
    plan.meshDispatchRangesByPass[orderedPassIndex] = bindings.meshDispatches;
    plan.bufferCopyRangesByPass[orderedPassIndex] = bindings.bufferCopies;
    plan.textureCopyRangesByPass[orderedPassIndex] = bindings.textureCopies;
    appendPassTexturePatch(orderedPassIndex, bindings.color,
                           RenderGraphPlan::PassColorTexturePatch{});
    appendPassTexturePatch(orderedPassIndex, bindings.colorResolve,
                           RenderGraphPlan::PassColorResolveTexturePatch{});
    appendPassTexturePatch(orderedPassIndex, bindings.depth,
                           RenderGraphPlan::PassDepthTexturePatch{});
    appendPassTexturePatch(orderedPassIndex, bindings.depthResolve,
                           RenderGraphPlan::PassDepthResolveTexturePatch{});
    for (uint32_t commandIndex = 0u; commandIndex < bindings.draws.count;
         ++commandIndex) {
      const DrawBindings binding =
          drawBindings_[bindings.draws.offset + commandIndex];
      appendCommandBufferPatch(orderedPassIndex, commandIndex, binding.vertex,
                               RenderGraphPlan::DrawVertexBufferPatch{});
      appendCommandBufferPatch(orderedPassIndex, commandIndex, binding.index,
                               RenderGraphPlan::DrawIndexBufferPatch{});
      appendCommandBufferPatch(orderedPassIndex, commandIndex, binding.indirect,
                               RenderGraphPlan::DrawIndirectBufferPatch{});
      appendCommandBufferPatch(orderedPassIndex, commandIndex,
                               binding.indirectCount,
                               RenderGraphPlan::DrawIndirectCountBufferPatch{});
    }
    for (uint32_t commandIndex = 0u;
         commandIndex < bindings.meshDispatches.count; ++commandIndex) {
      const MeshDispatchBindings binding =
          meshDispatchBindings_[bindings.meshDispatches.offset + commandIndex];
      appendCommandBufferPatch(
          orderedPassIndex, commandIndex, binding.indirect,
          RenderGraphPlan::MeshDispatchIndirectBufferPatch{});
      appendCommandBufferPatch(
          orderedPassIndex, commandIndex, binding.indirectCount,
          RenderGraphPlan::MeshDispatchIndirectCountBufferPatch{});
    }
    for (uint32_t commandIndex = 0u; commandIndex < bindings.bufferCopies.count;
         ++commandIndex) {
      const BufferCopyBindings binding =
          bufferCopyBindings_[bindings.bufferCopies.offset + commandIndex];
      appendCommandBufferPatch(orderedPassIndex, commandIndex, binding.source,
                               RenderGraphPlan::BufferCopySourcePatch{});
      appendCommandBufferPatch(orderedPassIndex, commandIndex,
                               binding.destination,
                               RenderGraphPlan::BufferCopyDestinationPatch{});
    }
    for (uint32_t commandIndex = 0u;
         commandIndex < bindings.textureCopies.count; ++commandIndex) {
      const TextureCopyBindings binding =
          textureCopyBindings_[bindings.textureCopies.offset + commandIndex];
      appendCommandTexturePatch(orderedPassIndex, commandIndex, binding.source,
                                RenderGraphPlan::TextureCopySourcePatch{});
      appendCommandTexturePatch(orderedPassIndex, commandIndex,
                                binding.destination,
                                RenderGraphPlan::TextureCopyDestinationPatch{});
    }
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
RenderGraphBuilder::compile(RenderGraphRuntime &runtime) {
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
  compiled.commands = buildFrameCommands(compiled.plan);
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
    for (const RenderGraphPlan::CommandResourcePatch &patch :
         compiled.plan.commandResourcePatches) {
      std::visit(
          [&](const auto &site) {
            using Site = std::decay_t<decltype(site)>;
            RenderPass &pass = executablePasses[site.orderedPassIndex];
            if constexpr (std::is_same_v<
                              Site, RenderGraphPlan::PassColorTexturePatch>) {
              pass.colorTexture = transientTexture(site.resourceIndex);
            } else if constexpr (std::is_same_v<
                                     Site,
                                     RenderGraphPlan::PassDepthTexturePatch>) {
              pass.depthTexture = transientTexture(site.resourceIndex);
            } else if constexpr (std::is_same_v<
                                     Site, RenderGraphPlan::
                                               PassColorResolveTexturePatch>) {
              pass.colorResolveTexture = transientTexture(site.resourceIndex);
            } else if constexpr (std::is_same_v<
                                     Site, RenderGraphPlan::
                                               PassDepthResolveTexturePatch>) {
              pass.depthResolveTexture = transientTexture(site.resourceIndex);
            } else if constexpr (
                std::is_same_v<Site, RenderGraphPlan::DrawVertexBufferPatch> ||
                std::is_same_v<Site, RenderGraphPlan::DrawIndexBufferPatch> ||
                std::is_same_v<Site,
                               RenderGraphPlan::DrawIndirectBufferPatch> ||
                std::is_same_v<Site,
                               RenderGraphPlan::DrawIndirectCountBufferPatch>) {
              const auto range =
                  compiled.plan.drawRangesByPass[site.orderedPassIndex];
              DrawItem &draw =
                  executableDrawItems[range.offset + site.commandIndex];
              const BufferHandle handle = transientBuffer(site.resourceIndex);
              if constexpr (std::is_same_v<
                                Site, RenderGraphPlan::DrawVertexBufferPatch>)
                draw.vertexBuffer = handle;
              else if constexpr (std::is_same_v<
                                     Site,
                                     RenderGraphPlan::DrawIndexBufferPatch>)
                draw.indexBuffer = handle;
              else if constexpr (std::is_same_v<
                                     Site,
                                     RenderGraphPlan::DrawIndirectBufferPatch>)
                draw.indirectBuffer = handle;
              else
                draw.indirectCountBuffer = handle;
            } else if constexpr (
                std::is_same_v<
                    Site, RenderGraphPlan::MeshDispatchIndirectBufferPatch> ||
                std::is_same_v<
                    Site,
                    RenderGraphPlan::MeshDispatchIndirectCountBufferPatch>) {
              const auto range =
                  compiled.plan.meshDispatchRangesByPass[site.orderedPassIndex];
              MeshDispatchItem &dispatch =
                  executableMeshDispatches[range.offset + site.commandIndex];
              if constexpr (std::is_same_v<Site,
                                           RenderGraphPlan::
                                               MeshDispatchIndirectBufferPatch>)
                dispatch.indirectBuffer = transientBuffer(site.resourceIndex);
              else
                dispatch.indirectCountBuffer =
                    transientBuffer(site.resourceIndex);
            } else if constexpr (
                std::is_same_v<Site, RenderGraphPlan::BufferCopySourcePatch> ||
                std::is_same_v<Site,
                               RenderGraphPlan::BufferCopyDestinationPatch>) {
              const auto range =
                  compiled.plan.bufferCopyRangesByPass[site.orderedPassIndex];
              BufferCopyRegion &copy =
                  executableBufferCopies[range.offset + site.commandIndex];
              if constexpr (std::is_same_v<
                                Site, RenderGraphPlan::BufferCopySourcePatch>)
                copy.srcBuffer = transientBuffer(site.resourceIndex);
              else
                copy.dstBuffer = transientBuffer(site.resourceIndex);
            } else {
              const auto range =
                  compiled.plan.textureCopyRangesByPass[site.orderedPassIndex];
              TextureCopyItem &copy =
                  executableTextureCopies[range.offset + site.commandIndex];
              if constexpr (std::is_same_v<
                                Site, RenderGraphPlan::TextureCopySourcePatch>)
                copy.sourceTexture = transientTexture(site.resourceIndex);
              else
                copy.destinationTexture = transientTexture(site.resourceIndex);
            }
          },
          patch);
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
        RecordingReferenceSet::append(
            references.buffers,
            std::span<const BufferHandle>(commands.bufferHandlesByResource));
        RecordingReferenceSet::append(
            references.buffers,
            std::span<const BufferHandle>(transientBufferHandles));
        RecordingReferenceSet::append(
            references.textures,
            std::span<const TextureHandle>(commands.textureHandlesByResource));
        RecordingReferenceSet::append(
            references.textures,
            std::span<const TextureHandle>(transientTextureHandles));
        RecordingReferenceSet::append(
            references.accelerationStructures,
            std::span<const AccelerationStructureHandle>(
                commands.accelerationStructureHandlesByResource));
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
          auto recordResult = gpu.recordGraphicsRangeWithReferences(
              recordingContexts[workerIndex], steps,
              recordingReferences[workerIndex].view());
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
          for (const RecordedCommandBufferHandle handle :
               recordedCommandBuffers) {
            if (nuri::isValid(handle)) {
              (void)gpu.discardRecordedGraphicsCommandBuffer(handle);
            }
          }
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
