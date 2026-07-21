#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/core/containers/hash_map.h"
#include "nuri/core/profiling.h"
#include "nuri/pch.h"
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
[[nodiscard]] uint64_t mixFingerprintSeed(uint64_t seed,
                                          uint64_t value) noexcept {
  constexpr uint64_t kMix = 0x9e3779b97f4a7c15ull;
  seed ^= value + kMix + (seed << 6u) + (seed >> 2u);
  return seed;
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
    mix(pass.externalTemporalDispatch.provider != nullptr ? 1u : 0u);
    for (const ComputeDispatchItem &dispatch : pass.preDispatches) {
      mix(static_cast<uint64_t>(dispatch.dependencyBuffers.size()));
      mix(static_cast<uint64_t>(dispatch.dependencyTextures.size()));
      mix(nuri::isValid(dispatch.rayQueryBinding) ? 1u : 0u);
    }
    for (const MeshDispatchItem &dispatch : pass.meshDispatches) {
      mix(static_cast<uint64_t>(dispatch.command));
      mix(nuri::isValid(dispatch.indirectBuffer) ? 1u : 0u);
      mix(nuri::isValid(dispatch.indirectCountBuffer) ? 1u : 0u);
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
                                                "addTextureCopyPass");
  }
  const bool computeOnly = isComputeOnlyExecutionMode(desc.executionMode);
  const bool externalTemporal =
      isExternalTemporalExecutionMode(desc.executionMode);
  if (!computeOnly && !externalTemporal) {
    if (desc.externalTemporalDispatch.provider != nullptr) {
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
  if (computeOnly && desc.externalTemporalDispatch.provider != nullptr) {
    return Result<bool, std::string>::makeError(
        std::string(caller) +
        ": compute-only pass cannot contain an external temporal dispatch");
  }
  if (externalTemporal && !desc.preDispatches.empty()) {
    return Result<bool, std::string>::makeError(
        std::string(caller) +
        ": external temporal pass cannot contain native compute dispatches");
  }
  if (externalTemporal && desc.externalTemporalDispatch.provider == nullptr) {
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
constexpr uint32_t kMinValidationItemsPerWorker = 256u;
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
makeValidationRanges(uint32_t itemCount, uint32_t workerCount) {
  return makeAdaptiveRanges(itemCount, workerCount,
                            kMinValidationItemsPerWorker);
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
      textureCopyBindings_(memory_), importedTextureIndicesByHandle_(memory_),
      importedBufferIndicesByHandle_(memory_),
      importedAccelerationStructureIndicesByHandle_(memory_),
      explicitTextureAccessIndicesByPassResource_(memory_),
      inferredTextureAccessIndicesByPassResource_(memory_),
      explicitBufferAccessIndicesByPassResource_(memory_),
      inferredBufferAccessIndicesByPassResource_(memory_),
      explicitAccelerationStructureAccessIndicesByPassResource_(memory_),
      inferredAccelerationStructureAccessIndicesByPassResource_(memory_),
      dependencyEdgeKeys_(memory_), dependencies_(memory_),
      passResourceAccesses_(memory_), frameOutputTextureSet_(memory_),
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
  textureCopyBindings_.clear();
  importedTextureIndicesByHandle_.clear();
  importedBufferIndicesByHandle_.clear();
  importedAccelerationStructureIndicesByHandle_.clear();
  explicitTextureAccessIndicesByPassResource_.clear();
  inferredTextureAccessIndicesByPassResource_.clear();
  explicitBufferAccessIndicesByPassResource_.clear();
  inferredBufferAccessIndicesByPassResource_.clear();
  explicitAccelerationStructureAccessIndicesByPassResource_.clear();
  inferredAccelerationStructureAccessIndicesByPassResource_.clear();
  dependencyEdgeKeys_.clear();
  dependencies_.clear();
  passResourceAccesses_.clear();
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
  mixStructure(static_cast<uint64_t>(passResourceAccesses_.size()));
  for (const PassResourceAccess &access : passResourceAccesses_) {
    mixStructure(access.passIndex);
    mixStructure(static_cast<uint64_t>(access.resourceKind));
    mixStructure(access.resourceIndex);
    mixStructure(static_cast<uint64_t>(access.mode));
    mixStructure(static_cast<uint64_t>(access.requestedState));
    mixStructure(access.inferred ? 1u : 0u);
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
      .passAccessCount = passResourceAccesses_.size(),
      .frameOutputCount = frameOutputTextureIndices_.size(),
      .sideEffectMarkCount = sideEffectPassMarks_.size(),
      .allPassesBorrowPayload = allPassesBorrowPayload_,
      .payloadLayoutHash = payloadLayoutHash,
      .structuralIdentityHash = structuralIdentityHash,
      .transientResourceDescriptorsHash = transientResourceDescriptorsHash_,
      .persistentHandlesVersion = persistentHandlesVersion_,
  };
}

void RenderGraphBuilder::refreshHandlesInCompileResult(
    RenderGraphCompileResult &result) const {
  for (size_t i = 0;
       i < textures_.size() && i < result.textureHandlesByResource.size();
       ++i) {
    if (textures_[i].imported) {
      result.textureHandlesByResource[i] = textures_[i].importedHandle;
    }
  }
  for (size_t i = 0;
       i < buffers_.size() && i < result.bufferHandlesByResource.size(); ++i) {
    if (buffers_[i].imported) {
      result.bufferHandlesByResource[i] = buffers_[i].importedHandle;
    }
  }
  for (size_t i = 0; i < accelerationStructures_.size() &&
                     i < result.accelerationStructureHandlesByResource.size();
       ++i) {
    result.accelerationStructureHandlesByResource[i] =
        accelerationStructures_[i].importedHandle;
  }
  for (size_t i = 0;
       i < result.resolvedDependencyBufferResourceIndices.size() &&
       i < result.resolvedDependencyBuffers.size();
       ++i) {
    const uint32_t resourceIndex =
        result.resolvedDependencyBufferResourceIndices[i];
    if (resourceIndex == UINT32_MAX || resourceIndex >= buffers_.size()) {
      continue;
    }
    if (buffers_[resourceIndex].imported) {
      result.resolvedDependencyBuffers[i] =
          buffers_[resourceIndex].importedHandle;
    }
  }
  for (size_t i = 0;
       i < result.resolvedDependencyTextureResourceIndices.size() &&
       i < result.resolvedDependencyTextures.size();
       ++i) {
    const uint32_t resourceIndex =
        result.resolvedDependencyTextureResourceIndices[i];
    if (resourceIndex == UINT32_MAX || resourceIndex >= textures_.size()) {
      continue;
    }
    if (textures_[resourceIndex].imported) {
      result.resolvedDependencyTextures[i] =
          textures_[resourceIndex].importedHandle;
    }
  }
  for (size_t i = 0;
       i < result.resolvedPreDispatchDependencyBufferResourceIndices.size() &&
       i < result.resolvedPreDispatchDependencyBuffers.size();
       ++i) {
    const uint32_t resourceIndex =
        result.resolvedPreDispatchDependencyBufferResourceIndices[i];
    if (resourceIndex == UINT32_MAX || resourceIndex >= buffers_.size()) {
      continue;
    }
    if (buffers_[resourceIndex].imported) {
      result.resolvedPreDispatchDependencyBuffers[i] =
          buffers_[resourceIndex].importedHandle;
    }
  }
  const size_t passCount = result.orderedPasses.size();
  constexpr std::array attachmentBindings{
      std::pair{&RenderPass::colorTexture, &PassBindings::color},
      std::pair{&RenderPass::colorResolveTexture, &PassBindings::colorResolve},
      std::pair{&RenderPass::depthTexture, &PassBindings::depth},
      std::pair{&RenderPass::depthResolveTexture, &PassBindings::depthResolve},
  };
  for (size_t i = 0; i < passCount; ++i) {
    const uint32_t passIndex = result.orderedPassIndices[i];
    for (const auto [target, binding] : attachmentBindings) {
      const uint32_t resource = passBindings_[passIndex].*binding;
      if (resource != UINT32_MAX && textures_[resource].imported) {
        result.orderedPasses[i].*target = textures_[resource].importedHandle;
      }
    }
  }
  for (size_t i = 0; i < passCount; ++i) {
    const uint32_t passIndex = result.orderedPassIndices[i];
    const RenderPass &sourcePass = passes_[passIndex];
    RenderPass &refreshedPass = result.orderedPasses[i];
    refreshedPass.color = sourcePass.color;
    refreshedPass.executionMode = sourcePass.executionMode;
    refreshedPass.hasColorAttachment = sourcePass.hasColorAttachment;
    refreshedPass.depth = sourcePass.depth;
    refreshedPass.useViewport = sourcePass.useViewport;
    refreshedPass.viewport = sourcePass.viewport;
    refreshedPass.payloadBorrowed = sourcePass.payloadBorrowed;
    refreshedPass.drawBuffersPreResolved = sourcePass.drawBuffersPreResolved;
    refreshedPass.debugLabel = sourcePass.debugLabel;
    refreshedPass.debugColor = sourcePass.debugColor;
    refreshedPass.accelerationStructureBuilds =
        sourcePass.accelerationStructureBuilds;
    const auto dependencyRange = result.dependencyBufferRangesByPass[i];
    if (dependencyRange.count > 0u &&
        dependencyRange.offset <= result.resolvedDependencyBuffers.size() &&
        dependencyRange.count <=
            result.resolvedDependencyBuffers.size() - dependencyRange.offset) {
      refreshedPass.dependencyBuffers = std::span<const BufferHandle>(
          result.resolvedDependencyBuffers.data() + dependencyRange.offset,
          dependencyRange.count);
    } else {
      refreshedPass.dependencyBuffers = {};
    }
    const auto dependencyTextureRange = result.dependencyTextureRangesByPass[i];
    if (dependencyTextureRange.count > 0u &&
        dependencyTextureRange.offset <=
            result.resolvedDependencyTextures.size() &&
        dependencyTextureRange.count <=
            result.resolvedDependencyTextures.size() -
                dependencyTextureRange.offset) {
      refreshedPass.dependencyTextures = std::span<const TextureHandle>(
          result.resolvedDependencyTextures.data() +
              dependencyTextureRange.offset,
          dependencyTextureRange.count);
    } else {
      refreshedPass.dependencyTextures = {};
    }
    const auto preDispatchRange = result.preDispatchRangesByPass[i];
    if (preDispatchRange.count > 0u) {
      if (sourcePass.preDispatches.size() == preDispatchRange.count &&
          preDispatchRange.offset <= result.ownedPreDispatches.size() &&
          preDispatchRange.count <=
              result.ownedPreDispatches.size() - preDispatchRange.offset) {
        for (uint32_t dispatchIndex = 0; dispatchIndex < preDispatchRange.count;
             ++dispatchIndex) {
          const uint32_t globalDispatchIndex =
              preDispatchRange.offset + dispatchIndex;
          if (globalDispatchIndex >=
              result.preDispatchDependencyRanges.size()) {
            break;
          }
          ComputeDispatchItem refreshedDispatch =
              sourcePass.preDispatches[dispatchIndex];
          const auto depRange =
              result.preDispatchDependencyRanges[globalDispatchIndex];
          if (depRange.count > 0u &&
              depRange.offset <=
                  result.resolvedPreDispatchDependencyBuffers.size() &&
              depRange.count <=
                  result.resolvedPreDispatchDependencyBuffers.size() -
                      depRange.offset) {
            refreshedDispatch.dependencyBuffers = std::span<const BufferHandle>(
                result.resolvedPreDispatchDependencyBuffers.data() +
                    depRange.offset,
                depRange.count);
          } else {
            refreshedDispatch.dependencyBuffers = {};
          }
          refreshedDispatch.dependencyBufferAccessModes = {};
          result.ownedPreDispatches[globalDispatchIndex] = refreshedDispatch;
        }
        refreshedPass.preDispatches = std::span<const ComputeDispatchItem>(
            result.ownedPreDispatches.data() + preDispatchRange.offset,
            preDispatchRange.count);
      } else {
        refreshedPass.preDispatches = {};
      }
    } else {
      refreshedPass.preDispatches = sourcePass.preDispatches;
    }
    const auto drawRange = result.drawRangesByPass[i];
    if (drawRange.count > 0u) {
      const uint32_t actualDrawCount =
          static_cast<uint32_t>(sourcePass.draws.size());
      if (actualDrawCount <= drawRange.count &&
          drawRange.offset <= result.ownedDrawItems.size() &&
          drawRange.count <= result.ownedDrawItems.size() - drawRange.offset) {
        for (uint32_t drawIndex = 0; drawIndex < actualDrawCount; ++drawIndex) {
          result.ownedDrawItems[drawRange.offset + drawIndex] =
              sourcePass.draws[drawIndex];
        }
        refreshedPass.draws = std::span<const DrawItem>(
            result.ownedDrawItems.data() + drawRange.offset, actualDrawCount);
      } else {
        refreshedPass.draws = {};
      }
    } else {
      refreshedPass.draws = sourcePass.draws;
    }
    const auto meshDispatchRange = result.meshDispatchRangesByPass[i];
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
        result
            .ownedMeshDispatchItems[meshDispatchRange.offset + dispatchIndex] =
            dispatch;
      }
      refreshedPass.meshDispatches = std::span<const MeshDispatchItem>(
          result.ownedMeshDispatchItems.data() + meshDispatchRange.offset,
          actualDispatchCount);
    } else {
      refreshedPass.meshDispatches = sourcePass.meshDispatches;
    }
    const auto textureCopyRange = result.textureCopyRangesByPass[i];
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
        result.ownedTextureCopyItems[textureCopyRange.offset + copyIndex] =
            copy;
      }
      refreshedPass.textureCopies = std::span<const TextureCopyItem>(
          result.ownedTextureCopyItems.data() + textureCopyRange.offset,
          actualCopyCount);
    } else {
      refreshedPass.textureCopies = sourcePass.textureCopies;
    }
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
    RenderGraphAccessMode mode, bool inferred) {
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
  const uint64_t key = foldPassResourceKey(pass.value, texture.value);
  auto &indexByKey = inferred ? inferredTextureAccessIndicesByPassResource_
                              : explicitTextureAccessIndicesByPassResource_;
  if (const auto existing = indexByKey.find(key);
      existing != indexByKey.end()) {
    PassResourceAccess &merged = passResourceAccesses_[existing->second];
    merged.mode = merged.mode | mode;
    return Result<bool, std::string>::makeResult(true);
  }
  const uint32_t accessIndex =
      static_cast<uint32_t>(passResourceAccesses_.size());
  if (accessIndex == UINT32_MAX) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addTextureAccessInternal: access count exceeds "
        "uint32_t");
  }
  passResourceAccesses_.push_back(PassResourceAccess{
      .passIndex = pass.value,
      .resourceKind = AccessResourceKind::Texture,
      .resourceIndex = texture.value,
      .mode = mode,
      .inferred = inferred,
  });
  indexByKey.emplace(key, accessIndex);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RenderGraphBuilder::addTextureAccess(RenderGraphPassId pass,
                                     RenderGraphTextureId texture,
                                     RenderGraphAccessMode mode) {
  return addTextureAccessInternal(pass, texture, mode, false);
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
    PassResourceAccess &merged = passResourceAccesses_[existing->second];
    if (merged.requestedState != RenderGraphResourceState::Unknown &&
        requestedState != RenderGraphResourceState::Unknown &&
        merged.requestedState != requestedState) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::addBufferAccessInternal: contradictory "
          "resource states");
    }
    merged.mode = merged.mode | mode;
    if (merged.requestedState == RenderGraphResourceState::Unknown) {
      merged.requestedState = requestedState;
    }
    return Result<bool, std::string>::makeResult(true);
  }
  const uint32_t accessIndex =
      static_cast<uint32_t>(passResourceAccesses_.size());
  if (accessIndex == UINT32_MAX) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addBufferAccessInternal: access count exceeds "
        "uint32_t");
  }
  passResourceAccesses_.push_back(PassResourceAccess{
      .passIndex = pass.value,
      .resourceKind = AccessResourceKind::Buffer,
      .resourceIndex = buffer.value,
      .mode = mode,
      .requestedState = requestedState,
      .inferred = inferred,
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
    PassResourceAccess &merged = passResourceAccesses_[existing->second];
    if (merged.requestedState != state) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::addAccelerationStructureAccessInternal: "
          "contradictory access for the same pass and resource");
    }
    merged.mode = merged.mode | mode;
    return Result<bool, std::string>::makeResult(true);
  }
  if (passResourceAccesses_.size() >= UINT32_MAX) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::addAccelerationStructureAccessInternal: access "
        "count exceeds uint32_t");
  }
  const uint32_t accessIndex =
      static_cast<uint32_t>(passResourceAccesses_.size());
  passResourceAccesses_.push_back(PassResourceAccess{
      .passIndex = pass.value,
      .resourceKind = AccessResourceKind::AccelerationStructure,
      .resourceIndex = accelerationStructure.value,
      .mode = mode,
      .requestedState = state,
      .inferred = inferred,
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
    dispatch.debugLabel =
        std::string_view(ownedPayload.preDispatchDebugLabels[i].data(),
                         ownedPayload.preDispatchDebugLabels[i].size());
  }
  ownedPayload.drawDebugLabels.reserve(desc.draws.size());
  ownedPayload.drawPushConstants.reserve(desc.draws.size());
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
  }
  ownedPayload.draws.resize(desc.draws.size());
  for (size_t i = 0; i < desc.draws.size(); ++i) {
    const DrawItem &sourceDraw = desc.draws[i];
    DrawItem &draw = ownedPayload.draws[i];
    draw = sourceDraw;
    draw.pushConstants =
        std::span<const std::byte>(ownedPayload.drawPushConstants[i].data(),
                                   ownedPayload.drawPushConstants[i].size());
    draw.debugLabel = std::string_view(ownedPayload.drawDebugLabels[i].data(),
                                       ownedPayload.drawDebugLabels[i].size());
  }
  ownedPayload.meshDispatchDebugLabels.reserve(desc.meshDispatches.size());
  ownedPayload.meshDispatchPushConstants.reserve(desc.meshDispatches.size());
  ownedPayload.meshDispatchDependencyBuffers.reserve(
      desc.meshDispatches.size());
  ownedPayload.meshDispatchDependencyTextures.reserve(
      desc.meshDispatches.size());
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
          RenderGraphCompileResult::MeshDispatchBufferBindingTarget::Indirect,
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
          RenderGraphCompileResult::MeshDispatchBufferBindingTarget::
              IndirectCount,
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
          std::pair<BufferHandle,
                    RenderGraphCompileResult::DrawBufferBindingTarget>,
          4>
          bindings = {{
              {draw.vertexBuffer,
               RenderGraphCompileResult::DrawBufferBindingTarget::Vertex},
              {draw.indexBuffer,
               RenderGraphCompileResult::DrawBufferBindingTarget::Index},
              {draw.indirectBuffer,
               RenderGraphCompileResult::DrawBufferBindingTarget::Indirect},
              {draw.indirectCountBuffer,
               RenderGraphCompileResult::DrawBufferBindingTarget::
                   IndirectCount},
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
  if (!desc.borrowPayload) {
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
  pass.debugColor = desc.debugColor;
  pass.payloadBorrowed = desc.borrowPayload;
  pass.drawBuffersPreResolved = desc.drawBuffersPreResolved;
  if (desc.borrowPayload) {
    pass.preDispatches = desc.preDispatches;
    pass.dependencyBuffers = desc.dependencyBuffers;
    pass.dependencyTextures = desc.dependencyTextures;
    pass.draws = desc.draws;
    pass.meshDispatches = desc.meshDispatches;
    pass.debugLabel = desc.debugLabel;
  } else {
    OwnedPassPayload ownedPayload = clonePassPayload(desc);
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
                                       RenderGraphAccessMode::Read);
    if (readResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          readResult.error());
    }
    auto writeResult = addTextureAccess(passId, copy.destinationTexture,
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

Result<bool, std::string> RenderGraphBuilder::bindDrawBuffer(
    RenderGraphPassId pass, uint32_t drawIndex,
    RenderGraphCompileResult::DrawBufferBindingTarget target,
    RenderGraphBufferId buffer, RenderGraphAccessMode mode) {
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
    RenderGraphCompileResult::MeshDispatchBufferBindingTarget target,
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
    RenderGraphCompileResult &compiled,
    RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  compiled.textureHandlesByResource.resize(textures_.size());
  for (uint32_t i = 0; i < textures_.size(); ++i) {
    const TextureResource &texture = textures_[i];
    if (texture.imported) {
      compiled.textureHandlesByResource[i] = texture.importedHandle;
      ++compiled.resourceStats.importedTextures;
    } else {
      ++compiled.resourceStats.transientTextures;
    }
  }
  compiled.bufferHandlesByResource.resize(buffers_.size());
  for (uint32_t i = 0; i < buffers_.size(); ++i) {
    const BufferResource &buffer = buffers_[i];
    if (buffer.imported) {
      compiled.bufferHandlesByResource[i] = buffer.importedHandle;
      ++compiled.resourceStats.importedBuffers;
    } else {
      ++compiled.resourceStats.transientBuffers;
    }
  }
  compiled.accelerationStructureHandlesByResource.resize(
      accelerationStructures_.size());
  for (uint32_t i = 0; i < accelerationStructures_.size(); ++i) {
    compiled.accelerationStructureHandlesByResource[i] =
        accelerationStructures_[i].importedHandle;
    ++compiled.resourceStats.importedAccelerationStructures;
  }
  work.passCount = static_cast<uint32_t>(passes_.size());
  work.activePassCount = work.passCount;
  compiled.declaredPassCount = work.passCount;
}

Result<bool, std::string> RenderGraphBuilder::compileStageC1C2BuildTopology(
    RenderGraphRuntime &runtime, RenderGraphCompileResult &compiled,
    RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_BARRIER);
  work.compiledAccesses = passResourceAccesses_;
  std::sort(work.compiledAccesses.begin(), work.compiledAccesses.end(),
            [](const PassResourceAccess &lhs, const PassResourceAccess &rhs) {
              const uint8_t lhsKind = static_cast<uint8_t>(lhs.resourceKind);
              const uint8_t rhsKind = static_cast<uint8_t>(rhs.resourceKind);
              if (lhsKind != rhsKind) {
                return lhsKind < rhsKind;
              }
              if (lhs.resourceIndex != rhs.resourceIndex) {
                return lhs.resourceIndex < rhs.resourceIndex;
              }
              return lhs.passIndex < rhs.passIndex;
            });
  if (!work.compiledAccesses.empty()) {
    size_t writeIndex = 0u;
    size_t groupBegin = 0u;
    while (groupBegin < work.compiledAccesses.size()) {
      size_t groupEnd = groupBegin + 1u;
      while (groupEnd < work.compiledAccesses.size() &&
             work.compiledAccesses[groupEnd].resourceKind ==
                 work.compiledAccesses[groupBegin].resourceKind &&
             work.compiledAccesses[groupEnd].resourceIndex ==
                 work.compiledAccesses[groupBegin].resourceIndex &&
             work.compiledAccesses[groupEnd].passIndex ==
                 work.compiledAccesses[groupBegin].passIndex) {
        ++groupEnd;
      }
      RenderGraphAccessMode explicitMode = RenderGraphAccessMode::None;
      RenderGraphAccessMode inferredMode = RenderGraphAccessMode::None;
      RenderGraphResourceState explicitState =
          RenderGraphResourceState::Unknown;
      RenderGraphResourceState inferredState =
          RenderGraphResourceState::Unknown;
      for (size_t i = groupBegin; i < groupEnd; ++i) {
        if (work.compiledAccesses[i].inferred) {
          inferredMode = inferredMode | work.compiledAccesses[i].mode;
          inferredState = work.compiledAccesses[i].requestedState;
          continue;
        }
        explicitMode = explicitMode | work.compiledAccesses[i].mode;
        explicitState = work.compiledAccesses[i].requestedState;
      }
      const bool hasExplicit =
          hasAccessFlag(explicitMode, RenderGraphAccessMode::Read) ||
          hasAccessFlag(explicitMode, RenderGraphAccessMode::Write);
      const RenderGraphAccessMode selectedMode =
          hasExplicit ? explicitMode : inferredMode;
      if (hasAccessFlag(selectedMode, RenderGraphAccessMode::Read) ||
          hasAccessFlag(selectedMode, RenderGraphAccessMode::Write)) {
        work.compiledAccesses[writeIndex++] = PassResourceAccess{
            .passIndex = work.compiledAccesses[groupBegin].passIndex,
            .resourceKind = work.compiledAccesses[groupBegin].resourceKind,
            .resourceIndex = work.compiledAccesses[groupBegin].resourceIndex,
            .mode = selectedMode,
            .requestedState = hasExplicit ? explicitState : inferredState,
            .inferred = !hasExplicit,
        };
      }
      groupBegin = groupEnd;
    }
    work.compiledAccesses.resize(writeIndex);
  }
  NURI_PROFILER_ZONE("RenderGraph.compile.build_topology",
                     NURI_PROFILER_COLOR_BARRIER);
  PmrHashSet<uint64_t> dependencyEdgeKeys(memory_);
  dependencyEdgeKeys.reserve(dependencies_.size() +
                             work.compiledAccesses.size() * 2u);
  std::pmr::vector<DependencyEdge> allDependencies(memory_);
  for (const DependencyEdge edge : dependencies_) {
    const uint64_t key =
        (static_cast<uint64_t>(edge.before) << 32u) | edge.after;
    if (dependencyEdgeKeys.insert(key).second) {
      allDependencies.push_back(edge);
    }
  }
  allDependencies.reserve(dependencies_.size() +
                          work.compiledAccesses.size() * 2u);
  struct ResourceAccessGroup {
    uint32_t resourceIndex = UINT32_MAX;
    size_t begin = 0u;
    size_t end = 0u;
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
  const auto addResourceHazards = [&](AccessResourceKind resourceKind) {
    std::vector<ResourceAccessGroup> groups{};
    groups.reserve(work.compiledAccesses.size());
    for (size_t i = 0u; i < work.compiledAccesses.size();) {
      if (work.compiledAccesses[i].resourceKind != resourceKind) {
        ++i;
        continue;
      }
      const uint32_t resourceIndex = work.compiledAccesses[i].resourceIndex;
      const size_t begin = i;
      do {
        ++i;
      } while (i < work.compiledAccesses.size() &&
               work.compiledAccesses[i].resourceKind == resourceKind &&
               work.compiledAccesses[i].resourceIndex == resourceIndex);
      groups.push_back(ResourceAccessGroup{
          .resourceIndex = resourceIndex,
          .begin = begin,
          .end = i,
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
        activeReaders.clear();
        uint32_t lastWriter = UINT32_MAX;
        for (size_t accessIndex = group.begin; accessIndex < group.end;
             ++accessIndex) {
          const PassResourceAccess &access = work.compiledAccesses[accessIndex];
          const bool hasRead =
              hasAccessFlag(access.mode, RenderGraphAccessMode::Read);
          const bool hasWrite =
              hasAccessFlag(access.mode, RenderGraphAccessMode::Write);
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
  addResourceHazards(AccessResourceKind::Texture);
  addResourceHazards(AccessResourceKind::Buffer);
  addResourceHazards(AccessResourceKind::AccelerationStructure);
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
      ++compiled.rootPassCount;
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
      for (const PassResourceAccess &access : work.compiledAccesses) {
        if (access.resourceKind != AccessResourceKind::Texture ||
            access.resourceIndex != textureIndex) {
          continue;
        }
        textureAccessByPass[access.passIndex] =
            textureAccessByPass[access.passIndex] | access.mode;
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
    compiled.culledPassCount = work.passCount - work.activePassCount;
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
    RenderGraphRuntime &runtime, RenderGraphCompileResult &compiled,
    const RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  compiled.passDebugNames.reserve(passDebugNames_.size());
  for (const std::pmr::string &name : passDebugNames_) {
    std::pmr::string copiedName(memory_);
    copiedName.assign(name.data(), name.size());
    compiled.passDebugNames.push_back(std::move(copiedName));
  }
  compiled.edges.reserve(work.scheduledDependencies.size());
  for (const DependencyEdge edge : work.scheduledDependencies) {
    compiled.edges.push_back(RenderGraphCompileResult::Edge{
        .before = edge.before, .after = edge.after});
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
    uint32_t quantizedDrawCount = 0u;
    uint32_t meshDispatchCount = 0u;
    uint32_t meshDispatchBindingOffset = 0u;
    uint32_t meshDispatchOutputOffset = 0u;
    uint32_t textureCopyCount = 0u;
    uint32_t textureCopyBindingOffset = 0u;
    uint32_t textureCopyOutputOffset = 0u;
    uint32_t unresolvedTextureOffset = 0u;
    uint32_t unresolvedTextureCount = 0u;
    uint32_t unresolvedDependencyOffset = 0u;
    uint32_t unresolvedDependencyCount = 0u;
    uint32_t unresolvedDependencyTextureOffset = 0u;
    uint32_t unresolvedDependencyTextureCount = 0u;
    uint32_t unresolvedPreDispatchDependencyOffset = 0u;
    uint32_t unresolvedPreDispatchDependencyCount = 0u;
    uint32_t unresolvedDrawOffset = 0u;
    uint32_t unresolvedDrawCount = 0u;
    uint32_t unresolvedMeshDispatchOffset = 0u;
    uint32_t unresolvedMeshDispatchCount = 0u;
    uint32_t unresolvedTextureCopyOffset = 0u;
    uint32_t unresolvedTextureCopyCount = 0u;
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
      plan.textureCopyCount = static_cast<uint32_t>(pass.textureCopies.size());
      plan.textureCopyBindingOffset =
          passBindings_[passIndex].textureCopies.offset;
      const auto unresolvedTexture = [&](uint32_t resourceIndex) {
        return resourceIndex != UINT32_MAX &&
               !textures_[resourceIndex].imported;
      };
      plan.unresolvedTextureCount =
          unresolvedTexture(plan.colorTextureIndex) +
          unresolvedTexture(plan.colorResolveTextureIndex) +
          unresolvedTexture(plan.depthTextureIndex) +
          unresolvedTexture(plan.depthResolveTextureIndex);
      plan.unresolvedDependencyCount = countTransientBindings(
          passDependencyBufferBindingResourceIndices_,
          plan.dependencyBindingOffset, plan.dependencyCount, buffers_);
      plan.unresolvedDependencyTextureCount =
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
        plan.unresolvedPreDispatchDependencyCount +=
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
            plan.unresolvedDrawCount +=
                resource != UINT32_MAX && !buffers_[resource].imported;
          }
        }
      }
      for (uint32_t i = 0; i < plan.meshDispatchCount; ++i) {
        const MeshDispatchBindings binding =
            meshDispatchBindings_[plan.meshDispatchBindingOffset + i];
        plan.unresolvedMeshDispatchCount +=
            (binding.indirect != UINT32_MAX &&
             !buffers_[binding.indirect].imported) +
            (binding.indirectCount != UINT32_MAX &&
             !buffers_[binding.indirectCount].imported);
      }
      for (uint32_t i = 0; i < plan.textureCopyCount; ++i) {
        const TextureCopyBindings binding =
            textureCopyBindings_[plan.textureCopyBindingOffset + i];
        plan.unresolvedTextureCopyCount += !textures_[binding.source].imported;
        plan.unresolvedTextureCopyCount +=
            !textures_[binding.destination].imported;
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
  compiled.usedParallelPayloadResolution = usedParallelPassResolution;
  size_t totalDependencyBufferSlots = 0u;
  size_t totalDependencyTextureSlots = 0u;
  size_t totalPreDispatchItems = 0u;
  size_t totalPreDispatchDependencySlots = 0u;
  size_t totalDrawItems = 0u;
  size_t totalMeshDispatchItems = 0u;
  size_t totalTextureCopyItems = 0u;
  size_t totalUnresolvedTextureBindings = 0u;
  size_t totalUnresolvedDependencyBufferBindings = 0u;
  size_t totalUnresolvedDependencyTextureBindings = 0u;
  size_t totalUnresolvedPreDispatchDependencyBufferBindings = 0u;
  size_t totalUnresolvedDrawBufferBindings = 0u;
  size_t totalUnresolvedMeshDispatchBufferBindings = 0u;
  size_t totalUnresolvedTextureCopyBindings = 0u;
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
    if (plan.drawCount != 0u && (plan.unresolvedDrawCount != 0u ||
                                 !passes_[plan.passIndex].payloadBorrowed)) {
      if (totalDrawItems > UINT32_MAX) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: draw item output offset exceeds "
            "uint32_t range");
      }
      plan.drawOutputOffset = static_cast<uint32_t>(totalDrawItems);
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
    plan.unresolvedTextureOffset =
        static_cast<uint32_t>(totalUnresolvedTextureBindings);
    totalUnresolvedTextureBindings += plan.unresolvedTextureCount;
    plan.unresolvedDependencyOffset =
        static_cast<uint32_t>(totalUnresolvedDependencyBufferBindings);
    totalUnresolvedDependencyBufferBindings += plan.unresolvedDependencyCount;
    plan.unresolvedDependencyTextureOffset =
        static_cast<uint32_t>(totalUnresolvedDependencyTextureBindings);
    totalUnresolvedDependencyTextureBindings +=
        plan.unresolvedDependencyTextureCount;
    plan.unresolvedPreDispatchDependencyOffset = static_cast<uint32_t>(
        totalUnresolvedPreDispatchDependencyBufferBindings);
    totalUnresolvedPreDispatchDependencyBufferBindings +=
        plan.unresolvedPreDispatchDependencyCount;
    plan.unresolvedDrawOffset =
        static_cast<uint32_t>(totalUnresolvedDrawBufferBindings);
    totalUnresolvedDrawBufferBindings += plan.unresolvedDrawCount;
    plan.unresolvedMeshDispatchOffset =
        static_cast<uint32_t>(totalUnresolvedMeshDispatchBufferBindings);
    totalUnresolvedMeshDispatchBufferBindings +=
        plan.unresolvedMeshDispatchCount;
    plan.unresolvedTextureCopyOffset =
        static_cast<uint32_t>(totalUnresolvedTextureCopyBindings);
    totalUnresolvedTextureCopyBindings += plan.unresolvedTextureCopyCount;
  }
  compiled.resolvedDependencyBuffers.resize(totalDependencyBufferSlots);
  compiled.resolvedDependencyBufferResourceIndices.assign(
      totalDependencyBufferSlots, UINT32_MAX);
  compiled.dependencyBufferRangesByPass.resize(work.order.size());
  compiled.resolvedDependencyTextures.resize(totalDependencyTextureSlots);
  compiled.resolvedDependencyTextureResourceIndices.assign(
      totalDependencyTextureSlots, UINT32_MAX);
  compiled.dependencyTextureRangesByPass.resize(work.order.size());
  compiled.ownedPreDispatches.resize(totalPreDispatchItems);
  compiled.preDispatchRangesByPass.resize(work.order.size());
  compiled.resolvedPreDispatchDependencyBuffers.resize(
      totalPreDispatchDependencySlots);
  compiled.resolvedPreDispatchDependencyBufferResourceIndices.assign(
      totalPreDispatchDependencySlots, UINT32_MAX);
  compiled.preDispatchDependencyRanges.resize(totalPreDispatchItems);
  compiled.ownedDrawItems.resize(totalDrawItems);
  compiled.drawRangesByPass.resize(work.order.size());
  compiled.ownedMeshDispatchItems.resize(totalMeshDispatchItems);
  compiled.ownedTextureCopyItems.resize(totalTextureCopyItems);
  {
    compiled.ownedMeshDispatchDebugLabels.clear();
    compiled.ownedMeshDispatchPushConstants.clear();
    compiled.ownedMeshDispatchDependencyBuffers.clear();
    compiled.ownedMeshDispatchDependencyTextures.clear();
    compiled.ownedMeshDispatchDebugLabels.reserve(totalMeshDispatchItems);
    compiled.ownedMeshDispatchPushConstants.reserve(totalMeshDispatchItems);
    compiled.ownedMeshDispatchDependencyBuffers.reserve(totalMeshDispatchItems);
    compiled.ownedMeshDispatchDependencyTextures.reserve(
        totalMeshDispatchItems);
    for (size_t i = 0u; i < totalMeshDispatchItems; ++i) {
      compiled.ownedMeshDispatchDebugLabels.emplace_back();
      compiled.ownedMeshDispatchPushConstants.emplace_back();
      compiled.ownedMeshDispatchDependencyBuffers.emplace_back();
      compiled.ownedMeshDispatchDependencyTextures.emplace_back();
    }
  }
  compiled.meshDispatchRangesByPass.resize(work.order.size());
  compiled.textureCopyRangesByPass.resize(work.order.size());
  compiled.orderedPasses.resize(work.order.size());
  compiled.orderedPassIndices.resize(work.order.size());
  compiled.recordedGraphicsPasses.resize(work.order.size());
  compiled.passBarrierPlans.resize(work.order.size());
  compiled.unresolvedTextureBindings.resize(totalUnresolvedTextureBindings);
  compiled.unresolvedDependencyBufferBindings.resize(
      totalUnresolvedDependencyBufferBindings);
  compiled.unresolvedDependencyTextureBindings.resize(
      totalUnresolvedDependencyTextureBindings);
  compiled.unresolvedPreDispatchDependencyBufferBindings.resize(
      totalUnresolvedPreDispatchDependencyBufferBindings);
  compiled.unresolvedDrawBufferBindings.resize(
      totalUnresolvedDrawBufferBindings);
  compiled.unresolvedMeshDispatchBufferBindings.resize(
      totalUnresolvedMeshDispatchBufferBindings);
  compiled.unresolvedTextureCopyBindings.resize(
      totalUnresolvedTextureCopyBindings);
  const auto fillPassRange = [&](uint32_t, RenderGraphContiguousRange range) {
    for (uint32_t orderedPassIndex = range.offset;
         orderedPassIndex < range.offset + range.count; ++orderedPassIndex) {
      const PassResolvePlan &plan = passPlans[orderedPassIndex];
      const uint32_t passIndex = plan.passIndex;
      const RenderPass &sourcePass = passes_[passIndex];
      RenderPass resolvedPass = sourcePass;
      uint32_t unresolvedTextureWriteOffset = plan.unresolvedTextureOffset;
      if (plan.colorTextureIndex != UINT32_MAX) {
        const TextureResource &resource = textures_[plan.colorTextureIndex];
        if (resource.imported) {
          resolvedPass.colorTexture = resource.importedHandle;
        } else {
          resolvedPass.colorTexture = {};
          compiled.unresolvedTextureBindings[unresolvedTextureWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .textureResourceIndex = plan.colorTextureIndex,
              .target =
                  RenderGraphCompileResult::PassTextureBindingTarget::Color};
        }
      }
      if (plan.colorResolveTextureIndex != UINT32_MAX) {
        const TextureResource &resource =
            textures_[plan.colorResolveTextureIndex];
        if (resource.imported) {
          resolvedPass.colorResolveTexture = resource.importedHandle;
        } else {
          resolvedPass.colorResolveTexture = {};
          compiled.unresolvedTextureBindings[unresolvedTextureWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .textureResourceIndex = plan.colorResolveTextureIndex,
              .target = RenderGraphCompileResult::PassTextureBindingTarget::
                  ColorResolve};
        }
      }
      if (plan.depthTextureIndex != UINT32_MAX) {
        const TextureResource &resource = textures_[plan.depthTextureIndex];
        if (resource.imported) {
          resolvedPass.depthTexture = resource.importedHandle;
        } else {
          resolvedPass.depthTexture = {};
          compiled.unresolvedTextureBindings[unresolvedTextureWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .textureResourceIndex = plan.depthTextureIndex,
              .target =
                  RenderGraphCompileResult::PassTextureBindingTarget::Depth};
        }
      }
      if (plan.depthResolveTextureIndex != UINT32_MAX) {
        const TextureResource &resource =
            textures_[plan.depthResolveTextureIndex];
        if (resource.imported) {
          resolvedPass.depthResolveTexture = resource.importedHandle;
        } else {
          resolvedPass.depthResolveTexture = {};
          compiled.unresolvedTextureBindings[unresolvedTextureWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .textureResourceIndex = plan.depthResolveTextureIndex,
              .target = RenderGraphCompileResult::PassTextureBindingTarget::
                  DepthResolve};
        }
      }
      compiled.dependencyBufferRangesByPass[orderedPassIndex] = {
          .offset = plan.resolvedDependencyOffset,
          .count = plan.dependencyCount};
      uint32_t unresolvedDependencyWriteOffset =
          plan.unresolvedDependencyOffset;
      for (uint32_t depIndex = 0; depIndex < plan.dependencyCount; ++depIndex) {
        const uint32_t resourceIndex =
            passDependencyBufferBindingResourceIndices_
                [plan.dependencyBindingOffset + depIndex];
        BufferHandle &resolvedHandle =
            compiled.resolvedDependencyBuffers[plan.resolvedDependencyOffset +
                                               depIndex];
        if (resourceIndex == UINT32_MAX) {
          resolvedHandle = {};
          continue;
        }
        const BufferResource &resource = buffers_[resourceIndex];
        if (resource.imported) {
          resolvedHandle = resource.importedHandle;
          compiled.resolvedDependencyBufferResourceIndices
              [plan.resolvedDependencyOffset + depIndex] = resourceIndex;
        } else {
          resolvedHandle = {};
          compiled.unresolvedDependencyBufferBindings
              [unresolvedDependencyWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .dependencyBufferIndex = depIndex,
              .bufferResourceIndex = resourceIndex};
        }
      }
      if (plan.dependencyCount > 0u) {
        resolvedPass.dependencyBuffers = std::span<const BufferHandle>(
            compiled.resolvedDependencyBuffers.data() +
                plan.resolvedDependencyOffset,
            plan.dependencyCount);
      } else {
        resolvedPass.dependencyBuffers = {};
      }
      compiled.dependencyTextureRangesByPass[orderedPassIndex] = {
          .offset = plan.resolvedDependencyTextureOffset,
          .count = plan.dependencyTextureCount};
      uint32_t unresolvedDependencyTextureWriteOffset =
          plan.unresolvedDependencyTextureOffset;
      for (uint32_t depIndex = 0; depIndex < plan.dependencyTextureCount;
           ++depIndex) {
        const uint32_t resourceIndex =
            passDependencyTextureBindingResourceIndices_
                [plan.dependencyTextureBindingOffset + depIndex];
        TextureHandle &resolvedHandle =
            compiled.resolvedDependencyTextures
                [plan.resolvedDependencyTextureOffset + depIndex];
        if (resourceIndex == UINT32_MAX) {
          resolvedHandle = {};
          continue;
        }
        const TextureResource &resource = textures_[resourceIndex];
        if (resource.imported) {
          resolvedHandle = resource.importedHandle;
          compiled.resolvedDependencyTextureResourceIndices
              [plan.resolvedDependencyTextureOffset + depIndex] = resourceIndex;
        } else {
          resolvedHandle = {};
          compiled.unresolvedDependencyTextureBindings
              [unresolvedDependencyTextureWriteOffset++] = {
              .orderedPassIndex = orderedPassIndex,
              .dependencyTextureIndex = depIndex,
              .textureResourceIndex = resourceIndex};
        }
      }
      if (plan.dependencyTextureCount > 0u) {
        resolvedPass.dependencyTextures = std::span<const TextureHandle>(
            compiled.resolvedDependencyTextures.data() +
                plan.resolvedDependencyTextureOffset,
            plan.dependencyTextureCount);
      } else {
        resolvedPass.dependencyTextures = {};
      }
      compiled.preDispatchRangesByPass[orderedPassIndex] = {
          .offset = plan.preDispatchOutputOffset,
          .count = plan.preDispatchCount};
      uint32_t nextPreDispatchDependencyOffset =
          plan.preDispatchDependencyOffset;
      uint32_t unresolvedPreDispatchWriteOffset =
          plan.unresolvedPreDispatchDependencyOffset;
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
        compiled.preDispatchDependencyRanges[plan.preDispatchOutputOffset +
                                             dispatchIndex] = {
            .offset = nextPreDispatchDependencyOffset,
            .count = dispatchDependencyCount};
        for (uint32_t depIndex = 0; depIndex < dispatchDependencyCount;
             ++depIndex) {
          const uint32_t resourceIndex =
              preDispatchDependencyBindingResourceIndices_
                  [dispatchDependencyOffset + depIndex];
          BufferHandle &resolvedHandle =
              compiled.resolvedPreDispatchDependencyBuffers
                  [nextPreDispatchDependencyOffset + depIndex];
          if (resourceIndex == UINT32_MAX) {
            resolvedHandle = {};
            continue;
          }
          const BufferResource &resource = buffers_[resourceIndex];
          if (resource.imported) {
            resolvedHandle = resource.importedHandle;
            compiled.resolvedPreDispatchDependencyBufferResourceIndices
                [nextPreDispatchDependencyOffset + depIndex] = resourceIndex;
          } else {
            resolvedHandle = {};
            compiled.unresolvedPreDispatchDependencyBufferBindings
                [unresolvedPreDispatchWriteOffset++] = {
                .orderedPassIndex = orderedPassIndex,
                .preDispatchIndex = dispatchIndex,
                .dependencyBufferIndex = depIndex,
                .bufferResourceIndex = resourceIndex};
          }
        }
        if (dispatchDependencyCount > 0u) {
          resolvedDispatch.dependencyBuffers = std::span<const BufferHandle>(
              compiled.resolvedPreDispatchDependencyBuffers.data() +
                  nextPreDispatchDependencyOffset,
              dispatchDependencyCount);
        } else {
          resolvedDispatch.dependencyBuffers = {};
        }
        resolvedDispatch.dependencyBufferAccessModes = {};
        compiled
            .ownedPreDispatches[plan.preDispatchOutputOffset + dispatchIndex] =
            resolvedDispatch;
        nextPreDispatchDependencyOffset += dispatchDependencyCount;
      }
      if (plan.preDispatchCount > 0u) {
        resolvedPass.preDispatches = std::span<const ComputeDispatchItem>(
            compiled.ownedPreDispatches.data() + plan.preDispatchOutputOffset,
            plan.preDispatchCount);
      } else {
        resolvedPass.preDispatches = {};
      }
      if (plan.drawCount > 0u && plan.unresolvedDrawCount == 0u &&
          sourcePass.payloadBorrowed) {
        compiled.drawRangesByPass[orderedPassIndex] = {};
        resolvedPass.draws = sourcePass.draws;
      } else if (plan.unresolvedDrawCount == 0u) {
        compiled.drawRangesByPass[orderedPassIndex] = {
            .offset = plan.drawOutputOffset, .count = plan.quantizedDrawCount};
        std::ranges::copy(sourcePass.draws, compiled.ownedDrawItems.begin() +
                                                plan.drawOutputOffset);
        resolvedPass.draws =
            plan.drawCount == 0u
                ? std::span<const DrawItem>{}
                : std::span<const DrawItem>(compiled.ownedDrawItems.data() +
                                                plan.drawOutputOffset,
                                            plan.drawCount);
      } else {
        compiled.drawRangesByPass[orderedPassIndex] = {
            .offset = plan.drawOutputOffset, .count = plan.quantizedDrawCount};
        uint32_t unresolvedDrawWriteOffset = plan.unresolvedDrawOffset;
        for (uint32_t drawIndex = 0; drawIndex < plan.drawCount; ++drawIndex) {
          const DrawItem &sourceDraw = sourcePass.draws[drawIndex];
          DrawItem resolvedDraw = sourceDraw;
          const uint32_t globalDrawIndex = plan.drawBindingOffset + drawIndex;
          const auto resolveDrawBinding =
              [&](uint32_t resourceIndex,
                  RenderGraphCompileResult::DrawBufferBindingTarget target,
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
                compiled
                    .unresolvedDrawBufferBindings[unresolvedDrawWriteOffset++] =
                    {.orderedPassIndex = orderedPassIndex,
                     .drawIndex = drawIndex,
                     .target = target,
                     .bufferResourceIndex = resourceIndex};
              };
          resolveDrawBinding(
              drawBindings_[globalDrawIndex].vertex,
              RenderGraphCompileResult::DrawBufferBindingTarget::Vertex,
              resolvedDraw.vertexBuffer);
          resolveDrawBinding(
              drawBindings_[globalDrawIndex].index,
              RenderGraphCompileResult::DrawBufferBindingTarget::Index,
              resolvedDraw.indexBuffer);
          resolveDrawBinding(
              drawBindings_[globalDrawIndex].indirect,
              RenderGraphCompileResult::DrawBufferBindingTarget::Indirect,
              resolvedDraw.indirectBuffer);
          resolveDrawBinding(
              drawBindings_[globalDrawIndex].indirectCount,
              RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount,
              resolvedDraw.indirectCountBuffer);
          compiled.ownedDrawItems[plan.drawOutputOffset + drawIndex] =
              resolvedDraw;
        }
        if (plan.drawCount > 0u) {
          resolvedPass.draws = std::span<const DrawItem>(
              compiled.ownedDrawItems.data() + plan.drawOutputOffset,
              plan.drawCount);
        } else {
          resolvedPass.draws = {};
        }
      }
      compiled.meshDispatchRangesByPass[orderedPassIndex] = {
          .offset = plan.meshDispatchOutputOffset,
          .count = plan.meshDispatchCount};
      if (plan.meshDispatchCount > 0u) {
        uint32_t unresolvedMeshDispatchWriteOffset =
            plan.unresolvedMeshDispatchOffset;
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
                  RenderGraphCompileResult::MeshDispatchBufferBindingTarget
                      target,
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
                compiled.unresolvedMeshDispatchBufferBindings
                    [unresolvedMeshDispatchWriteOffset++] = {
                    .orderedPassIndex = orderedPassIndex,
                    .meshDispatchIndex = dispatchIndex,
                    .target = target,
                    .bufferResourceIndex = resourceIndex};
              };
          resolveMeshDispatchBinding(
              meshDispatchBindings_[globalDispatchIndex].indirect,
              RenderGraphCompileResult::MeshDispatchBufferBindingTarget::
                  Indirect,
              resolvedDispatch.indirectBuffer);
          resolveMeshDispatchBinding(
              meshDispatchBindings_[globalDispatchIndex].indirectCount,
              RenderGraphCompileResult::MeshDispatchBufferBindingTarget::
                  IndirectCount,
              resolvedDispatch.indirectCountBuffer);
          std::pmr::string &debugLabel =
              compiled.ownedMeshDispatchDebugLabels[outputIndex];
          debugLabel.assign(sourceDispatch.debugLabel.data(),
                            sourceDispatch.debugLabel.size());
          resolvedDispatch.debugLabel =
              std::string_view(debugLabel.data(), debugLabel.size());
          std::pmr::vector<std::byte> &pushConstants =
              compiled.ownedMeshDispatchPushConstants[outputIndex];
          pushConstants.assign(sourceDispatch.pushConstants.begin(),
                               sourceDispatch.pushConstants.end());
          resolvedDispatch.pushConstants = std::span<const std::byte>(
              pushConstants.data(), pushConstants.size());
          std::pmr::vector<BufferHandle> &dependencyBuffers =
              compiled.ownedMeshDispatchDependencyBuffers[outputIndex];
          dependencyBuffers.assign(sourceDispatch.dependencyBuffers.begin(),
                                   sourceDispatch.dependencyBuffers.end());
          resolvedDispatch.dependencyBuffers = std::span<const BufferHandle>(
              dependencyBuffers.data(), dependencyBuffers.size());
          std::pmr::vector<TextureHandle> &dependencyTextures =
              compiled.ownedMeshDispatchDependencyTextures[outputIndex];
          dependencyTextures.assign(sourceDispatch.dependencyTextures.begin(),
                                    sourceDispatch.dependencyTextures.end());
          resolvedDispatch.dependencyTextures = std::span<const TextureHandle>(
              dependencyTextures.data(), dependencyTextures.size());
          compiled.ownedMeshDispatchItems[outputIndex] = resolvedDispatch;
        }
        resolvedPass.meshDispatches = std::span<const MeshDispatchItem>(
            compiled.ownedMeshDispatchItems.data() +
                plan.meshDispatchOutputOffset,
            plan.meshDispatchCount);
      } else {
        resolvedPass.meshDispatches = {};
      }
      compiled.textureCopyRangesByPass[orderedPassIndex] = {
          .offset = plan.textureCopyOutputOffset,
          .count = plan.textureCopyCount};
      if (plan.textureCopyCount > 0u) {
        uint32_t unresolvedTextureCopyWriteOffset =
            plan.unresolvedTextureCopyOffset;
        for (uint32_t copyIndex = 0; copyIndex < plan.textureCopyCount;
             ++copyIndex) {
          TextureCopyItem resolvedCopy = sourcePass.textureCopies[copyIndex];
          const uint32_t globalCopyIndex =
              plan.textureCopyBindingOffset + copyIndex;
          const auto resolveTextureCopyBinding =
              [&](uint32_t resourceIndex,
                  RenderGraphCompileResult::TextureCopyBindingTarget target,
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
                compiled.unresolvedTextureCopyBindings
                    [unresolvedTextureCopyWriteOffset++] = {
                    .orderedPassIndex = orderedPassIndex,
                    .textureCopyIndex = copyIndex,
                    .target = target,
                    .textureResourceIndex = resourceIndex};
              };
          resolveTextureCopyBinding(
              textureCopyBindings_[globalCopyIndex].source,
              RenderGraphCompileResult::TextureCopyBindingTarget::Source,
              resolvedCopy.sourceTexture);
          resolveTextureCopyBinding(
              textureCopyBindings_[globalCopyIndex].destination,
              RenderGraphCompileResult::TextureCopyBindingTarget::Destination,
              resolvedCopy.destinationTexture);
          compiled
              .ownedTextureCopyItems[plan.textureCopyOutputOffset + copyIndex] =
              resolvedCopy;
        }
        resolvedPass.textureCopies = std::span<const TextureCopyItem>(
            compiled.ownedTextureCopyItems.data() +
                plan.textureCopyOutputOffset,
            plan.textureCopyCount);
      } else {
        resolvedPass.textureCopies = {};
      }
      const std::pmr::string &compiledName = compiled.passDebugNames[passIndex];
      resolvedPass.debugLabel =
          std::string_view(compiledName.data(), compiledName.size());
      compiled.orderedPassIndices[orderedPassIndex] = passIndex;
      compiled.recordedGraphicsPasses[orderedPassIndex] = {
          .orderedPassIndex = orderedPassIndex, .declaredPassIndex = passIndex};
      compiled.passBarrierPlans[orderedPassIndex] = {.orderedPassIndex =
                                                         orderedPassIndex,
                                                     .barrierOffset = 0u,
                                                     .barrierCount = 0u};
      compiled.orderedPasses[orderedPassIndex] = resolvedPass;
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
    RenderGraphCompileResult &compiled,
    const RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_BARRIER);
  compiled.passBarrierRecords.clear();
  compiled.finalBarrierPlan = {};
  std::pmr::vector<uint32_t> executionRankByPass(memory_);
  executionRankByPass.resize(work.passCount, UINT32_MAX);
  for (uint32_t rank = 0; rank < work.order.size(); ++rank) {
    executionRankByPass[work.order[rank]] = rank;
  }
  std::pmr::vector<PassResourceAccess> orderedAccesses(memory_);
  orderedAccesses.reserve(work.compiledAccesses.size());
  for (const PassResourceAccess &access : work.compiledAccesses) {
    if (access.passIndex >= work.passCount ||
        work.activePassMask[access.passIndex] == 0u ||
        executionRankByPass[access.passIndex] == UINT32_MAX) {
      continue;
    }
    orderedAccesses.push_back(access);
  }
  std::sort(orderedAccesses.begin(), orderedAccesses.end(),
            [&executionRankByPass](const PassResourceAccess &lhs,
                                   const PassResourceAccess &rhs) {
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
  const auto resolveResourceState = [&](const PassResourceAccess &access) {
    if (access.requestedState != RenderGraphResourceState::Unknown) {
      return access.requestedState;
    }
    const bool hasWrite =
        hasAccessFlag(access.mode, RenderGraphAccessMode::Write);
    if (access.resourceKind == AccessResourceKind::Texture &&
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
  AccessResourceKind previousKind = AccessResourceKind::Texture;
  uint32_t previousResourceIndex = UINT32_MAX;
  RenderGraphAccessMode previousAccess = RenderGraphAccessMode::None;
  RenderGraphResourceState previousState = RenderGraphResourceState::Unknown;
  bool havePreviousResource = false;
  for (const PassResourceAccess &access : orderedAccesses) {
    const bool sameResource = havePreviousResource &&
                              previousKind == access.resourceKind &&
                              previousResourceIndex == access.resourceIndex;
    if (!sameResource) {
      previousKind = access.resourceKind;
      previousResourceIndex = access.resourceIndex;
      previousAccess = RenderGraphAccessMode::None;
      previousState = RenderGraphResourceState::Unknown;
      havePreviousResource = true;
    }
    const RenderGraphResourceState nextState = resolveResourceState(access);
    const bool needsBarrier =
        previousState == RenderGraphResourceState::Unknown ||
        previousState != nextState ||
        hasAccessFlag(previousAccess, RenderGraphAccessMode::Write) ||
        hasAccessFlag(access.mode, RenderGraphAccessMode::Write);
    if (needsBarrier) {
      const uint32_t orderedPassIndex = executionRankByPass[access.passIndex];
      stagedBarrierRecords.push_back(RenderGraphBarrierRecord{
          .resourceKind =
              [&]() {
                switch (access.resourceKind) {
                case AccessResourceKind::Texture:
                  return RenderGraphBarrierResourceKind::Texture;
                case AccessResourceKind::Buffer:
                  return RenderGraphBarrierResourceKind::Buffer;
                case AccessResourceKind::AccelerationStructure:
                  return RenderGraphBarrierResourceKind::AccelerationStructure;
                }
                return RenderGraphBarrierResourceKind::Texture;
              }(),
          .resourceIndex = access.resourceIndex,
          .beforeAccess = previousAccess,
          .afterAccess = access.mode,
          .beforeState = previousState,
          .afterState = nextState,
      });
      stagedBarrierPassIndices.push_back(orderedPassIndex);
      ++barrierCounts[orderedPassIndex];
    }
    if (access.resourceKind == AccessResourceKind::Texture &&
        access.resourceIndex < textures_.size()) {
      lastTextureAccessByResource[access.resourceIndex] = access.mode;
      lastTextureStateByResource[access.resourceIndex] = nextState;
      hasLastTextureAccess[access.resourceIndex] = 1u;
    }
    previousAccess = access.mode;
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
  compiled.passBarrierRecords.resize(stagedBarrierRecords.size() +
                                     stagedFinalBarrierRecords.size());
  std::pmr::vector<uint32_t> nextBarrierOffset(memory_);
  nextBarrierOffset.resize(compiled.passBarrierPlans.size(), 0u);
  uint32_t runningBarrierOffset = 0u;
  for (uint32_t orderedPassIndex = 0u;
       orderedPassIndex < compiled.passBarrierPlans.size();
       ++orderedPassIndex) {
    compiled.passBarrierPlans[orderedPassIndex].barrierOffset =
        runningBarrierOffset;
    compiled.passBarrierPlans[orderedPassIndex].barrierCount =
        barrierCounts[orderedPassIndex];
    nextBarrierOffset[orderedPassIndex] = runningBarrierOffset;
    runningBarrierOffset += barrierCounts[orderedPassIndex];
  }
  compiled.finalBarrierPlan = FinalBarrierPlan{
      .barrierOffset = runningBarrierOffset,
      .barrierCount = static_cast<uint32_t>(stagedFinalBarrierRecords.size()),
  };
  for (uint32_t i = 0u; i < stagedBarrierRecords.size(); ++i) {
    const uint32_t orderedPassIndex = stagedBarrierPassIndices[i];
    compiled.passBarrierRecords[nextBarrierOffset[orderedPassIndex]++] =
        stagedBarrierRecords[i];
  }
  for (uint32_t i = 0u; i < stagedFinalBarrierRecords.size(); ++i) {
    compiled.passBarrierRecords[compiled.finalBarrierPlan.barrierOffset + i] =
        stagedFinalBarrierRecords[i];
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RenderGraphBuilder::compileStageC5PlanTransientLifetimes(
    RenderGraphRuntime &runtime, RenderGraphCompileResult &compiled,
    RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  std::pmr::vector<uint32_t> executionRankByPass(memory_);
  executionRankByPass.resize(work.passCount, UINT32_MAX);
  for (uint32_t rank = 0; rank < work.order.size(); ++rank) {
    executionRankByPass[work.order[rank]] = rank;
  }
  std::pmr::vector<uint32_t> transientTextureFirstRank(memory_);
  std::pmr::vector<uint32_t> transientTextureLastRank(memory_);
  transientTextureFirstRank.resize(textures_.size(), UINT32_MAX);
  transientTextureLastRank.resize(textures_.size(), 0u);
  std::pmr::vector<uint32_t> transientBufferFirstRank(memory_);
  std::pmr::vector<uint32_t> transientBufferLastRank(memory_);
  transientBufferFirstRank.resize(buffers_.size(), UINT32_MAX);
  transientBufferLastRank.resize(buffers_.size(), 0u);
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
      const PassResourceAccess &access = work.compiledAccesses[accessIndex];
      if (access.passIndex >= work.passCount ||
          work.activePassMask[access.passIndex] == 0u) {
        continue;
      }
      const uint32_t rank = executionRankByPass[access.passIndex];
      if (rank == UINT32_MAX) {
        continue;
      }
      if (access.resourceKind == AccessResourceKind::Texture) {
        if (access.resourceIndex >= textures_.size() ||
            textures_[access.resourceIndex].imported) {
          continue;
        }
        updateLifetimeRanks(textureFirstRanks, textureLastRanks,
                            access.resourceIndex, rank);
        continue;
      }
      if (access.resourceKind == AccessResourceKind::Buffer) {
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
    std::pmr::vector<uint32_t> textureFirst;
    std::pmr::vector<uint32_t> textureLast;
    std::pmr::vector<uint32_t> bufferFirst;
    std::pmr::vector<uint32_t> bufferLast;
    WorkerLifetimeRanks(std::pmr::memory_resource *memory, size_t textureCount,
                        size_t bufferCount)
        : textureFirst(memory), textureLast(memory), bufferFirst(memory),
          bufferLast(memory) {
      textureFirst.resize(textureCount, UINT32_MAX);
      textureLast.resize(textureCount, 0u);
      bufferFirst.resize(bufferCount, UINT32_MAX);
      bufferLast.resize(bufferCount, 0u);
    }
  };
  bool usedParallelLifetimeAnalysis = false;
  if (!work.compiledAccesses.empty()) {
    const std::vector<RenderGraphContiguousRange> stdRanges =
        runtime.parallelCompileEnabled() && work.compiledAccesses.size() > 1u
            ? makeLifetimeRanges(
                  static_cast<uint32_t>(work.compiledAccesses.size()),
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
            analyzeAccessRange(std::span<uint32_t>(worker.textureFirst.data(),
                                                   worker.textureFirst.size()),
                               std::span<uint32_t>(worker.textureLast.data(),
                                                   worker.textureLast.size()),
                               std::span<uint32_t>(worker.bufferFirst.data(),
                                                   worker.bufferFirst.size()),
                               std::span<uint32_t>(worker.bufferLast.data(),
                                                   worker.bufferLast.size()),
                               range);
          });
      usedParallelLifetimeAnalysis = true;
      for (const WorkerLifetimeRanks &worker : workerRanks) {
        for (uint32_t textureIndex = 0; textureIndex < textures_.size();
             ++textureIndex) {
          if (worker.textureFirst[textureIndex] == UINT32_MAX) {
            continue;
          }
          updateLifetimeRanks(
              std::span<uint32_t>(transientTextureFirstRank.data(),
                                  transientTextureFirstRank.size()),
              std::span<uint32_t>(transientTextureLastRank.data(),
                                  transientTextureLastRank.size()),
              textureIndex, worker.textureFirst[textureIndex]);
          transientTextureLastRank[textureIndex] =
              std::max(transientTextureLastRank[textureIndex],
                       worker.textureLast[textureIndex]);
        }
        for (uint32_t bufferIndex = 0; bufferIndex < buffers_.size();
             ++bufferIndex) {
          if (worker.bufferFirst[bufferIndex] == UINT32_MAX) {
            continue;
          }
          updateLifetimeRanks(
              std::span<uint32_t>(transientBufferFirstRank.data(),
                                  transientBufferFirstRank.size()),
              std::span<uint32_t>(transientBufferLastRank.data(),
                                  transientBufferLastRank.size()),
              bufferIndex, worker.bufferFirst[bufferIndex]);
          transientBufferLastRank[bufferIndex] =
              std::max(transientBufferLastRank[bufferIndex],
                       worker.bufferLast[bufferIndex]);
        }
      }
    } else {
      analyzeAccessRange(
          std::span<uint32_t>(transientTextureFirstRank.data(),
                              transientTextureFirstRank.size()),
          std::span<uint32_t>(transientTextureLastRank.data(),
                              transientTextureLastRank.size()),
          std::span<uint32_t>(transientBufferFirstRank.data(),
                              transientBufferFirstRank.size()),
          std::span<uint32_t>(transientBufferLastRank.data(),
                              transientBufferLastRank.size()),
          RenderGraphContiguousRange{
              .offset = 0u,
              .count = static_cast<uint32_t>(work.compiledAccesses.size())});
    }
  }
  compiled.usedParallelHazardAnalysis = work.usedParallelHazardAnalysis;
  compiled.usedParallelLifetimeAnalysis = usedParallelLifetimeAnalysis;
  compiled.usedParallelCompile = compiled.usedParallelPayloadResolution ||
                                 compiled.usedParallelHazardAnalysis ||
                                 compiled.usedParallelLifetimeAnalysis;
  for (uint32_t textureIndex = 0; textureIndex < textures_.size();
       ++textureIndex) {
    if (textures_[textureIndex].imported ||
        transientTextureFirstRank[textureIndex] == UINT32_MAX) {
      continue;
    }
    compiled.transientTextureLifetimes.push_back(
        {.resourceIndex = textureIndex,
         .firstExecutionIndex = transientTextureFirstRank[textureIndex],
         .lastExecutionIndex = transientTextureLastRank[textureIndex]});
  }
  for (uint32_t bufferIndex = 0; bufferIndex < buffers_.size(); ++bufferIndex) {
    if (buffers_[bufferIndex].imported ||
        transientBufferFirstRank[bufferIndex] == UINT32_MAX) {
      continue;
    }
    compiled.transientBufferLifetimes.push_back(
        {.resourceIndex = bufferIndex,
         .firstExecutionIndex = transientBufferFirstRank[bufferIndex],
         .lastExecutionIndex = transientBufferLastRank[bufferIndex]});
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
RenderGraphBuilder::compileStageC6PlanTransientAliasing(
    RenderGraphCompileResult &compiled) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  {
    NURI_PROFILER_ZONE("RenderGraph.compile.plan_texture_aliasing",
                       NURI_PROFILER_COLOR_CREATE);
    std::pmr::vector<uint32_t> orderIndices(memory_);
    orderIndices.resize(compiled.transientTextureLifetimes.size(), 0u);
    std::iota(orderIndices.begin(), orderIndices.end(), 0u);
    std::sort(orderIndices.begin(), orderIndices.end(),
              [&compiled](uint32_t lhs, uint32_t rhs) {
                const auto &a = compiled.transientTextureLifetimes[lhs];
                const auto &b = compiled.transientTextureLifetimes[rhs];
                if (a.firstExecutionIndex != b.firstExecutionIndex) {
                  return a.firstExecutionIndex < b.firstExecutionIndex;
                }
                if (a.lastExecutionIndex != b.lastExecutionIndex) {
                  return a.lastExecutionIndex < b.lastExecutionIndex;
                }
                return a.resourceIndex < b.resourceIndex;
              });
    std::pmr::vector<uint32_t> slotLastUse(memory_);
    std::pmr::vector<uint32_t> slotRepresentativeResource(memory_);
    slotLastUse.reserve(compiled.transientTextureLifetimes.size());
    slotRepresentativeResource.reserve(
        compiled.transientTextureLifetimes.size());
    for (const uint32_t lifetimeIndex : orderIndices) {
      const auto &lifetime = compiled.transientTextureLifetimes[lifetimeIndex];
      uint32_t chosenSlot = UINT32_MAX;
      for (uint32_t slot = 0; slot < slotLastUse.size(); ++slot) {
        if (slotLastUse[slot] >= lifetime.firstExecutionIndex) {
          continue;
        }
        const uint32_t representative = slotRepresentativeResource[slot];
        if (representative >= textures_.size() ||
            lifetime.resourceIndex >= textures_.size()) {
          continue;
        }
        if (!isTextureDescAliasCompatible(
                textures_[representative].transientDesc,
                textures_[lifetime.resourceIndex].transientDesc)) {
          continue;
        }
        chosenSlot = slot;
        break;
      }
      if (chosenSlot == UINT32_MAX) {
        chosenSlot = static_cast<uint32_t>(slotLastUse.size());
        slotLastUse.push_back(lifetime.lastExecutionIndex);
        slotRepresentativeResource.push_back(lifetime.resourceIndex);
        TextureDesc desc = textures_[lifetime.resourceIndex].transientDesc;
        desc.data = {};
        compiled.transientTexturePhysicalAllocations.push_back(
            RenderGraphCompileResult::TransientTexturePhysicalAllocation{
                .allocationIndex = chosenSlot,
                .representativeResourceIndex = lifetime.resourceIndex,
                .desc = desc,
            });
      } else {
        slotLastUse[chosenSlot] = lifetime.lastExecutionIndex;
      }
      compiled.transientTextureAllocationByResource[lifetime.resourceIndex] =
          chosenSlot;
      compiled.transientTextureAllocations.push_back(
          RenderGraphCompileResult::TransientAllocation{
              .resourceIndex = lifetime.resourceIndex,
              .allocationIndex = chosenSlot,
          });
    }
    std::sort(compiled.transientTextureAllocations.begin(),
              compiled.transientTextureAllocations.end(),
              [](const auto &a, const auto &b) {
                return a.resourceIndex < b.resourceIndex;
              });
    compiled.transientTexturePhysicalCount = static_cast<uint32_t>(
        compiled.transientTexturePhysicalAllocations.size());
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("RenderGraph.compile.plan_buffer_aliasing",
                       NURI_PROFILER_COLOR_CREATE);
    std::pmr::vector<uint32_t> orderIndices(memory_);
    orderIndices.resize(compiled.transientBufferLifetimes.size(), 0u);
    std::iota(orderIndices.begin(), orderIndices.end(), 0u);
    std::sort(orderIndices.begin(), orderIndices.end(),
              [&compiled](uint32_t lhs, uint32_t rhs) {
                const auto &a = compiled.transientBufferLifetimes[lhs];
                const auto &b = compiled.transientBufferLifetimes[rhs];
                if (a.firstExecutionIndex != b.firstExecutionIndex) {
                  return a.firstExecutionIndex < b.firstExecutionIndex;
                }
                if (a.lastExecutionIndex != b.lastExecutionIndex) {
                  return a.lastExecutionIndex < b.lastExecutionIndex;
                }
                return a.resourceIndex < b.resourceIndex;
              });
    std::pmr::vector<uint32_t> slotLastUse(memory_);
    std::pmr::vector<uint32_t> slotRepresentativeResource(memory_);
    slotLastUse.reserve(compiled.transientBufferLifetimes.size());
    slotRepresentativeResource.reserve(
        compiled.transientBufferLifetimes.size());
    for (const uint32_t lifetimeIndex : orderIndices) {
      const auto &lifetime = compiled.transientBufferLifetimes[lifetimeIndex];
      uint32_t chosenSlot = UINT32_MAX;
      for (uint32_t slot = 0; slot < slotLastUse.size(); ++slot) {
        if (slotLastUse[slot] >= lifetime.firstExecutionIndex) {
          continue;
        }
        const uint32_t representative = slotRepresentativeResource[slot];
        if (representative >= buffers_.size() ||
            lifetime.resourceIndex >= buffers_.size()) {
          continue;
        }
        if (!isBufferDescAliasCompatible(
                buffers_[representative].transientDesc,
                buffers_[lifetime.resourceIndex].transientDesc)) {
          continue;
        }
        chosenSlot = slot;
        break;
      }
      if (chosenSlot == UINT32_MAX) {
        chosenSlot = static_cast<uint32_t>(slotLastUse.size());
        slotLastUse.push_back(lifetime.lastExecutionIndex);
        slotRepresentativeResource.push_back(lifetime.resourceIndex);
        BufferDesc desc = buffers_[lifetime.resourceIndex].transientDesc;
        desc.data = {};
        compiled.transientBufferPhysicalAllocations.push_back(
            RenderGraphCompileResult::TransientBufferPhysicalAllocation{
                .allocationIndex = chosenSlot,
                .representativeResourceIndex = lifetime.resourceIndex,
                .desc = desc,
            });
      } else {
        slotLastUse[chosenSlot] = lifetime.lastExecutionIndex;
      }
      compiled.transientBufferAllocationByResource[lifetime.resourceIndex] =
          chosenSlot;
      compiled.transientBufferAllocations.push_back(
          RenderGraphCompileResult::TransientAllocation{
              .resourceIndex = lifetime.resourceIndex,
              .allocationIndex = chosenSlot,
          });
    }
    std::sort(compiled.transientBufferAllocations.begin(),
              compiled.transientBufferAllocations.end(),
              [](const auto &a, const auto &b) {
                return a.resourceIndex < b.resourceIndex;
              });
    compiled.transientBufferPhysicalCount = static_cast<uint32_t>(
        compiled.transientBufferPhysicalAllocations.size());
    NURI_PROFILER_ZONE_END();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<RenderGraphCompileResult, std::string>
RenderGraphBuilder::compile(RenderGraphRuntime &runtime) const {
  NURI_PROFILER_FUNCTION();
  RenderGraphCompileResult compiled(memory_);
  compiled.frameIndex = frameIndex_;
  CompileWorkState work(memory_);
  compileStageC0BuildResourceTables(compiled, work);
  if (passes_.empty()) {
    return Result<RenderGraphCompileResult, std::string>::makeResult(
        std::move(compiled));
  }
  auto topologyResult = compileStageC1C2BuildTopology(runtime, compiled, work);
  if (topologyResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        topologyResult.error());
  }
  auto resolveResult =
      compileStageC3ResolvePassPayloads(runtime, compiled, work);
  if (resolveResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        resolveResult.error());
  }
  auto barrierResult = compileStageC4PlanBarriers(compiled, work);
  if (barrierResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        barrierResult.error());
  }
  auto lifetimeResult =
      compileStageC5PlanTransientLifetimes(runtime, compiled, work);
  if (lifetimeResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        lifetimeResult.error());
  }
  compiled.transientTextureAllocationByResource.resize(textures_.size(),
                                                       UINT32_MAX);
  compiled.transientBufferAllocationByResource.resize(buffers_.size(),
                                                      UINT32_MAX);
  auto aliasingResult = compileStageC6PlanTransientAliasing(compiled);
  if (aliasingResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        aliasingResult.error());
  }
  return Result<RenderGraphCompileResult, std::string>::makeResult(
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
                             const RenderGraphCompileResult &compiled,
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

Result<bool, std::string>
RenderGraphExecutor::executeInternal(RenderGraphRuntime *runtime,
                                     GPUDevice &gpu,
                                     const RenderGraphCompileResult &compiled,
                                     RenderGraphExecutionMetadata &metadata,
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
  transientTextureHandles.resize(compiled.transientTexturePhysicalCount,
                                 TextureHandle{});
  std::pmr::vector<TextureDesc> transientTextureDescs(memory_);
  transientTextureDescs.resize(compiled.transientTexturePhysicalCount,
                               TextureDesc{});
  std::pmr::vector<BufferHandle> transientBufferHandles(memory_);
  transientBufferHandles.resize(compiled.transientBufferPhysicalCount,
                                BufferHandle{});
  std::pmr::vector<BufferDesc> transientBufferDescs(memory_);
  transientBufferDescs.resize(compiled.transientBufferPhysicalCount,
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
         compiled.transientTexturePhysicalAllocations) {
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
    for (const auto &allocation : compiled.transientBufferPhysicalAllocations) {
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
  std::pmr::vector<RenderPass> executablePasses(memory_);
  std::pmr::vector<BufferHandle> executableDependencyBuffers(memory_);
  std::pmr::vector<TextureHandle> executableDependencyTextures(memory_);
  std::pmr::vector<ComputeDispatchItem> executablePreDispatches(memory_);
  std::pmr::vector<DrawItem> executableDrawItems(memory_);
  std::pmr::vector<MeshDispatchItem> executableMeshDispatches(memory_);
  std::pmr::vector<TextureCopyItem> executableTextureCopies(memory_);
  std::pmr::vector<BufferHandle> executablePreDispatchDependencyBuffers(
      memory_);
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.build_executable_payload",
                       NURI_PROFILER_COLOR_CMD_COPY);
    const bool needsMutableDependencyBuffers =
        !compiled.unresolvedDependencyBufferBindings.empty();
    const bool needsMutableDependencyTextures =
        !compiled.unresolvedDependencyTextureBindings.empty();
    const bool needsMutableDrawItems =
        !compiled.unresolvedDrawBufferBindings.empty();
    const bool needsMutableTextureCopies =
        !compiled.unresolvedTextureCopyBindings.empty();
    executablePasses = compiled.orderedPasses;
    if (needsMutableDependencyBuffers) {
      executableDependencyBuffers = compiled.resolvedDependencyBuffers;
    }
    if (needsMutableDependencyTextures) {
      executableDependencyTextures = compiled.resolvedDependencyTextures;
    }
    executablePreDispatches = compiled.ownedPreDispatches;
    if (needsMutableDrawItems) {
      executableDrawItems = compiled.ownedDrawItems;
    }
    executableMeshDispatches = compiled.ownedMeshDispatchItems;
    if (needsMutableTextureCopies) {
      executableTextureCopies = compiled.ownedTextureCopyItems;
    }
    executablePreDispatchDependencyBuffers =
        compiled.resolvedPreDispatchDependencyBuffers;
    for (uint32_t passIndex = 0; passIndex < executablePasses.size();
         ++passIndex) {
      RenderPass &pass = executablePasses[passIndex];
      const auto &bufferRange =
          compiled.dependencyBufferRangesByPass[passIndex];
      if (needsMutableDependencyBuffers) {
        pass.dependencyBuffers =
            std::span<const BufferHandle>(executableDependencyBuffers)
                .subspan(bufferRange.offset, bufferRange.count);
      }
      const auto &textureRange =
          compiled.dependencyTextureRangesByPass[passIndex];
      if (needsMutableDependencyTextures) {
        pass.dependencyTextures =
            std::span<const TextureHandle>(executableDependencyTextures)
                .subspan(textureRange.offset, textureRange.count);
      }
      const auto &preDispatchRange =
          compiled.preDispatchRangesByPass[passIndex];
      pass.preDispatches =
          std::span<const ComputeDispatchItem>(executablePreDispatches)
              .subspan(preDispatchRange.offset, preDispatchRange.count);
      for (uint32_t i = 0; i < preDispatchRange.count; ++i) {
        const uint32_t dispatchIndex = preDispatchRange.offset + i;
        const auto &range = compiled.preDispatchDependencyRanges[dispatchIndex];
        executablePreDispatches[dispatchIndex].dependencyBuffers =
            std::span<const BufferHandle>(
                executablePreDispatchDependencyBuffers)
                .subspan(range.offset, range.count);
      }
      const auto &drawRange = compiled.drawRangesByPass[passIndex];
      if (needsMutableDrawItems) {
        pass.draws = std::span<const DrawItem>(executableDrawItems)
                         .subspan(drawRange.offset, drawRange.count);
      }
      const auto &meshRange = compiled.meshDispatchRangesByPass[passIndex];
      pass.meshDispatches =
          std::span<const MeshDispatchItem>(executableMeshDispatches)
              .subspan(meshRange.offset, meshRange.count);
      const auto &copyRange = compiled.textureCopyRangesByPass[passIndex];
      if (needsMutableTextureCopies) {
        pass.textureCopies =
            std::span<const TextureCopyItem>(executableTextureCopies)
                .subspan(copyRange.offset, copyRange.count);
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.patch_unresolved_bindings",
                       NURI_PROFILER_COLOR_CMD_COPY);
    const auto transientTexture = [&](uint32_t resourceIndex) {
      return transientTextureHandles
          [compiled.transientTextureAllocationByResource[resourceIndex]];
    };
    const auto transientBuffer = [&](uint32_t resourceIndex) {
      return transientBufferHandles
          [compiled.transientBufferAllocationByResource[resourceIndex]];
    };
    for (const auto &binding : compiled.unresolvedTextureBindings) {
      RenderPass &pass = executablePasses[binding.orderedPassIndex];
      const TextureHandle texture =
          transientTexture(binding.textureResourceIndex);
      switch (binding.target) {
      case RenderGraphCompileResult::PassTextureBindingTarget::Color:
        pass.colorTexture = texture;
        break;
      case RenderGraphCompileResult::PassTextureBindingTarget::Depth:
        pass.depthTexture = texture;
        break;
      case RenderGraphCompileResult::PassTextureBindingTarget::ColorResolve:
        pass.colorResolveTexture = texture;
        break;
      case RenderGraphCompileResult::PassTextureBindingTarget::DepthResolve:
        pass.depthResolveTexture = texture;
        break;
      }
    }
    for (const auto &binding : compiled.unresolvedDependencyBufferBindings) {
      const auto &range =
          compiled.dependencyBufferRangesByPass[binding.orderedPassIndex];
      executableDependencyBuffers[range.offset +
                                  binding.dependencyBufferIndex] =
          transientBuffer(binding.bufferResourceIndex);
    }
    for (const auto &binding : compiled.unresolvedDependencyTextureBindings) {
      const auto &range =
          compiled.dependencyTextureRangesByPass[binding.orderedPassIndex];
      executableDependencyTextures[range.offset +
                                   binding.dependencyTextureIndex] =
          transientTexture(binding.textureResourceIndex);
    }
    for (const auto &binding : compiled.unresolvedTextureCopyBindings) {
      const auto &range =
          compiled.textureCopyRangesByPass[binding.orderedPassIndex];
      TextureCopyItem &copy =
          executableTextureCopies[range.offset + binding.textureCopyIndex];
      const TextureHandle texture =
          transientTexture(binding.textureResourceIndex);
      switch (binding.target) {
      case RenderGraphCompileResult::TextureCopyBindingTarget::Source:
        copy.sourceTexture = texture;
        break;
      case RenderGraphCompileResult::TextureCopyBindingTarget::Destination:
        copy.destinationTexture = texture;
        break;
      }
    }
    for (const auto &binding :
         compiled.unresolvedPreDispatchDependencyBufferBindings) {
      const auto &passRange =
          compiled.preDispatchRangesByPass[binding.orderedPassIndex];
      const auto &range =
          compiled.preDispatchDependencyRanges[passRange.offset +
                                               binding.preDispatchIndex];
      executablePreDispatchDependencyBuffers[range.offset +
                                             binding.dependencyBufferIndex] =
          transientBuffer(binding.bufferResourceIndex);
    }
    for (const auto &binding : compiled.unresolvedDrawBufferBindings) {
      const auto &range = compiled.drawRangesByPass[binding.orderedPassIndex];
      DrawItem &draw = executableDrawItems[range.offset + binding.drawIndex];
      const BufferHandle buffer = transientBuffer(binding.bufferResourceIndex);
      switch (binding.target) {
      case RenderGraphCompileResult::DrawBufferBindingTarget::Vertex:
        draw.vertexBuffer = buffer;
        break;
      case RenderGraphCompileResult::DrawBufferBindingTarget::Index:
        draw.indexBuffer = buffer;
        break;
      case RenderGraphCompileResult::DrawBufferBindingTarget::Indirect:
        draw.indirectBuffer = buffer;
        break;
      case RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount:
        draw.indirectCountBuffer = buffer;
        break;
      }
    }
    for (const auto &binding : compiled.unresolvedMeshDispatchBufferBindings) {
      const auto &range =
          compiled.meshDispatchRangesByPass[binding.orderedPassIndex];
      MeshDispatchItem &dispatch =
          executableMeshDispatches[range.offset + binding.meshDispatchIndex];
      const BufferHandle buffer = transientBuffer(binding.bufferResourceIndex);
      switch (binding.target) {
      case RenderGraphCompileResult::MeshDispatchBufferBindingTarget::Indirect:
        dispatch.indirectBuffer = buffer;
        break;
      case RenderGraphCompileResult::MeshDispatchBufferBindingTarget::
          IndirectCount:
        dispatch.indirectCountBuffer = buffer;
        break;
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  std::pmr::vector<GraphicsBarrierRecord> executableBarrierRecords(memory_);
  executableBarrierRecords.reserve(compiled.passBarrierRecords.size());
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.resolve_barriers",
                       NURI_PROFILER_COLOR_BARRIER);
    for (const RenderGraphBarrierRecord &barrier :
         compiled.passBarrierRecords) {
      if (barrier.resourceKind == RenderGraphBarrierResourceKind::Texture) {
        TextureHandle texture =
            compiled.textureHandlesByResource[barrier.resourceIndex];
        if (!nuri::isValid(texture)) {
          texture =
              transientTextureHandles[compiled
                                          .transientTextureAllocationByResource
                                              [barrier.resourceIndex]];
        }
        executableBarrierRecords.push_back(GraphicsBarrierRecord::ForTexture(
            texture, barrier.beforeAccess, barrier.afterAccess,
            barrier.beforeState, barrier.afterState));
      } else if (barrier.resourceKind ==
                 RenderGraphBarrierResourceKind::Buffer) {
        BufferHandle buffer =
            compiled.bufferHandlesByResource[barrier.resourceIndex];
        if (!nuri::isValid(buffer)) {
          buffer = transientBufferHandles
              [compiled
                   .transientBufferAllocationByResource[barrier.resourceIndex]];
        }
        executableBarrierRecords.push_back(GraphicsBarrierRecord::ForBuffer(
            buffer, barrier.beforeAccess, barrier.afterAccess,
            barrier.beforeState, barrier.afterState));
      } else {
        const AccelerationStructureHandle accelerationStructure =
            compiled
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
      metadata.usedParallelCompile = compiled.usedParallelCompile;
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
        for (uint32_t localIndex = 0u; localIndex < range.count; ++localIndex) {
          if (recordingFailed.load()) {
            return;
          }
          const uint32_t orderedPassIndex = range.offset + localIndex;
          const PassBarrierPlan &barrierPlan =
              compiled.passBarrierPlans[orderedPassIndex];
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
            compiled.finalBarrierPlan.barrierCount > 0u) {
          auto finalBarrierResult = gpu.recordGraphicsBarriers(
              recordingContexts[workerIndex],
              std::span<const GraphicsBarrierRecord>(executableBarrierRecords)
                  .subspan(compiled.finalBarrierPlan.barrierOffset,
                           compiled.finalBarrierPlan.barrierCount));
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
