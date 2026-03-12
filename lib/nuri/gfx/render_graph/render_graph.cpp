#include "nuri/pch.h"

#include "nuri/gfx/render_graph/render_graph.h"

#include "nuri/core/containers/hash_set.h"
#include "nuri/core/profiling.h"

#include <queue>

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

constexpr size_t kMaxReusableTransientTextures = 32u;
constexpr size_t kMaxReusableTransientBuffers = 64u;
constexpr uint32_t kMinValidationItemsPerWorker = 64u;
constexpr uint32_t kMinHazardGroupsPerWorker = 32u;
constexpr uint32_t kMinPayloadPassesPerWorker = 8u;
constexpr uint32_t kMinLifetimeItemsPerWorker = 64u;
constexpr uint32_t kMinRecordingPassesPerWorker = 4u;

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
  case RenderGraphExecutionFailureStage::ValidateCompiledMetadata:
    return "validate_compiled_metadata";
  case RenderGraphExecutionFailureStage::MaterializeTransients:
    return "materialize_transients";
  case RenderGraphExecutionFailureStage::BuildExecutablePayload:
    return "build_executable_payload";
  case RenderGraphExecutionFailureStage::PatchUnresolvedBindings:
    return "patch_unresolved_bindings";
  case RenderGraphExecutionFailureStage::ResolveBarriers:
    return "resolve_barriers";
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
  default:
    return "unknown";
  }
}

RenderGraphBuilder::RenderGraphBuilder(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      textures_(memory_), buffers_(memory_), ownedPassPayloads_(memory_),
      passes_(memory_), passDebugNames_(memory_),
      passColorTextureBindings_(memory_), passDepthTextureBindings_(memory_),
      passDependencyBufferBindingOffsets_(memory_),
      passDependencyBufferBindingCounts_(memory_),
      passDependencyBufferBindingResourceIndices_(memory_),
      passPreDispatchBindingOffsets_(memory_),
      passPreDispatchBindingCounts_(memory_),
      preDispatchDependencyBindingOffsets_(memory_),
      preDispatchDependencyBindingCounts_(memory_),
      preDispatchDependencyBindingResourceIndices_(memory_),
      passDrawBindingOffsets_(memory_), passDrawBindingCounts_(memory_),
      drawVertexBindingResourceIndices_(memory_),
      drawIndexBindingResourceIndices_(memory_),
      drawIndirectBindingResourceIndices_(memory_),
      drawIndirectCountBindingResourceIndices_(memory_),
      importedTextureIndicesByHandle_(memory_),
      importedBufferIndicesByHandle_(memory_),
      explicitTextureAccessIndicesByPassResource_(memory_),
      inferredTextureAccessIndicesByPassResource_(memory_),
      explicitBufferAccessIndicesByPassResource_(memory_),
      inferredBufferAccessIndicesByPassResource_(memory_),
      dependencyEdgeKeys_(memory_), dependencies_(memory_),
      passResourceAccesses_(memory_), frameOutputTextureSet_(memory_),
      frameOutputTextureIndices_(memory_),
      sideEffectMarkIndicesByPass_(memory_), sideEffectPassMarks_(memory_) {}

void RenderGraphBuilder::beginFrame(uint64_t frameIndex) {
  frameIndex_ = frameIndex;
  textures_.clear();
  buffers_.clear();
  ownedPassPayloads_.clear();
  passes_.clear();
  passDebugNames_.clear();
  passColorTextureBindings_.clear();
  passDepthTextureBindings_.clear();
  passDependencyBufferBindingOffsets_.clear();
  passDependencyBufferBindingCounts_.clear();
  passDependencyBufferBindingResourceIndices_.clear();
  passPreDispatchBindingOffsets_.clear();
  passPreDispatchBindingCounts_.clear();
  preDispatchDependencyBindingOffsets_.clear();
  preDispatchDependencyBindingCounts_.clear();
  preDispatchDependencyBindingResourceIndices_.clear();
  passDrawBindingOffsets_.clear();
  passDrawBindingCounts_.clear();
  drawVertexBindingResourceIndices_.clear();
  drawIndexBindingResourceIndices_.clear();
  drawIndirectBindingResourceIndices_.clear();
  drawIndirectCountBindingResourceIndices_.clear();
  importedTextureIndicesByHandle_.clear();
  importedBufferIndicesByHandle_.clear();
  explicitTextureAccessIndicesByPassResource_.clear();
  inferredTextureAccessIndicesByPassResource_.clear();
  explicitBufferAccessIndicesByPassResource_.clear();
  inferredBufferAccessIndicesByPassResource_.clear();
  dependencyEdgeKeys_.clear();
  dependencies_.clear();
  passResourceAccesses_.clear();
  frameOutputTextureSet_.clear();
  frameOutputTextureIndices_.clear();
  sideEffectMarkIndicesByPass_.clear();
  sideEffectPassMarks_.clear();
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
    RenderGraphAccessMode mode, bool inferred) {
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
    merged.mode = merged.mode | mode;
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

RenderGraphBuilder::OwnedPassPayload RenderGraphBuilder::clonePassPayload(
    const RenderGraphGraphicsPassDesc &desc) const {
  OwnedPassPayload ownedPayload(memory_);
  ownedPayload.debugLabel.assign(desc.debugLabel.data(),
                                 desc.debugLabel.size());
  ownedPayload.dependencyBuffers.assign(desc.dependencyBuffers.begin(),
                                        desc.dependencyBuffers.end());

  ownedPayload.preDispatchDebugLabels.reserve(desc.preDispatches.size());
  ownedPayload.preDispatchPushConstants.reserve(desc.preDispatches.size());
  ownedPayload.preDispatchDependencyBuffers.reserve(desc.preDispatches.size());
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

  return ownedPayload;
}

Result<bool, std::string> RenderGraphBuilder::applyImplicitPassRoots(
    RenderGraphPassId pass, const RenderGraphGraphicsPassDesc &desc) {
  if (nuri::isValid(desc.colorTexture)) {
    if (desc.markColorAsFrameOutput) {
      return markTextureAsFrameOutput(desc.colorTexture);
    }
    if (desc.markImplicitOutputSideEffect &&
        textures_[desc.colorTexture.value].imported) {
      return markPassSideEffect(pass);
    }
    return Result<bool, std::string>::makeResult(true);
  }

  if (!desc.markImplicitOutputSideEffect) {
    return Result<bool, std::string>::makeResult(true);
  }
  return markPassSideEffect(pass);
}

Result<bool, std::string> RenderGraphBuilder::bindImplicitPassResources(
    RenderGraphPassId pass, const RenderGraphGraphicsPassDesc &desc) {
  if (nuri::isValid(desc.colorTexture)) {
    auto bindResult = bindPassColorTexture(pass, desc.colorTexture);
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

  const std::string dependencyDebugName =
      makePassResourceDebugName(desc.debugLabel, "dependency_buffer");
  for (size_t i = 0; i < desc.dependencyBuffers.size(); ++i) {
    const BufferHandle dependency = desc.dependencyBuffers[i];
    if (!nuri::isValid(dependency)) {
      continue;
    }

    auto importResult = importBuffer(dependency, dependencyDebugName);
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }

    auto bindResult = bindPassDependencyBuffer(
        pass, static_cast<uint32_t>(i), importResult.value(),
        RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
    if (bindResult.hasError()) {
      return bindResult;
    }
  }

  const std::string preDispatchDependencyDebugName = makePassResourceDebugName(
      desc.debugLabel, "pre_dispatch_dependency_buffer");
  for (size_t dispatchIndex = 0; dispatchIndex < desc.preDispatches.size();
       ++dispatchIndex) {
    const ComputeDispatchItem &dispatch = desc.preDispatches[dispatchIndex];
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
          RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
      if (bindResult.hasError()) {
        return bindResult;
      }
    }
  }

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
             RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount},
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

  return Result<bool, std::string>::makeResult(true);
}

Result<RenderGraphPassId, std::string>
RenderGraphBuilder::addGraphicsPass(const RenderGraphGraphicsPassDesc &desc) {
  RenderPass pass{};
  pass.color = desc.color;
  pass.depth = desc.depth;
  pass.useViewport = desc.useViewport;
  pass.viewport = desc.viewport;
  pass.debugColor = desc.debugColor;

  auto addResult = addPassRecord(pass, clonePassPayload(desc), desc.debugLabel);
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
RenderGraphBuilder::addPassRecord(RenderPass pass,
                                  OwnedPassPayload ownedPayload,
                                  std::string_view debugName) {
  const uint32_t passIndex = static_cast<uint32_t>(passes_.size());
  if (passIndex == UINT32_MAX) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addPassRecord: pass count exceeds uint32_t");
  }

  const uint32_t dependencyBindingOffset =
      static_cast<uint32_t>(passDependencyBufferBindingResourceIndices_.size());
  const uint32_t preDispatchBindingOffset =
      static_cast<uint32_t>(preDispatchDependencyBindingOffsets_.size());
  const uint32_t drawBindingOffset =
      static_cast<uint32_t>(drawVertexBindingResourceIndices_.size());
  const RenderGraphPassId passId{.value = passIndex};

  const size_t dependencyCount = ownedPayload.dependencyBuffers.size();
  if (dependencyCount > UINT32_MAX) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addPassRecord: dependency buffer count "
        "exceeds uint32_t");
  }
  if (dependencyCount > kMaxDependencyBuffers) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addPassRecord: dependency buffer count "
        "exceeds kMaxDependencyBuffers");
  }

  const size_t preDispatchCount = ownedPayload.preDispatches.size();
  if (preDispatchCount > UINT32_MAX) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addPassRecord: pre-dispatch count exceeds "
        "uint32_t");
  }
  for (const ComputeDispatchItem &dispatch : ownedPayload.preDispatches) {
    if (dispatch.dependencyBuffers.size() > UINT32_MAX) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addPassRecord: pre-dispatch dependency "
          "buffer count exceeds uint32_t");
    }
    if (dispatch.dependencyBuffers.size() > kMaxDependencyBuffers) {
      return Result<RenderGraphPassId, std::string>::makeError(
          "RenderGraphBuilder::addPassRecord: pre-dispatch dependency "
          "buffer count exceeds kMaxDependencyBuffers");
    }
  }

  const size_t drawCount = ownedPayload.draws.size();
  if (drawCount > UINT32_MAX) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addPassRecord: draw count exceeds uint32_t");
  }

  ownedPassPayloads_.push_back(std::move(ownedPayload));
  OwnedPassPayload &storedPayload = ownedPassPayloads_.back();
  pass.preDispatches = std::span<const ComputeDispatchItem>(
      storedPayload.preDispatches.data(), storedPayload.preDispatches.size());
  pass.dependencyBuffers =
      std::span<const BufferHandle>(storedPayload.dependencyBuffers.data(),
                                    storedPayload.dependencyBuffers.size());
  pass.draws = std::span<const DrawItem>(storedPayload.draws.data(),
                                         storedPayload.draws.size());
  pass.debugLabel = std::string_view(storedPayload.debugLabel.data(),
                                     storedPayload.debugLabel.size());

  passes_.push_back(pass);
  std::pmr::string resolvedName(memory_);
  const std::string_view selectedName =
      !debugName.empty() ? debugName : pass.debugLabel;
  resolvedName.assign(selectedName.data(), selectedName.size());
  passDebugNames_.push_back(std::move(resolvedName));
  passColorTextureBindings_.push_back(UINT32_MAX);
  passDepthTextureBindings_.push_back(UINT32_MAX);
  passDependencyBufferBindingOffsets_.push_back(dependencyBindingOffset);
  passDependencyBufferBindingCounts_.push_back(
      static_cast<uint32_t>(dependencyCount));
  for (size_t i = 0; i < dependencyCount; ++i) {
    passDependencyBufferBindingResourceIndices_.push_back(UINT32_MAX);
  }

  passPreDispatchBindingOffsets_.push_back(preDispatchBindingOffset);
  passPreDispatchBindingCounts_.push_back(
      static_cast<uint32_t>(preDispatchCount));
  for (const ComputeDispatchItem &dispatch : storedPayload.preDispatches) {
    const uint32_t dispatchDependencyOffset = static_cast<uint32_t>(
        preDispatchDependencyBindingResourceIndices_.size());
    preDispatchDependencyBindingOffsets_.push_back(dispatchDependencyOffset);
    preDispatchDependencyBindingCounts_.push_back(
        static_cast<uint32_t>(dispatch.dependencyBuffers.size()));
    for (size_t i = 0; i < dispatch.dependencyBuffers.size(); ++i) {
      preDispatchDependencyBindingResourceIndices_.push_back(UINT32_MAX);
    }
  }

  passDrawBindingOffsets_.push_back(drawBindingOffset);
  passDrawBindingCounts_.push_back(static_cast<uint32_t>(drawCount));
  for (size_t i = 0; i < drawCount; ++i) {
    drawVertexBindingResourceIndices_.push_back(UINT32_MAX);
    drawIndexBindingResourceIndices_.push_back(UINT32_MAX);
    drawIndirectBindingResourceIndices_.push_back(UINT32_MAX);
    drawIndirectCountBindingResourceIndices_.push_back(UINT32_MAX);
  }
  return Result<RenderGraphPassId, std::string>::makeResult(passId);
}

Result<bool, std::string>
RenderGraphBuilder::bindPassColorTexture(RenderGraphPassId pass,
                                         RenderGraphTextureId texture) {
  if (!isValid(pass) || !isValid(texture)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassColorTexture: id is invalid");
  }
  if (!isValidPassIndex(pass.value) || !isValidTextureIndex(texture.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassColorTexture: id is out of range");
  }
  if (pass.value >= passColorTextureBindings_.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassColorTexture: pass binding table is out "
        "of sync");
  }

  passColorTextureBindings_[pass.value] = texture.value;

  const RenderPass &targetPass = passes_[pass.value];
  const RenderGraphAccessMode mode =
      attachmentAccessMode(targetPass.color.loadOp);
  if (mode == RenderGraphAccessMode::None) {
    return Result<bool, std::string>::makeResult(true);
  }
  return addTextureAccess(pass, texture, mode);
}

Result<bool, std::string>
RenderGraphBuilder::bindPassDepthTexture(RenderGraphPassId pass,
                                         RenderGraphTextureId texture) {
  if (!isValid(pass) || !isValid(texture)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDepthTexture: id is invalid");
  }
  if (!isValidPassIndex(pass.value) || !isValidTextureIndex(texture.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDepthTexture: id is out of range");
  }
  if (pass.value >= passDepthTextureBindings_.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDepthTexture: pass binding table is out "
        "of sync");
  }

  passDepthTextureBindings_[pass.value] = texture.value;

  const RenderPass &targetPass = passes_[pass.value];
  const RenderGraphAccessMode mode =
      attachmentAccessMode(targetPass.depth.loadOp);
  if (mode == RenderGraphAccessMode::None) {
    return Result<bool, std::string>::makeResult(true);
  }
  return addTextureAccess(pass, texture, mode);
}

Result<bool, std::string> RenderGraphBuilder::bindPassDependencyBuffer(
    RenderGraphPassId pass, uint32_t dependencyIndex,
    RenderGraphBufferId buffer, RenderGraphAccessMode mode) {
  if (!isValid(pass) || !isValid(buffer)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyBuffer: id is invalid");
  }
  if (!hasAccessFlag(mode, RenderGraphAccessMode::Read) &&
      !hasAccessFlag(mode, RenderGraphAccessMode::Write)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyBuffer: access mode must "
        "contain read or write");
  }
  if (!isValidPassIndex(pass.value) || !isValidBufferIndex(buffer.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyBuffer: id is out of range");
  }
  if (pass.value >= passDependencyBufferBindingOffsets_.size() ||
      pass.value >= passDependencyBufferBindingCounts_.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyBuffer: pass binding table is "
        "out of sync");
  }

  const uint32_t count = passDependencyBufferBindingCounts_[pass.value];
  if (dependencyIndex >= kMaxDependencyBuffers) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyBuffer: dependency index "
        "exceeds kMaxDependencyBuffers");
  }
  if (dependencyIndex >= count) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyBuffer: dependency index is "
        "out of range");
  }

  const uint32_t offset = passDependencyBufferBindingOffsets_[pass.value];
  if (offset > passDependencyBufferBindingResourceIndices_.size() ||
      count > passDependencyBufferBindingResourceIndices_.size() - offset) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPassDependencyBuffer: pass dependency "
        "binding range is invalid");
  }

  passDependencyBufferBindingResourceIndices_[offset + dependencyIndex] =
      buffer.value;
  return addBufferAccess(pass, buffer, mode);
}

Result<bool, std::string> RenderGraphBuilder::bindPreDispatchDependencyBuffer(
    RenderGraphPassId pass, uint32_t preDispatchIndex, uint32_t dependencyIndex,
    RenderGraphBufferId buffer, RenderGraphAccessMode mode) {
  if (!isValid(pass) || !isValid(buffer)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: id is invalid");
  }
  if (!hasAccessFlag(mode, RenderGraphAccessMode::Read) &&
      !hasAccessFlag(mode, RenderGraphAccessMode::Write)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: access mode "
        "must contain read or write");
  }
  if (!isValidPassIndex(pass.value) || !isValidBufferIndex(buffer.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: id is out of "
        "range");
  }
  if (pass.value >= passPreDispatchBindingOffsets_.size() ||
      pass.value >= passPreDispatchBindingCounts_.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: pass binding "
        "table is out of sync");
  }

  const uint32_t dispatchCount = passPreDispatchBindingCounts_[pass.value];
  if (preDispatchIndex >= dispatchCount) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: pre-dispatch "
        "index is out of range");
  }

  const uint32_t dispatchOffset = passPreDispatchBindingOffsets_[pass.value];
  if (dispatchOffset > preDispatchDependencyBindingOffsets_.size() ||
      dispatchCount >
          preDispatchDependencyBindingOffsets_.size() - dispatchOffset ||
      dispatchCount >
          preDispatchDependencyBindingCounts_.size() - dispatchOffset) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: pre-dispatch "
        "binding range is invalid");
  }

  const uint32_t globalDispatchIndex = dispatchOffset + preDispatchIndex;
  const uint32_t dependencyOffset =
      preDispatchDependencyBindingOffsets_[globalDispatchIndex];
  const uint32_t dependencyCount =
      preDispatchDependencyBindingCounts_[globalDispatchIndex];
  if (dependencyIndex >= kMaxDependencyBuffers) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: dependency "
        "index exceeds kMaxDependencyBuffers");
  }
  if (dependencyIndex >= dependencyCount) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: dependency "
        "index is out of range");
  }
  if (dependencyOffset > preDispatchDependencyBindingResourceIndices_.size() ||
      dependencyCount > preDispatchDependencyBindingResourceIndices_.size() -
                            dependencyOffset) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: dependency "
        "binding range is invalid");
  }

  preDispatchDependencyBindingResourceIndices_[dependencyOffset +
                                               dependencyIndex] = buffer.value;
  return addBufferAccess(pass, buffer, mode);
}

Result<bool, std::string> RenderGraphBuilder::bindDrawBuffer(
    RenderGraphPassId pass, uint32_t drawIndex,
    RenderGraphCompileResult::DrawBufferBindingTarget target,
    RenderGraphBufferId buffer, RenderGraphAccessMode mode) {
  if (!isValid(pass) || !isValid(buffer)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindDrawBuffer: id is invalid");
  }
  if (!hasAccessFlag(mode, RenderGraphAccessMode::Read) &&
      !hasAccessFlag(mode, RenderGraphAccessMode::Write)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindDrawBuffer: access mode must contain read "
        "or write");
  }
  if (!isValidPassIndex(pass.value) || !isValidBufferIndex(buffer.value)) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindDrawBuffer: id is out of range");
  }
  if (pass.value >= passDrawBindingOffsets_.size() ||
      pass.value >= passDrawBindingCounts_.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindDrawBuffer: pass binding table is out of "
        "sync");
  }

  const uint32_t drawCount = passDrawBindingCounts_[pass.value];
  if (drawIndex >= drawCount) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindDrawBuffer: draw index is out of range");
  }
  const uint32_t drawOffset = passDrawBindingOffsets_[pass.value];
  if (drawOffset > drawVertexBindingResourceIndices_.size() ||
      drawCount > drawVertexBindingResourceIndices_.size() - drawOffset ||
      drawCount > drawIndexBindingResourceIndices_.size() - drawOffset ||
      drawCount > drawIndirectBindingResourceIndices_.size() - drawOffset ||
      drawCount >
          drawIndirectCountBindingResourceIndices_.size() - drawOffset) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindDrawBuffer: draw binding range is invalid");
  }

  uint32_t *targetTableEntry = nullptr;
  switch (target) {
  case RenderGraphCompileResult::DrawBufferBindingTarget::Vertex:
    targetTableEntry =
        &drawVertexBindingResourceIndices_[drawOffset + drawIndex];
    break;
  case RenderGraphCompileResult::DrawBufferBindingTarget::Index:
    targetTableEntry =
        &drawIndexBindingResourceIndices_[drawOffset + drawIndex];
    break;
  case RenderGraphCompileResult::DrawBufferBindingTarget::Indirect:
    targetTableEntry =
        &drawIndirectBindingResourceIndices_[drawOffset + drawIndex];
    break;
  case RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount:
    targetTableEntry =
        &drawIndirectCountBindingResourceIndices_[drawOffset + drawIndex];
    break;
  default:
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindDrawBuffer: unsupported binding target");
  }

  *targetTableEntry = buffer.value;
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

Result<bool, std::string> RenderGraphBuilder::compileStageC0ValidateInputs(
    RenderGraphRuntime &runtime, RenderGraphCompileResult &compiled,
    RenderGraphBuilder::CompileWorkState &work) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  compiled.textureHandlesByResource.resize(textures_.size(), TextureHandle{});
  compiled.bufferHandlesByResource.resize(buffers_.size(), BufferHandle{});

  struct IndexedValidationError {
    bool hasError = false;
    uint32_t index = UINT32_MAX;
    std::string message{};
  };

  struct ResourceValidationCounts {
    uint32_t imported = 0u;
    uint32_t transient = 0u;
  };

  if (!textures_.empty()) {
    const uint32_t workerCount = std::max(1u, runtime.workerCount());
    std::vector<IndexedValidationError> textureErrors(workerCount);
    std::vector<ResourceValidationCounts> textureCounts(workerCount);
    const auto validateTextureRange = [&](uint32_t workerIndex,
                                          RenderGraphContiguousRange range) {
      ResourceValidationCounts counts{};
      for (uint32_t textureIndex = range.offset;
           textureIndex < range.offset + range.count; ++textureIndex) {
        const TextureResource &texture = textures_[textureIndex];
        if (texture.imported) {
          if (!nuri::isValid(texture.importedHandle)) {
            textureErrors[workerIndex] = IndexedValidationError{
                .hasError = true,
                .index = textureIndex,
                .message = "RenderGraphBuilder::compile: imported texture "
                           "handle is invalid",
            };
            return;
          }
          compiled.textureHandlesByResource[textureIndex] =
              texture.importedHandle;
          ++counts.imported;
          continue;
        }

        if (!isValidTransientTextureDesc(texture.transientDesc)) {
          textureErrors[workerIndex] = IndexedValidationError{
              .hasError = true,
              .index = textureIndex,
              .message = "RenderGraphBuilder::compile: transient texture "
                         "descriptor is invalid",
          };
          return;
        }
        ++counts.transient;
      }
      textureCounts[workerIndex] = counts;
    };

    bool ranTextureValidationInParallel = false;
    if (runtime.parallelCompileEnabled() && textures_.size() > 1u) {
      const std::vector<RenderGraphContiguousRange> stdRanges =
          makeValidationRanges(static_cast<uint32_t>(textures_.size()),
                               workerCount);
      if (stdRanges.size() > 1u) {
        std::pmr::vector<RenderGraphContiguousRange> ranges(memory_);
        ranges.assign(stdRanges.begin(), stdRanges.end());
        runtime.runRanges(std::span<const RenderGraphContiguousRange>(
                              ranges.data(), ranges.size()),
                          validateTextureRange);
        ranTextureValidationInParallel = true;
      } else if (!stdRanges.empty()) {
        validateTextureRange(0u, stdRanges[0u]);
      }
    } else {
      validateTextureRange(
          0u,
          RenderGraphContiguousRange{
              .offset = 0u, .count = static_cast<uint32_t>(textures_.size())});
    }
    work.usedParallelValidation =
        work.usedParallelValidation || ranTextureValidationInParallel;
    const IndexedValidationError *textureError = nullptr;
    for (const IndexedValidationError &error : textureErrors) {
      if (!error.hasError) {
        continue;
      }
      if (textureError == nullptr || error.index < textureError->index) {
        textureError = &error;
      }
    }
    if (textureError != nullptr) {
      return Result<bool, std::string>::makeError(textureError->message);
    }
    for (const ResourceValidationCounts &counts : textureCounts) {
      compiled.resourceStats.importedTextures += counts.imported;
      compiled.resourceStats.transientTextures += counts.transient;
    }
  }

  if (!buffers_.empty()) {
    const uint32_t workerCount = std::max(1u, runtime.workerCount());
    std::vector<IndexedValidationError> bufferErrors(workerCount);
    std::vector<ResourceValidationCounts> bufferCounts(workerCount);
    const auto validateBufferRange = [&](uint32_t workerIndex,
                                         RenderGraphContiguousRange range) {
      ResourceValidationCounts counts{};
      for (uint32_t bufferIndex = range.offset;
           bufferIndex < range.offset + range.count; ++bufferIndex) {
        const BufferResource &buffer = buffers_[bufferIndex];
        if (buffer.imported) {
          if (!nuri::isValid(buffer.importedHandle)) {
            bufferErrors[workerIndex] = IndexedValidationError{
                .hasError = true,
                .index = bufferIndex,
                .message = "RenderGraphBuilder::compile: imported buffer "
                           "handle is invalid",
            };
            return;
          }
          compiled.bufferHandlesByResource[bufferIndex] = buffer.importedHandle;
          ++counts.imported;
          continue;
        }

        if (!isValidTransientBufferDesc(buffer.transientDesc)) {
          bufferErrors[workerIndex] = IndexedValidationError{
              .hasError = true,
              .index = bufferIndex,
              .message = "RenderGraphBuilder::compile: transient buffer "
                         "descriptor is invalid",
          };
          return;
        }
        ++counts.transient;
      }
      bufferCounts[workerIndex] = counts;
    };

    bool ranBufferValidationInParallel = false;
    if (runtime.parallelCompileEnabled() && buffers_.size() > 1u) {
      const std::vector<RenderGraphContiguousRange> stdRanges =
          makeValidationRanges(static_cast<uint32_t>(buffers_.size()),
                               workerCount);
      if (stdRanges.size() > 1u) {
        std::pmr::vector<RenderGraphContiguousRange> ranges(memory_);
        ranges.assign(stdRanges.begin(), stdRanges.end());
        runtime.runRanges(std::span<const RenderGraphContiguousRange>(
                              ranges.data(), ranges.size()),
                          validateBufferRange);
        ranBufferValidationInParallel = true;
      } else if (!stdRanges.empty()) {
        validateBufferRange(0u, stdRanges[0u]);
      }
    } else {
      validateBufferRange(
          0u,
          RenderGraphContiguousRange{
              .offset = 0u, .count = static_cast<uint32_t>(buffers_.size())});
    }
    work.usedParallelValidation =
        work.usedParallelValidation || ranBufferValidationInParallel;
    const IndexedValidationError *bufferError = nullptr;
    for (const IndexedValidationError &error : bufferErrors) {
      if (!error.hasError) {
        continue;
      }
      if (bufferError == nullptr || error.index < bufferError->index) {
        bufferError = &error;
      }
    }
    if (bufferError != nullptr) {
      return Result<bool, std::string>::makeError(bufferError->message);
    }
    for (const ResourceValidationCounts &counts : bufferCounts) {
      compiled.resourceStats.importedBuffers += counts.imported;
      compiled.resourceStats.transientBuffers += counts.transient;
    }
  }

  if (!passResourceAccesses_.empty()) {
    const uint32_t workerCount = std::max(1u, runtime.workerCount());
    std::vector<IndexedValidationError> accessErrors(workerCount);
    const auto validateAccessRange = [&](uint32_t workerIndex,
                                         RenderGraphContiguousRange range) {
      for (uint32_t accessIndex = range.offset;
           accessIndex < range.offset + range.count; ++accessIndex) {
        const PassResourceAccess &access = passResourceAccesses_[accessIndex];
        if (!isValidPassIndex(access.passIndex)) {
          accessErrors[workerIndex] = IndexedValidationError{
              .hasError = true,
              .index = accessIndex,
              .message = "RenderGraphBuilder::compile: resource access "
                         "references out-of-range pass",
          };
          return;
        }
        if (access.resourceKind == AccessResourceKind::Texture &&
            !isValidTextureIndex(access.resourceIndex)) {
          accessErrors[workerIndex] = IndexedValidationError{
              .hasError = true,
              .index = accessIndex,
              .message = "RenderGraphBuilder::compile: texture access "
                         "references out-of-range resource",
          };
          return;
        }
        if (access.resourceKind == AccessResourceKind::Buffer &&
            !isValidBufferIndex(access.resourceIndex)) {
          accessErrors[workerIndex] = IndexedValidationError{
              .hasError = true,
              .index = accessIndex,
              .message = "RenderGraphBuilder::compile: buffer access "
                         "references out-of-range resource",
          };
          return;
        }
        if (!hasAccessFlag(access.mode, RenderGraphAccessMode::Read) &&
            !hasAccessFlag(access.mode, RenderGraphAccessMode::Write)) {
          accessErrors[workerIndex] = IndexedValidationError{
              .hasError = true,
              .index = accessIndex,
              .message = "RenderGraphBuilder::compile: resource access "
                         "mode is invalid",
          };
          return;
        }
      }
    };

    bool ranAccessValidationInParallel = false;
    if (runtime.parallelCompileEnabled() && passResourceAccesses_.size() > 1u) {
      const std::vector<RenderGraphContiguousRange> stdRanges =
          makeValidationRanges(
              static_cast<uint32_t>(passResourceAccesses_.size()), workerCount);
      if (stdRanges.size() > 1u) {
        std::pmr::vector<RenderGraphContiguousRange> ranges(memory_);
        ranges.assign(stdRanges.begin(), stdRanges.end());
        runtime.runRanges(std::span<const RenderGraphContiguousRange>(
                              ranges.data(), ranges.size()),
                          validateAccessRange);
        ranAccessValidationInParallel = true;
      } else if (!stdRanges.empty()) {
        validateAccessRange(0u, stdRanges[0u]);
      }
    } else {
      validateAccessRange(
          0u, RenderGraphContiguousRange{.offset = 0u,
                                         .count = static_cast<uint32_t>(
                                             passResourceAccesses_.size())});
    }
    work.usedParallelValidation =
        work.usedParallelValidation || ranAccessValidationInParallel;
    const IndexedValidationError *accessError = nullptr;
    for (const IndexedValidationError &error : accessErrors) {
      if (!error.hasError) {
        continue;
      }
      if (accessError == nullptr || error.index < accessError->index) {
        accessError = &error;
      }
    }
    if (accessError != nullptr) {
      return Result<bool, std::string>::makeError(accessError->message);
    }
  }

  if (passes_.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (passColorTextureBindings_.size() != passes_.size() ||
      passDepthTextureBindings_.size() != passes_.size() ||
      passDependencyBufferBindingOffsets_.size() != passes_.size() ||
      passDependencyBufferBindingCounts_.size() != passes_.size() ||
      passPreDispatchBindingOffsets_.size() != passes_.size() ||
      passPreDispatchBindingCounts_.size() != passes_.size() ||
      passDrawBindingOffsets_.size() != passes_.size() ||
      passDrawBindingCounts_.size() != passes_.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::compile: pass texture binding tables are out of "
        "sync");
  }
  if (drawVertexBindingResourceIndices_.size() !=
          drawIndexBindingResourceIndices_.size() ||
      drawVertexBindingResourceIndices_.size() !=
          drawIndirectBindingResourceIndices_.size() ||
      drawVertexBindingResourceIndices_.size() !=
          drawIndirectCountBindingResourceIndices_.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::compile: draw buffer binding tables are out of "
        "sync");
  }
  if (preDispatchDependencyBindingOffsets_.size() !=
      preDispatchDependencyBindingCounts_.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::compile: pre-dispatch dependency binding tables "
        "are out of sync");
  }
  work.passCount = static_cast<uint32_t>(passes_.size());
  work.activePassCount = work.passCount;
  compiled.declaredPassCount = work.passCount;
  return Result<bool, std::string>::makeResult(true);
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
      for (size_t i = groupBegin; i < groupEnd; ++i) {
        if (work.compiledAccesses[i].inferred) {
          inferredMode = inferredMode | work.compiledAccesses[i].mode;
          continue;
        }
        explicitMode = explicitMode | work.compiledAccesses[i].mode;
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
    NURI_ASSERT(!ready.empty(),
                "RenderGraphBuilder::compile: ready set is unexpectedly empty");
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

  struct IndexedResolveError {
    bool hasError = false;
    uint32_t orderedPassIndex = UINT32_MAX;
    std::string message{};
  };
  struct PassResolvePlan {
    uint32_t passIndex = UINT32_MAX;
    uint32_t colorTextureIndex = UINT32_MAX;
    uint32_t depthTextureIndex = UINT32_MAX;
    uint32_t dependencyCount = 0u;
    uint32_t dependencyBindingOffset = 0u;
    uint32_t resolvedDependencyOffset = 0u;
    uint32_t preDispatchCount = 0u;
    uint32_t preDispatchBindingOffset = 0u;
    uint32_t preDispatchOutputOffset = 0u;
    uint32_t preDispatchDependencyOffset = 0u;
    uint32_t preDispatchDependencyCount = 0u;
    uint32_t drawCount = 0u;
    uint32_t drawBindingOffset = 0u;
    uint32_t drawOutputOffset = 0u;
    uint32_t unresolvedTextureOffset = 0u;
    uint32_t unresolvedTextureCount = 0u;
    uint32_t unresolvedDependencyOffset = 0u;
    uint32_t unresolvedDependencyCount = 0u;
    uint32_t unresolvedPreDispatchDependencyOffset = 0u;
    uint32_t unresolvedPreDispatchDependencyCount = 0u;
    uint32_t unresolvedDrawOffset = 0u;
    uint32_t unresolvedDrawCount = 0u;
  };
  const uint32_t workerCount = std::max(1u, runtime.workerCount());
  std::vector<IndexedResolveError> resolveErrors(workerCount);
  std::pmr::vector<PassResolvePlan> passPlans(memory_);
  passPlans.resize(work.order.size());
  const auto validatePassRange = [&](uint32_t workerIndex,
                                     RenderGraphContiguousRange range) {
    for (uint32_t orderedPassIndex = range.offset;
         orderedPassIndex < range.offset + range.count; ++orderedPassIndex) {
      const uint32_t passIndex = work.order[orderedPassIndex];
      const RenderPass &sourcePass = passes_[passIndex];
      PassResolvePlan plan{};
      plan.passIndex = passIndex;
      plan.colorTextureIndex = passColorTextureBindings_[passIndex];
      plan.depthTextureIndex = passDepthTextureBindings_[passIndex];
      const uint32_t dependencyCount =
          passDependencyBufferBindingCounts_[passIndex];
      if (dependencyCount > kMaxDependencyBuffers) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message = "RenderGraphBuilder::compile: pass dependency buffer "
                       "count exceeds kMaxDependencyBuffers",
        };
        return;
      }
      const uint32_t dependencyOffset =
          passDependencyBufferBindingOffsets_[passIndex];
      if (sourcePass.dependencyBuffers.size() != dependencyCount) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message = "RenderGraphBuilder::compile: dependency buffer binding "
                       "count does not match pass dependency buffer count",
        };
        return;
      }
      if (dependencyOffset >
              passDependencyBufferBindingResourceIndices_.size() ||
          dependencyCount > passDependencyBufferBindingResourceIndices_.size() -
                                dependencyOffset) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message = "RenderGraphBuilder::compile: dependency buffer binding "
                       "range is invalid",
        };
        return;
      }
      plan.dependencyCount = dependencyCount;
      plan.dependencyBindingOffset = dependencyOffset;
      const uint32_t preDispatchCount =
          passPreDispatchBindingCounts_[passIndex];
      const uint32_t preDispatchOffset =
          passPreDispatchBindingOffsets_[passIndex];
      if (sourcePass.preDispatches.size() != preDispatchCount) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message =
                "RenderGraphBuilder::compile: pre-dispatch binding count "
                "does not match pass pre-dispatch count",
        };
        return;
      }
      if (preDispatchOffset > preDispatchDependencyBindingOffsets_.size() ||
          preDispatchCount >
              preDispatchDependencyBindingOffsets_.size() - preDispatchOffset ||
          preDispatchCount >
              preDispatchDependencyBindingCounts_.size() - preDispatchOffset) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message =
                "RenderGraphBuilder::compile: pre-dispatch binding range "
                "is invalid",
        };
        return;
      }
      plan.preDispatchCount = preDispatchCount;
      plan.preDispatchBindingOffset = preDispatchOffset;
      for (uint32_t i = 0; i < preDispatchCount; ++i) {
        const uint32_t depOffset =
            preDispatchDependencyBindingOffsets_[preDispatchOffset + i];
        const uint32_t depCount =
            preDispatchDependencyBindingCounts_[preDispatchOffset + i];
        if (depCount > kMaxDependencyBuffers) {
          resolveErrors[workerIndex] = IndexedResolveError{
              .hasError = true,
              .orderedPassIndex = orderedPassIndex,
              .message = "RenderGraphBuilder::compile: pre-dispatch dependency "
                         "buffer count exceeds kMaxDependencyBuffers",
          };
          return;
        }
        if (depOffset > preDispatchDependencyBindingResourceIndices_.size() ||
            depCount > preDispatchDependencyBindingResourceIndices_.size() -
                           depOffset) {
          resolveErrors[workerIndex] = IndexedResolveError{
              .hasError = true,
              .orderedPassIndex = orderedPassIndex,
              .message = "RenderGraphBuilder::compile: pre-dispatch dependency "
                         "binding range is invalid",
          };
          return;
        }
        plan.preDispatchDependencyCount += depCount;
      }

      const uint32_t drawCount = passDrawBindingCounts_[passIndex];
      const uint32_t drawOffset = passDrawBindingOffsets_[passIndex];
      if (sourcePass.draws.size() != drawCount) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message = "RenderGraphBuilder::compile: draw binding count does "
                       "not match pass draw count",
        };
        return;
      }
      if (drawOffset > drawVertexBindingResourceIndices_.size() ||
          drawCount > drawVertexBindingResourceIndices_.size() - drawOffset ||
          drawCount > drawIndexBindingResourceIndices_.size() - drawOffset ||
          drawCount > drawIndirectBindingResourceIndices_.size() - drawOffset ||
          drawCount >
              drawIndirectCountBindingResourceIndices_.size() - drawOffset) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message = "RenderGraphBuilder::compile: draw binding range is "
                       "invalid",
        };
        return;
      }
      if (nuri::isValid(sourcePass.colorTexture) &&
          passColorTextureBindings_[passIndex] == UINT32_MAX) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message = "RenderGraphBuilder::compile: pass requires explicit "
                       "color texture binding",
        };
        return;
      }
      if (nuri::isValid(sourcePass.depthTexture) &&
          passDepthTextureBindings_[passIndex] == UINT32_MAX) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message = "RenderGraphBuilder::compile: pass requires explicit "
                       "depth texture binding",
        };
        return;
      }
      if (plan.colorTextureIndex != UINT32_MAX &&
          !isValidTextureIndex(plan.colorTextureIndex)) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message = "RenderGraphBuilder::compile: color texture binding "
                       "references out-of-range texture",
        };
        return;
      }
      if (plan.depthTextureIndex != UINT32_MAX &&
          !isValidTextureIndex(plan.depthTextureIndex)) {
        resolveErrors[workerIndex] = IndexedResolveError{
            .hasError = true,
            .orderedPassIndex = orderedPassIndex,
            .message = "RenderGraphBuilder::compile: depth texture binding "
                       "references out-of-range texture",
        };
        return;
      }
      if (plan.colorTextureIndex != UINT32_MAX &&
          !textures_[plan.colorTextureIndex].imported) {
        ++plan.unresolvedTextureCount;
      }
      if (plan.depthTextureIndex != UINT32_MAX &&
          !textures_[plan.depthTextureIndex].imported) {
        ++plan.unresolvedTextureCount;
      }
      for (uint32_t depIndex = 0; depIndex < dependencyCount; ++depIndex) {
        const BufferHandle sourceHandle =
            sourcePass.dependencyBuffers[depIndex];
        const uint32_t resourceIndex =
            passDependencyBufferBindingResourceIndices_[dependencyOffset +
                                                        depIndex];
        if (nuri::isValid(sourceHandle) && resourceIndex == UINT32_MAX) {
          resolveErrors[workerIndex] = IndexedResolveError{
              .hasError = true,
              .orderedPassIndex = orderedPassIndex,
              .message = "RenderGraphBuilder::compile: pass requires explicit "
                         "dependency buffer binding",
          };
          return;
        }
        if (resourceIndex == UINT32_MAX) {
          continue;
        }
        if (!isValidBufferIndex(resourceIndex)) {
          resolveErrors[workerIndex] = IndexedResolveError{
              .hasError = true,
              .orderedPassIndex = orderedPassIndex,
              .message =
                  "RenderGraphBuilder::compile: dependency buffer binding "
                  "references out-of-range resource",
          };
          return;
        }
        if (!buffers_[resourceIndex].imported) {
          ++plan.unresolvedDependencyCount;
        }
      }
      for (uint32_t dispatchIndex = 0; dispatchIndex < preDispatchCount;
           ++dispatchIndex) {
        const ComputeDispatchItem &dispatch =
            sourcePass.preDispatches[dispatchIndex];
        const uint32_t depOffset =
            preDispatchDependencyBindingOffsets_[preDispatchOffset +
                                                 dispatchIndex];
        const uint32_t depCount =
            preDispatchDependencyBindingCounts_[preDispatchOffset +
                                                dispatchIndex];
        for (uint32_t depIndex = 0; depIndex < depCount; ++depIndex) {
          const BufferHandle sourceHandle =
              dispatch.dependencyBuffers[depIndex];
          const uint32_t resourceIndex =
              preDispatchDependencyBindingResourceIndices_[depOffset +
                                                           depIndex];
          if (nuri::isValid(sourceHandle) && resourceIndex == UINT32_MAX) {
            resolveErrors[workerIndex] = IndexedResolveError{
                .hasError = true,
                .orderedPassIndex = orderedPassIndex,
                .message =
                    "RenderGraphBuilder::compile: pass requires explicit "
                    "pre-dispatch dependency buffer binding",
            };
            return;
          }
          if (resourceIndex == UINT32_MAX) {
            continue;
          }
          if (!isValidBufferIndex(resourceIndex)) {
            resolveErrors[workerIndex] = IndexedResolveError{
                .hasError = true,
                .orderedPassIndex = orderedPassIndex,
                .message =
                    "RenderGraphBuilder::compile: pre-dispatch dependency "
                    "binding references out-of-range resource",
            };
            return;
          }
          if (!buffers_[resourceIndex].imported) {
            ++plan.unresolvedPreDispatchDependencyCount;
          }
        }
      }
      for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex) {
        const DrawItem &draw = sourcePass.draws[drawIndex];
        const uint32_t globalDrawIndex = drawOffset + drawIndex;
        const auto validateDrawBinding =
            [&](BufferHandle sourceHandle, uint32_t resourceIndex,
                std::string_view missingBindingMessage,
                std::string_view invalidBindingMessage) {
              if (nuri::isValid(sourceHandle) && resourceIndex == UINT32_MAX) {
                resolveErrors[workerIndex] = IndexedResolveError{
                    .hasError = true,
                    .orderedPassIndex = orderedPassIndex,
                    .message = std::string(missingBindingMessage),
                };
                return false;
              }
              if (resourceIndex == UINT32_MAX) {
                return true;
              }
              if (!isValidBufferIndex(resourceIndex)) {
                resolveErrors[workerIndex] = IndexedResolveError{
                    .hasError = true,
                    .orderedPassIndex = orderedPassIndex,
                    .message = std::string(invalidBindingMessage),
                };
                return false;
              }
              if (!buffers_[resourceIndex].imported) {
                ++plan.unresolvedDrawCount;
              }
              return true;
            };
        if (!validateDrawBinding(
                draw.vertexBuffer,
                drawVertexBindingResourceIndices_[globalDrawIndex],
                "RenderGraphBuilder::compile: pass requires explicit draw "
                "vertex buffer binding",
                "RenderGraphBuilder::compile: draw buffer binding "
                "references out-of-range resource")) {
          return;
        }
        if (!validateDrawBinding(
                draw.indexBuffer,
                drawIndexBindingResourceIndices_[globalDrawIndex],
                "RenderGraphBuilder::compile: pass requires explicit draw "
                "index buffer binding",
                "RenderGraphBuilder::compile: draw buffer binding "
                "references out-of-range resource")) {
          return;
        }
        if (!validateDrawBinding(
                draw.indirectBuffer,
                drawIndirectBindingResourceIndices_[globalDrawIndex],
                "RenderGraphBuilder::compile: pass requires explicit draw "
                "indirect buffer binding",
                "RenderGraphBuilder::compile: draw buffer binding "
                "references out-of-range resource")) {
          return;
        }
        if (!validateDrawBinding(
                draw.indirectCountBuffer,
                drawIndirectCountBindingResourceIndices_[globalDrawIndex],
                "RenderGraphBuilder::compile: pass requires explicit draw "
                "indirect-count buffer binding",
                "RenderGraphBuilder::compile: draw buffer binding "
                "references out-of-range resource")) {
          return;
        }
      }
      plan.drawCount = drawCount;
      plan.drawBindingOffset = drawOffset;
      passPlans[orderedPassIndex] = plan;
    }
  };

  bool usedParallelPassResolution = false;
  if (runtime.parallelCompileEnabled() && work.order.size() > 1u) {
    const std::vector<RenderGraphContiguousRange> stdRanges = makePayloadRanges(
        static_cast<uint32_t>(work.order.size()), workerCount);
    if (stdRanges.size() > 1u) {
      std::pmr::vector<RenderGraphContiguousRange> ranges(memory_);
      ranges.assign(stdRanges.begin(), stdRanges.end());
      runtime.runRanges(std::span<const RenderGraphContiguousRange>(
                            ranges.data(), ranges.size()),
                        validatePassRange);
      usedParallelPassResolution = true;
    } else if (!stdRanges.empty()) {
      validatePassRange(0u, stdRanges[0u]);
    }
  } else {
    validatePassRange(
        0u,
        RenderGraphContiguousRange{
            .offset = 0u, .count = static_cast<uint32_t>(work.order.size())});
  }

  compiled.usedParallelPayloadResolution = usedParallelPassResolution;
  const IndexedResolveError *resolveError = nullptr;
  for (const IndexedResolveError &error : resolveErrors) {
    if (!error.hasError) {
      continue;
    }
    if (resolveError == nullptr ||
        error.orderedPassIndex < resolveError->orderedPassIndex) {
      resolveError = &error;
    }
  }
  if (resolveError != nullptr) {
    return Result<bool, std::string>::makeError(resolveError->message);
  }

  size_t totalDependencyBufferSlots = 0u;
  size_t totalPreDispatchItems = 0u;
  size_t totalPreDispatchDependencySlots = 0u;
  size_t totalDrawItems = 0u;
  size_t totalUnresolvedTextureBindings = 0u;
  size_t totalUnresolvedDependencyBufferBindings = 0u;
  size_t totalUnresolvedPreDispatchDependencyBufferBindings = 0u;
  size_t totalUnresolvedDrawBufferBindings = 0u;
  for (uint32_t orderedPassIndex = 0u; orderedPassIndex < passPlans.size();
       ++orderedPassIndex) {
    PassResolvePlan &plan = passPlans[orderedPassIndex];
    plan.resolvedDependencyOffset =
        static_cast<uint32_t>(totalDependencyBufferSlots);
    totalDependencyBufferSlots += plan.dependencyCount;
    plan.preDispatchOutputOffset = static_cast<uint32_t>(totalPreDispatchItems);
    totalPreDispatchItems += plan.preDispatchCount;
    plan.preDispatchDependencyOffset =
        static_cast<uint32_t>(totalPreDispatchDependencySlots);
    totalPreDispatchDependencySlots += plan.preDispatchDependencyCount;
    plan.drawOutputOffset = static_cast<uint32_t>(totalDrawItems);
    totalDrawItems += plan.drawCount;
    plan.unresolvedTextureOffset =
        static_cast<uint32_t>(totalUnresolvedTextureBindings);
    totalUnresolvedTextureBindings += plan.unresolvedTextureCount;
    plan.unresolvedDependencyOffset =
        static_cast<uint32_t>(totalUnresolvedDependencyBufferBindings);
    totalUnresolvedDependencyBufferBindings += plan.unresolvedDependencyCount;
    plan.unresolvedPreDispatchDependencyOffset = static_cast<uint32_t>(
        totalUnresolvedPreDispatchDependencyBufferBindings);
    totalUnresolvedPreDispatchDependencyBufferBindings +=
        plan.unresolvedPreDispatchDependencyCount;
    plan.unresolvedDrawOffset =
        static_cast<uint32_t>(totalUnresolvedDrawBufferBindings);
    totalUnresolvedDrawBufferBindings += plan.unresolvedDrawCount;
  }

  compiled.resolvedDependencyBuffers.resize(totalDependencyBufferSlots);
  compiled.dependencyBufferRangesByPass.resize(work.order.size());
  compiled.ownedPreDispatches.resize(totalPreDispatchItems);
  compiled.preDispatchRangesByPass.resize(work.order.size());
  compiled.resolvedPreDispatchDependencyBuffers.resize(
      totalPreDispatchDependencySlots);
  compiled.preDispatchDependencyRanges.resize(totalPreDispatchItems);
  compiled.ownedDrawItems.resize(totalDrawItems);
  compiled.drawRangesByPass.resize(work.order.size());
  compiled.orderedPasses.resize(work.order.size());
  compiled.orderedPassIndices.resize(work.order.size());
  compiled.recordedGraphicsPasses.resize(work.order.size());
  compiled.passBarrierPlans.resize(work.order.size());
  compiled.unresolvedTextureBindings.resize(totalUnresolvedTextureBindings);
  compiled.unresolvedDependencyBufferBindings.resize(
      totalUnresolvedDependencyBufferBindings);
  compiled.unresolvedPreDispatchDependencyBufferBindings.resize(
      totalUnresolvedPreDispatchDependencyBufferBindings);
  compiled.unresolvedDrawBufferBindings.resize(
      totalUnresolvedDrawBufferBindings);

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
            preDispatchDependencyBindingOffsets_[globalDispatchBindingIndex];
        const uint32_t dispatchDependencyCount =
            preDispatchDependencyBindingCounts_[globalDispatchBindingIndex];
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

      compiled.drawRangesByPass[orderedPassIndex] = {
          .offset = plan.drawOutputOffset, .count = plan.drawCount};
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
                  .unresolvedDrawBufferBindings[unresolvedDrawWriteOffset++] = {
                  .orderedPassIndex = orderedPassIndex,
                  .drawIndex = drawIndex,
                  .target = target,
                  .bufferResourceIndex = resourceIndex};
            };

        resolveDrawBinding(
            drawVertexBindingResourceIndices_[globalDrawIndex],
            RenderGraphCompileResult::DrawBufferBindingTarget::Vertex,
            resolvedDraw.vertexBuffer);
        resolveDrawBinding(
            drawIndexBindingResourceIndices_[globalDrawIndex],
            RenderGraphCompileResult::DrawBufferBindingTarget::Index,
            resolvedDraw.indexBuffer);
        resolveDrawBinding(
            drawIndirectBindingResourceIndices_[globalDrawIndex],
            RenderGraphCompileResult::DrawBufferBindingTarget::Indirect,
            resolvedDraw.indirectBuffer);
        resolveDrawBinding(
            drawIndirectCountBindingResourceIndices_[globalDrawIndex],
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
      if (passIndex < compiled.passDebugNames.size()) {
        const std::pmr::string &compiledName =
            compiled.passDebugNames[passIndex];
        resolvedPass.debugLabel =
            std::string_view(compiledName.data(), compiledName.size());
      }

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

  if (usedParallelPassResolution && work.order.size() > 1u) {
    const std::vector<RenderGraphContiguousRange> stdRanges = makePayloadRanges(
        static_cast<uint32_t>(work.order.size()), workerCount);
    if (stdRanges.size() > 1u) {
      std::pmr::vector<RenderGraphContiguousRange> ranges(memory_);
      ranges.assign(stdRanges.begin(), stdRanges.end());
      runtime.runRanges(std::span<const RenderGraphContiguousRange>(
                            ranges.data(), ranges.size()),
                        fillPassRange);
    } else if (!stdRanges.empty()) {
      fillPassRange(0u, stdRanges[0u]);
    }
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
    const bool hasWrite =
        hasAccessFlag(access.mode, RenderGraphAccessMode::Write);
    if (access.resourceKind == AccessResourceKind::Texture &&
        (passColorTextureBindings_[access.passIndex] == access.resourceIndex ||
         passDepthTextureBindings_[access.passIndex] == access.resourceIndex)) {
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
          .resourceKind = access.resourceKind == AccessResourceKind::Texture
                              ? RenderGraphBarrierResourceKind::Texture
                              : RenderGraphBarrierResourceKind::Buffer,
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
                                      uint32_t resourceIndex,
                                      uint32_t rank) {
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
        : textureFirst(memory),
          textureLast(memory),
          bufferFirst(memory),
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
            analyzeAccessRange(
                std::span<uint32_t>(worker.textureFirst.data(),
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
          transientTextureLastRank[textureIndex] = std::max(
              transientTextureLastRank[textureIndex],
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

  compiled.usedParallelValidation = work.usedParallelValidation;
  compiled.usedParallelHazardAnalysis = work.usedParallelHazardAnalysis;
  compiled.usedParallelLifetimeAnalysis = usedParallelLifetimeAnalysis;
  compiled.usedParallelCompile = compiled.usedParallelValidation ||
                                 compiled.usedParallelPayloadResolution ||
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

Result<bool, std::string>
RenderGraphBuilder::compileStageC7ValidateCompiledMetadata(
    const RenderGraphCompileResult &compiled) const {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_BARRIER);
  {
    NURI_PROFILER_ZONE("RenderGraph.compile.validate_compiled_metadata",
                       NURI_PROFILER_COLOR_BARRIER);
    for (const auto &lifetime : compiled.transientTextureLifetimes) {
      if (lifetime.resourceIndex >=
          compiled.transientTextureAllocationByResource.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture allocation map is "
            "out of range");
      }
      const uint32_t allocationIndex =
          compiled.transientTextureAllocationByResource[lifetime.resourceIndex];
      if (allocationIndex == UINT32_MAX ||
          allocationIndex >= compiled.transientTexturePhysicalCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture allocation map is "
            "incomplete or invalid");
      }
    }

    for (const auto &lifetime : compiled.transientBufferLifetimes) {
      if (lifetime.resourceIndex >=
          compiled.transientBufferAllocationByResource.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer allocation map is "
            "out of range");
      }
      const uint32_t allocationIndex =
          compiled.transientBufferAllocationByResource[lifetime.resourceIndex];
      if (allocationIndex == UINT32_MAX ||
          allocationIndex >= compiled.transientBufferPhysicalCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer allocation map is "
            "incomplete or invalid");
      }
    }

    if (compiled.transientTextureAllocations.size() !=
        compiled.transientTextureLifetimes.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture allocation metadata "
          "count mismatch");
    }
    if (compiled.transientBufferAllocations.size() !=
        compiled.transientBufferLifetimes.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer allocation metadata "
          "count mismatch");
    }
    if (compiled.transientTexturePhysicalAllocations.size() !=
        compiled.transientTexturePhysicalCount) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture physical allocation "
          "metadata count mismatch");
    }
    if (compiled.transientBufferPhysicalAllocations.size() !=
        compiled.transientBufferPhysicalCount) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer physical allocation "
          "metadata count mismatch");
    }

    uint32_t previousTextureResourceIndex = UINT32_MAX;
    for (size_t i = 0; i < compiled.transientTextureAllocations.size(); ++i) {
      const auto &allocation = compiled.transientTextureAllocations[i];
      if (allocation.resourceIndex >=
          compiled.transientTextureAllocationByResource.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture allocation "
            "resource index is out of range");
      }
      if (allocation.allocationIndex >=
          compiled.transientTexturePhysicalCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture allocation slot "
            "index is out of range");
      }
      if (i > 0u && allocation.resourceIndex <= previousTextureResourceIndex) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture allocations are "
            "not strictly ordered by resource index");
      }
      previousTextureResourceIndex = allocation.resourceIndex;
      if (compiled
              .transientTextureAllocationByResource[allocation.resourceIndex] !=
          allocation.allocationIndex) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture allocation map "
            "entry mismatch");
      }
    }

    uint32_t previousBufferResourceIndex = UINT32_MAX;
    for (size_t i = 0; i < compiled.transientBufferAllocations.size(); ++i) {
      const auto &allocation = compiled.transientBufferAllocations[i];
      if (allocation.resourceIndex >=
          compiled.transientBufferAllocationByResource.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer allocation resource "
            "index is out of range");
      }
      if (allocation.allocationIndex >= compiled.transientBufferPhysicalCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer allocation slot "
            "index is out of range");
      }
      if (i > 0u && allocation.resourceIndex <= previousBufferResourceIndex) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer allocations are not "
            "strictly ordered by resource index");
      }
      previousBufferResourceIndex = allocation.resourceIndex;
      if (compiled
              .transientBufferAllocationByResource[allocation.resourceIndex] !=
          allocation.allocationIndex) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer allocation map "
            "entry mismatch");
      }
    }

    std::pmr::vector<uint8_t> seenTexturePhysicalSlots(memory_);
    seenTexturePhysicalSlots.resize(compiled.transientTexturePhysicalCount, 0u);
    for (const auto &allocation :
         compiled.transientTexturePhysicalAllocations) {
      if (allocation.allocationIndex >=
          compiled.transientTexturePhysicalCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture physical "
            "allocation slot index is out of range");
      }
      if (seenTexturePhysicalSlots[allocation.allocationIndex] != 0u) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture physical "
            "allocation slot index is duplicated");
      }
      seenTexturePhysicalSlots[allocation.allocationIndex] = 1u;

      if (allocation.representativeResourceIndex >=
          compiled.transientTextureAllocationByResource.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture physical "
            "allocation representative resource index is out of range");
      }
      if (allocation.representativeResourceIndex >= textures_.size() ||
          textures_[allocation.representativeResourceIndex].imported) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture physical "
            "allocation representative resource is invalid");
      }
      if (compiled.transientTextureAllocationByResource
              [allocation.representativeResourceIndex] !=
          allocation.allocationIndex) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture physical "
            "allocation representative map entry mismatch");
      }
    }
    for (uint32_t slot = 0; slot < compiled.transientTexturePhysicalCount;
         ++slot) {
      if (seenTexturePhysicalSlots[slot] == 0u) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient texture physical "
            "allocation metadata is missing a slot");
      }
    }

    std::pmr::vector<uint8_t> seenBufferPhysicalSlots(memory_);
    seenBufferPhysicalSlots.resize(compiled.transientBufferPhysicalCount, 0u);
    for (const auto &allocation : compiled.transientBufferPhysicalAllocations) {
      if (allocation.allocationIndex >= compiled.transientBufferPhysicalCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer physical allocation "
            "slot index is out of range");
      }
      if (seenBufferPhysicalSlots[allocation.allocationIndex] != 0u) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer physical allocation "
            "slot index is duplicated");
      }
      seenBufferPhysicalSlots[allocation.allocationIndex] = 1u;

      if (allocation.representativeResourceIndex >=
          compiled.transientBufferAllocationByResource.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer physical allocation "
            "representative resource index is out of range");
      }
      if (allocation.representativeResourceIndex >= buffers_.size() ||
          buffers_[allocation.representativeResourceIndex].imported) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer physical allocation "
            "representative resource is invalid");
      }
      if (compiled.transientBufferAllocationByResource
              [allocation.representativeResourceIndex] !=
          allocation.allocationIndex) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer physical allocation "
            "representative map entry mismatch");
      }
    }
    for (uint32_t slot = 0; slot < compiled.transientBufferPhysicalCount;
         ++slot) {
      if (seenBufferPhysicalSlots[slot] == 0u) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: transient buffer physical allocation "
            "metadata is missing a slot");
      }
    }

    if (compiled.passDebugNames.size() != compiled.declaredPassCount) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: pass debug-name metadata count "
          "mismatch");
    }
    if (compiled.orderedPasses.size() != compiled.orderedPassIndices.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: ordered pass metadata count mismatch");
    }
    if (compiled.recordedGraphicsPasses.size() !=
        compiled.orderedPassIndices.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: recorded graphics pass metadata count "
          "mismatch");
    }
    if (compiled.passBarrierPlans.size() !=
        compiled.orderedPassIndices.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: pass barrier plan metadata count "
          "mismatch");
    }
    if (compiled.orderedPasses.size() + compiled.culledPassCount !=
        compiled.declaredPassCount) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: ordered and culled pass counts are "
          "inconsistent");
    }
    if (compiled.rootPassCount > compiled.declaredPassCount) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: root pass count exceeds declared pass "
          "count");
    }

    std::pmr::vector<uint8_t> seenOrderedPassIndices(memory_);
    seenOrderedPassIndices.resize(compiled.declaredPassCount, 0u);
    for (const uint32_t passIndex : compiled.orderedPassIndices) {
      if (passIndex >= compiled.declaredPassCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: ordered pass index is out of range");
      }
      if (seenOrderedPassIndices[passIndex] != 0u) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: ordered pass index is duplicated");
      }
      seenOrderedPassIndices[passIndex] = 1u;
    }

    PmrHashSet<uint64_t> seenEdges(memory_);
    seenEdges.reserve(compiled.edges.size());
    std::pmr::vector<uint32_t> orderPositionByPass(memory_);
    orderPositionByPass.resize(compiled.declaredPassCount, UINT32_MAX);
    for (uint32_t i = 0u; i < compiled.orderedPassIndices.size(); ++i) {
      orderPositionByPass[compiled.orderedPassIndices[i]] = i;
    }
    for (const auto &edge : compiled.edges) {
      if (edge.before >= compiled.declaredPassCount ||
          edge.after >= compiled.declaredPassCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: dependency edge pass index is out of "
            "range");
      }
      if (edge.before == edge.after) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: dependency edge self-cycle is "
            "invalid");
      }
      if (seenOrderedPassIndices[edge.before] == 0u ||
          seenOrderedPassIndices[edge.after] == 0u) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: dependency edge references culled "
            "pass");
      }
      const uint64_t edgeKey =
          (static_cast<uint64_t>(edge.before) << 32u) | edge.after;
      if (!seenEdges.insert(edgeKey).second) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: dependency edge is duplicated");
      }
      const uint32_t beforeOrder = orderPositionByPass[edge.before];
      const uint32_t afterOrder = orderPositionByPass[edge.after];
      if (beforeOrder == UINT32_MAX || afterOrder == UINT32_MAX ||
          beforeOrder >= afterOrder) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: dependency edge violates ordered "
            "pass topology");
      }
    }

    for (uint32_t i = 0u; i < compiled.recordedGraphicsPasses.size(); ++i) {
      const auto &meta = compiled.recordedGraphicsPasses[i];
      if (meta.orderedPassIndex != i) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: recorded graphics pass ordered "
            "index mismatch");
      }
      if (meta.declaredPassIndex >= compiled.declaredPassCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: recorded graphics pass declared "
            "index is out of range");
      }
    }

    for (const auto &plan : compiled.passBarrierPlans) {
      if (plan.orderedPassIndex >= compiled.orderedPassIndices.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: pass barrier plan ordered pass "
            "index is out of range");
      }
      if (plan.barrierOffset > compiled.passBarrierRecords.size() ||
          plan.barrierCount >
              compiled.passBarrierRecords.size() - plan.barrierOffset) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: pass barrier plan range is out of "
            "bounds");
      }
    }
    if (compiled.finalBarrierPlan.barrierOffset >
            compiled.passBarrierRecords.size() ||
        compiled.finalBarrierPlan.barrierCount >
            compiled.passBarrierRecords.size() -
                compiled.finalBarrierPlan.barrierOffset) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: final barrier plan range is out of "
          "bounds");
    }
    if (compiled.textureHandlesByResource.size() != textures_.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: texture resource handle table count "
          "mismatch");
    }
    if (compiled.bufferHandlesByResource.size() != buffers_.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: buffer resource handle table count "
          "mismatch");
    }

    if (compiled.dependencyBufferRangesByPass.size() !=
        compiled.orderedPasses.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency buffer range metadata count "
          "mismatch");
    }
    if (compiled.preDispatchRangesByPass.size() !=
        compiled.orderedPasses.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: pre-dispatch range metadata count "
          "mismatch");
    }
    if (compiled.drawRangesByPass.size() != compiled.orderedPasses.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: draw range metadata count mismatch");
    }
    if (compiled.preDispatchDependencyRanges.size() !=
        compiled.ownedPreDispatches.size()) {
      return Result<bool, std::string>::makeError(
          "RenderGraphBuilder::compile: pre-dispatch dependency range metadata "
          "count mismatch");
    }

    for (uint32_t passExecIndex = 0;
         passExecIndex < compiled.orderedPasses.size(); ++passExecIndex) {
      const auto &depRange =
          compiled.dependencyBufferRangesByPass[passExecIndex];
      if (depRange.count > kMaxDependencyBuffers) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: dependency buffer range exceeds "
            "kMaxDependencyBuffers");
      }
      if (depRange.offset > compiled.resolvedDependencyBuffers.size() ||
          depRange.count >
              compiled.resolvedDependencyBuffers.size() - depRange.offset) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: dependency buffer range is out of "
            "bounds");
      }

      const auto &preDispatchRange =
          compiled.preDispatchRangesByPass[passExecIndex];
      if (preDispatchRange.offset > compiled.ownedPreDispatches.size() ||
          preDispatchRange.count >
              compiled.ownedPreDispatches.size() - preDispatchRange.offset) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: pre-dispatch range is out of bounds");
      }

      const auto &drawRange = compiled.drawRangesByPass[passExecIndex];
      if (drawRange.offset > compiled.ownedDrawItems.size() ||
          drawRange.count > compiled.ownedDrawItems.size() - drawRange.offset) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: draw range is out of bounds");
      }
    }

    for (const auto &binding : compiled.unresolvedTextureBindings) {
      if (binding.orderedPassIndex >= compiled.orderedPasses.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved texture binding pass "
            "index is out of range");
      }
      if (binding.textureResourceIndex >=
          compiled.transientTextureAllocationByResource.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved texture binding resource "
            "index is out of range");
      }
      if (binding.target !=
              RenderGraphCompileResult::PassTextureBindingTarget::Color &&
          binding.target !=
              RenderGraphCompileResult::PassTextureBindingTarget::Depth) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved texture binding target is "
            "invalid");
      }
      const uint32_t allocationIndex =
          compiled.transientTextureAllocationByResource
              [binding.textureResourceIndex];
      if (allocationIndex == UINT32_MAX ||
          allocationIndex >= compiled.transientTexturePhysicalCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved texture binding has no "
            "transient allocation");
      }
    }

    for (const auto &binding : compiled.unresolvedDependencyBufferBindings) {
      if (binding.orderedPassIndex >=
          compiled.dependencyBufferRangesByPass.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved dependency buffer binding "
            "pass index is out of range");
      }
      if (binding.bufferResourceIndex >=
          compiled.transientBufferAllocationByResource.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved dependency buffer binding "
            "resource index is out of range");
      }
      const uint32_t allocationIndex =
          compiled
              .transientBufferAllocationByResource[binding.bufferResourceIndex];
      if (allocationIndex == UINT32_MAX ||
          allocationIndex >= compiled.transientBufferPhysicalCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved dependency buffer binding "
            "has no transient allocation");
      }
      const auto &range =
          compiled.dependencyBufferRangesByPass[binding.orderedPassIndex];
      if (binding.dependencyBufferIndex >= range.count) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved dependency buffer binding "
            "slot index is out of range");
      }
    }

    for (const auto &binding :
         compiled.unresolvedPreDispatchDependencyBufferBindings) {
      if (binding.orderedPassIndex >= compiled.preDispatchRangesByPass.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
            "binding pass index is out of range");
      }
      if (binding.bufferResourceIndex >=
          compiled.transientBufferAllocationByResource.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
            "binding resource index is out of range");
      }
      const uint32_t allocationIndex =
          compiled
              .transientBufferAllocationByResource[binding.bufferResourceIndex];
      if (allocationIndex == UINT32_MAX ||
          allocationIndex >= compiled.transientBufferPhysicalCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
            "binding has no transient allocation");
      }
      const auto &preDispatchRange =
          compiled.preDispatchRangesByPass[binding.orderedPassIndex];
      if (binding.preDispatchIndex >= preDispatchRange.count) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
            "binding pre-dispatch index is out of range");
      }
      const uint32_t globalDispatchIndex =
          preDispatchRange.offset + binding.preDispatchIndex;
      if (globalDispatchIndex >= compiled.preDispatchDependencyRanges.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
            "binding global dispatch index is out of range");
      }
      const auto &depRange =
          compiled.preDispatchDependencyRanges[globalDispatchIndex];
      if (depRange.count > kMaxDependencyBuffers) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: pre-dispatch dependency range "
            "exceeds kMaxDependencyBuffers");
      }
      if (binding.dependencyBufferIndex >= depRange.count) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
            "binding slot index is out of range");
      }
    }

    for (const auto &binding : compiled.unresolvedDrawBufferBindings) {
      if (binding.orderedPassIndex >= compiled.drawRangesByPass.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved draw buffer binding pass "
            "index is out of range");
      }
      if (binding.bufferResourceIndex >=
          compiled.transientBufferAllocationByResource.size()) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved draw buffer binding "
            "resource index is out of range");
      }
      const uint32_t allocationIndex =
          compiled
              .transientBufferAllocationByResource[binding.bufferResourceIndex];
      if (allocationIndex == UINT32_MAX ||
          allocationIndex >= compiled.transientBufferPhysicalCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved draw buffer binding has "
            "no transient allocation");
      }
      const auto &drawRange =
          compiled.drawRangesByPass[binding.orderedPassIndex];
      if (binding.drawIndex >= drawRange.count) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved draw buffer binding draw "
            "index is out of range");
      }
      if (binding.target !=
              RenderGraphCompileResult::DrawBufferBindingTarget::Vertex &&
          binding.target !=
              RenderGraphCompileResult::DrawBufferBindingTarget::Index &&
          binding.target !=
              RenderGraphCompileResult::DrawBufferBindingTarget::Indirect &&
          binding.target != RenderGraphCompileResult::DrawBufferBindingTarget::
                                IndirectCount) {
        return Result<bool, std::string>::makeError(
            "RenderGraphBuilder::compile: unresolved draw buffer binding "
            "target is invalid");
      }
    }
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
  auto validateResult = compileStageC0ValidateInputs(runtime, compiled, work);
  if (validateResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        validateResult.error());
  }
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

  auto metadataValidationResult =
      compileStageC7ValidateCompiledMetadata(compiled);
  if (metadataValidationResult.hasError()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        metadataValidationResult.error());
  }

  return Result<RenderGraphCompileResult, std::string>::makeResult(
      std::move(compiled));
}

RenderGraphExecutor::RenderGraphExecutor(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      pendingFrames_(memory_), reusableTextures_(memory_),
      reusableBuffers_(memory_) {}

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
            reusableBuffers_.size() < kMaxReusableTransientBuffers) {
          ReusableBufferResource entry{};
          entry.handle = buffer;
          entry.desc = pending.bufferDescs[bufferIndex];
          entry.desc.data = {};
          reusableBuffers_.push_back(entry);
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
            reusableTextures_.size() < kMaxReusableTransientTextures) {
          ReusableTextureResource entry{};
          entry.handle = texture;
          entry.desc = pending.textureDescs[textureIndex];
          entry.desc.data = {};
          reusableTextures_.push_back(entry);
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
                             const RenderGraphCompileResult &compiled) {
  RenderGraphExecutionMetadata metadata(memory_);
  auto result = executeInternal(&runtime, gpu, compiled, &metadata);
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
                                     RenderGraphExecutionMetadata *metadata) {
  NURI_PROFILER_FUNCTION();

  const auto fail = [](RenderGraphExecutionFailureStage stage,
                       std::string_view message) {
    return Result<bool, std::string>::makeError(
        makeExecutionStageError(stage, message));
  };

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

  {
    NURI_PROFILER_ZONE("RenderGraph.execute.validate_compiled_metadata",
                       NURI_PROFILER_COLOR_BARRIER);
    if (compiled.orderedPasses.size() != compiled.orderedPassIndices.size()) {
      return fail(
          RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
          "RenderGraphExecutor::execute: ordered pass index metadata count "
          "mismatch");
    }
    if (compiled.orderedPasses.size() + compiled.culledPassCount !=
        compiled.declaredPassCount) {
      return fail(
          RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
          "RenderGraphExecutor::execute: declared/ordered/culled pass counts "
          "are inconsistent");
    }
    if (compiled.rootPassCount > compiled.declaredPassCount) {
      return fail(
          RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
          "RenderGraphExecutor::execute: root pass count exceeds declared pass "
          "count");
    }
    if (compiled.passDebugNames.size() != compiled.declaredPassCount) {
      return fail(
          RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
          "RenderGraphExecutor::execute: pass debug-name metadata count "
          "mismatch");
    }
    std::pmr::vector<uint8_t> seenOrderedPassIndices(memory_);
    seenOrderedPassIndices.resize(compiled.declaredPassCount, 0u);
    for (const uint32_t passIndex : compiled.orderedPassIndices) {
      if (passIndex >= compiled.declaredPassCount) {
        return fail(
            RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
            "RenderGraphExecutor::execute: ordered pass index is out of range");
      }
      if (seenOrderedPassIndices[passIndex] != 0u) {
        return fail(
            RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
            "RenderGraphExecutor::execute: ordered pass index is duplicated");
      }
      seenOrderedPassIndices[passIndex] = 1u;
    }
    std::pmr::vector<uint32_t> orderPositionByPass(memory_);
    orderPositionByPass.resize(compiled.declaredPassCount, UINT32_MAX);
    for (uint32_t i = 0u; i < compiled.orderedPassIndices.size(); ++i) {
      orderPositionByPass[compiled.orderedPassIndices[i]] = i;
    }
    PmrHashSet<uint64_t> seenEdges(memory_);
    seenEdges.reserve(compiled.edges.size());
    for (const auto &edge : compiled.edges) {
      if (edge.before >= compiled.declaredPassCount ||
          edge.after >= compiled.declaredPassCount) {
        return fail(
            RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
            "RenderGraphExecutor::execute: dependency edge pass index is out "
            "of "
            "range");
      }
      if (edge.before == edge.after) {
        return fail(
            RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
            "RenderGraphExecutor::execute: dependency edge self-cycle is "
            "invalid");
      }
      if (seenOrderedPassIndices[edge.before] == 0u ||
          seenOrderedPassIndices[edge.after] == 0u) {
        return fail(
            RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
            "RenderGraphExecutor::execute: dependency edge references culled "
            "pass");
      }
      const uint64_t edgeKey =
          (static_cast<uint64_t>(edge.before) << 32u) | edge.after;
      if (!seenEdges.insert(edgeKey).second) {
        return fail(
            RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
            "RenderGraphExecutor::execute: dependency edge is duplicated");
      }
      const uint32_t beforeOrder = orderPositionByPass[edge.before];
      const uint32_t afterOrder = orderPositionByPass[edge.after];
      if (beforeOrder == UINT32_MAX || afterOrder == UINT32_MAX ||
          beforeOrder >= afterOrder) {
        return fail(
            RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
            "RenderGraphExecutor::execute: dependency edge violates ordered "
            "pass "
            "topology");
      }
    }
    if (compiled.finalBarrierPlan.barrierOffset >
            compiled.passBarrierRecords.size() ||
        compiled.finalBarrierPlan.barrierCount >
            compiled.passBarrierRecords.size() -
                compiled.finalBarrierPlan.barrierOffset) {
      return fail(
          RenderGraphExecutionFailureStage::ValidateCompiledMetadata,
          "RenderGraphExecutor::execute: final barrier plan range is out of "
          "bounds");
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
    if (compiled.transientTexturePhysicalAllocations.size() !=
        compiled.transientTexturePhysicalCount) {
      return fail(
          RenderGraphExecutionFailureStage::MaterializeTransients,
          "RenderGraphExecutor::execute: transient texture allocation metadata "
          "count mismatch");
    }
    if (compiled.transientBufferPhysicalAllocations.size() !=
        compiled.transientBufferPhysicalCount) {
      return fail(
          RenderGraphExecutionFailureStage::MaterializeTransients,
          "RenderGraphExecutor::execute: transient buffer allocation metadata "
          "count mismatch");
    }

    for (const auto &allocation :
         compiled.transientTexturePhysicalAllocations) {
      if (allocation.allocationIndex >= transientTextureHandles.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::MaterializeTransients,
            "RenderGraphExecutor::execute: transient texture allocation index "
            "is out of range");
      }
      if (nuri::isValid(transientTextureHandles[allocation.allocationIndex])) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::MaterializeTransients,
            "RenderGraphExecutor::execute: transient texture allocation index "
            "is duplicated");
      }

      TextureDesc desc = allocation.desc;
      desc.data = {};
      transientTextureDescs[allocation.allocationIndex] = desc;

      TextureHandle transientTexture{};
      for (size_t poolIndex = 0u; poolIndex < reusableTextures_.size();) {
        const ReusableTextureResource &candidate = reusableTextures_[poolIndex];
        if (!nuri::isValid(candidate.handle) ||
            !gpu.isValid(candidate.handle)) {
          reusableTextures_[poolIndex] = reusableTextures_.back();
          reusableTextures_.pop_back();
          continue;
        }
        if (!isTextureDescAliasCompatible(candidate.desc, desc)) {
          ++poolIndex;
          continue;
        }

        transientTexture = candidate.handle;
        reusableTextures_[poolIndex] = reusableTextures_.back();
        reusableTextures_.pop_back();
        break;
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
      if (allocation.allocationIndex >= transientBufferHandles.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::MaterializeTransients,
            "RenderGraphExecutor::execute: transient buffer allocation index "
            "is "
            "out of range");
      }
      if (nuri::isValid(transientBufferHandles[allocation.allocationIndex])) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::MaterializeTransients,
            "RenderGraphExecutor::execute: transient buffer allocation index "
            "is "
            "duplicated");
      }

      BufferDesc desc = allocation.desc;
      desc.data = {};
      transientBufferDescs[allocation.allocationIndex] = desc;

      BufferHandle transientBuffer{};
      for (size_t poolIndex = 0u; poolIndex < reusableBuffers_.size();) {
        const ReusableBufferResource &candidate = reusableBuffers_[poolIndex];
        if (!nuri::isValid(candidate.handle) ||
            !gpu.isValid(candidate.handle)) {
          reusableBuffers_[poolIndex] = reusableBuffers_.back();
          reusableBuffers_.pop_back();
          continue;
        }
        if (!isBufferDescAliasCompatible(candidate.desc, desc)) {
          ++poolIndex;
          continue;
        }

        transientBuffer = candidate.handle;
        reusableBuffers_[poolIndex] = reusableBuffers_.back();
        reusableBuffers_.pop_back();
        break;
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
  std::pmr::vector<ComputeDispatchItem> executablePreDispatches(memory_);
  std::pmr::vector<DrawItem> executableDrawItems(memory_);
  std::pmr::vector<BufferHandle> executablePreDispatchDependencyBuffers(
      memory_);

  {
    NURI_PROFILER_ZONE("RenderGraph.execute.build_executable_payload",
                       NURI_PROFILER_COLOR_CMD_COPY);
    executablePasses = compiled.orderedPasses;
    executableDependencyBuffers = compiled.resolvedDependencyBuffers;
    executablePreDispatches = compiled.ownedPreDispatches;
    executableDrawItems = compiled.ownedDrawItems;
    executablePreDispatchDependencyBuffers =
        compiled.resolvedPreDispatchDependencyBuffers;

    if (compiled.dependencyBufferRangesByPass.size() !=
        executablePasses.size()) {
      destroyMaterializedResources();
      return fail(
          RenderGraphExecutionFailureStage::BuildExecutablePayload,
          "RenderGraphExecutor::execute: pass dependency buffer range metadata "
          "count mismatch");
    }
    if (compiled.preDispatchRangesByPass.size() != executablePasses.size()) {
      destroyMaterializedResources();
      return fail(
          RenderGraphExecutionFailureStage::BuildExecutablePayload,
          "RenderGraphExecutor::execute: pass pre-dispatch range metadata "
          "count "
          "mismatch");
    }
    if (compiled.drawRangesByPass.size() != executablePasses.size()) {
      destroyMaterializedResources();
      return fail(
          RenderGraphExecutionFailureStage::BuildExecutablePayload,
          "RenderGraphExecutor::execute: pass draw range metadata count "
          "mismatch");
    }
    if (compiled.preDispatchDependencyRanges.size() !=
        executablePreDispatches.size()) {
      destroyMaterializedResources();
      return fail(RenderGraphExecutionFailureStage::BuildExecutablePayload,
                  "RenderGraphExecutor::execute: pre-dispatch dependency range "
                  "metadata "
                  "count mismatch");
    }
    for (uint32_t orderedPassIndex = 0;
         orderedPassIndex < executablePasses.size(); ++orderedPassIndex) {
      const auto &range =
          compiled.dependencyBufferRangesByPass[orderedPassIndex];
      if (range.count > kMaxDependencyBuffers) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::BuildExecutablePayload,
            "RenderGraphExecutor::execute: pass dependency buffer range "
            "exceeds "
            "kMaxDependencyBuffers");
      }
      if (range.offset > executableDependencyBuffers.size() ||
          range.count > executableDependencyBuffers.size() - range.offset) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::BuildExecutablePayload,
            "RenderGraphExecutor::execute: pass dependency buffer range is out "
            "of bounds");
      }

      if (range.count == 0u) {
        executablePasses[orderedPassIndex].dependencyBuffers = {};
        continue;
      }

      executablePasses[orderedPassIndex].dependencyBuffers =
          std::span<const BufferHandle>(
              executableDependencyBuffers.data() + range.offset, range.count);

      const auto &preDispatchRange =
          compiled.preDispatchRangesByPass[orderedPassIndex];
      if (preDispatchRange.offset > executablePreDispatches.size() ||
          preDispatchRange.count >
              executablePreDispatches.size() - preDispatchRange.offset) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::BuildExecutablePayload,
            "RenderGraphExecutor::execute: pass pre-dispatch range is out of "
            "bounds");
      }
      if (preDispatchRange.count > 0u) {
        executablePasses[orderedPassIndex].preDispatches =
            std::span<const ComputeDispatchItem>(
                executablePreDispatches.data() + preDispatchRange.offset,
                preDispatchRange.count);
        for (uint32_t i = 0; i < preDispatchRange.count; ++i) {
          ComputeDispatchItem &dispatch =
              executablePreDispatches[preDispatchRange.offset + i];
          const auto &depRange =
              compiled.preDispatchDependencyRanges[preDispatchRange.offset + i];
          if (depRange.count > kMaxDependencyBuffers) {
            destroyMaterializedResources();
            return fail(
                RenderGraphExecutionFailureStage::BuildExecutablePayload,
                "RenderGraphExecutor::execute: pre-dispatch dependency range "
                "exceeds kMaxDependencyBuffers");
          }
          if (depRange.offset > executablePreDispatchDependencyBuffers.size() ||
              depRange.count > executablePreDispatchDependencyBuffers.size() -
                                   depRange.offset) {
            destroyMaterializedResources();
            return fail(
                RenderGraphExecutionFailureStage::BuildExecutablePayload,
                "RenderGraphExecutor::execute: pre-dispatch dependency range "
                "is out of bounds");
          }
          if (depRange.count > 0u) {
            dispatch.dependencyBuffers = std::span<const BufferHandle>(
                executablePreDispatchDependencyBuffers.data() + depRange.offset,
                depRange.count);
          } else {
            dispatch.dependencyBuffers = {};
          }
        }
      } else {
        executablePasses[orderedPassIndex].preDispatches = {};
      }

      const auto &drawRange = compiled.drawRangesByPass[orderedPassIndex];
      if (drawRange.offset > executableDrawItems.size() ||
          drawRange.count > executableDrawItems.size() - drawRange.offset) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::BuildExecutablePayload,
            "RenderGraphExecutor::execute: pass draw range is out of bounds");
      }
      if (drawRange.count > 0u) {
        executablePasses[orderedPassIndex].draws = std::span<const DrawItem>(
            executableDrawItems.data() + drawRange.offset, drawRange.count);
      } else {
        executablePasses[orderedPassIndex].draws = {};
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("RenderGraph.execute.patch_unresolved_bindings",
                       NURI_PROFILER_COLOR_CMD_COPY);
    for (const auto &binding : compiled.unresolvedTextureBindings) {
      if (binding.orderedPassIndex >= executablePasses.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved texture binding pass "
            "index is out of range");
      }
      if (binding.textureResourceIndex >=
          compiled.transientTextureAllocationByResource.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved texture binding resource "
            "index is out of range");
      }

      const uint32_t allocationIndex =
          compiled.transientTextureAllocationByResource
              [binding.textureResourceIndex];
      if (allocationIndex == UINT32_MAX ||
          allocationIndex >= transientTextureHandles.size() ||
          !nuri::isValid(transientTextureHandles[allocationIndex])) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved texture binding has no "
            "materialized allocation");
      }

      RenderPass &pass = executablePasses[binding.orderedPassIndex];
      if (binding.target ==
          RenderGraphCompileResult::PassTextureBindingTarget::Color) {
        pass.colorTexture = transientTextureHandles[allocationIndex];
      } else if (binding.target ==
                 RenderGraphCompileResult::PassTextureBindingTarget::Depth) {
        pass.depthTexture = transientTextureHandles[allocationIndex];
      } else {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved texture binding target "
            "is "
            "invalid");
      }
    }

    for (const auto &binding : compiled.unresolvedDependencyBufferBindings) {
      if (binding.orderedPassIndex >= executablePasses.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved dependency buffer "
            "binding "
            "pass index is out of range");
      }
      if (binding.bufferResourceIndex >=
          compiled.transientBufferAllocationByResource.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved dependency buffer "
            "binding "
            "resource index is out of range");
      }

      const auto &range =
          compiled.dependencyBufferRangesByPass[binding.orderedPassIndex];
      if (binding.dependencyBufferIndex >= range.count) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved dependency buffer "
            "binding "
            "slot index is out of range");
      }

      const uint32_t allocationIndex =
          compiled
              .transientBufferAllocationByResource[binding.bufferResourceIndex];
      if (allocationIndex == UINT32_MAX ||
          allocationIndex >= transientBufferHandles.size() ||
          !nuri::isValid(transientBufferHandles[allocationIndex])) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved dependency buffer "
            "binding "
            "has no materialized allocation");
      }

      executableDependencyBuffers[range.offset +
                                  binding.dependencyBufferIndex] =
          transientBufferHandles[allocationIndex];
    }

    for (const auto &binding :
         compiled.unresolvedPreDispatchDependencyBufferBindings) {
      if (binding.orderedPassIndex >= executablePasses.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved pre-dispatch dependency "
            "binding pass index is out of range");
      }
      if (binding.bufferResourceIndex >=
          compiled.transientBufferAllocationByResource.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved pre-dispatch dependency "
            "binding resource index is out of range");
      }

      const auto &preDispatchRange =
          compiled.preDispatchRangesByPass[binding.orderedPassIndex];
      if (binding.preDispatchIndex >= preDispatchRange.count) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved pre-dispatch dependency "
            "binding dispatch index is out of range");
      }
      const uint32_t globalDispatchIndex =
          preDispatchRange.offset + binding.preDispatchIndex;
      if (globalDispatchIndex >= compiled.preDispatchDependencyRanges.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved pre-dispatch dependency "
            "binding global dispatch index is out of range");
      }
      const auto &depRange =
          compiled.preDispatchDependencyRanges[globalDispatchIndex];
      if (binding.dependencyBufferIndex >= depRange.count) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved pre-dispatch dependency "
            "binding slot index is out of range");
      }

      const uint32_t allocationIndex =
          compiled
              .transientBufferAllocationByResource[binding.bufferResourceIndex];
      if (allocationIndex == UINT32_MAX ||
          allocationIndex >= transientBufferHandles.size() ||
          !nuri::isValid(transientBufferHandles[allocationIndex])) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved pre-dispatch dependency "
            "binding has no materialized allocation");
      }

      executablePreDispatchDependencyBuffers[depRange.offset +
                                             binding.dependencyBufferIndex] =
          transientBufferHandles[allocationIndex];
    }

    for (const auto &binding : compiled.unresolvedDrawBufferBindings) {
      if (binding.orderedPassIndex >= executablePasses.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved draw buffer binding pass "
            "index is out of range");
      }
      if (binding.bufferResourceIndex >=
          compiled.transientBufferAllocationByResource.size()) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved draw buffer binding "
            "resource index is out of range");
      }

      const auto &drawRange =
          compiled.drawRangesByPass[binding.orderedPassIndex];
      if (binding.drawIndex >= drawRange.count) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved draw buffer binding draw "
            "index is out of range");
      }

      const uint32_t allocationIndex =
          compiled
              .transientBufferAllocationByResource[binding.bufferResourceIndex];
      if (allocationIndex == UINT32_MAX ||
          allocationIndex >= transientBufferHandles.size() ||
          !nuri::isValid(transientBufferHandles[allocationIndex])) {
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved draw buffer binding has "
            "no materialized allocation");
      }

      DrawItem &draw =
          executableDrawItems[drawRange.offset + binding.drawIndex];
      switch (binding.target) {
      case RenderGraphCompileResult::DrawBufferBindingTarget::Vertex:
        draw.vertexBuffer = transientBufferHandles[allocationIndex];
        break;
      case RenderGraphCompileResult::DrawBufferBindingTarget::Index:
        draw.indexBuffer = transientBufferHandles[allocationIndex];
        break;
      case RenderGraphCompileResult::DrawBufferBindingTarget::Indirect:
        draw.indirectBuffer = transientBufferHandles[allocationIndex];
        break;
      case RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount:
        draw.indirectCountBuffer = transientBufferHandles[allocationIndex];
        break;
      default:
        destroyMaterializedResources();
        return fail(
            RenderGraphExecutionFailureStage::PatchUnresolvedBindings,
            "RenderGraphExecutor::execute: unresolved draw buffer binding "
            "target is invalid");
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  std::pmr::vector<GraphicsBarrierRecord> executableBarrierRecords(memory_);
  executableBarrierRecords.reserve(compiled.passBarrierRecords.size());
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.resolve_barriers",
                       NURI_PROFILER_COLOR_BARRIER);
    const auto toGraphicsAccess =
        [](RenderGraphAccessMode mode) -> GraphicsBarrierAccessMode {
      GraphicsBarrierAccessMode converted = GraphicsBarrierAccessMode::None;
      if (hasAccessFlag(mode, RenderGraphAccessMode::Read)) {
        converted = converted | GraphicsBarrierAccessMode::Read;
      }
      if (hasAccessFlag(mode, RenderGraphAccessMode::Write)) {
        converted = converted | GraphicsBarrierAccessMode::Write;
      }
      return converted;
    };
    const auto toGraphicsState =
        [](RenderGraphResourceState state) -> GraphicsBarrierState {
      switch (state) {
      case RenderGraphResourceState::Read:
        return GraphicsBarrierState::Read;
      case RenderGraphResourceState::Write:
        return GraphicsBarrierState::Write;
      case RenderGraphResourceState::Attachment:
        return GraphicsBarrierState::Attachment;
      case RenderGraphResourceState::Present:
        return GraphicsBarrierState::Present;
      case RenderGraphResourceState::Unknown:
      default:
        return GraphicsBarrierState::Unknown;
      }
    };

    for (const RenderGraphBarrierRecord &barrier :
         compiled.passBarrierRecords) {
      GraphicsBarrierRecord resolved{};
      resolved.beforeAccess = toGraphicsAccess(barrier.beforeAccess);
      resolved.afterAccess = toGraphicsAccess(barrier.afterAccess);
      resolved.beforeState = toGraphicsState(barrier.beforeState);
      resolved.afterState = toGraphicsState(barrier.afterState);

      if (barrier.resourceKind == RenderGraphBarrierResourceKind::Texture) {
        resolved.resourceKind = GraphicsBarrierResourceKind::Texture;
        if (barrier.resourceIndex >= compiled.textureHandlesByResource.size()) {
          destroyMaterializedResources();
          return fail(
              RenderGraphExecutionFailureStage::ResolveBarriers,
              "RenderGraphExecutor::execute: texture barrier resource index "
              "is out of range");
        }

        TextureHandle texture =
            compiled.textureHandlesByResource[barrier.resourceIndex];
        if (!nuri::isValid(texture)) {
          if (barrier.resourceIndex >=
              compiled.transientTextureAllocationByResource.size()) {
            destroyMaterializedResources();
            return fail(
                RenderGraphExecutionFailureStage::ResolveBarriers,
                "RenderGraphExecutor::execute: transient texture barrier "
                "resource index is out of range");
          }

          const uint32_t allocationIndex =
              compiled
                  .transientTextureAllocationByResource[barrier.resourceIndex];
          if (allocationIndex == UINT32_MAX ||
              allocationIndex >= transientTextureHandles.size() ||
              !nuri::isValid(transientTextureHandles[allocationIndex])) {
            destroyMaterializedResources();
            return fail(RenderGraphExecutionFailureStage::ResolveBarriers,
                        "RenderGraphExecutor::execute: texture barrier has no "
                        "materialized handle");
          }
          texture = transientTextureHandles[allocationIndex];
        }

        resolved.texture = texture;
      } else {
        resolved.resourceKind = GraphicsBarrierResourceKind::Buffer;
        if (barrier.resourceIndex >= compiled.bufferHandlesByResource.size()) {
          destroyMaterializedResources();
          return fail(
              RenderGraphExecutionFailureStage::ResolveBarriers,
              "RenderGraphExecutor::execute: buffer barrier resource index "
              "is out of range");
        }

        BufferHandle buffer =
            compiled.bufferHandlesByResource[barrier.resourceIndex];
        if (!nuri::isValid(buffer)) {
          if (barrier.resourceIndex >=
              compiled.transientBufferAllocationByResource.size()) {
            destroyMaterializedResources();
            return fail(
                RenderGraphExecutionFailureStage::ResolveBarriers,
                "RenderGraphExecutor::execute: transient buffer barrier "
                "resource index is out of range");
          }

          const uint32_t allocationIndex =
              compiled
                  .transientBufferAllocationByResource[barrier.resourceIndex];
          if (allocationIndex == UINT32_MAX ||
              allocationIndex >= transientBufferHandles.size() ||
              !nuri::isValid(transientBufferHandles[allocationIndex])) {
            destroyMaterializedResources();
            return fail(RenderGraphExecutionFailureStage::ResolveBarriers,
                        "RenderGraphExecutor::execute: buffer barrier has no "
                        "materialized handle");
          }
          buffer = transientBufferHandles[allocationIndex];
        }

        resolved.buffer = buffer;
      }

      executableBarrierRecords.push_back(resolved);
    }
    NURI_PROFILER_ZONE_END();
  }

  Result<bool, std::string> submitResult =
      Result<bool, std::string>::makeResult(true);
  SubmissionHandle frameSubmission{};
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.submit_frame",
                       NURI_PROFILER_COLOR_SUBMIT);
    if (metadata != nullptr) {
      metadata->usedParallelCompile = compiled.usedParallelCompile;
      metadata->usedParallelRecording = false;
      metadata->recordedCommandBuffers.clear();
      metadata->submitBatches.clear();
      metadata->passRanges.clear();
    }
    if (!executablePasses.empty()) {
      {
        NURI_PROFILER_ZONE("RenderGraph.execute.prepare_frame_output",
                           NURI_PROFILER_COLOR_WAIT);
        auto prepareFrameOutputResult = gpu.prepareFrameOutput();
        if (prepareFrameOutputResult.hasError()) {
          return failWithString(
              RenderGraphExecutionFailureStage::SubmitRecordedFrame,
              "RenderGraphExecutor::execute: failed to prepare frame output: " +
                  prepareFrameOutputResult.error());
        }
        NURI_PROFILER_ZONE_END();
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

      if (metadata != nullptr) {
        metadata->usedParallelRecording =
            supportsParallelRecording && ranges.size() > 1u;
        metadata->passRanges.reserve(ranges.size());
        for (uint32_t workerIndex = 0u; workerIndex < ranges.size();
             ++workerIndex) {
          metadata->passRanges.push_back(RenderGraphPassRange{
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
          if (orderedPassIndex >= compiled.passBarrierPlans.size()) {
            setRecordingFailure(makeExecutionStageError(
                RenderGraphExecutionFailureStage::RecordGraphicsBarriers,
                "RenderGraphExecutor::execute: ordered pass barrier plan index "
                "is out of range"));
            return;
          }
          const PassBarrierPlan &barrierPlan =
              compiled.passBarrierPlans[orderedPassIndex];
          if (barrierPlan.barrierCount > 0u) {
            auto barrierResult = gpu.recordGraphicsBarriers(
                recordingContexts[workerIndex],
                executableBarrierRecords.data() + barrierPlan.barrierOffset,
                barrierPlan.barrierCount);
            if (barrierResult.hasError()) {
              setRecordingFailure(makeExecutionStageError(
                  RenderGraphExecutionFailureStage::RecordGraphicsBarriers,
                  "RenderGraphExecutor::execute: failed to record graphics "
                  "barriers: " +
                      barrierResult.error()));
              return;
            }
          }
          auto recordResult =
              gpu.recordGraphicsPass(recordingContexts[workerIndex],
                                     executablePasses[orderedPassIndex]);
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
              executableBarrierRecords.data() +
                  compiled.finalBarrierPlan.barrierOffset,
              compiled.finalBarrierPlan.barrierCount);
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
          if (metadata != nullptr) {
            metadata->recordedCommandBuffers.push_back(
                RecordedCommandBufferMeta{
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
        if (metadata != nullptr) {
          metadata->submitBatches.assign(batches.begin(), batches.end());
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
          frameSubmission = submitFrameResult.value();
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
