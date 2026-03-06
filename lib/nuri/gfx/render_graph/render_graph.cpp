#include "nuri/pch.h"

#include "nuri/gfx/render_graph/render_graph.h"

#include "nuri/core/containers/hash_set.h"
#include "nuri/core/profiling.h"

#include <numeric>

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

[[nodiscard]] std::string_view resolveResourceDebugName(std::string_view name,
                                                        std::string_view fallback) {
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

[[nodiscard]] bool
isTextureDescAliasCompatible(const TextureDesc &a, const TextureDesc &b) {
  return a.type == b.type && a.format == b.format &&
         a.dimensions.width == b.dimensions.width &&
         a.dimensions.height == b.dimensions.height &&
         a.dimensions.depth == b.dimensions.depth && a.usage == b.usage &&
         a.storage == b.storage && a.numLayers == b.numLayers &&
         a.numSamples == b.numSamples && a.numMipLevels == b.numMipLevels &&
         a.dataNumMipLevels == b.dataNumMipLevels &&
         a.generateMipmaps == b.generateMipmaps;
}

[[nodiscard]] bool
isBufferDescAliasCompatible(const BufferDesc &a, const BufferDesc &b) {
  return a.usage == b.usage && a.storage == b.storage && a.size == b.size;
}

constexpr size_t kMaxReusableTransientTextures = 32u;
constexpr size_t kMaxReusableTransientBuffers = 64u;

[[nodiscard]] RenderGraphAccessMode attachmentAccessMode(LoadOp loadOp,
                                                         StoreOp storeOp) {
  RenderGraphAccessMode mode = RenderGraphAccessMode::None;
  if (loadOp == LoadOp::Load) {
    mode = mode | RenderGraphAccessMode::Read;
  }
  if (loadOp == LoadOp::Clear || storeOp == StoreOp::Store) {
    mode = mode | RenderGraphAccessMode::Write;
  }
  return mode;
}

} // namespace

RenderGraphBuilder::RenderGraphBuilder(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      textures_(memory_), buffers_(memory_), passes_(memory_),
      passDebugNames_(memory_), passColorTextureBindings_(memory_),
      passDepthTextureBindings_(memory_),
      passDependencyBufferBindingOffsets_(memory_),
      passDependencyBufferBindingCounts_(memory_),
      passDependencyBufferBindingResourceIndices_(memory_),
      passPreDispatchBindingOffsets_(memory_),
      passPreDispatchBindingCounts_(memory_),
      preDispatchDependencyBindingOffsets_(memory_),
      preDispatchDependencyBindingCounts_(memory_),
      preDispatchDependencyBindingResourceIndices_(memory_),
      passDrawBindingOffsets_(memory_),
      passDrawBindingCounts_(memory_),
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
      dependencyEdgeKeys_(memory_),
      dependencies_(memory_),
      passResourceAccesses_(memory_), frameOutputTextureSet_(memory_),
      frameOutputTextureIndices_(memory_), sideEffectMarkIndicesByPass_(memory_),
      sideEffectPassMarks_(memory_) {}

void RenderGraphBuilder::beginFrame(uint64_t frameIndex) {
  frameIndex_ = frameIndex;
  textures_.clear();
  buffers_.clear();
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

  const uint32_t textureIndex = static_cast<uint32_t>(textures_.size());
  if (textureIndex == UINT32_MAX) {
    return Result<RenderGraphTextureId, std::string>::makeError(
        "RenderGraphBuilder::importTexture: texture count exceeds uint32_t");
  }

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
RenderGraphBuilder::importBuffer(BufferHandle buffer, std::string_view debugName) {
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

  const uint32_t bufferIndex = static_cast<uint32_t>(buffers_.size());
  if (bufferIndex == UINT32_MAX) {
    return Result<RenderGraphBufferId, std::string>::makeError(
        "RenderGraphBuilder::importBuffer: buffer count exceeds uint32_t");
  }

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

  const uint32_t textureIndex = static_cast<uint32_t>(textures_.size());
  if (textureIndex == UINT32_MAX) {
    return Result<RenderGraphTextureId, std::string>::makeError(
        "RenderGraphBuilder::createTransientTexture: texture count exceeds "
        "uint32_t");
  }

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

  const uint32_t bufferIndex = static_cast<uint32_t>(buffers_.size());
  if (bufferIndex == UINT32_MAX) {
    return Result<RenderGraphBufferId, std::string>::makeError(
        "RenderGraphBuilder::createTransientBuffer: buffer count exceeds "
        "uint32_t");
  }

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
        "RenderGraphBuilder::addTextureAccessInternal: access mode must contain "
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

  const uint32_t accessIndex = static_cast<uint32_t>(passResourceAccesses_.size());
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

Result<bool, std::string> RenderGraphBuilder::addTextureAccess(
    RenderGraphPassId pass, RenderGraphTextureId texture,
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

  const uint32_t accessIndex = static_cast<uint32_t>(passResourceAccesses_.size());
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

Result<RenderGraphPassId, std::string>
RenderGraphBuilder::addGraphicsPass(const RenderGraphGraphicsPassDesc &desc) {
  RenderPass pass{};
  pass.color = desc.color;
  pass.depth = desc.depth;
  pass.useViewport = desc.useViewport;
  pass.viewport = desc.viewport;
  pass.preDispatches = desc.preDispatches;
  pass.dependencyBuffers = desc.dependencyBuffers;
  pass.draws = desc.draws;
  pass.debugLabel = desc.debugLabel;
  pass.debugColor = desc.debugColor;

  auto addResult = addPassRecord(pass, desc.debugLabel);
  if (addResult.hasError()) {
    return Result<RenderGraphPassId, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();

  if (nuri::isValid(desc.colorTexture)) {
    auto bindResult = bindPassColorTexture(passId, desc.colorTexture);
    if (bindResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(bindResult.error());
    }
    if (desc.markColorAsFrameOutput) {
      auto outputResult = markTextureAsFrameOutput(desc.colorTexture);
      if (outputResult.hasError()) {
        return Result<RenderGraphPassId, std::string>::makeError(
            outputResult.error());
      }
    }
  } else if (desc.markImplicitOutputSideEffect) {
    auto sideEffectResult = markPassSideEffect(passId);
    if (sideEffectResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          sideEffectResult.error());
    }
  }

  if (nuri::isValid(desc.depthTexture)) {
    auto bindResult = bindPassDepthTexture(passId, desc.depthTexture);
    if (bindResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(bindResult.error());
    }
  }

  const std::string dependencyDebugName =
      makePassResourceDebugName(desc.debugLabel, "dependency_buffer");
  for (uint32_t i = 0; i < desc.dependencyBuffers.size(); ++i) {
    const BufferHandle dependency = desc.dependencyBuffers[i];
    if (!nuri::isValid(dependency)) {
      continue;
    }
    auto importResult = importBuffer(dependency, dependencyDebugName);
    if (importResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(
          importResult.error());
    }
    auto bindResult = bindPassDependencyBuffer(
        passId, i, importResult.value(),
        RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
    if (bindResult.hasError()) {
      return Result<RenderGraphPassId, std::string>::makeError(bindResult.error());
    }
  }

  const std::string preDispatchDependencyDebugName =
      makePassResourceDebugName(desc.debugLabel, "pre_dispatch_dependency_buffer");
  for (uint32_t dispatchIndex = 0; dispatchIndex < desc.preDispatches.size();
       ++dispatchIndex) {
    const ComputeDispatchItem &dispatch = desc.preDispatches[dispatchIndex];
    for (uint32_t depIndex = 0; depIndex < dispatch.dependencyBuffers.size();
         ++depIndex) {
      const BufferHandle dependency = dispatch.dependencyBuffers[depIndex];
      if (!nuri::isValid(dependency)) {
        continue;
      }
      auto importResult = importBuffer(dependency, preDispatchDependencyDebugName);
      if (importResult.hasError()) {
        return Result<RenderGraphPassId, std::string>::makeError(
            importResult.error());
      }
      auto bindResult = bindPreDispatchDependencyBuffer(
          passId, dispatchIndex, depIndex, importResult.value(),
          RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
      if (bindResult.hasError()) {
        return Result<RenderGraphPassId, std::string>::makeError(
            bindResult.error());
      }
    }
  }

  const std::string drawDebugName =
      makePassResourceDebugName(desc.debugLabel, "draw_buffer");
  for (uint32_t drawIndex = 0; drawIndex < desc.draws.size(); ++drawIndex) {
    const DrawItem &draw = desc.draws[drawIndex];
    const std::array<std::pair<BufferHandle,
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
        return Result<RenderGraphPassId, std::string>::makeError(
            importResult.error());
      }
      auto bindResult =
          bindDrawBuffer(passId, drawIndex, target, importResult.value(),
                         RenderGraphAccessMode::Read);
      if (bindResult.hasError()) {
        return Result<RenderGraphPassId, std::string>::makeError(
            bindResult.error());
      }
    }
  }

  return Result<RenderGraphPassId, std::string>::makeResult(passId);
}

Result<RenderGraphPassId, std::string>
RenderGraphBuilder::addPassRecord(const RenderPass &pass,
                                  std::string_view debugName) {
  const uint32_t passIndex = static_cast<uint32_t>(passes_.size());
  if (passIndex == UINT32_MAX) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addPassRecord: pass count exceeds uint32_t");
  }

  passes_.push_back(pass);
  std::pmr::string resolvedName(memory_);
  const std::string_view selectedName =
      !debugName.empty() ? debugName : pass.debugLabel;
  resolvedName.assign(selectedName.data(), selectedName.size());
  passDebugNames_.push_back(std::move(resolvedName));
  passColorTextureBindings_.push_back(UINT32_MAX);
  passDepthTextureBindings_.push_back(UINT32_MAX);

  const uint32_t dependencyBindingOffset =
      static_cast<uint32_t>(passDependencyBufferBindingResourceIndices_.size());
  passDependencyBufferBindingOffsets_.push_back(dependencyBindingOffset);
  passDependencyBufferBindingCounts_.push_back(0u);

  const uint32_t preDispatchBindingOffset =
      static_cast<uint32_t>(preDispatchDependencyBindingOffsets_.size());
  passPreDispatchBindingOffsets_.push_back(preDispatchBindingOffset);
  passPreDispatchBindingCounts_.push_back(0u);

  const uint32_t drawBindingOffset =
      static_cast<uint32_t>(drawVertexBindingResourceIndices_.size());
  passDrawBindingOffsets_.push_back(drawBindingOffset);
  passDrawBindingCounts_.push_back(0u);

  const RenderGraphPassId passId{.value = passIndex};

  if (pass.dependencyBuffers.size() > UINT32_MAX) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addPassRecord: dependency buffer count "
        "exceeds uint32_t");
  }
  if (pass.dependencyBuffers.size() > kMaxDependencyBuffers) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addPassRecord: dependency buffer count "
        "exceeds kMaxDependencyBuffers");
  }
  for (const BufferHandle buffer : pass.dependencyBuffers) {
    (void)buffer;
    passDependencyBufferBindingResourceIndices_.push_back(UINT32_MAX);
  }
  passDependencyBufferBindingCounts_[passIndex] =
      static_cast<uint32_t>(pass.dependencyBuffers.size());

  if (pass.preDispatches.size() > UINT32_MAX) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addPassRecord: pre-dispatch count exceeds "
        "uint32_t");
  }
  for (const ComputeDispatchItem &dispatch : pass.preDispatches) {
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

    const uint32_t dispatchDependencyOffset =
        static_cast<uint32_t>(preDispatchDependencyBindingResourceIndices_.size());
    preDispatchDependencyBindingOffsets_.push_back(dispatchDependencyOffset);
    preDispatchDependencyBindingCounts_.push_back(
        static_cast<uint32_t>(dispatch.dependencyBuffers.size()));

    for (const BufferHandle buffer : dispatch.dependencyBuffers) {
      (void)buffer;
      preDispatchDependencyBindingResourceIndices_.push_back(UINT32_MAX);
    }
  }
  passPreDispatchBindingCounts_[passIndex] =
      static_cast<uint32_t>(pass.preDispatches.size());

  if (pass.draws.size() > UINT32_MAX) {
    return Result<RenderGraphPassId, std::string>::makeError(
        "RenderGraphBuilder::addPassRecord: draw count exceeds uint32_t");
  }
  for (const DrawItem &draw : pass.draws) {
    (void)draw;
    drawVertexBindingResourceIndices_.push_back(UINT32_MAX);
    drawIndexBindingResourceIndices_.push_back(UINT32_MAX);
    drawIndirectBindingResourceIndices_.push_back(UINT32_MAX);
    drawIndirectCountBindingResourceIndices_.push_back(UINT32_MAX);
  }
  passDrawBindingCounts_[passIndex] = static_cast<uint32_t>(pass.draws.size());
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
      attachmentAccessMode(targetPass.color.loadOp, targetPass.color.storeOp);
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
      attachmentAccessMode(targetPass.depth.loadOp, targetPass.depth.storeOp);
  if (mode == RenderGraphAccessMode::None) {
    return Result<bool, std::string>::makeResult(true);
  }
  return addTextureAccess(pass, texture, mode);
}

Result<bool, std::string> RenderGraphBuilder::bindPassDependencyBuffer(
    RenderGraphPassId pass, uint32_t dependencyIndex, RenderGraphBufferId buffer,
    RenderGraphAccessMode mode) {
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
      dispatchCount > preDispatchDependencyBindingOffsets_.size() - dispatchOffset ||
      dispatchCount > preDispatchDependencyBindingCounts_.size() - dispatchOffset) {
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
      dependencyCount >
          preDispatchDependencyBindingResourceIndices_.size() - dependencyOffset) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindPreDispatchDependencyBuffer: dependency "
        "binding range is invalid");
  }

  preDispatchDependencyBindingResourceIndices_[dependencyOffset + dependencyIndex] =
      buffer.value;
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
      drawCount > drawIndirectCountBindingResourceIndices_.size() - drawOffset) {
    return Result<bool, std::string>::makeError(
        "RenderGraphBuilder::bindDrawBuffer: draw binding range is invalid");
  }

  uint32_t *targetTableEntry = nullptr;
  switch (target) {
  case RenderGraphCompileResult::DrawBufferBindingTarget::Vertex:
    targetTableEntry = &drawVertexBindingResourceIndices_[drawOffset + drawIndex];
    break;
  case RenderGraphCompileResult::DrawBufferBindingTarget::Index:
    targetTableEntry = &drawIndexBindingResourceIndices_[drawOffset + drawIndex];
    break;
  case RenderGraphCompileResult::DrawBufferBindingTarget::Indirect:
    targetTableEntry = &drawIndirectBindingResourceIndices_[drawOffset + drawIndex];
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

Result<bool, std::string> RenderGraphBuilder::markPassSideEffectInternal(
    RenderGraphPassId pass, bool inferred) {
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

Result<RenderGraphCompileResult, std::string>
RenderGraphBuilder::compile() const {
  NURI_PROFILER_FUNCTION();

  RenderGraphCompileResult compiled(memory_);
  compiled.frameIndex = frameIndex_;
  {
    NURI_PROFILER_ZONE("RenderGraph.compile.validate_inputs",
                       NURI_PROFILER_COLOR_CREATE);
  for (const TextureResource &texture : textures_) {
    if (texture.imported) {
      if (!nuri::isValid(texture.importedHandle)) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: imported texture handle is invalid");
      }
      ++compiled.resourceStats.importedTextures;
      continue;
    }

    if (!isValidTransientTextureDesc(texture.transientDesc)) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture descriptor is "
          "invalid");
    }
    ++compiled.resourceStats.transientTextures;
  }

  for (const BufferResource &buffer : buffers_) {
    if (buffer.imported) {
      if (!nuri::isValid(buffer.importedHandle)) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: imported buffer handle is invalid");
      }
      ++compiled.resourceStats.importedBuffers;
      continue;
    }

    if (!isValidTransientBufferDesc(buffer.transientDesc)) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer descriptor is "
          "invalid");
    }
    ++compiled.resourceStats.transientBuffers;
  }

  for (const PassResourceAccess &access : passResourceAccesses_) {
    if (!isValidPassIndex(access.passIndex)) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: resource access references "
          "out-of-range pass");
    }
    if (access.resourceKind == AccessResourceKind::Texture &&
        !isValidTextureIndex(access.resourceIndex)) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: texture access references "
          "out-of-range resource");
    }
    if (access.resourceKind == AccessResourceKind::Buffer &&
        !isValidBufferIndex(access.resourceIndex)) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: buffer access references "
          "out-of-range resource");
    }
    if (!hasAccessFlag(access.mode, RenderGraphAccessMode::Read) &&
        !hasAccessFlag(access.mode, RenderGraphAccessMode::Write)) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: resource access mode is invalid");
    }
  }

  if (passes_.empty()) {
    return Result<RenderGraphCompileResult, std::string>::makeResult(
        std::move(compiled));
  }

  if (passColorTextureBindings_.size() != passes_.size() ||
      passDepthTextureBindings_.size() != passes_.size() ||
      passDependencyBufferBindingOffsets_.size() != passes_.size() ||
      passDependencyBufferBindingCounts_.size() != passes_.size() ||
      passPreDispatchBindingOffsets_.size() != passes_.size() ||
      passPreDispatchBindingCounts_.size() != passes_.size() ||
      passDrawBindingOffsets_.size() != passes_.size() ||
      passDrawBindingCounts_.size() != passes_.size()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: pass texture binding tables are out of "
        "sync");
  }
  if (drawVertexBindingResourceIndices_.size() !=
          drawIndexBindingResourceIndices_.size() ||
      drawVertexBindingResourceIndices_.size() !=
          drawIndirectBindingResourceIndices_.size() ||
      drawVertexBindingResourceIndices_.size() !=
          drawIndirectCountBindingResourceIndices_.size()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: draw buffer binding tables are out of "
        "sync");
  }
  if (preDispatchDependencyBindingOffsets_.size() !=
      preDispatchDependencyBindingCounts_.size()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: pre-dispatch dependency binding tables "
        "are out of sync");
  }
    NURI_PROFILER_ZONE_END();
  }

  const uint32_t passCount = static_cast<uint32_t>(passes_.size());
  compiled.declaredPassCount = passCount;
  std::pmr::vector<uint8_t> activePassMask(memory_);
  uint32_t activePassCount = passCount;
  std::pmr::vector<DependencyEdge> scheduledDependencies(memory_);
  std::pmr::vector<uint32_t> order(memory_);
  {
    NURI_PROFILER_ZONE("RenderGraph.compile.build_topology",
                       NURI_PROFILER_COLOR_BARRIER);
  PmrHashSet<uint64_t> dependencyEdgeKeys(memory_);
  dependencyEdgeKeys.reserve(dependencies_.size() +
                             passResourceAccesses_.size() * 2u);
  std::pmr::vector<DependencyEdge> allDependencies(memory_);
  for (const DependencyEdge edge : dependencies_) {
    const uint64_t key =
        (static_cast<uint64_t>(edge.before) << 32u) | edge.after;
    if (dependencyEdgeKeys.insert(key).second) {
      allDependencies.push_back(edge);
    }
  }
  allDependencies.reserve(dependencies_.size() + passResourceAccesses_.size() * 2u);

  const auto addDependencyEdge = [&allDependencies, &dependencyEdgeKeys](
                                     uint32_t before, uint32_t after) {
    if (before == after) {
      return;
    }
    const uint64_t key = (static_cast<uint64_t>(before) << 32u) | after;
    if (!dependencyEdgeKeys.insert(key).second) {
      return;
    }
    allDependencies.push_back(DependencyEdge{.before = before, .after = after});
  };

  std::pmr::vector<PassResourceAccess> mergedAccesses(memory_);
  mergedAccesses = passResourceAccesses_;
  std::sort(mergedAccesses.begin(), mergedAccesses.end(),
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

  if (!mergedAccesses.empty()) {
    size_t writeIndex = 0u;
    size_t groupBegin = 0u;
    while (groupBegin < mergedAccesses.size()) {
      size_t groupEnd = groupBegin + 1u;
      while (groupEnd < mergedAccesses.size() &&
             mergedAccesses[groupEnd].resourceKind ==
                 mergedAccesses[groupBegin].resourceKind &&
             mergedAccesses[groupEnd].resourceIndex ==
                 mergedAccesses[groupBegin].resourceIndex &&
             mergedAccesses[groupEnd].passIndex ==
                 mergedAccesses[groupBegin].passIndex) {
        ++groupEnd;
      }

      RenderGraphAccessMode explicitMode = RenderGraphAccessMode::None;
      RenderGraphAccessMode inferredMode = RenderGraphAccessMode::None;
      for (size_t i = groupBegin; i < groupEnd; ++i) {
        if (mergedAccesses[i].inferred) {
          inferredMode = inferredMode | mergedAccesses[i].mode;
          continue;
        }
        explicitMode = explicitMode | mergedAccesses[i].mode;
      }

      const bool hasExplicit =
          hasAccessFlag(explicitMode, RenderGraphAccessMode::Read) ||
          hasAccessFlag(explicitMode, RenderGraphAccessMode::Write);
      const RenderGraphAccessMode selectedMode =
          hasExplicit ? explicitMode : inferredMode;
      if (hasAccessFlag(selectedMode, RenderGraphAccessMode::Read) ||
          hasAccessFlag(selectedMode, RenderGraphAccessMode::Write)) {
        mergedAccesses[writeIndex++] = PassResourceAccess{
            .passIndex = mergedAccesses[groupBegin].passIndex,
            .resourceKind = mergedAccesses[groupBegin].resourceKind,
            .resourceIndex = mergedAccesses[groupBegin].resourceIndex,
            .mode = selectedMode,
            .inferred = !hasExplicit,
        };
      }
      groupBegin = groupEnd;
    }
    mergedAccesses.resize(writeIndex);
  }

  const auto addResourceHazards = [&](AccessResourceKind resourceKind) {
    std::pmr::vector<uint32_t> activeReaders(memory_);
    activeReaders.reserve(passCount);
    uint32_t currentResource = UINT32_MAX;
    uint32_t lastWriter = UINT32_MAX;

    for (const PassResourceAccess &access : mergedAccesses) {
      if (access.resourceKind != resourceKind) {
        continue;
      }
      if (access.resourceIndex != currentResource) {
        currentResource = access.resourceIndex;
        lastWriter = UINT32_MAX;
        activeReaders.clear();
      }

      const bool hasRead = hasAccessFlag(access.mode, RenderGraphAccessMode::Read);
      const bool hasWrite =
          hasAccessFlag(access.mode, RenderGraphAccessMode::Write);
      if (!hasRead && !hasWrite) {
        continue;
      }

      if (hasRead && lastWriter != UINT32_MAX) {
        addDependencyEdge(lastWriter, access.passIndex);
      }
      if (hasWrite) {
        if (lastWriter != UINT32_MAX) {
          addDependencyEdge(lastWriter, access.passIndex);
        }
        for (const uint32_t reader : activeReaders) {
          addDependencyEdge(reader, access.passIndex);
        }
        activeReaders.clear();
        lastWriter = access.passIndex;
      } else {
        activeReaders.push_back(access.passIndex);
      }
    }
  };

  addResourceHazards(AccessResourceKind::Texture);
  addResourceHazards(AccessResourceKind::Buffer);

  activePassMask.resize(passCount, 1u);
  if (!frameOutputTextureIndices_.empty() || !sideEffectPassMarks_.empty()) {
    std::fill(activePassMask.begin(), activePassMask.end(), 0u);

    std::pmr::vector<uint32_t> reverseCount(memory_);
    reverseCount.resize(passCount, 0u);
    for (const DependencyEdge edge : allDependencies) {
      if (edge.before >= passCount || edge.after >= passCount) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: dependency edge references "
            "out-of-range pass");
      }
      ++reverseCount[edge.after];
    }

    std::pmr::vector<uint32_t> reverseOffsets(memory_);
    reverseOffsets.resize(static_cast<size_t>(passCount) + 1u, 0u);
    for (uint32_t i = 0; i < passCount; ++i) {
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
    stack.reserve(passCount);
    const auto pushRoot = [&activePassMask, &stack, &compiled](uint32_t passIndex) {
      if (activePassMask[passIndex] != 0u) {
        return;
      }
      activePassMask[passIndex] = 1u;
      stack.push_back(passIndex);
      ++compiled.rootPassCount;
    };

    const bool hasExplicitFrameOutputRoots = !frameOutputTextureIndices_.empty();
    const bool suppressInferredSideEffectRoots =
        suppressInferredSideEffectsWhenExplicitOutputs_ &&
        hasExplicitFrameOutputRoots;
    for (const SideEffectPassMark &mark : sideEffectPassMarks_) {
      const uint32_t passIndex = mark.passIndex;
      if (!isValidPassIndex(passIndex)) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: side-effect pass index is out of "
            "range");
      }
      if (mark.inferred && suppressInferredSideEffectRoots) {
        continue;
      }
      pushRoot(passIndex);
    }

    std::pmr::vector<RenderGraphAccessMode> textureAccessByPass(memory_);
    textureAccessByPass.resize(passCount, RenderGraphAccessMode::None);
    for (const uint32_t textureIndex : frameOutputTextureIndices_) {
      if (!isValidTextureIndex(textureIndex)) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: frame-output texture index is out of "
            "range");
      }

      std::fill(textureAccessByPass.begin(), textureAccessByPass.end(),
                RenderGraphAccessMode::None);
      for (const PassResourceAccess &access : mergedAccesses) {
        if (access.resourceKind != AccessResourceKind::Texture ||
            access.resourceIndex != textureIndex) {
          continue;
        }
        textureAccessByPass[access.passIndex] =
            textureAccessByPass[access.passIndex] | access.mode;
      }
      for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
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
        if (activePassMask[predecessor] != 0u) {
          continue;
        }
        activePassMask[predecessor] = 1u;
        stack.push_back(predecessor);
      }
    }

    activePassCount = 0u;
    for (const uint8_t isActive : activePassMask) {
      activePassCount += static_cast<uint32_t>(isActive != 0u);
    }
    compiled.culledPassCount = passCount - activePassCount;
  }

  scheduledDependencies.reserve(allDependencies.size());
  for (const DependencyEdge edge : allDependencies) {
    if (activePassMask[edge.before] == 0u || activePassMask[edge.after] == 0u) {
      continue;
    }
    scheduledDependencies.push_back(edge);
  }

  std::pmr::vector<uint32_t> indegree(memory_);
  indegree.resize(passCount, 0u);

  std::pmr::vector<uint32_t> outgoingCount(memory_);
  outgoingCount.resize(passCount, 0u);

  for (const DependencyEdge edge : scheduledDependencies) {
    if (!isValidPassIndex(edge.before) || !isValidPassIndex(edge.after)) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency edge references "
          "out-of-range pass");
    }
    if (indegree[edge.after] == UINT32_MAX) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency indegree overflow");
    }
    ++indegree[edge.after];
    if (outgoingCount[edge.before] == UINT32_MAX) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency outgoing edge overflow");
    }
    ++outgoingCount[edge.before];
  }

  std::pmr::vector<uint32_t> outgoingOffsets(memory_);
  outgoingOffsets.resize(static_cast<size_t>(passCount) + 1u, 0u);
  for (uint32_t i = 0; i < passCount; ++i) {
    outgoingOffsets[i + 1u] = outgoingOffsets[i] + outgoingCount[i];
  }

  std::pmr::vector<uint32_t> outgoingEdges(memory_);
  outgoingEdges.resize(scheduledDependencies.size(), 0u);

  std::pmr::vector<uint32_t> outgoingCursor(memory_);
  outgoingCursor = outgoingOffsets;
  for (const DependencyEdge edge : scheduledDependencies) {
    const uint32_t cursor = outgoingCursor[edge.before]++;
    outgoingEdges[cursor] = edge.after;
  }

  std::pmr::vector<uint32_t> ready(memory_);
  ready.reserve(passCount);
  for (uint32_t i = 0; i < indegree.size(); ++i) {
    if (activePassMask[i] != 0u && indegree[i] == 0u) {
      ready.push_back(i);
    }
  }

  order.reserve(activePassCount);
  while (!ready.empty()) {
    auto minIt = std::min_element(ready.begin(), ready.end());
    NURI_ASSERT(minIt != ready.end(),
                "RenderGraphBuilder::compile: ready set is unexpectedly empty");
    const uint32_t current = *minIt;
    ready.erase(minIt);
    order.push_back(current);

    const uint32_t start = outgoingOffsets[current];
    const uint32_t end = outgoingOffsets[current + 1u];
    for (uint32_t edgeIndex = start; edgeIndex < end; ++edgeIndex) {
      const uint32_t next = outgoingEdges[edgeIndex];
      if (indegree[next] == 0u) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: indegree underflow");
      }
      --indegree[next];
      if (indegree[next] == 0u) {
        ready.push_back(next);
      }
    }
  }

  if (order.size() != activePassCount) {
    std::ostringstream message;
    message << "RenderGraphBuilder::compile: dependency cycle detected";

    bool hasCyclePass = false;
    for (uint32_t i = 0; i < passCount; ++i) {
      if (activePassMask[i] == 0u || indegree[i] == 0u) {
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

    return Result<RenderGraphCompileResult, std::string>::makeError(
        message.str());
  }
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("RenderGraph.compile.resolve_pass_payloads",
                       NURI_PROFILER_COLOR_CREATE);
  compiled.passDebugNames.reserve(passDebugNames_.size());
  for (const std::pmr::string &name : passDebugNames_) {
    std::pmr::string copiedName(memory_);
    copiedName.assign(name.data(), name.size());
    compiled.passDebugNames.push_back(std::move(copiedName));
  }

  compiled.edges.reserve(scheduledDependencies.size());
  for (const DependencyEdge edge : scheduledDependencies) {
    compiled.edges.push_back(
        RenderGraphCompileResult::Edge{.before = edge.before, .after = edge.after});
  }

  size_t totalDependencyBufferSlots = 0u;
  size_t totalPreDispatchItems = 0u;
  size_t totalPreDispatchDependencySlots = 0u;
  size_t totalDrawItems = 0u;
  for (const uint32_t passIndex : order) {
    const RenderPass &sourcePass = passes_[passIndex];

    const uint32_t count = passDependencyBufferBindingCounts_[passIndex];
    const uint32_t offset = passDependencyBufferBindingOffsets_[passIndex];
    if (sourcePass.dependencyBuffers.size() != count) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency buffer binding count does "
          "not match pass dependency buffer count");
    }
    if (offset > passDependencyBufferBindingResourceIndices_.size() ||
        count > passDependencyBufferBindingResourceIndices_.size() - offset) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency buffer binding range is "
          "invalid");
    }
    totalDependencyBufferSlots += count;

    const uint32_t preDispatchCount = passPreDispatchBindingCounts_[passIndex];
    const uint32_t preDispatchOffset = passPreDispatchBindingOffsets_[passIndex];
    if (sourcePass.preDispatches.size() != preDispatchCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: pre-dispatch binding count does not "
          "match pass pre-dispatch count");
    }
    if (preDispatchOffset > preDispatchDependencyBindingOffsets_.size() ||
        preDispatchCount >
            preDispatchDependencyBindingOffsets_.size() - preDispatchOffset ||
        preDispatchCount >
            preDispatchDependencyBindingCounts_.size() - preDispatchOffset) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: pre-dispatch binding range is "
          "invalid");
    }
    totalPreDispatchItems += preDispatchCount;
    for (uint32_t i = 0; i < preDispatchCount; ++i) {
      const uint32_t depOffset =
          preDispatchDependencyBindingOffsets_[preDispatchOffset + i];
      const uint32_t depCount =
          preDispatchDependencyBindingCounts_[preDispatchOffset + i];
      if (depOffset > preDispatchDependencyBindingResourceIndices_.size() ||
          depCount >
              preDispatchDependencyBindingResourceIndices_.size() - depOffset) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: pre-dispatch dependency binding "
            "range is invalid");
      }
      totalPreDispatchDependencySlots += depCount;
    }

    const uint32_t drawCount = passDrawBindingCounts_[passIndex];
    const uint32_t drawOffset = passDrawBindingOffsets_[passIndex];
    if (sourcePass.draws.size() != drawCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: draw binding count does not match "
          "pass draw count");
    }
    if (drawOffset > drawVertexBindingResourceIndices_.size() ||
        drawCount > drawVertexBindingResourceIndices_.size() - drawOffset ||
        drawCount > drawIndexBindingResourceIndices_.size() - drawOffset ||
        drawCount > drawIndirectBindingResourceIndices_.size() - drawOffset ||
        drawCount >
            drawIndirectCountBindingResourceIndices_.size() - drawOffset) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: draw binding range is invalid");
    }

    if (nuri::isValid(sourcePass.colorTexture) &&
        passColorTextureBindings_[passIndex] == UINT32_MAX) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: pass requires explicit color texture "
          "binding");
    }
    if (nuri::isValid(sourcePass.depthTexture) &&
        passDepthTextureBindings_[passIndex] == UINT32_MAX) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: pass requires explicit depth texture "
          "binding");
    }

    for (uint32_t depIndex = 0; depIndex < count; ++depIndex) {
      if (!nuri::isValid(sourcePass.dependencyBuffers[depIndex])) {
        continue;
      }
      if (passDependencyBufferBindingResourceIndices_[offset + depIndex] ==
          UINT32_MAX) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: pass requires explicit dependency "
            "buffer binding");
      }
    }

    for (uint32_t dispatchIndex = 0; dispatchIndex < preDispatchCount;
         ++dispatchIndex) {
      const ComputeDispatchItem &dispatch = sourcePass.preDispatches[dispatchIndex];
      const uint32_t depOffset =
          preDispatchDependencyBindingOffsets_[preDispatchOffset + dispatchIndex];
      const uint32_t depCount =
          preDispatchDependencyBindingCounts_[preDispatchOffset + dispatchIndex];
      for (uint32_t depIndex = 0; depIndex < depCount; ++depIndex) {
        if (!nuri::isValid(dispatch.dependencyBuffers[depIndex])) {
          continue;
        }
        if (preDispatchDependencyBindingResourceIndices_[depOffset + depIndex] ==
            UINT32_MAX) {
          return Result<RenderGraphCompileResult, std::string>::makeError(
              "RenderGraphBuilder::compile: pass requires explicit "
              "pre-dispatch dependency buffer binding");
        }
      }
    }

    for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex) {
      const DrawItem &draw = sourcePass.draws[drawIndex];
      const uint32_t globalDrawIndex = drawOffset + drawIndex;
      if (nuri::isValid(draw.vertexBuffer) &&
          drawVertexBindingResourceIndices_[globalDrawIndex] == UINT32_MAX) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: pass requires explicit draw vertex "
            "buffer binding");
      }
      if (nuri::isValid(draw.indexBuffer) &&
          drawIndexBindingResourceIndices_[globalDrawIndex] == UINT32_MAX) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: pass requires explicit draw index "
            "buffer binding");
      }
      if (nuri::isValid(draw.indirectBuffer) &&
          drawIndirectBindingResourceIndices_[globalDrawIndex] == UINT32_MAX) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: pass requires explicit draw "
            "indirect buffer binding");
      }
      if (nuri::isValid(draw.indirectCountBuffer) &&
          drawIndirectCountBindingResourceIndices_[globalDrawIndex] ==
              UINT32_MAX) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: pass requires explicit draw "
            "indirect-count buffer binding");
      }
    }

    totalDrawItems += drawCount;
  }

  compiled.resolvedDependencyBuffers.reserve(totalDependencyBufferSlots);
  compiled.dependencyBufferRangesByPass.resize(order.size());
  compiled.ownedPreDispatches.reserve(totalPreDispatchItems);
  compiled.preDispatchRangesByPass.resize(order.size());
  compiled.resolvedPreDispatchDependencyBuffers.reserve(
      totalPreDispatchDependencySlots);
  compiled.preDispatchDependencyRanges.reserve(totalPreDispatchItems);
  compiled.ownedDrawItems.reserve(totalDrawItems);
  compiled.drawRangesByPass.resize(order.size());

  compiled.orderedPasses.reserve(order.size());
  compiled.orderedPassIndices.reserve(order.size());
  for (const uint32_t passIndex : order) {
    const RenderPass &sourcePass = passes_[passIndex];
    RenderPass resolvedPass = sourcePass;
    const uint32_t orderedPassIndex =
        static_cast<uint32_t>(compiled.orderedPasses.size());

    const uint32_t colorTextureIndex = passColorTextureBindings_[passIndex];
    if (colorTextureIndex != UINT32_MAX) {
      if (!isValidTextureIndex(colorTextureIndex)) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: color texture binding references "
            "out-of-range texture");
      }
      const TextureResource &resource = textures_[colorTextureIndex];
      if (resource.imported) {
        resolvedPass.colorTexture = resource.importedHandle;
      } else {
        resolvedPass.colorTexture = {};
        compiled.unresolvedTextureBindings.push_back(
            RenderGraphCompileResult::PassTextureBinding{
                .orderedPassIndex = orderedPassIndex,
                .textureResourceIndex = colorTextureIndex,
                .target =
                    RenderGraphCompileResult::PassTextureBindingTarget::Color,
            });
      }
    }

    const uint32_t depthTextureIndex = passDepthTextureBindings_[passIndex];
    if (depthTextureIndex != UINT32_MAX) {
      if (!isValidTextureIndex(depthTextureIndex)) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: depth texture binding references "
            "out-of-range texture");
      }
      const TextureResource &resource = textures_[depthTextureIndex];
      if (resource.imported) {
        resolvedPass.depthTexture = resource.importedHandle;
      } else {
        resolvedPass.depthTexture = {};
        compiled.unresolvedTextureBindings.push_back(
            RenderGraphCompileResult::PassTextureBinding{
                .orderedPassIndex = orderedPassIndex,
                .textureResourceIndex = depthTextureIndex,
                .target =
                    RenderGraphCompileResult::PassTextureBindingTarget::Depth,
        });
      }
    }

    const uint32_t dependencyCount =
        passDependencyBufferBindingCounts_[passIndex];
    if (dependencyCount > kMaxDependencyBuffers) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: pass dependency buffer count exceeds "
          "kMaxDependencyBuffers");
    }
    const uint32_t dependencyOffset =
        passDependencyBufferBindingOffsets_[passIndex];
    if (dependencyOffset > passDependencyBufferBindingResourceIndices_.size() ||
        dependencyCount >
            passDependencyBufferBindingResourceIndices_.size() -
                dependencyOffset) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency buffer binding range "
          "references out-of-range data");
    }

    const uint32_t resolvedDependencyOffset =
        static_cast<uint32_t>(compiled.resolvedDependencyBuffers.size());
    compiled.dependencyBufferRangesByPass[orderedPassIndex] =
        RenderGraphCompileResult::PassDependencyBufferRange{
            .offset = resolvedDependencyOffset,
            .count = dependencyCount,
        };
    if (dependencyCount > 0u) {
      for (uint32_t i = 0; i < dependencyCount; ++i) {
        const uint32_t resourceIndex =
            passDependencyBufferBindingResourceIndices_[dependencyOffset + i];
        if (resourceIndex == UINT32_MAX) {
          compiled.resolvedDependencyBuffers.push_back(BufferHandle{});
          continue;
        }
        if (!isValidBufferIndex(resourceIndex)) {
          return Result<RenderGraphCompileResult, std::string>::makeError(
              "RenderGraphBuilder::compile: dependency buffer binding "
              "references out-of-range resource");
        }

        const BufferResource &resource = buffers_[resourceIndex];
        if (resource.imported) {
          compiled.resolvedDependencyBuffers.push_back(resource.importedHandle);
          continue;
        }

        compiled.resolvedDependencyBuffers.push_back(BufferHandle{});
        compiled.unresolvedDependencyBufferBindings.push_back(
            RenderGraphCompileResult::UnresolvedDependencyBufferBinding{
                .orderedPassIndex = orderedPassIndex,
                .dependencyBufferIndex = i,
                .bufferResourceIndex = resourceIndex,
            });
      }
      resolvedPass.dependencyBuffers = std::span<const BufferHandle>(
          compiled.resolvedDependencyBuffers.data() + resolvedDependencyOffset,
          dependencyCount);
    } else {
      resolvedPass.dependencyBuffers = {};
    }

    const uint32_t passPreDispatchCount = passPreDispatchBindingCounts_[passIndex];
    const uint32_t passPreDispatchOffset =
        passPreDispatchBindingOffsets_[passIndex];
    if (sourcePass.preDispatches.size() != passPreDispatchCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: pre-dispatch binding count does not "
          "match pass pre-dispatch count");
    }
    const uint32_t preDispatchOutputOffset =
        static_cast<uint32_t>(compiled.ownedPreDispatches.size());
    compiled.preDispatchRangesByPass[orderedPassIndex] =
        RenderGraphCompileResult::PassDispatchRange{
            .offset = preDispatchOutputOffset,
            .count = passPreDispatchCount,
        };
    for (uint32_t dispatchIndex = 0; dispatchIndex < passPreDispatchCount;
         ++dispatchIndex) {
      const ComputeDispatchItem &sourceDispatch =
          sourcePass.preDispatches[dispatchIndex];
      ComputeDispatchItem resolvedDispatch = sourceDispatch;

      const uint32_t globalDispatchBindingIndex =
          passPreDispatchOffset + dispatchIndex;
      const uint32_t dispatchDependencyOffset =
          preDispatchDependencyBindingOffsets_[globalDispatchBindingIndex];
      const uint32_t dispatchDependencyCount =
          preDispatchDependencyBindingCounts_[globalDispatchBindingIndex];
      if (dispatchDependencyCount > kMaxDependencyBuffers) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: pre-dispatch dependency buffer "
            "count exceeds kMaxDependencyBuffers");
      }
      if (sourceDispatch.dependencyBuffers.size() != dispatchDependencyCount) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            "RenderGraphBuilder::compile: pre-dispatch dependency binding "
            "count does not match dispatch dependency count");
      }

      const uint32_t resolvedDispatchDependencyOffset =
          static_cast<uint32_t>(
              compiled.resolvedPreDispatchDependencyBuffers.size());
      compiled.preDispatchDependencyRanges.push_back(
          RenderGraphCompileResult::DispatchDependencyBufferRange{
              .offset = resolvedDispatchDependencyOffset,
              .count = dispatchDependencyCount,
          });

      for (uint32_t depIndex = 0; depIndex < dispatchDependencyCount;
           ++depIndex) {
        const uint32_t resourceIndex =
            preDispatchDependencyBindingResourceIndices_[dispatchDependencyOffset +
                                                        depIndex];
        if (resourceIndex == UINT32_MAX) {
          compiled.resolvedPreDispatchDependencyBuffers.push_back(BufferHandle{});
          continue;
        }
        if (!isValidBufferIndex(resourceIndex)) {
          return Result<RenderGraphCompileResult, std::string>::makeError(
              "RenderGraphBuilder::compile: pre-dispatch dependency binding "
              "references out-of-range resource");
        }

        const BufferResource &resource = buffers_[resourceIndex];
        if (resource.imported) {
          compiled.resolvedPreDispatchDependencyBuffers.push_back(
              resource.importedHandle);
          continue;
        }

        compiled.resolvedPreDispatchDependencyBuffers.push_back(BufferHandle{});
        compiled.unresolvedPreDispatchDependencyBufferBindings.push_back(
            RenderGraphCompileResult::UnresolvedPreDispatchDependencyBufferBinding{
                .orderedPassIndex = orderedPassIndex,
                .preDispatchIndex = dispatchIndex,
                .dependencyBufferIndex = depIndex,
                .bufferResourceIndex = resourceIndex,
            });
      }

      if (dispatchDependencyCount > 0u) {
        resolvedDispatch.dependencyBuffers = std::span<const BufferHandle>(
            compiled.resolvedPreDispatchDependencyBuffers.data() +
                resolvedDispatchDependencyOffset,
            dispatchDependencyCount);
      } else {
        resolvedDispatch.dependencyBuffers = {};
      }

      compiled.ownedPreDispatches.push_back(resolvedDispatch);
    }
    if (passPreDispatchCount > 0u) {
      resolvedPass.preDispatches = std::span<const ComputeDispatchItem>(
          compiled.ownedPreDispatches.data() + preDispatchOutputOffset,
          passPreDispatchCount);
    } else {
      resolvedPass.preDispatches = {};
    }

    const uint32_t passDrawCount = passDrawBindingCounts_[passIndex];
    const uint32_t passDrawOffset = passDrawBindingOffsets_[passIndex];
    if (sourcePass.draws.size() != passDrawCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: draw binding count does not match pass "
          "draw count");
    }
    const uint32_t drawOutputOffset =
        static_cast<uint32_t>(compiled.ownedDrawItems.size());
    compiled.drawRangesByPass[orderedPassIndex] =
        RenderGraphCompileResult::PassDrawRange{
            .offset = drawOutputOffset,
            .count = passDrawCount,
        };

    for (uint32_t drawIndex = 0; drawIndex < passDrawCount; ++drawIndex) {
      const DrawItem &sourceDraw = sourcePass.draws[drawIndex];
      DrawItem resolvedDraw = sourceDraw;
      const uint32_t globalDrawIndex = passDrawOffset + drawIndex;

      const auto resolveDrawBinding = [&](uint32_t resourceIndex,
                                          RenderGraphCompileResult::
                                              DrawBufferBindingTarget target,
                                          BufferHandle &slotHandle) {
        if (resourceIndex == UINT32_MAX) {
          slotHandle = {};
          return Result<bool, std::string>::makeResult(true);
        }
        if (!isValidBufferIndex(resourceIndex)) {
          return Result<bool, std::string>::makeError(
              "RenderGraphBuilder::compile: draw buffer binding references "
              "out-of-range resource");
        }

        const BufferResource &resource = buffers_[resourceIndex];
        if (resource.imported) {
          slotHandle = resource.importedHandle;
          return Result<bool, std::string>::makeResult(true);
        }

        slotHandle = {};
        compiled.unresolvedDrawBufferBindings.push_back(
            RenderGraphCompileResult::UnresolvedDrawBufferBinding{
                .orderedPassIndex = orderedPassIndex,
                .drawIndex = drawIndex,
                .target = target,
                .bufferResourceIndex = resourceIndex,
            });
        return Result<bool, std::string>::makeResult(true);
      };

      auto vertexResult = resolveDrawBinding(
          drawVertexBindingResourceIndices_[globalDrawIndex],
          RenderGraphCompileResult::DrawBufferBindingTarget::Vertex,
          resolvedDraw.vertexBuffer);
      if (vertexResult.hasError()) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            vertexResult.error());
      }

      auto indexResult = resolveDrawBinding(
          drawIndexBindingResourceIndices_[globalDrawIndex],
          RenderGraphCompileResult::DrawBufferBindingTarget::Index,
          resolvedDraw.indexBuffer);
      if (indexResult.hasError()) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            indexResult.error());
      }

      auto indirectResult = resolveDrawBinding(
          drawIndirectBindingResourceIndices_[globalDrawIndex],
          RenderGraphCompileResult::DrawBufferBindingTarget::Indirect,
          resolvedDraw.indirectBuffer);
      if (indirectResult.hasError()) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            indirectResult.error());
      }

      auto indirectCountResult = resolveDrawBinding(
          drawIndirectCountBindingResourceIndices_[globalDrawIndex],
          RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount,
          resolvedDraw.indirectCountBuffer);
      if (indirectCountResult.hasError()) {
        return Result<RenderGraphCompileResult, std::string>::makeError(
            indirectCountResult.error());
      }

      compiled.ownedDrawItems.push_back(resolvedDraw);
    }
    if (passDrawCount > 0u) {
      resolvedPass.draws = std::span<const DrawItem>(
          compiled.ownedDrawItems.data() + drawOutputOffset, passDrawCount);
    } else {
      resolvedPass.draws = {};
    }

    compiled.orderedPassIndices.push_back(passIndex);
    compiled.orderedPasses.push_back(resolvedPass);
  }
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("RenderGraph.compile.plan_transient_lifetimes",
                       NURI_PROFILER_COLOR_CREATE);
  std::pmr::vector<uint32_t> executionRankByPass(memory_);
  executionRankByPass.resize(passCount, UINT32_MAX);
  for (uint32_t rank = 0; rank < order.size(); ++rank) {
    executionRankByPass[order[rank]] = rank;
  }

  std::pmr::vector<uint32_t> transientTextureFirstRank(memory_);
  std::pmr::vector<uint32_t> transientTextureLastRank(memory_);
  transientTextureFirstRank.resize(textures_.size(), UINT32_MAX);
  transientTextureLastRank.resize(textures_.size(), 0u);

  std::pmr::vector<uint32_t> transientBufferFirstRank(memory_);
  std::pmr::vector<uint32_t> transientBufferLastRank(memory_);
  transientBufferFirstRank.resize(buffers_.size(), UINT32_MAX);
  transientBufferLastRank.resize(buffers_.size(), 0u);

  for (const PassResourceAccess &access : passResourceAccesses_) {
    if (access.passIndex >= passCount || activePassMask[access.passIndex] == 0u) {
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
      if (transientTextureFirstRank[access.resourceIndex] == UINT32_MAX) {
        transientTextureFirstRank[access.resourceIndex] = rank;
        transientTextureLastRank[access.resourceIndex] = rank;
      } else {
        transientTextureFirstRank[access.resourceIndex] =
            std::min(transientTextureFirstRank[access.resourceIndex], rank);
        transientTextureLastRank[access.resourceIndex] =
            std::max(transientTextureLastRank[access.resourceIndex], rank);
      }
      continue;
    }

    if (access.resourceIndex >= buffers_.size() ||
        buffers_[access.resourceIndex].imported) {
      continue;
    }
    if (transientBufferFirstRank[access.resourceIndex] == UINT32_MAX) {
      transientBufferFirstRank[access.resourceIndex] = rank;
      transientBufferLastRank[access.resourceIndex] = rank;
    } else {
      transientBufferFirstRank[access.resourceIndex] =
          std::min(transientBufferFirstRank[access.resourceIndex], rank);
      transientBufferLastRank[access.resourceIndex] =
          std::max(transientBufferLastRank[access.resourceIndex], rank);
    }
  }

  for (uint32_t textureIndex = 0; textureIndex < textures_.size();
       ++textureIndex) {
    if (textures_[textureIndex].imported ||
        transientTextureFirstRank[textureIndex] == UINT32_MAX) {
      continue;
    }
    compiled.transientTextureLifetimes.push_back(
        RenderGraphCompileResult::TransientLifetime{
            .resourceIndex = textureIndex,
            .firstExecutionIndex = transientTextureFirstRank[textureIndex],
            .lastExecutionIndex = transientTextureLastRank[textureIndex],
        });
  }

  for (uint32_t bufferIndex = 0; bufferIndex < buffers_.size(); ++bufferIndex) {
    if (buffers_[bufferIndex].imported ||
        transientBufferFirstRank[bufferIndex] == UINT32_MAX) {
      continue;
    }
    compiled.transientBufferLifetimes.push_back(
        RenderGraphCompileResult::TransientLifetime{
            .resourceIndex = bufferIndex,
            .firstExecutionIndex = transientBufferFirstRank[bufferIndex],
            .lastExecutionIndex = transientBufferLastRank[bufferIndex],
        });
  }
    NURI_PROFILER_ZONE_END();
  }

  compiled.transientTextureAllocationByResource.resize(textures_.size(),
                                                       UINT32_MAX);
  compiled.transientBufferAllocationByResource.resize(buffers_.size(),
                                                      UINT32_MAX);

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
    slotRepresentativeResource.reserve(compiled.transientTextureLifetimes.size());

    for (const uint32_t lifetimeIndex : orderIndices) {
      const auto &lifetime = compiled.transientTextureLifetimes[lifetimeIndex];
      uint32_t chosenSlot = UINT32_MAX;
      for (uint32_t slot = 0; slot < slotLastUse.size(); ++slot) {
        if (slotLastUse[slot] >= lifetime.firstExecutionIndex) {
          continue;
        }
        const uint32_t representative = slotRepresentativeResource[slot];
        if (representative >= textures_.size() || lifetime.resourceIndex >= textures_.size()) {
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
    compiled.transientTexturePhysicalCount =
        static_cast<uint32_t>(compiled.transientTexturePhysicalAllocations.size());
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
    slotRepresentativeResource.reserve(compiled.transientBufferLifetimes.size());

    for (const uint32_t lifetimeIndex : orderIndices) {
      const auto &lifetime = compiled.transientBufferLifetimes[lifetimeIndex];
      uint32_t chosenSlot = UINT32_MAX;
      for (uint32_t slot = 0; slot < slotLastUse.size(); ++slot) {
        if (slotLastUse[slot] >= lifetime.firstExecutionIndex) {
          continue;
        }
        const uint32_t representative = slotRepresentativeResource[slot];
        if (representative >= buffers_.size() || lifetime.resourceIndex >= buffers_.size()) {
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
    compiled.transientBufferPhysicalCount =
        static_cast<uint32_t>(compiled.transientBufferPhysicalAllocations.size());
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("RenderGraph.compile.validate_compiled_metadata",
                       NURI_PROFILER_COLOR_BARRIER);
  for (const auto &lifetime : compiled.transientTextureLifetimes) {
    if (lifetime.resourceIndex >=
        compiled.transientTextureAllocationByResource.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture allocation map is "
          "out of range");
    }
    const uint32_t allocationIndex =
        compiled.transientTextureAllocationByResource[lifetime.resourceIndex];
    if (allocationIndex == UINT32_MAX ||
        allocationIndex >= compiled.transientTexturePhysicalCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture allocation map is "
          "incomplete or invalid");
    }
  }

  for (const auto &lifetime : compiled.transientBufferLifetimes) {
    if (lifetime.resourceIndex >=
        compiled.transientBufferAllocationByResource.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer allocation map is "
          "out of range");
    }
    const uint32_t allocationIndex =
        compiled.transientBufferAllocationByResource[lifetime.resourceIndex];
    if (allocationIndex == UINT32_MAX ||
        allocationIndex >= compiled.transientBufferPhysicalCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer allocation map is "
          "incomplete or invalid");
    }
  }

  if (compiled.transientTextureAllocations.size() !=
      compiled.transientTextureLifetimes.size()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: transient texture allocation metadata "
        "count mismatch");
  }
  if (compiled.transientBufferAllocations.size() !=
      compiled.transientBufferLifetimes.size()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: transient buffer allocation metadata "
        "count mismatch");
  }
  if (compiled.transientTexturePhysicalAllocations.size() !=
      compiled.transientTexturePhysicalCount) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: transient texture physical allocation "
        "metadata count mismatch");
  }
  if (compiled.transientBufferPhysicalAllocations.size() !=
      compiled.transientBufferPhysicalCount) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: transient buffer physical allocation "
        "metadata count mismatch");
  }

  uint32_t previousTextureResourceIndex = UINT32_MAX;
  for (size_t i = 0; i < compiled.transientTextureAllocations.size(); ++i) {
    const auto &allocation = compiled.transientTextureAllocations[i];
    if (allocation.resourceIndex >=
        compiled.transientTextureAllocationByResource.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture allocation resource "
          "index is out of range");
    }
    if (allocation.allocationIndex >= compiled.transientTexturePhysicalCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture allocation slot "
          "index is out of range");
    }
    if (i > 0u && allocation.resourceIndex <= previousTextureResourceIndex) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture allocations are not "
          "strictly ordered by resource index");
    }
    previousTextureResourceIndex = allocation.resourceIndex;
    if (compiled.transientTextureAllocationByResource[allocation.resourceIndex] !=
        allocation.allocationIndex) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture allocation map "
          "entry mismatch");
    }
  }

  uint32_t previousBufferResourceIndex = UINT32_MAX;
  for (size_t i = 0; i < compiled.transientBufferAllocations.size(); ++i) {
    const auto &allocation = compiled.transientBufferAllocations[i];
    if (allocation.resourceIndex >=
        compiled.transientBufferAllocationByResource.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer allocation resource "
          "index is out of range");
    }
    if (allocation.allocationIndex >= compiled.transientBufferPhysicalCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer allocation slot "
          "index is out of range");
    }
    if (i > 0u && allocation.resourceIndex <= previousBufferResourceIndex) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer allocations are not "
          "strictly ordered by resource index");
    }
    previousBufferResourceIndex = allocation.resourceIndex;
    if (compiled.transientBufferAllocationByResource[allocation.resourceIndex] !=
        allocation.allocationIndex) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer allocation map entry "
          "mismatch");
    }
  }

  std::pmr::vector<uint8_t> seenTexturePhysicalSlots(memory_);
  seenTexturePhysicalSlots.resize(compiled.transientTexturePhysicalCount, 0u);
  for (const auto &allocation : compiled.transientTexturePhysicalAllocations) {
    if (allocation.allocationIndex >= compiled.transientTexturePhysicalCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture physical allocation "
          "slot index is out of range");
    }
    if (seenTexturePhysicalSlots[allocation.allocationIndex] != 0u) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture physical allocation "
          "slot index is duplicated");
    }
    seenTexturePhysicalSlots[allocation.allocationIndex] = 1u;

    if (allocation.representativeResourceIndex >=
        compiled.transientTextureAllocationByResource.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture physical allocation "
          "representative resource index is out of range");
    }
    if (allocation.representativeResourceIndex >= textures_.size() ||
        textures_[allocation.representativeResourceIndex].imported) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture physical allocation "
          "representative resource is invalid");
    }
    if (compiled.transientTextureAllocationByResource
            [allocation.representativeResourceIndex] !=
        allocation.allocationIndex) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture physical allocation "
          "representative map entry mismatch");
    }
  }
  for (uint32_t slot = 0; slot < compiled.transientTexturePhysicalCount; ++slot) {
    if (seenTexturePhysicalSlots[slot] == 0u) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient texture physical allocation "
          "metadata is missing a slot");
    }
  }

  std::pmr::vector<uint8_t> seenBufferPhysicalSlots(memory_);
  seenBufferPhysicalSlots.resize(compiled.transientBufferPhysicalCount, 0u);
  for (const auto &allocation : compiled.transientBufferPhysicalAllocations) {
    if (allocation.allocationIndex >= compiled.transientBufferPhysicalCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer physical allocation "
          "slot index is out of range");
    }
    if (seenBufferPhysicalSlots[allocation.allocationIndex] != 0u) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer physical allocation "
          "slot index is duplicated");
    }
    seenBufferPhysicalSlots[allocation.allocationIndex] = 1u;

    if (allocation.representativeResourceIndex >=
        compiled.transientBufferAllocationByResource.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer physical allocation "
          "representative resource index is out of range");
    }
    if (allocation.representativeResourceIndex >= buffers_.size() ||
        buffers_[allocation.representativeResourceIndex].imported) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer physical allocation "
          "representative resource is invalid");
    }
    if (compiled.transientBufferAllocationByResource
            [allocation.representativeResourceIndex] !=
        allocation.allocationIndex) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer physical allocation "
          "representative map entry mismatch");
    }
  }
  for (uint32_t slot = 0; slot < compiled.transientBufferPhysicalCount; ++slot) {
    if (seenBufferPhysicalSlots[slot] == 0u) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: transient buffer physical allocation "
          "metadata is missing a slot");
    }
  }

  if (compiled.passDebugNames.size() != compiled.declaredPassCount) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: pass debug-name metadata count mismatch");
  }
  if (compiled.orderedPasses.size() != compiled.orderedPassIndices.size()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: ordered pass metadata count mismatch");
  }
  if (compiled.orderedPasses.size() + compiled.culledPassCount !=
      compiled.declaredPassCount) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: ordered and culled pass counts are "
        "inconsistent");
  }
  if (compiled.rootPassCount > compiled.declaredPassCount) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: root pass count exceeds declared pass "
        "count");
  }
  std::pmr::vector<uint8_t> seenOrderedPassIndices(memory_);
  seenOrderedPassIndices.resize(compiled.declaredPassCount, 0u);
  for (const uint32_t passIndex : compiled.orderedPassIndices) {
    if (passIndex >= compiled.declaredPassCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: ordered pass index is out of range");
    }
    if (seenOrderedPassIndices[passIndex] != 0u) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
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
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency edge pass index is out of "
          "range");
    }
    if (edge.before == edge.after) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency edge self-cycle is invalid");
    }
    if (seenOrderedPassIndices[edge.before] == 0u ||
        seenOrderedPassIndices[edge.after] == 0u) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency edge references culled "
          "pass");
    }
    const uint64_t edgeKey =
        (static_cast<uint64_t>(edge.before) << 32u) | edge.after;
    if (!seenEdges.insert(edgeKey).second) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency edge is duplicated");
    }
    const uint32_t beforeOrder = orderPositionByPass[edge.before];
    const uint32_t afterOrder = orderPositionByPass[edge.after];
    if (beforeOrder == UINT32_MAX || afterOrder == UINT32_MAX ||
        beforeOrder >= afterOrder) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency edge violates ordered pass "
          "topology");
    }
  }

  if (compiled.dependencyBufferRangesByPass.size() !=
      compiled.orderedPasses.size()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: dependency buffer range metadata count "
        "mismatch");
  }
  if (compiled.preDispatchRangesByPass.size() != compiled.orderedPasses.size()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: pre-dispatch range metadata count "
        "mismatch");
  }
  if (compiled.drawRangesByPass.size() != compiled.orderedPasses.size()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: draw range metadata count mismatch");
  }
  if (compiled.preDispatchDependencyRanges.size() !=
      compiled.ownedPreDispatches.size()) {
    return Result<RenderGraphCompileResult, std::string>::makeError(
        "RenderGraphBuilder::compile: pre-dispatch dependency range metadata "
        "count mismatch");
  }

  for (uint32_t passExecIndex = 0;
       passExecIndex < compiled.orderedPasses.size(); ++passExecIndex) {
    const auto &depRange = compiled.dependencyBufferRangesByPass[passExecIndex];
    if (depRange.count > kMaxDependencyBuffers) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency buffer range exceeds "
          "kMaxDependencyBuffers");
    }
    if (depRange.offset > compiled.resolvedDependencyBuffers.size() ||
        depRange.count >
            compiled.resolvedDependencyBuffers.size() - depRange.offset) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: dependency buffer range is out of "
          "bounds");
    }

    const auto &preDispatchRange =
        compiled.preDispatchRangesByPass[passExecIndex];
    if (preDispatchRange.offset > compiled.ownedPreDispatches.size() ||
        preDispatchRange.count >
            compiled.ownedPreDispatches.size() - preDispatchRange.offset) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: pre-dispatch range is out of bounds");
    }

    const auto &drawRange = compiled.drawRangesByPass[passExecIndex];
    if (drawRange.offset > compiled.ownedDrawItems.size() ||
        drawRange.count > compiled.ownedDrawItems.size() - drawRange.offset) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: draw range is out of bounds");
    }
  }

  for (const auto &binding : compiled.unresolvedTextureBindings) {
    if (binding.orderedPassIndex >= compiled.orderedPasses.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved texture binding pass index "
          "is out of range");
    }
    if (binding.textureResourceIndex >=
        compiled.transientTextureAllocationByResource.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved texture binding resource "
          "index is out of range");
    }
    if (binding.target !=
            RenderGraphCompileResult::PassTextureBindingTarget::Color &&
        binding.target !=
            RenderGraphCompileResult::PassTextureBindingTarget::Depth) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved texture binding target is "
          "invalid");
    }
    const uint32_t allocationIndex =
        compiled
            .transientTextureAllocationByResource[binding.textureResourceIndex];
    if (allocationIndex == UINT32_MAX ||
        allocationIndex >= compiled.transientTexturePhysicalCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved texture binding has no "
          "transient allocation");
    }
  }

  for (const auto &binding : compiled.unresolvedDependencyBufferBindings) {
    if (binding.orderedPassIndex >= compiled.dependencyBufferRangesByPass.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved dependency buffer binding "
          "pass index is out of range");
    }
    if (binding.bufferResourceIndex >=
        compiled.transientBufferAllocationByResource.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved dependency buffer binding "
          "resource index is out of range");
    }
    const uint32_t allocationIndex =
        compiled
            .transientBufferAllocationByResource[binding.bufferResourceIndex];
    if (allocationIndex == UINT32_MAX ||
        allocationIndex >= compiled.transientBufferPhysicalCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved dependency buffer binding "
          "has no transient allocation");
    }
    const auto &range =
        compiled.dependencyBufferRangesByPass[binding.orderedPassIndex];
    if (binding.dependencyBufferIndex >= range.count) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved dependency buffer binding "
          "slot index is out of range");
    }
  }

  for (const auto &binding :
       compiled.unresolvedPreDispatchDependencyBufferBindings) {
    if (binding.orderedPassIndex >= compiled.preDispatchRangesByPass.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
          "binding pass index is out of range");
    }
    if (binding.bufferResourceIndex >=
        compiled.transientBufferAllocationByResource.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
          "binding resource index is out of range");
    }
    const uint32_t allocationIndex =
        compiled
            .transientBufferAllocationByResource[binding.bufferResourceIndex];
    if (allocationIndex == UINT32_MAX ||
        allocationIndex >= compiled.transientBufferPhysicalCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
          "binding has no transient allocation");
    }
    const auto &preDispatchRange =
        compiled.preDispatchRangesByPass[binding.orderedPassIndex];
    if (binding.preDispatchIndex >= preDispatchRange.count) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
          "binding pre-dispatch index is out of range");
    }
    const uint32_t globalDispatchIndex =
        preDispatchRange.offset + binding.preDispatchIndex;
    if (globalDispatchIndex >= compiled.preDispatchDependencyRanges.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
          "binding global dispatch index is out of range");
    }
    const auto &depRange =
        compiled.preDispatchDependencyRanges[globalDispatchIndex];
    if (depRange.count > kMaxDependencyBuffers) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: pre-dispatch dependency range "
          "exceeds kMaxDependencyBuffers");
    }
    if (binding.dependencyBufferIndex >= depRange.count) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved pre-dispatch dependency "
          "binding slot index is out of range");
    }
  }

  for (const auto &binding : compiled.unresolvedDrawBufferBindings) {
    if (binding.orderedPassIndex >= compiled.drawRangesByPass.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved draw buffer binding pass "
          "index is out of range");
    }
    if (binding.bufferResourceIndex >=
        compiled.transientBufferAllocationByResource.size()) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved draw buffer binding "
          "resource index is out of range");
    }
    const uint32_t allocationIndex =
        compiled
            .transientBufferAllocationByResource[binding.bufferResourceIndex];
    if (allocationIndex == UINT32_MAX ||
        allocationIndex >= compiled.transientBufferPhysicalCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved draw buffer binding has "
          "no transient allocation");
    }
    const auto &drawRange = compiled.drawRangesByPass[binding.orderedPassIndex];
    if (binding.drawIndex >= drawRange.count) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved draw buffer binding draw "
          "index is out of range");
    }
    if (binding.target != RenderGraphCompileResult::DrawBufferBindingTarget::Vertex &&
        binding.target != RenderGraphCompileResult::DrawBufferBindingTarget::Index &&
        binding.target != RenderGraphCompileResult::DrawBufferBindingTarget::Indirect &&
        binding.target !=
            RenderGraphCompileResult::DrawBufferBindingTarget::IndirectCount) {
      return Result<RenderGraphCompileResult, std::string>::makeError(
          "RenderGraphBuilder::compile: unresolved draw buffer binding target "
          "is invalid");
    }
  }
    NURI_PROFILER_ZONE_END();
  }

  return Result<RenderGraphCompileResult, std::string>::makeResult(
      std::move(compiled));
}

RenderGraphExecutor::RenderGraphExecutor(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      pendingFrames_(memory_), reusableTextures_(memory_),
      reusableBuffers_(memory_) {}

void RenderGraphExecutor::collectRetiredResources(GPUDevice &gpu,
                                                  uint64_t completedFrameIndex) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_DESTROY);
  size_t writeIndex = 0u;
  for (size_t readIndex = 0u; readIndex < pendingFrames_.size(); ++readIndex) {
    PendingFrameResources &pending = pendingFrames_[readIndex];
    if (pending.retireAfterFrame > completedFrameIndex) {
      if (writeIndex != readIndex) {
        pendingFrames_[writeIndex] = std::move(pending);
      }
      ++writeIndex;
      continue;
    }

    const bool hasBufferDescs = pending.bufferDescs.size() == pending.buffers.size();
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

Result<bool, std::string>
RenderGraphExecutor::execute(GPUDevice &gpu,
                             const RenderGraphCompileResult &compiled) {
  NURI_PROFILER_FUNCTION();

  {
    NURI_PROFILER_ZONE("RenderGraph.execute.retire_resources",
                       NURI_PROFILER_COLOR_DESTROY);
  collectRetiredResources(gpu, compiled.frameIndex);
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("RenderGraph.execute.validate_compiled_metadata",
                       NURI_PROFILER_COLOR_BARRIER);
  if (compiled.orderedPasses.size() != compiled.orderedPassIndices.size()) {
    return Result<bool, std::string>::makeError(
        "RenderGraphExecutor::execute: ordered pass index metadata count "
        "mismatch");
  }
  if (compiled.orderedPasses.size() + compiled.culledPassCount !=
      compiled.declaredPassCount) {
    return Result<bool, std::string>::makeError(
        "RenderGraphExecutor::execute: declared/ordered/culled pass counts "
        "are inconsistent");
  }
  if (compiled.rootPassCount > compiled.declaredPassCount) {
    return Result<bool, std::string>::makeError(
        "RenderGraphExecutor::execute: root pass count exceeds declared pass "
        "count");
  }
  if (compiled.passDebugNames.size() != compiled.declaredPassCount) {
    return Result<bool, std::string>::makeError(
        "RenderGraphExecutor::execute: pass debug-name metadata count "
        "mismatch");
  }
  std::pmr::vector<uint8_t> seenOrderedPassIndices(memory_);
  seenOrderedPassIndices.resize(compiled.declaredPassCount, 0u);
  for (const uint32_t passIndex : compiled.orderedPassIndices) {
    if (passIndex >= compiled.declaredPassCount) {
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: ordered pass index is out of range");
    }
    if (seenOrderedPassIndices[passIndex] != 0u) {
      return Result<bool, std::string>::makeError(
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
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: dependency edge pass index is out of "
          "range");
    }
    if (edge.before == edge.after) {
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: dependency edge self-cycle is "
          "invalid");
    }
    if (seenOrderedPassIndices[edge.before] == 0u ||
        seenOrderedPassIndices[edge.after] == 0u) {
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: dependency edge references culled "
          "pass");
    }
    const uint64_t edgeKey =
        (static_cast<uint64_t>(edge.before) << 32u) | edge.after;
    if (!seenEdges.insert(edgeKey).second) {
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: dependency edge is duplicated");
    }
    const uint32_t beforeOrder = orderPositionByPass[edge.before];
    const uint32_t afterOrder = orderPositionByPass[edge.after];
    if (beforeOrder == UINT32_MAX || afterOrder == UINT32_MAX ||
        beforeOrder >= afterOrder) {
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: dependency edge violates ordered pass "
          "topology");
    }
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
    return Result<bool, std::string>::makeError(
        "RenderGraphExecutor::execute: transient texture allocation metadata "
        "count mismatch");
  }
  if (compiled.transientBufferPhysicalAllocations.size() !=
      compiled.transientBufferPhysicalCount) {
    return Result<bool, std::string>::makeError(
        "RenderGraphExecutor::execute: transient buffer allocation metadata "
        "count mismatch");
  }

  for (const auto &allocation : compiled.transientTexturePhysicalAllocations) {
    if (allocation.allocationIndex >= transientTextureHandles.size()) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: transient texture allocation index "
          "is out of range");
    }
    if (nuri::isValid(transientTextureHandles[allocation.allocationIndex])) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: transient texture allocation index "
          "is duplicated");
    }

    TextureDesc desc = allocation.desc;
    desc.data = {};
    transientTextureDescs[allocation.allocationIndex] = desc;

    TextureHandle transientTexture{};
    for (size_t poolIndex = 0u; poolIndex < reusableTextures_.size();) {
      const ReusableTextureResource &candidate = reusableTextures_[poolIndex];
      if (!nuri::isValid(candidate.handle) || !gpu.isValid(candidate.handle)) {
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
        return Result<bool, std::string>::makeError(
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
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: transient buffer allocation index is "
          "out of range");
    }
    if (nuri::isValid(transientBufferHandles[allocation.allocationIndex])) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: transient buffer allocation index is "
          "duplicated");
    }

    BufferDesc desc = allocation.desc;
    desc.data = {};
    transientBufferDescs[allocation.allocationIndex] = desc;

    BufferHandle transientBuffer{};
    for (size_t poolIndex = 0u; poolIndex < reusableBuffers_.size();) {
      const ReusableBufferResource &candidate = reusableBuffers_[poolIndex];
      if (!nuri::isValid(candidate.handle) || !gpu.isValid(candidate.handle)) {
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
        return Result<bool, std::string>::makeError(
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
  std::pmr::vector<BufferHandle> executablePreDispatchDependencyBuffers(memory_);

  {
    NURI_PROFILER_ZONE("RenderGraph.execute.build_executable_payload",
                       NURI_PROFILER_COLOR_CMD_COPY);
  executablePasses = compiled.orderedPasses;
  executableDependencyBuffers = compiled.resolvedDependencyBuffers;
  executablePreDispatches = compiled.ownedPreDispatches;
  executableDrawItems = compiled.ownedDrawItems;
  executablePreDispatchDependencyBuffers =
      compiled.resolvedPreDispatchDependencyBuffers;

  if (compiled.dependencyBufferRangesByPass.size() != executablePasses.size()) {
    destroyMaterializedResources();
    return Result<bool, std::string>::makeError(
        "RenderGraphExecutor::execute: pass dependency buffer range metadata "
        "count mismatch");
  }
  if (compiled.preDispatchRangesByPass.size() != executablePasses.size()) {
    destroyMaterializedResources();
    return Result<bool, std::string>::makeError(
        "RenderGraphExecutor::execute: pass pre-dispatch range metadata count "
        "mismatch");
  }
  if (compiled.drawRangesByPass.size() != executablePasses.size()) {
    destroyMaterializedResources();
    return Result<bool, std::string>::makeError(
        "RenderGraphExecutor::execute: pass draw range metadata count "
        "mismatch");
  }
  if (compiled.preDispatchDependencyRanges.size() !=
      executablePreDispatches.size()) {
    destroyMaterializedResources();
    return Result<bool, std::string>::makeError(
        "RenderGraphExecutor::execute: pre-dispatch dependency range metadata "
        "count mismatch");
  }
  for (uint32_t orderedPassIndex = 0; orderedPassIndex < executablePasses.size();
       ++orderedPassIndex) {
    const auto &range = compiled.dependencyBufferRangesByPass[orderedPassIndex];
    if (range.count > kMaxDependencyBuffers) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: pass dependency buffer range exceeds "
          "kMaxDependencyBuffers");
    }
    if (range.offset > executableDependencyBuffers.size() ||
        range.count > executableDependencyBuffers.size() - range.offset) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
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
      return Result<bool, std::string>::makeError(
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
          return Result<bool, std::string>::makeError(
              "RenderGraphExecutor::execute: pre-dispatch dependency range "
              "exceeds kMaxDependencyBuffers");
        }
        if (depRange.offset > executablePreDispatchDependencyBuffers.size() ||
            depRange.count > executablePreDispatchDependencyBuffers.size() -
                                 depRange.offset) {
          destroyMaterializedResources();
          return Result<bool, std::string>::makeError(
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
      return Result<bool, std::string>::makeError(
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
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved texture binding pass "
          "index is out of range");
    }
    if (binding.textureResourceIndex >=
        compiled.transientTextureAllocationByResource.size()) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved texture binding resource "
          "index is out of range");
    }

    const uint32_t allocationIndex =
        compiled
            .transientTextureAllocationByResource[binding.textureResourceIndex];
    if (allocationIndex == UINT32_MAX ||
        allocationIndex >= transientTextureHandles.size() ||
        !nuri::isValid(transientTextureHandles[allocationIndex])) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
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
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved texture binding target is "
          "invalid");
    }
  }

  for (const auto &binding : compiled.unresolvedDependencyBufferBindings) {
    if (binding.orderedPassIndex >= executablePasses.size()) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved dependency buffer binding "
          "pass index is out of range");
    }
    if (binding.bufferResourceIndex >=
        compiled.transientBufferAllocationByResource.size()) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved dependency buffer binding "
          "resource index is out of range");
    }

    const auto &range =
        compiled.dependencyBufferRangesByPass[binding.orderedPassIndex];
    if (binding.dependencyBufferIndex >= range.count) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved dependency buffer binding "
          "slot index is out of range");
    }

    const uint32_t allocationIndex =
        compiled
            .transientBufferAllocationByResource[binding.bufferResourceIndex];
    if (allocationIndex == UINT32_MAX ||
        allocationIndex >= transientBufferHandles.size() ||
        !nuri::isValid(transientBufferHandles[allocationIndex])) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved dependency buffer binding "
          "has no materialized allocation");
    }

    executableDependencyBuffers[range.offset + binding.dependencyBufferIndex] =
        transientBufferHandles[allocationIndex];
  }

  for (const auto &binding :
       compiled.unresolvedPreDispatchDependencyBufferBindings) {
    if (binding.orderedPassIndex >= executablePasses.size()) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved pre-dispatch dependency "
          "binding pass index is out of range");
    }
    if (binding.bufferResourceIndex >=
        compiled.transientBufferAllocationByResource.size()) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved pre-dispatch dependency "
          "binding resource index is out of range");
    }

    const auto &preDispatchRange =
        compiled.preDispatchRangesByPass[binding.orderedPassIndex];
    if (binding.preDispatchIndex >= preDispatchRange.count) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved pre-dispatch dependency "
          "binding dispatch index is out of range");
    }
    const uint32_t globalDispatchIndex =
        preDispatchRange.offset + binding.preDispatchIndex;
    if (globalDispatchIndex >= compiled.preDispatchDependencyRanges.size()) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved pre-dispatch dependency "
          "binding global dispatch index is out of range");
    }
    const auto &depRange = compiled.preDispatchDependencyRanges[globalDispatchIndex];
    if (binding.dependencyBufferIndex >= depRange.count) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
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
      return Result<bool, std::string>::makeError(
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
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved draw buffer binding pass "
          "index is out of range");
    }
    if (binding.bufferResourceIndex >=
        compiled.transientBufferAllocationByResource.size()) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved draw buffer binding "
          "resource index is out of range");
    }

    const auto &drawRange = compiled.drawRangesByPass[binding.orderedPassIndex];
    if (binding.drawIndex >= drawRange.count) {
      destroyMaterializedResources();
      return Result<bool, std::string>::makeError(
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
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved draw buffer binding has "
          "no materialized allocation");
    }

    DrawItem &draw = executableDrawItems[drawRange.offset + binding.drawIndex];
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
      return Result<bool, std::string>::makeError(
          "RenderGraphExecutor::execute: unresolved draw buffer binding "
          "target is invalid");
    }
  }
    NURI_PROFILER_ZONE_END();
  }

  Result<bool, std::string> submitResult =
      Result<bool, std::string>::makeResult(true);
  {
    NURI_PROFILER_ZONE("RenderGraph.execute.submit_frame",
                       NURI_PROFILER_COLOR_SUBMIT);
    if (!executablePasses.empty()) {
      const RenderFrame frame{
          .passes = std::span<const RenderPass>(executablePasses.data(),
                                                executablePasses.size()),
      };
      submitResult = gpu.submitFrame(frame);
    }
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("RenderGraph.execute.defer_transient_retire",
                       NURI_PROFILER_COLOR_DESTROY);
    PendingFrameResources pending(memory_);
    const uint64_t retireLag =
        static_cast<uint64_t>(std::max(1u, gpu.getSwapchainImageCount())) + 1ull;
    pending.retireAfterFrame = compiled.frameIndex + retireLag;
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
