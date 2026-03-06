#pragma once

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/containers/hash_set.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/gpu_render_types.h"

#include <cstdint>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>

namespace nuri {

struct NURI_API RenderGraphPassId {
  uint32_t value = UINT32_MAX;
};

struct NURI_API RenderGraphTextureId {
  uint32_t value = UINT32_MAX;
};

struct NURI_API RenderGraphBufferId {
  uint32_t value = UINT32_MAX;
};

[[nodiscard]] constexpr bool isValid(RenderGraphPassId id) {
  return id.value != UINT32_MAX;
}

[[nodiscard]] constexpr bool isValid(RenderGraphTextureId id) {
  return id.value != UINT32_MAX;
}

[[nodiscard]] constexpr bool isValid(RenderGraphBufferId id) {
  return id.value != UINT32_MAX;
}

enum class RenderGraphAccessMode : uint8_t {
  None = 0,
  Read = 1u << 0u,
  Write = 1u << 1u,
};

[[nodiscard]] constexpr RenderGraphAccessMode
operator|(RenderGraphAccessMode lhs, RenderGraphAccessMode rhs) {
  return static_cast<RenderGraphAccessMode>(static_cast<uint8_t>(lhs) |
                                            static_cast<uint8_t>(rhs));
}

[[nodiscard]] constexpr bool hasAccessFlag(RenderGraphAccessMode mode,
                                           RenderGraphAccessMode flag) {
  return (static_cast<uint8_t>(mode) & static_cast<uint8_t>(flag)) != 0u;
}

struct NURI_API RenderGraphGraphicsPassDesc {
  AttachmentColor color{};
  RenderGraphTextureId colorTexture{};
  AttachmentDepth depth{};
  RenderGraphTextureId depthTexture{};
  bool useViewport = false;
  Viewport viewport{};
  std::span<const ComputeDispatchItem> preDispatches{};
  std::span<const BufferHandle> dependencyBuffers{};
  std::span<const DrawItem> draws{};
  std::string_view debugLabel{};
  uint32_t debugColor = 0xffffffffu;
  bool markColorAsFrameOutput = true;
  bool markImplicitOutputSideEffect = true;
};

struct NURI_API RenderGraphCompileResult {
  struct Edge {
    uint32_t before = UINT32_MAX;
    uint32_t after = UINT32_MAX;
  };

  struct ResourceStats {
    uint32_t importedTextures = 0;
    uint32_t transientTextures = 0;
    uint32_t importedBuffers = 0;
    uint32_t transientBuffers = 0;
  };

  struct TransientLifetime {
    uint32_t resourceIndex = UINT32_MAX;
    uint32_t firstExecutionIndex = UINT32_MAX;
    uint32_t lastExecutionIndex = UINT32_MAX;
  };

  struct TransientAllocation {
    uint32_t resourceIndex = UINT32_MAX;
    uint32_t allocationIndex = UINT32_MAX;
  };

  struct TransientTexturePhysicalAllocation {
    uint32_t allocationIndex = UINT32_MAX;
    uint32_t representativeResourceIndex = UINT32_MAX;
    TextureDesc desc{};
  };

  struct TransientBufferPhysicalAllocation {
    uint32_t allocationIndex = UINT32_MAX;
    uint32_t representativeResourceIndex = UINT32_MAX;
    BufferDesc desc{};
  };

  enum class PassTextureBindingTarget : uint8_t {
    Color = 0,
    Depth = 1,
  };

  struct PassTextureBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t textureResourceIndex = UINT32_MAX;
    PassTextureBindingTarget target = PassTextureBindingTarget::Color;
  };

  struct PassDependencyBufferRange {
    uint32_t offset = 0;
    uint32_t count = 0;
  };

  struct UnresolvedDependencyBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t dependencyBufferIndex = UINT32_MAX;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };

  struct PassDispatchRange {
    uint32_t offset = 0;
    uint32_t count = 0;
  };

  struct PassDrawRange {
    uint32_t offset = 0;
    uint32_t count = 0;
  };

  struct DispatchDependencyBufferRange {
    uint32_t offset = 0;
    uint32_t count = 0;
  };

  enum class DrawBufferBindingTarget : uint8_t {
    Vertex = 0,
    Index = 1,
    Indirect = 2,
    IndirectCount = 3,
  };

  struct UnresolvedPreDispatchDependencyBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t preDispatchIndex = UINT32_MAX;
    uint32_t dependencyBufferIndex = UINT32_MAX;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };

  struct UnresolvedDrawBufferBinding {
    uint32_t orderedPassIndex = UINT32_MAX;
    uint32_t drawIndex = UINT32_MAX;
    DrawBufferBindingTarget target = DrawBufferBindingTarget::Vertex;
    uint32_t bufferResourceIndex = UINT32_MAX;
  };

  uint64_t frameIndex = 0;
  uint32_t declaredPassCount = 0;
  uint32_t culledPassCount = 0;
  uint32_t rootPassCount = 0;
  uint32_t transientTexturePhysicalCount = 0;
  uint32_t transientBufferPhysicalCount = 0;
  ResourceStats resourceStats{};
  std::pmr::vector<RenderPass> orderedPasses;
  std::pmr::vector<uint32_t> orderedPassIndices;
  std::pmr::vector<std::pmr::string> passDebugNames;
  std::pmr::vector<Edge> edges;
  std::pmr::vector<TransientLifetime> transientTextureLifetimes;
  std::pmr::vector<TransientLifetime> transientBufferLifetimes;
  std::pmr::vector<TransientAllocation> transientTextureAllocations;
  std::pmr::vector<TransientAllocation> transientBufferAllocations;
  std::pmr::vector<uint32_t> transientTextureAllocationByResource;
  std::pmr::vector<uint32_t> transientBufferAllocationByResource;
  std::pmr::vector<TransientTexturePhysicalAllocation>
      transientTexturePhysicalAllocations;
  std::pmr::vector<TransientBufferPhysicalAllocation>
      transientBufferPhysicalAllocations;
  std::pmr::vector<PassTextureBinding> unresolvedTextureBindings;
  std::pmr::vector<BufferHandle> resolvedDependencyBuffers;
  std::pmr::vector<PassDependencyBufferRange> dependencyBufferRangesByPass;
  std::pmr::vector<UnresolvedDependencyBufferBinding>
      unresolvedDependencyBufferBindings;
  std::pmr::vector<ComputeDispatchItem> ownedPreDispatches;
  std::pmr::vector<DrawItem> ownedDrawItems;
  std::pmr::vector<PassDispatchRange> preDispatchRangesByPass;
  std::pmr::vector<PassDrawRange> drawRangesByPass;
  std::pmr::vector<BufferHandle> resolvedPreDispatchDependencyBuffers;
  std::pmr::vector<DispatchDependencyBufferRange> preDispatchDependencyRanges;
  std::pmr::vector<UnresolvedPreDispatchDependencyBufferBinding>
      unresolvedPreDispatchDependencyBufferBindings;
  std::pmr::vector<UnresolvedDrawBufferBinding> unresolvedDrawBufferBindings;

  explicit RenderGraphCompileResult(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : orderedPasses(ensureMemory(memory)),
        orderedPassIndices(ensureMemory(memory)),
        passDebugNames(ensureMemory(memory)), edges(ensureMemory(memory)),
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
        ownedPreDispatches(ensureMemory(memory)),
        ownedDrawItems(ensureMemory(memory)),
        preDispatchRangesByPass(ensureMemory(memory)),
        drawRangesByPass(ensureMemory(memory)),
        resolvedPreDispatchDependencyBuffers(ensureMemory(memory)),
        preDispatchDependencyRanges(ensureMemory(memory)),
        unresolvedPreDispatchDependencyBufferBindings(ensureMemory(memory)),
        unresolvedDrawBufferBindings(ensureMemory(memory)) {}

private:
  static std::pmr::memory_resource *ensureMemory(std::pmr::memory_resource *m) {
    return m != nullptr ? m : std::pmr::get_default_resource();
  }
};

class NURI_API RenderGraphBuilder {
public:
  explicit RenderGraphBuilder(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());

  void beginFrame(uint64_t frameIndex);
  [[nodiscard]] Result<RenderGraphTextureId, std::string>
  importTexture(TextureHandle texture, std::string_view debugName = {});
  [[nodiscard]] Result<RenderGraphBufferId, std::string>
  importBuffer(BufferHandle buffer, std::string_view debugName = {});
  [[nodiscard]] Result<RenderGraphTextureId, std::string>
  createTransientTexture(const TextureDesc &desc,
                         std::string_view debugName = {});
  [[nodiscard]] Result<RenderGraphBufferId, std::string>
  createTransientBuffer(const BufferDesc &desc,
                        std::string_view debugName = {});
  [[nodiscard]] Result<bool, std::string>
  addTextureAccess(RenderGraphPassId pass, RenderGraphTextureId texture,
                   RenderGraphAccessMode mode);
  [[nodiscard]] Result<bool, std::string>
  addBufferAccess(RenderGraphPassId pass, RenderGraphBufferId buffer,
                  RenderGraphAccessMode mode);
  [[nodiscard]] Result<bool, std::string>
  addTextureRead(RenderGraphPassId pass, RenderGraphTextureId texture);
  [[nodiscard]] Result<bool, std::string>
  addTextureWrite(RenderGraphPassId pass, RenderGraphTextureId texture);
  [[nodiscard]] Result<bool, std::string>
  addBufferRead(RenderGraphPassId pass, RenderGraphBufferId buffer);
  [[nodiscard]] Result<bool, std::string>
  addBufferWrite(RenderGraphPassId pass, RenderGraphBufferId buffer);
  [[nodiscard]] Result<RenderGraphPassId, std::string>
  addGraphicsPass(const RenderGraphGraphicsPassDesc &desc);
  [[nodiscard]] Result<bool, std::string>
  bindPassColorTexture(RenderGraphPassId pass, RenderGraphTextureId texture);
  [[nodiscard]] Result<bool, std::string>
  bindPassDepthTexture(RenderGraphPassId pass, RenderGraphTextureId texture);
  [[nodiscard]] Result<bool, std::string> bindPassDependencyBuffer(
      RenderGraphPassId pass, uint32_t dependencyIndex,
      RenderGraphBufferId buffer,
      RenderGraphAccessMode mode = RenderGraphAccessMode::Read |
                                   RenderGraphAccessMode::Write);
  [[nodiscard]] Result<bool, std::string> bindPreDispatchDependencyBuffer(
      RenderGraphPassId pass, uint32_t preDispatchIndex,
      uint32_t dependencyIndex, RenderGraphBufferId buffer,
      RenderGraphAccessMode mode = RenderGraphAccessMode::Read |
                                   RenderGraphAccessMode::Write);
  [[nodiscard]] Result<bool, std::string>
  bindDrawBuffer(RenderGraphPassId pass, uint32_t drawIndex,
                 RenderGraphCompileResult::DrawBufferBindingTarget target,
                 RenderGraphBufferId buffer,
                 RenderGraphAccessMode mode = RenderGraphAccessMode::Read);
  [[nodiscard]] Result<bool, std::string>
  addDependency(RenderGraphPassId before, RenderGraphPassId after);
  [[nodiscard]] Result<bool, std::string>
  markPassSideEffect(RenderGraphPassId pass);
  [[nodiscard]] Result<bool, std::string>
  markTextureAsFrameOutput(RenderGraphTextureId texture);
  void setInferredSideEffectSuppression(bool enabled) noexcept {
    suppressInferredSideEffectsWhenExplicitOutputs_ = enabled;
  }
  [[nodiscard]] Result<RenderGraphCompileResult, std::string> compile() const;
  [[nodiscard]] size_t passCount() const noexcept { return passes_.size(); }

private:
  struct DependencyEdge {
    uint32_t before = UINT32_MAX;
    uint32_t after = UINT32_MAX;
  };

  enum class AccessResourceKind : uint8_t {
    Texture = 0,
    Buffer = 1,
  };

  struct PassResourceAccess {
    uint32_t passIndex = UINT32_MAX;
    AccessResourceKind resourceKind = AccessResourceKind::Texture;
    uint32_t resourceIndex = UINT32_MAX;
    RenderGraphAccessMode mode = RenderGraphAccessMode::None;
    bool inferred = false;
  };

  struct SideEffectPassMark {
    uint32_t passIndex = UINT32_MAX;
    bool inferred = false;
  };

  struct TextureResource {
    bool imported = false;
    TextureHandle importedHandle{};
    TextureDesc transientDesc{};
    std::pmr::string debugName;

    explicit TextureResource(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : debugName(memory) {}
  };

  struct BufferResource {
    bool imported = false;
    BufferHandle importedHandle{};
    BufferDesc transientDesc{};
    std::pmr::string debugName;

    explicit BufferResource(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : debugName(memory) {}
  };

  [[nodiscard]] bool isValidPassIndex(uint32_t passIndex) const {
    return passIndex < passes_.size();
  }
  [[nodiscard]] bool isValidTextureIndex(uint32_t textureIndex) const {
    return textureIndex < textures_.size();
  }
  [[nodiscard]] bool isValidBufferIndex(uint32_t bufferIndex) const {
    return bufferIndex < buffers_.size();
  }
  [[nodiscard]] Result<bool, std::string>
  addTextureAccessInternal(RenderGraphPassId pass, RenderGraphTextureId texture,
                           RenderGraphAccessMode mode, bool inferred);
  [[nodiscard]] Result<bool, std::string>
  addBufferAccessInternal(RenderGraphPassId pass, RenderGraphBufferId buffer,
                          RenderGraphAccessMode mode, bool inferred);
  [[nodiscard]] Result<bool, std::string>
  markPassSideEffectInternal(RenderGraphPassId pass, bool inferred);
  [[nodiscard]] Result<RenderGraphPassId, std::string>
  addPassRecord(const RenderPass &pass, std::string_view debugName);

  std::pmr::memory_resource *memory_ = nullptr;
  uint64_t frameIndex_ = 0;
  std::pmr::vector<TextureResource> textures_;
  std::pmr::vector<BufferResource> buffers_;
  std::pmr::vector<RenderPass> passes_;
  std::pmr::vector<std::pmr::string> passDebugNames_;
  std::pmr::vector<uint32_t> passColorTextureBindings_;
  std::pmr::vector<uint32_t> passDepthTextureBindings_;
  std::pmr::vector<uint32_t> passDependencyBufferBindingOffsets_;
  std::pmr::vector<uint32_t> passDependencyBufferBindingCounts_;
  std::pmr::vector<uint32_t> passDependencyBufferBindingResourceIndices_;
  std::pmr::vector<uint32_t> passPreDispatchBindingOffsets_;
  std::pmr::vector<uint32_t> passPreDispatchBindingCounts_;
  std::pmr::vector<uint32_t> preDispatchDependencyBindingOffsets_;
  std::pmr::vector<uint32_t> preDispatchDependencyBindingCounts_;
  std::pmr::vector<uint32_t> preDispatchDependencyBindingResourceIndices_;
  std::pmr::vector<uint32_t> passDrawBindingOffsets_;
  std::pmr::vector<uint32_t> passDrawBindingCounts_;
  std::pmr::vector<uint32_t> drawVertexBindingResourceIndices_;
  std::pmr::vector<uint32_t> drawIndexBindingResourceIndices_;
  std::pmr::vector<uint32_t> drawIndirectBindingResourceIndices_;
  std::pmr::vector<uint32_t> drawIndirectCountBindingResourceIndices_;
  PmrHashMap<uint64_t, uint32_t> importedTextureIndicesByHandle_;
  PmrHashMap<uint64_t, uint32_t> importedBufferIndicesByHandle_;
  PmrHashMap<uint64_t, uint32_t> explicitTextureAccessIndicesByPassResource_;
  PmrHashMap<uint64_t, uint32_t> inferredTextureAccessIndicesByPassResource_;
  PmrHashMap<uint64_t, uint32_t> explicitBufferAccessIndicesByPassResource_;
  PmrHashMap<uint64_t, uint32_t> inferredBufferAccessIndicesByPassResource_;
  PmrHashSet<uint64_t> dependencyEdgeKeys_;
  std::pmr::vector<DependencyEdge> dependencies_;
  std::pmr::vector<PassResourceAccess> passResourceAccesses_;
  PmrHashSet<uint32_t> frameOutputTextureSet_;
  std::pmr::vector<uint32_t> frameOutputTextureIndices_;
  PmrHashMap<uint32_t, uint32_t> sideEffectMarkIndicesByPass_;
  std::pmr::vector<SideEffectPassMark> sideEffectPassMarks_;
  bool suppressInferredSideEffectsWhenExplicitOutputs_ = false;
};

class NURI_API RenderGraphExecutor {
public:
  explicit RenderGraphExecutor(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  [[nodiscard]] Result<bool, std::string>
  execute(GPUDevice &gpu, const RenderGraphCompileResult &compiled);

private:
  struct PendingFrameResources {
    uint64_t retireAfterFrame = 0;
    std::pmr::vector<TextureHandle> textures;
    std::pmr::vector<BufferHandle> buffers;
    std::pmr::vector<TextureDesc> textureDescs;
    std::pmr::vector<BufferDesc> bufferDescs;

    explicit PendingFrameResources(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : textures(memory), buffers(memory), textureDescs(memory),
          bufferDescs(memory) {}
  };

  struct ReusableTextureResource {
    TextureHandle handle{};
    TextureDesc desc{};
  };

  struct ReusableBufferResource {
    BufferHandle handle{};
    BufferDesc desc{};
  };

  void collectRetiredResources(GPUDevice &gpu, uint64_t completedFrameIndex);

  std::pmr::memory_resource *memory_ = nullptr;
  std::pmr::vector<PendingFrameResources> pendingFrames_;
  std::pmr::vector<ReusableTextureResource> reusableTextures_;
  std::pmr::vector<ReusableBufferResource> reusableBuffers_;
};

} // namespace nuri
