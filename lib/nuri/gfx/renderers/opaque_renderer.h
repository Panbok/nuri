#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/scene/render_scene.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

using OpaqueRendererConfig = RuntimeOpaqueShaderConfig;
class ResourceManager;

class NURI_API OpaqueRenderer {
public:
  explicit OpaqueRenderer(
      GPUDevice &gpu, OpaqueRendererConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~OpaqueRenderer();

  OpaqueRenderer(const OpaqueRenderer &) = delete;
  OpaqueRenderer &operator=(const OpaqueRenderer &) = delete;
  OpaqueRenderer(OpaqueRenderer &&) = delete;
  OpaqueRenderer &operator=(OpaqueRenderer &&) = delete;

  void onAttach();
  void onDetach();
  void onResize(uint32_t width, uint32_t height);
  Result<bool, std::string> prepareOpaqueGraphPasses(RenderFrameContext &frame);
  [[nodiscard]] bool hasPreparedOpaqueMainPasses() const noexcept;
  [[nodiscard]] bool hasPreparedOpaquePickPasses() const noexcept;
  Result<bool, std::string> appendOpaqueMainPasses(RenderFrameContext &frame,
                                                   RenderGraphBuilder &graph);
  Result<bool, std::string> appendOpaquePickPasses(RenderFrameContext &frame,
                                                   RenderGraphBuilder &graph);

private:
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 224,
                "OpaqueRenderer::FrameData must match shader FrameDataBuffer "
                "layout");

  struct PushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t instanceRemapAddress = 0;
    uint64_t materialBufferAddress = 0;
    uint64_t instanceCentersPhaseAddress = 0;
    uint64_t instanceBaseMatricesAddress = 0;
    uint32_t instanceCount = 0;
    uint32_t materialIndex = 0;
    float timeSeconds = 0.0f;
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 6.0f;
    uint32_t debugVisualizationMode = 0;
  };
  static_assert(
      sizeof(PushConstants) <= 128,
      "OpaqueRenderer::PushConstants exceeds Vulkan minimum guarantee");

  // These templates hold non-owning borrowed pointers. Callers must ensure the
  // referenced scene/model data outlives the cached template usage.
  struct RenderableTemplate {
    const Renderable *renderable = nullptr;
    const Model *model = nullptr;
  };

  // These templates hold non-owning borrowed pointers. Callers must ensure the
  // referenced renderable/model/submesh data outlives the cached template
  // usage.
  struct MeshDrawTemplate {
    const Renderable *renderable = nullptr;
    const Submesh *submesh = nullptr;
    uint32_t submeshIndex = 0;
    uint32_t instanceIndex = 0;
    GeometryAllocationHandle geometryHandle{};
    BufferHandle indexBuffer{};
    uint64_t indexBufferOffset = 0;
    uint64_t vertexBufferAddress = 0;
    uint32_t materialIndex = kInvalidMaterialIndex;
    bool doubleSided = false;
  };

  struct TessCandidate {
    float distanceSq = 0.0f;
    uint32_t instanceId = 0;
  };

  struct DynamicBufferSlot {
    std::unique_ptr<Buffer> buffer;
    size_t capacityBytes = 0;
  };

  struct SingleInstanceBatchEntry {
    DrawItem draw{};
    uint64_t vertexBufferAddress = 0;
    uint32_t materialIndex = kInvalidMaterialIndex;
    size_t instanceCount = 0;
  };

  struct SingleInstanceBatchCache {
    bool valid = false;
    uint32_t requestedLod = 0;
    bool tessPipelineEnabled = false;
    RenderPipelineHandle basePipeline{};
    RenderPipelineHandle doubleSidedBasePipeline{};
    RenderPipelineHandle tessPipeline{};
    RenderPipelineHandle doubleSidedTessPipeline{};
    uint64_t templateRevision = 0;
    size_t remapCount = 0;
    std::pmr::vector<SingleInstanceBatchEntry> batches;

    explicit SingleInstanceBatchCache(std::pmr::memory_resource *memory)
        : batches(memory) {}
  };

  struct PreparedGraphPass {
    struct DependencyBufferBinding {
      uint32_t dependencyIndex = UINT32_MAX;
      BufferHandle buffer{};
      RenderGraphAccessMode mode =
          RenderGraphAccessMode::Read | RenderGraphAccessMode::Write;
    };

    struct DependencyTextureBinding {
      TextureHandle texture{};
      RenderGraphAccessMode mode = RenderGraphAccessMode::Read;
    };

    struct PreDispatchDependencyBinding {
      uint32_t preDispatchIndex = UINT32_MAX;
      uint32_t dependencyIndex = UINT32_MAX;
      BufferHandle buffer{};
      RenderGraphAccessMode mode =
          RenderGraphAccessMode::Read | RenderGraphAccessMode::Write;
    };

    struct DrawBufferBinding {
      uint32_t drawIndex = UINT32_MAX;
      RenderGraphDrawBufferBindingTarget target =
          RenderGraphDrawBufferBindingTarget::Vertex;
      BufferHandle buffer{};
      RenderGraphAccessMode mode = RenderGraphAccessMode::Read;
    };

    RenderGraphGraphicsPassDesc desc{};
    TextureHandle colorTextureHandle{};
    TextureHandle depthTextureHandle{};
    std::pmr::vector<DependencyBufferBinding> dependencyBufferBindings;
    std::pmr::vector<DependencyTextureBinding> dependencyTextureBindings;
    std::pmr::vector<PreDispatchDependencyBinding>
        preDispatchDependencyBindings;
    std::pmr::vector<DrawBufferBinding> drawBufferBindings;
    bool hasDraws = false;
    bool hasPreDispatch = false;
    bool hasIndirectDraws = false;
    bool isMainPass = false;
    bool isPickPass = false;

    explicit PreparedGraphPass(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : dependencyBufferBindings(memory), dependencyTextureBindings(memory),
          preDispatchDependencyBindings(memory), drawBufferBindings(memory) {}
  };

  struct IndirectPackCache {
    bool valid = false;
    uint64_t drawSignature = std::numeric_limits<uint64_t>::max();
    size_t requiredBytes = 0;
  };

  struct StaticBatchCache {
    bool valid = false;
    bool enableMeshLod = false;
    int32_t forcedMeshLod = -1;
    uint64_t generation = 1;
    uint64_t remapSignature = std::numeric_limits<uint64_t>::max();
    uint64_t indirectDrawSignature = std::numeric_limits<uint64_t>::max();
    std::pmr::vector<DrawItem> draws;
    std::pmr::vector<PushConstants> pushConstantsTemplates;
    std::pmr::vector<uint32_t> remap;

    explicit StaticBatchCache(std::pmr::memory_resource *memory)
        : draws(memory), pushConstantsTemplates(memory), remap(memory) {}
  };

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> recreateDepthTexture();
  Result<bool, std::string> recreatePickTexture();
  Result<bool, std::string>
  ensureCentersPhaseBufferCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceBaseMatricesBufferCapacity(size_t requiredBytes);
  Result<bool, std::string> ensureMaterialBufferCapacity(size_t requiredBytes);
  Result<bool, std::string> ensureRingBufferCount(uint32_t requiredCount);
  Result<bool, std::string>
  ensureInstanceMatricesRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceRemapRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureIndirectCommandRingCapacity(size_t requiredBytes);
  [[nodiscard]] uint32_t
  resolveSingleInstanceRequestedLod(const RenderSettings &settings,
                                    uint32_t forcedLod) const;
  [[nodiscard]] bool shouldEnableSingleInstanceTessPipeline(
      bool tessellationRequested, uint32_t requestedLod,
      const glm::vec3 &cameraPosition, float tessFarDistanceSq) const;
  Result<bool, std::string>
  ensureSingleInstanceBatchCache(uint32_t requestedLod,
                                 bool tessPipelineEnabled,
                                 const DrawItem &baseDraw);
  [[nodiscard]] size_t singleInstanceCacheIndex(uint32_t requestedLod,
                                                bool tessPipelineEnabled) const;
  Result<bool, std::string> buildIndirectDraws(uint32_t frameSlot,
                                               size_t remapCount,
                                               uint64_t drawSignature,
                                               bool drawSignatureValid);
  [[nodiscard]] uint64_t computeIndirectDrawSignature(size_t remapCount) const;
  [[nodiscard]] bool canReuseIndirectPack(uint64_t drawSignature) const;
  Result<bool, std::string> rebuildIndirectPack(uint32_t frameSlot,
                                                size_t remapCount,
                                                uint64_t drawSignature);
  Result<bool, std::string> refreshCachedIndirectPack(uint32_t frameSlot,
                                                      uint64_t drawSignature);
  Result<bool, std::string> rebuildSceneCache(const RenderScene &scene,
                                              const ResourceManager &resources,
                                              uint32_t materialCount);
  Result<bool, std::string>
  rebuildMaterialTextureAccessCache(const RenderScene &scene,
                                    const ResourceManager &resources);
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPipelines();
  Result<bool, std::string>
  buildOpaquePasses(RenderFrameContext &frame,
                    std::pmr::vector<PreparedGraphPass> &out);
  Result<bool, std::string>
  appendPreparedGraphPass(RenderFrameContext &frame, RenderGraphBuilder &graph,
                          const PreparedGraphPass &pass, uint32_t safeWidth,
                          uint32_t safeHeight);
  void cachePreparedGraphPassMetadata(PreparedGraphPass &pass) const;
  [[nodiscard]] bool
  shouldPublishSceneDepthGraphTexture(const RenderFrameContext &frame) const;
  [[nodiscard]] RenderPipelineHandle selectMeshPipeline(bool doubleSided,
                                                        bool tessellated) const;
  [[nodiscard]] RenderPipelineHandle
  selectPickPipeline(RenderPipelineHandle sourcePipeline) const;
  [[nodiscard]] bool isDoubleSidedPipeline(RenderPipelineHandle handle) const;
  [[nodiscard]] bool isTessPipeline(RenderPipelineHandle handle) const;
  Result<bool, std::string> ensureWireframePipeline();
  Result<bool, std::string> ensureTessWireframePipeline();
  Result<bool, std::string> ensureGsOverlayPipeline();
  Result<bool, std::string> ensureGsTessOverlayPipeline();
  void resetOverlayPipelineState();
  void invalidateAutoLodCache();
  void updateFastAutoLodCache(
      const Submesh *submesh, const glm::vec3 &cameraPosition,
      const std::array<float, 3> &sortedLodThresholds,
      const std::array<size_t, Submesh::kMaxLodCount> &bucketCounts,
      size_t remapCount, size_t instanceCount, uint64_t frameIndex);
  void invalidateStaticBatchCache();
  void invalidateSingleInstanceBatchCache();
  void invalidateIndirectPackCache();
  void resetPickState();
  void destroyMeshPipelineState();
  void resetMeshPipelineState();
  void destroyDepthTexture();
  void destroyPickTexture();
  void destroyBuffers();

  GPUDevice &gpu_;
  OpaqueRendererConfig config_{};
  std::unique_ptr<Shader> meshShader_;
  std::unique_ptr<Shader> meshTessShader_;
  std::unique_ptr<Shader> meshDebugOverlayShader_;
  std::unique_ptr<Shader> meshPickShader_;
  std::unique_ptr<Shader> computeShader_;
  std::unique_ptr<Pipeline> meshPipeline_;
  std::unique_ptr<Pipeline> computePipeline_;
  std::unique_ptr<Buffer> instanceCentersPhaseBuffer_;
  std::unique_ptr<Buffer> instanceBaseMatricesBuffer_;
  std::unique_ptr<Buffer> materialBuffer_;
  std::pmr::vector<DynamicBufferSlot> instanceMatricesRing_;
  std::pmr::vector<DynamicBufferSlot> instanceRemapRing_;
  std::pmr::vector<DynamicBufferSlot> indirectCommandRing_;
  TextureHandle depthTexture_{};
  TextureHandle pickIdTexture_{};

  ShaderHandle meshVertexShader_{};
  ShaderHandle meshTessVertexShader_{};
  ShaderHandle meshTessControlShader_{};
  ShaderHandle meshTessEvalShader_{};
  ShaderHandle meshFragmentShader_{};
  ShaderHandle meshDebugOverlayGeometryShader_{};
  ShaderHandle meshDebugOverlayFragmentShader_{};
  ShaderHandle meshPickFragmentShader_{};
  ShaderHandle computeShaderHandle_{};
  RenderPipelineHandle meshFillPipelineHandle_{};
  RenderPipelineHandle meshDoubleSidedFillPipelineHandle_{};
  RenderPipelineHandle meshTessPipelineHandle_{};
  RenderPipelineHandle meshDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshGsOverlayPipelineHandle_{};
  RenderPipelineHandle meshGsTessOverlayPipelineHandle_{};
  RenderPipelineHandle meshWireframePipelineHandle_{};
  RenderPipelineHandle meshTessWireframePipelineHandle_{};
  RenderPipelineHandle meshPickPipelineHandle_{};
  RenderPipelineHandle meshPickDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshPickTessPipelineHandle_{};
  RenderPipelineHandle meshPickDoubleSidedTessPipelineHandle_{};
  ComputePipelineHandle computePipelineHandle_{};

  size_t instanceCentersPhaseBufferCapacityBytes_ = 0;
  size_t instanceBaseMatricesBufferCapacityBytes_ = 0;
  size_t materialBufferCapacityBytes_ = 0;
  bool initialized_ = false;
  bool tessellationUnsupported_ = false;
  bool wireframePipelineInitialized_ = false;
  bool wireframePipelineUnsupported_ = false;
  bool tessWireframePipelineInitialized_ = false;
  bool tessWireframePipelineUnsupported_ = false;
  bool gsOverlayPipelineInitialized_ = false;
  bool gsOverlayPipelineUnsupported_ = false;
  bool gsTessOverlayPipelineInitialized_ = false;
  bool gsTessOverlayPipelineUnsupported_ = false;
  bool loggedWireframeFallbackUnsupported_ = false;
  bool loggedTessWireframeFallbackUnsupported_ = false;
  bool loggedGsOverlayUnsupported_ = false;
  bool loggedGsTessOverlayUnsupported_ = false;
  bool loggedMaterialFallbackWarning_ = false;
  bool loggedBlendMaterialUnsupportedWarning_ = false;

  const RenderScene *cachedScene_ = nullptr;
  uint64_t cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedGeometryMutationVersion_ =
      std::numeric_limits<uint64_t>::max();
  bool instanceStaticBuffersDirty_ = true;
  bool uniformSingleSubmeshPath_ = false;

  struct AutoLodCache {
    bool valid = false;
    glm::vec3 cameraPos{0.0f};
    std::array<float, 3> thresholds = {0.0f, 0.0f, 0.0f};
    std::array<size_t, Submesh::kMaxLodCount> bucketCounts{};
    size_t remapCount = 0;
    size_t instanceCount = 0;
    const Submesh *submesh = nullptr;
    uint64_t frameIndex = std::numeric_limits<uint64_t>::max();
  };
  static constexpr size_t kSingleInstanceCacheVariantCount =
      static_cast<size_t>(Submesh::kMaxLodCount) * 2u;
  AutoLodCache autoLodCache_{};
  std::pmr::vector<SingleInstanceBatchCache> singleInstanceBatchCaches_;
  uint64_t singleInstanceTemplateRevision_ = 1;
  IndirectPackCache indirectPackCache_{};
  StaticBatchCache staticBatchCache_;

  std::pmr::vector<RenderableTemplate> renderableTemplates_;
  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<size_t> indirectSourceDrawIndices_;
  std::pmr::vector<uint64_t> instanceMatricesUploadVersions_;
  std::pmr::vector<uint64_t> indirectUploadSignatures_;
  std::pmr::vector<uint64_t> remapUploadSignatures_;
  std::pmr::vector<uint32_t> templateBatchIndices_;
  std::pmr::vector<size_t> batchWriteOffsets_;
  std::pmr::vector<glm::vec4> instanceCentersPhase_;
  std::pmr::vector<glm::mat4> instanceBaseMatrices_;
  std::pmr::vector<InstanceData> instanceMatricesCpuCache_;
  std::pmr::vector<glm::vec4> instanceLodCentersInvRadiusSq_;
  std::pmr::vector<MaterialGpuData> materialGpuDataCache_;
  std::pmr::vector<TextureHandle> materialTextureAccessHandles_;
  std::pmr::vector<uint32_t> instanceAutoLodLevels_;
  std::pmr::vector<uint8_t> instanceTessSelection_;
  std::pmr::vector<TessCandidate> tessCandidates_;
  std::pmr::vector<uint32_t> instanceRemap_;
  std::pmr::vector<PushConstants> drawPushConstants_;
  std::pmr::vector<DrawItem> drawItems_;
  std::pmr::vector<DrawItem> indirectDrawItems_;
  std::pmr::vector<std::byte> indirectCommandUploadBytes_;
  std::pmr::vector<DrawItem> overlayDrawItems_;
  std::pmr::vector<DrawItem> pickDrawItems_;
  std::pmr::vector<DrawItem> passDrawItems_;
  std::pmr::vector<ComputeDispatchItem> preDispatches_;
  std::pmr::vector<BufferHandle> passDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode> passDependencyBufferAccessModes_;
  std::pmr::vector<BufferHandle> dispatchDependencyBuffers_;
  std::pmr::vector<TextureHandle> passDependencyTextures_;
  std::pmr::vector<PreparedGraphPass> preparedGraphPasses_;
  PushConstants computePushConstants_{};
  DrawItem baseMeshFillDraw_{};
  DrawItem baseMeshWireframeDraw_{};
  uint64_t cachedRemapSignature_ = std::numeric_limits<uint64_t>::max();
  bool cachedRemapSignatureValid_ = false;
  uint64_t boundStaticBatchGeneration_ = 0;
  uint64_t statsLogFrameCounter_ = 0;
  std::optional<OpaquePickRequest> pendingPickRequest_{};

  struct InFlightPickReadback {
    OpaquePickRequest request{};
    uint64_t submissionFrame = 0;
  };
  std::optional<InFlightPickReadback> inFlightPickReadback_{};
};

} // namespace nuri
