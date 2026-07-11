#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/gfx/shader.h"
#include "nuri/gfx/visibility/visibility.h"
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
#include <span>
#include <unordered_map>
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
  void publishFrameData(RenderFrameContext &frame);
  Result<bool, std::string> prepareOpaqueGraphPasses(RenderFrameContext &frame);
  void commitSubmittedFrame(uint64_t frameIndex) noexcept;
  void abandonPreparedFrame(uint64_t frameIndex) noexcept;
  [[nodiscard]] bool hasPreparedOpaqueMainPasses() const noexcept;
  [[nodiscard]] bool hasPreparedOpaquePrepassPasses() const noexcept;
  [[nodiscard]] bool hasPreparedOpaqueMainLightingPasses() const noexcept;
  [[nodiscard]] bool hasPreparedOpaquePickPasses() const noexcept;
  Result<bool, std::string> appendOpaqueMainPasses(RenderFrameContext &frame,
                                                   RenderGraphBuilder &graph);
  Result<bool, std::string>
  appendOpaquePrepassPasses(RenderFrameContext &frame,
                            RenderGraphBuilder &graph);
  Result<bool, std::string>
  appendOpaqueMainLightingPasses(RenderFrameContext &frame,
                                 RenderGraphBuilder &graph);
  Result<bool, std::string> appendOpaquePickPasses(RenderFrameContext &frame,
                                                   RenderGraphBuilder &graph);

private:
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 464,
                "OpaqueRenderer::FrameData must match shader FrameDataBuffer "
                "layout");

  struct PushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t previousInstanceMatricesAddress = 0;
    uint64_t instanceRemapAddress = 0;
    uint64_t instanceCentersPhaseAddress = 0;
    uint64_t instanceBaseMatricesAddress = 0;
    uint64_t velocityInstanceFlagsAddress = 0;
    uint64_t velocityFrameDataAddress = 0;
    uint32_t instanceCount = 0;
    uint32_t materialIndex = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    float timeSeconds = 0.0f;
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 6.0f;
    uint32_t debugVisualizationMode = 0;
    uint32_t shadowCascadeIndex = 0;
  };
  static_assert(sizeof(PushConstants) == 128,
                "OpaqueRenderer::PushConstants must match shader layout");
  static_assert(offsetof(PushConstants, instanceRemapAddress) == 40u);
  static_assert(offsetof(PushConstants, instanceCentersPhaseAddress) == 48u);
  static_assert(offsetof(PushConstants, instanceBaseMatricesAddress) == 56u);
  static_assert(offsetof(PushConstants, instanceCount) == 80u);
  static_assert(offsetof(PushConstants, shadowCascadeIndex) == 120u);

  struct MeshletPushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t instanceRemapAddress = 0;
    uint64_t instanceLodBoundsAddress = 0;
    glm::vec4 lodThresholds{0.0f};
    uint64_t meshletBatchBufferAddress = 0;
    uint64_t visibilityCounterBufferAddress = 0;
    uint64_t previousInstanceMatricesAddress = 0;
    uint64_t velocityInstanceFlagsAddress = 0;
    uint64_t velocityFrameDataAddress = 0;
    uint32_t batchBase = 0;
    uint32_t candidateOffset = 0;
    uint32_t sourceFrameIndex = 0;
    uint32_t meshletCounterFlags = 0;
  };
  static_assert(sizeof(MeshletPushConstants) == 104,
                "OpaqueRenderer::MeshletPushConstants must match shader "
                "layout");
  static_assert(offsetof(MeshletPushConstants, lodThresholds) == 32u);
  static_assert(offsetof(MeshletPushConstants, meshletBatchBufferAddress) ==
                48u);
  static_assert(offsetof(MeshletPushConstants,
                         visibilityCounterBufferAddress) == 56u);
  static_assert(offsetof(MeshletPushConstants,
                         previousInstanceMatricesAddress) == 64u);
  static_assert(offsetof(MeshletPushConstants, velocityInstanceFlagsAddress) ==
                72u);
  static_assert(offsetof(MeshletPushConstants, velocityFrameDataAddress) ==
                80u);
  static_assert(offsetof(MeshletPushConstants, batchBase) == 88u);

  struct alignas(16) MeshletBatchGpuData {
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint64_t meshletBufferAddress = 0;
    uint64_t meshletVertexIndexBufferAddress = 0;
    uint64_t meshletPrimitiveIndexBufferAddress = 0;
    uint64_t meshletLodRangeBufferAddress = 0;
    glm::uvec4 draw{0u};
    glm::uvec4 mesh{0u};
    glm::uvec4 flags{0u};
  };
  static_assert(sizeof(MeshletBatchGpuData) == 96,
                "OpaqueRenderer::MeshletBatchGpuData layout changed");
  static_assert(offsetof(MeshletBatchGpuData, draw) == 48u);
  static_assert(offsetof(MeshletBatchGpuData, mesh) == 64u);
  static_assert(offsetof(MeshletBatchGpuData, flags) == 80u);

  struct alignas(16) ShadowSdsmReducePushConstants {
    uint64_t resultBufferAddress = 0u;
    uint32_t sourceTexId = kInvalidShadowBindlessIndex;
    uint32_t sourceFrameIndex = 0u;
  };
  static_assert(sizeof(ShadowSdsmReducePushConstants) == 16u,
                "OpaqueRenderer::ShadowSdsmReducePushConstants layout "
                "changed");

  struct alignas(16) ShadowSdsmHistogramReducePushConstants {
    uint64_t resultBufferAddress = 0u;
    uint64_t resultBufferAddressPadding = 0u;
    glm::uvec4 sourceParams{0u};
    glm::uvec4 histogramParams{0u};
    glm::vec4 cameraParams{0.0f};
    glm::vec4 trimParams{0.0f};
  };
  static_assert(sizeof(ShadowSdsmHistogramReducePushConstants) == 80u,
                "OpaqueRenderer::ShadowSdsmHistogramReducePushConstants "
                "layout changed");
  static_assert(offsetof(ShadowSdsmHistogramReducePushConstants,
                         sourceParams) == 16u);
  static_assert(offsetof(ShadowSdsmHistogramReducePushConstants,
                         histogramParams) == 32u);
  static_assert(offsetof(ShadowSdsmHistogramReducePushConstants,
                         cameraParams) == 48u);
  static_assert(offsetof(ShadowSdsmHistogramReducePushConstants, trimParams) ==
                64u);

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
    BufferHandle baseVertexBuffer{};
    BufferHandle vertexBuffer{};
    BufferHandle baseVertexDecodeBuffer{};
    BufferHandle vertexDecodeBuffer{};
    uint64_t baseVertexBufferAddress = 0;
    uint64_t baseVertexDecodeBufferAddress = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t basePackedVertexFormat = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = kInvalidMaterialIndex;
    const Model::ModelMeshletGpuView *meshletView = nullptr;
    bool doubleSided = false;
    bool alphaMasked = false;
    bool materialNormalRequired = false;
  };

  struct TessCandidate {
    float distanceSq = 0.0f;
    uint32_t instanceId = 0;
  };

  struct MeshletBatchInfo {
    const Model::ModelMeshletGpuView *view = nullptr;
    BufferHandle vertexDecodeBuffer{};
    uint32_t meshletOffset = 0;
    uint32_t meshletCount = 0;
    uint32_t submeshIndex = 0;
    uint32_t meshletMaxCount = 0;
    uint32_t vertexOffset = 0;
    bool doubleSided = false;
    bool alphaMasked = false;
    bool materialNormalRequired = false;
  };

  struct BatchEntry {
    DrawItem draw{};
    BufferHandle vertexBuffer{};
    BufferHandle vertexDecodeBuffer{};
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = kInvalidMaterialIndex;
    const Model::ModelMeshletGpuView *meshletView = nullptr;
    uint32_t meshletOffset = 0;
    uint32_t meshletCount = 0;
    uint32_t submeshIndex = 0;
    uint32_t meshletMaxCount = 0;
    uint32_t vertexOffset = 0;
    bool doubleSided = false;
    size_t instanceCount = 0;
    size_t firstInstance = 0;
    bool alphaMasked = false;
    bool materialNormalRequired = false;
  };

  struct MeshletDispatchDependencyBuffers {
    std::pmr::vector<BufferHandle> buffers;

    explicit MeshletDispatchDependencyBuffers(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : buffers(memory) {}

    [[nodiscard]] std::span<const BufferHandle> span() const noexcept {
      return std::span<const BufferHandle>(buffers.data(), buffers.size());
    }
  };

  struct DynamicBufferSlot {
    std::unique_ptr<Buffer> buffer;
    size_t capacityBytes = 0;
  };

  struct alignas(16) VelocityFrameGpuData {
    glm::mat4 currentViewProjNoJitter{1.0f};
    glm::mat4 previousViewProjNoJitter{1.0f};
    glm::uvec4 instanceFlagsMode{0u};
    uint64_t previousGeometryAddress = 0u;
    uint64_t previousGeometryAddressPadding = 0u;
    glm::uvec4 previousGeometryInfo{0u};
  };
  static_assert(sizeof(VelocityFrameGpuData) == sizeof(glm::mat4) * 2u +
                                                    sizeof(glm::uvec4) * 2u +
                                                    sizeof(uint64_t) * 2u,
                "OpaqueRenderer::VelocityFrameGpuData layout changed");
  static_assert(offsetof(VelocityFrameGpuData, previousGeometryAddress) ==
                144u);
  static_assert(offsetof(VelocityFrameGpuData, previousGeometryInfo) == 160u);

  struct alignas(16) VelocityRenderableGeometryGpuData {
    uint64_t previousVertexBufferAddress = 0u;
    uint64_t previousVertexBufferAddressPadding = 0u;
    glm::uvec4 metadata{0u};
  };
  static_assert(sizeof(VelocityRenderableGeometryGpuData) == 32u,
                "OpaqueRenderer::VelocityRenderableGeometryGpuData layout "
                "changed");
  static_assert(offsetof(VelocityRenderableGeometryGpuData, metadata) == 16u);

  enum class VelocityInstanceFlagsMode : uint32_t {
    Buffer = 0u,
    AllValid = 1u,
    AllInvalid = 2u,
  };

  struct alignas(16) DepthMotionVectorPushConstants {
    uint32_t depthTexId = 0u;
    uint32_t pointSamplerId = 0u;
    uint32_t currentJitterUvXBits = 0u;
    uint32_t currentJitterUvYBits = 0u;
    glm::mat4 previousFromCurrentJitteredClip{1.0f};
  };
  static_assert(sizeof(DepthMotionVectorPushConstants) <= 128u,
                "Depth motion vector push constants exceed Vulkan minimum");

  struct SingleInstanceBatchEntry {
    DrawItem draw{};
    BufferHandle vertexBuffer{};
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = kInvalidMaterialIndex;
    size_t instanceCount = 0;
    bool alphaMasked = false;
    bool materialNormalRequired = false;
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
    TextureHandle colorResolveTextureHandle{};
    TextureHandle depthTextureHandle{};
    TextureHandle depthResolveTextureHandle{};
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
    bool isDepthPrepass = false;
    bool isTransmissionVisibilityDepthPass = false;
    bool isNormalPrepass = false;
    bool isDepthPyramidPass = false;
    bool isVelocityPass = false;
    bool isEarlyVelocityPass = false;
    bool isReactiveMaskPass = false;
    bool isEarlyReactiveMaskPass = false;
    bool isVisibilityComputePass = false;
    uint32_t depthPyramidLevel = UINT32_MAX;

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
    bool meshletRequested = false;
    bool meshletDispatchCacheValid = false;
    bool enableMeshLod = false;
    int32_t forcedMeshLod = -1;
    uint64_t generation = 1;
    uint64_t remapSignature = std::numeric_limits<uint64_t>::max();
    uint64_t indirectDrawSignature = std::numeric_limits<uint64_t>::max();
    uint64_t drawBufferSignature = std::numeric_limits<uint64_t>::max();
    uint64_t meshletDispatchSignature = std::numeric_limits<uint64_t>::max();
    uint64_t meshletCandidateCount = 0;
    std::pmr::vector<DrawItem> draws;
    std::pmr::vector<PushConstants> pushConstantsTemplates;
    std::pmr::vector<uint8_t> alphaMasked;
    std::pmr::vector<MeshletBatchInfo> meshletBatchInfos;
    std::pmr::vector<MeshDispatchItem> meshletDispatches;
    std::pmr::vector<MeshletPushConstants> meshletPushConstantsTemplates;
    std::pmr::vector<MeshletDispatchDependencyBuffers>
        meshletDispatchDependencyBuffers;
    std::pmr::vector<MeshletBatchGpuData> meshletBatchGpuData;
    std::pmr::vector<uint32_t> remap;

    explicit StaticBatchCache(std::pmr::memory_resource *memory)
        : draws(memory), pushConstantsTemplates(memory), alphaMasked(memory),
          meshletBatchInfos(memory), meshletDispatches(memory),
          meshletPushConstantsTemplates(memory),
          meshletDispatchDependencyBuffers(memory), meshletBatchGpuData(memory),
          remap(memory) {}
  };

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> recreatePickTexture();
  Result<bool, std::string>
  ensureCentersPhaseBufferCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceLodBoundsBufferCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceBaseMatricesBufferCapacity(size_t requiredBytes);
  Result<bool, std::string> ensureRingBufferCount(uint32_t requiredCount);
  Result<bool, std::string>
  ensureInstanceMatricesRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensurePreviousInstanceMatricesRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureVelocityInstanceFlagsRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureVelocityFrameDataRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureVelocityGeometryRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureMeshletBatchRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureVisibilityGpuRingCapacity(size_t candidateBytes,
                                  size_t visibleIndexBytes);
  Result<bool, std::string>
  ensureVisibilityMeshletIndirectRingCapacity(size_t dispatchBytes,
                                              size_t commandBytes);
  Result<bool, std::string>
  ensureVisibilityMeshletIndirectCommandRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureVisibilityCounterRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureVisibilityVisibleIndexRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureDynamicRingCapacity(std::pmr::vector<DynamicBufferSlot> &ring,
                            size_t requiredBytes, size_t minimumBytes,
                            std::string_view debugNamePrefix,
                            Storage storage = Storage::Device);
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
                                              uint32_t materialCount,
                                              bool excludeTransmission);
  Result<bool, std::string>
  rebuildMaterialTextureAccessCache(const RenderScene &scene,
                                    const ResourceManager &resources,
                                    bool excludeTransmission);
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPipelines();
  Result<bool, std::string> ensureMeshletPipelineState();
  void readLatestVisibilityGpuReadback(RenderFrameContext &frame);
  Result<bool, std::string> appendGpuVisibilityMainPass(
      RenderFrameContext &frame, uint32_t frameSlot,
      std::span<const VisibilityCandidate> candidates,
      std::span<const VisibilityCandidateGpu> candidateGpuData,
      std::span<const uint32_t> candidateIndices,
      const VisibilityPassRequest &request,
      const VisibilityResolvedSettings &settings,
      bool candidateIndicesPreculledByCpu,
      std::pmr::vector<PreparedGraphPass> &out);
  Result<bool, std::string>
  buildOpaquePasses(RenderFrameContext &frame,
                    std::pmr::vector<PreparedGraphPass> &out);
  Result<bool, std::string>
  ensureDepthPyramidTextures(const RenderSettings &settings);
  [[nodiscard]] bool requiresDepthPyramid(const RenderSettings &settings) const;
  [[nodiscard]] bool
  shouldBuildTransmissionVisibilityDepth(const RenderFrameContext &frame,
                                         const RenderSettings &settings) const;
  Result<bool, std::string> appendPreparedGraphPass(
      RenderFrameContext &frame, RenderGraphBuilder &graph,
      const PreparedGraphPass &pass, uint32_t safeWidth, uint32_t safeHeight,
      std::span<const RenderGraphBufferId> preResolvedDrawBufferIds);
  Result<bool, std::string>
  ensurePreResolvedDrawBufferIds(RenderFrameContext &frame,
                                 RenderGraphBuilder &graph);
  [[nodiscard]] static bool
  isPreLightingPass(const PreparedGraphPass &pass) noexcept;
  void cachePreparedGraphPassMetadata(PreparedGraphPass &pass) const;
  [[nodiscard]] bool
  shouldPublishSceneDepthGraphTexture(const RenderFrameContext &frame) const;
  [[nodiscard]] RenderPipelineHandle selectMeshPipeline(bool doubleSided,
                                                        bool tessellated) const;
  [[nodiscard]] RenderPipelineHandle
  selectVelocityPipeline(RenderPipelineHandle sourcePipeline) const;
  [[nodiscard]] RenderPipelineHandle
  selectReactiveMaskPipeline(RenderPipelineHandle sourcePipeline) const;
  [[nodiscard]] RenderPipelineHandle
  selectDepthPipeline(RenderPipelineHandle sourcePipeline, bool alphaMasked,
                      bool msaa) const;
  [[nodiscard]] RenderPipelineHandle
  selectNormalPipeline(RenderPipelineHandle sourcePipeline) const;
  [[nodiscard]] RenderPipelineHandle
  selectPickPipeline(RenderPipelineHandle sourcePipeline) const;
  [[nodiscard]] RenderPipelineHandle
  selectShadowInspectPipeline(RenderPipelineHandle sourcePipeline) const;
  [[nodiscard]] RenderPipelineHandle
  selectMsaaScenePipeline(RenderPipelineHandle sourcePipeline,
                          bool alphaMasked) const;
  [[nodiscard]] bool isDoubleSidedPipeline(RenderPipelineHandle handle) const;
  [[nodiscard]] bool isTessPipeline(RenderPipelineHandle handle) const;
  Result<bool, std::string> ensureSceneDepthSampler();
  Result<bool, std::string> ensureWireframePipeline(bool requireMsaa = false);
  Result<bool, std::string>
  ensureTessWireframePipeline(bool requireMsaa = false);
  Result<bool, std::string> ensureGsOverlayPipeline(bool requireMsaa = false);
  Result<bool, std::string>
  ensureGsTessOverlayPipeline(bool requireMsaa = false);
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
  void stagePreviousTransforms(const RenderScene &scene, uint64_t frameIndex);
  void destroyMeshletPipelineState();
  void destroyMeshPipelineState();
  void resetMeshletPipelineState();
  void resetMeshPipelineState();
  void destroyDepthPyramidTextures();
  Result<bool, std::string>
  ensureTransmissionVisibilityDepthTexture(TextureHandle sceneDepthTexture);
  void destroyTransmissionVisibilityDepthTexture();
  void destroyPickTexture();
  Result<bool, std::string> recreateShadowInspectTexture();
  void destroyShadowInspectTexture();
  void destroyBuffers();

  GPUDevice &gpu_;
  OpaqueRendererConfig config_{};
  std::unique_ptr<Shader> meshShader_;
  std::unique_ptr<Shader> meshTessShader_;
  std::unique_ptr<Shader> meshDebugOverlayShader_;
  std::unique_ptr<Shader> meshPickShader_;
  std::unique_ptr<Shader> meshShadowInspectShader_;
  std::unique_ptr<Shader> meshVelocityShader_;
  std::unique_ptr<Shader> meshReactiveMaskShader_;
  std::unique_ptr<Shader> meshNormalShader_;
  std::unique_ptr<Shader> depthShader_;
  std::unique_ptr<Shader> depthAlphaShader_;
  std::unique_ptr<Shader> depthPyramidShader_;
  std::unique_ptr<Shader> depthMotionVectorShader_;
  std::unique_ptr<Shader> computeShader_;
  std::unique_ptr<Shader> visibilityShader_;
  std::unique_ptr<Shader> visibilityIndirectDrawShader_;
  std::unique_ptr<Shader> visibilityIndirectMeshDispatchShader_;
  std::unique_ptr<Shader> meshletShader_;
  std::unique_ptr<Pipeline> meshPipeline_;
  std::unique_ptr<Pipeline> computePipeline_;
  std::unique_ptr<Pipeline> visibilityComputePipeline_;
  std::unique_ptr<Pipeline> visibilityIndirectDrawComputePipeline_;
  std::unique_ptr<Pipeline> visibilityIndirectMeshDispatchComputePipeline_;
  std::unique_ptr<Buffer> instanceCentersPhaseBuffer_;
  std::unique_ptr<Buffer> instanceLodBoundsBuffer_;
  std::unique_ptr<Buffer> instanceBaseMatricesBuffer_;
  std::pmr::vector<DynamicBufferSlot> instanceMatricesRing_;
  std::pmr::vector<DynamicBufferSlot> previousInstanceMatricesRing_;
  std::pmr::vector<DynamicBufferSlot> velocityInstanceFlagsRing_;
  std::pmr::vector<DynamicBufferSlot> velocityFrameDataRing_;
  std::pmr::vector<DynamicBufferSlot> velocityGeometryRing_;
  std::pmr::vector<DynamicBufferSlot> instanceRemapRing_;
  std::pmr::vector<DynamicBufferSlot> indirectCommandRing_;
  std::pmr::vector<DynamicBufferSlot> meshletBatchRing_;
  std::pmr::vector<DynamicBufferSlot> visibilityCandidateRing_;
  std::pmr::vector<DynamicBufferSlot> visibilityPassRing_;
  std::pmr::vector<DynamicBufferSlot> visibilityVisibleIndexRing_;
  std::pmr::vector<DynamicBufferSlot> visibilityCounterRing_;
  std::pmr::vector<DynamicBufferSlot> visibilityMeshletDispatchRing_;
  std::pmr::vector<DynamicBufferSlot> visibilityMeshletIndirectCommandRing_;
  bool cachedMeshletCounterValid_ = false;
  uint32_t cachedMeshletCounterSourceFrame_ = 0u;
  uint32_t cachedMeshletEmitted_ = 0u;
  uint32_t cachedMeshletTaskGroupsExecuted_ = 0u;
  TextureHandle pickIdTexture_{};
  TextureHandle shadowInspectTexture_{};
  TextureHandle transmissionVisibilityDepthTexture_{};
  std::array<TextureHandle, kMaxSceneDepthPyramidLevels>
      sceneDepthPyramidTextures_{};
  uint32_t sceneDepthPyramidLevelCount_ = 0;
  uint32_t sceneDepthPyramidWidth_ = 0;
  uint32_t sceneDepthPyramidHeight_ = 0;
  std::optional<uint64_t> sceneDepthPyramidSourceFrameIndex_{};
  std::optional<glm::mat4> sceneDepthPyramidSourceViewProj_{};
  SamplerHandle sceneDepthSampler_{};

  ShaderHandle meshVertexShader_{};
  ShaderHandle meshTessVertexShader_{};
  ShaderHandle meshTessControlShader_{};
  ShaderHandle meshTessEvalShader_{};
  ShaderHandle meshFragmentShader_{};
  ShaderHandle meshDebugOverlayGeometryShader_{};
  ShaderHandle meshDebugOverlayFragmentShader_{};
  ShaderHandle meshPickVertexShader_{};
  ShaderHandle meshPickTessVertexShader_{};
  ShaderHandle meshPickTessControlShader_{};
  ShaderHandle meshPickTessEvalShader_{};
  ShaderHandle meshPickFragmentShader_{};
  ShaderHandle meshShadowInspectFragmentShader_{};
  ShaderHandle meshVelocityVertexShader_{};
  ShaderHandle meshVelocityTessVertexShader_{};
  ShaderHandle meshVelocityTessControlShader_{};
  ShaderHandle meshVelocityTessEvalShader_{};
  ShaderHandle meshVelocityFragmentShader_{};
  ShaderHandle meshReactiveMaskVertexShader_{};
  ShaderHandle meshReactiveMaskFragmentShader_{};
  ShaderHandle meshNormalFragmentShader_{};
  ShaderHandle depthVertexShader_{};
  ShaderHandle depthTessVertexShader_{};
  ShaderHandle depthTessControlShader_{};
  ShaderHandle depthTessEvalShader_{};
  ShaderHandle depthAlphaVertexShader_{};
  ShaderHandle depthAlphaTessVertexShader_{};
  ShaderHandle depthAlphaTessControlShader_{};
  ShaderHandle depthAlphaTessEvalShader_{};
  ShaderHandle depthFragmentShader_{};
  ShaderHandle depthAlphaFragmentShader_{};
  ShaderHandle depthPyramidVertexShader_{};
  ShaderHandle depthPyramidFragmentShader_{};
  ShaderHandle depthMotionVectorVertexShader_{};
  ShaderHandle depthMotionVectorFragmentShader_{};
  ShaderHandle computeShaderHandle_{};
  ShaderHandle visibilityComputeShader_{};
  ShaderHandle visibilityIndirectDrawComputeShader_{};
  ShaderHandle visibilityIndirectMeshDispatchComputeShader_{};
  ShaderHandle meshletTaskShader_{};
  ShaderHandle meshletMeshShader_{};
  ShaderHandle meshletFragmentShader_{};
  ShaderHandle meshletDepthFragmentShader_{};
  ShaderHandle meshletDepthAlphaFragmentShader_{};
  ShaderHandle meshletSimpleNormalMeshShader_{};
  ShaderHandle meshletSimpleNormalFragmentShader_{};
  ShaderHandle meshletNormalFragmentShader_{};
  ShaderHandle meshletVelocityMeshShader_{};
  ShaderHandle meshletVelocityFragmentShader_{};
  ShaderHandle meshletReactiveMaskMeshShader_{};
  ShaderHandle meshletReactiveMaskFragmentShader_{};
  RenderPipelineHandle meshFillPipelineHandle_{};
  RenderPipelineHandle meshDoubleSidedFillPipelineHandle_{};
  RenderPipelineHandle meshTessPipelineHandle_{};
  RenderPipelineHandle meshDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshMsaaFillPipelineHandle_{};
  RenderPipelineHandle meshMsaaDoubleSidedFillPipelineHandle_{};
  RenderPipelineHandle meshMsaaTessPipelineHandle_{};
  RenderPipelineHandle meshMsaaDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshMsaaAlphaFillPipelineHandle_{};
  RenderPipelineHandle meshMsaaAlphaDoubleSidedFillPipelineHandle_{};
  RenderPipelineHandle meshMsaaAlphaTessPipelineHandle_{};
  RenderPipelineHandle meshMsaaAlphaDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshGsOverlayPipelineHandle_{};
  RenderPipelineHandle meshGsTessOverlayPipelineHandle_{};
  RenderPipelineHandle meshWireframePipelineHandle_{};
  RenderPipelineHandle meshTessWireframePipelineHandle_{};
  RenderPipelineHandle meshMsaaGsOverlayPipelineHandle_{};
  RenderPipelineHandle meshMsaaGsTessOverlayPipelineHandle_{};
  RenderPipelineHandle meshMsaaWireframePipelineHandle_{};
  RenderPipelineHandle meshMsaaTessWireframePipelineHandle_{};
  RenderPipelineHandle meshPickPipelineHandle_{};
  RenderPipelineHandle meshPickDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshPickTessPipelineHandle_{};
  RenderPipelineHandle meshPickDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshShadowInspectPipelineHandle_{};
  RenderPipelineHandle meshShadowInspectDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshShadowInspectTessPipelineHandle_{};
  RenderPipelineHandle meshShadowInspectDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshVelocityPipelineHandle_{};
  RenderPipelineHandle meshVelocityDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshVelocityTessPipelineHandle_{};
  RenderPipelineHandle meshVelocityDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshReactiveMaskPipelineHandle_{};
  RenderPipelineHandle meshReactiveMaskDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshNormalPipelineHandle_{};
  RenderPipelineHandle meshNormalDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshNormalTessPipelineHandle_{};
  RenderPipelineHandle meshNormalDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshDepthPipelineHandle_{};
  RenderPipelineHandle meshDepthDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshDepthTessPipelineHandle_{};
  RenderPipelineHandle meshDepthDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshDepthAlphaPipelineHandle_{};
  RenderPipelineHandle meshDepthAlphaDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshDepthAlphaTessPipelineHandle_{};
  RenderPipelineHandle meshDepthAlphaDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshMsaaDepthPipelineHandle_{};
  RenderPipelineHandle meshMsaaDepthDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshMsaaDepthTessPipelineHandle_{};
  RenderPipelineHandle meshMsaaDepthDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle meshMsaaDepthAlphaPipelineHandle_{};
  RenderPipelineHandle meshMsaaDepthAlphaDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshMsaaDepthAlphaTessPipelineHandle_{};
  RenderPipelineHandle meshMsaaDepthAlphaDoubleSidedTessPipelineHandle_{};
  RenderPipelineHandle depthPyramidPipelineHandle_{};
  RenderPipelineHandle depthMotionVectorPipelineHandle_{};
  ComputePipelineHandle computePipelineHandle_{};
  ComputePipelineHandle visibilityPipelineHandle_{};
  ComputePipelineHandle visibilityIndirectDrawPipelineHandle_{};
  ComputePipelineHandle visibilityIndirectMeshDispatchPipelineHandle_{};
  MeshletPipelineHandle meshletPipelineHandle_{};
  MeshletPipelineHandle meshletDoubleSidedPipelineHandle_{};
  MeshletPipelineHandle meshletMsaaPipelineHandle_{};
  MeshletPipelineHandle meshletMsaaDoubleSidedPipelineHandle_{};
  MeshletPipelineHandle meshletDepthPipelineHandle_{};
  MeshletPipelineHandle meshletDepthDoubleSidedPipelineHandle_{};
  MeshletPipelineHandle meshletDepthAlphaPipelineHandle_{};
  MeshletPipelineHandle meshletDepthAlphaDoubleSidedPipelineHandle_{};
  MeshletPipelineHandle meshletMsaaDepthPipelineHandle_{};
  MeshletPipelineHandle meshletMsaaDepthDoubleSidedPipelineHandle_{};
  MeshletPipelineHandle meshletMsaaDepthAlphaPipelineHandle_{};
  MeshletPipelineHandle meshletMsaaDepthAlphaDoubleSidedPipelineHandle_{};
  MeshletPipelineHandle meshletSimpleNormalPipelineHandle_{};
  MeshletPipelineHandle meshletSimpleNormalDoubleSidedPipelineHandle_{};
  MeshletPipelineHandle meshletNormalPipelineHandle_{};
  MeshletPipelineHandle meshletNormalDoubleSidedPipelineHandle_{};
  MeshletPipelineHandle meshletVelocityPipelineHandle_{};
  MeshletPipelineHandle meshletVelocityDoubleSidedPipelineHandle_{};
  MeshletPipelineHandle meshletReactiveMaskPipelineHandle_{};
  MeshletPipelineHandle meshletReactiveMaskDoubleSidedPipelineHandle_{};

  size_t instanceCentersPhaseBufferCapacityBytes_ = 0;
  size_t instanceLodBoundsBufferCapacityBytes_ = 0;
  size_t instanceBaseMatricesBufferCapacityBytes_ = 0;
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
  bool msaaWireframePipelineInitialized_ = false;
  bool msaaWireframePipelineUnsupported_ = false;
  bool msaaTessWireframePipelineInitialized_ = false;
  bool msaaTessWireframePipelineUnsupported_ = false;
  bool msaaGsOverlayPipelineInitialized_ = false;
  bool msaaGsOverlayPipelineUnsupported_ = false;
  bool msaaGsTessOverlayPipelineInitialized_ = false;
  bool msaaGsTessOverlayPipelineUnsupported_ = false;
  bool loggedWireframeFallbackUnsupported_ = false;
  bool loggedTessWireframeFallbackUnsupported_ = false;
  bool loggedGsOverlayUnsupported_ = false;
  bool loggedGsTessOverlayUnsupported_ = false;
  bool loggedDepthPrepassUnsupported_ = false;
  bool loggedTransmissionVisibilityDepthUnsupported_ = false;
  bool loggedNormalPrepassUnsupported_ = false;
  bool loggedDepthPyramidUnsupported_ = false;
  bool loggedMaterialFallbackWarning_ = false;
  bool loggedBlendMaterialUnsupportedWarning_ = false;
  bool loggedShadowSdsmReduceSkipWarning_ = false;
  bool meshletPipelineInitialized_ = false;
  bool meshletPipelineUnsupported_ = false;
  bool loggedMeshletUnsupportedWarning_ = false;
  bool loggedMeshletAssetWarning_ = false;
  bool loggedMeshletIncompatibleWarning_ = false;
  bool loggedVisibilityGpuUnsupportedWarning_ = false;

  const RenderScene *cachedScene_ = nullptr;
  uint64_t cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedGeometryMutationVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedVisibilityCandidateTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedVisibilityCandidateTransformVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedVisibilityCandidateDeformationVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedVisibilityCandidateGeometryVersion_ =
      std::numeric_limits<uint64_t>::max();
  bool cachedExcludeTransmission_ = true;
  bool cachedVisibilityCandidatesHadDeformedRenderable_ = false;
  uint64_t cachedAnimationSceneVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedVisibleBatchTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedVisibleBatchMaterialVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedVisibleBatchGeometryVersion_ =
      std::numeric_limits<uint64_t>::max();
  bool cachedVisibleBatchValid_ = false;
  bool cachedVisibleBatchMeshletRequested_ = false;
  bool cachedVisibleBatchEnableMeshLod_ = false;
  int32_t cachedVisibleBatchForcedMeshLod_ = -1;
  uint64_t currentDirectDrawBufferSignature_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t currentIndirectDrawBufferSignature_ =
      std::numeric_limits<uint64_t>::max();
  bool cachedAnimationSceneActive_ = false;
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
  std::pmr::vector<uint64_t> visibilityCounterRingPublishedFrames_;
  std::pmr::vector<uint32_t> visibilityExpectedVisibleIndexCounts_;
  std::pmr::vector<uint64_t> visibilityExpectedVisibleIndexHashes_;
  std::pmr::vector<uint32_t> visibilityVisibleIndexReadback_;
  std::pmr::vector<VisibilityCandidate> visibilityCandidates_;
  std::pmr::vector<VisibilityCandidateGpu> visibilityCandidateGpuData_;
  std::pmr::vector<uint32_t> templateBatchIndices_;
  std::pmr::vector<uint32_t> cachedVisibleTemplateBatchIndices_;
  std::pmr::vector<uint32_t> visibleBatchActiveRemap_;
  std::pmr::vector<BatchEntry> cachedVisibleBatchEntries_;
  std::pmr::vector<size_t> batchWriteOffsets_;
  std::pmr::vector<glm::vec4> instanceCentersPhase_;
  std::pmr::vector<glm::mat4> instanceBaseMatrices_;
  std::pmr::vector<InstanceData> instanceMatricesCpuCache_;
  std::pmr::vector<glm::vec4> instanceLodCentersInvRadiusSq_;
  std::pmr::vector<TextureHandle> materialTextureAccessHandles_;
  std::pmr::vector<uint32_t> instanceAutoLodLevels_;
  std::pmr::vector<uint8_t> instanceTessSelection_;
  std::pmr::vector<TessCandidate> tessCandidates_;
  std::pmr::vector<uint32_t> instanceRemap_;
  std::pmr::vector<PushConstants> drawPushConstants_;
  std::pmr::vector<DrawItem> drawItems_;
  std::pmr::vector<uint8_t> drawAlphaMasked_;
  std::pmr::vector<MeshletBatchInfo> meshletBatchInfos_;
  std::pmr::vector<DrawItem> indirectDrawItems_;
  std::pmr::vector<uint8_t> indirectAlphaMasked_;
  std::pmr::vector<std::byte> indirectCommandUploadBytes_;
  std::pmr::vector<DrawItem> overlayDrawItems_;
  std::pmr::vector<DrawItem> velocityDrawItems_;
  std::pmr::vector<DrawItem> reactiveMaskDrawItems_;
  std::pmr::vector<DrawItem> pickDrawItems_;
  std::pmr::vector<DrawItem> shadowInspectDrawItems_;
  std::pmr::vector<DrawItem> passDrawItems_;
  std::pmr::vector<DrawItem> msaaPassDrawItems_;
  std::pmr::vector<DrawItem> depthPrepassDrawItems_;
  std::pmr::vector<DrawItem> transmissionVisibilityDepthDrawItems_;
  std::pmr::vector<DrawItem> normalPrepassDrawItems_;
  std::pmr::vector<glm::uvec4> depthPyramidPushConstants_;
  std::pmr::vector<DrawItem> depthPyramidDrawItems_;
  DepthMotionVectorPushConstants depthMotionVectorPushConstants_{};
  DrawItem depthMotionVectorDrawItem_{};
  std::array<TextureHandle, 1> depthMotionVectorDependencyTextures_{};
  std::pmr::vector<MeshDispatchItem> meshletDepthPrepassDispatchItems_;
  std::pmr::vector<MeshletPushConstants> meshletDepthPrepassPushConstants_;
  std::pmr::vector<MeshletDispatchDependencyBuffers>
      meshletDepthPrepassDispatchDependencyBuffers_;
  std::pmr::vector<BufferHandle> meshletDepthPrepassDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode>
      meshletDepthPrepassDependencyBufferAccessModes_;
  std::pmr::vector<MeshDispatchItem> meshletNormalPrepassDispatchItems_;
  std::pmr::vector<MeshletPushConstants> meshletNormalPrepassPushConstants_;
  std::pmr::vector<MeshletDispatchDependencyBuffers>
      meshletNormalPrepassDispatchDependencyBuffers_;
  std::pmr::vector<BufferHandle> meshletNormalPrepassDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode>
      meshletNormalPrepassDependencyBufferAccessModes_;
  std::pmr::vector<MeshDispatchItem> meshletDispatchItems_;
  std::pmr::vector<MeshletPushConstants> meshletPushConstants_;
  std::pmr::vector<MeshletBatchGpuData> meshletBatchGpuData_;
  std::pmr::vector<MeshletDispatchDependencyBuffers>
      meshletDispatchDependencyBuffers_;
  std::pmr::vector<MeshDispatchItem> meshletVelocityDispatchItems_;
  std::pmr::vector<MeshletPushConstants> meshletVelocityPushConstants_;
  std::pmr::vector<MeshletDispatchDependencyBuffers>
      meshletVelocityDispatchDependencyBuffers_;
  std::pmr::vector<MeshDispatchItem> meshletReactiveMaskDispatchItems_;
  std::pmr::vector<MeshletPushConstants> meshletReactiveMaskPushConstants_;
  std::pmr::vector<MeshletDispatchDependencyBuffers>
      meshletReactiveMaskDispatchDependencyBuffers_;
  std::pmr::vector<TextureHandle> depthPyramidDependencyTextures_;
  std::pmr::vector<ShadowSdsmReducePushConstants>
      shadowSdsmReducePushConstants_;
  std::pmr::vector<ShadowSdsmHistogramReducePushConstants>
      shadowSdsmHistogramReducePushConstants_;
  std::pmr::vector<ComputeDispatchItem> shadowSdsmReduceDispatches_;
  std::pmr::vector<BufferHandle> shadowSdsmReduceDependencyBuffers_;
  std::pmr::vector<TextureHandle> shadowSdsmReduceDependencyTextures_;
  std::pmr::vector<ComputeDispatchItem> preDispatches_;
  std::pmr::vector<ComputeDispatchItem> mainPreDispatches_;
  std::pmr::vector<ComputeDispatchItem> visibilityGpuDispatches_;
  std::pmr::vector<VisibilityCandidateGpu> visibilityGpuCandidates_;
  std::pmr::vector<VisibilityPassGpuData> visibilityPassGpuData_;
  std::pmr::vector<VisibilityCounterGpuData> visibilityCounterClear_;
  std::pmr::vector<VisibilityGpuPushConstants> visibilityGpuPushConstants_;
  std::pmr::vector<VisibilityIndirectDrawPushConstants>
      visibilityIndirectDrawPushConstants_;
  std::pmr::vector<VisibilityMeshletDispatchGpuData>
      visibilityMeshletDispatchGpuData_;
  std::pmr::vector<uint32_t> visibilityMeshletCandidateMap_;
  std::pmr::vector<VisibilityIndirectMeshDispatchPushConstants>
      visibilityIndirectMeshDispatchPushConstants_;
  std::pmr::vector<ComputeDispatchItem> visibilityMeshletGpuDispatches_;
  std::pmr::vector<BufferHandle> visibilityMeshletGpuDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode>
      visibilityMeshletGpuDependencyBufferAccessModes_;
  std::pmr::vector<BufferHandle> visibilityGpuDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode>
      visibilityGpuDependencyBufferAccessModes_;
  std::pmr::vector<TextureHandle> visibilityGpuDependencyTextures_;
  std::pmr::vector<BufferHandle> passDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode> passDependencyBufferAccessModes_;
  std::pmr::vector<BufferHandle> preResolvedDecodeBuffers_;
  std::pmr::vector<BufferHandle> preResolvedDrawBuffers_;
  std::pmr::vector<RenderGraphBufferId> cachedPreResolvedDrawBufferIds_;
  std::pmr::vector<BufferHandle> dispatchDependencyBuffers_;
  std::pmr::vector<TextureHandle> passDependencyTextures_;
  std::pmr::vector<BufferHandle> mainPassDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode> mainPassDependencyBufferAccessModes_;
  std::pmr::vector<TextureHandle> mainPassDependencyTextures_;
  std::pmr::vector<RenderGraphAccessMode> mainPassDependencyTextureAccessModes_;
  std::pmr::vector<BufferHandle> velocityPassDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode>
      velocityPassDependencyBufferAccessModes_;
  std::pmr::vector<BufferHandle> reactivePassDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode>
      reactivePassDependencyBufferAccessModes_;
  std::pmr::unordered_map<RenderableId, glm::mat4> previousTransformById_;
  std::pmr::unordered_map<RenderableId, glm::mat4>
      pendingPreviousTransformById_;
  std::pmr::vector<InstanceData> previousInstanceMatricesCpuCache_;
  std::pmr::vector<uint32_t> velocityInstanceFlagsCpuCache_;
  std::pmr::vector<VelocityRenderableGeometryGpuData> velocityGeometryCpuCache_;
  std::pmr::vector<PushConstants> transmissionVisibilityDepthPushConstants_;
  std::pmr::vector<PreparedGraphPass> preparedGraphPasses_;
  PushConstants computePushConstants_{};
  DrawItem baseMeshFillDraw_{};
  DrawItem baseMeshWireframeDraw_{};
  uint64_t cachedPreResolvedBufferFrameIndex_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedPreResolvedBufferSignature_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedRemapSignature_ = std::numeric_limits<uint64_t>::max();
  bool cachedRemapSignatureValid_ = false;
  glm::vec3 cachedMeshLodThresholdsInput_{std::numeric_limits<float>::max()};
  std::array<float, 3> cachedSortedLodThresholds_{0.0f, 0.0f, 0.0f};

  PersistentBufferId persistentCentersPhaseBuffer_{};
  PersistentBufferId persistentBaseMatricesBuffer_{};
  BufferHandle registeredCentersPhaseBufferHandle_{};
  BufferHandle registeredBaseMatricesBufferHandle_{};
  uint64_t boundStaticBatchGeneration_ = 0;
  uint64_t previousTransformSceneId_ = 0u;
  uint64_t previousTransformCaptureFrameIndex_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t previousTransformCaptureTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t previousTransformCaptureTransformVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t pendingPreviousTransformSceneId_ = 0u;
  uint64_t pendingPreviousTransformFrameIndex_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t pendingPreviousTransformTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t pendingPreviousTransformTransformVersion_ =
      std::numeric_limits<uint64_t>::max();
  bool pendingPreviousTransformDataChanged_ = false;
  std::optional<OpaquePickRequest> pendingPickRequest_{};
  std::optional<ShadowInspectRequest> pendingShadowInspectRequest_{};

  struct InFlightPickReadback {
    OpaquePickRequest request{};
    uint64_t submissionFrame = 0;
  };
  std::optional<InFlightPickReadback> inFlightPickReadback_{};

  struct InFlightShadowInspectReadback {
    ShadowInspectRequest request{};
    uint64_t submissionFrame = 0;
  };
  std::optional<InFlightShadowInspectReadback> inFlightShadowInspectReadback_{};
};

} // namespace nuri
