#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/dynamic_buffer.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/owned_gpu_resource.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/gfx/renderers/scene_draw_database.h"
#include "nuri/gfx/shader.h"
#include "nuri/gfx/visibility/visibility.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/scene/render_scene.h"
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>
namespace nuri {

using OpaqueRendererConfig = RuntimeOpaqueShaderConfig;
class ResourceManager;
class RenderPipeline;

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
  Result<bool, std::string> prepareSceneCache(SceneDrawDatabase &database,
                                              const RenderScene &scene,
                                              const ResourceManager &resources);
  void commitSubmittedFrame(uint64_t frameIndex) noexcept;
  void abandonPreparedFrame(uint64_t frameIndex) noexcept;
  [[nodiscard]] bool hasPreparedOpaquePrepassPasses() const noexcept;
  [[nodiscard]] bool hasPreparedOpaqueMainLightingPasses() const noexcept;
  [[nodiscard]] bool hasPreparedOpaquePickPasses() const noexcept;
  Result<bool, std::string>
  appendOpaquePrepassPasses(RenderFrameContext &frame,
                            RenderGraphBuilder &graph);
  Result<bool, std::string>
  appendOpaqueMainLightingPasses(RenderFrameContext &frame,
                                 RenderGraphBuilder &graph);
  Result<bool, std::string> appendOpaquePickPasses(RenderFrameContext &frame,
                                                   RenderGraphBuilder &graph);

private:
  enum class PreparedPassPhase : uint8_t { PreLighting, MainLighting, Pick };
  enum class PreparedPassKind : uint8_t {
    Other,
    Main,
    Pick,
    Depth,
    TransmissionDepth,
    Normal,
    DepthPyramid,
    Velocity,
    ReactiveMask
  };
  enum class OverlayPipelineKind : uint8_t {
    Wireframe,
    TessWireframe,
    Geometry,
    TessGeometry,
    Count
  };
  enum ShaderSlot : uint8_t {
    MeshVertex,
    MeshTessVertex,
    MeshTessControl,
    MeshTessEval,
    MeshFragment,
    MeshDebugOverlayGeometry,
    MeshDebugOverlayFragment,
    MeshPickVertex,
    MeshPickTessVertex,
    MeshPickTessControl,
    MeshPickTessEval,
    MeshPickFragment,
    MeshShadowInspectFragment,
    MeshVelocityVertex,
    MeshVelocityTessVertex,
    MeshVelocityTessControl,
    MeshVelocityTessEval,
    MeshVelocityFragment,
    MeshReactiveMaskVertex,
    MeshReactiveMaskFragment,
    MeshNormalFragment,
    DepthVertex,
    DepthTessVertex,
    DepthTessControl,
    DepthTessEval,
    DepthAlphaVertex,
    DepthAlphaTessVertex,
    DepthAlphaTessControl,
    DepthAlphaTessEval,
    DepthFragment,
    DepthAlphaFragment,
    DepthPyramidVertex,
    DepthPyramidFragment,
    DepthMotionVectorVertex,
    DepthMotionVectorFragment,
    Compute,
    VisibilityCompute,
    VisibilityIndirectDrawCompute,
    VisibilityIndirectMeshDispatchCompute,
    MeshletCompactionCompute,
    MeshletTask,
    MeshletCompactedTask,
    MeshletMesh,
    MeshletFragment,
    MeshletDepthFragment,
    MeshletDepthAlphaFragment,
    MeshletSimpleNormalMesh,
    MeshletSimpleNormalFragment,
    MeshletNormalFragment,
    MeshletVelocityMesh,
    MeshletVelocityFragment,
    MeshletReactiveMaskMesh,
    MeshletReactiveMaskFragment,
    ShaderSlotCount
  };
  enum BufferRingSlot : uint8_t {
    InstanceMatricesRing,
    PreviousInstanceMatricesRing,
    VelocityInstanceFlagsRing,
    VelocityFrameDataRing,
    VelocityGeometryRing,
    InstanceRemapRing,
    IndirectCommandRing,
    MeshletBatchRing,
    VisibilityCandidateRing,
    VisibilityPassRing,
    VisibilityVisibleIndexRing,
    VisibilityCounterRing,
    VisibilityMeshletDispatchRing,
    VisibilityMeshletIndirectCommandRing,
    MeshletCompactionRing,
    BufferRingCount
  };
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 488,
                "OpaqueRenderer::FrameData must match shader FrameDataBuffer "
                "layout");
  using PushConstants = ForwardMeshPushConstants;
  struct MeshletPushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t instanceRemapAddress = 0;
    uint64_t instanceLodBoundsAddress = 0;
    glm::vec4 lodThresholds{0.0f};
    uint64_t meshletBatchBufferAddress = 0;
    uint64_t visibilityCounterBufferAddress = 0;
    uint64_t compactedMeshletBufferAddress = 0;
    uint64_t compactionCounterBufferAddress = 0;
    uint64_t velocityFrameDataAddress = 0;
    uint32_t batchBase = 0;
    uint32_t candidateOffset = 0;
    uint32_t sourceFrameIndex = 0;
    uint32_t meshletCounterFlags = 0;
    uint32_t currentDepthVerificationTexId = kInvalidTextureBindlessIndex;
    uint32_t currentDepthVerificationExtentPacked = 0u;
  };
  static_assert(sizeof(MeshletPushConstants) == 112,
                "OpaqueRenderer::MeshletPushConstants must match shader "
                "layout");
  static_assert(offsetof(MeshletPushConstants, lodThresholds) == 32u);
  static_assert(offsetof(MeshletPushConstants, meshletBatchBufferAddress) ==
                48u);
  static_assert(offsetof(MeshletPushConstants,
                         visibilityCounterBufferAddress) == 56u);
  static_assert(offsetof(MeshletPushConstants, compactedMeshletBufferAddress) ==
                64u);
  static_assert(offsetof(MeshletPushConstants,
                         compactionCounterBufferAddress) == 72u);
  static_assert(offsetof(MeshletPushConstants, velocityFrameDataAddress) ==
                80u);
  static_assert(offsetof(MeshletPushConstants, batchBase) == 88u);
  static_assert(offsetof(MeshletPushConstants, currentDepthVerificationTexId) ==
                104u);
  static_assert(offsetof(MeshletPushConstants,
                         currentDepthVerificationExtentPacked) == 108u);
  struct alignas(16) MeshletCompactionWorkItemGpuData {
    glm::uvec4 data{0u};
  };
  static_assert(sizeof(MeshletCompactionWorkItemGpuData) == 16u);
  struct alignas(16) CompactedMeshletGpuData {
    glm::uvec4 ids{0u};
  };
  static_assert(sizeof(CompactedMeshletGpuData) == 16u);
  struct alignas(16) MeshletCompactionPushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t instanceRemapAddress = 0;
    uint64_t instanceLodBoundsAddress = 0;
    uint64_t meshletBatchBufferAddress = 0;
    uint64_t workItemBufferAddress = 0;
    uint64_t compactedMeshletBufferAddress = 0;
    uint64_t compactionCounterBufferAddress = 0;
    uint64_t indirectCommandBufferAddress = 0;
    uint64_t visibilityCounterBufferAddress = 0;
    glm::vec4 lodThresholds{0.0f};
    uint32_t workItemCount = 0;
    uint32_t dispatchCount = 0;
    uint32_t compactGridWidth = 0;
    uint32_t sourceFrameIndex = 0;
    uint32_t meshletCounterFlags = 0;
    uint32_t flags = 0;
    uint32_t currentDepthVerificationTexId = kInvalidTextureBindlessIndex;
    uint32_t currentDepthVerificationExtentPacked = 0u;
  };
  static_assert(sizeof(MeshletCompactionPushConstants) == 128u);
  static_assert(offsetof(MeshletCompactionPushConstants, lodThresholds) == 80u);
  static_assert(offsetof(MeshletCompactionPushConstants, workItemCount) == 96u);
  static_assert(offsetof(MeshletCompactionPushConstants,
                         currentDepthVerificationTexId) == 120u);
  static_assert(offsetof(MeshletCompactionPushConstants,
                         currentDepthVerificationExtentPacked) == 124u);
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
  using RenderableTemplate = SceneInstanceRecord;
  using MeshDrawTemplate = SceneDrawRecord;
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
    uint32_t resolvedLod = 0;
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
    uint32_t resolvedLod = 0;
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
  struct alignas(16) VelocityFrameGpuData {
    glm::mat4 currentViewProjNoJitter{1.0f};
    glm::mat4 previousViewProjNoJitter{1.0f};
    uint64_t previousInstanceMatricesAddress = 0u;
    uint64_t velocityInstanceFlagsAddress = 0u;
    glm::uvec4 instanceFlagsMode{0u};
    uint64_t previousGeometryAddress = 0u;
    uint64_t previousGeometryAddressPadding = 0u;
    glm::uvec4 previousGeometryInfo{0u};
  };
  static_assert(sizeof(VelocityFrameGpuData) == sizeof(glm::mat4) * 2u +
                                                    sizeof(glm::uvec4) * 2u +
                                                    sizeof(uint64_t) * 4u,
                "OpaqueRenderer::VelocityFrameGpuData layout changed");
  static_assert(offsetof(VelocityFrameGpuData,
                         previousInstanceMatricesAddress) == 128u);
  static_assert(offsetof(VelocityFrameGpuData, velocityInstanceFlagsAddress) ==
                136u);
  static_assert(offsetof(VelocityFrameGpuData, previousGeometryAddress) ==
                160u);
  static_assert(offsetof(VelocityFrameGpuData, previousGeometryInfo) == 176u);
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
    bool automaticLod = false;
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
    RenderGraphGraphicsPassDesc desc{};
    TextureHandle colorTextureHandle{};
    TextureHandle colorResolveTextureHandle{};
    TextureHandle depthTextureHandle{};
    TextureHandle depthResolveTextureHandle{};
    PreparedPassPhase phase = PreparedPassPhase::MainLighting;
    PreparedPassKind kind = PreparedPassKind::Other;
    bool publishesDepth = false;
    uint32_t depthPyramidLevel = UINT32_MAX;
    explicit PreparedGraphPass(
        std::pmr::memory_resource * = std::pmr::get_default_resource()) {}
  };
  struct IndirectPackCache {
    bool valid = false;
    uint64_t drawSignature = std::numeric_limits<uint64_t>::max();
    size_t requiredBytes = 0;
  };
  struct FrameSlotState {
    uint64_t matricesUploadVersion = std::numeric_limits<uint64_t>::max();
    uint64_t indirectUploadSignature = std::numeric_limits<uint64_t>::max();
    uint64_t remapUploadSignature = std::numeric_limits<uint64_t>::max();
    uint64_t visibilityPublishedFrame = std::numeric_limits<uint64_t>::max();
    uint32_t expectedVisibleCount = 0;
    uint64_t expectedVisibleHash = 0;
    bool expectedVisibleListValid = false;
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
  [[nodiscard]] bool meshletPipelinesConfigured() const noexcept;
  Result<bool, std::string> recreatePickTexture();
  Result<bool, std::string> ensureStaticInstanceBufferCapacity(size_t count);
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
  ensureMeshletCompactionRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureVisibilityMeshletIndirectCommandRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureVisibilityCounterRingCapacity(size_t requiredBytes);
  void invalidateVisibilityReadbackSlot(size_t slot);
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
  ensureSingleInstanceBatchCache(uint32_t requestedLod, bool automaticLod,
                                 bool tessPipelineEnabled,
                                 const DrawItem &baseDraw);
  [[nodiscard]] size_t singleInstanceCacheIndex(uint32_t requestedLod,
                                                bool tessPipelineEnabled) const;
  Result<bool, std::string> buildIndirectDraws(uint32_t frameSlot,
                                               size_t remapCount,
                                               uint64_t drawSignature,
                                               bool drawSignatureValid);
  [[nodiscard]] uint64_t computeIndirectDrawSignature(size_t remapCount) const;
  Result<bool, std::string> rebuildIndirectPack(uint32_t frameSlot,
                                                uint64_t drawSignature);
  Result<bool, std::string> refreshCachedIndirectPack(uint32_t frameSlot,
                                                      uint64_t drawSignature);
  void rebuildSceneCache(const SceneDrawDatabase &database,
                         const RenderScene &scene, bool excludeTransmission);
  Result<bool, std::string>
  rebuildMaterialTextureAccessCache(const SceneDrawDatabase &database,
                                    const ResourceManager &resources,
                                    bool excludeTransmission);
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPipelines();
  Result<bool, std::string> createMeshletPipelineState();
  void readLatestVisibilityGpuReadback(RenderFrameContext &frame);
  Result<bool, std::string> appendGpuVisibilityMainPass(
      RenderFrameContext &frame, uint32_t frameSlot,
      std::span<const VisibilityCandidate> candidates,
      std::span<const VisibilityCandidateGpu> candidateGpuData,
      std::span<const uint32_t> candidateIndices,
      const VisibilityPassRequest &request,
      const VisibilityResolvedSettings &settings, bool validateVisibleList,
      std::pmr::vector<PreparedGraphPass> &out);
  Result<bool, std::string>
  buildOpaquePasses(RenderFrameContext &frame,
                    std::pmr::vector<PreparedGraphPass> &out);
  Result<bool, std::string>
  ensureDepthPyramidTextures(bool currentFrameVerificationRequired);
  [[nodiscard]] bool requiresDepthPyramid(const RenderSettings &settings) const;
  [[nodiscard]] bool
  shouldBuildTransmissionVisibilityDepth(const RenderFrameContext &frame,
                                         const RenderSettings &settings) const;
  void appendPreparedGraphPass(RenderFrameContext &frame,
                               RenderGraphBuilder &graph,
                               const PreparedGraphPass &pass,
                               uint32_t safeWidth, uint32_t safeHeight);
  Result<bool, std::string> appendPreparedPasses(RenderFrameContext &frame,
                                                 RenderGraphBuilder &graph,
                                                 PreparedPassPhase phase);
  [[nodiscard]] bool hasPreparedPasses(PreparedPassPhase phase) const noexcept;
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
                      CoverageMode coverage) const;
  [[nodiscard]] RenderPipelineHandle
  selectNormalPipeline(RenderPipelineHandle sourcePipeline) const;
  [[nodiscard]] RenderPipelineHandle
  selectPickPipeline(RenderPipelineHandle sourcePipeline) const;
  [[nodiscard]] RenderPipelineHandle
  selectShadowInspectPipeline(RenderPipelineHandle sourcePipeline) const;
  [[nodiscard]] RenderPipelineHandle
  selectMsaaScenePipeline(RenderPipelineHandle sourcePipeline, bool alphaMasked,
                          CoverageMode coverage) const;
  [[nodiscard]] MeshletPipelineHandle
  selectMeshletScenePipeline(bool compacted, CoverageMode coverage,
                             bool doubleSided) const;
  [[nodiscard]] MeshletPipelineHandle
  selectMeshletDepthPipeline(CoverageMode coverage, bool alphaMasked,
                             bool doubleSided) const;
  [[nodiscard]] bool isDoubleSidedPipeline(RenderPipelineHandle handle) const;
  [[nodiscard]] bool isTessPipeline(RenderPipelineHandle handle) const;
  Result<bool, std::string> ensureSceneDepthSampler();
  [[nodiscard]] static constexpr size_t
  overlayPipelineIndex(OverlayPipelineKind kind,
                       CoverageMode coverage) noexcept {
    return static_cast<size_t>(kind) +
           coverageModeIndex(coverage) *
               static_cast<size_t>(OverlayPipelineKind::Count);
  }
  [[nodiscard]] RenderPipelineHandle
  overlayPipeline(OverlayPipelineKind kind,
                  CoverageMode coverage) const noexcept;
  bool ensureOverlayPipeline(OverlayPipelineKind kind, CoverageMode coverage);
  void resetOverlayPipelineState();
  void invalidateAutoLodHistory();
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
  OwnedRenderPipelineHandle meshPipeline_;
  OwnedComputePipelineHandle computePipeline_;
  OwnedComputePipelineHandle visibilityComputePipeline_;
  OwnedComputePipelineHandle visibilityIndirectDrawComputePipeline_;
  OwnedComputePipelineHandle visibilityIndirectMeshDispatchComputePipeline_;
  OwnedComputePipelineHandle meshletCompactionComputePipeline_;
  std::unique_ptr<Buffer> instanceCentersPhaseBuffer_;
  std::unique_ptr<Buffer> instanceLodBoundsBuffer_;
  std::unique_ptr<Buffer> instanceBaseMatricesBuffer_;
  std::pmr::vector<std::pmr::vector<DynamicBufferSlot>> bufferRings_;
  bool cachedMeshletCounterValid_ = false;
  uint32_t cachedMeshletCounterSourceFrame_ = 0u;
  uint32_t cachedMeshletEmitted_ = 0u;
  uint32_t cachedMeshletTaskGroupsExecuted_ = 0u;
  TextureHandle pickIdTexture_{};
  TextureHandle shadowInspectTexture_{};
  TextureHandle transmissionVisibilityDepthTexture_{};
  TextureHandle currentFrameDepthVerificationTexture_{};
  std::array<TextureHandle, kMaxSceneDepthPyramidLevels>
      sceneDepthPyramidTextures_{};
  uint32_t sceneDepthPyramidLevelCount_ = 0;
  uint32_t sceneDepthPyramidWidth_ = 0;
  uint32_t sceneDepthPyramidHeight_ = 0;
  std::optional<uint64_t> sceneDepthPyramidSourceFrameIndex_{};
  std::optional<glm::mat4> sceneDepthPyramidSourceViewProj_{};
  SamplerHandle sceneDepthSampler_{};
  std::array<ShaderHandle, ShaderSlotCount> shaders_{};
  std::array<RenderPipelineHandle, 8 * kCoverageModeCount>
      meshScenePipelines_{};
  std::array<RenderPipelineHandle, 4> meshPickPipelines_{};
  std::array<RenderPipelineHandle, 4> meshShadowInspectPipelines_{};
  std::array<RenderPipelineHandle, 4> meshVelocityPipelines_{};
  std::array<RenderPipelineHandle, 2> meshReactiveMaskPipelines_{};
  std::array<RenderPipelineHandle, 4> meshNormalPipelines_{};
  std::array<RenderPipelineHandle, 8 * kCoverageModeCount>
      meshDepthPipelines_{};
  std::array<RenderPipelineHandle,
             static_cast<size_t>(OverlayPipelineKind::Count) *
                 kCoverageModeCount>
      overlayPipelines_{};
  RenderPipelineHandle currentFrameDepthVerificationPipelineHandle_{};
  RenderPipelineHandle depthPyramidPipelineHandle_{};
  RenderPipelineHandle depthMotionVectorPipelineHandle_{};
  std::array<MeshletPipelineHandle, 4 * kCoverageModeCount>
      meshletScenePipelines_{};
  std::array<MeshletPipelineHandle, 4 * kCoverageModeCount>
      meshletDepthPipelines_{};
  std::array<MeshletPipelineHandle, 4> meshletNormalPipelines_{};
  std::array<MeshletPipelineHandle, 2> meshletVelocityPipelines_{};
  std::array<MeshletPipelineHandle, 2> meshletReactiveMaskPipelines_{};
  size_t instanceCentersPhaseBufferCapacityBytes_ = 0;
  size_t instanceLodBoundsBufferCapacityBytes_ = 0;
  size_t instanceBaseMatricesBufferCapacityBytes_ = 0;
  bool initialized_ = false;
  bool tessellationUnsupported_ = false;
  std::array<bool, static_cast<size_t>(OverlayPipelineKind::Count) *
                       kCoverageModeCount>
      overlayPipelineUnsupported_{};
  bool meshletPipelineInitialized_ = false;
  const RenderScene *cachedScene_ = nullptr;
  uint64_t cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedModelMaterialBindingVersion_ =
      std::numeric_limits<uint64_t>::max();
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
  static constexpr size_t kSingleInstanceCacheVariantCount =
      static_cast<size_t>(Submesh::kMaxLodCount) * 2u;
  std::pmr::vector<SingleInstanceBatchCache> singleInstanceBatchCaches_;
  uint64_t singleInstanceTemplateRevision_ = 1;
  IndirectPackCache indirectPackCache_{};
  StaticBatchCache staticBatchCache_;
  std::pmr::vector<RenderableTemplate> renderableTemplates_;
  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<size_t> indirectSourceDrawIndices_;
  std::pmr::vector<FrameSlotState> frameSlotStates_;
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
  std::pmr::vector<glm::vec4> instanceAutoLodWorldErrors_;
  std::pmr::vector<uint8_t> instanceAutoLodCounts_;
  std::pmr::vector<TextureHandle> materialTextureAccessHandles_;
  bool materialTextureAccessCacheValid_ = false;
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
  std::pmr::vector<MeshletCompactionWorkItemGpuData>
      meshletCompactionWorkItems_;
  std::pmr::vector<MeshletCompactionPushConstants>
      meshletCompactionPushConstants_;
  std::pmr::vector<ComputeDispatchItem> meshletCompactionDispatches_;
  std::pmr::vector<uint32_t> meshletCompactionCounterClear_;
  std::pmr::vector<BufferHandle> meshletCompactionDependencyBuffers_;
  std::pmr::vector<BufferHandle> meshletCompactionFinalizeDependencyBuffers_;
  std::pmr::vector<TextureHandle> meshletCompactionDependencyTextures_;
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
  uint64_t cachedRemapSignature_ = std::numeric_limits<uint64_t>::max();
  bool cachedRemapSignatureValid_ = false;
  bool autoLodHistoryValid_ = false;
  bool autoLodWasActive_ = false;
  float cachedAutoLodTargetPixelError_ =
      std::numeric_limits<float>::quiet_NaN();
  float cachedAutoLodHysteresisRatio_ = std::numeric_limits<float>::quiet_NaN();
  float cachedAutoLodProjectionScaleY_ =
      std::numeric_limits<float>::quiet_NaN();
  float cachedAutoLodNearPlane_ = std::numeric_limits<float>::quiet_NaN();
  glm::uvec2 cachedAutoLodRenderExtent_{0u};
  ProjectionType cachedAutoLodProjectionType_ = ProjectionType::Perspective;
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

NURI_API OpaqueRenderer *registerOpaquePrepassStages(
    RenderPipeline &pipeline, GPUDevice &gpu, RuntimeOpaqueShaderConfig config,
    std::pmr::memory_resource *memory = std::pmr::get_default_resource(),
    SceneDrawDatabase *database = nullptr);
NURI_API void registerOpaqueMainStage(RenderPipeline &pipeline,
                                      OpaqueRenderer &renderer);

} // namespace nuri
