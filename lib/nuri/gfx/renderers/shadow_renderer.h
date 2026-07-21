#pragma once
#include "nuri/core/containers/hash_map.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/dynamic_buffer.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/gfx/renderers/detail/shadow_math.h"
#include "nuri/gfx/renderers/scene_draw_database.h"
#include "nuri/gfx/shader.h"
#include "nuri/gfx/visibility/visibility.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/scene/render_scene.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <string>
#include <vector>
namespace nuri {

using ShadowRendererConfig = RuntimeOpaqueShaderConfig;
class ResourceManager;
class RenderPipeline;

struct BufferDependency {
  BufferHandle handle{};
  RenderGraphAccessMode mode = RenderGraphAccessMode::Read;
};

struct TextureDependency {
  TextureHandle handle{};
  RenderGraphAccessMode mode = RenderGraphAccessMode::Read;
};

class NURI_API ShadowRenderer {
public:
  explicit ShadowRenderer(GPUDevice &gpu, std::pmr::memory_resource *memory =
                                              std::pmr::get_default_resource());
  ShadowRenderer(
      GPUDevice &gpu, const ShadowRendererConfig &config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~ShadowRenderer();
  ShadowRenderer(const ShadowRenderer &) = delete;
  ShadowRenderer &operator=(const ShadowRenderer &) = delete;
  ShadowRenderer(ShadowRenderer &&) = delete;
  ShadowRenderer &operator=(ShadowRenderer &&) = delete;
  Result<bool, std::string> publishFrameData(RenderFrameContext &frame);
  Result<bool, std::string> prepareShadowGraphPasses(RenderFrameContext &frame);
  Result<bool, std::string>
  prepareSceneCache(SceneDrawDatabase &database, const RenderScene &scene,
                    const ResourceManager &resources,
                    const RenderSettings *settings = nullptr);
  [[nodiscard]] bool hasPreparedShadowDepthPasses() const noexcept;
  Result<bool, std::string> appendShadowDepthPasses(RenderFrameContext &frame,
                                                    RenderGraphBuilder &graph);

private:
  enum BufferRingSlot : uint8_t {
    InstanceMatricesRing,
    InstanceRemapRing,
    ShadowDrawPacketRing,
    ShadowFrameRing,
    SdsmReduceResultRing,
    BufferRingCount
  };
  enum ShaderSlot : uint8_t {
    ShadowVertex,
    ShadowOpaqueVertex,
    DepthFragment,
    DepthAlphaFragment,
    SdsmReduceCompute,
    PreviewVertex,
    PreviewFragment,
    ShaderSlotCount
  };
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 488,
                "ShadowRenderer::FrameData must match shader layout");
  struct PushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t shadowDrawMetadataAddress = 0;
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
    float tessMaxFactor = 1.0f;
    uint32_t debugVisualizationMode = 0;
    uint32_t shadowCascadeIndex = 0;
  };
  static_assert(sizeof(PushConstants) == 128,
                "ShadowRenderer::PushConstants must match shader layout");
  static_assert(offsetof(PushConstants, shadowDrawMetadataAddress) == 32u);
  static_assert(offsetof(PushConstants, instanceRemapAddress) == 40u);
  static_assert(offsetof(PushConstants, instanceCentersPhaseAddress) == 48u);
  static_assert(offsetof(PushConstants, instanceBaseMatricesAddress) == 56u);
  static_assert(offsetof(PushConstants, instanceCount) == 80u);
  static_assert(offsetof(PushConstants, shadowCascadeIndex) == 120u);
  using MeshDrawTemplate = SceneDrawRecord;
  struct StaticShadowCasterLightSpaceBounds {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
  };
  struct StaticShadowCasterCacheEntry {
    uint32_t templateIndex = 0;
    uint32_t instanceIndex = 0;
    uint32_t batchIndex = std::numeric_limits<uint32_t>::max();
    BufferHandle indexBuffer{};
    uint64_t indexBufferOffset = 0;
    IndexFormat indexFormat = IndexFormat::U32;
    BufferHandle vertexBuffer{};
    BufferHandle vertexDecodeBuffer{};
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = 0;
    uint64_t rasterSignature = 0;
    uint32_t indexCount = 0;
    uint32_t firstIndex = 0;
    bool doubleSided = false;
    bool alphaMasked = false;
    bool hasCasterCullingBounds = false;
    std::array<glm::vec3, 8> casterWorldCorners{};
    StaticShadowCasterLightSpaceBounds lightBounds{};
  };
  struct StaticShadowBatchKey {
    RenderPipelineHandle pipeline{};
    BufferHandle vertexBuffer{};
    BufferHandle vertexDecodeBuffer{};
    BufferHandle indexBuffer{};
    uint64_t indexBufferOffset = 0;
    IndexFormat indexFormat = IndexFormat::U32;
    uint32_t indexCount = 0;
    uint32_t firstIndex = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = 0;
    bool operator==(const StaticShadowBatchKey &) const noexcept = default;
  };
  struct StaticShadowBatchTemplate : StaticShadowBatchKey {
    uint32_t firstInstanceIndex = 0;
    uint32_t instanceCount = 0;
    uint64_t rasterSignature = 0;
    uint64_t indexCountEstimate = 0;
    StaticShadowCasterLightSpaceBounds lightBounds{};
  };
  struct StaticShadowBatchKeyHash {
    size_t operator()(const StaticShadowBatchKey &key) const noexcept {
      constexpr uint64_t kOffsetBasis = 14695981039346656037ull;
      constexpr uint64_t kPrime = 1099511628211ull;
      const auto combine = [](uint64_t hash, uint64_t value) {
        hash ^= value;
        hash *= kPrime;
        return hash;
      };
      const auto fold = [](uint32_t index, uint32_t generation) {
        return (static_cast<uint64_t>(generation) << 32u) | index;
      };
      uint64_t hash = kOffsetBasis;
      hash = combine(hash, fold(key.pipeline.index, key.pipeline.generation));
      hash = combine(hash,
                     fold(key.vertexBuffer.index, key.vertexBuffer.generation));
      hash = combine(hash, fold(key.vertexDecodeBuffer.index,
                                key.vertexDecodeBuffer.generation));
      hash = combine(hash,
                     fold(key.indexBuffer.index, key.indexBuffer.generation));
      hash = combine(hash, key.indexBufferOffset);
      hash = combine(hash, static_cast<uint64_t>(key.indexFormat));
      hash = combine(hash, (static_cast<uint64_t>(key.indexCount) << 32u) |
                               key.firstIndex);
      hash = combine(hash, key.vertexBufferAddress);
      hash = combine(hash, key.vertexDecodeBufferAddress);
      hash =
          combine(hash, (static_cast<uint64_t>(key.vertexDecodeIndex) << 32u) |
                            key.packedVertexFormat);
      hash = combine(hash, key.materialIndex);
      return static_cast<size_t>(hash);
    }
  };
  struct StaticShadowCasterLightGridCell {
    uint32_t firstEntry = 0;
    uint32_t entryCount = 0;
  };
  struct StaticShadowCasterLightGrid {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
    glm::vec3 invCellSize{0.0f};
    glm::uvec3 dimensions{0u};
    bool valid = false;
  };
  enum StaticLightGridKind : size_t { CasterGrid, BatchGrid, LightGridCount };
  struct PreviewPushConstants {
    glm::uvec4 sourceTexIds{
        kInvalidShadowBindlessIndex, kInvalidShadowBindlessIndex,
        kInvalidShadowBindlessIndex, kInvalidShadowBindlessIndex};
    glm::uvec4 previewParams{0u, 0u, 0u, 0u};
    glm::vec4 depthParams{1.0f, 0.0f, 0.0f, 0.0f};
  };
  static_assert(sizeof(PreviewPushConstants) == 48,
                "ShadowRenderer::PreviewPushConstants layout changed");
  static_assert(
      sizeof(PreviewPushConstants) <= 128,
      "ShadowRenderer::PreviewPushConstants exceeds Vulkan minimum guarantee");
  struct alignas(16) SdsmReducePushConstants {
    uint64_t resultBufferAddress = 0;
    uint32_t sourceTexId = kInvalidShadowBindlessIndex;
    uint32_t sourceFrameIndex = 0;
  };
  static_assert(sizeof(SdsmReducePushConstants) == 16u,
                "ShadowRenderer::SdsmReducePushConstants layout changed");
  static_assert(sizeof(SdsmReducePushConstants) <= 128,
                "ShadowRenderer::SdsmReducePushConstants exceeds Vulkan "
                "minimum guarantee");
  struct alignas(16) SdsmGpuMinMaxResult {
    glm::vec2 rawDeviceMinMax{1.0f, 1.0f};
    uint32_t sourceFrameIndex = 0u;
    uint32_t valid = 0u;
  };
  static_assert(sizeof(SdsmGpuMinMaxResult) == 16u,
                "ShadowRenderer::SdsmGpuMinMaxResult layout changed");
  struct StaticOnlyCascadeReuseState {
    shadow_detail::DirectionalShadowFit renderedFit{};
    shadow_detail::DirectionalShadowFit rawFit{};
    TextureHandle shadowDepthTexture{};
    uint64_t rasterSignature = 0u;
    uint64_t lightViewProjSignature = 0u;
    uint64_t biasSignature = 0u;
    uint64_t casterSignature = 0u;
    uint32_t staticDrawCount = 0u;
    uint32_t dynamicDrawCount = 0u;
  };
  struct CascadeState {
    uint32_t drawCount = 0;
    uint32_t culledCount = 0;
    uint32_t dynamicDrawCount = 0;
    uint64_t indexCountEstimate = 0;
    uint64_t contentSignature = 0;
    uint64_t reusableContentSignature = 0;
    StaticOnlyCascadeReuseState reusable{};
    bool reuse = false;
    bool reusableValid = false;
  };
  struct CascadeStabilizationHistory {
    bool valid = false;
    LightId lightId = kInvalidLightId;
    uint32_t shadowMapSize = 0u;
    uint32_t cascadeCount = 0u;
    std::array<shadow_detail::DirectionalShadowFit, kMaxShadowCascades> fits{};
  };
  struct SdsmState {
    bool hasValidSdsmRange_ = false;
    bool hasValidSdsmFarCascadeTexelSize_ = false;
    uint32_t gpuReductionConsecutiveMissingResultFrames_ = 0u;
    float sdsmSmoothedMinDepth_ = 0.0f;
    float sdsmSmoothedMaxDepth_ = 0.0f;
    float sdsmFarCascadeTexelWorldSize_ = 0.0f;
    uint64_t lastValidSdsmSourceFrameIndex_ =
        std::numeric_limits<uint64_t>::max();
  };
  struct DiagnosticLogState {
    bool hasLastSignature = false;
    uint64_t lastSignature = 0u;
    uint64_t lastLoggedFrameIndex = std::numeric_limits<uint64_t>::max();
  };
  struct FrameSlotState {
    uint64_t instanceUploadVersion = std::numeric_limits<uint64_t>::max();
    uint64_t remapUploadSignature = std::numeric_limits<uint64_t>::max();
    uint64_t drawPacketUploadSignature = std::numeric_limits<uint64_t>::max();
    uint64_t frameUploadSignature = std::numeric_limits<uint64_t>::max();
    uint64_t sdsmPublishedFrame = std::numeric_limits<uint64_t>::max();
  };
  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPreviewShaders();
  Result<bool, std::string> createSdsmReduceShaders();
  Result<bool, std::string> createPipelines(Format depthFormat,
                                            RasterPipelineState rasterState);
  Result<bool, std::string> createPreviewPipeline();
  Result<bool, std::string> createSdsmReducePipeline();
  Result<bool, std::string>
  ensureShadowResources(const RenderSettings::ShadowSettings &settings);
  void ensureRingBufferCount(uint32_t requiredCount);
  Result<bool, std::string>
  ensureInstanceMatricesRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceRemapRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureShadowDrawPacketRingCapacity(size_t requiredBytes);
  Result<bool, std::string> ensureShadowFrameRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureSdsmReduceResultRingCount(uint32_t requiredCount);
  Result<bool, std::string>
  ensureSdsmReduceResultRingCapacity(size_t requiredBytes);
  Result<bool, std::string> ensureRingCapacity(
      BufferRingSlot slot, size_t requiredBytes, std::string_view debugName,
      uint64_t FrameSlotState::*version, Storage storage = Storage::Device);
  void rebuildSceneCache(const SceneDrawDatabase &database);
  void rebuildStaticShadowCasterCache(const RenderScene &scene,
                                      const RenderSettings &settings);
  Result<bool, std::string>
  updateShadowFrameData(RenderFrameContext &frame,
                        const RenderSettings::ShadowSettings &settings,
                        uint32_t shadowMapSize, int32_t forcedMeshLod);
  Result<bool, std::string>
  buildShadowDraws(RenderFrameContext &frame, uint32_t frameSlot,
                   const ForwardSceneGpuData &sceneGpu);
  [[nodiscard]] uint64_t shadowPipelineSignature() const noexcept;
  void invalidateStaticShadowCasterCache() noexcept;
  void invalidateReusableStaticOnlyCascadeCache() noexcept;
  void destroyShadowResources();
  void destroyBuffers();
  void destroyShaders();
  void destroyShadowDepthPipelineState();
  void destroyPipelineState();
  void resetCachedState();
  void resetFrameBuildState();
  void resetFrozenShadowFit();
  void resetCascadeStabilizationHistory();
  void resetSdsmState();
  GPUDevice &gpu_;
  ShadowRendererConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::pmr::vector<std::pmr::vector<DynamicBufferSlot>> bufferRings_;
  std::pmr::vector<FrameSlotState> frameSlotStates_;
  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  ScratchArena batchBuildScratchArena_;
  std::pmr::vector<uint32_t> staticShadowTemplateIndices_;
  std::pmr::vector<uint32_t> dynamicShadowTemplateIndices_;
  std::pmr::vector<StaticShadowCasterCacheEntry> staticShadowCasterCache_;
  std::pmr::vector<StaticShadowBatchTemplate> staticShadowBatchTemplates_;
  PmrHashMap<StaticShadowBatchKey, uint32_t, StaticShadowBatchKeyHash>
      staticShadowBatchIndexMap_;
  std::pmr::vector<uint32_t> staticShadowBatchInstanceIndices_;
  std::pmr::vector<BufferHandle> staticShadowCasterDrawBuffers_;
  std::pmr::vector<glm::vec3> staticShadowCasterFitPoints_;
  std::array<std::pmr::vector<StaticShadowCasterLightGridCell>, LightGridCount>
      staticLightGridCells_;
  std::array<std::pmr::vector<uint32_t>, LightGridCount>
      staticLightGridEntries_;
  std::array<std::pmr::vector<uint32_t>, LightGridCount>
      staticLargeLightGridEntries_;
  std::pmr::vector<uint32_t> staticShadowCasterLightGridQueryMarks_;
  std::pmr::vector<uint32_t> staticShadowCasterLightGridQueryEntries_;
  std::pmr::vector<InstanceData> instanceMatrices_;
  std::pmr::vector<uint32_t> instanceRemap_;
  std::array<std::pmr::vector<PushConstants>, kMaxShadowCascades>
      cascadePushConstants_{};
  std::array<std::pmr::vector<DrawItem>, kMaxShadowCascades>
      cascadeDrawItems_{};
  std::array<std::pmr::vector<PushConstants>, kMaxShadowCascades>
      cascadeIndirectPushConstants_{};
  std::array<std::pmr::vector<DrawItem>, kMaxShadowCascades>
      cascadeIndirectDrawItems_{};
  std::pmr::vector<std::byte> shadowDrawPacketUploadBytes_;
  std::array<CascadeState, kMaxShadowCascades> cascadeStates_{};
  std::pmr::vector<BufferDependency> passBufferDependencies_;
  std::pmr::vector<BufferHandle> passDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode> passDependencyBufferAccessModes_;
  std::pmr::vector<TextureHandle> passDependencyTextures_;
  std::pmr::vector<RenderGraphAccessMode> passDependencyTextureAccessModes_;
  std::pmr::vector<BufferHandle> preResolvedDrawBuffers_;
  std::pmr::vector<TextureDependency> passTextureDependencies_;
  std::pmr::vector<TextureDependency> previewTextureDependencies_;
  std::array<ShaderHandle, ShaderSlotCount> shaders_{};
  std::array<RenderPipelineHandle, 4> shadowPipelines_{};
  Format shadowDepthPipelineFormat_ = Format::Count;
  RasterPipelineState shadowPipelineRasterState_{};
  ComputePipelineHandle sdsmReducePipelineHandle_{};
  RenderPipelineHandle previewPipelineHandle_{};
  std::array<TextureHandle, kMaxShadowCascades> shadowDepthTextures_{};
  TextureHandle shadowDebugPreviewTexture_{};
  SamplerHandle rawDepthSampler_{};
  SamplerHandle compareDepthSampler_{};
  uint32_t shadowMapSize_ = 0u;
  uint32_t activeCascadeCount_ = 0u;
  bool initialized_ = false;
  bool hasPreparedShadowDepthPasses_ = false;
  bool hasPreparedShadowPreviewPass_ = false;
  bool hasActiveShadowLightForFrame_ = false;
  PreviewPushConstants previewPushConstants_{};
  DrawItem previewDraw_{};
  const RenderScene *cachedScene_ = nullptr;
  uint64_t cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedModelMaterialBindingVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedDeformationVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedGeometryMutationVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t staticShadowCasterCacheTransformVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t staticShadowCasterCachePipelineSignature_ = 0u;
  int32_t staticShadowCasterCacheForcedMeshLod_ =
      std::numeric_limits<int32_t>::min();
  glm::vec3 staticShadowCasterBoundsMin_{std::numeric_limits<float>::max()};
  glm::vec3 staticShadowCasterBoundsMax_{std::numeric_limits<float>::lowest()};
  glm::vec3 staticShadowCasterLightDepthDirection_{0.0f, -1.0f, 0.0f};
  glm::vec2 staticShadowCasterLightDepthBounds_{0.0f};
  glm::mat4 staticShadowCasterLightSpaceBoundsView_{1.0f};
  glm::vec3 staticShadowCasterLightSpaceBoundsMin_{
      std::numeric_limits<float>::max()};
  glm::vec3 staticShadowCasterLightSpaceBoundsMax_{
      std::numeric_limits<float>::lowest()};
  std::array<StaticShadowCasterLightGrid, LightGridCount> staticLightGrids_{};
  uint64_t staticShadowCasterCacheContentSignature_ = 0u;
  uint64_t staticShadowCasterCacheIndexCountEstimate_ = 0u;
  uint32_t staticShadowCasterLightGridQueryMarker_ = 1u;
  bool hasFrozenShadowFit_ = false;
  bool hasStaticShadowCasterBounds_ = false;
  bool hasStaticShadowCasterLightDepthBounds_ = false;
  bool hasStaticShadowCasterLightSpaceBounds_ = false;
  bool staticShadowCasterCacheValid_ = false;
  LightId frozenShadowLightId_ = kInvalidLightId;
  uint32_t frozenShadowMapSize_ = 0u;
  uint32_t frozenCascadeCount_ = 0u;
  CascadeStabilizationHistory cascadeStabilizationHistory_{};
  SdsmState sdsmState_{};
  DiagnosticLogState diagnosticLogState_{};
  std::array<shadow_detail::DirectionalShadowFit, kMaxShadowCascades>
      frozenShadowFits_{};
  std::array<shadow_detail::DirectionalShadowFit, kMaxShadowCascades>
      currentRawShadowFits_{};
  ShadowFrameGpuData shadowFrameCpuData_{};
  ShadowDebugFrameData shadowDebugFrameData_{};
};

NURI_API ShadowRenderer *registerShadowStage(
    RenderPipeline &pipeline, GPUDevice &gpu,
    std::pmr::memory_resource *memory = std::pmr::get_default_resource(),
    SceneDrawDatabase *database = nullptr);
NURI_API ShadowRenderer *registerShadowStage(
    RenderPipeline &pipeline, GPUDevice &gpu,
    const RuntimeOpaqueShaderConfig &config,
    std::pmr::memory_resource *memory = std::pmr::get_default_resource(),
    SceneDrawDatabase *database = nullptr);

} // namespace nuri
