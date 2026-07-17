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
  [[nodiscard]] bool hasPreparedShadowDepthPasses() const noexcept;
  Result<bool, std::string> appendShadowDepthPasses(RenderFrameContext &frame,
                                                    RenderGraphBuilder &graph);

private:
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 464,
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

  struct MeshDrawTemplate {
    const Renderable *renderable = nullptr;
    const Submesh *submesh = nullptr;
    uint32_t submeshIndex = 0;
    uint32_t instanceIndex = 0;
    BufferHandle indexBuffer{};
    uint64_t indexBufferOffset = 0;
    IndexFormat indexFormat = IndexFormat::U32;
    BufferHandle baseVertexBuffer{};
    BufferHandle vertexDecodeBuffer{};
    uint64_t vertexBufferByteOffset = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = 0;
    bool doubleSided = false;
    bool alphaMasked = false;
    bool dynamicCaster = false;
  };

  // Duplicates hot draw fields from MeshDrawTemplate intentionally so static
  // shadow cache traversal can stay compact and avoid template indirection.
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
  };

  struct StaticShadowBatchTemplate {
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
    uint32_t firstInstanceIndex = 0;
    uint32_t instanceCount = 0;
    uint64_t rasterSignature = 0;
    uint64_t indexCountEstimate = 0;
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
    bool operator==(const StaticShadowBatchKey &other) const noexcept {
      return pipeline.index == other.pipeline.index &&
             pipeline.generation == other.pipeline.generation &&
             vertexBuffer.index == other.vertexBuffer.index &&
             vertexBuffer.generation == other.vertexBuffer.generation &&
             vertexDecodeBuffer.index == other.vertexDecodeBuffer.index &&
             vertexDecodeBuffer.generation ==
                 other.vertexDecodeBuffer.generation &&
             indexBuffer.index == other.indexBuffer.index &&
             indexBuffer.generation == other.indexBuffer.generation &&
             indexBufferOffset == other.indexBufferOffset &&
             indexFormat == other.indexFormat &&
             indexCount == other.indexCount && firstIndex == other.firstIndex &&
             vertexBufferAddress == other.vertexBufferAddress &&
             vertexDecodeBufferAddress == other.vertexDecodeBufferAddress &&
             vertexDecodeIndex == other.vertexDecodeIndex &&
             packedVertexFormat == other.packedVertexFormat &&
             materialIndex == other.materialIndex;
    }
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

  struct StaticShadowCasterLightSpaceBounds {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
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
    ShadowSdsmStatus lastLoggedSdsmWarningStatus_ = ShadowSdsmStatus::Disabled;
    bool lastLoggedSdsmWarningFixedFallbackActive_ = false;
    bool lastLoggedSdsmWarningReusedCachedRange_ = false;
    uint64_t lastLoggedSdsmWarningSourceFrameIndex_ =
        std::numeric_limits<uint64_t>::max();
    uint64_t lastLoggedSdsmWarningFrameIndex_ =
        std::numeric_limits<uint64_t>::max();
    bool loggedGpuReductionFallbackWarning_ = false;
    bool loggedGpuReductionUnavailableWarning_ = false;
    bool loggedGpuResultRingDiagnosticWarning_ = false;
  };

  struct DiagnosticLogState {
    struct CascadeLightState {
      bool valid = false;
      glm::mat4 lightView{1.0f};
      glm::mat4 lightViewProj{1.0f};
      glm::vec4 lightSpaceBoundsMin{0.0f};
      glm::vec4 lightSpaceBoundsMax{0.0f};
      glm::vec4 unsnappedCenter{0.0f};
      glm::vec4 snappedCenter{0.0f};
    };

    bool hasLastSignature = false;
    uint64_t lastSignature = 0u;
    uint64_t lastLoggedFrameIndex = std::numeric_limits<uint64_t>::max();
    std::array<CascadeLightState, kMaxShadowCascades> cascadeLightStates{};
  };

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPreviewShaders();
  Result<bool, std::string> createSdsmReduceShaders();
  Result<bool, std::string> createPipelines(Format depthFormat,
                                            RasterPipelineState rasterState);
  Result<bool, std::string> createPreviewPipeline();
  Result<bool, std::string> createSdsmReducePipeline();
  Result<bool, std::string> ensureSdsmReduceResources();
  Result<bool, std::string>
  ensureShadowResources(const RenderSettings::ShadowSettings &settings);
  Result<bool, std::string> ensureRingBufferCount(uint32_t requiredCount);
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
  Result<bool, std::string>
  ensureRingCapacity(std::pmr::vector<DynamicBufferSlot> &ring,
                     size_t requiredBytes, std::string_view debugName,
                     std::span<uint64_t> uploadVersions,
                     Storage storage = Storage::Device);
  Result<bool, std::string> rebuildSceneCache(const RenderScene &scene,
                                              const ResourceManager &resources,
                                              uint32_t materialCount);
  Result<bool, std::string>
  rebuildStaticShadowCasterCache(const RenderScene &scene,
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
  std::unique_ptr<Shader> shadowShader_;
  std::unique_ptr<Shader> shadowOpaqueShader_;
  std::unique_ptr<Shader> depthShader_;
  std::unique_ptr<Shader> depthAlphaShader_;
  std::unique_ptr<Shader> sdsmReduceShader_;
  std::pmr::vector<DynamicBufferSlot> instanceMatricesRing_;
  std::pmr::vector<DynamicBufferSlot> instanceRemapRing_;
  std::pmr::vector<DynamicBufferSlot> shadowDrawPacketRing_;
  std::pmr::vector<DynamicBufferSlot> shadowFrameRing_;
  std::pmr::vector<DynamicBufferSlot> sdsmReduceResultRing_;
  std::pmr::vector<uint64_t> instanceDataRingUploadVersions_;
  std::pmr::vector<uint64_t> instanceRemapUploadSignatures_;
  std::pmr::vector<uint64_t> shadowDrawPacketUploadSignatures_;
  std::pmr::vector<uint64_t> shadowFrameUploadSignatures_;
  std::pmr::vector<uint64_t> sdsmReduceResultRingPublishedFrames_;
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
  std::pmr::vector<StaticShadowCasterLightSpaceBounds>
      staticShadowCasterLightSpaceBounds_;
  std::pmr::vector<StaticShadowCasterLightSpaceBounds>
      staticShadowBatchLightSpaceBounds_;
  std::pmr::vector<StaticShadowCasterLightGridCell>
      staticShadowCasterLightGridCells_;
  std::pmr::vector<uint32_t> staticShadowCasterLightGridEntries_;
  std::pmr::vector<uint32_t> staticShadowCasterLargeLightGridEntries_;
  std::pmr::vector<StaticShadowCasterLightGridCell>
      staticShadowBatchLightGridCells_;
  std::pmr::vector<uint32_t> staticShadowBatchLightGridEntries_;
  std::pmr::vector<uint32_t> staticShadowBatchLargeLightGridEntries_;
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
  std::array<uint32_t, kMaxShadowCascades> cascadeDrawCounts_{};
  std::array<uint32_t, kMaxShadowCascades> cascadeCulledCounts_{};
  std::array<uint32_t, kMaxShadowCascades> cascadeDynamicDrawCounts_{};
  std::array<uint64_t, kMaxShadowCascades> cascadeIndexCountEstimates_{};
  std::array<uint64_t, kMaxShadowCascades>
      staticOnlyCascadeContentSignatures_{};
  std::array<uint64_t, kMaxShadowCascades>
      reusableStaticOnlyCascadeContentSignatures_{};
  std::array<StaticOnlyCascadeReuseState, kMaxShadowCascades>
      reusableStaticOnlyCascadeStates_{};
  std::array<bool, kMaxShadowCascades> reuseStaticOnlyCascadePass_{};
  std::array<bool, kMaxShadowCascades> reusableStaticOnlyCascadeValid_{};
  std::pmr::vector<BufferDependency> passBufferDependencies_;
  std::pmr::vector<BufferHandle> passDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode> passDependencyBufferAccessModes_;
  std::pmr::vector<RenderGraphPreparedDependencyBufferBinding>
      passDependencyBufferBindings_;
  std::pmr::vector<RenderGraphPreparedDependencyTextureBinding>
      passDependencyTextureBindings_;
  std::pmr::vector<BufferHandle> preResolvedDrawBuffers_;
  std::pmr::vector<RenderGraphBufferId> preResolvedDrawBufferIds_;
  std::pmr::vector<TextureDependency> passTextureDependencies_;
  std::pmr::vector<TextureDependency> previewTextureDependencies_;

  ShaderHandle shadowVertexShader_{};
  ShaderHandle shadowOpaqueVertexShader_{};
  ShaderHandle depthFragmentShader_{};
  ShaderHandle depthAlphaFragmentShader_{};
  ShaderHandle sdsmReduceComputeShader_{};
  ShaderHandle previewVertexShader_{};
  ShaderHandle previewFragmentShader_{};
  RenderPipelineHandle shadowPipelineHandle_{};
  RenderPipelineHandle shadowDoubleSidedPipelineHandle_{};
  RenderPipelineHandle shadowAlphaPipelineHandle_{};
  RenderPipelineHandle shadowAlphaDoubleSidedPipelineHandle_{};
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
  StaticShadowCasterLightGrid staticShadowCasterLightGrid_{};
  StaticShadowCasterLightGrid staticShadowBatchLightGrid_{};
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

} // namespace nuri
