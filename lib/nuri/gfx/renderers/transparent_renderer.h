#pragma once

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/scene/render_scene.h"

namespace nuri {

// Transparent rendering reuses RuntimeOpaqueShaderConfig because the shader
// path/layout inputs match the opaque pipeline configuration today.
using TransparentRendererConfig = RuntimeOpaqueShaderConfig;

class ResourceManager;
class Shader;

class NURI_API TransparentRenderer {
public:
  explicit TransparentRenderer(
      GPUDevice &gpu, TransparentRendererConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~TransparentRenderer();

  TransparentRenderer(const TransparentRenderer &) = delete;
  TransparentRenderer &operator=(const TransparentRenderer &) = delete;
  TransparentRenderer(TransparentRenderer &&) = delete;
  TransparentRenderer &operator=(TransparentRenderer &&) = delete;

  void onAttach();
  void onDetach();
  void publishFrameData(RenderFrameContext &frame);
  Result<bool, std::string> prepareTransparentPasses(RenderFrameContext &frame);
  Result<bool, std::string>
  appendTransparentMainPass(RenderFrameContext &frame,
                            RenderGraphBuilder &graph);
  Result<bool, std::string>
  appendTransparentPickPass(RenderFrameContext &frame,
                            RenderGraphBuilder &graph);

private:
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 368,
                "TransparentRenderer::FrameData must match shader layout");

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
                "TransparentRenderer::PushConstants must match shader layout");
  static_assert(offsetof(PushConstants, previousInstanceMatricesAddress) == 32u,
                "TransparentRenderer::PushConstants "
                "previousInstanceMatricesAddress offset changed");
  static_assert(offsetof(PushConstants, instanceRemapAddress) == 40u);
  static_assert(offsetof(PushConstants, instanceCentersPhaseAddress) == 48u);
  static_assert(offsetof(PushConstants, instanceBaseMatricesAddress) == 56u);
  static_assert(offsetof(PushConstants, velocityInstanceFlagsAddress) == 64u,
                "TransparentRenderer::PushConstants "
                "velocityInstanceFlagsAddress offset changed");
  static_assert(offsetof(PushConstants, velocityFrameDataAddress) == 72u,
                "TransparentRenderer::PushConstants "
                "velocityFrameDataAddress offset changed");
  static_assert(offsetof(PushConstants, instanceCount) == 80u);
  static_assert(offsetof(PushConstants, shadowCascadeIndex) == 120u);

  struct MeshDrawTemplate {
    // These pointers reference scene-owned topology and remain valid only
    // while RenderScene::topologyVersion() is unchanged.
    const Renderable *renderable = nullptr;
    const Submesh *submesh = nullptr;
    uint32_t submeshIndex = 0;
    BufferHandle indexBuffer{};
    uint64_t indexBufferOffset = 0;
    // Original geometry buffer; use as the source handle for dependencies and
    // for resolving sub-range addresses.
    BufferHandle baseVertexBuffer{};
    // Byte offset into baseVertexBuffer when the draw uses a sliced view.
    uint64_t vertexBufferByteOffset = 0;
    // Current draw buffer view, possibly replaced by animated geometry.
    BufferHandle vertexBuffer{};
    // Secondary static decode data used by packed/quantized vertex formats.
    BufferHandle baseVertexDecodeBuffer{};
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = kInvalidMaterialIndex;
    uint32_t instanceIndex = 0;
    bool doubleSided = false;
  };

  struct DynamicBufferSlot {
    std::unique_ptr<Buffer> buffer;
    size_t capacityBytes = 0;
  };

  struct FixedDrawEntry {
    DrawItem draw{};
    uint32_t dependencyOffset = 0;
    uint32_t dependencyCount = 0;
  };

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> createShaders();
  Result<bool, std::string> ensurePipelines(Format colorFormat,
                                            Format depthFormat);
  Result<bool, std::string> ensureFeedbackCopyPipeline(Format colorFormat);
  Result<bool, std::string> ensureRingBufferCount(uint32_t requiredCount);
  Result<bool, std::string>
  ensureInstanceMatricesRingCapacity(size_t requiredBytes);
  Result<bool, std::string>
  ensureInstanceRemapRingCapacity(size_t requiredBytes);
  Result<bool, std::string> rebuildSceneCache(const RenderScene &scene,
                                              const ResourceManager &resources,
                                              uint32_t materialCount,
                                              bool excludeTransmissionBlend);
  Result<bool, std::string>
  rebuildMaterialTextureAccessCache(const ResourceManager &resources);
  Result<bool, std::string> collectContributorDraws(RenderFrameContext &frame);
  Result<bool, std::string> appendTransparentPass(
      RenderFrameContext &frame, RenderGraphBuilder &graph,
      TextureHandle colorTexture, TextureHandle depthTexture,
      RenderGraphTextureId sceneDepthGraphTexture,
      std::span<const TransparentStageSortableDraw> sortableDraws,
      std::span<const FixedDrawEntry> fixedDraws,
      std::span<const TextureHandle> textureReads,
      std::span<const BufferHandle> dependencyBuffers);
  Result<bool, std::string> appendTransparentDrawRun(
      RenderGraphBuilder &graph, TextureHandle colorTexture,
      TextureHandle depthTexture, RenderGraphTextureId sceneDepthGraphTexture,
      std::span<const DrawItem> draws,
      std::span<const TextureHandle> textureReads,
      std::span<const BufferHandle> dependencyBuffers,
      std::string_view debugLabel);
  Result<bool, std::string>
  appendTransparentTransmissionFeedbackRefresh(RenderFrameContext &frame,
                                               RenderGraphBuilder &graph);
  void collectEnvironmentTextureReads(const RenderScene &scene,
                                      const ResourceManager &resources);
  void resetCachedState();
  void resetFrameBuildState();
  void destroyPipelineState();
  void destroyShaders();
  void destroyBuffers();
  static void
  sortTransparentDraws(std::span<TransparentStageSortableDraw> draws);
  [[nodiscard]] RenderPipelineHandle selectMeshPipeline(bool doubleSided) const;
  [[nodiscard]] RenderPipelineHandle selectPickPipeline(bool doubleSided) const;

  GPUDevice &gpu_;
  TransparentRendererConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<Shader> meshShader_;
  std::unique_ptr<Shader> meshPickShader_;
  std::unique_ptr<Shader> feedbackCopyShader_;
  std::pmr::vector<DynamicBufferSlot> instanceMatricesRing_;
  std::pmr::vector<DynamicBufferSlot> instanceRemapRing_;

  ShaderHandle meshVertexShader_{};
  ShaderHandle meshFragmentShader_{};
  ShaderHandle meshPickVertexShader_{};
  ShaderHandle meshPickFragmentShader_{};
  ShaderHandle feedbackCopyVertexShader_{};
  ShaderHandle feedbackCopyFragmentShader_{};
  RenderPipelineHandle meshPipelineHandle_{};
  RenderPipelineHandle meshDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshPickPipelineHandle_{};
  RenderPipelineHandle meshPickDoubleSidedPipelineHandle_{};
  RenderPipelineHandle feedbackCopyPipelineHandle_{};

  Format meshPipelineColorFormat_ = Format::Count;
  Format meshPipelineDepthFormat_ = Format::Count;
  // The pick pass writes a fixed object-ID color target, so only the depth
  // format needs to be cached here.
  Format pickPipelineDepthFormat_ = Format::Count;
  Format feedbackCopyPipelineColorFormat_ = Format::Count;

  bool initialized_ = false;
  bool loggedMaterialFallbackWarning_ = false;
  bool loggedTransmissionFeedbackFallbackWarning_ = false;
  bool transparentUsesJitteredProjection_ = true;
  uint32_t loggedContributorCollections_ = 0u;
  uint64_t loggedAddressProbeTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();

  const RenderScene *cachedScene_ = nullptr;
  uint64_t cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedGeometryMutationVersion_ =
      std::numeric_limits<uint64_t>::max();
  bool cachedExcludeTransmissionBlend_ = true;

  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<InstanceData> instanceMatrices_;
  std::pmr::vector<uint32_t> instanceRemap_;
  std::pmr::vector<uint64_t> instanceDataRingUploadVersions_;
  std::pmr::vector<TextureHandle> materialTextureAccessHandles_;
  std::pmr::vector<TextureHandle> environmentTextureAccessHandles_;
  std::pmr::vector<TransparentStageSortableDraw> contributorSortableDraws_;
  std::pmr::vector<FixedDrawEntry> contributorFixedDraws_;
  std::pmr::vector<TextureHandle> contributorTextureReads_;
  std::pmr::vector<BufferHandle> contributorDependencyBuffers_;
  std::pmr::vector<PushConstants> drawPushConstants_;
  std::pmr::vector<PushConstants> pickPushConstants_;
  std::pmr::vector<TransparentStageSortableDraw> meshSortableDraws_;
  std::pmr::vector<TransparentStageSortableDraw> sortableDraws_;
  std::pmr::vector<FixedDrawEntry> fixedDraws_;
  std::pmr::vector<DrawItem> passDrawItems_;
  std::pmr::vector<DrawItem> transparentRunDrawItems_;
  std::pmr::vector<BufferHandle> transparentRunDependencyBuffers_;
  std::pmr::vector<BufferHandle> transparentCandidateDependencyBuffers_;
  std::pmr::vector<DrawItem> pickDrawItems_;
  std::pmr::vector<TextureHandle> passTextureReads_;
  std::pmr::vector<BufferHandle> passDependencyBuffers_;
  std::filesystem::path alphaPickFragmentPath_{};
  std::filesystem::path feedbackCopyVertexPath_{};
  std::filesystem::path feedbackCopyFragmentPath_{};
};

} // namespace nuri
