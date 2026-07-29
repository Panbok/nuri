#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/dynamic_buffer.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/renderers/detail/forward_rendering.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/gfx/renderers/scene_draw_database.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/scene/render_scene.h"
#include <array>
#include <cstdint>
#include <filesystem>
#include <glm/glm.hpp>
#include <limits>
#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>
namespace nuri {

using TransmissionRendererConfig = RuntimeOpaqueShaderConfig;

class ResourceManager;
class RenderPipeline;
class ForwardInstanceBuffers;

class NURI_API TransmissionRenderer {
public:
  explicit TransmissionRenderer(
      GPUDevice &gpu, const TransmissionRendererConfig &config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~TransmissionRenderer();
  TransmissionRenderer(const TransmissionRenderer &) = delete;
  TransmissionRenderer &operator=(const TransmissionRenderer &) = delete;
  TransmissionRenderer(TransmissionRenderer &&) = delete;
  TransmissionRenderer &operator=(TransmissionRenderer &&) = delete;
  void onAttach();
  void onDetach();
  void publishFrameData(RenderFrameContext &frame);
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept;
  Result<bool, std::string>
  buildTransparentStageContribution(RenderFrameContext &frame,
                                    TransparentStageContribution &out);
  [[nodiscard]] bool hasPreparedTransmissionMainPass() const noexcept;
  Result<bool, std::string>
  prepareTransmissionPasses(RenderFrameContext &frame);
  Result<bool, std::string>
  appendTransmissionMainPass(RenderFrameContext &frame,
                             RenderGraphBuilder &graph);

private:
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 488,
                "TransmissionRenderer::FrameData must match shader layout");
  using MeshPushConstants = ForwardMeshPushConstants;
  struct MeshDrawTemplate : SceneDrawRecord {
    glm::vec3 transmissionScale{1.0f};
    std::array<glm::vec3, 8> worldBoundsCorners{};
    DrawItem cachedDrawItem{};
    uint64_t cachedDrawLayoutSignature = std::numeric_limits<uint64_t>::max();
    bool sortedFeedback = false;
    explicit MeshDrawTemplate(const SceneDrawRecord &draw)
        : SceneDrawRecord(draw),
          sortedFeedback(draw.sortedTransmissionFeedback) {}
  };
  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> createShaders();
  Result<bool, std::string> ensurePipelines(Format colorFormat,
                                            Format depthFormat,
                                            RasterPipelineState rasterState);
  Result<bool, std::string> ensureFeedbackCopyPipeline(Format colorFormat);
  [[nodiscard]] TransparentStageFeedbackRefreshMode
  selectTransparentFeedbackRefreshMode(uint32_t visibleDrawCount);
  Result<bool, std::string>
  appendTransparentFeedbackRefresh(RenderFrameContext &frame,
                                   RenderGraphBuilder &graph);
  Result<bool, std::string> applyTransparentFeedbackBindings(DrawItem &draw);
  void rebuildSceneCache(const SceneDrawDatabase &database,
                         const RenderScene &scene);
  void refreshDrawTemplateTransforms();
  void resetCachedState();
  void resetFrameBuildState();
  void destroyPipelineState();
  void destroyShaders();
  void destroyBuffers();
  [[nodiscard]] RenderPipelineHandle
  selectMeshPipeline(bool doubleSided, bool useBlendPipeline) const;
  GPUDevice &gpu_;
  TransmissionRendererConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<ForwardInstanceBuffers> instanceBuffers_;
  std::unique_ptr<DynamicBufferRing> blendedFrameDataRing_;
  std::array<ShaderHandle, 4> shaders_{};
  std::array<RenderPipelineHandle, 4> meshPipelines_{};
  RenderPipelineHandle feedbackCopyPipelineHandle_{};
  Format meshPipelineColorFormat_ = Format::Count;
  Format meshPipelineDepthFormat_ = Format::Count;
  RasterPipelineState meshPipelineRasterState_{};
  Format feedbackCopyPipelineColorFormat_ = Format::Count;
  bool initialized_ = false;
  bool loggedTransparentFeedbackFallbackWarning_ = false;
  ForwardSceneDrawCache sceneCache_;
  EnvironmentHandles cachedEnvironmentHandles_{};
  bool environmentTextureAccessCacheValid_ = false;
  bool materialTextureAccessCacheValid_ = false;
  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<TextureHandle> staticPassTextureReads_;
  std::pmr::vector<MeshPushConstants> meshPushConstants_;
  std::pmr::vector<MeshPushConstants> blendedPushConstants_;
  std::pmr::vector<DrawItem> passDrawItems_;
  std::pmr::vector<TransparentStageSortableDraw> blendedSortableDraws_;
  std::pmr::vector<const MeshDrawTemplate *> sortedDepthTemplates_;
  std::pmr::vector<TextureHandle> passTextureReads_;
  std::pmr::vector<TextureHandle> blendedTextureReads_;
  std::pmr::vector<BufferHandle> passDependencyBuffers_;
  std::pmr::vector<BufferHandle> blendedDependencyBuffers_;
  std::pmr::vector<RenderGraphAccessMode> passDependencyBufferAccessModes_;
  std::pmr::vector<BufferHandle> preResolvedTemplateBuffers_;
  std::pmr::vector<BufferHandle> cachedPreResolvedDrawBuffers_;
  uint64_t cachedPreResolvedDrawBufferSignature_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedPassTextureReadSignature_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t cachedBlendedTextureReadSignature_ =
      std::numeric_limits<uint64_t>::max();
  std::filesystem::path transmissionVertexPath_{};
  std::filesystem::path transmissionFragmentPath_{};
  std::filesystem::path feedbackCopyVertexPath_{};
  std::filesystem::path feedbackCopyFragmentPath_{};
  TextureHandle preparedSceneColorTexture_{};
  TextureHandle preparedSceneColorHalfResTexture_{};
  TextureHandle preparedSceneColorQuarterResTexture_{};
  TextureHandle preparedFrameColorTexture_{};
  TextureHandle preparedDepthTexture_{};
  RenderGraphTextureId preparedSceneDepthGraphTexture_{};
  std::array<RenderGraphTextureId, kFrameCompositionSceneColorMipCount>
      preparedTransparentFeedbackTextures_{};
  std::array<PushConstantTextureBinding, kFrameCompositionSceneColorMipCount>
      preparedTransparentFeedbackBindings_{};
  std::array<TextureHandle, kFrameCompositionSceneColorMipCount>
      transparentFeedbackCaptureTextures_{};
  uint32_t preparedTransparentFeedbackCandidateCount_ = 0u;
};

NURI_API void registerTransmissionStage(
    RenderPipeline &pipeline, GPUDevice &gpu,
    const RuntimeOpaqueShaderConfig &config,
    std::pmr::memory_resource *memory = std::pmr::get_default_resource(),
    SceneDrawDatabase *database = nullptr);

} // namespace nuri
