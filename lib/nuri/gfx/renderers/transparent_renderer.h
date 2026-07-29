#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
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

using TransparentRendererConfig = RuntimeOpaqueShaderConfig;

class ResourceManager;
class RenderPipeline;
class ForwardInstanceBuffers;

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
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept;
  Result<bool, std::string> prepareTransparentPasses(RenderFrameContext &frame);
  Result<bool, std::string>
  appendTransparentMainPass(RenderFrameContext &frame,
                            RenderGraphBuilder &graph);
  Result<bool, std::string>
  appendTransparentPickPass(RenderFrameContext &frame,
                            RenderGraphBuilder &graph);

private:
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 488,
                "TransparentRenderer::FrameData must match shader layout");
  using PushConstants = ForwardMeshPushConstants;
  using MeshDrawTemplate = SceneDrawRecord;
  struct FixedDrawEntry {
    DrawItem draw{};
    uint32_t dependencyOffset = 0;
    uint32_t dependencyCount = 0;
  };
  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> createShaders();
  Result<bool, std::string> ensurePipelines(Format colorFormat,
                                            Format depthFormat);
  void rebuildSceneCache(const SceneDrawDatabase &database,
                         const RenderScene &scene,
                         bool excludeTransmissionBlend);
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
  std::unique_ptr<ForwardInstanceBuffers> instanceBuffers_;
  std::array<ShaderHandle, 4> shaders_{};
  RenderPipelineHandle meshPipelineHandle_{};
  RenderPipelineHandle meshDoubleSidedPipelineHandle_{};
  RenderPipelineHandle meshPickPipelineHandle_{};
  RenderPipelineHandle meshPickDoubleSidedPipelineHandle_{};
  Format meshPipelineColorFormat_ = Format::Count;
  Format meshPipelineDepthFormat_ = Format::Count;
  Format pickPipelineDepthFormat_ = Format::Count;
  bool initialized_ = false;
  bool transparentUsesJitteredProjection_ = true;
  ForwardSceneDrawCache sceneCache_;
  bool cachedExcludeTransmissionBlend_ = true;
  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<TransparentStageSortableDraw> contributorSortableDraws_;
  std::pmr::vector<FixedDrawEntry> contributorFixedDraws_;
  std::pmr::vector<TextureHandle> contributorTextureReads_;
  std::pmr::vector<BufferHandle> contributorDependencyBuffers_;
  TransparentStageFeedbackRefresh feedbackRefresh_{};
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
};

NURI_API void registerTransparentStages(
    RenderPipeline &pipeline, GPUDevice &gpu, RuntimeOpaqueShaderConfig config,
    std::pmr::memory_resource *memory = std::pmr::get_default_resource(),
    SceneDrawDatabase *database = nullptr);

} // namespace nuri
