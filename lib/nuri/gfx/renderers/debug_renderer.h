#pragma once

#include <memory>
#include <memory_resource>
#include <utility>

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/render_graph/render_graph.h"

#include <glm/glm.hpp>

namespace nuri {

using DebugRendererConfig = RuntimeDebugShaderConfig;

class DebugDraw3D;
class GPUDevice;
class Pipeline;
class Shader;

class NURI_API DebugRenderer {
public:
  explicit DebugRenderer(
      GPUDevice &gpu, DebugRendererConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~DebugRenderer();

  DebugRenderer(const DebugRenderer &) = delete;
  DebugRenderer &operator=(const DebugRenderer &) = delete;
  DebugRenderer(DebugRenderer &&) = delete;
  DebugRenderer &operator=(DebugRenderer &&) = delete;

  void onDetach();
  Result<bool, std::string> prepareDebugPasses(RenderFrameContext &frame);
  [[nodiscard]] bool hasPreparedDebugGridPass() const noexcept;
  [[nodiscard]] bool hasPreparedDebugSceneOverlayPass() const noexcept;
  Result<bool, std::string> appendDebugGridPass(RenderFrameContext &frame,
                                                RenderGraphBuilder &graph);
  Result<bool, std::string>
  appendDebugSceneOverlayPass(RenderFrameContext &frame,
                              RenderGraphBuilder &graph);
  Result<bool, std::string>
  buildTransparentStageContribution(RenderFrameContext &frame,
                                    TransparentStageContribution &out);

private:
  struct GridPushConstants {
    glm::mat4 mvp{1.0f};
    glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 origin{0.0f, 0.0f, 0.0f, 0.0f};
  };

  [[nodiscard]] Result<bool, std::string> ensureGridInitialized();
  [[nodiscard]] Result<bool, std::string> createGridShaders();
  [[nodiscard]] Result<bool, std::string>
  ensureGridPipeline(Format colorFormat, Format depthFormat);
  [[nodiscard]] Result<bool, std::string>
  prepareGridDraw(const RenderFrameContext &frame, TextureHandle depthTexture);
  [[nodiscard]] Result<bool, std::string>
  appendModelBoundsGraphPass(const RenderFrameContext &frame,
                             RenderGraphBuilder &graph,
                             TextureHandle sceneDepthTexture,
                             RenderGraphTextureId sceneDepthGraphTexture);
  [[nodiscard]] bool hasDebugWork(const RenderFrameContext &frame) const;
  [[nodiscard]] Result<bool, std::string>
  buildSceneDebugLines(const RenderFrameContext &frame,
                       TextureHandle depthTexture, float &outSortDepth);
  void resetGridState();

  GPUDevice &gpu_;
  DebugRendererConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<DebugDraw3D> debugDraw3D_;
  std::unique_ptr<Shader> gridShader_;
  std::unique_ptr<Pipeline> gridPipeline_;

  ShaderHandle gridVertexShader_{};
  ShaderHandle gridFragmentShader_{};
  RenderPipelineHandle gridPipelineHandle_{};

  Format gridPipelineColorFormat_ = Format::Count;
  Format gridPipelineDepthFormat_ = Format::Count;

  TextureHandle preparedSceneDepthTexture_{};
  TextureHandle preparedFrameColorTexture_{};
  RenderGraphTextureId preparedSceneDepthGraphTexture_{};
  bool preparedHasPriorColorPass_ = false;
  bool preparedGridPass_ = false;
  bool preparedSceneOverlayPass_ = false;

  GridPushConstants gridPushConstants_{};
  DrawItem gridDrawItem_{};
  std::pmr::vector<TransparentStageSortableDraw> transparentSortableDraws_;
  std::pmr::vector<DrawItem> transparentFixedDraws_;
  std::pmr::vector<BufferHandle> transparentDependencyBuffers_;
};

} // namespace nuri
