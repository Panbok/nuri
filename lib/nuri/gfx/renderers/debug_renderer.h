#pragma once
#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/owned_gpu_resource.h"
#include <glm/glm.hpp>
#include <memory>
#include <memory_resource>
#include <string>
#include <vector>
namespace nuri {

using DebugRendererConfig = RuntimeRasterShaderConfig;

class DebugDraw3D;
class GPUDevice;
class RenderPipeline;

class NURI_API DebugRenderer {
public:
  explicit DebugRenderer(
      GPUDevice &gpu, DebugRendererConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~DebugRenderer();
  Result<bool, std::string>
  buildTransparentStageContribution(RenderFrameContext &frame,
                                    TransparentStageContribution &out);

private:
  struct GridPushConstants {
    glm::mat4 mvp{1.0f};
    glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 origin{0.0f, 0.0f, 0.0f, 0.0f};
  };
  [[nodiscard]] Result<bool, std::string> createGridShaders();
  [[nodiscard]] Result<bool, std::string>
  ensureGridPipeline(Format colorFormat, Format depthFormat);
  [[nodiscard]] Result<bool, std::string>
  prepareGridDraw(const RenderFrameContext &frame, TextureHandle depthTexture);
  [[nodiscard]] bool hasDebugWork(const RenderFrameContext &frame) const;
  [[nodiscard]] bool buildSceneDebugLines(const RenderFrameContext &frame,
                                          TextureHandle depthTexture,
                                          float &outSortDepth);
  GPUDevice &gpu_;
  DebugRendererConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<DebugDraw3D> debugDraw3D_;
  OwnedRenderPipelineHandle gridPipeline_;
  ShaderHandle gridVertexShader_{};
  ShaderHandle gridFragmentShader_{};
  Format gridPipelineColorFormat_ = Format::Count;
  Format gridPipelineDepthFormat_ = Format::Count;
  GridPushConstants gridPushConstants_{};
  DrawItem gridDrawItem_{};
  std::pmr::vector<TransparentStageSortableDraw> transparentSortableDraws_;
  std::pmr::vector<DrawItem> transparentFixedDraws_;
  std::pmr::vector<BufferHandle> transparentDependencyBuffers_;
};

NURI_API void registerDebugStages(
    RenderPipeline &pipeline, GPUDevice &gpu, RuntimeRasterShaderConfig config,
    std::pmr::memory_resource *memory = std::pmr::get_default_resource());

} // namespace nuri
