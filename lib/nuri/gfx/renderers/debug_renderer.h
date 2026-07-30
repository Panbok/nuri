#pragma once
#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/pipeline/owned_program_bundle.h"
#include <array>
#include <glm/glm.hpp>
#include <memory>
#include <memory_resource>
#include <string>
#include <vector>
namespace nuri {

struct DebugRendererConfig {
  RuntimeRasterShaderConfig grid{};
  RuntimeDDGIShaderConfig ddgi{};
};

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
  void onFrameSubmitted(SubmissionHandle submission) noexcept;
  void onFrameAbandoned() noexcept;

private:
  struct GridPushConstants {
    glm::mat4 mvp{1.0f};
    glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 origin{0.0f, 0.0f, 0.0f, 0.0f};
  };
  struct DDGIProbePushConstants {
    glm::mat4 viewProjection{1.0f};
    glm::vec4 cameraRightAndScale{1.0f, 0.0f, 0.0f, 0.1f};
    glm::vec4 cameraUpAndDebugView{0.0f, 1.0f, 0.0f, 0.0f};
    uint64_t frameAddress = 0u;
    uint32_t volumeSlot = 0u;
    uint32_t submittedSequence = 0u;
  };
  static_assert(sizeof(DDGIProbePushConstants) == 112u);
  struct DDGIRayPushConstants {
    glm::mat4 viewProjection{1.0f};
    uint64_t diagnosticAddress = 0u;
    uint32_t rayCount = 0u;
    uint32_t reserved = 0u;
  };
  static_assert(sizeof(DDGIRayPushConstants) == 80u);
  [[nodiscard]] Result<bool, std::string>
  ensureGridPipeline(Format colorFormat, Format depthFormat);
  [[nodiscard]] Result<bool, std::string>
  prepareGridDraw(const RenderFrameContext &frame, TextureHandle depthTexture);
  [[nodiscard]] Result<bool, std::string>
  ensureDDGIProbePipeline(Format colorFormat, Format depthFormat);
  [[nodiscard]] Result<bool, std::string>
  prepareDDGIProbeDraws(const RenderFrameContext &frame,
                        TextureHandle depthTexture);
  [[nodiscard]] Result<bool, std::string>
  ensureDDGIRayPipeline(Format colorFormat, Format depthFormat);
  [[nodiscard]] Result<bool, std::string>
  prepareDDGIRayDraw(const RenderFrameContext &frame,
                     TextureHandle depthTexture);
  [[nodiscard]] bool hasDebugWork(const RenderFrameContext &frame) const;
  [[nodiscard]] bool buildSceneDebugLines(const RenderFrameContext &frame,
                                          TextureHandle depthTexture,
                                          float &outSortDepth);
  GPUDevice &gpu_;
  DebugRendererConfig config_{};
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::unique_ptr<DebugDraw3D> debugDraw3D_;
  OwnedProgramBundle gridProgram_{};
  OwnedProgramBundle ddgiProbeProgram_{};
  OwnedProgramBundle ddgiRayProgram_{};
  Format gridPipelineColorFormat_ = Format::Count;
  Format gridPipelineDepthFormat_ = Format::Count;
  Format ddgiProbePipelineColorFormat_ = Format::Count;
  Format ddgiProbePipelineDepthFormat_ = Format::Count;
  Format ddgiRayPipelineColorFormat_ = Format::Count;
  Format ddgiRayPipelineDepthFormat_ = Format::Count;
  GridPushConstants gridPushConstants_{};
  DrawItem gridDrawItem_{};
  std::array<DDGIProbePushConstants, kMaxDDGIVolumes> ddgiProbePushConstants_{};
  std::array<DrawItem, kMaxDDGIVolumes> ddgiProbeDrawItems_{};
  uint32_t ddgiProbeDrawCount_ = 0u;
  DDGIRayPushConstants ddgiRayPushConstants_{};
  DrawItem ddgiRayDrawItem_{};
  std::pmr::vector<TransparentStageSortableDraw> transparentSortableDraws_;
  std::pmr::vector<DrawItem> transparentFixedDraws_;
  std::pmr::vector<BufferHandle> transparentDependencyBuffers_;
};

NURI_API void registerDebugStages(
    RenderPipeline &pipeline, GPUDevice &gpu, DebugRendererConfig config,
    std::pmr::memory_resource *memory = std::pmr::get_default_resource());

} // namespace nuri
