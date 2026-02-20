#pragma once

#include <memory>
#include <memory_resource>
#include <utility>

#include "nuri/core/layer.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"

#include <glm/glm.hpp>

namespace nuri {

using DebugLayerConfig = RuntimeDebugShaderConfig;

class DebugDraw3D;
class GPUDevice;
class Pipeline;
class Shader;

class NURI_API DebugLayer final : public Layer {
public:
  explicit DebugLayer(
      GPUDevice &gpu, DebugLayerConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~DebugLayer() override;

  DebugLayer(const DebugLayer &) = delete;
  DebugLayer &operator=(const DebugLayer &) = delete;
  DebugLayer(DebugLayer &&) = delete;
  DebugLayer &operator=(DebugLayer &&) = delete;

  static std::unique_ptr<DebugLayer>
  create(GPUDevice &gpu, DebugLayerConfig config,
         std::pmr::memory_resource *memory = std::pmr::get_default_resource()) {
    return std::make_unique<DebugLayer>(gpu, std::move(config), memory);
  }

  void onDetach() override;
  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  struct GridPushConstants {
    glm::mat4 mvp{1.0f};
    glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 origin{0.0f, 0.0f, 0.0f, 0.0f};
  };

  [[nodiscard]] Result<bool, std::string> ensureGridInitialized();
  [[nodiscard]] Result<bool, std::string> createGridShaders();
  [[nodiscard]] Result<bool, std::string> ensureGridPipeline(
      Format colorFormat, Format depthFormat);
  [[nodiscard]] Result<RenderPass, std::string>
  buildGridPass(const RenderFrameContext &frame, bool hasPriorColorPass);
  [[nodiscard]] Result<bool, std::string>
  appendModelBoundsPass(const RenderFrameContext &frame, RenderPassList &out);
  void resetGridState();

  GPUDevice &gpu_;
  DebugLayerConfig config_{};
  std::unique_ptr<DebugDraw3D> debugDraw3D_;
  std::unique_ptr<Shader> gridShader_;
  std::unique_ptr<Pipeline> gridPipeline_;

  ShaderHandle gridVertexShader_{};
  ShaderHandle gridFragmentShader_{};
  RenderPipelineHandle gridPipelineHandle_{};

  Format gridPipelineColorFormat_ = Format::Count;
  Format gridPipelineDepthFormat_ = Format::Count;

  GridPushConstants gridPushConstants_{};
  DrawItem gridDrawItem_{};
};

} // namespace nuri
