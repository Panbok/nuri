#pragma once

#include "nuri/core/layer.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/layers/render_frame_context.h"
#include "nuri/resources/gpu/buffer.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>

namespace nuri {

using CompositeLayerConfig = RuntimeOpaqueShaderConfig;

class Shader;

class NURI_API CompositeLayer final : public Layer {
public:
  explicit CompositeLayer(GPUDevice &gpu, CompositeLayerConfig config);
  ~CompositeLayer() override;

  CompositeLayer(const CompositeLayer &) = delete;
  CompositeLayer &operator=(const CompositeLayer &) = delete;
  CompositeLayer(CompositeLayer &&) = delete;
  CompositeLayer &operator=(CompositeLayer &&) = delete;

  static std::unique_ptr<CompositeLayer> create(GPUDevice &gpu,
                                                CompositeLayerConfig config) {
    return std::make_unique<CompositeLayer>(gpu, std::move(config));
  }

  void onAttach() override;
  void onDetach() override;
  Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &frame,
                   RenderGraphBuilder &graph) override;

private:
  struct PushConstants {
    uint64_t frameDataAddress = 0;
    uint64_t vertexBufferAddress = 0;
    uint64_t instanceMatricesAddress = 0;
    uint64_t instanceRemapAddress = 0;
    uint64_t materialBufferAddress = 0;
    uint64_t instanceCentersPhaseAddress = 0;
    uint64_t instanceBaseMatricesAddress = 0;
    uint32_t instanceCount = 0;
    uint32_t materialIndex = 0;
    float timeSeconds = 0.0f;
    float tessNearDistance = 1.0f;
    float tessFarDistance = 8.0f;
    float tessMinFactor = 1.0f;
    float tessMaxFactor = 1.0f;
    uint32_t debugVisualizationMode = 0;
  };
  static_assert(sizeof(PushConstants) <= 128,
                "CompositeLayer::PushConstants exceeds Vulkan guarantee");

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> createShaders();
  Result<bool, std::string> ensurePipeline();
  Result<bool, std::string> ensureFrameBufferCapacity(size_t requiredBytes);
  void destroyPipelineState();
  void destroyShaders();
  void destroyBuffers();

  GPUDevice &gpu_;
  std::unique_ptr<Shader> shader_;
  std::unique_ptr<Buffer> frameBuffer_;

  ShaderHandle vertexShader_{};
  ShaderHandle fragmentShader_{};
  RenderPipelineHandle pipelineHandle_{};
  Format pipelineColorFormat_ = Format::Count;

  size_t frameBufferCapacityBytes_ = 0;
  bool initialized_ = false;
  bool frameDataUploadValid_ = false;

  ForwardSceneFrameData frameData_{};
  ForwardSceneFrameData uploadedFrameData_{};
  PushConstants pushConstants_{};
  DrawItem drawItem_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
};

} // namespace nuri
