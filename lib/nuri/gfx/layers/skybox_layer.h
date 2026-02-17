#pragma once

#include "nuri/core/layer.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include <glm/glm.hpp>

namespace nuri {

class NURI_API SkyboxLayer final : public Layer {
public:
  explicit SkyboxLayer(GPUDevice &gpu);
  ~SkyboxLayer() override;

  SkyboxLayer(const SkyboxLayer &) = delete;
  SkyboxLayer &operator=(const SkyboxLayer &) = delete;
  SkyboxLayer(SkyboxLayer &&) = delete;
  SkyboxLayer &operator=(SkyboxLayer &&) = delete;

  static std::unique_ptr<SkyboxLayer> create(GPUDevice &gpu) {
    return std::make_unique<SkyboxLayer>(gpu);
  }

  void onAttach() override;
  void onDetach() override;
  void onResize(int32_t width, int32_t height) override;
  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  struct PerFrameData {
    glm::mat4 model{1.0f};
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
    uint32_t albedoTexId = 0;
    uint32_t cubemapTexId = 0;
    uint32_t hasCubemap = 0;
    uint32_t _padding0 = 0;
  };

  struct PushConstants {
    uint64_t perFrameAddress = 0;
    uint64_t vertexBufferAddress = 0;
  };

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> ensurePerFrameBufferCapacity(size_t requiredBytes);
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPipeline();
  void destroyPerFrameBuffer();

  GPUDevice &gpu_;
  std::unique_ptr<Shader> skyboxShader_;
  std::unique_ptr<Pipeline> skyboxPipeline_;
  std::unique_ptr<Buffer> perFrameBuffer_;

  ShaderHandle skyboxVertexShader_{};
  ShaderHandle skyboxFragmentShader_{};
  RenderPipelineHandle skyboxPipelineHandle_{};

  size_t perFrameBufferCapacityBytes_ = 0;
  bool initialized_ = false;

  PerFrameData perFrameData_{};
  PushConstants pushConstants_{};
  DrawItem drawItem_{};
};

} // namespace nuri
