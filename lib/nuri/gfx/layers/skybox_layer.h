#pragma once

#include "nuri/core/layer.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <glm/glm.hpp>

namespace nuri {

using SkyboxLayerConfig = RuntimeSkyboxShaderConfig;

class NURI_API SkyboxLayer final : public Layer {
public:
  explicit SkyboxLayer(GPUDevice &gpu, SkyboxLayerConfig config);
  ~SkyboxLayer() override;

  SkyboxLayer(const SkyboxLayer &) = delete;
  SkyboxLayer &operator=(const SkyboxLayer &) = delete;
  SkyboxLayer(SkyboxLayer &&) = delete;
  SkyboxLayer &operator=(SkyboxLayer &&) = delete;

  static std::unique_ptr<SkyboxLayer> create(GPUDevice &gpu,
                                             SkyboxLayerConfig config) {
    return std::make_unique<SkyboxLayer>(gpu, std::move(config));
  }

  void onAttach() override;
  void onDetach() override;
  void onResize(int32_t width, int32_t height) override;
  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  enum FrameDataFlags : uint32_t {
    HasIblDiffuse = 1u << 0u,
    HasIblSpecular = 1u << 1u,
    HasIblSheen = 1u << 2u,
    HasBrdfLut = 1u << 3u,
    OutputLinearToSrgb = 1u << 4u,
  };

  struct FrameData {
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
    uint32_t cubemapTexId = 0;
    uint32_t hasCubemap = 0;
    uint32_t irradianceTexId = 0;
    uint32_t prefilteredGgxTexId = 0;
    uint32_t prefilteredCharlieTexId = 0;
    uint32_t brdfLutTexId = 0;
    uint32_t flags = 0;
    uint32_t cubemapSamplerId = 0;
  };
  static_assert(sizeof(FrameData) == 176,
                "SkyboxLayer::FrameData must match shader FrameDataBuffer "
                "layout");

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
  };

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> ensureFrameBufferCapacity(size_t requiredBytes);
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPipeline();
  void destroyFrameBuffer();

  GPUDevice &gpu_;
  SkyboxLayerConfig config_{};
  std::unique_ptr<Shader> skyboxShader_;
  std::unique_ptr<Pipeline> skyboxPipeline_;
  std::unique_ptr<Buffer> frameBuffer_;

  ShaderHandle skyboxVertexShader_{};
  ShaderHandle skyboxFragmentShader_{};
  RenderPipelineHandle skyboxPipelineHandle_{};

  size_t frameBufferCapacityBytes_ = 0;
  bool initialized_ = false;

  FrameData frameData_{};
  PushConstants pushConstants_{};
  DrawItem drawItem_{};
};

} // namespace nuri
