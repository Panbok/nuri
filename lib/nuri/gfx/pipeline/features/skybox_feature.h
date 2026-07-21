#pragma once
#include "nuri/core/result.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/owned_gpu_resource.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <vector>
namespace nuri {

class RenderPipeline;

using SkyboxFeatureConfig = RuntimeRasterShaderConfig;

class NURI_API SkyboxPass final {
public:
  explicit SkyboxPass(GPUDevice &gpu, const SkyboxFeatureConfig &config);
  ~SkyboxPass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);

private:
  struct FrameBufferSlot {
    BufferHandle buffer{};
    size_t capacityBytes = 0u;
  };
  enum FrameDataFlags : uint32_t {
    HasSceneColor = 1u << 5u,
  };
  using FrameData = ForwardSceneFrameData;
  static_assert(sizeof(FrameData) == 488,
                "SkyboxPass::FrameData must match shader FrameDataBuffer "
                "layout");
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
    float tessMaxFactor = 1.0f;
    uint32_t debugVisualizationMode = 0;
    uint32_t shadowCascadeIndex = 0;
    uint32_t reservedAfterShadowCascadeIndex = 0;
  };
  static_assert(sizeof(PushConstants) == 128,
                "SkyboxPass::PushConstants must match shader layout");
  static_assert(offsetof(PushConstants, instanceRemapAddress) == 40u);
  static_assert(offsetof(PushConstants, instanceCentersPhaseAddress) == 48u);
  static_assert(offsetof(PushConstants, instanceBaseMatricesAddress) == 56u);
  static_assert(offsetof(PushConstants, instanceCount) == 80u);
  static_assert(offsetof(PushConstants, shadowCascadeIndex) == 120u);
  Result<bool, std::string> ensureInitialized();
  void syncFrameBufferSlots();
  Result<bool, std::string> ensureFrameBufferCapacity(FrameBufferSlot &slot,
                                                      size_t requiredBytes);
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPipeline();
  Result<bool, std::string> prepareSkyboxDraw(FrameBuildContext &ctx);
  void destroyFrameBuffers();
  GPUDevice &gpu_;
  SkyboxFeatureConfig config_{};
  std::unique_ptr<Shader> skyboxShader_;
  std::array<OwnedRenderPipelineHandle, kCoverageModeCount> skyboxPipelines_;
  std::array<OwnedRenderPipelineHandle, kCoverageModeCount>
      skyboxDepthPipelines_;
  std::vector<FrameBufferSlot> frameBufferSlots_;
  BufferHandle preparedFrameBuffer_{};
  ShaderHandle skyboxVertexShader_{};
  ShaderHandle skyboxFragmentShader_{};
  bool initialized_ = false;
  FrameData frameData_{};
  PushConstants pushConstants_{};
  DrawItem drawItem_{};
  bool hasPreparedDraw_ = false;
};

NURI_API void registerSkyboxStage(RenderPipeline &pipeline, GPUDevice &gpu,
                                  SkyboxFeatureConfig config);

} // namespace nuri
