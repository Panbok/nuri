#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include <array>
#include <cstddef>
#include <string>
namespace nuri {

class RenderPipeline;

class NURI_API GTAOPass final {
public:
  explicit GTAOPass(GPUDevice &gpu, RuntimeOpaqueShaderConfig config);
  ~GTAOPass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx);
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);
  void observeTemporalPolicy(const AmbientOcclusionExecutionPlan &plan,
                             float strength) noexcept;
  static constexpr uint32_t kViewDepthMipCount = 5u;

private:
  GPUDevice &gpu_;
  RuntimeOpaqueShaderConfig config_{};
  std::array<ShaderHandle, 6> shaders_{};
  std::array<ComputePipelineHandle, 6> pipelines_{};
  std::array<SamplerHandle, 2> samplers_{};
  std::string initializationError_{};
  uint64_t lastTemporalPolicySignature_ = 0u;
  bool hasLastTemporalPolicySignature_ = false;
  bool temporalPolicyChanged_ = false;
  TextureHandle reconstructedNormalDebugTexture_{};
  std::array<TextureHandle, 2> captureTextures_{};
  Result<bool, std::string> initialize();
  Result<bool, std::string> ensureResources(FrameBuildContext &ctx);
  void destroyResources();
};

NURI_API void registerGTAOStage(RenderPipeline &pipeline, GPUDevice &gpu,
                                RuntimeOpaqueShaderConfig config);

} // namespace nuri
