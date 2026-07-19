#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>
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
  void observeTemporalPolicy(
      const RenderSettings::AmbientOcclusionSettings &ao) noexcept;
  static constexpr uint32_t kViewDepthMipCount = 5u;

private:
  static constexpr size_t kScratchEdges = kViewDepthMipCount;
  static constexpr size_t kScratchRawAmbientOcclusion = kScratchEdges + 1u;
  static constexpr size_t kScratchDenoise = kScratchEdges + 2u;
  static constexpr size_t kScratchTextureCount = kScratchEdges + 3u;
  struct FrameScratchTextures {
    std::array<TextureHandle, kScratchTextureCount> textures{};
  };
  static constexpr std::size_t kPushConstantBufferSize = 128u;
  GPUDevice &gpu_;
  RuntimeOpaqueShaderConfig config_{};
  std::array<ShaderHandle, 5> shaders_{};
  std::array<ComputePipelineHandle, 5> pipelines_{};
  std::array<SamplerHandle, 2> samplers_{};
  std::string initializationError_{};
  std::vector<FrameScratchTextures> scratchTextures_{};
  uint32_t scratchWidth_ = 0u;
  uint32_t scratchHeight_ = 0u;
  uint32_t scratchRingCount_ = 0u;
  uint64_t lastTemporalPolicySignature_ = 0u;
  bool hasLastTemporalPolicySignature_ = false;
  bool temporalPolicyChanged_ = false;
  std::array<TextureHandle, 8> depthPrefilterDependencies_{};
  std::array<RenderGraphAccessMode, 8> depthPrefilterAccessModes_{};
  std::array<ComputeDispatchItem, 1> depthPrefilterDispatches_{};
  std::array<TextureHandle, 3> edgeDependencies_{};
  std::array<RenderGraphAccessMode, 3> edgeAccessModes_{};
  std::array<ComputeDispatchItem, 1> edgeDispatches_{};
  std::array<TextureHandle, 8> mainDependencies_{};
  std::array<RenderGraphAccessMode, 8> mainAccessModes_{};
  std::array<ComputeDispatchItem, 1> mainDispatches_{};
  std::array<TextureHandle, 3> denoiseDependencies_{};
  std::array<RenderGraphAccessMode, 3> denoiseAccessModes_{};
  std::array<ComputeDispatchItem, 1> denoiseDispatches_{};
  std::array<TextureHandle, 9> temporalDependencies_{};
  std::array<RenderGraphAccessMode, 9> temporalAccessModes_{};
  std::array<ComputeDispatchItem, 1> temporalDispatches_{};
  std::array<std::byte, kPushConstantBufferSize> depthPrefilterPushBytes_{};
  std::array<std::byte, kPushConstantBufferSize> edgePushBytes_{};
  std::array<std::byte, kPushConstantBufferSize> mainPushBytes_{};
  std::array<std::byte, kPushConstantBufferSize> denoisePushBytes_{};
  std::array<std::byte, kPushConstantBufferSize> temporalPushBytes_{};
  Result<bool, std::string> initialize();
  Result<bool, std::string> ensureScratchTextures(FrameBuildContext &ctx);
  Result<bool, std::string>
  recreateScratchTextures(uint32_t width, uint32_t height, uint32_t ringCount);
  void destroyResources();
  void destroyScratchTextures();
  [[nodiscard]] FrameScratchTextures &
  currentScratch(uint64_t frameIndex) noexcept;
};

NURI_API void registerGTAOStage(RenderPipeline &pipeline, GPUDevice &gpu,
                                RuntimeOpaqueShaderConfig config);

} // namespace nuri
