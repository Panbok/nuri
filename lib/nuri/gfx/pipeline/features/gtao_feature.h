#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/shader.h"

#include <array>
#include <memory>
#include <span>
#include <vector>

namespace nuri {

class NURI_API GTAOPass final : public RenderFeaturePass {
public:
  explicit GTAOPass(GPUDevice &gpu, RuntimeOpaqueShaderConfig config);
  ~GTAOPass() override;

  GTAOPass(const GTAOPass &) = delete;
  GTAOPass &operator=(const GTAOPass &) = delete;
  GTAOPass(GTAOPass &&) = delete;
  GTAOPass &operator=(GTAOPass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "GTAOPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

  static constexpr uint32_t kViewDepthMipCount = 5u;

private:
  struct FrameScratchTextures {
    std::array<TextureHandle, kViewDepthMipCount> viewDepthMips{};
    TextureHandle rawAmbientOcclusion{};
    TextureHandle denoiseScratch{};
  };

  GPUDevice &gpu_;
  RuntimeOpaqueShaderConfig config_{};
  std::unique_ptr<Shader> depthPrefilterShader_{};
  std::unique_ptr<Shader> mainShader_{};
  std::unique_ptr<Shader> denoiseShader_{};
  ShaderHandle depthPrefilterShaderHandle_{};
  ShaderHandle mainShaderHandle_{};
  ShaderHandle denoiseShaderHandle_{};
  ComputePipelineHandle depthPrefilterPipeline_{};
  ComputePipelineHandle mainPipeline_{};
  ComputePipelineHandle denoisePipeline_{};
  SamplerHandle pointClampSampler_{};
  std::vector<FrameScratchTextures> scratchTextures_{};
  uint32_t scratchWidth_ = 0u;
  uint32_t scratchHeight_ = 0u;
  uint32_t scratchRingCount_ = 0u;

  std::array<TextureHandle, 8> depthPrefilterDependencies_{};
  std::array<RenderGraphAccessMode, 8> depthPrefilterAccessModes_{};
  std::array<ComputeDispatchItem, kViewDepthMipCount>
      depthPrefilterDispatches_{};
  std::array<TextureHandle, 8> mainDependencies_{};
  std::array<RenderGraphAccessMode, 8> mainAccessModes_{};
  std::array<ComputeDispatchItem, 1> mainDispatches_{};
  std::array<TextureHandle, 3> denoiseDependencies_{};
  std::array<RenderGraphAccessMode, 3> denoiseAccessModes_{};
  std::array<ComputeDispatchItem, 1> denoiseDispatches_{};
  std::array<std::array<std::byte, 128>, kViewDepthMipCount>
      depthPrefilterPushBytes_{};
  std::array<std::byte, 128> mainPushBytes_{};
  std::array<std::byte, 128> denoisePushBytes_{};

  Result<bool, std::string> ensureResources(FrameBuildContext &ctx);
  Result<bool, std::string>
  recreateScratchTextures(uint32_t width, uint32_t height, uint32_t ringCount);
  void destroyResources();
  void destroyScratchTextures();
  [[nodiscard]] FrameScratchTextures *
  currentScratch(uint64_t frameIndex) noexcept;
};

class NURI_API GTAOFeature final : public RenderFeature {
public:
  explicit GTAOFeature(GPUDevice &gpu, RuntimeOpaqueShaderConfig config);
  ~GTAOFeature() override = default;

  GTAOFeature(const GTAOFeature &) = delete;
  GTAOFeature &operator=(const GTAOFeature &) = delete;
  GTAOFeature(GTAOFeature &&) = delete;
  GTAOFeature &operator=(GTAOFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "GTAOFeature";
  }
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  GTAOPass pass_;
  std::array<RenderFeaturePass *, 1> passes_{&pass_};
};

} // namespace nuri
