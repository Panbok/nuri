#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/shader.h"

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace nuri {

class NURI_API SpatialAAPass final : public RenderFeaturePass {
public:
  explicit SpatialAAPass(GPUDevice &gpu, RuntimeCompositeConfig config);
  ~SpatialAAPass() override;

  SpatialAAPass(const SpatialAAPass &) = delete;
  SpatialAAPass &operator=(const SpatialAAPass &) = delete;
  SpatialAAPass(SpatialAAPass &&) = delete;
  SpatialAAPass &operator=(SpatialAAPass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "SpatialAAPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  struct FullscreenResources {
    std::unique_ptr<Shader> shader{};
    ShaderHandle vertexShader{};
    ShaderHandle fragmentShader{};
    RenderPipelineHandle pipeline{};
    Format pipelineColorFormat = Format::Count;
    std::filesystem::path vertexPath{};
    std::filesystem::path fragmentPath{};
    bool initialized = false;
  };

  GPUDevice &gpu_;
  RuntimeCompositeConfig config_{};
  FullscreenResources edgeResources_{};
  FullscreenResources blendResources_{};
  FullscreenResources neighborhoodResources_{};
  SamplerHandle linearClampSampler_{};
  SamplerHandle pointClampSampler_{};
  TextureHandle areaLutTexture_{};
  TextureHandle searchLutTexture_{};
  std::vector<TextureHandle> edgeTextures_{};
  std::vector<TextureHandle> blendTextures_{};
  uint32_t scratchWidth_ = 0u;
  uint32_t scratchHeight_ = 0u;
  uint32_t scratchRingCount_ = 0u;

  Result<bool, std::string> ensureResources(FrameBuildContext &ctx);
  void destroyResources();
  void destroyFullscreenResources(FullscreenResources &resources);
};

class NURI_API SpatialAAFeature final : public RenderFeature {
public:
  explicit SpatialAAFeature(GPUDevice &gpu, RuntimeCompositeConfig config);
  ~SpatialAAFeature() override = default;

  SpatialAAFeature(const SpatialAAFeature &) = delete;
  SpatialAAFeature &operator=(const SpatialAAFeature &) = delete;
  SpatialAAFeature(SpatialAAFeature &&) = delete;
  SpatialAAFeature &operator=(SpatialAAFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "SpatialAAFeature";
  }
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  SpatialAAPass spatialPass_;
  std::array<RenderFeaturePass *, 1> passes_{&spatialPass_};
};

} // namespace nuri
