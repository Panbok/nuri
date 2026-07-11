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

namespace nuri {

class NURI_API ReferenceTAAResolvePass final : public RenderFeaturePass {
public:
  ReferenceTAAResolvePass(GPUDevice &gpu, RuntimeCompositeConfig config);
  ~ReferenceTAAResolvePass() override;

  ReferenceTAAResolvePass(const ReferenceTAAResolvePass &) = delete;
  ReferenceTAAResolvePass &operator=(const ReferenceTAAResolvePass &) = delete;
  ReferenceTAAResolvePass(ReferenceTAAResolvePass &&) = delete;
  ReferenceTAAResolvePass &operator=(ReferenceTAAResolvePass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "ReferenceTAAResolvePass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  GPUDevice &gpu_;
  std::unique_ptr<Shader> shader_{};
  ShaderHandle vertexShader_{};
  ShaderHandle fragmentShader_{};
  SamplerHandle linearClampSampler_{};
  RenderPipelineHandle pipeline_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
};

class NURI_API ReferenceTAAFeature final : public RenderFeature {
public:
  ReferenceTAAFeature(GPUDevice &gpu, RuntimeCompositeConfig config)
      : resolvePass_(gpu, std::move(config)) {}
  ~ReferenceTAAFeature() override = default;

  ReferenceTAAFeature(const ReferenceTAAFeature &) = delete;
  ReferenceTAAFeature &operator=(const ReferenceTAAFeature &) = delete;
  ReferenceTAAFeature(ReferenceTAAFeature &&) = delete;
  ReferenceTAAFeature &operator=(ReferenceTAAFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "ReferenceTAAFeature";
  }
  [[nodiscard]] Result<bool, std::string>
  publishFrameData(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override {
    return passes_;
  }

private:
  ReferenceTAAResolvePass resolvePass_;
  std::array<RenderFeaturePass *, 1> passes_{&resolvePass_};
};

} // namespace nuri
