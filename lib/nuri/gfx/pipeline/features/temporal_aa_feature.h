#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/shader.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace nuri {

enum class TemporalInputPlacement : uint8_t {
  EarlyGtao = 0u,
  ColorReconstruction = 1u,
};

class NURI_API TemporalAAMotionVectorClearPass final
    : public RenderFeaturePass {
public:
  explicit TemporalAAMotionVectorClearPass(TemporalInputPlacement placement)
      : placement_(placement) {}
  ~TemporalAAMotionVectorClearPass() override = default;

  TemporalAAMotionVectorClearPass(const TemporalAAMotionVectorClearPass &) =
      delete;
  TemporalAAMotionVectorClearPass &
  operator=(const TemporalAAMotionVectorClearPass &) = delete;
  TemporalAAMotionVectorClearPass(TemporalAAMotionVectorClearPass &&) = delete;
  TemporalAAMotionVectorClearPass &
  operator=(TemporalAAMotionVectorClearPass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TemporalAAMotionVectorClearPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  TemporalInputPlacement placement_ =
      TemporalInputPlacement::ColorReconstruction;
};

class NURI_API TemporalAAReactiveMaskClearPass final
    : public RenderFeaturePass {
public:
  explicit TemporalAAReactiveMaskClearPass(TemporalInputPlacement placement)
      : placement_(placement) {}
  ~TemporalAAReactiveMaskClearPass() override = default;

  TemporalAAReactiveMaskClearPass(const TemporalAAReactiveMaskClearPass &) =
      delete;
  TemporalAAReactiveMaskClearPass &
  operator=(const TemporalAAReactiveMaskClearPass &) = delete;
  TemporalAAReactiveMaskClearPass(TemporalAAReactiveMaskClearPass &&) = delete;
  TemporalAAReactiveMaskClearPass &
  operator=(TemporalAAReactiveMaskClearPass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TemporalAAReactiveMaskClearPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  TemporalInputPlacement placement_ =
      TemporalInputPlacement::ColorReconstruction;
};

class NURI_API TemporalAABackgroundMotionPass final : public RenderFeaturePass {
public:
  explicit TemporalAABackgroundMotionPass(GPUDevice &gpu,
                                          RuntimeCompositeConfig config,
                                          TemporalInputPlacement placement);
  ~TemporalAABackgroundMotionPass() override;

  TemporalAABackgroundMotionPass(const TemporalAABackgroundMotionPass &) =
      delete;
  TemporalAABackgroundMotionPass &
  operator=(const TemporalAABackgroundMotionPass &) = delete;
  TemporalAABackgroundMotionPass(TemporalAABackgroundMotionPass &&) = delete;
  TemporalAABackgroundMotionPass &
  operator=(TemporalAABackgroundMotionPass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TemporalAABackgroundMotionPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  GPUDevice &gpu_;
  std::unique_ptr<Shader> shader_{};
  ShaderHandle vertexShader_{};
  ShaderHandle fragmentShader_{};
  RenderPipelineHandle pipeline_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
  TemporalInputPlacement placement_ =
      TemporalInputPlacement::ColorReconstruction;
};

class NURI_API TemporalAAMotionClassPass final : public RenderFeaturePass {
public:
  explicit TemporalAAMotionClassPass(GPUDevice &gpu,
                                     RuntimeCompositeConfig config,
                                     TemporalInputPlacement placement);
  ~TemporalAAMotionClassPass() override;

  TemporalAAMotionClassPass(const TemporalAAMotionClassPass &) = delete;
  TemporalAAMotionClassPass &
  operator=(const TemporalAAMotionClassPass &) = delete;
  TemporalAAMotionClassPass(TemporalAAMotionClassPass &&) = delete;
  TemporalAAMotionClassPass &operator=(TemporalAAMotionClassPass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TemporalAAMotionClassPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  GPUDevice &gpu_;
  std::unique_ptr<Shader> shader_{};
  ShaderHandle vertexShader_{};
  ShaderHandle fragmentShader_{};
  RenderPipelineHandle pipeline_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
  TemporalInputPlacement placement_ =
      TemporalInputPlacement::ColorReconstruction;
};

class NURI_API TemporalInputFeature final : public RenderFeature {
public:
  explicit TemporalInputFeature(GPUDevice &gpu, RuntimeCompositeConfig config);
  ~TemporalInputFeature() override = default;

  TemporalInputFeature(const TemporalInputFeature &) = delete;
  TemporalInputFeature &operator=(const TemporalInputFeature &) = delete;
  TemporalInputFeature(TemporalInputFeature &&) = delete;
  TemporalInputFeature &operator=(TemporalInputFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TemporalInputFeature";
  }
  [[nodiscard]] Result<bool, std::string>
  publishFrameData(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  TemporalAAMotionVectorClearPass motionVectorClearPass_{
      TemporalInputPlacement::EarlyGtao};
  TemporalAAReactiveMaskClearPass reactiveMaskClearPass_{
      TemporalInputPlacement::EarlyGtao};
  TemporalAABackgroundMotionPass backgroundMotionPass_;
  TemporalAAMotionClassPass motionClassPass_;
  std::array<RenderFeaturePass *, 4> passes_{
      &motionVectorClearPass_, &reactiveMaskClearPass_, &backgroundMotionPass_,
      &motionClassPass_};
};

class NURI_API TemporalAAResolvePass final : public RenderFeaturePass {
public:
  explicit TemporalAAResolvePass(GPUDevice &gpu, RuntimeCompositeConfig config);
  ~TemporalAAResolvePass() override;

  TemporalAAResolvePass(const TemporalAAResolvePass &) = delete;
  TemporalAAResolvePass &operator=(const TemporalAAResolvePass &) = delete;
  TemporalAAResolvePass(TemporalAAResolvePass &&) = delete;
  TemporalAAResolvePass &operator=(TemporalAAResolvePass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TemporalAAResolvePass";
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
  Format pipelineColorFormat_ = Format::Count;
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
  bool initialized_ = false;
};

class NURI_API TemporalAAFeature final : public RenderFeature {
public:
  explicit TemporalAAFeature(GPUDevice &gpu, RuntimeCompositeConfig config);
  ~TemporalAAFeature() override = default;

  TemporalAAFeature(const TemporalAAFeature &) = delete;
  TemporalAAFeature &operator=(const TemporalAAFeature &) = delete;
  TemporalAAFeature(TemporalAAFeature &&) = delete;
  TemporalAAFeature &operator=(TemporalAAFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TemporalAAFeature";
  }
  [[nodiscard]] Result<bool, std::string>
  publishFrameData(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  TemporalAAMotionVectorClearPass motionVectorClearPass_{
      TemporalInputPlacement::ColorReconstruction};
  TemporalAAReactiveMaskClearPass reactiveMaskClearPass_{
      TemporalInputPlacement::ColorReconstruction};
  TemporalAABackgroundMotionPass backgroundMotionPass_;
  TemporalAAMotionClassPass motionClassPass_;
  TemporalAAResolvePass resolvePass_;
  std::array<RenderFeaturePass *, 5> passes_{
      &motionVectorClearPass_, &reactiveMaskClearPass_, &backgroundMotionPass_,
      &motionClassPass_, &resolvePass_};
};

} // namespace nuri
