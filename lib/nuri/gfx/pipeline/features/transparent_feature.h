#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/renderers/transparent_renderer.h"

#include <array>
#include <memory_resource>
#include <span>

namespace nuri {

class NURI_API TransparentMainPass final : public RenderFeaturePass {
public:
  explicit TransparentMainPass(TransparentRenderer &renderer)
      : renderer_(renderer) {}
  ~TransparentMainPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TransparentMainPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  TransparentRenderer &renderer_;
};

class NURI_API TransparentPickPass final : public RenderFeaturePass {
public:
  explicit TransparentPickPass(TransparentRenderer &renderer)
      : renderer_(renderer) {}
  ~TransparentPickPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TransparentPickPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  TransparentRenderer &renderer_;
};

class NURI_API TransparentFeature final : public RenderFeature {
public:
  explicit TransparentFeature(
      GPUDevice &gpu, TransparentRendererConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~TransparentFeature() override;

  TransparentFeature(const TransparentFeature &) = delete;
  TransparentFeature &operator=(const TransparentFeature &) = delete;
  TransparentFeature(TransparentFeature &&) = delete;
  TransparentFeature &operator=(TransparentFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TransparentFeature";
  }
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx) override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  std::unique_ptr<TransparentRenderer> renderer_;
  TransparentMainPass mainPass_;
  TransparentPickPass pickPass_;
  std::array<RenderFeaturePass *, 2> passes_{&mainPass_, &pickPass_};
};

} // namespace nuri
