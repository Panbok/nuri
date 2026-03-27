#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/renderers/transmission_renderer.h"

#include <array>
#include <memory_resource>
#include <span>

namespace nuri {

class NURI_API TransmissionDownsamplePass final : public RenderFeaturePass {
public:
  explicit TransmissionDownsamplePass(TransmissionRenderer &renderer)
      : renderer_(renderer) {}
  ~TransmissionDownsamplePass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TransmissionDownsamplePass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  TransmissionRenderer &renderer_;
};

class NURI_API TransmissionCopyPass final : public RenderFeaturePass {
public:
  explicit TransmissionCopyPass(TransmissionRenderer &renderer)
      : renderer_(renderer) {}
  ~TransmissionCopyPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TransmissionCopyPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  TransmissionRenderer &renderer_;
};

class NURI_API TransmissionMainPass final : public RenderFeaturePass {
public:
  explicit TransmissionMainPass(TransmissionRenderer &renderer)
      : renderer_(renderer) {}
  ~TransmissionMainPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TransmissionMainPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  TransmissionRenderer &renderer_;
};

class NURI_API TransmissionFeature final : public RenderFeature {
public:
  explicit TransmissionFeature(
      GPUDevice &gpu, TransmissionRendererConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~TransmissionFeature() override;

  TransmissionFeature(const TransmissionFeature &) = delete;
  TransmissionFeature &operator=(const TransmissionFeature &) = delete;
  TransmissionFeature(TransmissionFeature &&) = delete;
  TransmissionFeature &operator=(TransmissionFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TransmissionFeature";
  }
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx) override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  // Keep renderer_ first: pass members and passes_ depend on it during
  // construction.
  std::unique_ptr<TransmissionRenderer> renderer_;
  TransmissionDownsamplePass downsamplePass_;
  TransmissionCopyPass copyPass_;
  TransmissionMainPass mainPass_;
  std::array<RenderFeaturePass *, 3> passes_{&downsamplePass_, &copyPass_,
                                             &mainPass_};
};

} // namespace nuri
