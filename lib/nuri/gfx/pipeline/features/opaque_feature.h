#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/renderers/opaque_renderer.h"

#include <array>
#include <memory_resource>
#include <span>

namespace nuri {

class NURI_API OpaqueMainPass final : public RenderFeaturePass {
public:
  explicit OpaqueMainPass(OpaqueRenderer &renderer) : renderer_(renderer) {}
  ~OpaqueMainPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "OpaqueMainPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  OpaqueRenderer &renderer_;
};

class NURI_API OpaquePickPass final : public RenderFeaturePass {
public:
  explicit OpaquePickPass(OpaqueRenderer &renderer) : renderer_(renderer) {}
  ~OpaquePickPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "OpaquePickPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  OpaqueRenderer &renderer_;
};

class NURI_API OpaqueFeature final : public RenderFeature {
public:
  explicit OpaqueFeature(
      GPUDevice &gpu, OpaqueRendererConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~OpaqueFeature() override;

  OpaqueFeature(const OpaqueFeature &) = delete;
  OpaqueFeature &operator=(const OpaqueFeature &) = delete;
  OpaqueFeature(OpaqueFeature &&) = delete;
  OpaqueFeature &operator=(OpaqueFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "OpaqueFeature";
  }
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  std::unique_ptr<OpaqueRenderer> renderer_;
  OpaqueMainPass mainPass_;
  OpaquePickPass pickPass_;
  std::array<RenderFeaturePass *, 2> passes_{&mainPass_, &pickPass_};
};

} // namespace nuri
