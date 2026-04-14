#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/renderers/shadow_renderer.h"

#include <array>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>

namespace nuri {

class NURI_API ShadowDepthPass final : public RenderFeaturePass {
public:
  explicit ShadowDepthPass(ShadowRenderer &renderer) : renderer_(renderer) {}
  ~ShadowDepthPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "ShadowDepthPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  ShadowRenderer &renderer_;
};

class NURI_API ShadowFeature final : public RenderFeature {
public:
  explicit ShadowFeature(GPUDevice &gpu, std::pmr::memory_resource *memory =
                                             std::pmr::get_default_resource());
  ~ShadowFeature() override = default;

  ShadowFeature(const ShadowFeature &) = delete;
  ShadowFeature &operator=(const ShadowFeature &) = delete;
  ShadowFeature(ShadowFeature &&) = delete;
  ShadowFeature &operator=(ShadowFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "ShadowFeature";
  }
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx) override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  std::unique_ptr<ShadowRenderer> renderer_;
  ShadowDepthPass depthPass_;
  std::array<RenderFeaturePass *, 1> passes_{&depthPass_};
};

} // namespace nuri
