#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"

#include <array>
#include <span>

namespace nuri {

class NURI_API TemporalAANoopPass final : public RenderFeaturePass {
public:
  TemporalAANoopPass() = default;
  ~TemporalAANoopPass() override = default;

  TemporalAANoopPass(const TemporalAANoopPass &) = delete;
  TemporalAANoopPass &operator=(const TemporalAANoopPass &) = delete;
  TemporalAANoopPass(TemporalAANoopPass &&) = delete;
  TemporalAANoopPass &operator=(TemporalAANoopPass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TemporalAANoopPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;
};

class NURI_API TemporalAAFeature final : public RenderFeature {
public:
  TemporalAAFeature() = default;
  ~TemporalAAFeature() override = default;

  TemporalAAFeature(const TemporalAAFeature &) = delete;
  TemporalAAFeature &operator=(const TemporalAAFeature &) = delete;
  TemporalAAFeature(TemporalAAFeature &&) = delete;
  TemporalAAFeature &operator=(TemporalAAFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "TemporalAAFeature";
  }
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  TemporalAANoopPass noopPass_{};
  std::array<RenderFeaturePass *, 1> passes_{&noopPass_};
};

} // namespace nuri
