#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"

#include <array>
#include <span>

namespace nuri {

class NURI_API TemporalAAMotionVectorClearPass final
    : public RenderFeaturePass {
public:
  TemporalAAMotionVectorClearPass() = default;
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
  [[nodiscard]] Result<bool, std::string>
  publishFrameData(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  TemporalAAMotionVectorClearPass motionVectorClearPass_{};
  std::array<RenderFeaturePass *, 1> passes_{&motionVectorClearPass_};
};

} // namespace nuri
