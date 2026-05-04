#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"

#include <array>
#include <span>

namespace nuri {

class GPUDevice;

class NURI_API MsaaResolvePass final : public RenderFeaturePass {
public:
  explicit MsaaResolvePass(GPUDevice &gpu) : gpu_(gpu) {}
  ~MsaaResolvePass() override = default;

  MsaaResolvePass(const MsaaResolvePass &) = delete;
  MsaaResolvePass &operator=(const MsaaResolvePass &) = delete;
  MsaaResolvePass(MsaaResolvePass &&) = delete;
  MsaaResolvePass &operator=(MsaaResolvePass &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "MsaaResolvePass";
  }
  [[nodiscard]] bool
  isEnabled(const FrameBuildContext &ctx) const noexcept override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  GPUDevice &gpu_;
};

class NURI_API MsaaResolveFeature final : public RenderFeature {
public:
  explicit MsaaResolveFeature(GPUDevice &gpu) : pass_(gpu) {}
  ~MsaaResolveFeature() override = default;

  MsaaResolveFeature(const MsaaResolveFeature &) = delete;
  MsaaResolveFeature &operator=(const MsaaResolveFeature &) = delete;
  MsaaResolveFeature(MsaaResolveFeature &&) = delete;
  MsaaResolveFeature &operator=(MsaaResolveFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "MsaaResolveFeature";
  }
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  MsaaResolvePass pass_;
  // passes_ stores a RenderFeaturePass pointer to the sibling pass_
  // MsaaResolvePass. Keep pass_ declared first; copy/move must remain deleted
  // to preserve the declaration-order and no-dangling-pointer invariant.
  std::array<RenderFeaturePass *, 1> passes_{&pass_};
};

} // namespace nuri
