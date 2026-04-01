#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/ui/editor_overlay_controller.h"

#include <array>
#include <span>

namespace nuri {

class EditorOverlayPass final : public RenderFeaturePass {
public:
  explicit EditorOverlayPass(EditorOverlayController *controller = nullptr)
      : controller_(controller) {}
  ~EditorOverlayPass() override = default;

  EditorOverlayPass(const EditorOverlayPass &) = delete;
  EditorOverlayPass &operator=(const EditorOverlayPass &) = delete;
  EditorOverlayPass(EditorOverlayPass &&) = delete;
  EditorOverlayPass &operator=(EditorOverlayPass &&) = delete;

  void setController(EditorOverlayController *controller) noexcept {
    controller_ = controller;
  }

  [[nodiscard]] std::string_view name() const noexcept override {
    return "EditorOverlayPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  EditorOverlayController *controller_ = nullptr;
};

class EditorOverlayFeature final : public RenderFeature {
public:
  explicit EditorOverlayFeature(EditorOverlayController *controller = nullptr)
      : pass_(controller) {}
  ~EditorOverlayFeature() override = default;

  EditorOverlayFeature(const EditorOverlayFeature &) = delete;
  EditorOverlayFeature &operator=(const EditorOverlayFeature &) = delete;
  EditorOverlayFeature(EditorOverlayFeature &&) = delete;
  EditorOverlayFeature &operator=(EditorOverlayFeature &&) = delete;

  void setController(EditorOverlayController *controller) noexcept {
    pass_.setController(controller);
  }

  [[nodiscard]] std::string_view name() const noexcept override {
    return "EditorOverlayFeature";
  }
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  EditorOverlayPass pass_;
  // `passes_` stores `&pass_`, so `pass_` must be declared first.
  std::array<RenderFeaturePass *, 1> passes_{&pass_};
};

} // namespace nuri
