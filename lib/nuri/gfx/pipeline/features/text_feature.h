#pragma once

#include <array>
#include <span>

#include "nuri/defines.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/text/text_system.h"

namespace nuri {

class NURI_API Text3DPass final : public RenderFeaturePass {
public:
  explicit Text3DPass(TextSystem &text) : text_(text) {}
  ~Text3DPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "Text3DPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  [[nodiscard]] Result<bool, std::string>
  prepare(FrameBuildContext &ctx) override;
  [[nodiscard]] Result<bool, std::string>
  build(FrameBuildContext &ctx) override;
  [[nodiscard]] TextSystem &textSystem() noexcept { return text_; }

private:
  TextSystem &text_;
};

class NURI_API Text3DFeature final : public RenderFeature {
public:
  explicit Text3DFeature(TextSystem &text) : pass_(text) {}
  ~Text3DFeature() override = default;

  Text3DFeature(const Text3DFeature &) = delete;
  Text3DFeature &operator=(const Text3DFeature &) = delete;
  Text3DFeature(Text3DFeature &&) = delete;
  Text3DFeature &operator=(Text3DFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "Text3DFeature";
  }
  [[nodiscard]] Result<bool, std::string>
  publishFrameData(FrameBuildContext &ctx) override;
  [[nodiscard]] Result<bool, std::string>
  prepare(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  Text3DPass pass_;
  std::array<RenderFeaturePass *, 1> passes_{&pass_};
};

class NURI_API Text2DPass final : public RenderFeaturePass {
public:
  explicit Text2DPass(TextSystem &text) : text_(text) {}
  ~Text2DPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "Text2DPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  [[nodiscard]] Result<bool, std::string>
  prepare(FrameBuildContext &ctx) override;
  [[nodiscard]] Result<bool, std::string>
  build(FrameBuildContext &ctx) override;
  [[nodiscard]] TextSystem &textSystem() noexcept { return text_; }

private:
  TextSystem &text_;
};

class NURI_API Text2DFeature final : public RenderFeature {
public:
  explicit Text2DFeature(TextSystem &text) : pass_(text) {}
  ~Text2DFeature() override = default;

  Text2DFeature(const Text2DFeature &) = delete;
  Text2DFeature &operator=(const Text2DFeature &) = delete;
  Text2DFeature(Text2DFeature &&) = delete;
  Text2DFeature &operator=(Text2DFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "Text2DFeature";
  }
  [[nodiscard]] bool isTerminalFeature() const noexcept override {
    return true;
  }
  [[nodiscard]] Result<bool, std::string>
  prepare(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  Text2DPass pass_;
  std::array<RenderFeaturePass *, 1> passes_{&pass_};
};

} // namespace nuri
