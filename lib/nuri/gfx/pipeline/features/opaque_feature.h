#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/renderers/opaque_renderer.h"

#include <array>
#include <memory>
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

class NURI_API OpaquePrepassPass final : public RenderFeaturePass {
public:
  explicit OpaquePrepassPass(OpaqueRenderer &renderer) : renderer_(renderer) {}
  ~OpaquePrepassPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "OpaquePrepassPass";
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

class NURI_API OpaqueMainLightingPass final : public RenderFeaturePass {
public:
  explicit OpaqueMainLightingPass(OpaqueRenderer &renderer)
      : renderer_(renderer) {}
  ~OpaqueMainLightingPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "OpaqueMainLightingPass";
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

using SharedOpaqueRenderer = std::shared_ptr<OpaqueRenderer>;

NURI_API SharedOpaqueRenderer makeSharedOpaqueRenderer(
    GPUDevice &gpu, OpaqueRendererConfig config,
    std::pmr::memory_resource *memory = std::pmr::get_default_resource());

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
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx) override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept override;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  // Keep renderer_/pass members in this order. passes_ stores pointers to the
  // pass members and is initialized from them in the constructor.
  std::unique_ptr<OpaqueRenderer> renderer_;
  OpaqueMainPass mainPass_;
  OpaquePickPass pickPass_;
  std::array<RenderFeaturePass *, 2> passes_{};
};

class NURI_API OpaquePrepassFeature final : public RenderFeature {
public:
  explicit OpaquePrepassFeature(SharedOpaqueRenderer renderer);
  ~OpaquePrepassFeature() override;

  OpaquePrepassFeature(const OpaquePrepassFeature &) = delete;
  OpaquePrepassFeature &operator=(const OpaquePrepassFeature &) = delete;
  OpaquePrepassFeature(OpaquePrepassFeature &&) = delete;
  OpaquePrepassFeature &operator=(OpaquePrepassFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "OpaquePrepassFeature";
  }
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx) override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  Result<bool, std::string>
  prepareSceneStep(RenderScenePreparationContext &ctx) override;
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept override;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  SharedOpaqueRenderer renderer_;
  OpaquePickPass pickPass_;
  OpaquePrepassPass prepass_;
  std::array<RenderFeaturePass *, 2> passes_{};
};

class NURI_API OpaqueMainFeature final : public RenderFeature {
public:
  // OpaqueMainFeature must share a SharedOpaqueRenderer with
  // OpaquePrepassFeature and be scheduled after it: OpaquePrepassFeature calls
  // publishFrameData() and prepare(), which populate renderer_ frame data and
  // prepared passes consumed by the main lighting pass.
  explicit OpaqueMainFeature(SharedOpaqueRenderer renderer);
  ~OpaqueMainFeature() override = default;

  OpaqueMainFeature(const OpaqueMainFeature &) = delete;
  OpaqueMainFeature &operator=(const OpaqueMainFeature &) = delete;
  OpaqueMainFeature(OpaqueMainFeature &&) = delete;
  OpaqueMainFeature &operator=(OpaqueMainFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "OpaqueMainFeature";
  }
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  SharedOpaqueRenderer renderer_;
  OpaqueMainLightingPass mainLightingPass_;
  std::array<RenderFeaturePass *, 1> passes_{};
};

} // namespace nuri
