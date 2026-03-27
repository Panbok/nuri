#pragma once

#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/renderers/debug_renderer.h"

#include <array>
#include <memory>
#include <memory_resource>
#include <span>

namespace nuri {

class NURI_API DebugGridPass final : public RenderFeaturePass {
public:
  explicit DebugGridPass(DebugRenderer &renderer) : renderer_(renderer) {}
  ~DebugGridPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "DebugGridPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  DebugRenderer &renderer_;
};

class NURI_API DebugSceneOverlayPass final : public RenderFeaturePass {
public:
  explicit DebugSceneOverlayPass(DebugRenderer &renderer)
      : renderer_(renderer) {}
  ~DebugSceneOverlayPass() override = default;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "DebugSceneOverlayPass";
  }
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> build(FrameBuildContext &ctx) override;

private:
  DebugRenderer &renderer_;
};

class NURI_API DebugFeature final : public RenderFeature {
public:
  explicit DebugFeature(
      GPUDevice &gpu, DebugRendererConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~DebugFeature() override;

  DebugFeature(const DebugFeature &) = delete;
  DebugFeature &operator=(const DebugFeature &) = delete;
  DebugFeature(DebugFeature &&) = delete;
  DebugFeature &operator=(DebugFeature &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "DebugFeature";
  }
  Result<bool, std::string> publishFrameData(FrameBuildContext &ctx) override;
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  [[nodiscard]] std::span<RenderFeaturePass *const> passes() noexcept override;

private:
  std::unique_ptr<DebugRenderer> renderer_;
  DebugGridPass gridPass_;
  DebugSceneOverlayPass overlayPass_;
  std::array<RenderFeaturePass *, 2> passes_{&gridPass_, &overlayPass_};
};

} // namespace nuri
