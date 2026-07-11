#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/pipeline/frame_data_provider.h"

#include <string>
#include <string_view>

namespace nuri {

class SceneRuntimeHost;

class NURI_API AnimationSceneFrameProvider final : public FrameDataProvider {
public:
  explicit AnimationSceneFrameProvider(SceneRuntimeHost &runtime)
      : runtime_(runtime) {}
  AnimationSceneFrameProvider(const AnimationSceneFrameProvider &) = delete;
  AnimationSceneFrameProvider &
  operator=(const AnimationSceneFrameProvider &) = delete;
  AnimationSceneFrameProvider(AnimationSceneFrameProvider &&) = delete;
  AnimationSceneFrameProvider &
  operator=(AnimationSceneFrameProvider &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "AnimationSceneFrameProvider";
  }
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept override;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept override;

private:
  SceneRuntimeHost &runtime_;
};

} // namespace nuri
