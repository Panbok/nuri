#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include <string>
#include <string_view>
namespace nuri {

class SceneRuntimeHost;

class NURI_API AnimationSceneFrameProvider final {
public:
  explicit AnimationSceneFrameProvider(SceneRuntimeHost &runtime)
      : runtime_(&runtime) {}
  AnimationSceneFrameProvider(const AnimationSceneFrameProvider &) = delete;
  AnimationSceneFrameProvider &
  operator=(const AnimationSceneFrameProvider &) = delete;
  AnimationSceneFrameProvider(AnimationSceneFrameProvider &&) = delete;
  AnimationSceneFrameProvider &
  operator=(AnimationSceneFrameProvider &&) = delete;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept;
  void bindRuntime(SceneRuntimeHost &runtime) noexcept { runtime_ = &runtime; }

private:
  SceneRuntimeHost *runtime_ = nullptr;
};

} // namespace nuri
