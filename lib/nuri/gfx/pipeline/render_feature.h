#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/pipeline/frame_build_context.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"

#include <span>
#include <string>
#include <string_view>

namespace nuri {

class RenderScene;
class ResourceManager;
class Camera;
struct RenderSettings;

struct RenderScenePreparationContext {
  RenderScene &scene;
  ResourceManager &resources;
  uint32_t maxOperations = 128u;
  const RenderSettings *settings = nullptr;
  const Camera *camera = nullptr;
  float aspectRatio = 1.0f;
  uint32_t renderWidth = 1u;
  uint32_t renderHeight = 1u;
};

class NURI_API RenderFeature {
public:
  virtual ~RenderFeature() = default;
  RenderFeature(const RenderFeature &) = delete;
  RenderFeature &operator=(const RenderFeature &) = delete;
  RenderFeature(RenderFeature &&) = delete;
  RenderFeature &operator=(RenderFeature &&) = delete;

protected:
  RenderFeature() = default;

public:
  virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual Result<bool, std::string>
  publishFrameData(FrameBuildContext &ctx) {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  [[nodiscard]] virtual Result<bool, std::string>
  prepare(FrameBuildContext &ctx) {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  // Advances renderer-owned, scene-derived caches without making the scene
  // active. Returning true means this feature is ready to render the exact
  // topology/resource versions supplied by ctx.
  [[nodiscard]] virtual Result<bool, std::string>
  prepareSceneStep(RenderScenePreparationContext &ctx) {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  [[nodiscard]] virtual bool isTerminalFeature() const noexcept {
    return false;
  }
  virtual void onFrameSubmitted(const RenderFrameContext &frame) noexcept {
    (void)frame;
  }
  virtual void onFrameAbandoned(const RenderFrameContext &frame) noexcept {
    (void)frame;
  }
  [[nodiscard]] virtual std::span<RenderFeaturePass *const>
  passes() noexcept = 0;
};

} // namespace nuri
