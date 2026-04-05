#include "nuri/pch.h"

#include "nuri/gfx/sim/animation_scene_frame_provider.h"

#include "nuri/scene_runtime/scene_runtime_host.h"

namespace nuri {

Result<bool, std::string>
AnimationSceneFrameProvider::prepare(FrameBuildContext &ctx) {
  auto prepareResult =
      runtime_.prepareAnimationSceneFrame(ctx.frame.frameIndex);
  if (prepareResult.hasError()) {
    NURI_LOG_WARNING("AnimationSceneFrameProvider::prepare: failed to prepare "
                     "frame %llu: %s",
                     static_cast<unsigned long long>(ctx.frame.frameIndex),
                     prepareResult.error().c_str());
    return prepareResult;
  }
  const AnimationSceneFrameData *frameData = runtime_.animationSceneFrameData();
  if (frameData == nullptr) {
    NURI_LOG_DEBUG("AnimationSceneFrameProvider::prepare: early exit because "
                   "frameData == nullptr");
    return Result<bool, std::string>::makeResult(true);
  }
  if (!nuri::isValid(frameData->instanceMatricesBuffer)) {
    NURI_LOG_DEBUG("AnimationSceneFrameProvider::prepare: early exit because "
                   "instanceMatricesBuffer is invalid");
    return Result<bool, std::string>::makeResult(true);
  }
  if (frameData->instanceMatricesAddress == 0u) {
    NURI_LOG_DEBUG("AnimationSceneFrameProvider::prepare: early exit because "
                   "instanceMatricesAddress == 0u");
    return Result<bool, std::string>::makeResult(true);
  }
  if (!frameData->preDispatches.empty()) {
    ctx.graph.mixDynamicPayloadVersion(frameData->version);
  }
  ctx.frame.sharedResources.animationSceneGpuData = *frameData;
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
