#pragma once
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/resources/cpu/mesh_data.h"
namespace nuri {

[[nodiscard]] inline const AnimationSceneFrameData *
resolveAnimationSceneFrameData(const RenderFrameContext &frame) noexcept {
  const auto &data = frame.sharedResources.animationSceneGpuData;
  return data ? &*data : nullptr;
}

[[nodiscard]] inline bool
animationSceneAnimatesRenderable(const AnimationSceneFrameData &data,
                                 size_t renderableIndex) noexcept {
  if (std::ranges::find(data.animatedRenderableIndices, renderableIndex) !=
      data.animatedRenderableIndices.end()) {
    return true;
  }
  const auto &geometry = data.geometryOverridesByRenderable[renderableIndex];
  return geometry.enabled && nuri::isValid(geometry.vertexBuffer);
}

[[nodiscard]] inline bool animationOverrideCoversSubmesh(
    const AnimatedRenderableGeometryOverride &geometry,
    const Submesh &submesh) noexcept {
  return static_cast<uint64_t>(geometry.vertexCount) >=
         static_cast<uint64_t>(submesh.vertexOffset) + submesh.vertexCount;
}

}
