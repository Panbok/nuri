#include "nuri/gfx/layers/debug_layer.h"

#include "nuri/core/profiling.h"
#include "nuri/scene/render_scene.h"

namespace nuri {

DebugLayer::DebugLayer(GPUDevice &gpu)
    : debugDraw3D_(std::make_unique<DebugDraw3D>(gpu)) {}

Result<bool, std::string>
DebugLayer::buildRenderPasses(RenderFrameContext &frame, RenderPassList &out) {
  NURI_PROFILER_FUNCTION();

  if (!debugDraw3D_ || !frame.scene) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!frame.settings || !frame.settings->debug.enabled ||
      !frame.settings->debug.modelBounds) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!nuri::isValid(frame.sharedDepthTexture)) {
    return Result<bool, std::string>::makeResult(true);
  }

  const std::span<const OpaqueRenderable> renderables =
      frame.scene->opaqueRenderables();
  if (renderables.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  debugDraw3D_->clear();
  debugDraw3D_->setMatrix(frame.camera.proj * frame.camera.view);
  for (const OpaqueRenderable &renderable : renderables) {
    if (!renderable.model) {
      continue;
    }
    debugDraw3D_->box(renderable.modelMatrix, renderable.model->bounds(),
                      glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
  }

  auto linePassResult = debugDraw3D_->buildRenderPass(frame.sharedDepthTexture);
  if (linePassResult.hasError()) {
    return Result<bool, std::string>::makeError(linePassResult.error());
  }

  out.push_back(linePassResult.value());
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
