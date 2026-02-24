#pragma once

#include "nuri/core/input_events.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/layers/render_frame_context.h"

#include <cstdint>
#include <memory_resource>
#include <string>
#include <vector>

namespace nuri {

using RenderPassList = std::pmr::vector<RenderPass>;

class NURI_API Layer {
public:
  virtual ~Layer() = default;

  Layer(const Layer &) = delete;
  Layer &operator=(const Layer &) = delete;
  Layer(Layer &&) = delete;
  Layer &operator=(Layer &&) = delete;

  virtual void onAttach() {}
  virtual void onDetach() {}
  virtual void onUpdate(double deltaTime) {}
  virtual void onResize(int32_t width, int32_t height) {}
  virtual bool onInput(const InputEvent &event) { return false; }
  // Called once per frame before any buildRenderPasses() calls.
  // Overlays can use this to publish frame-scoped requests consumed by layers.
  virtual void prepareFrameContext(RenderFrameContext &frame) {}
  virtual Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                                      RenderPassList &out);

protected:
  Layer() = default;
};

inline Result<bool, std::string> Layer::buildRenderPasses(RenderFrameContext &,
                                                          RenderPassList &) {
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
