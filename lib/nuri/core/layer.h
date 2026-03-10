#pragma once

#include "nuri/core/input_events.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/layers/render_frame_context.h"
#include "nuri/gfx/render_graph/render_graph.h"

#include <cstdint>
#include <string>

namespace nuri {

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
  // Called once per frame before any buildRenderGraph() calls.
  // Any layer (commonly overlays) may override this to publish frame-scoped
  // state into RenderFrameContext before other layers consume it via
  // buildRenderGraph().
  virtual void prepareFrameContext(RenderFrameContext &frame) {}
  virtual void publishFrameData(RenderFrameContext &frame) {
    prepareFrameContext(frame);
  }
  // `out` exposes non-owning spans into contributor-managed frame data.
  // Callers must consume or copy them before the contributor's next
  // clear()/beginFrame()-style reset invalidates the backing storage.
  virtual Result<bool, std::string>
  buildTransparentStageContribution([[maybe_unused]] RenderFrameContext &frame,
                                    TransparentStageContribution &out) {
    out = {};
    return Result<bool, std::string>::makeResult(true);
  }
  virtual Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &frame, RenderGraphBuilder &graph) = 0;

protected:
  Layer() = default;
};

} // namespace nuri
