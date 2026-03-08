#pragma once

#include "nuri/core/input_events.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/layers/render_frame_context.h"
#include "nuri/gfx/render_graph/render_graph.h"

#include <string>

namespace nuri {

class GizmoController {
public:
  GizmoController(const GizmoController &) = delete;
  GizmoController &operator=(const GizmoController &) = delete;
  GizmoController(GizmoController &&) = delete;
  GizmoController &operator=(GizmoController &&) = delete;

  [[nodiscard]] virtual bool onInput(const InputEvent &event) = 0;
  virtual void onFrame(RenderFrameContext &frame) = 0;
  virtual void drawUi() = 0;
  virtual void reset() = 0;

protected:
  GizmoController() = default;
  virtual ~GizmoController() = default;
};

class Editor {
public:
  virtual ~Editor() = default;

  Editor(const Editor &) = delete;
  Editor &operator=(const Editor &) = delete;
  Editor(Editor &&) = delete;
  Editor &operator=(Editor &&) = delete;

  virtual void beginFrame() = 0;
  virtual Result<RenderGraphGraphicsPassDesc, std::string> endFrame() = 0;

protected:
  Editor() = default;
};

} // namespace nuri
