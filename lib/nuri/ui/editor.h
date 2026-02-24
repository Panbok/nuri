#pragma once

#include "nuri/core/input_events.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/layers/render_frame_context.h"

#include <string>

namespace nuri {

class NURI_API GizmoController {
public:
  virtual ~GizmoController() = default;

  virtual bool onInput(const InputEvent &event) = 0;
  virtual void onFrame(RenderFrameContext &frame) = 0;
  virtual void drawUi() = 0;
  virtual void reset() = 0;
};

class NURI_API Editor {
public:
  virtual ~Editor() = default;

  Editor(const Editor &) = delete;
  Editor &operator=(const Editor &) = delete;
  Editor(Editor &&) = delete;
  Editor &operator=(Editor &&) = delete;

  virtual void beginFrame() = 0;
  virtual Result<RenderPass, std::string> endFrame() = 0;

protected:
  Editor() = default;
};

} // namespace nuri
