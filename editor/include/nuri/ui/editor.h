#pragma once

#include "nuri/core/input_events.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/render_graph/render_graph.h"

#include <string>
#include <string_view>

namespace nuri {

class RenderScene;

struct GizmoUiDrawConfig {
  // Non-owning views; referenced titles must outlive this config and the draw
  // call that consumes it.
  bool *showControlsWindow = nullptr;
  std::string_view controlsWindowTitle = "Gizmo Controls";
  bool *showLightsWindow = nullptr;
  std::string_view lightsWindowTitle = "Lights";
};

struct EditorSceneSelectionOption {
  // Non-owning views; referenced strings must outlive the option and any scene
  // selection UI update that consumes it.
  std::string_view id{};
  std::string_view label{};
};

struct EditorSceneLoadUiState {
  std::string_view pendingSceneId{};
  std::string_view phase{};
  std::string_view error{};
  float progress = 0.0f;
  bool cancellable = false;
  bool failed = false;
};

class GizmoController {
public:
  GizmoController(const GizmoController &) = delete;
  GizmoController &operator=(const GizmoController &) = delete;
  GizmoController(GizmoController &&) = delete;
  GizmoController &operator=(GizmoController &&) = delete;

  [[nodiscard]] virtual bool onInput(const InputEvent &event) = 0;
  virtual void onFrame(RenderFrameContext &frame) = 0;
  virtual void drawUi(const GizmoUiDrawConfig &config) = 0;
  virtual void setShadowInspectRequestsEnabled(bool enabled) = 0;
  virtual void invalidatePendingPicks() = 0;
  virtual void reset() = 0;
  virtual void bindScene(RenderScene &scene) = 0;

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
