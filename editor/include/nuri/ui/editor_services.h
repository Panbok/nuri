#pragma once

#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_handles.h"

#include <cstdint>

namespace nuri {

class RenderScene;
class CameraSystem;
class GPUDevice;
class TextSystem;
class ResourceManager;
class RenderGraphTelemetryService;
class RenderPipeline;
namespace bakery {
class BakerySystem;
}

enum class SceneSelectionKind : uint8_t {
  None = 0,
  NodeRenderable = 1,
  Light = 2,
};

struct SceneEditorSelectionState {
  SceneSelectionKind kind = SceneSelectionKind::None;
  NodeId node = kInvalidNodeId;
  RenderableId renderableId = kInvalidRenderableId;
  uint32_t renderableIndex = 0u;
  LightId lightId = kInvalidLightId;

  void clear() noexcept {
    kind = SceneSelectionKind::None;
    node = kInvalidNodeId;
    renderableId = kInvalidRenderableId;
    renderableIndex = 0u;
    lightId = kInvalidLightId;
  }
};

struct EditorServices {
  RenderScene *scene = nullptr;
  CameraSystem *cameraSystem = nullptr;
  GPUDevice *gpu = nullptr;
  ResourceManager *resources = nullptr;
  RenderPipeline *renderPipeline = nullptr;
  TextSystem *textSystem = nullptr;
  bakery::BakerySystem *bakery = nullptr;
  RenderGraphTelemetryService *renderGraphTelemetry = nullptr;
  SceneEditorSelectionState *selectionState = nullptr;

  [[nodiscard]] bool hasAllDependencies() const {
    // Gizmo controller dependencies.
    return scene != nullptr && cameraSystem != nullptr && gpu != nullptr;
  }
};

} // namespace nuri
