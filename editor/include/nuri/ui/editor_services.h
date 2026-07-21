#pragma once

#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_handles.h"

#include <cstdint>

namespace nuri {

class RenderScene;
class Application;
class CameraSystem;
class GPUDevice;
class TextSystem;
class ResourceManager;
class RenderGraphTelemetryService;
class RenderPipeline;
class EditorAnimationPlayerService;
namespace bakery {
class BakerySystem;
}

enum class SceneSelectionKind : uint8_t {
  None = 0,
  Node = 1,
  NodeRenderable = 2,
  Light = 3,
  DDGIVolume = 4,
};

// `kind` selects which payload members are meaningful:
// - `None`: no selection payload is valid.
// - `Node`: only `node` is valid.
// - `NodeRenderable`: `node`, `renderableId`, and `renderableIndex` are valid.
// - `Light`: `node` and `lightId` are valid.
// - `DDGIVolume`: `node` and `ddgiVolumeId` are valid.
struct SceneEditorSelectionState {
  SceneSelectionKind kind = SceneSelectionKind::None;
  NodeId node = kInvalidNodeId;
  RenderableId renderableId = kInvalidRenderableId;
  uint32_t renderableIndex = 0u;
  LightId lightId = kInvalidLightId;
  DDGIVolumeId ddgiVolumeId = kInvalidDDGIVolumeId;

  constexpr void clear() noexcept {
    kind = SceneSelectionKind::None;
    node = kInvalidNodeId;
    renderableId = kInvalidRenderableId;
    renderableIndex = 0u;
    lightId = kInvalidLightId;
    ddgiVolumeId = kInvalidDDGIVolumeId;
  }
};

struct EditorServices {
  Application *application = nullptr;
  RenderScene *scene = nullptr;
  CameraSystem *cameraSystem = nullptr;
  GPUDevice *gpu = nullptr;
  ResourceManager *resources = nullptr;
  RenderPipeline *renderPipeline = nullptr;
  TextSystem *textSystem = nullptr;
  bakery::BakerySystem *bakery = nullptr;
  RenderGraphTelemetryService *renderGraphTelemetry = nullptr;
  SceneEditorSelectionState *selectionState = nullptr;
  EditorAnimationPlayerService *animationPlayer = nullptr;

  [[nodiscard]] bool hasAllDependencies() const {
    // Gizmo controller dependencies.
    return scene != nullptr && cameraSystem != nullptr && gpu != nullptr;
  }
};

} // namespace nuri
