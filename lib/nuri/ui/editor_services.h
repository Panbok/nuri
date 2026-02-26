#pragma once

#include "nuri/defines.h"

namespace nuri {

class RenderScene;
class CameraSystem;
class GPUDevice;
class TextSystem;

struct EditorServices {
  RenderScene *scene = nullptr;
  CameraSystem *cameraSystem = nullptr;
  GPUDevice *gpu = nullptr;
  TextSystem *textSystem = nullptr;

  [[nodiscard]] bool hasAllDependencies() const {
    // Gizmo controller dependencies.
    return scene != nullptr && cameraSystem != nullptr && gpu != nullptr;
  }
};

} // namespace nuri
