#pragma once

#include "nuri/defines.h"

namespace nuri {

class RenderScene;
class CameraSystem;
class GPUDevice;

struct EditorServices {
  RenderScene *scene = nullptr;
  CameraSystem *cameraSystem = nullptr;
  GPUDevice *gpu = nullptr;

  [[nodiscard]] bool hasGizmoDependencies() const {
    return scene != nullptr && cameraSystem != nullptr && gpu != nullptr;
  }
};

} // namespace nuri
