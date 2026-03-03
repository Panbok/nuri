#pragma once

#include "nuri/defines.h"

namespace nuri {

class RenderScene;
class CameraSystem;
class GPUDevice;
class TextSystem;
namespace bakery {
class BakerySystem;
}

struct EditorServices {
  RenderScene *scene = nullptr;
  CameraSystem *cameraSystem = nullptr;
  GPUDevice *gpu = nullptr;
  TextSystem *textSystem = nullptr;
  bakery::BakerySystem *bakery = nullptr;

  [[nodiscard]] bool hasAllDependencies() const {
    // Gizmo controller dependencies.
    return scene != nullptr && cameraSystem != nullptr && gpu != nullptr;
  }
};

} // namespace nuri
