#pragma once

#include "nuri/defines.h"
#include "nuri/scene/camera.h"
#include "nuri/scene/camera_system.h"

#include <string>

#include <glm/glm.hpp>

namespace nuri {

struct CameraControllerWidgetState {
  glm::vec3 targetPosition{0.0f, 1.0f, -1.5f};
  float targetYawDegrees{0.0f};
  float targetPitchDegrees{0.0f};
  float durationSeconds{1.5f};
  std::string lastError{};
};

NURI_API void
syncCameraControllerWidgetStateFromCamera(const Camera &camera,
                                          CameraControllerWidgetState &state);

NURI_API void drawCameraControllerWidget(CameraSystem &cameraSystem,
                                         CameraControllerWidgetState &state);

} // namespace nuri
