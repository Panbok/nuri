#pragma once

#include <string>
#include <string_view>
#include <span>

#include <glm/glm.hpp>

#include "nuri/defines.h"
#include "nuri/scene/camera.h"
#include "nuri/scene/camera_system.h"

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

NURI_API bool drawScenePresetWidget(std::span<const char *const> presetNames,
                                    int &selectedIndex,
                                    std::string_view hotkeyHint = "Hotkey: F6");

} // namespace nuri
