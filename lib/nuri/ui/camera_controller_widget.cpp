#include "nuri/ui/camera_controller_widget.h"

#include "nuri/pch.h"

namespace nuri {

namespace {

glm::quat cameraOrientationFromYawPitch(float yawRadians, float pitchRadians) {
  const glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
  const glm::quat yawRotation = glm::angleAxis(yawRadians, worldUp);
  const glm::vec3 rightAxis =
      glm::normalize(yawRotation * glm::vec3(1.0f, 0.0f, 0.0f));
  const glm::quat pitchRotation = glm::angleAxis(pitchRadians, rightAxis);
  return glm::normalize(pitchRotation * yawRotation);
}

} // namespace

void syncCameraControllerWidgetStateFromCamera(
    const Camera &camera, CameraControllerWidgetState &state) {
  state.targetPosition = camera.position();
  const glm::vec3 forward = camera.forward();
  state.targetPitchDegrees =
      glm::degrees(std::asin(glm::clamp(forward.y, -1.0f, 1.0f)));
  state.targetYawDegrees = glm::degrees(std::atan2(forward.x, -forward.z));
}

void drawCameraControllerWidget(CameraSystem &cameraSystem,
                                CameraControllerWidgetState &state) {
  if (!ImGui::Begin("Camera Controller")) {
    ImGui::End();
    return;
  }

  CameraController *controller = cameraSystem.activeController();
  Camera *camera = cameraSystem.activeCamera();
  if (!controller || !camera) {
    ImGui::TextUnformatted("No active camera/controller");
    ImGui::End();
    return;
  }

  int presetIndex = static_cast<int>(controller->preset());
  const char *presetNames[] = {"FPS + Direct", "FPS + MoveTo"};
  if (ImGui::Combo("Preset", &presetIndex, presetNames,
                   IM_ARRAYSIZE(presetNames))) {
    controller->setPreset(static_cast<CameraPreset>(presetIndex));
  }

  if (controller->preset() == CameraPreset::FpsMoveTo) {
    if (ImGui::Button("Use Current Pose")) {
      syncCameraControllerWidgetStateFromCamera(*camera, state);
    }

    ImGui::InputFloat3("Target Position", &state.targetPosition.x);
    ImGui::InputFloat("Target Yaw (deg)", &state.targetYawDegrees);
    ImGui::InputFloat("Target Pitch (deg)", &state.targetPitchDegrees);
    ImGui::InputFloat("Duration (sec)", &state.durationSeconds);
    state.durationSeconds = std::max(state.durationSeconds, 0.01f);

    if (ImGui::Button("Start MoveTo")) {
      const float yawRadians = glm::radians(state.targetYawDegrees);
      const float pitchRadians = glm::radians(state.targetPitchDegrees);
      const MoveToRequest request{
          .targetPosition = state.targetPosition,
          .targetOrientation =
              cameraOrientationFromYawPitch(yawRadians, pitchRadians),
          .durationSeconds = state.durationSeconds,
          .easing = MoveToEasing::Smoothstep,
      };
      auto moveToResult = controller->startMoveTo(request);
      if (moveToResult.hasError()) {
        state.lastError = moveToResult.error();
      } else {
        state.lastError.clear();
      }
    }

    ImGui::SameLine();
    if (ImGui::Button("Cancel MoveTo")) {
      controller->cancelMoveTo();
      state.lastError.clear();
    }

    ImGui::Text("Status: %s",
                controller->isMoveToActive() ? "MoveTo Active" : "Idle");
    if (!state.lastError.empty()) {
      ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Last Error: %s",
                         state.lastError.c_str());
    }
  }

  ImGui::TextUnformatted("Manual look/move input cancels MoveTo.");

  ImGui::End();
}

} // namespace nuri
