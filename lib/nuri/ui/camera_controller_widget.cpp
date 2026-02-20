#include "nuri/pch.h"

#include "nuri/ui/camera_controller_widget.h"

namespace nuri {

namespace {

constexpr float kMinMoveToDurationSeconds = 0.01f;
constexpr const char *kPresetNames[] = {"FPS + Direct", "FPS + MoveTo"};

void drawCameraControlScheme() {
  ImGui::TextUnformatted("Camera Controls");
  ImGui::TextUnformatted("Move               : W / A / S / D");
  ImGui::TextUnformatted("Vertical Move      : Q / E");
  ImGui::TextUnformatted("Speed Boost        : LeftShift / RightShift");
  ImGui::TextUnformatted("Free Camera Look   : Hold RMB + Mouse");
  ImGui::TextUnformatted("Projection Toggle  : P");
}

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

  static_assert(static_cast<int>(CameraPreset::Count) == 2,
                "Update presetNames when CameraPreset changes");
  int presetIndex = static_cast<int>(controller->preset());
  if (ImGui::Combo("Preset", &presetIndex, kPresetNames,
                   IM_ARRAYSIZE(kPresetNames))) {
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
    state.durationSeconds =
        std::max(state.durationSeconds, kMinMoveToDurationSeconds);

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

    ImGui::TextUnformatted("Manual look/move input cancels MoveTo.");
  }

  ImGui::Separator();
  drawCameraControlScheme();

  ImGui::End();
}

bool drawScenePresetWidget(std::span<const char *const> presetNames,
                           int &selectedIndex, std::string_view hotkeyHint) {
  if (presetNames.empty()) {
    return false;
  }
  if (!ImGui::Begin("Scene Preset")) {
    ImGui::End();
    return false;
  }

  selectedIndex =
      std::clamp(selectedIndex, 0, static_cast<int>(presetNames.size()) - 1);
  const bool changed =
      ImGui::Combo("Preset", &selectedIndex, presetNames.data(),
                   static_cast<int>(presetNames.size()));
  ImGui::TextUnformatted(hotkeyHint.data(),
                         hotkeyHint.data() + hotkeyHint.size());
  ImGui::End();
  return changed;
}

} // namespace nuri
