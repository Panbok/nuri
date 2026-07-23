#include "nuri/editor_pch.h"

#include "nuri/ui/camera_controller_widget.h"

#include "nuri/core/log.h"

#include <cmath>
#include <format>

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

void logCurrentCameraPose(const Camera &camera) {
#if defined(NURI_LOGGING) && NURI_LOGGING
  const glm::vec3 position = camera.position();
  const glm::vec3 direction = camera.forward();
  NURI_LOG_INFO("Camera Controller: position=(%.6f, %.6f, %.6f) "
                "direction=(%.6f, %.6f, %.6f)",
                position.x, position.y, position.z, direction.x, direction.y,
                direction.z);
#else
  (void)camera;
#endif
}

std::string makeAutotestCameraReproJson(const Camera &camera) {
  const glm::vec3 position = camera.position();
  const glm::vec3 direction = camera.forward();
  const PerspectiveParams &perspective = camera.perspective();
  const ImGuiIO &io = ImGui::GetIO();
  const int framebufferWidth =
      std::max(1, static_cast<int>(std::lround(io.DisplaySize.x *
                                               io.DisplayFramebufferScale.x)));
  const int framebufferHeight =
      std::max(1, static_cast<int>(std::lround(io.DisplaySize.y *
                                               io.DisplayFramebufferScale.y)));

  return std::format("{{\n"
                     "  \"resolution\": [{}, {}],\n"
                     "  \"camera\": {{\n"
                     "    \"position\": [{:.9g}, {:.9g}, {:.9g}],\n"
                     "    \"direction\": [{:.9g}, {:.9g}, {:.9g}],\n"
                     "    \"verticalFovDegrees\": {:.9g},\n"
                     "    \"nearPlane\": {:.9g},\n"
                     "    \"farPlane\": {:.9g}\n"
                     "  }}\n"
                     "}}",
                     framebufferWidth, framebufferHeight, position.x,
                     position.y, position.z, direction.x, direction.y,
                     direction.z, glm::degrees(perspective.fovYRadians),
                     perspective.nearPlane, perspective.farPlane);
}

void drawActiveCameraPose(const Camera &camera) {
  const glm::vec3 position = camera.position();
  const glm::vec3 direction = camera.forward();
  const glm::quat orientation = camera.orientation();
  const float pitchDegrees =
      glm::degrees(std::asin(glm::clamp(direction.y, -1.0f, 1.0f)));
  const float yawDegrees = glm::degrees(std::atan2(direction.x, -direction.z));
  const ImGuiIO &io = ImGui::GetIO();
  const int framebufferWidth =
      std::max(1, static_cast<int>(std::lround(io.DisplaySize.x *
                                               io.DisplayFramebufferScale.x)));
  const int framebufferHeight =
      std::max(1, static_cast<int>(std::lround(io.DisplaySize.y *
                                               io.DisplayFramebufferScale.y)));

  ImGui::SeparatorText("Active Camera Pose");
  ImGui::Text("Position: [%.9g, %.9g, %.9g]", position.x, position.y,
              position.z);
  ImGui::Text("Direction: [%.9g, %.9g, %.9g]", direction.x, direction.y,
              direction.z);
  ImGui::Text("Yaw / Pitch: %.9g / %.9g deg", yawDegrees, pitchDegrees);
  ImGui::Text("Quaternion (wxyz): [%.9g, %.9g, %.9g, %.9g]", orientation.w,
              orientation.x, orientation.y, orientation.z);
  ImGui::Text("Framebuffer: %d x %d", framebufferWidth, framebufferHeight);

  if (camera.projectionType() == ProjectionType::Perspective) {
    const PerspectiveParams &perspective = camera.perspective();
    ImGui::Text("Perspective: FOV %.9g deg, near %.9g, far %.9g",
                glm::degrees(perspective.fovYRadians), perspective.nearPlane,
                perspective.farPlane);
    if (ImGui::Button("Copy Autotest Repro JSON")) {
      const std::string payload = makeAutotestCameraReproJson(camera);
      ImGui::SetClipboardText(payload.c_str());
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip(
          "Copies framebuffer resolution and an autotest-ready camera object.");
    }
  } else {
    const OrthographicParams &orthographic = camera.orthographic();
    ImGui::Text("Orthographic: height %.9g, near %.9g, far %.9g",
                orthographic.height, orthographic.nearPlane,
                orthographic.farPlane);
    ImGui::TextDisabled(
        "Autotest camera export requires perspective projection.");
  }
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

void drawCameraControllerContents(CameraSystem &cameraSystem,
                                  CameraControllerWidgetState &state) {
  CameraController *controller = cameraSystem.activeController();
  Camera *camera = cameraSystem.activeCamera();
  if (!controller || !camera) {
    ImGui::TextUnformatted("No active camera/controller");
    return;
  }

  static_assert(static_cast<int>(CameraPreset::Count) == 2,
                "Update presetNames when CameraPreset changes");
  int presetIndex = static_cast<int>(controller->preset());
  if (ImGui::Combo("Preset", &presetIndex, kPresetNames,
                   IM_ARRAYSIZE(kPresetNames))) {
    controller->setPreset(static_cast<CameraPreset>(presetIndex));
  }

  if (ImGui::Button("Log Camera Pose")) {
    logCurrentCameraPose(*camera);
  }

  drawActiveCameraPose(*camera);

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
}

void drawCameraHelpContents() { drawCameraControlScheme(); }

} // namespace nuri
