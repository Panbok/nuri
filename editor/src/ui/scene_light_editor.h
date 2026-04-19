#pragma once

#include "nuri/core/log.h"
#include "nuri/math/light.h"
#include "nuri/scene/scene_graph.h"

#include <ImGui.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace nuri {

struct LightEditorDraft {
  LightId id = kInvalidLightId;
  LightDesc light{};
  std::array<char, 128> nameBuffer{};
  std::array<char, 32> intensityBuffer{};
  std::array<char, 32> rangeBuffer{};
  std::array<char, 32> innerConeDegreesBuffer{};
  std::array<char, 32> outerConeDegreesBuffer{};
  std::array<char, 32> angularRadiusDegreesBuffer{};
};

[[nodiscard]] inline const char *lightTypeName(LightType type) {
  switch (type) {
  case LightType::Directional:
    return "Directional";
  case LightType::Point:
    return "Point";
  case LightType::Spot:
    return "Spot";
  }
  return "Unknown";
}

inline void writeFloatBuffer(std::span<char> buffer, float value,
                             const char *format) {
  if (buffer.empty()) {
    return;
  }
  std::snprintf(buffer.data(), buffer.size(), format, value);
  buffer.back() = '\0';
}

[[nodiscard]] inline bool tryParseFloatBuffer(const char *buffer,
                                              float &outValue) {
  if (buffer == nullptr || buffer[0] == '\0') {
    return false;
  }

  char *end = nullptr;
  errno = 0;
  const float parsed = std::strtof(buffer, &end);
  if (end == buffer || errno == ERANGE) {
    return false;
  }

  while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end)) != 0) {
    ++end;
  }
  if (*end != '\0' || !std::isfinite(parsed)) {
    return false;
  }

  outValue = parsed;
  return true;
}

[[nodiscard]] inline bool drawFloatTextStepper(std::string_view label,
                                               std::span<char> buffer,
                                               float &value, float step,
                                               float minValue, float maxValue,
                                               const char *format) {
  bool changed = false;

  ImGui::PushID(label.data(), label.data() + label.size());
  if (ImGui::Button("-")) {
    value = std::clamp(value - step, minValue, maxValue);
    writeFloatBuffer(buffer, value, format);
    changed = true;
  }
  ImGui::SameLine();
  ImGui::SetNextItemWidth(150.0f);
  const bool edited = ImGui::InputText("##value", buffer.data(), buffer.size(),
                                       ImGuiInputTextFlags_CharsScientific);
  if (edited) {
    float parsedValue = value;
    if (tryParseFloatBuffer(buffer.data(), parsedValue)) {
      value = std::clamp(parsedValue, minValue, maxValue);
      changed = true;
    }
  }
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    writeFloatBuffer(buffer, value, format);
  }
  ImGui::SameLine();
  if (ImGui::Button("+")) {
    value = std::clamp(value + step, minValue, maxValue);
    writeFloatBuffer(buffer, value, format);
    changed = true;
  }
  ImGui::SameLine();
  ImGui::TextUnformatted(label.data(), label.data() + label.size());
  ImGui::PopID();

  return changed;
}

[[nodiscard]] inline glm::vec3 directionFromLightAngles(float thetaDegrees,
                                                        float phiDegrees) {
  const float thetaRadians = glm::radians(std::remainder(
      std::isfinite(thetaDegrees) ? thetaDegrees : 0.0f, 360.0f));
  const float phiRadians = glm::radians(
      std::clamp(std::isfinite(phiDegrees) ? phiDegrees : 0.0f, -89.9f, 89.9f));
  const float cosPhi = std::cos(phiRadians);
  return glm::normalize(glm::vec3(std::sin(thetaRadians) * cosPhi,
                                  -std::sin(phiRadians),
                                  -std::cos(thetaRadians) * cosPhi));
}

[[nodiscard]] inline glm::vec2 lightAnglesFromDirection(glm::vec3 direction) {
  const float length = glm::length(direction);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return glm::vec2(0.0f);
  }
  direction /= length;
  const float phi =
      glm::degrees(std::asin(std::clamp(-direction.y, -1.0f, 1.0f)));
  const float horizontal =
      std::sqrt(std::max(1.0f - direction.y * direction.y, 0.0f));
  if (horizontal <= 1.0e-5f) {
    return glm::vec2(0.0f, phi);
  }
  const float theta = glm::degrees(std::atan2(direction.x, -direction.z));
  return glm::vec2(std::remainder(theta, 360.0f), phi);
}

[[nodiscard]] inline glm::quat rotationFromLightDirection(glm::vec3 direction) {
  const float length = glm::length(direction);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }
  direction /= length;
  const glm::vec3 up = std::abs(direction.y) < 0.99f
                           ? glm::vec3(0.0f, 1.0f, 0.0f)
                           : glm::vec3(0.0f, 0.0f, 1.0f);
  const glm::mat4 view = glm::lookAt(glm::vec3(0.0f), direction, up);
  return glm::normalize(glm::quat_cast(glm::inverse(view)));
}

inline void invalidateLightEditorDraft(LightEditorDraft &draft) {
  draft.id = kInvalidLightId;
  draft.light = LightDesc{};
  draft.nameBuffer.fill('\0');
  draft.intensityBuffer.fill('\0');
  draft.rangeBuffer.fill('\0');
  draft.innerConeDegreesBuffer.fill('\0');
  draft.outerConeDegreesBuffer.fill('\0');
  draft.angularRadiusDegreesBuffer.fill('\0');
}

inline void syncLightEditorDraft(LightEditorDraft &draft,
                                 LightId selectedLightId,
                                 const LightDesc &selectedLight) {
  const bool matchesSelectedLight =
      draft.id == selectedLightId && draft.light.type == selectedLight.type &&
      draft.light.name == selectedLight.name &&
      nuri::vec3ExactEqual(draft.light.position, selectedLight.position) &&
      nuri::quatExactEqual(draft.light.rotation, selectedLight.rotation) &&
      nuri::vec3ExactEqual(draft.light.color, selectedLight.color) &&
      draft.light.intensity == selectedLight.intensity &&
      draft.light.range == selectedLight.range &&
      draft.light.innerConeAngleRadians ==
          selectedLight.innerConeAngleRadians &&
      draft.light.outerConeAngleRadians ==
          selectedLight.outerConeAngleRadians &&
      draft.light.angularRadiusDegrees == selectedLight.angularRadiusDegrees &&
      draft.light.enabled == selectedLight.enabled;
  if (matchesSelectedLight) {
    return;
  }

  draft.id = selectedLightId;
  draft.light = selectedLight;
  std::snprintf(draft.nameBuffer.data(), draft.nameBuffer.size(), "%s",
                selectedLight.name.c_str());
  writeFloatBuffer(draft.intensityBuffer, selectedLight.intensity, "%.3f");
  writeFloatBuffer(draft.rangeBuffer, selectedLight.range, "%.3f");
  writeFloatBuffer(draft.innerConeDegreesBuffer,
                   glm::degrees(selectedLight.innerConeAngleRadians), "%.2f");
  writeFloatBuffer(draft.outerConeDegreesBuffer,
                   glm::degrees(selectedLight.outerConeAngleRadians), "%.2f");
  writeFloatBuffer(draft.angularRadiusDegreesBuffer,
                   selectedLight.angularRadiusDegrees, "%.3f");
}

inline void drawLightEditor(SceneGraph &graph, LightId selectedLightId,
                            const LightDesc &selectedLight,
                            LightEditorDraft &draft) {
  syncLightEditorDraft(draft, selectedLightId, selectedLight);
  ImGui::PushID(static_cast<int>(selectedLightId.value));

  LightDesc &edited = draft.light;
  bool changed = false;

  if (ImGui::InputText("Light Name", draft.nameBuffer.data(),
                       draft.nameBuffer.size())) {
    edited.name = draft.nameBuffer.data();
    changed = true;
  }
  if (ImGui::ColorEdit3("Light Color", glm::value_ptr(edited.color))) {
    changed = true;
  }
  if (ImGui::Checkbox("Light Enabled", &edited.enabled)) {
    changed = true;
  }

  ImGui::SeparatorText("Radiometry");
  if (drawFloatTextStepper("Intensity", draft.intensityBuffer, edited.intensity,
                           0.1f, 0.0f, std::numeric_limits<float>::max(),
                           "%.3f")) {
    changed = true;
  }

  if (edited.type != LightType::Directional &&
      drawFloatTextStepper("Range", draft.rangeBuffer, edited.range, 0.1f, 0.0f,
                           std::numeric_limits<float>::max(), "%.3f")) {
    changed = true;
  }
  if (edited.type == LightType::Directional) {
    if (drawFloatTextStepper(
            "Angular Radius (deg)", draft.angularRadiusDegreesBuffer,
            edited.angularRadiusDegrees, 0.01f, 0.0f, 10.0f, "%.3f")) {
      changed = true;
    }
    ImGui::SeparatorText("Direction");
    const glm::vec3 direction = lightDirectionFromRotation(edited.rotation);
    glm::vec2 angles = lightAnglesFromDirection(direction);
    const bool thetaChanged =
        ImGui::SliderFloat("Theta", &angles.x, -180.0f, 180.0f, "%.2f");
    const bool phiChanged =
        ImGui::SliderFloat("Phi", &angles.y, -85.0f, 85.0f, "%.2f");
    if (thetaChanged || phiChanged) {
      edited.rotation = rotationFromLightDirection(
          directionFromLightAngles(angles.x, angles.y));
      changed = true;
    }
  }
  if (edited.type == LightType::Spot) {
    ImGui::SeparatorText("Spot Cone");
    float innerConeDegrees = glm::degrees(edited.innerConeAngleRadians);
    float outerConeDegrees = glm::degrees(edited.outerConeAngleRadians);
    if (drawFloatTextStepper("Inner Cone Degrees", draft.innerConeDegreesBuffer,
                             innerConeDegrees, 0.5f, 0.0f, 89.0f, "%.2f")) {
      edited.innerConeAngleRadians = glm::radians(innerConeDegrees);
      changed = true;
    }
    if (drawFloatTextStepper("Outer Cone Degrees", draft.outerConeDegreesBuffer,
                             outerConeDegrees, 0.5f, 0.0f, 89.0f, "%.2f")) {
      edited.outerConeAngleRadians = glm::radians(outerConeDegrees);
      changed = true;
    }
    edited.innerConeAngleRadians =
        std::min(edited.innerConeAngleRadians, edited.outerConeAngleRadians);
    writeFloatBuffer(draft.innerConeDegreesBuffer,
                     glm::degrees(edited.innerConeAngleRadians), "%.2f");
  }

  if (changed && !graph.updateLight(selectedLightId, edited)) {
    NURI_LOG_WARNING("drawLightEditor: failed to update %s light (slot=%u)",
                     lightTypeName(selectedLight.type),
                     indexOf(selectedLightId));
  }

  ImGui::PopID();
}

} // namespace nuri
