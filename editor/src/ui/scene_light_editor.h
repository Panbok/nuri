#pragma once

#include "nuri/core/log.h"
#include "nuri/scene/scene_graph.h"

#include <ImGui.h>
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

  ImGui::PushID(label.data());
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
  ImGui::TextUnformatted(label.data());
  ImGui::PopID();

  return changed;
}

inline void invalidateLightEditorDraft(LightEditorDraft &draft) {
  draft.id = kInvalidLightId;
  draft.light = LightDesc{};
  draft.nameBuffer.fill('\0');
  draft.intensityBuffer.fill('\0');
  draft.rangeBuffer.fill('\0');
  draft.innerConeDegreesBuffer.fill('\0');
  draft.outerConeDegreesBuffer.fill('\0');
}

inline void syncLightEditorDraft(LightEditorDraft &draft,
                                 LightId selectedLightId,
                                 const LightDesc &selectedLight) {
  if (draft.id == selectedLightId) {
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
  }

  if (changed && !graph.updateLight(selectedLightId, edited)) {
    NURI_LOG_WARNING("drawLightEditor: failed to update %s light (slot=%u)",
                     lightTypeName(selectedLight.type),
                     indexOf(selectedLightId));
  }

  ImGui::PopID();
}

} // namespace nuri
