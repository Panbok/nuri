#include "nuri/editor_pch.h"

#include "nuri/ui/imgui_gizmo_controller.h"

#include "nuri/core/log.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/scene/camera_system.h"
#include "nuri/scene/render_scene.h"

#include <ImGuizmo.h>
#include <array>
#include <cerrno>
#include <cstdlib>

namespace nuri {
namespace {

enum class SelectionKind : uint8_t {
  None = 0,
  Renderable = 1,
  Light = 2,
};

struct Ray {
  glm::vec3 origin{0.0f};
  glm::vec3 direction{0.0f, 0.0f, -1.0f};
};

[[nodiscard]] glm::vec3 safeNormalize(const glm::vec3 &value,
                                      const glm::vec3 &fallback) {
  const float length = glm::length(value);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return fallback;
  }
  return value / length;
}

[[nodiscard]] float lightIconScale(const glm::vec3 &cameraPosition,
                                   const glm::vec3 &lightPosition) {
  const float distance = glm::length(cameraPosition - lightPosition);
  return std::clamp(distance * 0.08f, 0.2f, 3.0f);
}

[[nodiscard]] bool cursorFramebufferPosition(GPUDevice &gpu, uint32_t &outX,
                                             uint32_t &outY, uint32_t &outWidth,
                                             uint32_t &outHeight) {
  if (ImGui::GetCurrentContext() == nullptr) {
    return false;
  }

  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu.getFramebufferSize(framebufferWidth, framebufferHeight);
  if (framebufferWidth <= 0 || framebufferHeight <= 0) {
    return false;
  }

  const ImGuiIO &io = ImGui::GetIO();
  if (!std::isfinite(io.MousePos.x) || !std::isfinite(io.MousePos.y)) {
    return false;
  }

  double framebufferX = static_cast<double>(io.MousePos.x);
  double framebufferY = static_cast<double>(io.MousePos.y);
  if (io.DisplayFramebufferScale.x > 0.0f &&
      io.DisplayFramebufferScale.y > 0.0f) {
    framebufferX *= static_cast<double>(io.DisplayFramebufferScale.x);
    framebufferY *= static_cast<double>(io.DisplayFramebufferScale.y);
  }

  outWidth = static_cast<uint32_t>(framebufferWidth);
  outHeight = static_cast<uint32_t>(framebufferHeight);
  outX = std::min<uint32_t>(
      static_cast<uint32_t>(std::max(0.0, std::floor(framebufferX))),
      outWidth - 1u);
  outY = std::min<uint32_t>(
      static_cast<uint32_t>(std::max(0.0, std::floor(framebufferY))),
      outHeight - 1u);
  return true;
}

[[nodiscard]] Ray buildCameraRay(const RenderFrameContext &frame, uint32_t x,
                                 uint32_t y, uint32_t width, uint32_t height) {
  const float ndcX =
      ((static_cast<float>(x) + 0.5f) / std::max(float(width), 1.0f)) * 2.0f -
      1.0f;
  const float ndcY =
      1.0f -
      ((static_cast<float>(y) + 0.5f) / std::max(float(height), 1.0f)) * 2.0f;

  const glm::mat4 invViewProj =
      glm::inverse(frame.camera.proj * frame.camera.view);
  glm::vec4 nearPoint = invViewProj * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
  glm::vec4 farPoint = invViewProj * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
  nearPoint /= std::max(nearPoint.w, 1.0e-6f);
  farPoint /= std::max(farPoint.w, 1.0e-6f);

  Ray ray{};
  ray.origin = glm::vec3(nearPoint);
  ray.direction = safeNormalize(glm::vec3(farPoint - nearPoint),
                                glm::vec3(0.0f, 0.0f, -1.0f));
  return ray;
}

[[nodiscard]] bool intersectRaySphere(const Ray &ray, const glm::vec3 &center,
                                      float radius, float &outT) {
  const glm::vec3 oc = ray.origin - center;
  const float a = glm::dot(ray.direction, ray.direction);
  const float b = 2.0f * glm::dot(oc, ray.direction);
  const float c = glm::dot(oc, oc) - radius * radius;
  const float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f) {
    return false;
  }

  const float sqrtDiscriminant = std::sqrt(discriminant);
  const float invDenom = 0.5f / std::max(a, 1.0e-6f);
  const float t0 = (-b - sqrtDiscriminant) * invDenom;
  const float t1 = (-b + sqrtDiscriminant) * invDenom;
  if (t0 > 0.0f) {
    outT = t0;
    return true;
  }
  if (t1 > 0.0f) {
    outT = t1;
    return true;
  }
  return false;
}

[[nodiscard]] glm::mat4 lightTransformMatrix(const LightDesc &light) {
  return glm::translate(glm::mat4(1.0f), light.position) *
         glm::mat4_cast(light.rotation);
}

[[nodiscard]] glm::quat extractRotation(const glm::mat4 &transform) {
  glm::mat3 rotationMatrix(transform);
  rotationMatrix[0] =
      safeNormalize(rotationMatrix[0], glm::vec3(1.0f, 0.0f, 0.0f));
  rotationMatrix[1] =
      safeNormalize(rotationMatrix[1], glm::vec3(0.0f, 1.0f, 0.0f));
  rotationMatrix[2] =
      safeNormalize(rotationMatrix[2], glm::vec3(0.0f, 0.0f, 1.0f));
  return glm::normalize(glm::quat_cast(rotationMatrix));
}

[[nodiscard]] const char *lightTypeName(LightType type) {
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

void writeFloatBuffer(std::span<char> buffer, float value, const char *format) {
  if (buffer.empty()) {
    return;
  }
  std::snprintf(buffer.data(), buffer.size(), format, value);
  buffer.back() = '\0';
}

[[nodiscard]] bool tryParseFloatBuffer(const char *buffer, float &outValue) {
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

[[nodiscard]] bool drawFloatTextStepper(std::string_view label,
                                        std::span<char> buffer, float &value,
                                        float step, float minValue,
                                        float maxValue, const char *format) {
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

[[nodiscard]] bool imguiOwnsMouseInput() {
  if (ImGui::GetCurrentContext() == nullptr) {
    return false;
  }

  const ImGuiIO &io = ImGui::GetIO();
  if (io.WantCaptureMouse) {
    return true;
  }

  return ImGui::IsAnyItemHovered() || ImGui::IsAnyItemActive() ||
         ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
}

} // namespace

struct ImGuizmoController::Impl {
  struct LightEditorDraft {
    LightId id = kInvalidLightId;
    LightDesc light{};
    std::array<char, 128> nameBuffer{};
    std::array<char, 32> intensityBuffer{};
    std::array<char, 32> rangeBuffer{};
    std::array<char, 32> innerConeDegreesBuffer{};
    std::array<char, 32> outerConeDegreesBuffer{};
  };

  explicit Impl(const EditorServices &services)
      : scene(*services.scene), cameraSystem(*services.cameraSystem),
        gpu(*services.gpu) {
    NURI_ASSERT(services.hasAllDependencies(),
                "ImGuizmoController requires valid editor services");
  }

  bool onInput(const InputEvent &event) {
    if (event.type != InputEventType::MouseButton ||
        event.payload.mouseButton.button != MouseButton::Left ||
        event.payload.mouseButton.action != MouseAction::Press) {
      return false;
    }
    if (imguiOwnsMouseInput()) {
      return true;
    }
    if (isGizmoConsumingMouseInput()) {
      return true;
    }
    if (tryPickLightAtCursor()) {
      pendingPickRequest.reset();
      pickRequestFloorId = nextPickRequestId;
      return true;
    }
    queuePickAtCursor();
    return true;
  }

  void onFrame(RenderFrameContext &frameIn) {
    frame = &frameIn;
    if (pendingPickRequest.has_value()) {
      frame->opaquePickRequest = pendingPickRequest;
      pendingPickRequest.reset();
    }
    frame->channels.publish<LightId>(kFrameChannelSelectedLightId,
                                     selectionKind == SelectionKind::Light
                                         ? selectedLightId
                                         : kInvalidLightId);
  }

  void drawUi(const GizmoUiDrawConfig &config) {
    if (ImGui::GetCurrentContext() == nullptr) {
      gizmoHoverOrUsing = false;
      return;
    }
    updateSelectionFromPickResults();

    const Renderable *selectedRenderable = nullptr;
    if (selectionKind == SelectionKind::Renderable &&
        selectedOpaqueIndex.has_value()) {
      selectedRenderable = scene.renderable(*selectedOpaqueIndex);
      if (selectedRenderable == nullptr) {
        clearSelectionState();
      }
    }
    LightDesc selectedLight{};
    const bool hasSelectedLight =
        selectionKind == SelectionKind::Light &&
        scene.getLightDesc(selectedLightId, selectedLight);
    if (selectionKind == SelectionKind::Light && !hasSelectedLight) {
      clearSelectionState();
    }

    const bool wantsLightsWindow =
        config.showLightsWindow != nullptr && *config.showLightsWindow;
    if (wantsLightsWindow) {
      if (ImGui::Begin(config.lightsWindowTitle.data(),
                       config.showLightsWindow)) {
        if (selectionKind == SelectionKind::Light && hasSelectedLight) {
          ImGui::Text("Light: %s", lightTypeName(selectedLight.type));
          ImGui::Text("Light Slot: %u", indexOf(selectedLightId));
        } else {
          ImGui::TextUnformatted("No light selected");
        }

        if (selectionKind == SelectionKind::Light &&
            ImGui::Button("Clear Light Selection")) {
          clearSelectionState();
          selectedRenderable = nullptr;
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Lights");
        if (ImGui::Button("Add Directional")) {
          spawnLight(LightType::Directional);
        }
        ImGui::SameLine();
        if (ImGui::Button("Add Point")) {
          spawnLight(LightType::Point);
        }
        ImGui::SameLine();
        if (ImGui::Button("Add Spot")) {
          spawnLight(LightType::Spot);
        }

        if (selectionKind == SelectionKind::Light && hasSelectedLight) {
          if (ImGui::Button("Delete Selected Light")) {
            (void)scene.removeLight(selectedLightId);
            clearSelectionState();
          } else {
            drawSelectedLightEditor(selectedLight);
          }
        }
      }
      ImGui::End();
    }

    const bool wantsControlsWindow =
        config.showControlsWindow != nullptr && *config.showControlsWindow;
    if (wantsControlsWindow) {
      if (ImGui::Begin(config.controlsWindowTitle.data(),
                       config.showControlsWindow)) {
        if (selectionKind == SelectionKind::Renderable &&
            selectedOpaqueIndex.has_value()) {
          ImGui::Text("Renderable Index: %u", *selectedOpaqueIndex);
        } else if (selectionKind == SelectionKind::Light && hasSelectedLight) {
          ImGui::Text("Light: %s", lightTypeName(selectedLight.type));
          ImGui::Text("Light Slot: %u", indexOf(selectedLightId));
        } else {
          ImGui::TextUnformatted("No selection");
        }

        if (selectionKind != SelectionKind::None &&
            ImGui::Button("Clear Selection")) {
          clearSelectionState();
          selectedRenderable = nullptr;
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Gizmo");

        const bool scaleAllowed = selectionKind != SelectionKind::Light;
        if (ImGui::RadioButton("Translate",
                               gizmoOperation == ImGuizmo::TRANSLATE)) {
          gizmoOperation = ImGuizmo::TRANSLATE;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", gizmoOperation == ImGuizmo::ROTATE)) {
          gizmoOperation = ImGuizmo::ROTATE;
        }
        if (scaleAllowed) {
          ImGui::SameLine();
        }
        if (scaleAllowed &&
            ImGui::RadioButton("Scale", gizmoOperation == ImGuizmo::SCALE)) {
          gizmoOperation = ImGuizmo::SCALE;
        }

        if (gizmoOperation != ImGuizmo::SCALE) {
          if (ImGui::RadioButton("Local", gizmoMode == ImGuizmo::LOCAL)) {
            gizmoMode = ImGuizmo::LOCAL;
          }
          ImGui::SameLine();
          if (ImGui::RadioButton("World", gizmoMode == ImGuizmo::WORLD)) {
            gizmoMode = ImGuizmo::WORLD;
          }
        }

        ImGui::Checkbox("Snap", &gizmoUseSnap);
        if (gizmoUseSnap) {
          if (gizmoOperation == ImGuizmo::ROTATE) {
            ImGui::SliderFloat("Angle Snap", &gizmoAngleSnapDegrees, 1.0f,
                               90.0f, "%.1f");
          } else {
            ImGui::SliderFloat3("Snap", glm::value_ptr(gizmoSnapValues), 0.01f,
                                100.0f, "%.2f");
          }
        }
      }
      ImGui::End();
    }

    if (frame == nullptr) {
      clearSelectionState();
      return;
    }

    const Camera *activeCamera = cameraSystem.activeCamera();
    if (activeCamera == nullptr) {
      clearSelectionState();
      return;
    }

    float snapValues[3] = {
        gizmoOperation == ImGuizmo::ROTATE ? gizmoAngleSnapDegrees
                                           : gizmoSnapValues.x,
        gizmoOperation == ImGuizmo::ROTATE ? gizmoAngleSnapDegrees
                                           : gizmoSnapValues.y,
        gizmoOperation == ImGuizmo::ROTATE ? gizmoAngleSnapDegrees
                                           : gizmoSnapValues.z,
    };
    float *snap = gizmoUseSnap ? snapValues : nullptr;

    glm::mat4 modelMatrix(1.0f);
    ImGuizmo::OPERATION effectiveOperation = gizmoOperation;
    if (selectionKind == SelectionKind::Renderable) {
      if (!selectedOpaqueIndex.has_value() || selectedRenderable == nullptr) {
        clearSelectionState();
        return;
      }
      modelMatrix = selectedRenderable->modelMatrix;
    } else if (selectionKind == SelectionKind::Light && hasSelectedLight) {
      modelMatrix = lightTransformMatrix(selectedLight);
      if (selectedLight.type == LightType::Directional) {
        effectiveOperation = ImGuizmo::ROTATE;
      } else if (selectedLight.type == LightType::Point) {
        effectiveOperation = ImGuizmo::TRANSLATE;
      } else if (effectiveOperation == ImGuizmo::SCALE) {
        effectiveOperation = ImGuizmo::TRANSLATE;
      }
    } else {
      gizmoHoverOrUsing = false;
      return;
    }

    glm::mat4 view = frame->camera.view;
    glm::mat4 proj = frame->camera.proj;

    ImGuizmo::BeginFrame();
    ImGuizmo::SetOrthographic(activeCamera->projectionType() ==
                              ProjectionType::Orthographic);
    if (const ImGuiViewport *viewport = ImGui::GetMainViewport()) {
      ImGuizmo::SetRect(viewport->Pos.x, viewport->Pos.y, viewport->Size.x,
                        viewport->Size.y);
    }

    const bool manipulated = ImGuizmo::Manipulate(
        glm::value_ptr(view), glm::value_ptr(proj), effectiveOperation,
        gizmoMode, glm::value_ptr(modelMatrix), nullptr, snap);
    gizmoHoverOrUsing = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
    if (!manipulated) {
      return;
    }

    if (selectionKind == SelectionKind::Renderable) {
      if (!scene.setRenderableTransform(*selectedOpaqueIndex, modelMatrix)) {
        clearSelectionState();
      }
      return;
    }

    if (selectionKind == SelectionKind::Light && hasSelectedLight) {
      glm::vec3 position = glm::vec3(modelMatrix[3]);
      glm::quat rotation = selectedLight.rotation;
      if (effectiveOperation != ImGuizmo::TRANSLATE) {
        rotation = extractRotation(modelMatrix);
      }
      if (effectiveOperation == ImGuizmo::TRANSLATE &&
          selectedLight.type == LightType::Directional) {
        rotation = selectedLight.rotation;
      }
      if (!scene.setLightTransform(selectedLightId, position, rotation)) {
        clearSelectionState();
      }
    }
  }

  void reset() {
    clearSelectionState();
    frame = nullptr;
    pendingPickRequest.reset();
    pickRequestFloorId = nextPickRequestId;
  }

private:
  void queuePickAtCursor() {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!cursorFramebufferPosition(gpu, x, y, width, height)) {
      return;
    }
    pendingPickRequest = OpaquePickRequest{
        .x = x,
        .y = y,
        .requestId = nextPickRequestId++,
    };
  }

  bool tryPickLightAtCursor() {
    if (frame == nullptr || frame->scene == nullptr) {
      return false;
    }

    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!cursorFramebufferPosition(gpu, x, y, width, height)) {
      return false;
    }

    const Ray ray = buildCameraRay(*frame, x, y, width, height);
    float closestT = std::numeric_limits<float>::max();
    LightId closestLight = kInvalidLightId;
    frame->scene->forEachLightId([&](LightId lightId) {
      LightDesc light{};
      if (!frame->scene->getLightDesc(lightId, light)) {
        return;
      }
      float hitT = 0.0f;
      const float radius =
          lightIconScale(glm::vec3(frame->camera.cameraPos), light.position);
      if (intersectRaySphere(ray, light.position, radius, hitT) &&
          hitT < closestT) {
        closestT = hitT;
        closestLight = lightId;
      }
    });

    if (!isValid(closestLight)) {
      return false;
    }

    selectionKind = SelectionKind::Light;
    selectedLightId = closestLight;
    selectedOpaqueIndex.reset();
    return true;
  }

  void updateSelectionFromPickResults() {
    if (frame == nullptr || !frame->opaquePickResult.has_value()) {
      return;
    }
    const OpaquePickResult &pickResult = *frame->opaquePickResult;
    frame->opaquePickResult.reset();
    if (pickResult.requestId < pickRequestFloorId) {
      return;
    }
    if (!pickResult.hit) {
      clearSelectionState();
      return;
    }

    if (scene.renderable(pickResult.renderableIndex) == nullptr) {
      clearSelectionState();
      return;
    }
    selectionKind = SelectionKind::Renderable;
    selectedOpaqueIndex = pickResult.renderableIndex;
    selectedLightId = kInvalidLightId;
    invalidateLightEditorDraft();
  }

  [[nodiscard]] bool isGizmoConsumingMouseInput() const {
    if (gizmoHoverOrUsing) {
      return true;
    }
    if (ImGui::GetCurrentContext() == nullptr) {
      return false;
    }
    return ImGuizmo::IsOver() || ImGuizmo::IsUsing();
  }

  void spawnLight(LightType type) {
    LightDesc desc{};
    desc.type = type;
    desc.name = std::string(lightTypeName(type)) + " Light";
    desc.enabled = true;

    if (const Camera *camera = cameraSystem.activeCamera(); camera != nullptr) {
      desc.position = camera->position() + camera->forward() * 2.0f;
      desc.rotation = camera->orientation();
    }

    switch (type) {
    case LightType::Directional:
      desc.intensity = 2.0f;
      break;
    case LightType::Point:
      desc.intensity = 20.0f;
      desc.range = 15.0f;
      break;
    case LightType::Spot:
      desc.intensity = 20.0f;
      desc.range = 15.0f;
      desc.innerConeAngleRadians = glm::radians(20.0f);
      desc.outerConeAngleRadians = glm::radians(30.0f);
      break;
    }

    auto addResult = scene.addLight(desc);
    if (addResult.hasError()) {
      NURI_LOG_WARNING("ImGuizmoController::spawnLight: failed to spawn %s "
                       "light: %s",
                       lightTypeName(type), addResult.error().c_str());
      return;
    }
    selectionKind = SelectionKind::Light;
    selectedLightId = addResult.value();
    selectedOpaqueIndex.reset();
    NURI_LOG_INFO("ImGuizmoController::spawnLight: spawned %s light (slot=%u)",
                  lightTypeName(type), indexOf(selectedLightId));
  }

  void drawSelectedLightEditor(const LightDesc &selectedLight) {
    syncLightEditorDraft(selectedLight);
    ImGui::PushID(static_cast<int>(selectedLightId.value));

    LightDesc &edited = lightEditorDraft.light;
    bool changed = false;

    if (ImGui::InputText("Light Name", lightEditorDraft.nameBuffer.data(),
                         lightEditorDraft.nameBuffer.size())) {
      edited.name = lightEditorDraft.nameBuffer.data();
      changed = true;
    }
    if (ImGui::ColorEdit3("Light Color", glm::value_ptr(edited.color))) {
      changed = true;
    }
    if (ImGui::Checkbox("Light Enabled", &edited.enabled)) {
      changed = true;
    }

    ImGui::SeparatorText("Radiometry");
    if (drawFloatTextStepper("Intensity", lightEditorDraft.intensityBuffer,
                             edited.intensity, 0.1f, 0.0f,
                             std::numeric_limits<float>::max(), "%.3f")) {
      changed = true;
    }

    if (edited.type != LightType::Directional &&
        drawFloatTextStepper("Range", lightEditorDraft.rangeBuffer,
                             edited.range, 0.1f, 0.0f,
                             std::numeric_limits<float>::max(), "%.3f")) {
      changed = true;
    }
    if (edited.type == LightType::Spot) {
      ImGui::SeparatorText("Spot Cone");
      float innerConeDegrees = glm::degrees(edited.innerConeAngleRadians);
      float outerConeDegrees = glm::degrees(edited.outerConeAngleRadians);
      if (drawFloatTextStepper("Inner Cone Degrees",
                               lightEditorDraft.innerConeDegreesBuffer,
                               innerConeDegrees, 0.5f, 0.0f, 89.0f, "%.2f")) {
        edited.innerConeAngleRadians = glm::radians(innerConeDegrees);
        changed = true;
      }
      if (drawFloatTextStepper("Outer Cone Degrees",
                               lightEditorDraft.outerConeDegreesBuffer,
                               outerConeDegrees, 0.5f, 0.0f, 89.0f, "%.2f")) {
        edited.outerConeAngleRadians = glm::radians(outerConeDegrees);
        changed = true;
      }
      edited.innerConeAngleRadians =
          std::min(edited.innerConeAngleRadians, edited.outerConeAngleRadians);
    }

    if (changed && !scene.updateLight(selectedLightId, edited)) {
      NURI_LOG_WARNING("ImGuizmoController::drawSelectedLightEditor: failed to "
                       "update %s light (slot=%u)",
                       lightTypeName(selectedLight.type),
                       indexOf(selectedLightId));
    }

    ImGui::PopID();
  }

  void clearSelectionState() {
    selectionKind = SelectionKind::None;
    selectedOpaqueIndex.reset();
    selectedLightId = kInvalidLightId;
    invalidateLightEditorDraft();
    gizmoHoverOrUsing = false;
  }

  void invalidateLightEditorDraft() {
    lightEditorDraft.id = kInvalidLightId;
    lightEditorDraft.light = LightDesc{};
    lightEditorDraft.nameBuffer.fill('\0');
    lightEditorDraft.intensityBuffer.fill('\0');
    lightEditorDraft.rangeBuffer.fill('\0');
    lightEditorDraft.innerConeDegreesBuffer.fill('\0');
    lightEditorDraft.outerConeDegreesBuffer.fill('\0');
  }

  void syncLightEditorDraft(const LightDesc &selectedLight) {
    if (lightEditorDraft.id == selectedLightId) {
      return;
    }

    lightEditorDraft.id = selectedLightId;
    lightEditorDraft.light = selectedLight;
    std::snprintf(lightEditorDraft.nameBuffer.data(),
                  lightEditorDraft.nameBuffer.size(), "%s",
                  selectedLight.name.c_str());
    writeFloatBuffer(lightEditorDraft.intensityBuffer, selectedLight.intensity,
                     "%.3f");
    writeFloatBuffer(lightEditorDraft.rangeBuffer, selectedLight.range, "%.3f");
    writeFloatBuffer(lightEditorDraft.innerConeDegreesBuffer,
                     glm::degrees(selectedLight.innerConeAngleRadians), "%.2f");
    writeFloatBuffer(lightEditorDraft.outerConeDegreesBuffer,
                     glm::degrees(selectedLight.outerConeAngleRadians), "%.2f");
  }

  RenderScene &scene;
  CameraSystem &cameraSystem;
  GPUDevice &gpu;
  RenderFrameContext *frame = nullptr;
  std::optional<OpaquePickRequest> pendingPickRequest{};
  SelectionKind selectionKind = SelectionKind::None;
  std::optional<uint32_t> selectedOpaqueIndex{};
  LightId selectedLightId = kInvalidLightId;
  uint64_t nextPickRequestId = 1;
  uint64_t pickRequestFloorId = 1;
  ImGuizmo::OPERATION gizmoOperation = ImGuizmo::TRANSLATE;
  ImGuizmo::MODE gizmoMode = ImGuizmo::LOCAL;
  bool gizmoUseSnap = false;
  glm::vec3 gizmoSnapValues{1.0f, 1.0f, 1.0f};
  float gizmoAngleSnapDegrees = 15.0f;
  bool gizmoHoverOrUsing = false;
  LightEditorDraft lightEditorDraft{};
};

ImGuizmoController::ImGuizmoController(const EditorServices &services)
    : impl_(std::make_unique<Impl>(services)) {}

ImGuizmoController::~ImGuizmoController() = default;

bool ImGuizmoController::onInput(const InputEvent &event) {
  return impl_->onInput(event);
}

void ImGuizmoController::onFrame(RenderFrameContext &frame) {
  impl_->onFrame(frame);
}

void ImGuizmoController::drawUi(const GizmoUiDrawConfig &config) {
  impl_->drawUi(config);
}

void ImGuizmoController::reset() { impl_->reset(); }

std::shared_ptr<GizmoController>
createImGuizmoController(const EditorServices &services) {
  return std::make_shared<ImGuizmoController>(services);
}

} // namespace nuri
