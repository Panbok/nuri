#include "nuri/editor_pch.h"

#include "nuri/ui/imgui_gizmo_controller.h"

#include "nuri/core/log.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/scene/camera_system.h"
#include "nuri/scene/render_scene.h"
#include "scene_light_editor.h"

#include <ImGuizmo.h>

namespace nuri {
namespace {

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

[[nodiscard]] bool setNodeWorldTransform(RenderScene &scene, NodeId node,
                                         const glm::mat4 &worldTransform) {
  SceneGraph &graph = scene.graph();
  (void)graph.syncWorldTransforms();
  NodeId parent = kInvalidNodeId;
  if (!graph.getNodeParent(node, parent)) {
    return false;
  }

  glm::mat4 localTransform = worldTransform;
  if (isValid(parent)) {
    glm::mat4 parentWorld(1.0f);
    if (!graph.getCachedNodeWorldTransform(parent, parentWorld)) {
      return false;
    }
    localTransform = glm::inverse(parentWorld) * worldTransform;
  }

  return graph.setNodeLocalTransform(node, localTransform);
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
  explicit Impl(const EditorServices &services)
      : scene(*services.scene), cameraSystem(*services.cameraSystem),
        gpu(*services.gpu), selectionState(services.selectionState != nullptr
                                               ? services.selectionState
                                               : &localSelectionState) {
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
      frame->shadowInspectRequest =
          ShadowInspectRequest{.x = pendingPickRequest->x,
                               .y = pendingPickRequest->y,
                               .requestId = pendingPickRequest->requestId};
      pendingPickRequest.reset();
    }
    if (selectionState->kind == SceneSelectionKind::Light &&
        isValid(selectionState->lightId)) {
      frame->sharedResources.selectedLightId = selectionState->lightId;
    } else {
      frame->sharedResources.selectedLightId.reset();
    }
  }

  void drawUi(const GizmoUiDrawConfig &config) {
    if (ImGui::GetCurrentContext() == nullptr) {
      gizmoHoverOrUsing = false;
      return;
    }
    updateSelectionFromPickResults();

    SceneGraph &graph = scene.graph();
    (void)graph.syncWorldTransforms();
    const Renderable *selectedRenderable = nullptr;
    if (selectionState->kind == SceneSelectionKind::NodeRenderable) {
      selectedRenderable = scene.renderable(selectionState->renderableIndex);
      if (selectedRenderable == nullptr) {
        demoteToNodeSelection();
      } else if (selectedRenderable->id != selectionState->renderableId) {
        demoteToNodeSelection();
        selectedRenderable = nullptr;
      }
    }
    LightDesc selectedLight{};
    LightDesc selectedLightWorld{};
    NodeId selectedLightNode = kInvalidNodeId;
    const bool hasSelectedLight =
        selectionState->kind == SceneSelectionKind::Light &&
        graph.getLightDesc(selectionState->lightId, selectedLight) &&
        graph.getCachedLightWorldDesc(selectionState->lightId,
                                      selectedLightWorld) &&
        graph.getLightNode(selectionState->lightId, selectedLightNode);
    if (selectionState->kind == SceneSelectionKind::Light &&
        !hasSelectedLight) {
      demoteToNodeSelection();
    }

    const bool wantsLightsWindow =
        config.showLightsWindow != nullptr && *config.showLightsWindow;
    if (wantsLightsWindow) {
      if (ImGui::Begin(config.lightsWindowTitle.data(),
                       config.showLightsWindow)) {
        if (selectionState->kind == SceneSelectionKind::Light &&
            hasSelectedLight) {
          ImGui::Text("Light: %s", lightTypeName(selectedLight.type));
          ImGui::Text("Light Slot: %u", indexOf(selectionState->lightId));
        } else {
          ImGui::TextUnformatted("No light selected");
        }

        if (selectionState->kind == SceneSelectionKind::Light &&
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

        if (selectionState->kind == SceneSelectionKind::Light &&
            hasSelectedLight) {
          if (ImGui::Button("Delete Selected Light")) {
            (void)graph.removeLight(selectionState->lightId);
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
        if (selectionState->kind == SceneSelectionKind::NodeRenderable &&
            selectedRenderable != nullptr) {
          ImGui::Text("Renderable Index: %u", selectionState->renderableIndex);
        } else if (selectionState->kind == SceneSelectionKind::Light &&
                   hasSelectedLight) {
          ImGui::Text("Light: %s", lightTypeName(selectedLight.type));
          ImGui::Text("Light Slot: %u", indexOf(selectionState->lightId));
        } else if (isValid(selectionState->node)) {
          ImGui::Text("Node: %u", indexOf(selectionState->node));
        } else {
          ImGui::TextUnformatted("No selection");
        }

        if (isValid(selectionState->node) && ImGui::Button("Clear Selection")) {
          clearSelectionState();
          selectedRenderable = nullptr;
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Gizmo");

        const bool scaleAllowed =
            selectionState->kind != SceneSelectionKind::Light;
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
    NodeId selectedNode = selectionState->node;
    if (selectionState->kind == SceneSelectionKind::NodeRenderable) {
      if (selectedRenderable == nullptr) {
        demoteToNodeSelection();
        selectedNode = selectionState->node;
      } else {
        selectedNode = selectedRenderable->node;
      }
    } else if (selectionState->kind == SceneSelectionKind::Light &&
               hasSelectedLight) {
      selectedNode = selectedLightNode;
    }

    if (selectionState->kind == SceneSelectionKind::Light && hasSelectedLight) {
      if (!graph.getCachedNodeWorldTransform(selectedLightNode, modelMatrix)) {
        clearSelectionState();
        return;
      }
      if (selectedLightWorld.type == LightType::Directional) {
        effectiveOperation = ImGuizmo::ROTATE;
      } else if (selectedLightWorld.type == LightType::Point) {
        effectiveOperation = ImGuizmo::TRANSLATE;
      } else if (effectiveOperation == ImGuizmo::SCALE) {
        effectiveOperation = ImGuizmo::TRANSLATE;
      }
    } else if (isValid(selectedNode) &&
               graph.getCachedNodeWorldTransform(selectedNode, modelMatrix)) {
      // Node-only and renderable selections both manipulate the node transform.
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

    if (selectionState->kind == SceneSelectionKind::Light && hasSelectedLight) {
      if (!setNodeWorldTransform(scene, selectedLightNode, modelMatrix)) {
        clearSelectionState();
      }
      return;
    }

    if (!setNodeWorldTransform(scene, selectedNode, modelMatrix)) {
      clearSelectionState();
    }
  }

  void reset() {
    invalidatePendingPicks();
    clearSelectionState();
    frame = nullptr;
  }

  void invalidatePendingPicks() {
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
      if (!frame->scene->graph().getCachedLightWorldDesc(lightId, light)) {
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

    NodeId lightNode = kInvalidNodeId;
    if (!scene.graph().getLightNode(closestLight, lightNode)) {
      return false;
    }
    selectionState->kind = SceneSelectionKind::Light;
    selectionState->node = lightNode;
    selectionState->renderableId = kInvalidRenderableId;
    selectionState->renderableIndex = 0u;
    selectionState->lightId = closestLight;
    invalidateLightEditorDraft(lightEditorDraft);
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

    const Renderable *renderable = scene.renderable(pickResult.renderableIndex);
    if (renderable == nullptr || !isValid(renderable->node)) {
      clearSelectionState();
      return;
    }
    selectionState->kind = SceneSelectionKind::NodeRenderable;
    selectionState->node = renderable->node;
    selectionState->renderableId = renderable->id;
    selectionState->renderableIndex = pickResult.renderableIndex;
    selectionState->lightId = kInvalidLightId;
    invalidateLightEditorDraft(lightEditorDraft);
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
    SceneGraph &graph = scene.graph();
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

    auto nodeResult = graph.createNode(graph.rootNode(), desc.name,
                                       lightTransformMatrix(desc));
    if (nodeResult.hasError()) {
      NURI_LOG_WARNING("ImGuizmoController::spawnLight: failed to create %s "
                       "light node: %s",
                       lightTypeName(type), nodeResult.error().c_str());
      return;
    }
    desc.position = glm::vec3(0.0f);
    desc.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    auto addResult = graph.addLight(nodeResult.value(), desc);
    if (addResult.hasError()) {
      (void)graph.destroyNodeSubtree(nodeResult.value());
      NURI_LOG_WARNING("ImGuizmoController::spawnLight: failed to spawn %s "
                       "light: %s",
                       lightTypeName(type), addResult.error().c_str());
      return;
    }
    selectionState->kind = SceneSelectionKind::Light;
    selectionState->node = nodeResult.value();
    selectionState->renderableId = kInvalidRenderableId;
    selectionState->renderableIndex = 0u;
    selectionState->lightId = addResult.value();
    invalidateLightEditorDraft(lightEditorDraft);
    NURI_LOG_INFO("ImGuizmoController::spawnLight: spawned %s light (slot=%u)",
                  lightTypeName(type), indexOf(selectionState->lightId));
  }

  void drawSelectedLightEditor(const LightDesc &selectedLight) {
    drawLightEditor(scene.graph(), selectionState->lightId, selectedLight,
                    lightEditorDraft);
  }

  void clearSelectionState() {
    selectionState->clear();
    invalidateLightEditorDraft(lightEditorDraft);
    gizmoHoverOrUsing = false;
  }

  void demoteToNodeSelection() {
    selectionState->kind = SceneSelectionKind::Node;
    selectionState->renderableId = kInvalidRenderableId;
    selectionState->renderableIndex = 0u;
    selectionState->lightId = kInvalidLightId;
    invalidateLightEditorDraft(lightEditorDraft);
  }

  RenderScene &scene;
  CameraSystem &cameraSystem;
  GPUDevice &gpu;
  SceneEditorSelectionState localSelectionState{};
  SceneEditorSelectionState *selectionState = nullptr;
  RenderFrameContext *frame = nullptr;
  std::optional<OpaquePickRequest> pendingPickRequest{};
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

void ImGuizmoController::invalidatePendingPicks() {
  impl_->invalidatePendingPicks();
}

void ImGuizmoController::reset() { impl_->reset(); }

std::shared_ptr<GizmoController>
createImGuizmoController(const EditorServices &services) {
  return std::make_shared<ImGuizmoController>(services);
}

} // namespace nuri
