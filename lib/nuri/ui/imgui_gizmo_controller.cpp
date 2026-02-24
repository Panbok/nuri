#include "nuri/pch.h"

#include "nuri/ui/imgui_gizmo_controller.h"

#include "nuri/scene/camera_system.h"
#include "nuri/scene/render_scene.h"

#include <ImGuizmo.h>

namespace nuri {

struct ImGuizmoController::Impl {
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
    if (isGizmoConsumingMouseInput()) {
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
  }

  void drawUi() {
    if (ImGui::GetCurrentContext() == nullptr) {
      gizmoHoverOrUsing = false;
      return;
    }
    updateSelectionFromPickResults();

    const OpaqueRenderable *selectedRenderable = nullptr;
    if (selectedOpaqueIndex.has_value()) {
      selectedRenderable = scene.opaqueRenderable(*selectedOpaqueIndex);
      if (selectedRenderable == nullptr) {
        clearSelectionState();
      }
    }

    if (ImGui::Begin("Selection")) {
      if (selectedOpaqueIndex.has_value()) {
        ImGui::Text("Renderable Index: %u", *selectedOpaqueIndex);
      } else {
        ImGui::TextUnformatted("No selection");
      }

      if (selectedOpaqueIndex.has_value() && ImGui::Button("Clear Selection")) {
        clearSelectionState();
        selectedRenderable = nullptr;
      }

      ImGui::Separator();
      ImGui::TextUnformatted("Gizmo");

      if (ImGui::RadioButton("Translate",
                             gizmoOperation == ImGuizmo::TRANSLATE)) {
        gizmoOperation = ImGuizmo::TRANSLATE;
      }
      ImGui::SameLine();
      if (ImGui::RadioButton("Rotate", gizmoOperation == ImGuizmo::ROTATE)) {
        gizmoOperation = ImGuizmo::ROTATE;
      }
      ImGui::SameLine();
      if (ImGui::RadioButton("Scale", gizmoOperation == ImGuizmo::SCALE)) {
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
          ImGui::SliderFloat("Angle Snap", &gizmoAngleSnapDegrees, 1.0f, 90.0f,
                             "%.1f");
        } else {
          ImGui::SliderFloat3("Snap", glm::value_ptr(gizmoSnapValues), 0.01f,
                              100.0f, "%.2f");
        }
      }
    }
    ImGui::End();

    if (!selectedOpaqueIndex.has_value() || selectedRenderable == nullptr ||
        frame == nullptr) {
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

    glm::mat4 modelMatrix = selectedRenderable->modelMatrix;
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
        glm::value_ptr(view), glm::value_ptr(proj), gizmoOperation, gizmoMode,
        glm::value_ptr(modelMatrix), nullptr, snap);
    gizmoHoverOrUsing = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
    if (manipulated &&
        !scene.setOpaqueRenderableTransform(*selectedOpaqueIndex, modelMatrix)) {
      clearSelectionState();
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
    if (ImGui::GetCurrentContext() == nullptr) {
      return;
    }

    int32_t framebufferWidth = 0;
    int32_t framebufferHeight = 0;
    gpu.getFramebufferSize(framebufferWidth, framebufferHeight);
    if (framebufferWidth <= 0 || framebufferHeight <= 0) {
      return;
    }

    const ImGuiIO &io = ImGui::GetIO();
    if (!std::isfinite(io.MousePos.x) || !std::isfinite(io.MousePos.y)) {
      return;
    }

    double framebufferX = static_cast<double>(io.MousePos.x);
    double framebufferY = static_cast<double>(io.MousePos.y);
    if (io.DisplayFramebufferScale.x > 0.0f &&
        io.DisplayFramebufferScale.y > 0.0f) {
      framebufferX *= static_cast<double>(io.DisplayFramebufferScale.x);
      framebufferY *= static_cast<double>(io.DisplayFramebufferScale.y);
    }

    const uint32_t safeWidth = static_cast<uint32_t>(framebufferWidth);
    const uint32_t safeHeight = static_cast<uint32_t>(framebufferHeight);
    const uint32_t x = std::min<uint32_t>(
        static_cast<uint32_t>(std::max(0.0, std::floor(framebufferX))),
        safeWidth - 1u);
    const uint32_t y = std::min<uint32_t>(
        static_cast<uint32_t>(std::max(0.0, std::floor(framebufferY))),
        safeHeight - 1u);

    pendingPickRequest = OpaquePickRequest{
        .x = x,
        .y = y,
        .requestId = nextPickRequestId++,
    };
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

    if (scene.opaqueRenderable(pickResult.renderableIndex) == nullptr) {
      clearSelectionState();
      return;
    }
    selectedOpaqueIndex = pickResult.renderableIndex;
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

  void clearSelectionState() {
    selectedOpaqueIndex.reset();
    gizmoHoverOrUsing = false;
  }

  RenderScene &scene;
  CameraSystem &cameraSystem;
  GPUDevice &gpu;
  RenderFrameContext *frame = nullptr;
  std::optional<OpaquePickRequest> pendingPickRequest{};
  std::optional<uint32_t> selectedOpaqueIndex{};
  uint64_t nextPickRequestId = 1;
  uint64_t pickRequestFloorId = 1;
  ImGuizmo::OPERATION gizmoOperation = ImGuizmo::TRANSLATE;
  ImGuizmo::MODE gizmoMode = ImGuizmo::LOCAL;
  bool gizmoUseSnap = false;
  glm::vec3 gizmoSnapValues{1.0f, 1.0f, 1.0f};
  float gizmoAngleSnapDegrees = 15.0f;
  bool gizmoHoverOrUsing = false;
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

void ImGuizmoController::drawUi() { impl_->drawUi(); }

void ImGuizmoController::reset() { impl_->reset(); }

std::shared_ptr<GizmoController>
createImGuizmoController(const EditorServices &services) {
  return std::make_shared<ImGuizmoController>(services);
}

} // namespace nuri
