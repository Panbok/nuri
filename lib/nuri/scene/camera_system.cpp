#include "nuri/pch.h"

#include "nuri/scene/camera_system.h"

#include "nuri/core/log.h"

namespace nuri {

namespace {
const char *projectionTypeToString(ProjectionType type) {
  switch (type) {
  case ProjectionType::Perspective:
    return "Perspective";
  case ProjectionType::Orthographic:
    return "Orthographic";
  }
  return "Unknown";
}
} // namespace

CameraSystem::CameraSystem(std::pmr::memory_resource &memory)
    : cameras_(&memory) {
  NURI_LOG_DEBUG("CameraSystem::CameraSystem: Camera system created");
}

CameraHandle CameraSystem::addCamera(const Camera &camera,
                                     CameraController controller) {
  cameras_.push_back(CameraSlot{
      .camera = camera,
      .controller = std::move(controller),
  });

  const uint32_t index = static_cast<uint32_t>(cameras_.size() - 1);
  NURI_LOG_DEBUG("CameraSystem::addCamera: Added camera index=%u", index);
  return CameraHandle{index};
}

bool CameraSystem::setActiveCamera(CameraHandle handle, Window &window) {
  if (!isValidHandle(handle)) {
    NURI_LOG_WARNING(
        "CameraSystem::setActiveCamera: Invalid camera handle index=%u",
        handle.index);
    return false;
  }

  if (activeIndex_ == handle.index) {
    return true;
  }

  if (CameraSlot *previous = activeSlot()) {
    previous->controller.onDeactivate(window);
  }

  activeIndex_ = handle.index;

  if (CameraSlot *active = activeSlot()) {
    active->controller.onActivate(window);
  }
  NURI_LOG_INFO(
      "CameraSystem::setActiveCamera: Active camera switched to index=%u",
      activeIndex_);
  return true;
}

Camera *CameraSystem::activeCamera() {
  CameraSlot *slot = activeSlot();
  return slot ? &slot->camera : nullptr;
}

const Camera *CameraSystem::activeCamera() const {
  const CameraSlot *slot = activeSlot();
  return slot ? &slot->camera : nullptr;
}

CameraController *CameraSystem::activeController() {
  CameraSlot *slot = activeSlot();
  return slot ? &slot->controller : nullptr;
}

const CameraController *CameraSystem::activeController() const {
  const CameraSlot *slot = activeSlot();
  return slot ? &slot->controller : nullptr;
}

Camera *CameraSystem::camera(CameraHandle handle) {
  return isValidHandle(handle) ? &cameras_[handle.index].camera : nullptr;
}

const Camera *CameraSystem::camera(CameraHandle handle) const {
  return isValidHandle(handle) ? &cameras_[handle.index].camera : nullptr;
}

CameraController *CameraSystem::controller(CameraHandle handle) {
  return isValidHandle(handle) ? &cameras_[handle.index].controller : nullptr;
}

const CameraController *CameraSystem::controller(CameraHandle handle) const {
  return isValidHandle(handle) ? &cameras_[handle.index].controller : nullptr;
}

bool CameraSystem::onInput(const InputEvent &event, Window &window) {
  CameraSlot *slot = activeSlot();
  if (!slot) {
    return false;
  }

  if (event.type == InputEventType::Key && event.payload.key.key == Key::P &&
      event.payload.key.action == KeyAction::Press) {
    return toggleActiveProjection();
  }

  return slot->controller.onInput(event, window);
}

void CameraSystem::update(double deltaTime, const InputSystem &input) {
  CameraSlot *slot = activeSlot();
  if (!slot) {
    return;
  }

  slot->controller.update(deltaTime, input, slot->camera);
}

bool CameraSystem::toggleActiveProjection() {
  Camera *active = activeCamera();
  if (!active) {
    NURI_LOG_WARNING(
        "CameraSystem::toggleActiveProjection: No active camera to toggle");
    return false;
  }

  switch (active->projectionType()) {
  case ProjectionType::Perspective:
    active->setProjectionType(ProjectionType::Orthographic);
    break;
  case ProjectionType::Orthographic:
    active->setProjectionType(ProjectionType::Perspective);
    break;
  }

  NURI_LOG_INFO("CameraSystem::toggleActiveProjection: Active camera "
                "projection is now %s",
                projectionTypeToString(active->projectionType()));

  return true;
}

bool CameraSystem::isValidHandle(CameraHandle handle) const {
  return handle.isValid() && handle.index < cameras_.size();
}

CameraSystem::CameraSlot *CameraSystem::activeSlot() {
  if (activeIndex_ == CameraHandle::kInvalidIndex ||
      activeIndex_ >= cameras_.size()) {
    return nullptr;
  }
  return &cameras_[activeIndex_];
}

const CameraSystem::CameraSlot *CameraSystem::activeSlot() const {
  if (activeIndex_ == CameraHandle::kInvalidIndex ||
      activeIndex_ >= cameras_.size()) {
    return nullptr;
  }
  return &cameras_[activeIndex_];
}

} // namespace nuri
