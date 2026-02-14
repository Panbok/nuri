#pragma once

#include "nuri/defines.h"
#include "nuri/scene/camera.h"
#include "nuri/scene/camera_controller.h"

#include <cstdint>
#include <memory_resource>
#include <vector>

namespace nuri {

struct CameraHandle {
  static constexpr uint32_t kInvalidIndex = UINT32_MAX;
  uint32_t index = kInvalidIndex;

  [[nodiscard]] bool isValid() const noexcept { return index != kInvalidIndex; }
};

class NURI_API CameraSystem {
public:
  explicit CameraSystem(std::pmr::memory_resource &memory);
  ~CameraSystem() = default;

  CameraSystem(const CameraSystem &) = delete;
  CameraSystem &operator=(const CameraSystem &) = delete;
  CameraSystem(CameraSystem &&) = delete;
  CameraSystem &operator=(CameraSystem &&) = delete;

  CameraHandle addCamera(const Camera &camera,
                         const CameraController &controller);
  bool setActiveCamera(CameraHandle handle, Window &window);

  [[nodiscard]] Camera *activeCamera();
  [[nodiscard]] const Camera *activeCamera() const;
  [[nodiscard]] CameraController *activeController();
  [[nodiscard]] const CameraController *activeController() const;

  [[nodiscard]] Camera *camera(CameraHandle handle);
  [[nodiscard]] const Camera *camera(CameraHandle handle) const;
  [[nodiscard]] CameraController *controller(CameraHandle handle);
  [[nodiscard]] const CameraController *controller(CameraHandle handle) const;

  [[nodiscard]] CameraHandle activeCameraHandle() const noexcept {
    return CameraHandle{activeIndex_};
  }

  bool onInput(const InputEvent &event, Window &window);
  void update(double deltaTime, const InputSystem &input);
  bool toggleActiveProjection();

private:
  struct CameraSlot {
    Camera camera{};
    CameraController controller{};
  };

  [[nodiscard]] bool isValidHandle(CameraHandle handle) const;
  [[nodiscard]] CameraSlot *activeSlot();
  [[nodiscard]] const CameraSlot *activeSlot() const;

  std::pmr::vector<CameraSlot> cameras_;
  uint32_t activeIndex_ = CameraHandle::kInvalidIndex;
};

} // namespace nuri
