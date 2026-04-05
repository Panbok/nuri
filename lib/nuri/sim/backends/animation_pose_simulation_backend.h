#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/sim/animation_scene_frame_data.h"
#include "nuri/sim/backends/simulation_backend.h"

#include <memory_resource>

namespace nuri {

class AnimationGpuServices;

class NURI_API AnimationPoseSimulationBackend final
    : public ISimulationBackend {
public:
  explicit AnimationPoseSimulationBackend(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~AnimationPoseSimulationBackend() override;
  AnimationPoseSimulationBackend(const AnimationPoseSimulationBackend &) =
      delete;
  AnimationPoseSimulationBackend &
  operator=(const AnimationPoseSimulationBackend &) = delete;
  AnimationPoseSimulationBackend(AnimationPoseSimulationBackend &&) noexcept;
  AnimationPoseSimulationBackend &
  operator=(AnimationPoseSimulationBackend &&) noexcept;

  [[nodiscard]] SimulationKind kind() const noexcept override {
    return SimulationKind::AnimationPose;
  }

  [[nodiscard]] Result<bool, std::string>
  createInstance(SceneRuntimeHost &host, SimulationHandle handle,
                 const SimulationDesc &desc) override;
  [[nodiscard]] Result<bool, std::string>
  destroyInstance(SceneRuntimeHost &host, SimulationHandle handle) override;
  [[nodiscard]] Result<bool, std::string>
  updateParams(SceneRuntimeHost &host, SimulationHandle handle,
               std::span<const std::byte> params) override;
  [[nodiscard]] Result<bool, std::string>
  executePhase(SceneRuntimeHost &host, SimulationHandle handle,
               SimulationPhase phase,
               const SimulationExecutionContext &context) override;
  [[nodiscard]] Result<bool, std::string>
  prepareSceneFrame(SceneRuntimeHost &host, uint64_t frameIndex);

  void reset();
  void attachGpuServices(AnimationGpuServices *services) noexcept {
    services_ = services;
  }
  [[nodiscard]] const AnimationSceneFrameData *
  currentSceneFrameData() const noexcept;

private:
  struct Impl;
  std::pmr::memory_resource *memory_;
  AnimationGpuServices *services_ = nullptr;
  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
