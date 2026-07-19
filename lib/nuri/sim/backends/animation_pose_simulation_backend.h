#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/sim/animation_scene_frame_data.h"
#include "nuri/sim/simulation_desc.h"
#include "nuri/sim/simulation_execution_context.h"
#include <memory_resource>
namespace nuri {

class AnimationGpuServices;
class SceneRuntimeHost;

class NURI_API AnimationPoseSimulationBackend final {
public:
  explicit AnimationPoseSimulationBackend(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~AnimationPoseSimulationBackend();
  AnimationPoseSimulationBackend(const AnimationPoseSimulationBackend &) =
      delete;
  AnimationPoseSimulationBackend &
  operator=(const AnimationPoseSimulationBackend &) = delete;
  [[nodiscard]] Result<bool, std::string>
  createInstance(SceneRuntimeHost &host, SimulationHandle handle,
                 const SimulationDesc &desc);
  [[nodiscard]] Result<bool, std::string>
  destroyInstance(SceneRuntimeHost &host, SimulationHandle handle);
  [[nodiscard]] Result<bool, std::string>
  updateParams(SceneRuntimeHost &host, SimulationHandle handle,
               std::span<const std::byte> params);
  [[nodiscard]] Result<bool, std::string>
  executePhase(SceneRuntimeHost &host, SimulationHandle handle,
               SimulationPhase phase,
               const SimulationExecutionContext &context);
  [[nodiscard]] Result<bool, std::string>
  prepareSceneFrame(SceneRuntimeHost &host, uint64_t frameIndex);
  void commitSceneFrame(uint64_t frameIndex) noexcept;
  void abandonSceneFrame(uint64_t frameIndex) noexcept;
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
