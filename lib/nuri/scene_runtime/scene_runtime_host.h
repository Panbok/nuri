#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/sim/animation_scene_frame_data.h"
#include "nuri/gfx/sim/simulation_gpu_context.h"
#include "nuri/scene/render_scene.h"
#include "nuri/scene_runtime/scene_runtime_bindings.h"
#include "nuri/sim/animation_pose_simulation.h"
#include "nuri/sim/backends/null_simulation_backend.h"
#include "nuri/sim/simulation_controller.h"
#include "nuri/sim/simulation_registry.h"
#include "nuri/sim/simulation_scheduler.h"
#include "nuri/sim/simulation_writeback.h"

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace nuri {

class ISimulationBackend;
class AnimationGpuServices;
class AnimationPoseSimulationBackend;

class NURI_API SceneRuntimeHost {
public:
  explicit SceneRuntimeHost(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~SceneRuntimeHost();

  [[nodiscard]] SimulationController &simulations() noexcept {
    return controller_;
  }
  [[nodiscard]] const SimulationController &simulations() const noexcept {
    return controller_;
  }

  [[nodiscard]] SimulationTickResult tick(const SimulationTickInput &input);
  void bindScene(RenderScene *scene);
  [[nodiscard]] const RenderScene *scene() const noexcept { return scene_; }
  void attachAnimationGpuServices(AnimationGpuServices *services) noexcept;
  [[nodiscard]] Result<bool, std::string>
  prepareAnimationSceneFrame(uint64_t frameIndex);
  void commitAnimationSceneFrame(uint64_t frameIndex) noexcept;
  void abandonAnimationSceneFrame(uint64_t frameIndex) noexcept;
  [[nodiscard]] Result<SimulationHandle, std::string>
  createAnimationPoseSimulation(
      const AnimationPoseSimulationCreateInfo &createInfo);
  [[nodiscard]] bool
  destroyAnimationPoseSimulation(SimulationHandle handle) noexcept;
  [[nodiscard]] const AnimationSceneFrameData *
  animationSceneFrameData() const noexcept;
  void reset();

  [[nodiscard]] uint64_t simulationVersion() const noexcept {
    return simulationVersion_;
  }
  [[nodiscard]] uint64_t bindingVersion() const noexcept {
    return bindingVersion_;
  }
  [[nodiscard]] uint64_t deformationVersion() const noexcept {
    return deformationVersion_;
  }
  [[nodiscard]] uint64_t topologyVersion() const noexcept {
    return topologyVersion_;
  }
  [[nodiscard]] uint64_t transformVersion() const noexcept {
    return transformVersion_;
  }
  [[nodiscard]] uint64_t simulationControlVersion() const noexcept {
    return simulationControlVersion_;
  }
  [[nodiscard]] double remainingAccumulatorSeconds() const noexcept {
    return lastTickResult_.remainingAccumulatorSeconds;
  }

  [[nodiscard]] std::pmr::memory_resource *memoryResource() noexcept {
    return memory_;
  }

private:
  friend class AnimationPoseSimulationBackend;
  friend class SimulationController;
  friend class SimulationScheduler;
  friend class NullSimulationBackend;

  [[nodiscard]] Result<SimulationHandle, std::string>
  createSimulationInternal(const SimulationDesc &desc);
  [[nodiscard]] bool destroySimulationInternal(SimulationHandle handle);
  [[nodiscard]] bool setSimulationEnabledInternal(SimulationHandle handle,
                                                  bool enabled);
  [[nodiscard]] bool pauseSimulationInternal(SimulationHandle handle);
  [[nodiscard]] bool resumeSimulationInternal(SimulationHandle handle);
  [[nodiscard]] bool
  requestSimulationSingleStepInternal(SimulationHandle handle);
  [[nodiscard]] bool setSimulationTimeScaleInternal(SimulationHandle handle,
                                                    float timeScale);
  [[nodiscard]] bool setSimulationSubstepCountInternal(SimulationHandle handle,
                                                       uint32_t count);
  [[nodiscard]] bool
  setSimulationSolverIterationCountInternal(SimulationHandle handle,
                                            uint32_t count);
  [[nodiscard]] bool
  setSimulationParamsInternal(SimulationHandle handle,
                              std::span<const std::byte> params);
  [[nodiscard]] bool getSimulationStateInternal(SimulationHandle handle,
                                                SimulationState &out) const;
  [[nodiscard]] bool getSimulationDescInternal(SimulationHandle handle,
                                               SimulationDesc &out) const;
  [[nodiscard]] bool getSimulationStatsInternal(SimulationHandle handle,
                                                SimulationStats &out) const;

  [[nodiscard]] ISimulationBackend &
  backendFor(const SimulationRegistry::Record &record) noexcept;
  [[nodiscard]] bool
  validateBindingTarget(SimulationBindingTarget &target) noexcept;
  [[nodiscard]] bool
  validateBindingDesc(SimulationBindingDesc &binding) noexcept;
  void validateAllSimulationBindings();
  void faultSimulation(SimulationHandle handle, std::string_view reason);
  void noteSimulationMutation() noexcept;
  void noteBindingMutation() noexcept;
  void refreshSceneBindingsIfNeeded();
  void flushWritebacks();

  [[nodiscard]] SimulationRegistry &registry() noexcept { return registry_; }
  [[nodiscard]] const SimulationRegistry &registry() const noexcept {
    return registry_;
  }
  [[nodiscard]] SceneRuntimeBindings &bindings() noexcept { return bindings_; }
  [[nodiscard]] const SceneRuntimeBindings &bindings() const noexcept {
    return bindings_;
  }
  [[nodiscard]] SimulationGpuContext &gpuContext() noexcept {
    return gpuContext_;
  }
  [[nodiscard]] SimulationWritebackState &writebacks() noexcept {
    return writebacks_;
  }

  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  RenderScene *scene_ = nullptr;
  SimulationRegistry registry_;
  SceneRuntimeBindings bindings_;
  SimulationGpuContext gpuContext_;
  SimulationWritebackState writebacks_;
  SimulationScheduler scheduler_{};
  NullSimulationBackend nullBackend_{};
  std::unique_ptr<AnimationPoseSimulationBackend> animationPoseBackend_;
  std::optional<AnimationPoseSimulationCreatePayload>
      pendingAnimationPoseCreatePayload_;
  SimulationController controller_;
  uint64_t simulationVersion_ = 0u;
  uint64_t bindingVersion_ = 0u;
  uint64_t deformationVersion_ = 0u;
  uint64_t topologyVersion_ = 0u;
  uint64_t transformVersion_ = 0u;
  uint64_t simulationControlVersion_ = 0u;
  SimulationTickResult lastTickResult_{};
};

} // namespace nuri
