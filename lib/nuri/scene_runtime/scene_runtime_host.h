#pragma once
#include "nuri/defines.h"
#include "nuri/gfx/sim/animation_scene_frame_data.h"
#include "nuri/scene/render_scene.h"
#include "nuri/scene_runtime/scene_runtime_bindings.h"
#include "nuri/sim/animation_pose_simulation.h"
#include "nuri/sim/simulation_registry.h"
#include "nuri/sim/simulation_scheduler.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
namespace nuri {

class AnimationGpuServices;
class AnimationPoseSimulationBackend;
struct SimulationExecutionContext;

class NURI_API SceneRuntimeHost {
public:
  explicit SceneRuntimeHost(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~SceneRuntimeHost();
  [[nodiscard]] SceneRuntimeHost &simulations() noexcept { return *this; }
  [[nodiscard]] const SceneRuntimeHost &simulations() const noexcept {
    return *this;
  }
  [[nodiscard]] Result<SimulationHandle, std::string>
  createSimulation(const SimulationDesc &desc);
  [[nodiscard]] bool destroySimulation(SimulationHandle handle);
  [[nodiscard]] bool setEnabled(SimulationHandle handle, bool enabled);
  [[nodiscard]] bool pause(SimulationHandle handle);
  [[nodiscard]] bool resume(SimulationHandle handle);
  [[nodiscard]] bool requestSingleStep(SimulationHandle handle);
  [[nodiscard]] bool setTimeScale(SimulationHandle handle, float timeScale);
  [[nodiscard]] bool setSubstepCount(SimulationHandle handle, uint32_t count);
  [[nodiscard]] bool setSolverIterationCount(SimulationHandle handle,
                                             uint32_t count);
  [[nodiscard]] bool setParams(SimulationHandle handle,
                               std::span<const std::byte> params);
  [[nodiscard]] bool getState(SimulationHandle handle,
                              SimulationState &out) const;
  [[nodiscard]] bool getDesc(SimulationHandle handle,
                             SimulationDesc &out) const;
  [[nodiscard]] bool getStats(SimulationHandle handle,
                              SimulationStats &out) const;
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
  [[nodiscard]] uint64_t bindingVersion() const noexcept {
    return bindingVersion_;
  }
  [[nodiscard]] uint64_t topologyVersion() const noexcept {
    return topologyVersion_;
  }
  [[nodiscard]] double remainingAccumulatorSeconds() const noexcept {
    return lastTickResult_.remainingAccumulatorSeconds;
  }
  [[nodiscard]] std::pmr::memory_resource *memoryResource() noexcept {
    return memory_;
  }

private:
  friend class AnimationPoseSimulationBackend;
  friend class SimulationScheduler;
  [[nodiscard]] Result<bool, std::string>
  executeSimulationPhase(SimulationKind kind, SimulationHandle handle,
                         SimulationPhase phase,
                         const SimulationExecutionContext &context);
  [[nodiscard]] bool
  validateBindingTarget(SimulationBindingTarget &target) noexcept;
  [[nodiscard]] bool
  validateBindingDesc(SimulationBindingDesc &binding) noexcept;
  void validateAllSimulationBindings();
  void faultSimulation(SimulationHandle handle, std::string_view reason);
  void noteBindingMutation() noexcept;
  void refreshSceneBindingsIfNeeded();
  [[nodiscard]] SimulationRegistry &registry() noexcept { return registry_; }
  [[nodiscard]] const SimulationRegistry &registry() const noexcept {
    return registry_;
  }
  [[nodiscard]] SceneRuntimeBindings &bindings() noexcept { return bindings_; }
  [[nodiscard]] const SceneRuntimeBindings &bindings() const noexcept {
    return bindings_;
  }
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  RenderScene *scene_ = nullptr;
  SimulationRegistry registry_;
  SceneRuntimeBindings bindings_;
  SimulationScheduler scheduler_{};
  std::unique_ptr<AnimationPoseSimulationBackend> animationPoseBackend_;
  std::optional<AnimationPoseSimulationCreatePayload>
      pendingAnimationPoseCreatePayload_;
  std::unordered_map<const ScenePrefab *, std::weak_ptr<const ScenePrefab>>
      animationPrefabOwners_;
  std::unordered_map<const SceneInstantiationMap *,
                     std::weak_ptr<const SceneInstantiationMap>>
      animationInstantiationOwners_;
  uint64_t bindingVersion_ = 0u;
  uint64_t topologyVersion_ = 0u;
  SimulationTickResult lastTickResult_{};
};

} // namespace nuri
