#include "nuri/pch.h"

#include "nuri/sim/animation_pose_simulation.h"

namespace nuri {

Result<SimulationDesc, std::string> makeAnimationPoseSimulationDesc(
    const AnimationPoseSimulationCreateInfo &createInfo,
    std::pmr::memory_resource *memory) {
  if (createInfo.prefab == nullptr) {
    return Result<SimulationDesc, std::string>::makeError(
        "makeAnimationPoseSimulationDesc: prefab is null");
  }
  if (createInfo.instantiationMap == nullptr) {
    return Result<SimulationDesc, std::string>::makeError(
        "makeAnimationPoseSimulationDesc: instantiationMap is null");
  }
  if (!isValid(createInfo.rootNode)) {
    return Result<SimulationDesc, std::string>::makeError(
        "makeAnimationPoseSimulationDesc: rootNode is invalid");
  }

  SimulationDesc desc(memory);
  desc.kind = SimulationKind::AnimationPose;
  desc.debugName.assign(createInfo.debugName.data(),
                        createInfo.debugName.size());
  desc.binding.primaryTarget =
      SimulationBindingTarget::makePrefabRoot(createInfo.rootNode);
  desc.backendPreference = SimulationBackendPreference::GPUOnly;
  desc.allowGpuExecution = true;
  desc.startPaused = !createInfo.params.playing;
  desc.enabled = true;
  return Result<SimulationDesc, std::string>::makeResult(std::move(desc));
}

} // namespace nuri
