#include "nuri/pch.h"

#include "nuri/sim/animation_pose_simulation.h"

namespace nuri {
namespace {

constexpr float kAnimationPoseBlendWeightEpsilon = 1.0e-5f;

[[nodiscard]] bool clipStateValid(const ScenePrefab &prefab,
                                  const AnimationPoseClipState &clip) {
  return clip.clipIndex < prefab.animations.size();
}

} // namespace

Result<AnimationPoseSimulationParams, std::string>
decodeAnimationPoseSimulationParams(std::span<const std::byte> bytes) {
  if (bytes.size() != sizeof(AnimationPoseSimulationParams)) {
    return Result<AnimationPoseSimulationParams, std::string>::makeError(
        "decodeAnimationPoseSimulationParams: invalid params payload");
  }

  AnimationPoseSimulationParams params{};
  std::memcpy(&params, bytes.data(), sizeof(params));
  return Result<AnimationPoseSimulationParams, std::string>::makeResult(params);
}

void sanitizeAnimationPoseSimulationParams(
    AnimationPoseSimulationParams &params) noexcept {
  params.blendWeight = glm::clamp(params.blendWeight, 0.0f, 1.0f);
  if (params.blendWeight <= kAnimationPoseBlendWeightEpsilon) {
    params.blendWeight = 0.0f;
    params.blendMode = AnimationPoseBlendMode::Single;
  }
}

Result<bool, std::string> validateAnimationPoseSimulationParams(
    const ScenePrefab &prefab, const AnimationPoseSimulationParams &params) {
  if (!clipStateValid(prefab, params.primary)) {
    return Result<bool, std::string>::makeError(
        "validateAnimationPoseSimulationParams: primary clip index is out of "
        "range");
  }
  if (params.blendMode == AnimationPoseBlendMode::Lerp &&
      params.blendWeight > kAnimationPoseBlendWeightEpsilon &&
      !clipStateValid(prefab, params.secondary)) {
    return Result<bool, std::string>::makeError(
        "validateAnimationPoseSimulationParams: secondary clip index is out "
        "of range");
  }
  return Result<bool, std::string>::makeResult(true);
}

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
  desc.startPaused = !createInfo.params.primary.playing;
  desc.enabled = true;
  desc.initialParams = asBytes(createInfo.params);
  return Result<SimulationDesc, std::string>::makeResult(std::move(desc));
}

} // namespace nuri
