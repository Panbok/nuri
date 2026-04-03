#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/scene/scene_handles.h"
#include "nuri/scene/scene_prefab.h"
#include "nuri/sim/simulation_desc.h"

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <string_view>
#include <type_traits>

namespace nuri {

enum class AnimationPosePlaybackMode : uint8_t {
  Once = 0,
  Loop = 1,
};

struct NURI_API AnimationPoseSimulationParams {
  uint32_t clipIndex = 0u;
  float timeSeconds = 0.0f;
  AnimationPosePlaybackMode playbackMode = AnimationPosePlaybackMode::Loop;
  bool playing = true;
  uint8_t reserved0 = 0u;
  uint16_t reserved1 = 0u;
};
static_assert(std::is_trivially_copyable_v<AnimationPoseSimulationParams>);

struct NURI_API AnimationPoseSimulationCreatePayload {
  const ScenePrefab *prefab = nullptr;
  const SceneInstantiationMap *instantiationMap = nullptr;
  AnimationPoseSimulationParams params{};
};
static_assert(
    std::is_trivially_copyable_v<AnimationPoseSimulationCreatePayload>);

struct NURI_API AnimationPoseSimulationCreateInfo {
  const ScenePrefab *prefab = nullptr;
  const SceneInstantiationMap *instantiationMap = nullptr;
  NodeId rootNode = kInvalidNodeId;
  std::string_view debugName = "AnimationPose";
  AnimationPoseSimulationParams params{};
};

[[nodiscard]] NURI_API Result<SimulationDesc, std::string>
makeAnimationPoseSimulationDesc(
    const AnimationPoseSimulationCreateInfo &createInfo,
    std::pmr::memory_resource *memory = std::pmr::get_default_resource());

[[nodiscard]] constexpr std::span<const std::byte>
asBytes(const AnimationPoseSimulationParams &params) noexcept {
  return std::as_bytes(
      std::span<const AnimationPoseSimulationParams>(&params, size_t{1u}));
}

[[nodiscard]] constexpr std::span<const std::byte>
asBytes(const AnimationPoseSimulationCreatePayload &payload) noexcept {
  return std::as_bytes(std::span<const AnimationPoseSimulationCreatePayload>(
      &payload, size_t{1u}));
}

} // namespace nuri
