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

enum class AnimationPoseBlendMode : uint32_t {
  Single = 0u,
  Lerp = 1u,
};

enum class AnimationPoseBlendSyncMode : uint32_t {
  Independent = 0u,
  NormalizedTime = 1u,
};

struct NURI_API AnimationPoseClipState {
  uint32_t clipIndex = 0u;
  float timeSeconds = 0.0f;
  AnimationPosePlaybackMode playbackMode = AnimationPosePlaybackMode::Loop;
  bool playing = true;
  uint8_t reserved0[2] = {0u, 0u};
};
static_assert(std::is_trivially_copyable_v<AnimationPoseClipState>);
static_assert(sizeof(AnimationPoseClipState) == 12u);

struct NURI_API AnimationPoseSimulationParams {
  AnimationPoseClipState primary{};
  AnimationPoseClipState secondary{};
  float blendWeight = 0.0f;
  AnimationPoseBlendMode blendMode = AnimationPoseBlendMode::Single;
  AnimationPoseBlendSyncMode blendSyncMode =
      AnimationPoseBlendSyncMode::Independent;
  uint32_t reserved0 = 0u;
};
static_assert(std::is_trivially_copyable_v<AnimationPoseSimulationParams>);
static_assert(sizeof(AnimationPoseSimulationParams) == 40u);

// Borrowed pointers used only during in-process creation. These addresses are
// intentionally not serialized into SimulationDesc::initialParams.
struct NURI_API AnimationPoseSimulationCreatePayload {
  const ScenePrefab *prefab = nullptr;
  const SceneInstantiationMap *instantiationMap = nullptr;
  std::span<const uint32_t> controlledPrefabNodeIndices{};
  AnimationPoseSimulationParams params{};
};

struct NURI_API AnimationPoseSimulationCreateInfo {
  const ScenePrefab *prefab = nullptr;
  const SceneInstantiationMap *instantiationMap = nullptr;
  std::span<const uint32_t> controlledPrefabNodeIndices{};
  NodeId rootNode = kInvalidNodeId;
  std::string_view debugName = "AnimationPose";
  AnimationPoseSimulationParams params{};
};

[[nodiscard]] NURI_API Result<AnimationPoseSimulationParams, std::string>
decodeAnimationPoseSimulationParams(std::span<const std::byte> bytes);

NURI_API void sanitizeAnimationPoseSimulationParams(
    AnimationPoseSimulationParams &params) noexcept;

[[nodiscard]] NURI_API Result<bool, std::string>
validateAnimationPoseSimulationParams(
    const ScenePrefab &prefab, const AnimationPoseSimulationParams &params);

[[nodiscard]] NURI_API Result<SimulationDesc, std::string>
makeAnimationPoseSimulationDesc(
    const AnimationPoseSimulationCreateInfo &createInfo,
    std::pmr::memory_resource *memory = std::pmr::get_default_resource());

[[nodiscard]] constexpr std::span<const std::byte>
asBytes(const AnimationPoseSimulationParams &params) noexcept {
  return std::as_bytes(
      std::span<const AnimationPoseSimulationParams>(&params, size_t{1u}));
}
} // namespace nuri
