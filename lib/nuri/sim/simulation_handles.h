#pragma once
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include <cstdint>
#include <functional>
namespace nuri {

using SimulationHandle = PackedHandle<struct SimulationHandleTag>;

inline constexpr SimulationHandle kInvalidSimulationHandle{};

[[nodiscard]] constexpr SimulationHandle
makeSimulationHandle(uint32_t index, uint32_t generation) noexcept {
  return makePackedHandle<SimulationHandleTag>(index, generation);
}

enum class SimulationKind : uint8_t {
  Unknown = 0,
  AnimationPose = 1,
  Custom = 2,
};

enum class SimulationState : uint8_t {
  Stopped = 0,
  Running = 1,
  Paused = 2,
};

enum class SimulationPhase : uint8_t {
  PreSceneWrite = 0,
  Predict = 1,
  Project = 2,
  Finalize = 3,
  PostSceneWrite = 4,
  Count = 5,
};

} // namespace nuri
namespace std {

template <> struct hash<nuri::SimulationHandle> {
  size_t operator()(nuri::SimulationHandle handle) const noexcept {
    return hash<uint32_t>{}(handle.value);
  }
};

} // namespace std
