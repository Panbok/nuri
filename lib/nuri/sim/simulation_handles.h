#pragma once

#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"

#include <cstdint>
#include <functional>

namespace nuri {

struct NURI_API SimulationHandle {
  uint32_t value = 0;

  constexpr bool operator==(SimulationHandle other) const noexcept {
    return value == other.value;
  }
  constexpr bool operator!=(SimulationHandle other) const noexcept {
    return value != other.value;
  }
};

inline constexpr SimulationHandle kInvalidSimulationHandle{};

[[nodiscard]] constexpr bool isValid(SimulationHandle handle) noexcept {
  return handle.value != 0u;
}

[[nodiscard]] constexpr uint32_t indexOf(SimulationHandle handle) noexcept {
  return unpackResourceHandle(handle.value).index;
}

[[nodiscard]] constexpr uint32_t
generationOf(SimulationHandle handle) noexcept {
  return unpackResourceHandle(handle.value).generation;
}

[[nodiscard]] constexpr SimulationHandle
makeSimulationHandle(uint32_t index, uint32_t generation) noexcept {
  return SimulationHandle{packResourceHandle(index, generation)};
}

enum class SimulationKind : uint8_t {
  Unknown = 0,
  AnimationPose = 1,
  SecondaryMotion = 2,
  Cloth = 3,
  SoftBody = 4,
  RigidProxy = 5,
  Custom = 6,
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

enum class SimulationBackendPreference : uint8_t {
  Auto = 0,
  CPUOnly = 1,
  GPUOnly = 2,
  PreferGPU = 3,
};

} // namespace nuri

namespace std {

template <> struct hash<nuri::SimulationHandle> {
  size_t operator()(nuri::SimulationHandle handle) const noexcept {
    return hash<uint32_t>{}(handle.value);
  }
};

} // namespace std
