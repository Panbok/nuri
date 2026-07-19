#pragma once
#include "nuri/defines.h"
#include "nuri/scene/scene_handles.h"
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <string>
#include <vector>
namespace nuri {

inline constexpr uint32_t kInvalidSimulationBindingIndex =
    std::numeric_limits<uint32_t>::max();

enum class SimulationBindingTargetType : uint8_t {
  None = 0,
  Node = 1,
  Renderable = 2,
  PrefabRoot = 3,
};

enum class SimulationBindingFlags : uint32_t {
  None = 0u,
  ValidateTargets = 1u << 0u,
  FaultOnInvalidTarget = 1u << 1u,
};

[[nodiscard]] constexpr SimulationBindingFlags
operator|(SimulationBindingFlags lhs, SimulationBindingFlags rhs) noexcept {
  return static_cast<SimulationBindingFlags>(static_cast<uint32_t>(lhs) |
                                             static_cast<uint32_t>(rhs));
}

constexpr SimulationBindingFlags &
operator|=(SimulationBindingFlags &lhs, SimulationBindingFlags rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr SimulationBindingFlags
operator&(SimulationBindingFlags lhs, SimulationBindingFlags rhs) noexcept {
  return static_cast<SimulationBindingFlags>(static_cast<uint32_t>(lhs) &
                                             static_cast<uint32_t>(rhs));
}

constexpr SimulationBindingFlags &
operator&=(SimulationBindingFlags &lhs, SimulationBindingFlags rhs) noexcept {
  lhs = lhs & rhs;
  return lhs;
}

[[nodiscard]] constexpr SimulationBindingFlags
operator~(SimulationBindingFlags v) noexcept {
  return static_cast<SimulationBindingFlags>(~static_cast<uint32_t>(v));
}

[[nodiscard]] constexpr bool
hasSimulationBindingFlag(SimulationBindingFlags flags,
                         SimulationBindingFlags flag) noexcept {
  return static_cast<uint32_t>(flags & flag) != 0u;
}

struct NURI_API SimulationBindingTarget {
  SimulationBindingTargetType type = SimulationBindingTargetType::None;
  NodeId node = kInvalidNodeId;
  RenderableId renderable = kInvalidRenderableId;
  NodeId prefabRoot = kInvalidNodeId;
  uint32_t runtimeBindingIndex = kInvalidSimulationBindingIndex;
  [[nodiscard]] static constexpr SimulationBindingTarget
  makeNode(NodeId nodeId) noexcept {
    SimulationBindingTarget target{};
    target.type = SimulationBindingTargetType::Node;
    target.node = nodeId;
    return target;
  }
  [[nodiscard]] static constexpr SimulationBindingTarget
  makeRenderable(RenderableId renderableId) noexcept {
    SimulationBindingTarget target{};
    target.type = SimulationBindingTargetType::Renderable;
    target.renderable = renderableId;
    return target;
  }
  [[nodiscard]] static constexpr SimulationBindingTarget
  makePrefabRoot(NodeId rootNodeId) noexcept {
    SimulationBindingTarget target{};
    target.type = SimulationBindingTargetType::PrefabRoot;
    target.prefabRoot = rootNodeId;
    return target;
  }
};

struct NURI_API SimulationAttachmentBinding {
  uint32_t slot = 0u;
  SimulationBindingTarget target{};
};

struct NURI_API SimulationBindingDesc {
  explicit SimulationBindingDesc(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : secondaryTargets(memory), debugName(memory), attachmentSlots(memory) {}
  SimulationBindingTarget primaryTarget{};
  std::pmr::vector<SimulationBindingTarget> secondaryTargets;
  std::pmr::string debugName;
  SimulationBindingFlags flags = SimulationBindingFlags::ValidateTargets |
                                 SimulationBindingFlags::FaultOnInvalidTarget;
  std::pmr::vector<SimulationAttachmentBinding> attachmentSlots;
};

} // namespace nuri
