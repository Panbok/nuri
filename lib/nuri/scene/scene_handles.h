#pragma once
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include <cstdint>
#include <functional>
namespace nuri {

using NodeId = PackedHandle<struct NodeIdTag>;
using RenderableId = PackedHandle<struct RenderableIdTag>;
using DDGIVolumeId = PackedHandle<struct DDGIVolumeIdTag>;

inline constexpr NodeId kInvalidNodeId{};
inline constexpr RenderableId kInvalidRenderableId{};
inline constexpr DDGIVolumeId kInvalidDDGIVolumeId{};

[[nodiscard]] constexpr NodeId makeNodeId(uint32_t index,
                                          uint32_t generation) noexcept {
  return makePackedHandle<NodeIdTag>(index, generation);
}

[[nodiscard]] constexpr RenderableId
makeRenderableId(uint32_t index, uint32_t generation) noexcept {
  return makePackedHandle<RenderableIdTag>(index, generation);
}

[[nodiscard]] constexpr DDGIVolumeId
makeDDGIVolumeId(uint32_t index, uint32_t generation) noexcept {
  return makePackedHandle<DDGIVolumeIdTag>(index, generation);
}

} // namespace nuri
namespace std {

template <> struct hash<nuri::NodeId> {
  size_t operator()(nuri::NodeId id) const noexcept {
    return hash<uint32_t>{}(id.value);
  }
};

template <> struct hash<nuri::RenderableId> {
  size_t operator()(nuri::RenderableId id) const noexcept {
    return hash<uint32_t>{}(id.value);
  }
};

template <> struct hash<nuri::DDGIVolumeId> {
  size_t operator()(nuri::DDGIVolumeId id) const noexcept {
    return hash<uint32_t>{}(id.value);
  }
};

} // namespace std
