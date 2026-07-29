#pragma once
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include <cstdint>
namespace nuri {

using NodeId = PackedHandle<struct NodeIdTag>;
using RenderableId = PackedHandle<struct RenderableIdTag>;
using DDGIVolumeId = PackedHandle<struct DDGIVolumeIdTag>;

inline constexpr NodeId kInvalidNodeId{};
inline constexpr RenderableId kInvalidRenderableId{};
inline constexpr DDGIVolumeId kInvalidDDGIVolumeId{};

} // namespace nuri
