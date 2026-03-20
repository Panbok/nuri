#pragma once

#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"

#include <cstdint>
#include <functional>

namespace nuri {

struct NURI_API NodeId {
  uint32_t value = 0;

  constexpr bool operator==(NodeId other) const noexcept {
    return value == other.value;
  }
  constexpr bool operator!=(NodeId other) const noexcept {
    return value != other.value;
  }
};

struct NURI_API RenderableId {
  uint32_t value = 0;

  constexpr bool operator==(RenderableId other) const noexcept {
    return value == other.value;
  }
  constexpr bool operator!=(RenderableId other) const noexcept {
    return value != other.value;
  }
};

inline constexpr NodeId kInvalidNodeId{};
inline constexpr RenderableId kInvalidRenderableId{};

[[nodiscard]] constexpr bool isValid(NodeId id) noexcept {
  return id.value != 0u;
}

[[nodiscard]] constexpr bool isValid(RenderableId id) noexcept {
  return id.value != 0u;
}

[[nodiscard]] constexpr uint32_t indexOf(NodeId id) noexcept {
  return unpackResourceHandle(id.value).index;
}

[[nodiscard]] constexpr uint32_t generationOf(NodeId id) noexcept {
  return unpackResourceHandle(id.value).generation;
}

[[nodiscard]] constexpr uint32_t indexOf(RenderableId id) noexcept {
  return unpackResourceHandle(id.value).index;
}

[[nodiscard]] constexpr uint32_t generationOf(RenderableId id) noexcept {
  return unpackResourceHandle(id.value).generation;
}

[[nodiscard]] constexpr NodeId makeNodeId(uint32_t index,
                                          uint32_t generation) noexcept {
  return NodeId{packResourceHandle(index, generation)};
}

[[nodiscard]] constexpr RenderableId
makeRenderableId(uint32_t index, uint32_t generation) noexcept {
  return RenderableId{packResourceHandle(index, generation)};
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

} // namespace std
