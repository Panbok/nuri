#pragma once

#include "nuri/math/types.h"
#include <cstdint>

namespace nuri {

struct DDGISceneCoverageBounds {
  BoundingBox bounds{};
  uint64_t generation = 0u;
  bool valid = false;
  bool complete = false;
  [[nodiscard]] constexpr bool
  operator==(const DDGISceneCoverageBounds &other) const noexcept {
    return bounds.min_.x == other.bounds.min_.x &&
           bounds.min_.y == other.bounds.min_.y &&
           bounds.min_.z == other.bounds.min_.z &&
           bounds.max_.x == other.bounds.max_.x &&
           bounds.max_.y == other.bounds.max_.y &&
           bounds.max_.z == other.bounds.max_.z &&
           generation == other.generation && valid == other.valid &&
           complete == other.complete;
  }
};

} // namespace nuri
