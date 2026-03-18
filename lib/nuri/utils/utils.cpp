#include "nuri/pch.h"

#include "nuri/utils/utils.h"

namespace nuri {

size_t alignUp(size_t value, size_t alignment) {
  if (alignment == 0u) {
    return value;
  }

  const size_t mask = alignment - 1u;
  if ((alignment & mask) == 0u) {
    return (value + mask) & ~mask;
  }

  const size_t remainder = value % alignment;
  return remainder == 0u ? value : value + (alignment - remainder);
}

bool alignUpU64(uint64_t value, uint64_t alignment, uint64_t &out) {
  if (alignment <= 1u) {
    out = value;
    return true;
  }

  const uint64_t mask = alignment - 1u;
  if (value > (std::numeric_limits<uint64_t>::max() - mask)) {
    return false;
  }

  out = (value + mask) & ~mask;
  return true;
}

} // namespace nuri
