#pragma once

#include "nuri/defines.h"

#include <cstddef>
#include <cstdint>

namespace nuri {

[[nodiscard]] NURI_API size_t alignUp(size_t value, size_t alignment);
[[nodiscard]] NURI_API bool alignUpU64(uint64_t value, uint64_t alignment,
                                       uint64_t &out);

} // namespace nuri
